"""Render a batch of dynamics rollout videos from a trained checkpoint.

Mirrors the in-training video eval (scripts/train_dynamics.py) but runs
standalone, so it works from any checkpoint on any slice size -- including the
multi-host runs where live video eval is disabled. Each output is a side-by-side
mp4: decoded ground truth on the left, `context_frames` of real context followed
by generated frames on the right.

    uv run python scripts/eval_dynamics_videos.py \
        --checkpoint_dir gs://visionary-uc1/so101/checkpoints/dynamics_small_ft \
        --tokenizer_checkpoint_dir gs://visionary-uc1/so101/checkpoints/tokenizer \
        --data_dir data/so101/dyn/eval --output_dir artifacts/dynamics_videos --num_videos 20
"""

import argparse
import io
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import grain.python as grain
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from etils import epath
from hydra.utils import instantiate
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from visionary.common.checkpoint import (
    resolve_model_export_step,
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.common.jax import (
    data_parallel_mesh,
    global_batch_to_local,
    init_distributed_if_pod,
    local_batch_to_global,
    shard_from_host,
)
from visionary.dataset import decode_video_window
from visionary.eval.loading import build_raw_index
from visionary.models.dreamer4.dynamics import DynamicsModel, denormalize_latents, normalize_latents
from visionary.models.dreamer4.tokenizer_preprocessor import TokenizerPreprocessor
from visionary.shards import chunk_starts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectedClip:
    source: str
    embodiment_id: int
    index: int
    start_frame: int
    frame_count: int


def parse_selected_clips(value: str) -> list[SelectedClip]:
    clips = []
    for spec in value.split(","):
        source, embodiment_id, index, start_frame, frame_count = spec.split(":")
        clips.append(
            SelectedClip(
                source=source,
                embodiment_id=int(embodiment_id),
                index=int(index),
                start_frame=int(start_frame),
                frame_count=int(frame_count),
            )
        )
    return clips


def parse_source_dirs(value: str) -> dict[str, epath.Path]:
    sources = {}
    for spec in value.split(","):
        name, path = spec.split("=", maxsplit=1)
        sources[name] = epath.Path(path)
    return sources


def upload_output(source: Path, destination: str) -> None:
    subprocess.run(
        ["gcloud", "storage", "rsync", "--recursive", str(source), destination],
        check=True,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    init_distributed_if_pod(logger=logger)
    process_index = jax.process_index()
    process_count = jax.process_count()
    mesh = data_parallel_mesh()
    batch_sharding = NamedSharding(mesh, P("data"))
    replicated_sharding = NamedSharding(mesh, P())
    parser = argparse.ArgumentParser(description="Render dynamics rollout videos.")
    parser.add_argument("--checkpoint_dir", required=True, help="Dynamics checkpoint directory.")
    parser.add_argument("--tokenizer_checkpoint_dir", required=True, help="Tokenizer checkpoint directory.")
    parser.add_argument("--data_dir", help="Eval latent ArrayRecord directory.")
    parser.add_argument(
        "--source_dirs",
        help="Comma-separated source=directory pairs used with --clip_specs.",
    )
    parser.add_argument(
        "--clip_specs",
        help="Comma-separated source:embodiment:index:start_frame:frame_count selections.",
    )
    parser.add_argument("--output_dir", required=True, help="Where to write mp4 files.")
    parser.add_argument("--upload_dir", help="Optional GCS directory for the completed output.")
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument(
        "--random_clip_count",
        type=int,
        help="Select this many full records at random from --data_dir.",
    )
    parser.add_argument(
        "--required_indices",
        help="Comma-separated record indices that must be present in --random_clip_count.",
    )
    parser.add_argument(
        "--selection_source",
        default="eval",
        help="Source label written for clips selected from --data_dir.",
    )
    parser.add_argument(
        "--indices",
        help="Comma-separated video indices to render instead of 0..num_videos-1. "
        "Indices are deterministic given --seed, so the same clip can be re-rendered "
        "from a different checkpoint.",
    )
    parser.add_argument("--step", type=int, help="Checkpoint step. Defaults to latest.")
    parser.add_argument("--context_frames", type=int, default=4)
    parser.add_argument(
        "--generated_frames",
        type=int,
        default=60,
        help="-1 rolls out to the end of each record. Cost grows with the square "
        "of the length, since every generated frame re-runs the whole sequence.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=512,
        help="Upper bound on total frames when --generated_frames is -1.",
    )
    parser.add_argument(
        "--length_bucket",
        type=int,
        default=64,
        help="Round rollout lengths down to a multiple of this, so a handful of "
        "shapes are compiled instead of one per record.",
    )
    parser.add_argument(
        "--action_source",
        choices=("true", "zero", "shuffled", "gripper_open"),
        default="true",
        help="Probe how much the rollout depends on the actions. 'zero' uses zero actions, "
        "and 'shuffled' feeds another episode's actions. "
        "Scores that barely move mean the conditioning is being ignored.",
    )
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--rollout_seed",
        type=int,
        default=0,
        help="Offset added to the per-clip rollout noise seed; the crop stays fixed.",
    )
    parser.add_argument("--context_tau", type=float, default=0.9)
    parser.add_argument("--embodiment_id", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--raw_shards_dir",
        help="Packed video shards (eval split) so the reference panel is the "
        "original footage instead of the tokenizer's reconstruction.",
    )
    parser.add_argument(
        "--include_recon",
        action="store_true",
        help="Add a middle panel with the tokenizer reconstruction, which "
        "separates tokenizer error from dynamics error.",
    )
    args = parser.parse_args()

    if args.upload_dir and process_count != 1:
        parser.error("--upload_dir currently supports one-host jobs only")

    if bool(args.source_dirs) != bool(args.clip_specs):
        parser.error("--source_dirs and --clip_specs must be used together")
    if not args.clip_specs and not args.data_dir:
        parser.error("--data_dir is required unless --clip_specs is used")
    if args.clip_specs and args.action_source != "true":
        parser.error("--clip_specs supports only --action_source=true")
    if args.random_clip_count is not None:
        if args.clip_specs or args.indices:
            parser.error("--random_clip_count cannot be combined with --clip_specs or --indices")
        if args.random_clip_count <= 0:
            parser.error("--random_clip_count must be positive")
        if args.action_source != "true":
            parser.error("--random_clip_count supports only --action_source=true")
    elif args.required_indices:
        parser.error("--required_indices requires --random_clip_count")

    roll_to_end = args.generated_frames < 0
    fixed_total = None if roll_to_end else args.context_frames + args.generated_frames
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latents = None
    selected_sources = {}
    selected_clips = parse_selected_clips(args.clip_specs) if args.clip_specs else []
    if selected_clips:
        for name, source_dir in parse_source_dirs(args.source_dirs).items():
            paths = sorted(str(path) for path in source_dir.glob("*.arecord"))
            if not paths:
                raise FileNotFoundError(f"No .arecord files in {source_dir}")
            selected_sources[name] = grain.ArrayRecordDataSource(paths)
            logger.info("Eval source %s has %d records", name, len(selected_sources[name]))
        missing_sources = sorted({clip.source for clip in selected_clips} - selected_sources.keys())
        if missing_sources:
            raise ValueError(f"Missing source directories for: {missing_sources}")
    else:
        latent_paths = sorted(str(p) for p in Path(args.data_dir).glob("*.arecord"))
        if not latent_paths:
            raise FileNotFoundError(f"No .arecord files in {args.data_dir}")
        latents = grain.ArrayRecordDataSource(latent_paths)
        logger.info("Eval source has %d records", len(latents))

        if args.random_clip_count is not None:
            required = []
            for value in (args.required_indices or "").split(","):
                if value:
                    index = int(value)
                    if index not in required:
                        required.append(index)
            if len(required) > args.random_clip_count:
                raise ValueError(
                    f"Found {len(required)} required indices for a {args.random_clip_count}-clip sample"
                )

            frame_counts = {}
            for index in range(len(latents)):
                with np.load(io.BytesIO(latents[index])) as data:
                    frame_counts[index] = min(len(data["frames"]), args.max_frames)
            eligible = [index for index, count in frame_counts.items() if count > args.context_frames]
            missing = [index for index in required if index not in frame_counts]
            too_short = [index for index in required if index in frame_counts and index not in eligible]
            if missing:
                raise IndexError(f"Required indices are outside 0..{len(latents) - 1}: {missing}")
            if too_short:
                raise ValueError(f"Required indices are too short for context {args.context_frames}: {too_short}")

            candidates = np.asarray([index for index in eligible if index not in required], dtype=np.int64)
            rng = np.random.default_rng(args.seed)
            rng.shuffle(candidates)
            needed = args.random_clip_count - len(required)
            if len(candidates) < needed:
                raise ValueError(
                    f"Found only {len(eligible)} eligible records for {args.random_clip_count} clips"
                )
            selected_indices = required + candidates[:needed].tolist()
            selected_clips = [
                SelectedClip(
                    source=args.selection_source,
                    embodiment_id=args.embodiment_id,
                    index=index,
                    start_frame=0,
                    frame_count=frame_counts[index],
                )
                for index in selected_indices
            ]
            selected_sources[args.selection_source] = latents
            logger.info(
                "Selected %d full clips with seed %d, including %d required indices",
                len(selected_clips),
                args.seed,
                len(required),
            )

    raw_index = build_raw_index(args.raw_shards_dir) if args.raw_shards_dir else {}
    if args.raw_shards_dir:
        logger.info("Indexed %d original streams from %s", len(raw_index), args.raw_shards_dir)

    def sample_for(index: int):
        assert latents is not None
        with np.load(io.BytesIO(latents[index % len(latents)])) as data:
            video = np.asarray(data["frames"])
            actions = np.asarray(data["actions"], dtype=np.float32)
            prev_action = np.asarray(data["prev_action"], dtype=np.float32)
            key = (str(data["repo"]), int(data["episode"]), str(data["camera"]))
            record_start = int(data["start_index"]) if "start_index" in data.files else 0

        stride = args.stride
        usable_strided = (len(video) - 1) // stride + 1
        if roll_to_end:
            usable = min(usable_strided, args.max_frames)
            # bucket the length so jit compiles a few shapes, not one per record
            total = max(
                usable // args.length_bucket * args.length_bucket,
                args.context_frames + args.length_bucket,
            )
            total = min(total, usable_strided)
        else:
            total = fixed_total
            if total > usable_strided:
                raise ValueError(f"Record {index} too short: {len(video)} < span of {total}")

        span = (total - 1) * stride + 1
        rng = np.random.default_rng([args.seed, index])
        offset = int(rng.integers(0, max(len(video) - span, 0) + 1))
        if args.action_source == "shuffled":
            # a different episode's actions: still a plausible trajectory, just
            # the wrong one, so only a model that reads them will be hurt
            other = (index + len(latents) // 2) % len(latents)
            with np.load(io.BytesIO(latents[other])) as data:
                donor = np.asarray(data["actions"], dtype=np.float32)
            donor = np.resize(donor, (len(actions), donor.shape[1]))
            actions, prev_action = donor, donor[max(offset - 1, 0)]
        indices = offset + np.arange(total) * stride
        aligned = actions[indices - 1]
        if offset == 0:
            aligned[0] = prev_action
        if args.action_source == "gripper_open":
            # hold the gripper at its starting command; every other joint
            # follows the truth, so the grasp should never happen
            aligned[:, -1] = aligned[0, -1]
        elif args.action_source == "zero":
            aligned = np.zeros_like(aligned)
        return {
            "video": np.asarray(video[indices], dtype=np.float32)[None],
            "actions": np.asarray(aligned, dtype=np.float32)[None],
            "key": key,
            "absolute_start": record_start + offset,
            "span": span,
            "total": total,
        }

    if selected_clips:
        selected = []
    elif args.indices:
        selected = [int(i) for i in args.indices.split(",")]
    else:
        selected = []
        minimum_frames = fixed_total or args.context_frames + args.length_bucket
        for index in range(len(latents)):
            with np.load(io.BytesIO(latents[index])) as data:
                record_frames = len(data["frames"])
            usable_frames = (record_frames - 1) // args.stride + 1
            if usable_frames >= minimum_frames:
                selected.append(index)
                if len(selected) == args.num_videos:
                    break
        if len(selected) < args.num_videos:
            raise ValueError(f"Found only {len(selected)} records with at least {minimum_frames} usable frames")
    step = resolve_model_export_step(args.checkpoint_dir, args.step)
    export_cfg, params = restore_model_export_single_device(args.checkpoint_dir, step=step)
    model = instantiate(export_cfg)
    logger.info("Restored dynamics export from step %d", step)

    tokenizer_cfg, tokenizer_variables = restore_model_export_single_device(args.tokenizer_checkpoint_dir)
    tokenizer = instantiate(tokenizer_cfg)
    preprocessor = TokenizerPreprocessor.from_config(restore_preprocessor_export(args.tokenizer_checkpoint_dir))

    def rollout_fn(params, video, actions, embodiment_ids, seed, generated_frames):
        video = normalize_latents(video, model.latent_mean, model.latent_std)
        actions = jnp.asarray(actions, dtype=jnp.float32)
        actions = jnp.pad(actions, ((0, 0), (0, 0), (0, model.max_action_dim - actions.shape[-1])))
        embodiment_ids = jnp.asarray(embodiment_ids, dtype=jnp.int32)
        primed = jnp.zeros_like(video).at[:, : args.context_frames].set(video[:, : args.context_frames])
        context_key, sample_key = jax.random.split(seed)
        context_noise = jax.random.normal(context_key, video.shape, dtype=jnp.float32)
        sample_noise = jax.random.normal(
            sample_key, (video.shape[0], generated_frames, *video.shape[2:]), dtype=jnp.float32
        )
        generated, diagnostics = model.apply(
            params,
            primed,
            actions,
            embodiment_ids,
            context_noise,
            sample_noise,
            args.context_frames,
            context_tau=args.context_tau,
            sample_steps=args.sample_steps,
            method=DynamicsModel.generate_rollout,
        )
        return denormalize_latents(generated, model.latent_mean, model.latent_std), diagnostics

    if selected_clips:
        rollout = jax.jit(
            rollout_fn,
            in_shardings=(
                replicated_sharding,
                batch_sharding,
                batch_sharding,
                batch_sharding,
                replicated_sharding,
            ),
            out_shardings=(batch_sharding, replicated_sharding),
            static_argnames=("generated_frames",),
        )
    else:
        rollout = jax.jit(rollout_fn, static_argnames=("generated_frames",))

    def decode_chunk_fn(tokenizer_variables, latent_chunk):
        return preprocessor.patches_to_images(
            tokenizer.apply(tokenizer_variables, latent_chunk, method=type(tokenizer).decode)
        ).astype(jnp.float32)

    if selected_clips:
        decode_chunk = jax.jit(
            decode_chunk_fn,
            in_shardings=(replicated_sharding, batch_sharding),
            out_shardings=batch_sharding,
        )
    else:
        decode_chunk = jax.jit(decode_chunk_fn)

    def decode_all(latent: jnp.ndarray, window_length: int = 16, window_overlap: int = 8) -> jax.Array:
        pieces = []
        previous_stop = 0
        for start in chunk_starts(latent.shape[1], window_length, window_overlap):
            stop = min(start + window_length, latent.shape[1])
            piece = latent[:, start:stop]
            pad = window_length - piece.shape[1]
            if pad:
                piece = jnp.pad(piece, ((0, 0), (0, pad), (0, 0), (0, 0)))
            decoded = decode_chunk(tokenizer_variables, piece)[:, : stop - start]
            warmup = max(previous_stop - start, 0)
            pieces.append(decoded[:, warmup:])
            previous_stop = stop
        return jnp.concatenate(pieces, axis=1)

    def to_u8(x: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)

    if selected_clips:
        params = jax.tree_util.tree_map(lambda value: shard_from_host(value, replicated_sharding), params)
        tokenizer_variables = jax.tree_util.tree_map(
            lambda value: shard_from_host(value, replicated_sharding), tokenizer_variables
        )
        clips = sorted(selected_clips, key=lambda clip: clip.frame_count)
        global_batch_size = jax.device_count()
        local_batch_size = global_batch_size // process_count
        padding = (-len(clips)) % global_batch_size
        clip_items = [(clip, False) for clip in clips]
        clip_items.extend((clips[0], True) for _ in range(padding))
        clip_items.sort(key=lambda item: item[0].frame_count)
        if padding:
            logger.info("Padding %d selected batch slots; padded rows are not scored or written", padding)

        local_summary = []
        for batch_start in range(0, len(clip_items), global_batch_size):
            global_items = clip_items[batch_start : batch_start + global_batch_size]
            local_start = process_index * local_batch_size
            local_items = global_items[local_start : local_start + local_batch_size]
            total_frames = max(clip.frame_count for clip, _ in global_items)
            local_videos = []
            local_actions = []
            local_embodiment_ids = []
            local_metadata = []

            for clip, is_padding in local_items:
                source = selected_sources[clip.source]
                if clip.index >= len(source):
                    raise IndexError(f"{clip.source} index {clip.index} is outside 0..{len(source) - 1}")
                with np.load(io.BytesIO(source[clip.index])) as data:
                    video = np.asarray(data["frames"], dtype=np.float32)
                    actions = np.asarray(data["actions"], dtype=np.float32)
                    prev_action = np.asarray(data["prev_action"], dtype=np.float32)
                    stream = f"{str(data['repo'])}/ep{int(data['episode'])}/{str(data['camera'])}"
                stop_frame = clip.start_frame + clip.frame_count
                if clip.start_frame < 0 or clip.frame_count <= args.context_frames or stop_frame > len(video):
                    raise ValueError(
                        f"Invalid range for {clip.source} index {clip.index}: "
                        f"start={clip.start_frame} count={clip.frame_count} record={len(video)}"
                    )
                indices = np.arange(clip.start_frame, stop_frame)
                aligned_actions = actions[indices - 1].copy()
                if clip.start_frame == 0:
                    aligned_actions[0] = prev_action
                action_padding = model.max_action_dim - aligned_actions.shape[-1]
                if action_padding < 0:
                    raise ValueError(
                        f"Action dimension {aligned_actions.shape[-1]} exceeds model maximum {model.max_action_dim}"
                    )
                aligned_actions = np.pad(aligned_actions, ((0, 0), (0, action_padding)))
                frame_padding = total_frames - clip.frame_count
                local_videos.append(np.pad(video[indices], ((0, frame_padding), (0, 0), (0, 0))))
                local_actions.append(np.pad(aligned_actions, ((0, frame_padding), (0, 0))))
                local_embodiment_ids.append(np.full((total_frames,), clip.embodiment_id, dtype=np.int32))
                local_metadata.append((clip, stream, is_padding))

            video_batch = local_batch_to_global(np.stack(local_videos), batch_sharding)
            action_batch = local_batch_to_global(np.stack(local_actions), batch_sharding)
            embodiment_batch = local_batch_to_global(np.stack(local_embodiment_ids), batch_sharding)
            started = time.monotonic()
            generated_latent, diagnostics = rollout(
                params,
                video_batch,
                action_batch,
                embodiment_batch,
                jax.random.key(args.rollout_seed + batch_start),
                total_frames - args.context_frames,
            )
            decoded = to_u8(global_batch_to_local(decode_all(video_batch)))
            generated = to_u8(global_batch_to_local(decode_all(generated_latent)))

            for row, (clip, stream, is_padding) in enumerate(local_metadata):
                if is_padding:
                    continue
                stop_frame = clip.start_frame + clip.frame_count
                base = f"d{clip.index:06d}__f{clip.start_frame:04d}-{stop_frame:04d}"
                domain_dir = output_dir / clip.source
                decoded_dir = domain_dir / "decoded"
                rollout_dir = domain_dir / "rollout"
                decoded_dir.mkdir(parents=True, exist_ok=True)
                rollout_dir.mkdir(parents=True, exist_ok=True)
                decoded_frames = decoded[row, : clip.frame_count]
                generated_frames = generated[row, : clip.frame_count]
                imageio.mimsave(decoded_dir / f"{base}__decoded.mp4", decoded_frames, fps=args.fps)
                imageio.mimsave(
                    rollout_dir / f"{base}__rollout_s{args.sample_steps}.mp4", generated_frames, fps=args.fps
                )

                generated_slice = slice(args.context_frames, clip.frame_count)
                decoded_f = decoded_frames.astype(np.float32) / 255.0
                generated_f = generated_frames.astype(np.float32) / 255.0
                psnr = float(
                    peak_signal_noise_ratio(decoded_f[generated_slice], generated_f[generated_slice], data_range=1.0)
                )
                ssim = float(
                    np.mean(
                        [
                            structural_similarity(reference, prediction, data_range=1.0, channel_axis=-1)
                            for reference, prediction in zip(
                                decoded_f[generated_slice], generated_f[generated_slice], strict=True
                            )
                        ]
                    )
                )
                local_summary.append(
                    {
                        "source": clip.source,
                        "embodiment_id": clip.embodiment_id,
                        "index": clip.index,
                        "start_frame": clip.start_frame,
                        "frame_count": clip.frame_count,
                        "stream": stream,
                        "psnr_vs_decoded": psnr,
                        "ssim_vs_decoded": ssim,
                    }
                )
                logger.info(
                    "%s %d frames: PSNR %.2f SSIM %.4f",
                    base,
                    clip.frame_count,
                    psnr,
                    ssim,
                )
            logger.info("Finished selected batch in %.1fs", time.monotonic() - started)

        (output_dir / f"summary_process_{process_index}.json").write_text(
            json.dumps(
                {
                    "checkpoint_dir": args.checkpoint_dir,
                    "step": step,
                    "context_frames": args.context_frames,
                    "sample_steps": args.sample_steps,
                    "context_tau": args.context_tau,
                    "videos": local_summary,
                },
                indent=2,
            )
        )
        logger.info("Wrote %d selected clips on process %d", len(local_summary), process_index)
        if args.upload_dir:
            upload_output(output_dir, args.upload_dir)
            logger.info("Uploaded rollout output to %s", args.upload_dir)
        return

    summary = []
    for position, index in enumerate(selected):
        sample = sample_for(index)
        total_frames = sample["total"]
        started = time.monotonic()
        generated_latent, diagnostics = rollout(
            params,
            sample["video"],
            sample["actions"],
            np.full((1, total_frames), args.embodiment_id, dtype=np.int32),
            jax.random.key(index + args.rollout_seed),
            total_frames - args.context_frames,
        )
        recon = to_u8(np.asarray(jax.device_get(decode_all(jnp.asarray(sample["video"], dtype=jnp.float32))))[0])
        generated = to_u8(np.asarray(jax.device_get(decode_all(generated_latent)))[0])

        # the reference is the untouched footage when we can reach it, so the
        # score covers tokenizer error too rather than hiding it
        reference, reference_kind = recon, "tokenizer_recon"
        if sample["key"] in raw_index:
            try:
                raw = decode_video_window(
                    raw_index[sample["key"]],
                    sample["absolute_start"],
                    sample["span"],
                    tuple(preprocessor.resize_shape),
                )
                raw = raw[:: args.stride][:total_frames]
                if len(raw) == total_frames:
                    reference, reference_kind = raw, "raw"
                else:
                    logger.warning(
                        "video %d: decoded %d/%d raw frames, falling back to recon",
                        index,
                        len(raw),
                        total_frames,
                    )
            except Exception:
                logger.warning("video %d: raw decode failed, falling back to recon", index)
        elif raw_index:
            logger.warning("video %d: %s not in raw index", index, sample["key"])

        gen = slice(args.context_frames, total_frames)
        ref_f = reference.astype(np.float32) / 255.0
        gen_f = generated.astype(np.float32) / 255.0
        psnr = float(peak_signal_noise_ratio(ref_f[gen], gen_f[gen], data_range=1.0))
        ssim = float(
            np.mean(
                [
                    structural_similarity(t, g, data_range=1.0, channel_axis=-1)
                    for t, g in zip(ref_f[gen], gen_f[gen], strict=True)
                ]
            )
        )

        panels = [reference, recon, generated] if args.include_recon else [reference, generated]
        separator = np.full((panels[0].shape[1], 4, 3), 255, dtype=np.uint8)
        frames = []
        for parts in zip(*panels, strict=True):
            row = [parts[0]]
            for part in parts[1:]:
                row.extend([separator, part])
            frames.append(np.concatenate(row, axis=1))
        tag = "" if args.action_source == "true" else f"_act-{args.action_source}"
        path = output_dir / f"rollout_{step}_{index:02d}{tag}_s{args.sample_steps}_psnr{psnr:.1f}.mp4"
        imageio.mimsave(path, frames, fps=args.fps)
        summary.append(
            {
                "index": index,
                "psnr": psnr,
                "ssim": ssim,
                "reference": reference_kind,
                "stream": f"{sample['key'][0]}/ep{sample['key'][1]}/{sample['key'][2]}",
                "path": path.name,
                "x0_norm_by_step": np.asarray(diagnostics["x0_norm"]).mean(axis=0).tolist(),
                "update_mag_by_step": np.asarray(diagnostics["update_mag"]).mean(axis=0).tolist(),
            }
        )
        logger.info(
            "[%d/%d] %s  %d frames  psnr %.2f  ssim %.4f  (%.0fs)",
            position + 1,
            len(selected),
            path.name,
            total_frames,
            psnr,
            ssim,
            time.monotonic() - started,
        )

    mean_psnr = float(np.mean([s["psnr"] for s in summary]))
    mean_ssim = float(np.mean([s["ssim"] for s in summary]))
    suffix = "" if args.action_source == "true" else f"_act-{args.action_source}"
    (output_dir / f"summary{suffix}_s{args.sample_steps}.json").write_text(
        json.dumps(
            {
                "checkpoint_dir": args.checkpoint_dir,
                "step": step,
                "context_frames": args.context_frames,
                "generated_frames": args.generated_frames,
                "action_source": args.action_source,
                "sample_steps": args.sample_steps,
                "context_tau": args.context_tau,
                "raw_reference_videos": sum(1 for s in summary if s["reference"] == "raw"),
                "mean_psnr": mean_psnr,
                "mean_ssim": mean_ssim,
                "mean_x0_norm_by_step": np.mean(
                    [video["x0_norm_by_step"] for video in summary],
                    axis=0,
                ).tolist(),
                "mean_update_mag_by_step": np.mean(
                    [video["update_mag_by_step"] for video in summary],
                    axis=0,
                ).tolist(),
                "videos": summary,
            },
            indent=2,
        )
    )
    logger.info(
        "Wrote %d videos to %s | mean PSNR %.2f, mean SSIM %.4f",
        len(summary),
        output_dir,
        mean_psnr,
        mean_ssim,
    )
    if args.upload_dir:
        upload_output(output_dir, args.upload_dir)
        logger.info("Uploaded rollout output to %s", args.upload_dir)


if __name__ == "__main__":
    main()
