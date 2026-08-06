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
import functools
import io
import json
import logging
import time
from pathlib import Path

import grain.python as grain
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from visionary.common.checkpoint import (
    CheckpointManager,
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.common.train_state import DynamicsTrainState
from visionary.dataset import align_actions_to_frames, decode_video_window
from visionary.dynamics import DynamicsModel
from visionary.tokenizer_preprocessor import TokenizerPreprocessor

logger = logging.getLogger(__name__)


def build_raw_index(shards_dir: str) -> dict[tuple[str, int, str], bytes]:
    """Map (repo, episode, camera) -> mp4 bytes from the packed video shards.

    These are the same trimmed streams the latents were encoded from, so a
    latent record's start_index indexes directly into this video.
    """
    paths = sorted(str(p) for p in Path(shards_dir).glob("*.arecord"))
    if not paths:
        raise FileNotFoundError(f"No .arecord files in {shards_dir}")
    source = grain.ArrayRecordDataSource(paths)
    index: dict[tuple[str, int, str], bytes] = {}
    for i in range(len(source)):
        with np.load(io.BytesIO(source[i])) as data:
            key = (str(data["repo"]), int(data["episode"]), str(data["camera"]))
            index[key] = data["video"].tobytes()
    return index


def load_train_config(checkpoint_dir: str) -> OmegaConf:
    """The training run stores its own resolved config next to the checkpoints."""
    manager = CheckpointManager(
        checkpoint_dir,
        instantiate({"_target_": "orbax.checkpoint.CheckpointManagerOptions"}),
    )
    metadata = manager.load_metadata()
    manager.close()
    for key in ("dynamics_config", "config"):
        if key in metadata:
            return OmegaConf.create(metadata[key])
    raise KeyError(f"No config found in checkpoint metadata at {checkpoint_dir}")


def restore_params(cfg: OmegaConf, checkpoint_dir: str, step: int | None, sample_batch):
    model = instantiate(cfg.dynamics)
    optimizer = instantiate(cfg.optimizer)

    def make_state():
        params = model.init(
            {"params": jax.random.key(0), "sample": jax.random.key(1)},
            sample_batch,
            bootstrap_rows=0,
            method=DynamicsModel.loss,
        )
        return DynamicsTrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # shapes only: allocating real optimizer state would triple the memory for
    # no reason, and restore just needs the tree structure
    abstract_state = jax.eval_shape(make_state)
    # the training run sharded these over its own mesh, so without an explicit
    # sharding orbax reuses the saved topology and rejects a different slice
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    abstract_state = jax.tree_util.tree_map(
        lambda leaf: (
            jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding)
            if hasattr(leaf, "shape")
            else leaf
        ),
        abstract_state,
    )
    manager = CheckpointManager(checkpoint_dir, instantiate(cfg.checkpoint.manager.options))
    resolved = manager.latest_step() if step is None else int(step)
    params = manager.restore(target=abstract_state, step=resolved, params_only=True)
    manager.close()
    return model, params, resolved


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Render dynamics rollout videos.")
    parser.add_argument("--checkpoint_dir", required=True, help="Dynamics checkpoint directory.")
    parser.add_argument(
        "--tokenizer_checkpoint_dir", required=True, help="Tokenizer checkpoint directory."
    )
    parser.add_argument("--data_dir", required=True, help="Eval latent ArrayRecord directory.")
    parser.add_argument("--output_dir", required=True, help="Where to write mp4 files.")
    parser.add_argument("--num_videos", type=int, default=20)
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
        choices=("true", "none", "shuffled", "gripper_open"),
        default="true",
        help="Probe how much the rollout depends on the actions. 'none' takes the "
        "model's unconditional path, 'shuffled' feeds another episode's actions. "
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
    parser.add_argument(
        "--from_export",
        action="store_true",
        help="Load weights-only exports from <checkpoint_dir>/model/<step> instead "
        "of the full train state.",
    )
    parser.add_argument("--context_tau", type=float, default=0.9)
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

    roll_to_end = args.generated_frames < 0
    fixed_total = None if roll_to_end else args.context_frames + args.generated_frames
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_train_config(args.checkpoint_dir)
    logger.info("Loaded training config for exp %s", cfg.get("exp_name", "?"))

    # read records directly rather than through DynamicsDataSource: the crop
    # offset and provenance are needed to locate the original footage
    latent_paths = sorted(str(p) for p in Path(args.data_dir).glob("*.arecord"))
    if not latent_paths:
        raise FileNotFoundError(f"No .arecord files in {args.data_dir}")
    latents = grain.ArrayRecordDataSource(latent_paths)
    logger.info("Eval source has %d records", len(latents))

    raw_index = build_raw_index(args.raw_shards_dir) if args.raw_shards_dir else {}
    if args.raw_shards_dir:
        logger.info("Indexed %d original streams from %s", len(raw_index), args.raw_shards_dir)

    def sample_for(index: int):
        with np.load(io.BytesIO(latents[index % len(latents)])) as data:
            video = np.asarray(data["frames"])
            actions = np.asarray(data["actions"], dtype=np.float32)
            prev_action = np.asarray(data["prev_action"], dtype=np.float32)
            key = (str(data["repo"]), int(data["episode"]), str(data["camera"]))
            record_start = int(data["start_index"])

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
        return {
            "video": np.asarray(video[indices], dtype=np.float32)[None],
            "actions": np.asarray(aligned, dtype=np.float32)[None],
            "key": key,
            "absolute_start": record_start + offset,
            "span": span,
            "total": total,
        }

    first = sample_for(0)
    if args.from_export:
        export_cfg, params = restore_model_export_single_device(
            args.checkpoint_dir, step=args.step
        )
        model = instantiate(export_cfg)
        step = args.step
        logger.info("Restored dynamics export from step %d", step)
    else:
        # the model was initialised against the training batch_length, but the
        # rollout only ever sees total_frames, so init at that length
        model, params, step = restore_params(
            cfg,
            args.checkpoint_dir,
            args.step,
            {"video": first["video"], "actions": first["actions"]},
        )
        logger.info("Restored dynamics params from step %d", step)

    tokenizer_cfg, tokenizer_variables = restore_model_export_single_device(
        args.tokenizer_checkpoint_dir
    )
    tokenizer = instantiate(tokenizer_cfg)
    preprocessor = TokenizerPreprocessor.from_config(
        restore_preprocessor_export(args.tokenizer_checkpoint_dir)
    )

    @functools.partial(jax.jit, static_argnames=("generated_frames",))
    def rollout(params, video, actions, seed, generated_frames):
        video = jnp.asarray(video, dtype=jnp.float32)
        actions = None if args.action_source == "none" else jnp.asarray(actions, dtype=jnp.float32)
        primed = (
            jnp.zeros_like(video).at[:, : args.context_frames].set(video[:, : args.context_frames])
        )
        context_key, sample_key = jax.random.split(jax.random.key(seed))
        context_noise = jax.random.normal(context_key, video.shape, dtype=jnp.float32)
        sample_noise = jax.random.normal(
            sample_key, (video.shape[0], generated_frames, *video.shape[2:]), dtype=jnp.float32
        )
        return model.apply(
            params,
            primed,
            actions,
            context_noise,
            sample_noise,
            args.context_frames,
            context_tau=args.context_tau,
            sample_steps=args.sample_steps,
            method=DynamicsModel.generate_rollout,
        )

    @jax.jit
    def decode_chunk(tokenizer_variables, latent_chunk):
        return preprocessor.patches_to_images(
            tokenizer.apply(tokenizer_variables, latent_chunk, method=type(tokenizer).decode)
        ).astype(jnp.float32)

    def decode_all(latent: jnp.ndarray, chunk: int = 64) -> np.ndarray:
        """Decode in fixed-size chunks; a full-length clip at once will not fit."""
        pieces = []
        for start in range(0, latent.shape[1], chunk):
            piece = latent[:, start : start + chunk]
            if piece.shape[1] < chunk:  # keep one compiled shape
                pad = chunk - piece.shape[1]
                piece = jnp.pad(piece, ((0, 0), (0, pad), (0, 0), (0, 0)))
                pieces.append(
                    np.asarray(jax.device_get(decode_chunk(tokenizer_variables, piece)))[:, :-pad]
                )
            else:
                pieces.append(np.asarray(jax.device_get(decode_chunk(tokenizer_variables, piece))))
        return np.concatenate(pieces, axis=1)

    def to_u8(x: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)

    selected = (
        [int(i) for i in args.indices.split(",")] if args.indices else list(range(args.num_videos))
    )
    summary = []
    for position, index in enumerate(selected):
        sample = sample_for(index)
        total_frames = sample["total"]
        started = time.monotonic()
        generated_latent = rollout(
            params,
            sample["video"],
            sample["actions"],
            index + args.rollout_seed,
            total_frames - args.context_frames,
        )
        recon = to_u8(decode_all(jnp.asarray(sample["video"], dtype=jnp.float32))[0])
        generated = to_u8(decode_all(generated_latent)[0])

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
        path = (
            output_dir / f"rollout_{step}_{index:02d}{tag}_s{args.sample_steps}_psnr{psnr:.1f}.mp4"
        )
        imageio.mimsave(path, frames, fps=args.fps)
        summary.append(
            {
                "index": index,
                "psnr": psnr,
                "ssim": ssim,
                "reference": reference_kind,
                "stream": f"{sample['key'][0]}/ep{sample['key'][1]}/{sample['key'][2]}",
                "path": path.name,
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


if __name__ == "__main__":
    main()
