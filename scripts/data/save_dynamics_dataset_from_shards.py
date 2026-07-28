"""Encode the packed video shards into latent dynamics records.

Reads the tokenizer's ArrayRecord shards (one record per (episode, fixed
camera) stream: trimmed mp4 bytes plus aligned actions), encodes every frame
with a trained tokenizer export, and writes the chunked ArrayRecord format
train_dynamics.py reads. The input train/eval split is preserved, so
tokenizer-eval episodes stay out of dynamics-train.

Episodes are encoded in windows of the tokenizer's training length (its
temporal RoPE never saw longer offsets) and the window batch is sharded over
every local device. Actions are normalized to [-1, 1] with q01-q99 stats
computed from the shards themselves unless --action_stats is given.

    uv run python scripts/data/save_dynamics_dataset_from_shards.py \
        --checkpoint_dir gs://visionary-robot-bucket/so101/checkpoints/so101_tokenizer \
        --input_dir data/so101/shards --output_dir data/so101/dyn --frame_length 64
"""

import argparse
import functools
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import grain.python as grain
import jax
import numpy as np
from hydra.utils import instantiate
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from omegaconf import OmegaConf
from save_dynamics_dataset import (
    ShardWriter,
    SplitStats,
    build_action_normalizer,
    chunk_starts,
    record_bounds,
)

from visionary.common.checkpoint import (
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.dataset import AugmentVideoClip, decode_video_window
from visionary.tokenizer import Tokenizer
from visionary.tokenizer_preprocessor import TokenizerPreprocessor

logger = logging.getLogger(__name__)

GB = 1024**3


@dataclass
class Stream:
    episode_id: int
    patches: np.ndarray
    arrays: dict[str, np.ndarray]
    provenance: dict[str, Any]


class ShardedTokenizerEncoder:
    def __init__(
        self,
        checkpoint_dir: str,
        step: int | None,
        window_length: int,
        window_overlap: int,
        batch_size: int,
        latent_dtype: np.dtype,
    ) -> None:
        self.window_length = window_length
        self.window_overlap = window_overlap
        self.batch_size = batch_size
        self.latent_dtype = latent_dtype

        self.tokenizer_cfg, variables = restore_model_export_single_device(
            checkpoint_dir, step=step
        )
        self.preprocessor_cfg = restore_preprocessor_export(checkpoint_dir, step=step)
        self.model = instantiate(self.tokenizer_cfg)
        self.preprocessor = TokenizerPreprocessor.from_config(self.preprocessor_cfg)
        self.latents_per_frame = (
            int(self.model.num_latents),
            int(self.model.channel_dim),
        )

        mesh = Mesh(np.asarray(jax.local_devices()), ("data",))
        self.batch_sharding = NamedSharding(mesh, P("data"))
        self.variables = jax.device_put(variables, NamedSharding(mesh, P()))
        model = self.model

        @jax.jit
        def encode_step(variables, patch_batch):
            return model.apply(variables, {"video": patch_batch}, method=Tokenizer.encode)

        self.encode_fn = encode_step

    def encode_episodes(self, episodes: list[np.ndarray]) -> list[np.ndarray]:
        window_refs: list[tuple[int, int, int, int]] = []
        encoded = [
            np.empty((len(patches), *self.latents_per_frame), dtype=self.latent_dtype)
            for patches in episodes
        ]
        for episode_idx, patches in enumerate(episodes):
            prev_stop = 0
            for start in chunk_starts(len(patches), self.window_length, self.window_overlap):
                stop = min(start + self.window_length, len(patches))
                window_refs.append((episode_idx, start, stop, max(prev_stop - start, 0)))
                prev_stop = stop
        if not window_refs:
            return encoded

        token_shape = episodes[0].shape[1:]
        batch = np.zeros(
            (self.batch_size, self.window_length, *token_shape), dtype=episodes[0].dtype
        )
        lengths = np.zeros((self.batch_size,), dtype=np.int32)

        for batch_start in range(0, len(window_refs), self.batch_size):
            batch.fill(0)
            lengths.fill(0)
            refs = window_refs[batch_start : batch_start + self.batch_size]
            for window_idx, (episode_idx, start, stop, _) in enumerate(refs):
                batch[window_idx, : stop - start] = episodes[episode_idx][start:stop]
                lengths[window_idx] = stop - start

            sharded = jax.device_put(batch, self.batch_sharding)
            latents = np.asarray(
                jax.device_get(self.encode_fn(self.variables, sharded)), dtype=np.float32
            )
            for window_idx, (episode_idx, start, stop, overlap) in enumerate(refs):
                encoded[episode_idx][start + overlap : stop] = latents[
                    window_idx, overlap : lengths[window_idx]
                ].astype(self.latent_dtype, copy=False)

        return encoded


def load_stream(
    record_bytes: bytes,
    episode_id: int,
    copy: int,
    preprocessor: TokenizerPreprocessor,
    normalizer: Callable[[np.ndarray], np.ndarray],
    augment: AugmentVideoClip,
    seed: int,
) -> Stream:
    with np.load(io.BytesIO(record_bytes)) as data:
        video = data["video"].tobytes()
        length = int(data["length"])
        actions = np.asarray(data["actions"], dtype=np.float32)
        state = np.asarray(data["state"], dtype=np.float32)
        provenance = {
            "repo": str(data["repo"]),
            "episode": np.int32(data["episode"]),
            "camera": str(data["camera"]),
            "augment_copy": np.int32(copy),
        }

    frames = decode_video_window(video, 0, length, tuple(preprocessor.resize_shape))
    length = min(len(frames), len(actions))
    frames, actions, state = frames[:length], actions[:length], state[:length]
    if copy > 0:
        # one draw per episode; per-window draws would jump brightness mid-episode
        frames = augment.random_map(
            {"video": frames}, np.random.default_rng([seed, episode_id, copy])
        )["video"]

    return Stream(
        episode_id=episode_id,
        patches=preprocessor.preprocess_video(frames),
        arrays={
            "actions": normalizer(actions),
            "state": state,
            "rewards": np.zeros((length,), dtype=np.float32),  # teleop demos have none
        },
        provenance=provenance,
    )


def iter_loaded_streams(
    source: grain.ArrayRecordDataSource,
    items: list[tuple[int, int]],
    load_fn: Callable[..., Stream],
    read_workers: int,
    prefetch: int,
):
    with ThreadPoolExecutor(max_workers=read_workers) as executor:
        pending: dict[int, Any] = {}
        next_submit = 0
        next_yield = 0

        def submit(item_idx: int) -> Any:
            episode_id, copy = items[item_idx]
            return executor.submit(load_fn, source[episode_id], episode_id, copy)

        while next_submit < min(len(items), prefetch):
            pending[next_submit] = submit(next_submit)
            next_submit += 1

        while next_yield < len(items):
            try:
                stream = pending.pop(next_yield).result()
            except Exception:
                logger.exception("Skipping stream %s", items[next_yield])
                stream = None
            if next_submit < len(items):
                pending[next_submit] = submit(next_submit)
                next_submit += 1
            yield stream
            next_yield += 1


def encode_stream_record(stream: Stream, latents: np.ndarray, start: int, stop: int) -> bytes:
    actions = stream.arrays["actions"]
    payload: dict[str, Any] = {key: value[start:stop] for key, value in stream.arrays.items()}
    payload["frames"] = latents[start:stop]
    payload["prev_action"] = (
        actions[start - 1] if start > 0 else np.zeros(actions.shape[1:], dtype=actions.dtype)
    )
    payload["episode_id"] = np.asarray(stream.episode_id, dtype=np.int64)
    payload["start_index"] = np.asarray(start, dtype=np.int32)
    payload.update(stream.provenance)

    buffer = io.BytesIO()
    np.savez(buffer, **payload)
    return buffer.getvalue()


def write_split(
    split_name: str,
    source: grain.ArrayRecordDataSource,
    encoder: ShardedTokenizerEncoder,
    args: argparse.Namespace,
    normalizer: Callable[[np.ndarray], np.ndarray],
    output_dir: Path,
) -> SplitStats:
    copies = 1 + (args.augment_copies if split_name == "train" else 0)
    # global stream indices, so episode ids stay stable across ranged runs
    start = min(args.stream_start, len(source))
    stop = len(source) if args.stream_stop is None else min(args.stream_stop, len(source))
    if args.limit is not None:
        stop = min(stop, start + args.limit)
    items = [(idx, copy) for idx in range(start, stop) for copy in range(copies)]

    shard_writer = ShardWriter(output_dir, args.records_per_shard)
    stats = SplitStats(episodes_found=len(items))
    start_time = time.monotonic()
    pending: list[Stream] = []

    load_fn = functools.partial(
        load_stream,
        preprocessor=encoder.preprocessor,
        normalizer=normalizer,
        augment=AugmentVideoClip(),  # defaults match the tokenizer training config
        seed=args.seed,
    )

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        latents_batch = encoder.encode_episodes([stream.patches for stream in pending])
        for stream, latents in zip(pending, latents_batch, strict=True):
            bounds = record_bounds(
                len(latents),
                chunk_length=args.chunk_length,
                overlap=args.chunk_overlap,
                min_length=args.frame_length,
            )
            for start, stop in bounds:
                record_bytes = encode_stream_record(stream, latents, start, stop)
                shard_writer.write(record_bytes)
                stats.records_written += 1
                stats.payload_bytes += len(record_bytes)
            stats.frames_written += sum(stop - start for start, stop in bounds)
            stats.episodes_written += 1
        pending = []

    try:
        for count, stream in enumerate(
            iter_loaded_streams(source, items, load_fn, args.read_workers, args.prefetch_episodes),
            start=1,
        ):
            if stream is None or len(stream.patches) < args.frame_length:
                stats.episodes_skipped += 1
            else:
                pending.append(stream)
                if len(pending) == args.encode_episode_batch_size:
                    flush()
            if count % 50 == 0 or count == len(items):
                elapsed = time.monotonic() - start_time
                logger.info(
                    "%s split: processed %d/%d streams, wrote %d records in %.1fs (%.1f frames/s)",
                    split_name,
                    count,
                    len(items),
                    stats.records_written,
                    elapsed,
                    stats.frames_written / max(elapsed, 1e-6),
                )
        flush()
    finally:
        stats.shards_written = shard_writer.close()
    return stats


def compute_action_stats(source: grain.ArrayRecordDataSource, out_path: Path) -> None:
    chunks: list[np.ndarray] = []
    seen: set[tuple[str, int]] = set()
    for idx in range(len(source)):
        with np.load(io.BytesIO(source[idx])) as data:
            key = (str(data["repo"]), int(data["episode"]))
            if key in seen:  # multi-view streams share one trajectory
                continue
            seen.add(key)
            chunks.append(np.asarray(data["actions"], dtype=np.float64))
    actions = np.concatenate(chunks, axis=0)
    stats = {
        "n_episodes": len(chunks),
        "n_frames": int(actions.shape[0]),
        "units": "degrees",
        "normalization": "q01_q99_to_[-1,1]_clipped (gripper included)",
        "mean": actions.mean(0).round(4).tolist(),
        "std": actions.std(0).round(4).tolist(),
        "q01": np.quantile(actions, 0.01, axis=0).round(4).tolist(),
        "q99": np.quantile(actions, 0.99, axis=0).round(4).tolist(),
    }
    out_path.write_text(json.dumps(stats, indent=1))
    logger.info("Wrote action stats over %d episodes to %s", len(chunks), out_path)


def open_split_source(input_dir: Path, split: str) -> grain.ArrayRecordDataSource:
    paths = sorted(str(path) for path in (input_dir / split).glob("*.arecord"))
    if not paths:
        raise FileNotFoundError(f"No .arecord files found in {input_dir / split}")
    return grain.ArrayRecordDataSource(paths)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode packed video shards into chunked ArrayRecord dynamics data."
    )
    parser.add_argument("--checkpoint_dir", required=True, help="Tokenizer checkpoint directory.")
    parser.add_argument("--input_dir", required=True, help="Shard root with train/ and eval/.")
    parser.add_argument("--output_dir", required=True, help="Output directory for latent shards.")
    parser.add_argument(
        "--frame_length",
        type=int,
        required=True,
        help="Minimum frames per output record; shorter streams are skipped. "
        "Must be >= the dynamics batch_length.",
    )
    parser.add_argument("--step", type=int, help="Checkpoint step to restore. Defaults to latest.")
    parser.add_argument("--seed", type=int, default=42, help="Augmentation seed.")
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=512,
        help="Frames stored per output record before the tail chunk.",
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=0, help="Overlap between output records, in frames."
    )
    parser.add_argument(
        "--encode_window_length",
        type=int,
        default=32,
        help="Frames per tokenizer forward pass; must match the tokenizer's training length.",
    )
    parser.add_argument(
        "--encode_window_overlap",
        type=int,
        default=0,
        help="Warm-up frames re-encoded per window so kept latents see this much context.",
    )
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=64,
        help="Windows per forward pass, sharded over local devices.",
    )
    parser.add_argument(
        "--encode_episode_batch_size",
        type=int,
        default=8,
        help="Loaded streams grouped into one encode pass.",
    )
    parser.add_argument(
        "--read_workers", type=int, default=8, help="Threads decoding and patchifying video."
    )
    parser.add_argument(
        "--prefetch_episodes", type=int, default=8, help="Streams queued ahead of the encoder."
    )
    parser.add_argument(
        "--records_per_shard", type=int, default=1024, help="Maximum records per .arecord shard."
    )
    parser.add_argument(
        "--latent_dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Latent dtype stored on disk. The latents are held near N(0, 1), so "
        "float16 halves the dataset at no meaningful precision cost.",
    )
    parser.add_argument(
        "--action_stats",
        default=None,
        help="Path to norm_stats.json (q01/q99). Defaults to computing stats from the "
        "train shards and writing norm_stats.json into --output_dir.",
    )
    parser.add_argument(
        "--augment_copies",
        type=int,
        default=0,
        help="Extra augmented encodings written per train stream (eval stays clean).",
    )
    parser.add_argument("--limit", type=int, help="Streams per split, for smoke tests.")
    parser.add_argument(
        "--split",
        choices=("both", "train", "eval"),
        default="both",
        help="Which input split to encode.",
    )
    parser.add_argument(
        "--stream_start",
        type=int,
        default=0,
        help="First stream index to encode; episode ids stay global, so ranged "
        "runs compose into one resumable dataset.",
    )
    parser.add_argument(
        "--stream_stop", type=int, help="Stop stream index (exclusive). Defaults to the split end."
    )
    return parser


def main() -> None:
    # force: grain's import installs a root handler, which would swallow basicConfig
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    args = create_parser().parse_args()

    if jax.process_count() > 1:
        raise RuntimeError("Single-host script; run one process and it uses every local device.")
    if args.encode_batch_size % jax.local_device_count() != 0:
        raise ValueError(
            f"encode_batch_size must be divisible by the {jax.local_device_count()} local devices"
        )
    if not 0 <= args.encode_window_overlap < args.encode_window_length:
        raise ValueError("Expected 0 <= encode_window_overlap < encode_window_length")

    logger.info(
        "Initializing tokenizer from checkpoint %s (step=%s) on %d device(s)",
        args.checkpoint_dir,
        args.step if args.step is not None else "latest",
        jax.local_device_count(),
    )
    encoder = ShardedTokenizerEncoder(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        window_length=args.encode_window_length,
        window_overlap=args.encode_window_overlap,
        batch_size=args.encode_batch_size,
        latent_dtype=np.dtype(args.latent_dtype),
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    sources = {split: open_split_source(input_dir, split) for split in ("train", "eval")}
    logger.info("Found %d train and %d eval streams", len(sources["train"]), len(sources["eval"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.action_stats:
        stats_path = Path(args.action_stats)
    else:
        stats_path = output_dir / "norm_stats.json"
        logger.info("Computing action stats over the train shards")
        compute_action_stats(sources["train"], stats_path)
    normalizer = build_action_normalizer("continuous", str(stats_path))

    splits = ("train", "eval") if args.split == "both" else (args.split,)
    split_stats = {
        split: write_split(split, sources[split], encoder, args, normalizer, output_dir / split)
        for split in splits
    }

    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_step": args.step,
        "action_stats": str(stats_path),
        "config": {
            "tokenizer": OmegaConf.to_container(encoder.tokenizer_cfg, resolve=True),
            "preprocessor": dict(encoder.preprocessor_cfg),
        },
        "token_dataset": {
            "latent_shape_per_frame": list(encoder.latents_per_frame),
            "latent_dtype": args.latent_dtype,
            "frame_length": args.frame_length,
            "chunk_length": args.chunk_length,
            "chunk_overlap": args.chunk_overlap,
            "encode_window_length": args.encode_window_length,
            "encode_window_overlap": args.encode_window_overlap,
            "augment_copies": args.augment_copies,
            "split": args.split,
            "stream_start": args.stream_start,
            "stream_stop": args.stream_stop,
        },
        "stats": {split: stats.to_dict() for split, stats in split_stats.items()},
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info(
        "Wrote %.1f GB of latents and metadata to %s",
        sum(stats.payload_bytes for stats in split_stats.values()) / GB,
        metadata_path,
    )


if __name__ == "__main__":
    main()
