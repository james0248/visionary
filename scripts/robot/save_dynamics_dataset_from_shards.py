"""Encode the packed video shards into latent dynamics records.

Reads the tokenizer's ArrayRecord shards (one record per episode and camera),
encodes every frame with a trained tokenizer export, and writes one dynamics
record per episode. The input train/eval split is preserved, so
tokenizer-eval episodes stay out of dynamics-train.

Episodes are encoded in windows of the tokenizer's training length (its
temporal RoPE never saw longer offsets) and the window batch is sharded over
every local device. Actions are normalized to [-1, 1] with q01-q99 stats
computed from the train shards.

Each embodiment is encoded by its own run, selected by action dimensionality;
per-record striding unifies every source to --target_hz.

    uv run python scripts/robot/save_dynamics_dataset_from_shards.py \
        --checkpoint_dir gs://visionary-uc1/so101/checkpoints/tokenizer_v2 \
        --input_dir data/tokenizer/shards --output_dir data/so101/dynamics \
        --min_episode_length 8 --encode_window_length 16 --encode_window_overlap 8 \
        --action_dim 6
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
from visionary.common.checkpoint import (
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.dataset import decode_video_window
from visionary.models.dreamer4.tokenizer import Tokenizer
from visionary.models.dreamer4.tokenizer_preprocessor import TokenizerPreprocessor
from visionary.shards import (
    ShardWriter,
    SplitStats,
    build_action_normalizer,
    chunk_starts,
)

logger = logging.getLogger(__name__)

GB = 1024**3


class RunningLatentStats:
    def __init__(self) -> None:
        self.count = 0
        self.frames = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, latents: np.ndarray) -> None:
        self.frames += int(latents.shape[0])
        for start in range(0, len(latents), 64):
            values = latents[start : start + 64].reshape(-1, latents.shape[-1]).astype(np.float64)
            batch_count = int(values.shape[0])
            batch_mean = values.mean(axis=0)
            batch_m2 = np.sum((values - batch_mean) ** 2, axis=0)
            if self.mean is None:
                self.count = batch_count
                self.mean = batch_mean
                self.m2 = batch_m2
                continue
            total = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean += delta * (batch_count / total)
            self.m2 += batch_m2 + delta**2 * (self.count * batch_count / total)
            self.count = total

    def to_dict(self) -> dict[str, Any]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise ValueError("No training latents were encoded")
        std = np.sqrt(self.m2 / self.count)
        if not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("Invalid training latent standard deviation")
        return {
            "count": self.count,
            "frames": self.frames,
            "mean": self.mean.tolist(),
            "std": std.tolist(),
        }


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

        self.tokenizer_cfg, variables = restore_model_export_single_device(checkpoint_dir, step=step)
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
        encoded = [np.empty((len(patches), *self.latents_per_frame), dtype=self.latent_dtype) for patches in episodes]
        for episode_idx, patches in enumerate(episodes):
            prev_stop = 0
            for start in chunk_starts(len(patches), self.window_length, self.window_overlap):
                stop = min(start + self.window_length, len(patches))
                window_refs.append((episode_idx, start, stop, max(prev_stop - start, 0)))
                prev_stop = stop
        if not window_refs:
            return encoded

        token_shape = episodes[0].shape[1:]
        batch = np.zeros((self.batch_size, self.window_length, *token_shape), dtype=episodes[0].dtype)
        lengths = np.zeros((self.batch_size,), dtype=np.int32)

        for batch_start in range(0, len(window_refs), self.batch_size):
            batch.fill(0)
            lengths.fill(0)
            refs = window_refs[batch_start : batch_start + self.batch_size]
            for window_idx, (episode_idx, start, stop, _) in enumerate(refs):
                batch[window_idx, : stop - start] = episodes[episode_idx][start:stop]
                lengths[window_idx] = stop - start

            sharded = jax.device_put(batch, self.batch_sharding)
            latents = np.asarray(jax.device_get(self.encode_fn(self.variables, sharded)), dtype=np.float32)
            for window_idx, (episode_idx, start, stop, overlap) in enumerate(refs):
                encoded[episode_idx][start + overlap : stop] = latents[
                    window_idx, overlap : lengths[window_idx]
                ].astype(self.latent_dtype, copy=False)

        return encoded


def load_stream(
    record_bytes: bytes,
    episode_id: int,
    preprocessor: TokenizerPreprocessor,
    normalizer: Callable[[np.ndarray], np.ndarray],
    target_hz: float,
    action_dim: int,
) -> Stream | None:
    with np.load(io.BytesIO(record_bytes)) as data:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.shape[-1] != action_dim:  # other embodiment; encoded by its own run
            return None
        video = data["video"].tobytes()
        length = int(data["length"])
        fps = float(data["fps"])
        state = np.asarray(data["state"], dtype=np.float32)
        provenance = {
            "repo": str(data["repo"]),
            "episode": np.int32(data["episode"]),
            "camera": str(data["camera"]),
        }

    stride = max(int(round(fps / target_hz)), 1)
    frames = decode_video_window(video, 0, length, tuple(preprocessor.resize_shape))
    length = min(len(frames), len(actions), len(state))
    frames = frames[:length:stride]
    actions = actions[:length:stride]
    state = state[:length:stride]

    return Stream(
        episode_id=episode_id,
        patches=preprocessor.preprocess_video(frames),
        arrays={
            "actions": normalizer(actions),
            "state": state,
        },
        provenance=provenance,
    )


def iter_loaded_streams(
    source: grain.ArrayRecordDataSource,
    items: list[int],
    load_fn: Callable[..., Stream],
    read_workers: int,
    prefetch: int,
):
    with ThreadPoolExecutor(max_workers=read_workers) as executor:
        pending: dict[int, Any] = {}
        next_submit = 0
        next_yield = 0

        def submit(item_idx: int) -> Any:
            episode_id = items[item_idx]
            return executor.submit(load_fn, source[episode_id], episode_id)

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


def encode_stream_record(stream: Stream, latents: np.ndarray) -> bytes:
    payload: dict[str, Any] = dict(stream.arrays)
    payload["frames"] = latents
    payload["prev_action"] = np.zeros(stream.arrays["actions"].shape[1:], dtype=np.float32)
    payload["episode_id"] = np.asarray(stream.episode_id, dtype=np.int64)
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
    latent_stats: RunningLatentStats | None = None,
) -> SplitStats:
    # global stream indices, so episode ids stay stable across ranged runs
    start = min(args.stream_start, len(source))
    stop = len(source) if args.stream_stop is None else min(args.stream_stop, len(source))
    if args.limit is not None:
        stop = min(stop, start + args.limit)
    items = list(range(start, stop))

    shard_writer = ShardWriter(output_dir, args.records_per_shard)
    stats = SplitStats(episodes_found=len(items))
    start_time = time.monotonic()
    pending: list[Stream] = []

    load_fn = functools.partial(
        load_stream,
        preprocessor=encoder.preprocessor,
        normalizer=normalizer,
        target_hz=args.target_hz,
        action_dim=args.action_dim,
    )

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        latents_batch = encoder.encode_episodes([stream.patches for stream in pending])
        for stream, latents in zip(pending, latents_batch, strict=True):
            if latent_stats is not None:
                latent_stats.update(latents)
            record_bytes = encode_stream_record(stream, latents)
            shard_writer.write(record_bytes)
            stats.records_written += 1
            stats.payload_bytes += len(record_bytes)
            stats.frames_written += len(latents)
            stats.episodes_written += 1
        pending = []

    try:
        for count, stream in enumerate(
            iter_loaded_streams(source, items, load_fn, args.read_workers, args.prefetch_episodes),
            start=1,
        ):
            if stream is None or len(stream.patches) < args.min_episode_length:
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


def compute_action_stats(
    source: grain.ArrayRecordDataSource,
    out_path: Path,
    action_dim: int,
    target_hz: float,
) -> None:
    chunks: list[np.ndarray] = []
    seen: set[tuple[str, int]] = set()
    for idx in range(len(source)):
        with np.load(io.BytesIO(source[idx])) as data:
            actions_array = np.asarray(data["actions"], dtype=np.float64)
            if actions_array.shape[-1] != action_dim:
                continue
            key = (str(data["repo"]), int(data["episode"]))
            if key in seen:  # multi-view streams share one trajectory
                continue
            seen.add(key)
            fps = float(data["fps"])
            stride = max(int(round(fps / target_hz)), 1)
            chunks.append(actions_array[::stride])
    actions = np.concatenate(chunks, axis=0)
    stats = {
        "n_episodes": len(chunks),
        "n_frames": int(actions.shape[0]),
        "action_dim": action_dim,
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
    parser = argparse.ArgumentParser(description="Encode packed video shards into episode dynamics records.")
    parser.add_argument("--checkpoint_dir", required=True, help="Tokenizer checkpoint directory.")
    parser.add_argument("--input_dir", required=True, help="Shard root with train/ and eval/.")
    parser.add_argument("--output_dir", required=True, help="Output directory for latent shards.")
    parser.add_argument(
        "--min_episode_length",
        type=int,
        default=8,
        help="Episodes shorter than this after resampling are skipped.",
    )
    parser.add_argument("--step", type=int, required=True, help="Tokenizer export step.")
    parser.add_argument(
        "--action_dim",
        type=int,
        required=True,
        help="Embodiment filter: only streams with this action dimensionality are encoded.",
    )
    parser.add_argument(
        "--target_hz",
        type=float,
        default=5.0,
        help="Per-record stride = round(fps / target_hz), unifying the frame rate across sources.",
    )
    parser.add_argument(
        "--encode_window_length",
        type=int,
        default=16,
        help="Frames per tokenizer forward pass; must match the tokenizer's training length.",
    )
    parser.add_argument(
        "--encode_window_overlap",
        type=int,
        default=8,
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
    parser.add_argument("--read_workers", type=int, default=8, help="Threads decoding and patchifying video.")
    parser.add_argument("--prefetch_episodes", type=int, default=8, help="Streams queued ahead of the encoder.")
    parser.add_argument("--records_per_shard", type=int, default=256, help="Maximum records per .arecord shard.")
    parser.add_argument(
        "--latent_dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Latent dtype stored on disk.",
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
        help="First stream index to encode into an empty output directory.",
    )
    parser.add_argument("--stream_stop", type=int, help="Stop stream index (exclusive). Defaults to the split end.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    args = create_parser().parse_args()

    if jax.process_count() > 1:
        raise RuntimeError("Single-host script; run one process and it uses every local device.")
    if args.encode_batch_size % jax.local_device_count() != 0:
        raise ValueError(f"encode_batch_size must be divisible by the {jax.local_device_count()} local devices")
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
    splits = ("train", "eval") if args.split == "both" else (args.split,)
    needed = set(splits) | {"train"}
    sources = {split: open_split_source(input_dir, split) for split in sorted(needed)}
    for split, source in sources.items():
        logger.info("Found %d %s streams", len(source), split)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "norm_stats.json"
    if stats_path.exists():
        logger.info("Reusing action stats at %s", stats_path)
    else:
        logger.info("Computing action stats over the train shards")
        compute_action_stats(sources["train"], stats_path, args.action_dim, args.target_hz)
    normalizer = build_action_normalizer("continuous", str(stats_path))

    train_latent_stats = RunningLatentStats() if "train" in splits else None
    split_stats = {}
    for split in splits:
        split_stats[split] = write_split(
            split,
            sources[split],
            encoder,
            args,
            normalizer,
            output_dir / split,
            latent_stats=train_latent_stats if split == "train" else None,
        )

    latent_stats_payload = None
    if train_latent_stats is not None:
        latent_stats_payload = train_latent_stats.to_dict()
        latent_stats_path = output_dir / "latent_stats.json"
        latent_stats_path.write_text(json.dumps(latent_stats_payload, indent=2))
        logger.info(
            "Wrote latent stats over %d frames to %s",
            latent_stats_payload["frames"],
            latent_stats_path,
        )

    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_step": args.step,
        "action_stats": str(stats_path),
        "latent_stats": latent_stats_payload,
        "config": {
            "tokenizer": OmegaConf.to_container(encoder.tokenizer_cfg, resolve=True),
            "preprocessor": dict(encoder.preprocessor_cfg),
        },
        "token_dataset": {
            "latent_shape_per_frame": list(encoder.latents_per_frame),
            "latent_dtype": args.latent_dtype,
            "action_dim": args.action_dim,
            "target_hz": args.target_hz,
            "min_episode_length": args.min_episode_length,
            "encode_window_length": args.encode_window_length,
            "encode_window_overlap": args.encode_window_overlap,
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
