import argparse
import hashlib
import io
import itertools
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter
from omegaconf import OmegaConf

from visionary.common.tokenizer_checkpoint import (
    normalize_tokenizer_config,
    restore_tokenizer_checkpoint_bundle,
)
from visionary.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def chunk_starts(length: int, chunk_length: int, overlap: int) -> list[int]:
    if length <= chunk_length:
        return [0]
    stride = chunk_length - overlap
    starts = list(range(0, length - chunk_length + 1, stride))
    last = length - chunk_length
    if starts[-1] != last:
        starts.append(last)
    return starts


@dataclass(frozen=True)
class FileSplits:
    train_files: list[Path]
    eval_files: list[Path]
    metadata: dict[str, Any]

    @classmethod
    def from_build_config(cls, build_cfg: "BuildConfig") -> "FileSplits":
        def list_files(data_dir: str) -> list[Path]:
            return sorted(Path(data_dir).rglob("*.npz"), key=lambda path: path.as_posix())

        files = list_files(build_cfg.input_dir)

        train_files: list[Path] = []
        eval_files: list[Path] = []
        for path in files:
            digest = hashlib.sha256(f"{build_cfg.seed}:{path.as_posix()}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], byteorder="big") / 2**64
            (eval_files if bucket < build_cfg.eval_ratio else train_files).append(path)

        return cls(
            train_files=train_files,
            eval_files=eval_files,
            metadata={
                "split_mode": "hashed",
                "input_dir": build_cfg.input_dir,
                "eval_ratio": build_cfg.eval_ratio,
                "seed": build_cfg.seed,
            },
        )


@dataclass(frozen=True)
class BuildConfig:
    checkpoint_dir: str
    checkpoint_step: int | None
    input_dir: str
    output_dir: Path
    config_path: str
    seed: int
    eval_ratio: float
    chunk_length: int
    chunk_overlap: int
    min_length: int
    encode_window_length: int
    encode_window_overlap: int
    encode_batch_size: int
    encode_episode_batch_size: int
    read_workers: int
    prefetch_episodes: int
    record_workers: int
    records_per_shard: int
    latent_dtype_name: str

    @property
    def latent_dtype(self) -> np.dtype:
        return {
            "float16": np.dtype(np.float16),
            "float32": np.dtype(np.float32),
        }[self.latent_dtype_name]

    @classmethod
    def from_args(cls, args: argparse.Namespace, cfg: Any) -> "BuildConfig":
        encode_window_length = args.encode_window_length
        if encode_window_length is None:
            encode_window_length = int(cfg.dataset.frame_length)

        min_length = int(cfg.dataset.frame_length)
        encode_window_overlap = 0

        if not 0 <= args.chunk_overlap < args.chunk_length:
            raise ValueError(
                f"Expected 0 <= chunk_overlap < chunk_length, got "
                f"{args.chunk_overlap=} {args.chunk_length=}"
            )
        if args.encode_batch_size <= 0:
            raise ValueError(f"Expected encode_batch_size > 0, got {args.encode_batch_size}")
        if args.encode_episode_batch_size <= 0:
            raise ValueError(
                f"Expected encode_episode_batch_size > 0, got {args.encode_episode_batch_size}"
            )
        if args.read_workers <= 0:
            raise ValueError(f"Expected read_workers > 0, got {args.read_workers}")
        if args.prefetch_episodes <= 0:
            raise ValueError(f"Expected prefetch_episodes > 0, got {args.prefetch_episodes}")
        if args.record_workers <= 0:
            raise ValueError(f"Expected record_workers > 0, got {args.record_workers}")

        return cls(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_step=args.step,
            input_dir=args.input_dir,
            output_dir=Path(args.output_dir),
            config_path=args.config,
            seed=args.seed,
            eval_ratio=args.eval_ratio,
            chunk_length=args.chunk_length,
            chunk_overlap=args.chunk_overlap,
            min_length=min_length,
            encode_window_length=encode_window_length,
            encode_window_overlap=encode_window_overlap,
            encode_batch_size=args.encode_batch_size,
            encode_episode_batch_size=args.encode_episode_batch_size,
            read_workers=args.read_workers,
            prefetch_episodes=args.prefetch_episodes,
            record_workers=args.record_workers,
            records_per_shard=args.records_per_shard,
            latent_dtype_name=args.latent_dtype,
        )


@dataclass
class SplitStats:
    episodes_found: int = 0
    episodes_written: int = 0
    episodes_skipped: int = 0
    records_written: int = 0
    frames_written: int = 0
    payload_bytes: int = 0
    shards_written: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "episodes_found": self.episodes_found,
            "episodes_written": self.episodes_written,
            "episodes_skipped": self.episodes_skipped,
            "records_written": self.records_written,
            "frames_written": self.frames_written,
            "payload_bytes": self.payload_bytes,
            "shards_written": self.shards_written,
        }


class ShardWriter:
    def __init__(self, output_dir: Path, records_per_shard: int) -> None:
        self.output_dir = output_dir
        self.records_per_shard = records_per_shard
        self.shard_idx = 0
        self.records_in_shard = 0
        self.shards_written = 0

        self._prepare_output_dir()
        self._open_new_writer()

    def _prepare_output_dir(self) -> None:
        if self.output_dir.exists() and any(self.output_dir.glob("*.arecord")):
            raise FileExistsError(
                f"Output directory {self.output_dir} already contains .arecord files. "
                "Choose a new path or clear the directory first."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _open_new_writer(self) -> None:
        self.current_path = self.output_dir / f"shard-{self.shard_idx:05d}.arecord"
        self.writer = ArrayRecordWriter(self.current_path.as_posix(), "group_size:1")

    def write(self, record_bytes: bytes) -> None:
        self.writer.write(record_bytes)
        self.records_in_shard += 1
        if self.records_in_shard == self.records_per_shard:
            self._rotate()

    def _rotate(self) -> None:
        self.writer.close()
        self.shards_written += 1
        self.shard_idx += 1
        self.records_in_shard = 0
        self._open_new_writer()

    def close(self) -> int:
        self.writer.close()
        if self.records_in_shard > 0:
            self.shards_written += 1
        else:
            self.current_path.unlink(missing_ok=True)
        return self.shards_written


def load_episode_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as episode:
        return {key: np.asarray(episode[key]) for key in episode.files}


def encode_record(
    arrays: dict[str, np.ndarray],
    latents: np.ndarray,
    episode_id: int,
    start: int,
    stop: int,
) -> bytes:
    payload: dict[str, np.ndarray] = {
        "frames": latents[start:stop],
        "episode_id": np.asarray(episode_id, dtype=np.int64),
        "start_index": np.asarray(start, dtype=np.int32),
    }
    episode_length = int(arrays["frames"].shape[0])
    for key, value in arrays.items():
        if key != "frames" and value.ndim > 0 and value.shape[0] == episode_length:
            payload[key] = value[start:stop]
            if key == "actions":
                payload["prev_action"] = (
                    np.asarray(value[start - 1], dtype=value.dtype)
                    if start > 0
                    else np.full(value.shape[1:], -1, dtype=value.dtype)
                )

    buffer = io.BytesIO()
    np.savez(buffer, **payload)
    return buffer.getvalue()


def record_bounds(
    length: int,
    *,
    chunk_length: int,
    overlap: int,
    min_length: int,
) -> list[tuple[int, int]]:
    bounds = []
    for start in chunk_starts(length, chunk_length, overlap):
        stop = min(start + chunk_length, length)
        if stop - start >= min_length:
            bounds.append((start, stop))
    return bounds


def iter_loaded_episodes(
    files: list[Path],
    *,
    read_workers: int,
    prefetch_episodes: int,
):
    if read_workers == 1:
        for episode_id, path in enumerate(files):
            yield episode_id, path, load_episode_arrays(path)
        return

    with ThreadPoolExecutor(max_workers=read_workers) as executor:
        pending: dict[int, Any] = {}
        next_submit = 0
        next_yield = 0

        while next_submit < min(len(files), prefetch_episodes):
            pending[next_submit] = executor.submit(load_episode_arrays, files[next_submit])
            next_submit += 1

        while next_yield < len(files):
            yield next_yield, files[next_yield], pending.pop(next_yield).result()
            if next_submit < len(files):
                pending[next_submit] = executor.submit(load_episode_arrays, files[next_submit])
                next_submit += 1
            next_yield += 1


class TokenizerEncoder:
    def __init__(self, cfg: Any, build_cfg: BuildConfig) -> None:
        bundle = restore_tokenizer_checkpoint_bundle(
            build_cfg.checkpoint_dir,
            seed=build_cfg.seed,
            checkpoint_step=build_cfg.checkpoint_step,
            config=cfg,
        )
        self.cfg = bundle.config
        self.build_cfg = build_cfg
        self.model = bundle.model
        self.state = bundle.state
        self.latents_per_frame = (
            int(self.cfg.tokenizer.num_latents),
            int(self.cfg.tokenizer.channel_dim),
        )

        @jax.jit
        def encode_step(params, video_batch):
            return self.model.apply(
                params,
                {"video": jnp.asarray(video_batch)},
                method=Tokenizer.encode,
            )

        self.encode_fn = encode_step

    def encode_episode(self, frames: np.ndarray) -> np.ndarray:
        return self.encode_episodes([frames])[0]

    def encode_episodes(self, episodes: list[np.ndarray]) -> list[np.ndarray]:
        if not episodes:
            return []

        window_refs: list[tuple[int, int, int, int]] = []
        encoded = [
            np.empty((len(frames), *self.latents_per_frame), dtype=self.build_cfg.latent_dtype)
            for frames in episodes
        ]
        frame_shape = episodes[0].shape[1:]
        frame_dtype = episodes[0].dtype

        for episode_idx, frames in enumerate(episodes):
            prev_stop = 0
            for start in chunk_starts(
                len(frames),
                self.build_cfg.encode_window_length,
                self.build_cfg.encode_window_overlap,
            ):
                stop = min(start + self.build_cfg.encode_window_length, len(frames))
                overlap = max(prev_stop - start, 0)
                window_refs.append((episode_idx, start, stop, overlap))
                prev_stop = stop

        if not window_refs:
            return encoded

        batch_frames = np.zeros(
            (
                self.build_cfg.encode_batch_size,
                self.build_cfg.encode_window_length,
                *frame_shape,
            ),
            dtype=frame_dtype,
        )
        batch_lengths = np.zeros((self.build_cfg.encode_batch_size,), dtype=np.int32)

        for batch_start in range(0, len(window_refs), self.build_cfg.encode_batch_size):
            batch_frames.fill(0)
            batch_lengths.fill(0)
            batch_window_refs = window_refs[
                batch_start : batch_start + self.build_cfg.encode_batch_size
            ]

            for window_idx, (episode_idx, start, stop, _) in enumerate(batch_window_refs):
                length = stop - start
                batch_frames[window_idx, :length] = episodes[episode_idx][start:stop]
                batch_lengths[window_idx] = length

            batch_latents = np.asarray(
                jax.device_get(self.encode_fn(self.state.params, batch_frames)),
                dtype=np.float32,
            )
            for window_idx, (episode_idx, start, stop, overlap) in enumerate(batch_window_refs):
                encoded[episode_idx][start + overlap : stop] = batch_latents[
                    window_idx,
                    overlap : batch_lengths[window_idx],
                ].astype(self.build_cfg.latent_dtype, copy=False)

        return encoded


def iter_record_bytes(
    arrays: dict[str, np.ndarray],
    latents: np.ndarray,
    *,
    episode_id: int,
    chunk_length: int,
    overlap: int,
    min_length: int,
    executor: ThreadPoolExecutor | None,
):
    bounds = record_bounds(
        int(arrays["frames"].shape[0]),
        chunk_length=chunk_length,
        overlap=overlap,
        min_length=min_length,
    )

    if executor is None:
        for start, stop in bounds:
            yield encode_record(arrays, latents, episode_id, start, stop)
        return

    yield from executor.map(
        encode_record,
        itertools.repeat(arrays),
        itertools.repeat(latents),
        itertools.repeat(episode_id),
        (start for start, _ in bounds),
        (stop for _, stop in bounds),
    )


def write_split(
    split_name: str,
    files: list[Path],
    encoder: TokenizerEncoder,
    build_cfg: BuildConfig,
    output_dir: Path,
) -> SplitStats:
    shard_writer = ShardWriter(output_dir, build_cfg.records_per_shard)
    stats = SplitStats(episodes_found=len(files))
    start_time = time.monotonic()
    pending_episodes: list[tuple[int, dict[str, np.ndarray]]] = []
    record_executor = None
    if build_cfg.record_workers > 1:
        record_executor = ThreadPoolExecutor(max_workers=build_cfg.record_workers)

    def flush_pending() -> None:
        nonlocal pending_episodes
        if not pending_episodes:
            return

        latents_batch = encoder.encode_episodes(
            [arrays["frames"] for _, arrays in pending_episodes]
        )
        for (episode_id, arrays), latents in zip(pending_episodes, latents_batch, strict=True):
            bounds = record_bounds(
                int(arrays["frames"].shape[0]),
                chunk_length=build_cfg.chunk_length,
                overlap=build_cfg.chunk_overlap,
                min_length=build_cfg.min_length,
            )
            for record_bytes in iter_record_bytes(
                arrays,
                latents,
                episode_id=episode_id,
                chunk_length=build_cfg.chunk_length,
                overlap=build_cfg.chunk_overlap,
                min_length=build_cfg.min_length,
                executor=record_executor,
            ):
                shard_writer.write(record_bytes)
                stats.records_written += 1
                stats.payload_bytes += len(record_bytes)
            stats.frames_written += sum(stop - start for start, stop in bounds)
            stats.episodes_written += 1

        pending_episodes = []

    try:
        for episode_id, path, arrays in iter_loaded_episodes(
            files,
            read_workers=build_cfg.read_workers,
            prefetch_episodes=build_cfg.prefetch_episodes,
        ):
            if "frames" not in arrays:
                raise KeyError(f"Episode {path} does not contain a 'frames' array")
            episode_length = int(arrays["frames"].shape[0])
            if episode_length < build_cfg.min_length:
                stats.episodes_skipped += 1
                continue

            pending_episodes.append((episode_id, arrays))
            should_flush = len(pending_episodes) == build_cfg.encode_episode_batch_size
            is_last = episode_id + 1 == len(files)
            if should_flush or is_last:
                flush_pending()

            if (episode_id + 1) % 10 == 0 or episode_id + 1 == len(files):
                elapsed = time.monotonic() - start_time
                logger.info(
                    "%s split: processed %d/%d episodes, wrote %d records in %.1fs (%.1f frames/s)",
                    split_name,
                    episode_id + 1,
                    len(files),
                    stats.records_written,
                    elapsed,
                    stats.frames_written / max(elapsed, 1e-6),
                )
    finally:
        if record_executor is not None:
            record_executor.shutdown(wait=True)
        stats.shards_written = shard_writer.close()
    return stats


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenize raw episode NPZ files into chunked ArrayRecord dynamics data."
    )

    parser.add_argument("--checkpoint_dir", required=True, help="Tokenizer checkpoint directory.")
    parser.add_argument("--input_dir", required=True, help="Directory of raw episode .npz files.")
    parser.add_argument("--output_dir", required=True, help="Output directory for token shards.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config" / "breakout.yaml"),
        help="Tokenizer config used for dataset build settings and tokenizer restore.",
    )
    parser.add_argument("--step", type=int, help="Checkpoint step to restore. Defaults to latest.")
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="Eval ratio for hash split.")
    parser.add_argument("--seed", type=int, default=42, help="Hash-split and init seed.")
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=512,
        help="Frames stored per output token record before the tail chunk.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=0,
        help="Overlap between output token records, in frames.",
    )
    parser.add_argument(
        "--encode_window_length",
        type=int,
        help="Frames per tokenizer forward pass. Defaults to tokenizer frame_length.",
    )
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=16,
        help="Number of encode windows packed into one tokenizer forward pass.",
    )
    parser.add_argument(
        "--encode_episode_batch_size",
        type=int,
        default=2,
        help="Number of loaded episodes grouped into one tokenizer encode pass.",
    )
    parser.add_argument(
        "--read_workers",
        type=int,
        default=4,
        help="Number of background threads used to load and decompress episode files.",
    )
    parser.add_argument(
        "--prefetch_episodes",
        type=int,
        default=16,
        help="Number of episode files to keep queued ahead of the encoder.",
    )
    parser.add_argument(
        "--record_workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of threads used to build ArrayRecord payload bytes after encoding.",
    )
    parser.add_argument(
        "--records_per_shard", type=int, default=1024, help="Maximum records per .arecord shard."
    )
    parser.add_argument(
        "--latent_dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Latent dtype stored on disk.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = create_parser().parse_args()
    cfg = normalize_tokenizer_config(OmegaConf.load(args.config))
    build_cfg = BuildConfig.from_args(args, cfg)

    logger.info("Initializing tokenizer from %s", build_cfg.config_path)
    encoder = TokenizerEncoder(cfg, build_cfg)

    file_splits = FileSplits.from_build_config(build_cfg)
    logger.info(
        "Found %d train and %d eval episodes",
        len(file_splits.train_files),
        len(file_splits.eval_files),
    )

    build_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train_stats = write_split(
        "train",
        file_splits.train_files,
        encoder,
        build_cfg,
        build_cfg.output_dir / "train",
    )
    eval_stats = write_split(
        "eval",
        file_splits.eval_files,
        encoder,
        build_cfg,
        build_cfg.output_dir / "eval",
    )

    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint_dir": build_cfg.checkpoint_dir,
        "checkpoint_step": build_cfg.checkpoint_step,
        "config": {
            "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
            "tokenizer": OmegaConf.to_container(cfg.tokenizer, resolve=True),
        },
        "token_dataset": {
            "latent_shape_per_frame": encoder.latents_per_frame,
            "latent_dtype": build_cfg.latent_dtype_name,
            "chunk_length": build_cfg.chunk_length,
            "chunk_overlap": build_cfg.chunk_overlap,
            "min_length": build_cfg.min_length,
            "encode_window_length": build_cfg.encode_window_length,
            "encode_window_overlap": build_cfg.encode_window_overlap,
            "encode_batch_size": build_cfg.encode_batch_size,
            "encode_episode_batch_size": build_cfg.encode_episode_batch_size,
            "read_workers": build_cfg.read_workers,
            "prefetch_episodes": build_cfg.prefetch_episodes,
            "record_workers": build_cfg.record_workers,
            "records_per_shard": build_cfg.records_per_shard,
        },
        "split": file_splits.metadata,
        "stats": {
            "train": train_stats.to_dict(),
            "eval": eval_stats.to_dict(),
        },
    }
    metadata_path = build_cfg.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Wrote metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
