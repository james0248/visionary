"""Build a chunked latent dataset for the Dreamer 4 dynamics model.

The input is a directory of raw per-episode `.npz` files that contain aligned
`frames`, `actions`, and `rewards`. The output is a pair of `train/` and
`eval/` ArrayRecord directories whose records contain tokenizer latents instead
of pixels.
"""

import argparse
import hashlib
import io
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from array_record.python.array_record_module import ArrayRecordWriter
from hydra.utils import instantiate
from omegaconf import OmegaConf

from visionary.common.checkpoint import CheckpointManager
from visionary.common.train_state import TokenizerTrainState
from visionary.dataset import PreprocessAndPatchify
from visionary.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def chunk_starts(length: int, chunk_length: int, overlap: int) -> list[int]:
    if length <= chunk_length:
        return [0]
    stride = chunk_length - overlap
    return list(range(0, length, stride))


@dataclass(frozen=True)
class FileSplits:
    train_files: list[Path]
    eval_files: list[Path]
    metadata: dict[str, Any]

    @classmethod
    def from_build_config(cls, build_cfg: "BuildConfig") -> "FileSplits":
        def list_files(data_dir: str) -> list[Path]:
            return sorted(Path(data_dir).rglob("*.npz"), key=lambda path: path.as_posix())

        if build_cfg.eval_input_dir is not None:
            train_files = list_files(build_cfg.input_dir)
            eval_files = list_files(build_cfg.eval_input_dir)
            if build_cfg.max_files is not None:
                train_files = train_files[: build_cfg.max_files]
                eval_files = eval_files[: build_cfg.max_files]
            metadata = {
                "split_mode": "separate_dirs",
                "train_input_dir": build_cfg.input_dir,
                "eval_input_dir": build_cfg.eval_input_dir,
            }
            return cls(train_files=train_files, eval_files=eval_files, metadata=metadata)

        files = list_files(build_cfg.input_dir)
        if build_cfg.max_files is not None:
            files = files[: build_cfg.max_files]

        train_files: list[Path] = []
        eval_files: list[Path] = []
        for path in files:
            digest = hashlib.sha256(
                f"{build_cfg.seed}:{path.as_posix()}".encode("utf-8")
            ).digest()
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
    eval_input_dir: str | None
    output_dir: Path
    config_path: str
    seed: int
    eval_ratio: float
    max_files: int | None
    chunk_length: int
    chunk_overlap: int
    min_length: int
    encode_window_length: int
    encode_window_overlap: int
    records_per_shard: int
    latent_dtype_name: str
    compressed: bool

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

        min_length = args.min_length
        if min_length is None:
            min_length = int(cfg.dataset.frame_length)

        if not 0 <= args.chunk_overlap < args.chunk_length:
            raise ValueError(
                f"Expected 0 <= chunk_overlap < chunk_length, got "
                f"{args.chunk_overlap=} {args.chunk_length=}"
            )
        if not 0 <= args.encode_window_overlap < encode_window_length:
            raise ValueError(
                f"Expected 0 <= encode_window_overlap < encode_window_length, got "
                f"{args.encode_window_overlap=} {encode_window_length=}"
            )

        return cls(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_step=args.step,
            input_dir=args.input_dir,
            eval_input_dir=args.eval_input_dir,
            output_dir=Path(args.output_dir),
            config_path=args.config,
            seed=args.seed,
            eval_ratio=args.eval_ratio,
            max_files=args.max_files,
            chunk_length=args.chunk_length,
            chunk_overlap=args.chunk_overlap,
            min_length=min_length,
            encode_window_length=encode_window_length,
            encode_window_overlap=args.encode_window_overlap,
            records_per_shard=args.records_per_shard,
            latent_dtype_name=args.latent_dtype,
            compressed=bool(args.compressed),
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


class TokenizerEncoder:
    def __init__(self, cfg: Any, build_cfg: BuildConfig) -> None:
        self.cfg = cfg
        self.build_cfg = build_cfg
        self.model = instantiate(self.cfg.tokenizer)
        self.transform = PreprocessAndPatchify(
            int(cfg.dataset.patch_size),
            (int(cfg.dataset.pad_width[0]), int(cfg.dataset.pad_width[1])),
            tuple(int(value) for value in cfg.dataset.resize_shape),
        )
        self.latents_per_frame = [int(cfg.tokenizer.num_latents), int(cfg.tokenizer.channel_dim)]

        @jax.jit
        def encode_step(params, batch):
            return self.model.apply(params, batch, method=Tokenizer.encode)

        self.encode_fn = encode_step
        self.state = self._restore_state()

    def _restore_state(self) -> TokenizerTrainState:
        init_key, sample_key = jax.random.split(jax.random.key(self.build_cfg.seed))
        patch_count = int(self.cfg.tokenizer.x_len) * int(self.cfg.tokenizer.y_len)
        patch_dim = int(self.cfg.dataset.patch_size) * int(self.cfg.dataset.patch_size) * 3
        dummy_batch = {"video": np.zeros((1, 1, patch_count, patch_dim), dtype=np.uint8)}
        state = TokenizerTrainState.create(
            apply_fn=self.model.apply,
            params=self.model.init(
                {"params": init_key, "sample": sample_key},
                dummy_batch,
            ),
            tx=optax.adam(0.0),
            mse_sq_ema=jnp.ones((), dtype=jnp.float32),
            l1_sq_ema=jnp.ones((), dtype=jnp.float32),
            lpips_sq_ema=jnp.ones((), dtype=jnp.float32),
            motion_sq_ema=jnp.ones((), dtype=jnp.float32),
        )
        with CheckpointManager(
            self.build_cfg.checkpoint_dir, ocp.CheckpointManagerOptions()
        ) as manager:
            return manager.restore(target=state, step=self.build_cfg.checkpoint_step)

    def encode_episode(self, frames: np.ndarray) -> np.ndarray:
        starts = chunk_starts(
            len(frames),
            self.build_cfg.encode_window_length,
            self.build_cfg.encode_window_overlap,
        )
        rng = np.random.default_rng(0)
        encoded_parts: list[np.ndarray] = []
        prev_stop = 0

        for start in starts:
            stop = min(start + self.build_cfg.encode_window_length, len(frames))
            batch = self.transform.random_map({"video": frames[start:stop]}, rng)
            latent = self.encode_fn(self.state.params, {"video": batch["video"][None, ...]})
            latent = np.asarray(jax.device_get(latent[0]), dtype=np.float32)

            overlap = max(prev_stop - start, 0)
            if overlap:
                latent = latent[overlap:]
            if self.build_cfg.latent_dtype != np.float32:
                latent = latent.astype(self.build_cfg.latent_dtype, copy=False)

            encoded_parts.append(latent)
            prev_stop = stop

        if not encoded_parts:
            return np.empty((0, 0, 0), dtype=self.build_cfg.latent_dtype)
        return np.concatenate(encoded_parts, axis=0)


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

    for episode_id, path in enumerate(files):
        with np.load(path) as episode:
            arrays = {key: np.asarray(episode[key]) for key in episode.files}
        if "frames" not in arrays:
            raise KeyError(f"Episode {path} does not contain a 'frames' array")
        episode_length = int(arrays["frames"].shape[0])
        if episode_length < build_cfg.min_length:
            stats.episodes_skipped += 1
            continue

        latents = encoder.encode_episode(arrays["frames"])

        for start in chunk_starts(
            episode_length, build_cfg.chunk_length, build_cfg.chunk_overlap
        ):
            stop = min(start + build_cfg.chunk_length, episode_length)
            if stop - start < build_cfg.min_length:
                continue

            payload: dict[str, np.ndarray] = {
                "frames": latents[start:stop],
                "episode_id": np.asarray(episode_id, dtype=np.int64),
                "start_index": np.asarray(start, dtype=np.int32),
            }
            for key, value in arrays.items():
                if key != "frames" and value.ndim > 0 and value.shape[0] == episode_length:
                    payload[key] = value[start:stop]

            buffer = io.BytesIO()
            if build_cfg.compressed:
                np.savez_compressed(buffer, **payload)
            else:
                np.savez(buffer, **payload)
            record_bytes = buffer.getvalue()
            shard_writer.write(record_bytes)
            stats.records_written += 1
            stats.frames_written += stop - start
            stats.payload_bytes += len(record_bytes)

        stats.episodes_written += 1
        if (episode_id + 1) % 10 == 0 or episode_id + 1 == len(files):
            elapsed = time.monotonic() - start_time
            logger.info(
                "%s split: processed %d/%d episodes, wrote %d records in %.1fs",
                split_name,
                episode_id + 1,
                len(files),
                stats.records_written,
                elapsed,
            )

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
        help="Hydra config used to instantiate the tokenizer and preprocessing.",
    )
    parser.add_argument("--step", type=int, help="Checkpoint step to restore. Defaults to latest.")
    parser.add_argument(
        "--eval_input_dir",
        help="Optional separate eval directory. If omitted, train/eval are split by hash.",
    )
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="Eval ratio for hash split.")
    parser.add_argument("--seed", type=int, default=42, help="Hash-split and init seed.")
    parser.add_argument(
        "--max_files", type=int, help="Optional cap on the number of episodes per split."
    )
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
        "--min_length",
        type=int,
        help="Skip records and episodes shorter than this many frames. Defaults to tokenizer frame_length.",
    )
    parser.add_argument(
        "--encode_window_length",
        type=int,
        help="Frames per tokenizer forward pass. Defaults to tokenizer frame_length.",
    )
    parser.add_argument(
        "--encode_window_overlap",
        type=int,
        default=0,
        help="Overlap between tokenizer encode windows, in frames.",
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
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed for each payload instead of np.savez.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = create_parser().parse_args()
    cfg = OmegaConf.load(args.config)
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
            "compressed": build_cfg.compressed,
            "chunk_length": build_cfg.chunk_length,
            "chunk_overlap": build_cfg.chunk_overlap,
            "min_length": build_cfg.min_length,
            "encode_window_length": build_cfg.encode_window_length,
            "encode_window_overlap": build_cfg.encode_window_overlap,
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
