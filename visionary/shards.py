import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter


def chunk_starts(length: int, chunk_length: int, overlap: int) -> list[int]:
    if length <= chunk_length:
        return [0]
    stride = chunk_length - overlap
    starts = list(range(0, length - chunk_length + 1, stride))
    last = length - chunk_length
    if starts[-1] != last:
        starts.append(last)
    return starts


def record_bounds(
    length: int,
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


def build_action_normalizer(action_mode: str, action_stats_path: str | None):
    if action_mode != "continuous":
        return None
    if not action_stats_path:
        raise ValueError("--action_mode continuous requires --action_stats <norm_stats.json>")
    stats = json.loads(Path(action_stats_path).read_text())
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    span = np.where((q99 - q01) > 1e-6, q99 - q01, 1.0)

    def normalize(actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        return np.clip(2.0 * (actions - q01) / span - 1.0, -1.0, 1.0).astype(np.float32)

    return normalize
