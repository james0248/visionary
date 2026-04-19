import bisect
import hashlib
import io
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, TypedDict

from array_record.python import array_record_module
import grain.python as grain
import jax
import numpy as np
from einops import rearrange
from etils import epath


class VideoDataset(TypedDict):
    video: np.ndarray


class DynamicsBatch(TypedDict):
    video: np.ndarray
    actions: np.ndarray


class DynamicsDataset(DynamicsBatch):
    rewards: np.ndarray
    prev_action: np.ndarray


DEFAULT_MAX_OPEN_ARECORD_READERS = 64


def _max_open_arecord_readers() -> int:
    value = os.environ.get("VISIONARY_MAX_OPEN_ARECORD_READERS")
    if value is None:
        return DEFAULT_MAX_OPEN_ARECORD_READERS
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("VISIONARY_MAX_OPEN_ARECORD_READERS must be an integer") from exc
    if parsed < 1:
        raise ValueError("VISIONARY_MAX_OPEN_ARECORD_READERS must be >= 1")
    return parsed


def _create_arecord_reader(path: str) -> Any:
    return array_record_module.ArrayRecordReader(
        path,
        options="readahead_buffer_size:0",
        file_reader_buffer_size=32768,
    )


@dataclass(slots=True)
class _ReaderHandle:
    reader: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    refcount: int = 0


class _BoundedArrayRecordDataSource(grain.RandomAccessDataSource):
    def __init__(self, paths: list[str], max_open_readers: int):
        if max_open_readers < 1:
            raise ValueError("max_open_readers must be >= 1")
        self._paths = list(paths)
        self._max_open_readers = int(max_open_readers)
        self._cache_lock = threading.Lock()
        self._readers: OrderedDict[int, _ReaderHandle] = OrderedDict()
        self._prefix_sums: list[int] = []

        total = 0
        for path in self._paths:
            reader = _create_arecord_reader(path)
            try:
                total += int(reader.num_records())
            finally:
                reader.close()
            self._prefix_sums.append(total)
        self._num_records = total

    def __repr__(self) -> str:
        digest = hashlib.sha1()
        for path in self._paths:
            digest.update(path.encode())
        return (
            f"{type(self).__name__}("
            f"hash_of_paths={digest.hexdigest()}, "
            f"max_open_readers={self._max_open_readers})"
        )

    def __len__(self) -> int:
        return self._num_records

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        with self._cache_lock:
            readers = list(self._readers.values())
            self._readers.clear()
        for handle in readers:
            handle.reader.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_cache_lock", None)
        state.pop("_readers", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()
        self._readers = OrderedDict()

    def _reader_idx_and_position(self, record_key: int) -> tuple[int, int]:
        if record_key < 0 or record_key >= self._num_records:
            raise ValueError("Record key should be in [0, num_records)")
        reader_idx = bisect.bisect_right(self._prefix_sums, record_key)
        records_in_previous_shards = self._prefix_sums[reader_idx - 1] if reader_idx > 0 else 0
        return reader_idx, record_key - records_in_previous_shards

    def _evict_readers_locked(self) -> None:
        while len(self._readers) > self._max_open_readers:
            evict_idx = None
            for reader_idx, handle in self._readers.items():
                if handle.refcount == 0:
                    evict_idx = reader_idx
                    break
            if evict_idx is None:
                return
            handle = self._readers.pop(evict_idx)
            handle.reader.close()

    def _acquire_reader(self, reader_idx: int) -> _ReaderHandle:
        with self._cache_lock:
            handle = self._readers.get(reader_idx)
            if handle is None:
                handle = _ReaderHandle(reader=_create_arecord_reader(self._paths[reader_idx]))
                self._readers[reader_idx] = handle
            else:
                self._readers.move_to_end(reader_idx)
            handle.refcount += 1
            self._evict_readers_locked()
            return handle

    def _release_reader(self, reader_idx: int) -> None:
        with self._cache_lock:
            handle = self._readers.get(reader_idx)
            if handle is None:
                return
            handle.refcount -= 1
            self._evict_readers_locked()

    def __getitem__(self, record_key: int) -> bytes:
        reader_idx, position = self._reader_idx_and_position(int(record_key))
        handle = self._acquire_reader(reader_idx)
        try:
            with handle.lock:
                if hasattr(handle.reader, "read"):
                    return handle.reader.read([position])[0]
                return handle.reader[position]
        finally:
            self._release_reader(reader_idx)

    def __getitems__(self, record_keys: list[int]) -> list[bytes]:
        return [self[key] for key in record_keys]


def align_actions_to_frames(
    actions: np.ndarray,
    *,
    prev_action: np.ndarray | None = None,
) -> np.ndarray:
    aligned = np.empty_like(actions)
    if prev_action is None:
        prev_action = np.full(actions.shape[1:], -1, dtype=actions.dtype)
    aligned[0] = prev_action
    aligned[1:] = actions[:-1]
    return aligned


def _array_record_source(data_dir: str) -> _BoundedArrayRecordDataSource:
    shard_dir = epath.Path(data_dir)
    paths = sorted(
        [p for p in shard_dir.iterdir() if p.suffix == ".arecord"],
        key=lambda p: p.as_posix(),
    )
    if not paths:
        raise FileNotFoundError(f"No .arecord files found in {data_dir}")
    return _BoundedArrayRecordDataSource(
        [p.as_posix() for p in paths],
        max_open_readers=_max_open_arecord_readers(),
    )


class DynamicsDataSource(grain.RandomAccessDataSource):
    def __init__(self, data_dir: str):
        self._data_dir = epath.Path(data_dir).as_posix()
        self._source = _array_record_source(self._data_dir)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data_dir={self._data_dir!r})"

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx: int) -> DynamicsDataset:
        record_bytes = self._source[idx]
        with np.load(io.BytesIO(record_bytes)) as data:
            video = np.asarray(data["frames"])
            actions = np.asarray(data["actions"])
            rewards = np.asarray(data["rewards"])
            prev_action = (
                np.asarray(data["prev_action"])
                if "prev_action" in data
                else np.full(actions.shape[1:], -1, dtype=actions.dtype)
            )
        return DynamicsDataset(
            video=video,
            actions=actions,
            rewards=rewards,
            prev_action=prev_action,
        )


class VideoDataSource(grain.RandomAccessDataSource):
    def __init__(self, data_dir: str):
        self._data_dir = epath.Path(data_dir).as_posix()
        self._source = _array_record_source(self._data_dir)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data_dir={self._data_dir!r})"

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx: int) -> VideoDataset:
        record_bytes = self._source[idx]
        with np.load(io.BytesIO(record_bytes)) as data:
            video = np.asarray(data["frames"])
        return VideoDataset(video=video)


class RandomVideoCrop(grain.RandomMapTransform):
    def __init__(self, frame_length: int):
        self.frame_length = frame_length

    def random_map(self, element: VideoDataset, rng: np.random.Generator) -> VideoDataset:
        video = element["video"]
        start_idx = int(rng.integers(0, len(video) - self.frame_length + 1))
        return VideoDataset(video=video[start_idx : start_idx + self.frame_length].copy())


class RandomDynamicsCrop(grain.RandomMapTransform):
    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length

    def random_map(
        self,
        element: DynamicsDataset,
        rng: np.random.Generator,
    ) -> DynamicsBatch:
        video = element["video"]
        actions = element["actions"]
        prev_action = element["prev_action"]
        if len(video) < self.sequence_length:
            raise ValueError(f"Sequence shorter than crop: {len(video)} < {self.sequence_length}")
        if len(video) == self.sequence_length:
            return DynamicsBatch(
                video=video,
                actions=align_actions_to_frames(actions, prev_action=prev_action),
            )

        start_idx = int(rng.integers(0, len(video) - self.sequence_length + 1))
        stop_idx = start_idx + self.sequence_length
        cropped_actions = actions[start_idx:stop_idx]
        crop_prev_action = actions[start_idx - 1] if start_idx > 0 else prev_action
        return DynamicsBatch(
            video=video[start_idx:stop_idx],
            actions=align_actions_to_frames(cropped_actions, prev_action=crop_prev_action),
        )


class PreprocessAndPatchify(grain.RandomMapTransform):
    def __init__(
        self,
        patch_size: int,
        pad_width: tuple[int, int],
        resize_shape: tuple[int, int] | None = None,
    ):
        self.patch_size = patch_size
        self.pad_width = pad_width
        self.resize_shape = resize_shape

    def random_map(self, element: VideoDataset, rng: np.random.Generator) -> VideoDataset:
        video = element["video"]
        if self.resize_shape is not None:
            video = np.clip(
                np.rint(
                    np.asarray(
                        jax.image.resize(
                            video.astype(np.float32),
                            (video.shape[0], *self.resize_shape, video.shape[-1]),
                            method="linear",
                            antialias=True,
                        )
                    )
                ),
                0,
                255,
            ).astype(video.dtype)
        height_pad, width_pad = self.pad_width
        padded_video = np.pad(
            video,
            ((0, 0), (height_pad, height_pad), (width_pad, width_pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        patched_video = rearrange(
            padded_video,
            "t (h p1) (w p2) c -> t (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return {"video": patched_video}
