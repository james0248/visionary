import io
from typing import TypedDict

import grain.python as grain
import numpy as np
from etils import epath


class VideoDataset(TypedDict):
    video: np.ndarray


class DynamicsBatch(TypedDict):
    video: np.ndarray
    actions: np.ndarray


class DynamicsDataset(DynamicsBatch):
    rewards: np.ndarray
    prev_action: np.ndarray


def align_actions_to_frames(
    actions: np.ndarray,
    prev_action: np.ndarray | None = None,
) -> np.ndarray:
    aligned = np.empty_like(actions)
    if prev_action is None:
        prev_action = np.full(actions.shape[1:], -1, dtype=actions.dtype)
    aligned[0] = prev_action
    aligned[1:] = actions[:-1]
    return aligned


def _array_record_source_with_paths(data_dir: str) -> tuple[grain.ArrayRecordDataSource, list[str]]:
    shard_dir = epath.Path(data_dir)
    paths = sorted(
        [p for p in shard_dir.iterdir() if p.suffix == ".arecord"],
        key=lambda p: p.as_posix(),
    )
    if not paths:
        raise FileNotFoundError(f"No .arecord files found in {data_dir}")
    path_strings = [p.as_posix() for p in paths]
    return grain.ArrayRecordDataSource(path_strings), path_strings


def _describe_record_location(
    source: grain.ArrayRecordDataSource, paths: list[str], idx: int
) -> str:
    if hasattr(source, "_reader_idx_and_position"):
        try:
            reader_idx, position = source._reader_idx_and_position(idx)
            return f"shard={paths[reader_idx]!r} record={position}"
        except Exception:
            pass
    if len(paths) == 1:
        return f"shard={paths[0]!r}"
    return f"global_record={idx} among {len(paths)} shards"


class DynamicsDataSource(grain.RandomAccessDataSource):
    def __init__(self, data_dir: str):
        self._data_dir = epath.Path(data_dir).as_posix()
        self._source, self._paths = _array_record_source_with_paths(self._data_dir)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data_dir={self._data_dir!r})"

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx: int) -> DynamicsDataset:
        try:
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
        except Exception as exc:
            location = _describe_record_location(self._source, self._paths, idx)
            raise ValueError(
                f"Failed to decode dynamics record idx={idx} ({location}) from {self._data_dir}"
            ) from exc
        return DynamicsDataset(
            video=video,
            actions=actions,
            rewards=rewards,
            prev_action=prev_action,
        )


class VideoDataSource(grain.RandomAccessDataSource):
    def __init__(self, data_dir: str):
        self._data_dir = epath.Path(data_dir).as_posix()
        self._source, self._paths = _array_record_source_with_paths(self._data_dir)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data_dir={self._data_dir!r})"

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx: int) -> VideoDataset:
        try:
            record_bytes = self._source[idx]
            with np.load(io.BytesIO(record_bytes)) as data:
                video = np.asarray(data["frames"])
        except Exception as exc:
            location = _describe_record_location(self._source, self._paths, idx)
            raise ValueError(
                f"Failed to decode video record idx={idx} ({location}) from {self._data_dir}"
            ) from exc
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


def decode_video_window(
    source: str | bytes,
    start: int,
    length: int,
    decode_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    import av

    container = av.open(io.BytesIO(source) if isinstance(source, bytes) else source)
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        if start > 0:
            container.seek(int(start / fps / stream.time_base), stream=stream, backward=True)
        frames = []
        for frame in container.decode(video=0):
            if int(round(float(frame.time) * fps)) < start:
                continue
            if decode_hw is not None:
                frame = frame.reformat(width=decode_hw[1], height=decode_hw[0], format="rgb24")
            frames.append(np.asarray(frame.to_ndarray(format="rgb24")))
            if len(frames) >= length:
                break
    finally:
        container.close()
    if not frames:
        raise ValueError(f"Decoded no frames at start={start}")
    return np.stack(frames)


class VideoBytesDataSource(grain.RandomAccessDataSource):
    """ArrayRecord shards whose records hold an encoded mp4 plus its frame count."""

    def __init__(self, data_dir: str):
        self._data_dir = epath.Path(data_dir).as_posix()
        self._source, self._paths = _array_record_source_with_paths(self._data_dir)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data_dir={self._data_dir!r})"

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx: int) -> dict:
        try:
            with np.load(io.BytesIO(self._source[idx])) as data:
                return {"video": data["video"].tobytes(), "length": int(data["length"])}
        except Exception as exc:
            location = _describe_record_location(self._source, self._paths, idx)
            raise ValueError(
                f"Failed to decode video record idx={idx} ({location}) from {self._data_dir}"
            ) from exc


class DecodeRandomVideoClip(grain.RandomMapTransform):
    def __init__(self, frame_length: int, decode_hw: tuple[int, int] | None = None):
        self.frame_length = frame_length
        self.decode_hw = decode_hw

    def random_map(self, element: dict, rng: np.random.Generator) -> VideoDataset:
        length = element["length"]
        start = (
            0
            if length <= self.frame_length
            else int(rng.integers(0, length - self.frame_length + 1))
        )
        frames = decode_video_window(element["video"], start, self.frame_length, self.decode_hw)
        return VideoDataset(video=frames)
