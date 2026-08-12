import io
import json
from typing import TypedDict

import cv2
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


def _describe_record_location(source: grain.ArrayRecordDataSource, paths: list[str], idx: int) -> str:
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
            raise ValueError(f"Failed to decode dynamics record idx={idx} ({location}) from {self._data_dir}") from exc
        return DynamicsDataset(
            video=video,
            actions=actions,
            rewards=rewards,
            prev_action=prev_action,
        )


class SubsetDataSource(grain.RandomAccessDataSource):
    """A source restricted to the given record indices."""

    def __init__(self, source: grain.RandomAccessDataSource, indices: list[int]):
        self._source = source
        self._indices = list(indices)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._source!r}, {len(self._indices)} records)"

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._source[self._indices[idx]]


def load_record_lengths(data_dir: str) -> list[int] | None:
    """Frame counts per record, written by stitch_dynamics_records.py."""
    path = epath.Path(data_dir) / "lengths.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


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
            raise ValueError(f"Failed to decode video record idx={idx} ({location}) from {self._data_dir}") from exc
        return VideoDataset(video=video)


class RandomVideoCrop(grain.RandomMapTransform):
    def __init__(self, frame_length: int):
        self.frame_length = frame_length

    def random_map(self, element: VideoDataset, rng: np.random.Generator) -> VideoDataset:
        video = element["video"]
        start_idx = int(rng.integers(0, len(video) - self.frame_length + 1))
        return VideoDataset(video=video[start_idx : start_idx + self.frame_length].copy())


class RandomDynamicsCrop(grain.RandomMapTransform):
    def __init__(self, sequence_length: int, stride: int = 1):
        self.sequence_length = sequence_length
        self.stride = stride

    def random_map(
        self,
        element: DynamicsDataset,
        rng: np.random.Generator,
    ) -> DynamicsBatch:
        video = element["video"]
        actions = element["actions"]
        prev_action = element["prev_action"]
        span = (self.sequence_length - 1) * self.stride + 1
        if len(video) < span:
            raise ValueError(f"Sequence shorter than crop span: {len(video)} < {span}")
        start_idx = int(rng.integers(0, len(video) - span + 1))
        indices = start_idx + np.arange(self.sequence_length) * self.stride
        aligned = actions[indices - 1]
        if start_idx == 0:
            aligned[0] = prev_action
        return DynamicsBatch(video=video[indices], actions=aligned)


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
                return {
                    "video": data["video"].tobytes(),
                    "length": int(data["length"]),
                    "fps": float(data["fps"]) if "fps" in data else 0.0,
                }
        except Exception as exc:
            location = _describe_record_location(self._source, self._paths, idx)
            raise ValueError(f"Failed to decode video record idx={idx} ({location}) from {self._data_dir}") from exc


class DecodeRandomVideoClip(grain.RandomMapTransform):
    # stride matches the dynamics frame rate; short clips pad by repeating the last frame
    def __init__(
        self,
        frame_length: int,
        decode_hw: tuple[int, int] | None = None,
        stride: int = 1,
        target_hz: float | None = None,
    ):
        self.frame_length = frame_length
        self.decode_hw = decode_hw
        self.stride = max(int(stride), 1)
        self.target_hz = target_hz

    def _stride_for(self, element: dict) -> int:
        if self.target_hz and element.get("fps", 0.0) > 0:
            return max(int(round(element["fps"] / self.target_hz)), 1)
        return self.stride

    def random_map(self, element: dict, rng: np.random.Generator) -> VideoDataset:
        length = element["length"]
        stride = self._stride_for(element)
        span = (self.frame_length - 1) * stride + 1
        start = 0 if length <= span else int(rng.integers(0, length - span + 1))
        frames = decode_video_window(element["video"], start, span, self.decode_hw)
        frames = frames[::stride][: self.frame_length]
        if len(frames) < self.frame_length:
            pad = np.repeat(frames[-1:], self.frame_length - len(frames), axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        return VideoDataset(video=frames)


class WeightedVideoBytesDataSource(VideoBytesDataSource):
    """Repeats each record in the sampling index in proportion to its duration."""

    def __init__(self, data_dir: str, weight_seconds: float = 25.6):
        super().__init__(data_dir)
        base = epath.Path(data_dir)
        lengths = json.loads((base / "lengths.json").read_text())
        rates = json.loads((base / "fps.json").read_text())
        if len(lengths) != len(self._source) or len(rates) != len(self._source):
            raise ValueError(
                f"sidecar length mismatch in {data_dir}: "
                f"{len(lengths)} lengths, {len(rates)} fps, {len(self._source)} records"
            )
        self._index = []
        for record_idx, (n_frames, fps) in enumerate(zip(lengths, rates)):
            duration = n_frames / fps if fps > 0 else n_frames / 30.0
            repeats = max(int(np.ceil(duration / weight_seconds)), 1)
            self._index.extend([record_idx] * repeats)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        return super().__getitem__(self._index[idx])


class AugmentVideoClip(grain.RandomMapTransform):
    # Parameters are drawn once per clip, not per frame: a fresh crop each frame
    # would synthesize camera motion. No flip, which would contradict the actions.
    def __init__(
        self,
        crop_scale: float = 0.95,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.0,
        prob: float = 1.0,
    ):
        self.crop_scale = crop_scale
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob

    def random_map(self, element: VideoDataset, rng: np.random.Generator) -> VideoDataset:
        video = element["video"]
        if self.prob < 1.0 and rng.random() >= self.prob:
            return VideoDataset(video=video)

        _, height, width, _ = video.shape

        if self.crop_scale < 1.0:
            crop_h, crop_w = int(height * self.crop_scale), int(width * self.crop_scale)
            top = int(rng.integers(0, height - crop_h + 1))
            left = int(rng.integers(0, width - crop_w + 1))
            video = video[:, top : top + crop_h, left : left + crop_w, :]
            video = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in video])

        out = video.astype(np.float32)
        if self.saturation > 0:
            factor = 1.0 + float(rng.uniform(-self.saturation, self.saturation))
            gray = out @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            out = gray[..., None] + (out - gray[..., None]) * factor
        if self.contrast > 0:
            factor = 1.0 + float(rng.uniform(-self.contrast, self.contrast))
            out = out.mean() + (out - out.mean()) * factor
        if self.brightness > 0:
            out *= 1.0 + float(rng.uniform(-self.brightness, self.brightness))
        if self.hue > 0:
            shift = float(rng.uniform(-self.hue, self.hue)) * 180.0
            hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in np.clip(out, 0, 255).astype(np.uint8)])
            hsv[..., 0] = (hsv[..., 0].astype(np.float32) + shift) % 180
            out = np.stack([cv2.cvtColor(f, cv2.COLOR_HSV2RGB) for f in hsv]).astype(np.float32)

        return VideoDataset(video=np.clip(out, 0, 255).astype(np.uint8))
