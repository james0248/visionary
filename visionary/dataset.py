import io
import json
from dataclasses import dataclass
from typing import Iterator, NotRequired, TypedDict

import cv2
import grain.python as grain
import numpy as np
from etils import epath


class VideoDataset(TypedDict):
    video: np.ndarray


class EncodedVideoDataset(TypedDict):
    video: bytes
    length: int
    fps: float


class DynamicsBatch(TypedDict):
    video: np.ndarray
    actions: np.ndarray
    segment_ids: NotRequired[np.ndarray]


class DynamicsDataset(DynamicsBatch):
    rewards: np.ndarray
    prev_action: np.ndarray


@dataclass(frozen=True)
class LatentStats:
    count: int
    mean: np.ndarray
    std: np.ndarray


def load_latent_stats(path: str) -> LatentStats:
    payload = json.loads(epath.Path(path).read_text())
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    count = int(payload["count"])
    if mean.ndim != 1 or std.shape != mean.shape:
        raise ValueError(f"Invalid latent statistics shapes: mean={mean.shape}, std={std.shape}")
    if count <= 0 or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError(f"Invalid latent statistics in {path}")
    return LatentStats(count=count, mean=mean, std=std)


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
                latents = np.asarray(data["frames"])
                actions = np.asarray(data["actions"])
                rewards = np.asarray(data["rewards"])
                prev_action = np.asarray(data["prev_action"])
        except Exception as exc:
            location = _describe_record_location(self._source, self._paths, idx)
            raise ValueError(f"Failed to decode dynamics record idx={idx} ({location}) from {self._data_dir}") from exc
        return DynamicsDataset(
            video=latents,
            actions=actions,
            rewards=rewards,
            prev_action=prev_action,
        )


class RandomDynamicsCrop(grain.RandomMapTransform):
    def __init__(self, sequence_length: int, stride: int = 1):
        self.sequence_length = sequence_length
        self.stride = stride

    def random_map(
        self,
        element: DynamicsDataset,
        rng: np.random.Generator,
    ) -> DynamicsBatch:
        latents = element["video"]
        actions = element["actions"]
        prev_action = element["prev_action"]
        span = (self.sequence_length - 1) * self.stride + 1
        if len(latents) < span:
            raise ValueError(f"Sequence shorter than crop span: {len(latents)} < {span}")
        start_idx = int(rng.integers(0, len(latents) - span + 1))
        indices = start_idx + np.arange(self.sequence_length) * self.stride
        aligned_actions = actions[indices - 1]
        if start_idx == 0:
            aligned_actions[0] = prev_action
        return DynamicsBatch(video=latents[indices], actions=aligned_actions)


class NormalizeDynamicsLatents(grain.MapTransform):
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-8)

    def map(self, element: DynamicsBatch) -> DynamicsBatch:
        latents = np.asarray(element["video"], dtype=np.float32)
        if latents.shape[-1] != self.mean.shape[0]:
            raise ValueError(f"Latent channel mismatch: video={latents.shape[-1]}, stats={self.mean.shape[0]}")
        normalized = DynamicsBatch(
            video=(latents - self.mean) / self.std,
            actions=element["actions"],
        )
        if "segment_ids" in element:
            normalized["segment_ids"] = element["segment_ids"]
        return normalized


class PackDynamicsEpisodes:
    """Pack an episode stream into fixed rows and batches.

    Segment IDs preserve episode boundaries within each row.
    """

    def __init__(self, records: Iterator[DynamicsDataset], sequence_length: int, batch_size: int):
        self.records = records
        self.sequence_length = sequence_length
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[DynamicsBatch]:
        rows: list[DynamicsBatch] = []
        parts_video: list[np.ndarray] = []
        parts_actions: list[np.ndarray] = []
        parts_segments: list[np.ndarray] = []
        row_length = 0
        segment_id = 0

        for record in self.records:
            latents = record["video"]
            previous_actions = np.concatenate(
                [record["prev_action"][None], record["actions"][:-1]],
                axis=0,
            )
            position = 0

            while position < len(latents):
                part_length = min(
                    len(latents) - position,
                    self.sequence_length - row_length,
                )
                part = slice(position, position + part_length)
                parts_video.append(latents[part])
                parts_actions.append(previous_actions[part])
                parts_segments.append(np.full((part_length,), segment_id, dtype=np.int32))
                position += part_length
                row_length += part_length

                if row_length < self.sequence_length:
                    continue

                rows.append(
                    DynamicsBatch(
                        video=np.concatenate(parts_video, axis=0),
                        actions=np.concatenate(parts_actions, axis=0),
                        segment_ids=np.concatenate(parts_segments, axis=0),
                    )
                )
                parts_video, parts_actions, parts_segments = [], [], []
                row_length = 0
                segment_id = 0

                if len(rows) < self.batch_size:
                    continue

                yield DynamicsBatch(
                    video=np.stack([row["video"] for row in rows]),
                    actions=np.stack([row["actions"] for row in rows]),
                    segment_ids=np.stack([row["segment_ids"] for row in rows]),
                )
                rows = []

            segment_id += 1


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

    def random_map(
        self,
        element: VideoDataset,
        rng: np.random.Generator,
    ) -> VideoDataset:
        video = element["video"]
        start_idx = int(rng.integers(0, len(video) - self.frame_length + 1))
        return VideoDataset(video=video[start_idx : start_idx + self.frame_length].copy())


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

    def __getitem__(self, idx: int) -> EncodedVideoDataset:
        try:
            with np.load(io.BytesIO(self._source[idx])) as data:
                return EncodedVideoDataset(
                    video=data["video"].tobytes(),
                    length=int(data["length"]),
                    fps=float(data["fps"]) if "fps" in data else 0.0,
                )
        except Exception as exc:
            location = _describe_record_location(self._source, self._paths, idx)
            raise ValueError(f"Failed to decode video record idx={idx} ({location}) from {self._data_dir}") from exc


class WeightedVideoBytesDataSource(VideoBytesDataSource):
    """Sample records in proportion to duration."""

    def __init__(self, data_dir: str, weight_seconds: float = 25.6):
        super().__init__(data_dir)
        base = epath.Path(data_dir)
        lengths = json.loads((base / "lengths.json").read_text())
        rates = json.loads((base / "fps.json").read_text())
        if len(lengths) != len(self._source) or len(rates) != len(self._source):
            raise ValueError(
                f"sidecar length mismatch in {data_dir}: "
                f"{len(lengths)} lengths, {len(rates)} fps, "
                f"{len(self._source)} records"
            )

        self._index = []
        for record_idx, (num_frames, fps) in enumerate(zip(lengths, rates)):
            duration = num_frames / fps if fps > 0 else num_frames / 30.0
            repeats = max(int(np.ceil(duration / weight_seconds)), 1)
            self._index.extend([record_idx] * repeats)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int) -> EncodedVideoDataset:
        return super().__getitem__(self._index[idx])


class DecodeRandomVideoClip(grain.RandomMapTransform):
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

    def random_map(
        self,
        element: EncodedVideoDataset,
        rng: np.random.Generator,
    ) -> VideoDataset:
        length = element["length"]
        fps = element["fps"]
        stride = max(int(round(fps / self.target_hz)), 1) if self.target_hz and fps > 0 else self.stride
        span = (self.frame_length - 1) * stride + 1
        start = 0 if length <= span else int(rng.integers(0, length - span + 1))
        frames = decode_video_window(element["video"], start, span, self.decode_hw)
        frames = frames[::stride][: self.frame_length]
        if len(frames) < self.frame_length:
            pad = np.repeat(frames[-1:], self.frame_length - len(frames), axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        return VideoDataset(video=frames)


class AugmentVideoClip(grain.RandomMapTransform):
    """Apply one spatial and color transform to a complete clip."""

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
            video = np.stack(
                [
                    cv2.resize(
                        frame,
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for frame in video
                ]
            )

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
            hsv = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2HSV) for frame in np.clip(out, 0, 255).astype(np.uint8)])
            hsv[..., 0] = (hsv[..., 0].astype(np.float32) + shift) % 180
            out = np.stack([cv2.cvtColor(frame, cv2.COLOR_HSV2RGB) for frame in hsv]).astype(np.float32)

        return VideoDataset(video=np.clip(out, 0, 255).astype(np.uint8))
