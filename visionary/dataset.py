import io
from typing import TypedDict

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


def _array_record_source(data_dir: str) -> grain.ArrayRecordDataSource:
    shard_dir = epath.Path(data_dir)
    paths = sorted(
        [p for p in shard_dir.iterdir() if p.suffix == ".arecord"],
        key=lambda p: p.as_posix(),
    )
    if not paths:
        raise FileNotFoundError(f"No .arecord files found in {data_dir}")
    return grain.ArrayRecordDataSource([p.as_posix() for p in paths])


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
