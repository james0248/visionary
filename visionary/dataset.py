import io
from typing import TypedDict

import grain.python as grain
import jax
import numpy as np
from einops import rearrange
from etils import epath


class VideoDataset(TypedDict):
    video: np.ndarray


class VideoDataSource(grain.RandomAccessDataSource):
    def __init__(self, data_dir: str):
        shard_dir = epath.Path(data_dir)
        paths = sorted(
            [p for p in shard_dir.iterdir() if p.suffix == ".arecord"],
            key=lambda p: p.as_posix(),
        )
        if not paths:
            raise FileNotFoundError(f"No .arecord files found in {data_dir}")
        self._source = grain.ArrayRecordDataSource([p.as_posix() for p in paths])

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
