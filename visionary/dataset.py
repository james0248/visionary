import glob
from typing import TypedDict

import grain.python as grain
import numpy as np
from einops import rearrange


class VideoDataset(TypedDict):
    video: np.ndarray


class PreprocessedVideoDataset(TypedDict):
    video: np.ndarray
    mask_prob: float
    independent: bool


class EpisodeDataSource(grain.RandomAccessDataSource):
    def __init__(self, episode_dir):
        self.files = sorted(glob.glob(f"{episode_dir}/**/episode_*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> VideoDataset:
        data = np.load(self.files[idx])
        return VideoDataset(video=data["observation"])


class RandomVideoCrop(grain.RandomMapTransform):
    def __init__(self, frame_length: int):
        self.frame_length = frame_length

    def random_map(
        self, element: VideoDataset, rng: np.random.Generator
    ) -> VideoDataset:
        video = element["video"]
        start_idx = rng.integers(0, len(video) - self.frame_length + 1)
        sliced_video = video[start_idx : start_idx + self.frame_length]

        return VideoDataset(video=sliced_video)


class PreprocessAndPatchify(grain.RandomMapTransform):
    def __init__(self, frame_length: int, patch_size: int, pad_width: tuple[int, int]):
        self.frame_length = frame_length
        self.patch_size = patch_size

        height_pad, width_pad = pad_width
        self.pad_width = (
            (0, 0),
            (height_pad, height_pad),
            (width_pad, width_pad),
            (0, 0),
        )

    def random_map(
        self, element: VideoDataset, rng: np.random.Generator
    ) -> PreprocessedVideoDataset:
        video = element["video"]

        padded_video = np.pad(video, self.pad_width, mode="constant", constant_values=0)
        patched_video = rearrange(
            padded_video,
            "t (h p1) (w p2) c -> t (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        p = rng.uniform(0.0, 0.9, size=(video.shape[0],))
        independent = rng.random() < 0.3

        return {"video": patched_video, "mask_prob": p, "independent": independent}
