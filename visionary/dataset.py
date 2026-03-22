from typing import TypedDict
import grain.python as grain
import numpy as np
import glob
from einops import rearrange


class VideoDataset(TypedDict):
    video: np.ndarray


class PreprocessedVideoDataset(TypedDict):
    video: np.ndarray
    mask_prob: float


class EpisodeDataSource(grain.RandomAccessDataSource):
    def __init__(self, episode_dir):
        self.files = sorted(glob.glob(f"{episode_dir}/**/episode_*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> VideoDataset:
        data = np.load(self.files[idx])
        return VideoDataset(video=data["observation"])


class PreprocessAndPatchify(grain.MapTransform):
    def __init__(self, patch_size: int, pad_width: tuple[int, int]):
        self.height_pad, self.width_pad = pad_width
        self.patch_size = patch_size

    def map(self, element: VideoDataset) -> PreprocessedVideoDataset:
        video = element["video"]

        pad_width = (
            (0, 0),
            (self.height_pad, self.height_pad),
            (self.width_pad, self.width_pad),
            (0, 0),
        )
        padded_video = np.pad(video, pad_width, mode="constant", constant_values=0)

        patched_video = rearrange(
            padded_video,
            "t (h p1) (w p2) c -> t (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

        p = np.random.uniform(0.0, 0.9)

        return {"video": patched_video, "mask_prob": p}
