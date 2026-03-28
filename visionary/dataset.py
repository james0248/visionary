import hashlib
from collections.abc import Sequence
from typing import TypedDict

import grain.python as grain
import numpy as np
from einops import rearrange
from etils import epath


class VideoDataset(TypedDict):
    video: np.ndarray


class PreprocessedVideoDataset(TypedDict):
    video: np.ndarray
    mask_prob: np.ndarray
    independent: bool


class EpisodeDataSource(grain.RandomAccessDataSource):
    def __init__(self, files: Sequence[epath.Path]):
        self.files = list(files)

    @classmethod
    def from_split(
        cls, data_dir: str, eval_ratio: float, seed: int
    ) -> tuple["EpisodeDataSource", "EpisodeDataSource"]:
        train_files, eval_files = split_episode_files(data_dir, eval_ratio, seed)
        return cls(train_files), cls(eval_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> VideoDataset:
        with self.files[idx].open("rb") as file_obj:
            with np.load(file_obj) as data:
                if "frames" not in data:
                    raise KeyError(
                        f"Expected 'frames' in {self.files[idx]}"
                    )
                video = np.asarray(data["frames"])
        return VideoDataset(video=video)


def list_episode_files(data_dir: str) -> list[epath.Path]:
    root = epath.Path(data_dir)
    episode_files: list[epath.Path] = []
    for dirpath, _, filenames in root.walk():
        dirpath = _normalize_walk_dirpath(root, dirpath)
        for filename in filenames:
            if filename.endswith(".npz"):
                episode_files.append(dirpath / filename)
    return sorted(episode_files, key=lambda p: p.as_posix())


def _normalize_walk_dirpath(root: epath.Path, dirpath: epath.Path) -> epath.Path:
    root_parts = root.parts
    dirpath_str = dirpath.as_posix()
    if root_parts[:2] == ("/", "gs") and not dirpath_str.startswith(("gs://", "/gs/")):
        return epath.Path(f"/gs/{dirpath_str.lstrip('/')}")
    return dirpath

def split_episode_files(
    data_dir: str,
    eval_ratio: float,
    split_seed: int,
) -> tuple[list[epath.Path], list[epath.Path]]:
    train_files: list[epath.Path] = []
    eval_files: list[epath.Path] = []

    for path in list_episode_files(data_dir):
        digest = hashlib.sha256(
            f"{split_seed}:{path.as_posix()}".encode("utf-8")
        ).digest()
        bucket = int.from_bytes(digest[:8], byteorder="big") / 2**64
        if bucket < eval_ratio:
            eval_files.append(path)
        else:
            train_files.append(path)

    return train_files, eval_files


class RandomVideoCrop(grain.RandomMapTransform):
    def __init__(self, frame_length: int):
        self.frame_length = frame_length

    def random_map(
        self, element: VideoDataset, rng: np.random.Generator
    ) -> VideoDataset:
        video = element["video"]
        start_idx = int(rng.integers(0, len(video) - self.frame_length + 1))
        return VideoDataset(video=video[start_idx : start_idx + self.frame_length].copy())


class PreprocessAndPatchify(grain.RandomMapTransform):
    def __init__(self, patch_size: int, pad_width: tuple[int, int]):
        self.patch_size = patch_size
        self.pad_width = pad_width

    def random_map(
        self, element: VideoDataset, rng: np.random.Generator
    ) -> PreprocessedVideoDataset:
        video = element["video"]
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
        p = rng.uniform(0.0, 0.9, size=(video.shape[0],)).astype(np.float32)
        independent = rng.random() < 0.3

        return {"video": patched_video, "mask_prob": p, "independent": independent}
