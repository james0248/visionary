from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange

from visionary.dataset import VideoDataset


@dataclass(frozen=True, slots=True)
class TokenizerPreprocessor:
    resize_shape: tuple[int, int]
    pad_width: tuple[int, int]
    patch_size: int
    num_channels: int = 3

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "TokenizerPreprocessor":
        return cls(
            resize_shape=tuple(cfg["resize_shape"]),
            pad_width=tuple(cfg["pad_width"]),
            patch_size=int(cfg["patch_size"]),
        )

    @property
    def image_height(self) -> int:
        return self.resize_shape[0]

    @property
    def image_width(self) -> int:
        return self.resize_shape[1]

    @property
    def height_pad(self) -> int:
        return self.pad_width[0]

    @property
    def width_pad(self) -> int:
        return self.pad_width[1]

    @property
    def y_len(self) -> int:
        return (self.image_height + 2 * self.height_pad) // self.patch_size

    @property
    def x_len(self) -> int:
        return (self.image_width + 2 * self.width_pad) // self.patch_size

    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.patch_size * self.num_channels

    def export_config(self) -> dict[str, Any]:
        return {
            "resize_shape": list(self.resize_shape),
            "pad_width": list(self.pad_width),
            "patch_size": self.patch_size,
        }

    def preprocess_video(self, video: np.ndarray) -> np.ndarray:
        video = np.asarray(video)
        squeeze_batch = video.ndim == 4
        if squeeze_batch:
            video = video[None]

        batch_size, seq_len, _, _, num_channels = video.shape
        frames = video.reshape(batch_size * seq_len, *video.shape[2:])
        resized_frames = np.empty(
            (frames.shape[0], self.image_height, self.image_width, num_channels),
            dtype=np.uint8,
        )
        for index, frame in enumerate(frames):
            resized_frames[index] = cv2.resize(
                frame,
                (self.image_width, self.image_height),
                interpolation=cv2.INTER_LINEAR,
            )

        images = resized_frames.reshape(
            batch_size,
            seq_len,
            self.image_height,
            self.image_width,
            num_channels,
        )
        patches = rearrange(
            np.pad(
                images,
                (
                    (0, 0),
                    (0, 0),
                    (self.height_pad, self.height_pad),
                    (self.width_pad, self.width_pad),
                    (0, 0),
                ),
                mode="constant",
                constant_values=0,
            ),
            "b t (h p1) (w p2) c -> b t (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return patches[0] if squeeze_batch else patches

    def patches_to_images(self, patches: jax.Array | np.ndarray) -> jax.Array:
        patches = jnp.asarray(patches)
        squeeze_batch = patches.ndim == 3
        if squeeze_batch:
            patches = patches[None]
        images = rearrange(
            patches,
            "b t (h w) (p1 p2 c) -> b t (h p1) (w p2) c",
            h=self.y_len,
            w=self.x_len,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.num_channels,
        )
        images = images[
            :,
            :,
            self.height_pad : self.height_pad + self.image_height,
            self.width_pad : self.width_pad + self.image_width,
            :,
        ]
        return images[0] if squeeze_batch else images

    def mask_to_images(self, mask: jax.Array | np.ndarray) -> jax.Array:
        mask = jnp.asarray(mask)
        squeeze_batch = mask.ndim == 3
        if squeeze_batch:
            mask = mask[None]
        patch_mask = jnp.broadcast_to(mask[..., None], (*mask.shape, self.patch_dim)).astype(
            jnp.float32
        )
        images = self.patches_to_images(patch_mask)
        return images[0] if squeeze_batch else images

    def as_grain_transform(self) -> "TokenizerPreprocessTransform":
        return TokenizerPreprocessTransform(self)

    def as_grain_batch_transform(self) -> "BatchedTokenizerPreprocessTransform":
        return BatchedTokenizerPreprocessTransform(self)


class TokenizerPreprocessTransform(grain.RandomMapTransform):
    def __init__(self, preprocessor: TokenizerPreprocessor):
        self.preprocessor = preprocessor

    def __repr__(self) -> str:
        return f"{type(self).__name__}(preprocessor={self.preprocessor!r})"

    def random_map(self, element: VideoDataset, rng: np.random.Generator) -> VideoDataset:
        return {"video": self.preprocessor.preprocess_video(element["video"])}


class BatchedTokenizerPreprocessTransform(grain.MapTransform):
    def __init__(self, preprocessor: TokenizerPreprocessor):
        self.preprocessor = preprocessor

    def __repr__(self) -> str:
        return f"{type(self).__name__}(preprocessor={self.preprocessor!r})"

    def map(self, element: VideoDataset) -> VideoDataset:
        return {"video": self.preprocessor.preprocess_video(element["video"])}
