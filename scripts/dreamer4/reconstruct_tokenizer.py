import argparse
import io
from pathlib import Path

import grain.python as grain
import imageio
import jax
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from visionary.common.checkpoint import (
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.dataset import RandomVideoCrop
from visionary.tokenizer import Tokenizer
from visionary.tokenizer_preprocessor import TokenizerPreprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output", default="reconstruction.png")
    parser.add_argument("--step", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=8)
    parser.add_argument("--mask_prob", type=float, default=0.1)
    args = parser.parse_args()

    run_cfg = OmegaConf.load(Path(__file__).resolve().parent / "config" / "breakout.yaml")
    rng = np.random.default_rng(args.seed)
    shard_paths = sorted(Path(args.dataset_dir).glob("*.arecord"))
    shard_indices = rng.choice(
        len(shard_paths),
        size=min(args.num_episodes, len(shard_paths)),
        replace=False,
    )
    samples = []
    for shard_idx in np.atleast_1d(shard_indices):
        source = grain.ArrayRecordDataSource([shard_paths[int(shard_idx)].as_posix()])
        with np.load(io.BytesIO(source[int(rng.integers(len(source)))])) as data:
            sample = {"video": np.asarray(data["frames"])}
        sample = RandomVideoCrop(run_cfg.dataset.frame_length).random_map(sample, rng)
        samples.append(sample)
    batch = {
        "video": np.stack([sample["video"] for sample in samples]),
    }

    model_cfg, variables = restore_model_export_single_device(
        args.checkpoint_dir,
        step=args.step,
    )
    preprocessor_cfg = restore_preprocessor_export(args.checkpoint_dir, step=args.step)
    tokenizer = instantiate(model_cfg)
    preprocessor = TokenizerPreprocessor.from_config(preprocessor_cfg)
    patch_batch = {"video": preprocessor.preprocess_video(batch["video"])}
    patch_video = jax.numpy.asarray(patch_batch["video"], dtype=jax.numpy.float32) / 255.0
    reconstructed_patches, mask = tokenizer.apply(
        variables,
        patch_batch,
        mask_prob=float(args.mask_prob),
        method=Tokenizer.reconstruct,
        rngs={"sample": jax.random.key(args.seed + 1)},
    )
    original = preprocessor.patches_to_images(patch_video)
    reconstructed = preprocessor.patches_to_images(reconstructed_patches.astype(jax.numpy.float32))
    mask_images = preprocessor.mask_to_images(mask).astype(original.dtype)
    masked = original * (1.0 - mask_images)
    original, masked, reconstructed = jax.device_get(
        (
            jax.numpy.clip(original, 0.0, 1.0),
            jax.numpy.clip(masked, 0.0, 1.0),
            jax.numpy.clip(reconstructed, 0.0, 1.0),
        )
    )

    episode_indices = np.arange(original.shape[0])
    frame_indices = rng.integers(0, original.shape[1], size=original.shape[0])
    original = np.clip(np.rint(original[episode_indices, frame_indices] * 255.0), 0, 255).astype(
        np.uint8
    )
    masked = np.clip(np.rint(masked[episode_indices, frame_indices] * 255.0), 0, 255).astype(
        np.uint8
    )
    reconstructed = np.clip(
        np.rint(reconstructed[episode_indices, frame_indices] * 255.0), 0, 255
    ).astype(np.uint8)

    col_sep = np.full((original.shape[1], 2, 3), 255, dtype=np.uint8)
    rows = [
        np.concatenate([o, col_sep, m, col_sep, r], axis=1)
        for o, m, r in zip(original, masked, reconstructed, strict=True)
    ]
    row_sep = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
    grid = np.concatenate(
        [row if i == 0 else np.concatenate([row_sep, row], axis=0) for i, row in enumerate(rows)],
        axis=0,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output, grid)
    print(output)


if __name__ == "__main__":
    main()
