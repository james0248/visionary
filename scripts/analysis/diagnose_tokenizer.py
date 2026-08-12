import argparse
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from visionary.common.checkpoint import (
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.dataset import RandomVideoCrop, VideoDataSource
from visionary.models.dreamer4.tokenizer import Tokenizer
from visionary.models.dreamer4.tokenizer_preprocessor import TokenizerPreprocessor


def compute_mse(prediction: jax.Array, target: jax.Array) -> float:
    return float(jnp.mean(jnp.square(prediction - target)))


def build_grid(panels: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    episode_indices = np.arange(panels[0].shape[0])
    frame_indices = rng.integers(0, panels[0].shape[1], size=panels[0].shape[0])

    def sample_frames(images: np.ndarray) -> np.ndarray:
        return np.clip(
            np.rint(images[episode_indices, frame_indices] * 255.0),
            0,
            255,
        ).astype(np.uint8)

    panels = [sample_frames(p) for p in panels]
    col_sep = np.full((panels[0].shape[1], 2, 3), 255, dtype=np.uint8)
    rows = []
    for cells in zip(*panels, strict=True):
        row = [cells[0]]
        for cell in cells[1:]:
            row.extend([col_sep, cell])
        rows.append(np.concatenate(row, axis=1))
    row_sep = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate(
        [row if i == 0 else np.concatenate([row_sep, row], axis=0) for i, row in enumerate(rows)],
        axis=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("recon", "ablate"), default="ablate")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output", default="tokenizer_diagnostic.png")
    parser.add_argument("--step", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=8)
    parser.add_argument("--mask_prob", type=float, default=0.1)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_cfg = OmegaConf.load(args.config)
    rng = np.random.default_rng(args.seed)
    source = VideoDataSource(args.dataset_dir)
    sample_indices = rng.choice(
        len(source),
        size=min(args.num_episodes, len(source)),
        replace=False,
    )

    samples = []
    for sample_idx in np.atleast_1d(sample_indices):
        sample = source[int(sample_idx)]
        sample = RandomVideoCrop(run_cfg.dataset.frame_length).random_map(sample, rng)
        samples.append(sample)
    batch = {"video": np.stack([sample["video"] for sample in samples])}

    model_cfg, variables = restore_model_export_single_device(
        args.checkpoint_dir,
        step=args.step,
    )
    preprocessor_cfg = restore_preprocessor_export(args.checkpoint_dir, step=args.step)
    tokenizer = instantiate(model_cfg)
    preprocessor = TokenizerPreprocessor.from_config(preprocessor_cfg)
    patch_batch = {"video": preprocessor.preprocess_video(batch["video"])}
    patch_video = jnp.asarray(patch_batch["video"], dtype=jnp.float32) / 255.0
    original = preprocessor.patches_to_images(patch_video).astype(jnp.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "recon":
        reconstructed_patches, mask, _ = tokenizer.apply(
            variables,
            patch_batch,
            mask_prob=float(args.mask_prob),
            method=Tokenizer.reconstruct,
            rngs={"sample": jax.random.key(args.seed + 1)},
        )
        reconstructed = preprocessor.patches_to_images(reconstructed_patches.astype(jnp.float32))
        mask_images = preprocessor.mask_to_images(mask).astype(original.dtype)
        masked = original * (1.0 - mask_images)
        panels = jax.device_get([jnp.clip(p, 0.0, 1.0) for p in (original, masked, reconstructed)])
        imageio.imwrite(output, build_grid([np.asarray(p) for p in panels], rng))
        print(output)
        return

    def decode_images(latent: jax.Array) -> jax.Array:
        return preprocessor.patches_to_images(
            tokenizer.apply(
                variables,
                latent,
                method=Tokenizer.decode,
            ).astype(jnp.float32)
        )

    latent = tokenizer.apply(variables, patch_batch, method=Tokenizer.encode).astype(jnp.float32)
    reconstructed = decode_images(latent)
    zero_latent = decode_images(jnp.zeros_like(latent))

    shuffle_perm = jnp.asarray(rng.permutation(latent.shape[0]))
    shuffled_latent = decode_images(latent[shuffle_perm])

    mean_image = jnp.mean(original, axis=(0, 1), keepdims=True)
    mean_baseline = jnp.broadcast_to(mean_image, original.shape)

    flattened_latent = latent.reshape(-1, latent.shape[-2], latent.shape[-1])
    latent_stats = {
        "latent_mean": float(jnp.mean(latent)),
        "latent_std": float(jnp.std(latent)),
        "latent_example_std": float(jnp.mean(jnp.std(flattened_latent, axis=0))),
        "latent_min": float(jnp.min(latent)),
        "latent_max": float(jnp.max(latent)),
        "latent_saturation_ratio": float(jnp.mean(jnp.abs(latent) > 0.95)),
    }
    mse_stats = {
        "mse_reconstructed": compute_mse(reconstructed, original),
        "mse_zero_latent": compute_mse(zero_latent, original),
        "mse_shuffled_latent": compute_mse(shuffled_latent, original),
        "mse_mean_baseline": compute_mse(mean_baseline, original),
        "recon_vs_zero_l1": float(jnp.mean(jnp.abs(reconstructed - zero_latent))),
        "recon_vs_shuffled_l1": float(jnp.mean(jnp.abs(reconstructed - shuffled_latent))),
    }

    panels = jax.device_get(
        [jnp.clip(p, 0.0, 1.0) for p in (original, reconstructed, zero_latent, shuffled_latent, mean_baseline)]
    )
    imageio.imwrite(
        output,
        build_grid([np.asarray(p) for p in panels], np.random.default_rng(args.seed + 1)),
    )

    print("Saved diagnostic grid:", output)
    print()
    print("MSE stats")
    for key, value in mse_stats.items():
        print(f"  {key}: {value:.6f}")
    print()
    print("Latent stats")
    for key, value in latent_stats.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
