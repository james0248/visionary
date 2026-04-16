import argparse
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from visionary.common.checkpoint import restore_model_export
from visionary.dataset import RandomVideoCrop, VideoDataSource
from visionary.tokenizer import Tokenizer


def compute_mse(prediction: jax.Array, target: jax.Array) -> float:
    return float(jnp.mean(jnp.square(prediction - target)))


def build_grid(
    original: np.ndarray,
    reconstructed: np.ndarray,
    zero_latent: np.ndarray,
    shuffled_latent: np.ndarray,
    mean_baseline: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    episode_indices = np.arange(original.shape[0])
    frame_indices = rng.integers(0, original.shape[1], size=original.shape[0])

    def sample_frames(images: np.ndarray) -> np.ndarray:
        return np.clip(
            np.rint(images[episode_indices, frame_indices] * 255.0),
            0,
            255,
        ).astype(np.uint8)

    original = sample_frames(original)
    reconstructed = sample_frames(reconstructed)
    zero_latent = sample_frames(zero_latent)
    shuffled_latent = sample_frames(shuffled_latent)
    mean_baseline = sample_frames(mean_baseline)

    col_sep = np.full((original.shape[1], 2, 3), 255, dtype=np.uint8)
    rows = [
        np.concatenate(
            [o, col_sep, r, col_sep, z, col_sep, s, col_sep, m],
            axis=1,
        )
        for o, r, z, s, m in zip(
            original,
            reconstructed,
            zero_latent,
            shuffled_latent,
            mean_baseline,
            strict=True,
        )
    ]
    row_sep = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate(
        [row if i == 0 else np.concatenate([row_sep, row], axis=0) for i, row in enumerate(rows)],
        axis=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output", default="tokenizer_diagnostic.png")
    parser.add_argument("--step", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=8)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config" / "breakout.yaml"),
    )
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

    model_cfg, variables = restore_model_export(args.checkpoint_dir, step=args.step)
    tokenizer = instantiate(model_cfg)
    original = tokenizer.apply(
        variables,
        jnp.asarray(batch["video"]),
        method=Tokenizer.preprocess_video,
    )[0].astype(jnp.float32)

    latent = tokenizer.apply(variables, batch, method=Tokenizer.encode).astype(jnp.float32)
    reconstructed = tokenizer.apply(
        variables,
        latent,
        method=Tokenizer.decode,
    ).astype(jnp.float32)

    zero_latent = tokenizer.apply(
        variables,
        jnp.zeros_like(latent),
        method=Tokenizer.decode,
    ).astype(jnp.float32)

    shuffle_perm = jnp.asarray(rng.permutation(latent.shape[0]))
    shuffled_latent = tokenizer.apply(
        variables,
        latent[shuffle_perm],
        method=Tokenizer.decode,
    ).astype(jnp.float32)

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

    original_images = np.asarray(jax.device_get(jnp.clip(original, 0.0, 1.0)))
    reconstructed_images = np.asarray(jax.device_get(jnp.clip(reconstructed, 0.0, 1.0)))
    zero_latent_images = np.asarray(jax.device_get(jnp.clip(zero_latent, 0.0, 1.0)))
    shuffled_latent_images = np.asarray(jax.device_get(jnp.clip(shuffled_latent, 0.0, 1.0)))
    mean_baseline_images = np.asarray(jax.device_get(jnp.clip(mean_baseline, 0.0, 1.0)))

    grid = build_grid(
        original_images,
        reconstructed_images,
        zero_latent_images,
        shuffled_latent_images,
        mean_baseline_images,
        seed=args.seed + 1,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output, grid)

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
