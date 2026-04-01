import argparse
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf

from visionary.common.checkpoint import CheckpointManager
from visionary.common.train_state import TokenizerTrainState
from visionary.dataset import PreprocessAndPatchify, RandomVideoCrop, VideoDataSource
from visionary.tokenizer import Tokenizer


def unpatchify(images: jax.Array, patch_size: int, x_len: int, y_len: int) -> jax.Array:
    return rearrange(
        images,
        "b t (h w) (p1 p2 c) -> b t (h p1) (w p2) c",
        h=y_len,
        w=x_len,
        p1=patch_size,
        p2=patch_size,
    )


def trim_padding(images: np.ndarray, pad_width: tuple[int, int]) -> np.ndarray:
    h_pad, w_pad = (int(v) for v in pad_width)
    return images[
        :,
        :,
        slice(h_pad, -h_pad if h_pad else None),
        slice(w_pad, -w_pad if w_pad else None),
        :,
    ]


def patches_to_images(
    patches: jax.Array,
    *,
    patch_size: int,
    x_len: int,
    y_len: int,
    pad_width: tuple[int, int],
) -> np.ndarray:
    images = unpatchify(patches, patch_size, x_len, y_len).clip(0.0, 1.0)
    return trim_padding(np.asarray(jax.device_get(images)), pad_width)


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

    cfg = OmegaConf.load(args.config)
    rng = np.random.default_rng(args.seed)
    source = VideoDataSource(args.dataset_dir)
    sample_indices = rng.choice(
        len(source),
        size=min(args.num_episodes, len(source)),
        replace=False,
    )
    transform = PreprocessAndPatchify(
        cfg.dataset.patch_size,
        tuple(cfg.dataset.pad_width),
        tuple(cfg.dataset.resize_shape),
    )

    samples = []
    for sample_idx in np.atleast_1d(sample_indices):
        sample = source[int(sample_idx)]
        sample = RandomVideoCrop(cfg.dataset.frame_length).random_map(sample, rng)
        sample = transform.random_map(sample, rng)
        samples.append(sample)
    batch = {"video": np.stack([sample["video"] for sample in samples])}

    model = instantiate(cfg.tokenizer)
    init_key, sample_key = jax.random.split(jax.random.key(args.seed))
    state = TokenizerTrainState.create(
        apply_fn=model.apply,
        params=model.init({"params": init_key, "sample": sample_key}, batch),
        tx=optax.adam(0.0),
        mse_sq_ema=jnp.ones((), dtype=jnp.float32),
        lpips_sq_ema=jnp.ones((), dtype=jnp.float32),
    )
    with CheckpointManager(args.checkpoint_dir, ocp.CheckpointManagerOptions()) as manager:
        state = manager.restore(target=state, step=args.step)

    original = jnp.asarray(batch["video"], dtype=jnp.float32) / 255.0
    patch_dim = int(batch["video"].shape[-1])

    latent = state.apply_fn(state.params, batch, method=Tokenizer.encode).astype(
        jnp.float32
    )
    reconstructed = state.apply_fn(
        state.params,
        latent,
        patch_dim,
        method=Tokenizer.decode,
    ).astype(jnp.float32)

    zero_latent = state.apply_fn(
        state.params,
        jnp.zeros_like(latent),
        patch_dim,
        method=Tokenizer.decode,
    ).astype(jnp.float32)

    shuffle_perm = jnp.asarray(rng.permutation(latent.shape[0]))
    shuffled_latent = state.apply_fn(
        state.params,
        latent[shuffle_perm],
        patch_dim,
        method=Tokenizer.decode,
    ).astype(jnp.float32)

    mean_patch = jnp.mean(original, axis=(0, 1), keepdims=True)
    mean_baseline = jnp.broadcast_to(mean_patch, original.shape)

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
        "recon_vs_shuffled_l1": float(
            jnp.mean(jnp.abs(reconstructed - shuffled_latent))
        ),
    }

    original_images = patches_to_images(
        original,
        patch_size=cfg.dataset.patch_size,
        x_len=cfg.tokenizer.x_len,
        y_len=cfg.tokenizer.y_len,
        pad_width=tuple(cfg.dataset.pad_width),
    )
    reconstructed_images = patches_to_images(
        reconstructed,
        patch_size=cfg.dataset.patch_size,
        x_len=cfg.tokenizer.x_len,
        y_len=cfg.tokenizer.y_len,
        pad_width=tuple(cfg.dataset.pad_width),
    )
    zero_latent_images = patches_to_images(
        zero_latent,
        patch_size=cfg.dataset.patch_size,
        x_len=cfg.tokenizer.x_len,
        y_len=cfg.tokenizer.y_len,
        pad_width=tuple(cfg.dataset.pad_width),
    )
    shuffled_latent_images = patches_to_images(
        shuffled_latent,
        patch_size=cfg.dataset.patch_size,
        x_len=cfg.tokenizer.x_len,
        y_len=cfg.tokenizer.y_len,
        pad_width=tuple(cfg.dataset.pad_width),
    )
    mean_baseline_images = patches_to_images(
        mean_baseline,
        patch_size=cfg.dataset.patch_size,
        x_len=cfg.tokenizer.x_len,
        y_len=cfg.tokenizer.y_len,
        pad_width=tuple(cfg.dataset.pad_width),
    )

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
