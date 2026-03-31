import argparse
import io
from pathlib import Path

import grain.python as grain
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
from visionary.dataset import PreprocessAndPatchify, RandomVideoCrop


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

    cfg = OmegaConf.load(Path(__file__).resolve().parent / "config" / "breakout.yaml")
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
        sample = RandomVideoCrop(cfg.dataset.frame_length).random_map(sample, rng)
        sample = PreprocessAndPatchify(
            cfg.dataset.patch_size, tuple(cfg.dataset.pad_width)
        ).random_map(sample, rng)
        samples.append(sample)
    batch = {
        "video": np.stack([sample["video"] for sample in samples]),
    }

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

    reconstructed, mask = state.apply_fn(
        state.params,
        batch,
        mask_prob=float(args.mask_prob),
        rngs={"sample": jax.random.key(args.seed + 1)},
    )

    original = jnp.asarray(batch["video"], dtype=jnp.float32) / 255.0
    mask_f32 = jnp.expand_dims(mask, axis=-1).astype(original.dtype)
    masked = original * (1.0 - mask_f32)
    reconstructed = reconstructed.astype(jnp.float32)

    original = trim_padding(
        np.asarray(
            jax.device_get(
                unpatchify(
                    original, cfg.dataset.patch_size, cfg.tokenizer.x_len, cfg.tokenizer.y_len
                ).clip(0.0, 1.0)
            )
        ),
        tuple(cfg.dataset.pad_width),
    )
    masked = trim_padding(
        np.asarray(
            jax.device_get(
                unpatchify(
                    masked, cfg.dataset.patch_size, cfg.tokenizer.x_len, cfg.tokenizer.y_len
                ).clip(0.0, 1.0)
            )
        ),
        tuple(cfg.dataset.pad_width),
    )
    reconstructed = trim_padding(
        np.asarray(
            jax.device_get(
                unpatchify(
                    reconstructed,
                    cfg.dataset.patch_size,
                    cfg.tokenizer.x_len,
                    cfg.tokenizer.y_len,
                ).clip(0.0, 1.0)
            )
        ),
        tuple(cfg.dataset.pad_width),
    )

    episode_indices = np.arange(original.shape[0])
    frame_indices = rng.integers(0, original.shape[1], size=original.shape[0])
    original = np.clip(
        np.rint(original[episode_indices, frame_indices] * 255.0), 0, 255
    ).astype(np.uint8)
    masked = np.clip(
        np.rint(masked[episode_indices, frame_indices] * 255.0), 0, 255
    ).astype(np.uint8)
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
