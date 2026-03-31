import itertools
import logging
import time
from functools import lru_cache, partial

import grain.python as grain
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from einops import rearrange
from hydra.utils import instantiate
from jaxlpips import LPIPS
from omegaconf import DictConfig

from visionary.common.checkpoint import CheckpointManager
from visionary.common.train_state import TokenizerTrainState
from visionary.common.wandb import WandbLogger
from visionary.dataset import (
    PreprocessAndPatchify,
    PreprocessedVideoDataset,
    RandomVideoCrop,
    VideoDataSource,
)

logger = logging.getLogger(__name__)


LPIPS_PRETRAINED_NETWORK = "alexnet"
LOSS_RMS_DECAY = 0.99
LOSS_RMS_EPS = 1e-8


@lru_cache(maxsize=1)
def get_lpips_loss_fn():
    return LPIPS(pretrained_network=LPIPS_PRETRAINED_NETWORK)


def unpatchify(
    images: jax.Array, patch_size: int, width_tokens: int, height_tokens: int
) -> jax.Array:
    return rearrange(
        images,
        "b t (h w) (p1 p2 c) -> b t (h p1) (w p2) c",
        p1=patch_size,
        p2=patch_size,
        h=height_tokens,
        w=width_tokens,
    )


def compute_lpips_loss(
    original: jax.Array,
    reconstructed: jax.Array,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
) -> jax.Array:
    original_images = unpatchify(original, patch_size, width_tokens, height_tokens)
    reconstructed_images = unpatchify(
        reconstructed, patch_size, width_tokens, height_tokens
    )
    original_images, reconstructed_images = (
        original_images * 2.0 - 1.0,
        reconstructed_images * 2.0 - 1.0,
    )
    # jaxlpips expects NHWC images, so fold time into the batch axis.
    original_images = rearrange(original_images, "b t h w c -> (b t) h w c")
    reconstructed_images = rearrange(reconstructed_images, "b t h w c -> (b t) h w c")
    return jnp.mean(get_lpips_loss_fn()(original_images, reconstructed_images))


def update_loss_ema(
    state: TokenizerTrainState,
    mse_loss: jax.Array,
    lpips_loss: jax.Array,
) -> TokenizerTrainState:
    mse_loss = mse_loss.astype(jnp.float32)
    lpips_loss = lpips_loss.astype(jnp.float32)
    step_size = jnp.asarray(1.0 - LOSS_RMS_DECAY, dtype=mse_loss.dtype)
    return state.replace(
        mse_sq_ema=optax.incremental_update(
            jnp.square(mse_loss), state.mse_sq_ema.astype(mse_loss.dtype), step_size
        ),
        lpips_sq_ema=optax.incremental_update(
            jnp.square(lpips_loss),
            state.lpips_sq_ema.astype(lpips_loss.dtype),
            step_size,
        ),
    )


def compute_losses(
    params,
    state: TokenizerTrainState,
    batch: PreprocessedVideoDataset,
    sample_key: jax.Array,
    lpips_weight: float,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
):
    reconstructed, mask = state.apply_fn(
        params,
        batch,
        rngs={"sample": sample_key},
    )
    reconstructed_f32 = reconstructed.astype(jnp.float32)
    original = batch["video"].astype(jnp.float32) / 255.0
    mask_f32 = jnp.expand_dims(mask, axis=-1).astype(reconstructed_f32.dtype)
    num_masked = jnp.maximum(
        jnp.sum(mask_f32) * reconstructed_f32.shape[-1],
        1.0,
    )
    mse_loss = jnp.sum(jnp.square(reconstructed_f32 - original) * mask_f32) / num_masked
    mse_rms = jnp.sqrt(state.mse_sq_ema.astype(mse_loss.dtype) + LOSS_RMS_EPS)
    normalized_mse_loss = mse_loss / jax.lax.stop_gradient(mse_rms)

    lpips_rms = jnp.sqrt(state.lpips_sq_ema.astype(mse_loss.dtype) + LOSS_RMS_EPS)
    if lpips_weight > 0:
        inpainted_reconstruction = (
            reconstructed_f32 * mask_f32 + original * (1.0 - mask_f32)
        )
        lpips_loss = compute_lpips_loss(
            original,
            inpainted_reconstruction,
            patch_size=patch_size,
            width_tokens=width_tokens,
            height_tokens=height_tokens,
        )
    else:
        lpips_loss = jnp.zeros((), dtype=mse_loss.dtype)
    normalized_lpips_loss = lpips_loss / jax.lax.stop_gradient(lpips_rms)

    raw_loss = mse_loss + lpips_weight * lpips_loss
    loss = normalized_mse_loss + lpips_weight * normalized_lpips_loss
    metrics = {
        "loss": loss,
        "raw_loss": raw_loss,
        "mse_loss": mse_loss,
        "lpips_loss": lpips_loss,
        "normalized_mse_loss": normalized_mse_loss,
        "normalized_lpips_loss": normalized_lpips_loss,
        "mse_rms": mse_rms,
        "lpips_rms": lpips_rms,
        "mask_ratio": jnp.mean(mask_f32),
    }
    return loss, metrics


@partial(
    jax.jit,
    static_argnames=("lpips_weight", "patch_size", "width_tokens", "height_tokens"),
)
def train_step(
    state: TokenizerTrainState,
    batch: PreprocessedVideoDataset,
    sample_key: jax.Array,
    lpips_weight: float,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
):
    def loss_fn(params):
        return compute_losses(
            params,
            state,
            batch,
            sample_key,
            lpips_weight,
            patch_size,
            width_tokens,
            height_tokens,
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = update_loss_ema(state, metrics["mse_loss"], metrics["lpips_loss"])
    return state, metrics


@partial(
    jax.jit,
    static_argnames=("lpips_weight", "patch_size", "width_tokens", "height_tokens"),
)
def eval_step(
    state: TokenizerTrainState,
    batch: PreprocessedVideoDataset,
    sample_key: jax.Array,
    lpips_weight: float,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
):
    _, metrics = compute_losses(
        state.params,
        state,
        batch,
        sample_key,
        lpips_weight,
        patch_size,
        width_tokens,
        height_tokens,
    )
    return metrics


@jax.jit
def eval_visualization_step(
    state: TokenizerTrainState,
    batch: PreprocessedVideoDataset,
    sample_key: jax.Array,
):
    return state.apply_fn(
        state.params,
        batch,
        rngs={"sample": sample_key},
    )


def trim_padding(images: np.ndarray, pad_width: tuple[int, int]) -> np.ndarray:
    height_pad, width_pad = (int(p) for p in pad_width)
    h_slice = slice(height_pad, -height_pad if height_pad else None)
    w_slice = slice(width_pad, -width_pad if width_pad else None)
    return images[:, :, h_slice, w_slice, :]


def build_reconstruction_grid(
    batch: PreprocessedVideoDataset,
    reconstructed: jax.Array,
    mask: jax.Array,
    *,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
    pad_width: tuple[int, int],
    frame_key: jax.Array,
    num_frames: int,
) -> np.ndarray:
    original = jnp.asarray(batch["video"], dtype=jnp.float32) / 255.0
    mask_f32 = jnp.expand_dims(mask, axis=-1).astype(original.dtype)
    masked_input = original * (1.0 - mask_f32)
    original = rearrange(original, "b t n d -> (b t) n d")
    reconstructed = rearrange(reconstructed.astype(jnp.float32), "b t n d -> (b t) n d")
    masked_input = rearrange(masked_input, "b t n d -> (b t) n d")

    total_frames = original.shape[0]
    num_frames = min(int(num_frames), total_frames)
    frame_indices = np.asarray(
        jax.device_get(
            jax.random.choice(
                frame_key, total_frames, shape=(num_frames,), replace=False
            )
        )
    )

    original_images = trim_padding(
        np.asarray(
            jax.device_get(
                unpatchify(
                    rearrange(original[frame_indices], "f n d -> 1 f n d"),
                    patch_size,
                    width_tokens,
                    height_tokens,
                ).clip(0.0, 1.0)
            )
        ),
        pad_width,
    )[0]
    reconstructed_images = trim_padding(
        np.asarray(
            jax.device_get(
                unpatchify(
                    rearrange(reconstructed[frame_indices], "f n d -> 1 f n d"),
                    patch_size,
                    width_tokens,
                    height_tokens,
                ).clip(0.0, 1.0)
            )
        ),
        pad_width,
    )[0]
    masked_images = trim_padding(
        np.asarray(
            jax.device_get(
                unpatchify(
                    rearrange(masked_input[frame_indices], "f n d -> 1 f n d"),
                    patch_size,
                    width_tokens,
                    height_tokens,
                ).clip(0.0, 1.0)
            )
        ),
        pad_width,
    )[0]

    originals = np.clip(np.rint(original_images * 255.0), 0, 255).astype(np.uint8)
    reconstructions = np.clip(np.rint(reconstructed_images * 255.0), 0, 255).astype(
        np.uint8
    )
    masked_inputs = np.clip(np.rint(masked_images * 255.0), 0, 255).astype(np.uint8)

    col_sep = np.full((originals.shape[1], 2, 3), 255, dtype=np.uint8)
    rows = []
    for original_frame, reconstructed_frame, masked_frame in zip(
        originals, reconstructions, masked_inputs, strict=True
    ):
        rows.append(
            np.concatenate(
                [original_frame, col_sep, reconstructed_frame, col_sep, masked_frame],
                axis=1,
            )
        )
    row_sep = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate(
        [
            row if idx == 0 else np.concatenate([row_sep, row], axis=0)
            for idx, row in enumerate(rows)
        ],
        axis=0,
    )


@hydra.main(config_path="config", version_base=None)
def main(cfg: DictConfig):
    logger.info("JAX backend: %s, devices: %s", jax.default_backend(), jax.devices())
    wb = WandbLogger(cfg)
    total_steps = int(cfg.total_steps)

    train_source = VideoDataSource(cfg.dataset.train_dir)
    eval_source = VideoDataSource(cfg.dataset.eval_dir)
    logger.info(
        "Loaded %d training videos and %d eval videos",
        len(train_source),
        len(eval_source),
    )
    effective_read_threads = max(int(cfg.dataset.worker_count), 1) * int(
        cfg.dataset.num_threads
    )
    logger.info(
        "Data loader settings: worker_count=%d num_threads=%d "
        "prefetch_buffer_size=%d effective_read_threads=%d",
        int(cfg.dataset.worker_count),
        int(cfg.dataset.num_threads),
        int(cfg.dataset.prefetch_buffer_size),
        effective_read_threads,
    )
    transforms = [
        RandomVideoCrop(cfg.dataset.frame_length),
        PreprocessAndPatchify(cfg.dataset.patch_size, cfg.dataset.pad_width),
    ]

    def make_loader(source, *, shuffle: bool, drop_remainder: bool):
        sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.ShardByJaxProcess(),
            shuffle=shuffle,
            seed=cfg.seed,
        )
        read_options = grain.ReadOptions(
            num_threads=int(cfg.dataset.num_threads),
            prefetch_buffer_size=int(cfg.dataset.prefetch_buffer_size),
        )
        return grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=[
                *transforms,
                grain.Batch(
                    batch_size=cfg.dataset.batch_size,
                    drop_remainder=drop_remainder,
                ),
            ],
            worker_count=cfg.dataset.worker_count,
            read_options=read_options,
        )

    _t = time.monotonic()
    train_dataloader = make_loader(train_source, shuffle=True, drop_remainder=True)
    eval_dataloader = make_loader(eval_source, shuffle=False, drop_remainder=False)
    logger.info("DataLoader creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    sample_batch = next(iter(train_dataloader))
    logger.info("First batch fetch took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    key = jax.random.key(cfg.seed)
    init_key, init_sample_key, train_key, eval_key = jax.random.split(key, num=4)
    model = instantiate(cfg.tokenizer)
    params = model.init({"params": init_key, "sample": init_sample_key}, sample_batch)
    logger.info("Model init took %.1fs", time.monotonic() - _t)

    if cfg.lpips_weight > 0:
        _t = time.monotonic()
        get_lpips_loss_fn()
        logger.info("LPIPS init took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    optimizer = optax.adam(cfg.learning_rate)
    state = TokenizerTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        mse_sq_ema=jnp.ones((), dtype=jnp.float32),
        lpips_sq_ema=jnp.ones((), dtype=jnp.float32),
    )
    logger.info("TrainState creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    checkpoint_manager: CheckpointManager = instantiate(cfg.checkpoint.manager)
    logger.info("CheckpointManager creation took %.1fs", time.monotonic() - _t)
    if cfg.checkpoint.resume_step is not None:
        state = checkpoint_manager.restore(
            target=state,
            step=int(cfg.checkpoint.resume_step),
        )
        logger.info(f"Resumed tokenizer training from step {state.step}")

    step = int(state.step)
    logger.info("Tokenizer training target step: %d", total_steps)

    if step >= total_steps:
        logger.info(
            "Current step %d is already at or above total_steps=%d; exiting.",
            step,
            total_steps,
        )
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()
        wb.finish()
        return

    t0 = time.monotonic()
    for batch in train_dataloader:
        t1 = time.monotonic()

        batch = jax.device_put(batch)
        t2 = time.monotonic()

        sample_key = jax.random.fold_in(train_key, step)
        state, metrics = train_step(
            state,
            batch,
            sample_key,
            cfg.lpips_weight,
            patch_size=cfg.dataset.patch_size,
            width_tokens=cfg.tokenizer.x_len,
            height_tokens=cfg.tokenizer.y_len,
        )
        jax.block_until_ready(metrics)
        t3 = time.monotonic()

        step = int(state.step)

        t_eval = 0.0
        if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
            t_eval_start = time.monotonic()
            totals: dict[str, float] = {}
            num_batches = 0
            for batch_idx, eval_batch in enumerate(
                itertools.islice(iter(eval_dataloader), cfg.dataset.eval.max_batches)
            ):
                eval_sample_key = jax.random.fold_in(eval_key, batch_idx)
                batch_metrics = eval_step(
                    state,
                    eval_batch,
                    eval_sample_key,
                    cfg.lpips_weight,
                    patch_size=cfg.dataset.patch_size,
                    width_tokens=cfg.tokenizer.x_len,
                    height_tokens=cfg.tokenizer.y_len,
                )
                for k, v in batch_metrics.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                if batch_idx == 0:
                    reconstructed, mask = eval_visualization_step(
                        state, eval_batch, eval_sample_key
                    )
                    grid = build_reconstruction_grid(
                        eval_batch,
                        reconstructed,
                        mask,
                        patch_size=cfg.dataset.patch_size,
                        width_tokens=cfg.tokenizer.x_len,
                        height_tokens=cfg.tokenizer.y_len,
                        pad_width=tuple(cfg.dataset.pad_width),
                        frame_key=jax.random.fold_in(eval_key, step),
                        num_frames=int(cfg.dataset.eval.log_frames),
                    )
                    wb.log_image(
                        "eval/reconstructions",
                        grid,
                        step=step,
                        caption="Columns: original, reconstructed, masked input",
                    )
                num_batches += 1
            if num_batches > 0:
                eval_metrics = {k: v / num_batches for k, v in totals.items()}
                wb.log(
                    {f"eval/{k}": v for k, v in eval_metrics.items()},
                    step=step,
                )
            t_eval = time.monotonic() - t_eval_start
            logger.info(
                "Eval at step %d — %d batches in %.3fs", step, num_batches, t_eval
            )

        if step % cfg.log_interval == 0:
            wb.log(
                {k: float(v) for k, v in metrics.items()},
                step=step,
            )

        if checkpoint_manager.should_save(step):
            checkpoint_manager.save(step=step, state=state)

        t4 = time.monotonic()
        logger.info(
            "Step %d — data: %.3fs, transfer: %.3fs, compute: %.3fs, overhead: %.3fs",
            step,
            t1 - t0,
            t2 - t1,
            t3 - t2,
            t4 - t3 - t_eval,
        )
        if step >= total_steps:
            logger.info(
                "Reached total_steps=%d; stopping tokenizer training.", total_steps
            )
            break
        t0 = time.monotonic()

    if step >= total_steps and not checkpoint_manager.should_save(step):
        checkpoint_manager.save(step=step, state=state, force=True)
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    wb.finish()


if __name__ == "__main__":
    main()
