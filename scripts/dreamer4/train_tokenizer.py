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
from visionary.tokenizer import Tokenizer

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
    motion_loss: jax.Array,
) -> TokenizerTrainState:
    mse_loss = mse_loss.astype(jnp.float32)
    lpips_loss = lpips_loss.astype(jnp.float32)
    motion_loss = motion_loss.astype(jnp.float32)
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
        motion_sq_ema=optax.incremental_update(
            jnp.square(motion_loss),
            state.motion_sq_ema.astype(motion_loss.dtype),
            step_size,
        ),
    )


def compute_loss_metrics(
    state: TokenizerTrainState,
    batch: PreprocessedVideoDataset,
    reconstructed: jax.Array,
    mask: jax.Array,
    lpips_weight: float,
    motion_loss_weight: float,
    masked_mse: bool,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
):
    reconstructed_f32 = reconstructed.astype(jnp.float32)
    original = batch["video"].astype(jnp.float32) / 255.0
    mask_f32 = jnp.expand_dims(mask, axis=-1).astype(reconstructed_f32.dtype)
    mse_weights = mask_f32 if masked_mse else jnp.ones_like(mask_f32)
    num_mse = jnp.maximum(
        jnp.sum(mse_weights) * reconstructed_f32.shape[-1],
        1.0,
    )
    mse_loss = jnp.sum(jnp.square(reconstructed_f32 - original) * mse_weights) / num_mse
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

    if motion_loss_weight > 0 and reconstructed_f32.shape[1] > 1:
        reconstructed_diff = reconstructed_f32[:, 1:] - reconstructed_f32[:, :-1]
        original_diff = original[:, 1:] - original[:, :-1]
        motion_loss = jnp.mean(jnp.square(reconstructed_diff - original_diff))
    else:
        motion_loss = jnp.zeros((), dtype=mse_loss.dtype)
    motion_rms = jnp.sqrt(state.motion_sq_ema.astype(mse_loss.dtype) + LOSS_RMS_EPS)
    normalized_motion_loss = motion_loss / jax.lax.stop_gradient(motion_rms)

    raw_loss = mse_loss + lpips_weight * lpips_loss + motion_loss_weight * motion_loss
    loss = (
        normalized_mse_loss
        + lpips_weight * normalized_lpips_loss
        + motion_loss_weight * normalized_motion_loss
    )
    metrics = {
        "loss": loss,
        "raw_loss": raw_loss,
        "mse_loss": mse_loss,
        "lpips_loss": lpips_loss,
        "motion_loss": motion_loss,
        "normalized_mse_loss": normalized_mse_loss,
        "normalized_lpips_loss": normalized_lpips_loss,
        "normalized_motion_loss": normalized_motion_loss,
        "mse_rms": mse_rms,
        "lpips_rms": lpips_rms,
        "motion_rms": motion_rms,
        "mask_ratio": jnp.mean(mask_f32),
    }
    return loss, metrics


@partial(
    jax.jit,
    static_argnames=(
        "lpips_weight",
        "motion_loss_weight",
        "masked_mse",
        "patch_size",
        "width_tokens",
        "height_tokens",
    ),
)
def train_step(
    state: TokenizerTrainState,
    batch: PreprocessedVideoDataset,
    sample_key: jax.Array,
    lpips_weight: float,
    motion_loss_weight: float,
    masked_mse: bool,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
):
    def loss_fn(params):
        reconstructed, mask = state.apply_fn(
            params,
            batch,
            rngs={"sample": sample_key},
        )
        return compute_loss_metrics(
            state,
            batch,
            reconstructed,
            mask,
            lpips_weight,
            motion_loss_weight,
            masked_mse,
            patch_size,
            width_tokens,
            height_tokens,
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = update_loss_ema(
        state,
        metrics["mse_loss"],
        metrics["lpips_loss"],
        metrics["motion_loss"],
    )
    return state, metrics


@partial(
    jax.jit,
    static_argnames=(
        "lpips_weight",
        "motion_loss_weight",
        "masked_mse",
        "patch_size",
        "width_tokens",
        "height_tokens",
        "pad_width",
    ),
)
def eval_step(
    state: TokenizerTrainState,
    batch: PreprocessedVideoDataset,
    sample_key: jax.Array,
    lpips_weight: float,
    motion_loss_weight: float,
    masked_mse: bool,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
    pad_width: tuple[int, int],
):
    model_key, frame_key = jax.random.split(sample_key)
    reconstructed, mask = state.apply_fn(
        state.params,
        batch,
        method=Tokenizer.reconstruct_eval,
        rngs={"sample": model_key},
    )
    _, metrics = compute_loss_metrics(
        state,
        batch,
        reconstructed,
        mask,
        lpips_weight,
        motion_loss_weight,
        masked_mse,
        patch_size,
        width_tokens,
        height_tokens,
    )
    sampled_frames = sample_sequence_frames(
        batch,
        reconstructed,
        mask,
        patch_size=patch_size,
        width_tokens=width_tokens,
        height_tokens=height_tokens,
        pad_width=pad_width,
        frame_key=frame_key,
    )
    return metrics, sampled_frames


def trim_padding(images: np.ndarray, pad_width: tuple[int, int]) -> np.ndarray:
    height_pad, width_pad = (int(p) for p in pad_width)
    h_slice = slice(height_pad, -height_pad if height_pad else None)
    w_slice = slice(width_pad, -width_pad if width_pad else None)
    return images[:, :, h_slice, w_slice, :]


def patches_to_images(
    patches: jax.Array,
    *,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
    pad_width: tuple[int, int],
) -> jax.Array:
    images = unpatchify(
        rearrange(patches, "b n d -> b 1 n d"),
        patch_size,
        width_tokens,
        height_tokens,
    )
    images = trim_padding(images, pad_width)[:, 0]
    return jnp.clip(jnp.rint(images * 255.0), 0, 255).astype(jnp.uint8)


def sample_sequence_frames(
    batch: PreprocessedVideoDataset,
    reconstructed: jax.Array,
    mask: jax.Array,
    *,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
    pad_width: tuple[int, int],
    frame_key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    original = jnp.asarray(batch["video"], dtype=jnp.float32) / 255.0
    reconstructed = reconstructed.astype(jnp.float32)
    mask_f32 = jnp.expand_dims(mask, axis=-1).astype(original.dtype)
    masked_input = original * (1.0 - mask_f32)

    batch_size, seq_len, _, _ = original.shape
    batch_indices = jnp.arange(batch_size)
    frame_indices = jax.random.randint(frame_key, (batch_size,), 0, seq_len)

    original = original[batch_indices, frame_indices]
    reconstructed = reconstructed[batch_indices, frame_indices]
    masked_input = masked_input[batch_indices, frame_indices]

    return (
        patches_to_images(
            original,
            patch_size=patch_size,
            width_tokens=width_tokens,
            height_tokens=height_tokens,
            pad_width=pad_width,
        ),
        patches_to_images(
            reconstructed,
            patch_size=patch_size,
            width_tokens=width_tokens,
            height_tokens=height_tokens,
            pad_width=pad_width,
        ),
        patches_to_images(
            masked_input,
            patch_size=patch_size,
            width_tokens=width_tokens,
            height_tokens=height_tokens,
            pad_width=pad_width,
        ),
    )


def build_reconstruction_grid(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    masked_inputs: np.ndarray,
    *,
    frame_key: jax.Array,
    num_frames: int,
) -> np.ndarray:
    total_frames = originals.shape[0]
    num_frames = min(int(num_frames), total_frames)
    frame_indices = np.asarray(
        jax.device_get(
            jax.random.choice(
                frame_key, total_frames, shape=(num_frames,), replace=False
            )
        )
    )

    originals = originals[frame_indices]
    reconstructions = reconstructions[frame_indices]
    masked_inputs = masked_inputs[frame_indices]

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
        PreprocessAndPatchify(
            cfg.dataset.patch_size,
            tuple(cfg.dataset.pad_width),
            tuple(cfg.dataset.resize_shape),
        ),
    ]

    def make_loader(source, *, shuffle: bool, drop_remainder: bool, seed: int):
        sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.ShardByJaxProcess(),
            shuffle=shuffle,
            seed=seed,
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
    train_dataloader = make_loader(
        train_source,
        shuffle=True,
        drop_remainder=True,
        seed=int(cfg.seed),
    )
    logger.info("Train DataLoader creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    sample_batch = next(iter(train_dataloader))
    logger.info("First batch fetch took %.1fs", time.monotonic() - _t)
    if bool(cfg.overfit_single_batch):
        logger.info("Overfit mode enabled: reusing the first sampled batch for training.")

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
        motion_sq_ema=jnp.ones((), dtype=jnp.float32),
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
    train_batches = (
        itertools.repeat(sample_batch)
        if bool(cfg.overfit_single_batch)
        else train_dataloader
    )
    for batch in train_batches:
        t1 = time.monotonic()

        batch = jax.device_put(batch)
        t2 = time.monotonic()

        sample_key = jax.random.fold_in(train_key, step)
        state, metrics = train_step(
            state,
            batch,
            sample_key,
            cfg.lpips_weight,
            cfg.motion_loss_weight,
            bool(cfg.masked_mse),
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
            eval_run_key = jax.random.fold_in(eval_key, step)
            if bool(cfg.overfit_single_batch):
                eval_batches = [sample_batch]
            else:
                eval_dataloader = make_loader(
                    eval_source,
                    shuffle=True,
                    drop_remainder=False,
                    seed=int(cfg.seed) + step,
                )
                eval_batches = itertools.islice(
                    iter(eval_dataloader), cfg.dataset.eval.max_batches
                )
            vis_original_batches = []
            vis_reconstruction_batches = []
            vis_masked_batches = []
            for batch_idx, eval_batch in enumerate(eval_batches):
                eval_sample_key = jax.random.fold_in(eval_run_key, batch_idx)
                batch_metrics, sampled_frames = eval_step(
                    state,
                    eval_batch,
                    eval_sample_key,
                    cfg.lpips_weight,
                    cfg.motion_loss_weight,
                    bool(cfg.masked_mse),
                    patch_size=cfg.dataset.patch_size,
                    width_tokens=cfg.tokenizer.x_len,
                    height_tokens=cfg.tokenizer.y_len,
                    pad_width=tuple(cfg.dataset.pad_width),
                )
                for k, v in batch_metrics.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                sampled_originals, sampled_reconstructions, sampled_masked_inputs = (
                    jax.device_get(sampled_frames)
                )
                vis_original_batches.append(np.asarray(sampled_originals))
                vis_reconstruction_batches.append(np.asarray(sampled_reconstructions))
                vis_masked_batches.append(np.asarray(sampled_masked_inputs))
                num_batches += 1
            if num_batches > 0:
                eval_metrics = {k: v / num_batches for k, v in totals.items()}
                wb.log(
                    {f"eval/{k}": v for k, v in eval_metrics.items()},
                    step=step,
                )
                grid = build_reconstruction_grid(
                    np.concatenate(vis_original_batches, axis=0),
                    np.concatenate(vis_reconstruction_batches, axis=0),
                    np.concatenate(vis_masked_batches, axis=0),
                    frame_key=jax.random.fold_in(eval_run_key, num_batches),
                    num_frames=int(cfg.dataset.eval.log_frames),
                )
                wb.log_image(
                    "eval/reconstructions",
                    grid,
                    step=step,
                    caption="Columns: original, reconstructed, masked input",
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
