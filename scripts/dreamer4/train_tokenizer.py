import itertools
import logging
import time
from functools import lru_cache, partial

import grain.python as grain
import hydra
import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from flax.training.train_state import TrainState
from hydra.utils import instantiate
from jaxlpips import LPIPS
from omegaconf import DictConfig

from visionary.common.checkpoint import CheckpointManager
from visionary.common.wandb import WandbLogger
from visionary.dataset import (
    PreprocessAndPatchify,
    PreprocessedVideoDataset,
    RandomVideoCrop,
    VideoDataSource,
)

logger = logging.getLogger(__name__)


LPIPS_PRETRAINED_NETWORK = "alexnet"


@lru_cache(maxsize=1)
def get_lpips_loss_fn():
    return LPIPS(pretrained_network=LPIPS_PRETRAINED_NETWORK)


def compute_lpips_loss(
    original: jax.Array,
    reconstructed: jax.Array,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
) -> jax.Array:
    def unpatchify(images: jax.Array) -> jax.Array:
        return rearrange(
            images,
            "b t (h w) (p1 p2 c) -> b t (h p1) (w p2) c",
            p1=patch_size,
            p2=patch_size,
            h=height_tokens,
            w=width_tokens,
        )

    original_images = unpatchify(original)
    reconstructed_images = unpatchify(reconstructed)
    original_images, reconstructed_images = (
        original_images * 2.0 - 1.0,
        reconstructed_images * 2.0 - 1.0,
    )
    # jaxlpips expects NHWC images, so fold time into the batch axis.
    original_images = rearrange(original_images, "b t h w c -> (b t) h w c")
    reconstructed_images = rearrange(reconstructed_images, "b t h w c -> (b t) h w c")
    return jnp.mean(get_lpips_loss_fn()(original_images, reconstructed_images))


def compute_losses(
    params,
    state: TrainState,
    batch: PreprocessedVideoDataset,
    mask_key: jax.Array,
    lpips_weight: float,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
):
    reconstructed = state.apply_fn(params, batch, rngs={"mask": mask_key})
    reconstructed_f32 = reconstructed.astype(jnp.float32)
    original = batch["video"].astype(jnp.float32) / 255.0
    mse_loss = jnp.mean(jnp.square(reconstructed_f32 - original))
    if lpips_weight > 0:
        lpips_loss = compute_lpips_loss(
            original,
            reconstructed_f32,
            patch_size=patch_size,
            width_tokens=width_tokens,
            height_tokens=height_tokens,
        )
    else:
        lpips_loss = jnp.zeros((), dtype=mse_loss.dtype)
    loss = mse_loss + lpips_weight * lpips_loss
    return loss, {"loss": loss, "mse_loss": mse_loss, "lpips_loss": lpips_loss}


@partial(
    jax.jit,
    static_argnames=("lpips_weight", "patch_size", "width_tokens", "height_tokens"),
)
def train_step(
    state: TrainState,
    batch: PreprocessedVideoDataset,
    mask_key: jax.Array,
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
            mask_key,
            lpips_weight,
            patch_size,
            width_tokens,
            height_tokens,
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


@partial(
    jax.jit,
    static_argnames=("lpips_weight", "patch_size", "width_tokens", "height_tokens"),
)
def eval_step(
    state: TrainState,
    batch: PreprocessedVideoDataset,
    mask_key: jax.Array,
    lpips_weight: float,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
):
    _, metrics = compute_losses(
        state.params,
        state,
        batch,
        mask_key,
        lpips_weight,
        patch_size,
        width_tokens,
        height_tokens,
    )
    return metrics


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
    init_key, init_mask_key, train_key, eval_key = jax.random.split(key, num=4)
    model = instantiate(cfg.tokenizer)
    params = model.init({"params": init_key, "mask": init_mask_key}, sample_batch)
    logger.info("Model init took %.1fs", time.monotonic() - _t)

    if cfg.lpips_weight > 0:
        _t = time.monotonic()
        get_lpips_loss_fn()
        logger.info("LPIPS init took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    optimizer = optax.adam(cfg.learning_rate)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
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

        mask_key = jax.random.fold_in(train_key, step)
        state, metrics = train_step(
            state,
            batch,
            mask_key,
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
                eval_mask_key = jax.random.fold_in(eval_key, batch_idx)
                batch_metrics = eval_step(
                    state,
                    eval_batch,
                    eval_mask_key,
                    cfg.lpips_weight,
                    patch_size=cfg.dataset.patch_size,
                    width_tokens=cfg.tokenizer.x_len,
                    height_tokens=cfg.tokenizer.y_len,
                )
                for k, v in batch_metrics.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
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
