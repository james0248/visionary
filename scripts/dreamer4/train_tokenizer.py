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
from hydra.utils import instantiate
from jaxlpips import LPIPS
from omegaconf import DictConfig, OmegaConf

from visionary.common.checkpoint import CheckpointManager, save_model_export
from visionary.common.jax import fold_in_many, maybe_initialize_distributed
from visionary.common.train_state import TokenizerTrainState
from visionary.common.wandb import WandbLogger
from visionary.dataset import RandomVideoCrop, VideoDataset, VideoDataSource
from visionary.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


LPIPS_PRETRAINED_NETWORK = "alexnet"
LOSS_RMS_DECAY = 0.99
LOSS_RMS_EPS = 1e-8


def make_host_seed(*values: int) -> int:
    seed_sequence = np.random.SeedSequence([int(value) for value in values])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


@lru_cache(maxsize=1)
def get_lpips_loss_fn():
    return LPIPS(pretrained_network=LPIPS_PRETRAINED_NETWORK)


def compute_lpips_loss(
    original: jax.Array,
    reconstructed: jax.Array,
) -> jax.Array:
    original_images = original * 2.0 - 1.0
    reconstructed_images = reconstructed * 2.0 - 1.0
    original_images = original_images.reshape((-1, *original_images.shape[2:]))
    reconstructed_images = reconstructed_images.reshape((-1, *reconstructed_images.shape[2:]))
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


def compute_loss_metrics(
    state: TokenizerTrainState,
    original_images: jax.Array,
    reconstructed: jax.Array,
    mask: jax.Array,
    lpips_weight: float,
):
    original = original_images.astype(jnp.float32)
    reconstructed_f32 = reconstructed.astype(jnp.float32)
    mask_f32 = mask.astype(reconstructed_f32.dtype)
    reconstruction_error = reconstructed_f32 - original

    mse_loss = jnp.mean(jnp.square(reconstruction_error))
    mse_rms = jnp.sqrt(state.mse_sq_ema.astype(mse_loss.dtype) + LOSS_RMS_EPS)
    normalized_mse_loss = mse_loss / jax.lax.stop_gradient(mse_rms)

    lpips_rms = jnp.sqrt(state.lpips_sq_ema.astype(mse_loss.dtype) + LOSS_RMS_EPS)
    if lpips_weight > 0:
        lpips_loss = compute_lpips_loss(original, reconstructed_f32)
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


@partial(jax.jit, static_argnames=("lpips_weight",))
def train_step(
    state: TokenizerTrainState,
    batch: VideoDataset,
    base_sample_key: jax.Array,
    global_step: int,
    lpips_weight: float,
):
    sample_key = fold_in_many(base_sample_key, global_step)

    def loss_fn(params):
        original_images, reconstructed, mask = state.apply_fn(
            params,
            batch,
            method=Tokenizer.reconstruct,
            rngs={"sample": sample_key},
        )
        return compute_loss_metrics(
            state,
            original_images,
            reconstructed,
            mask,
            lpips_weight,
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = update_loss_ema(
        state,
        metrics["mse_loss"],
        metrics["lpips_loss"],
    )
    return state, metrics


@partial(jax.jit, static_argnames=("lpips_weight",))
def eval_step(
    state: TokenizerTrainState,
    batch: VideoDataset,
    base_sample_key: jax.Array,
    global_step: int,
    batch_index: int,
    lpips_weight: float,
):
    sample_key = fold_in_many(base_sample_key, global_step, batch_index)
    model_key, frame_key = jax.random.split(sample_key)
    original_images, reconstructed, mask = state.apply_fn(
        state.params,
        batch,
        mask_prob=0.1,
        independent=jnp.zeros((batch["video"].shape[0],), dtype=bool),
        method=Tokenizer.reconstruct,
        rngs={"sample": model_key},
    )
    _, metrics = compute_loss_metrics(
        state,
        original_images,
        reconstructed,
        mask,
        lpips_weight,
    )
    sampled_frames = sample_sequence_frames(
        original_images,
        reconstructed,
        mask,
        frame_key=frame_key,
    )
    return metrics, sampled_frames


def sample_sequence_frames(
    original: jax.Array,
    reconstructed: jax.Array,
    mask: jax.Array,
    frame_key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    original = original.astype(jnp.float32)
    reconstructed = reconstructed.astype(jnp.float32)
    mask_f32 = mask.astype(original.dtype)
    masked_input = original * (1.0 - mask_f32)

    batch_size, seq_len, _, _, _ = original.shape
    batch_indices = jnp.arange(batch_size)
    frame_indices = jax.random.randint(frame_key, (batch_size,), 0, seq_len)

    original = original[batch_indices, frame_indices]
    reconstructed = reconstructed[batch_indices, frame_indices]
    masked_input = masked_input[batch_indices, frame_indices]

    return (
        jnp.clip(jnp.rint(original * 255.0), 0, 255).astype(jnp.uint8),
        jnp.clip(jnp.rint(reconstructed * 255.0), 0, 255).astype(jnp.uint8),
        jnp.clip(jnp.rint(masked_input * 255.0), 0, 255).astype(jnp.uint8),
    )


def build_reconstruction_grid(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    masked_inputs: np.ndarray,
    frame_seed: int,
    num_frames: int,
) -> np.ndarray:
    total_frames = originals.shape[0]
    num_frames = min(int(num_frames), total_frames)
    frame_indices = np.random.default_rng(frame_seed).choice(
        total_frames, size=num_frames, replace=False
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
    maybe_initialize_distributed(logger=logger)
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
    effective_read_threads = max(int(cfg.dataset.worker_count), 1) * int(cfg.dataset.num_threads)
    logger.info(
        "Data loader settings: worker_count=%d num_threads=%d "
        "prefetch_buffer_size=%d effective_read_threads=%d",
        int(cfg.dataset.worker_count),
        int(cfg.dataset.num_threads),
        int(cfg.dataset.prefetch_buffer_size),
        effective_read_threads,
    )
    transforms = [RandomVideoCrop(cfg.dataset.frame_length)]

    def make_loader(source, shuffle: bool, drop_remainder: bool, seed: int):
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

    _t = time.monotonic()
    key = jax.random.key(cfg.seed)
    init_key, init_sample_key, train_key, eval_key = jax.random.split(key, num=4)
    model = instantiate(cfg.tokenizer)
    params = model.init(
        {"params": init_key, "sample": init_sample_key},
        sample_batch,
        method=Tokenizer.reconstruct,
    )
    logger.info("Model init took %.1fs", time.monotonic() - _t)

    if cfg.lpips_weight > 0:
        _t = time.monotonic()
        get_lpips_loss_fn()
        logger.info("LPIPS init took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    optimizer = optax.adam(cfg.learning_rate)
    state = TokenizerTrainState.create(
        model.apply,
        params,
        optimizer,
    )
    logger.info("TrainState creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    checkpoint_manager: CheckpointManager = instantiate(cfg.checkpoint.manager)
    checkpoint_manager.save_metadata({"config": OmegaConf.to_container(cfg, resolve=False)})
    logger.info("CheckpointManager creation took %.1fs", time.monotonic() - _t)
    train_iterator = iter(train_dataloader)

    def save_checkpoint(step: int, force: bool = False) -> None:
        checkpoint_manager.save(
            step=step,
            state=state,
            extra_items=iterator_items(),
            force=force,
        )
        save_model_export(checkpoint_manager.directory, step, cfg.tokenizer, state.params)

    def iterator_items():
        return {"train_iterator": train_iterator}

    resume_spec = cfg.checkpoint.resume_step
    resume_step = None
    if resume_spec is not None:
        if isinstance(resume_spec, str) and resume_spec.strip().lower() == "latest":
            resume_step = checkpoint_manager.latest_step()
            if resume_step is None:
                logger.info(
                    "No tokenizer checkpoint found in %s; starting fresh.",
                    checkpoint_manager.directory,
                )
        else:
            resume_step = int(resume_spec)

    if resume_step is not None:
        state = checkpoint_manager.restore(
            target=state, step=resume_step, extra_items=iterator_items()
        )
        logger.info("Resumed tokenizer training from step %d", int(state.step))

    step = int(jax.device_get(state.step))
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
    while True:
        current_step = step
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)
        t1 = time.monotonic()

        batch = jax.device_put(batch)
        t2 = time.monotonic()

        state, metrics = train_step(
            state,
            batch,
            train_key,
            current_step,
            cfg.lpips_weight,
        )

        step = current_step + 1

        should_log_train = step % cfg.log_interval == 0
        if should_log_train:
            train_metrics = jax.device_get(metrics)
            wb.log(
                {k: float(v) for k, v in train_metrics.items()},
                step=step,
            )

        t_eval = 0.0
        if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
            t_eval_start = time.monotonic()
            totals: dict[str, float] = {}
            num_batches = 0
            eval_dataloader = make_loader(
                eval_source,
                shuffle=True,
                drop_remainder=False,
                seed=int(cfg.seed) + step,
            )
            eval_batches = itertools.islice(iter(eval_dataloader), cfg.dataset.eval.max_batches)
            vis_original_batches = []
            vis_reconstruction_batches = []
            vis_masked_batches = []
            for batch_idx, eval_batch in enumerate(eval_batches):
                batch_metrics, sampled_frames = eval_step(
                    state,
                    eval_batch,
                    eval_key,
                    step,
                    batch_idx,
                    cfg.lpips_weight,
                )
                batch_metrics, sampled_frames = jax.device_get((batch_metrics, sampled_frames))
                for k, v in batch_metrics.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                sampled_originals, sampled_reconstructions, sampled_masked_inputs = sampled_frames
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
                    frame_seed=make_host_seed(cfg.seed, step, num_batches),
                    num_frames=int(cfg.dataset.eval.log_frames),
                )
                wb.log_image(
                    "eval/reconstructions",
                    grid,
                    step=step,
                    caption="Columns: original, reconstructed, masked input",
                )
            t_eval = time.monotonic() - t_eval_start
            logger.info("Eval at step %d - %d batches in %.3fs", step, num_batches, t_eval)

        if checkpoint_manager.should_save(step):
            save_checkpoint(step)

        t4 = time.monotonic()
        if should_log_train:
            logger.info(
                "Step %d - data: %.3fs, transfer: %.3fs, eval: %.3fs, loop: %.3fs",
                step,
                t1 - t0,
                t2 - t1,
                t_eval,
                t4 - t0,
            )
        if step >= total_steps:
            logger.info("Reached total_steps=%d; stopping tokenizer training.", total_steps)
            break
        t0 = time.monotonic()

    if step >= total_steps and not checkpoint_manager.should_save(step):
        save_checkpoint(step, force=True)
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    wb.finish()


if __name__ == "__main__":
    main()
