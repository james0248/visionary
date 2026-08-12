import itertools
import logging
import time
from functools import lru_cache

import grain.python as grain
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra.utils import instantiate
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf import DictConfig, OmegaConf

from visionary.common.checkpoint import (
    CheckpointManager,
    save_model_export,
    save_preprocessor_export,
)
from visionary.common.jax import (
    data_parallel_mesh,
    fold_in_many,
    global_batch_to_local,
    init_distributed_if_pod,
    local_batch_to_global,
)
from visionary.common.timing import PhaseTimer
from visionary.common.train_state import TokenizerTrainState
from visionary.common.wandb import WandbLogger
from visionary.dataset import VideoDataset
from visionary.lpips import LPIPS
from visionary.models.dreamer4.tokenizer import Tokenizer
from visionary.models.dreamer4.tokenizer_preprocessor import TokenizerPreprocessor

logger = logging.getLogger(__name__)


LPIPS_PRETRAINED_NETWORK = "alexnet"
LOSS_RMS_DECAY = 0.99
LOSS_RMS_EPS = 1e-8
DATA_AXIS = "data"


def build_optimizer(cfg) -> optax.GradientTransformation:

    def build_wsd_schedule(cfg) -> optax.Schedule:
        total = int(cfg.total_steps)
        warmup = max(int(total * cfg.optimizer.warmup_ratio), 1)
        decay = int(total * cfg.optimizer.decay_ratio)
        stable = max(total - warmup - decay, 0)
        return optax.join_schedules(
            [
                optax.linear_schedule(0.0, cfg.optimizer.peak_lr, warmup),
                optax.constant_schedule(cfg.optimizer.peak_lr),
                optax.linear_schedule(cfg.optimizer.peak_lr, 0.0, decay),
            ],
            boundaries=[warmup, warmup + stable],
        )

    schedule = build_wsd_schedule(cfg)
    opt = cfg.optimizer
    adam_ratio = float(opt.adam_lr_ratio)

    def adam_schedule(step):
        return schedule(step) * adam_ratio

    def muon_dimension_numbers(params):
        from optax.contrib import MuonDimensionNumbers

        # Do not use Muon on independent lookup vectors
        embedding_like = {"embedding", "latent_tokens", "image_tokens"}

        def mapper(path, x):
            names = {str(getattr(key, "key", key)).lower() for key in path}
            if names & embedding_like:
                return None

            # Apply Muon on all matrices (not 1D)
            return MuonDimensionNumbers() if x.ndim >= 2 else None

        return jax.tree_util.tree_map_with_path(mapper, params)

    return optax.contrib.muon(
        learning_rate=schedule,
        beta=opt.muon_beta,
        ns_steps=opt.ns_steps,
        nesterov=opt.nesterov,
        weight_decay=opt.weight_decay,
        adam_learning_rate=adam_schedule,
        adam_b1=opt.adam_b1,
        adam_b2=opt.adam_b2,
        adam_weight_decay=0.0,
        muon_weight_dimension_numbers=muon_dimension_numbers,
    )


@lru_cache(maxsize=1)
def get_lpips_loss_fn():
    return LPIPS(pretrained_network=LPIPS_PRETRAINED_NETWORK)


def compute_lpips_loss(
    original: jax.Array,
    reconstructed: jax.Array,
    preprocessor: TokenizerPreprocessor,
) -> jax.Array:
    original_images = preprocessor.patches_to_images(original)
    reconstructed_images = preprocessor.patches_to_images(reconstructed)
    original_images = original_images * 2.0 - 1.0
    reconstructed_images = reconstructed_images * 2.0 - 1.0
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
        mse_sq_ema=optax.incremental_update(jnp.square(mse_loss), state.mse_sq_ema.astype(mse_loss.dtype), step_size),
        lpips_sq_ema=optax.incremental_update(
            jnp.square(lpips_loss),
            state.lpips_sq_ema.astype(lpips_loss.dtype),
            step_size,
        ),
    )


def compute_loss_metrics(
    state: TokenizerTrainState,
    batch: VideoDataset,
    reconstructed: jax.Array,
    mask: jax.Array,
    latent: jax.Array,
    lpips_weight: float,
    lpips_frame_stride: int,
    preprocessor: TokenizerPreprocessor,
):
    original = batch["video"].astype(jnp.float32) / 255.0
    reconstructed = reconstructed.astype(jnp.float32)
    mask = jnp.expand_dims(mask, axis=-1).astype(reconstructed.dtype)
    reconstruction_error = reconstructed - original

    mse_loss = jnp.mean(jnp.square(reconstruction_error))
    mse_rms = jnp.sqrt(state.mse_sq_ema.astype(mse_loss.dtype) + LOSS_RMS_EPS)
    normalized_mse_loss = mse_loss / jax.lax.stop_gradient(mse_rms)

    lpips_rms = jnp.sqrt(state.lpips_sq_ema.astype(mse_loss.dtype) + LOSS_RMS_EPS)
    if lpips_weight > 0:
        stride = max(int(lpips_frame_stride), 1)
        lpips_loss = compute_lpips_loss(original[:, ::stride], reconstructed[:, ::stride], preprocessor)
    else:
        lpips_loss = jnp.zeros((), dtype=mse_loss.dtype)
    normalized_lpips_loss = lpips_loss / jax.lax.stop_gradient(lpips_rms)

    latent = latent.astype(jnp.float32).reshape(-1, latent.shape[-1])
    latent_mean = jnp.mean(latent, axis=0)
    latent_std = jnp.std(latent, axis=0)

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
        "mask_ratio": jnp.mean(mask),
        "latent_mean_absmax": jnp.max(jnp.abs(latent_mean)),
        "latent_std_min": jnp.min(latent_std),
        "latent_std_max": jnp.max(latent_std),
    }
    return loss, metrics


def train_step(
    state: TokenizerTrainState,
    batch: VideoDataset,
    base_sample_key: jax.Array,
    global_step: int,
    lpips_weight: float,
    lpips_frame_stride: int,
    preprocessor: TokenizerPreprocessor,
):
    model_key = fold_in_many(base_sample_key, global_step)

    def loss_fn(params):
        reconstructed, mask, latent = state.apply_fn(
            params, batch, method=Tokenizer.reconstruct, rngs={"sample": model_key}
        )
        return compute_loss_metrics(
            state, batch, reconstructed, mask, latent, lpips_weight, lpips_frame_stride, preprocessor
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = update_loss_ema(state, metrics["mse_loss"], metrics["lpips_loss"])
    return state, metrics


def eval_step(
    state: TokenizerTrainState,
    batch: VideoDataset,
    base_sample_key: jax.Array,
    global_step: int,
    batch_index: int,
    lpips_weight: float,
    lpips_frame_stride: int,
    preprocessor: TokenizerPreprocessor,
):
    sample_key = fold_in_many(base_sample_key, global_step, batch_index)
    model_key, frame_key = jax.random.split(sample_key)
    reconstructed, mask, latent = state.apply_fn(
        state.params,
        batch,
        mask_prob=0.1,
        independent=jnp.zeros((batch["video"].shape[0],), dtype=bool),
        method=Tokenizer.reconstruct,
        rngs={"sample": model_key},
    )
    _, metrics = compute_loss_metrics(
        state, batch, reconstructed, mask, latent, lpips_weight, lpips_frame_stride, preprocessor
    )
    sampled_frames = sample_sequence_frames(batch, reconstructed, mask, preprocessor, frame_key)
    return metrics, sampled_frames


def sample_sequence_frames(
    batch: VideoDataset,
    reconstructed: jax.Array,
    mask: jax.Array,
    preprocessor: TokenizerPreprocessor,
    frame_key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    original = batch["video"].astype(jnp.float32) / 255.0
    reconstructed = reconstructed.astype(jnp.float32)
    masked_input = original * (1.0 - jnp.expand_dims(mask, axis=-1).astype(original.dtype))

    batch_size, seq_len, _, _ = original.shape
    batch_indices = jnp.arange(batch_size)
    frame_indices = jax.random.randint(frame_key, (batch_size,), 0, seq_len)

    original = original[batch_indices, frame_indices]
    reconstructed = reconstructed[batch_indices, frame_indices]
    masked_input = masked_input[batch_indices, frame_indices]
    sampled_frames = []
    for patches in (original, reconstructed, masked_input):
        images = preprocessor.patches_to_images(patches)
        sampled_frames.append(jnp.clip(jnp.rint(images * 255.0), 0, 255).astype(jnp.uint8))
    return tuple(sampled_frames)


def build_reconstruction_grid(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    masked_inputs: np.ndarray,
    frame_seed: int,
    num_frames: int,
) -> np.ndarray:
    total_frames = originals.shape[0]
    num_frames = min(int(num_frames), total_frames)
    frame_indices = np.random.default_rng(frame_seed).choice(total_frames, size=num_frames, replace=False)

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
        [row if idx == 0 else np.concatenate([row_sep, row], axis=0) for idx, row in enumerate(rows)],
        axis=0,
    )


@hydra.main(config_path="config", config_name="tokenizer", version_base=None)
def main(cfg: DictConfig):
    init_distributed_if_pod(logger=logger)

    process_index = jax.process_index()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()
    is_primary_process = process_index == 0
    mesh = data_parallel_mesh(DATA_AXIS)
    batch_pspec = P(DATA_AXIS)
    batch_sharding = NamedSharding(mesh, batch_pspec)
    metrics_sharding = NamedSharding(mesh, P())

    logger.info(
        "JAX backend: %s process=%d/%d local_devices=%d global_devices=%d devices=%s",
        jax.default_backend(),
        process_index,
        process_count,
        local_device_count,
        jax.device_count(),
        jax.local_devices(),
    )
    logger.info("Data mesh: %d devices, batch_pspec=%s", mesh.shape[DATA_AXIS], batch_pspec)

    wb = WandbLogger(cfg, enabled=bool(cfg.wandb.enabled) and is_primary_process)
    total_steps = int(cfg.total_steps)

    train_source = instantiate(cfg.dataset.source, data_dir=cfg.dataset.train_dir)
    eval_source = instantiate(cfg.dataset.source, data_dir=cfg.dataset.eval_dir)
    logger.info("Loaded %d training videos and %d eval videos", len(train_source), len(eval_source))

    # Calculate per device batch size
    batch_size_global = int(cfg.dataset.batch_size)
    if batch_size_global % process_count != 0:
        raise ValueError(
            "cfg.dataset.batch_size is global and must be divisible by the process count; "
            f"got batch_size={batch_size_global} process_count={process_count}"
        )
    batch_size_per_process = batch_size_global // process_count
    if batch_size_per_process % local_device_count != 0:
        raise ValueError(
            "global batch_size / process_count must be divisible by "
            "jax.local_device_count() for FSDP input sharding; got "
            f"per_process={batch_size_per_process} local_device_count={local_device_count}"
        )
    logger.info(
        "Batch layout: global=%d per_process=%d per_device=%d",
        batch_size_global,
        batch_size_per_process,
        batch_size_per_process // local_device_count,
    )

    worker_count = int(cfg.dataset.worker_count)
    num_threads = int(cfg.dataset.num_threads)
    prefetch_buffer_size = int(cfg.dataset.prefetch_buffer_size)
    effective_read_threads = max(worker_count, 1) * num_threads
    logger.info(
        "Data loader settings: worker_count=%d num_threads=%d prefetch_buffer_size=%d effective_read_threads=%d",
        worker_count,
        num_threads,
        prefetch_buffer_size,
        effective_read_threads,
    )

    preprocessor = TokenizerPreprocessor(
        resize_shape=tuple(cfg.tokenizer.resize_shape),
        pad_width=tuple(cfg.tokenizer.pad_width),
        patch_size=int(cfg.tokenizer.patch_size),
    )
    logger.info(
        "LPIPS settings: weight=%.3f",
        float(cfg.lpips_weight),
    )

    clip_transform = instantiate(cfg.dataset.clip_transform, frame_length=cfg.dataset.frame_length)
    augment_transform = instantiate(cfg.dataset.augment) if cfg.dataset.augment else None
    logger.info("Train augmentation: %s", augment_transform or "disabled")
    preprocess_transform = preprocessor.as_grain_transform()

    def make_loader(source, shuffle: bool, drop_remainder: bool, seed: int, augment: bool = False):
        sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.ShardByJaxProcess() if process_count > 1 else grain.NoSharding(),
            shuffle=shuffle,
            seed=seed,
        )
        read_options = grain.ReadOptions(
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size,
        )
        operations = [clip_transform]
        if augment and augment_transform is not None:
            operations.append(augment_transform)
        return grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=[
                *operations,
                preprocess_transform,
                grain.Batch(
                    batch_size=batch_size_per_process,
                    drop_remainder=drop_remainder,
                ),
            ],
            worker_count=worker_count,
            read_options=read_options,
        )

    _t = time.monotonic()
    train_dataloader = make_loader(
        train_source,
        shuffle=True,
        drop_remainder=True,
        seed=int(cfg.seed),
        augment=True,
    )
    logger.info("Train DataLoader creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    eval_loader = make_loader(
        eval_source,
        shuffle=False,
        drop_remainder=True,
        seed=int(cfg.seed),
    )
    logger.info("Eval DataLoader creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    sample_batch = next(iter(train_dataloader))
    logger.info("First batch fetch took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    key = jax.random.key(cfg.seed)
    init_key, init_sample_key, train_key, eval_key = jax.random.split(key, num=4)
    model = instantiate(cfg.tokenizer)
    optimizer = build_optimizer(cfg)

    def init_state(batch):
        params = model.init({"params": init_key, "sample": init_sample_key}, batch, method=Tokenizer.reconstruct)
        return TokenizerTrainState.create(model.apply, params, optimizer)

    state = jax.jit(init_state, out_shardings=metrics_sharding)(local_batch_to_global(sample_batch, batch_sharding))
    state_shardings = jax.tree_util.tree_map(lambda _: metrics_sharding, state)
    logger.info("State init took %.1fs", time.monotonic() - _t)

    if cfg.lpips_weight > 0:
        _t = time.monotonic()
        get_lpips_loss_fn()
        logger.info("LPIPS init took %.1fs", time.monotonic() - _t)

    batch_input_shardings = {"video": batch_sharding}
    jit_train_step = jax.jit(
        train_step,
        in_shardings=(state_shardings, batch_input_shardings, metrics_sharding, metrics_sharding),
        out_shardings=(state_shardings, metrics_sharding),
        static_argnames=("lpips_weight", "lpips_frame_stride", "preprocessor"),
        donate_argnums=(0,),
    )
    jit_eval_step = jax.jit(
        eval_step,
        in_shardings=(state_shardings, batch_input_shardings, metrics_sharding, metrics_sharding, metrics_sharding),
        out_shardings=(metrics_sharding, (batch_sharding, batch_sharding, batch_sharding)),
        static_argnames=("lpips_weight", "lpips_frame_stride", "preprocessor"),
    )

    _t = time.monotonic()
    checkpoint_manager: CheckpointManager = instantiate(cfg.checkpoint.manager)
    if is_primary_process:
        checkpoint_manager.save_metadata({"config": OmegaConf.to_container(cfg, resolve=False)})
    logger.info("CheckpointManager creation took %.1fs", time.monotonic() - _t)
    train_iterator = iter(train_dataloader)
    last_checkpoint_step: int | None = None
    export_interval_steps = int(cfg.checkpoint.export_interval_steps)

    def should_export_model(step: int, force: bool) -> bool:
        return force or (export_interval_steps > 0 and step % export_interval_steps == 0)

    def save_checkpoint(step: int, force: bool = False) -> None:
        nonlocal last_checkpoint_step
        saved = checkpoint_manager.save(
            step=step,
            state=state,
            extra_items=iterator_items(),
            force=force,
        )
        if not saved:
            return
        last_checkpoint_step = int(step)
        if should_export_model(step, force=force):
            save_model_export(checkpoint_manager.directory, step, cfg.tokenizer, state.params)
            save_preprocessor_export(
                checkpoint_manager.directory,
                step,
                preprocessor.export_config(),
            )

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
        state = checkpoint_manager.restore(target=state, step=resume_step, extra_items=iterator_items())
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

    timing_start_step = step
    timer = PhaseTimer()
    logger.info(
        "Asynchronous timing mode enabled for tokenizer training; timing logs are averaged over each logging window."
    )
    while True:
        current_step = step
        with timer("data"):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
        with timer("transfer"):
            batch = local_batch_to_global(batch, batch_sharding)
        with timer("compute"):
            state, metrics = jit_train_step(
                state,
                batch,
                train_key,
                jnp.asarray(current_step, dtype=jnp.int32),
                float(cfg.lpips_weight),
                int(cfg.lpips_frame_stride),
                preprocessor,
            )
        step = current_step + 1

        t_eval = 0.0
        if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
            t_eval_start = time.monotonic()
            totals: dict[str, float] = {}
            num_batches = 0
            eval_batches = list(itertools.islice(iter(eval_loader), cfg.dataset.eval.max_batches))
            global_eval_batch_counts = np.asarray(
                multihost_utils.process_allgather(np.asarray(len(eval_batches), dtype=np.int32))
            )
            eval_batches = eval_batches[: int(np.min(global_eval_batch_counts))]
            vis_original_batches = []
            vis_reconstruction_batches = []
            vis_masked_batches = []
            for batch_idx, eval_batch in enumerate(eval_batches):
                batch_metrics, sampled_frames = jit_eval_step(
                    state,
                    local_batch_to_global(eval_batch, batch_sharding),
                    eval_key,
                    jnp.asarray(step, dtype=jnp.int32),
                    jnp.asarray(batch_idx, dtype=jnp.int32),
                    float(cfg.lpips_weight),
                    int(cfg.lpips_frame_stride),
                    preprocessor,
                )
                sampled_frames = global_batch_to_local(sampled_frames)
                sampled_frames = jax.device_get(sampled_frames)
                batch_metrics = jax.device_get(batch_metrics)
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
                if vis_original_batches:
                    grid = build_reconstruction_grid(
                        np.concatenate(vis_original_batches, axis=0),
                        np.concatenate(vis_reconstruction_batches, axis=0),
                        np.concatenate(vis_masked_batches, axis=0),
                        frame_seed=int(
                            jax.random.randint(fold_in_many(eval_key, step, num_batches), (), 0, np.iinfo(np.int32).max)
                        ),
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

        if step % cfg.log_interval == 0:
            # the jitted step is async; syncing on metrics charges the device wait to compute
            with timer("compute"):
                train_metrics = jax.device_get(metrics)
            timing_stats = timer.log(logger, step, step - timing_start_step, eval=t_eval)
            timing_start_step = step
            wb.log(
                {
                    **{k: float(v) for k, v in train_metrics.items()},
                    **{f"train/{k}": v for k, v in timing_stats.items()},
                },
                step=step,
            )
        if step >= total_steps:
            logger.info("Reached total_steps=%d; stopping tokenizer training.", total_steps)
            break

    if last_checkpoint_step != step:
        save_checkpoint(step, force=True)
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    wb.finish()


if __name__ == "__main__":
    main()
