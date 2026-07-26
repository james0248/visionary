import itertools
import logging
import time
from functools import lru_cache
from typing import Any

import grain.python as grain
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra.utils import instantiate
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from omegaconf import DictConfig, OmegaConf

from visionary.common.checkpoint import (
    CheckpointManager,
    save_model_export,
    save_preprocessor_export,
)
from visionary.common.jax import fold_in_many, maybe_initialize_distributed
from visionary.common.train_state import TokenizerTrainState
from visionary.common.wandb import WandbLogger
from visionary.dataset import VideoDataset
from visionary.lpips import LPIPS
from visionary.tokenizer import Tokenizer
from visionary.tokenizer_preprocessor import TokenizerPreprocessor

logger = logging.getLogger(__name__)


LPIPS_PRETRAINED_NETWORK = "alexnet"
LOSS_RMS_DECAY = 0.99
LOSS_RMS_EPS = 1e-8
DATA_AXIS = "data"
FSDP_AXIS = "fsdp"


def cfg_select(cfg: DictConfig, path: str, default: Any) -> Any:
    value = OmegaConf.select(cfg, path, default=default)
    return default if value is None else value


def parse_auto_bool(value: Any, auto_value: bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value == "auto":
        return auto_value
    if value in {"1", "true", "yes", "on", "enabled"}:
        return True
    if value in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(f"Expected a boolean or 'auto', got {value!r}")


def build_fsdp_mesh(cfg: DictConfig) -> tuple[Mesh, bool, int]:
    device_count = jax.device_count()
    auto_enabled = device_count > 1 or jax.process_count() > 1
    fsdp_enabled = parse_auto_bool(
        cfg_select(cfg, "fsdp.enabled", "auto"),
        auto_value=auto_enabled,
    )
    data_axis_size = int(cfg_select(cfg, "fsdp.data_axis_size", 1))
    if data_axis_size < 1:
        raise ValueError(f"fsdp.data_axis_size must be >= 1, got {data_axis_size}")
    if device_count % data_axis_size != 0:
        raise ValueError(
            "fsdp.data_axis_size must divide jax.device_count(); got "
            f"{data_axis_size=} and device_count={device_count}"
        )

    fsdp_axis_size = device_count // data_axis_size
    devices = mesh_utils.create_device_mesh((data_axis_size, fsdp_axis_size))
    mesh = Mesh(devices, (DATA_AXIS, FSDP_AXIS))
    if fsdp_enabled and fsdp_axis_size < 1:
        raise ValueError("FSDP requested but no devices are available.")
    return mesh, fsdp_enabled, fsdp_axis_size


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


def batch_partition_spec() -> P:
    return P((DATA_AXIS, FSDP_AXIS))


def choose_fsdp_partition_spec(
    value: Any,
    enabled: bool,
    fsdp_axis_size: int,
) -> P:
    if not hasattr(value, "shape"):
        if not np.isscalar(value):
            return P()
        dtype = np.int32 if isinstance(value, (int, np.integer)) else None
        value = np.asarray(value, dtype=dtype)
    shape = tuple(int(dim) for dim in value.shape)
    if (
        not enabled
        or fsdp_axis_size <= 1
        or not shape
    ):
        return P()

    ranked_dims = sorted(range(len(shape)), key=lambda axis: shape[axis], reverse=True)
    for axis in ranked_dims:
        if shape[axis] >= fsdp_axis_size and shape[axis] % fsdp_axis_size == 0:
            spec = [None] * len(shape)
            spec[axis] = FSDP_AXIS
            return P(*spec)
    return P()


def make_array_shardings(
    tree,
    mesh: Mesh,
    fsdp_enabled: bool,
    fsdp_axis_size: int,
):
    return jax.tree_util.tree_map(
        lambda value: NamedSharding(
            mesh,
            choose_fsdp_partition_spec(
                value,
                enabled=fsdp_enabled,
                fsdp_axis_size=fsdp_axis_size,
            ),
        ),
        tree,
    )


def make_global_array_from_host(value, sharding: NamedSharding):
    if not hasattr(value, "shape") and not np.isscalar(value):
        return value
    dtype = np.int32 if isinstance(value, (int, np.integer)) else None
    value = np.asarray(jax.device_get(value), dtype=dtype)
    return jax.make_array_from_callback(
        value.shape,
        sharding,
        lambda index: value[index],
        dtype=value.dtype,
    )


def put_replicated(value, mesh: Mesh):
    return jax.device_put(value, replicated_sharding(mesh))


def put_global_batch(batch: VideoDataset, batch_sharding: NamedSharding) -> VideoDataset:
    return jax.tree_util.tree_map(
        lambda value: jax.make_array_from_process_local_data(batch_sharding, value),
        batch,
    )


def host_local_batch(batch, mesh: Mesh, pspec: P):
    return multihost_utils.global_array_to_host_local_array(batch, mesh, pspec)


def log_sharding_summary(tree, shardings, prefix: str, max_entries: int = 12) -> None:
    leaves = []

    def visit(path, value, sharding):
        if not hasattr(value, "shape"):
            return
        spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
        leaves.append(
            (
                spec != P(),
                int(np.prod(value.shape)) if value.shape else 1,
                jax.tree_util.keystr(path, simple=True, separator="/"),
                tuple(int(dim) for dim in value.shape),
                spec,
            )
        )

    jax.tree_util.tree_map_with_path(visit, tree, shardings)
    num_leaves = len(leaves)
    num_sharded = sum(1 for sharded, *_ in leaves if sharded)
    total_values = sum(size for _, size, *_ in leaves)
    sharded_values = sum(size for sharded, size, *_ in leaves if sharded)
    logger.info(
        "%s sharding: %d/%d leaves sharded, %.1f%% of values in sharded leaves",
        prefix,
        num_sharded,
        num_leaves,
        100.0 * sharded_values / max(total_values, 1),
    )
    for sharded, size, path, shape, spec in sorted(leaves, reverse=True)[:max_entries]:
        logger.info(
            "%s sharding leaf: sharded=%s values=%d path=%s shape=%s spec=%s",
            prefix,
            sharded,
            size,
            path,
            shape,
            spec,
        )


def make_host_seed(*values: int) -> int:
    seed_sequence = np.random.SeedSequence([int(value) for value in values])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def log_train_timing(
    wb: WandbLogger,
    step: int,
    start_step: int,
    metrics,
    data_time: float,
    transfer_time: float,
    dispatch_time: float,
) -> dict[str, float]:
    window_steps = step - start_step
    avg_steps = max(window_steps, 1)
    sync_start = time.monotonic()
    train_metrics = jax.device_get(metrics)
    sync_time = time.monotonic() - sync_start

    total_time = data_time + transfer_time + dispatch_time + sync_time
    stats = {
        "sps": window_steps / max(total_time, 1e-8),
        "data_time": data_time / avg_steps,
        "transfer_time": transfer_time / avg_steps,
        "compute_time": (dispatch_time + sync_time) / avg_steps,
        "wall_time": total_time / avg_steps,
    }
    wb.log(
        {
            **{k: float(v) for k, v in train_metrics.items()},
            **{f"train/{k}": v for k, v in stats.items()},
        },
        step=step,
    )
    return stats


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
    batch: VideoDataset,
    reconstructed: jax.Array,
    mask: jax.Array,
    lpips_weight: float,
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
        lpips_loss = compute_lpips_loss(original, reconstructed, preprocessor)
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
        "mask_ratio": jnp.mean(mask),
    }
    return loss, metrics


def train_step(
    state: TokenizerTrainState,
    batch: VideoDataset,
    base_sample_key: jax.Array,
    global_step: int,
    lpips_weight: float,
    preprocessor: TokenizerPreprocessor,
):
    sample_key = fold_in_many(base_sample_key, global_step)

    def loss_fn(params):
        reconstructed, mask = state.apply_fn(
            params,
            batch,
            method=Tokenizer.reconstruct,
            rngs={"sample": sample_key},
        )
        return compute_loss_metrics(
            state,
            batch,
            reconstructed,
            mask,
            lpips_weight,
            preprocessor,
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = update_loss_ema(
        state,
        metrics["mse_loss"],
        metrics["lpips_loss"],
    )
    return state, metrics


def eval_step(
    state: TokenizerTrainState,
    batch: VideoDataset,
    base_sample_key: jax.Array,
    global_step: int,
    batch_index: int,
    lpips_weight: float,
    preprocessor: TokenizerPreprocessor,
):
    sample_key = fold_in_many(base_sample_key, global_step, batch_index)
    model_key, frame_key = jax.random.split(sample_key)
    reconstructed, mask = state.apply_fn(
        state.params,
        batch,
        mask_prob=0.1,
        independent=jnp.zeros((batch["video"].shape[0],), dtype=bool),
        method=Tokenizer.reconstruct,
        rngs={"sample": model_key},
    )
    _, metrics = compute_loss_metrics(
        state,
        batch,
        reconstructed,
        mask,
        lpips_weight,
        preprocessor,
    )
    sampled_frames = sample_sequence_frames(
        batch,
        reconstructed,
        mask,
        preprocessor,
        frame_key,
    )
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


@hydra.main(config_path="config", config_name="tokenizer", version_base=None)
def main(cfg: DictConfig):
    maybe_initialize_distributed(logger=logger)

    process_index = jax.process_index()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()
    is_primary_process = process_index == 0
    mesh, fsdp_enabled, fsdp_axis_size = build_fsdp_mesh(cfg)
    log_sharding = bool(cfg_select(cfg, "fsdp.log_sharding", True))
    batch_pspec = batch_partition_spec() if fsdp_enabled else P()
    batch_sharding = NamedSharding(mesh, batch_pspec)
    metrics_sharding = replicated_sharding(mesh)

    logger.info(
        "JAX backend: %s process=%d/%d local_devices=%d global_devices=%d devices=%s",
        jax.default_backend(),
        process_index,
        process_count,
        local_device_count,
        jax.device_count(),
        jax.local_devices(),
    )
    logger.info(
        "FSDP mesh: enabled=%s mesh_shape=%s data_axis=%d fsdp_axis=%d batch_pspec=%s",
        fsdp_enabled,
        mesh.devices.shape,
        mesh.shape[DATA_AXIS],
        mesh.shape[FSDP_AXIS],
        batch_pspec,
    )
    if not fsdp_enabled and process_count > 1:
        raise ValueError(
            "Multi-process tokenizer training requires fsdp.enabled=true or auto. "
            "Replicated batch sharding would receive different per-process Grain batches."
        )
    if not fsdp_enabled and jax.device_count() > 1:
        logger.warning(
            "Multiple JAX devices are visible but fsdp.enabled is false; tokenizer training "
            "will use replicated sharding."
        )

    wb = WandbLogger(cfg, enabled=bool(cfg.wandb.enabled) and is_primary_process)
    total_steps = int(cfg.total_steps)

    train_source = instantiate(cfg.dataset.source, data_dir=cfg.dataset.train_dir)
    eval_source = instantiate(cfg.dataset.source, data_dir=cfg.dataset.eval_dir)
    logger.info(
        "Loaded %d training videos and %d eval videos",
        len(train_source),
        len(eval_source),
    )
    batch_size_per_process = int(cfg.dataset.batch_size)
    if fsdp_enabled and batch_size_per_process % local_device_count != 0:
        raise ValueError(
            "cfg.dataset.batch_size is interpreted per process and must be divisible by "
            "jax.local_device_count() for FSDP input sharding; got "
            f"batch_size={batch_size_per_process} local_device_count={local_device_count}"
        )
    logger.info(
        "Batch layout: per_process=%d per_device=%d global=%d",
        batch_size_per_process,
        batch_size_per_process // local_device_count if fsdp_enabled else batch_size_per_process,
        batch_size_per_process * process_count,
    )
    worker_count = int(cfg.dataset.worker_count)
    num_threads = int(cfg.dataset.num_threads)
    prefetch_buffer_size = int(cfg.dataset.prefetch_buffer_size)
    effective_read_threads = max(worker_count, 1) * num_threads
    logger.info(
        "Data loader settings: worker_count=%d num_threads=%d "
        "prefetch_buffer_size=%d effective_read_threads=%d",
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
    clip_transform = instantiate(
        cfg.dataset.clip_transform, frame_length=cfg.dataset.frame_length
    )
    preprocess_transform = preprocessor.as_grain_transform()

    def make_loader(source, shuffle: bool, drop_remainder: bool, seed: int):
        sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.ShardByJaxProcess()
            if process_count > 1
            else grain.NoSharding(),
            shuffle=shuffle,
            seed=seed,
        )
        read_options = grain.ReadOptions(
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size,
        )
        return grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=[
                clip_transform,
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
    )
    logger.info("Train DataLoader creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    eval_loader = make_loader(
        eval_source,
        shuffle=False,
        drop_remainder=fsdp_enabled,
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
    state_shardings = make_array_shardings(
        state,
        mesh=mesh,
        fsdp_enabled=fsdp_enabled,
        fsdp_axis_size=fsdp_axis_size,
    )
    state = jax.tree_util.tree_map(make_global_array_from_host, state, state_shardings)
    if log_sharding:
        log_sharding_summary(params, state_shardings.params, prefix="Tokenizer params")
    logger.info("TrainState sharding took %.1fs", time.monotonic() - _t)

    batch_input_shardings = {"video": batch_sharding}
    sampled_frame_sharding = NamedSharding(mesh, batch_pspec)
    jit_train_step = jax.jit(
        train_step,
        in_shardings=(
            state_shardings,
            batch_input_shardings,
            metrics_sharding,
            metrics_sharding,
        ),
        out_shardings=(state_shardings, metrics_sharding),
        static_argnames=("lpips_weight", "preprocessor"),
        donate_argnums=(0,),
    )
    jit_eval_step = jax.jit(
        eval_step,
        in_shardings=(
            state_shardings,
            batch_input_shardings,
            metrics_sharding,
            metrics_sharding,
            metrics_sharding,
        ),
        out_shardings=(
            metrics_sharding,
            (sampled_frame_sharding, sampled_frame_sharding, sampled_frame_sharding),
        ),
        static_argnames=("lpips_weight", "preprocessor"),
    )
    train_key = put_replicated(train_key, mesh)
    eval_key = put_replicated(eval_key, mesh)

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
        export_model = should_export_model(step, force=force)
        if export_model:
            save_model_export(checkpoint_manager.directory, step, cfg.tokenizer, state.params)
        if is_primary_process and export_model:
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

    timing_start_step = step
    timing_data_time = timing_transfer_time = timing_dispatch_time = 0.0
    logger.info(
        "Asynchronous timing mode enabled for tokenizer training; timing logs are averaged "
        "over each logging window."
    )
    while True:
        step_start = time.monotonic()
        current_step = step
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)
        data_done = time.monotonic()

        batch = put_global_batch(batch, batch_sharding)
        transfer_done = time.monotonic()

        state, metrics = jit_train_step(
            state,
            batch,
            train_key,
            jnp.asarray(current_step, dtype=jnp.int32),
            float(cfg.lpips_weight),
            preprocessor,
        )
        train_dispatched = time.monotonic()

        step = current_step + 1
        timing_data_time += data_done - step_start
        timing_transfer_time += transfer_done - data_done
        timing_dispatch_time += train_dispatched - transfer_done

        timing_stats = None
        if step % cfg.log_interval == 0:
            timing_stats = log_train_timing(
                wb,
                step=step,
                start_step=timing_start_step,
                metrics=metrics,
                data_time=timing_data_time,
                transfer_time=timing_transfer_time,
                dispatch_time=timing_dispatch_time,
            )
            timing_start_step = step
            timing_data_time = timing_transfer_time = timing_dispatch_time = 0.0

        t_eval = 0.0
        if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
            t_eval_start = time.monotonic()
            totals: dict[str, float] = {}
            num_batches = 0
            eval_batches = list(itertools.islice(iter(eval_loader), cfg.dataset.eval.max_batches))
            if fsdp_enabled:
                global_eval_batch_counts = np.asarray(
                    multihost_utils.process_allgather(
                        np.asarray(len(eval_batches), dtype=np.int32)
                    )
                )
                eval_batches = eval_batches[: int(np.min(global_eval_batch_counts))]
            vis_original_batches = []
            vis_reconstruction_batches = []
            vis_masked_batches = []
            for batch_idx, eval_batch in enumerate(eval_batches):
                batch_metrics, sampled_frames = jit_eval_step(
                    state,
                    put_global_batch(eval_batch, batch_sharding),
                    eval_key,
                    jnp.asarray(step, dtype=jnp.int32),
                    jnp.asarray(batch_idx, dtype=jnp.int32),
                    float(cfg.lpips_weight),
                    preprocessor,
                )
                sampled_frames = host_local_batch(sampled_frames, mesh, batch_pspec)
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

        if timing_stats is not None:
            logger.info(
                "Step %d - sps: %.2f, data: %.3fs, transfer: %.3fs, compute: %.3fs, "
                "wall: %.3fs, eval: %.3fs",
                step,
                timing_stats["sps"],
                timing_stats["data_time"],
                timing_stats["transfer_time"],
                timing_stats["compute_time"],
                timing_stats["wall_time"],
                t_eval,
            )
        if step >= total_steps:
            logger.info("Reached total_steps=%d; stopping tokenizer training.", total_steps)
            break

    if step >= total_steps and last_checkpoint_step != step:
        save_checkpoint(step, force=True)
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    wb.finish()


if __name__ == "__main__":
    main()
