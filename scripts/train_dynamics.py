import itertools
import logging
import time
from pathlib import Path
from typing import Any

import grain.python as grain
import hydra
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from hydra.utils import instantiate, to_absolute_path
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from omegaconf import DictConfig, OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from visionary.common.checkpoint import (
    CheckpointManager,
    restore_model_export_single_device,
    restore_preprocessor_export,
    save_model_export,
)
from visionary.common.jax import fold_in_many, maybe_initialize_distributed
from visionary.common.train_state import DynamicsTrainState
from visionary.common.wandb import WandbLogger
from visionary.dataset import DynamicsBatch, DynamicsDataSource, RandomDynamicsCrop
from visionary.dynamics import DynamicsModel
from visionary.tokenizer_preprocessor import TokenizerPreprocessor

logger = logging.getLogger(__name__)


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
    if not enabled or fsdp_axis_size <= 1 or not shape:
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


def put_single_device_tree(tree):
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(jax.device_get(value), sharding)
        if hasattr(value, "shape")
        else value,
        tree,
    )


def put_global_batch(batch: DynamicsBatch, batch_sharding: NamedSharding) -> DynamicsBatch:
    return jax.tree_util.tree_map(
        lambda value: jax.make_array_from_process_local_data(batch_sharding, value),
        batch,
    )


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
    sequence_length: int,
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
            "train/sequence_length": sequence_length,
            **{f"train/{k}": v for k, v in stats.items()},
        },
        step=step,
    )
    return stats


def train_step(
    state: DynamicsTrainState,
    batch: DynamicsBatch,
    base_sample_key: jax.Array,
    global_step: jax.Array,
    bootstrap_rows: int,
):
    sample_key = fold_in_many(base_sample_key, global_step)

    def loss_fn(params):
        return state.apply_fn(
            params,
            batch,
            bootstrap_rows=bootstrap_rows,
            method=DynamicsModel.loss,
            rngs={"sample": sample_key},
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


def eval_step(
    state: DynamicsTrainState,
    batch: DynamicsBatch,
    base_sample_key: jax.Array,
    global_step: jax.Array,
    batch_index: jax.Array,
    bootstrap_rows: int,
):
    sample_key = fold_in_many(
        base_sample_key,
        global_step,
        batch_index,
    )
    _, metrics = state.apply_fn(
        state.params,
        batch,
        bootstrap_rows=bootstrap_rows,
        method=DynamicsModel.loss,
        rngs={"sample": sample_key},
    )
    return metrics


def log_video_eval(
    wb: WandbLogger,
    params,
    batch: DynamicsBatch,
    step: int,
    rollout_seed: int,
    video_cfg: DictConfig,
    output_dir: Path,
    tokenizer_variables,
    run_video_eval,
    context_tau_used: float,
) -> None:
    context_frames = int(video_cfg.context_frames)
    generated_frames = int(video_cfg.generated_frames)
    ground_truth_images, rollout_images = jax.device_get(
        run_video_eval(
            params,
            tokenizer_variables,
            batch["video"],
            batch["actions"],
            np.uint32(rollout_seed),
        )
    )
    ground_truth_images = np.asarray(ground_truth_images)
    rollout_images = np.asarray(rollout_images)
    ground_truth_images = np.clip(ground_truth_images, 0.0, 1.0)
    rollout_images = np.clip(rollout_images, 0.0, 1.0)

    generated_ground_truth = ground_truth_images[0, context_frames:]
    generated_rollout = rollout_images[0, context_frames:]
    psnr = np.asarray(
        [
            peak_signal_noise_ratio(target_frame, predicted_frame, data_range=1.0)
            for target_frame, predicted_frame in zip(
                generated_ground_truth, generated_rollout, strict=True
            )
        ],
        dtype=np.float32,
    )
    ssim = np.asarray(
        [
            structural_similarity(
                target_frame,
                predicted_frame,
                data_range=1.0,
                channel_axis=-1,
            )
            for target_frame, predicted_frame in zip(
                generated_ground_truth, generated_rollout, strict=True
            )
        ],
        dtype=np.float32,
    )
    metric_rows = [
        [context_frames + idx, idx + 1, float(psnr[idx]), float(ssim[idx])]
        for idx in range(generated_frames)
    ]

    video_path = output_dir / f"eval_side_by_side_{step}.mp4"
    ground_truth_video = np.clip(np.rint(ground_truth_images[0] * 255.0), 0, 255).astype(np.uint8)
    rollout_video = np.clip(np.rint(rollout_images[0] * 255.0), 0, 255).astype(np.uint8)
    separator = np.full((ground_truth_video.shape[1], 4, 3), 255, dtype=np.uint8)
    frames = [
        np.concatenate([gt_frame, separator, rollout_frame], axis=1)
        for gt_frame, rollout_frame in zip(ground_truth_video, rollout_video, strict=True)
    ]
    imageio.mimsave(video_path, frames, fps=int(video_cfg.fps))

    mean_psnr = float(np.mean(psnr))
    mean_ssim = float(np.mean(ssim))

    logger.info(
        "Video eval at step %d - mean PSNR %.3f, mean SSIM %.4f, context tau used %.4f",
        step,
        mean_psnr,
        mean_ssim,
        context_tau_used,
    )
    if wb.enabled:
        metric_table = wandb.Table(
            columns=["frame_index", "generated_frame", "psnr", "ssim"],
            data=metric_rows,
        )
        wb.log(
            {
                "eval/video": wandb.Video(
                    video_path.as_posix(),
                    caption=(
                        f"Left: decoded eval ground truth. Right: {context_frames} context "
                        f"frames followed by {generated_frames} generated frames."
                    ),
                ),
                "eval/video_frame_metrics": metric_table,
                "eval/video_psnr": wandb.plot.line(
                    metric_table,
                    "generated_frame",
                    "psnr",
                    title="Generated-frame PSNR",
                ),
                "eval/video_ssim": wandb.plot.line(
                    metric_table,
                    "generated_frame",
                    "ssim",
                    title="Generated-frame SSIM",
                ),
                "eval/video_mean_psnr": mean_psnr,
                "eval/video_mean_ssim": mean_ssim,
                "eval/video_context_tau_requested": float(video_cfg.context_tau),
                "eval/video_context_tau_used": context_tau_used,
                "eval/video_context_frames": context_frames,
                "eval/video_generated_frames": generated_frames,
                "eval/video_sample_steps": int(video_cfg.sample_steps),
            },
            step=step,
        )


@hydra.main(config_path="config", config_name="dynamics", version_base=None)
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
            "Multi-process dynamics training requires fsdp.enabled=true or auto. "
            "Replicated batch sharding would receive different per-process Grain batches."
        )
    if not fsdp_enabled and jax.device_count() > 1:
        logger.warning(
            "Multiple JAX devices are visible but fsdp.enabled is false; dynamics training "
            "will use replicated sharding."
        )

    wb = WandbLogger(cfg, enabled=bool(cfg.wandb.enabled) and is_primary_process)
    total_steps = int(cfg.total_steps)

    train_source = DynamicsDataSource(cfg.dataset.train_dir)
    eval_source = DynamicsDataSource(cfg.dataset.eval_dir)
    logger.info(
        "Loaded %d training sequences and %d eval sequences",
        len(train_source),
        len(eval_source),
    )
    effective_read_threads = max(cfg.dataset.worker_count, 1) * cfg.dataset.num_threads
    logger.info(
        "Data loader settings: worker_count=%d num_threads=%d "
        "prefetch_buffer_size=%d effective_read_threads=%d",
        cfg.dataset.worker_count,
        cfg.dataset.num_threads,
        cfg.dataset.prefetch_buffer_size,
        effective_read_threads,
    )

    train_sequence_lengths = (
        sorted({cfg.dataset.alternating_lengths.short, cfg.dataset.alternating_lengths.long})
        if cfg.dataset.alternating_lengths.enabled
        else [cfg.dataset.batch_length]
    )
    logger.info(
        "Training sequence lengths: %s; eval sequence length: %d",
        train_sequence_lengths,
        cfg.dataset.eval.batch_length,
    )
    batch_size_per_process = int(cfg.dataset.batch_size)
    if fsdp_enabled and batch_size_per_process % local_device_count != 0:
        raise ValueError(
            "cfg.dataset.batch_size is interpreted per process and must be divisible by "
            "jax.local_device_count() for FSDP input sharding; got "
            f"batch_size={batch_size_per_process} local_device_count={local_device_count}"
        )
    global_batch_size = batch_size_per_process * process_count
    batch_size_per_device = (
        batch_size_per_process // local_device_count if fsdp_enabled else batch_size_per_process
    )
    bootstrap_start_step = int(cfg.loss.bootstrap_start_step)
    target_bootstrap_ratio = float(cfg.loss.bootstrap_ratio)
    target_bootstrap_rows = min(
        max(int(round(target_bootstrap_ratio * global_batch_size)), 0),
        global_batch_size,
    )
    logger.info(
        "Batch layout: per_process=%d per_device=%d global=%d",
        batch_size_per_process,
        batch_size_per_device,
        global_batch_size,
    )
    logger.info(
        "Loss schedule: bootstrap_ratio=%.2f bootstrap_rows=%d bootstrap_start_step=%d",
        target_bootstrap_ratio,
        target_bootstrap_rows,
        bootstrap_start_step,
    )

    def bootstrap_rows_for_step(current_step: int) -> int:
        if current_step < bootstrap_start_step:
            return 0
        return target_bootstrap_rows

    def make_loader(
        source: DynamicsDataSource,
        sequence_length: int,
        shuffle: bool,
        drop_remainder: bool,
        seed: int,
    ) -> grain.DataLoader:
        sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.ShardByJaxProcess()
            if process_count > 1
            else grain.NoSharding(),
            shuffle=shuffle,
            seed=seed,
        )
        read_options = grain.ReadOptions(
            num_threads=cfg.dataset.num_threads,
            prefetch_buffer_size=cfg.dataset.prefetch_buffer_size,
        )
        return grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=[
                RandomDynamicsCrop(sequence_length),
                grain.Batch(
                    batch_size=batch_size_per_process,
                    drop_remainder=drop_remainder,
                ),
            ],
            worker_count=cfg.dataset.worker_count,
            read_options=read_options,
        )

    _t = time.monotonic()
    train_loaders = {
        sequence_length: make_loader(
            train_source,
            sequence_length=sequence_length,
            shuffle=True,
            drop_remainder=True,
            seed=cfg.seed + sequence_length,
        )
        for sequence_length in train_sequence_lengths
    }
    logger.info("Train DataLoader creation took %.1fs", time.monotonic() - _t)

    init_sequence_length = max(train_sequence_lengths)
    _t = time.monotonic()
    sample_batch = next(iter(train_loaders[init_sequence_length]))
    logger.info("First batch fetch took %.1fs", time.monotonic() - _t)

    overfit_batches = {init_sequence_length: sample_batch}
    if cfg.overfit_single_batch:
        logger.info("Overfit mode enabled: reusing one sampled batch per train sequence length.")
        for sequence_length in train_sequence_lengths:
            if sequence_length in overfit_batches:
                continue
            overfit_batches[sequence_length] = next(iter(train_loaders[sequence_length]))
        if cfg.dataset.eval.batch_length not in overfit_batches:
            overfit_batches[cfg.dataset.eval.batch_length] = next(
                iter(
                    make_loader(
                        eval_source,
                        sequence_length=cfg.dataset.eval.batch_length,
                        shuffle=False,
                        drop_remainder=fsdp_enabled,
                        seed=cfg.seed,
                    )
                )
            )
    else:
        _t = time.monotonic()
        eval_loader = make_loader(
            eval_source,
            sequence_length=cfg.dataset.eval.batch_length,
            shuffle=False,
            drop_remainder=fsdp_enabled,
            seed=cfg.seed,
        )
        logger.info("Eval DataLoader creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    key = jax.random.key(cfg.seed)
    init_key, init_sample_key, train_key, eval_key = jax.random.split(key, num=4)
    model = instantiate(cfg.dynamics)
    params = model.init(
        {"params": init_key, "sample": init_sample_key},
        sample_batch,
        bootstrap_rows=0,
        method=DynamicsModel.loss,
    )
    logger.info("Model init took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    optimizer = instantiate(cfg.optimizer)
    state = DynamicsTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
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
        log_sharding_summary(params, state_shardings.params, prefix="Dynamics params")
    logger.info("TrainState sharding took %.1fs", time.monotonic() - _t)

    batch_input_shardings = {"video": batch_sharding, "actions": batch_sharding}
    jit_train_step = jax.jit(
        train_step,
        in_shardings=(
            state_shardings,
            batch_input_shardings,
            metrics_sharding,
            metrics_sharding,
        ),
        out_shardings=(state_shardings, metrics_sharding),
        static_argnames=("bootstrap_rows",),
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
        out_shardings=metrics_sharding,
        static_argnames=("bootstrap_rows",),
    )
    train_key = put_replicated(train_key, mesh)
    eval_key = put_replicated(eval_key, mesh)

    video_cfg = cfg.video_eval
    video_eval_enabled = is_primary_process and process_count == 1
    video_output_dir = None
    run_video_eval = None
    context_tau_used = None
    tokenizer_variables = None
    if is_primary_process and process_count > 1:
        logger.warning(
            "Live dynamics video eval is disabled for multi-process FSDP training. "
            "Run video evaluation from a single-process export/checkpoint instead."
        )
    if video_eval_enabled:
        _t = time.monotonic()
        video_output_dir = Path(to_absolute_path(video_cfg.output_dir))
        video_output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer_checkpoint_dir = video_cfg.tokenizer.checkpoint_dir
        if "://" not in str(tokenizer_checkpoint_dir):
            tokenizer_checkpoint_dir = to_absolute_path(str(tokenizer_checkpoint_dir))
        tokenizer_cfg, tokenizer_variables = restore_model_export_single_device(
            tokenizer_checkpoint_dir,
            step=video_cfg.tokenizer.checkpoint_step,
        )
        preprocessor_cfg = restore_preprocessor_export(
            tokenizer_checkpoint_dir,
            step=video_cfg.tokenizer.checkpoint_step,
        )
        tokenizer = instantiate(tokenizer_cfg)
        preprocessor = TokenizerPreprocessor.from_config(preprocessor_cfg)

        requested_tau = float(video_cfg.context_tau)
        video_context_frames = int(video_cfg.context_frames)
        generated_frames = int(video_cfg.generated_frames)
        total_video_frames = video_context_frames + generated_frames
        context_step_count = 1 << (int(cfg.dynamics.max_step_size) - 1)
        context_tau_used = (
            min(max(round(requested_tau * context_step_count), 0), context_step_count - 1)
            / context_step_count
        )

        @jax.jit
        def run_video_eval(
            dynamics_params,
            tokenizer_variables,
            video_batch,
            action_batch,
            rollout_seed,
        ):
            video = jnp.asarray(video_batch[:1, :total_video_frames], dtype=jnp.float32)
            eval_action_dtype = (
                jnp.float32
                if str(cfg.dynamics.get("action_mode", "discrete")) == "continuous"
                else jnp.int32
            )
            actions = jnp.asarray(action_batch[:1, :total_video_frames], dtype=eval_action_dtype)
            rollout_video = jnp.zeros_like(video)
            rollout_video = rollout_video.at[:, :video_context_frames].set(
                video[:, :video_context_frames]
            )

            rollout_key = jax.random.key(rollout_seed)
            context_noise_key, sample_noise_key = jax.random.split(rollout_key)
            context_noise = jax.random.normal(context_noise_key, video.shape, dtype=jnp.float32)
            sample_noise = jax.random.normal(
                sample_noise_key,
                (video.shape[0], generated_frames, *video.shape[2:]),
                dtype=jnp.float32,
            )
            rollout_video = model.apply(
                dynamics_params,
                rollout_video,
                actions,
                context_noise,
                sample_noise,
                video_context_frames,
                context_tau=requested_tau,
                sample_steps=int(video_cfg.sample_steps),
                method=DynamicsModel.generate_rollout,
            )
            ground_truth_patches = tokenizer.apply(
                tokenizer_variables, video, method=type(tokenizer).decode
            )
            rollout_patches = tokenizer.apply(
                tokenizer_variables, rollout_video, method=type(tokenizer).decode
            )
            ground_truth_images = preprocessor.patches_to_images(ground_truth_patches).astype(
                jnp.float32
            )
            rollout_images = preprocessor.patches_to_images(rollout_patches).astype(jnp.float32)
            return ground_truth_images, rollout_images

        logger.info(
            "Video eval ready; context=%d generated=%d sample_steps=%d requested_tau=%.4f "
            "used_tau=%.4f",
            int(video_cfg.context_frames),
            int(video_cfg.generated_frames),
            int(video_cfg.sample_steps),
            requested_tau,
            context_tau_used,
        )
        logger.info("Video eval init took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    checkpoint_manager: CheckpointManager = instantiate(cfg.checkpoint.manager)
    if is_primary_process:
        checkpoint_manager.save_metadata(
            {
                "config": OmegaConf.to_container(cfg, resolve=False),
                "dynamics_config": OmegaConf.to_container(cfg, resolve=False),
            }
        )
    logger.info("CheckpointManager creation took %.1fs", time.monotonic() - _t)
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
            save_model_export(checkpoint_manager.directory, step, cfg.dynamics, state.params)

    train_iterators = {
        sequence_length: iter(loader) for sequence_length, loader in train_loaders.items()
    }

    def iterator_items():
        if cfg.overfit_single_batch:
            return None
        return {
            f"train_iterator_{sequence_length}": train_iterator
            for sequence_length, train_iterator in train_iterators.items()
        }

    def sequence_length_for_step(current_step: int) -> int:
        if cfg.dataset.alternating_lengths.enabled and (
            current_step >= total_steps - cfg.dataset.alternating_lengths.final_long_only_steps
            or (current_step + 1) % cfg.dataset.alternating_lengths.long_every == 0
        ):
            return cfg.dataset.alternating_lengths.long
        if cfg.dataset.alternating_lengths.enabled:
            return cfg.dataset.alternating_lengths.short
        return cfg.dataset.batch_length

    resume_spec = cfg.checkpoint.resume_step
    resume_step = None
    if resume_spec is not None:
        if isinstance(resume_spec, str) and resume_spec.strip().lower() == "latest":
            resume_step = checkpoint_manager.latest_step()
            if resume_step is None:
                logger.info(
                    "No dynamics checkpoint found in %s; starting fresh.",
                    checkpoint_manager.directory,
                )
        else:
            resume_step = int(resume_spec)

    if resume_step is not None:
        state = checkpoint_manager.restore(
            target=state,
            step=resume_step,
            extra_items=iterator_items(),
        )
        logger.info("Resumed dynamics training from step %d", int(state.step))

    step = int(jax.device_get(state.step))
    logger.info("Dynamics training target step: %d", total_steps)

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
        "Asynchronous timing mode enabled for dynamics training; timing logs are averaged "
        "over each logging window."
    )
    while True:
        current_step = step
        step_start = time.monotonic()

        sequence_length = sequence_length_for_step(current_step)

        if cfg.overfit_single_batch:
            batch = overfit_batches[sequence_length]
        else:
            try:
                batch = next(train_iterators[sequence_length])
            except StopIteration:
                train_iterators[sequence_length] = iter(train_loaders[sequence_length])
                batch = next(train_iterators[sequence_length])
        data_done = time.monotonic()

        batch = put_global_batch(batch, batch_sharding)
        transfer_done = time.monotonic()

        bootstrap_rows = bootstrap_rows_for_step(current_step)
        state, metrics = jit_train_step(
            state,
            batch,
            train_key,
            jnp.asarray(current_step, dtype=jnp.int32),
            bootstrap_rows,
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
                sequence_length=sequence_length,
            )
            timing_start_step = step
            timing_data_time = timing_transfer_time = timing_dispatch_time = 0.0

        t_eval = 0.0
        if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
            t_eval_start = time.monotonic()
            totals: dict[str, float] = {}
            if cfg.overfit_single_batch:
                eval_batches = [overfit_batches[cfg.dataset.eval.batch_length]]
            else:
                eval_batches = list(
                    itertools.islice(iter(eval_loader), cfg.dataset.eval.max_batches)
                )
            if fsdp_enabled:
                global_eval_batch_counts = np.asarray(
                    multihost_utils.process_allgather(
                        np.asarray(len(eval_batches), dtype=np.int32)
                    )
                )
                eval_batches = eval_batches[: int(np.min(global_eval_batch_counts))]

            num_batches = 0
            for batch_idx, eval_batch in enumerate(eval_batches):
                batch_metrics = jax.device_get(
                    jit_eval_step(
                        state,
                        put_global_batch(eval_batch, batch_sharding),
                        eval_key,
                        jnp.asarray(step, dtype=jnp.int32),
                        jnp.asarray(batch_idx, dtype=jnp.int32),
                        bootstrap_rows,
                    )
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
            if video_eval_enabled and num_batches > 0:
                log_video_eval(
                    wb,
                    put_single_device_tree(state.params),
                    eval_batches[0],
                    step=step,
                    rollout_seed=make_host_seed(cfg.seed, step, num_batches),
                    video_cfg=video_cfg,
                    output_dir=video_output_dir,
                    tokenizer_variables=tokenizer_variables,
                    run_video_eval=run_video_eval,
                    context_tau_used=context_tau_used,
                )
            t_eval = time.monotonic() - t_eval_start
            logger.info("Eval at step %d - %d batches in %.3fs", step, num_batches, t_eval)

        if checkpoint_manager.should_save(step):
            save_checkpoint(step)

        if timing_stats is not None:
            logger.info(
                "Step %d - seq: %d, sps: %.2f, data: %.3fs, transfer: %.3fs, "
                "compute: %.3fs, wall: %.3fs, eval: %.3fs",
                step,
                sequence_length,
                timing_stats["sps"],
                timing_stats["data_time"],
                timing_stats["transfer_time"],
                timing_stats["compute_time"],
                timing_stats["wall_time"],
                t_eval,
            )
        if step >= total_steps:
            logger.info("Reached total_steps=%d; stopping dynamics training.", total_steps)
            break

    multihost_utils.sync_global_devices("dynamics_train_complete")
    if step >= total_steps and last_checkpoint_step != step:
        save_checkpoint(step, force=True)
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    wb.finish()


if __name__ == "__main__":
    main()
