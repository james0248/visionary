import itertools
import logging
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import grain.python as grain
import flax.jax_utils as flax_jax_utils
import hydra
import imageio
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from hydra.utils import instantiate, to_absolute_path
from jax.experimental import multihost_utils
from omegaconf import DictConfig, OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from visionary.common.checkpoint import (
    CheckpointManager,
    restore_model_export_single_device,
)
from visionary.common.jax import fold_in_many, maybe_initialize_distributed
from visionary.common.train_state import DynamicsTrainState
from visionary.common.wandb import WandbLogger
from visionary.dataset import DynamicsBatch, DynamicsDataSource, RandomDynamicsCrop
from visionary.dynamics import DynamicsModel

logger = logging.getLogger(__name__)


PMAP_AXIS_NAME = "data"


def make_host_seed(*values: int) -> int:
    seed_sequence = np.random.SeedSequence([int(value) for value in values])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def block_until_ready(tree):
    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        tree,
    )


@partial(
    jax.pmap,
    axis_name=PMAP_AXIS_NAME,
    in_axes=(0, 0, None, None, None),
    static_broadcasted_argnums=(4,),
    donate_argnums=(0,),
)
def train_step(
    state: DynamicsTrainState,
    batch: DynamicsBatch,
    base_sample_key: jax.Array,
    global_step: int,
    bootstrap_rows: int,
):
    sample_key = fold_in_many(base_sample_key, global_step, lax.axis_index(PMAP_AXIS_NAME))

    def loss_fn(params):
        return state.apply_fn(
            params,
            batch,
            bootstrap_rows=bootstrap_rows,
            method=DynamicsModel.loss,
            rngs={"sample": sample_key},
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = lax.pmean(grads, axis_name=PMAP_AXIS_NAME)
    metrics = lax.pmean(metrics, axis_name=PMAP_AXIS_NAME)
    state = state.apply_gradients(grads=grads)
    return state, metrics


@partial(
    jax.pmap,
    axis_name=PMAP_AXIS_NAME,
    in_axes=(0, 0, None, None, None, None),
    static_broadcasted_argnums=(5,),
)
def eval_step(
    state: DynamicsTrainState,
    batch: DynamicsBatch,
    base_sample_key: jax.Array,
    global_step: int,
    batch_index: int,
    bootstrap_rows: int,
):
    sample_key = fold_in_many(
        base_sample_key,
        global_step,
        batch_index,
        lax.axis_index(PMAP_AXIS_NAME),
    )
    _, metrics = state.apply_fn(
        state.params,
        batch,
        bootstrap_rows=bootstrap_rows,
        method=DynamicsModel.loss,
        rngs={"sample": sample_key},
    )
    return lax.pmean(metrics, axis_name=PMAP_AXIS_NAME)


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
            rollout_seed,
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


@hydra.main(config_path="config", config_name="breakout_dynamics", version_base=None)
def main(cfg: DictConfig):
    maybe_initialize_distributed(logger=logger)

    process_index = jax.process_index()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()
    local_devices = jax.local_devices()
    distributed = process_count > 1 or local_device_count > 1
    is_primary_process = process_index == 0
    logger.info(
        "JAX backend: %s process=%d/%d local_devices=%d global_devices=%d devices=%s",
        jax.default_backend(),
        process_index,
        process_count,
        local_device_count,
        jax.device_count(),
        jax.local_devices(),
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
    if batch_size_per_process % local_device_count != 0:
        raise ValueError(
            "cfg.dataset.batch_size is interpreted per process and must be divisible by "
            f"jax.local_device_count(); got {cfg.dataset.batch_size=} {local_device_count=}"
        )
    batch_size_per_device = batch_size_per_process // local_device_count
    bootstrap_start_step = int(cfg.loss.bootstrap_start_step)
    target_bootstrap_ratio = float(cfg.loss.bootstrap_ratio)
    target_bootstrap_rows = min(
        max(int(round(target_bootstrap_ratio * batch_size_per_device)), 0),
        batch_size_per_device,
    )
    logger.info(
        "Batch layout: per_process=%d per_device=%d global=%d",
        batch_size_per_process,
        batch_size_per_device,
        batch_size_per_process * process_count,
    )
    logger.info(
        "Loss schedule: bootstrap_ratio=%.2f bootstrap_rows=%d bootstrap_start_step=%d",
        target_bootstrap_ratio,
        target_bootstrap_rows,
        bootstrap_start_step,
    )

    def reshape_batch(batch: DynamicsBatch) -> DynamicsBatch:
        return {
            key: np.reshape(value, (local_device_count, batch_size_per_device, *value.shape[1:]))
            for key, value in batch.items()
        }

    def bootstrap_rows_for_step(current_step: int) -> int:
        if current_step < bootstrap_start_step:
            return 0
        return target_bootstrap_rows

    def unreplicate(tree):
        return flax_jax_utils.unreplicate(tree)

    def to_host(tree):
        return jax.device_get(unreplicate(tree))

    def make_loader(
        source: DynamicsDataSource,
        sequence_length: int,
        shuffle: bool,
        drop_remainder: bool,
        seed: int,
    ) -> grain.DataLoader:
        sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.ShardByJaxProcess(),
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
                        drop_remainder=distributed,
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
            drop_remainder=distributed,
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
    optimizer = optax.adam(cfg.learning_rate)
    state = DynamicsTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
    logger.info("TrainState creation took %.1fs", time.monotonic() - _t)

    video_cfg = cfg.video_eval
    video_output_dir = None
    run_video_eval = None
    context_tau_used = None
    if is_primary_process:
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
        tokenizer = instantiate(tokenizer_cfg)

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
            actions = jnp.asarray(action_batch[:1, :total_video_frames], dtype=jnp.int32)
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
            ground_truth_images = tokenizer.apply(
                tokenizer_variables, video, method=type(tokenizer).decode
            )
            rollout_images = tokenizer.apply(
                tokenizer_variables, rollout_video, method=type(tokenizer).decode
            )
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
    checkpoint_manager.save_metadata({"dynamics_config": OmegaConf.to_container(cfg, resolve=False)})
    logger.info("CheckpointManager creation took %.1fs", time.monotonic() - _t)
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
                logger.info("No dynamics checkpoint found in %s; starting fresh.", checkpoint_manager.directory)
        else:
            resume_step = int(resume_spec)

    if resume_step is not None:
        state = checkpoint_manager.restore(target=state, step=resume_step, extra_items=iterator_items())
        logger.info("Resumed dynamics training from step %d", int(state.step))

    state = flax_jax_utils.replicate(state, devices=local_devices)
    step = int(jax.device_get(unreplicate(state.step)))
    logger.info("Dynamics training target step: %d", total_steps)

    profiling_cfg = cfg.get("profiling") or {}
    profiler_enabled = bool(profiling_cfg.get("enabled", False))
    profiler_active = False
    profiler_start_step = 0
    profiler_stop_step = 0
    profiler_log_dir = None
    profiler_create_perfetto_trace = False
    if profiler_enabled:
        configured_profiler_start_step = int(profiling_cfg.get("start_step", step))
        profiler_start_step = max(configured_profiler_start_step, step)
        profiler_num_steps = max(int(profiling_cfg.get("num_steps", cfg.log_interval)), 1)
        profiler_stop_step = min(profiler_start_step + profiler_num_steps, total_steps)
        if profiler_stop_step <= profiler_start_step:
            profiler_enabled = False
            logger.info(
                "Profiling window [%d, %d) is empty at current step %d; skipping profiler.",
                profiler_start_step,
                profiler_stop_step,
                step,
            )
        else:
            profiler_root_dir = Path(
                to_absolute_path(
                    str(
                        profiling_cfg.get(
                            "log_dir", "artifacts/dreamer4_dynamics_profile"
                        )
                    )
                )
            )
            profiler_log_dir = profiler_root_dir / f"process_{process_index}"
            profiler_log_dir.mkdir(parents=True, exist_ok=True)
            profiler_create_perfetto_trace = bool(
                profiling_cfg.get("create_perfetto_trace", True)
            )
            if profiler_start_step != configured_profiler_start_step:
                logger.info(
                    "Adjusted profiler start step from %d to %d because current step is %d.",
                    configured_profiler_start_step,
                    profiler_start_step,
                    step,
                )
            logger.info(
                "Profiler armed for steps [%d, %d) -> %s (create_perfetto_trace=%s)",
                profiler_start_step,
                profiler_stop_step,
                profiler_log_dir,
                profiler_create_perfetto_trace,
            )

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

    train_window_start_step = step
    train_window_data_time = 0.0
    train_window_reshape_time = 0.0
    train_window_dispatch_time = 0.0
    logger.info(
        "Asynchronous timing mode enabled for dynamics training; timing logs are averaged "
        "over each logging window."
    )
    while True:
        current_step = step
        step_start = time.monotonic()

        if profiler_enabled and not profiler_active and current_step == profiler_start_step:
            jax.profiler.start_trace(
                profiler_log_dir.as_posix(),
                create_perfetto_trace=profiler_create_perfetto_trace,
            )
            profiler_active = True
            logger.info(
                "Started JAX profiler trace at step %d -> %s",
                current_step,
                profiler_log_dir,
            )

        step_annotation = (
            jax.profiler.StepTraceAnnotation("train", step_num=current_step)
            if profiler_active
            else nullcontext()
        )
        with step_annotation:
            sequence_length = sequence_length_for_step(current_step)

            if cfg.overfit_single_batch:
                batch = overfit_batches[sequence_length]
            else:
                try:
                    batch = next(train_iterators[sequence_length])
                except StopIteration:
                    train_iterators[sequence_length] = iter(train_loaders[sequence_length])
                    batch = next(train_iterators[sequence_length])
            t1 = time.monotonic()

            batch = reshape_batch(batch)
            t2 = time.monotonic()

            bootstrap_rows = bootstrap_rows_for_step(current_step)
            state, metrics = train_step(
                state,
                batch,
                train_key,
                current_step,
                bootstrap_rows,
            )
            t3 = time.monotonic()

        step = current_step + 1
        data_time = t1 - step_start
        reshape_time = t2 - t1
        dispatch_time = t3 - t2
        train_window_data_time += data_time
        train_window_reshape_time += reshape_time
        train_window_dispatch_time += dispatch_time

        should_log_train = step % cfg.log_interval == 0
        train_sps = 0.0
        avg_data_time = 0.0
        avg_reshape_time = 0.0
        avg_compute_time = 0.0
        avg_wall_time = 0.0
        if should_log_train:
            window_steps = step - train_window_start_step
            sync_start = time.monotonic()
            train_metrics = to_host(metrics)
            sync_time = time.monotonic() - sync_start
            total_window_time = (
                train_window_data_time
                + train_window_reshape_time
                + train_window_dispatch_time
                + sync_time
            )
            train_sps = window_steps / max(total_window_time, 1e-8)
            avg_data_time = train_window_data_time / max(window_steps, 1)
            avg_reshape_time = train_window_reshape_time / max(window_steps, 1)
            avg_compute_time = (train_window_dispatch_time + sync_time) / max(window_steps, 1)
            avg_wall_time = total_window_time / max(window_steps, 1)
            wb.log(
                {
                    **{k: float(v) for k, v in train_metrics.items()},
                    "train/sequence_length": sequence_length,
                    "train/sps": float(train_sps),
                    "train/data_time": float(avg_data_time),
                    "train/reshape_time": float(avg_reshape_time),
                    "train/compute_time": float(avg_compute_time),
                    "train/wall_time": float(avg_wall_time),
                },
                step=step,
            )
            train_window_start_step = step
            train_window_data_time = 0.0
            train_window_reshape_time = 0.0
            train_window_dispatch_time = 0.0

        if profiler_active and step >= profiler_stop_step:
            profiler_sync_time = 0.0
            if not should_log_train:
                profiler_sync_start = time.monotonic()
                block_until_ready(metrics)
                profiler_sync_time = time.monotonic() - profiler_sync_start
            jax.profiler.stop_trace()
            profiler_active = False
            logger.info(
                "Stopped JAX profiler trace at step %d (final sync %.3fs) -> %s",
                step,
                profiler_sync_time,
                profiler_log_dir,
            )

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
            global_eval_batch_counts = np.asarray(
                multihost_utils.process_allgather(np.asarray(len(eval_batches), dtype=np.int32))
            )
            eval_batches = eval_batches[: int(np.min(global_eval_batch_counts))]

            num_batches = 0
            for batch_idx, eval_batch in enumerate(eval_batches):
                batch_metrics = to_host(
                    eval_step(
                        state,
                        reshape_batch(eval_batch),
                        eval_key,
                        step,
                        batch_idx,
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
            if is_primary_process and num_batches > 0:
                log_video_eval(
                    wb,
                    unreplicate(state.params),
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
            checkpoint_manager.save(step=step, state=to_host(state), extra_items=iterator_items())

        if should_log_train:
            logger.info(
                "Step %d - seq: %d, sps: %.2f, data: %.3fs, reshape: %.3fs, "
                "compute: %.3fs, wall: %.3fs, eval: %.3fs",
                step,
                sequence_length,
                train_sps,
                avg_data_time,
                avg_reshape_time,
                avg_compute_time,
                avg_wall_time,
                t_eval,
            )
        if step >= total_steps:
            logger.info("Reached total_steps=%d; stopping dynamics training.", total_steps)
            break

    if profiler_active:
        block_until_ready(state)
        jax.profiler.stop_trace()
        logger.info("Stopped JAX profiler trace during shutdown -> %s", profiler_log_dir)
    multihost_utils.sync_global_devices("dynamics_train_complete")
    if step >= total_steps and not checkpoint_manager.should_save(step):
        checkpoint_manager.save(
            step=step,
            state=to_host(state),
            extra_items=iterator_items(),
            force=True,
        )
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    wb.finish()


if __name__ == "__main__":
    main()
