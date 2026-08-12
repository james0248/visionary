import itertools
import logging
import time
from pathlib import Path

import grain.python as grain
import hydra
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from hydra.utils import instantiate, to_absolute_path
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P
from omegaconf import DictConfig, OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from visionary.common.checkpoint import (
    CheckpointManager,
    restore_model_export_single_device,
    restore_preprocessor_export,
    save_model_export,
)
from visionary.common.jax import (
    data_parallel_mesh,
    fold_in_many,
    init_distributed_if_pod,
    local_batch_to_global,
)
from visionary.common.timing import PhaseTimer
from visionary.common.train_state import DynamicsTrainState
from visionary.common.wandb import WandbLogger
from visionary.dataset import (
    DynamicsBatch,
    DynamicsDataSource,
    NormalizeDynamicsLatents,
    RandomDynamicsCrop,
    SubsetDataSource,
    load_latent_stats,
    load_record_lengths,
)
from visionary.models.dreamer4.dynamics import DynamicsModel, denormalize_latents
from visionary.models.dreamer4.tokenizer_preprocessor import TokenizerPreprocessor

logger = logging.getLogger(__name__)


DATA_AXIS = "data"


def put_single_device_tree(tree):
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(jax.device_get(value), sharding) if hasattr(value, "shape") else value,
        tree,
    )


def train_step(
    state: DynamicsTrainState,
    batch: DynamicsBatch,
    base_sample_key: jax.Array,
    global_step: jax.Array,
    bootstrap_rows: int,
    ema_decay: float,
):
    sample_key = fold_in_many(base_sample_key, global_step)

    def loss_fn(params):
        return state.apply_fn(
            params,
            batch,
            bootstrap_rows=bootstrap_rows,
            bootstrap_target_variables=state.ema_params,
            method=DynamicsModel.loss,
            rngs={"sample": sample_key},
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.update_ema(ema_decay)
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
        state.ema_params,
        batch,
        bootstrap_rows=bootstrap_rows,
        bootstrap_target_variables=state.ema_params,
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
            for target_frame, predicted_frame in zip(generated_ground_truth, generated_rollout, strict=True)
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
            for target_frame, predicted_frame in zip(generated_ground_truth, generated_rollout, strict=True)
        ],
        dtype=np.float32,
    )
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
        "Video eval at step %d - mean PSNR %.3f, mean SSIM %.4f",
        step,
        mean_psnr,
        mean_ssim,
    )
    if wb.enabled:
        wb.log(
            {
                "eval/video": wandb.Video(
                    video_path.as_posix(),
                    caption=(
                        f"Left: decoded eval ground truth. Right: {context_frames} context "
                        f"frames followed by {generated_frames} generated frames."
                    ),
                ),
                "eval/video_mean_psnr": mean_psnr,
                "eval/video_mean_ssim": mean_ssim,
            },
            step=step,
        )


def build_wsd_schedule(cfg) -> optax.Schedule:
    opt = cfg.optimizer
    total = int(cfg.total_steps)
    warmup = max(int(total * opt.warmup_ratio), 1)
    decay = int(total * opt.decay_ratio)
    stable = max(total - warmup - decay, 0)
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, opt.peak_lr, warmup),
            optax.constant_schedule(opt.peak_lr),
            optax.linear_schedule(opt.peak_lr, 0.0, decay),
        ],
        boundaries=[warmup, warmup + stable],
    )


def build_optimizer(cfg) -> optax.GradientTransformation:
    opt = cfg.optimizer
    if "_target_" in opt:
        return instantiate(opt)

    schedule = build_wsd_schedule(cfg)
    adam_ratio = float(opt.adam_lr_ratio)

    def adam_schedule(step):
        return schedule(step) * adam_ratio

    def muon_dimension_numbers(params):
        from optax.contrib import MuonDimensionNumbers

        embedding_like = {"embedding", "base_token", "register_tokens"}

        def mapper(path, x):
            names = {str(getattr(key, "key", key)).lower() for key in path}
            if names & embedding_like:
                return None
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


@hydra.main(config_path="config", config_name="dynamics", version_base=None)
def main(cfg: DictConfig):
    init_distributed_if_pod(logger=logger)

    process_index = jax.process_index()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()
    is_primary_process = process_index == 0
    mesh = data_parallel_mesh(DATA_AXIS)
    batch_sharding = NamedSharding(mesh, P(DATA_AXIS))
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
    logger.info("Data mesh: %d devices", mesh.shape[DATA_AXIS])

    latent_stats_path = to_absolute_path(str(cfg.dataset.latent_stats))
    latent_stats = load_latent_stats(latent_stats_path)
    latent_normalizer = NormalizeDynamicsLatents(latent_stats.mean, latent_stats.std)
    OmegaConf.update(cfg, "dynamics.latent_mean", latent_stats.mean.tolist(), force_add=True)
    OmegaConf.update(cfg, "dynamics.latent_std", latent_stats.std.tolist(), force_add=True)
    logger.info(
        "Loaded latent stats from %s: count=%d std=[%.4f, %.4f]",
        latent_stats_path,
        latent_stats.count,
        float(latent_stats.std.min()),
        float(latent_stats.std.max()),
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
        "Data loader settings: worker_count=%d num_threads=%d prefetch_buffer_size=%d effective_read_threads=%d",
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
    global_batch_size = int(cfg.dataset.batch_size)
    if global_batch_size % process_count != 0:
        raise ValueError(
            "cfg.dataset.batch_size is global and must be divisible by the process count; "
            f"got batch_size={global_batch_size} process_count={process_count}"
        )
    batch_size_per_process = global_batch_size // process_count
    if batch_size_per_process % local_device_count != 0:
        raise ValueError(
            "global batch_size / process_count must be divisible by "
            "jax.local_device_count() for input sharding; got "
            f"per_process={batch_size_per_process} local_device_count={local_device_count}"
        )
    batch_size_per_device = batch_size_per_process // local_device_count
    bootstrap_start_step = int(cfg.loss.bootstrap_start_step)
    target_bootstrap_ratio = float(cfg.loss.bootstrap_ratio)
    target_bootstrap_rows = min(
        max(int(round(target_bootstrap_ratio * global_batch_size)), 0),
        global_batch_size,
    )
    logger.info(
        "Batch layout: global=%d per_process=%d per_device=%d",
        global_batch_size,
        batch_size_per_process,
        batch_size_per_device,
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
        lengths: list[int] | None = None,
    ) -> grain.DataLoader:
        if lengths is not None:
            indices = [i for i, n in enumerate(lengths) if n >= sequence_length]
            if len(indices) < len(lengths):
                logger.info(
                    "Length %d: %d/%d records fit",
                    sequence_length,
                    len(indices),
                    len(lengths),
                )
            source = SubsetDataSource(source, indices)
        sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.ShardByJaxProcess() if process_count > 1 else grain.NoSharding(),
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
                latent_normalizer,
                grain.Batch(
                    batch_size=batch_size_per_process,
                    drop_remainder=drop_remainder,
                ),
            ],
            worker_count=cfg.dataset.worker_count,
            read_options=read_options,
        )

    _t = time.monotonic()
    train_lengths_manifest = load_record_lengths(cfg.dataset.train_dir)
    eval_lengths_manifest = load_record_lengths(cfg.dataset.eval_dir)
    train_loaders = {
        sequence_length: make_loader(
            train_source,
            sequence_length=sequence_length,
            shuffle=True,
            drop_remainder=True,
            seed=cfg.seed + sequence_length,
            lengths=train_lengths_manifest,
        )
        for sequence_length in train_sequence_lengths
    }
    logger.info("Train DataLoader creation took %.1fs", time.monotonic() - _t)

    init_sequence_length = max(train_sequence_lengths)
    _t = time.monotonic()
    sample_batch = next(iter(train_loaders[init_sequence_length]))
    logger.info("First batch fetch took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    eval_loader = make_loader(
        eval_source,
        sequence_length=cfg.dataset.eval.batch_length,
        shuffle=False,
        drop_remainder=True,
        seed=cfg.seed,
        lengths=eval_lengths_manifest,
    )
    logger.info("Eval DataLoader creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    key = jax.random.key(cfg.seed)
    init_key, train_key, eval_key = jax.random.split(key, num=3)
    model = instantiate(cfg.dynamics)
    optimizer = build_optimizer(cfg)

    def init_state(batch):
        video = jnp.asarray(batch["video"], dtype=jnp.float32)
        batch_size, sequence_length, _, _ = video.shape
        z = video.reshape(batch_size, sequence_length, model.num_obs_tokens, -1)
        step_levels = jnp.zeros((batch_size, sequence_length), dtype=jnp.int32)
        signal_indices = jnp.zeros((batch_size, sequence_length), dtype=jnp.int32)
        params = model.init(
            init_key,
            z,
            batch["actions"],
            step_levels,
            signal_indices,
        )
        return DynamicsTrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    state = jax.jit(init_state, out_shardings=metrics_sharding)(local_batch_to_global(sample_batch, batch_sharding))
    state_shardings = jax.tree_util.tree_map(lambda _: metrics_sharding, state)
    logger.info("State init took %.1fs", time.monotonic() - _t)

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
        static_argnames=("bootstrap_rows", "ema_decay"),
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

    video_cfg = cfg.video_eval
    video_eval_enabled = is_primary_process and process_count == 1
    video_output_dir = None
    run_video_eval = None
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
                jnp.float32 if str(cfg.dynamics.get("action_mode", "discrete")) == "continuous" else jnp.int32
            )
            actions = jnp.asarray(action_batch[:1, :total_video_frames], dtype=eval_action_dtype)
            rollout_video = jnp.zeros_like(video)
            rollout_video = rollout_video.at[:, :video_context_frames].set(video[:, :video_context_frames])

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
            raw_video = denormalize_latents(video, model.latent_mean, model.latent_std)
            raw_rollout = denormalize_latents(rollout_video, model.latent_mean, model.latent_std)
            ground_truth_patches = tokenizer.apply(tokenizer_variables, raw_video, method=type(tokenizer).decode)
            rollout_patches = tokenizer.apply(tokenizer_variables, raw_rollout, method=type(tokenizer).decode)
            ground_truth_images = preprocessor.patches_to_images(ground_truth_patches).astype(jnp.float32)
            rollout_images = preprocessor.patches_to_images(rollout_patches).astype(jnp.float32)
            return ground_truth_images, rollout_images

        logger.info(
            "Video eval ready; context=%d generated=%d sample_steps=%d tau=%.4f",
            int(video_cfg.context_frames),
            int(video_cfg.generated_frames),
            int(video_cfg.sample_steps),
            requested_tau,
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
            save_model_export(checkpoint_manager.directory, step, cfg.dynamics, state.ema_params)

    train_iterators = {sequence_length: iter(loader) for sequence_length, loader in train_loaders.items()}

    def iterator_items():
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
    timer = PhaseTimer()
    logger.info(
        "Asynchronous timing mode enabled for dynamics training; timing logs are averaged over each logging window."
    )
    while True:
        current_step = step

        sequence_length = sequence_length_for_step(current_step)

        with timer("data"):
            try:
                batch = next(train_iterators[sequence_length])
            except StopIteration:
                train_iterators[sequence_length] = iter(train_loaders[sequence_length])
                batch = next(train_iterators[sequence_length])
        with timer("transfer"):
            batch = local_batch_to_global(batch, batch_sharding)
        bootstrap_rows = bootstrap_rows_for_step(current_step)
        with timer("compute"):
            state, metrics = jit_train_step(
                state, batch, train_key, jnp.asarray(current_step, dtype=jnp.int32), bootstrap_rows, cfg.ema_decay
            )
        step = current_step + 1

        t_eval = 0.0
        if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
            t_eval_start = time.monotonic()
            totals: dict[str, float] = {}
            eval_batches = list(itertools.islice(iter(eval_loader), cfg.dataset.eval.max_batches))
            global_eval_batch_counts = np.asarray(
                multihost_utils.process_allgather(np.asarray(len(eval_batches), dtype=np.int32))
            )
            eval_batches = eval_batches[: int(np.min(global_eval_batch_counts))]

            num_batches = 0
            for batch_idx, eval_batch in enumerate(eval_batches):
                batch_metrics = jax.device_get(
                    jit_eval_step(
                        state,
                        local_batch_to_global(eval_batch, batch_sharding),
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
                    put_single_device_tree(state.ema_params),
                    eval_batches[(step // cfg.eval_steps) % num_batches],
                    step=step,
                    rollout_seed=int(
                        jax.random.randint(fold_in_many(eval_key, step, num_batches), (), 0, np.iinfo(np.int32).max)
                    ),
                    video_cfg=video_cfg,
                    output_dir=video_output_dir,
                    tokenizer_variables=tokenizer_variables,
                    run_video_eval=run_video_eval,
                )
            t_eval = time.monotonic() - t_eval_start
            logger.info("Eval at step %d - %d batches in %.3fs", step, num_batches, t_eval)

        if checkpoint_manager.should_save(step):
            save_checkpoint(step)

        if step % cfg.log_interval == 0:
            # Syncing on metrics charges the device wait to compute
            with timer("compute"):
                train_metrics = jax.device_get(metrics)
            timing_stats = timer.log(logger, step, step - timing_start_step, eval=t_eval)
            timing_start_step = step
            wb.log(
                {
                    **{k: float(v) for k, v in train_metrics.items()},
                    "train/sequence_length": sequence_length,
                    **{f"train/{k}": v for k, v in timing_stats.items()},
                },
                step=step,
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
