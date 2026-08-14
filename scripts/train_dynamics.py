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
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
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
    PackDynamicsEpisodes,
    interleave_dynamics_episodes,
    load_latent_stats,
    mix_dynamics_episodes,
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
    image_fraction: float,
    ema_decay: float,
):
    sample_key = fold_in_many(base_sample_key, global_step)

    def loss_fn(params):
        return state.apply_fn(
            params,
            batch,
            bootstrap_rows=bootstrap_rows,
            image_fraction=image_fraction,
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
    batch_index: jax.Array,
):
    sample_key = fold_in_many(base_sample_key, batch_index)
    _, online_metrics = state.apply_fn(
        state.params,
        batch,
        bootstrap_rows=0,
        image_fraction=0.0,
        bootstrap_target_variables=state.ema_params,
        method=DynamicsModel.loss,
        rngs={"sample": sample_key},
    )
    _, ema_metrics = state.apply_fn(
        state.ema_params,
        batch,
        bootstrap_rows=0,
        image_fraction=0.0,
        bootstrap_target_variables=state.ema_params,
        method=DynamicsModel.loss,
        rngs={"sample": sample_key},
    )
    return {
        "online_flow_mse": online_metrics["flow_mse"],
        "ema_flow_mse": ema_metrics["flow_mse"],
    }


def log_video_eval(
    wb: WandbLogger,
    online_params,
    ema_params,
    batch: DynamicsBatch,
    step: int,
    rollout_seed: int,
    video_cfg: DictConfig,
    output_dir: Path,
    tokenizer_variables,
    run_video_eval,
) -> None:
    context_frames = int(video_cfg.context_frames)
    horizon = int(video_cfg.horizon)
    (
        ground_truth_images,
        online_rollout_images,
        ema_rollout_images,
        online_diagnostics,
        ema_diagnostics,
    ) = jax.device_get(
        run_video_eval(
            online_params,
            ema_params,
            tokenizer_variables,
            batch["video"],
            batch["actions"],
            batch["embodiment_ids"],
            np.uint32(rollout_seed),
        )
    )
    ground_truth_images = np.asarray(ground_truth_images)
    ground_truth_images = np.clip(ground_truth_images, 0.0, 1.0)
    generated_ground_truth = ground_truth_images[:, context_frames:]
    rollout_images = {
        "online": np.clip(np.asarray(online_rollout_images), 0.0, 1.0),
        "ema": np.clip(np.asarray(ema_rollout_images), 0.0, 1.0),
    }
    rollout_diagnostics = {
        "online": online_diagnostics,
        "ema": ema_diagnostics,
    }
    rollout_metrics = {}
    for model_name, images in rollout_images.items():
        generated_rollout = images[:, context_frames:]
        psnr = np.asarray(
            [
                [
                    peak_signal_noise_ratio(target_frame, predicted_frame, data_range=1.0)
                    for target_frame, predicted_frame in zip(target_video, predicted_video, strict=True)
                ]
                for target_video, predicted_video in zip(generated_ground_truth, generated_rollout, strict=True)
            ],
            dtype=np.float32,
        )
        ssim = np.asarray(
            [
                [
                    structural_similarity(
                        target_frame,
                        predicted_frame,
                        data_range=1.0,
                        channel_axis=-1,
                    )
                    for target_frame, predicted_frame in zip(target_video, predicted_video, strict=True)
                ]
                for target_video, predicted_video in zip(generated_ground_truth, generated_rollout, strict=True)
            ],
            dtype=np.float32,
        )
        rollout_metrics[model_name] = {"psnr": psnr, "ssim": ssim}

    video_path = output_dir / f"eval_side_by_side_{step}.mp4"
    ground_truth_video = np.clip(np.rint(ground_truth_images * 255.0), 0, 255).astype(np.uint8)
    online_rollout_video = np.clip(np.rint(rollout_images["online"] * 255.0), 0, 255).astype(np.uint8)
    ema_rollout_video = np.clip(np.rint(rollout_images["ema"] * 255.0), 0, 255).astype(np.uint8)
    separator = np.full((ground_truth_video.shape[2], 4, 3), 255, dtype=np.uint8)
    frames = []
    for frame_index in range(ground_truth_video.shape[1]):
        rows = [
            np.concatenate(
                [
                    ground_truth_video[sample_index, frame_index],
                    separator,
                    online_rollout_video[sample_index, frame_index],
                    separator,
                    ema_rollout_video[sample_index, frame_index],
                ],
                axis=1,
            )
            for sample_index in range(ground_truth_video.shape[0])
        ]
        frames.append(np.concatenate(rows, axis=0))
    imageio.mimsave(video_path, frames, fps=int(video_cfg.fps))

    logger.info(
        "Video eval at step %d - online PSNR %.3f SSIM %.4f - EMA PSNR %.3f SSIM %.4f",
        step,
        float(np.mean(rollout_metrics["online"]["psnr"])),
        float(np.mean(rollout_metrics["online"]["ssim"])),
        float(np.mean(rollout_metrics["ema"]["psnr"])),
        float(np.mean(rollout_metrics["ema"]["ssim"])),
    )
    if wb.enabled:
        metric_log = {}
        for model_name, metrics in rollout_metrics.items():
            metric_log[f"eval/{model_name}_video_mean_psnr"] = float(np.mean(metrics["psnr"]))
            metric_log[f"eval/{model_name}_video_mean_ssim"] = float(np.mean(metrics["ssim"]))
            for metric_horizon in (1, 4, 8, 16, 32):
                if metric_horizon <= horizon:
                    metric_log[f"eval/{model_name}_video_psnr_h{metric_horizon}"] = float(
                        np.mean(metrics["psnr"][:, metric_horizon - 1])
                    )
                    metric_log[f"eval/{model_name}_video_ssim_h{metric_horizon}"] = float(
                        np.mean(metrics["ssim"][:, metric_horizon - 1])
                    )
            diagnostics = rollout_diagnostics[model_name]
            for ode_step, value in enumerate(np.mean(diagnostics["x0_norm"], axis=0)):
                metric_log[f"eval/{model_name}_x0_norm_step_{ode_step}"] = float(value)
            for ode_step, value in enumerate(np.mean(diagnostics["update_mag"], axis=0)):
                metric_log[f"eval/{model_name}_update_mag_step_{ode_step}"] = float(value)
        wb.log(
            {
                "eval/video": wandb.Video(
                    video_path.as_posix(),
                    caption=(
                        "Left: ground truth. Center: online. Right: EMA. "
                        f"Context: {context_frames}. Horizon: {horizon}."
                    ),
                ),
                **metric_log,
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

        adam_parameters = {
            "embedding",
            "base_token",
            "register_tokens",
            "action_projection_kernel",
            "action_projection_bias",
        }

        def mapper(path, x):
            names = {str(getattr(key, "key", key)).lower() for key in path}
            if names & adam_parameters or any(name.endswith("_bias") for name in names):
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

    total_steps = int(cfg.total_steps)

    source_configs = list(cfg.dataset.sources)
    max_action_dim = int(cfg.dynamics.max_action_dim)
    sampling_weights = [float(source.sampling_weight) for source in source_configs]
    wb = WandbLogger(cfg, enabled=bool(cfg.wandb.enabled) and is_primary_process)

    train_sources = []
    eval_sources = []
    for embodiment_id, source in enumerate(source_configs):
        train_source = DynamicsDataSource(source.train_dir, embodiment_id, source.action_dim)
        eval_source = DynamicsDataSource(source.eval_dir, embodiment_id, source.action_dim)
        train_sources.append(train_source)
        eval_sources.append(eval_source)
        logger.info(
            "Embodiment %d %s: action_dim=%d weight=%.4f train_sequences=%d eval_sequences=%d",
            embodiment_id,
            source.embodiment,
            source.action_dim,
            source.sampling_weight,
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

    pack_length = int(cfg.dataset.pack_length)
    logger.info("Packed sequence length: %d", pack_length)
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
    bootstrap_interval = int(cfg.loss.bootstrap_interval)
    image_fraction = float(cfg.loss.image_fraction)
    logger.info(
        "Batch layout: global=%d per_process=%d per_device=%d",
        global_batch_size,
        batch_size_per_process,
        batch_size_per_device,
    )
    logger.info(
        "Loss schedule: full-batch bootstrap every %d steps after step %d; image-only rows %.1f%%",
        bootstrap_interval,
        bootstrap_start_step,
        image_fraction * 100.0,
    )

    read_options = grain.ReadOptions(
        num_threads=cfg.dataset.num_threads,
        prefetch_buffer_size=cfg.dataset.prefetch_buffer_size,
    )

    def make_record_iterator(
        source: DynamicsDataSource,
        shuffle: bool,
        num_epochs: int | None,
        seed_offset: int,
    ):
        sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.ShardByJaxProcess() if process_count > 1 else grain.NoSharding(),
            shuffle=shuffle,
            num_epochs=num_epochs,
            seed=int(cfg.seed) + seed_offset,
        )
        loader = grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=[],
            worker_count=cfg.dataset.worker_count,
            read_options=read_options,
        )
        return iter(loader)

    _t = time.monotonic()
    train_record_iterators = [
        make_record_iterator(source, shuffle=True, num_epochs=None, seed_offset=index)
        for index, source in enumerate(train_sources)
    ]
    train_packed_iterator = iter(
        PackDynamicsEpisodes(
            mix_dynamics_episodes(train_record_iterators, sampling_weights),
            sequence_length=pack_length,
            batch_size=batch_size_per_process,
            max_action_dim=max_action_dim,
        )
    )
    logger.info("Train DataLoader creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    initial_train_batch = latent_normalizer.map(next(train_packed_iterator))
    logger.info("First batch fetch took %.1fs", time.monotonic() - _t)

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
            batch["embodiment_ids"],
            step_levels,
            signal_indices,
            batch["segment_ids"],
        )
        return DynamicsTrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    state = jax.jit(init_state, out_shardings=metrics_sharding)(
        local_batch_to_global(initial_train_batch, batch_sharding)
    )
    state_shardings = jax.tree_util.tree_map(lambda _: metrics_sharding, state)
    logger.info("State init took %.1fs", time.monotonic() - _t)

    packed_batch_shardings = {
        "video": batch_sharding,
        "actions": batch_sharding,
        "embodiment_ids": batch_sharding,
        "segment_ids": batch_sharding,
    }
    jit_train_step = jax.jit(
        train_step,
        in_shardings=(
            state_shardings,
            packed_batch_shardings,
            metrics_sharding,
            metrics_sharding,
        ),
        out_shardings=(state_shardings, metrics_sharding),
        static_argnames=("bootstrap_rows", "image_fraction", "ema_decay"),
        donate_argnums=(0,),
    )
    jit_eval_step = jax.jit(
        eval_step,
        in_shardings=(
            state_shardings,
            packed_batch_shardings,
            metrics_sharding,
            metrics_sharding,
        ),
        out_shardings=metrics_sharding,
    )

    video_cfg = cfg.video_eval
    video_eval_enabled = is_primary_process and process_count == 1
    video_output_dir = None
    run_video_eval = None
    tokenizer_variables = None
    rollout_eval_batch = None
    if is_primary_process and process_count > 1:
        logger.warning(
            "Live dynamics video eval is disabled for multi-process training. "
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
            tokenizer_checkpoint_dir, step=video_cfg.tokenizer.checkpoint_step
        )
        preprocessor_cfg = restore_preprocessor_export(
            tokenizer_checkpoint_dir, step=video_cfg.tokenizer.checkpoint_step
        )
        tokenizer = instantiate(tokenizer_cfg)
        preprocessor = TokenizerPreprocessor.from_config(preprocessor_cfg)

        requested_tau = float(video_cfg.context_tau)
        video_context_frames = int(video_cfg.context_frames)
        rollout_horizon = int(video_cfg.horizon)
        total_video_frames = video_context_frames + rollout_horizon
        rollout_samples = []
        for record_index in range(max(len(source) for source in eval_sources)):
            for eval_source in eval_sources:
                if record_index >= len(eval_source):
                    continue
                record = eval_source[record_index]
                if len(record["video"]) < total_video_frames:
                    continue
                start = (len(record["video"]) - total_video_frames) // 2
                stop = start + total_video_frames
                previous_actions = np.concatenate(
                    [record["prev_action"][None], record["actions"][:-1]],
                    axis=0,
                )
                if previous_actions.ndim == 2:
                    previous_actions = np.pad(
                        previous_actions,
                        ((0, 0), (0, max_action_dim - record["action_dim"])),
                    )
                rollout_samples.append(
                    latent_normalizer.map(
                        DynamicsBatch(
                            video=record["video"][start:stop],
                            actions=previous_actions[start:stop],
                            embodiment_ids=np.full(total_video_frames, record["embodiment_id"], dtype=np.int32),
                            segment_ids=np.zeros(total_video_frames, dtype=np.int32),
                        )
                    )
                )
                if len(rollout_samples) == int(video_cfg.num_samples):
                    break
            if len(rollout_samples) == int(video_cfg.num_samples):
                break
        if len(rollout_samples) < int(video_cfg.num_samples):
            raise ValueError(
                f"Need {video_cfg.num_samples} eval episodes with at least "
                f"{total_video_frames} frames, found {len(rollout_samples)}"
            )
        rollout_eval_batch = DynamicsBatch(
            video=np.stack([sample["video"] for sample in rollout_samples]),
            actions=np.stack([sample["actions"] for sample in rollout_samples]),
            embodiment_ids=np.stack([sample["embodiment_ids"] for sample in rollout_samples]),
            segment_ids=np.stack([sample["segment_ids"] for sample in rollout_samples]),
        )

        @jax.jit
        def run_video_eval(
            online_params,
            ema_params,
            tokenizer_variables,
            video_batch,
            action_batch,
            embodiment_id_batch,
            rollout_seed,
        ):
            video = jnp.asarray(video_batch, dtype=jnp.float32)
            eval_action_dtype = (
                jnp.float32 if str(cfg.dynamics.get("action_mode", "discrete")) == "continuous" else jnp.int32
            )
            actions = jnp.asarray(action_batch, dtype=eval_action_dtype)
            embodiment_ids = jnp.asarray(embodiment_id_batch, dtype=jnp.int32)

            rollout_key = jax.random.key(rollout_seed)
            context_noise_key, sample_noise_key = jax.random.split(rollout_key)
            context_noise = jax.random.normal(context_noise_key, video.shape, dtype=jnp.float32)
            sample_noise = jax.random.normal(
                sample_noise_key,
                (video.shape[0], rollout_horizon, *video.shape[2:]),
                dtype=jnp.float32,
            )
            rollout_videos = []
            rollout_diagnostics = []
            for dynamics_params in (online_params, ema_params):
                rollout_video = jnp.zeros_like(video)
                rollout_video = rollout_video.at[:, :video_context_frames].set(video[:, :video_context_frames])
                generated, diagnostics = model.apply(
                    dynamics_params,
                    rollout_video,
                    actions,
                    embodiment_ids,
                    context_noise,
                    sample_noise,
                    video_context_frames,
                    context_tau=requested_tau,
                    sample_steps=int(video_cfg.sample_steps),
                    method=DynamicsModel.generate_rollout,
                )
                rollout_videos.append(generated)
                rollout_diagnostics.append(diagnostics)

            raw_video = denormalize_latents(video, model.latent_mean, model.latent_std)
            ground_truth_patches = tokenizer.apply(tokenizer_variables, raw_video, method=type(tokenizer).decode)
            ground_truth_images = preprocessor.patches_to_images(ground_truth_patches).astype(jnp.float32)
            decoded_rollouts = []
            for rollout_video in rollout_videos:
                raw_rollout = denormalize_latents(rollout_video, model.latent_mean, model.latent_std)
                rollout_patches = tokenizer.apply(tokenizer_variables, raw_rollout, method=type(tokenizer).decode)
                decoded_rollouts.append(preprocessor.patches_to_images(rollout_patches).astype(jnp.float32))
            return (
                ground_truth_images,
                decoded_rollouts[0],
                decoded_rollouts[1],
                rollout_diagnostics[0],
                rollout_diagnostics[1],
            )

        logger.info(
            "Video eval ready; samples=%d context=%d horizon=%d sample_steps=%d tau=%.4f",
            int(video_cfg.num_samples),
            video_context_frames,
            rollout_horizon,
            int(video_cfg.sample_steps),
            requested_tau,
        )
        logger.info("Video eval init took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    checkpoint_manager: CheckpointManager = instantiate(cfg.checkpoint.manager)
    if is_primary_process:
        checkpoint_manager.save_metadata({"config": OmegaConf.to_container(cfg, resolve=False)})
    logger.info("CheckpointManager creation took %.1fs", time.monotonic() - _t)
    last_checkpoint_step: int | None = None
    export_interval_steps = int(cfg.checkpoint.export_interval_steps)

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
        if force or (export_interval_steps > 0 and step % export_interval_steps == 0):
            save_model_export(checkpoint_manager.directory, step, cfg.dynamics, state.ema_params)

    def iterator_items():
        # The current episode remainder lives in the packer, not Grain.
        return {
            f"train_iterator_{source_index}": iterator for source_index, iterator in enumerate(train_record_iterators)
        }

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
        train_packed_iterator = iter(
            PackDynamicsEpisodes(
                mix_dynamics_episodes(train_record_iterators, sampling_weights),
                sequence_length=pack_length,
                batch_size=batch_size_per_process,
                max_action_dim=max_action_dim,
            )
        )
        initial_train_batch = None
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
    latest_flow_metrics = None
    latest_bootstrap_metrics = None
    logger.info(
        "Asynchronous timing mode enabled for dynamics training; timing logs are averaged over each logging window."
    )
    while True:
        current_step = step

        with timer("data"):
            if initial_train_batch is not None:
                batch = initial_train_batch
                initial_train_batch = None
            else:
                batch = next(train_packed_iterator)
                batch = latent_normalizer.map(batch)
        with timer("transfer"):
            batch = local_batch_to_global(batch, batch_sharding)
        bootstrap_rows = (
            global_batch_size
            if current_step >= bootstrap_start_step and (current_step - bootstrap_start_step) % bootstrap_interval == 0
            else 0
        )
        with timer("compute"):
            state, metrics = jit_train_step(
                state,
                batch,
                train_key,
                jnp.asarray(current_step, dtype=jnp.int32),
                bootstrap_rows,
                image_fraction,
                cfg.ema_decay,
            )
        if bootstrap_rows:
            latest_bootstrap_metrics = metrics
        else:
            latest_flow_metrics = metrics
        step = current_step + 1

        t_eval = 0.0
        if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
            t_eval_start = time.monotonic()
            totals: dict[str, float] = {}
            eval_record_iterators = [
                make_record_iterator(source, shuffle=False, num_epochs=1, seed_offset=index)
                for index, source in enumerate(eval_sources)
            ]
            eval_packed_iterator = iter(
                PackDynamicsEpisodes(
                    interleave_dynamics_episodes(eval_record_iterators),
                    sequence_length=pack_length,
                    batch_size=batch_size_per_process,
                    max_action_dim=max_action_dim,
                )
            )
            eval_batches = [
                latent_normalizer.map(batch)
                for batch in itertools.islice(eval_packed_iterator, cfg.dataset.eval.max_batches)
            ]
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
                        jnp.asarray(batch_idx, dtype=jnp.int32),
                    )
                )
                for k, v in batch_metrics.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                num_batches += 1

            if num_batches > 0:
                eval_metrics = {k: v / num_batches for k, v in totals.items()}
                wb.log({f"eval/{name}": value for name, value in eval_metrics.items()}, step=step)
            if video_eval_enabled:
                log_video_eval(
                    wb,
                    put_single_device_tree(state.params),
                    put_single_device_tree(state.ema_params),
                    rollout_eval_batch,
                    step=step,
                    rollout_seed=int(video_cfg.seed),
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
            with timer("compute"):
                metrics_to_log = metrics if latest_flow_metrics is None else latest_flow_metrics
                train_metrics = jax.device_get(metrics_to_log)
                if latest_bootstrap_metrics is not None:
                    bootstrap_metrics = jax.device_get(latest_bootstrap_metrics)
                    train_metrics["bootstrap_loss"] = bootstrap_metrics["bootstrap_loss"]
                    train_metrics["bootstrap_target_norm"] = bootstrap_metrics["bootstrap_target_norm"]
            latest_flow_metrics = None
            latest_bootstrap_metrics = None
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
