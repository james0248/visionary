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
import orbax.checkpoint as ocp
import wandb
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from visionary.common.checkpoint import CheckpointManager
from visionary.common.train_state import DynamicsTrainState, TokenizerTrainState
from visionary.common.wandb import WandbLogger
from visionary.dataset import DynamicsBatch, DynamicsDataSource, RandomDynamicsCrop
from visionary.dynamics import DynamicsModel
from visionary.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@jax.jit
def train_step(
    state: DynamicsTrainState,
    batch: DynamicsBatch,
    sample_key: jax.Array,
):
    def loss_fn(params):
        return state.apply_fn(
            params,
            batch,
            method=DynamicsModel.loss,
            rngs={"sample": sample_key},
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


@jax.jit
def eval_step(
    state: DynamicsTrainState,
    batch: DynamicsBatch,
    sample_key: jax.Array,
):
    _, metrics = state.apply_fn(
        state.params,
        batch,
        method=DynamicsModel.loss,
        rngs={"sample": sample_key},
    )
    return metrics


def log_video_eval(
    wb: WandbLogger,
    state: DynamicsTrainState,
    batch: DynamicsBatch,
    *,
    step: int,
    rollout_key: jax.Array,
    video_cfg: DictConfig,
    output_dir: Path,
    tokenizer_params,
    decode_images,
    generate_next,
    context_tau_used: float,
) -> None:
    context_frames = int(video_cfg.context_frames)
    generated_frames = int(video_cfg.generated_frames)
    total_frames = context_frames + generated_frames

    video = jnp.asarray(batch["video"][:1, :total_frames], dtype=jnp.float32)
    actions = jnp.asarray(batch["actions"][:1, :total_frames], dtype=jnp.int32)
    rollout_video = jnp.zeros_like(video)
    rollout_video = rollout_video.at[:, :context_frames].set(video[:, :context_frames])

    context_noise_key, sample_noise_key = jax.random.split(rollout_key)
    context_noise = jax.random.normal(context_noise_key, video.shape, dtype=jnp.float32)
    sample_noise = jax.random.normal(sample_noise_key, video.shape, dtype=jnp.float32)

    for frame_idx in range(context_frames, total_frames):
        next_representation = generate_next(
            state.params,
            rollout_video,
            actions,
            context_noise,
            sample_noise[:, frame_idx],
            jnp.asarray(frame_idx, dtype=jnp.int32),
        ).astype(jnp.float32)
        rollout_video = rollout_video.at[:, frame_idx].set(next_representation)

    ground_truth_images = np.asarray(jax.device_get(decode_images(tokenizer_params, video)))
    rollout_images = np.asarray(jax.device_get(decode_images(tokenizer_params, rollout_video)))
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
    logger.info("JAX backend: %s, devices: %s", jax.default_backend(), jax.devices())
    wb = WandbLogger(cfg)
    total_steps = cfg.total_steps

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

    def make_loader(
        source: DynamicsDataSource,
        *,
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
                    batch_size=cfg.dataset.batch_size,
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
                        drop_remainder=False,
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
            drop_remainder=False,
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

    _t = time.monotonic()
    video_cfg = cfg.video_eval
    video_output_dir = Path(to_absolute_path(video_cfg.output_dir))
    video_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_cfg = OmegaConf.load(to_absolute_path(video_cfg.tokenizer.config_path))
    tokenizer = instantiate(tokenizer_cfg.tokenizer)
    tokenizer_patch_size = int(tokenizer_cfg.dataset.patch_size)
    tokenizer_pad_width = tuple(int(value) for value in tokenizer_cfg.dataset.pad_width)
    tokenizer_patch_dim = tokenizer_patch_size * tokenizer_patch_size * 3
    tokenizer_patch_count = int(tokenizer_cfg.tokenizer.x_len) * int(tokenizer_cfg.tokenizer.y_len)
    tokenizer_batch = {
        "video": np.zeros((1, 1, tokenizer_patch_count, tokenizer_patch_dim), dtype=np.uint8)
    }
    tokenizer_key, tokenizer_sample_key = jax.random.split(jax.random.key(cfg.seed))
    tokenizer_state = TokenizerTrainState.create(
        apply_fn=tokenizer.apply,
        params=tokenizer.init(
            {"params": tokenizer_key, "sample": tokenizer_sample_key},
            tokenizer_batch,
        ),
        tx=optax.adam(0.0),
        mse_sq_ema=jnp.ones((), dtype=jnp.float32),
        l1_sq_ema=jnp.ones((), dtype=jnp.float32),
        lpips_sq_ema=jnp.ones((), dtype=jnp.float32),
        motion_sq_ema=jnp.ones((), dtype=jnp.float32),
    )
    tokenizer_checkpoint_dir = video_cfg.tokenizer.checkpoint_dir
    if "://" not in str(tokenizer_checkpoint_dir):
        tokenizer_checkpoint_dir = to_absolute_path(str(tokenizer_checkpoint_dir))
    with CheckpointManager(tokenizer_checkpoint_dir, ocp.CheckpointManagerOptions()) as manager:
        tokenizer_state = manager.restore(
            target=tokenizer_state,
            step=video_cfg.tokenizer.checkpoint_step,
        )

    requested_tau = float(video_cfg.context_tau)
    context_step_count = 1 << (int(cfg.dynamics.max_step_size) - 1)
    context_tau_used = (
        min(max(round(requested_tau * context_step_count), 0), context_step_count - 1)
        / context_step_count
    )

    @jax.jit
    def decode_images(params, latent):
        return tokenizer.apply(
            params,
            latent,
            tokenizer_patch_size,
            tokenizer_pad_width,
            method=Tokenizer.decode_images,
        )

    @jax.jit
    def generate_next(params, video_prefix, actions, context_noise, sample_noise, target_index):
        return model.apply(
            params,
            video_prefix,
            actions,
            context_noise,
            sample_noise,
            target_index,
            context_tau=requested_tau,
            sample_steps=int(video_cfg.sample_steps),
            method=DynamicsModel.generate_next,
        )

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
    logger.info("CheckpointManager creation took %.1fs", time.monotonic() - _t)
    if cfg.checkpoint.resume_step is not None:
        state = checkpoint_manager.restore(
            target=state,
            step=cfg.checkpoint.resume_step,
        )
        logger.info("Resumed dynamics training from step %d", int(state.step))

    step = int(state.step)
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

    train_iterators = {
        sequence_length: iter(loader) for sequence_length, loader in train_loaders.items()
    }

    while True:
        t1 = time.monotonic()

        if cfg.dataset.alternating_lengths.enabled and (
            step >= total_steps - cfg.dataset.alternating_lengths.final_long_only_steps
            or (step + 1) % cfg.dataset.alternating_lengths.long_every == 0
        ):
            sequence_length = cfg.dataset.alternating_lengths.long
        elif cfg.dataset.alternating_lengths.enabled:
            sequence_length = cfg.dataset.alternating_lengths.short
        else:
            sequence_length = cfg.dataset.batch_length

        if cfg.overfit_single_batch:
            batch = overfit_batches[sequence_length]
        else:
            try:
                batch = next(train_iterators[sequence_length])
            except StopIteration:
                train_iterators[sequence_length] = iter(train_loaders[sequence_length])
                batch = next(train_iterators[sequence_length])
        t2 = time.monotonic()

        batch = jax.device_put(batch)
        t3 = time.monotonic()

        sample_key = jax.random.fold_in(train_key, step)
        state, metrics = train_step(state, batch, sample_key)
        jax.block_until_ready(metrics)
        t4 = time.monotonic()

        step = int(state.step)

        t_eval = 0.0
        if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
            t_eval_start = time.monotonic()
            totals: dict[str, float] = {}
            eval_run_key = jax.random.fold_in(eval_key, step)
            if cfg.overfit_single_batch:
                eval_batches = [overfit_batches[cfg.dataset.eval.batch_length]]
            else:
                eval_batches = list(
                    itertools.islice(iter(eval_loader), cfg.dataset.eval.max_batches)
                )

            num_batches = 0
            for batch_idx, eval_batch in enumerate(eval_batches):
                eval_sample_key = jax.random.fold_in(eval_run_key, batch_idx)
                batch_metrics = jax.device_get(
                    eval_step(state, jax.device_put(eval_batch), eval_sample_key)
                )
                for k, v in batch_metrics.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                num_batches += 1

            eval_metrics = {k: v / num_batches for k, v in totals.items()}
            wb.log(
                {f"eval/{k}": v for k, v in eval_metrics.items()},
                step=step,
            )
            log_video_eval(
                wb,
                state,
                eval_batches[0],
                step=step,
                rollout_key=jax.random.fold_in(eval_run_key, num_batches),
                video_cfg=video_cfg,
                output_dir=video_output_dir,
                tokenizer_params=tokenizer_state.params,
                decode_images=decode_images,
                generate_next=generate_next,
                context_tau_used=context_tau_used,
            )
            t_eval = time.monotonic() - t_eval_start
            logger.info("Eval at step %d - %d batches in %.3fs", step, num_batches, t_eval)

        if step % cfg.log_interval == 0:
            wb.log(
                {
                    **{k: float(v) for k, v in metrics.items()},
                    "train/sequence_length": sequence_length,
                },
                step=step,
            )

        if checkpoint_manager.should_save(step):
            checkpoint_manager.save(step=step, state=state)

        t5 = time.monotonic()
        logger.info(
            "Step %d - seq: %d, data: %.3fs, transfer: %.3fs, compute: %.3fs, overhead: %.3fs",
            step,
            sequence_length,
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t5 - t4 - t_eval,
        )
        if step >= total_steps:
            logger.info("Reached total_steps=%d; stopping dynamics training.", total_steps)
            break

    if step >= total_steps and not checkpoint_manager.should_save(step):
        checkpoint_manager.save(step=step, state=state, force=True)
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    wb.finish()


if __name__ == "__main__":
    main()
