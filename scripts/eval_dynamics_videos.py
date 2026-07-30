"""Render a batch of dynamics rollout videos from a trained checkpoint.

Mirrors the in-training video eval (scripts/train_dynamics.py) but runs
standalone, so it works from any checkpoint on any slice size -- including the
multi-host runs where live video eval is disabled. Each output is a side-by-side
mp4: decoded ground truth on the left, `context_frames` of real context followed
by generated frames on the right.

    uv run python scripts/eval_dynamics_videos.py \
        --checkpoint_dir gs://visionary-robot-bucket/so101/checkpoints/so101_dynamics \
        --tokenizer_checkpoint_dir gs://visionary-robot-bucket/so101/checkpoints/so101_tokenizer \
        --data_dir data/so101/dyn/eval --output_dir artifacts/dynamics_videos --num_videos 20
"""

import argparse
import json
import logging
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from visionary.common.checkpoint import (
    CheckpointManager,
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.common.train_state import DynamicsTrainState
from visionary.dataset import DynamicsDataSource, RandomDynamicsCrop
from visionary.dynamics import DynamicsModel
from visionary.tokenizer_preprocessor import TokenizerPreprocessor

logger = logging.getLogger(__name__)


def load_train_config(checkpoint_dir: str) -> OmegaConf:
    """The training run stores its own resolved config next to the checkpoints."""
    manager = CheckpointManager(
        checkpoint_dir,
        instantiate({"_target_": "orbax.checkpoint.CheckpointManagerOptions"}),
    )
    metadata = manager.load_metadata()
    manager.close()
    for key in ("dynamics_config", "config"):
        if key in metadata:
            return OmegaConf.create(metadata[key])
    raise KeyError(f"No config found in checkpoint metadata at {checkpoint_dir}")


def restore_params(cfg: OmegaConf, checkpoint_dir: str, step: int | None, sample_batch):
    model = instantiate(cfg.dynamics)
    optimizer = instantiate(cfg.optimizer)

    def make_state():
        params = model.init(
            {"params": jax.random.key(0), "sample": jax.random.key(1)},
            sample_batch,
            bootstrap_rows=0,
            method=DynamicsModel.loss,
        )
        return DynamicsTrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # shapes only: allocating real optimizer state would trible the memory for
    # no reason, and restore just needs the tree structure
    abstract_state = jax.eval_shape(make_state)
    manager = CheckpointManager(checkpoint_dir, instantiate(cfg.checkpoint.manager.options))
    resolved = manager.latest_step() if step is None else int(step)
    params = manager.restore(target=abstract_state, step=resolved, params_only=True)
    manager.close()
    return model, params, resolved


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Render dynamics rollout videos.")
    parser.add_argument("--checkpoint_dir", required=True, help="Dynamics checkpoint directory.")
    parser.add_argument(
        "--tokenizer_checkpoint_dir", required=True, help="Tokenizer checkpoint directory."
    )
    parser.add_argument("--data_dir", required=True, help="Eval latent ArrayRecord directory.")
    parser.add_argument("--output_dir", required=True, help="Where to write mp4 files.")
    parser.add_argument("--num_videos", type=int, default=20)
    parser.add_argument("--step", type=int, help="Checkpoint step. Defaults to latest.")
    parser.add_argument("--context_frames", type=int, default=4)
    parser.add_argument("--generated_frames", type=int, default=60)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--context_tau", type=float, default=0.9)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    total_frames = args.context_frames + args.generated_frames
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_train_config(args.checkpoint_dir)
    logger.info("Loaded training config for exp %s", cfg.get("exp_name", "?"))

    source = DynamicsDataSource(args.data_dir)
    logger.info("Eval source has %d records", len(source))
    crop = RandomDynamicsCrop(total_frames)

    def batch_for(index: int):
        record = source[index % len(source)]
        cropped = crop.random_map(record, np.random.default_rng([args.seed, index]))
        return {
            "video": np.asarray(cropped["video"], dtype=np.float32)[None],
            "actions": np.asarray(cropped["actions"], dtype=np.float32)[None],
        }

    # the model was initialised against the training batch_length, but the
    # rollout only ever sees total_frames, so init at that length
    model, params, step = restore_params(cfg, args.checkpoint_dir, args.step, batch_for(0))
    logger.info("Restored dynamics params from step %d", step)

    tokenizer_cfg, tokenizer_variables = restore_model_export_single_device(
        args.tokenizer_checkpoint_dir
    )
    tokenizer = instantiate(tokenizer_cfg)
    preprocessor = TokenizerPreprocessor.from_config(
        restore_preprocessor_export(args.tokenizer_checkpoint_dir)
    )

    @jax.jit
    def rollout(params, tokenizer_variables, video, actions, seed):
        video = jnp.asarray(video, dtype=jnp.float32)
        actions = jnp.asarray(actions, dtype=jnp.float32)
        primed = (
            jnp.zeros_like(video).at[:, : args.context_frames].set(video[:, : args.context_frames])
        )
        context_key, sample_key = jax.random.split(jax.random.key(seed))
        context_noise = jax.random.normal(context_key, video.shape, dtype=jnp.float32)
        sample_noise = jax.random.normal(
            sample_key, (video.shape[0], args.generated_frames, *video.shape[2:]), dtype=jnp.float32
        )
        generated = model.apply(
            params,
            primed,
            actions,
            context_noise,
            sample_noise,
            args.context_frames,
            context_tau=args.context_tau,
            sample_steps=args.sample_steps,
            method=DynamicsModel.generate_rollout,
        )
        truth_images = preprocessor.patches_to_images(
            tokenizer.apply(tokenizer_variables, video, method=type(tokenizer).decode)
        ).astype(jnp.float32)
        rollout_images = preprocessor.patches_to_images(
            tokenizer.apply(tokenizer_variables, generated, method=type(tokenizer).decode)
        ).astype(jnp.float32)
        return truth_images, rollout_images

    summary = []
    for index in range(args.num_videos):
        batch = batch_for(index)
        truth, generated = jax.device_get(
            rollout(params, tokenizer_variables, batch["video"], batch["actions"], index)
        )
        truth, generated = np.asarray(truth[0]), np.asarray(generated[0])

        gen_slice = slice(args.context_frames, total_frames)
        psnr = float(
            peak_signal_noise_ratio(truth[gen_slice], generated[gen_slice], data_range=1.0)
        )
        ssim = float(
            np.mean(
                [
                    structural_similarity(t, g, data_range=1.0, channel_axis=-1)
                    for t, g in zip(truth[gen_slice], generated[gen_slice], strict=True)
                ]
            )
        )

        to_u8 = lambda x: np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)  # noqa: E731
        left, right = to_u8(truth), to_u8(generated)
        separator = np.full((left.shape[1], 4, 3), 255, dtype=np.uint8)
        frames = [
            np.concatenate([lf, separator, rf], axis=1) for lf, rf in zip(left, right, strict=True)
        ]
        path = output_dir / f"rollout_{step}_{index:02d}_psnr{psnr:.1f}.mp4"
        imageio.mimsave(path, frames, fps=args.fps)
        summary.append({"index": index, "psnr": psnr, "ssim": ssim, "path": path.name})
        logger.info(
            "[%d/%d] %s  psnr %.2f  ssim %.4f", index + 1, args.num_videos, path.name, psnr, ssim
        )

    mean_psnr = float(np.mean([s["psnr"] for s in summary]))
    mean_ssim = float(np.mean([s["ssim"] for s in summary]))
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint_dir": args.checkpoint_dir,
                "step": step,
                "context_frames": args.context_frames,
                "generated_frames": args.generated_frames,
                "sample_steps": args.sample_steps,
                "context_tau": args.context_tau,
                "mean_psnr": mean_psnr,
                "mean_ssim": mean_ssim,
                "videos": summary,
            },
            indent=2,
        )
    )
    logger.info(
        "Wrote %d videos to %s | mean PSNR %.2f, mean SSIM %.4f",
        len(summary),
        output_dir,
        mean_psnr,
        mean_ssim,
    )


if __name__ == "__main__":
    main()
