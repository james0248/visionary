"""Rollout error vs teacher-forced error along the timeline, plus videos.

For each clip: (a) a normal autoregressive rollout, (b) a 1-step teacher-forced
pass where every frame t is generated from fully real context. Writes a
three-panel video (raw | teacher-forced | rollout) and a JSON of per-frame
latent error norms for both, together with the norm of the 0.1-scaled context
noise as the reference the model is trained to tolerate.

    uv run python scripts/analysis/eval_teacher_forced.py \
        --checkpoint_dir gs://.../so101_dynamics_small \
        --tokenizer_checkpoint_dir gs://.../so101_tokenizer \
        --data_dir data/so101/dyn/eval --raw_shards_dir data/so101/shards/eval
"""

import argparse
import io
import json
import logging
from pathlib import Path

import grain.python as grain
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visionary.common.checkpoint import (
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.dataset import decode_video_window
from visionary.dynamics import DynamicsModel
from visionary.tokenizer_preprocessor import TokenizerPreprocessor

from eval_dynamics_videos import build_raw_index, load_train_config, restore_params

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Teacher-forced vs rollout error.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--tokenizer_checkpoint_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--raw_shards_dir")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--output_dir", default="tf_videos")
    parser.add_argument("--output", default="tf_errors.json")
    parser.add_argument("--step", type=int)
    parser.add_argument("--indices", default="2,7,256")
    parser.add_argument("--context_frames", type=int, default=4)
    parser.add_argument("--total_frames", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--context_tau", type=float, default=0.9)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action_modes", default="real")
    parser.add_argument("--offset_shift", type=int, default=0)
    args = parser.parse_args()
    action_modes = args.action_modes.split(",")

    cfg = load_train_config(args.checkpoint_dir)
    latents = grain.ArrayRecordDataSource(
        sorted(str(p) for p in Path(args.data_dir).glob("*.arecord"))
    )
    raw_index = None if args.no_video else build_raw_index(args.raw_shards_dir)

    def sample_for(index: int, action_mode: str = "real"):
        with np.load(io.BytesIO(latents[index % len(latents)])) as data:
            video = np.asarray(data["frames"])
            actions = np.asarray(data["actions"], dtype=np.float32)
            prev_action = np.asarray(data["prev_action"], dtype=np.float32)
            key = (str(data["repo"]), int(data["episode"]), str(data["camera"]))
            record_start = int(data["start_index"])
        total = min(args.total_frames, (len(video) - 1) // args.stride + 1)
        span = (total - 1) * args.stride + 1
        rng = np.random.default_rng([args.seed, index])
        offset = int(rng.integers(0, max(len(video) - span, 0) + 1))
        offset = min(offset + args.offset_shift * args.stride, max(len(video) - span, 0))
        indices = offset + np.arange(total) * args.stride
        aligned = actions[indices - 1]
        if offset == 0:
            aligned[0] = prev_action
        if action_mode == "shuffled":
            aligned = aligned[np.random.default_rng([99, index]).permutation(len(aligned))]
        elif action_mode == "frozen":
            aligned = np.repeat(aligned[:1], len(aligned), axis=0)
        elif action_mode == "zero":
            aligned = np.zeros_like(aligned)
        return {
            "video": np.asarray(video[indices], dtype=np.float32)[None],
            "actions": np.asarray(aligned, dtype=np.float32)[None],
            "key": key,
            "absolute_start": record_start + offset,
            "span": span,
            "total": total,
        }

    first = sample_for(int(args.indices.split(",")[0]))
    model, params, step = restore_params(
        cfg, args.checkpoint_dir, args.step, {"video": first["video"], "actions": first["actions"]}
    )
    logger.info("Restored dynamics params from step %d", step)

    tokenizer_cfg, tokenizer_variables = restore_model_export_single_device(
        args.tokenizer_checkpoint_dir
    )
    tokenizer = instantiate(tokenizer_cfg)
    preprocessor = TokenizerPreprocessor.from_config(
        restore_preprocessor_export(args.tokenizer_checkpoint_dir)
    )

    @jax.jit
    def rollout_full(params, video, actions, seed):
        video = jnp.asarray(video, jnp.float32)
        primed = jnp.zeros_like(video).at[:, : args.context_frames].set(
            video[:, : args.context_frames]
        )
        ck, sk = jax.random.split(jax.random.key(seed))
        return model.apply(
            params,
            primed,
            jnp.asarray(actions, jnp.float32),
            jax.random.normal(ck, video.shape, jnp.float32),
            jax.random.normal(
                sk, (video.shape[0], video.shape[1] - args.context_frames, *video.shape[2:]),
                jnp.float32,
            ),
            args.context_frames,
            context_tau=args.context_tau,
            sample_steps=args.sample_steps,
            method=DynamicsModel.generate_rollout,
        )

    @jax.jit
    def tf_one(params, video, actions, t, seed):
        video = jnp.asarray(video, jnp.float32)
        mask = (jnp.arange(video.shape[1]) < t)[None, :, None, None]
        primed = jnp.where(mask, video, 0.0)
        ck, sk = jax.random.split(jax.random.key(seed))
        out = model.apply(
            params,
            primed,
            jnp.asarray(actions, jnp.float32),
            jax.random.normal(ck, video.shape, jnp.float32),
            jax.random.normal(sk, (video.shape[0], 1, *video.shape[2:]), jnp.float32),
            t,
            context_tau=args.context_tau,
            sample_steps=args.sample_steps,
            method=DynamicsModel.generate_rollout,
        )
        return jnp.take(out, t, axis=1)

    @jax.jit
    def decode_chunk(latent_chunk):
        return preprocessor.patches_to_images(
            tokenizer.apply(tokenizer_variables, latent_chunk, method=type(tokenizer).decode)
        ).astype(jnp.float32)

    def decode_all(latent: np.ndarray, chunk: int = 64) -> np.ndarray:
        pieces = []
        for start in range(0, len(latent), chunk):
            piece = latent[None, start : start + chunk]
            pad = chunk - piece.shape[1]
            if pad:
                piece = np.pad(piece, ((0, 0), (0, pad), (0, 0), (0, 0)))
            out = np.asarray(jax.device_get(decode_chunk(jnp.asarray(piece, jnp.float32))))[0]
            pieces.append(out[: chunk - pad if pad else chunk])
        return np.concatenate(pieces, axis=0)

    def to_u8(x):
        return np.clip(np.rint(np.clip(x, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for index, mode in ((i, m) for i in (int(i) for i in args.indices.split(",")) for m in action_modes):
        sample = sample_for(index, mode)
        video, actions, total = sample["video"], sample["actions"], sample["total"]
        truth = video[0]

        roll = np.asarray(jax.device_get(rollout_full(params, video, actions, index)))[0]
        tf = truth.copy()
        for t in range(args.context_frames, total):
            tf[t] = np.asarray(
                jax.device_get(tf_one(params, video, actions, t, 10_000 + 100 * index + t))
            )[0]

        rng = np.random.default_rng([7, index])
        noise = rng.standard_normal(truth.shape).astype(np.float32)
        frames = list(range(args.context_frames, total))
        entry = {
            "frames": frames,
            "rollout_err": [float(np.linalg.norm(roll[t] - truth[t])) for t in frames],
            "tf_err": [float(np.linalg.norm(tf[t] - truth[t])) for t in frames],
            "noise_ref": [float(np.linalg.norm(0.1 * noise[t])) for t in frames],
            "latent_norm": [float(np.linalg.norm(truth[t])) for t in frames],
            "persist_err": [float(np.linalg.norm(truth[t] - truth[t - 1])) for t in frames],
        }
        results[f"{index}_{mode}"] = entry
        logger.info(
            "clip %d [%s] | mean err: rollout %.2f  teacher-forced %.2f  copy %.2f  (tf/copy %.3f)",
            index,
            mode,
            np.mean(entry["rollout_err"]),
            np.mean(entry["tf_err"]),
            np.mean(entry["persist_err"]),
            np.mean(entry["tf_err"]) / np.mean(entry["persist_err"]),
        )

        if args.no_video:
            continue

        raw = decode_video_window(
            raw_index[sample["key"]], sample["absolute_start"], sample["span"],
            tuple(preprocessor.resize_shape),
        )[:: args.stride][:total]
        panels = [raw, to_u8(decode_all(tf)), to_u8(decode_all(roll))]
        sep = np.full((raw.shape[1], 4, 3), 255, np.uint8)
        out_frames = []
        for parts in zip(*panels, strict=True):
            row = [parts[0]]
            for panel in parts[1:]:
                row.extend([sep, panel])
            out_frames.append(np.concatenate(row, axis=1))
        imageio.mimsave(
            output_dir / f"tf_{step}_{index:03d}_{mode}_raw-tf-rollout.mp4", out_frames, fps=args.fps
        )

    Path(args.output).write_text(json.dumps({"step": step, "results": results}, indent=2))


if __name__ == "__main__":
    main()
