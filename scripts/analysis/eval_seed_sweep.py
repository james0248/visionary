"""Seed x sample-steps sweep on teacher-forced grasp frames.

Diagnostic for the disappearing-held-object failure: at carry frames (gripper
closed, joints moving — selected from the action stream), generate the same
teacher-forced frame under several noise seeds and denoise step counts. If the
object materializes in some samples, the model has the information and the fix
is sampling/training-length; if it never appears, the failure is learned and
data-side levers are needed.

Writes one contact sheet per (clip, frame): header row = raw t-1, raw t,
recon t; then one row per sample_steps setting, one column per seed. Also a
JSON of per-sample latent errors.

    uv run python scripts/analysis/eval_seed_sweep.py \
        --checkpoint_dir gs://.../so101_dynamics_small_ftg5 \
        --tokenizer_checkpoint_dir gs://.../so101_tokenizer \
        --data_dir data/so101/dyn_g5/eval --raw_shards_dir data/so101/shards/eval \
        --output_dir artifacts/seed_sweep_ftg5
"""

import argparse
import functools
import io
import json
import logging
from pathlib import Path

import cv2
import grain.python as grain
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate

from visionary.common.checkpoint import (
    resolve_model_export_step,
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.dataset import decode_video_window
from visionary.eval.loading import build_raw_index
from visionary.models.dreamer4.dynamics import DynamicsModel
from visionary.models.dreamer4.tokenizer_preprocessor import TokenizerPreprocessor

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Seed/sample-steps sweep at grasp frames.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--tokenizer_checkpoint_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--raw_shards_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--indices", default="2,256,259")
    parser.add_argument("--num_seeds", type=int, default=8)
    parser.add_argument("--steps_list", default="4,8,16")
    parser.add_argument("--frames_per_clip", type=int, default=6)
    parser.add_argument("--context_frames", type=int, default=4)
    parser.add_argument("--total_frames", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--context_tau", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    steps_list = [int(s) for s in args.steps_list.split(",")]

    latents = grain.ArrayRecordDataSource(sorted(str(p) for p in Path(args.data_dir).glob("*.arecord")))
    raw_index = build_raw_index(args.raw_shards_dir)

    def sample_for(index: int):
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
        indices = offset + np.arange(total) * args.stride
        aligned = actions[indices - 1]
        if offset == 0:
            aligned[0] = prev_action
        return {
            "video": np.asarray(video[indices], dtype=np.float32)[None],
            "actions": np.asarray(aligned, dtype=np.float32)[None],
            "key": key,
            "absolute_start": record_start + offset,
            "span": span,
            "total": total,
        }

    def carry_frames(actions: np.ndarray, total: int) -> list[int]:
        """Frames in the generation range where the gripper is closed and the
        arm is moving; evenly thinned to frames_per_clip. Falls back to evenly
        spaced frames when the heuristic finds too few."""
        a = actions[0]
        grip = a[:, -1]
        vel = np.abs(np.diff(a[:, :-1], axis=0)).mean(axis=1)
        closed = grip[1:] < np.median(grip)
        moving = vel > np.median(vel)
        cands = [t for t in range(args.context_frames, total) if closed[t - 1] and moving[t - 1]]
        if len(cands) < 2:
            cands = list(range(args.context_frames, total))
        picks = np.linspace(0, len(cands) - 1, min(args.frames_per_clip, len(cands)))
        return sorted({cands[int(round(p))] for p in picks})

    step = resolve_model_export_step(args.checkpoint_dir, args.step)
    export_cfg, params = restore_model_export_single_device(args.checkpoint_dir, step=step)
    model = instantiate(export_cfg)
    logger.info("Restored dynamics export from step %d", step)

    tokenizer_cfg, tokenizer_variables = restore_model_export_single_device(args.tokenizer_checkpoint_dir)
    tokenizer = instantiate(tokenizer_cfg)
    preprocessor = TokenizerPreprocessor.from_config(restore_preprocessor_export(args.tokenizer_checkpoint_dir))

    @functools.partial(jax.jit, static_argnames=("sample_steps",))
    def tf_one(params, video, actions, t, seed, sample_steps):
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
            sample_steps=sample_steps,
            method=DynamicsModel.generate_rollout,
        )
        return jnp.take(out, t, axis=1)

    @jax.jit
    def decode_chunk(latent_chunk):
        return preprocessor.patches_to_images(
            tokenizer.apply(tokenizer_variables, latent_chunk, method=type(tokenizer).decode)
        ).astype(jnp.float32)

    def decode_batch(lat: np.ndarray, chunk: int = 32) -> np.ndarray:
        pieces = []
        for start in range(0, len(lat), chunk):
            piece = lat[None, start : start + chunk]
            pad = chunk - piece.shape[1]
            if pad:
                piece = np.pad(piece, ((0, 0), (0, pad), (0, 0), (0, 0)))
            out = np.asarray(jax.device_get(decode_chunk(jnp.asarray(piece, jnp.float32))))[0]
            pieces.append(out[: chunk - pad if pad else chunk])
        return np.concatenate(pieces, axis=0)

    def to_u8(x):
        return np.clip(np.rint(np.clip(x, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)

    def label(img, text):
        img = img.copy()
        cv2.putText(img, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        return img

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"step": step, "num_seeds": args.num_seeds, "steps_list": steps_list, "clips": {}}
    for index in (int(i) for i in args.indices.split(",")):
        sample = sample_for(index)
        video, actions, total = sample["video"], sample["actions"], sample["total"]
        truth = video[0]
        frames = carry_frames(actions, total)
        logger.info("clip %d: carry frames %s", index, frames)

        raw = decode_video_window(
            raw_index[sample["key"]],
            sample["absolute_start"],
            sample["span"],
            tuple(preprocessor.resize_shape),
        )[:: args.stride][:total]
        recon = to_u8(decode_batch(truth[np.array(frames)]))

        clip_report = {}
        for fi, t in enumerate(frames):
            lat = []
            errs = {}
            for ss in steps_list:
                for s in range(args.num_seeds):
                    z = np.asarray(
                        jax.device_get(tf_one(params, video, actions, t, 50_000 + 1000 * index + 10 * t + s, ss))
                    )[0]
                    lat.append(z)
                    errs[f"ss{ss}_seed{s}"] = float(np.linalg.norm(z - truth[t]))
            px = to_u8(decode_batch(np.stack(lat)))

            header = [
                label(raw[t - 1], f"raw t-1={t - 1}"),
                label(raw[t], f"raw t={t}"),
                label(recon[fi], f"recon t={t}"),
            ] + [np.zeros_like(raw[0])] * (args.num_seeds - 3)
            rows = [np.concatenate(header[: args.num_seeds], axis=1)]
            k = 0
            for ss in steps_list:
                cells = []
                for s in range(args.num_seeds):
                    cells.append(label(px[k], f"ss={ss} seed={s}"))
                    k += 1
                rows.append(np.concatenate(cells, axis=1))
            sheet = np.concatenate(rows, axis=0)
            imageio.imwrite(output_dir / f"sweep_{step}_{index:03d}_t{t:02d}.png", sheet)
            clip_report[str(t)] = errs
        report["clips"][str(index)] = {
            "stream": f"{sample['key'][0]}/ep{sample['key'][1]}/{sample['key'][2]}",
            "frames": frames,
            "errors": clip_report,
        }
        logger.info("clip %d done (%d frames x %d samples)", index, len(frames), len(steps_list) * args.num_seeds)

    (output_dir / "seed_sweep.json").write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", output_dir / "seed_sweep.json")


if __name__ == "__main__":
    main()
