"""Teacher-forced vs rollout probe, scored in latent or pixel space.

--space latent: per-frame latent error norms for a normal autoregressive
rollout and a 1-step teacher-forced pass (every frame t generated from fully
real context), with the 0.1-scaled context-noise norm as the trained-tolerance
reference. Video: raw | tf | rollout.

--space pixel: latent L2 near the jitter floor cannot distinguish "copied the
background" from "predicted everything predictable", so score decoded frames
against the raw footage restricted to regions that actually move:

    recon   encode->decode of the GT frame       (tokenizer ceiling)
    tf      1-step teacher-forced prediction     (all-real context)
    roll    autoregressive rollout
    copy    decode of the previous GT latent     (copy-last-frame baseline)

Video: raw | recon | tf | roll, plus a contact sheet of the frames where tf
falls furthest below the tokenizer ceiling.

    uv run python scripts/analysis/eval_teacher_forced.py \
        --space pixel \
        --checkpoint_dir gs://.../so101_dynamics_small_ftg5 \
        --tokenizer_checkpoint_dir gs://.../so101_tokenizer \
        --data_dir data/so101/dyn_g5/eval --raw_shards_dir data/so101/shards/eval \
        --output_dir artifacts/ftg5_compare/ftg5 --label ftg5
"""

import argparse
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
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.dataset import decode_video_window
from visionary.models.dreamer4.dynamics import DynamicsModel
from visionary.eval.loading import build_raw_index, load_train_config, restore_params
from visionary.models.dreamer4.tokenizer_preprocessor import TokenizerPreprocessor

logger = logging.getLogger(__name__)


def moving_mask(raw: np.ndarray, thresh: float = 10.0, dilate: int = 5) -> np.ndarray:
    """Per-frame mask of pixels that changed since the previous strided frame.

    Covers both where content arrived and where it left, so a disappearing
    object stays inside the mask. First frame reuses the second's mask.
    """
    gray = raw.astype(np.float32).mean(axis=-1)
    diff = np.abs(np.diff(gray, axis=0))
    kernel = np.ones((dilate, dilate), np.uint8)
    masks = np.zeros(raw.shape[:3], bool)
    for t in range(1, len(raw)):
        m = (cv2.GaussianBlur(diff[t - 1], (5, 5), 0) > thresh).astype(np.uint8)
        masks[t] = cv2.dilate(m, kernel).astype(bool)
    masks[0] = masks[1]
    return masks


def masked_psnr(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float | None:
    if mask.sum() < 16:
        return None
    err = ((pred.astype(np.float32) - ref.astype(np.float32)) / 255.0) ** 2
    mse = float(err[mask].mean())
    return 10.0 * float(np.log10(1.0 / max(mse, 1e-10)))


def full_psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    err = ((pred.astype(np.float32) - ref.astype(np.float32)) / 255.0) ** 2
    return 10.0 * float(np.log10(1.0 / max(float(err.mean()), 1e-10)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Teacher-forced vs rollout probe.")
    parser.add_argument("--space", choices=("latent", "pixel"), default="latent")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--tokenizer_checkpoint_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--raw_shards_dir")
    parser.add_argument("--output_dir", default="tf_videos")
    parser.add_argument("--output", help="Metrics JSON path; defaults inside output_dir.")
    parser.add_argument("--label", default="tf", help="Tag prefixed to every output file.")
    parser.add_argument("--step", type=int)
    parser.add_argument("--from_export", action="store_true")
    parser.add_argument("--indices", default="2,7,256")
    parser.add_argument("--context_frames", type=int, default=4)
    parser.add_argument("--total_frames", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--context_tau", type=float, default=0.9)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--action_modes", default="real")
    parser.add_argument("--offset_shift", type=int, default=0)
    args = parser.parse_args()
    action_modes = args.action_modes.split(",")

    need_raw = args.space == "pixel" or not args.no_video
    if need_raw and not args.raw_shards_dir:
        parser.error("--raw_shards_dir is required unless --space latent --no_video")

    latents = grain.ArrayRecordDataSource(sorted(str(p) for p in Path(args.data_dir).glob("*.arecord")))
    raw_index = build_raw_index(args.raw_shards_dir) if need_raw else None

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
    if args.from_export:
        export_cfg, params = restore_model_export_single_device(args.checkpoint_dir, step=args.step)
        model = instantiate(export_cfg)
        step = args.step
        logger.info("Restored dynamics export from step %s", step)
    else:
        cfg = load_train_config(args.checkpoint_dir)
        model, params, step = restore_params(
            cfg, args.checkpoint_dir, args.step, {"video": first["video"], "actions": first["actions"]}
        )
        logger.info("Restored dynamics params from step %d", step)

    tokenizer_cfg, tokenizer_variables = restore_model_export_single_device(args.tokenizer_checkpoint_dir)
    tokenizer = instantiate(tokenizer_cfg)
    preprocessor = TokenizerPreprocessor.from_config(restore_preprocessor_export(args.tokenizer_checkpoint_dir))

    @jax.jit
    def rollout_full(params, video, actions, seed):
        video = jnp.asarray(video, jnp.float32)
        primed = jnp.zeros_like(video).at[:, : args.context_frames].set(video[:, : args.context_frames])
        ck, sk = jax.random.split(jax.random.key(seed))
        return model.apply(
            params,
            primed,
            jnp.asarray(actions, jnp.float32),
            jax.random.normal(ck, video.shape, jnp.float32),
            jax.random.normal(
                sk,
                (video.shape[0], video.shape[1] - args.context_frames, *video.shape[2:]),
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

    def write_panel_video(panels, path):
        sep = np.full((panels[0].shape[1], 4, 3), 255, np.uint8)
        out_frames = []
        for parts in zip(*panels, strict=True):
            row = [parts[0]]
            for panel in parts[1:]:
                row.extend([sep, panel])
            out_frames.append(np.concatenate(row, axis=1))
        imageio.mimsave(path, out_frames, fps=args.fps)

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
            tf[t] = np.asarray(jax.device_get(tf_one(params, video, actions, t, 10_000 + 100 * index + t)))[0]

        frames = list(range(args.context_frames, total))
        raw = None
        if need_raw:
            raw = decode_video_window(
                raw_index[sample["key"]],
                sample["absolute_start"],
                sample["span"],
                tuple(preprocessor.resize_shape),
            )[:: args.stride][:total]

        if args.space == "latent":
            rng = np.random.default_rng([7, index])
            noise = rng.standard_normal(truth.shape).astype(np.float32)
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
            if not args.no_video:
                write_panel_video(
                    [raw, to_u8(decode_all(tf)), to_u8(decode_all(roll))],
                    output_dir / f"{args.label}_{step}_{index:03d}_{mode}_raw-tf-rollout.mp4",
                )
            continue

        recon = to_u8(decode_all(truth))
        tf_px = to_u8(decode_all(tf))
        roll_px = to_u8(decode_all(roll))
        masks = moving_mask(raw)
        entry = {
            "stream": f"{sample['key'][0]}/ep{sample['key'][1]}/{sample['key'][2]}",
            "frames": frames,
            "mask_frac": [float(masks[t].mean()) for t in frames],
            "psnr_moving": {
                "recon": [masked_psnr(recon[t], raw[t], masks[t]) for t in frames],
                "tf": [masked_psnr(tf_px[t], raw[t], masks[t]) for t in frames],
                "roll": [masked_psnr(roll_px[t], raw[t], masks[t]) for t in frames],
                "copy": [masked_psnr(recon[t - 1], raw[t], masks[t]) for t in frames],
            },
            "psnr_full": {
                "recon": [full_psnr(recon[t], raw[t]) for t in frames],
                "tf": [full_psnr(tf_px[t], raw[t]) for t in frames],
                "roll": [full_psnr(roll_px[t], raw[t]) for t in frames],
                "copy": [full_psnr(recon[t - 1], raw[t]) for t in frames],
            },
            "latent_tf_err": [float(np.linalg.norm(tf[t] - truth[t])) for t in frames],
            "latent_persist_err": [float(np.linalg.norm(truth[t] - truth[t - 1])) for t in frames],
        }
        results[f"{index}_{mode}"] = entry

        def mean_of(vals):
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else float("nan")

        logger.info(
            "clip %d [%s] | moving-psnr  recon %.2f  tf %.2f  roll %.2f  copy %.2f",
            index,
            mode,
            *(mean_of(entry["psnr_moving"][k]) for k in ("recon", "tf", "roll", "copy")),
        )

        if not args.no_video:
            write_panel_video(
                [raw, recon, tf_px, roll_px],
                output_dir / f"{args.label}_{step}_{index:03d}_{mode}_raw-recon-tf-roll.mp4",
            )

        # contact sheet: frames where tf falls furthest below the tokenizer
        # ceiling on moving pixels — where content the tokenizer can carry
        # was lost by the dynamics model
        gaps = [
            (r - p, t)
            for t, r, p in zip(frames, entry["psnr_moving"]["recon"], entry["psnr_moving"]["tf"])
            if r is not None and p is not None
        ]
        worst = [t for _, t in sorted(gaps, reverse=True)[:6]]
        worst.sort()
        if worst:
            rows = []
            for name, imgs in (("raw", raw), ("recon", recon), ("tf", tf_px), ("roll", roll_px)):
                cells = []
                for t in worst:
                    cell = imgs[t].copy()
                    cv2.putText(
                        cell, f"{name} t={t}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA
                    )
                    cells.append(cell)
                rows.append(np.concatenate(cells, axis=1))
            sheet = np.concatenate(rows, axis=0)
            imageio.imwrite(output_dir / f"{args.label}_{step}_{index:03d}_{mode}_worstframes.png", sheet)

    out = {
        "label": args.label,
        "space": args.space,
        "checkpoint_dir": args.checkpoint_dir,
        "step": step,
        "data_dir": args.data_dir,
        "sample_steps": args.sample_steps,
        "context_tau": args.context_tau,
        "stride": args.stride,
        "seed": args.seed,
        "results": results,
    }
    out_path = Path(args.output) if args.output else output_dir / f"{args.label}_{args.space}_tf.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
