"""Decode noised latents to see how robust the latent space is to noise.

The dynamics model conditions on context mixed as tau*z + (1-tau)*noise
(tau=0.9 at inference). Decoding those mixtures shows directly what scene
evidence survives the corruption -- in particular whether small objects like
a carried duck are still legible to anything reading the latents.

    uv run python scripts/analysis/eval_latent_noise.py \
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
from visionary.models.dreamer4.tokenizer_preprocessor import TokenizerPreprocessor

from eval_dynamics_videos import build_raw_index

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Decode noised latents.")
    parser.add_argument("--tokenizer_checkpoint_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--raw_shards_dir", required=True)
    parser.add_argument("--output", default="latent_noise.json")
    parser.add_argument("--indices", default="2,7,256")
    parser.add_argument("--taus", default="1.0,0.95,0.9,0.8")
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--video_dir")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer_cfg, tokenizer_variables = restore_model_export_single_device(
        args.tokenizer_checkpoint_dir
    )
    tokenizer = instantiate(tokenizer_cfg)
    preprocessor = TokenizerPreprocessor.from_config(
        restore_preprocessor_export(args.tokenizer_checkpoint_dir)
    )
    latents = grain.ArrayRecordDataSource(
        sorted(str(p) for p in Path(args.data_dir).glob("*.arecord"))
    )
    raw_index = build_raw_index(args.raw_shards_dir)
    taus = [float(t) for t in args.taus.split(",")]

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

    def moving_pixels(raw: np.ndarray) -> np.ndarray:
        delta = np.abs(np.diff(raw.astype(np.float32), axis=0, prepend=raw[:1])).max(-1)
        mask = delta > 12.0
        for _ in range(8):
            mask |= (
                np.roll(mask, 1, 1) | np.roll(mask, -1, 1)
                | np.roll(mask, 1, 2) | np.roll(mask, -1, 2)
            )
        return mask

    results = []
    for index in (int(i) for i in args.indices.split(",")):
        with np.load(io.BytesIO(latents[index % len(latents)])) as data:
            z = np.asarray(data["frames"], np.float32)
            key = (str(data["repo"]), int(data["episode"]), str(data["camera"]))
            record_start = int(data["start_index"])
        total = min(args.max_frames, len(z))
        z = z[:total]
        raw = decode_video_window(
            raw_index[key], record_start, total, tuple(preprocessor.resize_shape)
        )
        if len(raw) != total:
            logger.warning("clip %d: raw decode short, skipping", index)
            continue
        mask = moving_pixels(raw)
        raw_f = raw.astype(np.float32) / 255.0
        rng = np.random.default_rng([args.seed, index])
        noise = rng.standard_normal(z.shape).astype(np.float32)
        panels = [raw]
        for tau in taus:
            zt = tau * z + (1.0 - tau) * noise
            decoded = np.clip(decode_all(zt), 0.0, 1.0)
            panels.append(np.clip(np.rint(decoded * 255.0), 0, 255).astype(np.uint8))
            err = (decoded - raw_f) ** 2
            entry = {
                "index": index,
                "tau": tau,
                "psnr_all": float(-10.0 * np.log10(err.mean() + 1e-12)),
                "psnr_moving": float(-10.0 * np.log10(err.mean(-1)[mask].mean() + 1e-12)),
            }
            results.append(entry)
            logger.info(
                "clip %d tau %.2f | psnr all %.2f moving %.2f",
                index, tau, entry["psnr_all"], entry["psnr_moving"],
            )
        if args.video_dir:
            video_dir = Path(args.video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            sep = np.full((raw.shape[1], 4, 3), 255, np.uint8)
            frames = []
            for parts in zip(*panels, strict=True):
                row = [parts[0]]
                for panel in parts[1:]:
                    row.extend([sep, panel])
                frames.append(np.concatenate(row, axis=1))
            tag = "-".join(f"{t:g}" for t in taus)
            imageio.mimsave(
                video_dir / f"noise_{index:03d}_raw-tau{tag}.mp4", frames, fps=args.fps
            )

    Path(args.output).write_text(json.dumps(results, indent=2))
    print("\ntau  | mean psnr moving (all clips)")
    for tau in taus:
        vals = [r["psnr_moving"] for r in results if r["tau"] == tau]
        if vals:
            print(f"{tau:4g} | {np.mean(vals):6.2f} dB")


if __name__ == "__main__":
    main()
