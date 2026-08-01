"""Decode temporally strided latent sequences.

The tokenizer's temporal attention was trained on 30fps sequences; a dynamics
model trained on strided latents will feed the decoder sequences at 15/10/7.5
fps, a distribution the decoder never saw. Decodes eval latents at several
strides, scores them against the matching raw frames, and writes side-by-side
videos so the degradation (if any) can be judged by eye.

    uv run python scripts/analysis/eval_latent_stride.py \
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
from visionary.tokenizer_preprocessor import TokenizerPreprocessor

from eval_dynamics_videos import build_raw_index

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Decode strided latent sequences.")
    parser.add_argument("--tokenizer_checkpoint_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--raw_shards_dir", required=True)
    parser.add_argument("--output", default="latent_stride.json")
    parser.add_argument("--indices", default="0,2,6,7")
    parser.add_argument("--strides", default="1,2,3,4")
    parser.add_argument("--frames_per_stride", type=int, default=64)
    parser.add_argument("--video_dir")
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
    strides = [int(s) for s in args.strides.split(",")]
    n = args.frames_per_stride

    @jax.jit
    def decode_chunk(latent_chunk):
        return preprocessor.patches_to_images(
            tokenizer.apply(tokenizer_variables, latent_chunk, method=type(tokenizer).decode)
        ).astype(jnp.float32)

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
        total = min(len(z), n * max(strides))
        raw = decode_video_window(
            raw_index[key], record_start, total, tuple(preprocessor.resize_shape)
        )
        if len(raw) != total:
            logger.warning("clip %d: raw decode short, skipping", index)
            continue
        for k in strides:
            idx = np.arange(min(n, len(z) // k)) * k
            zk = z[idx]
            decoded = np.asarray(
                jax.device_get(decode_chunk(jnp.asarray(zk[None], jnp.float32)))
            )[0]
            decoded = np.clip(decoded, 0.0, 1.0)
            raw_k = raw[idx]
            mask = moving_pixels(raw_k)
            err = (decoded - raw_k.astype(np.float32) / 255.0) ** 2
            psnr_all = float(-10.0 * np.log10(err.mean() + 1e-12))
            psnr_mov = float(-10.0 * np.log10(err.mean(-1)[mask].mean() + 1e-12))
            results.append(
                {"index": index, "stride": k, "psnr_all": psnr_all, "psnr_moving": psnr_mov}
            )
            logger.info(
                "clip %d stride %d | psnr all %.2f moving %.2f", index, k, psnr_all, psnr_mov
            )
            if args.video_dir:
                video_dir = Path(args.video_dir)
                video_dir.mkdir(parents=True, exist_ok=True)
                sep = np.full((raw_k.shape[1], 4, 3), 255, np.uint8)
                dec_u8 = np.clip(np.rint(decoded * 255.0), 0, 255).astype(np.uint8)
                frames = [np.concatenate([a, sep, b], axis=1) for a, b in zip(raw_k, dec_u8)]
                imageio.mimsave(
                    video_dir / f"stride_decode_{index:02d}_k{k}.mp4",
                    frames,
                    fps=max(round(30 / k), 1),
                )

    Path(args.output).write_text(json.dumps(results, indent=2))
    print("\nstride | mean psnr moving (all clips)")
    for k in strides:
        vals = [r["psnr_moving"] for r in results if r["stride"] == k]
        if vals:
            print(f"{k:6d} | {np.mean(vals):6.2f} dB")


if __name__ == "__main__":
    main()
