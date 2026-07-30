"""Is the tokenizer's temporal latent jitter decoder-relevant?

Latent deltas in moving regions are anti-correlated frame to frame while both
the raw pixels and the decoded reconstructions are smooth, so the encoder
appears to jitter in directions the decoder ignores. If that is right,
temporally filtering the latents should barely cost reconstruction quality --
and a filtered dataset is a far less noisy prediction target for the dynamics
model. Decodes several filtered variants of the eval latents and scores them
against the raw footage, in the regions that move.

    uv run python scripts/eval_latent_denoise.py \
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

from visionary.common.checkpoint import (
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.dataset import decode_video_window
from visionary.tokenizer_preprocessor import TokenizerPreprocessor

from eval_dynamics_videos import build_raw_index

logger = logging.getLogger(__name__)

FILTERS = {
    "identity": np.array([1.0]),
    "g3": np.array([0.25, 0.5, 0.25]),
    "g5": np.array([0.0614, 0.2448, 0.3877, 0.2448, 0.0614]),
    "box5": np.full(5, 0.2),
}


def filter_time(z: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Temporal convolution with reflect padding, axis 0."""
    r = len(kernel) // 2
    if r == 0:
        return z
    padded = np.concatenate([z[r:0:-1], z, z[-2 : -2 - r : -1]], axis=0)
    out = np.zeros_like(z)
    for k, w in enumerate(kernel):
        out += w * padded[k : k + len(z)]
    return out


def delta_cos(z: np.ndarray) -> float:
    """Temporal coherence of the top-decile moving tokens."""
    d = np.diff(z, axis=0)
    s = np.linalg.norm(d, axis=-1)
    cut = np.quantile(s, 0.9, axis=-1, keepdims=True)
    mov = (s[1:] >= cut[1:]) & (s[:-1] >= cut[:-1])
    cos = (d[1:] * d[:-1]).sum(-1) / (s[1:] * s[:-1] + 1e-6)
    return float(cos[mov].mean())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Decode temporally filtered latents.")
    parser.add_argument("--tokenizer_checkpoint_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--raw_shards_dir", required=True)
    parser.add_argument("--output", default="latent_denoise.json")
    parser.add_argument("--indices", default="0,2,6,7")
    parser.add_argument("--max_frames", type=int, default=192)
    parser.add_argument(
        "--video_dir",
        help="Also write side-by-side mp4s: raw | identity | g3 | g5 | box5, "
        "so the filters can be judged by eye rather than PSNR alone.",
    )
    parser.add_argument("--fps", type=int, default=30)
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
            logger.warning("clip %d: raw decode came back short, skipping", index)
            continue
        mask = moving_pixels(raw)
        raw_f = raw.astype(np.float32) / 255.0
        panels = [raw]
        for name, kernel in FILTERS.items():
            zf = filter_time(z, kernel)
            decoded = np.clip(decode_all(zf), 0.0, 1.0)
            panels.append(np.clip(np.rint(decoded * 255.0), 0, 255).astype(np.uint8))
            err = (decoded - raw_f) ** 2
            psnr_all = float(-10.0 * np.log10(err.mean() + 1e-12))
            psnr_mov = float(-10.0 * np.log10(err.mean(-1)[mask].mean() + 1e-12))
            entry = {
                "index": index,
                "filter": name,
                "psnr_all": psnr_all,
                "psnr_moving": psnr_mov,
                "latent_delta_cos": delta_cos(zf),
            }
            results.append(entry)
            logger.info(
                "clip %d %-8s | psnr all %.2f moving %.2f | delta-cos %+.3f",
                index, name, psnr_all, psnr_mov, entry["latent_delta_cos"],
            )
        if args.video_dir:
            video_dir = Path(args.video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            separator = np.full((panels[0].shape[1], 4, 3), 255, dtype=np.uint8)
            frames = []
            for parts in zip(*panels, strict=True):
                row = [parts[0]]
                for panel in parts[1:]:
                    row.extend([separator, panel])
                frames.append(np.concatenate(row, axis=1))
            path = video_dir / f"denoise_{index:02d}_raw-id-g3-g5-box5.mp4"
            imageio.mimsave(path, frames, fps=args.fps)
            logger.info("wrote %s", path)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print("\nfilter | mean psnr moving (all clips) | mean delta-cos")
    for name in FILTERS:
        vals = [r for r in results if r["filter"] == name]
        if vals:
            print(
                f"{name:8s} | {np.mean([r['psnr_moving'] for r in vals]):6.2f} dB "
                f"| {np.mean([r['latent_delta_cos'] for r in vals]):+.3f}"
            )


if __name__ == "__main__":
    main()
