"""Temporally filter the latents in an existing dynamics dataset.

The tokenizer's per-frame losses leave decoder-null temporal jitter in the
latents (anti-correlated frame-to-frame deltas that the decoder ignores, see
scripts/eval_latent_denoise.py). Filtering the stored latents removes most of
that noise at ~0.2-0.4 dB decoded cost, giving the dynamics model a target
whose frame-to-frame change is mostly signal. No re-encoding needed: the
filter runs directly on the stored records and rewrites the shards.

    uv run python scripts/data/filter_dynamics_latents.py \
        --input_dir data/so101/dyn/train --output_dir data/so101/dyn_g5/train
"""

import argparse
import io
import logging
from pathlib import Path

import grain.python as grain
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

logger = logging.getLogger(__name__)

KERNELS = {
    "g3": np.array([0.25, 0.5, 0.25]),
    "g5": np.array([0.0614, 0.2448, 0.3877, 0.2448, 0.0614]),
}


def filter_time(z: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    r = len(kernel) // 2
    padded = np.concatenate([z[r:0:-1], z, z[-2 : -2 - r : -1]], axis=0)
    out = np.zeros_like(z)
    for k, w in enumerate(kernel):
        out += w * padded[k : k + len(z)]
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Filter latents in dynamics shards.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--kernel", choices=sorted(KERNELS), default="g5")
    parser.add_argument("--records_per_shard", type=int, default=64)
    args = parser.parse_args()

    kernel = KERNELS[args.kernel]
    paths = sorted(str(p) for p in Path(args.input_dir).glob("*.arecord"))
    if not paths:
        raise FileNotFoundError(f"No .arecord files in {args.input_dir}")
    source = grain.ArrayRecordDataSource(paths)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*.arecord")):
        raise FileExistsError(f"{output_dir} already contains shards")

    writer = None
    shard_idx = written = 0
    for i in range(len(source)):
        with np.load(io.BytesIO(source[i])) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
        frames = payload["frames"]
        payload["frames"] = filter_time(frames.astype(np.float32), kernel).astype(frames.dtype)
        buffer = io.BytesIO()
        np.savez(buffer, **payload)
        if writer is None or written % args.records_per_shard == 0:
            if writer is not None:
                writer.close()
            path = output_dir / f"shard-{shard_idx:05d}.arecord"
            writer = ArrayRecordWriter(path.as_posix(), "group_size:1")
            shard_idx += 1
        writer.write(buffer.getvalue())
        written += 1
        if written % 100 == 0:
            logger.info("%d/%d records", written, len(source))
    if writer is not None:
        writer.close()
    logger.info("Wrote %d records to %d shards in %s", written, shard_idx, output_dir)


if __name__ == "__main__":
    main()
