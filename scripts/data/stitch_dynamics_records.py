"""Reassemble full episodes from chunked dynamics records.

The dataset was built with chunk_length=512, which caps every record at 512
frames. Strided training needs longer spans: a 128-frame window at stride 3
covers 382 frames. Records carry episode_id/start_index and chunks of one
episode are adjacent in record order, so this streams shard by shard, merges
each episode's chunks (deduping the right-aligned overlap of the final chunk),
and rewrites records capped at --max_length. Nothing is re-encoded.

Also writes lengths.json (frame count per record, in record order) so loaders
can restrict long-window sampling to records that fit without scanning shards.

    uv run python scripts/data/stitch_dynamics_records.py \
        --input_dir data/so101/dyn/train --output_dir data/so101/dyn_long/train
"""

import argparse
import io
import json
import logging
from pathlib import Path

import grain.python as grain
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

logger = logging.getLogger(__name__)

PER_FRAME_KEYS = ("frames", "actions", "state", "rewards")


def group_key(payload: dict) -> tuple:
    return tuple(str(payload[k]) for k in ("repo", "episode", "camera", "augment_copy") if k in payload) + (
        int(payload["episode_id"]),
    )


def merge(chunks: list[dict]) -> dict:
    chunks = sorted(chunks, key=lambda c: int(c["start_index"]))
    base = int(chunks[0]["start_index"])
    total = max(int(c["start_index"]) - base + len(c["frames"]) for c in chunks)
    merged = dict(chunks[0])
    for key in PER_FRAME_KEYS:
        if key not in chunks[0]:
            continue
        out = np.empty((total, *chunks[0][key].shape[1:]), dtype=chunks[0][key].dtype)
        for c in chunks:
            off = int(c["start_index"]) - base
            out[off : off + len(c[key])] = c[key]
        merged[key] = out
    return merged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Stitch chunked dynamics records.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--min_length", type=int, default=94)
    parser.add_argument("--records_per_shard", type=int, default=64)
    args = parser.parse_args()

    source = grain.ArrayRecordDataSource(sorted(str(p) for p in Path(args.input_dir).glob("*.arecord")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*.arecord")):
        raise FileExistsError(f"{output_dir} already contains shards")

    writer = None
    shard_idx = written = dropped = 0
    lengths: list[int] = []

    def emit(payload: dict) -> None:
        nonlocal writer, shard_idx, written, dropped
        episode = payload
        length = len(episode["frames"])
        for start in range(0, length, args.max_length):
            stop = min(start + args.max_length, length)
            if stop - start < args.min_length:
                if start == 0:
                    dropped += 1
                continue
            record = dict(episode)
            for key in PER_FRAME_KEYS:
                if key in record:
                    record[key] = episode[key][start:stop]
            record["start_index"] = np.asarray(int(episode["start_index"]) + start, dtype=np.int32)
            if start > 0:
                record["prev_action"] = np.asarray(episode["actions"][start - 1], dtype=episode["actions"].dtype)
            buffer = io.BytesIO()
            np.savez(buffer, **record)
            if writer is None or written % args.records_per_shard == 0:
                if writer is not None:
                    writer.close()
                path = output_dir / f"shard-{shard_idx:05d}.arecord"
                writer = ArrayRecordWriter(path.as_posix(), "group_size:1")
                shard_idx += 1
            writer.write(buffer.getvalue())
            lengths.append(stop - start)
            written += 1

    current_key = None
    current_chunks: list[dict] = []
    for i in range(len(source)):
        with np.load(io.BytesIO(source[i])) as data:
            payload = {k: np.asarray(data[k]) for k in data.files}
        key = group_key(payload)
        if key != current_key and current_chunks:
            emit(merge(current_chunks))
            current_chunks = []
        current_key = key
        current_chunks.append(payload)
        if (i + 1) % 500 == 0:
            logger.info("%d/%d records read, %d written", i + 1, len(source), written)
    if current_chunks:
        emit(merge(current_chunks))
    if writer is not None:
        writer.close()

    (output_dir / "lengths.json").write_text(json.dumps(lengths))
    arr = np.array(lengths)
    logger.info(
        "Wrote %d records (%d shards, %d dropped short) | frames min %d median %d max %d",
        written,
        shard_idx,
        dropped,
        arr.min(),
        int(np.median(arr)),
        arr.max(),
    )


if __name__ == "__main__":
    main()
