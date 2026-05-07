import argparse
import hashlib
import io
import itertools
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter
from etils import epath

logger = logging.getLogger(__name__)


def _normalize_walk_dirpath(root: epath.Path, dirpath: epath.Path) -> epath.Path:
    root_parts = root.parts
    dirpath_str = dirpath.as_posix()
    if root_parts[:2] == ("/", "gs") and not dirpath_str.startswith(("gs://", "/gs/")):
        return epath.Path(f"/gs/{dirpath_str.lstrip('/')}")
    return dirpath


def list_episode_files(data_dir: str) -> list[epath.Path]:
    root = epath.Path(data_dir)
    files = []
    for dirpath, _, filenames in root.walk():
        dirpath = _normalize_walk_dirpath(root, dirpath)
        for filename in filenames:
            if filename.endswith(".npz"):
                files.append(dirpath / filename)
    return sorted(files, key=lambda p: p.as_posix())


def split_files(
    files: list[epath.Path], eval_ratio: float, seed: int
) -> tuple[list[epath.Path], list[epath.Path]]:
    train, eval_ = [], []
    for path in files:
        digest = hashlib.sha256(f"{seed}:{path.as_posix()}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], byteorder="big") / 2**64
        (eval_ if bucket < eval_ratio else train).append(path)
    return train, eval_


def chunk_starts(length: int, chunk_length: int, overlap: int) -> list[int]:
    if length <= chunk_length:
        return [0]
    stride = chunk_length - overlap
    starts = list(range(0, length - chunk_length + 1, stride))
    last = length - chunk_length
    if starts[-1] != last:
        starts.append(last)
    return starts


def encode_record(arrays: dict[str, np.ndarray], start: int, stop: int) -> bytes:
    frames = arrays["frames"]
    payload: dict[str, np.ndarray] = {"frames": frames[start:stop]}
    for key, value in arrays.items():
        if key == "frames":
            continue
        if value.ndim > 0 and value.shape[0] == frames.shape[0]:
            payload[key] = value[start:stop]
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    return buffer.getvalue()


def encode_episode(
    path: str, chunk_length: int, overlap: int, frame_length: int
) -> tuple[list[bytes], bool]:
    with open(path, "rb") as f, np.load(f) as episode:
        arrays = {key: np.asarray(episode[key]) for key in episode.files}
    length = int(arrays["frames"].shape[0])
    if length < frame_length:
        return [], True

    records = []
    for start in chunk_starts(length, chunk_length, overlap):
        stop = min(start + chunk_length, length)
        if stop - start < frame_length:
            continue
        records.append(encode_record(arrays, start, stop))
    return records, False


def iter_encoded_episodes(
    files: list[epath.Path], chunk_length: int, overlap: int, frame_length: int, workers: int
):
    paths = [path.as_posix() for path in files]
    if workers == 1:
        for path in paths:
            yield encode_episode(path, chunk_length, overlap, frame_length)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(
            encode_episode,
            paths,
            itertools.repeat(chunk_length),
            itertools.repeat(overlap),
            itertools.repeat(frame_length),
        )


def write_shards(
    files: list[epath.Path],
    *,
    output_dir: epath.Path,
    chunk_length: int,
    overlap: int,
    frame_length: int,
    records_per_shard: int,
    workers: int,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    skipped = 0
    shard_idx = 0
    records_in_shard = 0
    shard_path = output_dir / f"shard-{shard_idx:05d}.arecord"
    writer = ArrayRecordWriter(shard_path.as_posix(), "group_size:1")

    for records, skipped_episode in iter_encoded_episodes(
        files, chunk_length, overlap, frame_length, workers
    ):
        skipped += int(skipped_episode)
        for record in records:
            writer.write(record)
            total += 1
            records_in_shard += 1
            if records_in_shard == records_per_shard:
                writer.close()
                logger.info("Wrote shard %s (%d total records)", shard_path, total)
                shard_idx += 1
                records_in_shard = 0
                shard_path = output_dir / f"shard-{shard_idx:05d}.arecord"
                writer = ArrayRecordWriter(shard_path.as_posix(), "group_size:1")

    writer.close()
    if records_in_shard:
        logger.info("Wrote shard %s (%d total records)", shard_path, total)
    else:
        shard_path.unlink(missing_ok=True)
    return total, skipped


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Convert NPZ episodes to chunked ArrayRecord")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--frame_length", type=int, default=32)
    parser.add_argument("--chunk_length", type=int, default=64)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--records_per_shard", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max_files", type=int)
    args = parser.parse_args()

    overlap = 0 if args.overlap is None else args.overlap
    if not 0 <= overlap < args.chunk_length:
        raise ValueError(
            f"Expected 0 <= overlap < chunk_length, got {overlap=} {args.chunk_length=}"
        )
    if args.frame_length > args.chunk_length:
        raise ValueError(
            f"frame_length={args.frame_length} must be <= chunk_length={args.chunk_length}"
        )

    logger.info("Listing episodes from %s", args.input_dir)
    files = list_episode_files(args.input_dir)
    if args.max_files is not None:
        files = files[: args.max_files]
        logger.info("Limiting conversion to first %d episodes", len(files))
    logger.info("Found %d episodes", len(files))

    train_files, eval_files = split_files(files, args.eval_ratio, args.seed)
    logger.info(
        "Split: %d train, %d eval, workers=%d", len(train_files), len(eval_files), args.workers
    )

    output = epath.Path(args.output_dir)
    n_train, skipped_train = write_shards(
        train_files,
        output_dir=output / "train",
        chunk_length=args.chunk_length,
        overlap=overlap,
        frame_length=args.frame_length,
        records_per_shard=args.records_per_shard,
        workers=args.workers,
    )
    n_eval, skipped_eval = write_shards(
        eval_files,
        output_dir=output / "eval",
        chunk_length=args.chunk_length,
        overlap=overlap,
        frame_length=args.frame_length,
        records_per_shard=args.records_per_shard,
        workers=args.workers,
    )
    logger.info(
        "Done. Wrote %d train and %d eval chunked records "
        "(skipped %d train and %d eval episodes shorter than frame_length).",
        n_train,
        n_eval,
        skipped_train,
        skipped_eval,
    )


if __name__ == "__main__":
    main()
