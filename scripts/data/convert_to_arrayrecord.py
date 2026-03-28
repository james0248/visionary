import argparse
import hashlib
import io
import logging

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
    episode_files: list[epath.Path] = []
    for dirpath, _, filenames in root.walk():
        dirpath = _normalize_walk_dirpath(root, dirpath)
        for filename in filenames:
            if filename.endswith(".npz"):
                episode_files.append(dirpath / filename)
    return sorted(episode_files, key=lambda p: p.as_posix())


def split_files(
    files: list[epath.Path], eval_ratio: float, seed: int
) -> tuple[list[epath.Path], list[epath.Path]]:
    train, eval_ = [], []
    for path in files:
        digest = hashlib.sha256(f"{seed}:{path.as_posix()}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], byteorder="big") / 2**64
        if bucket < eval_ratio:
            eval_.append(path)
        else:
            train.append(path)
    return train, eval_


def serialize_episode(path: epath.Path) -> bytes:
    with path.open("rb") as f:
        with np.load(f) as data:
            buf = io.BytesIO()
            np.savez(buf, **{k: data[k] for k in data.files})
            return buf.getvalue()


def write_shards(
    files: list[epath.Path],
    output_dir: epath.Path,
    episodes_per_shard: int,
    seed: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(files))

    shard_idx = 0
    total = 0
    for start in range(0, len(indices), episodes_per_shard):
        batch_indices = indices[start : start + episodes_per_shard]
        shard_path = output_dir / f"shard-{shard_idx:05d}.arecord"

        writer = ArrayRecordWriter(shard_path.as_posix())
        for idx in batch_indices:
            record = serialize_episode(files[idx])
            writer.write(record)
            total += 1
        writer.close()

        logger.info(
            "Wrote shard %s (%d episodes)", shard_path.as_posix(), len(batch_indices)
        )
        shard_idx += 1

    return total


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Convert NPZ episodes to ArrayRecord")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--episodes_per_shard", type=int, default=64)
    args = parser.parse_args()

    logger.info("Listing episodes from %s", args.input_dir)
    all_files = list_episode_files(args.input_dir)
    logger.info("Found %d episodes", len(all_files))

    train_files, eval_files = split_files(all_files, args.eval_ratio, args.seed)
    logger.info("Split: %d train, %d eval", len(train_files), len(eval_files))

    output = epath.Path(args.output_dir)

    n_train = write_shards(
        train_files, output / "train", args.episodes_per_shard, args.seed
    )
    n_eval = write_shards(
        eval_files, output / "eval", args.episodes_per_shard, args.seed
    )
    logger.info("Done. Wrote %d train, %d eval episodes.", n_train, n_eval)


if __name__ == "__main__":
    main()
