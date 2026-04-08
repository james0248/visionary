import argparse
import io
import statistics
import time

import grain.python as grain
import numpy as np
from etils import epath

from visionary.dataset import (
    PreprocessAndPatchify,
    RandomVideoCrop,
    VideoDataSource,
)


class EpisodeDataSource(grain.RandomAccessDataSource):
    def __init__(self, data_dir: str):
        shard_dir = epath.Path(data_dir)
        paths = sorted(
            [p for p in shard_dir.iterdir() if p.suffix == ".arecord"],
            key=lambda p: p.as_posix(),
        )
        if not paths:
            raise FileNotFoundError(f"No .arecord files found in {data_dir}")
        self._source = grain.ArrayRecordDataSource([p.as_posix() for p in paths])

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx: int):
        with np.load(io.BytesIO(self._source[idx])) as data:
            return {"video": np.asarray(data["frames"])}


def make_source(data_dir: str, source_type: str):
    if source_type == "episode":
        return EpisodeDataSource(data_dir)
    if source_type == "chunk":
        return VideoDataSource(data_dir)
    raise ValueError(f"Unknown source_type={source_type!r}")


def summarize_samples(source, indices: list[int]) -> None:
    item_times = []
    frame_lengths = []
    raw_bytes = []
    video_bytes = []
    for idx in indices:
        raw_start = time.monotonic()
        raw = source._source[idx]
        raw_time = time.monotonic() - raw_start

        item_start = time.monotonic()
        item = source[idx]
        item_time = time.monotonic() - item_start

        raw_bytes.append(len(raw))
        item_times.append(item_time)
        frame_lengths.append(int(item["video"].shape[0]))
        video_bytes.append(int(item["video"].nbytes))
        print(
            f"sample idx={idx} raw_fetch={raw_time:.4f}s decode={item_time:.4f}s "
            f"frames={item['video'].shape[0]} shape={item['video'].shape}"
        )

    print(
        "sample summary "
        f"raw_bytes_mean={statistics.mean(raw_bytes) / 1e3:.1f}KB "
        f"video_bytes_mean={statistics.mean(video_bytes) / 1e6:.1f}MB "
        f"frames_mean={statistics.mean(frame_lengths):.1f} "
        f"decode_mean={statistics.mean(item_times):.4f}s"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark ArrayRecord loading")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--source_type", choices=("episode", "chunk"), default="episode")
    parser.add_argument("--frame_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--worker_count", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=2)
    parser.add_argument("--prefetch_buffer_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--pad_height", type=int, default=7)
    parser.add_argument("--pad_width", type=int, default=0)
    parser.add_argument("--measure_batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t = time.monotonic()
    source = make_source(args.data_dir, args.source_type)
    print(
        f"source_init={time.monotonic() - t:.3f}s len={len(source)} source_type={args.source_type}"
    )

    sample_indices = sorted({0, min(1, len(source) - 1), len(source) // 2, len(source) - 1})
    summarize_samples(source, sample_indices)

    sampler = grain.IndexSampler(
        num_records=len(source),
        shard_options=grain.ShardByJaxProcess(),
        shuffle=True,
        seed=args.seed,
    )
    transforms = [
        RandomVideoCrop(args.frame_length),
        PreprocessAndPatchify(args.patch_size, (args.pad_height, args.pad_width)),
    ]

    t = time.monotonic()
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[
            *transforms,
            grain.Batch(batch_size=args.batch_size, drop_remainder=True),
        ],
        worker_count=args.worker_count,
        read_options=grain.ReadOptions(
            num_threads=args.num_threads,
            prefetch_buffer_size=args.prefetch_buffer_size,
        ),
    )
    print(f"dataloader_creation={time.monotonic() - t:.3f}s")

    t = time.monotonic()
    first_batch = next(iter(loader))
    print(f"first_batch={time.monotonic() - t:.3f}s batch_video_shape={first_batch['video'].shape}")

    iterator = iter(loader)
    next(iterator)
    t = time.monotonic()
    next(iterator)
    print(f"second_batch_same_iter={time.monotonic() - t:.3f}s")

    iterator = iter(loader)
    batch_times = []
    for _ in range(args.measure_batches):
        t = time.monotonic()
        next(iterator)
        batch_times.append(time.monotonic() - t)
    print(
        f"batch_mean={statistics.mean(batch_times):.3f}s "
        f"batch_median={statistics.median(batch_times):.3f}s "
        f"batch_min={min(batch_times):.3f}s "
        f"batch_max={max(batch_times):.3f}s "
        f"measured_batches={len(batch_times)}"
    )


if __name__ == "__main__":
    main()
