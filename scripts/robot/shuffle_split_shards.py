import argparse
import hashlib
import io
import json
import resource
from pathlib import Path

import grain.python as grain
import numpy as np

from visionary.shards import ShardWriter

GB = 1024**3


def raise_fd_limit(needed):
    """ArrayRecordDataSource holds every input file open, so the 1024 default
    fails once the corpus passes ~1000 sources."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = min(max(needed * 2 + 256, soft), hard)
    if want > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def is_eval(seed, repo, episode, eval_ratio):
    digest = hashlib.sha256(f"{seed}:{repo}:{episode}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < eval_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--records_per_shard", type=int, default=256)
    args = parser.parse_args()

    paths = sorted(str(p) for p in Path(args.input_dir).rglob("*.arecord"))
    if not paths:
        raise SystemExit(f"no .arecord files under {args.input_dir}")
    limit = raise_fd_limit(len(paths))
    if limit < len(paths) + 64:
        raise SystemExit(f"fd limit {limit} too low for {len(paths)} input files")
    source = grain.ArrayRecordDataSource(paths)
    print(f"{len(paths)} input files, {len(source):,} records, fd limit {limit}", flush=True)

    out = Path(args.output_dir)
    writers = {
        split: ShardWriter(out / split, args.records_per_shard) for split in ("train", "eval")
    }
    counts = {"train": 0, "eval": 0}
    payload = {"train": 0, "eval": 0}
    lengths = {"train": [], "eval": []}
    rates = {"train": [], "eval": []}
    order = np.random.default_rng(args.seed).permutation(len(source))
    for done, idx in enumerate(order, 1):
        record = source[int(idx)]
        with np.load(io.BytesIO(record)) as data:
            repo, episode = str(data["repo"]), int(data["episode"])
            n_frames = int(data["length"])
            rate = float(data["fps"]) if "fps" in data else 0.0
        split = "eval" if is_eval(args.seed, repo, episode, args.eval_ratio) else "train"
        writers[split].write(record)
        lengths[split].append(n_frames)
        rates[split].append(rate)
        counts[split] += 1
        payload[split] += len(record)
        if done % 2000 == 0:
            print(f"  {done:,}/{len(source):,}", flush=True)

    shards = {split: writer.close() for split, writer in writers.items()}
    for split in ("train", "eval"):
        (out / split / "lengths.json").write_text(json.dumps(lengths[split]))
        (out / split / "fps.json").write_text(json.dumps(rates[split]))
    summary = {
        "input_files": len(paths),
        "records": len(source),
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "records_per_shard": args.records_per_shard,
        "counts": counts,
        "shards": shards,
        "payload_bytes": payload,
    }
    (out / "shuffle_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"train {counts['train']:,} records / {shards['train']} shards "
          f"({payload['train'] / GB:.1f} GB), "
          f"eval {counts['eval']:,} records / {shards['eval']} shards "
          f"({payload['eval'] / GB:.1f} GB) -> {out}", flush=True)


if __name__ == "__main__":
    main()
