import argparse
import hashlib
import io
import json
from pathlib import Path

import grain.python as grain
import numpy as np

from visionary.shards import ShardWriter

GB = 1024**3


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
    source = grain.ArrayRecordDataSource(paths)
    print(f"{len(paths)} input files, {len(source):,} records", flush=True)

    out = Path(args.output_dir)
    writers = {
        split: ShardWriter(out / split, args.records_per_shard) for split in ("train", "eval")
    }
    counts = {"train": 0, "eval": 0}
    payload = {"train": 0, "eval": 0}
    order = np.random.default_rng(args.seed).permutation(len(source))
    for done, idx in enumerate(order, 1):
        record = source[int(idx)]
        with np.load(io.BytesIO(record)) as data:
            repo, episode = str(data["repo"]), int(data["episode"])
        split = "eval" if is_eval(args.seed, repo, episode, args.eval_ratio) else "train"
        writers[split].write(record)
        counts[split] += 1
        payload[split] += len(record)
        if done % 2000 == 0:
            print(f"  {done:,}/{len(source):,}", flush=True)

    shards = {split: writer.close() for split, writer in writers.items()}
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
