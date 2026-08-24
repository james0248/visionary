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
    parser.add_argument(
        "--single_split",
        choices=("train", "eval", "test"),
        help="Write every input record to one split without shuffling.",
    )
    parser.add_argument(
        "--policy_type",
        help="Keep this policy type when a record has a non-empty policy_type. Records without the field stay.",
    )
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
    splits = (args.single_split,) if args.single_split else ("train", "eval")
    writers = {split: ShardWriter(out / split, args.records_per_shard) for split in splits}
    counts = {split: 0 for split in splits}
    payload = {split: 0 for split in splits}
    lengths = {split: [] for split in splits}
    rates = {split: [] for split in splits}
    order = np.arange(len(source)) if args.single_split else np.random.default_rng(args.seed).permutation(len(source))
    filtered = 0
    for done, idx in enumerate(order, 1):
        record = source[int(idx)]
        with np.load(io.BytesIO(record)) as data:
            repo, episode = str(data["repo"]), int(data["episode"])
            n_frames = int(data["length"])
            rate = float(data["fps"]) if "fps" in data else 0.0
            policy_type = str(data["policy_type"]) if "policy_type" in data else ""
        if args.policy_type and policy_type and policy_type != args.policy_type:
            filtered += 1
            continue
        split = args.single_split or ("eval" if is_eval(args.seed, repo, episode, args.eval_ratio) else "train")
        writers[split].write(record)
        lengths[split].append(n_frames)
        rates[split].append(rate)
        counts[split] += 1
        payload[split] += len(record)
        if done % 2000 == 0:
            print(f"  {done:,}/{len(source):,}", flush=True)

    shards = {split: writer.close() for split, writer in writers.items()}
    for split in splits:
        (out / split / "lengths.json").write_text(json.dumps(lengths[split]))
        (out / split / "fps.json").write_text(json.dumps(rates[split]))
    summary = {
        "input_files": len(paths),
        "records": len(source),
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "records_per_shard": args.records_per_shard,
        "single_split": args.single_split,
        "policy_type": args.policy_type,
        "records_filtered": filtered,
        "counts": counts,
        "shards": shards,
        "payload_bytes": payload,
    }
    (out / "shuffle_summary.json").write_text(json.dumps(summary, indent=1))
    result = ", ".join(
        f"{split} {counts[split]:,} records / {shards[split]} shards ({payload[split] / GB:.1f} GB)" for split in splits
    )
    print(f"{result} -> {out}", flush=True)


if __name__ == "__main__":
    main()
