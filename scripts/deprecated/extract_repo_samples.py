import argparse
import io
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from array_record.python.array_record_data_source import ArrayRecordDataSource
from array_record.python.array_record_module import ArrayRecordWriter


def shard_list(src: str) -> list[str]:
    if src.startswith("gs://"):
        listed = subprocess.run(
            ["gcloud", "storage", "ls", src.rstrip("/") + "/*.arecord"],
            check=True, capture_output=True, text=True,
        ).stdout.split()
        return sorted(listed)
    return sorted(str(p) for p in Path(src).glob("*.arecord"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repos", required=True)
    ap.add_argument("--per_repo", type=int, default=1)
    args = ap.parse_args()

    wanted = json.load(open(args.repos))
    counts: dict[str, int] = {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = ArrayRecordWriter(str(out / "shard-00000.arecord"), "group_size:1")
    written = 0

    for path in shard_list(args.src):
        if len(counts) == len(wanted) and all(v >= args.per_repo for v in counts.values()):
            break
        local, tmp = path, None
        if path.startswith("gs://"):
            tmp = tempfile.NamedTemporaryFile(suffix=".arecord", delete=False)
            tmp.close()
            subprocess.run(["gcloud", "storage", "cp", path, tmp.name], check=True, capture_output=True)
            local = tmp.name
        source = ArrayRecordDataSource([local])
        for i in range(len(source)):
            record = source[i]
            with np.load(io.BytesIO(record)) as data:
                repo, camera = str(data["repo"]), str(data["camera"])
            if wanted.get(repo) == camera and counts.get(repo, 0) < args.per_repo:
                writer.write(record)
                counts[repo] = counts.get(repo, 0) + 1
                written += 1
        del source
        if tmp is not None:
            Path(tmp.name).unlink()
        print(f"{path}: have {len(counts)}/{len(wanted)} repos, {written} records", flush=True)

    writer.close()
    print(f"done: {written} records, {len(counts)}/{len(wanted)} repos")
    print("not found:", sorted(set(wanted) - set(counts)))


if __name__ == "__main__":
    main()
