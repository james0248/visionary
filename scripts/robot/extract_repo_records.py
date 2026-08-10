"""Extract every record belonging to one repo from packed arecord shards.

Records are copied byte-for-byte, so the output is a drop-in dataset dir for
the same consumers. Works on a local shard dir, or streams shards one at a
time from GCS (download, filter, delete) so the full set never has to fit on
disk.

    uv run python scripts/robot/extract_repo_records.py \
        --src data/so101/dyn_g5/eval --out data/so101/dyn_single/eval \
        --repo Loki0929/so100_lan
"""

import argparse
import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from array_record.python.array_record_data_source import ArrayRecordDataSource
from array_record.python.array_record_module import ArrayRecordWriter


def shard_list(src: str) -> list[str]:
    if src.startswith("gs://"):
        out = subprocess.run(
            ["gcloud", "storage", "ls", src.rstrip("/") + "/*.arecord"],
            check=True, capture_output=True, text=True,
        ).stdout.split()
        return sorted(out)
    return sorted(str(p) for p in Path(src).glob("*.arecord"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Local dir or gs:// dir of .arecord shards.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--records_per_shard", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_idx = written = scanned = 0
    writer = ArrayRecordWriter(str(out / f"shard-{shard_idx:05d}.arecord"), "group_size:1")

    def rotate():
        nonlocal writer, shard_idx
        writer.close()
        shard_idx += 1
        writer = ArrayRecordWriter(str(out / f"shard-{shard_idx:05d}.arecord"), "group_size:1")

    for path in shard_list(args.src):
        local = path
        tmp = None
        if path.startswith("gs://"):
            tmp = tempfile.NamedTemporaryFile(suffix=".arecord", delete=False)
            tmp.close()
            subprocess.run(
                ["gcloud", "storage", "cp", path, tmp.name],
                check=True, capture_output=True,
            )
            local = tmp.name
        source = ArrayRecordDataSource([local])
        for i in range(len(source)):
            record = source[i]
            scanned += 1
            with np.load(io.BytesIO(record)) as data:
                keep = str(data["repo"]) == args.repo
            if keep:
                writer.write(record)
                written += 1
                if written % args.records_per_shard == 0:
                    rotate()
        del source
        if tmp is not None:
            Path(tmp.name).unlink()
        print(f"{path}: scanned {scanned}, kept {written}", flush=True)

    writer.close()
    if written == 0:
        raise SystemExit(f"No records found for repo {args.repo!r}")
    print(f"done: {written}/{scanned} records -> {out} ({shard_idx + 1} shards)")


if __name__ == "__main__":
    main()
