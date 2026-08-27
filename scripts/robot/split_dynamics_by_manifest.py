"""Copy dynamics records into exact manifest-defined splits."""

import argparse
import io
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from array_record.python.array_record_data_source import ArrayRecordDataSource

from visionary.shards import ShardWriter


def parse_manifest_spec(spec: str) -> tuple[str, Path]:
    name, separator, path = spec.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"Expected NAME=PATH, got {spec!r}")
    return name, Path(path)


def record_key(record: dict) -> tuple[str, int, str]:
    return str(record["repo"]), int(record["episode"]), str(record["camera"])


def source_shards(source: str) -> list[str]:
    if not source.startswith("gs://"):
        return sorted(str(path) for path in Path(source).glob("*.arecord"))
    result = subprocess.run(
        ["gcloud", "storage", "ls", source.rstrip("/") + "/*.arecord"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(result.stdout.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--records_per_shard", type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifests = dict(parse_manifest_spec(spec) for spec in args.manifest)
    selected: dict[tuple[str, int, str], str] = {}
    expected: dict[str, int] = {}
    for split, path in manifests.items():
        payload = json.loads(path.read_text())
        records = payload["records"]
        expected[split] = len(records)
        for record in records:
            key = record_key(record)
            if key in selected:
                raise ValueError(f"Record {key} appears in both {selected[key]} and {split}")
            selected[key] = split

    writers = {
        split: ShardWriter(output_dir / split, args.records_per_shard)
        for split in manifests
    }
    written = {split: 0 for split in manifests}
    scanned = 0
    with tempfile.TemporaryDirectory(prefix="visionary-split-dynamics-") as temporary:
        for source_path in source_shards(args.source):
            local_path = source_path
            if source_path.startswith("gs://"):
                local_path = str(Path(temporary) / Path(source_path).name)
                subprocess.run(
                    ["gcloud", "storage", "cp", source_path, local_path],
                    check=True,
                )
            source = ArrayRecordDataSource([local_path])
            for index in range(len(source)):
                record = source[index]
                scanned += 1
                with np.load(io.BytesIO(record)) as data:
                    key = (str(data["repo"]), int(data["episode"]), str(data["camera"]))
                split = selected.get(key)
                if split is None:
                    continue
                writers[split].write(record)
                written[split] += 1
            del source
            if source_path.startswith("gs://"):
                Path(local_path).unlink()
            print(f"{source_path}: scanned={scanned} written={sum(written.values())}", flush=True)

    shard_counts = {split: writer.close() for split, writer in writers.items()}
    if written != expected:
        missing = {split: expected[split] - written[split] for split in expected}
        raise RuntimeError(f"Split counts do not match manifests: written={written} missing={missing}")
    summary = {
        "source": args.source,
        "manifests": {split: str(path) for split, path in manifests.items()},
        "records_scanned": scanned,
        "records_written": written,
        "shards_written": shard_counts,
    }
    (output_dir / "split_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
