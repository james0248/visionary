#!/usr/bin/env python3

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import grain.python as grain
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest_rows(root: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for path in sorted((root / "manifests").glob("*.records.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row["shard"], int(row["record_index"]))
            if key in rows:
                raise ValueError(f"Duplicate manifest key {key}")
            rows[key] = row
    return rows


def scalar(data, key: str, default):
    return data[key].item() if key in data else default


def row_from_record(data, shard: Path, record_index: int, length: int, fps: float) -> dict:
    repo = str(scalar(data, "repo", ""))
    row = {
        "shard": shard.name,
        "record_index": record_index,
        "dataset": repo,
        "format_dataset": str(scalar(data, "dataset", "")),
        "repo": repo,
        "episode": int(scalar(data, "episode", -1)),
        "camera": str(scalar(data, "camera", "")),
        "task": str(scalar(data, "task", "")),
        "frames": length,
        "fps": fps,
    }
    success = int(scalar(data, "success", -1))
    if success >= 0:
        row["success"] = bool(success)
    for key in ("success_class", "policy_repo_id", "policy_type"):
        value = str(scalar(data, key, ""))
        if value:
            row[key] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir", type=Path)
    args = parser.parse_args()

    root = args.split_dir.resolve()
    manifest_rows = load_manifest_rows(root)
    has_sidecars = bool(manifest_rows)
    combined = []
    lengths = []
    rates = []
    global_index = 0
    for shard in sorted(root.glob("*.arecord")):
        source = grain.ArrayRecordDataSource([str(shard)])
        for record_index in range(len(source)):
            key = (shard.name, record_index)
            if has_sidecars and key not in manifest_rows:
                raise ValueError(f"No manifest row for {key}")
            with np.load(io.BytesIO(source[record_index])) as data:
                length = int(data["length"])
                fps = float(data["fps"])
                actions = np.asarray(data["actions"])
                state = np.asarray(data["state"])
                actual = (str(data["repo"]), int(data["episode"]), str(data["camera"]))
                row = (
                    manifest_rows.pop(key) if has_sidecars else row_from_record(data, shard, record_index, length, fps)
                )
            if has_sidecars:
                expected = (row["repo"], int(row["episode"]), row["camera"])
                if actual != expected:
                    raise ValueError(f"Provenance mismatch in {key}: {actual} != {expected}")
            if actions.shape != (length, 6) or state.shape != (length, 6):
                raise ValueError(
                    f"Shape mismatch in {key}: length={length} actions={actions.shape} state={state.shape}"
                )
            if length != int(row["frames"]) or abs(fps - float(row["fps"])) > 1e-5:
                raise ValueError(f"Sidecar mismatch in {key}")
            row["global_index"] = global_index
            combined.append(row)
            lengths.append(length)
            rates.append(fps)
            global_index += 1
    if has_sidecars and manifest_rows:
        raise ValueError(f"Manifest has {len(manifest_rows)} records without ArrayRecord data")

    (root / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in combined))
    (root / "lengths.json").write_text(json.dumps(lengths))
    (root / "fps.json").write_text(json.dumps(rates))

    files = []
    for path in sorted(file for file in root.rglob("*") if file.is_file() and file.name != "upload_manifest.json"):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    summary = {
        "format": "Visionary tokenizer-input ArrayRecord",
        "split": root.name,
        "records": len(combined),
        "frames_native_rate": sum(lengths),
        "datasets": dict(sorted(Counter(row["dataset"] for row in combined).items())),
        "cameras": dict(sorted(Counter(f"{row['dataset']}/{row['camera']}" for row in combined).items())),
        "armnet_success_class": dict(
            sorted(Counter(row["success_class"] for row in combined if row.get("success_class")).items())
        ),
        "files": files,
        "total_bytes": sum(file["bytes"] for file in files),
    }
    (root / "upload_manifest.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {key: summary[key] for key in ("records", "frames_native_rate", "datasets", "total_bytes")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
