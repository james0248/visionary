import argparse
import hashlib
import io
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import grain.python as grain
from array_record.python.array_record_module import ArrayRecordReader
from etils import epath

from visionary.dataset import DynamicsDataSource, RandomDynamicsCrop

logger = logging.getLogger(__name__)


@dataclass
class BadRecord:
    record: int
    error_type: str
    error: str


@dataclass
class ShardReport:
    path: str
    size_bytes: int | None = None
    sha256: str | None = None
    num_records: int | None = None
    records_checked: int = 0
    valid_records: int = 0
    bad_records: list[BadRecord] = field(default_factory=list)
    open_error_type: str | None = None
    open_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.open_error is None and not self.bad_records


def iter_shards(data_dir: Path) -> list[Path]:
    shards = sorted(data_dir.glob("*.arecord"))
    if not shards:
        raise FileNotFoundError(f"No .arecord shards found in {data_dir}")
    return shards


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_npz_payload(record: bytes, keys: tuple[str, ...] | None) -> None:
    with np.load(io.BytesIO(record)) as data:
        requested_keys = keys if keys is not None else tuple(data.files)
        missing_keys = [key for key in requested_keys if key not in data]
        if missing_keys:
            raise KeyError(f"Missing keys: {', '.join(missing_keys)}")
        for key in requested_keys:
            np.asarray(data[key])


def diagnose_shard(
    shard_path: Path,
    keys: tuple[str, ...] | None,
    max_bad_records: int,
    hash_files: bool,
) -> ShardReport:
    report = ShardReport(path=shard_path.as_posix())
    try:
        report.size_bytes = shard_path.stat().st_size
        if hash_files:
            report.sha256 = file_sha256(shard_path)
    except Exception as exc:
        logger.warning("Could not stat/hash %s: %s", shard_path, exc)

    reader: ArrayRecordReader | None = None
    try:
        reader = ArrayRecordReader(shard_path.as_posix())
        report.num_records = int(reader.num_records())
    except Exception as exc:
        report.open_error_type = type(exc).__name__
        report.open_error = str(exc)
        logger.error("Could not open shard %s: %s: %s", shard_path, type(exc).__name__, exc)
        if reader is not None:
            reader.close()
        return report

    logger.info("Checking %s (%d records)", shard_path, report.num_records)
    try:
        for record_idx in range(report.num_records):
            report.records_checked += 1
            try:
                record = reader.read([record_idx])[0]
                validate_npz_payload(record, keys)
            except Exception as exc:
                bad_record = BadRecord(
                    record=record_idx,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                report.bad_records.append(bad_record)
                logger.error(
                    "Bad record: shard=%s record=%d error=%s: %s",
                    shard_path,
                    record_idx,
                    bad_record.error_type,
                    bad_record.error,
                )
                if len(report.bad_records) >= max_bad_records:
                    logger.error(
                        "Stopping %s after %d bad records",
                        shard_path,
                        max_bad_records,
                    )
                    break
                continue
            report.valid_records += 1
    finally:
        reader.close()

    return report


def report_payload(data_dirs: list[Path], reports: list[ShardReport]) -> dict[str, Any]:
    bad_shards = [report for report in reports if not report.ok]
    return {
        "data_dirs": [data_dir.as_posix() for data_dir in data_dirs],
        "totals": {
            "shards": len(reports),
            "bad_shards": len(bad_shards),
            "records_checked": sum(report.records_checked for report in reports),
            "valid_records": sum(report.valid_records for report in reports),
            "bad_records": sum(len(report.bad_records) for report in reports),
            "bytes": sum(report.size_bytes or 0 for report in reports),
        },
        "bad_shards": [shard_payload(report) for report in bad_shards],
        "shards": [shard_payload(report) for report in reports],
    }


def shard_payload(report: ShardReport) -> dict[str, Any]:
    return {
        "path": report.path,
        "size_bytes": report.size_bytes,
        "sha256": report.sha256,
        "num_records": report.num_records,
        "records_checked": report.records_checked,
        "valid_records": report.valid_records,
        "open_error_type": report.open_error_type,
        "open_error": report.open_error,
        "bad_records": [
            {
                "record": bad.record,
                "error_type": bad.error_type,
                "error": bad.error,
            }
            for bad in report.bad_records
        ],
    }


def write_report(path: str, payload: dict[str, Any]) -> None:
    out_path = epath.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Wrote report to %s", path)


def stress_dynamics_loader(args) -> dict[str, Any]:
    summaries = []
    total_batches = 0
    total_examples = 0
    for data_dir in args.data_dirs:
        source = DynamicsDataSource(data_dir.as_posix())
        sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.NoSharding(),
            shuffle=True,
            seed=int(args.seed),
        )
        loader = grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=[
                RandomDynamicsCrop(int(args.sequence_length)),
                grain.Batch(batch_size=int(args.batch_size), drop_remainder=True),
            ],
            worker_count=int(args.worker_count),
            read_options=grain.ReadOptions(
                num_threads=int(args.num_threads),
                prefetch_buffer_size=int(args.prefetch_buffer_size),
            ),
        )
        batches = 0
        examples = 0
        logger.info(
            "Stress reading %s: records=%d batches=%d batch_size=%d length=%d "
            "worker_count=%d num_threads=%d prefetch=%d",
            data_dir,
            len(source),
            args.batches,
            args.batch_size,
            args.sequence_length,
            args.worker_count,
            args.num_threads,
            args.prefetch_buffer_size,
        )
        iterator = iter(loader)
        for _ in range(int(args.batches)):
            batch = next(iterator)
            batches += 1
            examples += int(batch["video"].shape[0])
        total_batches += batches
        total_examples += examples
        summaries.append(
            {
                "data_dir": data_dir.as_posix(),
                "records": len(source),
                "batches": batches,
                "examples": examples,
            }
        )
    return {
        "mode": "stress_dynamics_loader",
        "settings": {
            "batch_size": int(args.batch_size),
            "sequence_length": int(args.sequence_length),
            "batches": int(args.batches),
            "worker_count": int(args.worker_count),
            "num_threads": int(args.num_threads),
            "prefetch_buffer_size": int(args.prefetch_buffer_size),
            "seed": int(args.seed),
        },
        "totals": {
            "batches": total_batches,
            "examples": total_examples,
        },
        "summaries": summaries,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose ArrayRecord shards by reading each record directly and reporting the "
            "exact shard/record that fails."
        )
    )
    parser.add_argument("data_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--stress_dynamics_loader",
        action="store_true",
        help="Use DynamicsDataSource plus Grain DataLoader to reproduce training-style reads.",
    )
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--worker_count", type=int, default=4)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--prefetch_buffer_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keys", nargs="+", help="NPZ keys to force-read.")
    parser.add_argument("--report_path", help="Optional local or gs:// JSON report path.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of shard scan threads.",
    )
    parser.add_argument(
        "--max_bad_records_per_shard",
        type=int,
        default=5,
        help="Stop scanning a shard after this many bad records.",
    )
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Include SHA-256 for every shard. This is useful for comparing disk copies.",
    )
    parser.add_argument(
        "--success_on_corruption",
        action="store_true",
        help="Exit 0 even if corrupt shards are found.",
    )
    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_bad_records_per_shard <= 0:
        raise ValueError("--max_bad_records_per_shard must be positive")

    data_dirs = [Path(data_dir) for data_dir in args.data_dirs]
    if args.stress_dynamics_loader:
        payload = stress_dynamics_loader(args)
        if args.report_path:
            write_report(args.report_path, payload)
        logger.info(
            "Stress summary: batches=%d examples=%d",
            payload["totals"]["batches"],
            payload["totals"]["examples"],
        )
        return 0

    shards = [shard for data_dir in data_dirs for shard in iter_shards(data_dir)]
    keys = tuple(args.keys) if args.keys else None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        reports = list(
            executor.map(
                diagnose_shard,
                shards,
                [keys] * len(shards),
                [args.max_bad_records_per_shard] * len(shards),
                [bool(args.sha256)] * len(shards),
            )
        )

    payload = report_payload(data_dirs, reports)
    if args.report_path:
        write_report(args.report_path, payload)

    totals = payload["totals"]
    logger.info(
        "Summary: shards=%d bad_shards=%d records_checked=%d valid_records=%d bad_records=%d",
        totals["shards"],
        totals["bad_shards"],
        totals["records_checked"],
        totals["valid_records"],
        totals["bad_records"],
    )
    if totals["bad_shards"]:
        logger.error("Found %d bad shards", totals["bad_shards"])
        return 0 if args.success_on_corruption else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
