import argparse
import io
import json
import logging
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from array_record.python.array_record_module import ArrayRecordReader, ArrayRecordWriter
from etils import epath

logger = logging.getLogger(__name__)


@dataclass
class ValidationSummary:
    input_dir: Path
    total_records: int = 0
    valid_records: int = 0
    corrupt_records: int = 0
    replaced_records: int = 0
    shard_failures: int = 0
    corrupt_examples: list[dict[str, Any]] = field(default_factory=list)


def iter_shards(data_dir: Path) -> list[Path]:
    shards = sorted(data_dir.glob("*.arecord"))
    if not shards:
        raise FileNotFoundError(f"No .arecord shards found in {data_dir}")
    return shards


def validate_npz_payload(record: bytes, keys: Iterable[str] | None) -> None:
    with np.load(io.BytesIO(record)) as data:
        requested_keys = list(keys) if keys is not None else list(data.files)
        missing_keys = [key for key in requested_keys if key not in data]
        if missing_keys:
            raise KeyError(f"Missing keys: {', '.join(missing_keys)}")
        for key in requested_keys:
            np.asarray(data[key])


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if any(output_dir.glob("*.arecord")):
            if not overwrite:
                raise FileExistsError(
                    f"{output_dir} already contains .arecord files; pass --overwrite to replace it"
                )
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def validate_dir(
    data_dir: Path,
    output_dir: Path | None,
    keys: tuple[str, ...] | None,
    replace_corrupt_with_previous: bool,
    max_failures: int | None,
    max_report_records: int,
) -> ValidationSummary:
    summary = ValidationSummary(input_dir=data_dir)
    last_valid_record: bytes | None = None

    for shard_path in iter_shards(data_dir):
        reader: ArrayRecordReader | None = None
        writer: ArrayRecordWriter | None = None
        output_path = output_dir / shard_path.name if output_dir is not None else None
        try:
            reader = ArrayRecordReader(str(shard_path))
            num_records = int(reader.num_records())
            if output_path is not None:
                writer = ArrayRecordWriter(str(output_path), "group_size:1")
        except Exception:
            logger.exception("Could not open shard %s", shard_path)
            if reader is not None:
                reader.close()
            summary.shard_failures += 1
            continue

        logger.info("Checking %s (%d records)", shard_path, num_records)
        try:
            for record_idx in range(num_records):
                summary.total_records += 1
                try:
                    record = reader.read([record_idx])[0]
                    validate_npz_payload(record, keys)
                except Exception as exc:
                    summary.corrupt_records += 1
                    logger.error(
                        "Corrupt record: shard=%s record=%d error=%s: %s",
                        shard_path,
                        record_idx,
                        type(exc).__name__,
                        exc,
                    )
                    if len(summary.corrupt_examples) < max_report_records:
                        summary.corrupt_examples.append(
                            {
                                "shard": shard_path.as_posix(),
                                "record": record_idx,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                    if writer is not None and replace_corrupt_with_previous:
                        if last_valid_record is None:
                            logger.error(
                                "Cannot replace %s record %d: no previous valid record exists",
                                shard_path,
                                record_idx,
                            )
                        else:
                            writer.write(last_valid_record)
                            summary.replaced_records += 1
                    if max_failures is not None and summary.corrupt_records >= max_failures:
                        logger.error("Stopping after %d corrupt records", max_failures)
                        return summary
                    continue

                summary.valid_records += 1
                last_valid_record = record
                if writer is not None:
                    writer.write(record)
        finally:
            if reader is not None:
                reader.close()
            if writer is not None:
                writer.close()
                logger.info("Wrote cleaned shard %s", output_path)

    return summary


def summary_payload(summary: ValidationSummary) -> dict[str, Any]:
    return {
        "input_dir": summary.input_dir.as_posix(),
        "total_records": summary.total_records,
        "valid_records": summary.valid_records,
        "corrupt_records": summary.corrupt_records,
        "replaced_records": summary.replaced_records,
        "shard_failures": summary.shard_failures,
        "corrupt_examples": summary.corrupt_examples,
    }


def write_report(report_path: str, summaries: list[ValidationSummary]) -> None:
    payload = {
        "summaries": [summary_payload(summary) for summary in summaries],
        "totals": {
            "total_records": sum(summary.total_records for summary in summaries),
            "valid_records": sum(summary.valid_records for summary in summaries),
            "corrupt_records": sum(summary.corrupt_records for summary in summaries),
            "replaced_records": sum(summary.replaced_records for summary in summaries),
            "shard_failures": sum(summary.shard_failures for summary in summaries),
        },
    }
    path = epath.Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Wrote validation report to %s", report_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(
        description="Validate ArrayRecord records containing NPZ payloads and optionally rewrite a clean copy."
    )
    parser.add_argument("data_dirs", nargs="+", type=Path, help="Directories containing .arecord shards.")
    parser.add_argument(
        "--output_root",
        type=Path,
        help="Optional root directory for cleaned copies. Each input is written under output_root/<name>.",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        help="NPZ keys to force-read. Defaults to all keys in each payload.",
    )
    parser.add_argument(
        "--max_failures",
        type=int,
        help="Stop after this many corrupt records.",
    )
    parser.add_argument(
        "--max_report_records",
        type=int,
        default=1000,
        help="Maximum corrupt record examples to include in the optional JSON report.",
    )
    parser.add_argument(
        "--report_path",
        help="Optional local or gs:// JSON report path.",
    )
    parser.add_argument(
        "--success_on_corruption",
        action="store_true",
        help="Exit 0 after reporting corrupt records. Infrastructure errors still raise.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing cleaned output directories that contain .arecord files.",
    )
    parser.add_argument(
        "--replace_corrupt_with_previous",
        action="store_true",
        help=(
            "When rewriting, keep record counts stable by replacing corrupt records with the "
            "previous valid record instead of dropping them."
        ),
    )
    args = parser.parse_args()

    if args.replace_corrupt_with_previous and args.output_root is None:
        raise ValueError("--replace_corrupt_with_previous requires --output_root")

    output_dirs: dict[Path, Path | None] = {}
    for data_dir in args.data_dirs:
        output_dir = None
        if args.output_root is not None:
            output_dir = args.output_root / data_dir.name
            if output_dir.resolve() == data_dir.resolve():
                raise ValueError(f"Refusing to rewrite {data_dir} in place")
            prepare_output_dir(output_dir, overwrite=args.overwrite)
        output_dirs[data_dir] = output_dir

    summaries = [
        validate_dir(
            data_dir=data_dir,
            output_dir=output_dir,
            keys=tuple(args.keys) if args.keys else None,
            replace_corrupt_with_previous=bool(args.replace_corrupt_with_previous),
            max_failures=args.max_failures,
            max_report_records=max(args.max_report_records, 0),
        )
        for data_dir, output_dir in output_dirs.items()
    ]

    if args.report_path:
        write_report(args.report_path, summaries)

    corrupt_total = 0
    for summary in summaries:
        corrupt_total += summary.corrupt_records + summary.shard_failures
        logger.info(
            "Summary for %s: total=%d valid=%d corrupt=%d replaced=%d shard_failures=%d",
            summary.input_dir,
            summary.total_records,
            summary.valid_records,
            summary.corrupt_records,
            summary.replaced_records,
            summary.shard_failures,
        )

    if corrupt_total:
        logger.error("Found %d corrupt records/shards", corrupt_total)
        return 0 if args.output_root is not None or args.success_on_corruption else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
