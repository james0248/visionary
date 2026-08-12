from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ORT WebGPU benchmark results.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def by_name(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keyed = {}
    entries = result.get("results", [])
    if isinstance(entries, dict):
        entries = entries.values()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = entry.get("mode") or entry.get("name")
        if key is not None:
            keyed[key] = entry
    return keyed


def main() -> int:
    args = parse_args()
    result = load_json(args.result)
    baseline = load_json(args.baseline)

    if result.get("status") == "blocked":
        print(f"streaming benchmark blocked: {result.get('blocked_reason', 'unknown reason')}")
        for missing in result.get("missing_artifacts", []):
            print(f"missing {missing.get('role')}: {missing.get('accepted_names')}")
        return 0

    if result.get("status") != "passed":
        print(f"benchmark failed: {result.get('message', 'unknown error')}", file=sys.stderr)
        return 1

    baseline_results = by_name(baseline)
    latest_results = by_name(result)
    ratio = float(baseline.get("policy", {}).get("p95_regression_ratio", 1.25))
    hard_fail = args.fail_on_regression or baseline.get("policy", {}).get("mode") == "fail"

    failures = []
    warnings = []
    for name, latest in latest_results.items():
        latency = latest.get("latency") or latest.get("timing", {}).get("streaming_frame")
        if latency is None:
            print(f"{name}: no timed latency samples")
            continue
        latest_p95 = float(latency["p95_ms"])
        baseline_entry = baseline_results.get(name)
        if baseline_entry is None:
            print(f"{name}: p95={latest_p95:.3f}ms (no baseline)")
            continue
        baseline_latency = baseline_entry.get("latency") or baseline_entry.get("timing", {}).get("streaming_frame")
        if baseline_latency is None:
            print(f"{name}: p95={latest_p95:.3f}ms (baseline has no comparable latency)")
            continue
        baseline_p95 = float(baseline_latency["p95_ms"])
        allowed = baseline_p95 * ratio
        line = f"{name}: p95={latest_p95:.3f}ms baseline={baseline_p95:.3f}ms allowed={allowed:.3f}ms"
        if latest_p95 > allowed:
            if hard_fail:
                failures.append(line)
            else:
                warnings.append(line)
        print(line)

    for warning in warnings:
        print(f"warning: regression: {warning}", file=sys.stderr)
    for failure in failures:
        print(f"error: regression: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
