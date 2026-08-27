"""Expand a task split while preserving fixed validation and evaluation records."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def load_records(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text())
    return payload, list(payload["records"])


def record_key(record: dict) -> tuple[str, int, str]:
    return str(record["repo"]), int(record["episode"]), str(record["camera"])


def rank(seed: int, task: str, episode: int) -> bytes:
    return hashlib.sha256(f"{seed}:{task}:{episode}".encode()).digest()


def summarize(records: list[dict], source: str, repo: str, seed: int) -> dict:
    task_counts: dict[str, int] = defaultdict(int)
    for record in records:
        task_counts[str(record["task"])] += 1
    return {
        "schema_version": 1,
        "source_records": source,
        "repo": repo,
        "selection_seed": seed,
        "num_tasks": len(task_counts),
        "num_trajectories": len(records),
        "total_frames": sum(int(record["frames"]) for record in records),
        "duration_s": sum(float(record["frames"]) / float(record["fps"]) for record in records),
        "task_counts": dict(sorted(task_counts.items())),
        "records": records,
    }


def allocate_proportionally(groups: dict[str, list[dict]], count: int) -> dict[str, int]:
    capacity = {task: len(records) for task, records in groups.items()}
    total_capacity = sum(capacity.values())
    if count > total_capacity:
        raise ValueError(f"Requested {count} records from a pool of {total_capacity}")
    exact = {task: count * size / total_capacity for task, size in capacity.items()}
    allocation = {task: min(math.floor(exact[task]), capacity[task]) for task in groups}
    remaining = count - sum(allocation.values())
    order = sorted(groups, key=lambda task: (-(exact[task] - allocation[task]), task))
    while remaining:
        changed = False
        for task in order:
            if allocation[task] >= capacity[task]:
                continue
            allocation[task] += 1
            remaining -= 1
            changed = True
            if not remaining:
                break
        if not changed:
            raise RuntimeError("Could not complete proportional allocation")
    return allocation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--total_train", type=int)
    target.add_argument("--per_task", type=int)
    parser.add_argument("--reserved_episodes", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_payload, base_train = load_records(args.base_train)
    _, validation = load_records(args.validation)
    _, source_test = load_records(args.test)
    repo = str(train_payload["repo"])
    source = str(train_payload["source_records"])
    reserved_episodes = {int(value) for value in args.reserved_episodes.split(",") if value}

    base_keys = {record_key(record) for record in base_train}
    validation_keys = {record_key(record) for record in validation}
    test_keys = {record_key(record) for record in source_test}
    if base_keys & validation_keys or base_keys & test_keys or validation_keys & test_keys:
        raise ValueError("Input manifests overlap")

    reserved = [record for record in source_test if int(record["episode"]) in reserved_episodes]
    if len(reserved) != len(reserved_episodes):
        found = {int(record["episode"]) for record in reserved}
        raise ValueError(f"Reserved episodes not found: {sorted(reserved_episodes - found)}")
    reserved_tasks = {str(record["task"]) for record in reserved}
    all_tasks = {str(record["task"]) for record in base_train + validation + source_test}
    if reserved_tasks != all_tasks:
        raise ValueError("Reserved evaluation records must cover every task exactly once")

    additional = []
    if args.per_task is not None:
        groups: dict[str, list[dict]] = defaultdict(list)
        for record in validation + source_test:
            if int(record["episode"]) not in reserved_episodes:
                groups[str(record["task"])].append(record)
        base_by_task: dict[str, int] = defaultdict(int)
        for record in base_train:
            base_by_task[str(record["task"])] += 1
        for task in sorted(all_tasks):
            needed = args.per_task - base_by_task[task]
            if needed < 0:
                raise ValueError(f"Base split has more than {args.per_task} records for {task}")
            ordered = sorted(groups[task], key=lambda record: rank(args.seed, task, int(record["episode"])))
            if len(ordered) < needed:
                raise ValueError(f"Task {task} needs {needed} additional records, found {len(ordered)}")
            additional.extend(ordered[:needed])
        total_train = args.per_task * len(all_tasks)
        final_validation = reserved
        candidate_heldout = validation + source_test
        selection_method = f"{args.per_task} per task with fixed reserved evaluation records"
    else:
        groups = defaultdict(list)
        for record in source_test:
            if int(record["episode"]) not in reserved_episodes:
                groups[str(record["task"])].append(record)
        additional_count = args.total_train - len(base_train)
        if additional_count < 0:
            raise ValueError("total_train is smaller than the base training split")
        allocation = allocate_proportionally(groups, additional_count)
        for task, records in sorted(groups.items()):
            ordered = sorted(records, key=lambda record: rank(args.seed, task, int(record["episode"])))
            additional.extend(ordered[: allocation[task]])
        total_train = args.total_train
        final_validation = validation
        candidate_heldout = source_test
        selection_method = "proportional task allocation with sha256 rank"

    train = base_train + additional
    train_keys = {record_key(record) for record in train}
    validation_output_keys = {record_key(record) for record in final_validation}
    test = [
        record
        for record in candidate_heldout
        if record_key(record) not in train_keys and record_key(record) not in validation_output_keys
    ]
    heldout = final_validation + test
    for records in (train, final_validation, test, heldout, reserved):
        records.sort(key=lambda record: (str(record["task"]), int(record["episode"])))

    if len(train) != total_train:
        raise RuntimeError(f"Expected {total_train} training records, got {len(train)}")
    if train_keys & {record_key(record) for record in heldout}:
        raise RuntimeError("Training and held-out records overlap")
    if not {record_key(record) for record in reserved}.issubset({record_key(record) for record in heldout}):
        raise RuntimeError("Reserved evaluation records are not all held out")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        f"train_{total_train}_seed{args.seed}.json": summarize(train, source, repo, args.seed),
        f"validation_seed{args.seed}.json": summarize(final_validation, source, repo, args.seed),
        f"test_seed{args.seed}.json": summarize(test, source, repo, args.seed),
        f"heldout_seed{args.seed}.json": summarize(heldout, source, repo, args.seed),
        f"reserved_eval_seed{args.seed}.json": summarize(reserved, source, repo, args.seed),
    }
    for name, payload in payloads.items():
        (args.output_dir / name).write_text(json.dumps(payload, indent=2) + "\n")
    summary = {
        "repo": repo,
        "selection_seed": args.seed,
        "selection_method": selection_method,
        "base_train_trajectories": len(base_train),
        "additional_train_trajectories": len(additional),
        "train_trajectories": len(train),
        "validation_trajectories": len(final_validation),
        "test_trajectories": len(test),
        "reserved_eval_trajectories": len(reserved),
        "train_duration_s": payloads[f"train_{total_train}_seed{args.seed}.json"]["duration_s"],
        "validation_duration_s": payloads[f"validation_seed{args.seed}.json"]["duration_s"],
        "test_duration_s": payloads[f"test_seed{args.seed}.json"]["duration_s"],
    }
    (args.output_dir / f"summary_seed{args.seed}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
