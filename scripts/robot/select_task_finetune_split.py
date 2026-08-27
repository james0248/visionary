import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def rank(seed: int, task: str, episode: int) -> bytes:
    return hashlib.sha256(f"{seed}:{task}:{episode}".encode()).digest()


def summarize(records: list[dict], source: str, repo: str, seed: int, per_task: int) -> dict:
    task_counts: dict[str, int] = defaultdict(int)
    for record in records:
        task_counts[str(record["task"])] += 1
    return {
        "schema_version": 1,
        "source_records": source,
        "repo": repo,
        "selection_seed": seed,
        "per_task": per_task,
        "num_tasks": len(task_counts),
        "num_trajectories": len(records),
        "total_frames": sum(int(record["frames"]) for record in records),
        "duration_s": sum(float(record["frames"]) / float(record["fps"]) for record in records),
        "task_counts": dict(sorted(task_counts.items())),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repo", default="HenryZhang/VLAReplica_SFT_data")
    parser.add_argument("--per_task", type=int, default=3)
    parser.add_argument("--validation_per_task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_path = Path(args.records)
    rows = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("repo") == args.repo]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)
    if not by_task:
        raise ValueError(f"No records found for {args.repo}")
    reserved_per_task = args.per_task + args.validation_per_task
    too_small = {task: len(group) for task, group in by_task.items() if len(group) <= reserved_per_task}
    if too_small:
        raise ValueError(f"Tasks need more than {reserved_per_task} records: {too_small}")

    train = []
    validation = []
    test = []
    heldout = []
    for task, group in sorted(by_task.items()):
        ordered = sorted(group, key=lambda row: rank(args.seed, task, int(row["episode"])))
        train.extend(ordered[: args.per_task])
        validation.extend(ordered[args.per_task : reserved_per_task])
        test.extend(ordered[reserved_per_task:])
        heldout.extend(ordered[args.per_task:])
    train.sort(key=lambda row: (str(row["task"]), int(row["episode"])))
    validation.sort(key=lambda row: (str(row["task"]), int(row["episode"])))
    test.sort(key=lambda row: (str(row["task"]), int(row["episode"])))
    heldout.sort(key=lambda row: (str(row["task"]), int(row["episode"])))

    train_keys = {(row["repo"], int(row["episode"]), row["camera"]) for row in train}
    validation_keys = {(row["repo"], int(row["episode"]), row["camera"]) for row in validation}
    test_keys = {(row["repo"], int(row["episode"]), row["camera"]) for row in test}
    heldout_keys = {(row["repo"], int(row["episode"]), row["camera"]) for row in heldout}
    if train_keys & heldout_keys:
        raise ValueError("Fine-tune and held-out records overlap")
    if validation_keys & test_keys:
        raise ValueError("Validation and test records overlap")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = "data/unseen_physics/shared_pipeline/tokenizer_input/test/records.jsonl"
    train_payload = summarize(train, source, args.repo, args.seed, args.per_task)
    validation_payload = summarize(
        validation, source, args.repo, args.seed, args.validation_per_task
    )
    test_payload = summarize(test, source, args.repo, args.seed, 0)
    heldout_payload = summarize(heldout, source, args.repo, args.seed, args.per_task)
    (output_dir / f"train_{args.per_task}_per_task_seed{args.seed}.json").write_text(
        json.dumps(train_payload, indent=2) + "\n"
    )
    (output_dir / f"heldout_seed{args.seed}.json").write_text(
        json.dumps(heldout_payload, indent=2) + "\n"
    )
    (output_dir / f"validation_{args.validation_per_task}_per_task_seed{args.seed}.json").write_text(
        json.dumps(validation_payload, indent=2) + "\n"
    )
    (output_dir / f"test_seed{args.seed}.json").write_text(
        json.dumps(test_payload, indent=2) + "\n"
    )
    summary = {
        "repo": args.repo,
        "selection_seed": args.seed,
        "selection_method": "sha256 rank within task",
        "per_task": args.per_task,
        "num_tasks": len(by_task),
        "train_trajectories": len(train),
        "validation_trajectories": len(validation),
        "test_trajectories": len(test),
        "heldout_trajectories": len(heldout),
        "train_duration_s": train_payload["duration_s"],
        "validation_duration_s": validation_payload["duration_s"],
        "test_duration_s": test_payload["duration_s"],
        "heldout_duration_s": heldout_payload["duration_s"],
    }
    (output_dir / f"summary_seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
