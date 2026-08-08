import argparse
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path


def run(cmd: list[str], check: bool = True) -> int:
    return subprocess.run(cmd, check=check).returncode


def gcs_exists(uri: str) -> bool:
    return subprocess.run(
        ["gcloud", "storage", "ls", uri], capture_output=True
    ).returncode == 0


def push(local: Path, uri: str) -> None:
    shutil.rmtree(local / ".cache", ignore_errors=True)
    run(["gcloud", "storage", "rsync", "--recursive", str(local), uri])


def fetch(repo_id: str, local: Path, workers: int, attempts: int = 8) -> None:
    from huggingface_hub import snapshot_download

    for attempt in range(attempts):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local),
                max_workers=workers,
                token=os.environ.get("HF_TOKEN"),
            )
            return
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            delay = min(600, (2 ** attempt) * 30) + random.uniform(0, 20)
            print(f"  retry {attempt + 1}/{attempts} after {delay:.0f}s: {type(exc).__name__}", flush=True)
            time.sleep(delay)


def archive_so101(args) -> None:

    repos = json.load(open(args.manifest))
    repos = [r if isinstance(r, str) else r["repo"] for r in repos]
    mine = [r for i, r in enumerate(sorted(repos)) if i % args.num_shards == args.shard]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} repos", flush=True)

    for n, repo in enumerate(mine, 1):
        dest = f"{args.dest}/so101/{repo}"
        if gcs_exists(f"{dest}/_DONE"):
            print(f"[{n}/{len(mine)}] skip {repo}", flush=True)
            continue
        local = Path(args.workdir) / repo.replace("/", "__")
        try:
            fetch(repo, local, args.workers)
        except Exception as exc:
            print(f"[{n}/{len(mine)}] FAILED {repo}: {type(exc).__name__} {exc}", flush=True)
            shutil.rmtree(local, ignore_errors=True)
            continue
        push(local, dest)
        subprocess.run(
            ["gcloud", "storage", "cp", "-", f"{dest}/_DONE"],
            input=b"ok", check=False,
        )
        shutil.rmtree(local, ignore_errors=True)
        print(f"[{n}/{len(mine)}] done {repo}", flush=True)


def archive_bridge(args) -> None:
    dest = f"{args.dest}/bridge_v2"
    if gcs_exists(f"{dest}/_DONE"):
        print("bridge already archived", flush=True)
        return
    local = Path(args.workdir) / "bridge_v2"
    fetch("nvidia/BridgeData2_LeRobot_v3", local, args.workers)
    push(local, dest)
    subprocess.run(["gcloud", "storage", "cp", "-", f"{dest}/_DONE"], input=b"ok", check=False)
    shutil.rmtree(local, ignore_errors=True)
    print("bridge done", flush=True)


def archive_soar(args) -> None:
    dest = f"{args.dest}/soar"
    if gcs_exists(f"{dest}/_DONE"):
        print("soar already archived", flush=True)
        return
    local = Path(args.workdir) / "soar"
    local.mkdir(parents=True, exist_ok=True)
    run([
        "wget", "--recursive", "--no-parent", "--no-host-directories",
        "--cut-dirs=2", "--reject", "index.html*", "--continue",
        "--tries=5", "--directory-prefix", str(local),
        "https://rail.eecs.berkeley.edu/datasets/soar_release/",
    ], check=False)
    push(local, dest)
    subprocess.run(["gcloud", "storage", "cp", "-", f"{dest}/_DONE"], input=b"ok", check=False)
    shutil.rmtree(local, ignore_errors=True)
    print("soar done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["so101", "bridge", "soar"])
    ap.add_argument("--dest", default="gs://visionary-uc1/raw")
    ap.add_argument("--manifest", default="/tmp/download_pool.json")
    ap.add_argument("--workdir", default="/var/tmp/archive")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    {"so101": archive_so101, "bridge": archive_bridge, "soar": archive_soar}[args.kind](args)
    print("ARCHIVE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
