"""Download tier datasets from the HF Hub into a local tree that mirrors the
LeRobot layout (meta/ + data/*.parquet + videos/<primary camera>/*.mp4), ready
for GCS upload and local analysis. Resumable: existing files are skipped.

    uv run python scripts/so101/download_datasets.py \
        --manifest artifacts/so101/tier2_clean.json --out-dir data/so101/hf
"""

import argparse
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def load_repos(manifest: str, min_episodes: int) -> list[str]:
    """Accept either a plain repo list or survey.json, filtering the latter.

    Selection lives here rather than in a separate curation step: the only
    dataset-level filter that survived review is a minimum episode count, and
    everything else about a dataset is judged from its pixels after download.
    """
    raw = json.load(open(manifest))
    if isinstance(raw, dict) and "results" in raw:
        rows = [r for r in raw["results"] if r.get("ok")]
        kept = [r for r in rows if (r.get("total_episodes") or 0) >= min_episodes]
        log(f"survey: {len(rows)} datasets, {len(kept)} with >={min_episodes} episodes "
            f"({sum(r.get('total_episodes') or 0 for r in kept):,} episodes, "
            f"{sum(r.get('total_frames') or 0 for r in kept):,} frames)")
        return [r["repo"] for r in kept]
    return [x.split(":", 1)[-1] if ":" in x else x for x in raw]


def dir_bytes(path: Path) -> int:
    # Tolerate files vanishing mid-walk (e.g. a concurrent purge): rglob lists
    # them, then stat() races the deletion. Size accounting must never fail the
    # download it is only reporting on.
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def download_repo(repo: str, out_root: Path, camera: str | None,
                  all_cameras: bool, max_workers: int, probe: bool = False,
                  camera_map: dict | None = None, no_videos: bool = False,
                  views: dict | None = None) -> dict:
    local = out_root / repo.replace("/", "__")
    info = json.load(open(hf_hub_download(repo, "meta/info.json", repo_type="dataset")))
    cam_keys = [k for k in info["features"] if k.startswith("observation.images")]

    if probe:
        # One episode of every camera, enough for analyze.py to classify each
        # view before committing bandwidth to the full dataset.
        snapshot_download(
            repo, repo_type="dataset", local_dir=local, max_workers=max_workers,
            allow_patterns=["meta/info.json", "meta/episodes.jsonl",
                            "videos/*/*/episode_000000.mp4"],
        )
        return {"repo": repo, "ok": True, "camera": None, "probe": True,
                "cameras": len(cam_keys), "bytes": dir_bytes(local)}

    if camera_map is not None:
        cam_key = camera_map.get(repo)
        if cam_key is None:
            return {"repo": repo, "ok": False, "error": "no verified fixed camera"}
    elif camera and f"observation.images.{camera}" in cam_keys:
        cam_key = f"observation.images.{camera}"
    else:
        cam_key = cam_keys[0] if cam_keys else None
    if cam_key is None and not all_cameras:
        return {"repo": repo, "ok": False, "error": "no camera"}

    patterns = ["meta/*", "data/*/*.parquet"]
    if not no_videos:
        if views is not None:
            # every pixel-verified FIXED view, and nothing else. Prevents
            # re-fetching moving-camera video that purge_moving_cameras removed.
            fixed = views.get(repo, {}).get("fixed")
            if fixed is None:                    # unclassified repo -> take all
                patterns.append("videos/*/*/*.mp4")
            elif not fixed:
                return {"repo": repo, "ok": False, "error": "no fixed view"}
            else:
                patterns += [f"videos/*/observation.images.{c}/*.mp4" for c in fixed]
        else:
            patterns.append("videos/*/*/*.mp4" if all_cameras
                            else f"videos/*/{cam_key}/*.mp4")
    snapshot_download(
        repo, repo_type="dataset", local_dir=local, max_workers=max_workers,
        allow_patterns=patterns,
    )
    # verify rather than trust the absence of an exception (rate limits can no-op)
    if not list(local.glob("data/*/*.parquet")):
        return {"repo": repo, "ok": False, "error": "no parquet after download"}
    return {
        "repo": repo, "ok": True, "camera": cam_key,
        "episodes": int(info.get("total_episodes", 0)),
        "frames": int(info.get("total_frames", 0)),
        "bytes": dir_bytes(local),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="data/so101/hf")
    ap.add_argument("--camera", default=None, help="force a camera key suffix")
    ap.add_argument("--all-cameras", action="store_true", default=True,
                    help="fetch every camera (default; filtering happens later from pixels)")
    ap.add_argument("--single-camera", dest="all_cameras", action="store_false",
                    help="fetch only the chosen/primary camera")
    ap.add_argument("--fixed-only", action="store_true",
                    help="download only pixel-verified fixed views (camera_views.json)")
    ap.add_argument("--views", default="artifacts/so101/camera_views.json")
    ap.add_argument("--no-videos", action="store_true",
                    help="fetch meta + parquet only (fast; unblocks QC)")
    ap.add_argument("--probe", action="store_true",
                    help="fetch only episode 0 of every camera, for wrist detection")
    ap.add_argument("--camera-map", default=None,
                    help="camera_choice.json from analyze.py; overrides name picking")
    ap.add_argument("--repo-workers", type=int, default=6, help="datasets in flight")
    ap.add_argument("--file-workers", type=int, default=8, help="files in flight per dataset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--min-episodes", type=int, default=5,
                    help="drop datasets with fewer episodes (mostly abandoned *_test uploads)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the selection without downloading")
    ap.add_argument("--min-free-gb", type=float, default=10.0,
                    help="abort if free disk falls below this")
    args = ap.parse_args()

    repos = load_repos(args.manifest, args.min_episodes)
    camera_map = json.load(open(args.camera_map)) if args.camera_map else None
    views = json.load(open(args.views)) if (args.fixed_only and Path(args.views).exists()) else None
    if camera_map is not None:
        repos = [r for r in repos if r in camera_map]
    repos = repos[args.skip:]
    if args.limit:
        repos = repos[: args.limit]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(out_root).free / 1024**3
    log(f"Downloading {len(repos)} datasets -> {out_root}  ({free:.0f} GB free)")
    if args.dry_run:
        log("dry run: nothing downloaded")
        return

    done = 0
    total_bytes = 0
    results = []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.repo_workers) as ex:
        futs = {
            ex.submit(download_repo, r, out_root, args.camera,
                      args.all_cameras, args.file_workers, args.probe, camera_map,
                      args.no_videos, views): r
            for r in repos
        }
        for fut in as_completed(futs):
            repo = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = {"repo": repo, "ok": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}
            results.append(res)
            done += 1
            total_bytes += res.get("bytes", 0)
            elapsed = time.monotonic() - start
            rate = total_bytes / max(elapsed, 1e-6) / 1024**2
            eta = (len(repos) - done) * (elapsed / max(done, 1)) / 60
            free_gb = shutil.disk_usage(out_root).free / 1024**3
            if free_gb < args.min_free_gb:
                log(f"ABORT: only {free_gb:.1f} GB free (< --min-free-gb "
                    f"{args.min_free_gb}). Re-run to resume after freeing space.")
                for f2 in futs:
                    f2.cancel()
                break
            status = "ok" if res["ok"] else f"FAIL {res.get('error', '')[:60]}"
            log(f"[{done}/{len(repos)}] {repo} {status} | "
                f"{total_bytes / 1024**3:.1f} GB, {rate:.1f} MB/s, ETA {eta:.0f} min")
            json.dump(results, open(out_root / "_download_status.json", "w"), indent=1)

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    log(f"\nDone: {len(ok)} ok, {len(bad)} failed, {total_bytes / 1024**3:.1f} GB in "
        f"{(time.monotonic() - start) / 60:.0f} min")
    if bad:
        log("Failures (re-run to retry; downloads resume):")
        for r in bad[:20]:
            log(f"  {r['repo']}: {r.get('error')}")


if __name__ == "__main__":
    main()
