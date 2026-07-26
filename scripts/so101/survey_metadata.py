"""Fetch each dataset's LeRobot meta/info.json (no video) and report the
distributions of fps, resolution, cameras, action/state dims, and totals.

    uv run python scripts/so101/survey_metadata.py --workers 48
"""

import argparse
import json
import urllib.request
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/meta/info.json"


def load_manifest(path: str) -> list[str]:
    raw = json.load(open(path))
    return [x.split(":", 1)[1] if ":" in x else x for x in raw]


def fetch_info(repo: str, timeout: float = 20.0) -> dict:
    url = RESOLVE.format(repo=repo)
    req = urllib.request.Request(url, headers={"User-Agent": "so101-survey/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            info = json.loads(r.read().decode("utf-8"))
        feats = info.get("features", {})
        cams = {k: v for k, v in feats.items() if k.startswith("observation.images")}
        cam_shapes = {k: tuple(v.get("shape", [])) for k, v in cams.items()}
        action = feats.get("action", {})
        state = feats.get("observation.state", {})
        return {
            "repo": repo,
            "ok": True,
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "action_shape": tuple(action.get("shape", [])),
            "action_names": action.get("names"),
            "state_shape": tuple(state.get("shape", [])),
            "n_cameras": len(cams),
            "camera_keys": sorted(cams.keys()),
            "camera_shapes": cam_shapes,
            "video_dtypes": sorted({v.get("dtype") for v in cams.values()}),
        }
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, Exception) as e:  # noqa: BLE001
        code = getattr(e, "code", None)
        return {"repo": repo, "ok": False, "error": f"{type(e).__name__}:{code or e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="scripts/so101/molmoact2_repo_list.json")
    ap.add_argument("--out", default="artifacts/so101/survey.json")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    repos = load_manifest(args.manifest)
    if args.limit:
        repos = repos[: args.limit]
    print(f"Surveying {len(repos)} datasets with {args.workers} workers...")

    results: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_info, r): r for r in repos}
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(repos)}")

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]

    def dist(key):
        return Counter(r[key] for r in ok).most_common()

    res_counter = Counter()
    cam_name_counter = Counter()
    action_dim_counter = Counter()
    for r in ok:
        for shp in r["camera_shapes"].values():
            res_counter[str(shp)] += 1
        for k in r["camera_keys"]:
            cam_name_counter[k.replace("observation.images.", "")] += 1
        action_dim_counter[str(r["action_shape"])] += 1

    summary = {
        "n_total": len(results),
        "n_ok": len(ok),
        "n_failed": len(bad),
        "total_episodes": sum(r["total_episodes"] or 0 for r in ok),
        "total_frames": sum(r["total_frames"] or 0 for r in ok),
        "codebase_version": dist("codebase_version"),
        "robot_type": dist("robot_type"),
        "fps": dist("fps"),
        "n_cameras": dist("n_cameras"),
        "action_shape": action_dim_counter.most_common(),
        "state_shape": Counter(str(r["state_shape"]) for r in ok).most_common(),
        "camera_resolution": res_counter.most_common(20),
        "camera_key_names_top40": cam_name_counter.most_common(40),
        "failures_sample": [r["repo"] + " :: " + r["error"] for r in bad[:30]],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=1)

    print("\n===== SO-100/101 COMMUNITY DATASET COMPATIBILITY SURVEY =====")
    print(json.dumps(summary, indent=1))
    print(f"\nWrote full per-dataset results to {args.out}")


if __name__ == "__main__":
    main()
