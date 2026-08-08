"""Measure the downloaded dataset and write the metadata the packer consumes.

Two phases, independently runnable because they cost very differently:

  --cameras   classify every camera fixed vs moving from pixels (decodes video)
  --episodes  per-episode trim bounds and quality flags (reads parquet only)

Camera names are not trustworthy -- roughly a third of cameras named `laptop`,
`phone` or `base` are wrist-mounted -- so views are judged on whether the image
itself holds still. Nothing here mutates the dataset; every decision lands in a
JSON artifact and can be re-derived with different thresholds.

    uv run python scripts/so101/analyze.py
    uv run python scripts/so101/analyze.py --episodes    # re-tune trims cheaply
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from visionary.dataset import decode_video_window

EP_PARQUET = re.compile(r"episode_(\d+)\.parquet$")

# Preference order used only to break ties between cameras already verified as
# fixed. It never decides fixed-vs-moving.
NAME_PRIORITY = [
    "front", "frontcam", "top", "above", "overhead", "topdown", "topcam",
    "realsense_top", "realsense_side", "side", "side_view", "side_camera",
    "third", "scene", "context", "global", "laptop", "webcam", "base", "phone",
]

# Camera verdict, calibrated on 8 datasets whose camera is literally named
# `wrist` (static_frac <= 0.11, shift_p90 >= 0.65) plus visually confirmed fixed
# cameras (static_frac >= 0.46, and one at 0.16 with shift_p90 0.40 where the arm
# merely fills the frame). Both conditions are required: a close-up fixed camera
# can have few static pixels, but only a moving camera also translates globally.
STATIC_MAX = 0.30
SHIFT_MIN = 0.5
# Fixed cameras sit at shift_p90 p50 0.34 / p90 0.66, so anything well above that
# is decided by egomotion alone and is not ambiguous.
BORDERLINE_SHIFT_MAX = 2.0

# Episode trim. Motion is deg/s (|dA| * fps) so thresholds are fps-invariant, and
# smoothed over ~0.5s so a single-frame blip cannot anchor the start at frame 0.
MOTION_DEG_S = 3.0
WINDOW_S = 0.5
PAD_S = 0.5
MIN_KEEP_S = 2.0
MID_GAP_S = 3.0
# Fast teleop is normal (peak velocity p50 267 deg/s, p99 708), so a flat cap
# only flags brisk motion. A real glitch is an isolated spike.
JUMP_DEG_S = 800.0
JUMP_RATIO = 5.0


def cam_short(key: str) -> str:
    return key.replace("observation.images.", "").lower()


def cam_rank(camera: str) -> int:
    for i, token in enumerate(NAME_PRIORITY):
        if token in camera:
            return i
    return len(NAME_PRIORITY)


# --------------------------------------------------------------------------
# phase 1: camera motion
# --------------------------------------------------------------------------


def camera_metrics(path: str, n_frames: int = 24, hw: tuple[int, int] = (96, 128)) -> dict:
    frames = decode_video_window(path, 0, 10**9, decode_hw=hw)
    if len(frames) < 4:
        raise ValueError(f"too few frames: {len(frames)}")
    idx = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)
    gray = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in idx]).astype(np.float32)

    std = gray.std(axis=0)
    static_frac = float((std < 8.0).mean())

    h, w = std.shape
    bh, bw = h // 5, w // 5
    edge = np.concatenate([std[:bh].ravel(), std[-bh:].ravel(),
                           std[:, :bw].ravel(), std[:, -bw:].ravel()])
    edge_ratio = float(edge.mean() / max(std[bh:-bh, bw:-bw].mean(), 1e-6))

    shifts = []
    for a, b in zip(gray[:-1], gray[1:]):
        (dx, dy), _ = cv2.phaseCorrelate(a, b)
        shifts.append((dx**2 + dy**2) ** 0.5)
    shift_p90 = float(np.percentile(shifts, 90)) if shifts else 0.0

    return {"static_frac": static_frac, "edge_ratio": edge_ratio, "shift_p90": shift_p90,
            "mean_absdiff": float(np.abs(np.diff(gray, axis=0)).mean())}


def is_moving(m: dict) -> bool:
    return m["static_frac"] < STATIC_MAX and m["shift_p90"] > SHIFT_MIN


def is_borderline(m: dict) -> bool:
    # The valley between the two static_frac modes (moving mean 0.04, fixed mean
    # 0.62), but only when egomotion does not already settle it: a camera that
    # translates by 20 px is moving no matter where static_frac lands.
    return 0.15 < m["static_frac"] < 0.45 and m["shift_p90"] < BORDERLINE_SHIFT_MAX


def scan_cameras(root: Path, episodes: int) -> list[dict]:
    rows = []
    for repo_dir in sorted(d for d in root.iterdir() if d.is_dir() and "__" in d.name):
        info_path = repo_dir / "meta" / "info.json"
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text())
        repo = repo_dir.name.replace("__", "/", 1)
        for key in [k for k in info["features"] if k.startswith("observation.images")]:
            per_episode = []
            for mp4 in sorted(repo_dir.glob(f"videos/*/{key}/episode_*.mp4"))[:episodes]:
                try:
                    per_episode.append(camera_metrics(str(mp4)))
                except Exception:  # noqa: BLE001
                    continue
            if not per_episode:
                continue
            row = {k: float(np.median([m[k] for m in per_episode])) for k in per_episode[0]}
            row.update(repo=repo, camera=cam_short(key), episodes_probed=len(per_episode))
            row["moving"] = is_moving(row)
            row["borderline"] = is_borderline(row)
            rows.append(row)
            print(f"{repo[:38]:38s} {row['camera'][:14]:14s} static={row['static_frac']:.2f} "
                  f"shift={row['shift_p90']:5.2f} -> {'MOVING' if row['moving'] else 'fixed'}",
                  flush=True)
    return rows


def write_camera_artifacts(rows: list[dict], out_dir: Path) -> None:
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)

    views, choice, dropped = {}, {}, []
    for repo, cams in by_repo.items():
        fixed = [c for c in cams if not c["moving"]]
        moving = [c for c in cams if c["moving"]]
        best = (min(fixed, key=lambda c: (cam_rank(c["camera"]), -c["static_frac"]))
                if fixed else None)
        if best:
            choice[repo] = f"observation.images.{best['camera']}"
        else:
            dropped.append(repo)
        views[repo] = {
            "fixed": sorted(c["camera"] for c in fixed),
            "moving": sorted(c["camera"] for c in moving),
            "chosen": best["camera"] if best else None,
            "borderline": sorted(c["camera"] for c in cams if c["borderline"]),
        }

    json.dump(rows, open(out_dir / "camera_motion.json", "w"), indent=1)
    json.dump(views, open(out_dir / "camera_views.json", "w"), indent=1)
    json.dump(choice, open(out_dir / "camera_choice.json", "w"), indent=1)

    n_moving = sum(r["moving"] for r in rows)
    print(f"\n{len(rows)} cameras over {len(by_repo)} datasets: {n_moving} moving, "
          f"{len(rows) - n_moving} fixed ({sum(r['borderline'] for r in rows)} borderline)")
    print(f"{len(choice)} datasets keep a fixed view, {len(dropped)} dropped")


# --------------------------------------------------------------------------
# phase 2: episode quality
# --------------------------------------------------------------------------


def episode_qc(action: np.ndarray, fps: float) -> dict:
    n = len(action)
    duration = n / fps
    if n < 3:
        return {"n_frames": n, "duration_s": duration, "flags": ["too_short"], "keep": False}

    velocity = np.abs(np.diff(action, axis=0)).max(1) * fps
    window = max(int(round(WINDOW_S * fps)), 1)
    smooth = np.convolve(velocity, np.ones(window) / window, mode="same")
    active = smooth > MOTION_DEG_S
    if not active.any():
        return {"n_frames": n, "duration_s": duration, "active_frac": 0.0,
                "flags": ["no_motion"], "keep": False}

    idx = np.flatnonzero(active)
    pad = int(round(PAD_S * fps))
    start = max(int(idx[0]) - pad, 0)
    end = min(int(idx[-1]) + pad, n - 1)
    head_idle, tail_idle = start / fps, (n - 1 - end) / fps
    kept = (end - start) / fps

    gaps, run = [], 0
    for a in active[idx[0]:idx[-1] + 1]:
        run = 0 if a else run + 1
        gaps.append(run)
    mid_gap = (max(gaps) if gaps else 0) / fps

    flags = []
    if head_idle >= 1.0:
        flags.append("stale_head")
    if tail_idle >= 1.0:
        flags.append("stale_tail")
    if mid_gap >= MID_GAP_S:
        flags.append("mid_pause")
    p95 = float(np.percentile(velocity, 95))
    if velocity.max() > JUMP_DEG_S and velocity.max() > JUMP_RATIO * max(p95, 1e-6):
        flags.append("action_jump")
    if action[:, :5].std(0).max() < 1.0 and action[:, 5].std() > 1.0:
        flags.append("gripper_only")

    return {
        "n_frames": n, "duration_s": round(duration, 2), "start": start, "end": end,
        "kept_s": round(kept, 2), "head_idle_s": round(head_idle, 2),
        "tail_idle_s": round(tail_idle, 2), "mid_gap_s": round(mid_gap, 2),
        "active_frac": round(float(active.mean()), 3),
        "max_vel_deg_s": round(float(velocity.max()), 1),
        "flags": flags, "keep": kept >= MIN_KEEP_S,
    }


def scan_episodes(root: Path) -> list[dict]:
    rows = []
    for repo_dir in sorted(d for d in root.iterdir() if d.is_dir() and "__" in d.name):
        info_path = repo_dir / "meta" / "info.json"
        if not info_path.exists():
            continue
        fps = float(json.loads(info_path.read_text()).get("fps", 30))
        repo = repo_dir.name.replace("__", "/", 1)
        for f in sorted(repo_dir.glob("data/*/episode_*.parquet")):
            m = EP_PARQUET.search(f.name)
            if not m:
                continue
            try:
                action = np.asarray(pq.read_table(f).column("action").to_pylist(),
                                    dtype=np.float32)
            except Exception:  # noqa: BLE001
                rows.append({"repo": repo, "episode": int(m.group(1)),
                             "flags": ["unreadable"], "keep": False})
                continue
            row = episode_qc(action, fps)
            row.update(repo=repo, episode=int(m.group(1)), fps=fps)
            rows.append(row)
    return rows


def write_episode_artifacts(rows: list[dict], out_dir: Path) -> None:
    bounds: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for r in rows:
        if r.get("keep") and "start" in r:
            bounds[r["repo"]][str(r["episode"])] = [r["start"], r["end"]]

    json.dump(rows, open(out_dir / "episode_qc.json", "w"), indent=1)
    json.dump(bounds, open(out_dir / "episode_bounds.json", "w"), indent=1)

    kept = [r for r in rows if r.get("keep")]
    total = sum(r.get("duration_s", 0) for r in rows)
    after = sum(r.get("kept_s", 0) for r in kept)
    flags = Counter(f for r in rows for f in r.get("flags", []))
    print(f"\n{len(rows)} episodes over {len({r['repo'] for r in rows})} datasets")
    print(f"  keep {len(kept)}, drop {len(rows) - len(kept)}")
    print(f"  {total / 3600:.1f} h -> {after / 3600:.1f} h after trim "
          f"({(1 - after / max(total, 1e-9)) * 100:.0f}% removed)")
    print(f"  flags: {dict(flags.most_common())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", default="data/so101/hf")
    ap.add_argument("--out-dir", default="artifacts/so101")
    ap.add_argument("--cameras", action="store_true", help="run only the camera phase")
    ap.add_argument("--episodes", action="store_true", help="run only the episode phase")
    ap.add_argument("--probe-episodes", type=int, default=1,
                    help="episodes sampled per camera when classifying")
    args = ap.parse_args()

    root = Path(args.local_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.cameras or not args.episodes:
        write_camera_artifacts(scan_cameras(root, args.probe_episodes), out)
    if args.episodes or not args.cameras:
        write_episode_artifacts(scan_episodes(root), out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
