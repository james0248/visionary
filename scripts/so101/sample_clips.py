"""Render annotated sample clips from the source tree, driven by analyze.py's
metadata, so the automated verdicts can be checked by eye.

  --flags     examples of each episode QC flag; trims draw the keep/cut boundary
  --cameras   moving vs fixed vs borderline views, with the deciding metrics

Every verdict here came from a threshold someone chose, so being able to look at
what those thresholds actually selected is the point.

    uv run python scripts/so101/sample_clips.py --per-group 6
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from visionary.dataset import decode_video_window

FLAG_DETAIL = {
    "stale_head": "head_idle_s", "stale_tail": "tail_idle_s", "mid_pause": "mid_gap_s",
    "action_jump": "max_vel_deg_s", "no_motion": None, "too_short": "duration_s",
    "gripper_only": None,
}


def find_video(repo: str, camera: str | None, episode: int | None = None) -> str | None:
    cam = f"observation.images.{camera}" if camera else "*"
    ep = f"episode_{episode:06d}" if episode is not None else "episode_*"
    hits = sorted(glob.glob(f"data/so101/hf/{repo.replace('/', '__')}/videos/*/{cam}/{ep}.mp4"))
    return hits[0] if hits else None


def render(path: str, out: Path, title: str, subtitle: str, colour: tuple[int, int, int],
           bounds: tuple[int, int] | None = None, fps: float = 30,
           hw: tuple[int, int] = (180, 240), max_seconds: float | None = None) -> bool:
    try:
        limit = int(max_seconds * fps) if max_seconds else 10**9
        frames = decode_video_window(path, 0, limit, decode_hw=hw)
    except Exception:  # noqa: BLE001
        return False
    if len(frames) < 2:
        return False

    out_frames = []
    for i, frame in enumerate(frames):
        frame = frame.copy()
        shade = colour
        if bounds is not None:
            keep = bounds[0] <= i <= bounds[1]
            if not keep:
                frame = (frame * 0.42).astype(np.uint8)
            shade = (80, 255, 80) if keep else (80, 80, 255)
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), shade, 3)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(frame, title, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, shade, 1)
        cv2.putText(frame, f"{subtitle}  {i / fps:5.1f}s", (4, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1)
        out_frames.append(frame)

    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack(out_frames)
    arr = arr[:, : arr.shape[1] // 2 * 2, : arr.shape[2] // 2 * 2]
    imageio.mimwrite(out, arr, fps=fps, codec="libx264", quality=5, macro_block_size=None)
    return True


def sample_flags(out_dir: Path, per_group: int) -> None:
    rows = json.load(open("artifacts/so101/episode_qc.json"))
    views_path = Path("artifacts/so101/camera_views.json")
    views = json.loads(views_path.read_text()) if views_path.exists() else {}

    by_flag = defaultdict(list)
    for r in rows:
        for f in r.get("flags", []):
            by_flag[f].append(r)

    for flag, records in sorted(by_flag.items(), key=lambda kv: -len(kv[1])):
        key = FLAG_DETAIL.get(flag)
        if key:
            records = sorted(records, key=lambda r: -(r.get(key) or 0))
        seen, picks = set(), []
        for r in records:
            if r["repo"] in seen:
                continue
            seen.add(r["repo"])
            picks.append(r)
            if len(picks) >= per_group:
                break

        written = 0
        for r in picks:
            camera = views.get(r["repo"], {}).get("chosen")
            path = find_video(r["repo"], camera, r["episode"])
            if not path:
                continue
            detail = f"{key}={r.get(key)}" if key else ""
            bounds = (r["start"], r["end"]) if "start" in r else None
            name = f"{r['repo'].replace('/', '__')}_ep{r['episode']:03d}.mp4"
            if render(path, out_dir / "flags" / flag / name, f"{flag}  {detail}",
                      f"{r['repo']} ep{r['episode']}", (60, 200, 255), bounds,
                      fps=r.get("fps", 30)):
                written += 1
        print(f"  {flag:16s} {len(by_flag[flag]):6d} episodes -> {written} samples")


def sample_cameras(out_dir: Path, per_group: int) -> None:
    rows = json.load(open("artifacts/so101/camera_motion.json"))
    groups = defaultdict(list)
    for r in rows:
        groups["moving" if r["moving"] else
               ("borderline" if r.get("borderline") else "fixed")].append(r)

    for group, records in groups.items():
        if group == "moving":                 # spread across camera NAMES
            by_name = defaultdict(list)
            for r in records:
                by_name[r["camera"]].append(r)
            picks, seen = [], set()
            for name in sorted(by_name, key=lambda n: -len(by_name[n])):
                for r in sorted(by_name[name], key=lambda r: r["static_frac"]):
                    if r["repo"] in seen:
                        continue
                    seen.add(r["repo"])
                    picks.append(r)
                    break
                if len(picks) >= per_group:
                    break
        else:                                  # spread across the static_frac range
            records = sorted(records, key=lambda r: r["static_frac"])
            step = max(len(records) // max(per_group, 1), 1)
            picks = records[::step][:per_group]

        written = 0
        for r in picks:
            path = find_video(r["repo"], r["camera"])
            if not path:
                continue
            name = f"{r['static_frac']:.2f}_{r['repo'].replace('/', '__')}_{r['camera']}.mp4"
            if render(path, out_dir / "cameras" / group / name,
                      f"{'MOVING' if r['moving'] else 'FIXED'}  \"{r['camera']}\"",
                      f"static={r['static_frac']:.2f} shift={r['shift_p90']:.1f} {r['repo']}",
                      (80, 80, 255) if r["moving"] else (80, 255, 80), max_seconds=8):
                written += 1
        print(f"  {group:12s} {len(groups[group]):6d} cameras -> {written} samples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/samples")
    ap.add_argument("--per-group", type=int, default=6)
    ap.add_argument("--flags", action="store_true", help="only episode-flag samples")
    ap.add_argument("--cameras", action="store_true", help="only camera samples")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if args.flags or not args.cameras:
        sample_flags(out, args.per_group)
    if args.cameras or not args.flags:
        sample_cameras(out, args.per_group)
    size = sum(f.stat().st_size for f in out.rglob("*.mp4")) / 1024**2
    print(f"\n{out}/  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
