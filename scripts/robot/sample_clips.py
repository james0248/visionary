"""Render annotated sample clips from the processed tree, driven by
pack_tokenizer_dataset.py's per-source analysis JSONs, so the automated
verdicts can be checked by eye.

  --flags     examples of each episode QC flag; trims draw the keep/cut boundary
  --cameras   moving vs fixed vs borderline views, with the deciding metrics

Every verdict here came from a threshold someone chose, so being able to look at
what those thresholds actually selected is the point.

    uv run python scripts/robot/sample_clips.py \
        --analysis-dir artifacts/robot/analysis --processed-dir data/robot/processed
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from visionary.dataset import decode_video_window

FLAG_DETAIL = {
    "stale_head": "head_idle_s",
    "stale_tail": "tail_idle_s",
    "mid_pause": "mid_gap_s",
    "action_jump": "max_vel_deg_s",
    "no_motion": None,
    "too_short": "duration_s",
    "gripper_only": None,
}


def load_analyses(analysis_dir: Path) -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text()) for path in sorted(analysis_dir.rglob("*.json"))}


def read_member(tar_path: Path, spec: dict) -> bytes:
    with open(tar_path, "rb") as f:
        f.seek(spec["offset"])
        return f.read(spec["size"])


def find_video(processed_dir: Path, source: str, camera: str | None, episode: int | None = None) -> bytes | None:
    for source_dir in (
        processed_dir.glob(f"*/{source}") if not (processed_dir / source).exists() else [processed_dir / source]
    ):
        for tar_path in sorted(source_dir.glob("shard-*.tar")):
            index = json.loads(tar_path.with_suffix(".index.json").read_text())
            for name in sorted(index):
                files = index[name]
                meta = json.loads(read_member(tar_path, files["meta.json"]))
                if episode is not None and meta["episode"] != episode:
                    continue
                if camera is not None and meta["camera"]["orig_name"] != camera:
                    continue
                return read_member(tar_path, files["video.mp4"])
    return None


def render(
    video: bytes,
    out: Path,
    title: str,
    subtitle: str,
    colour: tuple[int, int, int],
    bounds: tuple[int, int] | None = None,
    fps: float = 30,
    hw: tuple[int, int] = (180, 240),
    max_seconds: float | None = None,
) -> bool:
    try:
        limit = int(max_seconds * fps) if max_seconds else 10**9
        frames = decode_video_window(video, 0, limit, decode_hw=hw)
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
        cv2.putText(frame, f"{subtitle}  {i / fps:5.1f}s", (4, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1)
        out_frames.append(frame)

    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack(out_frames)
    arr = arr[:, : arr.shape[1] // 2 * 2, : arr.shape[2] // 2 * 2]
    imageio.mimwrite(out, arr, fps=fps, codec="libx264", quality=5, macro_block_size=None)
    return True


def best_fixed_camera(analysis: dict) -> str | None:
    fixed = [(c, m) for c, m in analysis["cameras"].items() if not m["moving"]]
    if not fixed:
        return None
    return max(fixed, key=lambda cm: cm[1]["static_frac"])[0]


def sample_flags(analyses: dict[str, dict], processed_dir: Path, out_dir: Path, per_group: int) -> None:
    by_flag = defaultdict(list)
    for source, analysis in analyses.items():
        for ep, qc in analysis["episodes"].items():
            for flag in qc.get("flags", []):
                row = dict(qc)
                row.update(source=source, episode=int(ep), fps=analysis["fps"])
                by_flag[flag].append(row)

    for flag, records in sorted(by_flag.items(), key=lambda kv: -len(kv[1])):
        key = FLAG_DETAIL.get(flag)
        if key:
            records = sorted(records, key=lambda r: -(r.get(key) or 0))
        seen, picks = set(), []
        for r in records:
            if r["source"] in seen:
                continue
            seen.add(r["source"])
            picks.append(r)
            if len(picks) >= per_group:
                break

        written = 0
        for r in picks:
            camera = best_fixed_camera(analyses[r["source"]])
            video = find_video(processed_dir, r["source"], camera, r["episode"])
            if video is None:
                continue
            detail = f"{key}={r.get(key)}" if key else ""
            bounds = (r["start"], r["end"]) if "start" in r else None
            name = f"{r['source']}_ep{r['episode']:03d}.mp4"
            if render(
                video,
                out_dir / "flags" / flag / name,
                f"{flag}  {detail}",
                f"{r['source']} ep{r['episode']}",
                (60, 200, 255),
                bounds,
                fps=r.get("fps", 30),
            ):
                written += 1
        print(f"  {flag:16s} {len(by_flag[flag]):6d} episodes -> {written} samples")


def sample_cameras(analyses: dict[str, dict], processed_dir: Path, out_dir: Path, per_group: int) -> None:
    rows = []
    for source, analysis in analyses.items():
        for camera, metrics in analysis["cameras"].items():
            row = dict(metrics)
            row.update(source=source, camera=camera)
            rows.append(row)
    groups = defaultdict(list)
    for r in rows:
        groups["moving" if r["moving"] else ("borderline" if r.get("borderline") else "fixed")].append(r)

    for group, records in groups.items():
        if group == "moving":  # spread across camera NAMES
            by_name = defaultdict(list)
            for r in records:
                by_name[r["camera"]].append(r)
            picks, seen = [], set()
            for name in sorted(by_name, key=lambda n: -len(by_name[n])):
                for r in sorted(by_name[name], key=lambda r: r["static_frac"]):
                    if r["source"] in seen:
                        continue
                    seen.add(r["source"])
                    picks.append(r)
                    break
                if len(picks) >= per_group:
                    break
        else:  # spread across the static_frac range
            records = sorted(records, key=lambda r: r["static_frac"])
            step = max(len(records) // max(per_group, 1), 1)
            picks = records[::step][:per_group]

        written = 0
        for r in picks:
            video = find_video(processed_dir, r["source"], r["camera"])
            if video is None:
                continue
            name = f"{r['static_frac']:.2f}_{r['source']}_{r['camera']}.mp4"
            if render(
                video,
                out_dir / "cameras" / group / name,
                f'{"MOVING" if r["moving"] else "FIXED"}  "{r["camera"]}"',
                f"static={r['static_frac']:.2f} shift={r['shift_p90']:.1f} {r['source']}",
                (80, 80, 255) if r["moving"] else (80, 255, 80),
                max_seconds=8,
            ):
                written += 1
        print(f"  {group:12s} {len(groups[group]):6d} cameras -> {written} samples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis-dir", required=True, help="dir of per-source analysis JSONs from pack_tokenizer_dataset.py"
    )
    ap.add_argument("--processed-dir", required=True, help="local processed tree: [<dataset>/]<source>/shard-*.tar")
    ap.add_argument("--out-dir", default="outputs/samples")
    ap.add_argument("--per-group", type=int, default=6)
    ap.add_argument("--flags", action="store_true", help="only episode-flag samples")
    ap.add_argument("--cameras", action="store_true", help="only camera samples")
    args = ap.parse_args()

    analyses = load_analyses(Path(args.analysis_dir))
    processed_dir = Path(args.processed_dir)
    out = Path(args.out_dir)
    if args.flags or not args.cameras:
        sample_flags(analyses, processed_dir, out, args.per_group)
    if args.cameras or not args.flags:
        sample_cameras(analyses, processed_dir, out, args.per_group)
    size = sum(f.stat().st_size for f in out.rglob("*.mp4")) / 1024**2
    print(f"\n{out}/  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
