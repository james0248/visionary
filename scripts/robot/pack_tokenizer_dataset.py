import argparse
import io
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

from visionary.dataset import decode_video_window

GB = 1024**3

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
MOTION_DEG_S = 6.0
WINDOW_S = 0.5
PAD_S = 0.5
MIN_KEEP_S = 2.0
MID_GAP_S = 3.0
# Fast teleop is normal (peak velocity p50 267 deg/s, p99 708), so a flat cap
# only flags brisk motion. A real glitch is an isolated spike.
JUMP_DEG_S = 800.0
JUMP_RATIO = 5.0


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def is_gcs(uri):
    return str(uri).startswith("gs://")


def gcs_exists(uri):
    return subprocess.run(["gcloud", "storage", "ls", uri], capture_output=True).returncode == 0


def gcs_rsync(src, dst):
    run(["gcloud", "storage", "rsync", "--recursive", str(src), str(dst)])


def gcs_cp(src, dst):
    run(["gcloud", "storage", "cp", str(src), str(dst)])


def list_sources(processed, dataset):
    prefix = f"{processed}/{dataset}"
    if is_gcs(processed):
        out = run(["gcloud", "storage", "ls", prefix + "/"], text=True).stdout
        return sorted(line.rstrip("/").rsplit("/", 1)[-1] for line in out.splitlines() if line.endswith("/"))
    return sorted(d.name for d in Path(prefix).iterdir() if d.is_dir())


def load_droplist(checklist):
    if not checklist or not Path(checklist).exists():
        return set()
    drop = set()
    for line in Path(checklist).read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 5 and parts[4].strip() == "n":
            drop.add(parts[1].strip())
    return drop


def read_member(tar_path, spec):
    with open(tar_path, "rb") as f:
        f.seek(spec["offset"])
        return f.read(spec["size"])


def load_entries(source_dir):
    entries = []
    for tar_path in sorted(Path(source_dir).glob("shard-*.tar")):
        index = json.loads(tar_path.with_suffix(".index.json").read_text())
        for name in sorted(index):
            files = index[name]
            meta = json.loads(read_member(tar_path, files["meta.json"]))
            entries.append((tar_path, files, meta))
    return entries


def camera_metrics(video, n_frames: int = 24, hw: tuple[int, int] = (96, 128)) -> dict:
    frames = decode_video_window(video, 0, 10**9, decode_hw=hw)
    if len(frames) < 4:
        raise ValueError(f"too few frames: {len(frames)}")
    idx = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)
    gray = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in idx]).astype(np.float32)

    std = gray.std(axis=0)
    static_frac = float((std < 8.0).mean())

    h, w = std.shape
    bh, bw = h // 5, w // 5
    edge = np.concatenate([std[:bh].ravel(), std[-bh:].ravel(), std[:, :bw].ravel(), std[:, -bw:].ravel()])
    edge_ratio = float(edge.mean() / max(std[bh:-bh, bw:-bw].mean(), 1e-6))

    shifts = []
    for a, b in zip(gray[:-1], gray[1:]):
        (dx, dy), _ = cv2.phaseCorrelate(a, b)
        shifts.append((dx**2 + dy**2) ** 0.5)
    shift_p90 = float(np.percentile(shifts, 90)) if shifts else 0.0

    return {
        "static_frac": static_frac,
        "edge_ratio": edge_ratio,
        "shift_p90": shift_p90,
        "mean_absdiff": float(np.abs(np.diff(gray, axis=0)).mean()),
    }


def is_moving(m: dict) -> bool:
    return m["static_frac"] < STATIC_MAX and m["shift_p90"] > SHIFT_MIN


def is_borderline(m: dict) -> bool:
    # The valley between the two static_frac modes (moving mean 0.04, fixed mean
    # 0.62), but only when egomotion does not already settle it: a camera that
    # translates by 20 px is moving no matter where static_frac lands.
    return 0.15 < m["static_frac"] < 0.45 and m["shift_p90"] < BORDERLINE_SHIFT_MAX


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
        return {
            "n_frames": n,
            "duration_s": duration,
            "active_frac": 0.0,
            "flags": ["no_motion"],
            "keep": False,
        }

    idx = np.flatnonzero(active)
    pad = int(round(PAD_S * fps))
    start = max(int(idx[0]) - pad, 0)
    end = min(int(idx[-1]) + pad, n - 1)
    head_idle, tail_idle = start / fps, (n - 1 - end) / fps
    kept = (end - start) / fps

    gaps, run_len = [], 0
    for a in active[idx[0] : idx[-1] + 1]:
        run_len = 0 if a else run_len + 1
        gaps.append(run_len)
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
    if action.shape[1] == 6 and action[:, :5].std(0).max() < 1.0 and action[:, 5].std() > 1.0:
        flags.append("gripper_only")

    return {
        "n_frames": n,
        "duration_s": round(duration, 2),
        "start": start,
        "end": end,
        "kept_s": round(kept, 2),
        "head_idle_s": round(head_idle, 2),
        "tail_idle_s": round(tail_idle, 2),
        "mid_gap_s": round(mid_gap, 2),
        "active_frac": round(float(active.mean()), 3),
        "max_vel_deg_s": round(float(velocity.max()), 1),
        "flags": flags,
        "keep": kept >= MIN_KEEP_S,
    }


ANALYSIS_PARAMS = {
    "motion_deg_s": MOTION_DEG_S,
    "pad_s": PAD_S,
    "window_s": WINDOW_S,
    "min_keep_s": MIN_KEEP_S,
    "static_max": STATIC_MAX,
    "shift_min": SHIFT_MIN,
}


def analyze_source(source, entries, fps, probe_episodes, trim):
    by_camera = defaultdict(list)
    for tar_path, files, meta in entries:
        by_camera[meta["camera"]["orig_name"]].append((tar_path, files, meta))

    cameras = {}
    for cam, cam_entries in sorted(by_camera.items()):
        per_episode = []
        for tar_path, files, meta in cam_entries[:probe_episodes]:
            try:
                per_episode.append(camera_metrics(read_member(tar_path, files["video.mp4"])))
            except Exception:  # noqa: BLE001
                continue
        if not per_episode:
            continue
        row = {k: float(np.median([m[k] for m in per_episode])) for k in per_episode[0]}
        row["episodes_probed"] = len(per_episode)
        row["moving"] = is_moving(row)
        row["borderline"] = is_borderline(row)
        meta0 = cam_entries[0][2]
        row["width"] = meta0.get("width")
        row["height"] = meta0.get("height")
        cameras[cam] = row

    episodes = {}
    for tar_path, files, meta in entries:
        ep = str(meta["episode"])
        if ep in episodes:
            continue
        with np.load(io.BytesIO(read_member(tar_path, files["frames.npz"]))) as d:
            actions = np.asarray(d["actions"], dtype=np.float32)
        if trim:
            episodes[ep] = episode_qc(actions, fps)
        else:
            n = len(actions)
            episodes[ep] = {
                "n_frames": n,
                "duration_s": round(n / fps, 2),
                "start": 0,
                "end": n - 1,
                "flags": [],
                "keep": n >= 3,
            }

    return {
        "source": source,
        "fps": fps,
        "cameras": cameras,
        "episodes": episodes,
        "params": {**ANALYSIS_PARAMS, "probe_episodes": probe_episodes, "trim": trim},
    }


def probe_dims(video_bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        path = tmp.name
    try:
        out = run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                path,
            ],
            text=True,
        )
        stream = json.loads(out.stdout)["streams"][0]
        return int(stream["width"]), int(stream["height"])
    finally:
        Path(path).unlink(missing_ok=True)


def trim_and_encode(video_bytes, start, stop, height, width, crf, preset, gop):
    """Cut [start, stop) and re-encode. Returns (mp4 bytes, frame count)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        src = Path(tmp.name)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        vf = f"select='between(n\\,{start}\\,{stop - 1})',setpts=N/FRAME_RATE/TB"
        if height and width:
            vf += (
                f",scale={width}:{height}:force_original_aspect_ratio=decrease"
                f":force_divisible_by=2:flags=lanczos"
                f",pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
            )
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-threads",
                "2",
                "-i",
                str(src),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-g",
                str(gop),
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(out),
            ],
        )
        data = out.read_bytes()
        probe = run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                str(out),
            ],
            text=True,
        )
        return data, int(probe.stdout.strip() or 0)
    finally:
        src.unlink(missing_ok=True)
        out.unlink(missing_ok=True)


def pack_source(source, entries, analysis, args, out_path):
    fps = analysis["fps"]
    cameras = analysis["cameras"]
    episodes = analysis["episodes"]

    if args.cameras:
        approved = {camera.strip() for camera in args.cameras.split(",") if camera.strip()}
        fixed = set(cameras) & approved
    else:
        fixed = {c for c, m in cameras.items() if not m["moving"]}
    dims = {}
    for cam in fixed:
        if cameras[cam].get("width") and cameras[cam].get("height"):
            dims[cam] = (cameras[cam]["width"], cameras[cam]["height"])
    if args.aspect:
        for tar_path, files, meta in entries:
            cam = meta["camera"]["orig_name"]
            if cam in fixed and cam not in dims:
                dims[cam] = probe_dims(read_member(tar_path, files["video.mp4"]))
        off_aspect = {c for c in fixed if not dims[c][1] or abs(dims[c][0] / dims[c][1] - args.aspect) > 0.02}
        fixed -= off_aspect
    else:
        off_aspect = set()

    repo = source.replace("__", "/", 1) if "__" in source else source
    writer = ArrayRecordWriter(str(out_path), "group_size:1")
    written = skipped = frames_kept = failures = 0
    for tar_path, files, meta in entries:
        cam = meta["camera"]["orig_name"]
        qc = episodes.get(str(meta["episode"]), {})
        if cam not in fixed or not qc.get("keep") or "start" not in qc:
            continue
        # only SOAR carries `success`; absent means the dataset does not label outcomes
        if args.require_success and meta.get("success") is False:
            failures += 1
            continue
        start, stop = qc["start"], qc["end"] + 1
        try:
            with np.load(io.BytesIO(read_member(tar_path, files["frames.npz"]))) as d:
                actions = np.asarray(d["actions"], dtype=np.float32)[start:stop]
                state = np.asarray(d["state"], dtype=np.float32)[start:stop]
            video, n = trim_and_encode(
                read_member(tar_path, files["video.mp4"]),
                start,
                stop,
                args.height,
                args.width,
                args.crf,
                args.preset,
                args.gop,
            )
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if n != len(actions):  # pixels and actions must stay aligned
            skipped += 1
            continue
        buf = io.BytesIO()
        np.savez(
            buf,
            video=np.frombuffer(video, dtype=np.uint8),
            length=np.int32(n),
            fps=np.float32(fps),
            actions=actions,
            state=state,
            repo=repo,
            episode=np.int32(meta["episode"]),
            camera=cam,
            dataset=str(meta.get("dataset", "")),
            task=str(meta.get("language", "")),
            success=np.int8(meta.get("success", -1)),
            success_class=str(meta.get("success_class", "")),
            policy_repo_id=str(meta.get("policy_repo_id", "")),
            policy_type=str(meta.get("policy_type", "")),
        )
        writer.write(buf.getvalue())
        written += 1
        frames_kept += n
    writer.close()

    return {
        "source": source,
        "fps": fps,
        "cameras_fixed": sorted(fixed),
        "cameras_moving": sorted(c for c, m in cameras.items() if m["moving"]),
        "cameras_off_aspect": sorted(off_aspect),
        "records": written,
        "skipped": skipped,
        "failures_dropped": failures,
        "frames": frames_kept,
        "bytes": out_path.stat().st_size if written else 0,
    }


def stage_source(processed, dataset, source, workdir):
    if not is_gcs(processed):
        return Path(processed) / dataset / source, False
    local = Path(workdir) / "raw" / source
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir(parents=True)
    gcs_rsync(f"{processed}/{dataset}/{source}", local)
    return local, True


def load_analysis(analysis_prefix, dataset, source, workdir):
    uri = f"{analysis_prefix}/{dataset}/{source}.json"
    if is_gcs(analysis_prefix):
        if not gcs_exists(uri):
            return None
        local = Path(workdir) / "_analysis.json"
        gcs_cp(uri, local)
        return json.loads(local.read_text())
    return json.loads(Path(uri).read_text()) if Path(uri).exists() else None


def save_analysis(analysis, analysis_prefix, dataset, source, workdir):
    uri = f"{analysis_prefix}/{dataset}/{source}.json"
    if is_gcs(analysis_prefix):
        local = Path(workdir) / "_analysis.json"
        local.write_text(json.dumps(analysis, indent=1))
        gcs_cp(local, uri)
    else:
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        Path(uri).write_text(json.dumps(analysis, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="gs://visionary-uc1/so101/data/processed/v1")
    parser.add_argument("--dataset", default="so101")
    parser.add_argument("--out", default="gs://visionary-uc1/so101/data/tokenizer/packed")
    parser.add_argument("--analysis", default="gs://visionary-uc1/so101/data/tokenizer/analysis")
    parser.add_argument("--workdir", default="/mnt/work")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--checklist", default="artifacts/repo_review/checklist.csv")
    parser.add_argument("--phase", choices=["analyze", "pack"], default="pack")
    parser.add_argument("--reanalyze", action="store_true")
    parser.add_argument("--probe_episodes", type=int, default=3)
    parser.add_argument("--trim", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--gop", type=int, default=25)
    parser.add_argument(
        "--fps",
        type=float,
        default=0,
        help="only pack sources at this rate; 0 packs every rate",
    )
    parser.add_argument(
        "--aspect",
        type=float,
        default=0,
        help="only pack streams at this aspect; 0 disables (fit-and-pad handles all)",
    )
    parser.add_argument(
        "--require_success",
        action="store_true",
        help="drop entries flagged success=false (SOAR only)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cameras",
        help="Comma-separated fixed camera allow-list from visual review. Disables the camera-motion verdict.",
    )
    args = parser.parse_args()

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    drop = load_droplist(args.checklist)
    sources = list_sources(args.processed, args.dataset)
    todo = [s for i, s in enumerate(sources) if i % args.num_shards == args.shard]
    if args.limit:
        todo = todo[: args.limit]
    print(f"shard {args.shard}/{args.num_shards}: {len(todo)} sources", flush=True)

    report = []
    for position, source in enumerate(todo, 1):
        repo = source.replace("__", "/", 1) if "__" in source else source
        if repo in drop:
            print(f"[{position}/{len(todo)}] skip(drop) {source}", flush=True)
            continue
        out_uri = f"{args.out}/{args.dataset}/{source}.arecord"
        if args.phase == "pack" and not args.overwrite:
            done = gcs_exists(out_uri) if is_gcs(args.out) else Path(out_uri).exists()
            if done:
                print(f"[{position}/{len(todo)}] skip(done) {source}", flush=True)
                continue

        local, staged = stage_source(args.processed, args.dataset, source, args.workdir)
        try:
            entries = load_entries(local)
            if not entries:
                print(f"[{position}/{len(todo)}] EMPTY {source}", flush=True)
                continue
            fps = float(entries[0][2]["fps"])
            if args.fps and fps != args.fps:
                report.append({"source": source, "fps": fps, "skipped_source": "off_rate"})
                print(f"[{position}/{len(todo)}] skip(fps={fps}) {source}", flush=True)
                continue

            trim = args.trim == "on" or (args.trim == "auto" and args.dataset == "so101")
            wanted = {**ANALYSIS_PARAMS, "probe_episodes": args.probe_episodes, "trim": trim}
            analysis = None
            if not args.reanalyze:
                analysis = load_analysis(args.analysis, args.dataset, source, args.workdir)
                if analysis is not None and analysis.get("params") != wanted:
                    analysis = None
            if analysis is None:
                analysis = analyze_source(source, entries, fps, args.probe_episodes, trim)
                save_analysis(analysis, args.analysis, args.dataset, source, args.workdir)
            if args.phase == "analyze":
                n_moving = sum(m["moving"] for m in analysis["cameras"].values())
                n_keep = sum(e.get("keep", False) for e in analysis["episodes"].values())
                print(
                    f"[{position}/{len(todo)}] analyzed {source}: "
                    f"{len(analysis['cameras'])} cams ({n_moving} moving), "
                    f"{n_keep}/{len(analysis['episodes'])} episodes keep",
                    flush=True,
                )
                continue

            out_local = Path(args.workdir) / "out" / f"{source}.arecord"
            out_local.parent.mkdir(parents=True, exist_ok=True)
            out_local.unlink(missing_ok=True)
            row = pack_source(source, entries, analysis, args, out_local)
            report.append(row)
            if row["records"]:
                if is_gcs(args.out):
                    gcs_cp(out_local, out_uri)
                else:
                    Path(out_uri).parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(out_local, out_uri)
            out_local.unlink(missing_ok=True)
            print(
                f"[{position}/{len(todo)}] done {source}: {row['records']} records "
                f"({row['skipped']} skipped, {row['bytes'] / GB:.2f} GB)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{position}/{len(todo)}] FAILED {source}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if staged:
                shutil.rmtree(local, ignore_errors=True)

    report_path = Path(args.workdir) / f"pack_report_{args.dataset}_{args.shard}.json"
    report_path.write_text(json.dumps(report, indent=1))
    total = sum(r.get("records", 0) for r in report)
    print(
        f"PACK_COMPLETE dataset={args.dataset} shard={args.shard} sources={len(report)} records={total}",
        flush=True,
    )


if __name__ == "__main__":
    main()
