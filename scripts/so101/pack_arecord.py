"""Pack the downloaded LeRobot tree into ArrayRecord shards.

Each record is one (episode, fixed camera) stream:

    video   : encoded mp4 bytes  (trimmed to the non-stale span)
    length  : frame count
    actions : (T, 6) float32     (trimmed identically)
    state   : (T, 6) float32
    repo, episode, camera

Video and actions are cut with the SAME bounds and the frame counts are asserted
equal -- a mismatch would desync pixels from actions and poison training without
raising anywhere downstream.

    uv run python scripts/so101/pack_arecord.py --out-dir data/so101/shards --dry-run
"""

import argparse
import io
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from array_record.python.array_record_module import ArrayRecordWriter

GB = 1024**3


def trim_and_encode(src: Path, start: int, stop: int, height: int, width: int,
                    crf: int, preset: str) -> tuple[bytes, int]:
    """Cut [start, stop) and re-encode. Returns (mp4 bytes, frame count)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        vf = f"select='between(n\\,{start}\\,{stop - 1})',setpts=N/FRAME_RATE/TB"
        if height and width:
            vf += f",scale={width}:{height}"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-threads", "2", "-i", str(src),
             "-vf", vf, "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-an", str(out)],
            check=True, capture_output=True,
        )
        data = out.read_bytes()
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True,
        )
        return data, int(probe.stdout.strip() or 0)
    finally:
        out.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/so101/hf")
    ap.add_argument("--out-dir", default="data/so101/shards")
    ap.add_argument("--views", default="artifacts/so101/camera_views.json")
    ap.add_argument("--bounds", default="artifacts/so101/episode_bounds.json")
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--eval-ratio", type=float, default=0.02)
    ap.add_argument("--records-per-shard", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    views = json.loads(Path(args.views).read_text())
    bounds = json.loads(Path(args.bounds).read_text())
    root = Path(args.data_dir)

    jobs = []
    for repo, v in views.items():
        rb = bounds.get(repo)
        if not rb or not v["fixed"]:
            continue
        d = root / repo.replace("/", "__")
        for cam in v["fixed"]:
            for ep_str, (start, stop) in rb.items():
                ep = int(ep_str)
                mp4 = sorted(d.glob(f"videos/*/observation.images.{cam}/episode_{ep:06d}.mp4"))
                pqt = sorted(d.glob(f"data/*/episode_{ep:06d}.parquet"))
                if mp4 and pqt:
                    jobs.append((repo, cam, ep, mp4[0], pqt[0], start, stop))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs):,} (episode, camera) streams to pack")
    if args.dry_run:
        kept = sum(stop - start for *_, start, stop in jobs)
        print(f"  frames after trim: {kept:,}")
        return

    import hashlib

    def is_eval(repo, ep):
        h = hashlib.sha256(f"{args.seed}:{repo}:{ep}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64 < args.eval_ratio

    out = Path(args.out_dir)
    writers, counts, shards = {}, {}, {}
    for split in ("train", "eval"):
        (out / split).mkdir(parents=True, exist_ok=True)
        shards[split] = 0
        counts[split] = 0
        writers[split] = ArrayRecordWriter(
            str(out / split / f"shard-{0:05d}.arecord"), "group_size:1")

    done = skipped = 0
    for repo, cam, ep, mp4, pqt, start, stop in jobs:
        try:
            actions = np.asarray(pq.read_table(pqt).column("action").to_pylist(),
                                 dtype=np.float32)[start:stop]
            state = np.asarray(pq.read_table(pqt).column("observation.state").to_pylist(),
                               dtype=np.float32)[start:stop]
            video, n = trim_and_encode(mp4, start, stop, args.height, args.width,
                                       args.crf, args.preset)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if n != len(actions):          # pixels and actions must stay aligned
            skipped += 1
            continue

        buf = io.BytesIO()
        np.savez(buf, video=np.frombuffer(video, dtype=np.uint8), length=np.int32(n),
                 actions=actions, state=state, repo=repo, episode=np.int32(ep), camera=cam)
        split = "eval" if is_eval(repo, ep) else "train"
        writers[split].write(buf.getvalue())
        counts[split] += 1
        done += 1
        if counts[split] % args.records_per_shard == 0:
            writers[split].close()
            shards[split] += 1
            writers[split] = ArrayRecordWriter(
                str(out / split / f"shard-{shards[split]:05d}.arecord"), "group_size:1")
        if done % 500 == 0:
            print(f"  {done:,}/{len(jobs):,} packed ({skipped} skipped)", flush=True)

    for w in writers.values():
        w.close()
    size = sum(f.stat().st_size for f in out.rglob("*.arecord")) / GB
    print(f"\npacked {done:,} records ({skipped} skipped) -> {size:.1f} GB in {out}")


if __name__ == "__main__":
    main()
