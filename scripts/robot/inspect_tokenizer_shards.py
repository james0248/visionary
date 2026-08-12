"""Validate packed ArrayRecord shards before training or upload.

Checks every record for the failures that would otherwise surface as bad
training rather than as an error:

  * the record parses and carries every expected field
  * the video decodes, and its frame count matches the declared length
  * actions and state have that same length -- a mismatch means pixels and
    actions are misaligned, which nothing downstream would notice
  * actions and state are finite

Optionally renders sample clips so the encode quality can be judged by eye.

    uv run python scripts/robot/inspect_tokenizer_shards.py --shards data/robot/tokenizer
    uv run python scripts/robot/inspect_tokenizer_shards.py --samples 12
"""

import argparse
import glob
import io
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

GB = 1024**3


def count_frames(video: bytes) -> int:
    """Decode without converting to arrays -- only the frame count is needed."""
    import av

    container = av.open(io.BytesIO(video))
    try:
        return sum(1 for _ in container.decode(video=0))
    finally:
        container.close()


def check_record(args) -> dict:
    shard, index, action_dim = args
    import grain.python as grain

    src = grain.ArrayRecordDataSource([shard])
    try:
        with np.load(io.BytesIO(src[index])) as data:
            missing = [k for k in ("video", "length", "actions", "state", "repo", "episode", "camera") if k not in data]
            if missing:
                return {"ok": False, "error": f"missing fields: {missing}"}
            video = bytes(data["video"])
            length = int(data["length"])
            actions = np.asarray(data["actions"])
            state = np.asarray(data["state"])
            repo, camera = str(data["repo"]), str(data["camera"])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"unreadable record: {type(e).__name__}"}

    if len(actions) != length or len(state) != length:
        return {
            "ok": False,
            "repo": repo,
            "error": f"length {length} but actions {len(actions)} state {len(state)}",
        }
    if not (np.isfinite(actions).all() and np.isfinite(state).all()):
        return {"ok": False, "repo": repo, "error": "non-finite actions or state"}
    if actions.ndim != 2 or (action_dim and actions.shape[1] != action_dim):
        return {"ok": False, "repo": repo, "error": f"action shape {actions.shape}"}

    try:
        decoded = count_frames(video)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "repo": repo, "error": f"video will not decode: {type(e).__name__}"}
    if decoded != length:
        return {
            "ok": False,
            "repo": repo,
            "error": f"video has {decoded} frames, record declares {length}",
        }

    return {
        "ok": True,
        "repo": repo,
        "camera": camera,
        "frames": length,
        "bytes": len(video),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
    }


def render_samples(shards: list[str], out: Path, n: int) -> None:
    import cv2
    import grain.python as grain
    import imageio.v2 as imageio

    from visionary.dataset import decode_video_window

    src = grain.ArrayRecordDataSource(shards)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(src), size=min(n, len(src)), replace=False)
    out.mkdir(parents=True, exist_ok=True)
    for i in picks:
        with np.load(io.BytesIO(src[int(i)])) as data:
            video, repo = bytes(data["video"]), str(data["repo"])
            episode, camera = int(data["episode"]), str(data["camera"])
        frames = decode_video_window(video, 0, 10**9)
        labelled = []
        for k, f in enumerate(frames):
            f = f.copy()
            cv2.rectangle(f, (0, 0), (f.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(
                f,
                f"{repo} ep{episode} {camera}",
                (4, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                f,
                f"frame {k}/{len(frames)}",
                (4, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (120, 220, 255),
                1,
            )
            labelled.append(f)
        arr = np.stack(labelled)
        arr = arr[:, : arr.shape[1] // 2 * 2, : arr.shape[2] // 2 * 2]
        imageio.mimwrite(
            out / f"{repo.replace('/', '__')}_ep{episode:03d}_{camera}.mp4",
            arr,
            fps=30,
            codec="libx264",
            quality=6,
            macro_block_size=None,
        )
    print(f"  rendered {len(picks)} sample clips -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/robot/tokenizer")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="check only the first N records")
    ap.add_argument("--samples", type=int, default=0, help="also render N clips to outputs/")
    ap.add_argument("--out", default="outputs/shard_samples")
    ap.add_argument("--report", default="artifacts/robot/shard_report.json")
    ap.add_argument("--action-dim", type=int, default=6, help="expected action dim; 0 disables")
    args = ap.parse_args()

    import grain.python as grain

    shards = sorted(glob.glob(f"{args.shards}/**/*.arecord", recursive=True))
    shards = [s for s in shards if Path(s).stat().st_size > 65536]
    if not shards:
        raise SystemExit(f"no usable shards under {args.shards}")
    size = sum(Path(s).stat().st_size for s in shards) / GB
    print(f"{len(shards)} shards, {size:.1f} GB")

    work, unreadable = [], []
    for shard in shards:
        try:
            n = len(grain.ArrayRecordDataSource([shard]))
        except Exception as e:  # noqa: BLE001
            # a shard still being written, or genuinely corrupt; report, do not crash
            unreadable.append((shard, f"{type(e).__name__}: {str(e)[:90]}"))
            continue
        work += [(shard, i, args.action_dim) for i in range(n)]
    if unreadable:
        print(f"  {len(unreadable)} shard(s) could not be opened:")
        for shard, err in unreadable[:5]:
            print(f"    {Path(shard).name}: {err}")
    if args.limit:
        work = work[: args.limit]
    print(f"{len(work):,} records to check")

    results, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(check_record, work, chunksize=8):
            results.append(r)
            done += 1
            if done % 2000 == 0:
                bad = sum(not x["ok"] for x in results)
                print(f"  {done:,}/{len(work):,}  ({bad} bad)", flush=True)

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    frames = sum(r["frames"] for r in ok)
    nbytes = sum(r["bytes"] for r in ok)
    repos = {r["repo"] for r in ok}
    amin = min((r["action_min"] for r in ok), default=0.0)
    amax = max((r["action_max"] for r in ok), default=0.0)

    print(f"\n{len(ok):,} valid, {len(bad)} invalid, {len(unreadable)} unreadable shards")
    print(f"  datasets    : {len(repos)}")
    print(f"  frames      : {frames:,}  ({frames / 30 / 3600:.1f} h at 30 fps)")
    print(f"  video bytes : {nbytes / GB:.1f} GB  ({nbytes / max(frames, 1):.0f} bytes/frame)")
    print(f"  action range: [{amin:.1f}, {amax:.1f}] (degrees, un-normalized)")
    if bad:
        print("\n  failures:")
        for kind, n in Counter(r["error"].split(":")[0] for r in bad).most_common():
            print(f"    {n:5d}  {kind}")
        for r in bad[:5]:
            print(f"      e.g. {r.get('repo', '?')}: {r['error']}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "shards": len(shards),
            "unreadable_shards": [s for s, _ in unreadable],
            "records": len(results),
            "valid": len(ok),
            "invalid": len(bad),
            "datasets": len(repos),
            "frames": frames,
            "bytes": nbytes,
            "failures": bad[:200],
        },
        open(args.report, "w"),
        indent=1,
    )
    print(f"\n-> {args.report}")

    if args.samples:
        render_samples(shards, Path(args.out), args.samples)
    raise SystemExit(1 if (bad or unreadable) else 0)


if __name__ == "__main__":
    main()
