import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
from array_record.python.array_record_data_source import ArrayRecordDataSource

from visionary.dataset import decode_video_window


def contact_sheet(video: bytes, n_frames: int, width: int) -> np.ndarray | None:
    frames = decode_video_window(video, 0, 900)
    if len(frames) < n_frames:
        return None
    picks = np.linspace(0, len(frames) - 1, n_frames).astype(int)
    height = int(width * frames.shape[1] / frames.shape[2])
    cells = []
    for i in picks:
        cell = cv2.resize(frames[i], (width, height), interpolation=cv2.INTER_AREA)
        cv2.putText(cell, f"f{i}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        cells.append(cell)
    return np.concatenate(cells, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", default="data/so101/raw/eval,data/so101/raw/top100_extra")
    ap.add_argument("--repo_hours", default="artifacts/repo_review/packed_repo_hours.json")
    ap.add_argument("--camera_choice", default="artifacts/so101/camera_choice.json")
    ap.add_argument("--out", default="artifacts/repo_review")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--cell_width", type=int, default=240)
    args = ap.parse_args()

    hours = json.load(open(args.repo_hours))
    cameras = json.load(open(args.camera_choice))
    ranked = sorted(hours.items(), key=lambda kv: -kv[1]["train_hours"])[: args.top]
    wanted = {repo: cameras.get(repo, "").split(".")[-1] for repo, _ in ranked}

    out = Path(args.out)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    paths = []
    for d in args.shards_dir.split(","):
        paths.extend(sorted(str(p) for p in Path(d).glob("*.arecord")))
    source = ArrayRecordDataSource(paths)
    picked: dict[str, bytes] = {}
    for i in range(len(source)):
        with np.load(io.BytesIO(source[i])) as data:
            repo, camera = str(data["repo"]), str(data["camera"])
            if repo in wanted and wanted[repo] == camera and repo not in picked:
                picked[repo] = data["video"].tobytes()

    rows = []
    for rank, (repo, stats) in enumerate(ranked, 1):
        slug = repo.replace("/", "__")
        if repo not in picked:
            rows.append((rank, repo, stats, wanted[repo], None))
            continue
        sheet = contact_sheet(picked[repo], args.frames, args.cell_width)
        if sheet is None:
            rows.append((rank, repo, stats, wanted[repo], None))
            continue
        path = out / "sheets" / f"{slug}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        rows.append((rank, repo, stats, wanted[repo], f"sheets/{slug}.jpg"))
        print(f"{rank:3d} {repo}")

    html = [
        "<meta charset='utf-8'><title>SO-101 repo review</title>",
        "<style>body{font-family:system-ui;background:#111;color:#eee;margin:0;padding:16px}"
        ".r{margin:0 0 22px;border-bottom:1px solid #333;padding-bottom:14px}"
        ".h{display:flex;gap:14px;align-items:baseline;margin-bottom:6px}"
        ".n{font-weight:600;font-size:15px}.m{color:#999;font-size:13px}"
        "img{width:100%;image-rendering:auto;border-radius:4px}"
        ".miss{color:#c66;font-size:13px}</style>",
        f"<h2>Top {len(rows)} SO-101 repos by train hours &mdash; training camera</h2>",
    ]
    for rank, repo, stats, camera, rel in rows:
        html.append("<div class='r'><div class='h'>")
        html.append(f"<span class='n'>{rank}. {repo}</span>")
        html.append(
            f"<span class='m'>{stats['train_hours']:.2f} h &middot; {stats.get('episodes', 0)} ep &middot; "
            f"{camera} &middot; cams: {','.join(stats.get('cameras', []))}</span>"
        )
        html.append("</div>")
        html.append(f"<img src='{rel}'>" if rel else "<div class='miss'>no eval sample locally</div>")
        html.append("</div>")
    (out / "index.html").write_text("\n".join(html))

    with open(out / "checklist.csv", "w") as fh:
        fh.write("rank,repo,train_hours,camera,keep,reason\n")
        for rank, repo, stats, camera, _ in rows:
            fh.write(f"{rank},{repo},{stats['train_hours']:.3f},{camera},,\n")
    print(f"\n{out}/index.html   ({sum(1 for r in rows if r[4])} sheets)")


if __name__ == "__main__":
    main()
