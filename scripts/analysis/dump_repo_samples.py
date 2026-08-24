"""Dump raw mp4 samples from the packed video shards, grouped by repo.

Writes <out>/<repo_slug>/ep<episode>_<camera>.mp4 for the requested repos so
the footage each repo contributes can be eyeballed directly.

    uv run python scripts/analysis/dump_repo_samples.py \
        --shards_dir data/so101/raw/eval --out artifacts/repo_probe/repo_samples \
        --repos VoicAndrei/so100_kitchen,Loki0929/so100_lan --per_repo 2
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import grain.python as grain
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repos", help="Comma-separated repo names. Omit to export every repo.")
    ap.add_argument("--per_repo", type=int, default=2)
    args = ap.parse_args()

    wanted = set(args.repos.split(",")) if args.repos else None
    counts: dict[str, int] = {}
    manifest = []
    out = Path(args.out)
    source = grain.ArrayRecordDataSource(sorted(str(p) for p in Path(args.shards_dir).glob("*.arecord")))
    for i in range(len(source)):
        with np.load(io.BytesIO(source[i])) as data:
            repo = str(data["repo"])
            if wanted is not None and repo not in wanted:
                continue
            if args.per_repo > 0 and counts.get(repo, 0) >= args.per_repo:
                continue
            episode, camera = int(data["episode"]), str(data["camera"])
            length, fps = int(data["length"]), float(data["fps"])
            video = data["video"].tobytes()
        slug = repo.replace("/", "__")
        (out / slug).mkdir(parents=True, exist_ok=True)
        path = out / slug / f"ep{episode:06d}_{camera.replace('.', '_')}.mp4"
        if path.exists():
            raise FileExistsError(path)
        path.write_bytes(video)
        counts[repo] = counts.get(repo, 0) + 1
        manifest.append(
            {
                "path": str(path.relative_to(out)),
                "bytes": len(video),
                "sha256": hashlib.sha256(video).hexdigest(),
                "repo": repo,
                "episode": episode,
                "camera": camera,
                "frames": length,
                "fps": fps,
            }
        )
        if len(manifest) % 500 == 0:
            print(f"wrote {len(manifest):,} videos", flush=True)
    missing = (wanted or set()) - set(counts)
    if missing:
        print(f"no records found for: {sorted(missing)}")
    summary = {
        "format": "pre-tokenization trajectory MP4",
        "records": len(manifest),
        "total_bytes": sum(row["bytes"] for row in manifest),
        "repos": counts,
        "files": manifest,
    }
    (out / "upload_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"complete: {len(manifest):,} videos, {summary['total_bytes'] / 1024**3:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
