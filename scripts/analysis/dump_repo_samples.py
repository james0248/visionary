"""Dump raw mp4 samples from the packed video shards, grouped by repo.

Writes <out>/<repo_slug>/ep<episode>_<camera>.mp4 for the requested repos so
the footage each repo contributes can be eyeballed directly.

    uv run python scripts/analysis/dump_repo_samples.py \
        --shards_dir data/so101/raw/eval --out artifacts/repo_probe/repo_samples \
        --repos VoicAndrei/so100_kitchen,Loki0929/so100_lan --per_repo 2
"""

import argparse
import io
from pathlib import Path

import grain.python as grain
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repos", required=True, help="Comma-separated repo names.")
    ap.add_argument("--per_repo", type=int, default=2)
    args = ap.parse_args()

    wanted = set(args.repos.split(","))
    counts: dict[str, int] = {}
    out = Path(args.out)
    source = grain.ArrayRecordDataSource(
        sorted(str(p) for p in Path(args.shards_dir).glob("*.arecord"))
    )
    for i in range(len(source)):
        with np.load(io.BytesIO(source[i])) as data:
            repo = str(data["repo"])
            if repo not in wanted or counts.get(repo, 0) >= args.per_repo:
                continue
            episode, camera = int(data["episode"]), str(data["camera"])
            video = data["video"].tobytes()
        slug = repo.replace("/", "__")
        (out / slug).mkdir(parents=True, exist_ok=True)
        path = out / slug / f"ep{episode:04d}_{camera.replace('.', '_')}.mp4"
        path.write_bytes(video)
        counts[repo] = counts.get(repo, 0) + 1
        print(f"wrote {path}")
    missing = wanted - set(counts)
    if missing:
        print(f"no records found for: {sorted(missing)}")


if __name__ == "__main__":
    main()
