"""Delete video for cameras that pixel analysis classified as MOVING (wrist /
egocentric). Those views are not used by the fixed-view world model, and
camera_views.json records every one so they can be re-downloaded on demand.

Only files under a verified-moving camera directory are touched -- parquet,
meta, and every fixed/borderline view are left alone. Borderline cameras count
as FIXED and are never deleted.

    uv run python scripts/so101/purge_moving_cameras.py --dry-run
    uv run python scripts/so101/purge_moving_cameras.py --yes
"""

import argparse
import json
from collections import Counter
from pathlib import Path

GB = 1024**3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="artifacts/so101/camera_views.json")
    ap.add_argument("--data-dir", default="data/so101/hf")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="actually delete")
    args = ap.parse_args()

    views = json.loads(Path(args.views).read_text())
    root = Path(args.data_dir)

    targets, by_name = [], Counter()
    total = 0
    for d in sorted(root.glob("*__*")):
        v = views.get(d.name.replace("__", "/", 1))
        if not v:
            continue                      # unclassified repo: leave untouched
        for cam in v["moving"]:
            if cam in v["fixed"] or cam in v.get("borderline", []):
                continue                  # belt and braces
            for f in (d / "videos").glob(f"*/observation.images.{cam}/*.mp4"):
                targets.append(f)
                total += f.stat().st_size
                by_name[cam] += 1

    print(f"{len(targets):,} files, {total / GB:.1f} GB across "
          f"{len({t.parents[3].name for t in targets})} datasets")
    print("top camera names:", dict(by_name.most_common(8)))

    if not args.yes or args.dry_run:
        print("\nDRY RUN — nothing deleted. Re-run with --yes to delete.")
        for f in targets[:5]:
            print("  would delete:", f)
        return

    freed = removed = 0
    for f in targets:
        try:
            n = f.stat().st_size
            f.unlink()
            freed += n
            removed += 1
        except OSError as e:
            print(f"  could not remove {f}: {e}")
    # drop the now-empty camera directories
    for d in sorted(root.glob("*__*/videos/*/observation.images.*"), reverse=True):
        try:
            next(d.iterdir())
        except StopIteration:
            d.rmdir()
        except OSError:
            pass
    print(f"\ndeleted {removed:,} files, freed {freed / GB:.1f} GB")


if __name__ == "__main__":
    main()
