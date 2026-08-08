"""Compute global action-normalization stats (q01-q99, min/max/mean/std) over the
ingested NPZ actions. Writes norm_stats.json for save_dynamics_dataset --action_stats.

    uv run python scripts/so101/compute_action_stats.py --raw-dir data/so101/raw
"""

import argparse
import json
from pathlib import Path

import numpy as np

CANON_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="dir of per-repo NPZ subfolders")
    ap.add_argument("--out", default="data/so101/norm_stats.json")
    ap.add_argument("--max-rows-per-episode", type=int, default=None,
                    help="optional subsample cap per episode to bound memory")
    args = ap.parse_args()

    files = sorted(Path(args.raw_dir).rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz under {args.raw_dir}")

    chunks: list[np.ndarray] = []
    n_eps = 0
    for f in files:
        try:
            with np.load(f) as d:
                if "actions" not in d.files:
                    continue
                a = np.asarray(d["actions"], dtype=np.float64)
        except Exception:  # noqa: BLE001
            continue
        if a.ndim != 2 or a.shape[1] != 6:
            continue
        if args.max_rows_per_episode and len(a) > args.max_rows_per_episode:
            idx = np.linspace(0, len(a) - 1, args.max_rows_per_episode).astype(int)
            a = a[idx]
        chunks.append(a)
        n_eps += 1

    if not chunks:
        raise RuntimeError("No valid (T,6) action arrays found.")
    alla = np.concatenate(chunks, axis=0)  # (N, 6)
    stats = {
        "joint_names": CANON_JOINTS,
        "n_episodes": n_eps,
        "n_frames": int(alla.shape[0]),
        "units": "degrees",
        "normalization": "q01_q99_to_[-1,1]_clipped (gripper included)",
        "min": alla.min(0).round(4).tolist(),
        "max": alla.max(0).round(4).tolist(),
        "mean": alla.mean(0).round(4).tolist(),
        "std": alla.std(0).round(4).tolist(),
        "q01": np.quantile(alla, 0.01, axis=0).round(4).tolist(),
        "q99": np.quantile(alla, 0.99, axis=0).round(4).tolist(),
        "q10": np.quantile(alla, 0.10, axis=0).round(4).tolist(),
        "q90": np.quantile(alla, 0.90, axis=0).round(4).tolist(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(args.out, "w"), indent=1)
    print(json.dumps(stats, indent=1))
    print(f"\nWrote {args.out}  ({n_eps} episodes, {alla.shape[0]} frames)")


if __name__ == "__main__":
    main()
