"""Convert a LeRobot SO-100/101 dataset into per-episode NPZ (frames, actions,
state, rewards) for the tokenizer/dynamics pipeline. One primary camera per
episode; actions kept raw (normalized later by compute_action_stats).

    uv run python scripts/so101/lerobot_to_npz.py --manifest artifacts/so101/tier2_clean.json \
        --out-dir data/so101/raw --resize 128,160 --fps-target 30
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from visionary.dataset import decode_video_window


def pick_camera(repo: str, cam_keys: list[str]) -> str | None:
    """Prefer the view analyze.py verified as fixed; fall back to the first camera."""
    views = Path("artifacts/so101/camera_views.json")
    if views.exists():
        chosen = json.loads(views.read_text()).get(repo, {}).get("chosen")
        if chosen:
            return f"observation.images.{chosen}"
    return cam_keys[0] if cam_keys else None


def get_info(repo: str) -> dict:
    p = hf_hub_download(repo, "meta/info.json", repo_type="dataset")
    return json.load(open(p))


def resample_indices(n: int, src_fps: float, tgt_fps: float) -> np.ndarray:
    if not tgt_fps or abs(src_fps - tgt_fps) < 1e-6 or tgt_fps >= src_fps:
        return np.arange(n)
    step = src_fps / tgt_fps
    return np.unique(np.floor(np.arange(0, n, step)).astype(int))


def convert_repo(repo: str, out_root: Path, camera: str | None, max_eps: int | None,
                 fps_target: float | None, resize_hw: tuple[int, int] | None,
                 camera_policy: str = "external") -> dict:
    info = get_info(repo)
    fps = float(info.get("fps", 30))
    feats = info["features"]
    cam_keys = [k for k in feats if k.startswith("observation.images")]
    if camera and f"observation.images.{camera}" in cam_keys:
        cam_key = f"observation.images.{camera}"
        wrist_fb = False
    else:
        cam_key, wrist_fb = pick_camera(repo, cam_keys), False
    if cam_key is None:
        print(f"  {repo}: no {camera_policy} camera -> skipped whole dataset")
        return {"repo": repo, "camera": None, "episodes_written": 0,
                "episodes_skipped": 0, "frames_written": 0, "skipped_dataset": True}
    chunks_size = int(info.get("chunks_size", 1000))
    data_tpl = info["data_path"]
    video_tpl = info["video_path"]
    n_eps = int(info["total_episodes"])
    if max_eps:
        n_eps = min(n_eps, max_eps)

    out_dir = out_root / repo.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    written, frames_total, skipped = 0, 0, 0

    for ep in range(n_eps):
        out_npz = out_dir / f"episode_{ep:06d}.npz"
        if out_npz.exists():  # resume: skip already-converted episodes
            written += 1
            continue
        chunk = ep // chunks_size
        dpath = data_tpl.format(episode_chunk=chunk, episode_index=ep)
        vpath = video_tpl.format(episode_chunk=chunk, video_key=cam_key, episode_index=ep)
        try:
            dfile = hf_hub_download(repo, dpath, repo_type="dataset")
            vfile = hf_hub_download(repo, vpath, repo_type="dataset")
            table = pq.read_table(dfile)
            cols = table.column_names
            action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
            state = (np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
                     if "observation.state" in cols else np.zeros_like(action))
            frames = decode_video_window(vfile, 0, len(action), decode_hw=resize_hw)
        except Exception as e:  # noqa: BLE001
            print(f"    ! ep{ep} skipped: {type(e).__name__}: {str(e)[:120]}")
            skipped += 1
            continue

        # Align lengths (video decode can be off by <=1 frame vs the parquet).
        T = min(len(frames), len(action), len(state))
        if T < 2:
            skipped += 1
            continue
        frames, action, state = frames[:T], action[:T], state[:T]

        idx = resample_indices(T, fps, fps_target) if fps_target else np.arange(T)
        frames, action, state = frames[idx], action[idx], state[idx]

        np.savez_compressed(
            out_dir / f"episode_{ep:06d}.npz",
            frames=frames.astype(np.uint8),
            actions=action.astype(np.float32),
            state=state.astype(np.float32),
            rewards=np.zeros(len(frames), dtype=np.float32),
        )
        written += 1
        frames_total += len(frames)

    meta = {
        "repo": repo, "camera": cam_key, "wrist_fallback": wrist_fb,
        "src_fps": fps, "eff_fps": (fps_target or fps),
        "episodes_written": written, "episodes_skipped": skipped,
        "frames_written": frames_total,
    }
    json.dump(meta, open(out_dir / "_source.json", "w"), indent=1)
    print(f"  {repo}: cam={cam_key.replace('observation.images.','')} "
          f"wrote {written} eps / {frames_total} frames (skipped {skipped})")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--manifest")
    ap.add_argument("--out-dir", default="data/so101/raw")
    ap.add_argument("--camera", default=None, help="force a camera key suffix; else auto")
    ap.add_argument("--camera-policy", choices=("external", "top"), default="external",
                    help="auto-pick best external view, or ONLY a top-down view "
                         "(datasets without one are skipped).")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--fps-target", type=float, default=None)
    ap.add_argument("--resize", default=None, help="H,W to shrink stored frames, e.g. 256,320")
    args = ap.parse_args()

    resize_hw = tuple(int(x) for x in args.resize.split(",")) if args.resize else None
    repos = [args.repo] if args.repo else [
        (x.split(":", 1)[1] if ":" in x else x) for x in json.load(open(args.manifest))
    ]
    out_root = Path(args.out_dir)
    all_meta = []
    for i, repo in enumerate(repos):
        print(f"[{i + 1}/{len(repos)}] {repo}")
        try:
            all_meta.append(convert_repo(repo, out_root, args.camera, args.max_episodes,
                                         args.fps_target, resize_hw, args.camera_policy))
        except Exception as e:  # noqa: BLE001
            print(f"  !! repo failed: {type(e).__name__}: {str(e)[:160]}")
    out_root.mkdir(parents=True, exist_ok=True)
    json.dump(all_meta, open(out_root / "_ingest_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
