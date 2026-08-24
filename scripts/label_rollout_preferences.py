"""Label three-panel rollout videos with the numeric keypad."""

import argparse
import csv
import os
from pathlib import Path

import cv2


LABELS = {ord("1"): "so101_only", ord("2"): "tie", ord("3"): "cotrained_v4"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos",
        type=Path,
        default=Path("artifacts/eval_split/random100_collage_seed42/three_panel_rollouts"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--speed", type=float, default=10.0)
    args = parser.parse_args()

    output = args.output or args.videos.parent / "human_labels.csv"
    videos = sorted(args.videos.glob("*.mp4"))
    labels = {}
    if output.exists():
        with output.open(newline="") as handle:
            labels = {row["video"]: row["label"] for row in csv.DictReader(handle)}

    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("video", "label"))
        if write_header:
            writer.writeheader()

        try:
            for position, video in enumerate(videos, 1):
                if video.name in labels:
                    continue
                cap = cv2.VideoCapture(str(video))
                fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
                delay = max(1, round(1000 / (fps * args.speed)))
                title = f"{position}/{len(videos)} | 1 SO-101 | 2 tie | 3 cotrain | q quit"
                print(f"{title}\n{video.name}")

                label = None
                while label is None:
                    ok, frame = cap.read()
                    if not ok:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    cv2.imshow(title, frame)
                    key = cv2.waitKeyEx(delay) & 0xFF
                    if key in LABELS:
                        label = LABELS[key]
                    elif key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                        return

                writer.writerow({"video": video.name, "label": label})
                handle.flush()
                os.fsync(handle.fileno())
                cap.release()
                cv2.destroyWindow(title)
        finally:
            cv2.destroyAllWindows()

    print(f"Saved {len(videos)} labels to {output}")


if __name__ == "__main__":
    main()
