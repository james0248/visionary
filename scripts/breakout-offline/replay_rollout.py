import argparse

import cv2
import numpy as np


def draw_hud(frame: np.ndarray, step: int, total: int, action: int,
             reward: float, cumulative_reward: float, fps: int, paused: bool) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (255, 255, 255)
    shadow = (0, 0, 0)
    thickness = 1

    lines = [
        f"Step: {step}/{total}",
        f"Action: {action}",
        f"Reward: {reward:+.1f}  Total: {cumulative_reward:.1f}",
        f"FPS: {fps}" + ("  [PAUSED]" if paused else ""),
    ]

    for j, line in enumerate(lines):
        y = 20 + j * 20
        cv2.putText(frame, line, (6, y + 1), font, scale, shadow, thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (5, y), font, scale, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Replay a saved rollout episode")
    parser.add_argument("path", help="Path to episode .npz file")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS (default: 30)")
    parser.add_argument("--scale", type=int, default=3, help="Display scale factor (default: 3)")
    args = parser.parse_args()

    data = np.load(args.path, mmap_mode="r")
    frames = data["frames"]
    actions = data["actions"]
    rewards = data["rewards"]

    total = len(frames)
    cum_rewards = np.cumsum(rewards)
    paused = False
    i = 0
    fps = args.fps

    print(f"Loaded {total} frames from {args.path}")
    print("Controls: SPACE=pause  LEFT/RIGHT=step  +/-=speed  Q=quit")

    last_i, last_fps, last_paused = -1, -1, None
    display = None

    while i < total:
        if i != last_i or fps != last_fps or paused != last_paused:
            frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_NEAREST)
            draw_hud(frame, i, total, actions[i], rewards[i], cum_rewards[i], fps, paused)
            display = frame
            last_i, last_fps, last_paused = i, fps, paused

        cv2.imshow("Rollout Replay", display)

        delay = 0 if paused else max(1, 1000 // fps)
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q") or key == 27:  # q or ESC
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("+") or key == ord("="):
            fps = min(240, fps + 5)
        elif key == ord("-") or key == ord("_"):
            fps = max(1, fps - 5)
        elif key == 2:  # left arrow
            i = max(0, i - 1)
            continue
        elif key == 3:  # right arrow
            if paused and i < total - 1:
                i += 1
            continue

        if not paused:
            i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
