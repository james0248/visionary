import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import numpy as np
from einops import rearrange
from hydra.utils import instantiate

from visionary.common.checkpoint import (
    resolve_model_export_step,
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.tokenizer import Tokenizer
from visionary.tokenizer_preprocessor import TokenizerPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a browser demo context artifact from an episode and tokenizer checkpoint."
    )
    parser.add_argument("--episode", type=Path, default=Path("episode_0.npz"))
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--tokenizer_step", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("webgpu_app/assets"))
    parser.add_argument("--name", default="breakout_demo_context")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument("--prefix_frames", type=int, default=4)
    parser.add_argument("--context_step_level", type=int, default=5)
    parser.add_argument("--context_signal_level", type=int, default=29)
    parser.add_argument("--num_obs_tokens", type=int, default=32)
    parser.add_argument("--num_actions", type=int, default=4)
    parser.add_argument(
        "--action_meanings",
        default=None,
        help="Comma-separated action meanings in action-id order. Defaults to Breakout's reduced action set.",
    )
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument(
        "--align_actions_to_frames",
        action="store_true",
        help="Store frame-aligned actions, matching dynamics training: [prev_action, actions[:-1]].",
    )
    parser.add_argument(
        "--clean_context",
        action="store_true",
        help="Store clean encoded prefix latents instead of training-style noised context latents.",
    )
    return parser.parse_args()


def parse_action_meanings(value: str | None, num_actions: int) -> dict[str, str]:
    if value is None:
        meanings = ["noop", "fire", "right", "left"]
    else:
        meanings = [item.strip() for item in value.split(",")]
    if len(meanings) != num_actions:
        raise ValueError(
            f"Expected {num_actions} action meanings, got {len(meanings)}: {meanings}"
        )
    return {str(index): meaning for index, meaning in enumerate(meanings)}


def align_prefix_actions(actions: np.ndarray, start: int, prefix_frames: int) -> np.ndarray:
    cropped_actions = actions[start : start + prefix_frames]
    prev_action = actions[start - 1] if start > 0 else np.asarray(0, dtype=actions.dtype)
    aligned_actions = np.empty_like(cropped_actions)
    aligned_actions[0] = prev_action
    aligned_actions[1:] = cropped_actions[:-1]
    return aligned_actions


def write_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(array)
    path.write_bytes(array.tobytes())
    return {
        "path": path.name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes": int(array.nbytes),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    action_meanings = parse_action_meanings(args.action_meanings, args.num_actions)

    tokenizer_step = resolve_model_export_step(args.tokenizer_dir, args.tokenizer_step)
    tokenizer_cfg, tokenizer_variables = restore_model_export_single_device(
        args.tokenizer_dir,
        step=tokenizer_step,
    )
    preprocessor_cfg = restore_preprocessor_export(args.tokenizer_dir, step=tokenizer_step)
    preprocessor = TokenizerPreprocessor.from_config(preprocessor_cfg)
    tokenizer = instantiate(tokenizer_cfg)

    data = np.load(args.episode)
    frames = np.asarray(data["frames"])
    actions = np.asarray(data["actions"], dtype=np.int32)
    if args.prefix_frames <= 0 or args.prefix_frames > args.context_length:
        raise ValueError(
            f"--prefix_frames must be in [1, context_length], got {args.prefix_frames}."
        )
    if args.start < 0 or args.start + args.prefix_frames > len(frames):
        raise ValueError(
            f"Prefix window [{args.start}, {args.start + args.prefix_frames}) is outside "
            f"episode length {len(frames)}."
        )

    prefix_frames = frames[args.start : args.start + args.prefix_frames]
    raw_prefix_actions = actions[args.start : args.start + args.prefix_frames]
    prefix_actions = (
        align_prefix_actions(actions, args.start, args.prefix_frames)
        if args.align_actions_to_frames
        else raw_prefix_actions
    )
    patches = preprocessor.preprocess_video(prefix_frames)[None]
    display_pixels = np.asarray(preprocessor.patches_to_images(patches), dtype=np.uint8)[0]

    @jax.jit
    def encode_step(variables, patch_batch):
        return tokenizer.apply(variables, {"video": patch_batch}, method=Tokenizer.encode)

    latents = np.asarray(jax.device_get(encode_step(tokenizer_variables, patches)), dtype=np.float32)
    prefix_z = rearrange(latents, "b t (n k) d -> b t n (k d)", n=args.num_obs_tokens).astype(
        np.float32,
        copy=False,
    )
    context_step_count = 1 << int(args.context_step_level)
    context_tau = np.float32(int(args.context_signal_level) / context_step_count)
    if args.clean_context:
        noised_prefix_z = prefix_z
        context_noise = None
    else:
        rng = np.random.default_rng(args.noise_seed)
        context_noise = rng.standard_normal(prefix_z.shape, dtype=np.float32)
        noised_prefix_z = context_tau * prefix_z + (np.float32(1.0) - context_tau) * context_noise

    prefix_offset = args.context_length - args.prefix_frames
    z = np.zeros(
        (1, args.context_length, args.num_obs_tokens, prefix_z.shape[-1]),
        dtype=np.float32,
    )
    z[:, prefix_offset:] = noised_prefix_z
    display_z = prefix_z.astype(np.float32, copy=False)
    context_actions = np.zeros((1, args.context_length), dtype=np.int32)
    context_actions[:, prefix_offset:] = prefix_actions[None]
    step_levels = np.full(
        (1, args.context_length),
        int(args.context_step_level),
        dtype=np.int32,
    )
    signal_levels = np.full(
        (1, args.context_length),
        int(args.context_signal_level),
        dtype=np.int32,
    )

    prefix = args.out_dir / args.name
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_episode": args.episode.as_posix(),
        "episode_start": int(args.start),
        "context_length": int(args.context_length),
        "prefix_frames": int(args.prefix_frames),
        "zero_pad_frames": int(prefix_offset),
        "prefix_slot_start": int(prefix_offset),
        "context_tau": float(context_tau),
        "context_step_level": int(args.context_step_level),
        "context_signal_level": int(args.context_signal_level),
        "clean_context": bool(args.clean_context),
        "action_alignment": "frame_aligned" if args.align_actions_to_frames else "raw",
        "raw_prefix_actions": raw_prefix_actions.astype(np.int32).tolist(),
        "stored_prefix_actions": prefix_actions.astype(np.int32).tolist(),
        "noise_seed": int(args.noise_seed),
        "tokenizer_dir": str(args.tokenizer_dir),
        "tokenizer_step": int(tokenizer_step),
        "preprocessor": {
            **preprocessor.export_config(),
            "image_height": int(preprocessor.image_height),
            "image_width": int(preprocessor.image_width),
            "x_len": int(preprocessor.x_len),
            "y_len": int(preprocessor.y_len),
            "patch_dim": int(preprocessor.patch_dim),
            "num_channels": int(preprocessor.num_channels),
        },
        "arrays": {
            "z": write_array(prefix.with_suffix(".z.f32.bin"), z),
            "display_z": write_array(prefix.with_suffix(".display_z.f32.bin"), display_z),
            "display_pixels": write_array(
                prefix.with_suffix(".display_pixels.u8.bin"), display_pixels
            ),
            "actions": write_array(prefix.with_suffix(".actions.i32.bin"), context_actions),
            "step_levels": write_array(prefix.with_suffix(".step_levels.i32.bin"), step_levels),
            "signal_levels": write_array(prefix.with_suffix(".signal_levels.i32.bin"), signal_levels),
        },
        "episode_actions": {
            "path": "actions",
            "num_actions": int(args.num_actions),
            "meanings": action_meanings,
        },
        "notes": [
            "The tokenizer encoder is intentionally offline-only for the web demo.",
            "The artifact pads the first context slots with zeros and places the real prefix at the end.",
            "Unless clean_context is true, stored prefix latents are noised using the same tau convention as dynamics rollout.",
            "display_z stores the clean prefix latents for the initial browser preview only.",
            "display_pixels stores the raw preprocessed prefix frames for the initial browser preview.",
            "Dynamics context z is packed from tokenizer latents with shape [1,64,32,32].",
        ],
    }
    manifest_path = prefix.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
