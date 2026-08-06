"""Seed-to-seed spread of rollouts at grasp moments, plus motion attenuation.

Two failure modes look identical in a single video but need opposite fixes. A
model that collapses to the conditional mean shows spread << error and damped
motion; contact would need a less noisy target or a distribution-aware loss.
A model that samples diverse-but-wrong outcomes shows spread ~ error; that is
missing information (actions carry no force/slip signal) and needs better
conditioning, not a better objective.

Also measures the predicted/true latent speed ratio on moving tokens: the mean
prediction of a noisy target under-moves by exactly the noise fraction, so this
number is the smoking gun for target-noise shrinkage.

    uv run python scripts/eval_rollout_variance.py \
        --checkpoint_dir gs://.../so101_dynamics --data_dir data/so101/dyn/eval
"""

import argparse
import functools
import io
import json
import logging
from pathlib import Path

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visionary.dataset import align_actions_to_frames
from visionary.dynamics import DynamicsModel

from eval_dynamics_horizon import moving_token_mask
from eval_dynamics_videos import load_train_config, restore_params

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Rollout spread at grasp moments.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="rollout_variance.json")
    parser.add_argument("--step", type=int)
    parser.add_argument("--indices", default="2,7")
    parser.add_argument("--num_seeds", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--total_frames", type=int, default=192)
    parser.add_argument("--context_tau", type=float, default=0.9)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--grip_delta", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_train_config(args.checkpoint_dir)
    source = grain.ArrayRecordDataSource(
        sorted(str(p) for p in Path(args.data_dir).glob("*.arecord"))
    )

    def sample_for(index: int):
        with np.load(io.BytesIO(source[index % len(source)])) as data:
            video = np.asarray(data["frames"])
            actions = np.asarray(data["actions"], dtype=np.float32)
            prev_action = np.asarray(data["prev_action"], dtype=np.float32)
        total = min(args.total_frames, len(video))
        rng = np.random.default_rng([args.seed, index])
        offset = int(rng.integers(0, max(len(video) - total, 0) + 1))
        stop = offset + total
        before = actions[offset - 1] if offset > 0 else prev_action
        raw_actions = actions[offset:stop]
        return (
            np.asarray(video[offset:stop], dtype=np.float32)[None],
            align_actions_to_frames(raw_actions, prev_action=before)[None],
            raw_actions,
        )

    video, actions, _ = sample_for(int(args.indices.split(",")[0]))
    model, params, step = restore_params(
        cfg, args.checkpoint_dir, args.step, {"video": video, "actions": actions}
    )
    logger.info("Restored params from step %d", step)

    @functools.partial(jax.jit, static_argnames=("start", "steps"))
    def rollout(params, video, actions, seed, start, steps):
        video = jnp.asarray(video, dtype=jnp.float32)
        primed = jnp.zeros_like(video).at[:, :start].set(video[:, :start])
        context_key, sample_key = jax.random.split(jax.random.key(seed))
        return model.apply(
            params,
            primed,
            jnp.asarray(actions, dtype=jnp.float32),
            jax.random.normal(context_key, video.shape, dtype=jnp.float32),
            jax.random.normal(sample_key, (video.shape[0], steps, *video.shape[2:]), jnp.float32),
            start,
            context_tau=args.context_tau,
            sample_steps=args.sample_steps,
            method=DynamicsModel.generate_rollout,
        )

    results = []
    for index in (int(i) for i in args.indices.split(",")):
        video, actions, raw_actions = sample_for(index)
        truth = video[0]
        mask = moving_token_mask(truth, 0.1)
        grip = raw_actions[:, -1]
        events = np.where(np.abs(np.diff(grip)) > args.grip_delta)[0]
        events = events[(events >= 40) & (events + args.horizon // 2 < len(truth))]
        if len(events) == 0:
            logger.warning("clip %d: no usable gripper event, using start 64", index)
            events = np.array([64])
        start = int(events[0]) - 8

        # multi-seed rollout across the grasp
        outs = []
        for s in range(args.num_seeds):
            outs.append(
                np.asarray(
                    jax.device_get(
                        rollout(params, video, actions, 1000 * index + s, start, args.horizon)
                    )
                )[0][start : start + args.horizon]
            )
        outs = np.stack(outs)  # (S, H, N, D)
        ref = truth[start : start + args.horizon]
        m = mask[start : start + args.horizon]
        spread = np.linalg.norm(outs.std(0), axis=-1)  # (H, N)
        err = np.linalg.norm(outs - ref[None], axis=-1).mean(0)
        scale = np.linalg.norm(ref, axis=-1)
        # motion attenuation over the generated segment, moving tokens
        d_out = np.linalg.norm(np.diff(outs, axis=1), axis=-1).mean(0)
        d_ref = np.linalg.norm(np.diff(ref, axis=0), axis=-1)
        dm = m[1:]
        entry = {
            "index": index,
            "start": start,
            "grip_event": int(events[0]),
            "spread_over_err_moving": float(spread[m].mean() / err[m].mean()),
            "rel_err_moving": float(err[m].mean() / scale[m].mean()),
            "speed_ratio_moving": float(d_out[dm].mean() / d_ref[dm].mean()),
            "speed_ratio_static": float(d_out[~dm].mean() / d_ref[~dm].mean()),
        }
        results.append(entry)
        logger.info(
            "clip %d start %d | spread/err %.3f | rel err moving %.3f | "
            "speed ratio moving %.3f static %.3f",
            index, start, entry["spread_over_err_moving"], entry["rel_err_moving"],
            entry["speed_ratio_moving"], entry["speed_ratio_static"],
        )

    Path(args.output).write_text(json.dumps({"step": step, "results": results}, indent=2))
    print("\nspread/err << 1 with speed ratio << 1 = mean collapse (noisy target).")
    print("spread/err ~ 1 = diverse-but-wrong samples (missing conditioning info).")


if __name__ == "__main__":
    main()
