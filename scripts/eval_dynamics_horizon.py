"""Measure how dynamics error grows with rollout horizon, in latent space.

Separates two explanations for the model's broken manipulation: either it never
learned contact physics, or it learned it and autoregressive error destroys it.
A one-step prediction from true context is a teacher-forced probe -- if that is
already bad on the moving tokens, no amount of rollout stability work will help.

Errors are reported on all tokens and on the moving tokens separately, since
~90% of the tokens are static background that both cases predict perfectly.

    uv run python scripts/eval_dynamics_horizon.py \
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

from visionary.dataset import align_actions_to_frames
from visionary.dynamics import DynamicsModel

from eval_dynamics_videos import load_train_config, restore_params

logger = logging.getLogger(__name__)


def moving_token_mask(latents: np.ndarray, fraction: float) -> np.ndarray:
    """Tokens that actually change frame to frame -- the ones carrying the motion."""
    delta = np.abs(np.diff(latents, axis=0, prepend=latents[:1])).mean(-1)
    cutoff = np.quantile(delta, 1.0 - fraction, axis=-1, keepdims=True)
    return delta >= cutoff


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Dynamics error vs rollout horizon.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="horizon.json")
    parser.add_argument("--step", type=int)
    parser.add_argument("--indices", default="2,7")
    parser.add_argument("--horizons", default="1,2,4,8,16,32,64")
    parser.add_argument("--starts", default="48,80,112", help="Where each rollout begins.")
    parser.add_argument("--total_frames", type=int, default=192)
    parser.add_argument("--context_tau", type=float, default=0.9)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--moving_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_train_config(args.checkpoint_dir)
    paths = sorted(str(p) for p in Path(args.data_dir).glob("*.arecord"))
    source = grain.ArrayRecordDataSource(paths)
    horizons = [int(h) for h in args.horizons.split(",")]
    starts = [int(s) for s in args.starts.split(",")]

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
        return (
            np.asarray(video[offset:stop], dtype=np.float32)[None],
            align_actions_to_frames(actions[offset:stop], prev_action=before)[None],
        )

    video, actions = sample_for(int(args.indices.split(",")[0]))
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
        video, actions = sample_for(index)
        truth = video[0]
        mask = moving_token_mask(truth, args.moving_fraction)
        for start in starts:
            for horizon in horizons:
                if start + horizon > truth.shape[0]:
                    continue
                out = np.asarray(
                    jax.device_get(rollout(params, video, actions, index, start, horizon))
                )[0]
                # only the last generated frame: that is the state after `horizon`
                # autoregressive steps, which is what the curve is about
                frame = start + horizon - 1
                err = ((out[frame] - truth[frame]) ** 2).mean(-1)
                scale = (truth[frame] ** 2).mean(-1)
                m = mask[frame]
                entry = {
                    "index": index,
                    "start": start,
                    "horizon": horizon,
                    "rel_err_all": float(err.mean() / scale.mean()),
                    "rel_err_moving": float(err[m].mean() / scale[m].mean()),
                    "rel_err_static": float(err[~m].mean() / scale[~m].mean()),
                }
                results.append(entry)
                logger.info(
                    "idx %d start %3d horizon %2d | rel err all %.4f moving %.4f static %.4f",
                    index,
                    start,
                    horizon,
                    entry["rel_err_all"],
                    entry["rel_err_moving"],
                    entry["rel_err_static"],
                )

    Path(args.output).write_text(json.dumps({"step": step, "results": results}, indent=2))
    print("\nhorizon | rel err moving tokens (mean over clips and starts)")
    for horizon in horizons:
        vals = [r["rel_err_moving"] for r in results if r["horizon"] == horizon]
        stat = [r["rel_err_static"] for r in results if r["horizon"] == horizon]
        if vals:
            print(f"{horizon:7d} | {np.mean(vals):.4f}   (static {np.mean(stat):.4f})")


if __name__ == "__main__":
    main()
