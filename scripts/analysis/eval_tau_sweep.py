"""Teacher-forced denoising error as a function of tau, split by token motion
and by proximity to gripper events.

The training loss weights tau as 0.9*tau + 0.1 while the sampler divides the
predicted velocity by (1 - tau), so under-training at high tau is amplified at
inference. This sweeps a fixed tau across the eval clips and reports where the
error actually sits. Two noising patterns: 'uniform' puts every frame at the
probe tau (the training marginal); 'alternating' holds even frames at the
inference context_tau and probes odd frames, which approximates the one-step
inference condition with dense statistics.

All tau values and both patterns reuse a single compiled forward, so this is
cheap despite the grid.

    uv run python scripts/eval_tau_sweep.py \
        --checkpoint_dir gs://.../so101_dynamics --data_dir data/so101/dyn/eval
"""

import argparse
import io
import json
import logging
from pathlib import Path

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visionary.dataset import align_actions_to_frames

from eval_dynamics_videos import load_train_config, restore_params

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Denoising error vs tau.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="tau_sweep.json")
    parser.add_argument("--step", type=int)
    parser.add_argument("--indices", default="0,1,2,3,6,7")
    parser.add_argument("--taus", default="0.25,0.5,0.75,0.875,0.9375")
    parser.add_argument("--context_tau", type=float, default=0.9)
    parser.add_argument("--total_frames", type=int, default=192)
    parser.add_argument("--moving_fraction", type=float, default=0.1)
    parser.add_argument("--grip_delta", type=float, default=0.08)
    parser.add_argument("--event_window", type=int, default=8)
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
        return (
            np.asarray(video[offset:stop], dtype=np.float32)[None],
            align_actions_to_frames(actions[offset:stop], prev_action=before)[None],
            actions[offset:stop],
        )

    video, actions, _ = sample_for(int(args.indices.split(",")[0]))
    model, params, step = restore_params(
        cfg, args.checkpoint_dir, args.step, {"video": video, "actions": actions}
    )
    logger.info("Restored params from step %d", step)

    # the finest shortcut level: tau lives on a grid of 1/step_count
    level = model.max_step_size - 1
    step_count = 1 << level

    @jax.jit
    def tau_probe(params, video, actions, seed, signal_levels):
        z_target = rearrange(
            jnp.asarray(video, jnp.float32),
            "b t (n k) d -> b t n (k d)",
            n=model.num_obs_tokens,
        )
        step_levels = jnp.full(signal_levels.shape, level, dtype=jnp.int32)
        tau = (signal_levels.astype(jnp.float32) / step_count)[..., None, None]
        z_noise = jax.random.normal(jax.random.key(seed), z_target.shape, dtype=jnp.float32)
        z_noised = tau * z_target + (1.0 - tau) * z_noise
        z_pred = model.apply(
            params, z_noised, jnp.asarray(actions, jnp.float32), step_levels, signal_levels
        )
        err = ((z_pred - z_target) ** 2).mean(-1)
        scale = (z_target**2).mean(-1)
        return err, scale

    taus = [float(t) for t in args.taus.split(",")]
    results = []
    for index in (int(i) for i in args.indices.split(",")):
        video, actions, raw_actions = sample_for(index)
        z = rearrange(video[0], "t (n k) d -> t n (k d)", n=model.num_obs_tokens)
        delta = np.abs(np.diff(z, axis=0, prepend=z[:1])).mean(-1)
        cutoff = np.quantile(delta, 1.0 - args.moving_fraction, axis=-1, keepdims=True)
        moving = delta >= cutoff
        grip = raw_actions[:, -1]
        near = np.zeros(len(z), bool)
        for e in np.where(np.abs(np.diff(grip)) > args.grip_delta)[0]:
            near[max(0, e - args.event_window) : e + args.event_window] = True
        T = len(z)
        ctx_signal = int(round(args.context_tau * step_count))
        for pattern in ("uniform", "alternating"):
            for tau in taus:
                sig = np.full((1, T), int(round(tau * step_count)), np.int32)
                scored = np.ones(T, bool)
                if pattern == "alternating":
                    sig[:, ::2] = ctx_signal
                    scored = np.arange(T) % 2 == 1
                err, scale = (
                    np.asarray(x)
                    for x in jax.device_get(
                        tau_probe(params, video, actions, args.seed + index, jnp.asarray(sig))
                    )
                )
                err, scale = err[0], scale[0]
                scored[:4] = False  # first frames have no meaningful context
                def sel(fm, tm):
                    m = fm[:, None] & tm
                    return float(err[m].sum() / max(scale[m].sum(), 1e-9)) if m.any() else None
                entry = {
                    "index": index,
                    "pattern": pattern,
                    "tau": tau,
                    "rel_err_moving": sel(scored, moving),
                    "rel_err_static": sel(scored, ~moving),
                    "rel_err_moving_grip": sel(scored & near, moving),
                    "rel_err_moving_free": sel(scored & ~near, moving),
                }
                results.append(entry)
                logger.info(
                    "idx %d %-11s tau %.4f | moving %.4f static %.4f | grip %s free %s",
                    index, pattern, tau,
                    entry["rel_err_moving"], entry["rel_err_static"],
                    f"{entry['rel_err_moving_grip']:.4f}" if entry["rel_err_moving_grip"] else "-",
                    f"{entry['rel_err_moving_free']:.4f}" if entry["rel_err_moving_free"] else "-",
                )

    Path(args.output).write_text(json.dumps({"step": step, "results": results}, indent=2))
    print("\npattern     tau    | moving  static  grip    free   (mean over clips)")
    for pattern in ("uniform", "alternating"):
        for tau in taus:
            rows = [r for r in results if r["pattern"] == pattern and r["tau"] == tau]
            def mean(key):
                vals = [r[key] for r in rows if r[key] is not None]
                return f"{np.mean(vals):.4f}" if vals else "  -   "
            print(
                f"{pattern:11s} {tau:.4f} | {mean('rel_err_moving')} {mean('rel_err_static')} "
                f"{mean('rel_err_moving_grip')} {mean('rel_err_moving_free')}"
            )


if __name__ == "__main__":
    main()
