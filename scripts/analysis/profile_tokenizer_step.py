import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra.utils import instantiate  # noqa: E402

from visionary.common.train_state import TokenizerTrainState  # noqa: E402
from visionary.models.dreamer4.tokenizer_preprocessor import TokenizerPreprocessor  # noqa: E402

from train_tokenizer import build_optimizer, train_step  # noqa: E402

V5E_PEAK_BF16 = 197e12
V6E_PEAK_BF16 = 918e12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "config" / "so101_tokenizer.yaml"))
    parser.add_argument("--per_device_batch", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--trace_dir")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--mode", choices=["full", "grad", "fwd"], default="full")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    n_dev = jax.local_device_count()
    batch_size = args.per_device_batch * n_dev
    frames = args.frames

    model = instantiate(cfg.tokenizer)
    preprocessor = TokenizerPreprocessor(
        resize_shape=tuple(cfg.tokenizer.resize_shape),
        pad_width=tuple(cfg.tokenizer.pad_width),
        patch_size=int(cfg.tokenizer.patch_size),
    )

    rng = np.random.default_rng(0)
    h, w = cfg.tokenizer.resize_shape
    video_u8 = rng.integers(0, 256, (batch_size, frames, h, w, 3), dtype=np.uint8)
    patches = preprocessor.preprocess_video(video_u8)
    batch = {"video": jnp.asarray(patches)}

    params = model.init({"params": jax.random.key(0), "sample": jax.random.key(1)}, batch)
    state = TokenizerTrainState.create(apply_fn=model.apply, params=params, tx=build_optimizer(cfg))
    n_params = sum(x.size for x in jax.tree.leaves(params))

    mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()), ("data",))
    batch = jax.device_put(batch, NamedSharding(mesh, P("data")))
    state = jax.device_put(state, NamedSharding(mesh, P()))

    from visionary.models.dreamer4.tokenizer import Tokenizer as _Tok

    def fwd_fn(state_, batch_, key_, step_, lpips_weight, lpips_frame_stride, preprocessor):
        recon, mask, latent = state_.apply_fn(state_.params, batch_, method=_Tok.reconstruct, rngs={"sample": key_})
        return state_, {"loss": jnp.mean(jnp.square(recon))}

    def grad_fn(state_, batch_, key_, step_, lpips_weight, lpips_frame_stride, preprocessor):
        def loss_fn(p):
            recon, mask, latent = state_.apply_fn(p, batch_, method=_Tok.reconstruct, rngs={"sample": key_})
            return jnp.mean(jnp.square(recon))

        loss, grads = jax.value_and_grad(loss_fn)(state_.params)
        gnorm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
        return state_, {"loss": loss + 0 * gnorm}

    step_fn = {"full": train_step, "grad": grad_fn, "fwd": fwd_fn}[args.mode]
    jit_step = jax.jit(
        step_fn,
        static_argnames=("lpips_weight", "lpips_frame_stride", "preprocessor"),
        donate_argnums=(0,),
    )
    key = jax.random.key(3)

    def run(step_idx, state):
        return jit_step(
            state,
            batch,
            key,
            step_idx,
            float(cfg.lpips_weight),
            int(cfg.lpips_frame_stride),
            preprocessor,
        )

    lowered = jit_step.lower(
        state,
        batch,
        key,
        0,
        float(cfg.lpips_weight),
        int(cfg.lpips_frame_stride),
        preprocessor,
    )
    compiled = lowered.compile()
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    flops = float(cost.get("flops", 0.0))

    for i in range(args.warmup):
        state, metrics = run(i, state)
    jax.block_until_ready(metrics["loss"])

    if args.trace_dir:
        jax.profiler.start_trace(args.trace_dir)
    start = time.perf_counter()
    for i in range(args.warmup, args.warmup + args.steps):
        state, metrics = run(i, state)
    jax.block_until_ready(metrics["loss"])
    elapsed = time.perf_counter() - start
    if args.trace_dir:
        jax.profiler.stop_trace()

    step_ms = elapsed / args.steps * 1000
    device_kind = jax.devices()[0].device_kind.lower()
    peak = V6E_PEAK_BF16 if "v6" in device_kind else V5E_PEAK_BF16
    mxu = flops / (elapsed / args.steps) / (peak * n_dev)

    result = {
        "tag": args.tag,
        "device": jax.devices()[0].device_kind,
        "n_devices": n_dev,
        "params_m": round(n_params / 1e6, 1),
        "batch_global": batch_size,
        "frames": frames,
        "step_ms": round(step_ms, 2),
        "steps_per_s": round(1000 / step_ms, 2),
        "tflops_per_step": round(flops / 1e12, 2),
        "mxu_pct": round(100 * mxu, 1),
        "loss": float(metrics["loss"]),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
