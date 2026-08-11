import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from visionary.models.dreamer4.tokenizer import Tokenizer

PRESETS = {
    "tiny": dict(
        num_layers=4, num_latents=8, num_heads=2, num_kv_heads=1, model_dim=64,
        mlp_hidden_dim=128, head_dim=32, channel_dim=8, resize_shape=[64, 64],
        pad_width=[0, 0], patch_size=16, base=10000.0, decoder_num_layers=4,
    ),
    "mid": dict(
        num_layers=4, num_latents=8, num_heads=4, num_kv_heads=2, model_dim=128,
        mlp_hidden_dim=256, head_dim=32, channel_dim=8, resize_shape=[240, 320],
        pad_width=[0, 0], patch_size=16, base=10000.0, decoder_num_layers=4,
    ),
    "full": dict(
        num_layers=12, num_latents=256, num_heads=12, num_kv_heads=3, model_dim=768,
        mlp_hidden_dim=2304, head_dim=64, channel_dim=16, resize_shape=[240, 320],
        pad_width=[0, 0], patch_size=16, base=10000.0, decoder_num_layers=8,
        remat=True, remat_policy="dots_with_no_batch_dims_saveable",
    ),
}
BATCH = {"tiny": (2, 4), "mid": (2, 4), "full": (2, 16)}


def tree_norms(tree):
    flat = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        key = "/".join(str(getattr(k, "key", k)) for k in path[:3])
        flat.setdefault(key, 0.0)
        flat[key] += float(jnp.sum(jnp.square(leaf.astype(jnp.float32))))
    return {k: float(np.sqrt(v)) for k, v in sorted(flat.items())}


def fingerprint(preset: str, splash: bool = False) -> dict:
    cfg = dict(PRESETS[preset])
    model = Tokenizer(**cfg, use_splash=splash, dtype=jnp.float32)
    batch_size, frames = BATCH[preset]
    tokens = (cfg["resize_shape"][0] // 16) * (cfg["resize_shape"][1] // 16)
    patch_dim = 16 * 16 * 3

    rng = np.random.default_rng(0)
    video = jnp.asarray(
        rng.integers(0, 256, (batch_size, frames, tokens, patch_dim)), jnp.float32
    )
    params = model.init(
        {"params": jax.random.key(0), "sample": jax.random.key(1)}, {"video": video}
    )

    def loss_fn(p):
        recon, mask, latent = model.apply(
            p,
            {"video": video / 255.0},
            mask_prob=0.5,
            independent=jnp.zeros((batch_size,), bool),
            method=Tokenizer.reconstruct,
            rngs={"sample": jax.random.key(2)},
        )
        loss = jnp.mean(jnp.square(recon - video / 255.0))
        return loss, (recon, latent, mask)

    (loss, (recon, latent, mask)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    recon = recon.astype(jnp.float32)
    latent = latent.astype(jnp.float32)
    return {
        "preset": preset,
        "loss": float(loss),
        "recon_mean": float(jnp.mean(recon)),
        "recon_std": float(jnp.std(recon)),
        "recon_absmax": float(jnp.max(jnp.abs(recon))),
        "latent_mean": float(jnp.mean(latent)),
        "latent_std": float(jnp.std(latent)),
        "latent_absmax": float(jnp.max(jnp.abs(latent))),
        "mask_ratio": float(jnp.mean(mask)),
        "grad_norms": tree_norms(grads),
    }


def compare(baseline: dict, current: dict, rtol: float) -> list[str]:
    failures = []

    def close(a, b, name):
        denom = max(abs(a), abs(b), 1e-8)
        rel = abs(a - b) / denom
        if rel > rtol:
            failures.append(f"{name}: baseline={a:.8g} current={b:.8g} rel={rel:.3g}")

    for key in ("loss", "recon_mean", "recon_std", "recon_absmax",
                "latent_mean", "latent_std", "latent_absmax", "mask_ratio"):
        close(baseline[key], current[key], key)
    for key in baseline["grad_norms"]:
        if key not in current["grad_norms"]:
            failures.append(f"grad_norms/{key}: missing in current")
            continue
        close(baseline["grad_norms"][key], current["grad_norms"][key], f"grad_norms/{key}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default="tiny")
    parser.add_argument("--save")
    parser.add_argument("--check")
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--splash", action="store_true")
    args = parser.parse_args()

    result = fingerprint(args.preset, splash=args.splash)
    if args.save:
        Path(args.save).write_text(json.dumps(result, indent=1))
        print(f"saved fingerprint to {args.save} (loss={result['loss']:.8g})")
    if args.check:
        baseline = json.loads(Path(args.check).read_text())
        if baseline.get("preset") != args.preset:
            raise SystemExit(f"preset mismatch: baseline={baseline.get('preset')}")
        failures = compare(baseline, result, args.rtol)
        if failures:
            print(f"NUMERICS MISMATCH ({len(failures)}):")
            for f in failures[:20]:
                print("  ", f)
            raise SystemExit(1)
        print(f"numerics match baseline within rtol={args.rtol} (loss={result['loss']:.8g})")


if __name__ == "__main__":
    main()
