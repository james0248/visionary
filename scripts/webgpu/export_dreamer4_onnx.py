import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax2onnx
import numpy as np
import onnx
import onnxruntime as ort

from visionary.common.checkpoint import (
    resolve_model_export_step,
    restore_model_export_single_device,
)
from visionary.export.onnx_wrappers import (
    apply_dynamics_uncached,
    apply_tokenizer_decoder,
    dynamics_shapes,
    tokenizer_shapes,
)


TOKENIZER_DECODER_NAME = "breakout_tokenizer_decoder_b1_t64"
DYNAMICS_UNCACHED_NAME = "breakout_dynamics_b1_t64"
MANIFEST_NAME = "breakout_onnx_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Dreamer4 tokenizer decoder and dynamics models to ONNX. "
            "The browser runtime intentionally does not export the tokenizer encoder; "
            "context latents should be precomputed as web artifacts."
        )
    )
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--tokenizer_step", type=int, default=None)
    parser.add_argument("--dynamics_dir", required=True)
    parser.add_argument("--dynamics_step", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("webgpu_app/assets"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--opset", type=int, default=23)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--context_latents",
        type=Path,
        default=None,
        help=(
            "Optional precomputed context latent artifact to record in the manifest. "
            "The exporter does not create browser encoder ONNX graphs."
        ),
    )
    parser.add_argument(
        "--export_cached",
        action="store_true",
        help=(
            "Reserved for the required cached dynamics graphs. Currently fails with a "
            "clear message until inference-only temporal KV kernels are implemented."
        ),
    )
    return parser.parse_args()


def require_static_phase1_args(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("Phase 1 ONNX export supports only --batch_size 1.")
    if args.seq_len != 64:
        raise ValueError("Phase 1 ONNX export supports only --seq_len 64.")


def ensure_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")


def external_data_path(path: Path) -> Path:
    return path.with_name(path.name + ".data")


def remove_existing_export(path: Path, *, overwrite: bool) -> None:
    ensure_output(path, overwrite=overwrite)
    sidecar = external_data_path(path)
    ensure_output(sidecar, overwrite=overwrite)
    if overwrite:
        path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version_or_unknown(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def seeded_inputs(
    *,
    seed: int,
    tokenizer_shape: tuple[int, int, int, int],
    dynamics_shape: tuple[int, int, int, int],
    level_shape: tuple[int, int],
    max_step_size: int,
    num_actions: int,
) -> dict[str, jax.Array]:
    latent_key, z_key, action_key, step_key, signal_key = jax.random.split(jax.random.key(seed), 5)
    step_levels = jax.random.randint(
        step_key,
        level_shape,
        minval=0,
        maxval=max_step_size,
        dtype=jnp.int32,
    )
    step_counts = 1 << step_levels
    signal_levels = jax.random.randint(
        signal_key,
        level_shape,
        minval=0,
        maxval=step_counts,
        dtype=jnp.int32,
    )
    return {
        "latent": jax.random.normal(latent_key, tokenizer_shape, dtype=jnp.float32),
        "z": jax.random.normal(z_key, dynamics_shape, dtype=jnp.float32),
        "actions": jax.random.randint(
            action_key,
            level_shape,
            minval=0,
            maxval=num_actions,
            dtype=jnp.int32,
        ),
        "step_levels": step_levels,
        "signal_levels": signal_levels,
    }


def export_to_onnx(
    *,
    fn,
    inputs: tuple[jax.Array, ...],
    output_path: Path,
    model_name: str,
    opset: int,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    overwrite: bool,
) -> None:
    remove_existing_export(output_path, overwrite=overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jax2onnx.to_onnx(
        fn,
        inputs=inputs,
        model_name=model_name,
        opset=opset,
        return_mode="file",
        output_path=output_path,
        input_names=input_names,
        output_names=output_names,
    )
    onnx.checker.check_model(output_path.as_posix())


def run_ort(path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    outputs = session.run(None, feeds)
    return {output.name: value for output, value in zip(session.get_outputs(), outputs)}


def compare_arrays(
    expected: np.ndarray, actual: np.ndarray, *, atol: float, rtol: float
) -> dict[str, Any]:
    diff = np.abs(expected - actual)
    denom = np.maximum(np.abs(expected), np.asarray(1e-8, dtype=expected.dtype))
    rel = diff / denom
    passed = bool(np.allclose(expected, actual, atol=atol, rtol=rtol))
    return {
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": float(np.max(diff)),
        "mean_abs_error": float(np.mean(diff)),
        "max_rel_error": float(np.max(rel)),
        "mean_rel_error": float(np.mean(rel)),
        "passed": passed,
    }


def tensor_spec(dtype: str, shape: tuple[int, ...]) -> dict[str, Any]:
    return {"dtype": dtype, "shape": list(shape)}


def export_file_metadata(path: Path) -> dict[str, Any]:
    metadata = {
        "path": path.name,
        "sha256": sha256_file(path),
    }
    sidecar = external_data_path(path)
    if sidecar.exists():
        metadata["external_data"] = [
            {
                "path": sidecar.name,
                "sha256": sha256_file(sidecar),
            }
        ]
    else:
        metadata["external_data"] = []
    return metadata


def cache_contract(dyn_shapes) -> dict[str, Any]:
    cache_shape = list(dyn_shapes.cache)
    return {
        "status": "contract_only",
        "reason": (
            "Cached ONNX graphs require inference-only temporal attention kernels. "
            "Do not treat the uncached dynamics graph as production browser success."
        ),
        "static_axes": True,
        "context_length": dyn_shapes.context_length,
        "temporal_blocks": dyn_shapes.temporal_blocks,
        "total_tokens": dyn_shapes.total_tokens,
        "num_kv_heads": dyn_shapes.num_kv_heads,
        "head_dim": dyn_shapes.head_dim,
        "tensors": {
            "k_cache": tensor_spec("float32", tuple(cache_shape)),
            "v_cache": tensor_spec("float32", tuple(cache_shape)),
            "cache_length": tensor_spec("int32", (1,)),
        },
        "ownership": "browser_runtime",
        "invalidation": [
            "new_episode",
            "checkpoint_or_config_change",
            "context_latent_change",
            "action_history_change",
            "latent_history_change",
            "context_tau_change",
            "batch_order_change",
        ],
        "target_frame_policy": (
            "Do not commit or reuse target-frame cache entries across diffusion/sample "
            "iterations unless z_t, action_t, step_level_t, and signal_level_t are identical."
        ),
    }


def export_cached_requested() -> None:
    raise NotImplementedError(
        "--export_cached was requested, but cached dynamics ONNX export requires "
        "inference-only temporal KV attention kernels that are not yet implemented. "
        "The current script emits the cache ABI in the manifest and exports the "
        "uncached baseline for conversion/parity validation."
    )


def validate_single_output(
    *,
    path: Path,
    feeds: dict[str, jax.Array],
    output_name: str,
    expected: jax.Array,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    ort_feeds = {name: np.asarray(jax.device_get(value)) for name, value in feeds.items()}
    actual = run_ort(path, ort_feeds)[output_name]
    return compare_arrays(
        np.asarray(jax.device_get(expected)),
        actual,
        atol=atol,
        rtol=rtol,
    )


def main() -> None:
    args = parse_args()
    require_static_phase1_args(args)
    if args.export_cached:
        export_cached_requested()

    tokenizer_step = resolve_model_export_step(args.tokenizer_dir, args.tokenizer_step)
    dynamics_step = resolve_model_export_step(args.dynamics_dir, args.dynamics_step)
    tokenizer_cfg, tokenizer_variables = restore_model_export_single_device(
        args.tokenizer_dir,
        step=tokenizer_step,
    )
    dynamics_cfg, dynamics_variables = restore_model_export_single_device(
        args.dynamics_dir,
        step=dynamics_step,
    )

    tok_shapes = tokenizer_shapes(
        tokenizer_cfg,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    dyn_shapes = dynamics_shapes(
        dynamics_cfg,
        tok_shapes,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    if dyn_shapes.context_length != args.seq_len:
        raise ValueError(
            "Phase 1 fixed-shape export expects --seq_len to match dynamics context_length, "
            f"got seq_len={args.seq_len} and context_length={dyn_shapes.context_length}."
        )

    inputs = seeded_inputs(
        seed=args.seed,
        tokenizer_shape=tok_shapes.latent,
        dynamics_shape=dyn_shapes.z,
        level_shape=dyn_shapes.levels,
        max_step_size=int(dynamics_cfg.max_step_size),
        num_actions=int(dynamics_cfg.num_actions),
    )

    decoder_path = args.out_dir / f"{TOKENIZER_DECODER_NAME}.onnx"
    dynamics_path = args.out_dir / f"{DYNAMICS_UNCACHED_NAME}.onnx"
    manifest_path = args.out_dir / MANIFEST_NAME
    ensure_output(manifest_path, overwrite=args.overwrite)

    def decoder_fn(latent: jax.Array) -> jax.Array:
        return apply_tokenizer_decoder(tokenizer_variables, tokenizer_cfg, latent)

    def dynamics_fn(
        z: jax.Array,
        actions: jax.Array,
        step_levels: jax.Array,
        signal_levels: jax.Array,
    ) -> jax.Array:
        return apply_dynamics_uncached(
            dynamics_variables,
            dynamics_cfg,
            z,
            actions,
            step_levels,
            signal_levels,
        )

    export_to_onnx(
        fn=decoder_fn,
        inputs=(inputs["latent"],),
        output_path=decoder_path,
        model_name=TOKENIZER_DECODER_NAME,
        opset=args.opset,
        input_names=("latent",),
        output_names=("patches",),
        overwrite=args.overwrite,
    )
    export_to_onnx(
        fn=dynamics_fn,
        inputs=(
            inputs["z"],
            inputs["actions"],
            inputs["step_levels"],
            inputs["signal_levels"],
        ),
        output_path=dynamics_path,
        model_name=DYNAMICS_UNCACHED_NAME,
        opset=args.opset,
        input_names=("z", "actions", "step_levels", "signal_levels"),
        output_names=("pred_z",),
        overwrite=args.overwrite,
    )

    validation = {
        TOKENIZER_DECODER_NAME: {"skipped": not args.validate},
        DYNAMICS_UNCACHED_NAME: {"skipped": not args.validate},
    }
    if args.validate:
        validation[TOKENIZER_DECODER_NAME] = validate_single_output(
            path=decoder_path,
            feeds={"latent": inputs["latent"]},
            output_name="patches",
            expected=decoder_fn(inputs["latent"]),
            atol=args.atol,
            rtol=args.rtol,
        )
        validation[DYNAMICS_UNCACHED_NAME] = validate_single_output(
            path=dynamics_path,
            feeds={
                "z": inputs["z"],
                "actions": inputs["actions"],
                "step_levels": inputs["step_levels"],
                "signal_levels": inputs["signal_levels"],
            },
            output_name="pred_z",
            expected=dynamics_fn(
                inputs["z"],
                inputs["actions"],
                inputs["step_levels"],
                inputs["signal_levels"],
            ),
            atol=args.atol,
            rtol=args.rtol,
        )

        failed = [name for name, result in validation.items() if not result.get("passed", False)]
        if failed:
            raise AssertionError(f"ONNX validation failed for: {failed}")

    context_artifact = None
    if args.context_latents is not None:
        context_artifact = {
            "path": args.context_latents.as_posix(),
            "sha256": sha256_file(args.context_latents) if args.context_latents.exists() else None,
            "expected_latent_shape": list(tok_shapes.latent),
            "expected_dynamics_shape": list(dyn_shapes.z),
        }

    decoder_files = export_file_metadata(decoder_path)
    dynamics_files = export_file_metadata(dynamics_path)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "opset": args.opset,
        "axes_policy": {
            "static": True,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "context_length": dyn_shapes.context_length,
        },
        "tool_versions": {
            "jax": version_or_unknown(jax),
            "jax2onnx": version_or_unknown(jax2onnx),
            "onnx": version_or_unknown(onnx),
            "onnxruntime": version_or_unknown(ort),
        },
        "checkpoints": {
            "tokenizer_dir": str(args.tokenizer_dir),
            "tokenizer_step": tokenizer_step,
            "dynamics_dir": str(args.dynamics_dir),
            "dynamics_step": dynamics_step,
        },
        "context_latents": context_artifact,
        "tokenizer": {
            "num_latents": tok_shapes.num_latents,
            "channel_dim": tok_shapes.channel_dim,
            "patch_count": tok_shapes.patch_count,
            "patch_dim": tok_shapes.patch_dim,
            "resize_shape": list(tokenizer_cfg.resize_shape),
            "pad_width": list(tokenizer_cfg.pad_width),
            "patch_size": int(tokenizer_cfg.patch_size),
        },
        "dynamics": {
            "context_length": dyn_shapes.context_length,
            "num_obs_tokens": dyn_shapes.num_obs_tokens,
            "token_dim": dyn_shapes.token_dim,
            "num_actions": int(dynamics_cfg.num_actions),
            "max_step_size": int(dynamics_cfg.max_step_size),
            "num_registers": int(dynamics_cfg.num_registers),
            "temporal_blocks": dyn_shapes.temporal_blocks,
            "total_tokens": dyn_shapes.total_tokens,
        },
        "exports": [
            {
                "name": TOKENIZER_DECODER_NAME,
                **decoder_files,
                "inputs": {"latent": tensor_spec("float32", tok_shapes.latent)},
                "outputs": {"patches": tensor_spec("float32", tok_shapes.patches)},
                "validation": validation[TOKENIZER_DECODER_NAME],
            },
            {
                "name": DYNAMICS_UNCACHED_NAME,
                **dynamics_files,
                "inputs": {
                    "z": tensor_spec("float32", dyn_shapes.z),
                    "actions": tensor_spec("int32", dyn_shapes.levels),
                    "step_levels": tensor_spec("int32", dyn_shapes.levels),
                    "signal_levels": tensor_spec("int32", dyn_shapes.levels),
                },
                "outputs": {"pred_z": tensor_spec("float32", dyn_shapes.z)},
                "validation": validation[DYNAMICS_UNCACHED_NAME],
                "production_browser_ready": False,
            },
        ],
        "cache_contract": cache_contract(dyn_shapes),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {decoder_path}")
    print(f"Wrote {dynamics_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
