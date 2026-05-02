import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
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
    apply_dynamics_cached_prefill,
    apply_dynamics_cached_prefill_layer_cache,
    apply_dynamics_cached_sample_step,
    apply_dynamics_cached_sample_step_append_context,
    apply_dynamics_cached_sample_step_append_context_full_cache,
    apply_dynamics_cached_sample_step_append_context_full_cache_entries,
    apply_dynamics_cached_sample_step_append_context_layer_cache,
    apply_dynamics_cached_step,
    apply_dynamics_uncached,
    apply_tokenizer_decode_z,
    apply_tokenizer_decoder,
    dynamics_shapes,
    tokenizer_shapes,
)


TOKENIZER_DECODER_NAME = "breakout_tokenizer_decoder_b1_t64"
TOKENIZER_DECODER_STEP_NAME = "breakout_tokenizer_decoder_b1_t1"
TOKENIZER_DECODE_Z_STEP_NAME = "breakout_tokenizer_decode_z_b1_t1"
DYNAMICS_UNCACHED_NAME = "breakout_dynamics_b1_t64"
DYNAMICS_CACHED_PREFILL_NAME = "breakout_dynamics_prefill_cached_b1_t64"
DYNAMICS_CACHED_PREFILL_LAYER_NAME = "breakout_dynamics_prefill_layer_cached_b1_t64"
DYNAMICS_CACHED_STEP_NAME = "breakout_dynamics_step_cached_b1_t1"
DYNAMICS_CACHED_SAMPLE_STEP_NAME = "breakout_dynamics_cached_sample_step_b1_t1_s4"
DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME = "breakout_dynamics_cached_sample_step_slide_b1_t1_s4"
DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME = "breakout_dynamics_sample_append_context_b1_t1_s4"
DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME = (
    "breakout_dynamics_sample_append_context_slide_b1_t1_s4"
)
DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME = (
    "breakout_dynamics_sample_append_context_slide_full_cache_b1_t1_s4"
)
DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME = (
    "breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4"
)
DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME = (
    "breakout_dynamics_sample_append_context_slide_layer_b1_t1_s4"
)
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
    parser.add_argument(
        "--raw_out_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory where the freshly exported ONNX files are copied before "
            "any simplification, ORT optimization, or WebGPU graph rewrites. Use this "
            "with scripts/webgpu/compare_onnx_artifacts.py as a behavior-preserving "
            "optimization gate."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--opset", type=int, default=23)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--context_tau", type=float, default=29 / 32)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--simplify_onnx",
        action="store_true",
        help=(
            "Run onnxsim on each exported artifact before optional ONNX Runtime "
            "optimization. This is experimental and intended to test whether shape "
            "graph simplification reduces WebGPU CPU reshape/provider-boundary copies."
        ),
    )
    parser.add_argument(
        "--simplify_demo_only",
        action="store_true",
        help=(
            "When --simplify_onnx is enabled, run onnxsim only on the browser demo "
            "artifacts used by the benchmark hot path."
        ),
    )
    parser.add_argument(
        "--skip_onnx_optimization",
        action="store_true",
        help=(
            "Skip the offline ONNX Runtime graph cleanup pass. By default exported "
            "browser artifacts are optimized with ORT_ENABLE_EXTENDED to fold static "
            "shape scaffolding before WebGPU benchmarking."
        ),
    )
    parser.add_argument(
        "--skip_singleton_reshape_rewrite",
        action="store_true",
        help=(
            "Skip the WebGPU layout rewrite that replaces safe singleton-only Reshape "
            "nodes with Squeeze/Unsqueeze. Enabled by default because ORT WebGPU keeps "
            "Squeeze/Unsqueeze on device but falls back to CPU for standalone Reshape."
        ),
    )
    parser.add_argument(
        "--head_projection_rewrite",
        choices=("einsum", "layout"),
        default="einsum",
        help=(
            "How to remove attention head projection Reshape nodes. 'einsum' replaces "
            "Gemm+Reshape with rank-aware Einsum kernels; 'layout' keeps the Gemm kernels "
            "and uses static Split/Squeeze/Unsqueeze/Concat layout ops."
        ),
    )
    parser.add_argument(
        "--rotary_embedding_rewrite",
        action="store_true",
        help=(
            "Experimental: replace exported RoPE Split/Mul/Add/Sub/Concat islands with "
            "ORT WebGPU's contrib RotaryEmbedding op. This is opt-in because the required "
            "layout transposes can be slower than the unfused graph on current ORT WebGPU."
        ),
    )
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
            "Export cached dynamics prefill/step graphs and a single-frame decoder for "
            "the browser demo benchmark."
        ),
    )
    return parser.parse_args()


def require_static_phase1_args(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("Phase 1 ONNX export supports only --batch_size 1.")
    if args.seq_len != 64:
        raise ValueError("Phase 1 ONNX export supports only --seq_len 64.")
    if args.sample_steps <= 0 or args.sample_steps & (args.sample_steps - 1):
        raise ValueError(
            f"--sample_steps must be a positive power of two, got {args.sample_steps}."
        )


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


def copy_onnx_artifact(src: Path, dst: Path, *, overwrite: bool) -> dict[str, Any]:
    ensure_output(dst, overwrite=overwrite)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        dst.unlink(missing_ok=True)
        external_data_path(dst).unlink(missing_ok=True)
    shutil.copy2(src, dst)

    copied = [{"path": dst.name, "sha256": sha256_file(dst), "size_bytes": dst.stat().st_size}]
    src_sidecar = external_data_path(src)
    if src_sidecar.exists():
        dst_sidecar = external_data_path(dst)
        ensure_output(dst_sidecar, overwrite=overwrite)
        if overwrite:
            dst_sidecar.unlink(missing_ok=True)
        shutil.copy2(src_sidecar, dst_sidecar)
        copied.append(
            {
                "path": dst_sidecar.name,
                "sha256": sha256_file(dst_sidecar),
                "size_bytes": dst_sidecar.stat().st_size,
            }
        )
    return {"path": dst.name, "files": copied}


def snapshot_raw_artifacts(
    exported_paths: dict[str, Path],
    raw_out_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    copied = {
        name: copy_onnx_artifact(path, raw_out_dir / path.name, overwrite=overwrite)
        for name, path in exported_paths.items()
    }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Raw jax2onnx exports captured before ONNX simplification, ORT optimization, "
            "or WebGPU-specific graph rewrites."
        ),
        "artifacts": copied,
    }
    raw_manifest_path = raw_out_dir / "raw_onnx_artifacts_manifest.json"
    ensure_output(raw_manifest_path, overwrite=overwrite)
    raw_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "enabled": True,
        "path": raw_out_dir.as_posix(),
        "manifest": raw_manifest_path.name,
        "artifacts": copied,
    }


def op_counts(path: Path) -> Counter[str]:
    model = onnx.load(path.as_posix(), load_external_data=False)
    return Counter(node.op_type for node in model.graph.node)


def simplify_onnx_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    if before.get("RMSNormalization", 0):
        return {
            "enabled": False,
            "reason": "onnxsim does not support RMSNormalization at opset 23",
            "tool": "onnxsim",
            "unsupported_ops": {"RMSNormalization": int(before["RMSNormalization"])},
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
        }

    helper = Path(__file__).with_name("simplify_onnx_file.py")
    result = subprocess.run(
        [sys.executable, helper.as_posix(), path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"onnxsim subprocess failed for {path} with exit code {result.returncode}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"onnxsim subprocess for {path} did not return JSON.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        ) from exc


def optimize_onnx_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    tmp_path = path.with_name(path.stem + ".ort_optimized_tmp.onnx")
    tmp_path.unlink(missing_ok=True)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session_options.optimized_model_filepath = tmp_path.as_posix()
    ort.InferenceSession(
        path.as_posix(),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    onnx.checker.check_model(tmp_path.as_posix())
    tmp_path.replace(path)

    onnx.load(path.as_posix(), load_external_data=True)

    after = op_counts(path)
    tracked_ops = ("Reshape", "Concat", "Expand", "Cast", "Transpose", "Einsum", "Gemm")
    return {
        "enabled": True,
        "tool": "onnxruntime",
        "graph_optimization_level": "ORT_ENABLE_EXTENDED",
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
    }


def rewrite_slide_static_cache_ops_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)

    value_shapes: dict[str, tuple[int, ...]] = {}
    for value_info in itertools.chain(
        model.graph.input, model.graph.output, model.graph.value_info
    ):
        tensor_type = value_info.type.tensor_type
        if tensor_type.HasField("shape"):
            dims = []
            static = True
            for dim in tensor_type.shape.dim:
                if dim.dim_value:
                    dims.append(int(dim.dim_value))
                else:
                    static = False
                    break
            if static:
                value_shapes[value_info.name] = tuple(dims)
    for initializer in model.graph.initializer:
        value_shapes[initializer.name] = tuple(initializer.dims)

    initializer_names = {initializer.name for initializer in model.graph.initializer}

    def add_initializer_once(array: np.ndarray, name: str) -> None:
        if name not in initializer_names:
            model.graph.initializer.append(onnx.numpy_helper.from_array(array, name=name))
            initializer_names.add(name)

    rewrites: list[dict[str, Any]] = []
    rewritten_nodes: list[onnx.NodeProto] = []
    skip_outputs: set[str] = set()

    for node in model.graph.node:
        if node.op_type == "Min" and node.output and node.output[0] == "candidate_cache_length":
            add_initializer_once(np.asarray([64], dtype=np.int32), node.output[0])
            rewrites.append(
                {
                    "node": node.name,
                    "op": node.op_type,
                    "kind": "steady_state_cache_length_output",
                }
            )
            continue
        if node.op_type == "Min" and node.output and node.output[0] == "minimum_out_0":
            add_initializer_once(np.asarray(64, dtype=np.int32), node.output[0])
            rewrites.append(
                {
                    "node": node.name,
                    "op": node.op_type,
                    "kind": "steady_state_cache_length_clamp",
                }
            )
            continue

        if (
            node.op_type == "Reshape"
            and len(node.input) == 2
            and len(node.output) == 1
            and value_shapes.get(node.input[0]) == (36, 128)
            and value_shapes.get(node.output[0]) == (1, 36, 1, 2, 64)
        ):
            prefix = node.name or node.output[0]
            unsqueezed = f"{prefix}__static_unsqueeze"
            split0 = f"{prefix}__static_split0"
            split1 = f"{prefix}__static_split1"
            split0_unsqueezed = f"{prefix}__static_split0_unsqueeze"
            split1_unsqueezed = f"{prefix}__static_split1_unsqueeze"
            axes_outer = f"{prefix}__static_outer_axes"
            split_sizes = f"{prefix}__static_split_sizes"
            axes_head = f"{prefix}__static_head_axes"
            add_initializer_once(np.asarray([0, 2], dtype=np.int64), axes_outer)
            add_initializer_once(np.asarray([64, 64], dtype=np.int64), split_sizes)
            add_initializer_once(np.asarray([3], dtype=np.int64), axes_head)
            rewritten_nodes.extend(
                [
                    onnx.helper.make_node(
                        "Unsqueeze",
                        [node.input[0], axes_outer],
                        [unsqueezed],
                        name=f"{prefix}__static_unsqueeze",
                    ),
                    onnx.helper.make_node(
                        "Split",
                        [unsqueezed, split_sizes],
                        [split0, split1],
                        name=f"{prefix}__static_split",
                        axis=3,
                    ),
                    onnx.helper.make_node(
                        "Unsqueeze",
                        [split0, axes_head],
                        [split0_unsqueezed],
                        name=f"{prefix}__static_split0_unsqueeze",
                    ),
                    onnx.helper.make_node(
                        "Unsqueeze",
                        [split1, axes_head],
                        [split1_unsqueezed],
                        name=f"{prefix}__static_split1_unsqueeze",
                    ),
                    onnx.helper.make_node(
                        "Concat",
                        [split0_unsqueezed, split1_unsqueezed],
                        [node.output[0]],
                        name=f"{prefix}__static_concat",
                        axis=3,
                    ),
                ]
            )
            rewrites.append(
                {
                    "node": node.name,
                    "op": node.op_type,
                    "kind": "static_cache_projection_reshape",
                    "input_shape": [36, 128],
                    "output_shape": [1, 36, 1, 2, 64],
                }
            )
            continue

        if (
            node.op_type == "Unsqueeze"
            and len(node.input) == 2
            and len(node.output) == 1
            and value_shapes.get(node.input[0]) == (65,)
            and value_shapes.get(node.output[0]) == (1, 1, 1, 65)
        ):
            cast_consumers = [
                candidate
                for candidate in model.graph.node
                if candidate.op_type == "Cast"
                and len(candidate.input) == 1
                and candidate.input[0] == node.output[0]
                and len(candidate.output) == 1
            ]
            cast_to_float = None
            if len(cast_consumers) == 1:
                cast_to = next(
                    (attr for attr in cast_consumers[0].attribute if attr.name == "to"),
                    None,
                )
                if cast_to is not None and cast_to.i == onnx.TensorProto.FLOAT:
                    cast_to_float = cast_consumers[0]
            if cast_to_float is not None:
                casted = f"{node.name or node.output[0]}__cast_before_unsqueeze"
                rewritten_nodes.extend(
                    [
                        onnx.helper.make_node(
                            "Cast",
                            [node.input[0]],
                            [casted],
                            name=f"{node.name or node.output[0]}__cast_before_unsqueeze",
                            to=onnx.TensorProto.FLOAT,
                        ),
                        onnx.helper.make_node(
                            "Unsqueeze",
                            [casted, node.input[1]],
                            [cast_to_float.output[0]],
                            name=f"{node.name or node.output[0]}__float_unsqueeze",
                        ),
                    ]
                )
                skip_outputs.add(cast_to_float.output[0])
                rewrites.append(
                    {
                        "node": node.name,
                        "op": node.op_type,
                        "kind": "bool_unsqueeze_to_float_unsqueeze",
                    }
                )
                continue

        if node.output and any(output in skip_outputs for output in node.output):
            continue
        rewritten_nodes.append(node)

    if rewrites:
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)
        external_data_path(path).unlink(missing_ok=True)
        onnx.checker.check_model(model)
        onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    return {
        "enabled": True,
        "tool": "custom_slide_static_cache_webgpu_rewrite",
        "reason": (
            "The browser demo steady-state slide graph always runs with a full "
            "64-token cache. Static cache-length and layout rewrites keep the hot "
            "graph on WebGPU and unblock graph-capture experiments."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "rewrites": len(rewrites),
        "rewrite_examples": rewrites[:12],
        "tracked_ops_before": {
            op: int(before.get(op, 0))
            for op in ("Min", "Reshape", "Unsqueeze", "Split", "Concat", "Cast")
        },
        "tracked_ops_after": {
            op: int(after.get(op, 0))
            for op in ("Min", "Reshape", "Unsqueeze", "Split", "Concat", "Cast")
        },
    }


def rewrite_gather_int64_casts_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    rewrites: list[dict[str, Any]] = []
    remove_nodes: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Cast" or len(node.input) != 1 or len(node.output) != 1:
            continue
        to_attr = next((attr for attr in node.attribute if attr.name == "to"), None)
        if to_attr is None or to_attr.i != onnx.TensorProto.INT64:
            continue
        cast_output = node.output[0]
        cast_consumers = consumers.get(cast_output, [])
        if not cast_consumers or any(consumer.op_type != "Gather" for consumer in cast_consumers):
            continue
        for consumer in cast_consumers:
            for idx, input_name in enumerate(consumer.input):
                if input_name == cast_output:
                    consumer.input[idx] = node.input[0]
        remove_nodes.add(node.name)
        rewrites.append(
            {
                "cast": node.name,
                "input": node.input[0],
                "output": cast_output,
                "consumers": [consumer.name for consumer in cast_consumers],
            }
        )

    if rewrites:
        kept_nodes = [node for node in model.graph.node if node.name not in remove_nodes]
        del model.graph.node[:]
        model.graph.node.extend(kept_nodes)
        external_data_path(path).unlink(missing_ok=True)
        onnx.save_model(model, path.as_posix(), save_as_external_data=False)
        if op_counts(path).get("SimplifiedLayerNormalization", 0) == 0:
            onnx.checker.check_model(path.as_posix())

    after = op_counts(path)
    tracked_ops = ("Cast", "Gather", "MemcpyFromHost", "Reshape", "SimplifiedLayerNormalization")
    return {
        "enabled": True,
        "tool": "custom_gather_int64_cast_rewrite",
        "reason": "Remove int32->int64 casts when the result only feeds Gather; Gather accepts int32 indices and ORT WebGPU cannot cast to int64 on device.",
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "rewrites": {"cast_to_int64_before_gather": len(rewrites)},
        "rewrite_examples": rewrites[:12],
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
    }


def _tensor_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            return None
        dims.append(int(dim.dim_value))
    return tuple(dims)


def _singleton_layout_plan(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> tuple[list[int], list[int]] | None:
    if input_shape == output_shape:
        return ([], [])

    if len(output_shape) > len(input_shape):
        candidate_axes = [axis for axis, dim in enumerate(output_shape) if dim == 1]
        for remove_count in range(len(output_shape) - len(input_shape), len(candidate_axes) + 1):
            for axes_tuple in itertools.combinations(candidate_axes, remove_count):
                axes = list(axes_tuple)
                if (
                    tuple(dim for axis, dim in enumerate(output_shape) if axis not in axes)
                    == input_shape
                ):
                    return ([], axes)

    if len(input_shape) > len(output_shape):
        candidate_axes = [axis for axis, dim in enumerate(input_shape) if dim == 1]
        for remove_count in range(len(input_shape) - len(output_shape), len(candidate_axes) + 1):
            for axes_tuple in itertools.combinations(candidate_axes, remove_count):
                axes = list(axes_tuple)
                if (
                    tuple(dim for axis, dim in enumerate(input_shape) if axis not in axes)
                    == output_shape
                ):
                    return (axes, [])

    if [dim for dim in input_shape if dim != 1] != [dim for dim in output_shape if dim != 1]:
        return None

    squeeze_axes = [axis for axis, dim in enumerate(input_shape) if dim == 1]
    squeezed_shape = tuple(dim for dim in input_shape if dim != 1)
    unsqueeze_axes: list[int] = []
    squeezed_index = 0
    for axis, dim in enumerate(output_shape):
        if dim == 1:
            unsqueeze_axes.append(axis)
        else:
            if squeezed_index >= len(squeezed_shape) or squeezed_shape[squeezed_index] != dim:
                return None
            squeezed_index += 1
    if squeezed_index != len(squeezed_shape):
        return None
    return squeeze_axes, unsqueeze_axes


def rewrite_singleton_reshapes_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    inferred = onnx.shape_inference.infer_shapes(model)
    value_shapes = {
        value.name: _tensor_shape(value)
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        )
    }

    rewritten_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    rewrites = Counter()
    rewrite_examples: list[dict[str, Any]] = []

    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 1 or len(node.output) != 1:
            rewritten_nodes.append(node)
            continue

        input_shape = value_shapes.get(node.input[0])
        output_shape = value_shapes.get(node.output[0])
        if input_shape is None or output_shape is None:
            rewritten_nodes.append(node)
            continue

        plan = _singleton_layout_plan(input_shape, output_shape)
        if plan is None:
            rewritten_nodes.append(node)
            continue

        squeeze_axes, unsqueeze_axes = plan
        if not squeeze_axes and not unsqueeze_axes:
            rewritten_nodes.append(node)
            continue

        current_name = node.input[0]
        if squeeze_axes:
            axes_name = f"{node.name or node.output[0]}__squeeze_axes"
            new_initializers.append(
                onnx.numpy_helper.from_array(np.asarray(squeeze_axes, dtype=np.int64), axes_name)
            )
            squeeze_output = node.output[0] if not unsqueeze_axes else f"{node.output[0]}__squeezed"
            rewritten_nodes.append(
                onnx.helper.make_node(
                    "Squeeze",
                    [current_name, axes_name],
                    [squeeze_output],
                    name=f"{node.name or node.output[0]}__squeeze",
                )
            )
            current_name = squeeze_output
        if unsqueeze_axes:
            axes_name = f"{node.name or node.output[0]}__unsqueeze_axes"
            new_initializers.append(
                onnx.numpy_helper.from_array(np.asarray(unsqueeze_axes, dtype=np.int64), axes_name)
            )
            rewritten_nodes.append(
                onnx.helper.make_node(
                    "Unsqueeze",
                    [current_name, axes_name],
                    [node.output[0]],
                    name=f"{node.name or node.output[0]}__unsqueeze",
                )
            )

        if squeeze_axes and unsqueeze_axes:
            rewrites["squeeze_unsqueeze"] += 1
        elif squeeze_axes:
            rewrites["squeeze"] += 1
        else:
            rewrites["unsqueeze"] += 1
        if len(rewrite_examples) < 12:
            rewrite_examples.append(
                {
                    "node": node.name,
                    "input_shape": list(input_shape),
                    "output_shape": list(output_shape),
                    "squeeze_axes": squeeze_axes,
                    "unsqueeze_axes": unsqueeze_axes,
                }
            )

    tracked_ops = (
        "Reshape",
        "Squeeze",
        "Unsqueeze",
        "Concat",
        "Expand",
        "Transpose",
        "Einsum",
        "Gemm",
    )
    if not rewrites:
        return {
            "enabled": True,
            "tool": "custom_singleton_reshape_rewrite",
            "reason": "No singleton-only Reshape nodes found.",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
            "tracked_ops_after": {op: int(before.get(op, 0)) for op in tracked_ops},
            "rewrites": {},
            "rewrite_examples": [],
        }

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(new_initializers)
    onnx.checker.check_model(model)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    return {
        "enabled": True,
        "tool": "custom_singleton_reshape_rewrite",
        "reason": (
            "Replace behavior-preserving singleton-only Reshape nodes with "
            "Squeeze/Unsqueeze so ORT WebGPU can keep those layout views on device."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": rewrite_examples,
    }


def rewrite_gqa_repeats_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    inferred = onnx.shape_inference.infer_shapes(model)
    value_shapes = {
        value.name: _tensor_shape(value)
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        )
    }

    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    producer = {output: node for node in model.graph.node for output in node.output}

    skip_nodes: set[str] = set()
    replacements: dict[str, list[onnx.NodeProto]] = {}
    new_initializers: list[onnx.TensorProto] = []
    rewrites = Counter()
    rewrite_examples: list[dict[str, Any]] = []

    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.output) != 1:
            continue
        input_shape = value_shapes.get(node.input[0])
        output_shape = value_shapes.get(node.output[0])
        if input_shape is None or output_shape is None:
            continue
        if len(input_shape) < 4 or len(input_shape) != len(output_shape) + 1:
            continue
        if input_shape[-3:] != (2, 4, 64) or output_shape[-2:] != (8, 64):
            continue
        if input_shape[:-3] != output_shape[:-2]:
            continue

        expand = producer.get(node.input[0])
        if expand is None or expand.op_type != "Expand":
            continue
        if len(consumers.get(expand.output[0], [])) != 1:
            continue

        source_producer = producer.get(expand.input[0])
        if source_producer is None or len(consumers.get(source_producer.output[0], [])) != 1:
            continue
        source_shape = value_shapes.get(source_producer.input[0])
        source_name = source_producer.input[0]
        replacement_nodes: list[onnx.NodeProto] = []

        if source_producer.op_type == "Unsqueeze":
            if (
                source_shape is None
                or source_shape[:-2] != output_shape[:-2]
                or source_shape[-2:] != (2, 64)
            ):
                continue
            skip_nodes.add(source_producer.name)
            rewrites["from_unsqueeze"] += 1
        elif source_producer.op_type == "Reshape":
            if source_shape is None or source_shape[-1] != 128:
                continue
            compact_shape = tuple(output_shape[:-2]) + (2, 64)
            if int(np.prod(source_shape)) != int(np.prod(compact_shape)):
                continue
            compact_shape_name = f"{source_producer.name or node.name}__gqa_compact_shape"
            compact_output = f"{source_producer.output[0]}__gqa_compact"
            new_initializers.append(
                onnx.numpy_helper.from_array(
                    np.asarray(compact_shape, dtype=np.int64), compact_shape_name
                )
            )
            replacement_nodes.append(
                onnx.helper.make_node(
                    "Reshape",
                    [source_name, compact_shape_name],
                    [compact_output],
                    name=f"{source_producer.name or node.name}__gqa_compact",
                )
            )
            source_name = compact_output
            skip_nodes.add(source_producer.name)
            rewrites["from_reshape"] += 1
        else:
            continue

        indices_name = f"{node.name or node.output[0]}__gqa_indices"
        new_initializers.append(
            onnx.numpy_helper.from_array(
                np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64), indices_name
            )
        )
        replacement_nodes.append(
            onnx.helper.make_node(
                "Gather",
                [source_name, indices_name],
                [node.output[0]],
                name=f"{node.name or node.output[0]}__gqa_gather",
                axis=len(output_shape) - 2,
            )
        )
        replacements[node.name] = replacement_nodes
        skip_nodes.add(expand.name)
        if len(rewrite_examples) < 12:
            rewrite_examples.append(
                {
                    "node": node.name,
                    "source": source_producer.op_type,
                    "input_shape": list(input_shape),
                    "output_shape": list(output_shape),
                    "gather_axis": len(output_shape) - 2,
                }
            )

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_gqa_repeat_rewrite",
            "rewrites": {},
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "tracked_ops_before": {},
            "tracked_ops_after": {},
            "rewrite_examples": [],
        }

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in skip_nodes:
            continue
        if node.name in replacements:
            rewritten_nodes.extend(replacements[node.name])
        else:
            rewritten_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(new_initializers)
    onnx.checker.check_model(model)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Reshape", "Unsqueeze", "Expand", "Gather", "Einsum", "Gemm")
    return {
        "enabled": True,
        "tool": "custom_gqa_repeat_rewrite",
        "reason": (
            "Replace GQA K/V head repeat materialization with Gather over the KV-head "
            "axis. This removes the WebGPU->CPU Reshape boundary between Expand and "
            "Einsum without changing repeat ordering."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": rewrite_examples,
    }


def _gemm_attrs(node: onnx.NodeProto) -> dict[str, Any]:
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    return {
        "alpha": float(attrs.get("alpha", 1.0)),
        "beta": float(attrs.get("beta", 1.0)),
        "transA": int(attrs.get("transA", 0)),
        "transB": int(attrs.get("transB", 0)),
    }


def rewrite_head_projection_reshapes_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    inferred = onnx.shape_inference.infer_shapes(model)
    value_shapes = {
        value.name: _tensor_shape(value)
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        )
    }
    initializers = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    initializer_names = set(initializers)
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    producer = {output: node for node in model.graph.node for output in node.output}

    replacements: dict[str, list[onnx.NodeProto]] = {}
    skip_nodes: set[str] = set()
    new_initializers: list[onnx.TensorProto] = []
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    def can_rewrite_gemm(gemm: onnx.NodeProto) -> bool:
        if gemm.op_type != "Gemm" or len(gemm.input) != 2:
            return False
        attrs = _gemm_attrs(gemm)
        return attrs == {"alpha": 1.0, "beta": 0.0, "transA": 0, "transB": 0}

    for reshape in model.graph.node:
        if reshape.op_type != "Reshape" or len(reshape.output) != 1:
            continue
        input_shape = value_shapes.get(reshape.input[0])
        output_shape = value_shapes.get(reshape.output[0])
        if input_shape is None or output_shape is None:
            continue

        # Pattern 1: Gemm([N,K], W[K,H*D]) -> head-shaped output with
        # optional singleton wrappers, e.g. [1,N,H,D], [N,1,H,D],
        # or [1,N,1,H,D].
        gemm = producer.get(reshape.input[0])
        prefix_shape = output_shape[:-2]
        if (
            gemm is not None
            and can_rewrite_gemm(gemm)
            and len(input_shape) == 2
            and len(output_shape) in (4, 5, 6)
            and output_shape[-1] == 64
            and output_shape[-2] in (2, 8)
            and prefix_shape.count(input_shape[0]) == 1
            and all(dim in (1, input_shape[0]) for dim in prefix_shape)
        ):
            source_shape = value_shapes.get(gemm.input[0])
            weight = initializers.get(gemm.input[1])
            if (
                source_shape is not None
                and len(source_shape) == 2
                and weight is not None
                and weight.shape == (source_shape[-1], output_shape[-2] * output_shape[-1])
                and source_shape[0] in prefix_shape
            ):
                squeezed_shape = tuple(dim for dim in output_shape if dim != 1)
                if squeezed_shape == (source_shape[0], output_shape[-2], output_shape[-1]):
                    weight_name = f"{gemm.name or reshape.name}__head_weight"
                    einsum_output = (
                        reshape.output[0]
                        if output_shape[0] != 1 and output_shape[1] != 1
                        else f"{reshape.output[0]}__head_projected"
                    )
                    new_initializers.append(
                        onnx.numpy_helper.from_array(
                            weight.reshape(source_shape[-1], output_shape[-2], output_shape[-1]),
                            weight_name,
                        )
                    )
                    replacement_nodes = [
                        onnx.helper.make_node(
                            "Einsum",
                            [gemm.input[0], weight_name],
                            [einsum_output],
                            name=f"{gemm.name or reshape.name}__head_project",
                            equation="nk,khd->nhd",
                        )
                    ]
                    if einsum_output != reshape.output[0]:
                        axes = [axis for axis, dim in enumerate(output_shape) if dim == 1]
                        axes_name = f"{reshape.name or reshape.output[0]}__head_unsqueeze_axes"
                        new_initializers.append(
                            onnx.numpy_helper.from_array(
                                np.asarray(axes, dtype=np.int64), axes_name
                            )
                        )
                        replacement_nodes.append(
                            onnx.helper.make_node(
                                "Unsqueeze",
                                [einsum_output, axes_name],
                                [reshape.output[0]],
                                name=f"{reshape.name or reshape.output[0]}__head_unsqueeze",
                            )
                        )
                    replacements[reshape.name] = replacement_nodes
                    skip_nodes.add(gemm.name)
                    rewrites["gemm_to_head"] += 1
                    if len(examples) < 12:
                        examples.append(
                            {
                                "kind": "gemm_to_head",
                                "gemm": gemm.name,
                                "reshape": reshape.name,
                                "input_shape": list(source_shape),
                                "output_shape": list(output_shape),
                            }
                        )
                    continue

        # Pattern 2: Reshape([1,N,H,D], [N,H*D]) -> Gemm(W[H*D,M]).
        gemm_consumer = _single_consumer(consumers, reshape.output[0], "Gemm")
        if (
            gemm_consumer is not None
            and can_rewrite_gemm(gemm_consumer)
            and len(input_shape) == 4
            and len(output_shape) == 2
            and input_shape[:2].count(1) == 1
            and input_shape[-2:] == (8, 64)
            and output_shape
            == (max(input_shape[0], input_shape[1]), input_shape[2] * input_shape[3])
        ):
            weight = initializers.get(gemm_consumer.input[1])
            if weight is not None and weight.shape[0] == output_shape[-1]:
                weight_name = f"{gemm_consumer.name or reshape.name}__merge_weight"
                equation = "bnhd,hdm->nm" if input_shape[0] == 1 else "nthd,hdm->nm"
                new_initializers.append(
                    onnx.numpy_helper.from_array(
                        weight.reshape(input_shape[2], input_shape[3], weight.shape[1]),
                        weight_name,
                    )
                )
                replacements[reshape.name] = [
                    onnx.helper.make_node(
                        "Einsum",
                        [reshape.input[0], weight_name],
                        [gemm_consumer.output[0]],
                        name=f"{gemm_consumer.name or reshape.name}__head_merge",
                        equation=equation,
                    )
                ]
                skip_nodes.add(gemm_consumer.name)
                rewrites["head_to_gemm"] += 1
                if len(examples) < 12:
                    examples.append(
                        {
                            "kind": "head_to_gemm",
                            "reshape": reshape.name,
                            "gemm": gemm_consumer.name,
                            "input_shape": list(input_shape),
                            "output_shape": list(gemm_consumer.output),
                        }
                    )

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_head_projection_reshape_rewrite",
            "rewrites": {},
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "tracked_ops_before": {},
            "tracked_ops_after": {},
            "rewrite_examples": [],
        }

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in skip_nodes:
            continue
        if node.name in replacements:
            rewritten_nodes.extend(replacements[node.name])
        else:
            rewritten_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(new_initializers)
    retained_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in initializer_names
        or any(initializer.name in node.input for node in model.graph.node)
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)
    onnx.checker.check_model(model)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Reshape", "Unsqueeze", "Einsum", "Gemm", "Gather", "Memcpy")
    return {
        "enabled": True,
        "tool": "custom_head_projection_reshape_rewrite",
        "reason": (
            "Replace attention head split/merge Gemm+Reshape patterns with rank-aware "
            "Einsum forms so ORT WebGPU does not cross to CPU for standalone Reshape."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
    }


def rewrite_head_projection_reshapes_with_layout_ops_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    inferred = onnx.shape_inference.infer_shapes(model)
    value_shapes = {
        value.name: _tensor_shape(value)
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        )
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    producer = {output: node for node in model.graph.node for output in node.output}

    replacements: dict[str, list[onnx.NodeProto]] = {}
    new_initializers: list[onnx.TensorProto] = []
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    def can_rewrite_gemm(gemm: onnx.NodeProto) -> bool:
        if gemm.op_type != "Gemm" or len(gemm.input) != 2:
            return False
        attrs = _gemm_attrs(gemm)
        return attrs == {"alpha": 1.0, "beta": 0.0, "transA": 0, "transB": 0}

    def add_i64_initializer(name: str, values: list[int]) -> str:
        new_initializers.append(
            onnx.numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)
        )
        return name

    def singleton_axes_for_head_shape(shape: tuple[int, ...], head_axis: int) -> list[int]:
        axes = [axis for axis, dim in enumerate(shape) if dim == 1]
        axes.append(head_axis)
        return sorted(set(axes))

    for reshape in model.graph.node:
        if reshape.op_type != "Reshape" or len(reshape.output) != 1:
            continue
        input_shape = value_shapes.get(reshape.input[0])
        output_shape = value_shapes.get(reshape.output[0])
        if input_shape is None or output_shape is None:
            continue

        # Pattern 1: keep Gemm([N,K], W[K,H*D]) and replace the head-view
        # Reshape with static split/unsqueeze/concat layout ops.
        gemm = producer.get(reshape.input[0])
        prefix_shape = output_shape[:-2]
        if (
            gemm is not None
            and can_rewrite_gemm(gemm)
            and len(input_shape) == 2
            and len(output_shape) in (4, 5, 6)
            and output_shape[-1] == 64
            and output_shape[-2] in (2, 8)
            and input_shape[-1] == output_shape[-2] * output_shape[-1]
            and prefix_shape.count(input_shape[0]) == 1
            and all(dim in (1, input_shape[0]) for dim in prefix_shape)
        ):
            head_count = output_shape[-2]
            head_dim = output_shape[-1]
            head_axis = len(output_shape) - 2
            split_outputs = [
                f"{reshape.output[0]}__flat_head_{head_idx}"
                for head_idx in range(head_count)
            ]
            split_sizes = add_i64_initializer(
                f"{reshape.name or reshape.output[0]}__split_sizes",
                [head_dim] * head_count,
            )
            replacement_nodes = [
                onnx.helper.make_node(
                    "Split",
                    [reshape.input[0], split_sizes],
                    split_outputs,
                    name=f"{reshape.name or reshape.output[0]}__head_split",
                    axis=1,
                )
            ]
            unsqueezed_outputs = []
            axes = singleton_axes_for_head_shape(output_shape, head_axis)
            axes_name = add_i64_initializer(
                f"{reshape.name or reshape.output[0]}__head_unsqueeze_axes",
                axes,
            )
            for head_idx, split_output in enumerate(split_outputs):
                unsqueezed = f"{reshape.output[0]}__head_{head_idx}"
                unsqueezed_outputs.append(unsqueezed)
                replacement_nodes.append(
                    onnx.helper.make_node(
                        "Unsqueeze",
                        [split_output, axes_name],
                        [unsqueezed],
                        name=f"{reshape.name or reshape.output[0]}__head_{head_idx}_unsqueeze",
                    )
                )
            replacement_nodes.append(
                onnx.helper.make_node(
                    "Concat",
                    unsqueezed_outputs,
                    [reshape.output[0]],
                    name=f"{reshape.name or reshape.output[0]}__head_concat",
                    axis=head_axis,
                )
            )
            replacements[reshape.name] = replacement_nodes
            rewrites["gemm_to_head_layout"] += 1
            if len(examples) < 12:
                examples.append(
                    {
                        "kind": "gemm_to_head_layout",
                        "gemm": gemm.name,
                        "reshape": reshape.name,
                        "input_shape": list(input_shape),
                        "output_shape": list(output_shape),
                    }
                )
            continue

        # Pattern 2: keep the output projection Gemm and replace the flattening
        # Reshape([singleton..., N, H, D] -> [N, H*D]) with static layout ops.
        gemm_consumer = _single_consumer(consumers, reshape.output[0], "Gemm")
        if (
            gemm_consumer is not None
            and can_rewrite_gemm(gemm_consumer)
            and len(input_shape) in (4, 5, 6)
            and len(output_shape) == 2
            and input_shape[-2:] == (8, 64)
            and input_shape[:-2].count(output_shape[0]) == 1
            and all(dim in (1, output_shape[0]) for dim in input_shape[:-2])
            and output_shape[1] == input_shape[-2] * input_shape[-1]
        ):
            head_count = input_shape[-2]
            head_dim = input_shape[-1]
            head_axis = len(input_shape) - 2
            split_outputs = [
                f"{reshape.output[0]}__ranked_head_{head_idx}"
                for head_idx in range(head_count)
            ]
            split_sizes = add_i64_initializer(
                f"{reshape.name or reshape.output[0]}__merge_split_sizes",
                [1] * head_count,
            )
            replacement_nodes = [
                onnx.helper.make_node(
                    "Split",
                    [reshape.input[0], split_sizes],
                    split_outputs,
                    name=f"{reshape.name or reshape.output[0]}__merge_split",
                    axis=head_axis,
                )
            ]
            squeezed_outputs = []
            axes = singleton_axes_for_head_shape(input_shape, head_axis)
            axes_name = add_i64_initializer(
                f"{reshape.name or reshape.output[0]}__merge_squeeze_axes",
                axes,
            )
            for head_idx, split_output in enumerate(split_outputs):
                squeezed = f"{reshape.output[0]}__flat_head_{head_idx}"
                squeezed_outputs.append(squeezed)
                replacement_nodes.append(
                    onnx.helper.make_node(
                        "Squeeze",
                        [split_output, axes_name],
                        [squeezed],
                        name=f"{reshape.name or reshape.output[0]}__head_{head_idx}_squeeze",
                    )
                )
            replacement_nodes.append(
                onnx.helper.make_node(
                    "Concat",
                    squeezed_outputs,
                    [reshape.output[0]],
                    name=f"{reshape.name or reshape.output[0]}__merge_concat",
                    axis=1,
                )
            )
            replacements[reshape.name] = replacement_nodes
            rewrites["head_to_gemm_layout"] += 1
            if len(examples) < 12:
                examples.append(
                    {
                        "kind": "head_to_gemm_layout",
                        "reshape": reshape.name,
                        "gemm": gemm_consumer.name,
                        "input_shape": list(input_shape),
                        "output_shape": list(output_shape),
                    }
                )

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_head_projection_layout_rewrite",
            "rewrites": {},
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "tracked_ops_before": {},
            "tracked_ops_after": {},
            "rewrite_examples": [],
        }

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in replacements:
            rewritten_nodes.extend(replacements[node.name])
        else:
            rewritten_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(new_initializers)
    onnx.checker.check_model(model)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = (
        "Reshape",
        "Split",
        "Squeeze",
        "Unsqueeze",
        "Concat",
        "Einsum",
        "Gemm",
        "Gather",
    )
    return {
        "enabled": True,
        "tool": "custom_head_projection_layout_rewrite",
        "reason": (
            "Replace attention head split/merge Reshape nodes with static layout ops "
            "while preserving the original Gemm kernels."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
    }


def rewrite_rmsnorm_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    initializers = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    producer = {output: node for node in model.graph.node for output in node.output}

    def constant_scalar(name: str) -> float | None:
        value = initializers.get(name)
        if value is None or value.size != 1:
            return None
        return float(value.reshape(()))

    def only_consumed_by(output_name: str, node: onnx.NodeProto) -> bool:
        return consumers.get(output_name, []) == [node]

    replacements: dict[str, list[onnx.NodeProto]] = {}
    skip_nodes: set[str] = set()
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    for scale_mul in model.graph.node:
        if scale_mul.op_type != "Mul" or len(scale_mul.input) != 2:
            continue

        div = None
        scale_input = None
        for input_name in scale_mul.input:
            candidate = producer.get(input_name)
            if candidate is not None and candidate.op_type == "Div":
                div = candidate
                scale_input = (
                    scale_mul.input[1] if input_name == scale_mul.input[0] else scale_mul.input[0]
                )
                break
        if div is None or scale_input is None or scale_input not in initializers:
            continue
        if not only_consumed_by(div.output[0], scale_mul):
            continue

        sqrt = producer.get(div.input[1]) if len(div.input) >= 2 else None
        root_input = div.input[0]
        if sqrt is None or sqrt.op_type != "Sqrt" or not only_consumed_by(sqrt.output[0], div):
            continue

        add = producer.get(sqrt.input[0])
        if add is None or add.op_type != "Add" or not only_consumed_by(add.output[0], sqrt):
            continue

        reduce_mean = None
        epsilon = None
        for input_name in add.input:
            candidate = producer.get(input_name)
            if candidate is not None and candidate.op_type == "ReduceMean":
                reduce_mean = candidate
            else:
                epsilon = constant_scalar(input_name)
        if reduce_mean is None or epsilon is None or epsilon <= 0:
            continue
        if not only_consumed_by(reduce_mean.output[0], add):
            continue

        square = producer.get(reduce_mean.input[0])
        if (
            square is None
            or square.op_type != "Mul"
            or len(square.input) != 2
            or square.input[0] != square.input[1]
            or square.input[0] != root_input
            or not only_consumed_by(square.output[0], reduce_mean)
        ):
            continue

        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in reduce_mean.attribute}
        if int(attrs.get("keepdims", 1)) != 1:
            continue
        axes = attrs.get("axes")
        if axes is None and len(reduce_mean.input) > 1:
            axes_value = initializers.get(reduce_mean.input[1])
            if axes_value is not None:
                axes = axes_value.astype(np.int64).reshape(-1).tolist()
        if axes is None or len(axes) != 1:
            continue

        replacements[scale_mul.name] = onnx.helper.make_node(
            "SimplifiedLayerNormalization",
            [root_input, scale_input],
            list(scale_mul.output),
            name=f"{scale_mul.name or scale_mul.output[0]}__rmsnorm",
            epsilon=float(epsilon),
            axis=int(axes[0]),
            stash_type=1,
        )
        skip_nodes.update({square.name, reduce_mean.name, add.name, sqrt.name, div.name})
        rewrites["rmsnorm"] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "scale_mul": scale_mul.name,
                    "root_input": root_input,
                    "scale_input": scale_input,
                    "axis": int(axes[0]),
                    "epsilon": float(epsilon),
                }
            )

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_rmsnorm_rewrite",
            "rewrites": {},
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "tracked_ops_before": {},
            "tracked_ops_after": {},
            "rewrite_examples": [],
        }

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in skip_nodes:
            continue
        rewritten_nodes.append(replacements.get(node.name, node))

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    external_data_path(path).unlink(missing_ok=True)
    # ONNX checker does not know ORT's SimplifiedLayerNormalization schema, but
    # ONNX Runtime WebGPU and ORT CPU both load it. Validation below uses ORT.
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = (
        "SimplifiedLayerNormalization",
        "ReduceMean",
        "Sqrt",
        "Div",
        "Mul",
        "Add",
    )
    return {
        "enabled": True,
        "tool": "custom_rmsnorm_rewrite",
        "reason": "Fuse decomposed RMSNorm arithmetic into ORT SimplifiedLayerNormalization for WebGPU.",
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
    }


def fuse_skip_simplified_layer_norm_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    value_shapes = {
        value.name: _tensor_shape(value)
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        )
    }
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    producer = {output: node for node in model.graph.node for output in node.output}

    gamma_aliases: dict[str, str] = {}
    new_initializers: list[onnx.TensorProto] = []
    replacements: dict[str, onnx.NodeProto] = {}
    skip_nodes: set[str] = set()
    examples: list[dict[str, Any]] = []

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def one_dim_gamma_name(name: str, width: int) -> str | None:
        if name in gamma_aliases:
            return gamma_aliases[name]
        initializer = initializers.get(name)
        if initializer is None:
            return None
        value = onnx.numpy_helper.to_array(initializer)
        if value.size != width:
            return None
        if value.ndim == 1:
            gamma_aliases[name] = name
            return name
        alias = f"{name}__skip_sln_1d"
        if alias not in initializers and alias not in gamma_aliases.values():
            new_initializers.append(
                onnx.numpy_helper.from_array(value.reshape((width,)).astype(value.dtype), alias)
            )
        gamma_aliases[name] = alias
        return alias

    for sln in model.graph.node:
        if sln.op_type != "SimplifiedLayerNormalization" or len(sln.input) < 2:
            continue
        add = producer.get(sln.input[0])
        if add is None or add.op_type != "Add" or len(add.input) != 2 or len(add.output) != 1:
            continue

        input_shape = value_shapes.get(sln.input[0])
        if input_shape is None or len(input_shape) not in {2, 3}:
            continue
        axis = int(attr_value(sln, "axis", -1))
        if axis < 0:
            axis += len(input_shape)
        if axis < 0 or axis >= len(input_shape):
            continue
        width = int(np.prod(input_shape[axis:]))
        gamma_name = one_dim_gamma_name(sln.input[1], width)
        if gamma_name is None:
            continue

        epsilon = float(attr_value(sln, "epsilon", 1.0e-6))
        replacements[sln.name] = onnx.helper.make_node(
            "SkipSimplifiedLayerNormalization",
            [add.input[0], add.input[1], gamma_name],
            [sln.output[0], "", "", add.output[0]],
            name=f"{sln.name or sln.output[0]}__skip_sln",
            domain="com.microsoft",
            epsilon=epsilon,
        )
        skip_nodes.add(add.name)
        if len(examples) < 12:
            examples.append(
                {
                    "add": add.name,
                    "simplified_layer_norm": sln.name,
                    "axis": axis,
                    "width": width,
                    "gamma": gamma_name,
                }
            )

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_skip_simplified_layer_norm_rewrite",
            "rewrites": 0,
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "tracked_ops_before": {},
            "tracked_ops_after": {},
            "rewrite_examples": [],
        }

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in skip_nodes:
            continue
        rewritten_nodes.append(replacements.get(node.name, node))

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(new_initializers)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = (
        "Add",
        "SimplifiedLayerNormalization",
        "SkipSimplifiedLayerNormalization",
    )
    return {
        "enabled": True,
        "tool": "custom_skip_simplified_layer_norm_rewrite",
        "reason": (
            "Fuse residual Add followed by SimplifiedLayerNormalization into "
            "ORT WebGPU's SkipSimplifiedLayerNormalization."
        ),
        "rewrites": len(replacements),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrite_examples": examples,
    }


def rewrite_rotary_embedding_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    inferred = onnx.shape_inference.infer_shapes(model)
    value_shapes = {
        value.name: _tensor_shape(value)
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        )
    }
    initializers = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    initializer_names = set(initializers)
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def single_consumer(value_name: str, op_type: str | None = None) -> onnx.NodeProto | None:
        value_consumers = consumers.get(value_name, [])
        if len(value_consumers) != 1:
            return None
        node = value_consumers[0]
        if op_type is not None and node.op_type != op_type:
            return None
        return node

    def split_mul_map(split_output: str) -> dict[str, onnx.NodeProto] | None:
        by_const: dict[str, onnx.NodeProto] = {}
        for mul in consumers.get(split_output, []):
            if mul.op_type != "Mul" or len(mul.input) != 2 or len(mul.output) != 1:
                return None
            if mul.input[0] == split_output:
                const_name = mul.input[1]
            elif mul.input[1] == split_output:
                const_name = mul.input[0]
            else:
                return None
            if const_name not in initializers or const_name in by_const:
                return None
            by_const[const_name] = mul
        if len(by_const) != 2:
            return None
        return by_const

    def add_cache_initializer(const_name: str, sequence_length: int, half_dim: int) -> str | None:
        alias = f"{const_name}__rotary_cache_2d_s{sequence_length}_d{half_dim}"
        if alias in initializer_names:
            return alias
        value = initializers.get(const_name)
        if value is None or value.size != sequence_length * half_dim:
            return None
        cache = value.reshape((sequence_length, half_dim)).astype(value.dtype, copy=False)
        model.graph.initializer.append(onnx.numpy_helper.from_array(cache, alias))
        initializer_names.add(alias)
        return alias

    position_ids_name = "rotary_position_zero_i64"
    if position_ids_name not in initializer_names:
        model.graph.initializer.append(
            onnx.numpy_helper.from_array(np.asarray(0, dtype=np.int64), position_ids_name)
        )
        initializer_names.add(position_ids_name)

    replacements: dict[str, onnx.NodeProto] = {}
    skip_nodes: set[str] = set()
    removed_value_info: set[str] = set()
    examples: list[dict[str, Any]] = []

    for split in model.graph.node:
        if split.op_type != "Split" or len(split.input) < 1 or len(split.output) != 2:
            continue
        input_shape = value_shapes.get(split.input[0])
        left_shape = value_shapes.get(split.output[0])
        right_shape = value_shapes.get(split.output[1])
        if (
            input_shape is None
            or len(input_shape) != 4
            or left_shape is None
            or right_shape is None
            or left_shape != right_shape
            or input_shape[-1] != left_shape[-1] * 2
            or input_shape[:-1] != left_shape[:-1]
        ):
            continue
        axis = int(attr_value(split, "axis", -1))
        if axis < 0:
            axis += len(input_shape)
        if axis != len(input_shape) - 1:
            continue

        left_muls = split_mul_map(split.output[0])
        right_muls = split_mul_map(split.output[1])
        if left_muls is None or right_muls is None:
            continue
        const_names = set(left_muls) & set(right_muls)
        if len(const_names) != 2:
            continue

        sub_node = None
        add_node = None
        cos_const = None
        sin_const = None
        for left_const in const_names:
            for right_const in const_names - {left_const}:
                left_cos = left_muls[left_const]
                right_sin = right_muls[right_const]
                sub = single_consumer(left_cos.output[0], "Sub")
                if (
                    sub is not None
                    and sub.input[0] == left_cos.output[0]
                    and sub.input[1] == right_sin.output[0]
                    and single_consumer(right_sin.output[0], "Sub") is sub
                ):
                    right_cos = right_muls[left_const]
                    left_sin = left_muls[right_const]
                    add = single_consumer(right_cos.output[0], "Add")
                    if (
                        add is not None
                        and set(add.input) == {right_cos.output[0], left_sin.output[0]}
                        and single_consumer(left_sin.output[0], "Add") is add
                    ):
                        sub_node = sub
                        add_node = add
                        cos_const = left_const
                        sin_const = right_const
                        break
            if sub_node is not None:
                break
        if sub_node is None or add_node is None or cos_const is None or sin_const is None:
            continue

        concat = single_consumer(sub_node.output[0], "Concat")
        if (
            concat is None
            or single_consumer(add_node.output[0], "Concat") is not concat
            or list(concat.input) != [sub_node.output[0], add_node.output[0]]
            or int(attr_value(concat, "axis", axis)) != axis
            or len(concat.output) != 1
            or value_shapes.get(concat.output[0]) != input_shape
        ):
            continue

        half_dim = int(left_shape[-1])
        direct_sequence_length = int(input_shape[-2])
        direct_num_heads = int(input_shape[-3])
        transposed_sequence_length = int(input_shape[-3])
        transposed_num_heads = int(input_shape[-2])
        cos_cache = add_cache_initializer(cos_const, direct_sequence_length, half_dim)
        sin_cache = add_cache_initializer(sin_const, direct_sequence_length, half_dim)
        use_transpose = False
        sequence_length = direct_sequence_length
        num_heads = direct_num_heads
        if cos_cache is None or sin_cache is None:
            cos_cache = add_cache_initializer(
                cos_const, transposed_sequence_length, half_dim
            )
            sin_cache = add_cache_initializer(
                sin_const, transposed_sequence_length, half_dim
            )
            use_transpose = True
            sequence_length = transposed_sequence_length
            num_heads = transposed_num_heads
        if cos_cache is None or sin_cache is None:
            continue

        rotary_input = split.input[0]
        rotary_output = concat.output[0]
        replacement_nodes = []
        if use_transpose:
            # ORT WebGPU's contrib RotaryEmbedding interprets rank-4 input as
            # [batch, heads, sequence, head_dim]. The exported RoPE islands use
            # [batch, sequence, heads, head_dim], so wrap the fused op with two
            # GPU-supported transposes.
            rotary_input = f"{split.input[0]}__rotary_bhsd"
            rotary_output = f"{concat.output[0]}__rotary_bhsd"
            replacement_nodes.append(
                onnx.helper.make_node(
                    "Transpose",
                    [split.input[0]],
                    [rotary_input],
                    name=f"{split.name}__rotary_to_bhsd",
                    perm=[0, 2, 1, 3],
                )
            )

        replacement_nodes.append(
            onnx.helper.make_node(
            "RotaryEmbedding",
            [rotary_input, position_ids_name, cos_cache, sin_cache],
            [rotary_output],
            name=f"{concat.name or concat.output[0]}__rotary_embedding",
            domain="com.microsoft",
            interleaved=0,
            num_heads=num_heads,
            rotary_embedding_dim=0,
            scale=1.0,
            )
        )
        if use_transpose:
            replacement_nodes.append(
                onnx.helper.make_node(
                    "Transpose",
                    [rotary_output],
                    [concat.output[0]],
                    name=f"{concat.name or concat.output[0]}__rotary_from_bhsd",
                    perm=[0, 2, 1, 3],
                )
            )
        replacements[split.name] = replacement_nodes
        matched_nodes = {
            split.name,
            concat.name,
            sub_node.name,
            add_node.name,
            *(node.name for node in left_muls.values()),
            *(node.name for node in right_muls.values()),
        }
        skip_nodes.update(matched_nodes - {split.name})
        removed_value_info.update(split.output)
        removed_value_info.update(node.output[0] for node in left_muls.values())
        removed_value_info.update(node.output[0] for node in right_muls.values())
        removed_value_info.update(sub_node.output)
        removed_value_info.update(add_node.output)
        if len(examples) < 12:
            examples.append(
                {
                    "split": split.name,
                    "output": concat.output[0],
                    "input_shape": list(input_shape),
                    "layout": "bshd" if use_transpose else "bhsd",
                    "num_heads": num_heads,
                    "sequence_length": sequence_length,
                    "cos_cache": cos_cache,
                    "sin_cache": sin_cache,
                }
            )

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_rotary_embedding_rewrite",
            "reason": "No supported RoPE Split/Mul/Sub/Add/Concat islands found.",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": 0,
            "rewrite_examples": [],
        }

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in replacements:
            rewritten_nodes.extend(replacements[node.name])
        elif node.name not in skip_nodes:
            rewritten_nodes.append(node)

    retained_value_info = [
        value_info
        for value_info in model.graph.value_info
        if value_info.name not in removed_value_info
    ]
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("RotaryEmbedding", "Split", "Mul", "Sub", "Add", "Concat")
    return {
        "enabled": True,
        "tool": "custom_rotary_embedding_rewrite",
        "reason": (
            "Replace non-interleaved RoPE Split/Mul/Sub/Add/Concat islands with "
            "ORT WebGPU's contrib RotaryEmbedding op."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": len(replacements),
        "rewrite_examples": examples,
    }


def _single_consumer(
    consumers: dict[str, list[onnx.NodeProto]],
    value_name: str,
    op_type: str | None = None,
) -> onnx.NodeProto | None:
    value_consumers = consumers.get(value_name, [])
    if len(value_consumers) != 1:
        return None
    node = value_consumers[0]
    if op_type is not None and node.op_type != op_type:
        return None
    return node


def run_ort(path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    typed_feeds = dict(feeds)
    for input_spec in session.get_inputs():
        if input_spec.name not in typed_feeds:
            continue
        if input_spec.type == "tensor(float16)":
            typed_feeds[input_spec.name] = typed_feeds[input_spec.name].astype(np.float16)
        elif input_spec.type == "tensor(float)":
            typed_feeds[input_spec.name] = typed_feeds[input_spec.name].astype(np.float32)
    outputs = session.run(None, typed_feeds)
    return {output.name: value for output, value in zip(session.get_outputs(), outputs)}


def compare_arrays(
    expected: np.ndarray, actual: np.ndarray, *, atol: float, rtol: float
) -> dict[str, Any]:
    expected = expected.astype(np.float32) if expected.dtype == np.float16 else expected
    actual = actual.astype(np.float32) if actual.dtype == np.float16 else actual
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


def cache_contract(dyn_shapes, *, available: bool, dtype: str = "float32") -> dict[str, Any]:
    cache_shape = list(dyn_shapes.cache)
    layer_cache_shape = list(dyn_shapes.layer_cache)
    return {
        "status": "available" if available else "contract_only",
        "reason": None
        if available
        else (
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
            "k_cache": tensor_spec(dtype, tuple(cache_shape)),
            "v_cache": tensor_spec(dtype, tuple(cache_shape)),
            "layer_cache": {
                **tensor_spec(dtype, tuple(layer_cache_shape)),
                "layers": dyn_shapes.temporal_blocks,
            },
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
            "Use committed cache for attention on all sample iterations. Discard candidate "
            "cache for sample steps 0-2. Commit candidate cache only on sample step 3."
        ),
    }


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


def validate_outputs(
    *,
    path: Path,
    feeds: dict[str, jax.Array],
    expected: dict[str, jax.Array],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    ort_feeds = {name: np.asarray(jax.device_get(value)) for name, value in feeds.items()}
    actual = run_ort(path, ort_feeds)
    results = {
        name: compare_arrays(
            np.asarray(jax.device_get(expected_value)),
            actual[name],
            atol=atol,
            rtol=rtol,
        )
        for name, expected_value in expected.items()
    }
    return {
        "atol": atol,
        "rtol": rtol,
        "passed": all(result["passed"] for result in results.values()),
        "outputs": results,
    }


def main() -> None:
    args = parse_args()
    require_static_phase1_args(args)

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
    decoder_step_path = args.out_dir / f"{TOKENIZER_DECODER_STEP_NAME}.onnx"
    decoder_z_step_path = args.out_dir / f"{TOKENIZER_DECODE_Z_STEP_NAME}.onnx"
    dynamics_path = args.out_dir / f"{DYNAMICS_UNCACHED_NAME}.onnx"
    dynamics_prefill_path = args.out_dir / f"{DYNAMICS_CACHED_PREFILL_NAME}.onnx"
    dynamics_prefill_layer_path = args.out_dir / f"{DYNAMICS_CACHED_PREFILL_LAYER_NAME}.onnx"
    dynamics_step_path = args.out_dir / f"{DYNAMICS_CACHED_STEP_NAME}.onnx"
    dynamics_sample_step_path = args.out_dir / f"{DYNAMICS_CACHED_SAMPLE_STEP_NAME}.onnx"
    dynamics_sample_step_slide_path = (
        args.out_dir / f"{DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME}.onnx"
    )
    dynamics_sample_append_context_path = (
        args.out_dir / f"{DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME}.onnx"
    )
    dynamics_sample_append_context_slide_path = (
        args.out_dir / f"{DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME}.onnx"
    )
    dynamics_sample_append_context_slide_full_cache_path = (
        args.out_dir / f"{DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME}.onnx"
    )
    dynamics_sample_append_context_slide_entry_path = (
        args.out_dir / f"{DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME}.onnx"
    )
    dynamics_sample_append_context_slide_layer_path = (
        args.out_dir / f"{DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME}.onnx"
    )
    manifest_path = args.out_dir / MANIFEST_NAME
    ensure_output(manifest_path, overwrite=args.overwrite)

    def decoder_fn(latent: jax.Array) -> jax.Array:
        return apply_tokenizer_decoder(
            tokenizer_variables,
            tokenizer_cfg,
            latent,
        )

    def decoder_step_fn(latent: jax.Array) -> jax.Array:
        return apply_tokenizer_decoder(
            tokenizer_variables,
            tokenizer_cfg,
            latent,
        )

    def decoder_z_step_fn(z: jax.Array) -> jax.Array:
        return apply_tokenizer_decode_z(
            tokenizer_variables,
            tokenizer_cfg,
            z,
            num_obs_tokens=dyn_shapes.num_obs_tokens,
        )

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

    def dynamics_prefill_fn(
        z: jax.Array,
        actions: jax.Array,
        step_levels: jax.Array,
        signal_levels: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return apply_dynamics_cached_prefill(
            dynamics_variables,
            dynamics_cfg,
            z,
            actions,
            step_levels,
            signal_levels,
        )

    def dynamics_prefill_layer_fn(
        z: jax.Array,
        actions: jax.Array,
        step_levels: jax.Array,
        signal_levels: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...], jax.Array]:
        return apply_dynamics_cached_prefill_layer_cache(
            dynamics_variables,
            dynamics_cfg,
            z,
            actions,
            step_levels,
            signal_levels,
        )

    def dynamics_step_fn(
        z: jax.Array,
        actions: jax.Array,
        step_levels: jax.Array,
        signal_levels: jax.Array,
        position_index: jax.Array,
        k_cache: jax.Array,
        v_cache: jax.Array,
        cache_length: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return apply_dynamics_cached_step(
            dynamics_variables,
            dynamics_cfg,
            z,
            actions,
            step_levels,
            signal_levels,
            position_index,
            k_cache,
            v_cache,
            cache_length,
        )

    def dynamics_sample_step_fn(
        sample_noise: jax.Array,
        actions: jax.Array,
        position_index: jax.Array,
        k_cache: jax.Array,
        v_cache: jax.Array,
        cache_length: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return apply_dynamics_cached_sample_step(
            dynamics_variables,
            dynamics_cfg,
            sample_noise,
            actions,
            position_index,
            k_cache,
            v_cache,
            cache_length,
            sample_steps=args.sample_steps,
        )

    def dynamics_sample_step_slide_fn(
        sample_noise: jax.Array,
        actions: jax.Array,
        position_index: jax.Array,
        k_cache: jax.Array,
        v_cache: jax.Array,
        cache_length: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return apply_dynamics_cached_sample_step(
            dynamics_variables,
            dynamics_cfg,
            sample_noise,
            actions,
            position_index,
            k_cache,
            v_cache,
            cache_length,
            sample_steps=args.sample_steps,
            cache_update="slide",
        )

    def dynamics_sample_append_context_fn(
        sample_noise: jax.Array,
        context_noise: jax.Array,
        actions: jax.Array,
        position_index: jax.Array,
        k_cache: jax.Array,
        v_cache: jax.Array,
        cache_length: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return apply_dynamics_cached_sample_step_append_context(
            dynamics_variables,
            dynamics_cfg,
            sample_noise,
            context_noise,
            actions,
            position_index,
            k_cache,
            v_cache,
            cache_length,
            context_tau=args.context_tau,
            sample_steps=args.sample_steps,
        )

    def dynamics_sample_append_context_slide_fn(
        sample_noise: jax.Array,
        context_noise: jax.Array,
        actions: jax.Array,
        position_index: jax.Array,
        k_cache: jax.Array,
        v_cache: jax.Array,
        cache_length: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return apply_dynamics_cached_sample_step_append_context(
            dynamics_variables,
            dynamics_cfg,
            sample_noise,
            context_noise,
            actions,
            position_index,
            k_cache,
            v_cache,
            cache_length,
            context_tau=args.context_tau,
            sample_steps=args.sample_steps,
            cache_update="slide",
        )

    def dynamics_sample_append_context_slide_full_cache_fn(
        sample_noise: jax.Array,
        context_noise: jax.Array,
        actions: jax.Array,
        k_cache: jax.Array,
        v_cache: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return apply_dynamics_cached_sample_step_append_context_full_cache(
            dynamics_variables,
            dynamics_cfg,
            sample_noise,
            context_noise,
            actions,
            k_cache,
            v_cache,
            context_tau=args.context_tau,
            sample_steps=args.sample_steps,
        )

    def dynamics_sample_append_context_slide_entry_fn(
        sample_noise: jax.Array,
        context_noise: jax.Array,
        actions: jax.Array,
        k_cache: jax.Array,
        v_cache: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return apply_dynamics_cached_sample_step_append_context_full_cache_entries(
            dynamics_variables,
            dynamics_cfg,
            sample_noise,
            context_noise,
            actions,
            k_cache,
            v_cache,
            context_tau=args.context_tau,
            sample_steps=args.sample_steps,
        )

    def dynamics_sample_append_context_slide_layer_fn(
        sample_noise: jax.Array,
        context_noise: jax.Array,
        actions: jax.Array,
        position_index: jax.Array,
        k_cache_0: jax.Array,
        k_cache_1: jax.Array,
        k_cache_2: jax.Array,
        k_cache_3: jax.Array,
        k_cache_4: jax.Array,
        k_cache_5: jax.Array,
        v_cache_0: jax.Array,
        v_cache_1: jax.Array,
        v_cache_2: jax.Array,
        v_cache_3: jax.Array,
        v_cache_4: jax.Array,
        v_cache_5: jax.Array,
        cache_length: jax.Array,
    ) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...], jax.Array]:
        return apply_dynamics_cached_sample_step_append_context_layer_cache(
            dynamics_variables,
            dynamics_cfg,
            sample_noise,
            context_noise,
            actions,
            position_index,
            (k_cache_0, k_cache_1, k_cache_2, k_cache_3, k_cache_4, k_cache_5),
            (v_cache_0, v_cache_1, v_cache_2, v_cache_3, v_cache_4, v_cache_5),
            cache_length,
            context_tau=args.context_tau,
            sample_steps=args.sample_steps,
            cache_update="slide",
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

    cached_inputs: dict[str, jax.Array] = {}
    if args.export_cached:
        cached_inputs = {
            "latent_step": inputs["latent"][:, :1],
            "z_step": inputs["z"][:, :1],
            "actions_step": inputs["actions"][:, :1],
            "step_levels_step": jnp.full(dyn_shapes.step_levels, 2, dtype=jnp.int32),
            "signal_levels_step": jnp.zeros(dyn_shapes.step_levels, dtype=jnp.int32),
            "position_index": jnp.asarray([dyn_shapes.context_length], dtype=jnp.int32),
            "k_cache": jnp.zeros(dyn_shapes.cache, dtype=jnp.float32),
            "v_cache": jnp.zeros(dyn_shapes.cache, dtype=jnp.float32),
            "layer_cache": jnp.zeros(dyn_shapes.layer_cache, dtype=jnp.float32),
            "cache_length": jnp.asarray([dyn_shapes.context_length], dtype=jnp.int32),
        }
        layer_count = dyn_shapes.temporal_blocks
        if layer_count != 6:
            raise ValueError(
                "Layer-cache export currently uses an explicit 6-layer ONNX ABI; "
                f"got temporal_blocks={layer_count}."
            )
        k_layer_names = tuple(f"k_cache_{i}" for i in range(layer_count))
        v_layer_names = tuple(f"v_cache_{i}" for i in range(layer_count))
        candidate_k_layer_names = tuple(f"candidate_k_cache_{i}" for i in range(layer_count))
        candidate_v_layer_names = tuple(f"candidate_v_cache_{i}" for i in range(layer_count))
        k_layer_cache_inputs = tuple(
            jnp.zeros(dyn_shapes.layer_cache, dtype=jnp.float32) for _ in range(layer_count)
        )
        v_layer_cache_inputs = tuple(
            jnp.zeros(dyn_shapes.layer_cache, dtype=jnp.float32) for _ in range(layer_count)
        )
        export_to_onnx(
            fn=decoder_step_fn,
            inputs=(cached_inputs["latent_step"],),
            output_path=decoder_step_path,
            model_name=TOKENIZER_DECODER_STEP_NAME,
            opset=args.opset,
            input_names=("latent",),
            output_names=("patches",),
            overwrite=args.overwrite,
        )
        export_to_onnx(
            fn=decoder_z_step_fn,
            inputs=(cached_inputs["z_step"],),
            output_path=decoder_z_step_path,
            model_name=TOKENIZER_DECODE_Z_STEP_NAME,
            opset=args.opset,
            input_names=("z",),
            output_names=("patches",),
            overwrite=args.overwrite,
        )
        export_to_onnx(
            fn=dynamics_prefill_fn,
            inputs=(
                inputs["z"],
                inputs["actions"],
                inputs["step_levels"],
                inputs["signal_levels"],
            ),
            output_path=dynamics_prefill_path,
            model_name=DYNAMICS_CACHED_PREFILL_NAME,
            opset=args.opset,
            input_names=("z", "actions", "step_levels", "signal_levels"),
            output_names=("pred_z", "k_cache", "v_cache", "cache_length"),
            overwrite=args.overwrite,
        )
        export_to_onnx(
            fn=dynamics_prefill_layer_fn,
            inputs=(
                inputs["z"],
                inputs["actions"],
                inputs["step_levels"],
                inputs["signal_levels"],
            ),
            output_path=dynamics_prefill_layer_path,
            model_name=DYNAMICS_CACHED_PREFILL_LAYER_NAME,
            opset=args.opset,
            input_names=("z", "actions", "step_levels", "signal_levels"),
            output_names=("pred_z", *k_layer_names, *v_layer_names, "cache_length"),
            overwrite=args.overwrite,
        )
        export_to_onnx(
            fn=dynamics_step_fn,
            inputs=(
                cached_inputs["z_step"],
                cached_inputs["actions_step"],
                cached_inputs["step_levels_step"],
                cached_inputs["signal_levels_step"],
                cached_inputs["position_index"],
                cached_inputs["k_cache"],
                cached_inputs["v_cache"],
                cached_inputs["cache_length"],
            ),
            output_path=dynamics_step_path,
            model_name=DYNAMICS_CACHED_STEP_NAME,
            opset=args.opset,
            input_names=(
                "z",
                "actions",
                "step_levels",
                "signal_levels",
                "position_index",
                "k_cache",
                "v_cache",
                "cache_length",
            ),
            output_names=(
                "pred_z",
                "candidate_k_cache",
                "candidate_v_cache",
                "candidate_cache_length",
            ),
            overwrite=args.overwrite,
        )
        export_to_onnx(
            fn=dynamics_sample_step_fn,
            inputs=(
                cached_inputs["z_step"],
                cached_inputs["actions_step"],
                cached_inputs["position_index"],
                cached_inputs["k_cache"],
                cached_inputs["v_cache"],
                cached_inputs["cache_length"],
            ),
            output_path=dynamics_sample_step_path,
            model_name=DYNAMICS_CACHED_SAMPLE_STEP_NAME,
            opset=args.opset,
            input_names=(
                "sample_noise",
                "actions",
                "position_index",
                "k_cache",
                "v_cache",
                "cache_length",
            ),
            output_names=(
                "final_z",
                "pred_z",
                "candidate_k_cache",
                "candidate_v_cache",
                "candidate_cache_length",
            ),
            overwrite=args.overwrite,
        )
        export_to_onnx(
            fn=dynamics_sample_step_slide_fn,
            inputs=(
                cached_inputs["z_step"],
                cached_inputs["actions_step"],
                cached_inputs["position_index"],
                cached_inputs["k_cache"],
                cached_inputs["v_cache"],
                cached_inputs["cache_length"],
            ),
            output_path=dynamics_sample_step_slide_path,
            model_name=DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME,
            opset=args.opset,
            input_names=(
                "sample_noise",
                "actions",
                "position_index",
                "k_cache",
                "v_cache",
                "cache_length",
            ),
            output_names=(
                "final_z",
                "pred_z",
                "candidate_k_cache",
                "candidate_v_cache",
                "candidate_cache_length",
            ),
            overwrite=args.overwrite,
        )
        export_to_onnx(
            fn=dynamics_sample_append_context_fn,
            inputs=(
                cached_inputs["z_step"],
                cached_inputs["z_step"],
                cached_inputs["actions_step"],
                cached_inputs["position_index"],
                cached_inputs["k_cache"],
                cached_inputs["v_cache"],
                cached_inputs["cache_length"],
            ),
            output_path=dynamics_sample_append_context_path,
            model_name=DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME,
            opset=args.opset,
            input_names=(
                "sample_noise",
                "context_noise",
                "actions",
                "position_index",
                "k_cache",
                "v_cache",
                "cache_length",
            ),
            output_names=(
                "final_z",
                "pred_z",
                "candidate_k_cache",
                "candidate_v_cache",
                "candidate_cache_length",
            ),
            overwrite=args.overwrite,
        )
        export_to_onnx(
            fn=dynamics_sample_append_context_slide_fn,
            inputs=(
                cached_inputs["z_step"],
                cached_inputs["z_step"],
                cached_inputs["actions_step"],
                cached_inputs["position_index"],
                cached_inputs["k_cache"],
                cached_inputs["v_cache"],
                cached_inputs["cache_length"],
            ),
            output_path=dynamics_sample_append_context_slide_path,
            model_name=DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME,
            opset=args.opset,
            input_names=(
                "sample_noise",
                "context_noise",
                "actions",
                "position_index",
                "k_cache",
                "v_cache",
                "cache_length",
            ),
            output_names=(
                "final_z",
                "pred_z",
                "candidate_k_cache",
                "candidate_v_cache",
                "candidate_cache_length",
            ),
            overwrite=args.overwrite,
        )
        export_to_onnx(
            fn=dynamics_sample_append_context_slide_full_cache_fn,
            inputs=(
                cached_inputs["z_step"],
                cached_inputs["z_step"],
                cached_inputs["actions_step"],
                cached_inputs["k_cache"],
                cached_inputs["v_cache"],
            ),
            output_path=dynamics_sample_append_context_slide_full_cache_path,
            model_name=DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME,
            opset=args.opset,
            input_names=(
                "sample_noise",
                "context_noise",
                "actions",
                "k_cache",
                "v_cache",
            ),
            output_names=(
                "final_z",
                "pred_z",
                "candidate_k_cache",
                "candidate_v_cache",
            ),
            overwrite=args.overwrite,
        )
        export_to_onnx(
            fn=dynamics_sample_append_context_slide_entry_fn,
            inputs=(
                cached_inputs["z_step"],
                cached_inputs["z_step"],
                cached_inputs["actions_step"],
                cached_inputs["k_cache"],
                cached_inputs["v_cache"],
            ),
            output_path=dynamics_sample_append_context_slide_entry_path,
            model_name=DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME,
            opset=args.opset,
            input_names=(
                "sample_noise",
                "context_noise",
                "actions",
                "k_cache",
                "v_cache",
            ),
            output_names=(
                "final_z",
                "pred_z",
                "candidate_k_entry",
                "candidate_v_entry",
            ),
            overwrite=args.overwrite,
        )
        export_to_onnx(
            fn=dynamics_sample_append_context_slide_layer_fn,
            inputs=(
                cached_inputs["z_step"],
                cached_inputs["z_step"],
                cached_inputs["actions_step"],
                cached_inputs["position_index"],
                *k_layer_cache_inputs,
                *v_layer_cache_inputs,
                cached_inputs["cache_length"],
            ),
            output_path=dynamics_sample_append_context_slide_layer_path,
            model_name=DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME,
            opset=args.opset,
            input_names=(
                "sample_noise",
                "context_noise",
                "actions",
                "position_index",
                *k_layer_names,
                *v_layer_names,
                "cache_length",
            ),
            output_names=(
                "final_z",
                "pred_z",
                *candidate_k_layer_names,
                *candidate_v_layer_names,
                "candidate_cache_length",
            ),
            overwrite=args.overwrite,
        )

    exported_paths = {
        TOKENIZER_DECODER_NAME: decoder_path,
        DYNAMICS_UNCACHED_NAME: dynamics_path,
    }
    if args.export_cached:
        exported_paths.update(
            {
                TOKENIZER_DECODER_STEP_NAME: decoder_step_path,
                TOKENIZER_DECODE_Z_STEP_NAME: decoder_z_step_path,
                DYNAMICS_CACHED_PREFILL_NAME: dynamics_prefill_path,
                DYNAMICS_CACHED_PREFILL_LAYER_NAME: dynamics_prefill_layer_path,
                DYNAMICS_CACHED_STEP_NAME: dynamics_step_path,
                DYNAMICS_CACHED_SAMPLE_STEP_NAME: dynamics_sample_step_path,
                DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME: dynamics_sample_step_slide_path,
                DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME: dynamics_sample_append_context_path,
                DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME: (
                    dynamics_sample_append_context_slide_path
                ),
                DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME: (
                    dynamics_sample_append_context_slide_full_cache_path
                ),
                DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME: (
                    dynamics_sample_append_context_slide_entry_path
                ),
                DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME: (
                    dynamics_sample_append_context_slide_layer_path
                ),
            }
        )

    raw_artifacts = (
        snapshot_raw_artifacts(
            exported_paths,
            args.raw_out_dir,
            overwrite=args.overwrite,
        )
        if args.raw_out_dir is not None
        else {"enabled": False, "reason": "--raw_out_dir not set"}
    )

    simplification = {
        name: {"enabled": False, "reason": "--simplify_onnx not set"} for name in exported_paths
    }
    if args.simplify_onnx:
        demo_simplification_names = {
            TOKENIZER_DECODE_Z_STEP_NAME,
            DYNAMICS_CACHED_PREFILL_NAME,
            DYNAMICS_CACHED_PREFILL_LAYER_NAME,
            DYNAMICS_CACHED_SAMPLE_STEP_NAME,
            DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME,
            DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME,
            DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME,
            DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME,
            DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME,
            DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME,
        }
        simplification = {
            name: simplify_onnx_for_webgpu(path)
            if not args.simplify_demo_only or name in demo_simplification_names
            else {
                "enabled": False,
                "reason": "--simplify_demo_only skips artifacts outside the browser demo hot path",
                "tool": "onnxsim",
            }
            for name, path in exported_paths.items()
        }

    optimization = {
        name: {"enabled": False, "reason": "--skip_onnx_optimization"} for name in exported_paths
    }
    if not args.skip_onnx_optimization:
        optimization = {
            name: optimize_onnx_for_webgpu(path) for name, path in exported_paths.items()
        }

    layout_rewrite = {
        name: {"enabled": False, "reason": "--skip_singleton_reshape_rewrite"}
        for name in exported_paths
    }
    if not args.skip_singleton_reshape_rewrite:
        layout_rewrite = {
            name: rewrite_singleton_reshapes_for_webgpu(path)
            for name, path in exported_paths.items()
        }
        gqa_repeat_rewrite = {
            name: rewrite_gqa_repeats_for_webgpu(path) for name, path in exported_paths.items()
        }
        head_projection_rewriter = (
            rewrite_head_projection_reshapes_with_layout_ops_for_webgpu
            if args.head_projection_rewrite == "layout"
            else rewrite_head_projection_reshapes_for_webgpu
        )
        head_projection_rewrite = {
            name: head_projection_rewriter(path) for name, path in exported_paths.items()
        }
    else:
        gqa_repeat_rewrite = {
            name: {"enabled": False, "reason": "--skip_singleton_reshape_rewrite"}
            for name in exported_paths
        }
        head_projection_rewrite = {
            name: {"enabled": False, "reason": "--skip_singleton_reshape_rewrite"}
            for name in exported_paths
        }
    slide_static_cache_rewrite = {
        name: rewrite_slide_static_cache_ops_for_webgpu(path)
        if name
        in {
            DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME,
            DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME,
            DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME,
        }
        else {"enabled": False, "reason": "not a steady-state slide artifact"}
        for name, path in exported_paths.items()
    }

    rmsnorm_rewrite = {
        name: rewrite_rmsnorm_for_webgpu(path) for name, path in exported_paths.items()
    }
    skip_simplified_layer_norm_rewrite = {
        name: fuse_skip_simplified_layer_norm_for_webgpu(path)
        for name, path in exported_paths.items()
    }
    gather_index_rewrite = {
        name: rewrite_gather_int64_casts_for_webgpu(path) for name, path in exported_paths.items()
    }
    rotary_embedding_rewrite = {
        name: {"enabled": False, "reason": "--rotary_embedding_rewrite not set"}
        for name in exported_paths
    }
    if args.rotary_embedding_rewrite:
        rotary_embedding_rewrite = {
            name: rewrite_rotary_embedding_for_webgpu(path)
            for name, path in exported_paths.items()
        }
    validation = {
        TOKENIZER_DECODER_NAME: {"skipped": not args.validate},
        DYNAMICS_UNCACHED_NAME: {"skipped": not args.validate},
        TOKENIZER_DECODER_STEP_NAME: {"skipped": not (args.validate and args.export_cached)},
        TOKENIZER_DECODE_Z_STEP_NAME: {"skipped": not (args.validate and args.export_cached)},
        DYNAMICS_CACHED_PREFILL_NAME: {"skipped": not (args.validate and args.export_cached)},
        DYNAMICS_CACHED_PREFILL_LAYER_NAME: {"skipped": not (args.validate and args.export_cached)},
        DYNAMICS_CACHED_STEP_NAME: {"skipped": not (args.validate and args.export_cached)},
        DYNAMICS_CACHED_SAMPLE_STEP_NAME: {"skipped": not (args.validate and args.export_cached)},
        DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME: {
            "skipped": not (args.validate and args.export_cached)
        },
        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME: {
            "skipped": not (args.validate and args.export_cached)
        },
        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME: {
            "skipped": not (args.validate and args.export_cached)
        },
        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME: {
            "skipped": not (args.validate and args.export_cached)
        },
        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME: {
            "skipped": not (args.validate and args.export_cached)
        },
        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME: {
            "skipped": not (args.validate and args.export_cached)
        },
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

        if args.export_cached:
            validation[TOKENIZER_DECODER_STEP_NAME] = validate_single_output(
                path=decoder_step_path,
                feeds={"latent": cached_inputs["latent_step"]},
                output_name="patches",
                expected=decoder_step_fn(cached_inputs["latent_step"]),
                atol=args.atol,
                rtol=args.rtol,
            )
            validation[TOKENIZER_DECODE_Z_STEP_NAME] = validate_single_output(
                path=decoder_z_step_path,
                feeds={"z": cached_inputs["z_step"]},
                output_name="patches",
                expected=decoder_z_step_fn(cached_inputs["z_step"]),
                atol=args.atol,
                rtol=args.rtol,
            )
            prefill_expected = dynamics_prefill_fn(
                inputs["z"],
                inputs["actions"],
                inputs["step_levels"],
                inputs["signal_levels"],
            )
            validation[DYNAMICS_CACHED_PREFILL_NAME] = validate_outputs(
                path=dynamics_prefill_path,
                feeds={
                    "z": inputs["z"],
                    "actions": inputs["actions"],
                    "step_levels": inputs["step_levels"],
                    "signal_levels": inputs["signal_levels"],
                },
                expected={
                    "pred_z": prefill_expected[0],
                    "k_cache": prefill_expected[1],
                    "v_cache": prefill_expected[2],
                    "cache_length": prefill_expected[3],
                },
                atol=args.atol,
                rtol=args.rtol,
            )
            prefill_layer_expected = dynamics_prefill_layer_fn(
                inputs["z"],
                inputs["actions"],
                inputs["step_levels"],
                inputs["signal_levels"],
            )
            validation[DYNAMICS_CACHED_PREFILL_LAYER_NAME] = validate_outputs(
                path=dynamics_prefill_layer_path,
                feeds={
                    "z": inputs["z"],
                    "actions": inputs["actions"],
                    "step_levels": inputs["step_levels"],
                    "signal_levels": inputs["signal_levels"],
                },
                expected={
                    "pred_z": prefill_layer_expected[0],
                    **{name: prefill_layer_expected[1][i] for i, name in enumerate(k_layer_names)},
                    **{name: prefill_layer_expected[2][i] for i, name in enumerate(v_layer_names)},
                    "cache_length": prefill_layer_expected[3],
                },
                atol=args.atol,
                rtol=args.rtol,
            )
            step_expected = dynamics_step_fn(
                cached_inputs["z_step"],
                cached_inputs["actions_step"],
                cached_inputs["step_levels_step"],
                cached_inputs["signal_levels_step"],
                cached_inputs["position_index"],
                cached_inputs["k_cache"],
                cached_inputs["v_cache"],
                cached_inputs["cache_length"],
            )
            validation[DYNAMICS_CACHED_STEP_NAME] = validate_outputs(
                path=dynamics_step_path,
                feeds={
                    "z": cached_inputs["z_step"],
                    "actions": cached_inputs["actions_step"],
                    "step_levels": cached_inputs["step_levels_step"],
                    "signal_levels": cached_inputs["signal_levels_step"],
                    "position_index": cached_inputs["position_index"],
                    "k_cache": cached_inputs["k_cache"],
                    "v_cache": cached_inputs["v_cache"],
                    "cache_length": cached_inputs["cache_length"],
                },
                expected={
                    "pred_z": step_expected[0],
                    "candidate_k_cache": step_expected[1],
                    "candidate_v_cache": step_expected[2],
                    "candidate_cache_length": step_expected[3],
                },
                atol=args.atol,
                rtol=args.rtol,
            )
            sample_step_expected = dynamics_sample_step_fn(
                cached_inputs["z_step"],
                cached_inputs["actions_step"],
                cached_inputs["position_index"],
                cached_inputs["k_cache"],
                cached_inputs["v_cache"],
                cached_inputs["cache_length"],
            )
            validation[DYNAMICS_CACHED_SAMPLE_STEP_NAME] = validate_outputs(
                path=dynamics_sample_step_path,
                feeds={
                    "sample_noise": cached_inputs["z_step"],
                    "actions": cached_inputs["actions_step"],
                    "position_index": cached_inputs["position_index"],
                    "k_cache": cached_inputs["k_cache"],
                    "v_cache": cached_inputs["v_cache"],
                    "cache_length": cached_inputs["cache_length"],
                },
                expected={
                    "final_z": sample_step_expected[0],
                    "pred_z": sample_step_expected[1],
                    "candidate_k_cache": sample_step_expected[2],
                    "candidate_v_cache": sample_step_expected[3],
                    "candidate_cache_length": sample_step_expected[4],
                },
                atol=args.atol,
                rtol=args.rtol,
            )
            sample_step_slide_expected = dynamics_sample_step_slide_fn(
                cached_inputs["z_step"],
                cached_inputs["actions_step"],
                cached_inputs["position_index"],
                cached_inputs["k_cache"],
                cached_inputs["v_cache"],
                cached_inputs["cache_length"],
            )
            validation[DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME] = validate_outputs(
                path=dynamics_sample_step_slide_path,
                feeds={
                    "sample_noise": cached_inputs["z_step"],
                    "actions": cached_inputs["actions_step"],
                    "position_index": cached_inputs["position_index"],
                    "k_cache": cached_inputs["k_cache"],
                    "v_cache": cached_inputs["v_cache"],
                    "cache_length": cached_inputs["cache_length"],
                },
                expected={
                    "final_z": sample_step_slide_expected[0],
                    "pred_z": sample_step_slide_expected[1],
                    "candidate_k_cache": sample_step_slide_expected[2],
                    "candidate_v_cache": sample_step_slide_expected[3],
                    "candidate_cache_length": sample_step_slide_expected[4],
                },
                atol=args.atol,
                rtol=args.rtol,
            )
            sample_append_context_expected = dynamics_sample_append_context_fn(
                cached_inputs["z_step"],
                cached_inputs["z_step"],
                cached_inputs["actions_step"],
                cached_inputs["position_index"],
                cached_inputs["k_cache"],
                cached_inputs["v_cache"],
                cached_inputs["cache_length"],
            )
            validation[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME] = validate_outputs(
                path=dynamics_sample_append_context_path,
                feeds={
                    "sample_noise": cached_inputs["z_step"],
                    "context_noise": cached_inputs["z_step"],
                    "actions": cached_inputs["actions_step"],
                    "position_index": cached_inputs["position_index"],
                    "k_cache": cached_inputs["k_cache"],
                    "v_cache": cached_inputs["v_cache"],
                    "cache_length": cached_inputs["cache_length"],
                },
                expected={
                    "final_z": sample_append_context_expected[0],
                    "pred_z": sample_append_context_expected[1],
                    "candidate_k_cache": sample_append_context_expected[2],
                    "candidate_v_cache": sample_append_context_expected[3],
                    "candidate_cache_length": sample_append_context_expected[4],
                },
                atol=args.atol,
                rtol=args.rtol,
            )
            sample_append_context_slide_expected = dynamics_sample_append_context_slide_fn(
                cached_inputs["z_step"],
                cached_inputs["z_step"],
                cached_inputs["actions_step"],
                cached_inputs["position_index"],
                cached_inputs["k_cache"],
                cached_inputs["v_cache"],
                cached_inputs["cache_length"],
            )
            validation[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME] = validate_outputs(
                path=dynamics_sample_append_context_slide_path,
                feeds={
                    "sample_noise": cached_inputs["z_step"],
                    "context_noise": cached_inputs["z_step"],
                    "actions": cached_inputs["actions_step"],
                    "position_index": cached_inputs["position_index"],
                    "k_cache": cached_inputs["k_cache"],
                    "v_cache": cached_inputs["v_cache"],
                    "cache_length": cached_inputs["cache_length"],
                },
                expected={
                    "final_z": sample_append_context_slide_expected[0],
                    "pred_z": sample_append_context_slide_expected[1],
                    "candidate_k_cache": sample_append_context_slide_expected[2],
                    "candidate_v_cache": sample_append_context_slide_expected[3],
                    "candidate_cache_length": sample_append_context_slide_expected[4],
                },
                atol=args.atol,
                rtol=args.rtol,
            )
            sample_append_context_slide_full_cache_expected = (
                dynamics_sample_append_context_slide_full_cache_fn(
                    cached_inputs["z_step"],
                    cached_inputs["z_step"],
                    cached_inputs["actions_step"],
                    cached_inputs["k_cache"],
                    cached_inputs["v_cache"],
                )
            )
            validation[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME] = (
                validate_outputs(
                    path=dynamics_sample_append_context_slide_full_cache_path,
                    feeds={
                        "sample_noise": cached_inputs["z_step"],
                        "context_noise": cached_inputs["z_step"],
                        "actions": cached_inputs["actions_step"],
                        "k_cache": cached_inputs["k_cache"],
                        "v_cache": cached_inputs["v_cache"],
                    },
                    expected={
                        "final_z": sample_append_context_slide_full_cache_expected[0],
                        "pred_z": sample_append_context_slide_full_cache_expected[1],
                        "candidate_k_cache": sample_append_context_slide_full_cache_expected[2],
                        "candidate_v_cache": sample_append_context_slide_full_cache_expected[3],
                    },
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
            sample_append_context_slide_entry_expected = (
                dynamics_sample_append_context_slide_entry_fn(
                    cached_inputs["z_step"],
                    cached_inputs["z_step"],
                    cached_inputs["actions_step"],
                    cached_inputs["k_cache"],
                    cached_inputs["v_cache"],
                )
            )
            validation[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME] = validate_outputs(
                path=dynamics_sample_append_context_slide_entry_path,
                feeds={
                    "sample_noise": cached_inputs["z_step"],
                    "context_noise": cached_inputs["z_step"],
                    "actions": cached_inputs["actions_step"],
                    "k_cache": cached_inputs["k_cache"],
                    "v_cache": cached_inputs["v_cache"],
                },
                expected={
                    "final_z": sample_append_context_slide_entry_expected[0],
                    "pred_z": sample_append_context_slide_entry_expected[1],
                    "candidate_k_entry": sample_append_context_slide_entry_expected[2],
                    "candidate_v_entry": sample_append_context_slide_entry_expected[3],
                },
                atol=args.atol,
                rtol=args.rtol,
            )
            sample_append_context_slide_layer_expected = (
                dynamics_sample_append_context_slide_layer_fn(
                    cached_inputs["z_step"],
                    cached_inputs["z_step"],
                    cached_inputs["actions_step"],
                    cached_inputs["position_index"],
                    *k_layer_cache_inputs,
                    *v_layer_cache_inputs,
                    cached_inputs["cache_length"],
                )
            )
            validation[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME] = validate_outputs(
                path=dynamics_sample_append_context_slide_layer_path,
                feeds={
                    "sample_noise": cached_inputs["z_step"],
                    "context_noise": cached_inputs["z_step"],
                    "actions": cached_inputs["actions_step"],
                    "position_index": cached_inputs["position_index"],
                    **{name: k_layer_cache_inputs[i] for i, name in enumerate(k_layer_names)},
                    **{name: v_layer_cache_inputs[i] for i, name in enumerate(v_layer_names)},
                    "cache_length": cached_inputs["cache_length"],
                },
                expected={
                    "final_z": sample_append_context_slide_layer_expected[0],
                    "pred_z": sample_append_context_slide_layer_expected[1],
                    **{
                        name: sample_append_context_slide_layer_expected[2][i]
                        for i, name in enumerate(candidate_k_layer_names)
                    },
                    **{
                        name: sample_append_context_slide_layer_expected[3][i]
                        for i, name in enumerate(candidate_v_layer_names)
                    },
                    "candidate_cache_length": sample_append_context_slide_layer_expected[4],
                },
                atol=args.atol,
                rtol=args.rtol,
            )

        failed = [
            name
            for name, result in validation.items()
            if not result.get("skipped", False) and not result.get("passed", False)
        ]
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
    decoder_step_files = export_file_metadata(decoder_step_path) if args.export_cached else None
    decoder_z_step_files = export_file_metadata(decoder_z_step_path) if args.export_cached else None
    dynamics_prefill_files = (
        export_file_metadata(dynamics_prefill_path) if args.export_cached else None
    )
    dynamics_prefill_layer_files = (
        export_file_metadata(dynamics_prefill_layer_path) if args.export_cached else None
    )
    dynamics_step_files = export_file_metadata(dynamics_step_path) if args.export_cached else None
    dynamics_sample_step_files = (
        export_file_metadata(dynamics_sample_step_path) if args.export_cached else None
    )
    dynamics_sample_step_slide_files = (
        export_file_metadata(dynamics_sample_step_slide_path) if args.export_cached else None
    )
    dynamics_sample_append_context_files = (
        export_file_metadata(dynamics_sample_append_context_path) if args.export_cached else None
    )
    dynamics_sample_append_context_slide_files = (
        export_file_metadata(dynamics_sample_append_context_slide_path)
        if args.export_cached
        else None
    )
    dynamics_sample_append_context_slide_full_cache_files = (
        export_file_metadata(dynamics_sample_append_context_slide_full_cache_path)
        if args.export_cached
        else None
    )
    dynamics_sample_append_context_slide_entry_files = (
        export_file_metadata(dynamics_sample_append_context_slide_entry_path)
        if args.export_cached
        else None
    )
    dynamics_sample_append_context_slide_layer_files = (
        export_file_metadata(dynamics_sample_append_context_slide_layer_path)
        if args.export_cached
        else None
    )
    exports = [
        {
            "name": TOKENIZER_DECODER_NAME,
            **decoder_files,
            "inputs": {"latent": tensor_spec("float32", tok_shapes.latent)},
            "outputs": {"patches": tensor_spec("float32", tok_shapes.patches)},
            "validation": validation[TOKENIZER_DECODER_NAME],
            "simplification": simplification[TOKENIZER_DECODER_NAME],
            "optimization": optimization[TOKENIZER_DECODER_NAME],
            "layout_rewrite": layout_rewrite[TOKENIZER_DECODER_NAME],
            "gqa_repeat_rewrite": gqa_repeat_rewrite[TOKENIZER_DECODER_NAME],
            "head_projection_rewrite": head_projection_rewrite[TOKENIZER_DECODER_NAME],
            "rmsnorm_rewrite": rmsnorm_rewrite[TOKENIZER_DECODER_NAME],
            "gather_index_rewrite": gather_index_rewrite[TOKENIZER_DECODER_NAME],
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
            "simplification": simplification[DYNAMICS_UNCACHED_NAME],
            "optimization": optimization[DYNAMICS_UNCACHED_NAME],
            "layout_rewrite": layout_rewrite[DYNAMICS_UNCACHED_NAME],
            "gqa_repeat_rewrite": gqa_repeat_rewrite[DYNAMICS_UNCACHED_NAME],
            "head_projection_rewrite": head_projection_rewrite[DYNAMICS_UNCACHED_NAME],
            "rmsnorm_rewrite": rmsnorm_rewrite[DYNAMICS_UNCACHED_NAME],
            "gather_index_rewrite": gather_index_rewrite[DYNAMICS_UNCACHED_NAME],
            "production_browser_ready": False,
        },
    ]
    if args.export_cached:
        exports.extend(
            [
                {
                    "name": TOKENIZER_DECODER_STEP_NAME,
                    **decoder_step_files,
                    "inputs": {
                        "latent": tensor_spec(
                            "float32",
                            (args.batch_size, 1, tok_shapes.num_latents, tok_shapes.channel_dim),
                        )
                    },
                    "outputs": {
                        "patches": tensor_spec(
                            "float32",
                            (args.batch_size, 1, tok_shapes.patch_count, tok_shapes.patch_dim),
                        )
                    },
                    "validation": validation[TOKENIZER_DECODER_STEP_NAME],
                    "simplification": simplification[TOKENIZER_DECODER_STEP_NAME],
                    "optimization": optimization[TOKENIZER_DECODER_STEP_NAME],
                    "layout_rewrite": layout_rewrite[TOKENIZER_DECODER_STEP_NAME],
                    "gqa_repeat_rewrite": gqa_repeat_rewrite[TOKENIZER_DECODER_STEP_NAME],
                    "head_projection_rewrite": head_projection_rewrite[TOKENIZER_DECODER_STEP_NAME],
                    "rmsnorm_rewrite": rmsnorm_rewrite[TOKENIZER_DECODER_STEP_NAME],
                    "gather_index_rewrite": gather_index_rewrite[TOKENIZER_DECODER_STEP_NAME],
                    "production_browser_ready": True,
                },
                {
                    "name": TOKENIZER_DECODE_Z_STEP_NAME,
                    **decoder_z_step_files,
                    "inputs": {"z": tensor_spec("float32", dyn_shapes.step_z)},
                    "outputs": {
                        "patches": tensor_spec(
                            "float32",
                            (args.batch_size, 1, tok_shapes.patch_count, tok_shapes.patch_dim),
                        )
                    },
                    "validation": validation[TOKENIZER_DECODE_Z_STEP_NAME],
                    "simplification": simplification[TOKENIZER_DECODE_Z_STEP_NAME],
                    "optimization": optimization[TOKENIZER_DECODE_Z_STEP_NAME],
                    "layout_rewrite": layout_rewrite[TOKENIZER_DECODE_Z_STEP_NAME],
                    "gqa_repeat_rewrite": gqa_repeat_rewrite[TOKENIZER_DECODE_Z_STEP_NAME],
                    "head_projection_rewrite": head_projection_rewrite[
                        TOKENIZER_DECODE_Z_STEP_NAME
                    ],
                    "rmsnorm_rewrite": rmsnorm_rewrite[TOKENIZER_DECODE_Z_STEP_NAME],
                    "gather_index_rewrite": gather_index_rewrite[TOKENIZER_DECODE_Z_STEP_NAME],
                    "production_browser_ready": True,
                    "decode_z": {
                        "source": "final_z_after_velocity_update",
                        "dynamics_shape": list(dyn_shapes.step_z),
                        "decoder_latent_shape": [
                            args.batch_size,
                            1,
                            tok_shapes.num_latents,
                            tok_shapes.channel_dim,
                        ],
                    },
                },
                {
                    "name": DYNAMICS_CACHED_PREFILL_NAME,
                    **dynamics_prefill_files,
                    "inputs": {
                        "z": tensor_spec("float32", dyn_shapes.z),
                        "actions": tensor_spec("int32", dyn_shapes.levels),
                        "step_levels": tensor_spec("int32", dyn_shapes.levels),
                        "signal_levels": tensor_spec("int32", dyn_shapes.levels),
                    },
                    "outputs": {
                        "pred_z": tensor_spec("float32", dyn_shapes.z),
                        "k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "v_cache": tensor_spec("float32", dyn_shapes.cache),
                        "cache_length": tensor_spec("int32", (1,)),
                    },
                    "validation": validation[DYNAMICS_CACHED_PREFILL_NAME],
                    "simplification": simplification[DYNAMICS_CACHED_PREFILL_NAME],
                    "optimization": optimization[DYNAMICS_CACHED_PREFILL_NAME],
                    "layout_rewrite": layout_rewrite[DYNAMICS_CACHED_PREFILL_NAME],
                    "gqa_repeat_rewrite": gqa_repeat_rewrite[DYNAMICS_CACHED_PREFILL_NAME],
                    "head_projection_rewrite": head_projection_rewrite[
                        DYNAMICS_CACHED_PREFILL_NAME
                    ],
                    "rmsnorm_rewrite": rmsnorm_rewrite[DYNAMICS_CACHED_PREFILL_NAME],
                    "gather_index_rewrite": gather_index_rewrite[DYNAMICS_CACHED_PREFILL_NAME],
                    "production_browser_ready": True,
                },
                {
                    "name": DYNAMICS_CACHED_PREFILL_LAYER_NAME,
                    **dynamics_prefill_layer_files,
                    "inputs": {
                        "z": tensor_spec("float32", dyn_shapes.z),
                        "actions": tensor_spec("int32", dyn_shapes.levels),
                        "step_levels": tensor_spec("int32", dyn_shapes.levels),
                        "signal_levels": tensor_spec("int32", dyn_shapes.levels),
                    },
                    "outputs": {
                        "pred_z": tensor_spec("float32", dyn_shapes.z),
                        **{
                            name: tensor_spec("float32", dyn_shapes.layer_cache)
                            for name in k_layer_names
                        },
                        **{
                            name: tensor_spec("float32", dyn_shapes.layer_cache)
                            for name in v_layer_names
                        },
                        "cache_length": tensor_spec("int32", (1,)),
                    },
                    "validation": validation[DYNAMICS_CACHED_PREFILL_LAYER_NAME],
                    "simplification": simplification[DYNAMICS_CACHED_PREFILL_LAYER_NAME],
                    "optimization": optimization[DYNAMICS_CACHED_PREFILL_LAYER_NAME],
                    "layout_rewrite": layout_rewrite[DYNAMICS_CACHED_PREFILL_LAYER_NAME],
                    "gqa_repeat_rewrite": gqa_repeat_rewrite[DYNAMICS_CACHED_PREFILL_LAYER_NAME],
                    "head_projection_rewrite": head_projection_rewrite[
                        DYNAMICS_CACHED_PREFILL_LAYER_NAME
                    ],
                    "rmsnorm_rewrite": rmsnorm_rewrite[DYNAMICS_CACHED_PREFILL_LAYER_NAME],
                    "gather_index_rewrite": gather_index_rewrite[
                        DYNAMICS_CACHED_PREFILL_LAYER_NAME
                    ],
                    "production_browser_ready": True,
                    "cache_layout": "per_temporal_layer",
                },
                {
                    "name": DYNAMICS_CACHED_STEP_NAME,
                    **dynamics_step_files,
                    "inputs": {
                        "z": tensor_spec("float32", dyn_shapes.step_z),
                        "actions": tensor_spec("int32", dyn_shapes.step_levels),
                        "step_levels": tensor_spec("int32", dyn_shapes.step_levels),
                        "signal_levels": tensor_spec("int32", dyn_shapes.step_levels),
                        "position_index": tensor_spec("int32", dyn_shapes.position_index),
                        "k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "v_cache": tensor_spec("float32", dyn_shapes.cache),
                        "cache_length": tensor_spec("int32", (1,)),
                    },
                    "outputs": {
                        "pred_z": tensor_spec("float32", dyn_shapes.step_z),
                        "candidate_k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "candidate_v_cache": tensor_spec("float32", dyn_shapes.cache),
                        "candidate_cache_length": tensor_spec("int32", (1,)),
                    },
                    "validation": validation[DYNAMICS_CACHED_STEP_NAME],
                    "simplification": simplification[DYNAMICS_CACHED_STEP_NAME],
                    "optimization": optimization[DYNAMICS_CACHED_STEP_NAME],
                    "layout_rewrite": layout_rewrite[DYNAMICS_CACHED_STEP_NAME],
                    "gqa_repeat_rewrite": gqa_repeat_rewrite[DYNAMICS_CACHED_STEP_NAME],
                    "head_projection_rewrite": head_projection_rewrite[DYNAMICS_CACHED_STEP_NAME],
                    "rmsnorm_rewrite": rmsnorm_rewrite[DYNAMICS_CACHED_STEP_NAME],
                    "gather_index_rewrite": gather_index_rewrite[DYNAMICS_CACHED_STEP_NAME],
                    "production_browser_ready": True,
                },
                {
                    "name": DYNAMICS_CACHED_SAMPLE_STEP_NAME,
                    **dynamics_sample_step_files,
                    "inputs": {
                        "sample_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "actions": tensor_spec("int32", dyn_shapes.step_levels),
                        "position_index": tensor_spec("int32", dyn_shapes.position_index),
                        "k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "v_cache": tensor_spec("float32", dyn_shapes.cache),
                        "cache_length": tensor_spec("int32", (1,)),
                    },
                    "outputs": {
                        "final_z": tensor_spec("float32", dyn_shapes.step_z),
                        "pred_z": tensor_spec("float32", dyn_shapes.step_z),
                        "candidate_k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "candidate_v_cache": tensor_spec("float32", dyn_shapes.cache),
                        "candidate_cache_length": tensor_spec("int32", (1,)),
                    },
                    "validation": validation[DYNAMICS_CACHED_SAMPLE_STEP_NAME],
                    "simplification": simplification[DYNAMICS_CACHED_SAMPLE_STEP_NAME],
                    "optimization": optimization[DYNAMICS_CACHED_SAMPLE_STEP_NAME],
                    "layout_rewrite": layout_rewrite[DYNAMICS_CACHED_SAMPLE_STEP_NAME],
                    "gqa_repeat_rewrite": gqa_repeat_rewrite[DYNAMICS_CACHED_SAMPLE_STEP_NAME],
                    "head_projection_rewrite": head_projection_rewrite[
                        DYNAMICS_CACHED_SAMPLE_STEP_NAME
                    ],
                    "rmsnorm_rewrite": rmsnorm_rewrite[DYNAMICS_CACHED_SAMPLE_STEP_NAME],
                    "gather_index_rewrite": gather_index_rewrite[DYNAMICS_CACHED_SAMPLE_STEP_NAME],
                    "production_browser_ready": True,
                    "sample_steps": args.sample_steps,
                    "sample_cache_policy": "read_committed_each_sample_commit_final_only",
                    "fallback": DYNAMICS_CACHED_STEP_NAME,
                    "cache_update": "fill",
                    "steady_state_export": DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME,
                },
                {
                    "name": DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME,
                    **dynamics_sample_step_slide_files,
                    "inputs": {
                        "sample_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "actions": tensor_spec("int32", dyn_shapes.step_levels),
                        "position_index": tensor_spec("int32", dyn_shapes.position_index),
                        "k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "v_cache": tensor_spec("float32", dyn_shapes.cache),
                        "cache_length": tensor_spec("int32", (1,)),
                    },
                    "outputs": {
                        "final_z": tensor_spec("float32", dyn_shapes.step_z),
                        "pred_z": tensor_spec("float32", dyn_shapes.step_z),
                        "candidate_k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "candidate_v_cache": tensor_spec("float32", dyn_shapes.cache),
                        "candidate_cache_length": tensor_spec("int32", (1,)),
                    },
                    "validation": validation[DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME],
                    "simplification": simplification[DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME],
                    "optimization": optimization[DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME],
                    "layout_rewrite": layout_rewrite[DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME],
                    "gqa_repeat_rewrite": gqa_repeat_rewrite[
                        DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME
                    ],
                    "head_projection_rewrite": head_projection_rewrite[
                        DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME
                    ],
                    "rmsnorm_rewrite": rmsnorm_rewrite[DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME],
                    "gather_index_rewrite": gather_index_rewrite[
                        DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME
                    ],
                    "production_browser_ready": True,
                    "sample_steps": args.sample_steps,
                    "sample_cache_policy": "read_committed_each_sample_commit_final_only",
                    "fallback": DYNAMICS_CACHED_STEP_NAME,
                    "cache_update": "slide",
                },
                {
                    "name": DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME,
                    **dynamics_sample_append_context_files,
                    "inputs": {
                        "sample_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "context_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "actions": tensor_spec("int32", dyn_shapes.step_levels),
                        "position_index": tensor_spec("int32", dyn_shapes.position_index),
                        "k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "v_cache": tensor_spec("float32", dyn_shapes.cache),
                        "cache_length": tensor_spec("int32", (1,)),
                    },
                    "outputs": {
                        "final_z": tensor_spec("float32", dyn_shapes.step_z),
                        "pred_z": tensor_spec("float32", dyn_shapes.step_z),
                        "candidate_k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "candidate_v_cache": tensor_spec("float32", dyn_shapes.cache),
                        "candidate_cache_length": tensor_spec("int32", (1,)),
                    },
                    "validation": validation[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME],
                    "simplification": simplification[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME],
                    "optimization": optimization[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME],
                    "layout_rewrite": layout_rewrite[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME],
                    "gqa_repeat_rewrite": gqa_repeat_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME
                    ],
                    "head_projection_rewrite": head_projection_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME
                    ],
                    "rmsnorm_rewrite": rmsnorm_rewrite[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME],
                    "gather_index_rewrite": gather_index_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME
                    ],
                    "production_browser_ready": True,
                    "sample_steps": args.sample_steps,
                    "context_tau": args.context_tau,
                    "sample_cache_policy": "sample_then_append_generated_context",
                    "fallback": DYNAMICS_CACHED_STEP_NAME,
                    "cache_update": "fill",
                    "steady_state_export": DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME,
                },
                {
                    "name": DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME,
                    **dynamics_sample_append_context_slide_files,
                    "inputs": {
                        "sample_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "context_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "actions": tensor_spec("int32", dyn_shapes.step_levels),
                        "position_index": tensor_spec("int32", dyn_shapes.position_index),
                        "k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "v_cache": tensor_spec("float32", dyn_shapes.cache),
                        "cache_length": tensor_spec("int32", (1,)),
                    },
                    "outputs": {
                        "final_z": tensor_spec("float32", dyn_shapes.step_z),
                        "pred_z": tensor_spec("float32", dyn_shapes.step_z),
                        "candidate_k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "candidate_v_cache": tensor_spec("float32", dyn_shapes.cache),
                        "candidate_cache_length": tensor_spec("int32", (1,)),
                    },
                    "validation": validation[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME],
                    "simplification": simplification[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME
                    ],
                    "optimization": optimization[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME],
                    "layout_rewrite": layout_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME
                    ],
                    "gqa_repeat_rewrite": gqa_repeat_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME
                    ],
                    "head_projection_rewrite": head_projection_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME
                    ],
                    "rmsnorm_rewrite": rmsnorm_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME
                    ],
                    "gather_index_rewrite": gather_index_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME
                    ],
                    "production_browser_ready": True,
                    "sample_steps": args.sample_steps,
                    "context_tau": args.context_tau,
                    "sample_cache_policy": "sample_then_append_generated_context",
                    "fallback": DYNAMICS_CACHED_STEP_NAME,
                    "cache_update": "slide",
                    "full_cache_export": DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME,
                },
                {
                    "name": DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME,
                    **dynamics_sample_append_context_slide_full_cache_files,
                    "inputs": {
                        "sample_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "context_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "actions": tensor_spec("int32", dyn_shapes.step_levels),
                        "k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "v_cache": tensor_spec("float32", dyn_shapes.cache),
                    },
                    "outputs": {
                        "final_z": tensor_spec("float32", dyn_shapes.step_z),
                        "pred_z": tensor_spec("float32", dyn_shapes.step_z),
                        "candidate_k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "candidate_v_cache": tensor_spec("float32", dyn_shapes.cache),
                    },
                    "validation": validation[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME
                    ],
                    "simplification": simplification[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME
                    ],
                    "optimization": optimization[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME
                    ],
                    "layout_rewrite": layout_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME
                    ],
                    "gqa_repeat_rewrite": gqa_repeat_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME
                    ],
                    "head_projection_rewrite": head_projection_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME
                    ],
                    "rmsnorm_rewrite": rmsnorm_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME
                    ],
                    "gather_index_rewrite": gather_index_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME
                    ],
                    "production_browser_ready": True,
                    "sample_steps": args.sample_steps,
                    "context_tau": args.context_tau,
                    "sample_cache_policy": "sample_then_append_generated_context",
                    "fallback": DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME,
                    "cache_update": "slide",
                    "steady_state_full_cache_specialized": True,
                    "entry_cache_export": DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME,
                },
                {
                    "name": DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME,
                    **dynamics_sample_append_context_slide_entry_files,
                    "inputs": {
                        "sample_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "context_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "actions": tensor_spec("int32", dyn_shapes.step_levels),
                        "k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "v_cache": tensor_spec("float32", dyn_shapes.cache),
                    },
                    "outputs": {
                        "final_z": tensor_spec("float32", dyn_shapes.step_z),
                        "pred_z": tensor_spec("float32", dyn_shapes.step_z),
                        "candidate_k_entry": tensor_spec(
                            "float32",
                            (
                                dyn_shapes.cache[0],
                                dyn_shapes.cache[1],
                                dyn_shapes.cache[2],
                                1,
                                dyn_shapes.cache[4],
                                dyn_shapes.cache[5],
                            ),
                        ),
                        "candidate_v_entry": tensor_spec(
                            "float32",
                            (
                                dyn_shapes.cache[0],
                                dyn_shapes.cache[1],
                                dyn_shapes.cache[2],
                                1,
                                dyn_shapes.cache[4],
                                dyn_shapes.cache[5],
                            ),
                        ),
                    },
                    "validation": validation[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME
                    ],
                    "simplification": simplification[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME
                    ],
                    "optimization": optimization[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME
                    ],
                    "layout_rewrite": layout_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME
                    ],
                    "gqa_repeat_rewrite": gqa_repeat_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME
                    ],
                    "head_projection_rewrite": head_projection_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME
                    ],
                    "rmsnorm_rewrite": rmsnorm_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME
                    ],
                    "gather_index_rewrite": gather_index_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME
                    ],
                    "production_browser_ready": True,
                    "sample_steps": args.sample_steps,
                    "context_tau": args.context_tau,
                    "sample_cache_policy": "sample_then_append_generated_context",
                    "fallback": DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME,
                    "cache_update": "slide",
                    "cache_update_contract": "webgpu_inplace_slide_rebase_entry",
                    "steady_state_full_cache_specialized": True,
                },
                {
                    "name": DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME,
                    **dynamics_sample_append_context_slide_layer_files,
                    "inputs": {
                        "sample_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "context_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "actions": tensor_spec("int32", dyn_shapes.step_levels),
                        "position_index": tensor_spec("int32", dyn_shapes.position_index),
                        **{
                            name: tensor_spec("float32", dyn_shapes.layer_cache)
                            for name in k_layer_names
                        },
                        **{
                            name: tensor_spec("float32", dyn_shapes.layer_cache)
                            for name in v_layer_names
                        },
                        "cache_length": tensor_spec("int32", (1,)),
                    },
                    "outputs": {
                        "final_z": tensor_spec("float32", dyn_shapes.step_z),
                        "pred_z": tensor_spec("float32", dyn_shapes.step_z),
                        **{
                            name: tensor_spec("float32", dyn_shapes.layer_cache)
                            for name in candidate_k_layer_names
                        },
                        **{
                            name: tensor_spec("float32", dyn_shapes.layer_cache)
                            for name in candidate_v_layer_names
                        },
                        "candidate_cache_length": tensor_spec("int32", (1,)),
                    },
                    "validation": validation[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME
                    ],
                    "simplification": simplification[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME
                    ],
                    "optimization": optimization[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME
                    ],
                    "layout_rewrite": layout_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME
                    ],
                    "gqa_repeat_rewrite": gqa_repeat_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME
                    ],
                    "head_projection_rewrite": head_projection_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME
                    ],
                    "rmsnorm_rewrite": rmsnorm_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME
                    ],
                    "gather_index_rewrite": gather_index_rewrite[
                        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME
                    ],
                    "production_browser_ready": True,
                    "sample_steps": args.sample_steps,
                    "context_tau": args.context_tau,
                    "sample_cache_policy": "sample_then_append_generated_context",
                    "fallback": DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME,
                    "cache_update": "slide",
                    "cache_layout": "per_temporal_layer",
                },
            ]
        )

    for entry in exports:
        entry["slide_static_cache_rewrite"] = slide_static_cache_rewrite[entry["name"]]
        entry["skip_simplified_layer_norm_rewrite"] = skip_simplified_layer_norm_rewrite[
            entry["name"]
        ]
        entry["rotary_embedding_rewrite"] = rotary_embedding_rewrite[entry["name"]]

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
        "attention_export": {"implementation": "patched_onnx_decomposition"},
        "layout_rewrite": {
            "singleton_reshape_to_squeeze_unsqueeze": not args.skip_singleton_reshape_rewrite,
            "gqa_repeat_to_gather": not args.skip_singleton_reshape_rewrite,
            "head_projection_reshape_to_einsum": not args.skip_singleton_reshape_rewrite,
        },
        "checkpoints": {
            "tokenizer_dir": str(args.tokenizer_dir),
            "tokenizer_step": tokenizer_step,
            "dynamics_dir": str(args.dynamics_dir),
            "dynamics_step": dynamics_step,
        },
        "context_latents": context_artifact,
        "raw_artifacts": raw_artifacts,
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
            "rope_base": float(dynamics_cfg.base),
            "num_kv_heads": dyn_shapes.num_kv_heads,
            "head_dim": dyn_shapes.head_dim,
        },
        "exports": exports,
        "cache_contract": cache_contract(
            dyn_shapes,
            available=args.export_cached,
            dtype="float32",
        ),
        "demo_generation": {
            "sample_steps": args.sample_steps,
            "context_tau": args.context_tau,
            "sample_cache_policy": "sample_then_append_generated_context",
            "preferred_step_export": DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME,
            "preferred_prefill_export": DYNAMICS_CACHED_PREFILL_NAME,
            "preferred_steady_state_step_export": (
                DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME
            ),
            "fallback_steady_state_step_export": (
                DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME
            ),
            "experimental_layer_prefill_export": DYNAMICS_CACHED_PREFILL_LAYER_NAME,
            "experimental_layer_steady_state_step_export": (
                DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME
            ),
            "legacy_sample_step_export": DYNAMICS_CACHED_SAMPLE_STEP_NAME,
            "legacy_steady_state_sample_step_export": DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME,
            "fallback_step_export": DYNAMICS_CACHED_STEP_NAME,
            "decode_z": {
                "source": "final_z_after_velocity_update",
                "dynamics_shape": list(dyn_shapes.step_z),
                "decoder_latent_shape": [
                    args.batch_size,
                    1,
                    tok_shapes.num_latents,
                    tok_shapes.channel_dim,
                ],
            },
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {decoder_path}")
    print(f"Wrote {dynamics_path}")
    if args.export_cached:
        print(f"Wrote {decoder_step_path}")
        print(f"Wrote {decoder_z_step_path}")
        print(f"Wrote {dynamics_prefill_path}")
        print(f"Wrote {dynamics_prefill_layer_path}")
        print(f"Wrote {dynamics_step_path}")
        print(f"Wrote {dynamics_sample_step_path}")
        print(f"Wrote {dynamics_sample_step_slide_path}")
        print(f"Wrote {dynamics_sample_append_context_path}")
        print(f"Wrote {dynamics_sample_append_context_slide_path}")
        print(f"Wrote {dynamics_sample_append_context_slide_full_cache_path}")
        print(f"Wrote {dynamics_sample_append_context_slide_entry_path}")
        print(f"Wrote {dynamics_sample_append_context_slide_layer_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
