import argparse
import copy
import itertools
import json
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

from webgpu_app.export.manifest import cache_contract, tensor_spec, version_or_unknown
from webgpu_app.export.onnx_artifacts import (
    ensure_output,
    export_file_metadata,
    external_data_path,
    remove_existing_export,
    sha256_file,
    snapshot_raw_artifacts,
)
from webgpu_app.export.validation import validate_outputs, validate_single_output
from webgpu_app.export.wasm_passes import (
    run_wasm_passes,
    wasm_pass_options_from_args,
)
from webgpu_app.export.webgpu_passes import (
    run_webgpu_passes,
    webgpu_pass_options_from_args,
)
from visionary.common.checkpoint import (
    resolve_model_export_step,
    restore_model_export_single_device,
)
from visionary.export.onnx_wrappers import (
    onnx_apply_dynamics_cached_sample_step_append_context_full_cache_entries,
    onnx_apply_dynamics_uncached,
    onnx_apply_tokenizer_decode_z,
    onnx_apply_tokenizer_decoder,
    dynamics_shapes,
    set_attention_export_layout,
    set_attention_export_lowering,
    tokenizer_shapes,
)


TOKENIZER_DECODER_NAME = "breakout_tokenizer_decoder_b1_t64"
TOKENIZER_DECODER_STEP_NAME = "breakout_tokenizer_decoder_b1_t1"
TOKENIZER_DECODE_Z_STEP_NAME = "breakout_tokenizer_decode_z_b1_t1"
DYNAMICS_UNCACHED_NAME = "breakout_dynamics_b1_t64"
MANIFEST_NAME = "breakout_onnx_manifest.json"


def sample_export_names(sample_steps: int) -> dict[str, str]:
    suffix = f"s{sample_steps}"
    return {
        "append_context_slide_entry": (
            f"breakout_dynamics_sample_append_context_slide_entry_b1_t1_{suffix}"
        ),
    }


def set_sample_export_names(sample_steps: int) -> None:
    global DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME

    names = sample_export_names(sample_steps)
    DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME = names["append_context_slide_entry"]


set_sample_export_names(4)


def split_wasm_dynamics_paths(source_path: Path) -> tuple[Path, Path]:
    stem = source_path.with_suffix("")
    return (
        source_path.with_name(f"{stem.name}_sample_only_final_z.onnx"),
        source_path.with_name(f"{stem.name}_context_entry_from_final_z.onnx"),
    )


def export_split_wasm_dynamics_models(source_path: Path, *, overwrite: bool) -> tuple[Path, Path]:
    sample_path, entry_path = split_wasm_dynamics_paths(source_path)
    ensure_output(sample_path, overwrite=overwrite)
    ensure_output(entry_path, overwrite=overwrite)
    onnx.utils.extract_model(
        str(source_path),
        str(sample_path),
        ["sample_noise", "actions", "k_cache", "v_cache"],
        ["final_z"],
        check_model=False,
    )
    onnx.utils.extract_model(
        str(source_path),
        str(entry_path),
        ["final_z", "context_noise", "actions", "k_cache", "v_cache"],
        ["candidate_k_entry", "candidate_v_entry"],
        check_model=False,
    )
    return sample_path, entry_path


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
            "any simplification, ORT optimization, or backend-specific graph rewrites. Use this "
            "with webgpu_app/export/compare_raw_optimized_onnx.py as a behavior-preserving "
            "optimization gate."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--opset", type=int, default=23)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_steps", type=int, default=2)
    parser.add_argument("--context_tau", type=float, default=29 / 32)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--export_target",
        choices=("webgpu", "wasm"),
        default="webgpu",
        help=(
            "Backend profile for post-export graph rewrites. The WebGPU target keeps "
            "the accepted WebGPU layout pipeline; the WASM target uses a separate "
            "pass pipeline for browser CPU execution."
        ),
    )
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
        "--skip_squeeze_concat_rewrite",
        action="store_true",
        help=(
            "Skip the WebGPU layout rewrite that factors repeated Squeeze nodes through "
            "Concat. Enabled by default because it removes hundreds of layout dispatches "
            "without reintroducing Reshape."
        ),
    )
    parser.add_argument(
        "--skip_unsqueeze_transpose_squeeze_rewrite",
        action="store_true",
        help=(
            "Skip the WebGPU layout rewrite that collapses Unsqueeze->Transpose->Squeeze "
            "chains into an equivalent lower-rank Transpose."
        ),
    )
    parser.add_argument(
        "--skip_spatial_qk_head_layout_rewrite",
        action="store_true",
        help=(
            "Skip the WebGPU layout rewrite that builds spatial Q/K RoPE inputs directly "
            "in B,H,S,D order and removes the matching Transpose wrappers."
        ),
    )
    parser.add_argument(
        "--skip_temporal_attention_bhsd_rewrite",
        action="store_true",
        help=(
            "Skip the WebGPU layout rewrite that feeds cached temporal attention in "
            "B,H,S,D order internally while preserving the exported cache-entry ABI."
        ),
    )
    parser.add_argument(
        "--skip_attention_scale_folding",
        action="store_true",
        help=(
            "Skip folding the attention score scale into query RMSNorm weights. Enabled "
            "by default to remove one logits-sized Mul before each attention Softmax."
        ),
    )
    parser.add_argument(
        "--head_projection_rewrite",
        choices=("einsum", "layout"),
        default="layout",
        help=(
            "How to remove attention head projection Reshape nodes. 'einsum' replaces "
            "Gemm+Reshape with rank-aware Einsum kernels; 'layout' keeps the Gemm kernels "
            "and uses static Split/Squeeze/Unsqueeze/Concat layout ops."
        ),
    )
    parser.add_argument(
        "--rotary_embedding_rewrite",
        dest="rotary_embedding_rewrite",
        action="store_true",
        default=True,
        help=(
            "Replace exported RoPE Split/Mul/Add/Sub/Concat islands with ORT WebGPU's "
            "contrib RotaryEmbedding op. Enabled by default for the browser WebGPU export."
        ),
    )
    parser.add_argument(
        "--skip_rotary_embedding_rewrite",
        dest="rotary_embedding_rewrite",
        action="store_false",
        help=(
            "Disable the accepted WebGPU RotaryEmbedding rewrite. This is mainly useful "
            "for controlled performance experiments."
        ),
    )
    parser.add_argument(
        "--pack_qkv_gemm",
        dest="pack_qkv_gemm",
        action="store_true",
        default=True,
        help=(
            "Pack sibling Q/K/V Gemm projections that share the same input into one wider "
            "Gemm followed by Split. Enabled by default for the browser WebGPU export."
        ),
    )
    parser.add_argument(
        "--skip_pack_qkv_gemm",
        dest="pack_qkv_gemm",
        action="store_false",
        help=(
            "Disable the accepted WebGPU Q/K/V Gemm packing pass. This is mainly useful "
            "for controlled performance experiments."
        ),
    )
    parser.add_argument(
        "--pack_qkv_head_projection",
        action="store_true",
        help=(
            "Experimental: replace sibling Q/K/V Gemm+head-Reshape projection groups "
            "with one packed rank-aware Einsum that directly emits head-ranked Q/K/V "
            "tensors. This removes per-head Split/Unsqueeze/Concat layout dispatches "
            "while preserving the original output tensor names."
        ),
    )
    parser.add_argument(
        "--pack_swiglu_gemm",
        dest="pack_swiglu_gemm",
        action="store_true",
        default=True,
        help=(
            "Pack sibling SwiGLU gate/value Gemm projections that share the same input "
            "into one wider Gemm followed by Split. Enabled by default for the browser "
            "WebGPU export."
        ),
    )
    parser.add_argument(
        "--skip_pack_swiglu_gemm",
        dest="pack_swiglu_gemm",
        action="store_false",
        help=(
            "Disable the accepted WebGPU SwiGLU Gemm packing pass. This is mainly useful "
            "for controlled performance experiments."
        ),
    )
    parser.add_argument(
        "--attention_lowering",
        choices=("manual", "native", "split_gqa"),
        default="manual",
        help=(
            "Attention lowering used during jax2onnx conversion. 'manual' uses the "
            "project's explicit einsum/softmax decomposition; 'native' leaves "
            "jax.nn.dot_product_attention for jax2onnx's built-in lowering; "
            "'split_gqa' avoids materializing repeated K/V heads by running one "
            "manual attention group per KV head."
        ),
    )
    parser.add_argument(
        "--attention_layout",
        choices=("bshd", "bnsh"),
        default="bshd",
        help=(
            "Export-only attention head layout. 'bshd' matches the training code's "
            "[batch, sequence, heads, dim] layout and is the accepted WebGPU demo path. "
            "'bnsh' keeps attention internals in [batch, heads, sequence, dim], but is "
            "experimental because it currently reintroduces CPU-only Reshape nodes."
        ),
    )
    parser.add_argument(
        "--fuse_gqa_attention",
        action="store_true",
        help=(
            "Experimental: post-export rewrite mask-free manual GQA attention islands "
            "into com.microsoft::GroupQueryAttention. This is intended for the fixed "
            "steady-state demo graph where the cache is full and no attention mask "
            "remains in the optimized ONNX graph."
        ),
    )
    parser.add_argument(
        "--fuse_spatial_gqa_attention",
        action="store_true",
        help=(
            "Experimental: also rewrite mask-free spatial attention islands into "
            "com.microsoft::GroupQueryAttention. Unlike cached temporal GQA, this "
            "uses no past-cache or seq_lens inputs, so the fused op is bidirectional."
        ),
    )
    parser.add_argument(
        "--fuse_mha_attention",
        action="store_true",
        help=(
            "Experimental: post-export rewrite mask-free manual attention islands "
            "into com.microsoft::MultiHeadAttention after K/V heads have already "
            "been materialized. This preserves behavior and tests whether ORT "
            "WebGPU's fused attention kernels beat the explicit Einsum/Softmax graph."
        ),
    )
    parser.add_argument(
        "--skip_wasm_mha_dynamics_fusion",
        dest="wasm_mha_dynamics_fusion",
        action="store_false",
        default=True,
        help=(
            "WASM export only: do not fuse the preferred full-cache dynamics step "
            "attention islands into com.microsoft::MultiHeadAttention."
        ),
    )
    parser.add_argument(
        "--wasm_mha_decoder_fusion",
        action="store_true",
        help=(
            "WASM export only: fuse matched single-frame decoder attention islands into "
            "com.microsoft::MultiHeadAttention. This includes masked BHQD islands when "
            "the attention bias can be passed through the fused op."
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
            "Export the browser hot path: single-frame decoder graphs and the full-cache "
            "dynamics entry graph."
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


def seeded_inputs(
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


def ensure_static_int32_output(path: Path, output_name: str, value: int) -> dict[str, Any]:
    model = onnx.load(path.as_posix(), load_external_data=True)
    graph_outputs = {output.name for output in model.graph.output}
    if output_name not in graph_outputs:
        return {
            "enabled": False,
            "reason": f"{output_name!r} is not a graph output",
        }

    produced = {output for node in model.graph.node for output in node.output}
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    if output_name in produced or output_name in initializer_names:
        return {
            "enabled": True,
            "changed": False,
            "reason": f"{output_name!r} already exists",
        }

    model.graph.initializer.append(
        onnx.numpy_helper.from_array(np.asarray([value], dtype=np.int32), name=output_name)
    )
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)
    return {
        "enabled": True,
        "changed": True,
        "output_name": output_name,
        "value": value,
    }


def op_counts(path: Path) -> Counter[str]:
    model = onnx.load(path.as_posix(), load_external_data=False)
    return Counter(node.op_type for node in model.graph.node)


def node_key(node: onnx.NodeProto) -> str:
    if node.name:
        return node.name
    if node.output:
        return node.output[0]
    return f"unnamed_{id(node)}"


def prune_graph_to_outputs(model: onnx.ModelProto) -> dict[str, int]:
    producer = {output: node for node in model.graph.node for output in node.output}
    required_values = {output.name for output in model.graph.output}
    required_values.update(input_value.name for input_value in model.graph.input)
    required_nodes: set[str] = set()
    pending = list(required_values)

    while pending:
        value_name = pending.pop()
        node = producer.get(value_name)
        if node is None:
            continue

        key = node_key(node)
        if key in required_nodes:
            continue
        required_nodes.add(key)
        for output_name in node.output:
            required_values.add(output_name)
        for input_name in node.input:
            if input_name and input_name not in required_values:
                required_values.add(input_name)
                pending.append(input_name)

    node_count_before = len(model.graph.node)
    initializer_count_before = len(model.graph.initializer)
    value_info_count_before = len(model.graph.value_info)

    kept_nodes = [node for node in model.graph.node if node_key(node) in required_nodes]
    kept_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in required_values
    ]
    graph_io = {value.name for value in model.graph.input}
    graph_io.update(value.name for value in model.graph.output)
    kept_value_info = [
        value
        for value in model.graph.value_info
        if value.name in required_values or value.name in graph_io
    ]

    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)

    return {
        "nodes_removed": node_count_before - len(kept_nodes),
        "initializers_removed": initializer_count_before - len(kept_initializers),
        "value_info_removed": value_info_count_before - len(kept_value_info),
    }


def rewrite_entry_final_z_only_for_webgpu(path: Path, enabled: bool = True) -> dict[str, Any]:
    before = op_counts(path)
    if not enabled:
        return {
            "enabled": False,
            "reason": "not the steady-state entry-cache demo artifact",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
        }

    model = onnx.load(path.as_posix(), load_external_data=True)
    output_names = [output.name for output in model.graph.output]
    if "pred_z" not in output_names:
        return {
            "enabled": True,
            "tool": "custom_final_z_only_prune",
            "reason": "Graph is already final_z-only.",
            "final_z_aliases_pred_z": "final_z" in output_names,
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "removed_outputs": [],
            "prune": {
                "nodes_removed": 0,
                "initializers_removed": 0,
                "value_info_removed": 0,
            },
        }
    if "final_z" not in output_names:
        return {
            "enabled": False,
            "tool": "custom_final_z_only_prune",
            "reason": "Graph has pred_z output but no final_z output to preserve.",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
        }

    producer = {output: node for node in model.graph.node for output in node.output}
    pred_node = producer.get("pred_z")
    final_node = producer.get("final_z")
    if pred_node is None or final_node is None:
        return {
            "enabled": False,
            "tool": "custom_final_z_only_prune",
            "reason": "Could not find producers for both pred_z and final_z.",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
        }
    if pred_node is final_node:
        return {
            "enabled": False,
            "tool": "custom_final_z_only_prune",
            "reason": "pred_z and final_z are produced by the same node; refusing ambiguous rewrite.",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
        }

    discarded_final_z = "final_z__discarded_before_pred_alias"
    for idx, output_name in enumerate(final_node.output):
        if output_name == "final_z":
            final_node.output[idx] = discarded_final_z
    for idx, output_name in enumerate(pred_node.output):
        if output_name == "pred_z":
            pred_node.output[idx] = "final_z"

    kept_outputs = [output for output in model.graph.output if output.name != "pred_z"]
    del model.graph.output[:]
    model.graph.output.extend(kept_outputs)
    kept_value_info = [
        value for value in model.graph.value_info if value.name not in {"pred_z", discarded_final_z}
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)

    prune_result = prune_graph_to_outputs(model)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Unsqueeze", "Squeeze", "Concat", "Gemm", "Einsum", "Transpose", "Identity")
    return {
        "enabled": True,
        "tool": "custom_final_z_only_prune",
        "reason": (
            "Expose the sampled pred_z producer as final_z for the steady-state demo "
            "graph, remove the redundant pred_z graph output, and prune the old "
            "terminal final_z branch without inserting Identity nodes."
        ),
        "final_z_aliases_pred_z": True,
        "removed_outputs": ["pred_z"],
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "prune": prune_result,
    }


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
            static_cache_length_name = f"{node.output[0]}__static_value"
            add_initializer_once(np.asarray([64], dtype=np.int32), static_cache_length_name)
            rewritten_nodes.append(
                onnx.helper.make_node(
                    "Identity",
                    [static_cache_length_name],
                    [node.output[0]],
                    name=f"{node.name or node.output[0]}__static_identity",
                )
            )
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


def _set_tensor_shape(value_info: onnx.ValueInfoProto, shape: tuple[int, ...]) -> None:
    dims = value_info.type.tensor_type.shape.dim
    del dims[:]
    for dim in shape:
        dims.add().dim_value = int(dim)


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
    external_data_path(path).unlink(missing_ok=True)
    # ONNX checker does not know ORT's SimplifiedLayerNormalization schema.
    # ORT validation is run separately by the export/accuracy checks.
    if not any(node.op_type == "SimplifiedLayerNormalization" for node in model.graph.node):
        onnx.checker.check_model(model)
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


def rewrite_squeeze_concat_for_webgpu(path: Path, enabled: bool = True) -> dict[str, Any]:
    before = op_counts(path)
    if not enabled:
        return {
            "enabled": False,
            "reason": "--skip_squeeze_concat_rewrite",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
        }

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
    initializers = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    producer = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    def node_key(node: onnx.NodeProto) -> str:
        return node.name or node.output[0]

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def squeeze_axes(node: onnx.NodeProto) -> tuple[int, ...] | None:
        if node.op_type != "Squeeze":
            return None
        if len(node.input) >= 2 and node.input[1] in initializers:
            return tuple(int(axis) for axis in np.asarray(initializers[node.input[1]]).reshape(-1))
        axes = attr_value(node, "axes", None)
        if axes is None:
            return None
        return tuple(int(axis) for axis in axes)

    replacements: dict[str, list[onnx.NodeProto]] = {}
    skip_nodes: set[str] = set()
    new_initializers: list[onnx.TensorProto] = []
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    for concat in model.graph.node:
        if concat.op_type != "Concat" or len(concat.input) < 2 or len(concat.output) != 1:
            continue
        if node_key(concat) in replacements:
            continue

        squeezes = [producer.get(input_name) for input_name in concat.input]
        if any(node is None or node.op_type != "Squeeze" for node in squeezes):
            continue
        if any(len(consumers.get(squeeze.output[0], [])) != 1 for squeeze in squeezes if squeeze):
            continue

        axes = squeeze_axes(squeezes[0])
        if axes is None:
            continue
        input_shapes = [value_shapes.get(squeeze.input[0]) for squeeze in squeezes if squeeze]
        output_shapes = [value_shapes.get(squeeze.output[0]) for squeeze in squeezes if squeeze]
        concat_output_shape = value_shapes.get(concat.output[0])
        if (
            any(shape is None for shape in input_shapes)
            or any(shape is None for shape in output_shapes)
            or concat_output_shape is None
        ):
            continue

        input_rank = len(input_shapes[0])
        normalized_axes = tuple(sorted(axis if axis >= 0 else input_rank + axis for axis in axes))
        if any(axis < 0 or axis >= input_rank for axis in normalized_axes):
            continue
        if any(shape[axis] != 1 for shape in input_shapes for axis in normalized_axes):
            continue
        expected_squeezed = tuple(
            dim for axis, dim in enumerate(input_shapes[0]) if axis not in normalized_axes
        )
        if any(shape != expected_squeezed for shape in output_shapes):
            continue
        if any(squeeze_axes(squeeze) != axes for squeeze in squeezes[1:] if squeeze):
            continue

        squeezed_rank = len(expected_squeezed)
        concat_axis = int(attr_value(concat, "axis", 0))
        concat_axis = concat_axis if concat_axis >= 0 else squeezed_rank + concat_axis
        if concat_axis < 0 or concat_axis >= squeezed_rank:
            continue
        surviving_axes = [axis for axis in range(input_rank) if axis not in normalized_axes]
        original_concat_axis = surviving_axes[concat_axis]

        first_shape = input_shapes[0]
        if any(
            any(
                dim != first_shape[axis]
                for axis, dim in enumerate(shape)
                if axis != original_concat_axis
            )
            for shape in input_shapes[1:]
        ):
            continue

        expected_unsqueezed_output = list(first_shape)
        expected_unsqueezed_output[original_concat_axis] = sum(
            shape[original_concat_axis] for shape in input_shapes
        )
        if (
            tuple(
                dim
                for axis, dim in enumerate(expected_unsqueezed_output)
                if axis not in normalized_axes
            )
            != concat_output_shape
        ):
            continue

        prefix = concat.name or concat.output[0]
        concat_before_squeeze = f"{prefix}__pre_squeeze_concat"
        axes_name = f"{prefix}__factored_squeeze_axes"
        new_initializers.append(
            onnx.numpy_helper.from_array(np.asarray(normalized_axes, dtype=np.int64), axes_name)
        )
        replacements[node_key(concat)] = [
            onnx.helper.make_node(
                "Concat",
                [squeeze.input[0] for squeeze in squeezes if squeeze],
                [concat_before_squeeze],
                name=f"{prefix}__factored_concat",
                axis=original_concat_axis,
            ),
            onnx.helper.make_node(
                "Squeeze",
                [concat_before_squeeze, axes_name],
                [concat.output[0]],
                name=f"{prefix}__factored_squeeze",
            ),
        ]
        skip_nodes.update(node_key(squeeze) for squeeze in squeezes if squeeze)
        rewrites["squeeze_through_concat"] += 1
        rewrites["removed_squeeze_nodes"] += len(squeezes) - 1
        if len(examples) < 12:
            examples.append(
                {
                    "concat": concat.name,
                    "inputs": len(concat.input),
                    "squeeze_axes": list(normalized_axes),
                    "old_concat_axis": concat_axis,
                    "new_concat_axis": original_concat_axis,
                    "input_shape": list(input_shapes[0]),
                    "output_shape": list(concat_output_shape),
                }
            )

    if rewrites:
        rewritten_nodes: list[onnx.NodeProto] = []
        for node in model.graph.node:
            key = node_key(node)
            if key in skip_nodes:
                continue
            if key in replacements:
                rewritten_nodes.extend(replacements[key])
            else:
                rewritten_nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)
        model.graph.initializer.extend(new_initializers)
        external_data_path(path).unlink(missing_ok=True)
        onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Squeeze", "Unsqueeze", "Split", "Concat", "Transpose", "Reshape")
    return {
        "enabled": True,
        "tool": "custom_squeeze_concat_rewrite",
        "reason": (
            "Factor repeated Squeeze inputs through Concat. This preserves the same "
            "static layout view while replacing N per-head Squeeze ops with one "
            "post-Concat Squeeze."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
    }


def rewrite_unsqueeze_transpose_squeeze_for_webgpu(
    path: Path, enabled: bool = True
) -> dict[str, Any]:
    before = op_counts(path)
    if not enabled:
        return {
            "enabled": False,
            "reason": "--skip_unsqueeze_transpose_squeeze_rewrite",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
        }

    model = onnx.load(path.as_posix(), load_external_data=True)
    initializers = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    producer = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    def node_key(node: onnx.NodeProto) -> str:
        return node.name or node.output[0]

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def axes_value(node: onnx.NodeProto) -> tuple[int, ...] | None:
        if len(node.input) >= 2 and node.input[1] in initializers:
            return tuple(int(axis) for axis in np.asarray(initializers[node.input[1]]).reshape(-1))
        axes = attr_value(node, "axes", None)
        if axes is None:
            return None
        return tuple(int(axis) for axis in axes)

    replacements: dict[str, list[onnx.NodeProto]] = {}
    skip_nodes: set[str] = set()
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    for squeeze in model.graph.node:
        if squeeze.op_type != "Squeeze" or len(squeeze.input) == 0 or len(squeeze.output) != 1:
            continue
        transpose = producer.get(squeeze.input[0])
        if transpose is None or transpose.op_type != "Transpose" or len(transpose.input) != 1:
            continue
        unsqueeze = producer.get(transpose.input[0])
        if (
            unsqueeze is None
            or unsqueeze.op_type != "Unsqueeze"
            or len(unsqueeze.input) == 0
            or len(unsqueeze.output) != 1
        ):
            continue
        if len(consumers.get(unsqueeze.output[0], [])) != 1:
            continue
        if len(consumers.get(transpose.output[0], [])) != 1:
            continue

        unsqueeze_axes = axes_value(unsqueeze)
        squeeze_axes = axes_value(squeeze)
        perm = tuple(int(axis) for axis in attr_value(transpose, "perm", ()))
        if unsqueeze_axes != (0,) or squeeze_axes != (0,) or not perm:
            continue
        if perm[0] != 0 or sorted(perm) != list(range(len(perm))):
            continue

        replacement_perm = [axis - 1 for axis in perm[1:]]
        if any(axis < 0 for axis in replacement_perm):
            continue

        prefix = squeeze.name or squeeze.output[0]
        replacements[node_key(squeeze)] = [
            onnx.helper.make_node(
                "Transpose",
                [unsqueeze.input[0]],
                [squeeze.output[0]],
                name=f"{prefix}__collapsed_transpose",
                perm=replacement_perm,
            )
        ]
        skip_nodes.add(node_key(unsqueeze))
        skip_nodes.add(node_key(transpose))
        rewrites["unsqueeze_transpose_squeeze_to_transpose"] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "unsqueeze": unsqueeze.name,
                    "transpose": transpose.name,
                    "squeeze": squeeze.name,
                    "old_perm": list(perm),
                    "new_perm": replacement_perm,
                }
            )

    if rewrites:
        rewritten_nodes: list[onnx.NodeProto] = []
        for node in model.graph.node:
            key = node_key(node)
            if key in skip_nodes:
                continue
            if key in replacements:
                rewritten_nodes.extend(replacements[key])
            else:
                rewritten_nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)
        external_data_path(path).unlink(missing_ok=True)
        onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Unsqueeze", "Transpose", "Squeeze", "Reshape")
    return {
        "enabled": True,
        "tool": "custom_unsqueeze_transpose_squeeze_rewrite",
        "reason": (
            "Collapse Unsqueeze(axis=0)->Transpose(0,...)->Squeeze(axis=0) chains "
            "into a lower-rank Transpose with the inserted axis removed."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
    }


def rewrite_spatial_qk_head_layout_for_webgpu(path: Path, enabled: bool = True) -> dict[str, Any]:
    before = op_counts(path)
    if not enabled:
        return {
            "enabled": False,
            "reason": "--skip_spatial_qk_head_layout_rewrite",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
        }

    model = onnx.load(path.as_posix(), load_external_data=True)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    producer = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    def node_key(node: onnx.NodeProto) -> str:
        return node.name or node.output[0]

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def equation(node: onnx.NodeProto) -> str | None:
        value = attr_value(node, "equation", None)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    def set_equation(node: onnx.NodeProto, value: str) -> None:
        for attr in node.attribute:
            if attr.name == "equation":
                attr.s = value.encode("utf-8")
                return
        node.attribute.extend([onnx.helper.make_attribute("equation", value)])

    def set_axis_attr(node: onnx.NodeProto, axis: int) -> None:
        for attr in node.attribute:
            if attr.name == "axis":
                attr.i = axis
                return
        node.attribute.extend([onnx.helper.make_attribute("axis", axis)])

    def axes_value(node: onnx.NodeProto) -> tuple[int, ...] | None:
        if len(node.input) >= 2 and node.input[1] in initializers:
            value = onnx.numpy_helper.to_array(initializers[node.input[1]])
            return tuple(int(axis) for axis in np.asarray(value).reshape(-1))
        axes = attr_value(node, "axes", None)
        if axes is None:
            return None
        return tuple(int(axis) for axis in axes)

    def replace_axes_initializer(name: str, axes: tuple[int, ...]) -> None:
        initializers[name].CopyFrom(
            onnx.numpy_helper.from_array(np.asarray(axes, dtype=np.int64), name)
        )

    rewrites = Counter()
    examples: list[dict[str, Any]] = []
    skip_nodes: set[str] = set()
    stale_value_info: set[str] = set()

    for transpose in model.graph.node:
        if (
            transpose.op_type != "Transpose"
            or len(transpose.input) != 1
            or len(transpose.output) != 1
        ):
            continue
        if tuple(int(axis) for axis in attr_value(transpose, "perm", ())) != (0, 2, 1, 3):
            continue

        rotary_consumers = consumers.get(transpose.output[0], [])
        if len(rotary_consumers) != 1 or rotary_consumers[0].op_type != "RotaryEmbedding":
            continue
        rotary = rotary_consumers[0]

        post_transpose = None
        post_consumer = None
        rotary_output_consumers = consumers.get(rotary.output[0], [])
        if len(rotary_output_consumers) == 1 and rotary_output_consumers[0].op_type == "Transpose":
            post_transpose = rotary_output_consumers[0]
            if tuple(int(axis) for axis in attr_value(post_transpose, "perm", ())) != (
                0,
                2,
                1,
                3,
            ):
                continue
            post_consumers = consumers.get(post_transpose.output[0], [])
            if len(post_consumers) != 1:
                continue
            post_consumer = post_consumers[0]
            is_spatial_q = (
                post_consumer.op_type == "Einsum"
                and equation(post_consumer) == "bqhd,bkhd->bhqk"
                and post_consumer.input[0] == post_transpose.output[0]
            )
            is_spatial_k = (
                post_consumer.op_type == "Gather"
                and int(attr_value(post_consumer, "axis", -1)) == 2
                and post_consumer.input[0] == post_transpose.output[0]
            )
            if not (is_spatial_q or is_spatial_k):
                continue
        elif any(consumer.op_type == "Transpose" for consumer in rotary_output_consumers):
            # Other post-RoPE transpose patterns are temporal/cache sites; those are
            # handled by rewrite_temporal_attention_bhsd_for_webgpu().
            continue

        norm = producer.get(transpose.input[0])
        if norm is None or norm.op_type != "SimplifiedLayerNormalization":
            continue
        if len(consumers.get(norm.output[0], [])) != 1:
            continue

        concat = producer.get(norm.input[0])
        if concat is None or concat.op_type != "Concat" or len(concat.output) != 1:
            continue
        if int(attr_value(concat, "axis", -1)) != 2:
            continue
        if len(consumers.get(concat.output[0], [])) != 1:
            continue

        unsqueezes = [producer.get(input_name) for input_name in concat.input]
        if any(node is None or node.op_type != "Unsqueeze" for node in unsqueezes):
            continue
        if any(
            len(consumers.get(node.output[0], [])) != 1
            or len(node.input) < 2
            or node.input[1] not in initializers
            for node in unsqueezes
            if node is not None
        ):
            continue
        if any(axes_value(node) != (0, 2) for node in unsqueezes if node is not None):
            continue

        for unsqueeze in unsqueezes:
            if unsqueeze is not None:
                replace_axes_initializer(unsqueeze.input[1], (0, 1))
        set_axis_attr(concat, 1)
        rotary.input[0] = norm.output[0]
        skip_nodes.add(node_key(transpose))
        if post_transpose is not None and post_consumer is not None:
            if post_consumer.op_type == "Einsum":
                post_consumer.input[0] = rotary.output[0]
                set_equation(post_consumer, "bhqd,bhkd->bhqk")
                rewrites["spatial_q_direct_bhsd"] += 1
            elif post_consumer.op_type == "Gather":
                post_consumer.input[0] = rotary.output[0]
                set_axis_attr(post_consumer, 1)
                stale_value_info.update(post_consumer.output)
                rewrites["spatial_k_direct_bhsd"] += 1
            skip_nodes.add(node_key(post_transpose))
            stale_value_info.update(post_transpose.output)
        else:
            rewrites["spatial_qk_direct_bhsd"] += 1
        stale_value_info.update({concat.output[0], norm.output[0], transpose.output[0]})

        if len(examples) < 12:
            examples.append(
                {
                    "transpose": transpose.name,
                    "rotary": rotary.name,
                    "post_transpose": post_transpose.name if post_transpose is not None else None,
                    "post_consumer": post_consumer.name if post_consumer is not None else None,
                    "post_consumer_op": post_consumer.op_type
                    if post_consumer is not None
                    else None,
                    "concat": concat.name,
                    "heads": len(concat.input),
                    "old_unsqueeze_axes": [0, 2],
                    "new_unsqueeze_axes": [0, 1],
                    "old_concat_axis": 2,
                    "new_concat_axis": 1,
                }
            )

    if rewrites:
        kept_nodes = [node for node in model.graph.node if node_key(node) not in skip_nodes]
        kept_value_info = [
            value for value in model.graph.value_info if value.name not in stale_value_info
        ]
        del model.graph.node[:]
        model.graph.node.extend(kept_nodes)
        del model.graph.value_info[:]
        model.graph.value_info.extend(kept_value_info)
        external_data_path(path).unlink(missing_ok=True)
        onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Transpose", "RotaryEmbedding", "Concat", "Unsqueeze", "Einsum", "Reshape")
    return {
        "enabled": True,
        "tool": "custom_spatial_qk_head_layout_rewrite",
        "reason": (
            "Build spatial Q/K head tensors directly in B,H,S,D order before "
            "RotaryEmbedding, removing Transpose wrappers while preserving attention "
            "inputs and outputs."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
    }


def rewrite_temporal_attention_bhsd_for_webgpu(path: Path, enabled: bool = True) -> dict[str, Any]:
    before = op_counts(path)
    if not enabled:
        return {
            "enabled": False,
            "reason": "--skip_temporal_attention_bhsd_rewrite",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
        }

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
    producer = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    def node_key(node: onnx.NodeProto) -> str:
        return node.name or node.output[0]

    graph_outputs = {output.name for output in model.graph.output}

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def set_axis_attr(node: onnx.NodeProto, axis: int) -> None:
        for attr in node.attribute:
            if attr.name == "axis":
                attr.i = axis
                return
        node.attribute.extend([onnx.helper.make_attribute("axis", axis)])

    def equation(node: onnx.NodeProto) -> str | None:
        value = attr_value(node, "equation", None)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    def set_equation(node: onnx.NodeProto, value: str) -> None:
        for attr in node.attribute:
            if attr.name == "equation":
                attr.s = value.encode("utf-8")
                return
        node.attribute.extend([onnx.helper.make_attribute("equation", value)])

    def has_perm(node: onnx.NodeProto, perm: tuple[int, ...]) -> bool:
        return tuple(int(axis) for axis in attr_value(node, "perm", ())) == perm

    def make_transpose(
        input_name: str,
        output_name: str,
        owner: onnx.NodeProto,
        suffix: str,
    ) -> onnx.NodeProto:
        owner_name = node_key(owner)
        return onnx.helper.make_node(
            "Transpose",
            [input_name],
            [output_name],
            name=f"{owner_name}__temporal_bhsd_{suffix}_transpose",
            perm=[0, 2, 1, 3],
        )

    def trace_temporal_rope_output(
        output_name: str,
    ) -> tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto, onnx.NodeProto] | None:
        post_transpose = producer.get(output_name)
        if (
            post_transpose is None
            or post_transpose.op_type != "Transpose"
            or not has_perm(post_transpose, (0, 2, 1, 3))
            or len(post_transpose.input) != 1
        ):
            return None
        rotary = producer.get(post_transpose.input[0])
        if rotary is None or rotary.op_type != "RotaryEmbedding" or len(rotary.input) < 1:
            return None
        pre_transpose = producer.get(rotary.input[0])
        if (
            pre_transpose is None
            or pre_transpose.op_type != "Transpose"
            or not has_perm(pre_transpose, (0, 2, 1, 3))
            or len(pre_transpose.input) != 1
        ):
            return None
        norm = producer.get(pre_transpose.input[0])
        if norm is None or norm.op_type != "SimplifiedLayerNormalization":
            return None
        concat = producer.get(norm.input[0])
        if concat is None or concat.op_type != "Concat" or int(attr_value(concat, "axis", -1)) != 2:
            return None
        input_shape = value_shapes.get(concat.output[0])
        if input_shape is None or len(input_shape) != 4:
            return None
        if input_shape[1] != 1 or input_shape[2] not in (2, 8):
            return None
        return post_transpose, rotary, pre_transpose, concat

    def find_temporal_kv_inputs(
        gather: onnx.NodeProto,
    ) -> tuple[onnx.NodeProto, str, str, onnx.NodeProto] | None:
        if gather.op_type != "Gather" or len(gather.input) < 1:
            return None
        if int(attr_value(gather, "axis", -1)) != 2:
            return None
        concat = producer.get(gather.input[0])
        if (
            concat is None
            or concat.op_type != "Concat"
            or int(attr_value(concat, "axis", -1)) != 1
            or len(concat.input) != 2
        ):
            return None
        current_inputs = [
            input_name
            for input_name in concat.input
            if value_shapes.get(input_name, (None, None, None, None))[1] == 1
        ]
        if len(current_inputs) != 1:
            return None
        current_input = current_inputs[0]
        cache_input = concat.input[1] if concat.input[0] == current_input else concat.input[0]
        if value_shapes.get(cache_input) is None:
            return None
        return concat, cache_input, current_input, gather

    def single_attention_value(score: onnx.NodeProto) -> onnx.NodeProto | None:
        softmax_consumers = [
            node
            for node in consumers.get(score.output[0], [])
            if node.op_type == "Softmax" and len(node.output) == 1
        ]
        if len(softmax_consumers) != 1:
            return None
        value_nodes = [
            node
            for node in consumers.get(softmax_consumers[0].output[0], [])
            if node.op_type == "Einsum"
            and equation(node) == "bhqk,bkhd->bqhd"
            and len(node.input) >= 2
        ]
        if len(value_nodes) != 1:
            return None
        return value_nodes[0]

    new_nodes_before: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    skip_nodes: set[str] = set()
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    for score in model.graph.node:
        if (
            score.op_type != "Einsum"
            or equation(score) != "bqhd,bkhd->bhqk"
            or len(score.input) < 2
        ):
            continue

        q_trace = trace_temporal_rope_output(score.input[0])
        if q_trace is None:
            continue
        q_post, q_rotary, q_pre, q_concat = q_trace

        k_gather = producer.get(score.input[1])
        if k_gather is None:
            continue
        k_inputs = find_temporal_kv_inputs(k_gather)
        if k_inputs is None:
            continue
        k_concat, k_cache, k_current, k_gather = k_inputs
        k_trace = trace_temporal_rope_output(k_current)
        if k_trace is None:
            continue
        k_post, k_rotary, k_pre, k_head_concat = k_trace

        value = single_attention_value(score)
        if value is None:
            continue
        v_gather = producer.get(value.input[1])
        if v_gather is None:
            continue
        v_inputs = find_temporal_kv_inputs(v_gather)
        if v_inputs is None:
            continue
        v_concat, v_cache, v_current, v_gather = v_inputs
        v_head_concat = producer.get(v_current)
        if (
            v_head_concat is None
            or v_head_concat.op_type != "Concat"
            or int(attr_value(v_head_concat, "axis", -1)) != 2
        ):
            continue

        # Q only feeds the temporal attention island, so both layout transposes
        # around RoPE can be removed.
        set_axis_attr(q_concat, 1)
        q_rotary.input[0] = q_pre.input[0]
        score.input[0] = q_rotary.output[0]
        skip_nodes.update({node_key(q_pre), node_key(q_post)})

        # K keeps the public B,S,H,D cache-entry output intact. The attention
        # path consumes the RoPE output before the post-RoPE transpose.
        set_axis_attr(k_head_concat, 1)
        k_rotary.input[0] = k_pre.input[0]
        k_concat.input[list(k_concat.input).index(k_current)] = k_rotary.output[0]
        skip_nodes.add(node_key(k_pre))

        # V has no RoPE. If the current V tensor is only used by this attention
        # concat, switch it directly to B,H,S,D. Otherwise add an attention-only
        # transpose so public outputs keep their original layout.
        if len(consumers.get(v_current, [])) == 1:
            set_axis_attr(v_head_concat, 1)
            v_attention_current = v_current
        else:
            v_attention_current = f"{node_key(value)}__temporal_bhsd_current_v_out"
            new_nodes_before[node_key(v_concat)].append(
                make_transpose(
                    v_current,
                    v_attention_current,
                    owner=value,
                    suffix="current_v",
                )
            )
            rewrites["current_v_transpose_inserted"] += 1
        v_concat.input[list(v_concat.input).index(v_current)] = v_attention_current

        k_cache_bhsd = f"{node_key(score)}__temporal_bhsd_cache_k_out"
        v_cache_bhsd = f"{node_key(value)}__temporal_bhsd_cache_v_out"
        new_nodes_before[node_key(k_concat)].append(
            make_transpose(k_cache, k_cache_bhsd, owner=score, suffix="cache_k")
        )
        new_nodes_before[node_key(v_concat)].append(
            make_transpose(v_cache, v_cache_bhsd, owner=value, suffix="cache_v")
        )
        k_concat.input[list(k_concat.input).index(k_cache)] = k_cache_bhsd
        v_concat.input[list(v_concat.input).index(v_cache)] = v_cache_bhsd

        set_axis_attr(k_concat, 2)
        set_axis_attr(v_concat, 2)
        set_axis_attr(k_gather, 1)
        set_axis_attr(v_gather, 1)
        set_equation(score, "bhqd,bhkd->bhqk")
        set_equation(value, "bhqk,bhkd->bqhd")

        rewrites["temporal_attention_bhsd"] += 1
        rewrites["cache_transpose_inserted"] += 2
        if len(examples) < 12:
            examples.append(
                {
                    "score_einsum": score.name,
                    "value_einsum": value.name,
                    "q_pre_transpose": q_pre.name,
                    "q_post_transpose": q_post.name,
                    "k_pre_transpose": k_pre.name,
                    "k_post_transpose_preserved": k_post.name,
                    "k_concat": k_concat.name,
                    "v_concat": v_concat.name,
                    "inserted_current_v_transpose": v_attention_current != v_current,
                }
            )

    if rewrites:
        used_input_names: set[str] = set(graph_outputs)
        for node in model.graph.node:
            if node_key(node) in skip_nodes:
                continue
            used_input_names.update(input_name for input_name in node.input if input_name)
        skip_nodes = {
            node_key(node)
            for node in model.graph.node
            if node_key(node) in skip_nodes
            and all(output not in used_input_names for output in node.output)
            and all(output not in graph_outputs for output in node.output)
        }

        rewritten_nodes: list[onnx.NodeProto] = []
        for node in model.graph.node:
            key = node_key(node)
            rewritten_nodes.extend(new_nodes_before.get(key, []))
            if key not in skip_nodes:
                rewritten_nodes.append(node)

        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)
        del model.graph.value_info[:]
        external_data_path(path).unlink(missing_ok=True)
        onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Transpose", "Reshape", "Einsum", "Gather", "Concat", "Squeeze")
    return {
        "enabled": True,
        "tool": "custom_temporal_attention_bhsd_rewrite",
        "reason": (
            "Run cached temporal attention internally in B,H,S,D layout to remove "
            "pre/post RoPE layout transposes while preserving public cache-entry outputs."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
    }


def rewrite_entry_cache_io_bhntd_for_webgpu(path: Path, enabled: bool = True) -> dict[str, Any]:
    before = op_counts(path)
    if not enabled:
        return {
            "enabled": False,
            "reason": "entry-cache BHNTD cache ABI rewrite disabled",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
        }

    model = onnx.load(path.as_posix(), load_external_data=True)
    graph_inputs = {value.name: value for value in model.graph.input}
    if "k_cache" not in graph_inputs or "v_cache" not in graph_inputs:
        return {
            "enabled": False,
            "tool": "custom_entry_cache_io_bhntd_rewrite",
            "reason": "graph does not expose monolithic k_cache/v_cache inputs",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
        }

    k_shape = _tensor_shape(graph_inputs["k_cache"])
    v_shape = _tensor_shape(graph_inputs["v_cache"])
    if k_shape is None or v_shape is None or len(k_shape) != 6 or k_shape != v_shape:
        return {
            "enabled": False,
            "tool": "custom_entry_cache_io_bhntd_rewrite",
            "reason": f"unsupported cache input shapes k={k_shape} v={v_shape}",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
        }

    layers, batch, tokens, context_length, heads, head_dim = k_shape
    if context_length <= heads:
        return {
            "enabled": False,
            "tool": "custom_entry_cache_io_bhntd_rewrite",
            "reason": f"cache shape already appears head-major or ambiguous: {k_shape}",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
        }

    new_cache_shape = (layers, batch, tokens, heads, context_length, head_dim)
    for name in ("k_cache", "v_cache"):
        _set_tensor_shape(graph_inputs[name], new_cache_shape)

    initializer_by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    initializer_arrays = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    def node_key(node: onnx.NodeProto) -> str:
        return node.name or node.output[0]

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def has_perm(node: onnx.NodeProto, perm: tuple[int, ...]) -> bool:
        return tuple(int(axis) for axis in attr_value(node, "perm", ())) == perm

    def replace_initializer(name: str, values: np.ndarray) -> None:
        initializer = initializer_by_name[name]
        initializer.CopyFrom(onnx.numpy_helper.from_array(values.astype(np.int64), name))

    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    for node in model.graph.node:
        if (
            node.op_type != "Slice"
            or len(node.input) < 4
            or node.input[0] not in {"k_cache", "v_cache"}
        ):
            continue
        starts_name, ends_name, axes_name = node.input[1], node.input[2], node.input[3]
        if not all(name in initializer_arrays for name in (starts_name, ends_name, axes_name)):
            continue
        axes = np.asarray(initializer_arrays[axes_name]).reshape(-1)
        starts = np.asarray(initializer_arrays[starts_name]).reshape(-1).copy()
        ends = np.asarray(initializer_arrays[ends_name]).reshape(-1).copy()
        if tuple(int(axis) for axis in axes) != (0, 1, 2, 3, 4, 5):
            continue
        old_starts = starts.copy()
        old_ends = ends.copy()
        starts[3], starts[4] = old_starts[4], old_starts[3]
        ends[3], ends[4] = old_ends[4], old_ends[3]
        replace_initializer(starts_name, starts)
        replace_initializer(ends_name, ends)
        rewrites["cache_slice_bounds_swapped"] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "slice": node.name,
                    "input": node.input[0],
                    "old_ends": old_ends.astype(int).tolist(),
                    "new_ends": ends.astype(int).tolist(),
                }
            )

    output_replacements: dict[str, str] = {}
    skip_nodes: set[str] = set()
    for node in model.graph.node:
        if (
            node.op_type != "Transpose"
            or not has_perm(node, (0, 2, 1, 3))
            or "__temporal_bhsd_cache_" not in node.name
            or len(node.input) != 1
            or len(node.output) != 1
        ):
            continue
        output_replacements[node.output[0]] = node.input[0]
        skip_nodes.add(node_key(node))
        rewrites["cache_transpose_removed"] += 1

    if not rewrites:
        return {
            "enabled": True,
            "tool": "custom_entry_cache_io_bhntd_rewrite",
            "reason": "no eligible cache slices/transposes found",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
        }

    for node in model.graph.node:
        if node_key(node) in skip_nodes:
            continue
        for idx, input_name in enumerate(node.input):
            if input_name in output_replacements:
                node.input[idx] = output_replacements[input_name]

    rewritten_nodes = [node for node in model.graph.node if node_key(node) not in skip_nodes]
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    # Shapes after this ABI rewrite are intentionally different from the raw
    # jax2onnx graph. Drop stale inferred value_info and let ORT infer locally.
    del model.graph.value_info[:]
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Transpose", "Reshape", "Einsum", "Gather", "Concat", "Squeeze")
    return {
        "enabled": True,
        "tool": "custom_entry_cache_io_bhntd_rewrite",
        "reason": (
            "Expose steady-state cache inputs as [layer,batch,token,head,time,dim] "
            "so cached temporal attention consumes cache slices in B,H,S,D layout "
            "without per-layer cache transposes."
        ),
        "cache_layout": "layer_batch_token_head_time_dim",
        "old_cache_shape": list(k_shape),
        "new_cache_shape": list(new_cache_shape),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
    }


def rewrite_prefill_cache_outputs_bhntd_for_webgpu(
    path: Path, enabled: bool = True
) -> dict[str, Any]:
    before = op_counts(path)
    if not enabled:
        return {
            "enabled": False,
            "reason": "prefill BHNTD cache output rewrite disabled",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
        }

    model = onnx.load(path.as_posix(), load_external_data=True)
    graph_outputs = {value.name: value for value in model.graph.output}
    producer = {output: node for node in model.graph.node for output in node.output}
    rewrites = Counter()
    examples: list[dict[str, Any]] = []
    new_nodes: list[onnx.NodeProto] = []

    for output_name in ("k_cache", "v_cache"):
        output = graph_outputs.get(output_name)
        if output is None:
            continue
        shape = _tensor_shape(output)
        if shape is None or len(shape) != 6:
            continue
        layers, batch, tokens, context_length, heads, head_dim = shape
        if context_length <= heads:
            continue
        node = producer.get(output_name)
        if node is None:
            continue
        hidden_output = f"{output_name}__bnt_hd_before_bhntd"
        for idx, node_output in enumerate(node.output):
            if node_output == output_name:
                node.output[idx] = hidden_output
                break
        new_shape = (layers, batch, tokens, heads, context_length, head_dim)
        new_nodes.append(
            onnx.helper.make_node(
                "Transpose",
                [hidden_output],
                [output_name],
                name=f"{output_name}__prefill_cache_to_bhntd",
                perm=[0, 1, 2, 4, 3, 5],
            )
        )
        _set_tensor_shape(output, new_shape)
        rewrites["prefill_cache_output_transpose"] += 1
        if len(examples) < 4:
            examples.append(
                {
                    "output": output_name,
                    "old_shape": list(shape),
                    "new_shape": list(new_shape),
                }
            )

    if not rewrites:
        return {
            "enabled": True,
            "tool": "custom_prefill_cache_outputs_bhntd_rewrite",
            "reason": "no eligible prefill cache outputs found",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
        }

    model.graph.node.extend(new_nodes)
    del model.graph.value_info[:]
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Transpose", "Reshape", "Einsum", "Gather", "Concat", "Squeeze")
    return {
        "enabled": True,
        "tool": "custom_prefill_cache_outputs_bhntd_rewrite",
        "reason": (
            "Expose prefill cache outputs as [layer,batch,token,head,time,dim] "
            "so the browser cache object can feed the steady-state entry graph directly."
        ),
        "cache_layout": "layer_batch_token_head_time_dim",
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
    }


def fold_attention_scale_into_query_norm_for_webgpu(
    path: Path, enabled: bool = True
) -> dict[str, Any]:
    before = op_counts(path)
    if not enabled:
        return {
            "enabled": False,
            "reason": "--skip_attention_scale_folding",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
        }

    model = onnx.load(path.as_posix(), load_external_data=True)
    initializers = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    initializer_by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    producer = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    def node_key(node: onnx.NodeProto) -> str:
        return node.name or node.output[0]

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def equation(node: onnx.NodeProto) -> str | None:
        value = attr_value(node, "equation", None)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    def const_scalar(name: str) -> float | None:
        if name not in initializers:
            return None
        value = np.asarray(initializers[name])
        if value.size != 1:
            return None
        return float(value.reshape(()))

    def trace_query_norm(name: str) -> onnx.NodeProto | None:
        node = producer.get(name)
        while node is not None and node.op_type in {"Transpose", "RotaryEmbedding"}:
            if len(node.input) == 0:
                return None
            node = producer.get(node.input[0])
        if node is not None and node.op_type == "SimplifiedLayerNormalization":
            return node
        return None

    def replace_uses(old_name: str, new_name: str, skip: set[str]) -> None:
        for node in model.graph.node:
            if node_key(node) in skip:
                continue
            for input_idx, input_name in enumerate(node.input):
                if input_name == old_name:
                    node.input[input_idx] = new_name
        for output in model.graph.output:
            if output.name == old_name:
                output.name = new_name

    skip_nodes: set[str] = set()
    scaled_initializers: dict[str, float] = {}
    rewrites = Counter()
    examples: list[dict[str, Any]] = []
    candidates: list[tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto, str, float]] = []
    query_norms_by_scale: dict[str, set[str]] = defaultdict(set)

    for einsum in model.graph.node:
        if einsum.op_type != "Einsum" or len(einsum.input) < 2 or len(einsum.output) != 1:
            continue
        if equation(einsum) not in {"bqhd,bkhd->bhqk", "bhqd,bhkd->bhqk"}:
            continue
        mul_consumers = [
            node
            for node in consumers.get(einsum.output[0], [])
            if node.op_type == "Mul" and len(node.input) == 2 and len(node.output) == 1
        ]
        if len(mul_consumers) != 1:
            continue
        scale_mul = mul_consumers[0]
        scale_inputs = [
            (input_idx, const_scalar(input_name))
            for input_idx, input_name in enumerate(scale_mul.input)
            if const_scalar(input_name) is not None
        ]
        if len(scale_inputs) != 1:
            continue
        scale_input_idx, scale = scale_inputs[0]
        assert scale is not None
        if not np.isfinite(scale) or scale == 0.0:
            continue
        if scale_mul.input[1 - scale_input_idx] != einsum.output[0]:
            continue
        if not consumers.get(scale_mul.output[0]):
            continue

        query_norm = trace_query_norm(einsum.input[0])
        if query_norm is None or len(query_norm.input) < 2:
            continue
        scale_name = query_norm.input[1]
        if scale_name not in initializers or scale_name not in initializer_by_name:
            continue
        candidates.append((einsum, scale_mul, query_norm, scale_name, scale))
        query_norms_by_scale[scale_name].add(node_key(query_norm))

    safe_scales: set[str] = set()
    for scale_name, query_norms in query_norms_by_scale.items():
        scale_consumers = consumers[scale_name]
        consumer_keys = {node_key(node) for node in scale_consumers}
        if all(node.op_type == "SimplifiedLayerNormalization" for node in scale_consumers) and (
            consumer_keys == query_norms
        ):
            safe_scales.add(scale_name)
        else:
            rewrites["shared_query_scale_initializers_skipped"] += 1

    for einsum, scale_mul, query_norm, scale_name, scale in candidates:
        if scale_name not in safe_scales:
            continue

        previous_scale = scaled_initializers.setdefault(scale_name, scale)
        if abs(previous_scale - scale) > 1e-8:
            continue
        replace_uses(scale_mul.output[0], einsum.output[0], skip={node_key(scale_mul)})
        skip_nodes.add(node_key(scale_mul))
        rewrites["attention_score_mul_removed"] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "einsum": einsum.name,
                    "scale_mul": scale_mul.name,
                    "query_norm": query_norm.name,
                    "query_scale_initializer": scale_name,
                    "scale": scale,
                }
            )

    if rewrites:
        for scale_name, scale in scaled_initializers.items():
            initializer = initializer_by_name[scale_name]
            scaled = np.asarray(initializers[scale_name], dtype=np.float32) * np.float32(scale)
            initializer.CopyFrom(onnx.numpy_helper.from_array(scaled, scale_name))
            rewrites["query_norm_initializers_scaled"] += 1

        rewritten_nodes = [node for node in model.graph.node if node_key(node) not in skip_nodes]
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)
        external_data_path(path).unlink(missing_ok=True)
        onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Einsum", "Mul", "Softmax", "SimplifiedLayerNormalization")
    return {
        "enabled": True,
        "tool": "custom_attention_scale_folding",
        "reason": (
            "Fold the constant attention score scale into query RMSNorm weights and "
            "bypass the logits-sized Mul before Softmax."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
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
        kv_heads, repeat_count, head_dim = input_shape[-3:]
        if kv_heads <= 0 or repeat_count <= 1 or head_dim <= 0:
            continue
        if output_shape[-2:] != (kv_heads * repeat_count, head_dim):
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
                or source_shape[-2:] != (kv_heads, head_dim)
            ):
                continue
            skip_nodes.add(source_producer.name)
            rewrites["from_unsqueeze"] += 1
        elif source_producer.op_type == "Reshape":
            if source_shape is None or source_shape[-1] != kv_heads * head_dim:
                continue
            compact_shape = tuple(output_shape[:-2]) + (kv_heads, head_dim)
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
                np.repeat(np.arange(kv_heads, dtype=np.int64), repeat_count),
                indices_name,
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
    external_data_path(path).unlink(missing_ok=True)
    # ONNX checker does not know ORT's SimplifiedLayerNormalization schema.
    # ORT validation is run separately by the export/accuracy checks.
    if not any(node.op_type == "SimplifiedLayerNormalization" for node in model.graph.node):
        onnx.checker.check_model(model)
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


def rewrite_packed_qkv_head_projection_for_webgpu(
    path: Path,
    enabled: bool,
) -> dict[str, Any]:
    before = op_counts(path)
    if not enabled:
        return {
            "enabled": False,
            "reason": "--pack_qkv_head_projection not set",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": 0,
            "rewrite_examples": [],
        }

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
    initializer_arrays = {
        name: onnx.numpy_helper.to_array(initializer) for name, initializer in initializers.items()
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    def node_key(node: onnx.NodeProto) -> str:
        return node.name or node.output[0]

    def can_rewrite_gemm(gemm: onnx.NodeProto) -> bool:
        if gemm.op_type != "Gemm" or len(gemm.input) != 2:
            return False
        if gemm.input[1] not in initializer_arrays:
            return False
        return _gemm_attrs(gemm) == {"alpha": 1.0, "beta": 0.0, "transA": 0, "transB": 0}

    def single_head_reshape(gemm: onnx.NodeProto) -> onnx.NodeProto | None:
        gemm_consumers = consumers.get(gemm.output[0], [])
        if len(gemm_consumers) != 1:
            return None
        reshape = gemm_consumers[0]
        if reshape.op_type != "Reshape" or len(reshape.output) != 1:
            return None
        input_shape = value_shapes.get(gemm.input[0])
        output_shape = value_shapes.get(reshape.output[0])
        weight = initializer_arrays.get(gemm.input[1])
        if (
            input_shape is None
            or output_shape is None
            or weight is None
            or len(input_shape) != 2
            or len(output_shape) != 4
            or output_shape[0] != 1
            or output_shape[1] != input_shape[0]
            or output_shape[-1] != 64
            or output_shape[-2] not in {2, 8}
            or weight.shape != (input_shape[-1], output_shape[-2] * output_shape[-1])
        ):
            return None
        return reshape

    grouped: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        if can_rewrite_gemm(node):
            grouped[node.input[0]].append(node)

    replacements: dict[str, list[onnx.NodeProto]] = {}
    skip_nodes: set[str] = set()
    new_initializers: list[onnx.TensorProto] = []
    examples: list[dict[str, Any]] = []
    rewrites = Counter()

    for input_name, gemms in grouped.items():
        candidates: list[tuple[onnx.NodeProto, onnx.NodeProto, tuple[int, ...]]] = []
        for gemm in gemms:
            reshape = single_head_reshape(gemm)
            if reshape is None:
                continue
            output_shape = value_shapes.get(reshape.output[0])
            assert output_shape is not None
            candidates.append((gemm, reshape, output_shape))
        if len(candidates) != 3:
            continue
        candidates_by_heads: dict[int, list[tuple[onnx.NodeProto, onnx.NodeProto, tuple[int, ...]]]]
        candidates_by_heads = defaultdict(list)
        for candidate in candidates:
            candidates_by_heads[candidate[2][-2]].append(candidate)
        if len(candidates_by_heads[8]) != 1 or len(candidates_by_heads[2]) != 2:
            continue

        q_candidate = candidates_by_heads[8][0]
        kv_candidates = sorted(candidates_by_heads[2], key=lambda item: item[0].name)
        ordered = [q_candidate, *kv_candidates]
        input_shape = value_shapes.get(input_name)
        if input_shape is None:
            continue
        input_width = input_shape[-1]
        weights = []
        split_sizes = []
        split_outputs = []
        unsqueeze_outputs = []
        for gemm, reshape, output_shape in ordered:
            weight = initializer_arrays[gemm.input[1]]
            head_count = output_shape[-2]
            weights.append(weight.reshape(input_width, head_count, output_shape[-1]))
            split_sizes.append(head_count)
            split_outputs.append(f"{reshape.output[0]}__packed_qkv_head")
            unsqueeze_outputs.append(reshape.output[0])

        first_gemm = min(
            (item[0] for item in ordered), key=lambda node: list(model.graph.node).index(node)
        )
        prefix = first_gemm.name or first_gemm.output[0]
        packed_weight_name = f"{prefix}__packed_qkv_head_weight"
        packed_output_name = f"{prefix}__packed_qkv_head_output"
        split_sizes_name = f"{prefix}__packed_qkv_head_split_sizes"
        unsqueeze_axes_name = f"{prefix}__packed_qkv_head_unsqueeze_axes"
        packed_weight = np.concatenate(weights, axis=1)
        new_initializers.extend(
            [
                onnx.numpy_helper.from_array(
                    packed_weight.astype(weights[0].dtype), packed_weight_name
                ),
                onnx.numpy_helper.from_array(
                    np.asarray(split_sizes, dtype=np.int64), split_sizes_name
                ),
                onnx.numpy_helper.from_array(np.asarray([0], dtype=np.int64), unsqueeze_axes_name),
            ]
        )
        replacement_nodes: list[onnx.NodeProto] = [
            onnx.helper.make_node(
                "Einsum",
                [input_name, packed_weight_name],
                [packed_output_name],
                name=f"{prefix}__packed_qkv_head_project",
                equation="nk,khd->nhd",
            ),
            onnx.helper.make_node(
                "Split",
                [packed_output_name, split_sizes_name],
                split_outputs,
                name=f"{prefix}__packed_qkv_head_split",
                axis=1,
            ),
        ]
        for split_output, original_output in zip(split_outputs, unsqueeze_outputs, strict=True):
            replacement_nodes.append(
                onnx.helper.make_node(
                    "Unsqueeze",
                    [split_output, unsqueeze_axes_name],
                    [original_output],
                    name=f"{original_output}__packed_qkv_head_unsqueeze",
                )
            )

        replacements[node_key(first_gemm)] = replacement_nodes
        for gemm, reshape, _ in ordered:
            if node_key(gemm) != node_key(first_gemm):
                skip_nodes.add(node_key(gemm))
            skip_nodes.add(node_key(reshape))
        rewrites["packed_qkv_head_projection"] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "input": input_name,
                    "input_shape": list(input_shape),
                    "gemms": [gemm.name for gemm, _, _ in ordered],
                    "outputs": [reshape.output[0] for _, reshape, _ in ordered],
                    "split_sizes": split_sizes,
                }
            )

    if rewrites:
        rewritten_nodes: list[onnx.NodeProto] = []
        for node in model.graph.node:
            key = node_key(node)
            if key in skip_nodes:
                continue
            if key in replacements:
                rewritten_nodes.extend(replacements[key])
            else:
                rewritten_nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)
        model.graph.initializer.extend(new_initializers)
        onnx.checker.check_model(model)
        external_data_path(path).unlink(missing_ok=True)
        onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Gemm", "Einsum", "Split", "Unsqueeze", "Concat", "Reshape")
    return {
        "enabled": True,
        "tool": "custom_packed_qkv_head_projection_rewrite",
        "reason": (
            "Pack sibling Q/K/V Gemm+head-Reshape projections into one rank-aware "
            "Einsum and split the packed head axis into the original Q/K/V tensors."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
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
        head_count = output_shape[-2] if len(output_shape) >= 2 else 0
        head_dim = output_shape[-1] if len(output_shape) >= 1 else 0
        if (
            gemm is not None
            and can_rewrite_gemm(gemm)
            and len(input_shape) == 2
            and len(output_shape) in (4, 5, 6)
            and head_count >= 2
            and head_dim >= 8
            and input_shape[-1] == head_count * head_dim
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
            and input_shape[-2] >= 2
            and input_shape[-1] >= 8
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
    external_data_path(path).unlink(missing_ok=True)
    # ONNX checker does not know ORT contrib/custom schemas that may already be
    # present when this rewrite is applied to an optimized artifact.
    if not any(
        node.op_type in {"SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization"}
        or node.domain == "com.microsoft"
        for node in model.graph.node
    ):
        onnx.checker.check_model(model)
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
        head_count = output_shape[-2] if len(output_shape) >= 2 else 0
        head_dim = output_shape[-1] if len(output_shape) >= 1 else 0
        if (
            gemm is not None
            and can_rewrite_gemm(gemm)
            and len(input_shape) == 2
            and len(output_shape) in (4, 5, 6)
            and head_count >= 2
            and head_dim >= 8
            and input_shape[-1] == head_count * head_dim
            and prefix_shape.count(input_shape[0]) == 1
            and all(dim in (1, input_shape[0]) for dim in prefix_shape)
        ):
            head_count = output_shape[-2]
            head_dim = output_shape[-1]
            head_axis = len(output_shape) - 2
            split_outputs = [
                f"{reshape.output[0]}__flat_head_{head_idx}" for head_idx in range(head_count)
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
            and input_shape[-2] >= 2
            and input_shape[-1] >= 8
            and input_shape[:-2].count(output_shape[0]) == 1
            and all(dim in (1, output_shape[0]) for dim in input_shape[:-2])
            and output_shape[1] == input_shape[-2] * input_shape[-1]
        ):
            head_count = input_shape[-2]
            head_dim = input_shape[-1]
            head_axis = len(input_shape) - 2
            split_outputs = [
                f"{reshape.output[0]}__ranked_head_{head_idx}" for head_idx in range(head_count)
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
    external_data_path(path).unlink(missing_ok=True)
    # ONNX checker does not know ORT contrib/custom schemas that may already be
    # present when this rewrite is applied to an optimized artifact.
    if not any(
        node.op_type in {"SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization"}
        or node.domain == "com.microsoft"
        for node in model.graph.node
    ):
        onnx.checker.check_model(model)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = (
        "Reshape",
        "Flatten",
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


def pack_sibling_gemms_for_webgpu(
    path: Path,
    pack_qkv: bool,
    pack_swiglu: bool,
) -> dict[str, Any]:
    before = op_counts(path)
    if not pack_qkv and not pack_swiglu:
        return {
            "enabled": False,
            "reason": "--pack_qkv_gemm and --pack_swiglu_gemm not set",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
        }

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
    initializer_arrays = {
        name: onnx.numpy_helper.to_array(initializer) for name, initializer in initializers.items()
    }

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def is_packable_gemm(node: onnx.NodeProto) -> bool:
        if node.op_type != "Gemm" or len(node.input) != 2 or len(node.output) != 1:
            return False
        if node.input[1] not in initializer_arrays:
            return False
        return (
            int(attr_value(node, "transA", 0)) == 0
            and int(attr_value(node, "transB", 0)) == 0
            and float(attr_value(node, "alpha", 1.0)) == 1.0
            and float(attr_value(node, "beta", 1.0)) == 0.0
        )

    grouped: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        if is_packable_gemm(node):
            grouped[node.input[0]].append(node)

    replacements: dict[str, list[onnx.NodeProto]] = {}
    skip_nodes: set[str] = set()
    remove_initializer_names: set[str] = set()
    new_initializers: list[onnx.TensorProto] = []
    packed_initializer_names: dict[tuple[str, tuple[str, ...]], tuple[str, str]] = {}
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    def maybe_pack(input_name: str, nodes: list[onnx.NodeProto]) -> None:
        output_shapes = [value_shapes.get(node.output[0]) for node in nodes]
        if any(shape is None or len(shape) != 2 for shape in output_shapes):
            return
        output_widths = [int(shape[-1]) for shape in output_shapes if shape is not None]
        kind: str | None = None
        input_widths = {
            int(initializer_arrays[node.input[1]].shape[0])
            for node in nodes
            if initializer_arrays[node.input[1]].ndim == 2
        }
        input_width = next(iter(input_widths)) if len(input_widths) == 1 else None
        sorted_widths = sorted(output_widths)
        if (
            pack_qkv
            and len(nodes) == 3
            and input_width is not None
            and len(sorted_widths) == 3
            and sorted_widths[0] == sorted_widths[1]
            and sorted_widths[2] == sorted_widths[0] * 4
            and sorted_widths[2] == input_width * 2
        ):
            kind = "qkv"
        elif (
            pack_swiglu
            and len(nodes) == 2
            and input_width is not None
            and output_widths[0] == output_widths[1]
            and output_widths[0] == input_width * 3
        ):
            kind = "swiglu"
        if kind is None:
            return

        weights = [initializer_arrays[node.input[1]] for node in nodes]
        if any(weight.ndim != 2 for weight in weights):
            return
        if len(input_widths) != 1:
            return
        if [int(weight.shape[1]) for weight in weights] != output_widths:
            return

        first_node = nodes[0]
        prefix = first_node.name or first_node.output[0]
        initializer_key = (kind, tuple(node.input[1] for node in nodes))
        if initializer_key in packed_initializer_names:
            packed_weight_name, split_sizes_name = packed_initializer_names[initializer_key]
        else:
            packed_weight_name = f"{prefix}__packed_{kind}_weight"
            split_sizes_name = f"{prefix}__packed_{kind}_split_sizes"
            packed_initializer_names[initializer_key] = (packed_weight_name, split_sizes_name)
            packed_weight = np.concatenate(weights, axis=1)
            new_initializers.append(
                onnx.numpy_helper.from_array(
                    packed_weight.astype(weights[0].dtype),
                    packed_weight_name,
                )
            )
            new_initializers.append(
                onnx.numpy_helper.from_array(
                    np.asarray(output_widths, dtype=np.int64),
                    split_sizes_name,
                )
            )
        packed_output_name = f"{prefix}__packed_{kind}_output"
        replacements[first_node.name] = [
            onnx.helper.make_node(
                "Gemm",
                [input_name, packed_weight_name],
                [packed_output_name],
                name=f"{prefix}__packed_{kind}_gemm",
                transA=0,
                transB=0,
                alpha=1.0,
                beta=0.0,
            ),
            onnx.helper.make_node(
                "Split",
                [packed_output_name, split_sizes_name],
                [node.output[0] for node in nodes],
                name=f"{prefix}__packed_{kind}_split",
                axis=1,
            ),
        ]
        for node in nodes[1:]:
            skip_nodes.add(node.name)
        remove_initializer_names.update(node.input[1] for node in nodes)
        rewrites[kind] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "kind": kind,
                    "input": input_name,
                    "gemms": [node.name for node in nodes],
                    "output_widths": output_widths,
                    "packed_width": int(sum(output_widths)),
                }
            )

    for input_name, nodes in grouped.items():
        maybe_pack(input_name, nodes)

    if rewrites:
        rewritten_nodes: list[onnx.NodeProto] = []
        for node in model.graph.node:
            if node.name in skip_nodes:
                continue
            if node.name in replacements:
                rewritten_nodes.extend(replacements[node.name])
            else:
                rewritten_nodes.append(node)
        used_inputs = {
            input_name for node in rewritten_nodes for input_name in node.input if input_name
        }
        kept_initializers = [
            initializer
            for initializer in model.graph.initializer
            if initializer.name in used_inputs or initializer.name not in remove_initializer_names
        ]
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept_initializers)
        model.graph.initializer.extend(new_initializers)
        external_data_path(path).unlink(missing_ok=True)
        onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Gemm", "Split", "Concat", "Reshape", "Einsum", "Transpose")
    return {
        "enabled": True,
        "tool": "custom_packed_sibling_gemm_rewrite",
        "reason": (
            "Pack sibling projection Gemm nodes with the same input into one wider "
            "Gemm followed by Split. Original output tensor names are preserved."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
    }


def rewrite_packed_qkv_split_partial_heads_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    initializer_arrays = {
        name: onnx.numpy_helper.to_array(initializer) for name, initializer in initializers.items()
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def split_sizes(node: onnx.NodeProto) -> tuple[int, ...] | None:
        if len(node.input) >= 2 and node.input[1] in initializer_arrays:
            return tuple(
                int(value) for value in np.asarray(initializer_arrays[node.input[1]]).reshape(-1)
            )
        return None

    replacements: dict[str, onnx.NodeProto] = {}
    skip_nodes: set[str] = set()
    stale_value_info: set[str] = set()
    new_initializers: list[onnx.TensorProto] = []
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    graph_output_names = {output.name for output in model.graph.output}

    for split in model.graph.node:
        if split.op_type != "Split" or len(split.output) < 2:
            continue
        split_axis = int(attr_value(split, "axis", -1))
        if split_axis < 0:
            continue
        sizes = split_sizes(split)
        if sizes is None or len(sizes) != len(split.output):
            continue

        new_sizes: list[int] = []
        new_outputs: list[str] = []
        child_splits: list[onnx.NodeProto] = []
        removed_outputs: list[str] = []

        for output_name, size in zip(split.output, sizes):
            if output_name in graph_output_names:
                new_sizes.append(int(size))
                new_outputs.append(output_name)
                continue
            output_consumers = consumers.get(output_name, [])
            child = output_consumers[0] if len(output_consumers) == 1 else None
            child_sizes = split_sizes(child) if child is not None else None
            if child is None or child.op_type != "Split" or child_sizes is None:
                new_sizes.append(int(size))
                new_outputs.append(output_name)
                continue
            if int(attr_value(child, "axis", -1)) != split_axis:
                new_sizes.append(int(size))
                new_outputs.append(output_name)
                continue
            if (
                len(child.output) != 2
                or len(child.output) != len(child_sizes)
                or sum(child_sizes) != int(size)
            ):
                new_sizes.append(int(size))
                new_outputs.append(output_name)
                continue

            new_sizes.extend(int(child_size) for child_size in child_sizes)
            new_outputs.extend(child.output)
            child_splits.append(child)
            removed_outputs.append(output_name)

        if not child_splits:
            continue

        size_name = f"{split.name or split.output[0]}__partial_head_split_sizes"
        new_initializers.append(
            onnx.numpy_helper.from_array(np.asarray(new_sizes, dtype=np.int64), size_name)
        )
        rewritten = onnx.helper.make_node(
            "Split",
            [split.input[0], size_name],
            new_outputs,
            name=f"{split.name or split.output[0]}__partial_head_split",
            axis=split_axis,
        )
        replacements[node_key(split)] = rewritten
        skip_nodes.update(node_key(child) for child in child_splits)
        stale_value_info.update(removed_outputs)
        rewrites["packed_qkv_partial_head_split"] += 1
        rewrites["removed_child_head_splits"] += len(child_splits)
        if len(examples) < 12:
            examples.append(
                {
                    "split": split.name,
                    "old_sizes": list(sizes),
                    "new_sizes": new_sizes,
                    "child_splits": [child.name for child in child_splits],
                    "removed_outputs": removed_outputs,
                }
            )

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_packed_qkv_partial_head_split_rewrite",
            "rewrites": {},
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "tracked_ops_before": {},
            "tracked_ops_after": {},
            "rewrite_examples": [],
        }

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        key = node_key(node)
        if key in skip_nodes:
            continue
        rewritten_nodes.append(replacements.get(key, node))

    kept_value_info = [
        value for value in model.graph.value_info if value.name not in stale_value_info
    ]
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    model.graph.initializer.extend(new_initializers)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Split", "Squeeze", "Unsqueeze", "Concat", "Gemm", "Einsum", "Transpose")
    return {
        "enabled": True,
        "tool": "custom_packed_qkv_partial_head_split_rewrite",
        "reason": (
            "Inline K/V two-head Split nodes into the packed QKV Split. This keeps the "
            "packed Gemm output identical but emits K/V head tensors directly, removing "
            "two Split dispatches per matched QKV projection."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
    }


def rewrite_q_head_split_gather_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    initializer_arrays = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def axes_input(node: onnx.NodeProto) -> tuple[int, ...] | None:
        if len(node.input) < 2 or node.input[1] not in initializer_arrays:
            return None
        return tuple(
            int(value) for value in np.asarray(initializer_arrays[node.input[1]]).reshape(-1)
        )

    indices_name = "head_gather_indices_8x32"
    unsqueeze_axis0_name = "head_gather_unsqueeze_axis0"
    unsqueeze_axis1_name = "head_gather_unsqueeze_axis1"
    required_initializers = {
        indices_name: np.arange(256, dtype=np.int64).reshape(8, 32),
        unsqueeze_axis0_name: np.asarray([0], dtype=np.int64),
        unsqueeze_axis1_name: np.asarray([1], dtype=np.int64),
    }

    replacements: dict[str, list[onnx.NodeProto]] = {}
    skip_nodes: set[str] = set()
    stale_value_info: set[str] = set()
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    for split in model.graph.node:
        if split.op_type != "Split" or len(split.output) != 8:
            continue
        if int(attr_value(split, "axis", -1)) != 1:
            continue
        if len(split.input) < 2 or split.input[1] not in initializer_arrays:
            continue
        split_sizes = tuple(
            int(value) for value in np.asarray(initializer_arrays[split.input[1]]).reshape(-1)
        )
        if split_sizes != (32, 32, 32, 32, 32, 32, 32, 32):
            continue

        split_consumers = [consumers.get(output_name, []) for output_name in split.output]
        if any(len(output_consumers) != 1 for output_consumers in split_consumers):
            continue
        unsqueezes = [output_consumers[0] for output_consumers in split_consumers]
        if any(node.op_type != "Unsqueeze" for node in unsqueezes):
            continue
        if any(len(consumers.get(node.output[0], [])) != 1 for node in unsqueezes):
            continue

        concat = consumers[unsqueezes[0].output[0]][0]
        if concat.op_type != "Concat":
            continue
        if any(consumers[node.output[0]][0] is not concat for node in unsqueezes):
            continue
        if list(concat.input) != [node.output[0] for node in unsqueezes]:
            continue

        unsqueeze_axes = [axes_input(node) for node in unsqueezes]
        if any(axes is None for axes in unsqueeze_axes) or len(set(unsqueeze_axes)) != 1:
            continue
        axes = unsqueeze_axes[0]
        concat_axis = int(attr_value(concat, "axis", -1))

        gather_output = f"{split.output[0]}__head_gathered"
        replacement_nodes = [
            onnx.helper.make_node(
                "Gather",
                [split.input[0], indices_name],
                [gather_output],
                name=f"{split.name or split.output[0]}__head_gather",
                axis=1,
            )
        ]
        if axes == (0, 1) and concat_axis == 1:
            unsqueeze_output = f"{split.output[0]}__head_gather_unsqueezed"
            replacement_nodes.extend(
                [
                    onnx.helper.make_node(
                        "Unsqueeze",
                        [gather_output, unsqueeze_axis0_name],
                        [unsqueeze_output],
                        name=f"{split.name or split.output[0]}__head_gather_unsqueeze0",
                    ),
                    onnx.helper.make_node(
                        "Transpose",
                        [unsqueeze_output],
                        [concat.output[0]],
                        name=f"{split.name or split.output[0]}__head_gather_transpose",
                        perm=[0, 2, 1, 3],
                    ),
                ]
            )
            rewrites["axis01_concat1"] += 1
        elif axes == (1, 2) and concat_axis == 2:
            replacement_nodes.append(
                onnx.helper.make_node(
                    "Unsqueeze",
                    [gather_output, unsqueeze_axis1_name],
                    [concat.output[0]],
                    name=f"{split.name or split.output[0]}__head_gather_unsqueeze1",
                )
            )
            rewrites["axis12_concat2"] += 1
        else:
            continue

        replacements[node_key(split)] = replacement_nodes
        skip_nodes.add(node_key(concat))
        stale_value_info.update(split.output)
        stale_value_info.update(node.output[0] for node in unsqueezes)
        skip_nodes.update(node_key(node) for node in unsqueezes)
        if len(examples) < 12:
            examples.append(
                {
                    "split": split.name,
                    "input": split.input[0],
                    "output": concat.output[0],
                    "unsqueeze_axes": list(axes),
                    "concat_axis": concat_axis,
                }
            )

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_q_head_split_gather_rewrite",
            "rewrites": {},
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "tracked_ops_before": {},
            "tracked_ops_after": {},
            "rewrite_examples": [],
        }

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for name, value in required_initializers.items():
        if name not in initializer_names:
            model.graph.initializer.append(onnx.numpy_helper.from_array(value, name))

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        key = node_key(node)
        if key in skip_nodes:
            continue
        rewritten_nodes.extend(replacements.get(key, [node]))

    kept_value_info = [
        value for value in model.graph.value_info if value.name not in stale_value_info
    ]
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Split", "Unsqueeze", "Concat", "Gather", "Transpose", "Einsum", "Gemm")
    return {
        "enabled": True,
        "tool": "custom_q_head_split_gather_rewrite",
        "reason": (
            "Replace 8-way Q-head Split/Unsqueeze/Concat layout islands with "
            "Gather plus a small rank-restoring layout step. This preserves the "
            "head ordering while reducing layout dispatch count."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
    }


def remove_zero_softmax_bias_adds_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    initializer_arrays = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    replacements: dict[str, str] = {}
    skip_nodes: set[str] = set()
    stale_value_info: set[str] = set()
    examples: list[dict[str, Any]] = []

    for node in model.graph.node:
        if node.op_type != "Add" or len(node.input) != 2 or len(node.output) != 1:
            continue
        output_name = node.output[0]
        output_consumers = consumers.get(output_name, [])
        if len(output_consumers) != 1 or output_consumers[0].op_type != "Softmax":
            continue

        zero_input_index = None
        for index, input_name in enumerate(node.input):
            value = initializer_arrays.get(input_name)
            if value is None:
                continue
            if np.all(value == 0):
                zero_input_index = index
                break
        if zero_input_index is None:
            continue

        replacements[output_name] = node.input[1 - zero_input_index]
        skip_nodes.add(node_key(node))
        stale_value_info.add(output_name)
        if len(examples) < 12:
            zero_name = node.input[zero_input_index]
            value = initializer_arrays[zero_name]
            examples.append(
                {
                    "add": node.name,
                    "softmax": output_consumers[0].name,
                    "zero_input": zero_name,
                    "zero_shape": list(value.shape),
                    "rewired_input": node.input[1 - zero_input_index],
                }
            )

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_zero_softmax_bias_add_prune",
            "rewrites": 0,
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "tracked_ops_before": {},
            "tracked_ops_after": {},
            "rewrite_examples": [],
        }

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node_key(node) in skip_nodes:
            continue
        for index, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[index] = replacements[input_name]
        rewritten_nodes.append(node)

    used_inputs = {
        input_name for node in rewritten_nodes for input_name in node.input if input_name
    }
    kept_initializers = [
        initializer for initializer in model.graph.initializer if initializer.name in used_inputs
    ]
    kept_value_info = [
        value for value in model.graph.value_info if value.name not in stale_value_info
    ]
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Add", "Softmax", "Mul", "Einsum")
    return {
        "enabled": True,
        "tool": "custom_zero_softmax_bias_add_prune",
        "reason": (
            "Remove Add nodes immediately before Softmax when one input is an "
            "all-zero initializer. This is exact for full-cache attention masks "
            "that fold to zero bias."
        ),
        "rewrites": len(replacements),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrite_examples": examples,
    }


def rewrite_cache_layer_slices_as_gather_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    initializer_arrays = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    skip_nodes: set[str] = set()
    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    stale_value_info: set[str] = set()
    examples: list[dict[str, Any]] = []
    rewrites = Counter()

    for node in model.graph.node:
        key = node_key(node)
        if key in skip_nodes:
            continue

        replacement: onnx.NodeProto | None = None
        if (
            node.op_type == "Slice"
            and len(node.input) >= 4
            and len(node.output) == 1
            and node.input[0] in {"k_cache", "v_cache"}
        ):
            starts = initializer_arrays.get(node.input[1])
            ends = initializer_arrays.get(node.input[2])
            axes = initializer_arrays.get(node.input[3])
            slice_consumers = consumers.get(node.output[0], [])
            if (
                starts is not None
                and ends is not None
                and axes is not None
                and len(slice_consumers) == 1
                and slice_consumers[0].op_type == "Squeeze"
                and len(slice_consumers[0].output) == 1
            ):
                starts = np.asarray(starts).reshape(-1)
                ends = np.asarray(ends).reshape(-1)
                axes = np.asarray(axes).reshape(-1)
                if (
                    tuple(int(axis) for axis in axes) == (0, 1, 2, 3, 4, 5)
                    and starts.size == 6
                    and ends.size == 6
                    and all(int(starts[index]) == 0 for index in range(1, 6))
                    and int(ends[0]) == int(starts[0]) + 1
                ):
                    layer = int(starts[0])
                    squeeze = slice_consumers[0]
                    gather_index_name = f"{key}__layer_gather_index"
                    new_initializers.append(
                        onnx.numpy_helper.from_array(
                            np.asarray(layer, dtype=np.int64), gather_index_name
                        )
                    )
                    replacement = onnx.helper.make_node(
                        "Gather",
                        [node.input[0], gather_index_name],
                        [squeeze.output[0]],
                        name=f"{key}__layer_gather",
                        axis=0,
                    )
                    skip_nodes.add(node_key(squeeze))
                    stale_value_info.update({node.output[0], squeeze.output[0]})
                    rewrites["cache_layer_slice_to_gather"] += 1
                    if len(examples) < 12:
                        examples.append(
                            {
                                "slice": node.name,
                                "squeeze": squeeze.name,
                                "cache": node.input[0],
                                "layer": layer,
                                "output": squeeze.output[0],
                            }
                        )

        new_nodes.append(replacement if replacement is not None else node)

    if not rewrites:
        return {
            "enabled": True,
            "tool": "custom_cache_layer_slice_gather_rewrite",
            "reason": "no eligible cache layer Slice/Squeeze pairs found",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
        }

    used_inputs = {input_name for node in new_nodes for input_name in node.input if input_name}
    kept_initializers = [
        initializer for initializer in model.graph.initializer if initializer.name in used_inputs
    ]
    kept_value_info = [
        value for value in model.graph.value_info if value.name not in stale_value_info
    ]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    model.graph.initializer.extend(new_initializers)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Slice", "Squeeze", "Gather", "Einsum", "Softmax")
    return {
        "enabled": True,
        "tool": "custom_cache_layer_slice_gather_rewrite",
        "reason": (
            "Replace full-shape cache layer Slice followed by layer-axis Squeeze "
            "with scalar Gather(axis=0). This preserves the per-layer cache tensor "
            "shape while removing one layout node per K/V layer."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
    }


def fuse_manual_gqa_attention_for_webgpu(
    path: Path, enabled: bool, fuse_spatial: bool = False
) -> dict[str, Any]:
    before = op_counts(path)
    if not enabled:
        return {
            "enabled": False,
            "reason": "--fuse_gqa_attention not set",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
        }

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
    initializers = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    producer = {output: node for node in model.graph.node for output in node.output}

    def node_key(node: onnx.NodeProto) -> str:
        return node.name or node.output[0]

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def equation(node: onnx.NodeProto) -> str:
        value = attr_value(node, "equation", b"")
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def single_consumer(value_name: str, op_type: str | None = None) -> onnx.NodeProto | None:
        value_consumers = consumers.get(value_name, [])
        if len(value_consumers) != 1:
            return None
        consumer = value_consumers[0]
        if op_type is not None and consumer.op_type != op_type:
            return None
        return consumer

    def const_scalar(name: str) -> float | None:
        if name not in initializers:
            return None
        array = np.asarray(initializers[name])
        if array.size != 1:
            return None
        return float(array.reshape(()))

    def squeeze_axes(node: onnx.NodeProto) -> tuple[int, ...] | None:
        if node.op_type != "Squeeze":
            return None
        if len(node.input) >= 2 and node.input[1] in initializers:
            return tuple(int(axis) for axis in np.asarray(initializers[node.input[1]]).reshape(-1))
        axes = attr_value(node, "axes", None)
        if axes is None:
            return None
        return tuple(int(axis) for axis in axes)

    def folded_output_merge(
        output_name: str,
        num_heads: int,
        head_dim: int,
    ) -> tuple[str, set[str], tuple[int, ...]] | None:
        split = single_consumer(output_name, "Split")
        if split is None or int(attr_value(split, "axis", 0)) != 2:
            return None
        if len(split.output) != num_heads:
            return None

        # Current WebGPU layout passes factor the per-head squeeze through the
        # merge, producing Split(heads) -> Concat(feature axis) -> Squeeze.
        # The fused GQA op returns a flattened [batch, query, heads * dim]
        # tensor, so only the non-head singleton axes still need squeezing.
        factored_concat: onnx.NodeProto | None = None
        factored = True
        for split_output in split.output:
            split_consumer = single_consumer(split_output, "Concat")
            if split_consumer is None:
                factored = False
                break
            if factored_concat is None:
                factored_concat = split_consumer
            elif node_key(factored_concat) != node_key(split_consumer):
                factored = False
                break
        if factored and factored_concat is not None:
            if int(attr_value(factored_concat, "axis", 0)) == 3 and list(
                factored_concat.input
            ) == list(split.output):
                squeeze = single_consumer(factored_concat.output[0], "Squeeze")
                axes = squeeze_axes(squeeze) if squeeze is not None else None
                if axes is not None and 2 in axes and all(axis in (0, 1, 2) for axis in axes):
                    gqa_squeeze_axes = tuple(axis for axis in axes if axis != 2)
                    output_shape = value_shapes.get(squeeze.output[0])
                    if (
                        output_shape is not None
                        and output_shape[-1] == num_heads * head_dim
                        and len(output_shape) == 3 - len(gqa_squeeze_axes)
                    ):
                        return (
                            squeeze.output[0],
                            {node_key(split), node_key(factored_concat), node_key(squeeze)},
                            gqa_squeeze_axes,
                        )

        squeezed_outputs: list[str] = []
        skip = {node_key(split)}
        concat: onnx.NodeProto | None = None
        gqa_squeeze_axes: tuple[int, ...] | None = None
        for split_output in split.output:
            squeeze = single_consumer(split_output, "Squeeze")
            axes = squeeze_axes(squeeze) if squeeze is not None else None
            if squeeze is None or axes is None or 2 not in axes:
                return None
            if any(axis not in (0, 1, 2) for axis in axes):
                return None
            candidate_gqa_axes = tuple(axis for axis in axes if axis != 2)
            if gqa_squeeze_axes is None:
                gqa_squeeze_axes = candidate_gqa_axes
            elif gqa_squeeze_axes != candidate_gqa_axes:
                return None
            squeezed_outputs.append(squeeze.output[0])
            skip.add(node_key(squeeze))
            squeeze_consumer = single_consumer(squeeze.output[0], "Concat")
            if squeeze_consumer is None:
                return None
            if concat is None:
                concat = squeeze_consumer
            elif node_key(concat) != node_key(squeeze_consumer):
                return None

        if concat is None or int(attr_value(concat, "axis", 0)) != 2:
            if concat is None:
                return None
            concat_axis = int(attr_value(concat, "axis", 0))
            if concat_axis != (2 - len(gqa_squeeze_axes or ())):
                return None
        else:
            concat_axis = 2
        if list(concat.input) != squeezed_outputs:
            return None
        output_shape = value_shapes.get(concat.output[0])
        if output_shape is None or output_shape[-1] != num_heads * head_dim:
            return None
        expected_rank = 3 - len(gqa_squeeze_axes or ())
        if len(output_shape) != expected_rank:
            return None
        skip.add(node_key(concat))
        return concat.output[0], skip, gqa_squeeze_axes or ()

    new_initializers: list[onnx.TensorProto] = []
    replacements: dict[str, list[onnx.NodeProto]] = {}
    skip_nodes: set[str] = set()
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    def flatten_heads(
        source_name: str,
        shape: tuple[int, ...],
        prefix: str,
    ) -> tuple[str, list[onnx.NodeProto]]:
        if len(shape) == 3:
            return source_name, []
        if len(shape) != 4:
            raise ValueError(f"Expected rank-3 or rank-4 attention input, got {shape}")
        num_heads = int(shape[2])
        split_sizes_name = f"{prefix}__split_sizes"
        squeeze_axes_name = f"{prefix}__squeeze_axes"
        new_initializers.append(
            onnx.numpy_helper.from_array(
                np.ones((num_heads,), dtype=np.int64),
                split_sizes_name,
            )
        )
        new_initializers.append(
            onnx.numpy_helper.from_array(np.asarray([2], dtype=np.int64), squeeze_axes_name)
        )
        split_outputs = [f"{prefix}__ranked_head_{idx}" for idx in range(num_heads)]
        flat_outputs = [f"{prefix}__flat_head_{idx}" for idx in range(num_heads)]
        nodes = [
            onnx.helper.make_node(
                "Split",
                [source_name, split_sizes_name],
                split_outputs,
                name=f"{prefix}__head_split",
                axis=2,
            )
        ]
        nodes.extend(
            onnx.helper.make_node(
                "Squeeze",
                [split_output, squeeze_axes_name],
                [flat_output],
                name=f"{prefix}__head_{idx}_squeeze",
            )
            for idx, (split_output, flat_output) in enumerate(zip(split_outputs, flat_outputs))
        )
        flat_name = f"{prefix}__flat"
        nodes.append(
            onnx.helper.make_node(
                "Concat",
                flat_outputs,
                [flat_name],
                name=f"{prefix}__head_concat",
                axis=2,
            )
        )
        return flat_name, nodes

    def slice_axis(
        source_name: str,
        axis: int,
        start: int,
        end: int,
        prefix: str,
    ) -> tuple[str, list[onnx.NodeProto]]:
        starts_name = f"{prefix}__starts"
        ends_name = f"{prefix}__ends"
        axes_name = f"{prefix}__axes"
        steps_name = f"{prefix}__steps"
        new_initializers.extend(
            [
                onnx.numpy_helper.from_array(np.asarray([start], dtype=np.int64), starts_name),
                onnx.numpy_helper.from_array(np.asarray([end], dtype=np.int64), ends_name),
                onnx.numpy_helper.from_array(np.asarray([axis], dtype=np.int64), axes_name),
                onnx.numpy_helper.from_array(np.asarray([1], dtype=np.int64), steps_name),
            ]
        )
        output_name = f"{prefix}__slice"
        return output_name, [
            onnx.helper.make_node(
                "Slice",
                [source_name, starts_name, ends_name, axes_name, steps_name],
                [output_name],
                name=f"{prefix}__slice",
            )
        ]

    for softmax in model.graph.node:
        if softmax.op_type != "Softmax":
            continue
        softmax_input = producer.get(softmax.input[0])
        first = None
        scale = None
        mul: onnx.NodeProto | None = None
        if (
            softmax_input is not None
            and softmax_input.op_type == "Mul"
            and len(softmax_input.input) == 2
        ):
            mul = softmax_input
            for input_name in mul.input:
                candidate = producer.get(input_name)
                if candidate is not None and candidate.op_type == "Einsum":
                    first = candidate
                else:
                    scale = const_scalar(input_name)
        elif softmax_input is not None and softmax_input.op_type == "Einsum":
            first = softmax_input
            scale = 1.0
        if first is None or scale is None or equation(first) != "bqhd,bkhd->bhqk":
            continue
        second = single_consumer(softmax.output[0], "Einsum")
        if second is None or equation(second) != "bhqk,bkhd->bqhd":
            continue
        if second.input[0] != softmax.output[0]:
            continue
        if mul is not None:
            if single_consumer(first.output[0]) is not mul:
                continue
            if single_consumer(mul.output[0]) is not softmax:
                continue
        elif single_consumer(first.output[0]) is not softmax:
            continue

        q_name = first.input[0]
        k_repeated_name = first.input[1]
        v_repeated_name = second.input[1]
        k_gather = producer.get(k_repeated_name)
        v_gather = producer.get(v_repeated_name)
        if k_gather is None or v_gather is None:
            continue
        if k_gather.op_type != "Gather" or v_gather.op_type != "Gather":
            continue
        if single_consumer(k_gather.output[0]) is not first:
            continue
        if single_consumer(v_gather.output[0]) is not second:
            continue

        k_compact_name = k_gather.input[0]
        v_compact_name = v_gather.input[0]
        q_shape = value_shapes.get(q_name)
        k_shape = value_shapes.get(k_compact_name)
        v_shape = value_shapes.get(v_compact_name)
        output_shape = value_shapes.get(second.output[0])
        if (
            q_shape is None
            or k_shape is None
            or v_shape is None
            or output_shape is None
            or len(q_shape) != 4
            or len(k_shape) != 4
            or len(v_shape) != 4
            or len(output_shape) != 4
        ):
            continue
        if q_shape[:2] != output_shape[:2] or q_shape[2:] != output_shape[2:]:
            continue
        if k_shape != v_shape:
            continue
        if q_shape[0] != k_shape[0] or q_shape[3] != k_shape[3]:
            continue
        use_past_cache = q_shape[1] == 1 and k_shape[1] > 1
        if not use_past_cache and not fuse_spatial:
            continue
        if not use_past_cache:
            # ONNX Runtime's GroupQueryAttention schema requires real seq_lens and
            # total_sequence_length inputs. In the WebGPU implementation, seq_lens
            # makes attention causal, which is not equivalent for bidirectional
            # spatial attention, so only cached temporal attention can be fused.
            continue
        num_heads = int(q_shape[2])
        kv_num_heads = int(k_shape[2])
        head_dim = int(q_shape[3])
        if num_heads <= kv_num_heads or num_heads % kv_num_heads != 0:
            continue
        if int(attr_value(k_gather, "axis", -1)) != 2 or int(attr_value(v_gather, "axis", -1)) != 2:
            continue

        folded = folded_output_merge(second.output[0], num_heads=num_heads, head_dim=head_dim)
        if folded is None:
            continue
        output_name, merge_skip_nodes, gqa_squeeze_axes = folded

        prefix = node_key(second)
        gqa_output_name = output_name if not gqa_squeeze_axes else f"{prefix}__gqa_flat_output"
        q_flat, q_nodes = flatten_heads(q_name, q_shape, prefix=f"{prefix}__gqa_q")
        if use_past_cache:
            past_length = int(k_shape[1]) - 1
            k_past, k_past_slice_nodes = slice_axis(
                k_compact_name,
                axis=1,
                start=0,
                end=past_length,
                prefix=f"{prefix}__gqa_k_past",
            )
            v_past, v_past_slice_nodes = slice_axis(
                v_compact_name,
                axis=1,
                start=0,
                end=past_length,
                prefix=f"{prefix}__gqa_v_past",
            )
            k_current, k_current_slice_nodes = slice_axis(
                k_compact_name,
                axis=1,
                start=past_length,
                end=int(k_shape[1]),
                prefix=f"{prefix}__gqa_k_current",
            )
            v_current, v_current_slice_nodes = slice_axis(
                v_compact_name,
                axis=1,
                start=past_length,
                end=int(v_shape[1]),
                prefix=f"{prefix}__gqa_v_current",
            )
            k_past_bnsk = f"{prefix}__gqa_k_past_bnsk"
            v_past_bnsk = f"{prefix}__gqa_v_past_bnsk"
            k_current_flat, k_current_flat_nodes = flatten_heads(
                k_current,
                (k_shape[0], 1, k_shape[2], k_shape[3]),
                prefix=f"{prefix}__gqa_k_current",
            )
            v_current_flat, v_current_flat_nodes = flatten_heads(
                v_current,
                (v_shape[0], 1, v_shape[2], v_shape[3]),
                prefix=f"{prefix}__gqa_v_current",
            )
            k_nodes = [
                *k_past_slice_nodes,
                onnx.helper.make_node(
                    "Transpose",
                    [k_past],
                    [k_past_bnsk],
                    name=f"{prefix}__gqa_k_past_to_bnsk",
                    perm=[0, 2, 1, 3],
                ),
                *k_current_slice_nodes,
                *k_current_flat_nodes,
            ]
            v_nodes = [
                *v_past_slice_nodes,
                onnx.helper.make_node(
                    "Transpose",
                    [v_past],
                    [v_past_bnsk],
                    name=f"{prefix}__gqa_v_past_to_bnsk",
                    perm=[0, 2, 1, 3],
                ),
                *v_current_slice_nodes,
                *v_current_flat_nodes,
            ]
            seq_lens_name = f"{prefix}__gqa_seq_lens"
            total_sequence_length_name = f"{prefix}__gqa_total_sequence_length"
            new_initializers.append(
                onnx.numpy_helper.from_array(
                    np.full((int(q_shape[0]),), past_length, dtype=np.int32),
                    seq_lens_name,
                )
            )
            new_initializers.append(
                onnx.numpy_helper.from_array(
                    np.asarray([int(k_shape[1])], dtype=np.int32),
                    total_sequence_length_name,
                )
            )
            gqa_inputs = [
                q_flat,
                k_current_flat,
                v_current_flat,
                k_past_bnsk,
                v_past_bnsk,
                seq_lens_name,
                total_sequence_length_name,
            ]
            gqa_outputs = [
                gqa_output_name,
                f"{prefix}__gqa_present_key_unused",
                f"{prefix}__gqa_present_value_unused",
            ]
            rewrite_kind = "temporal_group_query_attention"
        else:
            past_length = 0
            k_flat, k_nodes = flatten_heads(k_compact_name, k_shape, prefix=f"{prefix}__gqa_k")
            v_flat, v_nodes = flatten_heads(v_compact_name, v_shape, prefix=f"{prefix}__gqa_v")
            gqa_inputs = [q_flat, k_flat, v_flat]
            gqa_outputs = [gqa_output_name]
            rewrite_kind = "spatial_group_query_attention"

        gqa_nodes = [
            onnx.helper.make_node(
                "GroupQueryAttention",
                gqa_inputs,
                gqa_outputs,
                name=f"{prefix}__group_query_attention",
                domain="com.microsoft",
                num_heads=num_heads,
                kv_num_heads=kv_num_heads,
                scale=float(scale),
                softcap=0.0,
                do_rotary=0,
                rotary_interleaved=0,
                smooth_softmax=0,
                local_window_size=-1,
            )
        ]
        if gqa_squeeze_axes:
            squeeze_axes_name = f"{prefix}__gqa_output_squeeze_axes"
            new_initializers.append(
                onnx.numpy_helper.from_array(
                    np.asarray(gqa_squeeze_axes, dtype=np.int64), squeeze_axes_name
                )
            )
            gqa_nodes.append(
                onnx.helper.make_node(
                    "Squeeze",
                    [gqa_output_name, squeeze_axes_name],
                    [output_name],
                    name=f"{prefix}__gqa_output_squeeze",
                )
            )
        replacements[node_key(second)] = [
            *q_nodes,
            *k_nodes,
            *v_nodes,
            *gqa_nodes,
        ]
        skip_nodes.update(
            {
                node_key(first),
                node_key(softmax),
                node_key(k_gather),
                node_key(v_gather),
                *merge_skip_nodes,
            }
        )
        if mul is not None:
            skip_nodes.add(node_key(mul))
        rewrites["group_query_attention"] += 1
        rewrites[rewrite_kind] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "kind": rewrite_kind,
                    "first_einsum": first.name,
                    "second_einsum": second.name,
                    "output": output_name,
                    "query_shape": list(q_shape),
                    "key_shape": list(k_shape),
                    "num_heads": num_heads,
                    "kv_num_heads": kv_num_heads,
                    "head_dim": head_dim,
                    "past_length": past_length,
                    "scale": float(scale),
                    "output_squeeze_axes": list(gqa_squeeze_axes),
                }
            )

    if rewrites:
        rewritten_nodes: list[onnx.NodeProto] = []
        for node in model.graph.node:
            key = node_key(node)
            if key in skip_nodes:
                continue
            if key in replacements:
                rewritten_nodes.extend(replacements[key])
            else:
                rewritten_nodes.append(node)

        used_inputs = {
            input_name for node in rewritten_nodes for input_name in node.input if input_name
        }
        kept_initializers = [
            initializer
            for initializer in model.graph.initializer
            if initializer.name in used_inputs
        ]
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept_initializers)
        model.graph.initializer.extend(new_initializers)
        external_data_path(path).unlink(missing_ok=True)
        onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = (
        "GroupQueryAttention",
        "Einsum",
        "Softmax",
        "Mul",
        "Gather",
        "Split",
        "Squeeze",
        "Concat",
        "Gemm",
    )
    return {
        "enabled": True,
        "tool": "custom_group_query_attention_rewrite",
        "reason": (
            "Replace mask-free manual GQA attention islands with "
            "com.microsoft::GroupQueryAttention and fold the following head-merge "
            "layout into the fused op output."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
    }


def fuse_manual_mha_attention_for_webgpu(
    path: Path,
    enabled: bool,
    include_bhqd_attention: bool = False,
    include_attention_bias: bool = False,
) -> dict[str, Any]:
    before = op_counts(path)
    if not enabled:
        return {
            "enabled": False,
            "reason": "--fuse_mha_attention not set",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
            "include_bhqd_attention": include_bhqd_attention,
            "include_attention_bias": include_attention_bias,
        }

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
    initializers = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    producer = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    def node_key(node: onnx.NodeProto) -> str:
        return node.name or node.output[0]

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def equation(node: onnx.NodeProto) -> str:
        value = attr_value(node, "equation", b"")
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def single_consumer(value_name: str, op_type: str | None = None) -> onnx.NodeProto | None:
        value_consumers = consumers.get(value_name, [])
        if len(value_consumers) != 1:
            return None
        consumer = value_consumers[0]
        if op_type is not None and consumer.op_type != op_type:
            return None
        return consumer

    def const_scalar(name: str) -> float | None:
        if name not in initializers:
            return None
        array = np.asarray(initializers[name])
        if array.size != 1:
            return None
        return float(array.reshape(()))

    def parse_scaled_score(
        value_name: str,
    ) -> tuple[onnx.NodeProto | None, float | None, onnx.NodeProto | None]:
        candidate = producer.get(value_name)
        if candidate is not None and candidate.op_type == "Einsum":
            return candidate, 1.0, None
        if candidate is None or candidate.op_type != "Mul" or len(candidate.input) != 2:
            return None, None, None
        first = None
        scale = None
        for input_name in candidate.input:
            input_producer = producer.get(input_name)
            if input_producer is not None and input_producer.op_type == "Einsum":
                first = input_producer
            else:
                scale = const_scalar(input_name)
        return first, scale, candidate

    def squeeze_axes(node: onnx.NodeProto) -> tuple[int, ...] | None:
        if node.op_type != "Squeeze":
            return None
        if len(node.input) >= 2 and node.input[1] in initializers:
            return tuple(int(axis) for axis in np.asarray(initializers[node.input[1]]).reshape(-1))
        axes = attr_value(node, "axes", None)
        if axes is None:
            return None
        return tuple(int(axis) for axis in axes)

    def flattened_head_merge(
        output_name: str,
        num_heads: int,
        head_dim: int,
    ) -> tuple[str, set[str], tuple[int, ...]] | None:
        split = single_consumer(output_name, "Split")
        if split is None or int(attr_value(split, "axis", 0)) != 2:
            return None
        if len(split.output) != num_heads:
            return None

        direct_concat: onnx.NodeProto | None = None
        if all(len(consumers.get(split_output, [])) == 1 for split_output in split.output):
            candidate = consumers[split.output[0]][0]
            if (
                candidate.op_type == "Concat"
                and list(candidate.input) == list(split.output)
                and int(attr_value(candidate, "axis", 0)) == 3
            ):
                direct_concat = candidate
        if direct_concat is not None and len(consumers.get(direct_concat.output[0], [])) == 1:
            squeeze = consumers[direct_concat.output[0]][0]
            axes = squeeze_axes(squeeze) if squeeze.op_type == "Squeeze" else None
            concat_shape = value_shapes.get(direct_concat.output[0])
            output_shape = value_shapes.get(squeeze.output[0]) if squeeze is not None else None
            if (
                squeeze.op_type == "Squeeze"
                and axes is not None
                and 2 in axes
                and concat_shape is not None
                and output_shape is not None
                and len(concat_shape) == 4
                and concat_shape[2] == 1
                and concat_shape[-1] == num_heads * head_dim
            ):
                output_squeeze_axes = tuple(axis for axis in axes if axis != 2)
                expected_rank = 3 - len(output_squeeze_axes)
                if len(output_shape) == expected_rank and output_shape[-1] == num_heads * head_dim:
                    return (
                        squeeze.output[0],
                        {node_key(split), node_key(direct_concat), node_key(squeeze)},
                        output_squeeze_axes,
                    )

        squeezed_outputs: list[str] = []
        skip = {node_key(split)}
        concat: onnx.NodeProto | None = None
        output_squeeze_axes: tuple[int, ...] | None = None
        for split_output in split.output:
            squeeze = single_consumer(split_output, "Squeeze")
            axes = squeeze_axes(squeeze) if squeeze is not None else None
            if squeeze is None or axes is None or 2 not in axes:
                return None
            if any(axis not in (0, 1, 2) for axis in axes):
                return None
            candidate_axes = tuple(axis for axis in axes if axis != 2)
            if output_squeeze_axes is None:
                output_squeeze_axes = candidate_axes
            elif output_squeeze_axes != candidate_axes:
                return None
            squeezed_outputs.append(squeeze.output[0])
            skip.add(node_key(squeeze))
            squeeze_consumer = single_consumer(squeeze.output[0], "Concat")
            if squeeze_consumer is None:
                return None
            if concat is None:
                concat = squeeze_consumer
            elif node_key(concat) != node_key(squeeze_consumer):
                return None

        if concat is None or list(concat.input) != squeezed_outputs:
            return None
        concat_axis = int(attr_value(concat, "axis", 0))
        expected_axis = 2 - len(output_squeeze_axes or ())
        if concat_axis != expected_axis:
            return None
        output_shape = value_shapes.get(concat.output[0])
        expected_rank = 3 - len(output_squeeze_axes or ())
        if output_shape is None or len(output_shape) != expected_rank:
            return None
        if output_shape[-1] != num_heads * head_dim:
            return None
        skip.add(node_key(concat))
        return concat.output[0], skip, output_squeeze_axes or ()

    new_initializers: list[onnx.TensorProto] = []

    def flatten_heads(
        source_name: str,
        shape: tuple[int, ...],
        prefix: str,
    ) -> tuple[str, list[onnx.NodeProto]]:
        if len(shape) == 3:
            return source_name, []
        if len(shape) != 4:
            raise ValueError(f"Expected rank-3 or rank-4 attention input, got {shape}")
        num_heads = int(shape[2])
        split_sizes_name = f"{prefix}__split_sizes"
        squeeze_axes_name = f"{prefix}__squeeze_axes"
        new_initializers.append(
            onnx.numpy_helper.from_array(np.ones((num_heads,), dtype=np.int64), split_sizes_name)
        )
        new_initializers.append(
            onnx.numpy_helper.from_array(np.asarray([2], dtype=np.int64), squeeze_axes_name)
        )
        split_outputs = [f"{prefix}__ranked_head_{idx}" for idx in range(num_heads)]
        concat_output = f"{prefix}__pre_squeeze_flat"
        flat_name = f"{prefix}__flat"
        nodes = [
            onnx.helper.make_node(
                "Split",
                [source_name, split_sizes_name],
                split_outputs,
                name=f"{prefix}__head_split",
                axis=2,
            )
        ]
        nodes.append(
            onnx.helper.make_node(
                "Concat",
                split_outputs,
                [concat_output],
                name=f"{prefix}__head_concat",
                axis=3,
            )
        )
        nodes.append(
            onnx.helper.make_node(
                "Squeeze",
                [concat_output, squeeze_axes_name],
                [flat_name],
                name=f"{prefix}__head_squeeze",
            )
        )
        return flat_name, nodes

    def transpose_kv_heads_for_mha(
        source_name: str,
        shape: tuple[int, ...],
        prefix: str,
    ) -> tuple[str, list[onnx.NodeProto]]:
        if len(shape) != 4:
            return flatten_heads(source_name, shape, prefix=prefix)
        bnsd_name = f"{prefix}__bnsd"
        return (
            bnsd_name,
            [
                onnx.helper.make_node(
                    "Transpose",
                    [source_name],
                    [bnsd_name],
                    name=f"{prefix}__to_bnsd",
                    perm=[0, 2, 1, 3],
                )
            ],
        )

    def transpose_query_heads_for_mha(
        source_name: str,
        shape: tuple[int, ...],
        prefix: str,
    ) -> tuple[str, list[onnx.NodeProto]]:
        if len(shape) != 4:
            return flatten_heads(source_name, shape, prefix=prefix)
        bqhd_name = f"{prefix}__bqhd"
        flat_name, flat_nodes = flatten_heads(
            bqhd_name,
            (shape[0], shape[2], shape[1], shape[3]),
            prefix=prefix,
        )
        return (
            flat_name,
            [
                onnx.helper.make_node(
                    "Transpose",
                    [source_name],
                    [bqhd_name],
                    name=f"{prefix}__to_bqhd",
                    perm=[0, 2, 1, 3],
                ),
                *flat_nodes,
            ],
        )

    def infer_bhqd_attention_shapes(
        score_shape: tuple[int, ...] | None,
        output_shape: tuple[int, ...] | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
        if (
            score_shape is None
            or output_shape is None
            or len(score_shape) != 4
            or len(output_shape) != 4
        ):
            return None
        batch, num_heads, query_tokens, key_tokens = (int(dim) for dim in score_shape)
        if (
            batch <= 0
            or num_heads <= 0
            or query_tokens <= 0
            or key_tokens <= 0
            or int(output_shape[0]) != batch
            or int(output_shape[1]) != query_tokens
            or int(output_shape[2]) != num_heads
        ):
            return None
        head_dim = int(output_shape[3])
        if head_dim <= 0:
            return None
        return (
            (batch, num_heads, query_tokens, head_dim),
            (batch, num_heads, key_tokens, head_dim),
            (batch, key_tokens, num_heads, head_dim),
        )

    replacements: dict[str, list[onnx.NodeProto]] = {}
    skip_nodes: set[str] = set()
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    for softmax in model.graph.node:
        if softmax.op_type != "Softmax":
            continue
        bias_add = None
        attention_bias_name = None
        first, scale, mul = parse_scaled_score(softmax.input[0])
        if first is None and include_attention_bias:
            candidate_add = producer.get(softmax.input[0])
            if (
                candidate_add is not None
                and candidate_add.op_type == "Add"
                and len(candidate_add.input) == 2
            ):
                for input_name in candidate_add.input:
                    candidate_first, candidate_scale, candidate_mul = parse_scaled_score(input_name)
                    if candidate_first is not None and candidate_scale is not None:
                        other_input = (
                            candidate_add.input[1]
                            if input_name == candidate_add.input[0]
                            else candidate_add.input[0]
                        )
                        first = candidate_first
                        scale = candidate_scale
                        mul = candidate_mul
                        bias_add = candidate_add
                        attention_bias_name = other_input
                        break
        if first is None or scale is None:
            continue
        second = single_consumer(softmax.output[0], "Einsum")
        first_equation = equation(first)
        second_equation = equation(second) if second is not None else ""
        is_bqhd_attention = (
            first_equation == "bqhd,bkhd->bhqk" and second_equation == "bhqk,bkhd->bqhd"
        )
        is_bhqd_attention = (
            first_equation == "bhqd,bhkd->bhqk" and second_equation == "bhqk,bkhd->bqhd"
        )
        if not (is_bqhd_attention or is_bhqd_attention):
            continue
        if is_bhqd_attention and not include_bhqd_attention:
            continue
        if second.input[0] != softmax.output[0]:
            continue
        if bias_add is not None:
            if single_consumer(bias_add.output[0]) is not softmax:
                continue
            if mul is None:
                if single_consumer(first.output[0]) is not bias_add:
                    continue
            else:
                if single_consumer(first.output[0]) is not mul:
                    continue
                if single_consumer(mul.output[0]) is not bias_add:
                    continue
        elif mul is None or mul.op_type != "Mul":
            if single_consumer(first.output[0]) is not softmax:
                continue
        else:
            if single_consumer(first.output[0]) is not mul:
                continue
            if single_consumer(mul.output[0]) is not softmax:
                continue

        q_name = first.input[0]
        k_name = first.input[1]
        v_name = second.input[1]
        q_shape = value_shapes.get(q_name)
        k_shape = value_shapes.get(k_name)
        v_shape = value_shapes.get(v_name)
        output_shape = value_shapes.get(second.output[0])
        if is_bhqd_attention and (q_shape is None or k_shape is None or v_shape is None):
            inferred_shapes = infer_bhqd_attention_shapes(
                value_shapes.get(first.output[0]),
                output_shape,
            )
            if inferred_shapes is not None:
                q_shape, k_shape, v_shape = inferred_shapes
        if (
            q_shape is None
            or k_shape is None
            or v_shape is None
            or output_shape is None
            or len(q_shape) != 4
            or len(k_shape) != 4
            or len(v_shape) != 4
            or len(output_shape) != 4
        ):
            continue
        if is_bqhd_attention:
            if q_shape[0] != k_shape[0] or q_shape[0] != v_shape[0]:
                continue
            if k_shape[1] != v_shape[1] or k_shape[2:] != v_shape[2:]:
                continue
            if q_shape[2:] != k_shape[2:] or output_shape != q_shape:
                continue
            num_heads = int(q_shape[2])
            head_dim = int(q_shape[3])
        else:
            if q_shape[0] != k_shape[0] or q_shape[0] != v_shape[0]:
                continue
            if q_shape[1] != k_shape[1] or q_shape[1] != v_shape[2]:
                continue
            if q_shape[2] != output_shape[1] or q_shape[3] != output_shape[3]:
                continue
            if k_shape[2] != v_shape[1] or k_shape[3] != v_shape[3]:
                continue
            if output_shape[0] != q_shape[0] or output_shape[2] != q_shape[1]:
                continue
            num_heads = int(q_shape[1])
            head_dim = int(q_shape[3])
        if num_heads <= 0 or head_dim <= 0:
            continue

        merged = flattened_head_merge(second.output[0], num_heads=num_heads, head_dim=head_dim)
        if merged is None:
            continue
        output_name, merge_skip_nodes, output_squeeze_axes = merged

        prefix = node_key(second)
        mha_raw_output = output_name if not output_squeeze_axes else f"{prefix}__mha_flat_output"
        if is_bqhd_attention:
            q_flat, q_nodes = flatten_heads(q_name, q_shape, prefix=f"{prefix}__mha_q")
            k_flat, k_nodes = transpose_kv_heads_for_mha(k_name, k_shape, prefix=f"{prefix}__mha_k")
        else:
            q_flat, q_nodes = transpose_query_heads_for_mha(
                q_name, q_shape, prefix=f"{prefix}__mha_q"
            )
            k_flat, k_nodes = (k_name, [])
        v_flat, v_nodes = transpose_kv_heads_for_mha(v_name, v_shape, prefix=f"{prefix}__mha_v")
        mha_inputs = [q_flat, k_flat, v_flat]
        if attention_bias_name:
            mha_inputs.extend(["", "", attention_bias_name])
        mha_nodes = [
            onnx.helper.make_node(
                "MultiHeadAttention",
                mha_inputs,
                [mha_raw_output],
                name=f"{prefix}__multi_head_attention",
                domain="com.microsoft",
                num_heads=num_heads,
                scale=float(scale),
                mask_filter_value=-10000.0,
            )
        ]
        if output_squeeze_axes:
            squeeze_axes_name = f"{prefix}__mha_output_squeeze_axes"
            new_initializers.append(
                onnx.numpy_helper.from_array(
                    np.asarray(output_squeeze_axes, dtype=np.int64), squeeze_axes_name
                )
            )
            mha_nodes.append(
                onnx.helper.make_node(
                    "Squeeze",
                    [mha_raw_output, squeeze_axes_name],
                    [output_name],
                    name=f"{prefix}__mha_output_squeeze",
                )
            )

        replacements[node_key(second)] = [*q_nodes, *k_nodes, *v_nodes, *mha_nodes]
        skip_nodes.update({node_key(first), node_key(softmax), *merge_skip_nodes})
        if mul is not None and mul.op_type == "Mul":
            skip_nodes.add(node_key(mul))
        if bias_add is not None:
            skip_nodes.add(node_key(bias_add))
        rewrites["multi_head_attention"] += 1
        if is_bqhd_attention:
            rewrites["bqhd_multi_head_attention"] += 1
        else:
            rewrites["bhqd_multi_head_attention"] += 1
        if attention_bias_name:
            rewrites["attention_bias"] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "first_einsum": first.name,
                    "second_einsum": second.name,
                    "output": output_name,
                    "query_shape": list(q_shape),
                    "key_shape": list(k_shape),
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "scale": float(scale),
                    "attention_bias": attention_bias_name,
                    "output_squeeze_axes": list(output_squeeze_axes),
                }
            )

    if rewrites:
        rewritten_nodes: list[onnx.NodeProto] = []
        for node in model.graph.node:
            key = node_key(node)
            if key in skip_nodes:
                continue
            if key in replacements:
                rewritten_nodes.extend(replacements[key])
            else:
                rewritten_nodes.append(node)

        used_inputs = {
            input_name for node in rewritten_nodes for input_name in node.input if input_name
        }
        kept_initializers = [
            initializer
            for initializer in model.graph.initializer
            if initializer.name in used_inputs
        ]
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept_initializers)
        model.graph.initializer.extend(new_initializers)
        external_data_path(path).unlink(missing_ok=True)
        onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = (
        "MultiHeadAttention",
        "Einsum",
        "Softmax",
        "Mul",
        "Gather",
        "Split",
        "Squeeze",
        "Concat",
        "Transpose",
        "Gemm",
        "Reshape",
    )
    return {
        "enabled": True,
        "tool": "custom_multi_head_attention_rewrite",
        "reason": (
            "Replace matched manual attention islands with "
            "com.microsoft::MultiHeadAttention after K/V heads have already been "
            "materialized. This preserves the explicit graph's attention semantics "
            "while testing ORT fused attention kernels."
        ),
        "include_bhqd_attention": include_bhqd_attention,
        "include_attention_bias": include_attention_bias,
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
    }


def rewrite_attention_einsums_as_matmul_for_wasm(path: Path, enabled: bool = True) -> dict[str, Any]:
    before = op_counts(path)
    tracked_ops = ("Einsum", "MatMul", "FusedMatMul", "Transpose", "Softmax", "Gemm")
    if not enabled:
        return {
            "enabled": False,
            "reason": "not a selected WASM attention MatMul artifact",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
            "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
            "tracked_ops_after": {op: int(before.get(op, 0)) for op in tracked_ops},
        }

    model = onnx.load(path.as_posix(), load_external_data=True)

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def equation(node: onnx.NodeProto) -> str:
        value = attr_value(node, "equation", b"")
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    replacements: dict[str, list[onnx.NodeProto]] = {}
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    for node in model.graph.node:
        if node.op_type != "Einsum":
            continue
        node_equation = equation(node)
        prefix = node.name or node.output[0]
        if node_equation == "bhqd,bhkd->bhqk":
            k_t = f"{prefix}__matmul_k_t"
            replacements[prefix] = [
                onnx.helper.make_node(
                    "Transpose",
                    [node.input[1]],
                    [k_t],
                    name=f"{prefix}__matmul_transpose_k",
                    perm=[0, 1, 3, 2],
                ),
                onnx.helper.make_node(
                    "MatMul",
                    [node.input[0], k_t],
                    list(node.output),
                    name=f"{prefix}__matmul",
                ),
            ]
        elif node_equation == "bqhd,bkhd->bhqk":
            q_t = f"{prefix}__matmul_q_t"
            k_t = f"{prefix}__matmul_k_t"
            replacements[prefix] = [
                onnx.helper.make_node(
                    "Transpose",
                    [node.input[0]],
                    [q_t],
                    name=f"{prefix}__matmul_transpose_q",
                    perm=[0, 2, 1, 3],
                ),
                onnx.helper.make_node(
                    "Transpose",
                    [node.input[1]],
                    [k_t],
                    name=f"{prefix}__matmul_transpose_k",
                    perm=[0, 2, 3, 1],
                ),
                onnx.helper.make_node(
                    "MatMul",
                    [q_t, k_t],
                    list(node.output),
                    name=f"{prefix}__matmul",
                ),
            ]
        elif node_equation == "bhqk,bkhd->bqhd":
            v_t = f"{prefix}__matmul_v_t"
            matmul_output = f"{prefix}__matmul_bhqd"
            replacements[prefix] = [
                onnx.helper.make_node(
                    "Transpose",
                    [node.input[1]],
                    [v_t],
                    name=f"{prefix}__matmul_transpose_v",
                    perm=[0, 2, 1, 3],
                ),
                onnx.helper.make_node(
                    "MatMul",
                    [node.input[0], v_t],
                    [matmul_output],
                    name=f"{prefix}__matmul",
                ),
                onnx.helper.make_node(
                    "Transpose",
                    [matmul_output],
                    list(node.output),
                    name=f"{prefix}__matmul_transpose_out",
                    perm=[0, 2, 1, 3],
                ),
            ]
        else:
            continue
        rewrites[node_equation] += 1
        if len(examples) < 12:
            examples.append({"node": node.name, "equation": node_equation})

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_attention_einsum_matmul_rewrite",
            "reason": "No supported attention Einsum nodes found.",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
            "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
            "tracked_ops_after": {op: int(before.get(op, 0)) for op in tracked_ops},
        }

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        key = node.name or node.output[0]
        rewritten_nodes.extend(replacements.get(key, [node]))

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    return {
        "enabled": True,
        "tool": "custom_attention_einsum_matmul_rewrite",
        "reason": (
            "Replace attention Einsum equations with equivalent batched MatMul "
            "and explicit Transpose nodes for the ORT WASM CPU kernel path."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
    }


def rewrite_static_head_merges_for_wasm(path: Path, enabled: bool = True) -> dict[str, Any]:
    before = op_counts(path)
    tracked_ops = ("Split", "Concat", "Squeeze", "Reshape", "Transpose", "MatMul", "Gemm")
    if not enabled:
        return {
            "enabled": False,
            "reason": "not a selected WASM static head-merge artifact",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
            "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
            "tracked_ops_after": {op: int(before.get(op, 0)) for op in tracked_ops},
        }

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
    initializer_arrays = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def axes_input(node: onnx.NodeProto) -> tuple[int, ...] | None:
        if len(node.input) < 2 or node.input[1] not in initializer_arrays:
            return None
        return tuple(
            int(value) for value in np.asarray(initializer_arrays[node.input[1]]).reshape(-1)
        )

    replacements: dict[str, list[onnx.NodeProto]] = {}
    skip_nodes: set[str] = set()
    stale_value_info: set[str] = set()
    new_initializers: list[onnx.TensorProto] = []
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    for split in model.graph.node:
        if split.op_type != "Split" or len(split.input) < 2:
            continue
        if int(attr_value(split, "axis", 0)) != 2:
            continue
        split_sizes = initializer_arrays.get(split.input[1])
        if split_sizes is None:
            continue
        sizes = tuple(int(value) for value in np.asarray(split_sizes).reshape(-1))
        input_shape = value_shapes.get(split.input[0])
        if (
            not sizes
            or set(sizes) != {1}
            or input_shape is None
            or len(input_shape) != 4
            or len(split.output) != len(sizes)
        ):
            continue

        output_consumers = [consumers.get(output_name, []) for output_name in split.output]
        if any(len(value_consumers) != 1 for value_consumers in output_consumers):
            continue
        concat = output_consumers[0][0]
        if (
            concat.op_type != "Concat"
            or any(value_consumers[0] is not concat for value_consumers in output_consumers)
            or list(concat.input) != list(split.output)
            or int(attr_value(concat, "axis", 0)) != 3
        ):
            continue
        squeeze = _single_consumer(consumers, concat.output[0], "Squeeze")
        axes = axes_input(squeeze) if squeeze is not None else None
        if squeeze is None or axes is None or 2 not in axes:
            continue

        num_heads = int(input_shape[2])
        head_dim = int(input_shape[3])
        concat_shape = (int(input_shape[0]), int(input_shape[1]), 1, num_heads * head_dim)
        normalized_axes = tuple(axis if axis >= 0 else axis + len(concat_shape) for axis in axes)
        output_shape = value_shapes.get(squeeze.output[0])
        if (
            num_heads <= 0
            or head_dim <= 0
            or any(axis < 0 or axis >= len(concat_shape) for axis in normalized_axes)
            or any(concat_shape[axis] != 1 for axis in normalized_axes)
        ):
            continue
        expected_shape = tuple(
            dim for index, dim in enumerate(concat_shape) if index not in normalized_axes
        )
        if (
            len(expected_shape) < 2
            or any(dim <= 0 for dim in expected_shape)
            or num_heads <= 0
            or head_dim <= 0
            or output_shape != expected_shape
        ):
            continue

        prefix = node_key(split)
        shape_name = f"{prefix}__static_head_merge_shape"
        new_initializers.append(
            onnx.numpy_helper.from_array(
                np.asarray(expected_shape, dtype=np.int64),
                shape_name,
            )
        )
        replacements[prefix] = [
            onnx.helper.make_node(
                "Reshape",
                [split.input[0], shape_name],
                [squeeze.output[0]],
                name=f"{prefix}__static_head_merge_reshape",
            )
        ]
        skip_nodes.update({node_key(concat), node_key(squeeze)})
        stale_value_info.update(split.output)
        stale_value_info.update(concat.output)
        rewrite_kind = (
            "split_concat_squeeze_to_reshape"
            if len(expected_shape) == 2
            else "ranked_split_concat_squeeze_to_reshape"
        )
        rewrites[rewrite_kind] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "split": split.name,
                    "concat": concat.name,
                    "squeeze": squeeze.name,
                    "input_shape": list(input_shape),
                    "output": squeeze.output[0],
                    "output_shape": list(output_shape),
                    "squeeze_axes": list(normalized_axes),
                    "heads": num_heads,
                    "head_dim": head_dim,
                }
            )

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_static_head_merge_wasm_rewrite",
            "reason": "No eligible Split -> Concat -> Squeeze head-merge islands found.",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
            "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
            "tracked_ops_after": {op: int(before.get(op, 0)) for op in tracked_ops},
        }

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        key = node_key(node)
        if key in skip_nodes:
            continue
        rewritten_nodes.extend(replacements.get(key, [node]))

    kept_value_info = [
        value_info for value_info in model.graph.value_info if value_info.name not in stale_value_info
    ]
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    model.graph.initializer.extend(new_initializers)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    return {
        "enabled": True,
        "tool": "custom_static_head_merge_wasm_rewrite",
        "reason": (
            "Replace static attention head-merge Split/Concat/Squeeze islands with "
            "one equivalent Reshape for the ORT WASM CPU path."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
    }


def rewrite_singleton_key_attention_for_wasm(path: Path, enabled: bool = True) -> dict[str, Any]:
    before = op_counts(path)
    tracked_ops = (
        "MatMul",
        "Softmax",
        "Transpose",
        "RotaryEmbedding",
        "Gather",
        "Unsqueeze",
        "SimplifiedLayerNormalization",
    )
    if not enabled:
        return {
            "enabled": False,
            "reason": "not a selected WASM singleton-key attention artifact",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
            "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
            "tracked_ops_after": {op: int(before.get(op, 0)) for op in tracked_ops},
        }

    model = onnx.load(path.as_posix(), load_external_data=True)
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
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

    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    replacements: dict[str, str] = {}
    skip_nodes: set[str] = set()
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    for score_matmul in model.graph.node:
        if score_matmul.op_type != "MatMul" or len(score_matmul.output) != 1:
            continue
        score_shape = value_shapes.get(score_matmul.output[0])
        if score_shape is None or len(score_shape) < 2 or score_shape[-2:] != (1, 1):
            continue
        softmax = _single_consumer(consumers, score_matmul.output[0], "Softmax")
        value_matmul = _single_consumer(consumers, softmax.output[0], "MatMul") if softmax else None
        if (
            softmax is None
            or value_matmul is None
            or len(value_matmul.input) < 2
            or value_matmul.input[0] != softmax.output[0]
            or len(value_matmul.output) != 1
        ):
            continue
        value_input = value_matmul.input[1]
        if value_shapes.get(value_input) != value_shapes.get(value_matmul.output[0]):
            continue

        replacements[value_matmul.output[0]] = value_input
        skip_nodes.update({node_key(score_matmul), node_key(softmax), node_key(value_matmul)})
        rewrites["singleton_score_softmax_value_bypass"] += 1
        if len(examples) < 8:
            examples.append(
                {
                    "score_matmul": score_matmul.name,
                    "softmax": softmax.name,
                    "value_matmul": value_matmul.name,
                    "score_shape": list(score_shape),
                    "value_input": value_input,
                    "output": value_matmul.output[0],
                }
            )

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_singleton_key_attention_wasm_rewrite",
            "reason": "No eligible singleton-key MatMul -> Softmax -> MatMul chains found.",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
            "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
            "tracked_ops_after": {op: int(before.get(op, 0)) for op in tracked_ops},
        }

    for node in model.graph.node:
        if node_key(node) in skip_nodes:
            continue
        for index, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[index] = replacements[input_name]

    live_values = {output.name for output in model.graph.output}
    live_nodes: list[onnx.NodeProto] = []
    for node in reversed(model.graph.node):
        if node_key(node) in skip_nodes:
            continue
        if any(output_name in live_values for output_name in node.output):
            live_nodes.append(node)
            live_values.update(input_name for input_name in node.input if input_name)
    live_nodes.reverse()

    kept_value_info = [
        value_info for value_info in model.graph.value_info if value_info.name in live_values
    ]
    del model.graph.node[:]
    model.graph.node.extend(live_nodes)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    return {
        "enabled": True,
        "tool": "custom_singleton_key_attention_wasm_rewrite",
        "reason": (
            "Bypass attention chains whose score tensor has a singleton key axis. "
            "Softmax over a 1-wide axis is exactly one, so the attention result is "
            "the value tensor."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
    }


def rewrite_decoder_rmsnorm_as_primitives_for_wasm(
    path: Path, enabled: bool = True
) -> dict[str, Any]:
    before = op_counts(path)
    tracked_ops = (
        "SimplifiedLayerNormalization",
        "Mul",
        "ReduceMean",
        "Add",
        "Sqrt",
        "Div",
    )
    if not enabled:
        return {
            "enabled": False,
            "reason": "not a selected WASM decoder primitive RMSNorm artifact",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
            "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
            "tracked_ops_after": {op: int(before.get(op, 0)) for op in tracked_ops},
        }

    model = onnx.load(path.as_posix(), load_external_data=True)
    rewritten_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    for node in model.graph.node:
        if (
            node.op_type != "SimplifiedLayerNormalization"
            or len(node.input) < 2
            or len(node.output) != 1
        ):
            rewritten_nodes.append(node)
            continue

        prefix = node.name or node.output[0]
        axis = int(attr_value(node, "axis", -1))
        epsilon = float(attr_value(node, "epsilon", 1e-6))
        x_input = node.input[0]
        scale_input = node.input[1]
        square_output = f"{prefix}__primitive_square"
        mean_output = f"{prefix}__primitive_mean"
        add_output = f"{prefix}__primitive_add_eps"
        sqrt_output = f"{prefix}__primitive_sqrt"
        div_output = f"{prefix}__primitive_div"
        axes_name = f"{prefix}__primitive_axes"
        epsilon_name = f"{prefix}__primitive_epsilon"

        new_initializers.extend(
            [
                onnx.numpy_helper.from_array(np.asarray([axis], dtype=np.int64), axes_name),
                onnx.numpy_helper.from_array(np.asarray(epsilon, dtype=np.float32), epsilon_name),
            ]
        )
        rewritten_nodes.extend(
            [
                onnx.helper.make_node(
                    "Mul",
                    [x_input, x_input],
                    [square_output],
                    name=f"{prefix}__primitive_square",
                ),
                onnx.helper.make_node(
                    "ReduceMean",
                    [square_output, axes_name],
                    [mean_output],
                    name=f"{prefix}__primitive_mean",
                    keepdims=1,
                ),
                onnx.helper.make_node(
                    "Add",
                    [mean_output, epsilon_name],
                    [add_output],
                    name=f"{prefix}__primitive_add_eps",
                ),
                onnx.helper.make_node(
                    "Sqrt",
                    [add_output],
                    [sqrt_output],
                    name=f"{prefix}__primitive_sqrt",
                ),
                onnx.helper.make_node(
                    "Div",
                    [x_input, sqrt_output],
                    [div_output],
                    name=f"{prefix}__primitive_div",
                ),
                onnx.helper.make_node(
                    "Mul",
                    [div_output, scale_input],
                    list(node.output),
                    name=f"{prefix}__primitive_scale",
                ),
            ]
        )
        rewrites["simplified_layer_norm_to_rmsnorm_primitives"] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "node": node.name,
                    "axis": axis,
                    "epsilon": epsilon,
                    "output": node.output[0],
                }
            )

    if not rewrites:
        return {
            "enabled": True,
            "tool": "custom_decoder_rmsnorm_primitive_wasm_rewrite",
            "reason": "No SimplifiedLayerNormalization nodes found.",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
            "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
            "tracked_ops_after": {op: int(before.get(op, 0)) for op in tracked_ops},
        }

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(new_initializers)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    return {
        "enabled": True,
        "tool": "custom_decoder_rmsnorm_primitive_wasm_rewrite",
        "reason": (
            "Replace decoder SimplifiedLayerNormalization with equivalent RMSNorm "
            "primitive arithmetic for the ORT WASM CPU path."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
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
            cos_cache = add_cache_initializer(cos_const, transposed_sequence_length, half_dim)
            sin_cache = add_cache_initializer(sin_const, transposed_sequence_length, half_dim)
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


def rewrite_one_position_rotary_transposes_for_webgpu(path: Path) -> dict[str, Any]:
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
    initializer_arrays = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def has_perm(node: onnx.NodeProto, perm: tuple[int, ...]) -> bool:
        return tuple(int(axis) for axis in attr_value(node, "perm", ())) == perm

    replacements: dict[str, onnx.NodeProto] = {}
    skip_nodes: set[str] = set()
    stale_value_info: set[str] = set()
    new_initializers: list[onnx.TensorProto] = []
    rewrites = Counter()
    examples: list[dict[str, Any]] = []

    for pre_transpose in model.graph.node:
        if (
            pre_transpose.op_type != "Transpose"
            or len(pre_transpose.input) != 1
            or len(pre_transpose.output) != 1
            or not has_perm(pre_transpose, (0, 2, 1, 3))
        ):
            continue
        input_shape = value_shapes.get(pre_transpose.input[0])
        if (
            input_shape is None
            or len(input_shape) != 4
            or input_shape[1] != 1
            or input_shape[2] not in (2, 8)
        ):
            continue
        rotary_consumers = consumers.get(pre_transpose.output[0], [])
        if len(rotary_consumers) != 1 or rotary_consumers[0].op_type != "RotaryEmbedding":
            continue
        rotary = rotary_consumers[0]
        if len(rotary.input) < 4 or len(rotary.output) != 1:
            continue
        post_consumers = consumers.get(rotary.output[0], [])
        if len(post_consumers) != 1 or post_consumers[0].op_type != "Transpose":
            continue
        post_transpose = post_consumers[0]
        if (
            len(post_transpose.input) != 1
            or len(post_transpose.output) != 1
            or not has_perm(post_transpose, (0, 2, 1, 3))
        ):
            continue
        position_ids = initializer_arrays.get(rotary.input[1])
        cos_cache = initializer_arrays.get(rotary.input[2])
        sin_cache = initializer_arrays.get(rotary.input[3])
        if (
            position_ids is None
            or np.asarray(position_ids).size != 1
            or int(np.asarray(position_ids).reshape(())) != 0
            or cos_cache is None
            or sin_cache is None
            or cos_cache.shape != sin_cache.shape
            or len(cos_cache.shape) != 2
            or cos_cache.shape[0] != 1
        ):
            continue

        repeated_sequence = int(input_shape[2])
        cos_name = f"{node_key(rotary)}__direct_repeat_cos_s{repeated_sequence}"
        sin_name = f"{node_key(rotary)}__direct_repeat_sin_s{repeated_sequence}"
        new_initializers.append(
            onnx.numpy_helper.from_array(
                np.repeat(cos_cache, repeated_sequence, axis=0).astype(cos_cache.dtype),
                cos_name,
            )
        )
        new_initializers.append(
            onnx.numpy_helper.from_array(
                np.repeat(sin_cache, repeated_sequence, axis=0).astype(sin_cache.dtype),
                sin_name,
            )
        )
        replacements[node_key(pre_transpose)] = onnx.helper.make_node(
            "RotaryEmbedding",
            [pre_transpose.input[0], rotary.input[1], cos_name, sin_name],
            [post_transpose.output[0]],
            name=f"{node_key(rotary)}__direct_repeated_onepos",
            domain=rotary.domain,
            interleaved=int(attr_value(rotary, "interleaved", 0)),
            num_heads=1,
            rotary_embedding_dim=int(attr_value(rotary, "rotary_embedding_dim", 0)),
            scale=float(attr_value(rotary, "scale", 1.0)),
        )
        skip_nodes.update({node_key(rotary), node_key(post_transpose)})
        stale_value_info.update(pre_transpose.output)
        stale_value_info.update(rotary.output)
        rewrites["direct_repeated_one_position_rotary"] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "pre_transpose": pre_transpose.name,
                    "rotary": rotary.name,
                    "post_transpose": post_transpose.name,
                    "input_shape": list(input_shape),
                    "repeated_sequence": repeated_sequence,
                    "output": post_transpose.output[0],
                }
            )

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_one_position_rotary_transpose_rewrite",
            "reason": "no eligible one-position RotaryEmbedding transpose wrappers found",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": {},
            "rewrite_examples": [],
        }

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        key = node_key(node)
        if key in skip_nodes:
            continue
        rewritten_nodes.append(replacements.get(key, node))

    used_inputs = {
        input_name for node in rewritten_nodes for input_name in node.input if input_name
    }
    kept_initializers = [
        initializer for initializer in model.graph.initializer if initializer.name in used_inputs
    ]
    kept_value_info = [
        value for value in model.graph.value_info if value.name not in stale_value_info
    ]
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    model.graph.initializer.extend(new_initializers)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Transpose", "RotaryEmbedding", "Einsum", "Gemm")
    return {
        "enabled": True,
        "tool": "custom_one_position_rotary_transpose_rewrite",
        "reason": (
            "Replace one-position Transpose -> RotaryEmbedding -> Transpose islands "
            "with a direct-layout RotaryEmbedding whose single cos/sin row is repeated "
            "across the true head axis. The rotation is identical because the position "
            "is fixed to zero for these one-token temporal branches."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": {key: int(value) for key, value in rewrites.items()},
        "rewrite_examples": examples,
    }


def rewrite_final_output_head_slice_transposes_for_webgpu(path: Path) -> dict[str, Any]:
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
    initializer_arrays = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def int_initializer(name: str) -> tuple[int, ...] | None:
        value = initializer_arrays.get(name)
        if value is None:
            return None
        return tuple(int(item) for item in np.asarray(value).reshape(-1))

    replacements: dict[str, onnx.NodeProto] = {}
    squeeze_axis_replacements: dict[str, str] = {}
    skip_nodes: set[str] = set()
    stale_value_info: set[str] = set()
    new_initializers: list[onnx.TensorProto] = []
    examples: list[dict[str, Any]] = []

    for transpose in model.graph.node:
        if (
            transpose.op_type != "Transpose"
            or len(transpose.input) != 1
            or len(transpose.output) != 1
            or tuple(int(axis) for axis in attr_value(transpose, "perm", ())) != (0, 2, 1, 3)
        ):
            continue
        input_shape = value_shapes.get(transpose.input[0])
        output_shape = value_shapes.get(transpose.output[0])
        if (
            input_shape is None
            or output_shape is None
            or len(input_shape) != 4
            or len(output_shape) != 4
            or input_shape[0] != 1
            or input_shape[2] != 1
            or output_shape[0] != 1
            or output_shape[1] != 1
        ):
            continue
        slice_users = consumers.get(transpose.output[0], [])
        if len(slice_users) != 1 or slice_users[0].op_type != "Slice":
            continue
        slice_node = slice_users[0]
        if len(slice_node.input) < 4 or len(slice_node.output) != 1:
            continue
        starts = int_initializer(slice_node.input[1])
        ends = int_initializer(slice_node.input[2])
        axes = int_initializer(slice_node.input[3])
        steps = int_initializer(slice_node.input[4]) if len(slice_node.input) > 4 else None
        if (
            starts != (0, 0, 4, 0)
            or ends != output_shape
            or axes != (0, 1, 2, 3)
            or (steps is not None and steps != (1, 1, 1, 1))
        ):
            continue
        squeeze_users = consumers.get(slice_node.output[0], [])
        if len(squeeze_users) != 1 or squeeze_users[0].op_type != "Squeeze":
            continue
        squeeze = squeeze_users[0]
        if len(squeeze.input) < 2 or int_initializer(squeeze.input[1]) != (0, 1):
            continue

        direct_starts_name = f"{node_key(slice_node)}__pre_transpose_starts"
        direct_ends_name = f"{node_key(slice_node)}__pre_transpose_ends"
        direct_axes_name = f"{node_key(slice_node)}__pre_transpose_axes"
        squeeze_axes_name = f"{node_key(squeeze)}__pre_transpose_squeeze_axes"
        new_initializers.extend(
            [
                onnx.numpy_helper.from_array(
                    np.asarray([0, 4, 0, 0], dtype=np.int64),
                    direct_starts_name,
                ),
                onnx.numpy_helper.from_array(
                    np.asarray(input_shape, dtype=np.int64),
                    direct_ends_name,
                ),
                onnx.numpy_helper.from_array(
                    np.asarray([0, 1, 2, 3], dtype=np.int64),
                    direct_axes_name,
                ),
                onnx.numpy_helper.from_array(
                    np.asarray([0, 2], dtype=np.int64),
                    squeeze_axes_name,
                ),
            ]
        )
        replacements[node_key(slice_node)] = onnx.helper.make_node(
            "Slice",
            [transpose.input[0], direct_starts_name, direct_ends_name, direct_axes_name],
            [slice_node.output[0]],
            name=f"{node_key(slice_node)}__pre_transpose",
        )
        squeeze_axis_replacements[node_key(squeeze)] = squeeze_axes_name
        skip_nodes.add(node_key(transpose))
        stale_value_info.update(transpose.output)
        stale_value_info.update(slice_node.output)
        if len(examples) < 12:
            examples.append(
                {
                    "transpose": transpose.name,
                    "slice": slice_node.name,
                    "squeeze": squeeze.name,
                    "input_shape": list(input_shape),
                    "old_slice_shape": list(output_shape),
                    "output": squeeze.output[0],
                }
            )

    if not replacements:
        return {
            "enabled": True,
            "tool": "custom_final_output_head_slice_transpose_rewrite",
            "reason": "no eligible final output-head Transpose -> Slice -> Squeeze islands found",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": 0,
            "rewrite_examples": [],
        }

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        key = node_key(node)
        if key in skip_nodes:
            continue
        replacement = replacements.get(key)
        if replacement is not None:
            rewritten_nodes.append(replacement)
            continue
        if key in squeeze_axis_replacements:
            copied = copy.deepcopy(node)
            copied.input[1] = squeeze_axis_replacements[key]
            rewritten_nodes.append(copied)
            continue
        rewritten_nodes.append(node)

    kept_value_info = [
        value for value in model.graph.value_info if value.name not in stale_value_info
    ]
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(new_initializers)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Transpose", "Slice", "Squeeze", "Gemm")
    return {
        "enabled": True,
        "tool": "custom_final_output_head_slice_transpose_rewrite",
        "reason": (
            "Slice the final output-head token range before the singleton-axis "
            "transpose and retarget the following squeeze axes. This is exact for "
            "the full-cache output heads because the transposed axis is singleton."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": len(replacements),
        "rewrite_examples": examples,
    }


def fold_shared_gather_add_constants_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    initializer_arrays = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    output_replacements: dict[str, str] = {}
    removed_nodes: set[str] = set()
    initializer_updates: dict[str, np.ndarray] = {}
    examples: list[dict[str, Any]] = []

    for gather in model.graph.node:
        if gather.op_type != "Gather" or len(gather.input) < 2 or len(gather.output) != 1:
            continue
        data_name = gather.input[0]
        data = initializer_arrays.get(data_name)
        if data is None or data.ndim != 2:
            continue
        axis = 0
        for attr in gather.attribute:
            if attr.name == "axis":
                axis = int(onnx.helper.get_attribute_value(attr))
        if axis != 0:
            continue
        add_users = consumers.get(gather.output[0], [])
        if not add_users or any(user.op_type != "Add" for user in add_users):
            continue

        constants: list[np.ndarray] = []
        for add in add_users:
            if len(add.input) != 2 or len(add.output) != 1:
                constants = []
                break
            const_inputs = [name for name in add.input if name != gather.output[0]]
            if len(const_inputs) != 1:
                constants = []
                break
            const = initializer_arrays.get(const_inputs[0])
            if const is None or const.shape[-1:] != data.shape[-1:]:
                constants = []
                break
            if any(dim != 1 for dim in const.shape[:-1]):
                constants = []
                break
            constants.append(np.asarray(const).reshape(data.shape[-1]))
        if not constants:
            continue
        if any(not np.array_equal(constants[0], other) for other in constants[1:]):
            continue

        initializer_updates[data_name] = (data + constants[0].reshape(1, -1)).astype(data.dtype)
        for add in add_users:
            output_replacements[add.output[0]] = gather.output[0]
            removed_nodes.add(node_key(add))
        if len(examples) < 12:
            examples.append(
                {
                    "gather": gather.name,
                    "initializer": data_name,
                    "removed_adds": [add.name for add in add_users],
                    "fanout": len(add_users),
                    "data_shape": list(data.shape),
                }
            )

    if not removed_nodes:
        return {
            "enabled": True,
            "tool": "custom_shared_gather_add_constant_fold",
            "reason": "no eligible shared Gather + Add(constant) fanout found",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": 0,
            "rewrite_examples": [],
        }

    for initializer in model.graph.initializer:
        update = initializer_updates.get(initializer.name)
        if update is not None:
            initializer.CopyFrom(onnx.numpy_helper.from_array(update, initializer.name))

    for node in model.graph.node:
        if node_key(node) in removed_nodes:
            continue
        for index, input_name in enumerate(node.input):
            node.input[index] = output_replacements.get(input_name, input_name)

    kept_nodes = [node for node in model.graph.node if node_key(node) not in removed_nodes]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    kept_value_info = [
        value for value in model.graph.value_info if value.name not in output_replacements
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Gather", "Add", "Unsqueeze")
    return {
        "enabled": True,
        "tool": "custom_shared_gather_add_constant_fold",
        "reason": (
            "Fold identical constants added to every fanout of a shared embedding "
            "Gather into the embedding initializer, then wire consumers to the "
            "folded Gather output."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": len(removed_nodes),
        "rewrite_examples": examples,
    }


def rewrite_swiglu_rank2_islands_for_webgpu(path: Path) -> dict[str, Any]:
    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    initializer_arrays = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    node_by_output: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for output_name in node.output:
            if output_name:
                node_by_output[output_name] = node
        for input_name in node.input:
            consumers[input_name].append(node)

    def attr_value(node: onnx.NodeProto, name: str, default: Any) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def int_initializer(name: str) -> tuple[int, ...] | None:
        value = initializer_arrays.get(name)
        if value is None:
            return None
        if not np.issubdtype(value.dtype, np.integer):
            return None
        return tuple(int(item) for item in np.asarray(value).reshape(-1))

    def axes_initializer(axes: tuple[int, ...]) -> str:
        target = tuple(int(axis) for axis in axes)
        for name in initializer_arrays:
            if int_initializer(name) == target:
                return name
        name = "swiglu_rank2_unsqueeze_axes_" + "_".join(str(axis) for axis in target)
        tensor = onnx.numpy_helper.from_array(np.asarray(target, dtype=np.int64), name)
        model.graph.initializer.append(tensor)
        initializer_arrays[name] = onnx.numpy_helper.to_array(tensor)
        return name

    def axis_initializer(axis: int) -> str:
        return axes_initializer((axis,))

    rewrites: list[dict[str, Any]] = []
    rewrite_count = 0
    remove_nodes: set[str] = set()
    insert_before: dict[str, onnx.NodeProto] = {}
    stale_value_info: set[str] = set()

    for quick_gelu in model.graph.node:
        if quick_gelu.op_type != "QuickGelu" or len(quick_gelu.input) != 1:
            continue
        quick_input_unsqueeze = node_by_output.get(quick_gelu.input[0])
        if (
            quick_input_unsqueeze is None
            or quick_input_unsqueeze.op_type != "Unsqueeze"
            or len(quick_input_unsqueeze.input) < 2
            or len(quick_input_unsqueeze.output) != 1
        ):
            continue
        singleton_axis = int_initializer(quick_input_unsqueeze.input[1])
        if singleton_axis not in ((0,), (1,)):
            continue
        axis = singleton_axis[0]

        mul_users = consumers.get(quick_gelu.output[0], [])
        if len(mul_users) != 1 or mul_users[0].op_type != "Mul" or len(mul_users[0].input) != 2:
            continue
        mul = mul_users[0]
        other_mul_inputs = [name for name in mul.input if name != quick_gelu.output[0]]
        if len(other_mul_inputs) != 1:
            continue
        gate_input_unsqueeze = node_by_output.get(other_mul_inputs[0])
        if (
            gate_input_unsqueeze is None
            or gate_input_unsqueeze.op_type != "Unsqueeze"
            or len(gate_input_unsqueeze.input) < 2
            or len(gate_input_unsqueeze.output) != 1
            or int_initializer(gate_input_unsqueeze.input[1]) != singleton_axis
        ):
            continue

        post_mul_squeeze_users = consumers.get(mul.output[0], [])
        if (
            len(post_mul_squeeze_users) != 1
            or post_mul_squeeze_users[0].op_type != "Squeeze"
            or len(post_mul_squeeze_users[0].input) < 2
            or len(post_mul_squeeze_users[0].output) != 1
            or int_initializer(post_mul_squeeze_users[0].input[1]) != singleton_axis
        ):
            continue
        post_mul_squeeze = post_mul_squeeze_users[0]
        output_gemm_users = consumers.get(post_mul_squeeze.output[0], [])
        if len(output_gemm_users) != 1 or output_gemm_users[0].op_type != "Gemm":
            continue
        output_gemm = output_gemm_users[0]
        post_gemm_unsqueeze_users = consumers.get(output_gemm.output[0], [])
        if (
            len(post_gemm_unsqueeze_users) != 1
            or post_gemm_unsqueeze_users[0].op_type != "Unsqueeze"
            or len(post_gemm_unsqueeze_users[0].input) < 2
            or len(post_gemm_unsqueeze_users[0].output) != 1
            or int_initializer(post_gemm_unsqueeze_users[0].input[1]) != singleton_axis
        ):
            continue
        post_gemm_unsqueeze = post_gemm_unsqueeze_users[0]
        residual_add_users = consumers.get(post_gemm_unsqueeze.output[0], [])
        if (
            len(residual_add_users) != 1
            or residual_add_users[0].op_type != "Add"
            or len(residual_add_users[0].input) != 2
            or len(residual_add_users[0].output) != 1
        ):
            continue
        residual_add = residual_add_users[0]
        following_users = consumers.get(residual_add.output[0], [])
        following_transpose: onnx.NodeProto | None = None
        remove_post_add_unsqueeze: onnx.NodeProto | None = None
        restored_axes: tuple[int, ...] | None = None
        if (
            len(following_users) == 1
            and following_users[0].op_type == "Transpose"
            and tuple(int(item) for item in attr_value(following_users[0], "perm", ())) == (1, 0, 2)
        ):
            following_transpose = following_users[0]
            restored_axes = (1 - axis,)
        elif (
            len(following_users) == 1
            and following_users[0].op_type == "Unsqueeze"
            and len(following_users[0].input) >= 2
            and int_initializer(following_users[0].input[1]) == (0,)
        ):
            post_add_unsqueeze = following_users[0]
            post_add_transpose_users = consumers.get(post_add_unsqueeze.output[0], [])
            if (
                axis == 1
                and len(post_add_transpose_users) == 1
                and post_add_transpose_users[0].op_type == "Transpose"
                and tuple(int(item) for item in attr_value(post_add_transpose_users[0], "perm", ()))
                == (0, 2, 1, 3)
            ):
                following_transpose = post_add_transpose_users[0]
                remove_post_add_unsqueeze = post_add_unsqueeze
                restored_axes = (0, 1)
            elif (
                axis == 1
                and len(post_add_transpose_users) == 1
                and post_add_transpose_users[0].op_type == "Slice"
            ):
                following_transpose = post_add_unsqueeze
                restored_axes = (0, 2)
        if following_transpose is None or restored_axes is None:
            continue
        residual_inputs = [
            name for name in residual_add.input if name != post_gemm_unsqueeze.output[0]
        ]
        if len(residual_inputs) != 1:
            continue
        residual_input = residual_inputs[0]

        quick_gelu.input[0] = quick_input_unsqueeze.input[0]
        for index, input_name in enumerate(mul.input):
            if input_name == gate_input_unsqueeze.output[0]:
                mul.input[index] = gate_input_unsqueeze.input[0]
        output_gemm.input[0] = mul.output[0]

        residual_rank2 = f"{node_key(residual_add)}__swiglu_rank2_residual"
        insert_before[node_key(residual_add)] = onnx.helper.make_node(
            "Squeeze",
            [residual_input, axis_initializer(axis)],
            [residual_rank2],
            name=f"{node_key(residual_add)}__swiglu_rank2_residual_squeeze",
        )
        for index, input_name in enumerate(residual_add.input):
            if input_name == residual_input:
                residual_add.input[index] = residual_rank2
            elif input_name == post_gemm_unsqueeze.output[0]:
                residual_add.input[index] = output_gemm.output[0]

        following_transpose.op_type = "Unsqueeze"
        del following_transpose.attribute[:]
        del following_transpose.input[:]
        following_transpose.input.extend([residual_add.output[0], axes_initializer(restored_axes)])

        remove_nodes.update(
            {
                node_key(quick_input_unsqueeze),
                node_key(gate_input_unsqueeze),
                node_key(post_mul_squeeze),
                node_key(post_gemm_unsqueeze),
            }
        )
        if remove_post_add_unsqueeze is not None:
            remove_nodes.add(node_key(remove_post_add_unsqueeze))
            stale_value_info.update(remove_post_add_unsqueeze.output)
        stale_value_info.update(
            {
                quick_input_unsqueeze.output[0],
                gate_input_unsqueeze.output[0],
                post_mul_squeeze.output[0],
                post_gemm_unsqueeze.output[0],
                quick_gelu.output[0],
                mul.output[0],
                residual_add.output[0],
                residual_rank2,
            }
        )
        rewrite_count += 1
        if len(rewrites) < 12:
            rewrites.append(
                {
                    "quick_gelu": quick_gelu.name,
                    "removed_unsqueezes": [
                        quick_input_unsqueeze.name,
                        gate_input_unsqueeze.name,
                        post_gemm_unsqueeze.name,
                    ],
                    "removed_squeeze": post_mul_squeeze.name,
                    "residual_add": residual_add.name,
                    "removed_post_add_unsqueeze": (
                        remove_post_add_unsqueeze.name
                        if remove_post_add_unsqueeze is not None
                        else None
                    ),
                    "restored_axes": list(restored_axes),
                }
            )

    if not remove_nodes:
        return {
            "enabled": True,
            "tool": "custom_swiglu_rank2_island_rewrite",
            "reason": "no eligible rank-3 SwiGLU singleton islands found",
            "node_count_before": int(sum(before.values())),
            "node_count_after": int(sum(before.values())),
            "rewrites": 0,
            "rewrite_examples": [],
        }

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        key = node_key(node)
        if key in remove_nodes:
            continue
        if key in insert_before:
            rewritten_nodes.append(insert_before[key])
        rewritten_nodes.append(node)

    used_inputs = {
        input_name for node in rewritten_nodes for input_name in node.input if input_name
    }
    kept_initializers = [
        initializer for initializer in model.graph.initializer if initializer.name in used_inputs
    ]
    kept_value_info = [
        value for value in model.graph.value_info if value.name not in stale_value_info
    ]
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    external_data_path(path).unlink(missing_ok=True)
    onnx.save_model(model, path.as_posix(), save_as_external_data=False)

    after = op_counts(path)
    tracked_ops = ("Unsqueeze", "Squeeze", "Transpose", "QuickGelu", "Mul", "Gemm")
    return {
        "enabled": True,
        "tool": "custom_swiglu_rank2_island_rewrite",
        "reason": (
            "Keep singleton-axis SwiGLU activation islands rank-2 between the "
            "packed SwiGLU split and output Gemm, then restore the singleton axis "
            "at the residual boundary. This is exact because the removed axes are "
            "singleton and QuickGelu/Mul operate elementwise."
        ),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
        "rewrites": rewrite_count,
        "rewrite_examples": rewrites,
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


def main() -> None:
    args = parse_args()
    require_static_phase1_args(args)
    set_sample_export_names(args.sample_steps)
    set_attention_export_lowering(args.attention_lowering)
    set_attention_export_layout(args.attention_layout)

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
    dynamics_sample_append_context_slide_entry_path = (
        args.out_dir / f"{DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME}.onnx"
    )
    manifest_path = args.out_dir / MANIFEST_NAME
    ensure_output(manifest_path, overwrite=args.overwrite)

    def decoder_fn(latent: jax.Array) -> jax.Array:
        return onnx_apply_tokenizer_decoder(
            tokenizer_variables,
            tokenizer_cfg,
            latent,
        )

    def decoder_step_fn(latent: jax.Array) -> jax.Array:
        return onnx_apply_tokenizer_decoder(
            tokenizer_variables,
            tokenizer_cfg,
            latent,
        )

    def decoder_z_step_fn(z: jax.Array) -> jax.Array:
        return onnx_apply_tokenizer_decode_z(
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
        return onnx_apply_dynamics_uncached(
            dynamics_variables,
            dynamics_cfg,
            z,
            actions,
            step_levels,
            signal_levels,
        )

    def dynamics_sample_append_context_slide_entry_fn(
        sample_noise: jax.Array,
        context_noise: jax.Array,
        actions: jax.Array,
        k_cache: jax.Array,
        v_cache: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return onnx_apply_dynamics_cached_sample_step_append_context_full_cache_entries(
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

    if not args.export_cached:
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
            "k_cache": jnp.zeros(dyn_shapes.cache, dtype=jnp.float32),
            "v_cache": jnp.zeros(dyn_shapes.cache, dtype=jnp.float32),
        }
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

    exported_paths = {}
    if not args.export_cached:
        exported_paths[TOKENIZER_DECODER_NAME] = decoder_path
        exported_paths[DYNAMICS_UNCACHED_NAME] = dynamics_path
    if args.export_cached:
        exported_paths.update(
            {
                TOKENIZER_DECODER_STEP_NAME: decoder_step_path,
                TOKENIZER_DECODE_Z_STEP_NAME: decoder_z_step_path,
                DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME: (
                    dynamics_sample_append_context_slide_entry_path
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

    pass_names = {
        "tokenizer_decoder_step": TOKENIZER_DECODER_STEP_NAME,
        "tokenizer_decode_z_step": TOKENIZER_DECODE_Z_STEP_NAME,
        "dynamics_cached_sample_append_context_slide_entry": (
            DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME
        ),
    }
    pass_rewrites = {
        "simplify": simplify_onnx_for_webgpu,
        "optimize": optimize_onnx_for_webgpu,
        "singleton_reshapes": rewrite_singleton_reshapes_for_webgpu,
        "gqa_repeats": rewrite_gqa_repeats_for_webgpu,
        "packed_qkv_head_projection": rewrite_packed_qkv_head_projection_for_webgpu,
        "head_projection_layout": rewrite_head_projection_reshapes_with_layout_ops_for_webgpu,
        "head_projection_einsum": rewrite_head_projection_reshapes_for_webgpu,
        "pack_sibling_gemms": pack_sibling_gemms_for_webgpu,
        "packed_qkv_partial_head_split": rewrite_packed_qkv_split_partial_heads_for_webgpu,
        "q_head_split_gather": rewrite_q_head_split_gather_for_webgpu,
        "slide_static_cache": rewrite_slide_static_cache_ops_for_webgpu,
        "rmsnorm": rewrite_rmsnorm_for_webgpu,
        "skip_simplified_layer_norm": fuse_skip_simplified_layer_norm_for_webgpu,
        "gather_index": rewrite_gather_int64_casts_for_webgpu,
        "rotary_embedding": rewrite_rotary_embedding_for_webgpu,
        "fuse_gqa_attention": fuse_manual_gqa_attention_for_webgpu,
        "fuse_mha_attention": fuse_manual_mha_attention_for_webgpu,
        "attention_einsum_matmul": rewrite_attention_einsums_as_matmul_for_wasm,
        "static_head_merge_wasm": rewrite_static_head_merges_for_wasm,
        "singleton_key_attention_wasm": rewrite_singleton_key_attention_for_wasm,
        "decoder_rmsnorm_primitive_wasm": rewrite_decoder_rmsnorm_as_primitives_for_wasm,
        "squeeze_concat": rewrite_squeeze_concat_for_webgpu,
        "unsqueeze_transpose_squeeze": rewrite_unsqueeze_transpose_squeeze_for_webgpu,
        "attention_scale_folding": fold_attention_scale_into_query_norm_for_webgpu,
        "zero_softmax_bias_adds": remove_zero_softmax_bias_adds_for_webgpu,
        "spatial_qk_head_layout": rewrite_spatial_qk_head_layout_for_webgpu,
        "temporal_attention_bhsd": rewrite_temporal_attention_bhsd_for_webgpu,
        "entry_final_z_only": rewrite_entry_final_z_only_for_webgpu,
    }
    if args.export_target == "wasm":
        pass_results = run_wasm_passes(
            exported_paths,
            wasm_pass_options_from_args(args),
            names=pass_names,
            rewrites=pass_rewrites,
        )
    else:
        pass_results = run_webgpu_passes(
            exported_paths,
            webgpu_pass_options_from_args(args),
            names=pass_names,
            rewrites=pass_rewrites,
        )
    simplification = pass_results["simplification"]
    optimization = pass_results["optimization"]
    layout_rewrite = pass_results["layout_rewrite"]
    gqa_repeat_rewrite = pass_results["gqa_repeat_rewrite"]
    packed_qkv_head_projection_rewrite = pass_results["packed_qkv_head_projection_rewrite"]
    head_projection_rewrite = pass_results["head_projection_rewrite"]
    packed_gemm_rewrite = pass_results["packed_gemm_rewrite"]
    packed_qkv_partial_head_split_rewrite = pass_results["packed_qkv_partial_head_split_rewrite"]
    q_head_split_gather_rewrite = pass_results["q_head_split_gather_rewrite"]
    slide_static_cache_rewrite = pass_results["slide_static_cache_rewrite"]
    rmsnorm_rewrite = pass_results["rmsnorm_rewrite"]
    skip_simplified_layer_norm_rewrite = pass_results["skip_simplified_layer_norm_rewrite"]
    gather_index_rewrite = pass_results["gather_index_rewrite"]
    rotary_embedding_rewrite = pass_results["rotary_embedding_rewrite"]
    fused_gqa_attention_rewrite = pass_results["fused_gqa_attention_rewrite"]
    fused_mha_attention_rewrite = pass_results["fused_mha_attention_rewrite"]
    attention_einsum_matmul_rewrite = pass_results["attention_einsum_matmul_rewrite"]
    static_head_merge_wasm_rewrite = pass_results["static_head_merge_wasm_rewrite"]
    singleton_key_attention_wasm_rewrite = pass_results[
        "singleton_key_attention_wasm_rewrite"
    ]
    decoder_rmsnorm_primitive_wasm_rewrite = pass_results[
        "decoder_rmsnorm_primitive_wasm_rewrite"
    ]
    squeeze_concat_rewrite = pass_results["squeeze_concat_rewrite"]
    unsqueeze_transpose_squeeze_rewrite = pass_results["unsqueeze_transpose_squeeze_rewrite"]
    attention_scale_folding = pass_results["attention_scale_folding"]
    zero_softmax_bias_add_prune = pass_results["zero_softmax_bias_add_prune"]
    spatial_qk_head_layout_rewrite = pass_results["spatial_qk_head_layout_rewrite"]
    temporal_attention_bhsd_rewrite = pass_results["temporal_attention_bhsd_rewrite"]
    final_z_only_rewrite = pass_results["final_z_only_rewrite"]
    split_wasm_dynamics_model_paths = None
    if args.export_cached and args.export_target == "wasm":
        split_wasm_dynamics_model_paths = export_split_wasm_dynamics_models(
            dynamics_sample_append_context_slide_entry_path,
            overwrite=args.overwrite,
        )
    static_output_repairs = {
        name: {"enabled": False, "reason": "no static graph output repair needed"}
        for name in exported_paths
    }
    validation = {
        TOKENIZER_DECODER_NAME: {"skipped": not (args.validate and not args.export_cached)},
        DYNAMICS_UNCACHED_NAME: {"skipped": not (args.validate and not args.export_cached)},
        TOKENIZER_DECODER_STEP_NAME: {"skipped": not (args.validate and args.export_cached)},
        TOKENIZER_DECODE_Z_STEP_NAME: {"skipped": not (args.validate and args.export_cached)},
        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME: {
            "skipped": not (args.validate and args.export_cached)
        },
    }
    if args.validate and not args.export_cached:
        validation[TOKENIZER_DECODER_NAME] = validate_single_output(
            path=decoder_path,
            feeds={"latent": inputs["latent"]},
            output_name="patches",
            expected=decoder_fn(inputs["latent"]),
            atol=args.atol,
            rtol=args.rtol,
        )
    if args.validate:
        if not args.export_cached:
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
            sample_append_context_slide_entry_expected = (
                dynamics_sample_append_context_slide_entry_fn(
                    cached_inputs["z_step"],
                    cached_inputs["z_step"],
                    cached_inputs["actions_step"],
                    cached_inputs["k_cache"],
                    cached_inputs["v_cache"],
                )
            )
            entry_final_z_aliases_pred_z = bool(
                final_z_only_rewrite[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME].get(
                    "final_z_aliases_pred_z", False
                )
            )
            entry_expected = {
                "final_z": sample_append_context_slide_entry_expected[
                    1 if entry_final_z_aliases_pred_z else 0
                ],
                "candidate_k_entry": sample_append_context_slide_entry_expected[2],
                "candidate_v_entry": sample_append_context_slide_entry_expected[3],
            }
            if not entry_final_z_aliases_pred_z:
                entry_expected["pred_z"] = sample_append_context_slide_entry_expected[1]
            validation[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME] = validate_outputs(
                path=dynamics_sample_append_context_slide_entry_path,
                feeds={
                    "sample_noise": cached_inputs["z_step"],
                    "context_noise": cached_inputs["z_step"],
                    "actions": cached_inputs["actions_step"],
                    "k_cache": cached_inputs["k_cache"],
                    "v_cache": cached_inputs["v_cache"],
                },
                expected=entry_expected,
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

    decoder_files = export_file_metadata(decoder_path) if not args.export_cached else None
    dynamics_files = export_file_metadata(dynamics_path) if not args.export_cached else None
    decoder_step_files = export_file_metadata(decoder_step_path) if args.export_cached else None
    decoder_z_step_files = export_file_metadata(decoder_z_step_path) if args.export_cached else None
    dynamics_sample_append_context_slide_entry_files = (
        export_file_metadata(dynamics_sample_append_context_slide_entry_path)
        if args.export_cached
        else None
    )
    entry_manifest_final_z_aliases_pred_z = False
    if args.export_cached:
        entry_manifest_final_z_aliases_pred_z = bool(
            final_z_only_rewrite[DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME].get(
                "final_z_aliases_pred_z", False
            )
        )

    def entry_manifest_output_specs(include_cache_length: bool, final_z_aliases_pred_z: bool):
        outputs = {
            "final_z": tensor_spec("float32", dyn_shapes.step_z),
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
        }
        if include_cache_length:
            outputs["candidate_cache_length"] = tensor_spec("int32", (1,))
        if not final_z_aliases_pred_z:
            outputs["pred_z"] = tensor_spec("float32", dyn_shapes.step_z)
        return outputs

    entry_manifest_outputs = (
        entry_manifest_output_specs(
            include_cache_length=False,
            final_z_aliases_pred_z=entry_manifest_final_z_aliases_pred_z,
        )
        if args.export_cached
        else {}
    )
    exports = []
    if not args.export_cached:
        exports.extend(
            [
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
        )
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
                    "name": DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME,
                    **dynamics_sample_append_context_slide_entry_files,
                    "inputs": {
                        "sample_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "context_noise": tensor_spec("float32", dyn_shapes.step_z),
                        "actions": tensor_spec("int32", dyn_shapes.step_levels),
                        "k_cache": tensor_spec("float32", dyn_shapes.cache),
                        "v_cache": tensor_spec("float32", dyn_shapes.cache),
                    },
                    "outputs": entry_manifest_outputs,
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
                    "cache_update": "slide",
                    "cache_update_contract": "webgpu_inplace_slide_rebase_entry",
                    "final_z_aliases_pred_z": entry_manifest_final_z_aliases_pred_z,
                    "steady_state_full_cache_specialized": True,
                },
            ]
        )

    for entry in exports:
        entry["slide_static_cache_rewrite"] = slide_static_cache_rewrite[entry["name"]]
        entry["packed_qkv_head_projection_rewrite"] = packed_qkv_head_projection_rewrite[
            entry["name"]
        ]
        entry["packed_gemm_rewrite"] = packed_gemm_rewrite[entry["name"]]
        entry["packed_qkv_partial_head_split_rewrite"] = packed_qkv_partial_head_split_rewrite[
            entry["name"]
        ]
        entry["q_head_split_gather_rewrite"] = q_head_split_gather_rewrite[entry["name"]]
        entry["skip_simplified_layer_norm_rewrite"] = skip_simplified_layer_norm_rewrite[
            entry["name"]
        ]
        entry["rotary_embedding_rewrite"] = rotary_embedding_rewrite[entry["name"]]
        entry["fused_gqa_attention_rewrite"] = fused_gqa_attention_rewrite[entry["name"]]
        entry["fused_mha_attention_rewrite"] = fused_mha_attention_rewrite[entry["name"]]
        entry["attention_einsum_matmul_rewrite"] = attention_einsum_matmul_rewrite[
            entry["name"]
        ]
        entry["static_head_merge_wasm_rewrite"] = static_head_merge_wasm_rewrite[
            entry["name"]
        ]
        entry["singleton_key_attention_wasm_rewrite"] = (
            singleton_key_attention_wasm_rewrite[entry["name"]]
        )
        entry["decoder_rmsnorm_primitive_wasm_rewrite"] = (
            decoder_rmsnorm_primitive_wasm_rewrite[entry["name"]]
        )
        entry["squeeze_concat_rewrite"] = squeeze_concat_rewrite[entry["name"]]
        entry["unsqueeze_transpose_squeeze_rewrite"] = unsqueeze_transpose_squeeze_rewrite[
            entry["name"]
        ]
        entry["attention_scale_folding"] = attention_scale_folding[entry["name"]]
        entry["zero_softmax_bias_add_prune"] = zero_softmax_bias_add_prune[entry["name"]]
        entry["spatial_qk_head_layout_rewrite"] = spatial_qk_head_layout_rewrite[entry["name"]]
        entry["temporal_attention_bhsd_rewrite"] = temporal_attention_bhsd_rewrite[entry["name"]]
        entry["final_z_only_rewrite"] = final_z_only_rewrite[entry["name"]]
        entry["static_output_repair"] = static_output_repairs[entry["name"]]

    demo_generation = None
    if args.export_cached:
        demo_generation = {
            "sample_steps": args.sample_steps,
            "context_tau": args.context_tau,
            "sample_cache_policy": "sample_then_append_generated_context",
            "preferred_decoder_export": TOKENIZER_DECODER_STEP_NAME,
            "preferred_full_cache_step_export": (
                DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME
            ),
            "preferred_full_cache_step_export_wasm": (
                DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME
            ),
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
        }

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
        "export_target": args.export_target,
        "attention_export": {
            "implementation": args.attention_lowering,
            "layout": args.attention_layout,
            "manual": "patched_onnx_decomposition",
            "native": "jax2onnx_dot_product_attention",
            "split_gqa": "manual_attention_split_by_kv_head_without_kv_repeat",
            "post_export_gqa_fusion": args.fuse_gqa_attention,
            "post_export_spatial_gqa_fusion": args.fuse_spatial_gqa_attention,
            "post_export_mha_fusion": (
                args.fuse_mha_attention
                or (
                    args.export_target == "wasm"
                    and (args.wasm_mha_dynamics_fusion or args.wasm_mha_decoder_fusion)
                )
            ),
            "wasm_mha_dynamics_fusion": (
                args.export_target == "wasm" and args.wasm_mha_dynamics_fusion
            ),
            "wasm_mha_decoder_fusion": (
                args.export_target == "wasm" and args.wasm_mha_decoder_fusion
            ),
        },
        "layout_rewrite": {
            "singleton_reshape_to_squeeze_unsqueeze": not args.skip_singleton_reshape_rewrite,
            "gqa_repeat_to_gather": not args.skip_singleton_reshape_rewrite,
            "head_projection_reshape_to_einsum": not args.skip_singleton_reshape_rewrite,
            "packed_qkv_head_projection": args.pack_qkv_head_projection,
            "squeeze_concat_factorization": not args.skip_squeeze_concat_rewrite,
            "unsqueeze_transpose_squeeze_collapse": (
                not args.skip_unsqueeze_transpose_squeeze_rewrite
            ),
            "attention_scale_folding": not args.skip_attention_scale_folding,
            "zero_softmax_bias_add_prune": True,
            "spatial_qk_direct_bhsd": not args.skip_spatial_qk_head_layout_rewrite,
            "temporal_attention_bhsd": not args.skip_temporal_attention_bhsd_rewrite,
            "steady_state_entry_final_z_only": args.export_cached,
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
        "demo_generation": demo_generation,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not args.export_cached:
        print(f"Wrote {decoder_path}")
        print(f"Wrote {dynamics_path}")
    if args.export_cached:
        print(f"Wrote {decoder_step_path}")
        print(f"Wrote {decoder_z_step_path}")
        print(f"Wrote {dynamics_sample_append_context_slide_entry_path}")
        if split_wasm_dynamics_model_paths is not None:
            print(f"Wrote {split_wasm_dynamics_model_paths[0]}")
            print(f"Wrote {split_wasm_dynamics_model_paths[1]}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
