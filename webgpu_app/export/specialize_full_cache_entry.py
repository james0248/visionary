#!/usr/bin/env python
import argparse
import copy
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from webgpu_app.export.export_dreamer4_onnx import (
    fold_attention_scale_into_query_norm_for_webgpu,
    fold_shared_gather_add_constants_for_webgpu,
    remove_zero_softmax_bias_adds_for_webgpu,
    pack_sibling_gemms_for_webgpu,
    rewrite_cache_layer_slices_as_gather_for_webgpu,
    rewrite_final_output_head_slice_transposes_for_webgpu,
    rewrite_one_position_rotary_transposes_for_webgpu,
    rewrite_q_head_split_gather_for_webgpu,
    rewrite_packed_qkv_split_partial_heads_for_webgpu,
    rewrite_rotary_embedding_for_webgpu,
    rewrite_swiglu_rank2_islands_for_webgpu,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the packed WebGPU full-cache entry dynamics graph from the exported "
            "full-cache entry graph."
        )
    )
    parser.add_argument("--asset_dir", type=Path, required=True)
    parser.add_argument("--manifest", default="breakout_onnx_manifest.json")
    parser.add_argument(
        "--source_export",
        default="breakout_dynamics_sample_append_context_slide_entry_b1_t1_s2",
    )
    parser.add_argument(
        "--target_export",
        default="breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2",
    )
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument(
        "--skip_pack_gemm",
        action="store_true",
        help="Do not pack sibling QKV/SwiGLU Gemm nodes after specialization.",
    )
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def find_export(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in manifest["exports"]:
        if entry["name"] == name:
            return entry
    raise KeyError(f"Missing export {name}")


def attr_map(node: onnx.NodeProto) -> dict[str, Any]:
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def normalize_axes(axes: Any, rank: int) -> tuple[int, ...]:
    return tuple(int(axis if axis >= 0 else axis + rank) for axis in axes)


def replace_initializers(model: onnx.ModelProto, replacements: dict[str, np.ndarray]) -> None:
    remove = set(replacements)
    kept = [initializer for initializer in model.graph.initializer if initializer.name not in remove]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    for name, value in replacements.items():
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(value), name))


def constant_fold_small_nodes(model: onnx.ModelProto) -> int:
    folded = 0
    changed = True
    while changed:
        changed = False
        initializers = {
            initializer.name: numpy_helper.to_array(initializer)
            for initializer in model.graph.initializer
        }
        new_nodes = []
        replacements: dict[str, np.ndarray] = {}

        for node in model.graph.node:
            output = node.output[0] if len(node.output) == 1 else None
            value = None
            try:
                if output and node.op_type == "Squeeze" and node.input[0] in initializers:
                    data = initializers[node.input[0]]
                    attrs = attr_map(node)
                    if len(node.input) > 1 and node.input[1] in initializers:
                        axes = initializers[node.input[1]].astype(np.int64).tolist()
                    else:
                        axes = attrs.get("axes")
                    value = (
                        np.squeeze(data)
                        if axes is None
                        else np.squeeze(data, axis=normalize_axes(axes, data.ndim))
                    )
                elif output and node.op_type == "Unsqueeze" and node.input[0] in initializers:
                    data = initializers[node.input[0]]
                    attrs = attr_map(node)
                    if len(node.input) > 1 and node.input[1] in initializers:
                        axes = initializers[node.input[1]].astype(np.int64).tolist()
                    else:
                        axes = attrs.get("axes", [])
                    value = data
                    for axis in sorted(int(axis) for axis in axes):
                        value = np.expand_dims(value, axis=axis)
                elif (
                    output
                    and node.op_type == "Gather"
                    and node.input[0] in initializers
                    and node.input[1] in initializers
                ):
                    value = np.take(
                        initializers[node.input[0]],
                        initializers[node.input[1]],
                        axis=int(attr_map(node).get("axis", 0)),
                    )
                elif (
                    output
                    and node.op_type == "Sub"
                    and node.input[0] in initializers
                    and node.input[1] in initializers
                ):
                    value = initializers[node.input[0]] - initializers[node.input[1]]
                elif (
                    output
                    and node.op_type == "Mul"
                    and node.input[0] in initializers
                    and node.input[1] in initializers
                ):
                    value = initializers[node.input[0]] * initializers[node.input[1]]
            except Exception:
                value = None

            if value is None:
                new_nodes.append(node)
            else:
                replacements[output] = np.asarray(value)
                folded += 1
                changed = True

        if changed:
            del model.graph.node[:]
            model.graph.node.extend(new_nodes)
            replace_initializers(model, replacements)

    return folded


def fixed_full_cache_inputs(source_spec: dict[str, Any], context_length: int) -> dict[str, np.ndarray]:
    fixed: dict[str, np.ndarray] = {}
    if "sample_position_index" in source_spec["inputs"]:
        fixed["sample_position_index"] = np.asarray([context_length], dtype=np.int32)
    if "context_position_index" in source_spec["inputs"]:
        fixed["context_position_index"] = np.asarray([context_length - 1], dtype=np.int32)
    if "attention_mask" in source_spec["inputs"]:
        shape = source_spec["inputs"]["attention_mask"]["shape"]
        fixed["attention_mask"] = np.ones(shape, dtype=np.float32)
    return fixed


def specialize_model(
    source_path: Path,
    target_path: Path,
    source_spec: dict[str, Any],
    context_length: int,
    pack_gemm: bool,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="visionary-full-cache-") as tmp_dir:
        tmp_path = Path(tmp_dir) / target_path.name
        model = onnx.load(source_path.as_posix(), load_external_data=True)
        fixed = fixed_full_cache_inputs(source_spec, context_length)
        replace_initializers(model, fixed)
        kept_inputs = [
            graph_input for graph_input in model.graph.input if graph_input.name not in fixed
        ]
        del model.graph.input[:]
        model.graph.input.extend(kept_inputs)
        folded = constant_fold_small_nodes(model)
        onnx.save_model(model, tmp_path.as_posix(), save_as_external_data=False)
        rotary_report = rewrite_rotary_embedding_for_webgpu(tmp_path)

        optimized_path = Path(tmp_dir) / f"{target_path.stem}.optimized.onnx"
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        session_options.optimized_model_filepath = optimized_path.as_posix()
        ort.InferenceSession(
            tmp_path.as_posix(),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        optimized = onnx.load(optimized_path.as_posix(), load_external_data=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(optimized, target_path.as_posix(), save_as_external_data=False)
        pack_report = (
            pack_sibling_gemms_for_webgpu(target_path, pack_qkv=True, pack_swiglu=True)
            if pack_gemm
            else {"enabled": False, "reason": "--skip_pack_gemm"}
        )
        partial_split_report = (
            rewrite_packed_qkv_split_partial_heads_for_webgpu(target_path)
            if pack_gemm
            else {"enabled": False, "reason": "--skip_pack_gemm"}
        )
        q_head_gather_report = (
            rewrite_q_head_split_gather_for_webgpu(target_path)
            if pack_gemm
            else {"enabled": False, "reason": "--skip_pack_gemm"}
        )
        attention_scale_folding_report = fold_attention_scale_into_query_norm_for_webgpu(
            target_path,
            enabled=True,
        )
        zero_softmax_bias_report = remove_zero_softmax_bias_adds_for_webgpu(target_path)
        cache_layer_gather_report = rewrite_cache_layer_slices_as_gather_for_webgpu(target_path)
        one_position_rotary_report = rewrite_one_position_rotary_transposes_for_webgpu(target_path)
        final_output_head_slice_report = rewrite_final_output_head_slice_transposes_for_webgpu(
            target_path
        )
        shared_gather_add_fold_report = fold_shared_gather_add_constants_for_webgpu(target_path)
        swiglu_rank2_report = rewrite_swiglu_rank2_islands_for_webgpu(target_path)

    return {
        "folded_constant_nodes": folded,
        "rotary_embedding_rewrite": rotary_report,
        "packed_gemm_rewrite": pack_report,
        "packed_qkv_partial_head_split_rewrite": partial_split_report,
        "q_head_split_gather_rewrite": q_head_gather_report,
        "attention_scale_folding": attention_scale_folding_report,
        "zero_softmax_bias_add_prune": zero_softmax_bias_report,
        "cache_layer_gather_rewrite": cache_layer_gather_report,
        "one_position_rotary_transpose_rewrite": one_position_rotary_report,
        "final_output_head_slice_transpose_rewrite": final_output_head_slice_report,
        "shared_gather_add_constant_fold": shared_gather_add_fold_report,
        "swiglu_rank2_island_rewrite": swiglu_rank2_report,
        "sha256": sha256_file(target_path),
        "size_bytes": target_path.stat().st_size,
    }


def dtype_for_spec(spec: dict[str, Any]) -> np.dtype:
    dtype = spec["dtype"]
    if dtype == "int32":
        return np.int32
    if dtype == "float32":
        return np.float32
    raise ValueError(f"Unsupported validation dtype {dtype}")


def validation_feeds(
    source_spec: dict[str, Any],
    target_spec: dict[str, Any],
    context_length: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    fixed = fixed_full_cache_inputs(source_spec, context_length)
    target_feeds: dict[str, np.ndarray] = {}
    for name, spec in target_spec["inputs"].items():
        shape = tuple(spec["shape"])
        dtype = dtype_for_spec(spec)
        if dtype == np.int32:
            values = rng.integers(0, 4, size=shape, dtype=np.int32)
        else:
            values = rng.standard_normal(size=shape).astype(dtype)
        target_feeds[name] = values
    source_feeds = {**target_feeds, **fixed}
    return source_feeds, target_feeds


def validate_specialized_export(
    asset_dir: Path,
    source_spec: dict[str, Any],
    target_spec: dict[str, Any],
    context_length: int,
    seed: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    source_path = asset_dir / source_spec["path"]
    target_path = asset_dir / target_spec["path"]
    source_session = ort.InferenceSession(source_path.as_posix(), providers=["CPUExecutionProvider"])
    target_session = ort.InferenceSession(target_path.as_posix(), providers=["CPUExecutionProvider"])
    source_feeds, target_feeds = validation_feeds(source_spec, target_spec, context_length, seed)
    output_names = list(target_spec["outputs"])
    source_outputs = source_session.run(output_names, source_feeds)
    target_outputs = target_session.run(output_names, target_feeds)

    outputs = {}
    passed = True
    for name, source, target in zip(output_names, source_outputs, target_outputs):
        diff = np.abs(source - target)
        max_abs = float(diff.max(initial=0.0))
        mean_abs = float(diff.mean()) if diff.size else 0.0
        ok = bool(np.allclose(source, target, atol=atol, rtol=rtol))
        outputs[name] = {
            "allclose": ok,
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
        }
        passed = passed and ok
    return {
        "source_export": source_spec["name"],
        "context_length": context_length,
        "atol": atol,
        "rtol": rtol,
        "passed": passed,
        "outputs": outputs,
    }


def tensor_spec_from_value_info(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    dtype = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.INT32: "int32",
    }.get(tensor_type.elem_type)
    if dtype is None:
        raise ValueError(f"Unsupported tensor elem_type {tensor_type.elem_type}")
    shape = [
        int(dim.dim_value) if dim.HasField("dim_value") else str(dim.dim_param)
        for dim in tensor_type.shape.dim
    ]
    return {"dtype": dtype, "shape": shape}


def model_io_specs(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    model = onnx.shape_inference.infer_shapes(onnx.load(path.as_posix(), load_external_data=False))
    inputs = {value.name: tensor_spec_from_value_info(value) for value in model.graph.input}
    outputs = {value.name: tensor_spec_from_value_info(value) for value in model.graph.output}
    return inputs, outputs


def update_manifest(
    manifest_path: Path,
    source_spec: dict[str, Any],
    target_name: str,
    target_path: Path,
    specialization: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    source_spec = find_export(manifest, source_spec["name"])
    inputs, outputs = model_io_specs(target_path)
    entry = copy.deepcopy(source_spec)
    entry.update(
        {
            "name": target_name,
            "path": target_path.name,
            "sha256": specialization["sha256"],
            "external_data": [],
            "inputs": inputs,
            "outputs": outputs,
            "full_cache_specialization": {
                "source_export": source_spec["name"],
                "context_length": validation["context_length"],
                "folded_constant_nodes": specialization["folded_constant_nodes"],
                "rotary_embedding_rewrites": specialization["rotary_embedding_rewrite"].get(
                    "rewrites", 0
                ),
                "packed_gemm_rewrites": specialization["packed_gemm_rewrite"].get(
                    "rewrites", {}
                ),
                "packed_qkv_partial_head_split_rewrites": specialization[
                    "packed_qkv_partial_head_split_rewrite"
                ].get("rewrites", {}),
                "q_head_split_gather_rewrites": specialization[
                    "q_head_split_gather_rewrite"
                ].get("rewrites", {}),
                "attention_scale_folding": specialization["attention_scale_folding"].get(
                    "rewrites", {}
                ),
                "zero_softmax_bias_add_prunes": specialization[
                    "zero_softmax_bias_add_prune"
                ].get("rewrites", 0),
                "cache_layer_gather_rewrites": specialization[
                    "cache_layer_gather_rewrite"
                ].get("rewrites", {}),
                "one_position_rotary_transpose_rewrites": specialization[
                    "one_position_rotary_transpose_rewrite"
                ].get("rewrites", {}),
                "final_output_head_slice_transpose_rewrites": specialization[
                    "final_output_head_slice_transpose_rewrite"
                ].get("rewrites", 0),
                "shared_gather_add_constant_folds": specialization[
                    "shared_gather_add_constant_fold"
                ].get("rewrites", 0),
                "swiglu_rank2_island_rewrites": specialization[
                    "swiglu_rank2_island_rewrite"
                ].get("rewrites", 0),
            },
            "packed_gemm_rewrite": specialization["packed_gemm_rewrite"],
            "packed_qkv_partial_head_split_rewrite": specialization[
                "packed_qkv_partial_head_split_rewrite"
            ],
            "q_head_split_gather_rewrite": specialization["q_head_split_gather_rewrite"],
            "attention_scale_folding": specialization["attention_scale_folding"],
            "zero_softmax_bias_add_prune": specialization["zero_softmax_bias_add_prune"],
            "cache_layer_gather_rewrite": specialization["cache_layer_gather_rewrite"],
            "one_position_rotary_transpose_rewrite": specialization[
                "one_position_rotary_transpose_rewrite"
            ],
            "final_output_head_slice_transpose_rewrite": specialization[
                "final_output_head_slice_transpose_rewrite"
            ],
            "shared_gather_add_constant_fold": specialization[
                "shared_gather_add_constant_fold"
            ],
            "swiglu_rank2_island_rewrite": specialization[
                "swiglu_rank2_island_rewrite"
            ],
            "validation": validation,
        }
    )
    manifest["exports"] = [item for item in manifest["exports"] if item["name"] != target_name]
    insert_at = next(
        (
            index + 1
            for index, item in enumerate(manifest["exports"])
            if item["name"] == source_spec["name"]
        ),
        len(manifest["exports"]),
    )
    manifest["exports"].insert(insert_at, entry)
    manifest.setdefault("demo_generation", {})["preferred_full_cache_step_export"] = target_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return entry


def update_collection_manifest(asset_dir: Path, target_path: Path, target_name: str) -> None:
    collection_path = asset_dir.parent / "dream_arcade_assets_manifest.json"
    if not collection_path.exists():
        return
    collection = load_manifest(collection_path)
    game = asset_dir.name
    game_entry = collection.get("games", {}).get(game)
    if not game_entry:
        return
    relative_path = f"{game}/{target_path.name}"
    file_entry = {
        "bytes": target_path.stat().st_size,
        "path": relative_path,
        "sha256": sha256_file(target_path),
    }
    game_entry["files"] = [
        item for item in game_entry.get("files", []) if item.get("path") != relative_path
    ]
    game_entry["files"].append(file_entry)
    game_entry["files"].sort(key=lambda item: item["path"])
    game_entry["total_bytes"] = int(sum(item["bytes"] for item in game_entry["files"]))
    policy = collection.setdefault("asset_policy", {})
    included = policy.setdefault("included_onnx_exports", [])
    if target_name not in included:
        included.append(target_name)
    collection_path.write_text(json.dumps(collection, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    manifest_path = args.asset_dir / args.manifest
    manifest = load_manifest(manifest_path)
    source_spec = find_export(manifest, args.source_export)
    source_path = args.asset_dir / source_spec["path"]
    target_path = args.asset_dir / f"{args.target_export}.onnx"

    before = Counter(node.op_type for node in onnx.load(source_path.as_posix()).graph.node)
    specialization = specialize_model(
        source_path=source_path,
        target_path=target_path,
        source_spec=source_spec,
        context_length=args.context_length,
        pack_gemm=not args.skip_pack_gemm,
    )
    target_inputs, target_outputs = model_io_specs(target_path)
    target_spec = {
        **source_spec,
        "name": args.target_export,
        "path": target_path.name,
        "inputs": target_inputs,
        "outputs": target_outputs,
    }
    validation = validate_specialized_export(
        asset_dir=args.asset_dir,
        source_spec=source_spec,
        target_spec=target_spec,
        context_length=args.context_length,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
    )
    if not validation["passed"]:
        raise AssertionError(json.dumps(validation, indent=2, sort_keys=True))
    entry = update_manifest(
        manifest_path=manifest_path,
        source_spec=source_spec,
        target_name=args.target_export,
        target_path=target_path,
        specialization=specialization,
        validation=validation,
    )
    update_collection_manifest(args.asset_dir, target_path, args.target_export)

    after = Counter(node.op_type for node in onnx.load(target_path.as_posix()).graph.node)
    print(
        json.dumps(
            {
                "target_export": entry["name"],
                "path": target_path.as_posix(),
                "sha256": entry["sha256"],
                "size_bytes": target_path.stat().st_size,
                "node_count_before": int(sum(before.values())),
                "node_count_after": int(sum(after.values())),
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
