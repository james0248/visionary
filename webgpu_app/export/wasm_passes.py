from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from webgpu_app.export.pass_utils import RewriteFn, disabled, run_all


@dataclass(frozen=True)
class WasmPassOptions:
    simplify_onnx: bool
    simplify_demo_only: bool
    skip_onnx_optimization: bool
    skip_singleton_reshape_rewrite: bool
    pack_qkv_head_projection: bool
    head_projection_rewrite: str
    pack_qkv_gemm: bool
    pack_swiglu_gemm: bool
    rotary_embedding_rewrite: bool
    skip_squeeze_concat_rewrite: bool
    skip_unsqueeze_transpose_squeeze_rewrite: bool
    skip_attention_scale_folding: bool
    skip_spatial_qk_head_layout_rewrite: bool
    skip_temporal_attention_bhsd_rewrite: bool
    wasm_mha_dynamics_fusion: bool
    wasm_mha_decoder_fusion: bool


def wasm_pass_options_from_args(args) -> WasmPassOptions:
    return WasmPassOptions(
        simplify_onnx=args.simplify_onnx,
        simplify_demo_only=args.simplify_demo_only,
        skip_onnx_optimization=args.skip_onnx_optimization,
        skip_singleton_reshape_rewrite=args.skip_singleton_reshape_rewrite,
        pack_qkv_head_projection=args.pack_qkv_head_projection,
        head_projection_rewrite=args.head_projection_rewrite,
        pack_qkv_gemm=args.pack_qkv_gemm,
        pack_swiglu_gemm=args.pack_swiglu_gemm,
        rotary_embedding_rewrite=args.rotary_embedding_rewrite,
        skip_squeeze_concat_rewrite=args.skip_squeeze_concat_rewrite,
        skip_unsqueeze_transpose_squeeze_rewrite=args.skip_unsqueeze_transpose_squeeze_rewrite,
        skip_attention_scale_folding=args.skip_attention_scale_folding,
        skip_spatial_qk_head_layout_rewrite=args.skip_spatial_qk_head_layout_rewrite,
        skip_temporal_attention_bhsd_rewrite=args.skip_temporal_attention_bhsd_rewrite,
        wasm_mha_dynamics_fusion=args.wasm_mha_dynamics_fusion,
        wasm_mha_decoder_fusion=args.wasm_mha_decoder_fusion,
    )


def run_wasm_passes(
    exported_paths: dict[str, Path],
    options: WasmPassOptions,
    names: dict[str, str],
    rewrites: dict[str, RewriteFn],
) -> dict[str, dict[str, dict[str, Any]]]:
    demo_simplification_names = {
        names["tokenizer_decode_z_step"],
        names["dynamics_cached_sample_append_context_slide_entry"],
    }
    if options.simplify_onnx:
        simplification = {
            name: rewrites["simplify"](path)
            if not options.simplify_demo_only or name in demo_simplification_names
            else {
                "enabled": False,
                "reason": "--simplify_demo_only skips artifacts outside the browser demo hot path",
                "tool": "onnxsim",
            }
            for name, path in exported_paths.items()
        }
    else:
        simplification = disabled(exported_paths, "--simplify_onnx not set")

    optimization = (
        disabled(exported_paths, "--skip_onnx_optimization")
        if options.skip_onnx_optimization
        else run_all(exported_paths, rewrites["optimize"])
    )

    if options.skip_singleton_reshape_rewrite:
        layout_rewrite = disabled(exported_paths, "--skip_singleton_reshape_rewrite")
        gqa_repeat_rewrite = disabled(exported_paths, "--skip_singleton_reshape_rewrite")
        head_projection_rewrite = disabled(exported_paths, "--skip_singleton_reshape_rewrite")
        packed_qkv_head_projection_rewrite = disabled(exported_paths, "--skip_singleton_reshape_rewrite")
    else:
        layout_rewrite = run_all(exported_paths, rewrites["singleton_reshapes"])
        gqa_repeat_rewrite = run_all(exported_paths, rewrites["gqa_repeats"])
        packed_qkv_head_projection_rewrite = {
            name: rewrites["packed_qkv_head_projection"](
                path,
                enabled=options.pack_qkv_head_projection,
            )
            for name, path in exported_paths.items()
        }
        head_projection = (
            rewrites["head_projection_layout"]
            if options.head_projection_rewrite == "layout"
            else rewrites["head_projection_einsum"]
        )
        head_projection_rewrite = run_all(exported_paths, head_projection)

    packed_gemm_rewrite = {
        name: rewrites["pack_sibling_gemms"](
            path,
            pack_qkv=options.pack_qkv_gemm,
            pack_swiglu=options.pack_swiglu_gemm,
        )
        for name, path in exported_paths.items()
    }
    final_z_only_artifacts = {
        names["dynamics_cached_sample_append_context_slide_entry"],
    }

    packed_qkv_partial_head_split_rewrite = run_all(exported_paths, rewrites["packed_qkv_partial_head_split"])
    q_head_split_gather_rewrite = run_all(exported_paths, rewrites["q_head_split_gather"])
    slide_static_cache_rewrite = {
        name: {"enabled": False, "reason": "full-cache entry graph has no static slide inputs"}
        for name, path in exported_paths.items()
    }
    rmsnorm_rewrite = run_all(exported_paths, rewrites["rmsnorm"])
    skip_simplified_layer_norm_rewrite = run_all(exported_paths, rewrites["skip_simplified_layer_norm"])
    gather_index_rewrite = run_all(exported_paths, rewrites["gather_index"])
    rotary_embedding_rewrite = (
        run_all(exported_paths, rewrites["rotary_embedding"])
        if options.rotary_embedding_rewrite
        else disabled(exported_paths, "--rotary_embedding_rewrite not set")
    )
    fused_gqa_attention_rewrite = disabled(
        exported_paths,
        "WASM export uses the MHA fusion pass for the matched hot-path attention islands",
    )
    squeeze_concat_rewrite = {
        name: rewrites["squeeze_concat"](path, enabled=not options.skip_squeeze_concat_rewrite)
        for name, path in exported_paths.items()
    }
    unsqueeze_transpose_squeeze_rewrite = {
        name: rewrites["unsqueeze_transpose_squeeze"](
            path, enabled=not options.skip_unsqueeze_transpose_squeeze_rewrite
        )
        for name, path in exported_paths.items()
    }
    attention_scale_folding = {
        name: rewrites["attention_scale_folding"](path, enabled=not options.skip_attention_scale_folding)
        for name, path in exported_paths.items()
    }
    zero_softmax_bias_add_prune = run_all(exported_paths, rewrites["zero_softmax_bias_adds"])
    spatial_qk_head_layout_rewrite = {
        name: rewrites["spatial_qk_head_layout"](path, enabled=not options.skip_spatial_qk_head_layout_rewrite)
        for name, path in exported_paths.items()
    }
    temporal_attention_bhsd_rewrite = {
        name: rewrites["temporal_attention_bhsd"](path, enabled=not options.skip_temporal_attention_bhsd_rewrite)
        for name, path in exported_paths.items()
    }
    final_z_only_rewrite = {
        name: rewrites["entry_final_z_only"](path, enabled=name in final_z_only_artifacts)
        for name, path in exported_paths.items()
    }
    wasm_mha_artifacts = {
        names["dynamics_cached_sample_append_context_slide_entry"],
    }
    wasm_mha_decoder_artifacts = {
        names["tokenizer_decoder_step"],
    }
    fused_mha_attention_rewrite = {
        name: rewrites["fuse_mha_attention"](
            path,
            enabled=(
                (options.wasm_mha_dynamics_fusion and name in wasm_mha_artifacts)
                or (options.wasm_mha_decoder_fusion and name in wasm_mha_decoder_artifacts)
            ),
            include_bhqd_attention=(options.wasm_mha_decoder_fusion and name in wasm_mha_decoder_artifacts),
            include_attention_bias=(options.wasm_mha_decoder_fusion and name in wasm_mha_decoder_artifacts),
        )
        if name in wasm_mha_artifacts or name in wasm_mha_decoder_artifacts
        else rewrites["fuse_mha_attention"](path, enabled=False)
        for name, path in exported_paths.items()
    }
    attention_einsum_matmul_artifacts = {
        names["tokenizer_decoder_step"],
        names["tokenizer_decode_z_step"],
        names["dynamics_cached_sample_append_context_slide_entry"],
    }
    attention_einsum_matmul_rewrite = {
        name: rewrites["attention_einsum_matmul"](path, enabled=name in attention_einsum_matmul_artifacts)
        for name, path in exported_paths.items()
    }
    static_head_merge_artifacts = {
        names["tokenizer_decoder_step"],
        names["tokenizer_decode_z_step"],
        names["dynamics_cached_sample_append_context_slide_entry"],
    }
    static_head_merge_wasm_rewrite = {
        name: rewrites["static_head_merge_wasm"](path, enabled=name in static_head_merge_artifacts)
        for name, path in exported_paths.items()
    }
    singleton_key_attention_artifacts = {
        names["tokenizer_decoder_step"],
        names["tokenizer_decode_z_step"],
    }
    singleton_key_attention_wasm_rewrite = {
        name: rewrites["singleton_key_attention_wasm"](path, enabled=name in singleton_key_attention_artifacts)
        for name, path in exported_paths.items()
    }
    decoder_rmsnorm_primitive_artifacts = {
        names["tokenizer_decoder_step"],
        names["tokenizer_decode_z_step"],
    }
    decoder_rmsnorm_primitive_wasm_rewrite = {
        name: rewrites["decoder_rmsnorm_primitive_wasm"](path, enabled=name in decoder_rmsnorm_primitive_artifacts)
        for name, path in exported_paths.items()
    }

    return {
        "simplification": simplification,
        "optimization": optimization,
        "layout_rewrite": layout_rewrite,
        "gqa_repeat_rewrite": gqa_repeat_rewrite,
        "packed_qkv_head_projection_rewrite": packed_qkv_head_projection_rewrite,
        "head_projection_rewrite": head_projection_rewrite,
        "packed_gemm_rewrite": packed_gemm_rewrite,
        "packed_qkv_partial_head_split_rewrite": packed_qkv_partial_head_split_rewrite,
        "q_head_split_gather_rewrite": q_head_split_gather_rewrite,
        "slide_static_cache_rewrite": slide_static_cache_rewrite,
        "rmsnorm_rewrite": rmsnorm_rewrite,
        "skip_simplified_layer_norm_rewrite": skip_simplified_layer_norm_rewrite,
        "gather_index_rewrite": gather_index_rewrite,
        "rotary_embedding_rewrite": rotary_embedding_rewrite,
        "fused_gqa_attention_rewrite": fused_gqa_attention_rewrite,
        "fused_mha_attention_rewrite": fused_mha_attention_rewrite,
        "attention_einsum_matmul_rewrite": attention_einsum_matmul_rewrite,
        "static_head_merge_wasm_rewrite": static_head_merge_wasm_rewrite,
        "singleton_key_attention_wasm_rewrite": singleton_key_attention_wasm_rewrite,
        "decoder_rmsnorm_primitive_wasm_rewrite": decoder_rmsnorm_primitive_wasm_rewrite,
        "squeeze_concat_rewrite": squeeze_concat_rewrite,
        "unsqueeze_transpose_squeeze_rewrite": unsqueeze_transpose_squeeze_rewrite,
        "attention_scale_folding": attention_scale_folding,
        "zero_softmax_bias_add_prune": zero_softmax_bias_add_prune,
        "spatial_qk_head_layout_rewrite": spatial_qk_head_layout_rewrite,
        "temporal_attention_bhsd_rewrite": temporal_attention_bhsd_rewrite,
        "final_z_only_rewrite": final_z_only_rewrite,
    }
