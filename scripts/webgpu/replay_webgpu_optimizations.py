from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from scripts.webgpu.export_dreamer4_onnx import (
    DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME,
    DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME,
    DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME,
    DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME,
    DYNAMICS_CACHED_SAMPLE_STEP_NAME,
    DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME,
    DYNAMICS_CACHED_PREFILL_LAYER_NAME,
    DYNAMICS_CACHED_PREFILL_NAME,
    TOKENIZER_DECODE_Z_STEP_NAME,
    external_data_path,
    fold_attention_scale_into_query_norm_for_webgpu,
    fuse_manual_gqa_attention_for_webgpu,
    fuse_manual_mha_attention_for_webgpu,
    fuse_skip_simplified_layer_norm_for_webgpu,
    optimize_onnx_for_webgpu,
    pack_sibling_gemms_for_webgpu,
    rewrite_gather_int64_casts_for_webgpu,
    rewrite_gqa_repeats_for_webgpu,
    rewrite_head_projection_reshapes_for_webgpu,
    rewrite_head_projection_reshapes_with_layout_ops_for_webgpu,
    rewrite_packed_qkv_head_projection_for_webgpu,
    rewrite_rmsnorm_for_webgpu,
    rewrite_rotary_embedding_for_webgpu,
    rewrite_singleton_reshapes_for_webgpu,
    rewrite_slide_static_cache_ops_for_webgpu,
    rewrite_spatial_qk_head_layout_for_webgpu,
    rewrite_squeeze_concat_for_webgpu,
    rewrite_unsqueeze_transpose_squeeze_for_webgpu,
    simplify_onnx_for_webgpu,
)


DEMO_ARTIFACTS = (
    TOKENIZER_DECODE_Z_STEP_NAME,
    DYNAMICS_CACHED_PREFILL_NAME,
    DYNAMICS_CACHED_PREFILL_LAYER_NAME,
    DYNAMICS_CACHED_SAMPLE_STEP_NAME,
    DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME,
    DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME,
    DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME,
    DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME,
    DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the export script's WebGPU graph optimization passes from saved "
            "raw jax2onnx artifacts. This is intended for fast local trials after a "
            "full export has already populated --raw_dir."
        )
    )
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("webgpu_app/assets"))
    parser.add_argument(
        "--artifact",
        action="append",
        default=None,
        help="Artifact name to replay. Defaults to the demo-relevant artifacts.",
    )
    parser.add_argument("--simplify_onnx", action="store_true")
    parser.add_argument("--skip_onnx_optimization", action="store_true")
    parser.add_argument("--skip_singleton_reshape_rewrite", action="store_true")
    parser.add_argument("--skip_squeeze_concat_rewrite", action="store_true")
    parser.add_argument("--skip_unsqueeze_transpose_squeeze_rewrite", action="store_true")
    parser.add_argument("--skip_spatial_qk_head_layout_rewrite", action="store_true")
    parser.add_argument("--skip_attention_scale_folding", action="store_true")
    parser.add_argument(
        "--head_projection_rewrite",
        choices=("einsum", "layout"),
        default="einsum",
    )
    parser.add_argument("--rotary_embedding_rewrite", action="store_true")
    parser.add_argument("--pack_qkv_gemm", action="store_true")
    parser.add_argument("--pack_qkv_head_projection", action="store_true")
    parser.add_argument("--pack_swiglu_gemm", action="store_true")
    parser.add_argument("--fuse_gqa_attention", action="store_true")
    parser.add_argument("--fuse_spatial_gqa_attention", action="store_true")
    parser.add_argument("--fuse_mha_attention", action="store_true")
    return parser.parse_args()


def copy_raw_artifact(raw_dir: Path, out_dir: Path, name: str) -> Path:
    src = raw_dir / f"{name}.onnx"
    if not src.exists():
        raise FileNotFoundError(f"Missing raw artifact: {src}")
    dst = out_dir / src.name
    out_dir.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    external_data_path(dst).unlink(missing_ok=True)
    shutil.copy2(src, dst)
    src_external = external_data_path(src)
    if src_external.exists():
        shutil.copy2(src_external, external_data_path(dst))
    return dst


def run_pass(
    report: dict[str, Any],
    name: str,
    pass_name: str,
    fn: Callable[[], dict[str, Any]],
) -> None:
    result = fn()
    report.setdefault(name, {})[pass_name] = result


def main() -> int:
    args = parse_args()
    artifacts = tuple(args.artifact or DEMO_ARTIFACTS)
    head_projection_rewriter = (
        rewrite_head_projection_reshapes_with_layout_ops_for_webgpu
        if args.head_projection_rewrite == "layout"
        else rewrite_head_projection_reshapes_for_webgpu
    )
    slide_static_names = {
        DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME,
        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_NAME,
        DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME,
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "raw_dir": args.raw_dir.as_posix(),
        "out_dir": args.out_dir.as_posix(),
        "artifacts": {},
    }

    for name in artifacts:
        path = copy_raw_artifact(args.raw_dir, args.out_dir, name)
        report["artifacts"][name] = {"path": path.as_posix()}

        if args.simplify_onnx:
            run_pass(report["artifacts"], name, "simplification", lambda path=path: simplify_onnx_for_webgpu(path))
        if not args.skip_onnx_optimization:
            run_pass(report["artifacts"], name, "optimization", lambda path=path: optimize_onnx_for_webgpu(path))

        if not args.skip_singleton_reshape_rewrite:
            run_pass(report["artifacts"], name, "layout_rewrite", lambda path=path: rewrite_singleton_reshapes_for_webgpu(path))
            run_pass(report["artifacts"], name, "gqa_repeat_rewrite", lambda path=path: rewrite_gqa_repeats_for_webgpu(path))
            run_pass(
                report["artifacts"],
                name,
                "packed_qkv_head_projection_rewrite",
                lambda path=path: rewrite_packed_qkv_head_projection_for_webgpu(
                    path,
                    enabled=args.pack_qkv_head_projection,
                ),
            )
            run_pass(report["artifacts"], name, "head_projection_rewrite", lambda path=path: head_projection_rewriter(path))

        run_pass(
            report["artifacts"],
            name,
            "packed_gemm_rewrite",
            lambda path=path: pack_sibling_gemms_for_webgpu(
                path,
                pack_qkv=args.pack_qkv_gemm,
                pack_swiglu=args.pack_swiglu_gemm,
            ),
        )
        if name in slide_static_names:
            run_pass(report["artifacts"], name, "slide_static_cache_rewrite", lambda path=path: rewrite_slide_static_cache_ops_for_webgpu(path))

        run_pass(report["artifacts"], name, "rmsnorm_rewrite", lambda path=path: rewrite_rmsnorm_for_webgpu(path))
        run_pass(
            report["artifacts"],
            name,
            "skip_simplified_layer_norm_rewrite",
            lambda path=path: fuse_skip_simplified_layer_norm_for_webgpu(path),
        )
        run_pass(report["artifacts"], name, "gather_index_rewrite", lambda path=path: rewrite_gather_int64_casts_for_webgpu(path))

        if args.rotary_embedding_rewrite:
            run_pass(report["artifacts"], name, "rotary_embedding_rewrite", lambda path=path: rewrite_rotary_embedding_for_webgpu(path))

        run_pass(
            report["artifacts"],
            name,
            "fused_gqa_attention_rewrite",
            lambda path=path: fuse_manual_gqa_attention_for_webgpu(
                path,
                enabled=args.fuse_gqa_attention or args.fuse_spatial_gqa_attention,
                fuse_spatial=args.fuse_spatial_gqa_attention,
            ),
        )
        run_pass(
            report["artifacts"],
            name,
            "fused_mha_attention_rewrite",
            lambda path=path: fuse_manual_mha_attention_for_webgpu(
                path,
                enabled=args.fuse_mha_attention,
            ),
        )
        run_pass(
            report["artifacts"],
            name,
            "squeeze_concat_rewrite",
            lambda path=path: rewrite_squeeze_concat_for_webgpu(
                path,
                enabled=not args.skip_squeeze_concat_rewrite,
            ),
        )
        run_pass(
            report["artifacts"],
            name,
            "unsqueeze_transpose_squeeze_rewrite",
            lambda path=path: rewrite_unsqueeze_transpose_squeeze_for_webgpu(
                path,
                enabled=not args.skip_unsqueeze_transpose_squeeze_rewrite,
            ),
        )
        run_pass(
            report["artifacts"],
            name,
            "attention_scale_folding",
            lambda path=path: fold_attention_scale_into_query_norm_for_webgpu(
                path,
                enabled=not args.skip_attention_scale_folding,
            ),
        )
        run_pass(
            report["artifacts"],
            name,
            "spatial_qk_head_layout_rewrite",
            lambda path=path: rewrite_spatial_qk_head_layout_for_webgpu(
                path,
                enabled=not args.skip_spatial_qk_head_layout_rewrite,
            ),
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
