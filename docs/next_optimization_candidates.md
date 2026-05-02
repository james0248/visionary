# Next ONNX WebGPU Optimization Candidates

Date: 2026-05-02

Scope: source/export-level opportunities that preserve behavior for the `sample_steps=4` browser demo path. Production files were not edited.

## Bottleneck Summary

Current preferred steady-state step artifact:

- `webgpu_app/assets/breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4.onnx`
- Selected by `webgpu_app/bench/benchmark.js` before the full-cache and layer-cache variants.
- Inputs: `sample_noise`, `context_noise`, `actions`, `k_cache`, `v_cache`.
- Outputs: `final_z`, `pred_z`, `candidate_k_entry`, `candidate_v_entry`.
- Cache update is browser-side WebGPU in-place slide/rebase, not a full-cache ONNX output.

Current benchmark result file: `webgpu_app/bench/results/latest.json`.

- Profiling was requested, but ORT WebGPU emitted no raw profiling callbacks: `profiling.available=false`, `raw_events=0`.
- `streaming_frame` mean is `283.98 ms`, but that includes two `100 ms` profiling drain windows per frame. The session timings are more useful:
- Dynamics step/session: `75.83 ms` mean, `73.46 ms` median, `88.12 ms` p95.
- Decoder: `5.69 ms` mean, `5.84 ms` median, `6.10 ms` p95.
- Cache commit shader: `0.014 ms` mean.
- Pack/unpack: `0.004 ms` mean.

Current graph-capture notes in `.codex/onnx_webgpu_progress.md` show the entry-cache artifact is graph-capture compatible, but graph capture does not materially change the true steady-state cost:

- Graph capture, 64 timed frames: dynamics `80.19 / 89.36 / 90.27 ms` mean/median/p95.
- Streaming `86.73 / 96.54 / 97.59 ms` mean/median/p95.

Fresh op counts for the active entry graph:

- Total nodes: `5,950`; file size: `245.6 MB`.
- No `Reshape`, `Cast`, `Less`, `Shape`, or `Size`.
- Hot counts: `716 Einsum`, `366 Gemm`, `119 Softmax`, `478 SimplifiedLayerNormalization`, `119 QuickGelu`, `915 Unsqueeze`, `451 Squeeze`, `304 Concat`, `239 Gather`, `239 Split`, `1,200 Mul`.

Conclusion: previous CPU/provider-boundary issues from `Reshape` are solved on the active graph. The remaining bottleneck is dynamics compute and dispatch count inside the fused sample+append graph. Decoder and cache commit are no longer first-order bottlenecks.

## Candidate 1: Export Fused QKV Projections

Implement an export-only attention module that computes Q/K/V with one projection instead of three separate `Dense`/`Gemm` projections, then splits the packed result into `[q, k, v]`.

Expected impact: medium to high. The active graph has `366 Gemm` nodes; most are attention projections repeated across the unrolled `sample_steps=4` plus context append path. Combining Q/K/V should remove roughly two projection dispatches per attention block while preserving exact math.

Risk: medium. The math is exact if kernels are concatenated in the same order, but the Flax variable tree needs careful handling because existing checkpoints store separate `Dense_0`, `Dense_1`, and `Dense_2` kernels. The cleanest export path is to add an export-only module and transform/copy variables during export, not to change training modules.

Concrete pointers:

- `visionary/export/onnx_wrappers.py`
  - `_ExportAttention.__call__`
  - `_CachedTemporalAttention.__call__`
  - `_CachedTemporalStepAttention.__call__`
  - `_ExportTransformerBlock` and `_Cached*TransformerBlock` call sites
- `scripts/webgpu/export_dreamer4_onnx.py`
  - add a variable-preparation step near the cached export functions and manifest rewrite metadata
  - validate active artifacts around `DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME`
- Reference source modules for naming/behavior:
  - `visionary/transformer.py::Attention`
  - `visionary/dynamics.py::_CachedTemporalAttention`

Validation:

- Compare raw/fused-QKV ONNX outputs for `final_z`, `candidate_k_entry`, and `candidate_v_entry`.
- Confirm graph counts reduce `Gemm`/projection `Einsum` without reintroducing `Reshape`.
- Run `bun run benchmark:webgpu` and graph-capture benchmark separately.

## Candidate 2: Export-Native Fused Dynamics Attention

Lower dynamics attention to `com.microsoft::GroupQueryAttention` or an equivalent custom export primitive for both spatial and cached temporal attention, avoiding the current `Einsum -> Softmax -> Einsum` islands and GQA `Gather` plumbing.

Expected impact: high. The active graph still has `119 Softmax`, `716 Einsum`, and `239 Gather` nodes. Fusing attention is the main remaining way to remove whole attention islands rather than shaving layout nodes.

Risk: high. The current worktree already has a partial post-export `rewrite_cached_temporal_attention_to_gqa()` experiment that looks for no-mask full-cache attention islands and can cover both temporal and spatial-shaped cases, but it still inserts `Reshape` nodes around the GQA op and is gated behind `--fused_temporal_gqa`. The next version should be export-native or should emit rank-compatible flat tensors directly, otherwise it risks undoing the solved WebGPU placement work.

Concrete pointers:

- `visionary/export/onnx_wrappers.py`
  - `_attention_for_export`
  - `_export_dot_product_attention`
  - `_CachedTemporalStepAttention.__call__`
  - `_ExportAttention.__call__`
- `scripts/webgpu/export_dreamer4_onnx.py`
  - `--fused_temporal_gqa`
  - `rewrite_cached_temporal_attention_to_gqa`
  - manifest field `cached_temporal_gqa_fusion`
- Benchmark target:
  - `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4`

Validation:

- First prove one temporal attention replacement with no `Reshape`, no CPU fallback, and parity.
- Then extend to spatial attention.
- Browser check must confirm `GroupQueryAttention` runs on WebGPU and graph capture still succeeds.

## Candidate 3: Export Fused MLP Gate/Value Projection

Replace `SwiGLU`'s two input projections with one export-only packed projection, then split into gate/value before `swish(gate) * value`. Keep the output projection unchanged.

Expected impact: medium. There are `119 QuickGelu`/MLP blocks and many neighboring `Gemm`, `Unsqueeze`, `Mul`, and `Add` nodes. Fusing gate/value removes one large projection dispatch per block and may simplify surrounding shape adapters.

Risk: medium. This is exact if the two input kernels are packed without changing parameter values. As with QKV fusion, the main risk is variable naming and checkpoint compatibility in the export wrapper.

Concrete pointers:

- `visionary/transformer.py::SwiGLU` for behavior.
- `visionary/export/onnx_wrappers.py`
  - `_ExportTransformerBlock.__call__`
  - `_CachedPrefillTransformerBlock.__call__`
  - `_CachedStepTransformerBlock.__call__`
  - replace export-path `SwiGLU(...)` calls with an export-only packed SwiGLU.
- `scripts/webgpu/export_dreamer4_onnx.py`
  - add/check parameter packing for the export-only module.

Validation:

- CPU ORT parity on the active entry artifact.
- Check that packed MLP reduces projection op count and does not create unsupported layout ops.
- Benchmark after Candidate 1 independently if possible, because QKV and MLP fusion target different dispatch groups.

## Recommended Next Candidate

Implement Candidate 1 first: export fused QKV projections.

Reasoning:

- It is exact and narrower than fused attention.
- It targets a large repeated dispatch class in the current graph without relying on ORT contrib attention behavior.
- It is more likely than cache ABI or graph capture work to move the measured `75-90 ms` dynamics session time.
- It sets up Candidate 2 because fused attention prefers flat packed Q/K/V inputs anyway.

Suggested first implementation path:

1. Add an export-only packed attention projection in `visionary/export/onnx_wrappers.py` for `_ExportAttention`, `_CachedTemporalAttention`, and `_CachedTemporalStepAttention`.
2. Add export variable packing in `scripts/webgpu/export_dreamer4_onnx.py` for the cached dynamics artifacts, initially gated by a flag.
3. Validate only `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4` first, then expand to the other cached variants once parity and graph counts look good.
