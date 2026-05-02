# ORT WebGPU Operator Bottleneck Audit

Scope: read-only inspection of `onnxruntime-web@1.24.3` in `node_modules`, `webgpu_app/assets/breakout_onnx_manifest.json`, `webgpu_app/bench/results/latest.json`, and the generated ONNX assets. No source files were edited.

## Current benchmark and graph

Latest browser benchmark:

- Config: WebGPU, `graphCapture=true`, `timedRuns=64`, profiling disabled.
- Active step artifact: `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4`.
- Cached step / dynamics median: `84.345 ms`.
- Decoder median: `6.375 ms`.
- Streaming frame median: `91.922 ms`.
- Cache commit median: `0 ms`.

The active dynamics artifact is now compute/dispatch bound, not obviously CPU-fallback bound. Direct ONNX parse of `webgpu_app/assets/breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4.onnx`:

| Op | Count |
| --- | ---: |
| Unsqueeze | 915 |
| Einsum | 716 |
| Transpose | 537 |
| Squeeze | 451 |
| Gemm | 366 |
| SimplifiedLayerNormalization | 299 |
| Mul | 244 |
| Gather | 239 |
| RotaryEmbedding | 239 |
| SkipSimplifiedLayerNormalization | 179 |
| Softmax | 119 |
| QuickGelu | 119 |
| Add | 65 |
| Concat | 65 |
| Slice | 16 |
| Sub | 4 |
| Div | 3 |

Notably absent from the final hot graph: `Reshape`, `Shape`, `Min`, `Cast`, `Split`, and `GroupQueryAttention`.

## ORT WebGPU support evidence

Local support table:

- `Einsum` is listed as WebGPU-supported.
- `RotaryEmbedding` is listed as WebGPU-supported.
- `Squeeze` and `Unsqueeze` are listed as WebGPU-supported.
- `Transpose` is WebGPU-supported, but marked `need perf optimization`.
- `Reshape` and `Shape` are listed with `no GPU kernel`.
- `GroupQueryAttention` is listed as WebGPU-supported.

Local source:

- `op-resolve-rules.ts` registers JS WebGPU implementations for `Einsum`, `Gather`, `Gemm`, `GroupQueryAttention`, `RotaryEmbedding`, `Softmax`, `Transpose`, etc.
- There is no `ops/reshape.ts`, and `Reshape` is not in the JS WebGPU resolve map. This matches the support table warning.
- `Transpose` always materializes a new output buffer through `TransposeCopy`, `TransposeShared`, or generic `Transpose`; even reshape-like transposes are copies, not view aliases.
- `Einsum` uses a generic generated shader with scalar reduction loops, not a packed/matmul-specialized path.
- `RotaryEmbedding` is a single WebGPU dispatch per op and replaces a much larger arithmetic decomposition.
- `GroupQueryAttention` internally computes QK, softmax, and V-score as three WebGPU programs and supports KV-head repetition through `nReps`, avoiding explicit repeated-head materialization.

## Operator conclusions

### Transpose

Do not treat `Transpose` as a CPU fallback in the current artifact, but do treat it as expensive. The active graph has `537` `Transpose` nodes, all with perm `(0, 2, 1, 3)`. ORT's own WebGPU table marks it as needing perf optimization, and source shows it is a dispatching copy kernel.

Recommendation: reduce layout churn instead of preserving these transposes. The biggest target is attention layout: keep tensors in the layout expected by the next attention/projection kernel, or move to fused attention so Q/K/V transposes are inside one attention lowering rather than graph-level nodes.

### RotaryEmbedding

Keep `RotaryEmbedding`. It is WebGPU-supported and the manifest shows the rewrite is doing useful work: the active graph has `239 RotaryEmbedding` nodes with much lower surrounding `Add`/`Mul`/`Concat`/`Split` counts than the decomposed form. It is still one dispatch per Q/K rotary site, so it contributes to dispatch count, but reverting it would be worse.

Recommendation: keep the contrib op. The next improvement is not to decompose it, but to fold rotary into a fused attention path when using `GroupQueryAttention`/custom attention.

### Einsum

`Einsum` is the largest remaining compute bottleneck. It is WebGPU-supported, but source confirms it is generic scalar-loop code. The active graph has `716` `Einsum` nodes:

- `359` projection/head rewrite einsums: `nk,khd->nhd`
- `119` attention score einsums: `bqhd,bkhd->bhqk`
- `119` attention value einsums: `bhqk,bkhd->bqhd`
- `90` output projection einsums: `bnhd,hdm->nm`
- `29` output projection einsums: `nthd,hdm->nm`

Recommendation: do not add more `Einsum` as a generic replacement unless it removes a CPU fallback such as `Reshape`. That trade already paid off. From here, reduce `Einsum` count by fusing attention and projections, not by replacing one layout op with another generic einsum.

### Squeeze / Unsqueeze

Keep `Squeeze`/`Unsqueeze` as replacements for `Reshape` when the alternative is a CPU/provider boundary. The active graph-capture run succeeds and the hot graph has no `Reshape`, so these ops are not the current CPU fallback.

They are still too numerous: `915 Unsqueeze` plus `451 Squeeze`. Even if some are cheap metadata handling, they enlarge graph-capture replay and binding work; if they are true kernels in a given ORT path, they are a large dispatch class.

Recommendation: keep existing `Reshape -> Squeeze/Unsqueeze` rewrites, but do not stop there. Collapse adjacent singleton adapters around producer/consumer rewrites and prefer export-native rank/layout choices that avoid emitting them in the first place.

## CPU fallback risk

For the current active `slide_entry` artifact, CPU fallback is likely not first-order:

- Direct ONNX parse shows no `Reshape`, `Shape`, or `Min` in the active graph.
- `latest.json` shows graph capture enabled for the cached step and stable steady-state dynamics samples around `84-86 ms` after initial capture/warmup.
- The older known CPU fallback class, `Reshape`, is eliminated by the `head_projection_rewrite`; manifest records `478 -> 0 Reshape` for that pass.

Remaining CPU fallback risks to watch:

- Any new rewrite that reintroduces `Reshape` or `Shape`.
- Any scalar/index logic that emits unsupported ops such as `Min`.
- Any attempt to use unsupported dtypes for WebGPU kernels. Existing notes already showed BF16 breaks `Einsum`.

## Dispatch reduction recommendations

1. Prioritize fused attention over more shape rewrites.

   The graph has `119` attention islands, each still expressed as `Einsum(QK) -> Softmax -> Einsum(V)` plus surrounding `Transpose`, `Gather`, `Squeeze`, and `Unsqueeze`. ORT WebGPU has `GroupQueryAttention`, and its source handles KV repetition internally via `nReps`. A correct GQA/custom-attention rewrite can replace a repeated island with fewer kernels and remove explicit head-repeat/layout plumbing.

2. Make the GQA rewrite export-native or layout-native.

   A post-export GQA rewrite that inserts graph-level `Reshape` will likely undo the solved CPU-fallback work. The target ABI should feed flat `[B, S, hidden]` Q/K/V or already-compatible BNSH cache tensors directly to the fused op, with no graph-level `Reshape`.

3. Reduce projection dispatch count next.

   The current `Reshape`-avoidance rewrite converted many `Gemm + Reshape` patterns into `Einsum`, leaving `716` einsums. Pack QKV projections where possible, and pack MLP gate/value projections where exact. This reduces dispatch count without relying on generic `Einsum` for every head split/merge.

4. Keep `RotaryEmbedding`, but aim to absorb it into fused attention later.

   The RotaryEmbedding rewrite is worthwhile as a standalone improvement. Its remaining cost is dispatch count, not provider fallback.

5. Treat `Transpose`, `Squeeze`, and `Unsqueeze` as layout debt.

   They are worth keeping versus `Reshape` CPU fallback, but they should not be considered free. The current `1,903` combined layout/view nodes (`Transpose + Squeeze + Unsqueeze`) are too many for a 50 ms target.

## Practical next experiment

Build one isolated `GroupQueryAttention` prototype for a steady-state full-cache temporal attention island in the active `slide_entry` graph:

- Inputs must avoid graph-level `Reshape`.
- Keep cache in GQA-native layout, or change exporter/cache ABI so no transpose ladder is needed.
- Validate parity against the current artifact.
- Browser-profile with session profiling enabled to confirm `GroupQueryAttention` is on WebGPU and measure kernel count reduction.
- Compare against the current graph-capture baseline: dynamics median `84.345 ms`, streaming median `91.922 ms`.

Expected result if successful: fewer `Einsum`, `Softmax`, `Transpose`, `Gather`, `Squeeze`, and `Unsqueeze` nodes. This is the only path visible from local ORT WebGPU source that can remove whole attention dispatch groups rather than shaving individual layout kernels.
