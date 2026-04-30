# ONNX Graph Rewrite Path for WebGPU Reshape CPU Islands

Date: 2026-04-29

## Executive summary

The current exported static ONNX artifacts already have constant `Reshape` target inputs, and ORT offline optimization is folding a large amount of shape plumbing. The remaining issue is not dynamic shape construction. It is that `onnxruntime-web` WebGPU has no standalone GPU kernel for graph-level `Reshape`, so every remaining `Reshape` becomes a CPU node and creates WebGPU-to-CPU / CPU-to-WebGPU boundaries.

The highest-return route is a post-export ONNX pass in `scripts/webgpu/export_dreamer4_onnx.py`, after the existing ORT optimization step:

1. Rewrite flattened `Gemm` islands into rank-preserving `MatMul` plus optional `Add`, using reshaped constant weights where needed. This removes the dominant flatten/restore and projection-head reshapes.
2. Rewrite GQA head-repeat `Reshape -> Expand -> Reshape` into a single `Gather` on the KV-head axis with constant repeated indices.
3. Add conservative cleanup for direct/adjacent reshapes and elementwise-through-reshape cases that remain after the first two passes.
4. Validate each rewritten artifact against the pre-rewrite artifact using CPU ORT before replacing the file.

On the profiled graph, `breakout_dynamics_step_cached_b1_t1.onnx`, this targets the large majority of the 399 CPU `Reshape` nodes: roughly 148-172 from Gemm rank lifting and another 96 from GQA repeat lowering, before smaller cleanups.

## Evidence inspected

Local files and artifacts inspected:

- `scripts/webgpu/export_dreamer4_onnx.py`
- `visionary/export/onnx_wrappers.py`
- `webgpu_app/assets/*.onnx`
- `webgpu_app/assets/breakout_onnx_manifest.json`
- `webgpu_app/bench/results/session_profile_summary.json`
- `node_modules/onnxruntime-web/docs/webgpu-operators.md`
- `node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/op-resolve-rules.ts`
- `node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/ops/{matmul,gemm,expand,concat,transpose}.ts`
- `node_modules/onnxruntime-web/lib/wasm/jsep/init.ts`
- `.venv/lib/python3.11/site-packages/jax2onnx/plugins/jax/lax/{reshape,dot_general,broadcast_in_dim,transpose,concatenate}.py`
- `.venv/lib/python3.11/site-packages/jax2onnx/converter/{conversion_api,ir_optimizations}.py`

## Current state

The exporter already runs ORT `ORT_ENABLE_EXTENDED` unless `--skip_onnx_optimization` is set. The manifest shows that this removes many static shape nodes, but leaves many `Reshape` nodes:

| artifact | nodes before -> after ORT | Reshape before -> after ORT |
| --- | ---: | ---: |
| `breakout_tokenizer_decoder_b1_t64.onnx` | 811 -> 437 | 238 -> 124 |
| `breakout_dynamics_b1_t64.onnx` | 2255 -> 1306 | 683 -> 367 |
| `breakout_tokenizer_decoder_b1_t1.onnx` | 811 -> 437 | 238 -> 124 |
| `breakout_tokenizer_decode_z_b1_t1.onnx` | 813 -> 437 | 240 -> 124 |
| `breakout_dynamics_prefill_cached_b1_t64.onnx` | 2850 -> 1802 | 694 -> 384 |
| `breakout_dynamics_step_cached_b1_t1.onnx` | 2987 -> 1902 | 708 -> 399 |
| `breakout_dynamics_cached_sample_step_b1_t1_s4.onnx` | 11743 -> 7290 | 2796 -> 1511 |

All remaining `Reshape` target inputs in these artifacts are constant.

The session profile summary is for the cached step graph. It reports:

- 2680 total node events.
- 2275 WebGPU node events, 405 CPU node events.
- 399 CPU `Reshape` events, totaling 6.956 ms.
- 796 memcpy node events.
- `MemcpyToHost`: 394 events, 278.078 ms.
- `MemcpyFromHost`: 402 events, 44.020 ms.

So the main cost is not the CPU reshape arithmetic itself. It is the provider boundary traffic created around CPU reshape islands.

## Source-level cause

`onnxruntime-web` documents `Reshape` for WebGPU as having no GPU kernel. The local resolver maps `Gemm`, `MatMul`, `Einsum`, `Concat`, `Expand`, `Transpose`, `Squeeze`, `Unsqueeze`, and `Gather`, but not a WebGPU `Reshape` implementation.

The WebGPU `TensorView.reshape()` method in `node_modules/onnxruntime-web/lib/wasm/jsep/init.ts` only returns a new view over the same data pointer when used inside an existing WebGPU op implementation. That is exactly what we want: move reshape semantics into supported GPU ops or initializer metadata, not graph-level `Reshape` nodes.

JAX2ONNX 0.13.0 already emits a single constant initializer for fully static reshape shapes, and its IR optimizer already removes identity reshapes and some redundant reshape pairs. That confirms this repo is past the "constant shape target" problem; the remaining opportunity is semantic graph rewriting.

## Artifact pattern counts

The dominant remaining reshape patterns are stable across all exports:

| artifact | Reshape count | Gemm rank-lift candidates, rough reshape removal | GQA repeat candidates, reshape removal |
| --- | ---: | ---: | ---: |
| `breakout_dynamics_b1_t64.onnx` | 367 | 148-172 | 96 |
| `breakout_dynamics_prefill_cached_b1_t64.onnx` | 384 | 148-172 | 96 |
| `breakout_dynamics_step_cached_b1_t1.onnx` | 399 | 148-172 | 96 |
| `breakout_dynamics_cached_sample_step_b1_t1_s4.onnx` | 1511 | 592-688 | 384 |
| `breakout_tokenizer_decoder_b1_t64.onnx` | 124 | 52-60 | 32 |
| `breakout_tokenizer_decoder_b1_t1.onnx` | 124 | 52-60 | 32 |
| `breakout_tokenizer_decode_z_b1_t1.onnx` | 124 | 50-59 | 32 |

For `breakout_dynamics_step_cached_b1_t1.onnx`, the common transitions include:

- `(1, 36, 256) -> (36, 256)` before `Gemm`.
- `(36, 256) -> (1, 36, 256)` after `Gemm`.
- `(36, 128) -> (1, 36, 2, 64)` after Q/K projection.
- `(36, 512) -> (1, 36, 8, 64)` after V or output-head projection.
- `(1, 36, 2, 64) -> (1, 36, 2, 1, 64) -> Expand -> (1, 36, 8, 64)` for repeated KV heads.
- `(1, 36, 8, 64) -> (36, 512)` before output projection.

## Priority 1: rank-lift Gemm islands

### Pattern A: flatten -> Gemm -> restore

Current:

```text
x[P..., K]
  Reshape -> x2[prod(P), K]
  Gemm(x2, W[K, N], C)
  Reshape -> y[P..., N]
```

Rewrite:

```text
MatMul(x[P..., K], W[K, N]) -> y0[P..., N]
optional Add(y0, C) -> y[P..., N]
```

Eligibility:

- `Reshape` data input shape is `P + [K]`.
- `Reshape` output shape is `[prod(P), K]`.
- `Gemm` has `transA=0`, `transB=0`, `alpha=1`.
- `beta=0` means ignore `C`; `beta=1` means add `C` if nonzero.
- `C` is scalar, `[N]`, or otherwise ONNX-broadcastable to `P + [N]`.
- Downstream `Reshape` target is `P + [N]`, or downstream ops are rank-agnostic elementwise ops and can consume `P + [N]`.

This removes the CPU flatten reshape and often makes the restore reshape an identity that can be deleted.

### Pattern B: flatten -> Gemm -> split projection heads

Current:

```text
x[P..., K]
  Reshape -> x2[prod(P), K]
  Gemm(x2, W[K, H * D], C)
  Reshape -> y[P..., H, D]
```

Rewrite:

```text
W_head = initializer reshape W to [H, K, D]
x_mm = Unsqueeze(x, axis=-2)              # P..., 1, K
y_mm = MatMul(x_mm, W_head)              # P..., H, 1, D
y = Squeeze(y_mm, axis=-2)               # P..., H, D
optional Add with C reshaped/broadcast as [H, D]
```

This uses only WebGPU-supported graph ops plus an offline initializer reshape. It removes both standalone graph-level reshapes around Q/K/V-style projections.

Eligibility:

- Same `Gemm` constraints as Pattern A.
- The sole or primary consumer is `Reshape` to `P + tail`.
- `prod(tail) == N`, where `N` is the Gemm output width.
- `tail` length is at least 2 for the projection-head form.
- The first `len(P)` target dims match the pre-Gemm unflattened dims.

Implementation note: for general `tail`, reshape the weight initializer to `tail[:-1] + [K, tail[-1]]`. Use `Unsqueeze(x, axis=-2)`, `MatMul`, then `Squeeze(axis=-2)`. ONNX MatMul broadcasts batch dimensions, and ORT WebGPU supports rank-N MatMul.

### Why this should be first

This pass attacks the largest source of CPU reshape islands. In the profiled step graph, there are 170 `Gemm` nodes and 399 `Reshape` nodes. The graph has 74 flatten reshapes feeding Gemm and 98 Gemm-output restore/projection reshapes. The rewrite can remove most of these without changing math.

## Priority 2: replace GQA repeat-head reshape chains with Gather

Current GQA repeat pattern:

```text
k_or_v[P..., KV, D]
  Reshape -> [P..., KV, 1, D]
  Expand  -> [P..., KV, repeat, D]
  Reshape -> [P..., KV * repeat, D]
```

For this repo, typical shapes are:

```text
[1, 36, 2, 64] -> [1, 36, 2, 1, 64] -> Expand -> [1, 36, 8, 64]
```

Rewrite:

```text
indices = [0, 0, 0, 0, 1, 1, 1, 1]  # int64 initializer when KV=2, repeat=4
Gather(k_or_v, indices, axis=-2) -> [P..., KV * repeat, D]
```

Eligibility:

- `Reshape1` only inserts a singleton axis immediately after the KV-head axis, possibly while preserving the same leading dims.
- `Expand` expands that singleton axis to `repeat`.
- `Reshape2` merges `[KV, repeat]` into `KV * repeat`.
- No consumer depends on the intermediate `[KV, repeat]` rank.

This preserves `jnp.repeat(..., axis=-2)` ordering. Do not replace this with `Tile`; ONNX `Tile` would repeat the whole KV-head axis in a different order.

Expected removal:

- Dynamics step/prefill/uncached: 48 chains, 96 reshapes removed per artifact.
- Cached sample-step: 192 chains, 384 reshapes removed.
- Tokenizer decoders: 16 chains, 32 reshapes removed.

## Priority 3: compose adjacent reshape consumers

There are a few direct `Reshape -> Reshape` edges in cached graphs, especially around cache write/stack shapes. Many are shared-output cases, so the existing JAX2ONNX pair eliminator does not remove them.

Conservative rule:

- If `R1` has a single data consumer `R2`, replace `R1 -> R2` with one `Reshape` from `R1` input to `R2` target.
- If `R1` has multiple consumers, only rewrite a branch when it reduces total CPU reshapes. Replacing `R2` with a composed reshape while keeping `R1` does not reduce count, so skip unless a later pass removes `R1`.

This is lower priority because current counts are small compared with Gemm and GQA patterns.

## Priority 4: propagate rank through rank-agnostic ops

After Gemm rank lifting, additional reshapes become redundant if rank-agnostic ops are allowed to carry the unflattened rank:

- `Add`
- `Mul`
- `Sub`
- `Div`
- `QuickGelu`
- `Sigmoid`
- `Sqrt`
- `Cast`
- `Min`/`Max` only when broadcasting is unchanged

Rule:

```text
Reshape(flatten) -> elementwise -> ... -> Gemm
```

can often become:

```text
elementwise on rank-N -> ... -> MatMul
```

This should be implemented after the direct Gemm rewrite, because many of these opportunities become identity reshapes only after MatMul outputs have rank-N shapes.

## Lower-priority alternatives

### Generate shape-specialized exports

A more invasive export route is to change the JAX wrapper/exported computation so Dense-like projections operate rank-natively before JAX2ONNX sees them. That means avoiding the JAX/JAX2ONNX `reshape -> dot -> reshape` lowering in the first place, probably by using explicit rank-N matmul/einsum wrappers for export-only Dense paths.

This can work, but the post-export ONNX pass is less invasive and directly targets the already-exported artifacts.

### Use grouped GQA export or fused attention ops

The repo already has `--grouped_gqa_attention`. That may reduce materialized K/V repeat patterns at export time, but it should be benchmarked separately. A larger rewrite to `GroupQueryAttention` or `MultiHeadAttention` could remove many Q/K/V reshape and attention nodes, but mask/cache semantics make it a riskier route than local Gemm and GQA-repeat rewrites.

### Implement WebGPU Reshape in ORT Web

This is outside this repo. It would still require waiting for ORT Web updates and would not reduce graph fragmentation in current deployed artifacts.

## Proposed integration point

Add a new post-export pass in `scripts/webgpu/export_dreamer4_onnx.py`:

```text
jax2onnx.to_onnx(...)
onnx.checker.check_model(...)
optional onnxsim
existing ORT_ENABLE_EXTENDED optimization
new optimize_static_reshapes_for_webgpu(path)
optional float16
validation
manifest metadata
```

The pass should:

1. Load the model with external data when present.
2. Build producer/consumer maps, shape maps, initializer maps, and constant target-shape maps.
3. Run `rewrite_gemm_reshape_islands`.
4. Run `rewrite_gqa_repeat_reshape_expand`.
5. Run `remove_identity_reshapes` and `remove_single_consumer_adjacent_reshapes`.
6. Run dead-node and unused-initializer cleanup.
7. Run ONNX checker and shape inference if it succeeds.
8. Validate old vs new with ORT CPU on deterministic seeded inputs before replacing the artifact.

Manifest metadata should record:

- `reshape_webgpu_rewrite.enabled`
- node count before/after
- reshape count before/after
- counts by rewrite kind
- validation result

## Validation requirements

Use old artifact vs rewritten artifact, not JAX vs rewritten artifact only:

- Load old and new with CPU `onnxruntime`.
- Feed the same deterministic inputs already used by exporter validation.
- Compare all graph outputs with the existing `atol`/`rtol`.
- Run this per artifact before replacement.

For `--float16`, run the reshape rewrite before float16 conversion. That keeps initializer reshaping and numerical comparison simpler.

## Expected impact

For `breakout_dynamics_step_cached_b1_t1.onnx`, the profile has:

- 399 CPU `Reshape` nodes.
- 394 `MemcpyToHost` and 402 `MemcpyFromHost` events.
- 278.078 ms in `MemcpyToHost` and 44.020 ms in `MemcpyFromHost`.

If the first two rewrite classes remove roughly 240-268 of the 399 reshape nodes, the profile should show a much smaller CPU island count and, more importantly, fewer provider-boundary memcpy nodes. The direct CPU `Reshape` time is only about 7 ms; the real benchmark win should come from reducing host/device copies.

## Concrete implementation order

1. Implement utility functions: `get_initializer_array`, `replace_initializer_array`, `value_shape`, `const_shape_input`, `replace_all_uses_except`, `remove_dead_nodes`.
2. Implement Pattern A Gemm rank lifting and validate on tokenizer decoder first.
3. Extend to Pattern B projection-head weights and validate on `breakout_dynamics_step_cached_b1_t1.onnx`.
4. Implement GQA `Gather` rewrite.
5. Add manifest stats and run the existing profile diagnostic.
6. Only then consider rank-through-elementwise and adjacent shared-reshape cleanup.

The first benchmark target should be `breakout_dynamics_step_cached_b1_t1.onnx`, because `session_profile_summary.json` confirms it is the profiled graph and has a clean 399-reshape baseline.
