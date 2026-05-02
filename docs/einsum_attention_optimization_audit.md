# Einsum Attention Optimization Audit

Date: 2026-05-03

Scope: read-only inspection of:

- `visionary/export/onnx_wrappers.py`
- `scripts/webgpu/export_dreamer4_onnx.py`
- `webgpu_app/assets/breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4.onnx`
- `webgpu_app/assets/breakout_onnx_manifest.json`
- local `onnxruntime-web@1.24.3` WebGPU operator files under `node_modules/onnxruntime-web`

No source files were edited.

## Current Hot Graph

The active graph is numerically correct and has no `Reshape`, `Cast`, `Expand`, `Split`, `ReduceMean`, or `Sqrt` left. The current top operator counts are:

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

`Einsum` breaks down as:

| Equation | Count | Meaning |
| --- | ---: | --- |
| `nk,khd->nhd` | 359 | Q/K/V projection output split into heads |
| `bqhd,bkhd->bhqk` | 119 | attention score matmul |
| `bhqk,bkhd->bqhd` | 119 | attention value matmul |
| `bnhd,hdm->nm` | 90 | spatial attention output head merge |
| `nthd,hdm->nm` | 29 | temporal attention output head merge |

The important regression clue is in the manifest rewrite stats:

- Before `custom_head_projection_reshape_rewrite`: `Einsum=238`, `Gemm=844`, `Reshape=478`, `Unsqueeze=556`.
- After it: `Einsum=716`, `Gemm=366`, `Reshape=0`, `Unsqueeze=915`.
- This pass replaced 359 projection `Gemm -> Reshape` patterns and 119 output merge `Reshape -> Gemm` patterns with rank-aware `Einsum`.

So the current slowdown is probably not from the RotaryEmbedding rewrite alone. It is more likely from replacing 478 optimized dense/projection matmuls with generic `Einsum` kernels to avoid `Reshape`.

## Relevant Code Paths

`visionary/export/onnx_wrappers.py`:

- `_export_dot_product_attention()` directly exports attention as `jnp.einsum("bqhd,bkhd->bhqk")`, softmax, then `jnp.einsum("bhqk,bkhd->bqhd")`.
- `_ExportAttention`, `_CachedTemporalAttention`, and `_CachedTemporalStepAttention` all project Q/K/V with separate `Dense`, reshape to `b t h d`, apply RMSNorm/RoPE on Q/K, then call `_attention_for_export()`.
- The GQA path repeats K/V with `jnp.repeat()` before the two attention einsums.

`scripts/webgpu/export_dreamer4_onnx.py`:

- `rewrite_singleton_reshapes_for_webgpu()` replaces singleton-only `Reshape` with `Squeeze`/`Unsqueeze`.
- `rewrite_gqa_repeats_for_webgpu()` replaces `Expand -> Reshape` repeat materialization with `Gather` over KV heads.
- `rewrite_head_projection_reshapes_for_webgpu()` is the pass that converts projection split/merge `Gemm + Reshape` patterns into `Einsum`.
- `rewrite_rotary_embedding_for_webgpu()` fuses RoPE arithmetic into `com.microsoft::RotaryEmbedding`, but wraps almost every fused op in BSHD<->BHSD transposes because ORT WebGPU expects rank-4 rotary input as `[batch, heads, sequence, head_dim]`.

Local ORT WebGPU support confirms `MatMul`, `Gemm`, `Einsum`, `Squeeze`, `Unsqueeze`, `Transpose`, `Gather`, `Split`, `Concat`, `Softmax`, and `RotaryEmbedding` are registered. `Transpose` is marked "need perf optimization" in the local generated support table. `MatMul` uses the packed WebGPU matmul path for normal sizes, while `Einsum` is implemented as a generic equation shader.

## Candidate 1: Replace Projection Split/Merge Einsums With MatMul/Gemm Plus Layout Ops

Priority: high.

Current problem:

- 478 of 716 `Einsum` nodes were introduced by `rewrite_head_projection_reshapes_for_webgpu()`.
- Projection split examples are `Squeeze([1,36,256] -> [36,256]) -> Einsum(nk,khd->nhd) -> Unsqueeze([36,H,64] -> [1,36,H,64])`.
- Output merge examples are `Einsum([1,36,8,64], [8,64,256] -> [36,256]) -> Unsqueeze([1,36,256])`.

Concrete replacement shapes:

- Q projection: `[1,36,256] @ [256,512] -> [1,36,512]`, then split last axis into eight 64-wide chunks and concatenate those chunks along a new head axis to produce `[1,36,8,64]`.
- K/V projection: `[1,36,256] @ [256,128] -> [1,36,128]`, split into two 64-wide chunks to produce `[1,36,2,64]`.
- Temporal variants use `[36,1,256] @ [256,H*64] -> [36,1,H*64]`.
- Output merge can do the inverse: concatenate head slices into `[1,36,512]` or `[36,1,512]`, then `MatMul/Gemm` with `[512,256]`.

Likely implementation:

- Replace `rewrite_head_projection_reshapes_for_webgpu()` with a version that keeps dense projections as `Gemm` where rank-2 is already present, or emits `MatMul` on rank-3 inputs to avoid the surrounding `Squeeze`.
- Convert `[*, H*D] <-> [*, H, D]` using only static `Split`, `Unsqueeze`, `Squeeze`, and `Concat`, similar to the specialized static cache rewrite already used for `[36,128] -> [1,36,1,2,64]`.
- Gate this behind a flag and compare against the current einsum rewrite because it trades fewer generic matmuls for more simple layout dispatches.

Expected graph effect:

- Remove up to 359 `nk,khd->nhd` einsums and 119 head-merge einsums.
- Increase `MatMul`/`Gemm`, `Split`, and `Concat`.
- Potentially reduce many `Squeeze`/`Unsqueeze` if rank-3 `MatMul` is used directly from `[1,36,256]` / `[36,1,256]`.

Risks:

- If implemented with many per-head `Split`/`Concat` nodes, the dispatch count can rise. For H=8, replacing one einsum may add one split, eight unsqueezes, and one concat unless optimized carefully.
- `Split`/`Concat` must preserve ONNX row-major flattening exactly. A wrong axis or chunk order silently permutes heads.
- Rank-3 `MatMul` must benchmark better than rank-2 `Gemm + layout ops`; ORT WebGPU has optimized paths for both, but the exact shape matters.
- Bias is currently absent (`use_bias=False`), so this is simpler. If bias appears later, `MatMul + Add` or `Gemm` handling must be added.

## Candidate 2: Lower True Attention Einsums To Transpose + MatMul

Priority: high.

Current attention core:

- 90 spatial score/value pairs: `[1,36,8,64] x [1,36,8,64] -> [1,8,36,36]`, then `[1,8,36,36] x [1,36,8,64] -> [1,36,8,64]`.
- 29 temporal score/value pairs: `[36,1,8,64] x [36,65,8,64] -> [36,8,1,65]`, then `[36,8,1,65] x [36,65,8,64] -> [36,1,8,64]`.

Concrete lowering:

- Score:
  - `q_bshd -> q_bhsd` via `Transpose([0,2,1,3])`.
  - `k_bshd -> k_bhds` via `Transpose([0,2,3,1])`.
  - `MatMul(q_bhsd, k_bhds) -> [B,H,Q,K]`.
- Value:
  - `v_bshd -> v_bhsd` via `Transpose([0,2,1,3])`.
  - `MatMul(weights_bhqk, v_bhkd) -> [B,H,Q,D]`.
  - `Transpose([0,2,1,3]) -> [B,Q,H,D]`.

This directly replaces 238 true attention `Einsum` nodes with `MatMul` and transposes. It should be tested because ORT WebGPU's `MatMul` uses packed kernels while `Einsum` is generic.

The best implementation is not a blind post-export replacement. It should coordinate with RoPE layout:

- `RotaryEmbedding` already wants BHSD, and the current rewrite adds BSHD<->BHSD transposes around each RoPE node.
- If attention internals stay BHSD after RoPE, the post-RoPE `rotary_from_bhsd` transposes can be removed and the score `MatMul` can consume Q/K in the layout it wants.
- V can be projected/split directly to BHSD if Candidate 1 is also implemented in BHSD.

Risks:

- This can increase `Transpose` count if implemented locally around each einsum. It only becomes compelling if paired with a broader attention-internal BHSD layout.
- Mask shape must stay broadcast-compatible with `[B,H,Q,K]`. Existing masks are already effectively `[B,1,Q,K]`, but each spatial/temporal path should be validated.
- Softmax axis remains the last axis; any layout drift here changes behavior.
- Temporal attention has `B=36`, `Q=1`, `K=65`; the local `MatMul` implementation has a batched vec-mat optimization for `M == 1`, which is promising but must be benchmarked.

## Candidate 3: Canonicalize Export Attention Layout To BHSD

Priority: high if Candidate 2 is pursued.

Current export uses BSHD throughout JAX wrappers because the model code naturally works as `b t h d`. ORT WebGPU's `RotaryEmbedding` expects BHSD and current graph surgery wraps each fused RoPE island with transposes. The graph has 537 `Transpose` nodes, all with perm `[0,2,1,3]`.

Concrete approach:

- In export-only wrappers, after Q/K/V projection and head split, immediately transpose or directly build Q/K/V as `[B,H,S,D]`.
- Apply Q/K RMSNorm in a way that preserves last-axis normalization.
- Feed Q/K directly to `RotaryEmbedding` without the wrapper transposes.
- Do score/value attention as BHSD/BHDK `MatMul`.
- Only transpose back to BSHD if the downstream output merge path needs it. If Candidate 1's output merge accepts BHSD directly, this final transpose can also be avoided.

Expected graph effect:

- Remove many of the 537 rotary transposes.
- Make Candidate 2 cheaper by avoiding extra layout conversions around every attention block.
- Simplify GQA repeat/gather axes by making head axis consistently `1` instead of `2`.

Risks:

- This is more invasive than a post-export rewrite because it touches wrapper-level export semantics.
- `_CachedTemporalAttention` returns `k, v` for caches. Cache ABI currently stores `[batch, token, time, kv_heads, head_dim]`; a BHSD internal layout must still return cache entries in the browser contract's existing layout or update the cache ABI everywhere.
- RMSNorm axes and output merge weights are safe only if the last dimension remains `head_dim`; validating shape metadata is mandatory.

## Candidate 4: Avoid Materialized GQA Head Repeats

Priority: medium.

Current graph has 239 `Gather` nodes from `rewrite_gqa_repeats_for_webgpu()`. This is better than the previous `Expand -> Reshape` repeat, but it still materializes K/V as 8 heads before attention.

Concrete alternatives:

- Grouped MatMul attention without K/V repeat:
  - Reshape/reorder Q from 8 heads to `(kv_head=2, repeat=4)`.
  - MatMul Q groups against K with KV heads only.
  - Carry grouped logits through softmax and grouped value matmul.
  - Flatten `(kv_head, repeat)` back to head order `[0,0,0,0,1,1,1,1]`.
- Or revisit `com.microsoft::GroupQueryAttention`, but only if the inputs can be exported in the op's native flat/cache layout without graph-level `Reshape`.

Expected effect:

- Remove many `Gather` nodes and avoid duplicate K/V reads.
- Potentially reduce the value-side memory footprint.

Risks:

- A grouped MatMul lowering may need reshapes. If those become ONNX `Reshape`, it reintroduces the CPU/provider-boundary issue the current graph solved.
- The exact repeated-head order must match `jnp.repeat(axis=-2)`.
- Existing local notes show the current gated GQA fusion matched 29 temporal islands but regressed when it reintroduced `Reshape`, so this should not be repeated as a simple post-export contrib-op graft.

## Candidate 5: Reorder Existing Rewrites Or Add a MatMul Rewrite After Rotary

Priority: medium.

The current rewrite order is:

1. ORT extended optimization
2. singleton `Reshape -> Squeeze/Unsqueeze`
3. GQA repeat rewrite
4. head projection rewrite
5. slide/static cache rewrite
6. RMSNorm and skip norm rewrites
7. gather cast rewrite
8. RotaryEmbedding rewrite

Because `rewrite_head_projection_reshapes_for_webgpu()` runs before `rewrite_rotary_embedding_for_webgpu()`, it cannot see the final RoPE-transpose structure and cannot coordinate the head layout with the eventual `RotaryEmbedding` op.

Concrete experiment:

- Add a late attention-layout rewrite after RotaryEmbedding that recognizes:
  - `RotaryEmbedding -> Transpose -> Einsum(bqhd,bkhd->bhqk)`
  - `Softmax -> Einsum(bhqk,bkhd->bqhd)`
- Replace the island with BHSD `MatMul` while deleting redundant rotary output transposes.

Risks:

- Post-export attention island matching is brittle because spatial and temporal shapes differ and some K/V inputs come from caches or gathers.
- If it only replaces einsum with local transposes and does not remove rotary transposes, it may regress.
- Dead value info and initializer cleanup needs the same care as the existing RoPE rewrite.

## Recommended Next Step

Implement Candidate 1 first as a gated alternative to `rewrite_head_projection_reshapes_for_webgpu()`.

Reason:

- It targets 478 of the 716 current `Einsum` nodes.
- It is narrower than changing attention semantics.
- It should preserve model behavior exactly because it is only changing how `[H*D]` is viewed as `[H,D]` around existing linear projections.
- It can be validated with graph counts before touching the true attention `Softmax` islands.

Suggested first validation:

1. On a temp copy of `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4.onnx`, replace only projection split/merge einsums with a MatMul/Gemm + static layout alternative.
2. Confirm `Reshape=0`.
3. Confirm projection `Einsum` equations drop from `nk,khd->nhd=359`, `bnhd,hdm->nm=90`, and `nthd,hdm->nm=29`.
4. Run CPU ORT parity on `final_z`, `pred_z`, `candidate_k_entry`, and `candidate_v_entry`.
5. Run the normal WebGPU benchmark before graph-capture benchmark. Graph capture did not materially solve the current dispatch/compute cost by itself.

If Candidate 1 does not improve latency, move to Candidate 2 and implement it together with Candidate 3. A naive `Einsum -> MatMul` rewrite that adds more transposes is unlikely to be enough.

## Summary

The most concrete optimization candidate is replacing the current projection split/merge `Einsum` rewrite with a gated `MatMul`/`Gemm` plus static layout rewrite. The current graph has 716 `Einsum` nodes, and 478 of them were introduced by `rewrite_head_projection_reshapes_for_webgpu()` to avoid `Reshape`. That pass likely fixed WebGPU placement but moved too much dense linear work from optimized matmul kernels into generic einsum kernels.

The true attention einsums should also be lowered to `MatMul`, but only together with a BHSD attention layout cleanup that removes redundant RotaryEmbedding transposes. A local post-export `Einsum -> MatMul` replacement that simply adds transposes risks trading one bottleneck for another.
