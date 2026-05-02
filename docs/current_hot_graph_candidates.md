# Current Hot Graph Optimization Candidates

Date: 2026-05-03

Scope: fp32 ONNX Runtime WebGPU optimization candidates for the current browser
Dreamer4 dynamics hot path. This analysis avoids fp16 and bf16 and does not
propose behavior-changing sampling shortcuts.

## Source Of Truth

Current hot artifact on disk:

- `webgpu_app/assets/breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4.onnx`
- Inputs: `sample_noise`, `context_noise`, `actions`, `k_cache`, `v_cache`
- Outputs: `final_z`, `pred_z`, `candidate_k_entry`, `candidate_v_entry`
- Size: `245,785,271` bytes, about `234.4 MiB`
- Current SHA256 on disk: `1f1ccb34d7a0d5303a7b7862dc0551516e1adb933f5599991679501002054160`

Benchmark source:

- `webgpu_app/bench/results/latest.json`
- `graphCapture=false`
- Dynamics frame: `100.93 ms` mean, `97.45 ms` median, `114.04 ms` p95
- Decoder frame: `6.34 ms` mean, `6.14 ms` median, `7.21 ms` p95
- Cache commit shader: `0.004 ms` mean
- Streaming frame: `107.39 ms` mean, `103.75 ms` median, `121.41 ms` p95

Accuracy source:

- `webgpu_app/bench/results/raw_optimized_onnx_accuracy.json` passes for the
  entry artifact.
- `webgpu_app/bench/results/entry_cache_update_accuracy.json` passes entry
  cache reconstruction against the full-cache artifact. The full-cache vs
  entry-cache K update max abs error is `2.3841858e-7`; V cache, `final_z`, and
  `pred_z` are exact in that check.

Important freshness note:

- The manifest timestamp is `2026-05-03 00:53:00`.
- The latest benchmark timestamp is `2026-05-03 00:55:16`.
- The current entry ONNX file timestamp is `2026-05-03 00:57:42`.
- The manifest SHA for the entry artifact is stale relative to the file now on
  disk. The graph counts below use the actual ONNX file on disk, not the
  manifest. Before attributing any speedup to a new change, refresh the manifest
  and rerun the benchmark for this exact artifact.

## Hot Graph Facts

Current op counts for the entry artifact:

| Op | Count |
| --- | ---: |
| `Mul` | 1200 |
| `Unsqueeze` | 915 |
| `Einsum` | 716 |
| `Squeeze` | 451 |
| `Gemm` | 366 |
| `Add` | 304 |
| `Concat` | 304 |
| `SimplifiedLayerNormalization` | 299 |
| `Sub` | 243 |
| `Gather` | 239 |
| `Split` | 239 |
| `SkipSimplifiedLayerNormalization` | 179 |
| `Softmax` | 119 |
| `QuickGelu` | 119 |
| `Transpose` | 59 |
| `Slice` | 16 |
| `Div` | 3 |

Total nodes: `5771`.

Notably absent in the current hot entry file: `Reshape`, `Cast`, `Expand`,
`Less`, `Shape`, and `Memcpy*`. That means the previous major provider-boundary
class is already handled for this artifact.

The graph is still mostly five unrolled transformer passes:

- Four sample predict passes from `sample_step_predict_only`.
- One context-entry pass from `step_entries`, which emits only K/V entries for
  the browser-side in-place cache slide/rebase update.
- The context pass is behaviorally required by the current cache policy because
  it runs with `noised_context_z`, the context shortcut level, and the context
  signal level. Reusing final sample-step K/V would change behavior.

Attention shape breakdown:

- `90` spatial attention softmaxes with output shape `[1, 8, 36, 36]`.
- `29` cached temporal attention softmaxes with output shape `[36, 8, 1, 65]`.
- `119` score equations: `bqhd,bkhd->bhqk`.
- `119` value equations: `bhqk,bkhd->bqhd`.
- `239` GQA repeat gathers remain after the previous repeat-materialization
  rewrite.

Projection and MLP breakdown:

- `359` head projection `Einsum` nodes with equation `nk,khd->nhd`.
- `90` spatial output projection `Einsum` nodes with equation `bnhd,hdm->nm`.
- `29` temporal output projection `Einsum` nodes with equation `nthd,hdm->nm`.
- `238` MLP gate/value `Gemm` nodes of shape `[36,256] -> [36,768]`.
- `119` MLP output `Gemm` nodes of shape `[36,768] -> [36,256]`.
- `119` `QuickGelu` nodes.

RoPE breakdown:

- `239` RoPE split inputs:
  - `90` x `[1,36,2,64]`
  - `90` x `[1,36,8,64]`
  - `30` x `[36,1,2,64]`
  - `29` x `[36,1,8,64]`
- These lower to repeated `Split`, `Mul`, `Add`, `Sub`, and `Concat` islands.
  The ORT WebGPU support table includes `com.microsoft::RotaryEmbedding`, so
  this is now the most concrete remaining layout/elementwise fusion target.

## Ranked Candidates

### 0. First: Lock Down The Current Baseline

This is not a speedup candidate, but it should happen before comparing new
exports.

The current ONNX file is newer than both the manifest and the benchmark result,
and it contains `179` `SkipSimplifiedLayerNormalization` nodes that are not
reflected in the manifest's recorded `rmsnorm_rewrite` counts. Refresh the
manifest/accuracy artifacts and rerun the browser benchmark for the exact file
on disk before measuring any next candidate.

Acceptance gate:

- Manifest SHA matches the current ONNX file.
- Raw-vs-optimized and entry-cache reconstruction checks still pass.
- Baseline benchmark result is regenerated for the same artifact.

### 1. Replace RoPE Decomposition With `com.microsoft::RotaryEmbedding`

Expected impact: high.

Why this is the best next graph rewrite:

- It targets `239` repeated RoPE islands that currently account for most of the
  remaining `Split`, `Sub`, and many `Mul`/`Add`/`Concat` nodes.
- ORT WebGPU 1.24.3 lists `RotaryEmbedding | com.microsoft(1+)` as supported.
- The implementation convention matches this repo's non-interleaved RoPE:
  left/right halves, `left * cos - right * sin`, `left * sin + right * cos`.
- It does not require fp16/bf16, cache ABI changes, or a full attention rewrite.

Concrete implementation path:

- Add a post-export ONNX rewrite that matches `apply_rotary_embedding()` after
  the graph has static 4D Q/K shapes.
- Replace each exact pattern:
  `Split(axis=-1) -> Mul/Mul/Sub and Mul/Mul/Add -> Concat(axis=-1)`
  with one `com.microsoft::RotaryEmbedding` node.
- Use attributes:
  - `interleaved=0`
  - `num_heads=8` for Q, `num_heads=2` for K
  - `rotary_embedding_dim=0`
  - `scale=1.0`
- Use `position_ids` instead of graph arithmetic:
  - spatial dynamics attention can use position offset `0` over sequence length
    `36`;
  - temporal sample/context paths should use the same fixed positions already
    represented by the exported cos/sin constants, or a fully materialized
    `[batch, sequence]` `position_ids` initializer if matching pass identity is
    simpler.
- Place the pass after head projection rewrite and before final metadata
  capture. It should leave `Reshape=0`, `Cast=0`, and `Expand=0`.

Risk:

- Medium. The op is WebGPU-supported, but the exact position-id behavior must be
  validated against the current constants.
- If Python ORT cannot execute the contrib op, add a deterministic browser
  comparison between unfused and fused entry artifacts.

Acceptance gate:

- Entry artifact parity for `final_z`, `candidate_k_entry`, and
  `candidate_v_entry`.
- Entry-cache reconstruction vs full-cache still passes.
- Browser session/profile confirms no CPU fallback or reintroduced `Reshape`.
- Benchmark improves the dynamics frame, not just node count.

### 2. Validate Existing Packed QKV Projection Export

Expected impact: medium to high.

The current working-tree export code already has `--packed_qkv_projection` wired
through the dynamics export path and `_packed_qkv_projection()` in
`visionary/export/onnx_wrappers.py`. The current manifest has it disabled.

Why it is worth testing:

- Current graph has `359` head projection `Einsum` nodes.
- Packing Q/K/V should replace three projection dispatches per attention block
  with one wider projection plus a split.
- It preserves fp32 math and reads the existing checkpoint kernels in order:
  `Dense_0`, `Dense_1`, `Dense_2`.

Main risk:

- The packed path may interact badly with `rewrite_head_projection_reshapes_for_webgpu()`.
  If it reintroduces standalone `Reshape`, `Cast`, or `Expand`, reject it until
  the head projection rewrite understands the packed shape directly.

Acceptance gate:

- Export only to a temporary output directory first.
- Compare raw and optimized ONNX outputs.
- Check graph counts for lower projection dispatch count and still `Reshape=0`.
- Run the entry-cache reconstruction check.
- Benchmark packed-QKV alone before combining it with packed SwiGLU or RoPE.

### 3. Validate Existing Packed SwiGLU Gate/Value Export

Expected impact: medium.

The current working-tree export code also has `--packed_swiglu_projection` and an
export-only `_ExportSwiGLU`.

Why it is worth testing:

- Current graph has `238` MLP gate/value `Gemm` nodes plus `119` MLP output
  `Gemm` nodes.
- Packing gate/value should remove one large projection dispatch per transformer
  block, replacing two `[36,256] -> [36,768]` GEMMs with one
  `[36,256] -> [36,1536]` projection plus a split.
- It is behavior-preserving if kernels are concatenated exactly and split back
  in the original gate/value order.

Main risk:

- Wider GEMM may be less favorable for ORT WebGPU than two smaller GEMMs on this
  workload. The only reliable answer is a benchmark.

Acceptance gate:

- Test independently from packed QKV.
- Graph still has no `Reshape`, `Cast`, or `Expand`.
- `QuickGelu` count should remain `119`; do not decompose it on fp32.
- CPU/browser parity and benchmark pass.

### 4. Revisit Fused Attention Only As A Flat, No-Reshape GQA Export

Expected impact: high if solved, high risk.

What it targets:

- `119` `Softmax` nodes.
- `238` attention score/value `Einsum` nodes.
- `239` GQA repeat `Gather` nodes.

Do not simply enable the current `--fused_temporal_gqa` path as-is. Prior notes
show the corrected GQA fusion smoke-tested but regressed badly: streaming frame
around `195 ms` mean versus the restored fp32 baseline around `116 ms` at that
time. The current rewrite path also introduces reshape/transpose overhead around
the contrib op.

A viable next GQA attempt should be stricter:

- Emit or rewrite flat Q/K/V tensors directly for `GroupQueryAttention`.
- Keep graph-level `Reshape=0`.
- Store past K/V in GQA-native BNSH layout for fused artifacts so each temporal
  attention block does not pay two past-cache `Transpose` dispatches.
- Consider `GroupQueryAttention` `do_rotary=1` only after standalone
  `RotaryEmbedding` parity is proven.
- Prune dead original attention nodes after fusion.
- Add post-fusion accuracy gates, not only pre-fusion validation.

This should come after Candidate 1 or Candidate 2, because fused GQA benefits
from the same flat projection and RoPE work.

### 5. Trim Unused Runtime Outputs From The Hot Entry Artifact

Expected impact: low.

The browser result records that it fetches GPU `final_z` and GPU cache entry
outputs once per frame and does not fetch `pred_z`. Keeping `pred_z` as an ONNX
graph output likely does not add major compute because `final_z` already depends
on the final prediction, but it can still force an output binding/allocation.

Candidate:

- Add a production entry artifact variant that outputs only `final_z`,
  `candidate_k_entry`, and `candidate_v_entry`.
- Keep the current `pred_z` artifact for diagnostics and parity tests.

Acceptance gate:

- Same generated frames and cache reconstruction.
- Browser harness no longer expects or exposes `pred_z` for the production
  steady-state path.
- Benchmark shows a measurable allocation/output-binding improvement. If not,
  drop it.

## Not Recommended Now

- Do not pursue fp16 or bf16 for the hot dynamics graph in this pass. Prior notes
  show bf16 is not accepted by key WebGPU paths, and fp16 changes the precision
  problem rather than preserving the current fp32 behavior.
- Do not re-run the older grouped-GQA decomposition as the primary path. It was
  browser-compatible but slower and still decomposes attention into many generic
  ops.
- Do not re-enable current post-export GQA fusion without removing its
  reshape/transpose overhead and adding a post-fusion parity gate.
- Do not replace fixed `sample_steps=4` unrolling with ONNX `Loop` for the
  browser hot path. It may shrink the file, but it is unlikely to reduce WebGPU
  compute or dispatch cost and can make provider placement less predictable.

## Recommended Order

1. Refresh the manifest and benchmark for the exact current entry artifact.
2. Implement or prototype the `RotaryEmbedding` rewrite on the entry artifact
   only.
3. Independently validate `--packed_qkv_projection`.
4. Independently validate `--packed_swiglu_projection`.
5. Combine the winners and rerun raw-vs-optimized, entry-cache reconstruction,
   normal WebGPU benchmark, and graph-capture benchmark.
6. Revisit flat GQA only after the projection/RoPE candidates establish a cleaner
   attention boundary.
