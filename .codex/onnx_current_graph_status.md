# ONNX Current Graph Status

Date: 2026-05-02

## Scope

This note records the current ONNX/WebGPU state after regenerating the Breakout assets with the fp32 export path:

```bash
uv run --python 3.11 --group onnx python scripts/webgpu/export_dreamer4_onnx.py \
  --tokenizer_dir gs://visionary-exp/breakout/checkpoints/tokenizer_l8p8 \
  --tokenizer_step 1000000 \
  --dynamics_dir gs://visionary-exp/breakout/checkpoints/dynamics_l24 \
  --dynamics_step 1000000 \
  --out_dir webgpu_app/assets \
  --seq_len 64 \
  --export_cached \
  --validate \
  --overwrite
```

No `--float16`, no `--grouped_gqa_attention`, no `--fused_temporal_gqa`, and no `--simplify_onnx` were used for this snapshot.

## Current Demo Contract

- Preferred prefill: `breakout_dynamics_prefill_cached_b1_t64`
- Preferred steady-state step: `breakout_dynamics_sample_append_context_slide_b1_t1_s4`
- Preferred decoder: `breakout_tokenizer_decode_z_b1_t1`
- Precision: fp32
- Sample steps: 4
- Context length/cache length: 64
- Cache shape per K/V tensor: `[6, 1, 36, 64, 2, 64]`
- K/V cache bytes per tensor: 7.08 MB
- K/V cache bytes input plus output per frame: about 27 MB of GPU-side tensor payload

## Validation

The regenerated artifacts passed export-time ONNX Runtime CPU validation. The hot steady-state graph reported:

- `final_z` max absolute error: `3.278e-6`
- `pred_z` max absolute error: `3.278e-6`
- cache tensor max absolute errors: roughly `1e-6` to `2e-5`
- validation tolerance: `atol=0.05`, `rtol=0.05`

## Browser Benchmark

Command:

```bash
bun install --frozen-lockfile
bun run benchmark:webgpu
```

Result file:

- `webgpu_app/bench/results/latest.json`

Environment:

- Chromium through Playwright
- WebGPU adapter vendor: `apple`
- architecture: `metal-3`
- ORT Web package: `onnxruntime-web@1.24.3`

Clean benchmark result:

| Phase | Mean | Median | P95 | Throughput |
|---|---:|---:|---:|---:|
| prefill | 718.71 ms | 718.71 ms | 718.71 ms | 1.39 Hz |
| dynamics steady step | 103.70 ms | 101.35 ms | 112.72 ms | 9.64 Hz |
| decoder | 5.85 ms | 5.74 ms | 6.36 ms | 170.98 Hz |
| streaming frame | 109.65 ms | 107.30 ms | 119.14 ms | 9.12 Hz |

Conclusion: the demo is currently dynamics-bound. Decoder work is only about 5-6 ms/frame, so decoder optimization alone cannot get us near 20 fps.

The WebGPU profiling test passed, but ORT WebGPU did not emit raw profiling events in this environment:

- profiling enabled: true
- profiling available: false
- raw events: 0

So current bottleneck attribution has to come from graph structure plus wall-clock phase timings.

## Hot Graph Counts

`breakout_dynamics_sample_append_context_slide_b1_t1_s4.onnx`:

- file size: 235.1 MB
- nodes: 6,183
- inputs: 7
- outputs: 5
- domains: `ai.onnx`: 6,064, `com.microsoft`: 119

Key op counts:

| Op | Count |
|---|---:|
| Mul | 1,106 |
| Unsqueeze | 936 |
| Einsum | 715 |
| Add | 614 |
| SimplifiedLayerNormalization | 478 |
| Squeeze | 453 |
| Gemm | 367 |
| Concat | 323 |
| Split | 246 |
| Gather | 245 |
| Div | 122 |
| Softmax | 119 |
| QuickGelu | 119 |
| Transpose | 59 |
| Slice | 28 |
| Cast | 1 |
| Min | 1 |

Important absence:

- `Reshape`: 0
- `Expand`: 0
- `Shape`: 0
- `Size`: 0

This means the previous reshape-driven CPU transfer issue is solved for the hot steady-state graph. The remaining problem is graph size, dispatch count, attention cost, and full cache read/write cost.

## Other Graph Counts

`breakout_tokenizer_decode_z_b1_t1.onnx`:

- nodes: 407
- `Reshape`: 0
- `Einsum`: 48
- `Gemm`: 26
- `Softmax`: 8
- `SimplifiedLayerNormalization`: 32

`breakout_dynamics_prefill_cached_b1_t64.onnx`:

- nodes: 1,241
- `Reshape`: 268
- `Einsum`: 48
- `Gemm`: 170
- `Softmax`: 24
- `SimplifiedLayerNormalization`: 96

The prefill graph still contains many `Reshape` nodes, but it runs once at startup/context reset. It is not the steady-state bottleneck.

`breakout_dynamics_sample_append_context_slide_layer_b1_t1_s4.onnx`:

- nodes: 6,145
- `Reshape`: 0
- `Einsum`: 715
- `Squeeze`: 441
- `Unsqueeze`: 924
- `Slice`: 16
- `Concat`: 321

The layer-cache variant is only slightly smaller than the stacked-cache variant. It is not currently preferred, and its current graph shape does not look like a major win by itself.

## Existing Rewrites That Are Working

Current manifest flags:

- `gqa_repeat_to_gather`: enabled
- `head_projection_reshape_to_einsum`: enabled
- `singleton_reshape_to_squeeze_unsqueeze`: enabled
- RMSNorm decomposition replaced by `com.microsoft::SimplifiedLayerNormalization`
- hot steady-state static cache rewrite: enabled

Observed effects in the preferred hot graph:

- GQA materialized repeat `Expand -> Reshape -> Einsum` is gone.
- Head split/merge `Gemm -> Reshape` and `Reshape -> Gemm` patterns are gone.
- Steady-state graph has no `Reshape`.
- Decomposed RMSNorm arithmetic is fused to 478 `SimplifiedLayerNormalization` nodes.

## Remaining Patterns

The graph has a large number of layout-only nodes even after removing `Reshape`:

- 936 `Unsqueeze`
- 453 `Squeeze`
- 323 `Concat`
- 246 `Split`
- 245 `Gather`

Frequent local patterns in the hot graph:

- `SimplifiedLayerNormalization -> Squeeze -> Einsum`: 119
- `SimplifiedLayerNormalization -> Squeeze -> Gemm`: 119
- `Mul -> Squeeze -> Gemm`: 119
- `Einsum -> Unsqueeze -> SimplifiedLayerNormalization`: 239
- `Gemm -> Unsqueeze -> QuickGelu`: 119
- `Gemm -> Unsqueeze -> Mul`: 119
- `Gemm -> Unsqueeze -> Add`: 119
- `Concat/Gather -> Einsum -> Div/Softmax`: 119 attention score paths
- `Softmax/Gather -> Einsum`: 119 attention value paths

This suggests the graph is now paying for many small GPU dispatches and rank-adapter ops, not host/device copies.

## Ranked Optimization Points

### 1. Run `onnxsim` in the fresh export pipeline for demo artifacts

Current snapshot was exported without `--simplify_onnx`. Earlier notes show the exporter can run simplification before `SimplifiedLayerNormalization` is introduced, avoiding the `onnxsim` failure on contrib RMSNorm. This is the lowest-risk next experiment.

Expected target:

- reduce total node count before the custom rewrites
- fold dead constants and shape plumbing
- possibly remove some `Squeeze`/`Unsqueeze`/`Concat` chains

Use:

```bash
uv run --python 3.11 --group onnx python scripts/webgpu/export_dreamer4_onnx.py \
  ... \
  --export_cached \
  --validate \
  --overwrite \
  --simplify_onnx \
  --simplify_demo_only
```

Then compare hot graph node counts and `bun run benchmark:webgpu`.

### 2. Reduce full-cache output/update cost

The current steady-state graph reads full K/V cache inputs and returns full K/V cache outputs every frame:

- input K/V: about 13.5 MB
- output K/V: about 13.5 MB
- total cache tensor payload touched at the graph boundary: about 27 MB/frame

This is currently GPU-buffer based, so it is not a host transfer. But it still forces large output allocation/copy/update work and keeps the graph ABI heavy.

Best behavior-preserving direction:

- output only the new K/V slice per layer/token, not the full shifted cache
- maintain a browser-owned GPU ring buffer for the cache
- pass cache position metadata to attention
- avoid rebuilding and returning the full cache every step

This is larger than graph surgery. It should be implemented from the export/inference wrapper, not by renaming graph outputs after export.

### 3. Export-native fused attention or attention plugin path

The hot graph contains:

- 119 `Softmax`
- 238 attention `Einsum` nodes
- 245 `Gather` nodes from GQA repeat removal
- many rank adapters around attention

The current graph uses generic ONNX ops. The main remaining structural win is to express attention as a fused operator that ORT WebGPU can execute as fewer kernels, or to change the wrapper lowering so grouped attention is computed without the current split/gather/einsum plumbing.

The previous `--grouped_gqa_attention` path validated poorly for speed and was not the same as a fused attention op. It changed the decomposition but still produced a large graph. The goal now should be fewer dispatches, not merely a different GQA decomposition.

### 4. Rank-preserving dense/MLP lowering

There are many repeated patterns around dense blocks:

- `Squeeze -> Gemm -> Unsqueeze`
- `Gemm -> Unsqueeze -> QuickGelu`
- `Gemm -> Unsqueeze -> Add`

These are probably on GPU now, but each adapter is still a node dispatch. A rank-preserving export of dense/MLP operations could trade 2D `Gemm` plus adapters for rank-aware `Einsum`/`MatMul` forms. This needs benchmark validation because `Gemm` may be faster than `Einsum`; the win only exists if dispatch reduction beats kernel cost.

### 5. Keep fp32 for now

BF16 is not usable in the installed ORT WebGPU runtime for this graph. FP16 caused demo instability/black frames in prior attempts. Since this snapshot validates and is behaviorally stable in fp32, precision should not be the immediate optimization axis.

### 6. Prefill and decoder are secondary

Prefill is slow at about 719 ms, but it happens once for a context. The decoder is about 5-6 ms/frame. The live frame budget is dominated by the 4-sample dynamics graph.

## Immediate Next Experiment

The next concrete experiment should be:

1. Re-export with `--simplify_onnx --simplify_demo_only`.
2. Verify numerical parity.
3. Recount the hot graph.
4. Run `bun run benchmark:webgpu`.
5. If it is still around 100 ms/frame, stop spending time on decoder/prefill and move to cache-delta ABI or fused attention.

