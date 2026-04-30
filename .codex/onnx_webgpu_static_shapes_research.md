# ONNX Runtime WebGPU Static-Shape Transfer Findings

Date: 2026-04-29

## Scope

Inspected local artifacts and installed `onnxruntime-web@1.24.3` under:

- `webgpu_app/assets/*.onnx`
- `webgpu_app/bench/results/latest.json`
- `webgpu_app/bench/results/session_profile_summary.json`
- `node_modules/onnxruntime-web/docs/webgpu-operators.md`
- `node_modules/onnxruntime-web/lib/wasm/jsep/**`

No production code was edited.

## Short Answer

The exported fused sample graph `breakout_dynamics_cached_sample_step_b1_t1_s4.onnx` no longer has `Reshape`, and the remaining major ops are WebGPU-supported by local ORT WebGPU. The remaining CPU/device copies shown in the current session profile are mostly from the profiled fallback graph `breakout_dynamics_step_cached_b1_t1.onnx`, not from the fused sample-step graph used by `latest.json`.

One caveat remains in the fused sample graph: node `3`, `node_Cast_2`, casts `actions` to `int64` before action embedding `Gather`. Local WebGPU `Cast` does not support `int64`, so this is still a likely tiny CPU/index upload unless ORT optimizes it away at session creation.

The concrete remaining transfer causes in the fallback profile are:

1. `bool[65]` attention-mask shape work: `Less` runs on WebGPU, then a `bool` `Unsqueeze` runs on CPU, forcing `MemcpyToHost bool[65]` and `MemcpyFromHost bool[1,1,1,65]`.
2. Scalar cache/index control: `Min` is not listed as a WebGPU op, and `Cast(to=int64)` is not supported by the local WebGPU cast shader. This forces CPU work and uploads of tiny `int64` gather indices.
3. Output location choices: any output not requested as `gpu-buffer` is copied back to CPU by ORT Web. The current fused benchmark avoids `candidate_cache_length`, but the fallback diagnostic requests CPU outputs.

## Artifact State

Operator counts from exported ONNX files:

| Artifact | Nodes | Relevant ops |
|---|---:|---|
| `breakout_dynamics_cached_sample_step_b1_t1_s4.onnx` | 6866 | `Mul:1636`, `Add:872`, `Unsqueeze:756`, `Einsum:576`, `Div:483`, `ReduceMean:384`, `Sqrt:384`, `Squeeze:368`, `Gemm:296`, `Concat:258`, `Gather:196`, `Sub:196`, `Split:192`, `Softmax:96`, `QuickGelu:96`, `Transpose:48`, `Slice:28`, `Cast:1` (`actions` to `int64`); **`Reshape:0`** |
| `breakout_dynamics_step_cached_b1_t1.onnx` | 1800 | includes `Unsqueeze:211`, `Einsum:144`, `Gemm:74`, `Gather:53`, plus `Cast:5`, `Min:2`, `Less:1`; **`Reshape:0`** |
| `breakout_dynamics_prefill_cached_b1_t64.onnx` | 1724 | still has `Reshape:268` |
| `breakout_tokenizer_decode_z_b1_t1.onnx` | 405 | has `Reshape:1` |

The latest benchmark (`webgpu_app/bench/results/latest.json`) uses:

- Step artifact: `breakout_dynamics_cached_sample_step_b1_t1_s4`
- Step preferred outputs: `pred_z`, `candidate_k_cache`, `candidate_v_cache`, and `final_z` as `gpu-buffer`
- No `candidate_cache_length` output in the fused sample graph
- Current step timing: mean `67.34 ms`, median `63.82 ms`, min `59.91 ms`, max `97.70 ms`

The diagnostic session profile (`session_profile_summary.json`) instead profiles `breakout_dynamics_step_cached_b1_t1`, with fallback-style cache/index inputs and `candidate_cache_length`.

## ORT WebGPU Support Check

The installed local WebGPU operator table lists support for the remaining sample-step ops:

- `Add`, `Mul`, `Sub`, `Div`
- `Cast`
- `Concat`, `Split`, `Slice`
- `Einsum`
- `Gather`
- `Gemm`, `MatMul`
- `ReduceMean`
- `Sqrt`
- `Softmax`
- `QuickGelu`
- `Squeeze`, `Unsqueeze`
- `Transpose`

Important caveats from local sources:

- `Reshape` is listed in `docs/webgpu-operators.md` with comment `no GPU kernel`.
- `Shape` is also listed as `no GPU kernel`.
- `Min` is not listed as a WebGPU op. `ReduceMin` is listed, but variadic elementwise `Min` is not.
- `Cast` is listed, but `lib/wasm/jsep/webgpu/ops/unary-op.ts` only accepts cast targets `float16`, `float`, `uint32`, `int32`, and `bool`. `Cast(to=7)` means `int64`, so those casts cannot use the local WebGPU cast shader.
- The TypeScript WebGPU op map in `op-resolve-rules.ts` does not include `Reshape`, `Shape`, `Squeeze`, or `Unsqueeze`; however, the profile shows most `Squeeze`/`Unsqueeze` nodes assigned to `WebGpuExecutionProvider`. Treat the generated operator table plus profiler assignment as the practical support signal for `Squeeze`/`Unsqueeze`, with dtype-specific exceptions.

## Profile Evidence

From `session_profile_summary.json`, for the profiled fallback step graph:

| Provider/op | Count | Total |
|---|---:|---:|
| `WebGpuExecutionProvider` | 1812 | `59.877 ms` |
| `CPUExecutionProvider` | 7 | `0.182 ms` |
| `WebGpuExecutionProvider|MemcpyToHost` | 3 | `12.444 ms` |
| `WebGpuExecutionProvider|MemcpyFromHost` | 5 | `0.183 ms` |
| `CPUExecutionProvider|Cast` | 4 | `0.098 ms` |
| `CPUExecutionProvider|Min` | 2 | `0.045 ms` |
| `CPUExecutionProvider|Unsqueeze` | 1 | `0.039 ms` |

Top transfer:

- `Memcpy_token_58_kernel_time`: `MemcpyToHost`, `bool[65]`, `10.274 ms`.

Fallback graph node sequence around that transfer:

```text
0  Squeeze(cache_length) -> squeeze_out_28
1  Add(squeeze_out_28, 1) -> add_out_218
2  Less(arange_out_9, add_out_218) -> lt_out_5
3  Unsqueeze(lt_out_5) -> bcast_reshape_out_169
4  Cast(bcast_reshape_out_169, to=float) -> convert_out_23
```

The profile assigns that `Unsqueeze` to CPU for `bool[65]`, while adjacent `Less`/`Cast` are WebGPU. That creates GPU-to-CPU for `lt_out_5`, then CPU-to-GPU for the cast input.

Other CPU-producing fallback nodes:

```text
8    Min(position_index, 64) -> minimum_out_0
9    Cast(minimum_out_0, to=int64) -> take_indices_int64_0
15   Cast(signal_levels, to=int64) -> embed_idx_i64_2
17   Cast(step_levels, to=int64) -> embed_idx_i64_1
21   Cast(actions, to=int64) -> embed_idx_i64_0
1799 Min(cache_length + 1, 64) -> candidate_cache_length
```

These explain the tiny `int64` `MemcpyFromHost` events before WebGPU `Gather`.

## Why This Still Matters For <=50 ms

The fused sample-step graph likely removed the worst CPU partition sources:

- no `cache_length` input or output
- no `position_index`
- no `step_levels` or `signal_levels`
- no `Min`
- no `Less`
- only one remaining `Cast(to=int64)` for `actions`, instead of five in the fallback graph
- no `Reshape`

So the current `~67 ms` fused step is probably dominated by many small GPU kernels and dispatch overhead, not CPU fallback. The graph has 6866 nodes for one frame: 756 `Unsqueeze`, 368 `Squeeze`, 576 `Einsum`, 296 `Gemm`, 258 `Concat`, 196 `Gather`, and many elementwise/reduction kernels. Even when all run on GPU, this is a large number of dispatches for a <=50 ms target.

The fallback profile is still useful because it identifies exact patterns to avoid in any graph that still carries scalar cache/index logic.

## Actionable Findings

1. Profile the actual fused sample artifact with ORT session profiling.

   The existing `profile_diagnostic.js` default and spec target `breakout_dynamics_step_cached_b1_t1`. For the <=50 ms path, the diagnostic should be run against `breakout_dynamics_cached_sample_step_b1_t1_s4` with the same preferred output policy as the benchmark. Otherwise the profile overstates CPU copies that are not on the production fused path.

2. Keep fused sample outputs GPU-only.

   `wasm-core-impl.ts` defaults preferred outputs to `cpu`; only `preferredOutputLocation: 'gpu-buffer'` keeps outputs resident. The latest benchmark already does this for `pred_z`, `candidate_k_cache`, `candidate_v_cache`, and `final_z`. Preserve that. Do not reintroduce `candidate_cache_length` as a fetched CPU output on the fused path.

3. Avoid `Cast(to=int64)` before `Gather`.

   ONNX `Gather` can use integer indices, and the local WebGPU `Gather` shader accepts the index tensor's data type. The local WebGPU `Cast` shader does not support `int64`. Export or rewrite embedding/gather indices as `int32` so `actions`, `step_levels`, `signal_levels`, `position_index`, and similar indices do not route through CPU `Cast`. This still applies to the fused sample graph because `node_Cast_2` casts `actions` to `int64`.

4. Remove or rewrite scalar `Min`.

   WebGPU supports `ReduceMin`, not variadic elementwise `Min`. Replace clamp-like scalar `Min(x, 64)` patterns with an export-time constant, a JS-side preclamped feed, or a WebGPU-supported expression such as `Where(Less(x, 64), x, 64)` if the value must remain in-graph.

5. Avoid `bool` `Unsqueeze` in mask construction.

   The observed fallback transfer is `Less -> bool[65] -> CPU Unsqueeze -> GPU Cast`. Prefer generating the mask directly at `[1,1,1,65]`, or cast the `bool[65]` to `float` on GPU before reshaping/unsqueezing if float `Unsqueeze` stays on WebGPU. This targets the largest observed transfer (`10.274 ms`).

6. Treat remaining fused-step work as GPU dispatch count reduction.

   Since the fused graph's remaining ops are supported, the next performance work should target graph simplification/fusion:

   - reduce `Unsqueeze`/`Squeeze` count by emitting tensors in final rank/layout earlier
   - fuse RMSNorm decomposition (`Mul`, `ReduceMean`, `Add`, `Sqrt`, `Div`, `Mul`) into `SimplifiedLayerNormalization`/`SkipSimplifiedLayerNormalization` if numerically acceptable and supported
   - use ORT contrib attention/GQA only if it can cover masks and past/present for this exact cache contract; local docs still say `MultiHeadAttention` needs mask and past/present work
   - inspect whether `Einsum` head projection/merge patterns can become `MatMul`/`Gemm` shapes that use faster paths
   - keep cache update as one fused write per frame, which the current sample graph already does

## Verification Commands Used

```bash
uv run python - <<'PY'
from pathlib import Path
from collections import Counter
import onnx
for path in sorted(Path('webgpu_app/assets').glob('*.onnx')):
    m = onnx.load(path, load_external_data=False)
    print(path.name, len(m.graph.node), Counter(n.op_type for n in m.graph.node))
PY
```

```bash
node - <<'NODE'
import fs from 'node:fs';
const s = JSON.parse(fs.readFileSync('webgpu_app/bench/results/session_profile_summary.json', 'utf8'));
console.log(s.counts);
console.log(s.totals.ops.slice(0, 30));
NODE
```

## Bottom Line

`Reshape` removal did what it was supposed to do for the fused sample graph. The remaining copies in the available profile are from the fallback step graph's scalar/index/mask side path, especially `bool` `Unsqueeze`, `Min`, and `Cast(to=int64)`. For the current fused graph, remove the one remaining action `Cast(to=int64)`, then treat the <=50 ms blocker as the sheer number of supported WebGPU kernels and dispatches rather than unsupported `Einsum`/`Gemm`/`Gather`.
