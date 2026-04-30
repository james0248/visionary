# ONNX Runtime WebGPU Optimization Research

Date: 2026-04-29
Repo: `/Users/hyeonseok/Documents/Sources/visionary`

## Executive Summary

The current bottleneck is primarily ORT WebGPU provider-boundary traffic, not the CPU cost of reshape math.

Local session profiling for `breakout_dynamics_step_cached_b1_t1.onnx` shows:

- `WebGpuExecutionProvider|MemcpyToHost`: 394 events, 278.078 ms total.
- `WebGpuExecutionProvider|MemcpyFromHost`: 402 events, 44.020 ms total.
- `CPUExecutionProvider|Reshape`: 399 events, only 6.956 ms total.
- `WebGpuExecutionProvider|Concat/Expand/Einsum/Transpose`: placed on WebGPU in this trace.
- Diagnostic run: WebGPU session run was 415.54 ms, while WASM was 93.55 ms for the same diagnostic path.

So the high-cost pattern is:

`WebGPU op -> MemcpyToHost -> CPU Reshape -> MemcpyFromHost -> WebGPU op`

This happens repeatedly in attention, especially around GQA K/V repeat materialization:

`Reshape [36,65,2,64] -> [36,65,2,1,64]`
`Expand -> [36,65,2,4,64]`
`Reshape -> [36,65,8,64]`

## Is jax2onnx Causing It?

Partly, but not in the most important sense.

jax2onnx is emitting valid ONNX graph structure for the JAX/einops program. Its local plugins explicitly lower:

- `jax.lax.reshape` to ONNX `Reshape`: `.venv/lib/python3.11/site-packages/jax2onnx/plugins/jax/lax/reshape.py`
- `jax.lax.broadcast_in_dim` to `Reshape` + `Expand`: `.venv/lib/python3.11/site-packages/jax2onnx/plugins/jax/lax/broadcast_in_dim.py`
- `jax.lax.transpose` to `Transpose`: `.venv/lib/python3.11/site-packages/jax2onnx/plugins/jax/lax/transpose.py`
- `jax.lax.concatenate` to `Concat`: `.venv/lib/python3.11/site-packages/jax2onnx/plugins/jax/lax/concatenate.py`
- complex/high-rank `dot_general` to `Einsum`: `.venv/lib/python3.11/site-packages/jax2onnx/plugins/jax/lax/dot_general.py`

The exporter is also already static-shaped and already runs offline ORT optimization. The manifest reports `static: true`, and for the cached step graph ORT optimization reduced:

- nodes: 2987 -> 1902
- `Reshape`: 708 -> 399
- `Concat`: 420 -> 76
- `Expand`: 172 -> 48

That means static shapes and ORT graph cleanup are working, but many semantic view/broadcast nodes remain.

The CPU fallback itself is ORT WebGPU placement behavior. In installed `onnxruntime-web@1.24.3`, the generated WebGPU operator table says `Reshape` has "no GPU kernel" while `Concat`, `Einsum`, `Expand`, and `Transpose` are WebGPU-supported. Local source confirms WebGPU JS shader implementations for `Concat`, `Expand`, `Einsum`, and `Transpose`, but no WebGPU JS `Reshape` implementation is registered in `op-resolve-rules.ts`.

Important nuance: an ORT maintainer states that WebGPU/JSEP `Reshape`, `Squeeze`, and `Unsqueeze` are intended to be metadata-only and not modify tensor data, with `Reshape` aliasing input data. However, this repo's actual 1.24.3 session trace shows `CPUExecutionProvider|Reshape` bracketed by host/device copies, so the installed/runtime behavior is not achieving metadata-only placement for this graph.

## Provider Placement Findings

From local `node_modules/onnxruntime-web/docs/webgpu-operators.md`:

- `Concat`: WebGPU supported.
- `Einsum`: WebGPU supported.
- `Expand`: WebGPU supported.
- `Transpose`: WebGPU supported, "need perf optimization".
- `Reshape`: supported but comment is "no GPU kernel".
- `Shape`: supported but comment is "no GPU kernel; an ORT warning is generated".
- `Min` is not listed as a WebGPU operator. The trace has small `CPUExecutionProvider|Min` nodes from cache length logic.

From local `node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/op-resolve-rules.ts`:

- `Concat`, `Einsum`, `Expand`, and `Transpose` are registered in `WEBGPU_OP_RESOLVE_RULES`.
- `Reshape` is not registered there.

From local WebGPU op sources:

- `Expand` reads the shape tensor on CPU via `inputs[1].getBigInt64Array()` and emits a WebGPU compute shader for the data output.
- `Concat` emits a WebGPU shader.
- `Einsum` emits a general WebGPU shader.
- `Transpose` emits WebGPU kernels, including optimized/shared and copy-as-transpose cases.

## IO Binding / GPUBuffer Findings

Official ORT WebGPU docs say default model inputs/outputs are CPU tensors; WebGPU runs copy inputs to GPU and outputs back to CPU. They recommend IO binding for transformer-style repeated execution.

Usable APIs in installed package:

- `ort.Tensor.fromGpuBuffer(gpuBuffer, { dataType, dims })`
- session option `preferredOutputLocation: 'gpu-buffer'` or per-output map.
- preallocated output tensors passed as `fetches` object: `{ outputName: gpuTensor }`.
- `ort.env.webgpu.device` is available after WebGPU initialization.

Local implementation details:

- `preferredOutputLocation` creates IO binding only when at least one output wants GPU/ML tensor location.
- With `enableGraphCapture: true`, if no `preferredOutputLocation` is supplied, outputs default to `gpu-buffer`.
- With `enableGraphCapture: true`, every external input/output tensor must be `gpu-buffer`; CPU tensors are rejected.
- Proxy worker does not support non-CPU tensor locations, and official docs say the proxy worker cannot work with WebGPU EP because GPU buffers are not transferable.

Current benchmark state:

- `webgpu_app/bench/benchmark.js` already uses `preferredOutputLocationFor()` for cache outputs, fused final z, and decoder output.
- `webgpu_app/bench/profile_diagnostic.js` does not use `preferredOutputLocation`, so its trace includes default output-to-CPU behavior. It still clearly exposes internal CPU reshape boundaries.
- `makeFloatTensor()`, `makeIntTensor()`, and `makeScalarFillTensor()` still create CPU tensors. This affects initial `z`, action, step/signal levels, position index, and any CPU-side generated feed.

## Static Shapes / Graph Capture

Static shapes help, but they are not sufficient.

Evidence:

- The exported manifest records static axes.
- jax2onnx's reshape plugin tries to fold all-constant target shapes into a single initializer and avoid dynamic `Shape/Gather/Concat` plumbing.
- ORT offline optimization already folds hundreds of shape nodes.

But static shapes do not make ORT WebGPU place `Reshape` on WebGPU in this trace. Static shapes are a prerequisite for `enableGraphCapture`, not a fix for CPU fallback.

Graph capture is not currently viable for the cached step graph as traced:

- Official docs: graph capture needs static shapes and all computing kernels running on WebGPU EP.
- Trace still has CPU `Reshape`, `Min`, and `Cast`.
- Local `wasm-core-impl.ts` rejects CPU external buffers when graph capture is enabled.

Graph capture should be treated as a later optimization after CPU provider nodes are removed or proven metadata-only.

## Session Options / Transforms

Current `createSession()` in `webgpu_app/bench/benchmark.js` uses:

```js
{
  executionProviders: [{ name: 'webgpu' }],
  externalData,
  graphOptimizationLevel: 'all',
  ...sessionOptions,
}
```

Available relevant session/provider options in installed types:

- `graphOptimizationLevel`: already `all` in browser and offline ORT optimization is already used during export.
- `preferredOutputLocation`: useful and already partially used in benchmark.
- `enableGraphCapture`: blocked until all external tensors are GPU buffers and all compute is WebGPU.
- `freeDimensionOverrides`: useful only for symbolic/free dims; current artifacts are already concrete static shapes.
- WebGPU EP `preferredLayout: 'NCHW' | 'NHWC'`: mainly layout-sensitive ops such as conv/pool; unlikely to affect transformer reshape traffic.
- WebGPU EP `forceCpuNodeNames`: only forces more CPU placement. There is no inverse "force WebGPU or fail" option in installed types.
- WebGPU EP `validationMode`: validation/debugging only, not placement.

Browser `optimizedModelFilePath` is not useful unless ORT Web is rebuilt with optimized-model output support. The export script's offline Python ORT optimization is the right place for this repo.

## Ranked Options

### 1. Highest Likelihood: Remove GQA Repeat Materialization From Export

The largest repeated copies line up with K/V repeat expansion:

- `Reshape [36,65,2,64] -> [36,65,2,1,64]`
- `Expand [36,65,2,1,64] -> [36,65,2,4,64]`
- `Reshape [36,65,2,4,64] -> [36,65,8,64]`

This comes from `visionary/export/onnx_wrappers.py`:

- `_export_dot_product_attention()`
- current non-grouped GQA path uses `jnp.repeat(key, repeat, axis=-2)` and `jnp.repeat(value, repeat, axis=-2)`.
- `--grouped_gqa_attention` exists but is currently disabled in the manifest.

Concrete next tests:

1. Re-export with `scripts/webgpu/export_dreamer4_onnx.py --export_cached --grouped_gqa_attention --validate`.
2. Run `bun run benchmark:webgpu:profile:session-diagnostic` and compare:
   - CPU `Reshape` count.
   - `MemcpyToHost/FromHost` totals.
   - `Expand [*,2,4,64]` patterns.
3. If current grouped path still emits `broadcast_to -> Expand -> Reshape`, rewrite `_export_dot_product_attention()` to compute grouped GQA without materializing repeated K/V. Prefer equations over explicit repeat:
   - logits: grouped query `[b, kv, r, q, d]` with key `[b, kv, s, d]`.
   - output: weights `[b, kv, r, q, s]` with value `[b, kv, s, d]`.
   - return `[b, q, kv*r, d]`.

This targets the dominant provider-boundary pattern directly.

### 2. High Likelihood: Remove `candidate_cache_length` From ONNX Step Output

Trace has small but graph-capture-blocking CPU `Min` and `Cast` around cache length. WebGPU table does not list `Min`, and the code computes:

- `visionary/export/onnx_wrappers.py`: `candidate_cache_length = jnp.minimum(cache_length + 1, self.context_length).astype(jnp.int32)`

The browser already controls sequence/cache progression. Track cache length in JS and remove it from hot ONNX step outputs if possible.

Files/APIs:

- `visionary/export/onnx_wrappers.py`: `_CachedSpatioTemporalTransformer.step()`, `_CachedDynamicsModel.step()`, `sample_step()`.
- `scripts/webgpu/export_dreamer4_onnx.py`: cached output names for `candidate_cache_length`.
- `webgpu_app/bench/benchmark.js`: `cacheFromOutputs()`, `cacheOutputNames()`, fallback length handling.

Expected benefit is smaller than option 1 for raw time, but it removes a CPU op that blocks graph capture.

### 3. High Likelihood: Use GPUBuffer Feeds and Preallocated Fetches in the Benchmark

This will not fix internal CPU `Reshape`, but it removes avoidable run-boundary copies and allocation churn.

Files/APIs:

- `webgpu_app/bench/benchmark.js`
  - Replace hot CPU tensor factories where inputs are persistent or updated every run:
    - `makeFloatTensor()`
    - `makeIntTensor()`
    - `makeScalarFillTensor()`
  - Add GPU buffer helpers using:
    - `const device = ort.env.webgpu.device`
    - `device.createBuffer({ usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE, size: align16(bytes) })`
    - `device.queue.writeBuffer(buffer, 0, typedArray)`
    - `ort.Tensor.fromGpuBuffer(buffer, { dataType, dims })`
  - Use object fetches for preallocated outputs:
    - `{ [outputName]: ort.Tensor.fromGpuBuffer(buffer, { dataType, dims }) }`

Use double buffering for cache outputs because the next step feeds the previous cache while the current step writes the candidate cache.

### 4. Medium Likelihood: Try `onnxruntime-web@dev` / Newer ORT Web Against the Same Artifacts

Official docs explicitly suggest trying the nightly/dev package for latest WebGPU features. This is a low-code experiment:

- `package.json`: temporarily change `onnxruntime-web`.
- Re-run the same session diagnostic.

Success condition:

- `Reshape` is no longer assigned to `CPUExecutionProvider`, or reshape aliases no longer trigger large host/device copies.

This is worth testing because ORT maintainers describe `Reshape` as metadata-only in WebGPU/JSEP, but the installed 1.24.3 trace is not behaving that way for this graph.

### 5. Medium Likelihood: Run `--simplify_onnx` Before ORT Optimization

The export script has a built-in flag:

- `scripts/webgpu/export_dreamer4_onnx.py --simplify_onnx`

Current manifest says simplification was not run. ORT optimization already removed many shape nodes, but onnxsim may remove more shape scaffolding before ORT's pass.

Success condition:

- fewer `Reshape` nodes after optimization,
- fewer CPU provider nodes and memcpys in session profile.

Risk:

- onnxsim may not remove semantically necessary view nodes.
- It may fail on newer ops; the script already skips `RMSNormalization` cases.

### 6. Medium / Larger Change: Replace Generic ONNX Attention Decomposition With ORT WebGPU-Friendly Attention

Installed WebGPU operator table includes contrib/internal attention ops:

- `Attention`
- `MultiHeadAttention`
- `GroupQueryAttention`

Comments say some mask/past-present support is incomplete. Still, for this model's fixed cached step ABI, a custom export to `GroupQueryAttention` or `MultiHeadAttention` may avoid thousands of generic `Reshape/Expand/Einsum/Concat` nodes.

Files:

- `visionary/export/onnx_wrappers.py`
- possibly jax2onnx custom function/plugin path if directly emitting contrib ONNX nodes.

This is likely the path toward 50 ms if generic decomposition remains too fragmented.

### 7. Tactical Fallback: Prefer WASM for This Specific Step Until WebGPU Placement Is Fixed

The session diagnostic reports WASM at 93.55 ms vs WebGPU at 415.54 ms for the cached step diagnostic path. That is not the final target, but it is currently much closer.

Files:

- `webgpu_app/bench/benchmark.js`: make provider per role/session configurable.
- Keep decoder on WebGPU if it benefits; test dynamics cached step on WASM.

This is a pragmatic baseline, not the final WebGPU optimization.

## What Not To Expect

- `preferredOutputLocation` does not control internal intermediate placement. It only controls graph outputs.
- `enableGraphCapture` will not solve CPU fallback; it requires the fallback problem to already be solved.
- `freeDimensionOverrides` is unlikely to help current exported artifacts because shapes are already static.
- `preferredLayout` is unlikely to affect transformer reshape/copy traffic.
- `forceCpuNodeNames` moves nodes to CPU; it cannot force unsupported nodes onto WebGPU.

## Verification Checklist

After each experiment, use:

```bash
bun run benchmark:webgpu:profile:session-diagnostic
bun run benchmark:webgpu:profile:session-summary
```

Compare:

- `webgpu_app/bench/results/session_profile_summary.json`
- `totals.ops`:
  - `WebGpuExecutionProvider|MemcpyToHost`
  - `WebGpuExecutionProvider|MemcpyFromHost`
  - `CPUExecutionProvider|Reshape`
  - `CPUExecutionProvider|Min`
  - `CPUExecutionProvider|Cast`
- chronological trace around the largest memcpys.

Primary target before graph capture:

- eliminate or drastically reduce `MemcpyToHost/FromHost` around `CPUExecutionProvider|Reshape`.

Only after that:

- try `enableGraphCapture: true`.
- convert all hot inputs/fetches to `gpu-buffer`.
- preallocate fetch buffers.

## Sources Checked

Local repo:

- `webgpu_app/bench/results/session_profile_summary.json`
- `webgpu_app/bench/results/profile_diagnostic_latest.json`
- `webgpu_app/assets/breakout_onnx_manifest.json`
- `webgpu_app/bench/benchmark.js`
- `webgpu_app/bench/profile_diagnostic.js`
- `scripts/webgpu/export_dreamer4_onnx.py`
- `visionary/export/onnx_wrappers.py`

Local installed packages:

- `node_modules/onnxruntime-web@1.24.3`
- `node_modules/onnxruntime-web/docs/webgpu-operators.md`
- `node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/op-resolve-rules.ts`
- `node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/ops/{concat,expand,einsum,transpose}.ts`
- `node_modules/onnxruntime-web/lib/wasm/wasm-core-impl.ts`
- `node_modules/onnxruntime-common/lib/inference-session.ts`
- `.venv/lib/python3.11/site-packages/jax2onnx`

Web primary docs/source:

- ONNX Runtime WebGPU docs: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
- ONNX Runtime Web env/session options: https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html
- ONNX Runtime Web performance diagnosis: https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html
- ORT maintainer discussion on WebGPU/JSEP Reshape metadata behavior: https://github.com/microsoft/onnxruntime/discussions/15937
