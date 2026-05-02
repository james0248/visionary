# ONNX WebGPU Optimization Progress

## Goal

Make the Dreamer4 ONNX export fast enough for a live browser demo using ONNX Runtime WebGPU.

Demo benchmark contract:
- Prefill the cached dynamics model once with 64 context frames.
- Generate each new frame from the committed KV cache.
- Decode only the newly predicted frame.
- Benchmark only demo-relevant paths: cached prefill, cached step/sample frame, decoder, full streaming frame.

## Current Baseline

Current assets were generated with `--grouped_gqa_attention`.

Latest browser benchmark:
- Status: passed.
- WebGPU hardware: Apple/Metal via Chromium.
- Cached prefill: needs re-read from `webgpu_app/bench/results/latest.json`.
- Streaming frame: slower than the previous best baseline.

Known previous best:
- Fused `cached_sample_step` + `decode_z` baseline was roughly 0.86 s/frame.
- Native attention export was slower.
- Grouped GQA with 5D einsum was browser-incompatible.
- Grouped GQA lowered to matmul is browser-compatible but currently slower.

## Current Hypothesis

The bottleneck is not the CPU time of reshape itself. The major cost is data moving between WebGPU and CPU because ONNX Runtime WebGPU assigns parts of the graph to CPU. The largest suspicious paths are:
- reshape/transpose/shape plumbing around GQA repeat or grouped-GQA lowering,
- repeated layout changes in spatiotemporal attention,
- possible dynamic shape tensors that prevent ORT WebGPU from keeping a graph segment on device.

`jax2onnx` source inspection shows both `jax.lax.reshape` and `jax.numpy.reshape` lowerers try to emit a constant initializer for the reshape target when all dimensions are static. Next step is to inspect the exported ONNX graph and confirm whether our reshape targets are actually constant or dynamic.

## Iteration Log

### 2026-04-29

- Started auto loop: inspect graph/profile -> change export/model wrapper -> export ONNX -> benchmark -> analyze -> iterate.
- Inspected local `jax2onnx` reshape lowerers from `.venv`.
- Finding: `jax2onnx` has explicit logic to fold static reshape shape tensors into initializers and avoid `Shape/Gather/Concat` when the target shape is known.
- Next action: count `Reshape` nodes with constant vs dynamic second input in exported ONNX files, then decide whether to patch our export graph or change wrapper code.

### Static Reshape Check

Result: reshape target sizes are already static in the exported ONNX graphs.

Current grouped-GQA assets:
- `breakout_dynamics_cached_sample_step_b1_t1_s4.onnx`: 2064 `Reshape`, 2064 constant shape inputs, 0 dynamic shape inputs.
- `breakout_dynamics_prefill_cached_b1_t64.onnx`: 522 `Reshape`, 522 constant shape inputs, 0 dynamic shape inputs.
- `breakout_dynamics_step_cached_b1_t1.onnx`: 538 `Reshape`, 538 constant shape inputs, 0 dynamic shape inputs.
- tokenizer decoder/decode graphs: all reshape targets are constant.

Conclusion: the CPU reshape issue is not caused by variable reshape sizes from `jax2onnx`. ORT WebGPU is assigning these reshape/layout operations to CPU despite constant target shapes, or CPU fallback is triggered by neighboring unsupported layout/broadcast patterns.

Next action: re-export the default non-grouped attention path and benchmark it against the grouped-GQA matmul path. If default is faster, keep grouped GQA experimental and optimize a different axis.

### Default Attention Re-Benchmark

Re-exported without `--grouped_gqa_attention`.

Result:
- Cached prefill: 1419.08 ms.
- Cached sample step / dynamics frame: 812.41 ms mean, 806.73 ms median, 886.99 ms p95.
- Decoder frame: 109.79 ms mean.
- Streaming frame: 922.30 ms mean, 911.54 ms median, 1004.51 ms p95.

Conclusion: default patched ONNX attention is faster than grouped-GQA matmul. Keep grouped-GQA experimental; do not use it as the default demo artifact.

### Fixed Full Cache Attempt

Change tried:
- Added a fused sample-step export path that assumes the browser demo always has a full 64-frame cache after prefill.
- Removed runtime `position_index`, `cache_length`, and `candidate_cache_length` from the fused sample-step ONNX graph.

Result:
- Cached sample step / dynamics frame: 967.78 ms mean.
- Decoder frame: 121.52 ms mean.
- Streaming frame: 1089.40 ms mean.
- Graph delta was small: `Reshape` 1511 -> 1508, total nodes 7290 -> 7274.

Conclusion: this made the demo slower. Runtime scalar cache inputs are not the bottleneck. Reverted the export script to the original faster cached sample-step contract.

Next action:
- Restore the faster default artifacts.
- Investigate larger structural changes: full-cache output/write cost, GQA repeat/expand transfer cost, and ORT WebGPU support for graph capture or preallocated outputs.

### Restored Default Export

Re-exported the original cached sample-step contract after reverting the fixed-full-cache export path.

Latest result:
- Cached prefill: 1257.76 ms.
- Cached sample step / dynamics frame: 782.11 ms mean, 768.49 ms median, 870.02 ms p95.
- Decoder frame: 102.76 ms mean.
- Streaming frame: 884.96 ms mean, 866.58 ms median, 984.44 ms p95.

Current best result in this loop: 884.96 ms/frame.

Graph:
- Inputs: `z`, `actions`, `position_index`, `k_cache`, `v_cache`, `cache_length`.
- Outputs: `final_z`, `pred_z`, `candidate_k_cache`, `candidate_v_cache`, `candidate_cache_length`.
- Nodes: 7290 total, 1511 `Reshape`, 48 `Transpose`, 192 `Expand`, 192 `Einsum`, 258 `Concat`.

Next action:
- Investigate float16 conversion/export because the browser GPU reports `shader-f16` support and current artifacts are fp32.

### Float16 Conversion Attempt

Change tried:
- Added dependency: `onnxconverter-common==1.16.0` via `uv add --group onnx onnxconverter-common`.
- Added exporter flag: `--float16`.
- Conversion uses `convert_float_to_float16(..., keep_io_types=True, disable_shape_infer=True)` so JS inputs/outputs can remain float32.

Result:
- Export/conversion started, but ORT CPU validation rejected `breakout_dynamics_step_cached_b1_t1.onnx`.
- Error: output of an inserted Cast node had `tensor(float16)` where ONNX Runtime expected `tensor(float)`.

Conclusion:
- Naive full-graph fp16 conversion is currently blocked. It needs either shape/type inference enabled, a block list for problematic nodes, or a custom mixed-precision pass.
- Restore fp32 artifacts before continuing.

### Reshape Elimination And Current Bottleneck

Implemented post-export rewrites:
- singleton `Reshape` replacement,
- GQA repeat materialization replacement,
- head projection `Gemm -> Reshape` and `Reshape -> Gemm` replacement with rank-aware `Einsum`,
- full-cache fused sample-step export for the demo hot path.

Current graph counts after full-cache export:
- `breakout_dynamics_cached_sample_step_b1_t1_s4`: `Reshape=0`, `Einsum=576`, `Unsqueeze=756`, `Gemm=296`.
- `breakout_tokenizer_decode_z_b1_t1`: `Reshape=1`, `RMSNormalization=32`.

Latest browser benchmark:
- Dynamics frame: 63.21 ms mean, 61.70 ms median, 70.79 ms p95.
- Decoder frame: 57.26 ms mean, 56.44 ms median, 64.60 ms p95.
- Streaming frame: 120.57 ms mean, 118.01 ms median, 132.70 ms p95.

Session profiling:
- Fused sample-step graph has essentially no CPU transfer issue: 6910 WebGPU nodes, 1 CPU node, 0.54 ms CPU.
- Decoder graph still has transfer bottleneck: `RMSNormalization` executes on CPU 32 times, causing 32 `MemcpyToHost` events totaling ~110 ms in the session profile.

Next action:
- Replace tokenizer decoder's original `SpatioTemporalTransformer` during ONNX export with an export-only transformer using decomposed `_ExportRMSNorm`, preserving parameter scopes and model behavior.

### 2026-04-29 Hot Path Status

Changes landed since the previous note:
- Tokenizer decoder export now uses the export-only transformer/RMSNorm path, then fuses decomposed RMSNorm into `SimplifiedLayerNormalization`.
- `apply_tokenizer_decode_z` packs `[1,1,32,32] -> [1,1,64,16]` without ONNX `Reshape`.
- Fused sample-step export now removes the remaining int32-to-int64 `Cast` before `Gather`.
- Export script accepts `--sample_steps` values other than 4 for demo-performance experiments.
- Benchmark result metadata now reports `sample_steps` from the manifest.

Validation:
- `--validate` passed for the current `sample_steps=1` export.
- Current sample-step artifact validation: `final_z` max abs error `~2.6e-6` versus the Flax/JAX inference function for the same sample-step count.

Rejected experiments:
- `--grouped_gqa_attention`: validates but is much slower; it reintroduces many `Reshape` nodes and produced ~696 ms streaming frames.
- `--float16`: ORT CPU validation currently fails on a fp16/fp32 type mismatch in the cached step graph.

Current benchmark, short run:
- `sample_steps=1`
- Dynamics frame: 33.94 ms mean, 35.56 ms median, 42.15 ms p95.
- Decoder frame: 6.16 ms mean, 5.62 ms median, 8.67 ms p95.
- Streaming frame: 40.19 ms mean, 41.25 ms median, 47.68 ms p95.

Current benchmark, longer 6 warmup / 24 timed run:
- Dynamics frame: 34.84 ms mean, 35.52 ms median, 38.10 ms p95.
- Decoder frame: 5.50 ms mean, 5.50 ms median, 5.80 ms p95.
- Streaming frame: 40.40 ms mean, 41.11 ms median, 43.78 ms p95.

Conclusion:
- The 50 ms target is reached only by reducing the demo flow sample count to 1. The behavior-preserving `sample_steps=4` path remains about 89-90 ms/frame after RMSNorm and CPU-transfer fixes.
- The remaining behavior-preserving path to <50 ms is still true fused attention / cache ABI work, not more CPU transfer cleanup.

### 2026-04-29 Sample-Steps Requirement And WASM Baseline

User constraint update:
- `sample_steps` must stay at 4 for the demo benchmark.
- The previous `sample_steps=1` and `sample_steps=2` speed trials are useful diagnostics, but are not acceptable demo configurations.

Export restored:
- Command used `--sample_steps 4 --export_cached --validate --overwrite`.
- `breakout_dynamics_cached_sample_step_b1_t1_s4` validation passed.

WASM CPU-only browser benchmark:
- Command: `WEBGPU_BENCHMARK_PROVIDER=wasm WEBGPU_BENCHMARK_WARMUP_RUNS=1 WEBGPU_BENCHMARK_TIMED_RUNS=8 bun run benchmark:webgpu`.
- Provider: ONNX Runtime Web `wasm`.
- Sampling: `sample_steps=4`, `generated_frames=8`.
- Prefill: 798.43 ms.
- Dynamics frame: 97.58 ms mean, 97.81 ms median, 99.20 ms p95.
- Decoder frame: 37.18 ms mean, 35.53 ms median, 44.18 ms p95.
- Streaming frame: 134.83 ms mean, 133.35 ms median, 143.43 ms p95, 7.42 FPS.

Interpretation:
- WASM is slower than WebGPU for the `sample_steps=4` demo path, as expected.
- The CPU-only result is still a useful fallback baseline, but it is not viable for the target interactive speed.

### 2026-04-30 ONNX Simplifier Attempt

Finding:
- `onnxsim` cannot run on already-fused artifacts because it does not recognize `SimplifiedLayerNormalization` at opset 23.
- The viable order is fresh export -> `onnxsim` -> ORT optimization/custom rewrites -> RMSNorm fusion.

Fresh export:
- Command used `--sample_steps 4 --export_cached --validate --overwrite --simplify_onnx`.
- Validation passed.
- `onnxsim` ran on all hot-path artifacts except the uncached dynamics model, which still has raw `RMSNormalization` and is not used by the demo benchmark.

Simplification effect:
- `breakout_dynamics_cached_sample_step_b1_t1_s4`: raw graph `11741 -> 7649` nodes by `onnxsim`, then final rewritten artifact has 4840 nodes, 0 `Reshape`, 384 `SimplifiedLayerNormalization`.
- `breakout_tokenizer_decode_z_b1_t1`: raw graph `1005 -> 629` nodes by `onnxsim`, then final rewritten artifact has 397 nodes, 0 `Reshape`, 32 `SimplifiedLayerNormalization`.

Benchmark after `onnxsim`:
- Provider: WebGPU.
- Sampling: `sample_steps=4`, `generated_frames=8`.
- Dynamics frame: 82.39 ms mean, 85.11 ms median, 85.60 ms p95.
- Decoder frame: 5.91 ms mean, 5.82 ms median, 6.43 ms p95.
- Streaming frame: 88.40 ms mean, 90.85 ms median, 91.93 ms p95, 11.31 FPS.

Conclusion:
- `onnxsim` now works in the fresh-export pipeline and should stay available as `--simplify_onnx`.
- It reduces graph size materially, but does not change the main latency class; `sample_steps=4` remains around 88-91 ms/frame.

### 2026-04-30 BF16 / GQA / Softmax Notes

Softmax precision:
- The export-only attention wrapper casts attention logits to `float32` before `jax.nn.softmax`, then casts the attention weights back to the value dtype. This keeps the numerically sensitive softmax accumulation in fp32 even if a future mixed-precision path is used.

### 2026-04-30 FP16 And Static Slide Rewrite Iteration

Goal:
- Keep `sample_steps=4`.
- Try to reduce the browser demo `streaming_frame` latency toward 50 ms.

Trials:
- All-FP16 export with Softmax blocked to fp32 and QuickGelu decomposed:
  - Export and validation passed.
  - Browser ran, but did not improve latency: `streaming_frame` was about 107 ms.
  - Main issue: dynamics stayed around 103 ms; decoder is only about 4 ms, so FP16 decoder wins do not move the total.
- Decoder-only FP16 with public fp32 IO:
  - Export and validation passed.
  - Browser ran, but was slower: about 111 ms without simplification and about 125 ms when combined with the demo-only simplifier.
  - Conclusion: decoder precision is not the bottleneck.
- Demo-only `onnxsim`:
  - Added `--simplify_demo_only` so simplification skips full-window artifacts outside the benchmark hot path.
  - This avoids the earlier all-artifact simplifier instability and keeps simplification focused on prefill, fused step, and single-frame decoder artifacts.
- Static steady-state slide rewrite:
  - Added `rewrite_slide_static_cache_ops_for_webgpu`.
  - Rewrites the steady-state slide graph under the assumption that the benchmark hot path runs with a full cache length of 64.
  - Replaced two `Min(cache_length, 64)` clamps with static 64 initializers.
  - Replaced one final cache-projection `Reshape([36,128] -> [1,36,1,2,64])` with `Unsqueeze + Split + Unsqueeze + Concat`.
  - Moved one bool mask cast before `Unsqueeze` to avoid a bool-layout fallback.
  - Validation passed for the hot graph; max errors stayed in the same tiny range as the original fp32 graph.

### 2026-04-30 FP16 QuickGelu Precision Trial

Trial:
- Kept `QuickGelu` in fp32 for dynamics graphs while keeping tokenizer decoder `QuickGelu` decomposed in fp16.
- Export command used `--sample_steps 4 --export_cached --validate --overwrite --simplify_onnx --simplify_demo_only --float16`.
- Validation passed and browser execution passed.

Benchmark:
- Long graph-capture run, 6 warmup / 24 timed samples.
- `cached_step` median: ~81.61 ms, p95: ~83.64 ms.
- `decoder_frame` median: ~11.34 ms, p95: ~13.00 ms.
- `streaming_frame` median: ~96.29 ms, p95: ~97.82 ms.

Conclusion:
- This was slower than the previous stable full-fp16/decomposed-QuickGelu graph-capture result, which had a streaming median around 93-94 ms.
- Do not keep this as the default precision strategy.

### Artifact Tracking Policy

Generated ONNX assets and benchmark JSON outputs are not suitable for normal commits.

Current policy:
- Ignore `webgpu_app/assets/*.onnx`.
- Ignore `webgpu_app/assets/*.onnx.data`.
- Ignore `webgpu_app/assets/*.pre_*.onnx`.
- Ignore `webgpu_app/bench/results/*.json`.
- Keep source changes, benchmark tooling, manifests generated by the export command, and result summaries in `.codex` as the reviewable history.

### 2026-04-30 QuickGelu And Step-Artifact Trials

QuickGelu fusion trial:
- Added experimental `--keep_quickgelu` to skip the fp16 QuickGelu decomposition.
- CPU ORT validation passed.
- Browser WebGPU failed immediately:
  - `Failed to create a WebGPU compute pipeline`
  - invalid shader module: `QuickGelu`
- Conclusion: fp16 QuickGelu must remain decomposed for ORT WebGPU 1.24.3 on this Chromium/Apple Metal path.

Benchmark override:
- Added `WEBGPU_BENCHMARK_STEP_ARTIFACT` so the benchmark can select a specific dynamics step export without editing the fallback list.

Legacy no-append-context comparison:
- Artifact: `breakout_dynamics_cached_sample_step_slide_b1_t1_s4`.
- Long graph-capture run, 6 warmup / 24 timed samples.
- `dynamics_frame` median: ~78.09 ms, p95: ~80.01 ms.
- `decoder_frame` median: ~5.04 ms, p95: ~5.25 ms.
- `streaming_frame` median: ~83.96 ms, p95: ~86.07 ms.
- This is faster than the behavior-preserving append-context artifact (~94 ms median) because it avoids the extra context-cache update pass.
- It is not equivalent to the current demo semantics, because the committed cache no longer comes from the noised context latent used by `sample_step_append_context`.

Legacy no-append-context profile:
- `breakout_dynamics_cached_sample_step_slide_b1_t1_s4` profile has 5209 WebGPU node events.
- Top costs:
  - `Mul`: 1085 events, ~27.25 ms profile total.
  - `Einsum`: 576 events, ~21.25 ms.
  - `Unsqueeze`: 754 events, ~12.14 ms.
  - `SimplifiedLayerNormalization`: 384 events, ~9.99 ms.
  - `Gemm`: 296 events, ~8.88 ms.
  - cache/layout update ops (`Concat`, `Squeeze`, `Gather`, `Split`) together remain meaningful but are not enough alone to reach 50 ms.
- Interpretation: after removing the fifth append-context pass, the core four denoise passes still dominate. Reaching 50 ms without changing `sample_steps=4` likely needs either fewer kernels per transformer pass or a cache ABI that avoids full-cache outputs plus custom GPU cache update.
  - Final hot graph has 0 `Min` and 0 `Reshape`.

Benchmark after static slide rewrite:
- Provider: WebGPU.
- Dynamics frame: 99.97 ms mean, 101.34 ms median, 102.91 ms p95.
- Decoder frame: 5.53 ms mean, 5.45 ms median, 5.88 ms p95.
- Streaming frame: 105.61 ms mean, 106.85 ms median, 108.44 ms p95.

Graph capture:
- After removing CPU fallback nodes, graph capture got past the provider-partition error.
- It then failed with `Buffer is not registered`.
- Current interpretation: ORT WebGPU graph capture needs stable registered GPU buffers across runs. The streaming cache loop currently receives new GPU output buffers for the cache each step and feeds them into the next step, which does not satisfy graph-capture's buffer registration constraints.

Full-FP16 with the static slide rewrite:
- Failed CPU validation before producing active artifacts.
- Failure was a type mismatch around a rewritten mask `Unsqueeze` after fp16 conversion.
- Active artifacts were restored to the validated fp32 static-slide export.

Current best stable artifact:
- Validated fp32 static-slide export with demo-only simplification.
- Current browser benchmark: `streaming_frame` about 105.6 ms.
- The remaining bottleneck is not decoder precision; it is the fused dynamics step doing 4 internal denoise/sample passes and writing the updated cache.

BF16 status:
- A direct bf16 graph conversion is not currently usable in ORT WebGPU for this model. The browser rejected bf16 inputs to `Einsum`, so enabling bf16 globally breaks the hot dynamics graph instead of speeding it up.
- Keep the valid demo artifacts in fp32 until a selective mixed-precision pass can prove that the specific lowered WebGPU ops accept bf16/fp16.

GQA fusion attempt:
- Implemented a gated post-export `GroupQueryAttention` fusion matcher for cached temporal attention.
- First packed-KV attempt was invalid because ORT WebGPU requires current query/key sequence lengths to match unless past K/V inputs are supplied separately.
- Corrected attempt passed `past_key`/`past_value` separately and passed the demo smoke test, but benchmark regressed badly.

Benchmark with corrected GQA fusion:
- Dynamics frame: ~191.20 ms mean, ~163.99 ms median, ~305.49 ms p95.
- Decoder frame: ~3.90 ms mean.
- Streaming frame: ~195.22 ms mean, ~167.88 ms median, ~310.06 ms p95.

Restored baseline after GQA regression:
- Dynamics frame: 112.13 ms mean, 111.80 ms median, 118.49 ms p95.
- Decoder frame: 4.14 ms mean, 4.17 ms median, 4.31 ms p95.
- Streaming frame: 116.37 ms mean, 115.98 ms median, 122.70 ms p95.

Conclusion:
- ORT WebGPU `GroupQueryAttention` is not a win for this graph as currently wired; the required reshapes/transposes and implementation overhead dominate.
- The valid hot artifact is restored to the faster non-GQA baseline. The GQA rewrite remains gated/off for future experiments.

### 2026-04-30 BF16 References And Mixed Precision

Target:
- Keep `sample_steps=4`.
- Keep attention softmax numerically stable by accumulating softmax in fp32.
- Reach 20 FPS, meaning `streaming_frame <= 50 ms`.

References checked:
- ONNX Runtime WebGPU docs recommend static shapes, graph capture when all kernels can stay on WebGPU, and IO binding / GPU-resident tensors for repeated transformer runs: https://fs-eire.github.io/onnxruntime/docs/tutorials/web/ep-webgpu.html
- Microsoft's ONNX Runtime WebGPU announcement calls out mixed precision as FP16, specifically browser `shader-f16` support. It does not describe BF16 support: https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/
- ORT issue #13001 is the broad BF16 support tracker. It is not WebGPU-specific and does not establish ORT WebGPU BF16 kernel support: https://github.com/microsoft/onnxruntime/issues/13001

Local ORT Web evidence:
- `node_modules/onnxruntime-web/lib/wasm/wasm-common.ts` defines `DataType.bfloat16 = 16`, but `calculateTensorSizeInBytes()` maps it to `-1`, meaning unsupported storage sizing.
- `tensorDataTypeStringToEnum()` and `tensorDataTypeEnumToString()` do not expose a `'bfloat16'` JS tensor type.
- WebGPU shader helper maps only `u32`, `f16`, `f32`, and `i32` to ORT data types.
- `backend-webgpu.ts` requests the browser `shader-f16` feature when available, not any BF16 feature.
- The previous bf16 graph conversion failed in-browser because `Einsum` rejected `tensor(bfloat16)`.

Conclusion:
- BF16 is an ONNX type and appears in the generic ORT enum, but the installed ORT WebGPU runtime does not implement it as a usable tensor/shader type for this model.
- Implementing true BF16 in ORT WebGPU would require runtime work: JS tensor type exposure, byte-size handling, WGSL storage/load/store representation, per-op kernel support or conversion shims, and type resolver updates.
- The practical mixed-precision path for this installed runtime is FP16, with fp32 softmax islands.

Selective FP16 attempt:
- Converted only the hot steady-state dynamics step graph to fp16 with `keep_io_types=True`.
- Blocked `Softmax` and `SimplifiedLayerNormalization` so softmax and norm islands stay fp32.
- Result: browser demo smoke hung after session creation / execution start and did not complete.
- Restored the valid fp32 hot artifact: sha256 `336d124912fda24233b39b7e2406e085961b65dad1e1f69885891679609203ed`.

Decision:
- Do not keep the selective fp16 artifact active.
- Continue with fp32 artifact while looking for structural graph/kernel-count reductions rather than unsupported bf16 or unstable fp16 conversion.

### 2026-04-30 Rank Cache Rewrite And Split-Cache Attempt

Hot graph profile:
- Ran session profiling against the actual steady-state graph `breakout_dynamics_sample_append_context_slide_b1_t1_s4`.
- Profile showed 6068 WebGPU node events and 1 CPU node.
- The lone CPU node was `node_Reshape_15414`, reshaping `[36,128] -> [1,36,1,2,64]` during cache update.
- This CPU node blocked WebGPU graph capture.

Rank-cache rewrite:
- Replaced `node_Gemm_15305 + node_Reshape_15414` with a rank-aware `Einsum + Unsqueeze`.
- Resulting hot graph has 0 `Reshape` ops.
- Demo smoke passed.
- Graph capture session creation now succeeds.

Benchmark after rank-cache rewrite, graph capture off:
- Dynamics frame: 100.15 ms mean, 102.10 ms median, 103.71 ms p95.
- Decoder frame: 5.52 ms mean.
- Streaming frame: 105.77 ms mean, 107.64 ms median, 109.45 ms p95.

Benchmark after rank-cache rewrite, graph capture on:
- Graph capture enabled successfully for the cached step session.
- Timing was unstable: first several samples reported ~1.4 ms but later samples settled around 95-102 ms.
- Streaming median stayed around ~101 ms, so graph capture is not enough by itself.

Split-cache ABI attempt:
- Tried graph surgery to expose per-layer cache inputs/outputs (`k_cache_0..5`, `v_cache_0..5`) to remove layer-axis `Slice` and final stack `Concat`.
- Benchmark harness was updated to understand both stacked and per-layer cache names.
- ORT shape inference rejected the surgically exposed internal outputs due rank mismatches (`inferred=4 declared=5`, then `inferred=6 declared=5`).
- Restored valid stacked-cache artifacts.

Conclusion:
- The rank-cache rewrite is kept; it is valid and improves the hot graph by roughly 10 ms/frame.
- Split-cache remains promising but should be done from export code or a more careful graph rewrite with fresh value names and `Identity` outputs, not by renaming tensors in place.
- Current valid performance is still ~106 ms/frame, above the 50 ms target.

### 2026-04-30 Graph Capture Fixed IO Attempt

Problem:
- ORT WebGPU graph capture failed with `Buffer is not registered`.
- Local ORT source showed graph-capture runs register/unregister each GPU input/output buffer.
- The benchmark was feeding the same GPU buffer as both `sample_noise` and `context_noise`, so ORT tried to unregister the same buffer twice.

Fix:
- Added separate fixed GPU buffers for `sample_noise` and `context_noise`.
- Added fixed GPU inputs for graph-capture scalar inputs (`cache_length`, `position_index`).
- Added fixed GPU cache buffers for `k_cache` and `v_cache`, with GPU-to-GPU copies from candidate cache outputs into the fixed buffers before the next captured run.
- Avoided fetching `candidate_cache_length` in graph-capture mode and retained the known steady-state committed length.

Result:
- Graph capture benchmark now runs and passes.
- Short 8-frame smoke can report low mean because early captured runs return before queued GPU work fully surfaces:
  - Streaming mean: ~19.75 ms
  - Median: ~2.10 ms
  - Last frames: ~46 ms and ~100 ms
- Longer run with 6 warmup / 24 timed frames shows true steady state still around the old cost:
  - Dynamics mean: ~68.96 ms, median ~93.68 ms, p95 ~97.69 ms
  - Streaming mean: ~74.13 ms, median ~100.58 ms, p95 ~103.92 ms

Conclusion:
- Graph capture now works mechanically, but it does not reduce the real steady-state frame latency on this model.
- The remaining 50 ms target requires reducing hot graph compute/dispatch cost, most likely via working FP16 or structural graph reduction.

### 2026-04-30 FP16 And Export-Level Layer Cache

FP16 conversion:
- Full FP16 export now validates after three fixes:
  - Keep `Softmax` in FP32 for attention stability.
  - Decompose FP16 `QuickGelu` to `Mul/Sigmoid/Mul` because ORT WebGPU 1.24.3 fails to compile the fused FP16 QuickGelu shader.
  - Strip stale intermediate value_info and repair FP16 binary cast mismatches after conversion.
- Stable monolithic-cache browser result with graph capture remains roughly:
  - Dynamics median: ~89 ms
  - Decoder median: ~5 ms
  - Streaming median: ~95 ms

Export-level layer-cache attempt:
- Added a real per-layer cache export path instead of graph surgery:
  - Prefill output: `k_cache_0..5`, `v_cache_0..5`
  - Step input/output: `k_cache_0..5`, `v_cache_0..5`, `candidate_k_cache_0..5`, `candidate_v_cache_0..5`
  - Layer cache shape: `[1, 36, 64, 2, 64]`
- Numerical validation passed for the new layer-cache artifacts.
- Browser benchmark selected:
  - `breakout_dynamics_prefill_layer_cached_b1_t64`
  - `breakout_dynamics_sample_append_context_slide_layer_b1_t1_s4`

Layer-cache result:
- No graph capture, 8 timed frames:
  - Dynamics median: ~95.66 ms
  - Streaming median: ~100.97 ms
- Graph capture, 6 warmup / 24 timed:
  - Dynamics median: ~84.46 ms, p95 ~89.15 ms
  - Streaming median: ~97.18 ms, p95 ~105.70 ms

Conclusion:
- The layer-cache ABI removes some internal slice/stack work, but the many cache inputs/outputs add enough ORT/browser overhead that end-to-end streaming is not better.
- Keep the layer-cache artifacts as experimental exports, but default the benchmark/demo back to the monolithic steady-state graph for now.
- Generated ONNX files are too large/noisy for normal commits; `.gitignore` now ignores generated ONNX assets and benchmark JSONs. Already tracked ONNX files must be excluded by explicit path staging or moved to a large-artifact mechanism later.

### 2026-05-02 Entry-Cache Artifact And Browser-Side Cache Update

Goal:
- Avoid returning/copying the full `[6,1,36,64,2,64]` K/V cache every generated frame.
- Export a steady-state artifact that returns only the new per-frame K/V entries and update the persistent full cache in-place in WebGPU.

Implementation:
- Added `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4`.
- Entry graph inputs: `sample_noise, context_noise, actions, k_cache, v_cache`.
- Entry graph outputs: `final_z, pred_z, candidate_k_entry, candidate_v_entry`.
- Added browser WebGPU compute shader for in-place cache slide:
  - V cache: shift left and write new entry at slot 63.
  - K cache: shift left, apply one-step RoPE rebase, and write new entry at slot 63.
- Added `scripts/webgpu/verify_entry_cache_update.py` to compare:
  - full-cache artifact output
  - entry artifact output + the same slide/rebase update in NumPy

Accuracy:
- Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
- Entry-cache reconstruction vs full-cache artifact passed:
  - `final_z`: exact
  - `pred_z`: exact
  - `candidate_v_cache_from_entry`: exact
  - `candidate_k_cache_from_entry`: max abs error `2.3841858e-7`

Graph cleanup:
- First entry export still had one final `Reshape` on `candidate_v_entry`, shape `[36,128] -> [1,1,36,1,2,64]`.
- Extended `rewrite_head_projection_reshapes_for_webgpu()` to rank-6 head outputs.
- Regenerated graph has:
  - `Reshape=0`
  - `Less=0`
  - `Cast=0`

Benchmark, normal mode:
- Selected step artifact: `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4`.
- Dynamics mean/median/p95: `105.29 / 102.55 / 116.26 ms`.
- Streaming mean/median/p95: `109.26 / 106.42 / 120.36 ms`.
- Result: not faster than the full-cache specialized graph.

Benchmark, graph capture, 64 timed frames:
- Graph capture now succeeds after removing the final `Reshape`.
- Dynamics mean/median/p95: `80.19 / 89.36 / 90.27 ms`.
- Streaming mean/median/p95: `86.73 / 96.54 / 97.59 ms`.
- Result: essentially unchanged from full-cache graph capture. Full-cache output/copy was not the dominant bottleneck.

Conclusion:
- Entry-cache artifact is numerically valid and graph-capture compatible.
- It should be kept as a useful runtime ABI, but it does not move us toward the 50 ms target by itself.
- Remaining bottleneck is dynamics graph compute/dispatch: hundreds of `Einsum`, `SimplifiedLayerNormalization`, `Softmax`, and pointwise ops inside the fused sample+append graph.

### 2026-05-03 Export Cleanup And Skip-SimplifiedLayerNorm Fusion

Cleanup:
- Removed failed packed QKV / packed SwiGLU export branches from `visionary/export/onnx_wrappers.py` and the corresponding exporter flags.
- Removed the stale post-export `com.microsoft::GroupQueryAttention` fusion path from `scripts/webgpu/export_dreamer4_onnx.py`; local ORT WebGPU support is for contrib attention ops only, and the GQA trial was accurate but slower.
- Kept the live fp32 entry-cache demo path unchanged.

Reproducible optimization:
- Added `fuse_skip_simplified_layer_norm_for_webgpu()` to the exporter.
- It fuses residual `Add` followed by `SimplifiedLayerNormalization` into ORT WebGPU's `SkipSimplifiedLayerNormalization`.
- Hot entry graph after regeneration:
  - nodes: `5771`
  - `SkipSimplifiedLayerNormalization`: `179`
  - `SimplifiedLayerNormalization`: `299`
  - `Add`: `304`
  - `Reshape`: `0`

Accuracy:
- Full raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
- Entry-cache reconstruction still passed at `atol=5e-4`, `rtol=5e-4`.
- Hot artifact max abs errors:
  - `final_z`: `1.4305e-6`
  - `pred_z`: `1.4305e-6`
  - `candidate_k_entry`: `7.6294e-6`
  - `candidate_v_entry`: `4.8280e-6`

Benchmark:
- Normal browser mode:
  - Dynamics median: `96.34 ms`
  - Streaming median: `102.31 ms`
- Graph capture, 64 timed frames:
  - Dynamics median: `84.17 ms`, p95 `84.67 ms`
  - Streaming median: `94.88 ms`, p95 `95.50 ms`

Conclusion:
- The skip-layernorm fusion is valid and reproducible, and graph-capture dynamics improved from roughly `88-89 ms` to roughly `84 ms`.
- Overall streaming remains about `95 ms` because decoder time is still around `10 ms` in this graph-capture run.
- This is useful cleanup/optimization, but still not enough for the 50 ms target.

### 2026-05-03 Attention And Projection Rewrite Trials

Failed attention lowering:
- Changed export-only attention from `Einsum -> Softmax -> Einsum` to local `Transpose -> MatMul -> Softmax -> MatMul -> Transpose`.
- Numerical export/validation passed, but browser latency regressed badly.
- Normal WebGPU benchmark:
  - Dynamics median: `299.89 ms`
  - Decoder median: `31.84 ms`
  - Streaming median: `330.21 ms`
- Conclusion: local attention matmul lowering adds too much layout/kernel overhead in the current BSHD graph. Reverted the default export wrapper to the previous `Einsum` attention.

RotaryEmbedding rewrite:
- Added an opt-in `--rotary_embedding_rewrite` flag.
- The rewrite fuses non-interleaved RoPE `Split/Mul/Sub/Add/Concat` islands into ORT's `com.microsoft::RotaryEmbedding`.
- CPU parity was acceptable in isolated checks, but browser latency was not better because the fused op needs BSHD/BHSD transposes around most current RoPE sites.
- Default remains disabled.

Projection layout rewrite:
- Added `--head_projection_rewrite {einsum,layout}`.
- `einsum` is the previous default: removes head projection `Reshape` by replacing `Gemm + Reshape` / `Reshape + Gemm` with rank-aware `Einsum`.
- `layout` is the new experimental path: keeps original `Gemm` kernels and replaces only the head-view `Reshape` nodes with static `Split/Squeeze/Unsqueeze/Concat`.

Accuracy for `--head_projection_rewrite layout`:
- Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
- Entry-cache reconstruction passed at `atol=5e-4`, `rtol=5e-4`.
- Hot entry artifact max abs errors:
  - `final_z`: `1.4305e-6`
  - `pred_z`: `1.4305e-6`
  - `candidate_k_entry`: `7.6294e-6`
  - `candidate_v_entry`: `4.8280e-6`

Graph for `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4` with layout rewrite:
- nodes: `8752`
- `Reshape`: `0`
- `Einsum`: `238`
- `Gemm`: `844`
- `Split`: `717`
- `Unsqueeze`: `1988`
- `Squeeze`: `1403`
- `Concat`: `782`

Benchmark for `--head_projection_rewrite layout`:
- Normal WebGPU:
  - Prefill: `721.71 ms`
  - Dynamics mean/median/p95: `72.93 / 67.30 / 97.28 ms`
  - Decoder mean/median/p95: `4.85 / 4.91 / 5.14 ms`
  - Streaming mean/median/p95: `77.90 / 72.14 / 102.61 ms`
- Graph capture, 64 timed frames:
  - Prefill: `714.23 ms`
  - Dynamics mean/median/p95: `53.32 / 58.51 / 59.23 ms`
  - Decoder mean/median/p95: `3.73 / 4.00 / 4.65 ms`
  - Streaming mean/median/p95: `57.42 / 63.03 / 63.75 ms`

Conclusion:
- The layout projection rewrite is the best behavior-preserving result so far.
- It improves graph-capture streaming median from the previous best `~94.88 ms` to `~63.03 ms`.
- It is still above the `50 ms` target. The remaining hot cost is dynamics: four denoise passes still take about `58.5 ms` median before the decoder.
- The next useful work is not more standalone reshape cleanup; it should reduce whole attention/projection dispatch groups, likely by changing the export/cache ABI to a layout that lets ORT WebGPU consume fused attention or fewer layout adapters.
