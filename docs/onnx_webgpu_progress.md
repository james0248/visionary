# ONNX WebGPU Optimization Progress

## Goal

Make the Dreamer4 ONNX export fast enough for a live browser demo using ONNX Runtime WebGPU.

Demo benchmark contract:
- Prefill the cached dynamics model once with 64 context frames.
- Generate each new frame from the committed KV cache.
- Decode only the newly predicted frame.
- Benchmark only demo-relevant paths: cached prefill, cached step/sample frame, decoder, full streaming frame.

## Current State

The branch now keeps the fp32 WebGPU path as the maintained demo/export target. The current demo
uses:
- `sample_steps=2`.
- Curated Breakout assets under `webgpu_app/dream_arcade_assets/breakout`.
- A cache-length entry dynamics step graph while filling the initial short cache:
  `breakout_dynamics_sample_append_context_cache_length_entry_b1_t1_s2.onnx`.
- A packed and partial-head-split-rewritten full-cache entry dynamics step graph after the logical
  cache reaches 64:
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2.onnx`.
- A packed and partial-head-split-rewritten single-frame tokenizer decode graph:
  `breakout_tokenizer_decode_z_b1_t1.onnx`.
- Offline context/cache artifacts generated from the first Breakout episode frames.

The maintained benchmark surface is latency plus graph capture:
- `bun run benchmark:webgpu` runs the browser streaming benchmark and graph-capture check.
- `bun run benchmark:webgpu:smoke` runs the smoke subset.
- Benchmark controls should be passed as wrapper flags after `--`, for example
  `--webgpu-benchmark-asset-base` or `--webgpu-benchmark-timed-runs`, instead of leading shell
  environment assignments.
- Generated results stay under `webgpu_app/bench/results/` and should not be committed.

Rejected or inactive paths:
- `--grouped_gqa_attention` validated in some forms but was slower or browser-incompatible.
- Native ONNX/ORT attention fusion did not produce a better WebGPU artifact for this model.
- fp16/bf16 and int quantization experiments either regressed speed, failed validation, or produced
  unstable outputs. The stable branch target is fp32.
- ORT WebGPU profiling callbacks/session profiling did not provide reliable actionable attribution,
  so the maintained workflow keeps timing and graph capture only.

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
- Keep source changes, benchmark tooling, manifests generated by the export command, and result summaries in `docs/` as the reviewable history.

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

### 2026-05-03 Packed Projection Trial For Sub-25 Push

New target:
- Keep `sample_steps=4`.
- Keep numerical parity against raw/unoptimized ONNX artifacts.
- Push graph-capture streaming below `25 ms` if possible.

Native attention lowering trial:
- Exported with `--attention_lowering native` after patching the export override so jax2onnx could use its native `dot_product_attention` lowering.
- Raw-vs-optimized validation and entry-cache reconstruction passed.
- Browser result was not viable:
  - Dynamics mean/median: `628.00 / 664.00 ms`
  - Decoder mean: `52.00 ms`
  - Streaming mean/median: `680.00 / 706.00 ms`
- Conclusion: jax2onnx native attention does not map to a fast WebGPU graph for this model. Keep manual attention lowering.

Restored best manual path:
- Exported with `--simplify_onnx --simplify_demo_only --head_projection_rewrite layout --rotary_embedding_rewrite --attention_lowering manual`.
- Validation passed:
  - hot `final_z`/`pred_z` max abs: `2.1755695e-6`
  - `candidate_k_entry` max abs: `5.9008598e-6`
  - `candidate_v_entry` max abs: `6.1392784e-6`
  - decoder patches max abs: `8.0466270e-7`
  - prefill K/V max abs: `1.0132789e-5 / 5.00679e-6`
- Entry-cache reconstruction passed:
  - `candidate_k_cache_from_entry` max abs: `2.3841858e-7`
  - `candidate_v_cache`, `final_z`, and `pred_z`: exact
- Graph-capture benchmark:
  - Dynamics mean/median/p95: `48.44 / 54.43 / 54.89 ms`
  - Decoder mean/median/p95: `3.50 / 3.89 / 4.15 ms`
  - Streaming mean/median/p95: `52.30 / 58.71 / 59.18 ms`

Packed QKV + SwiGLU trial:
- Added post-export `--pack_qkv_gemm` and `--pack_swiglu_gemm`.
- This preserves the original tensor output names by replacing sibling biasless `Gemm` groups with one wider `Gemm` followed by `Split`.
- Validation stayed within the same tolerance as the restored manual path:
  - hot `final_z`/`pred_z` max abs: `2.1755695e-6`
  - `candidate_k_entry` max abs: `5.9008598e-6`
  - `candidate_v_entry` max abs: `6.1392784e-6`
  - decoder patches max abs: `8.0466270e-7`
  - prefill K/V max abs: `1.0132789e-5 / 5.00679e-6`
- Entry-cache reconstruction passed:
  - `candidate_k_cache_from_entry` max abs: `2.3841858e-7`
  - V/final_z/pred_z: exact
- Hot entry graph changed:
  - `Gemm`: `844 -> 487`
  - `Split`: `478 -> 716`
  - final nodes: `7438`
- Graph-capture benchmark:
  - Dynamics mean/median/p95: `42.06 / 47.70 / 48.24 ms`
  - Decoder mean/median/p95: `3.10 / 3.41 / 3.66 ms`
  - Streaming mean/median/p95: `45.47 / 51.51 / 51.94 ms`
  - Steady-state FPS: `21.99`

Conclusion:
- Packed projections are a real behavior-preserving speedup: roughly `7 ms` mean and `7 ms` median off the full streaming frame compared with the restored layout+rotary baseline.
- The result is still above the sub-25 target. The next controlled trials should separate QKV-only and SwiGLU-only packing, then move back to attention dispatch reduction if neither gets close.

### 2026-05-03 Runtime Pinning And GQA Follow-Up

Current hot artifact:
- `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4.onnx`
- `sample_steps=4`
- `head_projection_rewrite=layout`
- packed QKV/SwiGLU projections enabled
- fp32 graph, with softmax kept in fp32

Validation:
- Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
- Entry-cache reconstruction passed at `atol=5e-4`, `rtol=5e-4`.
- Current hot graph max abs errors stayed tiny:
  - `final_z`/`pred_z`: `2.1755695e-6`
  - `candidate_k_entry`: `5.9008598e-6`
  - `candidate_v_entry`: `6.1392784e-6`
  - `candidate_k_cache_from_entry`: `2.3841858e-7`

Rejected GQA experiment:
- Tried grouped-GQA attention with rank-5 `Einsum` to avoid materializing repeated K/V heads.
- CPU numerical validation passed.
- Browser WebGPU rejected the generated shader for an `Einsum` node, so the artifact was restored.
- Conclusion: rank-5 grouped `Einsum` is not a usable lowering for ONNX Runtime WebGPU here.

Runtime pinning experiments:
- Preallocated hot output tensors for the step and decoder sessions.
- Reused a persistent GPU tensor for the streaming latent input in the normal benchmark path.
- Both changes were valid but only moved latency slightly:
  - Restored baseline normal streaming: `51.786 / 51.725 ms` mean/median.
  - Preallocated outputs: `51.577 / 51.270 ms` mean/median.
  - Persistent GPU input: `51.414 / 50.987 ms` mean/median.

Conclusion:
- Runtime tensor pinning is close to exhausted; it saves fractions of a millisecond, not the 25+ ms needed for the new goal.
- The bottleneck is still the dynamics graph itself, mainly attention/projection dispatch count and GQA-related layout work.
- Next trial: split GQA by KV head using browser-safe rank-4/rank-3 attention `Einsum`s. This avoids `Gather`-based K/V repeat without using the WebGPU-incompatible rank-5 grouped `Einsum`.

Split-GQA trial result:
- Export flags matched the current best path, except `--attention_lowering split_gqa`.
- Numerical validation passed against the raw split-GQA graph:
  - `final_z`/`pred_z`: `2.1755695e-6`
  - `candidate_k_entry`: `5.9008598e-6`
  - `candidate_v_entry`: `6.1392784e-6`
  - entry-cache reconstruction passed.
- Hot graph changes:
  - nodes: `7319 -> 8985`
  - `Gather`: `239 -> 1`
  - `Einsum`: `238 -> 476`
  - `Softmax`: `119 -> 238`
  - `Slice`: `16 -> 730`
- Normal WebGPU benchmark:
  - Dynamics mean/median/p95: `63.50 / 63.29 / 66.32 ms`
  - Decoder mean/median/p95: `4.59 / 4.43 / 5.76 ms`
  - Streaming mean/median/p95: `68.15 / 68.02 / 70.82 ms`

Conclusion:
- Split-GQA is numerically valid but slower.
- Removing K/V repeat is not worth doubling the attention kernels in ONNX Runtime WebGPU.
- Restore the manual attention lowering for the active demo artifacts.

QKV-only packing trial:
- Export flags matched the current best path, except `--pack_swiglu_gemm` was disabled.
- Numerical validation passed against `/private/tmp/visionary_qkv_only_raw`:
  - `final_z`/`pred_z`: `2.1755695e-6`
  - `candidate_k_entry`: `5.9008598e-6`
  - `candidate_v_entry`: `6.1392784e-6`
  - entry-cache reconstruction passed.
- Hot graph changes versus both-pack baseline:
  - nodes: `7438`
  - `Gemm`: `487 -> 606`
  - `Split`: `716 -> 597`
- Normal WebGPU benchmark:
  - Dynamics mean/median/p95: `50.22 / 50.06 / 52.27 ms`
  - Decoder mean/median/p95: `3.72 / 3.65 / 3.98 ms`
  - Streaming mean/median/p95: `53.99 / 53.94 / 56.16 ms`
  - Steady-state FPS: `18.52`

Conclusion:
- QKV-only is slightly slower than the restored both-pack baseline.
- The added `Gemm` launches cost more than the removed `Split` nodes save.
- Keep QKV and SwiGLU packing together unless a later browser run shows a clear reversal.

SwiGLU-only packing trial:
- Export flags matched the current best path, except `--pack_qkv_gemm` was disabled.
- Numerical validation passed against `/private/tmp/visionary_swiglu_only_raw`:
  - `final_z`/`pred_z`: `2.1755695e-6`
  - `candidate_k_entry`: `5.9008598e-6`
  - `candidate_v_entry`: `6.1392784e-6`
  - entry-cache reconstruction passed.
- Hot graph changes versus both-pack baseline:
  - nodes: `7438 -> 7557`
  - `Gemm`: `487 -> 725`
  - `Split`: `716 -> 597`
- Normal WebGPU benchmark:
  - Dynamics mean/median/p95: `54.21 / 51.47 / 68.96 ms`
  - Decoder mean/median/p95: `3.75 / 3.72 / 3.95 ms`
  - Streaming mean/median/p95: `58.01 / 55.31 / 72.87 ms`
  - Steady-state FPS: `17.24`

Conclusion:
- SwiGLU-only is worse than both-pack.
- The active demo graph should use both `--pack_qkv_gemm` and `--pack_swiglu_gemm`.

Temporal `GroupQueryAttention` fusion trial:
- Export flags matched the current best both-pack path, with `--fuse_gqa_attention` added.
- The post-export rewrite fused only cached temporal attention islands into `com.microsoft::GroupQueryAttention`.
- Spatial attention was intentionally left manual because ORT WebGPU's `GroupQueryAttention` path uses the `seq_lens` input and behaves causally, which is not equivalent to bidirectional spatial attention.
- Numerical validation passed against `/private/tmp/visionary_temporal_gqa_raw`:
  - `final_z`/`pred_z`: `2.1755695e-6`
  - `candidate_k_entry`: `5.9008598e-6`
  - `candidate_v_entry`: `6.1392784e-6`
  - entry-cache reconstruction passed.
- Hot graph changes versus both-pack baseline:
  - nodes: `7438 -> 7728`
  - `GroupQueryAttention`: `0 -> 29`
  - `Einsum`: `238 -> 180`
  - `Softmax`: `119 -> 90`
  - `Unsqueeze`: `1869 -> 1988`
  - `Squeeze`: `1432 -> 1548`
  - `Slice`: `16 -> 132`
- Normal WebGPU benchmark:
  - Dynamics mean/median/p95: `53.94 / 53.75 / 56.39 ms`
  - Decoder mean/median/p95: `3.62 / 3.54 / 3.90 ms`
  - Streaming mean/median/p95: `57.61 / 57.43 / 60.10 ms`
  - Steady-state FPS: `17.36`

Conclusion:
- Temporal `GroupQueryAttention` is numerically valid but slower than manual attention.
- The fused op removes some attention kernels but adds layout work around past/current K/V and likely pays for present-cache outputs that the demo graph does not use.
- Reject this path unless ORT WebGPU exposes a cheaper no-present-output or non-causal GQA mode.

BNSH internal attention layout trial:
- Export flags matched the current best both-pack manual path, with `--attention_layout bnsh` added.
- The public ONNX cache/API shapes stayed unchanged. Attention internals were changed from `[B, S, H, D]` style work to `[B, H, S, D]` style work to see whether avoiding repeated attention transposes would help WebGPU.
- Numerical validation passed against both `/private/tmp/visionary_bnsh_raw` and the restored BSHD raw baseline:
  - `final_z`/`pred_z`: `6.9141388e-6`
  - `candidate_k_entry`: `9.536743e-6`
  - `candidate_v_entry`: `9.179115e-6`
  - entry-cache reconstruction passed.
- Hot graph changes versus both-pack BSHD baseline:
  - nodes: `7438 -> 9358`
  - `Transpose`: `537 -> 774`
  - `Reshape`: `0 -> 238`
  - `Mul`: `244 -> 734`
  - `Expand`: `0 -> 238`
  - `SimplifiedLayerNormalization`: `299 -> 66`
- Normal WebGPU benchmark:
  - Dynamics mean/median/p95: `403.77 / 380.82 / 501.24 ms`
  - Decoder mean/median/p95: `45.91 / 46.96 / 55.89 ms`
  - Streaming mean/median/p95: `449.75 / 430.24 / 553.68 ms`
  - Steady-state FPS: `2.22`

Conclusion:
- BNSH is numerically valid but a severe regression in ORT WebGPU.
- The exporter/optimizer no longer recognizes many layer norm patterns and introduces CPU-fallback `Reshape` nodes, so any theoretical transpose reduction is erased.
- Keep the active graph on BSHD/manual attention unless the BNSH export can recover `SimplifiedLayerNormalization` and eliminate `Reshape`.

MultiHeadAttention fusion trial:
- Export flags matched the current best packed manual path, with `--fuse_mha_attention` added.
- The first version rewrote manual attention islands to `com.microsoft::MultiHeadAttention` with flattened Q/K/V inputs.
- A second version used ORT WebGPU's supported 4D K/V form, keeping query flattened but passing key/value as `[B, H, S, D]` to avoid internal K/V reshape work.
- Numerical validation passed against `/private/tmp/visionary_mha_kv4d_raw_20260503`:
  - raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`
  - entry-cache reconstruction passed at `atol=5e-4`, `rtol=5e-4`
- Hot entry graph with K/V-4D MHA:
  - nodes: `6495`
  - `MultiHeadAttention`: `119`
  - `Einsum`: `0`
  - `Softmax`: `0`
  - `Transpose`: `775`
  - `Reshape`: `0`
- Browser benchmark, plain `bun run benchmark:webgpu`, graph capture enabled by the benchmark:
  - Prefill: `727.20 ms`
  - Dynamics after graph-capture warmup: mean/median/p95 `75.48 / 75.41 / 75.90 ms`
  - Decoder after graph-capture warmup: mean/median/p95 `5.43 / 5.39 / 5.58 ms`
  - Streaming after graph-capture warmup: mean/median/p95 `81.68 / 81.67 / 82.14 ms`

Conclusion:
- ORT WebGPU `MultiHeadAttention` is behavior-preserving in this graph, but slower than the packed manual attention baseline.
- The fused op removes `Einsum`/`Softmax`, but the required layout adapters and MHA implementation cost dominate.
- Reject `--fuse_mha_attention` for the active demo artifacts and restore the packed manual attention path.

Spatial `GroupQueryAttention` schema check:
- Goal: replace bidirectional spatial manual GQA attention islands with `com.microsoft::GroupQueryAttention` to avoid K/V head materialization.
- Initial export failed ONNX Runtime validation because `GroupQueryAttention` in this ORT build requires at least 7 inputs and 3 outputs.
- A direct graph probe with empty optional past/seq-len slots also failed: input 5 (`seq_lens`) is marked required by the schema.
- ORT WebGPU source shows that when `seq_lens` is present, the softmax path uses `past_sequence_length + query_index + 1`, which makes the attention causal.

Conclusion:
- Spatial `GroupQueryAttention` is not behavior-preserving for the decoder/dynamics spatial attention in the current ORT WebGPU schema.
- Keep spatial attention on the manual `Einsum`/`Softmax` lowering for now.
- The exporter now skips non-cached spatial GQA fusion even when the experimental flag is present, so the flag cannot produce an invalid or causally masked graph.

## 2026-05-03 KST: Late BHSD and MatMul Trials

Accepted baseline before these trials:
- Active artifact: `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4.onnx`
- Graph capture benchmark, plain `bun run benchmark:webgpu`:
  - Streaming frame: `49.54 ms`
  - Dynamics: `45.76 ms`
  - Decoder: `3.40 ms`
  - FPS: `20.18`

BHSD spatial attention layout rewrite:
- Rewrote spatial score attention to consume Q/K in the BHSD layout already produced by `RotaryEmbedding`.
- Dynamics graph:
  - Spatial attention sites rewritten: `90`
  - Nodes: `6148 -> 5968`
  - `Transpose`: `537 -> 357`
- Decoder graph:
  - Spatial attention sites rewritten: `6`
  - Nodes: `433 -> 421`
  - `Transpose`: `28 -> 16`
- Numerical validation passed against the raw unoptimized ONNX artifacts:
  - Entry step max abs error: about `6.14e-6`
  - Decoder max abs error: about `8.05e-7`
  - Prefill max abs error: about `1.01e-5`
  - Entry-cache reconstruction passed.
- Browser benchmark:
  - Streaming frame: `49.54 ms`
  - Dynamics: `45.76 ms`
  - Decoder: `3.40 ms`
  - FPS: `20.18`

Conclusion:
- Accepted.
- This was the first trial in this round that produced a real latency reduction without changing behavior.
- The win came from avoiding redundant `Transpose` nodes around spatial attention after RoPE.

Temporal q-only BHSD attention trial:
- Rewrote only the remaining temporal Q layout after RoPE.
- Dynamics graph:
  - Temporal sites rewritten: `29`
  - Nodes: `5968 -> 5939`
  - `Transpose`: `357 -> 328`
- Numerical validation passed with the same tolerance as the accepted baseline.
- Browser benchmark:
  - Streaming frame: `49.55 ms`
  - Dynamics: `45.58 ms`
  - Decoder: `3.54 ms`
  - FPS: `20.18`

Conclusion:
- Rejected as neutral.
- The tiny dynamics improvement was erased by decoder/frame noise.
- Keep the accepted BHSD spatial rewrite, but do not count q-only temporal as a useful optimization.

Score-side `Einsum -> MatMul` trial:
- Replaced attention score equations with `MatMul`:
  - `bhqd,bhkd->bhqk`
  - `bqhd,bkhd->bhqk`
- Left value-side attention `Einsum` unchanged.
- Dynamics graph:
  - Score sites rewritten: `119`
  - Nodes: `5968 -> 6116`
  - `MatMul`: `0 -> 119`
  - `Einsum`: `238 -> 119`
  - `Transpose`: `357 -> 505`
- Decoder graph:
  - Score sites rewritten: `8`
  - Nodes: `421 -> 431`
  - `MatMul`: `0 -> 8`
  - `Einsum`: `16 -> 8`
  - `Transpose`: `16 -> 26`
- Numerical validation passed:
  - Entry step max abs error: about `5.90e-6`
  - Decoder max abs error: about `8.05e-7`
  - Prefill unchanged and passed.
  - Entry-cache reconstruction passed.
- Browser benchmark:
  - Streaming frame: `53.44 ms`
  - Dynamics: `49.23 ms`
  - Decoder: `3.76 ms`
  - FPS: `18.71`

Conclusion:
- Rejected.
- ORT WebGPU's `MatMul` did not offset the extra transpose work in this shape regime.
- Future attention work should avoid local `MatMul` substitution unless it also removes surrounding layout kernels.

Benchmark stability note:
- A browser launch failure occurred while Google Chrome attempted to read the user's real Crashpad settings.
- `playwright.config.ts` now launches Chrome with an isolated `HOME` at `/private/tmp/visionary-chrome-home`.
- This does not affect model math or measured model execution; it avoids benchmark startup failures in the sandboxed environment.

Rank-preserving `MatMul` projection trial:
- Tried replacing `Squeeze -> Gemm -> Unsqueeze` projection islands with rank-preserving `MatMul` plus optional bias `Add`.
- Dynamics graph:
  - Rewrites: `128`
  - Nodes: `6023 -> 5776`
  - `Gemm`: `487 -> 359`
  - `MatMul`: `0 -> 128`
  - `Squeeze`: `570 -> 442`
  - `Unsqueeze`: `1988 -> 1860`
- Decoder graph:
  - Rewrites: `10`
  - Nodes: `424 -> 406`
  - `Gemm`: `34 -> 24`
  - `MatMul`: `0 -> 10`
- Numerical validation passed against raw ONNX:
  - Entry step max abs error stayed about `6.14e-6` or lower.
  - Decoder max abs error stayed about `8.05e-7`.
  - Entry-cache reconstruction passed.
- Browser benchmark result:
  - Rejected before timing because graph capture refused the session: not all nodes were partitioned to the WebGPU EP.

Conclusion:
- Rejected.
- ONNX `MatMul` is WebGPU-supported in general, but this rank-preserving form is not graph-capture viable in the current ORT WebGPU partitioner.
- Keep `Gemm`-based projection islands for the active demo graph.

Attention output `Flatten(axis=2)` trial:
- Tried replacing the value-attention head merge:
  - `Split(axis=2) -> Concat(axis=3) -> Squeeze -> Gemm`
  - with `Flatten(axis=2) -> Gemm`.
- Dynamics graph:
  - Rewrites: `119`
  - Nodes: `6023 -> 5785`
  - `Split`: `488 -> 369`
  - `Concat`: `543 -> 424`
  - `Squeeze`: `570 -> 451`
  - `Flatten`: `0 -> 119`
- Decoder graph:
  - Rewrites: `8`
  - Nodes: `424 -> 408`
  - `Split`: `37 -> 29`
  - `Concat`: `38 -> 30`
  - `Squeeze`: `38 -> 30`
  - `Flatten`: `0 -> 8`
- Numerical validation passed:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
  - Entry-cache reconstruction passed.
- Browser benchmark result:
  - Rejected before timing because graph capture again refused the session: not all nodes were partitioned to WebGPU.

Conclusion:
- Rejected.
- Although ORT WebGPU documents `Flatten` as supported, this graph-capture configuration does not accept the resulting graph.
- Do not replace existing `Squeeze/Concat` merge ladders with `Flatten` in the active graph.

Rank-2 SwiGLU activation trial:
- Tried keeping packed SwiGLU activation in rank-2 form:
  - Before: `Split -> Unsqueeze -> QuickGelu`, another `Unsqueeze`, `Mul -> Squeeze -> Gemm`.
  - After: `Split -> QuickGelu`, `Mul -> Gemm`.
- Dynamics graph:
  - Rewrites: `119`
  - Nodes: `6023 -> 5666`
  - `Unsqueeze`: `1988 -> 1750`
  - `Squeeze`: `570 -> 451`
  - `QuickGelu`, `Mul`, and `Gemm` counts unchanged.
- Decoder graph:
  - Rewrites: `8`
  - Nodes: `424 -> 400`
  - `Unsqueeze`: `134 -> 118`
  - `Squeeze`: `38 -> 30`
- Numerical validation passed:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
  - Entry-cache reconstruction passed.
- Browser benchmark:
  - Streaming frame: `50.01 ms`
  - Dynamics: `47.70 ms`
  - Decoder: `1.87 ms`
  - FPS: `20.00`

Conclusion:
- Rejected.
- The graph-capture path accepted the rewrite, but removing these rank adapters did not reduce the steady-state critical path.
- This confirms that small shape-node count reductions are not enough; the next trials should target repeated attention/projection work or cache update cost.

Spatial attention `Einsum` to `MatMul` trial:
- Rewrote spatial attention score/value pairs:
  - Score: `Einsum("bhqd,bhkd->bhqk")` to `Transpose(K) -> MatMul`.
  - Value: `Einsum("bhqk,bkhd->bqhd")` to `Transpose(V) -> MatMul -> Transpose`.
- Dynamics graph:
  - Spatial attention sites rewritten: `90`
  - Nodes: `6023 -> 6293`
  - `Einsum`: `238 -> 58`
  - `MatMul`: `0 -> 180`
  - `Transpose`: `302 -> 572`
- Numerical validation passed:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
  - Entry-cache reconstruction passed.
- Browser benchmark:
  - Streaming frame: `51.21 ms`
  - Dynamics: `47.63 ms`
  - Decoder: `3.18 ms`
  - FPS: `19.53`

Conclusion:
- Rejected.
- Replacing generic attention `Einsum` with `MatMul` does not help when it adds this much transpose traffic.
- Keep the accepted BHSD spatial `Einsum` layout; future attention work must remove layout kernels, not just swap the core multiply op.

Packed QKV head projection trial:
- Replayed the current raw-to-WebGPU optimization pipeline with `--pack_qkv_head_projection` plus the existing packed QKV/SwiGLU Gemm passes.
- Dynamics graph:
  - Nodes: `6023 -> 5157`
  - `Gemm`: `487 -> 397`
  - `Einsum`: `238 -> 328`
  - `Unsqueeze`: `1988 -> 1127`
  - `Concat`: `543 -> 273`
  - `Transpose`: `302 -> 537`
  - `Reshape`: `0`
- Numerical validation passed:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
  - Entry-cache reconstruction passed.
- Browser benchmark:
  - Streaming frame: `56.74 ms`
  - Dynamics: `52.62 ms`
  - Decoder: `3.61 ms`
  - FPS: `17.63`

Conclusion:
- Rejected.
- The pass removes many layout nodes, but it breaks the accepted low-transpose BHSD attention layout and adds generic `Einsum` projection work.
- Do not enable `--pack_qkv_head_projection` for the current demo graph.

Spatial Q/K direct BHSD layout trial:
- Rewrote spatial Q/K head construction before `RotaryEmbedding`.
  - Before: per-head tensors were concatenated as `B,S,H,D`, then transposed to `B,H,S,D` for RoPE.
  - After: per-head tensors are unsqueezed/concatenated directly as `B,H,S,D`, so RoPE consumes the tensor without a wrapper transpose.
- Dynamics graph:
  - Spatial Q/K sites rewritten: `180`
  - Nodes: `6023 -> 5843`
  - `Transpose`: `302 -> 122`
  - `RotaryEmbedding`: unchanged at `239`
  - `Reshape`: `0`
- Decoder graph:
  - Spatial Q/K sites rewritten: `12`
  - Nodes: `424 -> 412`
  - `Transpose`: `13 -> 1`
- Numerical validation:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
  - Largest observed cache error stayed around `1.01e-5`; generated latents and decoded patches stayed around `1e-6` or lower.
  - Entry-cache reconstruction passed.
- Browser benchmark:
  - Streaming frame: `48.34 ms`
  - Dynamics: `44.73 ms`
  - Decoder: `3.15 ms`
  - FPS: `20.69`
  - Graph capture: passed.

Conclusion:
- Accepted.
- This is the current best graph. It is a real but modest win because it removes layout traffic around RoPE without changing attention math.
- The accepted graph depends on the `bnsh` attention export layout; older `bshd` raw artifacts still require RoPE transposes back to `B,S,H,D`.

Temporal `GroupQueryAttention` fusion trial:
- Applied the existing post-export GQA fusion to the current accepted graph.
- Dynamics graph:
  - Temporal GQA sites rewritten: `29`
  - `GroupQueryAttention`: `0 -> 29`
  - `Einsum`: `238 -> 180`
  - `Softmax`: `119 -> 90`
  - `Split`: `488 -> 546`
  - `Squeeze`: `570 -> 918`
  - `Concat`: `543 -> 601`
- Numerical validation:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
- Browser benchmark:
  - Rejected because graph capture fails in ORT WebGPU:
    `GroupQueryAttention ... Invalid dispatch group size (0, 1, 1)`.

Conclusion:
- Rejected for the demo path.
- ORT WebGPU has a `GroupQueryAttention` resolver, but this particular generated temporal/cache shape is not graph-capture viable.
- The fusion also adds substantial layout glue, so even if the dispatch bug were avoided, it would need a fresh benchmark before being considered.

Graph-capture preallocated-output trial:
- Tried enabling preallocated GPU output tensors for graph-capture benchmark runs.
- This does not change ONNX math; it only attempts to avoid per-frame output tensor allocation.
- Browser benchmark:
  - Rejected because ORT WebGPU graph capture fails with:
    `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`.

Conclusion:
- Rejected.
- Keep the existing benchmark behavior: preallocated hot outputs are used for non-graph-capture runs only.

Cache-update precomputed RoPE constants trial:
- Moved the entry-cache slide/rebase shader's one-step RoPE `cos/sin` values from per-work-item `pow/cos/sin` calls into precomputed GPU buffers.
- ONNX graph validation still passed because the model graph was unchanged.
- Browser benchmark:
  - Streaming frame: `48.05 ms`
  - Dynamics: `44.46 ms`
  - Decoder: `3.13 ms`
  - FPS: `20.81`

Conclusion:
- Accepted as a small runtime improvement.
- The effect is modest, which suggests the cache-update shader is not the dominant cost.

ORT graph optimization level trial:
- Changed browser session creation from `graphOptimizationLevel: "all"` to `"extended"`.
- Browser benchmark:
  - Streaming frame: `49.25 ms`
  - Dynamics: `45.53 ms`
  - Decoder: `3.26 ms`
  - FPS: `20.30`

Conclusion:
- Rejected.
- Keep `graphOptimizationLevel: "all"`.

Spatial attention value plus output projection fusion trial:
- Fused the spatial value-attention output projection:
  - Before: `Einsum("bhqk,bkhd->bqhd") -> Split -> Concat -> Squeeze -> Gemm -> Unsqueeze`.
  - After: `Einsum("bhqk,bkhd,hdm->bqm")` with the output projection weight reshaped from `(512,256)` to `(8,64,256)`.
- Applied only to spatial sites first.
- Dynamics graph:
  - Rewrites: `90`
  - `Gemm`: `487 -> 397`
  - `Split`: `488 -> 398`
  - `Concat`: `543 -> 453`
  - `Squeeze`: `570 -> 480`
  - `Unsqueeze`: `1988 -> 1898`
  - `Einsum`: unchanged at `238`
- Numerical validation:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
  - Entry-cache reconstruction passed.
- Browser benchmark:
  - Streaming frame: `179.44 ms`
  - Dynamics: `159.94 ms`
  - Decoder: `15.03 ms`
  - FPS: `5.57`

Conclusion:
- Rejected.
- ORT WebGPU's 3-input `Einsum` lowering is far slower than the separate value `Einsum` plus `Gemm` path, despite removing many layout nodes.
- Do not fuse output projection into n-ary `Einsum`.

MLP down-projection rank-3 `MatMul` trial:
- Replaced MLP down-projection ladders:
  - Before: rank-3 `Mul -> Squeeze -> Gemm(768,256) -> Unsqueeze`.
  - After: rank-3 `MatMul` with the original `(768,256)` weight.
- Dynamics graph:
  - Rewrites: `119`
  - `MatMul`: `0 -> 119`
  - `Gemm`: `487 -> 368`
  - `Squeeze`: `570 -> 451`
  - `Unsqueeze`: `1988 -> 1869`
  - `Reshape`: `0`
- Numerical validation:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
  - Entry-cache reconstruction passed.
- Browser benchmark:
  - Streaming frame: `48.44 ms`
  - Dynamics: `44.81 ms`
  - Decoder: `3.13 ms`
  - FPS: `20.65`

Conclusion:
- Rejected.
- Rank-3 `MatMul` is graph-capture safe here, but it does not beat ORT WebGPU's `Squeeze/Gemm/Unsqueeze` path.

SwiGLU input-projection rank-3 `MatMul` trial:
- Replaced packed SwiGLU input projection ladders:
  - Before: rank-3 input `Squeeze -> packed Gemm(256,1536) -> Split(axis=1) -> Unsqueeze x2`.
  - After: rank-3 `MatMul -> Split(axis=2)`, feeding the original rank-3 QuickGELU/Mul path.
- Dynamics graph:
  - Rewrites: `119`
  - `MatMul`: `0 -> 119`
  - `Gemm`: `487 -> 368`
  - `Squeeze`: `570 -> 451`
  - `Unsqueeze`: `1988 -> 1750`
  - `Split`: unchanged at `488`
  - `Reshape`: `0`
- Numerical validation:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
  - Entry-cache reconstruction passed.
- Browser benchmark:
  - Streaming frame: `49.91 ms`
  - Dynamics: `46.20 ms`
  - Decoder: `3.23 ms`
  - FPS: `20.04`

Conclusion:
- Rejected.
- Rank-3 `MatMul` is graph-capture safe but slower than the current packed `Gemm` plus layout adapters.

Temporal `MultiHeadAttention` fusion trial:
- Applied the existing `com.microsoft::MultiHeadAttention` rewrite to the hot entry-cache dynamics artifact only.
- The matcher fused the cached temporal attention sites:
  - `MultiHeadAttention`: `0 -> 29`
  - `Einsum`: `238 -> 180`
  - `Softmax`: `119 -> 90`
  - `Transpose`: `122 -> 180`
  - `Squeeze`: `570 -> 599`
  - Nodes: `5843 -> 5872`
- Numerical validation:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
  - Entry-cache reconstruction passed.
- Browser benchmark:
  - Streaming frame: `58.13 ms`
  - Dynamics: `53.50 ms`
  - Decoder: `3.66 ms`
  - FPS: `17.20`

Conclusion:
- Rejected.
- The fused MHA kernel itself is browser-compatible and graph-capture-safe, but this post-export rewrite only hits temporal sites and adds enough K/V layout transposes and output squeezes to lose about `9.3 ms/frame`.
- Do not enable the current `--fuse_mha_attention` pass for the live entry-cache graph.

Spatial `MultiHeadAttention` plus output `Gemm` trial:
- Built a temporary post-export rewrite for spatial attention sites:
  - Replaced `Einsum("bhqd,bhkd->bhqk") -> Softmax -> Einsum("bhqk,bkhd->bqhd")`.
  - Removed the following head merge ladder and reused the output projection as `Gemm`.
  - Fed `com.microsoft::MultiHeadAttention` with flat query and BNSH K/V layout.
- Dynamics graph:
  - Rewrites: `90`
  - `MultiHeadAttention`: `0 -> 90`
  - `Einsum`: `238 -> 58`
  - `Softmax`: `119 -> 29`
  - `Transpose`: `122 -> 302`
  - `Squeeze`: `570 -> 660`
  - Nodes: `5843 -> 5933`
- Numerical validation:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
  - Entry-cache reconstruction passed.
- Browser benchmark:
  - Streaming frame: `70.91 ms`
  - Dynamics: `64.88 ms`
  - Decoder: `4.58 ms`
  - FPS: `14.10`

Conclusion:
- Rejected.
- The fused MHA op is graph-capture-safe, but for the fixed spatial shape `[B=1, H=8, S=36, D=64]` the required layout adapters and MHA kernel are much slower than the current manual attention plus `Gemm` path.
- Do not pursue MHA fusion unless the graph can emit MHA-native layouts directly from export without post-export transposes.

Attention `Einsum` to explicit `MatMul` trial:
- Built a temporary post-export rewrite over the hot entry-cache dynamics artifact:
  - `bhqd,bhkd->bhqk`: `90` sites replaced by `Transpose(K) -> MatMul`.
  - `bqhd,bkhd->bhqk`: `29` sites replaced by `Transpose(Q) -> Transpose(K) -> MatMul`.
  - `bhqk,bkhd->bqhd`: `119` sites replaced by `Transpose(V) -> MatMul -> Transpose(output)`.
- Dynamics graph:
  - `Einsum`: `238 -> 0`
  - `MatMul`: `0 -> 238`
  - `Transpose`: `122 -> 508`
  - Nodes: `5843 -> 6229`
- Numerical validation:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
  - Entry-cache reconstruction passed.
- Browser benchmark with `bun run benchmark:webgpu`:
  - Streaming frame: `59.02 ms`
  - Dynamics: `54.95 ms`
  - Decoder: `3.74 ms`
  - FPS: `16.94`

Conclusion:
- Rejected.
- ORT WebGPU's `MatMul` kernels are not enough to offset the extra static transposes. The existing `Einsum` attention path is faster for this graph.
- This reinforces that post-export attention layout adapters are usually worse than the current accepted manual attention graph.

Export-native flat B=1/T=1 step-layout trial:
- Added a temporary export-only branch that squeezed the cached single-step spatial path from `[B=1, T=1, N, D]` to `[N, D]` before spatial transformer blocks, then restored the original output ABI.
- Goal:
  - Avoid redundant singleton dimensions in the source JAX graph so jax2onnx would emit less layout churn for the demo-only entry-cache artifact.
- Numerical validation:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
  - Entry-cache reconstruction passed.
- Dynamics graph result:
  - Nodes: `5843 -> 6786`
  - `Reshape`: `0 -> 598`
  - `Transpose`: `122 -> 535`
  - `Einsum`: `238 -> 356`
  - `Gemm`: `487 -> 726`
- Browser benchmark with `bun run benchmark:webgpu`:
  - Regular streaming tests passed, but graph-capture benchmark failed session creation.
  - ORT error: graph capture could not be used because not all compute graph nodes were partitioned to WebGPU.

Conclusion:
- Rejected and reverted.
- The source-level rank reduction made the exported graph less WebGPU-friendly. It reintroduced many `Reshape` nodes, which are CPU-only in ORT WebGPU, and broke graph capture.
- The accepted baseline was restored from `/private/tmp/visionary_active_before_flat_step_trial_20260503`.

Full temporal BHSD layout rewrite trial:
- Built a temporary post-export graph rewrite over the accepted entry-cache artifact.
- Goal:
  - Keep temporal Q/K/V attention islands in `[B,H,S,D]` instead of converting back to `[B,S,H,D]`.
  - Remove the remaining temporal RoPE transpose wrappers without changing attention math.
- Matched all temporal islands:
  - Temporal sites: `29`
  - Nodes: `5843 -> 5785`
  - `Transpose`: `122 -> 64`
  - `Reshape`: stayed `0`
- Validation result:
  - Rejected before benchmarking.
  - ORT refused to load the rewritten graph because final cache-entry output shape inference no longer matched the declared public ABI.
  - Concrete failure: `node_Concat_14694` inferred dimension `1` where the declared cache-entry layout expects dimension `2`.

Conclusion:
- Rejected and reverted.
- The current public K/V cache ABI is `[B,T,H,D]`. A full temporal `[B,H,T,D]` rewrite changes the layout of K/V entries that are also graph outputs.
- Adding transposes back only for those outputs would erase most or all of the intended dispatch reduction.
- A useful version of this idea requires a deliberate cache ABI change: store and stream temporal K/V cache in `[B,H,T,D]` layout across prefill, entry graph outputs, and the browser cache-update shader, then validate against the raw graph with explicit output/input transposes in the validator.

### 2026-05-03 Temporal Attention Fusion Retests

Goal: reduce the current accepted graph-capture steady-state latency of roughly 48.6 ms/frame without changing sample_steps=4 or numerical outputs.

Validation rule used before browser benchmarking:
- Raw jax2onnx artifact comparison via `scripts/webgpu/compare_raw_optimized_onnx.py`.
- Entry-cache reconstruction comparison via `scripts/webgpu/verify_entry_cache_update.py`.

Trial: full-KV temporal `GroupQueryAttention` without past-cache inputs.
- Replaced 29 temporal manual attention sites.
- Intended graph delta: `Softmax 119 -> 90`, `Einsum 238 -> 180`, `Gather 239 -> 181`.
- Rejected before benchmark. ORT CPU validation rejected the graph because `GroupQueryAttention` requires schema inputs/outputs and then rejects no-past query length 1 with key length 65. This is not safe under the raw-vs-optimized validation gate.

Trial: temporal `MultiHeadAttention`.
- Replaced 29 temporal manual attention sites.
- Validation passed; max raw-vs-optimized error stayed around 6e-6.
- Browser benchmark passed, including graph capture, but was slower.
- Result: streaming frame mean 57.72 ms, dynamics mean 53.12 ms, decoder mean 3.69 ms.
- Accepted baseline before the trial: streaming frame mean 48.62 ms, dynamics mean 45.03 ms.
- Conclusion: reject. ORT WebGPU's MHA path adds enough layout/work overhead that the manual temporal attention graph is faster on this model.

Trial: spatial-only `MultiHeadAttention`.
- Replaced 90 spatial manual attention sites while leaving the temporal attention path unchanged.
- Graph delta:
  - `MultiHeadAttention`: `0 -> 90`
  - `Einsum`: `238 -> 58`
  - `Softmax`: `119 -> 29`
  - `Transpose`: `122 -> 212`
  - `Squeeze`: `570 -> 660`
  - Nodes: unchanged at `5843`
- Numerical validation:
  - Raw-vs-optimized ONNX comparison passed; max errors stayed at the accepted fp32 baseline scale.
  - Entry-cache reconstruction passed.
- Browser benchmark with `bun run benchmark:webgpu`:
  - Graph-capture streaming frame: `68.77 ms`
  - Dynamics after graph-capture warmup: `63.27 ms`
  - Decoder after graph-capture warmup: `4.63 ms`
- Accepted baseline before the trial:
  - Graph-capture streaming frame: `48.62 ms`
  - Dynamics: `45.03 ms`
  - Decoder: `3.09 ms`
- Conclusion: reject. Removing many small attention ops was outweighed by ORT WebGPU's MHA layout work and added dispatch cost. The accepted manual spatial attention graph is faster.

Trial: temporal past-cache `GroupQueryAttention`.
- Replaced 29 temporal manual attention sites with `com.microsoft::GroupQueryAttention` using current-token K/V plus past-cache K/V.
- Graph delta:
  - `GroupQueryAttention`: `0 -> 29`
  - `Einsum`: `238 -> 180`
  - `Softmax`: `119 -> 90`
  - `Gather`: `239 -> 181`
  - `Split`: `488 -> 546`
  - `Squeeze`: `570 -> 918`
  - `Concat`: `543 -> 601`
  - Nodes: `5843 -> 6365`
- Numerical validation:
  - Raw-vs-optimized ONNX comparison passed.
  - Entry-cache reconstruction passed.
- Browser benchmark:
  - Non-graph-capture WebGPU smoke and streaming tests passed.
  - Graph-capture test failed inside ORT WebGPU `GroupQueryAttention`.
  - Error: `Invalid dispatch group size (0, 1, 1)`.
- Conclusion: reject. Even though the math validates, the ORT WebGPU GQA kernel is not reliable in the graph-capture demo path for this shape.

Trial: packed QKV head projection pass.
- Ran the existing `rewrite_packed_qkv_head_projection_for_webgpu` pass on the current accepted step graph.
- Result: no matched rewrites, no graph changes.
- Conclusion: no-op on the current optimized artifact.

Trial: export-native final context-entry K/V-only block.
- Added a temporary inference-only source path that skipped the final temporal block's attention output and MLP during cache-entry writing, computing only the K/V entries needed for the rolling context cache.
- Goal:
  - Avoid one full temporal attention/MLP block in the context-cache update part of the demo frame.
  - Keep the public ONNX input/output contract unchanged.
- Numerical validation:
  - Raw-vs-optimized ONNX comparison passed at `atol=5e-4`, `rtol=5e-4`.
  - Entry-cache reconstruction passed.
  - Max errors stayed around `1e-5` for K/V entries and `7e-6` for sampled latent output.
- Exported graph result:
  - Nodes: `5843 -> 8148`
  - `Einsum`: `238 -> 716`
  - `Transpose`: `122 -> 774`
  - `Reshape`: `0 -> 238`
  - `Gather`: `239 -> 5`
  - `SimplifiedLayerNormalization`: `299 -> 66`
- Browser benchmark:
  - Non-graph-capture tests became much slower.
  - Graph-capture session creation failed because the graph was no longer fully assigned to WebGPU.
- Conclusion: reject and revert.
  - The source-level shortcut changed jax2onnx lowering enough to reintroduce CPU-only `Reshape` nodes and many more layout ops.
  - Numerical accuracy alone was not sufficient; WebGPU partitioning and graph-capture viability must stay in the validation gate.

## 2026-05-04 KST: Final Output And Runtime Fusion Trials

Baseline entering this round:
- Active artifact: `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4.onnx`.
- Graph shape: `Reshape=0`, `Einsum=238`, `Gemm=487`, `Transpose=98`, `Unsqueeze=1988`, `Squeeze=570`.
- Typical graph-capture benchmark range before this round:
  - Dynamics after graph-capture warmup: about `44.5-45.1 ms`.
  - Streaming frame after graph-capture warmup: about `48.5-49.0 ms`.

Numerical validation policy:
- Fixed deterministic inputs/noise.
- Compare optimized ONNX against raw unoptimized ONNX on CPU.
- For the final-output-only graph, compare raw `pred_z` and raw `final_z` against optimized `final_z` because the optimized graph intentionally removes the redundant `pred_z` output.
- Keep tolerance at `atol=5e-4`, `rtol=5e-4`; observed max absolute errors stayed around `1e-6`.

Accepted trial: final-output-only steady step.
- Rewrote the optimized entry step output contract to remove `pred_z`.
- The graph now outputs only:
  - `final_z`
  - `candidate_k_entry`
  - `candidate_v_entry`
- Reason:
  - At the fourth and final sampler update, `final_z` is mathematically the same value as `pred_z`.
  - The browser only needs this final latent for the next frame and decoder.
- Graph result:
  - Nodes: `5791`
  - `Identity=0`
  - `Reshape=0`
  - `Transpose=74`
  - `Einsum=238`
  - `Gemm=487`
- Numerical validation:
  - Raw `final_z` vs optimized `final_z`: max abs `2.1755695e-6`.
  - Raw `pred_z` vs optimized `final_z`: max abs `2.1755695e-6`.
  - `candidate_k_entry`: max abs `6.4373016e-6`.
  - `candidate_v_entry`: max abs `5.9604645e-6`.
  - Decoder patches stayed valid: max abs `1.1920929e-6`.
  - Entry-cache reconstruction passed; full-cache `pred_z` vs entry `final_z` was exact in the check.
- Browser benchmark:
  - Graph capture passed.
  - Best run in this round: streaming frame after graph-capture warmup about `47.96 ms`.
  - Later repeat after cleanup: streaming frame after graph-capture warmup about `48.63 ms`.
- Conclusion:
  - Accept as a small, behavior-preserving output-contract cleanup.
  - The gain is modest and close to benchmark noise, but it removes a redundant graph output and keeps graph capture viable.

Rejected trial: attention `Einsum` to `MatMul`.
- Replaced all fixed attention `Einsum` equations with equivalent `Transpose` + `MatMul` patterns:
  - `bhqd,bhkd->bhqk`
  - `bhqk,bhkd->bqhd`
  - `bhqk,bkhd->bqhd`
- Numerical validation passed:
  - Max errors stayed identical to the accepted final-output-only graph.
- Graph result:
  - `Einsum`: `238 -> 0`
  - `MatMul`: `0 -> 238`
  - `Transpose`: `74 -> 402`
  - Nodes: `5791 -> 6119`
- Browser benchmark:
  - Streaming frame after graph-capture warmup: about `54.56 ms`.
  - Dynamics after graph-capture warmup: about `50.24 ms`.
- Conclusion:
  - Reject.
  - ORT WebGPU `MatMul` does not offset the extra transpose traffic for this attention layout.

Rejected trial: layer-cache prefill and steady-state step.
- Existing layer-cache artifacts validated against raw ONNX:
  - Layer prefill passed.
  - Layer steady-state step passed.
- Browser result:
  - Normal benchmark failed because runtime cache handling tried to read GPU cache tensors as CPU data.
  - Graph-capture benchmark failed session creation because not all layer-cache step nodes were assigned to WebGPU.
- Conclusion:
  - Reject for the demo path.
  - Keep the entry-cache artifact plus browser-side in-place cache slide/rebase.

Rejected trial: composite step plus decoder ONNX session.
- Built a temporary composite graph:
  - Inputs: same as the accepted entry step.
  - Outputs: `final_z`, `candidate_k_entry`, `candidate_v_entry`, `patches`.
  - The decoder consumed `final_z` inside the same ONNX graph.
- Numerical validation passed against a raw composite graph:
  - `patches`: max abs `7.1525574e-7`.
  - `final_z`, K entry, V entry stayed at accepted error levels.
- Browser benchmark:
  - Separate decoder timing dropped to `0 ms`, as intended.
  - But the combined graph made the captured step slower.
  - Streaming frame after graph-capture warmup: about `52.39 ms`.
  - Dynamics/combined step after graph-capture warmup: about `51.95 ms`.
- Conclusion:
  - Reject and remove the temporary artifact.
  - Combining sessions increases the step graph enough that it loses more than the separate decoder call costs.

Runtime graph optimization level trial:
- Tested session `graphOptimizationLevel` values on the accepted final-output-only artifact.
- `all`: accepted default, current range around `48-49 ms` streaming after graph-capture warmup.
- `extended`: passed, about `48.18 ms` streaming after graph-capture warmup.
- `basic`: passed, about `48.15 ms` streaming after graph-capture warmup.
- `disabled`: passed but regressed to about `49.55 ms`.
- Conclusion:
  - Keep `basic` for now as a tiny runtime-level improvement.
  - This is not a structural speedup; it does not change the main bottleneck.

Current bottleneck:
- The demo is still dynamics-bound.
- Steady graph-capture dynamics remains around `44-45 ms`; decoder plus copy overhead is around `3.5-4.0 ms`.
- The remaining practical bottleneck is the cost of five unrolled transformer-style passes for four sampler steps plus the context-entry cache update path.
- Small layout/output changes are now exhausted; reaching `25 ms` likely requires a genuinely faster attention/MLP execution path or fewer equivalent compute passes, not another local shape rewrite.

## 2026-05-04 KST: Status After Layer-Cache Retry

Current accepted browser path:
- Prefill artifact priority is back to `breakout_dynamics_prefill_cached_b1_t64`.
- Step artifact priority is back to `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4`.
- The latest saved `latest.json` may still show the rejected layer-cache trial if no newer benchmark has overwritten it.

Latest accepted numerical validation:
- Raw optimized comparison passed for `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4`.
- `final_z` max abs error: `4.5299530e-6`.
- `pred_z` compared to optimized `final_z` max abs error: `4.5299530e-6`.
- `candidate_k_entry` max abs error: `3.7431717e-5`.
- `candidate_v_entry` max abs error: `1.5676022e-5`.
- Tolerance remains `atol=5e-4`, `rtol=5e-4`.

Latest accepted timing before the layer-cache retry:
- Runtime graph optimization level: `basic`.
- Dynamics after graph-capture warmup: about `43.45 ms`.
- Decoder after graph-capture warmup: about `3.80 ms`.
- Streaming frame after graph-capture warmup: about `48.02 ms`.
- This is viable for roughly `20 fps`, but not close to the `25 ms` target.

Rejected layer-cache retry:
- Layer-cache artifacts passed numerical validation.
- The layer-cache step graph was larger than the entry-cache step graph:
  - Nodes: about `6734`.
  - `Transpose`: about `441`.
- Browser timing regressed to roughly `53.6 ms` per streaming frame after graph-capture warmup.
- Normal benchmark tests also exposed GPU-buffer cache handling issues for the layer-cache output contract.

Conclusion:
- Keep the entry-cache path.
- The remaining speed target is no longer blocked by host-device cache copies; those are now tiny.
- The dominant cost is the steady fp32 dynamics graph itself, especially repeated attention and MLP work across four sampler steps plus the context-entry update.
- Reaching `25 ms` probably requires changing the execution layer, such as specialized ORT WebGPU kernels for the current `Einsum` attention equations or fused packed SwiGLU/projection kernels. Another small ONNX shape rewrite is unlikely to cut the frame time in half.

## 2026-05-04 KST: Two-Step Dynamics Sampler Trial

Goal:
- Test whether exporting the fused dynamics sampler with `--sample_steps 2` gives a useful speedup.
- This intentionally changes rollout semantics compared with the accepted `sample_steps=4` demo path, so the validation target is raw two-step ONNX vs optimized two-step ONNX, not equivalence to the four-step sampler.

Command:
- `uv run python scripts/webgpu/export_dreamer4_onnx.py ... --sample_steps 2 --raw_out_dir webgpu_app/assets_raw_s2 --export_cached --validate --overwrite`

Important naming caveat:
- The exporter still writes the sample-step artifacts with `_s4` in the filename.
- The manifest correctly records `sample_steps: 2`.
- The browser benchmark uses the manifest sample-step metadata for the fused entry-cache graph, so the benchmark result is still a valid two-step timing.

Numerical validation:
- Raw-vs-optimized ONNX comparison passed for `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4`.
- Tolerance: `atol=5e-4`, `rtol=5e-4`.
- `final_z` max abs error: `1.8477440e-6`.
- `pred_z` compared to optimized `final_z` max abs error: `1.8477440e-6`.
- `candidate_k_entry` max abs error: `7.3909760e-6`.
- `candidate_v_entry` max abs error: `5.5991113e-6`.
- Entry-cache reconstruction passed:
  - K cache max abs error: `5.0067902e-6`.
  - V cache max abs error: `3.0994415e-6`.
  - `final_z` max abs error: `1.1920929e-7`.

Graph size:
- Nodes: `3543`.
- `Gemm`: `291`.
- `Einsum`: `142`.
- `Softmax`: `71`.
- `QuickGelu`: `71`.
- This is much smaller than the four-step accepted graph because the fused sampler now unrolls two model passes instead of four plus the context-entry update.

Browser benchmark:
- Command: `bun run benchmark:webgpu`.
- Result: passed smoke, streaming, and graph-capture tests.
- Sampling config in result:
  - `sample_steps: 2`.
  - `sample_step_level: 1`.
  - `generated_frames: 64`.
- Cached dynamics step after graph-capture warmup:
  - Mean: `28.80 ms`.
  - Median: `32.05 ms`.
  - P95: `37.52 ms`.
- Decoder after graph-capture warmup:
  - Mean: `5.19 ms`.
  - Median: `5.45 ms`.
  - P95: `7.93 ms`.
- Streaming frame after graph-capture warmup:
  - Mean: `34.64 ms`.
  - Median: `38.82 ms`.
  - P95: `43.86 ms`.
  - Throughput: `28.87 fps`.

Conclusion:
- The two-step export works and is substantially faster than the four-step export.
- It does not reach the `25 ms` target, but it moves the browser demo from about `20 fps` to about `29 fps`.
- The tradeoff is model behavior: this is a lower-quality/fewer-solver-step sampler, so rollout quality must be judged visually or against an offline metric before accepting it as the main demo setting.

## 2026-05-04 KST: INT4 Weight-Only Quantization Trial

Goal:
- Test whether ORT INT4 weight-only quantization improves the current optimized two-step dynamics step model.
- Keep the active demo artifact restored to fp32 after the trial because INT4 output drift is not yet acceptable.

Method:
- ORT INT4 quantization targets constant-weight `MatMul` nodes, but the current optimized graph uses `Gemm` for dense layers.
- Rewrote all eligible `Gemm` nodes to numerically equivalent `MatMul` plus optional `Add`.
- The fp32 rewrite was exactly equivalent on CPU for the tested deterministic feeds:
  - `final_z` max abs error: `0.0`.
  - `candidate_k_entry` max abs error: `0.0`.
  - `candidate_v_entry` max abs error: `0.0`.
- Applied ORT `MatMulNBits` INT4 quantization with symmetric weights and block size 32.

Full INT4 b32 graph:
- Converted `291` dense matmuls to `MatMulNBits`.
- File size dropped from about `191 MiB` to about `30.7 MiB`.
- CPU output drift against fp32:
  - `final_z` mean abs error: `0.0626`, max abs error: `0.4488`.
  - `candidate_k_entry` mean abs error: `0.1372`, max abs error: `1.9553`.
  - `candidate_v_entry` mean abs error: `0.0691`, max abs error: `0.9422`.
- WebGPU benchmark passed, including graph capture.
- WebGPU timing after graph-capture warmup:
  - Dynamics median: `23.33 ms`, p95: `24.00 ms`.
  - Streaming frame median: `29.13 ms`, p95: `29.63 ms`.

Comparison with fp32 in the same run window:
- Fp32 active graph size: about `191 MiB` on disk, `200,322,981` bytes over HTTP.
- Fp32 WebGPU timing after graph-capture warmup:
  - Dynamics median: `25.99 ms`, p95: `26.43 ms`.
  - Streaming frame median: `31.90 ms`, p95: `32.60 ms`.
- Full INT4 b32 speedup:
  - Dynamics median improved by about `10.2%`.
  - Streaming median improved by about `8.7%`.
  - Session creation also improved because the model payload is much smaller.

Rejected variants:
- Full INT4 b128:
  - Smaller file, about `26.3 MiB`.
  - Slower than fp32 in WebGPU: streaming median `37.00 ms`, p95 `37.84 ms`.
  - Worse output drift than b32.
- Selective INT4 b32 for qkv plus SwiGLU projections:
  - File size about `56.3 MiB`.
  - Streaming median `31.32 ms`, p95 `32.00 ms`, barely faster than fp32.
  - Output drift was still far above the normal raw-vs-optimized tolerance.

Conclusion:
- INT4 b32 does speed up ORT WebGPU, but only modestly for this graph.
- The current full INT4 model is not numerically safe as a replacement for the fp32 demo model.
- The useful takeaway is that `MatMulNBits` runs on ORT WebGPU and graph capture accepts it; a production INT4 path would need quantization-aware validation or a more selective policy with a clear quality metric.

## 2026-05-04 KST: INT8 Quantization Trial

Goal:
- Test whether INT8 is a better speed/quality tradeoff than INT4 for the optimized two-step dynamics step model.
- Keep the active demo artifact restored to fp32 after the trial.

Dynamic INT8:
- ORT dynamic quantization on the original `Gemm` graph barely changed the graph:
  - Most dense nodes stayed as `Gemm`.
  - File size stayed about `191 MiB`.
  - CPU outputs were exactly equal for the tested feeds because the hot dense path was effectively not quantized.
- ORT dynamic quantization on the `Gemm -> MatMul` rewrite produced:
  - `291` `MatMulInteger` nodes.
  - `290` `DynamicQuantizeLinear` nodes.
  - File size about `49.7 MiB`.
- CPU accuracy was better than INT4 for signed INT8 but still far outside the normal raw-vs-optimized tolerance:
  - `final_z` mean abs error: `0.0185`, max abs error: `0.1560`.
  - `candidate_k_entry` mean abs error: `0.0413`, max abs error: `2.6112`.
  - `candidate_v_entry` mean abs error: `0.0216`, max abs error: `0.6397`.
- Browser benchmark with `bun run benchmark:webgpu`:
  - Smoke and normal streaming tests passed.
  - Graph capture failed because not all compute nodes were assigned to WebGPU.
  - This path is not viable for the production browser hot path.

Weight-only INT8 with `DequantizeLinear`:
- Manually quantized constant matmul weights to signed int8, per output channel.
- Inserted `291` `DequantizeLinear` nodes before fp32 `MatMul`.
- File size dropped from about `191 MiB` to about `49.2 MiB`.
- CPU accuracy:
  - `final_z` mean abs error: `0.0133`, max abs error: `0.1000`.
  - `candidate_k_entry` mean abs error: `0.0253`, max abs error: `2.0496`.
  - `candidate_v_entry` mean abs error: `0.0133`, max abs error: `0.4941`.
- Browser benchmark passed, including graph capture.
- WebGPU timing after graph-capture warmup:
  - Dynamics median: `34.28 ms`, p95: `34.80 ms`.
  - Streaming frame median: `38.43 ms`, p95: `38.86 ms`.
- This is slower than fp32 because the graph still does fp32 matmul and now also dequantizes weights in the runtime graph.

Comparison with fp32 baseline:
- Fp32 dynamics median: `25.99 ms`, p95 `26.43 ms`.
- Fp32 streaming median: `31.90 ms`, p95 `32.60 ms`.
- Weight-DQ INT8 is smaller on disk but slower at runtime.

Conclusion:
- INT8 is not useful for the current ORT WebGPU demo graph.
- Dynamic INT8 uses unsupported/non-WebGPU-partitioned operators for graph capture.
- Weight-DQ INT8 stays WebGPU compatible but adds dequantization work and regresses latency.
- INT4 b32 remains the only quantized variant that improved WebGPU runtime, but it is not accurate enough yet.

## 2026-05-04 KST: FP16 Weight-Only Storage With FP32 Compute

Goal:
- Test a safer weight-only compression path than INT8/INT4.
- Store dense weights as fp16, cast them back to fp32 in the ONNX graph, and keep activations plus matmul/Gemm compute in fp32.
- Keep the active demo artifact restored to fp32 after the trial.

Method:
- Converted the `291` dense-layer weights to fp16 initializers.
- Inserted one fp16-to-fp32 `Cast` per unique dense weight initializer:
  - `194` unique dense weights.
  - Direct `Gemm` variant: `291` `Gemm`, `194` `Cast`.
  - `Gemm -> MatMul` variant: `291` `MatMul`, `194` `Cast`.
- Both variants passed ONNX Runtime CPU execution and ORT WebGPU graph capture.

Size:
- Fp32 active graph: about `191 MiB` on disk, `200,322,981` bytes over HTTP.
- FP16-weight direct `Gemm`: about `96.0 MiB` on disk, `100,705,698` bytes over HTTP.
- FP16-weight `MatMul`: about `96.0 MiB` on disk, `100,690,535` bytes over HTTP.

Numerical drift against fp32:
- Both variants had the same CPU drift for the deterministic validation feed.
- `final_z`:
  - Mean abs error: `1.3271e-4`.
  - P95 abs error: `3.4848e-4`.
  - Max abs error: `8.2517e-4`.
- `candidate_k_entry`:
  - Mean abs error: `3.0646e-4`.
  - P95 abs error: `8.5173e-4`.
  - Max abs error: `6.9528e-3`.
- `candidate_v_entry`:
  - Mean abs error: `1.2516e-4`.
  - P95 abs error: `4.0520e-4`.
  - Max abs error: `3.2272e-3`.

WebGPU timing after graph-capture warmup:
- Fp32:
  - Dynamics median: `25.99 ms`, p95 `26.43 ms`.
  - Streaming median: `31.90 ms`, p95 `32.60 ms`.
- FP16-weight direct `Gemm`:
  - Dynamics median: `25.95 ms`, p95 `26.39 ms`.
  - Streaming median: `32.04 ms`, p95 `32.55 ms`.
- FP16-weight `MatMul`:
  - Dynamics median: `26.19 ms`, p95 `26.69 ms`.
  - Streaming median: `32.22 ms`, p95 `32.81 ms`.

Conclusion:
- FP16 weight-only storage is the best compression-only option tested so far.
- It cuts the dynamics step model payload roughly in half and preserves graph-capture compatibility.
- It does not improve hot-path latency because the graph still computes in fp32 and the weight casts do not reduce the fp32 matmul/Gemm work.
- The direct `Gemm` variant is preferable to the `MatMul` variant if we use this path, because it preserves the current graph structure and is slightly faster.
- Accuracy is much better than INT8/INT4, but it is not exactly equivalent to fp32 and exceeds the previous strict `5e-4` max-error tolerance on cache entries.

## 2026-05-04 KST: Q4F16 MatMulNBits Trial

Goal:
- Test the browser-LLM style path: packed INT4 weights through `MatMulNBits`, fp16 activations inside the dense kernels, and fp32 boundaries around the rest of the graph.
- Keep the active demo artifact restored to fp32 after the trial unless numerical validation is acceptable.

Method:
- Started from the accepted fp32 two-step dynamics step artifact:
  - `webgpu_app/assets/breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4.onnx`.
- Rewrote `291` `Gemm` nodes into equivalent `MatMul` plus optional `Add` nodes.
- Applied ORT `MatMulNBits` INT4 quantization with:
  - bits: `4`.
  - block size: `32`.
  - symmetric weights: enabled.
- Inserted fp16 activation/output casts around every `MatMulNBits`:
  - `Cast(fp32 -> fp16)` before the `MatMulNBits` input activation.
  - `Cast(fp16 -> fp32)` after the `MatMulNBits` output.
- Converted `194` scale initializers from fp32 to fp16 so WebGPU shader operands use the same fp16 data type inside `MatMulNBits`.

Graph size:
- Accepted fp32 step graph: about `191 MiB`.
- Full INT4 fp32-activation graph: `32,195,648` bytes.
- Q4F16 graph: `29,202,453` bytes.

CPU numerical validation against the accepted fp32 graph:
- Result: failed both `5e-4` and `1e-2` allclose checks.
- `final_z`:
  - Mean abs error: `0.0613`.
  - P95 abs error: `0.1550`.
  - Max abs error: `0.7848`.
- `candidate_k_entry`:
  - Mean abs error: `0.1344`.
  - P95 abs error: `0.4033`.
  - Max abs error: `2.7824`.
- `candidate_v_entry`:
  - Mean abs error: `0.0678`.
  - P95 abs error: `0.2429`.
  - Max abs error: `2.2024`.

Browser benchmark:
- Attempted the normal benchmark command: `bun run benchmark:webgpu`.
- The q4f16 benchmark did not reach model loading. Chromium crashed during startup with Crashpad permission errors before the page or ORT session was created.
- A fp32 run immediately before the q4f16 swap passed, so the current accepted fp32 artifact was restored and `latest.json` was restored to the fp32 result.

Conclusion:
- Q4F16 does not solve the accuracy problem. Its output drift is essentially in the same range as the earlier full INT4 b32 graph, meaning INT4 weight error dominates more than activation dtype.
- Since numerical validation fails by a large margin, this is not a candidate replacement for the demo graph even if a later browser run shows better latency.
- The useful result is that the expected q4f16 transformation is mechanically possible and shrinks the graph slightly more than fp32-activation INT4, but accuracy remains the blocker.

## 2026-05-05 KST: Cache-Length Entry Graph For 4-Frame Demo Prefix

Goal:
- Replace the browser demo's two dynamics sessions with one entry-cache graph that accepts the logical `cache_length`.
- Keep the physical K/V cache at the fixed 64-slot shape, but only attend to committed slots plus the current generated token.
- Use the first 4 real frames as prefix, then fill slots 4 through 63 before switching to slide/rebase updates.

Changes:
- Added a cached append-context entry export:
  - `breakout_dynamics_sample_append_context_cache_length_entry_b1_t1_s4.onnx`.
  - Inputs: `sample_noise`, `context_noise`, `actions`, `k_cache`, `v_cache`, `cache_length`.
  - Outputs: `final_z`, `candidate_k_entry`, `candidate_v_entry`, `candidate_cache_length`.
- Fixed the cached temporal attention mask for partial caches:
  - Old mask allowed indices `< cache_length + 1`, which accidentally masked out the appended current token because the current K/V is concatenated at the fixed final slot.
  - New mask allows committed cache slots `< cache_length` plus the appended current slot at `context_length`.
- Updated the demo cache updater:
  - Fill mode writes the one returned K/V entry into `slot = cache_length`.
  - Slide mode shifts/rebases once `cache_length == 64`.
- Regenerated `breakout_demo_initial_cache.*` from the first 4 prefix frames. The manifest reports `cache_length = 4`.

Validation:
- Export-time ONNX validation passed for the new graph.
- CPU comparison at `cache_length = 4` against the full candidate-cache append graph:
  - `candidate_cache_length`: `[5]` vs `[5]`.
  - `final_z` compared to the rewritten graph's `pred_z` alias: max abs `0.0`.
  - `candidate_k_entry` vs old graph slot 4: max abs `4.1127e-6`.
  - `candidate_v_entry` vs old graph slot 4: max abs `4.8280e-6`.
- Demo smoke:
  - `bun run demo:webgpu:smoke`: passed, 2 tests.
- Benchmark:
  - `bun run benchmark:webgpu`: passed after graph capture was classified as blocked when ORT reports that not all nodes partitioned to WebGPU.
  - The graph-capture result is blocked for this artifact, not a numerical or normal streaming failure.

Conclusion:
- This is the correct cache ABI for the public demo: one fixed GPU cache, a logical length scalar, and per-frame K/V entry updates.
- It removes the need to download and instantiate both the fill full-cache graph and the steady-state entry graph for the website.

## 2026-05-06 KST: Small Checkpoint Fast Path Restore

Goal:
- Re-export the new smaller tokenizer/dynamics checkpoints while keeping the browser rollout fast.
- Preserve the 4-frame prefix cache-length demo behavior, but use the full-cache steady-state graph once the cache reaches 64 slots.

Issue found:
- The small checkpoint uses smaller attention head dimensions:
  - tokenizer head dim: `8`.
  - dynamics head dim: `32`.
- Two post-export rewrites still assumed the previous larger model head dim:
  - GQA repeat rewrite expected repeated K/V heads ending in `64`.
  - head projection rewrite expected `head_dim == 64`.
- Because those rewrites missed the small model, the hot graph kept `Expand` and `Reshape` nodes. ORT WebGPU could not capture the full graph and normal inference regressed to about `121 ms/frame`.

Fix:
- Generalized the GQA repeat rewrite to infer `kv_heads`, repeat count, and `head_dim` from static shapes.
- Generalized the head projection layout rewrite to infer `head_count` and `head_dim` instead of requiring `64`.
- Kept the demo/runtime policy:
  - use cache-length entry graph while filling slots 4 through 63.
  - use steady-state slide entry graph after the cache is full.

Validation:
- Re-exported:
  - `tokenizer_small`
  - `dynamics_small`
  - `--sample_steps 2`
- Export-time ONNX validation passed.
- Entry-cache reconstruction passed:
  - K cache max abs error: `4.0531e-6`.
  - V cache max abs error: `1.0729e-6`.
  - `final_z` max abs error: `1.1921e-7`.
- Hot graph counts after export:
  - `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s2.onnx`: `Expand=0`, `Reshape=0`.
  - `breakout_dynamics_sample_append_context_cache_length_entry_b1_t1_s2.onnx`: `Expand=0`, `Reshape=0`.
  - `breakout_tokenizer_decode_z_b1_t1.onnx`: `Expand=0`, `Reshape=0`.
- Browser checks:
  - `bun run typecheck`: passed.
  - `bun run build:webgpu:browser`: passed.
  - `bun run demo:webgpu:smoke`: passed.
  - `bun run benchmark:webgpu`: passed, including graph capture.

Benchmark:
- Normal WebGPU benchmark:
  - Dynamics mean/median/p95: `15.61 / 15.49 / 17.08 ms`.
  - Decoder mean/median/p95: `2.59 / 2.57 / 2.85 ms`.
  - Streaming frame mean/median/p95: `18.24 / 18.14 / 19.79 ms`.
  - Throughput: `54.83 fps`.
- Graph-capture benchmark:
  - Dynamics after warmup mean/median/p95: `8.02 / 10.25 / 11.27 ms`.
  - Decoder after warmup mean/median/p95: `1.80 / 2.22 / 2.90 ms`.
  - Streaming frame after warmup mean/median/p95: `10.07 / 12.90 / 13.97 ms`.
  - Throughput: `99.30 fps`.

Conclusion:
- The slowdown was not caused by the cache-length demo logic itself. It came from small-model shape assumptions in post-export graph rewrites.
- The small checkpoint is now faster than the previous accepted path and comfortably meets the live demo target.

Follow-up: restored one-dynamics-artifact demo contract.
- The temporary two-dynamics setup loaded:
  - cache-length entry graph for filling a short prefix cache.
  - slide-entry graph after the cache reached 64 slots.
- That recovered speed, but it was the wrong deployment contract because it made the website download and compile a second dynamics model.
- The demo and benchmark now use only:
  - `breakout_dynamics_sample_append_context_cache_length_entry_b1_t1_s2.onnx`.
- The browser derives `sample_position_index`, `context_position_index`, and `attention_mask` from the logical cache length each frame.
- The physical K/V cache remains fixed-size, and the runtime writes the returned K/V entry into the fill/slide slot.
- Validation after reverting to one dynamics artifact:
  - `bun run typecheck`: passed.
  - `bun run build:webgpu:browser`: passed.
  - `bun run demo:webgpu:smoke`: passed.
  - `bun run benchmark:webgpu`: passed, including graph capture.
- Benchmark:
  - Normal streaming frame mean/median/p95: `20.34 / 20.16 / 21.80 ms`.
  - Normal throughput: `49.17 fps`.
  - Graph-capture streaming after warmup mean/median/p95: `11.38 / 13.09 / 18.61 ms`.
  - Graph-capture throughput after warmup: `87.89 fps`.

Follow-up: pruned retired browser export paths.
- Removed runtime and benchmark artifact fallback lists for the old slide/full-cache/layer variants.
- The public demo now resolves only:
  - `breakout_dynamics_sample_append_context_cache_length_entry_b1_t1_s2.onnx`.
  - `breakout_tokenizer_decode_z_b1_t1.onnx`.
  - precomputed context/cache artifacts.
- The benchmark still keeps `breakout_dynamics_prefill_cached_b1_t64.onnx` only for the benchmark-only prefill timing mode.
- Removed `scripts/webgpu/verify_entry_cache_update.py`; it validated the retired slide-entry vs full-cache path and is no longer part of the active demo path.
- Validation after cleanup:
  - `bun run typecheck`: passed.
  - `bun run build:webgpu:browser`: passed.
  - `bun run demo:webgpu:smoke`: passed.
  - `bun run benchmark:webgpu`: passed, including graph capture.
- Latest normal streaming frame mean/median/p95: `21.12 / 21.09 / 22.35 ms`.

## 2026-05-10 KST: Curated Breakout Assets And Full-Cache Entry Specialization

Issue found:
- The root `webgpu_app/assets` manifest was stale for this run: it described a Space Invaders
  checkpoint while the demo page and benchmark were being treated as Breakout.
- The maintained Breakout assets are the curated small-checkpoint artifacts under
  `webgpu_app/dream_arcade_assets/breakout`, with dynamics/tokenizer checkpoint step `1000000`.
- Leading shell environment assignments before `bun run ...` are unreliable in this environment, so
  benchmark controls now use wrapper flags after `--`.

Runtime/tooling changes:
- Changed the Breakout demo and benchmark defaults to
  `/webgpu_app/dream_arcade_assets/breakout`.
- Added wrapper flags in `scripts/webgpu/run_playwright_chrome_home.ts` for benchmark controls such
  as `--webgpu-benchmark-asset-base`, `--webgpu-benchmark-step-artifact`, and
  `--playwright-benchmark-attempts`.
- Made the demo request WebGPU graph capture by default, but only run captured dynamics after
  `cache_length >= 64`. Using captured dynamics during the partial-cache fill produced repeated
  canvas frames, and the smoke test caught it.
- Restored a two-step runtime policy for the demo:
  - cache-length entry graph while filling slots `4..63`.
  - full-cache specialized entry graph after the cache is full.

Full-cache entry artifact:
- Added `scripts/webgpu/specialize_full_cache_entry.py`.
- The script starts from
  `breakout_dynamics_sample_append_context_cache_length_entry_b1_t1_s2.onnx`, fixes
  `sample_position_index=64`, `context_position_index=63`, and the full attention mask, folds
  constants, applies the existing WebGPU rotary rewrite, then runs ORT extended optimization.
- Generated:
  `breakout_dynamics_sample_append_context_full_cache_entry_b1_t1_s2.onnx`.
- Size: `22,936,833` bytes, compared with `23,057,120` bytes for the cache-length entry graph.
- Node count from the specialization run: `4420 -> 4053`.
- SHA-256:
  `fcaf8490b7e2fe0b75473d5f4772e802a3688b77fd99340b4ec26947438b8601`.

Validation:
- CPU validation against the original cache-length graph at full cache passed with
  `atol=5e-4`, `rtol=5e-4`.
- Max / mean absolute errors:
  - `final_z`: `3.847e-05 / 9.24e-07`.
  - `candidate_k_entry`: `7.200e-05 / 1.12e-06`.
  - `candidate_v_entry`: `4.482e-05 / 5.20e-07`.
- Browser/code checks:
  - `uv run python -m py_compile scripts/webgpu/specialize_full_cache_entry.py`: passed.
  - `bun run typecheck`: passed.
  - `bun run build:webgpu:browser`: passed.
  - `bun run demo:webgpu:smoke`: passed, 5 tests.
  - `bun run benchmark:webgpu -- --grep @smoke --playwright-benchmark-attempts 5`: passed.
  - `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 5`: passed.

Benchmark results on the curated Breakout assets:
- Normal WebGPU smoke result (`webgpu_app/bench/results/latest.json`):
  - Dynamics mean/median/p95: `32.64 / 32.68 / 34.17 ms`.
  - Decoder mean/median/p95: `4.26 / 4.22 / 4.51 ms`.
  - Streaming frame mean/median/p95: `36.94 / 36.99 / 38.79 ms`.
  - Throughput: `27.07 fps`.
- Graph-capture result (`webgpu_app/bench/results/graph_capture_latest.json`):
  - Dynamics after graph-capture warmup mean/median/p95: `21.56 / 22.81 / 23.33 ms`.
  - Decoder after graph-capture warmup mean/median/p95: `3.21 / 3.21 / 3.84 ms`.
  - Streaming frame after graph-capture warmup mean/median/p95: `25.10 / 26.52 / 26.95 ms`.
  - Throughput after graph-capture warmup: `39.84 fps`.

Operational note:
- Chromium startup still intermittently failed before page/model loading with Crashpad permission
  errors under Playwright. Adding `--no-zygote` did not eliminate the issue, but rerunning through
  the wrapper eventually produced valid benchmark results.

Conclusion:
- The full-cache specialized artifact is numerically valid and graph-capture compatible, but it does
  not reach 60 FPS on this measured run.
- The current bottleneck remains the fp32 dynamics graph compute/dispatch cost; graph capture plus
  fixed full-cache inputs gets the steady stream to roughly `25-27 ms/frame`, not the `<=16.67 ms`
  needed for 60 FPS.
- The earlier May 6 graph-capture result around `11 ms/frame` was not reproduced with the curated
  Breakout assets and longer steady-state samples in this run.

### Packed Full-Cache Follow-Up

Change tried:
- Generalized `pack_sibling_gemms_for_webgpu` so it matches the small Dream Arcade checkpoint
  widths instead of only the older large-model widths.
- Reused packed initializers by `(kind, weight initializer tuple)` so repeated packed QKV/SwiGLU
  patterns do not duplicate weight blobs.
- Made `scripts/webgpu/specialize_full_cache_entry.py` pack the specialized full-cache entry graph
  by default.
- Packed the single-frame tokenizer decoder in place.

Generated artifacts:
- Full-cache dynamics:
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2.onnx`.
- Dynamics size: `23,010,850` bytes.
- Dynamics SHA-256:
  `8866517393f8a19342482c420a83e9c466845cf51689a1d695307f1d8cbd72b3`.
- Dynamics node/op counts:
  - cache-length source: `4420` nodes, `504` Gemm, `357` Split.
  - unpacked full-cache: `4053` nodes, `504` Gemm, `286` Split.
  - packed full-cache: `3982` nodes, `291` Gemm, `428` Split.
- Decoder node/op counts after packing: `485` nodes, `34` Gemm, `49` Split. The decoder Gemm
  count dropped from `58` to `34`.

Validation:
- Packed dynamics CPU validation against the cache-length graph at full cache passed with
  `atol=5e-4`, `rtol=5e-4`.
- Max / mean absolute errors:
  - `final_z`: `3.8474798e-05 / 9.243740e-07`.
  - `candidate_k_entry`: `7.200241e-05 / 1.120458e-06`.
  - `candidate_v_entry`: `4.482269e-05 / 5.202440e-07`.
- Direct source-vs-packed CPU validation for the dynamics packing pass was exact on the checked
  outputs before full-cache specialization.
- Packed decoder CPU validation against the prior decoder was exact on `patches`.

Browser/code checks:
- `uv run --no-cache python -m py_compile scripts/webgpu/export_dreamer4_onnx.py scripts/webgpu/specialize_full_cache_entry.py`: passed.
- `bun run typecheck`: passed.
- `bun run build:webgpu:browser`: passed.
- `bun run demo:webgpu:build`: passed.
- `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 10`: passed, 5 tests.
- `bun run benchmark:webgpu -- --grep @smoke --playwright-benchmark-attempts 10`: passed.
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 10`: passed.

Fresh benchmark results on the packed curated Breakout assets:
- Normal WebGPU smoke result (`webgpu_app/bench/results/latest.json`, created
  `2026-05-10T14:24:34.258Z`):
  - Dynamics mean/median/p95: `30.38 / 30.11 / 32.20 ms`.
  - Decoder mean/median/p95: `4.10 / 4.03 / 4.68 ms`.
  - Streaming frame mean/median/p95: `34.52 / 34.28 / 36.51 ms`.
  - Throughput: `28.97 fps`.
- Graph-capture result (`webgpu_app/bench/results/graph_capture_latest.json`, created
  `2026-05-10T14:24:41.531Z`):
  - Dynamics after graph-capture warmup mean/median/p95: `19.17 / 20.43 / 21.02 ms`.
  - Decoder after graph-capture warmup mean/median/p95: `3.00 / 3.09 / 3.74 ms`.
  - Streaming frame after graph-capture warmup mean/median/p95: `22.44 / 24.00 / 24.43 ms`.
  - Throughput after graph-capture warmup: `44.57 fps`.

Negative result:
- A full-cache slide graph copied from a separate full export was slower than the entry-output graph:
  graph-capture streaming after warmup was `27.53 / 29.10 / 29.43 ms`, or `36.33 fps`.
- Do not replace the entry-output graph with that slide graph path.

Conclusion:
- Small-checkpoint QKV/SwiGLU packing is valid and gives a measurable graph-capture improvement
  over the unpacked full-cache entry path (`39.84 fps` -> `44.57 fps` reported).
- It is still short of the 60 FPS target. The consistent frame distribution is closer to the
  `24 ms` median/p95 range, so the next optimization needs a larger reduction in dynamics dispatch
  or compute cost rather than only decoder/cache plumbing.

### Packed QKV Partial-Head Split Follow-Up

Change tried:
- Generalized `rewrite_packed_qkv_split_partial_heads_for_webgpu` so it handles the small checkpoint
  QKV widths: parent packed QKV splits of `(64, 256, 64)` and two-way K/V child head splits of
  `(32, 32)`.
- Added that rewrite to `scripts/webgpu/specialize_full_cache_entry.py` after sibling Gemm packing.
- Applied the same constrained K/V split rewrite to the packed single-frame tokenizer decoder.
- Added a benchmark wrapper flag for `--webgpu-benchmark-graph-optimization-level` so `basic`,
  `extended`, and `all` can be compared without leading shell environment assignments.

Validation:
- Full-cache dynamics CPU validation against the cache-length graph at full cache passed with
  `atol=5e-4`, `rtol=5e-4`.
- Max / mean absolute errors stayed unchanged:
  - `final_z`: `3.8474798e-05 / 9.243740e-07`.
  - `candidate_k_entry`: `7.200241e-05 / 1.120458e-06`.
  - `candidate_v_entry`: `4.482269e-05 / 5.202440e-07`.
- Decoder CPU validation against the pre-rewrite packed decoder was exact on `patches`.

Generated artifact changes:
- Dynamics SHA-256:
  `ae82d8187689fab0a905c3defe7187c54baf19538a164bf8f14997bb33893b71`.
- Dynamics size: `22,995,592` bytes.
- Dynamics node/op counts:
  - before partial-head split rewrite: `3982` nodes, `428` Split.
  - after constrained K/V split rewrite: `3840` nodes, `286` Split.
  - rewrite count: `71` packed QKV parent splits, `142` removed child splits.
- Decoder SHA-256:
  `f84bead9f54da56ab99d1d08a4959900d29004b3accf766a984c89a3bf5cac9b`.
- Decoder size: `8,059,969` bytes.
- Decoder node/op counts:
  - before partial-head split rewrite: `485` nodes, `49` Split.
  - after constrained K/V split rewrite: `469` nodes, `33` Split.
  - rewrite count: `8` packed QKV parent splits, `16` removed child splits.

Negative result:
- A more aggressive version also inlined the 8-way Q head split into the parent QKV Split, reducing
  dynamics to `3769` nodes and `215` Split nodes.
- Browser WebGPU rejected that graph at runtime with `Too many storage buffers in shader`
  (`Current: 11, Max is 10`), so only the two-way K/V child splits are inlined.
- `graphOptimizationLevel=extended` and `graphOptimizationLevel=all` did not improve the default
  graph-capture path. On the packed graph before the split rewrite they measured `44.76 fps` and
  `44.52 fps` respectively, versus `44.57 fps` for `basic`. On the split-rewritten graph,
  `extended` measured `44.99 fps` versus `45.48 fps` for `basic`.
- `preferredLayout=NHWC` also did not improve the split-rewritten graph-capture path, measuring
  `45.83 fps` versus the default-layout run at `46.60 fps`.

Browser/code checks:
- `uv run --no-cache python -m py_compile scripts/webgpu/export_dreamer4_onnx.py scripts/webgpu/specialize_full_cache_entry.py`: passed.
- `bun run typecheck`: passed.
- `bun run build:webgpu:browser`: passed.
- `bun run demo:webgpu:build`: passed.
- `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 10`: passed, 5 tests.
- `bun run benchmark:webgpu -- --grep @smoke --playwright-benchmark-attempts 10`: passed.
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 10`: passed.

Fresh benchmark results after the constrained partial-head split rewrite:
- Normal WebGPU smoke result (`webgpu_app/bench/results/latest.json`, created
  `2026-05-10T14:40:03.475Z`):
  - Dynamics mean/median/p95: `29.22 / 28.96 / 31.28 ms`.
  - Decoder mean/median/p95: `4.00 / 3.90 / 4.90 ms`.
  - Streaming frame mean/median/p95: `33.25 / 32.99 / 35.39 ms`.
  - Throughput: `30.07 fps`.
- Graph-capture result (`webgpu_app/bench/results/graph_capture_latest.json`, created
  `2026-05-10T14:41:08.171Z`):
  - Dynamics after graph-capture warmup mean/median/p95: `18.42 / 20.02 / 20.36 ms`.
  - Decoder after graph-capture warmup mean/median/p95: `2.74 / 2.90 / 2.99 ms`.
  - Streaming frame after graph-capture warmup mean/median/p95: `21.46 / 23.26 / 23.57 ms`.
  - Throughput after graph-capture warmup: `46.60 fps`.

Conclusion:
- Constrained partial-head split rewriting is valid and improves both normal and graph-capture
  benchmark paths, but the gain is modest (`44.57 fps` -> `46.60 fps` reported with graph capture).
- The remaining gap to 60 FPS is still in the dynamics graph; even after graph capture, the median
  frame is about `23.2 ms`, so another `6+ ms` needs to come out of the steady-state frame path.

### Demo Runtime Allocation Reuse Follow-Up

Change tried:
- Added `NormalNoiseGenerator.fillTensorData()` so the demo can refill existing noise buffers
  instead of allocating a new `Float32Array` for every random tensor.
- Reused per-step CPU input tensors for `sample_noise`, `context_noise`, and `actions` in the live
  demo frame loop. The tensors are refilled before each ORT run, then uploaded into the fixed graph
  capture GPU input buffers.
- Reused the display `ImageData` backing object for tokenizer patches before calling
  `putImageData()`.
- Kept the model inputs, sample-step count, graph capture policy, and cache update policy unchanged.

Validation:
- `bun run typecheck`: passed.
- `bun run build:webgpu:browser`: passed.
- `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 10`: passed, 5 tests.

Measurement note:
- This is a live-demo bridge/allocation cleanup, so the model benchmark does not capture most of its
  benefit. A direct ad hoc Playwright measurement outside the wrapper hit the same intermittent
  Chrome Crashpad launch failure seen earlier; the wrapper-based demo smoke remains the runtime
  guard for this change.

### Additional Rejected Trials After Partial-Head Split

Static head-merge `Reshape` trial:
- Replaced `36` static `Split -> Concat -> Squeeze` head-merge islands in a temporary full-cache
  dynamics graph with constant-shape `Reshape` nodes.
- CPU validation against the accepted packed full-cache graph was exact for the checked outputs.
- Browser graph capture rejected the graph because not all nodes partitioned to the WebGPU execution
  provider.
- Conclusion: reject. Even static `Reshape` is not viable in this graph-capture path.

Packed QKV head `Einsum` trial:
- Replaced `36` current-layout packed QKV projection groups with rank-aware head `Einsum` nodes in a
  temporary artifact.
- Graph changes: `3444` nodes, `255` Gemm, `178` Einsum, `250` Split, `849` Unsqueeze, `253`
  Concat, `341` Squeeze, `285` Transpose.
- CPU validation against the accepted packed full-cache graph was exact for the checked outputs.
- Graph capture passed, but was slower:
  - Dynamics after graph-capture warmup: `20.17 / 21.98 / 22.90 ms`.
  - Decoder after graph-capture warmup: `2.11 / 2.24 / 2.95 ms`.
  - Streaming frame after graph-capture warmup: `22.35 / 24.36 / 25.37 ms`.
  - Throughput after graph-capture warmup: `44.75 fps`.
- Conclusion: reject. Removing layout nodes by replacing packed `Gemm` with `Einsum` loses more in
  dynamics compute than it saves in dispatch count.

Graph-capture preallocated-output retest:
- Retested enabling preallocated GPU output tensors for graph-capture benchmark runs on the current
  packed full-cache entry artifact.
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 5` failed
  consistently with the same ORT WebGPU error seen in the earlier trial:
  `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`.
- Conclusion: still reject. Keep preallocated hot outputs disabled for graph-capture runs.

ONNX Runtime Web `1.26.0` trial:
- The registry reports `onnxruntime-web@1.26.0`; the repo remains on `1.24.3`.
- Installed `1.26.0` temporarily using Bun's `--cache-dir` flag to avoid the sandbox tempdir issue.
- `bun run typecheck`: passed.
- `bun run build:webgpu:browser`: passed.
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 5`: passed,
  but regressed performance:
  - Dynamics after graph-capture warmup: `19.78 / 20.78 / 21.42 ms`.
  - Decoder after graph-capture warmup: `2.80 / 2.94 / 3.34 ms`.
  - Streaming frame after graph-capture warmup: `22.86 / 24.05 / 24.73 ms`.
  - Throughput after graph-capture warmup: `43.74 fps`.
- Restored `onnxruntime-web@1.24.3` and the original package version range.

Restored baseline check:
- After reverting the rejected preallocation and runtime-version trials:
  - `bun run typecheck`: passed.
  - `bun run build:webgpu:browser`: passed.
  - `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 5`: passed.
- Restored graph-capture result (`webgpu_app/bench/results/graph_capture_latest.json`, created
  `2026-05-10T15:02:44.669Z`):
  - Dynamics after graph-capture warmup: `18.96 / 20.41 / 20.94 ms`.
  - Decoder after graph-capture warmup: `2.85 / 2.91 / 3.57 ms`.
  - Streaming frame after graph-capture warmup: `22.10 / 23.77 / 24.17 ms`.
  - Throughput after graph-capture warmup: `45.25 fps`.

Current conclusion:
- The best observed accepted graph-capture result in this round remains the earlier split-rewritten
  run at `46.60 fps`, with a streaming median/p95 around `23.26 / 23.57 ms`.
- The current graph is still dynamics-bound and short of consistent 60 FPS without a larger
  attention/MLP execution-path change.

### Zero Softmax Bias Add Prune

Change accepted:
- Added `remove_zero_softmax_bias_adds_for_webgpu()` and wired it into the full-cache
  specialization script after packed QKV/SwiGLU and partial-head split rewrites.
- The rewrite only removes an `Add` immediately before `Softmax` when exactly one Add input is an
  all-zero initializer and the Add output has a single Softmax consumer.
- In the full-cache specialized graph, all `71` attention bias/mask Adds matched because the
  full-cache attention masks fold to zero bias.

Validation:
- Regenerated
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2.onnx`.
- CPU validation against the cache-length graph at full cache passed with `atol=5e-4`,
  `rtol=5e-4`.
- Max / mean absolute errors stayed unchanged:
  - `final_z`: `3.8474798e-05 / 9.243740e-07`.
  - `candidate_k_entry`: `7.200241e-05 / 1.120458e-06`.
  - `candidate_v_entry`: `4.482269e-05 / 5.202440e-07`.

Generated artifact changes:
- Dynamics SHA-256:
  `8db1234aee11154f8aecadd14289bf6194f9f16958dcc911c2d945719b11d120`.
- Dynamics size: `22,981,168` bytes.
- Dynamics node/op counts:
  - before zero-bias Add prune: `3840` nodes, `147` Add.
  - after zero-bias Add prune: `3769` nodes, `76` Add.
  - removed `71` `Add -> Softmax` nodes.

Browser/code checks:
- `uv run --no-cache python -m py_compile scripts/webgpu/export_dreamer4_onnx.py scripts/webgpu/specialize_full_cache_entry.py`: passed.
- `bun run typecheck`: passed.
- `bun run build:webgpu:browser`: passed.
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 5`: passed.
- `bun run benchmark:webgpu -- --grep @smoke --playwright-benchmark-attempts 5`: passed.
- `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 5`: passed, 5 tests.

Fresh benchmark results after zero-bias Add prune:
- Normal WebGPU smoke result (`webgpu_app/bench/results/latest.json`, created
  `2026-05-10T15:12:31.694Z`):
  - Dynamics mean/median/p95: `28.84 / 28.66 / 31.39 ms`.
  - Throughput: `30.32 fps`.
- Graph-capture result (`webgpu_app/bench/results/graph_capture_latest.json`, created
  `2026-05-10T15:12:12.345Z`):
  - Dynamics after graph-capture warmup mean/median/p95: `18.10 / 19.58 / 20.11 ms`.
  - Decoder after graph-capture warmup mean/median/p95: `2.92 / 3.11 / 3.57 ms`.
  - Streaming frame after graph-capture warmup mean/median/p95: `21.25 / 23.00 / 23.69 ms`.
  - Throughput after graph-capture warmup: `47.06 fps`.

Conclusion:
- Accept. This is an exact full-cache graph cleanup and gives a small but measurable graph-capture
  improvement over the previous best reported `46.60 fps`.
- The graph remains dynamics-bound and still misses consistent 60 FPS; the median frame is now about
  `23.0 ms`, so roughly another `6.3 ms` must come out of the steady frame path.

Follow-up:
- This zero-bias-only artifact was superseded by the attention-scale folding artifact below, which
  keeps the zero-bias Add prune and removes the attention score scale Mul nodes as well.

### Generalized Attention Scale Folding

Change accepted:
- Generalized `fold_attention_scale_into_query_norm_for_webgpu()` so it no longer only matches the
  older `0.125` attention scale.
- The current small Dream Arcade full-cache graph uses scale `0.1767766922712326`, so the previous
  pass left `71` logits-sized `Mul` nodes before `Softmax`.
- Wired the generalized scale fold into `scripts/webgpu/specialize_full_cache_entry.py` before the
  zero-bias Add prune.
- The pass scales each matched query RMSNorm weight and bypasses the corresponding attention score
  `Mul`. This preserves `(q * k) * scale == (q * scale) * k`.

Validation:
- Regenerated
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2.onnx`.
- CPU validation against the cache-length graph at full cache passed with `atol=5e-4`,
  `rtol=5e-4`.
- Max / mean absolute errors:
  - `final_z`: `2.3514032e-05 / 7.052768e-07`.
  - `candidate_k_entry`: `4.4345856e-05 / 9.355436e-07`.
  - `candidate_v_entry`: `2.8848648e-05 / 4.194887e-07`.

Generated artifact changes:
- Dynamics SHA-256:
  `c514d932ba978e5be2f73da9f5f865bd3f70c2732cefdbd14e6ad771058afcd8`.
- Dynamics size: `22,976,866` bytes.
- Dynamics node/op counts after both scale folding and zero-bias Add pruning:
  - nodes: `3698`.
  - `Mul`: `145 -> 74`.
  - `Add`: `147 -> 76`.
  - `Softmax`: unchanged at `71`.
  - `Einsum`: unchanged at `142`.
- Rewrite counts:
  - attention score Mul removed: `71`.
  - query RMSNorm initializers scaled: `71`.
  - zero-bias Add prunes: `71`.

Browser/code checks:
- `uv run --no-cache python -m py_compile scripts/webgpu/export_dreamer4_onnx.py scripts/webgpu/specialize_full_cache_entry.py`: passed.
- `bun run typecheck`: passed.
- `bun run build:webgpu:browser`: passed.
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 5`: passed.
- `bun run benchmark:webgpu -- --grep @smoke --playwright-benchmark-attempts 5`: passed.
- `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 5`: passed, 5 tests.
- `bun run demo:webgpu:build`: passed.

Fresh benchmark results after generalized attention scale folding:
- Normal WebGPU smoke result (`webgpu_app/bench/results/latest.json`, created
  `2026-05-10T15:15:47.771Z`):
  - Dynamics mean/median/p95: `28.37 / 28.12 / 30.73 ms`.
  - Decoder mean/median/p95: `4.13 / 4.06 / 4.61 ms`.
  - Throughput: `30.72 fps`.
- Graph-capture result (`webgpu_app/bench/results/graph_capture_latest.json`, created
  `2026-05-10T15:15:25.777Z`):
  - Dynamics after graph-capture warmup mean/median/p95: `16.25 / 17.82 / 18.51 ms`.
  - Decoder after graph-capture warmup mean/median/p95: `4.60 / 4.86 / 5.56 ms`.
  - Streaming frame after graph-capture warmup mean/median/p95: `21.12 / 23.08 / 23.85 ms`.
  - Throughput after graph-capture warmup: `47.35 fps`.
- Repeat after restoring a rejected decoder-capture-off benchmark experiment also passed, with
  graph-capture streaming frame after warmup at `21.30 / 23.09 / 23.96 ms`, or `46.94 fps`.

Rejected follow-up:
- Disabling graph capture for the decoder session while keeping dynamics graph capture enabled was
  slower: graph-capture streaming after warmup was `21.42 / 22.98 / 25.03 ms`, or `46.68 fps`.
- Keep decoder graph capture enabled in the benchmark path.

Conclusion:
- Accept. Dynamics alone now reaches about `61.5 fps` by mean after graph-capture warmup, but the
  full frame is still held to `47.35 fps` by decoder/cache/render serialization.
- The next useful target is the post-dynamics frame tail: decoder timing, cache-update ordering, or
  live-demo CPU readback/rendering.

### Decoder Attention Scale And Zero-Bias Cleanup

Change accepted:
- Applied the generalized attention scale folding pass to
  `breakout_tokenizer_decode_z_b1_t1.onnx`.
- Applied the zero-bias `Add -> Softmax` prune to the decoder as well.
- Wired `remove_zero_softmax_bias_adds_for_webgpu()` into the general export pipeline so future
  full exports can reproduce this decoder cleanup.

Validation:
- Compared the rewritten decoder against the previous packed/partial-head-split decoder on CPU with
  deterministic random `z`.
- `patches` allclose passed at `atol=5e-4`, `rtol=5e-4`.
- Max / mean absolute error: `6.7353249e-06 / 5.435383e-08`.

Generated artifact changes:
- Decoder SHA-256:
  `206de41b8a965aaab190888b303495532224488feba415c055bacc3dd7fe5b56`.
- Decoder size: `8,011,928` bytes.
- Decoder node/op counts:
  - before: `469` nodes, `16` Mul, `12` Add.
  - after: `459` nodes, `8` Mul, `10` Add.
  - attention score Mul removed: `8`.
  - query RMSNorm initializers scaled: `8`.
  - zero-bias Add prunes: `2`.

Browser/code checks:
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 5`: passed.
- `bun run benchmark:webgpu -- --grep @smoke --playwright-benchmark-attempts 5`: passed.
- `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 5`: passed, 5 tests.
- `bun run demo:webgpu:build`: passed.

Fresh benchmark results after decoder cleanup:
- Normal WebGPU smoke result (`webgpu_app/bench/results/latest.json`, created
  `2026-05-10T15:20:17.602Z`):
  - Dynamics mean/median/p95: `28.08 / 27.98 / 30.50 ms`.
  - Decoder mean/median/p95: `4.01 / 3.89 / 4.62 ms`.
  - Throughput: `31.12 fps`.
- Graph-capture result (`webgpu_app/bench/results/graph_capture_latest.json`, created
  `2026-05-10T15:19:19.397Z`):
  - Dynamics after graph-capture warmup mean/median/p95: `16.71 / 18.53 / 18.85 ms`.
  - Decoder after graph-capture warmup mean/median/p95: `3.20 / 3.41 / 3.77 ms`.
  - Streaming frame after graph-capture warmup mean/median/p95: `20.34 / 22.41 / 22.93 ms`.
  - Throughput after graph-capture warmup: `49.17 fps`.

Conclusion:
- Accept. The full streaming frame improved from the previous best `47.35 fps` to `49.17 fps`.
- The remaining gap to consistent 60 FPS is now roughly `5.7 ms` at the median frame level.

### Latent Decoder Input

Change accepted:
- Added `breakout_tokenizer_decoder_b1_t1.onnx` to the curated Dream Arcade Breakout assets.
- The new decoder takes `latent: [1,1,64,16]` instead of `z: [1,1,32,32]`.
- This is an exact reinterpretation for the current dynamics output: `[1,1,32,32]` and
  `[1,1,64,16]` have identical row-major storage order.
- The artifact was derived from the optimized `breakout_tokenizer_decode_z_b1_t1.onnx` by removing
  the initial `Split/Slice/Concat` decode-z packing island.
- The benchmark and demo now prefer `manifest.demo_generation.preferred_decoder_export`, then
  `breakout_tokenizer_decoder_b1_t1`, and fall back to `breakout_tokenizer_decode_z_b1_t1`.
- WebGPU runtime code copies the same GPU bytes into a fixed latent-shaped decoder input when the
  selected decoder does not expose `z`.

Validation:
- CPU comparison against the optimized `decode_z` decoder passed at `atol=5e-4`, `rtol=5e-4`.
- Observed `patches` max / mean absolute error: `0.0 / 0.0`.

Generated artifact changes:
- Latent decoder SHA-256:
  `54581c2b852dbea7b9249e52b271e1255c43cfa494a2091070a5387991433e45`.
- Latent decoder size: `7,931,443` bytes.
- Decoder node/op counts:
  - `decode_z`: `459` nodes, including `65` Slice and `34` Concat.
  - latent decoder: `393` nodes, with the decode-z packing island removed.

Rejected runtime follow-ups:
- A `vec4<f32>` browser cache-update shader compiled and passed graph capture, but was slightly
  slower: graph-capture streaming after warmup measured `20.50 / 22.51 / 23.03 ms`, or
  `48.79 fps`. Restored the scalar in-place slide/rebase shader.
- Reapplying the temporal BHSD attention rewrite after full-cache specialization was exact on CPU
  but slower in Chrome: graph-capture streaming after warmup measured `20.63 / 22.79 / 23.00 ms`,
  or `48.46 fps`. Restored the previous dynamics artifact.

Browser/code checks:
- `bun run typecheck`: passed.
- `bun run build:webgpu:browser`: passed.
- `bun run demo:webgpu:build`: passed.
- `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 5`: passed, 5 tests.
- `bun run benchmark:webgpu -- --grep @smoke --playwright-benchmark-attempts 5`: passed.
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 5`: passed.

Fresh benchmark results after latent decoder selection:
- Normal WebGPU smoke result (`webgpu_app/bench/results/latest.json`, created
  `2026-05-10T15:39:50.386Z`):
  - Dynamics mean/median/p95: `27.72 / 27.79 / 29.81 ms`.
  - Decoder mean/median/p95: `3.04 / 3.00 / 3.37 ms`.
  - Streaming frame mean/median/p95: `30.80 / 30.84 / 32.65 ms`.
  - Throughput: `32.47 fps`.
- Graph-capture result (`webgpu_app/bench/results/graph_capture_latest.json`, created
  `2026-05-10T15:39:58.096Z`):
  - Dynamics after graph-capture warmup mean/median/p95: `17.98 / 19.91 / 20.88 ms`.
  - Decoder after graph-capture warmup mean/median/p95: `2.07 / 2.29 / 2.35 ms`.
  - Streaming frame after graph-capture warmup mean/median/p95: `20.34 / 22.50 / 23.65 ms`.
  - Throughput after graph-capture warmup: `49.17 fps`.
- Best graph-capture repeat observed during the latent decoder trial was `49.32 fps`, with decoder
  after-warmup mean/median/p95 at `2.03 / 2.27 / 2.34 ms`.

Conclusion:
- Accept as an exact decoder/runtime cleanup. The decoder itself improves clearly, and normal smoke
  improves from `31.12 fps` to `32.47 fps`.
- The graph-capture full-frame result is effectively unchanged because the frame remains dominated by
  dynamics variance. The steady median frame is still about `22.5 ms`, short of consistent 60 FPS.

Additional rejected follow-ups:
- Direct cache-length entry graph under graph capture:
  - Command used the wrapper flag:
    `bun run benchmark:webgpu -- --grep @graph-capture --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_cache_length_entry_b1_t1_s2 --playwright-benchmark-attempts 5`.
  - Result: slower than the full-cache specialized graph.
  - Dynamics after warmup: `23.97 / 25.11 / 25.54 ms`.
  - Streaming after warmup: `26.33 / 27.59 / 27.99 ms`, or `37.98 fps`.
  - Conclusion: keep the full-cache specialized steady-state artifact.
- Squeezed K-entry output ABI:
  - Temporarily removed the final `12` K-entry output `Unsqueeze` nodes and exposed
    `candidate_k_entry` as `[432,1,2,32]`.
  - CPU comparison against the accepted graph was exact for `final_z`, `candidate_v_entry`, and
    the reshaped `candidate_k_entry`.
  - Browser graph capture was slower:
    - Dynamics after warmup: `18.59 / 20.51 / 21.51 ms`.
    - Streaming after warmup: `20.99 / 23.20 / 24.21 ms`, or `47.65 fps`.
  - Restored the accepted dynamics artifact SHA:
    `c514d932ba978e5be2f73da9f5f865bd3f70c2732cefdbd14e6ad771058afcd8`.
- Runtime `graphOptimizationLevel=all` retest on the current accepted artifact:
  - Command used the wrapper flag:
    `bun run benchmark:webgpu -- --grep @graph-capture --webgpu-benchmark-graph-optimization-level all --playwright-benchmark-attempts 5`.
  - Result: slower than `basic`.
  - Dynamics after warmup: `18.97 / 20.92 / 21.92 ms`.
  - Streaming after warmup: `21.41 / 23.67 / 24.61 ms`, or `46.71 fps`.
  - Keep the default `basic` graph optimization level.
- Root `webgpu_app/assets` control run:
  - Command used the wrapper flag:
    `bun run benchmark:webgpu -- --grep @graph-capture --webgpu-benchmark-asset-base /webgpu_app/assets --playwright-benchmark-attempts 5`.
  - The manifest there selected the cache-length entry graph, not the full-cache specialized graph.
  - Streaming after warmup: `25.68 / 26.80 / 27.87 ms`, or `38.93 fps`.
  - Conclusion: the older high-FPS root-asset note is not reproducible under the current benchmark
    configuration and is not a replacement for the curated Breakout full-cache path.
- Current-graph composite dynamics+decoder trial:
  - Built a temporary composite ONNX graph by appending `breakout_tokenizer_decode_z_b1_t1` after
    `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2`.
  - Outputs were `final_z`, `candidate_k_entry`, `candidate_v_entry`, and `patches`; the benchmark
    skipped the separate decoder session for this trial.
  - Graph capture passed after a retry around the known Chrome Crashpad startup issue.
  - The combined captured step was slower than separate sessions:
    - Combined dynamics/decoder after warmup: `22.23 / 24.72 / 24.87 ms`.
    - Streaming after warmup: `22.80 / 25.36 / 25.51 ms`, or `43.86 fps`.
  - Restored the manifest, benchmark runtime, and accepted separate-session artifact path.
- FP16 precision trials:
  - Dynamics conversion with `keep_io_types=True`, `disable_shape_infer=True`, and
    `Softmax`/`QuickGelu` blocked from conversion failed the exactness gate before browser timing.
  - CPU deltas versus the accepted FP32 dynamics graph:
    - `final_z` max/mean absolute error: `0.00968 / 0.000875`.
    - `candidate_k_entry` max/mean absolute error: `0.02836 / 0.001624`.
    - `candidate_v_entry` max/mean absolute error: `0.00913 / 0.000524`.
  - Decoder-only FP16 also failed the `5e-4` and `1e-3` allclose gates:
    - `patches` max/mean/p95 absolute error: `0.002953 / 0.000050 / 0.000230`.
  - Conclusion: reject FP16 for now. It is likely a real speed lever, but it changes numerical
    behavior enough that it needs a separate model-quality decision, not a transparent runtime
    optimization.
- Runtime knob retests on the accepted graph:
  - `preferredLayout=NHWC` with graph capture and 10 attempts measured:
    - Dynamics after warmup: `17.99 / 19.87 / 20.15 ms`.
    - Streaming after warmup: `20.41 / 22.52 / 22.81 ms`, or `48.99 fps`.
  - `graphOptimizationLevel=extended` with graph capture and 10 attempts measured:
    - Dynamics after warmup: `17.92 / 19.85 / 20.15 ms`.
    - Streaming after warmup: `20.32 / 22.53 / 22.81 ms`, or `49.21 fps`.
  - Same-window default `basic` rerun measured:
    - Dynamics after warmup: `18.01 / 19.89 / 20.14 ms`.
    - Streaming after warmup: `20.38 / 22.54 / 22.73 ms`, or `49.07 fps`.
  - Conclusion: NHWC and `extended` are within run noise here and are not worth changing defaults.
- Rank-aware `MatMul` replacement trial:
  - Replaced `76` exact `Squeeze -> Gemm -> Unsqueeze` islands with rank-aware `MatMul`, adding
    bias `Add` nodes only for the `5` biased Gemms.
  - Node count changed from `3698` to `3551`.
  - CPU comparison against the accepted graph was exact for `final_z`, `candidate_k_entry`, and
    `candidate_v_entry`.
  - Browser graph capture rejected the graph because not all nodes partitioned to the WebGPU
    execution provider.
  - Restored the manifest, moved the temporary artifact out of the served asset directory, and
    reran the accepted graph-capture baseline.
- Narrow rank-3 `MatMul` replacement trial:
  - Replaced only the `71` no-bias single-axis `Squeeze -> Gemm -> Unsqueeze` islands.
  - CPU comparison against the accepted graph was exact for `final_z`, `candidate_k_entry`, and
    `candidate_v_entry`.
  - Browser graph capture passed, but performance was slightly slower:
    - Dynamics after warmup: `18.19 / 20.04 / 20.20 ms`.
    - Streaming after warmup: `20.64 / 22.64 / 22.81 ms`, or `48.45 fps`.
  - Conclusion: reject. The WebGPU `Gemm` path plus explicit layout ops remains faster than the
    rank-aware `MatMul` kernel in this graph.
- Restored accepted graph-capture baseline after the rejected `MatMul` trials:
  - `webgpu_app/bench/results/graph_capture_latest.json`, created `2026-05-10T16:13:25.041Z`.
  - Dynamics after warmup: `17.92 / 19.80 / 19.99 ms`.
  - Decoder after warmup: `2.09 / 2.28 / 2.34 ms`.
  - Streaming after warmup: `20.33 / 22.42 / 22.55 ms`, or `49.19 fps`.
- Offline ORT `ENABLE_ALL` optimized artifact:
  - Re-serialized the accepted dynamics artifact with ORT `GraphOptimizationLevel.ORT_ENABLE_ALL`.
  - CPU comparison against the accepted graph was exact for `final_z`, `candidate_k_entry`, and
    `candidate_v_entry`.
  - Node/op counts were unchanged (`3698` nodes with the same op histogram), so this was not a
    distinct browser candidate. The earlier browser `graphOptimizationLevel=all` runtime retest was
    already slower.
  - Moved the temporary artifact out of the served asset directory.
- Attention head-merge `Flatten(axis=2)` trial:
  - Replaced `71` exact head-merge islands of
    `Split(axis=2) -> Concat(axis=3) -> Squeeze([0,2])` with `Flatten(axis=2)`.
  - Node count changed from `3698` to `3556`.
  - CPU comparison against the accepted graph was exact for `final_z`, `candidate_k_entry`, and
    `candidate_v_entry`.
  - Browser graph capture rejected the graph because not all nodes partitioned to the WebGPU
    execution provider, matching the earlier static `Reshape` rejection pattern.
- Attention head-merge projection `Einsum` trial:
  - Replaced the same `71` head-merge islands plus their following output-projection `Gemm` and
    rank-restoring `Unsqueeze` with rank-aware `Einsum("bthd,hdm->btm")`.
  - Node count changed from `3698` to `3414`; CPU comparison against the accepted graph was exact
    for `final_z`, `candidate_k_entry`, and `candidate_v_entry`.
  - Browser graph capture passed but was slower:
    - Dynamics after warmup: `19.96 / 22.47 / 22.74 ms`.
    - Decoder after warmup: `2.58 / 2.86 / 2.89 ms`.
    - Streaming after warmup: `22.88 / 25.67 / 25.93 ms`, or `43.71 fps`.
  - Conclusion: reject. Removing the merge layout kernels is not worth replacing these projection
    `Gemm` kernels with generic `Einsum` in ORT WebGPU.
- Restored accepted graph-capture baseline after the rejected head-merge trials:
  - `webgpu_app/bench/results/graph_capture_latest.json`, created `2026-05-10T16:20:48.242Z`.
  - Dynamics after warmup: `17.88 / 19.75 / 19.99 ms`.
  - Decoder after warmup: `2.09 / 2.27 / 2.31 ms`.
  - Streaming after warmup: `20.28 / 22.38 / 22.53 ms`, or `49.31 fps`.

Live demo render-loop cleanup:
- Added a precomputed patch render map for the Dream Arcade demo. The CPU bridge from tokenizer
  patches to `ImageData` now precomputes the source offset for every output pixel, reuses the
  `ImageData` allocation, and runs a single flat pixel loop.
- This preserves the same patch-to-pixel rounding and clamping behavior, including the packed
  float16 fallback path.
- This change does not affect `benchmark:webgpu`, because the benchmark measures GPU step/decoder
  timings and does not render decoder patches through the canvas CPU bridge.
- Browser/code checks after the render-loop cleanup:
  - `bun run typecheck`: passed.
  - `bun run build:webgpu:browser`: passed.
  - `bun run demo:webgpu:build`: passed.
  - `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 10`: passed, 5 tests.
  - `git diff --check`: passed.
- Accepted ONNX artifact hashes were rechecked after the rejected trials:
  - Dynamics:
    `c514d932ba978e5be2f73da9f5f865bd3f70c2732cefdbd14e6ad771058afcd8`.
  - Latent decoder:
    `54581c2b852dbea7b9249e52b271e1255c43cfa494a2091070a5387991433e45`.

### Q-Head Gather Layout Rewrite

Change accepted:
- Added `rewrite_q_head_split_gather_for_webgpu()` to
  `scripts/webgpu/export_dreamer4_onnx.py`.
- Wired the rewrite into `scripts/webgpu/specialize_full_cache_entry.py` after packed QKV partial
  head-split rewriting.
- The rewrite replaces exact 8-way Q-head layout islands:
  - `Split(axis=1)` into eight 32-wide heads.
  - Per-head `Unsqueeze`.
  - `Concat` to restore the ranked head tensor.
- Replacement uses `Gather` with a static `[8,32]` index table plus the minimal rank-restoring
  `Unsqueeze`/`Transpose` needed for the two observed layouts.

Validation:
- Canonical full-cache dynamics artifact regenerated through the specialization script.
- Validation against the cache-length source graph at full cache passed with `atol=5e-4`,
  `rtol=5e-4`.
- Max / mean absolute errors:
  - `final_z`: `2.351e-05 / 7.05e-07`.
  - `candidate_k_entry`: `4.435e-05 / 9.36e-07`.
  - `candidate_v_entry`: `2.885e-05 / 4.19e-07`.
- Latent decoder rewrite was exact against its previous artifact on CPU: `patches` max/mean
  absolute error `0.0 / 0.0`.

Generated artifact changes:
- Dynamics SHA-256:
  `5d3fb86ef9895633f92d7926d013cda11622eadd6005f8f641a0cf0add4d14d9`.
- Dynamics node count after the full specialization pipeline: `3166`.
- Dynamics Q-head gather rewrites: `36` `axis01_concat1` and `35` `axis12_concat2`.
- Latent decoder SHA-256:
  `b9689f234ce20bb9dd1702f0c80710e5cb34b42c6682e0f0610af40ad52591af`.
- Latent decoder node count: `393 -> 335`.
- The Breakout ONNX manifest and collection manifest were updated with the new hashes.

Rejected/neutral follow-ups:
- Rewriting only the `axis01_concat1` Q-head layout subset was exact and graph-capture compatible,
  but measured within noise of the combined rewrite:
  - Streaming after warmup: `20.12 / 22.35 / 22.45 ms`, or `49.70 fps`.
- Rewriting only the `axis12_concat2` subset was also exact and graph-capture compatible:
  - Streaming after warmup: `20.00 / 22.13 / 22.26 ms`, or `50.00 fps`.
- The combined rewrite remains the accepted variant because it gives the best observed p95 and
  removes the most layout dispatches without a measurable regression.
- Rewriting the two remaining final cache-entry 2-head layout islands was exact and graph-capture
  compatible, but it did not improve performance:
  - Streaming after warmup: `19.90 / 22.04 / 22.63 ms`, or `50.24 fps`.
  - Conclusion: reject. Replacing two small `Split/Unsqueeze/Concat` islands with `Gather` is not
    enough to matter and slightly worsened p95 in this run.
- Retesting runtime knobs on the new graph:
  - `graphOptimizationLevel=extended` regressed sharply to `39.26 fps`.
  - `graphOptimizationLevel=all` was not stable: one run measured `50.51 fps`, but the repeat
    measured `49.91 fps`.
  - `preferredLayout=NHWC` measured `50.31 fps`, essentially the same as the accepted default
    window, so the default layout stays unchanged.
- Decoder-only preallocated output under graph capture passed but was slower:
  - Streaming after warmup: `20.13 / 22.20 / 23.06 ms`, or `49.67 fps`.
  - Decoder after warmup: `2.21 / 2.43 / 2.52 ms`.
  - Conclusion: reject. The benchmark runtime patch was reverted, and decoder output preallocation
    stays disabled for graph-capture runs.
- Packed-QKV K/V two-head gather trial:
  - Replaced the remaining packed-QKV K/V two-head `Split -> Unsqueeze -> Concat` islands with
    grouped 64-wide K/V split chunks plus static `Gather` and the minimal rank-restoring layout
    step.
  - CPU comparison against the accepted full-cache entry artifact was exact for `final_z`,
    `candidate_k_entry`, and `candidate_v_entry`.
  - Browser graph capture passed, but the result was slower:
    - Streaming after warmup: `20.37 / 22.50 / 22.70 ms`, or `49.10 fps`.
    - Dynamics after warmup: `18.06 / 19.92 / 20.13 ms`.
  - Conclusion: reject. The extra `Gather`/`Transpose` work is not worth the removed
    `Unsqueeze`/`Concat` nodes.
- Packed-SwiGLU pre-split unsqueeze trial:
  - Replaced packed-SwiGLU `Split(axis=1) -> two Unsqueeze` sites with
    `Unsqueeze -> Split(axis=2)`, preserving the rank-3 QuickGelu/Mul path while removing one
    layout node per site.
  - CPU comparison against the accepted full-cache entry artifact was exact for `final_z`,
    `candidate_k_entry`, and `candidate_v_entry`.
  - Browser graph capture passed, but the result was slower:
    - Streaming after warmup: `20.35 / 22.57 / 22.89 ms`, or `49.14 fps`.
    - Dynamics after warmup: `17.79 / 19.71 / 19.97 ms`.
  - Conclusion: reject. Moving the unsqueeze before the split reduces node count but does not
    improve ORT WebGPU steady-state latency.
- WebGPU profiling diagnostic:
  - Added benchmark wrapper flags and query params for ORT WebGPU profiling:
    `--webgpu-benchmark-profiling`, `--webgpu-benchmark-profiling-required`,
    `--webgpu-benchmark-profiling-drain-ms`, and `--webgpu-benchmark-profiling-top-k`.
  - The benchmark installs `ort.env.webgpu.profiling.ondata` before creating sessions and records a
    grouped summary when events are available.
  - A diagnostic graph-capture run with profiling required completed the model path, but failed the
    profiling gate because no profiling events were received:
    `WebGPU profiling was required but no profiling events were received.`
  - Conclusion: blocked. Profiling remains inert unless explicitly enabled, but this environment did
    not provide usable ORT WebGPU kernel timing data for the graph-capture run.
- Normal graph-capture result restored after the failed profiling-required result file:
  - `webgpu_app/bench/results/graph_capture_latest.json`, created `2026-05-10T17:01:22.746Z`.
  - Dynamics after graph-capture warmup: `17.92 / 19.82 / 20.97 ms`.
  - Decoder after graph-capture warmup: `2.31 / 2.45 / 3.08 ms`.
  - Streaming frame after graph-capture warmup: `20.55 / 22.69 / 23.90 ms`.
  - Throughput after graph-capture warmup: `48.66 fps`.
  - Conclusion: valid but lower than the earlier accepted `49.5-50.3 fps` window, so treat it as
    timing variance rather than an accepted improvement.
- Head-time cache ABI trial:
  - Built temporary BHNTD prefill and full-cache step artifacts:
    - Prefill cache outputs changed from `[layer,batch,token,time,head,dim]` to
      `[layer,batch,token,head,time,dim]`.
    - Step cache inputs changed to the same head-time layout.
    - Temporal attention concatenated cache/current K/V along the time axis in `[B,H,T,D]`,
      switched the matching GQA `Gather` axes, removed `35` Q post-RoPE transposes and `24` K
      post-RoPE transposes, and inserted `35` V-current transposes.
  - Step graph node count changed `3166 -> 3142`; `Transpose` changed `249 -> 225`.
  - CPU comparison against the accepted artifacts was exact after explicit cache-layout transposes:
    - Step `final_z`, `candidate_k_entry`, `candidate_v_entry`: max/mean absolute error `0.0 / 0.0`.
    - Prefill `pred_z`, transposed `k_cache`, transposed `v_cache`, `cache_length`: max/mean
      absolute error `0.0 / 0.0`.
  - Browser graph capture passed with temporary manifest entries and wrapper flags:
    - Dynamics after warmup: `19.81 / 22.13 / 22.68 ms`.
    - Decoder after warmup: `0.58 / 0.62 / 0.76 ms`.
    - Streaming after warmup: `20.44 / 22.76 / 23.38 ms`, or `48.91 fps`.
  - Conclusion: reject. The lower transpose count and head-time cache memory layout did not improve
    end-to-end graph-capture throughput; dynamics got slower, and the streaming result stayed within
    or below the accepted timing window.
- Restored accepted artifacts and manifest after the BHNTD trial:
  - Moved the temporary served trial ONNX files back out of
    `webgpu_app/dream_arcade_assets/breakout`.
  - `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 3`: passed.
  - Restored graph-capture result (`webgpu_app/bench/results/graph_capture_latest.json`, created
    `2026-05-10T17:12:39.631Z`):
    - Dynamics after warmup: `17.76 / 19.82 / 20.17 ms`.
    - Decoder after warmup: `2.23 / 2.44 / 2.66 ms`.
    - Streaming after warmup: `20.31 / 22.72 / 23.02 ms`, or `49.23 fps`.
- Deferred cache-update scheduling trial:
  - Moved the browser cache slide/rebase submission after decoder/render work in both the benchmark
    loop and live demo frame loop.
  - This preserves model outputs and cache contents before the next dynamics call, but changes queue
    ordering so the visible frame can be produced before the cache update needed by the next frame.
  - Browser graph capture passed. The clean measured runs were neutral/noisy:
    - First run: streaming after warmup `20.09 / 22.24 / 23.16 ms`, or `49.77 fps`.
    - Clean repeat: streaming after warmup `20.36 / 22.60 / 23.37 ms`, or `49.12 fps`.
  - Conclusion: keep only as a live-display scheduling cleanup; it is not counted as a
    model-benchmark FPS improvement.
- WebGPU canvas patch renderer:
  - Added a live-demo WebGPU render path for tokenizer `patches` output when the WebGPU backend is
    active. The decoder now returns `patches` as a GPU buffer, and a small render pipeline maps the
    patch tensor directly to the canvas.
  - The existing CPU `patches -> ImageData -> putImageData` path remains as the fallback for CPU/2D
    rendering.
  - The visible WebGPU canvas cannot also expose a 2D context, so the demo smoke canvas-change check
    now hashes `locator('#frame').screenshot()` instead of requiring `getContext('2d')`.
  - Initial smoke attempts exposed a fallback bug (`2D canvas context is unavailable`) after a
    failed renderer setup claimed the canvas. The renderer now sets up on a replacement canvas and
    swaps it into the DOM only after successful creation.
  - Follow-up: the renderer now attaches the WebGPU canvas lazily on the first GPU-backed patch
    frame. If ORT still returns CPU patches, the demo disables that renderer and the 2D fallback
    replaces any claimed canvas with a fresh 2D canvas instead of throwing
    `2D canvas context is unavailable`.
  - Browser/code checks:
    - `bun run typecheck`: passed.
    - `bun run build:webgpu:browser`: passed.
    - `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 3`: passed, 5 tests.
    - `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 3`:
      passed. This model benchmark does not exercise the live canvas render path; the latest run
      measured streaming after warmup `20.77 / 23.08 / 23.67 ms`, or `48.14 fps`.
  - Conclusion: accept as a live-demo readback/render cleanup. It removes per-frame decoder patch
    CPU readback from the live WebGPU path, but it does not change ONNX numerical outputs or the
    maintained `benchmark:webgpu` model FPS.
- Cache-layer Slice/Squeeze to Gather rewrite:
  - Added `rewrite_cache_layer_slices_as_gather_for_webgpu` and wired it into the full-cache
    specialization path after the existing WebGPU graph cleanups.
  - The pass replaces full-shape `Slice(k_cache|v_cache, layer i) -> Squeeze(axis=0)` extraction
    islands with scalar `Gather(axis=0)`, removing one layout node per K/V layer.
  - Final accepted artifact:
    `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2.onnx`.
    - SHA-256: `646d1088b41a68eb90354f306c8df4d84ea457fd2c74e3c569acee465f3fe76b`.
    - Node count: `3166 -> 3142`.
    - Tracked op counts: `Slice 26 -> 2`, `Squeeze 341 -> 317`, `Gather 214 -> 238`.
    - Manifest records `cache_layer_slice_to_gather: 24`.
  - CPU validation:
    - The specialization script passed source-vs-full-cache validation at `5e-4` tolerance:
      `final_z` max/mean `2.35e-05 / 7.05e-07`, `candidate_k_entry`
      `4.43e-05 / 9.36e-07`, `candidate_v_entry` `2.88e-05 / 4.19e-07`.
    - A direct CPU comparison between the regenerated accepted artifact and the temporary trial
      artifact was exact for all three outputs (`array_equal`, max/mean `0.0 / 0.0`).
  - Browser graph-capture trial runs:
    - Trial run: streaming after warmup `20.27 / 22.59 / 23.04 ms`, or `49.34 fps`.
    - Trial repeat: streaming after warmup `19.81 / 21.89 / 22.64 ms`, or `50.48 fps`.
    - Same-window default artifact comparison: streaming after warmup
      `20.53 / 22.97 / 23.32 ms`, or `48.71 fps`.
  - Browser/code checks after regenerating the accepted artifact:
    - `.venv/bin/python -m py_compile scripts/webgpu/export_dreamer4_onnx.py scripts/webgpu/specialize_full_cache_entry.py`:
      passed.
    - `bun run typecheck`: passed.
    - `bun run build:webgpu:browser`: passed.
    - `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 3`: passed, 5 tests.
    - `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 3`:
      passed twice after acceptance. The repeat measured streaming after warmup
      `20.45 / 22.77 / 23.22 ms`, or `48.90 fps`.
    - No `*trial*` artifacts remain in `webgpu_app/dream_arcade_assets/breakout`.
  - Conclusion: accept as an exact graph simplification with a small positive same-window trial
    result. The latest accepted benchmark remains noisy and below 60 FPS; dynamics is still the
    dominant frame cost.
- Attention value/output projection three-input `Einsum` trial:
  - Built a temporary graph that replaced `36` chains of
    `Einsum("bhqk,bkhd->bqhd") -> Split(axis=2) -> Concat(axis=3) -> Squeeze([0,2]) -> Gemm -> Unsqueeze`
    with one `Einsum("bhqk,bkhd,hdm->bqm")`.
  - Node count changed `3142 -> 2962`; op counts changed by `-36` each for `Split`, `Concat`,
    `Squeeze`, `Gemm`, and `Unsqueeze`.
  - CPU comparison against the accepted full-cache artifact was exact for `final_z`,
    `candidate_k_entry`, and `candidate_v_entry` (`array_equal`, max/mean `0.0 / 0.0`).
  - Browser graph capture passed, but performance regressed sharply:
    - Dynamics after warmup: `35.25 / 39.94 / 40.69 ms`.
    - Decoder after warmup: `4.47 / 4.98 / 5.40 ms`.
    - Streaming after warmup: `40.82 / 46.13 / 46.70 ms`, or `24.50 fps`.
  - Restored the accepted manifest and moved the temporary served artifact back out of
    `webgpu_app/dream_arcade_assets/breakout`.
  - Conclusion: reject. ORT WebGPU's generic three-input `Einsum` path is much slower than the
    current two-stage attention-output plus `Gemm` projection, despite the lower node count.
- Final K/V entry sibling-Gemm pack trial:
  - Packed the two remaining final K/V entry `Gemm(128,64)` projections from the same input into
    one `Gemm(128,128)` plus one four-way `Split`, replacing the two following two-way head splits.
  - Node count changed `3142 -> 3140`; op counts changed `Gemm 291 -> 290` and `Split 215 -> 214`.
  - CPU comparison against the accepted full-cache artifact was exact for `final_z`,
    `candidate_k_entry`, and `candidate_v_entry` (`array_equal`, max/mean `0.0 / 0.0`).
  - Browser graph capture passed, but it was neutral in a same-window comparison:
    - Trial: streaming after warmup `20.53 / 22.82 / 23.29 ms`, or `48.70 fps`.
    - Default repeat: streaming after warmup `20.54 / 22.60 / 23.97 ms`, or `48.68 fps`.
  - Restored the accepted manifest and moved the temporary served artifact back out of
    `webgpu_app/dream_arcade_assets/breakout`.
  - Conclusion: reject for now. The rewrite is exact, but it is too small to produce a measurable
    speedup and would add a special-case export pass.
- One-position RoPE transpose removal:
  - Added `rewrite_one_position_rotary_transposes_for_webgpu` and wired it into the reproducible
    full-cache specialization path after the existing graph cleanups.
  - The pass targets one-position temporal RoPE islands:
    `Transpose([0,2,1,3]) -> RotaryEmbedding -> Transpose([0,2,1,3])`.
  - For these branches the true sequence length is `1` and the position id is fixed to zero. The
    replacement runs `RotaryEmbedding` directly on the original layout with `num_heads=1` and repeats
    the single cos/sin row across the true-head axis (`2` or `8`). This preserves the exact
    elementwise rotation while avoiding both layout transposes.
  - Final accepted artifact:
    `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2.onnx`.
    - SHA-256: `cbdde638de674e73d4b70a4b8a420a26d570c238262ff6b48179a4961853ef64`.
    - Node count: `3142 -> 3000`.
    - Tracked op counts: `Transpose 249 -> 107`; `RotaryEmbedding` remains `143`.
    - Manifest records `direct_repeated_one_position_rotary: 71`.
  - CPU validation:
    - The specialization script passed source-vs-full-cache validation at `5e-4` tolerance:
      `final_z` max/mean `2.35e-05 / 7.05e-07`, `candidate_k_entry`
      `4.43e-05 / 9.36e-07`, `candidate_v_entry` `2.88e-05 / 4.19e-07`.
    - A direct CPU comparison between the regenerated accepted artifact and the temporary trial
      artifact was exact for all three outputs (`array_equal`, max/mean `0.0 / 0.0`).
  - Browser graph-capture results:
    - Temporary trial: streaming after warmup `19.59 / 22.18 / 23.05 ms`, or `51.06 fps`.
    - Temporary trial repeat: streaming after warmup `19.62 / 22.22 / 22.72 ms`, or `50.98 fps`.
    - Same-window default artifact comparison: streaming after warmup
      `20.46 / 22.59 / 23.03 ms`, or `48.88 fps`.
    - Regenerated accepted artifact: streaming after warmup `19.94 / 22.50 / 23.12 ms`, or
      `50.16 fps`.
    - Regenerated accepted repeat: streaming after warmup `19.79 / 22.37 / 22.95 ms`, or
      `50.54 fps`.
    - Later default rerun after runtime-knob retests: streaming after warmup
      `19.65 / 22.32 / 22.82 ms`, or `50.89 fps`.
    - Post-restore default rerun after the rejected spatial Q-head retest: streaming after warmup
      `19.90 / 22.54 / 23.29 ms`, or `50.26 fps`.
  - Browser/code checks after regenerating the accepted artifact:
    - `.venv/bin/python -m py_compile scripts/webgpu/export_dreamer4_onnx.py scripts/webgpu/specialize_full_cache_entry.py`:
      passed.
    - `bun run typecheck`: passed.
    - `bun run build:webgpu:browser`: passed.
    - `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 3`:
      passed twice.
    - `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 3`: passed, 5 tests.
  - Conclusion: accept. This is an exact full-cache graph cleanup and gives the best accepted
    graph-capture result so far in this round, but the demo remains short of consistent 60 FPS.
- Decoder one-position RoPE transpose trial:
  - Applied a shape-light variant of the same direct one-position RoPE rewrite to the accepted
    latent decoder artifact.
  - It removed `4` decoder RoPE transpose wrappers: node count `335 -> 327`, `Transpose 18 -> 10`.
  - CPU comparison against the accepted decoder artifact was exact for `patches`
    (`array_equal`, max/mean `0.0 / 0.0`).
  - Browser graph capture passed after temporarily swapping the decoder artifact, but the full frame
    did not improve:
    - Dynamics after warmup: `17.68 / 19.88 / 20.55 ms`.
    - Decoder after warmup: `2.09 / 2.29 / 2.72 ms`.
    - Streaming after warmup: `19.98 / 22.52 / 23.11 ms`, or `50.04 fps`.
  - Restored the accepted decoder artifact and manifest.
  - Conclusion: reject for now. The decoder cleanup is exact but too small to move the maintained
    full-frame benchmark.
- Runtime knob retests after one-position RoPE transpose removal:
  - `preferredLayout=NHWC` with graph capture:
    - Streaming after warmup: `19.87 / 22.41 / 23.05 ms`, or `50.32 fps`.
  - `graphOptimizationLevel=all` with graph capture:
    - Streaming after warmup: `19.83 / 22.40 / 23.14 ms`, or `50.43 fps`.
  - `graphOptimizationLevel=extended` with graph capture:
    - Streaming after warmup: `19.91 / 22.41 / 23.18 ms`, or `50.22 fps`.
  - Conclusion: keep the default `basic` graph optimization level and default layout. The retests
    are within the RoPE-reduced default timing band and do not justify changing runtime defaults.
- Spatial Q-head split/concat retest after one-position RoPE transpose removal:
  - Reverted only the `36` spatial Q-head `Gather -> Unsqueeze -> Transpose` layout islands back to
    `Split(axis=1) -> 8x Unsqueeze([0,1]) -> Concat(axis=1)`.
  - This removed `36` transposes and `36` gathers, but increased total node count `3000 -> 3252`.
  - CPU comparison against the accepted full-cache artifact was exact for `final_z`,
    `candidate_k_entry`, and `candidate_v_entry` (`array_equal`, max/mean `0.0 / 0.0`).
  - Browser graph capture passed, but the result did not beat the accepted RoPE-reduced graph:
    - Dynamics after warmup: `17.31 / 19.39 / 20.20 ms`.
    - Decoder after warmup: `2.24 / 2.37 / 3.13 ms`.
    - Streaming after warmup: `19.90 / 22.45 / 23.07 ms`, or `50.26 fps`.
  - Restored the accepted manifest and moved the temporary served artifact back out of
    `webgpu_app/dream_arcade_assets/breakout`.
  - Conclusion: reject. Removing these transposes is not worth reintroducing the larger
    Split/Unsqueeze/Concat layout islands.
- Residual layout transpose canonicalization attempts:
  - The accepted graph still has `69` residual `Transpose([1,0,2])` nodes between residual-block
    outputs and the following normalization/projection paths.
  - A chain-aware trial removed all `69` such transposes and retargeted adjacent squeeze/unsqueeze
    axes based on inferred original shapes (`3000 -> 2931`, `Transpose 107 -> 38`), but CPU
    validation failed at `node_Reshape_309__squeeze` because the rewritten graph attempted to
    squeeze axis `1` of a `{1,36,128}` tensor.
  - A stricter axis-0 canonical trial also removed all `69` residual transposes, but CPU execution
    failed later at `node_Sub_2932` with an invalid broadcast around the final sample-update output
    head.
  - Conclusion: reject the broad rewrite. Any future residual-layout work needs a narrower scope
    that avoids the final sample/context output heads or proves the output-head axes separately.
- Graph-capture preallocated output trial:
  - Temporarily allowed the benchmark's preallocated GPU output fetch tensors to be used while
    `enableGraphCapture` is active.
  - Browser graph capture failed during ORT output binding with
    `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`.
  - Restored the benchmark to use preallocated output tensors only for non-graph-capture WebGPU.
  - A follow-up graph-capture benchmark passed after restoration and measured streaming after
    warmup `19.59 / 22.19 / 22.59 ms`, or `51.05 fps` by the benchmark mean. The median remains
    above the 60 FPS frame budget, so this is not a completion signal.
- Playwright Chrome home hardening:
  - The benchmark wrapper could still hit Chrome Crashpad launch failures that attempted to use the
    real macOS home directory despite `HOME=/private/tmp/visionary-chrome-home`.
  - Added `CFFIXED_USER_HOME` and pre-created both `Google/Chrome/Crashpad` and
    `Google/Chrome for Testing/Crashpad` under the temporary Chrome home in
    `scripts/webgpu/run_playwright_chrome_home.ts` and `playwright.config.ts`.
  - Verification after the launcher change:
    - `bun run typecheck`: passed.
    - `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 3`:
      passed on the accepted artifact, measuring streaming after warmup
      `19.49 / 21.97 / 22.37 ms`, or `51.30 fps` by the benchmark mean.
- Live decoder graph-capture attempt:
  - The live demo now tries to create the decoder session with `enableGraphCapture` when WebGPU
    graph capture is requested and a fixed GPU decoder input can be used, falling back to the normal
    decoder session if ORT rejects it.
  - This aligns the live demo with the benchmark's decoder graph-capture path. It does not change
    model outputs, cache policy, or sample-step count.
  - `bun run build:webgpu:browser`: passed.
  - `bun run typecheck`: passed.
  - `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 3`: passed, 5 tests.
- Final output-head `Transpose -> Slice` trial:
  - Built a temporary dynamics graph that rewired the two final output-head islands from
    `Transpose([0,2,1,3]) -> Slice(axis=2, 4:36) -> Squeeze([0,1])` to a direct slice on the
    pre-transpose tensor plus `Squeeze([0,2])`.
  - Node count changed `3000 -> 2998`; `Transpose 107 -> 105`.
  - CPU comparison against the accepted artifact was exact for `final_z`, `candidate_k_entry`, and
    `candidate_v_entry` (`array_equal`, max/mean `0.0 / 0.0`).
  - Browser trial attempts repeatedly failed during Chrome launch before the page loaded, so there
    is no WebGPU performance result for this rewrite.
  - Restored the accepted artifact and verified its SHA-256 remained
    `cbdde638de674e73d4b70a4b8a420a26d570c238262ff6b48179a4961853ef64`.
  - Conclusion: do not accept yet. The rewrite is exact but too small to justify changing the
    maintained artifact without a passing browser measurement.

Browser/code checks:
- `.venv/bin/python -m py_compile scripts/webgpu/export_dreamer4_onnx.py scripts/webgpu/specialize_full_cache_entry.py`: passed.
- `bun run typecheck`: passed.
- `bun run build:webgpu:browser`: passed.
- `bun run benchmark:webgpu -- --grep @smoke --playwright-benchmark-attempts 10`: passed.
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 10`: passed.
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 3`: passed
  after the profiling-required diagnostic failure, restoring
  `webgpu_app/bench/results/graph_capture_latest.json` to a passing result.
- `bun run demo:webgpu:build`: passed.
- `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 10`: passed on wrapper retry,
  5 tests. The first attempt had a transient Pacman key-state expectation failure.
- `bun run build:webgpu:browser`: passed after adding the profiling query controls.
- `git diff --check`: passed.
- Artifact hash check still matches the accepted dynamics and decoder artifacts:
  - Dynamics:
    `cbdde638de674e73d4b70a4b8a420a26d570c238262ff6b48179a4961853ef64`.
  - Decoder:
    `b9689f234ce20bb9dd1702f0c80710e5cb34b42c6682e0f0610af40ad52591af`.
- No `*trial*` artifacts remain in `webgpu_app/dream_arcade_assets/breakout`.

Fresh benchmark results after accepting the Q-head gather rewrite:
- Normal WebGPU smoke result (`webgpu_app/bench/results/latest.json`, created
  `2026-05-10T16:30:01.028Z`):
  - Dynamics mean/median/p95: `24.81 / 24.78 / 26.37 ms`.
  - Decoder mean/median/p95: `2.78 / 2.70 / 3.48 ms`.
  - Streaming frame mean/median/p95: `27.63 / 27.57 / 29.16 ms`.
  - Throughput: `36.19 fps`.
- Graph-capture result (`webgpu_app/bench/results/graph_capture_latest.json`, created
  `2026-05-10T16:52:06.120Z`, after restoring the accepted artifact from the rejected layout
  trials):
  - Dynamics after graph-capture warmup mean/median/p95: `17.66 / 19.42 / 19.81 ms`.
  - Decoder after graph-capture warmup mean/median/p95: `2.21 / 2.43 / 2.58 ms`.
  - Streaming frame after graph-capture warmup mean/median/p95: `20.18 / 22.23 / 22.66 ms`.
  - Throughput after graph-capture warmup: `49.56 fps`.
  - The earlier accepted run on the same artifact measured `50.29 fps`; treat the current accepted
    window as roughly `49.5-50.3 fps`.

Conclusion:
- Accept. This is the first follow-up in this round that is both numerically valid and measurably
  faster under graph capture.
- The demo is still short of consistent 60 FPS. The remaining gap is roughly `5.4 ms` versus the
  `16.67 ms` frame budget, and dynamics still dominates the steady frame.

Follow-up benchmark launcher and final-slice retry:
- Removed Playwright's macOS `--no-zygote` launch flag from the local benchmark config. It had not
  eliminated the Crashpad bootstrap failures, and it is not required for the WebGPU path.
- `bun run typecheck`: passed after the launcher edit.
- Accepted artifact check after restoring from the trial swap:
  - Dynamics SHA-256:
    `cbdde638de674e73d4b70a4b8a420a26d570c238262ff6b48179a4961853ef64`.
  - Decoder SHA-256:
    `b9689f234ce20bb9dd1702f0c80710e5cb34b42c6682e0f0610af40ad52591af`.
- Accepted graph-capture benchmark with system Chrome passed:
  - Command:
    `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 3`.
  - Result file: `webgpu_app/bench/results/graph_capture_latest.json`, created
    `2026-05-10T18:50:55.549Z`.
  - Dynamics after graph-capture warmup: `17.12 / 19.06 / 19.72 ms`.
  - Decoder after graph-capture warmup: `2.08 / 2.34 / 2.74 ms`.
  - Streaming after graph-capture warmup: `19.56 / 21.95 / 22.49 ms`, or `51.13 fps` by mean.
- Accepted graph-capture benchmark with bundled Chromium also passed after the `--no-zygote`
  removal:
  - Command:
    `bun run benchmark:webgpu -- --grep @graph-capture --playwright-benchmark-attempts 3`.
  - Result file: `webgpu_app/bench/results/graph_capture_latest.json`, created
    `2026-05-10T18:52:12.784Z`.
  - Dynamics after graph-capture warmup: `17.17 / 19.06 / 19.64 ms`.
  - Decoder after graph-capture warmup: `2.09 / 2.34 / 2.66 ms`.
  - Streaming after graph-capture warmup: `19.61 / 21.87 / 22.45 ms`, or `51.00 fps` by mean.
- Retried the exact final output-head slice rewrite by temporarily swapping in
  `/private/tmp/breakout_final_slice_no_transpose_trial.onnx` under the accepted dynamics filename
  and restoring the accepted artifact with a shell trap.
  - Bundled Chrome failed launch on all 5 attempts with the known Crashpad bootstrap permission
    error before page load.
  - System Chrome also failed launch on all 3 trial attempts with the same pre-page Crashpad
    bootstrap error, despite the accepted graph-capture run passing immediately before the trial.
  - The accepted artifact was restored and its SHA-256 was rechecked.
- Conclusion: keep rejecting the final-slice rewrite for the maintained artifact. Its CPU parity is
  exact, but there is still no browser performance result, and the rewrite is too small to accept
  on parity alone.

Follow-up full64 initial-cache experiment:
- Hypothesis: the live Breakout demo starts from an initial logical cache length of `4`, so the
  first roughly 60 generated frames fill the cache before the full-cache captured step can be used.
  A full 64-frame offline context/cache could start the page directly on the full-cache path.
- Added `prefill` mode to `scripts/webgpu/create_demo_initial_cache.py`, using the packaged
  `breakout_dynamics_prefill_cached_b1_t64` export to create initial K/V cache artifacts from a
  stored browser demo context in one CPU ORT pass.
- Added a `--demo-query` wrapper flag so demo smoke tests can select context/cache artifacts without
  leading shell environment assignments.
- Added diagnostic demo query controls:
  - `dynamicsGraphCapture`, defaulting to `graphCapture`.
  - `decoderGraphCapture`, defaulting to `graphCapture`.
- Generated temporary full64 context/cache artifacts from Breakout episode `0`, start frame `300`,
  `prefix_frames=64`, `prefix_slot_start=0`, with frame-aligned actions and prefill-created
  `cache_length=64`.

Validation:
- Full64 artifacts loaded and the start smoke rendered a frame.
- Full64 with default WebGPU graph capture failed the generated-frame smoke: frame count advanced,
  but screenshots through the early generated frames were identical.
- Control checks:
  - `bun run demo:webgpu:smoke -- --playwright-channel chrome --playwright-benchmark-attempts 1 --demo-query '?assetBase=/webgpu_app/dream_arcade_assets/breakout&contextName=breakout_demo_context_full64.json&initialCacheName=breakout_demo_initial_cache_full64.json&graphCapture=false' --grep 'world model demo changes'`:
    passed.
  - `bun run demo:webgpu:smoke -- --playwright-channel chrome --playwright-benchmark-attempts 1 --demo-query '?assetBase=/webgpu_app/dream_arcade_assets/breakout&contextName=breakout_demo_context_full64.json&initialCacheName=breakout_demo_initial_cache_full64.json&backend=wasm' --grep 'world model demo changes'`:
    passed.
  - Captured dynamics with non-captured decoder still failed the generated-frame smoke.
  - Non-captured dynamics with captured decoder still failed the generated-frame smoke.

Conclusion:
- Reject full64 as a default demo startup optimization for now. The cache artifact itself is usable
  without graph capture, but the intended captured WebGPU startup path can produce visually static
  output while the frame counter advances.
- Removed the generated `*full64*` context/cache artifacts from the served Breakout asset directory.
- Keep the prefill-cache generator mode and diagnostic query flags; they are useful for future cache
  and graph-capture isolation without changing the maintained default behavior.

Post-cleanup checks:
- `.venv/bin/python -m py_compile scripts/webgpu/create_demo_initial_cache.py scripts/webgpu/create_demo_context.py`:
  passed.
- `bun run typecheck`: passed.
- `bun run build:webgpu:browser`: passed.
- `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 3`: passed, 5 tests.
- `git diff --check -- docs/onnx_webgpu_progress.md playwright.config.ts scripts/webgpu/create_demo_initial_cache.py scripts/webgpu/run_playwright_chrome_home.ts webgpu_app/demo/main.ts`:
  passed.
- Accepted artifact hashes still match:
  - Dynamics:
    `cbdde638de674e73d4b70a4b8a420a26d570c238262ff6b48179a4961853ef64`.
  - Decoder:
    `b9689f234ce20bb9dd1702f0c80710e5cb34b42c6682e0f0610af40ad52591af`.
- No `*full64*` or `*trial*` artifacts remain in `webgpu_app/dream_arcade_assets/breakout`.

## 2026-05-11 KST: Canvas Fallback Fix And Runtime Retests

Canvas fallback fix:
- The live demo could throw `2D canvas context is unavailable` when CPU patch rendering ran after
  the visible canvas had already been claimed by a WebGPU patch renderer.
- `webgpu_app/demo/main.ts` now replaces the claimed canvas with a fresh canvas if `getContext('2d')`
  returns null, then retries the 2D context creation.
- Follow-up hardening acquires the replacement canvas' 2D context before swapping it into the DOM,
  catches browser `getContext('2d')` exceptions, and falls back to a minimal replacement canvas before
  surfacing `2D canvas context is unavailable`.
- If a decoder session returns CPU patches while a WebGPU patch renderer exists, the runtime drops
  the patch renderer and renders through the 2D fallback path.
- The WebGPU decoder patch renderer is only kept active for GPU-buffer patch outputs.

Benchmark and diagnostic controls:
- Added separate graph-capture query/runtime controls for dynamics and decoder sessions:
  `dynamicsGraphCapture` and `decoderGraphCapture`.
- Added matching benchmark wrapper flags:
  `--webgpu-benchmark-dynamics-graph-capture` and
  `--webgpu-benchmark-decoder-graph-capture`.
- Added `--playwright-headless` to the wrapper for launch diagnostics without leading shell
  environment assignments.

Decoder graph-capture isolation:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 3 --webgpu-benchmark-decoder-graph-capture false`.
- Result file: `webgpu_app/bench/results/graph_capture_latest.json`, created
  `2026-05-10T19:06:37.302Z`.
- Streaming after graph-capture warmup: `19.69 / 21.89 / 23.30 ms`, or `50.78 fps` by mean.
- Decoder after warmup worsened to `3.06 ms` median.
- Conclusion: reject. Keep decoder graph capture enabled.

Residual transpose squeeze/unsqueeze trial:
- Replaced each remaining residual `Transpose([1,0,2])` with a shape-preserving
  `Squeeze`/`Unsqueeze` pair in a temporary graph.
- CPU validation against the accepted dynamics artifact was exact for `final_z`,
  `candidate_k_entry`, and `candidate_v_entry`.
- Op counts changed `Transpose 107 -> 38`, with `69` added squeezes and `69` added unsqueezes.
- Browser timing remains unavailable: repeated system Chrome, bundled Chromium, and headless trial
  attempts failed during Chrome launch before page/model load.
- Conclusion: do not accept yet. The graph is CPU-exact, but the maintained artifact should not
  change without a passing browser measurement.

ORT runtime upgrade retest:
- Temporarily upgraded `onnxruntime-web` from `1.24.3` to `1.26.0`, rebuilt the browser bundle, and
  reran the accepted graph-capture benchmark.
- Result file: `webgpu_app/bench/results/graph_capture_latest.json`, created
  `2026-05-10T19:12:16.258Z`.
- Streaming after graph-capture warmup: `20.36 / 22.20 / 22.61 ms`, or `49.12 fps` by mean.
- Dynamics after warmup: `17.85 / 19.20 / 19.73 ms`.
- Decoder after warmup: `2.14 / 2.36 / 3.01 ms`.
- Conclusion: reject. Restored `onnxruntime-web@1.24.3`, rebuilt the browser bundle, and left
  `package.json` / `bun.lock` with no runtime-version diff from the accepted branch state.

Accepted baseline after restoring ORT 1.24.3:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Result file: `webgpu_app/bench/results/graph_capture_latest.json`, created
  `2026-05-10T19:12:56.485Z`.
- Streaming after graph-capture warmup: `19.49 / 22.10 / 22.39 ms`, or `51.31 fps` by mean.
- Dynamics after warmup: `17.04 / 19.10 / 19.70 ms`.
- Decoder after warmup: `2.09 / 2.34 / 2.86 ms`.
- The benchmark mean is around `51 fps`, but the median frame is still about `22 ms`, so this does
  not meet the consistent 60 FPS target.

Headless diagnostic run:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-headless --playwright-benchmark-attempts 3`.
- Result file: `webgpu_app/bench/results/graph_capture_latest.json`, created
  `2026-05-10T19:15:03.302Z`.
- Streaming after graph-capture warmup: `19.36 / 21.83 / 22.23 ms`, or `51.65 fps` by mean.
- Dynamics after warmup: `16.94 / 18.94 / 19.51 ms`.
- Decoder after warmup: `2.07 / 2.34 / 2.66 ms`.
- Conclusion: headless mode is useful for diagnostics and still uses the Apple Metal WebGPU
  adapter, but it does not close the 60 FPS gap.

Final verification after the canvas fallback/docs update:
- `bun run typecheck`: passed.
- `bun run build:webgpu:browser`: passed.
- `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 3`: passed, 5 tests.
- `bun run demo:webgpu:smoke -- --playwright-channel chrome --playwright-benchmark-attempts 2`:
  passed, 5 tests after the replacement-context hardening.
- `bun run demo:webgpu:smoke -- --demo-query '?backend=wasm' --playwright-channel chrome --playwright-benchmark-attempts 2`:
  passed, 5 tests after the replacement-context hardening.
- Normal non-headless graph-capture benchmark passed:
  - Command:
    `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
  - Result file: `webgpu_app/bench/results/graph_capture_latest.json`, created
    `2026-05-10T19:18:16.871Z`.
  - Streaming after graph-capture warmup: `19.46 / 22.05 / 22.46 ms`, or `51.40 fps` by mean.
  - Dynamics after warmup: `17.03 / 19.19 / 19.74 ms`.
  - Decoder after warmup: `2.07 / 2.34 / 2.40 ms`.

Artifact state:
- Accepted dynamics SHA-256:
  `cbdde638de674e73d4b70a4b8a420a26d570c238262ff6b48179a4961853ef64`.
- Accepted decoder SHA-256:
  `b9689f234ce20bb9dd1702f0c80710e5cb34b42c6682e0f0610af40ad52591af`.
- No `*full64*` or `*trial*` artifacts remain in `webgpu_app/dream_arcade_assets/breakout`.

## 2026-05-11 KST: Final Output-Head Slice Cleanup

Benchmark procedure note:
- Browser launch failures were reproducible when a large ONNX artifact was copied into place in the
  same shell command that launched Playwright. Staging the trial artifact in one command, then
  launching Playwright in a separate command, avoided the false pre-page Crashpad failures.

Residual transpose squeeze/unsqueeze retry:
- The CPU-exact residual `Transpose([1,0,2]) -> Squeeze/Unsqueeze` trial finally produced a browser
  timing with separated staging.
- Trial graph result (`2026-05-10T19:25:15.429Z`):
  - Streaming after graph-capture warmup: `19.58 / 21.90 / 22.32 ms`, or `51.08 fps`.
  - Dynamics after warmup: `18.05 / 20.19 / 20.63 ms`.
  - Decoder after warmup: `1.40 / 1.59 / 1.67 ms`.
- Same-window accepted repeat (`2026-05-10T19:25:49.722Z`):
  - Streaming after graph-capture warmup: `19.42 / 22.11 / 22.34 ms`, or `51.50 fps`.
  - Dynamics after warmup: `17.02 / 19.37 / 19.62 ms`.
- Conclusion: reject. The full-frame median was slightly lower, but dynamics got materially slower
  and mean FPS did not beat the accepted artifact.

Accepted final output-head slice rewrite:
- Added `rewrite_final_output_head_slice_transposes_for_webgpu()` to
  `scripts/webgpu/export_dreamer4_onnx.py` and wired it into
  `scripts/webgpu/specialize_full_cache_entry.py`.
- The pass targets the two final output-head islands:
  `Transpose([0,2,1,3]) -> Slice(axis=2, 4:36) -> Squeeze([0,1])`.
- It slices the pre-transpose tensor directly and retargets the following squeeze axes to `[0,2]`.
  The transposed axis is singleton, so the rewrite is exact.
- Regenerated maintained dynamics artifact:
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2.onnx`.
- New SHA-256:
  `7d208ea39260656a0422b35725e48e08c398dea917384efc7e169b09598ad3c1`.
- Node/op counts:
  - Nodes: `3000 -> 2998`.
  - `Transpose`: `107 -> 105`.
  - `Slice`: remains `2`.
- Validation:
  - Full-cache specialization validation against the cache-length source passed at
    `atol=5e-4`, `rtol=5e-4`.
  - Max / mean absolute errors:
    - `final_z`: `2.35e-05 / 7.05e-07`.
    - `candidate_k_entry`: `4.43e-05 / 9.36e-07`.
    - `candidate_v_entry`: `2.88e-05 / 4.19e-07`.
  - Direct CPU comparison between the regenerated artifact and the temporary final-slice trial was
    exact for `final_z`, `candidate_k_entry`, and `candidate_v_entry`.

Browser graph-capture results:
- Temporary final-slice trial (`2026-05-10T19:26:10.551Z`):
  - Streaming after warmup: `19.40 / 21.95 / 22.27 ms`, or `51.56 fps`.
  - Dynamics after warmup: `16.95 / 19.10 / 19.53 ms`.
- Temporary final-slice repeat (`2026-05-10T19:26:58.531Z`):
  - Streaming after warmup: `19.41 / 22.05 / 22.36 ms`, or `51.53 fps`.
  - Dynamics after warmup: `16.92 / 19.13 / 19.68 ms`.
- Regenerated maintained artifact (`2026-05-10T19:30:40.469Z`):
  - Streaming after warmup: `19.40 / 21.86 / 22.32 ms`, or `51.55 fps`.
  - Dynamics after warmup: `16.91 / 19.04 / 19.62 ms`.
  - Decoder after warmup: `2.13 / 2.35 / 2.73 ms`.
- Final verification run after rebuilding the browser bundle and demo smoke
  (`2026-05-10T19:32:01.075Z`):
  - Streaming after warmup: `19.76 / 21.96 / 22.29 ms`, or `50.61 fps`.
  - Dynamics after warmup: `17.12 / 18.97 / 19.55 ms`.
  - Decoder after warmup: `2.21 / 2.35 / 2.89 ms`.

Conclusion:
- Accept. This is a small exact cleanup and only a marginal browser improvement, but it is
  reproducible and keeps the maintained graph moving in the right direction.
- The demo is still not at consistent 60 FPS: the accepted median frame remains about `21.9 ms`,
  above the `16.67 ms` frame budget.

### Shared Gather Add Constant Fold

Change accepted:
- Added `fold_shared_gather_add_constants_for_webgpu()` and wired it into the full-cache
  specialization path after the final output-head slice rewrite.
- The pass targets a shared action embedding `Gather` whose only consumers are three
  `Add(Gather, identical_constant)` nodes.
- It folds the identical constant into the embedding initializer, rewires the three downstream
  consumers to the folded `Gather` output, and removes the three `Add` nodes.

Validation:
- Full-cache specialization validation against the cache-length source passed at `atol=5e-4`,
  `rtol=5e-4`.
- Direct CPU comparison against the previously accepted final-slice artifact was exact for
  `final_z`, `candidate_k_entry`, and `candidate_v_entry`.
- Node/op counts:
  - Nodes: `2998 -> 2995`.
  - `Add`: `76 -> 73`.
  - `Gather`: remains `238`.

Generated artifact:
- Dynamics:
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2.onnx`.
- SHA-256:
  `2c6f5ec24d1c56e07dabf9a26a61b09201ea5029e7b020ffdc18bedb3b57ec13`.
- Decoder SHA-256 remains:
  `b9689f234ce20bb9dd1702f0c80710e5cb34b42c6682e0f0610af40ad52591af`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Result file: `webgpu_app/bench/results/graph_capture_latest.json`, created
  `2026-05-10T19:36:31.323Z`.
- Streaming after graph-capture warmup: `19.53 / 22.00 / 22.23 ms`, or `51.21 fps`.
- Dynamics after warmup: `17.05 / 19.10 / 19.47 ms`.
- Decoder after warmup: `2.09 / 2.35 / 2.91 ms`.

Conclusion:
- Accept as an exact micro-cleanup. The improvement is within benchmark noise, but the graph is
  strictly smaller with no numerical change.
- Still not complete: median streaming frame time is about `22.0 ms`, above the 60 FPS frame budget.

### Live Demo Max-Speed Scheduler Cleanup

Change accepted:
- Updated the live demo stream loop to schedule uncapped follow-up frames with a `MessageChannel`
  task instead of always using `setTimeout(0)`.
- Added a pending-loop guard so repeated start/schedule events cannot queue duplicate generation
  loops.
- Starting the demo now schedules the first frame outside the click handler instead of awaiting
  inference in the UI event callback.

Scope:
- This is a TypeScript bridge / scheduling cleanup only. It does not modify ONNX artifacts, model
  inputs, cache update semantics, or the number of dynamics flow steps.
- Numerical validation against JAX is not applicable because no model graph or tensor math changed.

Verification:
- `bun run build:webgpu:browser`: passed.
- `bun run typecheck`: passed.
- `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 3`: passed, 5 tests.
- Model graph-capture benchmark still reflects the maintained inference bottleneck:
  - Command:
    `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
  - Result file: `webgpu_app/bench/results/graph_capture_latest.json`, created
    `2026-05-10T19:49:02.542Z`.
  - Streaming after graph-capture warmup: `19.37 / 21.98 / 22.26 ms`, or `51.64 fps`.
  - Dynamics after graph-capture warmup: `16.92 / 19.10 / 19.53 ms`.
  - Decoder after graph-capture warmup: `2.09 / 2.37 / 2.69 ms`.

Conclusion:
- Accept for live demo responsiveness and to avoid timer-clamp overhead in the uncapped loop.
- This does not close the model-inference FPS gap by itself; the maintained graph-capture median
  frame time still needs to get below `16.67 ms` for consistent 60 FPS.

### SwiGLU Rank-2 Island Rewrite

Change accepted:
- Added `rewrite_swiglu_rank2_islands_for_webgpu()` and wired it into
  `scripts/webgpu/specialize_full_cache_entry.py` after the existing exact full-cache graph
  cleanups.
- The pass targets 69 singleton-axis SwiGLU/MLP islands:
  `Unsqueeze -> QuickGelu`, sibling `Unsqueeze`, elementwise `Mul`, `Squeeze -> Gemm`,
  `Unsqueeze -> Add -> Transpose([1,0,2])`.
- It keeps `QuickGelu`, `Mul`, the output `Gemm`, and the residual `Add` rank-2, then restores the
  singleton axis at the residual boundary with `Unsqueeze`. This is exact because the removed axes
  are singleton and the changed operators are elementwise or already rank-2 `Gemm` operations.

Validation:
- Full-cache specialization validation against the cache-length source passed at `atol=5e-4`,
  `rtol=5e-4`.
- Direct CPU comparison against the previously accepted full-cache artifact was exact for
  `final_z`, `candidate_k_entry`, and `candidate_v_entry`.
- Node/op counts:
  - Nodes: `2995 -> 2788`.
  - `Unsqueeze`: `676 -> 538`.
  - `Transpose`: `105 -> 36`.
  - `Squeeze`: remains `317` because the pass replaces the post-Mul squeezes with residual
    squeezes.

Generated artifact:
- Dynamics:
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2.onnx`.
- SHA-256:
  `34be016e41f86510619eea76c011497b97fda55e5cdf9a5dd618630a600a2719`.
- Manifest hash matches the generated file, and the specialization report records
  `swiglu_rank2_island_rewrites: 69`.

Browser graph-capture results:
- Temporary trial (`2026-05-10T19:53:37.056Z`):
  - Streaming after warmup: `18.89 / 21.65 / 22.07 ms`, or `52.93 fps`.
  - Dynamics after warmup: `15.55 / 17.64 / 18.33 ms`.
  - Decoder after warmup: `3.08 / 3.44 / 3.88 ms`.
- Temporary trial repeat (`2026-05-10T19:54:19.726Z`):
  - Streaming after warmup: `19.18 / 21.79 / 22.33 ms`, or `52.13 fps`.
  - Dynamics after warmup: `15.87 / 18.01 / 18.44 ms`.
  - Decoder after warmup: `3.05 / 3.47 / 3.55 ms`.
- Same-window accepted repeat before accepting the rewrite (`2026-05-10T19:53:59.818Z`):
  - Streaming after warmup: `19.40 / 22.06 / 22.35 ms`, or `51.54 fps`.
  - Dynamics after warmup: `16.93 / 19.11 / 19.58 ms`.
  - Decoder after warmup: `2.11 / 2.37 / 2.92 ms`.
- Regenerated maintained artifact (`2026-05-10T19:58:35.159Z`):
  - Streaming after warmup: `19.13 / 21.88 / 22.31 ms`, or `52.26 fps`.
  - Dynamics after warmup: `15.69 / 17.86 / 18.49 ms`.
  - Decoder after warmup: `3.18 / 3.45 / 4.14 ms`.
- Final default repeat after the decoder graph-capture isolation (`2026-05-10T20:00:40.627Z`):
  - Streaming after warmup: `19.30 / 21.79 / 22.39 ms`, or `51.80 fps`.
  - Dynamics after warmup: `15.89 / 17.82 / 18.47 ms`.
  - Decoder after warmup: `3.16 / 3.46 / 4.12 ms`.

Verification:
- `.venv/bin/python -m py_compile scripts/webgpu/export_dreamer4_onnx.py scripts/webgpu/specialize_full_cache_entry.py`:
  passed.
- `bun run build:webgpu:browser`: passed.
- `bun run typecheck`: passed.
- First combined build/typecheck/smoke command hit the known pre-page Chrome Crashpad launch issue
  before loading any test page.
- Retried smoke in a fresh command:
  `bun run demo:webgpu:smoke -- --playwright-benchmark-attempts 3`: passed, 5 tests.
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`:
  passed.

Conclusion:
- Accept. This is the largest exact node-count reduction in the current window and materially
  improves the dynamics segment.
- Still not complete: the full streaming median remains about `21.9 ms`, above the `16.67 ms`
  frame budget for consistent 60 FPS. The next bottleneck to investigate is why the measured
  decoder segment increases to about `3.45 ms` after this dynamics-side rewrite.

Follow-up decoder graph-capture isolation:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-decoder-graph-capture false`.
- Result file copied to `/private/tmp/graph_capture_swiglu_rank2_decoder_gc_false.json`, created
  `2026-05-10T20:00:13.610Z`.
- Streaming after warmup: `18.93 / 21.84 / 22.59 ms`, or `52.82 fps`.
- Dynamics after warmup: `15.24 / 17.91 / 18.44 ms`.
- Decoder after warmup: `3.54 / 3.57 / 4.17 ms`.
- Conclusion: reject. Disabling decoder graph capture still worsens the decoder segment and does
  not materially improve full-frame median time.

### Decoder SwiGLU Rank-2 Add-Residual Rewrite

Change accepted:
- Extended `rewrite_swiglu_rank2_islands_for_webgpu()` to also cover decoder add-residual
  SwiGLU islands, including the final `Add -> Unsqueeze(axis=0) -> Transpose([0,2,1,3])`
  shape-restoration pattern.
- Applied the pass to the maintained single-frame decoder artifact
  `breakout_tokenizer_decoder_b1_t1.onnx`.
- The pass rewrote 4 decoder add-residual islands. It intentionally left the 4
  `SkipSimplifiedLayerNormalization` SwiGLU chains untouched.

Validation:
- Direct CPU comparison against the previously accepted decoder artifact was exact for `patches`
  at `atol=0`, `rtol=0`.
- Node/op counts:
  - Nodes: `335 -> 322`.
  - `Unsqueeze`: `75 -> 66`.
  - `Transpose`: `18 -> 14`.
  - `Squeeze`: remains `35`.

Generated artifact and manifests:
- Decoder SHA-256:
  `5285111dcd426121dfb31fe27d1eb958d2133d12dd9faeaf607ecc02008b472b`.
- Previous decoder SHA-256:
  `b9689f234ce20bb9dd1702f0c80710e5cb34b42c6682e0f0610af40ad52591af`.
- Updated `breakout_onnx_manifest.json` and `dream_arcade_assets_manifest.json` with the new
  decoder hash and artifact byte count.
- Result file copied to `/private/tmp/graph_capture_decoder_rank2_add_maintained.json`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `19.26 / 21.86 / 22.20 ms`, or `51.92 fps`.
- Dynamics after warmup: `15.89 / 17.96 / 18.50 ms`.
- Decoder after warmup: `2.85 / 3.12 / 3.72 ms`.

Verification:
- `.venv/bin/python -m py_compile scripts/webgpu/export_dreamer4_onnx.py scripts/webgpu/specialize_full_cache_entry.py`:
  passed.
- `bun run build:webgpu:browser`: passed.
- `bun run typecheck`: passed.
- `bun run demo:webgpu:smoke -- --playwright-channel chrome --playwright-benchmark-attempts 3`:
  passed, 5 tests.

Conclusion:
- Accept as an exact decoder graph cleanup. It improves the measured decoder segment versus the
  post-dynamics-rank-2 baseline, but full-frame median remains noise-bound around `21.9 ms`.
- Still not complete: the maintained graph-capture streaming frame is above the `16.67 ms` budget
  required for consistent 60 FPS.

### Dynamics SwiGLU Slice-Restore Rank-2 Extension

Change accepted:
- Extended `rewrite_swiglu_rank2_islands_for_webgpu()` to cover the two remaining dynamics
  SwiGLU islands whose residual output is restored through `Unsqueeze -> Slice -> Squeeze -> Gemm`
  rather than through a transpose.
- The new case keeps the SwiGLU activation/output projection rank-2, then changes the following
  shape restoration to `Unsqueeze(axes=[0, 2])` so the downstream slice sees the same
  `[1, 36, 1, 128]` tensor shape as before.

Validation:
- Temporary copy CPU comparison against the previously accepted dynamics artifact was exact at
  `atol=0`, `rtol=0` for `final_z`, `candidate_k_entry`, and `candidate_v_entry`.
- Maintained full-cache specialization validation against the cache-length source passed at
  `atol=5e-4`, `rtol=5e-4`.
- Node/op counts after the maintained regeneration:
  - Nodes: `2788 -> 2782` versus the previous accepted full-cache artifact.
  - `Unsqueeze`: `538 -> 532`.
  - `Squeeze`: remains `317`.
  - Full specialization report now records `swiglu_rank2_island_rewrites: 71`.

Generated artifact and manifests:
- Dynamics SHA-256:
  `0a85800175a0015f124e8c440da03dc45a99696964dc7912615cc10bda9c7290`.
- Previous dynamics SHA-256:
  `34be016e41f86510619eea76c011497b97fda55e5cdf9a5dd618630a600a2719`.
- Updated `breakout_onnx_manifest.json` and `dream_arcade_assets_manifest.json`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Result file copied to `/private/tmp/graph_capture_dynamics_rank2_slice_maintained.json`.
- Streaming after graph-capture warmup: `19.28 / 21.95 / 22.25 ms`, or `51.88 fps`.
- Dynamics after warmup: `15.88 / 17.94 / 18.52 ms`.
- Decoder after warmup: `2.88 / 3.13 / 3.72 ms`.

Verification:
- `.venv/bin/python -m py_compile scripts/webgpu/export_dreamer4_onnx.py scripts/webgpu/specialize_full_cache_entry.py`:
  passed.
- `bun run build:webgpu:browser`: passed.
- `bun run typecheck`: passed.
- `bun run demo:webgpu:smoke -- --playwright-channel chrome --playwright-benchmark-attempts 3`:
  passed, 5 tests.

Profiling note:
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 3 --webgpu-benchmark-profiling true --webgpu-benchmark-profiling-required true --webgpu-benchmark-profiling-top-k 50`
  reached the page but failed because ORT emitted zero `ort.env.webgpu.profiling.ondata` events.
- Re-running without `--webgpu-benchmark-profiling-required true` passed, but the result still
  contained `event_count: 0`, so WebGPU profiling is not currently usable on this setup.

Conclusion:
- Accept as an exact graph cleanup. It removes the final two rank-3 SwiGLU activation islands in
  the maintained full-cache dynamics graph.
- It is performance-neutral in the browser benchmark. The full streaming median remains about
  `21.9 ms`, still above the `16.67 ms` 60 FPS frame budget.

### Graph-Capture Output Buffer Reuse

Change accepted:
- The benchmark now allows preallocated GPU output fetch tensors when graph capture is enabled
  instead of only using them on the non-graph-capture hot path.
- The live demo now preallocates pinned WebGPU output tensors for graph-captured step outputs and
  decoder `patches`, reuses those tensors across frames, and skips disposal for pinned outputs.
- CPU and non-graph-capture fetch paths still use the previous output-name arrays.

Scope:
- This is a JS/WebGPU bridge allocation cleanup only. It does not modify model math, ONNX graph
  semantics, cache update order, or the number of dynamics flow steps.
- Numerical validation against JAX is not applicable because model inputs/outputs and graph
  operations are unchanged.

Browser graph-capture results:
- Trial 1:
  - Streaming after warmup: `19.07 / 21.88 / 22.18 ms`, or `52.44 fps`.
  - Dynamics after warmup: `15.69 / 17.82 / 18.44 ms`.
  - Decoder after warmup: `2.86 / 3.11 / 3.77 ms`.
- Trial repeat:
  - Streaming after warmup: `19.23 / 21.86 / 22.31 ms`, or `52.00 fps`.
  - Dynamics after warmup: `15.81 / 18.01 / 18.55 ms`.
  - Decoder after warmup: `2.90 / 3.12 / 3.75 ms`.
- Accepted run:
  - Command:
    `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
  - Result file copied to `/private/tmp/graph_capture_preallocated_graph_capture_fetches_accepted.json`.
  - Streaming after warmup: `19.08 / 21.90 / 22.42 ms`, or `52.41 fps`.
  - Dynamics after warmup: `15.75 / 17.90 / 18.50 ms`.
  - Decoder after warmup: `2.81 / 3.13 / 3.64 ms`.

Verification:
- `bun run build:webgpu:browser`: passed.
- `bun run typecheck`: passed.
- `bun run demo:webgpu:smoke -- --playwright-channel chrome --playwright-benchmark-attempts 3`:
  passed, 5 tests.

Conclusion:
- Accept as a small bridge cleanup. Reusing fixed graph-capture output buffers avoids avoidable
  per-frame GPU tensor allocation/disposal and slightly improves the observed benchmark median in
  this run window.
- Still not complete: the maintained graph-capture full-frame median remains about `21.9 ms`, above
  the `16.67 ms` 60 FPS budget.

### Rejected: Direct GQA Concat/Gather Fold

Trial:
- Prototyped replacing direct `Concat(two compact value heads) -> Gather(axis=2, [0,0,0,0,1,1,1,1])`
  islands with one wider `Concat` whose inputs repeat the two head tensors in gather order.
- The trial removed 36 `Gather` nodes:
  - Nodes: `2782 -> 2746`.
  - `Gather`: `238 -> 202`.
- CPU comparison against the accepted dynamics artifact was exact at `atol=0`, `rtol=0` for
  `final_z`, `candidate_k_entry`, and `candidate_v_entry`.

Browser graph-capture results:
- Trial 1:
  - Streaming after warmup: `19.14 / 21.85 / 22.45 ms`, or `52.24 fps`.
  - Dynamics after warmup: `16.61 / 18.86 / 19.54 ms`.
  - Decoder after warmup: `2.17 / 2.39 / 2.80 ms`.
- Trial repeat:
  - Streaming after warmup: `19.10 / 21.77 / 22.40 ms`, or `52.36 fps`.
  - Dynamics after warmup: `16.61 / 18.81 / 19.47 ms`.
  - Decoder after warmup: `2.15 / 2.38 / 3.01 ms`.

Conclusion:
- Reject. Although full-frame median moved slightly, mean FPS was not better than the accepted
  output-buffer-reuse run and the measured dynamics segment regressed consistently by about
  `0.9 ms`.
- Restored the accepted dynamics artifact:
  `0a85800175a0015f124e8c440da03dc45a99696964dc7912615cc10bda9c7290`.

### Rejected: QKV SimplifiedLayerNormalization Rank-2 Branch

Trial:
- Prototyped bypassing the shared singleton `Unsqueeze` only on the QKV
  `SimplifiedLayerNormalization` branch:
  `Unsqueeze -> SimplifiedLayerNormalization(axis=2) -> Squeeze -> Gemm`
  became `SimplifiedLayerNormalization(axis=1) -> Gemm`.
- The shared `Unsqueeze` was kept for the residual `SkipSimplifiedLayerNormalization` branch.
- Added branch-local squeezed scale initializers because the original SLN scale tensors are
  `[1, 1, 128]`.
- The trial removed 68 `Squeeze` nodes:
  - Nodes: `2782 -> 2714`.
  - `Squeeze`: `317 -> 249`.
- CPU comparison against the accepted dynamics artifact was exact at `atol=0`, `rtol=0` for
  `final_z`, `candidate_k_entry`, and `candidate_v_entry`.

Browser graph-capture results:
- Trial 1:
  - Streaming after warmup: `19.22 / 22.03 / 22.36 ms`, or `52.04 fps`.
  - Dynamics after warmup: `15.81 / 17.92 / 18.59 ms`.
  - Decoder after warmup: `2.88 / 3.14 / 3.77 ms`.
- Trial repeat:
  - Streaming after warmup: `19.16 / 21.88 / 22.34 ms`, or `52.20 fps`.
  - Dynamics after warmup: `15.77 / 17.92 / 18.51 ms`.
  - Decoder after warmup: `2.86 / 3.12 / 3.76 ms`.

Conclusion:
- Reject. The graph is smaller and exact, but the browser benchmark was neutral to slightly worse
  than the accepted output-buffer-reuse artifact.
- Restored the accepted dynamics artifact:
  `0a85800175a0015f124e8c440da03dc45a99696964dc7912615cc10bda9c7290`.

### Live Demo Static Graph-Capture Scalars

Change accepted:
- The live demo now initializes graph-capture scalar and attention-mask GPU tensors with their
  full-cache steady-state values when the runtime is created.
- During generated frames, the live graph-capture path now updates only the dynamic tensors
  (`sample_noise`, optional `context_noise`, and `actions`) instead of rewriting fixed
  `cache_length`, position indices, and `attention_mask` every frame.
- This aligns the live demo bridge with the benchmark, which already uses fixed graph-capture
  scalar tensors.

Scope:
- TypeScript bridge cleanup only. ONNX graphs, cache semantics, model math, and dynamics flow-step
  count are unchanged.
- Numerical validation against JAX is not applicable because the fixed values are the same
  full-cache values already supplied to graph-captured sessions.

Verification:
- `bun run build:webgpu:browser`: passed.
- `bun run typecheck`: passed.
- `bun run demo:webgpu:smoke -- --playwright-channel chrome --playwright-benchmark-attempts 3`:
  passed, 5 tests.
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`:
  passed with the restored accepted artifacts.
  - Result copied to `/private/tmp/graph_capture_after_live_static_scalars.json`.
  - Streaming after warmup: `19.37 / 22.01 / 22.32 ms`, or `51.62 fps`.
  - Dynamics after warmup: `15.99 / 17.97 / 18.55 ms`.
  - Decoder after warmup: `2.87 / 3.13 / 3.70 ms`.

Conclusion:
- Accept as a live-demo bridge cleanup. It removes avoidable CPU tensor allocation and GPU writes
  from each graph-captured generated frame.
- It is not counted as a model-benchmark FPS improvement; the benchmark already used fixed scalar
  tensors and the latest browser run remains in the same `~21.9-22.0 ms` median range.

### Dynamics Final-Z / Decoder Input Buffer Sharing

Change accepted:
- The benchmark and live demo now reuse the decoder's fixed graph-capture input tensor as the
  preallocated dynamics `final_z` output buffer when dtype and shape match.
- This makes the dynamics output tensor already be the decoder input tensor, so the existing
  `copyTensorToGpu()` source-equals-target guard skips the per-frame GPU copy before decoder
  inference.

Scope:
- TypeScript bridge cleanup only. Model math, ONNX graph semantics, cache update semantics, and
  dynamics flow-step count are unchanged.
- Numerical validation against JAX is not applicable because the same `final_z` bytes are written
  to the same decoder input buffer instead of being copied through an intermediate GPU buffer.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Result copied to `/private/tmp/graph_capture_finalz_into_decoder_input_accepted.json`.
- Streaming after warmup: `19.34 / 21.95 / 22.20 ms`, or `51.72 fps`.
- Dynamics after warmup: `15.94 / 17.84 / 18.48 ms`.
- Decoder after warmup: `2.86 / 3.13 / 3.73 ms`.
- Pack/copy segment after warmup: `0.087 / 0.100 / 0.130 ms`.

Verification:
- `bun run build:webgpu:browser`: passed.
- `bun run typecheck`: passed.
- `bun run demo:webgpu:smoke -- --playwright-channel chrome --playwright-benchmark-attempts 3`:
  passed, 5 tests.

Conclusion:
- Accept as a narrow bridge cleanup. It removes an avoidable GPU copy in the graph-captured
  dynamics-to-decoder path.
- It is not a meaningful model-benchmark improvement; full-frame median remains about `21.9 ms`,
  above the `16.67 ms` 60 FPS budget.

### 2D Canvas Unavailable Fallback Hardening

Change accepted:
- The CPU frame render path no longer throws when every `getContext('2d')` attempt fails.
- If 2D canvas acquisition is unavailable even after replacing the display canvas, the demo now
  renders `ImageData` through an `<img class="frame-fallback">` backed by a generated BMP blob.
- The WebGPU/canvas render path hides the fallback image again when a canvas-backed renderer is
  restored.

Scope:
- TypeScript/CSS demo rendering fallback only. ONNX graphs, model outputs, cache semantics, and
  benchmark timing paths are unchanged.

Verification:
- `bun run build:webgpu:browser`: passed.
- `bun run typecheck`: passed.
- `bun run demo:webgpu:smoke -- --playwright-channel chrome --playwright-benchmark-attempts 3`:
  passed, 6 tests.

### Rejected: No-Bias 2D Gemm to MatMul

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_gemm2matmul2d_trial.onnx`.
- Replaced the `286` no-bias `Gemm` nodes with `alpha=1`, `beta=0`, `transA=0`, and `transB=0`
  by plain `MatMul` nodes, leaving tensor ranks and surrounding layout nodes unchanged.
- The goal was to test ORT WebGPU's `MatMul` dispatch path without repeating the earlier
  rank-aware MatMul layout rewrites.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact for two deterministic
  input sets:
  - `final_z`: max/mean/p95 absolute error `0 / 0 / 0`.
  - `candidate_k_entry`: max/mean/p95 absolute error `0 / 0 / 0`.
  - `candidate_v_entry`: max/mean/p95 absolute error `0 / 0 / 0`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_gemm2matmul2d_trial`.
- Streaming after graph-capture warmup: `0.72 / 21.82 / 22.25 ms`, or `52.90 fps`.
- Dynamics after graph-capture warmup: `0.61 / 18.88 / 19.39 ms`.
- Decoder after graph-capture warmup: `0.09 / 2.44 / 2.87 ms`.

Same-session accepted comparison:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.73 / 21.72 / 22.46 ms`, or `52.60 fps`.
- Dynamics after graph-capture warmup: `0.61 / 17.75 / 18.52 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.11 / 3.76 ms`.

Conclusion:
- Reject. The rewrite is exact, but the ORT WebGPU `MatMul` path is slower than the accepted
  `GemmShared` path for the dynamics graph. The decoder median improved in this browser run, but
  the full streaming median and dynamics median did not beat the accepted artifact.
- Removed the temporary trial artifact and manifest entry; no served `*trial*` artifacts remain.

### Rejected: ORT WebGPU GemmShared Tile Size

Trial:
- Temporarily switched the benchmark import from `ort.webgpu.bundle.min.mjs` to the readable
  `ort.all.mjs`.
- Temporarily changed ORT WebGPU's `GemmShared` tile size in
  `node_modules/onnxruntime-web/dist/ort.all.mjs` from `16` to `32`, then to `8`.
- Model graphs, manifest entries, cache update behavior, and dynamics flow-step count were
  unchanged. The trial only changed the runtime dispatch geometry for `Gemm`.

Browser graph-capture results:
- Tile size `32`:
  - Command:
    `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
  - Streaming after graph-capture warmup: `0.76 / 22.00 / 22.34 ms`, or `52.21 fps`.
  - Dynamics after graph-capture warmup: `0.65 / 18.12 / 18.56 ms`.
  - Decoder after graph-capture warmup: `0.09 / 3.14 / 3.57 ms`.
- Tile size `8`:
  - Command:
    `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
  - Streaming after graph-capture warmup: `0.77 / 21.71 / 22.11 ms`, or `52.69 fps`.
  - Dynamics after graph-capture warmup: `0.66 / 17.81 / 18.44 ms`.
  - Decoder after graph-capture warmup: `0.09 / 3.11 / 3.75 ms`.

Restored accepted comparison:
- Restored the benchmark import to `ort.webgpu.bundle.min.mjs` and restored `GemmShared` tile size
  `16` in the readable bundle.
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.79 / 22.12 / 22.31 ms`, or `51.99 fps`.
- Dynamics after graph-capture warmup: `0.66 / 17.96 / 18.62 ms`.
- Decoder after graph-capture warmup: `0.08 / 3.14 / 3.75 ms`.

Conclusion:
- Reject. Tile size `32` is clearly worse for the dynamics median. Tile size `8` is effectively
  neutral and does not improve the full streaming median enough to justify carrying a patched ORT
  runtime.
- Restored the ORT bundle import and readable bundle edit; the current result file is a passing
  accepted-artifact run.

### Rejected: Alias Graph-Capture Noise Inputs

Trial:
- Temporarily changed the benchmark bridge so graph-captured `sample_noise` and `context_noise`
  used the same fixed GPU tensor, with duplicate copy targets deduplicated.
- This was intended to test whether avoiding one tiny per-frame GPU copy could reduce the streaming
  frame overhead outside the dynamics and decoder sessions.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.70 / 21.98 / 22.18 ms`, or `52.11 fps`.
- Dynamics after graph-capture warmup: `0.60 / 18.06 / 18.50 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.13 / 3.74 ms`.

Restored accepted comparison:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.76 / 21.96 / 22.43 ms`, or `51.37 fps`.
- Dynamics after graph-capture warmup: `0.63 / 17.90 / 18.76 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.13 / 3.80 ms`.

Conclusion:
- Reject. The benchmark-only alias is not a valid live-demo change because the demo's frame input
  state fills `sample_noise` and `context_noise` as distinct normal tensors. The timing result is
  also neutral.
- Reverted the benchmark bridge and restored the current result file to a passing accepted-artifact
  run.

### Rejected Before Benchmark: One-Position RoPE Identity Bypass

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_onepos_rope_identity_trial.onnx`.
- Bypassed the `71` temporal `RotaryEmbedding` nodes whose names include
  `direct_repeated_onepos`, rewiring consumers directly to the pre-RoPE tensor.
- The hypothesis was that these fixed one-position branches might be identity rotations and could
  remove 71 dispatches.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact failed with large differences:
  - Seed `1234`:
    - `final_z` max/mean/p95 absolute error: `2.88 / 0.455 / 1.30`.
    - `candidate_k_entry` max/mean/p95 absolute error: `18.21 / 1.41 / 5.96`.
    - `candidate_v_entry` max/mean/p95 absolute error: `5.56 / 0.212 / 0.892`.
  - Seed `5678`:
    - `final_z` max/mean/p95 absolute error: `2.49 / 0.280 / 0.839`.
    - `candidate_k_entry` max/mean/p95 absolute error: `19.81 / 1.35 / 5.77`.
    - `candidate_v_entry` max/mean/p95 absolute error: `4.26 / 0.158 / 0.682`.

Conclusion:
- Reject without browser benchmarking. The one-position RoPE nodes are not identity in the accepted
  artifact, despite the zero-position-looking naming.
- Removed the temporary artifact and restored the manifest; no served `*trial*` artifacts remain.

### Rejected: ORT WebGPU Non-Shared Gemm Shader

Trial:
- Temporarily switched the benchmark import from `ort.webgpu.bundle.min.mjs` to the readable
  `ort.all.mjs`.
- Temporarily changed ORT WebGPU `Gemm` lowering from the shared-memory `GemmShared` shader to the
  simple per-output shader by setting the internal `useShared` flag to `false`.
- Model artifacts, cache behavior, graph capture, and dynamics flow-step count were unchanged.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.78 / 21.97 / 22.25 ms`, or `51.89 fps`.
- Dynamics after graph-capture warmup: `0.67 / 18.03 / 18.54 ms`.
- Decoder after graph-capture warmup: `0.08 / 3.12 / 3.29 ms`.

Restored accepted comparison:
- Restored the benchmark import to `ort.webgpu.bundle.min.mjs` and restored ORT's shared-memory
  `GemmShared` path.
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.77 / 21.87 / 22.26 ms`, or `52.41 fps`.
- Dynamics after graph-capture warmup: `0.64 / 17.88 / 18.54 ms`.
- Decoder after graph-capture warmup: `0.08 / 3.13 / 3.81 ms`.

Conclusion:
- Reject. The simple Gemm shader is neutral to slightly worse for the dynamics median and does not
  move the full streaming frame toward consistent 60 fps.
- Restored the ORT bundle import and readable bundle edit; the current result file is a passing
  accepted-artifact run.

### Rejected: Cache Layer Gather to Split Fan-Out

Diagnostic:
- Tried to collect ORT WebGPU profiling events on the non-graph-captured benchmark path with:
  `bun run benchmark:webgpu -- --grep "webgpu demo streaming benchmark$" --playwright-channel chrome --playwright-benchmark-attempts 3 --webgpu-benchmark-profiling 1 --webgpu-benchmark-profiling-drain-ms 1000 --webgpu-benchmark-profiling-top-k 40 --webgpu-benchmark-timed-runs 8`.
- The run failed before reaching the benchmark page on all wrapper attempts because Chrome aborted
  during startup with crashpad bootstrap permission errors. No timing or profiling data was
  collected.

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_cache_layer_split_trial.onnx`.
- Replaced the `24` per-layer `Gather(axis=0)` cache reads over `k_cache` and `v_cache` with two
  `Split(axis=0)` nodes, one per cache tensor.
- Because `Gather` removes the layer axis but `Split` preserves it, rewrote the matching `24`
  cache-layer `Squeeze` nodes to squeeze both the layer and singleton batch axes.
- Public graph inputs/outputs, cache ABI, and dynamics flow-step count were unchanged.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact for two deterministic
  input sets:
  - `final_z`: max/mean/p95 absolute error `0 / 0 / 0`.
  - `candidate_k_entry`: max/mean/p95 absolute error `0 / 0 / 0`.
  - `candidate_v_entry`: max/mean/p95 absolute error `0 / 0 / 0`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_cache_layer_split_trial`.
- Failed on all wrapper attempts with:
  `Too many storage buffers in shader. Current: 11, Max is 10`.

Restored accepted comparison:
- Removed the temporary trial artifact and manifest entry.
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.74 / 21.87 / 22.38 ms`, or `52.01 fps`.
- Dynamics after graph-capture warmup: `0.62 / 17.94 / 18.52 ms`.
- Decoder after graph-capture warmup: `0.08 / 3.13 / 3.77 ms`.

Conclusion:
- Reject. The graph rewrite is exact but not ORT WebGPU compatible for this adapter because the
  12-output `Split` shader exceeds the storage-buffer-per-shader limit.
- No served `*trial*` artifacts remain; the current result file is a passing accepted-artifact run.

### Rejected: Cache Layer Gather to Hierarchical Split Fan-Out

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_cache_layer_split6_trial.onnx`.
- Replaced the `24` per-layer `Gather(axis=0)` cache reads with a two-stage Split fan-out per cache
  tensor:
  - first Split outputs layers `0..5` plus a tail chunk for layers `6..11`;
  - second Split expands the tail chunk to layers `6..11`.
- This avoided the storage-buffer limit hit by the single 12-output Split trial while preserving
  public inputs/outputs, cache ABI, and dynamics flow-step count.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact for two deterministic
  input sets:
  - `final_z`: max/mean/p95 absolute error `0 / 0 / 0`.
  - `candidate_k_entry`: max/mean/p95 absolute error `0 / 0 / 0`.
  - `candidate_v_entry`: max/mean/p95 absolute error `0 / 0 / 0`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_cache_layer_split6_trial`.
- Streaming after graph-capture warmup: `0.74 / 21.92 / 22.32 ms`, or `52.41 fps`.
- Dynamics after graph-capture warmup: `0.65 / 19.24 / 19.77 ms`.
- Decoder after graph-capture warmup: `0.08 / 2.31 / 2.39 ms`.

Same-session accepted comparison:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.71 / 21.83 / 22.41 ms`, or `51.53 fps`.
- Dynamics after graph-capture warmup: `0.61 / 17.83 / 18.58 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.13 / 3.76 ms`.

Conclusion:
- Reject. The storage-limit-safe Split variant is exact and browser-compatible, but it regresses the
  dynamics median by about `1.4 ms`; the decoder improvement in this run does not translate into a
  better full streaming median.
- Removed the temporary trial artifact and manifest entry.

### Rejected: Commute Spatial Q Transpose And SLN

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_transpose_sln_commute_trial.onnx`.
- Reordered the `36` spatial Q-branch
  `Transpose([0,2,1,3]) -> SimplifiedLayerNormalization(axis=3)` pairs to
  `SimplifiedLayerNormalization(axis=3) -> Transpose([0,2,1,3])`.
- This is exact because the normalization axis is the last dimension and the transpose only swaps
  the head and sequence axes.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact for two deterministic
  input sets:
  - `final_z`: max/mean/p95 absolute error `0 / 0 / 0`.
  - `candidate_k_entry`: max/mean/p95 absolute error `0 / 0 / 0`.
  - `candidate_v_entry`: max/mean/p95 absolute error `0 / 0 / 0`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_transpose_sln_commute_trial`.
- Streaming after graph-capture warmup: `0.79 / 21.88 / 22.28 ms`, or `51.69 fps`.
- Dynamics after graph-capture warmup: `0.65 / 17.89 / 18.57 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.11 / 3.54 ms`.

Same-session accepted comparison:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.68 / 21.88 / 22.33 ms`, or `52.16 fps`.
- Dynamics after graph-capture warmup: `0.59 / 17.92 / 18.59 ms`.
- Decoder after graph-capture warmup: `0.08 / 3.12 / 3.75 ms`.

Conclusion:
- Reject. The reorder is exact and browser-compatible, but it is neutral and does not improve the
  full streaming median.
- Removed the temporary trial artifact and manifest entry.

### Rejected: Decoder No-Bias 2D Gemm to MatMul

Trial:
- Built a temporary decoder artifact
  `breakout_tokenizer_decoder_b1_t1_gemm2matmul_trial.onnx`.
- Replaced the `32` decoder no-bias `Gemm` nodes with `alpha=1`, `beta=0`, `transA=0`, and
  `transB=0` by plain `MatMul` nodes.
- Temporarily set `demo_generation.preferred_decoder_export` to the trial decoder while keeping the
  accepted dynamics artifact unchanged.

Validation:
- CPU comparison against the accepted decoder artifact was exact for two deterministic input sets:
  - `patches`: max/mean/p95 absolute error `0 / 0 / 0`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.77 / 21.85 / 22.33 ms`, or `52.11 fps`.
- Dynamics after graph-capture warmup: `0.66 / 17.88 / 18.53 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.14 / 3.66 ms`.

Restored accepted comparison:
- Restored the accepted decoder artifact preference and removed the trial artifact/manifest entry.
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.76 / 21.81 / 22.30 ms`, or `52.44 fps`.
- Dynamics after graph-capture warmup: `0.64 / 17.84 / 18.54 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.13 / 3.79 ms`.

Conclusion:
- Reject. The decoder rewrite is exact, but `MatMul` does not improve decoder or full-frame median
  timing compared with the accepted `GemmShared` path.
- Removed the temporary decoder artifact and manifest entry.

### Rejected: Start Decoder Before Cache Commit

Trial:
- Temporarily changed the benchmark bridge so it started the decoder `session.run()` promise, then
  committed the K/V entry cache while the decoder promise was in flight, and awaited the decoder
  afterward.
- Model artifacts, graph math, cache contents, decoder inputs, and dynamics flow-step count were
  unchanged. The intent was to overlap a small amount of JS/cache-submit overhead with the decoder
  path.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.77 / 21.92 / 22.22 ms`, or `52.06 fps`.
- Dynamics after graph-capture warmup: `0.63 / 18.07 / 18.51 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.13 / 3.44 ms`.

Restored accepted comparison:
- Restored the original benchmark bridge ordering: await decoder, then commit cache.
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.82 / 21.85 / 22.33 ms`, or `52.24 fps`.
- Dynamics after graph-capture warmup: `0.69 / 17.88 / 18.62 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.13 / 3.78 ms`.

Conclusion:
- Reject. Starting the decoder before cache commit is neutral to slightly worse in the measured
  frame path and does not improve the model FPS.
- Reverted the benchmark bridge change.

### Rejected: Unfold Bias Gemm C Inputs

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_gemm_bias_add_unfold_trial.onnx`.
- Split the `5` remaining `Gemm` nodes with C/bias inputs into no-bias `Gemm` plus explicit
  `Add`.
- The motivation was the reverse of the rejected residual-add fold: ORT WebGPU's C-input `Gemm`
  path has been slower in this graph, so avoiding C-input `Gemm` for the few true bias projections
  might help.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact for two deterministic
  input sets:
  - `final_z`: max/mean/p95 absolute error `0 / 0 / 0`.
  - `candidate_k_entry`: max/mean/p95 absolute error `0 / 0 / 0`.
  - `candidate_v_entry`: max/mean/p95 absolute error `0 / 0 / 0`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_gemm_bias_add_unfold_trial`.
- Failed on all wrapper attempts with:
  `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`.

Restored accepted comparison:
- Removed the temporary trial artifact and manifest entry.
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.75 / 21.76 / 22.24 ms`, or `52.35 fps`.
- Dynamics after graph-capture warmup: `0.64 / 17.98 / 18.49 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.11 / 3.72 ms`.

Conclusion:
- Reject. The rewrite is exact but not graph-capture compatible with the current ORT WebGPU
  preallocated-output path.
- Removed the temporary trial artifact and manifest entry.

### Rejected: Hot Feed-Map Reuse in Benchmark and Demo Bridge

Trial:
- Precomputed the cached-step feed names in `webgpu_app/bench/benchmark.ts` and reused mutable
  feed objects for warmup/timed step and decoder calls instead of cloning feed maps and resolving
  input names before each `session.run`.
- Mirrored the same allocation cleanup in `webgpu_app/demo/main.ts` for graph-captured step feeds
  and fixed decoder input feeds.
- This changed only JavaScript object allocation around identical ORT tensor feeds; ONNX graphs,
  tensor values, cache updates, and the number of dynamics flow steps were unchanged.

Validation:
- `bun run typecheck`: passed.
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`:
  passed.
- Repeat of the same benchmark: passed.

Results:
- First run after the feed-map reuse patch:
  - Streaming after graph-capture warmup: median `21.88 ms`, `52.08 fps`.
  - Dynamics after graph-capture warmup: median `17.90 ms`.
  - Decoder after graph-capture warmup: median `3.14 ms`.
- Repeat:
  - Streaming after graph-capture warmup: median `21.86 ms`, `52.06 fps`.
  - Dynamics after graph-capture warmup: median `17.85 ms`.
  - Decoder after graph-capture warmup: median `3.13 ms`.

Conclusion:
- Reject. The cleanup did not improve the measured graph-capture FPS and stayed slightly below the
  accepted same-machine range.
- Reverted the feed-map reuse changes.
- Restored accepted graph-capture benchmark:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`
  passed with:
  - Streaming after graph-capture warmup: median `21.75 ms`, `52.65 fps`.
  - Dynamics after graph-capture warmup: median `17.92 ms`.
  - Decoder after graph-capture warmup: median `3.11 ms`.

### Rejected Before Artifact: Batch Same-Weight Gemm Groups Across Unrolled Branches

Observation:
- The accepted full-cache dynamics graph has `291` `Gemm` nodes.
- `289` of those `Gemm` nodes share weights in repeated groups, and `92` groups have exactly three
  same-weight, same-output-shape `Gemm` nodes that initially looked batchable.

Trial plan:
- Replace each candidate group with `Concat(axis=0) -> Gemm -> Split(axis=0)` so the same weight
  runs once over a taller row batch, then restore the original per-branch outputs.

Preflight result:
- Rebuilding the graph with all candidate groups could not be topologically sorted.
- A dependency audit showed `0` acyclic groups and `92` cyclic groups: each repeated-weight group is
  an unrolled sequential reuse, not independent parallel branches.

Conclusion:
- Reject before creating a served trial artifact. Batching these `Gemm` calls would change the
  graph dependencies or create cycles, so it is not a behavior-preserving optimization.
- No temporary artifact was written into `webgpu_app/dream_arcade_assets/breakout`.

### Rejected: WebGPU Profiling Drain Retest

Trial:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-profiling 1 --webgpu-benchmark-profiling-drain-ms 1000 --webgpu-benchmark-profiling-top-k 40`.

Result:
- Benchmark passed, but profiling still reported:
  - `enabled: true`
  - `event_count: 0`
  - `top_programs: []`
- The profiled run was slower/noisier:
  - Streaming after graph-capture warmup: median `21.89 ms`, `51.27 fps`.
  - Dynamics after graph-capture warmup: median `17.99 ms`.
  - Decoder after graph-capture warmup: median `3.13 ms`.

Conclusion:
- Reject. ORT WebGPU profiling remains unusable for this graph-captured benchmark path, even with a
  longer drain.
- Restored the normal non-profiled benchmark result after this retest.
  - Streaming after graph-capture warmup: median `21.88 ms`, `52.47 fps`.
  - Dynamics after graph-capture warmup: median `17.91 ms`.
  - Decoder after graph-capture warmup: median `3.13 ms`.

### Rejected: ORT WebGPU Replay Dispatch Batch Size

Trial:
- Temporarily switched the benchmark import from
  `/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs` to the readable
  `/node_modules/onnxruntime-web/dist/ort.all.mjs`.
- Temporarily changed ORT WebGPU's internal `maxDispatchNumber` from `16` to `256` in
  `node_modules/onnxruntime-web/dist/ort.all.mjs`.
- Motivation: graph-capture replay flushes the command encoder every `maxDispatchNumber`
  dispatches. If queue submission or compute-pass fragmentation were the hidden bottleneck, a
  larger batch should improve the graph-captured steady state.
- This was a runtime-only experiment. ONNX graphs, tensor values, cache updates, and dynamics flow
  step count were unchanged.

Result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Benchmark passed with:
  - Streaming after graph-capture warmup: median `21.90 ms`, `52.54 fps`.
  - Dynamics after graph-capture warmup: median `17.83 ms`.
  - Decoder after graph-capture warmup: median `3.12 ms`.

Conclusion:
- Reject. Increasing the replay dispatch batch size is neutral relative to the accepted
  same-machine run (`52.47 fps`), so the remaining gap is not primarily caused by ORT's default
  16-dispatch flush cadence.
- Restored the benchmark import and the temporary `ort.all.mjs` edit.

### Rejected: ONNX Runtime Web 1.25.1 Without Graph-Capture Preallocated Outputs

Trial:
- Temporarily installed `onnxruntime-web@1.25.1`.
- Temporarily restored the older benchmark fetch policy for graph-capture runs:
  `usePreallocatedHotOutputs = gpuDevice && !debugStats && !graphCapture`.
- Motivation: the earlier `1.25.1` retest failed with
  `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`; this checked whether
  that failure was caused only by preallocated output tensors.

Result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- All five wrapper attempts failed with the same error:
  `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`.

Conclusion:
- Reject. ORT Web 1.25.1 is still incompatible with the current graph-captured benchmark path even
  when graph-capture preallocated outputs are disabled.
- Restored `onnxruntime-web@1.24.3`, restored the accepted graph-capture preallocated-output fetch
  policy, and restored the dependency range to `^1.24.3`.
- Accepted graph-capture benchmark after restore:
  - Streaming after graph-capture warmup: median `21.84 ms`, `52.19 fps`.
  - Dynamics after graph-capture warmup: median `17.93 ms`.
  - Decoder after graph-capture warmup: median `3.12 ms`.
- Added a smoke test that forces `HTMLCanvasElement.prototype.getContext('2d')` to throw and
  verifies the `.frame-fallback` renderer appears without a page error.

### Rejected: Rank-3 Add Plus Pre-Norm SkipSimplifiedLayerNormalization Fusion

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_add_sln_skip_sln_rank3_trial.onnx`.
- Kept the new `SkipSimplifiedLayerNormalization` sites rank-3 to match the accepted graph-capture
  shape pattern, replacing `68` MLP residual `Add -> Unsqueeze -> SimplifiedLayerNormalization`
  islands while leaving the following `Squeeze` nodes in place.
- Node count stayed at `2782`.
- Op deltas:
  - `Add`: `73 -> 5`.
  - `Unsqueeze`: `532 -> 600`.
  - `SimplifiedLayerNormalization`: `215 -> 147`.
  - `SkipSimplifiedLayerNormalization`: `71 -> 139`.
  - `Squeeze`: unchanged at `317`.
- Trial SHA-256:
  `2c2dd77132d0dac3ac1640ea91c8649fb79c2df8b85d4ae07a1b4f04652971f7`.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was close but not bit-exact:
  - `final_z` max/mean/p95 absolute error:
    `7.09e-06 / 5.72e-07 / 1.46e-06`.
  - `candidate_k_entry` max/mean/p95 absolute error:
    `1.73e-05 / 7.06e-07 / 2.03e-06`.
  - `candidate_v_entry` max/mean/p95 absolute error:
    `1.03e-05 / 2.85e-07 / 1.07e-06`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_add_sln_skip_sln_rank3_trial`.
- The graph-capture benchmark failed on all wrapper attempts with:
  `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`.

Conclusion:
- Reject. Keeping the fused skip normalization rank-3 did not avoid the ORT WebGPU graph-capture
  output-binding failure seen with the rank-2 variant.
- Restored the manifest to the accepted full-cache dynamics artifact and removed the temporary
  trial artifact.

Restored accepted baseline after this rejected trial:
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`:
  passed.
  - Streaming after graph-capture warmup: `0.77 / 21.93 / 22.24 ms`, or `52.23 fps`.
  - Dynamics after graph-capture warmup: `0.66 / 17.92 / 18.48 ms`.
  - Decoder after graph-capture warmup: `0.10 / 3.14 / 3.75 ms`.

### Rejected: ONNX Runtime Web 1.25.1

Trial:
- Checked the registry after `onnxruntime-web@1.25.0` failed to resolve; there is no stable
  `1.25.0`, but `1.25.1` is published.
- Temporarily installed `onnxruntime-web@1.25.1` with Bun's `--cache-dir` flag. The package files
  were restored to the accepted `^1.24.3` range after the trial.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- The benchmark failed on all wrapper attempts with:
  `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`.

Conclusion:
- Reject. `1.25.1` regresses the accepted graph-capture path before timing can run, matching the
  metadata binding failure seen in the rejected `1.26.0` runtime trial.
- Restored `onnxruntime-web@1.24.3`, `package.json`, and `bun.lock`.

Restored accepted baseline after this runtime trial:
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`:
  passed.
  - Streaming after graph-capture warmup: `0.77 / 21.95 / 22.42 ms`, or `52.24 fps`.
  - Dynamics after graph-capture warmup: `0.65 / 18.05 / 18.55 ms`.
  - Decoder after graph-capture warmup: `0.09 / 3.13 / 3.39 ms`.

### Rejected: ONNX Runtime Web 1.24.2

Trial:
- Temporarily installed `onnxruntime-web@1.24.2` with Bun's `--cache-dir` flag to check whether the
  previous 1.24 patch release was faster than the accepted `1.24.3`.
- Model artifacts, graph-capture settings, cache semantics, and dynamics flow-step count were
  unchanged.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- The graph-capture benchmark passed.
- Streaming after graph-capture warmup: `0.75 / 21.93 / 22.36 ms`, or `51.87 fps`.
- Dynamics after graph-capture warmup: `0.64 / 17.93 / 18.60 ms`.
- Decoder after graph-capture warmup: `0.10 / 3.12 / 3.73 ms`.

Conclusion:
- Reject. `1.24.2` is graph-capture compatible, but it does not beat the accepted `1.24.3` baseline
  and has worse mean FPS in this run.
- Restored `onnxruntime-web@1.24.3`, `package.json`, and `bun.lock`.

Restored accepted baseline after the 1.24.2 runtime trial:
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`:
  passed.
  - Streaming after graph-capture warmup: `0.72 / 21.71 / 22.63 ms`, or `52.40 fps`.
  - Dynamics after graph-capture warmup: `0.61 / 18.02 / 18.64 ms`.
  - Decoder after graph-capture warmup: `0.08 / 3.12 / 3.78 ms`.

### Rejected: Cache Updater Workgroup Size Changes

Trial:
- Changed the WebGPU entry-cache updater compute workgroup size from `64` to `128`, then to `256`,
  in both the benchmark and live demo bridge.
- This only changed cache-maintenance dispatch shape. Model graphs, model math, cache contents,
  dynamics flow-step count, and exported artifacts were unchanged.

Same-session baseline before the trial:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `19.13 / 21.82 / 22.23 ms`, or `52.28 fps`.
- Dynamics after graph-capture warmup: `15.79 / 17.89 / 18.50 ms`.
- Decoder after graph-capture warmup: `2.82 / 3.12 / 3.74 ms`.
- Cache commit segment: `0.108 / 0.135 / 0.160 ms`.

Results:
- Workgroup size `128`:
  - Streaming after graph-capture warmup: `19.29 / 21.82 / 22.24 ms`, or `51.84 fps`.
  - Dynamics after graph-capture warmup: `15.87 / 17.90 / 18.47 ms`.
  - Decoder after graph-capture warmup: `2.89 / 3.12 / 3.79 ms`.
  - Cache commit segment: `0.117 / 0.135 / 0.174 ms`.
- Workgroup size `256`:
  - Streaming after graph-capture warmup: `19.38 / 22.03 / 22.34 ms`, or `51.60 fps`.
  - Dynamics after graph-capture warmup: `15.93 / 17.86 / 18.48 ms`.
  - Decoder after graph-capture warmup: `2.90 / 3.12 / 3.76 ms`.
  - Cache commit segment: `0.114 / 0.135 / 0.164 ms`.

Conclusion:
- Reject. Larger cache-updater workgroups did not reduce the frame path and slightly worsened the
  full-frame mean FPS.
- Restored the accepted `64` workgroup size in both bridge implementations.

### Rejected: MLP SkipSimplifiedLayerNormalization Rank-2 Branch

Trial:
- Built a temporary dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_skip_sln_rank2_trial.onnx`.
- Rewrote eligible MLP pre-norm `SkipSimplifiedLayerNormalization` islands to run rank-2 directly:
  - bypassed the skip-input `Unsqueeze`,
  - bypassed the normalized-output `Squeeze`,
  - bypassed the residual-output `Squeeze`,
  - left the shared main-branch `Unsqueeze` in place when the QKV branch still consumed it.
- Rewrites: `68` rank-2 `SkipSimplifiedLayerNormalization` branches.
- Node count: `2782 -> 2578`.
- Op deltas:
  - `Unsqueeze`: `532 -> 464`.
  - `Squeeze`: `317 -> 181`.
  - Other model op counts unchanged.
- CPU validation against the accepted dynamics artifact was exact at `atol=0` for
  `candidate_k_entry`, `candidate_v_entry`, and `final_z`.

Browser graph-capture results:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_skip_sln_rank2_trial`.
- First run:
  - Streaming after graph-capture warmup: `19.13 / 21.77 / 22.24 ms`, or `52.28 fps`.
  - Dynamics after graph-capture warmup: `15.77 / 17.90 / 18.53 ms`.
  - Decoder after graph-capture warmup: `2.84 / 3.13 / 3.79 ms`.
- Repeat:
  - Streaming after graph-capture warmup: `19.09 / 21.92 / 22.20 ms`, or `52.38 fps`.
  - Dynamics after graph-capture warmup: `15.73 / 18.00 / 18.50 ms`.
  - Decoder after graph-capture warmup: `2.85 / 3.12 / 3.78 ms`.

Conclusion:
- Reject. The graph is substantially smaller and exact, but the browser path is neutral and the
  dynamics median does not improve over the same-session accepted baseline.
- Removed the temporary trial artifact and manifest entry.

### Rejected: ORT Parallel Execution Mode

Trial:
- Set the benchmark WebGPU session option `executionMode: 'parallel'` for the prefill, dynamics,
  and decoder sessions.
- Model artifacts, graph capture policy, model math, cache update semantics, and dynamics flow-step
  count were unchanged.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `19.15 / 21.93 / 22.30 ms`, or `52.21 fps`.
- Dynamics after graph-capture warmup: `15.80 / 17.88 / 18.55 ms`.
- Decoder after graph-capture warmup: `2.84 / 3.13 / 3.76 ms`.

Conclusion:
- Reject. Parallel execution mode did not improve WebGPU graph-capture throughput and slightly
  worsened the full-frame median/p95.
- Restored the default sequential execution mode.

### Rejected: Direct-Z Decoder Preference

Trial:
- Temporarily changed `demo_generation.preferred_decoder_export` from
  `breakout_tokenizer_decoder_b1_t1` to `breakout_tokenizer_decode_z_b1_t1`.
- Motivation: the direct-`z` decoder input shape matches dynamics `final_z`, so the bridge could
  potentially avoid the `final_z` to latent decoder-input copy/reinterpretation path.
- Model math and dynamics artifact were unchanged.

Result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- The graph-capture benchmark failed consistently across all wrapper retries with:
  `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`.

Conclusion:
- Reject. The direct-`z` decoder is not usable in the current graph-capture/preallocated-output
  benchmark path.
- Restored `demo_generation.preferred_decoder_export` to `breakout_tokenizer_decoder_b1_t1`.

### Rejected: Composite Dynamics Plus Direct-Z Decoder

Trial:
- Built a temporary full-cache step graph that appended the direct-`z` decoder to the accepted
  packed dynamics artifact, so one graph-captured session would output `candidate_k_entry`,
  `candidate_v_entry`, `final_z`, and `patches`.
- Temporary artifact:
  `breakout_dynamics_decode_frame_full_cache_entry_packed_b1_t1_s2_trial.onnx`.
- SHA-256:
  `7b136417ac543467e7f444f1bd9b7b3d82ece616a25699a98fde1dec4617aa87`.
- CPU validation against the separate accepted dynamics artifact plus the direct-`z` decoder was
  exact at `atol=0` for `candidate_k_entry`, `candidate_v_entry`, `final_z`, and `patches`.

Result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_decode_frame_full_cache_entry_packed_b1_t1_s2_trial`.
- The graph-capture benchmark failed consistently across all wrapper retries with:
  `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`.

Conclusion:
- Reject. Folding the direct-`z` decoder into the step graph hits the same ORT WebGPU metadata
  failure as using the direct-`z` decoder as the preferred decoder export.
- Removed the temporary composite artifact, its manifest entry, and the temporary benchmark
  fused-`patches` output path.

### Rejected: Direct-Z Decoder Without Decoder Graph Capture

Trial:
- Temporarily changed `demo_generation.preferred_decoder_export` from
  `breakout_tokenizer_decoder_b1_t1` to `breakout_tokenizer_decode_z_b1_t1`.
- Disabled only decoder graph capture with `--webgpu-benchmark-decoder-graph-capture false`.
- Dynamics graph capture, model math, cache update semantics, and dynamics flow-step count were
  unchanged.

Browser graph-capture results:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-decoder-graph-capture false`.
- Streaming after graph-capture warmup: `18.84 / 21.59 / 22.80 ms`, or `53.09 fps`.
- Dynamics after graph-capture warmup: `14.82 / 17.56 / 18.38 ms`.
- Decoder after graph-capture warmup: `3.96 / 3.96 / 4.59 ms`.
- Result copied to `/private/tmp/graph_capture_direct_z_decoder_gc_false.json`.

Conclusion:
- Reject. The trial avoids the direct-`z` graph-capture metadata failure, but it makes the decoder
  materially slower and worsens the full-frame p95. The slightly better reported mean is not a
  consistent 60 fps path and appears dominated by the same graph-capture warmup-sample artifact in
  the benchmark summary.
- Restored `demo_generation.preferred_decoder_export` to `breakout_tokenizer_decoder_b1_t1`.

### Rejected: Direct-Z Decoder Graph Capture Without Preallocated Output Fetch

Trial:
- Temporarily changed `demo_generation.preferred_decoder_export` from
  `breakout_tokenizer_decoder_b1_t1` to `breakout_tokenizer_decode_z_b1_t1`.
- Kept decoder graph capture enabled, but temporarily skipped the benchmark's preallocated decoder
  output fetch tensor for z-input decoders.
- Motivation: isolate whether the direct-`z` graph-capture metadata failure was caused by the
  preallocated decoder output fetch path.

Result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 3`.
- The graph-capture benchmark still failed consistently across all wrapper retries with:
  `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`.

Conclusion:
- Reject. The direct-`z` graph-capture failure is not fixed by avoiding preallocated decoder output
  fetches.
- Restored the benchmark fetch policy and `demo_generation.preferred_decoder_export`.

### Rejected: Same-Byte Final-Z Output Buffer Reinterpretation

Trial:
- Temporarily relaxed the benchmark's graph-captured final-z output buffer reuse check from exact
  shape equality to equal dtype plus equal element count.
- Motivation: the accepted dynamics `final_z` output has shape `[1,1,32,32]`, while the accepted
  decoder `latent` input has shape `[1,1,64,16]`; the byte count is identical, so in principle this
  could remove the small GPU copy/reinterpret bridge before the decoder.

Result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- The graph-capture benchmark failed consistently across all wrapper retries with:
  `Got invalid dimensions for output: final_z ... Got: 64 Expected: 32 ... Got: 16 Expected: 32`.

Conclusion:
- Reject. ORT validates preallocated output tensor dimensions, so the step session cannot write
  `[1,1,32,32]` output metadata into the `[1,1,64,16]` decoder input tensor even when the buffer
  size matches.
- Restored the strict same-shape guard in the benchmark.

### Accepted: Throttle Live Demo Stats DOM Writes

Change:
- Throttled the live demo's visible `frame-count`, `latency`, and `fps` text updates to the first
  generated frame and then every `250 ms`.
- The model inference loop, generated frame count, cache update semantics, rendering path, ONNX
  artifacts, and dynamics flow-step count are unchanged.
- This only removes avoidable per-frame DOM text writes from max-speed live playback; it is not
  expected to change the model-only benchmark.

Validation:
- `bun run build:webgpu:browser`: passed.
- `bun run typecheck`: passed.
- `bun run demo:webgpu:smoke -- --playwright-channel chrome --playwright-benchmark-attempts 3`:
  passed, 6 tests.

Conclusion:
- Accept as a narrow live-bridge cleanup. It should make the actual demo loop do less browser UI
  work during max-speed playback, while preserving the inference and rendering behavior.

### Accepted: Pairwise Normal Noise Fill

Change:
- Optimized `NormalNoiseGenerator.fillTensorData()` to fill `Float32Array` values two at a time
  from each Box-Muller transform instead of calling `normal()` for every element.
- The generated noise sequence is unchanged; the new path emits the same `cos` then `sin` pair that
  repeated `normal()` calls previously returned through the `spare` value.
- ONNX artifacts, model math, cache update semantics, and dynamics flow-step count are unchanged.

Validation:
- A direct Bun comparison against the previous scalar fill algorithm passed for odd, even, and
  frame-sized tensors, including the case where a `spare` value is already pending.
- `bun run typecheck`: passed.
- `bun run build:webgpu:browser`: passed.
- `bun run demo:webgpu:smoke -- --playwright-channel chrome --playwright-benchmark-attempts 3`:
  passed, 6 tests.

Conclusion:
- Accept as a live JS bridge cleanup. It reduces per-frame main-thread overhead for the two
  generated noise tensors without changing the stochastic rollout stream.

### Accepted: Prefill Next Live Noise Slot

Change:
- Double-buffered the live demo's CPU `sample_noise` and `context_noise` tensors.
- The frame loop now uses the ready slot for the current ORT run, starts the dynamics run, then
  fills the alternate slot for the next generated frame while the current GPU step is already in
  flight.
- The graph-capture action input buffer is now rewritten only when the selected action changes.
- ONNX artifacts, model math, cache update semantics, rendered output path, and dynamics flow-step
  count are unchanged.

Validation:
- A direct Bun comparison verified that the double-buffered prefill emits the same per-frame
  `sample_noise` then `context_noise` sequence as the previous scalar frame order.
- `bun run typecheck`: passed.
- `bun run build:webgpu:browser`: passed.
- `bun run demo:webgpu:smoke -- --playwright-channel chrome --playwright-benchmark-attempts 3`:
  passed, 6 tests.

Conclusion:
- Accept as a live-demo bridge cleanup. It moves deterministic CPU noise generation off the
  immediate pre-dispatch path for the next frame and removes redundant graph-capture action
  `writeBuffer` calls when input is unchanged.
- It is not expected to change the model-only benchmark because that benchmark does not use the
  live demo noise generator or action-upload path.

### Rejected: Fold Gather Plus Unsqueeze Axis 1

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_gather_unsqueeze_axis1_trial.onnx`.
- Replaced `35` `Gather -> Unsqueeze(axis=1)` islands with a single `Gather` using an expanded
  rank-3 constant index tensor.
- Node count changed from `2782` to `2747`; `Unsqueeze` changed from `532` to `497`.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact at `atol=0` for
  `final_z`, `candidate_k_entry`, and `candidate_v_entry`.

Browser graph-capture results:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_gather_unsqueeze_axis1_trial`.
- First run:
  - Streaming after graph-capture warmup: `19.10 / 21.91 / 22.37 ms`, or `52.35 fps`.
  - Dynamics after graph-capture warmup: `15.73 / 17.86 / 18.60 ms`.
  - Decoder after graph-capture warmup: `2.85 / 3.13 / 3.75 ms`.
- Repeat:
  - Streaming after graph-capture warmup: `18.98 / 21.83 / 22.23 ms`, or `52.69 fps`.
  - Dynamics after graph-capture warmup: `15.68 / 17.90 / 18.53 ms`.
  - Decoder after graph-capture warmup: `2.80 / 3.13 / 3.71 ms`.

Same-state accepted comparison:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `18.99 / 21.80 / 22.20 ms`, or `52.67 fps`.
- Dynamics after graph-capture warmup: `15.62 / 17.87 / 18.47 ms`.
- Decoder after graph-capture warmup: `2.85 / 3.14 / 3.80 ms`.

Conclusion:
- Reject. The rewrite is exact and smaller, but the browser timing is neutral; the same-state
  accepted artifact matched or slightly beat the trial median/p95.
- Removed the temporary trial artifact and manifest entry.

### Rejected: Fold All Head Gather Plus Unsqueeze Islands

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_gather_unsqueeze_all_trial.onnx`.
- Folded both head-layout cases:
  - `35` `Gather -> Unsqueeze(axis=1)` islands into a `Gather` with rank-3 constant indices.
  - `36` `Gather -> Unsqueeze(axis=0) -> Transpose(0,2,1,3)` islands into a `Gather` with rank-3
    constant indices plus `Transpose(1,2,0,3)`.
- Node count changed from `2782` to `2711`; `Unsqueeze` changed from `532` to `461`.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact at `atol=0` for
  `final_z`, `candidate_k_entry`, and `candidate_v_entry`.

Browser graph-capture results:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_gather_unsqueeze_all_trial`.
- First run:
  - Streaming after graph-capture warmup: `19.05 / 21.83 / 22.18 ms`, or `52.51 fps`.
  - Dynamics after graph-capture warmup: `15.71 / 17.85 / 18.42 ms`.
  - Decoder after graph-capture warmup: `2.82 / 3.13 / 3.75 ms`.
- Repeat:
  - Streaming after graph-capture warmup: `19.25 / 21.90 / 22.49 ms`, or `51.94 fps`.
  - Dynamics after graph-capture warmup: `15.89 / 18.02 / 18.56 ms`.
  - Decoder after graph-capture warmup: `2.82 / 3.14 / 3.78 ms`.

Conclusion:
- Reject. Removing all 71 head-gather `Unsqueeze` nodes is exact, but the browser timing is
  inconsistent and regresses on repeat.
- Removed the temporary trial artifact and manifest entry.

### Rejected: Flatten SimplifiedLayerNormalization Scale Initializers

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_sln_scale1d_trial.onnx`.
- Flattened `215` `SimplifiedLayerNormalization` scale initializers from singleton-broadcast
  shapes such as `[1, 1, 1, 32]` and `[1, 1, 128]` to the normalized 1D shape.
- Node count and graph topology were unchanged.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact at `atol=0` for
  `final_z`, `candidate_k_entry`, and `candidate_v_entry`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_sln_scale1d_trial`.
- Streaming after graph-capture warmup: `19.08 / 21.83 / 22.22 ms`, or `52.42 fps`.
- Dynamics after graph-capture warmup: `15.71 / 17.89 / 18.50 ms`.
- Decoder after graph-capture warmup: `2.85 / 3.12 / 3.73 ms`.
- Result copied to `/private/tmp/graph_capture_sln_scale1d_trial.json`.

Accepted artifact comparison:
- A same-state accepted rerun after removing the trial measured streaming after graph-capture warmup
  at `19.16 / 21.98 / 22.27 ms`, dynamics at `15.79 / 17.86 / 18.56 ms`, and decoder at
  `2.87 / 3.12 / 3.78 ms`.
- The previously maintained accepted run in this batch measured streaming at
  `19.01 / 21.80 / 22.19 ms`.

Conclusion:
- Reject. Flattening the SLN scale tensors is exact, but it does not clearly improve the accepted
  graph-capture window; the small timing differences are within the observed run-to-run variance.
- Removed the temporary trial artifact and manifest entry.

### Rejected: GatherND Cache Layer Slice

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_gathernd_cache_slice_trial.onnx`.
- Replaced the `24` cache layer-read `Gather(layer) -> Squeeze(batch)` pairs with
  `GatherND([layer, 0])`, directly emitting the `[36, 64, 2, 32]` cache slice.
- Node count changed from `2782` to `2758`; `Gather` changed from `238` to `214`,
  `Squeeze` from `317` to `293`, and `GatherND` from `0` to `24`.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact at `atol=0` for
  `final_z`, `candidate_k_entry`, and `candidate_v_entry`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_gathernd_cache_slice_trial`.
- The graph-capture benchmark failed consistently across all wrapper retries with:
  `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`.
- Result copied to `/private/tmp/graph_capture_gathernd_cache_slice_trial_failed.json`.

Conclusion:
- Reject. The rewrite is exact on CPU, but `GatherND` is not graph-capture safe in the current ORT
  WebGPU path.
- Removed the temporary trial artifact and manifest entry.

### Rejected: Latent-Shaped Final-Z Output Reshape

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_finalz_latent_trial.onnx`.
- Kept internal `final_z` consumers on the original `[1,1,32,32]` tensor, but added one output
  `Reshape` so the exported `final_z` graph output had the decoder latent shape `[1,1,64,16]`.
- Motivation: unlike the rejected same-byte output-buffer reinterpretation trial, this changes the
  graph output metadata itself so ORT's exact-shape preallocated output check could reuse the
  decoder's fixed latent input buffer.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact at `atol=0` for
  `candidate_k_entry` and `candidate_v_entry`.
- The trial `final_z` output was exact at `atol=0` after reshaping the accepted `[1,1,32,32]`
  output to `[1,1,64,16]`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_finalz_latent_trial`.
- After temporarily rebuilding the benchmark validator to allow latent-shaped `final_z`, ORT blocked
  graph capture session creation with:
  `all compute graph nodes have not been partitioned to the WebGpuExecutionProvider`.
- Result copied to `/private/tmp/graph_capture_finalz_latent_output_trial_blocked.json`.

Conclusion:
- Reject. The added `Reshape` is CPU-exact but reintroduces a non-WebGPU node, so it is not viable
  for the graph-captured path.
- Removed the temporary trial artifact and manifest entry, then restored the strict benchmark
  validation that requires dynamics `final_z` to remain `[1,1,32,32]`.

### Rejected: Alias Final-Z GPU Buffer as Decoder Latent

Trial:
- Temporarily changed the benchmark bridge so, when decoder graph capture was disabled, a GPU
  `final_z` output with shape `[1,1,32,32]` could be re-exposed as a no-op-dispose
  `[1,1,64,16]` `ort.Tensor` over the same `GPUBuffer`.
- Motivation: avoid the per-frame GPU copy into the fixed latent decoder input without changing
  ONNX artifacts, model math, cache update semantics, rendered output path, or dynamics flow-step
  count.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 3 --webgpu-benchmark-decoder-graph-capture false`.
- Streaming after graph-capture warmup: `3.04 / 21.68 / 22.53 ms`, or `53.08 fps`.
- Dynamics after graph-capture warmup: `0.67 / 17.85 / 18.50 ms`.
- Decoder after graph-capture warmup: `2.34 / 3.24 / 3.94 ms`.

Same-code accepted comparison:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 3`.
- Streaming after graph-capture warmup: `0.75 / 21.88 / 22.24 ms`, or `52.28 fps`.
- Dynamics after graph-capture warmup: `0.63 / 17.85 / 18.54 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.12 / 3.61 ms`.

Conclusion:
- Reject. The alias path is bridge-clean and numerically neutral, but it requires disabling decoder
  graph capture and regresses decoder median/p95. The small streaming-median difference is within
  observed run-to-run variance and does not move the demo toward consistent 60 fps.
- Removed the temporary benchmark code and restored the accepted graph-capture result file.

### Rejected: Action Embedding Unsqueeze CSE

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_action_unsqueeze_cse_trial.onnx`.
- Rewired the three identical dynamic action-embedding `Unsqueeze(embed_out_0, axis=0)` nodes to
  share one output, removing two duplicate `Unsqueeze` nodes.
- Node count changed from `2782` to `2780`. ONNX artifacts outside the trial, cache update
  semantics, rendered output path, and dynamics flow-step count were unchanged.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact at `atol=0` for
  `final_z`, `candidate_k_entry`, and `candidate_v_entry`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_action_unsqueeze_cse_trial`.
- Streaming after graph-capture warmup: `0.74 / 21.78 / 22.31 ms`, or `52.38 fps`.
- Dynamics after graph-capture warmup: `0.62 / 17.98 / 18.55 ms`.
- Decoder after graph-capture warmup: `0.08 / 3.12 / 3.77 ms`.

Same-state accepted comparison:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.72 / 21.84 / 22.20 ms`, or `51.87 fps`.
- Dynamics after graph-capture warmup: `0.62 / 17.89 / 18.52 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.13 / 3.64 ms`.

Conclusion:
- Reject. The rewrite is exact, but removing two tiny layout nodes is within browser run-to-run
  variance and slightly worsens the measured dynamics median/p95.
- Removed the temporary trial artifact and manifest entry.

### Rejected: Graph Optimization Disabled

Trial:
- Ran the accepted full-cache dynamics and decoder artifacts with ORT `graphOptimizationLevel`
  set to `disabled`, leaving graph capture and all ONNX artifacts unchanged.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-graph-optimization-level disabled`.
- Streaming after graph-capture warmup: `0.72 / 21.90 / 22.23 ms`, or `52.27 fps`.
- Dynamics after graph-capture warmup: `0.58 / 17.89 / 18.53 ms`.
- Decoder after graph-capture warmup: `0.08 / 3.13 / 3.76 ms`.

Conclusion:
- Reject. Disabling ORT graph optimizations is neutral to slightly worse in the graph-captured
  browser path and does not move the steady streaming median toward the 16.67 ms target.
- Restored the default `basic` graph optimization setting.

### Rejected: Transposed Gemm Weight Layout

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_gemm_transb_trial.onnx`.
- Rewrote all `291` `Gemm` nodes by transposing constant B weight initializers and setting
  `transB=1`.
- Node count and graph topology were otherwise unchanged.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact at `atol=0` for
  `final_z`, `candidate_k_entry`, and `candidate_v_entry`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_gemm_transb_trial`.
- Streaming after graph-capture warmup: `0.76 / 22.11 / 22.52 ms`, or `51.84 fps`.
- Dynamics after graph-capture warmup: `0.64 / 18.19 / 18.72 ms`.
- Decoder after graph-capture warmup: `0.08 / 3.14 / 3.54 ms`.

Conclusion:
- Reject. The weight-layout rewrite is mathematically exact, but ORT WebGPU is slower with
  `transB=1` weights in this graph.
- Removed the temporary trial artifact and manifest entry.

### Rejected: Gemm C-Input Residual Add Fold

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_gemm_addc_trial.onnx`.
- Rewrote `71` exact `Gemm -> Add(residual)` islands into `Gemm(A, B, residual)` with
  `beta=1`, removing the separate residual `Add` nodes.
- Node count changed from `2782` to `2711`; `Add` changed from `73` to `2`.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact at `atol=0` for
  `final_z`, `candidate_k_entry`, and `candidate_v_entry`.

Browser graph-capture results:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_gemm_addc_trial`.
- First run:
  - Streaming after graph-capture warmup: `0.66 / 21.96 / 22.28 ms`, or `52.73 fps`.
  - Dynamics after graph-capture warmup: `0.56 / 18.80 / 19.43 ms`.
  - Decoder after graph-capture warmup: `0.07 / 2.46 / 2.80 ms`.
- Repeat:
  - Streaming after graph-capture warmup: `0.70 / 21.86 / 22.18 ms`, or `52.87 fps`.
  - Dynamics after graph-capture warmup: `0.58 / 18.70 / 19.38 ms`.
  - Decoder after graph-capture warmup: `0.09 / 2.41 / 3.02 ms`.

Same-window accepted comparison:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.73 / 21.74 / 22.19 ms`, or `52.43 fps`.
- Dynamics after graph-capture warmup: `0.60 / 17.89 / 18.49 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.13 / 3.51 ms`.

Conclusion:
- Reject. Folding the residual add into `Gemm` is exact and removes `71` Add nodes, but ORT WebGPU's
  `Gemm` path is consistently slower when the C input is present. Full-frame timing stays in the
  same run-to-run noise band and does not beat the same-window accepted median.
- Removed the temporary trial artifact and manifest entry.

### Rejected: Add Plus Pre-Norm SkipSimplifiedLayerNormalization Fusion

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_add_sln_skip_sln_trial.onnx`.
- Replaced `68` MLP residual `Add -> Unsqueeze -> SimplifiedLayerNormalization -> Squeeze` pre-norm
  islands with rank-2 `SkipSimplifiedLayerNormalization`, preserving the rank-2 residual sum for
  the existing downstream skip path.
- Node count changed from `2782` to `2646`.
- Op deltas:
  - `Add`: `73 -> 5`.
  - `SimplifiedLayerNormalization`: `215 -> 147`.
  - `Squeeze`: `317 -> 249`.
  - `SkipSimplifiedLayerNormalization`: `71 -> 139`.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was close but not bit-exact:
  - `final_z` max/mean/p95 absolute error: `9.30e-06 / 8.10e-07 / 3.69e-06`.
  - `candidate_k_entry` max/mean/p95 absolute error: `1.34e-05 / 8.47e-07 / 2.62e-06`.
  - `candidate_v_entry` max/mean/p95 absolute error: `1.92e-05 / 3.72e-07 / 1.31e-06`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_add_sln_skip_sln_trial`.
- The graph-capture benchmark failed on all wrapper attempts with:
  `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`.

Conclusion:
- Reject. The rewrite is numerically close and substantially smaller, but the added
  rank-2 `SkipSimplifiedLayerNormalization` pattern is not compatible with the current ORT WebGPU
  graph-capture output-binding path.
- Removed the temporary trial artifact and manifest entry.

Restored accepted baseline after this rejected-trial batch:
- Dynamics artifact SHA-256:
  `0a85800175a0015f124e8c440da03dc45a99696964dc7912615cc10bda9c7290`.
- Decoder artifact SHA-256:
  `5285111dcd426121dfb31fe27d1eb958d2133d12dd9faeaf607ecc02008b472b`.
- Breakout manifest SHA-256:
  `5db08d518d5e26b81567d10ec77f8b50940f03a2257572ecba3f2ec95b08a957`.
- No served `*trial*` artifacts remain in `webgpu_app/dream_arcade_assets/breakout`.
- `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`:
  passed.
  - Streaming after graph-capture warmup: `0.75 / 21.84 / 22.23 ms`, or `52.51 fps`.
  - Dynamics after graph-capture warmup: `0.63 / 17.83 / 18.48 ms`.
  - Decoder after graph-capture warmup: `0.09 / 3.13 / 3.58 ms`.
- `bun run demo:webgpu:smoke -- --playwright-channel chrome --playwright-benchmark-attempts 3`:
  passed, 6 tests.

### Rejected: Direct Rank-2 Terminal Output Slices

Trial:
- Built a temporary full-cache dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_direct_rank2_output_slice_trial.onnx`.
- Replaced the two terminal
  `Unsqueeze([0,2]) -> Slice(starts [0,4,0,0], ends [1,36,1,128]) -> Squeeze([0,2])`
  output-head islands with direct rank-2 `Slice(starts [4,0], ends [36,128])` feeding the same
  output `Gemm`.
- Removed `4` layout nodes. ONNX artifacts outside the trial, cache update semantics, rendered
  output path, and dynamics flow-step count were unchanged.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact at `atol=0` for
  seeds `1234` and `5678`:
  - `final_z`: max/mean/p95 absolute error `0 / 0 / 0`.
  - `candidate_k_entry`: max/mean/p95 absolute error `0 / 0 / 0`.
  - `candidate_v_entry`: max/mean/p95 absolute error `0 / 0 / 0`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_direct_rank2_output_slice_trial`.
- Streaming after graph-capture warmup: `0.75 / 21.82 / 22.30 ms`, or `51.75 fps`.
- Dynamics after graph-capture warmup: median `17.87 ms`, p95 `18.68 ms`.
- Decoder after graph-capture warmup: unchanged within measurement noise.

Restored accepted comparison:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: `0.74 / 21.93 / 22.25 ms`, or `52.46 fps`.
- Dynamics after graph-capture warmup: `0.63 / 18.00 / 18.53 ms`.
- Decoder after graph-capture warmup: `0.09 / 3.12 / 3.76 ms`.

Conclusion:
- Reject. The rewrite is exact and slightly improves dynamics median in this run, but worsens
  dynamics p95 and does not improve full streaming FPS. It remains in the browser run-to-run noise
  band and does not move the demo toward consistent 60 fps.
- Removed the temporary trial artifact, removed its manifest entry, and restored
  `graph_capture_latest.json` to the accepted artifact.

### Rejected: Queue Decoder Run Behind Dynamics Submission

Trial:
- Prototyped a benchmark-only bridge scheduling change for the graph-capture path.
- Because the step `final_z` output fetch is already aliased to the fixed decoder input tensor,
  the trial submitted `decoder.session.run(...)` immediately after calling `step.session.run(...)`
  instead of awaiting the dynamics promise first.
- This was intended to test whether ORT WebGPU exposed a costly await/synchronization boundary
  between the dynamics and decoder sessions.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Trial scheduling path:
  - Streaming after graph-capture warmup: median `21.76 ms`, p95 `22.13 ms`, or `52.68 fps`.
  - Dynamics after graph-capture warmup: median `17.76 ms`, p95 `18.40 ms`.
  - Decoder after graph-capture warmup: median `3.11 ms`, p95 `3.74 ms`.
- Same-window accepted rerun after reverting the benchmark-only patch:
  - Streaming after graph-capture warmup: median `21.88 ms`, p95 `22.45 ms`, or `52.22 fps`.
  - Dynamics after graph-capture warmup: median `17.86 ms`, p95 `18.61 ms`.
  - Decoder after graph-capture warmup: median `3.13 ms`, p95 `3.81 ms`.

Conclusion:
- Reject. The decoder timing stayed near the normal standalone decoder cost, so this did not expose
  a meaningful ORT await-boundary win. Full-frame timing remained in the accepted run-to-run noise
  band.
- Reverted the temporary benchmark scheduling patch and restored `graph_capture_latest.json` with
  an accepted-artifact rerun.

### Diagnostic Blocked: Non-Graph-Capture ORT WebGPU Profiling

Attempt:
- Retried a short non-graph-capture profiling run to collect ORT WebGPU kernel timings:
  `bun run benchmark:webgpu -- --grep "webgpu demo streaming benchmark$" --playwright-channel chrome --playwright-benchmark-attempts 3 --webgpu-benchmark-profiling 1 --webgpu-benchmark-profiling-drain-ms 1000 --webgpu-benchmark-profiling-top-k 40 --webgpu-benchmark-timed-runs 8`.
- Also tried bundled Chromium with one attempt:
  `bun run benchmark:webgpu -- --grep "webgpu demo streaming benchmark$" --playwright-channel bundled --playwright-benchmark-attempts 1 --webgpu-benchmark-profiling 1 --webgpu-benchmark-profiling-drain-ms 1000 --webgpu-benchmark-profiling-top-k 40 --webgpu-benchmark-timed-runs 4`.

Result:
- Both diagnostics failed before page load with Chrome crashpad bootstrap errors:
  `bootstrap_check_in ... Permission denied (1100)` and `ReadExactly: expected 4, observed 0`.

Conclusion:
- No profiling data collected. Do not treat this as performance evidence; it only confirms that this
  profiling launch path is currently blocked in this environment.

### Rejected: ORT WebGPU Workgroup-Size Micro-Tuning

Trial:
- Temporarily pointed the benchmark harness at the readable ORT module
  `/node_modules/onnxruntime-web/dist/ort.all.mjs` and changed one kernel family at a time.
- Softmax local workgroup trials:
  - Default local override is `WG = 64`, with the existing row-special-case logic preserved.
  - Tried `WG = 32` and `WG = 128`.
- Einsum local workgroup trials:
  - Changed the WebGPU Einsum program from its default `WORKGROUP_SIZE`/`outputSize / 64`
    dispatch shape to local sizes `128` and `256`.
- RotaryEmbedding local workgroup trial:
  - Changed the WebGPU RotaryEmbedding program from `WORKGROUP_SIZE` to `128`.
- No ONNX artifact, cache ABI, dynamics flow-step count, or model numerics changed. These were
  runtime-only diagnostics and were not kept in the production bundle.

Browser graph-capture results:
- Softmax `WG = 32`:
  - Streaming after graph-capture warmup: median `21.86 ms`, p95 `22.33 ms`, or `52.42 fps`.
  - Dynamics after graph-capture warmup: median `17.93 ms`, p95 `18.52 ms`.
- Softmax `WG = 128`:
  - Streaming after graph-capture warmup: median `21.95 ms`, p95 `22.35 ms`, or `52.22 fps`.
  - Dynamics after graph-capture warmup: median `17.92 ms`, p95 `18.52 ms`.
- Einsum local size `128`:
  - Streaming after graph-capture warmup: median `21.77 ms`, p95 `22.19 ms`, or `52.08 fps`.
  - Dynamics after graph-capture warmup: median `17.86 ms`, p95 `18.44 ms`.
- Einsum local size `256`:
  - Streaming after graph-capture warmup: median `21.87 ms`, p95 `22.29 ms`, or `51.72 fps`.
  - Dynamics after graph-capture warmup: median `17.94 ms`, p95 `18.56 ms`.
- RotaryEmbedding local size `128`:
  - Streaming after graph-capture warmup: median `21.93 ms`, p95 `22.31 ms`, or `52.36 fps`.
  - Dynamics after graph-capture warmup: median `17.90 ms`, p95 `18.54 ms`.
- GemmShared tile size `12`:
  - Motivation: hot dynamics Gemms have `M=36`, so a 12-row tile avoids the default 16-row tile's
    padded row band.
  - Streaming after graph-capture warmup: median `21.76 ms`, p95 `22.18 ms`, or `52.47 fps`.
  - Dynamics after graph-capture warmup: median `17.82 ms`, p95 `18.46 ms`.
- GemmShared tile size `15`:
  - Motivation: keep the same three `M=36` row tiles as the default while slightly reducing row
    padding.
  - Streaming after graph-capture warmup: median `21.82 ms`, p95 `22.30 ms`, or `52.29 fps`.
  - Dynamics after graph-capture warmup: median `17.85 ms`, p95 `18.48 ms`.

Restored accepted comparison:
- Restored the benchmark harness to
  `/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs`.
- Restored readable ORT defaults for Softmax, Einsum, and RotaryEmbedding.
- Restored readable ORT GemmShared default tile size `16`.
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: median `21.77 ms`, p95 `22.29 ms`, or `52.40 fps`.
- Dynamics after graph-capture warmup: median `17.88 ms`, p95 `18.54 ms`.
- Decoder after graph-capture warmup: median `3.12 ms`, p95 `3.78 ms`.

Conclusion:
- Reject. These runtime workgroup-size changes stayed inside normal browser run-to-run noise and did
  not move the full streaming frame time toward the `16.67 ms` target for consistent 60 fps.
- Keep the production benchmark import on the minified WebGPU bundle and keep readable ORT restored
  to defaults.

### Diagnostic Rejected: Skip Entry-Cache GPU Update

Trial:
- Temporarily changed the benchmark-only `updateCacheFromEntries()` path to dispose the dynamics
  `candidate_k_entry` and `candidate_v_entry` outputs but not dispatch the GPU cache slide/rebase
  shader.
- This intentionally invalidates rollout/cache semantics and was only used to test whether the
  cache-update shader was hidden in the next frame's dynamics await instead of the tiny
  `cache_commit` CPU timer.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 3`.
- Streaming after graph-capture warmup: median `21.82 ms`, p95 `22.27 ms`, or `52.13 fps`.
- Dynamics after graph-capture warmup: median `17.80 ms`, p95 `18.48 ms`.
- Decoder after graph-capture warmup: median `3.12 ms`, p95 `3.77 ms`.

Conclusion:
- Reject as an optimization and as a direction. Skipping the GPU cache update did not improve the
  frame path, so a correct cache-ring redesign is unlikely to recover the missing `~5 ms` by itself.
- Restored the valid cache update path and reran the accepted production-bundle graph-capture
  benchmark:
  - Streaming after graph-capture warmup: median `21.84 ms`, p95 `22.45 ms`, or `52.39 fps`.
  - Dynamics after graph-capture warmup: median `17.88 ms`, p95 `18.54 ms`.
  - Decoder after graph-capture warmup: median `3.12 ms`, p95 `3.52 ms`.

### Rejected: Reuse Benchmark Graph-Capture Step Feeds

Trial:
- Temporarily reused one graph-capture dynamics feed object in the benchmark instead of rebuilding
  it with object spreads every generated frame.
- Fixed GPU tensors, preallocated outputs, cache tensors, model artifacts, and dynamics flow-step
  count were unchanged.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: median `21.96 ms`, p95 `22.30 ms`, or `52.19 fps`.
- Dynamics after graph-capture warmup: median `17.97 ms`, p95 `18.59 ms`.
- Decoder after graph-capture warmup: median `3.13 ms`, p95 `3.29 ms`.

Conclusion:
- Reject. Reusing the feed object did not reduce full-frame timing and made the benchmark bridge
  more stateful.
- Reverted the benchmark-only patch.

### Rejected: Rectangular No-Bias GemmShared Shaders

Trial:
- Temporarily changed the readable ORT WebGPU Gemm kernel for no-bias, non-transposed `Gemm`
  nodes only, leaving biased Gemms on the default `GemmShared` path.
- Motivation: the hot dynamics projections mostly have `M=36`, so the default square `16x16`
  tile launches three row tiles and many column tiles. Rectangular tiling might reduce dispatch
  count for large `N` projections without changing ONNX graphs or arithmetic semantics.
- Runtime-only diagnostics:
  - `8x32`: one output per thread, workgroup size `32x8`.
  - `16x32`: two output columns per thread, workgroup size `16x16`.
- No ONNX artifact, cache ABI, dynamics flow-step count, or model numerics changed.

Browser graph-capture results:
- `8x32`:
  - Streaming after graph-capture warmup: median `21.97 ms`, p95 `22.34 ms`, or `52.28 fps`.
  - Dynamics after graph-capture warmup: median `17.95 ms`, p95 `18.67 ms`.
- `16x32`, first run:
  - Streaming after graph-capture warmup: median `21.89 ms`, p95 `22.06 ms`, or `52.45 fps`.
  - Dynamics after graph-capture warmup: median `17.74 ms`, p95 `18.41 ms`.
- `16x32`, repeat:
  - Streaming after graph-capture warmup: median `21.88 ms`, p95 `22.74 ms`, or `52.27 fps`.
  - Dynamics after graph-capture warmup: median `17.84 ms`, p95 `18.92 ms`.

Restored accepted comparison:
- Restored the benchmark harness to
  `/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs`.
- Restored readable ORT GemmShared to the default `16x16` shader.
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: median `21.94 ms`, p95 `22.34 ms`, or `51.57 fps`.
- Dynamics after graph-capture warmup: median `17.96 ms`, p95 `18.45 ms`.
- Decoder after graph-capture warmup: median `3.12 ms`, p95 `3.73 ms`.

Conclusion:
- Reject. The custom rectangular shaders were graph-capture compatible, but full-frame timing
  stayed inside the current browser noise band and the repeat run worsened p95.
- Do not carry a patched ORT Gemm shader without a much larger and repeatable win.

### Rejected: Gemm Residual Fold With Direct C-Input Shader

Trial:
- Built a temporary dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_gemm_addc_directc_trial.onnx`.
- Reused the exact `Gemm -> Add(residual)` fold from the earlier rejected C-input trial:
  `71` residual Adds became `Gemm(A, B, residual)` with `beta=1`.
- Added a temporary readable-ORT WebGPU shader specialization for the C-input case where the C
  tensor shape exactly matches the `Gemm` output shape, replacing the generic broadcasted C offset
  calculation with a direct `c[m * N + n]` read.
- This was a runtime-only specialization plus a temporary trial artifact. The dynamics flow-step
  count and cache ABI were unchanged.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact at `atol=0` for
  seeds `1234` and `5678`, covering `final_z`, `candidate_k_entry`, and `candidate_v_entry`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_gemm_addc_directc_trial`.
- Streaming after graph-capture warmup: median `21.79 ms`, p95 `22.32 ms`, or `53.01 fps`.
- Dynamics after graph-capture warmup: median `18.76 ms`, p95 `19.48 ms`.
- Decoder after graph-capture warmup: median `2.43 ms`, p95 `2.58 ms`.

Restored accepted comparison:
- Restored the manifest, removed the temporary trial artifact, restored readable ORT, and restored
  the benchmark harness to `/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs`.
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: median `21.97 ms`, p95 `22.52 ms`, or `51.71 fps`.
- Dynamics after graph-capture warmup: median `17.87 ms`, p95 `18.66 ms`.
- Decoder after graph-capture warmup: median `3.14 ms`, p95 `3.82 ms`.

Conclusion:
- Reject. Even with direct same-shape C reads, the folded C-input dynamics graph is slower than the
  accepted separate `Gemm + Add` path. The full-frame FPS increase came from decoder timing noise,
  not an improvement in the dynamics bottleneck.

### Rejected: Duplicate Initializer Deduplication

Trial:
- Built a temporary dynamics artifact
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_dedup_initializers_trial.onnx`.
- Rewired exact duplicate initializer tensors to a canonical initializer and removed the duplicate
  initializer entries.
- Removed `439` initializer entries and about `108,760` initializer bytes, mostly repeated split
  sizes, RoPE constants, and repeated norm scale constants across unrolled branches.
- Node topology, ONNX operators, cache ABI, dynamics flow-step count, and arithmetic were unchanged.

Validation:
- CPU comparison against the accepted full-cache dynamics artifact was exact at `atol=0` for
  seeds `1234` and `5678`, covering `final_z`, `candidate_k_entry`, and `candidate_v_entry`.

Browser graph-capture result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_dedup_initializers_trial`.
- Streaming after graph-capture warmup: median `21.80 ms`, p95 `22.26 ms`, or `52.44 fps`.
- Dynamics after graph-capture warmup: median `17.84 ms`, p95 `18.46 ms`.
- Decoder after graph-capture warmup: median `3.13 ms`, p95 `3.62 ms`.

Same-window accepted comparison:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: median `21.86 ms`, p95 `22.24 ms`, or `52.15 fps`.
- Dynamics after graph-capture warmup: median `17.89 ms`, p95 `18.51 ms`.
- Decoder after graph-capture warmup: median `3.13 ms`, p95 `3.77 ms`.

Conclusion:
- Reject for FPS. The cleanup is exact and slightly smaller on disk, but it does not change the
  runtime kernel sequence and the browser result is within the usual run-to-run noise band.
- Removed the trial artifact and restored the manifest.

### Rejected: ORT SkipSimplifiedLayerNormalization Workgroup Size

Trial:
- Temporarily switched the benchmark harness to readable ORT:
  `/node_modules/onnxruntime-web/dist/ort.all.mjs`.
- Changed the ORT WebGPU `SkipLayerNormalization` shader workgroup size from `64` to `32`, `16`,
  and `8`.
- Motivation: the active full-cache dynamics graph has `71` `SkipSimplifiedLayerNormalization`
  kernels with hidden size `128`. With ORT's 4-wide vectorization, hidden size `128` has only `32`
  vector lanes per row, so the default workgroup size `64` leaves most lanes idle before reduction.
- This was a runtime-only kernel experiment. The ONNX graph, cache ABI, dynamics flow-step count,
  and arithmetic were unchanged.

Browser graph-capture results:
- Workgroup size `32`, first run:
  - Streaming after graph-capture warmup: median `21.78 ms`, p95 `22.20 ms`, or `52.48 fps`.
  - Dynamics after graph-capture warmup: median `17.81 ms`, p95 `18.50 ms`.
- Workgroup size `16`:
  - Streaming after graph-capture warmup: median `21.94 ms`, p95 `22.27 ms`, or `52.33 fps`.
  - Dynamics after graph-capture warmup: median `17.85 ms`, p95 `18.47 ms`.
- Workgroup size `8`:
  - Streaming after graph-capture warmup: median `21.97 ms`, p95 `22.49 ms`, or `51.42 fps`.
  - Dynamics after graph-capture warmup: median `18.07 ms`, p95 `18.53 ms`.
- Workgroup size `32`, repeat:
  - Streaming after graph-capture warmup: median `21.87 ms`, p95 `22.24 ms`, or `52.34 fps`.
  - Dynamics after graph-capture warmup: median `17.83 ms`, p95 `18.52 ms`.

Restored accepted comparison:
- Restored readable ORT to the default `64`-thread skip-layer-norm workgroup and restored the
  benchmark harness to `/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs`.
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: median `22.06 ms`, p95 `22.23 ms`, or `52.36 fps`.
- Dynamics after graph-capture warmup: median `17.86 ms`, p95 `18.52 ms`.
- Decoder after graph-capture warmup: median `3.13 ms`, p95 `3.80 ms`.

Conclusion:
- Reject. The `32`-thread variant is graph-capture compatible and slightly improves dynamics median
  in one run, but the restored production bundle matched the FPS and p95 within noise.
- `16` and `8` are worse. Do not carry a patched ORT skip-layer-norm shader without a larger,
  repeatable full-frame win.

### Rejected: ORT Common Unary Workgroup Size

Trial:
- Temporarily switched the benchmark harness to readable ORT:
  `/node_modules/onnxruntime-web/dist/ort.all.mjs`.
- Changed ORT's common unary elementwise WebGPU shader workgroup size from the default `64` to
  `128` and `32`, updating the dispatch divisor to match.
- Motivation: the active dynamics graph still has `71` `QuickGelu` kernels on `(36,384)` tensors,
  and `QuickGelu` uses this common unary shader path.
- This was a runtime-only kernel experiment. The ONNX graph, cache ABI, dynamics flow-step count,
  and arithmetic were unchanged.

Browser graph-capture results:
- Workgroup size `128`:
  - Streaming after graph-capture warmup: median `21.79 ms`, p95 `22.22 ms`, or `52.41 fps`.
  - Dynamics after graph-capture warmup: median `17.88 ms`, p95 `18.49 ms`.
- Workgroup size `32`:
  - Streaming after graph-capture warmup: median `21.90 ms`, p95 `22.33 ms`, or `52.42 fps`.
  - Dynamics after graph-capture warmup: median `17.95 ms`, p95 `18.61 ms`.

Comparison:
- Latest restored production-bundle comparison from the skip-layer-norm workgroup trial:
  streaming after graph-capture warmup median `22.06 ms`, p95 `22.23 ms`, or `52.36 fps`, with
  dynamics median `17.86 ms`, p95 `18.52 ms`.

Conclusion:
- Reject. The common-unary workgroup size variants are graph-capture compatible but stay inside the
  current browser noise band and do not improve the dynamics p95.
- Restored readable ORT and the benchmark harness to the production bundle.

### Rejected: ORT Common Binary Workgroup Size

Trial:
- Temporarily switched the benchmark harness to readable ORT:
  `/node_modules/onnxruntime-web/dist/ort.all.mjs`.
- Changed ORT's common binary elementwise WebGPU shader workgroup size from the default `64` to
  `128` and `32`, updating the dispatch divisor to match.
- Motivation: the active dynamics graph still has high-count binary elementwise work
  (`Mul`/`Add` around the MLP and residual paths), and this shader is the common path for those ops.
- This was a runtime-only kernel experiment. The ONNX graph, cache ABI, dynamics flow-step count,
  and arithmetic were unchanged.

Browser graph-capture results:
- Workgroup size `128`:
  - Streaming after graph-capture warmup: median `21.93 ms`, p95 `22.22 ms`, or `52.15 fps`.
  - Dynamics after graph-capture warmup: median `17.91 ms`, p95 `18.48 ms`.
- Workgroup size `32`:
  - Streaming after graph-capture warmup: median `22.04 ms`, p95 `22.27 ms`, or `51.84 fps`.
  - Dynamics after graph-capture warmup: median `17.89 ms`, p95 `18.52 ms`.

Comparison:
- Latest restored production-bundle comparison from the skip-layer-norm workgroup trial:
  streaming after graph-capture warmup median `22.06 ms`, p95 `22.23 ms`, or `52.36 fps`, with
  dynamics median `17.86 ms`, p95 `18.52 ms`.

Conclusion:
- Reject. Binary elementwise workgroup changes are graph-capture compatible but do not improve the
  full-frame result; `32` is clearly worse and `128` is below the restored production comparison.
- Restored readable ORT and the benchmark harness to the production bundle.

### Rejected: ORT Concat and Gather Workgroup Sizes

Trial:
- Temporarily switched the benchmark harness to readable ORT:
  `/node_modules/onnxruntime-web/dist/ort.all.mjs`.
- Changed ORT's `Concat` WebGPU shader workgroup size from the default `64` to `128` and `32`.
- Restored `Concat`, then changed ORT's `Gather` WebGPU shader workgroup size from `64` to `128`
  and `32`.
- Motivation: the active dynamics graph has high-count layout kernels (`Concat 290`, `Gather 238`)
  after previous graph-level gather/concat rewrites had already been rejected.
- This was a runtime-only kernel experiment. The ONNX graph, cache ABI, dynamics flow-step count,
  and arithmetic were unchanged.

Browser graph-capture results:
- `Concat` workgroup size `128`:
  - Streaming after graph-capture warmup: median `21.98 ms`, p95 `22.23 ms`, or `52.37 fps`.
  - Dynamics after graph-capture warmup: median `17.88 ms`, p95 `18.53 ms`.
- `Concat` workgroup size `32`:
  - Streaming after graph-capture warmup: median `21.77 ms`, p95 `22.29 ms`, or `51.79 fps`.
  - Dynamics after graph-capture warmup: median `17.91 ms`, p95 `18.49 ms`.
- `Gather` workgroup size `128`:
  - Streaming after graph-capture warmup: median `22.00 ms`, p95 `22.26 ms`, or `52.26 fps`.
  - Dynamics after graph-capture warmup: median `17.85 ms`, p95 `18.54 ms`.
- `Gather` workgroup size `32`:
  - Streaming after graph-capture warmup: median `21.91 ms`, p95 `22.37 ms`, or `52.21 fps`.
  - Dynamics after graph-capture warmup: median `17.90 ms`, p95 `18.66 ms`.

Restored accepted comparison:
- Restored readable ORT and the benchmark harness to
  `/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs`.
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: median `21.81 ms`, p95 `22.40 ms`, or `51.97 fps`.
- Dynamics after graph-capture warmup: median `17.87 ms`, p95 `18.68 ms`.
- Decoder after graph-capture warmup: median `3.12 ms`, p95 `3.58 ms`.

Conclusion:
- Reject. The layout-kernel workgroup variants are graph-capture compatible but remain in the same
  full-frame noise band and do not produce a reliable dynamics p95 improvement.
- The latest production comparison is a lower same-machine run than the prior `52.36 fps`
  production run, reinforcing that these small workgroup movements are below the benchmark's
  run-to-run variance.

### Rejected: ORT Split Workgroup Size

Trial:
- Temporarily switched the benchmark harness to readable ORT:
  `/node_modules/onnxruntime-web/dist/ort.all.mjs`.
- Changed ORT's `Split` WebGPU shader workgroup size from the default `64` to `128` and `32`,
  updating the dispatch divisor to match.
- Motivation: the active dynamics graph has `215` `Split` kernels, mostly around attention and MLP
  layout paths.
- This was a runtime-only kernel experiment. The ONNX graph, cache ABI, dynamics flow-step count,
  and arithmetic were unchanged.

Browser graph-capture results:
- Workgroup size `128`:
  - Streaming after graph-capture warmup: median `22.01 ms`, p95 `22.32 ms`, or `52.26 fps`.
  - Dynamics after graph-capture warmup: median `17.84 ms`, p95 `18.57 ms`.
- Workgroup size `32`:
  - Streaming after graph-capture warmup: median `21.73 ms`, p95 `22.27 ms`, or `52.18 fps`.
  - Dynamics after graph-capture warmup: median `17.89 ms`, p95 `18.53 ms`.

Restored accepted comparison:
- Restored readable ORT and the benchmark harness to
  `/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs`.
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: median `21.73 ms`, p95 `22.22 ms`, or `52.23 fps`.
- Dynamics after graph-capture warmup: median `17.82 ms`, p95 `18.46 ms`.
- Decoder after graph-capture warmup: median `3.12 ms`, p95 `3.74 ms`.

Conclusion:
- Reject. Split workgroup changes are graph-capture compatible but do not improve full-frame FPS or
  dynamics p95 over the restored production bundle.

### Diagnostic Rejected: BiasSplitGelu as Fused QuickGelu Gate

Trial:
- Built a temporary dynamics artifact:
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_bias_split_quickgelu_trial.onnx`.
- Replaced each packed SwiGLU island
  `Split -> QuickGelu(alpha=1.0) -> Mul` with `com.microsoft::BiasSplitGelu` plus a zero bias.
- Direct rank-2 replacement was rejected by ORT model load before JS kernel selection:
  `[ShapeInferenceError] input shall be 3 dimensions`.
- The runnable dynamics trial inserted `Unsqueeze -> BiasSplitGelu -> Squeeze` around each fused
  gate, replacing `71` `Split`, `71` `QuickGelu`, and `71` `Mul` nodes with `71`
  `BiasSplitGelu` nodes plus `71` `Unsqueeze` and `71` `Squeeze` nodes.
- Temporarily patched readable ORT's WebGPU `BiasSplitGelu` shader to compute the exact
  `x * sigmoid(x)` gate used by the graph's `QuickGelu(alpha=1.0)` nodes instead of GELU, and
  temporarily switched the benchmark harness to `/node_modules/onnxruntime-web/dist/ort.all.mjs`.
- This was a diagnostic runtime semantic patch, not a valid standalone ONNX artifact under the
  stock ORT runtime.

Dynamics-only trial result:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5 --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_full_cache_entry_packed_bias_split_quickgelu_trial`.
- Streaming after graph-capture warmup: median `21.60 ms`, p95 `21.88 ms`, or `53.84 fps`.
- Dynamics after graph-capture warmup: median `16.89 ms`, p95 `17.53 ms`.
- Decoder after graph-capture warmup: median `3.81 ms`, p95 `4.40 ms`.

Same-window readable-ORT accepted comparison:
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: median `21.77 ms`, p95 `22.24 ms`, or `52.31 fps`.
- Dynamics after graph-capture warmup: median `18.09 ms`, p95 `18.69 ms`.
- Decoder after graph-capture warmup: median `3.10 ms`, p95 `3.44 ms`.

Decoder follow-up:
- Built a temporary decoder artifact:
  `breakout_tokenizer_decoder_b1_t1_bias_split_quickgelu_trial.onnx`.
- The decoder's packed SwiGLU halves are ordered opposite to the dynamics graph, so the trial
  swapped the local packed-Gemm output-weight halves before replacing the eight decoder gated MLP
  islands with the same patched `BiasSplitGelu` shader.
- Combined dynamics+decoder fused repeat:
  - Streaming after graph-capture warmup: median `22.00 ms`, p95 `22.89 ms`, or `52.68 fps`.
  - Dynamics after graph-capture warmup: median `18.82 ms`, p95 `19.60 ms`.
  - Decoder after graph-capture warmup: median `2.31 ms`, p95 `2.97 ms`.
- Decoder-only fused run with accepted dynamics:
  - Streaming after graph-capture warmup: median `21.80 ms`, p95 `23.00 ms`, or `51.31 fps`.
  - Dynamics after graph-capture warmup: median `19.27 ms`, p95 `20.34 ms`.
  - Decoder after graph-capture warmup: median `2.12 ms`, p95 `2.62 ms`.

Restored accepted comparison:
- Restored the manifest, removed both temporary trial artifacts, restored readable ORT's
  `BiasSplitGelu` shader, and restored the benchmark harness to
  `/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs`.
- Command:
  `bun run benchmark:webgpu -- --grep @graph-capture --playwright-channel chrome --playwright-benchmark-attempts 5`.
- Streaming after graph-capture warmup: median `22.25 ms`, p95 `23.08 ms`, or `50.92 fps`.
- Dynamics after graph-capture warmup: median `18.46 ms`, p95 `19.18 ms`.
- Decoder after graph-capture warmup: median `3.15 ms`, p95 `3.60 ms`.

Conclusion:
- Reject. The patched fused gate can improve isolated dynamics timing in one window, but it requires
  changing stock ORT `BiasSplitGelu` semantics and the full-frame result did not move toward
  consistent 60 fps once decoder/cache queue interactions were included.
- This remains a useful signal: a proper `BiasSplitQuickGelu`/SwiGLU fused WebGPU op with native
  rank-2 shape inference could be worth implementing, but the current stock-op substitution is not
  valid to ship and was not full-frame stable.

### Live Demo Cache-Fill Transition Fix

Issue:
- The live Breakout page decoded the prefix latent on reset whenever the WebGPU patch renderer was
  available, instead of showing the stored clean `display_pixels` preview.
- The page also enabled dynamics graph capture by default. Earlier full64 diagnostics showed that
  captured live dynamics could advance the frame counter while producing visually static output; the
  default four-frame startup only exposed that once `cache_length` reached `64`.

Change:
- The reset path now always renders the stored pixel preview when it is present.
- The WebGPU patch renderer reattaches its canvas if a later reset temporarily replaces it with a
  2D preview canvas.
- Live demo defaults now keep decoder graph capture and the cache-length dynamics capture setup, but
  leave the full-cache dynamics graph-capture session off unless `fullDynamicsGraphCapture=true` is
  explicitly supplied.
- Added a demo smoke regression that runs past frame `66` with full-cache dynamics graph capture
  disabled.

Reasoning:
- The benchmark's ~50 fps graph-capture number is a steady-state full-cache harness. The live page
  starts from a four-frame cache and spends roughly the first 60 generated frames on the partial
  cache-fill path, then switches to the full-cache path. Keeping full-cache dynamics capture off in
  the live page preserves transition correctness while the decoder can still use its fixed
  captured input/output path.

Follow-up correction:
- Tried to make the full-cache dynamics graph-capture path live by giving it fixed K/V cache buffers
  and making it the only ORT graph-capture session in that mode.
- That avoided the ORT `gpuBufferMetadata` crash only in a narrow smoke case, but manual inspection
  still showed the same broken frame after the cache-fill transition. A pixel-change assertion was too
  weak because the broken frame can still change slightly.
- Restored the shippable default to non-captured dynamics/decoder execution. Demo graph capture is
  now an unsafe diagnostic opt-in only, because captured dynamics can freeze the latent stream and
  captured decoder output can crash or fail validation depending on the bundle/browser.

### Safari Valid-Path Performance Check

Current validated Safari baseline:
- URL: `/webgpu_app/bench/index.html?browserProfile=safari&timedRuns=3&validationFrames=2`.
- Result: passed output validation, about `1.45 fps`.
- Mean frame time was about `691 ms`, with dynamics around `622 ms` and decoder around `62 ms`.
- Safari exposed `shader-f16` but not `subgroups`; Chrome on the same valid non-captured path exposed
  `subgroups` and ran about `35-36 fps`.
- ORT WebGPU profiling hooks emitted `0` events in both Safari and Chrome for this bundle, so the
  actionable timing split remains the benchmark's session-level dynamics/decoder measurements.

Rejected Safari controls:
- `graphOptimizationLevel=basic` was valid but much slower than `disabled` in Safari.
- `graphOptimizationLevel=extended` crashed the Safari automation session during the current matrix.
- The unpacked full-cache entry and cache-length entry step artifacts were valid but slower than the
  packed full-cache entry artifact.
- Disabling preallocated output tensors was noisy: a short two-frame run improved slightly, but a
  longer five-frame run was slower than the default preallocated path. Keep preallocation enabled.
- `navigator.ml` is not exposed in this Safari environment, so WebNN is not available as an alternate
  acceleration provider.
- The readable/all ORT bundles are not a Safari fix: `ort.webgpu.mjs` was valid but slower than the
  default WebGPU bundle, and `ort.all.mjs`/`ort.all.bundle.min.mjs` were extremely slow and failed
  generated-frame output validation.
- Pure WASM is valid but not competitive with Chrome. With the WASM bundle selected explicitly and
  `wasmNumThreads` varied on Safari:
  - `1` thread: about `1.71 fps`, `586 ms/frame`.
  - `2` threads: about `3.31 fps`, `302 ms/frame`.
  - `4` threads: about `3.71 fps`, `270 ms/frame`.
  - `8` threads: regressed to about `0.39 fps`, `2590 ms/frame`.
  This is faster than Safari's one-thread WASM and sometimes faster than Safari WebGPU, but still
  much slower than Chrome's validated WebGPU path.

Rejected mixed-FP16 artifact trial:
- Tried converting the packed full-cache dynamics step to keep public IO as fp32 while using fp16
  internally for Safari's `shader-f16` support.
- Blocking `Softmax`, layer norm, `QuickGelu`, `RotaryEmbedding`, and then `Einsum` still produced
  invalid fp16/fp32 type joins after ORT's converter rewrites.
- A narrower "Gemm-only fp16" conversion also produced contradictory Cast/value types around packed
  QKV/SwiGLU Gemm rewrites.
- A full-fp16 public-IO trial avoided the mixed-IO joins, but Safari WebGPU failed creating the
  fp16 `QuickGelu` pipeline. Keeping `QuickGelu` fp32 reintroduced mixed `Mul` input types. Replacing
  `QuickGelu` with primitive `Sigmoid`/`Mul` before full-fp16 conversion got past the immediate
  pipeline error, but the trial timed out after five minutes and was not a speed path.
- Conclusion: the current rewritten dynamics graph is not a usable target for ORT's generic fp16
  converter. Matching Chrome in Safari would require either a correct custom mixed-precision pass for
  this graph, fixing ORT WebGPU graph capture correctness for the full dynamics graph, or custom
  Safari-tuned WebGPU kernels for the hot dynamics blocks.

Graph-capture correctness probe:
- Built a temporary page that ran the packed full-cache dynamics step with several graph-capture
  input strategies.
- Normal non-captured WebGPU output changed across three different `sample_noise`/`context_noise`
  feeds.
- Replayed graph capture returned the first captured `final_z` hash for every later run, regardless
  of whether inputs were updated with `queue.writeBuffer`, copied through upload buffers, fenced with
  `queue.onSubmittedWorkDone()`, replaced with fresh GPU tensors, or changed for K/V/action inputs.
- Creating a new captured session for every frame produced changing hashes, but each frame paid the
  full first-capture cost and was slower than the valid non-captured Chrome path.
- Conclusion: the fast replayed graph-capture path is not a valid live or benchmark path for the
  current full dynamics graph. The failure is in ORT's captured full-graph replay observing only the
  initially captured inputs, not in the benchmark's input upload strategy.

Rejected Gemm-to-MatMul Safari trial:
- Built a temporary exact artifact replacing all `291` `Gemm` nodes in the packed full-cache
  dynamics step with `MatMul` plus bias `Add`.
- CPU validation against the accepted fp32 artifact was exact for `final_z`, `candidate_k_entry`,
  and `candidate_v_entry`.
- Safari benchmark with output validation passed, but regressed in the same run window:
  - Baseline packed `Gemm`: about `1.60 fps`, dynamics about `547 ms`.
  - Temporary `MatMul` artifact: about `1.31 fps`, dynamics about `669 ms`.
- Conclusion: Safari's ORT WebGPU `MatMul` kernels are not a replacement for the current `Gemm`
  kernels on this graph.
