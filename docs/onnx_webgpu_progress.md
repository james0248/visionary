# ONNX WebGPU Optimization Progress

## Goal

Make the Dreamer4 ONNX export fast enough for a live browser demo using ONNX Runtime WebGPU.

Demo benchmark contract:
- Start from an offline full 64-frame K/V cache artifact.
- Generate each new frame from the committed full cache.
- Decode only the newly predicted frame.
- Benchmark the real demo stream loop, not a parallel benchmark-only runtime.
- Validate visible generated frames with screenshot hashes and a loose Breakout brick-band coverage
  check, and validate WASM generated latents with finite changing `final_z` hashes before treating
  the FPS as valid.

## Current State

The branch now keeps the fp32 WebGPU path as the maintained demo/export target. The current demo
uses:
- `sample_steps=2`.
- Curated Breakout assets under `webgpu_app/dream_arcade_assets/breakout`.
- A separate WASM-targeted asset set under
  `webgpu_app/dream_arcade_assets/breakout_wasm_default_mha` when `backend=wasm`.
- Offline full-cache context/cache artifacts:
  `breakout_demo_context_noop60_fire4.*` and `breakout_demo_initial_cache_noop60_fire4.*`.
- A packed and partial-head-split-rewritten full-cache entry dynamics step graph for every generated
  frame:
  `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2.onnx`.
- On the WASM path, the entry-slide graph selected from the WASM asset set:
  `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s2.onnx`.
- On Chrome/Chromium WASM, the demo now tries a split-dynamics schedule when the derived
  `*_sample_only_final_z.onnx` and `*_context_entry_from_final_z.onnx` files are present. This
  starts the decoder worker after `final_z` is ready, then computes the context-cache entry on the
  main thread. Safari/WebKit keeps the unsplit graph by default because the split regressed its
  longer validated windows.
- A single-frame tokenizer decoder graph:
  `breakout_tokenizer_decoder_b1_t1.onnx`.

The maintained benchmark surface is latency plus graph capture:
- `bun run benchmark:webgpu` runs the actual demo streaming benchmark and graph-capture check.
- `bun run benchmark:webgpu:smoke` runs the smoke subset.
- Benchmark controls should be passed as wrapper flags after `--`, for example
  `--webgpu-benchmark-asset-base` or `--webgpu-benchmark-timed-runs`, instead of leading shell
  environment assignments.
- WASM split-dynamics experiments can be toggled with
  `--webgpu-benchmark-split-wasm-dynamics true|false`.
- `provider=wasm` benchmark defaults now explicitly use the WASM asset set and
  `ort.wasm.min.mjs`, then let the demo choose the browser-specific WASM thread count
  (`4` in Chrome, `3` in Safari/WebKit). The decoder worker pipeline defaults to
  `decoderWorkerNumThreads=3`.
- Generated results stay under `webgpu_app/bench/results/` and should not be committed.

Rejected or inactive paths:
- `--grouped_gqa_attention` validated in some forms but was slower or browser-incompatible.
- Native ONNX/ORT attention fusion did not produce a better WebGPU artifact for this model.
- fp16/bf16 and int quantization experiments either regressed speed, failed validation, or produced
  unstable outputs. The stable branch target is fp32.
- ORT WebGPU profiling callbacks/session profiling did not provide reliable actionable attribution,
  so the maintained workflow keeps timing and graph capture only.

### 2026-05-21 Actual Demo Benchmark Proxy

- Replaced the standalone `/bench/index.html` benchmark runtime with a Playwright benchmark that
  opens `/demo/index.html`, drives the visible Start/Pause stream loop, and records generated-frame
  timings from the demo debug API. This removes the previous proxy gap where benchmark FPS could
  disagree with the real demo.
- Added `visionaryDemoDebug.frameStats` from `recordGeneratedFrame()` so the benchmark can report
  per-frame latency and frame intervals from the actual demo loop.
- Replaced benchmark-side `setTimeout(0)` frame polling with a `visionaryDemoDebug.waitForFrameCount`
  promise that resolves directly from `recordGeneratedFrame()`. This avoids the benchmark observer
  competing with the demo's main-thread stream loop.
- The benchmark now writes `schema_version: 3` with `benchmark_kind: actual_demo_stream`,
  `demo.initial`/`demo.final` runtime snapshots, and `streaming_frame.output_validation` based on
  visible screenshot hashes plus a loose brick-band coverage check. Validation fails if generated
  frames are static or the visible Breakout brick band catastrophically disappears.
- Added an on-demand numerical validation hook for the actual demo benchmark. During the validation
  window only, the demo records CPU `final_z` summaries for WASM frames; the benchmark now requires
  at least two unique finite latent hashes for `provider=wasm`. The timed window turns the hook off,
  so the reported FPS remains the normal demo stream path.
- Updated the Playwright wrapper to rebuild `demo/main.js` before running the demo smoke or actual
  demo benchmark specs. The generated bundle is ignored by git, so this prevents a clean worktree
  from accidentally benchmarking stale code after local runtime probes.
- Removed the benchmark-only browser entry from the build. `bun run build:webgpu:browser` now builds
  only `demo/main.ts`; the benchmark uses the built demo bundle directly.
- Added a Playwright `webkit` project so the same WASM actual-demo benchmark can run under a
  Safari-family engine in addition to Chrome.
- Baseline after the proxy fix:
  - Chrome WebGPU default actual demo after removing benchmark polling: `37.7 fps`, visible
    validation passed.
  - Chrome WebGPU graph-capture actual demo after removing benchmark polling: `39.7 fps`, visible
    validation passed.
  - Chrome WASM actual demo after removing benchmark polling: `23.3 fps`, visible validation
    passed, decoder worker enabled.
  - WebKit/Safari-family WASM actual demo after removing benchmark polling: `21.5 fps`, visible
    validation passed, decoder worker enabled.
- Conclusion: the old standalone WASM benchmark was optimistic. The real demo path is currently
  about `22 fps` for WASM, so the remaining optimization target is roughly a `2.7x` speedup to reach
  `60 fps` without changing `sample_steps=2`.

### 2026-05-21 WASM Artifact Selection

- Added benchmark stage timing from the actual demo frame loop. For Chrome WASM with the previous
  default packed full-cache entry graph, visible validation passed but the frame was dominated by
  dynamics: `23.3 fps`, `42.9 ms/frame`, `dynamicsMs ~= 40.3 ms`, cache update wait
  `~= 1.6 ms`, and render `~= 0.06 ms`. The decoder worker is overlapped with dynamics; its total
  worker time was `~= 43.6 ms`, but `decoderWaitMs` was effectively zero.
- Tested available dynamics artifacts without changing `sample_steps=2`:
  - `breakout_dynamics_sample_append_context_cache_length_entry_b1_t1_s2`: Chrome WASM
    `22.9 fps`, validation passed; reject.
  - `breakout_dynamics_sample_append_context_full_cache_entry_b1_t1_s2`: Chrome WASM
    `24.2 fps`, validation passed; WebKit/Safari-family WASM `22.2 fps`, validation passed.
- Promoted `breakout_dynamics_sample_append_context_full_cache_entry_b1_t1_s2` as the WASM default
  full-cache step artifact. WebGPU keeps the packed artifact, and Safari-profile WebGPU keeps the
  Safari-specific graph-capture-safe artifact.

### 2026-05-21 Restored WASM Runtime Defaults

- Found a post-refactor default mismatch: `demo/index.html` still provided
  `data-ort-module=/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs`, so `backend=wasm`
  could load the larger WebGPU bundle unless the benchmark passed an explicit `ortModule`.
- Split the demo defaults so `backend=wasm` uses:
  - `assetBase=/dream_arcade_assets/breakout_wasm_default_mha`
  - `ortModule=/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs`
  - `wasmNumThreads=4` in Chrome and `3` in Safari/WebKit
  - decoder worker pipeline with `decoderWorkerNumThreads=3`
- The benchmark now passes those same defaults for `provider=wasm` and records the resolved
  `asset_base`, `ort_module_url`, `wasm_num_threads`, and `graph_optimization_level` from the
  actual demo runtime snapshot.
- No ONNX graph or sampling config changed; this restores the previously JAX/export-validated
  WASM asset set as the default runtime path.
- Short actual-demo validation windows after the fix:
  - Chrome WASM no-override: `33.54 fps`, output validation passed, selected
    `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s2`,
    `ort.wasm.min.mjs`, `wasmNumThreads=4`.
  - WebKit/Safari-family WASM no-override: `27.39 fps`, output validation passed with the same
    WASM artifact/runtime defaults.
  - Chrome WebGPU no-override: `36.59 fps`, output validation passed and still selected the WebGPU
    asset set under `dream_arcade_assets/breakout`.
- Rejected immediate follow-up runtime/serialization trials on the restored WASM default:
  - `ort.wasm.bundle.min.mjs` was slower than the standard WASM loader: `31.5 fps`.
  - `graphOptimizationLevel=extended` was slower than `all`: `31.4 fps`.
  - Disabling the decoder worker was slower: `27.0 fps`; the decoder itself is about `9.3 ms`
    sequentially, but overlapping it still wins despite worker contention.
  - Decoder worker thread counts `1`, `2`, and `4` did not beat the default `3`.
  - Main/decoder thread splits `5/1`, `5/2`, `5/3`, `6/1`, `6/2`, and `3/2` did not beat the
    default `4/3` split.
  - Serializing the entry graph through ORT `ORT_ENABLE_ALL` was CPU-exact against the accepted
    graph and reduced `3419 -> 3383` nodes, but browser validation was slower at `31.8 fps`.
    The ignored trial artifact and manifest entry were removed.
- WebKit/Safari-family thread retest found a better main-thread count than Chrome:
  - `4/3` adjacent control: `29.83 fps`.
  - `3/3`: `32.76 fps` on a short validated window.
  - `4/2`, `4/1`, `3/2`, and `2/2` did not beat `3/3`.
  Promoted the demo's Safari/WebKit WASM main-thread default to `3` while leaving Chrome at `4`;
  the benchmark no longer forces `wasmNumThreads=4` for provider `wasm`, so it records the actual
  browser-selected runtime default.
- Rejected follow-up runtime/data-movement probes on the actual-demo proxy:
  - Chrome high main-thread counts were clearly worse than the `4/3` split: `7/2` and `7/3` were
    about `24 fps`, `8/2` and `8/3` were about `21-23 fps`, and `10/2` was about `20 fps`.
  - Temporarily exposing `ort.env.wasm.simd="relaxed"` still validated but regressed a same-window
    Chrome control from `33.68 fps` to `31.57 fps`.
  - A SharedArrayBuffer-backed WASM K/V cache trial validated, but did not reduce worker cache wait
    and measured `31.65 fps`; ORT's input handling for SAB-backed tensors is not a win here.
  - Retested the latest published `onnxruntime-web@1.26.0` with matching JS and `.wasm` files staged
    under ignored `node_modules/ort126`; output validation passed but Chrome measured `31.58 fps`,
    slower than the pinned `1.24.3` control window.
  - Native ORT CPU profiling of the current entry graph still points at the same structural costs:
    `Gemm`, `Transpose`, `Gather`, `MultiHeadAttention`, `SimplifiedLayerNormalization`, and
    `Unsqueeze` dominate. This matches the prior rejected GQA/materialization and layout-cleanup
    trials; a larger dynamics-graph change is needed for another material WASM jump.
  - Retested the existing `breakout_dynamics_sample_append_context_slide_full_cache_b1_t1_s2`
    artifact, which avoids the entry-cache worker but returns the whole cache. It validated but was
    slower at `25.98 fps`, with dynamics around `35.3 ms`.
  - Temporarily instrumented normal-noise prefill in the demo loop. It measured only about
    `0.05 ms/frame`, so moving noise generation out of the hot path is not a meaningful route.
  - Forced the WASM entry-cache update back onto the main thread to check whether the cache worker
    now contends with the decoder worker. Output validation passed, but Chrome measured
    `31.20 fps`; keep the worker cache updater.

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

### Chrome Pure-WASM Export Split

Goal:
- Improve the pure WASM CPU fallback without changing `sample_steps=2`.
- Keep graph changes gated by exporter `--validate` against the JAX baseline.
- Keep WASM artifacts separate from WebGPU artifacts because ORT WebGPU and ORT WASM support
  different useful fused/layout operations.

What worked:
- Added `export_dreamer4_onnx.py --export_target wasm` with a separate WASM pass pipeline.
- The WASM profile uses the full-cache slide dynamics step
  `breakout_dynamics_sample_append_context_slide_full_cache_b1_t1_s2`; it returns full
  `candidate_k_cache`/`candidate_v_cache` tensors and avoids the JavaScript entry-cache update.
- The browser WASM path uses the external ORT `.wasm` loader:
  `/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs`.
- The WASM pass fuses only the dynamics hot path's temporal/BQHD attention islands into
  `com.microsoft::MultiHeadAttention`. The validated default export rewrote `35` temporal MHA
  islands and left decoder/spatial attention unfused.
- Best headed Chrome pure-WASM result in this iteration:
  - Artifact: `/dream_arcade_assets/breakout_wasm_default_mha`
  - Provider: `wasm`
  - ORT module: `/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs`
  - Threads: `wasmNumThreads=4`
  - Runtime graph optimization: `graphOptimizationLevel=basic`
  - Timed runs: `16`
  - Output validation: passed
  - JAX export validation: passed
  - Streaming: `19.84 fps`, `50.40 ms/frame`
  - Dynamics: `36.77 ms`
  - Decoder: `13.50 ms`

Rejected WASM controls from this iteration:
- Fusing all matched dynamics MHA islands, including spatial/BHQD, was JAX-valid but slower:
  `17.14 fps`, dynamics about `41.41 ms`, decoder about `16.77 ms`.
- Disabling the singleton reshape rewrite was JAX-valid but lost temporal MHA fusion and regressed
  to `13.91 fps`.
- Decoder MHA fusion was JAX-valid but only rewrote `2` decoder islands and regressed to
  `17.00 fps`.
- Disabling SwiGLU Gemm packing was JAX-valid but slower at `17.72 fps`.
- `onnxsim` on the demo hot path was JAX-valid but slower at `16.84 fps`.
- WASM thread count:
  - `1` thread: `10.42 fps`
  - `2` threads: `16.87 fps`
  - `4` threads: best at `19.84 fps`
  - `8` threads: `14.01 fps`
- Runtime `graphOptimizationLevel`:
  - `basic`: best at `19.84 fps`
  - `extended`: valid but slightly slower at `19.75 fps`
  - `disabled`: slower at `16.23 fps`
  - `all`: Chrome closed during navigation, so no valid result
- Browser ORT WASM session options did not help:
  - `enableCpuMemArena=true`: `18.04 fps`
  - `enableMemPattern=true`: `14.19 fps`
  - both enabled: `5.11 fps`
  - `executionMode=parallel`: `15.62 fps`

Current bottleneck:
- The best validated pure-WASM path is still well short of `30 fps`.
- Dynamics remains the dominant cost at roughly `37 ms/frame`; decoder adds roughly `13.5 ms`.
- Native ORT CPU profiling on the exported graph pointed at `Gemm` plus layout-heavy
  `Unsqueeze`/`Concat`/`Gather`/`Transpose` work, but the graph-level controls above did not
  produce a faster browser WASM path.

### Chrome Pure-WASM Decoder MHA Trial

Goal:
- Continue the pure WASM search without changing `sample_steps=2`.
- Gate graph changes with exporter `--validate` against the JAX baseline before benchmarking.

What changed:
- Added an opt-in WASM export flag, `--wasm_mha_decoder_fusion`, that extends the existing MHA
  rewrite to the single-frame decoder artifact.
- The decoder path now can fuse both:
  - unmasked `bqhd,bkhd->bhqk` / `bhqk,bkhd->bqhd` islands
  - masked spatial `bhqd,bhkd->bhqk` / `bhqk,bkhd->bqhd` islands when the attention bias can be
    passed as the fused op's optional attention-bias input.

Validated export:

```bash
uv run python webgpu_app/export/export_dreamer4_onnx.py \
  --tokenizer_dir gs://visionary-exp/dream-arcade/checkpoints/breakout_tokenizer_small_2x \
  --tokenizer_step 1000000 \
  --dynamics_dir gs://visionary-exp/dream-arcade/checkpoints/breakout_dynamics_small_2x \
  --dynamics_step 1000000 \
  --out_dir webgpu_app/dream_arcade_assets/breakout_wasm_decoder_bhqd_mha \
  --export_target wasm \
  --wasm_mha_decoder_fusion \
  --seq_len 64 \
  --sample_steps 2 \
  --export_cached \
  --validate \
  --overwrite
```

Validation/result:
- JAX export validation passed.
- `breakout_tokenizer_decoder_b1_t1` rewrote `8` decoder MHA islands:
  `6` BHQD masked islands with attention bias and `2` BQHD islands.
- `breakout_dynamics_sample_append_context_slide_full_cache_b1_t1_s2` kept the previously accepted
  `35` temporal/BQHD dynamics MHA rewrites.
- Headed Chrome pure-WASM benchmark with output validation passed:
  - Artifact: `/dream_arcade_assets/breakout_wasm_decoder_bhqd_mha`
  - ORT module: `/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs`
  - Threads: `wasmNumThreads=4`
  - Runtime graph optimization: `basic`
  - Timed runs: `16`
  - Streaming: `18.21 fps`, `54.93 ms/frame`
  - Dynamics: `40.06 ms`
  - Decoder: `14.76 ms`

Interpretation:
- This is JAX-valid and directionally improves the fresh same-session default rerun
  (`17.47 fps`, dynamics `41.42 ms`, decoder `15.66 ms`), but it does not beat the previous best
  recorded WASM run (`19.84 fps`).
- Native ORT CPU profiling of the decoder showed the rewrite replaced the decoder's large
  `Einsum`/`Softmax` work with `MultiHeadAttention` and reduced total decoder attention time, but
  browser end-to-end FPS is still dominated by dynamics and runtime variance.

Additional controls before stopping:
- `ort.jspi.min.mjs` was output-valid but slower: `16.05 fps`, dynamics about `45.5 ms`, decoder
  about `16.6 ms`.
- Existing slide-entry artifact was output-valid but much slower: `11.28 fps`; the JavaScript CPU
  cache slide/rebase path cost about `31.4 ms/frame`.
- Explicit `breakout_tokenizer_decode_z_b1_t1` decoder artifact was output-valid but slower than
  the default decoder path in the same window: `17.92 fps`, decoder about `15.0 ms`.
- A longer `64`-run headed default rerun completed after the stop request and stayed below target:
  `16.53 fps`, dynamics about `43.18 ms`, decoder about `17.23 ms`.

### 2026-05-19 KST: Pure-WASM S2 Retest And Rejected Cleanup Trials

Constraint:
- `sample_steps` must not be changed. The maintained WASM comparison target remains
  `sample_steps=2`.
- A temporary `sample_steps=1` WASM export was started before this clarification and did pass its
  own JAX/export and browser output-validation gates, but it is rejected as a non-candidate because
  it changes sampler semantics. The generated `breakout_wasm_s1_default_mha` trial directory was
  removed from the served assets.

Validated s2 WASM export:
- Regenerated `webgpu_app/dream_arcade_assets/breakout_wasm_default_mha` with:
  `--export_target wasm --sample_steps 2 --export_cached --validate --overwrite`.
- The manifest reports `export_target: wasm` and
  `preferred_full_cache_step_export_wasm:
  breakout_dynamics_sample_append_context_slide_entry_b1_t1_s2`.
- JAX/export validation passed for the preferred entry-cache slide step:
  - `final_z` max abs error: `4.3511391e-6`.
  - `candidate_k_entry` max abs error: `1.1205673e-5`.
  - `candidate_v_entry` max abs error: `9.8943710e-6`.
- The WASM MHA pass now rewrites `35` temporal/BQHD dynamics attention islands into
  `com.microsoft::MultiHeadAttention` for both the full-cache slide step and the preferred
  entry-cache slide step; spatial/BHQD attention remains unfused.

Browser WASM benchmark:
- Command shape:
  `bun run benchmark:webgpu -- --grep @smoke --webgpu-benchmark-provider wasm
  --webgpu-benchmark-asset-base /dream_arcade_assets/breakout_wasm_default_mha
  --webgpu-benchmark-ort-module /node_modules/onnxruntime-web/dist/ort.wasm.min.mjs
  --webgpu-benchmark-wasm-num-threads 4
  --webgpu-benchmark-step-artifact breakout_dynamics_sample_append_context_slide_full_cache_b1_t1_s2
  --webgpu-benchmark-graph-optimization-level basic --webgpu-benchmark-timed-runs 16`.
- Fresh local repeat with output validation passed:
  - Dynamics mean/median/p95: `36.57 / 34.68 / 42.61 ms`.
  - Decoder mean/median/p95: `12.80 / 12.52 / 14.08 ms`.
  - Streaming mean/median/p95: `49.51 / 47.42 / 56.92 ms`, or `20.20 fps`.
- This restored the local validated s2 full-cache WASM baseline after the non-candidate s1 run.
- Post-cleanup rerun of the same command also passed output validation, but was slower:
  dynamics `40.42 / 40.10 / 44.91 ms`, decoder `13.66 / 13.12 / 17.06 ms`, and streaming
  `54.21 / 53.74 / 61.34 ms` (`18.45 fps`). A later same-window full-cache control after the
  CPU entry-cache updater work was output-valid at streaming `46.50 / 45.82 / 49.65 ms`
  (`21.51 fps`), with dynamics `34.13 / 33.44 / 36.89 ms` and decoder
  `12.28 / 12.29 / 12.74 ms`. Treat full-cache s2 WASM as noisy but consistently below target.

Rejected exact graph cleanup trial:
- Applied the latest exact WebGPU full-cache cleanups to a copy of the s2 WASM export:
  cache-layer `Slice/Squeeze -> Gather`, one-position RoPE transpose removal, final output-head
  slice transpose cleanup, shared gather/add constant folding, and rank-2 SwiGLU islands on the
  step and decoder artifacts.
- CPU comparison against the JAX-validated s2 WASM export was exact for two dynamics seeds and one
  decoder seed:
  - `final_z`, `pred_z`, `candidate_k_cache`, `candidate_v_cache`, and `patches` all had
    max/mean absolute error `0.0`.
- Browser output validation passed, but performance regressed:
  - Streaming mean/median/p95: `47.69 / 46.89 / 50.91 ms`, or `20.97 fps`.
- Conclusion: reject for WASM. The graph is smaller, but ORT WASM does not benefit from these
  layout-oriented WebGPU cleanups.

Rejected output-contract trial:
- Removed the redundant `pred_z` output from the full-cache slide graph by aliasing the sampled
  `pred_z` producer as `final_z`.
- CPU comparison against the s2 default graph stayed within tiny float noise on cache outputs
  (`candidate_k_cache` max about `5.25e-6`, `candidate_v_cache` max about `3.58e-6`) and `final_z`
  exactly matched the default graph's `pred_z`.
- Browser output validation passed, but streaming regressed to `48.01 / 47.08 / 53.67 ms`
  (`20.83 fps`).
- Conclusion: reject. Removing the output does not help the WASM path.

Rejected raw entry-cache MHA trial:
- Fused the temporal/BQHD attention islands in the smaller entry-cache step graph:
  `35` `MultiHeadAttention` rewrites, CPU-exact against the default entry graph.
- Browser output validation passed and the dynamics segment improved to
  `30.29 / 30.58 / 31.89 ms`, but the JavaScript CPU cache slide/rebase commit still cost
  `20.29 ms` mean.
- Full streaming regressed badly to `63.08 / 59.97 / 79.71 ms` (`15.85 fps`).
- Conclusion: reject. Full-cache slide output remains better for pure WASM because it avoids the
  CPU-side entry-cache update.

Rejected decoder MHA trial:
- Fused all eight decoder attention islands, including six masked BHQD sites with attention bias.
- CPU comparison against the default decoder was exact for `patches`.
- Browser output validation passed, but it did not materially improve the full s2 frame:
  streaming `45.74 / 45.90 / 47.51 ms` (`21.87 fps`) in the short run.
- Conclusion: neutral/noisy and still below target; do not replace the default decoder on this basis.

Rejected runtime controls:
- WASM `numThreads=3` on the validated s2 export:
  streaming `53.24 / 52.97 / 57.11 ms` (`18.78 fps`).
- WASM `numThreads=5`:
  streaming `51.13 / 50.65 / 53.73 ms` (`19.56 fps`).
- WASM `numThreads=6`:
  streaming `50.96 / 50.80 / 52.34 ms` (`19.62 fps`).
- Keep `wasmNumThreads=4` as the current local default. It is the best validated setting in this
  run window, though still far short of `30 fps`.

Additional 2026-05-19 controls:
- Native ORT CPU profiling of the accepted s2 WASM artifacts, with ORT basic graph optimization
  and four native CPU threads, still points at broad graph cost rather than one removable node:
  dynamics time is spread across `Gemm`, `Transpose`, `SimplifiedLayerNormalization`, `Concat`,
  `Gather`, `MultiHeadAttention`, `Unsqueeze`, and remaining `Einsum`; decoder time is dominated
  by attention `Einsum` plus RMSNorm. This matches the browser result: the frame is compute-bound
  across many dense/layout kernels, not JavaScript overhead.
- Tried WASM CPU preallocated outputs in the benchmark. Decoder-only preallocation stayed output
  valid but did not beat the no-preallocation same-window control. Step-output preallocation is
  invalid for this path: browser output validation failed because sampled latent/frame hashes became
  static. Rejected and removed the experimental benchmark code.
- Layer-cache slide control used the existing JAX-validated
  `breakout_dynamics_prefill_layer_cached_b1_t64` plus
  `breakout_dynamics_sample_append_context_slide_layer_b1_t1_s2`.
  Browser output validation passed, but streaming regressed to
  `54.06 / 53.86 / 55.58 ms` (`18.50 fps`). A temporal MHA follow-up on a copied layer artifact
  found `0` eligible rewrites, so this path was not pursued further.
- Exported a validated s2 WASM no-QKV-pack trial with `--skip_pack_qkv_gemm`. It kept temporal MHA
  fusion (`35` rewrites), increased the full-cache step from `291` to `433` `Gemm` nodes, and
  passed browser output validation. Its short run was `47.15 / 46.54 / 50.58 ms` (`21.21 fps`),
  essentially the same as a same-window default rerun at `47.39 / 47.06 / 50.86 ms`
  (`21.10 fps`). Reject as noise-level and not a path to `30 fps`.
- Temporary WASM runtime controls:
  - ORT WASM worker proxy failed before timing with
    `ArrayBuffer at index 3 is a duplicate of an earlier ArrayBuffer`, because the benchmark feeds
    the same latent tensor as both sample and context noise. Rejected without modifying the demo
    feed semantics.
  - Explicit `env.wasm.simd="relaxed"` passed output validation and produced one short run at
    `45.80 / 45.72 / 47.68 ms` (`21.83 fps`), while `env.wasm.simd="fixed"` was
    `48.19 / 46.50 / 54.82 ms` (`20.75 fps`) and the same-window default was
    `48.58 / 48.24 / 52.15 ms` (`20.59 fps`). The ORT loader still imports the same
    `ort-wasm-simd-threaded` module; the setting only changes feature checking, so treat this as
    noise-level and not a real optimization. The temporary benchmark wiring was removed.
- Exported a validated s2 WASM head-projection `Einsum` trial with
  `--head_projection_rewrite einsum`. It removed the layout head-projection pattern but also lost
  the accepted temporal MHA fusion (`35 -> 0`) and moved the step to `428` `Einsum` nodes.
  Browser output validation passed, but streaming regressed to
  `58.63 / 58.56 / 62.84 ms` (`17.06 fps`). Reject.
- Exported a validated s2 WASM no-squeeze-concat trial with `--skip_squeeze_concat_rewrite`.
  It preserved temporal MHA fusion but left the full-cache step with more `Squeeze` nodes
  (`376 -> 628`). Browser output validation passed. The short run was
  `47.58 / 47.03 / 51.59 ms` (`21.02 fps`) versus a same-window default rerun at
  `49.73 / 49.82 / 51.71 ms` (`20.11 fps`). This is within the existing default-run noise window
  and still far below `30 fps`; reject as insufficient evidence to replace the default artifact.
- Exported a validated s2 WASM no-spatial-QK-layout trial with
  `--skip_spatial_qk_head_layout_rewrite`. The full-cache slide step stayed numerically valid
  against JAX with `sample_steps=2`: `final_z` max abs error `4.3511391e-6`,
  `candidate_k_cache` max abs error `1.2278557e-5`, and `candidate_v_cache` max abs error
  `8.9444220e-6`. Skipping that layout rewrite let the WASM MHA pass rewrite `71` dynamics
  attention islands instead of `35` and removed the remaining step `Einsum`/`Softmax` nodes, but
  increased `Transpose` nodes to `499` and `Squeeze` nodes to `412`. Browser output validation
  passed. Short-run streaming was `47.47 / 47.74 / 48.80 ms` (`21.06 fps`). A longer `64`-frame
  run was `45.25 / 45.14 / 46.43 ms` (`22.10 fps`) versus a same-window default at
  `46.21 / 45.45 / 49.74 ms` (`21.64 fps`), but a repeat after promoting the asset name was
  `46.10 / 45.05 / 49.92 ms` (`21.69 fps`). Reject as noise-level: dynamics sometimes crosses
  the 30 FPS boundary, but the full frame remains around `45-46 ms` and the improvement is not
  large or stable enough to replace `breakout_wasm_default_mha`. The trial asset was removed.
- Exported a combined no-spatial-QK plus decoder-MHA trial with
  `--skip_spatial_qk_head_layout_rewrite --wasm_mha_decoder_fusion`. JAX/export validation passed
  with the same full-cache slide errors as the no-spatial-QK-only trial. The dynamics step had
  `71` MHA rewrites and the decoder had `8` MHA rewrites with attention bias at the six masked
  sites. Browser output validation passed, but the `64`-frame WASM benchmark regressed to
  streaming `49.44 / 47.39 / 61.71 ms` (`20.23 fps`), with decoder mean `14.16 ms`.
  Same-window default after removing the trial was `50.12 / 48.56 / 55.10 ms` (`19.95 fps`).
  Reject: the decoder MHA fusion does not combine constructively with the no-spatial-QK dynamics
  variant and remains well below `30 fps`.
- Exported a validated s2 WASM no-RotaryEmbedding trial with `--skip_rotary_embedding_rewrite`.
  The full-cache slide step stayed valid against JAX: `final_z` max abs error
  `5.3048134e-6`, `candidate_k_cache` max abs error `1.3470650e-5`, and
  `candidate_v_cache` max abs error `1.0572374e-5`. Without the contrib RoPE op, the dynamics
  step still fused `71` MHA islands and reduced `Transpose` nodes to `213`, but primitive RoPE
  raised `Mul` to `695`, `Split` to `406`, and `Concat` to `505`. Browser output validation
  passed. The `64`-frame run was streaming `48.16 / 47.56 / 51.51 ms` (`20.76 fps`) versus a
  same-window default rerun at `45.41 / 45.10 / 47.09 ms` (`22.02 fps`). Reject: ORT WASM's
  contrib `RotaryEmbedding` remains better than the decomposed primitive graph for the full frame.
- Exported a validated s2 WASM no-SwiGLU-pack trial with `--skip_pack_swiglu_gemm` while keeping
  QKV packing and temporal dynamics MHA. Validation matched the default numerical envelope:
  `final_z` max abs error `4.3511391e-6`, `candidate_k_cache` max abs error `1.2278557e-5`, and
  `candidate_v_cache` max abs error `8.9444220e-6`. The full-cache step kept `35` MHA rewrites
  but increased `Gemm` from `291` to `362`. Browser output validation passed. The `64`-frame run
  was streaming `45.82 / 44.97 / 48.71 ms` (`21.82 fps`); adjacent default windows were
  `45.41 / 45.10 / 47.09 ms` and `47.56 / 46.65 / 52.56 ms`. Reject as noise-level and not a path
  to `30 fps`; the accepted packed SwiGLU graph remains the default.
- Runtime loader control on the accepted s2 WASM export:
  `/node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs` passed browser output validation,
  but measured streaming `46.69 / 45.79 / 51.63 ms` (`21.42 fps`) versus the restored
  `/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs` latest at `46.24 / 45.15 / 52.69 ms`
  (`21.63 fps`). Reject; the bundled loader does not change steady-state CPU execution enough to
  matter.
- Exported a validated s2 WASM no-attention-scale-folding trial with
  `--skip_attention_scale_folding`. The full-cache slide step stayed valid against JAX:
  `final_z` max abs error `2.8014183e-6`, `candidate_k_cache` max abs error `1.3828278e-5`, and
  `candidate_v_cache` max abs error `9.4175339e-6`. The graph kept the accepted `35` temporal MHA
  rewrites and only reintroduced the expected attention score `Mul`s (`Mul` `123 -> 159` after MHA
  fusion). Browser output validation passed, but the `64`-frame run was streaming
  `46.99 / 46.11 / 51.11 ms` (`21.28 fps`) and the restored default latest was
  `49.19 / 48.83 / 53.97 ms` (`20.33 fps`). Reject as noise-level and not a path to `30 fps`;
  keeping scale folded into the query norm remains the simpler default.
- Exported a validated s2 WASM no-Unsqueeze-Transpose-Squeeze-collapse trial with
  `--skip_unsqueeze_transpose_squeeze_rewrite`. The full-cache slide step matched the default
  validation envelope: `final_z` max abs error `4.3511391e-6`, `candidate_k_cache` max abs error
  `1.2278557e-5`, and `candidate_v_cache` max abs error `8.9444220e-6`. The graph kept the
  accepted `35` temporal MHA rewrites but left more shape-view nodes (`Squeeze` `376 -> 445` after
  MHA fusion). Browser output validation passed. The `64`-frame run was streaming
  `46.50 / 45.80 / 50.12 ms` (`21.50 fps`) and the restored default latest was
  `47.24 / 46.12 / 52.05 ms` (`21.17 fps`). Reject as noise-level; the collapse remains the
  accepted default because it keeps the graph smaller without a stable WASM regression.
- Exported a validated s2 WASM no-offline-ORT-optimization trial with `--skip_onnx_optimization`.
  The full-cache slide step stayed numerically valid against JAX, though with slightly larger tiny
  float noise: `final_z` max abs error `1.2338161e-5`, `candidate_k_cache` max abs error
  `3.7074089e-5`, and `candidate_v_cache` max abs error `2.5033951e-5`. Skipping offline ORT
  cleanup prevented the temporal MHA rewrite (`35 -> 0`), left `286` `Reshape` nodes after the MHA
  pass point, and raised hot-step `Gemm` to `504` plus `Mul` to `1772`. Browser output validation
  passed after Playwright retried an initial Chrome startup failure, but the short `16`-frame run
  regressed to streaming `60.25 / 59.68 / 63.41 ms` (`16.60 fps`) versus restored default at
  `48.53 / 48.16 / 53.40 ms` (`20.61 fps`). Reject; offline ORT cleanup is required for the
  current WASM graph pipeline.
- Temporarily added and tested a WASM export switch to skip the
  `Add + SimplifiedLayerNormalization -> SkipSimplifiedLayerNormalization` fusion. The full-cache
  slide step stayed numerically valid: `final_z` max abs error `2.6822090e-6`,
  `candidate_k_cache` max abs error `7.1525574e-6`, and `candidate_v_cache` max abs error
  `6.5565109e-6`. The accepted `35` temporal MHA rewrites stayed intact. Browser output validation
  passed, but the `64`-frame run was streaming `47.47 / 46.59 / 53.08 ms` (`21.07 fps`) versus a
  same-window default at `46.36 / 45.48 / 50.73 ms` (`21.57 fps`). Reject; the fused
  `SkipSimplifiedLayerNormalization` path remains at least as good for WASM. The temporary exporter
  switch was removed after the trial.
- Exported a validated s2 WASM packed-QKV-head trial with `--pack_qkv_head_projection`. The
  full-cache slide step stayed numerically valid against JAX with the same envelope as the accepted
  default: `final_z` max abs error `4.3511391e-6`, `candidate_k_cache` max abs error
  `1.2278557e-5`, and `candidate_v_cache` max abs error `8.9444220e-6`. The hot step still had the
  accepted `35` temporal MHA rewrites after the MHA pass point, and browser output validation
  passed. The `64`-frame run was streaming `46.26 / 45.22 / 49.99 ms` (`21.62 fps`) versus a
  same-window default at `46.31 / 45.77 / 49.06 ms` (`21.59 fps`). Reject as identical/noise-level
  for WASM and still far below `30 fps`; the trial asset was removed.
- Runtime-version control: staged `onnxruntime-web@1.26.0` under ignored `node_modules` and
  temporarily symlinked the served ORT WASM dist directory so the browser loaded matching 1.26.0 JS
  and `.wasm` files, without changing `package.json`/`bun.lock` or the accepted ONNX artifacts.
  The accepted s2 export was already JAX-valid; browser output validation passed for all runtime
  variants below:
  - Standard `ort.wasm.min.mjs`, `wasmNumThreads=4`, `graphOptimizationLevel=basic`:
    dynamics `32.59 / 32.46 / 33.92 ms`, decoder `12.27 / 12.26 / 12.71 ms`, streaming
    `44.94 / 44.81 / 46.73 ms` (`22.25 fps`).
  - Same-window pinned 1.24.3 control:
    dynamics `33.45 / 33.34 / 35.09 ms`, decoder `12.16 / 12.19 / 12.57 ms`, streaming
    `45.69 / 45.46 / 47.50 ms` (`21.89 fps`).
  - 1.26.0 default ORT thread setting was effectively identical to explicit four threads:
    streaming `44.96 / 44.47 / 47.74 ms` (`22.24 fps`).
  - 1.26.0 JSPI loader was valid but slower than the standard loader:
    streaming `45.39 / 45.21 / 46.87 ms` (`22.03 fps`).
  - 1.26.0 runtime graph optimization `extended` and `all` were both valid, but neutral/slower:
    `44.92 / 44.69 / 46.96 ms` (`22.26 fps`) and `45.25 / 45.16 / 46.44 ms` (`22.10 fps`).
  - 1.26.0 plus the explicit `breakout_tokenizer_decode_z_b1_t1` decoder artifact was also neutral:
    streaming `45.06 / 44.90 / 46.30 ms` (`22.19 fps`).
  Reject as insufficient for the target: 1.26.0 modestly improves dynamics, which now crosses
  `30 fps` alone in favorable windows, but the full sequential frame is still around `45 ms`.
  Keep the pinned 1.24.3 dependency until a larger runtime win justifies the package churn.
- Accepted follow-up: optimized the JavaScript CPU entry-cache updater for both
  `layer_batch_token_time_head_dim` and `layer_batch_token_head_time_dim` cache layouts. The fast
  path removes per-element cache-index helper calls from K rotation, uses typed-array block moves
  for V-cache sliding/appending, and keeps the fill path for partially populated caches in the demo.
  After also enabling the WASM MHA pass on the entry-cache slide artifact, the preferred WASM step is
  now `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s2` while keeping
  `sample_steps=2`.
- The promoted `breakout_wasm_default_mha` artifact was re-exported with validation and copied with
  the demo context/cache sidecars. Entry-step validation against JAX passed with `final_z` max abs
  `4.351139e-6`, `candidate_k_entry` max abs `1.1205673e-5`, and `candidate_v_entry` max abs
  `9.894371e-6`. The entry step has `35` temporal MHA rewrites after export.
- Browser output validation passed for the fused entry step. The accepted trial measured dynamics
  `26.92 / 26.58 / 28.59 ms`, decoder `11.97 / 12.02 / 12.45 ms`, cache commit
  `1.54 / 1.49 / 1.58 ms`, and streaming `40.47 / 39.93 / 42.71 ms` (`24.71 fps`).
- After promotion, the default WASM benchmark was run without explicitly passing
  `--webgpu-benchmark-step-artifact`; the manifest selected the entry-cache slide step. Output
  validation passed with `sample_steps=2`, dynamics `27.81 / 26.72 / 31.73 ms`, decoder
  `12.04 / 12.01 / 12.53 ms`, cache commit `1.54 / 1.48 / 1.64 ms`, and streaming
  `41.43 / 40.25 / 46.29 ms` (`24.13 fps`).
- Demo validation passed after the sidecars were copied into the promoted WASM asset:
  `world model demo starts and renders a frame @demo` and
  `world model demo changes the display over generated frames @demo` both passed in headed Chrome
  with `?backend=wasm&assetBase=/dream_arcade_assets/breakout_wasm_default_mha`.
- Decoder MHA combined with the fused entry step was output-valid but neutral/slower:
  dynamics `26.82 / 26.58 / 28.43 ms`, decoder `12.24 / 12.18 / 12.81 ms`, and streaming
  `40.65 / 40.32 / 43.27 ms` (`24.60 fps`). Reject; the decoder MHA graph is not an improvement
  over the promoted entry-only WASM path.
- Runtime-version control on the promoted entry-MHA path with temporary `onnxruntime-web@1.26.0`
  was also output-valid but neutral: dynamics `27.01 / 26.44 / 29.64 ms`, decoder
  `11.91 / 11.86 / 12.44 ms`, cache commit `1.54 / 1.48 / 1.73 ms`, and streaming
  `40.49 / 39.73 / 44.00 ms` (`24.70 fps`). Keep the pinned runtime unchanged.
- Runtime-thread control on the promoted entry-MHA path stayed output-valid and selected the
  manifest-preferred entry step with `sample_steps=2`, but did not beat four threads:
  - `wasmNumThreads=1`: streaming `87.20 / 86.97 / 88.56 ms` (`11.47 fps`).
  - `wasmNumThreads=2`: streaming `53.16 / 52.95 / 54.73 ms` (`18.81 fps`).
  - `wasmNumThreads=3`: streaming `45.11 / 44.99 / 46.42 ms` (`22.17 fps`).
  - `wasmNumThreads=5`: streaming `44.94 / 44.69 / 46.02 ms` (`22.25 fps`), with faster decoder
    than four threads but much slower dynamics.
  - `wasmNumThreads=6`: streaming `46.10 / 45.86 / 48.02 ms` (`21.69 fps`), with faster decoder
    but much slower dynamics.
  - `wasmNumThreads=7`: streaming `46.76 / 46.26 / 48.36 ms` (`21.39 fps`), again with faster
    decoder but much slower dynamics.
  - `wasmNumThreads=8`: streaming `47.12 / 46.91 / 48.97 ms` (`21.22 fps`).
  - Same-window `wasmNumThreads=4` control: streaming `41.06 / 40.17 / 44.60 ms`
    (`24.35 fps`).
  Keep four threads as the default.
- Runtime graph optimization control on the promoted entry-MHA path:
  - Explicit `graphOptimizationLevel=extended` passed browser output validation and measured
    dynamics `27.05 / 26.81 / 28.49 ms`, decoder `11.42 / 11.37 / 11.80 ms`, cache commit
    `1.54 / 1.48 / 1.76 ms`, and streaming `40.05 / 39.70 / 42.24 ms` (`24.97 fps`).
  - Explicit `graphOptimizationLevel=all` also passed, but was slightly slower:
    streaming `40.40 / 39.93 / 42.90 ms` (`24.76 fps`).
  - Promoted a WASM-only default of `graphOptimizationLevel=extended` in the benchmark and demo
    while leaving WebGPU on `basic`. A default WASM rerun with no explicit graph-optimization flag
    recorded `graphOptimizationLevel: "extended"`, selected
    `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s2`, passed output validation, and
    measured dynamics `26.91 / 26.65 / 28.87 ms`, decoder `11.49 / 11.51 / 12.00 ms`, cache commit
    `1.54 / 1.48 / 1.62 ms`, and streaming `39.98 / 39.55 / 41.98 ms` (`25.01 fps`).
  - Demo smoke on the default WASM path passed both `starts and renders a frame @demo` and
    `changes the display over generated frames @demo`.
- Explicit `breakout_tokenizer_decode_z_b1_t1` decoder on the promoted entry-MHA path with the new
  `extended` WASM runtime default passed output validation, but was slower/noisier than the default
  decoder: dynamics `27.79 / 26.85 / 33.74 ms`, decoder `11.72 / 11.56 / 12.34 ms`, cache commit
  `1.54 / 1.48 / 1.74 ms`, and streaming `41.09 / 39.86 / 47.78 ms` (`24.34 fps`). Keep
  `breakout_tokenizer_decoder_b1_t1` as the preferred decoder.
- Tried a benchmark-only scheduling change that starts the threaded WASM decoder promise before the
  CPU entry-cache commit, then awaits the decoder output. Browser output validation passed, but the
  frame did not improve: streaming `40.43 / 39.84 / 43.62 ms` (`24.73 fps`) versus the default
  `extended` control at `39.98 / 39.55 / 41.98 ms` (`25.01 fps`). Rejected and removed; the ORT
  WASM decoder run does not overlap usefully with the main-thread cache commit in this path.
- Runtime-loader control on the promoted entry-MHA path with `extended`: the bundled WASM loader
  `/node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs` passed output validation and measured
  streaming `39.80 / 39.64 / 41.30 ms` (`25.12 fps`), while the same-window standard loader control
  was `40.03 / 39.67 / 41.97 ms` (`24.98 fps`). This is too small to justify changing the default;
  keep `/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs`.
- JSPI loader control on the promoted entry-MHA path with the pinned `onnxruntime-web@1.24.3`
  runtime was output-valid but slower than the standard loader: dynamics
  `27.07 / 26.84 / 28.65 ms`, decoder `11.98 / 12.04 / 12.39 ms`, cache commit
  `1.59 / 1.50 / 1.74 ms`, and streaming `40.69 / 40.34 / 42.35 ms` (`24.58 fps`). Reject and
  keep the standard `/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs` loader.
- Additional ORT WASM session-option controls on the promoted entry-MHA path all passed browser
  output validation but did not improve the default:
  - `executionMode=parallel`: streaming `41.95 / 41.57 / 44.11 ms` (`23.84 fps`).
  - `enableCpuMemArena=true`: streaming `40.45 / 39.91 / 43.97 ms` (`24.72 fps`).
  - `enableMemPattern=true`: streaming `40.19 / 39.98 / 42.05 ms` (`24.88 fps`).
  Keep the default ORT session settings other than the accepted WASM-only `extended` graph
  optimization level.
- Retried ORT WASM proxy mode on the promoted entry-MHA path after avoiding the earlier duplicate
  transferable buffer for sample/context noise. It still failed before timing because proxy mode
  transfers and detaches the persistent K/V cache input buffers; the CPU entry-cache updater then
  raises `Cannot perform %TypedArray%.prototype.copyWithin on a detached or out-of-bounds
  ArrayBuffer`. Making proxy mode work would require cloning the full K/V cache every frame, which
  would add large memory copies. Reject and keep proxy disabled.
- Exported a validated WASM trial with `--skip_temporal_attention_bhsd_rewrite` while keeping
  `sample_steps=2` and the entry-step MHA fusion. JAX/export validation matched the accepted
  envelope: `final_z` max abs `4.351139e-6`, `candidate_k_entry` max abs `1.1205673e-5`, and
  `candidate_v_entry` max abs `9.894371e-6`. Browser output validation passed and the manifest still
  selected `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s2`, but performance regressed:
  dynamics `28.43 / 28.36 / 31.81 ms`, decoder `11.75 / 11.78 / 12.24 ms`, cache commit
  `1.54 / 1.48 / 1.60 ms`, and streaming `41.76 / 41.74 / 45.84 ms` (`23.95 fps`). Reject; the
  temporal BHSD rewrite remains part of the accepted WASM export pipeline.
- Tried a benchmark-only split runtime graph-optimization control: dynamics step sessions at
  `graphOptimizationLevel=all`, decoder sessions at `extended`. The first run passed validation and
  measured streaming `40.06 / 39.85 / 41.87 ms` (`24.96 fps`), but the repeat regressed to
  `41.29 / 40.70 / 43.95 ms` (`24.22 fps`). Reject as unstable/noise-level and keep the simpler
  single WASM default of `extended` for all sessions.
- Follow-up split session controls on the promoted entry-MHA path also passed browser output
  validation but did not beat the accepted single `extended` default:
  - Decoder `basic` with dynamics still at `extended`: streaming `39.90 / 39.60 / 42.08 ms`
    (`25.06 fps`), decoder `11.46 / 11.45 / 11.87 ms`.
  - Decoder `disabled`: streaming `39.81 / 39.54 / 41.96 ms` (`25.12 fps`), decoder
    `11.51 / 11.48 / 12.04 ms`.
  - Decoder `all`: streaming `40.18 / 39.65 / 42.99 ms` (`24.89 fps`), decoder
    `11.51 / 11.48 / 12.10 ms`.
  - Dynamics `basic` with decoder still at `extended`: streaming `40.51 / 39.96 / 42.79 ms`
    (`24.69 fps`), dynamics `27.26 / 26.80 / 29.37 ms`.
  - Dynamics `disabled`: streaming `41.55 / 41.07 / 45.43 ms` (`24.07 fps`), dynamics
    `27.97 / 27.54 / 30.35 ms`.
  Reject; the temporary benchmark wiring was removed and the accepted default was restored in
  `bench/results/latest.json` with output validation passed at streaming
  `40.09 / 39.83 / 41.56 ms` (`24.95 fps`).
- Tried a benchmark-only decoder pipeline on the promoted entry-MHA WASM path: after each dynamics
  step, the benchmark committed the entry-cache update and let the next dynamics step run while the
  previous decoder promise was pending, then measured displayed-frame intervals. Browser output
  validation passed and the manifest still selected the entry slide step at `sample_steps=2`, but
  ORT WASM contention erased the intended overlap. Dynamics regressed to
  `38.12 / 38.14 / 39.86 ms`, decoder was `12.18 / 12.18 / 12.62 ms`, and displayed-frame
  intervals were `41.12 / 39.79 / 52.82 ms` (`24.32 fps`). Reject; separate decoder and dynamics
  runs do not pipeline usefully on the shared ORT WASM runtime. The temporary benchmark wiring was
  removed and the accepted default was restored in `bench/results/latest.json` at streaming
  `40.04 / 39.80 / 42.01 ms` (`24.97 fps`).
- Retried `onnxsim` on the promoted entry-MHA WASM path with
  `--simplify_onnx --simplify_demo_only`, keeping `sample_steps=2`. The entry slide step stayed
  JAX/export valid (`final_z` max abs `4.351139e-6`, `candidate_k_entry` max abs
  `1.1205673e-5`, `candidate_v_entry` max abs `9.894371e-6`), and `onnxsim` reduced the raw entry
  step from `8867` to `5897` nodes before downstream rewrites while preserving the accepted `35`
  temporal MHA rewrites. Browser output validation passed, but performance regressed: dynamics
  `27.38 / 26.61 / 29.71 ms`, decoder `12.02 / 12.01 / 12.58 ms`, cache commit
  `1.58 / 1.47 / 1.74 ms`, and streaming `41.02 / 40.01 / 44.11 ms` (`24.38 fps`). Reject; the
  generated trial asset was removed.
- Restored the promoted default asset in `bench/results/latest.json` after the simplifier trial.
  With the manifest-selected entry slide step, output validation passed at `sample_steps=2` and
  measured dynamics `26.77 / 26.61 / 28.11 ms`, decoder `11.45 / 11.50 / 11.89 ms`, cache commit
  `1.54 / 1.48 / 1.61 ms`, and streaming `39.80 / 39.62 / 41.64 ms` (`25.12 fps`).
- Retested `--skip_spatial_qk_head_layout_rewrite` on the current entry-cache WASM path, not just
  the older full-cache path. JAX/export validation passed with the accepted numerical envelope:
  `final_z` max abs `4.351139e-6`, `candidate_k_entry` max abs `1.1205673e-5`, and
  `candidate_v_entry` max abs `9.894371e-6`. The entry slide graph fused `71` MHA islands instead
  of `35`, removing the remaining attention `Einsum`/`Softmax` nodes but raising `Transpose` to
  `499` and `Squeeze` to `412`. Browser output validation passed at `sample_steps=2`, but the
  full frame regressed: dynamics `26.85 / 25.99 / 32.05 ms`, decoder `12.27 / 12.24 / 13.08 ms`,
  cache commit `1.54 / 1.48 / 1.62 ms`, and streaming `40.70 / 39.78 / 46.51 ms` (`24.57 fps`).
  Reject; the extra MHA rewrites do not overcome the added layout cost on ORT WASM. The trial asset
  was removed and the accepted default was restored in `bench/results/latest.json` with streaming
  `40.42 / 39.64 / 44.25 ms` (`24.74 fps`).
- Native ORT CPU profiling of the accepted entry-MHA artifacts with `ORT_ENABLE_EXTENDED` reinforced
  that the decoder is real graph work, not just JavaScript/session overhead. Over the profiled run,
  decoder node time was led by `Einsum`, `SimplifiedLayerNormalization`, and `Gemm`; the entry step
  was spread across `Gemm`, `Unsqueeze`, `SimplifiedLayerNormalization`, `Einsum`, `Transpose`,
  `Gather`, `MultiHeadAttention`, `Concat`, and `Split`.
- Tested a WASM-only export trial that skipped the RMSNorm-to-`SimplifiedLayerNormalization` fusion.
  JAX/export validation passed at `sample_steps=2`: entry-step `final_z` max abs `6.139278e-6`,
  `candidate_k_entry` max abs `1.966953e-5`, `candidate_v_entry` max abs `1.385063e-5`; decoder
  `patches` max abs `3.755093e-6`. The graph kept primitive `ReduceMean/Sqrt/Div/Mul/Add` norm
  arithmetic and also changed the entry-step MHA match count to `71`, but browser output validation
  showed it was slower: dynamics `28.22 / 28.16 / 30.27 ms`, decoder
  `12.54 / 12.58 / 12.81 ms`, cache commit `1.55 / 1.49 / 1.70 ms`, and streaming
  `42.36 / 42.37 / 44.51 ms` (`23.61 fps`). Reject; ORT WASM's fused normalization remains better
  than the decomposed primitive graph. The temporary export flag and trial asset were removed, and
  the accepted default was restored in `bench/results/latest.json` with streaming
  `39.89 / 39.56 / 41.76 ms` (`25.07 fps`).
- Retested projection packing controls on the current entry-cache WASM path, because the earlier
  no-pack trials were measured against the older full-cache path:
  - `--skip_pack_swiglu_gemm` was JAX-valid at the accepted envelope (`final_z` max abs
    `4.351139e-6`, `candidate_k_entry` `1.120567e-5`, `candidate_v_entry` `9.894371e-6`) and
    preserved the `35` temporal MHA rewrites. It traded step `Split` nodes down from `251` to
    `180`, but raised `Gemm` from `291` to `362`; decoder `Gemm` rose from `34` to `42`.
    Browser output validation passed, but streaming regressed to
    `40.79 / 39.99 / 43.81 ms` (`24.52 fps`). Reject; the extra Gemm work costs more than the
    removed splits on ORT WASM.
  - `--skip_pack_qkv_gemm` was also JAX-valid with the accepted error envelope and preserved the
    `35` temporal MHA rewrites. It raised the entry step to `433` `Gemm` and `322` `Split` nodes,
    and decoder `Gemm` rose to `50`. Browser output validation passed, but streaming was
    `40.43 / 40.21 / 42.46 ms` (`24.73 fps`). Reject and keep both QKV and SwiGLU packing enabled.
  Both trial assets were removed, and the accepted default was restored in `bench/results/latest.json`
  with streaming `40.18 / 40.01 / 41.75 ms` (`24.89 fps`).
- Retested `--pack_qkv_head_projection` on the current entry-cache WASM path while keeping
  `sample_steps=2`. Export/JAX validation passed with the accepted entry-step envelope, but the
  pass found no sibling head-projection patterns to rewrite: `rewrites: {}` for both the preferred
  entry slide step and the t1 decoder. The resulting ONNX files were byte-identical to the accepted
  default asset (`sha256` entry step `25bd583654093fe44c47f4bd4756c2999c38a6cba6cd68b8a8027ae7be7bda60`,
  decoder `eae50cd813d08181b174a3a19f6a1dbbd6eeffb73fe218d5640b09d3491fc718`), so no browser
  benchmark was run for this no-op trial. The trial asset was removed.
- Accepted WASM attention `Einsum -> MatMul` rewrite:
  - Added a WASM attention pass that rewrites the supported attention equations
    `bhqd,bhkd->bhqk`, `bqhd,bkhd->bhqk`, and `bhqk,bkhd->bqhd` into equivalent batched `MatMul`
    with explicit `Transpose` layout nodes. The pass runs after optional MHA fusion, so temporal
    dynamics MHA rewrites stay fused and only the remaining explicit attention `Einsum`s are
    converted.
  - A copied ONNX-only prototype was bit-exact against the accepted JAX-validated decoder on CPU
    (`patches` max/mean abs `0.0`) and improved native ORT CPU decoder timing from about
    `7.54 ms` to `6.69 ms` in a 60-run local profile.
  - Re-exported a validated s2 WASM trial with the pass enabled by default. JAX/export validation
    passed for the t1 decoder with the accepted envelope (`patches` max abs `4.976988e-6`) and for
    the preferred entry step (`final_z` max abs `4.351139e-6`, `candidate_k_entry`
    `1.1205673e-5`, `candidate_v_entry` `9.894371e-6`).
  - The t1 decoder rewrite count was `16` total: `6` `bhqd,bhkd->bhqk`, `2`
    `bqhd,bkhd->bhqk`, and `8` `bhqk,bkhd->bqhd`. Decoder ops changed from
    `Einsum=16, MatMul=0, Transpose=12` to `Einsum=0, MatMul=16, Transpose=38`.
  - Browser output validation passed for the validated trial at `sample_steps=2`, measuring
    dynamics `27.05 / 26.74 / 28.81 ms`, decoder `10.34 / 10.29 / 10.84 ms`, cache commit
    `1.54 / 1.48 / 1.74 ms`, and streaming `38.97 / 38.64 / 41.30 ms` (`25.66 fps`). A
    same-window default control before promotion was slower at decoder `11.63 / 11.69 / 12.07 ms`
    and streaming `41.42 / 41.62 / 44.07 ms` (`24.15 fps`).
  - Promoted the validated MatMul decoder into `breakout_wasm_default_mha`. The promoted default
    benchmark passed browser output validation and selected the manifest-preferred entry slide step:
    dynamics `26.81 / 26.57 / 28.07 ms`, decoder `10.33 / 10.34 / 10.76 ms`, cache commit
    `1.56 / 1.50 / 1.84 ms`, and streaming `38.74 / 38.50 / 40.54 ms` (`25.82 fps`).
  - Demo smoke on the promoted default WASM asset passed both `starts and renders a frame @demo`
    and `changes the display over generated frames @demo`.
  - Extended the same MatMul rewrite to the preferred entry slide dynamics artifact after temporal
    MHA fusion. A copied-asset prototype was bit-exact against the promoted JAX-validated entry graph
    on CPU for `final_z`, `candidate_k_entry`, and `candidate_v_entry`, and browser output validation
    passed with dynamics `25.71 / 25.36 / 27.81 ms`, decoder `10.23 / 10.25 / 10.72 ms`, cache
    commit `1.60 / 1.49 / 1.78 ms`, and streaming `37.57 / 37.20 / 39.89 ms` (`26.61 fps`).
  - Re-exported a validated combined MatMul artifact with `sample_steps=2`. JAX/export validation
    passed for the preferred entry step (`final_z` max abs `4.351139e-6`, `candidate_k_entry`
    `1.1205673e-5`, `candidate_v_entry` `9.894371e-6`) and the t1 decoder (`patches` max abs
    `4.976988e-6`). Entry-step rewrites were `36` `bhqd,bhkd->bhqk` and `36`
    `bhqk,bkhd->bqhd`, while the accepted `35` temporal MHA rewrites remained in place.
  - Promoted the combined validated artifact into `breakout_wasm_default_mha`. The latest default
    browser benchmark passed output validation and selected
    `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s2`: dynamics
    `25.65 / 25.38 / 27.29 ms`, decoder `10.19 / 10.20 / 10.72 ms`, cache commit
    `1.56 / 1.50 / 1.85 ms`, and streaming `37.44 / 37.28 / 39.49 ms` (`26.71 fps`).
    Demo smoke passed again after promotion.
- Runtime controls retested on the combined MatMul graph all passed browser output validation but did
  not beat the accepted four-thread `extended` default:
  - `wasmNumThreads=5`: decoder improved to `9.71 / 9.71 / 9.94 ms`, but dynamics regressed to
    `31.63 / 31.37 / 32.61 ms` and full streaming regressed to `42.94 / 42.59 / 44.44 ms`
    (`23.29 fps`).
  - `wasmNumThreads=3`: dynamics `28.32 / 28.18 / 29.66 ms`, decoder
    `13.24 / 13.25 / 13.58 ms`, and streaming `43.15 / 43.03 / 44.44 ms` (`23.17 fps`).
  - `graphOptimizationLevel=all` with four threads: dynamics `25.67 / 25.47 / 27.19 ms`, decoder
    `10.39 / 10.39 / 10.76 ms`, and streaming `37.63 / 37.45 / 39.79 ms` (`26.58 fps`).
    Keep the WASM default at four threads and `extended`.
- Rejected a decoder singleton-key attention bypass on top of the combined MatMul graph. Two decoder
  attention islands had scalar score shape `[256, 8, 1, 1]`; replacing their output with the value
  tensor was bit-exact against the accepted decoder on CPU and improved native ORT decoder timing
  from about `6.66 ms` to `5.38 ms`, but browser WASM output-validation timing regressed:
  dynamics `26.72 / 26.42 / 28.82 ms`, decoder `10.17 / 10.20 / 10.59 ms`, cache commit
  `1.54 / 1.48 / 1.62 ms`, and streaming `38.47 / 38.14 / 40.75 ms` (`26.00 fps`). Reject; the
  ORT WASM browser path does not benefit from this exact decoder cleanup. The trial asset was
  removed and `bench/results/latest.json` was restored to the accepted default at streaming
  `37.70 / 37.43 / 39.30 ms` (`26.52 fps`).
- Rejected a JavaScript CPU cache-commit loop-order change. The trial changed the K-cache rotation
  loops to process each time slot contiguously and used typed-array appends for K, keeping the same
  arithmetic and passing browser output validation, but cache commit regressed from the accepted
  roughly `1.55 ms` window to `1.68 / 1.66 / 1.73 ms`; full streaming also regressed to
  `38.34 / 37.58 / 42.73 ms` (`26.08 fps`). The loop-order change was reverted. After restoring the
  accepted updater, browser output validation passed again with a noisy window of dynamics
  `27.01 / 25.90 / 29.48 ms`, decoder `10.63 / 10.36 / 10.87 ms`, cache commit
  `1.55 / 1.49 / 1.68 ms`, and streaming `39.23 / 37.81 / 42.59 ms` (`25.49 fps`).
- Rejected cache-layer `Slice/Squeeze -> Gather` on the current entry MatMul path. The existing exact
  rewrite removed `24` K/V cache layer slice pairs from the preferred entry step and was CPU-exact
  against the accepted graph for `final_z`, `candidate_k_entry`, and `candidate_v_entry`. Browser
  output validation passed, but it was indistinguishable from the same-window default: trial
  streaming `37.68 / 37.31 / 40.16 ms` (`26.54 fps`) versus default
  `37.66 / 37.36 / 40.07 ms` (`26.55 fps`). Reject as noise-level; replacing slice kernels with
  more gathers does not move the WASM path.
- Retested disabling temporal dynamics MHA now that explicit attention uses MatMul. The validated
  export used `--skip_wasm_mha_dynamics_fusion`, kept `sample_steps=2`, and rewrote all preferred
  entry-step attention islands to MatMul: `36` `bhqd,bhkd->bhqk`, `35` `bqhd,bkhd->bhqk`, and
  `71` `bhqk,bkhd->bqhd`. JAX/export validation passed with the accepted entry envelope, and
  browser output validation passed, but performance regressed: dynamics `26.66 / 26.67 / 27.76 ms`,
  decoder `10.28 / 10.31 / 10.68 ms`, cache commit `1.53 / 1.48 / 1.63 ms`, and streaming
  `38.51 / 38.45 / 40.04 ms` (`25.97 fps`). Keep the accepted hybrid of temporal MHA plus MatMul
  for the remaining explicit attention.
- Retested the bundled ORT WASM loader on the combined MatMul graph. It passed output validation and
  measured dynamics `25.70 / 25.44 / 27.32 ms`, decoder `10.25 / 10.26 / 10.75 ms`, cache commit
  `1.54 / 1.48 / 1.65 ms`, and streaming `37.53 / 37.16 / 39.45 ms` (`26.65 fps`), only about
  `0.1 ms` ahead of the adjacent standard-loader control (`37.63 / 37.35 / 40.28 ms`,
  `26.57 fps`). Keep the standard `/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs` loader.
  `bench/results/latest.json` was restored to the standard loader; the final restored window was
  noisy at streaming `38.72 / 38.21 / 43.41 ms` (`25.82 fps`) but remained output-valid.

Current WASM conclusion:
- The best behavior-preserving pure-WASM path is now the s2 entry-cache slide graph with temporal
  dynamics MHA, the MatMul attention rewrite on the decoder and remaining entry-step attention
  islands, the optimized JavaScript cache updater,
  `wasmNumThreads=4`, and the WASM-only ORT runtime graph optimization level `extended`.
- The full frame improved from roughly `45-50 ms` on the full-cache path to roughly `37.4-37.8 ms`
  in favorable local windows, or about `26.5-26.7 fps`; noisy repeats can still land around
  `39 ms`. Dynamics alone is now usually about `25.7 ms`, cache commit is about `1.5-1.6 ms`, and
  decoder is about `10.2 ms`, but the sequential full generated frame remains below the `30 fps`
  target.
- Further progress likely needs a genuinely faster CPU execution path for the decoder/dense blocks
  or a model/runtime-level change; the exact layout cleanups that help WebGPU are not translating
  into WASM speedups.

### 2026-05-19 KST: WASM Static Head-Merge Cleanup

Goal:
- Keep `sample_steps=2` and continue looking for behavior-preserving CPU/WASM wins after the
  combined attention MatMul path.
- Validate graph changes against the JAX/export gate before accepting them, and keep browser output
  validation plus demo smoke as runtime gates.

Accepted:
- Added a WASM-only static head-merge pass that replaces attention output
  `Split -> Concat -> Squeeze` islands with one equivalent static `Reshape`. This is deliberately
  narrower than restoring general `Reshape` lowering: it runs only after the accepted attention/MHA
  passes and only on the hot WASM decoder, decode-z, and preferred entry-slide artifacts.
- A prototype copy was CPU-exact against the accepted JAX-validated graphs:
  - Decoder `patches` max/mean abs error `0.0 / 0.0`.
  - Entry-step `final_z`, `candidate_k_entry`, and `candidate_v_entry` max/mean abs error all
    `0.0 / 0.0`.
- Re-exported `breakout_wasm_default_mha` with
  `--export_target wasm --sample_steps 2 --export_cached --validate --overwrite`. JAX/export
  validation passed with the same envelope:
  - Entry step: `final_z` max abs `4.351139e-6`, `candidate_k_entry` `1.1205673e-5`,
    `candidate_v_entry` `9.894371e-6`.
  - Decoder: `patches` max abs `4.976988e-6`.
- Manifest hot-path rewrites after the validated export:
  - Entry slide: `36` head-merge rewrites, node count `3561 -> 3489`.
  - Decoder: `8` head-merge rewrites, node count `403 -> 387`.
  - Decode-z: `8` head-merge rewrites, node count `469 -> 453`.
- Promoted-asset browser validation passed using the manifest-selected
  `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s2` step:
  - dynamics `25.34 / 25.29 ms` mean/median
  - decoder `10.26 / 10.25 ms`
  - cache commit `1.54 / 1.48 ms`
  - streaming `37.17 / 36.98 ms`, `26.90 fps`
  - output validation passed with `6` unique frame hashes and `6` unique latent hashes.
- Demo smoke on the same promoted WASM asset passed both `starts and renders a frame @demo` and
  `changes the display over generated frames @demo` in Chrome.
- The adjacent pre-promotion merge-only trial measured `37.22 / 37.05 ms` (`26.87 fps`), while the
  adjacent default control measured `37.41 / 37.24 ms` (`26.73 fps`). Treat the improvement as small
  but valid; it does not change the overall latency class.

Rejected / kept out:
- A broader static-layout prototype also replaced Q-head split/unsqueeze/concat islands. It was
  CPU-exact and reduced graph size more aggressively, but it added transposes and did not improve the
  browser full frame: streaming `38.29 / 37.63 ms` (`26.12 fps`) with decoder `10.47 ms`. Reject and
  keep only the direct head-merge `Reshape` cleanup.
- Retried the existing Q-head split-gather rewrite as a late pass after the accepted WASM
  attention/static-head passes, because that later graph still exposed `axis01_concat1` islands.
  The copied-asset trial was CPU-exact against the accepted graphs (`final_z`, K/V entries, decoder,
  and decode-z all max/mean abs `0.0 / 0.0`) and reduced the preferred entry step from `3489` to
  `3237` nodes, but browser output validation regressed: dynamics `28.01 / 26.73 ms`, decoder
  `10.82 / 10.82 ms`, cache commit `1.54 / 1.49 ms`, streaming `40.41 / 39.13 ms`
  (`24.75 fps`). Reject; the extra WASM `Gather`/`Transpose` work costs more than the removed
  `Split`/`Unsqueeze`/`Concat` dispatches.
- Retested the existing rank-2 SwiGLU island rewrite in isolation on the current entry-cache
  MatMul/static path. The copied-asset trial was CPU-exact against the accepted entry, decoder, and
  decode-z graphs, rewrote `71` entry SwiGLU islands, and reduced entry node count
  `3489 -> 3274`, but browser output validation was slower: dynamics `27.52 / 27.09 ms`, decoder
  `10.44 / 10.48 ms`, cache commit `1.53 / 1.48 ms`, streaming `39.53 / 38.89 ms`
  (`25.30 fps`). Reject; keeping these activation islands rank-2 removes layout nodes but hurts ORT
  WASM's steady-state execution on the current graph.
- Retested no-bias `Gemm -> MatMul` on the current entry-cache WASM path. The copied-asset trial
  rewrote `286` entry Gemms plus `32` decoder Gemms, and was CPU-exact against the accepted entry,
  decoder, and decode-z graphs. Browser output validation passed at streaming
  `37.28 / 37.07 ms` (`26.82 fps`), but the adjacent accepted default control was slightly faster:
  `37.21 / 36.96 ms` (`26.88 fps`). Reject as neutral/noise-level; ORT WASM's Gemm kernels remain
  at least as good as plain MatMul for the current dense projections.
- Retested the latest available `onnxruntime-web@1.26.0` runtime by temporarily serving its WASM
  `dist` files, without changing `package.json`, `bun.lock`, or the promoted artifacts. Both runs
  passed browser output validation, measuring `36.91 / 36.73 ms` (`27.10 fps`) and
  `37.46 / 37.25 ms` (`26.69 fps`), while adjacent pinned `1.24.3` controls measured
  `37.56 / 36.99 ms` (`26.62 fps`) and `37.01 / 36.82 ms` (`27.02 fps`). Reject as noise-level;
  the latest published runtime still does not move the current graph to `30 fps`.
- Retested `wasmNumThreads=2` on the current static-head path. Browser output validation passed, but
  dynamics and decoder both slowed sharply: streaming `50.50 / 50.14 ms` (`19.80 fps`). Keep the
  accepted four-thread default.
- Applied the accepted explicit-attention MatMul rewrite plus static head-merge cleanup to a copied
  full-cache slide artifact. The full-cache graph was CPU-exact against the accepted full-cache
  graph and improved that older path, but it was still slower than entry-cache: dynamics
  `31.51 / 31.30 ms`, decoder `10.31 / 10.33 ms`, near-zero cache commit, and streaming
  `41.91 / 41.66 ms` (`23.86 fps`). Reject; eliminating the JavaScript entry-cache commit is not
  enough to offset writing the full K/V cache from the dynamics graph.
- Applied the accepted explicit-attention MatMul rewrite plus static head-merge cleanup to a copied
  `cache_length_entry` artifact. It was CPU-exact against the accepted cache-length graph and
  output-valid in the browser, but stayed slower than the accepted slide-entry path: dynamics
  `27.75 / 27.60 ms`, decoder `10.41 / 10.42 ms`, cache commit `1.53 / 1.48 ms`, streaming
  `39.74 / 39.52 ms` (`25.16 fps`). Reject.
- Retried the `cache_length_entry` artifact with bias-capable MHA fusion. This fused `71` attention
  islands and was CPU-exact against the accepted cache-length graph, but browser output validation
  still measured only dynamics `26.90 / 26.78 ms`, decoder `10.33 / 10.31 ms`, cache commit
  `1.53 / 1.48 ms`, and streaming `38.79 / 38.68 ms` (`25.78 fps`). Reject; the accepted
  slide-entry graph remains faster than the dynamic-mask cache-length contract.
- Retested explicit `breakout_tokenizer_decode_z_b1_t1` after the static-head cleanup. Browser
  output validation passed and decoder timing was neutral/slightly faster in isolation
  (`10.17 / 10.20 ms`), but full-frame streaming stayed in the same band at
  `37.12 / 36.86 ms` (`26.94 fps`) versus the accepted manifest-default restore below. Keep
  `breakout_tokenizer_decoder_b1_t1` as the preferred decoder.
- Retested the exact one-position RoPE transpose cleanup on the current entry-cache WASM path. It
  removed `71` entry islands (`3489 -> 3347` nodes, `Transpose 391 -> 249`) plus `4` decoder
  islands, and was CPU-exact against the accepted entry, decoder, and decode-z graphs. Browser
  output validation passed, but streaming was neutral/slower at `37.60 / 37.49 ms`
  (`26.59 fps`). Reject for WASM; the transpose removal that helps the WebGPU full-cache path does
  not improve this CPU path.
- Retested the exact final output-head `Transpose -> Slice` cleanup on the current entry-cache path.
  It removes only two transposes and was CPU-exact for `final_z`, `candidate_k_entry`, and
  `candidate_v_entry`. Browser output validation passed twice, measuring `37.18 / 37.00 ms`
  (`26.90 fps`) and `37.87 / 37.11 ms` (`26.41 fps`), while the adjacent default control was
  `37.89 / 37.45 ms` (`26.39 fps`). Reject as neutral/noise-level; the graph change is too small to
  justify another accepted WASM export pass.
- Retested current-path runtime controls after the final static-head path:
  - `wasmNumThreads=6` passed output validation and improved decoder timing to `9.03 / 9.01 ms`,
    but dynamics regressed to `32.89 / 32.79 ms`; full streaming fell to `43.49 / 43.34 ms`
    (`22.99 fps`). Keep four threads.
  - The bundled WASM loader `/node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs` passed
    output validation but stayed slower/noisier than the standard loader: streaming
    `37.86 / 37.13 ms` (`26.41 fps`). Keep `/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs`.
  - The JSPI loader `/node_modules/onnxruntime-web/dist/ort.jspi.min.mjs` passed output validation
    on the current static-head artifact and measured `37.29 / 37.01 ms` (`26.82 fps`), essentially
    identical to the adjacent standard-loader restore. Keep the standard loader.
- Retested `graphOptimizationLevel=all` after introducing the static merge `Reshape`s. Browser
  output validation passed, but it remained slightly slower than `extended`: streaming
  `37.40 / 37.13 ms` (`26.74 fps`). Keep the WASM default at `extended`.
- Retested `graphOptimizationLevel=basic` after the static-head cleanup. Browser output validation
  passed, but it was still slower than the accepted `extended` default: streaming
  `38.61 / 38.08 ms` (`25.90 fps`). Keep `extended`.
- Retested the ORT default thread count by leaving `wasmNumThreads` unset on the current
  static-head artifact. Browser output validation passed, but timing stayed in the same band as the
  explicit four-thread default rather than improving: dynamics `25.55 / 25.27 ms`, decoder
  `10.23 / 10.24 ms`, cache commit `1.53 / 1.47 ms`, and streaming `37.34 / 37.19 ms`
  (`26.78 fps`). Keep the explicit four-thread default for reproducibility.
- Retested runtime `graphOptimizationLevel=disabled` on the current static-head artifact. Browser
  output validation passed, but disabling ORT's runtime graph optimizations was slower than
  `extended`: dynamics `26.98 / 26.56 ms`, decoder `10.53 / 10.53 ms`, cache commit
  `1.54 / 1.48 ms`, and streaming `39.08 / 38.71 ms` (`25.59 fps`). Keep `extended`.
- Retested a copied direct GQA-repeat fold on the current WASM path. The copied entry graph replaced
  `36` compact-head `Concat -> Gather(axis=2)` repeats with repeated-input `Concat`s
  (`3489 -> 3453` nodes, `Gather 178 -> 142`), and the copied decoder/decode-z graphs each removed
  `8` gathers. CPU comparison against the accepted artifacts was exact for `final_z`,
  `candidate_k_entry`, `candidate_v_entry`, and `patches`. Browser output validation passed twice,
  measuring streaming `37.40 / 37.08 ms` (`26.74 fps`) and `37.60 / 37.29 ms` (`26.59 fps`).
  Adjacent accepted-default controls moved from a noisy `38.60 / 38.67 ms` window back to
  `37.17 / 36.95 ms`; reject as neutral/noise-level.
- Retested replacing attention score `Transpose(K) -> MatMul` pairs with
  `com.microsoft::FusedMatMul(transB=1)` on copied artifacts. This was CPU-exact and removed `36`
  entry K-transposes plus `6` decoder/decode-z K-transposes, but browser output validation measured
  only streaming `37.77 / 37.09 ms` (`26.48 fps`). Reject; the contrib op is browser-compatible but
  does not improve the current ORT WASM path.
- Combined the copied direct GQA-repeat fold with the copied `FusedMatMul(transB=1)` rewrite. The
  combined entry graph was CPU-exact and shrank to `3417` nodes (`Gather 142`, `Transpose 355`,
  `MatMul 36`, `FusedMatMul 36`), with decoder/decode-z also exact. Browser output validation
  passed at streaming `37.26 / 37.04 ms` (`26.84 fps`), but the adjacent accepted default was
  slightly faster at `37.17 / 36.95 ms` (`26.90 fps`). Reject the combination.
- Tested a copied pure z-reshape decoder to avoid the benchmark/demo JavaScript copy from
  `final_z [1,1,32,32]` to decoder latent `[1,1,64,16]`. The trial replaced the existing
  `breakout_tokenizer_decode_z_b1_t1` file in a copied asset with a one-node `Reshape` into the
  accepted `breakout_tokenizer_decoder_b1_t1` graph. CPU comparison against the accepted decoder
  plus the same JS reshape was exact for `patches` (`0.0 / 0.0` max/mean abs). Browser output
  validation passed with the explicit decode-z decoder, but timing was neutral/slower:
  dynamics `26.34 / 25.50 ms`, decoder `10.27 / 10.28 ms`, pack/unpack near zero, cache commit
  `1.53 / 1.48 ms`, and streaming `38.18 / 37.27 ms` (`26.19 fps`). Reject; the JS reshape is not
  a material bottleneck.
- Temporarily exposed ORT `env.wasm.simd="relaxed"` as a benchmark query control. Browser output
  validation passed, but relaxed SIMD did not improve the current path: dynamics
  `26.09 / 25.29 ms`, decoder `10.27 / 10.26 ms`, cache commit `1.53 / 1.48 ms`, and streaming
  `37.93 / 37.03 ms` (`26.36 fps`). The temporary benchmark knob was removed; keep the standard
  WASM SIMD behavior.
- Tested a copied offline ORT `ENABLE_ALL` serialization of the current static-head entry,
  decoder, and decode-z artifacts. Serialization was CPU-exact and persisted ORT's
  MatMul-transpose fusions (`FusedMatMul 0 -> 36`, `Transpose 391 -> 355` on entry; decoder and
  decode-z each `FusedMatMul 0 -> 6`, `Transpose 38 -> 28`). Browser output validation passed, but
  runtime `extended` measured only streaming `37.52 / 37.18 ms` (`26.65 fps`), and runtime `basic`
  on the pre-fused graph regressed to `38.95 / 37.35 ms` (`25.67 fps`). Reject; persisting ORT's
  internal MatMul-transpose fusion does not beat the accepted runtime-optimized graph.
- Tested a copied head-time cache ABI trial on the current WASM asset. The prefill graph could
  expose `k_cache`/`v_cache` as `[layer,batch,token,head,time,dim]` exactly: after transposing the
  trial outputs back to the accepted `[layer,batch,token,time,head,dim]` layout, `pred_z`,
  `k_cache`, `v_cache`, and `cache_length` all matched with `0.0 / 0.0` max/mean abs error. The
  entry graph was rejected before browser timing: simply swapping cache slice bounds left the
  current WASM graph's time/head concat path inconsistent, and ORT refused to load it at
  `node_Concat_250` (`inferred=2`, `declared=64` on dimension `2`). A useful head-time cache ABI
  would need a broader temporal attention layout rewrite, not just the cache input shape swap.
  The copied trial asset was removed.
- Microbenchmarked a time-major JavaScript CPU cache-update loop and an unrolled variant against the
  accepted updater. The time-major order was slower in local V8 (`~1.15 ms` vs `~1.09 ms` after
  warmup), while full unrolling was only about `0.13 ms` faster in the synthetic loop and would not
  materially change the `37 ms` frame. Keep the clearer accepted cache updater.
- Accepted a WASM-only decoder singleton-key attention bypass. The pass targets decoder attention
  score tensors with shape `[256,8,1,1]`; `Softmax` over the 1-wide key axis is exactly one, so the
  following attention-value `MatMul` can be replaced by its value input. It rewrote `2` chains in
  both `breakout_tokenizer_decoder_b1_t1` and `breakout_tokenizer_decode_z_b1_t1`, reducing the
  t1 decoder from `387 -> 349` nodes (`MatMul 16 -> 12`, `Softmax 8 -> 6`, `Transpose 38 -> 26`,
  `Gather 18 -> 14`). CPU comparison against the accepted decoder was exact for both decoder
  artifacts (`patches` max/mean abs `0.0 / 0.0`).
- Re-exported the promoted `breakout_wasm_default_mha` artifact with `--export_target wasm`,
  `--sample_steps 2`, `--export_cached`, and `--validate`. JAX/export validation passed: entry
  `final_z` max abs `4.351139e-6`, `candidate_k_entry` `1.1205673e-5`,
  `candidate_v_entry` `9.894371e-6`; t1 decoder `patches` max abs `4.976988e-6`.
- Browser WASM output validation on the copied singleton-bypass trial passed twice:
  - trial: dynamics `25.50 / 25.32 ms`, decoder `9.52 / 9.49 ms`, cache commit
    `1.53 / 1.47 ms`, streaming `36.59 / 36.30 ms` (`27.33 fps`);
  - repeat: dynamics `25.58 / 25.28 ms`, decoder `9.51 / 9.50 ms`, cache commit
    `1.53 / 1.48 ms`, streaming `36.66 / 36.31 ms` (`27.28 fps`).
  The adjacent accepted-default control before promotion measured `37.52 / 36.95 ms`
  (`26.65 fps`) with decoder `10.29 / 10.28 ms`.
- Promoted default after re-export passed browser output validation with `6` unique frame hashes
  and `6` unique latent hashes. Latest timing: dynamics `25.40 / 25.18 ms`, decoder
  `9.42 / 9.39 ms`, cache commit `1.53 / 1.48 ms`, and streaming `36.39 / 36.09 ms`
  (`27.48 fps`). WASM demo smoke also passed `starts and renders a frame @demo` and
  `changes the display over generated frames @demo` with
  `?backend=wasm&assetBase=/dream_arcade_assets/breakout_wasm_default_mha`.
- Retested extending the singleton-key bypass to the preferred dynamics slide-entry graph. The
  copied-asset pass found no eligible singleton-key `MatMul -> Softmax -> MatMul` chains in the
  dynamics graph (`node_count 3489 -> 3489`), so there was no ONNX change to benchmark. The copied
  trial asset was removed.
- Temporarily exposed ORT WASM session allocation/scheduling controls in the benchmark only, then
  removed the temporary wiring after the runs:
  - `executionMode=parallel` passed output validation but regressed to dynamics
    `26.97 / 26.74 ms`, decoder `9.92 / 9.94 ms`, and streaming `38.46 / 38.12 ms`
    (`26.00 fps`). Reject.
  - `enableCpuMemArena=true` passed output validation but measured streaming
    `36.71 / 36.33 ms` (`27.24 fps`), behind the promoted control. Reject.
  - `enableMemPattern=true` passed output validation but measured streaming
    `37.13 / 36.39 ms` (`26.94 fps`). Reject.
  - Enabling both CPU arena and memory pattern passed output validation at
    `36.44 / 36.10 ms` (`27.45 fps`), essentially tied with the adjacent default
    `36.47 / 36.16 ms` (`27.42 fps`). Reject as noise-level.
- Tested reusing CPU preallocated output tensors for the WASM step and decoder, mirroring the
  WebGPU preallocated-output path. ORT accepted the fetch objects but returned static zero outputs:
  browser output validation failed with `1` unique frame hash and `1` unique latent hash, and the
  latest frame summary was all zeros. Reject as invalid for ORT WASM; the temporary patch was
  removed.
- Retested explicit `breakout_tokenizer_decode_z_b1_t1` after the decoder singleton-key bypass.
  Browser output validation passed, but timing was neutral/slower than the manifest-default
  decoder: dynamics `26.05 / 25.36 ms`, decoder `9.52 / 9.54 ms`, cache commit
  `1.54 / 1.48 ms`, and streaming `37.14 / 36.31 ms` (`26.93 fps`). Keep
  `breakout_tokenizer_decoder_b1_t1` as the preferred decoder.
- Retested the shared `Gather + Add(constant)` fold on the current WASM entry-cache path. The copied
  graph removed only three dynamics `Add` nodes (`3489 -> 3486`) and was CPU-exact against the
  accepted entry graph for `final_z`, `candidate_k_entry`, and `candidate_v_entry` (`0.0 / 0.0`
  max/mean abs). Browser output validation passed, first at streaming `36.37 / 36.08 ms`
  (`27.50 fps`), but a repeat regressed to `36.86 / 36.48 ms` (`27.13 fps`) and matched the
  adjacent default noise window. Reject; the cleanup is too small to justify another accepted pass.
  The copied trial asset was removed.
- Retested untried WASM thread-count controls after the decoder bypass. Both passed output
  validation but were much slower than the accepted four-thread default:
  - `wasmNumThreads=3`: dynamics `28.07 / 27.52 ms`, decoder `12.26 / 12.23 ms`, streaming
    `41.91 / 41.29 ms` (`23.86 fps`).
  - `wasmNumThreads=5`: dynamics `31.30 / 31.00 ms`, decoder `8.71 / 8.69 ms`, streaming
    `41.59 / 41.20 ms` (`24.04 fps`).
  Keep `wasmNumThreads=4`.
- Retested runtime `graphOptimizationLevel=all` after the decoder singleton-key bypass. Browser
  output validation passed, but it was still slower than the accepted `extended` control:
  dynamics `25.95 / 25.59 ms`, decoder `9.47 / 9.48 ms`, cache commit `1.53 / 1.48 ms`, and
  streaming `36.99 / 36.57 ms` (`27.03 fps`). Keep `extended`.
- Used native ORT CPU profiling as a direction finder on the accepted graphs. The entry profile was
  still spread across `Gemm`, `SimplifiedLayerNormalization`, shape ops, `Transpose`, `Gather`, MHA,
  and `Concat`, while the t1 decoder showed `SimplifiedLayerNormalization` as the largest native
  slice. This suggested trying a WASM-specific decoder RMSNorm lowering, but the native profile was
  not treated as browser evidence.
- Tested a copied decoder primitive RMSNorm rewrite. The trial replaced each decoder
  `SimplifiedLayerNormalization` with equivalent RMSNorm arithmetic (`Mul`, `ReduceMean`, `Add`,
  `Sqrt`, `Div`, `Mul`) in `breakout_tokenizer_decoder_b1_t1`, increasing that graph from
  `349 -> 429` nodes and removing all `16` decoder `SimplifiedLayerNormalization` nodes. CPU
  comparison against the accepted decoder passed with `patches` max/mean abs
  `3.2186508e-6 / 4.4784e-8`. Browser output validation passed on the copied asset:
  - first run: dynamics `26.41 / 25.48 ms`, decoder `9.22 / 9.17 ms`, cache commit
    `1.54 / 1.48 ms`, streaming `37.21 / 36.21 ms` (`26.88 fps`);
  - adjacent accepted default: dynamics `25.43 / 25.31 ms`, decoder `9.39 / 9.35 ms`, cache commit
    `1.53 / 1.48 ms`, streaming `36.39 / 36.17 ms` (`27.48 fps`);
  - repeat trial: dynamics `25.52 / 25.31 ms`, decoder `9.16 / 9.16 ms`, cache commit
    `1.53 / 1.48 ms`, streaming `36.25 / 36.01 ms` (`27.59 fps`).
  Accept the decoder-only rewrite because it consistently reduced decoder median time by roughly
  `0.18-0.25 ms` without changing output validity.
- Promoted the decoder primitive RMSNorm rewrite into the WASM export pipeline for
  `breakout_tokenizer_decoder_b1_t1` and `breakout_tokenizer_decode_z_b1_t1` only; the preferred
  dynamics entry graph is explicitly marked disabled for this pass. Re-exported
  `breakout_wasm_default_mha` with `--export_target wasm`, `--sample_steps 2`, `--export_cached`,
  and `--validate`. Manifest checks kept `demo_generation.sample_steps = 2`. JAX/export validation
  passed: entry `final_z` max abs `4.351139e-6`, `candidate_k_entry` `1.1205673e-5`,
  `candidate_v_entry` `9.894371e-6`; t1 decoder `patches` max abs `3.904104e-6`; decode-z
  `patches` max abs `3.337860e-6`. The promoted pass rewrote `16` RMSNorms in both decoder
  artifacts (`349 -> 429` and `415 -> 495` nodes) and left the entry graph at `3489` nodes.
- Promoted browser WASM output validation passed twice with the standard WASM loader,
  `wasmNumThreads=4`, and `graphOptimizationLevel=extended`:
  - first run: dynamics `25.65 / 25.30 ms`, decoder `9.18 / 9.18 ms`, cache commit
    `1.53 / 1.48 ms`, streaming `36.39 / 36.00 ms` (`27.48 fps`);
  - repeat: dynamics `25.53 / 25.26 ms`, decoder `9.19 / 9.19 ms`, cache commit
    `1.52 / 1.48 ms`, streaming `36.28 / 35.87 ms` (`27.56 fps`).
  Output validation reported `6` unique frame hashes and `6` unique latent hashes. WASM demo smoke
  passed both `starts and renders a frame @demo` and `changes the display over generated frames
  @demo` with `?backend=wasm&assetBase=/dream_arcade_assets/breakout_wasm_default_mha`.
- Tested applying the decoder primitive RMSNorm rewrite to the preferred dynamics entry graph in a
  copied asset. It removed all `215` entry `SimplifiedLayerNormalization` nodes but expanded the
  entry graph from `3489 -> 4564` nodes. CPU comparison against the accepted entry graph passed:
  `final_z` max abs `3.5762787e-6`, `candidate_k_entry` `1.0848045e-5`, and
  `candidate_v_entry` `5.826354e-6`. Browser output validation passed, but dynamics and full-frame
  timing regressed to dynamics `25.83 / 25.74 ms`, decoder `9.22 / 9.21 ms`, cache commit
  `1.53 / 1.48 ms`, and streaming `36.62 / 36.49 ms` (`27.31 fps`). The adjacent accepted default
  was faster at streaming `36.34 / 36.08 ms` (`27.52 fps`). Reject; the primitive norm lowering is
  useful only for the small decoder graphs.
- Accepted an extension to the WASM static head-merge pass for the dynamics MHA query path. The
  copied trial replaced `35` ranked `Split(axis=2) -> Concat(axis=3) -> Squeeze(axis=2)` islands
  shaped `[36,1,8,32] -> [36,1,256]` with one static `Reshape` each, reducing the promoted entry
  graph from `3489 -> 3419` nodes. CPU comparison against the accepted entry graph was exact for
  `final_z`, `candidate_k_entry`, and `candidate_v_entry` (`0.0 / 0.0` max/mean abs).
  Browser output validation passed twice:
  - trial: dynamics `25.38 / 25.09 ms`, decoder `9.19 / 9.18 ms`, cache commit
    `1.54 / 1.49 ms`, streaming `36.15 / 35.85 ms` (`27.67 fps`);
  - repeat: dynamics `25.22 / 24.97 ms`, decoder `9.22 / 9.21 ms`, cache commit
    `1.54 / 1.49 ms`, streaming `36.02 / 35.73 ms` (`27.77 fps`).
  Adjacent accepted defaults stayed slower at `36.34 / 36.08 ms` (`27.52 fps`) and
  `36.37 / 36.06 ms` (`27.49 fps`), so this is a small but repeatable entry-graph win.
- Re-exported the promoted `breakout_wasm_default_mha` artifact with `--export_target wasm`,
  `--sample_steps 2`, `--export_cached`, and `--validate`. Manifest checks kept
  `demo_generation.sample_steps = 2`. JAX/export validation passed with the accepted envelope:
  entry `final_z` max abs `4.351139e-6`, `candidate_k_entry` `1.1205673e-5`,
  `candidate_v_entry` `9.894371e-6`; t1 decoder `patches` max abs `3.904104e-6`; decode-z
  `patches` max abs `3.337860e-6`. The static head-merge pass now rewrites both the original
  `36` 2D head merges and the `35` ranked MHA query merges, reducing the entry graph
  `3561 -> 3419` during the export pass (`Split 180`, `Concat 255`, `Squeeze 305`, `Reshape 71`
  after the pass).
- Promoted browser WASM output validation passed twice after the ranked head-merge export:
  - first run: dynamics `25.58 / 25.20 ms`, decoder `9.24 / 9.22 ms`, cache commit
    `1.53 / 1.48 ms`, streaming `36.38 / 36.01 ms` (`27.49 fps`);
  - repeat: dynamics `25.12 / 25.03 ms`, decoder `9.22 / 9.21 ms`, cache commit
    `1.62 / 1.56 ms`, streaming `36.00 / 35.83 ms` (`27.78 fps`).
  Output validation again reported `6` unique frame hashes and `6` unique latent hashes. WASM demo
  smoke passed both `starts and renders a frame @demo` and `changes the display over generated
  frames @demo` with `?backend=wasm&assetBase=/dream_arcade_assets/breakout_wasm_default_mha`.
  After the rejected follow-up trials, the accepted `extended` restore run measured dynamics
  `25.25 / 25.08 ms`, decoder `9.25 / 9.24 ms`, cache commit `1.53 / 1.48 ms`, and streaming
  `36.06 / 35.84 ms` (`27.73 fps`).
- Retested runtime `graphOptimizationLevel=basic` on the newly smaller entry graph. Browser output
  validation passed but it regressed to dynamics `26.61 / 25.97 ms`, decoder `9.24 / 9.23 ms`,
  cache commit `1.53 / 1.48 ms`, and streaming `37.42 / 36.75 ms` (`26.72 fps`). Keep
  rejecting `basic`.
- Tested a copied decoder inverse-transpose-pair cleanup after the singleton-key bypass and decoder
  primitive RMSNorm promotion. The copied decoder/decode-z graphs each removed two exact inverse
  `Transpose -> Transpose` pairs (`429 -> 425` and `495 -> 491` nodes), and CPU comparison was
  exact for `patches` on both artifacts. Browser output validation passed, but timing was
  neutral/slower: dynamics `26.16 / 25.33 ms`, decoder `9.44 / 9.27 ms`, cache commit
  `1.53 / 1.48 ms`, and streaming `37.18 / 36.10 ms` (`26.90 fps`). Reject; ORT's optimizer
  already appears to handle or hide most of this small cleanup.
- Retested runtime `graphOptimizationLevel=all` after both the decoder primitive RMSNorm promotion
  and the ranked MHA query head-merge export. This is a session/runtime change only; the ONNX graphs
  and JAX/export validation envelope are unchanged. Explicit `all` browser output validation passed
  twice with `wasmNumThreads=4`:
  - first run: dynamics `25.17 / 24.89 ms`, decoder `9.21 / 9.21 ms`, cache commit
    `1.53 / 1.48 ms`, streaming `35.95 / 35.57 ms` (`27.81 fps`);
  - repeat: dynamics `25.00 / 24.88 ms`, decoder `9.22 / 9.22 ms`, cache commit
    `1.54 / 1.48 ms`, streaming `35.80 / 35.57 ms` (`27.93 fps`).
  The adjacent `extended` run passed but was slower at dynamics `25.72 / 25.05 ms`, decoder
  `9.23 / 9.22 ms`, cache commit `1.54 / 1.48 ms`, and streaming `36.53 / 35.88 ms`
  (`27.38 fps`). Promoted the WASM default in the benchmark and demo from `extended` to `all`.
  A no-override default validation confirmed the benchmark selected `graphOptimizationLevel: "all"`
  and passed browser output validation at `sample_steps=2`: dynamics `25.23 / 24.99 ms`, decoder
  `9.22 / 9.21 ms`, cache commit `1.52 / 1.48 ms`, streaming `36.01 / 35.66 ms`
  (`27.77 fps`), with `6` unique frame hashes and `6` unique latent hashes. The default WASM demo
  smoke passed both `starts and renders a frame @demo` and `changes the display over generated
  frames @demo` with `?backend=wasm&assetBase=/dream_arcade_assets/breakout_wasm_default_mha`.
- Tested ORT WASM `executionMode=parallel` as a runtime-only scheduling control on the current
  artifact. Browser output validation passed, but it regressed both sessions: dynamics
  `26.69 / 26.53 ms`, decoder `9.71 / 9.69 ms`, cache commit `1.53 / 1.48 ms`, and streaming
  `37.97 / 37.78 ms` (`26.34 fps`). The adjacent default sequential run returned to dynamics
  `25.18 / 24.78 ms`, decoder `9.25 / 9.23 ms`, cache commit `1.55 / 1.48 ms`, and streaming
  `36.01 / 35.53 ms` (`27.77 fps`). Reject; ORT's per-session parallel graph scheduler adds
  overhead for these already-threaded WASM kernels.
- Retested the current npm `onnxruntime-web@1.26.0` runtime after the later decoder RMSNorm,
  ranked head-merge, and `graphOptimizationLevel=all` changes. The 1.26.0 package was staged under
  an ignored node_modules trial path so the JS loader and `.wasm` files matched; no dependency files
  or ONNX artifacts were changed. Browser output validation passed, but timing was neutral/slower:
  dynamics `25.50 / 25.04 ms`, decoder `9.22 / 9.19 ms`, cache commit `1.53 / 1.47 ms`, and
  streaming `36.29 / 35.82 ms` (`27.55 fps`). The adjacent pinned `1.24.3` control was faster at
  dynamics `25.05 / 24.88 ms`, decoder `9.24 / 9.22 ms`, cache commit `1.53 / 1.47 ms`, and
  streaming `35.86 / 35.62 ms` (`27.89 fps`). Reject; keep the pinned runtime for now.
- Retested the pinned `1.24.3` bundled WASM loader,
  `/node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs`, on the current accepted graph.
  Browser output validation passed, but the result stayed in the standard-loader noise band:
  dynamics `25.20 / 24.95 ms`, decoder `9.22 / 9.21 ms`, cache commit `1.53 / 1.47 ms`, and
  streaming `35.99 / 35.71 ms` (`27.78 fps`). The adjacent standard-loader control remained
  slightly faster at `35.86 / 35.62 ms` (`27.89 fps`). Reject; keep the smaller standard WASM
  loader.
- Retested the pinned `1.24.3` JSPI loader,
  `/node_modules/onnxruntime-web/dist/ort.jspi.min.mjs`, on the same graph. Browser output
  validation passed, but timing was slower than the accepted standard loader: dynamics
  `25.30 / 25.23 ms`, decoder `9.28 / 9.28 ms`, cache commit `1.53 / 1.48 ms`, and streaming
  `36.15 / 36.01 ms` (`27.66 fps`). Reject; JSPI call plumbing is not a win for this current
  threaded WASM path.
- Tested the current npm dev tag, `onnxruntime-web@1.27.0-dev.20260506-673c3320fc`, with matching
  JS and `.wasm` files staged under an ignored trial path. Browser output validation passed twice:
  first at dynamics `25.30 / 25.19 ms`, decoder `9.23 / 9.22 ms`, cache commit
  `1.54 / 1.48 ms`, streaming `36.11 / 35.93 ms` (`27.69 fps`), and repeat at dynamics
  `25.25 / 25.09 ms`, decoder `9.23 / 9.23 ms`, cache commit `1.53 / 1.48 ms`, streaming
  `36.06 / 35.90 ms` (`27.73 fps`). Same-window pinned controls were noisy
  (`36.82 / 36.48 ms` then `36.27 / 35.98 ms`), while the earlier accepted pinned window was
  faster at `35.86 / 35.62 ms`. Reject; the prerelease runtime is not a stable/material win and
  still does not approach `30 fps`.
- Tested ORT WASM allocation session options on the accepted pinned runtime. All variants passed
  browser output validation at `sample_steps=2`, but none produced a full-frame win:
  - `enableCpuMemArena=true`: dynamics `25.13 / 24.95 ms`, decoder `9.20 / 9.19 ms`, cache commit
    `1.53 / 1.48 ms`, streaming `35.89 / 35.66 ms` (`27.87 fps`);
  - `enableMemPattern=true`: dynamics `25.41 / 25.05 ms`, decoder `9.07 / 9.07 ms`, cache commit
    `1.54 / 1.48 ms`, streaming `36.05 / 35.68 ms` (`27.74 fps`);
  - both enabled: dynamics `25.54 / 25.30 ms`, decoder `9.08 / 9.07 ms`, cache commit
    `1.53 / 1.48 ms`, streaming `36.19 / 35.90 ms` (`27.63 fps`).
  Reject; memory pattern can help decoder slightly, but the dynamics regression and overall frame
  noise leave the default session options better.
- After removing the temporary runtime hooks, the accepted no-override WASM benchmark passed output
  validation again with standard `ort.wasm.min.mjs`, `wasmNumThreads=4`, and default
  `graphOptimizationLevel=all`: dynamics `25.21 / 24.90 ms`, decoder `9.23 / 9.23 ms`, cache
  commit `1.54 / 1.49 ms`, and streaming `36.02 / 35.62 ms` (`27.76 fps`), with `6` unique frame
  hashes and `6` unique latent hashes.
- Accepted a runtime-only WASM cache scheduling change. The benchmark and demo now use a Web Worker
  for the entry-cache slide/rebase on the WASM backend, copying the small K/V entry tensors while
  transferring the persistent K/V cache buffers, and start that async update before the decoder
  runs. This does not change the ONNX graphs, so the existing export/JAX validation envelope for the
  promoted s2 artifact still applies.
  - Explicit worker-cache benchmark runs passed browser output validation with `6` unique frame
    hashes and `6` unique latent hashes. Timings were dynamics `25.49 / 25.34 ms`, decoder
    `9.20 / 9.19 ms`, cache wait `0.241 / 0.230 ms`, streaming `35.02 / 34.89 ms`
    (`28.56 fps`), and repeat dynamics `25.66 / 25.50 ms`, decoder `9.25 / 9.26 ms`, cache wait
    `0.235 / 0.225 ms`, streaming `35.24 / 35.05 ms` (`28.38 fps`).
  - The adjacent synchronous-cache default measured dynamics `25.27 / 24.95 ms`, decoder
    `9.24 / 9.23 ms`, cache commit `1.54 / 1.49 ms`, and streaming `36.08 / 35.65 ms`
    (`27.72 fps`), so the worker path is a real cache-overlap win even though dynamics and decoder
    remain the dominant costs.
  - Promoted the WASM benchmark default to `workerCacheUpdate: true`. A no-override default run
    confirmed the default and passed output validation: dynamics `25.91 / 25.44 ms`, decoder
    `9.31 / 9.27 ms`, cache wait `0.236 / 0.225 ms`, and streaming `35.55 / 34.93 ms`
    (`28.13 fps`).
  - Promoted the same worker updater into the live demo for WASM, with cleanup on failed runtime
    construction. WASM demo smoke passed `starts and renders a frame @demo`,
    `changes the display over generated frames @demo`, and a new cache lifecycle test that fills
    from the initial artifact cache to the full context length and then keeps the cache full on the
    full-cache slide path, all with
    `?backend=wasm&assetBase=/dream_arcade_assets/breakout_wasm_default_mha`.
- Fixed the benchmark's `provider=wasm` defaults to match the accepted WASM/demo runtime:
  `ortModule=/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs` and `wasmNumThreads=4` now apply
  when the caller does not pass explicit overrides. This is a runtime harness/default fix only; it
  does not change ONNX graphs or the JAX validation envelope, and it preserves `sample_steps=2`.
  A no-override WASM benchmark then confirmed the defaults in the reported config and passed output
  validation with `6` unique frame hashes and `6` unique latent hashes: dynamics
  `26.01 / 25.62 ms`, decoder `9.26 / 9.25 ms`, cache wait `0.246 / 0.237 ms`, and streaming
  `35.63 / 35.11 ms` (`28.06 fps`). The same run without those explicit settings had left
  `wasmNumThreads` unset and loaded the WebGPU bundle, measuring only `37.82 / 37.36 ms`
  (`26.44 fps`), so the default fix avoids a misleading slow WASM benchmark path.
- Rejected a copied decoder `SkipSimplifiedLayerNormalization` primitive-lowering trial. The trial
  replaced `12` decoder skip-SLN nodes with equivalent `Add/Mul/ReduceMean/Sqrt/Div/Mul`
  arithmetic (`429 -> 501` decoder nodes). CPU comparison against the accepted decoder was valid
  (`patches` max/mean abs `9.536743e-7 / 3.998058e-8`), and browser output validation passed, but
  decoder timing regressed to `9.66 / 9.64 ms` and streaming to `38.34 / 37.71 ms`
  (`26.08 fps`). The adjacent accepted-default control was faster at decoder
  `9.40 / 9.37 ms`, streaming `37.82 / 37.36 ms` (`26.44 fps`) under the same temporarily
  unset-thread benchmark defaults. The trial asset was removed.
- Accepted a runtime-only decoder worker pipeline for pure WASM. The dynamics step still runs on the
  main ORT WASM instance with `wasmNumThreads=4`, while the decoder runs in a separate Web Worker
  with its own ORT WASM instance. The generated latent is copied to the decoder worker, the cache
  worker update is awaited immediately so the next dynamics frame can start, and decoded patches are
  displayed one frame later. This changes scheduling only; the ONNX graphs and the existing
  JAX/export validation envelope are unchanged, and browser output validation hashes are computed
  from the actual worker-decoded frames.
  - One decoder worker thread crossed the target: dynamics `26.22 / 25.40 ms`, worker decoder
    `30.43 / 30.38 ms`, cache wait `1.57 / 1.40 ms`, and displayed-frame interval
    `31.08 / 30.34 ms` (`32.18 fps`).
  - Two worker threads passed output validation at `28.60 / 27.42 ms` (`34.97 fps`), with dynamics
    `26.51 / 25.74 ms`, worker decoder `16.45 / 16.31 ms`, and cache wait `1.59 / 1.44 ms`.
  - Three worker threads were best and repeatable: first run `28.43 / 27.53 ms` (`35.18 fps`),
    repeat `28.41 / 27.70 ms` (`35.21 fps`). Worker decoder was about `12.15-12.18 ms`, while
    dynamics stayed around `26.39-26.45 ms` and cache wait around `1.51-1.60 ms`.
  - Four worker threads still passed output validation but increased dynamics contention:
    `29.95 / 29.51 ms` (`33.39 fps`). Keep the decoder worker default at `3` threads.
  - Promoted the benchmark default for `provider=wasm` to `decoderWorkerPipeline: true` and
    `decoderWorkerNumThreads: 3`. The no-override WASM benchmark confirmed the config and passed
    output validation with `6` unique frame hashes and `6` unique latent hashes: dynamics
    `26.64 / 26.13 ms`, worker decoder `12.21 / 12.14 ms`, cache wait `1.55 / 1.42 ms`, and
    displayed-frame interval `28.62 / 27.75 ms` (`34.94 fps`).
  - Promoted the same one-frame decoder pipeline into the live WASM demo. Demo smoke passed
    `starts and renders a frame @demo`, `changes the display over generated frames @demo`, and the
    WASM cache lifecycle test, which now also checks that the demo has the decoder worker enabled.
    The manual/debug frame generator remains sequential so the cache lifecycle test can directly
    verify K/V fill and full-cache sliding behavior.
- Rejected a dynamics-worker ownership trial. The prototype moved the WASM dynamics session and
  K/V cache mutation into a dedicated Worker so the persistent cache buffers stayed resident in the
  same thread that runs ORT, and only action/noise inputs plus `final_z` crossed the worker
  boundary. Chrome output validation passed, but the worker path was slower than the accepted main
  dynamics plus cache-worker pipeline:
  - Dynamics worker with `wasmNumThreads=4`: `29.60 fps`, `33.78 ms/frame`,
    worker dynamics `29.97 ms`, main-observed dynamics `31.31 ms`, cache update `1.21 ms`.
  - Dynamics worker with `wasmNumThreads=3`: `28.10 fps`, `35.59 ms/frame`,
    worker dynamics `31.52 ms`, main-observed dynamics `32.87 ms`, cache update `1.20 ms`.
  - Adjacent accepted default control: `30.41 fps`, `32.88 ms/frame`,
    dynamics `28.77 ms`, cache wait `1.42 ms`.
  - Conclusion: keeping K/V ownership inside the dynamics worker removes the cache transfer shape,
    but ORT dynamics itself slows down under Worker/decoder contention enough to lose overall. The
    prototype was reverted; no runtime code remains from this trial.
- Post-revert sanity on the accepted default bundle passed output validation in both browser
  families:
  - WebKit/Safari-family WASM: `31.22 fps`, `32.03 ms/frame`, dynamics `28.18 ms`, cache wait
    `1.12 ms`, `wasmNumThreads=3`.
  - Chrome WASM: `29.14 fps`, `34.31 ms/frame`, dynamics `30.04 ms`, cache wait `1.53 ms`,
    `wasmNumThreads=4`.
- Retested lighter Chrome decoder-worker thread counts on the accepted bundle:
  - `decoderWorkerNumThreads=2`: output validation passed at `30.26 fps`, `33.05 ms/frame`,
    dynamics `28.81 ms`, cache wait `1.50 ms`.
  - `decoderWorkerNumThreads=1`: output validation passed at `30.46 fps`, `32.83 ms/frame`,
    dynamics `28.34 ms`, cache wait `1.35 ms`, but decoder wait became nonzero
    (`0.46 ms` mean).
  - Adjacent default `decoderWorkerNumThreads=3`: output validation passed at `31.53 fps`,
    `31.72 ms/frame`, dynamics `27.60 ms`, cache wait `1.52 ms`.
  - Conclusion: the current `3`-thread decoder worker remains the best default; lighter settings
    are noise-level or slower and can expose decoder wait.
- Retested Safari/WebKit main WASM thread counts on the accepted bundle:
  - `wasmNumThreads=4`: output validation passed but regressed to `28.71 fps`,
    `34.83 ms/frame`, dynamics `30.97 ms`, cache wait `1.13 ms`.
  - `wasmNumThreads=2`: output validation passed but regressed to `28.56 fps`,
    `35.01 ms/frame`, dynamics `31.10 ms`, cache wait `1.19 ms`.
  - The adjacent no-override WebKit sanity run with the current default `wasmNumThreads=3`
    remained faster at `31.22 fps`; keep the Safari/WebKit default at `3`.
- Strengthened the benchmark validation contract with WASM numerical latent checks:
  - The demo now exposes an on-demand debug toggle that records CPU `final_z` hash/finite summaries
    when validation asks for them. It is disabled during warmup/timed windows.
  - The actual-demo benchmark now fails `provider=wasm` if the sampled generated latents are static
    or non-finite, in addition to the existing visible screenshot hash and brick-band checks.
  - Chrome WASM validation passed with `3` measured latent samples, `3` unique finite hashes,
    `30.13 fps`, `33.19 ms/frame`.
  - WebKit/Safari-family WASM validation passed with `3` measured latent samples, `3` unique finite
    hashes, `31.96 fps`, `31.29 ms/frame`.
- Rejected direct import of `/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs` as the
  demo `ortModule`. The page never reached Ready; `configureOrt` failed because the direct module
  does not expose the same `ort.env.wasm` surface as `ort.wasm.min.mjs`, and the browser also
  reported a missing resource. Keep the standard `ort.wasm.min.mjs` loader.
- Fresh native ORT CPU profiling on the current accepted WASM artifacts still shows broad graph
  cost rather than one isolated bottleneck. Over three dynamics runs with native ORT CPU profiling,
  top op-family totals were `Gemm` ~`10.1 ms`, `Transpose` ~`9.5 ms`, `Gather` ~`7.9 ms`,
  `MultiHeadAttention` ~`7.8 ms`, `SimplifiedLayerNormalization` ~`7.7 ms`, `Unsqueeze`
  ~`7.0 ms`, and `Concat` ~`4.7 ms`. Decoder profiling was led by `Gemm` ~`2.7 ms`, then
  attention/pointwise/norm families around `0.4-1.2 ms` each. This matches the browser behavior:
  more single-pattern cleanup is unlikely to reach `60 fps`; a larger structural change is needed.
- Accepted a Chrome/Chromium-only split-dynamics schedule for WASM:
  - Used ONNX extraction to split the current entry-slide dynamics graph into
    `*_sample_only_final_z.onnx` and `*_context_entry_from_final_z.onnx`.
  - Native ORT comparison against the original full graph was exact for `final_z`,
    `candidate_k_entry`, and `candidate_v_entry` (`max abs diff 0.0` for all three). Native CPU
    timing was full graph `~14.0 ms`, sample-only `~9.6 ms`, and context-entry `~4.6 ms`.
  - The demo can start the decoder worker after sample-only `final_z`, then run the context-entry
    graph and cache update before advancing the next frame. This preserves `sample_steps=2` and the
    original cache semantics.
  - Export now generates the split models for WASM cached exports, and the static demo-site builder
    copies the derived files when they exist. The runtime falls back to the original full graph if
    the split files are absent or fail to compile.
  - Chrome default WASM benchmark, actual demo, 64 timed frames: output and numerical validation
    passed, `33.61 fps`, `29.76 ms/frame`, `split_wasm_dynamics=true`, `wasmNumThreads=4`,
    dynamics split into sample `18.20 ms` and entry `9.20 ms`.
  - Same-day Chrome split-disabled control, 16 timed frames: output and numerical validation passed
    at `27.71 fps`; a same-window split-enabled 16-frame run was `31.23 fps`.
  - WebKit/Safari-family split was rejected as a default. A 64-frame split-enabled run validated
    but regressed to `29.15 fps`, while the split-disabled control validated at `32.33 fps`.
    Safari/WebKit therefore keeps `split_wasm_dynamics=false` by default, with an override flag for
    future experiments.
  - WebKit/Safari-family default after the guard, 64 timed frames: output and numerical validation
    passed, `32.92 fps`, `30.37 ms/frame`, `split_wasm_dynamics=false`, `wasmNumThreads=3`.
- Rejected post-split Chromium runtime retunes:
  - Decoder worker thread counts under split remained worse than the `3`-thread default:
    `1` thread `29.86 fps`, `2` threads `29.63 fps`, `4` threads `29.32 fps`; all passed output
    and latent validation but regressed the same short window.
  - Main WASM thread counts under split also remained worse than the `4`-thread Chromium default:
    `2` threads `25.93 fps`, `3` threads `28.73 fps`, and `5` threads `26.06 fps`; all passed
    validation but were slower.
  - Releasing the original full-step session after split session compilation validated but slowed
    the short Chromium window to `28.99 fps`; keep the full-step session resident as the fallback.
  - Running ORT `ORT_ENABLE_EXTENDED` optimization on the extracted split models reduced the sample
    graph `2308 -> 2284` nodes and the entry graph `1178 -> 1166` nodes, but browser validation
    regressed the short Chromium window to `28.81 fps`; keep the plain extracted split files.
  - Forcing the WASM entry-cache update back onto the main thread under split validated but
    measured only `29.60 fps` and introduced decoder wait; keep the worker cache updater.
  - Retested WASM `executionMode: 'parallel'` on the main split sessions. It passed validation but
    regressed the short Chromium window to `27.89 fps`; keep the default ORT execution mode.
  - Retested split-session `graphOptimizationLevel` after rebuilding the browser bundle from the
    committed source: `all` remained best at `31.15 fps` on the short Chromium window, while
    `extended` measured `29.34 fps` and `basic` measured `28.63 fps`.
  - Retested WASM session memory options under split. `enableMemPattern=true` validated but
    measured `29.56 fps`; `enableCpuMemArena=true` validated but measured `29.50 fps`. Keep the
    default session memory options.
- After adding automatic wrapper rebuilds, short default validation windows still passed:
  Chromium WASM `30.24 fps` with `split_wasm_dynamics=true`; WebKit/Safari-family WASM
  `31.78 fps` with `split_wasm_dynamics=false`.
- Moved the WASM split/decoder-worker backend label update out of the per-frame split hot loop and
  into runtime setup. This removes an unnecessary DOM write from measured generation. Short
  validation windows passed afterward: Chromium WASM `31.01 fps` with split enabled; WebKit/Safari
  family `31.26 fps` with split disabled.
- Retested decoder start ordering under split. Delaying the decoder worker until after the
  context-entry graph reduced decoder contention but lost overlap; a 64-frame Chromium validation
  measured `32.90 fps`, effectively the same as the adjacent immediate-start control at
  `32.93 fps`. Keep immediate decoder start after sample `final_z`.
- Rejected folding the hot split graphs' action embedding `Gather -> Add(constant)` patterns into
  pre-added gather tables. The rewrite was exactly equivalent in native ORT for split sample
  `final_z` and split entry `candidate_k_entry` / `candidate_v_entry` (`max abs diff 0.0`), but it
  only changed native timing by noise-level amounts (`sample ~10.09 ms -> ~10.08 ms`,
  `entry ~4.79 ms -> ~4.73 ms`) and the actual Chromium demo benchmark regressed to `29.22 fps`
  despite passing output and latent validation. Restore the plain extracted split ONNX files.
- Rejected replacing the accepted temporal `MultiHeadAttention` split graphs with
  `GroupQueryAttention`. An export trial that enabled the existing GQA pass before MHA did not
  match the WASM slide-entry graph after the current layout rewrites (`0` GQA rewrites), so it fell
  back to the slower explicit MatMul attention path. A direct prototype that rewrote each accepted
  MHA back to compact-K/V GQA was CPU-exact (`max abs diff 0.0`) but regressed native ORT timing:
  split sample `~10.0 ms -> ~13.0 ms`, split entry `~4.7 ms -> ~6.1 ms`. Keep the accepted MHA
  fusion for temporal dynamics.
- Retested decoder-worker contention under split by disabling the decoder worker pipeline. Dynamics
  got faster (`26.49 ms` mean, sample `17.87 ms`, entry `8.62 ms`) because the worker no longer
  contended for CPU, but decode became sequential (`9.19 ms`) and full streaming regressed to
  `27.15 fps` / `36.83 ms`. Keep the decoder worker pipeline; even a hypothetical non-contending
  decoder would still leave the sequential dynamics/cache path well above the `16.7 ms` target.
- Rejected a SharedArrayBuffer-backed WASM cache-updater prototype. Keeping the K/V cache tensors
  stable and shared with the worker preserved output and latent validation, but did not reduce cache
  wait and regressed the short Chromium window to `29.66 fps` / `33.72 ms` with cache wait
  `1.52 ms`. Keep the transferable ArrayBuffer cache updater.
- Rejected rewriting the current split graphs' no-bias `Gemm(A, B)` nodes to `MatMul(A, B)` before
  a browser trial. The rewrite was CPU-exact (`max abs diff 0.0`) for split sample and entry, but
  native ORT timing was noise-level and mixed: sample `~10.17 ms -> ~10.09 ms`, entry
  `~4.73 ms -> ~4.77 ms`. This is too small and inconsistent to move the actual demo frame budget.
- Rejected moving next-frame noise prefill until after the async cache worker launch. The intent was
  to overlap random tensor filling with the worker cache update, but the short Chromium validation
  regressed to `29.49 fps` / `33.91 ms` and cache timing did not improve (`cache wait 1.56 ms`,
  `cache total 1.66 ms`). Keep the previous ordering.
- Rejected replacing the zero-delay `MessageChannel` stream-loop scheduler with `queueMicrotask`.
  Chrome and WebKit both passed output/latent validation, but the short windows stayed in the
  normal noise band (Chrome `30.63 fps`, WebKit `31.30 fps`) and the microtask loop risks starving
  browser paints in the live demo. Keep the `MessageChannel` scheduler for zero-delay streaming.
- Retested ORT re-optimization of the extracted split WASM models against the current generated
  assets. A controlled Chrome A/B regenerated plain split files from the current full-step graph,
  then re-ran `ORT_ENABLE_EXTENDED` on only those split files. Plain extraction passed output and
  latent validation at `34.23 fps` / `29.21 ms`; re-optimized split files also validated but
  regressed to `32.46 fps` / `30.81 ms` despite reducing the sample graph `2308 -> 2284` nodes and
  the entry graph `1178 -> 1166` nodes. Keep the plain extracted split files.

Current WASM conclusion:
- The accepted pure-WASM path is now the s2 entry-cache slide graph with temporal dynamics MHA,
  MatMul attention for the remaining explicit attention islands, the extended static head-merge
  cleanup including ranked MHA query merges, the decoder singleton-key attention bypass, the decoder
  primitive RMSNorm rewrite, the worker-backed JavaScript cache updater for WASM, the decoder
  worker pipeline with `decoderWorkerNumThreads=3`, browser-specific main WASM threads
  (`4` for Chrome/Chromium and `3` for Safari/WebKit), ORT WASM `graphOptimizationLevel=all`, and
  Chrome/Chromium-only split dynamics when the derived split files are present.
- The current validated default windows are about `33.6 fps` in headed Chrome and `32.9 fps` in
  WebKit/Safari-family, both preserving `sample_steps=2` and passing visual plus WASM latent
  validation. The `30 fps` target is reached, but the requested `60 fps` target is not.
