# ORT WebGPU Demo Benchmark

This benchmark measures the browser path needed by the interactive demo. It intentionally does not
run the exported full-window, no-cache ONNX models.

Node.js/Bun is only used to launch Playwright and serve files. ONNX Runtime WebGPU runs inside a real
Chrome browser.

## Setup

```bash
bun install
bunx playwright install chrome
```

## Commands

```bash
bun run benchmark:webgpu:smoke
bun run benchmark:webgpu
bun run benchmark:webgpu:ci
```

The default scripts launch headed Google Chrome and require a hardware WebGPU adapter. On Apple
Silicon this should report the M-series GPU in `webgpu_app/bench/results/latest.json`; it should not
report SwiftShader.

Benchmark controls are wrapper flags passed after `--`. Prefer these over leading shell environment
assignments:

```bash
bun run benchmark:webgpu -- --grep @graph-capture --webgpu-benchmark-timed-runs 64
bun run benchmark:webgpu -- --webgpu-benchmark-asset-base /webgpu_app/dream_arcade_assets/breakout
bun run benchmark:webgpu -- --webgpu-benchmark-graph-optimization-level extended
bun run benchmark:webgpu -- --webgpu-benchmark-browser-profile safari
bun run benchmark:webgpu -- --webgpu-benchmark-provider wasm --webgpu-benchmark-ort-module /node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs --webgpu-benchmark-wasm-num-threads 4
```

When opened directly in Safari, the benchmark uses the same valid dynamics path as the demo:
WebGPU dynamics without graph capture. The interactive demo also defaults Safari to the CPU canvas
presentation path so the visible frame is read back and drawn through 2D canvas instead of relying on
Safari's WebGPU canvas presentation. Safari's WebGPU backend currently reports fast
captured-dynamics timings, but captured dynamics repeats a static frame, so dynamics graph capture
remains a diagnostic rather than the Safari default.

Manual Safari URL while the static server is running:

```text
http://127.0.0.1:4173/webgpu_app/bench/index.html?browserProfile=safari
```

For a functional-only check in headless Chromium/SwiftShader:

```bash
bun run benchmark:webgpu:headless-smoke
```

## Required Demo Assets

The benchmark only runs when these cached demo artifacts are present in
`webgpu_app/dream_arcade_assets/breakout/breakout_onnx_manifest.json`:

- `breakout_dynamics_prefill_cached_b1_t64.onnx`
- `breakout_dynamics_sample_append_context_cache_length_entry_b1_t1_s2.onnx`
- `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2.onnx`
- `breakout_tokenizer_decoder_b1_t1.onnx` preferred, with
  `breakout_tokenizer_decode_z_b1_t1.onnx` as the fallback
- `breakout_demo_context.*`
- `breakout_demo_initial_cache.*`

If those artifacts are missing, the benchmark writes a structured `blocked` result instead of running
the old full-window graphs.

Create the demo artifacts with:

```bash
uv run python scripts/webgpu/export_dreamer4_onnx.py \
  --tokenizer_dir gs://visionary-exp/dream-arcade/checkpoints/breakout_tokenizer_small_2x \
  --tokenizer_step 1000000 \
  --dynamics_dir gs://visionary-exp/dream-arcade/checkpoints/breakout_dynamics_small_2x \
  --dynamics_step 1000000 \
  --out_dir webgpu_app/dream_arcade_assets/breakout \
  --seq_len 64 \
  --sample_steps 2 \
  --export_cached \
  --validate \
  --overwrite
uv run python scripts/webgpu/specialize_full_cache_entry.py \
  --asset_dir webgpu_app/dream_arcade_assets/breakout
```

## Measured Path

The benchmark models one generated demo frame as:

```text
run the full-cache entry dynamics step artifact
update the rolling K/V cache from the returned entry tensors
copy predicted z to the decoder input
decode the accepted single frame
```

The interactive demo still uses the cache-length entry artifact while filling a short prefix cache,
then switches to the packed full-cache entry artifact once the logical cache length reaches 64.

Sampling constants are recorded in the result:

```text
sample_steps = 2
sample_step_level = 1
context_step_level = 5
context_tau_effective = 29 / 32
```

## Metrics

`results/latest.json` uses `schema_version: 2` and reports:

- `cached_prefill`: context cache creation time
- `cached_step`: cached dynamics target-forward time and full dynamics frame time
- `streaming_frame`: full steady-state generated-frame time and FPS
- `streaming_frame.output_validation`: untimed hashes from generated decoder frames and the latent
  tensor passed to the decoder; `status: failed` means the timed path was producing a static/stale
  frame and its FPS is not a valid demo result

Fetch time, session creation time, warmup, and browser metadata are reported separately from
steady-state frame timing.

Safari profile runs use the valid WebGPU path without graph capture. Dynamics and decoder graph
capture are disabled there because Safari currently returns stale captured frames, and
`graphOptimizationLevel=disabled` is the fastest valid Safari setting measured so far.
On the same machine, the validated Safari path is still far behind Chrome: Safari exposes
`shader-f16` but not `subgroups`, while Chrome exposes `subgroups`; the current ORT WebGPU
transformer kernels are much slower without that feature. Treat Safari graph-capture FPS as invalid
unless `streaming_frame.output_validation.status` is `passed`.

The pure WASM bundle is a valid Safari control path and can be selected with `provider=wasm`,
`ortModule=/node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs`, and `wasmNumThreads=4`.
It is faster than one-thread WASM but still far below Chrome's WebGPU path on this model.

## Graph Capture

The regular benchmark also includes a graph-capture test case:

```bash
bun run benchmark:webgpu
```

When graph capture succeeds, `results/graph_capture_latest.json` records graph-capture
warmup-adjusted timings under the `*_after_graph_capture_warmup` fields. If ONNX Runtime rejects
graph capture because part of the graph cannot be assigned to WebGPU, the test records a structured
`blocked` result instead of failing the whole benchmark run.

## Baselines

`webgpu_app/bench/baselines/webgpu_benchmark_baseline.json` starts in warning mode. After cached
demo artifacts exist and stable results are collected on the target machine, add representative
`streaming_frame` entries from `results/latest.json` and switch `policy.mode` to `fail` when
regressions should break CI.
