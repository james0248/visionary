# ORT WebGPU Demo Benchmark

This benchmark drives the interactive demo page itself. It clicks the same Start/Pause controls,
measures generated frames from `window.visionaryDemoDebug.frameStats`, and validates screenshots
from the visible frame surface. It intentionally does not keep a separate benchmark-only ONNX
runtime path.

Node.js/Bun is only used to launch Playwright and serve files. ONNX Runtime runs inside the browser
that Playwright launches.

## Setup

```bash
cd webgpu_app
bun install --frozen-lockfile
bunx playwright install chrome
```

## Commands

```bash
bun run benchmark:webgpu:smoke
bun run benchmark:webgpu
bun run benchmark:webgpu:ci
```

The default scripts launch headed Google Chrome and require a hardware WebGPU adapter. On Apple
Silicon this should report the M-series GPU in `bench/results/latest.json`; it should not
report SwiftShader.

Benchmark controls are wrapper flags passed after `--`. Prefer these over leading shell environment
assignments:

```bash
bun run benchmark:webgpu -- --grep @graph-capture --webgpu-benchmark-timed-runs 64
bun run benchmark:webgpu -- --webgpu-benchmark-asset-base /dream_arcade_assets/breakout
bun run benchmark:webgpu -- --webgpu-benchmark-graph-optimization-level extended
bun run benchmark:webgpu -- --webgpu-benchmark-browser-profile safari
bun run benchmark:webgpu -- --webgpu-benchmark-provider wasm
```

For a Safari-family automation check of the WASM path, run the same benchmark under the WebKit
project:

```bash
bun scripts/run_playwright_chrome_home.ts test bench/run_webgpu_benchmark.spec.ts --project=webkit --grep @output_validation --webgpu-benchmark-provider wasm
```

For a functional-only check in headless Chromium/SwiftShader:

```bash
bun run benchmark:webgpu:headless-smoke
```

## Required Demo Assets

The benchmark only runs when these cached demo artifacts are present in
`webgpu_app/dream_arcade_assets/breakout/breakout_onnx_manifest.json`:

- `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2.onnx`
- `breakout_dynamics_sample_append_context_full_cache_entry_b1_t1_s2.onnx`
- `breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2_final_z_add_zero_safari_trial.onnx`
- `breakout_tokenizer_decoder_b1_t1.onnx` preferred, with
  `breakout_tokenizer_decode_z_b1_t1.onnx` as the fallback
- `breakout_demo_context_noop60_fire4.*`
- `breakout_demo_initial_cache_noop60_fire4.*`

If those artifacts are missing, the demo fails to reach `Ready` and the benchmark fails with the
page status and recent browser diagnostics.

Create the demo artifacts from the repository root with:

```bash
uv run python webgpu_app/export/export_dreamer4_onnx.py \
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
uv run python webgpu_app/export/specialize_full_cache_entry.py \
  --asset_dir webgpu_app/dream_arcade_assets/breakout
```

Create WASM-specific artifacts in a separate directory so backend-specific ONNX rewrites do not
overwrite the WebGPU artifacts:

```bash
uv run python webgpu_app/export/export_dreamer4_onnx.py \
  --tokenizer_dir gs://visionary-exp/dream-arcade/checkpoints/breakout_tokenizer_small_2x \
  --tokenizer_step 1000000 \
  --dynamics_dir gs://visionary-exp/dream-arcade/checkpoints/breakout_dynamics_small_2x \
  --dynamics_step 1000000 \
  --out_dir webgpu_app/dream_arcade_assets/breakout_wasm \
  --export_target wasm \
  --seq_len 64 \
  --sample_steps 2 \
  --export_cached \
  --validate \
  --overwrite
```

## Measured Path

The benchmark measures the demo's actual stream loop:

```text
click Start
let the demo run warmup frames
reset to the initial full-cache artifact
click Start again
time the generated-frame stream until the target frame count is reached
pause and write results/latest.json
```

The interactive demo starts from a full offline cache and uses the full-cache entry artifact for
every generated frame.

Sampling constants are recorded in the result:

```text
sample_steps = 2
sample_step_level = 1
context_step_level = 5
context_tau_effective = 29 / 32
```

## Metrics

`results/latest.json` uses `schema_version: 3` and reports:

- `benchmark_kind: actual_demo_stream`
- `streaming_frame.timing`: generated-frame latency, generated-frame intervals, measured window
  FPS, and warmup-window timing from the real demo stream loop
- `streaming_frame.output_validation`: screenshot hashes from visible generated frames plus a loose
  Breakout brick-band coverage check; `status: failed` means the measured FPS is not a valid demo
  result
- `demo.initial` and `demo.final`: backend, graph-capture state, decoder-worker state, cache length,
  selected ONNX exports, and sample-step metadata observed from the demo runtime

Fetch time, session creation time, warmup, and browser metadata are reported separately from
steady-state frame timing.

Safari-profile and WebKit runs are valid only when `streaming_frame.output_validation.status` is
`passed`. The benchmark records the selected graph-capture state in `demo.final`, so a fast number
without visible-frame validation should be treated as invalid.

The pure WASM path can be selected with `provider=wasm`; the demo then defaults to
`ortModule=/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs`, `wasmNumThreads=4`, and the
decoder worker pipeline with `decoderWorkerNumThreads=3`.
The current actual-demo WASM baseline is far below the 60 fps target: about `24.2 fps` in Chrome
and `22.2 fps` in WebKit on the local machine with the un-packed full-cache entry graph. Keep
validating `wasmNumThreads` and decoder worker settings per browser and machine.

## Graph Capture

The regular benchmark also includes a graph-capture test case:

```bash
bun run benchmark:webgpu
```

When graph capture succeeds, `results/graph_capture_latest.json` records the same schema v3 actual
demo result and the observed capture state under `demo.final`. If graph capture produces stale or
broken visible frames, `streaming_frame.output_validation.status` fails and the test fails.

## Baselines

`bench/baselines/webgpu_benchmark_baseline.json` starts in warning mode. After cached
demo artifacts exist and stable results are collected on the target machine, add representative
`streaming_frame` entries from `results/latest.json` and switch `policy.mode` to `fail` when
regressions should break CI.
