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

For a functional-only check in headless Chromium/SwiftShader:

```bash
bun run benchmark:webgpu:headless-smoke
```

## Required Demo Assets

The benchmark only runs when these cached demo artifacts are present in
`webgpu_app/assets/breakout_onnx_manifest.json`:

- `breakout_dynamics_prefill_cached_b1_t64.onnx`
- `breakout_dynamics_step_cached_b1_t1.onnx`
- `breakout_tokenizer_decoder_b1_t1.onnx` or `breakout_decoder_b1_t1.onnx`

If those artifacts are missing, the benchmark writes a structured `blocked` result instead of running
the old full-window graphs.

Create the demo artifacts with:

```bash
uv run python scripts/webgpu/export_dreamer4_onnx.py \
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

## Measured Path

The benchmark models one generated demo frame as:

```text
cached dynamics target forward 1
cached dynamics target forward 2
cached dynamics target forward 3
cached dynamics target forward 4
commit only the final cache
reshape/copy predicted z for the decoder
decode the accepted single frame
```

Sampling constants are recorded in the result:

```text
sample_steps = 4
sample_step_level = 2
context_step_level = 5
context_tau_effective = 29 / 32
```

## Metrics

`results/latest.json` uses `schema_version: 2` and reports:

- `cached_prefill`: context cache creation time
- `cached_step`: cached dynamics target-forward time and four-forward frame time
- `streaming_frame`: full steady-state generated-frame time and FPS

Fetch time, session creation time, warmup, and browser metadata are reported separately from
steady-state frame timing.

## Baselines

`webgpu_app/bench/baselines/webgpu_benchmark_baseline.json` starts in warning mode. After cached
demo artifacts exist and stable results are collected on the target machine, add representative
`streaming_frame` entries from `results/latest.json` and switch `policy.mode` to `fail` when
regressions should break CI.
