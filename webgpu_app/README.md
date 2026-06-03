# WebGPU App

This directory owns the browser-side ONNX Runtime WebGPU validation workspace.
The root project remains focused on world-model and RL research code.

## Layout

- `bench/`: Playwright benchmark page, benchmark specs, baselines, and the benchmark threshold checker.
- `demo/`: Minimal browser validation page for inspecting generated frames, cache behavior, and runtime performance.
- `export/`: ONNX export, WebGPU graph rewrite, artifact comparison, and demo context/cache generation scripts.
- `scripts/`: local TypeScript build, static server, Playwright wrapper, and cleanup helpers.
- `assets/`, `assets_raw/`, `dream_arcade_assets/`, `dist/`, `bench/results/`, and generated browser bundles are local generated outputs and are ignored by git.

## Setup

Run JavaScript commands from this directory:

```sh
cd webgpu_app
bun install --frozen-lockfile
```

Python export scripts still use the root `pyproject.toml` environment and should be run from the repository root:

```sh
uv run --no-cache python webgpu_app/export/export_dreamer4_onnx.py --help
```

## Validation

Use wrapper flags after `--` for benchmark and demo controls. Do not prefix `bun run` with environment variables in this project.

```sh
cd webgpu_app
bun run typecheck
bun run build:webgpu:browser
bun run demo:webgpu:smoke
bun run benchmark:webgpu -- --grep @smoke
bun run benchmark:webgpu -- --grep @output_validation
```

For graph-capture changes, also run:

```sh
cd webgpu_app
bun run benchmark:webgpu -- --grep @graph-capture
```

The Playwright wrapper defaults to one attempt so speed and output-validation failures are not
hidden by a later faster retry. `--playwright-benchmark-attempts N` is still available for launch
failures; completed actual-demo benchmark failures are not retried.

Export-script syntax and import checks:

```sh
uv run --no-cache python -m py_compile webgpu_app/export/*.py webgpu_app/bench/check_webgpu_benchmark.py
uv run --no-cache python webgpu_app/export/export_dreamer4_onnx.py --help
uv run --no-cache python webgpu_app/export/specialize_full_cache_entry.py --help
```

Raw-vs-optimized ONNX comparison needs a raw artifact snapshot generated from the same export run as the optimized assets:

```sh
uv run --no-cache python webgpu_app/export/compare_raw_optimized_onnx.py \
  --raw_dir webgpu_app/assets_raw \
  --optimized_dir webgpu_app/assets \
  --manifest webgpu_app/assets/breakout_onnx_manifest.json
```

## Cleanup

The cleanup command removes generated browser bundles, static-site output, Playwright reports, and benchmark JSON results. It intentionally leaves model assets alone.

```sh
cd webgpu_app
bun run clean:generated
```
