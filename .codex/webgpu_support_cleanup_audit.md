# WebGPU Support Cleanup Audit

Scope audited: `webgpu_app/**`, `scripts/webgpu/*.mjs`, and `package.json` scripts.

This audit is based on the current working tree. I did not edit the scoped implementation files. The only branch-added report artifact from this pass is this file.

## Branch-Added Scoped Inventory

Evidence: `git diff --name-status main...HEAD -- webgpu_app scripts/webgpu package.json`

- `package.json` is branch-added and defines the WebGPU package scripts.
- Branch-added `scripts/webgpu/*.mjs`: `serve_static.mjs`, `summarize_ort_session_profile.mjs`, `summarize_webgpu_profile.mjs`.
- Branch-added app/demo/benchmark files include the WebGPU bench pages, Playwright specs, demo page, demo JS/CSS, and `webgpu_app/bench/results/.gitkeep`.
- Ignored/generated runtime artifacts are present in the workspace but are not branch-added/tracked: `webgpu_app/assets/`, `webgpu_app/assets_raw/`, `scripts/webgpu/__pycache__/`, and generated JSON files under `webgpu_app/bench/results/`.

Evidence for ignored/generated paths:

- `.gitignore:38-44` ignores generated ONNX/assets/results: `webgpu_app/assets/*.onnx`, `webgpu_app/assets/breakout_onnx_manifest.json`, `webgpu_app/assets/breakout_demo_context.*`, `webgpu_app/assets_raw/`, and `webgpu_app/bench/results/*.json`.
- `git status --short --ignored webgpu_app/bench/results scripts/webgpu/__pycache__ webgpu_app/assets webgpu_app/assets_raw` reports those generated paths with `!!`.

## Reference Coverage

I did not find an obviously orphaned branch-added `.mjs`, page JS, or Playwright spec entry point.

Evidence:

- `package.json:4-14` wires the benchmark, profiling, demo server, and demo smoke scripts.
- `playwright.config.js:34-35` also references `scripts/webgpu/serve_static.mjs` as the Playwright web server.
- `webgpu_app/bench/index.html:9-10` imports `./benchmark.js`.
- `webgpu_app/bench/profile_diagnostic.html:9-10` imports `./profile_diagnostic.js`.
- `webgpu_app/demo/index.html:7` imports `./styles.css`; `webgpu_app/demo/index.html:57` imports `./main.js`.
- `webgpu_app/demo/main.js:2` imports `./jax_noise.js`.
- `webgpu_app/bench/run_webgpu_benchmark.spec.js:53-60` loads the benchmark page and waits for `window.__WEBGPU_BENCHMARK_RESULT__`.
- `webgpu_app/bench/run_webgpu_profile_diagnostic.spec.js:42-49` loads the diagnostic page and waits for `window.__WEBGPU_PROFILE_DIAGNOSTIC_RESULT__`.
- `webgpu_app/demo/run_demo_smoke.spec.js:4-12` loads the demo and verifies a rendered frame.
- `package.json:10-11` references both summary `.mjs` scripts, and `package.json:13` references `serve_static.mjs`.

Search evidence: `rg -n "serve_static|summarize_webgpu_profile|summarize_ort_session_profile|profile_diagnostic|run_webgpu_benchmark|run_demo_smoke|demo:webgpu|benchmark:webgpu" .`

## Findings

### 1. `webgpu_app/bench/results/.gitkeep` looks unnecessary

Confidence: high.

`webgpu_app/bench/results/.gitkeep` is the only tracked file under `webgpu_app/bench/results`. The result writers already create the directory before writing generated JSON:

- `webgpu_app/bench/run_webgpu_benchmark.spec.js:73-75` calls `mkdir(RESULT_DIR, { recursive: true })` before writing `latest.json`.
- `webgpu_app/bench/run_webgpu_profile_diagnostic.spec.js:58-60` calls `mkdir(RESULT_DIR, { recursive: true })` before writing diagnostic JSON.
- `rg -n "\.gitkeep|RESULT_DIR|mkdir\(RESULT_DIR|results/\.gitkeep" webgpu_app/bench scripts/webgpu/*.mjs package.json playwright.config.js` finds no `.gitkeep` consumer.

Cleanup option: remove `webgpu_app/bench/results/.gitkeep` and allow the specs to create the generated-output directory on demand.

Tradeoff: keeping it is harmless if the project wants empty output directories visible in fresh clones, but it is not required by the current code paths.

### 2. Demo runtime depends on an ignored generated context file that is not documented by the WebGPU README or package scripts

Confidence: high.

The demo hard-requires `webgpu_app/assets/breakout_demo_context.json`:

- `webgpu_app/demo/main.js:4-6` defines `CONTEXT_URL = /webgpu_app/assets/breakout_demo_context.json`.
- `webgpu_app/demo/main.js:337-340` fetches that context JSON during `loadRuntime()`.
- `webgpu_app/demo/run_demo_smoke.spec.js:4-12` expects the page to reach `Ready` and render a frame.

But the file is generated/ignored and not currently present in this workspace:

- `test -e webgpu_app/assets/breakout_demo_context.json` returned exit code `1`.
- `find webgpu_app/assets -maxdepth 1 -type f -name '*demo*' -print` returned no files.
- `.gitignore:42` explicitly ignores `webgpu_app/assets/breakout_demo_context.*`.
- `scripts/webgpu/create_demo_context.py:29` has a default output name of `breakout_demo_context`, but `rg -n "breakout_demo_context|create_demo_context|demo:webgpu|demo smoke|run_demo" webgpu_app scripts package.json README.md pyproject.toml playwright.config.js` only found the demo reference, the script default, and the package demo scripts. I found no package script or WebGPU README command that generates the required context.

Cleanup/refactor option: add a documented generation path for the demo context, or make the demo/test fail earlier with a clear "run create_demo_context.py" message. If the context is intended to be checked in, the ignore rule would need to change.

### 3. Demo, benchmark, and README drift on the accepted decoder artifact names

Confidence: medium-high.

The benchmark accepts multiple decoder export names:

- `webgpu_app/bench/benchmark.js:26-38` defines `REQUIRED_ARTIFACTS.decoder` as `breakout_tokenizer_decode_z_b1_t1`, `breakout_tokenizer_decoder_b1_t1`, or `breakout_decoder_b1_t1`.

The demo accepts only one decoder export:

- `webgpu_app/demo/main.js:347` calls `findExport(manifest, 'breakout_tokenizer_decode_z_b1_t1')`.

The README lists a different required decoder set and omits `breakout_tokenizer_decode_z_b1_t1`:

- `webgpu_app/bench/README.md:36-41` says the benchmark requires `breakout_tokenizer_decoder_b1_t1.onnx` or `breakout_decoder_b1_t1.onnx`.

Impact: a manifest can satisfy the benchmark/README expectation while the demo still fails at startup, or vice versa. This is a small supportability issue rather than a dead file.

Cleanup/refactor option: centralize the demo artifact-name contract or align the demo with the benchmark's fallback list using `findFirstExport()` for the decoder as well.

### 4. Browser helper code is duplicated across benchmark, profile diagnostic, and demo files

Confidence: medium.

Repeated helper families appear in multiple branch-added browser files:

- Float16 conversion appears in `webgpu_app/bench/benchmark.js:78-168`, `webgpu_app/bench/profile_diagnostic.js:86-132`, and `webgpu_app/demo/main.js:69-128`.
- PRNG/tensor feed helpers appear in both `webgpu_app/bench/benchmark.js:70-157` and `webgpu_app/bench/profile_diagnostic.js:78-151`.
- WebGPU adapter checks appear in both `webgpu_app/bench/benchmark.js:347-387` and `webgpu_app/bench/profile_diagnostic.js:179-210`.
- External-data mapping appears in both `webgpu_app/bench/benchmark.js:613-618` and `webgpu_app/bench/profile_diagnostic.js:164-169`.

The duplication is substantial enough to make subtle fixes easy to miss. For example, the demo has `isNativeFloat16Array()` handling at `webgpu_app/demo/main.js:109-117`, while the benchmark always routes float16 reads through `float16BitsToFloat32()` at `webgpu_app/bench/benchmark.js:138-142`.

Cleanup/refactor option: add one small browser module, for example `webgpu_app/shared/tensors.js` and optionally `webgpu_app/shared/webgpu.js`, then import it from the bench/demo pages. Keep it browser-native and dependency-free so static serving remains simple.

### 5. WebGPU server command/port is duplicated between package scripts and Playwright config

Confidence: medium.

The same static server command appears in two places:

- `package.json:13`: `node scripts/webgpu/serve_static.mjs --host 127.0.0.1 --port 4173`
- `playwright.config.js:34-35`: the same command and health URL.

Impact: changing the host/port/server script requires updating multiple files. This is minor, but it is exactly the kind of branch-local support glue that tends to drift.

Cleanup/refactor option: keep `serve_static.mjs` as the single server implementation and consider a package script alias used by humans, while Playwright keeps its direct command for reliability. At minimum, define the port once in the Playwright config if more WebGPU scripts are added.

### 6. Package benchmark scripts repeat a long Playwright invocation

Confidence: low-medium.

`package.json:4-12` repeats `playwright test webgpu_app/bench/run_webgpu_benchmark.spec.js --project=chromium` across the default, CI, headless smoke, profiling, and smoke scripts. This is not broken, but it makes script drift more likely.

Cleanup/refactor option: if this grows further, introduce a narrower helper command or a Playwright grep/project convention. I would not prioritize this over the artifact-contract and generated-context issues.

## Suggested Order

1. Decide whether `webgpu_app/bench/results/.gitkeep` is worth keeping as a visible output-directory placeholder. It is not required by current writers.
2. Add/document the demo context generation path before relying on `demo:webgpu:smoke`.
3. Align decoder artifact selection across demo, benchmark, and README.
4. Extract duplicated browser tensor/WebGPU helpers only after the artifact contract is stable.
