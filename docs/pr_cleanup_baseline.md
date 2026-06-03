# PR Cleanup Baseline

Recorded before unused-code cleanup on `2026-06-03 19:00:20 KST`.

- Branch: `webgpu-perf`
- Commit: `dd5fcbc`
- Worktree: clean before benchmark
- Runtime path: actual demo stream, `backend=wasm`, full head-time dynamics, full 64-frame cache,
  decoder worker pipeline, `sample_steps=2`
- Dynamics artifact:
  `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s2_full_headtime`
- ORT module: `/node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs`

## Chrome

Command:

```bash
cd webgpu_app
bun run benchmark:wasm:chrome
```

Result from `webgpu_app/bench/results/latest_chromium.json`:

- Status: passed
- Steady-state FPS: `45.46229472246595`
- Steady-state frame time: `21.99624999364217 ms`
- Visual validation: passed
- Numerical latent validation: passed
- Speed gate: passed, minimum `45 fps`
- Runtime defaults:
  - `wasm_num_threads=3`
  - `decoder_worker_num_threads=3`
  - `graph_optimization_level=all`
  - decoder artifact: `breakout_tokenizer_decoder_b1_t1`
- Mean stages:
  - dynamics: `20.42130207022031 ms`
  - cache update: `1.2856770555178325 ms`
  - decoder wait: `0.15390626589457193 ms`
  - decoder total: `23.54432295759519 ms`

## Native Safari

Command:

```bash
cd webgpu_app
bun run benchmark:wasm:safari
```

Two back-to-back runs were made because native Safari is known to vary around the current `45 fps`
gate. Both runs produced valid output and valid latent hashes, but both missed the speed gate.

Attempt 1:

- Steady-state FPS: `44.75`
- Steady-state frame time: `22.35 ms`
- Visual validation: passed
- Numerical latent validation: passed
- Speed gate: failed, minimum `45 fps`

Attempt 2, current `latest_safari.json`:

- Steady-state FPS: `43.375715021552296`
- Steady-state frame time: `23.054375000000007 ms`
- Visual validation: passed
- Numerical latent validation: passed
- Speed gate: failed, minimum `45 fps`

Use this as the cleanup comparison point: refactors should preserve visual/numerical validity and
avoid making the Chrome path slower. Safari should be compared against the current noisy validated
range unless the cleanup touches a Safari-specific hot path.
