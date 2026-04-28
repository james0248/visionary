# ONNX export research for the interactive world-model demo

Date: 2026-04-28

## Goal

Convert the trained Dreamer4-style tokenizer and dynamics exports to ONNX files that can run in `onnxruntime-web`, preferably with the WebGPU execution provider, without changing the training path or core model definitions. Any model-call changes should be inference-only wrappers or helper functions.

## Current repo facts

- Runtime stack in this workspace: JAX `0.9.1`, Flax `0.12.5`.
- Tokenizer model: `visionary/tokenizer.py`.
  - `Tokenizer.encode(batch)` consumes patchified `batch["video"]` shaped like `(B, T, patch_count, patch_dim)` with `uint8`/float input scaled by `/ 255`.
  - Breakout config gives `resize_shape=[128, 96]`, `patch_size=8`, `pad_width=[0, 0]`, so `y_len=16`, `x_len=12`, `patch_count=192`, `patch_dim=192`.
  - Output latent is `(B, T, num_latents=64, channel_dim=16)`.
  - `Tokenizer.decode(latent)` maps `(B, T, 64, 16)` back to `(B, T, 192, 192)` sigmoid patch values.
- Dynamics model: `visionary/dynamics.py`.
  - Main deterministic forward is `DynamicsModel.__call__(z, actions, step_levels, signal_levels)`.
  - Breakout config gives `num_obs_tokens=32`, `num_actions=4`, `num_registers=2`, `context_length=64`, `num_layers=24`, `model_dim=256`.
  - It expects `z` shaped `(B, T, 32, token_dim)`. Because tokenizer latents are `(B, T, 64, 16)`, the training code rearranges that to `(B, T, 32, 32)`.
  - Output shape matches `z`: `(B, T, 32, 32)`.
- Checkpoint exports already exist through `visionary/common/checkpoint.py`.
  - `restore_model_export_single_device(directory, step=None)` returns `(config, variables)` from local or `gs://...` export directories.
  - Training scripts save model exports via `save_model_export(..., cfg.tokenizer/cfg.dynamics, state.params)`.

## Current converter landscape

The best current path is `jax2onnx`.

- PyPI lists `jax2onnx 0.13.0`, released 2026-04-22, Python `>=3.11`, and says it converts JAX, Flax NNX, Flax Linen, and Equinox functions directly to ONNX.
- Its docs expose `to_onnx(fn, inputs=..., return_mode="file", output_path=...)`, with support for concrete arrays, `ShapeDtypeStruct`, shape tuples, symbolic dimensions like `"B"`, `input_names`, `output_names`, and `allclose(...)` validation against ONNX Runtime.
- Its Flax coverage page reports no missing dedicated Flax neural coverage, and specifically covers Embedding/RMSNorm equivalents. JAX LAX coverage reports direct or indirect coverage for core primitives we use heavily: `dot_general`, `gather`, `reshape`, `concatenate`, `broadcast_in_dim`, `dynamic_slice`, `dynamic_update_slice`, `select`, `scan`, and `fori_loop`.
- Still validate the actual Visionary graph. `visionary/transformer.py` calls `jax.nn.dot_product_attention`; depending on JAX lowering, this can become supported matmul/softmax primitives or an unsupported attention primitive. First script should export and run ORT parity tests, not just write files.

## ONNX Runtime Web constraints

- Use `onnxruntime-web` for browser inference.
- To use WebGPU, import `onnxruntime-web/webgpu` and create sessions with `executionProviders: ["webgpu"]`.
- ONNX Runtime Web docs say WASM supports all ONNX operators, but WebGPU/WebGL/WebNN support only a subset. Therefore, WebGPU viability needs browser-side or ORT-Web-node smoke tests; Python ORT parity alone is not enough.
- Static shapes are preferred. ORT WebGPU graph capture may improve performance when shapes are static and all kernels run on WebGPU.
- Keep repeated rollout tensors on GPU where possible. ORT Web supports GPU tensors via `Tensor.fromGpuBuffer`, preallocated output tensors, and `preferredOutputLocation: "gpu-buffer"`. This matters because the dynamics model will be called repeatedly during rollout.
- Browser support from ORT docs: WebGPU is supported in current Chromium/Edge desktop and Android Chromium; Safari/iOS is not supported for WebGPU. WASM fallback should remain available for compatibility, but likely too slow for full rollout.

## Recommended exported model boundaries

Export three ONNX files first:

1. `tokenizer_encoder.onnx`
   - Input: `video_patches`, likely `float32` or `uint8` converted to float before ONNX.
   - Shape: fixed demo shape, e.g. `(1, T_ctx, 192, 192)`.
   - Output: `latents` `(1, T_ctx, 64, 16)`.
   - For an interactive Breakout demo, encoding live frames may be less performance-critical than dynamics generation.

2. `tokenizer_decoder.onnx`
   - Input: `latents` `(1, T_decode, 64, 16)` or possibly one-frame `(1, 1, 64, 16)`.
   - Output: reconstructed patches `(1, T_decode, 192, 192)`.
   - Prefer a one-frame decoder for rollout display to reduce repeated work and simplify canvas upload.

3. `dynamics_step.onnx`
   - Input:
     - `z`: `(1, context_length_or_demo_window, 32, 32)`.
     - `actions`: `(1, context_length_or_demo_window)`, `int32`, with `-1` allowed for unknown/no-op slots if needed.
     - `step_levels`: `(1, context_length_or_demo_window)`, `int32`.
     - `signal_levels`: `(1, context_length_or_demo_window)`, `int32`.
   - Output: predicted `z` for the full sequence `(1, T, 32, 32)`, with the JS loop selecting/updating the target index.
   - Avoid exporting `generate_next` or `generate_rollout` initially. They include Python `math`, static `sample_steps`, dynamic index updates, and loops that are easier to express and tune in JS around a static ONNX forward.

## Inference-only wrappers to add

Create a new module such as `visionary/onnx_inference.py` with deterministic wrappers:

- `apply_tokenizer_encoder(params, cfg, video_patches) -> latents`
- `apply_tokenizer_decoder(params, cfg, latents) -> patches`
- `apply_dynamics_step(params, cfg, z, actions, step_levels, signal_levels) -> z_pred`

These wrappers should:

- Instantiate the existing Flax modules from config.
- Call `model.apply({"params": params}, ...)`.
- Avoid RNG, dropout, training losses, checkpoint managers, dataset iterators, WandB, and any mutation.
- Use stable input/output names.
- Prefer fixed shapes for the first web demo build.
- Optionally cast variables and/or activation dtype to `float32` for broad ONNX compatibility, then test whether `float16` export is viable for WebGPU.

## Export script plan

Create `scripts/webgpu/export_dreamer4_onnx.py`:

- CLI arguments:
  - `--tokenizer_dir URL_OR_PATH`
  - `--dynamics_dir URL_OR_PATH`
  - `--tokenizer_step optional int`
  - `--dynamics_step optional int`
  - `--out_dir webgpu_app/assets`
  - `--batch_size 1`
  - `--encoder_frames 16`
  - `--decoder_frames 1`
  - `--dynamics_context 17` or another static demo window
  - `--validate`
- Load exports with `restore_model_export_single_device`.
- Build dummy inputs from restored config shapes.
- Export with `jax2onnx.to_onnx(..., return_mode="file", output_path=...)`.
- Write `manifest.json` with model filenames, fixed shapes, patch/grid metadata, tokenizer/dynamics export steps, and config subset needed by JS.
- If `--validate`, run:
  - JAX wrapper output on deterministic dummy inputs.
  - Python ONNX Runtime output on the same inputs.
  - Shape and numeric tolerance checks.
  - Record results in `manifest.json` or a separate `validation.json`.

## Performance notes

- Full parameter size is substantial:
  - Tokenizer: 8 transformer layers for encoder plus 8 for decoder.
  - Dynamics: 24 transformer layers.
  - Rough size likely hundreds of MB in fp32 across all three files. Browser demo may require fp16/quantization or smaller checkpoints.
- First viable demo should use:
  - batch size 1,
  - fixed sequence/window length,
  - one-frame decoder,
  - a small sample step count,
  - WebGPU as primary execution provider,
  - WASM only as a correctness fallback.
- After baseline export works, consider:
  - ONNX external data only if model exceeds browser-loading constraints, but a single `.onnx` is simpler for static hosting.
  - ORT format / optimized model if supported in the target web packaging path.
  - fp16 conversion and WebGPU testing.
  - browser-side profiling with ORT Web performance tools and graph capture.

## Sources checked

- `jax2onnx` PyPI: https://pypi.org/project/jax2onnx/
- `jax2onnx` docs home: https://enpasos.github.io/jax2onnx/
- `jax2onnx` API reference: https://enpasos.github.io/jax2onnx/user_guide/api/
- `jax2onnx` Flax API coverage: https://enpasos.github.io/jax2onnx/user_guide/flax_api_coverage/
- `jax2onnx` JAX LAX coverage: https://enpasos.github.io/jax2onnx/user_guide/jax_lax_coverage/
- ONNX Runtime Web getting started: https://onnxruntime.ai/docs/get-started/with-javascript/web.html
- ONNX Runtime Web overview: https://onnxruntime.ai/docs/tutorials/web/
- ONNX Runtime WebGPU EP: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
