# Implementation plan: Dreamer4 ONNX export

Date: 2026-04-28

## Non-goals

- Do not edit training loops or core model behavior.
- Do not make the browser demo yet.
- Do not assume WebGPU support from successful Python ONNX Runtime validation.

## Phase 1: Exportable inference wrappers

Add `visionary/onnx_inference.py`.

Functions:

- `create_tokenizer_from_config(cfg, *, dtype=None)`.
- `create_dynamics_from_config(cfg, *, dtype=None)`.
- `tokenizer_encode_apply(params, cfg, video_patches)`.
- `tokenizer_decode_apply(params, cfg, latents)`.
- `dynamics_step_apply(params, cfg, z, actions, step_levels, signal_levels)`.
- Shape helpers:
  - `tokenizer_patch_shape(cfg) -> (patch_count, patch_dim)`.
  - `tokenizer_latent_shape(cfg) -> (num_latents, channel_dim)`.
  - `dynamics_token_shape(cfg, tokenizer_cfg) -> (num_obs_tokens, token_dim)`.

Implementation detail:

- Use the existing module methods:
  - `Tokenizer.encode({"video": video_patches})`
  - `Tokenizer.decode(latents)`
  - `DynamicsModel.__call__(z, actions, step_levels, signal_levels)`
- Pass `{"params": params}` into `.apply(...)`.
- Keep all wrappers deterministic and RNG-free.

## Phase 2: Export script

Add `scripts/webgpu/export_dreamer4_onnx.py`.

CLI:

```text
uv run python scripts/webgpu/export_dreamer4_onnx.py \
  --tokenizer_dir gs://visionary-exp/breakout/checkpoints/tokenizer_l8p8 \
  --dynamics_dir gs://visionary-exp/breakout/checkpoints/dynamics_l24 \
  --out_dir webgpu_app/assets \
  --validate
```

Script flow:

1. Parse args.
2. Restore tokenizer and dynamics exports with `restore_model_export_single_device`.
3. Derive shapes from configs.
4. Create fixed dummy inputs:
   - encoder: `(B, encoder_frames, patch_count, patch_dim)`, `float32`.
   - decoder: `(B, decoder_frames, num_latents, channel_dim)`, `float32`.
   - dynamics: `(B, dynamics_context, num_obs_tokens, token_dim)`, `float32`; action and level tensors as `int32`.
5. Call `jax2onnx.to_onnx` for each wrapper.
6. Write `manifest.json`.
7. If `--validate`, compare JAX vs Python ORT outputs.

## Phase 3: Validation and compatibility gates

Required local checks:

- `uv run ruff check visionary/onnx_inference.py scripts/webgpu/export_dreamer4_onnx.py`
- Run export against a local or GCS checkpoint.
- Load generated ONNX files with Python `onnxruntime.InferenceSession`.
- Compare outputs against JAX wrappers.

Required web checks before calling it demo-ready:

- Create ORT Web sessions with `executionProviders: ["webgpu", "wasm"]` or explicit WebGPU first/fallback logic.
- Verify session creation does not fall back silently for important nodes.
- Run one dynamics step and one decoder step in browser.
- Profile latency for target hardware.

## Open technical risks

- `jax.nn.dot_product_attention` lowering may not convert cleanly or may produce ONNX ops unsupported by ORT WebGPU.
- Current JAX version is `0.9.1`, while converter compatibility should be tested empirically with the installed environment.
- Model size may be too large for a comfortable browser demo unless fp16 or a smaller checkpoint is used.
- `bfloat16` in model configs may need override to `float32` or `float16` during export; browser support for bf16 is not a good assumption.
- Dynamics rollout quality and speed depend on how much of `generate_next` is kept in JS vs ONNX. Start with a static `dynamics_step` export.

## Suggested first milestone

Produce three validated ONNX files plus manifest for:

- `B=1`
- `encoder_frames=16`
- `decoder_frames=1`
- `dynamics_context=17`
- `sample_steps=1` or JS-side single denoise iteration for smoke testing

Only after that should we optimize the browser loop and model precision.
