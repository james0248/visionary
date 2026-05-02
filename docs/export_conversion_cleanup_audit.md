# ONNX export conversion cleanup audit

Scope: `scripts/webgpu/export_dreamer4_onnx.py` only. Live path assumed to be fp32 with `--export_cached --simplify_onnx --simplify_demo_only`, using `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4` as the steady-state entry-cache artifact.

## Safe removals

- Float16 conversion path: `--float16`, `--float16_decoder_only`, `--keep_quickgelu`, `convert_onnx_to_float16_for_webgpu`, `repair_cast_output_types`, `strip_intermediate_value_info`, `repair_float16_binary_cast_mismatches`, `decompose_quickgelu_for_fp16_webgpu`, plus manifest fields `precision_export`, per-export `precision`, `cast_type_repair`, `quickgelu_decomposition`, `value_info_strip`, and `float16_binary_cast_repair`.
  - Reason: the live path is fp32. These functions are only selected through `float16_export_names`; with no float16 flag they either do not run or write disabled metadata.

- Experimental attention/projection export toggles: `--native_attention`, `--grouped_gqa_attention`, `--packed_qkv_projection`, `--packed_swiglu_projection`, their validation in `require_static_phase1_args`, their passthrough arguments in every local wrapper, and manifest fields under `attention_export` that only record those choices.
  - Reason: the live path uses default patched ONNX decomposition and does not pass any of these flags. Removing them would keep the default behavior if wrapper calls are simplified to their default arguments.

- Experimental post-export GQA fusion path: `--fused_temporal_gqa`, `_single_consumer`, `_producer_input_by_op`, `_prune_dead_onnx_nodes`, `rewrite_cached_temporal_attention_to_gqa`, the `math` import added for that fusion, the `fused_temporal_gqa` processing block, manifest `layout_rewrite.cached_temporal_gqa_fusion`, and per-export `fused_temporal_gqa`.
  - Reason: the live invocation does not pass `--fused_temporal_gqa`. This path is isolated after validation/precision work and does not affect entry-cache fp32 export unless the flag is set.

- Raw artifact snapshot path: `--raw_out_dir`, `copy_onnx_artifact`, `snapshot_raw_artifacts`, and manifest top-level `raw_artifacts`.
  - Reason: the live invocation does not set `--raw_out_dir`; the path only copies already-exported ONNX files for comparison.

- Optional context latent manifest path: `--context_latents` and manifest top-level `context_latents`.
  - Reason: this script only records metadata for a precomputed artifact. The live export path and entry-cache ONNX graph do not consume it.

- Non-live cached artifact families can be removed from this exporter if benchmark/demo fallback support is intentionally dropped: `DYNAMICS_CACHED_SAMPLE_STEP_NAME`, `DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME`, `DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME`, their local wrapper functions, export calls, validation blocks, `exported_paths` entries, manifest export entries, and `demo_generation.legacy_sample_step_export`, `demo_generation.legacy_steady_state_sample_step_export`, `demo_generation.experimental_layer_steady_state_step_export`.
  - Reason: they are not the stated live entry-cache artifact. `--simplify_demo_only` currently includes them because they are treated as benchmark/demo candidates, not because entry-cache export requires them.

- Uncached/full-sequence exports can be removed from the live export script if the script is narrowed to browser cached assets: `TOKENIZER_DECODER_NAME`, `DYNAMICS_UNCACHED_NAME`, `decoder_fn`, `dynamics_fn`, their unconditional export/validation/manifest entries, and `production_browser_ready: false` for the uncached dynamics entry.
  - Reason: these are always exported today but are not on the stated cached live demo path. They are useful as contract/diagnostic outputs, not as the entry-cache steady-state artifact.

## Risky removals

- `DYNAMICS_CACHED_PREFILL_NAME`, `TOKENIZER_DECODE_Z_STEP_NAME`, and their wrapper/export/manifest/validation paths.
  - Reason: even if the steady-state artifact is entry-cache, the browser/benchmark flow still needs a cached prefix/prefill phase and a decode-z frame path. `webgpu_app/bench/benchmark.js` and `webgpu_app/demo/main.js` both reference these names.

- `DYNAMICS_CACHED_STEP_NAME` and `DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_NAME`.
  - Reason: they look like older/fill-mode paths relative to the entry-cache steady-state artifact, but `webgpu_app/demo/main.js` still loads `breakout_dynamics_step_cached_b1_t1` for prefix stepping and `breakout_dynamics_sample_append_context_b1_t1_s4` for sampling.

- `DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME`.
  - Reason: it is no longer the stated preferred live artifact, but it is still referenced as a fallback in the manifest (`fallback_steady_state_step_export` and `entry_cache_export` chain), in `webgpu_app/demo/main.js`, in `webgpu_app/bench/benchmark.js`, and by `scripts/webgpu/verify_entry_cache_update.py`.

- `DYNAMICS_CACHED_PREFILL_LAYER_NAME` and `DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME`.
  - Reason: layer-cache exports look experimental, but `scripts/webgpu/compare_raw_optimized_onnx.py` includes the layer prefill/step names in its default comparison list, and `webgpu_app/bench/benchmark.js` lists the layer step artifact as a candidate.

- `--skip_onnx_optimization` and `--skip_singleton_reshape_rewrite`.
  - Reason: live defaults run optimization and WebGPU rewrites, so the skip flags are not live. Removing the flags is safe only if the default passes remain unconditional; removing the underlying optimization/rewrite functions would change the live exported graph.

- `--validate`, `--atol`, `--rtol`, `run_ort`, `compare_arrays`, `validate_single_output`, and `validate_outputs`.
  - Reason: validation is off in the stated live invocation, but these are still the only in-script correctness gate for export variants. They are safe for a production-only exporter, risky for development.

- `TOKENIZER_DECODER_STEP_NAME`.
  - Reason: the current demo prefers `breakout_tokenizer_decode_z_b1_t1`, but `webgpu_app/bench/benchmark.js` lists `breakout_tokenizer_decoder_b1_t1` as a decoder fallback.

## Looks dead but still referenced

- `breakout_dynamics_step_cached_b1_t1`: still loaded by `webgpu_app/demo/main.js` for prefix stepping and listed by benchmark/profile diagnostics.
- `breakout_dynamics_sample_append_context_slide_full_cache_b1_t1_s4`: still used as a demo/benchmark fallback and by entry-cache verification.
- `breakout_dynamics_sample_append_context_slide_layer_b1_t1_s4`: still listed by benchmark and raw/optimized comparison tooling.
- `breakout_dynamics_cached_sample_step_b1_t1_s4` and `breakout_dynamics_cached_sample_step_slide_b1_t1_s4`: legacy paths, but still listed by benchmark fallback selection.
- `breakout_tokenizer_decoder_b1_t1`: not the preferred decoder for the current decode-z path, but still listed by benchmark fallback selection.
- `demo_generation.decode_z.export`: `compare_raw_optimized_onnx.py` attempts to read it, but the exporter currently writes only `source`, `dynamics_shape`, and `decoder_latent_shape`; it then appends the hard-coded `breakout_tokenizer_decode_z_b1_t1` fallback. This field looks like a missing/obsolete manifest contract rather than live runtime data.
