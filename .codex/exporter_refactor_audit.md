# Dreamer4 ONNX Exporter Refactor Audit

Scope: `scripts/webgpu/export_dreamer4_onnx.py` and `visionary/export/onnx_wrappers.py`.

Goal preserved: current fp32 cached browser export path, especially `DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME` / `breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4`.

Validation performed: read-through plus `rg` reference checks and `uv run ruff check scripts/webgpu/export_dreamer4_onnx.py visionary/export/onnx_wrappers.py` (passed, re-run after the concurrent exporter change noted below).

Concurrent worktree note: while this audit was in progress, `scripts/webgpu/export_dreamer4_onnx.py` changed in the worktree to remove the FP16 export branch. I did not make or revert that change. Findings below describe the current worktree unless explicitly marked as already addressed by that concurrent change.

## Safe For Current fp32 Entry-Cache Path

- `pack_tokenizer_latents_for_dynamics()` and `unpack_dynamics_latents()` in `visionary/export/onnx_wrappers.py` appear unused in-repo. `rg` found only their definitions. They duplicate shape conversion logic now done directly in `apply_tokenizer_decode_z()` with `jnp.split` + `jnp.concatenate` to avoid ONNX `Reshape` in the hot path. Safe to remove for this repo; risky only if this module is treated as an external public API.

- `_CachedDynamicsModel.sample_step_full_cache()` and `apply_dynamics_cached_sample_step_full_cache()` are unused in-repo and are not exported by `scripts/webgpu/export_dreamer4_onnx.py`. The current entry-cache path uses `sample_step_append_context_full_cache_entries()` via `apply_dynamics_cached_sample_step_append_context_full_cache_entries()`, not the non-append full-cache sample helper. Safe to remove for the stated path.

- `attention_logit_soft_cap` fields in `_ExportAttention`, `_ExportTransformerBlock`, and `_ExportSpatioTemporalTransformer` are carried through but not used in the export attention math. `_ExportSpatioTemporalTransformer` may still need to accept the constructor argument for compatibility with original model config, but the internal `_ExportTransformerBlock` -> `_ExportAttention` plumbing is dead as written. Safe to reduce after confirming constructor compatibility.

- `_CachedDynamicsModel._tokens()` types `actions` as `jnp.ndarray | None`, but every caller passes an array and the method immediately feeds `actions` to `_ExportActionEmbedding`. Narrowing this type is safe and clarifies there is no supported `None` path.

- `rewrite_singleton_reshapes_for_webgpu()` always saves/checks the model even when no `Reshape` nodes were rewritten. The other rewrite passes generally return early when no replacements exist. Adding an early no-op return is safe and reduces unnecessary file churn in the fp32 path.

- Already addressed by concurrent worktree change: the `--float16`, `--float16_decoder_only`, and `--keep_quickgelu` flags plus the FP16-only helpers were dead for the requested fp32 entry-cache export path: `convert_onnx_to_float16_for_webgpu()`, `repair_cast_output_types()`, `decompose_quickgelu_for_fp16_webgpu()`, `strip_intermediate_value_info()`, and `repair_float16_binary_cast_mismatches()`. They are no longer present in the current exporter file.

## Safe Candidates With Small ABI Caveats

- `--native_attention` and `_native_dot_product_attention_for_export()` are experimental and not used by the default fp32 export path. Removing them would preserve the current default patched export but would drop an explicit CLI comparison mode recorded in manifest `attention_export`. Safe for current path; risky for GQA/native-lowering experiments.

- `--grouped_gqa_attention` is also experimental and unused by the default path, but it is not dead internally: it is propagated into `export_overrides()`, `_export_dot_product_attention()`, and the export transformer partials. Removing it is safe only if grouped-GQA lowering experiments are out of scope.

- `--skip_onnx_optimization` and `--skip_singleton_reshape_rewrite` are debug escape hatches. Removing them and always running the current default passes preserves the current fp32 path, but it removes useful bisect controls for graph-regression work.

- `--simplify_onnx` and `--simplify_demo_only` are not part of the current default path. Dropping onnxsim support would shrink exporter logic and manifest fields, but it removes a behavior-preserving optimization gate used to compare graph simplification.

## Risky To Remove

- `DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME`, `dynamics_sample_append_context_slide_entry_fn()`, `_CachedDynamicsModel.sample_step_append_context_full_cache_entries()`, `_CachedSpatioTemporalTransformer.step_entries()`, and `apply_dynamics_cached_sample_step_append_context_full_cache_entries()` are the current fp32 entry-cache path. Do not remove when preserving the requested path.

- `DYNAMICS_CACHED_PREFILL_NAME` / `apply_dynamics_cached_prefill()` remain part of the browser contract. The manifest marks it as `preferred_prefill_export`, and the benchmark candidates include `breakout_dynamics_prefill_cached_b1_t64`.

- `DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_FULL_CACHE_NAME` is the manifest fallback for the entry-cache artifact. Removing it does not break the entry artifact itself, but it weakens fallback/debug behavior and raw-vs-optimized comparison coverage.

- `DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_LAYER_NAME` and `DYNAMICS_CACHED_PREFILL_LAYER_NAME` are experimental, but they are referenced by `webgpu_app/bench/benchmark.js` candidate lists and `scripts/webgpu/compare_raw_optimized_onnx.py`. Removing them is risky unless the layer-cache benchmark ABI is retired at the same time.

- `DYNAMICS_CACHED_STEP_NAME`, `DYNAMICS_CACHED_SAMPLE_STEP_NAME`, and `DYNAMICS_CACHED_SAMPLE_STEP_SLIDE_NAME` are legacy/fallback artifacts in the manifest. They are outside the entry-cache hot path, but `webgpu_app/bench/benchmark.js` and `scripts/webgpu/compare_jax_onnx_rollout.py` still reference the sample-step artifacts. Remove only with coordinated benchmark/script cleanup.

- `rewrite_slide_static_cache_ops_for_webgpu()` is not applied to the entry artifact, but it is applied to slide full-cache/layer artifacts. It contains shape-specific rewrites for steady-state slide graphs, so removing it is risky while those fallbacks/experiments remain exportable.

## Duplicate Helper Logic

- `_CachedDynamicsModel.sample_step_append_context()`, `sample_step_append_context_full_cache_entries()`, and `sample_step_append_context_layer_cache()` duplicate the same context append setup: `context_step_level`, `context_step_count`, clamped `context_signal_level`, `context_tau_used`, `noised_context_z`, `context_step_levels`, and `context_signal_levels`. Extracting a small array-only helper is safe for behavior and should not affect Flax module naming.

- `_CachedDynamicsModel.step()`, `step_entries()`, `step_layer_cache()`, `predict_step()`, and `predict_step_layer_cache()` duplicate token/rope/mask setup before dispatching to transformer methods. A helper that returns `(tokens, total_tokens, observation_offset, token_dim, spatial_rope, temporal_rope, spatial_mask)` would cut repetition. Keep it array-only; moving module construction into helpers would be risky under Flax compact naming.

- `_CachedSpatioTemporalTransformer.step()`, `step_layer_cache()`, `step_entries()`, `predict_step()`, and `predict_step_layer_cache()` repeat the spatial block loop and temporal cache preparation. This is the biggest LOC reduction opportunity, but it is risky: small changes can alter Flax submodule names, cache layout ABI, or ONNX output ordering. Prefer leaving this explicit unless there is strong test coverage around exported graph names and validation outputs.

- `scripts/webgpu/export_dreamer4_onnx.py` repeats the same artifact lifecycle many times: path constants, inner export function, `export_to_onnx()` call, validation block, metadata collection, manifest entry, and print. A declarative artifact table would save substantial LOC, but it is risky because the explicit code currently documents the ABI and output ordering for each ONNX artifact. If refactored, start only with metadata/manifest field generation, not export function signatures.

- The graph rewrite passes duplicate value shape collection and producer/consumer indexing (`rewrite_slide_static_cache_ops_for_webgpu()`, `rewrite_singleton_reshapes_for_webgpu()`, `rewrite_gqa_repeats_for_webgpu()`, `rewrite_head_projection_reshapes_for_webgpu()`, `rewrite_rmsnorm_for_webgpu()`, `fuse_skip_simplified_layer_norm_for_webgpu()`). Shared helpers for `value_shapes`, `producer`, and `consumers` are safe if kept minimal, but do not abstract the pattern matching itself.

## Suggested Order

1. Remove repo-unused helpers: `pack_tokenizer_latents_for_dynamics()`, `unpack_dynamics_latents()`, `_CachedDynamicsModel.sample_step_full_cache()`, and `apply_dynamics_cached_sample_step_full_cache()`.
2. Add no-op early return to `rewrite_singleton_reshapes_for_webgpu()`.
3. Extract the duplicated append-context input helper in `_CachedDynamicsModel`.
4. Decide explicitly whether native/grouped-GQA/simplification flags are still supported experiments before deleting those larger feature branches. The FP16 branch appears already removed by concurrent worktree changes.
5. Treat legacy/layer/fallback artifact removal as a coordinated ABI cleanup with benchmark and comparison scripts, not a local exporter-only refactor.
