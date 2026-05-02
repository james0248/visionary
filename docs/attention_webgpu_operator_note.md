# ORT WebGPU Attention Operator Note

Date: 2026-05-03

## Question

Can standard `ai.onnx::Attention` be used on ORT WebGPU for the current hot artifact, and are `com.microsoft::Attention`, `com.microsoft::MultiHeadAttention`, or `com.microsoft::GroupQueryAttention` likely useful?

Hot artifact inspected:

- `webgpu_app/assets/breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4.onnx`

## Local Sources

- `package.json`, `bun.lock`: browser package is `onnxruntime-web@1.24.3`.
- `pyproject.toml`, `uv.lock`: local ONNX tooling includes `onnx==1.21.0`, `onnxruntime==1.25.0`.
- `node_modules/onnxruntime-web/docs/webgpu-operators.md`: generated WebGPU EP support table.
- `node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/op-resolve-rules.ts`: WebGPU JS op dispatch map.
- `node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/ops/attention.ts`
- `node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/ops/multihead-attention.ts`
- `node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/ops/group-query-attention.ts`
- `scripts/webgpu/export_dreamer4_onnx.py`
- `webgpu_app/assets/breakout_onnx_manifest.json`
- Existing local notes: `.codex/fused_attention_path.md`, `.codex/temporal_gqa_export_review.md`, `.codex/onnx_webgpu_progress.md`

## Findings

Standard `ai.onnx::Attention` should not be treated as usable on ORT WebGPU here. Local ONNX `1.21.0` does define standard-domain `Attention` schemas in the empty/`ai.onnx` domain at opsets 23 and 24, but the installed ORT WebGPU support table lists only `Attention | com.microsoft(1+)`; it does not list `Attention | ai.onnx(...)`. The WebGPU `attention.ts` implementation also validates the contrib-style fused input/weights/bias form, not the standard ONNX Q/K/V attention contract. Raising or keeping the model opset at 23/24 does not by itself put standard `ai.onnx::Attention` on WebGPU.

The current hot artifact does not contain any `Attention`, `MultiHeadAttention`, or `GroupQueryAttention` nodes. Its inspected op counts are:

- `ai.onnx`: 5,831 nodes
- `com.microsoft`: 119 nodes, all `QuickGelu`
- attention-like core: `Softmax=119`, `Einsum=716`, `Gemm=366`, `Transpose=59`, `Gather=239`, `Reshape=0`

The manifest confirms the active export path is `attention_export.implementation = patched_onnx_decomposition` and `fused_temporal_gqa.enabled = false` for this artifact.

`com.microsoft::Attention` is probably not the right target. The WebGPU implementation has explicit gaps for mask and past handling, and the operator shape is the older fused projection form. The model already has separate cached K/V and GQA-style KV heads, so adapting to this op would fight the current cache layout.

`com.microsoft::MultiHeadAttention` is also a poor fit for the model's grouped-query attention. The local implementation expects ordinary MHA shapes: 3D key/value hidden size equal to query hidden size, or 4D key/value using `num_heads`. This model has 8 query heads but only 2 KV heads. Using MHA would require materializing/repeating KV heads up to 8 heads, which is exactly the expensive pattern the current rewrites tried to remove. The WebGPU docs also still annotate MHA with incomplete mask and past/present support.

`com.microsoft::GroupQueryAttention` is the only contrib attention op that is semantically aligned with the model. The local implementation accepts `num_heads` and `kv_num_heads`, computes `nReps = num_heads / kv_num_heads`, and has paths for past K/V. The repo already has a gated post-export pass, `rewrite_cached_temporal_attention_to_gqa()`, for no-mask full-cache temporal attention islands.

On a temp copy of the hot artifact, that pass matched 29 temporal islands:

- `GroupQueryAttention`: `0 -> 29`
- `Softmax`: `119 -> 90`
- `Einsum`: `716 -> 658`
- `Gather`: `239 -> 181`
- `Reshape`: `0 -> 87`
- nodes: `5950 -> 5892` after pruning

That means GQA is relevant, but not automatically a win. Existing local benchmark notes say the corrected GQA fusion path passed demo smoke but regressed streaming performance, and the current faster artifact restored the non-GQA baseline. The likely cause remains the same: the current GQA rewrite reintroduces graph-level `Reshape` nodes and past-K/V `Transpose` work around the fused op, while `Reshape` is documented as having no WebGPU kernel in the local support table.

## Conclusions

Do not target standard `ai.onnx::Attention` for ORT WebGPU in this repo. It exists in local ONNX schemas, but the installed ORT WebGPU package documents and implements only the Microsoft contrib attention variants.

Do not spend effort on `com.microsoft::Attention` or `com.microsoft::MultiHeadAttention` for the hot Dreamer dynamics path unless the model/cache layout changes substantially. They do not line up with 8-query-head/2-KV-head GQA without reintroducing repeated KV heads or unsupported mask/cache cases.

`com.microsoft::GroupQueryAttention` remains the plausible fused attention operator for this workload, but only with a better graph/export layout: flat Q/K/V tensors into the op, cache already in GQA-native BNSH layout, no graph-level `Reshape`, dead original attention nodes pruned, and post-fusion accuracy plus browser WebGPU placement validation. As currently wired, the repo keeps it gated/off for good reason.
