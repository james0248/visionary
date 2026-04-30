# ORT WebGPU Fused Attention Path

Date: 2026-04-29

## Bottom Line

Yes, the cached temporal attention shape we care about can target ORT WebGPU's fused `com.microsoft::GroupQueryAttention` path:

- query length: `1`
- key/value length: `65` (`64` cached + current token)
- query heads: `8`
- KV heads: `2`
- head dim: `64`
- flattened temporal batch: `36`

The viable fused op is `GroupQueryAttention`, not `MultiHeadAttention`, and not standard ONNX `Attention`/`ScaledDotProductAttention`.

The best first implementation is a post-export ONNX rewrite for the cached temporal attention subgraphs, gated to the full-cache streaming case. If that profiles well, move the lowering into a small jax2onnx custom primitive/plugin used by the export wrappers. Wrapper-only JAX changes will not emit the ORT contrib fused op with the installed jax2onnx lowering.

## Evidence

Current exported graphs contain no fused attention nodes:

- `breakout_dynamics_step_cached_b1_t1.onnx`: 1,902 nodes, 24 `Softmax`, 48 `Einsum`, 48 `Expand`, 399 `Reshape`, no `*Attention` nodes.
- `breakout_dynamics_cached_sample_step_b1_t1_s4.onnx`: 7,290 nodes, 96 `Softmax`, 192 `Einsum`, 192 `Expand`, 1,511 `Reshape`, no `*Attention` nodes.
- Temporal cached attention softmax shape is `(36, 8, 1, 65)` in the step graph.

Relevant local sources:

- ORT WebGPU support table lists `GroupQueryAttention | com.microsoft(1+)`, `MultiHeadAttention | com.microsoft(1+)`, and `Reshape ... no GPU kernel`: `node_modules/onnxruntime-web/docs/webgpu-operators.md:61`, `:76`, `:95`.
- `GroupQueryAttention` accepts 3D query/key/value shapes `(B,S,Dq)`, `(B,L,Dkv)`, with `query.dims[2] % key.dims[2] == 0`, and takes `num_heads`, `kv_num_heads`, `scale`, optional past KV, `seqlens`, and `total_sequence_length`: `node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/ops/group-query-attention.ts:69-72`, `:139-143`, `:196-199`, `:315-405`.
- The GQA kernel handles KV head repetition internally via `nReps = numHeads / kvNumHeads` in the shared attention shaders: `node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/ops/attention.ts:796-886`.
- Installed jax2onnx expands grouped KV heads into ordinary ops before attention: `expand_grouped_kv_heads` is called for K and V in `.venv/lib/python3.11/site-packages/jax2onnx/plugins/jax/nn/dot_product_attention.py:899-909` and `:1357-1367`.
- This repo's export wrapper also lowers attention to `Einsum`/`Softmax`, and the default GQA path explicitly repeats K/V heads: `visionary/export/onnx_wrappers.py:190-283`. The cached step currently concatenates cache + current K/V and builds a mask before that lowered attention: `visionary/export/onnx_wrappers.py:536-549`.

## Why Other Fused Ops Are Poor Fits

`MultiHeadAttention` is not a good match for Dreamer4 GQA. Its validator requires K/V hidden size equal to Q hidden size for 3D inputs, and past cache shape uses `num_heads`, not `kv_num_heads`: `multihead-attention.ts:126-129`, `:88-100`. We would have to materialize 2 KV heads into 8 heads before the fused op, which is exactly the repeated-head work causing the current bad pattern.

Standard ONNX `Attention` from ONNX opset 24 is also not the target. The local ONNX schema exists, but ORT WebGPU 1.24.3's support table only lists contrib `com.microsoft::Attention`, and the WebGPU `Attention` implementation expects input/weights/bias style contrib inputs, not standard `Q,K,V` inputs. Raising the export opset from 23 to 24 will not by itself put this path on WebGPU.

## GQA Shape Contract For This Repo

For each temporal block in cached step, flatten `(batch, token)` to `Bflat = 36`.

Preferred full-cache GQA input contract:

- `query`: `[36, 1, 512]`
- `key`: `[36, 65, 128]`
- `value`: `[36, 65, 128]`
- attributes: `num_heads=8`, `kv_num_heads=2`, `scale=0.125`, `softcap=0`, `do_rotary=0`, `rotary_interleaved=0`, `smooth_softmax=0`, `local_window_size=-1`
- output: `[36, 1, 512]`

This replaces the current temporal attention core:

`Expand/Reshape K,V to 8 heads -> Einsum QK -> mask Add -> Softmax -> Einsum WV -> Reshape`

with:

`GroupQueryAttention -> output projection`

Generic non-full cache is possible, but needs `seqlens` shaped `[36]` rather than the current scalar `[1]`. The GQA softmax uses `seqLens` to cap the attended length (`past_sequence_length + query_index + 1`), so a scalar cache length must be broadcast/repeated to the flattened temporal batch.

Past-key/value mode is possible but has one important ORT WebGPU quirk: `applyAttention` only includes past KV when the node has present KV outputs. If using inputs `past_key`/`past_value`, the GQA node must declare three outputs, even if the present outputs are later sliced or discarded.

## Post-Export Prototype

Start with a narrow graph rewrite for `breakout_dynamics_step_cached_b1_t1.onnx`, then apply the same rewrite to the four repeated copies in `breakout_dynamics_cached_sample_step_b1_t1_s4.onnx`.

1. Pattern-match temporal attention islands by `Softmax` input/output shape `(36, 8, 1, 65)`.
2. Trace each island back to the post-RMSNorm/post-rotary `q`, concatenated `keys`, and concatenated `values`.
3. Insert or reuse flattening to produce GQA inputs:
   - `q`: `[36, 1, 8, 64] -> [36, 1, 512]`
   - `keys`: `[36, 65, 2, 64] -> [36, 65, 128]`
   - `values`: `[36, 65, 2, 64] -> [36, 65, 128]`
4. Replace the island with one `com.microsoft::GroupQueryAttention` node.
5. Wire the `[36, 1, 512]` output into the existing output projection path.

This first rewrite may still need a few `Reshape` nodes around the GQA boundary. That is acceptable for the probe: it deletes the expensive repeated-head `Expand -> Reshape -> Einsum` path and verifies ORT WebGPU dispatches GQA correctly. If the remaining reshape copies dominate, move to the exporter/plugin path below.

## Exporter/Plugin Path

After the post-export probe validates performance, implement a jax2onnx lowering rather than relying on fragile graph surgery:

1. Add a small custom primitive used only by `visionary/export/onnx_wrappers.py` for ONNX export.
2. Lower that primitive directly to `com.microsoft::GroupQueryAttention`.
3. Keep explicit RMSNorm and rotary before the custom op at first; do not use GQA `do_rotary` until the non-interleaved RoPE convention is verified against this repo's `apply_rotary_embedding`.
4. Add a full-cache cached-sample export variant so temporal GQA can omit dynamic mask/seqlens in the streaming benchmark.
5. If generic partial-cache support is required, make the wrapper produce `seqlens: [36]` and `total_sequence_length: [1]`/scalar for GQA.
6. Consider changing the temporal cache export layout to store flat hidden KV as `[layers, batch, tokens, context, 128]` for ONNX-only assets. That avoids repeated `[... 2, 64] <-> [... 128]` reshapes around fused attention while leaving core model code unchanged.

## Validation

For each candidate export:

- Python ORT parity against the current export for `pred_z`, `candidate_k_cache`, `candidate_v_cache`, and `candidate_cache_length`.
- Browser session profile must show `GroupQueryAttention` assigned to WebGPU and fewer `MemcpyToHost`/`MemcpyFromHost` events.
- Check specifically that temporal `(36,65,2,4,64) -> (36,65,8,64)` and `(36,8,1,65)` attention islands disappear from the hot path.
- Benchmark both single cached step and fused `sample_step_s4`; the latter multiplies any remaining attention-boundary cost by four.

## Expected Impact

This will not remove every CPU boundary because ORT WebGPU still has no `Reshape` kernel and the model has many dense-layer flatten/unflatten reshapes. It should remove the most attention-specific repeated-head work in cached temporal attention. The likely largest win is in the s4 streaming graph, where the same temporal attention pattern is replicated across sample iterations.
