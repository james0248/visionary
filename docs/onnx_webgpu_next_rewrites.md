# ONNX WebGPU Next Rewrites

Date: 2026-04-29

Scope: read-only analysis of the current benchmark JS, current exported ONNX assets, ORT WebGPU support files, and existing benchmark/profile results. No production code was edited.

## Current State

The hot path is no longer the old graph-level `Reshape` CPU-island problem for the fused streaming artifact.

Current `breakout_dynamics_cached_sample_step_b1_t1_s4.onnx`:

- 6,866 nodes.
- 0 `Reshape` nodes in the fused s4 graph.
- 756 `Unsqueeze`, 368 `Squeeze`, 576 `Einsum`, 296 `Gemm`, 384 `ReduceMean`, 384 `Sqrt`, 483 `Div`, 96 `Softmax`, 258 `Concat`, 196 `Gather`.
- Spatial attention: 72 softmaxes with shape `(1, 8, 36, 36)`.
- Cached temporal attention: 24 softmaxes with shape `(36, 8, 1, 65)`.
- Cache output surface: `candidate_k_cache` and `candidate_v_cache`, each `[6, 1, 36, 64, 2, 64]`, 1,769,472 fp32 elements each, about 7.1 MB each.

Current `breakout_tokenizer_decode_z_b1_t1.onnx`:

- 405 nodes.
- 1 remaining `Reshape`, at the input `z [1,1,32,32] -> [64,16]`.
- 48 `Einsum`, 26 `Gemm`, 32 `RMSNormalization`, 8 `Softmax`, 62 `Unsqueeze`, 29 `Squeeze`.
- Spatial decoder attention: 6 softmaxes with shape `(1, 8, 256, 256)`.
- Temporal decoder attention: 2 softmaxes with shape `(256, 8, 1, 1)`.

Latest non-profiling baseline before the current profiling run:

- Streaming frame: 121.91 ms mean.
- Fused dynamics frame: 66.69 ms mean.
- Decoder frame: 55.14 ms mean.

Current `latest.json` was run with profiling enabled and `profilingDrainMs=100`, so its 314 ms streaming-frame number includes two 100 ms drain windows per frame. Its per-session timings are still useful: dynamics around 67 ms and decoder around 45 ms in that run.

## Main Conclusion

Getting to `<=50 ms` will not come from more singleton reshape rewrites. The remaining gap is roughly:

- Dynamics s4 graph: needs about 2x to 3x reduction.
- Decoder: needs about 2x reduction or must be partially fused with rendering/patch output.
- Runtime path: needs stable GPU-buffer reuse/graph capture so repeated per-frame session runs stop rebinding and reallocating the same large outputs.

The next concrete work should target semantic fusion and ABI changes, not only local ONNX cleanup.

## 1. Make True GQA The Primary Rewrite

### Why

The current graph still implements attention as projection kernels, `Gather` head repeat, `Einsum`, mask arithmetic, `Softmax`, another `Einsum`, then output projection. This is WebGPU-compatible now, but it creates many small kernels:

- Dynamics s4: 96 softmaxes and 576 einsums.
- Decoder: 8 softmaxes and 48 einsums.

ORT WebGPU supports `com.microsoft::GroupQueryAttention`. The local implementation handles true KV head sharing internally with `num_heads=8`, `kv_num_heads=2`, and `nReps=numHeads/kvNumHeads`. That is the right primitive for the current model.

### Dynamics Rewrite

Replace both dynamics attention forms:

1. Spatial attention, no mask:
   - Current shape: query/key/value around `[1, 36, 8, 64]` and repeated K/V `[1, 36, 8, 64]`.
   - Target GQA inputs:
     - `query [1, 36, 512]`
     - `key [1, 36, 128]`
     - `value [1, 36, 128]`
     - `num_heads=8`, `kv_num_heads=2`, `scale=0.125`

2. Cached temporal attention, full-cache streaming:
   - Current shape: query `[36, 1, 8, 64]`, K/V `[36, 65, 2, 64]`.
   - Target GQA inputs:
     - `query [36, 1, 512]`
     - `key [36, 65, 128]`
     - `value [36, 65, 128]`
     - `seqlens [36]` all 64, `total_sequence_length [65]`

The existing post-export `rewrite_cached_temporal_attention_to_gqa()` is aimed at only the temporal case and inserts `Reshape` nodes. The next version should be export-native or plugin-native so Q/K/V are emitted flat as `[B,S,hidden]` before the GQA node, avoiding new graph-level `Reshape`.

### Decoder Rewrite

The decoder spatial mask is block structured:

- Latent queries attend latent keys.
- Image queries attend latent and image keys.
- Latent queries do not attend image keys.

A single GQA node cannot represent that arbitrary block mask because the local ORT WebGPU GQA path does not support an additive mask. Split the decoder spatial attention exactly:

1. Latent stream GQA:
   - `query/key/value` from latent tokens only, length 64.

2. Image stream GQA:
   - query from image tokens, length 192.
   - key/value from concatenated latent + image tokens, length 256.

This preserves the decoder mask while removing the masked latent-to-image work and enabling fused GQA.

Expected impact: highest. This is the only rewrite class that can remove dozens of attention kernels per frame rather than just making individual kernels smaller.

## 2. Specialize The Decoder For Single-Frame Decode

The decoder is now a first-order bottleneck: about 45-55 ms by itself.

The exported decoder always has `seq_len=1`. Its temporal attention softmax shape is `(256, 8, 1, 1)`. For sequence length 1, temporal attention is exact identity weighting over the single value:

```text
DPA(q, k, v) == v
```

after the value projection and GQA head repetition. Q projection, K projection, Q/K RMSNorm, RoPE, QK logits, softmax, and attention masking are dead work for these temporal layers.

Concrete exact rewrite:

- In tokenizer decoder export-only path, replace temporal attention blocks at `t=1` with `V projection -> output projection`.
- Then fold `V projection + repeat-to-query-heads + output projection` into one Dense/Einsum where possible.
- Keep the block residual and MLP unchanged.

This targets the two decoder temporal blocks and removes the two `(256, 8, 1, 1)` softmax islands plus their Q/K plumbing.

Second decoder rewrite:

- Keep latent and image token streams separate through spatial layers.
- Only project the final 192 image tokens to patches.
- Preserve latent updates through latent-only attention and MLP because image tokens in later layers depend on updated latent tokens.

This is more invasive than post-export surgery but matches the decoder mask and avoids computing masked latent-to-image logits.

## 3. Replace Dynamics RMSNorm Decomposition With A Supported Fused Norm

The fused s4 dynamics graph still has decomposed RMSNorm:

- 384 `ReduceMean`
- 384 `Sqrt`
- 483 `Div`
- 1,636 `Mul`

The decoder graph has 32 `RMSNormalization` nodes, but local ORT WebGPU resolver files do not show a WebGPU kernel for `RMSNormalization`. `LayerNormalization` is present; `RMSNormalization` and `SimplifiedLayerNormalization` are not visible in the local resolver.

Concrete next experiment:

1. Browser-profile a decoder session specifically to confirm where `RMSNormalization` lands.
2. For dynamics, try a post-export rewrite from the exact RMS pattern:
   - `ReduceMean(x*x, axis=-1, keepdims=1) -> Add(eps) -> Sqrt/Div or Rsqrt/Mul -> Mul(scale)`
   - into the fastest WebGPU-supported fused norm available in this ORT build.
3. If no RMS-style fused WebGPU op is actually available, do not emit `RMSNormalization` for dynamics. Instead fuse RMSNorm into neighboring custom attention/MLP kernels or leave it decomposed until true fused kernels are introduced.

Expected impact: medium. This removes hundreds of tiny elementwise/reduction kernels in s4, but it will not by itself halve total frame time unless it also eliminates provider boundaries.

## 4. Change The Cache ABI To Avoid Full-Cache Outputs

Current fused sample step returns full candidate K/V cache tensors every frame:

```text
candidate_k_cache [6,1,36,64,2,64]
candidate_v_cache [6,1,36,64,2,64]
```

They stay on `gpu-buffer`, so this is not a CPU download. It is still a large per-frame graph output allocation/binding surface and forces the ONNX graph to build full shifted cache tensors via `Slice` + `Concat`.

Concrete rewrite:

- Change the model/export ABI to output only the new cache entries:
  - `new_k [6,1,36,1,2,64]`
  - `new_v [6,1,36,1,2,64]`
- Keep a persistent browser-owned ring buffer for full K/V cache.
- Update that ring buffer with a tiny WebGPU compute copy or queue write.
- Feed the same persistent cache buffer back into the next ONNX run.
- Pass a logical cursor/index vector if chronological order must be reconstructed inside the graph.

This reduces cache output size from about 14.2 MB per frame to about 0.22 MB per frame. It also removes the final full-cache `Concat` outputs from the ONNX critical path.

Pure ONNX cannot update an external cache buffer in place portably, so this is intentionally a runtime ABI rewrite, not just graph surgery.

## 5. Add GPU Buffer Pinning And Graph Capture As A Benchmark Mode

The benchmark already uses `preferredOutputLocation`:

- Prefill cache outputs: `gpu-buffer`.
- Fused step `final_z`, `candidate_k_cache`, `candidate_v_cache`: `gpu-buffer`.
- Decoder output: `gpu-buffer`.

ORT Web also supports `enableGraphCapture`, but it requires GPU-buffer locations for captured inputs/outputs. The current benchmark still has CPU-created inputs such as actions and first-frame `z`, and it receives fresh output tensors each run.

Concrete benchmark/export experiment:

- Add a separate graph-capture benchmark mode.
- Preallocate stable GPU buffers for step inputs and outputs.
- Feed `final_z` directly as the next step's `z` and decoder input without CPU materialization.
- Keep action/step metadata in GPU buffers, or specialize actions for the benchmark if needed.
- Use the cache ABI rewrite above so the same cache buffers remain stable across frames.

This should reduce per-run binding/allocation overhead. It will not fix the compute graph alone, but it is required for a realistic <=50 ms streaming loop once graph compute is reduced.

## 6. Keep Output Fetches Minimal

The current fetch policy is mostly correct:

- Fused step fetches `final_z` and final candidate cache only.
- It does not fetch `pred_z` unless needed for debug.
- Decoder output stays on `gpu-buffer` unless `debugStats=true`.

Next small cleanup:

- Remove `pred_z` as a production graph output from the fused s4 artifact, or export a separate benchmark/demo artifact without it.
- If the renderer can consume a GPU texture/buffer directly, avoid any future `getData()` path for `patches`.
- Consider a decoder variant whose final output is already render-layout friendly, so a later renderer does not need a CPU-side patch unshuffle.

Expected impact: low to medium. The current output fetch policy is not the dominant issue, but stale graph outputs can block graph capture and stable output binding.

## Ranked Next Work

1. Implement true GQA export/plugin for dynamics spatial and cached temporal attention.
2. Implement true GQA split-stream decoder spatial attention.
3. Specialize decoder temporal layers for `seq_len=1` and fold the value/output projection.
4. Add a cache-delta/ring-buffer ABI to stop returning full shifted K/V caches.
5. Add graph-capture benchmark mode with preallocated GPU buffers.
6. Investigate fused RMSNorm only after confirming actual ORT WebGPU placement for `RMSNormalization`.

## Validation Checklist

For every rewrite:

- Validate ONNX CPU parity against the current artifact for `final_z`, cache outputs/deltas, and decoder `patches`.
- Browser-profile the exact artifact, not only the single-step fallback graph.
- Check that `GroupQueryAttention` is assigned to WebGPU.
- Check that decomposed attention islands disappear:
  - Dynamics spatial `(1,8,36,36)`.
  - Dynamics temporal `(36,8,1,65)`.
  - Decoder spatial `(1,8,256,256)`.
  - Decoder temporal `(256,8,1,1)`.
- Run non-profiling benchmark separately; profiling drain adds 100 ms per profiled scope and invalidates streaming-frame wall time.
- Track both session time and full streaming-frame time, because graph capture/buffer reuse changes the gap between those numbers.

## Practical Target Split

A plausible <=50 ms frame budget needs something like:

- Dynamics fused s4: 20-25 ms.
- Decoder: 15-20 ms.
- Runtime/buffer/render overhead: 5-10 ms.

Local reshapes and scalar cache cleanup are already below the scale needed for that. The remaining path requires fused attention, decoder specialization, and a cache/runtime ABI that stops treating full K/V cache tensors as ordinary per-frame graph outputs.
