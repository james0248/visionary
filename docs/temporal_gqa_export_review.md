# Temporal GQA Export Review

## Scope

Reviewed the current temporal GQA export path in:

- `scripts/webgpu/export_dreamer4_onnx.py`
- `visionary/export/onnx_wrappers.py`
- prior local notes in `.codex/onnx_webgpu_progress.md` and `.codex/onnx_current_graph_status.md`

This report only covers the existing `--fused_temporal_gqa` and related grouped-attention paths.

## 1. How The Rewrite Works

There are two separate GQA-related paths.

`--grouped_gqa_attention` changes the JAX export lowering in `visionary/export/onnx_wrappers.py`. Instead of `jnp.repeat(key/value, repeat, axis=-2)` followed by generic `Einsum` attention, it groups query heads by KV head and computes grouped attention using reshapes/rearranges, `jnp.matmul`, softmax, and another `jnp.matmul`. This avoids explicit K/V repeat in Python-level math, but previous benchmarks showed it generated a larger/slower ONNX graph.

`--fused_temporal_gqa` is a post-export ONNX rewrite in `rewrite_cached_temporal_attention_to_gqa()`. It runs after normal export, ORT optimization, custom reshape rewrites, RMSNorm fusion, and CPU validation. It targets cached temporal attention islands shaped like the steady-state demo path:

- softmax output shape: `[36, 8, 1, 65]`
- query shape: `[36, 1, 8, 64]`
- concatenated key/value shape: `[36, 65, 2, 64]`

For each matched attention island, it replaces only the value-side `Einsum` node with:

- `Reshape(query)` to `[36, 1, 512]`
- `Reshape(current_key)` to `[36, 1, 128]`
- `Reshape(current_value)` to `[36, 1, 128]`
- `Transpose(past_key)` from `[36, 64, 2, 64]` to `[36, 2, 64, 64]`
- `Transpose(past_value)` likewise
- `com.microsoft::GroupQueryAttention`
- `Reshape(output)` back to `[36, 1, 8, 64]`

It hardcodes:

- `num_heads=8`
- `kv_num_heads=2`
- `scale=0.125`
- `seq_lens = [64] * 36`
- `total_sequence_length = [65]`

The rewrite is applied to several cached artifacts when `--fused_temporal_gqa` is enabled, including the preferred entry-cache steady-state artifact.

## 2. Likely Correctness Risks

The largest correctness risk is that the rewrite hardcodes full-cache steady-state semantics, but it is applied to more than just the final steady-state demo artifact. `seq_lens=[64]` and `total_sequence_length=[65]` are correct only when the cache is full and the current token is appended after 64 prior temporal positions. They are suspect for early-cache, partial-cache, or generic cached-step artifacts.

The matcher relies mostly on shapes and nearby op types. It does not verify the `Einsum` equations, the GQA repeat `Gather` indices, the attention scale source, or that the mask being removed is equivalent to the hardcoded `seq_lens` behavior. A same-shaped non-temporal island would be unlikely, but the current guard is still looser than it should be for a graph surgery pass.

The fused graph is not numerically validated after the rewrite. Validation happens before `--fused_temporal_gqa`; then the graph is changed and saved. The local comment says this is browser-targeted and not CPU-ORT validated. That leaves fused attention correctness dependent on smoke tests and visual output unless we add a deterministic raw-vs-fused comparison.

The head ordering must exactly match the existing repeat ordering. The current code assumes query heads are ordered as `kv0 repeat heads, kv1 repeat heads`, matching repeated indices `[0,0,0,0,1,1,1,1]`. This is probably correct for the current graph, but it should be asserted from the graph pattern rather than assumed.

The fixed constants make the pass config-fragile. `36`, `8`, `2`, `64`, and `0.125` should be derived from `dyn_shapes` or inferred shapes before the pass is considered robust.

## 3. Likely WebGPU Graph-Capture Blockers

The current fused rewrite reintroduces graph-level `Reshape` nodes. That is the biggest graph-capture blocker. The stable hot graph currently has `Reshape=0`; `GroupQueryAttention` fusion adds four standalone reshapes per fused attention island. Our previous analysis found graph-level `Reshape` falls back to CPU in ORT WebGPU, while `Squeeze`/`Unsqueeze` and rank-aware `Einsum` can stay on device.

The pass also adds two `Transpose` nodes per attention island for past K/V. `Transpose` is WebGPU-supported in the current graph, but it is still a separate dispatch and probably eats into any fused-attention win. If its shape is unsupported by ORT WebGPU for this contrib path, it can also break capture.

The old attention score path is not explicitly removed. The replacement dictionary swaps the value `Einsum`, but the original query-key `Einsum`, scale, mask add, softmax, and GQA-repeat gathers become dead unless the runtime prunes them. Since the fusion runs after offline ORT optimization, the saved graph can still contain dead nodes. This can hurt load time, diagnostics, and possibly provider partitioning.

`GroupQueryAttention` itself is a contrib op. It may be WebGPU-supported in the target ORT Web version, but graph capture is stricter than normal execution. The pass should treat this as unproven until the preferred artifact passes graph capture with zero CPU/provider-boundary nodes.

The GQA node produces present K/V outputs that are currently unused. Depending on ORT implementation, it may still allocate or compute them. If so, that overhead is pure waste in the current path.

## 4. Recommended Code Changes

First, restrict `--fused_temporal_gqa` to only the preferred steady-state demo artifact until correctness is proven. Do not apply the hardcoded `[64]/[65]` sequence semantics to generic cached-step or partial-cache artifacts. The initial target should be `DYNAMICS_CACHED_SAMPLE_APPEND_CONTEXT_SLIDE_ENTRY_NAME` only.

Second, add a post-fusion accuracy gate. Preferred order:

1. Run the unfused optimized artifact and fused artifact on the same deterministic inputs.
2. Compare `final_z`, `pred_z`, `candidate_k_entry`, and `candidate_v_entry`.
3. Use the existing raw artifact comparison style and fail export if max error exceeds the fp32 tolerance.

If Python ORT cannot execute `GroupQueryAttention`, add a Playwright/ORT-Web deterministic comparison that runs both artifacts in the browser with fixed inputs and writes a JSON diff.

Third, eliminate the four new graph-level `Reshape` nodes before expecting graph capture to work. Concrete approach:

- Make the query projection emit `[36, 1, 512]` directly for the fused GQA path instead of `[36, 1, 8, 64]`.
- Make current K/V projections emit `[36, 1, 128]` directly instead of `[36, 1, 2, 64]`.
- Make the GQA output stay flat as `[36, 1, 512]`.
- Rewrite the downstream output projection to consume the flat output directly, instead of restoring `[36, 1, 8, 64]`.

This should be implemented as a GQA-specific extension of the existing `rewrite_head_projection_reshapes_for_webgpu()` logic, not as more post-GQA `Reshape` cleanup.

Fourth, remove the past-K/V `Transpose` nodes by changing the fused-GQA cache ABI. For fused artifacts, store cache as BNSH layout:

- current layout: `[36, 64, 2, 64]`
- fused GQA layout: `[36, 2, 64, 64]`

That lets `GroupQueryAttention` consume past K/V directly. It also means the browser cache updater needs a GQA-specific path, but that is better than paying two transposes per attention block.

Fifth, add a pure ONNX dead-node pruning pass after GQA fusion. Walk backwards from graph outputs, keep only required nodes and initializers, then save. Do not rely on browser ORT to clean this up at session load time.

Sixth, strengthen the matcher before enabling benchmarks:

- verify `Einsum` equations for score and value paths;
- verify the scale equals `1 / sqrt(head_dim)`;
- verify GQA repeat gather indices match expected head ordering;
- verify key/value concat axis and input ordering;
- verify no explicit mask semantics are being dropped unless full-cache steady-state makes the mask all-valid.

Seventh, add an export-time graph-capture readiness check for the selected hot artifact. At minimum, fail or warn if the preferred artifact contains `Reshape`, unsupported contrib ops, or known CPU fallback ops after all rewrites.

## Bottom Line

The current `GroupQueryAttention` rewrite is conceptually targeting the right bottleneck: too many generic attention kernels. But as written, it reintroduces the exact standalone `Reshape` CPU fallback problem that the stable path already solved, and it is applied more broadly than its hardcoded full-cache assumptions justify.

The next useful implementation step is not another benchmark of the current fused path. It is a fused-GQA-specific graph shape rewrite that feeds `GroupQueryAttention` flat Q/K/V tensors without graph-level `Reshape`, keeps cache in GQA-native BNSH layout, prunes dead original attention nodes, and validates fused outputs against the unfused artifact before browser benchmarking.
