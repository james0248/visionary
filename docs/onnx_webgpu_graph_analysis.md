# ONNX WebGPU Graph Analysis

Date: 2026-04-29

Inputs inspected:

- ONNX assets: `webgpu_app/assets/*.onnx`
- Latest benchmark summary: `webgpu_app/bench/results/latest.json`
- Session profile: `webgpu_app/bench/results/profile_diagnostic_latest.json`
- Parsed summary: `webgpu_app/bench/results/session_profile_summary.json`

The session profile covers `breakout_dynamics_step_cached_b1_t1.onnx` with ORT Web 1.24.3 WebGPU. The s4 fused sample graph is four copies of the same step structure plus loop-output plumbing, so the boundary pattern scales roughly 4x there.

## Bottom Line

This is not primarily a jax2onnx dynamic-shape export problem. The exported assets have static I/O shapes, zero `Shape` nodes, zero `Size` nodes, and every `Reshape` shape input is an initializer. ORT WebGPU placement is the immediate cause: ORT assigns all 399 `Reshape` nodes in the profiled step graph to CPU, then inserts GPU/CPU copies around them.

The expensive work is not the CPU `Reshape` compute itself. In the profiled WebGPU run:

- Node events: 2,680
- WebGPU node events: 2,275
- CPU node events: 405
- CPU ops: `Reshape` 399, `Cast` 4, `Min` 2
- Memcpy events: 796
- `MemcpyToHost`: 394 events, 278.078 ms total
- `MemcpyFromHost`: 402 events, 44.020 ms total
- CPU op time: about 7.072 ms total
- Boundary copy time: about 322.098 ms total

## Static Shape Evidence

All current assets have concrete I/O dims and no runtime shape graph:

| Asset | Nodes | Shape | Size | Reshape | Const-shape Reshape | Reshape with `-1` |
|---|---:|---:|---:|---:|---:|---:|
| `breakout_dynamics_b1_t64.onnx` | 1,306 | 0 | 0 | 367 | 367 | 74 |
| `breakout_dynamics_prefill_cached_b1_t64.onnx` | 1,802 | 0 | 0 | 384 | 384 | 74 |
| `breakout_dynamics_step_cached_b1_t1.onnx` | 1,902 | 0 | 0 | 399 | 399 | 74 |
| `breakout_dynamics_cached_sample_step_b1_t1_s4.onnx` | 7,290 | 0 | 0 | 1,511 | 1,511 | 296 |
| `breakout_tokenizer_decode_z_b1_t1.onnx` | 437 | 0 | 0 | 124 | 124 | 25 |
| `breakout_tokenizer_decoder_b1_t1.onnx` | 437 | 0 | 0 | 124 | 124 | 26 |
| `breakout_tokenizer_decoder_b1_t64.onnx` | 437 | 0 | 0 | 124 | 124 | 26 |

The `-1` values are in static initializer shape tensors, not values computed by `Shape`/`Gather`/`Concat`. They still appear to trigger CPU placement for `Reshape` in ORT WebGPU.

## Profiled Step Graph Counts

`breakout_dynamics_step_cached_b1_t1.onnx` graph op counts:

- `Mul`: 409
- `Reshape`: 399
- `Add`: 219
- `Gemm`: 170
- `Div`: 120
- `ReduceMean`: 96
- `Sqrt`: 96
- `Concat`: 76
- `Sub`: 49
- `Split`: 48
- `Expand`: 48
- `Einsum`: 48
- `Slice`: 25
- `Softmax`: 24
- `QuickGelu`: 24
- `Squeeze`: 14
- `Transpose`: 12
- `Unsqueeze`: 12
- `Cast`: 5
- `Gather`: 5
- `Min`: 2
- `Less`: 1

ORT WebGPU placement in the profile:

- All `Expand`, `Einsum`, `Transpose`, `Concat`, `Gemm`, `Softmax`, and arithmetic kernels are on WebGPU.
- All `Reshape` kernels are on CPU.
- Four `Cast` and two `Min` scalar/index nodes are on CPU.
- `pred_z` producer `node_Reshape_3076` is CPU.
- `candidate_cache_length` producer `node_Min_3058` is CPU.
- `candidate_k_cache` and `candidate_v_cache` producer concats are WebGPU.

## Exact Boundary Patterns

The canonical boundary is:

`WebGPU producer -> MemcpyToHost -> CPU Reshape -> MemcpyFromHost -> WebGPU consumer`

The most important exact CPU `Reshape` patterns in `breakout_dynamics_step_cached_b1_t1.onnx` are:

| Count | Pattern | Shape | Example |
|---:|---|---|---|
| 36 | `Expand -> Reshape -> Einsum` | `(1,36,2,4,64) -> (1,36,8,64)` | `node_Reshape_135` |
| 36 | `Gemm -> Reshape -> Add` | `(36,256) -> (1,36,256)` | `node_Reshape_152` |
| 18 | `Concat -> Reshape -> Expand` | `(1,36,2,64) -> (1,36,2,1,64)` | `node_Reshape_128` |
| 18 | `Gemm -> Reshape -> Mul` | `(36,768) -> (1,36,768)` | `node_Reshape_169` |
| 18 | `Gemm -> Reshape -> QuickGelu` | `(36,768) -> (1,36,768)` | `node_Reshape_163` |
| 18 | `Mul -> Reshape -> Gemm` | `(1,36,768) -> (36,768)` | `node_Reshape_172` |
| 12 | `Squeeze -> Reshape -> Concat` | `(1,36,64,2,64) -> (36,64,2,64)` | `node_Reshape_420` |
| 12 | `Concat -> Reshape -> Expand` | `(36,65,2,64) -> (36,65,2,1,64)` | `node_Reshape_506` |
| 12 | `Expand -> Reshape -> Einsum` | `(36,65,2,4,64) -> (36,65,8,64)` | `node_Reshape_508` |
| 12 | `Gemm -> Reshape -> Add` | `(36,256) -> (36,1,256)` | `node_Reshape_521` |
| 6 | `Add -> Reshape -> Transpose` | `(1,36,256) -> (1,1,36,256)` | `node_Reshape_412` |
| 6 | `Transpose -> Reshape -> arithmetic` | `(1,36,1,256) -> (36,1,256)` | `node_Reshape_414` |
| 6 | `Concat -> Reshape -> Concat` | `(36,1,2,64) -> (1,36,1,2,64)` | `node_Reshape_545` |

The `Expand -> Reshape -> Einsum` rows are the clearest GQA materialization boundary. K/V heads are repeated from 2 KV heads to 8 query heads via `Expand` to a 5D tensor and `Reshape` back to the head dimension expected by `Einsum`.

Representative timeline windows:

- `Gemm/WebGPU (36,128) -> MemcpyToHost -> Reshape/CPU (36,128)->(1,36,2,1,64) -> MemcpyFromHost -> Expand/WebGPU`
- `Expand/WebGPU (1,36,2,1,64)->(1,36,2,4,64) -> MemcpyToHost -> Reshape/CPU (1,36,2,4,64)->(1,36,8,64) -> MemcpyFromHost -> Einsum/WebGPU`
- `Einsum/WebGPU (1,8,36,36)->(1,36,8,64) -> MemcpyToHost -> Reshape/CPU (1,36,8,64)->(36,512) -> MemcpyFromHost -> Gemm/WebGPU`
- `Concat/WebGPU (36,65,2,64) -> MemcpyToHost -> Reshape/CPU (36,65,2,64)->(36,65,2,1,64) -> MemcpyFromHost -> Expand/WebGPU`
- `Expand/WebGPU (36,65,2,1,64)->(36,65,2,4,64) -> MemcpyToHost -> Reshape/CPU (36,65,2,4,64)->(36,65,8,64) -> MemcpyFromHost -> Einsum/WebGPU`

## Memcpy Hot Shapes

Largest repeated copy groups:

| Count | Copy | Shape | Total |
|---:|---|---|---:|
| 48 | ToHost | `(36,128)` | 35.894 ms |
| 48 | ToHost | `(36,256)` | 33.658 ms |
| 48 | ToHost | `(36,768)` | 27.185 ms |
| 42 | ToHost | `(1,36,256)` | 24.140 ms |
| 18 | ToHost | `(1,36,8,64)` | 20.527 ms |
| 12 | ToHost | `(36,65,2,4,64)` | 18.424 ms |
| 12 | FromHost | `(36,65,8,64)` | 18.278 ms |
| 24 | ToHost | `(36,512)` | 17.845 ms |
| 36 | ToHost | `(1,36,2,4,64)` | 15.981 ms |
| 12 | ToHost | `(1,36,64,2,64)` | 15.559 ms |
| 18 | ToHost | `(1,36,2,64)` | 14.172 ms |
| 12 | ToHost | `(36,65,2,64)` | 9.996 ms |
| 18 | ToHost | `(36,1,256)` | 9.747 ms |

The `(36,65,...)` copies are from temporal cached attention: 36 tokens, cache length plus current token = 65, 2 KV heads, repeat factor 4, head dim 64.

## Why It Happens

The export wrapper currently decomposes attention into ordinary JAX reshapes/rearranges and `einsum`. In the non-`grouped_gqa` path it materializes repeated K/V heads:

```python
key = jnp.repeat(key, repeat, axis=-2)
value = jnp.repeat(value, repeat, axis=-2)
logits = jnp.einsum("bqhd,bkhd->bhqk", query, key) * scale
...
return jnp.einsum("bhqk,bkhd->bqhd", weights, value)
```

That lowers to the recurring:

`Reshape/Unsqueeze-like singleton insertion -> Expand -> Reshape -> Einsum`

ORT WebGPU supports `Expand` and `Einsum`, but places the connecting constant-shape `Reshape` on CPU. Since these tensors are much larger than the scalar shape tensors, the provider boundary dominates runtime.

The cached step wrapper also flattens and unflattens `(b,n)` and head dimensions around dense layers:

- `(1,36,256) <-> (36,256)`
- `(36,768) <-> (1,36,768)`
- `(1,36,8,64) <-> (36,512)`
- `(36,65,2,4,64) <-> (36,65,8,64)`

Those are behavior-preserving layout views in JAX, but in ORT WebGPU they become CPU execution-provider islands.

## Ranked Fixes

1. Avoid materializing repeated GQA heads before `Einsum`.

   Use the existing `grouped_gqa` export path as the first experiment, but inspect whether it lowers to WebGPU-supported `MatMul`/`Reshape` patterns without large `Expand -> Reshape -> Einsum` copies. The target is to remove the repeated-head tensors shaped `(1,36,2,4,64)` and `(36,65,2,4,64)` and compute attention grouped as `(b, kv, repeat, q, d)` instead.

   Expected win: removes the clearest GQA boundary pattern, including 36 step copies around `(1,36,2,4,64)->(1,36,8,64)` and 12 cached temporal copies around `(36,65,2,4,64)->(36,65,8,64)`.

2. Replace singleton-insertion `Reshape` nodes with `Unsqueeze` where possible.

   ORT WebGPU places `Unsqueeze` on WebGPU in this profile, while every `Reshape` is CPU. Rewrite wrapper expressions that only add a size-1 axis:

   - `(1,36,2,64) -> (1,36,2,1,64)`
   - `(36,65,2,64) -> (36,65,2,1,64)`
   - `(36,1,2,64) -> (1,36,1,2,64)`
   - `(1,36,256) -> (1,1,36,256)`

   Prefer `jnp.expand_dims`/explicit `None` indexing over einops `rearrange` when the transform only inserts singleton axes. This should preserve behavior and may lower to `Unsqueeze`, avoiding CPU placement.

3. Remove flatten/unflatten around dense blocks where the dense can operate on rank-3 tensors.

   Many boundaries are pure flatten/view steps before or after `Gemm`: `(1,36,256)->(36,256)`, `(36,256)->(1,36,256)`, `(36,768)->(1,36,768)`, `(1,36,8,64)->(36,512)`. If the wrapper can keep activations rank-3 and let `nn.Dense` lower as a batched matmul or equivalent WebGPU-supported op, the graph can avoid the CPU `Reshape` islands around projections and MLPs.

   This is a higher-risk rewrite because it can change the exact ONNX lowering around `Gemm`. Validate node placement and numerical parity after export.

4. Replace pack/unpack `Reshape` around head repetition with direct equations.

   If grouped GQA does not help, rewrite attention equations to avoid the 5D `Expand -> Reshape` materialization. For example, keep K/V as `(b,k,kv,d)` and Q as `(b,q,kv,repeat,d)`, then use grouped `matmul`/`einsum` without flattening `kv*repeat` until after the weighted value product.

   Desired lowering: no `(2,4)->8` reshape between WebGPU ops.

5. Defer or remove scalar CPU outputs from the WebGPU critical path.

   `candidate_cache_length` is produced by CPU `Min`, and the profile shows scalar `MemcpyToHost` around cache length/index logic. These are small compared with tensor copies, but they add scheduling boundaries. If runtime policy already knows the length advances by one and clamps at 64, consider keeping cache length as browser-side metadata and not as an ONNX output/input for the hot step graph.

6. Treat post-export graph surgery as a fallback, not the primary fix.

   A surgical ONNX pass can replace safe `Reshape` nodes with `Unsqueeze`/`Squeeze`/`Flatten`-like alternatives when shape changes are only singleton insertion/removal or batch flattening. This preserves model math but requires careful validation because ORT WebGPU placement is version-sensitive. It is less maintainable than wrapper-level lowering changes.

## Validation Plan For Fixes

For each rewrite/export:

1. Check graph static-shape invariants: no symbolic I/O dims, no `Shape`/`Size`, all intended shape tensors constant.
2. Run ORT validation against JAX for `pred_z`, cache tensors, and `cache_length`.
3. Run WebGPU session profiling and compare:
   - CPU node count
   - `MemcpyToHost`/`MemcpyFromHost` count and total time
   - presence of `Expand -> Reshape -> Einsum` boundaries
   - top memcpy shapes above
4. Benchmark `cached_step` and fused s4 `cached_sample_step`, because the s4 graph multiplies remaining boundaries.

