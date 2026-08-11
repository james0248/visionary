# Tokenizer training optimization log

Branch `tokenizer-opt`, profiled 2026-08-11 on a spot v5litepod-4 (us-west4-a;
no v6e capacity anywhere in any provisioning model that day). Per-device batch
matched to the planned run: 2 clips x 16 frames per chip, 137.3M params.
Numerics gate: `scripts/analysis/tokenizer_numerics.py` (fp32 fingerprint, rtol
2e-4) against a TPU baseline. MXU% = XLA cost-analysis FLOPs / step time /
(197 TFLOP/s bf16 per v5e chip).

## Baseline

312 ms/step, 3.2 steps/s, 19.2 TF/step, **7.8% MXU**.
Split: forward 60 ms, backward +223 ms, Muon update ~30 ms.

## Attribution (xprof op_profile + hlo_stats)

- Every top op is **HBM-bandwidth-bound** (XLA `bound_by: HBM`); matmul fusions
  are 73% of time at ~30% internal efficiency, ~55-78% HBM utilization.
- Largest recurring tensor class: attention scores `bf16[32,3,4,556,556]`
  (237 MB per instance) written and re-read by softmax and both backward
  matmuls in every one of the 20 layers. XLA:TPU has no auto flash attention,
  so these always round-trip HBM.
- LPIPS is only ~4% (ablation 312->300 ms). Remat ~8%. Muon ~10%.
- Batch sweep 2/4 per chip: time scales linearly, MXU flat ~7-8% — the
  workload is bandwidth-limited at every size, not batch-starved.

## Changes

| # | change | step time | MXU | numerics |
|---|---|---|---|---|
| 1 | remat: false (batch 2/chip fits without it) | 312 -> 287 ms (+8.7% sps) | 8.3% | unchanged (same math) |
| 2 | query-chunked attention (exact softmax, lax.map + checkpoint) | 330 ms (regression) | 6.9% | 1 metric at rel 2.2e-4 | **reverted** |
| 3 | XLA flash-attention flag | n/a — flag does not exist in this libtpu | | |
| 4 | splash attention, block 128 | 442 ms (regression) | 5.0% | — |
| 5 | splash attention, block 256/512 | 351 ms (regression) | 6.3% | — |
| 6 | splash attention, block 640 = whole padded seq (**landed**) | 287 -> 261 ms (+9.9% sps) | 8.4% | 4 metrics at rel 2.2-3.1e-4, all < 5e-4; loss rel 1e-6. Kernel accumulation-order reassociation. |

## Splash notes (branch `splash-attn`, jax 0.9.1 -> 0.10.2)

- Pallas needs libtpu <= 1 month old: forced the jax bump. XLA fingerprint
  identical across the bump (loss 0.13069603); XLA baseline step time
  unchanged (286.6 ms).
- The kernel must be constructed per trace (caching it leaks tracers) and
  wrapped in `shard_map` over the batch axes (Mosaic ops cannot be
  GSPMD-partitioned). Scripts set `transformer.SPLASH_MESH` / `SPLASH_AXIS`.
- Spatial seq is 556 (pad 640): tiled flash loses to XLA here — kernel
  overhead beats the saved score traffic. A single 640 block (fused
  attention, no tiling) is the only winning point.
- Expected 1.5-2.5x, got 1.10x: score round-trips were the largest single
  tensor class but not the majority of HBM traffic; the rest of the step
  (matmul fusions at ~30% efficiency) is untouched by attention fusion.

## Conclusions

1. Splash attention landed at +9.9% (287 -> 261 ms), not the estimated
   1.5-2.5x: at seq 556 the score tensors are a real but minority share of
   HBM traffic. Total banked vs original baseline: 312 -> 261 ms (+19.5% sps).
2. The remaining ceiling is the bandwidth-bound matmul fusions themselves —
   only bigger arithmetic intensity (larger dims) or faster HBM (v6e) moves
   them.
3. v6e has 2x the HBM bandwidth of v5e (1.6 TB/s), so the planned v6e-16 run
   should land roughly 2x faster per chip than these numbers at the same MXU%.
   Estimated ~130 ms/step at global batch 32 -> ~7.7 steps/s -> 100k steps in
   ~3.6 h (~$55-65 spot).
4. Raising per-chip batch does NOT improve efficiency here (flat MXU), so the
   small-batch-for-iteration-speed choice costs nothing in efficiency.
