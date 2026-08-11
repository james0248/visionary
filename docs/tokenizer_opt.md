# Tokenizer training optimization log

Branch `tokenizer-opt`, profiled on an on-demand v6e-1 (us-central1-b), 3-hour
budget. Per-device batch matched to the planned v6e-16 run: batch 2 x 16 frames
per chip. Numerical consistency is enforced by
`scripts/analysis/tokenizer_numerics.py` (fp32 fingerprint of loss, recon,
latent, and per-module grad norms; run `--save` on main, `--check` after every
change, rtol 2e-4).

MXU utilization = XLA cost-analysis FLOPs per step / measured step time /
918 TFLOP/s (v6e bf16 peak).

## Baseline

(pending profile)

## Changes

| # | change | step time | MXU | numerics |
|---|---|---|---|---|
