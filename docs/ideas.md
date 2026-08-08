# Ideas (not in any reference recipe)

- **Tube masking** (VideoMAE-style): share the MAE spatial mask across all frames
  of a clip so the decoder cannot inpaint a masked patch from neighboring frames,
  forcing the latent to carry it. Neither Dreamer 4 (masks per image,
  p~U(0,0.9)) nor Open Dreamer does this — both mask per frame. Implemented
  once, removed in favor of the reference recipe. Revisit if temporal flicker
  persists after tanh + per-frame MAE + SigReg removal.
