# Visionary

World models that you can play in your browser. Small Dreamer 4–style
(~7M parameter) models are trained to imagine Atari games, exported to ONNX, and
run locally in the browser via ONNX Runtime Web (WebGPU, with a WASM fallback).

Live demo: **[Dream Atari](https://www.hyeonseokjung.com/dream-atari)**

## Layout

| Path | What's in it |
| --- | --- |
| `visionary/` | JAX/Flax model code — `tokenizer.py`, `dynamics.py`, `transformer.py`, dataset, LPIPS |
| `visionary/export/`, `webgpu_app/export/` | ONNX export and graph-optimization passes (WebGPU + WASM profiles) |
| `webgpu_app/demo/` | Browser demo (TypeScript, served with Bun) |
| `scripts/` | Training entrypoints (`train_tokenizer.py`, `train_dynamics.py`) and configs |
| `cloud/` | TPU / cloud setup helpers |
| `docs/` | ONNX optimization notes and deployment guide |

## Setup

```sh
uv sync                          # Python deps (training + export)
cd webgpu_app && bun install     # web demo deps
```

## Training

```sh
uv run python scripts/train_tokenizer.py
uv run python scripts/train_dynamics.py
```

The transformer ends in an RMSNorm, so the parameter tree does not match
checkpoints trained before that was added. To load an older Atari checkpoint or
reproduce the shipped demo weights, check out `abe92ca`.

## Export to ONNX

```sh
# WebGPU build
uv run python webgpu_app/export/export_dreamer4_onnx.py --export_target webgpu

# WASM build (what the public demo ships)
uv run python webgpu_app/export/export_dreamer4_onnx.py --export_target wasm
```

The two targets use different graph passes because ORT WebGPU and ORT WASM
support different fused/layout ops. See `docs/onnx_webgpu_progress.md` for the
optimization log and `docs/webgpu_demo_deploy.md` for deployment.

## Run the demo locally

```sh
cd webgpu_app
bun run demo:webgpu        # serves at http://127.0.0.1:4173
```

Runtime knobs (query params): `?backend=webgpu|wasm|auto`, `wasmNumThreads`,
`fps`, `assetBase`. WebGPU is tried first and falls back to WASM.
