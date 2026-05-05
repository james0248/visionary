# WebGPU Demo Deployment

## Recommendation

Use a static site host for the HTML/CSS/JS shell and object storage for the large model
artifacts.

The current demo payload that the browser actually needs is about 235 MiB:

- `breakout_dynamics_sample_append_context_cache_length_entry_b1_t1_s2.onnx`: 191 MiB
- `breakout_tokenizer_decode_z_b1_t1.onnx`: 28.9 MiB
- `breakout_demo_initial_cache.*`: 13.5 MiB
- manifests/context metadata: about 1.0 MiB

Cloudflare Pages is a good shell host, but not a model host, because Pages has a
25 MiB single-file asset limit. Put the model artifacts in Cloudflare R2 behind a
public custom domain, then point the demo's `assetBase` at that domain.

## Build

Build the publishable shell:

```sh
bun run demo:webgpu:build -- --out webgpu_app/dist --asset-base https://static.example.com/breakout
```

For a fully local static bundle, including the large model files:

```sh
bun run demo:webgpu:build -- --out webgpu_app/dist --copy-assets
```

The local bundle is useful for smoke testing, but it is not the preferred Git deploy
shape because the ONNX files are large generated artifacts.

## Runtime Knobs

The demo reads configuration from script data attributes, query parameters, or
`window.VISIONARY_DEMO_CONFIG`.

- `assetBase`: directory containing `breakout_onnx_manifest.json` and the referenced model artifacts.
- `ortModule`: ONNX Runtime Web module URL.
- `ortWasmBase`: directory containing ONNX Runtime WASM fallback files.
- `backend`: `auto`, `webgpu`, or `wasm`.
- `fps`: desired frame cap. `0` means uncapped and is the default.

Examples:

```text
/?assetBase=https://static.example.com/breakout&fps=30
/?backend=wasm&fps=0
```

## Caching

Serve immutable model artifacts with long browser cache headers:

```text
Cache-Control: public, max-age=31536000, immutable
Access-Control-Allow-Origin: *
```

When replacing artifacts, publish them under a versioned prefix such as
`/breakout/v2/` and update `assetBase`. That avoids stale model/cache mismatches.

The page shell should use cross-origin isolation headers so ONNX Runtime can use the
fastest WASM path when WebGPU is unavailable:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## Expected Behavior

- WebGPU is attempted first.
- If WebGPU is unavailable or session creation fails, the demo falls back to WASM.
- The target FPS setting is a cap. If inference is slower than the cap, the loop runs
  as fast as the backend can finish frames.
- Mobile users get touch controls for fire, left, right, and noop.

## References

- Cloudflare Pages limits: https://developers.cloudflare.com/pages/platform/limits/
- Cloudflare R2 pricing: https://developers.cloudflare.com/r2/pricing/
- Cloudflare Pages headers: https://developers.cloudflare.com/pages/configuration/headers/
- GitHub repository limits: https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits
- GitHub Pages limits: https://docs.github.com/enterprise-cloud@latest/pages/getting-started-with-github-pages/github-pages-limits
- Vercel limits: https://vercel.com/docs/limits/overview
