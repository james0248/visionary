#!/usr/bin/env node
import {
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { basename, join, resolve } from 'node:path';

const args = new Map();
for (let index = 2; index < process.argv.length; index += 1) {
  const key = process.argv[index];
  if (!key.startsWith('--')) continue;
  const next = process.argv[index + 1];
  if (next && !next.startsWith('--')) {
    args.set(key, next);
    index += 1;
  } else {
    args.set(key, 'true');
  }
}

const outDir = resolve(args.get('--out') ?? 'webgpu_app/dist');
const copyAssets = args.get('--copy-assets') === 'true';
const assetBase = args.get('--asset-base') ?? (copyAssets ? './assets' : '/webgpu_app/assets');
const ortModule =
  args.get('--ort-module') ?? './vendor/onnxruntime-web/ort.webgpu.bundle.min.mjs';
const ortWasmBase = args.get('--ort-wasm-base') ?? './vendor/onnxruntime-web/';

const demoDir = resolve('webgpu_app/demo');
const ortDistDir = resolve('node_modules/onnxruntime-web/dist');
const assetDir = resolve('webgpu_app/assets');
const baseAssets = [
  'breakout_onnx_manifest.json',
  'breakout_demo_context.json',
  'breakout_demo_initial_cache.json',
  'breakout_demo_initial_cache.cache_length.i32.bin',
  'breakout_demo_initial_cache.k_cache.f32.bin',
  'breakout_demo_initial_cache.v_cache.f32.bin',
  'breakout_tokenizer_decode_z_b1_t1.onnx',
];

function demoModelAssets() {
  const manifestPath = join(assetDir, 'breakout_onnx_manifest.json');
  if (!existsSync(manifestPath)) return [];
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const preferredStep =
    manifest.demo_generation?.preferred_step_export ??
    manifest.demo_generation?.preferred_steady_state_step_export;
  if (!preferredStep) return [];
  const exportSpec = manifest.exports?.find((entry) => entry.name === preferredStep);
  return exportSpec?.path ? [exportSpec.path] : [];
}

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

for (const file of ['styles.css', 'main.js', 'jax_noise.js']) {
  copyFileSync(join(demoDir, file), join(outDir, file));
}

let html = readFileSync(join(demoDir, 'index.html'), 'utf8');
html = html
  .replace(/data-asset-base="[^"]*"/, `data-asset-base="${assetBase}"`)
  .replace(/data-ort-module="[^"]*"/, `data-ort-module="${ortModule}"`)
  .replace(/data-ort-wasm-base="[^"]*"/, `data-ort-wasm-base="${ortWasmBase}"`);
writeFileSync(join(outDir, 'index.html'), html);

const vendorDir = join(outDir, 'vendor/onnxruntime-web');
mkdirSync(vendorDir, { recursive: true });
copyFileSync(
  join(ortDistDir, 'ort.webgpu.bundle.min.mjs'),
  join(vendorDir, 'ort.webgpu.bundle.min.mjs'),
);
for (const file of readdirSync(ortDistDir)) {
  if (/^ort-wasm-simd-threaded(?!\.asyncify).*\.(mjs|wasm)$/.test(file)) {
    copyFileSync(join(ortDistDir, file), join(vendorDir, file));
  }
}

if (copyAssets) {
  const outAssetDir = join(outDir, 'assets');
  mkdirSync(outAssetDir, { recursive: true });
  const minimalAssets = [...baseAssets, ...demoModelAssets()];
  for (const file of minimalAssets) {
    const source = join(assetDir, file);
    if (!existsSync(source)) {
      throw new Error(`Missing demo asset ${source}`);
    }
    cpSync(source, join(outAssetDir, basename(file)));
  }
}

writeFileSync(
  join(outDir, '_headers'),
  `/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp

/vendor/*
  Cache-Control: public, max-age=31536000, immutable

/assets/*
  Cache-Control: public, max-age=31536000, immutable
  Access-Control-Allow-Origin: *
`,
);

console.log(`Wrote ${outDir}`);
console.log(`Asset base: ${assetBase}`);
