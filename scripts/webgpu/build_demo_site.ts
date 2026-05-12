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
import { buildDemoBrowserBundle } from './build_browser_entrypoints';

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
const assetBase =
  args.get('--asset-base') ??
  (copyAssets ? './assets' : '/webgpu_app/dream_arcade_assets/breakout');
const pacmanAssetBase =
  args.get('--pacman-asset-base') ??
  (copyAssets ? './assets/pacman' : siblingAssetBase(assetBase, 'breakout', 'pacman'));
const ortModule =
  args.get('--ort-module') ?? './vendor/onnxruntime-web/ort.webgpu.bundle.min.mjs';
const ortWasmBase = args.get('--ort-wasm-base') ?? './vendor/onnxruntime-web/';

const demoDir = resolve('webgpu_app/demo');
const ortDistDir = resolve('node_modules/onnxruntime-web/dist');
const assetDir = resolve(args.get('--asset-dir') ?? 'webgpu_app/dream_arcade_assets/breakout');
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
  const preferredExports = [
    manifest.demo_generation?.preferred_step_export,
    manifest.demo_generation?.preferred_full_cache_step_export,
    'breakout_tokenizer_decoder_b1_t1',
    'breakout_tokenizer_decode_z_b1_t1',
  ].filter(Boolean);
  return [
    ...new Set(
      preferredExports
        .map((name) => manifest.exports?.find((entry) => entry.name === name)?.path)
        .filter(Boolean),
    ),
  ];
}

function siblingAssetBase(base: string, fromName: string, toName: string) {
  const normalized = base.replace(/\/$/, '');
  if (normalized.endsWith(`/${fromName}`)) {
    return `${normalized.slice(0, -fromName.length)}${toName}`;
  }
  return `${normalized}/${toName}`;
}

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

for (const file of ['styles.css']) {
  copyFileSync(join(demoDir, file), join(outDir, file));
}
await buildDemoBrowserBundle(outDir);

for (const htmlFile of ['index.html', 'pacman.html']) {
  if (!existsSync(join(demoDir, htmlFile))) continue;
  const pageAssetBase = htmlFile === 'pacman.html' ? pacmanAssetBase : assetBase;
  let html = readFileSync(join(demoDir, htmlFile), 'utf8');
  html = html
    .replace(/data-asset-base="[^"]*"/, `data-asset-base="${pageAssetBase}"`)
    .replace(/data-ort-module="[^"]*"/, `data-ort-module="${ortModule}"`)
    .replace(/data-ort-wasm-base="[^"]*"/, `data-ort-wasm-base="${ortWasmBase}"`);
  writeFileSync(join(outDir, htmlFile), html);
}

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
