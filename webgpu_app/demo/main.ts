import { NormalNoiseGenerator } from './jax_noise';

const params = new URLSearchParams(window.location.search);
const scriptElement = document.getElementById('visionary-demo-main') as HTMLElement | null;
const globalConfig = window.VISIONARY_DEMO_CONFIG ?? {};

function configValue(name, fallback) {
  return params.get(name) ?? scriptElement?.dataset?.[name] ?? globalConfig[name] ?? fallback;
}

function resolveUrl(value) {
  return new URL(value, window.location.href).href;
}

function resolveBaseUrl(value) {
  return resolveUrl(value).replace(/\/$/, '');
}

const ASSET_DIR = resolveBaseUrl(configValue('assetBase', '/webgpu_app/assets'));
const MANIFEST_URL = `${ASSET_DIR}/breakout_onnx_manifest.json`;
const CONTEXT_URL = `${ASSET_DIR}/breakout_demo_context.json`;
const INITIAL_CACHE_URL = `${ASSET_DIR}/breakout_demo_initial_cache.json`;
const DEFAULT_TARGET_FPS = 0;
const DEFAULT_ORT_MODULE = `/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs`;
const DEFAULT_ORT_WASM_BASE = `/node_modules/onnxruntime-web/dist/`;
const ort = await import(resolveUrl(configValue('ortModule', DEFAULT_ORT_MODULE)));

ort.env.wasm ??= {};
ort.env.wasm.wasmPaths = resolveUrl(configValue('ortWasmBase', DEFAULT_ORT_WASM_BASE));
ort.env.webgpu ??= {};
ort.env.webgpu.powerPreference = 'high-performance';

const ACTIONS = {
  noop: 0,
  fire: 1,
  right: 2,
  left: 3,
};

const ACTION_LABELS = {
  0: 'noop',
  1: 'fire',
  2: 'right',
  3: 'left',
};

const CONTEXT_TENSOR_SIZE = 32 * 32;
const float32Scratch = new Float32Array(1);
const uint32Scratch = new Uint32Array(float32Scratch.buffer);

const elements = {
  canvas: document.getElementById('frame') as HTMLCanvasElement,
  status: document.getElementById('status'),
  start: document.getElementById('start') as HTMLButtonElement,
  reset: document.getElementById('reset') as HTMLButtonElement,
  fps: document.getElementById('fps'),
  action: document.getElementById('action'),
  frameCount: document.getElementById('frame-count'),
  latency: document.getElementById('latency'),
  context: document.getElementById('context'),
  backend: document.getElementById('backend'),
  payload: document.getElementById('payload'),
  targetFps: document.getElementById('target-fps') as HTMLSelectElement,
  loadFill: document.getElementById('load-progress-fill'),
  loadText: document.getElementById('load-progress-text'),
  loadLog: document.getElementById('load-log'),
  keys: {
    noop: document.getElementById('key-noop'),
    fire: document.getElementById('key-fire'),
    left: document.getElementById('key-left'),
    right: document.getElementById('key-right'),
  },
};

const ctx = elements.canvas.getContext('2d', { alpha: false });
ctx.imageSmoothingEnabled = false;

let runtime = null;
let running = false;
let frameCount = 0;
let currentAction = ACTIONS.noop;
let lastFrameTime = performance.now();
let noiseGenerator = new NormalNoiseGenerator(0);
let targetFps = parseTargetFps(configValue('fps', DEFAULT_TARGET_FPS));
const throttleMbps = Number(params.get('throttleMbps') ?? 0);
let loadEvents = [];

function parseTargetFps(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : DEFAULT_TARGET_FPS;
}

function setStatus(message) {
  elements.status.textContent = message;
}

function formatBytes(bytes) {
  if (bytes == null) return '';
  const mib = bytes / (1024 * 1024);
  return `${mib.toFixed(mib >= 100 ? 0 : 1)} MiB`;
}

function formatMs(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(0)} ms`;
}

function recordLoadEvent(label, elapsedMs, bytes = null) {
  loadEvents = [...loadEvents, { label, elapsedMs, bytes }];
  const totalBytes = loadEvents.reduce((total, event) => total + (event.bytes ?? 0), 0);
  elements.payload.textContent = formatBytes(totalBytes);
  elements.loadLog.replaceChildren(
    ...loadEvents.map((event) => {
      const item = document.createElement('li');
      const size = event.bytes == null ? '' : ` · ${formatBytes(event.bytes)}`;
      item.textContent = `${event.label}: ${formatMs(event.elapsedMs)}${size}`;
      return item;
    }),
  );
}

function updateLoadProgress(label, received, total) {
  if (!elements.loadFill || !elements.loadText) return;
  const percent = total > 0 ? Math.min(100, (received / total) * 100) : 0;
  elements.loadFill.style.width = `${percent}%`;
  const size = total > 0 ? `${formatBytes(received)} / ${formatBytes(total)}` : formatBytes(received);
  elements.loadText.textContent = `${label} ${size}`;
}

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function dtypeArray(dtype) {
  if (dtype === 'float32') return Float32Array;
  if (dtype === 'float16') return Uint16Array;
  if (dtype === 'int32') return Int32Array;
  if (dtype === 'uint8') return Uint8Array;
  throw new Error(`Unsupported artifact dtype ${dtype}`);
}

function mul(shape) {
  return shape.reduce((total, value) => total * value, 1);
}

function tensorByteLength(dtype, shape) {
  const bytesPerElement =
    dtype === 'float32' || dtype === 'int32' || dtype === 'uint32'
      ? 4
      : dtype === 'float16'
        ? 2
        : dtype === 'uint8'
          ? 1
          : 8;
  return mul(shape) * bytesPerElement;
}

function tensorDataBytes(tensor) {
  const data = tensor.data;
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

function createGpuTensorFromCpu(device, tensor) {
  const byteLength = Math.max(16, tensorByteLength(tensor.type, tensor.dims));
  const buffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(tensorDataBytes(tensor));
  buffer.unmap();
  return ort.Tensor.fromGpuBuffer(buffer, {
    dataType: tensor.type,
    dims: tensor.dims,
    dispose: () => buffer.destroy(),
  });
}

function float32ToFloat16Bits(value) {
  float32Scratch[0] = value;
  const bits = uint32Scratch[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  if (exponent === 0xff) return sign | (mantissa ? 0x7e00 : 0x7c00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const subnormal = (mantissa | 0x800000) >>> (1 - halfExponent);
    return sign | ((subnormal + 0x1000) >>> 13);
  }
  return sign | (halfExponent << 10) | ((mantissa + 0x1000) >>> 13);
}

function float16BitsToFloat32(bits) {
  const sign = (bits & 0x8000) << 16;
  let exponent = (bits >>> 10) & 0x1f;
  let mantissa = bits & 0x03ff;
  if (exponent === 0) {
    if (mantissa === 0) {
      uint32Scratch[0] = sign;
      return float32Scratch[0];
    }
    exponent = 1;
    while ((mantissa & 0x0400) === 0) {
      mantissa <<= 1;
      exponent -= 1;
    }
    mantissa &= 0x03ff;
  } else if (exponent === 0x1f) {
    uint32Scratch[0] = sign | 0x7f800000 | (mantissa << 13);
    return float32Scratch[0];
  }
  uint32Scratch[0] = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  return float32Scratch[0];
}

function isNativeFloat16Array(values) {
  return typeof Float16Array !== 'undefined' && values instanceof Float16Array;
}

function floatTensorValue(tensor, index) {
  if (tensor.type !== 'float16' || isNativeFloat16Array(tensor.data)) {
    return tensor.data[index];
  }
  return float16BitsToFloat32(tensor.data[index]);
}

function makeFloatTensor(dtype, values, shape) {
  if (dtype === 'float16') {
    const packed = new Uint16Array(values.length);
    for (let index = 0; index < values.length; index += 1) {
      packed[index] = float32ToFloat16Bits(values[index]);
    }
    return new ort.Tensor('float16', packed, shape);
  }
  return new ort.Tensor('float32', new Float32Array(values), shape);
}

async function fetchBytes(url, label) {
  const started = performance.now();
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  const total = Number(response.headers.get('content-length') ?? 0);
  let received = 0;

  if (!response.body) {
    const buffer = await response.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    recordLoadEvent(label, performance.now() - started, bytes.byteLength);
    updateLoadProgress(label, bytes.byteLength, bytes.byteLength);
    return bytes;
  }

  const reader = response.body.getReader();
  const chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    updateLoadProgress(label, received, total);
    if (throttleMbps > 0) {
      const targetElapsed = (received * 8 * 1000) / (throttleMbps * 1_000_000);
      const actualElapsed = performance.now() - started;
      if (targetElapsed > actualElapsed) {
        await delay(targetElapsed - actualElapsed);
      }
    }
  }

  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  recordLoadEvent(label, performance.now() - started, received);
  updateLoadProgress(label, received, total || received);
  return bytes;
}

async function fetchJson(url, label) {
  const bytes = await fetchBytes(url, label);
  return JSON.parse(new TextDecoder().decode(bytes));
}

async function fetchTensorFromArtifact(baseUrl, spec, label) {
  const bytes = await fetchBytes(`${baseUrl}/${spec.path}`, label);
  const ArrayType = dtypeArray(spec.dtype);
  return new ort.Tensor(spec.dtype, new ArrayType(bytes.buffer), spec.shape);
}

function findExport(manifest, name) {
  const entry = manifest.exports.find((item) => item.name === name);
  if (!entry) throw new Error(`Missing export ${name}`);
  return entry;
}

function findFirstExport(manifest, names) {
  for (const name of names.filter(Boolean)) {
    const entry = manifest.exports.find((item) => item.name === name);
    if (entry) return entry;
  }
  throw new Error(`Missing exports: ${names.filter(Boolean).join(', ')}`);
}

function outputName(spec, preferred) {
  if (spec.outputs?.[preferred]) return preferred;
  return Object.keys(spec.outputs ?? {})[0];
}

function requiredOutputName(spec, preferred) {
  if (spec.outputs?.[preferred]) return preferred;
  throw new Error(`${spec.name} must output ${preferred}`);
}

function optionalOutputName(spec, preferred) {
  return spec.outputs?.[preferred] ? preferred : null;
}

function formatShape(shape) {
  return `[${shape.join(',')}]`;
}

function sameShape(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function assertTensorMatchesSpec(tensor, spec, label, name) {
  if (!spec) return;
  if (tensor.type !== spec.dtype || !sameShape(tensor.dims, spec.shape)) {
    throw new Error(
      `${label} does not match the exported ${name} input. ` +
        `Artifact has ${tensor.type} ${formatShape(tensor.dims)}, ` +
        `model expects ${spec.dtype} ${formatShape(spec.shape)}. ` +
        'Regenerate it with `uv run python scripts/webgpu/create_demo_initial_cache.py --asset_dir webgpu_app/assets --overwrite`.',
    );
  }
}

function validateInitialCache(stepSpec, initialK, initialV, initialLength) {
  const inputs = stepSpec.inputs ?? {};
  assertTensorMatchesSpec(initialK, inputs.k_cache, 'Initial K cache', 'k_cache');
  assertTensorMatchesSpec(initialV, inputs.v_cache, 'Initial V cache', 'v_cache');
  assertTensorMatchesSpec(initialLength, inputs.cache_length, 'Initial cache length', 'cache_length');
}

function stepNamesForSpec(stepSpec) {
  return {
    finalZ: requiredOutputName(stepSpec, 'final_z'),
    k: requiredOutputName(stepSpec, 'candidate_k_entry'),
    v: requiredOutputName(stepSpec, 'candidate_v_entry'),
    length: optionalOutputName(stepSpec, 'candidate_cache_length'),
  };
}

async function createSession(spec, label, modelBytes, backend, options = {}) {
  const started = performance.now();
  setStatus(`Compiling ${label} · ${backend}`);
  const backendOptions = backend === 'webgpu' ? options : {};
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: backend === 'webgpu' ? [{ name: 'webgpu' }] : ['wasm'],
    graphOptimizationLevel: 'basic',
    externalData: (spec.external_data ?? []).map((entry) => ({
      path: entry.path,
      data: `${ASSET_DIR}/${entry.path}`,
    })),
    ...backendOptions,
  });
  recordLoadEvent(`${label} ${backend} compile`, performance.now() - started);
  return session;
}

function randomNormalTensor(shape, dtype = 'float32') {
  const size = shape.reduce((total, value) => total * value, 1);
  return makeFloatTensor(dtype, noiseGenerator.tensorData(size), shape);
}

function cacheAttentionMaskTensor(inputSpec, cacheLength, contextLength) {
  const validLength = Math.min(Math.max(cacheLength, 0), contextLength);
  const values = new Float32Array(mul(inputSpec.shape));
  for (let index = 0; index < values.length; index += 1) {
    const position = index % (contextLength + 1);
    values[index] = position < validLength || position === contextLength ? 1 : 0;
  }
  return makeFloatTensor(inputSpec.dtype, values, inputSpec.shape);
}

function stepPositionFeeds(stepSpec, cacheLength, contextLength, cacheLengthTensor) {
  const inputs = stepSpec.inputs ?? {};
  const feeds: Record<string, unknown> = {};
  if (inputs.sample_position_index) {
    feeds.sample_position_index = new ort.Tensor(
      'int32',
      new Int32Array([Math.min(cacheLength, contextLength)]),
      inputs.sample_position_index.shape,
    );
  }
  if (inputs.context_position_index) {
    feeds.context_position_index = new ort.Tensor(
      'int32',
      new Int32Array([Math.min(cacheLength, contextLength - 1)]),
      inputs.context_position_index.shape,
    );
  }
  if (inputs.attention_mask) {
    feeds.attention_mask = cacheAttentionMaskTensor(inputs.attention_mask, cacheLength, contextLength);
  }
  if (inputs.cache_length) {
    feeds.cache_length = cacheLengthTensor;
  }
  return feeds;
}

function advanceCacheLength(cacheLengthTensor, contextLength) {
  cacheLengthTensor.data[0] = Math.min(cacheLengthTensor.data[0] + 1, contextLength);
}

function disposeGpuTensor(tensor) {
  if (tensor?.location === 'gpu-buffer') {
    tensor.dispose();
  }
}

function disposeCache(cache) {
  disposeGpuTensor(cache?.k);
  disposeGpuTensor(cache?.v);
}

function contextFrameTensor(tensor, frameIndex, dtype = 'float32') {
  const start = frameIndex * CONTEXT_TENSOR_SIZE;
  const end = start + CONTEXT_TENSOR_SIZE;
  return makeFloatTensor(dtype, tensor.data.slice(start, end), [1, 1, 32, 32]);
}

function renderPixelTensor(tensor, frameIndex) {
  const [frames, height, width, channels] = tensor.dims;
  const clampedFrame = Math.max(0, Math.min(frames - 1, frameIndex));
  const sourceFrameOffset = clampedFrame * height * width * channels;
  const image = new ImageData(width, height);
  for (let index = 0; index < height * width; index += 1) {
    const source = sourceFrameOffset + index * channels;
    const target = index * 4;
    image.data[target] = tensor.data[source];
    image.data[target + 1] = tensor.data[source + 1];
    image.data[target + 2] = tensor.data[source + 2];
    image.data[target + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

function patchesToImageData(patchesTensor, preprocessor) {
  const width = preprocessor.image_width;
  const height = preprocessor.image_height;
  const patchSize = preprocessor.patch_size;
  const xLen = preprocessor.x_len;
  const yLen = preprocessor.y_len;
  const channels = preprocessor.num_channels;
  const patchDim = preprocessor.patch_dim;
  const image = new ImageData(width, height);

  for (let py = 0; py < yLen; py += 1) {
    for (let px = 0; px < xLen; px += 1) {
      const patchIndex = py * xLen + px;
      const patchOffset = patchIndex * patchDim;
      for (let iy = 0; iy < patchSize; iy += 1) {
        const y = py * patchSize + iy - preprocessor.pad_width[0];
        if (y < 0 || y >= height) continue;
        for (let ix = 0; ix < patchSize; ix += 1) {
          const x = px * patchSize + ix - preprocessor.pad_width[1];
          if (x < 0 || x >= width) continue;
          const source = patchOffset + (iy * patchSize + ix) * channels;
          const target = (y * width + x) * 4;
          const r = floatTensorValue(patchesTensor, source);
          const g = floatTensorValue(patchesTensor, source + 1);
          const b = floatTensorValue(patchesTensor, source + 2);
          image.data[target] = Math.max(0, Math.min(255, Math.round(r * 255)));
          image.data[target + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
          image.data[target + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
          image.data[target + 3] = 255;
        }
      }
    }
  }
  return image;
}

function cacheContractFromSpec(spec, manifest) {
  const cacheSpec = spec.inputs?.k_cache;
  const entrySpec = spec.outputs?.candidate_k_entry;
  if (!cacheSpec || !entrySpec) {
    throw new Error('Entry-cache update requires k_cache input and candidate_k_entry output specs.');
  }
  if (cacheSpec.dtype !== 'float32' || entrySpec.dtype !== 'float32') {
    throw new Error('Entry-cache update currently supports float32 caches only.');
  }
  const cacheLayout =
    manifest.cache_contract?.tensors?.k_cache?.layout ?? 'layer_batch_token_time_head_dim';
  let layers;
  let batch;
  let tokens;
  let contextLength;
  let heads;
  let headDim;
  if (cacheLayout === 'layer_batch_token_head_time_dim') {
    [layers, batch, tokens, heads, contextLength, headDim] = cacheSpec.shape;
  } else {
    [layers, batch, tokens, contextLength, heads, headDim] = cacheSpec.shape;
  }
  return {
    cacheLayout,
    cacheSpec,
    contextLength,
    batch,
    entrySpec,
    halfHeadDim: headDim / 2,
    headDim,
    heads,
    layers,
    tokens,
  };
}

function createEntryCacheUpdater(device, spec, manifest) {
  const {
    cacheLayout,
    contextLength,
    batch,
    halfHeadDim,
    headDim,
    heads,
    layers,
    tokens,
  } = cacheContractFromSpec(spec, manifest);
  const ropeBase = Number(manifest.dynamics?.rope_base ?? manifest.dynamics?.base ?? 10000);
  const workgroupSize = 64;
  const makeReadonlyBuffer = (label, values) => {
    const buffer = device.createBuffer({
      label,
      size: Math.max(16, values.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(values.buffer));
    buffer.unmap();
    return buffer;
  };
  const cosValues = new Float32Array(halfHeadDim);
  const sinValues = new Float32Array(halfHeadDim);
  for (let dim = 0; dim < halfHeadDim; dim += 1) {
    const theta = 1 / (ropeBase ** (dim / halfHeadDim));
    cosValues[dim] = Math.cos(theta);
    sinValues[dim] = Math.sin(theta);
  }
  const cosBuffer = makeReadonlyBuffer('visionary-entry-cache-cos', cosValues);
  const sinBuffer = makeReadonlyBuffer('visionary-entry-cache-sin', sinValues);
  const slotBuffer = device.createBuffer({
    label: 'visionary-entry-cache-slot',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const shader = device.createShaderModule({
    label: 'visionary-entry-cache-fill-slide-rebase',
    code: `
const LAYERS: u32 = ${layers}u;
const BATCH: u32 = ${batch}u;
const TOKENS: u32 = ${tokens}u;
const CONTEXT: u32 = ${contextLength}u;
const CONTEXT_MINUS_ONE: u32 = ${contextLength - 1}u;
const HEADS: u32 = ${heads}u;
const HEAD_DIM: u32 = ${headDim}u;
const HALF_HEAD_DIM: u32 = ${halfHeadDim}u;

@group(0) @binding(0) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(1) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(2) var<storage, read> k_entry: array<f32>;
@group(0) @binding(3) var<storage, read> v_entry: array<f32>;
@group(0) @binding(4) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(5) var<storage, read> sin_cache: array<f32>;

struct Params {
  slot: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(6) var<uniform> params: Params;

fn cache_index(layer: u32, batch: u32, token: u32, time: u32, head: u32, dim: u32) -> u32 {
  ${
    cacheLayout === 'layer_batch_token_head_time_dim'
      ? 'return (((((layer * BATCH + batch) * TOKENS + token) * HEADS + head) * CONTEXT + time) * HEAD_DIM + dim);'
      : 'return (((((layer * BATCH + batch) * TOKENS + token) * CONTEXT + time) * HEADS + head) * HEAD_DIM + dim);'
  }
}

fn entry_index(layer: u32, batch: u32, token: u32, head: u32, dim: u32) -> u32 {
  return ((((layer * BATCH + batch) * TOKENS + token) * HEADS + head) * HEAD_DIM + dim);
}

@compute @workgroup_size(${workgroupSize})
fn fill(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let total = LAYERS * BATCH * TOKENS * HEADS * HEAD_DIM;
  if (idx >= total) {
    return;
  }

  let dim = idx % HEAD_DIM;
  var remaining = idx / HEAD_DIM;
  let head = remaining % HEADS;
  remaining = remaining / HEADS;
  let token = remaining % TOKENS;
  remaining = remaining / TOKENS;
  let batch = remaining % BATCH;
  let layer = remaining / BATCH;
  let dst_time = min(params.slot, CONTEXT_MINUS_ONE);
  let src = entry_index(layer, batch, token, head, dim);
  let dst = cache_index(layer, batch, token, dst_time, head, dim);
  k_cache[dst] = k_entry[src];
  v_cache[dst] = v_entry[src];
}

@compute @workgroup_size(${workgroupSize})
fn slide(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let key_total = LAYERS * BATCH * TOKENS * HEADS * HALF_HEAD_DIM;
  let value_total = LAYERS * BATCH * TOKENS * HEADS * HEAD_DIM;

  if (idx < key_total) {
    let half_dim = idx % HALF_HEAD_DIM;
    var remaining = idx / HALF_HEAD_DIM;
    let head = remaining % HEADS;
    remaining = remaining / HEADS;
    let token = remaining % TOKENS;
    remaining = remaining / TOKENS;
    let batch = remaining % BATCH;
    let layer = remaining / BATCH;

    let cos_theta = cos_cache[half_dim];
    let sin_theta = sin_cache[half_dim];
    for (var time = 0u; time < CONTEXT_MINUS_ONE; time = time + 1u) {
      let src_left = cache_index(layer, batch, token, time + 1u, head, half_dim);
      let src_right = cache_index(layer, batch, token, time + 1u, head, HALF_HEAD_DIM + half_dim);
      let dst_left = cache_index(layer, batch, token, time, head, half_dim);
      let dst_right = cache_index(layer, batch, token, time, head, HALF_HEAD_DIM + half_dim);
      let left = k_cache[src_left];
      let right = k_cache[src_right];
      k_cache[dst_left] = left * cos_theta + right * sin_theta;
      k_cache[dst_right] = right * cos_theta - left * sin_theta;
    }

    let src_entry_left = entry_index(layer, batch, token, head, half_dim);
    let src_entry_right = entry_index(layer, batch, token, head, HALF_HEAD_DIM + half_dim);
    let dst_entry_left = cache_index(layer, batch, token, CONTEXT_MINUS_ONE, head, half_dim);
    let dst_entry_right = cache_index(layer, batch, token, CONTEXT_MINUS_ONE, head, HALF_HEAD_DIM + half_dim);
    k_cache[dst_entry_left] = k_entry[src_entry_left];
    k_cache[dst_entry_right] = k_entry[src_entry_right];
  }

  if (idx < value_total) {
    let dim = idx % HEAD_DIM;
    var remaining = idx / HEAD_DIM;
    let head = remaining % HEADS;
    remaining = remaining / HEADS;
    let token = remaining % TOKENS;
    remaining = remaining / TOKENS;
    let batch = remaining % BATCH;
    let layer = remaining / BATCH;

    for (var time = 0u; time < CONTEXT_MINUS_ONE; time = time + 1u) {
      let src = cache_index(layer, batch, token, time + 1u, head, dim);
      let dst = cache_index(layer, batch, token, time, head, dim);
      v_cache[dst] = v_cache[src];
    }
    let src_entry = entry_index(layer, batch, token, head, dim);
    let dst_entry = cache_index(layer, batch, token, CONTEXT_MINUS_ONE, head, dim);
    v_cache[dst_entry] = v_entry[src_entry];
  }
}
`,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'visionary-entry-cache-slide-rebase-bindings',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const fillPipeline = device.createComputePipeline({
    label: 'visionary-entry-cache-fill',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: shader, entryPoint: 'fill' },
  });
  const slidePipeline = device.createComputePipeline({
    label: 'visionary-entry-cache-slide-rebase',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: shader, entryPoint: 'slide' },
  });
  const fillDispatchCount = Math.ceil(
    (layers * batch * tokens * heads * headDim) / workgroupSize,
  );
  const slideDispatchCount = Math.ceil(
    Math.max(
      layers * batch * tokens * heads * halfHeadDim,
      layers * batch * tokens * heads * headDim,
    ) / workgroupSize,
  );
  let cachedBindGroup = null;
  let cachedBuffers = null;

  return {
    update(cache, kEntry, vEntry, cacheLength) {
      const logicalLength = cacheLength?.data?.[0] ?? contextLength;
      const slot = Math.min(Math.max(logicalLength, 0), contextLength - 1);
      device.queue.writeBuffer(slotBuffer, 0, new Uint32Array([slot, 0, 0, 0]));
      const buffers = [cache.k.gpuBuffer, cache.v.gpuBuffer, kEntry.gpuBuffer, vEntry.gpuBuffer];
      if (!cachedBindGroup || !cachedBuffers?.every((buffer, index) => buffer === buffers[index])) {
        cachedBindGroup = device.createBindGroup({
          label: 'visionary-entry-cache-slide-rebase-bind-group',
          layout: bindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: cache.k.gpuBuffer } },
            { binding: 1, resource: { buffer: cache.v.gpuBuffer } },
            { binding: 2, resource: { buffer: kEntry.gpuBuffer } },
            { binding: 3, resource: { buffer: vEntry.gpuBuffer } },
            { binding: 4, resource: { buffer: cosBuffer } },
            { binding: 5, resource: { buffer: sinBuffer } },
            { binding: 6, resource: { buffer: slotBuffer } },
          ],
        });
        cachedBuffers = buffers;
      }
      const useFill = logicalLength < contextLength;
      const encoder = device.createCommandEncoder({ label: 'visionary-entry-cache-update' });
      const pass = encoder.beginComputePass({
        label: useFill ? 'visionary-entry-cache-fill' : 'visionary-entry-cache-slide-rebase',
      });
      pass.setPipeline(useFill ? fillPipeline : slidePipeline);
      pass.setBindGroup(0, cachedBindGroup);
      pass.dispatchWorkgroups(useFill ? fillDispatchCount : slideDispatchCount);
      pass.end();
      device.queue.submit([encoder.finish()]);
      return cache;
    },
  };
}

function createCpuEntryCacheUpdater(spec, manifest) {
  const {
    cacheLayout,
    contextLength,
    batch,
    halfHeadDim,
    headDim,
    heads,
    layers,
    tokens,
  } = cacheContractFromSpec(spec, manifest);
  const ropeBase = Number(manifest.dynamics?.rope_base ?? manifest.dynamics?.base ?? 10000);
  const cosValues = new Float32Array(halfHeadDim);
  const sinValues = new Float32Array(halfHeadDim);
  for (let dim = 0; dim < halfHeadDim; dim += 1) {
    const theta = 1 / (ropeBase ** (dim / halfHeadDim));
    cosValues[dim] = Math.cos(theta);
    sinValues[dim] = Math.sin(theta);
  }

  const cacheIndex =
    cacheLayout === 'layer_batch_token_head_time_dim'
      ? (layer, batchIndex, token, time, head, dim) =>
          (((((layer * batch + batchIndex) * tokens + token) * heads + head) * contextLength +
            time) *
            headDim +
            dim)
      : (layer, batchIndex, token, time, head, dim) =>
          (((((layer * batch + batchIndex) * tokens + token) * contextLength + time) * heads +
            head) *
            headDim +
            dim);
  const entryIndex = (layer, batchIndex, token, head, dim) =>
    ((((layer * batch + batchIndex) * tokens + token) * heads + head) * headDim + dim);

  return {
    update(cache, kEntry, vEntry, cacheLength) {
      const logicalLength = cacheLength?.data?.[0] ?? contextLength;
      const kCache = cache.k.data;
      const vCache = cache.v.data;
      const kEntryData = kEntry.data;
      const vEntryData = vEntry.data;

      if (logicalLength < contextLength) {
        const dstTime = Math.min(Math.max(logicalLength, 0), contextLength - 1);
        for (let layer = 0; layer < layers; layer += 1) {
          for (let batchIndex = 0; batchIndex < batch; batchIndex += 1) {
            for (let token = 0; token < tokens; token += 1) {
              for (let head = 0; head < heads; head += 1) {
                for (let dim = 0; dim < headDim; dim += 1) {
                  const src = entryIndex(layer, batchIndex, token, head, dim);
                  const dst = cacheIndex(layer, batchIndex, token, dstTime, head, dim);
                  kCache[dst] = kEntryData[src];
                  vCache[dst] = vEntryData[src];
                }
              }
            }
          }
        }
        return cache;
      }

      const lastTime = contextLength - 1;
      for (let layer = 0; layer < layers; layer += 1) {
        for (let batchIndex = 0; batchIndex < batch; batchIndex += 1) {
          for (let token = 0; token < tokens; token += 1) {
            for (let head = 0; head < heads; head += 1) {
              for (let halfDim = 0; halfDim < halfHeadDim; halfDim += 1) {
                const cosTheta = cosValues[halfDim];
                const sinTheta = sinValues[halfDim];
                for (let time = 0; time < lastTime; time += 1) {
                  const srcLeft = cacheIndex(layer, batchIndex, token, time + 1, head, halfDim);
                  const srcRight = cacheIndex(
                    layer,
                    batchIndex,
                    token,
                    time + 1,
                    head,
                    halfHeadDim + halfDim,
                  );
                  const dstLeft = cacheIndex(layer, batchIndex, token, time, head, halfDim);
                  const dstRight = cacheIndex(
                    layer,
                    batchIndex,
                    token,
                    time,
                    head,
                    halfHeadDim + halfDim,
                  );
                  const left = kCache[srcLeft];
                  const right = kCache[srcRight];
                  kCache[dstLeft] = left * cosTheta + right * sinTheta;
                  kCache[dstRight] = right * cosTheta - left * sinTheta;
                }

                const srcEntryLeft = entryIndex(layer, batchIndex, token, head, halfDim);
                const srcEntryRight = entryIndex(
                  layer,
                  batchIndex,
                  token,
                  head,
                  halfHeadDim + halfDim,
                );
                const dstEntryLeft = cacheIndex(layer, batchIndex, token, lastTime, head, halfDim);
                const dstEntryRight = cacheIndex(
                  layer,
                  batchIndex,
                  token,
                  lastTime,
                  head,
                  halfHeadDim + halfDim,
                );
                kCache[dstEntryLeft] = kEntryData[srcEntryLeft];
                kCache[dstEntryRight] = kEntryData[srcEntryRight];
              }

              for (let dim = 0; dim < headDim; dim += 1) {
                for (let time = 0; time < lastTime; time += 1) {
                  const src = cacheIndex(layer, batchIndex, token, time + 1, head, dim);
                  const dst = cacheIndex(layer, batchIndex, token, time, head, dim);
                  vCache[dst] = vCache[src];
                }
                const srcEntry = entryIndex(layer, batchIndex, token, head, dim);
                const dstEntry = cacheIndex(layer, batchIndex, token, lastTime, head, dim);
                vCache[dstEntry] = vEntryData[srcEntry];
              }
            }
          }
        }
      }
      return cache;
    },
  };
}

function cloneCpuTensor(tensor) {
  const ArrayType = dtypeArray(tensor.type);
  return new ort.Tensor(tensor.type, new ArrayType(tensor.data), [...tensor.dims]);
}

function cacheFromInitialArtifacts(device, initialCache, backend) {
  if (backend !== 'webgpu') {
    return {
      k: cloneCpuTensor(initialCache.k),
      v: cloneCpuTensor(initialCache.v),
      length: cloneCpuTensor(initialCache.length),
    };
  }
  return {
    k: createGpuTensorFromCpu(device, initialCache.k),
    v: createGpuTensorFromCpu(device, initialCache.v),
    length: cloneCpuTensor(initialCache.length),
  };
}

async function renderLatent(zTensor) {
  const decoderOutputs = await runtime.sessions.decoder.run({ z: zTensor }, [runtime.names.patches]);
  const patches = decoderOutputs[runtime.names.patches];
  const image = patchesToImageData(patches, runtime.preprocessor);
  ctx.putImageData(image, 0, 0);
}

function setAction(action) {
  currentAction = action;
  const label = ACTION_LABELS[currentAction];
  elements.action.textContent = label;
  for (const [name, element] of Object.entries(elements.keys)) {
    const active = name === label;
    element.classList.toggle('active', active);
    element.setAttribute('aria-pressed', active ? 'true' : 'false');
  }
}

function actionFromKeys(event, pressed) {
  if (event.code === 'ArrowLeft') {
    event.preventDefault();
    setAction(pressed ? ACTIONS.left : ACTIONS.noop);
  } else if (event.code === 'ArrowRight') {
    event.preventDefault();
    setAction(pressed ? ACTIONS.right : ACTIONS.noop);
  } else if (event.code === 'Space' || event.code === 'ArrowUp') {
    event.preventDefault();
    setAction(pressed ? ACTIONS.fire : ACTIONS.noop);
  }
}

function bindActionButton(element, action) {
  element.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    element.setPointerCapture?.(event.pointerId);
    setAction(action);
  });
  const release = (event) => {
    event.preventDefault();
    if (currentAction === action || action === ACTIONS.noop) setAction(ACTIONS.noop);
  };
  element.addEventListener('pointerup', release);
  element.addEventListener('pointercancel', release);
  element.addEventListener('lostpointercapture', () => {
    if (currentAction === action) setAction(ACTIONS.noop);
  });
}

function preferredBackends() {
  const requested = String(configValue('backend', 'auto')).toLowerCase();
  if (requested === 'wasm' || requested === 'cpu') return ['wasm'];
  if (requested === 'webgpu') return ['webgpu'];
  return ['webgpu', 'wasm'];
}

async function releaseSession(session) {
  try {
    await session?.release?.();
  } catch {
    // Releasing after a failed compile is best-effort only.
  }
}

async function createRuntimeForBackend(backend, loaded) {
  let stepSession = null;
  let decoderSession = null;
  try {
    stepSession = await createSession(loaded.stepSpec, 'dynamics', loaded.stepModelBytes, backend, {
      preferredOutputLocation: {
        final_z: 'gpu-buffer',
        candidate_k_entry: 'gpu-buffer',
        candidate_v_entry: 'gpu-buffer',
      },
    });
    decoderSession = await createSession(
      loaded.decoderSpec,
      'decoder',
      loaded.decoderModelBytes,
      backend,
      {
        preferredOutputLocation: { patches: 'cpu' },
      },
    );

    const device = backend === 'webgpu' ? ort.env.webgpu?.device : null;
    if (backend === 'webgpu' && !device) {
      throw new Error('WebGPU session was created but ORT did not expose a GPU device.');
    }

    const cacheUpdater =
      backend === 'webgpu'
        ? createEntryCacheUpdater(device, loaded.stepSpec, loaded.manifest)
        : createCpuEntryCacheUpdater(loaded.stepSpec, loaded.manifest);

    const loadedRuntime = {
      backend,
      contextManifest: loaded.contextManifest,
      initialCacheManifest: loaded.initialCacheManifest,
      preprocessor: loaded.contextManifest.preprocessor,
      device,
      sessions: {
        step: stepSession,
        decoder: decoderSession,
      },
      specs: {
        step: loaded.stepSpec,
        decoder: loaded.decoderSpec,
      },
      names: {
        step: stepNamesForSpec(loaded.stepSpec),
        patches: outputName(loaded.decoderSpec, 'patches'),
      },
      dtypes: {
        sampleNoise: loaded.stepSpec.inputs.sample_noise.dtype,
      },
      initialCache: {
        k: loaded.initialK,
        v: loaded.initialV,
        length: loaded.initialLength,
      },
      contextLength:
        loaded.manifest.cache_contract?.context_length ??
        loaded.initialCacheManifest.arrays.k_cache.shape[3],
      displayZ: loaded.displayZ,
      displayPixels: loaded.displayPixels,
      cacheUpdater,
      cache: null,
    };

    loadedRuntime.cache = cacheFromInitialArtifacts(device, loadedRuntime.initialCache, backend);
    elements.backend.textContent = backend;
    return loadedRuntime;
  } catch (error) {
    await Promise.all([releaseSession(stepSession), releaseSession(decoderSession)]);
    throw error;
  }
}

async function loadRuntime() {
  const loadStarted = performance.now();
  setStatus('Loading manifests');
  const [manifest, contextManifest, initialCacheManifest] = await Promise.all([
    fetchJson(MANIFEST_URL, 'ONNX manifest'),
    fetchJson(CONTEXT_URL, 'context manifest'),
    fetchJson(INITIAL_CACHE_URL, 'initial cache manifest'),
  ]);
  const stepSpec = findFirstExport(manifest, [
    manifest.demo_generation?.preferred_step_export,
    'breakout_dynamics_sample_append_context_cache_length_entry_b1_t1_s2',
  ]);
  const decoderSpec = findExport(manifest, 'breakout_tokenizer_decode_z_b1_t1');

  setStatus('Loading context preview and initial cache');
  const displayPixelsPromise = contextManifest.arrays.display_pixels
    ? fetchTensorFromArtifact(ASSET_DIR, contextManifest.arrays.display_pixels, 'context preview pixels')
    : Promise.resolve(null);
  const [displayZ, displayPixels, initialK, initialV, initialLength] = await Promise.all([
    fetchTensorFromArtifact(ASSET_DIR, contextManifest.arrays.display_z, 'context preview'),
    displayPixelsPromise,
    fetchTensorFromArtifact(ASSET_DIR, initialCacheManifest.arrays.k_cache, 'initial K cache'),
    fetchTensorFromArtifact(ASSET_DIR, initialCacheManifest.arrays.v_cache, 'initial V cache'),
    fetchTensorFromArtifact(ASSET_DIR, initialCacheManifest.arrays.cache_length, 'cache length'),
  ]);
  validateInitialCache(stepSpec, initialK, initialV, initialLength);
  elements.context.textContent = `${contextManifest.prefix_frames} frames @ ${contextManifest.episode_start}`;

  setStatus('Loading ONNX models');
  const [stepModelBytes, decoderModelBytes] = await Promise.all([
    fetchBytes(`${ASSET_DIR}/${stepSpec.path}`, 'dynamics model'),
    fetchBytes(`${ASSET_DIR}/${decoderSpec.path}`, 'decoder model'),
  ]);

  const loaded = {
    manifest,
    contextManifest,
    initialCacheManifest,
    stepSpec,
    decoderSpec,
    displayZ,
    displayPixels,
    initialK,
    initialV,
    initialLength,
    stepModelBytes,
    decoderModelBytes,
  };
  let lastError = null;
  for (const backend of preferredBackends()) {
    try {
      const result = await createRuntimeForBackend(backend, loaded);
      recordLoadEvent('total load', performance.now() - loadStarted);
      return result;
    } catch (error) {
      lastError = error;
      recordLoadEvent(`${backend} unavailable`, 0);
      elements.backend.textContent = `${backend} failed`;
      setStatus(`${backend} unavailable`);
    }
  }
  throw lastError ?? new Error('No ONNX Runtime backend is available.');
}

async function resetDemo() {
  running = false;
  elements.start.textContent = 'Start';
  disposeCache(runtime.cache);
  runtime.cache = cacheFromInitialArtifacts(runtime.device, runtime.initialCache, runtime.backend);
  frameCount = 0;
  noiseGenerator = new NormalNoiseGenerator(runtime.contextManifest.noise_seed ?? 0);
  elements.frameCount.textContent = '0';
  elements.latency.textContent = '-- ms';
  const prefixFrames = runtime.contextManifest.prefix_frames ?? 1;
  if (runtime.displayPixels) {
    renderPixelTensor(runtime.displayPixels, prefixFrames - 1);
  } else {
    const previewTensor = contextFrameTensor(
      runtime.displayZ,
      prefixFrames - 1,
      runtime.specs.decoder.inputs.z.dtype,
    );
    await renderLatent(previewTensor);
  }
  setStatus(`Ready · ${runtime.backend} · cache length ${runtime.initialCache.length.data[0]}`);
}

async function generateFrame() {
  const started = performance.now();
  const action = new ort.Tensor('int32', new Int32Array([currentAction]), [1, 1]);
  const cacheLengthBefore = runtime.cache.length.data[0];
  const stepInputs = runtime.specs.step.inputs ?? {};
  const sampleNoise = randomNormalTensor(
    stepInputs.sample_noise.shape,
    stepInputs.sample_noise.dtype,
  );
  const contextNoise = randomNormalTensor(
    stepInputs.context_noise.shape,
    stepInputs.context_noise.dtype,
  );
  const fetches = [runtime.names.step.finalZ, runtime.names.step.k, runtime.names.step.v];
  if (runtime.names.step.length) fetches.push(runtime.names.step.length);
  const outputs = await runtime.sessions.step.run(
    {
      sample_noise: sampleNoise,
      context_noise: contextNoise,
      actions: action,
      k_cache: runtime.cache.k,
      v_cache: runtime.cache.v,
      ...stepPositionFeeds(
        runtime.specs.step,
        cacheLengthBefore,
        runtime.contextLength,
        runtime.cache.length,
      ),
    },
    fetches,
  );
  runtime.cacheUpdater.update(
    runtime.cache,
    outputs[runtime.names.step.k],
    outputs[runtime.names.step.v],
    runtime.cache.length,
  );
  if (runtime.names.step.length) {
    runtime.cache.length = outputs[runtime.names.step.length];
  } else {
    advanceCacheLength(runtime.cache.length, runtime.contextLength);
  }
  disposeGpuTensor(outputs[runtime.names.step.k]);
  disposeGpuTensor(outputs[runtime.names.step.v]);
  const zOutput = outputs[runtime.names.step.finalZ];

  const decoderOutputs = await runtime.sessions.decoder.run({ z: zOutput }, [runtime.names.patches]);
  disposeGpuTensor(zOutput);

  const image = patchesToImageData(decoderOutputs[runtime.names.patches], runtime.preprocessor);
  ctx.putImageData(image, 0, 0);

  frameCount += 1;
  const elapsed = performance.now() - started;
  const now = performance.now();
  const fps = 1000 / Math.max(now - lastFrameTime, 1);
  lastFrameTime = now;
  elements.frameCount.textContent = String(frameCount);
  elements.latency.textContent = `${elapsed.toFixed(1)} ms`;
  elements.fps.textContent = `${fps.toFixed(1)} fps`;
}

async function streamLoop() {
  if (!running) return;
  const frameStarted = performance.now();
  try {
    await generateFrame();
  } catch (error) {
    running = false;
    elements.start.textContent = 'Start';
    setStatus(error instanceof Error ? error.message : String(error));
    throw error;
  }
  const frameElapsed = performance.now() - frameStarted;
  const delayMs = targetFps > 0 ? Math.max(0, 1000 / targetFps - frameElapsed) : 0;
  window.setTimeout(streamLoop, delayMs);
}

elements.start.addEventListener('click', async () => {
  if (!runtime?.cache) return;
  running = !running;
  elements.start.textContent = running ? 'Pause' : 'Start';
  if (running) {
    lastFrameTime = performance.now();
    await streamLoop();
  }
});

elements.reset.addEventListener('click', resetDemo);
elements.targetFps.addEventListener('change', () => {
  targetFps = parseTargetFps(elements.targetFps.value);
});
window.addEventListener('keydown', (event) => actionFromKeys(event, true));
window.addEventListener('keyup', (event) => actionFromKeys(event, false));
bindActionButton(elements.keys.noop, ACTIONS.noop);
bindActionButton(elements.keys.fire, ACTIONS.fire);
bindActionButton(elements.keys.left, ACTIONS.left);
bindActionButton(elements.keys.right, ACTIONS.right);

setAction(ACTIONS.noop);
if ([...elements.targetFps.options].some((option) => option.value === String(targetFps))) {
  elements.targetFps.value = String(targetFps);
} else {
  targetFps = DEFAULT_TARGET_FPS;
  elements.targetFps.value = String(DEFAULT_TARGET_FPS);
}
elements.start.disabled = true;
elements.reset.disabled = true;

try {
  runtime = await loadRuntime();
  await resetDemo();
  elements.start.disabled = false;
  elements.reset.disabled = false;
} catch (error) {
  setStatus(error instanceof Error ? error.message : String(error));
  throw error;
}

window.visionaryDemoDebug = {
  get runtime() {
    return runtime;
  },
  get loadEvents() {
    return loadEvents;
  },
  async generateFrame() {
    await generateFrame();
  },
};
