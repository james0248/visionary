import { NormalNoiseGenerator } from './jax_noise';

const params = new URLSearchParams(window.location.search);
const scriptElement = document.getElementById('visionary-demo-main') as HTMLElement | null;
const globalConfig = window.VISIONARY_DEMO_CONFIG ?? {};

function configValue(name, fallback) {
  return params.get(name) ?? scriptElement?.dataset?.[name] ?? globalConfig[name] ?? fallback;
}

function detectBrowserProfile(userAgent) {
  if (/Version\/[\d.]+ Safari\//.test(userAgent) && !/(Chrome|Chromium|CriOS|Edg)\//.test(userAgent)) {
    return 'safari';
  }
  if (/(Chrome|Chromium|CriOS|Edg)\//.test(userAgent)) return 'chromium';
  if (/Firefox\//.test(userAgent)) return 'firefox';
  return 'unknown';
}

function resolveUrl(value) {
  return new URL(value, window.location.href).href;
}

function resolveBaseUrl(value) {
  return resolveUrl(value).replace(/\/$/, '');
}

const ASSET_DIR = resolveBaseUrl(configValue('assetBase', '/webgpu_app/assets'));
const MANIFEST_URL = `${ASSET_DIR}/${configValue('manifestName', 'breakout_onnx_manifest.json')}`;
const CONTEXT_URL = `${ASSET_DIR}/${configValue('contextName', 'breakout_demo_context.json')}`;
const INITIAL_CACHE_URL = `${ASSET_DIR}/${configValue('initialCacheName', 'breakout_demo_initial_cache.json')}`;
const requestedBrowserProfile = String(configValue('browserProfile', 'auto')).toLowerCase();
const detectedBrowserProfile = detectBrowserProfile(navigator.userAgent);
const browserProfile =
  requestedBrowserProfile === 'auto' ? detectedBrowserProfile : requestedBrowserProfile;
const DECODER_EXPORT_NAME = configValue('decoderExport', null);
const FULL_CACHE_STEP_EXPORT_NAME = configValue('fullCacheStepExport', null);
const SAFARI_SAFE_FULL_CACHE_STEP_EXPORT_NAME =
  'breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2_final_z_add_zero_safari_trial';
const PREFILL_EXPORT_FALLBACKS = parseConfigJson('prefillExportFallbacks', [
  'breakout_dynamics_prefill_cached_b1_t64',
]);
const DECODER_EXPORT_FALLBACKS = parseConfigJson('decoderExportFallbacks', [
  'breakout_tokenizer_decoder_b1_t1',
  'breakout_tokenizer_decode_z_b1_t1',
]);
const STEP_EXPORT_FALLBACKS = parseConfigJson('stepExportFallbacks', [
  'breakout_dynamics_sample_append_context_cache_length_entry_b1_t1_s2',
]);
const FULL_CACHE_STEP_EXPORT_FALLBACKS = parseConfigJson('fullCacheStepExportFallbacks', [
  ...(browserProfile === 'safari' ? [SAFARI_SAFE_FULL_CACHE_STEP_EXPORT_NAME] : []),
  'breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2',
  'breakout_dynamics_sample_append_context_full_cache_entry_b1_t1_s2',
  'breakout_dynamics_sample_append_context_slide_entry_b1_t1_s2',
]);
const DEFAULT_TARGET_FPS = 0;
const DEFAULT_DYNAMICS_GRAPH_CAPTURE = false;
const DEFAULT_FULL_DYNAMICS_GRAPH_CAPTURE = browserProfile === 'safari';
const DEFAULT_DECODER_GRAPH_CAPTURE = browserProfile === 'safari';
const DEFAULT_PREFILL_INITIAL_CACHE = browserProfile === 'safari';
const DEFAULT_GPU_PATCH_RENDERER = true;
const DEFAULT_GRAPH_OPTIMIZATION_LEVEL = 'basic';
const DEFAULT_ORT_MODULE = `/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs`;
const DEFAULT_ORT_WASM_BASE = `/node_modules/onnxruntime-web/dist/`;
const ort = await import(resolveUrl(configValue('ortModule', DEFAULT_ORT_MODULE)));

ort.env.wasm ??= {};
ort.env.wasm.wasmPaths = resolveUrl(configValue('ortWasmBase', DEFAULT_ORT_WASM_BASE));
ort.env.webgpu ??= {};
ort.env.webgpu.powerPreference = 'high-performance';

const DEFAULT_ACTION_DEFINITIONS = [
  { id: 0, name: 'noop', label: 'noop', keys: [] },
  { id: 1, name: 'fire', label: 'fire', keys: [['Space'], ['ArrowUp']] },
  { id: 2, name: 'right', label: 'right', keys: [['ArrowRight']] },
  { id: 3, name: 'left', label: 'left', keys: [['ArrowLeft']] },
];
const ACTION_DEFINITIONS = normalizeActionDefinitions(
  parseConfigJson('actions', DEFAULT_ACTION_DEFINITIONS),
);
const ACTIONS = Object.fromEntries(ACTION_DEFINITIONS.map((action) => [action.name, action.id]));
const ACTION_BY_ID = new Map(ACTION_DEFINITIONS.map((action) => [action.id, action]));
const NOOP_ACTION = ACTIONS.noop ?? ACTION_DEFINITIONS[0]?.id ?? 0;
const BOUND_KEY_CODES = new Set(
  ACTION_DEFINITIONS.flatMap((action) => action.keyCombos.flatMap((combo) => combo)),
);

const CONTEXT_TENSOR_SIZE = 32 * 32;
const STATS_UPDATE_INTERVAL_MS = 250;
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
  actionButtons: [...document.querySelectorAll('[data-action-id]')] as HTMLButtonElement[],
};

let ctx: CanvasRenderingContext2D | null = null;
let canvas2dUnavailable = false;
let imageDataFallback: HTMLImageElement | null = null;
let imageDataFallbackUrl: string | null = null;
let previewOverlay: HTMLImageElement | null = null;
let previewOverlayUrl: string | null = null;

function createDisplayCanvasLike(source: HTMLCanvasElement) {
  const canvas = document.createElement('canvas');
  for (const attribute of Array.from(source.attributes)) {
    canvas.setAttribute(attribute.name, attribute.value);
  }
  canvas.width = source.width;
  canvas.height = source.height;
  return canvas;
}

function replaceDisplayCanvas(canvas: HTMLCanvasElement) {
  elements.canvas.replaceWith(canvas);
  elements.canvas = canvas;
  ctx = null;
  canvas2dUnavailable = false;
  hideImageDataFallback();
}

function getCanvas2dContext(canvas: HTMLCanvasElement) {
  try {
    return canvas.getContext('2d', { alpha: false }) ?? canvas.getContext('2d');
  } catch {
    return null;
  }
}

function createMinimalDisplayCanvasLike(source: HTMLCanvasElement) {
  const canvas = document.createElement('canvas');
  canvas.id = source.id;
  canvas.width = source.width;
  canvas.height = source.height;
  canvas.setAttribute(
    'aria-label',
    source.getAttribute('aria-label') ?? 'Generated Breakout frame',
  );
  return canvas;
}

function replaceDisplayCanvasWith2dContext(canvas: HTMLCanvasElement, context: CanvasRenderingContext2D) {
  elements.canvas.replaceWith(canvas);
  elements.canvas = canvas;
  ctx = context;
  canvas2dUnavailable = false;
  hideImageDataFallback();
  ctx.imageSmoothingEnabled = false;
  return ctx;
}

function canvas2dContext() {
  if (canvas2dUnavailable) return null;
  if (!ctx) {
    ctx = getCanvas2dContext(elements.canvas);
    if (ctx) {
      ctx.imageSmoothingEnabled = false;
      hideImageDataFallback();
      return ctx;
    }
    const replacement = createDisplayCanvasLike(elements.canvas);
    const replacementCtx = getCanvas2dContext(replacement);
    if (replacementCtx) return replaceDisplayCanvasWith2dContext(replacement, replacementCtx);
    const minimalReplacement = createMinimalDisplayCanvasLike(elements.canvas);
    const minimalCtx = getCanvas2dContext(minimalReplacement);
    if (minimalCtx) return replaceDisplayCanvasWith2dContext(minimalReplacement, minimalCtx);
    canvas2dUnavailable = true;
    return null;
  }
  return ctx;
}

function hideImageDataFallback() {
  elements.canvas.hidden = false;
  if (imageDataFallback) imageDataFallback.hidden = true;
}

function hidePreviewOverlay() {
  if (previewOverlay) previewOverlay.hidden = true;
}

function imageDataToBmpBlob(image: ImageData) {
  const width = image.width;
  const height = image.height;
  const stride = width * 4;
  const pixelBytes = stride * height;
  const fileBytes = 54 + pixelBytes;
  const buffer = new ArrayBuffer(fileBytes);
  const view = new DataView(buffer);
  view.setUint8(0, 0x42);
  view.setUint8(1, 0x4d);
  view.setUint32(2, fileBytes, true);
  view.setUint32(10, 54, true);
  view.setUint32(14, 40, true);
  view.setInt32(18, width, true);
  view.setInt32(22, height, true);
  view.setUint16(26, 1, true);
  view.setUint16(28, 32, true);
  view.setUint32(34, pixelBytes, true);
  view.setInt32(38, 2835, true);
  view.setInt32(42, 2835, true);
  const bytes = new Uint8Array(buffer);
  const source = image.data;
  let target = 54;
  for (let y = height - 1; y >= 0; y -= 1) {
    const row = y * stride;
    for (let x = 0; x < width; x += 1) {
      const pixel = row + x * 4;
      bytes[target] = source[pixel + 2];
      bytes[target + 1] = source[pixel + 1];
      bytes[target + 2] = source[pixel];
      bytes[target + 3] = 255;
      target += 4;
    }
  }
  return new Blob([buffer], { type: 'image/bmp' });
}

function renderImageDataFallback(image: ImageData) {
  hidePreviewOverlay();
  if (!imageDataFallback) {
    imageDataFallback = document.createElement('img');
    imageDataFallback.className = 'frame-fallback';
    imageDataFallback.alt =
      elements.canvas.getAttribute('aria-label') ?? 'Generated game frame';
    elements.canvas.insertAdjacentElement('afterend', imageDataFallback);
  }
  imageDataFallback.width = image.width;
  imageDataFallback.height = image.height;
  imageDataFallback.hidden = false;
  elements.canvas.hidden = true;
  if (imageDataFallbackUrl) URL.revokeObjectURL(imageDataFallbackUrl);
  imageDataFallbackUrl = URL.createObjectURL(imageDataToBmpBlob(image));
  imageDataFallback.src = imageDataFallbackUrl;
}

function renderImageDataPreviewOverlay(image: ImageData) {
  if (!previewOverlay) {
    previewOverlay = document.createElement('img');
    previewOverlay.className = 'frame-preview';
    previewOverlay.alt = elements.canvas.getAttribute('aria-label') ?? 'Initial game frame';
    elements.canvas.insertAdjacentElement('afterend', previewOverlay);
  }
  previewOverlay.width = image.width;
  previewOverlay.height = image.height;
  previewOverlay.hidden = false;
  if (previewOverlayUrl) URL.revokeObjectURL(previewOverlayUrl);
  previewOverlayUrl = URL.createObjectURL(imageDataToBmpBlob(image));
  previewOverlay.src = previewOverlayUrl;
}

let runtime = null;
let running = false;
let frameCount = 0;
let currentAction = NOOP_ACTION;
let lastFrameTime = performance.now();
let lastStatsUpdateTime = 0;
let statsFramesSinceUpdate = 0;
let noiseGenerator = new NormalNoiseGenerator(0);
let targetFps = parseTargetFps(configValue('fps', DEFAULT_TARGET_FPS));
const graphCaptureConfig = configValue('graphCapture', null);
const graphCaptureRequested =
  graphCaptureConfig == null ? null : parseBooleanConfig(graphCaptureConfig, true);
// Short-cache dynamics graph capture can freeze the latent stream. Safari's full-cache path is
// allowed only when the manifest selects the materialized final_z artifact below.
const unsafeGraphCaptureAllowed = parseBooleanConfig(
  configValue('allowUnsafeGraphCapture', false),
  false,
);
const safariSafeGraphCaptureAllowed = parseBooleanConfig(
  configValue('allowSafariGraphCapture', browserProfile === 'safari'),
  browserProfile === 'safari',
);
const rawDynamicsGraphCaptureRequested = parseBooleanConfig(
  configValue('dynamicsGraphCapture', graphCaptureRequested ?? DEFAULT_DYNAMICS_GRAPH_CAPTURE),
  graphCaptureRequested ?? DEFAULT_DYNAMICS_GRAPH_CAPTURE,
);
const rawFullDynamicsGraphCaptureRequested = parseBooleanConfig(
  configValue('fullDynamicsGraphCapture', DEFAULT_FULL_DYNAMICS_GRAPH_CAPTURE),
  DEFAULT_FULL_DYNAMICS_GRAPH_CAPTURE,
);
const dynamicsGraphCaptureRequested =
  rawDynamicsGraphCaptureRequested && unsafeGraphCaptureAllowed;
const fullDynamicsGraphCaptureRequested = rawFullDynamicsGraphCaptureRequested;
const prefillInitialCacheDefault = DEFAULT_PREFILL_INITIAL_CACHE;
const prefillInitialCacheRequested = parseBooleanConfig(
  configValue('prefillInitialCache', prefillInitialCacheDefault),
  prefillInitialCacheDefault,
);
const dynamicsGraphCaptureEnabled = dynamicsGraphCaptureRequested && !fullDynamicsGraphCaptureRequested;
const decoderGraphCaptureRequested = parseBooleanConfig(
  configValue('decoderGraphCapture', graphCaptureRequested ?? DEFAULT_DECODER_GRAPH_CAPTURE),
  graphCaptureRequested ?? DEFAULT_DECODER_GRAPH_CAPTURE,
);
const decoderGraphCaptureEnabled =
  decoderGraphCaptureRequested &&
  (unsafeGraphCaptureAllowed || (browserProfile === 'safari' && safariSafeGraphCaptureAllowed));
const gpuPatchRendererEnabled = parseBooleanConfig(
  configValue('gpuPatchRenderer', DEFAULT_GPU_PATCH_RENDERER),
  DEFAULT_GPU_PATCH_RENDERER,
);
const fullCacheStepEnabled = parseBooleanConfig(configValue('fullCacheStep', true), true);
const skipShortCacheStepWhenFull = parseBooleanConfig(
  configValue('skipShortCacheStepWhenFull', browserProfile === 'safari'),
  browserProfile === 'safari',
);
const graphCaptureUploadFenceEnabled = parseBooleanConfig(
  configValue('graphCaptureUploadFence', false),
  false,
);
const graphCaptureInputUploadMode = String(
  configValue('graphCaptureInputUploadMode', 'write'),
).toLowerCase();
const cacheUpdateFenceEnabled = parseBooleanConfig(
  configValue('cacheUpdateFence', false),
  false,
);
const preallocateStepOutputsEnabled = parseBooleanConfig(
  configValue('preallocateStepOutputs', browserProfile !== 'safari'),
  browserProfile !== 'safari',
);
const preallocateDecoderOutputsEnabled = parseBooleanConfig(
  configValue('preallocateDecoderOutputs', browserProfile !== 'safari'),
  browserProfile !== 'safari',
);
const graphOptimizationLevel = String(
  configValue('graphOptimizationLevel', DEFAULT_GRAPH_OPTIMIZATION_LEVEL),
);
const preferredLayout = configValue('preferredLayout', null);
const throttleMbps = Number(params.get('throttleMbps') ?? 0);
let loadEvents = [];
let keySequence = 0;
const activeKeyCodes = new Map();
let streamLoopPending = false;
let streamLoopChannel: MessageChannel | null = null;

function parseTargetFps(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : DEFAULT_TARGET_FPS;
}

function parseConfigJson(name, fallback) {
  const value = configValue(name, null);
  if (value == null) return fallback;
  if (typeof value !== 'string') return value;
  const trimmed = value.trim();
  if (!trimmed) return fallback;
  try {
    return JSON.parse(trimmed);
  } catch (error) {
    throw new Error(`Invalid JSON demo config ${name}: ${error instanceof Error ? error.message : error}`);
  }
}

function parseBooleanConfig(value, fallback) {
  if (typeof value === 'boolean') return value;
  const normalized = String(value).toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
  return fallback;
}

function normalizeKeyCombos(keys) {
  if (!Array.isArray(keys)) return [];
  if (keys.every((key) => typeof key === 'string')) return [keys];
  return keys
    .filter((combo) => Array.isArray(combo))
    .map((combo) => combo.filter((key) => typeof key === 'string'))
    .filter((combo) => combo.length > 0);
}

function normalizeActionDefinitions(actions) {
  return actions.map((action) => {
    const id = Number(action.id);
    if (!Number.isInteger(id) || id < 0) {
      throw new Error(`Invalid action id ${action.id}`);
    }
    const name = String(action.name ?? action.label ?? id);
    return {
      id,
      name,
      label: String(action.label ?? name),
      keyCombos: normalizeKeyCombos(action.keys ?? action.keyCombos),
    };
  });
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

function writeCpuTensorToGpu(device, gpuTensor, cpuTensor) {
  device.queue.writeBuffer(gpuTensor.gpuBuffer, 0, tensorDataBytes(cpuTensor));
}

function createGraphCaptureUploadBuffer(device, gpuTensor, cpuTensor) {
  const byteLength = Math.max(16, tensorByteLength(cpuTensor.type, cpuTensor.dims));
  const buffer = device.createBuffer({
    label: 'visionary-graph-capture-input-upload',
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  gpuTensor.__visionaryUploadBuffer = buffer;
  gpuTensor.__visionaryUploadByteLength = tensorByteLength(cpuTensor.type, cpuTensor.dims);
  return buffer;
}

function copyCpuTensorToGpuViaUploadBuffer(device, gpuTensor, cpuTensor) {
  const byteLength = tensorByteLength(cpuTensor.type, cpuTensor.dims);
  const uploadBuffer =
    gpuTensor.__visionaryUploadBuffer &&
    gpuTensor.__visionaryUploadByteLength === byteLength
      ? gpuTensor.__visionaryUploadBuffer
      : createGraphCaptureUploadBuffer(device, gpuTensor, cpuTensor);
  device.queue.writeBuffer(uploadBuffer, 0, tensorDataBytes(cpuTensor));
  const encoder = device.createCommandEncoder({ label: 'visionary-graph-capture-input-upload' });
  encoder.copyBufferToBuffer(uploadBuffer, 0, gpuTensor.gpuBuffer, 0, byteLength);
  device.queue.submit([encoder.finish()]);
}

function writeGraphCaptureInputTensor(device, gpuTensor, cpuTensor) {
  if (graphCaptureInputUploadMode === 'copy') {
    copyCpuTensorToGpuViaUploadBuffer(device, gpuTensor, cpuTensor);
    return;
  }
  writeCpuTensorToGpu(device, gpuTensor, cpuTensor);
}

function copyGpuTensor(device, source, target) {
  if (source === target) return;
  const byteLength = tensorByteLength(source.type, source.dims);
  const encoder = device.createCommandEncoder({ label: 'visionary-demo-gpu-copy' });
  encoder.copyBufferToBuffer(source.gpuBuffer, 0, target.gpuBuffer, 0, byteLength);
  device.queue.submit([encoder.finish()]);
}

function copyTensorToGpu(device, source, target) {
  if (source?.location === 'gpu-buffer') {
    copyGpuTensor(device, source, target);
  } else {
    writeCpuTensorToGpu(device, target, source);
  }
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

function createEmptyGpuTensor(device, spec) {
  const byteLength = Math.max(16, tensorByteLength(spec.dtype, spec.shape));
  const buffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  return ort.Tensor.fromGpuBuffer(buffer, {
    dataType: spec.dtype,
    dims: spec.shape,
    dispose: () => buffer.destroy(),
  });
}

function cloneTensorToGpu(device, tensor) {
  if (tensor.location === 'gpu-buffer') {
    const clone = createEmptyGpuTensor(device, { dtype: tensor.type, shape: tensor.dims });
    copyTensorToGpu(device, tensor, clone);
    return clone;
  }
  return createGpuTensorFromCpu(device, tensor);
}

function createGpuOutputFetches(device, spec, names) {
  if (!device) return null;
  return Object.fromEntries(
    names.map((name) => {
      const outputSpec = spec.outputs?.[name];
      if (!outputSpec) throw new Error(`Cannot preallocate unknown output ${name}`);
      return [name, createEmptyGpuTensor(device, outputSpec)];
    }),
  );
}

function createGraphCaptureFixedCache(device, stepSpec, names) {
  const inputs = stepSpec.inputs ?? {};
  return {
    k: createGpuTensorFromCpu(device, makeZeroTensorFromSpec(inputs[names.kCache])),
    v: createGpuTensorFromCpu(device, makeZeroTensorFromSpec(inputs[names.vCache])),
  };
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
  return new ort.Tensor('float32', values instanceof Float32Array ? values : new Float32Array(values), shape);
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

function findFirstOptionalExport(manifest, names) {
  for (const name of names.filter(Boolean)) {
    const entry = manifest.exports.find((item) => item.name === name);
    if (entry) return entry;
  }
  return null;
}

function isSafariSafeFullCacheStepExport(manifest, spec) {
  if (browserProfile !== 'safari' || !safariSafeGraphCaptureAllowed || !spec) return false;
  const manifestPreferred = manifest.demo_generation?.preferred_full_cache_step_export_safari;
  return spec.name === manifestPreferred || spec.name === SAFARI_SAFE_FULL_CACHE_STEP_EXPORT_NAME;
}

function fullDynamicsGraphCaptureRequestedForSpec(manifest, spec) {
  if (!fullDynamicsGraphCaptureRequested) return false;
  return unsafeGraphCaptureAllowed || isSafariSafeFullCacheStepExport(manifest, spec);
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

function prefillCacheOutputNames(prefillSpec) {
  return {
    k: requiredOutputName(prefillSpec, 'k_cache'),
    v: requiredOutputName(prefillSpec, 'v_cache'),
    length: requiredOutputName(prefillSpec, 'cache_length'),
  };
}

function prefillFeedsFromContext(prefillSpec, loaded) {
  const sources = {
    z: loaded.contextZ,
    actions: loaded.contextActions,
    step_levels: loaded.contextStepLevels,
    signal_levels: loaded.contextSignalLevels,
  };
  return Object.fromEntries(
    Object.keys(prefillSpec.inputs ?? {}).map((name) => {
      const tensor = sources[name];
      if (!tensor) throw new Error(`Prefill cache input ${name} is missing from demo context.`);
      return [name, tensor];
    }),
  );
}

async function createSession(spec, label, modelBytes, backend, options = {}) {
  const started = performance.now();
  setStatus(`Compiling ${label} · ${backend}`);
  const backendOptions = backend === 'webgpu' ? options : {};
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders:
      backend === 'webgpu' ? [{ name: 'webgpu', validationMode: 'disabled' }] : ['wasm'],
    graphOptimizationLevel,
    externalData: (spec.external_data ?? []).map((entry) => ({
      path: entry.path,
      data: `${ASSET_DIR}/${entry.path}`,
    })),
    ...(backend === 'webgpu' && preferredLayout ? { preferredLayout } : {}),
    ...backendOptions,
  });
  recordLoadEvent(`${label} ${backend} compile`, performance.now() - started);
  return session;
}

function randomNormalTensor(shape, dtype = 'float32') {
  const size = shape.reduce((total, value) => total * value, 1);
  return makeFloatTensor(dtype, noiseGenerator.tensorData(size), shape);
}

function mutableFloatTensorFromSpec(inputSpec) {
  const size = mul(inputSpec.shape);
  if (inputSpec.dtype === 'float16') {
    return new ort.Tensor('float16', new Uint16Array(size), inputSpec.shape);
  }
  return new ort.Tensor('float32', new Float32Array(size), inputSpec.shape);
}

function fillRandomNormalTensor(tensor) {
  if (tensor.type === 'float16') {
    const values = tensor.data;
    for (let index = 0; index < values.length; index += 1) {
      values[index] = float32ToFloat16Bits(noiseGenerator.normal());
    }
    return tensor;
  }
  noiseGenerator.fillTensorData(tensor.data);
  return tensor;
}

function createNoiseInputSlot(inputs) {
  return {
    sampleNoise: mutableFloatTensorFromSpec(inputs.sample_noise),
    contextNoise: mutableFloatTensorFromSpec(inputs.context_noise),
  };
}

function fillNoiseInputSlot(slot) {
  fillRandomNormalTensor(slot.sampleNoise);
  fillRandomNormalTensor(slot.contextNoise);
  return slot;
}

function createPositionInputState(stepSpec) {
  const inputs = stepSpec.inputs ?? {};
  return {
    samplePositionIndex: inputs.sample_position_index
      ? scalarFillTensor(inputs.sample_position_index, 0)
      : null,
    contextPositionIndex: inputs.context_position_index
      ? scalarFillTensor(inputs.context_position_index, 0)
      : null,
    attentionMask: inputs.attention_mask ? cacheAttentionMaskTensor(inputs.attention_mask, 0, 0) : null,
  };
}

function createFrameInputState(stepSpec) {
  const inputs = stepSpec.inputs ?? {};
  const actionShape = inputs.actions?.shape ?? [1, 1];
  const noiseSlots = [createNoiseInputSlot(inputs), createNoiseInputSlot(inputs)];
  return {
    spec: stepSpec,
    noiseSlots,
    noiseSlotReady: [false, false],
    noiseSlotIndex: 0,
    action: new ort.Tensor('int32', new Int32Array(mul(actionShape)), actionShape),
    actionValue: null,
    positionInputs: createPositionInputState(stepSpec),
  };
}

function frameInputStateFor(stepSpec) {
  if (!runtime.frameInputState || runtime.frameInputState.spec !== stepSpec) {
    runtime.frameInputState = createFrameInputState(stepSpec);
  }
  return runtime.frameInputState;
}

function currentNoiseInputSlot(frameInputs) {
  const index = frameInputs.noiseSlotIndex;
  if (!frameInputs.noiseSlotReady[index]) {
    fillNoiseInputSlot(frameInputs.noiseSlots[index]);
    frameInputs.noiseSlotReady[index] = true;
  }
  return frameInputs.noiseSlots[index];
}

function prefillNextNoiseInputSlot(frameInputs) {
  const consumedIndex = frameInputs.noiseSlotIndex;
  frameInputs.noiseSlotReady[consumedIndex] = false;
  frameInputs.noiseSlotIndex = 1 - consumedIndex;
  const nextIndex = frameInputs.noiseSlotIndex;
  if (!frameInputs.noiseSlotReady[nextIndex]) {
    fillNoiseInputSlot(frameInputs.noiseSlots[nextIndex]);
    frameInputs.noiseSlotReady[nextIndex] = true;
  }
}

function actionTensorForFrame(frameInputs) {
  if (frameInputs.actionValue !== currentAction) {
    frameInputs.action.data.fill(currentAction);
    frameInputs.actionValue = currentAction;
  }
  return frameInputs.action;
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

function scalarFillTensor(inputSpec, value) {
  const size = mul(inputSpec.shape);
  if (inputSpec.dtype === 'int32') {
    return new ort.Tensor('int32', new Int32Array(size).fill(value), inputSpec.shape);
  }
  if (inputSpec.dtype === 'float16') {
    return makeFloatTensor('float16', new Float32Array(size).fill(value), inputSpec.shape);
  }
  return new ort.Tensor('float32', new Float32Array(size).fill(value), inputSpec.shape);
}

function fillScalarTensor(tensor, value) {
  if (tensor.type === 'float16') {
    tensor.data.fill(float32ToFloat16Bits(value));
    return;
  }
  tensor.data.fill(value);
}

function fillAttentionMaskTensor(tensor, cacheLength, contextLength) {
  const validLength = Math.min(Math.max(cacheLength, 0), contextLength);
  const width = contextLength + 1;
  const data = tensor.data;
  if (tensor.type === 'float16') {
    const one = float32ToFloat16Bits(1);
    const zero = float32ToFloat16Bits(0);
    for (let index = 0; index < data.length; index += 1) {
      const position = index % width;
      data[index] = position < validLength || position === contextLength ? one : zero;
    }
    return;
  }
  for (let index = 0; index < data.length; index += 1) {
    const position = index % width;
    data[index] = position < validLength || position === contextLength ? 1 : 0;
  }
}

function stepPositionFeeds(stepSpec, cacheLength, contextLength, cacheLengthTensor, positionInputs = null) {
  const inputs = stepSpec.inputs ?? {};
  const feeds: Record<string, unknown> = {};
  if (inputs.sample_position_index) {
    const tensor =
      positionInputs?.samplePositionIndex ?? scalarFillTensor(inputs.sample_position_index, 0);
    fillScalarTensor(tensor, Math.min(cacheLength, contextLength));
    feeds.sample_position_index = tensor;
  }
  if (inputs.context_position_index) {
    const tensor =
      positionInputs?.contextPositionIndex ?? scalarFillTensor(inputs.context_position_index, 0);
    fillScalarTensor(tensor, Math.min(cacheLength, contextLength - 1));
    feeds.context_position_index = tensor;
  }
  if (inputs.attention_mask) {
    const tensor =
      positionInputs?.attentionMask ?? cacheAttentionMaskTensor(inputs.attention_mask, 0, contextLength);
    fillAttentionMaskTensor(tensor, cacheLength, contextLength);
    feeds.attention_mask = tensor;
  }
  if (inputs.cache_length) {
    feeds.cache_length = cacheLengthTensor;
  }
  return feeds;
}

function findInputName(spec, candidates) {
  const inputs = Object.keys(spec.inputs ?? {});
  for (const candidate of candidates) {
    const exact = inputs.find((name) => name === candidate);
    if (exact) return exact;
  }
  return inputs.find((name) => candidates.some((candidate) => name.includes(candidate))) ?? null;
}

function makeZeroTensorFromSpec(spec) {
  if (spec.dtype === 'int32') {
    return new ort.Tensor('int32', new Int32Array(mul(spec.shape)), spec.shape);
  }
  if (spec.dtype === 'float16') {
    return makeFloatTensor('float16', new Float32Array(mul(spec.shape)), spec.shape);
  }
  return new ort.Tensor('float32', new Float32Array(mul(spec.shape)), spec.shape);
}

function graphCaptureStepInputNames(stepSpec) {
  const inputs = stepSpec.inputs ?? {};
  const names = {
    sampleNoise: findInputName(stepSpec, ['sample_noise', 'z']),
    contextNoise: findInputName(stepSpec, ['context_noise']),
    actions: findInputName(stepSpec, ['actions', 'action']),
    kCache: inputs.k_cache ? 'k_cache' : null,
    vCache: inputs.v_cache ? 'v_cache' : null,
    cacheLength: findInputName(stepSpec, ['cache_length']),
    samplePositionIndex: findInputName(stepSpec, ['sample_position_index']),
    contextPositionIndex: findInputName(stepSpec, ['context_position_index']),
    attentionMask: findInputName(stepSpec, ['attention_mask']),
    positionIndex: inputs.position_index ? 'position_index' : null,
  };
  if (!names.sampleNoise || !names.actions || !names.kCache || !names.vCache) return null;
  return names;
}

function createGraphCaptureStepState(
  device,
  stepSpec,
  contextLength,
  options: {
    fixedCache?: boolean;
    preallocateOutputs?: boolean;
  } = {},
) {
  const names = graphCaptureStepInputNames(stepSpec);
  if (!names) return null;
  const inputs = stepSpec.inputs ?? {};
  const outputNames = stepNamesForSpec(stepSpec);
  const tensors = {
    sampleNoise: createGpuTensorFromCpu(device, makeZeroTensorFromSpec(inputs[names.sampleNoise])),
    contextNoise: names.contextNoise
      ? createGpuTensorFromCpu(device, makeZeroTensorFromSpec(inputs[names.contextNoise]))
      : null,
    actions: createGpuTensorFromCpu(device, makeZeroTensorFromSpec(inputs[names.actions])),
    cacheLength: names.cacheLength
      ? createGpuTensorFromCpu(
          device,
          scalarFillTensor(inputs[names.cacheLength], contextLength),
        )
      : null,
    samplePositionIndex: names.samplePositionIndex
      ? createGpuTensorFromCpu(
          device,
          scalarFillTensor(inputs[names.samplePositionIndex], contextLength),
        )
      : null,
    contextPositionIndex: names.contextPositionIndex
      ? createGpuTensorFromCpu(
          device,
          scalarFillTensor(inputs[names.contextPositionIndex], Math.max(contextLength - 1, 0)),
        )
      : null,
    attentionMask: names.attentionMask
      ? createGpuTensorFromCpu(
          device,
          cacheAttentionMaskTensor(inputs[names.attentionMask], contextLength, contextLength),
        )
      : null,
    positionIndex: names.positionIndex
      ? createGpuTensorFromCpu(
          device,
          scalarFillTensor(inputs[names.positionIndex], contextLength),
        )
      : null,
  };
  const outputFetches = options.preallocateOutputs
    ? createGpuOutputFetches(device, stepSpec, [outputNames.finalZ, outputNames.k, outputNames.v])
    : null;
  return {
    enabled: true,
    names,
    tensors,
    outputFetches,
    staticScalars: Boolean(options.fixedCache),
    fixedCache: options.fixedCache ? createGraphCaptureFixedCache(device, stepSpec, names) : null,
    fixedCacheReady: false,
    capturedOnce: false,
  };
}

function reuseFinalZOutputAsDecoderInput(stepState, stepSpec, stepNames, decoderInput) {
  if (!stepState?.outputFetches || !decoderInput?.tensor || !stepNames?.finalZ) return;
  const outputSpec = stepSpec.outputs?.[stepNames.finalZ];
  if (
    outputSpec?.dtype === decoderInput.tensor.type &&
    sameShape(outputSpec.shape, decoderInput.tensor.dims)
  ) {
    stepState.outputFetches[stepNames.finalZ] = decoderInput.tensor;
  }
}

function decoderInputName(decoderSpec) {
  const inputs = decoderSpec.inputs ?? {};
  if (inputs.z) return 'z';
  if (inputs.latent) return 'latent';
  return Object.keys(inputs)[0] ?? null;
}

function decoderInputDtype(decoderSpec) {
  const name = decoderInputName(decoderSpec);
  return name ? decoderSpec.inputs[name]?.dtype ?? 'float32' : 'float32';
}

function createDecoderInputState(device, decoderSpec, backend, forceFixedInput = false) {
  if (backend !== 'webgpu' || (decoderSpec.inputs?.z && !forceFixedInput)) return null;
  const name = decoderInputName(decoderSpec);
  if (!name) return null;
  return {
    name,
    tensor: createGpuTensorFromCpu(device, makeZeroTensorFromSpec(decoderSpec.inputs[name])),
  };
}

async function createDecoderSessionsForBackend(
  backend,
  decoderSpec,
  decoderModelBytes,
  patchRenderer,
  decoderGraphInput,
) {
  const options = {
    preferredOutputLocation: { patches: patchRenderer ? 'gpu-buffer' : 'cpu' },
  };
  const session = await createSession(decoderSpec, 'decoder', decoderModelBytes, backend, options);
  if (backend !== 'webgpu' || !decoderGraphInput || !decoderGraphCaptureEnabled) {
    return { session, graphCaptureSession: null, graphCaptureEnabled: false };
  }
  try {
    return {
      session,
      graphCaptureSession: await createSession(decoderSpec, 'decoder graph capture', decoderModelBytes, backend, {
        ...options,
        enableGraphCapture: true,
      }),
      graphCaptureEnabled: true,
    };
  } catch (error) {
    recordLoadEvent('decoder graph capture unavailable', 0);
    console.warn('Falling back to WebGPU decoder without graph capture:', error);
    return { session, graphCaptureSession: null, graphCaptureEnabled: false };
  }
}

function latentTensorFromZ(zTensor) {
  if (mul(zTensor.dims) !== 1024) {
    throw new Error(`Cannot reinterpret decoder input ${zTensor.dims.join('x')} as [1,1,64,16].`);
  }
  const ArrayType = dtypeArray(zTensor.type);
  return new ort.Tensor(zTensor.type, new ArrayType(zTensor.data), [1, 1, 64, 16]);
}

function decoderFeedsFromZ(runtime, zTensor, decoderInput = runtime.decoderInput) {
  if (decoderInput) {
    copyTensorToGpu(runtime.device, zTensor, decoderInput.tensor);
    return { [decoderInput.name]: decoderInput.tensor };
  }
  if (runtime.specs.decoder.inputs?.z) {
    return { z: zTensor };
  }
  return { [decoderInputName(runtime.specs.decoder)]: latentTensorFromZ(zTensor) };
}

function updateGraphCaptureStepInputs(
  runtime,
  stepRuntime,
  sampleNoise,
  contextNoise,
  action,
  cacheLength,
) {
  const state = stepRuntime.graphCapture;
  const inputs = stepRuntime.spec.inputs ?? {};
  const names = state.names;
  const tensors = state.tensors;
  const samplePosition = Math.min(cacheLength, runtime.contextLength);
  const contextPosition = Math.min(cacheLength, runtime.contextLength - 1);

  writeGraphCaptureInputTensor(runtime.device, tensors.sampleNoise, sampleNoise);
  if (tensors.contextNoise) {
    writeGraphCaptureInputTensor(runtime.device, tensors.contextNoise, contextNoise);
  }
  const actionValue = action.data[0];
  if (state.uploadedActionValue !== actionValue) {
    writeGraphCaptureInputTensor(runtime.device, tensors.actions, action);
    state.uploadedActionValue = actionValue;
  }
  if (state.staticScalars) return;
  if (tensors.cacheLength) {
    writeGraphCaptureInputTensor(
      runtime.device,
      tensors.cacheLength,
      scalarFillTensor(inputs[names.cacheLength], cacheLength),
    );
  }
  if (tensors.samplePositionIndex) {
    writeGraphCaptureInputTensor(
      runtime.device,
      tensors.samplePositionIndex,
      scalarFillTensor(inputs[names.samplePositionIndex], samplePosition),
    );
  }
  if (tensors.contextPositionIndex) {
    writeGraphCaptureInputTensor(
      runtime.device,
      tensors.contextPositionIndex,
      scalarFillTensor(inputs[names.contextPositionIndex], contextPosition),
    );
  }
  if (tensors.attentionMask) {
    writeGraphCaptureInputTensor(
      runtime.device,
      tensors.attentionMask,
      cacheAttentionMaskTensor(inputs[names.attentionMask], cacheLength, runtime.contextLength),
    );
  }
  if (tensors.positionIndex) {
    writeGraphCaptureInputTensor(
      runtime.device,
      tensors.positionIndex,
      scalarFillTensor(inputs[names.positionIndex], samplePosition),
    );
  }
}

function graphCaptureStepFeeds(runtime, stepRuntime) {
  const state = stepRuntime.graphCapture;
  const names = state.names;
  const tensors = state.tensors;
  const cache = state.fixedCache ?? runtime.cache;
  return {
    [names.sampleNoise]: tensors.sampleNoise,
    ...(names.contextNoise ? { [names.contextNoise]: tensors.contextNoise } : {}),
    [names.actions]: tensors.actions,
    [names.kCache]: cache.k,
    [names.vCache]: cache.v,
    ...(names.cacheLength ? { [names.cacheLength]: tensors.cacheLength } : {}),
    ...(names.samplePositionIndex
      ? { [names.samplePositionIndex]: tensors.samplePositionIndex }
      : {}),
    ...(names.contextPositionIndex
      ? { [names.contextPositionIndex]: tensors.contextPositionIndex }
      : {}),
    ...(names.attentionMask ? { [names.attentionMask]: tensors.attentionMask } : {}),
    ...(names.positionIndex ? { [names.positionIndex]: tensors.positionIndex } : {}),
  };
}

function assertGraphCaptureGpuTensors(label, tensors) {
  for (const [name, tensor] of Object.entries(tensors ?? {})) {
    const gpuTensor = tensor as any;
    if (!gpuTensor) throw new Error(`${label} ${name} is missing.`);
    if (gpuTensor.location !== 'gpu-buffer') {
      throw new Error(`${label} ${name} must be a GPU tensor, got ${gpuTensor.location}.`);
    }
    if (!gpuTensor.gpuBuffer) throw new Error(`${label} ${name} has no GPU buffer.`);
  }
}

function advanceCacheLength(cacheLengthTensor, contextLength) {
  cacheLengthTensor.data[0] = Math.min(cacheLengthTensor.data[0] + 1, contextLength);
}

function disposeGpuTensor(tensor) {
  if (tensor?.location === 'gpu-buffer') {
    tensor.dispose();
  }
}

function disposeGpuTensorAfterSubmittedWork(device, tensor) {
  if (tensor?.location !== 'gpu-buffer') return;
  const dispose = () => {
    try {
      tensor.dispose();
    } catch {
      // The tensor may already have been released by ORT or by a reset path.
    }
  };
  if (device?.queue?.onSubmittedWorkDone) {
    void device.queue.onSubmittedWorkDone().then(dispose, dispose);
  } else {
    dispose();
  }
}

function tensorIsPinned(tensor, pinnedTensors = []) {
  return pinnedTensors.some((pinned) => pinned === tensor);
}

function disposeGpuTensorUnlessPinned(tensor, pinnedTensors = []) {
  if (!tensorIsPinned(tensor, pinnedTensors)) {
    disposeGpuTensor(tensor);
  }
}

function disposeGpuTensorUnlessPinnedAfterSubmittedWork(device, tensor, pinnedTensors = []) {
  if (!tensorIsPinned(tensor, pinnedTensors)) {
    disposeGpuTensorAfterSubmittedWork(device, tensor);
  }
}

function disposeStepGpuTensor(tensor, preserve) {
  if (preserve) return;
  disposeGpuTensor(tensor);
}

function disposeCache(cache) {
  disposeGpuTensor(cache?.k);
  disposeGpuTensor(cache?.v);
}

function cacheMatchesGraphCaptureFixedCache(cache, state) {
  return Boolean(
    cache &&
      state?.fixedCache &&
      cache.k === state.fixedCache.k &&
      cache.v === state.fixedCache.v,
  );
}

function cacheMatchesAnyGraphCaptureFixedCache(runtime, cache) {
  return (
    cacheMatchesGraphCaptureFixedCache(cache, runtime?.graphCapture) ||
    cacheMatchesGraphCaptureFixedCache(cache, runtime?.fullGraphCapture)
  );
}

function disposeRuntimeCache(runtime, cache) {
  if (cacheMatchesAnyGraphCaptureFixedCache(runtime, cache)) return;
  disposeCache(cache);
}

function resetGraphCaptureFixedCacheState(runtime) {
  if (runtime?.graphCapture) {
    runtime.graphCapture.fixedCacheReady = false;
    runtime.graphCapture.capturedOnce = false;
  }
  if (runtime?.fullGraphCapture) {
    runtime.fullGraphCapture.fixedCacheReady = false;
    runtime.fullGraphCapture.capturedOnce = false;
  }
}

function ensureGraphCaptureFixedCache(runtime, state) {
  if (!state?.fixedCache || cacheMatchesGraphCaptureFixedCache(runtime.cache, state)) {
    if (state?.fixedCache) state.fixedCacheReady = true;
    return;
  }
  const previousCache = runtime.cache;
  copyTensorToGpu(runtime.device, previousCache.k, state.fixedCache.k);
  copyTensorToGpu(runtime.device, previousCache.v, state.fixedCache.v);
  runtime.cache = {
    k: state.fixedCache.k,
    v: state.fixedCache.v,
    length: previousCache.length,
  };
  state.fixedCacheReady = true;
  disposeRuntimeCache(runtime, previousCache);
}

function contextFrameTensor(tensor, frameIndex, dtype = 'float32') {
  const start = frameIndex * CONTEXT_TENSOR_SIZE;
  const end = start + CONTEXT_TENSOR_SIZE;
  return makeFloatTensor(dtype, tensor.data.slice(start, end), [1, 1, 32, 32]);
}

function pixelTensorToImageData(tensor, frameIndex) {
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
  return image;
}

function renderPixelTensor(tensor, frameIndex) {
  const image = pixelTensorToImageData(tensor, frameIndex);
  const context = canvas2dContext();
  if (context) {
    context.putImageData(image, 0, 0);
  } else {
    renderImageDataFallback(image);
  }
}

function createPatchRenderMap(preprocessor) {
  const width = preprocessor.image_width;
  const height = preprocessor.image_height;
  const patchSize = preprocessor.patch_size;
  const xLen = preprocessor.x_len;
  const channels = preprocessor.num_channels;
  const patchDim = preprocessor.patch_dim;
  const sourceOffsets = new Int32Array(width * height).fill(-1);

  for (let py = 0; py < preprocessor.y_len; py += 1) {
    for (let px = 0; px < xLen; px += 1) {
      const patchIndex = py * xLen + px;
      const patchOffset = patchIndex * patchDim;
      for (let iy = 0; iy < patchSize; iy += 1) {
        const y = py * patchSize + iy - preprocessor.pad_width[0];
        if (y < 0 || y >= height) continue;
        for (let ix = 0; ix < patchSize; ix += 1) {
          const x = px * patchSize + ix - preprocessor.pad_width[1];
          if (x < 0 || x >= width) continue;
          sourceOffsets[y * width + x] = patchOffset + (iy * patchSize + ix) * channels;
        }
      }
    }
  }

  return { width, height, sourceOffsets };
}

function patchesToImageData(patchesTensor, preprocessor, targetImage = null, renderMap = null) {
  const width = preprocessor.image_width;
  const height = preprocessor.image_height;
  const image =
    targetImage?.width === width && targetImage?.height === height
      ? targetImage
      : new ImageData(width, height);

  const map = renderMap ?? createPatchRenderMap(preprocessor);
  const output = image.data;
  const patches = patchesTensor.data;
  const packedFloat16 = patchesTensor.type === 'float16' && !isNativeFloat16Array(patches);
  for (let pixel = 0, target = 0; pixel < map.sourceOffsets.length; pixel += 1, target += 4) {
    const source = map.sourceOffsets[pixel];
    if (source < 0) {
      output[target] = 0;
      output[target + 1] = 0;
      output[target + 2] = 0;
      output[target + 3] = 255;
      continue;
    }
    const r = packedFloat16 ? float16BitsToFloat32(patches[source]) : patches[source];
    const g = packedFloat16 ? float16BitsToFloat32(patches[source + 1]) : patches[source + 1];
    const b = packedFloat16 ? float16BitsToFloat32(patches[source + 2]) : patches[source + 2];
    output[target] = Math.max(0, Math.min(255, Math.round(r * 255)));
    output[target + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
    output[target + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
    output[target + 3] = 255;
  }
  return image;
}

function createWebgpuPatchRenderer(device, canvas, preprocessor) {
  if (!navigator.gpu || typeof canvas.getContext !== 'function' || !preprocessor) return null;
  const renderCanvas = createDisplayCanvasLike(canvas);
  const context = renderCanvas.getContext('webgpu');
  if (!context) return null;

  const width = preprocessor.image_width;
  const height = preprocessor.image_height;
  const patchSize = preprocessor.patch_size;
  const xLen = preprocessor.x_len;
  const yLen = preprocessor.y_len;
  const channels = preprocessor.num_channels;
  const patchDim = preprocessor.patch_dim;
  const padY = preprocessor.pad_width?.[0] ?? 0;
  const padX = preprocessor.pad_width?.[1] ?? 0;
  const format = navigator.gpu.getPreferredCanvasFormat();

  renderCanvas.width = width;
  renderCanvas.height = height;
  context.configure({
    device,
    format,
    alphaMode: 'opaque',
  });

  const shader = device.createShaderModule({
    label: 'visionary-patch-renderer',
    code: `
const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const PATCH_SIZE: u32 = ${patchSize}u;
const X_LEN: u32 = ${xLen}u;
const Y_LEN: u32 = ${yLen}u;
const CHANNELS: u32 = ${channels}u;
const PATCH_DIM: u32 = ${patchDim}u;
const PAD_X: u32 = ${padX}u;
const PAD_Y: u32 = ${padY}u;

@group(0) @binding(0) var<storage, read> patches: array<f32>;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  var out: VertexOut;
  out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
  return out;
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  let x = u32(position.x);
  let y = u32(position.y);
  if (x >= WIDTH || y >= HEIGHT) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let padded_x = x + PAD_X;
  let padded_y = y + PAD_Y;
  let patch_x = padded_x / PATCH_SIZE;
  let patch_y = padded_y / PATCH_SIZE;
  if (patch_x >= X_LEN || patch_y >= Y_LEN) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let in_patch_x = padded_x % PATCH_SIZE;
  let in_patch_y = padded_y % PATCH_SIZE;
  let patch_index = patch_y * X_LEN + patch_x;
  let source = patch_index * PATCH_DIM + (in_patch_y * PATCH_SIZE + in_patch_x) * CHANNELS;
  return vec4<f32>(
    clamp(patches[source], 0.0, 1.0),
    clamp(patches[source + 1u], 0.0, 1.0),
    clamp(patches[source + 2u], 0.0, 1.0),
    1.0,
  );
}
`,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'visionary-patch-renderer-bindings',
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }],
  });
  const pipeline = device.createRenderPipeline({
    label: 'visionary-patch-renderer',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module: shader, entryPoint: 'vs_main' },
    fragment: {
      module: shader,
      entryPoint: 'fs_main',
      targets: [{ format }],
    },
    primitive: { topology: 'triangle-list' },
  });
  let cachedBindGroup = null;
  let cachedBuffer = null;
  let attached = false;

  function attach() {
    if (attached && renderCanvas.isConnected && elements.canvas === renderCanvas) return;
    replaceDisplayCanvas(renderCanvas);
    attached = true;
  }

  return {
    kind: 'webgpu_patch_renderer',
    render(patchesTensor) {
      if (patchesTensor.location !== 'gpu-buffer') {
        throw new Error(`WebGPU patch renderer requires a GPU tensor, got ${patchesTensor.location}`);
      }
      attach();
      if (!cachedBindGroup || cachedBuffer !== patchesTensor.gpuBuffer) {
        cachedBindGroup = device.createBindGroup({
          label: 'visionary-patch-renderer-bind-group',
          layout: bindGroupLayout,
          entries: [{ binding: 0, resource: { buffer: patchesTensor.gpuBuffer } }],
        });
        cachedBuffer = patchesTensor.gpuBuffer;
      }
      const encoder = device.createCommandEncoder({ label: 'visionary-patch-renderer' });
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, cachedBindGroup);
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
    },
  };
}

function renderPatches(runtime, patchesTensor) {
  hidePreviewOverlay();
  if (runtime.patchRenderer && patchesTensor.location === 'gpu-buffer') {
    runtime.patchRenderer.render(patchesTensor);
    return;
  }
  if (runtime.patchRenderer && patchesTensor.location !== 'gpu-buffer') {
    runtime.patchRenderer = null;
  }
  const image = patchesToImageData(
    patchesTensor,
    runtime.preprocessor,
    runtime.frameImageData,
    runtime.patchRenderMap,
  );
  runtime.frameImageData = image;
  const context = canvas2dContext();
  if (context) {
    context.putImageData(image, 0, 0);
  } else {
    renderImageDataFallback(image);
  }
}

async function renderGpuPatchesPreviewOverlay(runtime, patchesTensor) {
  if (patchesTensor.location !== 'gpu-buffer' || typeof patchesTensor.getData !== 'function') {
    return;
  }
  const data = await patchesTensor.getData(false);
  const cpuPatches = new ort.Tensor(patchesTensor.type, data, patchesTensor.dims);
  renderImageDataPreviewOverlay(
    patchesToImageData(cpuPatches, runtime.preprocessor, null, runtime.patchRenderMap),
  );
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
    k: cloneTensorToGpu(device, initialCache.k),
    v: cloneTensorToGpu(device, initialCache.v),
    length: cloneCpuTensor(initialCache.length),
  };
}

function resetCacheFromInitialArtifacts(runtime) {
  resetGraphCaptureFixedCacheState(runtime);
  if (
    runtime.backend === 'webgpu' &&
    runtime.graphCapture?.enabled &&
    runtime.cache &&
    !cacheMatchesAnyGraphCaptureFixedCache(runtime, runtime.cache)
  ) {
    copyTensorToGpu(runtime.device, runtime.initialCache.k, runtime.cache.k);
    copyTensorToGpu(runtime.device, runtime.initialCache.v, runtime.cache.v);
    runtime.cache.length = cloneCpuTensor(runtime.initialCache.length);
    return;
  }
  const previousCache = runtime.cache;
  runtime.cache = cacheFromInitialArtifacts(runtime.device, runtime.initialCache, runtime.backend);
  disposeRuntimeCache(runtime, previousCache);
}

async function renderLatent(zTensor, options: { previewOverlay?: boolean } = {}) {
  const decoderOutputs = await runDecoder(runtime, zTensor);
  const patches = decoderOutputs[runtime.names.patches];
  renderPatches(runtime, patches);
  if (options.previewOverlay) {
    await renderGpuPatchesPreviewOverlay(runtime, patches);
  }
  disposeGpuTensorUnlessPinnedAfterSubmittedWork(runtime.device, patches, runtime.pinnedOutputTensors);
}

function shouldUseDecoderGraphCapture(runtime) {
  return Boolean(
    runtime.decoderGraphCapture &&
      (!runtime.deferDecoderGraphCapture || runtime.fullGraphCapture?.capturedOnce),
  );
}

async function runDecoder(runtime, zTensor) {
  let useGraphCapture = shouldUseDecoderGraphCapture(runtime);
  if (useGraphCapture && !runtime.sessions.decoderGraphCapture) {
    useGraphCapture = await ensureDecoderGraphCaptureSession(runtime);
  }
  const session = useGraphCapture ? runtime.sessions.decoderGraphCapture : runtime.sessions.decoder;
  const decoderInput = useGraphCapture ? runtime.decoderGraphInput : runtime.decoderInput;
  try {
    return await session.run(
      decoderFeedsFromZ(runtime, zTensor, decoderInput),
      useGraphCapture && runtime.decoderOutputFetches
        ? runtime.decoderOutputFetches
        : [runtime.names.patches],
    );
  } catch (error) {
    if (!useGraphCapture) throw error;
    runtime.decoderGraphCapture = false;
    runtime.decoderGraphCapturePending = false;
    console.warn('Falling back to WebGPU decoder without graph capture:', error);
    return runtime.sessions.decoder.run(
      decoderFeedsFromZ(runtime, zTensor, runtime.decoderInput),
      [runtime.names.patches],
    );
  }
}

async function ensureDecoderGraphCaptureSession(runtime) {
  if (runtime.sessions.decoderGraphCapture) return true;
  if (!runtime.decoderGraphCapturePending || !runtime.decoderGraphInput) return false;
  try {
    runtime.sessions.decoderGraphCapture = await createSession(
      runtime.specs.decoder,
      'decoder graph capture',
      runtime.decoderModelBytes,
      runtime.backend,
      {
        preferredOutputLocation: { patches: runtime.patchRenderer ? 'gpu-buffer' : 'cpu' },
        enableGraphCapture: true,
      },
    );
    runtime.decoderOutputFetches = createGpuOutputFetches(runtime.device, runtime.specs.decoder, [
      runtime.names.patches,
    ]);
    runtime.decoderGraphCapturePending = false;
    runtime.decoderGraphCapture = true;
    return true;
  } catch (error) {
    recordLoadEvent('decoder graph capture unavailable', 0);
    console.warn('Falling back to WebGPU decoder without graph capture:', error);
    runtime.decoderGraphCapturePending = false;
    runtime.decoderGraphCapture = false;
    return false;
  }
}

function setAction(action) {
  currentAction = Number(action);
  const definition = ACTION_BY_ID.get(currentAction) as { label: string } | undefined;
  elements.action.textContent = definition?.label ?? String(currentAction);
  for (const element of elements.actionButtons) {
    const active = Number(element.dataset.actionId) === currentAction;
    element.classList.toggle('active', active);
    element.setAttribute('aria-pressed', active ? 'true' : 'false');
  }
}

function actionFromPressedKeys() {
  let bestAction = null;
  let bestScore = -1;
  for (const action of ACTION_DEFINITIONS) {
    for (const combo of action.keyCombos) {
      if (!combo.every((code) => activeKeyCodes.has(code))) continue;
      const recency = Math.max(...combo.map((code) => activeKeyCodes.get(code)));
      const score = combo.length * 100000 + recency;
      if (score > bestScore) {
        bestAction = action;
        bestScore = score;
      }
    }
  }
  return bestAction?.id ?? NOOP_ACTION;
}

function actionFromKeys(event, pressed) {
  if (!BOUND_KEY_CODES.has(event.code)) return;
  event.preventDefault();
  if (pressed) {
    if (!activeKeyCodes.has(event.code)) {
      activeKeyCodes.set(event.code, ++keySequence);
    }
  } else {
    activeKeyCodes.delete(event.code);
  }
  setAction(actionFromPressedKeys());
}

function bindActionButton(element, action) {
  element.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    element.setPointerCapture?.(event.pointerId);
    setAction(action);
  });
  const release = (event) => {
    event.preventDefault();
    if (currentAction === action) setAction(actionFromPressedKeys());
  };
  element.addEventListener('pointerup', release);
  element.addEventListener('pointercancel', release);
  element.addEventListener('lostpointercapture', () => {
    if (currentAction === action) setAction(actionFromPressedKeys());
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

async function createPrefilledInitialCache(backend, loaded) {
  if (!prefillInitialCacheRequested || !loaded.prefillSpec) return null;
  const names = prefillCacheOutputNames(loaded.prefillSpec);
  let prefillSession = null;
  try {
    prefillSession = await createSession(
      loaded.prefillSpec,
      'prefill cache',
      loaded.prefillModelBytes,
      backend,
      backend === 'webgpu'
        ? {
            preferredOutputLocation: {
              [names.k]: 'cpu',
              [names.v]: 'cpu',
              [names.length]: 'cpu',
            },
          }
        : {},
    );
    setStatus(`Creating prefilled cache · ${backend}`);
    const outputs = await prefillSession.run(
      prefillFeedsFromContext(loaded.prefillSpec, loaded),
      [names.k, names.v, names.length],
    );
    return {
      k: outputs[names.k],
      v: outputs[names.v],
      length: outputs[names.length],
    };
  } finally {
    await releaseSession(prefillSession);
  }
}

function graphCaptureStepOutputLocations(stepSpec) {
  return Object.fromEntries(Object.keys(stepSpec.outputs ?? {}).map((name) => [name, 'gpu-buffer']));
}

function graphCaptureUnavailableReason(stepSpec, stepNames, requested) {
  if (!requested) return 'disabled by config';
  if (stepNames.length) return 'step graph returns cache_length on GPU';
  if (!graphCaptureStepInputNames(stepSpec)) return 'step graph has unsupported inputs';
  return null;
}

async function createStepSessionsForBackend(
  backend,
  stepSpec,
  stepModelBytes,
  stepNames,
  graphCaptureRequestedForSpec = dynamicsGraphCaptureEnabled,
) {
  const normalSession = await createSession(stepSpec, 'dynamics', stepModelBytes, backend, {
    preferredOutputLocation: {
      final_z: 'gpu-buffer',
      candidate_k_entry: 'gpu-buffer',
      candidate_v_entry: 'gpu-buffer',
    },
  });
  const graphCaptureReason =
    backend === 'webgpu'
      ? graphCaptureUnavailableReason(stepSpec, stepNames, graphCaptureRequestedForSpec)
      : 'backend is not webgpu';
  if (backend === 'webgpu' && !graphCaptureReason) {
    try {
      return {
        session: normalSession,
        graphCaptureSession: await createSession(stepSpec, 'dynamics graph capture', stepModelBytes, backend, {
          preferredOutputLocation: graphCaptureStepOutputLocations(stepSpec),
          enableGraphCapture: true,
        }),
        graphCaptureEnabled: true,
        graphCaptureReason: null,
      };
    } catch (error) {
      recordLoadEvent('graph capture unavailable', 0);
      console.warn('Falling back to WebGPU without graph capture:', error);
    }
  }

  return {
    session: normalSession,
    graphCaptureSession: null,
    graphCaptureEnabled: false,
    graphCaptureReason,
  };
}

async function createRuntimeForBackend(backend, loaded) {
  let stepSession = null;
  let stepGraphCaptureSession = null;
  let fullStepSession = null;
  let fullStepGraphCaptureSession = null;
  let decoderSession = null;
  let decoderGraphCaptureSession = null;
  try {
    const stepNames = stepNamesForSpec(loaded.stepSpec);
    const contextLength =
      loaded.manifest.cache_contract?.context_length ??
      loaded.initialCacheManifest.arrays.k_cache.shape[3];
    const initialCacheLength = loaded.initialLength.data[0];
    const fullStepNames = loaded.fullStepSpec ? stepNamesForSpec(loaded.fullStepSpec) : null;
    const skipStepRuntime =
      skipShortCacheStepWhenFull &&
      loaded.fullStepSpec &&
      fullStepNames &&
      initialCacheLength >= contextLength;
    const stepRuntime = skipStepRuntime
      ? {
          session: null,
          graphCaptureSession: null,
          graphCaptureEnabled: false,
          graphCaptureReason: 'skipped because initial cache is full',
        }
      : await createStepSessionsForBackend(
          backend,
          loaded.stepSpec,
          loaded.stepModelBytes,
          stepNames,
        );
    stepSession = stepRuntime.session;
    stepGraphCaptureSession = stepRuntime.graphCaptureSession;
    const fullStepRuntime = loaded.fullStepSpec
      ? await createStepSessionsForBackend(
          backend,
          loaded.fullStepSpec,
          loaded.fullStepModelBytes,
          fullStepNames,
          fullDynamicsGraphCaptureRequestedForSpec(loaded.manifest, loaded.fullStepSpec),
        )
      : null;
    fullStepSession = fullStepRuntime?.session ?? null;
    fullStepGraphCaptureSession = fullStepRuntime?.graphCaptureSession ?? null;
    const device = backend === 'webgpu' ? ort.env.webgpu?.device : null;
    if (backend === 'webgpu' && !device) {
      throw new Error('WebGPU session was created but ORT did not expose a GPU device.');
    }
    const prefilledInitialCache = await createPrefilledInitialCache(backend, loaded);
    const initialCache = prefilledInitialCache ?? {
      k: loaded.initialK,
      v: loaded.initialV,
      length: loaded.initialLength,
    };
    const patchRenderer =
      backend === 'webgpu' && device && gpuPatchRendererEnabled
        ? createWebgpuPatchRenderer(device, elements.canvas, loaded.contextManifest.preprocessor)
        : null;
    const deferDecoderGraphCapture = false;
    const decoderInput = createDecoderInputState(
      device,
      loaded.decoderSpec,
      backend,
      backend === 'webgpu',
    );
    const decoderGraphInput =
      backend === 'webgpu' && decoderGraphCaptureEnabled
        ? deferDecoderGraphCapture
          ? createDecoderInputState(device, loaded.decoderSpec, backend, true)
          : decoderInput
        : null;
    const decoderGraphCapturePending = Boolean(deferDecoderGraphCapture && decoderGraphInput);
    const decoderRuntime = await createDecoderSessionsForBackend(
      backend,
      loaded.decoderSpec,
      loaded.decoderModelBytes,
      patchRenderer,
      decoderGraphCapturePending ? null : decoderGraphInput,
    );
    decoderSession = decoderRuntime.session;
    decoderGraphCaptureSession = decoderRuntime.graphCaptureSession;
    const decoderOutputFetches =
      backend === 'webgpu' && decoderRuntime.graphCaptureEnabled && preallocateDecoderOutputsEnabled
        ? createGpuOutputFetches(device, loaded.decoderSpec, [
            outputName(loaded.decoderSpec, 'patches'),
          ])
        : null;
    const stepOutputFetches =
      backend === 'webgpu' &&
      device &&
      preallocateStepOutputsEnabled &&
      stepRuntime.session &&
      !stepRuntime.graphCaptureEnabled
        ? createGpuOutputFetches(device, loaded.stepSpec, [stepNames.finalZ, stepNames.k, stepNames.v])
        : null;
    const fullStepOutputFetches =
      backend === 'webgpu' &&
      device &&
      preallocateStepOutputsEnabled &&
      loaded.fullStepSpec &&
      fullStepNames &&
      !fullStepRuntime?.graphCaptureEnabled
        ? createGpuOutputFetches(device, loaded.fullStepSpec, [
            fullStepNames.finalZ,
            fullStepNames.k,
            fullStepNames.v,
          ])
        : null;

    let graphCapture = null;
    if (stepRuntime.graphCaptureEnabled) {
      graphCapture = createGraphCaptureStepState(
        device,
        loaded.stepSpec,
        loaded.manifest.cache_contract?.context_length ??
          loaded.initialCacheManifest.arrays.k_cache.shape[3],
        {
          preallocateOutputs: preallocateStepOutputsEnabled,
        },
      );
      if (!graphCapture) {
        throw new Error('Graph capture was enabled, but fixed GPU inputs could not be created.');
      }
    }
    let fullGraphCapture = null;
    if (fullStepRuntime?.graphCaptureEnabled) {
      fullGraphCapture = createGraphCaptureStepState(
        device,
        loaded.fullStepSpec,
        loaded.manifest.cache_contract?.context_length ??
          loaded.initialCacheManifest.arrays.k_cache.shape[3],
        {
          fixedCache: true,
          preallocateOutputs: preallocateStepOutputsEnabled,
        },
      );
      if (!fullGraphCapture) {
        throw new Error('Full-cache graph capture was enabled, but fixed GPU inputs could not be created.');
      }
    }
    reuseFinalZOutputAsDecoderInput(graphCapture, loaded.stepSpec, stepNames, decoderInput);
    reuseFinalZOutputAsDecoderInput(fullGraphCapture, loaded.fullStepSpec, fullStepNames, decoderInput);

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
      patchRenderer,
      sessions: {
        step: stepSession,
        stepGraphCapture: stepGraphCaptureSession,
        fullStep: fullStepSession,
        fullStepGraphCapture: fullStepGraphCaptureSession,
        decoder: decoderSession,
        decoderGraphCapture: decoderGraphCaptureSession,
      },
      specs: {
        step: loaded.stepSpec,
        fullStep: loaded.fullStepSpec,
        decoder: loaded.decoderSpec,
      },
      names: {
        step: stepNames,
        fullStep: fullStepNames,
        patches: outputName(loaded.decoderSpec, 'patches'),
      },
      dtypes: {
        sampleNoise: loaded.stepSpec.inputs.sample_noise.dtype,
      },
      initialCache,
      initialCacheSource: prefilledInitialCache ? 'prefill' : 'artifact',
      contextLength,
      contextZ: loaded.contextZ,
      displayZ: loaded.displayZ,
      displayPixels: loaded.displayPixels,
      patchRenderMap: createPatchRenderMap(loaded.contextManifest.preprocessor),
      cacheUpdater,
      graphCapture,
      fullGraphCapture,
      decoderGraphCapture: decoderRuntime.graphCaptureEnabled || decoderGraphCapturePending,
      decoderGraphCapturePending,
      deferDecoderGraphCapture,
      decoderInput,
      decoderGraphInput,
      decoderOutputFetches,
      stepOutputFetches,
      fullStepOutputFetches,
      decoderModelBytes: loaded.decoderModelBytes,
      pinnedOutputTensors: [
        ...Object.values(graphCapture?.outputFetches ?? {}),
        ...Object.values(fullGraphCapture?.outputFetches ?? {}),
        ...Object.values(decoderOutputFetches ?? {}),
        ...Object.values(stepOutputFetches ?? {}),
        ...Object.values(fullStepOutputFetches ?? {}),
      ],
      cache: null,
    };

    loadedRuntime.cache = cacheFromInitialArtifacts(device, loadedRuntime.initialCache, backend);
    const readyGraphCaptures = [];
    if (graphCapture?.enabled || fullGraphCapture?.enabled) {
      readyGraphCaptures.push('dynamics graph capture');
    }
    if (decoderRuntime.graphCaptureEnabled) {
      readyGraphCaptures.push('decoder graph capture');
    }
    const graphCaptureLabel = readyGraphCaptures.length
      ? ` + ${readyGraphCaptures.join(' + ')} ready`
      : '';
    elements.backend.textContent = `${backend}${graphCaptureLabel}`;
    return loadedRuntime;
  } catch (error) {
    await Promise.all([
      releaseSession(stepSession),
      releaseSession(stepGraphCaptureSession),
      releaseSession(fullStepSession),
      releaseSession(fullStepGraphCaptureSession),
      releaseSession(decoderSession),
      releaseSession(decoderGraphCaptureSession),
    ]);
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
    ...STEP_EXPORT_FALLBACKS,
  ]);
  const fullStepSpec = fullCacheStepEnabled
    ? findFirstOptionalExport(manifest, [
        FULL_CACHE_STEP_EXPORT_NAME,
        browserProfile === 'safari'
          ? manifest.demo_generation?.preferred_full_cache_step_export_safari
          : null,
        manifest.demo_generation?.preferred_full_cache_step_export,
        ...FULL_CACHE_STEP_EXPORT_FALLBACKS,
      ])
    : null;
  const prefillSpec = prefillInitialCacheRequested
    ? findFirstExport(manifest, [
        manifest.demo_generation?.preferred_prefill_export,
        ...PREFILL_EXPORT_FALLBACKS,
      ])
    : null;
  const decoderSpec = DECODER_EXPORT_NAME
    ? findExport(manifest, DECODER_EXPORT_NAME)
    : findFirstExport(manifest, [
        manifest.demo_generation?.preferred_decoder_export,
        ...DECODER_EXPORT_FALLBACKS,
      ]);

  setStatus('Loading context preview and initial cache');
  const displayPixelsPromise = contextManifest.arrays.display_pixels
    ? fetchTensorFromArtifact(ASSET_DIR, contextManifest.arrays.display_pixels, 'context preview pixels')
    : Promise.resolve(null);
  const [
    displayZ,
    displayPixels,
    initialK,
    initialV,
    initialLength,
    contextZ,
    contextActions,
    contextStepLevels,
    contextSignalLevels,
  ] = await Promise.all([
    fetchTensorFromArtifact(ASSET_DIR, contextManifest.arrays.display_z, 'context preview'),
    displayPixelsPromise,
    fetchTensorFromArtifact(ASSET_DIR, initialCacheManifest.arrays.k_cache, 'initial K cache'),
    fetchTensorFromArtifact(ASSET_DIR, initialCacheManifest.arrays.v_cache, 'initial V cache'),
    fetchTensorFromArtifact(ASSET_DIR, initialCacheManifest.arrays.cache_length, 'cache length'),
    prefillSpec
      ? fetchTensorFromArtifact(ASSET_DIR, contextManifest.arrays.z, 'prefill context')
      : Promise.resolve(null),
    prefillSpec
      ? fetchTensorFromArtifact(ASSET_DIR, contextManifest.arrays.actions, 'prefill actions')
      : Promise.resolve(null),
    prefillSpec
      ? fetchTensorFromArtifact(ASSET_DIR, contextManifest.arrays.step_levels, 'prefill step levels')
      : Promise.resolve(null),
    prefillSpec
      ? fetchTensorFromArtifact(ASSET_DIR, contextManifest.arrays.signal_levels, 'prefill signal levels')
      : Promise.resolve(null),
  ]);
  validateInitialCache(stepSpec, initialK, initialV, initialLength);
  elements.context.textContent = `${contextManifest.prefix_frames} frames @ ${contextManifest.episode_start}`;

  setStatus('Loading ONNX models');
  const [stepModelBytes, fullStepModelBytes, prefillModelBytes, decoderModelBytes] = await Promise.all([
    fetchBytes(`${ASSET_DIR}/${stepSpec.path}`, 'dynamics model'),
    fullStepSpec
      ? fetchBytes(`${ASSET_DIR}/${fullStepSpec.path}`, 'full-cache dynamics model')
      : Promise.resolve(null),
    prefillSpec
      ? fetchBytes(`${ASSET_DIR}/${prefillSpec.path}`, 'prefill model')
      : Promise.resolve(null),
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
    contextZ,
    contextActions,
    contextStepLevels,
    contextSignalLevels,
    initialK,
    initialV,
    initialLength,
    stepModelBytes,
    fullStepSpec,
    fullStepModelBytes,
    prefillSpec,
    prefillModelBytes,
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
  resetCacheFromInitialArtifacts(runtime);
  frameCount = 0;
  lastStatsUpdateTime = 0;
  statsFramesSinceUpdate = 0;
  noiseGenerator = new NormalNoiseGenerator(runtime.contextManifest.noise_seed ?? 0);
  elements.frameCount.textContent = '0';
  elements.latency.textContent = '-- ms';
  const prefixFrames = runtime.contextManifest.prefix_frames ?? 1;
  const usePrefillPreview = runtime.initialCacheSource === 'prefill' && runtime.contextZ;
  const avoidGpuPreviewBeforeGraphCapture =
    browserProfile === 'safari' &&
    Boolean(runtime.patchRenderer) &&
    Boolean(runtime.graphCapture?.enabled || runtime.fullGraphCapture?.enabled);
  const previewSource = usePrefillPreview ? runtime.contextZ : runtime.displayZ;
  const previewFrameIndex = usePrefillPreview ? runtime.contextLength - 1 : prefixFrames - 1;
  const previewTensor = contextFrameTensor(
    previewSource,
    previewFrameIndex,
    decoderInputDtype(runtime.specs.decoder),
  );
  if (runtime.displayPixels && (!usePrefillPreview || avoidGpuPreviewBeforeGraphCapture)) {
    if (runtime.patchRenderer && !avoidGpuPreviewBeforeGraphCapture) {
      await renderLatent(previewTensor);
      renderImageDataPreviewOverlay(pixelTensorToImageData(runtime.displayPixels, prefixFrames - 1));
    } else {
      renderPixelTensor(runtime.displayPixels, prefixFrames - 1);
    }
  } else {
    await renderLatent(previewTensor, {
      previewOverlay: browserProfile === 'safari' && Boolean(runtime.patchRenderer),
    });
  }
  const readyGraphCaptures = [];
  if (runtime.graphCapture?.enabled || runtime.fullGraphCapture?.enabled) {
    readyGraphCaptures.push('dynamics graph capture');
  }
  if (runtime.decoderGraphCapture) {
    readyGraphCaptures.push('decoder graph capture');
  }
  const graphCaptureLabel = readyGraphCaptures.length
    ? ` · ${readyGraphCaptures.join(' + ')} ready`
    : '';
  setStatus(
    `Ready · ${runtime.backend}${graphCaptureLabel} · cache length ${runtime.initialCache.length.data[0]}`,
  );
}

async function generateFrame() {
  const started = performance.now();
  const cacheLengthBefore = runtime.cache.length.data[0];
  const useFullCacheStep =
    runtime.sessions.fullStep &&
    runtime.specs.fullStep &&
    runtime.names.fullStep &&
    cacheLengthBefore >= runtime.contextLength;
  const activeStep = {
    spec: useFullCacheStep ? runtime.specs.fullStep : runtime.specs.step,
    names: useFullCacheStep ? runtime.names.fullStep : runtime.names.step,
    session: useFullCacheStep ? runtime.sessions.fullStep : runtime.sessions.step,
    graphCaptureSession: useFullCacheStep
      ? runtime.sessions.fullStepGraphCapture
      : runtime.sessions.stepGraphCapture,
    graphCapture: useFullCacheStep ? runtime.fullGraphCapture : runtime.graphCapture,
    outputFetches: useFullCacheStep ? runtime.fullStepOutputFetches : runtime.stepOutputFetches,
  };
  if (!activeStep.session) {
    throw new Error('Short-cache dynamics session was skipped, but the cache is not full.');
  }
  const useGraphCaptureStep =
    activeStep.graphCapture?.enabled &&
    activeStep.graphCaptureSession &&
    (!activeStep.graphCapture.fixedCache || cacheLengthBefore >= runtime.contextLength);
  if (runtime.backend === 'webgpu') {
    const activeCapture = useGraphCaptureStep
      ? ' + dynamics graph capture'
      : shouldUseDecoderGraphCapture(runtime)
        ? ' + decoder graph capture'
        : '';
    elements.backend.textContent = `${runtime.backend}${activeCapture}`;
  }
  const frameInputs = frameInputStateFor(activeStep.spec);
  const action = actionTensorForFrame(frameInputs);
  const noiseInputs = currentNoiseInputSlot(frameInputs);
  const sampleNoise = noiseInputs.sampleNoise;
  const contextNoise = noiseInputs.contextNoise;
  const fetchNames = [activeStep.names.finalZ, activeStep.names.k, activeStep.names.v];
  if (activeStep.names.length) fetchNames.push(activeStep.names.length);
  const normalStepFeeds = {
    sample_noise: sampleNoise,
    context_noise: contextNoise,
    actions: action,
    k_cache: runtime.cache.k,
    v_cache: runtime.cache.v,
    ...stepPositionFeeds(
      activeStep.spec,
      cacheLengthBefore,
      runtime.contextLength,
      runtime.cache.length,
      frameInputs.positionInputs,
    ),
  };
  let usedGraphCaptureStep = useGraphCaptureStep;
  let outputs;
  try {
    if (usedGraphCaptureStep) {
      ensureGraphCaptureFixedCache(runtime, activeStep.graphCapture);
      const graphCaptureFeeds = (() => {
        updateGraphCaptureStepInputs(
          runtime,
          activeStep,
          sampleNoise,
          contextNoise,
          action,
          cacheLengthBefore,
        );
        return graphCaptureStepFeeds(runtime, activeStep);
      })();
      if (graphCaptureUploadFenceEnabled) await runtime.device.queue.onSubmittedWorkDone();
      outputs = await activeStep.graphCaptureSession.run(
        graphCaptureFeeds,
        activeStep.graphCapture?.outputFetches ?? fetchNames,
      );
    } else {
      outputs = await activeStep.session.run(normalStepFeeds, activeStep.outputFetches ?? fetchNames);
    }
  } catch (error) {
    if (!usedGraphCaptureStep) throw error;
    activeStep.graphCapture.enabled = false;
    activeStep.graphCapture.failedReason = error instanceof Error ? error.message : String(error);
    usedGraphCaptureStep = false;
    elements.backend.textContent = shouldUseDecoderGraphCapture(runtime)
      ? `${runtime.backend} + decoder graph capture`
      : runtime.backend;
    console.warn('Falling back to WebGPU without dynamics graph capture:', error);
    outputs = await activeStep.session.run(
      {
        sample_noise: sampleNoise,
        context_noise: contextNoise,
        actions: action,
        k_cache: runtime.cache.k,
        v_cache: runtime.cache.v,
        ...stepPositionFeeds(
          activeStep.spec,
          cacheLengthBefore,
          runtime.contextLength,
          runtime.cache.length,
          frameInputs.positionInputs,
        ),
      },
      activeStep.outputFetches ?? fetchNames,
    );
  }
  prefillNextNoiseInputSlot(frameInputs);
  if (usedGraphCaptureStep) activeStep.graphCapture.capturedOnce = true;
  const preserveStepOutputs = Boolean(
    (usedGraphCaptureStep && activeStep.graphCapture?.outputFetches) || activeStep.outputFetches,
  );
  const zOutput = outputs[activeStep.names.finalZ];

  const decoderOutputs = await runDecoder(runtime, zOutput);
  disposeStepGpuTensor(zOutput, preserveStepOutputs);

  const patches = decoderOutputs[runtime.names.patches];
  renderPatches(runtime, patches);
  disposeGpuTensorUnlessPinnedAfterSubmittedWork(runtime.device, patches, runtime.pinnedOutputTensors);

  runtime.cacheUpdater.update(
    runtime.cache,
    outputs[activeStep.names.k],
    outputs[activeStep.names.v],
    runtime.cache.length,
  );
  if (cacheUpdateFenceEnabled && runtime.device) await runtime.device.queue.onSubmittedWorkDone();
  if (activeStep.names.length) {
    runtime.cache.length = outputs[activeStep.names.length];
  } else {
    advanceCacheLength(runtime.cache.length, runtime.contextLength);
  }
  if (!preserveStepOutputs) {
    disposeGpuTensorAfterSubmittedWork(runtime.device, outputs[activeStep.names.k]);
    disposeGpuTensorAfterSubmittedWork(runtime.device, outputs[activeStep.names.v]);
  }

  frameCount += 1;
  statsFramesSinceUpdate += 1;
  const elapsed = performance.now() - started;
  const now = performance.now();
  const statsWindowStart = lastStatsUpdateTime || lastFrameTime;
  const shouldUpdateStats =
    frameCount === 1 || now - statsWindowStart >= STATS_UPDATE_INTERVAL_MS;
  lastFrameTime = now;
  if (shouldUpdateStats) {
    const fps = (statsFramesSinceUpdate * 1000) / Math.max(now - statsWindowStart, 1);
    lastStatsUpdateTime = now;
    statsFramesSinceUpdate = 0;
    elements.frameCount.textContent = String(frameCount);
    elements.latency.textContent = `${elapsed.toFixed(1)} ms`;
    elements.fps.textContent = `${fps.toFixed(1)} fps`;
  }
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
  scheduleStreamLoop(delayMs);
}

function runScheduledStreamLoop() {
  streamLoopPending = false;
  void streamLoop();
}

function scheduleStreamLoop(delayMs) {
  if (streamLoopPending) return;
  streamLoopPending = true;
  if (delayMs > 0 || typeof MessageChannel === 'undefined') {
    window.setTimeout(runScheduledStreamLoop, delayMs);
    return;
  }
  if (!streamLoopChannel) {
    streamLoopChannel = new MessageChannel();
    streamLoopChannel.port1.onmessage = runScheduledStreamLoop;
  }
  streamLoopChannel.port2.postMessage(null);
}

elements.start.addEventListener('click', () => {
  if (!runtime?.cache) return;
  running = !running;
  elements.start.textContent = running ? 'Pause' : 'Start';
  if (running) {
    lastFrameTime = performance.now();
    lastStatsUpdateTime = lastFrameTime;
    statsFramesSinceUpdate = 0;
    scheduleStreamLoop(0);
  }
});

elements.reset.addEventListener('click', resetDemo);
elements.targetFps.addEventListener('change', () => {
  targetFps = parseTargetFps(elements.targetFps.value);
});
window.addEventListener('keydown', (event) => actionFromKeys(event, true));
window.addEventListener('keyup', (event) => actionFromKeys(event, false));
window.addEventListener('blur', () => {
  activeKeyCodes.clear();
  setAction(NOOP_ACTION);
});
for (const button of elements.actionButtons) {
  bindActionButton(button, Number(button.dataset.actionId));
}
(window as any).visionaryDemoActionsReady = true;

setAction(NOOP_ACTION);
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
  get frameCount() {
    return frameCount;
  },
  get loadEvents() {
    return loadEvents;
  },
  async generateFrame() {
    await generateFrame();
  },
};
