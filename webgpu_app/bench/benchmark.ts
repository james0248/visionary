// @ts-nocheck
import { applyCacheFeeds, cacheOutputNames, inputCacheNames } from '../runtime/cache';
import { byExportName, findSpec, sameShape } from '../runtime/manifest';
import { configureOrt } from '../runtime/ort';
import {
  copyGpuTensor,
  copyTensorToGpu,
  createEmptyGpuTensor as createEmptyGpuTensorWithOrt,
  createGpuTensorFromCpu as createGpuTensorFromCpuWithOrt,
  mul,
  tensorByteLength,
  tensorDataBytes,
  writeCpuTensorToGpu,
} from '../runtime/tensors';

export {};

let ASSET_DIR = '/dream_arcade_assets/breakout';
let MANIFEST_URL = `${ASSET_DIR}/breakout_onnx_manifest.json`;
const moduleParams = new URLSearchParams(window.location.search);
const requestedProvider = moduleParams.get('provider') ?? 'webgpu';
const DEFAULT_WASM_NUM_THREADS = 4;
const ORT_MODULE_URL =
  moduleParams.get('ortModule') ??
  (requestedProvider === 'wasm'
    ? '/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs'
    : '/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs');
const CAPTURE_CONSOLE = ['1', 'true', 'yes', 'on'].includes(
  (moduleParams.get('captureConsole') ?? '').toLowerCase(),
);
const capturedConsoleMessages = [];
if (CAPTURE_CONSOLE) {
  for (const level of ['log', 'warn', 'error']) {
    const original = console[level].bind(console);
    console[level] = (...args) => {
      capturedConsoleMessages.push({
        level,
        text: args
          .map((arg) => {
            if (typeof arg === 'string') return arg;
            try {
              return JSON.stringify(arg);
            } catch {
              return String(arg);
            }
          })
          .join(' '),
      });
      original(...args);
    };
  }
}
const ort = await import(ORT_MODULE_URL);
const wasmNumThreadsParam = moduleParams.get('wasmNumThreads');
const parsedWasmNumThreads = Number(wasmNumThreadsParam);
const WASM_NUM_THREADS =
  wasmNumThreadsParam == null
    ? requestedProvider === 'wasm'
      ? DEFAULT_WASM_NUM_THREADS
      : null
    : !Number.isInteger(parsedWasmNumThreads) || parsedWasmNumThreads <= 0
    ? null
    : parsedWasmNumThreads;
configureOrt(ort, {
  wasmPaths: '/node_modules/onnxruntime-web/dist/',
  wasmNumThreads: WASM_NUM_THREADS,
});
const createGpuTensorFromCpu = (device, tensor) =>
  createGpuTensorFromCpuWithOrt(ort, device, tensor);
const createEmptyGpuTensor = (device, spec) => createEmptyGpuTensorWithOrt(ort, device, spec);
const DEFAULT_TIMED_RUNS = 64;
const GRAPH_CAPTURE_STEADY_STATE_DROP = 8;
const SAMPLE_STEPS = 2;
const SAMPLE_STEP_LEVEL = 1;
const CONTEXT_STEP_LEVEL = 5;
const CONTEXT_TAU_EFFECTIVE = 29 / 32;
const SAFARI_GRAPH_CAPTURE_STEP_ARTIFACT =
  'breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2_final_z_add_zero_safari_trial';
const SAFARI_GRAPH_CAPTURE_EXPECTED_FRAME_HASHES = [
  '8b011283',
  'e86c61dc',
  'ab3ef873',
  '54cc2b94',
  '960b4a73',
  'ebcf25cd',
  'c3204639',
  '9bdf2a06',
];
const SAFARI_GRAPH_CAPTURE_EXPECTED_LATENT_HASHES = [
  'f90f159f',
  'a2d64ac5',
  '577bdd02',
  'edef5069',
  '05f7a366',
  'ef6586cf',
  'cdbfd640',
  '2579363d',
];
const DEFAULT_CONFIG = {
  mode: 'streaming',
  provider: 'webgpu',
  warmupRuns: 1,
  timedRuns: DEFAULT_TIMED_RUNS,
  requireHardwareGpu: true,
  debugStats: false,
  graphCapture: false,
  dynamicsGraphCapture: null,
  decoderGraphCapture: null,
  preferredLayout: null,
  graphOptimizationLevel: 'basic',
  prefillArtifact: null,
  stepArtifact: 'breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2',
  decoderArtifact: null,
  assetBase: ASSET_DIR,
  browserProfile: 'auto',
  profiling: false,
  profilingMode: 'default',
  profilingRequired: false,
  profilingDrainMs: 100,
  profilingTopK: 20,
  ortDebug: false,
  ortLogLevel: null,
  captureConsole: CAPTURE_CONSOLE,
  graphCaptureFreshInput: false,
  graphCaptureFinalZOnly: false,
  primeGraphCapture: false,
  preallocateStepOutputs: true,
  preallocateDecoderOutputs: true,
  workerCacheUpdate: false,
  decoderWorkerPipeline: false,
  decoderWorkerNumThreads: 3,
  ortModule: ORT_MODULE_URL,
  wasmNumThreads: WASM_NUM_THREADS,
  validateOutput: true,
  validationFrames: 6,
};
const REQUIRED_ARTIFACTS = {
  prefill: ['breakout_dynamics_prefill_cached_b1_t64'],
  step: [
    'breakout_dynamics_sample_append_context_full_cache_entry_packed_b1_t1_s2',
    'breakout_dynamics_sample_append_context_full_cache_entry_b1_t1_s2',
    'breakout_dynamics_sample_append_context_cache_length_entry_b1_t1_s2',
  ],
  decoder: ['breakout_tokenizer_decoder_b1_t1', 'breakout_tokenizer_decode_z_b1_t1'],
};

function setStatus(message) {
  document.getElementById('status').textContent = message;
  console.log(`WEBGPU_BENCHMARK_STATUS ${message}`);
}

function parseConfig() {
  const params = new URLSearchParams(window.location.search);
  const booleanParam = (name, fallback) => {
    const value = params.get(name);
    if (value == null) return fallback;
    return ['1', 'true', 'yes', 'on'].includes(value.toLowerCase());
  };
  const detectedBrowserProfile = detectBrowserProfile(navigator.userAgent);
  const requestedBrowserProfile = params.get('browserProfile') ?? DEFAULT_CONFIG.browserProfile;
  const browserProfile =
    requestedBrowserProfile === 'auto' ? detectedBrowserProfile : requestedBrowserProfile;
  const browserDefaults = browserProfileDefaults(browserProfile);
  const provider = params.get('provider') ?? DEFAULT_CONFIG.provider;
  const graphCapture = booleanParam(
    'graphCapture',
    browserDefaults.graphCapture ?? DEFAULT_CONFIG.graphCapture,
  );
  return {
    mode: params.get('mode') ?? DEFAULT_CONFIG.mode,
    provider,
    warmupRuns: Number(
      params.get('warmupRuns') ?? browserDefaults.warmupRuns ?? DEFAULT_CONFIG.warmupRuns,
    ),
    timedRuns: Number(params.get('timedRuns') ?? DEFAULT_CONFIG.timedRuns),
    requireHardwareGpu:
      (params.get('requireHardwareGpu') ?? String(DEFAULT_CONFIG.requireHardwareGpu)) === 'true',
    debugStats: (params.get('debugStats') ?? String(DEFAULT_CONFIG.debugStats)) === 'true',
    graphCapture,
    dynamicsGraphCapture: booleanParam(
      'dynamicsGraphCapture',
      browserDefaults.dynamicsGraphCapture ?? DEFAULT_CONFIG.dynamicsGraphCapture ?? graphCapture,
    ),
    decoderGraphCapture: booleanParam(
      'decoderGraphCapture',
      browserDefaults.decoderGraphCapture ?? DEFAULT_CONFIG.decoderGraphCapture ?? graphCapture,
    ),
    preferredLayout: params.get('preferredLayout') ?? DEFAULT_CONFIG.preferredLayout,
    graphOptimizationLevel:
      params.get('graphOptimizationLevel') ??
      browserDefaults.graphOptimizationLevel ??
      (provider === 'wasm' ? 'all' : DEFAULT_CONFIG.graphOptimizationLevel),
    prefillArtifact: params.get('prefillArtifact') ?? DEFAULT_CONFIG.prefillArtifact,
    stepArtifact:
      params.get('stepArtifact') ??
      browserDefaults.stepArtifact ??
      (provider === 'wasm' ? null : DEFAULT_CONFIG.stepArtifact),
    decoderArtifact: params.get('decoderArtifact') ?? DEFAULT_CONFIG.decoderArtifact,
    assetBase: params.get('assetBase') ?? DEFAULT_CONFIG.assetBase,
    browserProfile,
    detectedBrowserProfile,
    profiling: booleanParam('profiling', DEFAULT_CONFIG.profiling),
    profilingMode: params.get('profilingMode') ?? DEFAULT_CONFIG.profilingMode,
    profilingRequired: booleanParam('profilingRequired', DEFAULT_CONFIG.profilingRequired),
    profilingDrainMs: Number(params.get('profilingDrainMs') ?? DEFAULT_CONFIG.profilingDrainMs),
    profilingTopK: Number(params.get('profilingTopK') ?? DEFAULT_CONFIG.profilingTopK),
    ortDebug: booleanParam('ortDebug', DEFAULT_CONFIG.ortDebug),
    ortLogLevel: params.get('ortLogLevel') ?? DEFAULT_CONFIG.ortLogLevel,
    captureConsole: CAPTURE_CONSOLE,
    graphCaptureFreshInput: booleanParam(
      'graphCaptureFreshInput',
      DEFAULT_CONFIG.graphCaptureFreshInput,
    ),
    graphCaptureFinalZOnly: booleanParam(
      'graphCaptureFinalZOnly',
      DEFAULT_CONFIG.graphCaptureFinalZOnly,
    ),
    primeGraphCapture: booleanParam(
      'primeGraphCapture',
      browserDefaults.primeGraphCapture ?? DEFAULT_CONFIG.primeGraphCapture,
    ),
    preallocateStepOutputs: booleanParam(
      'preallocateStepOutputs',
      DEFAULT_CONFIG.preallocateStepOutputs,
    ),
    preallocateDecoderOutputs: booleanParam(
      'preallocateDecoderOutputs',
      DEFAULT_CONFIG.preallocateDecoderOutputs,
    ),
    workerCacheUpdate: booleanParam(
      'workerCacheUpdate',
      provider === 'wasm' ? true : DEFAULT_CONFIG.workerCacheUpdate,
    ),
    decoderWorkerPipeline: booleanParam(
      'decoderWorkerPipeline',
      provider === 'wasm' ? true : DEFAULT_CONFIG.decoderWorkerPipeline,
    ),
    decoderWorkerNumThreads: Number(
      params.get('decoderWorkerNumThreads') ?? DEFAULT_CONFIG.decoderWorkerNumThreads,
    ),
    ortModule: ORT_MODULE_URL,
    wasmNumThreads: WASM_NUM_THREADS,
    validateOutput: booleanParam('validateOutput', DEFAULT_CONFIG.validateOutput),
    validationFrames: Number(params.get('validationFrames') ?? DEFAULT_CONFIG.validationFrames),
  };
}

function detectBrowserProfile(userAgent) {
  if (/Version\/[\d.]+ Safari\//.test(userAgent) && !/(Chrome|Chromium|CriOS|Edg)\//.test(userAgent)) {
    return 'safari';
  }
  if (/(Chrome|Chromium|CriOS|Edg)\//.test(userAgent)) return 'chromium';
  if (/Firefox\//.test(userAgent)) return 'firefox';
  return 'unknown';
}

function browserProfileDefaults(browserProfile) {
  if (browserProfile === 'safari') {
    return {
      graphCapture: true,
      dynamicsGraphCapture: true,
      decoderGraphCapture: true,
      graphOptimizationLevel: 'basic',
      stepArtifact: SAFARI_GRAPH_CAPTURE_STEP_ARTIFACT,
      warmupRuns: 0,
      primeGraphCapture: false,
    };
  }
  return {};
}

function configureAssetBase(assetBase) {
  ASSET_DIR = assetBase.replace(/\/$/, '');
  MANIFEST_URL = `${ASSET_DIR}/breakout_onnx_manifest.json`;
}

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function setupWebgpuProfiling(config) {
  if (config.provider !== 'webgpu' || !config.profiling) {
    return {
      enabled: false,
      events: [],
      reason: config.provider !== 'webgpu' ? 'provider is not webgpu' : 'profiling query param is false',
    };
  }
  const events = [];
  ort.env.webgpu ??= {};
  ort.env.webgpu.profiling ??= {};
  ort.env.webgpu.profiling.mode = config.profilingMode;
  ort.env.webgpu.profiling.ondata = (data) => {
    events.push(data);
  };
  return {
    enabled: true,
    events,
    source: 'ort.env.webgpu.profiling.ondata',
  };
}

function setupOrtDiagnostics(config) {
  if (config.ortDebug) {
    ort.env.debug = true;
  }
  if (config.ortLogLevel) {
    ort.env.logLevel = config.ortLogLevel;
  }
}

function summarizeProfiling(profiler, config) {
  if (!profiler.enabled) {
    return {
      enabled: false,
      event_count: 0,
      reason: profiler.reason,
      source: profiler.source ?? null,
    };
  }

  const groups = new Map();
  for (const event of profiler.events) {
    const durationMs = Math.max(0, (event.endTime - event.startTime) / 1_000_000);
    const key = [
      event.programName ?? '<unknown program>',
      event.kernelName ?? '<unknown kernel>',
      event.kernelType ?? '<unknown type>',
    ].join(' | ');
    const current = groups.get(key) ?? {
      program_name: event.programName ?? null,
      kernel_name: event.kernelName ?? null,
      kernel_type: event.kernelType ?? null,
      count: 0,
      total_ms: 0,
      min_ms: Number.POSITIVE_INFINITY,
      max_ms: 0,
      inputs: event.inputsMetadata ?? [],
      outputs: event.outputsMetadata ?? [],
    };
    current.count += 1;
    current.total_ms += durationMs;
    current.min_ms = Math.min(current.min_ms, durationMs);
    current.max_ms = Math.max(current.max_ms, durationMs);
    groups.set(key, current);
  }

  const topK = Math.max(1, Math.floor(config.profilingTopK || DEFAULT_CONFIG.profilingTopK));
  const topPrograms = [...groups.values()]
    .map((entry) => ({
      ...entry,
      mean_ms: entry.count > 0 ? entry.total_ms / entry.count : 0,
      min_ms: Number.isFinite(entry.min_ms) ? entry.min_ms : 0,
    }))
    .sort((a, b) => b.total_ms - a.total_ms)
    .slice(0, topK);

  return {
    enabled: true,
    event_count: profiler.events.length,
    source: profiler.source,
    top_programs: topPrograms,
  };
}

function makePrng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

const float32Scratch = new Float32Array(1);
const uint32Scratch = new Uint32Array(float32Scratch.buffer);

function float32ToFloat16Bits(value) {
  float32Scratch[0] = value;
  const bits = uint32Scratch[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  if (exponent === 0xff) {
    return sign | (mantissa ? 0x7e00 : 0x7c00);
  }
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const shifted = (mantissa | 0x800000) >>> (1 - halfExponent);
    return sign | ((shifted + 0x1000) >>> 13);
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
    while ((mantissa & 0x0400) === 0) {
      mantissa <<= 1;
      exponent -= 1;
    }
    exponent += 1;
    mantissa &= ~0x0400;
  } else if (exponent === 0x1f) {
    uint32Scratch[0] = sign | 0x7f800000 | (mantissa << 13);
    return float32Scratch[0];
  }
  uint32Scratch[0] = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  return float32Scratch[0];
}

function makeFloatData(dtype, length, valueFn) {
  if (dtype === 'float16') {
    const values = new Uint16Array(length);
    for (let i = 0; i < values.length; i += 1) {
      values[i] = float32ToFloat16Bits(valueFn(i));
    }
    return values;
  }
  const values = new Float32Array(length);
  for (let i = 0; i < values.length; i += 1) {
    values[i] = valueFn(i);
  }
  return values;
}

function tensorValue(tensor, index) {
  if (tensor.type === 'float16') {
    return float16BitsToFloat32(tensor.data[index]);
  }
  return tensor.data[index];
}

function makeFloatTensor(shape, seed, dtype = 'float32') {
  const random = makePrng(seed);
  const values = makeFloatData(dtype, mul(shape), () => random() * 2 - 1);
  return new ort.Tensor(dtype, values, shape);
}

function makeIntTensor(shape, seed, maxExclusive) {
  const random = makePrng(seed);
  const values = new Int32Array(mul(shape));
  for (let i = 0; i < values.length; i += 1) {
    values[i] = Math.floor(random() * maxExclusive);
  }
  return new ort.Tensor('int32', values, shape);
}

function makeScalarFillTensor(dtype, shape, value) {
  const length = mul(shape);
  const values =
    dtype === 'float32'
      ? new Float32Array(length).fill(value)
      : dtype === 'float16'
        ? new Uint16Array(length).fill(float32ToFloat16Bits(value))
        : new Int32Array(length).fill(value);
  return new ort.Tensor(dtype, values, shape);
}

function makeCacheAttentionMaskTensor(dtype, shape, cacheLength) {
  const contextLength = shape[shape.length - 1] - 1;
  const validLength = Math.min(Math.max(cacheLength, 0), contextLength);
  const values = makeFloatData(dtype, mul(shape), (index) => {
    const position = index % (contextLength + 1);
    return position < validLength || position === contextLength ? 1 : 0;
  });
  return new ort.Tensor(dtype, values, shape);
}

function copyGpuTensorToTargets(device, source, targets) {
  const copies = targets.filter((target) => target && source !== target);
  if (!copies.length) return;
  const byteLength = tensorByteLength(source.type, source.dims);
  const encoder = device.createCommandEncoder();
  for (const target of copies) {
    encoder.copyBufferToBuffer(source.gpuBuffer, 0, target.gpuBuffer, 0, byteLength);
  }
  device.queue.submit([encoder.finish()]);
}

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  const index = (sorted.length - 1) * p;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

function summarize(samples) {
  if (samples.length === 0) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  const mean = samples.reduce((total, value) => total + value, 0) / samples.length;
  const variance =
    samples.reduce((total, value) => total + (value - mean) ** 2, 0) / samples.length;
  return {
    mean_ms: mean,
    median_ms: percentile(sorted, 0.5),
    p90_ms: percentile(sorted, 0.9),
    p95_ms: percentile(sorted, 0.95),
    min_ms: sorted[0],
    max_ms: sorted[sorted.length - 1],
    stddev_ms: Math.sqrt(variance),
    throughput_hz: 1000 / mean,
    samples_ms: samples,
  };
}

function summarizeAfter(samples, dropCount) {
  if (dropCount <= 0) return summarize(samples);
  if (samples.length <= dropCount) return null;
  const summary = summarize(samples.slice(dropCount));
  return summary == null
    ? null
    : {
        ...summary,
        dropped_warmup_samples: dropCount,
      };
}

function summarizeGraphCaptureSteady(samples, config) {
  return config.dynamicsGraphCapture || config.decoderGraphCapture
    ? summarizeAfter(samples, GRAPH_CAPTURE_STEADY_STATE_DROP)
    : null;
}

async function timeAsync(fn) {
  const start = performance.now();
  const value = await fn();
  return { value, elapsedMs: performance.now() - start };
}

function timeSync(fn) {
  const start = performance.now();
  const value = fn();
  return { value, elapsedMs: performance.now() - start };
}

function tensorStats(tensor) {
  const values = tensor.data;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const value = tensor.type === 'float16' ? float16BitsToFloat32(values[i]) : values[i];
    if (!Number.isFinite(value)) {
      throw new Error(`Output ${tensor.dims.join('x')} contains non-finite value ${value}`);
    }
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
  }
  return {
    dtype: tensor.type,
    dims: tensor.dims,
    min,
    max,
    mean: sum / values.length,
  };
}

function tensorSummary(tensor) {
  if (tensor.location && tensor.location !== 'cpu' && tensor.location !== 'cpu-pinned') {
    return {
      dtype: tensor.type,
      dims: tensor.dims,
      location: tensor.location,
      size: tensor.size,
    };
  }
  return {
    ...tensorStats(tensor),
    location: tensor.location ?? 'cpu',
    size: tensor.size,
  };
}

async function readGpuTensorBytes(device, tensor) {
  if (!device || !tensor?.gpuBuffer) {
    throw new Error(`Cannot validate GPU tensor ${tensor?.dims?.join('x') ?? '<unknown>'}: GPUBuffer is unavailable.`);
  }
  const byteLength = tensorByteLength(tensor.type, tensor.dims);
  const paddedByteLength = Math.max(16, 4 * Math.ceil(byteLength / 4));
  const readback = device.createBuffer({
    size: paddedByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(tensor.gpuBuffer, 0, readback, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ, 0, paddedByteLength);
  const bytes = new Uint8Array(readback.getMappedRange(0, paddedByteLength)).slice(0, byteLength);
  readback.destroy();
  return bytes;
}

async function tensorDataForHash(tensor, device = null) {
  if (tensor.location && tensor.location !== 'cpu' && tensor.location !== 'cpu-pinned') {
    if (device && tensor.gpuBuffer) {
      return readGpuTensorBytes(device, tensor);
    }
    if (typeof tensor.getData !== 'function') {
      return readGpuTensorBytes(device, tensor);
    }
    return tensor.getData(false);
  }
  return tensor.data;
}

async function tensorContentHash(tensor, device = null) {
  const values = await tensorDataForHash(tensor, device);
  const bytes =
    values instanceof Uint8Array
      ? values
      : new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let hash = 2166136261 >>> 0;
  for (let index = 0; index < bytes.length; index += 1) {
    hash ^= bytes[index];
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

function outputValidationSummary(hashes, config, latentHashes = []) {
  if (!config.validateOutput) {
    return {
      status: 'skipped',
      reason: 'disabled by validateOutput=false',
      sample_count: 0,
      unique_hashes: 0,
      hashes: [],
      unique_latent_hashes: 0,
      latent_hashes: [],
    };
  }
  const uniqueHashes = new Set(hashes);
  const uniqueLatentHashes = new Set(latentHashes);
  const expected = expectedOutputValidationHashes(config);
  const matchesExpected =
    expected == null
      ? null
      : hashes.every((hash, index) => hash === expected.frame_hashes[index]) &&
        latentHashes.every((hash, index) => hash === expected.latent_hashes[index]);
  return {
    status: expected
      ? uniqueHashes.size > 1 && matchesExpected
        ? 'passed'
        : 'failed'
      : uniqueHashes.size > 1
        ? 'passed'
        : 'failed',
    sample_count: hashes.length,
    unique_hashes: uniqueHashes.size,
    hashes,
    unique_latent_hashes: uniqueLatentHashes.size,
    latent_hashes: latentHashes,
    expected_hashes: expected?.frame_hashes ?? null,
    expected_latent_hashes: expected?.latent_hashes ?? null,
    matches_expected_hashes: matchesExpected,
  };
}

function expectedOutputValidationHashes(config) {
  if (
    config.provider === 'webgpu' &&
    config.graphCapture &&
    config.dynamicsGraphCapture &&
    config.stepArtifact === SAFARI_GRAPH_CAPTURE_STEP_ARTIFACT
  ) {
    return {
      frame_hashes: SAFARI_GRAPH_CAPTURE_EXPECTED_FRAME_HASHES.slice(0, config.validationFrames),
      latent_hashes: SAFARI_GRAPH_CAPTURE_EXPECTED_LATENT_HASHES.slice(0, config.validationFrames),
    };
  }
  return null;
}

function assertDims(name, actual, expected) {
  if (actual.length !== expected.length || actual.some((value, index) => value !== expected[index])) {
    throw new Error(`${name} shape mismatch: expected ${expected}, got ${actual}`);
  }
}

async function fetchManifest() {
  setStatus('fetching manifest');
  const response = await fetch(MANIFEST_URL);
  if (!response.ok) {
    throw new Error(`Failed to fetch manifest ${MANIFEST_URL}: ${response.status}`);
  }
  return response.json();
}

async function fetchSize(url) {
  const start = performance.now();
  const response = await fetch(url, { method: 'HEAD' });
  if (!response.ok) {
    throw new Error(`Failed to stat ${url}: ${response.status}`);
  }
  return {
    bytes: Number(response.headers.get('content-length') ?? 0),
    elapsed_ms: performance.now() - start,
  };
}

async function createSession(modelUrl, externalData = [], sessionOptions = {}) {
  const { provider = DEFAULT_CONFIG.provider, ...ortSessionOptions } = sessionOptions;
  const executionProvider =
    provider === 'webgpu'
      ? {
          name: provider,
          validationMode: 'disabled',
          ...(ortSessionOptions.preferredLayout
            ? { preferredLayout: ortSessionOptions.preferredLayout }
            : {}),
        }
      : { name: provider };
  delete ortSessionOptions.preferredLayout;
  setStatus(`creating session ${modelUrl}`);
  return timeAsync(() =>
    ort.InferenceSession.create(modelUrl, {
      executionProviders: [executionProvider],
      externalData,
      graphOptimizationLevel:
        ortSessionOptions.graphOptimizationLevel ?? DEFAULT_CONFIG.graphOptimizationLevel,
      ...ortSessionOptions,
    }),
  );
}

async function gpuInfo(config) {
  if (config.provider !== 'webgpu') {
    return {
      provider: config.provider,
      skipped: true,
      reason: 'WebGPU adapter check is only used for provider=webgpu.',
    };
  }
  setStatus('checking WebGPU adapter');
  if (!navigator.gpu) {
    throw new Error('navigator.gpu is unavailable');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('navigator.gpu.requestAdapter() returned null');
  }
  const info = adapter.info ?? {};
  return {
    vendor: info.vendor ?? null,
    architecture: info.architecture ?? null,
    device: info.device ?? null,
    description: info.description ?? null,
    features: [...adapter.features],
    limits: Object.fromEntries(Object.entries(adapter.limits ?? {})),
  };
}

function isSoftwareGpu(gpu) {
  const values = [gpu.vendor, gpu.architecture, gpu.device, gpu.description]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
  return values.includes('swiftshader') || values.includes('software') || gpu.vendor === 'google';
}

function externalDataForSpec(spec) {
  return (spec.external_data ?? []).map((entry) => ({
    path: entry.path,
    data: `${ASSET_DIR}/${entry.path}`,
  }));
}

function resolveDemoSpecs(manifest, config = DEFAULT_CONFIG) {
  const exportsByName = byExportName(manifest);
  const prefillNames = config.prefillArtifact
    ? [config.prefillArtifact, ...REQUIRED_ARTIFACTS.prefill]
    : REQUIRED_ARTIFACTS.prefill;
  const manifestStepNames = [
    config.provider === 'wasm'
      ? manifest.demo_generation?.preferred_full_cache_step_export_wasm
      : null,
    manifest.demo_generation?.preferred_step_export,
  ].filter(Boolean);
  const manifestDecoderNames = [manifest.demo_generation?.preferred_decoder_export].filter(Boolean);
  const stepNames = config.stepArtifact
    ? [config.stepArtifact, ...REQUIRED_ARTIFACTS.step]
    : [...manifestStepNames, ...REQUIRED_ARTIFACTS.step];
  const decoderNames = config.decoderArtifact
    ? [config.decoderArtifact, ...REQUIRED_ARTIFACTS.decoder]
    : [...manifestDecoderNames, ...REQUIRED_ARTIFACTS.decoder];
  return {
    prefill: findSpec(exportsByName, prefillNames),
    step: findSpec(exportsByName, stepNames),
    decoder: findSpec(exportsByName, decoderNames),
  };
}

function missingDemoArtifacts(specs) {
  return Object.entries(specs)
    .filter(([, spec]) => spec == null)
    .map(([role]) => ({
      role,
      accepted_names: REQUIRED_ARTIFACTS[role],
    }));
}

function requireTensorSpec(label, spec, expectedShape) {
  if (!spec) {
    throw new Error(`Missing tensor spec for ${label}`);
  }
  if (spec.dtype !== 'float32' && spec.dtype !== 'float16' && spec.dtype !== 'int32') {
    throw new Error(`${label} has unsupported dtype ${spec.dtype}`);
  }
  if (!sameShape(spec.shape, expectedShape)) {
    throw new Error(`${label} shape mismatch: expected ${expectedShape}, got ${spec.shape}`);
  }
}

function validateDemoSpecs(specs, manifest) {
  const tensors = manifest.cache_contract?.tensors ?? {};
  const cacheShape = tensors.k_cache?.shape;
  const cacheLayout = tensors.k_cache?.layout ?? 'layer_batch_token_time_head_dim';
  const cacheLengthShape = tensors.cache_length?.shape;
  const layerCacheShape = manifest.cache_contract?.tensors?.layer_cache?.shape;
  const layerCacheCount = manifest.cache_contract?.tensors?.layer_cache?.layers ?? 0;
  const hasLayerPrefill = layerCacheCount > 0 && Boolean(specs.prefill.outputs?.k_cache_0);
  const hasLayerStep = layerCacheCount > 0 && Boolean(specs.step.inputs?.k_cache_0);
  const hasEntryStep = Boolean(specs.step.outputs?.candidate_k_entry);
  const entryShape =
    cacheShape && cacheLayout === 'layer_batch_token_head_time_dim'
      ? [cacheShape[0], cacheShape[1], cacheShape[2], 1, cacheShape[3], cacheShape[5]]
      : cacheShape
        ? [...cacheShape.slice(0, 3), 1, ...cacheShape.slice(4)]
        : null;
  if (!hasLayerPrefill) {
    requireTensorSpec('prefill.outputs.k_cache', specs.prefill.outputs?.k_cache, cacheShape);
    requireTensorSpec('prefill.outputs.v_cache', specs.prefill.outputs?.v_cache, cacheShape);
  }
  requireTensorSpec('prefill.outputs.cache_length', specs.prefill.outputs?.cache_length, cacheLengthShape);
  if (hasLayerStep) {
    for (let i = 0; i < layerCacheCount; i += 1) {
      requireTensorSpec(`step.inputs.k_cache_${i}`, specs.step.inputs?.[`k_cache_${i}`], layerCacheShape);
      requireTensorSpec(`step.inputs.v_cache_${i}`, specs.step.inputs?.[`v_cache_${i}`], layerCacheShape);
      requireTensorSpec(`step.outputs.candidate_k_cache_${i}`, specs.step.outputs?.[`candidate_k_cache_${i}`], layerCacheShape);
      requireTensorSpec(`step.outputs.candidate_v_cache_${i}`, specs.step.outputs?.[`candidate_v_cache_${i}`], layerCacheShape);
    }
  } else {
    requireTensorSpec('step.inputs.k_cache', specs.step.inputs?.k_cache, cacheShape);
    requireTensorSpec('step.inputs.v_cache', specs.step.inputs?.v_cache, cacheShape);
    if (hasEntryStep) {
      requireTensorSpec('step.outputs.candidate_k_entry', specs.step.outputs?.candidate_k_entry, entryShape);
      requireTensorSpec('step.outputs.candidate_v_entry', specs.step.outputs?.candidate_v_entry, entryShape);
    } else {
      requireTensorSpec('step.outputs.candidate_k_cache', specs.step.outputs?.candidate_k_cache, cacheShape);
      requireTensorSpec('step.outputs.candidate_v_cache', specs.step.outputs?.candidate_v_cache, cacheShape);
    }
  }
  if (specs.step.inputs?.cache_length) {
    requireTensorSpec('step.inputs.cache_length', specs.step.inputs.cache_length, cacheLengthShape);
  }
  if (specs.step.outputs?.candidate_cache_length) {
    requireTensorSpec(
      'step.outputs.candidate_cache_length',
      specs.step.outputs.candidate_cache_length,
      cacheLengthShape,
    );
  }
  const stepPredOrFinal = specs.step.outputs?.pred_z ?? specs.step.outputs?.final_z;
  requireTensorSpec('step.outputs.pred_z_or_final_z', stepPredOrFinal, [1, 1, 32, 32]);
  if (specs.step.outputs?.final_z) {
    requireTensorSpec('step.outputs.final_z', specs.step.outputs.final_z, [1, 1, 32, 32]);
  }
  if (specs.decoder.inputs?.z) {
    requireTensorSpec('decoder.inputs.z', specs.decoder.inputs.z, [1, 1, 32, 32]);
  } else {
    requireTensorSpec('decoder.inputs.latent', specs.decoder.inputs?.latent, [1, 1, 64, 16]);
  }
  if (specs.prefill.name.includes('b1_t64') && !specs.prefill.name.includes('cached')) {
    throw new Error(`Prefill artifact is not cached: ${specs.prefill.name}`);
  }
  const stepHasCacheIo =
    (Boolean(specs.step.inputs?.k_cache) &&
      Boolean(specs.step.inputs?.v_cache) &&
      Boolean(specs.step.outputs?.candidate_k_cache) &&
      Boolean(specs.step.outputs?.candidate_v_cache)) ||
    (Boolean(specs.step.inputs?.k_cache) &&
      Boolean(specs.step.inputs?.v_cache) &&
      Boolean(specs.step.outputs?.candidate_k_entry) &&
      Boolean(specs.step.outputs?.candidate_v_entry)) ||
    (Boolean(specs.step.inputs?.k_cache_0) &&
      Boolean(specs.step.inputs?.v_cache_0) &&
      Boolean(specs.step.outputs?.candidate_k_cache_0) &&
      Boolean(specs.step.outputs?.candidate_v_cache_0));
  if (specs.step.name.includes('b1_t64') || !stepHasCacheIo) {
    throw new Error(`Step artifact is not cached: ${specs.step.name}`);
  }
  if (specs.decoder.name.endsWith('b1_t64')) {
    throw new Error(`Decoder artifact is not single-frame: ${specs.decoder.name}`);
  }
}

function cacheAbi(manifest) {
  const tensors = manifest.cache_contract?.tensors ?? {};
  return {
    status: manifest.cache_contract?.status ?? null,
    ownership: manifest.cache_contract?.ownership ?? null,
    target_frame_policy: manifest.cache_contract?.target_frame_policy ?? null,
    invalidation: manifest.cache_contract?.invalidation ?? [],
    k_cache: tensors.k_cache ?? null,
    v_cache: tensors.v_cache ?? null,
    layer_cache: tensors.layer_cache ?? null,
    cache_length: tensors.cache_length ?? null,
  };
}

function samplingConfig(specs = null, generatedFrames = DEFAULT_TIMED_RUNS) {
  const sampleSteps = specs?.step?.sample_steps ?? SAMPLE_STEPS;
  return {
    sample_steps: sampleSteps,
    sample_step_level: Math.log2(sampleSteps),
    context_step_level: CONTEXT_STEP_LEVEL,
    context_tau_effective: CONTEXT_TAU_EFFECTIVE,
    generated_frames: generatedFrames,
  };
}

function compactManifest(manifest) {
  const demoExportNames = new Set(Object.values(REQUIRED_ARTIFACTS).flat());
  return {
    opset: manifest.opset,
    schema_version: manifest.schema_version,
    checkpoints: manifest.checkpoints,
    axes_policy: manifest.axes_policy,
    dynamics: manifest.dynamics,
    tokenizer: manifest.tokenizer,
    exports: (manifest.exports ?? [])
      .filter((entry) => demoExportNames.has(entry.name))
      .map((entry) => ({
        name: entry.name,
        path: entry.path,
        sha256: entry.sha256,
        external_data: entry.external_data,
        inputs: entry.inputs,
        outputs: entry.outputs,
        production_browser_ready: entry.production_browser_ready,
        sample_steps: entry.sample_steps,
      })),
  };
}

function blockedResult({ config, manifest, gpu, missing }) {
  return {
    schema_version: 2,
    status: 'blocked',
    streaming_contract_status: 'blocked',
    blocked_reason:
      'Cached dynamics prefill, cached dynamics step, and single-frame decoder artifacts are not present; manifest cache support is contract_only.',
    missing_artifacts: missing,
    benchmark_modes: ['cached_prefill', 'cached_step', 'streaming_frame'],
    config,
    created_at: new Date().toISOString(),
    user_agent: navigator.userAgent,
    platform: navigator.platform,
    ort_version: ort.version ?? null,
    provider_options: {
      executionProviders: [
        {
          name: config.provider,
          ...(config.provider === 'webgpu' && config.preferredLayout
            ? { preferredLayout: config.preferredLayout }
            : {}),
        },
      ],
      graphOptimizationLevel: config.graphOptimizationLevel,
    },
    gpu,
    sampling: samplingConfig(null, config.timedRuns),
    cache_abi: cacheAbi(manifest),
    manifest: compactManifest(manifest),
    results: [],
  };
}

function contextLengthFromStepSpec(spec) {
  const maskShape = spec.inputs?.attention_mask?.shape;
  if (maskShape?.length) return maskShape[maskShape.length - 1] - 1;
  return spec.inputs?.k_cache?.shape?.[3] ?? 64;
}

function makeFeedForInput(name, inputSpec, seed, graphSpec = null) {
  const shape = inputSpec.shape;
  const dtype = inputSpec.dtype;
  const contextLength = graphSpec ? contextLengthFromStepSpec(graphSpec) : 64;
  if (name === 'attention_mask') {
    return makeCacheAttentionMaskTensor(dtype, shape, contextLength);
  }
  if (name === 'sample_position_index') {
    return makeScalarFillTensor('int32', shape, contextLength);
  }
  if (name === 'context_position_index') {
    return makeScalarFillTensor('int32', shape, Math.max(contextLength - 1, 0));
  }
  if (dtype === 'float32' || dtype === 'float16') {
    if (name === 'cache_length') return makeScalarFillTensor('float32', shape, 64);
    return makeFloatTensor(shape, seed, dtype);
  }
  if (name.includes('step_level')) {
    return makeScalarFillTensor('int32', shape, SAMPLE_STEP_LEVEL);
  }
  if (name.includes('signal_level')) {
    return makeScalarFillTensor('int32', shape, 0);
  }
  if (name.includes('cache_length')) {
    return makeScalarFillTensor('int32', shape, 64);
  }
  return makeIntTensor(shape, seed, 4);
}

function makeFeedsFromSpec(spec, seedBase = 100) {
  const feeds = {};
  let index = 0;
  for (const [name, inputSpec] of Object.entries(spec.inputs ?? {})) {
    feeds[name] = makeFeedForInput(name, inputSpec, seedBase + index * 13, spec);
    index += 1;
  }
  return feeds;
}

function stepPredOutputName(spec) {
  const outputs = Object.keys(spec.outputs ?? {});
  return (
    outputs.find((name) => name === 'pred_z') ??
    outputs.find((name) => name.endsWith('pred_z')) ??
    outputs.find((name) => name === 'final_z') ??
    outputs[0]
  );
}

function stepFinalZOutputName(spec) {
  const outputs = Object.keys(spec.outputs ?? {});
  return outputs.find((name) => name === 'final_z') ?? stepPredOutputName(spec);
}

function stepUsesFusedSampleStep(spec, predName, finalName) {
  return Boolean(
    finalName &&
      (finalName !== predName ||
        spec.final_z_aliases_pred_z ||
        (spec.sample_steps > 1 && specsHasFinalWithoutPred(spec))),
  );
}

function specsHasFinalWithoutPred(spec) {
  return Boolean(spec.outputs?.final_z && !spec.outputs?.pred_z);
}

function decoderOutputName(spec) {
  const outputs = Object.keys(spec.outputs ?? {});
  return outputs.find((name) => name === 'patches') ?? outputs[0];
}

function decoderInputName(spec) {
  const inputs = Object.keys(spec.inputs ?? {});
  return (
    inputs.find((name) => name === 'z') ??
    inputs.find((name) => name.includes('latent')) ??
    inputs[0]
  );
}

function replaceNamedFeed(feeds, patterns, tensor) {
  const next = { ...feeds };
  const name = Object.keys(next).find((key) => patterns.some((pattern) => key.includes(pattern)));
  if (!name) {
    throw new Error(`Could not find feed matching any of: ${patterns.join(', ')}`);
  }
  next[name] = tensor;
  return next;
}

function setStepInputs(feeds, { z, contextNoise, action, stepLevel, signalLevel, positionIndex }) {
  let next = replaceNamedFeed(feeds, ['sample_noise', 'z'], z);
  if (Object.keys(next).some((name) => name.includes('context_noise'))) {
    next = replaceNamedFeed(next, ['context_noise'], contextNoise ?? z);
  }
  next = replaceNamedFeed(next, ['action'], action);
  if (Object.keys(next).some((name) => name.includes('step_level'))) {
    next = replaceNamedFeed(next, ['step_level'], stepLevel);
  }
  if (Object.keys(next).some((name) => name.includes('signal_level'))) {
    next = replaceNamedFeed(next, ['signal_level'], signalLevel);
  }
  if (Object.keys(next).some((name) => name === 'position_index')) {
    next.position_index = positionIndex;
  }
  return next;
}

function cacheFromOutputs(outputs, names, fallbackLength = null) {
  if (!names.k || !names.v || !names.length) {
    if (!fallbackLength) {
      throw new Error('Cached graph must output k_cache, v_cache, and cache_length');
    }
  }
  if (!names.k || !names.v) {
    throw new Error('Cached graph must output k_cache and v_cache');
  }
  return {
    k: Array.isArray(names.k) ? names.k.map((name) => outputs[name]) : outputs[names.k],
    v: Array.isArray(names.v) ? names.v.map((name) => outputs[name]) : outputs[names.v],
    length: names.length && outputs[names.length] ? outputs[names.length] : fallbackLength,
  };
}

function makeZeroTensorFromSpec(spec) {
  if (!spec) {
    throw new Error('Missing tensor spec for fixed GPU tensor');
  }
  return makeScalarFillTensor(spec.dtype, spec.shape, 0);
}

function createFixedGpuCache(device, spec) {
  const names = inputCacheNames(spec);
  if (!names.k || !names.v) {
    throw new Error('Step graph does not expose cache inputs for fixed GPU cache');
  }
  const makeCacheTensor = (name) => createGpuTensorFromCpu(device, makeZeroTensorFromSpec(spec.inputs[name]));
  return {
    k: Array.isArray(names.k) ? names.k.map(makeCacheTensor) : makeCacheTensor(names.k),
    v: Array.isArray(names.v) ? names.v.map(makeCacheTensor) : makeCacheTensor(names.v),
    pinned: null,
  };
}

function findInputName(spec, patterns) {
  return Object.keys(spec.inputs ?? {}).find((name) => patterns.some((pattern) => name.includes(pattern)));
}

function createFixedGpuScalarInputs(device, spec) {
  const fixed = {};
  const maskSpec = spec.inputs?.attention_mask ?? null;
  const cacheSpec = spec.inputs?.k_cache ?? null;
  const contextLength = contextLengthFromStepSpec(spec);
  const cacheLengthName = findInputName(spec, ['cache_length']);
  if (cacheLengthName) {
    const inputSpec = spec.inputs[cacheLengthName];
    fixed.cacheLength = createGpuTensorFromCpu(
      device,
      makeScalarFillTensor(inputSpec.dtype, inputSpec.shape, 64),
    );
  }
  const samplePositionName = findInputName(spec, ['sample_position_index']);
  if (samplePositionName) {
    const inputSpec = spec.inputs[samplePositionName];
    fixed.samplePositionIndex = createGpuTensorFromCpu(
      device,
      makeScalarFillTensor(inputSpec.dtype, inputSpec.shape, contextLength),
    );
  }
  const contextPositionName = findInputName(spec, ['context_position_index']);
  if (contextPositionName) {
    const inputSpec = spec.inputs[contextPositionName];
    fixed.contextPositionIndex = createGpuTensorFromCpu(
      device,
      makeScalarFillTensor(inputSpec.dtype, inputSpec.shape, Math.max(contextLength - 1, 0)),
    );
  }
  if (maskSpec) {
    fixed.attentionMask = createGpuTensorFromCpu(
      device,
      makeCacheAttentionMaskTensor(maskSpec.dtype, maskSpec.shape, contextLength),
    );
  }
  const positionIndexName = spec.inputs?.position_index ? 'position_index' : null;
  if (positionIndexName) {
    const inputSpec = spec.inputs[positionIndexName];
    fixed.positionIndex = createGpuTensorFromCpu(
      device,
      makeScalarFillTensor(inputSpec.dtype, inputSpec.shape, 64),
    );
  }
  return fixed;
}

function fixedCachePinnedTensors(fixedCache) {
  if (!fixedCache) return [];
  return [fixedCache.k, fixedCache.v].flat().filter(Boolean);
}

function fixedInputPinnedTensors(fixedInputs, fixedScalars) {
  return [
    fixedInputs?.z,
    fixedInputs?.contextNoise,
    fixedInputs?.action,
    fixedScalars?.cacheLength,
    fixedScalars?.samplePositionIndex,
    fixedScalars?.contextPositionIndex,
    fixedScalars?.attentionMask,
    fixedScalars?.positionIndex,
  ].filter(Boolean);
}

function createFixedGpuDecoderInput(device, spec) {
  const name = decoderInputName(spec);
  const inputSpec = spec.inputs?.[name];
  if (!inputSpec) {
    throw new Error('Decoder graph does not expose an input tensor.');
  }
  return {
    name,
    tensor: createGpuTensorFromCpu(device, makeZeroTensorFromSpec(inputSpec)),
  };
}

function copyCacheIntoFixedGpu(device, sourceCache, fixedCache) {
  if (Array.isArray(fixedCache.k)) {
    fixedCache.k.forEach((target, index) => copyTensorToGpu(device, sourceCache.k[index], target));
    fixedCache.v.forEach((target, index) => copyTensorToGpu(device, sourceCache.v[index], target));
  } else {
    copyTensorToGpu(device, sourceCache.k, fixedCache.k);
    copyTensorToGpu(device, sourceCache.v, fixedCache.v);
  }
  return {
    k: fixedCache.k,
    v: fixedCache.v,
    length: sourceCache.length,
  };
}

function createEntryCacheUpdater(device, spec, manifest) {
  const cacheSpec = spec.inputs?.k_cache;
  const entrySpec = spec.outputs?.candidate_k_entry;
  if (!cacheSpec || !entrySpec) {
    throw new Error('Entry-cache update requires k_cache input and candidate_k_entry output specs.');
  }
  if (cacheSpec.dtype !== 'float32' || entrySpec.dtype !== 'float32') {
    throw new Error(
      `Entry-cache update currently supports float32 caches only, got ${cacheSpec.dtype}/${entrySpec.dtype}.`,
    );
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
  const halfHeadDim = headDim / 2;
  if (!Number.isInteger(halfHeadDim)) {
    throw new Error(`Entry-cache update requires an even head_dim, got ${headDim}.`);
  }
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
  const shader = device.createShaderModule({
    label: 'visionary-entry-cache-slide-rebase',
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
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    ],
  });
  const pipeline = device.createComputePipeline({
    label: 'visionary-entry-cache-slide-rebase',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: shader, entryPoint: 'main' },
  });
  const dispatchCount = Math.ceil(
    Math.max(
      layers * batch * tokens * heads * halfHeadDim,
      layers * batch * tokens * heads * headDim,
    ) / workgroupSize,
  );
  let cachedBindGroup = null;
  let cachedBindGroupBuffers = null;
  const bindGroupFor = (cache, kEntry, vEntry) => {
    const buffers = [cache.k.gpuBuffer, cache.v.gpuBuffer, kEntry.gpuBuffer, vEntry.gpuBuffer];
    if (
      cachedBindGroup &&
      cachedBindGroupBuffers?.every((buffer, index) => buffer === buffers[index])
    ) {
      return cachedBindGroup;
    }
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
      ],
    });
    cachedBindGroupBuffers = buffers;
    return cachedBindGroup;
  };
  return {
    kind: 'webgpu_inplace_slide_rebase_entry',
    rope_base: ropeBase,
    cache_layout: cacheLayout,
    cache_shape: cacheSpec.shape,
    entry_shape: entrySpec.shape,
    update(cache, kEntry, vEntry) {
      const bindGroup = bindGroupFor(cache, kEntry, vEntry);
      const encoder = device.createCommandEncoder({ label: 'visionary-entry-cache-update' });
      const pass = encoder.beginComputePass({ label: 'visionary-entry-cache-slide-rebase' });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(dispatchCount);
      pass.end();
      device.queue.submit([encoder.finish()]);
      return cache;
    },
  };
}

function createCpuEntryCacheUpdater(spec, manifest) {
  const cacheSpec = spec.inputs?.k_cache;
  const entrySpec = spec.outputs?.candidate_k_entry;
  if (!cacheSpec || !entrySpec) {
    throw new Error('Entry-cache update requires k_cache input and candidate_k_entry output specs.');
  }
  if (cacheSpec.dtype !== 'float32' || entrySpec.dtype !== 'float32') {
    throw new Error(
      `CPU entry-cache update currently supports float32 caches only, got ${cacheSpec.dtype}/${entrySpec.dtype}.`,
    );
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
  const halfHeadDim = headDim / 2;
  if (!Number.isInteger(halfHeadDim)) {
    throw new Error(`Entry-cache update requires an even head_dim, got ${headDim}.`);
  }
  const ropeBase = Number(manifest.dynamics?.rope_base ?? manifest.dynamics?.base ?? 10000);
  const cosValues = new Float32Array(halfHeadDim);
  const sinValues = new Float32Array(halfHeadDim);
  for (let dim = 0; dim < halfHeadDim; dim += 1) {
    const theta = 1 / (ropeBase ** (dim / halfHeadDim));
    cosValues[dim] = Math.cos(theta);
    sinValues[dim] = Math.sin(theta);
  }
  const cacheIndex = (layer, batchIndex, token, time, head, dim) =>
    cacheLayout === 'layer_batch_token_head_time_dim'
      ? (((((layer * batch + batchIndex) * tokens + token) * heads + head) * contextLength + time) * headDim + dim)
      : (((((layer * batch + batchIndex) * tokens + token) * contextLength + time) * heads + head) * headDim + dim);
  const entryIndex = (layer, batchIndex, token, head, dim) =>
    ((((layer * batch + batchIndex) * tokens + token) * heads + head) * headDim + dim);
  return {
    kind: 'cpu_inplace_slide_rebase_entry',
    rope_base: ropeBase,
    cache_layout: cacheLayout,
    cache_shape: cacheSpec.shape,
    entry_shape: entrySpec.shape,
    update(cache, kEntry, vEntry) {
      const kCache = cache.k.data;
      const vCache = cache.v.data;
      const kEntryData = kEntry.data;
      const vEntryData = vEntry.data;
      if (cacheLayout === 'layer_batch_token_time_head_dim') {
        const timeStride = heads * headDim;
        const tokenStride = contextLength * timeStride;
        const entryTokenStride = heads * headDim;
        for (let layer = 0; layer < layers; layer += 1) {
          for (let batchIndex = 0; batchIndex < batch; batchIndex += 1) {
            for (let token = 0; token < tokens; token += 1) {
              const cacheTokenBase =
                ((layer * batch + batchIndex) * tokens + token) * tokenStride;
              const entryTokenBase =
                ((layer * batch + batchIndex) * tokens + token) * entryTokenStride;
              for (let head = 0; head < heads; head += 1) {
                const cacheHeadBase = cacheTokenBase + head * headDim;
                const entryHeadBase = entryTokenBase + head * headDim;
                for (let dim = 0; dim < halfHeadDim; dim += 1) {
                  const cosTheta = cosValues[dim];
                  const sinTheta = sinValues[dim];
                  let dstLeft = cacheHeadBase + dim;
                  let dstRight = cacheHeadBase + halfHeadDim + dim;
                  let srcLeft = dstLeft + timeStride;
                  let srcRight = dstRight + timeStride;
                  for (let time = 0; time < contextLength - 1; time += 1) {
                    const left = kCache[srcLeft];
                    const right = kCache[srcRight];
                    kCache[dstLeft] = left * cosTheta + right * sinTheta;
                    kCache[dstRight] = right * cosTheta - left * sinTheta;
                    dstLeft += timeStride;
                    dstRight += timeStride;
                    srcLeft += timeStride;
                    srcRight += timeStride;
                  }
                  kCache[cacheHeadBase + (contextLength - 1) * timeStride + dim] =
                    kEntryData[entryHeadBase + dim];
                  kCache[
                    cacheHeadBase + (contextLength - 1) * timeStride + halfHeadDim + dim
                  ] = kEntryData[entryHeadBase + halfHeadDim + dim];
                }
              }
              vCache.copyWithin(
                cacheTokenBase,
                cacheTokenBase + timeStride,
                cacheTokenBase + tokenStride,
              );
              vCache.set(
                vEntryData.subarray(entryTokenBase, entryTokenBase + entryTokenStride),
                cacheTokenBase + tokenStride - timeStride,
              );
            }
          }
        }
        return cache;
      }
      if (cacheLayout === 'layer_batch_token_head_time_dim') {
        const headStride = contextLength * headDim;
        const tokenStride = heads * headStride;
        const entryTokenStride = heads * headDim;
        for (let layer = 0; layer < layers; layer += 1) {
          for (let batchIndex = 0; batchIndex < batch; batchIndex += 1) {
            for (let token = 0; token < tokens; token += 1) {
              const cacheTokenBase =
                ((layer * batch + batchIndex) * tokens + token) * tokenStride;
              const entryTokenBase =
                ((layer * batch + batchIndex) * tokens + token) * entryTokenStride;
              for (let head = 0; head < heads; head += 1) {
                const cacheHeadBase = cacheTokenBase + head * headStride;
                const entryHeadBase = entryTokenBase + head * headDim;
                for (let dim = 0; dim < halfHeadDim; dim += 1) {
                  const cosTheta = cosValues[dim];
                  const sinTheta = sinValues[dim];
                  let dstLeft = cacheHeadBase + dim;
                  let dstRight = cacheHeadBase + halfHeadDim + dim;
                  let srcLeft = dstLeft + headDim;
                  let srcRight = dstRight + headDim;
                  for (let time = 0; time < contextLength - 1; time += 1) {
                    const left = kCache[srcLeft];
                    const right = kCache[srcRight];
                    kCache[dstLeft] = left * cosTheta + right * sinTheta;
                    kCache[dstRight] = right * cosTheta - left * sinTheta;
                    dstLeft += headDim;
                    dstRight += headDim;
                    srcLeft += headDim;
                    srcRight += headDim;
                  }
                  kCache[cacheHeadBase + (contextLength - 1) * headDim + dim] =
                    kEntryData[entryHeadBase + dim];
                  kCache[cacheHeadBase + (contextLength - 1) * headDim + halfHeadDim + dim] =
                    kEntryData[entryHeadBase + halfHeadDim + dim];
                }
                vCache.copyWithin(
                  cacheHeadBase,
                  cacheHeadBase + headDim,
                  cacheHeadBase + headStride,
                );
                vCache.set(
                  vEntryData.subarray(entryHeadBase, entryHeadBase + headDim),
                  cacheHeadBase + headStride - headDim,
                );
              }
            }
          }
        }
        return cache;
      }
      for (let layer = 0; layer < layers; layer += 1) {
        for (let batchIndex = 0; batchIndex < batch; batchIndex += 1) {
          for (let token = 0; token < tokens; token += 1) {
            for (let head = 0; head < heads; head += 1) {
              for (let dim = 0; dim < halfHeadDim; dim += 1) {
                const cosTheta = cosValues[dim];
                const sinTheta = sinValues[dim];
                for (let time = 0; time < contextLength - 1; time += 1) {
                  const srcLeft = cacheIndex(layer, batchIndex, token, time + 1, head, dim);
                  const srcRight = cacheIndex(
                    layer,
                    batchIndex,
                    token,
                    time + 1,
                    head,
                    halfHeadDim + dim,
                  );
                  const dstLeft = cacheIndex(layer, batchIndex, token, time, head, dim);
                  const dstRight = cacheIndex(layer, batchIndex, token, time, head, halfHeadDim + dim);
                  const left = kCache[srcLeft];
                  const right = kCache[srcRight];
                  kCache[dstLeft] = left * cosTheta + right * sinTheta;
                  kCache[dstRight] = right * cosTheta - left * sinTheta;
                }
                const entryLeft = entryIndex(layer, batchIndex, token, head, dim);
                const entryRight = entryIndex(layer, batchIndex, token, head, halfHeadDim + dim);
                kCache[cacheIndex(layer, batchIndex, token, contextLength - 1, head, dim)] =
                  kEntryData[entryLeft];
                kCache[
                  cacheIndex(layer, batchIndex, token, contextLength - 1, head, halfHeadDim + dim)
                ] = kEntryData[entryRight];
              }
              for (let dim = 0; dim < headDim; dim += 1) {
                for (let time = 0; time < contextLength - 1; time += 1) {
                  vCache[cacheIndex(layer, batchIndex, token, time, head, dim)] =
                    vCache[cacheIndex(layer, batchIndex, token, time + 1, head, dim)];
                }
                vCache[cacheIndex(layer, batchIndex, token, contextLength - 1, head, dim)] =
                  vEntryData[entryIndex(layer, batchIndex, token, head, dim)];
              }
            }
          }
        }
      }
      return cache;
    },
  };
}

function createWorkerEntryCacheUpdater(ortModule, spec, manifest) {
  const cacheSpec = spec.inputs?.k_cache;
  const entrySpec = spec.outputs?.candidate_k_entry;
  if (!cacheSpec || !entrySpec) {
    throw new Error('Entry-cache update requires k_cache input and candidate_k_entry output specs.');
  }
  if (cacheSpec.dtype !== 'float32' || entrySpec.dtype !== 'float32') {
    throw new Error(
      `Worker entry-cache update currently supports float32 caches only, got ${cacheSpec.dtype}/${entrySpec.dtype}.`,
    );
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
  const halfHeadDim = headDim / 2;
  if (!Number.isInteger(halfHeadDim)) {
    throw new Error(`Entry-cache update requires an even head_dim, got ${headDim}.`);
  }
  const ropeBase = Number(manifest.dynamics?.rope_base ?? manifest.dynamics?.base ?? 10000);
  const cosValues = Array.from({ length: halfHeadDim }, (_, dim) => {
    const theta = 1 / (ropeBase ** (dim / halfHeadDim));
    return Math.cos(theta);
  });
  const sinValues = Array.from({ length: halfHeadDim }, (_, dim) => {
    const theta = 1 / (ropeBase ** (dim / halfHeadDim));
    return Math.sin(theta);
  });
  const workerSource = `
const LAYERS = ${layers};
const BATCH = ${batch};
const TOKENS = ${tokens};
const CONTEXT = ${contextLength};
const HEADS = ${heads};
const HEAD_DIM = ${headDim};
const HALF_HEAD_DIM = ${halfHeadDim};
const CACHE_LAYOUT = ${JSON.stringify(cacheLayout)};
const cosValues = new Float32Array(${JSON.stringify(cosValues)});
const sinValues = new Float32Array(${JSON.stringify(sinValues)});

function updateTimeHead(kCache, vCache, kEntryData, vEntryData) {
  const timeStride = HEADS * HEAD_DIM;
  const tokenStride = CONTEXT * timeStride;
  const entryTokenStride = HEADS * HEAD_DIM;
  for (let layer = 0; layer < LAYERS; layer += 1) {
    for (let batchIndex = 0; batchIndex < BATCH; batchIndex += 1) {
      for (let token = 0; token < TOKENS; token += 1) {
        const cacheTokenBase = ((layer * BATCH + batchIndex) * TOKENS + token) * tokenStride;
        const entryTokenBase = ((layer * BATCH + batchIndex) * TOKENS + token) * entryTokenStride;
        for (let head = 0; head < HEADS; head += 1) {
          const cacheHeadBase = cacheTokenBase + head * HEAD_DIM;
          const entryHeadBase = entryTokenBase + head * HEAD_DIM;
          for (let dim = 0; dim < HALF_HEAD_DIM; dim += 1) {
            const cosTheta = cosValues[dim];
            const sinTheta = sinValues[dim];
            let dstLeft = cacheHeadBase + dim;
            let dstRight = cacheHeadBase + HALF_HEAD_DIM + dim;
            let srcLeft = dstLeft + timeStride;
            let srcRight = dstRight + timeStride;
            for (let time = 0; time < CONTEXT - 1; time += 1) {
              const left = kCache[srcLeft];
              const right = kCache[srcRight];
              kCache[dstLeft] = left * cosTheta + right * sinTheta;
              kCache[dstRight] = right * cosTheta - left * sinTheta;
              dstLeft += timeStride;
              dstRight += timeStride;
              srcLeft += timeStride;
              srcRight += timeStride;
            }
            kCache[cacheHeadBase + (CONTEXT - 1) * timeStride + dim] =
              kEntryData[entryHeadBase + dim];
            kCache[cacheHeadBase + (CONTEXT - 1) * timeStride + HALF_HEAD_DIM + dim] =
              kEntryData[entryHeadBase + HALF_HEAD_DIM + dim];
          }
        }
        vCache.copyWithin(cacheTokenBase, cacheTokenBase + timeStride, cacheTokenBase + tokenStride);
        vCache.set(
          vEntryData.subarray(entryTokenBase, entryTokenBase + entryTokenStride),
          cacheTokenBase + tokenStride - timeStride,
        );
      }
    }
  }
}

function updateHeadTime(kCache, vCache, kEntryData, vEntryData) {
  const headStride = CONTEXT * HEAD_DIM;
  const tokenStride = HEADS * headStride;
  const entryTokenStride = HEADS * HEAD_DIM;
  for (let layer = 0; layer < LAYERS; layer += 1) {
    for (let batchIndex = 0; batchIndex < BATCH; batchIndex += 1) {
      for (let token = 0; token < TOKENS; token += 1) {
        const cacheTokenBase = ((layer * BATCH + batchIndex) * TOKENS + token) * tokenStride;
        const entryTokenBase = ((layer * BATCH + batchIndex) * TOKENS + token) * entryTokenStride;
        for (let head = 0; head < HEADS; head += 1) {
          const cacheHeadBase = cacheTokenBase + head * headStride;
          const entryHeadBase = entryTokenBase + head * HEAD_DIM;
          for (let dim = 0; dim < HALF_HEAD_DIM; dim += 1) {
            const cosTheta = cosValues[dim];
            const sinTheta = sinValues[dim];
            let dstLeft = cacheHeadBase + dim;
            let dstRight = cacheHeadBase + HALF_HEAD_DIM + dim;
            let srcLeft = dstLeft + HEAD_DIM;
            let srcRight = dstRight + HEAD_DIM;
            for (let time = 0; time < CONTEXT - 1; time += 1) {
              const left = kCache[srcLeft];
              const right = kCache[srcRight];
              kCache[dstLeft] = left * cosTheta + right * sinTheta;
              kCache[dstRight] = right * cosTheta - left * sinTheta;
              dstLeft += HEAD_DIM;
              dstRight += HEAD_DIM;
              srcLeft += HEAD_DIM;
              srcRight += HEAD_DIM;
            }
            kCache[cacheHeadBase + (CONTEXT - 1) * HEAD_DIM + dim] =
              kEntryData[entryHeadBase + dim];
            kCache[cacheHeadBase + (CONTEXT - 1) * HEAD_DIM + HALF_HEAD_DIM + dim] =
              kEntryData[entryHeadBase + HALF_HEAD_DIM + dim];
          }
          vCache.copyWithin(cacheHeadBase, cacheHeadBase + HEAD_DIM, cacheHeadBase + headStride);
          vCache.set(
            vEntryData.subarray(entryHeadBase, entryHeadBase + HEAD_DIM),
            cacheHeadBase + headStride - HEAD_DIM,
          );
        }
      }
    }
  }
}

self.onmessage = (event) => {
  const { id, kCacheBuffer, vCacheBuffer, kEntryBuffer, vEntryBuffer } = event.data;
  const kCache = new Float32Array(kCacheBuffer);
  const vCache = new Float32Array(vCacheBuffer);
  const kEntryData = new Float32Array(kEntryBuffer);
  const vEntryData = new Float32Array(vEntryBuffer);
  if (CACHE_LAYOUT === 'layer_batch_token_head_time_dim') {
    updateHeadTime(kCache, vCache, kEntryData, vEntryData);
  } else {
    updateTimeHead(kCache, vCache, kEntryData, vEntryData);
  }
  self.postMessage({ id, kCacheBuffer, vCacheBuffer }, [kCacheBuffer, vCacheBuffer]);
};
`;
  const workerUrl = URL.createObjectURL(
    new Blob([workerSource], { type: 'text/javascript; charset=utf-8' }),
  );
  const worker = new Worker(workerUrl);
  let nextId = 0;
  const pending = new Map();
  worker.onmessage = (event) => {
    const { id, kCacheBuffer, vCacheBuffer } = event.data;
    const callbacks = pending.get(id);
    if (!callbacks) return;
    pending.delete(id);
    callbacks.resolve({ kCacheBuffer, vCacheBuffer });
  };
  worker.onerror = (event) => {
    for (const callbacks of pending.values()) {
      callbacks.reject(new Error(event.message || 'worker cache update failed'));
    }
    pending.clear();
  };
  const fullBufferView = (tensor, label) => {
    const view = tensor.data;
    if (!(view?.buffer instanceof ArrayBuffer)) {
      throw new Error(`Worker cache update requires transferable ArrayBuffer for ${label}.`);
    }
    if (view.byteOffset !== 0 || view.byteLength !== view.buffer.byteLength) {
      throw new Error(`Worker cache update requires a full-buffer view for ${label}.`);
    }
    return view.buffer;
  };
  const copiedBufferView = (tensor, label) => {
    const view = tensor.data;
    if (!(view?.buffer instanceof ArrayBuffer)) {
      throw new Error(`Worker cache update requires ArrayBuffer data for ${label}.`);
    }
    return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
  };
  return {
    kind: 'worker_inplace_slide_rebase_entry',
    async: true,
    rope_base: ropeBase,
    cache_layout: cacheLayout,
    cache_shape: cacheSpec.shape,
    entry_shape: entrySpec.shape,
    async update(cache, kEntry, vEntry) {
      const id = nextId;
      nextId += 1;
      const kCacheBuffer = fullBufferView(cache.k, 'k_cache');
      const vCacheBuffer = fullBufferView(cache.v, 'v_cache');
      const kEntryBuffer = copiedBufferView(kEntry, 'candidate_k_entry');
      const vEntryBuffer = copiedBufferView(vEntry, 'candidate_v_entry');
      const result = await new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        worker.postMessage(
          { id, kCacheBuffer, vCacheBuffer, kEntryBuffer, vEntryBuffer },
          [kCacheBuffer, vCacheBuffer, kEntryBuffer, vEntryBuffer],
        );
      });
      return {
        k: new ortModule.Tensor('float32', new Float32Array(result.kCacheBuffer), cache.k.dims),
        v: new ortModule.Tensor('float32', new Float32Array(result.vCacheBuffer), cache.v.dims),
        length: cache.length,
      };
    },
  };
}

async function createWorkerDecoderRunner(spec, config) {
  const inputName = decoderInputName(spec);
  const outputName = decoderOutputName(spec);
  const inputSpec = spec.inputs?.[inputName];
  const outputSpec = spec.outputs?.[outputName];
  if (!inputSpec || !outputSpec) {
    throw new Error('Decoder worker requires static decoder input and output specs.');
  }
  if (inputSpec.dtype !== 'float32' || outputSpec.dtype !== 'float32') {
    throw new Error(
      `Decoder worker currently supports float32 tensors only, got ${inputSpec.dtype}/${outputSpec.dtype}.`,
    );
  }
  const workerSource = `
let ort = null;
let session = null;
let inputName = null;
let outputName = null;
let inputDims = null;
let outputDims = null;

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === 'setup') {
      ort = await import(message.ortModule);
      ort.env.wasm ??= {};
      ort.env.wasm.wasmPaths = message.wasmPaths;
      if (message.wasmNumThreads != null) {
        ort.env.wasm.numThreads = message.wasmNumThreads;
      }
      inputName = message.inputName;
      outputName = message.outputName;
      inputDims = message.inputDims;
      outputDims = message.outputDims;
      session = await ort.InferenceSession.create(message.modelUrl, {
        executionProviders: ['wasm'],
        externalData: message.externalData,
        graphOptimizationLevel: message.graphOptimizationLevel,
      });
      self.postMessage({ id: message.id, ok: true });
      return;
    }
    if (!session) throw new Error('decoder worker has not been set up');
    const started = performance.now();
    const input = new ort.Tensor('float32', new Float32Array(message.inputBuffer), inputDims);
    const outputs = await session.run({ [inputName]: input }, [outputName]);
    const output = outputs[outputName];
    const elapsedMs = performance.now() - started;
    const outputBuffer = output.data.buffer.slice(
      output.data.byteOffset,
      output.data.byteOffset + output.data.byteLength,
    );
    self.postMessage(
      {
        id: message.id,
        ok: true,
        elapsedMs,
        outputBuffer,
        outputDims,
      },
      [outputBuffer],
    );
  } catch (error) {
    self.postMessage({
      id: message.id,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
`;
  const workerUrl = URL.createObjectURL(
    new Blob([workerSource], { type: 'text/javascript; charset=utf-8' }),
  );
  const worker = new Worker(workerUrl, { type: 'module' });
  let nextId = 0;
  const pending = new Map();
  worker.onmessage = (event) => {
    const { id, ok, error, ...result } = event.data;
    const callbacks = pending.get(id);
    if (!callbacks) return;
    pending.delete(id);
    if (ok) callbacks.resolve(result);
    else callbacks.reject(new Error(error || 'decoder worker failed'));
  };
  worker.onerror = (event) => {
    for (const callbacks of pending.values()) {
      callbacks.reject(new Error(event.message || 'decoder worker failed'));
    }
    pending.clear();
  };
  const request = (message, transfer = []) =>
    new Promise((resolve, reject) => {
      const id = nextId;
      nextId += 1;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, ...message }, transfer);
    });
  const decoderWorkerNumThreads =
    Number.isInteger(config.decoderWorkerNumThreads) && config.decoderWorkerNumThreads > 0
      ? config.decoderWorkerNumThreads
      : null;
  await request({
    type: 'setup',
    ortModule: new URL(ORT_MODULE_URL, window.location.href).href,
    wasmPaths: new URL('/node_modules/onnxruntime-web/dist/', window.location.href).href,
    wasmNumThreads: decoderWorkerNumThreads,
    modelUrl: new URL(`${ASSET_DIR}/${spec.path}`, window.location.href).href,
    externalData: externalDataForSpec(spec).map((entry) => ({
      ...entry,
      data: new URL(entry.data, window.location.href).href,
    })),
    graphOptimizationLevel: config.graphOptimizationLevel,
    inputName,
    outputName,
    inputDims: inputSpec.shape,
    outputDims: outputSpec.shape,
  });
  return {
    worker: true,
    num_threads: decoderWorkerNumThreads,
    release() {
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
    },
    async run(latent) {
      const view = latent.data;
      if (!(view?.buffer instanceof ArrayBuffer)) {
        throw new Error('Decoder worker requires CPU ArrayBuffer input data.');
      }
      const inputBuffer = view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
      const result = await request({ type: 'run', inputBuffer }, [inputBuffer]);
      return {
        elapsedMs: result.elapsedMs,
        value: {
          [outputName]: new ort.Tensor(
            'float32',
            new Float32Array(result.outputBuffer),
            result.outputDims,
          ),
        },
      };
    },
  };
}

function updateCacheFromEntries(updater, cache, outputs, names, pinned = [], device = null) {
  const kEntry = outputs[names.entryK];
  const vEntry = outputs[names.entryV];
  if (!kEntry || !vEntry) {
    throw new Error('Entry-cache step did not return candidate_k_entry/candidate_v_entry.');
  }
  if (updater.kind?.startsWith('webgpu') && (kEntry.location !== 'gpu-buffer' || vEntry.location !== 'gpu-buffer')) {
    throw new Error(
      `Entry-cache update requires GPU entry tensors, got ${kEntry.location}/${vEntry.location}.`,
    );
  }
  const updatedCache = updater.update(cache, kEntry, vEntry);
  disposeTensorUnlessPinnedAfterSubmittedWork(device, kEntry, pinned);
  disposeTensorUnlessPinnedAfterSubmittedWork(device, vEntry, pinned);
  return updatedCache;
}

async function updateCacheFromEntriesAsync(updater, cache, outputs, names, pinned = [], device = null) {
  const kEntry = outputs[names.entryK];
  const vEntry = outputs[names.entryV];
  if (!kEntry || !vEntry) {
    throw new Error('Entry-cache step did not return candidate_k_entry/candidate_v_entry.');
  }
  if (updater.kind?.startsWith('webgpu') && (kEntry.location !== 'gpu-buffer' || vEntry.location !== 'gpu-buffer')) {
    throw new Error(
      `Entry-cache update requires GPU entry tensors, got ${kEntry.location}/${vEntry.location}.`,
    );
  }
  const updatedCache = await updater.update(cache, kEntry, vEntry);
  disposeTensorUnlessPinnedAfterSubmittedWork(device, kEntry, pinned);
  disposeTensorUnlessPinnedAfterSubmittedWork(device, vEntry, pinned);
  return updatedCache;
}

function cacheFetches(names) {
  if (names.entryK && names.entryV) return [names.entryK, names.entryV];
  return [...new Set([names.k, names.v, names.length].flat().filter(Boolean))];
}

function disposeTensorIfOwned(tensor) {
  if (tensor?.location === 'gpu-buffer') {
    tensor.dispose();
  }
}

function tensorIsInList(tensor, list) {
  return list.some((entry) => entry === tensor);
}

function disposeTensorUnlessPinned(tensor, pinned = []) {
  if (!tensorIsInList(tensor, pinned)) {
    disposeTensorIfOwned(tensor);
  }
}

function disposeTensorAfterSubmittedWork(device, tensor) {
  if (tensor?.location !== 'gpu-buffer') {
    disposeTensorIfOwned(tensor);
    return;
  }
  if (!device?.queue?.onSubmittedWorkDone) {
    disposeTensorIfOwned(tensor);
    return;
  }
  device.queue.onSubmittedWorkDone().then(
    () => tensor.dispose(),
    () => tensor.dispose(),
  );
}

function disposeTensorUnlessPinnedAfterSubmittedWork(device, tensor, pinned = []) {
  if (!tensorIsInList(tensor, pinned)) {
    disposeTensorAfterSubmittedWork(device, tensor);
  }
}

function disposeCache(cache, pinned = []) {
  if (Array.isArray(cache?.k)) cache.k.forEach((tensor) => disposeTensorUnlessPinned(tensor, pinned));
  else disposeTensorUnlessPinned(cache?.k, pinned);
  if (Array.isArray(cache?.v)) cache.v.forEach((tensor) => disposeTensorUnlessPinned(tensor, pinned));
  else disposeTensorUnlessPinned(cache?.v, pinned);
}

function createPreallocatedFetches(device, spec, names) {
  if (!device || !names.length) return null;
  return Object.fromEntries(
    names.map((name) => {
      const outputSpec = spec.outputs?.[name];
      if (!outputSpec) {
        throw new Error(`Cannot preallocate unknown output ${name}`);
      }
      return [name, createEmptyGpuTensor(device, outputSpec)];
    }),
  );
}

function assertGraphCaptureGpuTensors(label, feeds, fetches) {
  const entries = [
    ...Object.entries(feeds ?? {}).map(([name, tensor]) => [`feed:${name}`, tensor]),
    ...(fetches && !Array.isArray(fetches)
      ? Object.entries(fetches).map(([name, tensor]) => [`fetch:${name}`, tensor])
      : []),
  ];
  for (const [name, tensor] of entries) {
    if (tensor?.location === 'gpu-buffer' && !tensor.gpuBuffer) {
      throw new Error(`${label} ${name} is marked gpu-buffer but has no GPUBuffer.`);
    }
  }
}

function preallocatedPinnedTensors(fetches) {
  return fetches ? Object.values(fetches) : [];
}

function latentFromPredZ(predZ) {
  if (mul(predZ.dims) !== 1024) {
    throw new Error(`Cannot reshape pred_z ${predZ.dims.join('x')} to decoder latent [1,1,64,16]`);
  }
  const values =
    predZ.type === 'float16' ? new Uint16Array(predZ.data) : new Float32Array(predZ.data);
  return new ort.Tensor(predZ.type, values, [1, 1, 64, 16]);
}

function nextSampleZ(currentZ, predZ, signalLevel) {
  const tau = signalLevel / SAMPLE_STEPS;
  const stepSize = 1 / SAMPLE_STEPS;
  const denom = Math.max(1 - tau, 1e-6);
  const values = makeFloatData(currentZ.type, currentZ.data.length, (i) => {
    const current = tensorValue(currentZ, i);
    const predicted = tensorValue(predZ, i);
    const velocity = (predicted - current) / denom;
    return current + velocity * stepSize;
  });
  return new ort.Tensor(currentZ.type, values, currentZ.dims);
}

function stepZInputDtype(spec) {
  const input =
    Object.entries(spec.inputs ?? {}).find(([name]) => name.includes('sample_noise') || name === 'z')?.[1] ??
    null;
  return input?.dtype ?? 'float32';
}

function decoderInputDtype(spec) {
  const input =
    Object.entries(spec.inputs ?? {}).find(([name]) => name === 'z' || name.includes('latent'))?.[1] ??
    null;
  return input?.dtype ?? 'float32';
}

function replaceDecoderLatent(feeds, latent) {
  const next = { ...feeds };
  const latentName = decoderInputName({ inputs: next });
  next[latentName] = latent;
  return next;
}

function preferredOutputLocationFor(role, spec, config) {
  if (config.provider !== 'webgpu') {
    return undefined;
  }
  if (role === 'cached_prefill') {
    const names = cacheOutputNames(spec);
    const locations = {};
    for (const name of [names.k, names.v].flat().filter(Boolean)) locations[name] = 'gpu-buffer';
    if (names.length) locations[names.length] = 'cpu';
    return locations;
  }
  if (role === 'cached_step') {
    if (config.dynamicsGraphCapture) {
      return Object.fromEntries(Object.keys(spec.outputs ?? {}).map((name) => [name, 'gpu-buffer']));
    }
    const names = cacheOutputNames(spec);
    const predName = stepPredOutputName(spec);
    const finalName = stepFinalZOutputName(spec);
    const usesFusedSampleStep = stepUsesFusedSampleStep(spec, predName, finalName);
    const locations = {
      [predName]: usesFusedSampleStep && !config.debugStats ? 'gpu-buffer' : 'cpu',
    };
    for (const name of [names.k, names.v].flat().filter(Boolean)) locations[name] = 'gpu-buffer';
    for (const name of [names.entryK, names.entryV].filter(Boolean)) locations[name] = 'gpu-buffer';
    if (names.length && !config.dynamicsGraphCapture) locations[names.length] = 'cpu';
    if (usesFusedSampleStep) {
      locations[finalName] = 'gpu-buffer';
    }
    return locations;
  }
  if (role === 'single_frame_decoder') {
    return {
      [decoderOutputName(spec)]: config.debugStats ? 'cpu' : 'gpu-buffer',
    };
  }
  if (role === 'single_frame_decoder_validation') {
    return {
      [decoderOutputName(spec)]: 'cpu',
    };
  }
  return undefined;
}

async function createBenchSession(role, spec, config) {
  const modelUrl = `${ASSET_DIR}/${spec.path}`;
  const modelFetch = await fetchSize(modelUrl);
  const preferredOutputLocation = preferredOutputLocationFor(role, spec, config);
  const enableGraphCapture =
    config.provider === 'webgpu' &&
    ((role === 'cached_step' && config.dynamicsGraphCapture) ||
      (role === 'single_frame_decoder' && config.decoderGraphCapture));
  const graphOptimizationLevel = config.graphOptimizationLevel;
  const sessionCreate = await createSession(
    modelUrl,
    externalDataForSpec(spec),
    {
      provider: config.provider,
      graphOptimizationLevel,
      ...(config.preferredLayout ? { preferredLayout: config.preferredLayout } : {}),
      ...(preferredOutputLocation ? { preferredOutputLocation } : {}),
      ...(enableGraphCapture ? { enableGraphCapture: true } : {}),
    },
  );
  return {
    role,
    spec,
    session: sessionCreate.value,
    model_url: modelUrl,
    model_fetch_ms: modelFetch.elapsed_ms,
    model_bytes: modelFetch.bytes,
    session_create_ms: sessionCreate.elapsedMs,
    preferred_output_location: preferredOutputLocation ?? null,
    graph_optimization_level: graphOptimizationLevel,
    graph_capture: enableGraphCapture,
  };
}

async function runDemoBenchmark({ config, specs, manifest }) {
  const prefill = await createBenchSession('cached_prefill', specs.prefill, config);
  const step = await createBenchSession('cached_step', specs.step, config);
  const decoder = await createBenchSession('single_frame_decoder', specs.decoder, config);
  const validationDecoder = config.validateOutput
    ? await createBenchSession('single_frame_decoder_validation', specs.decoder, config)
    : null;
  const decoderWorker =
    config.provider === 'wasm' && config.decoderWorkerPipeline
      ? await createWorkerDecoderRunner(specs.decoder, config)
      : null;
  const prefillFeeds = makeFeedsFromSpec(specs.prefill, 1000);
  const stepBaseFeeds = makeFeedsFromSpec(specs.step, 2000);
  const decoderBaseFeeds = makeFeedsFromSpec(specs.decoder, 3000);
  const prefillCacheNames = cacheOutputNames(specs.prefill);
  const stepCacheNames = cacheOutputNames(specs.step);
  const usesEntryCacheStep = Boolean(stepCacheNames.entryK && stepCacheNames.entryV);
  const predName = stepPredOutputName(specs.step);
  const finalZName = stepFinalZOutputName(specs.step);
  const usesFusedSampleStep = stepUsesFusedSampleStep(specs.step, predName, finalZName);
  const decoderName = decoderOutputName(specs.decoder);
  const prefillFetches = cacheFetches(prefillCacheNames);
  const stepCacheFetches =
    step.graph_capture && !usesEntryCacheStep
      ? [stepCacheNames.k, stepCacheNames.v].flat().filter(Boolean)
      : cacheFetches(stepCacheNames);
  const stepPredFetches = [...new Set([predName, finalZName])];
  const stepCommitFetches = config.graphCaptureFinalZOnly && step.graph_capture
    ? [finalZName]
    : usesFusedSampleStep && !config.debugStats
    ? [...new Set([finalZName, ...stepCacheFetches])]
    : [...new Set([predName, finalZName, ...stepCacheFetches])];
  const decoderFetches = [decoderName];
  const actionTensor = makeIntTensor([1, 1], 4000, 4);
  const zDtype = stepZInputDtype(specs.step);
  const gpuDevice = config.provider === 'webgpu' ? (ort.env.webgpu?.device ?? null) : null;
  const usesSafariMaterializedGraphCapture =
    config.stepArtifact === SAFARI_GRAPH_CAPTURE_STEP_ARTIFACT;
  const usePreallocatedStepOutputs =
    gpuDevice &&
    config.preallocateStepOutputs &&
    !config.debugStats &&
    !(
      config.browserProfile === 'safari' &&
      config.dynamicsGraphCapture &&
      !usesSafariMaterializedGraphCapture
    );
  const usePreallocatedDecoderOutputs =
    gpuDevice &&
    config.preallocateDecoderOutputs &&
    !config.debugStats &&
    !(
      config.browserProfile === 'safari' &&
      (config.dynamicsGraphCapture || config.decoderGraphCapture) &&
      !usesSafariMaterializedGraphCapture
    );
  const stepCommitFetchArg = usePreallocatedStepOutputs
    ? createPreallocatedFetches(gpuDevice, specs.step, stepCommitFetches)
    : stepCommitFetches;
  const decoderFetchArg = usePreallocatedDecoderOutputs
    ? createPreallocatedFetches(gpuDevice, specs.decoder, decoderFetches)
    : decoderFetches;
  const entryCacheUpdater =
    usesEntryCacheStep
      ? gpuDevice
        ? createEntryCacheUpdater(gpuDevice, specs.step, manifest)
        : config.workerCacheUpdate
          ? createWorkerEntryCacheUpdater(ort, specs.step, manifest)
        : createCpuEntryCacheUpdater(specs.step, manifest)
      : null;
  if (usesEntryCacheStep && !entryCacheUpdater) {
    throw new Error('Entry-cache artifact requires provider=webgpu and an ORT WebGPU device.');
  }
  const graphCaptureStepInputs =
    step.graph_capture && gpuDevice
      ? {
          action: createGpuTensorFromCpu(gpuDevice, actionTensor),
          z: createGpuTensorFromCpu(gpuDevice, makeFloatTensor([1, 1, 32, 32], 5999, zDtype)),
          contextNoise: createGpuTensorFromCpu(
            gpuDevice,
            makeFloatTensor([1, 1, 32, 32], 6001, zDtype),
          ),
        }
      : null;
  const streamingInputZ =
    !step.graph_capture && gpuDevice && usesFusedSampleStep
      ? createGpuTensorFromCpu(gpuDevice, makeFloatTensor([1, 1, 32, 32], 5999, zDtype))
      : null;
  const graphCaptureFixedCache =
    step.graph_capture && gpuDevice
      ? createFixedGpuCache(gpuDevice, specs.step)
      : null;
  const graphCaptureFixedScalars =
    step.graph_capture && gpuDevice
      ? createFixedGpuScalarInputs(gpuDevice, specs.step)
      : null;
  const graphCaptureDecoderInput =
    gpuDevice && (decoder.graph_capture || !specs.decoder.inputs?.z)
      ? createFixedGpuDecoderInput(gpuDevice, specs.decoder)
      : null;
  if (
    graphCaptureDecoderInput &&
    stepCommitFetchArg &&
    !Array.isArray(stepCommitFetchArg) &&
    specs.step.outputs?.[finalZName]?.dtype === graphCaptureDecoderInput.tensor.type &&
    sameShape(specs.step.outputs?.[finalZName]?.shape, graphCaptureDecoderInput.tensor.dims)
  ) {
    stepCommitFetchArg[finalZName] = graphCaptureDecoderInput.tensor;
  }
  const graphCapturePinnedTensors = [
    ...fixedCachePinnedTensors(graphCaptureFixedCache),
    ...fixedInputPinnedTensors(graphCaptureStepInputs, graphCaptureFixedScalars),
    graphCaptureDecoderInput?.tensor,
    ...(entryCacheUpdater?.pinned_tensors ?? []),
    ...preallocatedPinnedTensors(stepCommitFetchArg),
    ...preallocatedPinnedTensors(decoderFetchArg),
    streamingInputZ,
  ];
  const stepLevelTensor = makeScalarFillTensor('int32', [1, 1], SAMPLE_STEP_LEVEL);
  const stepInputs = specs.step.inputs ?? {};
  const needsSignalLevelInput = Object.keys(stepInputs).some((name) => name.includes('signal_level'));
  const needsPositionIndexInput = Object.keys(stepInputs).some((name) => name === 'position_index');

  setStatus('demo benchmark: first prefill');
  const prefillFirst = await timeAsync(() => prefill.session.run(prefillFeeds, prefillFetches));
  let persistentCache = cacheFromOutputs(prefillFirst.value, prefillCacheNames);
  if (graphCaptureFixedCache) {
    const prefillCache = persistentCache;
    persistentCache = copyCacheIntoFixedGpu(gpuDevice, prefillCache, graphCaptureFixedCache);
    disposeCache(prefillCache, graphCapturePinnedTensors);
  }
  let streamingZ = graphCaptureStepInputs?.z ?? streamingInputZ ?? null;

  for (let i = 0; i < config.warmupRuns; i += 1) {
    setStatus(`demo benchmark: warmup frame ${i + 1}/${config.warmupRuns}`);
    let candidateCache = persistentCache;
    let currentZ =
      config.graphCaptureFreshInput && step.graph_capture
        ? makeFloatTensor([1, 1, 32, 32], 5000 + i, zDtype)
        : streamingInputZ ?? makeFloatTensor([1, 1, 32, 32], 5000 + i, zDtype);
    let predZ = null;
    let pendingEntryCacheOutputs = null;
    const sampleCount = usesFusedSampleStep ? 1 : SAMPLE_STEPS;
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const signalLevelTensor = needsSignalLevelInput
        ? makeScalarFillTensor('int32', [1, 1], sample)
        : null;
      const positionTensor = needsPositionIndexInput
        ? makeScalarFillTensor('int32', [1], persistentCache.length.data[0])
        : null;
      if (graphCaptureStepInputs?.z && currentZ.location === 'gpu-buffer') {
        copyGpuTensorToTargets(gpuDevice, currentZ, [
          graphCaptureStepInputs.z,
          graphCaptureStepInputs.contextNoise,
        ]);
      } else if (graphCaptureStepInputs?.z) {
        writeCpuTensorToGpu(gpuDevice, graphCaptureStepInputs.z, currentZ);
        writeCpuTensorToGpu(gpuDevice, graphCaptureStepInputs.contextNoise, currentZ);
      }
      const feeds = setStepInputs(applyCacheFeeds(stepBaseFeeds, persistentCache), {
        z: graphCaptureStepInputs?.z ?? currentZ,
        contextNoise: graphCaptureStepInputs?.contextNoise ?? currentZ,
        action: graphCaptureStepInputs?.action ?? actionTensor,
        stepLevel: stepLevelTensor,
        signalLevel: signalLevelTensor,
        positionIndex: positionTensor,
      });
      if (graphCaptureFixedScalars?.cacheLength) {
        feeds.cache_length = graphCaptureFixedScalars.cacheLength;
      }
      if (graphCaptureFixedScalars?.samplePositionIndex) {
        feeds.sample_position_index = graphCaptureFixedScalars.samplePositionIndex;
      }
      if (graphCaptureFixedScalars?.contextPositionIndex) {
        feeds.context_position_index = graphCaptureFixedScalars.contextPositionIndex;
      }
      if (graphCaptureFixedScalars?.attentionMask) {
        feeds.attention_mask = graphCaptureFixedScalars.attentionMask;
      }
      if (graphCaptureFixedScalars?.positionIndex) {
        feeds.position_index = graphCaptureFixedScalars.positionIndex;
      }
      const fetches = sample === sampleCount - 1 ? stepCommitFetchArg : stepPredFetches;
      if (step.graph_capture) {
        assertGraphCaptureGpuTensors('warmup cached_step graph capture', feeds, fetches);
      }
      const outputs = await step.session.run(feeds, fetches);
      predZ = outputs[predName] ?? null;
      if (sample === sampleCount - 1) {
        if (usesEntryCacheStep) {
          pendingEntryCacheOutputs =
            config.graphCaptureFinalZOnly && step.graph_capture ? null : outputs;
        } else {
          candidateCache = cacheFromOutputs(outputs, stepCacheNames, persistentCache.length);
        }
        if (!usesEntryCacheStep && graphCaptureFixedCache) {
          const outputCache = candidateCache;
          candidateCache = copyCacheIntoFixedGpu(gpuDevice, outputCache, graphCaptureFixedCache);
          disposeCache(outputCache, graphCapturePinnedTensors);
        }
      }
      currentZ = usesFusedSampleStep ? outputs[finalZName] : nextSampleZ(currentZ, predZ, sample);
    }
    const cacheUpdateTimed =
      pendingEntryCacheOutputs && entryCacheUpdater?.async
        ? timeAsync(() =>
            updateCacheFromEntriesAsync(
              entryCacheUpdater,
              persistentCache,
              pendingEntryCacheOutputs,
              stepCacheNames,
              graphCapturePinnedTensors,
              gpuDevice,
            ),
          )
        : null;
    if (cacheUpdateTimed) {
      pendingEntryCacheOutputs = null;
    }
    const decoderInput = graphCaptureDecoderInput
      ? timeSync(() => {
          copyTensorToGpu(gpuDevice, currentZ, graphCaptureDecoderInput.tensor);
          return graphCaptureDecoderInput.tensor;
        }).value
      : specs.decoder.inputs?.z
        ? currentZ
        : latentFromPredZ(currentZ);
    const decoderOutputs = decoderWorker
      ? (await decoderWorker.run(decoderInput)).value
      : await decoder.session.run(replaceDecoderLatent(decoderBaseFeeds, decoderInput), decoderFetchArg);
    if (cacheUpdateTimed) {
      candidateCache = (await cacheUpdateTimed).value;
    } else if (pendingEntryCacheOutputs) {
      candidateCache = await updateCacheFromEntriesAsync(
        entryCacheUpdater,
        persistentCache,
        pendingEntryCacheOutputs,
        stepCacheNames,
        graphCapturePinnedTensors,
        gpuDevice,
      );
    }
    const oldCache = persistentCache;
    persistentCache = candidateCache;
    if (oldCache !== persistentCache) disposeCache(oldCache, graphCapturePinnedTensors);
    disposeTensorUnlessPinnedAfterSubmittedWork(
      gpuDevice,
      decoderOutputs[decoderName],
      graphCapturePinnedTensors,
    );
    if (usesFusedSampleStep) {
      if (streamingInputZ && currentZ !== streamingInputZ) {
        copyGpuTensor(gpuDevice, currentZ, streamingInputZ);
        disposeTensorUnlessPinnedAfterSubmittedWork(gpuDevice, currentZ, graphCapturePinnedTensors);
      }
      streamingZ = streamingInputZ ?? currentZ;
    } else {
      disposeTensorAfterSubmittedWork(gpuDevice, currentZ);
    }
  }

  const prefillSamples = [];
  const dynamicsFrameSamples = [];
  const decoderFrameSamples = [];
  const cacheCommitSamples = [];
  const packUnpackSamples = [];
  const streamingFrameSamples = [];
  const targetForwardSamples = [];
  const validationFrameHashes = [];
  const validationLatentHashes = [];
  let latestPredStats = null;
  let latestFrameStats = null;
  let pendingDecoderFrame = null;
  let lastDecoderDisplayTime = null;

  const processPipelinedDecoderFrame = async (record) => {
    const decoderTimed = await record.promise;
    const displayTime = performance.now();
    decoderFrameSamples.push(decoderTimed.elapsedMs);
    streamingFrameSamples.push(
      lastDecoderDisplayTime == null
        ? displayTime - record.frameStart
        : displayTime - lastDecoderDisplayTime,
    );
    lastDecoderDisplayTime = displayTime;

    latestPredStats = record.predZ ? tensorSummary(record.predZ) : tensorSummary(record.finalZ);
    const frameOutput = decoderTimed.value[decoderName];
    if (!frameOutput) {
      throw new Error(`Single-frame decoder did not return output ${decoderName}`);
    }
    latestFrameStats = tensorSummary(frameOutput);
    if (
      validationDecoder &&
      validationFrameHashes.length < Math.max(1, Math.floor(config.validationFrames))
    ) {
      validationLatentHashes.push(await tensorContentHash(record.latent, gpuDevice));
      validationFrameHashes.push(await tensorContentHash(frameOutput));
    }
    if (record.predZ) {
      assertDims('cached_step.pred_z', record.predZ.dims, specs.step.outputs[predName].shape);
    }
    assertDims('cached_step.final_z', record.finalZ.dims, specs.step.outputs[finalZName].shape);
    assertDims('single_frame_decoder.output', frameOutput.dims, specs.decoder.outputs[decoderName].shape);
    disposeTensorUnlessPinnedAfterSubmittedWork(gpuDevice, frameOutput, graphCapturePinnedTensors);
    if (usesFusedSampleStep) {
      if (streamingInputZ && record.finalZ !== streamingInputZ) {
        copyGpuTensor(gpuDevice, record.finalZ, streamingInputZ);
        disposeTensorUnlessPinnedAfterSubmittedWork(gpuDevice, record.finalZ, graphCapturePinnedTensors);
      }
      if (streamingZ !== record.finalZ) {
        disposeTensorUnlessPinnedAfterSubmittedWork(gpuDevice, streamingZ, graphCapturePinnedTensors);
      }
      streamingZ = streamingInputZ ?? record.finalZ;
    }
  };

  setStatus('demo benchmark: timed prefill');
  const timedPrefill = await timeAsync(() => prefill.session.run(prefillFeeds, prefillFetches));
  prefillSamples.push(timedPrefill.elapsedMs);
  disposeCache(persistentCache, graphCapturePinnedTensors);
  persistentCache = cacheFromOutputs(timedPrefill.value, prefillCacheNames);
  if (graphCaptureFixedCache) {
    const prefillCache = persistentCache;
    persistentCache = copyCacheIntoFixedGpu(gpuDevice, prefillCache, graphCaptureFixedCache);
    disposeCache(prefillCache, graphCapturePinnedTensors);
  }
  if (usesSafariMaterializedGraphCapture && step.graph_capture && config.primeGraphCapture) {
    setStatus('demo benchmark: priming graph capture');
    let currentZ = makeFloatTensor([1, 1, 32, 32], 6000, zDtype);
    let primeOutputs = null;
    const sampleCount = usesFusedSampleStep ? 1 : SAMPLE_STEPS;
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const signalLevelTensor = needsSignalLevelInput
        ? makeScalarFillTensor('int32', [1, 1], sample)
        : null;
      const positionTensor = needsPositionIndexInput
        ? makeScalarFillTensor('int32', [1], persistentCache.length.data[0])
        : null;
      if (graphCaptureStepInputs?.z && currentZ.location === 'gpu-buffer') {
        copyGpuTensorToTargets(gpuDevice, currentZ, [
          graphCaptureStepInputs.z,
          graphCaptureStepInputs.contextNoise,
        ]);
      } else if (graphCaptureStepInputs?.z) {
        writeCpuTensorToGpu(gpuDevice, graphCaptureStepInputs.z, currentZ);
        writeCpuTensorToGpu(gpuDevice, graphCaptureStepInputs.contextNoise, currentZ);
      }
      const feeds = setStepInputs(applyCacheFeeds(stepBaseFeeds, persistentCache), {
        z: graphCaptureStepInputs?.z ?? currentZ,
        contextNoise: graphCaptureStepInputs?.contextNoise ?? currentZ,
        action: graphCaptureStepInputs?.action ?? actionTensor,
        stepLevel: stepLevelTensor,
        signalLevel: signalLevelTensor,
        positionIndex: positionTensor,
      });
      if (graphCaptureFixedScalars?.cacheLength) {
        feeds.cache_length = graphCaptureFixedScalars.cacheLength;
      }
      if (graphCaptureFixedScalars?.samplePositionIndex) {
        feeds.sample_position_index = graphCaptureFixedScalars.samplePositionIndex;
      }
      if (graphCaptureFixedScalars?.contextPositionIndex) {
        feeds.context_position_index = graphCaptureFixedScalars.contextPositionIndex;
      }
      if (graphCaptureFixedScalars?.attentionMask) {
        feeds.attention_mask = graphCaptureFixedScalars.attentionMask;
      }
      if (graphCaptureFixedScalars?.positionIndex) {
        feeds.position_index = graphCaptureFixedScalars.positionIndex;
      }
      const fetches = sample === sampleCount - 1 ? stepCommitFetchArg : stepPredFetches;
      assertGraphCaptureGpuTensors('prime cached_step graph capture', feeds, fetches);
      primeOutputs = await step.session.run(feeds, fetches);
      const predZ = primeOutputs[predName] ?? null;
      currentZ = usesFusedSampleStep ? primeOutputs[finalZName] : nextSampleZ(currentZ, predZ, sample);
    }
    const decoderInput = graphCaptureDecoderInput
      ? timeSync(() => {
          copyTensorToGpu(gpuDevice, currentZ, graphCaptureDecoderInput.tensor);
          return graphCaptureDecoderInput.tensor;
        }).value
      : specs.decoder.inputs?.z
        ? currentZ
        : latentFromPredZ(currentZ);
    const decoderFeeds = replaceDecoderLatent(decoderBaseFeeds, decoderInput);
    if (decoder.graph_capture) {
      assertGraphCaptureGpuTensors('prime decoder graph capture', decoderFeeds, decoderFetchArg);
    }
    const decoderOutputs = decoderWorker
      ? (await decoderWorker.run(decoderInput)).value
      : await decoder.session.run(decoderFeeds, decoderFetchArg);
    disposeTensorUnlessPinnedAfterSubmittedWork(
      gpuDevice,
      decoderOutputs[decoderName],
      graphCapturePinnedTensors,
    );
    if (primeOutputs) {
      disposeTensorUnlessPinnedAfterSubmittedWork(
        gpuDevice,
        primeOutputs[stepCacheNames.entryK],
        graphCapturePinnedTensors,
      );
      disposeTensorUnlessPinnedAfterSubmittedWork(
        gpuDevice,
        primeOutputs[stepCacheNames.entryV],
        graphCapturePinnedTensors,
      );
      disposeTensorUnlessPinnedAfterSubmittedWork(
        gpuDevice,
        primeOutputs[finalZName],
        graphCapturePinnedTensors,
      );
    }
    streamingZ = null;
  }

  for (let frame = 0; frame < config.timedRuns; frame += 1) {
    setStatus(`demo benchmark: generated frame ${frame + 1}/${config.timedRuns}`);
    const frameStart = performance.now();
    let candidateCache = persistentCache;
    let currentZ =
      config.graphCaptureFreshInput && step.graph_capture
        ? makeFloatTensor([1, 1, 32, 32], 6000 + frame, zDtype)
        : streamingInputZ ??
          (step.graph_capture && streamingZ?.location === 'gpu-buffer'
            ? streamingZ
            : makeFloatTensor([1, 1, 32, 32], 6000 + frame, zDtype));
    let predZ = null;
    let pendingEntryCacheOutputs = null;
    const frameForwardSamples = [];

    const sampleCount = usesFusedSampleStep ? 1 : SAMPLE_STEPS;
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const committedLengthBefore = persistentCache.length.data[0];
      const signalLevelTensor = needsSignalLevelInput
        ? makeScalarFillTensor('int32', [1, 1], sample)
        : null;
      const positionTensor = needsPositionIndexInput
        ? makeScalarFillTensor('int32', [1], committedLengthBefore)
        : null;
      if (graphCaptureStepInputs?.z && currentZ.location === 'gpu-buffer') {
        copyGpuTensorToTargets(gpuDevice, currentZ, [
          graphCaptureStepInputs.z,
          graphCaptureStepInputs.contextNoise,
        ]);
      } else if (graphCaptureStepInputs?.z) {
        writeCpuTensorToGpu(gpuDevice, graphCaptureStepInputs.z, currentZ);
        writeCpuTensorToGpu(gpuDevice, graphCaptureStepInputs.contextNoise, currentZ);
      }
      const feeds = setStepInputs(applyCacheFeeds(stepBaseFeeds, persistentCache), {
        z: graphCaptureStepInputs?.z ?? currentZ,
        contextNoise: graphCaptureStepInputs?.contextNoise ?? currentZ,
        action: graphCaptureStepInputs?.action ?? actionTensor,
        stepLevel: stepLevelTensor,
        signalLevel: signalLevelTensor,
        positionIndex: positionTensor,
      });
      if (graphCaptureFixedScalars?.cacheLength) {
        feeds.cache_length = graphCaptureFixedScalars.cacheLength;
      }
      if (graphCaptureFixedScalars?.samplePositionIndex) {
        feeds.sample_position_index = graphCaptureFixedScalars.samplePositionIndex;
      }
      if (graphCaptureFixedScalars?.contextPositionIndex) {
        feeds.context_position_index = graphCaptureFixedScalars.contextPositionIndex;
      }
      if (graphCaptureFixedScalars?.attentionMask) {
        feeds.attention_mask = graphCaptureFixedScalars.attentionMask;
      }
      if (graphCaptureFixedScalars?.positionIndex) {
        feeds.position_index = graphCaptureFixedScalars.positionIndex;
      }
      const fetches = sample === sampleCount - 1 ? stepCommitFetchArg : stepPredFetches;
      if (step.graph_capture) {
        assertGraphCaptureGpuTensors('timed cached_step graph capture', feeds, fetches);
      }
      const timed = await timeAsync(() => step.session.run(feeds, fetches));
      predZ = timed.value[predName] ?? null;
      if (!usesFusedSampleStep && !predZ) {
        throw new Error(`Cached step did not return output ${predName}`);
      }
      if (sample === sampleCount - 1) {
        if (usesEntryCacheStep) {
          pendingEntryCacheOutputs =
            config.graphCaptureFinalZOnly && step.graph_capture ? null : timed.value;
        } else {
          candidateCache = cacheFromOutputs(timed.value, stepCacheNames, persistentCache.length);
        }
        if (!usesEntryCacheStep && graphCaptureFixedCache) {
          const outputCache = candidateCache;
          candidateCache = copyCacheIntoFixedGpu(gpuDevice, outputCache, graphCaptureFixedCache);
          disposeCache(outputCache, graphCapturePinnedTensors);
        }
      }
      currentZ = usesFusedSampleStep ? timed.value[finalZName] : nextSampleZ(currentZ, predZ, sample);
      if (sample < sampleCount - 1 && persistentCache.length.data[0] !== committedLengthBefore) {
        throw new Error('Committed cache_length changed before the final sample step');
      }
      frameForwardSamples.push(timed.elapsedMs);
      targetForwardSamples.push(timed.elapsedMs);
    }

    const cacheUpdateTimed =
      pendingEntryCacheOutputs && entryCacheUpdater?.async
        ? timeAsync(() =>
            updateCacheFromEntriesAsync(
              entryCacheUpdater,
              persistentCache,
              pendingEntryCacheOutputs,
              stepCacheNames,
              graphCapturePinnedTensors,
              gpuDevice,
            ),
          )
        : null;
    if (cacheUpdateTimed) {
      pendingEntryCacheOutputs = null;
    }
    const packTimed = timeSync(() => {
      if (graphCaptureDecoderInput) {
        copyTensorToGpu(gpuDevice, currentZ, graphCaptureDecoderInput.tensor);
        return graphCaptureDecoderInput.tensor;
      }
      return specs.decoder.inputs?.z ? currentZ : latentFromPredZ(currentZ);
    });
    if (decoderWorker) {
      const decoderPromise = decoderWorker.run(packTimed.value);
      const commitTimed = await timeAsync(async () => {
        if (cacheUpdateTimed) {
          candidateCache = (await cacheUpdateTimed).value;
        }
        if (pendingEntryCacheOutputs) {
          candidateCache = await updateCacheFromEntriesAsync(
            entryCacheUpdater,
            persistentCache,
            pendingEntryCacheOutputs,
            stepCacheNames,
            graphCapturePinnedTensors,
            gpuDevice,
          );
        }
        const oldCache = persistentCache;
        persistentCache = candidateCache;
        if (oldCache !== persistentCache) disposeCache(oldCache, graphCapturePinnedTensors);
      });

      const dynamicsFrameMs = frameForwardSamples.reduce((total, value) => total + value, 0);
      dynamicsFrameSamples.push(dynamicsFrameMs);
      cacheCommitSamples.push(commitTimed.elapsedMs);
      packUnpackSamples.push(packTimed.elapsedMs);
      if (pendingDecoderFrame) {
        await processPipelinedDecoderFrame(pendingDecoderFrame);
      }
      pendingDecoderFrame = {
        promise: decoderPromise,
        latent: packTimed.value,
        predZ,
        finalZ: currentZ,
        frameStart,
      };
      continue;
    }
    const decoderTimed = await timeAsync(() =>
      {
        const decoderFeeds = replaceDecoderLatent(decoderBaseFeeds, packTimed.value);
        if (decoder.graph_capture) {
          assertGraphCaptureGpuTensors('timed decoder graph capture', decoderFeeds, decoderFetchArg);
        }
        return decoder.session.run(decoderFeeds, decoderFetchArg);
      },
    );
    const commitTimed = await timeAsync(async () => {
      if (cacheUpdateTimed) {
        candidateCache = (await cacheUpdateTimed).value;
      }
      if (pendingEntryCacheOutputs) {
        candidateCache = await updateCacheFromEntriesAsync(
          entryCacheUpdater,
          persistentCache,
          pendingEntryCacheOutputs,
          stepCacheNames,
          graphCapturePinnedTensors,
          gpuDevice,
        );
      }
      const oldCache = persistentCache;
      persistentCache = candidateCache;
      if (oldCache !== persistentCache) disposeCache(oldCache, graphCapturePinnedTensors);
    });

    const dynamicsFrameMs = frameForwardSamples.reduce((total, value) => total + value, 0);
    dynamicsFrameSamples.push(dynamicsFrameMs);
    decoderFrameSamples.push(decoderTimed.elapsedMs);
    cacheCommitSamples.push(commitTimed.elapsedMs);
    packUnpackSamples.push(packTimed.elapsedMs);
    streamingFrameSamples.push(performance.now() - frameStart);

    latestPredStats = predZ ? tensorSummary(predZ) : tensorSummary(currentZ);
    const frameOutput = decoderTimed.value[decoderName];
    if (!frameOutput) {
      throw new Error(`Single-frame decoder did not return output ${decoderName}`);
    }
    latestFrameStats = tensorSummary(frameOutput);
    if (
      validationDecoder &&
      validationFrameHashes.length < Math.max(1, Math.floor(config.validationFrames))
    ) {
      validationLatentHashes.push(await tensorContentHash(packTimed.value, gpuDevice));
      const validationOutputs = await validationDecoder.session.run(
        replaceDecoderLatent(decoderBaseFeeds, packTimed.value),
        [decoderName],
      );
      validationFrameHashes.push(await tensorContentHash(validationOutputs[decoderName]));
      disposeTensorAfterSubmittedWork(gpuDevice, validationOutputs[decoderName]);
    }
    if (predZ) {
      assertDims('cached_step.pred_z', predZ.dims, specs.step.outputs[predName].shape);
    }
    assertDims('cached_step.final_z', currentZ.dims, specs.step.outputs[finalZName].shape);
    assertDims('single_frame_decoder.output', frameOutput.dims, specs.decoder.outputs[decoderName].shape);
    disposeTensorUnlessPinnedAfterSubmittedWork(gpuDevice, frameOutput, graphCapturePinnedTensors);
    if (usesFusedSampleStep) {
      if (streamingInputZ && currentZ !== streamingInputZ) {
        copyGpuTensor(gpuDevice, currentZ, streamingInputZ);
        disposeTensorUnlessPinnedAfterSubmittedWork(gpuDevice, currentZ, graphCapturePinnedTensors);
      }
      if (streamingZ !== currentZ) {
        disposeTensorUnlessPinnedAfterSubmittedWork(gpuDevice, streamingZ, graphCapturePinnedTensors);
      }
      streamingZ = streamingInputZ ?? currentZ;
    }
  }
  if (pendingDecoderFrame) {
    await processPipelinedDecoderFrame(pendingDecoderFrame);
    pendingDecoderFrame = null;
  }
  decoderWorker?.release?.();

  const targetForwardAfterGraphCaptureWarmup = summarizeGraphCaptureSteady(
    targetForwardSamples,
    config,
  );
  const dynamicsAfterGraphCaptureWarmup = summarizeGraphCaptureSteady(
    dynamicsFrameSamples,
    config,
  );
  const decoderAfterGraphCaptureWarmup = summarizeGraphCaptureSteady(decoderFrameSamples, config);
  const streamingAfterGraphCaptureWarmup = summarizeGraphCaptureSteady(
    streamingFrameSamples,
    config,
  );
  const steady = summarize(streamingFrameSamples);
  const outputValidation = outputValidationSummary(validationFrameHashes, config, validationLatentHashes);
  return [
    {
      mode: 'cached_prefill',
      artifact_role: 'dynamics_prefill',
      name: prefill.spec.name,
      model_url: prefill.model_url,
      model_fetch_ms: prefill.model_fetch_ms,
      model_bytes: prefill.model_bytes,
      session_create_ms: prefill.session_create_ms,
      preferred_output_location: prefill.preferred_output_location,
      inputs: prefill.spec.inputs,
      outputs: prefill.spec.outputs,
      timing: {
        prefill: summarize(prefillSamples),
      },
      cache: {
        outputs: prefillCacheNames,
        k_cache: Array.isArray(persistentCache.k)
          ? persistentCache.k.map((tensor) => tensorSummary(tensor))
          : tensorSummary(persistentCache.k),
        v_cache: Array.isArray(persistentCache.v)
          ? persistentCache.v.map((tensor) => tensorSummary(tensor))
          : tensorSummary(persistentCache.v),
        cache_length: tensorSummary(persistentCache.length),
      },
    },
    {
      mode: 'cached_step',
      artifact_role: 'dynamics_step',
      name: step.spec.name,
      model_url: step.model_url,
      model_fetch_ms: step.model_fetch_ms,
      model_bytes: step.model_bytes,
      session_create_ms: step.session_create_ms,
      preferred_output_location: step.preferred_output_location,
      graph_capture: step.graph_capture,
      inputs: step.spec.inputs,
      outputs: step.spec.outputs,
      timing: {
        target_forward: summarize(targetForwardSamples),
        target_forward_after_graph_capture_warmup: targetForwardAfterGraphCaptureWarmup,
        dynamics_frame: summarize(dynamicsFrameSamples),
        dynamics_frame_after_graph_capture_warmup: dynamicsAfterGraphCaptureWarmup,
      },
      cache: {
        outputs: stepCacheNames,
        entry_cache_update: usesEntryCacheStep ? entryCacheUpdater : null,
        commit_policy: usesFusedSampleStep
          ? 'fused graph reads committed cache for all samples and returns final candidate cache once per frame'
          : 'discard non-final sample forwards; commit the final sample forward once per frame',
        fetch_policy: usesFusedSampleStep
          ? config.debugStats
            ? 'fetch final_z, pred_z, and GPU cache outputs once per frame'
            : 'fetch GPU final_z and GPU cache outputs once per frame; do not fetch pred_z'
          : 'fetch pred_z for non-final sample forwards; fetch pred_z and GPU cache outputs for the final sample forward',
      },
      output: latestPredStats,
    },
    {
      mode: 'streaming_frame',
      artifact_role: 'demo_frame',
      name: 'dreamer4_demo_streaming_frame',
      inputs: {
        context: specs.prefill.inputs,
        step: specs.step.inputs,
        decoder: specs.decoder.inputs,
      },
      outputs: {
        pred_z_t: specs.step.outputs[predName],
        frame_t: specs.decoder.outputs[decoderName],
      },
      timing: {
        target_forward: summarize(targetForwardSamples),
        target_forward_after_graph_capture_warmup: targetForwardAfterGraphCaptureWarmup,
        dynamics_frame: summarize(dynamicsFrameSamples),
        dynamics_frame_after_graph_capture_warmup: dynamicsAfterGraphCaptureWarmup,
        decoder_frame: summarize(decoderFrameSamples),
        decoder_frame_after_graph_capture_warmup: decoderAfterGraphCaptureWarmup,
        cache_commit: summarize(cacheCommitSamples),
        pack_unpack: summarize(packUnpackSamples),
        streaming_frame: steady,
        streaming_frame_after_graph_capture_warmup: streamingAfterGraphCaptureWarmup,
        steady_state_ms_per_frame: steady?.mean_ms ?? null,
        steady_state_fps: steady == null ? null : 1000 / steady.mean_ms,
        steady_state_after_graph_capture_warmup_ms_per_frame:
          streamingAfterGraphCaptureWarmup?.mean_ms ?? null,
        steady_state_after_graph_capture_warmup_fps:
          streamingAfterGraphCaptureWarmup == null
            ? null
            : 1000 / streamingAfterGraphCaptureWarmup.mean_ms,
        cold_stream_ms_per_frame:
          steady == null
            ? null
            : (prefillSamples[0] + streamingFrameSamples.reduce((total, value) => total + value, 0)) /
              streamingFrameSamples.length,
      },
      cache: {
        commit_policy: usesFusedSampleStep
          ? usesEntryCacheStep
            ? 'cache is updated in-place from per-frame K/V entry outputs; logical cache_length controls the attention mask and advances until full context'
            : 'cache_length advances once per generated frame from fused final candidate cache'
          : 'cache_length advances once per generated frame',
      },
      output_validation: outputValidation,
      output: latestFrameStats,
    },
  ];
}

async function runBenchmark() {
  const config = parseConfig();
  configureAssetBase(config.assetBase);
  setupOrtDiagnostics(config);
  const profiler = setupWebgpuProfiling(config);
  setStatus('starting demo benchmark');
  const manifest = await fetchManifest();
  const gpu = await gpuInfo(config);
  if (config.requireHardwareGpu && isSoftwareGpu(gpu)) {
    throw new Error(
      `WebGPU is using a software adapter instead of the hardware GPU: ${JSON.stringify(gpu)}`,
    );
  }
  const specs = resolveDemoSpecs(manifest, config);
  const missing = missingDemoArtifacts(specs);
  if (missing.length > 0 || manifest.cache_contract?.status === 'contract_only') {
    return blockedResult({ config, manifest, gpu, missing });
  }
  validateDemoSpecs(specs, manifest);

  const results = await runDemoBenchmark({ config, specs, manifest });
  const outputValidation = results.find((entry) => entry.mode === 'streaming_frame')
    ?.output_validation ?? { status: 'skipped' };
  if (profiler.enabled && config.profilingDrainMs > 0) {
    await delay(config.profilingDrainMs);
  }
  const profiling = summarizeProfiling(profiler, config);
  if (config.profilingRequired && profiler.enabled && profiling.event_count === 0) {
    throw new Error('WebGPU profiling was required but no profiling events were received.');
  }
  return {
    schema_version: 2,
    status: outputValidation.status === 'failed' ? 'failed' : 'passed',
    streaming_contract_status:
      outputValidation.status === 'failed' ? 'failed' : 'available',
    ...(outputValidation.status === 'failed'
      ? { message: 'Generated frame output validation failed: sampled frames were static.' }
      : {}),
    benchmark_modes: ['cached_prefill', 'cached_step', 'streaming_frame'],
    config,
    created_at: new Date().toISOString(),
    user_agent: navigator.userAgent,
    platform: navigator.platform,
    ort_version: ort.version ?? null,
    provider_options: {
      executionProviders: [{ name: config.provider }],
      graphOptimizationLevel: config.graphOptimizationLevel,
    },
    gpu,
    sampling: samplingConfig(specs, config.timedRuns),
    cache_abi: cacheAbi(manifest),
    manifest: compactManifest(manifest),
    profiling,
    diagnostics: config.captureConsole ? capturedConsoleMessages.slice(-500) : [],
    results,
  };
}

function finish(result) {
  window.__WEBGPU_BENCHMARK_RESULT__ = result;
  document.getElementById('status').textContent = JSON.stringify(result, null, 2);
  console.log(`WEBGPU_BENCHMARK_RESULT ${JSON.stringify(result)}`);
}

function graphCaptureBlockedResult(error) {
  const config = parseConfig();
  const message = error?.message ?? String(error);
  if (
    !(config.graphCapture || config.dynamicsGraphCapture || config.decoderGraphCapture) ||
    !message.includes('cannot use the graph capture feature') ||
    !message.includes('WebGpuExecutionProvider')
  ) {
    return null;
  }
  return {
    schema_version: 2,
    status: 'blocked',
    streaming_contract_status: 'blocked',
    benchmark_modes: ['cached_prefill', 'cached_step', 'streaming_frame'],
    created_at: new Date().toISOString(),
    user_agent: navigator.userAgent,
    platform: navigator.platform,
    config,
    blocked_reason:
      'ORT WebGPU graph capture is unavailable because at least one node was not assigned to the WebGPU execution provider.',
    message,
    stack: error?.stack ?? null,
  };
}

runBenchmark()
  .then(finish)
  .catch((error) => {
    const result = graphCaptureBlockedResult(error) ?? {
      schema_version: 2,
      status: 'failed',
      streaming_contract_status: 'failed',
      benchmark_modes: ['cached_prefill', 'cached_step', 'streaming_frame'],
      created_at: new Date().toISOString(),
      user_agent: navigator.userAgent,
      platform: navigator.platform,
      message: error?.message ?? String(error),
      stack: error?.stack ?? null,
    };
    finish(result);
  });
