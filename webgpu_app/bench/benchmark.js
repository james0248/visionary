import * as ort from '/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs';

const ASSET_DIR = '/webgpu_app/assets';
const MANIFEST_URL = `${ASSET_DIR}/breakout_onnx_manifest.json`;
ort.env.wasm ??= {};
ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
const GENERATED_FRAMES = 8;
const SAMPLE_STEPS = 4;
const SAMPLE_STEP_LEVEL = 2;
const CONTEXT_STEP_LEVEL = 5;
const CONTEXT_TAU_EFFECTIVE = 29 / 32;
const DEFAULT_CONFIG = {
  mode: 'streaming',
  provider: 'webgpu',
  warmupRuns: 1,
  timedRuns: GENERATED_FRAMES,
  requireHardwareGpu: true,
  profiling: false,
  profilingRequired: false,
  profilingDrainMs: 100,
  profilingTopK: 20,
  debugStats: false,
  graphCapture: false,
  stepArtifact: null,
};
const REQUIRED_ARTIFACTS = {
  prefill: ['breakout_dynamics_prefill_cached_b1_t64', 'breakout_dynamics_prefill_layer_cached_b1_t64'],
  step: [
    'breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4',
    'breakout_dynamics_sample_append_context_slide_full_cache_b1_t1_s4',
    'breakout_dynamics_sample_append_context_slide_b1_t1_s4',
    'breakout_dynamics_sample_append_context_slide_layer_b1_t1_s4',
    'breakout_dynamics_sample_append_context_b1_t1_s4',
    'breakout_dynamics_cached_sample_step_slide_b1_t1_s4',
    'breakout_dynamics_cached_sample_step_b1_t1_s4',
    'breakout_dynamics_step_cached_b1_t1',
  ],
  decoder: ['breakout_tokenizer_decode_z_b1_t1', 'breakout_tokenizer_decoder_b1_t1', 'breakout_decoder_b1_t1'],
};

function setStatus(message) {
  document.getElementById('status').textContent = message;
  console.log(`WEBGPU_BENCHMARK_STATUS ${message}`);
}

function parseConfig() {
  const params = new URLSearchParams(window.location.search);
  return {
    mode: params.get('mode') ?? DEFAULT_CONFIG.mode,
    provider: params.get('provider') ?? DEFAULT_CONFIG.provider,
    warmupRuns: Number(params.get('warmupRuns') ?? DEFAULT_CONFIG.warmupRuns),
    timedRuns: Number(params.get('timedRuns') ?? DEFAULT_CONFIG.timedRuns),
    requireHardwareGpu:
      (params.get('requireHardwareGpu') ?? String(DEFAULT_CONFIG.requireHardwareGpu)) === 'true',
    profiling: (params.get('profiling') ?? String(DEFAULT_CONFIG.profiling)) === 'true',
    profilingRequired:
      (params.get('profilingRequired') ?? String(DEFAULT_CONFIG.profilingRequired)) === 'true',
    profilingDrainMs: Number(params.get('profilingDrainMs') ?? DEFAULT_CONFIG.profilingDrainMs),
    profilingTopK: Number(params.get('profilingTopK') ?? DEFAULT_CONFIG.profilingTopK),
    debugStats: (params.get('debugStats') ?? String(DEFAULT_CONFIG.debugStats)) === 'true',
    graphCapture: (params.get('graphCapture') ?? String(DEFAULT_CONFIG.graphCapture)) === 'true',
    stepArtifact: params.get('stepArtifact') ?? DEFAULT_CONFIG.stepArtifact,
  };
}

function mul(shape) {
  return shape.reduce((total, value) => total * value, 1);
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

function writeCpuTensorToGpu(device, gpuTensor, cpuTensor) {
  device.queue.writeBuffer(gpuTensor.gpuBuffer, 0, tensorDataBytes(cpuTensor));
}

function copyGpuTensor(device, source, target) {
  if (source === target) return;
  const byteLength = tensorByteLength(source.type, source.dims);
  const encoder = device.createCommandEncoder();
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

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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
  setStatus(`creating session ${modelUrl}`);
  return timeAsync(() =>
    ort.InferenceSession.create(modelUrl, {
      executionProviders: [{ name: provider }],
      externalData,
      graphOptimizationLevel: 'all',
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

function hasTimestampQuery(gpu) {
  return (
    gpu.features.includes('timestamp-query') ||
    gpu.features.includes('chromium-experimental-timestamp-query-inside-passes')
  );
}

function groupProfileEvents(events, keyFn, topK) {
  const groups = new Map();
  for (const event of events) {
    const key = keyFn(event);
    if (key == null) continue;
    const entry = groups.get(key) ?? {
      key,
      event_count: 0,
      total_ms: 0,
      min_ms: Number.POSITIVE_INFINITY,
      max_ms: 0,
      examples: [],
    };
    entry.event_count += 1;
    entry.total_ms += event.duration_ms;
    entry.min_ms = Math.min(entry.min_ms, event.duration_ms);
    entry.max_ms = Math.max(entry.max_ms, event.duration_ms);
    if (entry.examples.length < 3) {
      entry.examples.push({
        kernelName: event.kernelName,
        kernelType: event.kernelType,
        programName: event.programName,
        inputsMetadata: event.inputsMetadata,
        outputsMetadata: event.outputsMetadata,
      });
    }
    groups.set(key, entry);
  }
  return [...groups.values()]
    .map((entry) => ({
      ...entry,
      mean_ms: entry.event_count === 0 ? 0 : entry.total_ms / entry.event_count,
      min_ms: entry.min_ms === Number.POSITIVE_INFINITY ? 0 : entry.min_ms,
    }))
    .sort((a, b) => b.total_ms - a.total_ms)
    .slice(0, topK);
}

function summarizeProfileEvents(events, scopes, topK) {
  const byRole = {};
  for (const role of ['cached_prefill', 'cached_step', 'single_frame_decoder']) {
    const roleEvents = events.filter((event) => event.role === role);
    byRole[role] = {
      event_count: roleEvents.length,
      total_ms: roleEvents.reduce((total, event) => total + event.duration_ms, 0),
      top_kernels: groupProfileEvents(
        roleEvents,
        (event) => `${event.programName}|${event.kernelName}|${event.kernelType}`,
        topK,
      ),
    };
  }
  const byScope = scopes.map((scope) => {
    const scopeEvents = events.filter((event) => event.scope_id === scope.scope_id);
    return {
      scope_id: scope.scope_id,
      run_id: scope.run_id,
      role: scope.role,
      phase: scope.phase,
      frame: scope.frame ?? null,
      sample: scope.sample ?? null,
      event_count: scopeEvents.length,
      total_ms: scopeEvents.reduce((total, event) => total + event.duration_ms, 0),
      top_kernels: groupProfileEvents(
        scopeEvents,
        (event) => `${event.programName}|${event.kernelName}|${event.kernelType}`,
        topK,
      ),
    };
  });
  return {
    by_role: byRole,
    by_scope: byScope,
    top_programs: groupProfileEvents(events, (event) => event.programName, topK),
    top_kernel_names: groupProfileEvents(events, (event) => event.kernelName, topK),
    top_kernel_types: groupProfileEvents(events, (event) => event.kernelType, topK),
  };
}

function createProfilingCollector({ config, gpu }) {
  const enabled = config.profiling;
  const available = enabled && hasTimestampQuery(gpu);
  const reason = enabled
    ? available
      ? null
      : 'WebGPU adapter does not expose timestamp-query'
    : 'profiling query param is false';
  const collector = {
    enabled,
    required: config.profilingRequired,
    available,
    reason,
    mode: available ? 'default' : null,
    source: 'ort.env.webgpu.profiling.ondata',
    time_unit: 'ns',
    drainMs: config.profilingDrainMs,
    topK: config.profilingTopK,
    activeScope: null,
    nextRunId: 0,
    hasOpenedScope: false,
    scopes: [],
    rawEvents: [],
    lateEvents: [],
    unscopedEvents: [],
    normalize(event, scope) {
      const durationNs = event.endTime - event.startTime;
      return {
        scope_id: scope?.scope_id ?? null,
        run_id: scope?.run_id ?? null,
        role: scope?.role ?? null,
        phase: scope?.phase ?? null,
        frame: scope?.frame ?? null,
        sample: scope?.sample ?? null,
        kernelId: event.kernelId,
        kernelType: event.kernelType,
        kernelName: event.kernelName,
        programName: event.programName,
        inputsMetadata: event.inputsMetadata,
        outputsMetadata: event.outputsMetadata,
        startTime: event.startTime,
        endTime: event.endTime,
        duration_ns: durationNs,
        duration_ms: durationNs / 1_000_000,
      };
    },
    async profileScope(scopeInfo, fn) {
      if (!this.available) return fn();
      if (this.activeScope != null) {
        throw new Error(`Profiling scope overlap: ${this.activeScope.scope_id}`);
      }
      const runId = this.nextRunId;
      this.nextRunId += 1;
      const scope = {
        scope_id: [
          scopeInfo.role,
          scopeInfo.phase,
          scopeInfo.frame == null ? null : `frame=${scopeInfo.frame}`,
          scopeInfo.sample == null ? null : `sample=${scopeInfo.sample}`,
          `run=${runId}`,
        ]
          .filter(Boolean)
          .join(':'),
        run_id: runId,
        role: scopeInfo.role,
        phase: scopeInfo.phase,
        frame: scopeInfo.frame ?? null,
        sample: scopeInfo.sample ?? null,
        started_at_ms: performance.now(),
        ended_at_ms: null,
        drain_ms: this.drainMs,
        event_count: 0,
      };
      this.activeScope = scope;
      this.hasOpenedScope = true;
      try {
        const value = await fn();
        await delay(this.drainMs);
        scope.ended_at_ms = performance.now();
        scope.event_count = this.rawEvents.filter((event) => event.scope_id === scope.scope_id).length;
        this.scopes.push(scope);
        return value;
      } finally {
        this.activeScope = null;
      }
    },
    result() {
      const callbackEventsEmitted =
        !this.enabled || !this.available || this.scopes.length === 0 || this.rawEvents.length > 0;
      const available = this.available && callbackEventsEmitted;
      const reason = callbackEventsEmitted
        ? this.reason
        : 'ORT WebGPU profiling was configured, but the runtime did not emit profiling callbacks';
      return {
        enabled: this.enabled,
        required: this.required,
        available,
        reason,
        mode: available ? this.mode : null,
        source: this.source,
        time_unit: this.time_unit,
        attribution: {
          strategy: 'single active scope around timed session.run plus per-phase drain',
          strict_scope_protocol: true,
          drain_ms: this.drainMs,
          scope_count: this.scopes.length,
          late_event_count: this.lateEvents.length,
          unscoped_event_count: this.unscopedEvents.length,
        },
        scopes: this.scopes,
        raw_events: this.rawEvents,
        late_events: this.lateEvents,
        unscoped_events: this.unscopedEvents,
        summary: summarizeProfileEvents(this.rawEvents, this.scopes, this.topK),
      };
    },
  };

  ort.env.webgpu ??= {};
  if (enabled && available) {
    ort.env.webgpu.profilingMode = 'default';
    ort.env.webgpu.profiling = {
      mode: 'default',
      ondata: (event) => {
        if (collector.activeScope != null) {
          collector.rawEvents.push(collector.normalize(event, collector.activeScope));
        } else if (collector.hasOpenedScope) {
          collector.lateEvents.push(collector.normalize(event, null));
        } else {
          collector.unscopedEvents.push(collector.normalize(event, null));
        }
      },
    };
  } else {
    ort.env.webgpu.profilingMode = 'off';
    ort.env.webgpu.profiling = { mode: 'off' };
  }

  if (enabled && !available && config.profilingRequired) {
    throw new Error(reason);
  }

  return collector;
}

function externalDataForSpec(spec) {
  return (spec.external_data ?? []).map((entry) => ({
    path: entry.path,
    data: `${ASSET_DIR}/${entry.path}`,
  }));
}

function byExportName(manifest) {
  return Object.fromEntries((manifest.exports ?? []).map((entry) => [entry.name, entry]));
}

function findSpec(exportsByName, names) {
  for (const name of names) {
    if (exportsByName[name]) return exportsByName[name];
  }
  return null;
}

function resolveDemoSpecs(manifest, config = DEFAULT_CONFIG) {
  const exportsByName = byExportName(manifest);
  const stepNames = config.stepArtifact
    ? [config.stepArtifact, ...REQUIRED_ARTIFACTS.step]
    : REQUIRED_ARTIFACTS.step;
  return {
    prefill: findSpec(exportsByName, REQUIRED_ARTIFACTS.prefill),
    step: findSpec(exportsByName, stepNames),
    decoder: findSpec(exportsByName, REQUIRED_ARTIFACTS.decoder),
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

function sameShape(actual, expected) {
  return (
    Array.isArray(actual) &&
    Array.isArray(expected) &&
    actual.length === expected.length &&
    actual.every((value, index) => value === expected[index])
  );
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
  const cacheLengthShape = tensors.cache_length?.shape;
  const layerCacheShape = manifest.cache_contract?.tensors?.layer_cache?.shape;
  const layerCacheCount = manifest.cache_contract?.tensors?.layer_cache?.layers ?? 0;
  const hasLayerPrefill = layerCacheCount > 0 && Boolean(specs.prefill.outputs?.k_cache_0);
  const hasLayerStep = layerCacheCount > 0 && Boolean(specs.step.inputs?.k_cache_0);
  const hasEntryStep = Boolean(specs.step.outputs?.candidate_k_entry);
  const entryShape = cacheShape ? [...cacheShape.slice(0, 3), 1, ...cacheShape.slice(4)] : null;
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
  requireTensorSpec('step.outputs.pred_z', specs.step.outputs?.pred_z, [1, 1, 32, 32]);
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

function samplingConfig(specs = null) {
  const sampleSteps = specs?.step?.sample_steps ?? SAMPLE_STEPS;
  return {
    sample_steps: sampleSteps,
    sample_step_level: Math.log2(sampleSteps),
    context_step_level: CONTEXT_STEP_LEVEL,
    context_tau_effective: CONTEXT_TAU_EFFECTIVE,
    generated_frames: GENERATED_FRAMES,
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

function blockedResult({ config, manifest, gpu, missing, profiler }) {
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
      executionProviders: [{ name: config.provider }],
      graphOptimizationLevel: 'all',
    },
    gpu,
    profiling: profiler.result(),
    sampling: samplingConfig(),
    cache_abi: cacheAbi(manifest),
    manifest: compactManifest(manifest),
    results: [],
  };
}

function makeFeedForInput(name, spec, seed) {
  const shape = spec.shape;
  const dtype = spec.dtype;
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
    feeds[name] = makeFeedForInput(name, inputSpec, seedBase + index * 13);
    index += 1;
  }
  return feeds;
}

function cacheOutputNames(spec) {
  const outputs = Object.keys(spec.outputs ?? {});
  const kLayers = outputs.filter((name) => /^k_cache_\d+$/.test(name)).sort();
  const vLayers = outputs.filter((name) => /^v_cache_\d+$/.test(name)).sort();
  const candidateKLayers = outputs.filter((name) => /^candidate_k_cache_\d+$/.test(name)).sort();
  const candidateVLayers = outputs.filter((name) => /^candidate_v_cache_\d+$/.test(name)).sort();
  return {
    k: candidateKLayers.length ? candidateKLayers : kLayers.length ? kLayers : outputs.find((name) => name === 'k_cache' || name.endsWith('_k_cache')),
    v: candidateVLayers.length ? candidateVLayers : vLayers.length ? vLayers : outputs.find((name) => name === 'v_cache' || name.endsWith('_v_cache')),
    entryK: outputs.find((name) => name === 'candidate_k_entry' || name.endsWith('_k_entry')),
    entryV: outputs.find((name) => name === 'candidate_v_entry' || name.endsWith('_v_entry')),
    length: outputs.find((name) => name === 'cache_length' || name.endsWith('_cache_length')),
  };
}

function stepPredOutputName(spec) {
  const outputs = Object.keys(spec.outputs ?? {});
  return (
    outputs.find((name) => name === 'pred_z') ??
    outputs.find((name) => name.endsWith('pred_z')) ??
    outputs[0]
  );
}

function stepFinalZOutputName(spec) {
  const outputs = Object.keys(spec.outputs ?? {});
  return outputs.find((name) => name === 'final_z') ?? stepPredOutputName(spec);
}

function decoderOutputName(spec) {
  const outputs = Object.keys(spec.outputs ?? {});
  return outputs.find((name) => name === 'patches') ?? outputs[0];
}

function applyCacheFeeds(feeds, cache) {
  const next = { ...feeds };
  for (const name of Object.keys(next)) {
    const kLayer = /^k_cache_(\d+)$/.exec(name);
    const vLayer = /^v_cache_(\d+)$/.exec(name);
    if (kLayer && Array.isArray(cache.k)) next[name] = cache.k[Number(kLayer[1])];
    else if (name === 'k_cache' || name.endsWith('_k_cache')) next[name] = cache.k;
    if (vLayer && Array.isArray(cache.v)) next[name] = cache.v[Number(vLayer[1])];
    else if (name === 'v_cache' || name.endsWith('_v_cache')) next[name] = cache.v;
    if (name === 'cache_length' || name.endsWith('_cache_length')) next[name] = cache.length;
  }
  return next;
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
  if (Object.keys(next).some((name) => name.includes('position_index'))) {
    next = replaceNamedFeed(next, ['position_index'], positionIndex);
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

function inputCacheNames(spec) {
  const inputs = Object.keys(spec.inputs ?? {});
  const kLayers = inputs.filter((name) => /^k_cache_\d+$/.test(name)).sort();
  const vLayers = inputs.filter((name) => /^v_cache_\d+$/.test(name)).sort();
  return {
    k: kLayers.length ? kLayers : inputs.find((name) => name === 'k_cache' || name.endsWith('_k_cache')),
    v: vLayers.length ? vLayers : inputs.find((name) => name === 'v_cache' || name.endsWith('_v_cache')),
    length: inputs.find((name) => name === 'cache_length' || name.endsWith('_cache_length')),
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
  const cacheLengthName = findInputName(spec, ['cache_length']);
  if (cacheLengthName) {
    const inputSpec = spec.inputs[cacheLengthName];
    fixed.cacheLength = createGpuTensorFromCpu(
      device,
      makeScalarFillTensor(inputSpec.dtype, inputSpec.shape, 64),
    );
  }
  const positionIndexName = findInputName(spec, ['position_index']);
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
  const [layers, batch, tokens, contextLength, heads, headDim] = cacheSpec.shape;
  const halfHeadDim = headDim / 2;
  if (!Number.isInteger(halfHeadDim)) {
    throw new Error(`Entry-cache update requires an even head_dim, got ${headDim}.`);
  }
  const ropeBase = Number(manifest.dynamics?.rope_base ?? manifest.dynamics?.base ?? 10000);
  const workgroupSize = 64;
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
const ROPE_BASE: f32 = ${ropeBase.toFixed(1)};

@group(0) @binding(0) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(1) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(2) var<storage, read> k_entry: array<f32>;
@group(0) @binding(3) var<storage, read> v_entry: array<f32>;

fn cache_index(layer: u32, batch: u32, token: u32, time: u32, head: u32, dim: u32) -> u32 {
  return (((((layer * BATCH + batch) * TOKENS + token) * CONTEXT + time) * HEADS + head) * HEAD_DIM + dim);
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

    let theta = 1.0 / pow(ROPE_BASE, f32(half_dim) / f32(HALF_HEAD_DIM));
    let cos_theta = cos(theta);
    let sin_theta = sin(theta);
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
  return {
    kind: 'webgpu_inplace_slide_rebase_entry',
    rope_base: ropeBase,
    cache_shape: cacheSpec.shape,
    entry_shape: entrySpec.shape,
    update(cache, kEntry, vEntry) {
      const bindGroup = device.createBindGroup({
        label: 'visionary-entry-cache-slide-rebase-bind-group',
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: cache.k.gpuBuffer } },
          { binding: 1, resource: { buffer: cache.v.gpuBuffer } },
          { binding: 2, resource: { buffer: kEntry.gpuBuffer } },
          { binding: 3, resource: { buffer: vEntry.gpuBuffer } },
        ],
      });
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

function updateCacheFromEntries(updater, cache, outputs, names) {
  const kEntry = outputs[names.entryK];
  const vEntry = outputs[names.entryV];
  if (!kEntry || !vEntry) {
    throw new Error('Entry-cache step did not return candidate_k_entry/candidate_v_entry.');
  }
  if (kEntry.location !== 'gpu-buffer' || vEntry.location !== 'gpu-buffer') {
    throw new Error(
      `Entry-cache update requires GPU entry tensors, got ${kEntry.location}/${vEntry.location}.`,
    );
  }
  updater.update(cache, kEntry, vEntry);
  disposeTensorIfOwned(kEntry);
  disposeTensorIfOwned(vEntry);
  return cache;
}

function cacheFetches(names) {
  if (names.entryK && names.entryV) return [names.entryK, names.entryV];
  return [names.k, names.v, names.length].flat().filter(Boolean);
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

function disposeCache(cache, pinned = []) {
  if (Array.isArray(cache?.k)) cache.k.forEach((tensor) => disposeTensorUnlessPinned(tensor, pinned));
  else disposeTensorUnlessPinned(cache?.k, pinned);
  if (Array.isArray(cache?.v)) cache.v.forEach((tensor) => disposeTensorUnlessPinned(tensor, pinned));
  else disposeTensorUnlessPinned(cache?.v, pinned);
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
  const latentName =
    Object.keys(next).find((name) => name === 'z') ??
    Object.keys(next).find((name) => name.includes('latent')) ??
    Object.keys(next)[0];
  next[latentName] = latent;
  return next;
}

function preferredOutputLocationFor(role, spec, config) {
  if (config.provider !== 'webgpu') {
    return undefined;
  }
  if (role === 'cached_prefill') {
    const names = cacheOutputNames(spec);
    const locations = {
    };
    for (const name of [names.k, names.v].flat().filter(Boolean)) locations[name] = 'gpu-buffer';
    if (names.length) locations[names.length] = 'cpu';
    return locations;
  }
  if (role === 'cached_step') {
    if (config.graphCapture) {
      return Object.fromEntries(Object.keys(spec.outputs ?? {}).map((name) => [name, 'gpu-buffer']));
    }
    const names = cacheOutputNames(spec);
    const predName = stepPredOutputName(spec);
    const finalName = stepFinalZOutputName(spec);
    const usesFusedSampleStep = finalName !== predName;
    const locations = {
      [predName]: usesFusedSampleStep && !config.debugStats ? 'gpu-buffer' : 'cpu',
    };
    for (const name of [names.k, names.v].flat().filter(Boolean)) locations[name] = 'gpu-buffer';
    for (const name of [names.entryK, names.entryV].filter(Boolean)) locations[name] = 'gpu-buffer';
    if (names.length && !config.graphCapture) locations[names.length] = 'cpu';
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
  return undefined;
}

async function createBenchSession(role, spec, config) {
  const modelUrl = `${ASSET_DIR}/${spec.path}`;
  const modelFetch = await fetchSize(modelUrl);
  const preferredOutputLocation = preferredOutputLocationFor(role, spec, config);
  const enableGraphCapture =
    config.provider === 'webgpu' &&
    config.graphCapture &&
    (role === 'cached_step' || role === 'single_frame_decoder');
  const graphOptimizationLevel = 'all';
  const sessionCreate = await createSession(
    modelUrl,
    externalDataForSpec(spec),
    {
      provider: config.provider,
      graphOptimizationLevel,
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

async function runDemoBenchmark({ config, specs, manifest, profiler }) {
  const prefill = await createBenchSession('cached_prefill', specs.prefill, config);
  const step = await createBenchSession('cached_step', specs.step, config);
  const decoder = await createBenchSession('single_frame_decoder', specs.decoder, config);
  const prefillFeeds = makeFeedsFromSpec(specs.prefill, 1000);
  const stepBaseFeeds = makeFeedsFromSpec(specs.step, 2000);
  const decoderBaseFeeds = makeFeedsFromSpec(specs.decoder, 3000);
  const prefillCacheNames = cacheOutputNames(specs.prefill);
  const stepCacheNames = cacheOutputNames(specs.step);
  const usesEntryCacheStep = Boolean(stepCacheNames.entryK && stepCacheNames.entryV);
  const predName = stepPredOutputName(specs.step);
  const finalZName = stepFinalZOutputName(specs.step);
  const usesFusedSampleStep = finalZName !== predName;
  const decoderName = decoderOutputName(specs.decoder);
  const prefillFetches = cacheFetches(prefillCacheNames);
  const stepCacheFetches =
    config.graphCapture && step.graph_capture && !usesEntryCacheStep
      ? [stepCacheNames.k, stepCacheNames.v].flat().filter(Boolean)
      : cacheFetches(stepCacheNames);
  const stepPredFetches = [...new Set([predName, finalZName])];
  const stepCommitFetches = usesFusedSampleStep && !config.debugStats
    ? [...new Set([finalZName, ...stepCacheFetches])]
    : [...new Set([predName, finalZName, ...stepCacheFetches])];
  const decoderFetches = [decoderName];
  const actionTensor = makeIntTensor([1, 1], 4000, 4);
  const zDtype = stepZInputDtype(specs.step);
  const gpuDevice = config.provider === 'webgpu' ? (ort.env.webgpu?.device ?? null) : null;
  const entryCacheUpdater =
    usesEntryCacheStep && gpuDevice ? createEntryCacheUpdater(gpuDevice, specs.step, manifest) : null;
  if (usesEntryCacheStep && !entryCacheUpdater) {
    throw new Error('Entry-cache artifact requires provider=webgpu and an ORT WebGPU device.');
  }
  const graphCaptureStepInputs =
    config.graphCapture && step.graph_capture && gpuDevice
      ? {
          action: createGpuTensorFromCpu(gpuDevice, actionTensor),
          z: createGpuTensorFromCpu(gpuDevice, makeFloatTensor([1, 1, 32, 32], 5999, zDtype)),
          contextNoise: createGpuTensorFromCpu(
            gpuDevice,
            makeFloatTensor([1, 1, 32, 32], 6001, zDtype),
          ),
        }
      : null;
  const graphCaptureFixedCache =
    config.graphCapture && step.graph_capture && gpuDevice
      ? createFixedGpuCache(gpuDevice, specs.step)
      : null;
  const graphCaptureFixedScalars =
    config.graphCapture && step.graph_capture && gpuDevice
      ? createFixedGpuScalarInputs(gpuDevice, specs.step)
      : null;
  const graphCapturePinnedTensors = fixedCachePinnedTensors(graphCaptureFixedCache);
  const stepLevelTensor = makeScalarFillTensor('int32', [1, 1], SAMPLE_STEP_LEVEL);

  setStatus('demo benchmark: first prefill');
  const prefillFirst = await timeAsync(() => prefill.session.run(prefillFeeds, prefillFetches));
  let persistentCache = cacheFromOutputs(prefillFirst.value, prefillCacheNames);
  if (graphCaptureFixedCache) {
    const prefillCache = persistentCache;
    persistentCache = copyCacheIntoFixedGpu(gpuDevice, prefillCache, graphCaptureFixedCache);
    disposeCache(prefillCache, graphCapturePinnedTensors);
  }
  let streamingZ = graphCaptureStepInputs?.z ?? null;

  for (let i = 0; i < config.warmupRuns; i += 1) {
    setStatus(`demo benchmark: warmup frame ${i + 1}/${config.warmupRuns}`);
    let candidateCache = persistentCache;
    let currentZ = makeFloatTensor([1, 1, 32, 32], 5000 + i, zDtype);
    let predZ = null;
    const sampleCount = usesFusedSampleStep ? 1 : SAMPLE_STEPS;
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const signalLevelTensor = makeScalarFillTensor('int32', [1, 1], sample);
      const positionTensor = makeScalarFillTensor('int32', [1], persistentCache.length.data[0]);
      if (graphCaptureStepInputs?.z && currentZ.location === 'gpu-buffer') {
        copyGpuTensor(gpuDevice, currentZ, graphCaptureStepInputs.z);
        copyGpuTensor(gpuDevice, currentZ, graphCaptureStepInputs.contextNoise);
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
      if (graphCaptureFixedScalars?.positionIndex) {
        feeds.position_index = graphCaptureFixedScalars.positionIndex;
      }
      const fetches = sample === sampleCount - 1 ? stepCommitFetches : stepPredFetches;
      const outputs = await step.session.run(feeds, fetches);
      predZ = outputs[predName] ?? null;
      if (sample === sampleCount - 1) {
        if (usesEntryCacheStep) {
          candidateCache = updateCacheFromEntries(
            entryCacheUpdater,
            persistentCache,
            outputs,
            stepCacheNames,
          );
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
    const oldCache = persistentCache;
    persistentCache = candidateCache;
    if (oldCache !== persistentCache) disposeCache(oldCache, graphCapturePinnedTensors);
    const decoderInput = specs.decoder.inputs?.z ? currentZ : latentFromPredZ(currentZ);
    const decoderOutputs = await decoder.session.run(
      replaceDecoderLatent(decoderBaseFeeds, decoderInput),
      decoderFetches,
    );
    disposeTensorIfOwned(decoderOutputs[decoderName]);
    if (usesFusedSampleStep) {
      streamingZ = currentZ;
    } else {
      disposeTensorIfOwned(currentZ);
    }
  }

  const prefillSamples = [];
  const dynamicsFrameSamples = [];
  const decoderFrameSamples = [];
  const cacheCommitSamples = [];
  const packUnpackSamples = [];
  const streamingFrameSamples = [];
  const targetForwardSamples = [];
  let latestPredStats = null;
  let latestFrameStats = null;

  setStatus('demo benchmark: timed prefill');
  const timedPrefill = await profiler.profileScope(
    {
      role: 'cached_prefill',
      phase: 'prefill',
    },
    () => timeAsync(() => prefill.session.run(prefillFeeds, prefillFetches)),
  );
  prefillSamples.push(timedPrefill.elapsedMs);
  disposeCache(persistentCache, graphCapturePinnedTensors);
  persistentCache = cacheFromOutputs(timedPrefill.value, prefillCacheNames);
  if (graphCaptureFixedCache) {
    const prefillCache = persistentCache;
    persistentCache = copyCacheIntoFixedGpu(gpuDevice, prefillCache, graphCaptureFixedCache);
    disposeCache(prefillCache, graphCapturePinnedTensors);
  }

  for (let frame = 0; frame < config.timedRuns; frame += 1) {
    setStatus(`demo benchmark: generated frame ${frame + 1}/${config.timedRuns}`);
    const frameStart = performance.now();
    let candidateCache = persistentCache;
    let currentZ =
      config.graphCapture && streamingZ?.location === 'gpu-buffer'
        ? streamingZ
        : makeFloatTensor([1, 1, 32, 32], 6000 + frame, zDtype);
    let predZ = null;
    const frameForwardSamples = [];

    const sampleCount = usesFusedSampleStep ? 1 : SAMPLE_STEPS;
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const committedLengthBefore = persistentCache.length.data[0];
      const signalLevelTensor = makeScalarFillTensor('int32', [1, 1], sample);
      const positionTensor = makeScalarFillTensor('int32', [1], committedLengthBefore);
      if (graphCaptureStepInputs?.z && currentZ.location === 'gpu-buffer') {
        copyGpuTensor(gpuDevice, currentZ, graphCaptureStepInputs.z);
        copyGpuTensor(gpuDevice, currentZ, graphCaptureStepInputs.contextNoise);
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
      if (graphCaptureFixedScalars?.positionIndex) {
        feeds.position_index = graphCaptureFixedScalars.positionIndex;
      }
      const timed = await profiler.profileScope(
        {
          role: 'cached_step',
          phase: 'target_forward',
          frame,
          sample,
        },
        () => {
          const fetches = sample === sampleCount - 1 ? stepCommitFetches : stepPredFetches;
          return timeAsync(() => step.session.run(feeds, fetches));
        },
      );
      predZ = timed.value[predName] ?? null;
      if (!usesFusedSampleStep && !predZ) {
        throw new Error(`Cached step did not return output ${predName}`);
      }
      if (sample === sampleCount - 1) {
        if (usesEntryCacheStep) {
          candidateCache = updateCacheFromEntries(
            entryCacheUpdater,
            persistentCache,
            timed.value,
            stepCacheNames,
          );
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

    const packTimed = timeSync(() =>
      specs.decoder.inputs?.z ? currentZ : latentFromPredZ(currentZ),
    );
    const decoderTimed = await profiler.profileScope(
      {
        role: 'single_frame_decoder',
        phase: 'decoder_frame',
        frame,
      },
      () =>
        timeAsync(() =>
          decoder.session.run(replaceDecoderLatent(decoderBaseFeeds, packTimed.value), decoderFetches),
        ),
    );
    const commitTimed = timeSync(() => {
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
    if (predZ) {
      assertDims('cached_step.pred_z', predZ.dims, specs.step.outputs[predName].shape);
    }
    assertDims('cached_step.final_z', currentZ.dims, specs.step.outputs[finalZName].shape);
    assertDims('single_frame_decoder.output', frameOutput.dims, specs.decoder.outputs[decoderName].shape);
    disposeTensorIfOwned(frameOutput);
    if (usesFusedSampleStep) {
      if (streamingZ !== currentZ) {
        disposeTensorIfOwned(streamingZ);
      }
      streamingZ = currentZ;
    }
  }

  const steady = summarize(streamingFrameSamples);
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
        dynamics_frame: summarize(dynamicsFrameSamples),
      },
      cache: {
        outputs: stepCacheNames,
        entry_cache_update: usesEntryCacheStep ? entryCacheUpdater : null,
        commit_policy: usesFusedSampleStep
          ? 'fused graph reads committed cache for all samples and returns final candidate cache once per frame'
          : 'discard sample forwards 1-3; commit sample forward 4 once per frame',
        fetch_policy: usesFusedSampleStep
          ? config.debugStats
            ? 'fetch final_z, pred_z, and GPU cache outputs once per frame'
            : 'fetch GPU final_z and GPU cache outputs once per frame; do not fetch pred_z'
          : 'fetch pred_z for sample forwards 1-3; fetch pred_z and GPU cache outputs for sample forward 4',
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
        dynamics_frame: summarize(dynamicsFrameSamples),
        decoder_frame: summarize(decoderFrameSamples),
        cache_commit: summarize(cacheCommitSamples),
        pack_unpack: summarize(packUnpackSamples),
        streaming_frame: steady,
        steady_state_ms_per_frame: steady?.mean_ms ?? null,
        steady_state_fps: steady == null ? null : 1000 / steady.mean_ms,
        cold_stream_ms_per_frame:
          steady == null
            ? null
            : (prefillSamples[0] + streamingFrameSamples.reduce((total, value) => total + value, 0)) /
              streamingFrameSamples.length,
      },
      cache: {
        commit_policy: usesFusedSampleStep
          ? usesEntryCacheStep
            ? 'cache is updated in-place from per-frame K/V entry outputs; cache_length stays fixed at full context'
            : 'cache_length advances once per generated frame from fused final candidate cache'
          : 'cache_length advances once per generated frame',
      },
      output: latestFrameStats,
    },
  ];
}

async function runBenchmark() {
  const config = parseConfig();
  setStatus('starting demo benchmark');
  const manifest = await fetchManifest();
  const gpu = await gpuInfo(config);
  if (config.requireHardwareGpu && isSoftwareGpu(gpu)) {
    throw new Error(
      `WebGPU is using a software adapter instead of the hardware GPU: ${JSON.stringify(gpu)}`,
    );
  }
  const profiler = createProfilingCollector({ config, gpu });

  const specs = resolveDemoSpecs(manifest, config);
  const missing = missingDemoArtifacts(specs);
  if (missing.length > 0 || manifest.cache_contract?.status === 'contract_only') {
    return blockedResult({ config, manifest, gpu, missing, profiler });
  }
  validateDemoSpecs(specs, manifest);

  const results = await runDemoBenchmark({ config, specs, manifest, profiler });
  return {
    schema_version: 2,
    status: 'passed',
    streaming_contract_status: 'available',
    benchmark_modes: ['cached_prefill', 'cached_step', 'streaming_frame'],
    config,
    created_at: new Date().toISOString(),
    user_agent: navigator.userAgent,
    platform: navigator.platform,
    ort_version: ort.version ?? null,
    provider_options: {
      executionProviders: [{ name: config.provider }],
      graphOptimizationLevel: 'all',
    },
    gpu,
    profiling: profiler.result(),
    sampling: samplingConfig(specs),
    cache_abi: cacheAbi(manifest),
    manifest: compactManifest(manifest),
    results,
  };
}

function finish(result) {
  window.__WEBGPU_BENCHMARK_RESULT__ = result;
  document.getElementById('status').textContent = JSON.stringify(result, null, 2);
  console.log(`WEBGPU_BENCHMARK_RESULT ${JSON.stringify(result)}`);
}

runBenchmark()
  .then(finish)
  .catch((error) => {
    const result = {
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
