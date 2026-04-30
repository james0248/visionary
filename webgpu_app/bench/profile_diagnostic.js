const ASSET_DIR = '/webgpu_app/assets';
const MANIFEST_URL = `${ASSET_DIR}/breakout_onnx_manifest.json`;
const DEFAULT_CONFIG = {
  importKind: 'dist_bundle',
  runs: 1,
  modelName: 'breakout_dynamics_step_cached_b1_t1',
  drainMs: 500,
  requireHardwareGpu: true,
  sessionProfiling: false,
};
const IMPORTS = {
  dist_bundle: {
    label: 'dist/ort.webgpu.bundle.min.mjs',
    url: '/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs',
    package_export: 'onnxruntime-web/webgpu default',
  },
  dist_external_wasm: {
    label: 'dist/ort.webgpu.min.mjs',
    url: '/node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs',
    package_export: 'onnxruntime-web/webgpu onnxruntime-web-use-extern-wasm',
  },
  dist_unminified: {
    label: 'dist/ort.webgpu.mjs',
    url: '/node_modules/onnxruntime-web/dist/ort.webgpu.mjs',
    package_export: 'debug dist file',
  },
};

function setStatus(message) {
  document.getElementById('status').textContent = message;
  console.log(`WEBGPU_PROFILE_DIAGNOSTIC_STATUS ${message}`);
}

function parseConfig() {
  const params = new URLSearchParams(window.location.search);
  return {
    importKind: params.get('importKind') ?? DEFAULT_CONFIG.importKind,
    runs: Number(params.get('runs') ?? DEFAULT_CONFIG.runs),
    modelName: params.get('modelName') ?? DEFAULT_CONFIG.modelName,
    drainMs: Number(params.get('drainMs') ?? DEFAULT_CONFIG.drainMs),
    requireHardwareGpu:
      (params.get('requireHardwareGpu') ?? String(DEFAULT_CONFIG.requireHardwareGpu)) === 'true',
    sessionProfiling:
      (params.get('sessionProfiling') ?? String(DEFAULT_CONFIG.sessionProfiling)) === 'true',
  };
}

function installConsoleCapture() {
  const captured = [];
  for (const method of ['log', 'info', 'warn', 'error']) {
    const original = console[method].bind(console);
    console[method] = (...args) => {
      const text = args
        .map((arg) => {
          if (typeof arg === 'string') return arg;
          try {
            return JSON.stringify(arg);
          } catch {
            return String(arg);
          }
        })
        .join(' ');
      captured.push({
        method,
        text,
        at_ms: performance.now(),
      });
      original(...args);
    };
  }
  return captured;
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
  if (exponent === 0xff) return sign | (mantissa ? 0x7e00 : 0x7c00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const shifted = (mantissa | 0x800000) >>> (1 - halfExponent);
    return sign | ((shifted + 0x1000) >>> 13);
  }
  return sign | (halfExponent << 10) | ((mantissa + 0x1000) >>> 13);
}

function makeFloatTensor(ort, shape, seed, dtype = 'float32') {
  const random = makePrng(seed);
  const values = dtype === 'float16' ? new Uint16Array(mul(shape)) : new Float32Array(mul(shape));
  for (let i = 0; i < values.length; i += 1) {
    const value = random() * 2 - 1;
    values[i] = dtype === 'float16' ? float32ToFloat16Bits(value) : value;
  }
  return new ort.Tensor(dtype, values, shape);
}

function makeIntTensor(ort, shape, seed, maxExclusive) {
  const random = makePrng(seed);
  const values = new Int32Array(mul(shape));
  for (let i = 0; i < values.length; i += 1) {
    values[i] = Math.floor(random() * maxExclusive);
  }
  return new ort.Tensor('int32', values, shape);
}

function makeScalarFillTensor(ort, dtype, shape, value) {
  const values =
    dtype === 'float32'
      ? new Float32Array(mul(shape)).fill(value)
      : dtype === 'float16'
        ? new Uint16Array(mul(shape)).fill(float32ToFloat16Bits(value))
      : new Int32Array(mul(shape)).fill(value);
  return new ort.Tensor(dtype, values, shape);
}

function makeFeedForInput(ort, name, spec, seed) {
  if (spec.dtype === 'float32' || spec.dtype === 'float16') {
    return makeFloatTensor(ort, spec.shape, seed, spec.dtype);
  }
  if (name.includes('step_level')) {
    return makeScalarFillTensor(ort, 'int32', spec.shape, 2);
  }
  if (name.includes('signal_level')) {
    return makeScalarFillTensor(ort, 'int32', spec.shape, 0);
  }
  if (name.includes('cache_length')) {
    return makeScalarFillTensor(ort, 'int32', spec.shape, 64);
  }
  if (name.includes('position_index')) {
    return makeScalarFillTensor(ort, 'int32', spec.shape, 64);
  }
  return makeIntTensor(ort, spec.shape, seed, 4);
}

function makeFeedsFromSpec(ort, spec, seedBase) {
  const feeds = {};
  let index = 0;
  for (const [name, inputSpec] of Object.entries(spec.inputs ?? {})) {
    feeds[name] = makeFeedForInput(ort, name, inputSpec, seedBase + index * 17);
    index += 1;
  }
  return feeds;
}

function externalDataForSpec(spec) {
  return (spec.external_data ?? []).map((entry) => ({
    path: entry.path,
    data: `${ASSET_DIR}/${entry.path}`,
  }));
}

function findProfileSpec(manifest, modelName) {
  return (manifest.exports ?? []).find((entry) => entry.name === modelName);
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function gpuInfo() {
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

async function timed(label, fn) {
  const start = performance.now();
  const value = await fn();
  return {
    label,
    elapsed_ms: performance.now() - start,
    value,
  };
}

function outputSummary(outputs) {
  return Object.fromEntries(
    Object.entries(outputs).map(([name, tensor]) => [
      name,
      {
        type: tensor.type,
        dims: tensor.dims,
      },
    ]),
  );
}

async function runTrial({ ort, session, feeds, drainMs, mode }) {
  const callbackEvents = [];
  ort.env.webgpu ??= {};
  if (mode === 'callback') {
    ort.env.webgpu.profilingMode = 'default';
    ort.env.webgpu.profiling = {
      mode: 'default',
      ondata: (event) => callbackEvents.push(event),
    };
  } else {
    ort.env.webgpu.profilingMode = 'default';
    ort.env.webgpu.profiling = { mode: 'default' };
  }
  const run = await timed(mode, () => session.run(feeds));
  await delay(drainMs);
  return {
    mode,
    elapsed_ms: run.elapsed_ms,
    callback_event_count: callbackEvents.length,
    callback_events: callbackEvents,
    outputs: outputSummary(run.value),
  };
}

async function runSessionProfilingTrial({ ort, spec, feeds, provider }) {
  ort.env.webgpu ??= {};
  ort.env.webgpu.profilingMode = 'off';
  ort.env.webgpu.profiling = { mode: 'off' };
  const modelUrl = `${ASSET_DIR}/${spec.path}`;
  const create = await timed(`create_${provider}_profile_session`, () =>
    ort.InferenceSession.create(modelUrl, {
      executionProviders: [{ name: provider }],
      externalData: externalDataForSpec(spec),
      graphOptimizationLevel: 'all',
      enableProfiling: true,
      profileFilePrefix: `visionary_${provider}_profile`,
    }),
  );
  const run = await timed(`${provider}_profile_run`, () => create.value.run(feeds));
  const beforeEnd = performance.now();
  const endResult = create.value.endProfiling();
  return {
    provider,
    session_create_ms: create.elapsed_ms,
    run_ms: run.elapsed_ms,
    end_profiling_elapsed_ms: performance.now() - beforeEnd,
    end_profiling_return: endResult ?? null,
    outputs: outputSummary(run.value),
  };
}

async function runDiagnostic() {
  const config = parseConfig();
  const consoleMessages = installConsoleCapture();
  const importSpec = IMPORTS[config.importKind];
  if (!importSpec) {
    throw new Error(`Unknown importKind ${config.importKind}; expected one of ${Object.keys(IMPORTS)}`);
  }

  setStatus(`importing ${importSpec.label}`);
  const ort = await import(importSpec.url);
  ort.env.wasm ??= {};
  ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';

  setStatus('checking gpu');
  const gpu = await gpuInfo();
  if (config.requireHardwareGpu && isSoftwareGpu(gpu)) {
    throw new Error(`WebGPU is using a software adapter: ${JSON.stringify(gpu)}`);
  }

  setStatus('fetching manifest');
  const manifest = await fetch(MANIFEST_URL).then((response) => {
    if (!response.ok) throw new Error(`Failed to fetch manifest: ${response.status}`);
    return response.json();
  });
  const spec = findProfileSpec(manifest, config.modelName);
  if (!spec) {
    return {
      status: 'blocked',
      reason: `missing ${config.modelName}`,
      config,
      import: importSpec,
      gpu,
      timestamp_query: hasTimestampQuery(gpu),
      ort_version: ort.version ?? ort.env?.versions?.web ?? null,
    };
  }

  const modelUrl = `${ASSET_DIR}/${spec.path}`;
  setStatus(`creating session ${modelUrl}`);
  const create = await timed('create_session', () =>
    ort.InferenceSession.create(modelUrl, {
      executionProviders: [{ name: 'webgpu' }],
      externalData: externalDataForSpec(spec),
      graphOptimizationLevel: 'all',
    }),
  );
  const feeds = makeFeedsFromSpec(ort, spec, 9000);

  setStatus('running callback profiling trial');
  let callbackTrial = null;
  for (let i = 0; i < config.runs; i += 1) {
    callbackTrial = await runTrial({
      ort,
      session: create.value,
      feeds,
      drainMs: config.drainMs,
      mode: 'callback',
    });
  }

  setStatus('running console profiling trial');
  const beforeConsole = consoleMessages.length;
  let consoleTrial = null;
  for (let i = 0; i < config.runs; i += 1) {
    consoleTrial = await runTrial({
      ort,
      session: create.value,
      feeds,
      drainMs: config.drainMs,
      mode: 'console',
    });
  }
  const consoleProfileMessages = consoleMessages
    .slice(beforeConsole)
    .filter((entry) => entry.text.includes('[profiling]'));
  const sessionProfilingTrials = [];
  let sessionProfilingConsoleStart = consoleMessages.length;
  if (config.sessionProfiling) {
    setStatus('running webgpu session profiling trial');
    sessionProfilingTrials.push(
      await runSessionProfilingTrial({
        ort,
        spec,
        feeds,
        provider: 'webgpu',
      }),
    );
    setStatus('running wasm session profiling trial');
    sessionProfilingTrials.push(
      await runSessionProfilingTrial({
        ort,
        spec,
        feeds,
        provider: 'wasm',
      }),
    );
  }
  const sessionProfilingConsoleMessages = consoleMessages.slice(sessionProfilingConsoleStart);

  return {
    schema_version: 1,
    status: 'passed',
    created_at: new Date().toISOString(),
    user_agent: navigator.userAgent,
    platform: navigator.platform,
    config,
    import: importSpec,
    ort_version: ort.version ?? ort.env?.versions?.web ?? null,
    gpu,
    timestamp_query: hasTimestampQuery(gpu),
    model: {
      name: spec.name,
      path: spec.path,
      inputs: spec.inputs,
      outputs: spec.outputs,
    },
    timings: {
      session_create_ms: create.elapsed_ms,
    },
    trials: [callbackTrial, consoleTrial],
    console_profile_messages: consoleProfileMessages,
    session_profiling_trials: sessionProfilingTrials,
    session_profiling_console_messages: sessionProfilingConsoleMessages,
    conclusion: {
      callback_events: callbackTrial.callback_event_count,
      console_profile_messages: consoleProfileMessages.length,
      session_profiling_console_messages: sessionProfilingConsoleMessages.length,
      profiling_visible:
        callbackTrial.callback_event_count > 0 ||
        consoleProfileMessages.length > 0 ||
        sessionProfilingConsoleMessages.length > 0,
    },
  };
}

function finish(result) {
  window.__WEBGPU_PROFILE_DIAGNOSTIC_RESULT__ = result;
  document.getElementById('status').textContent = JSON.stringify(result, null, 2);
  console.log(`WEBGPU_PROFILE_DIAGNOSTIC_RESULT ${JSON.stringify(result)}`);
}

runDiagnostic()
  .then(finish)
  .catch((error) => {
    finish({
      schema_version: 1,
      status: 'failed',
      created_at: new Date().toISOString(),
      message: error.message,
      stack: error.stack,
      config: parseConfig(),
    });
  });
