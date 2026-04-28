import * as ort from '/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs';

const ASSET_DIR = '/webgpu_app/assets';
const MANIFEST_URL = `${ASSET_DIR}/breakout_onnx_manifest.json`;
const GENERATED_FRAMES = 8;
const SAMPLE_STEPS = 4;
const SAMPLE_STEP_LEVEL = 2;
const CONTEXT_STEP_LEVEL = 5;
const CONTEXT_TAU_EFFECTIVE = 29 / 32;
const DEFAULT_CONFIG = {
  mode: 'streaming',
  warmupRuns: 1,
  timedRuns: GENERATED_FRAMES,
  requireHardwareGpu: true,
};
const REQUIRED_ARTIFACTS = {
  prefill: ['breakout_dynamics_prefill_cached_b1_t64'],
  step: ['breakout_dynamics_step_cached_b1_t1'],
  decoder: ['breakout_tokenizer_decoder_b1_t1', 'breakout_decoder_b1_t1'],
};

function setStatus(message) {
  document.getElementById('status').textContent = message;
  console.log(`WEBGPU_BENCHMARK_STATUS ${message}`);
}

function parseConfig() {
  const params = new URLSearchParams(window.location.search);
  return {
    mode: params.get('mode') ?? DEFAULT_CONFIG.mode,
    warmupRuns: Number(params.get('warmupRuns') ?? DEFAULT_CONFIG.warmupRuns),
    timedRuns: Number(params.get('timedRuns') ?? DEFAULT_CONFIG.timedRuns),
    requireHardwareGpu:
      (params.get('requireHardwareGpu') ?? String(DEFAULT_CONFIG.requireHardwareGpu)) === 'true',
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

function makeFloatTensor(shape, seed) {
  const random = makePrng(seed);
  const values = new Float32Array(mul(shape));
  for (let i = 0; i < values.length; i += 1) {
    values[i] = random() * 2 - 1;
  }
  return new ort.Tensor('float32', values, shape);
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
    dtype === 'float32' ? new Float32Array(length).fill(value) : new Int32Array(length).fill(value);
  return new ort.Tensor(dtype, values, shape);
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
  for (const value of values) {
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

async function createSession(modelUrl, externalData = []) {
  setStatus(`creating session ${modelUrl}`);
  return timeAsync(() =>
    ort.InferenceSession.create(modelUrl, {
      executionProviders: [{ name: 'webgpu' }],
      externalData,
      graphOptimizationLevel: 'all',
    }),
  );
}

async function gpuInfo() {
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

function byExportName(manifest) {
  return Object.fromEntries((manifest.exports ?? []).map((entry) => [entry.name, entry]));
}

function findSpec(exportsByName, names) {
  for (const name of names) {
    if (exportsByName[name]) return exportsByName[name];
  }
  return null;
}

function resolveDemoSpecs(manifest) {
  const exportsByName = byExportName(manifest);
  return {
    prefill: findSpec(exportsByName, REQUIRED_ARTIFACTS.prefill),
    step: findSpec(exportsByName, REQUIRED_ARTIFACTS.step),
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
  if (spec.dtype !== 'float32' && spec.dtype !== 'int32') {
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
  requireTensorSpec('prefill.outputs.k_cache', specs.prefill.outputs?.k_cache, cacheShape);
  requireTensorSpec('prefill.outputs.v_cache', specs.prefill.outputs?.v_cache, cacheShape);
  requireTensorSpec('prefill.outputs.cache_length', specs.prefill.outputs?.cache_length, cacheLengthShape);
  requireTensorSpec('step.inputs.k_cache', specs.step.inputs?.k_cache, cacheShape);
  requireTensorSpec('step.inputs.v_cache', specs.step.inputs?.v_cache, cacheShape);
  requireTensorSpec('step.inputs.cache_length', specs.step.inputs?.cache_length, cacheLengthShape);
  requireTensorSpec('step.outputs.candidate_k_cache', specs.step.outputs?.candidate_k_cache, cacheShape);
  requireTensorSpec('step.outputs.candidate_v_cache', specs.step.outputs?.candidate_v_cache, cacheShape);
  requireTensorSpec(
    'step.outputs.candidate_cache_length',
    specs.step.outputs?.candidate_cache_length,
    cacheLengthShape,
  );
  requireTensorSpec('step.outputs.pred_z', specs.step.outputs?.pred_z, [1, 1, 32, 32]);
  requireTensorSpec('decoder.inputs.latent', specs.decoder.inputs?.latent, [1, 1, 64, 16]);
  if (specs.prefill.name.includes('b1_t64') && !specs.prefill.name.includes('cached')) {
    throw new Error(`Prefill artifact is not cached: ${specs.prefill.name}`);
  }
  if (specs.step.name.includes('b1_t64') || !specs.step.name.includes('cached')) {
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
    cache_length: tensors.cache_length ?? null,
  };
}

function samplingConfig() {
  return {
    sample_steps: SAMPLE_STEPS,
    sample_step_level: SAMPLE_STEP_LEVEL,
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
      executionProviders: [{ name: 'webgpu' }],
      graphOptimizationLevel: 'all',
    },
    gpu,
    sampling: samplingConfig(),
    cache_abi: cacheAbi(manifest),
    manifest: compactManifest(manifest),
    results: [],
  };
}

function makeFeedForInput(name, spec, seed) {
  const shape = spec.shape;
  const dtype = spec.dtype;
  if (dtype === 'float32') {
    if (name === 'cache_length') return makeScalarFillTensor('float32', shape, 64);
    return makeFloatTensor(shape, seed);
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
  return {
    k: outputs.find((name) => name === 'k_cache' || name.endsWith('_k_cache')),
    v: outputs.find((name) => name === 'v_cache' || name.endsWith('_v_cache')),
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

function decoderOutputName(spec) {
  const outputs = Object.keys(spec.outputs ?? {});
  return outputs.find((name) => name === 'patches') ?? outputs[0];
}

function applyCacheFeeds(feeds, cache) {
  const next = { ...feeds };
  for (const name of Object.keys(next)) {
    if (name === 'k_cache' || name.endsWith('_k_cache')) next[name] = cache.k;
    if (name === 'v_cache' || name.endsWith('_v_cache')) next[name] = cache.v;
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

function setStepInputs(feeds, { z, action, stepLevel, signalLevel, positionIndex }) {
  let next = replaceNamedFeed(feeds, ['z'], z);
  next = replaceNamedFeed(next, ['action'], action);
  next = replaceNamedFeed(next, ['step_level'], stepLevel);
  next = replaceNamedFeed(next, ['signal_level'], signalLevel);
  next = replaceNamedFeed(next, ['position_index'], positionIndex);
  return next;
}

function cacheFromOutputs(outputs, names) {
  if (!names.k || !names.v || !names.length) {
    throw new Error('Cached graph must output k_cache, v_cache, and cache_length');
  }
  return {
    k: outputs[names.k],
    v: outputs[names.v],
    length: outputs[names.length],
  };
}

function latentFromPredZ(predZ) {
  if (mul(predZ.dims) !== 1024) {
    throw new Error(`Cannot reshape pred_z ${predZ.dims.join('x')} to decoder latent [1,1,64,16]`);
  }
  const values = new Float32Array(predZ.data);
  return new ort.Tensor('float32', values, [1, 1, 64, 16]);
}

function nextSampleZ(currentZ, predZ, signalLevel) {
  const tau = signalLevel / SAMPLE_STEPS;
  const stepSize = 1 / SAMPLE_STEPS;
  const denom = Math.max(1 - tau, 1e-6);
  const values = new Float32Array(currentZ.data.length);
  for (let i = 0; i < values.length; i += 1) {
    const current = currentZ.data[i];
    const predicted = predZ.data[i];
    const velocity = (predicted - current) / denom;
    values[i] = current + velocity * stepSize;
  }
  return new ort.Tensor('float32', values, currentZ.dims);
}

function replaceDecoderLatent(feeds, latent) {
  const next = { ...feeds };
  const latentName = Object.keys(next).find((name) => name.includes('latent')) ?? Object.keys(next)[0];
  next[latentName] = latent;
  return next;
}

async function createBenchSession(role, spec) {
  const modelUrl = `${ASSET_DIR}/${spec.path}`;
  const modelFetch = await fetchSize(modelUrl);
  const sessionCreate = await createSession(modelUrl, externalDataForSpec(spec));
  return {
    role,
    spec,
    session: sessionCreate.value,
    model_url: modelUrl,
    model_fetch_ms: modelFetch.elapsed_ms,
    model_bytes: modelFetch.bytes,
    session_create_ms: sessionCreate.elapsedMs,
  };
}

async function runDemoBenchmark({ config, specs }) {
  const prefill = await createBenchSession('cached_prefill', specs.prefill);
  const step = await createBenchSession('cached_step', specs.step);
  const decoder = await createBenchSession('single_frame_decoder', specs.decoder);
  const prefillFeeds = makeFeedsFromSpec(specs.prefill, 1000);
  const stepBaseFeeds = makeFeedsFromSpec(specs.step, 2000);
  const decoderBaseFeeds = makeFeedsFromSpec(specs.decoder, 3000);
  const prefillCacheNames = cacheOutputNames(specs.prefill);
  const stepCacheNames = cacheOutputNames(specs.step);
  const predName = stepPredOutputName(specs.step);
  const decoderName = decoderOutputName(specs.decoder);
  const actionTensor = makeIntTensor([1, 1], 4000, 4);
  const stepLevelTensor = makeScalarFillTensor('int32', [1, 1], SAMPLE_STEP_LEVEL);

  setStatus('demo benchmark: first prefill');
  const prefillFirst = await timeAsync(() => prefill.session.run(prefillFeeds));
  let persistentCache = cacheFromOutputs(prefillFirst.value, prefillCacheNames);

  for (let i = 0; i < config.warmupRuns; i += 1) {
    setStatus(`demo benchmark: warmup frame ${i + 1}/${config.warmupRuns}`);
    let candidateCache = persistentCache;
    let currentZ = makeFloatTensor([1, 1, 32, 32], 5000 + i);
    let predZ = null;
    for (let sample = 0; sample < SAMPLE_STEPS; sample += 1) {
      const signalLevelTensor = makeScalarFillTensor('int32', [1, 1], sample);
      const positionTensor = makeScalarFillTensor('int32', [1], persistentCache.length.data[0]);
      const feeds = setStepInputs(applyCacheFeeds(stepBaseFeeds, persistentCache), {
        z: currentZ,
        action: actionTensor,
        stepLevel: stepLevelTensor,
        signalLevel: signalLevelTensor,
        positionIndex: positionTensor,
      });
      const outputs = await step.session.run(feeds);
      predZ = outputs[predName];
      candidateCache = cacheFromOutputs(outputs, stepCacheNames);
      currentZ = nextSampleZ(currentZ, predZ, sample);
    }
    const latent = latentFromPredZ(currentZ);
    await decoder.session.run(replaceDecoderLatent(decoderBaseFeeds, latent));
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
  const timedPrefill = await timeAsync(() => prefill.session.run(prefillFeeds));
  prefillSamples.push(timedPrefill.elapsedMs);
  persistentCache = cacheFromOutputs(timedPrefill.value, prefillCacheNames);

  for (let frame = 0; frame < config.timedRuns; frame += 1) {
    setStatus(`demo benchmark: generated frame ${frame + 1}/${config.timedRuns}`);
    const frameStart = performance.now();
    let candidateCache = persistentCache;
    let currentZ = makeFloatTensor([1, 1, 32, 32], 6000 + frame);
    let predZ = null;
    const frameForwardSamples = [];

    for (let sample = 0; sample < SAMPLE_STEPS; sample += 1) {
      const committedLengthBefore = persistentCache.length.data[0];
      const signalLevelTensor = makeScalarFillTensor('int32', [1, 1], sample);
      const positionTensor = makeScalarFillTensor('int32', [1], committedLengthBefore);
      const feeds = setStepInputs(applyCacheFeeds(stepBaseFeeds, persistentCache), {
        z: currentZ,
        action: actionTensor,
        stepLevel: stepLevelTensor,
        signalLevel: signalLevelTensor,
        positionIndex: positionTensor,
      });
      const timed = await timeAsync(() => step.session.run(feeds));
      predZ = timed.value[predName];
      if (!predZ) {
        throw new Error(`Cached step did not return output ${predName}`);
      }
      candidateCache = cacheFromOutputs(timed.value, stepCacheNames);
      currentZ = nextSampleZ(currentZ, predZ, sample);
      if (sample < SAMPLE_STEPS - 1 && persistentCache.length.data[0] !== committedLengthBefore) {
        throw new Error('Committed cache_length changed before the final sample step');
      }
      frameForwardSamples.push(timed.elapsedMs);
      targetForwardSamples.push(timed.elapsedMs);
    }

    const packTimed = timeSync(() => latentFromPredZ(currentZ));
    const decoderTimed = await timeAsync(() =>
      decoder.session.run(replaceDecoderLatent(decoderBaseFeeds, packTimed.value)),
    );
    const commitTimed = timeSync(() => {
      persistentCache = candidateCache;
    });

    const dynamicsFrameMs = frameForwardSamples.reduce((total, value) => total + value, 0);
    dynamicsFrameSamples.push(dynamicsFrameMs);
    decoderFrameSamples.push(decoderTimed.elapsedMs);
    cacheCommitSamples.push(commitTimed.elapsedMs);
    packUnpackSamples.push(packTimed.elapsedMs);
    streamingFrameSamples.push(performance.now() - frameStart);

    latestPredStats = tensorStats(predZ);
    const frameOutput = decoderTimed.value[decoderName];
    if (!frameOutput) {
      throw new Error(`Single-frame decoder did not return output ${decoderName}`);
    }
    latestFrameStats = tensorStats(frameOutput);
    assertDims('cached_step.pred_z', predZ.dims, specs.step.outputs[predName].shape);
    assertDims('single_frame_decoder.output', frameOutput.dims, specs.decoder.outputs[decoderName].shape);
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
      inputs: prefill.spec.inputs,
      outputs: prefill.spec.outputs,
      timing: {
        prefill: summarize(prefillSamples),
      },
      cache: {
        outputs: prefillCacheNames,
        k_cache: tensorStats(persistentCache.k),
        v_cache: tensorStats(persistentCache.v),
        cache_length: tensorStats(persistentCache.length),
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
      inputs: step.spec.inputs,
      outputs: step.spec.outputs,
      timing: {
        target_forward: summarize(targetForwardSamples),
        dynamics_frame: summarize(dynamicsFrameSamples),
      },
      cache: {
        outputs: stepCacheNames,
        commit_policy: 'discard sample forwards 1-3; commit sample forward 4 once per frame',
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
        commit_policy: 'cache_length advances once per generated frame',
      },
      output: latestFrameStats,
    },
  ];
}

async function runBenchmark() {
  const config = parseConfig();
  setStatus('starting demo benchmark');
  const manifest = await fetchManifest();
  const gpu = await gpuInfo();
  if (config.requireHardwareGpu && isSoftwareGpu(gpu)) {
    throw new Error(
      `WebGPU is using a software adapter instead of the hardware GPU: ${JSON.stringify(gpu)}`,
    );
  }

  const specs = resolveDemoSpecs(manifest);
  const missing = missingDemoArtifacts(specs);
  if (missing.length > 0 || manifest.cache_contract?.status === 'contract_only') {
    return blockedResult({ config, manifest, gpu, missing });
  }
  validateDemoSpecs(specs, manifest);

  const results = await runDemoBenchmark({ config, specs });
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
      executionProviders: [{ name: 'webgpu' }],
      graphOptimizationLevel: 'all',
    },
    gpu,
    sampling: samplingConfig(),
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
