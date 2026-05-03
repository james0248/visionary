import * as ort from '/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs';
import { NormalNoiseGenerator } from './jax_noise.js';

const ASSET_DIR = '/webgpu_app/assets';
const MANIFEST_URL = `${ASSET_DIR}/breakout_onnx_manifest.json`;
const CONTEXT_URL = `${ASSET_DIR}/breakout_demo_context.json`;

ort.env.wasm ??= {};
ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';

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
const CACHE_SHAPE = [6, 1, 36, 64, 2, 64];
const float32Scratch = new Float32Array(1);
const uint32Scratch = new Uint32Array(float32Scratch.buffer);

const elements = {
  canvas: document.getElementById('frame'),
  status: document.getElementById('status'),
  start: document.getElementById('start'),
  reset: document.getElementById('reset'),
  fps: document.getElementById('fps'),
  action: document.getElementById('action'),
  frameCount: document.getElementById('frame-count'),
  latency: document.getElementById('latency'),
  context: document.getElementById('context'),
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

function setStatus(message) {
  elements.status.textContent = message;
}

function dtypeArray(dtype) {
  if (dtype === 'float32') return Float32Array;
  if (dtype === 'float16') return Uint16Array;
  if (dtype === 'int32') return Int32Array;
  throw new Error(`Unsupported artifact dtype ${dtype}`);
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

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  return response.json();
}

async function fetchTensorFromArtifact(baseUrl, spec) {
  const response = await fetch(`${baseUrl}/${spec.path}`);
  if (!response.ok) throw new Error(`Failed to fetch ${spec.path}: ${response.status}`);
  const buffer = await response.arrayBuffer();
  const ArrayType = dtypeArray(spec.dtype);
  return new ort.Tensor(spec.dtype, new ArrayType(buffer), spec.shape);
}

function findExport(manifest, name) {
  const entry = manifest.exports.find((item) => item.name === name);
  if (!entry) throw new Error(`Missing export ${name}`);
  return entry;
}

function findFirstExport(manifest, names) {
  for (const name of names) {
    const entry = manifest.exports.find((item) => item.name === name);
    if (entry) return entry;
  }
  throw new Error(`Missing exports ${names.join(', ')}`);
}

function outputName(spec, preferred) {
  if (spec.outputs?.[preferred]) return preferred;
  return Object.keys(spec.outputs ?? {})[0];
}

function optionalOutputName(spec, preferred) {
  return spec.outputs?.[preferred] ? preferred : null;
}

async function createSession(spec, options = {}) {
  const executionProviders = options.executionProviders ?? [{ name: 'webgpu' }];
  const sessionOptions = { ...options };
  delete sessionOptions.executionProviders;
  return ort.InferenceSession.create(`${ASSET_DIR}/${spec.path}`, {
    executionProviders,
    graphOptimizationLevel: 'all',
    externalData: (spec.external_data ?? []).map((entry) => ({
      path: entry.path,
      data: `${ASSET_DIR}/${entry.path}`,
    })),
    ...sessionOptions,
  });
}

function randomNormalTensor(shape, dtype = 'float32') {
  const size = shape.reduce((total, value) => total * value, 1);
  return makeFloatTensor(dtype, noiseGenerator.tensorData(size), shape);
}

function zeroTensor(dtype, shape) {
  const size = shape.reduce((total, value) => total * value, 1);
  const ArrayType = dtypeArray(dtype);
  return new ort.Tensor(dtype, new ArrayType(size), shape);
}

function scalarTensor(value, shape = [1, 1]) {
  return new ort.Tensor('int32', new Int32Array([value]), shape);
}

function contextFrameTensor(tensor, frameIndex, dtype = 'float32') {
  const start = frameIndex * CONTEXT_TENSOR_SIZE;
  const end = start + CONTEXT_TENSOR_SIZE;
  return makeFloatTensor(dtype, tensor.data.slice(start, end), [1, 1, 32, 32]);
}

function contextScalarTensor(tensor, frameIndex) {
  return new ort.Tensor('int32', new Int32Array([tensor.data[frameIndex]]), [1, 1]);
}

function disposeGpuTensor(tensor) {
  if (tensor?.location === 'gpu-buffer') {
    tensor.dispose();
  }
}

function patchesToImageData(patchesTensor, preprocessor) {
  const patches = patchesTensor.data;
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
          const source = patchOffset + ((iy * patchSize + ix) * channels);
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

function tensorStats(tensor) {
  const values = tensor.data;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let finite = 0;
  let nonzero = 0;

  for (let index = 0; index < values.length; index += 1) {
    const value = floatTensorValue(tensor, index);
    if (!Number.isFinite(value)) continue;
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    finite += 1;
    if (value !== 0) nonzero += 1;
  }

  return {
    type: tensor.type,
    dims: tensor.dims,
    min,
    max,
    mean: finite ? sum / finite : Number.NaN,
    finite,
    nonzero,
  };
}

function canvasStats() {
  const data = ctx.getImageData(0, 0, elements.canvas.width, elements.canvas.height).data;
  let min = 255;
  let max = 0;
  let sum = 0;
  let nonzero = 0;
  for (let index = 0; index < data.length; index += 4) {
    const value = data[index] + data[index + 1] + data[index + 2];
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    if (value !== 0) nonzero += 1;
  }
  const pixels = data.length / 4;
  return { min, max, mean: sum / pixels, nonzero };
}

async function renderLatent(zTensor) {
  const decoderOutputs = await runtime.sessions.decoder.run(
    {
      z: zTensor,
    },
    [runtime.names.patches],
  );
  const patches = decoderOutputs[runtime.names.patches];
  const image = patchesToImageData(patches, runtime.preprocessor);
  ctx.putImageData(image, 0, 0);
  return {
    patches: tensorStats(patches),
    canvas: canvasStats(),
  };
}

function updateActionUi() {
  const label = ACTION_LABELS[currentAction];
  elements.action.textContent = label;
  for (const [name, element] of Object.entries(elements.keys)) {
    element.classList.toggle('active', name === label);
  }
}

function actionFromKeys(event, pressed) {
  if (event.code === 'ArrowLeft') {
    currentAction = pressed ? ACTIONS.left : ACTIONS.noop;
  } else if (event.code === 'ArrowRight') {
    currentAction = pressed ? ACTIONS.right : ACTIONS.noop;
  } else if (event.code === 'Space' || event.code === 'ArrowUp') {
    event.preventDefault();
    currentAction = pressed ? ACTIONS.fire : ACTIONS.noop;
  } else {
    return;
  }
  updateActionUi();
}

async function loadRuntime() {
  setStatus('Loading manifest');
  const [manifest, contextManifest] = await Promise.all([
    fetchJson(MANIFEST_URL),
    fetchJson(CONTEXT_URL),
  ]);
  const prefixStepSpec = findExport(manifest, 'breakout_dynamics_step_cached_b1_t1');
  const sampleStepSpec = findExport(manifest, 'breakout_dynamics_sample_append_context_b1_t1_s4');
  const sampleStepSlideSpec = findFirstExport(manifest, [
    'breakout_dynamics_sample_append_context_slide_full_cache_b1_t1_s4',
    'breakout_dynamics_sample_append_context_slide_b1_t1_s4',
  ]);
  const decoderSpec = findExport(manifest, 'breakout_tokenizer_decode_z_b1_t1');

  setStatus('Loading context');
  const [contextZ, displayZ, contextActions, stepLevels, signalLevels] = await Promise.all([
    fetchTensorFromArtifact(ASSET_DIR, contextManifest.arrays.z),
    contextManifest.arrays.display_z
      ? fetchTensorFromArtifact(ASSET_DIR, contextManifest.arrays.display_z)
      : Promise.resolve(null),
    fetchTensorFromArtifact(ASSET_DIR, contextManifest.arrays.actions),
    fetchTensorFromArtifact(ASSET_DIR, contextManifest.arrays.step_levels),
    fetchTensorFromArtifact(ASSET_DIR, contextManifest.arrays.signal_levels),
  ]);
  elements.context.textContent =
    `${contextManifest.prefix_frames} frames @ ${contextManifest.episode_start}`;

  setStatus('Creating prefix session');
  const prefixStepSession = await createSession(prefixStepSpec, {
    preferredOutputLocation: {
      pred_z: 'cpu',
      candidate_k_cache: 'gpu-buffer',
      candidate_v_cache: 'gpu-buffer',
      candidate_cache_length: 'cpu',
    },
  });
  setStatus('Creating sampling session');
  const sampleStepSession = await createSession(sampleStepSpec, {
    preferredOutputLocation: {
      final_z: 'gpu-buffer',
      candidate_k_cache: 'gpu-buffer',
      candidate_v_cache: 'gpu-buffer',
      candidate_cache_length: 'cpu',
    },
  });
  setStatus('Creating steady sampling session');
  const sampleSlideOutputLocation = {
    final_z: 'gpu-buffer',
    candidate_k_cache: 'gpu-buffer',
    candidate_v_cache: 'gpu-buffer',
  };
  if (sampleStepSlideSpec.outputs?.candidate_cache_length) {
    sampleSlideOutputLocation.candidate_cache_length = 'cpu';
  }
  const sampleStepSlideSession = await createSession(sampleStepSlideSpec, {
    preferredOutputLocation: sampleSlideOutputLocation,
  });
  setStatus('Creating decoder session');
  const decoderSession = await createSession(decoderSpec, {
    preferredOutputLocation: {
      patches: 'cpu',
    },
  });

  return {
    contextManifest,
    preprocessor: contextManifest.preprocessor,
    sessions: {
      prefixStep: prefixStepSession,
      sampleStep: sampleStepSession,
      sampleStepSlide: sampleStepSlideSession,
      decoder: decoderSession,
    },
    specs: {
      prefixStep: prefixStepSpec,
      sampleStep: sampleStepSpec,
      sampleStepSlide: sampleStepSlideSpec,
      decoder: decoderSpec,
    },
    names: {
      prefixK: outputName(prefixStepSpec, 'candidate_k_cache'),
      prefixV: outputName(prefixStepSpec, 'candidate_v_cache'),
      prefixCacheLength: outputName(prefixStepSpec, 'candidate_cache_length'),
      finalZ: outputName(sampleStepSpec, 'final_z'),
      sampleK: outputName(sampleStepSpec, 'candidate_k_cache'),
      sampleV: outputName(sampleStepSpec, 'candidate_v_cache'),
      sampleCacheLength: optionalOutputName(sampleStepSpec, 'candidate_cache_length'),
      sampleSlideK: outputName(sampleStepSlideSpec, 'candidate_k_cache'),
      sampleSlideV: outputName(sampleStepSlideSpec, 'candidate_v_cache'),
      sampleSlideCacheLength: optionalOutputName(sampleStepSlideSpec, 'candidate_cache_length'),
      patches: outputName(decoderSpec, 'patches'),
    },
    context: {
      z: contextZ,
      displayZ,
      actions: contextActions,
      stepLevels,
      signalLevels,
    },
    dtypes: {
      prefixZ: prefixStepSpec.inputs.z.dtype,
      cache: prefixStepSpec.inputs.k_cache.dtype,
      sampleNoise: sampleStepSpec.inputs.sample_noise.dtype,
    },
    cache: null,
  };
}

async function prefill() {
  const prefixFrames = runtime.contextManifest.prefix_frames ?? runtime.contextManifest.context_length;
  const prefixSlotStart =
    runtime.contextManifest.prefix_slot_start ??
    Math.max(0, runtime.contextManifest.context_length - prefixFrames);
  setStatus(`Prefilling ${prefixFrames} prefix frames`);
  disposeGpuTensor(runtime.cache?.k);
  disposeGpuTensor(runtime.cache?.v);
  runtime.cache = {
    k: zeroTensor(runtime.dtypes.cache, CACHE_SHAPE),
    v: zeroTensor(runtime.dtypes.cache, CACHE_SHAPE),
    length: scalarTensor(0, [1]),
  };
  for (let offset = 0; offset < prefixFrames; offset += 1) {
    const index = prefixSlotStart + offset;
    const outputs = await runtime.sessions.prefixStep.run(
      {
        z: contextFrameTensor(runtime.context.z, index, runtime.dtypes.prefixZ),
        actions: contextScalarTensor(runtime.context.actions, index),
        step_levels: contextScalarTensor(runtime.context.stepLevels, index),
        signal_levels: contextScalarTensor(runtime.context.signalLevels, index),
        position_index: runtime.cache.length,
        k_cache: runtime.cache.k,
        v_cache: runtime.cache.v,
        cache_length: runtime.cache.length,
      },
      [runtime.names.prefixK, runtime.names.prefixV, runtime.names.prefixCacheLength],
    );
    const oldCache = runtime.cache;
    runtime.cache = {
      k: outputs[runtime.names.prefixK],
      v: outputs[runtime.names.prefixV],
      length: outputs[runtime.names.prefixCacheLength],
    };
    disposeGpuTensor(oldCache.k);
    disposeGpuTensor(oldCache.v);
  }
  frameCount = 0;
  noiseGenerator = new NormalNoiseGenerator(runtime.contextManifest.noise_seed ?? 0);
  elements.frameCount.textContent = '0';
  const previewTensor = runtime.context.displayZ
    ? contextFrameTensor(runtime.context.displayZ, prefixFrames - 1, runtime.dtypes.prefixZ)
    : contextFrameTensor(runtime.context.z, prefixSlotStart + prefixFrames - 1, runtime.dtypes.prefixZ);
  const preview = await renderLatent(previewTensor);
  runtime.lastPreviewStats = preview;
  setStatus(`Ready with cache length ${runtime.cache.length.data[0]}`);
}

async function generateFrame() {
  const started = performance.now();
  const action = new ort.Tensor('int32', new Int32Array([currentAction]), [1, 1]);
  const sampleNoise = randomNormalTensor([1, 1, 32, 32], runtime.dtypes.sampleNoise);
  const contextNoise = randomNormalTensor([1, 1, 32, 32], runtime.dtypes.sampleNoise);
  const sampleSession =
    runtime.cache.length.data[0] >= CACHE_SHAPE[3]
      ? runtime.sessions.sampleStepSlide
      : runtime.sessions.sampleStep;
  const sampleNames =
    runtime.cache.length.data[0] >= CACHE_SHAPE[3]
      ? {
          k: runtime.names.sampleSlideK,
          v: runtime.names.sampleSlideV,
          length: runtime.names.sampleSlideCacheLength,
        }
      : {
          k: runtime.names.sampleK,
          v: runtime.names.sampleV,
          length: runtime.names.sampleCacheLength,
        };
  const fetches = [runtime.names.finalZ, sampleNames.k, sampleNames.v];
  if (sampleNames.length) fetches.push(sampleNames.length);
  const sampleSpec =
    sampleSession === runtime.sessions.sampleStepSlide
      ? runtime.specs.sampleStepSlide
      : runtime.specs.sampleStep;
  const feeds = {
    sample_noise: sampleNoise,
    context_noise: contextNoise,
    actions: action,
    k_cache: runtime.cache.k,
    v_cache: runtime.cache.v,
  };
  if (sampleSpec.inputs?.position_index) feeds.position_index = runtime.cache.length;
  if (sampleSpec.inputs?.cache_length) feeds.cache_length = runtime.cache.length;
  const outputs = await sampleSession.run(feeds, fetches);
  const oldCache = runtime.cache;
  runtime.cache = {
    k: outputs[sampleNames.k],
    v: outputs[sampleNames.v],
    length: sampleNames.length ? outputs[sampleNames.length] : oldCache.length,
  };
  disposeGpuTensor(oldCache.k);
  disposeGpuTensor(oldCache.v);

  const decoderOutputs = await runtime.sessions.decoder.run({ z: outputs[runtime.names.finalZ] }, [
    runtime.names.patches,
  ]);
  disposeGpuTensor(outputs[runtime.names.finalZ]);

  const image = patchesToImageData(decoderOutputs[runtime.names.patches], runtime.preprocessor);
  ctx.putImageData(image, 0, 0);
  runtime.lastFrameStats = {
    patches: tensorStats(decoderOutputs[runtime.names.patches]),
    canvas: canvasStats(),
  };

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
  try {
    await generateFrame();
    requestAnimationFrame(streamLoop);
  } catch (error) {
    running = false;
    elements.start.textContent = 'Start';
    setStatus(error instanceof Error ? error.message : String(error));
    throw error;
  }
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

elements.reset.addEventListener('click', async () => {
  running = false;
  elements.start.textContent = 'Start';
  await prefill();
});

window.addEventListener('keydown', (event) => actionFromKeys(event, true));
window.addEventListener('keyup', (event) => actionFromKeys(event, false));

updateActionUi();
elements.start.disabled = true;
elements.reset.disabled = true;

try {
  runtime = await loadRuntime();
  await prefill();
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
  canvasStats,
  tensorStats,
  async renderContext(frameIndex = runtime.contextManifest.context_length - 1) {
    return renderLatent(contextFrameTensor(runtime.context.z, frameIndex, runtime.dtypes.prefixZ));
  },
  async generateFrame() {
    await generateFrame();
    return runtime.lastFrameStats;
  },
};
