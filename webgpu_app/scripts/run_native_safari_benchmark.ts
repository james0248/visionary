import { mkdir, writeFile } from 'node:fs/promises';

const DEFAULT_BASE_URL = 'http://127.0.0.1:4173';
const DEFAULT_DRIVER_PORT = 4444;
const DEFAULT_WARMUP_FRAMES = 32;
const DEFAULT_TIMED_FRAMES = 128;
const DEFAULT_VALIDATION_FRAMES = 64;
const DEFAULT_MIN_FPS = 40;
const DEFAULT_MAX_FRAME_P95_MS = 50;
const DEFAULT_MAX_FRAME_INTERVAL_MS = 100;
const RESULT_DIR = 'bench/results';
const RESULT_PATH = `${RESULT_DIR}/latest_safari.json`;
const LATEST_PATH = `${RESULT_DIR}/latest.json`;

type CliOptions = {
  baseUrl: string;
  driverPort: number;
  warmupFrames: number;
  timedFrames: number;
  validationFrames: number;
  minFps: number;
  maxFrameP95Ms: number;
  maxFrameIntervalMs: number;
  assetBase: string;
  ortModule: string;
  demoQuery: string | null;
};

function numberOption(value: string | undefined, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseArgs(argv: string[]): CliOptions {
  const args = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith('--')) continue;
    const [key, inlineValue] = arg.split('=', 2);
    const value = inlineValue ?? argv[index + 1];
    if (inlineValue == null) index += 1;
    args.set(key, value);
  }
  return {
    baseUrl: args.get('--base-url') ?? DEFAULT_BASE_URL,
    driverPort: numberOption(args.get('--driver-port'), DEFAULT_DRIVER_PORT),
    warmupFrames: numberOption(args.get('--warmup-runs'), DEFAULT_WARMUP_FRAMES),
    timedFrames: numberOption(args.get('--timed-runs'), DEFAULT_TIMED_FRAMES),
    validationFrames: numberOption(args.get('--validation-frames'), DEFAULT_VALIDATION_FRAMES),
    minFps: numberOption(args.get('--min-fps'), DEFAULT_MIN_FPS),
    maxFrameP95Ms: numberOption(args.get('--max-frame-p95-ms'), DEFAULT_MAX_FRAME_P95_MS),
    maxFrameIntervalMs: numberOption(
      args.get('--max-frame-interval-ms'),
      DEFAULT_MAX_FRAME_INTERVAL_MS,
    ),
    assetBase: args.get('--asset-base') ?? '/dream_arcade_assets/breakout_wasm_default_mha',
    ortModule:
      args.get('--ort-module') ?? '/node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs',
    demoQuery: args.get('--demo-query') ?? null,
  };
}

async function fetchJson(url: string, init: RequestInit = {}) {
  const response = await fetch(url, {
    ...init,
    headers: {
      'content-type': 'application/json',
      ...(init.headers ?? {}),
    },
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`${init.method ?? 'GET'} ${url} failed: ${response.status} ${text}`);
  }
  return text ? JSON.parse(text) : null;
}

async function waitForHttp(url: string, timeoutMs: number) {
  const started = Date.now();
  let lastError: unknown = null;
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch (error) {
      lastError = error;
    }
    await Bun.sleep(100);
  }
  throw new Error(`Timed out waiting for ${url}: ${lastError instanceof Error ? lastError.message : lastError}`);
}

async function ensureStaticServer(baseUrl: string) {
  try {
    const response = await fetch(`${baseUrl}/health`);
    if (response.ok) return null;
  } catch {
    // Start a local server below.
  }
  const parsed = new URL(baseUrl);
  const proc = Bun.spawn(
    [
      'bun',
      'scripts/serve_static.ts',
      '--host',
      parsed.hostname,
      '--port',
      parsed.port || '4173',
    ],
    { stdout: 'pipe', stderr: 'pipe' },
  );
  await waitForHttp(`${baseUrl}/health`, 120_000);
  return proc;
}

async function ensureSafariDriver(port: number) {
  const statusUrl = `http://127.0.0.1:${port}/status`;
  try {
    const response = await fetch(statusUrl);
    if (response.ok) return null;
  } catch {
    // Start safaridriver below.
  }
  const proc = Bun.spawn(['safaridriver', '-p', String(port)], { stdout: 'pipe', stderr: 'pipe' });
  await waitForHttp(statusUrl, 30_000);
  return proc;
}

function stopSafariAutomation() {
  if (process.platform !== 'darwin') return;
  try {
    Bun.spawnSync(['pkill', '-f', 'Safari --automation -ApplePersistenceIgnoreStateQuietly YES'], {
      stdout: 'ignore',
      stderr: 'ignore',
    });
  } catch {
    // Best-effort cleanup only.
  }
}

function demoUrl(options: CliOptions) {
  const params = new URLSearchParams();
  params.set('fps', '0');
  params.set('backend', 'wasm');
  params.set('assetBase', options.assetBase);
  params.set('ortModule', options.ortModule);
  if (options.demoQuery) {
    const extra = new URLSearchParams(options.demoQuery.replace(/^[?&]/, ''));
    for (const [key, value] of extra) params.set(key, value);
  }
  return `${options.baseUrl}/demo/index.html?${params.toString()}`;
}

function summarize(samples: number[]) {
  if (samples.length === 0) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  const sum = samples.reduce((total, value) => total + value, 0);
  const percentile = (p: number) =>
    sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
  return {
    count: samples.length,
    mean_ms: sum / samples.length,
    median_ms: percentile(0.5),
    p95_ms: percentile(0.95),
    min_ms: sorted[0],
    max_ms: sorted[sorted.length - 1],
  };
}

function timingFromWindow(window: any) {
  const stats = window.frameStats ?? [];
  const latencies = stats.map((entry: any) => entry.elapsedMs).filter(Number.isFinite);
  const intervals = [];
  for (let index = 1; index < stats.length; index += 1) {
    const interval = stats[index].completed - stats[index - 1].completed;
    if (Number.isFinite(interval)) intervals.push(interval);
  }
  const windowMsPerFrame = window.elapsedMs / Math.max(window.endFrame - window.startFrame, 1);
  const stageSamples = new Map<string, number[]>();
  for (const entry of stats) {
    for (const [name, value] of Object.entries(entry.stages ?? {})) {
      if (!Number.isFinite(value)) continue;
      const samples = stageSamples.get(name) ?? [];
      samples.push(value as number);
      stageSamples.set(name, samples);
    }
  }
  return {
    measured_frames: window.endFrame - window.startFrame,
    elapsed_ms: window.elapsedMs,
    window_ms_per_frame: windowMsPerFrame,
    window_fps: 1000 / windowMsPerFrame,
    frame_latency: summarize(latencies),
    frame_interval: summarize(intervals),
    stages: Object.fromEntries([...stageSamples].map(([name, samples]) => [name, summarize(samples)])),
  };
}

function benchmarkScript(config: CliOptions) {
  return `
const done = arguments[arguments.length - 1];
(async () => {
  const MIN_BRICK_COVERAGE = 0.45;
  const MIN_UNIQUE_FRAME_HASHES = 2;
  const MIN_UNIQUE_LATENT_HASHES = 2;
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const waitFor = async (fn, timeoutMs, label) => {
    const start = performance.now();
    while (performance.now() - start < timeoutMs) {
      const value = fn();
      if (value) return value;
      await sleep(100);
    }
    throw new Error('timeout waiting for ' + label);
  };
  const fnvHash = (bytes) => {
    let value = 2166136261 >>> 0;
    for (let index = 0; index < bytes.length; index += 1) {
      value ^= bytes[index];
      value = Math.imul(value, 16777619) >>> 0;
    }
    return value.toString(16).padStart(8, '0');
  };
  const visibleFrame = () =>
    document.querySelector('#frame:not([hidden]), .frame-preview:not([hidden]), .frame-fallback:not([hidden])');
  const pauseDemo = () => {
    const button = document.querySelector('#start');
    if (button?.textContent?.includes('Pause')) button.click();
  };
  const resetDemo = async () => {
    document.querySelector('#reset')?.click();
    await waitFor(
      () => window.visionaryDemoDebug?.frameCount === 0 && document.querySelector('#status')?.textContent?.includes('Ready'),
      30000,
      'reset',
    );
  };
  const runToFrame = async (targetFrame) => {
    const debug = window.visionaryDemoDebug;
    const button = document.querySelector('#start');
    const wait = debug.waitForFrameCount(targetFrame, 300000);
    if (!button.textContent?.includes('Pause')) button.click();
    await wait;
    pauseDemo();
  };
  const runWindow = async (frames) => {
    const debug = window.visionaryDemoDebug;
    const button = document.querySelector('#start');
    const startFrame = debug.frameCount;
    const targetFrame = startFrame + frames;
    const startMs = performance.now();
    const wait = debug.waitForFrameCount(targetFrame, 300000);
    if (!button.textContent?.includes('Pause')) button.click();
    await wait;
    const endMs = performance.now();
    pauseDemo();
    const frameStats = debug.frameStats
      .filter((entry) => entry.frame > startFrame && entry.frame <= targetFrame)
      .slice(-frames);
    return {
      startFrame,
      targetFrame,
      endFrame: debug.frameCount,
      startMs,
      endMs,
      elapsedMs: endMs - startMs,
      frameStats,
    };
  };
  const measureBrickCoverage = () => {
    const debug = window.visionaryDemoDebug;
    const runtime = debug.runtime;
    const expectedTensor = runtime.displayPixels;
    if (!expectedTensor) return { status: 'skipped', reason: 'display_pixels artifact is unavailable' };
    const [frames, height, width, channels] = expectedTensor.dims;
    const referenceFrame = Math.min(frames - 1, (runtime.contextManifest.prefix_frames ?? 1) - 1);
    const frameOffset = referenceFrame * height * width * channels;
    const sourceFrame = visibleFrame();
    if (!sourceFrame) return { status: 'failed', reason: 'visible frame is unavailable' };
    const scratch = document.createElement('canvas');
    scratch.width = width;
    scratch.height = height;
    const context = scratch.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('2D scratch context is unavailable');
    context.imageSmoothingEnabled = false;
    context.drawImage(sourceFrame, 0, 0, width, height);
    const actual = context.getImageData(0, 0, width, height).data;
    const expected = expectedTensor.data;
    const isBrickColor = (red, green, blue) => {
      const max = Math.max(red, green, blue);
      const min = Math.min(red, green, blue);
      return max > 50 && max - min > 18;
    };
    let expectedCount = 0;
    let actualCount = 0;
    let actualColoredTotal = 0;
    let brightnessTotal = 0;
    const y0 = Math.floor(height * 0.2);
    const y1 = Math.floor(height * 0.55);
    for (let y = y0; y <= y1; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const source = frameOffset + (y * width + x) * channels;
        const target = (y * width + x) * 4;
        const expectedBrick = isBrickColor(expected[source], expected[source + 1], expected[source + 2]);
        const actualBrick = isBrickColor(actual[target], actual[target + 1], actual[target + 2]);
        if (expectedBrick) {
          expectedCount += 1;
          if (actualBrick) actualCount += 1;
        }
        if (actualBrick) actualColoredTotal += 1;
        brightnessTotal += (actual[target] + actual[target + 1] + actual[target + 2]) / 3;
      }
    }
    const sampledPixels = (y1 - y0 + 1) * width;
    return {
      status: 'measured',
      width,
      height,
      hash: fnvHash(actual),
      expected_brick_pixels: expectedCount,
      actual_brick_pixels: actualCount,
      brick_coverage: expectedCount > 0 ? actualCount / expectedCount : 0,
      colored_ratio: actualColoredTotal / sampledPixels,
      mean_brightness: brightnessTotal / sampledPixels,
    };
  };
  const collectVisualValidation = async (frames) => {
    const debug = window.visionaryDemoDebug;
    debug.setRecordLatentSummaries(true);
    await resetDemo();
    const sampleFrames = [...new Set([1, 2, 4, frames].filter((frame) => frame <= frames))];
    const samples = [];
    for (const frame of sampleFrames) {
      await runToFrame(frame);
      const stats = debug.frameStats;
      const numeric = stats.at(-1)?.latent ?? { status: 'skipped', reason: 'latent summary unavailable' };
      samples.push({ frame, visual: measureBrickCoverage(), numeric });
    }
    debug.setRecordLatentSummaries(false);
    const measuredCoverage = samples.map((sample) => sample.visual).filter((visual) => visual.status === 'measured');
    const frameHashes = measuredCoverage.map((visual) => visual.hash);
    const numericSamples = samples.map((sample) => sample.numeric).filter((numeric) => numeric?.status === 'measured');
    const latentHashes = numericSamples.map((numeric) => numeric.hash);
    const minBrickCoverage = Math.min(...measuredCoverage.map((visual) => visual.brick_coverage));
    const numericalPassed =
      numericSamples.length === 0 ||
      (new Set(latentHashes).size >= MIN_UNIQUE_LATENT_HASHES && numericSamples.every((numeric) => numeric.finite));
    return {
      status:
        new Set(frameHashes).size >= MIN_UNIQUE_FRAME_HASHES &&
        measuredCoverage.length > 0 &&
        minBrickCoverage >= MIN_BRICK_COVERAGE &&
        numericalPassed
          ? 'passed'
          : 'failed',
      sample_count: samples.length,
      unique_hashes: new Set(frameHashes).size,
      hashes: frameHashes,
      min_brick_coverage: Number.isFinite(minBrickCoverage) ? minBrickCoverage : null,
      minimum_required_brick_coverage: MIN_BRICK_COVERAGE,
      numerical: {
        status:
          numericSamples.length === 0
            ? 'skipped'
            : new Set(latentHashes).size >= MIN_UNIQUE_LATENT_HASHES && numericSamples.every((numeric) => numeric.finite)
              ? 'passed'
              : 'failed',
        sample_count: numericSamples.length,
        unique_hashes: new Set(latentHashes).size,
        hashes: latentHashes,
        minimum_required_unique_hashes: MIN_UNIQUE_LATENT_HASHES,
        all_finite: numericSamples.every((numeric) => numeric.finite),
      },
      samples,
    };
  };
  const runtimeSnapshot = () => {
    const debug = window.visionaryDemoDebug;
    const runtime = debug.runtime;
    return {
      backend: runtime.backend,
      backend_text: document.querySelector('#backend')?.textContent ?? null,
      status_text: document.querySelector('#status')?.textContent ?? null,
      frame_count: debug.frameCount,
      cache_length: runtime.cache.length.data[0],
      context_length: runtime.contextLength,
      initial_cache_source: runtime.initialCacheSource,
      initial_cache_length: runtime.initialCache.length.data[0],
      graph_capture: Boolean(runtime.graphCapture?.enabled),
      full_graph_capture: Boolean(runtime.fullGraphCapture?.enabled),
      decoder_graph_capture: Boolean(runtime.decoderGraphCapture),
      decoder_worker: Boolean(runtime.decoderWorker),
      split_wasm_dynamics: Boolean(runtime.splitWasmDynamics),
      full_wasm_head_time_dynamics: Boolean(runtime.fullWasmHeadTimeDynamics),
      full_step_cache_layout: runtime.fullStepCacheLayout ?? null,
      step_export: runtime.specs.fullStep?.name ?? runtime.specs.step?.name ?? null,
      decoder_export: runtime.specs.decoder?.name ?? null,
      sample_steps: runtime.manifest?.demo_generation?.sample_steps ?? null,
      asset_base: runtime.assetBase ?? null,
      ort_module_url: runtime.ortModuleUrl ?? null,
      wasm_num_threads: runtime.wasmNumThreads ?? null,
      decoder_worker_num_threads: runtime.decoderWorkerNumThreads ?? null,
      graph_optimization_level: runtime.graphOptimizationLevel ?? null,
      context_name: runtime.contextName ?? null,
      initial_cache_name: runtime.initialCacheName ?? null,
    };
  };

  await waitFor(
    () => window.visionaryDemoDebug && document.querySelector('#status')?.textContent?.includes('Ready'),
    180000,
    'ready',
  );
  const initialRuntime = runtimeSnapshot();
  const outputValidationFrames = Math.max(${config.validationFrames}, ${config.timedFrames});
  const outputValidation = await collectVisualValidation(outputValidationFrames);
  await resetDemo();
  const warmup = await runWindow(${config.warmupFrames});
  await resetDemo();
  const timed = await runWindow(${config.timedFrames});
  const finalRuntime = runtimeSnapshot();
  pauseDemo();
  done({
    ok: true,
    initialRuntime,
    finalRuntime,
    warmup,
    timed,
    outputValidation,
    outputValidationFrames,
    userAgent: navigator.userAgent,
  });
})().catch((error) => {
  done({
    ok: false,
    error: error instanceof Error ? error.message : String(error),
    stack: error?.stack,
    statusText: document.querySelector('#status')?.textContent ?? null,
  });
});
`;
}

async function createSession(driverUrl: string) {
  const response = await fetchJson(`${driverUrl}/session`, {
    method: 'POST',
    body: JSON.stringify({
      capabilities: {
        alwaysMatch: {
          browserName: 'Safari',
        },
      },
    }),
  });
  const sessionId = response.sessionId ?? response.value?.sessionId;
  if (!sessionId) throw new Error(`safaridriver did not return a session id: ${JSON.stringify(response)}`);
  return sessionId;
}

async function runNativeSafariBenchmark(options: CliOptions) {
  const driverUrl = `http://127.0.0.1:${options.driverPort}`;
  const url = demoUrl(options);
  const sessionId = await createSession(driverUrl);
  try {
    await fetchJson(`${driverUrl}/session/${sessionId}/url`, {
      method: 'POST',
      body: JSON.stringify({ url }),
    });
    const response = await fetchJson(`${driverUrl}/session/${sessionId}/execute/async`, {
      method: 'POST',
      body: JSON.stringify({
        script: benchmarkScript(options),
        args: [],
      }),
    });
    const value = response.value ?? response;
    if (!value.ok) {
      throw new Error(
        [
          value.error ?? 'native Safari benchmark failed',
          value.statusText ? `status: ${value.statusText}` : null,
          value.stack,
        ]
          .filter(Boolean)
          .join('\n'),
      );
    }
    return { value, url };
  } finally {
    await fetch(`${driverUrl}/session/${sessionId}/window`, { method: 'DELETE' }).catch(() => null);
    await fetch(`${driverUrl}/session/${sessionId}`, { method: 'DELETE' }).catch(() => null);
  }
}

async function writeResult(result: any) {
  await mkdir(RESULT_DIR, { recursive: true });
  const text = `${JSON.stringify(result, null, 2)}\n`;
  await writeFile(RESULT_PATH, text);
  await writeFile(LATEST_PATH, text);
}

function resultFromSafariValue(options: CliOptions, value: any, url: string) {
  const timing = timingFromWindow(value.timed);
  const warmupTiming = timingFromWindow(value.warmup);
  const steadyStateFps = timing.window_fps ?? 0;
  const p95FrameMs = timing.frame_interval?.p95_ms ?? timing.frame_latency?.p95_ms ?? null;
  const maxFrameMs = timing.frame_interval?.max_ms ?? timing.frame_latency?.max_ms ?? null;
  const speedValidation = {
    status: steadyStateFps >= options.minFps ? 'passed' : 'failed',
    steady_state_fps: steadyStateFps,
    minimum_required_fps: options.minFps,
  };
  const frameStabilityValidation = {
    status:
      Number.isFinite(p95FrameMs) &&
      Number.isFinite(maxFrameMs) &&
      p95FrameMs <= options.maxFrameP95Ms &&
      maxFrameMs <= options.maxFrameIntervalMs
        ? 'passed'
        : 'failed',
    p95_frame_ms: p95FrameMs,
    max_frame_ms: maxFrameMs,
    maximum_allowed_p95_ms: options.maxFrameP95Ms,
    maximum_allowed_frame_ms: options.maxFrameIntervalMs,
  };
  const status =
    value.outputValidation.status === 'passed' &&
    speedValidation.status === 'passed' &&
    frameStabilityValidation.status === 'passed'
      ? 'passed'
      : 'failed';
  return {
    schema_version: 3,
    status,
    message:
      status === 'passed'
        ? 'native Safari actual demo benchmark passed'
        : 'native Safari actual demo benchmark failed validation',
    benchmark_kind: 'actual_demo_stream',
    benchmark_modes: ['streaming_frame'],
    mode: 'native_safari_wasm',
    config: {
      provider: 'wasm',
      warmupFrames: options.warmupFrames,
      timedFrames: options.timedFrames,
      validationFrames: options.validationFrames,
      outputValidationFrames: value.outputValidationFrames,
      minFps: options.minFps,
      maxFrameP95Ms: options.maxFrameP95Ms,
      maxFrameIntervalMs: options.maxFrameIntervalMs,
      assetBase: options.assetBase,
      ortModule: options.ortModule,
      demoQuery: options.demoQuery,
      demo_url: url,
    },
    created_at: new Date().toISOString(),
    completed_at: new Date().toISOString(),
    user_agent: value.userAgent,
    demo: {
      initial: value.initialRuntime,
      final: value.finalRuntime,
    },
    results: [
      {
        mode: 'streaming_frame',
        artifact_role: 'actual_demo_frame',
        name: 'native_safari_actual_demo_streaming_frame',
        timing: {
          warmup_stream: warmupTiming,
          streaming_frame: timing.frame_interval ?? timing.frame_latency,
          frame_latency: timing.frame_latency,
          frame_interval: timing.frame_interval,
          stages: timing.stages,
          measured_frames: timing.measured_frames,
          elapsed_ms: timing.elapsed_ms,
          steady_state_ms_per_frame: timing.window_ms_per_frame,
          steady_state_fps: timing.window_fps,
        },
        output_validation: value.outputValidation,
        speed_validation: speedValidation,
        frame_stability_validation: frameStabilityValidation,
      },
    ],
  };
}

const options = parseArgs(Bun.argv.slice(2));
let serverProcess: Bun.Subprocess | null = null;
let driverProcess: Bun.Subprocess | null = null;

try {
  serverProcess = await ensureStaticServer(options.baseUrl);
  driverProcess = await ensureSafariDriver(options.driverPort);
  const { value, url } = await runNativeSafariBenchmark(options);
  const result = resultFromSafariValue(options, value, url);
  await writeResult(result);
  const summary = result.results[0];
  console.log(
    [
      `${result.message}`,
      `fps=${summary.timing.steady_state_fps.toFixed(2)}`,
      `ms=${summary.timing.steady_state_ms_per_frame.toFixed(2)}`,
      `validation=${summary.output_validation.status}`,
      `numerical=${summary.output_validation.numerical.status}`,
      `result=${RESULT_PATH}`,
    ].join(' '),
  );
  if (result.status !== 'passed') process.exitCode = 1;
} finally {
  driverProcess?.kill();
  serverProcess?.kill();
  stopSafariAutomation();
}
