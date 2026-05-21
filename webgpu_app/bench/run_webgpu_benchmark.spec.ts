import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { expect, test, type Page } from '@playwright/test';

const RESULT_DIR = 'bench/results';
const RESULT_PATH = path.join(RESULT_DIR, 'latest.json');
const GRAPH_CAPTURE_RESULT_PATH = path.join(RESULT_DIR, 'graph_capture_latest.json');

type BenchmarkOptions = {
  graphCapture?: boolean;
};

type BenchmarkResult = any;

const DEFAULT_WARMUP_FRAMES = 8;
const DEFAULT_TIMED_FRAMES = 64;
const DEFAULT_VALIDATION_FRAMES = 6;
const DEFAULT_MIN_FPS = 30;
const DEFAULT_MAX_FRAME_P95_MS = 50;
const DEFAULT_MAX_FRAME_INTERVAL_MS = 100;
const MIN_BRICK_COVERAGE = 0.45;
const MIN_UNIQUE_FRAME_HASHES = 2;
const MIN_UNIQUE_LATENT_HASHES = 2;

function envFlag(name: string) {
  const value = process.env[name];
  return value != null && ['1', 'true', 'yes', 'on'].includes(value.toLowerCase());
}

function numberEnv(name: string, fallback: number) {
  const parsed = Number(process.env[name]);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function summarize(samples: number[]) {
  if (samples.length === 0) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  const sum = samples.reduce((total, value) => total + value, 0);
  const percentile = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
  return {
    count: samples.length,
    mean_ms: sum / samples.length,
    median_ms: percentile(0.5),
    p95_ms: percentile(0.95),
    min_ms: sorted[0],
    max_ms: sorted[sorted.length - 1],
  };
}

function fnvHash(bytes: Buffer | Uint8Array) {
  let value = 2166136261 >>> 0;
  for (const byte of bytes) {
    value ^= byte;
    value = Math.imul(value, 16777619) >>> 0;
  }
  return value.toString(16).padStart(8, '0');
}

function visibleFrame(page: Page) {
  return page
    .locator('#frame:not([hidden]), .frame-preview:not([hidden]), .frame-fallback:not([hidden])')
    .first();
}

function benchmarkConfig(options: BenchmarkOptions) {
  const graphCapture = envFlag('WEBGPU_BENCHMARK_GRAPH_CAPTURE') || Boolean(options.graphCapture);
  const provider = process.env.WEBGPU_BENCHMARK_PROVIDER ?? 'webgpu';
  const wasmDefaults = provider === 'wasm';
  return {
    provider,
    warmupFrames: numberEnv('WEBGPU_BENCHMARK_WARMUP_RUNS', DEFAULT_WARMUP_FRAMES),
    timedFrames: numberEnv('WEBGPU_BENCHMARK_TIMED_RUNS', DEFAULT_TIMED_FRAMES),
    validationFrames: numberEnv('WEBGPU_BENCHMARK_VALIDATION_FRAMES', DEFAULT_VALIDATION_FRAMES),
    minFps: numberEnv('WEBGPU_BENCHMARK_MIN_FPS', DEFAULT_MIN_FPS),
    maxFrameP95Ms: numberEnv('WEBGPU_BENCHMARK_MAX_FRAME_P95_MS', DEFAULT_MAX_FRAME_P95_MS),
    maxFrameIntervalMs: numberEnv(
      'WEBGPU_BENCHMARK_MAX_FRAME_INTERVAL_MS',
      DEFAULT_MAX_FRAME_INTERVAL_MS,
    ),
    graphCapture,
    dynamicsGraphCapture: process.env.WEBGPU_BENCHMARK_DYNAMICS_GRAPH_CAPTURE ?? (graphCapture ? 'true' : null),
    decoderGraphCapture: process.env.WEBGPU_BENCHMARK_DECODER_GRAPH_CAPTURE ?? (graphCapture ? 'true' : null),
    preferredLayout: process.env.WEBGPU_BENCHMARK_PREFERRED_LAYOUT ?? null,
    graphOptimizationLevel: process.env.WEBGPU_BENCHMARK_GRAPH_OPTIMIZATION_LEVEL ?? null,
    stepArtifact: process.env.WEBGPU_BENCHMARK_STEP_ARTIFACT ?? null,
    decoderArtifact: process.env.WEBGPU_BENCHMARK_DECODER_ARTIFACT ?? null,
    contextName: process.env.WEBGPU_BENCHMARK_CONTEXT_NAME ?? null,
    initialCacheName: process.env.WEBGPU_BENCHMARK_INITIAL_CACHE_NAME ?? null,
    assetBase:
      process.env.WEBGPU_BENCHMARK_ASSET_BASE ??
      (wasmDefaults ? '/dream_arcade_assets/breakout_wasm_default_mha' : null),
    browserProfile: process.env.WEBGPU_BENCHMARK_BROWSER_PROFILE ?? null,
    ortModule:
      process.env.WEBGPU_BENCHMARK_ORT_MODULE ??
      (wasmDefaults ? '/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs' : null),
    wasmNumThreads: process.env.WEBGPU_BENCHMARK_WASM_NUM_THREADS ?? null,
    decoderWorkerPipeline:
      process.env.WEBGPU_BENCHMARK_DECODER_WORKER_PIPELINE ?? (wasmDefaults ? 'true' : null),
    decoderWorkerNumThreads:
      process.env.WEBGPU_BENCHMARK_DECODER_WORKER_NUM_THREADS ?? (wasmDefaults ? '3' : null),
    splitWasmDynamics: process.env.WEBGPU_BENCHMARK_SPLIT_WASM_DYNAMICS ?? null,
  };
}

function demoUrl(config: ReturnType<typeof benchmarkConfig>) {
  const params = new URLSearchParams();
  params.set('fps', '0');
  if (config.provider) params.set('backend', config.provider);
  if (config.assetBase) params.set('assetBase', config.assetBase);
  if (config.browserProfile) params.set('browserProfile', config.browserProfile);
  if (config.contextName) params.set('contextName', config.contextName);
  if (config.initialCacheName) params.set('initialCacheName', config.initialCacheName);
  if (config.stepArtifact) params.set('fullCacheStepExport', config.stepArtifact);
  if (config.decoderArtifact) params.set('decoderExport', config.decoderArtifact);
  if (config.preferredLayout) params.set('preferredLayout', config.preferredLayout);
  if (config.graphOptimizationLevel) params.set('graphOptimizationLevel', config.graphOptimizationLevel);
  if (config.ortModule) params.set('ortModule', config.ortModule);
  if (config.wasmNumThreads) params.set('wasmNumThreads', config.wasmNumThreads);
  if (config.decoderWorkerPipeline) params.set('decoderWorkerPipeline', config.decoderWorkerPipeline);
  if (config.decoderWorkerNumThreads) params.set('decoderWorkerNumThreads', config.decoderWorkerNumThreads);
  if (config.splitWasmDynamics) params.set('splitWasmDynamics', config.splitWasmDynamics);
  if (config.graphCapture) {
    params.set('graphCapture', 'true');
    params.set('allowUnsafeGraphCapture', 'true');
    params.set('allowSafariGraphCapture', 'true');
  }
  if (config.dynamicsGraphCapture) {
    params.set('fullDynamicsGraphCapture', config.dynamicsGraphCapture);
  }
  if (config.decoderGraphCapture) {
    params.set('decoderGraphCapture', config.decoderGraphCapture);
  }
  return `/demo/index.html?${params.toString()}`;
}

async function waitForReady(page: Page) {
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });
  await expect(visibleFrame(page)).toBeVisible();
}

async function resetDemo(page: Page) {
  await page.locator('#reset').click();
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });
  await expect
    .poll(() => page.evaluate(() => (window as any).visionaryDemoDebug?.frameCount ?? -1), {
      timeout: 30_000,
    })
    .toBe(0);
}

async function pauseDemo(page: Page) {
  if ((await page.locator('#start').textContent())?.includes('Pause')) {
    await page.locator('#start').click();
  }
}

async function runDemoStreamWindow(page: Page, frames: number) {
  return page.evaluate(async (targetFrames) => {
    const debug = (window as any).visionaryDemoDebug;
    const button = document.querySelector<HTMLButtonElement>('#start');
    if (!debug || !button) throw new Error('Demo debug API is unavailable');
    const startFrame = debug.frameCount;
    const targetFrame = startFrame + targetFrames;
    const startMs = performance.now();
    const done = debug.waitForFrameCount(targetFrame, 300_000);
    if (!button.textContent?.includes('Pause')) button.click();
    await done;

    const endMs = performance.now();
    if (button.textContent?.includes('Pause')) button.click();
    const frameStats = debug.frameStats
      .filter((entry: any) => entry.frame > startFrame && entry.frame <= targetFrame)
      .slice(-targetFrames);
    return {
      startFrame,
      targetFrame,
      endFrame: debug.frameCount,
      startMs,
      endMs,
      elapsedMs: endMs - startMs,
      frameStats,
    };
  }, frames);
}

async function screenshotHash(page: Page) {
  const screenshot = await visibleFrame(page).screenshot({ animations: 'disabled' });
  return {
    hash: fnvHash(screenshot),
    base64: screenshot.toString('base64'),
  };
}

async function measureBrickCoverage(page: Page, screenshotBase64: string) {
  return page.evaluate(async (base64) => {
    const debug = (window as any).visionaryDemoDebug;
    const runtime = debug.runtime;
    const expectedTensor = runtime.displayPixels;
    if (!expectedTensor) return { status: 'skipped', reason: 'display_pixels artifact is unavailable' };
    const [frames, height, width, channels] = expectedTensor.dims;
    const referenceFrame = Math.min(frames - 1, (runtime.contextManifest.prefix_frames ?? 1) - 1);
    const frameOffset = referenceFrame * height * width * channels;
    const image = new Image();
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error('Failed to decode rendered frame screenshot'));
      image.src = `data:image/png;base64,${base64}`;
    });
    const scratch = document.createElement('canvas');
    scratch.width = width;
    scratch.height = height;
    const context = scratch.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('2D scratch context is unavailable');
    context.imageSmoothingEnabled = false;
    context.drawImage(image, 0, 0, width, height);
    const actual = context.getImageData(0, 0, width, height).data;
    const expected = expectedTensor.data;
    const isBrickColor = (red: number, green: number, blue: number) => {
      const max = Math.max(red, green, blue);
      const min = Math.min(red, green, blue);
      return max > 50 && max - min > 18;
    };
    let expectedCount = 0;
    let actualCount = 0;
    let actualColoredTotal = 0;
    let brightnessTotal = 0;
    for (let y = Math.floor(height * 0.2); y <= Math.floor(height * 0.55); y += 1) {
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
    const sampledPixels = (Math.floor(height * 0.55) - Math.floor(height * 0.2) + 1) * width;
    return {
      status: 'measured',
      width,
      height,
      expected_brick_pixels: expectedCount,
      actual_brick_pixels: actualCount,
      brick_coverage: expectedCount > 0 ? actualCount / expectedCount : 0,
      colored_ratio: actualColoredTotal / sampledPixels,
      mean_brightness: brightnessTotal / sampledPixels,
    };
  }, screenshotBase64);
}

async function collectVisualValidation(page: Page, frames: number) {
  await page.evaluate(() => (window as any).visionaryDemoDebug?.setRecordLatentSummaries?.(true));
  await resetDemo(page);
  const sampleFrames = [...new Set([1, 2, 4, frames].filter((frame) => frame <= frames))];
  const samples = [];
  for (const frame of sampleFrames) {
    await page.locator('#start').click();
    await page.evaluate(
      (targetFrame) => (window as any).visionaryDemoDebug.waitForFrameCount(targetFrame, 240_000),
      frame,
    );
    await pauseDemo(page);
    const screenshot = await screenshotHash(page);
    const numeric = await page.evaluate(() => {
      const stats = (window as any).visionaryDemoDebug?.frameStats ?? [];
      return stats.at(-1)?.latent ?? { status: 'skipped', reason: 'latent summary unavailable' };
    });
    samples.push({
      frame,
      hash: screenshot.hash,
      visual: await measureBrickCoverage(page, screenshot.base64),
      numeric,
    });
  }
  await page.evaluate(() => (window as any).visionaryDemoDebug?.setRecordLatentSummaries?.(false));
  const hashes = samples.map((sample) => sample.hash);
  const numericSamples = samples
    .map((sample) => sample.numeric)
    .filter((numeric: any) => numeric?.status === 'measured');
  const latentHashes = numericSamples.map((numeric: any) => numeric.hash);
  const measuredCoverage = samples
    .map((sample) => sample.visual)
    .filter((visual: any) => visual.status === 'measured');
  const minBrickCoverage = Math.min(
    ...measuredCoverage.map((visual: any) => visual.brick_coverage),
  );
  const status =
    new Set(hashes).size >= MIN_UNIQUE_FRAME_HASHES &&
    measuredCoverage.length > 0 &&
    minBrickCoverage >= MIN_BRICK_COVERAGE &&
    (numericSamples.length === 0 ||
      (new Set(latentHashes).size >= MIN_UNIQUE_LATENT_HASHES &&
        numericSamples.every((numeric: any) => numeric.finite)))
      ? 'passed'
      : 'failed';
  return {
    status,
    sample_count: samples.length,
    unique_hashes: new Set(hashes).size,
    hashes,
    min_brick_coverage: Number.isFinite(minBrickCoverage) ? minBrickCoverage : null,
    minimum_required_brick_coverage: MIN_BRICK_COVERAGE,
    numerical: {
      status:
        numericSamples.length === 0
          ? 'skipped'
          : new Set(latentHashes).size >= MIN_UNIQUE_LATENT_HASHES &&
              numericSamples.every((numeric: any) => numeric.finite)
            ? 'passed'
            : 'failed',
      sample_count: numericSamples.length,
      unique_hashes: new Set(latentHashes).size,
      hashes: latentHashes,
      minimum_required_unique_hashes: MIN_UNIQUE_LATENT_HASHES,
      all_finite: numericSamples.every((numeric: any) => numeric.finite),
    },
    samples,
  };
}

async function runtimeSnapshot(page: Page) {
  return page.evaluate(() => {
    const debug = (window as any).visionaryDemoDebug;
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
      split_wasm_cache_layout: runtime.splitWasmCacheLayout ?? null,
      step_export: runtime.specs.fullStep?.name ?? runtime.specs.step?.name ?? null,
      decoder_export: runtime.specs.decoder?.name ?? null,
      sample_steps:
        runtime.specs.fullStep?.sample_steps ??
        runtime.specs.step?.sample_steps ??
        runtime.manifest.demo_generation?.sample_steps ??
        null,
      asset_base: runtime.assetBase ?? null,
      ort_module_url: runtime.ortModuleUrl ?? null,
      wasm_num_threads: runtime.wasmNumThreads ?? null,
      decoder_worker_num_threads: runtime.decoderWorkerNumThreads ?? null,
      graph_optimization_level: runtime.graphOptimizationLevel ?? null,
      context_name: runtime.contextManifest.name ?? null,
      initial_cache_name: runtime.initialCacheManifest.name ?? null,
    };
  });
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

async function runBenchmark(
  page: Page,
  mode: string,
  options: BenchmarkOptions = {},
): Promise<BenchmarkResult> {
  const diagnostics: string[] = [];
  const pageErrors: string[] = [];
  page.on('console', (message) => {
    diagnostics.push(`console.${message.type()}: ${message.text()}`);
  });
  page.on('pageerror', (error) => {
    pageErrors.push(`${error.message}\n${error.stack ?? ''}`);
    diagnostics.push(`pageerror: ${error.message}\n${error.stack ?? ''}`);
  });
  page.on('requestfailed', (request) => {
    diagnostics.push(
      `requestfailed: ${request.method()} ${request.url()} ${request.failure()?.errorText ?? ''}`,
    );
  });

  const config = benchmarkConfig(options);
  const url = demoUrl(config);
  const startedAt = new Date().toISOString();
  await page.goto(url);
  try {
    await waitForReady(page);
    const initialRuntime = await runtimeSnapshot(page);
    const outputValidationFrames = Math.max(config.validationFrames, config.timedFrames);
    const outputValidation = await collectVisualValidation(page, outputValidationFrames);
    await resetDemo(page);
    const warmup = await runDemoStreamWindow(page, config.warmupFrames);
    await resetDemo(page);
    const timed = await runDemoStreamWindow(page, config.timedFrames);
    const finalRuntime = await runtimeSnapshot(page);
    await pauseDemo(page);

    const timing = timingFromWindow(timed);
    const steadyStateFps = timing.window_fps ?? 0;
    const p95FrameMs = timing.frame_interval?.p95_ms ?? timing.frame_latency?.p95_ms ?? null;
    const maxFrameMs = timing.frame_interval?.max_ms ?? timing.frame_latency?.max_ms ?? null;
    const speedValidation = {
      status: steadyStateFps >= config.minFps ? 'passed' : 'failed',
      steady_state_fps: steadyStateFps,
      minimum_required_fps: config.minFps,
    };
    const frameStabilityValidation = {
      status:
        Number.isFinite(p95FrameMs) &&
        Number.isFinite(maxFrameMs) &&
        p95FrameMs <= config.maxFrameP95Ms &&
        maxFrameMs <= config.maxFrameIntervalMs
          ? 'passed'
          : 'failed',
      p95_frame_ms: p95FrameMs,
      max_frame_ms: maxFrameMs,
      maximum_allowed_p95_ms: config.maxFrameP95Ms,
      maximum_allowed_frame_ms: config.maxFrameIntervalMs,
    };
    const status =
      outputValidation.status === 'passed' &&
      speedValidation.status === 'passed' &&
      frameStabilityValidation.status === 'passed' &&
      pageErrors.length === 0
        ? 'passed'
        : 'failed';
    return {
      schema_version: 3,
      status,
      message:
        status === 'passed'
          ? 'actual demo benchmark passed'
          : 'actual demo benchmark failed validation',
      benchmark_kind: 'actual_demo_stream',
      benchmark_modes: ['streaming_frame'],
      mode,
      config: {
        ...config,
        outputValidationFrames,
        demo_url: url,
      },
      created_at: startedAt,
      completed_at: new Date().toISOString(),
      user_agent: await page.evaluate(() => navigator.userAgent),
      diagnostics,
      page_errors: pageErrors,
      demo: {
        initial: initialRuntime,
        final: finalRuntime,
      },
      results: [
        {
          mode: 'streaming_frame',
          artifact_role: 'actual_demo_frame',
          name: 'actual_demo_streaming_frame',
          timing: {
            warmup_stream: timingFromWindow(warmup),
            streaming_frame: timing.frame_interval ?? timing.frame_latency,
            frame_latency: timing.frame_latency,
            frame_interval: timing.frame_interval,
            stages: timing.stages,
            measured_frames: timing.measured_frames,
            elapsed_ms: timing.elapsed_ms,
            steady_state_ms_per_frame: timing.window_ms_per_frame,
            steady_state_fps: timing.window_fps,
          },
          output_validation: outputValidation,
          speed_validation: speedValidation,
          frame_stability_validation: frameStabilityValidation,
        },
      ],
    };
  } catch (error) {
    const statusText = await page.locator('#status').textContent().catch(() => null);
    throw new Error(
      [
        error instanceof Error ? error.message : String(error),
        `status: ${statusText ?? '<missing>'}`,
        `url: ${url}`,
        ...diagnostics.slice(-50),
      ].join('\n'),
    );
  }
}

async function writeResult(result: BenchmarkResult, resultPath = RESULT_PATH) {
  await mkdir(RESULT_DIR, { recursive: true });
  await writeFile(resultPath, `${JSON.stringify(result, null, 2)}\n`);
}

async function writeLatestResults(
  result: BenchmarkResult,
  testInfo: { project: { name: string } },
  resultPath = RESULT_PATH,
) {
  await writeResult(result, resultPath);
  const parsedPath = path.parse(resultPath);
  const projectName = testInfo.project.name.replace(/[^a-z0-9_-]+/gi, '_').toLowerCase();
  await writeResult(
    result,
    path.join(parsedPath.dir, `${parsedPath.name}_${projectName}${parsedPath.ext}`),
  );
}

function expectPassedBenchmark(result: BenchmarkResult) {
  expect(result.status, result.message ?? '').toBe('passed');
  expect(result.schema_version).toBe(3);
  expect(result.benchmark_kind).toBe('actual_demo_stream');
  expect(result.benchmark_modes).toEqual(['streaming_frame']);
  const streaming = result.results.find((entry) => entry.mode === 'streaming_frame');
  expect(streaming?.output_validation).toMatchObject({ status: 'passed' });
  expect(streaming?.output_validation.unique_hashes).toBeGreaterThanOrEqual(MIN_UNIQUE_FRAME_HASHES);
  expect(streaming?.speed_validation).toMatchObject({ status: 'passed' });
  expect(streaming?.frame_stability_validation).toMatchObject({ status: 'passed' });
  if (result.demo.final.backend === 'wasm') {
    expect(streaming?.output_validation.numerical).toMatchObject({ status: 'passed' });
    expect(streaming?.output_validation.numerical.unique_hashes).toBeGreaterThanOrEqual(
      MIN_UNIQUE_LATENT_HASHES,
    );
  }
  expect(streaming?.timing.steady_state_fps).toBeGreaterThanOrEqual(result.config.minFps);
  expect(result.demo.final.cache_length).toBe(result.demo.final.context_length);
}

test('webgpu benchmark smoke @smoke', async ({ page }, testInfo) => {
  const result = await runBenchmark(page, 'streaming');
  await writeLatestResults(result, testInfo);
  expectPassedBenchmark(result);
});

test('webgpu demo streaming benchmark @output_validation', async ({ page }, testInfo) => {
  const result = await runBenchmark(page, 'streaming');
  await writeLatestResults(result, testInfo);
  expectPassedBenchmark(result);
});

test('webgpu demo streaming benchmark graph capture @graph-capture', async ({ page }, testInfo) => {
  const result = await runBenchmark(page, 'streaming', { graphCapture: true });
  await writeLatestResults(result, testInfo, GRAPH_CAPTURE_RESULT_PATH);
  expectPassedBenchmark(result);
  expect(result.demo.final.full_graph_capture || result.demo.final.decoder_graph_capture).toBe(true);
});
