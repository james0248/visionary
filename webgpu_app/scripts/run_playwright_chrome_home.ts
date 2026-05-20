#!/usr/bin/env node
import { existsSync, mkdirSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';

const chromeHome = process.env.PLAYWRIGHT_CHROME_HOME ?? '/private/tmp/visionary-chrome-home';
mkdirSync(`${chromeHome}/Library/Application Support/Google/Chrome/Crashpad`, { recursive: true });
mkdirSync(`${chromeHome}/Library/Application Support/Google/Chrome for Testing/Crashpad`, {
  recursive: true,
});
mkdirSync('/private/tmp/visionary-chrome-crashpad', { recursive: true });

const benchmarkFlagEnv = new Map([
  ['--webgpu-benchmark-provider', 'WEBGPU_BENCHMARK_PROVIDER'],
  ['--webgpu-benchmark-warmup-runs', 'WEBGPU_BENCHMARK_WARMUP_RUNS'],
  ['--webgpu-benchmark-timed-runs', 'WEBGPU_BENCHMARK_TIMED_RUNS'],
  ['--webgpu-benchmark-graph-capture', 'WEBGPU_BENCHMARK_GRAPH_CAPTURE'],
  ['--webgpu-benchmark-dynamics-graph-capture', 'WEBGPU_BENCHMARK_DYNAMICS_GRAPH_CAPTURE'],
  ['--webgpu-benchmark-decoder-graph-capture', 'WEBGPU_BENCHMARK_DECODER_GRAPH_CAPTURE'],
  ['--webgpu-benchmark-preferred-layout', 'WEBGPU_BENCHMARK_PREFERRED_LAYOUT'],
  ['--webgpu-benchmark-graph-optimization-level', 'WEBGPU_BENCHMARK_GRAPH_OPTIMIZATION_LEVEL'],
  ['--webgpu-benchmark-prefill-artifact', 'WEBGPU_BENCHMARK_PREFILL_ARTIFACT'],
  ['--webgpu-benchmark-step-artifact', 'WEBGPU_BENCHMARK_STEP_ARTIFACT'],
  ['--webgpu-benchmark-decoder-artifact', 'WEBGPU_BENCHMARK_DECODER_ARTIFACT'],
  ['--webgpu-benchmark-context-name', 'WEBGPU_BENCHMARK_CONTEXT_NAME'],
  ['--webgpu-benchmark-initial-cache-name', 'WEBGPU_BENCHMARK_INITIAL_CACHE_NAME'],
  ['--webgpu-benchmark-asset-base', 'WEBGPU_BENCHMARK_ASSET_BASE'],
  ['--webgpu-benchmark-browser-profile', 'WEBGPU_BENCHMARK_BROWSER_PROFILE'],
  ['--webgpu-benchmark-profiling', 'WEBGPU_BENCHMARK_PROFILING'],
  ['--webgpu-benchmark-profiling-required', 'WEBGPU_BENCHMARK_PROFILING_REQUIRED'],
  ['--webgpu-benchmark-profiling-drain-ms', 'WEBGPU_BENCHMARK_PROFILING_DRAIN_MS'],
  ['--webgpu-benchmark-profiling-top-k', 'WEBGPU_BENCHMARK_PROFILING_TOP_K'],
  ['--webgpu-benchmark-ort-module', 'WEBGPU_BENCHMARK_ORT_MODULE'],
  ['--webgpu-benchmark-wasm-num-threads', 'WEBGPU_BENCHMARK_WASM_NUM_THREADS'],
  ['--webgpu-benchmark-decoder-worker-pipeline', 'WEBGPU_BENCHMARK_DECODER_WORKER_PIPELINE'],
  ['--webgpu-benchmark-decoder-worker-num-threads', 'WEBGPU_BENCHMARK_DECODER_WORKER_NUM_THREADS'],
  ['--webgpu-benchmark-validation-frames', 'WEBGPU_BENCHMARK_VALIDATION_FRAMES'],
  ['--demo-query', 'DEMO_QUERY'],
  ['--allow-software-webgpu', 'ALLOW_SOFTWARE_WEBGPU'],
]);

function parseArgs(args: string[]) {
  const passthrough: string[] = [];
  const env: Record<string, string> = {};
  let attempts = Number.parseInt(process.env.PLAYWRIGHT_BENCHMARK_ATTEMPTS ?? '3', 10);

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    const [key, inlineValue] = arg.split('=', 2);
    if (key === '--playwright-benchmark-attempts') {
      const value = inlineValue ?? args[index + 1];
      attempts = Number.parseInt(value ?? String(attempts), 10);
      if (inlineValue == null) index += 1;
      continue;
    }
    if (key === '--playwright-channel') {
      const value = inlineValue ?? args[index + 1];
      if (value) env.PLAYWRIGHT_CHANNEL = value;
      if (inlineValue == null && args[index + 1] && !args[index + 1].startsWith('--')) {
        index += 1;
      }
      continue;
    }
    if (key === '--playwright-headless') {
      const value = inlineValue ?? args[index + 1];
      env.PLAYWRIGHT_HEADLESS = value && !value.startsWith('--') ? value : '1';
      if (inlineValue == null && args[index + 1] && !args[index + 1].startsWith('--')) {
        index += 1;
      }
      continue;
    }

    const envName = benchmarkFlagEnv.get(key);
    if (envName) {
      const nextValue = args[index + 1];
      const value = inlineValue ?? (nextValue && !nextValue.startsWith('--') ? nextValue : undefined);
      env[envName] = value ?? '1';
      if (inlineValue == null && value != null) {
        index += 1;
      }
      continue;
    }

    passthrough.push(arg);
  }

  return { passthrough, env, attempts: Number.isFinite(attempts) ? attempts : 3 };
}

const parsedArgs = parseArgs(process.argv.slice(2));
const maxAttempts = parsedArgs.attempts;

function playwrightCommand() {
  if (process.env.PLAYWRIGHT_CLI) return process.env.PLAYWRIGHT_CLI;
  const localBin = resolve(
    process.platform === 'win32' ? 'node_modules/.bin/playwright.cmd' : 'node_modules/.bin/playwright',
  );
  return existsSync(localBin) ? localBin : 'playwright';
}

function defaultBrowsersPath() {
  const home = process.env.HOME;
  if (!home) return undefined;
  if (process.platform === 'darwin') return `${home}/Library/Caches/ms-playwright`;
  if (process.platform === 'win32') {
    return `${process.env.LOCALAPPDATA ?? `${home}/AppData/Local`}/ms-playwright`;
  }
  return `${home}/.cache/ms-playwright`;
}

function runAttempt(attempt) {
  if (attempt > 1) {
    console.warn(`Retrying Playwright benchmark after startup failure (${attempt}/${maxAttempts})...`);
  }

  const childEnv: Record<string, string | undefined> = {
    ...process.env,
    ...parsedArgs.env,
    HOME: chromeHome,
    CFFIXED_USER_HOME: chromeHome,
    PLAYWRIGHT_CHROME_HOME: chromeHome,
  };
  const browsersPath = process.env.PLAYWRIGHT_BROWSERS_PATH ?? defaultBrowsersPath();
  if (browsersPath) {
    childEnv.PLAYWRIGHT_BROWSERS_PATH = browsersPath;
  }

  const child = spawn(playwrightCommand(), parsedArgs.passthrough, {
    stdio: 'inherit',
    env: childEnv,
  });

  child.on('exit', (code, signal) => {
    if (signal) {
      if (attempt < maxAttempts) {
        runAttempt(attempt + 1);
        return;
      }
      process.kill(process.pid, signal);
      return;
    }

    if (code !== 0 && attempt < maxAttempts) {
      runAttempt(attempt + 1);
      return;
    }

    process.exit(code ?? 1);
  });
}

runAttempt(1);
