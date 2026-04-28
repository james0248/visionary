import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { test, expect } from '@playwright/test';

const RESULT_DIR = 'webgpu_app/bench/results';
const RESULT_PATH = path.join(RESULT_DIR, 'latest.json');

async function runBenchmark(page, mode) {
  const diagnostics = [];
  page.on('console', (message) => {
    diagnostics.push(`console.${message.type()}: ${message.text()}`);
  });
  page.on('pageerror', (error) => {
    diagnostics.push(`pageerror: ${error.message}\n${error.stack ?? ''}`);
  });
  page.on('requestfailed', (request) => {
    diagnostics.push(
      `requestfailed: ${request.method()} ${request.url()} ${request.failure()?.errorText ?? ''}`,
    );
  });

  const requireHardwareGpu = process.env.ALLOW_SOFTWARE_WEBGPU === '1' ? 'false' : 'true';
  await page.goto(
    `/webgpu_app/bench/index.html?mode=${mode}&requireHardwareGpu=${requireHardwareGpu}`,
  );
  try {
    const result = await page.waitForFunction(
      () => window.__WEBGPU_BENCHMARK_RESULT__ ?? null,
      null,
      { timeout: 880_000 },
    );
    return result.jsonValue();
  } catch (error) {
    const statusText = await page.locator('#status').textContent().catch(() => null);
    throw new Error(
      [
        error.message,
        `status: ${statusText ?? '<missing>'}`,
        ...diagnostics.slice(-50),
      ].join('\n'),
    );
  }
}

async function writeResult(result) {
  await mkdir(RESULT_DIR, { recursive: true });
  await writeFile(RESULT_PATH, `${JSON.stringify(result, null, 2)}\n`);
}

test('webgpu benchmark smoke @smoke', async ({ page }) => {
  const result = await runBenchmark(page, 'streaming');
  await writeResult(result);
  expect(['passed', 'blocked']).toContain(result.status);
  expect(result.benchmark_modes).toEqual(['cached_prefill', 'cached_step', 'streaming_frame']);
  expect(result.results.map((entry) => entry.mode)).not.toContain('uncached_window');
});

test('webgpu demo streaming benchmark', async ({ page }) => {
  const result = await runBenchmark(page, 'streaming');
  await writeResult(result);
  expect(['passed', 'blocked']).toContain(result.status);
  expect(result.schema_version).toBe(2);
  expect(result.benchmark_modes).toEqual(['cached_prefill', 'cached_step', 'streaming_frame']);
  expect(result.manifest.exports.map((entry) => entry.name)).not.toContain('breakout_dynamics_b1_t64');
  expect(result.manifest.exports.map((entry) => entry.name)).not.toContain(
    'breakout_tokenizer_decoder_b1_t64',
  );

  if (result.status === 'blocked') {
    expect(result.streaming_contract_status).toBe('blocked');
    expect(result.blocked_reason).toContain('Cached dynamics prefill');
    expect(result.results).toEqual([]);
    return;
  }

  expect(result.streaming_contract_status).toBe('available');
  expect(result.results.map((entry) => entry.mode)).toEqual([
    'cached_prefill',
    'cached_step',
    'streaming_frame',
  ]);
  expect(result.results.find((entry) => entry.mode === 'streaming_frame').timing.steady_state_fps)
    .toBeGreaterThan(0);
});
