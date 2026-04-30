import { mkdir, copyFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { test, expect } from '@playwright/test';

const RESULT_DIR = 'webgpu_app/bench/results';
const RESULT_PATH = path.join(RESULT_DIR, 'profile_diagnostic_latest.json');
const MATRIX_PATH = path.join(RESULT_DIR, 'profile_diagnostic_import_matrix.json');
const BASELINE_PREFIX = 'profile_baseline';

async function archiveLatestProfileBaseline() {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const target = path.join(RESULT_DIR, `${BASELINE_PREFIX}_${timestamp}.json`);
  try {
    await copyFile(path.join(RESULT_DIR, 'latest.json'), target);
    return target;
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

async function runDiagnostic(page, importKind) {
  const diagnostics = [];
  page.on('console', (message) => {
    diagnostics.push(`console.${message.type()}: ${message.text()}`);
  });
  page.on('pageerror', (error) => {
    diagnostics.push(`pageerror: ${error.message}\n${error.stack ?? ''}`);
  });

  const requireHardwareGpu = process.env.ALLOW_SOFTWARE_WEBGPU === '1' ? 'false' : 'true';
  const params = new URLSearchParams({
    importKind,
    requireHardwareGpu,
    runs: process.env.WEBGPU_PROFILE_DIAGNOSTIC_RUNS ?? '1',
    modelName:
      process.env.WEBGPU_PROFILE_DIAGNOSTIC_MODEL ??
      'breakout_dynamics_step_cached_b1_t1',
    drainMs: process.env.WEBGPU_PROFILE_DIAGNOSTIC_DRAIN_MS ?? '500',
    sessionProfiling: process.env.WEBGPU_PROFILE_DIAGNOSTIC_SESSION_PROFILING === '1' ? 'true' : 'false',
  });
  await page.goto(`/webgpu_app/bench/profile_diagnostic.html?${params.toString()}`);
  try {
    const result = await page.waitForFunction(
      () => window.__WEBGPU_PROFILE_DIAGNOSTIC_RESULT__ ?? null,
      null,
      { timeout: 300_000 },
    );
    return result.jsonValue();
  } catch (error) {
    const statusText = await page.locator('#status').textContent().catch(() => null);
    throw new Error(
      [error.message, `status: ${statusText ?? '<missing>'}`, ...diagnostics.slice(-50)].join('\n'),
    );
  }
}

async function writeJson(file, value) {
  await mkdir(RESULT_DIR, { recursive: true });
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`);
}

test('webgpu profiling diagnostic import matrix @profile-diagnostic', async ({ browser }) => {
  const baseline_path = await archiveLatestProfileBaseline();
  const importKinds = (process.env.WEBGPU_PROFILE_DIAGNOSTIC_IMPORTS ?? 'dist_bundle,dist_external_wasm,dist_unminified')
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean);

  const results = [];
  for (const importKind of importKinds) {
    const page = await browser.newPage();
    try {
      const result = await runDiagnostic(page, importKind);
      results.push(result);
    } finally {
      await page.close();
    }
  }

  const matrix = {
    schema_version: 1,
    status: results.some((result) => result.status !== 'failed') ? 'passed' : 'failed',
    created_at: new Date().toISOString(),
    baseline_path,
    results,
    summary: results.map((result) => ({
      importKind: result.config?.importKind,
      status: result.status,
      ort_version: result.ort_version,
      timestamp_query: result.timestamp_query,
      callback_events: result.conclusion?.callback_events ?? 0,
      console_profile_messages: result.conclusion?.console_profile_messages ?? 0,
      session_profiling_console_messages:
        result.conclusion?.session_profiling_console_messages ?? 0,
      profiling_visible: result.conclusion?.profiling_visible ?? false,
      message: result.message ?? null,
    })),
  };

  await writeJson(MATRIX_PATH, matrix);
  await writeJson(RESULT_PATH, results[0] ?? matrix);
  expect(matrix.status).toBe('passed');
});
