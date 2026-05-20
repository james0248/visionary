import { expect, test, type Page } from '@playwright/test';

const demoPath = `/demo/index.html${process.env.DEMO_QUERY ?? ''}`;

function visibleFrame(page: Page) {
  return page.locator('#frame:not([hidden]), .frame-fallback:not([hidden])').first();
}

async function generatedFrameCount(page: Page) {
  return page.evaluate(() => (window as any).visionaryDemoDebug?.frameCount ?? 0);
}

async function waitForGeneratedFrame(page: Page, targetFrame: number, previousFrame = 0) {
  const minimumFrame = Math.max(targetFrame, previousFrame + 1);
  await expect
    .poll(async () => generatedFrameCount(page), {
      timeout: 240_000,
    })
    .toBeGreaterThanOrEqual(minimumFrame);
  return generatedFrameCount(page);
}

async function visibleFramePixelHash(page: Page) {
  const hash = await page.evaluate(() => {
    const canvas = document.querySelector<HTMLCanvasElement>('#frame:not([hidden])');
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return null;
    const bytes = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let value = 2166136261 >>> 0;
    for (let index = 0; index < bytes.length; index += 1) {
      value ^= bytes[index];
      value = Math.imul(value, 16777619) >>> 0;
    }
    return value.toString(16).padStart(8, '0');
  });
  if (hash) return hash;
  return (await visibleFrame(page).screenshot()).toString('base64');
}

async function generateFrameInPage(page: Page) {
  await page.evaluate(() => (window as any).visionaryDemoDebug.generateFrame());
  return generatedFrameCount(page);
}

async function cacheSlotStats(page: Page) {
  return page.evaluate(() => (window as any).visionaryDemoDebug.cacheSlotStats());
}

function expectContiguousFilledCache(stats: any, expectedLength: number) {
  expect(stats.cacheLength).toBe(expectedLength);
  expect(stats.activeFinite).toBe(true);
  expect(stats.futureFinite).toBe(true);
  expect(stats.activeNonZeroSlots).toBe(expectedLength);
  expect(stats.activeMinMaxAbs).toBeGreaterThan(0);
  expect(stats.futureMaxAbs).toBe(0);
}

test('world model demo loads its stylesheet @demo', async ({ page }) => {
  await page.goto(demoPath);
  await expect
    .poll(async () =>
      page.locator('.shell').evaluate((element) => getComputedStyle(element).display),
    )
    .toBe('grid');
  await expect
    .poll(async () =>
      page.locator('.machine').evaluate((element) => getComputedStyle(element).borderTopStyle),
    )
    .toBe('solid');
});

test('world model demo starts and renders a frame @demo', async ({ page }) => {
  await page.goto(demoPath);
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });
  await page.locator('#start').click();
  await expect
    .poll(async () => Number(await page.locator('#frame-count').textContent()), {
      timeout: 180_000,
    })
    .toBeGreaterThan(0);
  await expect(page.locator('#latency')).not.toHaveText('-- ms');
});

test('world model demo changes the display over generated frames @demo', async ({ page }) => {
  await page.goto(demoPath);
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });
  await page.locator('#start').click();

  const samples = [];
  const frame = visibleFrame(page);
  await expect(frame).toBeVisible();
  for (const targetFrame of [1, 2, 3, 4, 5, 8, 12]) {
    await expect
      .poll(async () => Number(await page.locator('#frame-count').textContent()), {
        timeout: 180_000,
      })
      .toBeGreaterThanOrEqual(targetFrame);
    samples.push((await frame.screenshot()).toString('base64'));
  }

  expect(new Set(samples).size).toBeGreaterThan(1);
});

test('world model demo fills and keeps the wasm cache over generated frames @demo', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto(demoPath);
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });
  const backend = await page.locator('#backend').textContent();
  test.skip(!backend?.includes('wasm'), 'requires WASM');

  await expect(visibleFrame(page)).toBeVisible();
  const initialCacheState = await page.evaluate(() => {
    const runtime = (window as any).visionaryDemoDebug.runtime;
    return {
      asyncCacheUpdater: Boolean(runtime.cacheUpdater?.async),
      contextLength: runtime.contextLength,
      decoderWorker: Boolean(runtime.decoderWorker),
      initialLength: runtime.cache.length.data[0],
    };
  });
  expect(initialCacheState.asyncCacheUpdater).toBe(true);
  expect(initialCacheState.decoderWorker).toBe(true);
  expect(initialCacheState.initialLength).toBeLessThan(initialCacheState.contextLength);

  const sampledFrameHashes = new Map<number, string>();
  const cacheLengths = new Map<number, number>();
  const framesToFull = initialCacheState.contextLength - initialCacheState.initialLength;
  const sampleFrames = new Set([1, 2, 4, framesToFull, framesToFull + 1, framesToFull + 2]);
  for (let frameNumber = 1; frameNumber <= framesToFull + 2; frameNumber += 1) {
    await generateFrameInPage(page);
    if (!sampleFrames.has(frameNumber)) continue;
    sampledFrameHashes.set(frameNumber, await visibleFramePixelHash(page));
    cacheLengths.set(
      frameNumber,
      await page.evaluate(() => (window as any).visionaryDemoDebug.runtime.cache.length.data[0]),
    );
  }

  expect(cacheLengths.get(1)).toBe(initialCacheState.initialLength + 1);
  expect(cacheLengths.get(2)).toBe(initialCacheState.initialLength + 2);
  expect(cacheLengths.get(4)).toBe(initialCacheState.initialLength + 4);
  expect(cacheLengths.get(framesToFull)).toBe(initialCacheState.contextLength);
  expect(cacheLengths.get(framesToFull + 1)).toBe(initialCacheState.contextLength);
  expect(cacheLengths.get(framesToFull + 2)).toBe(initialCacheState.contextLength);
  expect(new Set([...sampledFrameHashes.values()]).size).toBe(sampledFrameHashes.size);
  expect(pageErrors).toEqual([]);
});

test('world model demo keeps safari profile on the valid dynamics path @demo', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto(
    '/demo/index.html?browserProfile=safari&graphCapture=true&dynamicsGraphCapture=true&fullDynamicsGraphCapture=true&decoderGraphCapture=true&allowSafariDynamicsGraphCapture=true',
  );
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });

  const captureState = await page.evaluate(() => ({
    dynamics: Boolean((window as any).visionaryDemoDebug.runtime.graphCapture?.enabled),
    fullDynamics: Boolean((window as any).visionaryDemoDebug.runtime.fullGraphCapture?.enabled),
    decoder: Boolean((window as any).visionaryDemoDebug.runtime.decoderGraphCapture),
    initialCacheSource: (window as any).visionaryDemoDebug.runtime.initialCacheSource,
    prefillSkipReason: (window as any).visionaryDemoDebug.runtime.prefillSkipReason,
    initialCacheLength: (window as any).visionaryDemoDebug.runtime.initialCache.length.data[0],
  }));
  expect(captureState).toEqual({
    dynamics: false,
    fullDynamics: true,
    decoder: true,
    initialCacheSource: 'artifact-prefill-skipped',
    prefillSkipReason:
      'prefillInitialCache skipped because the context artifact has padded prefix slots',
    initialCacheLength: 4,
  });

  await expect(visibleFrame(page)).toBeVisible();
  expect(await visibleFramePixelHash(page)).toBe('4d1cdf9b');

  const sampledFrameHashes = new Map<number, string>();
  const cacheLengths = new Map<number, number>();
  for (let frameNumber = 1; frameNumber <= 66; frameNumber += 1) {
    await generateFrameInPage(page);
    if (![1, 2, 4, 64, 65, 66].includes(frameNumber)) continue;
    sampledFrameHashes.set(frameNumber, await visibleFramePixelHash(page));
    cacheLengths.set(
      frameNumber,
      await page.evaluate(() => (window as any).visionaryDemoDebug.runtime.cache.length.data[0]),
    );
  }

  expect(sampledFrameHashes.size).toBe(6);
  expect(cacheLengths.get(64)).toBe(64);
  expect(cacheLengths.get(65)).toBe(64);
  expect(cacheLengths.get(66)).toBe(64);
  expect(new Set([...sampledFrameHashes.values()]).size).toBe(sampledFrameHashes.size);
  expect(
    new Set([
      sampledFrameHashes.get(64),
      sampledFrameHashes.get(65),
      sampledFrameHashes.get(66),
    ]).size,
  ).toBe(3);
  expect(pageErrors).toEqual([]);
});

test('world model demo fills the WebGPU cache contiguously before full-cache mode @demo', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto(demoPath);
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });
  const backend = await page.locator('#backend').textContent();
  test.skip(!backend?.includes('webgpu'), 'requires WebGPU');

  const initialStats = await cacheSlotStats(page);
  expect(initialStats.cacheLength).toBeLessThan(initialStats.contextLength);
  expectContiguousFilledCache(initialStats, initialStats.cacheLength);

  const framesToFull = initialStats.contextLength - initialStats.cacheLength;
  const checkpoints = new Set([
    1,
    2,
    4,
    Math.max(1, Math.floor(framesToFull / 2)),
    framesToFull,
    framesToFull + 1,
  ]);
  for (let frameNumber = 1; frameNumber <= framesToFull + 1; frameNumber += 1) {
    await generateFrameInPage(page);
    if (!checkpoints.has(frameNumber)) continue;
    const stats = await cacheSlotStats(page);
    const expectedLength = Math.min(
      initialStats.cacheLength + frameNumber,
      initialStats.contextLength,
    );
    expectContiguousFilledCache(stats, expectedLength);
  }
  expect(pageErrors).toEqual([]);
});

test('world model demo skips padded prefill cache for WebGPU startup @demo', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto(
    '/demo/index.html?backend=webgpu&assetBase=/dream_arcade_assets/breakout&prefillInitialCache=1&skipShortCacheStepWhenFull=1',
  );
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });
  const backend = await page.locator('#backend').textContent();
  test.skip(!backend?.includes('webgpu'), 'requires WebGPU');

  const state = await page.evaluate(() => {
    const runtime = (window as any).visionaryDemoDebug.runtime;
    return {
      initialCacheSource: runtime.initialCacheSource,
      prefillSkipReason: runtime.prefillSkipReason,
      initialLength: runtime.initialCache.length.data[0],
      runtimeLength: runtime.cache.length.data[0],
      contextLength: runtime.contextLength,
      hasShortStepSession: Boolean(runtime.sessions.step),
      hasFullStepSession: Boolean(runtime.sessions.fullStep),
      previewVisible: Boolean(document.querySelector('.frame-preview:not([hidden])')),
    };
  });
  expect(state).toEqual({
    initialCacheSource: 'artifact-prefill-skipped',
    prefillSkipReason:
      'prefillInitialCache skipped because the context artifact has padded prefix slots',
    initialLength: 4,
    runtimeLength: 4,
    contextLength: state.contextLength,
    hasShortStepSession: true,
    hasFullStepSession: true,
    previewVisible: true,
  });

  await generateFrameInPage(page);
  expect(await page.evaluate(() => (window as any).visionaryDemoDebug.runtime.cache.length.data[0]))
    .toBe(5);
  expect(pageErrors).toEqual([]);
});

test('world model demo stays on the validated path after the cache fills @demo', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto(demoPath);
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });
  const backend = await page.locator('#backend').textContent();
  test.skip(!backend?.includes('webgpu'), 'requires WebGPU');

  const captureState = await page.evaluate(() => ({
    dynamics: Boolean((window as any).visionaryDemoDebug.runtime.graphCapture?.enabled),
    fullDynamics: Boolean((window as any).visionaryDemoDebug.runtime.fullGraphCapture?.enabled),
    decoder: Boolean((window as any).visionaryDemoDebug.runtime.decoderGraphCapture),
  }));
  expect(captureState).toEqual({ dynamics: false, fullDynamics: false, decoder: false });

  await page.locator('#start').click();
  const frame = visibleFrame(page);
  await expect(frame).toBeVisible();
  const cacheFillSamples: string[] = [];
  let lastSampledFrame = 0;
  for (const targetFrame of [64, 65, 66]) {
    lastSampledFrame = await waitForGeneratedFrame(page, targetFrame, lastSampledFrame);
    cacheFillSamples.push((await frame.screenshot()).toString('base64'));
  }
  expect(new Set(cacheFillSamples).size).toBeGreaterThan(1);
  expect(pageErrors).toEqual([]);
});

test('world model demo falls back when 2D canvas is unavailable @demo', async ({ page }) => {
  const pageErrors: string[] = [];
  const canvasConsoleMessages: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  page.on('console', (message) => {
    if (/2D canvas|forced 2D canvas/.test(message.text())) {
      canvasConsoleMessages.push(`console.${message.type()}: ${message.text()}`);
    }
  });
  await page.addInitScript(() => {
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function getContext(
      this: HTMLCanvasElement,
      contextId: string,
      options?: unknown,
    ) {
      if (contextId === '2d') throw new Error('forced 2D canvas unavailable');
      return (originalGetContext as any).call(this, contextId, options);
    } as typeof HTMLCanvasElement.prototype.getContext;
  });

  await page.goto('/demo/index.html?backend=wasm');
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });
  await expect(page.locator('.frame-fallback')).toBeVisible();
  await page.locator('#start').click();
  await expect
    .poll(async () => Number(await page.locator('#frame-count').textContent()), {
      timeout: 180_000,
    })
    .toBeGreaterThan(0);
  await expect(page.locator('.frame-fallback')).toBeVisible();
  expect(pageErrors).toEqual([]);
  expect(canvasConsoleMessages).toEqual([]);
});
