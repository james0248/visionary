import { expect, test, type Page } from '@playwright/test';

const demoPath = `/webgpu_app/demo/index.html${process.env.DEMO_QUERY ?? ''}`;
const pacmanDemoPath = `/webgpu_app/demo/pacman.html${process.env.DEMO_QUERY ?? ''}`;

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

test('world model demo keeps safari profile on the valid dynamics path @demo', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto(
    '/webgpu_app/demo/index.html?browserProfile=safari&graphCapture=true&dynamicsGraphCapture=true&fullDynamicsGraphCapture=true&decoderGraphCapture=true&allowSafariDynamicsGraphCapture=true',
  );
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });

  const captureState = await page.evaluate(() => ({
    dynamics: Boolean((window as any).visionaryDemoDebug.runtime.graphCapture?.enabled),
    fullDynamics: Boolean((window as any).visionaryDemoDebug.runtime.fullGraphCapture?.enabled),
    decoder: Boolean((window as any).visionaryDemoDebug.runtime.decoderGraphCapture),
    initialCacheSource: (window as any).visionaryDemoDebug.runtime.initialCacheSource,
    initialCacheLength: (window as any).visionaryDemoDebug.runtime.initialCache.length.data[0],
  }));
  expect(captureState).toEqual({
    dynamics: false,
    fullDynamics: false,
    decoder: false,
    initialCacheSource: 'artifact',
    initialCacheLength: 4,
  });

  await expect(visibleFrame(page)).toBeVisible();
  expect(await visibleFramePixelHash(page)).toBe('4d1cdf9b');

  const expectedFrameHashes = new Map([
    [1, '04fb5f97'],
    [2, 'b12b92c0'],
    [4, '64bbd049'],
    [64, 'bfafbf81'],
    [65, 'a5c0e02a'],
    [66, 'c90b4fc3'],
  ]);
  const frameHashes = new Map<number, string>();
  const cacheLengths = new Map<number, number>();
  for (let frameNumber = 1; frameNumber <= 66; frameNumber += 1) {
    await generateFrameInPage(page);
    if (!expectedFrameHashes.has(frameNumber)) continue;
    frameHashes.set(frameNumber, await visibleFramePixelHash(page));
    cacheLengths.set(
      frameNumber,
      await page.evaluate(() => (window as any).visionaryDemoDebug.runtime.cache.length.data[0]),
    );
  }

  expect(Object.fromEntries(frameHashes)).toEqual(Object.fromEntries(expectedFrameHashes));
  expect(cacheLengths.get(64)).toBe(64);
  expect(cacheLengths.get(65)).toBe(64);
  expect(cacheLengths.get(66)).toBe(64);
  expect(new Set([...frameHashes.values()]).size).toBe(frameHashes.size);
  expect(new Set([frameHashes.get(64), frameHashes.get(65), frameHashes.get(66)]).size).toBe(3);
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

  await page.goto('/webgpu_app/demo/index.html?backend=wasm');
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

test('pacman demo wires cardinal and diagonal actions @demo', async ({ page }) => {
  await page.goto(pacmanDemoPath);
  await expect(page.locator('.pacman-grid [data-action-id]')).toHaveCount(9);
  await expect
    .poll(async () => page.evaluate(() => (window as any).visionaryDemoActionsReady === true), {
      timeout: 30_000,
    })
    .toBe(true);

  await page.keyboard.down('ArrowUp');
  await expect(page.locator('#action')).toHaveText('up');
  await page.keyboard.down('ArrowRight');
  await expect(page.locator('#action')).toHaveText('up+right');
  await page.keyboard.up('ArrowUp');
  await expect(page.locator('#action')).toHaveText('right');
  await page.keyboard.up('ArrowRight');
  await expect(page.locator('#action')).toHaveText('noop');
});

test('pacman demo starts and renders a frame @demo', async ({ page }) => {
  await page.goto(pacmanDemoPath);
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });
  await page.locator('#start').click();
  await expect
    .poll(async () => Number(await page.locator('#frame-count').textContent()), {
      timeout: 180_000,
    })
    .toBeGreaterThan(0);
});
