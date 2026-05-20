#!/usr/bin/env bun
import { mkdirSync } from 'node:fs';
import { chromium, type Browser, type Page } from '@playwright/test';

type Args = {
  frames: number;
  headless: boolean;
  maxGap: number;
  minCoverage: number;
  minRowCoverage: number;
  urls: { label: string; url: string }[];
};

type BrickBandMetrics = {
  start: number;
  end: number;
  expectedCount: number;
  actualCount: number;
  coverage: number;
  largestGap: number;
  minRowCoverage: number;
};

type FrameMetrics = {
  frame: number;
  state: {
    backend: string;
    cacheLength: number;
    contextLength: number;
    frameCount: number;
    initialCacheSource: string;
  };
  width: number;
  height: number;
  bands: BrickBandMetrics[];
};

const DEFAULT_BASE_URL =
  'http://127.0.0.1:4173/demo/index.html?backend=webgpu&assetBase=/dream_arcade_assets/breakout';

function defaultCases() {
  return [
    { label: 'full-cache', url: DEFAULT_BASE_URL },
    { label: 'safari-profile', url: `${DEFAULT_BASE_URL}&browserProfile=safari` },
  ];
}

function parseNumberFlag(name: string, value: string | undefined, fallback: number) {
  const parsed = Number(value ?? fallback);
  if (!Number.isFinite(parsed)) throw new Error(`${name} must be numeric, got ${value}`);
  return parsed;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    frames: 1,
    headless: process.env.PLAYWRIGHT_HEADLESS === '1',
    maxGap: 8,
    minCoverage: 0.95,
    minRowCoverage: 0.85,
    urls: defaultCases(),
  };
  let customUrl: string | null = null;
  let customLabel = 'custom';
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const [key, inlineValue] = arg.split('=', 2);
    const value = inlineValue ?? argv[index + 1];
    const consumesNext = inlineValue == null && value != null && !value.startsWith('--');
    if (key === '--url') {
      customUrl = value ?? customUrl;
      if (consumesNext) index += 1;
    } else if (key === '--label') {
      customLabel = value ?? customLabel;
      if (consumesNext) index += 1;
    } else if (key === '--frames') {
      args.frames = parseNumberFlag(key, value, args.frames);
      if (consumesNext) index += 1;
    } else if (key === '--min-coverage') {
      args.minCoverage = parseNumberFlag(key, value, args.minCoverage);
      if (consumesNext) index += 1;
    } else if (key === '--min-row-coverage') {
      args.minRowCoverage = parseNumberFlag(key, value, args.minRowCoverage);
      if (consumesNext) index += 1;
    } else if (key === '--max-gap') {
      args.maxGap = parseNumberFlag(key, value, args.maxGap);
      if (consumesNext) index += 1;
    } else if (key === '--headless') {
      args.headless = true;
      if (inlineValue == null && value === '1') index += 1;
    } else if (key === '--headed') {
      args.headless = false;
    } else if (key === '--help' || key === '-h') {
      console.log(`Usage: bun scripts/validate_demo_bricks.ts [options]

E2E-checks that generated Breakout frames keep the top brick band mostly intact.
The script drives the visible UI by clicking Start, waiting for generated frames, and
measuring the rendered game screenshot.

Options:
  --url <url>                 Check one URL instead of the default full-cache + Safari cases
  --label <name>              Label for --url output
  --frames <n>                UI-generated frame count to validate at (default: 1)
  --min-coverage <ratio>      Minimum band coverage (default: 0.95)
  --min-row-coverage <ratio>  Minimum per-row coverage inside the band (default: 0.85)
  --max-gap <pixels>          Maximum contiguous missing run in one brick row (default: 8)
  --headless                  Run Chrome headless
  --headed                    Run Chrome headed
`);
      process.exit(0);
    }
  }
  if (customUrl) args.urls = [{ label: customLabel, url: customUrl }];
  if (!Number.isInteger(args.frames) || args.frames <= 0) {
    throw new Error(`--frames must be a positive integer, got ${args.frames}`);
  }
  if (args.minCoverage < 0 || args.minCoverage > 1) {
    throw new Error(`--min-coverage must be in [0, 1], got ${args.minCoverage}`);
  }
  if (args.minRowCoverage < 0 || args.minRowCoverage > 1) {
    throw new Error(`--min-row-coverage must be in [0, 1], got ${args.minRowCoverage}`);
  }
  if (!Number.isInteger(args.maxGap) || args.maxGap < 0) {
    throw new Error(`--max-gap must be a non-negative integer, got ${args.maxGap}`);
  }
  return args;
}

async function healthOk(url: URL): Promise<boolean> {
  try {
    const response = await fetch(new URL('/health', url.origin));
    return response.ok && (await response.text()) === 'ok';
  } catch {
    return false;
  }
}

async function ensureLocalServer(url: URL) {
  if (await healthOk(url)) return null;
  if (url.hostname !== '127.0.0.1' && url.hostname !== 'localhost') {
    throw new Error(`No server is responding at ${url.origin}; refusing to spawn for non-local URL.`);
  }
  const server = Bun.spawn(
    ['bun', 'scripts/serve_static.ts', '--host', url.hostname, '--port', url.port || '4173'],
    {
      cwd: import.meta.dir.replace(/\/scripts$/, ''),
      stdout: 'inherit',
      stderr: 'inherit',
    },
  );
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (await healthOk(url)) return server;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  server.kill();
  throw new Error(`Timed out waiting for ${url.origin}`);
}

function prepareChromeHome() {
  const chromeHome = process.env.PLAYWRIGHT_CHROME_HOME ?? '/private/tmp/visionary-chrome-home';
  mkdirSync(`${chromeHome}/Library/Application Support/Google/Chrome/Crashpad`, {
    recursive: true,
  });
  mkdirSync(`${chromeHome}/Library/Application Support/Google/Chrome for Testing/Crashpad`, {
    recursive: true,
  });
  mkdirSync('/private/tmp/visionary-chrome-crashpad', { recursive: true });
  process.env.HOME = chromeHome;
  process.env.CFFIXED_USER_HOME = chromeHome;
  process.env.PLAYWRIGHT_CHROME_HOME = chromeHome;
}

function assertBrickMetrics(label: string, metrics: FrameMetrics, args: Args) {
  if (metrics.state.backend !== 'webgpu') {
    throw new Error(`${label}: expected WebGPU backend, got ${metrics.state.backend}`);
  }
  if (!metrics.bands.length) {
    throw new Error(`${label}: no reference brick bands found`);
  }
  for (const band of metrics.bands) {
    const bandLabel = `${label}: frame ${metrics.frame}, rows ${band.start}-${band.end}`;
    if (band.coverage < args.minCoverage) {
      throw new Error(
        `${bandLabel}: brick coverage ${band.coverage.toFixed(4)} is below ${args.minCoverage}`,
      );
    }
    if (band.minRowCoverage < args.minRowCoverage) {
      throw new Error(
        `${bandLabel}: min row coverage ${band.minRowCoverage.toFixed(4)} is below ${args.minRowCoverage}`,
      );
    }
    if (band.largestGap > args.maxGap) {
      throw new Error(
        `${bandLabel}: largest missing run ${band.largestGap}px exceeds ${args.maxGap}px`,
      );
    }
  }
}

async function visibleFrameScreenshot(page: Page) {
  const frame = page
    .locator('#frame:not([hidden]), .frame-preview:not([hidden]), .frame-fallback:not([hidden])')
    .first();
  await frame.waitFor({ state: 'visible', timeout: 30_000 });
  return frame.screenshot({ animations: 'disabled' });
}

async function measureFrame(page: Page, targetFrame: number) {
  const screenshot = await visibleFrameScreenshot(page);
  return page.evaluate(async ({ requestedFrame, screenshotBase64 }) => {
    const debug = (window as any).visionaryDemoDebug;
    const runtime = debug.runtime;
    const expectedTensor = runtime.displayPixels;
    if (!expectedTensor) throw new Error('runtime.displayPixels is unavailable');
    const [frames, height, width, channels] = expectedTensor.dims;
    const referenceFrame = Math.max(
      0,
      Math.min(frames - 1, (runtime.contextManifest.prefix_frames ?? 1) - 1),
    );
    const frameOffset = referenceFrame * height * width * channels;
    const image = new Image();
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error('Failed to decode rendered frame screenshot'));
      image.src = `data:image/png;base64,${screenshotBase64}`;
    });
    const scratch = document.createElement('canvas');
    scratch.width = width;
    scratch.height = height;
    const scratchContext = scratch.getContext('2d', { willReadFrequently: true });
    if (!scratchContext) throw new Error('2D scratch context is unavailable');
    scratchContext.imageSmoothingEnabled = false;
    scratchContext.drawImage(image, 0, 0, width, height);
    const actual = scratchContext.getImageData(0, 0, width, height).data;
    const expectedData = expectedTensor.data;

    const isBrickColor = (red: number, green: number, blue: number) => {
      const max = Math.max(red, green, blue);
      const min = Math.min(red, green, blue);
      return max > 50 && max - min > 18;
    };
    const expectedColored = (x: number, y: number) => {
      const source = frameOffset + (y * width + x) * channels;
      return isBrickColor(expectedData[source], expectedData[source + 1], expectedData[source + 2]);
    };
    const actualColored = (x: number, y: number) => {
      const source = (y * width + x) * 4;
      return isBrickColor(actual[source], actual[source + 1], actual[source + 2]);
    };

    const rowStart = Math.floor(height * 0.2);
    const rowEnd = Math.floor(height * 0.55);
    const minimumExpectedPixels = Math.floor(width * 0.4);
    const rowHasBricks = new Array(height).fill(false);
    for (let y = rowStart; y <= rowEnd; y += 1) {
      let colored = 0;
      for (let x = 0; x < width; x += 1) {
        if (expectedColored(x, y)) colored += 1;
      }
      rowHasBricks[y] = colored > minimumExpectedPixels;
    }

    const bands: BrickBandMetrics[] = [];
    let y = rowStart;
    while (y <= rowEnd) {
      while (y <= rowEnd && !rowHasBricks[y]) y += 1;
      if (y > rowEnd) break;
      const start = y;
      while (y <= rowEnd && rowHasBricks[y]) y += 1;
      const end = y - 1;
      let expectedCount = 0;
      let actualCount = 0;
      let largestGap = 0;
      let minRowCoverage = 1;
      for (let row = start; row <= end; row += 1) {
        let rowExpected = 0;
        let rowActual = 0;
        let currentGap = 0;
        for (let x = 0; x < width; x += 1) {
          if (!expectedColored(x, row)) {
            currentGap = 0;
            continue;
          }
          rowExpected += 1;
          expectedCount += 1;
          if (actualColored(x, row)) {
            rowActual += 1;
            actualCount += 1;
            currentGap = 0;
          } else {
            currentGap += 1;
            largestGap = Math.max(largestGap, currentGap);
          }
        }
        if (rowExpected > 0) {
          minRowCoverage = Math.min(minRowCoverage, rowActual / rowExpected);
        }
      }
      bands.push({
        start,
        end,
        expectedCount,
        actualCount,
        coverage: expectedCount > 0 ? actualCount / expectedCount : 0,
        largestGap,
        minRowCoverage,
      });
    }

    return {
      frame: debug.frameCount || requestedFrame,
      state: {
        backend: runtime.backend,
        cacheLength: runtime.cache.length.data[0],
        contextLength: runtime.contextLength,
        frameCount: debug.frameCount,
        initialCacheSource: runtime.initialCacheSource,
      },
      width,
      height,
      bands,
    } satisfies FrameMetrics;
  }, { requestedFrame: targetFrame, screenshotBase64: screenshot.toString('base64') });
}

async function runFramesThroughUi(page: Page, frames: number) {
  const startButton = page.locator('#start');
  await startButton.click();
  await page.waitForFunction(
    (targetFrame) => ((window as any).visionaryDemoDebug?.frameCount ?? 0) >= targetFrame,
    frames,
    { timeout: 240_000 },
  );
  if ((await startButton.textContent())?.includes('Pause')) {
    await startButton.click();
  }
}

async function validateCase(browser: Browser, args: Args, label: string, url: string) {
  const page = await browser.newPage();
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto(url);
  await page.waitForFunction(
    () => document.querySelector('#status')?.textContent?.includes('Ready'),
    null,
    { timeout: 180_000 },
  );
  console.log(JSON.stringify({ label, status: 'ready', url }, null, 2));
  await runFramesThroughUi(page, args.frames);
  const metrics = await measureFrame(page, args.frames);
  console.log(JSON.stringify({ label, ...metrics }, null, 2));
  assertBrickMetrics(label, metrics, args);
  if (pageErrors.length) {
    throw new Error(`${label}: page errors:\n${pageErrors.join('\n')}`);
  }
  await page.close();
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const firstUrl = new URL(args.urls[0].url);
  const server = await ensureLocalServer(firstUrl);
  prepareChromeHome();
  const browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL ?? 'chrome',
    headless: args.headless,
    args: ['--enable-unsafe-webgpu'],
  });
  const failures: string[] = [];
  try {
    for (const current of args.urls) {
      try {
        await validateCase(browser, args, current.label, current.url);
      } catch (error) {
        const message = error instanceof Error ? error.stack || error.message : String(error);
        failures.push(message);
        console.error(message);
      }
    }
    if (failures.length) {
      throw new Error(`Validation failed for ${failures.length} case(s).`);
    }
    console.log(JSON.stringify({ status: 'passed', frames: args.frames }, null, 2));
  } finally {
    await browser.close();
    server?.kill();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : error);
  process.exit(1);
});
