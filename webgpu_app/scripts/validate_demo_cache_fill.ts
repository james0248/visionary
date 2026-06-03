#!/usr/bin/env bun
import { mkdirSync } from 'node:fs';
import { chromium } from '@playwright/test';

type Args = {
  frames: number;
  headless: boolean;
  url: string;
};

const DEFAULT_URL =
  'http://127.0.0.1:4173/demo/index.html?backend=webgpu&assetBase=/dream_arcade_assets/breakout';

function parseArgs(argv: string[]): Args {
  const args: Args = {
    frames: 62,
    headless: process.env.PLAYWRIGHT_HEADLESS === '1',
    url: DEFAULT_URL,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const [key, inlineValue] = arg.split('=', 2);
    const value = inlineValue ?? argv[index + 1];
    if (key === '--url') {
      args.url = value ?? args.url;
      if (inlineValue == null) index += 1;
    } else if (key === '--frames') {
      args.frames = Number.parseInt(value ?? String(args.frames), 10);
      if (inlineValue == null) index += 1;
    } else if (key === '--headless') {
      args.headless = true;
      if (inlineValue == null && value === '1') index += 1;
    } else if (key === '--headed') {
      args.headless = false;
    }
  }
  if (!Number.isInteger(args.frames) || args.frames <= 0) {
    throw new Error(`--frames must be a positive integer, got ${args.frames}`);
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
    [
      'bun',
      'scripts/serve_static.ts',
      '--host',
      url.hostname,
      '--port',
      url.port || '4173',
    ],
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

function checkpointSet(initialLength: number, contextLength: number, frames: number) {
  const framesToFull = Math.max(0, contextLength - initialLength);
  return new Set(
    [1, 2, 4, Math.max(1, Math.floor(framesToFull / 2)), framesToFull, framesToFull + 1, frames]
      .filter((value) => value >= 1 && value <= frames),
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const url = new URL(args.url);
  const server = await ensureLocalServer(url);
  prepareChromeHome();

  const browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL ?? 'chrome',
    headless: args.headless,
    args: ['--enable-unsafe-webgpu'],
  });
  try {
    const page = await browser.newPage();
    page.on('pageerror', (error) => {
      throw error;
    });
    await page.goto(url.href);
    await page.waitForFunction(
      () => document.querySelector('#status')?.textContent?.includes('Ready'),
      null,
      { timeout: 180_000 },
    );
    const initial = await page.evaluate(() => {
      const runtime = (window as any).visionaryDemoDebug.runtime;
      return {
        backend: runtime.backend,
        cacheLength: runtime.cache.length.data[0],
        contextLength: runtime.contextLength,
        initialCacheSource: runtime.initialCacheSource,
        step: runtime.specs.step?.name,
        fullStep: runtime.specs.fullStep?.name ?? null,
      };
    });
    if (initial.backend !== 'webgpu') {
      throw new Error(`Expected WebGPU backend, got ${initial.backend}`);
    }
    const checkpoints = checkpointSet(initial.cacheLength, initial.contextLength, args.frames);
    console.log(
      JSON.stringify(
        {
          status: 'started',
          url: url.href,
          frames: args.frames,
          initial,
        },
        null,
        2,
      ),
    );

    for (let frame = 1; frame <= args.frames; frame += 1) {
      const result = await page.evaluate(() =>
        (window as any).visionaryDemoDebug.generateFrame({ debugCacheUpdate: true }),
      );
      if (!result?.passed) {
        throw new Error(
          `Cache update diverged at frame ${frame}:\n${JSON.stringify(result, null, 2)}`,
        );
      }
      const expectedLength = Math.min(initial.cacheLength + frame, initial.contextLength);
      if (result.cacheLengthAfter !== expectedLength) {
        throw new Error(
          `Unexpected cache length at frame ${frame}: got ${result.cacheLengthAfter}, expected ${expectedLength}`,
        );
      }
      if (checkpoints.has(frame)) {
        console.log(
          JSON.stringify(
            {
              frame,
              cacheLengthBefore: result.cacheLengthBefore,
              cacheLengthAfter: result.cacheLengthAfter,
              kMaxAbs: result.k.maxAbs,
              vMaxAbs: result.v.maxAbs,
            },
            null,
            2,
          ),
        );
      }
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
