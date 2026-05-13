import { defineConfig, devices } from '@playwright/test';
import { mkdirSync } from 'node:fs';

const headless = process.env.PLAYWRIGHT_HEADLESS === '1';
const browserChannel = process.env.PLAYWRIGHT_CHANNEL;
const chromeHome = process.env.PLAYWRIGHT_CHROME_HOME ?? '/private/tmp/visionary-chrome-home';
mkdirSync(`${chromeHome}/Library/Application Support/Google/Chrome/Crashpad`, { recursive: true });
mkdirSync(`${chromeHome}/Library/Application Support/Google/Chrome for Testing/Crashpad`, {
  recursive: true,
});
mkdirSync('/private/tmp/visionary-chrome-crashpad', { recursive: true });

const crashpadArgs = [
  '--disable-crash-reporter',
  '--disable-crashpad',
  '--disable-crashpad-handler-for-testing',
  '--disable-crashpad-for-testing',
  '--crash-dumps-dir=/private/tmp/visionary-chrome-crashpad',
];

export default defineConfig({
  testDir: '.',
  timeout: 900_000,
  expect: {
    timeout: 30_000,
  },
  use: {
    baseURL: 'http://127.0.0.1:4173',
    browserName: 'chromium',
    ...(browserChannel && browserChannel !== 'bundled' ? { channel: browserChannel } : {}),
    headless,
    launchOptions: {
      env: {
        ...process.env,
        HOME: chromeHome,
        CFFIXED_USER_HOME: chromeHome,
      },
      args: [
        '--enable-unsafe-webgpu',
        '--disable-dawn-features=disallow_unsafe_apis',
        '--ignore-gpu-blocklist',
        '--enable-gpu-rasterization',
        '--disable-gpu-sandbox',
        ...crashpadArgs,
      ],
    },
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'bun scripts/serve_static.ts --host 127.0.0.1 --port 4173',
    url: 'http://127.0.0.1:4173/health',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
