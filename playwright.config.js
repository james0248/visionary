import { defineConfig, devices } from '@playwright/test';

const headless = process.env.PLAYWRIGHT_HEADLESS === '1';
const browserChannel = process.env.PLAYWRIGHT_CHANNEL;
const crashpadArgs =
  process.env.PLAYWRIGHT_CRASHPAD_ARGS === '1'
    ? [
        '--disable-crash-reporter',
        '--disable-crashpad',
        '--disable-crashpad-handler-for-testing',
        '--disable-crashpad-for-testing',
        '--crash-dumps-dir=/private/tmp/visionary-chrome-crashpad',
      ]
    : [];

export default defineConfig({
  testDir: '.',
  timeout: 900_000,
  expect: {
    timeout: 30_000,
  },
  use: {
    baseURL: 'http://127.0.0.1:4173',
    browserName: 'chromium',
    ...(browserChannel === 'bundled' ? {} : { channel: browserChannel ?? 'chrome' }),
    headless,
    launchOptions: {
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
    command: 'node scripts/webgpu/serve_static.mjs --host 127.0.0.1 --port 4173',
    url: 'http://127.0.0.1:4173/health',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
