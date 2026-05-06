#!/usr/bin/env node
import { mkdirSync } from 'node:fs';
import { spawn } from 'node:child_process';

const chromeHome = process.env.PLAYWRIGHT_CHROME_HOME ?? '/private/tmp/visionary-chrome-home';
const maxAttempts = Number.parseInt(process.env.PLAYWRIGHT_BENCHMARK_ATTEMPTS ?? '3', 10);
mkdirSync(`${chromeHome}/Library/Application Support/Google/Chrome/Crashpad`, { recursive: true });

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
    HOME: chromeHome,
    PLAYWRIGHT_CHROME_HOME: chromeHome,
  };
  const browsersPath = process.env.PLAYWRIGHT_BROWSERS_PATH ?? defaultBrowsersPath();
  if (browsersPath) {
    childEnv.PLAYWRIGHT_BROWSERS_PATH = browsersPath;
  }

  const child = spawn('playwright', process.argv.slice(2), {
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
