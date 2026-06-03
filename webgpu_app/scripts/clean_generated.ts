import { rmSync } from 'node:fs';

const generatedPaths = [
  '__pycache__',
  'bench/benchmark.js',
  'bench/benchmark.js.map',
  'bench/__pycache__',
  'bench/results',
  'demo/main.js',
  'demo/main.js.map',
  'dist',
  'export/__pycache__',
  'playwright-report',
  'test-results',
];

for (const path of generatedPaths) {
  rmSync(path, { recursive: true, force: true });
}

console.log(`Removed ${generatedPaths.length} generated browser output paths.`);
