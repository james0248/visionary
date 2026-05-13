import { rmSync } from 'node:fs';

const generatedPaths = [
  'bench/benchmark.js',
  'bench/benchmark.js.map',
  'bench/results',
  'demo/main.js',
  'demo/main.js.map',
  'dist',
  'playwright-report',
  'test-results',
];

for (const path of generatedPaths) {
  rmSync(path, { recursive: true, force: true });
}

console.log(`Removed ${generatedPaths.length} generated browser output paths.`);
