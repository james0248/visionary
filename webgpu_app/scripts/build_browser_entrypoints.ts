import { rmSync } from 'node:fs';
import { join, resolve } from 'node:path';

type BrowserBuildEntry = {
  entrypoint: string;
  outdir: string;
  outfile: string;
};

const defaultEntries: BrowserBuildEntry[] = [
  {
    entrypoint: 'demo/main.ts',
    outdir: 'demo',
    outfile: 'demo/main.js',
  },
];

function cleanOutput(outfile: string) {
  rmSync(resolve(outfile), { force: true });
  rmSync(resolve(`${outfile}.map`), { force: true });
}

async function buildEntry(entry: BrowserBuildEntry) {
  cleanOutput(entry.outfile);
  const result = await Bun.build({
    entrypoints: [resolve(entry.entrypoint)],
    outdir: resolve(entry.outdir),
    target: 'browser',
    format: 'esm',
    splitting: false,
    sourcemap: 'none',
    minify: false,
    naming: '[name].[ext]',
  });

  if (!result.success) {
    for (const log of result.logs) {
      console.error(log);
    }
    throw new Error(`Browser build failed for ${entry.entrypoint}`);
  }
}

export async function buildBrowserEntrypoints(entries: BrowserBuildEntry[] = defaultEntries) {
  for (const entry of entries) {
    await buildEntry(entry);
  }
}

export async function buildDemoBrowserBundle(outDir: string) {
  await buildEntry({
    entrypoint: 'demo/main.ts',
    outdir: outDir,
    outfile: join(outDir, 'main.js'),
  });
}

if (import.meta.main) {
  await buildBrowserEntrypoints();
}
