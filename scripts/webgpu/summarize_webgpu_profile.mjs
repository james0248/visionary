import { readFile } from 'node:fs/promises';

const path = process.argv[2] ?? 'webgpu_app/bench/results/latest.json';
const raw = await readFile(path, 'utf8');
const result = JSON.parse(raw);
const profiling = result.profiling;

if (!profiling) {
  console.log(`No profiling block in ${path}`);
  process.exit(0);
}

console.log(`file: ${path}`);
console.log(`status: ${result.status}`);
console.log(`profiling enabled: ${profiling.enabled}`);
console.log(`profiling available: ${profiling.available}`);
if (profiling.reason) console.log(`reason: ${profiling.reason}`);
console.log(`scopes: ${profiling.scopes?.length ?? 0}`);
console.log(`raw events: ${profiling.raw_events?.length ?? 0}`);
console.log(`late events: ${profiling.late_events?.length ?? 0}`);
console.log(`unscoped events: ${profiling.unscoped_events?.length ?? 0}`);

if (!profiling.available || !profiling.raw_events?.length) {
  console.log('No ORT WebGPU kernel events are available to summarize.');
  process.exit(0);
}

const frameMs =
  result.results
    ?.find((entry) => entry.mode === 'streaming_frame')
    ?.timing?.streaming_frame?.mean_ms ?? null;

for (const [role, summary] of Object.entries(profiling.summary?.by_role ?? {})) {
  console.log('');
  console.log(`${role}: ${summary.event_count} events, ${summary.total_ms.toFixed(3)} ms total`);
  for (const kernel of summary.top_kernels ?? []) {
    const pct = frameMs == null ? null : (kernel.total_ms / frameMs) * 100;
    const pctText = pct == null ? '' : `, ${pct.toFixed(1)}% of frame`;
    console.log(
      `  ${kernel.total_ms.toFixed(3)} ms total, ${kernel.mean_ms.toFixed(3)} ms mean, ${kernel.event_count}x${pctText}: ${kernel.key}`,
    );
  }
}
