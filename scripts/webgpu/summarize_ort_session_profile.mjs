import { readFile, writeFile } from 'node:fs/promises';

const inputPath = process.argv[2] ?? 'webgpu_app/bench/results/profile_diagnostic_latest.json';
const outputPath = process.argv[3] ?? 'webgpu_app/bench/results/session_profile_summary.json';

function parseTraceFromConsoleMessages(messages) {
  const texts = messages.map((entry) => (typeof entry === 'string' ? entry : entry.text ?? ''));
  const start = texts.findIndex((text) => text.trim() === '[');
  if (start < 0) {
    throw new Error('Could not find ORT profile JSON start marker "[" in console messages');
  }
  let end = texts.findIndex((text, index) => index > start && text.trim() === ']');
  if (end < 0) end = texts.length;
  const jsonText = texts.slice(start, end + 1).join('\n').replace(/,\s*\]/g, ']');
  return JSON.parse(jsonText);
}

function groupBy(events, keyFn) {
  const groups = new Map();
  for (const event of events) {
    const key = keyFn(event);
    const entry = groups.get(key) ?? {
      key,
      count: 0,
      total_us: 0,
      max_us: 0,
      examples: [],
      node_indexes: new Set(),
    };
    const dur = event.dur ?? 0;
    entry.count += 1;
    entry.total_us += dur;
    entry.max_us = Math.max(entry.max_us, dur);
    if (event.args?.node_index != null) entry.node_indexes.add(String(event.args.node_index));
    if (entry.examples.length < 5) entry.examples.push(event.name);
    groups.set(key, entry);
  }
  return [...groups.values()]
    .map((entry) => ({
      ...entry,
      total_ms: entry.total_us / 1000,
      mean_ms: entry.count === 0 ? 0 : entry.total_us / entry.count / 1000,
      max_ms: entry.max_us / 1000,
      node_indexes: [...entry.node_indexes].sort((a, b) => Number(a) - Number(b)),
    }))
    .sort((a, b) => b.total_us - a.total_us);
}

function compactNode(event) {
  return {
    name: event.name,
    op_name: event.args?.op_name ?? null,
    provider: event.args?.provider ?? null,
    node_index: event.args?.node_index ?? null,
    dur_ms: (event.dur ?? 0) / 1000,
    output_size: event.args?.output_size ?? null,
    activation_size: event.args?.activation_size ?? null,
    parameter_size: event.args?.parameter_size ?? null,
    input_type_shape: event.args?.input_type_shape ?? null,
    output_type_shape: event.args?.output_type_shape ?? null,
  };
}

const diagnostic = JSON.parse(await readFile(inputPath, 'utf8'));
const trace = parseTraceFromConsoleMessages(diagnostic.session_profiling_console_messages ?? []);
const nodes = trace.filter((event) => event.cat === 'Node');
const sessions = trace.filter((event) => event.cat === 'Session');
const cpuNodes = nodes.filter((event) => event.args?.provider === 'CPUExecutionProvider');
const webgpuNodes = nodes.filter((event) => event.args?.provider === 'WebGpuExecutionProvider');
const memcpyNodes = nodes.filter((event) => String(event.args?.op_name ?? '').startsWith('Memcpy'));

const summary = {
  schema_version: 1,
  source: inputPath,
  created_at: new Date().toISOString(),
  diagnostic: {
    status: diagnostic.status,
    ort_version: diagnostic.ort_version,
    timestamp_query: diagnostic.timestamp_query,
    conclusion: diagnostic.conclusion,
    session_profiling_trials: diagnostic.session_profiling_trials,
  },
  counts: {
    trace_events: trace.length,
    session_events: sessions.length,
    node_events: nodes.length,
    webgpu_node_events: webgpuNodes.length,
    cpu_node_events: cpuNodes.length,
    memcpy_node_events: memcpyNodes.length,
  },
  totals: {
    providers: groupBy(nodes, (event) => event.args?.provider ?? 'unknown'),
    ops: groupBy(nodes, (event) => `${event.args?.provider ?? 'unknown'}|${event.args?.op_name ?? 'unknown'}`),
    cpu_ops: groupBy(cpuNodes, (event) => event.args?.op_name ?? 'unknown'),
    memcpy_ops: groupBy(memcpyNodes, (event) => `${event.args?.provider ?? 'unknown'}|${event.args?.op_name ?? 'unknown'}`),
  },
  top_nodes: nodes
    .slice()
    .sort((a, b) => (b.dur ?? 0) - (a.dur ?? 0))
    .slice(0, 50)
    .map(compactNode),
  top_cpu_nodes: cpuNodes
    .slice()
    .sort((a, b) => (b.dur ?? 0) - (a.dur ?? 0))
    .slice(0, 50)
    .map(compactNode),
  top_memcpy_nodes: memcpyNodes
    .slice()
    .sort((a, b) => (b.dur ?? 0) - (a.dur ?? 0))
    .slice(0, 50)
    .map(compactNode),
};

await writeFile(outputPath, `${JSON.stringify(summary, null, 2)}\n`);

console.log(`Wrote ${outputPath}`);
console.log(`Node events: ${summary.counts.node_events}`);
for (const provider of summary.totals.providers) {
  console.log(
    `${provider.key}: ${provider.count} events, ${provider.total_ms.toFixed(3)} ms total, ${provider.mean_ms.toFixed(3)} ms mean`,
  );
}
console.log('Top ops:');
for (const op of summary.totals.ops.slice(0, 10)) {
  console.log(
    `  ${op.key}: ${op.count} events, ${op.total_ms.toFixed(3)} ms total, ${op.mean_ms.toFixed(3)} ms mean`,
  );
}
