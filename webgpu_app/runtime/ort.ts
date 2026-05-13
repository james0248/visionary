export function configureOrt(
  ort,
  {
    wasmPaths = null,
    wasmNumThreads = null,
    webgpuPowerPreference = 'high-performance',
  } = {},
) {
  ort.env.wasm ??= {};
  if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths;
  if (wasmNumThreads != null) ort.env.wasm.numThreads = wasmNumThreads;
  ort.env.webgpu ??= {};
  ort.env.webgpu.powerPreference = webgpuPowerPreference;
}
