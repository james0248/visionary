export {};

declare global {
  interface Window {
    VISIONARY_DEMO_CONFIG?: Record<string, string | number | boolean>;
    visionaryDemoDebug?: unknown;
    __WEBGPU_BENCHMARK_RESULT__?: unknown;
  }
}
