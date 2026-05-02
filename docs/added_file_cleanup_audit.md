# Added File Cleanup Audit

Date: 2026-05-03

Scope requested: added files from `git diff --name-status main...HEAD`, untracked files from `git status --short`, usage checks with `rg`, and package/config references. No source files were edited.

## Definitely safe to remove

These are not tracked branch additions, are not referenced by package/config/runtime paths, and look like local scratch/reference material rather than repo assets.

- `dreamer4-jax/`
  - Evidence: `git status --short -- .codex dreamer4-jax dreamer4_paper.md` reports `?? dreamer4-jax/`.
  - Evidence: `du -sh dreamer4-jax` reports `10M`.
  - Evidence: `git -C dreamer4-jax remote -v` reports `origin https://github.com/edwhu/dreamer4-jax.git`, so this is a nested external checkout.
  - Evidence: `rg -n --hidden -S "dreamer4-jax" . --glob '!.git/**' --glob '!dreamer4-jax/**' --glob '!uv.lock' --glob '!bun.lock'` produced no source/package/config references outside the checkout itself.
  - Recommendation: do not add it to this repo; remove or move it outside the working tree if it is only a reference checkout.

- `dreamer4_paper.md`
  - Evidence: `git status --short -- .codex dreamer4-jax dreamer4_paper.md` reports `?? dreamer4_paper.md`.
  - Evidence: `du -sh dreamer4_paper.md` reports `80K`.
  - Evidence: `rg -n --hidden -S "dreamer4_paper" . --glob '!.git/**' --glob '!dreamer4-jax/**' --glob '!uv.lock' --glob '!bun.lock'` produced no references.
  - Recommendation: do not add it unless the branch intentionally vendors paper notes.

- `webgpu_app/bench/results/.gitkeep`
  - Evidence: `git diff --name-status main...HEAD` reports it as an added file.
  - Evidence: `webgpu_app/bench/run_webgpu_benchmark.spec.js:73-75` creates `webgpu_app/bench/results` with `mkdir(RESULT_DIR, { recursive: true })` before writing `latest.json`.
  - Evidence: `webgpu_app/bench/run_webgpu_profile_diagnostic.spec.js:58-60` creates the same result directory before writing diagnostic JSON.
  - Evidence: `rg -n --hidden -S "\\.gitkeep|RESULT_DIR|mkdir\\(RESULT_DIR|results/\\.gitkeep" webgpu_app/bench scripts/webgpu/*.mjs package.json playwright.config.js` found no `.gitkeep` consumer.
  - Recommendation: safe to remove if the repo does not need empty output directories visible in fresh clones.

Aside from `.gitkeep`, I did not find a tracked added product/runtime file that is definitely unused. The tracked additions are either wired into package/config entry points, imported by export/test code, or are diagnostic docs/scripts whose removal is a repo-policy decision.

## Probably keep

- `package.json`, `bun.lock`, `playwright.config.js`
  - Evidence: `package.json:4-14` defines the WebGPU benchmark/demo scripts.
  - Evidence: `package.json:17` adds `onnxruntime-web`; `package.json:20` adds `@playwright/test`.
  - Evidence: `playwright.config.js:33-35` starts `node scripts/webgpu/serve_static.mjs --host 127.0.0.1 --port 4173`.
  - Evidence: `webgpu_app/bench/README.md:18-22` documents the package scripts.

- `webgpu_app/bench/**` and `webgpu_app/demo/**`, excluding `webgpu_app/bench/results/.gitkeep`
  - Evidence: `package.json:4-12` invokes `webgpu_app/bench/run_webgpu_benchmark.spec.js` and `webgpu_app/bench/run_webgpu_profile_diagnostic.spec.js`.
  - Evidence: `package.json:14` invokes `webgpu_app/demo/run_demo_smoke.spec.js`.
  - Evidence: `webgpu_app/bench/index.html:10` loads `./benchmark.js`; `webgpu_app/bench/profile_diagnostic.html:10` loads `./profile_diagnostic.js`.
  - Evidence: `webgpu_app/demo/index.html:7` loads `./styles.css`; `webgpu_app/demo/index.html:57` loads `./main.js`; `webgpu_app/demo/main.js:2` imports `./jax_noise.js`.
  - Evidence: `webgpu_app/bench/run_webgpu_benchmark.spec.js:5-7` writes `webgpu_app/bench/results/latest.json`; `webgpu_app/bench/run_webgpu_profile_diagnostic.spec.js:5-8` writes profile result JSONs.

- `scripts/webgpu/check_webgpu_benchmark.py`
  - Evidence: `package.json:5` calls it in `benchmark:webgpu:ci`.
  - Evidence: it compares result and baseline arguments at `scripts/webgpu/check_webgpu_benchmark.py:10-15`.

- `scripts/webgpu/serve_static.mjs`
  - Evidence: `package.json:13` and `playwright.config.js:34` call it.
  - Evidence: it serves ONNX/WebGPU-safe headers at `scripts/webgpu/serve_static.mjs:56-61`.

- `scripts/webgpu/export_dreamer4_onnx.py` and `scripts/webgpu/simplify_onnx_file.py`
  - Evidence: `webgpu_app/bench/README.md:46-59` documents the export command.
  - Evidence: `scripts/webgpu/export_dreamer4_onnx.py:24-38` imports `visionary.export.onnx_wrappers`.
  - Evidence: `scripts/webgpu/export_dreamer4_onnx.py:118-132` defines `--simplify_onnx` / `--simplify_demo_only`.
  - Evidence: `scripts/webgpu/export_dreamer4_onnx.py:383-385` invokes sibling helper `simplify_onnx_file.py`.
  - Evidence: `pyproject.toml:35-41` defines the `onnx` dependency group used by this exporter.

- `visionary/export/__init__.py`, `visionary/export/onnx_wrappers.py`, and `tests/test_onnx_attention_export.py`
  - Evidence: `scripts/webgpu/export_dreamer4_onnx.py:24-38` imports the wrappers.
  - Evidence: `scripts/webgpu/compare_jax_onnx_rollout.py:18-21` imports wrapper functions.
  - Evidence: `tests/test_onnx_attention_export.py:5` imports `_export_dot_product_attention`.
  - Evidence: `tests/test_onnx_attention_export.py:8-30`, `:33-58`, and `:61-86` validate grouped-GQA parity.

- `webgpu_app/bench/baselines/webgpu_benchmark_baseline.json`
  - Evidence: `package.json:5` passes it to `check_webgpu_benchmark.py`.
  - Evidence: `webgpu_app/bench/README.md:158-163` documents the baseline policy.

- `scripts/webgpu/summarize_webgpu_profile.mjs` and `scripts/webgpu/summarize_ort_session_profile.mjs`
  - Evidence: `package.json:10-11` calls both scripts.
  - Evidence: `webgpu_app/bench/README.md:142-146` documents profile summary usage.

- `.gitignore` generated artifact rules
  - Evidence: `.gitignore:37-44` ignores generated ONNX assets, demo context artifacts, raw assets, and benchmark JSONs.
  - Evidence: `.codex/onnx_webgpu_progress.md:279-288` explicitly says generated ONNX assets and benchmark JSON outputs are not suitable for normal commits, while source changes/tooling and `.codex` summaries are the reviewable history.

## Unclear/risky

These look removable from an automated build/runtime perspective, but they may be valuable for reproducibility, validation, handoff, or branch history. I would not remove them without deciding what audit/debug history belongs in the repo.

- Added tracked `.codex/*.md` notes:
  - `.codex/attention_webgpu_operator_note.md`
  - `.codex/export_conversion_cleanup_audit.md`
  - `.codex/fused_attention_path.md`
  - `.codex/onnx_current_graph_status.md`
  - `.codex/onnx_export_implementation_plan.md`
  - `.codex/onnx_export_research.md`
  - `.codex/onnx_graph_rewrite_path.md`
  - `.codex/onnx_webgpu_graph_analysis.md`
  - `.codex/onnx_webgpu_next_rewrites.md`
  - `.codex/onnx_webgpu_progress.md`
  - `.codex/onnx_webgpu_research.md`
  - `.codex/onnx_webgpu_static_shapes_research.md`
  - Evidence: `git diff --name-status main...HEAD` reports all of these as `A`.
  - Evidence: `rg -n --hidden -S "^(#|Scope:|Date:|Goal:)" .codex/*.md` shows they are analysis/progress/research notes, not code entry points.
  - Evidence: `.codex/attention_webgpu_operator_note.md:24` references other `.codex` notes, so some notes cross-link.
  - Evidence: `.codex/onnx_webgpu_progress.md:279-288` says `.codex` result summaries are intended as reviewable history.
  - Cleanup read: safe for runtime/build removal, risky for handoff/history removal.

- Untracked `.codex/current_hot_graph_candidates.md`
  - Evidence: `git status --short -- .codex dreamer4-jax dreamer4_paper.md` reports `?? .codex/current_hot_graph_candidates.md`.
  - Evidence: `.codex/current_hot_graph_candidates.md:1-7` identifies it as optimization analysis for the current browser Dreamer4 hot path.
  - Evidence: `.codex/current_hot_graph_candidates.md:37-45` includes freshness warnings about benchmark/manifest timestamps.
  - Cleanup read: not required by package/config, but likely useful if continuing ONNX/WebGPU optimization work.

- Untracked `.codex/next_optimization_candidates.md`
  - Evidence: `git status --short -- .codex dreamer4-jax dreamer4_paper.md` reports `?? .codex/next_optimization_candidates.md`.
  - Evidence: `.codex/next_optimization_candidates.md:5` says production files were not edited.
  - Evidence: `.codex/next_optimization_candidates.md:120-135` recommends the next implementation path.
  - Cleanup read: not required by package/config, but useful as planning context.

- Untracked `.codex/temporal_gqa_export_review.md`
  - Evidence: `git status --short -- .codex dreamer4-jax dreamer4_paper.md` reports `?? .codex/temporal_gqa_export_review.md`.
  - Evidence: `.codex/temporal_gqa_export_review.md:5-11` says it reviews existing GQA export paths.
  - Evidence: `.codex/temporal_gqa_export_review.md:109-113` gives the bottom-line recommendation.
  - Evidence: `.codex/attention_webgpu_operator_note.md:24` references `.codex/temporal_gqa_export_review.md`.
  - Cleanup read: not required by package/config, but cross-referenced by a tracked note.

- Untracked `.codex/exporter_refactor_audit.md`
  - Evidence: `git status --short -- .codex` reports `?? .codex/exporter_refactor_audit.md`.
  - Evidence: `.codex/exporter_refactor_audit.md:1-7` identifies it as a read-through/`rg`/ruff audit of the exporter and ONNX wrappers.
  - Evidence: `.codex/exporter_refactor_audit.md:61-67` contains suggested cleanup order.
  - Cleanup read: not required by package/config, but likely useful as a refactor audit.

- Untracked `.codex/webgpu_support_cleanup_audit.md`
  - Evidence: `git status --short -- .codex` reports `?? .codex/webgpu_support_cleanup_audit.md`.
  - Evidence: `.codex/webgpu_support_cleanup_audit.md:1-5` identifies it as an audit of `webgpu_app/**`, `scripts/webgpu/*.mjs`, and package scripts.
  - Evidence: `.codex/webgpu_support_cleanup_audit.md:40-54` independently flags `webgpu_app/bench/results/.gitkeep` as unnecessary.
  - Cleanup read: not required by package/config, but useful as support-cleanup context.

- `scripts/webgpu/create_demo_context.py`
  - Evidence: no package script invokes it.
  - Evidence: it creates `breakout_demo_context` by default at `scripts/webgpu/create_demo_context.py:21-42`; the browser demo consumes `webgpu_app/assets/breakout_demo_context.json` at `webgpu_app/demo/main.js:6`.
  - Evidence: `.gitignore:42` ignores generated `webgpu_app/assets/breakout_demo_context.*`.
  - Cleanup read: keep or document it if the demo context needs to be reproducible; otherwise it is a manual generator with no automated entry point.

- `scripts/webgpu/compare_jax_onnx_rollout.py`
  - Evidence: no package script invokes it.
  - Evidence: it imports model/export helpers at `scripts/webgpu/compare_jax_onnx_rollout.py:13-23`.
  - Evidence: it writes `webgpu_app/bench/results/rollout_compare.json` by default at `scripts/webgpu/compare_jax_onnx_rollout.py:39`.
  - Cleanup read: manual validation tool. Removable only if rollout parity diagnostics are intentionally out of scope.

- `scripts/webgpu/compare_raw_optimized_onnx.py`
  - Evidence: no package script invokes it.
  - Evidence: it is tied to exporter raw snapshots by `scripts/webgpu/export_dreamer4_onnx.py:80-88` and `scripts/webgpu/compare_raw_optimized_onnx.py:211-214`.
  - Evidence: `.codex/current_hot_graph_candidates.md:28-35` and `.codex/onnx_webgpu_progress.md:516-523` cite raw-vs-optimized accuracy as a gate.
  - Cleanup read: probably keep if graph rewrite/export safety matters, but it is not automated.

- `scripts/webgpu/verify_entry_cache_update.py`
  - Evidence: no package script invokes it.
  - Evidence: it verifies entry-cache vs full-cache equivalence at `scripts/webgpu/verify_entry_cache_update.py:17-35` and compares cache reconstruction at `scripts/webgpu/verify_entry_cache_update.py:163-188`.
  - Evidence: `.codex/onnx_webgpu_progress.md:505-515` says this was added for the entry-cache artifact, and `.codex/onnx_webgpu_progress.md:516-523` records passing accuracy.
  - Cleanup read: probably keep with the entry-cache runtime ABI, but it is currently a manual gate.

## Exact evidence references

- Added tracked files: `git diff --name-status main...HEAD` reports added files across `.codex/`, `package.json`, `playwright.config.js`, `scripts/webgpu/`, `tests/test_onnx_attention_export.py`, `visionary/export/`, and `webgpu_app/`.
- Untracked files: `git status --short -- .codex dreamer4-jax dreamer4_paper.md` reports:
  - `?? .codex/current_hot_graph_candidates.md`
  - `?? .codex/exporter_refactor_audit.md`
  - `?? .codex/next_optimization_candidates.md`
  - `?? .codex/temporal_gqa_export_review.md`
  - `?? .codex/webgpu_support_cleanup_audit.md`
  - `?? dreamer4-jax/`
  - `?? dreamer4_paper.md`
- Package/config references:
  - `package.json:4-14`
  - `package.json:17`
  - `package.json:20`
  - `playwright.config.js:33-35`
  - `pyproject.toml:35-41`
  - `.gitignore:37-44`
- Browser/demo linkage:
  - `webgpu_app/bench/index.html:10`
  - `webgpu_app/bench/profile_diagnostic.html:10`
  - `webgpu_app/demo/index.html:7`
  - `webgpu_app/demo/index.html:57`
  - `webgpu_app/demo/main.js:2`
  - `webgpu_app/demo/main.js:6`
- Export/test linkage:
  - `scripts/webgpu/export_dreamer4_onnx.py:24-38`
  - `scripts/webgpu/export_dreamer4_onnx.py:80-88`
  - `scripts/webgpu/export_dreamer4_onnx.py:118-132`
  - `scripts/webgpu/export_dreamer4_onnx.py:383-385`
  - `tests/test_onnx_attention_export.py:5`
- Manual diagnostic linkage:
  - `scripts/webgpu/create_demo_context.py:21-42`
  - `scripts/webgpu/compare_jax_onnx_rollout.py:39`
  - `scripts/webgpu/compare_raw_optimized_onnx.py:211-214`
  - `scripts/webgpu/verify_entry_cache_update.py:17-35`
  - `scripts/webgpu/verify_entry_cache_update.py:163-188`
- `.codex` note evidence:
  - `.codex/onnx_webgpu_progress.md:279-288`
  - `.codex/attention_webgpu_operator_note.md:24`
  - `.codex/current_hot_graph_candidates.md:1-7`
  - `.codex/exporter_refactor_audit.md:1-7`
  - `.codex/next_optimization_candidates.md:5`
  - `.codex/temporal_gqa_export_review.md:5-11`
  - `.codex/webgpu_support_cleanup_audit.md:1-5`
