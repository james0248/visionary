from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


MANIFEST_NAME = "breakout_onnx_manifest.json"
RAW_MANIFEST_NAME = "raw_onnx_artifacts_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw jax2onnx artifacts against optimized WebGPU artifacts with "
            "deterministic ONNX Runtime CPU inputs."
        )
    )
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--optimized_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument(
        "--artifact",
        action="append",
        default=None,
        help="Artifact name to compare. Defaults to the demo prefill/step/decode_z exports.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("webgpu_app/bench/results/raw_optimized_onnx_accuracy.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(args: argparse.Namespace) -> Path:
    return args.manifest if args.manifest is not None else args.optimized_dir / MANIFEST_NAME


def artifact_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        entry["name"]: entry
        for entry in manifest.get("exports", [])
        if isinstance(entry, dict) and entry.get("name")
    }


def default_artifacts(manifest: dict[str, Any]) -> list[str]:
    demo = manifest.get("demo_generation", {})
    names = [
        demo.get("preferred_prefill_export"),
        demo.get("preferred_step_export"),
        demo.get("preferred_steady_state_step_export"),
        demo.get("experimental_layer_prefill_export"),
        demo.get("experimental_layer_steady_state_step_export"),
    ]
    decode_z = demo.get("decode_z")
    if isinstance(decode_z, dict):
        names.append(decode_z.get("export"))
    names.append("breakout_tokenizer_decode_z_b1_t1")
    seen = set()
    return [name for name in names if name and not (name in seen or seen.add(name))]


def make_float(shape: list[int], rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=0.5, size=tuple(shape)).astype(np.float32)


def make_int(
    name: str, shape: list[int], manifest: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    dynamics = manifest.get("dynamics", {})
    context_length = int(dynamics.get("context_length", 64))
    max_step_size = int(dynamics.get("max_step_size", 6))
    num_actions = int(dynamics.get("num_actions", 4))
    if name == "actions":
        return rng.integers(0, num_actions, size=tuple(shape), dtype=np.int32)
    if name == "step_levels":
        return np.full(tuple(shape), min(2, max_step_size - 1), dtype=np.int32)
    if name == "signal_levels":
        return np.zeros(tuple(shape), dtype=np.int32)
    if name in {"position_index", "cache_length"}:
        return np.full(tuple(shape), context_length, dtype=np.int32)
    return np.zeros(tuple(shape), dtype=np.int32)


def make_feed(
    name: str,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    dtype = spec.get("dtype")
    shape = [int(dim) for dim in spec.get("shape", [])]
    if dtype == "float32":
        return make_float(shape, rng)
    if dtype == "float16":
        return make_float(shape, rng).astype(np.float16)
    if dtype == "int32":
        return make_int(name, shape, manifest, rng)
    raise ValueError(f"Unsupported input dtype for {name}: {dtype!r}")


def run_ort(path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, feeds)
    return dict(zip(output_names, outputs))


def compare_arrays(
    raw: np.ndarray,
    optimized: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    raw_f = raw.astype(np.float32) if raw.dtype.kind in {"f", "i", "u", "b"} else raw
    opt_f = (
        optimized.astype(np.float32) if optimized.dtype.kind in {"f", "i", "u", "b"} else optimized
    )
    delta = opt_f - raw_f
    abs_delta = np.abs(delta)
    rel_delta = abs_delta / np.maximum(np.abs(raw_f), np.asarray(1e-12, dtype=np.float32))
    passed = bool(np.allclose(optimized, raw, atol=atol, rtol=rtol))
    return {
        "shape": list(raw.shape),
        "raw_dtype": str(raw.dtype),
        "optimized_dtype": str(optimized.dtype),
        "max_abs_error": float(np.max(abs_delta)) if abs_delta.size else 0.0,
        "mean_abs_error": float(np.mean(abs_delta)) if abs_delta.size else 0.0,
        "max_rel_error": float(np.max(rel_delta)) if rel_delta.size else 0.0,
        "mean_rel_error": float(np.mean(rel_delta)) if rel_delta.size else 0.0,
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
    }


def compare_artifact(
    *,
    name: str,
    entry: dict[str, Any],
    raw_dir: Path,
    optimized_dir: Path,
    manifest: dict[str, Any],
    seed: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    raw_path = raw_dir / entry["path"]
    optimized_path = optimized_dir / entry["path"]
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw artifact for {name}: {raw_path}")
    if not optimized_path.exists():
        raise FileNotFoundError(f"Missing optimized artifact for {name}: {optimized_path}")

    rng = np.random.default_rng(seed)
    feeds = {
        input_name: make_feed(input_name, input_spec, manifest, rng)
        for input_name, input_spec in entry.get("inputs", {}).items()
    }
    raw_outputs = run_ort(raw_path, feeds)
    optimized_outputs = run_ort(optimized_path, feeds)
    output_names = sorted(raw_outputs)
    comparisons = {
        output_name: compare_arrays(
            raw_outputs[output_name],
            optimized_outputs[output_name],
            atol=atol,
            rtol=rtol,
        )
        for output_name in output_names
    }
    return {
        "name": name,
        "raw_path": raw_path.as_posix(),
        "optimized_path": optimized_path.as_posix(),
        "raw_sha256": sha256_file(raw_path),
        "optimized_sha256": sha256_file(optimized_path),
        "passed": all(result["passed"] for result in comparisons.values()),
        "outputs": comparisons,
    }


def main() -> int:
    args = parse_args()
    manifest = load_json(manifest_path(args))
    if not (args.raw_dir / RAW_MANIFEST_NAME).exists():
        raise FileNotFoundError(
            f"{args.raw_dir / RAW_MANIFEST_NAME} does not exist. Re-export with --raw_out_dir."
        )
    exports = artifact_map(manifest)
    artifacts = args.artifact or default_artifacts(manifest)
    missing = [name for name in artifacts if name not in exports]
    if missing:
        raise KeyError(f"Artifacts not present in manifest: {missing}")

    results = [
        compare_artifact(
            name=name,
            entry=exports[name],
            raw_dir=args.raw_dir,
            optimized_dir=args.optimized_dir,
            manifest=manifest,
            seed=args.seed + index,
            atol=args.atol,
            rtol=args.rtol,
        )
        for index, name in enumerate(artifacts)
    ]
    report = {
        "schema_version": 1,
        "raw_dir": args.raw_dir.as_posix(),
        "optimized_dir": args.optimized_dir.as_posix(),
        "manifest": manifest_path(args).as_posix(),
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "passed": all(result["passed"] for result in results),
        "artifacts": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
