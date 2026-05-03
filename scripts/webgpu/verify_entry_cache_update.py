from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


MANIFEST_NAME = "breakout_onnx_manifest.json"
FULL_CACHE_ARTIFACT = "breakout_dynamics_sample_append_context_slide_full_cache_b1_t1_s4"
ENTRY_ARTIFACT = "breakout_dynamics_sample_append_context_slide_entry_b1_t1_s4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the small entry-cache step artifact plus the browser-side "
            "slide/rebase cache update is numerically equivalent to the full-cache "
            "steady-state artifact."
        )
    )
    parser.add_argument("--dir", type=Path, default=Path("webgpu_app/assets"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("webgpu_app/bench/results/entry_cache_update_accuracy.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def artifact_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        entry["name"]: entry
        for entry in manifest.get("exports", [])
        if isinstance(entry, dict) and entry.get("name")
    }


def make_float(shape: list[int], rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=0.5, size=tuple(shape)).astype(np.float32)


def make_int(
    name: str, shape: list[int], manifest: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    dynamics = manifest.get("dynamics", {})
    num_actions = int(dynamics.get("num_actions", 4))
    if name == "actions":
        return rng.integers(0, num_actions, size=tuple(shape), dtype=np.int32)
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


def adapt_feeds_for_spec(
    feeds: dict[str, np.ndarray],
    target_inputs: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    adapted: dict[str, np.ndarray] = {}
    for name, value in feeds.items():
        target = target_inputs.get(name)
        if target is None:
            continue
        shape = tuple(int(dim) for dim in target.get("shape", []))
        if value.shape == shape:
            adapted[name] = value
            continue
        if (
            value.ndim == 6
            and len(shape) == 6
            and value.shape[:3] == shape[:3]
            and value.shape[3] == shape[4]
            and value.shape[4] == shape[3]
            and value.shape[5] == shape[5]
        ):
            adapted[name] = np.transpose(value, (0, 1, 2, 4, 3, 5)).copy()
            continue
        raise ValueError(
            f"Cannot adapt input {name}: source shape {value.shape}, target shape {shape}."
        )
    return adapted


def run_ort(path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, feeds)
    return dict(zip(output_names, outputs))


def slide_rebase_key_cache(
    k_cache: np.ndarray,
    k_entry: np.ndarray,
    *,
    base: float,
    head_dim: int,
) -> np.ndarray:
    kept = k_cache[:, :, :, 1:, :, :]
    half = head_dim // 2
    theta = 1.0 / (base ** (np.arange(half, dtype=np.float32) / half))
    cos = np.cos(theta).astype(k_cache.dtype).reshape(1, 1, 1, 1, 1, half)
    sin = np.sin(theta).astype(k_cache.dtype).reshape(1, 1, 1, 1, 1, half)
    left = kept[..., :half]
    right = kept[..., half:]
    rebased = np.concatenate([left * cos + right * sin, right * cos - left * sin], axis=-1)
    return np.concatenate([rebased, k_entry], axis=3)


def slide_value_cache(v_cache: np.ndarray, v_entry: np.ndarray) -> np.ndarray:
    return np.concatenate([v_cache[:, :, :, 1:, :, :], v_entry], axis=3)


def compare_arrays(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    expected_f = expected.astype(np.float32)
    actual_f = actual.astype(np.float32)
    diff = np.abs(expected_f - actual_f)
    rel = diff / np.maximum(np.abs(expected_f), np.asarray(1e-12, dtype=np.float32))
    return {
        "shape": list(expected.shape),
        "expected_dtype": str(expected.dtype),
        "actual_dtype": str(actual.dtype),
        "max_abs_error": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_error": float(np.mean(diff)) if diff.size else 0.0,
        "max_rel_error": float(np.max(rel)) if rel.size else 0.0,
        "mean_rel_error": float(np.mean(rel)) if rel.size else 0.0,
        "passed": bool(np.allclose(expected, actual, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest if args.manifest is not None else args.dir / MANIFEST_NAME
    manifest = load_json(manifest_path)
    exports = artifact_map(manifest)
    full_spec = exports[FULL_CACHE_ARTIFACT]
    entry_spec = exports[ENTRY_ARTIFACT]

    rng = np.random.default_rng(args.seed)
    feeds = {
        input_name: make_feed(input_name, input_spec, manifest, rng)
        for input_name, input_spec in full_spec.get("inputs", {}).items()
    }
    full_outputs = run_ort(args.dir / full_spec["path"], feeds)
    entry_outputs = run_ort(
        args.dir / entry_spec["path"],
        adapt_feeds_for_spec(feeds, entry_spec.get("inputs", {})),
    )
    entry_pred_output = "final_z" if entry_spec.get("final_z_aliases_pred_z") else "pred_z"

    dynamics = manifest.get("dynamics", {})
    base = float(dynamics.get("rope_base", dynamics.get("base", 10000.0)))
    head_dim = int(dynamics.get("head_dim", feeds["k_cache"].shape[-1]))
    reconstructed_k = slide_rebase_key_cache(
        feeds["k_cache"],
        entry_outputs["candidate_k_entry"],
        base=base,
        head_dim=head_dim,
    )
    reconstructed_v = slide_value_cache(feeds["v_cache"], entry_outputs["candidate_v_entry"])

    comparisons = {
        "final_z": compare_arrays(
            full_outputs["final_z"],
            entry_outputs["final_z"],
            atol=args.atol,
            rtol=args.rtol,
        ),
        "pred_z": compare_arrays(
            full_outputs["pred_z"],
            entry_outputs[entry_pred_output],
            atol=args.atol,
            rtol=args.rtol,
        ),
        "candidate_k_cache_from_entry": compare_arrays(
            full_outputs["candidate_k_cache"],
            reconstructed_k,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "candidate_v_cache_from_entry": compare_arrays(
            full_outputs["candidate_v_cache"],
            reconstructed_v,
            atol=args.atol,
            rtol=args.rtol,
        ),
    }
    report = {
        "schema_version": 1,
        "manifest": manifest_path.as_posix(),
        "dir": args.dir.as_posix(),
        "seed": args.seed,
        "full_cache_artifact": FULL_CACHE_ARTIFACT,
        "entry_artifact": ENTRY_ARTIFACT,
        "rope_base": base,
        "head_dim": head_dim,
        "passed": all(result["passed"] for result in comparisons.values()),
        "outputs": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
