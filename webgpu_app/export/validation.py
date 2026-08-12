from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
import onnxruntime as ort


def run_ort(path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    typed_feeds = dict(feeds)
    for input_spec in session.get_inputs():
        if input_spec.name not in typed_feeds:
            continue
        if input_spec.type == "tensor(float16)":
            typed_feeds[input_spec.name] = typed_feeds[input_spec.name].astype(np.float16)
        elif input_spec.type == "tensor(float)":
            typed_feeds[input_spec.name] = typed_feeds[input_spec.name].astype(np.float32)
    outputs = session.run(None, typed_feeds)
    return {output.name: value for output, value in zip(session.get_outputs(), outputs)}


def compare_arrays(expected: np.ndarray, actual: np.ndarray, atol: float, rtol: float) -> dict[str, Any]:
    expected = expected.astype(np.float32) if expected.dtype == np.float16 else expected
    actual = actual.astype(np.float32) if actual.dtype == np.float16 else actual
    diff = np.abs(expected - actual)
    denom = np.maximum(np.abs(expected), np.asarray(1e-8, dtype=expected.dtype))
    rel = diff / denom
    passed = bool(np.allclose(expected, actual, atol=atol, rtol=rtol))
    return {
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": float(np.max(diff)),
        "mean_abs_error": float(np.mean(diff)),
        "max_rel_error": float(np.max(rel)),
        "mean_rel_error": float(np.mean(rel)),
        "passed": passed,
    }


def validate_single_output(
    path: Path,
    feeds: dict[str, jax.Array],
    output_name: str,
    expected: jax.Array,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    ort_feeds = {name: np.asarray(jax.device_get(value)) for name, value in feeds.items()}
    actual = run_ort(path, ort_feeds)[output_name]
    return compare_arrays(
        np.asarray(jax.device_get(expected)),
        actual,
        atol=atol,
        rtol=rtol,
    )


def validate_outputs(
    path: Path,
    feeds: dict[str, jax.Array],
    expected: dict[str, jax.Array],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    ort_feeds = {name: np.asarray(jax.device_get(value)) for name, value in feeds.items()}
    actual = run_ort(path, ort_feeds)
    results = {
        name: compare_arrays(
            np.asarray(jax.device_get(expected_value)),
            actual[name],
            atol=atol,
            rtol=rtol,
        )
        for name, expected_value in expected.items()
    }
    return {
        "atol": atol,
        "rtol": rtol,
        "passed": all(result["passed"] for result in results.values()),
        "outputs": results,
    }
