import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create browser demo initial K/V cache artifacts from the exported prefix-step ONNX model."
    )
    parser.add_argument("--asset_dir", type=Path, default=Path("webgpu_app/assets"))
    parser.add_argument("--onnx_manifest", default="breakout_onnx_manifest.json")
    parser.add_argument("--context_manifest", default="breakout_demo_context.json")
    parser.add_argument("--mode", choices=("step", "prefill"), default="step")
    parser.add_argument("--prefix_step_export", default="breakout_dynamics_step_cached_b1_t1")
    parser.add_argument("--prefill_export", default="breakout_dynamics_prefill_cached_b1_t64")
    parser.add_argument("--name", default="breakout_demo_initial_cache")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def find_export(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in manifest.get("exports", []):
        if entry.get("name") == name:
            return entry
    raise KeyError(f"Missing export {name!r} in manifest.")


def dtype_array(dtype: str) -> type[np.ndarray]:
    if dtype == "float32":
        return np.float32
    if dtype == "int32":
        return np.int32
    raise ValueError(f"Unsupported artifact dtype {dtype!r}.")


def load_array(asset_dir: Path, spec: dict[str, Any]) -> np.ndarray:
    path = asset_dir / spec["path"]
    array = np.fromfile(path, dtype=dtype_array(spec["dtype"]))
    return array.reshape(tuple(spec["shape"]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_array(path: Path, array: np.ndarray, overwrite: bool) -> dict[str, Any]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    array = np.ascontiguousarray(array)
    path.write_bytes(array.tobytes())
    return {
        "path": path.name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes": int(array.nbytes),
        "sha256": sha256_file(path),
    }


def output_name(session: ort.InferenceSession, candidates: tuple[str, ...]) -> str:
    outputs = [output.name for output in session.get_outputs()]
    for candidate in candidates:
        if candidate in outputs:
            return candidate
    for output in outputs:
        if any(output.endswith(candidate) for candidate in candidates):
            return output
    raise KeyError(f"Could not find any output matching {candidates}; available={outputs}")


def scalar_tensor(value: int) -> np.ndarray:
    return np.asarray([value], dtype=np.int32)


def frame_tensor(tensor: np.ndarray, frame_index: int) -> np.ndarray:
    return tensor[:, frame_index : frame_index + 1]


def cache_from_step_replay(
    asset_dir: Path,
    manifest: dict[str, Any],
    context_manifest: dict[str, Any],
    context_z: np.ndarray,
    context_actions: np.ndarray,
    step_levels: np.ndarray,
    signal_levels: np.ndarray,
    export_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    prefix_step = find_export(manifest, export_name)
    model_path = asset_dir / prefix_step["path"]
    cache_shape = tuple(prefix_step["inputs"]["k_cache"]["shape"])
    cache = {
        "k": np.zeros(cache_shape, dtype=np.float32),
        "v": np.zeros(cache_shape, dtype=np.float32),
        "length": scalar_tensor(0),
    }
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    k_name = output_name(session, ("candidate_k_cache", "k_cache"))
    v_name = output_name(session, ("candidate_v_cache", "v_cache"))
    length_name = output_name(session, ("candidate_cache_length", "cache_length"))

    prefix_frames = int(context_manifest.get("prefix_frames", 0))
    prefix_slot_start = int(context_manifest.get("prefix_slot_start", 0))
    for offset in range(prefix_frames):
        frame_index = prefix_slot_start + offset
        feeds = {
            "z": frame_tensor(context_z, frame_index),
            "actions": frame_tensor(context_actions, frame_index),
            "step_levels": frame_tensor(step_levels, frame_index),
            "signal_levels": frame_tensor(signal_levels, frame_index),
            "position_index": cache["length"],
            "k_cache": cache["k"],
            "v_cache": cache["v"],
            "cache_length": cache["length"],
        }
        outputs = session.run(None, feeds)
        by_name = dict(zip([output.name for output in session.get_outputs()], outputs, strict=True))
        cache = {
            "k": np.asarray(by_name[k_name], dtype=np.float32),
            "v": np.asarray(by_name[v_name], dtype=np.float32),
            "length": np.asarray(by_name[length_name], dtype=np.int32).reshape((1,)),
        }

    return cache, {
        "source_cache_export": export_name,
        "source_cache_model": prefix_step["path"],
        "source_cache_model_sha256": sha256_file(model_path),
    }


def cache_from_prefill(
    asset_dir: Path,
    manifest: dict[str, Any],
    context_z: np.ndarray,
    context_actions: np.ndarray,
    step_levels: np.ndarray,
    signal_levels: np.ndarray,
    export_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    prefill = find_export(manifest, export_name)
    model_path = asset_dir / prefill["path"]
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    k_name = output_name(session, ("k_cache", "candidate_k_cache"))
    v_name = output_name(session, ("v_cache", "candidate_v_cache"))
    length_name = output_name(session, ("cache_length", "candidate_cache_length"))
    feeds = {
        "z": context_z,
        "actions": context_actions,
        "step_levels": step_levels,
        "signal_levels": signal_levels,
    }
    outputs = session.run(None, feeds)
    by_name = dict(zip([output.name for output in session.get_outputs()], outputs, strict=True))
    return {
        "k": np.asarray(by_name[k_name], dtype=np.float32),
        "v": np.asarray(by_name[v_name], dtype=np.float32),
        "length": np.asarray(by_name[length_name], dtype=np.int32).reshape((1,)),
    }, {
        "source_cache_export": export_name,
        "source_cache_model": prefill["path"],
        "source_cache_model_sha256": sha256_file(model_path),
    }


def main() -> None:
    args = parse_args()
    asset_dir = args.asset_dir
    manifest_path = asset_dir / args.onnx_manifest
    context_manifest_path = asset_dir / args.context_manifest
    manifest = load_json(manifest_path)
    context_manifest = load_json(context_manifest_path)

    context_arrays = context_manifest["arrays"]
    context_z = load_array(asset_dir, context_arrays["z"])
    context_actions = load_array(asset_dir, context_arrays["actions"])
    step_levels = load_array(asset_dir, context_arrays["step_levels"])
    signal_levels = load_array(asset_dir, context_arrays["signal_levels"])

    if args.mode == "prefill":
        cache, source_info = cache_from_prefill(
            asset_dir,
            manifest,
            context_z,
            context_actions,
            step_levels,
            signal_levels,
            args.prefill_export,
        )
    else:
        cache, source_info = cache_from_step_replay(
            asset_dir,
            manifest,
            context_manifest,
            context_z,
            context_actions,
            step_levels,
            signal_levels,
            args.prefix_step_export,
        )

    prefix = asset_dir / args.name
    manifest_out = prefix.with_suffix(".json")
    if manifest_out.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_out} already exists. Pass --overwrite to replace it.")

    cache_length = cache["length"]

    cache_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_onnx_manifest": args.onnx_manifest,
        "source_context_manifest": args.context_manifest,
        "cache_creation_mode": args.mode,
        **source_info,
        "cache_length": int(cache_length[0]),
        "context_length": int(context_manifest["context_length"]),
        "prefix_frames": int(context_manifest.get("prefix_frames", 0)),
        "prefix_slot_start": int(context_manifest.get("prefix_slot_start", 0)),
        "arrays": {
            "k_cache": write_array(
                prefix.with_suffix(".k_cache.f32.bin"),
                cache["k"],
                args.overwrite,
            ),
            "v_cache": write_array(
                prefix.with_suffix(".v_cache.f32.bin"),
                cache["v"],
                args.overwrite,
            ),
            "cache_length": write_array(
                prefix.with_suffix(".cache_length.i32.bin"),
                cache_length,
                args.overwrite,
            ),
        },
        "notes": [
            "This cache is generated offline from the stored browser demo context latents.",
            "The browser continues rollout with the cache-length entry graph and updates the fixed-size cache from per-frame K/V entries.",
        ],
    }
    manifest_out.write_text(json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest_out}")


if __name__ == "__main__":
    main()
