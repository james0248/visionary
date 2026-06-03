from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")


def external_data_path(path: Path) -> Path:
    return path.with_name(path.name + ".data")


def remove_existing_export(path: Path, overwrite: bool) -> None:
    ensure_output(path, overwrite=overwrite)
    sidecar = external_data_path(path)
    ensure_output(sidecar, overwrite=overwrite)
    if overwrite:
        path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_onnx_artifact(src: Path, dst: Path, overwrite: bool) -> dict[str, Any]:
    ensure_output(dst, overwrite=overwrite)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        dst.unlink(missing_ok=True)
        external_data_path(dst).unlink(missing_ok=True)
    shutil.copy2(src, dst)

    copied = [{"path": dst.name, "sha256": sha256_file(dst), "size_bytes": dst.stat().st_size}]
    src_sidecar = external_data_path(src)
    if src_sidecar.exists():
        dst_sidecar = external_data_path(dst)
        ensure_output(dst_sidecar, overwrite=overwrite)
        if overwrite:
            dst_sidecar.unlink(missing_ok=True)
        shutil.copy2(src_sidecar, dst_sidecar)
        copied.append(
            {
                "path": dst_sidecar.name,
                "sha256": sha256_file(dst_sidecar),
                "size_bytes": dst_sidecar.stat().st_size,
            }
        )
    return {"path": dst.name, "files": copied}


def snapshot_raw_artifacts(
    exported_paths: dict[str, Path],
    raw_out_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    copied = {
        name: copy_onnx_artifact(path, raw_out_dir / path.name, overwrite=overwrite)
        for name, path in exported_paths.items()
    }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Raw jax2onnx exports captured before ONNX simplification, ORT optimization, "
            "or WebGPU-specific graph rewrites."
        ),
        "artifacts": copied,
    }
    raw_manifest_path = raw_out_dir / "raw_onnx_artifacts_manifest.json"
    ensure_output(raw_manifest_path, overwrite=overwrite)
    raw_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "enabled": True,
        "path": raw_out_dir.as_posix(),
        "manifest": raw_manifest_path.name,
        "artifacts": copied,
    }


def export_file_metadata(path: Path) -> dict[str, Any]:
    metadata = {
        "path": path.name,
        "sha256": sha256_file(path),
    }
    sidecar = external_data_path(path)
    if sidecar.exists():
        metadata["external_data"] = [
            {
                "path": sidecar.name,
                "sha256": sha256_file(sidecar),
            }
        ]
    else:
        metadata["external_data"] = []
    return metadata
