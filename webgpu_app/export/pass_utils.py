from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


RewriteFn = Callable[..., dict[str, Any]]


def disabled(exported_paths: dict[str, Path], reason: str) -> dict[str, dict[str, Any]]:
    return {name: {"enabled": False, "reason": reason} for name in exported_paths}


def run_all(exported_paths: dict[str, Path], rewrite: RewriteFn) -> dict[str, dict[str, Any]]:
    return {name: rewrite(path) for name, path in exported_paths.items()}
