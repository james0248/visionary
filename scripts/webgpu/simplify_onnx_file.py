import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


def external_data_path(path: Path) -> Path:
    return path.with_name(path.name + ".data")


def op_counts(path: Path) -> Counter[str]:
    model = onnx.load(path.as_posix(), load_external_data=False)
    return Counter(node.op_type for node in model.graph.node)


def simplify_file(path: Path) -> dict[str, Any]:
    try:
        from onnxsim import simplify
    except ImportError as exc:
        raise ImportError(
            "onnxsim is required for --simplify_onnx. Install it with "
            "`uv add --group onnx 'onnx-simplifier==0.4.36'`."
        ) from exc

    before = op_counts(path)
    model = onnx.load(path.as_posix(), load_external_data=True)
    simplified_model, check_ok = simplify(model)
    if not check_ok:
        raise RuntimeError(f"onnxsim validation failed for {path}.")
    onnx.checker.check_model(simplified_model)
    onnx.save_model(
        simplified_model,
        path.as_posix(),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path(path).name,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.load(path.as_posix(), load_external_data=True)

    after = op_counts(path)
    tracked_ops = ("Reshape", "Concat", "Expand", "Cast", "Transpose", "Einsum", "Gemm")
    return {
        "enabled": True,
        "tool": "onnxsim",
        "check_ok": bool(check_ok),
        "node_count_before": int(sum(before.values())),
        "node_count_after": int(sum(after.values())),
        "tracked_ops_before": {op: int(before.get(op, 0)) for op in tracked_ops},
        "tracked_ops_after": {op: int(after.get(op, 0)) for op in tracked_ops},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplify one ONNX file with onnxsim.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(simplify_file(args.path), sort_keys=True))


if __name__ == "__main__":
    main()
