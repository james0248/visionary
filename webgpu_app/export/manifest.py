from __future__ import annotations

from typing import Any


def version_or_unknown(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def tensor_spec(dtype: str, shape: tuple[int, ...]) -> dict[str, Any]:
    return {"dtype": dtype, "shape": list(shape)}


def cache_contract(dyn_shapes, available: bool, dtype: str = "float32") -> dict[str, Any]:
    cache_shape = list(dyn_shapes.cache)
    layer_cache_shape = list(dyn_shapes.layer_cache)
    return {
        "status": "available" if available else "contract_only",
        "reason": None
        if available
        else (
            "Cached ONNX graphs require inference-only temporal attention kernels. "
            "Do not treat the uncached dynamics graph as production browser success."
        ),
        "static_axes": True,
        "context_length": dyn_shapes.context_length,
        "temporal_blocks": dyn_shapes.temporal_blocks,
        "total_tokens": dyn_shapes.total_tokens,
        "num_kv_heads": dyn_shapes.num_kv_heads,
        "head_dim": dyn_shapes.head_dim,
        "tensors": {
            "k_cache": tensor_spec(dtype, tuple(cache_shape)),
            "v_cache": tensor_spec(dtype, tuple(cache_shape)),
            "layer_cache": {
                **tensor_spec(dtype, tuple(layer_cache_shape)),
                "layers": dyn_shapes.temporal_blocks,
            },
            "cache_length": tensor_spec("int32", (1,)),
        },
        "ownership": "browser_runtime",
        "invalidation": [
            "new_episode",
            "checkpoint_or_config_change",
            "context_latent_change",
            "action_history_change",
            "latent_history_change",
            "context_tau_change",
            "batch_order_change",
        ],
        "target_frame_policy": (
            "Use committed cache for attention on all sample iterations. Discard candidate "
            "cache for sample steps 0-2. Commit candidate cache only on sample step 3."
        ),
    }
