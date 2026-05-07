import itertools
import json
import logging
import os
import re
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import fsspec
import grain.python as grain
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra.utils import instantiate
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from omegaconf import DictConfig, OmegaConf

from visionary.common.checkpoint import (
    CheckpointManager,
    save_model_export,
    save_preprocessor_export,
)
from visionary.common.jax import fold_in_many, maybe_initialize_distributed
from visionary.common.train_state import TokenizerTrainState
from visionary.common.wandb import WandbLogger
from visionary.dataset import RandomVideoCrop, VideoDataset, VideoDataSource
from visionary.lpips import LPIPS
from visionary.tokenizer import Tokenizer
from visionary.tokenizer_preprocessor import TokenizerPreprocessor

logger = logging.getLogger(__name__)


LPIPS_PRETRAINED_NETWORK = "alexnet"
LOSS_RMS_DECAY = 0.99
LOSS_RMS_EPS = 1e-8
DATA_AXIS = "data"
FSDP_AXIS = "fsdp"


def cfg_select(cfg: DictConfig, path: str, default: Any) -> Any:
    value = OmegaConf.select(cfg, path, default=default)
    return default if value is None else value


def uri_join(base: str, *parts: str) -> str:
    base = str(base).rstrip("/")
    suffix = "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))
    if not suffix:
        return base
    return f"{base}/{suffix}"


def sanitize_path_component(value: Any, default: str = "run") -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip("-._")
    return sanitized or default


def ensure_artifact_parent(uri: str) -> None:
    if "://" not in uri:
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        return

    fs, _, paths = fsspec.get_fs_token_paths(uri)
    if not paths:
        return
    parent = os.path.dirname(paths[0])
    if parent:
        fs.makedirs(parent, exist_ok=True)


def write_text_artifact(uri: str, payload: str) -> None:
    ensure_artifact_parent(uri)
    with fsspec.open(uri, "w") as handle:
        handle.write(payload)


def write_bytes_artifact(uri: str, payload: bytes) -> None:
    ensure_artifact_parent(uri)
    with fsspec.open(uri, "wb") as handle:
        handle.write(payload)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(jax.device_get(value))
        return array.item() if array.shape == () else array.tolist()
    if isinstance(value, (Path, P)):
        return str(value)
    return value


def parse_auto_bool(value: Any, *, auto_value: bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value == "auto":
        return auto_value
    if value in {"1", "true", "yes", "on", "enabled"}:
        return True
    if value in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(f"Expected a boolean or 'auto', got {value!r}")


def build_fsdp_mesh(cfg: DictConfig) -> tuple[Mesh, bool, int]:
    device_count = jax.device_count()
    auto_enabled = device_count > 1 or jax.process_count() > 1
    fsdp_enabled = parse_auto_bool(
        cfg_select(cfg, "fsdp.enabled", "auto"),
        auto_value=auto_enabled,
    )
    data_axis_size = int(cfg_select(cfg, "fsdp.data_axis_size", 1))
    if data_axis_size < 1:
        raise ValueError(f"fsdp.data_axis_size must be >= 1, got {data_axis_size}")
    if device_count % data_axis_size != 0:
        raise ValueError(
            "fsdp.data_axis_size must divide jax.device_count(); got "
            f"{data_axis_size=} and device_count={device_count}"
        )

    fsdp_axis_size = device_count // data_axis_size
    devices = mesh_utils.create_device_mesh((data_axis_size, fsdp_axis_size))
    mesh = Mesh(devices, (DATA_AXIS, FSDP_AXIS))
    if fsdp_enabled and fsdp_axis_size < 1:
        raise ValueError("FSDP requested but no devices are available.")
    return mesh, fsdp_enabled, fsdp_axis_size


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


def batch_partition_spec() -> P:
    return P((DATA_AXIS, FSDP_AXIS))


def choose_fsdp_partition_spec(
    value: Any,
    *,
    enabled: bool,
    fsdp_axis_size: int,
    min_shard_size: int,
) -> P:
    if not hasattr(value, "shape"):
        if not np.isscalar(value):
            return P()
        dtype = np.int32 if isinstance(value, (int, np.integer)) else None
        value = np.asarray(value, dtype=dtype)
    shape = tuple(int(dim) for dim in value.shape)
    if (
        not enabled
        or fsdp_axis_size <= 1
        or not shape
        or int(np.prod(shape)) < min_shard_size
    ):
        return P()

    ranked_dims = sorted(range(len(shape)), key=lambda axis: shape[axis], reverse=True)
    for axis in ranked_dims:
        if shape[axis] >= fsdp_axis_size and shape[axis] % fsdp_axis_size == 0:
            spec = [None] * len(shape)
            spec[axis] = FSDP_AXIS
            return P(*spec)
    return P()


def make_array_shardings(
    tree,
    *,
    mesh: Mesh,
    fsdp_enabled: bool,
    fsdp_axis_size: int,
    min_shard_size: int,
):
    return jax.tree_util.tree_map(
        lambda value: NamedSharding(
            mesh,
            choose_fsdp_partition_spec(
                value,
                enabled=fsdp_enabled,
                fsdp_axis_size=fsdp_axis_size,
                min_shard_size=min_shard_size,
            ),
        ),
        tree,
    )


def make_global_array_from_host(value, sharding: NamedSharding):
    if not hasattr(value, "shape") and not np.isscalar(value):
        return value
    dtype = np.int32 if isinstance(value, (int, np.integer)) else None
    value = np.asarray(jax.device_get(value), dtype=dtype)
    return jax.make_array_from_callback(
        value.shape,
        sharding,
        lambda index: value[index],
        dtype=value.dtype,
    )


def put_replicated(value, mesh: Mesh):
    return jax.device_put(value, replicated_sharding(mesh))


def put_global_batch(batch: VideoDataset, batch_sharding: NamedSharding) -> VideoDataset:
    return jax.tree_util.tree_map(
        lambda value: jax.make_array_from_process_local_data(batch_sharding, value),
        batch,
    )


def host_local_batch(batch, mesh: Mesh, pspec: P):
    return multihost_utils.global_array_to_host_local_array(batch, mesh, pspec)


def log_sharding_summary(tree, shardings, *, prefix: str, max_entries: int = 12) -> None:
    leaves = []

    def visit(path, value, sharding):
        if not hasattr(value, "shape"):
            return
        spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
        leaves.append(
            (
                spec != P(),
                int(np.prod(value.shape)) if value.shape else 1,
                jax.tree_util.keystr(path, simple=True, separator="/"),
                tuple(int(dim) for dim in value.shape),
                spec,
            )
        )

    jax.tree_util.tree_map_with_path(visit, tree, shardings)
    num_leaves = len(leaves)
    num_sharded = sum(1 for sharded, *_ in leaves if sharded)
    total_values = sum(size for _, size, *_ in leaves)
    sharded_values = sum(size for sharded, size, *_ in leaves if sharded)
    logger.info(
        "%s sharding: %d/%d leaves sharded, %.1f%% of values in sharded leaves",
        prefix,
        num_sharded,
        num_leaves,
        100.0 * sharded_values / max(total_values, 1),
    )
    for sharded, size, path, shape, spec in sorted(leaves, reverse=True)[:max_entries]:
        logger.info(
            "%s sharding leaf: sharded=%s values=%d path=%s shape=%s spec=%s",
            prefix,
            sharded,
            size,
            path,
            shape,
            spec,
        )


def collect_local_memory_stats() -> list[dict[str, Any]]:
    stats = []
    for device in jax.local_devices():
        memory_stats = None
        if hasattr(device, "memory_stats"):
            try:
                memory_stats = device.memory_stats()
            except Exception as exc:  # pragma: no cover - backend-specific diagnostic path.
                memory_stats = {"error": f"{type(exc).__name__}: {exc}"}
        stats.append(
            {
                "id": str(device),
                "platform": getattr(device, "platform", ""),
                "memory_stats": jsonable(memory_stats),
            }
        )
    return stats


def aggregate_float(records: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = [float(record[key]) for record in records if key in record and record[key] is not None]
    if not values:
        return None
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "last": values[-1],
    }


class TrainingProfiler:
    def __init__(
        self,
        cfg: DictConfig,
        *,
        metadata: dict[str, Any],
        process_index: int,
    ) -> None:
        self.enabled = bool(cfg_select(cfg, "profile.enabled", False))
        self.process_index = process_index
        self.records: list[dict[str, Any]] = []
        self.metadata = jsonable(metadata)
        self.flush_every_records = max(int(cfg_select(cfg, "profile.flush_every_records", 1)), 1)
        self.collect_memory_stats = bool(cfg_select(cfg, "profile.collect_memory_stats", False))
        self.blocking_timing = bool(cfg_select(cfg, "profile.blocking_timing", False))
        self.trace_enabled = self.enabled and bool(cfg_select(cfg, "profile.trace_enabled", False))
        self.trace_start_step = max(int(cfg_select(cfg, "profile.trace_start_step", 20)), 1)
        self.trace_num_steps = max(int(cfg_select(cfg, "profile.trace_num_steps", 5)), 0)
        self.trace_end_step = self.trace_start_step + self.trace_num_steps - 1
        self.sync_trace_steps = bool(cfg_select(cfg, "profile.sync_trace_steps", True))
        self.create_perfetto_trace = bool(cfg_select(cfg, "profile.create_perfetto_trace", True))
        self.save_device_memory_profile = bool(
            cfg_select(cfg, "profile.save_device_memory_profile", False)
        )
        self._trace_started = False
        self._trace_stopped = False
        self._flush_warning_logged = False

        exp_name = sanitize_path_component(cfg_select(cfg, "exp_name", "tokenizer"))
        local_profile_root = f"/tmp/visionary-tokenizer-profiles/{exp_name}"
        default_output_dir = uri_join(local_profile_root, "json")
        self.output_dir = str(cfg_select(cfg, "profile.output_dir", default_output_dir))
        self.events_path = uri_join(
            self.output_dir,
            f"tokenizer_profile_process_{process_index}.jsonl",
        )
        self.summary_path = uri_join(
            self.output_dir,
            f"tokenizer_profile_process_{process_index}.json",
        )
        self.memory_profile_path = uri_join(
            self.output_dir,
            f"tokenizer_device_memory_process_{process_index}.prof",
        )

        trace_dir = cfg_select(cfg, "profile.trace_dir", None)
        if trace_dir:
            self.trace_dir = uri_join(str(trace_dir), f"process_{process_index}")
        else:
            self.trace_dir = uri_join(local_profile_root, "traces", f"process_{process_index}")

    def log_configuration(self) -> None:
        if not self.enabled:
            logger.info("Tokenizer profiler disabled.")
            return
        logger.info(
            "Tokenizer profiler enabled: events=%s summary=%s trace_enabled=%s "
            "trace_steps=%s..%s trace_dir=%s blocking_timing=%s",
            self.events_path,
            self.summary_path,
            self.trace_enabled,
            self.trace_start_step,
            self.trace_end_step if self.trace_num_steps else "disabled",
            self.trace_dir,
            self.blocking_timing,
        )

    def should_trace_step(self, step: int) -> bool:
        return (
            self.trace_enabled
            and self.trace_num_steps > 0
            and self.trace_start_step <= step <= self.trace_end_step
        )

    def should_sync_step(self, step: int) -> bool:
        return self.blocking_timing or (self.sync_trace_steps and self.should_trace_step(step))

    def maybe_start_trace(self, step: int) -> None:
        if not self.should_trace_step(step) or self._trace_started:
            return
        Path(self.trace_dir).mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(
            self.trace_dir,
            create_perfetto_trace=self.create_perfetto_trace,
        )
        self._trace_started = True
        self.record(
            {
                "type": "trace_start",
                "step": step,
                "trace_dir": self.trace_dir,
                "create_perfetto_trace": self.create_perfetto_trace,
            },
            flush=True,
        )
        logger.info("Started JAX profiler trace at step %d: %s", step, self.trace_dir)

    def maybe_stop_trace(self, step: int) -> None:
        if not self._trace_started or self._trace_stopped or step < self.trace_end_step:
            return
        jax.profiler.stop_trace()
        self._trace_stopped = True
        self.record(
            {
                "type": "trace_stop",
                "step": step,
                "trace_dir": self.trace_dir,
            },
            flush=True,
        )
        logger.info("Stopped JAX profiler trace at step %d: %s", step, self.trace_dir)

    def step_context(self, step: int):
        if self.should_trace_step(step):
            return jax.profiler.StepTraceAnnotation("tokenizer_train", step_num=step)
        return nullcontext()

    def annotation(self, name: str, step: int):
        if self.should_trace_step(step):
            return jax.profiler.TraceAnnotation(name)
        return nullcontext()

    def record(self, record: dict[str, Any], *, flush: bool = False) -> None:
        if not self.enabled:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "process_index": self.process_index,
            **jsonable(record),
        }
        self.records.append(payload)
        if flush or len(self.records) % self.flush_every_records == 0:
            self.flush()

    def build_summary(self) -> dict[str, Any]:
        train_windows = [record for record in self.records if record.get("type") == "train_window"]
        step_records = [record for record in self.records if record.get("type") == "train_step"]
        event_counts: dict[str, int] = {}
        for record in self.records:
            event_type = str(record.get("type", "unknown"))
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        return {
            "metadata": self.metadata,
            "paths": {
                "events": self.events_path,
                "summary": self.summary_path,
                "trace_dir": self.trace_dir if self.trace_enabled else None,
                "device_memory_profile": self.memory_profile_path
                if self.save_device_memory_profile
                else None,
            },
            "event_counts": event_counts,
            "latest_train_window": train_windows[-1] if train_windows else None,
            "latest_train_step": step_records[-1] if step_records else None,
            "aggregates": {
                "train_windows": {
                    key: aggregate_float(train_windows, key)
                    for key in (
                        "steps_per_second",
                        "examples_per_second",
                        "data_seconds_per_step",
                        "transfer_seconds_per_step",
                        "dispatch_seconds_per_step",
                        "sync_seconds_per_step",
                        "compute_seconds_per_step",
                        "wall_seconds_per_step",
                    )
                },
                "train_steps": {
                    key: aggregate_float(step_records, key)
                    for key in (
                        "total_seconds",
                        "data_seconds",
                        "transfer_seconds",
                        "dispatch_seconds",
                        "sync_seconds",
                    )
                },
            },
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def flush(self) -> None:
        if not self.enabled:
            return
        try:
            write_text_artifact(
                self.events_path,
                "".join(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    for record in self.records
                ),
            )
            write_text_artifact(
                self.summary_path,
                json.dumps(self.build_summary(), indent=2, sort_keys=True) + "\n",
            )
        except Exception as exc:  # pragma: no cover - depends on cloud/local filesystems.
            if not self._flush_warning_logged:
                logger.warning(
                    "Failed to flush tokenizer profile artifacts to %s: %s",
                    self.output_dir,
                    exc,
                )
                self._flush_warning_logged = True

    def close(self, *, final_step: int) -> None:
        if not self.enabled:
            return
        if self._trace_started and not self._trace_stopped:
            jax.profiler.stop_trace()
            self._trace_stopped = True
            self.record(
                {
                    "type": "trace_stop",
                    "step": final_step,
                    "trace_dir": self.trace_dir,
                    "reason": "training_end",
                },
            )
        if self.save_device_memory_profile:
            try:
                write_bytes_artifact(
                    self.memory_profile_path,
                    jax.profiler.device_memory_profile(),
                )
            except Exception as exc:  # pragma: no cover - backend-specific diagnostic path.
                self.record(
                    {
                        "type": "device_memory_profile_error",
                        "step": final_step,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        self.record({"type": "profile_close", "step": final_step}, flush=True)


def make_host_seed(*values: int) -> int:
    seed_sequence = np.random.SeedSequence([int(value) for value in values])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def log_train_timing(
    wb: WandbLogger,
    step: int,
    start_step: int,
    metrics,
    data_time: float,
    transfer_time: float,
    dispatch_time: float,
    sync_time: float,
    global_batch_size: int,
) -> tuple[dict[str, float], dict[str, float], float]:
    window_steps = step - start_step
    avg_steps = max(window_steps, 1)
    sync_start = time.monotonic()
    train_metrics = jax.device_get(metrics)
    sync_time += time.monotonic() - sync_start
    wall_time = data_time + transfer_time + dispatch_time + sync_time

    stats = {
        "sps": window_steps / max(wall_time, 1e-8),
        "examples_per_second": window_steps * global_batch_size / max(wall_time, 1e-8),
        "data_time": data_time / avg_steps,
        "transfer_time": transfer_time / avg_steps,
        "compute_time": (dispatch_time + sync_time) / avg_steps,
        "dispatch_time": dispatch_time / avg_steps,
        "sync_time": sync_time / avg_steps,
        "wall_time": wall_time / avg_steps,
    }
    wb.log(
        {
            **{k: float(v) for k, v in train_metrics.items()},
            **{f"train/{k}": v for k, v in stats.items()},
        },
        step=step,
    )
    return stats, {k: float(v) for k, v in train_metrics.items()}, sync_time


@lru_cache(maxsize=1)
def get_lpips_loss_fn():
    return LPIPS(pretrained_network=LPIPS_PRETRAINED_NETWORK)


def compute_lpips_loss(
    original: jax.Array,
    reconstructed: jax.Array,
    preprocessor: TokenizerPreprocessor,
) -> jax.Array:
    original_images = preprocessor.patches_to_images(original)
    reconstructed_images = preprocessor.patches_to_images(reconstructed)
    original_images = original_images * 2.0 - 1.0
    reconstructed_images = reconstructed_images * 2.0 - 1.0
    original_images = original_images.reshape((-1, *original_images.shape[2:]))
    reconstructed_images = reconstructed_images.reshape((-1, *reconstructed_images.shape[2:]))
    return jnp.mean(get_lpips_loss_fn()(original_images, reconstructed_images))


def update_loss_ema(
    state: TokenizerTrainState,
    mse_loss: jax.Array,
    lpips_loss: jax.Array,
) -> TokenizerTrainState:
    mse_loss = mse_loss.astype(jnp.float32)
    lpips_loss = lpips_loss.astype(jnp.float32)
    step_size = jnp.asarray(1.0 - LOSS_RMS_DECAY, dtype=mse_loss.dtype)
    return state.replace(
        mse_sq_ema=optax.incremental_update(
            jnp.square(mse_loss), state.mse_sq_ema.astype(mse_loss.dtype), step_size
        ),
        lpips_sq_ema=optax.incremental_update(
            jnp.square(lpips_loss),
            state.lpips_sq_ema.astype(lpips_loss.dtype),
            step_size,
        ),
    )


def compute_loss_metrics(
    state: TokenizerTrainState,
    batch: VideoDataset,
    reconstructed: jax.Array,
    mask: jax.Array,
    lpips_weight: float,
    preprocessor: TokenizerPreprocessor,
):
    original = batch["video"].astype(jnp.float32) / 255.0
    reconstructed = reconstructed.astype(jnp.float32)
    mask = jnp.expand_dims(mask, axis=-1).astype(reconstructed.dtype)
    reconstruction_error = reconstructed - original

    mse_loss = jnp.mean(jnp.square(reconstruction_error))
    mse_rms = jnp.sqrt(state.mse_sq_ema.astype(mse_loss.dtype) + LOSS_RMS_EPS)
    normalized_mse_loss = mse_loss / jax.lax.stop_gradient(mse_rms)

    lpips_rms = jnp.sqrt(state.lpips_sq_ema.astype(mse_loss.dtype) + LOSS_RMS_EPS)
    if lpips_weight > 0:
        lpips_loss = compute_lpips_loss(original, reconstructed, preprocessor)
    else:
        lpips_loss = jnp.zeros((), dtype=mse_loss.dtype)
    normalized_lpips_loss = lpips_loss / jax.lax.stop_gradient(lpips_rms)

    raw_loss = mse_loss + lpips_weight * lpips_loss
    loss = normalized_mse_loss + lpips_weight * normalized_lpips_loss
    metrics = {
        "loss": loss,
        "raw_loss": raw_loss,
        "mse_loss": mse_loss,
        "lpips_loss": lpips_loss,
        "normalized_mse_loss": normalized_mse_loss,
        "normalized_lpips_loss": normalized_lpips_loss,
        "mse_rms": mse_rms,
        "lpips_rms": lpips_rms,
        "mask_ratio": jnp.mean(mask),
    }
    return loss, metrics


def train_step(
    state: TokenizerTrainState,
    batch: VideoDataset,
    base_sample_key: jax.Array,
    global_step: int,
    lpips_weight: float,
    preprocessor: TokenizerPreprocessor,
):
    sample_key = fold_in_many(base_sample_key, global_step)

    def loss_fn(params):
        reconstructed, mask = state.apply_fn(
            params,
            batch,
            method=Tokenizer.reconstruct,
            rngs={"sample": sample_key},
        )
        return compute_loss_metrics(
            state,
            batch,
            reconstructed,
            mask,
            lpips_weight,
            preprocessor,
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = update_loss_ema(
        state,
        metrics["mse_loss"],
        metrics["lpips_loss"],
    )
    return state, metrics


def eval_step(
    state: TokenizerTrainState,
    batch: VideoDataset,
    base_sample_key: jax.Array,
    global_step: int,
    batch_index: int,
    lpips_weight: float,
    preprocessor: TokenizerPreprocessor,
):
    sample_key = fold_in_many(base_sample_key, global_step, batch_index)
    model_key, frame_key = jax.random.split(sample_key)
    reconstructed, mask = state.apply_fn(
        state.params,
        batch,
        mask_prob=0.1,
        independent=jnp.zeros((batch["video"].shape[0],), dtype=bool),
        method=Tokenizer.reconstruct,
        rngs={"sample": model_key},
    )
    _, metrics = compute_loss_metrics(
        state,
        batch,
        reconstructed,
        mask,
        lpips_weight,
        preprocessor,
    )
    sampled_frames = sample_sequence_frames(
        batch,
        reconstructed,
        mask,
        preprocessor,
        frame_key,
    )
    return metrics, sampled_frames


def sample_sequence_frames(
    batch: VideoDataset,
    reconstructed: jax.Array,
    mask: jax.Array,
    preprocessor: TokenizerPreprocessor,
    frame_key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    original = batch["video"].astype(jnp.float32) / 255.0
    reconstructed = reconstructed.astype(jnp.float32)
    masked_input = original * (1.0 - jnp.expand_dims(mask, axis=-1).astype(original.dtype))

    batch_size, seq_len, _, _ = original.shape
    batch_indices = jnp.arange(batch_size)
    frame_indices = jax.random.randint(frame_key, (batch_size,), 0, seq_len)

    original = original[batch_indices, frame_indices]
    reconstructed = reconstructed[batch_indices, frame_indices]
    masked_input = masked_input[batch_indices, frame_indices]
    sampled_frames = []
    for patches in (original, reconstructed, masked_input):
        images = preprocessor.patches_to_images(patches)
        sampled_frames.append(jnp.clip(jnp.rint(images * 255.0), 0, 255).astype(jnp.uint8))
    return tuple(sampled_frames)


def build_reconstruction_grid(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    masked_inputs: np.ndarray,
    frame_seed: int,
    num_frames: int,
) -> np.ndarray:
    total_frames = originals.shape[0]
    num_frames = min(int(num_frames), total_frames)
    frame_indices = np.random.default_rng(frame_seed).choice(
        total_frames, size=num_frames, replace=False
    )

    originals = originals[frame_indices]
    reconstructions = reconstructions[frame_indices]
    masked_inputs = masked_inputs[frame_indices]

    col_sep = np.full((originals.shape[1], 2, 3), 255, dtype=np.uint8)
    rows = []
    for original_frame, reconstructed_frame, masked_frame in zip(
        originals, reconstructions, masked_inputs, strict=True
    ):
        rows.append(
            np.concatenate(
                [original_frame, col_sep, reconstructed_frame, col_sep, masked_frame],
                axis=1,
            )
        )
    row_sep = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate(
        [
            row if idx == 0 else np.concatenate([row_sep, row], axis=0)
            for idx, row in enumerate(rows)
        ],
        axis=0,
    )


@hydra.main(config_path="config", version_base=None)
def main(cfg: DictConfig):
    maybe_initialize_distributed(logger=logger)

    process_index = jax.process_index()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()
    is_primary_process = process_index == 0
    mesh, fsdp_enabled, fsdp_axis_size = build_fsdp_mesh(cfg)
    min_shard_size = int(cfg_select(cfg, "fsdp.min_shard_size", 4096))
    log_sharding = bool(cfg_select(cfg, "fsdp.log_sharding", True))
    batch_pspec = batch_partition_spec() if fsdp_enabled else P()
    batch_sharding = NamedSharding(mesh, batch_pspec)
    metrics_sharding = replicated_sharding(mesh)

    logger.info(
        "JAX backend: %s process=%d/%d local_devices=%d global_devices=%d devices=%s",
        jax.default_backend(),
        process_index,
        process_count,
        local_device_count,
        jax.device_count(),
        jax.local_devices(),
    )
    logger.info(
        "FSDP mesh: enabled=%s mesh_shape=%s data_axis=%d fsdp_axis=%d batch_pspec=%s "
        "min_shard_size=%d",
        fsdp_enabled,
        mesh.devices.shape,
        mesh.shape[DATA_AXIS],
        mesh.shape[FSDP_AXIS],
        batch_pspec,
        min_shard_size,
    )
    if not fsdp_enabled and process_count > 1:
        raise ValueError(
            "Multi-process tokenizer training requires fsdp.enabled=true or auto. "
            "Replicated batch sharding would receive different per-process Grain batches."
        )
    if not fsdp_enabled and jax.device_count() > 1:
        logger.warning(
            "Multiple JAX devices are visible but fsdp.enabled is false; tokenizer training "
            "will use replicated sharding."
        )

    wb = WandbLogger(cfg, enabled=bool(cfg.wandb.enabled) and is_primary_process)
    total_steps = int(cfg.total_steps)

    train_source = VideoDataSource(cfg.dataset.train_dir)
    eval_source = VideoDataSource(cfg.dataset.eval_dir)
    logger.info(
        "Loaded %d training videos and %d eval videos",
        len(train_source),
        len(eval_source),
    )
    batch_size_per_process = int(cfg.dataset.batch_size)
    if fsdp_enabled and batch_size_per_process % local_device_count != 0:
        raise ValueError(
            "cfg.dataset.batch_size is interpreted per process and must be divisible by "
            "jax.local_device_count() for FSDP input sharding; got "
            f"batch_size={batch_size_per_process} local_device_count={local_device_count}"
        )
    logger.info(
        "Batch layout: per_process=%d per_device=%d global=%d",
        batch_size_per_process,
        batch_size_per_process // local_device_count if fsdp_enabled else batch_size_per_process,
        batch_size_per_process * process_count,
    )
    global_batch_size = batch_size_per_process * process_count
    effective_read_threads = max(int(cfg.dataset.worker_count), 1) * int(cfg.dataset.num_threads)
    logger.info(
        "Data loader settings: worker_count=%d num_threads=%d "
        "prefetch_buffer_size=%d effective_read_threads=%d",
        int(cfg.dataset.worker_count),
        int(cfg.dataset.num_threads),
        int(cfg.dataset.prefetch_buffer_size),
        effective_read_threads,
    )
    profiler = TrainingProfiler(
        cfg,
        metadata={
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "jax": {
                "version": jax.__version__,
                "backend": jax.default_backend(),
                "process_index": process_index,
                "process_count": process_count,
                "local_device_count": local_device_count,
                "global_device_count": jax.device_count(),
                "local_devices": [str(device) for device in jax.local_devices()],
            },
            "fsdp": {
                "enabled": fsdp_enabled,
                "mesh_shape": dict(mesh.shape),
                "data_axis_size": int(mesh.shape[DATA_AXIS]),
                "fsdp_axis_size": int(mesh.shape[FSDP_AXIS]),
                "batch_partition_spec": str(batch_pspec),
                "min_shard_size": min_shard_size,
            },
            "batch": {
                "per_process": batch_size_per_process,
                "global": global_batch_size,
                "per_device": batch_size_per_process // local_device_count
                if fsdp_enabled
                else batch_size_per_process,
            },
            "dataset": {
                "train_dir": str(cfg.dataset.train_dir),
                "eval_dir": str(cfg.dataset.eval_dir),
                "frame_length": int(cfg.dataset.frame_length),
                "worker_count": int(cfg.dataset.worker_count),
                "num_threads": int(cfg.dataset.num_threads),
                "prefetch_buffer_size": int(cfg.dataset.prefetch_buffer_size),
            },
            "training": {
                "learning_rate": float(cfg.learning_rate),
                "lpips_weight": float(cfg.lpips_weight),
                "log_interval": int(cfg.log_interval),
                "eval_steps": int(cfg.eval_steps),
                "total_steps": total_steps,
            },
            "tokenizer": OmegaConf.to_container(cfg.tokenizer, resolve=True),
        },
        process_index=process_index,
    )
    profiler.log_configuration()
    preprocessor = TokenizerPreprocessor(
        resize_shape=tuple(cfg.tokenizer.resize_shape),
        pad_width=tuple(cfg.tokenizer.pad_width),
        patch_size=int(cfg.tokenizer.patch_size),
    )
    logger.info(
        "LPIPS settings: weight=%.3f",
        float(cfg.lpips_weight),
    )
    crop_transform = RandomVideoCrop(cfg.dataset.frame_length)
    preprocess_transform = preprocessor.as_grain_transform()

    def make_loader(source, shuffle: bool, drop_remainder: bool, seed: int):
        sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.ShardByJaxProcess()
            if process_count > 1
            else grain.NoSharding(),
            shuffle=shuffle,
            seed=seed,
        )
        read_options = grain.ReadOptions(
            num_threads=int(cfg.dataset.num_threads),
            prefetch_buffer_size=int(cfg.dataset.prefetch_buffer_size),
        )
        return grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=[
                crop_transform,
                preprocess_transform,
                grain.Batch(
                    batch_size=batch_size_per_process,
                    drop_remainder=drop_remainder,
                ),
            ],
            worker_count=cfg.dataset.worker_count,
            read_options=read_options,
        )

    _t = time.monotonic()
    train_dataloader = make_loader(
        train_source,
        shuffle=True,
        drop_remainder=True,
        seed=int(cfg.seed),
    )
    logger.info("Train DataLoader creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    eval_loader = make_loader(
        eval_source,
        shuffle=False,
        drop_remainder=fsdp_enabled,
        seed=int(cfg.seed),
    )
    logger.info("Eval DataLoader creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    sample_batch = next(iter(train_dataloader))
    logger.info("First batch fetch took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    key = jax.random.key(cfg.seed)
    init_key, init_sample_key, train_key, eval_key = jax.random.split(key, num=4)
    model = instantiate(cfg.tokenizer)
    params = model.init(
        {"params": init_key, "sample": init_sample_key},
        sample_batch,
        method=Tokenizer.reconstruct,
    )
    logger.info("Model init took %.1fs", time.monotonic() - _t)

    if cfg.lpips_weight > 0:
        _t = time.monotonic()
        get_lpips_loss_fn()
        logger.info("LPIPS init took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    optimizer = optax.adam(cfg.learning_rate)
    state = TokenizerTrainState.create(
        model.apply,
        params,
        optimizer,
    )
    logger.info("TrainState creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    state_shardings = make_array_shardings(
        state,
        mesh=mesh,
        fsdp_enabled=fsdp_enabled,
        fsdp_axis_size=fsdp_axis_size,
        min_shard_size=min_shard_size,
    )
    state = jax.tree_util.tree_map(make_global_array_from_host, state, state_shardings)
    if log_sharding:
        log_sharding_summary(params, state_shardings.params, prefix="Tokenizer params")
    logger.info("TrainState sharding took %.1fs", time.monotonic() - _t)

    batch_input_shardings = {"video": batch_sharding}
    sampled_frame_sharding = NamedSharding(mesh, batch_pspec)
    jit_train_step = jax.jit(
        train_step,
        in_shardings=(
            state_shardings,
            batch_input_shardings,
            metrics_sharding,
            metrics_sharding,
        ),
        out_shardings=(state_shardings, metrics_sharding),
        static_argnames=("lpips_weight", "preprocessor"),
        donate_argnums=(0,),
    )
    jit_eval_step = jax.jit(
        eval_step,
        in_shardings=(
            state_shardings,
            batch_input_shardings,
            metrics_sharding,
            metrics_sharding,
            metrics_sharding,
        ),
        out_shardings=(
            metrics_sharding,
            (sampled_frame_sharding, sampled_frame_sharding, sampled_frame_sharding),
        ),
        static_argnames=("lpips_weight", "preprocessor"),
    )
    train_key = put_replicated(train_key, mesh)
    eval_key = put_replicated(eval_key, mesh)

    _t = time.monotonic()
    checkpoint_manager: CheckpointManager = instantiate(cfg.checkpoint.manager)
    if is_primary_process:
        checkpoint_manager.save_metadata({"config": OmegaConf.to_container(cfg, resolve=False)})
    logger.info("CheckpointManager creation took %.1fs", time.monotonic() - _t)
    train_iterator = iter(train_dataloader)
    last_checkpoint_step: int | None = None

    def save_checkpoint(step: int, force: bool = False) -> None:
        nonlocal last_checkpoint_step
        saved = checkpoint_manager.save(
            step=step,
            state=state,
            extra_items=iterator_items(),
            force=force,
        )
        if not saved:
            return
        last_checkpoint_step = int(step)
        save_model_export(checkpoint_manager.directory, step, cfg.tokenizer, state.params)
        if is_primary_process:
            save_preprocessor_export(
                checkpoint_manager.directory,
                step,
                preprocessor.export_config(),
            )

    def iterator_items():
        return {"train_iterator": train_iterator}

    resume_spec = cfg.checkpoint.resume_step
    resume_step = None
    if resume_spec is not None:
        if isinstance(resume_spec, str) and resume_spec.strip().lower() == "latest":
            resume_step = checkpoint_manager.latest_step()
            if resume_step is None:
                logger.info(
                    "No tokenizer checkpoint found in %s; starting fresh.",
                    checkpoint_manager.directory,
                )
        else:
            resume_step = int(resume_spec)

    if resume_step is not None:
        state = checkpoint_manager.restore(
            target=state, step=resume_step, extra_items=iterator_items()
        )
        logger.info("Resumed tokenizer training from step %d", int(state.step))

    step = int(jax.device_get(state.step))
    logger.info("Tokenizer training target step: %d", total_steps)

    if step >= total_steps:
        logger.info(
            "Current step %d is already at or above total_steps=%d; exiting.",
            step,
            total_steps,
        )
        profiler.close(final_step=step)
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()
        wb.finish()
        return

    timing_start_step = step
    timing_data_time = timing_transfer_time = timing_dispatch_time = timing_sync_time = 0.0
    logger.info(
        "Asynchronous timing mode enabled for tokenizer training; timing logs are averaged "
        "over each logging window."
    )
    while True:
        current_step = step
        profile_step = current_step + 1
        profiler.maybe_start_trace(profile_step)
        step_start = time.monotonic()
        sync_time = 0.0
        with profiler.step_context(profile_step):
            with profiler.annotation("tokenizer_data", profile_step):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_dataloader)
                    batch = next(train_iterator)
                data_done = time.monotonic()

            with profiler.annotation("tokenizer_host_to_global_array", profile_step):
                batch = put_global_batch(batch, batch_sharding)
                transfer_done = time.monotonic()

            with profiler.annotation("tokenizer_jit_train_step", profile_step):
                state, metrics = jit_train_step(
                    state,
                    batch,
                    train_key,
                    jnp.asarray(current_step, dtype=jnp.int32),
                    float(cfg.lpips_weight),
                    preprocessor,
                )
                train_dispatched = time.monotonic()

            if profiler.should_sync_step(profile_step):
                with profiler.annotation("tokenizer_sync", profile_step):
                    sync_start = time.monotonic()
                    metrics = jax.block_until_ready(metrics)
                    sync_time = time.monotonic() - sync_start
        step_total_time = time.monotonic() - step_start
        profiler.maybe_stop_trace(profile_step)

        step = current_step + 1
        timing_data_time += data_done - step_start
        timing_transfer_time += transfer_done - data_done
        timing_dispatch_time += train_dispatched - transfer_done
        timing_sync_time += sync_time

        if profiler.blocking_timing or profiler.should_trace_step(profile_step):
            step_record = {
                "type": "train_step",
                "step": step,
                "data_seconds": data_done - step_start,
                "transfer_seconds": transfer_done - data_done,
                "dispatch_seconds": train_dispatched - transfer_done,
                "sync_seconds": sync_time,
                "total_seconds": step_total_time,
                "global_batch_size": global_batch_size,
            }
            if profiler.collect_memory_stats:
                step_record["memory"] = collect_local_memory_stats()
            if profiler.should_sync_step(profile_step):
                step_record["metrics"] = {
                    k: float(v) for k, v in jax.device_get(metrics).items()
                }
            profiler.record(step_record)

        timing_stats = None
        if step % cfg.log_interval == 0:
            timing_stats, train_metrics, timing_sync_time = log_train_timing(
                wb,
                step=step,
                start_step=timing_start_step,
                metrics=metrics,
                data_time=timing_data_time,
                transfer_time=timing_transfer_time,
                dispatch_time=timing_dispatch_time,
                sync_time=timing_sync_time,
                global_batch_size=global_batch_size,
            )
            profile_record = {
                "type": "train_window",
                "step": step,
                "start_step": timing_start_step,
                "window_steps": step - timing_start_step,
                "global_batch_size": global_batch_size,
                "steps_per_second": timing_stats["sps"],
                "examples_per_second": timing_stats["examples_per_second"],
                "data_seconds_per_step": timing_stats["data_time"],
                "transfer_seconds_per_step": timing_stats["transfer_time"],
                "dispatch_seconds_per_step": timing_stats["dispatch_time"],
                "sync_seconds_per_step": timing_stats["sync_time"],
                "compute_seconds_per_step": timing_stats["compute_time"],
                "wall_seconds_per_step": timing_stats["wall_time"],
                "metrics": train_metrics,
            }
            if profiler.collect_memory_stats:
                profile_record["memory"] = collect_local_memory_stats()
            profiler.record(
                profile_record,
                flush=True,
            )
            timing_start_step = step
            timing_data_time = timing_transfer_time = timing_dispatch_time = 0.0
            timing_sync_time = 0.0

        t_eval = 0.0
        if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
            t_eval_start = time.monotonic()
            totals: dict[str, float] = {}
            num_batches = 0
            eval_batches = list(itertools.islice(iter(eval_loader), cfg.dataset.eval.max_batches))
            if fsdp_enabled:
                global_eval_batch_counts = np.asarray(
                    multihost_utils.process_allgather(
                        np.asarray(len(eval_batches), dtype=np.int32)
                    )
                )
                eval_batches = eval_batches[: int(np.min(global_eval_batch_counts))]
            vis_original_batches = []
            vis_reconstruction_batches = []
            vis_masked_batches = []
            for batch_idx, eval_batch in enumerate(eval_batches):
                batch_metrics, sampled_frames = jit_eval_step(
                    state,
                    put_global_batch(eval_batch, batch_sharding),
                    eval_key,
                    jnp.asarray(step, dtype=jnp.int32),
                    jnp.asarray(batch_idx, dtype=jnp.int32),
                    float(cfg.lpips_weight),
                    preprocessor,
                )
                sampled_frames = host_local_batch(sampled_frames, mesh, batch_pspec)
                sampled_frames = jax.device_get(sampled_frames)
                batch_metrics = jax.device_get(batch_metrics)
                for k, v in batch_metrics.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                sampled_originals, sampled_reconstructions, sampled_masked_inputs = sampled_frames
                vis_original_batches.append(np.asarray(sampled_originals))
                vis_reconstruction_batches.append(np.asarray(sampled_reconstructions))
                vis_masked_batches.append(np.asarray(sampled_masked_inputs))
                num_batches += 1
            if num_batches > 0:
                eval_metrics = {k: v / num_batches for k, v in totals.items()}
                wb.log(
                    {f"eval/{k}": v for k, v in eval_metrics.items()},
                    step=step,
                )
                if vis_original_batches:
                    grid = build_reconstruction_grid(
                        np.concatenate(vis_original_batches, axis=0),
                        np.concatenate(vis_reconstruction_batches, axis=0),
                        np.concatenate(vis_masked_batches, axis=0),
                        frame_seed=make_host_seed(cfg.seed, step, num_batches),
                        num_frames=int(cfg.dataset.eval.log_frames),
                    )
                    wb.log_image(
                        "eval/reconstructions",
                        grid,
                        step=step,
                        caption="Columns: original, reconstructed, masked input",
                    )
            t_eval = time.monotonic() - t_eval_start
            logger.info("Eval at step %d - %d batches in %.3fs", step, num_batches, t_eval)

        if checkpoint_manager.should_save(step):
            save_checkpoint(step)

        if timing_stats is not None:
            logger.info(
                "Step %d - sps: %.2f, data: %.3fs, transfer: %.3fs, compute: %.3fs, "
                "sync: %.3fs, wall: %.3fs, eval: %.3fs",
                step,
                timing_stats["sps"],
                timing_stats["data_time"],
                timing_stats["transfer_time"],
                timing_stats["compute_time"],
                timing_stats["sync_time"],
                timing_stats["wall_time"],
                t_eval,
            )
        if step >= total_steps:
            logger.info("Reached total_steps=%d; stopping tokenizer training.", total_steps)
            break

    if step >= total_steps and last_checkpoint_step != step:
        save_checkpoint(step, force=True)
    profiler.close(final_step=step)
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    wb.finish()


if __name__ == "__main__":
    main()
