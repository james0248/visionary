import logging
import os
from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

TPU_DISTRIBUTED_ENV_VARS = (
    "TPU_ML_PLATFORM",
    "TPU_PROCESS_ADDRESSES",
    "TPU_WORKER_ID",
    "CLOUD_TPU_TASK_ID",
)


@dataclass(frozen=True)
class DistributedInitResult:
    mode: str
    is_initialized: bool


def fold_in_many(key: jax.Array, *terms: int | jax.Array) -> jax.Array:
    """
    Derive a deterministic subkey from a base key and any number of counters.
    """
    for term in terms:
        key = jax.random.fold_in(key, term)
    return key


def init_distributed_if_pod(
    logger: logging.Logger | None = None,
    environ: Mapping[str, str] | None = None,
) -> DistributedInitResult:
    """
    Call first thing in main. Joins the pod's jax.distributed cluster when
    TPU pod env vars are present; no-op on a single machine.
    """
    environ = os.environ if environ is None else environ

    if jax.distributed.is_initialized():
        return DistributedInitResult(mode="already_initialized", is_initialized=True)

    if any(name in environ for name in TPU_DISTRIBUTED_ENV_VARS):
        if logger is not None:
            logger.info(
                "Initializing JAX distributed for a Cloud TPU pod job. "
                "On multi-host jobs, start this script on every worker."
            )
        jax.distributed.initialize()
        return DistributedInitResult(mode="auto_detect", is_initialized=True)

    return DistributedInitResult(mode="none", is_initialized=False)


def shard_from_host(value, sharding: NamedSharding):
    """
    Turn a host value that is identical on every process into a global array
    laid out per `sharding`; each device pulls its own slice, no collectives.
    For distributing host-built state; prefer jit(init, out_shardings=...).
    """
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


def replicate_to_mesh(value, mesh: Mesh):
    """
    Eagerly place a small host value (PRNG keys included) replicated across the
    mesh. Rarely needed: jit auto-commits identical host-local inputs to
    replicated in_shardings; use only before any jit boundary.
    """
    sharding = NamedSharding(mesh, P())
    if jax.process_count() == 1:
        return jax.device_put(value, sharding)
    # host_local_array_to_global_array requires contiguous per-host subcubes,
    # which TPU ICI-optimized meshes violate; assemble the replicated array
    # from explicit per-device copies instead
    value = jnp.asarray(value)
    is_key = jnp.issubdtype(value.dtype, jax.dtypes.prng_key)
    data = jax.random.key_data(value) if is_key else value
    data = np.asarray(data)
    arrays = [jax.device_put(data, d) for d in mesh.local_devices]
    out = jax.make_array_from_single_device_arrays(data.shape, sharding, arrays)
    return jax.random.wrap_key_data(out, impl=jax.random.key_impl(value)) if is_key else out


def data_parallel_mesh(axis_name: str = "data") -> Mesh:
    """
    1-D data-parallel mesh over every device in the job. Build once at startup;
    shard batches with P(axis_name), replicate state with P().
    """
    # explicit Auto: make_mesh defaults to explicit-sharding axes, which our
    # models are not annotated for
    return jax.make_mesh((jax.device_count(),), (axis_name,), axis_types=(jax.sharding.AxisType.Auto,))


def local_batch_to_global(batch, batch_sharding: NamedSharding):
    """
    Declare each process's data-loader output as its slice of the global batch;
    rows land on local devices only. Call on every batch before the jitted step.
    """
    return jax.tree_util.tree_map(
        lambda value: jax.make_array_from_process_local_data(batch_sharding, value),
        batch,
    )


def global_batch_to_local(batch):
    """
    Pull this process's rows of a batch-sharded global array back to host numpy.
    For per-example outputs like eval images; replicated scalars just need
    jax.device_get.
    """
    if jax.process_count() == 1:
        return batch

    def to_local(x):
        shards = {}
        for s in x.addressable_shards:
            start = s.index[0].start if x.ndim and s.index else 0
            shards.setdefault(start or 0, s.data)
        return (
            np.concatenate([np.asarray(shards[k]) for k in sorted(shards)], axis=0)
            if x.ndim
            else np.asarray(next(iter(shards.values())))
        )

    return jax.tree_util.tree_map(to_local, batch)
