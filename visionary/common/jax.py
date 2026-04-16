import logging
import os
from dataclasses import dataclass
from typing import Mapping

import jax

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
    for term in terms:
        key = jax.random.fold_in(key, term)
    return key


def maybe_initialize_distributed(
    logger: logging.Logger | None = None,
    environ: Mapping[str, str] | None = None,
) -> DistributedInitResult:
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
