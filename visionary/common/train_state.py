from typing import Any

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.training.train_state import TrainState


class TargetTrainState(TrainState):
    target_params: FrozenDict[str, Any]


class TokenizerTrainState(TrainState):
    mse_sq_ema: jax.Array
    l1_sq_ema: jax.Array
    lpips_sq_ema: jax.Array
    motion_sq_ema: jax.Array

    @classmethod
    def create(cls, apply_fn, params, tx):
        return super().create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            mse_sq_ema=jnp.ones((), dtype=jnp.float32),
            l1_sq_ema=jnp.ones((), dtype=jnp.float32),
            lpips_sq_ema=jnp.ones((), dtype=jnp.float32),
            motion_sq_ema=jnp.ones((), dtype=jnp.float32),
        )


class DynamicsTrainState(TrainState):
    pass
