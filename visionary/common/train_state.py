from typing import Any

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState


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
    ema_params: Any

    @classmethod
    def create(cls, apply_fn, params, tx):
        return super().create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            ema_params=params,
        )

    def update_ema(self, decay: float):
        ema_params = jax.tree_util.tree_map(
            lambda ema, online: decay * ema + (1.0 - decay) * online,
            self.ema_params,
            self.params,
        )
        return self.replace(ema_params=ema_params)
