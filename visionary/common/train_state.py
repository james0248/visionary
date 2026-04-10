from typing import Any

import jax
from flax.core import FrozenDict
from flax.training.train_state import TrainState


class TargetTrainState(TrainState):
    target_params: FrozenDict[str, Any]


class TokenizerTrainState(TrainState):
    mse_sq_ema: jax.Array
    l1_sq_ema: jax.Array
    lpips_sq_ema: jax.Array
    motion_sq_ema: jax.Array


class DynamicsTrainState(TrainState):
    pass
