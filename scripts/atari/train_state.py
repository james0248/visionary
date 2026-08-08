from typing import Any

from flax.core import FrozenDict
from flax.training.train_state import TrainState


class TargetTrainState(TrainState):
    target_params: FrozenDict[str, Any]
