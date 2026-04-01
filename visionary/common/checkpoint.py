from __future__ import annotations

import logging
from os import PathLike
from typing import Any

import jax
import orbax.checkpoint as ocp
from etils import epath
from flax.training.train_state import TrainState

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Small Orbax wrapper for async Flax TrainState checkpoints."""

    def __init__(
        self,
        directory: str | PathLike[str],
        options: ocp.CheckpointManagerOptions,
    ) -> None:
        self.directory = epath.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self._manager = ocp.CheckpointManager(
            self.directory,
            options=options,
        )

    def __enter__(self) -> CheckpointManager:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.wait_until_finished()
        self.close()

    def all_steps(self) -> list[int]:
        return [int(step) for step in self._manager.all_steps()]

    def should_save(self, step: int) -> bool:
        return self._manager.should_save(step)

    def save(
        self,
        *,
        step: int,
        state: TrainState,
        metrics: Any | None = None,
        force: bool = False,
        wait: bool = False,
    ) -> bool:
        saved = self._manager.save(
            step,
            args=ocp.args.StandardSave(state),
            metrics=metrics,
            force=force,
        )
        if saved and wait:
            self.wait_until_finished()
        if saved:
            logger.info("Checkpoint scheduled at step %d in %s", step, self.directory)
        return saved

    def restore(
        self,
        *,
        target: TrainState,
        step: int | None = None,
        params_only: bool = False,
    ) -> Any:
        self.wait_until_finished()

        if step is None:
            step = self._manager.latest_step()
            if step is None:
                raise FileNotFoundError(f"No checkpoints found in {self.directory}")

        abstract_target = jax.tree_util.tree_map(
            ocp.utils.to_shape_dtype_struct, target
        )
        try:
            restored = self._manager.restore(
                step,
                args=ocp.args.StandardRestore(abstract_target),
            )
        except ValueError as exc:
            if "Composite" not in str(exc):
                raise
            restored = self._manager.restore(
                step,
                args=ocp.args.Composite(
                    default=ocp.args.StandardRestore(abstract_target)
                ),
            )["default"]
        logger.info("Checkpoint restored from step %d in %s", step, self.directory)

        if params_only:
            return restored.params
        return restored

    def wait_until_finished(self) -> None:
        self._manager.wait_until_finished()

    def close(self) -> None:
        self._manager.close()
