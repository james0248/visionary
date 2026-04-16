from __future__ import annotations

import json
import logging
from os import PathLike
from typing import Any

import grain.checkpoint as grain_checkpoint
import jax
import orbax.checkpoint as ocp
from etils import epath
from flax.training.train_state import TrainState

logger = logging.getLogger(__name__)
METADATA_FILENAME = "metadata.json"


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

    def latest_step(self) -> int | None:
        step = self._manager.latest_step()
        return None if step is None else int(step)

    def should_save(self, step: int) -> bool:
        return self._manager.should_save(step)

    def save_metadata(
        self,
        metadata: dict[str, Any],
        filename: str = METADATA_FILENAME,
    ) -> None:
        path = self.directory / filename
        with path.open("w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    def load_metadata(self, filename: str = METADATA_FILENAME) -> dict[str, Any]:
        path = self.directory / filename
        if not path.exists():
            raise FileNotFoundError(f"No metadata file found at {path}")
        with path.open() as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected checkpoint metadata at {path} to be a JSON object.")
        return loaded

    def save(
        self,
        step: int,
        state: TrainState,
        extra_items: dict[str, Any] | None = None,
        metrics: Any | None = None,
        force: bool = False,
        wait: bool = False,
    ) -> bool:
        args = ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            **{
                name: grain_checkpoint.CheckpointSave(item)
                for name, item in (extra_items or {}).items()
            },
        )

        saved = self._manager.save(step, args=args, metrics=metrics, force=force)
        if saved and wait:
            self.wait_until_finished()
        if saved:
            logger.info("Checkpoint scheduled at step %d in %s", step, self.directory)
        return saved

    def restore(
        self,
        target: TrainState,
        step: int | None = None,
        extra_items: dict[str, Any] | None = None,
        params_only: bool = False,
    ) -> Any:
        self.wait_until_finished()

        if step is None:
            step = self.latest_step()
            if step is None:
                raise FileNotFoundError(f"No checkpoints found in {self.directory}")

        abstract_target = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, target)
        restored = self._manager.restore(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_target),
                **{
                    name: grain_checkpoint.CheckpointRestore(item)
                    for name, item in (extra_items or {}).items()
                },
            ),
        )["state"]
        logger.info("Checkpoint restored from step %d in %s", step, self.directory)

        if params_only:
            return restored.params
        return restored

    def wait_until_finished(self) -> None:
        self._manager.wait_until_finished()

    def close(self) -> None:
        self._manager.close()
