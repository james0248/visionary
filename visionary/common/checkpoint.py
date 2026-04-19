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
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
METADATA_FILENAME = "metadata.json"
MODEL_EXPORT_DIRNAME = "model"
PREPROCESSOR_EXPORT_DIRNAME = "preprocessor"
PREPROCESSOR_CONFIG_FIELDS = ("resize_shape", "pad_width", "patch_size")


def model_export_dir(directory: str | PathLike[str]) -> epath.Path:
    return epath.Path(directory) / MODEL_EXPORT_DIRNAME


def preprocessor_export_dir(directory: str | PathLike[str]) -> epath.Path:
    return epath.Path(directory) / PREPROCESSOR_EXPORT_DIRNAME


def _latest_export_step(export_dir: epath.Path) -> int | None:
    steps = _export_steps(export_dir)
    return steps[-1] if steps else None


def _export_steps(export_dir: epath.Path) -> list[int]:
    if not export_dir.exists():
        return []

    return sorted(
        int(path.name)
        for path in export_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def resolve_model_export_step(directory: str | PathLike[str], step: int | None) -> int:
    if step is not None:
        return int(step)

    latest_step = latest_model_export_step(directory)
    if latest_step is None:
        raise FileNotFoundError(f"No model exports found in {model_export_dir(directory)}")
    return latest_step


def latest_model_export_step(directory: str | PathLike[str]) -> int | None:
    return _latest_export_step(model_export_dir(directory))


def latest_preprocessor_export_step(directory: str | PathLike[str]) -> int | None:
    return _latest_export_step(preprocessor_export_dir(directory))


def _find_preprocessor_export_step(
    directory: str | PathLike[str],
    step: int | None,
) -> int | None:
    export_steps = _export_steps(preprocessor_export_dir(directory))
    if not export_steps:
        return None

    if step is None:
        return export_steps[-1]

    requested_step = int(step)
    for export_step in reversed(export_steps):
        if export_step <= requested_step:
            return export_step
    return export_steps[0]


def model_export_path(directory: str | PathLike[str], step: int) -> epath.Path:
    return model_export_dir(directory) / str(int(step))


def preprocessor_export_path(directory: str | PathLike[str], step: int) -> epath.Path:
    return preprocessor_export_dir(directory) / str(int(step))


def _model_export_checkpointer() -> ocp.Checkpointer:
    return ocp.Checkpointer(
        ocp.CompositeCheckpointHandler(
            config=ocp.JsonCheckpointHandler(),
            variables=ocp.StandardCheckpointHandler(),
        )
    )


def _json_checkpointer() -> ocp.Checkpointer:
    return ocp.Checkpointer(ocp.JsonCheckpointHandler())


def _save_json_export(path: epath.Path, payload: dict[str, Any]) -> None:
    checkpointer = _json_checkpointer()
    checkpointer.save(
        path.as_posix(),
        args=ocp.args.JsonSave(payload),
        force=True,
    )
    checkpointer.close()


def _restore_json_export(path: epath.Path) -> dict[str, Any]:
    checkpointer = _json_checkpointer()
    restored = checkpointer.restore(
        path.as_posix(),
        args=ocp.args.JsonRestore(),
    )
    checkpointer.close()
    return dict(restored)


def _preprocessor_config_from_model_config(model_config: DictConfig) -> dict[str, Any]:
    missing = [field for field in PREPROCESSOR_CONFIG_FIELDS if field not in model_config]
    if missing:
        raise FileNotFoundError(
            "No preprocessor export found and model config does not contain the "
            f"required preprocessor fields: {missing}"
        )
    return {
        "resize_shape": list(model_config["resize_shape"]),
        "pad_width": list(model_config["pad_width"]),
        "patch_size": int(model_config["patch_size"]),
    }


def save_model_export(
    directory: str | PathLike[str],
    step: int,
    model_config: DictConfig,
    variables: Any,
) -> None:
    export_dir = model_export_dir(directory)
    export_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.create(OmegaConf.to_container(model_config, resolve=False))
    checkpointer = _model_export_checkpointer()
    checkpointer.save(
        model_export_path(directory, step).as_posix(),
        args=ocp.args.Composite(
            config=ocp.args.JsonSave(OmegaConf.to_container(config, resolve=True)),
            variables=ocp.args.StandardSave(variables),
        ),
        force=True,
    )
    checkpointer.close()


def save_preprocessor_export(
    directory: str | PathLike[str],
    step: int,
    preprocessor_config: DictConfig | dict[str, Any],
) -> None:
    export_dir = preprocessor_export_dir(directory)
    existing_step = _latest_export_step(export_dir)
    if existing_step is not None:
        logger.debug(
            "Skipping preprocessor export for step %s; already saved at step %s.",
            step,
            existing_step,
        )
        return

    export_dir.mkdir(parents=True, exist_ok=True)

    if OmegaConf.is_config(preprocessor_config):
        config = OmegaConf.to_container(preprocessor_config, resolve=True)
    else:
        config = dict(preprocessor_config)

    _save_json_export(preprocessor_export_path(directory, step), config)


def restore_model_export(
    directory: str | PathLike[str],
    step: int | None = None,
) -> tuple[DictConfig, Any]:
    step = resolve_model_export_step(directory, step)
    checkpointer = _model_export_checkpointer()
    restored = checkpointer.restore(
        model_export_path(directory, step).as_posix(),
        args=ocp.args.Composite(
            config=ocp.args.JsonRestore(),
            variables=None,
        ),
    )
    checkpointer.close()
    return OmegaConf.create(restored.config), restored.variables


def restore_preprocessor_export(
    directory: str | PathLike[str],
    step: int | None = None,
) -> dict[str, Any]:
    preprocessor_step = _find_preprocessor_export_step(directory, step)
    if preprocessor_step is not None:
        return _restore_json_export(preprocessor_export_path(directory, preprocessor_step))

    model_step = latest_model_export_step(directory) if step is None else int(step)
    if model_step is None:
        raise FileNotFoundError(
            f"No preprocessor or model exports found in {preprocessor_export_dir(directory)}"
        )

    model_config, _ = restore_model_export(directory, step=model_step)
    return _preprocessor_config_from_model_config(model_config)


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
        restore_args = ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_target),
            **{
                name: grain_checkpoint.CheckpointRestore(item)
                for name, item in (extra_items or {}).items()
            },
        )
        try:
            restored = self._manager.restore(step, args=restore_args)["state"]
        except ValueError as err:
            should_retry_without_iterators = (
                extra_items
                and "DataSource in checkpoint does not match datasource in dataloader"
                in str(err)
            )
            if not should_retry_without_iterators:
                raise
            logger.warning(
                "Checkpoint iterator state could not be restored at step %d in %s; "
                "falling back to restoring model state only. This usually means the "
                "checkpoint was created before datasource reprs were made stable.",
                step,
                self.directory,
            )
            restored = self._manager.restore(
                step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(abstract_target),
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
