import io
from pathlib import Path

import grain.python as grain
import jax
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from visionary.common.checkpoint import CheckpointManager
from visionary.common.train_state import DynamicsTrainState
from visionary.dynamics import DynamicsModel


def build_raw_index(shards_dir: str) -> dict[tuple[str, int, str], bytes]:
    """Map (repo, episode, camera) -> mp4 bytes from the packed video shards.

    These are the same trimmed streams the latents were encoded from, so a
    latent record's start_index indexes directly into this video.
    """
    paths = sorted(str(p) for p in Path(shards_dir).glob("*.arecord"))
    if not paths:
        raise FileNotFoundError(f"No .arecord files in {shards_dir}")
    source = grain.ArrayRecordDataSource(paths)
    index: dict[tuple[str, int, str], bytes] = {}
    for i in range(len(source)):
        with np.load(io.BytesIO(source[i])) as data:
            key = (str(data["repo"]), int(data["episode"]), str(data["camera"]))
            index[key] = data["video"].tobytes()
    return index


def load_train_config(checkpoint_dir: str) -> OmegaConf:
    """The training run stores its own resolved config next to the checkpoints."""
    manager = CheckpointManager(
        checkpoint_dir,
        instantiate({"_target_": "orbax.checkpoint.CheckpointManagerOptions"}),
    )
    metadata = manager.load_metadata()
    manager.close()
    for key in ("dynamics_config", "config"):
        if key in metadata:
            return OmegaConf.create(metadata[key])
    raise KeyError(f"No config found in checkpoint metadata at {checkpoint_dir}")


def restore_params(cfg: OmegaConf, checkpoint_dir: str, step: int | None, sample_batch):
    model = instantiate(cfg.dynamics)
    optimizer = instantiate(cfg.optimizer)

    def make_state():
        params = model.init(
            {"params": jax.random.key(0), "sample": jax.random.key(1)},
            sample_batch,
            bootstrap_rows=0,
            method=DynamicsModel.loss,
        )
        return DynamicsTrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # shapes only: allocating real optimizer state would triple the memory for
    # no reason, and restore just needs the tree structure
    abstract_state = jax.eval_shape(make_state)
    # the training run sharded these over its own mesh, so without an explicit
    # sharding orbax reuses the saved topology and rejects a different slice
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    abstract_state = jax.tree_util.tree_map(
        lambda leaf: (
            jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding)
            if hasattr(leaf, "shape")
            else leaf
        ),
        abstract_state,
    )
    manager = CheckpointManager(checkpoint_dir, instantiate(cfg.checkpoint.manager.options))
    resolved = manager.latest_step() if step is None else int(step)
    params = manager.restore(target=abstract_state, step=resolved, params_only=True)
    manager.close()
    return model, params, resolved
