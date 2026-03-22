import logging
import os

from flax import serialization

logger = logging.getLogger(__name__)


def load_checkpoint(ckpt_path: str, target):
    with open(ckpt_path, "rb") as f:
        params = serialization.from_bytes(target, f.read())
    logger.info("Checkpoint loaded from %s", ckpt_path)
    return params


def save_checkpoint(params, output_dir: str, step: int) -> None:
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"step_{step}.msgpack")
    with open(ckpt_path, "wb") as f:
        f.write(serialization.to_bytes(params))
    logger.info("Checkpoint saved to %s", ckpt_path)
