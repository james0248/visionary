from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from visionary.common.checkpoint import CheckpointManager
from visionary.common.train_state import TokenizerTrainState
from visionary.tokenizer import Tokenizer


TOKENIZER_CONFIG_METADATA_KEY = "tokenizer_config"


@dataclass(frozen=True)
class TokenizerCheckpointBundle:
    config: DictConfig
    model: Tokenizer
    state: TokenizerTrainState


def normalize_tokenizer_config(cfg: DictConfig) -> DictConfig:
    normalized = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    tokenizer_cfg = normalized.get("tokenizer")
    if tokenizer_cfg is None:
        raise KeyError("Expected tokenizer config under cfg.tokenizer")

    dataset_cfg = normalized.get("dataset")
    for key in ("resize_shape", "patch_size", "pad_width"):
        if key not in tokenizer_cfg and dataset_cfg is not None and key in dataset_cfg:
            tokenizer_cfg[key] = dataset_cfg[key]

    for key in (
        "x_len",
        "y_len",
        "decoder_single_image_token",
        "bottleneck_norm",
        "attention_logit_soft_cap",
    ):
        if key in tokenizer_cfg:
            del tokenizer_cfg[key]

    return normalized


def tokenizer_checkpoint_metadata(cfg: DictConfig) -> dict:
    cfg = normalize_tokenizer_config(cfg)
    return {
        TOKENIZER_CONFIG_METADATA_KEY: OmegaConf.to_container(cfg, resolve=False),
    }


def build_tokenizer_checkpoint_bundle(
    cfg: DictConfig,
    seed: int,
) -> TokenizerCheckpointBundle:
    cfg = normalize_tokenizer_config(cfg)
    model = instantiate(cfg.tokenizer)
    init_batch = {
        "video": np.zeros(
            (1, 1, int(model.image_height), int(model.image_width), 3),
            dtype=np.uint8,
        )
    }
    init_key, sample_key = jax.random.split(jax.random.key(seed))
    state = TokenizerTrainState.create(
        model.apply,
        model.init({"params": init_key, "sample": sample_key}, init_batch, method=Tokenizer.reconstruct),
        optax.adam(0.0),
    )
    return TokenizerCheckpointBundle(
        config=cfg,
        model=model,
        state=state,
    )


def restore_tokenizer_checkpoint_bundle(
    checkpoint_dir: str | PathLike[str],
    seed: int,
    checkpoint_step: int | None = None,
    config: DictConfig | None = None,
) -> TokenizerCheckpointBundle:
    with CheckpointManager(checkpoint_dir, ocp.CheckpointManagerOptions()) as manager:
        if config is None:
            metadata = manager.load_metadata()
            if TOKENIZER_CONFIG_METADATA_KEY not in metadata:
                raise KeyError(
                    f"Checkpoint metadata in {checkpoint_dir} is missing {TOKENIZER_CONFIG_METADATA_KEY!r}."
                )
            config = OmegaConf.create(metadata[TOKENIZER_CONFIG_METADATA_KEY])

        bundle = build_tokenizer_checkpoint_bundle(config, seed=seed)
        return TokenizerCheckpointBundle(
            config=bundle.config,
            model=bundle.model,
            state=manager.restore(target=bundle.state, step=checkpoint_step),
        )
