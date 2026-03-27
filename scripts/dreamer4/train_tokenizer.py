import itertools
import logging
from functools import lru_cache, partial

import grain.python as grain
import hydra
import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from flax.training.train_state import TrainState
from hydra.utils import instantiate
from jaxlpips import LPIPS
from omegaconf import DictConfig

from visionary.common.checkpoint import CheckpointManager
from visionary.common.wandb import WandbLogger
from visionary.dataset import (
    EpisodeDataSource,
    PreprocessAndPatchify,
    PreprocessedVideoDataset,
    RandomVideoCrop,
)

logger = logging.getLogger(__name__)


LPIPS_PRETRAINED_NETWORK = "alexnet"


@lru_cache(maxsize=1)
def get_lpips_loss_fn():
    return LPIPS(pretrained_network=LPIPS_PRETRAINED_NETWORK)


def compute_lpips_loss(
    original: jax.Array,
    reconstructed: jax.Array,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
) -> jax.Array:
    def unpatchify(images: jax.Array) -> jax.Array:
        return rearrange(
            images,
            "b t (h w) (p1 p2 c) -> b t (h p1) (w p2) c",
            p1=patch_size,
            p2=patch_size,
            h=height_tokens,
            w=width_tokens,
        )

    original_images = unpatchify(original)
    reconstructed_images = unpatchify(reconstructed)
    original_images, reconstructed_images = (
        original_images * 2.0 - 1.0,
        reconstructed_images * 2.0 - 1.0,
    )
    return jnp.mean(get_lpips_loss_fn()(original_images, reconstructed_images))


def compute_losses(
    params,
    state: TrainState,
    batch: PreprocessedVideoDataset,
    mask_key: jax.Array,
    lpips_weight: float,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
):
    reconstructed = state.apply_fn(params, batch, rngs={"mask": mask_key})
    reconstructed_f32 = reconstructed.astype(jnp.float32)
    original = batch["video"].astype(jnp.float32) / 255.0
    mse_loss = jnp.mean(jnp.square(reconstructed_f32 - original))
    if lpips_weight > 0:
        lpips_loss = compute_lpips_loss(
            original,
            reconstructed_f32,
            patch_size=patch_size,
            width_tokens=width_tokens,
            height_tokens=height_tokens,
        )
    else:
        lpips_loss = jnp.zeros((), dtype=mse_loss.dtype)
    loss = mse_loss + lpips_weight * lpips_loss
    return loss, {"loss": loss, "mse_loss": mse_loss, "lpips_loss": lpips_loss}


@partial(
    jax.jit,
    static_argnames=("lpips_weight", "patch_size", "width_tokens", "height_tokens"),
)
def train_step(
    state: TrainState,
    batch: PreprocessedVideoDataset,
    mask_key: jax.Array,
    lpips_weight: float,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
):
    def loss_fn(params):
        return compute_losses(
            params,
            state,
            batch,
            mask_key,
            lpips_weight,
            patch_size,
            width_tokens,
            height_tokens,
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


@partial(
    jax.jit,
    static_argnames=("lpips_weight", "patch_size", "width_tokens", "height_tokens"),
)
def eval_step(
    state: TrainState,
    batch: PreprocessedVideoDataset,
    mask_key: jax.Array,
    lpips_weight: float,
    patch_size: int,
    width_tokens: int,
    height_tokens: int,
):
    _, metrics = compute_losses(
        state.params,
        state,
        batch,
        mask_key,
        lpips_weight,
        patch_size,
        width_tokens,
        height_tokens,
    )
    return metrics


@hydra.main(config_path="config", config_name="train_tokenizer", version_base=None)
def main(cfg: DictConfig):
    wb = WandbLogger(cfg)

    train_source, eval_source = EpisodeDataSource.from_split(
        cfg.dataset.data_dir, cfg.dataset.eval_ratio, cfg.dataset.split_seed
    )
    logger.info(
        "Loaded %d training videos and %d eval videos",
        len(train_source),
        len(eval_source),
    )

    transforms = [
        RandomVideoCrop(cfg.dataset.frame_length),
        PreprocessAndPatchify(cfg.dataset.patch_size, cfg.dataset.pad_width),
    ]

    def make_loader(source, *, shuffle: bool, drop_remainder: bool):
        sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.ShardByJaxProcess(),
            shuffle=shuffle,
            seed=cfg.seed,
        )
        return grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=[
                *transforms,
                grain.Batch(
                    batch_size=cfg.dataset.batch_size,
                    drop_remainder=drop_remainder,
                ),
            ],
            worker_count=cfg.dataset.worker_count,
        )

    train_dataloader = make_loader(train_source, shuffle=True, drop_remainder=True)
    eval_dataloader = make_loader(eval_source, shuffle=False, drop_remainder=False)
    sample_batch = next(iter(train_dataloader))

    key = jax.random.key(cfg.seed)
    init_key, init_mask_key, train_key, eval_key = jax.random.split(key, num=4)
    model = instantiate(cfg.tokenizer)
    params = model.init({"params": init_key, "mask": init_mask_key}, sample_batch)

    optimizer = optax.adam(cfg.learning_rate)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    checkpoint_manager: CheckpointManager = instantiate(cfg.checkpoint.manager)
    if cfg.checkpoint.resume_step is not None:
        state = checkpoint_manager.restore(
            target=state,
            step=int(cfg.checkpoint.resume_step),
        )
        logger.info(f"Resumed tokenizer training from step {state.step}")

    step = int(state.step)

    for batch in train_dataloader:
        mask_key = jax.random.fold_in(train_key, step)
        state, metrics = train_step(
            state,
            batch,
            mask_key,
            cfg.lpips_weight,
            patch_size=cfg.dataset.patch_size,
            width_tokens=cfg.tokenizer.x_len,
            height_tokens=cfg.tokenizer.y_len,
        )
        step = int(state.step)

        if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
            totals: dict[str, float] = {}
            num_batches = 0
            for batch_idx, eval_batch in enumerate(
                itertools.islice(iter(eval_dataloader), cfg.dataset.eval.max_batches)
            ):
                eval_mask_key = jax.random.fold_in(eval_key, batch_idx)
                batch_metrics = eval_step(
                    state,
                    eval_batch,
                    eval_mask_key,
                    cfg.lpips_weight,
                    patch_size=cfg.dataset.patch_size,
                    width_tokens=cfg.tokenizer.x_len,
                    height_tokens=cfg.tokenizer.y_len,
                )
                for k, v in batch_metrics.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                num_batches += 1
            if num_batches > 0:
                eval_metrics = {k: v / num_batches for k, v in totals.items()}
                wb.log(
                    {f"eval/{k}": v for k, v in eval_metrics.items()},
                    step=step,
                )

        if step % cfg.log_interval == 0:
            wb.log(
                {k: float(v) for k, v in metrics.items()},
                step=step,
            )

        if checkpoint_manager.should_save(step):
            checkpoint_manager.save(step=step, state=state)
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    wb.finish()


if __name__ == "__main__":
    main()
