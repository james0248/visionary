import functools
import itertools
import logging

import grain.python as grain
import hydra
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from hydra.utils import instantiate
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


def load_dataset(cfg: DictConfig, seed: int):
    source = EpisodeDataSource(cfg.data_dir)
    sampler = grain.IndexSampler(
        num_records=len(source),
        shard_options=grain.ShardByJaxProcess(),
        shuffle=True,
        seed=seed,
    )
    operations = [
        RandomVideoCrop(cfg.frame_length),
        PreprocessAndPatchify(cfg.frame_length, cfg.patch_size, cfg.pad_width),
        grain.Batch(batch_size=cfg.batch_size, drop_remainder=True),
    ]
    dataloader = grain.DataLoader(
        source=source, sampler=sampler, operations=operations, worker_count=8
    )
    return dataloader


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(
    state: TrainState,
    batch: PreprocessedVideoDataset,
    lpips_weight: float,
    mask_key: jax.Array,
):
    def compute_loss(state: TrainState, params: dict, batch: PreprocessedVideoDataset):
        reconstructed = state.apply_fn(params, batch, rngs={"mask": mask_key})
        original = batch["video"]
        mse_loss = jnp.mean((reconstructed - original) ** 2)

        # TODO: Add LPIPS loss
        lpips_loss = lpips_weight * 0.0

        loss = mse_loss + lpips_loss
        return loss, (mse_loss, lpips_loss)

    (loss, (mse_loss, lpips_loss)), grads = jax.value_and_grad(
        compute_loss, argnums=1, has_aux=True
    )(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss, (mse_loss, lpips_loss)


@hydra.main(config_path="config", config_name="train_tokenizer", version_base=None)
def main(cfg: DictConfig):
    wb = WandbLogger(cfg)

    dataloader = load_dataset(cfg.dataset, cfg.seed)
    sample_batch = next(iter(dataloader))

    key = jax.random.key(cfg.seed)
    init_key, mask_key, train_key = jax.random.split(key, num=3)
    model = instantiate(cfg.tokenizer)
    params = model.init({"params": init_key, "mask": mask_key}, sample_batch)

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

    for batch in dataloader:
        mask_key = jax.random.fold_in(train_key, step)
        state, loss, (mse_loss, lpips_loss) = train_step(
            state,
            batch,
            cfg.lpips_weight,
            mask_key,
        )
        step = int(state.step)

        if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
            eval_loss = 0.0
            wb.log({"eval/steps": step, "eval/loss": eval_loss}, step=step)

        if step % cfg.log_interval == 0:
            wb.log(
                {
                    "loss": float(loss),
                    "mse_loss": float(mse_loss),
                    "lpips_loss": float(lpips_loss),
                },
                step=step,
            )

        if checkpoint_manager.should_save(step):
            checkpoint_manager.save(step=step, state=state)
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    wb.finish()


if __name__ == "__main__":
    main()
