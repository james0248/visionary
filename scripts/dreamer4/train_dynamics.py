import itertools
import logging
import time

import grain.python as grain
import hydra
import jax
import optax
from hydra.utils import instantiate
from omegaconf import DictConfig

from visionary.common.checkpoint import CheckpointManager
from visionary.common.train_state import DynamicsTrainState
from visionary.common.wandb import WandbLogger
from visionary.dataset import DynamicsBatch, DynamicsDataSource, RandomDynamicsCrop
from visionary.dynamics import DynamicsModel

logger = logging.getLogger(__name__)


@jax.jit
def train_step(
    state: DynamicsTrainState,
    batch: DynamicsBatch,
    sample_key: jax.Array,
):
    def loss_fn(params):
        return state.apply_fn(
            params,
            batch,
            method=DynamicsModel.loss,
            rngs={"sample": sample_key},
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


@jax.jit
def eval_step(
    state: DynamicsTrainState,
    batch: DynamicsBatch,
    sample_key: jax.Array,
):
    _, metrics = state.apply_fn(
        state.params,
        batch,
        method=DynamicsModel.loss,
        rngs={"sample": sample_key},
    )
    return metrics


@hydra.main(config_path="config", config_name="breakout_dynamics", version_base=None)
def main(cfg: DictConfig):
    logger.info("JAX backend: %s, devices: %s", jax.default_backend(), jax.devices())
    wb = WandbLogger(cfg)
    total_steps = cfg.total_steps

    train_source = DynamicsDataSource(cfg.dataset.train_dir)
    eval_source = DynamicsDataSource(cfg.dataset.eval_dir)
    logger.info(
        "Loaded %d training sequences and %d eval sequences",
        len(train_source),
        len(eval_source),
    )
    effective_read_threads = max(cfg.dataset.worker_count, 1) * cfg.dataset.num_threads
    logger.info(
        "Data loader settings: worker_count=%d num_threads=%d "
        "prefetch_buffer_size=%d effective_read_threads=%d",
        cfg.dataset.worker_count,
        cfg.dataset.num_threads,
        cfg.dataset.prefetch_buffer_size,
        effective_read_threads,
    )

    train_sequence_lengths = (
        sorted({cfg.dataset.alternating_lengths.short, cfg.dataset.alternating_lengths.long})
        if cfg.dataset.alternating_lengths.enabled
        else [cfg.dataset.batch_length]
    )
    logger.info(
        "Training sequence lengths: %s; eval sequence length: %d",
        train_sequence_lengths,
        cfg.dataset.eval.batch_length,
    )

    def make_loader(
        source: DynamicsDataSource,
        *,
        sequence_length: int,
        shuffle: bool,
        drop_remainder: bool,
        seed: int,
    ) -> grain.DataLoader:
        sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.ShardByJaxProcess(),
            shuffle=shuffle,
            seed=seed,
        )
        read_options = grain.ReadOptions(
            num_threads=cfg.dataset.num_threads,
            prefetch_buffer_size=cfg.dataset.prefetch_buffer_size,
        )
        return grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=[
                RandomDynamicsCrop(sequence_length),
                grain.Batch(
                    batch_size=cfg.dataset.batch_size,
                    drop_remainder=drop_remainder,
                ),
            ],
            worker_count=cfg.dataset.worker_count,
            read_options=read_options,
        )

    _t = time.monotonic()
    train_loaders = {
        sequence_length: make_loader(
            train_source,
            sequence_length=sequence_length,
            shuffle=True,
            drop_remainder=True,
            seed=cfg.seed + sequence_length,
        )
        for sequence_length in train_sequence_lengths
    }
    logger.info("Train DataLoader creation took %.1fs", time.monotonic() - _t)

    init_sequence_length = max(train_sequence_lengths)
    _t = time.monotonic()
    sample_batch = next(iter(train_loaders[init_sequence_length]))
    logger.info("First batch fetch took %.1fs", time.monotonic() - _t)

    overfit_batches = {init_sequence_length: sample_batch}
    if cfg.overfit_single_batch:
        logger.info("Overfit mode enabled: reusing one sampled batch per train sequence length.")
        for sequence_length in train_sequence_lengths:
            if sequence_length in overfit_batches:
                continue
            overfit_batches[sequence_length] = next(iter(train_loaders[sequence_length]))
        if cfg.dataset.eval.batch_length not in overfit_batches:
            overfit_batches[cfg.dataset.eval.batch_length] = next(
                iter(
                    make_loader(
                        eval_source,
                        sequence_length=cfg.dataset.eval.batch_length,
                        shuffle=False,
                        drop_remainder=False,
                        seed=cfg.seed,
                    )
                )
            )
    else:
        _t = time.monotonic()
        eval_loader = make_loader(
            eval_source,
            sequence_length=cfg.dataset.eval.batch_length,
            shuffle=False,
            drop_remainder=False,
            seed=cfg.seed,
        )
        logger.info("Eval DataLoader creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    key = jax.random.key(cfg.seed)
    init_key, init_sample_key, train_key, eval_key = jax.random.split(key, num=4)
    model = instantiate(cfg.dynamics)
    params = model.init(
        {"params": init_key, "sample": init_sample_key},
        sample_batch,
        method=DynamicsModel.loss,
    )
    logger.info("Model init took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    optimizer = optax.adam(cfg.learning_rate)
    state = DynamicsTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
    logger.info("TrainState creation took %.1fs", time.monotonic() - _t)

    _t = time.monotonic()
    checkpoint_manager: CheckpointManager = instantiate(cfg.checkpoint.manager)
    logger.info("CheckpointManager creation took %.1fs", time.monotonic() - _t)
    if cfg.checkpoint.resume_step is not None:
        state = checkpoint_manager.restore(
            target=state,
            step=cfg.checkpoint.resume_step,
        )
        logger.info("Resumed dynamics training from step %d", int(state.step))

    step = int(state.step)
    logger.info("Dynamics training target step: %d", total_steps)

    if step >= total_steps:
        logger.info(
            "Current step %d is already at or above total_steps=%d; exiting.",
            step,
            total_steps,
        )
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()
        wb.finish()
        return

    train_iterators = {
        sequence_length: iter(loader) for sequence_length, loader in train_loaders.items()
    }

    while True:
        t1 = time.monotonic()

        if cfg.dataset.alternating_lengths.enabled and (
            step >= total_steps - cfg.dataset.alternating_lengths.final_long_only_steps
            or (step + 1) % cfg.dataset.alternating_lengths.long_every == 0
        ):
            sequence_length = cfg.dataset.alternating_lengths.long
        elif cfg.dataset.alternating_lengths.enabled:
            sequence_length = cfg.dataset.alternating_lengths.short
        else:
            sequence_length = cfg.dataset.batch_length

        if cfg.overfit_single_batch:
            batch = overfit_batches[sequence_length]
        else:
            try:
                batch = next(train_iterators[sequence_length])
            except StopIteration:
                train_iterators[sequence_length] = iter(train_loaders[sequence_length])
                batch = next(train_iterators[sequence_length])
        t2 = time.monotonic()

        batch = jax.device_put(batch)
        t3 = time.monotonic()

        sample_key = jax.random.fold_in(train_key, step)
        state, metrics = train_step(state, batch, sample_key)
        jax.block_until_ready(metrics)
        t4 = time.monotonic()

        step = int(state.step)

        t_eval = 0.0
        if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
            t_eval_start = time.monotonic()
            totals: dict[str, float] = {}
            if cfg.overfit_single_batch:
                eval_batches = (overfit_batches[cfg.dataset.eval.batch_length],)
            else:
                eval_batches = itertools.islice(iter(eval_loader), cfg.dataset.eval.max_batches)

            num_batches = 0
            for batch_idx, eval_batch in enumerate(eval_batches):
                eval_sample_key = jax.random.fold_in(eval_key, batch_idx)
                batch_metrics = jax.device_get(
                    eval_step(state, jax.device_put(eval_batch), eval_sample_key)
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
            t_eval = time.monotonic() - t_eval_start
            logger.info("Eval at step %d - %d batches in %.3fs", step, num_batches, t_eval)

        if step % cfg.log_interval == 0:
            wb.log(
                {
                    **{k: float(v) for k, v in metrics.items()},
                    "train/sequence_length": sequence_length,
                },
                step=step,
            )

        if checkpoint_manager.should_save(step):
            checkpoint_manager.save(step=step, state=state)

        t5 = time.monotonic()
        logger.info(
            "Step %d - seq: %d, data: %.3fs, transfer: %.3fs, compute: %.3fs, overhead: %.3fs",
            step,
            sequence_length,
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t5 - t4 - t_eval,
        )
        if step >= total_steps:
            logger.info("Reached total_steps=%d; stopping dynamics training.", total_steps)
            break

    if step >= total_steps and not checkpoint_manager.should_save(step):
        checkpoint_manager.save(step=step, state=state, force=True)
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    wb.finish()


if __name__ == "__main__":
    main()
