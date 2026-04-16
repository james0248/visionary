import argparse
import math

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from visionary.dynamics import DynamicsModel


def build_synthetic_batch(
    key: jax.Array,
    *,
    batch_size: int,
    sequence_length: int,
    num_actions: int,
    mode: str,
) -> dict[str, jnp.ndarray]:
    key_state, key_action = jax.random.split(key)
    if mode == "stable":
        state_dim = 2
        initial_state = jax.random.normal(key_state, (batch_size, state_dim), dtype=jnp.float32) * 0.2
        action_effect = jnp.array(
            [
                [0.05, -0.02],
                [-0.04, 0.03],
                [0.02, 0.06],
            ],
            dtype=jnp.float32,
        )
        transition = jnp.array([[0.85, 0.10], [-0.05, 0.90]], dtype=jnp.float32)
    elif mode == "integrator":
        state_dim = 4
        initial_state = jax.random.normal(key_state, (batch_size, state_dim), dtype=jnp.float32) * 0.05
        action_effect = jnp.array(
            [
                [0.15, 0.0, 0.0, 0.0],
                [-0.12, 0.0, 0.0, 0.0],
                [0.0, 0.18, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        transition = jnp.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.97, 0.0],
                [0.0, 0.0, 0.0, 0.96],
            ],
            dtype=jnp.float32,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    actions = jax.random.randint(
        key_action,
        (batch_size, sequence_length),
        0,
        num_actions,
        dtype=jnp.int32,
    )
    states = [initial_state]
    for step_idx in range(sequence_length - 1):
        states.append(states[-1] @ transition.T + action_effect[actions[:, step_idx]])
    states = jnp.stack(states, axis=1)

    aligned_actions = -jnp.ones((batch_size, sequence_length), dtype=jnp.int32)
    aligned_actions = aligned_actions.at[:, 1:].set(actions[:, :-1])
    return {
        "video": states[:, :, None, :],
        "actions": aligned_actions,
    }


def rollout_mse(
    model: DynamicsModel,
    params,
    batch: dict[str, jnp.ndarray],
    *,
    context_frames: int,
    context_tau: float,
    sample_steps: int,
) -> float:
    video = jnp.asarray(batch["video"], dtype=jnp.float32)
    actions = jnp.asarray(batch["actions"], dtype=jnp.int32)
    rollout_video = jnp.zeros_like(video)
    rollout_video = rollout_video.at[:, :context_frames].set(video[:, :context_frames])

    rollout_key = jax.random.key(123)
    context_noise_key, sample_noise_key = jax.random.split(rollout_key)
    context_noise = jax.random.normal(context_noise_key, video.shape, dtype=jnp.float32)
    sample_noise = jax.random.normal(sample_noise_key, video.shape, dtype=jnp.float32)

    for frame_idx in range(context_frames, video.shape[1]):
        next_representation = model.apply(
            params,
            rollout_video,
            actions,
            context_noise,
            sample_noise[:, frame_idx],
            jnp.asarray(frame_idx, dtype=jnp.int32),
            context_tau=context_tau,
            sample_steps=sample_steps,
            method=DynamicsModel.generate_next,
        ).astype(jnp.float32)
        rollout_video = rollout_video.at[:, frame_idx].set(next_representation)

    return float(jnp.mean((rollout_video[:, context_frames:] - video[:, context_frames:]) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic smoke test for the dynamics loss.")
    parser.add_argument("--mode", choices=("stable", "integrator"), default="integrator")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--context_frames", type=int, default=4)
    parser.add_argument("--num_actions", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--bootstrap_ratio", type=float, default=0.25)
    parser.add_argument("--bootstrap_start_step", type=int, default=100)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--context_tau", type=float, default=0.9)
    args = parser.parse_args()

    key = jax.random.key(0)
    train_batch = build_synthetic_batch(
        jax.random.fold_in(key, 1),
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_actions=args.num_actions,
        mode=args.mode,
    )
    eval_batch = build_synthetic_batch(
        jax.random.fold_in(key, 2),
        batch_size=args.eval_batch_size,
        sequence_length=args.sequence_length,
        num_actions=args.num_actions,
        mode=args.mode,
    )

    latent_dim = int(train_batch["video"].shape[-1])
    model = DynamicsModel(
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        num_registers=1,
        num_obs_tokens=1,
        num_actions=args.num_actions,
        max_step_size=6,
        model_dim=64,
        head_dim=32,
        mlp_hidden_dim=128,
        context_length=args.sequence_length,
        dtype=jnp.float32,
    )
    params = model.init(
        {"params": jax.random.fold_in(key, 3), "sample": jax.random.fold_in(key, 4)},
        train_batch,
        bootstrap_ratio=0.0,
        method=DynamicsModel.loss,
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(args.learning_rate))

    @jax.jit
    def train_step(
        state: TrainState,
        batch: dict[str, jnp.ndarray],
        step: jax.Array,
        bootstrap_ratio: float,
    ):
        def loss_fn(params):
            return model.apply(
                params,
                batch,
                bootstrap_ratio=bootstrap_ratio,
                method=DynamicsModel.loss,
                rngs={"sample": jax.random.fold_in(jax.random.key(999), step)},
            )

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), metrics

    @jax.jit
    def teacher_forced_mse(params, batch):
        video = jnp.asarray(batch["video"], dtype=jnp.float32)
        actions = jnp.asarray(batch["actions"], dtype=jnp.int32)
        step_level = 5
        signal_level = (1 << step_level) - 1
        prediction = model.apply(
            params,
            video,
            actions,
            jnp.full(actions.shape, step_level, dtype=jnp.int32),
            jnp.full(actions.shape, signal_level, dtype=jnp.int32),
        )
        return jnp.mean((prediction - video) ** 2)

    print(
        f"synthetic mode={args.mode} state_dim={latent_dim} "
        f"bootstrap_ratio={args.bootstrap_ratio:.2f} "
        f"bootstrap_start_step={args.bootstrap_start_step}"
    )
    for step in range(args.steps + 1):
        bootstrap_ratio = (
            args.bootstrap_ratio
            if step >= args.bootstrap_start_step
            else 0.0
        )
        state, metrics = train_step(
            state,
            train_batch,
            jnp.asarray(step, dtype=jnp.int32),
            bootstrap_ratio,
        )
        if step % max(args.steps // 4, 1) == 0 or step == args.steps:
            print(
                {
                    "step": step,
                    "loss": round(float(metrics["loss"]), 6),
                    "active_flow_loss": round(float(metrics["active_flow_loss"]), 6),
                    "active_bootstrap_loss": round(float(metrics["active_bootstrap_loss"]), 6),
                    "bootstrap_active_fraction": round(
                        float(metrics["bootstrap_active_fraction"]),
                        4,
                    ),
                    "teacher_forced_mse": round(float(teacher_forced_mse(state.params, eval_batch)), 6),
                    "rollout_mse": round(
                        rollout_mse(
                            model,
                            state.params,
                            eval_batch,
                            context_frames=args.context_frames,
                            context_tau=args.context_tau,
                            sample_steps=args.sample_steps,
                        ),
                        6,
                    ),
                }
            )


if __name__ == "__main__":
    main()
