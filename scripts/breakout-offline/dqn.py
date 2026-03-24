import functools
import logging

import flax.linen as nn
import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from visionary.common.buffers import (
    ReplayBuffer,
    get_action_dim,
    get_obs_shape,
)
from visionary.common.checkpoint import save_checkpoint
from visionary.common.env import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    make_vec_env,
)
from visionary.common.rollout import record_rollout
from visionary.common.train_state import TargetTrainState
from visionary.common.wandb import WandbLogger

logger = logging.getLogger(__name__)


class DQN(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32) / 255.0
        x = rearrange(x, "b s h w c -> b h w (s c)")
        x = nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = rearrange(x, "b h w c -> b (h w c)")
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_size)(x)

        return x


@functools.partial(jax.jit, static_argnums=(4,))
def select_action(
    state: TargetTrainState,
    obs: jnp.ndarray,
    key: jax.random.PRNGKey,
    epsilon: jnp.ndarray,
    action_size: int,
) -> jnp.ndarray:
    epsilon_key, action_key = jax.random.split(key)
    q_values = state.apply_fn(state.params, obs)
    greedy_action = jnp.argmax(q_values, axis=-1)
    random_action = jax.random.randint(action_key, greedy_action.shape, 0, action_size)
    explore = jax.random.uniform(epsilon_key, shape=greedy_action.shape) < epsilon
    return jnp.where(explore, random_action, greedy_action)


@functools.partial(jax.jit, static_argnums=(1,))
def train_step(
    state: TargetTrainState,
    gamma: float,
    batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tuple[TargetTrainState, jnp.ndarray]:

    def compute_loss(
        state: TargetTrainState,
        params: dict,
        gamma: float,
        batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        obs, next_obs, actions, rewards, dones = batch
        q_values = state.apply_fn(params, obs)
        q_values_next = state.apply_fn(state.target_params, next_obs)

        q_values = jnp.take_along_axis(
            q_values, actions.astype(jnp.int32), axis=1
        ).squeeze(-1)
        q_values_next = jnp.max(q_values_next, axis=1)
        q_values_target = jax.lax.stop_gradient(
            rewards + (1 - dones) * gamma * q_values_next
        )
        loss = jnp.mean((q_values_target - q_values) ** 2)

        return loss, (jnp.mean(q_values), jnp.mean(q_values_next))

    (loss, (mean_q, mean_q_next)), grads = jax.value_and_grad(
        compute_loss, argnums=1, has_aux=True
    )(state, state.params, gamma, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss, mean_q, mean_q_next


@jax.jit
def update_target(state: TargetTrainState, tau: float) -> TargetTrainState:
    target_params = optax.incremental_update(state.params, state.target_params, tau)
    return state.replace(target_params=target_params)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


@hydra.main(config_path="config", config_name="dqn", version_base=None)
def main(cfg: DictConfig):
    key = jax.random.key(cfg.seed)

    def make_env(eval=False):
        env = gym.make(cfg.env, render_mode="rgb_array", frameskip=1)
        env = gym.wrappers.AtariPreprocessing(
            env,
            frame_skip=cfg.frame_skip,
            screen_size=cfg.screen_size,
            grayscale_obs=True,
            grayscale_newaxis=True,
            scale_obs=False,
        )
        if not eval:
            env = EpisodicLifeEnv(env)
            env = ClipRewardEnv(env)
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env, fire_on_life_loss=eval)
        return env

    env = make_vec_env(make_env, n_envs=cfg.n_envs)

    replay_buffer = ReplayBuffer(
        cfg.buffer_size,
        get_obs_shape(env.single_observation_space),
        env.single_observation_space.dtype,
        get_action_dim(env.single_action_space),
        env.single_action_space.dtype,
        jax.devices()[0],
        n_envs=cfg.n_envs,
    )

    wb = WandbLogger(cfg)

    obs, info = env.reset()
    model = DQN(action_size=env.single_action_space.n)
    init_key, key = jax.random.split(key)
    params = model.init(init_key, obs)

    optimizer = optax.adam(cfg.learning_rate)
    state = TargetTrainState.create(
        apply_fn=model.apply,
        params=params,
        target_params=params,
        tx=optimizer,
    )

    eval_env = make_env(eval=True)
    output_dir = HydraConfig.get().runtime.output_dir

    action_size = env.single_action_space.n
    exploration_duration = cfg.exploration_fraction * cfg.total_steps
    episode_rewards = np.zeros(cfg.n_envs)
    episode_lengths = np.zeros(cfg.n_envs, dtype=int)
    global_step = 0
    while global_step < cfg.total_steps:
        key, action_key = jax.random.split(key)
        epsilon = linear_schedule(
            cfg.start_epsilon, cfg.end_epsilon, exploration_duration, global_step
        )
        actions = select_action(state, obs, action_key, jnp.array(epsilon), action_size)
        next_obs, rewards, terminated, truncated, infos = env.step(
            jax.device_get(actions)
        )
        dones = np.logical_or(terminated, truncated)

        if np.any(truncated):
            real_next_obs = next_obs.copy()
            mask = np.where(truncated)[0]
            real_next_obs[mask] = np.stack([infos["final_obs"][i] for i in mask])
        else:
            real_next_obs = next_obs

        replay_buffer.add(
            obs,
            real_next_obs,
            np.asarray(actions).reshape(cfg.n_envs, 1),
            rewards,
            terminated,
        )

        episode_rewards += rewards
        episode_lengths += 1
        for i in range(cfg.n_envs):
            if dones[i]:
                wb.log(
                    {
                        "episode_reward": episode_rewards[i],
                        "episode_length": int(episode_lengths[i]),
                    },
                    step=global_step,
                )
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0

        obs = next_obs
        global_step += cfg.n_envs

        if cfg.eval_steps > 0 and global_step % cfg.eval_steps < cfg.n_envs:
            save_checkpoint(state.params, output_dir, global_step)
            steps, reward, video_path = record_rollout(
                eval_env, state, output_dir, global_step
            )
            wb.log({"eval/steps": steps, "eval/reward": reward}, step=global_step)
            wb.log_video("eval/rollout", video_path, step=global_step)

        if (
            global_step >= cfg.learning_starts
            and global_step % cfg.train_freq < cfg.n_envs
        ):
            n_updates = max(1, cfg.n_envs // cfg.train_freq)
            for _ in range(n_updates):
                batch = replay_buffer.sample(cfg.batch_size)
                state, loss, mean_q, mean_q_next = train_step(state, cfg.gamma, batch)
            if global_step % cfg.target_update_freq < cfg.n_envs:
                state = update_target(state, cfg.tau)
            if global_step % cfg.log_interval < cfg.n_envs:
                wb.log(
                    {
                        "train/loss": float(loss),
                        "train/epsilon": epsilon,
                        "train/q_values": float(mean_q),
                        "train/q_values_next": float(mean_q_next),
                    },
                    step=global_step,
                )

    wb.finish()
    eval_env.close()


if __name__ == "__main__":
    main()
