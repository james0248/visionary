import logging
import os
import time
from collections.abc import Callable

import flax.linen as nn
import gymnasium as gym
import hydra
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from flax.training.train_state import TrainState
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from visionary.common.checkpoint import CheckpointManager
from visionary.common.env import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    has_fire_action,
    make_vec_env,
)
from visionary.common.wandb import WandbLogger

logger = logging.getLogger(__name__)


def _scaled_init(scale: float = 1.0):
    # Avoid orthogonal initialization here: JAX implements it through QR, which
    # can require a cuSolver FFI path that is unavailable in some CUDA builds.
    return nn.initializers.variance_scaling(scale**2, "fan_avg", "truncated_normal")


class ActorCritic(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = x.astype(jnp.float32) / 255.0
        x = rearrange(x, "b s h w c -> b h w (s c)")
        x = nn.Conv(
            32,
            (8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=_scaled_init(np.sqrt(2)),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            (4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=_scaled_init(np.sqrt(2)),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            (3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=_scaled_init(np.sqrt(2)),
        )(x)
        x = nn.relu(x)
        x = rearrange(x, "b h w c -> b (h w c)")
        x = nn.Dense(256, kernel_init=_scaled_init(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(448, kernel_init=_scaled_init(np.sqrt(2)))(x)
        x = nn.relu(x)

        value_features = x + nn.relu(
            nn.Dense(448, kernel_init=_scaled_init(np.sqrt(0.1)), name="extra_value_fc")(x)
        )
        policy_features = x + nn.relu(
            nn.Dense(448, kernel_init=_scaled_init(np.sqrt(0.1)), name="extra_policy_fc")(x)
        )

        logits = nn.Dense(
            self.action_size,
            kernel_init=_scaled_init(np.sqrt(0.01)),
            name="policy",
        )(policy_features)
        int_value = nn.Dense(1, kernel_init=_scaled_init(np.sqrt(0.01)), name="int_value")(
            value_features
        )
        ext_value = nn.Dense(1, kernel_init=_scaled_init(np.sqrt(0.01)), name="ext_value")(
            value_features
        )
        return logits, int_value.squeeze(-1), ext_value.squeeze(-1)


class RNDTarget(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = rearrange(x, "b s h w c -> b h w (s c)")
        x = nn.Conv(
            32,
            (8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=_scaled_init(np.sqrt(2)),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(
            64,
            (4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=_scaled_init(np.sqrt(2)),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(
            64,
            (3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=_scaled_init(np.sqrt(2)),
        )(x)
        x = nn.leaky_relu(x)
        x = rearrange(x, "b h w c -> b (h w c)")
        return nn.Dense(512, kernel_init=_scaled_init(np.sqrt(2)))(x)


class RNDPredictor(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = rearrange(x, "b s h w c -> b h w (s c)")
        x = nn.Conv(
            32,
            (8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=_scaled_init(np.sqrt(2)),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(
            64,
            (4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=_scaled_init(np.sqrt(2)),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(
            64,
            (3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=_scaled_init(np.sqrt(2)),
        )(x)
        x = nn.leaky_relu(x)
        x = rearrange(x, "b h w c -> b (h w c)")
        x = nn.Dense(512, kernel_init=_scaled_init(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=_scaled_init(np.sqrt(2)))(x)
        x = nn.relu(x)
        return nn.Dense(512, kernel_init=_scaled_init(np.sqrt(2)))(x)


class PPORNDTrainState(TrainState):
    running_mean: jax.Array
    running_var: jax.Array
    running_count: jax.Array
    reward_rms_mean: jax.Array
    reward_rms_var: jax.Array
    reward_rms_count: jax.Array
    global_step: jax.Array
    rollout_idx: jax.Array


def categorical_log_prob(logits: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)


def categorical_entropy(logits: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


def normalize_obs(
    obs: jnp.ndarray,
    running_mean: jnp.ndarray,
    running_var: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    obs = obs.astype(jnp.float32)
    return jnp.clip((obs - running_mean) / jnp.sqrt(running_var + eps), -5.0, 5.0)


def update_running_stats(
    mean: jnp.ndarray,
    var: jnp.ndarray,
    count: jnp.ndarray,
    batch: jnp.ndarray,
    batch_axes: tuple[int, ...],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch = batch.astype(jnp.float32)
    batch_mean = jnp.mean(batch, axis=batch_axes)
    batch_var = jnp.var(batch, axis=batch_axes)
    batch_count = jnp.asarray(np.prod([batch.shape[axis] for axis in batch_axes]), dtype=jnp.float32)

    delta = batch_mean - mean
    total_count = count + batch_count
    new_mean = mean + delta * batch_count / total_count
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta**2 * count * batch_count / total_count
    new_var = m2 / total_count
    return new_mean, new_var, total_count


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_value: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[t].astype(np.float32)
        next_values = next_value if t == rewards.shape[0] - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def make_update_fn(
    policy_model: ActorCritic,
    target_model: RNDTarget,
    predictor_model: RNDPredictor,
    cfg: DictConfig,
) -> Callable:
    clip_range = float(cfg.clip_range)
    ent_coeff = float(cfg.ent_coeff)
    value_coeff = float(cfg.value_coeff)
    rnd_coeff = float(cfg.rnd_coeff)
    predictor_proportion = float(cfg.predictor_proportion)

    @jax.jit
    def update_minibatch(
        state: PPORNDTrainState,
        batch: dict[str, jnp.ndarray],
        key: jax.Array,
    ) -> tuple[PPORNDTrainState, dict[str, jnp.ndarray]]:
        def loss_fn(params):
            logits, int_values, ext_values = policy_model.apply(params["policy"], batch["obs"])
            log_prob = categorical_log_prob(logits, batch["actions"])
            entropy = jnp.mean(categorical_entropy(logits))
            log_ratio = log_prob - batch["old_log_probs"]
            ratio = jnp.exp(log_ratio)
            unclipped = ratio * batch["advantages"]
            clipped = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * batch["advantages"]
            policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

            value_ext_loss = jnp.mean((ext_values - batch["returns_ext"]) ** 2)
            value_int_loss = jnp.mean((int_values - batch["returns_int"]) ** 2)

            target_features = jax.lax.stop_gradient(
                target_model.apply(params["rnd_target"], batch["rnd_obs"])
            )
            predictor_features = predictor_model.apply(params["rnd_predictor"], batch["rnd_obs"])
            rnd_error = jnp.mean((predictor_features - target_features) ** 2, axis=-1)
            mask = jax.random.uniform(key, rnd_error.shape) < predictor_proportion
            mask = mask.astype(jnp.float32)
            rnd_loss = jnp.sum(mask * rnd_error) / jnp.maximum(jnp.sum(mask), 1.0)

            total_loss = (
                policy_loss
                + value_coeff * (value_ext_loss + value_int_loss)
                - ent_coeff * entropy
                + rnd_coeff * rnd_loss
            )
            approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
            clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32))
            metrics = {
                "loss/policy": policy_loss,
                "loss/value_ext": value_ext_loss,
                "loss/value_int": value_int_loss,
                "loss/rnd": rnd_loss,
                "loss/entropy": entropy,
                "loss/total": total_loss,
                "stats/approx_kl": approx_kl,
                "stats/clip_fraction": clip_fraction,
                "stats/rnd_mask_fraction": jnp.mean(mask),
            }
            return total_loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        del loss
        state = state.apply_gradients(grads=grads)
        return state, metrics

    return update_minibatch


def make_policy_fns(
    policy_model: ActorCritic,
    target_model: RNDTarget,
    predictor_model: RNDPredictor,
):
    @jax.jit
    def select_action(
        params: dict,
        obs: jnp.ndarray,
        key: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        logits, int_values, ext_values = policy_model.apply(params["policy"], obs)
        actions = jax.random.categorical(key, logits, axis=-1)
        log_probs = categorical_log_prob(logits, actions)
        return actions, log_probs, int_values, ext_values

    @jax.jit
    def values(params: dict, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        _, int_values, ext_values = policy_model.apply(params["policy"], obs)
        return int_values, ext_values

    @jax.jit
    def intrinsic_rewards(
        params: dict,
        obs: jnp.ndarray,
        running_mean: jnp.ndarray,
        running_var: jnp.ndarray,
        reward_rms_var: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        rnd_obs = normalize_obs(obs, running_mean, running_var)
        target_features = jax.lax.stop_gradient(target_model.apply(params["rnd_target"], rnd_obs))
        predictor_features = predictor_model.apply(params["rnd_predictor"], rnd_obs)
        raw = jnp.mean((predictor_features - target_features) ** 2, axis=-1)
        normalized = raw / jnp.sqrt(reward_rms_var + 1e-8)
        return raw, normalized

    return select_action, values, intrinsic_rewards


def flatten_rollout(array: np.ndarray | jnp.ndarray) -> jnp.ndarray:
    array = jnp.asarray(array)
    return array.reshape((array.shape[0] * array.shape[1],) + array.shape[2:])


def minibatches(
    rng: np.random.Generator,
    batch: dict[str, jnp.ndarray],
    n_mini_batch: int,
) -> list[dict[str, jnp.ndarray]]:
    batch_size = next(iter(batch.values())).shape[0]
    indices = rng.permutation(batch_size)
    splits = np.array_split(indices, n_mini_batch)
    return [{key: value[idx] for key, value in batch.items()} for idx in splits]


def record_eval_rollout(
    env: gym.Env,
    policy_model: ActorCritic,
    params: dict,
    output_dir: str,
    global_step: int,
) -> tuple[int, float, str]:
    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"ppo_rnd_rollout_{global_step}.mp4")

    @jax.jit
    def get_action(policy_params, obs):
        logits, _, _ = policy_model.apply(policy_params, obs[None])
        return jnp.argmax(logits, axis=-1)

    obs, _ = env.reset()
    fps = env.metadata.get("render_fps", 30)
    frames = [env.render()]
    total_reward = 0.0
    steps = 0

    while True:
        action = int(get_action(params["policy"], jnp.asarray(obs)).item())
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        frames.append(env.render())
        if terminated or truncated:
            break

    imageio.mimsave(video_path, frames, fps=fps, macro_block_size=1)
    return steps, total_reward, video_path


@hydra.main(config_path="config", config_name="ppo_rnd", version_base=None)
def main(cfg: DictConfig):
    key = jax.random.key(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    def make_env(eval: bool = False):
        env = gym.make(
            cfg.env,
            render_mode="rgb_array",
            frameskip=1,
            max_episode_steps=cfg.max_frames_episode,
        )
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
        if cfg.fire_reset and has_fire_action(env.unwrapped.get_action_meanings()):
            env = FireResetEnv(env, fire_on_life_loss=eval)
        return env

    env = make_vec_env(make_env, n_envs=cfg.n_envs)
    eval_env = make_env(eval=True)
    obs, _ = env.reset()

    action_size = env.single_action_space.n
    obs_shape = env.single_observation_space.shape
    policy_model = ActorCritic(action_size=action_size)
    target_model = RNDTarget()
    predictor_model = RNDPredictor()

    policy_key, target_key, predictor_key, key = jax.random.split(key, 4)
    dummy_obs = jnp.asarray(obs[:1], dtype=jnp.uint8)
    dummy_rnd_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)
    params = {
        "policy": policy_model.init(policy_key, dummy_obs),
        "rnd_target": target_model.init(target_key, dummy_rnd_obs),
        "rnd_predictor": predictor_model.init(predictor_key, dummy_rnd_obs),
    }

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.learning_rate),
    )
    state = PPORNDTrainState.create(
        apply_fn=policy_model.apply,
        params=params,
        tx=optimizer,
        running_mean=jnp.zeros(obs_shape, dtype=jnp.float32),
        running_var=jnp.ones(obs_shape, dtype=jnp.float32),
        running_count=jnp.asarray(1e-4, dtype=jnp.float32),
        reward_rms_mean=jnp.zeros((), dtype=jnp.float32),
        reward_rms_var=jnp.ones((), dtype=jnp.float32),
        reward_rms_count=jnp.asarray(1e-4, dtype=jnp.float32),
        global_step=jnp.zeros((), dtype=jnp.int32),
        rollout_idx=jnp.zeros((), dtype=jnp.int32),
    )

    select_action, values_fn, intrinsic_rewards_fn = make_policy_fns(
        policy_model,
        target_model,
        predictor_model,
    )
    update_minibatch = make_update_fn(policy_model, target_model, predictor_model, cfg)

    wb = WandbLogger(cfg)
    output_dir = HydraConfig.get().runtime.output_dir
    checkpoint_manager = CheckpointManager(
        directory=os.path.join(output_dir, "checkpoints"),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=1,
            enable_async_checkpointing=True,
        ),
    )

    episode_rewards = np.zeros(cfg.n_envs, dtype=np.float32)
    episode_lengths = np.zeros(cfg.n_envs, dtype=np.int32)
    global_step = 0
    last_checkpoint_step: int | None = None
    start_time = time.time()

    for _ in range(int(cfg.steps_initial_normalization)):
        actions = np_rng.integers(0, action_size, size=(cfg.n_envs,), dtype=np.int32)
        obs, _, _, _, _ = env.step(actions)
        running_mean, running_var, running_count = update_running_stats(
            state.running_mean,
            state.running_var,
            state.running_count,
            jnp.asarray(obs),
            batch_axes=(0,),
        )
        state = state.replace(
            running_mean=running_mean,
            running_var=running_var,
            running_count=running_count,
        )
        global_step += cfg.n_envs

    for rollout_idx in range(int(cfg.total_rollouts_per_env)):
        obs_buf = np.empty((cfg.rollout_length, cfg.n_envs) + obs_shape, dtype=np.uint8)
        actions_buf = np.empty((cfg.rollout_length, cfg.n_envs), dtype=np.int32)
        log_probs_buf = np.empty((cfg.rollout_length, cfg.n_envs), dtype=np.float32)
        ext_rewards_buf = np.empty((cfg.rollout_length, cfg.n_envs), dtype=np.float32)
        int_values_buf = np.empty((cfg.rollout_length, cfg.n_envs), dtype=np.float32)
        ext_values_buf = np.empty((cfg.rollout_length, cfg.n_envs), dtype=np.float32)
        dones_buf = np.empty((cfg.rollout_length, cfg.n_envs), dtype=bool)
        next_obs_buf = np.empty((cfg.rollout_length, cfg.n_envs) + obs_shape, dtype=np.uint8)

        for t in range(cfg.rollout_length):
            key, action_key = jax.random.split(key)
            actions, log_probs, int_values, ext_values = select_action(
                state.params,
                jnp.asarray(obs),
                action_key,
            )
            actions_np = np.asarray(jax.device_get(actions), dtype=np.int32)
            next_obs, rewards, terminated, truncated, infos = env.step(actions_np)
            dones = np.logical_or(terminated, truncated)

            real_next_obs = next_obs
            if np.any(truncated) and "final_obs" in infos:
                real_next_obs = next_obs.copy()
                mask = np.where(truncated)[0]
                real_next_obs[mask] = np.stack([infos["final_obs"][i] for i in mask])

            obs_buf[t] = obs
            actions_buf[t] = actions_np
            log_probs_buf[t] = np.asarray(jax.device_get(log_probs), dtype=np.float32)
            ext_rewards_buf[t] = rewards
            int_values_buf[t] = np.asarray(jax.device_get(int_values), dtype=np.float32)
            ext_values_buf[t] = np.asarray(jax.device_get(ext_values), dtype=np.float32)
            dones_buf[t] = terminated
            next_obs_buf[t] = real_next_obs

            episode_rewards += rewards
            episode_lengths += 1
            for i in range(cfg.n_envs):
                if dones[i]:
                    wb.log(
                        {
                            "episode_reward": float(episode_rewards[i]),
                            "episode_length": int(episode_lengths[i]),
                        },
                        step=global_step,
                    )
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0

            obs = next_obs
            global_step += cfg.n_envs

        flat_next_obs = next_obs_buf.reshape((-1,) + obs_shape)
        running_mean, running_var, running_count = update_running_stats(
            state.running_mean,
            state.running_var,
            state.running_count,
            jnp.asarray(flat_next_obs),
            batch_axes=(0,),
        )
        state = state.replace(
            running_mean=running_mean,
            running_var=running_var,
            running_count=running_count,
        )

        raw_int_rewards, int_rewards = intrinsic_rewards_fn(
            state.params,
            jnp.asarray(flat_next_obs),
            state.running_mean,
            state.running_var,
            state.reward_rms_var,
        )
        raw_int_rewards_np = np.asarray(jax.device_get(raw_int_rewards), dtype=np.float32)
        int_rewards_np = np.asarray(jax.device_get(int_rewards), dtype=np.float32).reshape(
            cfg.rollout_length,
            cfg.n_envs,
        )

        rew_mean, rew_var, rew_count = update_running_stats(
            state.reward_rms_mean,
            state.reward_rms_var,
            state.reward_rms_count,
            jnp.asarray(raw_int_rewards_np),
            batch_axes=(0,),
        )
        state = state.replace(
            reward_rms_mean=rew_mean,
            reward_rms_var=rew_var,
            reward_rms_count=rew_count,
            global_step=jnp.asarray(global_step, dtype=jnp.int32),
            rollout_idx=jnp.asarray(rollout_idx + 1, dtype=jnp.int32),
        )

        next_int_value, next_ext_value = values_fn(state.params, jnp.asarray(obs))
        next_int_value_np = np.asarray(jax.device_get(next_int_value), dtype=np.float32)
        next_ext_value_np = np.asarray(jax.device_get(next_ext_value), dtype=np.float32)

        ext_adv, ext_returns = compute_gae(
            ext_rewards_buf,
            ext_values_buf,
            next_ext_value_np,
            dones_buf,
            float(cfg.ext_gamma),
            float(cfg["lambda"]),
        )
        int_adv, int_returns = compute_gae(
            int_rewards_np,
            int_values_buf,
            next_int_value_np,
            np.zeros_like(dones_buf),
            float(cfg.int_gamma),
            float(cfg["lambda"]),
        )
        advantages = cfg.ext_adv_coeff * ext_adv + cfg.int_adv_coeff * int_adv
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        rnd_obs = normalize_obs(
            jnp.asarray(flatten_rollout(next_obs_buf)),
            state.running_mean,
            state.running_var,
        )
        train_batch = {
            "obs": flatten_rollout(obs_buf),
            "actions": flatten_rollout(actions_buf).astype(jnp.int32),
            "old_log_probs": flatten_rollout(log_probs_buf),
            "advantages": flatten_rollout(advantages),
            "returns_ext": flatten_rollout(ext_returns),
            "returns_int": flatten_rollout(int_returns),
            "rnd_obs": rnd_obs,
        }

        metrics_accum: dict[str, list[float]] = {}
        for _ in range(cfg.n_epochs):
            for mb in minibatches(np_rng, train_batch, int(cfg.n_mini_batch)):
                key, update_key = jax.random.split(key)
                state, metrics = update_minibatch(state, mb, update_key)
                metrics_np = jax.device_get(metrics)
                for name, value in metrics_np.items():
                    metrics_accum.setdefault(name, []).append(float(value))

        current_rollout = rollout_idx + 1
        should_checkpoint = (
            current_rollout == 1
            or current_rollout % cfg.checkpoint_interval_rollouts == 0
            or current_rollout == int(cfg.total_rollouts_per_env)
        )
        if should_checkpoint:
            checkpoint_manager.save(step=global_step, state=state, force=True)
            last_checkpoint_step = global_step

        if cfg.eval_steps > 0 and global_step % cfg.eval_steps < cfg.n_envs:
            steps, reward, video_path = record_eval_rollout(
                eval_env,
                policy_model,
                state.params,
                output_dir,
                global_step,
            )
            wb.log({"eval/steps": steps, "eval/reward": reward}, step=global_step)
            wb.log_video("eval/rollout", video_path, step=global_step)

        if rollout_idx % cfg.log_interval_rollouts == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            log_data = {
                "global_step": global_step,
                "rollout_idx": rollout_idx + 1,
                "stats/sps": global_step / elapsed,
                "stats/extrinsic_reward": float(ext_rewards_buf.mean()),
                "stats/intrinsic_reward": float(int_rewards_np.mean()),
                "stats/intrinsic_reward_raw": float(raw_int_rewards_np.mean()),
                "stats/advantage_mean": float(advantages.mean()),
                "stats/advantage_std": float(advantages.std()),
                "stats/running_count": float(jax.device_get(state.running_count)),
            }
            for name, values in metrics_accum.items():
                log_data[name] = float(np.mean(values))
            wb.log(log_data, step=global_step)

    if last_checkpoint_step != global_step:
        checkpoint_manager.save(step=global_step, state=state, force=True, wait=True)
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    wb.finish()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
