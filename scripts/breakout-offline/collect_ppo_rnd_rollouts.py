import logging
import os

import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import yaml
from omegaconf import DictConfig
from ppo_rnd import ActorCritic, PPORNDTrainState, RNDPredictor, RNDTarget

from visionary.common.checkpoint import CheckpointManager
from visionary.common.env import FireResetEnv, FrameRecorder, has_fire_action, make_vec_env

logger = logging.getLogger(__name__)


def load_run_config(run_dir: str) -> dict:
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_episode(episode_buffer: list, output_dir: str, step: int, episode_idx: int) -> None:
    step_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)

    frames, actions, rewards, terminations, truncations = zip(*episode_buffer)
    np.savez_compressed(
        os.path.join(step_dir, f"episode_{episode_idx}.npz"),
        frames=np.array(frames, dtype=np.uint8),
        actions=np.array(actions, dtype=np.int32),
        rewards=np.array(rewards, dtype=np.float32),
        terminations=np.array(terminations, dtype=bool),
        truncations=np.array(truncations, dtype=bool),
    )
    logger.info(
        "Saved episode %d for step %d (%d transitions)",
        episode_idx,
        step,
        len(frames),
    )


def collect_rollouts_for_checkpoint(
    env,
    recorders: list[FrameRecorder],
    get_actions,
    params,
    n_envs: int,
    output_dir: str,
    step: int,
) -> None:
    episode_buffers = [[] for _ in range(n_envs)]
    done = np.zeros(n_envs, dtype=bool)

    obs, _ = env.reset()
    for recorder in recorders:
        recorder.pop_frames()

    step_count = 0
    while not np.all(done):
        actions = np.asarray(get_actions(params, jnp.asarray(obs)))
        obs, _, terminated, truncated, _ = env.step(actions)

        for i in range(n_envs):
            raw = recorders[i].pop_frames()
            if not done[i]:
                for entry in raw:
                    if entry is None:
                        break
                    episode_buffers[i].append(entry)

        episode_done = (terminated | truncated) & ~done
        for i in range(n_envs):
            if episode_done[i]:
                save_episode(episode_buffers[i], output_dir, step, i)
                done[i] = True

        step_count += 1
        if step_count % 250 == 0:
            logger.info("step=%d, done=%d/%d", step_count, done.sum(), n_envs)


def create_initial_state(
    run_cfg: dict,
    obs_shape: tuple[int, ...],
    action_size: int,
) -> tuple[PPORNDTrainState, ActorCritic]:
    policy_model = ActorCritic(action_size=action_size)
    target_model = RNDTarget()
    predictor_model = RNDPredictor()

    dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    dummy_rnd_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)
    policy_key, target_key, predictor_key = jax.random.split(jax.random.key(0), 3)
    params = {
        "policy": policy_model.init(policy_key, dummy_obs),
        "rnd_target": target_model.init(target_key, dummy_rnd_obs),
        "rnd_predictor": predictor_model.init(predictor_key, dummy_rnd_obs),
    }

    optimizer = optax.chain(
        optax.clip_by_global_norm(run_cfg.get("max_grad_norm", 0.5)),
        optax.adam(run_cfg["learning_rate"]),
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
    return state, policy_model


@hydra.main(config_path="config", config_name="collect_ppo_rnd_rollouts", version_base=None)
def main(cfg: DictConfig):
    run_cfg = load_run_config(cfg.run_dir)
    if run_cfg.get("algorithm") != "ppo_rnd":
        raise ValueError(f"Expected a PPO-RND run, got algorithm={run_cfg.get('algorithm')!r}")

    env_id = run_cfg["env"]
    frame_skip = run_cfg["frame_skip"]
    fire_reset = run_cfg.get("fire_reset", True)
    max_episode_steps = run_cfg.get("max_frames_episode", 4500)

    output_dir = os.path.join(cfg.run_dir, "rollouts")
    checkpoint_manager = CheckpointManager(
        os.path.join(cfg.run_dir, "checkpoints"),
        options=ocp.CheckpointManagerOptions(
            enable_async_checkpointing=False,
            read_only=True,
        ),
    )

    checkpoints = checkpoint_manager.all_steps()
    if cfg.start_from_step is not None:
        checkpoints = [step for step in checkpoints if step >= cfg.start_from_step]
    logger.info("Found %d checkpoints in %s", len(checkpoints), cfg.run_dir)

    recorders: list[FrameRecorder] = []

    def make_rollout_env(env_id: str, screen_size: int = 84):
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            frameskip=1,
            max_episode_steps=max_episode_steps,
        )
        recorder = FrameRecorder(env)
        recorders.append(recorder)
        env = gym.wrappers.AtariPreprocessing(
            recorder,
            frame_skip=frame_skip,
            screen_size=screen_size,
            grayscale_obs=True,
            grayscale_newaxis=True,
            scale_obs=False,
        )
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        if fire_reset and has_fire_action(env.unwrapped.get_action_meanings()):
            env = FireResetEnv(env, fire_on_life_loss=True)
        return env

    env = make_vec_env(lambda: make_rollout_env(env_id, cfg.screen_size), cfg.n_envs)
    action_size = env.single_action_space.n
    obs_shape = env.single_observation_space.shape
    init_state, policy_model = create_initial_state(run_cfg, obs_shape, action_size)

    @jax.jit
    def get_argmax_actions(params, obs):
        logits, _, _ = policy_model.apply(params["policy"], obs)
        return jnp.argmax(logits, axis=-1)

    @jax.jit
    def get_sample_actions(params, obs, key, temperature):
        logits, _, _ = policy_model.apply(params["policy"], obs)
        logits = logits / jnp.maximum(temperature, 1e-6)
        return jax.random.categorical(key, logits, axis=-1)

    key_box = [jax.random.key(run_cfg.get("seed", 0))]

    def get_actions(params, obs):
        if cfg.policy_mode == "argmax":
            return get_argmax_actions(params, obs)
        if cfg.policy_mode == "sample":
            key_box[0], action_key = jax.random.split(key_box[0])
            return get_sample_actions(params, obs, action_key, jnp.asarray(cfg.temperature))
        raise ValueError(f"Unknown policy_mode={cfg.policy_mode!r}")

    for step in checkpoints:
        logger.info("Collecting PPO-RND rollouts for checkpoint step=%d", step)
        restored_state = checkpoint_manager.restore(
            step=step,
            target=init_state,
            params_only=False,
        )
        collect_rollouts_for_checkpoint(
            env,
            recorders,
            get_actions,
            restored_state.params,
            cfg.n_envs,
            output_dir,
            step,
        )

    checkpoint_manager.close()
    env.close()
    logger.info("Done. Rollouts saved to %s", output_dir)


if __name__ == "__main__":
    main()
