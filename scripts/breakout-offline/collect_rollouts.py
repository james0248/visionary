import glob
import logging
import os
import re

import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from omegaconf import DictConfig

from visionary.common.checkpoint import load_checkpoint
from visionary.common.env import FireResetEnv, FrameRecorder, make_vec_env
from dqn import DQN

logger = logging.getLogger(__name__)


def load_run_config(run_dir: str) -> dict:
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def discover_checkpoints(run_dir: str) -> list[tuple[int, str]]:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    pattern = os.path.join(ckpt_dir, "step_*.msgpack")
    paths = glob.glob(pattern)
    results = []
    for path in paths:
        match = re.search(r"step_(\d+)\.msgpack", os.path.basename(path))
        if match:
            results.append((int(match.group(1)), path))
    results.sort(key=lambda x: x[0])
    return results


def init_model(env, seed: int = 0):
    action_size = env.single_action_space.n
    model = DQN(action_size=action_size)
    obs_shape = env.single_observation_space.shape
    dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    init_params = model.init(jax.random.key(seed), dummy_obs)
    return model, init_params


def save_episode(
    episode_buffer: list, output_dir: str, step: int, episode_idx: int
) -> None:
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
    for r in recorders:
        r.pop_frames()

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


@hydra.main(config_path="config", config_name="collect_rollouts", version_base=None)
def main(cfg: DictConfig):
    run_cfg = load_run_config(cfg.run_dir)
    env_id = run_cfg["env"]
    frame_skip = run_cfg["frame_skip"]

    output_dir = os.path.join(cfg.run_dir, "rollouts")

    checkpoints = discover_checkpoints(cfg.run_dir)
    if cfg.start_from_step is not None:
        checkpoints = [(s, p) for s, p in checkpoints if s >= cfg.start_from_step]
    logger.info("Found %d checkpoints in %s", len(checkpoints), cfg.run_dir)

    recorders: list[FrameRecorder] = []

    def make_rollout_env(env_id: str, screen_size: int = 84):
        env = gym.make(
            env_id, render_mode="rgb_array", frameskip=1, max_episode_steps=10_800
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
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env, fire_on_life_loss=True)
        return env

    env = make_vec_env(lambda: make_rollout_env(env_id, cfg.screen_size), cfg.n_envs)
    model, init_params = init_model(env)

    @jax.jit
    def get_actions(params, obs):
        q_values = model.apply(params, obs)
        return jnp.argmax(q_values, axis=-1)

    for step, ckpt_path in checkpoints:
        logger.info("Collecting rollouts for checkpoint step=%d", step)
        params = load_checkpoint(ckpt_path, init_params)
        collect_rollouts_for_checkpoint(
            env,
            recorders,
            get_actions,
            params,
            cfg.n_envs,
            output_dir,
            step,
        )

    env.close()
    logger.info("Done. Rollouts saved to %s", output_dir)


if __name__ == "__main__":
    main()
