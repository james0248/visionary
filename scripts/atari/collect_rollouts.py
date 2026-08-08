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
from train_dqn import DQN
from omegaconf import DictConfig

from visionary.common.checkpoint import CheckpointManager
from env import FireResetEnv, FrameRecorder, has_fire_action, make_vec_env
from train_state import TargetTrainState

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


def clear_step_output(output_dir: str, step: int) -> None:
    step_dir = os.path.join(output_dir, f"step_{step}")
    if not os.path.isdir(step_dir):
        return

    removed = 0
    for filename in os.listdir(step_dir):
        if filename.endswith(".npz"):
            os.remove(os.path.join(step_dir, filename))
            removed += 1
    if removed:
        logger.info("Removed %d existing rollout files from %s", removed, step_dir)


def _repeated_action_entry(group: list) -> tuple[np.ndarray, int, float, bool, bool]:
    # FrameRecorder captures pre-step frames, so the first entry matches this action.
    frame, action, *_ = group[0]
    return (
        frame,
        int(action),
        sum(float(entry[2]) for entry in group),
        any(bool(entry[3]) for entry in group),
        any(bool(entry[4]) for entry in group),
    )


def make_repeated_action_entries(
    raw_entries: list,
    frame_skip: int,
    env_idx: int,
    step_count: int,
) -> list[tuple[np.ndarray, int, float, bool, bool]]:
    entries = []
    group = []
    for entry in raw_entries:
        if entry is None:
            break
        group.append(entry)
        _, _, _, terminated, truncated = entry
        if len(group) == frame_skip or terminated or truncated:
            entries.append(_repeated_action_entry(group))
            group = []

    if group:
        entries.append(_repeated_action_entry(group))

    if not entries:
        raise RuntimeError(
            "Expected at least one recorded ALE frame for "
            f"env={env_idx} outer_step={step_count}."
        )
    return entries


def collect_rollouts_for_checkpoint(
    env,
    recorders: list[FrameRecorder],
    get_actions,
    params,
    n_envs: int,
    output_dir: str,
    step: int,
    frame_skip: int,
) -> None:
    episode_buffers = [[] for _ in range(n_envs)]
    done = np.zeros(n_envs, dtype=bool)
    clear_step_output(output_dir, step)

    obs, _ = env.reset()
    for r in recorders:
        r.pop_frames()

    step_count = 0
    while not np.all(done):
        actions = np.asarray(get_actions(params, jnp.asarray(obs)))
        obs, _, terminated, truncated, _ = env.step(actions)

        for i in range(n_envs):
            raw = recorders[i].pop_frames()
            if done[i]:
                continue
            episode_buffers[i].extend(
                make_repeated_action_entries(raw, frame_skip, i, step_count)
            )

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
    fire_reset = run_cfg.get("fire_reset", True)
    max_episode_steps = cfg.get("max_episode_steps", 10_800)

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
        env_kwargs = {"render_mode": "rgb_array", "frameskip": 1}
        if max_episode_steps is not None:
            env_kwargs["max_episode_steps"] = int(max_episode_steps)
        env = gym.make(env_id, **env_kwargs)
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
    model = DQN(action_size=action_size)
    obs_shape = env.single_observation_space.shape
    dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.uint8)
    params = model.init(jax.random.key(0), dummy_obs)
    optimizer = optax.adam(run_cfg["learning_rate"])
    init_state = TargetTrainState.create(
        apply_fn=model.apply,
        params=params,
        target_params=params,
        tx=optimizer,
    )

    @jax.jit
    def get_actions(params, obs):
        q_values = model.apply(params, obs)
        return jnp.argmax(q_values, axis=-1)

    for step in checkpoints:
        logger.info("Collecting rollouts for checkpoint step=%d", step)
        params = checkpoint_manager.restore(
            step=step,
            target=init_state,
            params_only=True,
        )
        collect_rollouts_for_checkpoint(
            env,
            recorders,
            get_actions,
            params,
            cfg.n_envs,
            output_dir,
            step,
            frame_skip,
        )

    checkpoint_manager.close()
    env.close()
    logger.info("Done. Rollouts saved to %s", output_dir)


if __name__ == "__main__":
    main()
