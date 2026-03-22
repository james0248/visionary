import os

import gymnasium as gym
import imageio
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState


def record_rollout(
    env: gym.Env,
    state: TrainState,
    output_dir: str,
    global_step: int,
) -> tuple[int, float, str]:
    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"rollout_{global_step}.mp4")

    obs, info = env.reset()
    fps = env.metadata.get("render_fps", 30)
    frames = [env.render()]

    @jax.jit
    def get_action(params, obs):
        q_values = state.apply_fn(params, obs[None])
        return jnp.argmax(q_values, axis=-1)

    total_reward = 0.0
    steps = 0

    while True:
        action = get_action(state.params, jnp.asarray(obs)).item()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        frames.append(env.render())

        if terminated or truncated:
            break

    imageio.mimsave(video_path, frames, fps=fps)

    return steps, total_reward, video_path
