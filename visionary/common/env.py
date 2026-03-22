from collections.abc import Callable

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode

gym.register_envs(ale_py)


def make_vec_env(make_env: Callable, n_envs: int):
    return gym.vector.SyncVectorEnv(
        [make_env for _ in range(n_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )


class EpisodicLifeEnv(gym.Wrapper):
    """Make loss of life terminal (but only reset on true game over)."""

    def __init__(self, env):
        super().__init__(env)
        self.was_real_done = True
        self.lives = 0

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        self.was_real_done = True
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info


class ClipRewardEnv(gym.Wrapper):
    """Clip rewards to {-1, 0, +1} by taking their sign."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, np.sign(reward), terminated, truncated, info


class FrameRecorder(gym.Wrapper):
    """Records every ALE frame passing through, even those hidden by frame_skip or FireResetEnv."""

    def __init__(self, env):
        super().__init__(env)
        self.frame_buffer = []

    def step(self, action):
        frame = self.env.render()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frame_buffer.append((frame, action, reward, terminated, truncated))
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.frame_buffer.append(None)
        return self.env.reset(**kwargs)

    def pop_frames(self):
        frames = self.frame_buffer
        self.frame_buffer = []
        return frames


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset so Atari games (e.g. Breakout) start immediately."""

    def __init__(self, env, fire_on_life_loss=False):
        super().__init__(env)
        self.fire_on_life_loss = fire_on_life_loss

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.fire_on_life_loss:
            lives = self.env.unwrapped.ale.lives()
            if 0 < lives < self.lives:
                obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
                if terminated or truncated:
                    obs, info = self.env.reset()
            self.lives = lives
        return obs, reward, terminated, truncated, info
