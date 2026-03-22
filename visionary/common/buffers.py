import gymnasium.spaces as spaces
import jax
import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple[int, ...],
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: jax.Device,
        n_envs: int = 1,
    ):
        self.pos = 0
        self.full = False
        self.buffer_size = buffer_size // n_envs
        self.n_envs = n_envs
        self.device = device

        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.obs = np.empty((self.buffer_size, n_envs, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.empty((self.buffer_size, n_envs, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty(
            (self.buffer_size, n_envs, action_dim), dtype=action_dtype
        )
        self.rewards = np.empty((self.buffer_size, n_envs), dtype=np.float32)
        self.dones = np.empty((self.buffer_size, n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: bool,
    ):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        upper = self.buffer_size if self.full else self.pos
        batch_indices = np.random.randint(0, upper, batch_size)
        env_indices = np.random.randint(0, self.n_envs, batch_size)

        return jax.device_put(
            (
                self.obs[batch_indices, env_indices],
                self.next_obs[batch_indices, env_indices],
                self.actions[batch_indices, env_indices],
                self.rewards[batch_indices, env_indices],
                self.dones[batch_indices, env_indices],
            ),
            self.device,
        )


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """Get the shape of the observation of gymnasium spaces."""

    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {
            key: get_obs_shape(subspace)
            for (key, subspace) in observation_space.spaces.items()
        }  # type: ignore[misc]

    else:
        raise NotImplementedError(
            f"{observation_space} observation space is not supported"
        )


def get_action_dim(action_space: spaces.Space) -> int:
    """Get the dimension of the action space of gymnasium spaces."""
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(action_space.n, int), (
            f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        )
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
