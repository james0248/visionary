import flax.linen as nn
import jax.numpy as jnp


class SwiGLU(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.swish(nn.Dense(self.hidden_dim, use_bias=False)(x))
        value = nn.Dense(self.hidden_dim, use_bias=False)(x)
        hidden = gate * value

        return nn.Dense(x.shape[-1], use_bias=False)(hidden)
