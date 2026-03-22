import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange


class Transformer(nn.Module):
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, t, n, d = x.shape

        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        k = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False)(x)
        v = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False)(x)

        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.num_kv_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.num_kv_heads)
