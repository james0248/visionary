import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange


class SwiGLU(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.swish(nn.Dense(self.hidden_dim, use_bias=False)(x))
        value = nn.Dense(self.hidden_dim, use_bias=False)(x)
        hidden = gate * value

        return nn.Dense(x.shape[-1], use_bias=False)(hidden)


def apply_rotary_embedding(
    x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> jnp.ndarray:
    x_rot = jnp.stack([-x[..., 1::2], x[..., 0::2]], axis=-1)
    x_rot = x_rot.reshape(*x_rot.shape[:-2], -1)

    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    emb = x * cos + x_rot * sin
    return emb


def create_temporal_rope(
    base: float, head_dim: int, seq_len: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    theta = 1 / (base ** (jnp.arange(0, head_dim, 2) / head_dim))
    indicies = jnp.arange(seq_len)
    angles = jnp.outer(indicies, theta)
    cos_emb = jnp.cos(angles).repeat(2, axis=-1)
    sin_emb = jnp.sin(angles).repeat(2, axis=-1)
    return cos_emb, sin_emb


def create_spatial_rope(
    base: float, head_dim: int, x_len: int, y_len: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    theta = 1 / (base ** (jnp.arange(0, head_dim, 4) / head_dim))
    indicies = jnp.arange(x_len * y_len)
    x_indicies = indicies % x_len
    y_indicies = indicies // x_len

    x_angles = jnp.outer(x_indicies, theta)
    y_angles = jnp.outer(y_indicies, theta)

    x_cos_emb = jnp.cos(x_angles).repeat(2, axis=-1)
    x_sin_emb = jnp.sin(x_angles).repeat(2, axis=-1)
    y_cos_emb = jnp.cos(y_angles).repeat(2, axis=-1)
    y_sin_emb = jnp.sin(y_angles).repeat(2, axis=-1)

    cos_emb = jnp.concatenate([x_cos_emb, y_cos_emb], axis=-1)
    sin_emb = jnp.concatenate([x_sin_emb, y_sin_emb], axis=-1)
    return cos_emb, sin_emb


def pad_rope_for_latents(
    rope_cos: jnp.ndarray, rope_sin: jnp.ndarray, num_latents: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return (
        jnp.pad(rope_cos, ((num_latents, 0), (0, 0)), constant_values=1),
        jnp.pad(rope_sin, ((num_latents, 0), (0, 0)), constant_values=0),
    )


class Attention(nn.Module):
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_emb: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        num_groups = self.num_heads // self.num_kv_heads

        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        k = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False)(x)
        v = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False)(x)

        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_kv_heads)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_kv_heads)

        q = nn.RMSNorm()(q)
        k = nn.RMSNorm()(k)

        if rope_emb is not None:
            q = apply_rotary_embedding(q, rope_emb[0], rope_emb[1])
            k = apply_rotary_embedding(k, rope_emb[0], rope_emb[1])

        k = jnp.repeat(k, repeats=num_groups, axis=2)
        v = jnp.repeat(v, repeats=num_groups, axis=2)

        scores = jnp.einsum("b q h d, b k h d -> b h q k", q, k)
        scores = scores / jnp.sqrt(self.head_dim)
        if mask is not None:
            scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)

        attn_weights = nn.softmax(scores, axis=-1)
        out = jnp.einsum("b h q k, b k h d -> b q h d", attn_weights, v)
        out = rearrange(out, "b t h d -> b t (h d)")
        out = nn.Dense(self.model_dim, use_bias=False)(out)

        return out


class TransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    mlp_hidden_dim: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        residual = x
        x = nn.RMSNorm()(x)
        x = residual + Attention(
            self.model_dim,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )(x, rope_emb, mask)

        residual = x
        x = nn.RMSNorm()(x)
        x = residual + SwiGLU(self.mlp_hidden_dim)(x)

        return x


class SpatioTemporalTransformer(nn.Module):
    num_layers: int
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    mlp_hidden_dim: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        t: int,
        total_tokens: int,
        spatial_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        spatial_mask: jnp.ndarray,
        temporal_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        temporal_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        # (b, t, t) -> (b*n, 1, t, t) for head broadcast
        temporal_mask = jnp.repeat(temporal_mask, total_tokens, axis=0)
        temporal_mask = temporal_mask[:, None, :, :]

        for i in range(1, self.num_layers + 1):
            if i % 4 == 0:
                x = rearrange(x, "b t n d -> (b n) t d")
                rope_emb, mask = temporal_rope_emb, temporal_mask
            else:
                x = rearrange(x, "b t n d -> (b t) n d")
                rope_emb, mask = spatial_rope_emb, spatial_mask

            x = TransformerBlock(
                self.model_dim,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.mlp_hidden_dim,
            )(x, rope_emb, mask)

            if i % 4 == 0:
                x = rearrange(x, "(b n) t d -> b t n d", n=total_tokens)
            else:
                x = rearrange(x, "(b t) n d -> b t n d", t=t)

        return x
