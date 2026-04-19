import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange


class SwiGLU(nn.Module):
    hidden_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.swish(nn.Dense(self.hidden_dim, use_bias=False, dtype=self.dtype)(x))
        value = nn.Dense(self.hidden_dim, use_bias=False, dtype=self.dtype)(x)
        hidden = gate * value

        return nn.Dense(x.shape[-1], use_bias=False, dtype=self.dtype)(hidden)


def apply_rotary_embedding(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    x_rot = jnp.stack([-x[..., 1::2], x[..., 0::2]], axis=-1)
    x_rot = x_rot.reshape(*x_rot.shape[:-2], -1)

    cos = cos.astype(x.dtype)[None, :, None, :]
    sin = sin.astype(x.dtype)[None, :, None, :]

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
    attention_logit_soft_cap: float | None = 50.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_emb: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False, dtype=self.dtype)(x)
        k = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype)(x)
        v = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype)(x)

        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_kv_heads)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_kv_heads)

        q = nn.RMSNorm(dtype=self.dtype)(q)
        k = nn.RMSNorm(dtype=self.dtype)(k)

        if rope_emb is not None:
            q = apply_rotary_embedding(q, rope_emb[0], rope_emb[1])
            k = apply_rotary_embedding(k, rope_emb[0], rope_emb[1])

        out = jax.nn.dot_product_attention(q, k, v, mask=mask, scale=1.0 / jnp.sqrt(self.head_dim))
        out = rearrange(out, "b t h d -> b t (h d)")
        out = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype)(out)

        return out


class TransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    mlp_hidden_dim: int
    attention_logit_soft_cap: float | None = 50.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        residual = x
        x = nn.RMSNorm(dtype=self.dtype)(x)
        x = residual + Attention(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            attention_logit_soft_cap=self.attention_logit_soft_cap,
            dtype=self.dtype,
        )(x, rope_emb, mask)

        residual = x
        x = nn.RMSNorm(dtype=self.dtype)(x)
        x = residual + SwiGLU(self.mlp_hidden_dim, dtype=self.dtype)(x)

        return x


class SpatioTemporalTransformer(nn.Module):
    num_layers: int
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    mlp_hidden_dim: int
    temporal_layer_period: int = 4
    attention_logit_soft_cap: float | None = 50.0
    dtype: jnp.dtype = jnp.bfloat16

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
        if self.num_layers % self.temporal_layer_period != 0:
            raise ValueError(
                "num_layers must be divisible by temporal_layer_period, "
                f"got num_layers={self.num_layers} and "
                f"temporal_layer_period={self.temporal_layer_period}"
            )

        batch_size = x.shape[0]

        # (b, t, t) -> (b*n, 1, t, t) for head broadcast
        temporal_mask = jnp.repeat(temporal_mask, total_tokens, axis=0)
        temporal_mask = temporal_mask[:, None, :, :]

        def apply_block(
            block_idx: int,
            x: jnp.ndarray,
            rope_emb: tuple[jnp.ndarray, jnp.ndarray],
            mask: jnp.ndarray,
        ) -> jnp.ndarray:
            return TransformerBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                attention_logit_soft_cap=self.attention_logit_soft_cap,
                dtype=self.dtype,
                name=f"TransformerBlock_{block_idx}",
            )(x, rope_emb, mask)

        block_idx = 0
        num_groups = self.num_layers // self.temporal_layer_period
        for _ in range(num_groups):
            x = rearrange(x, "b t n d -> (b t) n d")
            for _ in range(self.temporal_layer_period - 1):
                x = apply_block(block_idx, x, spatial_rope_emb, spatial_mask)
                block_idx += 1
            x = rearrange(x, "(b t) n d -> b t n d", b=batch_size, t=t)

            x = rearrange(x, "b t n d -> (b n) t d")
            x = apply_block(block_idx, x, temporal_rope_emb, temporal_mask)
            block_idx += 1
            x = rearrange(x, "(b n) t d -> b t n d", b=batch_size, n=total_tokens)

        return x
