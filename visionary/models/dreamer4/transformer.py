import os
from functools import lru_cache

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange

SPLASH_INTERPRET = os.environ.get("SPLASH_INTERPRET") == "1"


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
    x_left, x_right = jnp.split(x, 2, axis=-1)

    cos = cos.astype(x.dtype)[None, :, None, :]
    sin = sin.astype(x.dtype)[None, :, None, :]

    rotated_left = x_left * cos - x_right * sin
    rotated_right = x_right * cos + x_left * sin
    return jnp.concatenate([rotated_left, rotated_right], axis=-1)


def create_temporal_rope(
    base: float, head_dim: int, seq_len: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    half_head_dim = head_dim // 2
    theta = 1 / (base ** (jnp.arange(half_head_dim) / half_head_dim))
    indicies = jnp.arange(seq_len)
    angles = jnp.outer(indicies, theta)
    cos_emb = jnp.cos(angles)
    sin_emb = jnp.sin(angles)
    return cos_emb, sin_emb


def create_spatial_rope(
    base: float, head_dim: int, x_len: int, y_len: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    quarter_head_dim = head_dim // 4
    theta = 1 / (base ** (jnp.arange(quarter_head_dim) / quarter_head_dim))
    indicies = jnp.arange(x_len * y_len)
    x_indicies = indicies % x_len
    y_indicies = indicies // x_len

    x_angles = jnp.outer(x_indicies, theta)
    y_angles = jnp.outer(y_indicies, theta)

    x_cos_emb = jnp.cos(x_angles)
    x_sin_emb = jnp.sin(x_angles)
    y_cos_emb = jnp.cos(y_angles)
    y_sin_emb = jnp.sin(y_angles)

    cos_emb = jnp.concatenate([x_cos_emb, y_cos_emb], axis=-1)
    sin_emb = jnp.concatenate([x_sin_emb, y_sin_emb], axis=-1)
    return cos_emb, sin_emb


def pad_rope_for_latents(
    rope_cos: jnp.ndarray, rope_sin: jnp.ndarray, num_latents: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    latent_cos = jnp.ones((num_latents, rope_cos.shape[-1]), dtype=rope_cos.dtype)
    latent_sin = jnp.zeros((num_latents, rope_sin.shape[-1]), dtype=rope_sin.dtype)
    return (
        jnp.concatenate([latent_cos, rope_cos], axis=0),
        jnp.concatenate([latent_sin, rope_sin], axis=0),
    )


def resolve_remat_policy(name: str | None):
    if name is None:
        return None
    policy = getattr(jax.checkpoint_policies, name, None)
    if policy is None:
        raise ValueError(f"Unknown remat_policy: {name!r}")
    return policy


@lru_cache(maxsize=8)
def _splash_mask(n_latents, n_image, encoder, seq_pad):
    seq = n_latents + n_image
    mask = np.zeros((seq_pad, seq_pad), dtype=bool)
    mask[:n_latents, :seq] = True
    mask[n_latents:seq, n_latents:seq] = True
    if encoder:
        mask[n_latents:seq, :n_latents] = False
    else:
        mask[:n_latents, n_latents:seq] = False
        mask[n_latents:seq, :n_latents] = True
    np.fill_diagonal(mask, True)
    return mask


def _splash_kernel(n_latents, n_image, encoder, num_heads, seq_pad, interpret):
    # constructed per trace: the kernel closes over arrays it creates, and
    # caching those across jit traces leaks tracers
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sk,
        splash_attention_mask as sm,
    )

    mask = _splash_mask(n_latents, n_image, encoder, seq_pad)
    b = min(int(os.environ.get("SPLASH_BLOCK", "640")), seq_pad)
    block = sk.BlockSizes(
        block_q=b, block_kv=b, block_kv_compute=b,
        block_q_dkv=b, block_kv_dkv=b, block_kv_dkv_compute=b,
        block_q_dq=b, block_kv_dq=b,
    )
    return sk.make_splash_mha(
        mask=sm.MultiHeadMask([sm.NumpyMask(mask)] * num_heads),
        block_sizes=block,
        head_shards=1,
        q_seq_shards=1,
        interpret=interpret,
    )


SPLASH_MESH = None
SPLASH_AXIS = "data"


def splash_attention(q, k, v, spec, scale):
    n_latents, n_image, encoder = spec
    batch, seq, num_heads, head_dim = q.shape
    seq_pad = ((seq + 127) // 128) * 128

    repeats = num_heads // k.shape[2]
    k = jnp.repeat(k, repeats, axis=2)
    v = jnp.repeat(v, repeats, axis=2)

    def prep(x):
        x = jnp.pad(x, ((0, 0), (0, seq_pad - seq), (0, 0), (0, 0)))
        return jnp.swapaxes(x, 1, 2)

    def run(q, k, v):
        kernel = _splash_kernel(
            n_latents, n_image, encoder, num_heads, seq_pad, SPLASH_INTERPRET
        )
        out = jax.vmap(kernel)(prep(q), prep(k), prep(v))
        return jnp.swapaxes(out, 1, 2)[:, :seq]

    if SPLASH_MESH is not None:
        # Mosaic kernels cannot be auto-partitioned by GSPMD
        p = jax.sharding.PartitionSpec(SPLASH_AXIS)
        run = jax.shard_map(
            run, mesh=SPLASH_MESH, in_specs=(p, p, p), out_specs=p, check_vma=False
        )
    return run(q * scale, k, v)


class Attention(nn.Module):
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    attention_logit_soft_cap: float | None = 50.0
    splash_spec: tuple | None = None
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

        scale = 1.0 / jnp.sqrt(self.head_dim)
        if self.splash_spec is not None and (
            jax.default_backend() == "tpu" or SPLASH_INTERPRET
        ):
            out = splash_attention(q, k, v, self.splash_spec, scale)
        else:
            out = jax.nn.dot_product_attention(q, k, v, mask=mask, scale=scale)
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
    splash_spec: tuple | None = None
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
            splash_spec=self.splash_spec,
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
    temporal_layer_offset: int = 1
    attention_logit_soft_cap: float | None = 50.0
    splash_spec: tuple | None = None
    remat: bool = False
    remat_policy: str | None = None
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
        if not 0 <= self.temporal_layer_offset < self.temporal_layer_period:
            raise ValueError(
                f"temporal_layer_offset={self.temporal_layer_offset} must be in "
                f"[0, {self.temporal_layer_period})"
            )
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

        # Recomputes each block in the backward pass instead of keeping its
        # activations. Same parameter tree either way, so checkpoints carry over.
        # remat_policy names an attribute of jax.checkpoint_policies; None keeps
        # only the block input and recomputes everything else.
        block_cls = (
            nn.remat(TransformerBlock, policy=resolve_remat_policy(self.remat_policy))
            if self.remat
            else TransformerBlock
        )

        def apply_block(
            block_idx: int,
            x: jnp.ndarray,
            rope_emb: tuple[jnp.ndarray, jnp.ndarray],
            mask: jnp.ndarray,
            splash_spec: tuple | None = None,
        ) -> jnp.ndarray:
            return block_cls(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                attention_logit_soft_cap=self.attention_logit_soft_cap,
                splash_spec=splash_spec,
                dtype=self.dtype,
                name=f"TransformerBlock_{block_idx}",
            )(x, rope_emb, mask)

        def spatial_run(x, block_idx, count):
            if count == 0:
                return x, block_idx
            x = rearrange(x, "b t n d -> (b t) n d")
            for _ in range(count):
                x = apply_block(block_idx, x, spatial_rope_emb, spatial_mask, self.splash_spec)
                block_idx += 1
            return rearrange(x, "(b t) n d -> b t n d", b=batch_size, t=t), block_idx

        offset = self.temporal_layer_offset
        block_idx = 0
        num_groups = self.num_layers // self.temporal_layer_period
        for _ in range(num_groups):
            x, block_idx = spatial_run(x, block_idx, offset)

            x = rearrange(x, "b t n d -> (b n) t d")
            x = apply_block(block_idx, x, temporal_rope_emb, temporal_mask)
            block_idx += 1
            x = rearrange(x, "(b n) t d -> b t n d", b=batch_size, n=total_tokens)

            x, block_idx = spatial_run(x, block_idx, self.temporal_layer_period - 1 - offset)

        return nn.RMSNorm(dtype=self.dtype, name="final_norm")(x)
