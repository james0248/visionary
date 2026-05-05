from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig

import visionary.dynamics as dynamics_module
import visionary.tokenizer as tokenizer_module
import visionary.transformer as transformer_module
from visionary.dynamics import DynamicsModel
from visionary.tokenizer import Tokenizer
from visionary.transformer import SwiGLU, apply_rotary_embedding


@dataclass(frozen=True)
class TokenizerShapes:
    batch_size: int
    seq_len: int
    num_latents: int
    channel_dim: int
    patch_count: int
    patch_dim: int

    @property
    def latent(self) -> tuple[int, int, int, int]:
        return (self.batch_size, self.seq_len, self.num_latents, self.channel_dim)

    @property
    def patches(self) -> tuple[int, int, int, int]:
        return (self.batch_size, self.seq_len, self.patch_count, self.patch_dim)


@dataclass(frozen=True)
class DynamicsShapes:
    batch_size: int
    seq_len: int
    num_obs_tokens: int
    token_dim: int
    total_tokens: int
    temporal_blocks: int
    num_kv_heads: int
    head_dim: int
    context_length: int

    @property
    def z(self) -> tuple[int, int, int, int]:
        return (self.batch_size, self.seq_len, self.num_obs_tokens, self.token_dim)

    @property
    def levels(self) -> tuple[int, int]:
        return (self.batch_size, self.seq_len)

    @property
    def cache(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.temporal_blocks,
            self.batch_size,
            self.total_tokens,
            self.context_length,
            self.num_kv_heads,
            self.head_dim,
        )

    @property
    def layer_cache(self) -> tuple[int, int, int, int, int]:
        return (
            self.batch_size,
            self.total_tokens,
            self.context_length,
            self.num_kv_heads,
            self.head_dim,
        )

    @property
    def step_z(self) -> tuple[int, int, int, int]:
        return (self.batch_size, 1, self.num_obs_tokens, self.token_dim)

    @property
    def step_levels(self) -> tuple[int, int]:
        return (self.batch_size, 1)

    @property
    def position_index(self) -> tuple[int]:
        return (1,)


def create_tokenizer(cfg: DictConfig, *, dtype: Any | None = jnp.float32) -> Tokenizer:
    if dtype is None:
        return instantiate(cfg)
    return instantiate(cfg, dtype=dtype)


def create_dynamics(cfg: DictConfig, *, dtype: Any | None = jnp.float32) -> DynamicsModel:
    if dtype is None:
        return instantiate(cfg)
    return instantiate(cfg, dtype=dtype)


def tokenizer_shapes(
    cfg: DictConfig,
    *,
    batch_size: int,
    seq_len: int,
) -> TokenizerShapes:
    patch_size = int(cfg.patch_size)
    height = int(cfg.resize_shape[0])
    width = int(cfg.resize_shape[1])
    height_pad = int(cfg.pad_width[0])
    width_pad = int(cfg.pad_width[1])
    y_len = (height + 2 * height_pad) // patch_size
    x_len = (width + 2 * width_pad) // patch_size
    return TokenizerShapes(
        batch_size=batch_size,
        seq_len=seq_len,
        num_latents=int(cfg.num_latents),
        channel_dim=int(cfg.channel_dim),
        patch_count=x_len * y_len,
        patch_dim=patch_size * patch_size * 3,
    )


def dynamics_shapes(
    cfg: DictConfig,
    tokenizer: TokenizerShapes,
    *,
    batch_size: int,
    seq_len: int,
) -> DynamicsShapes:
    num_obs_tokens = int(cfg.num_obs_tokens)
    latent_width = tokenizer.num_latents * tokenizer.channel_dim
    if latent_width % num_obs_tokens != 0:
        raise ValueError(
            "Tokenizer latent width must be divisible by dynamics num_obs_tokens, "
            f"got {latent_width=} and {num_obs_tokens=}."
        )
    temporal_period = int(cfg.temporal_layer_period)
    num_layers = int(cfg.num_layers)
    if num_layers % temporal_period != 0:
        raise ValueError(
            "Dynamics num_layers must be divisible by temporal_layer_period, "
            f"got num_layers={num_layers} and temporal_layer_period={temporal_period}."
        )
    return DynamicsShapes(
        batch_size=batch_size,
        seq_len=seq_len,
        num_obs_tokens=num_obs_tokens,
        token_dim=latent_width // num_obs_tokens,
        total_tokens=1 + 1 + int(cfg.num_registers) + num_obs_tokens,
        temporal_blocks=num_layers // temporal_period,
        num_kv_heads=int(cfg.num_kv_heads),
        head_dim=int(cfg.head_dim),
        context_length=int(cfg.context_length),
    )


def validate_sample_steps(sample_steps: int) -> int:
    sample_steps = int(sample_steps)
    if sample_steps <= 0 or sample_steps & (sample_steps - 1):
        raise ValueError(f"sample_steps must be a positive power of two, got {sample_steps}.")
    return int(round(math.log2(sample_steps)))


def flow_update_z(
    current_z: jnp.ndarray,
    pred_z: jnp.ndarray,
    *,
    signal_level: int,
    sample_steps: int,
) -> jnp.ndarray:
    tau = jnp.asarray(signal_level / sample_steps, dtype=jnp.float32)
    step_size = jnp.asarray(1.0 / sample_steps, dtype=jnp.float32)
    denom = jnp.maximum(1.0 - tau, 1e-6)
    velocity = (pred_z.astype(jnp.float32) - current_z.astype(jnp.float32)) / denom
    return current_z.astype(jnp.float32) + velocity * step_size


def _export_dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    *,
    bias: jnp.ndarray | None = None,
    mask: jnp.ndarray | None = None,
    scale: float | jnp.ndarray | None = None,
    is_causal: bool = False,
    **_kwargs: Any,
) -> jnp.ndarray:
    if bias is not None:
        raise NotImplementedError("Export attention wrapper does not support attention bias.")
    if is_causal:
        raise NotImplementedError("Export attention wrapper expects explicit masks, not is_causal.")

    num_heads = int(query.shape[-2])
    num_kv_heads = int(key.shape[-2])
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads must be divisible by num_kv_heads, got {num_heads=} {num_kv_heads=}."
        )

    if scale is None:
        scale = jnp.asarray(1.0 / (query.shape[-1] ** 0.5), dtype=query.dtype)
    else:
        scale = jnp.asarray(scale, dtype=query.dtype)

    if num_heads == num_kv_heads:
        logits = (jnp.einsum("bqhd,bkhd->bhqk", query, key) * scale).astype(jnp.float32)
        if mask is not None:
            mask = mask.astype(logits.dtype)
            logits = logits + (1.0 - mask) * jnp.asarray(-1.0e9, dtype=logits.dtype)
        weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
        return jnp.einsum("bhqk,bkhd->bqhd", weights, value)

    repeat = num_heads // num_kv_heads
    if _ATTENTION_EXPORT_LOWERING == "split_gqa":
        outputs = []
        for kv_idx in range(num_kv_heads):
            group_start = kv_idx * repeat
            q_group = query[..., group_start : group_start + repeat, :]
            k_head = key[..., kv_idx, :]
            v_head = value[..., kv_idx, :]

            logits = (jnp.einsum("bqhd,bkd->bhqk", q_group, k_head) * scale).astype(
                jnp.float32
            )
            if mask is not None:
                mask = mask.astype(logits.dtype)
                logits = logits + (1.0 - mask) * jnp.asarray(-1.0e9, dtype=logits.dtype)
            weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
            outputs.append(jnp.einsum("bhqk,bkd->bqhd", weights, v_head))
        return jnp.concatenate(outputs, axis=-2)

    key = jnp.repeat(key, repeat, axis=-2)
    value = jnp.repeat(value, repeat, axis=-2)

    logits = (jnp.einsum("bqhd,bkhd->bhqk", query, key) * scale).astype(jnp.float32)
    if mask is not None:
        mask = mask.astype(logits.dtype)
        logits = logits + (1.0 - mask) * jnp.asarray(-1.0e9, dtype=logits.dtype)
    weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, value)


def _attention_for_export(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    *,
    mask: jnp.ndarray | None = None,
    scale: float | jnp.ndarray | None = None,
) -> jnp.ndarray:
    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=mask,
        scale=scale,
    )


def _export_create_temporal_rope(
    base: float,
    head_dim: int,
    seq_len: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    half_head_dim = head_dim // 2
    theta = 1.0 / (base ** (jnp.arange(half_head_dim, dtype=jnp.float32) / half_head_dim))
    indices = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(indices, theta)
    return jnp.cos(angles), jnp.sin(angles)


def _export_create_spatial_rope(
    base: float,
    head_dim: int,
    x_len: int,
    y_len: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    quarter_head_dim = head_dim // 4
    theta = 1.0 / (base ** (jnp.arange(quarter_head_dim, dtype=jnp.float32) / quarter_head_dim))
    indices = jnp.arange(x_len * y_len, dtype=jnp.float32)
    x_indices = jnp.mod(indices, float(x_len))
    y_indices = jnp.floor(indices / float(x_len))

    x_angles = jnp.outer(x_indices, theta)
    y_angles = jnp.outer(y_indices, theta)

    cos_emb = jnp.concatenate([jnp.cos(x_angles), jnp.cos(y_angles)], axis=-1)
    sin_emb = jnp.concatenate([jnp.sin(x_angles), jnp.sin(y_angles)], axis=-1)
    return cos_emb, sin_emb


def _export_create_temporal_mask(independent: jnp.ndarray, t: int) -> jnp.ndarray:
    positions = jnp.arange(t, dtype=jnp.int32)
    query_positions = positions[:, None]
    key_positions = positions[None, :]
    causal = key_positions <= query_positions
    return jnp.broadcast_to(causal[None], (independent.shape[0], t, t))


class _ExportRMSNorm(nn.Module):
    dtype: jnp.dtype = jnp.bfloat16
    epsilon: float = 1.0e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        y = x_f32 * jax.lax.rsqrt(variance + jnp.asarray(self.epsilon, dtype=jnp.float32))
        y = y * scale.astype(jnp.float32)
        return y.astype(self.dtype)


class _ExportAttention(nn.Module):
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

        if _ATTENTION_EXPORT_LAYOUT == "bnsh":
            q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
            k = rearrange(k, "b t (h d) -> b h t d", h=self.num_kv_heads)
            v = rearrange(v, "b t (h d) -> b h t d", h=self.num_kv_heads)
        else:
            q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
            k = rearrange(k, "b t (h d) -> b t h d", h=self.num_kv_heads)
            v = rearrange(v, "b t (h d) -> b t h d", h=self.num_kv_heads)

        q = _ExportRMSNorm(dtype=self.dtype, name="RMSNorm_0")(q)
        k = _ExportRMSNorm(dtype=self.dtype, name="RMSNorm_1")(k)

        if rope_emb is not None:
            if _ATTENTION_EXPORT_LAYOUT == "bnsh":
                q = _apply_rotary_embedding_bnsh(q, rope_emb[0], rope_emb[1])
                k = _apply_rotary_embedding_bnsh(k, rope_emb[0], rope_emb[1])
            else:
                q = apply_rotary_embedding(q, rope_emb[0], rope_emb[1])
                k = apply_rotary_embedding(k, rope_emb[0], rope_emb[1])

        if _ATTENTION_EXPORT_LAYOUT == "bnsh":
            out = _attention_for_export_bnsh(
                q,
                k,
                v,
                mask=mask,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            out = rearrange(out, "b h t d -> b t (h d)")
        else:
            out = _attention_for_export(
                q,
                k,
                v,
                mask=mask,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            out = rearrange(out, "b t h d -> b t (h d)")
        out = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype, name="Dense_3")(out)
        return out


class _ExportTransformerBlock(nn.Module):
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
        x = _ExportRMSNorm(dtype=self.dtype, name="RMSNorm_0")(x)
        x = residual + _ExportAttention(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            attention_logit_soft_cap=self.attention_logit_soft_cap,
            dtype=self.dtype,
            name="Attention_0",
        )(x, rope_emb, mask)

        residual = x
        x = _ExportRMSNorm(dtype=self.dtype, name="RMSNorm_1")(x)
        x = residual + SwiGLU(self.mlp_hidden_dim, dtype=self.dtype)(x)
        return x


class _ExportSpatioTemporalTransformer(nn.Module):
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
        temporal_mask = jnp.repeat(temporal_mask, total_tokens, axis=0)[:, None, :, :]

        block_idx = 0
        num_groups = self.num_layers // self.temporal_layer_period
        for _ in range(num_groups):
            x = rearrange(x, "b t n d -> (b t) n d")
            for _ in range(self.temporal_layer_period - 1):
                x = _ExportTransformerBlock(
                    model_dim=self.model_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    mlp_hidden_dim=self.mlp_hidden_dim,
                    attention_logit_soft_cap=self.attention_logit_soft_cap,
                    dtype=self.dtype,
                    name=f"TransformerBlock_{block_idx}",
                )(x, spatial_rope_emb, spatial_mask)
                block_idx += 1
            x = rearrange(x, "(b t) n d -> b t n d", b=batch_size, t=t)

            x = rearrange(x, "b t n d -> (b n) t d")
            x = _ExportTransformerBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                attention_logit_soft_cap=self.attention_logit_soft_cap,
                dtype=self.dtype,
                name=f"TransformerBlock_{block_idx}",
            )(x, temporal_rope_emb, temporal_mask)
            block_idx += 1
            x = rearrange(x, "(b n) t d -> b t n d", b=batch_size, n=total_tokens)

        return x


_ExportSpatioTemporalTransformer.__name__ = "SpatioTemporalTransformer"
_ExportSpatioTemporalTransformer.__qualname__ = "SpatioTemporalTransformer"


_ATTENTION_EXPORT_LOWERING = "manual"
_ATTENTION_EXPORT_LAYOUT = "bshd"


def set_attention_export_lowering(mode: str) -> None:
    if mode not in {"manual", "native", "split_gqa"}:
        raise ValueError(f"Unsupported attention export lowering: {mode}")
    global _ATTENTION_EXPORT_LOWERING
    _ATTENTION_EXPORT_LOWERING = mode


def set_attention_export_layout(mode: str) -> None:
    if mode not in {"bshd", "bnsh"}:
        raise ValueError(f"Unsupported attention export layout: {mode}")
    global _ATTENTION_EXPORT_LAYOUT
    _ATTENTION_EXPORT_LAYOUT = mode


def _apply_rotary_embedding_bnsh(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> jnp.ndarray:
    x_left, x_right = jnp.split(x, 2, axis=-1)
    cos = cos.astype(x.dtype)[None, None, :, :]
    sin = sin.astype(x.dtype)[None, None, :, :]
    rotated_left = x_left * cos - x_right * sin
    rotated_right = x_right * cos + x_left * sin
    return jnp.concatenate([rotated_left, rotated_right], axis=-1)


def _attention_for_export_bnsh(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    *,
    mask: jnp.ndarray | None = None,
    scale: float | jnp.ndarray | None = None,
) -> jnp.ndarray:
    num_heads = int(query.shape[1])
    num_kv_heads = int(key.shape[1])
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads must be divisible by num_kv_heads, got {num_heads=} {num_kv_heads=}."
        )
    if scale is None:
        scale = jnp.asarray(1.0 / (query.shape[-1] ** 0.5), dtype=query.dtype)
    else:
        scale = jnp.asarray(scale, dtype=query.dtype)

    repeat = num_heads // num_kv_heads
    if repeat == 1:
        logits = (jnp.einsum("bhqd,bhkd->bhqk", query, key) * scale).astype(jnp.float32)
        if mask is not None:
            mask = mask.astype(logits.dtype)
            logits = logits + (1.0 - mask) * jnp.asarray(-1.0e9, dtype=logits.dtype)
        weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
        return jnp.einsum("bhqk,bhkd->bhqd", weights, value)

    if _ATTENTION_EXPORT_LOWERING == "split_gqa":
        outputs = []
        for kv_idx in range(num_kv_heads):
            group_start = kv_idx * repeat
            q_group = query[:, group_start : group_start + repeat, :, :]
            k_head = key[:, kv_idx, :, :]
            v_head = value[:, kv_idx, :, :]
            logits = (jnp.einsum("bhqd,bkd->bhqk", q_group, k_head) * scale).astype(
                jnp.float32
            )
            if mask is not None:
                mask = mask.astype(logits.dtype)
                logits = logits + (1.0 - mask) * jnp.asarray(-1.0e9, dtype=logits.dtype)
            weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
            outputs.append(jnp.einsum("bhqk,bkd->bhqd", weights, v_head))
        return jnp.concatenate(outputs, axis=1)

    key = jnp.repeat(key, repeat, axis=1)
    value = jnp.repeat(value, repeat, axis=1)
    logits = (jnp.einsum("bhqd,bhkd->bhqk", query, key) * scale).astype(jnp.float32)
    if mask is not None:
        mask = mask.astype(logits.dtype)
        logits = logits + (1.0 - mask) * jnp.asarray(-1.0e9, dtype=logits.dtype)
    weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, value)


class _CachedTemporalAttention(nn.Module):
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        mask: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False, dtype=self.dtype)(x)
        k = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype)(x)
        v = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype)(x)

        if _ATTENTION_EXPORT_LAYOUT == "bnsh":
            q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
            k = rearrange(k, "b t (h d) -> b h t d", h=self.num_kv_heads)
            v = rearrange(v, "b t (h d) -> b h t d", h=self.num_kv_heads)
        else:
            q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
            k = rearrange(k, "b t (h d) -> b t h d", h=self.num_kv_heads)
            v = rearrange(v, "b t (h d) -> b t h d", h=self.num_kv_heads)

        q = _ExportRMSNorm(dtype=self.dtype, name="RMSNorm_0")(q)
        k = _ExportRMSNorm(dtype=self.dtype, name="RMSNorm_1")(k)

        if _ATTENTION_EXPORT_LAYOUT == "bnsh":
            q = _apply_rotary_embedding_bnsh(q, rope_emb[0], rope_emb[1])
            k = _apply_rotary_embedding_bnsh(k, rope_emb[0], rope_emb[1])
        else:
            q = apply_rotary_embedding(q, rope_emb[0], rope_emb[1])
            k = apply_rotary_embedding(k, rope_emb[0], rope_emb[1])

        if _ATTENTION_EXPORT_LAYOUT == "bnsh":
            out = _attention_for_export_bnsh(
                q,
                k,
                v,
                mask=mask,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            out = rearrange(out, "b h t d -> b t (h d)")
            k_out = rearrange(k, "b h t d -> b t h d")
            v_out = rearrange(v, "b h t d -> b t h d")
        else:
            out = _attention_for_export(
                q,
                k,
                v,
                mask=mask,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            out = rearrange(out, "b t h d -> b t (h d)")
            k_out = k
            v_out = v
        out = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype, name="Dense_3")(out)
        return out, k_out, v_out


class _CachedTemporalStepAttention(nn.Module):
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    context_length: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        cache_length: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False, dtype=self.dtype)(x)
        k = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype)(x)
        v = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype)(x)

        if _ATTENTION_EXPORT_LAYOUT == "bnsh":
            q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
            k = rearrange(k, "b t (h d) -> b h t d", h=self.num_kv_heads)
            v = rearrange(v, "b t (h d) -> b h t d", h=self.num_kv_heads)
        else:
            q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
            k = rearrange(k, "b t (h d) -> b t h d", h=self.num_kv_heads)
            v = rearrange(v, "b t (h d) -> b t h d", h=self.num_kv_heads)

        q = _ExportRMSNorm(dtype=self.dtype, name="RMSNorm_0")(q)
        k = _ExportRMSNorm(dtype=self.dtype, name="RMSNorm_1")(k)

        if _ATTENTION_EXPORT_LAYOUT == "bnsh":
            q = _apply_rotary_embedding_bnsh(q, rope_emb[0], rope_emb[1])
            k = _apply_rotary_embedding_bnsh(k, rope_emb[0], rope_emb[1])
            keys = jnp.concatenate(
                [rearrange(k_cache, "b t h d -> b h t d").astype(k.dtype), k],
                axis=2,
            )
            values = jnp.concatenate(
                [rearrange(v_cache, "b t h d -> b h t d").astype(v.dtype), v],
                axis=2,
            )
        else:
            q = apply_rotary_embedding(q, rope_emb[0], rope_emb[1])
            k = apply_rotary_embedding(k, rope_emb[0], rope_emb[1])
            keys = jnp.concatenate([k_cache.astype(k.dtype), k], axis=1)
            values = jnp.concatenate([v_cache.astype(v.dtype), v], axis=1)
        cache_positions = jnp.arange(self.context_length + 1, dtype=jnp.int32)
        current_position = jnp.asarray(self.context_length, dtype=jnp.int32)
        valid_cache = cache_positions < cache_length[0]
        valid_current = cache_positions == current_position
        valid = jnp.logical_or(valid_cache, valid_current)
        mask = valid[None, None, None, :]

        if _ATTENTION_EXPORT_LAYOUT == "bnsh":
            out = _attention_for_export_bnsh(
                q,
                keys,
                values,
                mask=mask,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            out = rearrange(out, "b h t d -> b t (h d)")
            k_out = rearrange(k, "b h t d -> b t h d")
            v_out = rearrange(v, "b h t d -> b t h d")
        else:
            out = _attention_for_export(
                q,
                keys,
                values,
                mask=mask,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            out = rearrange(out, "b t h d -> b t (h d)")
            k_out = k
            v_out = v
        out = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype, name="Dense_3")(out)
        return out, k_out, v_out


class _CachedPrefillTransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    mlp_hidden_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        residual = x
        x = _ExportRMSNorm(dtype=self.dtype, name="RMSNorm_0")(x)
        attn_out, k, v = _CachedTemporalAttention(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            name="Attention_0",
        )(x, rope_emb, mask)
        x = residual + attn_out

        residual = x
        x = _ExportRMSNorm(dtype=self.dtype, name="RMSNorm_1")(x)
        x = residual + SwiGLU(self.mlp_hidden_dim, dtype=self.dtype)(x)
        return x, k, v


class _CachedStepTransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    mlp_hidden_dim: int
    context_length: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        cache_length: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        residual = x
        x = _ExportRMSNorm(dtype=self.dtype, name="RMSNorm_0")(x)
        attn_out, k, v = _CachedTemporalStepAttention(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            context_length=self.context_length,
            dtype=self.dtype,
            name="Attention_0",
        )(x, rope_emb, k_cache, v_cache, cache_length)
        x = residual + attn_out

        residual = x
        x = _ExportRMSNorm(dtype=self.dtype, name="RMSNorm_1")(x)
        x = residual + SwiGLU(self.mlp_hidden_dim, dtype=self.dtype)(x)
        return x, k, v


class _CachedSpatioTemporalTransformer(nn.Module):
    num_layers: int
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    mlp_hidden_dim: int
    context_length: int
    base: float
    temporal_layer_period: int = 4
    dtype: jnp.dtype = jnp.bfloat16
    cache_update: str = "fill"

    @nn.compact
    def prefill(
        self,
        x: jnp.ndarray,
        t: int,
        total_tokens: int,
        spatial_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        spatial_mask: jnp.ndarray,
        temporal_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        temporal_mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        batch_size = x.shape[0]
        temporal_mask = jnp.repeat(temporal_mask, total_tokens, axis=0)[:, None, :, :]
        block_idx = 0
        cache_ks = []
        cache_vs = []
        num_groups = self.num_layers // self.temporal_layer_period
        for _ in range(num_groups):
            x = rearrange(x, "b t n d -> (b t) n d")
            for _ in range(self.temporal_layer_period - 1):
                x = _ExportTransformerBlock(
                    model_dim=self.model_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    mlp_hidden_dim=self.mlp_hidden_dim,
                    dtype=self.dtype,
                    name=f"TransformerBlock_{block_idx}",
                )(x, spatial_rope_emb, spatial_mask)
                block_idx += 1
            x = rearrange(x, "(b t) n d -> b t n d", b=batch_size, t=t)

            x = rearrange(x, "b t n d -> (b n) t d")
            x, k, v = _CachedPrefillTransformerBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                dtype=self.dtype,
                name=f"TransformerBlock_{block_idx}",
            )(x, temporal_rope_emb, temporal_mask)
            cache_ks.append(rearrange(k, "(b n) t h d -> b n t h d", b=batch_size, n=total_tokens))
            cache_vs.append(rearrange(v, "(b n) t h d -> b n t h d", b=batch_size, n=total_tokens))
            block_idx += 1
            x = rearrange(x, "(b n) t d -> b t n d", b=batch_size, n=total_tokens)

        return x, jnp.stack(cache_ks, axis=0), jnp.stack(cache_vs, axis=0)

    @nn.compact
    def step(
        self,
        x: jnp.ndarray,
        total_tokens: int,
        spatial_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        spatial_mask: jnp.ndarray,
        temporal_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        cache_length: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        batch_size = x.shape[0]
        block_idx = 0
        temporal_idx = 0
        candidate_ks = []
        candidate_vs = []
        num_groups = self.num_layers // self.temporal_layer_period
        for _ in range(num_groups):
            x = rearrange(x, "b t n d -> (b t) n d")
            for _ in range(self.temporal_layer_period - 1):
                x = _ExportTransformerBlock(
                    model_dim=self.model_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    mlp_hidden_dim=self.mlp_hidden_dim,
                    dtype=self.dtype,
                    name=f"TransformerBlock_{block_idx}",
                )(x, spatial_rope_emb, spatial_mask)
                block_idx += 1
            x = rearrange(x, "(b t) n d -> b t n d", b=batch_size, t=1)

            x = rearrange(x, "b t n d -> (b n) t d")
            block_k_cache = rearrange(
                k_cache[temporal_idx],
                "b n t h d -> (b n) t h d",
            )
            block_v_cache = rearrange(
                v_cache[temporal_idx],
                "b n t h d -> (b n) t h d",
            )
            x, k, v = _CachedStepTransformerBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                context_length=self.context_length,
                dtype=self.dtype,
                name=f"TransformerBlock_{block_idx}",
            )(x, temporal_rope_emb, block_k_cache, block_v_cache, cache_length)
            k = rearrange(k, "(b n) one h d -> b n one h d", b=batch_size, n=total_tokens)
            v = rearrange(v, "(b n) one h d -> b n one h d", b=batch_size, n=total_tokens)
            if self.cache_update == "slide":
                candidate_ks.append(
                    jnp.concatenate(
                        [
                            _slide_rebased_key_cache(
                                k_cache[temporal_idx],
                                base=float(self.base),
                                head_dim=self.head_dim,
                            ),
                            k,
                        ],
                        axis=2,
                    )
                )
            else:
                candidate_ks.append(
                    _append_cache_entry(
                        k_cache[temporal_idx],
                        k,
                        cache_length,
                        update=self.cache_update,
                    )
                )
            candidate_vs.append(
                _append_cache_entry(
                    v_cache[temporal_idx],
                    v,
                    cache_length,
                    update=self.cache_update,
                )
            )
            block_idx += 1
            temporal_idx += 1
            x = rearrange(x, "(b n) t d -> b t n d", b=batch_size, n=total_tokens)

        candidate_cache_length = jnp.minimum(cache_length + 1, self.context_length).astype(
            jnp.int32
        )
        return (
            x,
            jnp.stack(candidate_ks, axis=0),
            jnp.stack(candidate_vs, axis=0),
            candidate_cache_length,
        )

    @nn.compact
    def step_layer_cache(
        self,
        x: jnp.ndarray,
        total_tokens: int,
        spatial_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        spatial_mask: jnp.ndarray,
        temporal_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        k_caches: tuple[jnp.ndarray, ...],
        v_caches: tuple[jnp.ndarray, ...],
        cache_length: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], jnp.ndarray]:
        batch_size = x.shape[0]
        block_idx = 0
        temporal_idx = 0
        candidate_ks = []
        candidate_vs = []
        num_groups = self.num_layers // self.temporal_layer_period
        for _ in range(num_groups):
            x = rearrange(x, "b t n d -> (b t) n d")
            for _ in range(self.temporal_layer_period - 1):
                x = _ExportTransformerBlock(
                    model_dim=self.model_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    mlp_hidden_dim=self.mlp_hidden_dim,
                    dtype=self.dtype,
                    name=f"TransformerBlock_{block_idx}",
                )(x, spatial_rope_emb, spatial_mask)
                block_idx += 1
            x = rearrange(x, "(b t) n d -> b t n d", b=batch_size, t=1)

            x = rearrange(x, "b t n d -> (b n) t d")
            block_k_cache = rearrange(
                k_caches[temporal_idx],
                "b n t h d -> (b n) t h d",
            )
            block_v_cache = rearrange(
                v_caches[temporal_idx],
                "b n t h d -> (b n) t h d",
            )
            x, k, v = _CachedStepTransformerBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                context_length=self.context_length,
                dtype=self.dtype,
                name=f"TransformerBlock_{block_idx}",
            )(x, temporal_rope_emb, block_k_cache, block_v_cache, cache_length)
            k = rearrange(k, "(b n) one h d -> b n one h d", b=batch_size, n=total_tokens)
            v = rearrange(v, "(b n) one h d -> b n one h d", b=batch_size, n=total_tokens)
            if self.cache_update == "slide":
                candidate_ks.append(
                    jnp.concatenate(
                        [
                            _slide_rebased_key_cache(
                                k_caches[temporal_idx],
                                base=float(self.base),
                                head_dim=self.head_dim,
                            ),
                            k,
                        ],
                        axis=2,
                    )
                )
            else:
                candidate_ks.append(
                    _append_cache_entry(
                        k_caches[temporal_idx],
                        k,
                        cache_length,
                        update=self.cache_update,
                    )
                )
            candidate_vs.append(
                _append_cache_entry(
                    v_caches[temporal_idx],
                    v,
                    cache_length,
                    update=self.cache_update,
                )
            )
            block_idx += 1
            temporal_idx += 1
            x = rearrange(x, "(b n) t d -> b t n d", b=batch_size, n=total_tokens)

        candidate_cache_length = jnp.minimum(cache_length + 1, self.context_length).astype(
            jnp.int32
        )
        return x, tuple(candidate_ks), tuple(candidate_vs), candidate_cache_length

    @nn.compact
    def step_entries(
        self,
        x: jnp.ndarray,
        total_tokens: int,
        spatial_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        spatial_mask: jnp.ndarray,
        temporal_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        cache_length: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        batch_size = x.shape[0]
        block_idx = 0
        temporal_idx = 0
        entry_ks = []
        entry_vs = []
        num_groups = self.num_layers // self.temporal_layer_period
        for _ in range(num_groups):
            x = rearrange(x, "b t n d -> (b t) n d")
            for _ in range(self.temporal_layer_period - 1):
                x = _ExportTransformerBlock(
                    model_dim=self.model_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    mlp_hidden_dim=self.mlp_hidden_dim,
                    dtype=self.dtype,
                    name=f"TransformerBlock_{block_idx}",
                )(x, spatial_rope_emb, spatial_mask)
                block_idx += 1
            x = rearrange(x, "(b t) n d -> b t n d", b=batch_size, t=1)

            x = rearrange(x, "b t n d -> (b n) t d")
            block_k_cache = rearrange(
                k_cache[temporal_idx],
                "b n t h d -> (b n) t h d",
            )
            block_v_cache = rearrange(
                v_cache[temporal_idx],
                "b n t h d -> (b n) t h d",
            )
            x, k, v = _CachedStepTransformerBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                context_length=self.context_length,
                dtype=self.dtype,
                name=f"TransformerBlock_{block_idx}",
            )(x, temporal_rope_emb, block_k_cache, block_v_cache, cache_length)
            entry_ks.append(
                rearrange(k, "(b n) one h d -> b n one h d", b=batch_size, n=total_tokens)
            )
            entry_vs.append(
                rearrange(v, "(b n) one h d -> b n one h d", b=batch_size, n=total_tokens)
            )
            block_idx += 1
            temporal_idx += 1
            x = rearrange(x, "(b n) t d -> b t n d", b=batch_size, n=total_tokens)

        return x, jnp.stack(entry_ks, axis=0), jnp.stack(entry_vs, axis=0)

    @nn.compact
    def predict_step(
        self,
        x: jnp.ndarray,
        total_tokens: int,
        spatial_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        spatial_mask: jnp.ndarray,
        temporal_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        cache_length: jnp.ndarray,
    ) -> jnp.ndarray:
        batch_size = x.shape[0]
        block_idx = 0
        temporal_idx = 0
        num_groups = self.num_layers // self.temporal_layer_period
        for _ in range(num_groups):
            x = rearrange(x, "b t n d -> (b t) n d")
            for _ in range(self.temporal_layer_period - 1):
                x = _ExportTransformerBlock(
                    model_dim=self.model_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    mlp_hidden_dim=self.mlp_hidden_dim,
                    dtype=self.dtype,
                    name=f"TransformerBlock_{block_idx}",
                )(x, spatial_rope_emb, spatial_mask)
                block_idx += 1
            x = rearrange(x, "(b t) n d -> b t n d", b=batch_size, t=1)

            x = rearrange(x, "b t n d -> (b n) t d")
            block_k_cache = rearrange(
                k_cache[temporal_idx],
                "b n t h d -> (b n) t h d",
            )
            block_v_cache = rearrange(
                v_cache[temporal_idx],
                "b n t h d -> (b n) t h d",
            )
            x, _, _ = _CachedStepTransformerBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                context_length=self.context_length,
                dtype=self.dtype,
                name=f"TransformerBlock_{block_idx}",
            )(x, temporal_rope_emb, block_k_cache, block_v_cache, cache_length)
            block_idx += 1
            temporal_idx += 1
            x = rearrange(x, "(b n) t d -> b t n d", b=batch_size, n=total_tokens)

        return x

    @nn.compact
    def predict_step_layer_cache(
        self,
        x: jnp.ndarray,
        total_tokens: int,
        spatial_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        spatial_mask: jnp.ndarray,
        temporal_rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        k_caches: tuple[jnp.ndarray, ...],
        v_caches: tuple[jnp.ndarray, ...],
        cache_length: jnp.ndarray,
    ) -> jnp.ndarray:
        batch_size = x.shape[0]
        block_idx = 0
        temporal_idx = 0
        num_groups = self.num_layers // self.temporal_layer_period
        for _ in range(num_groups):
            x = rearrange(x, "b t n d -> (b t) n d")
            for _ in range(self.temporal_layer_period - 1):
                x = _ExportTransformerBlock(
                    model_dim=self.model_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    mlp_hidden_dim=self.mlp_hidden_dim,
                    dtype=self.dtype,
                    name=f"TransformerBlock_{block_idx}",
                )(x, spatial_rope_emb, spatial_mask)
                block_idx += 1
            x = rearrange(x, "(b t) n d -> b t n d", b=batch_size, t=1)

            x = rearrange(x, "b t n d -> (b n) t d")
            block_k_cache = rearrange(
                k_caches[temporal_idx],
                "b n t h d -> (b n) t h d",
            )
            block_v_cache = rearrange(
                v_caches[temporal_idx],
                "b n t h d -> (b n) t h d",
            )
            x, _, _ = _CachedStepTransformerBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                context_length=self.context_length,
                dtype=self.dtype,
                name=f"TransformerBlock_{block_idx}",
            )(x, temporal_rope_emb, block_k_cache, block_v_cache, cache_length)
            block_idx += 1
            temporal_idx += 1
            x = rearrange(x, "(b n) t d -> b t n d", b=batch_size, n=total_tokens)

        return x


def _append_cache_entry(
    cache: jnp.ndarray,
    entry: jnp.ndarray,
    cache_length: jnp.ndarray,
    *,
    update: str,
) -> jnp.ndarray:
    if update == "slide":
        return jnp.concatenate([cache[:, :, 1:], entry], axis=2)
    if update != "fill":
        raise ValueError(f"Unsupported cache update mode {update!r}.")
    write_index = jnp.minimum(cache_length[0], cache.shape[2] - 1)
    return jax.lax.dynamic_update_slice(cache, entry, (0, 0, write_index, 0, 0))


def _slide_rebased_key_cache(
    cache: jnp.ndarray,
    *,
    base: float,
    head_dim: int,
) -> jnp.ndarray:
    kept = cache[:, :, 1:]
    half_head_dim = head_dim // 2
    theta = 1 / (base ** (jnp.arange(half_head_dim, dtype=jnp.float32) / half_head_dim))
    cos = jnp.cos(theta).astype(cache.dtype)
    sin = jnp.sin(theta).astype(cache.dtype)
    left, right = jnp.split(kept, 2, axis=-1)
    cos = cos[None, None, None, None, :]
    sin = sin[None, None, None, None, :]
    return jnp.concatenate(
        [
            left * cos + right * sin,
            right * cos - left * sin,
        ],
        axis=-1,
    )


class _ExportActionEmbedding(nn.Module):
    model_dim: int
    num_actions: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        actions: jnp.ndarray,
        batch_time_shape: tuple[int, int],
    ) -> jnp.ndarray:
        batch_size, seq_len = batch_time_shape
        base_token = self.param(
            "base_token",
            nn.initializers.normal(stddev=0.02),
            (self.model_dim,),
        ).astype(self.dtype)
        action_tokens = nn.Embed(
            num_embeddings=self.num_actions,
            features=self.model_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="action_embedding",
        )(jnp.asarray(actions, dtype=jnp.int32))
        return action_tokens + jnp.broadcast_to(base_token, (batch_size, seq_len, self.model_dim))


class _CachedDynamicsModel(nn.Module):
    cfg: Any
    dtype: jnp.dtype = jnp.float32
    cache_update: str = "fill"

    def setup(self):
        self.shortcut_embedding = dynamics_module.ShortcutEmbedding(
            model_dim=int(self.cfg.model_dim),
            max_step_size=int(self.cfg.max_step_size),
            dtype=self.dtype,
        )
        self.action_embedding = _ExportActionEmbedding(
            model_dim=int(self.cfg.model_dim),
            num_actions=int(self.cfg.num_actions),
            dtype=self.dtype,
        )
        self.register_tokens = self.param(
            "register_tokens",
            nn.initializers.normal(stddev=0.02),
            (int(self.cfg.num_registers), int(self.cfg.model_dim)),
        )
        self.transformer = _CachedSpatioTemporalTransformer(
            num_layers=int(self.cfg.num_layers),
            model_dim=int(self.cfg.model_dim),
            num_heads=int(self.cfg.num_heads),
            num_kv_heads=int(self.cfg.num_kv_heads),
            head_dim=int(self.cfg.head_dim),
            mlp_hidden_dim=int(self.cfg.mlp_hidden_dim),
            context_length=int(self.cfg.context_length),
            base=float(self.cfg.base),
            temporal_layer_period=int(self.cfg.temporal_layer_period),
            dtype=self.dtype,
            cache_update=self.cache_update,
            name="transformer",
        )

    def _tokens(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray | None,
        step_levels: jnp.ndarray,
        signal_levels: jnp.ndarray,
    ) -> tuple[jnp.ndarray, int, int]:
        batch_size, seq_len, num_obs_tokens, _ = z.shape
        action_tokens = self.action_embedding(actions, (batch_size, seq_len))[:, :, None, :]
        shortcut_tokens = self.shortcut_embedding(step_levels, signal_levels)[:, :, None, :]
        register_tokens = jnp.broadcast_to(
            self.register_tokens.astype(self.dtype),
            (batch_size, seq_len, int(self.cfg.num_registers), int(self.cfg.model_dim)),
        )
        observation_tokens = nn.Dense(
            int(self.cfg.model_dim),
            dtype=self.dtype,
            name="Dense_0",
        )(z.astype(self.dtype))
        tokens = jnp.concatenate(
            [action_tokens, shortcut_tokens, register_tokens, observation_tokens], axis=2
        )
        total_tokens = 1 + 1 + int(self.cfg.num_registers) + num_obs_tokens
        observation_offset = 1 + 1 + int(self.cfg.num_registers)
        return tokens, total_tokens, observation_offset

    def _project_output(
        self, hidden: jnp.ndarray, observation_offset: int, token_dim: int
    ) -> jnp.ndarray:
        observation_hidden = hidden[:, :, observation_offset:, :]
        return nn.Dense(
            token_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="Dense_1",
        )(observation_hidden)

    @nn.compact
    def prefill(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray,
        step_levels: jnp.ndarray,
        signal_levels: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        _, seq_len, _, token_dim = z.shape
        tokens, total_tokens, observation_offset = self._tokens(
            z, actions, step_levels, signal_levels
        )
        spatial_rope = _export_create_temporal_rope(
            float(self.cfg.base), int(self.cfg.head_dim), total_tokens
        )
        temporal_rope = _export_create_temporal_rope(
            float(self.cfg.base), int(self.cfg.head_dim), seq_len
        )
        spatial_mask = jnp.ones((total_tokens, total_tokens), dtype=bool)
        query_positions = jnp.arange(seq_len)[:, None]
        key_positions = jnp.arange(seq_len)[None, :]
        temporal_mask = key_positions <= query_positions
        temporal_mask = temporal_mask & (
            key_positions >= query_positions - (int(self.cfg.context_length) - 1)
        )
        temporal_mask = jnp.broadcast_to(temporal_mask[None], (z.shape[0], seq_len, seq_len))
        hidden, k_cache, v_cache = self.transformer.prefill(
            tokens,
            seq_len,
            total_tokens,
            spatial_rope,
            spatial_mask,
            temporal_rope,
            temporal_mask,
        )
        pred_z = self._project_output(hidden, observation_offset, token_dim)
        cache_length = jnp.asarray([seq_len], dtype=jnp.int32)
        return pred_z, k_cache.astype(jnp.float32), v_cache.astype(jnp.float32), cache_length

    @nn.compact
    def prefill_layer_cache(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray,
        step_levels: jnp.ndarray,
        signal_levels: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], jnp.ndarray]:
        pred_z, k_cache, v_cache, cache_length = self.prefill(
            z,
            actions,
            step_levels,
            signal_levels,
        )
        num_groups = int(self.cfg.num_layers) // int(self.cfg.temporal_layer_period)
        k_caches = tuple(k_cache[i].astype(jnp.float32) for i in range(num_groups))
        v_caches = tuple(v_cache[i].astype(jnp.float32) for i in range(num_groups))
        return pred_z, k_caches, v_caches, cache_length

    @nn.compact
    def step(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray,
        step_levels: jnp.ndarray,
        signal_levels: jnp.ndarray,
        position_index: jnp.ndarray,
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        cache_length: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        _, _, _, token_dim = z.shape
        tokens, total_tokens, observation_offset = self._tokens(
            z, actions, step_levels, signal_levels
        )
        spatial_rope = _export_create_temporal_rope(
            float(self.cfg.base), int(self.cfg.head_dim), total_tokens
        )
        full_temporal_rope = _export_create_temporal_rope(
            float(self.cfg.base),
            int(self.cfg.head_dim),
            int(self.cfg.context_length) + 1,
        )
        pos = jnp.minimum(position_index[0], int(self.cfg.context_length))
        temporal_rope = (
            jnp.take(full_temporal_rope[0], pos, axis=0)[None],
            jnp.take(full_temporal_rope[1], pos, axis=0)[None],
        )
        spatial_mask = jnp.ones((total_tokens, total_tokens), dtype=bool)
        hidden, candidate_k, candidate_v, candidate_cache_length = self.transformer.step(
            tokens,
            total_tokens,
            spatial_rope,
            spatial_mask,
            temporal_rope,
            k_cache,
            v_cache,
            cache_length,
        )
        pred_z = self._project_output(hidden, observation_offset, token_dim)
        return (
            pred_z,
            candidate_k.astype(jnp.float32),
            candidate_v.astype(jnp.float32),
            candidate_cache_length,
        )

    @nn.compact
    def step_entries(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray,
        step_levels: jnp.ndarray,
        signal_levels: jnp.ndarray,
        position_index: jnp.ndarray,
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        cache_length: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        _, _, _, token_dim = z.shape
        tokens, total_tokens, observation_offset = self._tokens(
            z, actions, step_levels, signal_levels
        )
        spatial_rope = _export_create_temporal_rope(
            float(self.cfg.base), int(self.cfg.head_dim), total_tokens
        )
        full_temporal_rope = _export_create_temporal_rope(
            float(self.cfg.base),
            int(self.cfg.head_dim),
            int(self.cfg.context_length) + 1,
        )
        pos = jnp.minimum(position_index[0], int(self.cfg.context_length))
        temporal_rope = (
            jnp.take(full_temporal_rope[0], pos, axis=0)[None],
            jnp.take(full_temporal_rope[1], pos, axis=0)[None],
        )
        spatial_mask = jnp.ones((total_tokens, total_tokens), dtype=bool)
        hidden, entry_k, entry_v = self.transformer.step_entries(
            tokens,
            total_tokens,
            spatial_rope,
            spatial_mask,
            temporal_rope,
            k_cache,
            v_cache,
            cache_length,
        )
        pred_z = self._project_output(hidden, observation_offset, token_dim)
        return pred_z, entry_k.astype(jnp.float32), entry_v.astype(jnp.float32)

    @nn.compact
    def step_layer_cache(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray,
        step_levels: jnp.ndarray,
        signal_levels: jnp.ndarray,
        position_index: jnp.ndarray,
        k_caches: tuple[jnp.ndarray, ...],
        v_caches: tuple[jnp.ndarray, ...],
        cache_length: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], jnp.ndarray]:
        tokens, total_tokens, observation_offset = self._tokens(
            z, actions, step_levels, signal_levels
        )
        token_dim = z.shape[-1]
        spatial_rope = _export_create_temporal_rope(
            float(self.cfg.base), int(self.cfg.head_dim), total_tokens
        )
        full_temporal_rope = _export_create_temporal_rope(
            float(self.cfg.base),
            int(self.cfg.head_dim),
            int(self.cfg.context_length) + 1,
        )
        pos = jnp.minimum(position_index[0], int(self.cfg.context_length))
        temporal_rope = (
            jnp.take(full_temporal_rope[0], pos, axis=0)[None],
            jnp.take(full_temporal_rope[1], pos, axis=0)[None],
        )
        spatial_mask = jnp.ones((total_tokens, total_tokens), dtype=bool)
        hidden, candidate_ks, candidate_vs, candidate_cache_length = (
            self.transformer.step_layer_cache(
                tokens,
                total_tokens,
                spatial_rope,
                spatial_mask,
                temporal_rope,
                k_caches,
                v_caches,
                cache_length,
            )
        )
        pred_z = self._project_output(hidden, observation_offset, token_dim)
        candidate_ks = tuple(cache.astype(jnp.float32) for cache in candidate_ks)
        candidate_vs = tuple(cache.astype(jnp.float32) for cache in candidate_vs)
        return pred_z, candidate_ks, candidate_vs, candidate_cache_length

    @nn.compact
    def predict_step(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray,
        step_levels: jnp.ndarray,
        signal_levels: jnp.ndarray,
        position_index: jnp.ndarray,
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        cache_length: jnp.ndarray,
    ) -> jnp.ndarray:
        _, _, _, token_dim = z.shape
        tokens, total_tokens, observation_offset = self._tokens(
            z, actions, step_levels, signal_levels
        )
        spatial_rope = _export_create_temporal_rope(
            float(self.cfg.base), int(self.cfg.head_dim), total_tokens
        )
        full_temporal_rope = _export_create_temporal_rope(
            float(self.cfg.base),
            int(self.cfg.head_dim),
            int(self.cfg.context_length) + 1,
        )
        pos = jnp.minimum(position_index[0], int(self.cfg.context_length))
        temporal_rope = (
            jnp.take(full_temporal_rope[0], pos, axis=0)[None],
            jnp.take(full_temporal_rope[1], pos, axis=0)[None],
        )
        spatial_mask = jnp.ones((total_tokens, total_tokens), dtype=bool)
        hidden = self.transformer.predict_step(
            tokens,
            total_tokens,
            spatial_rope,
            spatial_mask,
            temporal_rope,
            k_cache,
            v_cache,
            cache_length,
        )
        return self._project_output(hidden, observation_offset, token_dim)

    @nn.compact
    def predict_step_layer_cache(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray,
        step_levels: jnp.ndarray,
        signal_levels: jnp.ndarray,
        position_index: jnp.ndarray,
        k_caches: tuple[jnp.ndarray, ...],
        v_caches: tuple[jnp.ndarray, ...],
        cache_length: jnp.ndarray,
    ) -> jnp.ndarray:
        _, _, _, token_dim = z.shape
        tokens, total_tokens, observation_offset = self._tokens(
            z, actions, step_levels, signal_levels
        )
        spatial_rope = _export_create_temporal_rope(
            float(self.cfg.base), int(self.cfg.head_dim), total_tokens
        )
        full_temporal_rope = _export_create_temporal_rope(
            float(self.cfg.base),
            int(self.cfg.head_dim),
            int(self.cfg.context_length) + 1,
        )
        pos = jnp.minimum(position_index[0], int(self.cfg.context_length))
        temporal_rope = (
            jnp.take(full_temporal_rope[0], pos, axis=0)[None],
            jnp.take(full_temporal_rope[1], pos, axis=0)[None],
        )
        spatial_mask = jnp.ones((total_tokens, total_tokens), dtype=bool)
        hidden = self.transformer.predict_step_layer_cache(
            tokens,
            total_tokens,
            spatial_rope,
            spatial_mask,
            temporal_rope,
            k_caches,
            v_caches,
            cache_length,
        )
        return self._project_output(hidden, observation_offset, token_dim)

    @nn.compact
    def sample_step_layer_cache(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray,
        position_index: jnp.ndarray,
        k_caches: tuple[jnp.ndarray, ...],
        v_caches: tuple[jnp.ndarray, ...],
        cache_length: jnp.ndarray,
        *,
        sample_steps: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        sample_step_level = validate_sample_steps(sample_steps)
        current_z = z.astype(jnp.float32)
        pred_z = current_z
        step_levels = jnp.full(z.shape[:2], sample_step_level, dtype=jnp.int32)

        for signal_level in range(sample_steps):
            signal_levels = jnp.full(z.shape[:2], signal_level, dtype=jnp.int32)
            pred_z = self.predict_step_layer_cache(
                current_z,
                actions,
                step_levels,
                signal_levels,
                position_index,
                k_caches,
                v_caches,
                cache_length,
            )
            current_z = flow_update_z(
                current_z,
                pred_z,
                signal_level=signal_level,
                sample_steps=sample_steps,
            )

        return current_z, pred_z

    @nn.compact
    def sample_step_predict_only(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray,
        position_index: jnp.ndarray,
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        cache_length: jnp.ndarray,
        *,
        sample_steps: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        sample_step_level = validate_sample_steps(sample_steps)
        current_z = z.astype(jnp.float32)
        pred_z = current_z
        step_levels = jnp.full(z.shape[:2], sample_step_level, dtype=jnp.int32)

        for signal_level in range(sample_steps):
            signal_levels = jnp.full(z.shape[:2], signal_level, dtype=jnp.int32)
            pred_z = self.predict_step(
                current_z,
                actions,
                step_levels,
                signal_levels,
                position_index,
                k_cache,
                v_cache,
                cache_length,
            )
            current_z = flow_update_z(
                current_z,
                pred_z,
                signal_level=signal_level,
                sample_steps=sample_steps,
            )

        return current_z, pred_z

    @nn.compact
    def sample_step(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray,
        position_index: jnp.ndarray,
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        cache_length: jnp.ndarray,
        *,
        sample_steps: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        sample_step_level = validate_sample_steps(sample_steps)
        current_z = z.astype(jnp.float32)
        candidate_k = k_cache
        candidate_v = v_cache
        candidate_cache_length = cache_length
        pred_z = current_z
        step_levels = jnp.full(z.shape[:2], sample_step_level, dtype=jnp.int32)

        for signal_level in range(sample_steps):
            signal_levels = jnp.full(z.shape[:2], signal_level, dtype=jnp.int32)
            if signal_level == sample_steps - 1:
                pred_z, candidate_k, candidate_v, candidate_cache_length = self.step(
                    current_z,
                    actions,
                    step_levels,
                    signal_levels,
                    position_index,
                    k_cache,
                    v_cache,
                    cache_length,
                )
            else:
                pred_z = self.predict_step(
                    current_z,
                    actions,
                    step_levels,
                    signal_levels,
                    position_index,
                    k_cache,
                    v_cache,
                    cache_length,
                )
            current_z = flow_update_z(
                current_z,
                pred_z,
                signal_level=signal_level,
                sample_steps=sample_steps,
            )

        return (
            current_z,
            pred_z,
            candidate_k.astype(jnp.float32),
            candidate_v.astype(jnp.float32),
            candidate_cache_length,
        )

    @nn.compact
    def sample_step_append_context(
        self,
        sample_noise: jnp.ndarray,
        context_noise: jnp.ndarray,
        actions: jnp.ndarray,
        position_index: jnp.ndarray,
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        cache_length: jnp.ndarray,
        *,
        context_tau: float,
        sample_steps: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        final_z, pred_z = self.sample_step_predict_only(
            sample_noise,
            actions,
            position_index,
            k_cache,
            v_cache,
            cache_length,
            sample_steps=sample_steps,
        )

        context_step_level = int(self.cfg.max_step_size) - 1
        context_step_count = 1 << context_step_level
        context_signal_level = min(
            max(int(round(float(context_tau) * context_step_count)), 0),
            context_step_count - 1,
        )
        context_tau_used = jnp.asarray(context_signal_level / context_step_count, dtype=jnp.float32)
        noised_context_z = context_tau_used * final_z.astype(jnp.float32) + (
            jnp.asarray(1.0, dtype=jnp.float32) - context_tau_used
        ) * context_noise.astype(jnp.float32)
        context_step_levels = jnp.full(
            final_z.shape[:2],
            context_step_level,
            dtype=jnp.int32,
        )
        context_signal_levels = jnp.full(
            final_z.shape[:2],
            context_signal_level,
            dtype=jnp.int32,
        )
        if self.cache_update == "slide":
            context_position_index = jnp.asarray(
                [int(self.cfg.context_length) - 1], dtype=jnp.int32
            )
        else:
            context_position_index = position_index
        _, context_k, context_v, context_cache_length = self.step(
            noised_context_z,
            actions,
            context_step_levels,
            context_signal_levels,
            context_position_index,
            k_cache,
            v_cache,
            cache_length,
        )
        return (
            final_z,
            pred_z,
            context_k.astype(jnp.float32),
            context_v.astype(jnp.float32),
            context_cache_length,
        )

    @nn.compact
    def sample_step_append_context_full_cache(
        self,
        sample_noise: jnp.ndarray,
        context_noise: jnp.ndarray,
        actions: jnp.ndarray,
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        *,
        context_tau: float,
        sample_steps: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if self.cache_update != "slide":
            raise ValueError("Full-cache append-context export requires slide cache updates.")
        cache_length = jnp.asarray([int(self.cfg.context_length)], dtype=jnp.int32)
        position_index = jnp.asarray([int(self.cfg.context_length)], dtype=jnp.int32)
        final_z, pred_z, candidate_k, candidate_v, _ = self.sample_step_append_context(
            sample_noise,
            context_noise,
            actions,
            position_index,
            k_cache,
            v_cache,
            cache_length,
            context_tau=context_tau,
            sample_steps=sample_steps,
        )
        return final_z, pred_z, candidate_k, candidate_v

    @nn.compact
    def sample_step_append_context_full_cache_entries(
        self,
        sample_noise: jnp.ndarray,
        context_noise: jnp.ndarray,
        actions: jnp.ndarray,
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        *,
        context_tau: float,
        sample_steps: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if self.cache_update != "slide":
            raise ValueError("Full-cache entry export requires slide cache updates.")
        cache_length = jnp.asarray([int(self.cfg.context_length)], dtype=jnp.int32)
        sample_position_index = jnp.asarray([int(self.cfg.context_length)], dtype=jnp.int32)
        final_z, pred_z = self.sample_step_predict_only(
            sample_noise,
            actions,
            sample_position_index,
            k_cache,
            v_cache,
            cache_length,
            sample_steps=sample_steps,
        )

        context_step_level = int(self.cfg.max_step_size) - 1
        context_step_count = 1 << context_step_level
        context_signal_level = min(
            max(int(round(float(context_tau) * context_step_count)), 0),
            context_step_count - 1,
        )
        context_tau_used = jnp.asarray(context_signal_level / context_step_count, dtype=jnp.float32)
        noised_context_z = context_tau_used * final_z.astype(jnp.float32) + (
            jnp.asarray(1.0, dtype=jnp.float32) - context_tau_used
        ) * context_noise.astype(jnp.float32)
        context_step_levels = jnp.full(
            final_z.shape[:2],
            context_step_level,
            dtype=jnp.int32,
        )
        context_signal_levels = jnp.full(
            final_z.shape[:2],
            context_signal_level,
            dtype=jnp.int32,
        )
        context_position_index = jnp.asarray([int(self.cfg.context_length) - 1], dtype=jnp.int32)
        _, entry_k, entry_v = self.step_entries(
            noised_context_z,
            actions,
            context_step_levels,
            context_signal_levels,
            context_position_index,
            k_cache,
            v_cache,
            cache_length,
        )
        return final_z, pred_z, entry_k.astype(jnp.float32), entry_v.astype(jnp.float32)

    @nn.compact
    def sample_step_append_context_cache_length_entries(
        self,
        sample_noise: jnp.ndarray,
        context_noise: jnp.ndarray,
        actions: jnp.ndarray,
        k_cache: jnp.ndarray,
        v_cache: jnp.ndarray,
        cache_length: jnp.ndarray,
        *,
        context_tau: float,
        sample_steps: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        context_length = int(self.cfg.context_length)
        max_sample_position = jnp.asarray([context_length], dtype=jnp.int32)
        max_context_position = jnp.asarray([context_length - 1], dtype=jnp.int32)
        sample_position_index = jnp.minimum(cache_length, max_sample_position)
        context_position_index = jnp.minimum(cache_length, max_context_position)
        final_z, pred_z = self.sample_step_predict_only(
            sample_noise,
            actions,
            sample_position_index,
            k_cache,
            v_cache,
            cache_length,
            sample_steps=sample_steps,
        )

        context_step_level = int(self.cfg.max_step_size) - 1
        context_step_count = 1 << context_step_level
        context_signal_level = min(
            max(int(round(float(context_tau) * context_step_count)), 0),
            context_step_count - 1,
        )
        context_tau_used = jnp.asarray(context_signal_level / context_step_count, dtype=jnp.float32)
        noised_context_z = context_tau_used * final_z.astype(jnp.float32) + (
            jnp.asarray(1.0, dtype=jnp.float32) - context_tau_used
        ) * context_noise.astype(jnp.float32)
        context_step_levels = jnp.full(
            final_z.shape[:2],
            context_step_level,
            dtype=jnp.int32,
        )
        context_signal_levels = jnp.full(
            final_z.shape[:2],
            context_signal_level,
            dtype=jnp.int32,
        )
        _, entry_k, entry_v = self.step_entries(
            noised_context_z,
            actions,
            context_step_levels,
            context_signal_levels,
            context_position_index,
            k_cache,
            v_cache,
            cache_length,
        )
        candidate_cache_length = jnp.minimum(
            cache_length + 1,
            jnp.asarray([context_length], dtype=jnp.int32),
        ).astype(jnp.int32)
        return (
            final_z,
            pred_z,
            entry_k.astype(jnp.float32),
            entry_v.astype(jnp.float32),
            candidate_cache_length,
        )

    @nn.compact
    def sample_step_append_context_layer_cache(
        self,
        sample_noise: jnp.ndarray,
        context_noise: jnp.ndarray,
        actions: jnp.ndarray,
        position_index: jnp.ndarray,
        k_caches: tuple[jnp.ndarray, ...],
        v_caches: tuple[jnp.ndarray, ...],
        cache_length: jnp.ndarray,
        *,
        context_tau: float,
        sample_steps: int,
    ) -> tuple[
        jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], jnp.ndarray
    ]:
        final_z, pred_z = self.sample_step_layer_cache(
            sample_noise,
            actions,
            position_index,
            k_caches,
            v_caches,
            cache_length,
            sample_steps=sample_steps,
        )

        context_step_level = int(self.cfg.max_step_size) - 1
        context_step_count = 1 << context_step_level
        context_signal_level = min(
            max(int(round(float(context_tau) * context_step_count)), 0),
            context_step_count - 1,
        )
        context_tau_used = jnp.asarray(context_signal_level / context_step_count, dtype=jnp.float32)
        noised_context_z = context_tau_used * final_z.astype(jnp.float32) + (
            jnp.asarray(1.0, dtype=jnp.float32) - context_tau_used
        ) * context_noise.astype(jnp.float32)
        context_step_levels = jnp.full(
            final_z.shape[:2],
            context_step_level,
            dtype=jnp.int32,
        )
        context_signal_levels = jnp.full(
            final_z.shape[:2],
            context_signal_level,
            dtype=jnp.int32,
        )
        if self.cache_update == "slide":
            context_position_index = jnp.asarray(
                [int(self.cfg.context_length) - 1], dtype=jnp.int32
            )
        else:
            context_position_index = position_index
        _, context_ks, context_vs, context_cache_length = self.step_layer_cache(
            noised_context_z,
            actions,
            context_step_levels,
            context_signal_levels,
            context_position_index,
            k_caches,
            v_caches,
            cache_length,
        )
        return final_z, pred_z, context_ks, context_vs, context_cache_length

@contextmanager
def export_overrides():
    original = jax.nn.dot_product_attention
    original_transformer_spatiotemporal = transformer_module.SpatioTemporalTransformer
    original_transformer_temporal_rope = transformer_module.create_temporal_rope
    original_transformer_spatial_rope = transformer_module.create_spatial_rope
    original_tokenizer_spatiotemporal = tokenizer_module.SpatioTemporalTransformer
    original_tokenizer_temporal_rope = tokenizer_module.create_temporal_rope
    original_tokenizer_spatial_rope = tokenizer_module.create_spatial_rope
    original_tokenizer_temporal_mask = tokenizer_module.create_temporal_mask
    original_dynamics_spatiotemporal = dynamics_module.SpatioTemporalTransformer
    original_dynamics_temporal_rope = dynamics_module.create_temporal_rope
    if _ATTENTION_EXPORT_LOWERING in {"manual", "split_gqa"}:
        jax.nn.dot_product_attention = _export_dot_product_attention
    transformer_module.SpatioTemporalTransformer = _ExportSpatioTemporalTransformer
    tokenizer_module.SpatioTemporalTransformer = _ExportSpatioTemporalTransformer
    dynamics_module.SpatioTemporalTransformer = _ExportSpatioTemporalTransformer
    transformer_module.create_temporal_rope = _export_create_temporal_rope
    transformer_module.create_spatial_rope = _export_create_spatial_rope
    tokenizer_module.create_temporal_rope = _export_create_temporal_rope
    tokenizer_module.create_spatial_rope = _export_create_spatial_rope
    tokenizer_module.create_temporal_mask = _export_create_temporal_mask
    dynamics_module.create_temporal_rope = _export_create_temporal_rope
    try:
        yield
    finally:
        jax.nn.dot_product_attention = original
        transformer_module.SpatioTemporalTransformer = original_transformer_spatiotemporal
        transformer_module.create_temporal_rope = original_transformer_temporal_rope
        transformer_module.create_spatial_rope = original_transformer_spatial_rope
        tokenizer_module.SpatioTemporalTransformer = original_tokenizer_spatiotemporal
        tokenizer_module.create_temporal_rope = original_tokenizer_temporal_rope
        tokenizer_module.create_spatial_rope = original_tokenizer_spatial_rope
        tokenizer_module.create_temporal_mask = original_tokenizer_temporal_mask
        dynamics_module.SpatioTemporalTransformer = original_dynamics_spatiotemporal
        dynamics_module.create_temporal_rope = original_dynamics_temporal_rope


def onnx_apply_tokenizer_decoder(
    variables: Any,
    cfg: DictConfig,
    latent: jnp.ndarray,
    *,
    dtype: Any | None = jnp.float32,
) -> jnp.ndarray:
    model = create_tokenizer(cfg, dtype=dtype)
    with export_overrides():
        return model.apply(variables, latent, method=Tokenizer.decode)


def onnx_apply_tokenizer_decode_z(
    variables: Any,
    cfg: DictConfig,
    z: jnp.ndarray,
    *,
    num_obs_tokens: int | None = None,
    dtype: Any | None = jnp.float32,
) -> jnp.ndarray:
    model = create_tokenizer(cfg, dtype=dtype)
    if len(z.shape) != 4:
        raise ValueError(f"decode_z expects rank-4 input, got shape={tuple(z.shape)}.")

    _, _, token_count, token_dim = z.shape
    channel_dim = int(cfg.channel_dim)
    with export_overrides():
        if token_count == int(cfg.num_latents) and token_dim == channel_dim:
            return model.apply(variables, z, method=Tokenizer.decode)

        if num_obs_tokens is not None and token_count != int(num_obs_tokens):
            raise ValueError(
                "decode_z dynamics token count mismatch: "
                f"got shape={tuple(z.shape)}, num_obs_tokens={num_obs_tokens}."
            )
        if token_dim % channel_dim != 0:
            raise ValueError(
                "decode_z cannot pack dynamics z into tokenizer latents: "
                f"got shape={tuple(z.shape)}, channel_dim={channel_dim}."
            )
        latents_per_obs = token_dim // channel_dim
        if token_count * latents_per_obs != int(cfg.num_latents):
            raise ValueError(
                "decode_z latent count mismatch: "
                f"got shape={tuple(z.shape)}, inferred_latents={token_count * latents_per_obs}, "
                f"expected_latents={int(cfg.num_latents)}, channel_dim={channel_dim}."
            )

        # Avoid ONNX Reshape in the browser hot path. For the demo shape this is
        # equivalent to einops "b t n (k d) -> b t (n k) d".
        latent = jnp.concatenate(jnp.split(z, latents_per_obs, axis=-1), axis=2)
        return model.apply(variables, latent, method=Tokenizer.decode)


def onnx_apply_dynamics_uncached(
    variables: Any,
    cfg: DictConfig,
    z: jnp.ndarray,
    actions: jnp.ndarray,
    step_levels: jnp.ndarray,
    signal_levels: jnp.ndarray,
    *,
    dtype: Any | None = jnp.float32,
) -> jnp.ndarray:
    model = create_dynamics(cfg, dtype=dtype)
    with export_overrides():
        return model.apply(
            variables,
            z,
            actions,
            step_levels,
            signal_levels,
            method=DynamicsModel.__call__,
        )


def onnx_apply_dynamics_cached_prefill(
    variables: Any,
    cfg: DictConfig,
    z: jnp.ndarray,
    actions: jnp.ndarray,
    step_levels: jnp.ndarray,
    signal_levels: jnp.ndarray,
    *,
    dtype: Any | None = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    model = _CachedDynamicsModel(
        cfg,
        dtype=dtype or jnp.float32,
    )
    with export_overrides():
        return model.apply(
            variables,
            z,
            actions,
            step_levels,
            signal_levels,
            method=_CachedDynamicsModel.prefill,
        )


def onnx_apply_dynamics_cached_prefill_layer_cache(
    variables: Any,
    cfg: DictConfig,
    z: jnp.ndarray,
    actions: jnp.ndarray,
    step_levels: jnp.ndarray,
    signal_levels: jnp.ndarray,
    *,
    dtype: Any | None = jnp.float32,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], jnp.ndarray]:
    model = _CachedDynamicsModel(cfg, dtype=dtype or jnp.float32)
    with export_overrides():
        return model.apply(
            variables,
            z,
            actions,
            step_levels,
            signal_levels,
            method=_CachedDynamicsModel.prefill_layer_cache,
        )


def onnx_apply_dynamics_cached_step(
    variables: Any,
    cfg: DictConfig,
    z: jnp.ndarray,
    actions: jnp.ndarray,
    step_levels: jnp.ndarray,
    signal_levels: jnp.ndarray,
    position_index: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    cache_length: jnp.ndarray,
    *,
    dtype: Any | None = jnp.float32,
    cache_update: str = "fill",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    model = _CachedDynamicsModel(
        cfg,
        dtype=dtype or jnp.float32,
        cache_update=cache_update,
    )
    with export_overrides():
        return model.apply(
            variables,
            z,
            actions,
            step_levels,
            signal_levels,
            position_index,
            k_cache,
            v_cache,
            cache_length,
            method=_CachedDynamicsModel.step,
        )


def onnx_apply_dynamics_cached_sample_step(
    variables: Any,
    cfg: DictConfig,
    z: jnp.ndarray,
    actions: jnp.ndarray,
    position_index: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    cache_length: jnp.ndarray,
    *,
    sample_steps: int,
    dtype: Any | None = jnp.float32,
    cache_update: str = "fill",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    model = _CachedDynamicsModel(
        cfg,
        dtype=dtype or jnp.float32,
        cache_update=cache_update,
    )
    with export_overrides():
        return model.apply(
            variables,
            z,
            actions,
            position_index,
            k_cache,
            v_cache,
            cache_length,
            sample_steps=sample_steps,
            method=_CachedDynamicsModel.sample_step,
        )


def onnx_apply_dynamics_cached_sample_step_append_context(
    variables: Any,
    cfg: DictConfig,
    sample_noise: jnp.ndarray,
    context_noise: jnp.ndarray,
    actions: jnp.ndarray,
    position_index: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    cache_length: jnp.ndarray,
    *,
    context_tau: float,
    sample_steps: int,
    dtype: Any | None = jnp.float32,
    cache_update: str = "fill",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    model = _CachedDynamicsModel(
        cfg,
        dtype=dtype or jnp.float32,
        cache_update=cache_update,
    )
    with export_overrides():
        return model.apply(
            variables,
            sample_noise,
            context_noise,
            actions,
            position_index,
            k_cache,
            v_cache,
            cache_length,
            context_tau=context_tau,
            sample_steps=sample_steps,
            method=_CachedDynamicsModel.sample_step_append_context,
        )


def onnx_apply_dynamics_cached_sample_step_append_context_layer_cache(
    variables: Any,
    cfg: DictConfig,
    sample_noise: jnp.ndarray,
    context_noise: jnp.ndarray,
    actions: jnp.ndarray,
    position_index: jnp.ndarray,
    k_caches: tuple[jnp.ndarray, ...],
    v_caches: tuple[jnp.ndarray, ...],
    cache_length: jnp.ndarray,
    *,
    context_tau: float,
    sample_steps: int,
    dtype: Any | None = jnp.float32,
    cache_update: str = "fill",
) -> tuple[jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], jnp.ndarray]:
    model = _CachedDynamicsModel(
        cfg,
        dtype=dtype or jnp.float32,
        cache_update=cache_update,
    )
    with export_overrides():
        return model.apply(
            variables,
            sample_noise,
            context_noise,
            actions,
            position_index,
            k_caches,
            v_caches,
            cache_length,
            context_tau=context_tau,
            sample_steps=sample_steps,
            method=_CachedDynamicsModel.sample_step_append_context_layer_cache,
        )


def onnx_apply_dynamics_cached_sample_step_append_context_full_cache(
    variables: Any,
    cfg: DictConfig,
    sample_noise: jnp.ndarray,
    context_noise: jnp.ndarray,
    actions: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    *,
    context_tau: float,
    sample_steps: int,
    dtype: Any | None = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    model = _CachedDynamicsModel(
        cfg,
        dtype=dtype or jnp.float32,
        cache_update="slide",
    )
    with export_overrides():
        return model.apply(
            variables,
            sample_noise,
            context_noise,
            actions,
            k_cache,
            v_cache,
            context_tau=context_tau,
            sample_steps=sample_steps,
            method=_CachedDynamicsModel.sample_step_append_context_full_cache,
        )


def onnx_apply_dynamics_cached_sample_step_append_context_full_cache_entries(
    variables: Any,
    cfg: DictConfig,
    sample_noise: jnp.ndarray,
    context_noise: jnp.ndarray,
    actions: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    *,
    context_tau: float,
    sample_steps: int,
    dtype: Any | None = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    model = _CachedDynamicsModel(
        cfg,
        dtype=dtype or jnp.float32,
        cache_update="slide",
    )
    with export_overrides():
        return model.apply(
            variables,
            sample_noise,
            context_noise,
            actions,
            k_cache,
            v_cache,
            context_tau=context_tau,
            sample_steps=sample_steps,
            method=_CachedDynamicsModel.sample_step_append_context_full_cache_entries,
        )


def onnx_apply_dynamics_cached_sample_step_append_context_cache_length_entries(
    variables: Any,
    cfg: DictConfig,
    sample_noise: jnp.ndarray,
    context_noise: jnp.ndarray,
    actions: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    cache_length: jnp.ndarray,
    *,
    context_tau: float,
    sample_steps: int,
    dtype: Any | None = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    model = _CachedDynamicsModel(
        cfg,
        dtype=dtype or jnp.float32,
        cache_update="fill",
    )
    with export_overrides():
        return model.apply(
            variables,
            sample_noise,
            context_noise,
            actions,
            k_cache,
            v_cache,
            cache_length,
            context_tau=context_tau,
            sample_steps=sample_steps,
            method=_CachedDynamicsModel.sample_step_append_context_cache_length_entries,
        )


# Compatibility aliases for older export scripts. New ONNX-only call sites should
# use the onnx_ names so they are easy to distinguish from core model methods.
apply_tokenizer_decoder = onnx_apply_tokenizer_decoder
apply_tokenizer_decode_z = onnx_apply_tokenizer_decode_z
apply_dynamics_uncached = onnx_apply_dynamics_uncached
apply_dynamics_cached_prefill = onnx_apply_dynamics_cached_prefill
apply_dynamics_cached_prefill_layer_cache = onnx_apply_dynamics_cached_prefill_layer_cache
apply_dynamics_cached_step = onnx_apply_dynamics_cached_step
apply_dynamics_cached_sample_step = onnx_apply_dynamics_cached_sample_step
apply_dynamics_cached_sample_step_append_context = (
    onnx_apply_dynamics_cached_sample_step_append_context
)
apply_dynamics_cached_sample_step_append_context_layer_cache = (
    onnx_apply_dynamics_cached_sample_step_append_context_layer_cache
)
apply_dynamics_cached_sample_step_append_context_full_cache = (
    onnx_apply_dynamics_cached_sample_step_append_context_full_cache
)
apply_dynamics_cached_sample_step_append_context_full_cache_entries = (
    onnx_apply_dynamics_cached_sample_step_append_context_full_cache_entries
)
apply_dynamics_cached_sample_step_append_context_cache_length_entries = (
    onnx_apply_dynamics_cached_sample_step_append_context_cache_length_entries
)
