from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig

import visionary.dynamics as dynamics_module
import visionary.tokenizer as tokenizer_module
import visionary.transformer as transformer_module
from visionary.dynamics import DynamicsModel
from visionary.tokenizer import Tokenizer


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


def pack_tokenizer_latents_for_dynamics(
    latents: jnp.ndarray,
    *,
    num_obs_tokens: int,
) -> jnp.ndarray:
    return rearrange(latents, "b t (n k) d -> b t n (k d)", n=num_obs_tokens)


def unpack_dynamics_latents(
    z: jnp.ndarray,
    *,
    latent_dim: int,
) -> jnp.ndarray:
    return rearrange(z, "b t n (k d) -> b t (n k) d", d=latent_dim)


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

    num_heads = query.shape[-2]
    num_kv_heads = key.shape[-2]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads must be divisible by num_kv_heads, got {num_heads=} {num_kv_heads=}."
        )
    if num_heads != num_kv_heads:
        repeat = num_heads // num_kv_heads
        key = jnp.repeat(key, repeat, axis=-2)
        value = jnp.repeat(value, repeat, axis=-2)

    if scale is None:
        scale = jnp.asarray(1.0 / (query.shape[-1] ** 0.5), dtype=query.dtype)
    else:
        scale = jnp.asarray(scale, dtype=query.dtype)

    logits = jnp.einsum("bqhd,bkhd->bhqk", query, key) * scale
    if mask is not None:
        mask = mask.astype(logits.dtype)
        logits = logits + (1.0 - mask) * jnp.asarray(-1.0e9, dtype=logits.dtype)
    weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, value)


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


@contextmanager
def export_overrides():
    original = jax.nn.dot_product_attention
    original_transformer_temporal_rope = transformer_module.create_temporal_rope
    original_transformer_spatial_rope = transformer_module.create_spatial_rope
    original_tokenizer_temporal_rope = tokenizer_module.create_temporal_rope
    original_tokenizer_spatial_rope = tokenizer_module.create_spatial_rope
    original_tokenizer_temporal_mask = tokenizer_module.create_temporal_mask
    original_dynamics_temporal_rope = dynamics_module.create_temporal_rope
    jax.nn.dot_product_attention = _export_dot_product_attention
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
        transformer_module.create_temporal_rope = original_transformer_temporal_rope
        transformer_module.create_spatial_rope = original_transformer_spatial_rope
        tokenizer_module.create_temporal_rope = original_tokenizer_temporal_rope
        tokenizer_module.create_spatial_rope = original_tokenizer_spatial_rope
        tokenizer_module.create_temporal_mask = original_tokenizer_temporal_mask
        dynamics_module.create_temporal_rope = original_dynamics_temporal_rope


def apply_tokenizer_decoder(
    variables: Any,
    cfg: DictConfig,
    latent: jnp.ndarray,
    *,
    dtype: Any | None = jnp.float32,
) -> jnp.ndarray:
    model = create_tokenizer(cfg, dtype=dtype)
    with export_overrides():
        return model.apply(variables, latent, method=Tokenizer.decode)


def apply_dynamics_uncached(
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
