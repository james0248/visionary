from contextlib import contextmanager
from dataclasses import dataclass
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
from visionary.transformer import SwiGLU, TransformerBlock, apply_rotary_embedding


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

        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_kv_heads)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_kv_heads)

        q = nn.RMSNorm(dtype=self.dtype)(q)
        k = nn.RMSNorm(dtype=self.dtype)(k)

        q = apply_rotary_embedding(q, rope_emb[0], rope_emb[1])
        k = apply_rotary_embedding(k, rope_emb[0], rope_emb[1])

        out = _export_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            scale=1.0 / jnp.sqrt(self.head_dim),
        )
        out = rearrange(out, "b t h d -> b t (h d)")
        out = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype)(out)
        return out, k, v


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

        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_kv_heads)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_kv_heads)

        q = nn.RMSNorm(dtype=self.dtype)(q)
        k = nn.RMSNorm(dtype=self.dtype)(k)

        q = apply_rotary_embedding(q, rope_emb[0], rope_emb[1])
        k = apply_rotary_embedding(k, rope_emb[0], rope_emb[1])

        keys = jnp.concatenate([k_cache.astype(k.dtype), k], axis=1)
        values = jnp.concatenate([v_cache.astype(v.dtype), v], axis=1)
        cache_positions = jnp.arange(self.context_length + 1, dtype=jnp.int32)
        valid = cache_positions < (cache_length[0] + 1)
        mask = valid[None, None, None, :]

        out = _export_dot_product_attention(
            q,
            keys,
            values,
            mask=mask,
            scale=1.0 / jnp.sqrt(self.head_dim),
        )
        out = rearrange(out, "b t h d -> b t (h d)")
        out = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype)(out)
        return out, k, v


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
        x = nn.RMSNorm(dtype=self.dtype)(x)
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
        x = nn.RMSNorm(dtype=self.dtype)(x)
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
        x = nn.RMSNorm(dtype=self.dtype)(x)
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
        x = nn.RMSNorm(dtype=self.dtype)(x)
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
    temporal_layer_period: int = 4
    dtype: jnp.dtype = jnp.bfloat16

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
                x = TransformerBlock(
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
                x = TransformerBlock(
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
            candidate_ks.append(_append_cache_entry(k_cache[temporal_idx], k, cache_length))
            candidate_vs.append(_append_cache_entry(v_cache[temporal_idx], v, cache_length))
            block_idx += 1
            temporal_idx += 1
            x = rearrange(x, "(b n) t d -> b t n d", b=batch_size, n=total_tokens)

        candidate_cache_length = jnp.minimum(cache_length + 1, self.context_length).astype(jnp.int32)
        return x, jnp.stack(candidate_ks, axis=0), jnp.stack(candidate_vs, axis=0), candidate_cache_length


def _append_cache_entry(
    cache: jnp.ndarray,
    entry: jnp.ndarray,
    cache_length: jnp.ndarray,
) -> jnp.ndarray:
    del cache_length
    return jnp.concatenate([cache[:, :, 1:], entry], axis=2)


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
            temporal_layer_period=int(self.cfg.temporal_layer_period),
            dtype=self.dtype,
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
        observation_tokens = nn.Dense(int(self.cfg.model_dim), dtype=self.dtype)(z.astype(self.dtype))
        tokens = jnp.concatenate(
            [action_tokens, shortcut_tokens, register_tokens, observation_tokens], axis=2
        )
        total_tokens = 1 + 1 + int(self.cfg.num_registers) + num_obs_tokens
        observation_offset = 1 + 1 + int(self.cfg.num_registers)
        return tokens, total_tokens, observation_offset

    def _project_output(self, hidden: jnp.ndarray, observation_offset: int, token_dim: int) -> jnp.ndarray:
        observation_hidden = hidden[:, :, observation_offset:, :]
        return nn.Dense(
            token_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
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
        tokens, total_tokens, observation_offset = self._tokens(z, actions, step_levels, signal_levels)
        spatial_rope = _export_create_temporal_rope(float(self.cfg.base), int(self.cfg.head_dim), total_tokens)
        temporal_rope = _export_create_temporal_rope(float(self.cfg.base), int(self.cfg.head_dim), seq_len)
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
        tokens, total_tokens, observation_offset = self._tokens(z, actions, step_levels, signal_levels)
        spatial_rope = _export_create_temporal_rope(float(self.cfg.base), int(self.cfg.head_dim), total_tokens)
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
        return pred_z, candidate_k.astype(jnp.float32), candidate_v.astype(jnp.float32), candidate_cache_length


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


def apply_dynamics_cached_prefill(
    variables: Any,
    cfg: DictConfig,
    z: jnp.ndarray,
    actions: jnp.ndarray,
    step_levels: jnp.ndarray,
    signal_levels: jnp.ndarray,
    *,
    dtype: Any | None = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    model = _CachedDynamicsModel(cfg, dtype=dtype or jnp.float32)
    with export_overrides():
        return model.apply(
            variables,
            z,
            actions,
            step_levels,
            signal_levels,
            method=_CachedDynamicsModel.prefill,
        )


def apply_dynamics_cached_step(
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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    model = _CachedDynamicsModel(cfg, dtype=dtype or jnp.float32)
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
