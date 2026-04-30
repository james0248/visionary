import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from visionary.dataset import DynamicsBatch
from visionary.transformer import (
    SpatioTemporalTransformer,
    SwiGLU,
    apply_rotary_embedding,
    create_temporal_rope,
    TransformerBlock,
)


class ActionEmbedding(nn.Module):
    model_dim: int
    num_actions: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        actions: jnp.ndarray | None,
        batch_time_shape: tuple[int, int],
    ) -> jnp.ndarray:
        batch_size, seq_len = batch_time_shape
        base_token = self.param(
            "base_token",
            nn.initializers.normal(stddev=0.02),
            (self.model_dim,),
        ).astype(self.dtype)

        if actions is None:
            return jnp.broadcast_to(base_token, (batch_size, seq_len, self.model_dim))

        actions = jnp.asarray(actions, dtype=jnp.int32)
        valid_actions = actions >= 0
        safe_actions = jnp.where(valid_actions, actions, 0)
        action_tokens = nn.Embed(
            num_embeddings=self.num_actions,
            features=self.model_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="action_embedding",
        )(safe_actions)
        action_tokens = jnp.where(valid_actions[..., None], action_tokens, 0)
        return action_tokens + base_token


class ShortcutEmbedding(nn.Module):
    model_dim: int
    max_step_size: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, step_levels: jnp.ndarray, signal_levels: jnp.ndarray) -> jnp.ndarray:
        step_dim = self.model_dim // 2
        signal_dim = self.model_dim - step_dim

        step_tokens = nn.Embed(
            num_embeddings=self.max_step_size,
            features=step_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="step_embedding",
        )(jnp.asarray(step_levels, dtype=jnp.int32))
        signal_tokens = nn.Embed(
            num_embeddings=1 << self.max_step_size,
            features=signal_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="signal_embedding",
        )(jnp.asarray(signal_levels, dtype=jnp.int32))

        return jnp.concatenate([step_tokens, signal_tokens], axis=-1)


def validate_sample_steps(sample_steps: int) -> int:
    sample_steps = int(sample_steps)
    if sample_steps <= 0 or sample_steps & (sample_steps - 1):
        raise ValueError(f"sample_steps must be a positive power of two, got {sample_steps}.")
    return int(round(math.log2(sample_steps)))


def flow_update_z(
    current_z: jnp.ndarray, pred_z: jnp.ndarray, signal_level: int, sample_steps: int
) -> jnp.ndarray:
    tau = jnp.asarray(signal_level / sample_steps, dtype=jnp.float32)
    step_size = jnp.asarray(1.0 / sample_steps, dtype=jnp.float32)
    denom = jnp.maximum(1.0 - tau, 1e-6)
    velocity = (pred_z.astype(jnp.float32) - current_z.astype(jnp.float32)) / denom
    return current_z.astype(jnp.float32) + velocity * step_size


class _InferenceRMSNorm(nn.Module):
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


class _InferenceAttention(nn.Module):
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
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

        q = _InferenceRMSNorm(dtype=self.dtype, name="RMSNorm_0")(q)
        k = _InferenceRMSNorm(dtype=self.dtype, name="RMSNorm_1")(k)

        if rope_emb is not None:
            q = apply_rotary_embedding(q, rope_emb[0], rope_emb[1])
            k = apply_rotary_embedding(k, rope_emb[0], rope_emb[1])

        out = jax.nn.dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            scale=1.0 / jnp.sqrt(self.head_dim),
        )
        out = rearrange(out, "b t h d -> b t (h d)")
        return nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype)(out)


class _InferenceTransformerBlock(nn.Module):
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
    ) -> jnp.ndarray:
        residual = x
        x = _InferenceRMSNorm(dtype=self.dtype, name="RMSNorm_0")(x)
        x = residual + _InferenceAttention(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            name="Attention_0",
        )(x, rope_emb, mask)

        residual = x
        x = _InferenceRMSNorm(dtype=self.dtype, name="RMSNorm_1")(x)
        return residual + SwiGLU(self.mlp_hidden_dim, dtype=self.dtype)(x)


class _CachedTemporalAttention(nn.Module):
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    context_length: int | None = None
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        mask: jnp.ndarray | None,
        k_cache: jnp.ndarray | None = None,
        v_cache: jnp.ndarray | None = None,
        cache_length: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False, dtype=self.dtype)(x)
        k = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype)(x)
        v = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype)(x)

        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_kv_heads)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_kv_heads)

        q = _InferenceRMSNorm(dtype=self.dtype, name="RMSNorm_0")(q)
        k = _InferenceRMSNorm(dtype=self.dtype, name="RMSNorm_1")(k)

        q = apply_rotary_embedding(q, rope_emb[0], rope_emb[1])
        k = apply_rotary_embedding(k, rope_emb[0], rope_emb[1])

        keys = k
        values = v
        if k_cache is not None and v_cache is not None and cache_length is not None:
            keys = jnp.concatenate([k_cache.astype(k.dtype), k], axis=1)
            values = jnp.concatenate([v_cache.astype(v.dtype), v], axis=1)
            cache_positions = jnp.arange(int(self.context_length) + 1, dtype=jnp.int32)
            valid = cache_positions < (cache_length[0] + 1)
            mask = valid[None, None, None, :]

        out = jax.nn.dot_product_attention(
            q,
            keys,
            values,
            mask=mask,
            scale=1.0 / jnp.sqrt(self.head_dim),
        )
        out = rearrange(out, "b t h d -> b t (h d)")
        out = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype)(out)
        return out, k, v


class _CachedTransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    mlp_hidden_dim: int
    context_length: int | None = None
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_emb: tuple[jnp.ndarray, jnp.ndarray],
        mask: jnp.ndarray | None,
        k_cache: jnp.ndarray | None = None,
        v_cache: jnp.ndarray | None = None,
        cache_length: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        residual = x
        x = _InferenceRMSNorm(dtype=self.dtype, name="RMSNorm_0")(x)
        attn_out, k, v = _CachedTemporalAttention(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            context_length=self.context_length,
            dtype=self.dtype,
            name="Attention_0",
        )(x, rope_emb, mask, k_cache, v_cache, cache_length)
        x = residual + attn_out

        residual = x
        x = _InferenceRMSNorm(dtype=self.dtype, name="RMSNorm_1")(x)
        x = residual + SwiGLU(self.mlp_hidden_dim, dtype=self.dtype)(x)
        return x, k, v


def _append_cache_entry(
    cache: jnp.ndarray,
    entry: jnp.ndarray,
    cache_length: jnp.ndarray,
) -> jnp.ndarray:
    write_index = jnp.minimum(cache_length[0], cache.shape[2] - 1)
    return jax.lax.dynamic_update_slice(cache, entry, (0, 0, write_index, 0, 0))


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
                x = _InferenceTransformerBlock(
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
            x, k, v = _CachedTransformerBlock(
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
                x = _InferenceTransformerBlock(
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
            block_k_cache = rearrange(k_cache[temporal_idx], "b n t h d -> (b n) t h d")
            block_v_cache = rearrange(v_cache[temporal_idx], "b n t h d -> (b n) t h d")
            x, k, v = _CachedTransformerBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                context_length=self.context_length,
                dtype=self.dtype,
                name=f"TransformerBlock_{block_idx}",
            )(x, temporal_rope_emb, None, block_k_cache, block_v_cache, cache_length)
            k = rearrange(k, "(b n) one h d -> b n one h d", b=batch_size, n=total_tokens)
            v = rearrange(v, "(b n) one h d -> b n one h d", b=batch_size, n=total_tokens)
            candidate_ks.append(_append_cache_entry(k_cache[temporal_idx], k, cache_length))
            candidate_vs.append(_append_cache_entry(v_cache[temporal_idx], v, cache_length))
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
            block_k_cache = rearrange(k_cache[temporal_idx], "b n t h d -> (b n) t h d")
            block_v_cache = rearrange(v_cache[temporal_idx], "b n t h d -> (b n) t h d")
            x, _, _ = _CachedTransformerBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                context_length=self.context_length,
                dtype=self.dtype,
                name=f"TransformerBlock_{block_idx}",
            )(x, temporal_rope_emb, None, block_k_cache, block_v_cache, cache_length)
            block_idx += 1
            temporal_idx += 1
            x = rearrange(x, "(b n) t d -> b t n d", b=batch_size, n=total_tokens)

        return x


class DynamicsModel(nn.Module):
    num_layers: int
    num_heads: int
    num_kv_heads: int
    num_registers: int
    num_obs_tokens: int
    num_actions: int

    max_step_size: int

    model_dim: int
    head_dim: int
    mlp_hidden_dim: int
    context_length: int
    temporal_layer_period: int = 4
    base: float = 10000.0
    attention_logit_soft_cap: float | None = 50.0
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.shortcut_embedding = ShortcutEmbedding(
            model_dim=self.model_dim,
            max_step_size=self.max_step_size,
            dtype=self.dtype,
        )
        self.action_embedding = ActionEmbedding(
            model_dim=self.model_dim,
            num_actions=self.num_actions,
            dtype=self.dtype,
        )
        self.register_tokens = self.param(
            "register_tokens",
            nn.initializers.normal(stddev=0.02),
            (self.num_registers, self.model_dim),
        )
        self.transformer = SpatioTemporalTransformer(
            num_layers=self.num_layers,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            mlp_hidden_dim=self.mlp_hidden_dim,
            attention_logit_soft_cap=self.attention_logit_soft_cap,
            temporal_layer_period=self.temporal_layer_period,
            dtype=self.dtype,
        )

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray | None,
        step_levels: jnp.ndarray,
        signal_levels: jnp.ndarray,
    ) -> jnp.ndarray:
        batch_size, seq_len, num_obs_tokens, token_dim = z.shape

        action_tokens = self.action_embedding(actions, (batch_size, seq_len))[:, :, None, :]
        shortcut_tokens = self.shortcut_embedding(step_levels, signal_levels)[:, :, None, :]
        register_tokens = jnp.broadcast_to(
            self.register_tokens.astype(self.dtype),
            (batch_size, seq_len, self.num_registers, self.model_dim),
        )
        observation_tokens = nn.Dense(
            self.model_dim,
            dtype=self.dtype,
            name="Dense_0",
        )(z.astype(self.dtype))

        num_tokens = 1 + 1 + self.num_registers + num_obs_tokens
        tokens = jnp.concatenate(
            [action_tokens, shortcut_tokens, register_tokens, observation_tokens], axis=2
        )

        spatial_rope = create_temporal_rope(self.base, self.head_dim, num_tokens)
        temporal_rope = create_temporal_rope(self.base, self.head_dim, seq_len)
        spatial_mask = jnp.ones((num_tokens, num_tokens), dtype=bool)

        query_positions = jnp.arange(seq_len)[:, None]
        key_positions = jnp.arange(seq_len)[None, :]
        temporal_mask = key_positions <= query_positions
        temporal_mask = temporal_mask & (
            key_positions >= query_positions - (self.context_length - 1)
        )
        temporal_mask = jnp.broadcast_to(
            temporal_mask[None, :, :],
            (batch_size, seq_len, seq_len),
        )

        hidden = self.transformer(
            x=tokens,
            t=seq_len,
            total_tokens=num_tokens,
            spatial_rope_emb=spatial_rope,
            spatial_mask=spatial_mask,
            temporal_rope_emb=temporal_rope,
            temporal_mask=temporal_mask,
        )
        observation_offset = 1 + 1 + self.num_registers
        observation_hidden = hidden[:, :, observation_offset:, :]
        return nn.Dense(
            token_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="Dense_1",
        )(observation_hidden)

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
            (batch_size, seq_len, self.num_registers, self.model_dim),
        )
        observation_tokens = nn.Dense(self.model_dim, dtype=self.dtype)(z.astype(self.dtype))
        tokens = jnp.concatenate(
            [action_tokens, shortcut_tokens, register_tokens, observation_tokens], axis=2
        )
        total_tokens = 1 + 1 + self.num_registers + num_obs_tokens
        observation_offset = 1 + 1 + self.num_registers
        return tokens, total_tokens, observation_offset

    def _project_observations(
        self,
        hidden: jnp.ndarray,
        observation_offset: int,
        token_dim: int,
    ) -> jnp.ndarray:
        observation_hidden = hidden[:, :, observation_offset:, :]
        return nn.Dense(
            token_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(observation_hidden)

    @nn.compact
    def cached_prefill(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray | None,
        step_levels: jnp.ndarray,
        signal_levels: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        _, seq_len, _, token_dim = z.shape
        tokens, total_tokens, observation_offset = self._tokens(
            z, actions, step_levels, signal_levels
        )
        spatial_rope = create_temporal_rope(self.base, self.head_dim, total_tokens)
        temporal_rope = create_temporal_rope(self.base, self.head_dim, seq_len)
        spatial_mask = jnp.ones((total_tokens, total_tokens), dtype=bool)

        query_positions = jnp.arange(seq_len)[:, None]
        key_positions = jnp.arange(seq_len)[None, :]
        temporal_mask = key_positions <= query_positions
        temporal_mask = temporal_mask & (
            key_positions >= query_positions - (self.context_length - 1)
        )
        temporal_mask = jnp.broadcast_to(temporal_mask[None], (z.shape[0], seq_len, seq_len))

        hidden, k_cache, v_cache = _CachedSpatioTemporalTransformer(
            num_layers=self.num_layers,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            mlp_hidden_dim=self.mlp_hidden_dim,
            context_length=self.context_length,
            temporal_layer_period=self.temporal_layer_period,
            dtype=self.dtype,
            name="cached_transformer",
        ).prefill(
            tokens,
            seq_len,
            total_tokens,
            spatial_rope,
            spatial_mask,
            temporal_rope,
            temporal_mask,
        )
        pred_z = self._project_observations(hidden, observation_offset, token_dim)
        cache_length = jnp.asarray([seq_len], dtype=jnp.int32)
        return pred_z, k_cache.astype(jnp.float32), v_cache.astype(jnp.float32), cache_length

    @nn.compact
    def cached_step(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray | None,
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
        spatial_rope = create_temporal_rope(self.base, self.head_dim, total_tokens)
        full_temporal_rope = create_temporal_rope(
            self.base,
            self.head_dim,
            self.context_length + 1,
        )
        pos = jnp.minimum(position_index[0], self.context_length)
        temporal_rope = (
            jnp.take(full_temporal_rope[0], pos, axis=0)[None],
            jnp.take(full_temporal_rope[1], pos, axis=0)[None],
        )
        spatial_mask = jnp.ones((total_tokens, total_tokens), dtype=bool)

        hidden, candidate_k, candidate_v, candidate_cache_length = _CachedSpatioTemporalTransformer(
            num_layers=self.num_layers,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            mlp_hidden_dim=self.mlp_hidden_dim,
            context_length=self.context_length,
            temporal_layer_period=self.temporal_layer_period,
            dtype=self.dtype,
            name="cached_transformer",
        ).step(
            tokens,
            total_tokens,
            spatial_rope,
            spatial_mask,
            temporal_rope,
            k_cache,
            v_cache,
            cache_length,
        )
        pred_z = self._project_observations(hidden, observation_offset, token_dim)
        return (
            pred_z,
            candidate_k.astype(jnp.float32),
            candidate_v.astype(jnp.float32),
            candidate_cache_length,
        )

    @nn.compact
    def cached_predict_step(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray | None,
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
        spatial_rope = create_temporal_rope(self.base, self.head_dim, total_tokens)
        full_temporal_rope = create_temporal_rope(
            self.base,
            self.head_dim,
            self.context_length + 1,
        )
        pos = jnp.minimum(position_index[0], self.context_length)
        temporal_rope = (
            jnp.take(full_temporal_rope[0], pos, axis=0)[None],
            jnp.take(full_temporal_rope[1], pos, axis=0)[None],
        )
        spatial_mask = jnp.ones((total_tokens, total_tokens), dtype=bool)

        hidden = _CachedSpatioTemporalTransformer(
            num_layers=self.num_layers,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            mlp_hidden_dim=self.mlp_hidden_dim,
            context_length=self.context_length,
            temporal_layer_period=self.temporal_layer_period,
            dtype=self.dtype,
            name="cached_transformer",
        ).predict_step(
            tokens,
            total_tokens,
            spatial_rope,
            spatial_mask,
            temporal_rope,
            k_cache,
            v_cache,
            cache_length,
        )
        return self._project_observations(hidden, observation_offset, token_dim)

    @nn.compact
    def cached_sample_step(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray | None,
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
                pred_z, candidate_k, candidate_v, candidate_cache_length = self.cached_step(
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
                pred_z = self.cached_predict_step(
                    current_z,
                    actions,
                    step_levels,
                    signal_levels,
                    position_index,
                    k_cache,
                    v_cache,
                    cache_length,
                )
            current_z = flow_update_z(current_z, pred_z, signal_level, sample_steps)

        return current_z, pred_z, candidate_k, candidate_v, candidate_cache_length

    def generate_next(
        self,
        video_prefix: jnp.ndarray,
        actions: jnp.ndarray | None,
        context_noise: jnp.ndarray,
        sample_noise: jnp.ndarray,
        target_index: jnp.ndarray,
        context_tau: float,
        sample_steps: int,
    ) -> jnp.ndarray:
        batch_size, seq_len, _, latent_dim = video_prefix.shape
        target_index = jnp.asarray(target_index, dtype=jnp.int32)

        context_step_level = self.max_step_size - 1
        context_step_count = 1 << context_step_level
        context_signal_level = min(
            max(int(round(context_tau * context_step_count)), 0),
            context_step_count - 1,
        )
        context_tau = jnp.float32(context_signal_level / context_step_count)

        sample_step_level = int(round(math.log2(sample_steps)))
        sample_step_count = 1 << sample_step_level
        sample_step_size = jnp.float32(1.0 / sample_step_count)

        z_prefix = rearrange(
            video_prefix,
            "b t (n k) d -> b t n (k d)",
            n=self.num_obs_tokens,
        )
        z_context_noise = rearrange(
            context_noise,
            "b t (n k) d -> b t n (k d)",
            n=self.num_obs_tokens,
        )
        z_sample_noise = rearrange(
            sample_noise[:, None],
            "b t (n k) d -> b t n (k d)",
            n=self.num_obs_tokens,
        )[:, 0]
        _, _, num_obs_tokens, token_dim = z_prefix.shape

        past_mask = jnp.arange(seq_len, dtype=jnp.int32) < target_index
        past_mask_z = past_mask[None, :, None, None]
        past_mask_t = jnp.broadcast_to(past_mask[None, :], (batch_size, seq_len))

        noised_prefix = context_tau * z_prefix.astype(jnp.float32) + (
            1.0 - context_tau
        ) * z_context_noise.astype(jnp.float32)
        base_z = jnp.where(
            past_mask_z,
            noised_prefix,
            jnp.zeros((batch_size, seq_len, num_obs_tokens, token_dim), dtype=jnp.float32),
        )
        base_step_levels = jnp.where(
            past_mask_t,
            jnp.full((batch_size, seq_len), context_step_level, dtype=jnp.int32),
            jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        )
        base_signal_levels = jnp.where(
            past_mask_t,
            jnp.full((batch_size, seq_len), context_signal_level, dtype=jnp.int32),
            jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        )

        current_z = z_sample_noise.astype(jnp.float32)
        for sample_signal_level in range(sample_step_count):
            step_levels = base_step_levels.at[:, target_index].set(sample_step_level)
            signal_levels = base_signal_levels.at[:, target_index].set(sample_signal_level)
            z_input = base_z.at[:, target_index].set(current_z)
            predicted = self(z_input, actions, step_levels, signal_levels)[:, target_index].astype(
                jnp.float32
            )

            tau = jnp.float32(sample_signal_level / sample_step_count)
            velocity = (predicted - current_z) / jnp.maximum(1.0 - tau, 1e-6)
            current_z = current_z + velocity * sample_step_size

        return rearrange(current_z[:, None], "b t n (k d) -> b t (n k) d", d=latent_dim)[:, 0]

    def generate_rollout(
        self,
        video_prefix: jnp.ndarray,
        actions: jnp.ndarray | None,
        context_noise: jnp.ndarray,
        sample_noise: jnp.ndarray,
        start_index: jnp.ndarray,
        context_tau: float,
        sample_steps: int,
    ) -> jnp.ndarray:
        start_index = jnp.asarray(start_index, dtype=jnp.int32)
        rollout_steps = sample_noise.shape[1]

        def body_fn(offset: int, current_video: jnp.ndarray) -> jnp.ndarray:
            target_index = start_index + offset
            next_frame = self.generate_next(
                current_video,
                actions,
                context_noise,
                sample_noise[:, offset],
                target_index,
                context_tau=context_tau,
                sample_steps=sample_steps,
            )
            return current_video.at[:, target_index].set(next_frame)

        return jax.lax.fori_loop(0, rollout_steps, body_fn, video_prefix)

    def loss(
        self,
        batch: DynamicsBatch,
        bootstrap_rows: int = 0,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        z_target = rearrange(
            jnp.asarray(batch["video"], dtype=jnp.float32),
            "b t (n k) d -> b t n (k d)",
            n=self.num_obs_tokens,
        )
        actions = jnp.asarray(batch["actions"], dtype=jnp.int32)

        batch_size, seq_len, _, _ = z_target.shape
        bootstrap_rows = min(max(int(bootstrap_rows), 0), batch_size)
        bootstrap_start = batch_size - bootstrap_rows
        bootstrap_row_mask = jnp.arange(batch_size) >= bootstrap_start
        bootstrap_row_mask = jnp.broadcast_to(
            bootstrap_row_mask[:, None],
            (batch_size, seq_len),
        )
        sample_rng = self.make_rng("sample")
        step_rng, signal_rng, noise_rng = jax.random.split(sample_rng, 3)

        sampled_bootstrap_levels = jax.random.randint(
            step_rng,
            shape=(batch_size, seq_len),
            minval=0,
            maxval=self.max_step_size - 1,
            dtype=jnp.int32,
        )
        step_levels = jnp.full((batch_size, seq_len), self.max_step_size - 1, dtype=jnp.int32)
        step_levels = jnp.where(bootstrap_row_mask, sampled_bootstrap_levels, step_levels)
        step_counts = 1 << step_levels
        signal_levels = jax.random.randint(
            signal_rng,
            shape=(batch_size, seq_len),
            minval=0,
            maxval=step_counts,
            dtype=jnp.int32,
        )

        tau = signal_levels.astype(jnp.float32) / step_counts.astype(jnp.float32)
        step_sizes = 1.0 / step_counts.astype(jnp.float32)
        tau = tau[..., None, None]
        step_sizes = step_sizes[..., None, None]

        # Full step prediction
        z_noise = jax.random.normal(noise_rng, z_target.shape, dtype=jnp.float32)
        z_noised = tau * z_target + (1.0 - tau) * z_noise
        z_pred_1 = self(z_noised, actions, step_levels, signal_levels)

        flow_loss = (z_pred_1 - z_target) ** 2
        loss_weight = 0.9 * tau + 0.1
        weighted_flow_loss = loss_weight * flow_loss

        bootstrap_loss_metric = jnp.asarray(0.0, dtype=jnp.float32)
        weighted_loss = weighted_flow_loss
        if bootstrap_rows > 0:
            bootstrap_slice = slice(bootstrap_start, batch_size)
            z_noised_bootstrap = z_noised[bootstrap_slice]
            actions_bootstrap = actions[bootstrap_slice]
            tau_bootstrap = tau[bootstrap_slice]
            step_sizes_bootstrap = step_sizes[bootstrap_slice]
            step_levels_bootstrap = step_levels[bootstrap_slice]
            signal_levels_bootstrap = signal_levels[bootstrap_slice]

            # Bootstrap is only used for tail rows, so avoid the extra forwards elsewhere.
            half_step_levels = jnp.minimum(
                step_levels_bootstrap + 1,
                self.max_step_size - 1,
            )
            z_pred_2 = self(
                z_noised_bootstrap,
                actions_bootstrap,
                half_step_levels,
                signal_levels_bootstrap * 2,
            )
            b1 = (z_pred_2 - z_noised_bootstrap) / (1.0 - tau_bootstrap)

            half_step_sizes = step_sizes_bootstrap / 2.0
            half_noised = z_noised_bootstrap + b1 * half_step_sizes
            z_pred_3 = self(
                half_noised,
                actions_bootstrap,
                half_step_levels,
                signal_levels_bootstrap * 2 + 1,
            )
            b2 = (z_pred_3 - half_noised) / (1.0 - (tau_bootstrap + half_step_sizes))

            bootstrap_target = jax.lax.stop_gradient((b1 + b2) / 2.0)
            bootstrap_loss = (
                (z_pred_1[bootstrap_slice] - z_noised_bootstrap)
                - (1.0 - tau_bootstrap) * bootstrap_target
            ) ** 2
            weighted_bootstrap_loss = loss_weight[bootstrap_slice] * bootstrap_loss
            bootstrap_loss_metric = jnp.mean(weighted_bootstrap_loss)
            weighted_loss = weighted_loss.at[bootstrap_slice].set(weighted_bootstrap_loss)

        total_loss = jnp.mean(weighted_loss)

        def masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
            mask = jnp.broadcast_to(mask, values.shape).astype(values.dtype)
            return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)

        use_bootstrap_loss = bootstrap_row_mask[..., None, None]
        flow_mask = (~use_bootstrap_loss).astype(jnp.float32)

        metrics = {
            "loss": total_loss,
            "flow_loss": jnp.mean(weighted_flow_loss),
            "bootstrap_loss": bootstrap_loss_metric,
            "active_flow_loss": masked_mean(weighted_flow_loss, flow_mask),
            "active_bootstrap_loss": bootstrap_loss_metric,
            "mean_tau": jnp.mean(tau),
            "mean_step_size": jnp.mean(step_sizes),
            "min_step_fraction": jnp.mean(
                (step_levels == self.max_step_size - 1).astype(jnp.float32)
            ),
            "bootstrap_active_fraction": jnp.mean(bootstrap_row_mask.astype(jnp.float32)),
            "bootstrap_active_rows": jnp.asarray(bootstrap_rows, dtype=jnp.float32),
        }
        return total_loss, metrics
