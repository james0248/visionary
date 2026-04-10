import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from visionary.dataset import DynamicsBatch
from visionary.transformer import SpatioTemporalTransformer, create_temporal_rope


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

        action_tokens = nn.Embed(
            num_embeddings=self.num_actions,
            features=self.model_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="action_embedding",
        )(jnp.asarray(actions, dtype=jnp.int32))
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
        observation_tokens = nn.Dense(self.model_dim, dtype=self.dtype)(z.astype(self.dtype))

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
        )(observation_hidden)

    def loss(self, batch: DynamicsBatch) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        z_target = jnp.asarray(batch["video"], dtype=jnp.float32)
        z_target = rearrange(z_target, "b t (n k) d -> b t n (k d)", n=self.num_obs_tokens)
        actions = jnp.asarray(batch["actions"], dtype=jnp.int32)

        batch_size, seq_len, _, _ = z_target.shape
        sample_rng = self.make_rng("sample")
        step_rng, signal_rng, noise_rng = jax.random.split(sample_rng, 3)

        step_levels = jax.random.randint(
            step_rng,
            shape=(batch_size, seq_len),
            minval=0,
            maxval=self.max_step_size,
            dtype=jnp.int32,
        )
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

        # Half step prediction from tau
        half_step_levels = jnp.minimum(step_levels + 1, self.max_step_size - 1)
        z_pred_2 = self(z_noised, actions, half_step_levels, signal_levels * 2)
        b1 = (z_pred_2 - z_noised) / (1.0 - tau)

        # Half step prediction from tau + d/2
        half_step_sizes = step_sizes / 2.0
        half_noised = z_noised + b1 * half_step_sizes
        z_pred_3 = self(half_noised, actions, half_step_levels, signal_levels * 2 + 1)
        b2 = (z_pred_3 - half_noised) / (1.0 - (tau + half_step_sizes))

        # Loss computation
        flow_loss = (z_pred_1 - z_target) ** 2
        bootstrap_target = jax.lax.stop_gradient((b1 + b2) / 2.0)
        bootstrap_loss = ((z_pred_1 - z_noised) - (1.0 - tau) * bootstrap_target) ** 2

        is_min_step = (step_levels == self.max_step_size - 1)[..., None, None]
        loss = jnp.where(is_min_step, flow_loss, bootstrap_loss)
        loss_weight = 0.9 * tau + 0.1
        weighted_loss = loss * loss_weight
        total_loss = jnp.mean(weighted_loss)

        metrics = {
            "loss": total_loss,
            "flow_loss": jnp.mean(loss_weight * flow_loss),
            "bootstrap_loss": jnp.mean(loss_weight * bootstrap_loss),
            "mean_tau": jnp.mean(tau),
            "mean_step_size": jnp.mean(step_sizes),
            "min_step_fraction": jnp.mean(is_min_step),
        }
        return total_loss, metrics
