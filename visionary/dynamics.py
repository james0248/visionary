import math

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
