import math
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from visionary.dataset import DynamicsBatch
from visionary.models.dreamer4.transformer import (
    SpatioTemporalTransformer,
    create_temporal_rope,
)


def normalize_latents(latents: jnp.ndarray, mean: jax.typing.ArrayLike, std: jax.typing.ArrayLike) -> jnp.ndarray:
    latents = jnp.asarray(latents, dtype=jnp.float32)
    std = jnp.maximum(jnp.asarray(std, dtype=jnp.float32), 1e-8)
    return (latents - jnp.asarray(mean, dtype=jnp.float32)) / std


def denormalize_latents(latents: jnp.ndarray, mean: jax.typing.ArrayLike, std: jax.typing.ArrayLike) -> jnp.ndarray:
    latents = jnp.asarray(latents, dtype=jnp.float32)
    std = jnp.maximum(jnp.asarray(std, dtype=jnp.float32), 1e-8)
    return latents * std + jnp.asarray(mean, dtype=jnp.float32)


class ActionEmbedding(nn.Module):
    model_dim: int
    num_actions: int
    num_embodiments: int
    max_action_dim: int
    action_mode: Literal["discrete", "continuous"]
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, actions: jnp.ndarray, embodiment_ids: jnp.ndarray) -> jnp.ndarray:
        base_token = self.param(
            "base_token",
            nn.initializers.normal(stddev=0.02),
            (self.model_dim,),
        ).astype(self.dtype)

        if self.action_mode == "continuous":
            actions = jnp.asarray(actions, dtype=self.dtype)
            embodiment_ids = jnp.asarray(embodiment_ids, dtype=jnp.int32)
            action_kernels = self.param(
                "action_projection_kernel",
                nn.initializers.lecun_normal(),
                (self.num_embodiments, self.max_action_dim, self.model_dim),
            ).astype(self.dtype)
            action_biases = self.param(
                "action_projection_bias",
                nn.initializers.zeros,
                (self.num_embodiments, self.model_dim),
            ).astype(self.dtype)
            selected_kernels = jnp.take(action_kernels, embodiment_ids, axis=0)
            selected_biases = jnp.take(action_biases, embodiment_ids, axis=0)
            action_tokens = jnp.einsum("bta,btad->btd", actions, selected_kernels) + selected_biases
        elif self.action_mode == "discrete":
            actions = jnp.asarray(actions, dtype=jnp.int32)
            valid_actions = actions >= 0
            safe_actions = jnp.where(valid_actions, actions, 0)
            action_tokens = nn.Embed(
                num_embeddings=self.num_actions,
                features=self.model_dim,
                embedding_init=nn.initializers.normal(stddev=1.0),
                dtype=self.dtype,
                name="action_embedding",
            )(safe_actions)
            action_tokens = jnp.where(valid_actions[..., None], action_tokens, 0)
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode!r}")

        return action_tokens + base_token


class ShortcutEmbedding(nn.Module):
    model_dim: int
    max_step_size: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, step_levels: jnp.ndarray, signal_levels: jnp.ndarray) -> jnp.ndarray:
        half_dim = self.model_dim // 2
        step_tokens = nn.Embed(
            num_embeddings=self.max_step_size,
            features=half_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="step_embedding",
        )(jnp.asarray(step_levels, dtype=jnp.int32))
        signal_tokens = nn.Embed(
            num_embeddings=(1 << (self.max_step_size - 1)) + 1,
            features=half_dim,
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
    num_embodiments: int
    max_action_dim: int

    max_step_size: int

    model_dim: int
    head_dim: int
    mlp_hidden_dim: int
    context_length: int
    latent_mean: tuple[float, ...]
    latent_std: tuple[float, ...]

    action_mode: Literal["discrete", "continuous"]
    temporal_layer_period: int = 4
    base: float = 10000.0
    remat_policy: str | None = None
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.action_embedding = ActionEmbedding(
            model_dim=self.model_dim,
            num_actions=self.num_actions,
            num_embodiments=self.num_embodiments,
            max_action_dim=self.max_action_dim,
            action_mode=self.action_mode,
            dtype=self.dtype,
        )
        self.shortcut_embedding = ShortcutEmbedding(
            model_dim=self.model_dim,
            max_step_size=self.max_step_size,
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
            temporal_layer_period=self.temporal_layer_period,
            remat_policy=self.remat_policy,
            dtype=self.dtype,
        )

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray,
        embodiment_ids: jnp.ndarray,
        step_levels: jnp.ndarray,
        signal_levels: jnp.ndarray,
        segment_ids: jnp.ndarray,
    ) -> jnp.ndarray:
        batch_size, seq_len, input_obs_tokens, token_dim = z.shape

        action_tokens = self.action_embedding(actions, embodiment_ids)[:, :, None, :]
        shortcut_tokens = self.shortcut_embedding(step_levels, signal_levels)[:, :, None, :]
        register_tokens = jnp.broadcast_to(
            self.register_tokens.astype(self.dtype),
            (batch_size, seq_len, self.num_registers, self.model_dim),
        )
        observation_tokens = nn.Dense(
            self.model_dim,
            dtype=self.dtype,
            name="observation_projection",
        )(z.astype(self.dtype))

        observation_offset = 1 + 1 + self.num_registers  # action, shortcut, register
        num_tokens = observation_offset + input_obs_tokens
        tokens = jnp.concatenate([action_tokens, shortcut_tokens, register_tokens, observation_tokens], axis=2)

        spatial_rope = create_temporal_rope(self.base, self.head_dim, num_tokens)
        temporal_rope = create_temporal_rope(self.base, self.head_dim, seq_len)

        # Actions cannot see latent tokens.
        token_levels = jnp.concatenate([jnp.zeros(1), jnp.ones(num_tokens - 1)], dtype=jnp.int32)
        spatial_mask = token_levels[:, None] >= token_levels[None, :]

        # Context length windowed mask
        temporal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        temporal_mask = jnp.triu(temporal_mask, k=1 - self.context_length)
        temporal_mask = jnp.broadcast_to(temporal_mask[None, :, :], (batch_size, seq_len, seq_len))

        # Block attention
        segment_ids = jnp.asarray(segment_ids, dtype=jnp.int32)
        same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
        temporal_mask = temporal_mask & same_segment

        hidden = self.transformer(
            x=tokens,
            t=seq_len,
            total_tokens=num_tokens,
            spatial_rope_emb=spatial_rope,
            spatial_mask=spatial_mask,
            temporal_rope_emb=temporal_rope,
            temporal_mask=temporal_mask,
        )
        observation_hidden = hidden[:, :, observation_offset:, :]
        return nn.Dense(
            token_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="prediction_projection",
        )(observation_hidden)

    def generate_next(
        self,
        video_prefix: jnp.ndarray,
        actions: jnp.ndarray,
        embodiment_ids: jnp.ndarray,
        context_noise: jnp.ndarray,
        sample_noise: jnp.ndarray,
        target_index: jnp.ndarray,
        context_tau: float,
        sample_steps: int,
        clean_until: jnp.ndarray | int,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Generate one latent frame at each row's target index."""

        batch_size, seq_len, _, latent_dim = video_prefix.shape
        target_index = jnp.asarray(target_index, dtype=jnp.int32)
        clean_until = jnp.asarray(clean_until, dtype=jnp.int32)

        k_max = 1 << (self.max_step_size - 1)
        finest_step_level = self.max_step_size - 1
        if not 0.0 <= context_tau <= 1.0:
            raise ValueError(f"context_tau must be in [0, 1], got {context_tau}")

        sample_step_level = int(round(math.log2(sample_steps)))
        sample_step_count = 1 << sample_step_level
        sample_step_size = jnp.float32(1.0 / sample_step_count)
        sample_signal_stride = k_max // sample_step_count
        context_step_level = sample_step_level if sample_step_level == finest_step_level else finest_step_level - 1
        context_step_count = 1 << context_step_level
        context_signal_level = int(context_tau * context_step_count)
        context_signal_index = context_signal_level * (k_max // context_step_count)
        context_tau = jnp.float32(context_signal_level / context_step_count)

        z_prefix = rearrange(video_prefix, "b t (n k) d -> b t n (k d)", n=self.num_obs_tokens)
        z_context_noise = rearrange(context_noise, "b t (n k) d -> b t n (k d)", n=self.num_obs_tokens)
        z_sample_noise = rearrange(sample_noise[:, None], "b t (n k) d -> b t n (k d)", n=self.num_obs_tokens)[:, 0]
        _, _, num_obs_tokens, token_dim = z_prefix.shape
        segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

        positions = jnp.arange(seq_len, dtype=jnp.int32)
        past_mask = positions < target_index
        past_mask_z = past_mask[None, :, None, None]
        past_mask_t = jnp.broadcast_to(past_mask[None, :], (batch_size, seq_len))

        clean_mask = positions < clean_until
        clean_mask_z = clean_mask[None, :, None, None]
        clean_mask_t = jnp.broadcast_to(clean_mask[None, :], (batch_size, seq_len))

        z_prefix = z_prefix.astype(jnp.float32)
        noised_prefix = context_tau * z_prefix + (1.0 - context_tau) * z_context_noise.astype(jnp.float32)
        base_z = jnp.where(
            past_mask_z,
            jnp.where(clean_mask_z, z_prefix, noised_prefix),
            jnp.zeros((batch_size, seq_len, num_obs_tokens, token_dim), dtype=jnp.float32),
        )
        base_step_levels = jnp.where(
            past_mask_t,
            jnp.where(clean_mask_t, finest_step_level, context_step_level),
            0,
        ).astype(jnp.int32)
        base_signal_indices = jnp.where(
            past_mask_t,
            jnp.where(clean_mask_t, k_max, context_signal_index),
            0,
        ).astype(jnp.int32)

        def sample_step(
            sample_signal_level: int,
            carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            current_z, x0_norm, update_mag = carry
            step_levels = base_step_levels.at[:, target_index].set(sample_step_level)
            signal_indices = base_signal_indices.at[:, target_index].set(sample_signal_level * sample_signal_stride)
            z_input = base_z.at[:, target_index].set(current_z)
            predicted = self(z_input, actions, embodiment_ids, step_levels, signal_indices, segment_ids)[
                :, target_index
            ].astype(jnp.float32)

            tau = jnp.float32(sample_signal_level / sample_step_count)
            velocity = (predicted - current_z) / jnp.maximum(1.0 - tau, 1e-6)
            update = velocity * sample_step_size
            x0_norm = x0_norm.at[sample_signal_level].set(jnp.mean(jnp.abs(predicted)))
            update_mag = update_mag.at[sample_signal_level].set(jnp.mean(jnp.abs(update)))
            return current_z + update, x0_norm, update_mag

        current_z, x0_norm, update_mag = jax.lax.fori_loop(
            0,
            sample_step_count,
            sample_step,
            (
                z_sample_noise.astype(jnp.float32),
                jnp.zeros(sample_step_count, dtype=jnp.float32),
                jnp.zeros(sample_step_count, dtype=jnp.float32),
            ),
        )
        frame = rearrange(current_z[:, None], "b t n (k d) -> b t (n k) d", d=latent_dim)[:, 0]
        return frame, {"x0_norm": x0_norm, "update_mag": update_mag}

    def generate_rollout(
        self,
        video_prefix: jnp.ndarray,
        actions: jnp.ndarray,
        embodiment_ids: jnp.ndarray,
        context_noise: jnp.ndarray,
        sample_noise: jnp.ndarray,
        start_index: jnp.ndarray,
        context_tau: float,
        sample_steps: int,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Generate consecutive frames into a preallocated video tensor."""

        start_index = jnp.asarray(start_index, dtype=jnp.int32)
        rollout_steps = sample_noise.shape[1]
        sample_step_count = 1 << int(round(math.log2(sample_steps)))

        def body_fn(
            offset: int,
            carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            current_video, x0_norm, update_mag = carry
            target_index = start_index + offset
            next_frame, diagnostics = self.generate_next(
                current_video,
                actions,
                embodiment_ids,
                context_noise,
                sample_noise[:, offset],
                target_index,
                context_tau=context_tau,
                sample_steps=sample_steps,
                clean_until=start_index,
            )
            current_video = current_video.at[:, target_index].set(next_frame)
            x0_norm = x0_norm.at[offset].set(diagnostics["x0_norm"])
            update_mag = update_mag.at[offset].set(diagnostics["update_mag"])
            return current_video, x0_norm, update_mag

        generated, x0_norm, update_mag = jax.lax.fori_loop(
            0,
            rollout_steps,
            body_fn,
            (
                video_prefix,
                jnp.zeros((rollout_steps, sample_step_count), dtype=jnp.float32),
                jnp.zeros((rollout_steps, sample_step_count), dtype=jnp.float32),
            ),
        )
        return generated, {"x0_norm": x0_norm, "update_mag": update_mag}

    def loss(
        self,
        batch: DynamicsBatch,
        bootstrap_target_variables,
        bootstrap_rows: int,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute shortcut forcing loss. Empirical rows use flow loss. Final rows use targets from two EMA half-steps."""

        z_target = rearrange(
            jnp.asarray(batch["video"], dtype=jnp.float32), "b t (n k) d -> b t n (k d)", n=self.num_obs_tokens
        )
        action_dtype = jnp.float32 if self.action_mode == "continuous" else jnp.int32
        actions = jnp.asarray(batch["actions"], dtype=action_dtype)
        embodiment_ids = jnp.asarray(batch["embodiment_ids"], dtype=jnp.int32)
        segment_ids = jnp.asarray(batch["segment_ids"], dtype=jnp.int32)

        batch_size, seq_len, _, _ = z_target.shape
        if not 0 <= bootstrap_rows <= batch_size:
            raise ValueError(f"Expected 0 <= bootstrap_rows <= {batch_size}, got {bootstrap_rows}")

        transition_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
        transition_mask = transition_mask.at[:, 0].set(False)
        same_segment = segment_ids[:, 1:] == segment_ids[:, :-1]
        transition_mask = transition_mask.at[:, 1:].set(same_segment)

        bootstrap_start = batch_size - bootstrap_rows
        bootstrap_slice = slice(bootstrap_start, batch_size)
        bootstrap_row_mask = jnp.arange(batch_size)[:, None] >= bootstrap_start
        bootstrap_row_mask = jnp.broadcast_to(bootstrap_row_mask, (batch_size, seq_len))

        step_rng, signal_rng, noise_rng = jax.random.split(self.make_rng("sample"), 3)

        finest_step_level = self.max_step_size - 1
        sampled_bootstrap_levels = jax.random.randint(
            step_rng,
            shape=(batch_size, seq_len),
            minval=0,
            maxval=finest_step_level,
            dtype=jnp.int32,
        )
        step_levels = jnp.full((batch_size, seq_len), finest_step_level, dtype=jnp.int32)
        step_levels = jnp.where(bootstrap_row_mask, sampled_bootstrap_levels, step_levels)
        step_counts = 1 << step_levels

        include_endpoint = jnp.where(bootstrap_row_mask, 0, 1)
        signal_levels = jax.random.randint(
            signal_rng,
            shape=(batch_size, seq_len),
            minval=0,
            maxval=step_counts + include_endpoint,
            dtype=jnp.int32,
        )

        signal_indices = signal_levels << (finest_step_level - step_levels)

        tau = signal_levels.astype(jnp.float32) / step_counts.astype(jnp.float32)
        step_sizes = 1.0 / step_counts.astype(jnp.float32)
        tau = tau[..., None, None]
        step_sizes = step_sizes[..., None, None]

        z_noise = jax.random.normal(noise_rng, z_target.shape, dtype=jnp.float32)
        z_noised = tau * z_target + (1.0 - tau) * z_noise
        z_pred_1 = self(z_noised, actions, embodiment_ids, step_levels, signal_indices, segment_ids)

        flow_loss = (z_pred_1 - z_target) ** 2
        loss_weight = 0.9 * tau + 0.1
        weighted_flow_loss = loss_weight * flow_loss

        weighted_loss = weighted_flow_loss
        if bootstrap_rows > 0:
            z_noised_bootstrap = z_noised[bootstrap_slice]
            actions_bootstrap = actions[bootstrap_slice]
            embodiment_ids_bootstrap = embodiment_ids[bootstrap_slice]
            tau_bootstrap = tau[bootstrap_slice]
            step_sizes_bootstrap = step_sizes[bootstrap_slice]
            step_levels_bootstrap = step_levels[bootstrap_slice]
            signal_indices_bootstrap = signal_indices[bootstrap_slice]
            segment_ids_bootstrap = segment_ids[bootstrap_slice]

            bootstrap_forward = nn.apply(
                lambda model, z, action, embodiment, level, signal, segments: model(
                    z, action, embodiment, level, signal, segments
                ),
                self.clone(parent=None),
            )

            # First EMA half-step
            half_step_levels = step_levels_bootstrap + 1
            z_pred_2 = bootstrap_forward(
                bootstrap_target_variables,
                z_noised_bootstrap,
                actions_bootstrap,
                embodiment_ids_bootstrap,
                half_step_levels,
                signal_indices_bootstrap,
                segment_ids_bootstrap,
            )
            first_step_divisor = jnp.maximum(1.0 - tau_bootstrap, 1e-5)
            b1 = (jax.lax.stop_gradient(z_pred_2) - z_noised_bootstrap) / first_step_divisor

            # Second EMA half-step
            half_step_sizes = step_sizes_bootstrap / 2.0
            half_noised_unclipped = z_noised_bootstrap + b1 * half_step_sizes
            half_noised = jnp.clip(half_noised_unclipped, -4.0, 4.0)
            half_signal_delta = 1 << (self.max_step_size - 2 - step_levels_bootstrap)
            z_pred_3 = bootstrap_forward(
                bootstrap_target_variables,
                half_noised,
                actions_bootstrap,
                embodiment_ids_bootstrap,
                half_step_levels,
                signal_indices_bootstrap + half_signal_delta,
                segment_ids_bootstrap,
            )
            second_step_divisor = jnp.maximum(1.0 - (tau_bootstrap + half_step_sizes), 1e-5)
            b2 = (jax.lax.stop_gradient(z_pred_3) - half_noised) / second_step_divisor

            bootstrap_target_unclipped = jax.lax.stop_gradient((b1 + b2) / 2.0)
            bootstrap_target_norm = jnp.mean(jnp.abs(bootstrap_target_unclipped))
            bootstrap_target = jnp.clip(bootstrap_target_unclipped, -4.0, 4.0)
            bootstrap_error = (z_pred_1[bootstrap_slice] - z_noised_bootstrap) - (
                1.0 - tau_bootstrap
            ) * bootstrap_target
            bootstrap_loss = bootstrap_error**2
            weighted_bootstrap_loss = loss_weight[bootstrap_slice] * bootstrap_loss
            weighted_loss = weighted_loss.at[bootstrap_slice].set(weighted_bootstrap_loss)

        loss_mask = jnp.broadcast_to(transition_mask[..., None, None], weighted_loss.shape)
        loss_mask = loss_mask.astype(jnp.float32)
        loss_denominator = jnp.maximum(jnp.sum(loss_mask, axis=(1, 2, 3)), 1.0)
        row_loss = jnp.sum(weighted_loss * loss_mask, axis=(1, 2, 3)) / loss_denominator
        flow_row_loss = jnp.sum(weighted_flow_loss * loss_mask, axis=(1, 2, 3)) / loss_denominator
        flow_row_mse = jnp.sum(flow_loss * loss_mask, axis=(1, 2, 3)) / loss_denominator

        total_loss = jnp.mean(row_loss)
        flow_loss_metric = (
            jnp.mean(flow_row_loss[:bootstrap_start]) if bootstrap_start > 0 else jnp.asarray(0.0, dtype=jnp.float32)
        )
        flow_mse_metric = (
            jnp.mean(flow_row_mse[:bootstrap_start]) if bootstrap_start > 0 else jnp.asarray(0.0, dtype=jnp.float32)
        )
        bootstrap_loss_metric = (
            jnp.mean(row_loss[bootstrap_slice]) if bootstrap_rows > 0 else jnp.asarray(0.0, dtype=jnp.float32)
        )

        flow_token_mse = jnp.mean(flow_loss, axis=(-2, -1))
        empirical_mask = jnp.arange(batch_size)[:, None] < bootstrap_start
        flow_metric_mask = transition_mask & empirical_mask

        def sigma_mse(sigma_mask: jnp.ndarray) -> jnp.ndarray:
            metric_mask = flow_metric_mask & sigma_mask
            return jnp.sum(flow_token_mse * metric_mask) / jnp.maximum(jnp.sum(metric_mask), 1.0)

        tau_tokens = tau[..., 0, 0]
        flow_mse_low = sigma_mse(tau_tokens < 0.25)
        flow_mse_mid = sigma_mse((tau_tokens >= 0.25) & (tau_tokens < 0.75))
        flow_mse_high = sigma_mse(tau_tokens >= 0.75)

        metrics = {
            "loss": total_loss,
            "flow_loss": flow_loss_metric,
            "flow_mse": flow_mse_metric,
            "flow_mse_low": flow_mse_low,
            "flow_mse_mid": flow_mse_mid,
            "flow_mse_high": flow_mse_high,
            "bootstrap_loss": bootstrap_loss_metric,
            "bootstrap_target_norm": (
                bootstrap_target_norm if bootstrap_rows > 0 else jnp.asarray(0.0, dtype=jnp.float32)
            ),
        }
        return total_loss, metrics
