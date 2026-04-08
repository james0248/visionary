import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from visionary.dataset import DynamicsDataset
from visionary.transformer import SpatioTemporalTransformer


class ActionEmbedding(nn.Module):
    model_dim: int
    num_actions: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        action_tokens = self.param(
            "action_tokens",
            nn.initializers.normal(stddev=0.02),
            (self.num_actions, self.model_dim),
        ).astype(self.dtype)

        return action_tokens.take(actions, axis=0)


class SignalEmbedding(nn.Module):
    model_dim: int
    max_steps: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, step_levels: jnp.ndarray, signal_levels: jnp.ndarray) -> jnp.ndarray:
        signal_tokens = self.param(
            "signal_tokens",
            nn.initializers.normal(stddev=0.02),
            ((1 << (self.max_steps + 1)) - 1, self.model_dim),
        ).astype(self.dtype)

        indices = (1 << step_levels) + signal_levels - 1
        return signal_tokens.take(indices, axis=0)


class DynamicsModel(nn.Module):
    num_layers: int
    num_heads: int
    num_kv_heads: int
    num_registers: int
    num_actions: int

    max_step_size: int

    model_dim: int
    head_dim: int
    mlp_hidden_dim: int
    attention_logit_soft_cap: float | None = 50.0
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.signal_embedding = SignalEmbedding(
            model_dim=self.model_dim,
            max_steps=self.max_step_size,
            dtype=self.dtype,
        )
        self.action_embedding = ActionEmbedding(
            model_dim=self.model_dim,
            num_actions=self.num_actions,
            dtype=self.dtype,
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
        self.register_tokens = self.param(
            "register_tokens",
            nn.initializers.normal(stddev=0.02),
            (self.num_registers, self.model_dim),
        ).astype(self.dtype)

    def __call__(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray,
        step_levels: jnp.ndarray,
        signal_levels: jnp.ndarray,
    ) -> jnp.ndarray:
        batch_size, seq_len, num_spatial_tokens, _ = z.shape

        z = jnp.asarray(z, dtype=self.dtype)
        actions = jnp.asarray(actions)
        step_levels = jnp.asarray(step_levels)
        signal_levels = jnp.asarray(signal_levels)

        action_tokens = self.action_embedding(actions)[:, :, None, :]
        signal_tokens = self.signal_embedding(step_levels, signal_levels)[:, :, None, :]
        register_tokens = jnp.broadcast_to(
            self.register_tokens,
            (batch_size, seq_len, self.num_registers, self.model_dim),
        )

        return self.transformer(
            x=jnp.concatenate(
                [action_tokens, signal_tokens, register_tokens, z],
                axis=2,
            ),
            t=seq_len,
            total_tokens=num_spatial_tokens,
            spatial_rope_emb=None,
            spatial_mask=None,
            temporal_rope_emb=None,
            temporal_mask=None,
        )

    def loss(self, batch: DynamicsDataset) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        video = jnp.asarray(batch["video"], dtype=self.dtype)
        actions = jnp.asarray(batch["actions"])

        batch_size, seq_len, patch_len, _ = video.shape

        # Sample a dyadic step level and convert it to the actual step size d = 2^-level.
        step_levels = jax.random.randint(
            self.make_rng("sample"),
            shape=(batch_size, seq_len),
            minval=0,
            maxval=self.max_step_size,
        )
        signal_levels = jax.random.randint(
            self.make_rng("sample"),
            shape=(batch_size, seq_len),
            minval=0,
            maxval=1 << step_levels,
        )

        z_target = rearrange(video, "b t (n s) d -> b t n (s d)", s=patch_len)

        step_sizes = 1.0 / ((1 << step_levels).astype(self.dtype))
        step_sizes = step_sizes[..., None, None]
        tau = signal_levels.astype(self.dtype) * step_sizes
        tau = tau[..., None, None]

        # Full step prediction
        z_noised = tau * z_target + (1.0 - tau) * jax.random.normal(
            self.make_rng("sample"), z_target.shape, dtype=self.dtype
        )
        z_pred_1 = self(z_noised, actions, step_levels, signal_levels)

        # Half step prediction from tau
        half_step_sizes = step_sizes / 2.0
        half_step_levels = step_levels + 1

        z_pred_2 = self(z_noised, actions, half_step_levels, signal_levels * 2)
        b1 = (z_pred_2 - z_noised) / (1.0 - tau)

        # Half step prediction from tau + d/2
        half_noised_tokens = z_noised + b1 * half_step_sizes
        z_pred_3 = self(half_noised_tokens, actions, half_step_levels, signal_levels * 2 + 1)
        b2 = (z_pred_3 - half_noised_tokens) / (1.0 - (tau + half_step_sizes))

        # Loss calculation
        flow_loss = (z_pred_1 - z_target) ** 2
        bootstrap_loss = (
            (z_pred_1 - z_noised) - (1.0 - tau) * jax.lax.stop_gradient((b1 + b2) / 2.0)
        ) ** 2

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
            "min_step_fraction": jnp.mean(is_min_step.astype(jnp.float32)),
        }
        return total_loss, metrics
