import flax.linen as nn
import jax.numpy as jnp
import jax

from visionary.dataset import PreprocessedVideoDataset


class Tokenizer(nn.Module):
    model_dim: int
    num_latents: int

    @nn.compact
    def __call__(self, batch: PreprocessedVideoDataset) -> jnp.ndarray:
        x, p = batch["video"], batch["mask_prob"]

        b, t, n, _ = x.shape

        x = nn.Dense(self.model_dim)(x)

        # Apply masking to patches
        rng = self.make_rng("mask")
        mask_token = self.param(
            "mask_token", nn.initializers.normal(stddev=0.02), (self.model_dim,)
        )
        rand_vals = jax.random.uniform(rng, shape=(b, t, n))
        p = jnp.expand_dims(p, axis=-1)
        is_masked = jnp.expand_dims(rand_vals < p, axis=-1)
        x = jnp.where(is_masked, mask_token, x)

        # Concatenate latent tokens
        latent_tokens = self.param(
            "latent_tokens",
            nn.initializers.normal(stddev=0.02),
            (self.num_latents, self.model_dim),
        )
        latent_tokens_bcast = jnp.broadcast_to(
            latent_tokens, (b, t, self.num_latents, self.model_dim)
        )
        x = jnp.concatenate([x, latent_tokens_bcast], axis=2)

        return x
