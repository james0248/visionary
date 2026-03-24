from einops import rearrange
import flax.linen as nn
import jax
import jax.numpy as jnp

from visionary.dataset import PreprocessedVideoDataset
from visionary.transformer import (
    TransformerBlock,
    create_spatial_rope,
    create_temporal_rope,
)


def create_encoder_spatial_mask(token_len: int, latent_len: int) -> jnp.ndarray:
    return jnp.block(
        [
            [
                jnp.ones((latent_len, latent_len), dtype=bool),
                jnp.ones((latent_len, token_len), dtype=bool),
            ],
            [
                jnp.zeros(
                    (token_len, latent_len), dtype=bool
                ),  # block images -> latent attention
                jnp.ones((token_len, token_len), dtype=bool),
            ],
        ]
    )


def create_decoder_spatial_mask(token_len: int, latent_len: int) -> jnp.ndarray:
    return jnp.block(
        [
            [
                jnp.ones((latent_len, latent_len), dtype=bool),
                jnp.zeros(
                    (latent_len, token_len), dtype=bool
                ),  # block latent -> images attention
            ],
            [
                jnp.ones((token_len, latent_len), dtype=bool),
                jnp.ones((token_len, token_len), dtype=bool),
            ],
        ]
    )


def create_temporal_mask(seq_len: int) -> jnp.ndarray:
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


class TokenizerEncoder(nn.Module):
    num_layers: int
    num_latents: int
    num_heads: int
    num_kv_heads: int

    model_dim: int
    head_dim: int
    mlp_hidden_dim: int
    channel_dim: int
    x_len: int
    y_len: int

    base: float

    @nn.compact
    def __call__(self, batch: PreprocessedVideoDataset) -> jnp.ndarray:
        x, p = batch["video"], batch["mask_prob"]

        b, t, num_tokens, _ = x.shape
        assert num_tokens == self.x_len * self.y_len
        total_tokens = num_tokens + self.num_latents

        x = nn.Dense(self.model_dim)(x)

        # Apply masking to patches
        rng = self.make_rng("mask")
        mask_token = self.param(
            "mask_token", nn.initializers.normal(stddev=0.02), (self.model_dim,)
        )
        rand_vals = jax.random.uniform(rng, shape=(b, t, num_tokens))
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
        x = jnp.concatenate([latent_tokens_bcast, x], axis=2)

        # Create RoPE and mask
        spatial_rope_cos, spatial_rope_sin = create_spatial_rope(
            self.base, self.head_dim, self.x_len, self.y_len
        )
        spatial_rope_emb = (
            jnp.pad(
                spatial_rope_cos,
                ((self.num_latents, 0), (0, 0)),
                mode="constant",
                constant_values=1,
            ),
            jnp.pad(
                spatial_rope_sin,
                ((self.num_latents, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            ),
        )
        spatial_mask = create_encoder_spatial_mask(num_tokens, self.num_latents)
        temporal_rope_emb = create_temporal_rope(self.base, self.head_dim, t)
        temporal_mask = create_temporal_mask(t)

        # Encoder
        for i in range(1, self.num_layers + 1):
            if i % 4 == 0:
                x = rearrange(x, "b t n d -> (b n) t d")
                rope_emb, mask = temporal_rope_emb, temporal_mask
            else:
                x = rearrange(x, "b t n d -> (b t) n d")
                rope_emb, mask = spatial_rope_emb, spatial_mask

            x = TransformerBlock(
                self.model_dim,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.mlp_hidden_dim,
            )(x, rope_emb, mask)

            if i % 4 == 0:
                x = rearrange(x, "(b n) t d -> b t n d", n=total_tokens)
            else:
                x = rearrange(x, "(b t) n d -> b t n d", t=t)

        latent_tokens = x[:, :, : self.num_latents, :]
        latent = nn.Dense(self.channel_dim)(latent_tokens)
        latent = jnp.tanh(latent)
        return latent


class TokenizerDecoder(nn.Module):
    num_layers: int
    num_latents: int
    num_heads: int
    num_kv_heads: int

    model_dim: int
    head_dim: int
    mlp_hidden_dim: int
    channel_dim: int
    x_len: int
    y_len: int

    base: float

    @nn.compact
    def __call__(self, latent: jnp.ndarray) -> jnp.ndarray:
        b, t, num_latents, _ = latent.shape
        assert num_latents == self.num_latents

        num_tokens = self.x_len * self.y_len
        total_tokens = num_tokens + self.num_latents

        image_tokens = self.param(
            "image_tokens",
            nn.initializers.normal(stddev=0.02),
            (num_tokens, self.model_dim),
        )
        image_tokens_bcast = jnp.broadcast_to(
            image_tokens, (b, t, num_tokens, self.model_dim)
        )

        x = jnp.concatenate([latent, image_tokens_bcast], axis=2)

        # Create RoPE and mask
        spatial_rope_cos, spatial_rope_sin = create_spatial_rope(
            self.base, self.head_dim, self.x_len, self.y_len
        )
        spatial_rope_emb = (
            jnp.pad(
                spatial_rope_cos,
                ((self.num_latents, 0), (0, 0)),
                mode="constant",
                constant_values=1,
            ),
            jnp.pad(
                spatial_rope_sin,
                ((self.num_latents, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            ),
        )
        spatial_mask = create_decoder_spatial_mask(num_tokens, self.num_latents)
        temporal_rope_emb = create_temporal_rope(self.base, self.head_dim, t)
        temporal_mask = create_temporal_mask(t)

        for i in range(1, self.num_layers + 1):
            if i % 4 == 0:
                x = rearrange(x, "b t n d -> (b n) t d")
                rope_emb, mask = temporal_rope_emb, temporal_mask
            else:
                x = rearrange(x, "b t n d -> (b t) n d")
                rope_emb, mask = spatial_rope_emb, spatial_mask

            x = TransformerBlock(
                self.model_dim,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.mlp_hidden_dim,
            )(x, rope_emb, mask)

            if i % 4 == 0:
                x = rearrange(x, "(b n) t d -> b t n d", n=total_tokens)
            else:
                x = rearrange(x, "(b t) n d -> b t n d", t=t)

        x = nn.Dense(self.channel_dim)(x)
        return x
