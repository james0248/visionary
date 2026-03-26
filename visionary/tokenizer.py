import flax.linen as nn
import jax
import jax.numpy as jnp

from visionary.dataset import PreprocessedVideoDataset
from visionary.transformer import (
    SpatioTemporalTransformer,
    create_spatial_rope,
    create_temporal_rope,
    pad_rope_for_latents,
)


def create_spatial_mask(
    num_image_tokens: int, num_latent_tokens: int, *, encoder: bool
) -> jnp.ndarray:
    n_img, n_lat = num_image_tokens, num_latent_tokens

    latent_to_latent = jnp.ones((n_lat, n_lat), dtype=bool)
    image_to_image = jnp.ones((n_img, n_img), dtype=bool)

    if encoder:
        latent_to_image = jnp.ones((n_lat, n_img), dtype=bool)
        image_to_latent = jnp.zeros((n_img, n_lat), dtype=bool)
    else:
        latent_to_image = jnp.zeros((n_lat, n_img), dtype=bool)
        image_to_latent = jnp.ones((n_img, n_lat), dtype=bool)

    return jnp.block([
        [latent_to_latent, latent_to_image],
        [image_to_latent,  image_to_image],
    ])


def create_temporal_mask(independent: jnp.ndarray, t: int) -> jnp.ndarray:
    causal = jnp.tril(jnp.ones((t, t), dtype=bool))
    identity = jnp.eye(t, dtype=bool)
    return jnp.where(independent[:, None, None], identity, causal)


def _build_rope_embeddings(
    base: float,
    head_dim: int,
    x_len: int,
    y_len: int,
    num_latents: int,
    seq_len: int,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    spatial_rope = pad_rope_for_latents(
        *create_spatial_rope(base, head_dim, x_len, y_len),
        num_latents,
    )
    temporal_rope = create_temporal_rope(base, head_dim, seq_len)
    return spatial_rope, temporal_rope


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
    def __call__(
        self, x: jnp.ndarray, mask_prob: jnp.ndarray, temporal_mask: jnp.ndarray
    ) -> jnp.ndarray:
        batch_size, seq_len, num_tokens, _ = x.shape
        expected_num_tokens = self.x_len * self.y_len
        if num_tokens != expected_num_tokens:
            raise ValueError(
                "Tokenizer encoder expected "
                f"{expected_num_tokens} spatial tokens, got {num_tokens}."
            )

        x = nn.Dense(self.model_dim)(x)

        # Apply masking to patches
        rng = self.make_rng("mask")
        mask_token = self.param(
            "mask_token", nn.initializers.normal(stddev=0.02), (self.model_dim,)
        )
        rand_vals = jax.random.uniform(rng, shape=(batch_size, seq_len, num_tokens))
        mask_threshold = jnp.expand_dims(mask_prob, axis=-1)
        is_masked = jnp.expand_dims(rand_vals < mask_threshold, axis=-1)
        x = jnp.where(is_masked, mask_token, x)

        # Prepend latent tokens
        latent_tokens = self.param(
            "latent_tokens",
            nn.initializers.normal(stddev=0.02),
            (self.num_latents, self.model_dim),
        )
        latent_tokens = jnp.broadcast_to(
            latent_tokens, (batch_size, seq_len, self.num_latents, self.model_dim)
        )
        x = jnp.concatenate([latent_tokens, x], axis=2)

        spatial_rope, temporal_rope = _build_rope_embeddings(
            self.base,
            self.head_dim,
            self.x_len,
            self.y_len,
            self.num_latents,
            seq_len,
        )

        x = SpatioTemporalTransformer(
            num_layers=self.num_layers,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            mlp_hidden_dim=self.mlp_hidden_dim,
        )(
            x=x,
            t=seq_len,
            total_tokens=num_tokens + self.num_latents,
            spatial_rope_emb=spatial_rope,
            spatial_mask=create_spatial_mask(num_tokens, self.num_latents, encoder=True),
            temporal_rope_emb=temporal_rope,
            temporal_mask=temporal_mask,
        )

        latent = x[:, :, : self.num_latents, :]
        latent = nn.Dense(self.channel_dim)(latent)
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
    def __call__(
        self,
        latent: jnp.ndarray,
        temporal_mask: jnp.ndarray,
        patch_dim: int,
    ) -> jnp.ndarray:
        batch_size, seq_len, num_latents, _ = latent.shape
        if num_latents != self.num_latents:
            raise ValueError(
                "Tokenizer decoder expected "
                f"{self.num_latents} latent tokens, got {num_latents}."
            )
        num_tokens = self.x_len * self.y_len

        image_tokens = self.param(
            "image_tokens",
            nn.initializers.normal(stddev=0.02),
            (num_tokens, self.model_dim),
        )
        image_tokens = jnp.broadcast_to(
            image_tokens, (batch_size, seq_len, num_tokens, self.model_dim)
        )

        latent = nn.Dense(self.model_dim)(latent)
        x = jnp.concatenate([latent, image_tokens], axis=2)

        spatial_rope, temporal_rope = _build_rope_embeddings(
            self.base,
            self.head_dim,
            self.x_len,
            self.y_len,
            self.num_latents,
            seq_len,
        )

        x = SpatioTemporalTransformer(
            num_layers=self.num_layers,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            mlp_hidden_dim=self.mlp_hidden_dim,
        )(
            x=x,
            t=seq_len,
            total_tokens=num_tokens + self.num_latents,
            spatial_rope_emb=spatial_rope,
            spatial_mask=create_spatial_mask(num_tokens, self.num_latents, encoder=False),
            temporal_rope_emb=temporal_rope,
            temporal_mask=temporal_mask,
        )

        x = x[:, :, self.num_latents :, :]
        x = nn.Dense(patch_dim)(x)
        return x


class Tokenizer(nn.Module):
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

    def setup(self):
        shared = dict(
            num_layers=self.num_layers,
            num_latents=self.num_latents,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            model_dim=self.model_dim,
            head_dim=self.head_dim,
            mlp_hidden_dim=self.mlp_hidden_dim,
            channel_dim=self.channel_dim,
            x_len=self.x_len,
            y_len=self.y_len,
            base=self.base,
        )
        self.encoder = TokenizerEncoder(**shared)
        self.decoder = TokenizerDecoder(**shared)

    def _validate_batch(self, batch: PreprocessedVideoDataset) -> int:
        num_tokens = batch["video"].shape[-2]
        expected_num_tokens = self.x_len * self.y_len
        if num_tokens != expected_num_tokens:
            raise ValueError(
                "Tokenizer expected "
                f"{expected_num_tokens} spatial tokens from x_len*y_len, got "
                f"{num_tokens}."
            )
        patch_dim = batch["video"].shape[-1]
        if patch_dim <= 0:
            raise ValueError(f"Tokenizer expected positive patch_dim, got {patch_dim}.")
        return patch_dim

    def __call__(self, batch: PreprocessedVideoDataset) -> jnp.ndarray:
        patch_dim = self._validate_batch(batch)
        temporal_mask = create_temporal_mask(
            batch["independent"], batch["video"].shape[1]
        )
        latent = self.encoder(batch["video"], batch["mask_prob"], temporal_mask)
        return self.decoder(latent, temporal_mask, patch_dim)

    def encode(self, batch: PreprocessedVideoDataset) -> jnp.ndarray:
        self._validate_batch(batch)
        temporal_mask = create_temporal_mask(
            batch["independent"], batch["video"].shape[1]
        )
        return self.encoder(batch["video"], batch["mask_prob"], temporal_mask)
