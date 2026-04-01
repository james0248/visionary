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
    num_image_tokens: int, num_latent_tokens: int, encoder: bool
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

    return jnp.block(
        [
            [latent_to_latent, latent_to_image],
            [image_to_latent, image_to_image],
        ]
    )


def create_temporal_mask(independent: jnp.ndarray, t: int) -> jnp.ndarray:
    causal = jnp.tril(jnp.ones((t, t), dtype=bool))
    identity = jnp.eye(t, dtype=bool)
    return jnp.where(independent[:, None, None], identity, causal)


def build_rope_embeddings(
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
    bottleneck_norm: str = "tanh"
    attention_logit_soft_cap: float | None = 50.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        temporal_mask: jnp.ndarray,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        batch_size, seq_len, num_tokens, _ = x.shape

        x = nn.Dense(self.model_dim, dtype=self.dtype)(x)

        # Apply masking to patches
        mask_token = self.param(
            "mask_token", nn.initializers.normal(stddev=0.02), (self.model_dim,)
        ).astype(self.dtype)
        if mask is None:
            mask = jnp.zeros((batch_size, seq_len, num_tokens), dtype=bool)
        x = jnp.where(jnp.expand_dims(mask, axis=-1), mask_token, x)

        # Prepend latent tokens
        latent_tokens = self.param(
            "latent_tokens",
            nn.initializers.normal(stddev=0.02),
            (self.num_latents, self.model_dim),
        ).astype(self.dtype)
        latent_tokens = jnp.broadcast_to(
            latent_tokens, (batch_size, seq_len, self.num_latents, self.model_dim)
        )
        x = jnp.concatenate([latent_tokens, x], axis=2)

        spatial_rope, temporal_rope = build_rope_embeddings(
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
            attention_logit_soft_cap=self.attention_logit_soft_cap,
            dtype=self.dtype,
        )(
            x=x,
            t=seq_len,
            total_tokens=num_tokens + self.num_latents,
            spatial_rope_emb=spatial_rope,
            spatial_mask=create_spatial_mask(
                num_tokens, self.num_latents, encoder=True
            ),
            temporal_rope_emb=temporal_rope,
            temporal_mask=temporal_mask,
        )

        latent = x[:, :, : self.num_latents, :]
        latent = nn.Dense(self.channel_dim, dtype=self.dtype)(latent)
        if self.bottleneck_norm == "none":
            pass
        elif self.bottleneck_norm == "tanh":
            latent = jnp.tanh(latent)
        elif self.bottleneck_norm == "rmsnorm":
            latent = nn.RMSNorm(dtype=self.dtype, name="bottleneck_rmsnorm")(latent)
        else:
            raise ValueError(f"Unknown bottleneck_norm: {self.bottleneck_norm}")
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
    single_image_token: bool = False
    attention_logit_soft_cap: float | None = 50.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        latent: jnp.ndarray,
        temporal_mask: jnp.ndarray,
        patch_dim: int,
    ) -> jnp.ndarray:
        batch_size, seq_len, num_latents, _ = latent.shape
        num_tokens = self.x_len * self.y_len

        image_token_count = 1 if self.single_image_token else num_tokens
        image_tokens = self.param(
            "image_tokens",
            nn.initializers.normal(stddev=0.02),
            (image_token_count, self.model_dim),
        ).astype(self.dtype)
        if self.single_image_token:
            image_tokens = jnp.broadcast_to(image_tokens, (num_tokens, self.model_dim))
        image_tokens = jnp.broadcast_to(
            image_tokens, (batch_size, seq_len, num_tokens, self.model_dim)
        )

        latent = nn.Dense(self.model_dim, dtype=self.dtype)(latent)
        x = jnp.concatenate([latent, image_tokens], axis=2)

        spatial_rope, temporal_rope = build_rope_embeddings(
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
            attention_logit_soft_cap=self.attention_logit_soft_cap,
            dtype=self.dtype,
        )(
            x=x,
            t=seq_len,
            total_tokens=num_tokens + self.num_latents,
            spatial_rope_emb=spatial_rope,
            spatial_mask=create_spatial_mask(
                num_tokens, self.num_latents, encoder=False
            ),
            temporal_rope_emb=temporal_rope,
            temporal_mask=temporal_mask,
        )

        x = x[:, :, self.num_latents :, :]
        x = nn.Dense(patch_dim, dtype=self.dtype)(x)
        return nn.sigmoid(x)


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
    decoder_single_image_token: bool = False
    bottleneck_norm: str = "tanh"
    independent_prob: float = 0.3
    mask_prob_min: float = 0.0
    mask_prob_max: float = 0.9
    attention_logit_soft_cap: float | None = 50.0
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        encoder_kwargs = dict(
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
            bottleneck_norm=self.bottleneck_norm,
            attention_logit_soft_cap=self.attention_logit_soft_cap,
            dtype=self.dtype,
        )
        decoder_kwargs = dict(
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
            attention_logit_soft_cap=self.attention_logit_soft_cap,
            dtype=self.dtype,
        )
        self.encoder = TokenizerEncoder(**encoder_kwargs)
        self.decoder = TokenizerDecoder(
            **decoder_kwargs,
            single_image_token=self.decoder_single_image_token,
        )

    def sample_independent(self, batch_size: int) -> jnp.ndarray:
        rng = self.make_rng("sample")
        return jax.random.bernoulli(rng, p=self.independent_prob, shape=(batch_size,))

    def sample_mask(
        self,
        video_shape: tuple[int, int, int],
        mask_prob: float | None = None,
    ) -> jnp.ndarray:
        batch_size, seq_len, num_tokens = video_shape
        rng = self.make_rng("sample")

        if mask_prob is None:
            prob_rng, rng = jax.random.split(rng)
            mask_prob = jax.random.uniform(
                prob_rng,
                shape=(batch_size, seq_len),
                minval=self.mask_prob_min,
                maxval=self.mask_prob_max,
            )
        else:
            mask_prob = jnp.full((batch_size, seq_len), mask_prob)
        rand_vals = jax.random.uniform(
            rng, shape=(batch_size, seq_len, num_tokens)
        )
        return rand_vals < jnp.expand_dims(mask_prob, axis=-1)

    def __call__(
        self,
        batch: PreprocessedVideoDataset,
        mask_prob: float | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, seq_len, patch_len, patch_dim = batch["video"].shape

        independent = self.sample_independent(batch_size)
        mask = self.sample_mask((batch_size, seq_len, patch_len), mask_prob=mask_prob)
        temporal_mask = create_temporal_mask(independent, seq_len)

        video = jnp.asarray(batch["video"], dtype=jnp.float32) / 255.0
        latent = self.encoder(video, temporal_mask, mask=mask)
        reconstructed = self.decoder(latent, temporal_mask, patch_dim)
        return reconstructed, mask

    def reconstruct_eval(
        self, batch: PreprocessedVideoDataset
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, seq_len, patch_len, patch_dim = batch["video"].shape

        independent = jnp.zeros((batch_size,), dtype=bool)
        mask = self.sample_mask((batch_size, seq_len, patch_len), mask_prob=0.1)
        temporal_mask = create_temporal_mask(independent, seq_len)

        video = jnp.asarray(batch["video"], dtype=jnp.float32) / 255.0
        latent = self.encoder(video, temporal_mask, mask=mask)
        reconstructed = self.decoder(latent, temporal_mask, patch_dim)
        return reconstructed, mask

    def encode(self, batch: PreprocessedVideoDataset) -> jnp.ndarray:
        batch_size, seq_len, _, _ = batch["video"].shape
        temporal_mask = create_temporal_mask(
            jnp.zeros((batch_size,), dtype=bool), seq_len
        )
        video = jnp.asarray(batch["video"], dtype=jnp.float32) / 255.0
        return self.encoder(video, temporal_mask)

    def decode(self, latent: jnp.ndarray, patch_dim: int) -> jnp.ndarray:
        batch_size, seq_len, _, _ = latent.shape
        temporal_mask = create_temporal_mask(
            jnp.zeros((batch_size,), dtype=bool), seq_len
        )
        return self.decoder(latent, temporal_mask, patch_dim)
