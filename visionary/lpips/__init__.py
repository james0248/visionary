import jax
import jax.numpy as jnp
from flax import nnx

from jaxlpips.utils import load_model


class LPIPS(nnx.Module):
    def __init__(self, pretrained_network: str = "alexnet"):
        if pretrained_network == "alex":
            pretrained_network = "alexnet"
        if pretrained_network == "vgg":
            pretrained_network = "vgg16"

        valid_networks = ["alexnet", "vgg16"]
        if pretrained_network not in valid_networks:
            raise ValueError("Only 'alexnet' and 'vgg16' pretrained networks are supported.")

        self.pretrained_network = pretrained_network
        self.model, self.linear_weights = load_model(self.pretrained_network)

    def __call__(self, ref: jax.Array, tgt: jax.Array):
        if ref.shape != tgt.shape:
            raise ValueError(
                f"Expected LPIPS inputs to have the same shape, got {ref.shape} and {tgt.shape}"
            )

        combined = scale_shift(jnp.concatenate([ref, tgt], axis=0))
        combined_feats = self.model(combined)
        batch_size = ref.shape[0]

        layer_dists = []
        for pair_feats, w in zip(combined_feats, self.linear_weights, strict=True):
            ref_ft = pair_feats[:batch_size]
            tgt_ft = pair_feats[batch_size:]
            layer_dists.append(
                jnp.mean(
                    jnp.sum(
                        ((normalise(ref_ft) - normalise(tgt_ft)) ** 2) * w, axis=3, keepdims=True
                    ),
                    axis=(1, 2),
                    keepdims=True,
                )
            )
        dist_score = jnp.sum(jnp.stack(layer_dists, axis=0), axis=0)
        return dist_score


def scale_shift(x: jax.Array):
    shift = jnp.array([-0.030, -0.088, -0.188])[None, None, None, :]
    scale = jnp.array([0.458, 0.448, 0.450])[None, None, None, :]
    return (x - shift) / scale


def normalise(x: jax.Array, eps=1e-10):
    norm_factor = jnp.sqrt(jnp.sum(x**2, axis=3, keepdims=True))
    return x / (norm_factor + eps)
