import jax
import jax.numpy as jnp


def epps_pulley_statistic(x: jnp.ndarray, knots: int) -> jnp.ndarray:
    """Epps-Pulley statistic of x: (samples, projections) against N(0, 1)."""
    t = jnp.linspace(0.0, 3.0, knots, dtype=jnp.float32)
    dt = 3.0 / (knots - 1)
    weights = jnp.full((knots,), 2.0 * dt, dtype=jnp.float32)
    weights = weights.at[jnp.array([0, knots - 1])].set(dt)
    phi = jnp.exp(-jnp.square(t) / 2.0)

    x_t = x[..., None] * t
    err = jnp.square(jnp.mean(jnp.cos(x_t), axis=-3) - phi) + jnp.square(
        jnp.mean(jnp.sin(x_t), axis=-3)
    )
    return (err @ (weights * phi)) * x.shape[-2]


def sigreg_loss(
    z: jnp.ndarray,
    rng: jax.Array,
    knots: int = 17,
    num_proj: int = 1024,
) -> jnp.ndarray:
    """SIGReg: pulls the pooled distribution of z toward N(0, I).

    Every leading axis of z is a sample axis. Projection directions are redrawn
    per call, so pass a per-step rng.
    """
    dim = z.shape[-1]
    samples = z.reshape(-1, dim).astype(jnp.float32)
    directions = jax.random.normal(rng, (dim, num_proj), dtype=jnp.float32)
    directions = directions / jnp.linalg.norm(directions, axis=0)

    # without remat the backward keeps the (samples, num_proj, knots) sin/cos
    def statistic(samples, directions):
        return jnp.mean(epps_pulley_statistic(samples @ directions, knots))

    return jax.checkpoint(statistic)(samples, directions)
