import jax
import jax.numpy as jnp
import numpy as np

from visionary.export.onnx_wrappers import _export_dot_product_attention


def test_grouped_gqa_matches_repeat_export_attention():
    key = jax.random.key(0)
    query_key, key_key, value_key = jax.random.split(key, 3)
    query = jax.random.normal(query_key, (2, 3, 8, 16), dtype=jnp.float32)
    key_states = jax.random.normal(key_key, (2, 5, 2, 16), dtype=jnp.float32)
    value_states = jax.random.normal(value_key, (2, 5, 2, 16), dtype=jnp.float32)

    expected = _export_dot_product_attention(
        query,
        key_states,
        value_states,
        scale=1.0 / np.sqrt(16),
        grouped_gqa=False,
    )
    actual = _export_dot_product_attention(
        query,
        key_states,
        value_states,
        scale=1.0 / np.sqrt(16),
        grouped_gqa=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_grouped_gqa_matches_repeat_export_attention_with_mask():
    key = jax.random.key(1)
    query_key, key_key, value_key = jax.random.split(key, 3)
    query = jax.random.normal(query_key, (1, 2, 8, 16), dtype=jnp.float32)
    key_states = jax.random.normal(key_key, (1, 5, 2, 16), dtype=jnp.float32)
    value_states = jax.random.normal(value_key, (1, 5, 2, 16), dtype=jnp.float32)
    mask = jnp.asarray([[[[True, True, True, False, False], [True, False, True, False, True]]]])

    expected = _export_dot_product_attention(
        query,
        key_states,
        value_states,
        mask=mask,
        scale=1.0 / np.sqrt(16),
        grouped_gqa=False,
    )
    actual = _export_dot_product_attention(
        query,
        key_states,
        value_states,
        mask=mask,
        scale=1.0 / np.sqrt(16),
        grouped_gqa=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_grouped_gqa_matches_repeat_export_attention_with_rank2_mask():
    key = jax.random.key(2)
    query_key, key_key, value_key = jax.random.split(key, 3)
    query = jax.random.normal(query_key, (4, 6, 8, 16), dtype=jnp.float32)
    key_states = jax.random.normal(key_key, (4, 6, 2, 16), dtype=jnp.float32)
    value_states = jax.random.normal(value_key, (4, 6, 2, 16), dtype=jnp.float32)
    mask = jnp.tril(jnp.ones((6, 6), dtype=bool))

    expected = _export_dot_product_attention(
        query,
        key_states,
        value_states,
        mask=mask,
        scale=1.0 / np.sqrt(16),
        grouped_gqa=False,
    )
    actual = _export_dot_product_attention(
        query,
        key_states,
        value_states,
        mask=mask,
        scale=1.0 / np.sqrt(16),
        grouped_gqa=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
