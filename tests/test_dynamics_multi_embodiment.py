import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from omegaconf import OmegaConf

from visionary.dataset import PackDynamicsEpisodes
from visionary.models.dreamer4.dynamics import ActionEmbedding, DynamicsModel
from visionary.models.dreamer4.onnx_wrappers import onnx_apply_dynamics_cached_prefill


class PackDynamicsEpisodesTest(unittest.TestCase):
    def test_packs_embodiment_ids_and_pads_actions(self):
        records = iter(
            [
                {
                    "video": np.arange(12, dtype=np.float32).reshape(3, 2, 2),
                    "actions": np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
                    "prev_action": np.asarray([-1.0, -2.0], dtype=np.float32),
                    "embodiment_id": 0,
                    "action_dim": 2,
                },
                {
                    "video": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
                    "actions": np.asarray([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32),
                    "prev_action": np.asarray([-3.0, -4.0, -5.0], dtype=np.float32),
                    "embodiment_id": 1,
                    "action_dim": 3,
                },
            ]
        )
        batch = next(iter(PackDynamicsEpisodes(records, sequence_length=4, batch_size=1, max_action_dim=3)))

        np.testing.assert_array_equal(batch["embodiment_ids"], [[0, 0, 0, 1]])
        np.testing.assert_array_equal(batch["segment_ids"], [[0, 0, 0, 1]])
        np.testing.assert_allclose(
            batch["actions"],
            [[[-1.0, -2.0, 0.0], [1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [-3.0, -4.0, -5.0]]],
        )


class ActionEmbeddingTest(unittest.TestCase):
    def test_selects_one_projection_per_token(self):
        module = ActionEmbedding(
            model_dim=4,
            num_actions=1,
            num_embodiments=2,
            max_action_dim=3,
            action_mode="continuous",
            dtype=jnp.float32,
        )
        actions = jnp.asarray([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        embodiment_ids = jnp.asarray([[0, 1]], dtype=jnp.int32)
        variables = unfreeze(module.init(jax.random.key(0), actions, embodiment_ids))
        np.testing.assert_array_equal(
            variables["params"]["action_projection_bias"],
            np.zeros((2, 4), dtype=np.float32),
        )
        kernels = jnp.zeros((2, 3, 4), dtype=jnp.float32)
        kernels = kernels.at[0, 0, 0].set(1.0)
        kernels = kernels.at[1, 0, 1].set(1.0)
        variables["params"]["action_projection_kernel"] = kernels

        output = module.apply(freeze(variables), actions, embodiment_ids)
        np.testing.assert_allclose(np.asarray(output[0, 0] - output[0, 1]), [1.0, -1.0, 0.0, 0.0])


class DynamicsModelTest(unittest.TestCase):
    def setUp(self):
        self.model = DynamicsModel(
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            num_registers=1,
            num_obs_tokens=2,
            num_actions=3,
            num_embodiments=2,
            max_action_dim=3,
            max_step_size=3,
            model_dim=8,
            head_dim=4,
            mlp_hidden_dim=16,
            context_length=2,
            latent_mean=(0.0, 0.0),
            latent_std=(1.0, 1.0),
            action_mode="continuous",
            temporal_layer_period=2,
            dtype=jnp.float32,
        )
        self.z = jnp.zeros((1, 2, 2, 2), dtype=jnp.float32)
        self.actions = jnp.zeros((1, 2, 3), dtype=jnp.float32)
        self.embodiment_ids = jnp.asarray([[0, 1]], dtype=jnp.int32)
        levels = jnp.zeros((1, 2), dtype=jnp.int32)
        self.variables = self.model.init(
            jax.random.key(1),
            self.z,
            self.actions,
            self.embodiment_ids,
            levels,
            levels,
            jnp.asarray([[0, 1]], dtype=jnp.int32),
        )

    def test_loss_accepts_packed_embodiments(self):
        batch = {
            "video": self.z,
            "actions": self.actions,
            "embodiment_ids": self.embodiment_ids,
            "segment_ids": jnp.asarray([[0, 1]], dtype=jnp.int32),
        }
        loss, metrics = self.model.apply(
            self.variables,
            batch,
            bootstrap_target_variables=self.variables,
            bootstrap_rows=1,
            image_fraction=0.5,
            method=DynamicsModel.loss,
            rngs={"sample": jax.random.key(2)},
        )
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertEqual(
            set(metrics),
            {
                "loss",
                "flow_loss",
                "flow_mse",
                "flow_mse_low",
                "flow_mse_mid",
                "flow_mse_high",
                "bootstrap_loss",
                "bootstrap_target_norm",
            },
        )

    def test_image_rows_train_without_temporal_transitions(self):
        batch = {
            "video": jnp.ones_like(self.z),
            "actions": self.actions,
            "embodiment_ids": self.embodiment_ids,
            "segment_ids": jnp.asarray([[0, 1]], dtype=jnp.int32),
        }
        no_image_loss, _ = self.model.apply(
            self.variables,
            batch,
            bootstrap_target_variables=self.variables,
            bootstrap_rows=0,
            image_fraction=0.0,
            method=DynamicsModel.loss,
            rngs={"sample": jax.random.key(2)},
        )
        image_loss, _ = self.model.apply(
            self.variables,
            batch,
            bootstrap_target_variables=self.variables,
            bootstrap_rows=0,
            image_fraction=1.0,
            method=DynamicsModel.loss,
            rngs={"sample": jax.random.key(2)},
        )
        self.assertEqual(float(no_image_loss), 0.0)
        self.assertGreater(float(image_loss), 0.0)

    def test_generation_returns_per_step_diagnostics(self):
        frame, diagnostics = self.model.apply(
            self.variables,
            self.z,
            self.actions,
            self.embodiment_ids,
            jnp.zeros_like(self.z),
            jnp.zeros((1, 2, 2), dtype=jnp.float32),
            jnp.asarray(1, dtype=jnp.int32),
            context_tau=0.9,
            sample_steps=2,
            clean_until=1,
            method=DynamicsModel.generate_next,
        )
        self.assertEqual(frame.shape, (1, 2, 2))
        self.assertEqual(diagnostics["x0_norm"].shape, (2,))
        self.assertEqual(diagnostics["update_mag"].shape, (2,))

    def test_rejects_context_tau_outside_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "context_tau must be in"):
            self.model.apply(
                self.variables,
                self.z,
                self.actions,
                self.embodiment_ids,
                jnp.zeros_like(self.z),
                jnp.zeros((1, 2, 2), dtype=jnp.float32),
                jnp.asarray(1, dtype=jnp.int32),
                context_tau=1.1,
                sample_steps=2,
                clean_until=1,
                method=DynamicsModel.generate_next,
            )

    def test_cached_onnx_wrapper_uses_model_parameter_names(self):
        cfg = OmegaConf.create(
            {
                "num_layers": 2,
                "num_heads": 2,
                "num_kv_heads": 1,
                "num_registers": 1,
                "num_obs_tokens": 2,
                "num_actions": 3,
                "num_embodiments": 2,
                "max_action_dim": 3,
                "max_step_size": 3,
                "model_dim": 8,
                "head_dim": 4,
                "mlp_hidden_dim": 16,
                "context_length": 2,
                "action_mode": "continuous",
                "temporal_layer_period": 2,
                "base": 10000.0,
            }
        )
        levels = jnp.zeros((1, 2), dtype=jnp.int32)
        prediction, *_ = onnx_apply_dynamics_cached_prefill(
            self.variables,
            cfg,
            self.z,
            self.actions,
            levels,
            levels,
            dtype=jnp.float32,
        )
        self.assertEqual(prediction.shape, self.z.shape)


if __name__ == "__main__":
    unittest.main()
