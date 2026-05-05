import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
from einops import rearrange
from hydra.utils import instantiate

from visionary.common.checkpoint import (
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.dynamics import DynamicsModel
from visionary.export.onnx_wrappers import (
    onnx_apply_dynamics_cached_sample_step,
    onnx_apply_dynamics_cached_step,
)
from visionary.tokenizer import Tokenizer
from visionary.tokenizer_preprocessor import TokenizerPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pure JAX rollout with exported ONNX rollout.")
    parser.add_argument("--episode", type=Path, default=Path("episode_0.npz"))
    parser.add_argument("--assets_dir", type=Path, default=Path("webgpu_app/assets"))
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--tokenizer_step", type=int, default=1000000)
    parser.add_argument("--dynamics_dir", required=True)
    parser.add_argument("--dynamics_step", type=int, default=1000000)
    parser.add_argument("--prefix_frames", type=int, default=4)
    parser.add_argument("--generated_frames", type=int, default=4)
    parser.add_argument("--context_tau", type=float, default=29 / 32)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("webgpu_app/bench/results/rollout_compare.json"))
    return parser.parse_args()


def pack_z(latents: np.ndarray, *, num_obs_tokens: int = 32) -> np.ndarray:
    return rearrange(latents, "b t (n k) d -> b t n (k d)", n=num_obs_tokens).astype(np.float32)


def unpack_z(z: np.ndarray, *, channel_dim: int = 16) -> np.ndarray:
    return rearrange(z, "b t n (k d) -> b t (n k) d", d=channel_dim).astype(np.float32)


def run_ort_named(session: ort.InferenceSession, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, feeds)
    return dict(zip(output_names, outputs))


def stats(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    delta = actual.astype(np.float32) - expected.astype(np.float32)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "actual_norm": float(np.linalg.norm(actual)),
        "expected_norm": float(np.linalg.norm(expected)),
    }


def compare_jax_cached_wrapper(
    *,
    dynamics_variables: Any,
    dynamics_cfg: Any,
    actions: np.ndarray,
    prefix_noised_z: np.ndarray,
    sample_noise_z: np.ndarray,
    generated_context_noise_z: np.ndarray,
    context_step_level: int,
    context_signal_level: int,
    context_tau_used: np.float32,
    prefix_frames: int,
    generated_frames: int,
    sample_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    cache_shape = (6, 1, 36, 64, 2, 64)
    cache = {
        "k": jnp.zeros(cache_shape, dtype=jnp.float32),
        "v": jnp.zeros(cache_shape, dtype=jnp.float32),
        "length": jnp.asarray([0], dtype=jnp.int32),
    }

    def append_context(cache_state: dict[str, jax.Array], z: np.ndarray, action: int) -> dict[str, jax.Array]:
        _, candidate_k, candidate_v, candidate_length = onnx_apply_dynamics_cached_step(
            dynamics_variables,
            dynamics_cfg,
            jnp.asarray(z),
            jnp.asarray([[action]], dtype=jnp.int32),
            jnp.asarray([[context_step_level]], dtype=jnp.int32),
            jnp.asarray([[context_signal_level]], dtype=jnp.int32),
            cache_state["length"],
            cache_state["k"],
            cache_state["v"],
            cache_state["length"],
        )
        return {
            "k": candidate_k,
            "v": candidate_v,
            "length": candidate_length,
        }

    for index in range(prefix_frames):
        cache = append_context(cache, prefix_noised_z[:, index : index + 1], int(actions[0, index]))

    sample_cache = cache
    append_cache = cache
    sample_commit = []
    context_append = []
    for offset in range(generated_frames):
        final_z, _, candidate_k, candidate_v, candidate_length = onnx_apply_dynamics_cached_sample_step(
            dynamics_variables,
            dynamics_cfg,
            jnp.asarray(sample_noise_z[:, offset : offset + 1]),
            jnp.asarray([[int(actions[0, prefix_frames + offset])]], dtype=jnp.int32),
            sample_cache["length"],
            sample_cache["k"],
            sample_cache["v"],
            sample_cache["length"],
            sample_steps=sample_steps,
        )
        sample_commit.append(np.asarray(jax.device_get(final_z), dtype=np.float32))
        sample_cache = {
            "k": candidate_k,
            "v": candidate_v,
            "length": candidate_length,
        }

        final_z_append, _, _, _, _ = onnx_apply_dynamics_cached_sample_step(
            dynamics_variables,
            dynamics_cfg,
            jnp.asarray(sample_noise_z[:, offset : offset + 1]),
            jnp.asarray([[int(actions[0, prefix_frames + offset])]], dtype=jnp.int32),
            append_cache["length"],
            append_cache["k"],
            append_cache["v"],
            append_cache["length"],
            sample_steps=sample_steps,
        )
        final_z_append_np = np.asarray(jax.device_get(final_z_append), dtype=np.float32)
        context_append.append(final_z_append_np)
        noised_generated_context = (
            context_tau_used * final_z_append_np
            + (np.float32(1.0) - context_tau_used)
            * generated_context_noise_z[:, offset : offset + 1]
        )
        append_cache = append_context(
            append_cache,
            noised_generated_context,
            int(actions[0, prefix_frames + offset]),
        )

    return np.concatenate(sample_commit, axis=1), np.concatenate(context_append, axis=1)


def main() -> None:
    args = parse_args()
    data = np.load(args.episode)
    frames = np.asarray(data["frames"][: args.prefix_frames + args.generated_frames])
    actions = np.asarray(data["actions"][: args.prefix_frames + args.generated_frames], dtype=np.int32)[
        None
    ]

    tokenizer_cfg, tokenizer_variables = restore_model_export_single_device(
        args.tokenizer_dir,
        step=args.tokenizer_step,
    )
    preprocessor_cfg = restore_preprocessor_export(args.tokenizer_dir, step=args.tokenizer_step)
    tokenizer = instantiate(tokenizer_cfg)
    preprocessor = TokenizerPreprocessor.from_config(preprocessor_cfg)
    patches = preprocessor.preprocess_video(frames)[None]

    @jax.jit
    def encode_step(variables, patch_batch):
        return tokenizer.apply(variables, {"video": patch_batch}, method=Tokenizer.encode)

    latents = np.asarray(jax.device_get(encode_step(tokenizer_variables, patches)), dtype=np.float32)

    dynamics_cfg, dynamics_variables = restore_model_export_single_device(
        args.dynamics_dir,
        step=args.dynamics_step,
    )
    dynamics = instantiate(dynamics_cfg)
    if not isinstance(dynamics, DynamicsModel):
        raise TypeError(f"Expected DynamicsModel, got {type(dynamics)!r}")

    rollout_key = jax.random.key(args.seed)
    context_noise_key, sample_noise_key = jax.random.split(rollout_key)
    context_noise = np.asarray(
        jax.random.normal(context_noise_key, latents.shape, dtype=jnp.float32),
        dtype=np.float32,
    )
    sample_noise = np.asarray(
        jax.random.normal(
            sample_noise_key,
            (1, args.generated_frames, *latents.shape[2:]),
            dtype=jnp.float32,
        ),
        dtype=np.float32,
    )

    video_prefix = np.zeros_like(latents)
    video_prefix[:, : args.prefix_frames] = latents[:, : args.prefix_frames]

    jax_rollout = np.asarray(
        jax.device_get(
            dynamics.apply(
                dynamics_variables,
                jnp.asarray(video_prefix),
                jnp.asarray(actions),
                jnp.asarray(context_noise),
                jnp.asarray(sample_noise),
                jnp.asarray(args.prefix_frames, dtype=jnp.int32),
                context_tau=args.context_tau,
                sample_steps=args.sample_steps,
                method=DynamicsModel.generate_rollout,
            )
        ),
        dtype=np.float32,
    )
    jax_generated_z = pack_z(
        jax_rollout[:, args.prefix_frames : args.prefix_frames + args.generated_frames]
    )

    step_session = ort.InferenceSession(
        (args.assets_dir / "breakout_dynamics_step_cached_b1_t1.onnx").as_posix(),
        providers=["CPUExecutionProvider"],
    )
    sample_session = ort.InferenceSession(
        (args.assets_dir / "breakout_dynamics_cached_sample_step_b1_t1_s4.onnx").as_posix(),
        providers=["CPUExecutionProvider"],
    )

    context_step_level = int(dynamics.max_step_size) - 1
    context_step_count = 1 << context_step_level
    context_signal_level = min(
        max(int(round(args.context_tau * context_step_count)), 0),
        context_step_count - 1,
    )
    context_tau_used = np.float32(context_signal_level / context_step_count)
    prefix_z = pack_z(latents[:, : args.prefix_frames])
    prefix_context_noise_z = pack_z(context_noise[:, : args.prefix_frames])
    prefix_noised_z = context_tau_used * prefix_z + (np.float32(1.0) - context_tau_used) * prefix_context_noise_z
    sample_noise_z = pack_z(sample_noise)
    generated_context_noise_z = pack_z(
        context_noise[:, args.prefix_frames : args.prefix_frames + args.generated_frames]
    )
    jax_cached_sample_commit_z, jax_cached_context_append_z = compare_jax_cached_wrapper(
        dynamics_variables=dynamics_variables,
        dynamics_cfg=dynamics_cfg,
        actions=actions,
        prefix_noised_z=prefix_noised_z,
        sample_noise_z=sample_noise_z,
        generated_context_noise_z=generated_context_noise_z,
        context_step_level=context_step_level,
        context_signal_level=context_signal_level,
        context_tau_used=context_tau_used,
        prefix_frames=args.prefix_frames,
        generated_frames=args.generated_frames,
        sample_steps=args.sample_steps,
    )

    zero_cache = np.zeros((6, 1, 36, 64, 2, 64), dtype=np.float32)

    def append_context(cache: dict[str, np.ndarray], z: np.ndarray, action: int) -> dict[str, np.ndarray]:
        outputs = run_ort_named(
            step_session,
            {
                "z": z.astype(np.float32),
                "actions": np.asarray([[action]], dtype=np.int32),
                "step_levels": np.asarray([[context_step_level]], dtype=np.int32),
                "signal_levels": np.asarray([[context_signal_level]], dtype=np.int32),
                "position_index": cache["length"],
                "k_cache": cache["k"],
                "v_cache": cache["v"],
                "cache_length": cache["length"],
            },
        )
        return {
            "k": outputs["candidate_k_cache"],
            "v": outputs["candidate_v_cache"],
            "length": outputs["candidate_cache_length"],
        }

    def initial_cache() -> dict[str, np.ndarray]:
        cache = {
            "k": zero_cache.copy(),
            "v": zero_cache.copy(),
            "length": np.asarray([0], dtype=np.int32),
        }
        for index in range(args.prefix_frames):
            cache = append_context(cache, prefix_noised_z[:, index : index + 1], int(actions[0, index]))
        return cache

    def sample_once(cache: dict[str, np.ndarray], offset: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        outputs = run_ort_named(
            sample_session,
            {
                "sample_noise": sample_noise_z[:, offset : offset + 1],
                "actions": np.asarray([[int(actions[0, args.prefix_frames + offset])]], dtype=np.int32),
                "position_index": cache["length"],
                "k_cache": cache["k"],
                "v_cache": cache["v"],
                "cache_length": cache["length"],
            },
        )
        sample_cache = {
            "k": outputs["candidate_k_cache"],
            "v": outputs["candidate_v_cache"],
            "length": outputs["candidate_cache_length"],
        }
        return outputs["final_z"], sample_cache

    sample_cache = initial_cache()
    append_cache = initial_cache()
    onnx_sample_commit = []
    onnx_context_append = []
    for offset in range(args.generated_frames):
        final_z, candidate_cache = sample_once(sample_cache, offset)
        onnx_sample_commit.append(final_z)
        sample_cache = candidate_cache

        final_z_append, _ = sample_once(append_cache, offset)
        onnx_context_append.append(final_z_append)
        noised_generated_context = (
            context_tau_used * final_z_append
            + (np.float32(1.0) - context_tau_used)
            * generated_context_noise_z[:, offset : offset + 1]
        )
        append_cache = append_context(
            append_cache,
            noised_generated_context,
            int(actions[0, args.prefix_frames + offset]),
        )

    sample_commit_z = np.concatenate(onnx_sample_commit, axis=1)
    context_append_z = np.concatenate(onnx_context_append, axis=1)
    result = {
        "prefix_frames": args.prefix_frames,
        "generated_frames": args.generated_frames,
        "context_tau_requested": args.context_tau,
        "context_tau_used": float(context_tau_used),
        "sample_steps": args.sample_steps,
        "seed": args.seed,
        "comparisons": {
            "jax_cached_commit_sample_cache_vs_jax_full_rollout": stats(
                jax_cached_sample_commit_z,
                jax_generated_z,
            ),
            "jax_cached_append_final_as_context_vs_jax_full_rollout": stats(
                jax_cached_context_append_z,
                jax_generated_z,
            ),
            "onnx_commit_sample_cache_vs_jax": stats(sample_commit_z, jax_generated_z),
            "onnx_append_final_as_context_vs_jax": stats(context_append_z, jax_generated_z),
            "onnx_commit_sample_cache_vs_jax_cached_commit_sample_cache": stats(
                sample_commit_z,
                jax_cached_sample_commit_z,
            ),
            "onnx_append_final_as_context_vs_jax_cached_append_final_as_context": stats(
                context_append_z,
                jax_cached_context_append_z,
            ),
        },
        "per_frame": [
            {
                "frame": int(index),
                "jax_cached_sample_cache": stats(
                    jax_cached_sample_commit_z[:, index],
                    jax_generated_z[:, index],
                ),
                "jax_cached_append_context": stats(
                    jax_cached_context_append_z[:, index],
                    jax_generated_z[:, index],
                ),
                "sample_cache": stats(sample_commit_z[:, index], jax_generated_z[:, index]),
                "append_context": stats(context_append_z[:, index], jax_generated_z[:, index]),
            }
            for index in range(args.generated_frames)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
