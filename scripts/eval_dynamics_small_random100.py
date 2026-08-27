"""Tokenize and roll out the saved SO-101 dynamics_small model on random100."""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import subprocess
import time
from pathlib import Path

import grain.python as grain
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from etils import epath
from hydra.utils import instantiate
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from visionary.common.checkpoint import (
    restore_model_export_single_device,
    restore_preprocessor_export,
)
from visionary.dataset import decode_video_window
from visionary.models.dreamer4.dynamics import DynamicsModel
from visionary.models.dreamer4.tokenizer import Tokenizer
from visionary.models.dreamer4.tokenizer_preprocessor import TokenizerPreprocessor

logger = logging.getLogger(__name__)

ACTION_Q01 = np.asarray(
    [
        -42.26002521871019,
        45.0397494636546,
        35.07994561935998,
        5.092905081995654,
        -65.86201352805867,
        -0.30093924505488123,
    ],
    dtype=np.float32,
)
ACTION_Q99 = np.asarray(
    [
        48.62998446685902,
        186.13242897117001,
        173.63312190159667,
        93.48411058383967,
        43.53720968938328,
        44.68202021420624,
    ],
    dtype=np.float32,
)

EXPECTED_DYNAMICS = {
    "_target_": "visionary.models.dreamer4.dynamics.DynamicsModel",
    "num_layers": 12,
    "num_heads": 12,
    "num_kv_heads": 3,
    "temporal_layer_period": 4,
    "num_registers": 2,
    "num_obs_tokens": 160,
    "num_actions": 6,
    "action_mode": "continuous",
    "max_step_size": 6,
    "model_dim": 1024,
    "head_dim": 64,
    "mlp_hidden_dim": 3072,
    "context_length": 24,
}
EXPECTED_TOKENIZER = {
    "_target_": "visionary.models.dreamer4.tokenizer.Tokenizer",
    "num_layers": 12,
    "num_latents": 160,
    "model_dim": 768,
    "channel_dim": 16,
    "resize_shape": [240, 320],
    "patch_size": 16,
    "bounded_latent": False,
}


def scalar(value) -> object:
    value = np.asarray(value).item()
    return value.decode() if isinstance(value, bytes) else value


def stream_key(stream: str) -> tuple[str, int, str]:
    prefix, camera = stream.rsplit("/", 1)
    repo, episode = prefix.rsplit("/ep", 1)
    return repo, int(episode), camera


def normalize_actions(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[-1] != len(ACTION_Q01):
        raise ValueError(f"Expected six actions, got {actions.shape}")
    return np.clip(2.0 * (actions - ACTION_Q01) / (ACTION_Q99 - ACTION_Q01) - 1.0, -1, 1)


def resample_indices(length: int, source_fps: float, target_fps: float = 30.0) -> np.ndarray:
    if target_fps >= source_fps or abs(source_fps - target_fps) < 1e-6:
        return np.arange(length, dtype=np.int64)
    step = source_fps / target_fps
    return np.unique(np.floor(np.arange(0, length, step)).astype(np.int64))


def validate_config(config, expected: dict[str, object], label: str) -> None:
    actual = {key: config.get(key) for key in expected}
    mismatches = {
        key: (expected[key], actual[key]) for key in expected if actual[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"{label} export does not match the saved run: {mismatches}")
    logger.info("Validated saved %s config: %s", label, actual)


def shard_source(uri: str) -> grain.ArrayRecordDataSource:
    paths = sorted(
        (path.as_posix() for path in epath.Path(uri).iterdir() if path.suffix == ".arecord")
    )
    if not paths:
        raise FileNotFoundError(f"No ArrayRecord shards at {uri}")
    logger.info("Opening %d raw eval shards", len(paths))
    return grain.ArrayRecordDataSource(paths)


def load_selected_raw(
    raw_shards: str,
    wanted: set[tuple[str, int, str]],
) -> dict[tuple[str, int, str], dict[str, object]]:
    source = shard_source(raw_shards)
    found: dict[tuple[str, int, str], dict[str, object]] = {}
    started = time.monotonic()
    for index in range(len(source)):
        with np.load(io.BytesIO(source[index])) as data:
            key = (
                str(scalar(data["repo"])),
                int(scalar(data["episode"])),
                str(scalar(data["camera"])),
            )
            if key in wanted:
                found[key] = {
                    "video": data["video"].tobytes(),
                    "length": int(scalar(data["length"])),
                    "fps": float(scalar(data["fps"])),
                    "actions": np.asarray(data["actions"], dtype=np.float32),
                }
        if len(found) == len(wanted):
            break
        if (index + 1) % 500 == 0:
            logger.info(
                "Scanned %d/%d records; matched %d/%d",
                index + 1,
                len(source),
                len(found),
                len(wanted),
            )
    missing = wanted - found.keys()
    if missing:
        raise KeyError(f"Missing {len(missing)} selected streams: {sorted(missing)[:10]}")
    logger.info("Matched all %d streams in %.1fs", len(found), time.monotonic() - started)
    return found


def sync(local_dir: Path, uri: str) -> None:
    subprocess.run(
        ["gcloud", "storage", "rsync", "--recursive", local_dir.as_posix(), uri],
        check=True,
    )


def fetch_existing(uri: str, local_dir: Path) -> None:
    result = subprocess.run(
        ["gcloud", "storage", "ls", uri],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        local_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["gcloud", "storage", "rsync", "--recursive", uri, local_dir.as_posix()],
            check=True,
        )


class TokenizerRuntime:
    def __init__(self, checkpoint_dir: str, step: int, mesh: Mesh, batch_sharding: NamedSharding):
        config, variables = restore_model_export_single_device(checkpoint_dir, step=step)
        config["bounded_latent"] = False
        validate_config(config, EXPECTED_TOKENIZER, "tokenizer")
        self.model = instantiate(config)
        self.preprocessor = TokenizerPreprocessor.from_config(
            restore_preprocessor_export(checkpoint_dir, step=step)
        )
        self.variables = jax.device_put(variables, NamedSharding(mesh, P()))
        self.batch_sharding = batch_sharding

        model = self.model

        @jax.jit
        def encode_step(variables, patches):
            return model.apply(variables, {"video": patches}, method=Tokenizer.encode)

        @jax.jit
        def decode_step(variables, latents):
            patches = model.apply(variables, latents, method=Tokenizer.decode)
            return self.preprocessor.patches_to_images(patches).astype(jnp.float32)

        self.encode_step = encode_step
        self.decode_step = decode_step

    def encode(self, frames: np.ndarray, window: int, batch_size: int) -> np.ndarray:
        latents = np.empty(
            (len(frames), self.model.num_latents, self.model.channel_dim), np.float16
        )
        refs = [
            (start, min(start + window, len(frames))) for start in range(0, len(frames), window)
        ]
        for batch_start in range(0, len(refs), batch_size):
            batch = np.zeros((batch_size, window, *frames.shape[1:]), dtype=np.uint8)
            active = refs[batch_start : batch_start + batch_size]
            for row, (start, stop) in enumerate(active):
                batch[row, : stop - start] = frames[start:stop]
            patches = self.preprocessor.preprocess_video(batch)
            encoded = np.asarray(
                jax.device_get(
                    self.encode_step(self.variables, jax.device_put(patches, self.batch_sharding))
                ),
                dtype=np.float32,
            )
            for row, (start, stop) in enumerate(active):
                latents[start:stop] = encoded[row, : stop - start].astype(np.float16)
        return latents

    def decode(self, latents: np.ndarray, window: int = 32) -> np.ndarray:
        pieces = []
        for start in range(0, latents.shape[1], window):
            piece = latents[:, start : start + window]
            real_length = piece.shape[1]
            if real_length < window:
                piece = np.pad(piece, ((0, 0), (0, window - real_length), (0, 0), (0, 0)))
            decoded = np.asarray(
                jax.device_get(
                    self.decode_step(
                        self.variables,
                        jax.device_put(piece.astype(np.float32), self.batch_sharding),
                    )
                )
            )[:, :real_length]
            pieces.append(decoded)
        return np.concatenate(pieces, axis=1)


def tokenize_selected(
    manifest: list[dict[str, object]],
    raw: dict[tuple[str, int, str], dict[str, object]],
    runtime: TokenizerRuntime,
    output_dir: Path,
    output_uri: str,
    max_rollout_frames: int,
    stride: int,
    encode_window: int,
    batch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(manifest)
    max_source_frames = (max_rollout_frames - 1) * stride + 1
    completed = 0
    for item in manifest:
        index = int(item["index"])
        output = output_dir / f"clip_{index:06d}.npz"
        if output.exists():
            completed += 1
            continue
        key = stream_key(str(item["stream"]))
        record = raw[key]
        source_fps = float(record["fps"])
        source_length = min(int(record["length"]), len(record["actions"]))
        if source_fps > 30.0:
            decode_length = min(
                source_length,
                int(math.ceil((max_source_frames - 1) * source_fps / 30.0)) + 2,
            )
        else:
            decode_length = min(source_length, max_source_frames)
        frames = decode_video_window(
            record["video"], 0, decode_length, tuple(runtime.preprocessor.resize_shape)
        )
        usable = min(len(frames), len(record["actions"]))
        sample = resample_indices(usable, source_fps)[:max_source_frames]
        frames = frames[sample]
        actions = normalize_actions(np.asarray(record["actions"])[sample])
        latents = runtime.encode(frames, encode_window, batch_size)
        np.savez(
            output,
            frames=latents,
            actions=actions,
            prev_action=np.zeros(actions.shape[1:], dtype=np.float32),
            repo=np.asarray(key[0]),
            episode=np.asarray(key[1], dtype=np.int32),
            camera=np.asarray(key[2]),
            source_fps=np.asarray(source_fps, dtype=np.float32),
            pre_stride_fps=np.asarray(min(source_fps, 30.0), dtype=np.float32),
            source_index=np.asarray(index, dtype=np.int32),
            start_index=np.asarray(0, dtype=np.int32),
        )
        completed += 1
        logger.info(
            "Tokenized %d/%d: index=%d, %d source frames",
            completed,
            total,
            index,
            len(latents),
        )
        if completed % 10 == 0:
            sync(output_dir, output_uri)
    sync(output_dir, output_uri)


def prepare_rollout(path: Path, max_frames: int, stride: int) -> dict[str, object]:
    with np.load(path) as data:
        latents = np.asarray(data["frames"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        prev_action = np.asarray(data["prev_action"], dtype=np.float32)
        source_fps = float(scalar(data["source_fps"]))
        pre_stride_fps = float(scalar(data["pre_stride_fps"]))
        key = (str(scalar(data["repo"])), int(scalar(data["episode"])), str(scalar(data["camera"])))
        index = int(scalar(data["source_index"]))
    indices = np.arange(0, len(latents), stride, dtype=np.int64)[:max_frames]
    video = latents[indices]
    aligned = actions[indices - 1].copy()
    aligned[0] = prev_action
    return {
        "index": index,
        "key": key,
        "video": video,
        "actions": aligned,
        "length": len(video),
        "source_fps": source_fps,
        "fps": pre_stride_fps / stride,
    }


def pad_batch(samples: list[dict[str, object]], batch_size: int, frames: int):
    video_shape = samples[0]["video"].shape[1:]
    action_shape = samples[0]["actions"].shape[1:]
    videos = np.zeros((batch_size, frames, *video_shape), dtype=np.float32)
    actions = np.zeros((batch_size, frames, *action_shape), dtype=np.float32)
    seeds = np.zeros((batch_size,), dtype=np.int32)
    for row in range(batch_size):
        sample = samples[min(row, len(samples) - 1)]
        length = int(sample["length"])
        videos[row, :length] = sample["video"]
        actions[row, :length] = sample["actions"]
        if length < frames:
            videos[row, length:] = sample["video"][-1]
            actions[row, length:] = sample["actions"][-1]
        seeds[row] = 42 + int(sample["index"])
    return videos, actions, seeds


def to_u8(images: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(images * 255.0), 0, 255).astype(np.uint8)


def write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def video_paths(output_dir: Path, index: int, sample_steps: int) -> tuple[Path, Path, Path]:
    stem = f"d{index:06d}"
    return (
        output_dir / "decoded" / f"{stem}__decoded.mp4",
        output_dir / "rollout" / f"{stem}__dynamics_small_s{sample_steps}.mp4",
        output_dir / "side_by_side" / f"{stem}__decoded__dynamics_small.mp4",
    )


def score_video(
    sample: dict[str, object], reference: np.ndarray, prediction: np.ndarray, context_frames: int
) -> dict[str, object]:
    length = int(sample["length"])
    generated_slice = slice(context_frames, length)
    ref_f = reference[generated_slice].astype(np.float32) / 255.0
    pred_f = prediction[generated_slice].astype(np.float32) / 255.0
    psnr = float(peak_signal_noise_ratio(ref_f, pred_f, data_range=1.0))
    ssim = float(
        np.mean(
            [
                structural_similarity(a, b, data_range=1.0, channel_axis=-1)
                for a, b in zip(ref_f, pred_f, strict=True)
            ]
        )
    )
    key = sample["key"]
    return {
        "index": int(sample["index"]),
        "stream": f"{key[0]}/ep{key[1]}/{key[2]}",
        "frame_count": length,
        "source_fps": float(sample["source_fps"]),
        "rollout_fps": float(sample["fps"]),
        "psnr_vs_decoded": psnr,
        "ssim_vs_decoded": ssim,
    }


def rollout_all(
    manifest: list[dict[str, object]],
    tokenized_dir: Path,
    output_dir: Path,
    output_uri: str,
    tokenizer: TokenizerRuntime,
    dynamics_checkpoint: str,
    dynamics_step: int,
    mesh: Mesh,
    batch_sharding: NamedSharding,
    batch_size: int,
    max_frames: int,
    stride: int,
    context_frames: int,
    sample_steps: int,
    context_tau: float,
) -> list[dict[str, object]]:
    samples = [
        prepare_rollout(tokenized_dir / f"clip_{int(item['index']):06d}.npz", max_frames, stride)
        for item in manifest
    ]
    if min(int(sample["length"]) for sample in samples) <= context_frames:
        raise ValueError("A selected trajectory is too short for the context")

    config, variables = restore_model_export_single_device(dynamics_checkpoint, step=dynamics_step)
    validate_config(config, EXPECTED_DYNAMICS, "dynamics")
    model = instantiate(config)
    variables = jax.device_put(variables, NamedSharding(mesh, P()))

    @jax.jit
    def generate(variables, video, actions, seeds):
        primed = jnp.zeros_like(video).at[:, :context_frames].set(video[:, :context_frames])
        keys = jax.vmap(jax.random.PRNGKey)(seeds)
        context_keys, sample_keys = jax.vmap(jax.random.split)(keys).transpose((1, 0, 2))
        context_noise = jax.vmap(
            lambda key: jax.random.normal(key, video.shape[1:], dtype=jnp.float32)
        )(context_keys)
        sample_noise = jax.vmap(
            lambda key: jax.random.normal(
                key,
                (max_frames - context_frames, *video.shape[2:]),
                dtype=jnp.float32,
            )
        )(sample_keys)
        return model.apply(
            variables,
            primed,
            actions,
            context_noise,
            sample_noise,
            context_frames,
            context_tau=context_tau,
            sample_steps=sample_steps,
            method=DynamicsModel.generate_rollout,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for start in range(0, len(samples), batch_size):
        active = samples[start : start + batch_size]
        paths = [video_paths(output_dir, int(sample["index"]), sample_steps) for sample in active]
        if all(all(path.exists() for path in sample_paths) for sample_paths in paths):
            for sample, sample_paths in zip(active, paths, strict=True):
                reference = np.asarray(imageio.mimread(sample_paths[0]))[: int(sample["length"])]
                prediction = np.asarray(imageio.mimread(sample_paths[1]))[: int(sample["length"])]
                summary.append(score_video(sample, reference, prediction, context_frames))
            logger.info(
                "Reused retained rollouts through clip %d/%d", start + len(active), len(samples)
            )
            continue
        videos, actions, seeds = pad_batch(active, batch_size, max_frames)
        started = time.monotonic()
        generated = np.asarray(
            jax.device_get(
                generate(
                    variables,
                    jax.device_put(videos, batch_sharding),
                    jax.device_put(actions, batch_sharding),
                    jax.device_put(seeds, batch_sharding),
                )
            )
        )
        decoded = to_u8(tokenizer.decode(videos))
        rolled = to_u8(tokenizer.decode(generated))
        for row, sample in enumerate(active):
            length = int(sample["length"])
            index = int(sample["index"])
            reference = decoded[row, :length]
            prediction = rolled[row, :length]
            fps = float(sample["fps"])
            decoded_path, rollout_path, side_path = video_paths(output_dir, index, sample_steps)
            write_video(decoded_path, reference, fps)
            write_video(rollout_path, prediction, fps)
            separator = np.full((length, reference.shape[1], 4, 3), 255, dtype=np.uint8)
            side_by_side = np.concatenate([reference, separator, prediction], axis=2)
            write_video(side_path, side_by_side, fps)
            summary.append(score_video(sample, reference, prediction, context_frames))
        (output_dir / "partial_summary.json").write_text(json.dumps({"videos": summary}, indent=2))
        logger.info(
            "Rolled out %d/%d clips in %.1fs",
            min(start + len(active), len(samples)),
            len(samples),
            time.monotonic() - started,
        )
        sync(output_dir, output_uri)
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw_shards", required=True)
    parser.add_argument("--tokenizer_checkpoint", required=True)
    parser.add_argument("--tokenizer_step", type=int, required=True)
    parser.add_argument("--dynamics_checkpoint", required=True)
    parser.add_argument("--dynamics_step", type=int, required=True)
    parser.add_argument("--tokenized_dir", required=True)
    parser.add_argument("--tokenized_uri", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_uri", required=True)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--encode_window", type=int, default=32)
    parser.add_argument("--context_frames", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=8)
    parser.add_argument("--context_tau", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    if jax.process_count() != 1:
        raise RuntimeError("This evaluator supports one TPU host")
    device_count = jax.local_device_count()
    if device_count < 1:
        raise RuntimeError("No JAX devices found")
    logger.info("Using %d local devices", device_count)

    manifest_payload = json.loads(Path(args.manifest).read_text())
    manifest = manifest_payload["videos"]
    if len(manifest) != 100 or len({item["stream"] for item in manifest}) != 100:
        raise ValueError("The manifest must contain the exact 100 unique streams")
    if not 1 <= args.limit <= len(manifest):
        raise ValueError(f"Expected limit in [1, {len(manifest)}], got {args.limit}")
    manifest = manifest[: args.limit]

    tokenized_dir = Path(args.tokenized_dir)
    output_dir = Path(args.output_dir)
    fetch_existing(args.tokenized_uri, tokenized_dir)
    fetch_existing(args.output_uri, output_dir)
    completed = {int(path.stem.rsplit("_", 1)[1]) for path in tokenized_dir.glob("clip_*.npz")}
    missing_manifest = [item for item in manifest if int(item["index"]) not in completed]

    mesh = Mesh(np.asarray(jax.local_devices()), ("data",))
    batch_sharding = NamedSharding(mesh, P("data"))
    tokenizer = TokenizerRuntime(
        args.tokenizer_checkpoint,
        args.tokenizer_step,
        mesh,
        batch_sharding,
    )
    if missing_manifest:
        wanted = {stream_key(str(item["stream"])) for item in missing_manifest}
        raw = load_selected_raw(args.raw_shards, wanted)
        tokenize_selected(
            missing_manifest,
            raw,
            tokenizer,
            tokenized_dir,
            args.tokenized_uri,
            args.max_frames,
            args.stride,
            args.encode_window,
            device_count,
        )
    logger.info("All %d selected eval streams are tokenized", len(manifest))

    summary = rollout_all(
        manifest,
        tokenized_dir,
        output_dir,
        args.output_uri,
        tokenizer,
        args.dynamics_checkpoint,
        args.dynamics_step,
        mesh,
        batch_sharding,
        device_count,
        args.max_frames,
        args.stride,
        args.context_frames,
        args.sample_steps,
        args.context_tau,
    )
    summary.sort(key=lambda item: int(item["index"]))
    payload = {
        "dynamics_checkpoint": args.dynamics_checkpoint,
        "dynamics_step": args.dynamics_step,
        "tokenizer_checkpoint": args.tokenizer_checkpoint,
        "tokenizer_step": args.tokenizer_step,
        "saved_model_config": EXPECTED_DYNAMICS,
        "historical_preprocessing": {
            "target_fps": 30,
            "stride": args.stride,
            "encode_window": args.encode_window,
            "action_norm": "MolmoAct2 SO100/SO101 q01-q99",
        },
        "context_frames": args.context_frames,
        "sample_steps": args.sample_steps,
        "context_tau": args.context_tau,
        "max_frames": args.max_frames,
        "mean_psnr_vs_decoded": float(np.mean([item["psnr_vs_decoded"] for item in summary])),
        "mean_ssim_vs_decoded": float(np.mean([item["ssim_vs_decoded"] for item in summary])),
        "videos": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    sync(output_dir, args.output_uri)
    logger.info("Complete: %d videos", len(summary))


if __name__ == "__main__":
    main()
