import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COORD = "localhost:29511"
NUM_PROCESSES = 4
DEVICES_PER_PROCESS = 4


def child(args):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={DEVICES_PER_PROCESS}"
    )
    import jax

    jax.distributed.initialize(
        coordinator_address=COORD,
        num_processes=NUM_PROCESSES,
        process_id=args.process_id,
    )
    sys.argv = [
        "train_tokenizer.py",
        "--config-name",
        "so101_tokenizer",
        *args.overrides,
    ]
    runpy.run_path(str(REPO / "scripts" / "train_tokenizer.py"), run_name="__main__")


def build_overrides(data_dir: Path, ckpt_dir: Path, total_steps: int) -> list[str]:
    return [
        f"dataset.train_dir={data_dir / 'train'}",
        f"dataset.eval_dir={data_dir / 'eval'}",
        f"checkpoint.manager.directory={ckpt_dir}",
        "checkpoint.resume_step=latest",
        "checkpoint.manager.options.save_interval_steps=5",
        "checkpoint.export_interval_steps=5",
        f"total_steps={total_steps}",
        "eval_steps=3",
        "log_interval=1",
        "wandb.enabled=false",
        "dataset.batch_size=16",
        "dataset.frame_length=8",
        "dataset.worker_count=0",
        "dataset.num_threads=2",
        "dataset.prefetch_buffer_size=2",
        "dataset.eval.max_batches=2",
        "dataset.eval.log_frames=2",
        "fsdp.data_axis_size=16",
        "tokenizer.num_layers=4",
        "tokenizer.decoder_num_layers=4",
        "tokenizer.num_latents=8",
        "tokenizer.num_heads=4",
        "tokenizer.num_kv_heads=2",
        "tokenizer.model_dim=64",
        "tokenizer.mlp_hidden_dim=128",
        "tokenizer.head_dim=32",
        "tokenizer.channel_dim=8",
        "tokenizer.resize_shape=[64,64]",
    ]


def launch_round(data_dir: Path, ckpt_dir: Path, total_steps: int, tag: str) -> None:
    overrides = build_overrides(data_dir, ckpt_dir, total_steps)
    procs = []
    logs = []
    for pid in range(NUM_PROCESSES):
        log = open(data_dir / f"proc{pid}_{tag}.log", "w")
        logs.append(log)
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--child",
                    "--process-id",
                    str(pid),
                    "--",
                    *overrides,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=REPO,
            )
        )
    codes = [p.wait() for p in procs]
    for log in logs:
        log.close()
    if any(codes):
        for pid, code in enumerate(codes):
            if code:
                print(f"--- proc{pid} ({tag}) exit={code}, log tail ---")
                print("\n".join((data_dir / f"proc{pid}_{tag}.log").read_text().splitlines()[-30:]))
        raise SystemExit(f"preflight {tag} FAILED: exit codes {codes}")
    print(f"preflight {tag}: all {NUM_PROCESSES} processes exited 0")


def stage_data(data_dir: Path, n_train: int, n_eval: int) -> None:
    base = "gs://visionary-uc1/so101/data/tokenizer/shards"
    for split, count in (("train", n_train), ("eval", n_eval)):
        out = data_dir / split
        out.mkdir(parents=True, exist_ok=True)
        shards = [f"{base}/{split}/shard-{i:05d}.arecord" for i in range(count)]
        subprocess.run(["gcloud", "storage", "cp", "-n", *shards, str(out)], check=True)
        for name in ("lengths.json", "fps.json"):
            full = json.loads(
                subprocess.run(
                    ["gcloud", "storage", "cat", f"{base}/{split}/{name}"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            (out / name).write_text(json.dumps(full[: count * 256]))
        print(f"staged {count} {split} shards")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument("--data-dir", default="/tmp/preflight_pod")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    if args.child:
        child(args)
        return

    data_dir = Path(args.data_dir)
    stage_data(data_dir, n_train=4, n_eval=2)
    ckpt_dir = data_dir / "ckpt"
    launch_round(data_dir, ckpt_dir, total_steps=7, tag="fresh")
    launch_round(data_dir, ckpt_dir, total_steps=12, tag="resume")
    print("preflight PASSED: fresh run + resume both clean")


if __name__ == "__main__":
    main()
