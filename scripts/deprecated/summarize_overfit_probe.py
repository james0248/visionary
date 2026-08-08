import argparse
import json
from pathlib import Path

import numpy as np


def mean_of(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def load_chunks(root: Path, label: str, chunks: list[str]):
    out = {}
    for chunk in chunks:
        path = root / f"{label}_{chunk}" / f"{label}_{chunk}_pixel_tf.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for index, entry in payload["results"].items():
            key = f"{chunk}:{index}"
            out[key] = {
                "stream": entry["stream"],
                "recon": mean_of(entry["psnr_moving"]["recon"]),
                "tf": mean_of(entry["psnr_moving"]["tf"]),
                "roll": mean_of(entry["psnr_moving"]["roll"]),
                "copy": mean_of(entry["psnr_moving"]["copy"]),
            }
    return out


def section(title, overfit, base, keys):
    lines = [f"\n## {title}\n", "| clip | ceiling | base tf | ovf tf | tf gain | base gap | ovf gap | gap closed |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    gains, base_gaps, ovf_gaps = [], [], []
    for key in keys:
        o, b = overfit[key], base[key]
        bg, og = b["recon"] - b["tf"], o["recon"] - o["tf"]
        gains.append(o["tf"] - b["tf"])
        base_gaps.append(bg)
        ovf_gaps.append(og)
        lines.append(
            f"| {key} | {b['recon']:.2f} | {b['tf']:.2f} | {o['tf']:.2f} | "
            f"{o['tf'] - b['tf']:+.2f} | {bg:.2f} | {og:.2f} | {bg - og:+.2f} |"
        )
    mb, mo = float(np.mean(base_gaps)), float(np.mean(ovf_gaps))
    lines.append(
        f"\n**mean tf gain {float(np.mean(gains)):+.3f} dB** | "
        f"gap {mb:.3f} -> {mo:.3f} ({(mb - mo) / mb * 100:+.1f}% closed)"
    )
    return "\n".join(lines), mb, mo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    eval_chunks = ["c0", "c1", "c2", "c3"]
    train_chunks = ["t0", "t1"]
    overfit = load_chunks(root, "overfit", eval_chunks + train_chunks)
    base = load_chunks(root, "base215", eval_chunks + train_chunks)

    shared = sorted(set(overfit) & set(base))
    eval_keys = [k for k in shared if k.split(":")[0] in eval_chunks]
    train_keys = [k for k in shared if k.split(":")[0] in train_chunks]

    parts = ["# Single-repo overfit probe (Loki0929/so100_lan)",
             "",
             "base215 = 2.15M ftg5 checkpoint (147h, 733 repos)",
             "overfit = +100k steps on so100_lan only (2.0h, 1 rig)",
             "All numbers are moving-region PSNR in dB; gap = tokenizer ceiling - teacher-forced."]
    summary = {}
    if eval_keys:
        text, mb, mo = section("Held-out lan episodes (generalization)", overfit, base, eval_keys)
        parts.append(text)
        summary["eval"] = {"base_gap": mb, "overfit_gap": mo, "n": len(eval_keys)}
    if train_keys:
        text, mb, mo = section("Memorized lan episodes (train split)", overfit, base, train_keys)
        parts.append(text)
        summary["train"] = {"base_gap": mb, "overfit_gap": mo, "n": len(train_keys)}
    if "eval" in summary and "train" in summary:
        gen_gap = summary["eval"]["overfit_gap"] - summary["train"]["overfit_gap"]
        parts.append(
            f"\n## Memorization gap\n\novertfit model: train gap {summary['train']['overfit_gap']:.3f} dB, "
            f"held-out gap {summary['eval']['overfit_gap']:.3f} dB, difference **{gen_gap:+.3f} dB**"
        )

    Path(args.out).write_text("\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
