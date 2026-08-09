"""
One-time patch: adds p_low/p_high into an EXISTING checkpoint that was
saved before train.py/train_baseline.py/finetune_ssim.py started baking
them in automatically. Run this once on your current best checkpoint
(e.g. ./checkpoints_baseline/best_model.pt) so infer.py can use it
without needing calib_stats.json or a train/ folder present.

Usage:
    python bake_calib_into_checkpoint.py /path/to/data_root /path/to/checkpoint.pt
"""

import sys

import torch

from sem_dataset import load_calib_stats


def main():
    if len(sys.argv) < 3:
        print("Usage: python bake_calib_into_checkpoint.py /path/to/data_root /path/to/checkpoint.pt")
        return
    data_root = sys.argv[1]
    ckpt_path = sys.argv[2]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "p_low" in ckpt and "p_high" in ckpt:
        print(f"{ckpt_path} already has p_low={ckpt['p_low']}, p_high={ckpt['p_high']} -- nothing to do.")
        return

    p_low, p_high = load_calib_stats(data_root)
    ckpt["p_low"] = p_low
    ckpt["p_high"] = p_high
    torch.save(ckpt, ckpt_path)
    print(f"Baked p_low={p_low:.6f}, p_high={p_high:.6f} into {ckpt_path}")


if __name__ == "__main__":
    main()