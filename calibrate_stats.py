"""
Step 1: Global Stats Calibration.

Computes p01/p99 ONCE over the training set's noisy-LR images, and saves
them to a small JSON file. The SAME two numbers are then reused to
normalize LR, GT, and test data everywhere else in the pipeline --
recomputing per-file or per-batch would make every image's "1.0" mean a
different physical intensity, silently corrupting training/eval.

Run this once before training:
    python calibrate_stats.py /path/to/data_root

It writes calib_stats.json into data_root.
"""

import glob
import json
import os
import sys

import numpy as np


def compute_global_percentiles(lr_dir, p_low=1.0, p_high=99.0, sample_cap=None):
    """
    Computes p01/p99 over all (or a capped random sample of) LR training
    files. We calibrate off the LR/noisy images specifically, since that's
    where the out-of-range speckle spikes live -- the clean GT presumably
    doesn't need this correction, but we apply the same transform to both
    so 0/1 mean the same physical intensity in LR and GT alike.
    """
    files = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))
    if not files:
        raise ValueError(f"No .npy files found in {lr_dir}")

    if sample_cap is not None and len(files) > sample_cap:
        rng = np.random.default_rng(42)
        files = list(rng.choice(files, size=sample_cap, replace=False))

    print(f"Computing global percentiles over {len(files)} LR files...")

    all_values = []
    for f in files:
        arr = np.load(f).astype(np.float32)
        all_values.append(arr.ravel())

    all_values = np.concatenate(all_values)
    p_lo = float(np.percentile(all_values, p_low))
    p_hi = float(np.percentile(all_values, p_high))

    if p_hi <= p_lo:
        raise ValueError(
            f"p{p_high}={p_hi} <= p{p_low}={p_lo} -- data looks degenerate "
            f"(near-constant?). Check the files in {lr_dir}."
        )

    print(f"p{p_low} = {p_lo:.6f}, p{p_high} = {p_hi:.6f}")
    print(f"raw min = {all_values.min():.6f}, raw max = {all_values.max():.6f}")
    frac_clipped_low = (all_values < p_lo).mean()
    frac_clipped_high = (all_values > p_hi).mean()
    print(f"fraction below p{p_low}: {frac_clipped_low:.4%}, "
          f"fraction above p{p_high}: {frac_clipped_high:.4%}")

    return p_lo, p_hi


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data"
    lr_dir = os.path.join(data_root, "train", "lossylr")

    p_lo, p_hi = compute_global_percentiles(lr_dir, p_low=1.0, p_high=99.5)

    stats = {"p_low": p_lo, "p_high": p_hi, "p_low_pct": 1.0, "p_high_pct": 99.0}
    out_path = os.path.join(data_root, "calib_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved calibration stats to {out_path}")
    print("Reuse this exact file for train / val / test normalization -- "
          "do not recompute per-split.")


if __name__ == "__main__":
    main()
