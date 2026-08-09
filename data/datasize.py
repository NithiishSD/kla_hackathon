"""
Prints the actual shape of a few real files from each folder, so there's
no ambiguity about what resolution you're actually working with.

Usage:
    python check_shapes.py /path/to/data_root
"""

import glob
import os
import sys

import numpy as np


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else ".././data"

    for label, path in [
        ("train/gt", os.path.join(data_root, "train", "gt")),
        ("train/NoisyLR", os.path.join(data_root, "train", "NoisyLR")),
        ("test", os.path.join(data_root, "test")),
    ]:
        files = sorted(glob.glob(os.path.join(path, "*.npy")))[:5]
        if not files:
            print(f"{label}: no .npy files found at {path}")
            continue
        print(f"\n{label} ({path}):")
        for f in files:
            arr = np.load(f)
            print(f"  {os.path.basename(f)}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()