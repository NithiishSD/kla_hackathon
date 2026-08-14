"""
KLA PS01 Submission -- Evaluation Script (Component 2, mandatory)
====================================================================
This IS the benchmarking entry point KLA's team will run AS-IS.

Usage:
    python evaluate.py <path_to_test_images_dir> <path_to_output_dir>

Loads the trained SemiRestoreNet_V2 model (weights bundled in this repo
under ./weights/final_model.pt) and runs inference on every .npy image
found in <path_to_test_images_dir>, writing restored .npy outputs (same
filenames) to <path_to_output_dir>.

No ground truth, no calib_stats.json, and no manual edits are required --
normalization (1st/99th percentile robust scaling) is computed fresh,
per-image, directly from each test image. This makes the script fully
self-contained: it has no dependency on training-set statistics and is
robust to any distribution shift between the training data and the
benchmarking team's actual test set.

Requires model.py to be present alongside this script (same repo).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# model.py must live in the same directory / be importable from the repo root
from model import SemiRestoreNet_V2

# ---------------------------------------------------------------------------
# Config -- final submitted checkpoint
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "weights", "final_model.pt")

# Percentiles used for per-image robust scaling. Must match the convention
# used during training's calib_stats.json generation (p_low_pct=1.0,
# p_high_pct=99.0) -- only the SOURCE of p_low/p_high differs (per-image
# here vs. a fixed global JSON at train time), not the percentile choice.
P_LOW_PCT = 1.0
P_HIGH_PCT = 99.0


def normalize_per_image(arr):
    """Affine map: p_low -> 0, p_high -> 1, values outside clamped.
    Returns (normalized_array, p_low, p_high) -- p_low/p_high are kept so
    the model's output can be mapped back to the original intensity scale."""
    p_low = np.percentile(arr, P_LOW_PCT)
    p_high = np.percentile(arr, P_HIGH_PCT)
    if p_high <= p_low:
        # Degenerate flat image -- avoid divide-by-zero, just return zeros
        return np.zeros_like(arr, dtype=np.float32), float(p_low), float(p_low) + 1.0
    normed = (arr - p_low) / (p_high - p_low)
    normed = np.clip(normed, 0.0, 1.0).astype(np.float32)
    return normed, float(p_low), float(p_high)


def denormalize(arr, p_low, p_high):
    """Inverse of normalize_per_image -- maps model output back to the
    original physical intensity scale of the input image."""
    return arr * (p_high - p_low) + p_low


def load_model(weights_path, device):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {weights_path}. Make sure weights/final_model.pt "
            f"is present in the repo (see README for download instructions if using Git LFS "
            f"or an external link)."
        )
    ckpt = torch.load(weights_path, map_location=device)
    cfg = ckpt["config"]
    model = SemiRestoreNet_V2(
        dim=cfg["dim"],
        num_blocks=cfg["num_blocks"],
        scale_factor=cfg["scale_factor"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print(f"Loaded model from {weights_path} "
          f"(epoch {ckpt.get('epoch', -1) + 1}, val_psnr={ckpt.get('val_psnr', float('nan')):.2f}dB, "
          f"dim={cfg['dim']}, num_blocks={cfg['num_blocks']}, scale_factor={cfg['scale_factor']})")
    return model


def run_inference(input_dir, output_dir, weights_path=WEIGHTS_PATH):
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(weights_path, device)

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".npy")
    )
    if not files:
        raise ValueError(f"No .npy files found in {input_dir}")
    print(f"Found {len(files)} test images in {input_dir}")

    total_start = time.time()
    with torch.no_grad():
        for i, fname in enumerate(files):
            in_path = os.path.join(input_dir, fname)
            raw = np.load(in_path).astype(np.float32)

            normed, p_low, p_high = normalize_per_image(raw)
            lr_t = torch.from_numpy(normed).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

            pred_t = model(lr_t)
            pred = pred_t.squeeze(0).squeeze(0).detach().cpu().numpy()

            restored = denormalize(pred, p_low, p_high)

            out_path = os.path.join(output_dir, fname)
            np.save(out_path, restored.astype(np.float32))

            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"  [{i + 1}/{len(files)}] processed")

    elapsed = time.time() - total_start
    print(f"\nDone. {len(files)} images restored in {elapsed:.2f}s "
          f"({elapsed / len(files) * 1000:.1f} ms/image). Outputs saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="KLA PS01 -- standalone inference script")
    parser.add_argument("input_dir", type=str, help="Path to test images directory (.npy files)")
    parser.add_argument("output_dir", type=str, help="Path to write restored .npy outputs")
    args = parser.parse_args()

    run_inference(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()