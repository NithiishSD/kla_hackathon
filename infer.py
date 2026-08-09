"""
Submission inference script.

Usage (exact required contract -- no other REQUIRED arguments):
    python infer.py <test_images_dir> <output_dir>

Before submitting: set CHECKPOINT_PATH below to point at your final
chosen checkpoint file, and make sure that .pt file is included
alongside this script. That is the ONE manual edit this script needs,
done once before packaging -- nothing about a specific test run
requires editing this file.

What it does:
  - Loads the model + calibration stats (p_low/p_high) FROM THE
    CHECKPOINT ITSELF -- no train/ folder, no calib_stats.json needed
    next to the test directory.
  - Reads every .npy file in test_images_dir, runs restoration, writes
    the restored array (same filename) to output_dir as raw .npy --
    no plots, no chrome, just the deliverable image.
  - Times the FULL path (script start, model load, per-file I/O,
    inference, write) since that's how grading measures speed, not
    just the forward pass in isolation.
"""

import os
import sys
import time

import numpy as np
import torch

from model import SemiRestoreNet_V2

# ---------------------------------------------------------------------
# ONE-TIME EDIT before submission: point this at your final checkpoint.
# ---------------------------------------------------------------------
CHECKPOINT_PATH = "./checkpoints_baseline/best_model.pt"  # <-- EDIT THIS ONCE before submission


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Set CHECKPOINT_PATH "
            f"at the top of infer.py to point at your submitted model file."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "p_low" not in ckpt or "p_high" not in ckpt:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} has no baked-in p_low/p_high. "
            f"Run bake_calib_into_checkpoint.py on it first (one-time fix "
            f"for checkpoints saved before calibration was embedded), or "
            f"retrain with the current train.py/train_baseline.py, which "
            f"saves these automatically."
        )

    cfg = ckpt["config"]
    model = SemiRestoreNet_V2(
        dim=cfg["dim"], num_blocks=cfg["num_blocks"], scale_factor=cfg["scale_factor"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    p_low, p_high = ckpt["p_low"], ckpt["p_high"]
    return model, p_low, p_high, cfg


def normalize(arr, p_low, p_high):
    """Same affine map used during training: p_low -> 0, p_high -> 1,
    clipped. MUST match sem_dataset.py's normalize() exactly."""
    arr = (arr.astype(np.float32) - p_low) / (p_high - p_low)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def denormalize(arr, p_low, p_high):
    """Inverse of normalize() -- maps the model's [0,1] output back to
    the original physical intensity units the input was measured in,
    so the deliverable is comparable to the raw sensor scale, not left
    in an arbitrary normalized range."""
    return arr * (p_high - p_low) + p_low


def main():
    if len(sys.argv) != 3:
        print("Usage: python infer.py <test_images_dir> <output_dir>")
        sys.exit(1)

    test_images_dir = sys.argv[1]
    output_dir = sys.argv[2]

    t_start = time.time()

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, p_low, p_high, cfg = load_model(CHECKPOINT_PATH, device)
    model = torch.compile(model)
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    t_model_ready = time.time()

    files = sorted(f for f in os.listdir(test_images_dir) if f.endswith(".npy"))
    if not files:
        print(f"No .npy files found in {test_images_dir}")
        sys.exit(1)

    per_file_times = []
    with torch.no_grad():
        for fname in files:
            t0 = time.time()

            raw = np.load(os.path.join(test_images_dir, fname))
            x = normalize(raw, p_low, p_high)
            x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

            if use_bf16:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(x_t)
            else:
                pred = model(x_t)

            pred_np = pred.squeeze(0).squeeze(0).float().cpu().numpy()
            restored = denormalize(pred_np, p_low, p_high)

            np.save(os.path.join(output_dir, fname), restored.astype(np.float32))

            per_file_times.append(time.time() - t0)

    t_end = time.time()

    total_time = t_end - t_start
    n = len(files)
    print(f"Processed {n} files.")
    print(f"Model load time: {t_model_ready - t_start:.3f}s")
    print(f"Total end-to-end time (incl. model load, I/O, inference, write): "
          f"{total_time:.3f}s")
    print(f"Average per-file time (I/O + inference + write): "
          f"{sum(per_file_times) / n * 1000:.1f}ms")
    print(f"Average end-to-end throughput: {n / total_time:.2f} images/sec")
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()