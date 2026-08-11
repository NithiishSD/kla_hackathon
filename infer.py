"""
Submission inference script -- the actual deliverable per the KLA spec
(page 17): standalone, non-notebook, accepts (test_images_dir, output_dir),
loads the trained model, runs inference on every input image, writes
restored outputs back to disk. Runs with zero manual edits.

Deliberately does NOT import evaluation.py (pulls in matplotlib) or
anything else not needed for inference -- the benchmark times script
startup + model init + disk I/O + inference as one number, so extra
import weight here is a direct latency cost.

Calibration stats are read from the checkpoint itself (calib_p_low /
calib_p_high, baked in at save time) rather than a colocated
calib_stats.json -- this script has to work against a bare test-images
directory with no train/ folder next to it.

Usage:
    python infer.py /path/to/test_images_dir /path/to/output_dir
    python infer.py /path/to/test_images_dir /path/to/output_dir --checkpoint checkpoints_swinir/best_model.pt
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from sem_dataset import SEMTestDataset
from model import SemiRestoreNet_V2
from swinir_model import SwinIR

DEFAULT_CHECKPOINT = "checkpoints/best_model2.pt"


def build_model(cfg, device):
    model_type = cfg.get("model_type", "restormer")
    if model_type == "swinir":
        model = SwinIR(
            embed_dim=cfg["embed_dim"], depths=cfg["depths"], num_heads=cfg["num_heads"],
            window_size=cfg["window_size"], mlp_ratio=cfg["mlp_ratio"],
            scale_factor=cfg["scale_factor"],
        )
    else:
        model = SemiRestoreNet_V2(dim=cfg["dim"], num_blocks=cfg["num_blocks"],
                                   scale_factor=cfg["scale_factor"])
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_dir", help="Directory of degraded .npy input images")
    parser.add_argument("output_dir", help="Directory to write restored .npy outputs")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Model checkpoint path")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    p_low, p_high = ckpt.get("calib_p_low"), ckpt.get("calib_p_high")
    if p_low is None or p_high is None:
        raise RuntimeError(
            f"{args.checkpoint} has no baked-in calibration stats "
            f"(calib_p_low/calib_p_high) -- this checkpoint predates that fix, "
            f"re-save it with calibration stats attached before using it here."
        )

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = SEMTestDataset(args.test_dir, p_low, p_high)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=min(8, os.cpu_count() or 1),
                         pin_memory=(device == "cuda"))

    use_bf16 = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    precision = torch.bfloat16 if use_bf16 else torch.float16

    n_written = 0
    with torch.no_grad():
        for lr_batch, names in loader:
            lr_batch = lr_batch.to(device, non_blocking=True)
            if device == "cuda":
                with torch.amp.autocast('cuda', dtype=precision):
                    pred = model(lr_batch)
            else:
                pred = model(lr_batch)

            # De-normalize back to the original raw intensity scale -- the
            # model operates in the shared [0,1] calibration space, but
            # whatever ground truth this gets scored against is in the
            # original scale, so the output has to match it.
            pred_raw = pred.float().cpu().numpy() * (p_high - p_low) + p_low
            for i, name in enumerate(names):
                np.save(os.path.join(args.output_dir, name), pred_raw[i, 0])
                n_written += 1

    print(f"Wrote {n_written} restored images to {args.output_dir}")


if __name__ == "__main__":
    main()
