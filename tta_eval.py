"""
Test-Time Augmentation (TTA) check -- zero retraining cost. Runs each
validation LR input through all 8 dihedral-group (D4) transforms
(4 rotations x optional horizontal flip -- the same bit-exact transforms
sem_dataset.py already uses for training augmentation), inverse-transforms
each prediction back to canonical orientation, and averages them.

Rationale specific to the CD-bias diagnosis in deepseek_validation.txt:
if the model's edge-slope distortion isn't perfectly symmetric under
flips/rotations, averaging over all 8 orientations should partially
cancel out a directionally-consistent bias, the way it would for any
other orientation-dependent systematic error. Costs nothing to check
against an existing checkpoint -- no new training.

Reuses calculate_metrology_kpis/compute_psnr/compute_ssim from
evaluation.py so numbers are directly comparable to your existing
scorecards, and evaluates the SAME held-out val split (fixed seed=42).

Usage:
    python tta_eval.py /path/to/data_root /path/to/checkpoint.pt
"""
import os
import sys

import numpy as np
import torch

from sem_dataset import SEMPairDataset, load_calib_stats
from model import SemiRestoreNet_V2
from evaluation import compute_psnr, compute_ssim, calculate_metrology_kpis


def apply_transform(x, flip, k):
    if flip:
        x = torch.flip(x, dims=[-1])
    x = torch.rot90(x, k, dims=[-2, -1])
    return x


def invert_transform(x, flip, k):
    x = torch.rot90(x, (-k) % 4, dims=[-2, -1])
    if flip:
        x = torch.flip(x, dims=[-1])
    return x


D4_TRANSFORMS = [(flip, k) for flip in (False, True) for k in range(4)]


def tta_predict(model, lr_img):
    """lr_img: (1, 1, H, W). Returns the 8-view-averaged prediction."""
    preds = []
    for flip, k in D4_TRANSFORMS:
        x = apply_transform(lr_img, flip, k)
        with torch.no_grad():
            y = model(x)
        y = invert_transform(y, flip, k)
        preds.append(y)
    return torch.stack(preds, dim=0).mean(dim=0)


def main():
    if len(sys.argv) < 3:
        print("Usage: python tta_eval.py /path/to/data_root /path/to/checkpoint.pt")
        return
    data_root, ckpt_path = sys.argv[1], sys.argv[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print(f"Loaded checkpoint: epoch {ckpt['epoch']+1}, val_psnr={ckpt['val_psnr']:.2f}dB")

    model = SemiRestoreNet_V2(dim=cfg["dim"], num_blocks=cfg["num_blocks"],
                               scale_factor=cfg["scale_factor"]).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    p_low, p_high = load_calib_stats(data_root)
    full_train = SEMPairDataset(os.path.join(data_root, "train", "gt"),
                                 os.path.join(data_root, "train", "NoisyLR"),
                                 p_low, p_high, scale_factor=cfg["scale_factor"], augment=False)
    n_val = int(len(full_train) * 0.1)
    _, val_subset = torch.utils.data.random_split(
        full_train, [len(full_train) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))

    print(f"\nEvaluating on {len(val_subset)} held-out validation samples "
          f"(plain single-pass vs 8-view D4 TTA)...")

    plain_results, tta_results = [], []
    for idx in val_subset.indices:
        lr_img, gt_img = full_train[idx]
        lr_t = lr_img.unsqueeze(0).to(device)
        gt_t = gt_img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_plain = model(lr_t)
        pred_tta = tta_predict(model, lr_t)

        for results, pred in ((plain_results, pred_plain), (tta_results, pred_tta)):
            psnr = compute_psnr(pred, gt_t)
            ssim = compute_ssim(pred, gt_t)
            kpi = calculate_metrology_kpis(pred, gt_t)
            results.append((psnr, ssim, kpi))

    def summarize(results, label):
        psnrs = np.array([r[0] for r in results])
        ssims = np.array([r[1] for r in results])
        kpis = [r[2] for r in results if r[2] is not None]
        print(f"\n=== {label} (n={len(results)}) ===")
        print(f"PSNR: mean={psnrs.mean():.2f}dB  std={psnrs.std():.2f}  "
              f"min={psnrs.min():.2f}  max={psnrs.max():.2f}")
        print(f"SSIM: mean={ssims.mean():.4f}  std={ssims.std():.4f}")
        if kpis:
            avg_bias = np.mean([k['cd_bias_px'] for k in kpis])
            avg_ler = np.mean([k['ler_error_px'] for k in kpis])
            avg_lwr = np.mean([k['lwr_error_px'] for k in kpis])
            avg_slope = np.mean([k['slope_fidelity'] for k in kpis])
            print(f"Mean CD Bias:   {avg_bias:.3f} px")
            print(f"Mean LER Error: {avg_ler:.3f} px")
            print(f"Mean LWR Error: {avg_lwr:.3f} px")
            print(f"Slope Fidelity: {avg_slope*100:.1f} %")
        return psnrs.mean(), avg_bias if kpis else None

    plain_psnr, plain_bias = summarize(plain_results, "PLAIN (single-pass)")
    tta_psnr, tta_bias = summarize(tta_results, "TTA (8-view D4 average)")

    print(f"\n=== Delta (TTA - plain) ===")
    print(f"PSNR:    {tta_psnr - plain_psnr:+.3f} dB")
    if plain_bias is not None:
        print(f"CD Bias: {tta_bias - plain_bias:+.3f} px  "
              f"(closer to 0 is better; plain={plain_bias:.3f}, tta={tta_bias:.3f})")


if __name__ == "__main__":
    main()
