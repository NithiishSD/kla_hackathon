"""
Evaluation script -- run after training finishes.

What it does:
  1. Loads the best checkpoint.
  2. Computes PSNR + SSIM over the held-out VALIDATION set (the only split
     with ground truth besides train itself) -- reports overall mean/std,
     plus the worst-N and best-N samples by PSNR, so you see the spread,
     not just one averaged number.
  3. If any of your named difficulty samples (e.g. 000218, 000352, 000425)
     are findable in train or val, evaluates them individually and saves
     a side-by-side (input / prediction / ground truth) plot for each.
  4. Runs the TEST set through the model and saves prediction images for
     visual inspection. NOTE: the test set has no ground truth, so no
     PSNR/SSIM is possible there -- including your OOD "Woven Grid"
     sample (000014) -- this script gives you visuals only for that split,
     not a number. Don't report a test-set PSNR you don't actually have.

Usage:
    python evaluate.py /path/to/data_root ./checkpoints/best_model.pt
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sem_dataset import SEMPairDataset, SEMTestDataset, load_calib_stats
from model import SemiRestoreNet_V2

try:
    import lpips
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    print("WARNING: 'lpips' package not installed (pip install lpips) -- "
          "LPIPS will be skipped. PSNR/SSIM still reported.")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def compute_lpips(lpips_model, pred, target):
    """LPIPS expects 3-channel images in roughly [-1, 1]. Our data is
    1-channel [0, 1] -- replicate the channel and rescale. This is a
    reasonable adaptation, not a perfect one: LPIPS's AlexNet backbone was
    trained on natural RGB photos, not SEM imagery, so treat the absolute
    number as directionally useful (lower = more perceptually similar),
    not as validated ground truth for this domain."""
    def prep(x):
        x3 = x.repeat(1, 3, 1, 1)
        return x3 * 2.0 - 1.0
    with torch.no_grad():
        d = lpips_model(prep(pred), prep(target))
    return d.item()


def _gaussian_window(window_size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)


def compute_ssim(pred, target, window_size=11, data_range=1.0):
    """Standard single-channel SSIM. pred/target: (1, 1, H, W), same device."""
    window = _gaussian_window(window_size, device=pred.device)
    pad = window_size // 2
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(pred, window, padding=pad)
    mu2 = F.conv2d(target, window, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def save_comparison_plot(lr_np, pred_np, gt_np, title, out_path):
    n_panels = 3 if gt_np is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(lr_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Input (noisy LR)")
    axes[0].axis('off')

    axes[1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Model prediction")
    axes[1].axis('off')

    if gt_np is not None:
        axes[2].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title("Ground truth")
        axes[2].axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py /path/to/data_root /path/to/checkpoint.pt")
        return

    data_root = sys.argv[1]
    ckpt_path = sys.argv[2]
    out_dir = "./eval_outputs_baseline_hardtrained"
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- Load checkpoint + rebuild model with the SAME config it trained with ----
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']+1}, "
          f"val_loss={ckpt['val_loss']:.4f}, val_psnr={ckpt['val_psnr']:.2f}dB "
          f"(as recorded during training)")
    print(f"Checkpoint config: {cfg}")

    model = SemiRestoreNet_V2(
        dim=cfg["dim"], num_blocks=cfg["num_blocks"], scale_factor=cfg["scale_factor"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    lpips_model = None
    if _LPIPS_AVAILABLE:
        print("Loading LPIPS (AlexNet backbone)...")
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()

    p_low, p_high = load_calib_stats(data_root)

    # ---- Held-out validation set: THE source of real PSNR/SSIM numbers ----
    # Rebuild with the exact same split as training (same seed, same
    # val_fraction) so this is genuinely the held-out set, not train data.
    full_train = SEMPairDataset(
        gt_dir=os.path.join(data_root, "train", "gt"),
        lr_dir=os.path.join(data_root, "train", "NoisyLR"),
        p_low=p_low, p_high=p_high,
        scale_factor=cfg["scale_factor"], augment=False,
    )
    val_fraction = 0.1
    seed = 42
    n_val = max(1, int(len(full_train) * val_fraction))
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [n_train, n_val], generator=generator
    )

    print(f"\nEvaluating on {len(val_subset)} held-out validation samples...")
    results = []  # (psnr, ssim, lpips, filename)
    with torch.no_grad():
        for idx in val_subset.indices:
            lr_img, gt_img = full_train[idx]
            lr_img = lr_img.unsqueeze(0).to(device)
            gt_img = gt_img.unsqueeze(0).to(device)
            pred = model(lr_img)
            psnr = compute_psnr(pred, gt_img)
            ssim = compute_ssim(pred, gt_img)
            lp = compute_lpips(lpips_model, pred, gt_img) if lpips_model is not None else None
            fname = os.path.basename(full_train.pairs[idx][0])
            results.append((psnr, ssim, lp, fname))

    psnrs = np.array([r[0] for r in results])
    ssims = np.array([r[1] for r in results])
    print(f"\n=== Validation set summary (n={len(results)}) ===")
    print(f"PSNR: mean={psnrs.mean():.2f}dB  std={psnrs.std():.2f}  "
          f"min={psnrs.min():.2f}  max={psnrs.max():.2f}")
    print(f"SSIM: mean={ssims.mean():.4f}  std={ssims.std():.4f}  "
          f"min={ssims.min():.4f}  max={ssims.max():.4f}")
    if lpips_model is not None:
        lpipss = np.array([r[2] for r in results])
        print(f"LPIPS: mean={lpipss.mean():.4f}  std={lpipss.std():.4f}  "
              f"min={lpipss.min():.4f}  max={lpipss.max():.4f}  "
              f"(lower = more perceptually similar; AlexNet backbone trained "
              f"on natural photos, treat as directional for SEM data, not "
              f"an absolute validated benchmark)")

    results_sorted = sorted(results, key=lambda r: r[0])
    print(f"\nWorst 5 samples by PSNR (your weakest cases -- inspect these):")
    for psnr, ssim, lp, fname in results_sorted[:5]:
        lp_str = f", LPIPS={lp:.4f}" if lp is not None else ""
        print(f"  {fname}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}{lp_str}")
    print(f"\nBest 5 samples by PSNR:")
    for psnr, ssim, lp, fname in results_sorted[-5:]:
        lp_str = f", LPIPS={lp:.4f}" if lp is not None else ""
        print(f"  {fname}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}{lp_str}")

    # Save plots for the worst 3 -- these are your real weak spots, worth
    # having visuals ready for the presentation.
    name_to_idx = {os.path.basename(full_train.pairs[i][0]): i for i in val_subset.indices}
    for psnr, ssim, lp, fname in results_sorted[:3]:
        idx = name_to_idx[fname]
        lr_img, gt_img = full_train[idx]
        with torch.no_grad():
            pred = model(lr_img.unsqueeze(0).to(device)).squeeze(0).cpu()
        lr_up = F.interpolate(lr_img.unsqueeze(0), scale_factor=cfg["scale_factor"],
                               mode='nearest').squeeze(0)
        save_comparison_plot(
            lr_up[0].numpy(), pred[0].numpy(), gt_img[0].numpy(),
            title=f"{fname} (worst case: PSNR={psnr:.2f}dB, SSIM={ssim:.4f})",
            out_path=os.path.join(out_dir, f"worst_{fname.replace('.npy','')}.png"),
        )
    print(f"\nSaved worst-3 comparison plots to {out_dir}/")

    # ---- Your specifically-named difficulty samples, if findable ----
    named_samples = ["000218.npy", "000352.npy", "000425.npy"]
    all_train_names = {os.path.basename(p[0]): i for i, p in enumerate(full_train.pairs)}
    print(f"\n=== Named difficulty samples (from your diagnostic plots) ===")
    for name in named_samples:
        if name not in all_train_names:
            print(f"  {name}: not found in train/gt -- check exact filename.")
            continue
        idx = all_train_names[name]
        split = "VALIDATION (held-out)" if idx in val_subset.indices else "TRAIN (model saw this during training)"
        lr_img, gt_img = full_train[idx]
        with torch.no_grad():
            pred = model(lr_img.unsqueeze(0).to(device))
        psnr = compute_psnr(pred, gt_img.unsqueeze(0).to(device))
        ssim = compute_ssim(pred, gt_img.unsqueeze(0).to(device))
        print(f"  {name} [{split}]: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")
        # CHANGE: plot named samples regardless of split -- an anomalously
        # bad score on a TRAIN sample (model has seen it dozens of times)
        # is worth inspecting visually just as much as a bad val sample.
        lr_up = F.interpolate(lr_img.unsqueeze(0), scale_factor=cfg["scale_factor"],
                               mode='nearest').squeeze(0)
        save_comparison_plot(
            lr_up[0].numpy(), pred.squeeze(0)[0].cpu().numpy(), gt_img[0].numpy(),
            title=f"{name} [{split}]: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}",
            out_path=os.path.join(out_dir, f"named_{name.replace('.npy','')}.png"),
        )
        if idx not in val_subset.indices:
            print(f"    (in training set -- this number reflects fit, not "
                  f"generalization; don't present it as a held-out result -- "
                  f"but a LOW score here despite training on it is still "
                  f"worth investigating, it suggests the sample itself is "
                  f"unusually hard or possibly anomalous data)")

    # ---- Test set: qualitative only, NO ground truth available ----
    print(f"\n=== Test set (no ground truth -- visuals only, no PSNR/SSIM) ===")
    test_set = SEMTestDataset(os.path.join(data_root, "test"), p_low, p_high)
    n_show = min(5, len(test_set))
    for i in range(n_show):
        lr_img, fname = test_set[i]
        with torch.no_grad():
            pred = model(lr_img.unsqueeze(0).to(device)).squeeze(0).cpu()
        lr_up = F.interpolate(lr_img.unsqueeze(0), scale_factor=cfg["scale_factor"],
                               mode='nearest').squeeze(0)
        save_comparison_plot(
            lr_up[0].numpy(), pred[0].numpy(), None,
            title=f"TEST (no GT): {fname}",
            out_path=os.path.join(out_dir, f"test_{fname.replace('.npy','')}.png"),
        )
    print(f"Saved {n_show} test-set prediction visuals to {out_dir}/ "
          f"(no numeric score possible -- eyeball these for artifacts, "
          f"especially any known OOD / 'Woven Grid' samples)")

    # ---- Save numeric summary to disk -- stdout doesn't survive after
    # the terminal closes, and you'll want to compare runs later. ----
    summary = {
        "checkpoint": ckpt_path,
        "checkpoint_epoch": ckpt["epoch"] + 1,
        "config": cfg,
        "n_val_samples": len(results),
        "psnr": {"mean": float(psnrs.mean()), "std": float(psnrs.std()),
                 "min": float(psnrs.min()), "max": float(psnrs.max())},
        "ssim": {"mean": float(ssims.mean()), "std": float(ssims.std()),
                 "min": float(ssims.min()), "max": float(ssims.max())},
        "per_sample": [
            {"filename": r[3], "psnr": r[0], "ssim": r[1], "lpips": r[2]}
            for r in results
        ],
    }
    if lpips_model is not None:
        summary["lpips"] = {"mean": float(lpipss.mean()), "std": float(lpipss.std()),
                             "min": float(lpipss.min()), "max": float(lpipss.max())}

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved numeric summary to {summary_path}")

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()