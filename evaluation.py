import json
import os
import sys
import time

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
    print("WARNING: 'lpips' package not installed. LPIPS will be skipped.")

# ---------------------------------------------------------------------------
# 1. NEW: Metrology Support (JSON Encoder + KPI Logic)
# ---------------------------------------------------------------------------
class MetrologyJSONEncoder(json.JSONEncoder):
    """Handles float32 and numpy types for the summary.json"""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.float16)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        return super().default(obj)

def calculate_metrology_kpis(pred, gt, threshold=0.5):
    """Computes CD Bias, LER, LWR, and Slope Fidelity for Metrology Scorecard."""
    p = pred.detach().squeeze().cpu().numpy()
    g = gt.detach().squeeze().cpu().numpy()
    
    def get_line_stats(img):
        cd_per_row, left_edges, slopes = [], [], []
        for row in img:
            edges = np.where(np.diff((row > threshold).astype(int)) != 0)[0]
            if len(edges) >= 2:
                cd_per_row.append(edges[-1] - edges[0])
                left_edges.append(edges[0])
                slopes.append(np.max(np.abs(np.gradient(row))))
        if not cd_per_row: return None
        return np.mean(cd_per_row), 3 * np.std(left_edges), 3 * np.std(cd_per_row), np.mean(slopes)

    p_s = get_line_stats(p)
    g_s = get_line_stats(g)
    if p_s is None or g_s is None: return None
    
    return {
        "cd_bias_px": p_s[0] - g_s[0],
        "ler_error_px": abs(p_s[1] - g_s[1]),
        "lwr_error_px": abs(p_s[2] - g_s[2]),
        "slope_fidelity": p_s[3] / (g_s[3] + 1e-6)
    }

# ---------------------------------------------------------------------------
# 2. Standard Metrics (Unchanged)
# ---------------------------------------------------------------------------
def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0: return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)

def compute_lpips(lpips_model, pred, target):
    if lpips_model is None: return 0.0
    def prep(x): return x.repeat(1, 3, 1, 1) * 2.0 - 1.0
    with torch.no_grad():
        d = lpips_model(prep(pred), prep(target))
    return d.item()

def _gaussian_window(window_size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0)

def compute_ssim(pred, target, window_size=11, data_range=1.0):
    window = _gaussian_window(window_size, device=pred.device)
    pad = window_size // 2
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu1, mu2 = F.conv2d(pred, window, padding=pad), F.conv2d(target, window, padding=pad)
    sigma1_sq = F.conv2d(pred * pred, window, padding=pad) - mu1**2
    sigma2_sq = F.conv2d(target * target, window, padding=pad) - mu2**2
    sigma12 = F.conv2d(pred * target, window, padding=pad) - mu1*mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

# ---------------------------------------------------------------------------
# 3. Enhanced Visualization
# ---------------------------------------------------------------------------
def save_comparison_plot(lr_np, pred_np, gt_np, title, out_path):
    n_panels = 3 if gt_np is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(lr_np, cmap='gray', vmin=0, vmax=1); axes[0].set_title("Input (Noisy LR)"); axes[0].axis('off')
    axes[1].imshow(pred_np, cmap='gray', vmin=0, vmax=1); axes[1].set_title("Model Prediction"); axes[1].axis('off')
    if gt_np is not None:
        axes[2].imshow(gt_np, cmap='gray', vmin=0, vmax=1); axes[2].set_title("Ground Truth"); axes[2].axis('off')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py /path/to/data_root /path/to/checkpoint.pt")
        return

    data_root, ckpt_path = sys.argv[1], sys.argv[2]
    out_dir = "./eval_outputs_baseline_scale_v1"
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    print(f"Loaded checkpoint: epoch {ckpt['epoch']+1}, val_psnr={ckpt['val_psnr']:.2f}dB")

    model = SemiRestoreNet_V2(dim=cfg["dim"], num_blocks=cfg["num_blocks"], scale_factor=cfg["scale_factor"]).to(device)
    # Using strict=False to handle potential metrology_gain parameter
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    lpips_model = lpips.LPIPS(net='alex').to(device) if _LPIPS_AVAILABLE else None
    p_low, p_high = load_calib_stats(data_root)

    full_train = SEMPairDataset(os.path.join(data_root, "train", "gt"), os.path.join(data_root, "train", "NoisyLR"),
                                 p_low, p_high, scale_factor=cfg["scale_factor"], augment=False)
    n_val = int(len(full_train) * 0.1)
    _, val_subset = torch.utils.data.random_split(full_train, [len(full_train)-n_val, n_val], 
                                                 generator=torch.Generator().manual_seed(42))

    print(f"\nEvaluating on {len(val_subset)} held-out validation samples...")
    results = [] 
    with torch.no_grad():
        for idx in val_subset.indices:
            lr_img, gt_img = full_train[idx]
            lr_t, gt_t = lr_img.unsqueeze(0).to(device), gt_img.unsqueeze(0).to(device)
            pred_t = model(lr_t)
            
            psnr = compute_psnr(pred_t, gt_t)
            ssim = compute_ssim(pred_t, gt_t)
            lp = compute_lpips(lpips_model, pred_t, gt_t)
            kpis = calculate_metrology_kpis(pred_t, gt_t)
            
            fname = os.path.basename(full_train.pairs[idx][0])
            results.append({"psnr": psnr, "ssim": ssim, "lpips": lp, "name": fname, "kpis": kpis, "idx": idx})

    # Summary Statistics
    psnrs = np.array([r["psnr"] for r in results])
    ssims = np.array([r["ssim"] for r in results])
    print(f"\n=== Validation set summary (n={len(results)}) ===")
    print(f"PSNR: mean={psnrs.mean():.2f}dB  std={psnrs.std():.2f}  min={psnrs.min():.2f}  max={psnrs.max():.2f}")
    print(f"SSIM: mean={ssims.mean():.4f}  std={ssims.std():.4f}  min={ssims.min():.4f}  max={ssims.max():.4f}")
    
    # Worst/Best 5
    results_sorted = sorted(results, key=lambda r: r["psnr"])
    print(f"\nWorst 5 samples by PSNR:")
    for r in results_sorted[:5]:
        print(f"  {r['name']}: PSNR={r['psnr']:.2f}dB, SSIM={r['ssim']:.4f}")
    print(f"\nBest 5 samples by PSNR:")
    for r in results_sorted[-5:]:
        print(f"  {r['name']}: PSNR={r['psnr']:.2f}dB, SSIM={r['ssim']:.4f}")

    # Visualization of Worst 3 with Metrology info in Title
    for r in results_sorted[:3]:
        lr_img, gt_img = full_train[r['idx']]
        with torch.no_grad():
            pred = model(lr_img.unsqueeze(0).to(device)).detach().cpu().squeeze(0).numpy()
        lr_up = F.interpolate(lr_img.unsqueeze(0), scale_factor=cfg["scale_factor"], mode='nearest').squeeze(0).numpy()
        bias_str = f"Bias: {r['kpis']['cd_bias_px']:.2f}px" if r['kpis'] else "Bias: N/A"
        save_comparison_plot(lr_up[0], pred[0], gt_img.numpy()[0], 
                             f"{r['name']} | PSNR: {r['psnr']:.2f} | SSIM: {r['ssim']:.4f} | {bias_str}", 
                             os.path.join(out_dir, f"worst_{r['name']}.png"))

    # Named Samples Check
    named_samples = ["000218.npy", "000352.npy", "000425.npy"]
    all_names = {os.path.basename(p[0]): i for i, p in enumerate(full_train.pairs)}
    print(f"\n=== Named Difficulty Samples ===")
    for n in named_samples:
        if n in all_names:
            idx = all_names[n]
            lr_img, gt_img = full_train[idx]
            with torch.no_grad(): pred_t = model(lr_img.unsqueeze(0).to(device))
            psnr, ssim = compute_psnr(pred_t, gt_img.unsqueeze(0).to(device)), compute_ssim(pred_t, gt_img.unsqueeze(0).to(device))
            kpi = calculate_metrology_kpis(pred_t, gt_img.unsqueeze(0).to(device))
            print(f"  {n}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")
            bias_str = f"Bias: {kpi['cd_bias_px']:.2f}px" if kpi else "Bias: N/A"
            save_comparison_plot(F.interpolate(lr_img.unsqueeze(0), scale_factor=cfg["scale_factor"], mode='nearest').squeeze().numpy(), 
                                 pred_t.detach().cpu().squeeze().numpy(), gt_img.squeeze().numpy(), 
                                 f"Named: {n} | PSNR: {psnr:.2f} | {bias_str}", os.path.join(out_dir, f"named_{n}.png"))

    # METROLOGY SCORECARD (Additional Metrics)
    valid_kpis = [r["kpis"] for r in results if r["kpis"] is not None]
    if valid_kpis:
        avg_bias = np.mean([k['cd_bias_px'] for k in valid_kpis])
        avg_ler = np.mean([k['ler_error_px'] for k in valid_kpis])
        avg_lwr = np.mean([k['lwr_error_px'] for k in valid_kpis])
        avg_slope = np.mean([k['slope_fidelity'] for k in valid_kpis])
        print(f"\n{'='*45}\n        FINAL METROLOGY SCORECARD\n{'='*45}")
        print(f"Mean CD Bias:   {avg_bias:.3f} px")
        print(f"Mean LER Error: {avg_ler:.3f} px")
        print(f"Mean LWR Error: {avg_lwr:.3f} px")
        print(f"Slope Fidelity: {avg_slope*100:.1f} %")
        print(f"{'='*45}")

    # Test Set Visuals
    print(f"\n=== Processing Test Set (no GT) ===")
    test_set = SEMTestDataset(os.path.join(data_root, "test"), p_low, p_high)
    for i in range(min(5, len(test_set))):
        lr_img, fname = test_set[i]
        with torch.no_grad(): pred = model(lr_img.unsqueeze(0).to(device)).detach().cpu().squeeze().numpy()
        lr_up = F.interpolate(lr_img.unsqueeze(0), scale_factor=cfg["scale_factor"], mode='nearest').squeeze().numpy()
        save_comparison_plot(lr_up, pred, None, f"TEST: {fname}", os.path.join(out_dir, f"test_{fname}.png"))

    # Final Summary Save
    summary = {
        "overall": {"psnr_avg": float(psnrs.mean()), "ssim_avg": float(ssims.mean())},
        "metrology": {"avg_cd_bias": avg_bias, "avg_ler_err": avg_ler, "avg_lwr_err": avg_lwr, "avg_slope_fid": avg_slope} if valid_kpis else {},
        "per_sample": results
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=MetrologyJSONEncoder)
    print(f"\nDone. Detailed visuals and summary in {out_dir}/")

if __name__ == "__main__":
    main()