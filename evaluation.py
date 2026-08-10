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
    print("WARNING: 'lpips' package not installed. LPIPS will show as N/A.")

# ---------------------------------------------------------------------------
# 1. JSON Serialization Fix
# ---------------------------------------------------------------------------
class MetrologyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

# ---------------------------------------------------------------------------
# 2. Metrology KPI Logic (CD, LER, LWR, Slope)
# ---------------------------------------------------------------------------
def calculate_metrology_kpis(pred, gt, threshold=0.5):
    p = pred.squeeze().cpu().numpy()
    g = gt.squeeze().cpu().numpy()
    
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
# 3. Standard Metrics
# ---------------------------------------------------------------------------
def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    return 10 * np.log10(1.0 / mse) if mse > 0 else 100.0

def compute_ssim(pred, target):
    # Simplified Gaussian Window SSIM
    window_size = 11
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    window = (g / g.sum()).unsqueeze(1) @ (g / g.sum()).unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    mu1 = F.conv2d(pred, window, padding=5)
    mu2 = F.conv2d(target, window, padding=5)
    sigma1_sq = F.conv2d(pred*pred, window, padding=5) - mu1**2
    sigma2_sq = F.conv2d(target*target, window, padding=5) - mu2**2
    sigma12 = F.conv2d(pred*target, window, padding=5) - mu1*mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def compute_lpips(lpips_model, pred, target):
    if lpips_model is None: return 0.0
    def prep(x): return x.repeat(1, 3, 1, 1) * 2.0 - 1.0
    with torch.no_grad():
        return lpips_model(prep(pred), prep(target)).item()

# ---------------------------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py /path/to/data_root /path/to/checkpoint.pt")
        return

    data_root, ckpt_path = sys.argv[1], sys.argv[2]
    out_dir = "./eval_outputs_ssim_v3"
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = SemiRestoreNet_V2(dim=cfg["dim"], num_blocks=cfg["num_blocks"], scale_factor=cfg["scale_factor"]).to(device)
    model.load_state_dict(ckpt["model_state"],strict=False)
    model.eval()

    lpips_model = lpips.LPIPS(net='alex').to(device) if _LPIPS_AVAILABLE else None
    p_low, p_high = load_calib_stats(data_root)
    full_train = SEMPairDataset(os.path.join(data_root, "train", "gt"), os.path.join(data_root, "train", "NoisyLR"),
                                 p_low, p_high, scale_factor=cfg["scale_factor"], augment=False)
    
    # 90/10 Split Consistency
    n_val = int(len(full_train) * 0.1)
    _, val_subset = torch.utils.data.random_split(full_train, [len(full_train)-n_val, n_val], 
                                                 generator=torch.Generator().manual_seed(42))

    print(f"\nEvaluating on {len(val_subset)} validation samples...")
    results = []
    
    with torch.no_grad():
        for idx in val_subset.indices:
            lr, gt = full_train[idx]
            lr_t, gt_t = lr.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)
            pred_t = model(lr_t)
            
            # 1. AI Metrics
            psnr = compute_psnr(pred_t, gt_t)
            ssim = compute_ssim(pred_t, gt_t)
            lp = compute_lpips(lpips_model, pred_t, gt_t)
            
            # 2. Metrology Metrics
            kpis = calculate_metrology_kpis(pred_t, gt_t)
            
            results.append({"name": os.path.basename(full_train.pairs[idx][0]), 
                            "psnr": psnr, "ssim": ssim, "lpips": lp, "kpis": kpis})

    # --- FINAL AGGREGATED SCORECARD ---
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])
    avg_lpips = np.mean([r['lpips'] for r in results]) if _LPIPS_AVAILABLE else 0.0

    valid_kpis = [r['kpis'] for r in results if r['kpis'] is not None]
    avg_bias = np.mean([k['cd_bias_px'] for k in valid_kpis]) if valid_kpis else 0
    avg_ler = np.mean([k['ler_error_px'] for k in valid_kpis]) if valid_kpis else 0
    avg_lwr = np.mean([k['lwr_error_px'] for k in valid_kpis]) if valid_kpis else 0
    avg_slope = np.mean([k['slope_fidelity'] for k in valid_kpis]) if valid_kpis else 0

    print(f"\n{'='*45}")
    print(f"        FINAL METROLOGY SCORECARD")
    print(f"{'='*45}")
    print(f"PSNR:           {avg_psnr:.2f} dB")
    print(f"SSIM:           {avg_ssim:.4f}")
    print(f"LPIPS:          {avg_lpips:.4f} {'(N/A)' if not _LPIPS_AVAILABLE else ''}")
    print(f"-"*45)
    print(f"Mean CD Bias:   {avg_bias:.3f} px")
    print(f"Mean LER Error: {avg_ler:.3f} px")
    print(f"Mean LWR Error: {avg_lwr:.3f} px")
    print(f"Slope Fidelity: {avg_slope*100:.1f} %")
    print(f"{'='*45}")

    # Output to file
    with open(os.path.join(out_dir, "summary_metrology.json"), "w") as f:
        json.dump(results, f, indent=2, cls=MetrologyJSONEncoder)
    print(f"\nDone. All outputs and metrology stats in {out_dir}/")

if __name__ == "__main__":
    main()