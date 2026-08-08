import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# ==========================================
# CONFIGURATION: Update these paths
# ==========================================
DATA_ROOT = ".././data" # Change this to your data directory
TARGETS = {
    "High Complexity": "train/NoisyLR/000352.npy",
    "High Noise": "train/NoisyLR/000218.npy",
    "Low Contrast": "train/NoisyLR/000425.npy",
    "Test Sample": "test/000014.npy" # No GT available for this
}

def run_analytical_test(sample_name, lr_path, gt_path=None):
    if not os.path.exists(lr_path):
        print(f"File not found: {lr_path}")
        return

    # 1. Load Data
    lr = np.load(lr_path).astype(np.float32)
    has_gt = gt_path is not None and os.path.exists(gt_path)
    
    if has_gt:
        gt = np.load(gt_path).astype(np.float32)
        h, w = lr.shape
        gt_down = cv2.resize(gt, (w, h), interpolation=cv2.INTER_AREA)
        residual = lr - gt_down
        v_min, v_max = np.percentile(gt, (1, 99))
    else:
        v_min, v_max = np.percentile(lr, (1, 99))

    # Create Dashboard
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    plt.suptitle(f"KLA SCIENTIFIC AUDIT: {sample_name} ({os.path.basename(lr_path)})", 
                 fontsize=22, fontweight='bold', color='#2c3e50')

    # --- PLOT 1: Hardware View ---
    axs[0, 0].imshow(lr, cmap='gray', vmin=v_min, vmax=v_max)
    axs[0, 0].set_title("Input: SEM Sensor View (Grayscale)", fontsize=14)
    axs[0, 0].axis('off')

    # --- PLOT 2: FFT Spectrum (The Artifact Detector) ---
    f_shift = np.fft.fftshift(np.fft.fft2(lr))
    spectrum = 20 * np.log(np.abs(f_shift) + 1e-6)
    im2 = axs[0, 1].imshow(spectrum, cmap='viridis')
    axs[0, 1].set_title("FFT Spectrum: Look for bright lines/dots", fontsize=14)
    axs[0, 1].axis('off')

    # --- PLOT 3: Intensity Histogram ---
    axs[0, 2].hist(lr.flatten(), bins=100, color='red', alpha=0.3, label='Noisy Input', log=True)
    if has_gt:
        axs[0, 2].hist(gt_down.flatten(), bins=100, color='green', alpha=0.5, label='Ground Truth', log=True)
    axs[0, 2].set_title("Intensity Dist. (Log Scale)", fontsize=14)
    axs[0, 2].legend()

    if has_gt:
        # --- PLOT 4: Absolute Error Map ---
        im4 = axs[1, 0].imshow(np.abs(residual), cmap='hot')
        plt.colorbar(im4, ax=axs[1, 0])
        axs[1, 0].set_title("Degradation Map (Error)", fontsize=14)
        axs[1, 0].axis('off')

        # --- PLOT 5: Edge Profile ---
        best_row = np.var(gt_down, axis=1).argmax()
        axs[1, 1].plot(gt_down[best_row, :], label='Ground Truth', color='green', linewidth=3)
        axs[1, 1].plot(lr[best_row, :], label='Noisy Input', color='red', alpha=0.6)
        axs[1, 1].set_title(f"Edge Sharpness (Row {best_row})", fontsize=14)
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.2)

        # --- PLOT 6: Noise Signature (Scatter) ---
        sampled_gt = gt_down.flatten()[::5]
        sampled_res = residual.flatten()[::5]
        axs[1, 2].scatter(sampled_gt, sampled_res, alpha=0.05, s=1, color='purple')
        axs[1, 2].axhline(y=0, color='black', linestyle='--')
        axs[1, 2].set_xlabel("Signal Intensity (GT)")
        axs[1, 2].set_ylabel("Noise Magnitude (LR - GT)")
        axs[1, 2].set_title("Noise Signature (Look for Fan shape)", fontsize=14)
    else:
        # Placeholder for Test Sample
        axs[1, 0].text(0.5, 0.5, "NO GROUND TRUTH\nAVAILABLE FOR TEST SET", 
                      ha='center', va='center', fontsize=16, color='gray')
        axs[1, 0].axis('off')
        
        # Plot Local Variance for Test Set instead
        local_var = cv2.GaussianBlur(lr**2, (5,5), 0) - cv2.GaussianBlur(lr, (5,5), 0)**2
        axs[1, 1].imshow(local_var, cmap='magma')
        axs[1, 1].set_title("Estimated Noise Variance Map", fontsize=14)
        axs[1, 1].axis('off')

        axs[1, 2].text(0.5, 0.5, "SCATTER ANALYSIS\nREQUIRES GT PAIR", 
                      ha='center', va='center', fontsize=16, color='gray')
        axs[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- EXECUTE THE AUDIT ---
for cat, path in TARGETS.items():
    lr_full_path = os.path.join(DATA_ROOT, path)
    
    # Try to find matching GT if it's a training file
    gt_full_path = None
    if "train" in path:
        # Assuming path like 'train/NoisyLR/000352.npy' -> 'train/gt/000352.npy'
        gt_full_path = lr_full_path.replace("NoisyLR", "gt")
    
    run_analytical_test(cat, lr_full_path, gt_full_path)