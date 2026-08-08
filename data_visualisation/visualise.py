import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

def run_diagnostic(lr_path, gt_path):
    # 1. Load raw float32 data
    lr = np.load(lr_path)  # 128x128
    gt = np.load(gt_path)  # 256x256
    
    # 2. Rescale GT for direct pixel comparison
    gt_down = cv2.resize(gt, (128, 128), interpolation=cv2.INTER_AREA)
    residual = lr - gt_down # The "Noise + Artifact" signal

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    plt.suptitle(f"KLA Metrology Diagnostic: {os.path.basename(lr_path)}", fontsize=16)

    # --- PLOT 1: Visual Side-by-Side ---
    axs[0, 0].imshow(lr, cmap='magma')
    axs[0, 0].set_title("Noisy LR (Input)")
    axs[0, 1].imshow(gt, cmap='magma')
    axs[0, 1].set_title("Ground Truth (Target)")

    # --- PLOT 2: The Error Map (What the AI must learn) ---
    # High intensity in the error map means high noise/blur areas
    im_err = axs[0, 2].imshow(np.abs(residual), cmap='hot')
    plt.colorbar(im_err, ax=axs[0, 2])
    axs[0, 2].set_title("Absolute Error Map")

    # --- PLOT 3: Intensity Histogram (Outlier Check) ---
    axs[1, 0].hist(lr.flatten(), bins=100, color='blue', alpha=0.5, label='LR (Noisy)')
    axs[1, 0].hist(gt.flatten(), bins=100, color='green', alpha=0.5, label='GT')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title("Intensity Histogram (Log Scale)")
    axs[1, 0].legend()

    # --- PLOT 4: Edge Profile Analysis (Blur Check) ---
    # We take a horizontal "slice" through a circuit line
    mid_row = 64
    axs[1, 1].plot(gt_down[mid_row, :], label='GT Edge', color='green', linewidth=2)
    axs[1, 1].plot(lr[mid_row, :], label='Noisy Edge', color='blue', alpha=0.7)
    axs[1, 1].set_title("Edge Profile Cross-Section")
    axs[1, 1].legend()

    # --- PLOT 5: Fourier Spectrum (Periodic Noise Check) ---
    f = np.fft.fft2(lr)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
    axs[1, 2].imshow(magnitude_spectrum, cmap='viridis')
    axs[1, 2].set_title("FFT Power Spectrum (Look for spikes)")

    plt.tight_layout()
    plt.show()

# Run on a few samples from your data/train/ folder
# run_diagnostic('path/to/lr.npy', 'path/to/gt.npy')