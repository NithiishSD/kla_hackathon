import numpy as np
import matplotlib.pyplot as plt
import os
import random

def browse_npy_data(lr_dir, gt_dir, num_samples=4):
    filenames = [f for f in os.listdir(lr_dir) if f.endswith('.npy')]
    samples = random.sample(filenames, num_samples)
    
    fig, axs = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    plt.suptitle("Semiconductor Structure Browser (Noisy LR vs GT)", fontsize=16)

    for i, fname in enumerate(samples):
        # Load
        lr = np.load(os.path.join(lr_dir, fname))
        gt = np.load(os.path.join(gt_dir, fname))
        
        # Simple Normalization for visualization ONLY
        # (Using local min/max just so we can see the shapes)
        lr_vis = (lr - lr.min()) / (lr.max() - lr.min() + 1e-6)
        gt_vis = (gt - gt.min()) / (gt.max() - gt.min() + 1e-6)
        
        # Display
        axs[i, 0].imshow(lr_vis, cmap='magma')
        axs[i, 0].set_title(f"Noisy LR: {fname}")
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(gt_vis, cmap='magma')
        axs[i, 1].set_title(f"Ground Truth: {fname}")
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# --- RUN IT ---
browse_npy_data('.././data/train/NoisyLR', '.././data/train/gt', num_samples=4)