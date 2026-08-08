import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def get_targeted_samples(lr_dir, gt_dir, num_per_cat=1):
    files = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))
    stats = []

    print("Scanning dataset for extremes...")
    for f in files[:500]: # Scan a subset for speed
        img = np.load(f)
        # Calculate Edge Density (structural complexity)
        edges = np.mean(np.abs(np.gradient(img)))
        # Calculate Variance (noise level)
        var = np.var(img)
        # Calculate Mean (brightness)
        mean = np.mean(img)
        stats.append({'name': os.path.basename(f), 'edges': edges, 'var': var, 'mean': mean})

    # --- CATEGORY 1: The "High Complexity" (Dense Lines/Vias) ---
    cat_high_complex = sorted(stats, key=lambda x: x['edges'])[-1]
    
    # --- CATEGORY 2: The "High Noise" (Worst Speckle) ---
    cat_high_noise = sorted(stats, key=lambda x: x['var'])[-1]

    # --- CATEGORY 3: The "Low Contrast" (Deep Trenches) ---
    cat_low_contrast = sorted(stats, key=lambda x: x['mean'])[0]

    # --- CATEGORY 4: The "OOD Test Check" (Random Test Sample) ---
    # Pick a random sample from the test folder
    test_files = glob.glob(os.path.join('data/test', "*.npy"))
    test_sample = os.path.basename(test_files[0]) if test_files else None

    return {
        "High Complexity": cat_high_complex['name'],
        "High Noise": cat_high_noise['name'],
        "Low Contrast": cat_low_contrast['name'],
        "Test Sample": test_sample
    }

# Execute and get the filenames
targets = get_targeted_samples('.././data/train/lossylr', '.././data/train/gt')
print(targets)