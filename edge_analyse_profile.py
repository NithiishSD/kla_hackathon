"""
Multi-threshold edge profile analysis.
Reveals whether CD bias is spatial shift or profile deformation.
"""
import numpy as np
import matplotlib.pyplot as plt

def extract_profile(image, y_pos=None):
    """Extract horizontal intensity profile at given y position."""
    if y_pos is None:
        y_pos = image.shape[0] // 2
    return image[y_pos, :]

def find_best_profile_row(image):
    """Find a row with clear features (high contrast)."""
    best_row = 0
    best_contrast = 0
    for y in range(image.shape[0]):
        row = image[y, :]
        contrast = row.max() - row.min()
        if contrast > best_contrast:
            best_contrast = contrast
            best_row = y
    return best_row

def measure_edges(profile, thresholds):
    """Find edge positions at multiple thresholds."""
    p_min = profile.min()
    p_max = profile.max()
    p_norm = (profile - p_min) / (p_max - p_min + 1e-8)
    
    results = {}
    for t in thresholds:
        above = p_norm > t
        edges = np.where(np.diff(above.astype(int)))[0]
        
        if len(edges) >= 2:
            rising = edges[0] if not above[0] else edges[1] if len(edges) > 1 else None
            falling = edges[1] if not above[0] else edges[2] if len(edges) > 2 else None
            if rising is not None and falling is not None:
                results[t] = {
                    'width': falling - rising,
                    'rising': rising,
                    'falling': falling,
                    'midpoint': (rising + falling) / 2
                }
    return results

def main():
    # Load first sample
    gt = np.load("./cd_baseline_test/sample_0000_gt.npy")
    bicubic = np.load("./cd_baseline_test/sample_0000_bicubic.npy")
    model = np.load("./cd_baseline_test/sample_0000_model.npy")
    
    # Find best row
    row = find_best_profile_row(gt)
    print(f"Analyzing row {row}")
    
    # Extract profiles
    p_gt = extract_profile(gt, row)
    p_bicubic = extract_profile(bicubic, row)
    p_model = extract_profile(model, row)
    
    # Normalize for display
    p_gt_norm = (p_gt - p_gt.min()) / (p_gt.max() - p_gt.min() + 1e-8)
    p_bicubic_norm = (p_bicubic - p_bicubic.min()) / (p_bicubic.max() - p_bicubic.min() + 1e-8)
    p_model_norm = (p_model - p_model.min()) / (p_model.max() - p_model.min() + 1e-8)
    
    # Measure at multiple thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    edges_gt = measure_edges(p_gt, thresholds)
    edges_bicubic = measure_edges(p_bicubic, thresholds)
    edges_model = measure_edges(p_model, thresholds)
    
    # Compute CD bias for each threshold
    valid_thresholds = []
    bias_bicubic = []
    bias_model = []
    
    for t in thresholds:
        if t in edges_gt and t in edges_bicubic and t in edges_model:
            valid_thresholds.append(t)
            bias_bicubic.append(edges_bicubic[t]['width'] - edges_gt[t]['width'])
            bias_model.append(edges_model[t]['width'] - edges_gt[t]['width'])
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Profiles
    ax = axes[0]
    ax.plot(p_gt, 'k-', label='GT', linewidth=2, alpha=0.8)
    ax.plot(p_bicubic, 'b--', label='Bicubic', linewidth=1.5, alpha=0.7)
    ax.plot(p_model, 'r-', label='Model', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Pixel Position')
    ax.set_ylabel('Intensity')
    ax.set_title('Edge Profiles (Raw)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Normalized profiles with thresholds
    ax = axes[1]
    ax.plot(p_gt_norm, 'k-', label='GT', linewidth=2)
    ax.plot(p_model_norm, 'r-', label='Model', linewidth=1.5)
    for t in [0.3, 0.5, 0.7]:
        ax.axhline(y=t, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Pixel Position')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Normalized Profiles with Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. CD Bias vs Threshold
    ax = axes[2]
    ax.plot(valid_thresholds, bias_bicubic, 'bs--', label='Bicubic - GT', markersize=8)
    ax.plot(valid_thresholds, bias_model, 'ro-', label='Model - GT', markersize=8)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('CD Bias (pixels)')
    ax.set_title('CD Bias vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('edge_profile_analysis.png', dpi=150)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"EDGE PROFILE ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(f"{'Threshold':<12} {'Bicubic Bias':<15} {'Model Bias':<15}")
    print(f"{'-'*42}")
    for t, bb, bm in zip(valid_thresholds, bias_bicubic, bias_model):
        print(f"{t:<12.1f} {bb:<15.2f} {bm:<15.2f}")
    
    if 0.5 in valid_thresholds:
        idx = valid_thresholds.index(0.5)
        print(f"\nAt 50% threshold:")
        print(f"  Bicubic CD bias: {bias_bicubic[idx]:.2f} px")
        print(f"  Model CD bias:   {bias_model[idx]:.2f} px")
    
    print(f"\nSaved edge_profile_analysis.png")
    
    # Key interpretation
    print(f"\n{'='*50}")
    print(f"INTERPRETATION:")
    print(f"{'='*50}")
    if len(bias_bicubic) > 0 and len(bias_model) > 0:
        bicubic_range = max(bias_bicubic) - min(bias_bicubic)
        model_range = max(bias_model) - min(bias_model)
        
        if bicubic_range > 5:
            print("⚠️  Bicubic CD bias varies with threshold → profile deformation exists in input")
        if model_range > 5:
            print("⚠️  Model CD bias varies with threshold → model changes edge profile shape")
        if abs(bias_bicubic[0]) > 5:
            print("⚠️  Bicubic already has significant CD bias → check 128→256 generation")
        if abs(bias_model[0] - bias_bicubic[0]) < 3:
            print("✅ Model preserves bicubic CD bias → problem is upstream, not the model")

if __name__ == "__main__":
    main()