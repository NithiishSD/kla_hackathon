"""
Quick diagnostic: extracts line profiles from prediction vs ground truth
to visually confirm CD Bias reduction.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SemiRestoreNet_V2

def extract_line_profile(img, y_center=None, x_range=None, num_lines=1):
    """
    Extract horizontal intensity profile through the center of the image.
    If the image has obvious line structures, this profile crosses them
    and shows edge sharpness and position.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = img.squeeze()  # Remove batch/channel dims
    
    H, W = img.shape
    if y_center is None:
        y_center = H // 2
    if x_range is None:
        x_range = (0, W)
    
    # Average multiple adjacent lines for noise reduction
    profiles = []
    for dy in range(-(num_lines//2), num_lines//2 + 1):
        y = min(max(y_center + dy, 0), H-1)
        profiles.append(img[y, x_range[0]:x_range[1]])
    return np.mean(profiles, axis=0)

def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python diagnose_cd.py /path/to/best_model.pt /path/to/low_res.npy /path/to/gt.npy")
        return
    
    ckpt_path = sys.argv[1]
    lr_path = sys.argv[2]
    gt_path = sys.argv[3]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {"dim": 64, "num_blocks": 2, "scale_factor": 2})
    model = SemiRestoreNet_V2(dim=config["dim"], num_blocks=config["num_blocks"],
                               scale_factor=config["scale_factor"]).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    
    # Load data
    lr = torch.from_numpy(np.load(lr_path)).float().unsqueeze(0).unsqueeze(0).to(device)
    gt = torch.from_numpy(np.load(gt_path)).float().squeeze()
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = model(lr).cpu().squeeze().numpy()
    lr_upsampled = torch.nn.functional.interpolate(
        lr, scale_factor=config["scale_factor"], mode='bicubic', align_corners=False
    ).cpu().squeeze().numpy()
    
    # Extract line profiles
    profile_pred = extract_line_profile(pred, num_lines=5)
    profile_gt = extract_line_profile(gt, num_lines=5)
    profile_lr = extract_line_profile(lr_upsampled, num_lines=5)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].imshow(lr_upsampled, cmap='gray')
    axes[0].set_title('Low-Res (Bicubic)')
    axes[0].axhline(y=pred.shape[0]//2, color='r', linestyle='--', alpha=0.5)
    
    axes[1].imshow(pred, cmap='gray')
    axes[1].set_title('Model Output')
    axes[1].axhline(y=pred.shape[0]//2, color='r', linestyle='--', alpha=0.5)
    
    axes[2].imshow(gt, cmap='gray')
    axes[2].set_title('Ground Truth')
    axes[2].axhline(y=gt.shape[0]//2, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('diagnostic_images.png', dpi=150)
    
    # Line profile plot
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(profile_gt))
    ax.plot(x, profile_gt, 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(x, profile_pred, 'b-', linewidth=2, label='Model Output', alpha=0.8)
    ax.plot(x, profile_lr, 'gray', linewidth=1, label='Bicubic Input', alpha=0.5)
    ax.set_xlabel('Pixel Position')
    ax.set_ylabel('Intensity')
    ax.set_title('Line Profile Comparison (crossing features)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('diagnostic_profile.png', dpi=150)
    
    print("Saved diagnostic_images.png and diagnostic_profile.png")
    
    # Estimate CD bias from the profile
    # Find 50% threshold crossing points
    thresh = 0.5
    gt_crossings = np.where(np.diff((profile_gt > thresh).astype(int)))[0]
    pred_crossings = np.where(np.diff((profile_pred > thresh).astype(int)))[0]
    
    print(f"\nGT edge positions (50% threshold): {gt_crossings}")
    print(f"Pred edge positions (50% threshold): {pred_crossings}")
    
    if len(gt_crossings) >= 2 and len(pred_crossings) >= 2:
        gt_width = gt_crossings[1] - gt_crossings[0] if len(gt_crossings) >= 2 else None
        pred_width = pred_crossings[1] - pred_crossings[0] if len(pred_crossings) >= 2 else None
        if gt_width and pred_width:
            cd_bias = pred_width - gt_width
            print(f"GT feature width: {gt_width:.1f} px")
            print(f"Pred feature width: {pred_width:.1f} px")
            print(f"Estimated CD Bias: {cd_bias:.1f} px (negative = shrinking)")

if __name__ == "__main__":
    main()