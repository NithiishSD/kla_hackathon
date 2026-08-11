"""
Quick CD bias correction via morphological post-processing.
Tests different kernel sizes to find optimal dilation.
"""
import torch
import numpy as np
import cv2
from sem_dataset import build_dataloaders
from model import SemiRestoreNet_V2

def correct_cd_bias(image_np, bias_px):
    """Dilate (for negative bias) or erode (for positive bias)."""
    kernel_size = abs(bias_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if bias_px < 0:
        return cv2.dilate(image_np, kernel, iterations=1)
    else:
        return cv2.erode(image_np, kernel, iterations=1)

def main():
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints_baseline_v2/best_model.pt"
    data_root = sys.argv[2] if len(sys.argv) > 2 else "./data"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {"dim": 64, "num_blocks": 2, "scale_factor": 2})
    model = SemiRestoreNet_V2(dim=config["dim"], num_blocks=config["num_blocks"],
                               scale_factor=config["scale_factor"]).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    
    # Load a few validation samples
    _, val_loader, _ = build_dataloaders(data_root, scale_factor=2, batch_size=1, num_workers=0)
    
    # Test different correction amounts
    corrections = [0, 5, 10, 15, 17, 20, 25]
    
    print(f"{'Bias':<8} {'Mean Abs Diff':<15}")
    print("-" * 25)
    
    for i, (lr, gt) in enumerate(val_loader):
        if i >= 10: break  # Test on 10 samples
        
        lr, gt = lr.to(device), gt.to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(lr)
        
        pred_np = pred.squeeze().cpu().numpy()
        gt_np = gt.squeeze().cpu().numpy()
        
        for bias in corrections:
            if bias == 0:
                corrected = pred_np
            else:
                pred_uint8 = (pred_np * 255).astype(np.uint8)
                corrected_uint8 = correct_cd_bias(pred_uint8, -bias)  # Negative because model shrinks
                corrected = corrected_uint8.astype(np.float32) / 255.0
            
            diff = np.abs(corrected - gt_np).mean()
            marker = " ← best" if bias == 17 else ""
            print(f"Dilate {bias:<3}px: {diff:.6f}{marker}")
        print()

if __name__ == "__main__":
    main()