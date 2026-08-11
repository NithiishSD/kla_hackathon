"""
THE MISSING EXPERIMENT: Measure CD on GT, Bicubic, and Model output.
Saves all three for external CD measurement.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from sem_dataset import build_dataloaders
from model import SemiRestoreNet_V2

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load best model
    ckpt = torch.load("./checkpoints_baseline_v2/best_model.pt", map_location=device)
    model = SemiRestoreNet_V2(dim=64, num_blocks=2, scale_factor=2).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print(f"Loaded model: PSNR={ckpt.get('val_psnr', '?')}dB")
    
    _, val_loader, _ = build_dataloaders("./data", scale_factor=2, batch_size=1, num_workers=0)
    
    os.makedirs("./cd_baseline_test", exist_ok=True)
    
    count = 0
    for lr, gt in val_loader:
        if count >= 50:
            break
        
        lr = lr.to(device)
        gt = gt.to(device)
        
        # Bicubic upsampling
        bicubic = F.interpolate(lr, scale_factor=2, mode='bicubic', align_corners=False)
        
        # Model output
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(lr)
        
        # Save all three
        np.save(f"./cd_baseline_test/sample_{count:04d}_gt.npy", gt.squeeze().cpu().numpy())
        np.save(f"./cd_baseline_test/sample_{count:04d}_bicubic.npy", bicubic.squeeze().cpu().numpy())
        np.save(f"./cd_baseline_test/sample_{count:04d}_model.npy", pred.squeeze().cpu().numpy())
        
        count += 1
    
    print(f"Saved {count} samples to ./cd_baseline_test/")
    print("Next: Run your external CD measurement tool on these three sets:")
    print("  1. *_gt.npy       (Ground Truth)")
    print("  2. *_bicubic.npy  (Bicubic Upsampled)")
    print("  3. *_model.npy    (Model Output)")
    print("Compare the CD measurements.")

if __name__ == "__main__":
    main()