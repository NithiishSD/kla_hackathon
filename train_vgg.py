"""
Experiment: VGG Edge Loss
Tests if perceptual edge features reduce CD bias.
"""
import os, sys, time, torch
from sem_dataset import build_dataloaders
from model import SemiRestoreNet_V2, KLAMetrologyLoss, VGGEdgeLoss, build_optimizer
from train_engine import HardwareEngine

def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0: return float('inf')
    return (10 * torch.log10(max_val ** 2 / mse)).item()

def main():
    data_root = sys.argv[1]
    resume_ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    
    config = dict(
        dim=64, num_blocks=2, batch_size=4, target_batch_size=32, scale_factor=2,
        edge_weight=0.0,          # Sobel off
        freq_weight=0.0,          # Off
        ssim_weight=0.0,          # Off
        vgg_weight=0.1,           # VGG edge loss — ONLY change from baseline_v3
        charbonnier_weight=1.0,
        lr=5e-05, weight_decay=1e-04,
        scheduler_patience=5, early_stop_patience=10,
        num_workers=8, num_epochs=20, use_compile=False,
        checkpoint_dir="./checkpoints_vgg_v1",
    )
    
    print("=== VGG EDGE LOSS EXPERIMENT ===")
    print("Config:", config)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    train_loader, val_loader, _ = build_dataloaders(
        data_root, scale_factor=config["scale_factor"],
        batch_size=config["batch_size"], num_workers=config["num_workers"])
    
    model = SemiRestoreNet_V2(dim=config["dim"], num_blocks=config["num_blocks"],
                               scale_factor=config["scale_factor"])
    
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location='cuda')
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"Resumed from {resume_ckpt}")
    
    engine = HardwareEngine(model, batch_size=config["batch_size"],
                             target_batch_size=config["target_batch_size"])
    
    criterion_char = KLAMetrologyLoss(edge_weight=0.0, freq_weight=0.0,
                                   ssim_weight=0.0, charbonnier_weight=1.0)
    criterion_vgg = VGGEdgeLoss(device=engine.device)
    
    optimizer = build_optimizer(engine.model, lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config["scheduler_patience"])
    
    best_val_loss = float('inf')
    
    for epoch in range(config["num_epochs"]):
        t0 = time.time()
        
        # Train
        train_losses = []
        for lr_img, gt_img in train_loader:
            lr_img = lr_img.to(engine.device)
            gt_img = gt_img.to(engine.device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=engine.precision):
                pred = engine.model(lr_img)
                loss = criterion_char(pred, gt_img) + config["vgg_weight"] * criterion_vgg(pred, gt_img)
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train = sum(train_losses) / len(train_losses)
        
        # Validate
        val_losses, val_psnrs = [], []
        for lr_img, gt_img in val_loader:
            lr_img = lr_img.to(engine.device)
            gt_img = gt_img.to(engine.device)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=engine.precision):
                    pred = engine.model(lr_img)
                    loss = criterion_char(pred, gt_img)
            val_losses.append(loss.item())
            val_psnrs.append(compute_psnr(pred, gt_img))
        
        avg_val = sum(val_losses) / len(val_losses)
        avg_psnr = sum(val_psnrs) / len(val_psnrs)
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val)
        new_lr = optimizer.param_groups[0]['lr']
        
        dt = time.time() - t0
        print(f"[VGG] E{epoch+1}/{config['num_epochs']} [{dt:.0f}s] "
              f"train={avg_train:.4f} val={avg_val:.4f} psnr={avg_psnr:.2f}dB")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({"model_state": engine.model.state_dict(), "config": config,
                        "epoch": epoch, "val_loss": avg_val, "val_psnr": avg_psnr},
                       os.path.join(config["checkpoint_dir"], "best_model.pt"))
    
    print("Done. Evaluate with evaluate.py")

if __name__ == "__main__":
    main()