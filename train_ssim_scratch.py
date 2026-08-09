import os
import sys
import time
import torch
import torch.nn as nn

from sem_dataset import build_dataloaders
from model import SemiRestoreNet_V2, KLAMetrologyLoss, build_optimizer
from train_engine import HardwareEngine

def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0: return float('inf')
    return (10 * torch.log10(max_val ** 2 / mse)).item()

def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data"

    # ---- CONFIGURATION FOR FROM-SCRATCH RUN ----
    config = dict(
        dim=64,
        num_blocks=2,          # High depth for deblurring
        scale_factor=2,
        edge_weight=0.5,       # Sharp edges
        ssim_weight=0.1,       # CRITICAL: Active from the start
        freq_weight=0.0,       # Clean the sandstorm noise
        lr=2e-4,               # Standard starting LR
        weight_decay=1e-4,     # Standard regularization
        batch_size=4,          # Safe for 6GB VRAM (RTX 4050)
        target_batch_size=32,  # Simulated batch size via HardwareEngine
        num_epochs=60,         # Longer run for from-scratch
        scheduler_patience=3,
        early_stop_patience=8,
        checkpoint_dir="./checkpoints_scratch_ssim",
    )

    print("=== STARTING FROM-SCRATCH STRUCTURAL TRAINING ===")
    print("Config:", config)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    if not torch.cuda.is_available():
        print("Error: CUDA not found.")
        return

    # 1. DATA LOADERS
    train_loader, val_loader, _ = build_dataloaders(
        data_root, 
        scale_factor=config["scale_factor"],
        batch_size=config["batch_size"]
    )

    # 2. MODEL INITIALIZATION (Zero-Residual Init is handled in model.py)
    model = SemiRestoreNet_V2(
        dim=config["dim"], 
        num_blocks=config["num_blocks"],
        scale_factor=config["scale_factor"]
    )

    # 3. LOSS & ENGINE
    criterion = KLAMetrologyLoss(
        edge_weight=config["edge_weight"],
        freq_weight=config["freq_weight"],
        ssim_weight=config["ssim_weight"]
    )

    engine = HardwareEngine(model, batch_size=config["batch_size"], 
                             target_batch_size=config["target_batch_size"])
    criterion = criterion.to(engine.device)

    # 4. OPTIMIZER & SCHEDULER
    optimizer = build_optimizer(engine.model, lr=config["lr"], 
                                 weight_decay=config["weight_decay"])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config["scheduler_patience"], verbose=True
    )

    best_val_loss = float('inf')
    epochs_since_improvement = 0

    # 5. MAIN LOOP
    for epoch in range(config["num_epochs"]):
        t0 = time.time()
        
        # Training Phase
        engine.model.train()
        train_losses = []
        for lr_img, gt_img in train_loader:
            loss = engine.train_step(optimizer, criterion, None, lr_img, gt_img)
            train_losses.append(loss)
        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation Phase
        engine.model.eval()
        val_losses, val_psnrs = [], []
        with torch.no_grad():
            for lr_img, gt_img in val_loader:
                loss, pred = engine.eval_step(criterion, lr_img, gt_img)
                val_losses.append(loss)
                val_psnrs.append(compute_psnr(pred, gt_img.to(engine.device)))
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_psnr = sum(val_psnrs) / len(val_psnrs)

        # Step Scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{config['num_epochs']} [{dt:.1f}s] "
              f"LR: {old_lr:.2e} | Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | PSNR: {avg_val_psnr:.2f}dB")

        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            ckpt_path = os.path.join(config["checkpoint_dir"], "best_model.pt")
            torch.save({
                "model_state": engine.model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_psnr": avg_val_psnr
            }, ckpt_path)
            print(f"  -> saved new best model ({avg_val_psnr:.2f}dB)")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= config["early_stop_patience"]:
                print("Early stopping triggered.")
                break

    print("Scratch training finished.")

if __name__ == "__main__":
    main()