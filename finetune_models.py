"""
Phase 3 fine-tune: resumes from your EXISTING Phase 2 checkpoint and
continues training with SSIM loss added, at a lower LR. This is
deliberately NOT a from-scratch run -- the architecture is unchanged
(same num_blocks/dim), so the existing weights are a valid starting
point; only the loss function changes. Much cheaper than 50 fresh
epochs, and doesn't throw away what Phase 2 already learned.

If you later adopt an architecture change (e.g. low-rank MDTA), THAT
does need a fresh run -- this script is specifically for the loss-only
change discussed (SSIM term addressing the "melting"/over-smoothing
diagnosis).

Usage:
    python finetune_ssim.py /path/to/data_root ./checkpoints/best_model.pt
"""

import os
import sys
import time

import torch

from sem_dataset import build_dataloaders
from model import SemiRestoreNet_V2, KLAMetrologyLoss, build_optimizer
from train_engine import HardwareEngine


def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(max_val ** 2 / mse)).item()


def main():
    if len(sys.argv) < 3:
        print("Usage: python finetune_ssim.py /path/to/data_root /path/to/checkpoint.pt")
        return
    data_root = sys.argv[1]
    resume_ckpt_path = sys.argv[2]

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found.")
        return

    ckpt = torch.load(resume_ckpt_path, map_location="cuda")
    base_cfg = ckpt["config"]
    print(f"Resuming from {resume_ckpt_path} "
          f"(epoch {ckpt['epoch']+1}, val_psnr={ckpt['val_psnr']:.2f}dB)")

    # Same architecture as the checkpoint -- only the loss changes here.
    config = dict(base_cfg)
#     config.update(dict(
#     num_blocks=3,  # keep same architecture
#     ssim_weight=0.8,        # small, single new signal — was 0.4
#     lr=5e-5,                # real fine-tune LR, 20x lower — was 1e-4
#     edge_weight=1.5,       # KEEP baseline's 0.5, don't touch
#     weight_decay=1e-6,     # KEEP baseline's 1e-4, don't touch
#     freq_weight=0.15,        # still off — isolate one variable
#     num_epochs=20,
#     scheduler_patience=2,
#     early_stop_patience=5,
#     checkpoint_dir="./checkpoints_ssim_v4",
# ))
    print("=== baseline FINE-TUNE ===")
    print("Config:", config)

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root, scale_factor=config["scale_factor"],
        batch_size=config["batch_size"], num_workers=config.get("num_workers", 4),
    )

    model = SemiRestoreNet_V2(dim=config["dim"], num_blocks=config["num_blocks"],
                               scale_factor=config["scale_factor"])
    model.load_state_dict(ckpt["model_state"], strict=False) 
    print("Loaded Phase 2 weights. Missing 'metrology_gain' initialized to default (1.0).")

    criterion = KLAMetrologyLoss(edge_weight=config["edge_weight"],
                                  freq_weight=config.get("freq_weight", 0.0),
                                  ssim_weight=config["ssim_weight"])

    engine = HardwareEngine(model, batch_size=config["batch_size"],
                             target_batch_size=config["target_batch_size"])
    criterion = criterion.to(engine.device)

    optimizer = build_optimizer(engine.model, lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config["scheduler_patience"]
    )

    best_val_loss = float('inf')
    epochs_since_improvement = 0
    target_ssim_weight = config["ssim_weight"] # e.g., 0.8
    ramp_epochs = 5 # Number of epochs to reach full strength
    for epoch in range(config["num_epochs"]):
        t0 = time.time()
        current_ssim_w = min(target_ssim_weight, target_ssim_weight * (epoch + 1) / ramp_epochs)
        criterion.ssim_weight = current_ssim_w
    
        print(f"\n--- Epoch {epoch+1} | Active SSIM Weight: {current_ssim_w:.4f} ---")
        train_losses = []
        for lr_img, gt_img in train_loader:
            loss = engine.train_step(optimizer, criterion, None, lr_img, gt_img)
            train_losses.append(loss)
        avg_train_loss = sum(train_losses) / len(train_losses)

        val_losses, val_psnrs = [], []
        for lr_img, gt_img in val_loader:
            loss, pred = engine.eval_step(criterion, lr_img, gt_img)
            val_losses.append(loss)
            val_psnrs.append(compute_psnr(pred, gt_img.to(engine.device)))
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_psnr = sum(val_psnrs) / len(val_psnrs)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        dt = time.time() - t0
        lr_msg = (f" [LR: {old_lr:.2e}]" if old_lr == new_lr
                  else f" [LR REDUCED: {old_lr:.2e} -> {new_lr:.2e}]")
        print(f"[SSIM-FT] Epoch {epoch+1}/{config['num_epochs']} "
              f"[{dt:.1f}s]{lr_msg} train_loss={avg_train_loss:.4f} "
              f"val_loss={avg_val_loss:.4f} val_psnr={avg_val_psnr:.2f}dB")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            ckpt_path = os.path.join(config["checkpoint_dir"], "best_model.pt")
            state_dict = (engine.model.module.state_dict()
                          if isinstance(engine.model, torch.nn.DataParallel)
                          else engine.model.state_dict())
            torch.save({"model_state": state_dict, "config": config,
                        "epoch": epoch, "val_loss": avg_val_loss,
                        "val_psnr": avg_val_psnr, "lr": new_lr}, ckpt_path)
            print(f"  -> new best val_loss, saved to {ckpt_path}")
        else:
            epochs_since_improvement += 1
            print(f"  -> val_loss did not improve "
                  f"({epochs_since_improvement}/{config['early_stop_patience']})")
            if epochs_since_improvement >= config["early_stop_patience"]:
                if new_lr < config["lr"]:
                    print("STOPPING: fine-tune has plateaued at lower LR.")
                    break
                else:
                    print("WAITING: val_loss stalled but LR hasn't dropped yet.")

    engine.flush(optimizer)
    print("Fine-tune finished. Compare against Phase 2 and baseline "
          "checkpoints using evaluate.py -- and specifically look at "
          "the worst-case samples (like 000352/002982) for whether "
          "SSIM loss actually reduced the blur/melting.")


if __name__ == "__main__":
    main()