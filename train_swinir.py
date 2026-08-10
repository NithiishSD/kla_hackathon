"""
Training script for the SwinIR backbone -- mirrors train.py's loop exactly
(same HardwareEngine, same loss, same dataloaders), only the model and its
config keys differ. Kept as a separate script rather than branching inside
train.py, matching this repo's existing train.py / train_baseline.py split.

Usage:
    python calibrate_stats.py /path/to/data_root
    python train_swinir.py /path/to/data_root
"""

import os
import sys
import time

import torch

from sem_dataset import build_dataloaders
from swinir_model import SwinIR, KLAMetrologyLoss, build_optimizer
from train_engine import HardwareEngine


def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(max_val ** 2 / mse)).item()


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data"

    config = dict(
        model_type="swinir",
        embed_dim=60,
        depths=(4, 4, 4, 4),
        num_heads=(6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2.0,
        batch_size=4,
        target_batch_size=32,
        scale_factor=2,
        edge_weight=0.5,
        freq_weight=0.0,
        ssim_weight=0.0,
        lr=2e-4,
        weight_decay=1e-4,
        scheduler_patience=3,
        early_stop_patience=7,
        num_workers=min(8, os.cpu_count() or 1),
        num_epochs=50,
        use_compile=False,
        checkpoint_dir="./checkpoints_swinir",
    )
    print("Config:", config)

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found. This will run on CPU (slow, "
              "for debugging only) -- HardwareEngine requires CUDA for the "
              "actual hackathon run.")
        return

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root,
        scale_factor=config["scale_factor"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    model = SwinIR(
        embed_dim=config["embed_dim"],
        depths=config["depths"],
        num_heads=config["num_heads"],
        window_size=config["window_size"],
        mlp_ratio=config["mlp_ratio"],
        scale_factor=config["scale_factor"],
    )
    criterion = KLAMetrologyLoss(
        edge_weight=config["edge_weight"],
        freq_weight=config["freq_weight"],
        ssim_weight=config["ssim_weight"],
    )

    engine = HardwareEngine(
        model,
        batch_size=config["batch_size"],
        target_batch_size=config["target_batch_size"],
    )
    criterion = criterion.to(engine.device)

    optimizer = build_optimizer(engine.model, lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config["scheduler_patience"]
    )

    if config["use_compile"]:
        engine.compile_model()

    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(config["num_epochs"]):
        t0 = time.time()

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
        print(f"Epoch {epoch+1}/{config['num_epochs']} "
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
                    print("STOPPING: model has fine-tuned at a lower LR and plateaued.")
                    break
                else:
                    print("WAITING: val_loss stalled but LR hasn't dropped yet.")

    engine.flush(optimizer)
    print("Training finished.")


if __name__ == "__main__":
    main()
