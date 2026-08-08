"""
Ready-to-run training script.

Usage:
    python calibrate_stats.py /path/to/data_root      # once, before training
    python train.py /path/to/data_root

Config is at the top of main() -- edit dim/num_blocks/freq_weight there for
your baseline vs. metrology HPO runs.
"""

import os
import sys
import time

import torch

from sem_dataset import build_dataloaders
from model import SemiRestoreNet_V2, KLAMetrologyLoss, build_optimizer
from train_engine import HardwareAwareEngine


def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(max_val ** 2 / mse)).item()


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data"

    # ---- Config: edit this block for baseline vs. metrology HPO runs ----
    config = dict(
        dim=64,
        num_blocks=2,
        scale_factor=2,
        edge_weight=0.5,
        freq_weight=0.0,      # set to 0.1 for the "metrology" run
        lr=2e-4,
        weight_decay=1e-4,
        batch_size=8,
        num_workers=min(8, os.cpu_count() or 1),
        num_epochs=30,
        use_compile=False,    # flip on only after verifying eager works
        checkpoint_dir="./checkpoints",
        patience=5,           # epochs of rising val loss before early stop
    )
    print("Config:", config)

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found. This will run on CPU (slow, "
              "for debugging only) -- HardwareAwareEngine requires CUDA "
              "for the actual hackathon run.")
        return

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # ---- Data ----
    train_loader, val_loader, test_loader = build_dataloaders(
        data_root,
        scale_factor=config["scale_factor"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # ---- Model / loss / optimizer ----
    model = SemiRestoreNet_V2(
        dim=config["dim"],
        num_blocks=config["num_blocks"],
        scale_factor=config["scale_factor"],
    )
    criterion = KLAMetrologyLoss(
        edge_weight=config["edge_weight"],
        freq_weight=config["freq_weight"],
    )

    engine = HardwareAwareEngine(model)
    criterion = criterion.to(engine.device)

    optimizer = build_optimizer(engine.model, lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    total_steps = config["num_epochs"] * len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["lr"], total_steps=total_steps
    )

    if config["use_compile"]:
        engine.compile_model()

    # ---- Train loop with explicit overfitting guard ----
    # This is the check your team's plan named as the most important rule
    # ("if val loss rises while train loss falls, stop immediately") --
    # implemented here as an actual early-stop trigger, not just something
    # to watch for manually.
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(config["num_epochs"]):
        t0 = time.time()

        train_losses = []
        for lr_img, gt_img in train_loader:
            loss = engine.train_step(optimizer, criterion, lr_scheduler, lr_img, gt_img)
            train_losses.append(loss)
        avg_train_loss = sum(train_losses) / len(train_losses)

        val_losses, val_psnrs = [], []
        for lr_img, gt_img in val_loader:
            loss, pred = engine.eval_step(criterion, lr_img, gt_img)
            val_losses.append(loss)
            val_psnrs.append(compute_psnr(pred, gt_img.to(engine.device)))
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_psnr = sum(val_psnrs) / len(val_psnrs)

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{config['num_epochs']} "
              f"[{dt:.1f}s] train_loss={avg_train_loss:.4f} "
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
                        "val_psnr": avg_val_psnr}, ckpt_path)
            print(f"  -> new best val_loss, saved to {ckpt_path}")
        else:
            epochs_since_improvement += 1
            print(f"  -> val_loss did not improve "
                  f"({epochs_since_improvement}/{config['patience']})")
            if epochs_since_improvement >= config["patience"]:
                print(f"STOPPING: val_loss has not improved for "
                      f"{config['patience']} epochs while training "
                      f"continued -- this is the overfitting signal. "
                      f"Best checkpoint is saved at "
                      f"{config['checkpoint_dir']}/best_model.pt")
                break

    print("Training finished.")


if __name__ == "__main__":
    main()
