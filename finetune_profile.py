"""
Phase 3b fine-tune: resumes from checkpoints_baseline_v2/best_model.pt
(documented as the best starting point in deepseek_validation.txt) and
adds IntensityProfileLoss (losses/intensity_profile_loss.py) at a
conservative weight -- a GT-edge-masked gradient-magnitude loss that
targets the diagnosed root cause (edge slope shape mismatch) rather than
edge position, which is what the six prior loss-based attempts
(ThresholdedCDLoss, MS-SSIM) already tried and failed at.

Deliberately NOT the previous version of this file -- that one declared
use_profile_loss/num_profiles/profile_width config keys that were never
actually read anywhere; the loss it built was just SSIM. This one
actually wires the new loss in.

Usage:
    python finetune_profile.py /path/to/data_root
"""

import os
import sys
import time

import torch

from sem_dataset import build_dataloaders, load_calib_stats
from model import SemiRestoreNet_V2, KLAMetrologyLoss, build_optimizer
from train_engine import HardwareEngine
from losses.intensity_profile_loss import IntensityProfileLoss


def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(max_val ** 2 / mse)).item()


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data"
    base_ckpt_path = "checkpoints_baseline_v2/best_model.pt"

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found.")
        return
    device = "cuda"

    ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    base_cfg = ckpt["config"]
    print(f"Resuming from {base_ckpt_path} "
          f"(epoch {ckpt['epoch']+1}, val_psnr={ckpt['val_psnr']:.2f}dB)")

    config = dict(
        dim=base_cfg["dim"], num_blocks=base_cfg["num_blocks"], scale_factor=base_cfg["scale_factor"],
        batch_size=4, target_batch_size=32,
        edge_weight=0.5, freq_weight=0.0, ssim_weight=0.0,
        profile_weight=1.0,   # NEW -- conservative starting weight, not the "nuclear option"
        profile_edge_percentile=85.0,
        lr=5e-5, weight_decay=1e-4,
        scheduler_patience=2, early_stop_patience=5,
        num_workers=min(8, os.cpu_count() or 1),
        num_epochs=15,
        checkpoint_dir="./checkpoints_profile_v1",
        finetune_from=base_ckpt_path,
    )
    print("=== PROFILE-LOSS FINE-TUNE ===")
    print("Config:", config)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    p_low, p_high = load_calib_stats(data_root)
    train_loader, val_loader, test_loader = build_dataloaders(
        data_root, scale_factor=config["scale_factor"],
        batch_size=config["batch_size"], num_workers=config["num_workers"])

    model = SemiRestoreNet_V2(dim=config["dim"], num_blocks=config["num_blocks"],
                               scale_factor=config["scale_factor"])
    model.load_state_dict(ckpt["model_state"], strict=False)
    print("Loaded baseline_v2 weights.")

    base_criterion = KLAMetrologyLoss(edge_weight=config["edge_weight"],
                                       freq_weight=config["freq_weight"],
                                       ssim_weight=config["ssim_weight"])
    profile_criterion = IntensityProfileLoss(edge_percentile=config["profile_edge_percentile"])

    engine = HardwareEngine(model, batch_size=config["batch_size"],
                             target_batch_size=config["target_batch_size"])
    base_criterion = base_criterion.to(engine.device)
    profile_criterion = profile_criterion.to(engine.device)

    def criterion(pred, target):
        return base_criterion(pred, target) + config["profile_weight"] * profile_criterion(pred, target)

    optimizer = build_optimizer(engine.model, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config["scheduler_patience"])

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
        print(f"[PROFILE-FT] Epoch {epoch+1}/{config['num_epochs']} "
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
                        "val_psnr": avg_val_psnr, "lr": new_lr,
                        "p_low": p_low, "p_high": p_high}, ckpt_path)
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
    print("Profile-loss fine-tune finished. Compare against baseline_v2 with "
          "evaluate.py -- specifically check Mean CD Bias, not just PSNR/SSIM.")


if __name__ == "__main__":
    main()
