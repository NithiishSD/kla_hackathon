"""
KLA PS01 Submission -- Unified Training Script
================================================
Reproduces the full training process that produced the submitted model
(weights/final_model.pt) in two automated stages:

  Stage 1 (baseline): trains SemiRestoreNet_V2 from scratch with the base
      KLAMetrologyLoss (Charbonnier + Sobel edge loss). dim=64, num_blocks=2,
      50 epochs. Saves to ./checkpoints_baseline_v2/best_model.pt.

  Stage 2 (finetune): resumes from the Stage 1 checkpoint and fine-tunes
      with IntensityProfileLoss added on top (a GT-edge-masked,
      gradient-magnitude-matching loss -- see losses/intensity_profile_loss.py
      for why this targets edge-slope shape rather than edge position).
      profile_weight=3.0, lr=1e-5 (5x lower than Stage 1), 15 epochs.
      Saves to ./checkpoints_profile_w_lr3p0/best_model.pt.

This two-stage recipe -- not a single end-to-end run -- is what actually
produced the submitted weights: Stage 1's config was validated first, then
Stage 2's loss/LR combination was chosen via ablation (see report/log for
the full sweep across profile_weight in {0, 1.0, 2.0, 3.0} and the LR
sweep at profile_weight=3.0). Both are baked in here as the documented
best configuration, not re-derived at runtime.

Usage:
    python train.py /path/to/data_root                  # runs both stages
    python train.py /path/to/data_root --stage baseline  # Stage 1 only
    python train.py /path/to/data_root --stage finetune  # Stage 2 only
                                                           # (requires Stage 1
                                                           # checkpoint to
                                                           # already exist)

Note on reproducibility: results will be close to, but not bit-identical
to, the submitted checkpoint's reported metrics (GPU kernel / dataloader
non-determinism). See README.md for details.
"""

import argparse
import os
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


def run_training_loop(tag, config, model, criterion, train_loader, val_loader,
                       p_low, p_high, extra_ckpt_fields=None):
    """Shared training loop used by both stages -- identical logic to the
    original train_baseline.py / finetune_profile.py scripts, just
    factored out so the two stages don't duplicate ~80 lines each."""
    engine = HardwareEngine(model, batch_size=config["batch_size"],
                             target_batch_size=config["target_batch_size"])
    criterion = criterion.to(engine.device) if hasattr(criterion, "to") else criterion

    optimizer = build_optimizer(engine.model, lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config["scheduler_patience"])

    best_val_loss = float('inf')
    epochs_since_improvement = 0
    ckpt_path = os.path.join(config["checkpoint_dir"], "best_model.pt")

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
        print(f"[{tag}] Epoch {epoch+1}/{config['num_epochs']} "
              f"[{dt:.1f}s]{lr_msg} train_loss={avg_train_loss:.4f} "
              f"val_loss={avg_val_loss:.4f} val_psnr={avg_val_psnr:.2f}dB")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            state_dict = (engine.model.module.state_dict()
                          if isinstance(engine.model, torch.nn.DataParallel)
                          else engine.model.state_dict())
            save_dict = {"model_state": state_dict, "config": config,
                         "epoch": epoch, "val_loss": avg_val_loss,
                         "val_psnr": avg_val_psnr, "lr": new_lr,
                         "p_low": p_low, "p_high": p_high}
            if extra_ckpt_fields:
                save_dict.update(extra_ckpt_fields)
            torch.save(save_dict, ckpt_path)
            print(f"  -> new best val_loss, saved to {ckpt_path}")
        else:
            epochs_since_improvement += 1
            print(f"  -> val_loss did not improve "
                  f"({epochs_since_improvement}/{config['early_stop_patience']})")
            if epochs_since_improvement >= config["early_stop_patience"]:
                if new_lr < config["lr"]:
                    print(f"STOPPING [{tag}]: model has fine-tuned at a lower LR and plateaued.")
                    break
                else:
                    print(f"WAITING [{tag}]: val_loss stalled but LR hasn't dropped yet.")

    engine.flush(optimizer)
    print(f"[{tag}] finished. Best checkpoint: {ckpt_path}")
    return ckpt_path


def train_baseline(data_root, p_low, p_high, train_loader, val_loader):
    """Stage 1: baseline_v2. dim=64, num_blocks=2, 50 epochs, base loss only."""
    config = dict(
        dim=64,
        num_blocks=2,
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
        checkpoint_dir="./checkpoints_baseline_v2",
    )
    print("\n" + "=" * 60)
    print("STAGE 1: BASELINE TRAINING (baseline_v2)")
    print("=" * 60)
    print("Config:", config)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    model = SemiRestoreNet_V2(dim=config["dim"], num_blocks=config["num_blocks"],
                               scale_factor=config["scale_factor"])
    criterion = KLAMetrologyLoss(edge_weight=config["edge_weight"],
                                  freq_weight=config["freq_weight"],
                                  ssim_weight=config["ssim_weight"])

    return run_training_loop("BASELINE", config, model, criterion,
                              train_loader, val_loader, p_low, p_high)


def train_finetune(data_root, p_low, p_high, train_loader, val_loader,
                    base_ckpt_path, profile_weight=3.0, lr=1e-5):
    """Stage 2: profile-loss fine-tune from baseline_v2.
    profile_weight=3.0, lr=1e-5 is the documented best configuration from
    the ablation sweep (see report/log) -- these are the defaults, not
    arbitrary starting points."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    base_cfg = ckpt["config"]
    print(f"\nResuming from {base_ckpt_path} "
          f"(epoch {ckpt['epoch']+1}, val_psnr={ckpt['val_psnr']:.2f}dB)")

    weight_tag = str(profile_weight).replace(".", "p")
    config = dict(
        dim=base_cfg["dim"], num_blocks=base_cfg["num_blocks"], scale_factor=base_cfg["scale_factor"],
        batch_size=4, target_batch_size=32,
        edge_weight=0.5, freq_weight=0.0, ssim_weight=0.0,
        profile_weight=profile_weight,
        profile_edge_percentile=85.0,
        lr=lr, weight_decay=1e-4,
        scheduler_patience=2, early_stop_patience=5,
        num_workers=min(8, os.cpu_count() or 1),
        num_epochs=15,
        checkpoint_dir=f"./checkpoints_profile_w_lr{weight_tag}",
        finetune_from=base_ckpt_path,
    )
    print("\n" + "=" * 60)
    print("STAGE 2: PROFILE-LOSS FINE-TUNE (final submitted model)")
    print("=" * 60)
    print("Config:", config)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    model = SemiRestoreNet_V2(dim=config["dim"], num_blocks=config["num_blocks"],
                               scale_factor=config["scale_factor"])
    model.load_state_dict(ckpt["model_state"], strict=False)
    print("Loaded baseline_v2 weights.")

    base_criterion = KLAMetrologyLoss(edge_weight=config["edge_weight"],
                                       freq_weight=config["freq_weight"],
                                       ssim_weight=config["ssim_weight"])
    profile_criterion = IntensityProfileLoss(edge_percentile=config["profile_edge_percentile"])

    device_for_loss = "cuda" if torch.cuda.is_available() else "cpu"
    base_criterion = base_criterion.to(device_for_loss)
    profile_criterion = profile_criterion.to(device_for_loss)

    def criterion(pred, target):
        return base_criterion(pred, target) + config["profile_weight"] * profile_criterion(pred, target)

    return run_training_loop("PROFILE-FT", config, model, criterion,
                              train_loader, val_loader, p_low, p_high)


def main():
    parser = argparse.ArgumentParser(description="KLA PS01 -- two-stage training pipeline")
    parser.add_argument("data_root", type=str, help="Path to data root (must contain "
                         "train/gt, train/NoisyLR, calib_stats.json)")
    parser.add_argument("--stage", choices=["baseline", "finetune", "all"], default="all",
                         help="Which stage(s) to run. 'all' (default) runs both automatically.")
    parser.add_argument("--profile-weight", type=float, default=3.0,
                         help="Profile loss weight for Stage 2 (default: 3.0, the documented best)")
    parser.add_argument("--finetune-lr", type=float, default=1e-5,
                         help="Learning rate for Stage 2 (default: 1e-5, the documented best)")
    parser.add_argument("--base-ckpt", type=str, default="./checkpoints_baseline_v2/best_model.pt",
                         help="Path to Stage 1 checkpoint (only used if --stage finetune)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found. Training will be very slow on CPU.")

    p_low, p_high = load_calib_stats(args.data_root)
    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_root, scale_factor=2, batch_size=4,
        num_workers=min(8, os.cpu_count() or 1))

    baseline_ckpt = args.base_ckpt

    if args.stage in ("baseline", "all"):
        baseline_ckpt = train_baseline(args.data_root, p_low, p_high, train_loader, val_loader)

    if args.stage in ("finetune", "all"):
        final_ckpt = train_finetune(args.data_root, p_low, p_high, train_loader, val_loader,
                                     base_ckpt_path=baseline_ckpt,
                                     profile_weight=args.profile_weight,
                                     lr=args.finetune_lr)
        print(f"\nPipeline complete. Final model: {final_ckpt}")
        print(f"Copy this file to weights/final_model.pt before running inference.py.")


if __name__ == "__main__":
    main()