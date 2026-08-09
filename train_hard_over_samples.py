"""
Hard-example oversampling fine-tune. Directly targets your stated goal
(steady PSNR/SSIM, lower std) by giving the model more exposure to the
samples it currently struggles with (002982, 000352, and similar), rather
than assuming they're a data bug to fix.

How it works:
  1. Loads your baseline checkpoint, runs it over the full TRAIN split
     (not val -- we need per-sample loss on data the model has already
     fit reasonably well elsewhere, to find genuine outliers) to find
     per-sample loss.
  2. Builds a WeightedRandomSampler that oversamples the worst decile by
     a configurable factor (default 4x) -- these samples appear more
     often per epoch, giving the optimizer more gradient signal on them.
  3. Fine-tunes from the baseline checkpoint with this sampler, all other
     hyperparameters pinned to baseline's proven values (no compounding
     changes, per the ablation methodology used throughout).

Usage:
    python train_hard_oversample.py /path/to/data_root ./checkpoints_baseline/best_model.pt
"""

import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from sem_dataset import SEMPairDataset, load_calib_stats, build_dataloaders
from model import SemiRestoreNet_V2, KLAMetrologyLoss, build_optimizer
from train_engine import HardwareEngine


def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(max_val ** 2 / mse)).item()


def find_hard_samples(model, full_train, train_indices, device, oversample_factor=4.0,
                       hard_fraction=0.1):
    """Runs the current model over the TRAIN split, ranks samples by loss,
    and returns per-sample weights for WeightedRandomSampler -- the worst
    `hard_fraction` get `oversample_factor`x weight, everyone else gets 1x."""
    model.eval()
    losses = []
    print(f"Scoring {len(train_indices)} training samples to find hard cases...")
    with torch.no_grad():
        for idx in train_indices:
            lr_img, gt_img = full_train[idx]
            lr_img = lr_img.unsqueeze(0).to(device)
            gt_img = gt_img.unsqueeze(0).to(device)
            pred = model(lr_img)
            mse = torch.mean((pred - gt_img) ** 2).item()
            losses.append(mse)

    losses = np.array(losses)
    n_hard = max(1, int(len(losses) * hard_fraction))
    hard_threshold = np.sort(losses)[-n_hard]
    is_hard = losses >= hard_threshold

    weights = np.ones(len(losses), dtype=np.float32)
    weights[is_hard] = oversample_factor

    hard_indices_local = np.where(is_hard)[0]
    hard_fnames = [os.path.basename(full_train.pairs[train_indices[i]][0])
                   for i in hard_indices_local]
    print(f"Identified {n_hard} hard samples (top {hard_fraction:.0%} by MSE), "
          f"weighted {oversample_factor}x. Hard-sample MSE range: "
          f"[{losses[is_hard].min():.4f}, {losses[is_hard].max():.4f}] "
          f"vs. easy-sample range [{losses[~is_hard].min():.4f}, {losses[~is_hard].max():.4f}]")
    print(f"Hard samples include (first 10): {hard_fnames[:10]}")

    return weights


def main():
    if len(sys.argv) < 3:
        print("Usage: python train_hard_oversample.py /path/to/data_root /path/to/checkpoint.pt")
        return
    data_root = sys.argv[1]
    resume_ckpt_path = sys.argv[2]

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found.")
        return

    ckpt = torch.load(resume_ckpt_path, map_location="cuda")
    base_cfg = ckpt["config"]
    print(f"Resuming from {resume_ckpt_path} (val_psnr={ckpt['val_psnr']:.2f}dB)")

    # CHANGE vs. baseline: ONLY the sampler changes. Every other
    # hyperparameter is pinned to baseline's proven values -- no
    # compounding changes, per the ablation discipline used throughout.
    config = dict(base_cfg)
    config.update(dict(
        oversample_factor=4.0,
        hard_fraction=0.10,
        num_epochs=20,
        checkpoint_dir="./checkpoints_hard_oversample",
    ))
    print("=== HARD-EXAMPLE OVERSAMPLING FINE-TUNE ===")
    print("Config:", config)

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    p_low, p_high = load_calib_stats(data_root)

    # Build the same train/val split as every other script (same seed),
    # so "train_indices" here matches exactly what the model was
    # originally trained on.
    full_train = SEMPairDataset(
        gt_dir=os.path.join(data_root, "train", "gt"),
        lr_dir=os.path.join(data_root, "train", "NoisyLR"),
        p_low=p_low, p_high=p_high,
        scale_factor=config["scale_factor"], augment=True,
    )
    n_val = max(1, int(len(full_train) * 0.1))
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [n_train, n_val], generator=generator
    )

    model = SemiRestoreNet_V2(dim=config["dim"], num_blocks=config["num_blocks"],
                               scale_factor=config["scale_factor"])
    model.load_state_dict(ckpt["model_state"])
    model.to("cuda")
    print("Loaded baseline weights as starting point.")

    # ---- Score hard samples using the CURRENT (baseline) model ----
    sample_weights = find_hard_samples(
        model, full_train, train_subset.indices, "cuda",
        oversample_factor=config["oversample_factor"],
        hard_fraction=config["hard_fraction"],
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Custom DataLoader using the weighted sampler (build_dataloaders'
    # default random shuffle doesn't support per-sample weighting, so we
    # construct this one directly rather than going through it).
    train_loader = DataLoader(
        train_subset, batch_size=config["batch_size"], sampler=sampler,
        num_workers=config.get("num_workers", 4), pin_memory=True, drop_last=True,
    )
    # Val loader: reuse the normal (unweighted, unaugmented) path via
    # build_dataloaders for a fair, standard validation measurement.
    _, val_loader, _ = build_dataloaders(
        data_root, scale_factor=config["scale_factor"],
        batch_size=config["batch_size"], num_workers=config.get("num_workers", 4),
    )

    criterion = KLAMetrologyLoss(edge_weight=config["edge_weight"],
                                  freq_weight=config.get("freq_weight", 0.0),
                                  ssim_weight=config.get("ssim_weight", 0.0))

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
        val_psnr_std = float(np.std(val_psnrs))

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        dt = time.time() - t0
        lr_msg = (f" [LR: {old_lr:.2e}]" if old_lr == new_lr
                  else f" [LR REDUCED: {old_lr:.2e} -> {new_lr:.2e}]")
        print(f"[HARD-OS] Epoch {epoch+1}/{config['num_epochs']} "
              f"[{dt:.1f}s]{lr_msg} train_loss={avg_train_loss:.4f} "
              f"val_loss={avg_val_loss:.4f} val_psnr={avg_val_psnr:.2f}dB "
              f"val_psnr_std={val_psnr_std:.2f}dB  <- watch THIS drop, "
              f"that's the goal, not just mean PSNR")

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
                  f"({epochs_since_improvement}/{config.get('early_stop_patience', 7)})")
            if epochs_since_improvement >= config.get("early_stop_patience", 7):
                if new_lr < config["lr"]:
                    print("STOPPING: fine-tune has plateaued at lower LR.")
                    break
                else:
                    print("WAITING: val_loss stalled but LR hasn't dropped yet.")

    engine.flush(optimizer)
    print("Hard-oversample fine-tune finished. Compare val_psnr_std against "
          "baseline's 4.55dB using evaluate.py -- if this run's std is "
          "meaningfully lower, oversampling helped; if the SAME files "
          "(002982, 000352) are still the worst, the model may simply lack "
          "the capacity/features to represent this failure mode, which is "
          "itself useful, honest information for your presentation.")


if __name__ == "__main__":
    main()