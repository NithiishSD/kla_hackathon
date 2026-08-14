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
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sem_dataset import build_dataloaders, load_calib_stats
from model import SemiRestoreNet_V2, KLAMetrologyLoss, build_optimizer




class GradAccumState:
    """Tracks micro-batch position for gradient accumulation across the
    whole training run (not reset per epoch)."""

    def __init__(self, accum_steps):
        self.accum_steps = accum_steps
        self.micro_step = 0
        self.pending = False  # accumulated grads waiting on a step

    def should_zero_grad(self):
        return self.micro_step % self.accum_steps == 0

    def after_backward(self):
        self.micro_step += 1
        self.pending = True
        do_step = (self.micro_step % self.accum_steps == 0)
        if do_step:
            self.pending = False
        return do_step

    def flush_needed(self):
        return self.pending


class HardwareEngine:
    def __init__(self, model, batch_size, target_batch_size=32, device="cuda"):
        """
        batch_size: the ACTUAL batch_size your DataLoader yields (i.e. what
            you passed to build_dataloaders / DataLoader(..., batch_size=X)).
            This is the single source of truth for accumulation math --
            there is no separate "local_batch_size" concept with
            DataParallel, since one train_step already gets the full
            DataLoader batch regardless of GPU count.
        target_batch_size: the effective batch size you want to simulate
            via accumulation.
        """
        self.device = torch.device(device)
        self.gpu_name = torch.cuda.get_device_name(0)
        self.major, self.minor = torch.cuda.get_device_capability(0)
        self.num_gpus = torch.cuda.device_count()

        if self.major >= 8:
            self.precision = torch.bfloat16
            self.use_scaler = False
            print(f"--- [Modern GPU] Using bfloat16 on {self.gpu_name} ---")
        else:
            self.precision = torch.float16
            self.use_scaler = True
            print(f"--- [Legacy GPU] Using float16 + Scaler on {self.gpu_name} ---")

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_scaler)

        # CHANGE: accum_steps derived from the DataLoader's real batch_size,
        # no num_gpus multiplier (see module docstring for why).
        self.accum_steps = max(1, target_batch_size // batch_size)
        actual_effective_batch = self.accum_steps * batch_size
        print(f"--- [Compute] DataLoader batch_size={batch_size}, "
              f"target_batch_size={target_batch_size} "
              f"-> accum_steps={self.accum_steps} "
              f"(actual effective batch = {actual_effective_batch}) ---")
        if actual_effective_batch != target_batch_size:
            print(f"--- [Compute] NOTE: {target_batch_size} is not a multiple "
                  f"of batch_size={batch_size}, so actual effective batch is "
                  f"{actual_effective_batch}, not exactly {target_batch_size}. ---")

        self._accum_state = GradAccumState(self.accum_steps)

        self.model = model.to(self.device)
        self._compiled = False

        if self.num_gpus > 1:
            print(f"--- [Multi-GPU] {self.num_gpus} GPUs found. Wrapping in "
                  f"DataParallel. One train_step still consumes the full "
                  f"DataLoader batch_size={batch_size} total (split across "
                  f"GPUs internally) -- accum_steps above already accounts "
                  f"for this correctly. GPU-0 carries a heavier memory load "
                  f"than the rest; don't assume linear scaling. ---")
            self.model = nn.DataParallel(self.model)

    def compile_model(self):
        if self._compiled:
            return self.model
        try:
            print("--- [Optimizer] Attempting torch.compile... ---")
            if isinstance(self.model, nn.DataParallel):
                inner = torch.compile(self.model.module)
                self.model = nn.DataParallel(inner)
            else:
                self.model = torch.compile(self.model)
            self._compiled = True
            print("--- [Optimizer] torch.compile requested successfully "
                  "(verify with a real batch before trusting it). ---")
        except Exception as e:
            print(f"--- [Optimizer] Compile failed: {e}. Using eager mode. ---")
        return self.model

    def train_step(self, optimizer, criterion, lr_scheduler, input_img, gt_img):
        """No `i` argument needed anymore -- accumulation position is
        tracked internally and persists across epochs."""
        self.model.train()

        if self._accum_state.should_zero_grad():
            optimizer.zero_grad(set_to_none=True)

        input_img = input_img.to(self.device, non_blocking=True)
        gt_img = gt_img.to(self.device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=self.precision):
            output = self.model(input_img)
            loss = criterion(output, gt_img) / self.accum_steps

        if self.use_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        do_step = self._accum_state.after_backward()

        if do_step:
            if self.use_scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            if lr_scheduler is not None and not isinstance(
                    lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step()

        return loss.item() * self.accum_steps

    def flush(self, optimizer):
        """Call once, after the ENTIRE training loop ends (or right before
        an early-stop break) -- steps on any leftover accumulated gradient
        from a partial final group so it isn't silently discarded."""
        if not self._accum_state.flush_needed():
            return False
        if self.use_scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        self._accum_state.pending = False
        print("--- [Compute] Flushed leftover accumulated gradient at "
              "end of training. ---")
        return True

    @torch.no_grad()
    def eval_step(self, criterion, input_img, gt_img):
        self.model.eval()
        input_img = input_img.to(self.device, non_blocking=True)
        gt_img = gt_img.to(self.device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=self.precision):
            output = self.model(input_img)
            loss = criterion(output, gt_img)

        return loss.item(), output
class IntensityProfileLoss(nn.Module):
    def __init__(self, edge_percentile=85.0):
        super().__init__()
        self.edge_percentile = edge_percentile
        self.register_buffer('kernel_x', torch.tensor([[-1., 0., 1.]]).view(1, 1, 1, 3) / 2.0)
        self.register_buffer('kernel_y', torch.tensor([[-1.], [0.], [1.]]).view(1, 1, 3, 1) / 2.0)

    def _gradient_magnitude(self, img):
        gx = F.conv2d(img, self.kernel_x, padding=(0, 1))
        gy = F.conv2d(img, self.kernel_y, padding=(1, 0))
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred, target):
        pred_f, target_f = pred.float(), target.float()
        target_grad = self._gradient_magnitude(target_f)
        pred_grad = self._gradient_magnitude(pred_f)

        # Fixed GT-only edge mask, detached -- the model cannot reduce
        # this loss by changing which pixels get penalized, only by
        # matching the actual slope magnitude at GT's true edges.
        with torch.no_grad():
            b = target_grad.shape[0]
            mask = torch.zeros_like(target_grad)
            for i in range(b):
                flat = target_grad[i].reshape(-1)
                thresh = torch.quantile(flat, self.edge_percentile / 100.0)
                mask[i] = (target_grad[i] > thresh).float()

        diff = (pred_grad - target_grad).abs() * mask
        return diff.sum() / (mask.sum() + 1e-6)
def ensure_calib_stats(data_root, p_low_pct=1.0, p_high_pct=99.0):
    """Computes calib_stats.json automatically if it doesn't already exist,
    so train.py is fully self-contained -- no separate calibrate_stats.py
    run required before this script.

    Scans every .npy file under train/gt and train/NoisyLR, pools all pixel
    values, and computes the p_low_pct/p_high_pct percentiles across that
    pool. This mirrors the normalization scheme in sem_dataset.py's
    normalize() (p_low -> 0, p_high -> 1, clamp outside that range) and the
    1.0/99.0 percentile convention already recorded in this project's
    existing calib_stats.json.

    If you have the original calibrate_stats.py and it computes stats
    differently (e.g. GT-only, or a different percentile method), swap this
    function out for that logic instead -- what matters is that whatever
    computed the calib_stats.json baked into your existing checkpoints is
    exactly what's used here, so new runs stay comparable to your logged
    results.
    """
    calib_path = os.path.join(data_root, "calib_stats.json")
    if os.path.exists(calib_path):
        print(f"Found existing {calib_path}, reusing it.")
        return

    print(f"No calib_stats.json found at {calib_path} -- computing it now "
          f"from train/gt and train/NoisyLR ({p_low_pct}th/{p_high_pct}th percentile)...")

    gt_files = glob.glob(os.path.join(data_root, "train", "gt", "*.npy"))
    lr_files = glob.glob(os.path.join(data_root, "train", "NoisyLR", "*.npy"))
    all_files = gt_files + lr_files
    if not all_files:
        raise FileNotFoundError(
            f"No .npy files found under {data_root}/train/gt or "
            f"{data_root}/train/NoisyLR -- cannot compute calibration stats."
        )

    # Pool pixel values across all files. For large datasets this can use
    # significant RAM; if that becomes a problem, switch to an online/
    # streaming percentile estimate (e.g. a running histogram) instead.
    all_pixels = []
    for f in all_files:
        arr = np.load(f).astype(np.float32).reshape(-1)
        all_pixels.append(arr)
    pooled = np.concatenate(all_pixels)

    p_low = float(np.percentile(pooled, p_low_pct))
    p_high = float(np.percentile(pooled, p_high_pct))

    stats = {"p_low": p_low, "p_high": p_high,
              "p_low_pct": p_low_pct, "p_high_pct": p_high_pct}
    with open(calib_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Computed and saved calib_stats.json: p_low={p_low:.6f}, p_high={p_high:.6f}")


def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(max_val ** 2 / mse)).item()


def run_training_loop(tag, config, model, criterion, train_loader, val_loader,
                       p_low, p_high, extra_ckpt_fields=None, save_path=None):
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
            if save_path:
                torch.save(save_dict, save_path)
                print(f"  -> new best val_loss, saved to {save_path}")
            else:
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
        checkpoint_dir=f"./weights",  # final submitted model
        finetune_from=base_ckpt_path,
    )
    print("\n" + "=" * 60)
    print("STAGE 2: PROFILE-LOSS FINE-TUNE (final submitted model)")
    print("=" * 60)
    print("Config:", config)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    final_model_path = os.path.join(config["checkpoint_dir"], "final_model.pt")
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
                              train_loader, val_loader, p_low, p_high, save_path=final_model_path)


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

    ensure_calib_stats(args.data_root)
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