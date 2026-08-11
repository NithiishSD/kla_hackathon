"""
Phase 4: MS-SSIM Fine-Tune
Resumes from Phase 3 checkpoint, adds multi-scale SSIM for
structural recovery at all spatial scales.

Usage:
    python finetune_msssim.py /path/to/data_root /path/to/checkpoint.pt
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
        print("Usage: python finetune_msssim.py /path/to/data_root /path/to/checkpoint.pt")
        return
    data_root = sys.argv[1]
    resume_ckpt_path = sys.argv[2]

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found.")
        return

    ckpt = torch.load(resume_ckpt_path, map_location="cuda")
    print(f"Resuming from {resume_ckpt_path} "
          f"(epoch {ckpt['epoch']+1}, val_psnr={ckpt['val_psnr']:.2f}dB)")

    config = dict(
    dim=64,
    num_blocks=2,
    batch_size=4,
    target_batch_size=32,
    scale_factor=2,
    
    edge_weight=0.5,          # Unchanged
    freq_weight=0.0,          # Unchanged
    ssim_weight=0.15,         # Only change — conservative MS-SSIM
    charbonnier_weight=1.0,   # Unchanged
    
    lr=2e-04,                 # Unchanged
    weight_decay=1e-04,       # Unchanged
    scheduler_patience=5,     # Unchanged
    early_stop_patience=10,   # Unchanged
    num_workers=8,
    num_epochs=20,
    use_compile=False,
    checkpoint_dir="./checkpoints_msssim_v2",
    
    use_multiscale_ssim=True,
    ms_ssim_scales=[1.0, 0.5, 0.25],
    ms_ssim_weights=[0.5, 0.3, 0.2],
)

    print("=== PHASE 4: MS-SSIM STRUCTURAL RECOVERY ===")
    print("Config:", config)

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root, scale_factor=config["scale_factor"],
        batch_size=config["batch_size"], num_workers=config.get("num_workers", 4),
    )

    model = SemiRestoreNet_V2(dim=config["dim"], num_blocks=config["num_blocks"],
                               scale_factor=config["scale_factor"])
    model.load_state_dict(ckpt["model_state"], strict=False)
    print("Loaded checkpoint weights. Missing keys initialized to default.")

    criterion = KLAMetrologyLoss(
        edge_weight=config["edge_weight"],
        freq_weight=config.get("freq_weight", 0.0),
        ssim_weight=config["ssim_weight"],
        use_multiscale_ssim=config.get("use_multiscale_ssim", False),
        ms_ssim_scales=config.get("ms_ssim_scales", [1.0, 0.5, 0.25]),
        ms_ssim_weights=config.get("ms_ssim_weights", [0.5, 0.3, 0.2]),
        charbonnier_weight=config.get("charbonnier_weight", 1.0),
    )

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

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        dt = time.time() - t0
        lr_msg = (f" [LR: {old_lr:.2e}]" if old_lr == new_lr
                  else f" [LR REDUCED: {old_lr:.2e} -> {new_lr:.2e}]")
        print(f"[MS-SSIM] Epoch {epoch+1}/{config['num_epochs']} "
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
            if epochs_since_improvement >= config['early_stop_patience']:
                if new_lr < config["lr"]:
                    print("STOPPING: plateaued at lower LR.")
                    break
                else:
                    print("WAITING: val_loss stalled but LR hasn't dropped yet.")

    engine.flush(optimizer)
    print("MS-SSIM fine-tune finished. Evaluate with evaluate.py")


if __name__ == "__main__":
    main()