"""
Training script for AirNet -- see airnet_model.py for why this
architecture (OOD-generalization via a self-supervised degradation
representation, not speed or raw quality like the SwinIR/Restormer
comparison).

Does NOT reuse HardwareEngine (train_engine.py): AirNet's joint
restoration + contrastive loss with a per-step momentum-encoder update
doesn't fit HardwareEngine's single-criterion(output, gt) train_step, so
this reimplements the same bf16-autocast / cross-epoch gradient
accumulation conventions directly (accumulation position is a global
step counter, not reset per epoch, matching why HardwareEngine tracks it
that way -- a partial group at an epoch boundary must carry into the
next epoch instead of being silently dropped).

Usage:
    python calibrate_stats.py /path/to/data_root
    python train_airnet.py /path/to/data_root
"""

import os
import sys
import time

import torch

from sem_dataset import build_dataloaders
from model import KLAMetrologyLoss, build_optimizer
from airnet_model import AirNet


def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(max_val ** 2 / mse)).item()


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data"

    config = dict(
        model_type="airnet",
        dim=64, num_blocks=6, feat_dim=256, proj_dim=128,
        queue_size=1024, momentum=0.999, temperature=0.07,
        contrastive_weight=0.1,
        batch_size=4, target_batch_size=32, scale_factor=2,
        edge_weight=0.1, freq_weight=0.0,
        lr=2e-4, weight_decay=1e-4,
        scheduler_patience=2, early_stop_patience=6,
        num_workers=min(8, os.cpu_count() or 1),
        num_epochs=50,
        checkpoint_dir="./checkpoints_airnet",
    )
    print("Config:", config)

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found. This will run on CPU (slow, "
              "for debugging only).")
        return
    device = "cuda"
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root, scale_factor=config["scale_factor"],
        batch_size=config["batch_size"], num_workers=config["num_workers"])

    model = AirNet(
        dim=config["dim"], num_blocks=config["num_blocks"],
        feat_dim=config["feat_dim"], proj_dim=config["proj_dim"],
        queue_size=config["queue_size"], momentum=config["momentum"],
        temperature=config["temperature"], scale_factor=config["scale_factor"],
    ).to(device)
    criterion = KLAMetrologyLoss(edge_weight=config["edge_weight"],
                                  freq_weight=config["freq_weight"]).to(device)
    optimizer = build_optimizer(model, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config["scheduler_patience"])

    major, _ = torch.cuda.get_device_capability(0)
    use_scaler = major < 8
    precision = torch.float16 if use_scaler else torch.bfloat16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    print(f"--- Using {'float16 + scaler' if use_scaler else 'bfloat16'} on {torch.cuda.get_device_name(0)} ---")

    accum_steps = max(1, config["target_batch_size"] // config["batch_size"])
    print(f"--- accum_steps={accum_steps} (effective batch = {accum_steps * config['batch_size']}) ---")

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    def train_step(lr_img, gt_img):
        nonlocal global_step
        lr_img = lr_img.to(device, non_blocking=True)
        gt_img = gt_img.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=precision):
            pred = model(lr_img)
            restoration_loss = criterion(pred, gt_img)
            contrastive_loss = model.contrastive_loss(lr_img)
            loss = (restoration_loss + config["contrastive_weight"] * contrastive_loss) / accum_steps

        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        global_step += 1
        if global_step % accum_steps == 0:
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        return restoration_loss.item(), contrastive_loss.item()

    def flush():
        if global_step % accum_steps != 0:
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            print("--- Flushed leftover accumulated gradient at end of training. ---")

    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(config["num_epochs"]):
        t0 = time.time()
        model.train()
        restoration_losses, contrastive_losses = [], []
        for lr_img, gt_img in train_loader:
            r_loss, c_loss = train_step(lr_img, gt_img)
            restoration_losses.append(r_loss)
            contrastive_losses.append(c_loss)
        avg_restoration_loss = sum(restoration_losses) / len(restoration_losses)
        avg_contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)

        model.eval()
        val_losses, val_psnrs = [], []
        with torch.no_grad():
            for lr_img, gt_img in val_loader:
                lr_img = lr_img.to(device, non_blocking=True)
                gt_img = gt_img.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=precision):
                    pred = model(lr_img)
                    loss = criterion(pred, gt_img)  # restoration-only for val; contrastive is a training aux
                val_losses.append(loss.item())
                val_psnrs.append(compute_psnr(pred, gt_img))
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_psnr = sum(val_psnrs) / len(val_psnrs)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{config['num_epochs']} [{dt:.1f}s] "
              f"restoration_loss={avg_restoration_loss:.4f} contrastive_loss={avg_contrastive_loss:.4f} "
              f"val_loss={avg_val_loss:.4f} val_psnr={avg_val_psnr:.2f}dB lr={new_lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            ckpt_path = os.path.join(config["checkpoint_dir"], "best_model.pt")
            torch.save({"model_state": model.state_dict(), "config": config,
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

    flush()
    print("AirNet training finished.")


if __name__ == "__main__":
    main()
