"""
Hardware-aware training engine: auto-selects bf16 (no scaler) vs fp16
(with GradScaler) based on GPU compute capability, and optionally wraps
in DataParallel for multi-GPU.

Works on RTX 4050, Kaggle Dual T4, or a cloud H100 without edits.

Known tradeoffs (see CHANGES_round4.md for the full writeup):
- DataParallel is used here for hackathon-timeline simplicity. It is not
  the fastest multi-GPU option (PyTorch recommends DistributedDataParallel
  even for single-node multi-GPU) and GPU-0 carries a heavier memory load
  than other GPUs since outputs/loss are gathered there. Don't assume a
  clean 2x from batch size or throughput -- measure it.
- torch.compile is applied to the model BEFORE DataParallel wraps it
  (fixed from an earlier draft that compiled the DataParallel wrapper
  itself, which is not a supported combination).
"""
import torch
import torch.nn as nn

class HardwareEngine:
    def __init__(self, model, device="cuda", target_batch_size=32, local_batch_size=4):
        self.device = torch.device(device)
        self.gpu_name = torch.cuda.get_device_name(0)
        self.major, self.minor = torch.cuda.get_device_capability(0)
        self.num_gpus = torch.cuda.device_count()

        # 1. Precision Detection (RTX 4050/H100 get bf16, T4 gets fp16)
        if self.major >= 8:
            self.precision = torch.bfloat16
            self.use_scaler = False
            print(f"--- [Modern GPU] Using bfloat16 on {self.gpu_name} ---")
        else:
            self.precision = torch.float16
            self.use_scaler = True
            print(f"--- [Legacy GPU] Using float16 + Scaler on {self.gpu_name} ---")

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_scaler)

        # 2. Gradient Accumulation Logic (The Bridge)
        # Calculates how many steps to wait before updating weights
        self.accum_steps = max(1, target_batch_size // (local_batch_size * self.num_gpus))
        print(f"--- [Compute] Simulating Batch {target_batch_size} via {self.accum_steps} accumulation steps ---")

        # 3. Model Setup & Multi-GPU
        self.model = model.to(self.device)
        self._compiled = False
        if self.num_gpus > 1:
            print(f"--- [Multi-GPU] Wrapping in DataParallel (Kaggle Mode) ---")
            self.model = nn.DataParallel(self.model)

    def compile_model(self):
        """Compiles the inner model correctly to avoid DataParallel conflicts"""
        if self._compiled: return self.model
        try:
            if isinstance(self.model, nn.DataParallel):
                inner = torch.compile(self.model.module)
                self.model = nn.DataParallel(inner)
            else:
                self.model = torch.compile(self.model)
            self._compiled = True
            print("--- [Optimizer] torch.compile successful ---")
        except Exception as e:
            print(f"--- [Optimizer] Compile failed: {e}. Using Eager mode. ---")
        return self.model

    def train_step(self, i, optimizer, criterion, lr_scheduler, input_img, gt_img):
        self.model.train()
        
        # Only zero gradients at the start of a new 'Target Batch'
        if i % self.accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        input_img = input_img.to(self.device, non_blocking=True)
        gt_img = gt_img.to(self.device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=self.precision):
            output = self.model(input_img)
            # Scale loss by accumulation steps so the math stays consistent
            loss = criterion(output, gt_img) / self.accum_steps

        # Backward pass
        if self.use_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights only after N steps
        if (i + 1) % self.accum_steps == 0:
            if self.use_scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            
            if lr_scheduler is not None:
                # If using OneCycleLR, step here. If ReduceLROnPlateau, step in epoch loop.
                if not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step()

        return loss.item() * self.accum_steps