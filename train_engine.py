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


class HardwareAwareEngine:
    def __init__(self, model, device="cuda"):
        self.device = torch.device(device)
        self.gpu_name = torch.cuda.get_device_name(0)
        self.major, self.minor = torch.cuda.get_device_capability(0)
        self.num_gpus = torch.cuda.device_count()

        # 1. Hardware-specific precision tuning.
        # Ampere+ (capability >= 8, e.g. RTX 30/40-series, A100, H100) has
        # native bf16 tensor-core support. Turing and earlier (T4, capability
        # 7.5) does not -- bf16 there is much slower than fp16, so use fp16
        # + GradScaler instead.
        if self.major >= 8:
            self.precision = torch.bfloat16
            self.use_scaler = False
            print(f"--- [Modern GPU] Compute capability {self.major}.{self.minor} "
                  f"({self.gpu_name}) -- using bfloat16, no GradScaler. ---")
        else:
            self.precision = torch.float16
            self.use_scaler = True
            print(f"--- [Legacy/T4 GPU] Compute capability {self.major}.{self.minor} "
                  f"({self.gpu_name}) -- using float16 + GradScaler. ---")

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_scaler)

        # 2. Move + (optionally) compile the RAW model BEFORE wrapping in
        # DataParallel. torch.compile on a DataParallel-wrapped module is
        # not a supported combination (DataParallel replicates the module
        # per forward call, which conflicts with how Inductor traces/caches
        # a compiled graph) -- compiling the inner module first and letting
        # DataParallel replicate the *compiled* module is the correct order.
        self.model = model.to(self.device)
        self._compiled = False

        # 3. Multi-GPU wrapping happens last, after any compile call.
        if self.num_gpus > 1:
            print(f"--- [Multi-GPU] {self.num_gpus} GPUs found. Wrapping in "
                  f"DataParallel. Note: GPU-0 will carry a heavier memory "
                  f"load than the others (outputs/loss are gathered there); "
                  f"don't assume linear scaling. ---")
            self.model = nn.DataParallel(self.model)

    def compile_model(self):
        """Compile the model. Call this BEFORE training starts, and only
        once. Safe to call even if it's a no-op on unsupported setups --
        falls back to eager on failure."""
        if self._compiled:
            return self.model
        try:
            print("--- [Optimizer] Attempting torch.compile... ---")
            if isinstance(self.model, nn.DataParallel):
                # Compile the inner module, then rewrap -- see note above.
                inner = torch.compile(self.model.module)
                self.model = nn.DataParallel(inner)
            else:
                self.model = torch.compile(self.model)
            self._compiled = True
            print("--- [Optimizer] Compile requested successfully "
                  "(graph builds lazily on first forward pass -- verify "
                  "with a real batch before trusting it in training). ---")
        except Exception as e:
            print(f"--- [Optimizer] Compile failed/unsupported: {e}. "
                  f"Falling back to eager. ---")
        return self.model

    def train_step(self, optimizer, criterion, lr_scheduler, input_img, gt_img):
        self.model.train()
        optimizer.zero_grad(set_to_none=True)

        input_img = input_img.to(self.device, non_blocking=True)
        gt_img = gt_img.to(self.device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=self.precision):
            output = self.model(input_img)
            loss = criterion(output, gt_img)

        if self.use_scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        return loss.item()

    @torch.no_grad()
    def eval_step(self, criterion, input_img, gt_img):
        self.model.eval()
        input_img = input_img.to(self.device, non_blocking=True)
        gt_img = gt_img.to(self.device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=self.precision):
            output = self.model(input_img)
            loss = criterion(output, gt_img)

        return loss.item(), output
