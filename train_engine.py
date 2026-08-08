"""
Hardware-aware training engine with gradient accumulation.

Works on RTX 4050, Kaggle Dual T4, or a cloud H100 without edits.

Two things fixed in this version vs. your draft:

1. MULTI-GPU ACCUM MATH. This engine uses nn.DataParallel: ONE process,
   ONE DataLoader. Every train_step() call already receives the FULL
   `batch_size` from that single loader -- DataParallel splits it across
   GPUs *inside* the forward call. So one train_step already consumes
   `batch_size` samples total, across however many GPUs you have.
   `accum_steps` should therefore be `target_batch_size // batch_size`,
   full stop -- no num_gpus multiplier.
   (The `local_batch_size * num_gpus` formula from the draft is the right
   idea for DistributedDataParallel, where each GPU runs its own process
   with its own DataLoader yielding a per-process local batch -- that's
   not what's implemented here, so that formula silently computed the
   wrong accum_steps as soon as local_batch_size didn't happen to equal
   the DataLoader's actual per-GPU share.)

2. CROSS-EPOCH ACCUMULATION BOUNDARY. Micro-step counting is now a
   continuous counter across the WHOLE run, not reset every epoch. If
   steps_per_epoch isn't a multiple of accum_steps, the leftover
   micro-batches at the end of an epoch now correctly carry over into
   the next epoch's accumulation group, instead of being silently
   dropped when the epoch loop's local index reset to 0. Call
   engine.flush(optimizer) once after your training loop fully ends (or
   right before an early-stop break) to step on any leftover gradient
   from the true end of the run.

Known tradeoffs (unchanged from before):
- DataParallel is simpler to set up than DistributedDataParallel but not
  the fastest option, and GPU-0 carries more memory load than the rest.
  Don't assume linear scaling -- measure it.
- torch.compile is applied to the model BEFORE DataParallel wraps it.
"""

import torch
import torch.nn as nn


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