# Round 4 changes: hardware-aware engine + full training script

## train_engine.py (new)
Rewrite of the "Auto-Hardware Metrology Engine" with one real bug fixed:

- **Compile ordering fixed.** Original called `torch.compile()` on the
  model AFTER wrapping it in `nn.DataParallel`. `torch.compile` on a
  DataParallel-wrapped module is not a supported combination -- Data
  Parallel replicates the module per forward call, which conflicts with
  how a compiled graph is traced/cached. Fixed: the raw model is moved to
  device and (optionally) compiled first; `nn.DataParallel` wraps the
  already-compiled module afterward, in `__init__`. `compile_model()` also
  handles the case where you call it after DataParallel is already
  applied, by unwrapping, compiling the inner module, and rewrapping.
- Dropped the unused `from torch.cuda.amp import autocast` import (never
  used -- the code calls `torch.amp.autocast('cuda', ...)` instead).
  Switched `GradScaler` to the non-deprecated `torch.amp.GradScaler('cuda', ...)`
  constructor.
- Added `eval_step()` (wasn't in the original) since `train.py` needs a
  no-grad forward pass for validation.
- Precision auto-selection (bf16 no-scaler on capability>=8, fp16+scaler
  below) and set_to_none=True kept as-is -- these were already correct.
- DataParallel imbalance and "measure, don't assume 1.7x" caveats are
  documented in the module docstring and printed at runtime, not just
  asserted in prose.

## train.py (new)
Ready-to-run script wiring calibrate_stats.py's output through
sem_dataset.py and model.py, using the fixed engine. Notable pieces:

- **Config block at the top** for the baseline vs. metrology HPO runs
  from the team plan (`freq_weight`, `num_blocks`, etc. all in one place).
- **Overfitting guard is now an actual early-stop**, not just a rule to
  watch for manually: tracks best validation loss, saves a checkpoint
  every time it improves, and stops training after `patience` epochs of
  no improvement. This directly implements the "if val loss rises while
  train loss falls, stop immediately" rule from the war-room plan.
- Reports val PSNR each epoch alongside loss, since that's the metric
  you'll actually be judged on.
- Checkpoint saving correctly unwraps `DataParallel` before calling
  `.state_dict()` (saving a DataParallel-wrapped state dict adds a
  `module.` prefix to every key, which silently breaks loading the
  checkpoint into a plain model later for inference/submission).
- Exits cleanly with a clear warning if no CUDA device is found, rather
  than crashing partway through (verified below).

## Verified
- All five files (`calibrate_stats.py`, `sem_dataset.py`, `model.py`,
  `train_engine.py`, `train.py`) compile with no syntax errors.
- `train.py` on a CPU-only machine (this sandbox) prints its config and
  exits cleanly with an explanatory warning instead of crashing --
  confirms the CUDA-availability guard works.
- Full data -> model -> loss -> optimizer wiring dry-run on synthetic data
  matching your real folder structure, with `freq_weight=0.1` (the
  "metrology run" setting) and the weight-decay-grouped optimizer both
  exercised: produced a valid loss and completed a full backward/step
  with no errors.
- Not verified here (no GPU in this environment): the actual bf16 vs
  fp16 autocast branch, DataParallel multi-GPU path, and torch.compile.
  Run `train.py` on your real hardware (4050 / Kaggle T4s / H100) to
  confirm those -- the CPU dry-run only proves the data/model/loss logic
  is correct, not the hardware-specific paths.

## Still on you
- `freq_weight`, `num_blocks`, `dim` are still hyperparameters to sweep
  against validation curves, not values to trust as final from this code.
- Benchmark actual DataParallel speedup and torch.compile benefit on your
  specific hardware before citing any number in your presentation.
