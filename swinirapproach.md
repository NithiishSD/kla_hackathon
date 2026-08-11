# SwinIR Approach

Branch: `swinir-approach`. Adds SwinIR as a second, directly comparable
backbone to the Restormer model (`model.py`), reusing the same data
pipeline, loss, training engine, and evaluation script -- only the
architecture differs.

## What's in this branch

- `swinir_model.py` -- SwinIR backbone: RSTB blocks (shifted-window
  self-attention + relative position bias), reflect-pad to window
  multiples, PixelShuffle 2x upsample. Same global-residual +
  zero-init-last-conv trick as `SemiRestoreNet_V2` (Restormer model), so
  training starts at the bicubic baseline instead of noise. Reuses
  `KLAMetrologyLoss` / `build_optimizer` from `model.py` rather than
  duplicating them.
- `train_swinir.py` -- mirrors `train.py`'s loop exactly (same
  `HardwareEngine`, dataloaders, checkpointing), saves to
  `./checkpoints_swinir/`, matching the existing `train.py` /
  `train_baseline.py` convention.
- `evaluation.py` -- added `build_model(cfg, device)`, which branches on
  `cfg.get("model_type", "restormer")` so the same eval script now works
  for both checkpoint types without duplicating the eval/plotting logic.
- `restoration_pipeline.ipynb` -- the whole pipeline (calibration, dataset,
  both models, training engine, both training loops, evaluation, plus a
  generic `finetune_restormer(...)` for loss-ablation experiments)
  consolidated into one notebook. Cells carry real executed output (not
  just source) -- run live top-to-bottom, including the corrected
  MS-SSIM checkpoint evaluation in section 11.
- `model.py` -- added multi-scale SSIM (`ms_ssim` / `ms_ssim_loss` /
  `ms_ssim_weight` on `KLAMetrologyLoss`), and `evaluation.py` got the
  matching `compute_ms_ssim` metric.
- `infer.py` -- the actual submission-required inference script (see
  "Submission compliance" below).
- `requirements.txt` -- `pip freeze` output from the environment both
  models were actually trained in.

Config used: `embed_dim=60, depths=(4,4,4,4), num_heads=(6,6,6,6),
window_size=8, mlp_ratio=2` -- lighter than official SwinIR-light, given
this task's speed requirement.

## Bug found and fixed while building it

Two `transpose(1, 2).view(...)` chains (in `RSTB.forward` and
`SwinIR.forward`) would have crashed at runtime -- PyTorch refuses
`.view()` on a non-contiguous tensor, and `transpose()` always produces
one. Fixed by adding `.contiguous()` before each `.view()`, matching the
pattern the official SwinIR/`PatchUnEmbed` implementation uses.

## Real data used

`data/train/gt` (3200, 256x256), `data/train/NoisyLR` (3200, 128x128),
`data/test` (400, 128x128, no GT) -- the 256->128 resolution pair.
Calibration: `p1=0.0090, p99.5=1.1852` (raw LR range was
`[-0.279, 2.158]`, confirming the spec's speckle-exceeds-GT-range note
on this exact dataset).

## Quick-check run: 12 epochs each, same val split

Both scripts default to `num_epochs=50`; a full run is ~1.9h (Restormer)
/ ~2.9h (SwinIR) on the local RTX 5050 Laptop GPU (8GB). For a same-session
comparison, both were run for a bounded 12 epochs instead (identical
config otherwise), then evaluated with the same `evaluation.py` against
the same held-out 320-sample val split (deterministic split, seed=42).

| Metric | Restormer | SwinIR |
|---|---|---|
| PSNR mean | 28.87 dB | 29.01 dB |
| PSNR std | 4.34 | 4.47 |
| SSIM mean | 0.7654 | 0.7697 |
| Worst-case (002982.npy) | 12.26 dB / 0.272 SSIM | 12.19 dB / 0.274 SSIM |
| Best-case (001827.npy) | 42.93 dB / 0.960 SSIM | 44.65 dB / 0.968 SSIM |
| Params | 0.59M | 0.79M |

SwinIR is marginally ahead on average quality (+0.14 dB / +0.004 SSIM) --
within noise for a 12-epoch run. Both models hit essentially the same
floor on the same worst-case sample (~12.2 dB either way), which is the
"melting"/over-smoothing failure on fine-texture content -- neither
architecture fixes it; **this looks like a loss-function problem (no
SSIM/perceptual term was active in either run), not an architecture
problem.**

One process note: the SwinIR run's epoch 4 logged 8272s instead of the
normal ~255s -- the laptop went to sleep mid-run. Training resumed
correctly afterward (loss/PSNR kept moving in the right direction); it's
a wall-clock artifact, not a training issue.

## Full 50-epoch run: converged comparison

Both models re-run at their actual configured budget (`num_epochs=50`,
unmodified `train.py` / `train_swinir.py`), evaluated the same way.
Restormer ran the full 50 epochs without early-stopping (still improving,
though plateaued in its last ~15 epochs). SwinIR ran the full 50 too,
needed an LR drop at epoch 47 to squeeze out its last gains.

| Metric | Restormer | SwinIR |
|---|---|---|
| PSNR mean | 29.43 dB | **29.85 dB** |
| PSNR std | 4.53 | 4.67 |
| SSIM mean | 0.7866 | **0.7951** |
| Worst-case (002982.npy) | 12.64 dB / 0.322 SSIM | 12.49 dB / 0.325 SSIM |
| Best-case (001827.npy) | 43.72 dB / 0.969 SSIM | **45.71 dB / 0.966 SSIM** |
| Params | 0.59M | 0.79M |

Unlike the 12-epoch check, SwinIR's quality lead is now real:
**+0.42 dB PSNR, +0.0085 SSIM**, both runs fully converged. The shared
worst-case sample (002982.npy) is still ~12.5 dB for both -- full training
didn't move it either, reinforcing that it's a ceiling, not a
training-budget problem (see the SSIM-loss experiment below for why).

One process note from this run: Restormer's checkpoint mtime tracking
showed it training steadily throughout (~155s/epoch, no interruptions);
SwinIR ran cleanly this time too (~250-260s/epoch, no repeat of the
12-epoch check's sleep-induced stall).

## Latency (idle GPU, no training contention)

Measured twice -- once during the 12-epoch check, once after the full
50-epoch run finished (both idle-GPU reads, consistent with each other):

| | Restormer | SwinIR |
|---|---|---|
| batch=1 | 19.0-19.1 ms/image | 26.9-27.2 ms/image |
| batch=8 | 14.6-14.7 ms/image | 28.0-28.7 ms/image |

Restormer is ~1.4-1.9x faster, and the gap widens with batch size.
Consistent with expectations: window attention does real spatial mixing
per window; channel attention (Restormer) doesn't pay that cost. This is
on a laptop RTX 5050, not the H100 used for actual benchmarking -- the
relative ordering should hold, absolute ms won't transfer, and this
doesn't include the I/O/startup overhead the real benchmark measures.

## Verdict (converged, full 50-epoch budget)

No clean winner. **SwinIR wins quality by a real, converged margin
(+0.42 dB PSNR). Restormer wins speed by a consistent, real margin
(~1.4-1.9x).** KLA's scoring prefers the faster pipeline "when quality is
comparable" -- a 0.42 dB gap is arguably not comparable, so this is a
genuine trade-off call, not something resolved by more training. Pick
based on how the actual competition weighs the two; nothing here settles
it further.

## Housekeeping gotcha found (fixed)

Both `evaluate.py` runs wrote to the same `./eval_outputs/` filenames
(e.g. `worst0_002982.png`), so running Restormer then SwinIR back to back
silently overwrote Restormer's visual outputs with SwinIR's.
**Fixed** -- output filenames are now tagged with `model_type`
(`worst0_restormer_002982.png` / `worst0_swinir_002982.png`), verified
both coexist after running both models back to back.

## LPIPS added -- real three-metric comparison

Installed `lpips` (AlexNet backbone; grayscale replicated to 3 channels
since there's no single-channel variant). Added to `evaluation.py` as
`compute_lpips()`, lazy-loaded so it only costs anything in this
diagnostics script, never in `infer.py` (the timed submission script).

| Metric | Restormer | SwinIR |
|---|---|---|
| PSNR mean | 29.43 dB | **29.85 dB** |
| SSIM mean | 0.7866 | **0.7951** |
| LPIPS mean (lower better) | **0.2547** | 0.2559 |

A genuine three-way tension, not just quality-vs-speed: **Restormer is
marginally better on LPIPS** despite losing on PSNR/SSIM. The gap is
small (0.0012) but consistent with expectations -- SwinIR's higher
PSNR/SSIM could partly reflect fitting pixel-level/structural targets
slightly more aggressively, without a matching perceptual-similarity
gain. On the specific worst-case sample (002982.npy) the pattern reverses
sharply: SwinIR's LPIPS (0.2876) is much better than Restormer's (0.3409)
despite near-identical PSNR (~12.5 dB both) -- SwinIR's output is
perceptually closer even though pixel-error is a wash there. Reinforces
that no single metric tells the whole story; worth reporting all three
whenever comparing checkpoints going forward.

## Submission compliance: infer.py, baked calibration, requirements.txt

All four of KLA's required submission components (page 17) now exist:

1. **`infer.py`** -- the actual required inference script, built to spec:
   standalone (non-notebook), accepts `(test_images_dir, output_dir)` as
   positional CLI args, loads a checkpoint, runs inference on every
   image, writes raw restored `.npy` outputs (not diagnostic plots) back
   to disk, runs with zero manual edits. Deliberately does *not* import
   `evaluation.py` (which pulls in matplotlib) -- every import in the
   timed script costs startup latency. Defaults to the Restormer
   checkpoint; swap via `--checkpoint checkpoints_swinir/best_model.pt`.
2. **`restoration_pipeline.ipynb`** -- training script deliverable
   (notebook form), see above.
3. **Denoised test outputs** -- generated by actually running `infer.py`
   against all 400 real test images for both checkpoints
   (`submission_test_outputs_restormer/`, `submission_test_outputs_swinir/`).
   Verified: 400/400 files each, correct shape (256x256, upscaled from
   128x128 input), correct dtype (float32), correctly de-normalized back
   to the original raw intensity scale (not left in the internal [0,1]
   model space) -- range matches the baked calibration bounds.
4. **`requirements.txt`** -- generated via `pip freeze` from the actual
   environment both training runs used. Confirms `torch==2.13.0+cu130`,
   `numpy`, `matplotlib`, `torchvision`, `lpips`. Note: this is the
   global Python 3.14 install, not an isolated project venv, so the file
   (136 lines) includes some unrelated packages too -- left as-is since
   the spec's own FAQ says to submit the full `pip freeze` output, not a
   hand-pruned one.

**Also fixed:** calibration stats (`p_low`/`p_high`) are now baked
directly into both final checkpoints (`calib_p_low` / `calib_p_high`
keys added to the saved dict) instead of requiring a colocated
`calib_stats.json` next to a `train/` folder -- `infer.py` reads them
from the checkpoint, so it works against a bare test-images directory
with no other structure around it, which is what the real benchmark will
actually hand it.

**Real end-to-end latency** (script startup + model init + disk read +
inference + disk write -- exactly what the spec says gets measured,
timed externally around the whole `python infer.py ...` invocation, on
all 400 real test images):

| | Restormer | SwinIR |
|---|---|---|
| End-to-end | 17.16s / 400 images = **42.9 ms/image** | 19.85s / 400 images = **49.6 ms/image** |

Same ordering as the pure forward-pass latency numbers above, but the
gap narrows (1.16x here vs 1.4-1.9x forward-only) -- fixed overhead
(script startup, model load, DataLoader setup) is a larger fraction of
the total at this image count, so it dilutes the per-model compute
difference somewhat. Still on the laptop RTX 5050, not H100.

## Other architectures considered, not yet built

- **NAFNet** (top candidate for next experiment) -- U-Net-shaped,
  attention-free (simplified channel gating), proven on real
  denoising/deblurring, should beat both current models on speed.
- **RRDBNet** (Real-ESRGAN backbone) -- pure CNN, closest published
  precedent to this exact "combined noise + downsampling, invert in one
  pass" problem. Candidate if NAFNet's quality ceiling disappoints.
- **Uformer** -- U-Net-shaped window attention, same speed class as
  SwinIR, not pursued since it doesn't address the speed gap.
- **HAT** -- better SR quality in the literature but heavier than
  SwinIR (which already lost on speed); ruled out given the speed
  constraint.
- **Diffusion / GAN-based restoration** -- ruled out: multi-step
  sampling / adversarial training cost too much inference latency for an
  H100-latency-graded benchmark.

## SSIM-loss experiment: negative result

Hypothesis: the shared worst-case floor (~12.5 dB on 002982.npy, both
architectures, both training budgets) is a loss-function problem --
Charbonnier/L1-style losses are documented to over-smooth ("melt") fine
texture -- so adding an SSIM term should recover some of it.

Test: fine-tuned the converged Restormer checkpoint
(`checkpoints/best_model2.pt`, 27.91 dB) with `ssim_weight=0.3` added to
`KLAMetrologyLoss`, `lr=5e-5`, up to 15 epochs (early-stopped at 13).
002982.npy stays in the held-out val split (fixed seed=42) throughout, so
it's genuinely unseen during the fine-tune too. Saved to
`checkpoints/best_model_ssim.pt` (does not overwrite the baseline).

| | Aggregate val PSNR | 002982.npy PSNR | 002982.npy SSIM |
|---|---|---|---|
| Before | 27.91 dB | 12.64 dB | 0.3220 |
| After | ~24.2 dB (best epoch) | 12.48 dB | 0.2985 |
| Delta | **-3.7 dB** | **-0.16 dB** | **-0.0235** |

**Hypothesis not confirmed -- both got worse.** Two things going on:

1. `ssim_weight=0.3` was likely too aggressive on its own -- a 3.7 dB
   aggregate regression from one added term is a real destabilization,
   not a gentle refinement. Picked from general SR-literature convention,
   not tuned for this task.
2. More importantly: the worst-case sample not improving *at all* (even
   slightly negative) suggests the problem isn't "wrong loss" so much as
   an information-theoretic ceiling. That sample's ground truth is
   near-random, fine stochastic texture (see the eval_outputs plot) --
   for content that's genuinely unpredictable at the pixel level, no
   deterministic loss (L1, SSIM, or any combination) can make a
   feed-forward model output the *specific* random realization in the
   ground truth. A smooth/averaged prediction is the mathematically
   correct response to "minimize expected error against an unpredictable
   target." Fixing this would need a generative approach (GAN/diffusion)
   that samples a plausible texture instead of regressing to a mean --
   and those trade PSNR/SSIM away for perceptual realism (LPIPS), they
   don't improve PSNR/SSIM on this kind of sample either.

**Conclusion: don't chase this specific sample further.** Treat it as an
expected outlier in the worst-5 report, not a fixable bug.
`checkpoints/best_model_ssim.pt` is strictly worse than the baseline on
every measured axis -- not promoted, baseline checkpoints remain the best
result.

## MS-SSIM loss experiment: also negative, and it exposed a real bug

Follow-up to the single-scale SSIM result: implemented proper multi-scale
SSIM (Wang/Simoncelli/Bovik 2003 -- 5 scales via successive 2x
downsampling, contrast*structure combined at each scale, full SSIM incl.
luminance only at the coarsest scale) as both an evaluation metric
(`compute_ms_ssim`) and a loss term (`KLAMetrologyLoss.ms_ssim_weight`).
Sanity-checked first: identical images -> MS-SSIM = 1.0 exactly, random
vs. random -> 0.22, gradients finite on a synthetic batch.

Ran the same fine-tune protocol as the SSIM experiment (resume from
`checkpoints/best_model2.pt`, `lr=5e-5`, up to 15 epochs), but with a more
conservative `ms_ssim_weight=0.15` (vs. the earlier `ssim_weight=0.3`),
on the theory that weight magnitude was part of what hurt last time.

**Training collapsed at epoch 4.** val_psnr went 19.19 -> 20.30 -> 20.95dB
(epochs 1-3, already a big regression from 27.91dB, same shape as the
SSIM run) then **cratered to 7.73dB at epoch 4 and never recovered**,
early-stopping at epoch 8 with val_psnr flat at 7.73dB. Likely cause:
MS-SSIM's per-scale terms are combined via `value ** weight` with
fractional weights (~0.04-0.30) -- the *forward* value is safely clamped
away from zero, but the *gradient* of `x**w` is `w * x**(w-1)`, and
`w-1` is negative, so as a per-scale SSIM value approaches the clamp
floor (plausible for the flat/saturated/degenerate patches this dataset's
own histograms show), the backward gradient can blow up even though the
forward pass looks numerically fine. A known failure mode of naive
power-law MS-SSIM implementations, not something the `eps` clamp alone
protects against.

**Process note -- a bug in my own comparison script, caught and fixed
before reporting anything:** the first version of the before/after check
evaluated the *live end-of-training model object* (i.e. the collapsed
epoch-8 state) instead of reloading the checkpoint that actually got
saved (`checkpoints/best_model_msssim.pt`, saved at epoch 3, before the
collapse -- checkpoints only save on improvement). That first pass
reported a nonsensical PSNR=6.27dB/SSIM=0.0001 "result." Caught this by
checking the saved checkpoint's own recorded epoch/PSNR before trusting
the comparison script, then re-ran the check correctly (reload from disk,
not the live object) -- **live-executed in the notebook**
(`restoration_pipeline.ipynb`, section 11) rather than re-described from
a scratch log, so the corrected numbers below are from an actual fresh
execution, not a copy-paste.

Corrected result (the checkpoint that actually exists on disk, epoch 3,
before the collapse):

| | Aggregate val PSNR | Aggregate val SSIM | 002982.npy PSNR | 002982.npy SSIM | 002982.npy MS-SSIM |
|---|---|---|---|---|---|
| Before (baseline) | 27.91 dB | 0.7866 | 12.64 dB | 0.3220 | 0.7650 |
| After (ep. 3, pre-collapse) | 23.02 dB | 0.5441 | 12.48 dB | 0.2870 | 0.6807 |
| Delta | **-4.89 dB** | **-0.2425** | -0.16 dB | -0.0350 | -0.0844 |

Worse than the single-scale SSIM result on every axis, including the
worst-case sample -- and the aggregate SSIM's minimum across the val set
was -0.0398 (a negative SSIM, i.e. structurally anti-correlated with the
target on at least one sample), a sign this checkpoint was already
unstable even before the epoch-4 collapse, not just "regressed."

**Conclusion: same as the SSIM experiment, more strongly.** The shared
worst-case sample still doesn't improve (now with two different
structural-similarity losses tested), reinforcing the information-
theoretic-ceiling read. On top of that, `ms_ssim_weight=0.15` at
`lr=5e-5` is numerically unsafe with this implementation and this
optimizer/precision setup (bf16 autocast) -- a smaller weight or a
log-space-combined MS-SSIM formulation (avoids the `x**w` gradient blowup
by summing `w * log(x)` instead of multiplying `x**w`) would need to be
tried before concluding MS-SSIM itself is unusable here, but that's a
different, lower-priority experiment now given two structural-similarity
losses have both hit the same wall. `checkpoints/best_model_msssim.pt`
is not promoted -- baseline checkpoints remain the best result.

## Next steps under discussion

- SSIM-loss experiment: done (see above, negative result).
- MS-SSIM-loss experiment: done (see above, negative and worse, plus
  exposed a gradient-stability issue in naive power-law MS-SSIM).
- A smaller `ssim_weight`/`ms_ssim_weight`, or a log-space MS-SSIM
  formulation that avoids the `x**w` gradient blowup, are technically
  untested, but the ceiling argument (same worst-case sample failing
  under two different structural-similarity losses now) means they're
  unlikely to move that specific sample even if they stabilize training
  -- lower priority now.
- NAFNet / RRDBNet -- still the more promising unexplored direction for
  the actual speed-vs-quality trade-off, since neither model swap nor
  loss engineering (two variants now) closed that gap so far.

## Compliance status

All the gaps from the earlier Restormer-only review are now closed (see
"Submission compliance" and "LPIPS added" above): LPIPS is measured,
`infer.py` matches the required CLI contract, outputs are real restored
`.npy` files, calibration is baked into the checkpoints, `requirements.txt`
exists, and end-to-end latency is measured on real data.

What's genuinely still open:

- **Which checkpoint to actually submit is still an open call**, not a
  compliance gap -- `infer.py` defaults to Restormer (speed) but SwinIR
  (quality) is the `--checkpoint` swap away. Both are equally
  submission-ready now; this is the same quality-vs-speed trade-off from
  the verdict above, unresolved by anything done here.
- `infer.py` has only been run on the local RTX 5050, not the actual
  H100 benchmark environment -- the CLI contract and logic are
  hardware-agnostic so this should transfer, but it hasn't been verified
  on the real target hardware.
- LPIPS is measured as a metric but not used as a loss term anywhere --
  given the SSIM-loss experiment's negative result and its
  information-theoretic-ceiling explanation, adding LPIPS to training
  would very likely hit the same wall on the same worst-case sample, so
  this is deliberately not prioritized, not an oversight.
