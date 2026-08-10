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

## Latency (idle GPU, no training contention)

| | Restormer | SwinIR |
|---|---|---|
| batch=1 | 19.0 ms/image | 26.9 ms/image |
| batch=8 | 14.7 ms/image | 28.7 ms/image |

Restormer is ~1.4-1.9x faster, and the gap widens with batch size.
Consistent with expectations: window attention does real spatial mixing
per window; channel attention (Restormer) doesn't pay that cost. This is
on a laptop RTX 5050, not the H100 used for actual benchmarking -- the
relative ordering should hold, absolute ms won't transfer, and this
doesn't include the I/O/startup overhead the real benchmark measures.

## Verdict (as of the 12-epoch check)

Quality: effectively tied. Speed: not tied -- Restormer is meaningfully
and consistently faster. Since KLA's actual scoring prefers the faster
pipeline when quality is comparable, **Restormer is the better choice
right now** unless a longer SwinIR run opens a real quality gap that's
worth the latency cost. Not re-run at full 50-epoch budget yet -- ranking
could shift.

## Housekeeping gotcha found

Both `evaluate.py` runs write to the same `./eval_outputs/` filenames
(e.g. `worst0_002982.png`), so running Restormer then SwinIR back to back
silently overwrote Restormer's visual outputs with SwinIR's. Worth
namespacing those filenames by `model_type` before doing another
side-by-side comparison.

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

## Next step under discussion

Before building a third architecture: run the SSIM-loss experiment on
the existing checkpoints (cheap, reuses everything) to check whether the
shared 12.2 dB worst-case floor is loss-driven before spending more GPU
time on new backbones. Not yet run.

## Known compliance gaps (apply regardless of which backbone wins)

Carried over from the earlier Restormer-only review, still open:

- No LPIPS anywhere (it's an official scored metric alongside PSNR/SSIM).
- `evaluation.py` doesn't match the required submission CLI contract
  (`test_images_dir`, `output_dir`); it's a diagnostics script, not the
  deliverable inference script.
- Eval outputs are comparison plots, not raw restored images for
  submission.
- Calibration stats (`calib_stats.json`) aren't portable to a bare test
  directory -- need to be baked into the checkpoint or shipped alongside it.
- No `pip freeze` / requirements.txt in the repo.
- No end-to-end (script startup + I/O + inference) latency benchmark --
  only forward-pass latency has been measured so far.
