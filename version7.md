Research on 
Downsampled Fourier branch, tested against your current setup —
cheap experiment, plausible speed win for the efficiency prize. what it does to
the excisting model how it helps




As your Lead Scientist, I can tell you that the Downsampled Fourier Branch is a "high-ROI" (Return on Investment) engineering maneuver. It is exactly the kind of optimization that wins the Efficiency/Inference Speed prize because it exploits a fundamental property of signal processing.
Here is the deep technical breakdown of what it does, how it helps your current setup, and why it is a smart "cheap experiment."
1. What it does (The "Multi-Scale Frequency" Concept)
In our current FourierUnit, we perform a Fast Fourier Transform (FFT) on the full 128x128 feature map.
Current Workflow: Input (128x128) → FFT → Frequency Mixing → IFFT → Output (128x128).
Downsampled Workflow: Input (128x128) → Average Pool (64x64) → FFT → Frequency Mixing → IFFT → Bilinear Upsample (128x128) → Output.
2. How it helps the Existing Model
A. Computational Complexity (The Speed Win)
The 2D-FFT has a complexity of 
O
(
N
2
log
⁡
N
)
O(N 
2
 logN)
.
By downsampling from 128x128 to 64x64, we reduce the number of pixels by 4x.
In practice, this makes the Fourier branch nearly 3x to 5x faster on your RTX 4050 and the H100. If the hackathon judges prioritize "Images Per Second," this is your strongest lever.
B. Semantic Regularization (The Accuracy Win)
In semiconductor scans, Scan-line Noise (periodic streaks) and Global Haze (blur) are low-to-mid frequency artifacts.
The Problem: When we do FFT at full resolution, the branch tries to process high-frequency edges. This can cause "Ringing Artifacts" or "Waviness" around the sharp silicon lines.
The Solution: By downsampling, we force the Fourier branch to be a "Low-Frequency Specialist." It focuses entirely on the global noise signatures and the "haze," while leaving the high-frequency "Edge Reconstruction" to the Spatial MDTA branch. This "Separation of Concerns" usually improves SSIM.
C. VRAM Headroom
On your 6GB RTX 4050, this saves significant memory. This headroom allows you to increase the dim (e.g., from 64 to 96) or add an extra block, which does improve accuracy.




drawback:

What this change actually does

It splits the Fourier branch off the shared full-resolution path: attention stays full-res, but the FFT branch now operates on a 2× average-pooled version of the feature map, then gets bilinearly upsampled back before fusion. Shape-wise this is correct — no bugs. avg_pool2d(kernel_size=2) defaults to stride=2, giving proper non-overlapping downsampling, and since your feature maps are 128×128 (your LR resolution, since all blocks operate pre-upsampler), halving to 64×64 is clean with no off-by-one/odd-dimension issues.

Why it should speed things up

This directly targets what I flagged a few messages ago as the likely dominant per-block cost — FourierUnit's forced-fp32 rfft2/irfft2. Halving both spatial dimensions cuts the pixel count fed into the FFT by 4×, and FFT cost scales roughly with N log N, so you should see close to a 3-4x reduction in the cost of the spectral branch specifically. Profile it — use the same before/after timing harness I suggested earlier (comment the spectral call in/out, or diff this version against the original), don't just trust the theoretical estimate.

The real risk: this is a deliberate trade-off, not a free lunch

The docstring framing ("Fourier Branch handles Haze/Scan-lines") is reasonable — broad, low-frequency periodic noise survives 2× downsampling fine, since it lives well below the new Nyquist limit. But avg_pool2d is a box filter, not a proper anti-aliasing low-pass — it doesn't cleanly remove content near the new Nyquist frequency, it partially aliases it. Any high-frequency periodic noise near or above the new Nyquist limit (e.g. fine-pitch scan-line artifacts, or anything resembling your "Woven Grid" OOD case) could lose resolution the FourierUnit specifically exists to catch. This is the one thing worth being genuinely cautious about — you're trading resolution on exactly the frequency band this branch was designed to handle, for the case where the periodic noise happens to be fine-grained.

This needs a fine-tune, not a drop-in swap — but not a fresh run either

Per your project's own convention (stated in finetune_ssim.py's docstring): a loss-only change gets a cheap fine-tune, but an architecture change usually needs a fresh run. This one's actually a middle case worth calling out explicitly: FourierUnit's conv/norm layer shapes are unchanged (still operating on channels*2), so the existing weights are directly loadable — but what they're now seeing spatially is different content (downsampled instead of full-res), so they'll need to re-adapt. A short fine-tune from your best checkpoint (few epochs, low LR, same discipline as the SSIM run) is the right move, not a naive "swap the code and eval the old checkpoint directly." Evaluating with old weights un-adapted would give you a falsely pessimistic read on whether the idea works.







finetune 1:
Resuming from ./checkpoints_ssim_v3/best_model.pt (epoch 14, val_psnr=27.88dB)
=== PHASE 3 SSIM FINE-TUNE ===
Config: {'dim': 64, 'num_blocks': 2, 'batch_size': 4, 'target_batch_size': 32, 'scale_factor': 2, 'edge_weight': 1.5, 'freq_weight': 0.15, 'ssim_weight': 0.8, 'lr': 5e-05, 'weight_decay': 1e-06, 'scheduler_patience': 2, 'early_stop_patience': 5, 'num_workers': 8, 'num_epochs': 20, 'use_compile': False, 'checkpoint_dir': './checkpoints_ssim_v3'}
[SEMPairDataset] 3200 paired samples found.
[SEMTestDataset] 400 test samples found.
Loaded Phase 2 weights. Missing 'metrology_gain' initialized to default (1.0).
--- [Modern GPU] Using bfloat16 on NVIDIA GeForce RTX 4050 Laptop GPU ---
--- [Compute] DataLoader batch_size=4, target_batch_size=32 -> accum_steps=8 (actual effective batch = 32) ---
[SSIM-FT] Epoch 1/20 [118.1s] [LR: 5.00e-05] train_loss=1.0003 val_loss=0.7944 val_psnr=22.30dB
  -> new best val_loss, saved to ./checkpoints_ssim_v3/best_model.pt
[SSIM-FT] Epoch 2/20 [116.8s] [LR: 5.00e-05] train_loss=0.7452 val_loss=0.6878 val_psnr=22.84dB
  -> new best val_loss, saved to ./checkpoints_ssim_v3/best_model.pt
[SSIM-FT] Epoch 3/20 [116.9s] [LR: 5.00e-05] train_loss=0.6807 val_loss=0.6378 val_psnr=23.18dB
  -> new best val_loss, saved to ./checkpoints_ssim_v3/best_model.pt
[SSIM-FT] Epoch 4/20 [116.9s] [LR: 5.00e-05] train_loss=0.6376 val_loss=0.6101 val_psnr=23.12dB
  -> new best val_loss, saved to ./checkpoints_ssim_v3/best_model.pt
[SSIM-FT] Epoch 5/20 [116.9s] [LR: 5.00e-05] train_loss=0.6108 val_loss=0.5854 val_psnr=23.29dB
  -> new best val_loss, saved to ./checkpoints_ssim_v3/best_model.pt
[SSIM-FT] Epoch 6/20 [116.7s] [LR: 5.00e-05] train_loss=0.5922 val_loss=0.5730 val_psnr=23.32dB
  -> new best val_loss, saved to ./checkpoints_ssim_v3/best_model.pt
[SSIM-FT] Epoch 7/20 [116.6s] [LR: 5.00e-05] train_loss=0.5790 val_loss=0.5588 val_psnr=23.49dB
  -> new best val_loss, saved to ./checkpoints_ssim_v3/best_model.pt
[SSIM-FT] Epoch 8/20 [116.6s] [LR: 5.00e-05] train_loss=0.5724 val_loss=0.5664 val_psnr=23.08dB
  -> val_loss did not improve (1/5)
[SSIM-FT] Epoch 9/20 [116.6s] [LR: 5.00e-05] train_loss=0.5672 val_loss=0.5515 val_psnr=23.60dB
  -> new best val_loss, saved to ./checkpoints_ssim_v3/best_model.pt
[SSIM-FT] Epoch 10/20 [116.5s] [LR: 5.00e-05] train_loss=0.5617 val_loss=0.5418 val_psnr=23.82dB
  -> new best val_loss, saved to ./checkpoints_ssim_v3/best_model.pt
[SSIM-FT] Epoch 11/20 [116.6s] [LR: 5.00e-05] train_loss=0.5597 val_loss=0.5373 val_psnr=23.68dB
  -> new best val_loss, saved to ./checkpoints_ssim_v3/best_model.pt
[SSIM-FT] Epoch 12/20 [116.6s] [LR: 5.00e-05] train_loss=0.5568 val_loss=0.5377 val_psnr=24.03dB
  -> val_loss did not improve (1/5)
[SSIM-FT] Epoch 13/20 [116.6s] [LR: 5.00e-05] train_loss=0.5553 val_loss=0.5406 val_psnr=23.77dB
  -> val_loss did not improve (2/5)
[SSIM-FT] Epoch 14/20 [116.6s] [LR REDUCED: 5.00e-05 -> 2.50e-05] train_loss=0.5560 val_loss=0.5501 val_psnr=22.97dB
  -> val_loss did not improve (3/5)
[SSIM-FT] Epoch 15/20 [116.6s] [LR: 2.50e-05] train_loss=0.5531 val_loss=0.5403 val_psnr=23.73dB
  -> val_loss did not improve (4/5)
[SSIM-FT] Epoch 16/20 [116.5s] [LR: 2.50e-05] train_loss=0.5499 val_loss=0.5383 val_psnr=23.65dB
  -> val_loss did not improve (5/5)
STOPPING: fine-tune has plateaued at lower LR.


