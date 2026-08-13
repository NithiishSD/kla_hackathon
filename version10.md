Step 1: Post-Processing CD Correction (10 minutes)
    (/home/nithiish/Documents/kla_hackathon/.conda) nithiish@nithiish-pc:~/Documents/kla_hackathon$ python correct_cd.py
[SEMPairDataset] 3200 paired samples found.
[SEMTestDataset] 400 test samples found.
Bias     Mean Abs Diff  
-------------------------
Dilate 0  px: 0.047287
Dilate 5  px: 0.074901
Dilate 10 px: 0.101678
Dilate 15 px: 0.115882
Dilate 17 px: 0.120847 ← best
Dilate 20 px: 0.130310
Dilate 25 px: 0.139862

Dilate 0  px: 0.010813
Dilate 5  px: 0.032483
Dilate 10 px: 0.068657
Dilate 15 px: 0.095511
Dilate 17 px: 0.106190 ← best
Dilate 20 px: 0.127099
Dilate 25 px: 0.150839

Dilate 0  px: 0.019162
Dilate 5  px: 0.040536
Dilate 10 px: 0.073104
Dilate 15 px: 0.095750
Dilate 17 px: 0.104325 ← best
Dilate 20 px: 0.120008
Dilate 25 px: 0.136634

Dilate 0  px: 0.015806
Dilate 5  px: 0.027765
Dilate 10 px: 0.053319
Dilate 15 px: 0.071871
Dilate 17 px: 0.078768 ← best
Dilate 20 px: 0.090921
Dilate 25 px: 0.104223

Dilate 0  px: 0.008284
Dilate 5  px: 0.014290
Dilate 10 px: 0.027131
Dilate 15 px: 0.037079
Dilate 17 px: 0.040722 ← best
Dilate 20 px: 0.049318
Dilate 25 px: 0.058424

Dilate 0  px: 0.033750
Dilate 5  px: 0.060311
Dilate 10 px: 0.089693
Dilate 15 px: 0.105220
Dilate 17 px: 0.110698 ← best
Dilate 20 px: 0.120551
Dilate 25 px: 0.129970

Dilate 0  px: 0.020524
Dilate 5  px: 0.048741
Dilate 10 px: 0.099067
Dilate 15 px: 0.133474
Dilate 17 px: 0.145692 ← best
Dilate 20 px: 0.169555
Dilate 25 px: 0.194207

Dilate 0  px: 0.030481
Dilate 5  px: 0.035389
Dilate 10 px: 0.043582
Dilate 15 px: 0.049522
Dilate 17 px: 0.052041 ← best
Dilate 20 px: 0.057611
Dilate 25 px: 0.063239

Dilate 0  px: 0.062817
Dilate 5  px: 0.144570
Dilate 10 px: 0.227062
Dilate 15 px: 0.269953
Dilate 17 px: 0.283450 ← best
Dilate 20 px: 0.306546
Dilate 25 px: 0.327387

Dilate 0  px: 0.007488
Dilate 5  px: 0.011648
Dilate 10 px: 0.020916
Dilate 15 px: 0.027997
Dilate 17 px: 0.031000 ← best
Dilate 20 px: 0.037619
Dilate 25 px: 0.044665


2:
finetune vgg training:
Evaluating on 320 held-out validation samples...

=== Validation set summary (n=320) ===
PSNR: mean=29.41dB  std=4.63  min=12.07  max=45.52
SSIM: mean=0.7777  std=0.1552  min=0.2860  max=0.9728

Worst 5 samples by PSNR:
  002982.npy: PSNR=12.07dB, SSIM=0.2902
  001908.npy: PSNR=19.12dB, SSIM=0.3502
  001387.npy: PSNR=20.61dB, SSIM=0.7993
  000108.npy: PSNR=20.61dB, SSIM=0.6733
  001386.npy: PSNR=20.72dB, SSIM=0.8054

Best 5 samples by PSNR:
  001082.npy: PSNR=39.95dB, SSIM=0.9433
  002217.npy: PSNR=40.27dB, SSIM=0.9581
  003117.npy: PSNR=42.52dB, SSIM=0.9615
  002225.npy: PSNR=43.24dB, SSIM=0.9641
  001827.npy: PSNR=45.52dB, SSIM=0.9644

=== Named Difficulty Samples ===
  000218.npy: PSNR=31.90dB, SSIM=0.8428
  000352.npy: PSNR=12.17dB, SSIM=0.2971
  000425.npy: PSNR=34.48dB, SSIM=0.8862

=============================================
        FINAL METROLOGY SCORECARD
=============================================
Mean CD Bias:   -16.969 px
Mean LER Error: 23.623 px
Mean LWR Error: 27.193 px
Slope Fidelity: 76.6 %
=============================================

=== Processing Test Set (no GT) ===
[SEMTestDataset] 400 test samples found.

Done. Detailed visuals and summary in ./eval_outputs_cd_v2/



now doing CD Baseline Test:









test completed 



baseline_v2 evaluation:
Evaluating on 320 held-out validation samples...

=== Validation set summary (n=320) ===
PSNR: mean=29.54dB  std=4.64  min=12.50  max=45.66
SSIM: mean=0.7873  std=0.1473  min=0.3008  max=0.9796

Worst 5 samples by PSNR:
  002982.npy: PSNR=12.50dB, SSIM=0.3140
  001908.npy: PSNR=19.46dB, SSIM=0.4064
  000108.npy: PSNR=20.59dB, SSIM=0.6761
  001387.npy: PSNR=20.69dB, SSIM=0.8045
  001315.npy: PSNR=20.76dB, SSIM=0.6604

Best 5 samples by PSNR:
  001082.npy: PSNR=40.12dB, SSIM=0.9467
  002217.npy: PSNR=40.69dB, SSIM=0.9628
  003117.npy: PSNR=42.38dB, SSIM=0.9608
  002225.npy: PSNR=43.52dB, SSIM=0.9680
  001827.npy: PSNR=45.66dB, SSIM=0.9652

=== Named Difficulty Samples ===
  000218.npy: PSNR=32.22dB, SSIM=0.8600
  000352.npy: PSNR=12.60dB, SSIM=0.3213
  000425.npy: PSNR=34.79dB, SSIM=0.8933

=============================================
        FINAL METROLOGY SCORECARD
=============================================
Mean CD Bias:   -14.459 px
Mean LER Error: 22.811 px
Mean LWR Error: 25.227 px
Slope Fidelity: 76.6 %
=============================================








### Chapter 11: TTA and IntensityProfileLoss (Claude Code session)

**The Ask:** Cross-checked this file against a scale-up run in progress
(dim=96/num_blocks=4, 100 epochs, queued after the user noted "even after
scaling the increases in accuracy was very low"). Chapter 9's scale_v1
result (dim=128/num_blocks=4: +0.5dB PSNR, CD bias unchanged) already
answered that question -- the queued run was retesting an answered
hypothesis and was killed before completion. Picked two of the
"Tools Explored But Not Tested" list (Chapter 10) instead: Test-Time
Augmentation and IntensityProfileLoss.

**Finding: IntensityProfileLoss did not actually exist.** `finetune_profile.py`
declared `use_profile_loss`/`num_profiles`/`profile_width` config keys that
were never read anywhere in the script -- the loss it actually built was
plain SSIM. Implemented for real in `losses/intensity_profile_loss.py`.

**Experiment 1 -- TTA (8-view D4 dihedral average), zero retraining, against
baseline_v2:**

| Metric | Plain | TTA (8-view avg) | Delta |
|--------|-------|-------------------|-------|
| PSNR | 29.54 | 29.77 | +0.24 dB |
| SSIM | 0.7873 | 0.7931 | +0.006 |
| CD Bias | -14.454 | -15.213 | **-0.76 px (worse)** |
| LER Error | 22.811 | 24.223 | +1.41 px (worse) |
| LWR Error | 25.218 | 26.831 | +1.61 px (worse) |
| Slope Fidelity | 76.6% | 74.1% | -2.5 pts (worse) |

**Finding: TTA helps generic PSNR/SSIM slightly (expected -- averaging
reduces variance) but makes the metrology metrics worse.** The edge-slope
distortion isn't asymmetric enough under D4 transforms for averaging to
cancel it; averaging just smooths the edge, which is the wrong direction
for a slope-shape problem. Not adopted.

**Experiment 2 -- IntensityProfileLoss fine-tune from baseline_v2.**
Design deliberately different from the failed ThresholdedCDLoss (Chapter 3):
masks by a GT-only, detached edge map (fixed, not influenced by the
model's own predictions), so there's no "spam edges everywhere" degenerate
minimum available -- the model can only reduce the loss by matching the
actual gradient-magnitude VALUE at true edge locations, not by changing
which pixels get penalized. `profile_weight=1.0` (conservative first pass,
given 6/6 prior loss additions on this branch made things worse or
collapsed). Smoke-tested for stability (10 real steps, no blowup) before
committing to the full 15-epoch run.

| Metric | baseline_v2 | profile_v1 (weight=1.0) | Delta |
|--------|-------------|--------------------------|-------|
| PSNR | 29.54 | 29.21 | -0.33 dB |
| SSIM | 0.7873 | 0.7799 | -0.0074 |
| **CD Bias** | **-14.454** | **-8.594** | **+5.86 px closer to zero (40% reduction)** |
| **LER Error** | 22.811 | 18.670 | **-4.14 px (18% better)** |
| **LWR Error** | 25.218 | 19.421 | **-5.80 px (23% better)** |
| **Slope Fidelity** | 76.6% | 82.6% | **+6.0 pts** |

**Finding: this is the first approach (out of 7 tried across both this
project's history and a parallel branch's SwinIR/AirNet comparison) that
actually moves CD Bias substantially in the right direction without
collapsing PSNR/SSIM or flipping the bias sign.** Slope Fidelity improving
alongside CD Bias/LER/LWR is a direct confirmation of the Chapter 8
diagnosis -- the problem really is edge-slope shape, not edge position,
and a loss that targets slope shape at fixed GT locations (rather than
predicted-edge position, which is what every prior loss-based attempt
used) is what finally moves it. Training was stable throughout (no
collapse, val_loss still gently improving at epoch 15) -- profile_weight=1.0
was a deliberately conservative starting point, not a tuned optimum.

**Next:** testing a higher profile_weight to see how far CD Bias can be
pushed and where the PSNR/SSIM cost becomes unacceptable.




Tier 0 — known dead ends, don't repeat these
Seven things already tried, all flat-to-negative on PSNR/SSIM or the metrology metrics: CD-distance-transform loss (both variants), MS-SSIM loss (both branches), post-processing dilation correction, pure architecture scaling (2→3→4 blocks, 64→128 dim), TTA, hard-sample oversampling (naive version). Worth stating plainly so effort doesn't get re-spent rediscovering these.

Tier 1 — cheap, near-term, direct extensions of what's already working
Finish the profile_weight sweep (in progress). Try 0.5, 1.0 (done), 2.0, 3.0 (running) — find where the PSNR-cost/CD-bias-gain trade-off stops being worth it. This is the only lever so far that's moved the real metric in the right direction.
Rebalance the base loss around it — right now edge_weight=0.5 (Sobel) sits alongside the new profile loss, but Chapter 6 already showed Sobel edge loss is neutral (doesn't help or hurt). Dropping it to 0 and letting profile_weight carry that role might free up capacity without cost.
Ensembling independently trained checkpoints (not TTA — genuine model diversity, not multi-view of one model). Average predictions from baseline_v2 + profile_v1 + a third run with a different random seed. This is a different mechanism from TTA (which failed) because the errors between differently-trained models are less correlated than between differently-oriented views of the same model — classic ensembling, usually gives a real, reliable, small PSNR/SSIM bump. Cheap: no new training beyond what already exists, just an averaging script.
Tier 2 — the highest-leverage genuinely untried direction
Degradation-diversity augmentation. Still untried on either branch, despite being flagged repeatedly (by me, independently by the deepseek log's own Priority 2, and explicitly by the KLA spec itself for OOD generalization). Concretely: synthesize extra noise/blur severity variation on top of the existing GT during training (not just the one fixed degradation profile in the 3200 pairs), so the model has to generalize across a wider distribution instead of memorizing one. This is the one lever that hasn't been tested at all — everything else has been loss or architecture, this is data.
Two-stage refinement, done correctly this time — not a naive cascade (already ruled out earlier for lacking intermediate supervision), but MPRNet-style: a second small network takes (LR input, stage-1 output) and refines it, trained end-to-end against the same final GT, with the profile loss applied at both stages. Real precedent (MPRNet's Supervised Attention Module) for this working where flat scaling didn't.
Tier 3 — bigger investment, real trade-offs to accept going in
LPIPS as a training-time loss term (not just a reported metric) — pushes toward perceptual realism specifically. Real cost: will likely trade a bit of PSNR for it, same perception-distortion trade-off discussed earlier. Worth it only if perceptual quality is weighted meaningfully in whatever's actually being graded.
A generative/diffusion pass, hard-samples-only — for the specific information-theoretic-ceiling samples (002982.npy-type — genuinely unpredictable fine texture), a sampling-based model could produce a plausible texture instead of a blurred average. Real cost: inference-time latency (fights the speed requirement) and it won't raise PSNR against one fixed GT realization, only perceptual scores. I'd only reach for this if the grading rewards realism over exact pixel match.






p3.0 result:
Loaded checkpoint: epoch 15, val_psnr=26.75dB
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:207: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:222: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: /home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/lpips/weights/v0.1/alex.pth
[SEMPairDataset] 3200 paired samples found.

Evaluating on 320 held-out validation samples...

=== Validation set summary (n=320) ===
PSNR: mean=28.25dB  std=4.51  min=11.43  max=43.75
SSIM: mean=0.7385  std=0.1575  min=0.2523  max=0.9646

Worst 5 samples by PSNR:
  002982.npy: PSNR=11.43dB, SSIM=0.2893
  000641.npy: PSNR=19.56dB, SSIM=0.3246
  001908.npy: PSNR=19.65dB, SSIM=0.4925
  000108.npy: PSNR=19.82dB, SSIM=0.6621
  001334.npy: PSNR=19.92dB, SSIM=0.3508

Best 5 samples by PSNR:
  000823.npy: PSNR=37.87dB, SSIM=0.9229
  002460.npy: PSNR=38.09dB, SSIM=0.9311
  002217.npy: PSNR=39.54dB, SSIM=0.9494
  002225.npy: PSNR=42.07dB, SSIM=0.9481
  001827.npy: PSNR=43.75dB, SSIM=0.9506

=== Named Difficulty Samples ===
  000218.npy: PSNR=31.21dB, SSIM=0.8021
  000352.npy: PSNR=11.52dB, SSIM=0.2962
  000425.npy: PSNR=33.45dB, SSIM=0.8786

=============================================
        FINAL METROLOGY SCORECARD
=============================================
Mean CD Bias:   0.789 px
Mean LER Error: 16.090 px
Mean LWR Error: 17.027 px
Slope Fidelity: 88.4 %
=============================================


v1 version:


Evaluating on 320 held-out validation samples...

=== Validation set summary (n=320) ===
PSNR: mean=29.21dB  std=4.58  min=12.24  max=44.55
SSIM: mean=0.7799  std=0.1458  min=0.3219  max=0.9741

Worst 5 samples by PSNR:
  002982.npy: PSNR=12.24dB, SSIM=0.3283
  001908.npy: PSNR=19.68dB, SSIM=0.4526
  000108.npy: PSNR=20.35dB, SSIM=0.6764
  001387.npy: PSNR=20.47dB, SSIM=0.7998
  001315.npy: PSNR=20.51dB, SSIM=0.6552

Best 5 samples by PSNR:
  002506.npy: PSNR=38.95dB, SSIM=0.9480
  003117.npy: PSNR=39.78dB, SSIM=0.9226
  002217.npy: PSNR=40.40dB, SSIM=0.9615
  002225.npy: PSNR=42.73dB, SSIM=0.9573
  001827.npy: PSNR=44.55dB, SSIM=0.9557

=== Named Difficulty Samples ===
  000218.npy: PSNR=31.92dB, SSIM=0.8503
  000352.npy: PSNR=12.34dB, SSIM=0.3372
  000425.npy: PSNR=34.49dB, SSIM=0.9011

=============================================
        FINAL METROLOGY SCORECARD
=============================================
Mean CD Bias:   -8.595 px
Mean LER Error: 18.669 px
Mean LWR Error: 19.420 px
Slope Fidelity: 82.6 %
=============================================



p2p0:



Evaluating on 320 held-out validation samples...

=== Validation set summary (n=320) ===
PSNR: mean=28.80dB  std=4.54  min=12.01  max=44.24
SSIM: mean=0.7631  std=0.1490  min=0.3009  max=0.9694

Worst 5 samples by PSNR:
  002982.npy: PSNR=12.01dB, SSIM=0.3213
  001908.npy: PSNR=19.98dB, SSIM=0.5056
  000108.npy: PSNR=20.09dB, SSIM=0.6703
  001387.npy: PSNR=20.19dB, SSIM=0.7908
  001315.npy: PSNR=20.24dB, SSIM=0.6500

Best 5 samples by PSNR:
  003117.npy: PSNR=38.47dB, SSIM=0.8973
  002460.npy: PSNR=38.57dB, SSIM=0.9382
  002217.npy: PSNR=40.17dB, SSIM=0.9586
  002225.npy: PSNR=42.47dB, SSIM=0.9499
  001827.npy: PSNR=44.24dB, SSIM=0.9482

=== Named Difficulty Samples ===
  000218.npy: PSNR=31.47dB, SSIM=0.8260
  000352.npy: PSNR=12.11dB, SSIM=0.3304
  000425.npy: PSNR=34.29dB, SSIM=0.8982

=============================================
        FINAL METROLOGY SCORECARD
=============================================
Mean CD Bias:   -3.861 px
Mean LER Error: 17.776 px
Mean LWR Error: 18.272 px
Slope Fidelity: 84.7 %
=============================================

=== Processing Test Set (no GT) ===
[SEMTestDataset] 400 test samples found.

Done. Detailed visuals and summary in ./eval_outputs_checkpoints_profile_w2p0/



w_edhe3p0:edge=0


Loaded checkpoint: epoch 15, val_psnr=26.00dB
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:207: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:222: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: /home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/lpips/weights/v0.1/alex.pth
[SEMPairDataset] 3200 paired samples found.

Evaluating on 320 held-out validation samples...

=== Validation set summary (n=320) ===
PSNR: mean=27.64dB  std=4.77  min=10.05  max=43.95
SSIM: mean=0.7122  std=0.1630  min=0.2390  max=0.9672

Worst 5 samples by PSNR:
  002982.npy: PSNR=10.05dB, SSIM=0.2999
  000641.npy: PSNR=18.64dB, SSIM=0.3603
  000108.npy: PSNR=18.80dB, SSIM=0.6278
  001334.npy: PSNR=18.85dB, SSIM=0.3612
  001387.npy: PSNR=18.90dB, SSIM=0.7527

Best 5 samples by PSNR:
  000823.npy: PSNR=37.98dB, SSIM=0.9261
  002460.npy: PSNR=38.31dB, SSIM=0.9335
  002217.npy: PSNR=39.63dB, SSIM=0.9503
  002225.npy: PSNR=42.00dB, SSIM=0.9500
  001827.npy: PSNR=43.95dB, SSIM=0.9504

=== Named Difficulty Samples ===
  000218.npy: PSNR=30.15dB, SSIM=0.7416
  000352.npy: PSNR=10.14dB, SSIM=0.3064
  000425.npy: PSNR=33.93dB, SSIM=0.8883

=============================================
        FINAL METROLOGY SCORECARD
=============================================
Mean CD Bias:   5.613 px
Mean LER Error: 16.534 px
Mean LWR Error: 17.839 px
Slope Fidelity: 89.6 %
=============================================

=== Processing Test Set (no GT) ===
[SEMTestDataset] 400 test samples found.

Done. Detailed visuals and summary in ./eval_outputs_checkpoints_profile_w_edge3p0/





Evaluating on 320 held-out validation samples...

=== Validation set summary (n=320) ===
PSNR: mean=28.63dB  std=4.51  min=11.95  max=43.96
SSIM: mean=0.7590  std=0.1488  min=0.3037  max=0.9692

Worst 5 samples by PSNR:
  002982.npy: PSNR=11.95dB, SSIM=0.3254
  000108.npy: PSNR=20.00dB, SSIM=0.6698
  001908.npy: PSNR=20.00dB, SSIM=0.5186
  001387.npy: PSNR=20.13dB, SSIM=0.7933
  001315.npy: PSNR=20.20dB, SSIM=0.6533

Best 5 samples by PSNR:
  002506.npy: PSNR=38.35dB, SSIM=0.9371
  002460.npy: PSNR=38.36dB, SSIM=0.9371
  002217.npy: PSNR=39.99dB, SSIM=0.9570
  002225.npy: PSNR=42.22dB, SSIM=0.9508
  001827.npy: PSNR=43.96dB, SSIM=0.9500

=== Named Difficulty Samples ===
  000218.npy: PSNR=31.44dB, SSIM=0.8252
  000352.npy: PSNR=12.04dB, SSIM=0.3335
  000425.npy: PSNR=34.26dB, SSIM=0.8954

=============================================
        FINAL METROLOGY SCORECARD
=============================================
Mean CD Bias:   -1.718 px
Mean LER Error: 16.693 px
Mean LWR Error: 16.821 px
Slope Fidelity: 88.1 %
=============================================

=== Processing Test Set (no GT) ===
[SEMTestDataset] 400 test samples found.

Done. Detailed visuals and summary in ./eval_outputs_checkpoints_profile_w_lr3p0/






ensemble testing:

Evaluating ensemble on 320 held-out validation samples...

=== Ensemble validation summary (n=320) ===
PSNR: mean=29.24dB  std=4.61  min=12.28  max=45.04
SSIM: mean=0.7798  std=0.1458  min=0.3205  max=0.9769

Worst 5 samples by PSNR:
  002982.npy: PSNR=12.28dB, SSIM=0.3233
  001908.npy: PSNR=19.84dB, SSIM=0.4745
  000108.npy: PSNR=20.35dB, SSIM=0.6762
  001387.npy: PSNR=20.48dB, SSIM=0.8014
  001315.npy: PSNR=20.54dB, SSIM=0.6604

Best 5 samples by PSNR:
  001082.npy: PSNR=39.46dB, SSIM=0.9325
  003117.npy: PSNR=40.02dB, SSIM=0.9280
  002217.npy: PSNR=40.63dB, SSIM=0.9630
  002225.npy: PSNR=43.13dB, SSIM=0.9631
  001827.npy: PSNR=45.04dB, SSIM=0.9606

=== Named Difficulty Samples ===
  000218.npy: PSNR=32.03dB, SSIM=0.8527
  000352.npy: PSNR=12.38dB, SSIM=0.3312
  000425.npy: PSNR=34.80dB, SSIM=0.9030

=============================================
     ENSEMBLE METROLOGY SCORECARD
=============================================
Mean CD Bias:   -8.524 px
Mean LER Error: 19.028 px
Mean LWR Error: 20.434 px
Slope Fidelity: 81.2 %
=============================================

=== Processing Test Set (no GT) ===
[SEMTestDataset] 400 test samples found.

Done. Detailed visuals and summary in ./eval_outputs_ensemble_checkpoints_baseline_v2_checkpoints_profile_w_lr3p0/





=== Ensemble validation summary (n=320) ===
PSNR: mean=28.96dB  std=4.55  min=12.11  max=44.30
SSIM: mean=0.7715  std=0.1468  min=0.3139  max=0.9726

Worst 5 samples by PSNR:
  002982.npy: PSNR=12.11dB, SSIM=0.3278
  001908.npy: PSNR=19.87dB, SSIM=0.4894
  000108.npy: PSNR=20.19dB, SSIM=0.6741
  001387.npy: PSNR=20.32dB, SSIM=0.7976
  001315.npy: PSNR=20.37dB, SSIM=0.6560

Best 5 samples by PSNR:
  002460.npy: PSNR=38.55dB, SSIM=0.9394
  002506.npy: PSNR=38.74dB, SSIM=0.9443
  002217.npy: PSNR=40.23dB, SSIM=0.9597
  002225.npy: PSNR=42.53dB, SSIM=0.9550
  001827.npy: PSNR=44.30dB, SSIM=0.9535

=== Named Difficulty Samples ===
  000218.npy: PSNR=31.72dB, SSIM=0.8404
  000352.npy: PSNR=12.21dB, SSIM=0.3363
  000425.npy: PSNR=34.40dB, SSIM=0.8994

=============================================
     ENSEMBLE METROLOGY SCORECARD
=============================================
Mean CD Bias:   -5.299 px
Mean LER Error: 17.687 px
Mean LWR Error: 17.467 px
Slope Fidelity: 85.1 %
=============================================

=== Processing Test Set (no GT) ===
[SEMTestDataset] 400 test samples found.

Done. Detailed visuals and summary in ./eval_outputs_ensemble_checkpoints_profile_v1_checkpoints_profile_w_lr3p0/