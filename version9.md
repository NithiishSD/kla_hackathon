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