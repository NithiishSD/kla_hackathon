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