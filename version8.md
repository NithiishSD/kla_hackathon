

Again rerunnning the baseline model for increasing teh metrix with addition to the new metrix
finetunning:
esuming from ./checkpoints_baseline/best_model.pt (epoch 1, val_psnr=27.82dB)
=== baseline FINE-TUNE ===
Config: {'dim': 64, 'num_blocks': 2, 'batch_size': 4, 'target_batch_size': 32, 'scale_factor': 2, 'edge_weight': 0.5, 'freq_weight': 0.0, 'ssim_weight': 0.0, 'lr': 0.0002, 'weight_decay': 0.0001, 'scheduler_patience': 3, 'early_stop_patience': 7, 'num_workers': 8, 'num_epochs': 50, 'use_compile': False, 'checkpoint_dir': './checkpoints_baseline_v2'}
[SEMPairDataset] 3200 paired samples found.
[SEMTestDataset] 400 test samples found.
Loaded Phase 2 weights. Missing 'metrology_gain' initialized to default (1.0).
--- [Modern GPU] Using bfloat16 on NVIDIA GeForce RTX 4050 Laptop GPU ---
--- [Compute] DataLoader batch_size=4, target_batch_size=32 -> accum_steps=8 (actual effective batch = 32) ---

--- Epoch 1 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 1/50 [108.7s] [LR: 2.00e-04] train_loss=0.0793 val_loss=0.0781 val_psnr=27.84dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 2 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 2/50 [107.9s] [LR: 2.00e-04] train_loss=0.0790 val_loss=0.0783 val_psnr=27.83dB
  -> val_loss did not improve (1/7)

--- Epoch 3 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 3/50 [108.0s] [LR: 2.00e-04] train_loss=0.0790 val_loss=0.0782 val_psnr=27.82dB
  -> val_loss did not improve (2/7)

--- Epoch 4 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 4/50 [108.2s] [LR: 2.00e-04] train_loss=0.0791 val_loss=0.0780 val_psnr=27.85dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 5 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 5/50 [107.9s] [LR: 2.00e-04] train_loss=0.0789 val_loss=0.0782 val_psnr=27.87dB
  -> val_loss did not improve (1/7)

--- Epoch 6 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 6/50 [107.9s] [LR: 2.00e-04] train_loss=0.0788 val_loss=0.0780 val_psnr=27.86dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 7 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 7/50 [107.9s] [LR: 2.00e-04] train_loss=0.0789 val_loss=0.0781 val_psnr=27.81dB
  -> val_loss did not improve (1/7)

--- Epoch 8 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 8/50 [107.8s] [LR: 2.00e-04] train_loss=0.0789 val_loss=0.0778 val_psnr=27.86dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 9 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 9/50 [108.0s] [LR: 2.00e-04] train_loss=0.0788 val_loss=0.0779 val_psnr=27.83dB
  -> val_loss did not improve (1/7)

--- Epoch 10 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 10/50 [108.1s] [LR: 2.00e-04] train_loss=0.0786 val_loss=0.0780 val_psnr=27.83dB
  -> val_loss did not improve (2/7)

--- Epoch 11 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 11/50 [108.0s] [LR: 2.00e-04] train_loss=0.0787 val_loss=0.0778 val_psnr=27.88dB
  -> val_loss did not improve (3/7)

--- Epoch 12 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 12/50 [108.0s] [LR: 2.00e-04] train_loss=0.0787 val_loss=0.0777 val_psnr=27.88dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 13 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 13/50 [107.9s] [LR: 2.00e-04] train_loss=0.0786 val_loss=0.0777 val_psnr=27.90dB
  -> val_loss did not improve (1/7)

--- Epoch 14 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 14/50 [107.9s] [LR: 2.00e-04] train_loss=0.0786 val_loss=0.0775 val_psnr=27.89dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 15 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 15/50 [108.0s] [LR: 2.00e-04] train_loss=0.0785 val_loss=0.0778 val_psnr=27.91dB
  -> val_loss did not improve (1/7)

--- Epoch 16 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 16/50 [108.1s] [LR: 2.00e-04] train_loss=0.0785 val_loss=0.0776 val_psnr=27.88dB
  -> val_loss did not improve (2/7)

--- Epoch 17 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 17/50 [107.8s] [LR: 2.00e-04] train_loss=0.0785 val_loss=0.0775 val_psnr=27.88dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 18 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 18/50 [107.8s] [LR: 2.00e-04] train_loss=0.0784 val_loss=0.0775 val_psnr=27.91dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 19 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 19/50 [108.0s] [LR: 2.00e-04] train_loss=0.0784 val_loss=0.0773 val_psnr=27.91dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 20 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 20/50 [108.1s] [LR: 2.00e-04] train_loss=0.0783 val_loss=0.0774 val_psnr=27.89dB
  -> val_loss did not improve (1/7)

--- Epoch 21 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 21/50 [108.1s] [LR: 2.00e-04] train_loss=0.0782 val_loss=0.0775 val_psnr=27.90dB
  -> val_loss did not improve (2/7)

--- Epoch 22 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 22/50 [107.9s] [LR: 2.00e-04] train_loss=0.0783 val_loss=0.0773 val_psnr=27.91dB
  -> val_loss did not improve (3/7)

--- Epoch 23 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 23/50 [108.0s] [LR REDUCED: 2.00e-04 -> 1.00e-04] train_loss=0.0783 val_loss=0.0773 val_psnr=27.91dB
  -> val_loss did not improve (4/7)

--- Epoch 24 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 24/50 [107.9s] [LR: 1.00e-04] train_loss=0.0779 val_loss=0.0771 val_psnr=27.92dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 25 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 25/50 [108.2s] [LR: 1.00e-04] train_loss=0.0778 val_loss=0.0770 val_psnr=27.94dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 26 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 26/50 [108.2s] [LR: 1.00e-04] train_loss=0.0778 val_loss=0.0770 val_psnr=27.93dB
  -> val_loss did not improve (1/7)

--- Epoch 27 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 27/50 [108.1s] [LR: 1.00e-04] train_loss=0.0778 val_loss=0.0770 val_psnr=27.95dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 28 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 28/50 [108.1s] [LR: 1.00e-04] train_loss=0.0777 val_loss=0.0769 val_psnr=27.94dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 29 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 29/50 [108.1s] [LR: 1.00e-04] train_loss=0.0777 val_loss=0.0770 val_psnr=27.94dB
  -> val_loss did not improve (1/7)

--- Epoch 30 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 30/50 [108.1s] [LR: 1.00e-04] train_loss=0.0777 val_loss=0.0769 val_psnr=27.94dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 31 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 31/50 [108.0s] [LR: 1.00e-04] train_loss=0.0777 val_loss=0.0770 val_psnr=27.94dB
  -> val_loss did not improve (1/7)

--- Epoch 32 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 32/50 [108.1s] [LR: 1.00e-04] train_loss=0.0777 val_loss=0.0770 val_psnr=27.93dB
  -> val_loss did not improve (2/7)

--- Epoch 33 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 33/50 [107.9s] [LR: 1.00e-04] train_loss=0.0776 val_loss=0.0770 val_psnr=27.94dB
  -> val_loss did not improve (3/7)

--- Epoch 34 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 34/50 [107.8s] [LR REDUCED: 1.00e-04 -> 5.00e-05] train_loss=0.0777 val_loss=0.0770 val_psnr=27.93dB
  -> val_loss did not improve (4/7)

--- Epoch 35 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 35/50 [107.8s] [LR: 5.00e-05] train_loss=0.0774 val_loss=0.0768 val_psnr=27.94dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 36 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 36/50 [107.9s] [LR: 5.00e-05] train_loss=0.0775 val_loss=0.0768 val_psnr=27.95dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 37 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 37/50 [107.8s] [LR: 5.00e-05] train_loss=0.0774 val_loss=0.0769 val_psnr=27.96dB
  -> val_loss did not improve (1/7)

--- Epoch 38 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 38/50 [107.8s] [LR: 5.00e-05] train_loss=0.0773 val_loss=0.0767 val_psnr=27.95dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 39 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 39/50 [107.8s] [LR: 5.00e-05] train_loss=0.0775 val_loss=0.0768 val_psnr=27.96dB
  -> val_loss did not improve (1/7)

--- Epoch 40 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 40/50 [108.0s] [LR: 5.00e-05] train_loss=0.0775 val_loss=0.0767 val_psnr=27.95dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 41 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 41/50 [108.0s] [LR: 5.00e-05] train_loss=0.0774 val_loss=0.0768 val_psnr=27.92dB
  -> val_loss did not improve (1/7)

--- Epoch 42 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 42/50 [108.0s] [LR: 5.00e-05] train_loss=0.0774 val_loss=0.0767 val_psnr=27.96dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 43 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 43/50 [107.9s] [LR: 5.00e-05] train_loss=0.0774 val_loss=0.0767 val_psnr=27.97dB
  -> val_loss did not improve (1/7)

--- Epoch 44 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 44/50 [107.8s] [LR: 5.00e-05] train_loss=0.0773 val_loss=0.0767 val_psnr=27.95dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 45 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 45/50 [107.8s] [LR: 5.00e-05] train_loss=0.0773 val_loss=0.0767 val_psnr=27.95dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt

--- Epoch 46 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 46/50 [107.7s] [LR: 5.00e-05] train_loss=0.0773 val_loss=0.0767 val_psnr=27.96dB
  -> val_loss did not improve (1/7)

--- Epoch 47 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 47/50 [107.8s] [LR: 5.00e-05] train_loss=0.0773 val_loss=0.0767 val_psnr=27.96dB
  -> val_loss did not improve (2/7)

--- Epoch 48 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 48/50 [107.8s] [LR: 5.00e-05] train_loss=0.0773 val_loss=0.0767 val_psnr=27.96dB
  -> val_loss did not improve (3/7)

--- Epoch 49 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 49/50 [107.7s] [LR REDUCED: 5.00e-05 -> 2.50e-05] train_loss=0.0772 val_loss=0.0768 val_psnr=27.97dB
  -> val_loss did not improve (4/7)

--- Epoch 50 | Active SSIM Weight: 0.0000 ---
[SSIM-FT] Epoch 50/50 [107.8s] [LR: 2.50e-05] train_loss=0.0774 val_loss=0.0766 val_psnr=27.96dB
  -> new best val_loss, saved to ./checkpoints_baseline_v2/best_model.pt
Fine-tune finished. Compare against Phase 2 and baseline checkpoints using evaluate.py -- and specifically look at the worst-case samples (like 000352/002982) for whether SSIM loss actually reduced the blur/melting.
(/home/nithiish/Documents/kla_hackathon/.conda) nithiish@nithiish-pc:~/Documents/kla_hackathon$ python evaluation.py ./data ./checkpoints_baseline_v2/best_model.pt
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:207: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:222: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: /home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/lpips/weights/v0.1/alex.pth
[SEMPairDataset] 3200 paired samples found.

Evaluating on 320 validation samples...

=============================================
        FINAL METROLOGY SCORECARD
=============================================
PSNR:           29.54 dB
SSIM:           0.7873
LPIPS:          0.2411 
---------------------------------------------
Mean CD Bias:   -14.459 px
Mean LER Error: 22.811 px
Mean LWR Error: 25.227 px
Slope Fidelity: 76.6 %
=============================================

Done. All outputs and metrology stats in ./eval_outputs_baseline_v2/


step1:

focusing on cd reduction :

Run analyze_failures.py on sample 002982. Understand the spatial pattern of failure.

Implement ThresholdedCDLoss in a new file. Test it on a single batch to verify it runs and produces reasonable values.

Create ./hard_val_set/ with your 20 worst samples.

Launch Phase 4 training with the config above.

Go for coffee while epoch 1 runs. When it finishes, check the validation CD Bias. If it moved from -14.7 toward zero, you've won. Everything else is tuning.


.pt
Resuming from ./checkpoints_baseline_v2/best_model.pt (epoch 50, val_psnr=27.96dB)
=== PHASE 4: CD BIAS EXTERMINATION ===
Config: {'dim': 64, 'num_blocks': 2, 'batch_size': 4, 'target_batch_size': 32, 'scale_factor': 2, 'edge_weight': 2.5, 'freq_weight': 0.15, 'ssim_weight': 0.3, 'lr': 2.5e-05, 'weight_decay': 1e-06, 'scheduler_patience': 3, 'early_stop_patience': 7, 'num_workers': 8, 'num_epochs': 30, 'use_compile': False, 'checkpoint_dir': './checkpoints_cd_v1', 'use_cd_loss': True, 'cd_edge_percentile': 90.0, 'cd_max_distance': 30.0}
[SEMPairDataset] 3200 paired samples found.
[SEMTestDataset] 400 test samples found.
Loaded Phase 2 weights. Missing 'metrology_gain' initialized to default (1.0).
--- [Modern GPU] Using bfloat16 on NVIDIA GeForce RTX 4050 Laptop GPU ---
--- [Compute] DataLoader batch_size=4, target_batch_size=32 -> accum_steps=8 (actual effective batch = 32) ---
[CD-EXTERM] Epoch 1/30 [118.8s] [LR: 2.50e-05] train_loss=0.3669 val_loss=0.3766 val_psnr=21.27dB
  -> new best val_loss, saved to ./checkpoints_cd_v1/best_model.pt
[CD-EXTERM] Epoch 2/30 [118.2s] [LR: 2.50e-05] train_loss=0.3547 val_loss=0.3413 val_psnr=22.10dB
  -> new best val_loss, saved to ./checkpoints_cd_v1/best_model.pt
[CD-EXTERM] Epoch 3/30 [118.3s] [LR: 2.50e-05] train_loss=0.3369 val_loss=0.3338 val_psnr=22.37dB
  -> new best val_loss, saved to ./checkpoints_cd_v1/best_model.pt
[CD-EXTERM] Epoch 4/30 [118.5s] [LR: 2.50e-05] train_loss=0.3336 val_loss=0.3325 val_psnr=22.49dB
  -> new best val_loss, saved to ./checkpoints_cd_v1/best_model.pt
[CD-EXTERM] Epoch 5/30 [118.7s] [LR: 2.50e-05] train_loss=0.3315 val_loss=0.3307 val_psnr=22.65dB
  -> new best val_loss, saved to ./checkpoints_cd_v1/best_model.pt
[CD-EXTERM] Epoch 6/30 [118.4s] [LR: 2.50e-05] train_loss=0.3298 val_loss=0.3280 val_psnr=22.88dB
  -> new best val_loss, saved to ./checkpoints_cd_v1/best_model.pt
[CD-EXTERM] Epoch 7/30 [127.9s] [LR: 2.50e-05] train_loss=0.3286 val_loss=0.3267 val_psnr=23.00dB
  -> new best val_loss, saved to ./checkpoints_cd_v1/best_model.pt
[CD-EXTERM] Epoch 8/30 [118.6s] [LR: 2.50e-05] train_loss=0.3274 val_loss=0.3260 val_psnr=23.10dB
  -> new best val_loss, saved to ./checkpoints_cd_v1/best_model.pt
[CD-EXTERM] Epoch 9/30 [118.1s] [LR: 2.50e-05] train_loss=0.3266 val_loss=0.3241 val_psnr=23.28dB
  -> new best val_loss, saved to ./checkpoints_cd_v1/best_model.pt
[CD-EXTERM] Epoch 10/30 [118.1s] [LR: 2.50e-05] train_loss=0.3260 val_loss=0.3248 val_psnr=23.30dB
  -> val_loss did not improve (1/7)
[CD-EXTERM] Epoch 11/30 [118.1s] [LR: 2.50e-05] train_loss=0.3258 val_loss=0.3242 val_psnr=23.40dB
  -> val_loss did not improve (2/7)
[CD-EXTERM] Epoch 12/30 [118.1s] [LR: 2.50e-05] train_loss=0.3253 val_loss=0.3230 val_psnr=23.54dB
  -> new best val_loss, saved to ./checkpoints_cd_v1/best_model.pt
[CD-EXTERM] Epoch 13/30 [118.1s] [LR: 2.50e-05] train_loss=0.3255 val_loss=0.3256 val_psnr=23.43dB
  -> val_loss did not improve (1/7)
[CD-EXTERM] Epoch 14/30 [118.1s] [LR: 2.50e-05] train_loss=0.3262 val_loss=0.3240 val_psnr=23.57dB
  -> val_loss did not improve (2/7)
[CD-EXTERM] Epoch 15/30 [118.1s] [LR: 2.50e-05] train_loss=0.3260 val_loss=0.3255 val_psnr=23.53dB
  -> val_loss did not improve (3/7)
[CD-EXTERM] Epoch 16/30 [118.1s] [LR REDUCED: 2.50e-05 -> 1.25e-05] train_loss=0.3260 val_loss=0.3234 val_psnr=23.73dB
  -> val_loss did not improve (4/7)
[CD-EXTERM] Epoch 17/30 [118.1s] [LR: 1.25e-05] train_loss=0.3265 val_loss=0.3248 val_psnr=23.63dB
  -> val_loss did not improve (5/7)
[CD-EXTERM] Epoch 18/30 [118.1s] [LR: 1.25e-05] train_loss=0.3264 val_loss=0.3240 val_psnr=23.71dB
  -> val_loss did not improve (6/7)
[CD-EXTERM] Epoch 19/30 [118.1s] [LR: 1.25e-05] train_loss=0.3264 val_loss=0.3254 val_psnr=23.64dB
  -> val_loss did not improve (7/7)
STOPPING: fine-tune has plateaued at lower LR.
Fine-tune finished. Compare against Phase 2 and baseline checkpoints using evaluate.py -- and specifically look at the worst-case samples (like 000352/002982) for whether SSIM loss actually reduced the blur/melting.
(/home/nithiish/Documents/kla_hackathon/.conda) nithiish@nithiish-pc:~/Documents/kla_hackathon$ python evaluation.py ./data ./checkpoints_cd_v1/best_model.pt
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:207: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:222: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: /home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/lpips/weights/v0.1/alex.pth
[SEMPairDataset] 3200 paired samples found.

Evaluating on 320 held-out validation samples...

=== Validation set summary (n=320) ===
PSNR: mean=24.86dB  std=4.40  min=12.45  max=44.09
SSIM: mean=0.5933  std=0.2039  min=0.1228  max=0.9639
LPIPS: mean=0.3538  std=0.1717  min=0.0450  max=0.7902

Worst 5 samples by PSNR:
  002982.npy: PSNR=12.45dB, SSIM=0.3057
  000398.npy: PSNR=18.14dB, SSIM=0.2035
  002483.npy: PSNR=18.23dB, SSIM=0.1465
  001927.npy: PSNR=18.30dB, SSIM=0.1713
  000397.npy: PSNR=18.30dB, SSIM=0.1652

=== Named Difficulty Samples ===
  000218.npy: PSNR=21.19dB, SSIM=0.4990
  000352.npy: PSNR=12.55dB, SSIM=0.3123
  000425.npy: PSNR=33.60dB, SSIM=0.8855

=============================================
        FINAL METROLOGY SCORECARD
=============================================
Mean CD Bias:   17.188 px
Mean LER Error: 24.829 px
Mean LWR Error: 35.425 px
Slope Fidelity: 114.1 %
=============================================

=== Processing Test Set (no GT) ===
[SEMTestDataset] 400 test samples found.

Done. Visuals and metrology summary in ./eval_outputs_cd_v1/



trainied the ms-ssim :

Resuming from ./checkpoints_baseline_v2/best_model.pt (epoch 50, val_psnr=27.96dB)
=== PHASE 4: MS-SSIM STRUCTURAL RECOVERY ===
Config: {'dim': 64, 'num_blocks': 2, 'batch_size': 4, 'target_batch_size': 32, 'scale_factor': 2, 'edge_weight': 0.5, 'freq_weight': 0.0, 'ssim_weight': 0.15, 'charbonnier_weight': 1.0, 'lr': 0.0002, 'weight_decay': 0.0001, 'scheduler_patience': 5, 'early_stop_patience': 10, 'num_workers': 8, 'num_epochs': 20, 'use_compile': False, 'checkpoint_dir': './checkpoints_msssim_v2', 'use_multiscale_ssim': True, 'ms_ssim_scales': [1.0, 0.5, 0.25], 'ms_ssim_weights': [0.5, 0.3, 0.2]}
[SEMPairDataset] 3200 paired samples found.
[SEMTestDataset] 400 test samples found.
Loaded checkpoint weights. Missing keys initialized to default.
--- [Modern GPU] Using bfloat16 on NVIDIA GeForce RTX 4050 Laptop GPU ---
--- [Compute] DataLoader batch_size=4, target_batch_size=32 -> accum_steps=8 (actual effective batch = 32) ---
[MS-SSIM] Epoch 1/20 [120.1s] [LR: 2.00e-04] train_loss=0.2407 val_loss=0.2025 val_psnr=22.25dB
  -> new best val_loss, saved to ./checkpoints_msssim_v2/best_model.pt
[MS-SSIM] Epoch 2/20 [119.4s] [LR: 2.00e-04] train_loss=0.1983 val_loss=0.1897 val_psnr=22.18dB
  -> new best val_loss, saved to ./checkpoints_msssim_v2/best_model.pt
[MS-SSIM] Epoch 3/20 [119.3s] [LR: 2.00e-04] train_loss=0.1885 val_loss=0.1780 val_psnr=22.84dB
  -> new best val_loss, saved to ./checkpoints_msssim_v2/best_model.pt
[MS-SSIM] Epoch 4/20 [119.2s] [LR: 2.00e-04] train_loss=0.1834 val_loss=0.1731 val_psnr=23.07dB
  -> new best val_loss, saved to ./checkpoints_msssim_v2/best_model.pt
[MS-SSIM] Epoch 5/20 [119.2s] [LR: 2.00e-04] train_loss=0.1809 val_loss=0.1681 val_psnr=23.29dB
  -> new best val_loss, saved to ./checkpoints_msssim_v2/best_model.pt
[MS-SSIM] Epoch 6/20 [119.2s] [LR: 2.00e-04] train_loss=0.1777 val_loss=0.1712 val_psnr=22.92dB
  -> val_loss did not improve (1/10)
[MS-SSIM] Epoch 7/20 [119.1s] [LR: 2.00e-04] train_loss=0.1756 val_loss=0.1684 val_psnr=23.18dB
  -> val_loss did not improve (2/10)
[MS-SSIM] Epoch 8/20 [119.1s] [LR: 2.00e-04] train_loss=0.1849 val_loss=0.1704 val_psnr=23.22dB
  -> val_loss did not improve (3/10)
[MS-SSIM] Epoch 9/20 [119.1s] [LR: 2.00e-04] train_loss=0.1773 val_loss=0.1746 val_psnr=22.84dB
  -> val_loss did not improve (4/10)
[MS-SSIM] Epoch 10/20 [119.1s] [LR: 2.00e-04] train_loss=0.1759 val_loss=0.1733 val_psnr=22.86dB
  -> val_loss did not improve (5/10)
[MS-SSIM] Epoch 11/20 [119.1s] [LR REDUCED: 2.00e-04 -> 1.00e-04] train_loss=0.1731 val_loss=0.1694 val_psnr=23.00dB
  -> val_loss did not improve (6/10)
[MS-SSIM] Epoch 12/20 [119.2s] [LR: 1.00e-04] train_loss=0.1714 val_loss=0.1637 val_psnr=23.25dB
  -> new best val_loss, saved to ./checkpoints_msssim_v2/best_model.pt
[MS-SSIM] Epoch 13/20 [119.1s] [LR: 1.00e-04] train_loss=0.1714 val_loss=0.1698 val_psnr=23.20dB
  -> val_loss did not improve (1/10)
[MS-SSIM] Epoch 14/20 [119.1s] [LR: 1.00e-04] train_loss=0.1702 val_loss=0.1636 val_psnr=23.29dB
  -> new best val_loss, saved to ./checkpoints_msssim_v2/best_model.pt
[MS-SSIM] Epoch 15/20 [119.1s] [LR: 1.00e-04] train_loss=0.1705 val_loss=0.1670 val_psnr=23.03dB
  -> val_loss did not improve (1/10)
[MS-SSIM] Epoch 16/20 [119.1s] [LR: 1.00e-04] train_loss=0.1706 val_loss=0.1674 val_psnr=23.02dB
  -> val_loss did not improve (2/10)
[MS-SSIM] Epoch 17/20 [119.1s] [LR: 1.00e-04] train_loss=0.1700 val_loss=0.1657 val_psnr=23.05dB
  -> val_loss did not improve (3/10)
[MS-SSIM] Epoch 18/20 [119.1s] [LR: 1.00e-04] train_loss=0.1693 val_loss=0.1608 val_psnr=23.35dB
  -> new best val_loss, saved to ./checkpoints_msssim_v2/best_model.pt
[MS-SSIM] Epoch 19/20 [119.1s] [LR: 1.00e-04] train_loss=0.1681 val_loss=0.1595 val_psnr=23.57dB
  -> new best val_loss, saved to ./checkpoints_msssim_v2/best_model.pt
[MS-SSIM] Epoch 20/20 [119.1s] [LR: 1.00e-04] train_loss=0.1677 val_loss=0.1654 val_psnr=23.06dB
  -> val_loss did not improve (1/10)
MS-SSIM fine-tune finished. Evaluate with evaluate.py
(/home/nithiish/Documents/kla_hackathon/.conda) nithiish@nithiish-pc:~/Documents/kla_hackathon$ python evaluation.py ./data ./checkpoints_msssim_v2/best_model.p
t
Loaded checkpoint: epoch 19, val_psnr=23.57dB
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:207: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:222: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: /home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/lpips/weights/v0.1/alex.pth
[SEMPairDataset] 3200 paired samples found.

Evaluating on 320 held-out validation samples...

=== Validation set summary (n=320) ===
PSNR: mean=25.05dB  std=4.60  min=12.37  max=43.03
SSIM: mean=0.6152  std=0.1976  min=0.1308  max=0.9497

Worst 5 samples by PSNR:
  002982.npy: PSNR=12.37dB, SSIM=0.2914
  001101.npy: PSNR=17.26dB, SSIM=0.1308
  001927.npy: PSNR=17.27dB, SSIM=0.1679
  002483.npy: PSNR=17.33dB, SSIM=0.1582
  001435.npy: PSNR=17.60dB, SSIM=0.1938

Best 5 samples by PSNR:
  000823.npy: PSNR=38.26dB, SSIM=0.9288
  003117.npy: PSNR=39.04dB, SSIM=0.9189
  002217.npy: PSNR=39.84dB, SSIM=0.9497
  002225.npy: PSNR=41.39dB, SSIM=0.9272
  001827.npy: PSNR=43.03dB, SSIM=0.9248

=== Named Difficulty Samples ===
  000218.npy: PSNR=20.94dB, SSIM=0.5043
  000352.npy: PSNR=12.46dB, SSIM=0.2971
  000425.npy: PSNR=34.13dB, SSIM=0.8692

=============================================
        FINAL METROLOGY SCORECARD
=============================================
Mean CD Bias:   20.679 px
Mean LER Error: 30.878 px
Mean LWR Error: 38.969 px
Slope Fidelity: 107.1 %
=============================================

=== Processing Test Set (no GT) ===
[SEMTestDataset] 400 test samples found.

Done. Detailed visuals and summary in ./eval_outputs_md_ssim/


(/home/nithiish/Documents/kla_hackathon/.conda) nithiish@nithiish-pc:~/Documents/kla_hackathon$ python train_baseline.py ./data ./checkpoints_baseline_v2/best_m
odel.pt
=== BASELINE ABLATION RUN ===
Config: {'dim': 64, 'num_blocks': 2, 'batch_size': 4, 'target_batch_size': 32, 'scale_factor': 2, 'edge_weight': 0.0, 'freq_weight': 0.0, 'ssim_weight': 0.0, 'charbonnier_weight': 1.0, 'lr': 0.0002, 'weight_decay': 0.0001, 'scheduler_patience': 3, 'early_stop_patience': 7, 'num_workers': 8, 'num_epochs': 50, 'use_compile': False, 'checkpoint_dir': './checkpoints_baseline_v3'}
[SEMPairDataset] 3200 paired samples found.
[SEMTestDataset] 400 test samples found.
--- [Modern GPU] Using bfloat16 on NVIDIA GeForce RTX 4050 Laptop GPU ---
--- [Compute] DataLoader batch_size=4, target_batch_size=32 -> accum_steps=8 (actual effective batch = 32) ---
[BASELINE] Epoch 1/50 [108.2s] [LR: 2.00e-04] train_loss=0.0380 val_loss=0.0333 val_psnr=26.25dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 2/50 [107.7s] [LR: 2.00e-04] train_loss=0.0335 val_loss=0.0322 val_psnr=26.58dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 3/50 [107.7s] [LR: 2.00e-04] train_loss=0.0317 val_loss=0.0305 val_psnr=26.90dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 4/50 [107.6s] [LR: 2.00e-04] train_loss=0.0308 val_loss=0.0300 val_psnr=27.00dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 5/50 [107.6s] [LR: 2.00e-04] train_loss=0.0303 val_loss=0.0296 val_psnr=27.12dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 6/50 [107.6s] [LR: 2.00e-04] train_loss=0.0301 val_loss=0.0294 val_psnr=27.18dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 7/50 [107.8s] [LR: 2.00e-04] train_loss=0.0298 val_loss=0.0291 val_psnr=27.25dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 8/50 [107.8s] [LR: 2.00e-04] train_loss=0.0295 val_loss=0.0289 val_psnr=27.28dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 9/50 [107.8s] [LR: 2.00e-04] train_loss=0.0293 val_loss=0.0287 val_psnr=27.32dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 10/50 [107.7s] [LR: 2.00e-04] train_loss=0.0291 val_loss=0.0285 val_psnr=27.38dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 11/50 [107.8s] [LR: 2.00e-04] train_loss=0.0290 val_loss=0.0284 val_psnr=27.42dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 12/50 [108.0s] [LR: 2.00e-04] train_loss=0.0288 val_loss=0.0283 val_psnr=27.43dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 13/50 [107.9s] [LR: 2.00e-04] train_loss=0.0287 val_loss=0.0282 val_psnr=27.47dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 14/50 [107.9s] [LR: 2.00e-04] train_loss=0.0286 val_loss=0.0280 val_psnr=27.51dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 15/50 [107.8s] [LR: 2.00e-04] train_loss=0.0285 val_loss=0.0279 val_psnr=27.55dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 16/50 [107.8s] [LR: 2.00e-04] train_loss=0.0284 val_loss=0.0278 val_psnr=27.57dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 17/50 [107.7s] [LR: 2.00e-04] train_loss=0.0283 val_loss=0.0277 val_psnr=27.59dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 18/50 [107.9s] [LR: 2.00e-04] train_loss=0.0282 val_loss=0.0277 val_psnr=27.61dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 19/50 [107.9s] [LR: 2.00e-04] train_loss=0.0281 val_loss=0.0276 val_psnr=27.63dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 20/50 [107.8s] [LR: 2.00e-04] train_loss=0.0280 val_loss=0.0275 val_psnr=27.64dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 21/50 [107.9s] [LR: 2.00e-04] train_loss=0.0280 val_loss=0.0274 val_psnr=27.68dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 22/50 [107.9s] [LR: 2.00e-04] train_loss=0.0279 val_loss=0.0275 val_psnr=27.66dB
  -> val_loss did not improve (1/7)
[BASELINE] Epoch 23/50 [107.8s] [LR: 2.00e-04] train_loss=0.0279 val_loss=0.0274 val_psnr=27.70dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 24/50 [107.7s] [LR: 2.00e-04] train_loss=0.0278 val_loss=0.0273 val_psnr=27.73dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 25/50 [107.8s] [LR: 2.00e-04] train_loss=0.0278 val_loss=0.0272 val_psnr=27.74dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 26/50 [107.8s] [LR: 2.00e-04] train_loss=0.0277 val_loss=0.0271 val_psnr=27.76dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 27/50 [107.7s] [LR: 2.00e-04] train_loss=0.0276 val_loss=0.0271 val_psnr=27.77dB
  -> val_loss did not improve (1/7)
[BASELINE] Epoch 28/50 [107.7s] [LR: 2.00e-04] train_loss=0.0276 val_loss=0.0271 val_psnr=27.78dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 29/50 [107.7s] [LR: 2.00e-04] train_loss=0.0275 val_loss=0.0270 val_psnr=27.79dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 30/50 [107.7s] [LR: 2.00e-04] train_loss=0.0275 val_loss=0.0270 val_psnr=27.81dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 31/50 [107.7s] [LR: 2.00e-04] train_loss=0.0275 val_loss=0.0270 val_psnr=27.82dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 32/50 [107.6s] [LR: 2.00e-04] train_loss=0.0274 val_loss=0.0269 val_psnr=27.83dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 33/50 [107.7s] [LR: 2.00e-04] train_loss=0.0274 val_loss=0.0269 val_psnr=27.83dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 34/50 [107.7s] [LR: 2.00e-04] train_loss=0.0274 val_loss=0.0269 val_psnr=27.84dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 35/50 [107.7s] [LR: 2.00e-04] train_loss=0.0273 val_loss=0.0268 val_psnr=27.84dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 36/50 [107.6s] [LR: 2.00e-04] train_loss=0.0273 val_loss=0.0269 val_psnr=27.82dB
  -> val_loss did not improve (1/7)
[BASELINE] Epoch 37/50 [107.7s] [LR: 2.00e-04] train_loss=0.0273 val_loss=0.0268 val_psnr=27.84dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 38/50 [107.7s] [LR: 2.00e-04] train_loss=0.0273 val_loss=0.0268 val_psnr=27.87dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 39/50 [107.7s] [LR: 2.00e-04] train_loss=0.0272 val_loss=0.0268 val_psnr=27.84dB
  -> val_loss did not improve (1/7)
[BASELINE] Epoch 40/50 [107.7s] [LR: 2.00e-04] train_loss=0.0272 val_loss=0.0267 val_psnr=27.87dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 41/50 [107.6s] [LR: 2.00e-04] train_loss=0.0272 val_loss=0.0267 val_psnr=27.88dB
  -> val_loss did not improve (1/7)
[BASELINE] Epoch 42/50 [107.7s] [LR: 2.00e-04] train_loss=0.0272 val_loss=0.0267 val_psnr=27.88dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 43/50 [107.6s] [LR: 2.00e-04] train_loss=0.0272 val_loss=0.0267 val_psnr=27.87dB
  -> val_loss did not improve (1/7)
[BASELINE] Epoch 44/50 [107.7s] [LR: 2.00e-04] train_loss=0.0271 val_loss=0.0268 val_psnr=27.89dB
  -> val_loss did not improve (2/7)
[BASELINE] Epoch 45/50 [107.6s] [LR: 2.00e-04] train_loss=0.0271 val_loss=0.0266 val_psnr=27.91dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 46/50 [107.6s] [LR: 2.00e-04] train_loss=0.0271 val_loss=0.0266 val_psnr=27.91dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 47/50 [107.6s] [LR: 2.00e-04] train_loss=0.0271 val_loss=0.0266 val_psnr=27.91dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 48/50 [107.7s] [LR: 2.00e-04] train_loss=0.0270 val_loss=0.0266 val_psnr=27.92dB
  -> val_loss did not improve (1/7)
[BASELINE] Epoch 49/50 [107.7s] [LR: 2.00e-04] train_loss=0.0270 val_loss=0.0265 val_psnr=27.92dB
  -> new best val_loss, saved to ./checkpoints_baseline_v3/best_model.pt
[BASELINE] Epoch 50/50 [107.9s] [LR: 2.00e-04] train_loss=0.0270 val_loss=0.0266 val_psnr=27.91dB
  -> val_loss did not improve (1/7)
Baseline run finished. Compare against Phase 2's ./checkpoints/best_model.pt using evaluate.py on both.
(/home/nithiish/Documents/kla_hackathon/.conda) nithiish@nithiish-pc:~/Documents/kla_hackathon$ python evaluation.py ./data ./checkpoints_baseline_v3/best_model
.pt
Loaded checkpoint: epoch 49, val_psnr=27.92dB
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:207: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:222: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: /home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/lpips/weights/v0.1/alex.pth
[SEMPairDataset] 3200 paired samples found.

Evaluating on 320 held-out validation samples...

=== Validation set summary (n=320) ===
PSNR: mean=29.50dB  std=4.63  min=12.61  max=45.32
SSIM: mean=0.7828  std=0.1541  min=0.2761  max=0.9727

Worst 5 samples by PSNR:
  002982.npy: PSNR=12.61dB, SSIM=0.2842
  001908.npy: PSNR=19.30dB, SSIM=0.3788
  001387.npy: PSNR=20.42dB, SSIM=0.7858
  000108.npy: PSNR=20.50dB, SSIM=0.6628
  001386.npy: PSNR=20.53dB, SSIM=0.7917

Best 5 samples by PSNR:
  001082.npy: PSNR=40.31dB, SSIM=0.9483
  002217.npy: PSNR=40.57dB, SSIM=0.9630
  003117.npy: PSNR=42.21dB, SSIM=0.9629
  002225.npy: PSNR=43.28dB, SSIM=0.9673
  001827.npy: PSNR=45.32dB, SSIM=0.9702

=== Named Difficulty Samples ===
  000218.npy: PSNR=32.12dB, SSIM=0.8577
  000352.npy: PSNR=12.69dB, SSIM=0.2906
  000425.npy: PSNR=34.52dB, SSIM=0.8837

=============================================
        FINAL METROLOGY SCORECARD
=============================================
Mean CD Bias:   -17.259 px
Mean LER Error: 25.098 px
Mean LWR Error: 29.892 px
Slope Fidelity: 75.0 %
=============================================

=== Processing Test Set (no GT) ===
[SEMTestDataset] 400 test samples found.

Done. Detailed visuals and summary in ./eval_outputs_baseline_v3/