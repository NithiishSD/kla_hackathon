Three pieces: (1) bake calibration stats into checkpoints going forward, (2) a one-time patch script so your existing baseline checkpoint doesn't need a retrain just to gain this, (3) the actual infer.py matching the required contract.


scripts done

ython bake_calib_into_checkpoints.py ./data ./checkpoints_baseline/best_model.pt 
python infer.py <test_images_dir> <output_dir>

pip install lpips   # one-time, on your laptop (downloads AlexNet weights, needs normal internet)
python train_hard_oversample.py ./data ./checkpoints_baseline/best_model.pt
python evaluate.py ./data ./checkpoints_hard_oversample/best_model.pt


complete the validation and packaging process for submission



so choose baseline model first then phase 2 model avaoid ssim model


/home/nithiish/Documents/kla_hackathon/.conda) nithiish@nithiish-pc:~/Documents/kla_hackathon$ python infer.py ./data/test ./test_results
Processed 400 files.
Model load time: 1.962s
Total end-to-end time (incl. model load, I/O, inference, write): 6.910s
Average per-file time (I/O + inference + write): 12.4ms
Average end-to-end throughput: 57.89 images/sec
Outputs written to ./test_results



Using device: cuda
Loaded checkpoint from epoch 18, val_loss=0.0776, val_psnr=27.87dB (as recorded during training)
Checkpoint config: {'dim': 64, 'num_blocks': 2, 'batch_size': 4, 'target_batch_size': 32, 'scale_factor': 2, 'edge_weight': 0.5, 'freq_weight': 0.0, 'ssim_weight': 0.0, 'lr': 0.0002, 'weight_decay': 0.0001, 'scheduler_patience': 3, 'early_stop_patience': 7, 'num_workers': 8, 'num_epochs': 20, 'use_compile': False, 'checkpoint_dir': './checkpoints_hard_oversample', 'oversample_factor': 4.0, 'hard_fraction': 0.1}
Loading LPIPS (AlexNet backbone)...
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:207: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/torchvision/models/_utils.py:222: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: /home/nithiish/Documents/kla_hackathon/.conda/lib/python3.12/site-packages/lpips/weights/v0.1/alex.pth
[SEMPairDataset] 3200 paired samples found.

Evaluating on 320 held-out validation samples...

=== Validation set summary (n=320) ===
PSNR: mean=29.41dB  std=4.57  min=12.49  max=45.52
SSIM: mean=0.7824  std=0.1473  min=0.2991  max=0.9636
LPIPS: mean=0.2452  std=0.1354  min=0.0278  max=0.7190  (lower = more perceptually similar; AlexNet backbone trained on natural photos, treat as directional for SEM data, not an absolute validated benchmark)

Worst 5 samples by PSNR (your weakest cases -- inspect these):
  002982.npy: PSNR=12.49dB, SSIM=0.3172, LPIPS=0.3059
  001908.npy: PSNR=19.54dB, SSIM=0.4245, LPIPS=0.4741
  000108.npy: PSNR=20.52dB, SSIM=0.6741, LPIPS=0.2405
  001387.npy: PSNR=20.62dB, SSIM=0.8002, LPIPS=0.0902
  001386.npy: PSNR=20.74dB, SSIM=0.8060, LPIPS=0.0867

Best 5 samples by PSNR:
  001082.npy: PSNR=39.52dB, SSIM=0.9431, LPIPS=0.1216
  002217.npy: PSNR=40.44dB, SSIM=0.9611, LPIPS=0.0778
  003117.npy: PSNR=42.16dB, SSIM=0.9602, LPIPS=0.1062
  002225.npy: PSNR=43.13dB, SSIM=0.9624, LPIPS=0.1187
  001827.npy: PSNR=45.52dB, SSIM=0.9636, LPIPS=0.0432

Saved worst-3 comparison plots to ./eval_outputs_baseline_hardtrained/

=== Named difficulty samples (from your diagnostic plots) ===
  000218.npy [TRAIN (model saw this during training)]: PSNR=32.10dB, SSIM=0.8539
    (in training set -- this number reflects fit, not generalization; don't present it as a held-out result -- but a LOW score here despite training on it is still worth investigating, it suggests the sample itself is unusually hard or possibly anomalous data)
  000352.npy [TRAIN (model saw this during training)]: PSNR=12.60dB, SSIM=0.3240
    (in training set -- this number reflects fit, not generalization; don't present it as a held-out result -- but a LOW score here despite training on it is still worth investigating, it suggests the sample itself is unusually hard or possibly anomalous data)
  000425.npy [TRAIN (model saw this during training)]: PSNR=34.57dB, SSIM=0.8868
    (in training set -- this number reflects fit, not generalization; don't present it as a held-out result -- but a LOW score here despite training on it is still worth investigating, it suggests the sample itself is unusually hard or possibly anomalous data)

=== Test set (no ground truth -- visuals only, no PSNR/SSIM) ===
[SEMTestDataset] 400 test samples found.
Saved 5 test-set prediction visuals to ./eval_outputs_baseline_hardtrained/ (no numeric score possible -- eyeball these for artifacts, especially any known OOD / 'Woven Grid' samples)
Saved numeric summary to ./eval_outputs_baseline_hardtrained/summary.json