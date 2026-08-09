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


Recommended order from here
Retry the SSIM fine-tune with the controlled config above — this is still your best lever for the bulk of your samples, the earlier failure was a hyperparameter mistake, not evidence the idea is bad.
2)Set 000352/002982-class samples aside for your main accuracy push — don't let a handful of information-theoretically hard samples eat more of your remaining time chasing PSNR gains that may not be achievable.
Skip the 80/20 exclusion experiment — you now have a real explanation for the worst cases; changing the split ratio won't change the underlying loss-vs-stochastic-target mismatch.
Optionally, if time permits: report a trimmed-mean or median PSNR/SSIM alongside the mean in your final writeup, so 2–3 structurally unsolvable samples don't dominate your headline number — with a note explaining why, which reads as rigor, not cherry-picking.




for ( 2)

Concrete next steps

Quantify how much of an outlier this is. Compute per-sample noise std (or SNR estimate: std(LR - GT)) across your whole training set, then rank it. If 000352/002982 sit in the extreme tail (e.g., top 1–2% noise level), that confirms this is a distribution-tail problem, not a fluke — and tells you whether it's one or two isolated cases or a broader class you should be worried about for the KLA test set too.
Check if the radial degradation pattern shows up on other samples too. If it's systematic across many images (not just this one), that's a real, fixable insight — e.g., feeding the model a normalized radial-distance channel, or checking whether your data augmentation/crop strategy is centering crops in a way that biases training away from edge regions.
Don't exclude these from train or val. They're legitimate data, and the KLA test set (400 unseen samples) likely contains similarly extreme cases — excluding your hardest known examples to inflate your validation PSNR would make your reported metric less representative of real test performance, not more.
For your writeup: report a trimmed or percentile-based PSNR/SSIM (e.g. P10/median/P90) alongside the mean, and call out that a small number of extreme low-SNR captures represent a known, expected floor — this is honest, defensible framing rather than something to hide.


— this file resolves the mystery, and it's not a split mismatch after all. Splits match exactly (val_fraction=0.1, seed=42, same SEMPairDataset construction, same files) between build_dataloaders and evaluate.py's manual reconstruction. The real cause is a metric computation difference.

The actual bug: batch-averaged PSNR vs per-image-averaged PSNR

Look at how PSNR gets computed during training:

python
for lr_img, gt_img in val_loader:
    loss, pred = engine.eval_step(criterion, lr_img, gt_img)
    val_psnrs.append(compute_psnr(pred, gt_img.to(engine.device)))
avg_val_psnr = sum(val_psnrs) / len(val_psnrs)

pred and gt_img here are a whole batch (batch_size=4). Inside compute_psnr:

python
mse = torch.mean((pred - target) ** 2)   # averages over ALL 4 images in the batch at once
return 10 * torch.log10(1.0 / mse)        # ONE psnr value for the whole batch

So the training loop computes one PSNR per batch of 4 images, then averages those batch-level PSNR values across batches.

evaluate.py, by contrast, loops one image at a time and averages individual per-image PSNR values.

These are mathematically different quantities, and not just by a rounding error. PSNR is -10·log10(MSE), a concave function of MSE. By Jensen's inequality, when MSE varies across images (which it clearly does — your std is 4.57dB), averaging MSE first (across a batch) then taking one log will always come out lower than averaging individually-computed PSNR values. A single noisy/hard image in a batch of 4 drags down that whole batch's blended MSE heavily, and that batch then reports one low PSNR — whereas in the per-image approach, that same hard image just contributes one low number among many, without contaminating its batch-mates' scores.

This exactly explains the direction and rough size of every discrepancy you've seen:

Hard-oversample checkpoint: training log said val_psnr=27.87dB → evaluate.py said mean 29.41dB (+1.5dB)
Baseline checkpoint: training log said val_psnr=27.85dB → your earlier three-way table said 29.37dB via evaluate.py (+1.5dB, same size gap)

Consistent ~1.5dB bias, both times, same direction. That's not noise — that's the batching artifact.

What this means practically
Your past comparisons are still valid. Baseline vs Phase 2 vs hard-oversample were all compared via evaluate.py's per-image numbers, consistently. Nothing to redo there.
In-training val_psnr logs (what you're watching right now during the SSIM run) are self-consistent with each other, just biased low vs. evaluate.py. Epoch 1→4 going 21.65→23.38dB is real, legitimate recovery-in-progress — just don't compare that 23.38dB directly against evaluate.py's 29.37dB baseline number. Compare it against the training log's own resumed starting point (27.85dB, same metric) — so it's still ~4.5dB below where it started, recovering but not there yet.
When this run finishes, run evaluate.py on the resulting checkpoint — that's the only number to trust for your final PSNR comparison against baseline's 29.37dB / hard-oversample's 29.41dB.

Step 1 (do this first): re-run the SSIM fine-tune, corrected

Your last attempt failed because too much changed at once (ssim_weight 0→0.4, edge_weight 0.5→1.0, lr only halved). Fix all three:

python
config.update(dict(
    ssim_weight=0.1,        # small, single new signal — was 0.4
    lr=1e-5,                # real fine-tune LR, 20x lower — was 1e-4
    edge_weight=base_cfg["edge_weight"],       # KEEP baseline's 0.5, don't touch
    weight_decay=base_cfg["weight_decay"],     # KEEP baseline's 1e-4, don't touch
    freq_weight=0.0,        # still off — isolate one variable
    num_epochs=15,
    scheduler_patience=2,
    early_stop_patience=5,
    checkpoint_dir="./checkpoints_ssim_v2",
))

And change the resume source in the script to your baseline checkpoint:

bash
python finetune_ssim.py ./data ./checkpoints_baseline/best_model.pt

What to watch for: if val_psnr in epoch 1-2 stays close to baseline's 29.37dB (not collapsing like last time), you know the fine-tune is stable. If it's still unstable even at lr=1e-5, drop to 5e-6 next.

Then run evaluate.py on the resulting checkpoint and compare against baseline using the same percentile/tail breakdown I ran last message — not just the mean, since that's what actually tells you if it helped the tail.

Step 2: only if Step 1 succeeds, consider freq_weight

Your FourierUnit is running in every block but getting no direct supervision (freq_weight=0.0). If Step 1 gives a clean, stable checkpoint, do one more single-variable fine-tune from that checkpoint adding freq_weight=0.05–0.1, everything else pinned. Don't do this simultaneously with Step 1 — you want to know which change did what.

Direct answer: don't do hard-oversampling on the SSIM model — not yet, and possibly not at all

Here's the reasoning, not just the verdict:

You already tested oversampling in isolation and it did nothing (std 4.55→4.57, statistically flat). There's no new evidence to suggest it'll behave differently on top of a different loss — running it again without a reason to expect a different outcome is just spending a training run to confirm what you already know.
It would also break your ablation discipline. If you stack oversampling on top of a new SSIM-tuned model and something changes, you won't know whether it was the SSIM loss, the oversampling, or their interaction. You've been doing single-variable changes well so far — keep that up.
If Step 1 succeeds and you still see a stubborn tail (the ~57 sub-25dB samples), the right move isn't to oversample blindly again — it's to re-score hard samples using the new SSIM-tuned model (the hard set may have shifted, since SSIM loss changes what "hard" means) and only then decide if oversampling is worth a third try, now with a specific, falsifiable hypothesis rather than a repeat.
Summary of what to run, in order
Priority	Action	Change vs last known-good	Expected outcome to check
1	SSIM fine-tune v2, from baseline ckpt	ssim_weight=0.1, lr=1e-5 only	val_psnr stays near 29.3-29.5dB, SSIM improves, no collapse
2 (conditional)	Add freq_weight=0.05-0.1, from Step 1's ckpt	one more single variable	small further SSIM/PSNR gain, or no change (also useful info)
3 (only if tail persists)	Re-score hard samples on best model, decide fresh	new hypothesis, not a repeat	—


running the ssim finetune model

1)
config.update(dict(
    ssim_weight=0.1,        # small, single new signal — was 0.4
    lr=1e-5,                # real fine-tune LR, 20x lower — was 1e-4
    edge_weight=base_cfg["edge_weight"],       # KEEP baseline's 0.5, don't touch
    weight_decay=base_cfg["weight_decay"],     # KEEP baseline's 1e-4, don't touch
    freq_weight=0.0,        # still off — isolate one variable
    num_epochs=15,
    scheduler_patience=2,
    early_stop_patience=5,
    checkpoint_dir="./checkpoints_ssim_v2",
))

result:



--- [Compute] DataLoader batch_size=4, target_batch_size=32 -> accum_steps=8 (actual effective batch = 32) ---
[SSIM-FT] Epoch 1/15 [117.7s] [LR: 1.00e-05] train_loss=0.1912 val_loss=0.2616 val_psnr=21.65dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 2/15 [116.6s] [LR: 1.00e-05] train_loss=0.2287 val_loss=0.1984 val_psnr=22.78dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 3/15 [116.6s] [LR: 1.00e-05] train_loss=0.1915 val_loss=0.1786 val_psnr=23.22dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 4/15 [116.6s] [LR: 1.00e-05] train_loss=0.1788 val_loss=0.1705 val_psnr=23.38dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 5/15 [116.7s] [LR: 1.00e-05] train_loss=0.1710 val_loss=0.1635 val_psnr=23.57dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 6/15 [116.7s] [LR: 1.00e-05] train_loss=0.1643 val_loss=0.1579 val_psnr=23.68dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 7/15 [116.7s] [LR: 1.00e-05] train_loss=0.1588 val_loss=0.1523 val_psnr=23.87dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 8/15 [116.7s] [LR: 1.00e-05] train_loss=0.1540 val_loss=0.1487 val_psnr=23.95dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 9/15 [116.7s] [LR: 1.00e-05] train_loss=0.1504 val_loss=0.1456 val_psnr=24.04dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 10/15 [116.7s] [LR: 1.00e-05] train_loss=0.1476 val_loss=0.1425 val_psnr=24.18dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 11/15 [116.7s] [LR: 1.00e-05] train_loss=0.1454 val_loss=0.1408 val_psnr=24.21dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 12/15 [116.7s] [LR: 1.00e-05] train_loss=0.1436 val_loss=0.1397 val_psnr=24.21dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 13/15 [116.7s] [LR: 1.00e-05] train_loss=0.1422 val_loss=0.1383 val_psnr=24.22dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 14/15 [116.7s] [LR: 1.00e-05] train_loss=0.1411 val_loss=0.1374 val_psnr=24.23dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt
[SSIM-FT] Epoch 15/15 [116.7s] [LR: 1.00e-05] train_loss=0.1403 val_loss=0.1374 val_psnr=24.15dB
  -> new best val_loss, saved to ./checkpoints_ssim_v2/best_model.pt

/home/nithiish/Documents/kla_hackathon/.conda) nithiish@nithiish-pc:~/Documents/kla_hackathon$ python evaluation.py ./data ./checkpoints_ssim_v2/best_model.pt
Using device: cuda
Loaded checkpoint from epoch 15, val_loss=0.1374, val_psnr=24.15dB (as recorded during training)
Checkpoint config: {'dim': 64, 'num_blocks': 2, 'batch_size': 4, 'target_batch_size': 32, 'scale_factor': 2, 'edge_weight': 0.5, 'freq_weight': 0.0, 'ssim_weight': 0.1, 'lr': 1e-05, 'weight_decay': 0.0001, 'scheduler_patience': 2, 'early_stop_patience': 5, 'num_workers': 8, 'num_epochs': 15, 'use_compile': False, 'checkpoint_dir': './checkpoints_ssim_v2'}
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
PSNR: mean=25.42dB  std=4.33  min=12.44  max=44.19
SSIM: mean=0.5967  std=0.2091  min=0.1328  max=0.9544
LPIPS: mean=0.3050  std=0.1699  min=0.0387  max=0.8620  (lower = more perceptually similar; AlexNet backbone trained on natural photos, treat as directional for SEM data, not an absolute validated benchmark)

Worst 5 samples by PSNR (your weakest cases -- inspect these):
  002982.npy: PSNR=12.44dB, SSIM=0.2982, LPIPS=0.4730
  001908.npy: PSNR=18.57dB, SSIM=0.4421, LPIPS=0.4123
  000398.npy: PSNR=18.93dB, SSIM=0.1612, LPIPS=0.8204
  001927.npy: PSNR=18.95dB, SSIM=0.1397, LPIPS=0.8162
  000397.npy: PSNR=19.28dB, SSIM=0.1426, LPIPS=0.7493

Best 5 samples by PSNR:
  000823.npy: PSNR=38.13dB, SSIM=0.9286, LPIPS=0.1039
  003117.npy: PSNR=39.53dB, SSIM=0.9274, LPIPS=0.2226
  002217.npy: PSNR=39.94dB, SSIM=0.9544, LPIPS=0.0494
  002225.npy: PSNR=42.42dB, SSIM=0.9487, LPIPS=0.0844
  001827.npy: PSNR=44.19dB, SSIM=0.9517, LPIPS=0.0387

Saved worst-3 comparison plots to ./eval_outputs_ssim_v2/

=== Named difficulty samples (from your diagnostic plots) ===
  000218.npy [TRAIN (model saw this during training)]: PSNR=23.13dB, SSIM=0.5165
    (in training set -- this number reflects fit, not generalization; don't present it as a held-out result -- but a LOW score here despite training on it is still worth investigating, it suggests the sample itself is unusually hard or possibly anomalous data)
  000352.npy [TRAIN (model saw this during training)]: PSNR=12.54dB, SSIM=0.3043
    (in training set -- this number reflects fit, not generalization; don't present it as a held-out result -- but a LOW score here despite training on it is still worth investigating, it suggests the sample itself is unusually hard or possibly anomalous data)
  000425.npy [TRAIN (model saw this during training)]: PSNR=33.87dB, SSIM=0.8842
    (in training set -- this number reflects fit, not generalization; don't present it as a held-out result -- but a LOW score here despite training on it is still worth investigating, it suggests the sample itself is unusually hard or possibly anomalous data)

=== Test set (no ground truth -- visuals only, no PSNR/SSIM) ===
[SEMTestDataset] 400 test samples found.
Saved 5 test-set prediction visuals to ./eval_outputs_ssim_v2/ (no numeric score possible -- eyeball these for artifacts, especially any known OOD / 'Woven Grid' samples)
Saved numeric summary to ./eval_outputs_ssim_v2/summary.json





test2)


chenging ssim back to zero and running for extra 15 epochs:



result:











now running teh ssim from scratch: