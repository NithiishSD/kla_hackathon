problem :
The PSNR number — 7.81dB is very low

Working backward from the formula: PSNR = 10·log10(1/MSE) → 7.81dB implies MSE ≈ 0.166, i.e. an average squared error of ~0.17 on values that live in [0,1]. That's roughly what you'd get from a fairly chaotic, uncorrelated prediction — not "slightly noisy," but structurally quite far off. Train loss of 0.49 (which includes an L1-like Charbonnier term) points the same direction: half-scale average error is large.

solution in this solution:

The standard fix (used in EDSR/RCAN-style residual SR networks): zero-initialize the last conv layer's weights and bias, so at step 0 the residual really is zero and the network starts exactly at the bicubic baseline — then learns to move away from it


problem2:

The Problem: The "Cool-Down" Crisis
The OneCycleLR scheduler is like a flight plan for a 30-epoch journey:
Phase 1 (The Climb): Epochs 1–10. Learning rate (LR) goes up to "warm up" the model.
Phase 2 (The Cruise): Epochs 10–20. LR stays relatively high to explore the data.
Phase 3 (The Landing): Epochs 20–30. This is the most important part. The LR drops to almost zero. This is where the model stops "jumping around" and starts doing fine-tuning—the tiny pixel-level adjustments that give you those high SSIM scores.
The Expert's Warning:
If your Early Stopping (patience=5) triggers at Epoch 15 because the validation loss hasn't improved much, you have effectively "ejected" from the plane while it was still at 30,000 feet.
Because the LR was still high at Epoch 15, your "Best Checkpoint" will be a model that was still vibrating/jittering.
You missed the "Phase 3 Landing" where the model settles into the perfect, sharp-edge restoration.

solution:

Choice A: The "Let it Finish" Strategy
Increase your patience to 10 or 15, or disable Early Stopping entirely.
Pros: You are guaranteed to hit that final "Phase 3" low-LR fine-tuning.
Cons: If the model starts overfitting badly at Epoch 10, you waste 20 epochs of H100/T4 time.


Choice B: The "Metrology-Safe" Scheduler (Lead Scientist Recommendation)
Switch from OneCycleLR to ReduceLROnPlateau.
How it works: Instead of a pre-planned 30-epoch curve, it watches your validation loss. If the loss stops dropping for 2 or 3 epochs, it automatically cuts the learning rate by 10x.