"""
Synthetic degradation-diversity augmentation.

Broadens the training distribution beyond the fixed noise/blur severity
literally present in the paired dataset, per the review's OOD point:
"explore data augmentation ... that simulates varying noise levels and
noise types" for generalization.

Design constraints (consistent with everything established earlier):
- Applied ONLY to the LR input, never GT -- GT must stay the clean
  target, or this stops being a valid restoration task.
- Multiplicative/speckle-style noise, not generic additive Gaussian --
  matches the fan-shaped noise-vs-intensity pattern your own diagnostic
  plots showed (noise magnitude scales with signal, doesn't just add a
  flat offset).
- Applied stochastically (probability + randomized severity per sample)
  ON TOP OF the real degraded LR that's already there -- this ADDS
  variety, it doesn't replace real degradation with fake degradation.
- Only in the LR's own [0,1] normalized space, output re-clipped to
  [0,1] -- consistent with what sem_dataset.py already produces, no
  change to the calibration/normalization contract.
- Deliberately does NOT include the per-image contrast-stretch trick
  flagged earlier as harmful for metrology -- this is variety in
  degradation, not manufactured dynamic range.
"""

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def apply_synthetic_degradation(lr, rng, noise_prob=0.5, noise_sigma_range=(0.02, 0.10),
                                 blur_prob=0.3, blur_sigma_range=(0.3, 0.8)):
    """
    lr: numpy float32 array, already normalized to [0,1] (post-calibration).
    rng: numpy random Generator, for reproducibility if seeded upstream.

    Returns a further-degraded copy, still clipped to [0,1].
    """
    out = lr.copy()

    if rng.random() < noise_prob:
        sigma = rng.uniform(*noise_sigma_range)
        # Multiplicative speckle-style: noise scales with local intensity,
        # matching the fan-shaped noise-vs-signal pattern from the
        # diagnostic plots, rather than a flat additive offset.
        speckle = rng.normal(loc=0.0, scale=sigma, size=out.shape).astype(np.float32)
        out = out * (1.0 + speckle)

    if blur_prob > 0 and _SCIPY_AVAILABLE and rng.random() < blur_prob:
        sigma = rng.uniform(*blur_sigma_range)
        out = gaussian_filter(out, sigma=sigma).astype(np.float32)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


class SyntheticAugmentMixin:
    """Mixin that adds synthetic degradation on top of an existing
    SEMPairDataset's __getitem__, LR side only. Use via
    SEMPairDatasetSyntheticAug below rather than directly."""

    def _init_synthetic_aug(self, seed=123, **aug_kwargs):
        self._synth_rng = np.random.default_rng(seed)
        self._aug_kwargs = aug_kwargs


def make_synthetic_aug_dataset(base_dataset_cls):
    """Factory: wraps a SEMPairDataset subclass so its LR tensor gets
    synthetic degradation applied post-hoc, without modifying
    sem_dataset.py itself (keeps the already-verified base pipeline
    untouched for every other script)."""

    class _Wrapped(base_dataset_cls, SyntheticAugmentMixin):
        def __init__(self, *args, synthetic_aug=True, noise_prob=0.5,
                     noise_sigma_range=(0.02, 0.10), blur_prob=0.3,
                     blur_sigma_range=(0.3, 0.8), seed=123, **kwargs):
            super().__init__(*args, **kwargs)
            self.synthetic_aug = synthetic_aug
            self._init_synthetic_aug(
                seed=seed, noise_prob=noise_prob,
                noise_sigma_range=noise_sigma_range, blur_prob=blur_prob,
                blur_sigma_range=blur_sigma_range,
            )

        def __getitem__(self, idx):
            lr_t, gt_t = super().__getitem__(idx)
            if self.synthetic_aug:
                lr_np = lr_t.squeeze(0).numpy()
                lr_np = apply_synthetic_degradation(lr_np, self._synth_rng, **self._aug_kwargs)
                import torch
                lr_t = torch.from_numpy(lr_np).unsqueeze(0)
            return lr_t, gt_t

    return _Wrapped