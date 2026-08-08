"""
Dataset / DataLoader using shared global percentile calibration.

Requires calib_stats.json (see calibrate_stats.py) so LR and GT are
normalized with the EXACT SAME affine transform -- 0/1 means the same
physical intensity in both, and the same transform is reused for
train/val/test so metrics stay comparable across the whole pipeline.

Folder layout assumed:
    train/gt/*.npy       -- clean, 256x256
    train/lossylr/*.npy  -- noisy, 128x128
    test/*.npy           -- noisy, 128x128, no gt
"""

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_calib_stats(data_root):
    path = os.path.join(data_root, "calib_stats.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run calibrate_stats.py {data_root} first -- "
            f"normalization must be computed once and reused everywhere, "
            f"not recomputed ad hoc."
        )
    with open(path) as f:
        stats = json.load(f)
    return stats["p_low"], stats["p_high"]


def normalize(arr, p_low, p_high):
    """Affine map: p_low -> 0, p_high -> 1. Values outside [p_low, p_high]
    (the speckle spikes) are clamped, not stretched -- this is the
    'robust percentile scaling' step."""
    arr = (arr - p_low) / (p_high - p_low)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


class SEMPairDataset(Dataset):
    """Paired GT / noisy-LR dataset for training, using shared calibration."""

    def __init__(self, gt_dir, lr_dir, p_low, p_high, scale_factor=2,
                 augment=True, strict_name_match=True):
        self.gt_dir = gt_dir
        self.lr_dir = lr_dir
        self.p_low = p_low
        self.p_high = p_high
        self.scale_factor = scale_factor
        self.augment = augment

        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
        lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))

        if strict_name_match:
            gt_names = {os.path.basename(f) for f in gt_files}
            lr_names = {os.path.basename(f) for f in lr_files}
            common = gt_names & lr_names
            if len(common) != len(gt_names) or len(common) != len(lr_names):
                missing_in_lr = gt_names - lr_names
                missing_in_gt = lr_names - gt_names
                raise ValueError(
                    f"Filename mismatch between gt/ and lossylr/.\n"
                    f"In gt but not lossylr ({len(missing_in_lr)}): {list(missing_in_lr)[:5]}...\n"
                    f"In lossylr but not gt ({len(missing_in_gt)}): {list(missing_in_gt)[:5]}...\n"
                    f"Pass strict_name_match=False if names differ but are index-aligned."
                )
            self.pairs = [(os.path.join(gt_dir, n), os.path.join(lr_dir, n))
                          for n in sorted(common)]
        else:
            if len(gt_files) != len(lr_files):
                raise ValueError(
                    f"gt/ has {len(gt_files)} files, lossylr/ has {len(lr_files)} "
                    f"-- can't pair positionally."
                )
            self.pairs = list(zip(gt_files, lr_files))

        print(f"[SEMPairDataset] {len(self.pairs)} paired samples found.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_path, lr_path = self.pairs[idx]
        gt = normalize(np.load(gt_path).astype(np.float32), self.p_low, self.p_high)
        lr = normalize(np.load(lr_path).astype(np.float32), self.p_low, self.p_high)

        expected_h = gt.shape[0] // self.scale_factor
        expected_w = gt.shape[1] // self.scale_factor
        if (lr.shape[0], lr.shape[1]) != (expected_h, expected_w):
            raise ValueError(
                f"Shape mismatch at {os.path.basename(lr_path)}: gt={gt.shape}, "
                f"lr={lr.shape}, expected lr={(expected_h, expected_w)} for "
                f"scale_factor={self.scale_factor}."
            )

        gt_t = torch.from_numpy(gt).unsqueeze(0)  # (1, H, W)
        lr_t = torch.from_numpy(lr).unsqueeze(0)  # (1, h, w)

        if self.augment:
            gt_t, lr_t = self._augment(gt_t, lr_t)

        return lr_t, gt_t

    @staticmethod
    def _augment(gt_t, lr_t):
        # Bit-perfect augmentation only -- torch.rot90 / torch.flip are exact
        # index permutations, unlike torchvision's interpolated rotate(),
        # which would blur precision float32 metrology data.
        if torch.rand(1).item() < 0.5:
            gt_t = torch.flip(gt_t, dims=[-1])
            lr_t = torch.flip(lr_t, dims=[-1])
        if torch.rand(1).item() < 0.5:
            gt_t = torch.flip(gt_t, dims=[-2])
            lr_t = torch.flip(lr_t, dims=[-2])
        k = torch.randint(0, 4, (1,)).item()
        if k:
            gt_t = torch.rot90(gt_t, k, dims=[-2, -1])
            lr_t = torch.rot90(lr_t, k, dims=[-2, -1])
        return gt_t, lr_t


class SEMTestDataset(Dataset):
    """Test set: noisy-LR only, no ground truth. Uses the SAME calibration
    stats as training -- never fit fresh stats on the test set."""

    def __init__(self, test_dir, p_low, p_high):
        self.p_low = p_low
        self.p_high = p_high
        self.files = sorted(glob.glob(os.path.join(test_dir, "*.npy")))
        if not self.files:
            raise ValueError(f"No .npy files found in {test_dir}")
        print(f"[SEMTestDataset] {len(self.files)} test samples found.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        lr = normalize(np.load(path).astype(np.float32), self.p_low, self.p_high)
        lr_t = torch.from_numpy(lr).unsqueeze(0)
        return lr_t, os.path.basename(path)


class _NoAugWrapper(Dataset):
    """Wraps a Subset of SEMPairDataset and forces augment=False for reads
    that go through it, without permanently mutating the shared underlying
    dataset (fixes the earlier train/val augmentation-sharing issue)."""

    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        ds = self.subset.dataset
        real_idx = self.subset.indices[idx]
        was_augment = ds.augment
        ds.augment = False
        try:
            item = ds[real_idx]
        finally:
            ds.augment = was_augment
        return item


def build_dataloaders(data_root, scale_factor=2, batch_size=8, num_workers=8,
                       val_fraction=0.1, seed=42):
    p_low, p_high = load_calib_stats(data_root)

    full_train = SEMPairDataset(
        gt_dir=os.path.join(data_root, "train", "gt"),
        lr_dir=os.path.join(data_root, "train", "lossylr"),
        p_low=p_low, p_high=p_high,
        scale_factor=scale_factor,
        augment=True,
    )

    n_val = max(1, int(len(full_train) * val_fraction))
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [n_train, n_val], generator=generator
    )
    # CHANGE: val now goes through a wrapper that forces augment=False
    # instead of silently sharing the train-side augment flag.
    val_set = _NoAugWrapper(val_subset)

    test_set = SEMTestDataset(os.path.join(data_root, "test"), p_low, p_high)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True,
                               drop_last=True, prefetch_factor=2 if num_workers > 0 else None)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             prefetch_factor=2 if num_workers > 0 else None)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              prefetch_factor=2 if num_workers > 0 else None)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data"

    train_loader, val_loader, test_loader = build_dataloaders(data_root, scale_factor=2, batch_size=4)

    lr, gt = next(iter(train_loader))
    print(f"Train batch: lr={lr.shape} dtype={lr.dtype}, gt={gt.shape}")
    print(f"lr range: [{lr.min():.3f}, {lr.max():.3f}], gt range: [{gt.min():.3f}, {gt.max():.3f}]")

    lr_test, names = next(iter(test_loader))
    print(f"Test batch: lr={lr_test.shape}, filenames={names}")
