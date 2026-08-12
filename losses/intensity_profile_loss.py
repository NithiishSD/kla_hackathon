"""
IntensityProfileLoss -- targets the root cause diagnosed in
deepseek_validation.txt: CD bias isn't a spatial shift (dilation testing
in correct_cd.py already ruled that out -- features ARE in the right
place), it's that the model's edge transition has a different SLOPE
than ground truth's, which shifts where a threshold-based CD measurement
lands even though the edge position itself is correct.

This is deliberately NOT another edge-position loss like
losses/metrology_cd_losses.py's ThresholdedCDLoss, which already failed
(Chapter 3 of deepseek_validation.txt): that loss weighted the penalty by
the model's OWN predicted edge map, so the model's cheapest way to
minimize it was to dilate/spam edges everywhere rather than fix slope
shape. Here the mask is built from the ground truth only and detached
(no_grad) -- it cannot be gamed by changing what the model predicts,
since the model's output only affects the loss through the gradient
VALUE compared at those fixed locations, not through which locations
get penalized.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntensityProfileLoss(nn.Module):
    def __init__(self, edge_percentile=85.0):
        super().__init__()
        self.edge_percentile = edge_percentile
        self.register_buffer('kernel_x', torch.tensor([[-1., 0., 1.]]).view(1, 1, 1, 3) / 2.0)
        self.register_buffer('kernel_y', torch.tensor([[-1.], [0.], [1.]]).view(1, 1, 3, 1) / 2.0)

    def _gradient_magnitude(self, img):
        gx = F.conv2d(img, self.kernel_x, padding=(0, 1))
        gy = F.conv2d(img, self.kernel_y, padding=(1, 0))
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred, target):
        pred_f, target_f = pred.float(), target.float()
        target_grad = self._gradient_magnitude(target_f)
        pred_grad = self._gradient_magnitude(pred_f)

        # Fixed GT-only edge mask, detached -- the model cannot reduce
        # this loss by changing which pixels get penalized, only by
        # matching the actual slope magnitude at GT's true edges.
        with torch.no_grad():
            b = target_grad.shape[0]
            mask = torch.zeros_like(target_grad)
            for i in range(b):
                flat = target_grad[i].reshape(-1)
                thresh = torch.quantile(flat, self.edge_percentile / 100.0)
                mask[i] = (target_grad[i] > thresh).float()

        diff = (pred_grad - target_grad).abs() * mask
        return diff.sum() / (mask.sum() + 1e-6)
