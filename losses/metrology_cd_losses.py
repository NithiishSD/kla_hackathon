import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

class ThresholdedCDLoss(nn.Module):
    """
    Penalizes the distance between predicted edges and ground truth edges.
    Directly minimizes CD Bias in pixel units.
    """
    def __init__(self, threshold_percentile=50.0, max_distance=30.0):
        super().__init__()
        self.threshold_percentile = threshold_percentile
        self.max_distance = max_distance
        
        # Edge detection kernels
        self.register_buffer('kernel_x', torch.tensor([[-1., 0., 1.]]) / 2.0)
        self.register_buffer('kernel_y', torch.tensor([[-1.], [0.], [1.]]) / 2.0)
    
    def extract_edges(self, img):
        """Extract thin edges using gradient magnitude thresholding."""
        grad_x = F.conv2d(img, self.kernel_x.view(1,1,1,3), padding=(0,1))
        grad_y = F.conv2d(img, self.kernel_y.view(1,1,3,1), padding=(1,0))
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Adaptive threshold: keep top percentile of gradient magnitudes
        batch_size = grad_mag.shape[0]
        edges = []
        for b in range(batch_size):
            flat = grad_mag[b].view(-1)
            thresh = torch.quantile(flat, self.threshold_percentile / 100.0)
            edges.append((grad_mag[b] > thresh).float().unsqueeze(0))
        return torch.cat(edges, dim=0)
    
    def compute_distance_map(self, edge_map):
        """Compute Euclidean distance transform for a single edge map (CPU numpy)."""
        edge_np = edge_map.cpu().numpy()
        # Distance to nearest edge pixel
        dist = distance_transform_edt(1.0 - edge_np)
        # Clamp to prevent single outliers from dominating
        dist = np.clip(dist, 0, self.max_distance)
        return torch.from_numpy(dist).to(edge_map.device)
    
    def forward(self, pred, target):
        """
        pred, target: (B, 1, H, W) tensors normalized to ~[0,1]
        Returns: CD loss in approximate pixel units
        """
        pred_edges = self.extract_edges(pred)
        target_edges = self.extract_edges(target)
        
        # Distance from each predicted edge pixel to nearest target edge
        total_loss = 0.0
        for b in range(pred.shape[0]):
            dist_map = self.compute_distance_map(target_edges[b, 0])
            # Penalize predicted edge pixels by their distance from true edges
            loss = (pred_edges[b, 0] * dist_map).mean()
            total_loss += loss
        
        return total_loss / pred.shape[0]