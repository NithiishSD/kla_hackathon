"""
SemiRestoreNet_V2 - model + loss.
Normalization now lives entirely in sem_dataset.py (shared global
percentile calibration) -- no in-model log transform, so there's no
log/linear-space mismatch to worry about.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fourier Unit
# ---------------------------------------------------------------------------
class FourierUnit(nn.Module):
    """Global frequency-domain mixer. Learned, not a hand-crafted notch
    filter -- gives cheap global receptive field, still needs training
    signal to actually suppress periodic scan-line noise."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, 1)
        self.norm = nn.GroupNorm(1, channels * 2)

    def forward(self, x):
        B, C, H, W = x.shape
        fft = torch.fft.rfft2(x.float(), norm='ortho')
        fft_real = torch.stack([fft.real, fft.imag], dim=1)
        fft_filt = self.norm(self.conv(fft_real.view(B, 2 * C, H, -1)))
        fft_filt = fft_filt.view(B, 2, C, H, -1)
        fft_final = torch.complex(fft_filt[:, 0], fft_filt[:, 1])
        return torch.fft.irfft2(fft_final, s=(H, W), norm='ortho').to(x.dtype)


# ---------------------------------------------------------------------------
# Restormer-style block
# ---------------------------------------------------------------------------
class MetrologyRestormerBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.qkv_dw = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3)
        self.project_out = nn.Conv2d(channels, channels, 1)
        self.spectral = FourierUnit(channels)

        # GDFN expansion: use 2x for large dim to save VRAM
        expansion = 2 if channels >= 128 else 4
        self.gdfn_in = nn.Conv2d(channels, channels * expansion, 1)
        self.gdfn_dw = nn.Conv2d(channels * expansion, channels * expansion, 3, padding=1, groups=channels * expansion)
        self.gdfn_out = nn.Conv2d(channels * expansion // 2, channels, 1)

    def _attn_forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        q = F.normalize(q.view(b, self.num_heads, -1, h * w), dim=-1)
        k = F.normalize(k.view(b, self.num_heads, -1, h * w), dim=-1)
        v = v.view(b, self.num_heads, -1, h * w)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        return self.project_out((attn.softmax(dim=-1) @ v).view(b, c, h, w))

    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self._attn_forward(x_norm) + self.spectral(x_norm)

        x1, x2 = self.gdfn_dw(self.gdfn_in(self.norm2(x))).chunk(2, dim=1)
        x = x + self.gdfn_out(F.gelu(x1) * x2)
        return x


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------
class SemiRestoreNet_V2(nn.Module):
    def __init__(self, dim=64, num_blocks=2, scale_factor=2, num_heads=8):
        super().__init__()
        self.metrology_gain = nn.Parameter(torch.ones(1))
        self.scale_factor = scale_factor
        self.embed = nn.Conv2d(1, dim, 3, padding=1)

        self.encoder = nn.Sequential(
            *[MetrologyRestormerBlock(dim, num_heads=num_heads) for _ in range(num_blocks)]
        )
        self.bottleneck = MetrologyRestormerBlock(dim, num_heads=num_heads)
        self.decoder = nn.Sequential(
            *[MetrologyRestormerBlock(dim, num_heads=num_heads) for _ in range(num_blocks)]
        )

        if scale_factor > 1:
            self.upsampler = nn.Sequential(
                nn.Conv2d(dim, dim * (scale_factor ** 2), 3, padding=1),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(dim, 1, 3, padding=1),
            )
            last_conv = self.upsampler[-1]
        else:
            self.upsampler = nn.Conv2d(dim, 1, 3, padding=1)
            last_conv = self.upsampler

        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, x):
        feat = self.embed(x)
        res = self.encoder(feat)
        res = self.bottleneck(res)
        res = self.decoder(res + feat)
        out = self.upsampler(res)

        if self.scale_factor > 1:
            base = F.interpolate(x, scale_factor=self.scale_factor,
                                  mode='bicubic', align_corners=False)
        else:
            base = x

        return torch.clamp((out + base) * self.metrology_gain, 0.0, 1.0)


# ---------------------------------------------------------------------------
# VGG Edge Loss
# ---------------------------------------------------------------------------
class VGGEdgeLoss(nn.Module):
    """Perceptual loss using VGG16 early layers for edge sensitivity."""
    def __init__(self, device='cuda'):
        super().__init__()
        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:6]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
        self.vgg.eval()
    
    def forward(self, pred, target):
        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)
        pred_feat = self.vgg(pred_rgb)
        target_feat = self.vgg(target_rgb)
        return F.l1_loss(pred_feat, target_feat)


# ---------------------------------------------------------------------------
# Intensity Profile Loss
# ---------------------------------------------------------------------------
class IntensityProfileLoss(nn.Module):
    """Metrology-grade loss based on intensity profile gradient matching."""
    def __init__(self, num_profiles=32, profile_width=5, edge_zone_weight=2.0):
        super().__init__()
        self.num_profiles = num_profiles
        self.profile_width = profile_width
        self.edge_zone_weight = edge_zone_weight
    
    def extract_profiles(self, img):
        B, C, H, W = img.shape
        profiles = []
        if H > self.num_profiles:
            indices = torch.randperm(H)[:self.num_profiles]
        else:
            indices = torch.arange(H)
        for idx in indices:
            profiles.append(img[:, :, idx, :])
        if W > self.num_profiles:
            indices = torch.randperm(W)[:self.num_profiles]
        else:
            indices = torch.arange(W)
        for idx in indices:
            profiles.append(img[:, :, :, idx])
        return profiles
    
    def profile_gradient_loss(self, pred_profile, target_profile):
        pred_grad = pred_profile[:, :, 1:] - pred_profile[:, :, :-1]
        target_grad = target_profile[:, :, 1:] - target_profile[:, :, :-1]
        return F.l1_loss(pred_grad, target_grad)
    
    def forward(self, pred, target):
        pred_profiles = self.extract_profiles(pred)
        target_profiles = self.extract_profiles(target)
        total_loss = 0.0
        for p_prof, t_prof in zip(pred_profiles, target_profiles):
            total_loss += self.profile_gradient_loss(p_prof, t_prof)
        return total_loss / len(pred_profiles)


# ---------------------------------------------------------------------------
# CDMetrologyLoss (deprecated - kept for backward compatibility)
# ---------------------------------------------------------------------------
class CDMetrologyLoss(nn.Module):
    """Thresholded edge-position loss. DEPRECATED - use IntensityProfileLoss."""
    def __init__(self, edge_percentile=90.0, max_distance=30.0):
        super().__init__()
        self.edge_percentile = edge_percentile
        self.max_distance = max_distance
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def extract_edges(self, img):
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        B = grad_mag.shape[0]
        edges = []
        for b in range(B):
            flat = grad_mag[b].reshape(-1)
            k = max(1, int(flat.numel() * (1.0 - self.edge_percentile / 100.0)))
            thresh = flat.topk(k, largest=True).values[-1]
            edges.append((grad_mag[b] >= thresh).float().unsqueeze(0))
        return torch.cat(edges, dim=0)
    
    def _distance_transform(self, edge_map):
        inv = 1.0 - edge_map
        kernel = torch.ones(1, 1, 3, 3, device=edge_map.device)
        dist = torch.zeros_like(inv)
        current = inv.clone()
        for d in range(1, int(self.max_distance) + 1):
            dilated = F.conv2d(current, kernel, padding=1)
            dilated = (dilated > 0).float()
            newly_reached = (dilated - current) > 0
            dist[newly_reached] = d
            current = dilated
        dist = torch.clamp(dist, 0, self.max_distance)
        return dist
    
    def forward(self, pred, target):
        pred_edges = self.extract_edges(pred)
        target_edges = self.extract_edges(target)
        total_loss = 0.0
        for b in range(pred.shape[0]):
            dist_to_target = self._distance_transform(target_edges[b:b+1])
            loss_pred_to_target = (pred_edges[b:b+1] * dist_to_target).mean()
            dist_to_pred = self._distance_transform(pred_edges[b:b+1])
            loss_target_to_pred = (target_edges[b:b+1] * dist_to_pred).mean()
            total_loss += (loss_pred_to_target + loss_target_to_pred)
        return total_loss / pred.shape[0]


# ---------------------------------------------------------------------------
# KLAMetrologyLoss - main loss class
# ---------------------------------------------------------------------------
class KLAMetrologyLoss(nn.Module):
    """Charbonnier + edge (Sobel/CD/Profile) + optional frequency + optional SSIM loss."""

    def __init__(self, edge_weight=0.5, freq_weight=0.0, ssim_weight=0.0,
                 use_cd_loss=False, cd_edge_percentile=90.0, cd_max_distance=30.0,
                 use_profile_loss=False, num_profiles=32, profile_width=5,
                 use_multiscale_ssim=False, ms_ssim_scales=None, ms_ssim_weights=None,
                 charbonnier_weight=1.0):
        super().__init__()
        self.edge_weight = edge_weight
        self.freq_weight = freq_weight
        self.ssim_weight = ssim_weight
        self.use_cd_loss = use_cd_loss
        self.use_profile_loss = use_profile_loss
        self.charbonnier_weight = charbonnier_weight
        
        self.use_multiscale_ssim = use_multiscale_ssim
        self.ms_ssim_scales = ms_ssim_scales or [1.0, 0.5, 0.25]
        self.ms_ssim_weights = ms_ssim_weights or [0.5, 0.3, 0.2]
        
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.cd_loss = CDMetrologyLoss(
            edge_percentile=cd_edge_percentile,
            max_distance=cd_max_distance
        ) if use_cd_loss else None
        
        self.profile_loss = IntensityProfileLoss(
            num_profiles=num_profiles,
            profile_width=profile_width,
        ) if use_profile_loss else None
        
        self.register_buffer('_ssim_window', self._make_gaussian_window(11, 1.5))
        self.register_buffer('_ssim_window_small', self._make_gaussian_window(7, 1.0))

    @staticmethod
    def _make_gaussian_window(window_size, sigma):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
        return window_2d.unsqueeze(0).unsqueeze(0)

    def charbonnier_loss(self, pred, target, eps=1e-6):
        return torch.mean(torch.sqrt((pred - target) ** 2 + eps))

    def edge_loss(self, pred, target):
        pred_f, target_f = pred.float(), target.float()
        p_edge = F.conv2d(pred_f, self.sobel_x, padding=1) ** 2 + \
                 F.conv2d(pred_f, self.sobel_y, padding=1) ** 2
        t_edge = F.conv2d(target_f, self.sobel_x, padding=1) ** 2 + \
                 F.conv2d(target_f, self.sobel_y, padding=1) ** 2
        return F.l1_loss(p_edge, t_edge)

    def frequency_loss(self, pred, target):
        pred_fft = torch.fft.rfft2(pred.float(), norm='ortho')
        gt_fft = torch.fft.rfft2(target.float(), norm='ortho')
        return F.l1_loss(torch.abs(pred_fft), torch.abs(gt_fft))

    def ssim_loss_single_scale(self, pred, target, window, data_range=1.0):
        pred_f, target_f = pred.float(), target.float()
        pad = window.shape[-1] // 2
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        mu1 = F.conv2d(pred_f, window, padding=pad)
        mu2 = F.conv2d(target_f, window, padding=pad)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1_sq = F.conv2d(pred_f * pred_f, window, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(target_f * target_f, window, padding=pad) - mu2_sq
        sigma12 = F.conv2d(pred_f * target_f, window, padding=pad) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()

    def multiscale_ssim_loss(self, pred, target, data_range=1.0):
        total = 0.0
        for scale, w in zip(self.ms_ssim_scales, self.ms_ssim_weights):
            if scale == 1.0:
                p, t = pred.float(), target.float()
            else:
                h = int(pred.shape[2] * scale)
                w_new = int(pred.shape[3] * scale)
                p = F.interpolate(pred.float(), size=(h, w_new), mode='bilinear', align_corners=False)
                t = F.interpolate(target.float(), size=(h, w_new), mode='bilinear', align_corners=False)
            window = self._ssim_window if scale >= 0.5 else self._ssim_window_small
            total += w * self.ssim_loss_single_scale(p, t, window, data_range)
        return total

    def ssim_loss(self, pred, target, data_range=1.0):
        if self.use_multiscale_ssim:
            return self.multiscale_ssim_loss(pred, target, data_range)
        return self.ssim_loss_single_scale(pred, target, self._ssim_window, data_range)

    def forward(self, pred, target):
        l_char = self.charbonnier_weight * self.charbonnier_loss(pred, target)
        loss = l_char
        
        if self.edge_weight > 0:
            if self.use_profile_loss and self.profile_loss is not None:
                l_edge = self.profile_loss(pred, target)
            elif self.use_cd_loss and self.cd_loss is not None:
                l_edge = self.cd_loss(pred, target)
            else:
                l_edge = self.edge_loss(pred, target)
            loss = loss + self.edge_weight * l_edge
        
        if self.freq_weight > 0:
            loss = loss + self.freq_weight * self.frequency_loss(pred, target)
        if self.ssim_weight > 0:
            loss = loss + self.ssim_weight * self.ssim_loss(pred, target)
        return loss


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def build_optimizer(model, lr=2e-4, weight_decay=1e-4):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or 'norm' in n or 'temperature' in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=lr)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test standard model
    model = SemiRestoreNet_V2(dim=64, num_blocks=2, scale_factor=2).to(device)
    print(f"Standard model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Test scaled model
    model_scaled = SemiRestoreNet_V2(dim=128, num_blocks=4, scale_factor=2).to(device)
    print(f"Scaled model params: {sum(p.numel() for p in model_scaled.parameters())/1e6:.2f}M")
    
    criterion = KLAMetrologyLoss().to(device)
    optimizer = build_optimizer(model, lr=2e-4, weight_decay=1e-4)

    dummy_lr = torch.rand(2, 1, 128, 128, device=device)
    dummy_gt = torch.rand(2, 1, 256, 256, device=device)

    optimizer.zero_grad(set_to_none=True)
    if device == "cuda":
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            restored_img = model(dummy_lr)
            loss = criterion(restored_img, dummy_gt)
    else:
        restored_img = model(dummy_lr)
        loss = criterion(restored_img, dummy_gt)
    loss.backward()
    optimizer.step()

    print(f"Sanity check ok. Loss: {loss.item():.4f}")