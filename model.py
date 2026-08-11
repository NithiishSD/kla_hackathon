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
        fft = torch.fft.rfft2(x.float(), norm='ortho')  # fp32: not autocast-safe otherwise
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

        self.gdfn_in = nn.Conv2d(channels, channels * 4, 1)
        self.gdfn_dw = nn.Conv2d(channels * 4, channels * 4, 3, padding=1, groups=channels * 4)
        self.gdfn_out = nn.Conv2d(channels * 2, channels, 1)

    def _attn_forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        q = F.normalize(q.view(b, self.num_heads, -1, h * w), dim=-1)
        k = F.normalize(k.view(b, self.num_heads, -1, h * w), dim=-1)
        v = v.view(b, self.num_heads, -1, h * w)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        return self.project_out((attn.softmax(dim=-1) @ v).view(b, c, h, w))

    def forward(self, x):
        # CHANGE: norm1(x) computed once and reused, was being computed
        # twice (once per branch) in the previous version -- same result,
        # free speedup.
        x_norm = self.norm1(x)
        x = x + self._attn_forward(x_norm) + self.spectral(x_norm)

        x1, x2 = self.gdfn_dw(self.gdfn_in(self.norm2(x))).chunk(2, dim=1)
        x = x + self.gdfn_out(F.gelu(x1) * x2)
        return x


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------
class SemiRestoreNet_V2(nn.Module):
    def __init__(self, dim=64, num_blocks=2, scale_factor=2):
        super().__init__()
        self.metrology_gain = nn.Parameter(torch.ones(1))
        self.scale_factor = scale_factor
        self.embed = nn.Conv2d(1, dim, 3, padding=1)

        self.encoder = nn.Sequential(*[MetrologyRestormerBlock(dim) for _ in range(num_blocks)])
        self.bottleneck = MetrologyRestormerBlock(dim)
        self.decoder = nn.Sequential(*[MetrologyRestormerBlock(dim) for _ in range(num_blocks)])

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

        # CHANGE: zero-init the final conv so the residual ("out") starts
        # at exactly zero. Without this, "out" is whatever a randomly
        # initialized network produces at step 0 -- essentially noise added
        # on top of the bicubic baseline -- so training starts from a much
        # worse point than the "global residual learning" framing implies.
        # With this init, forward() at step 0 returns clamp(0 + base, 0, 1)
        # == the bicubic baseline itself, which is a far better starting
        # point and should noticeably raise epoch-1 PSNR.
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, x):
        # x is assumed already normalized to [0, 1] by sem_dataset.py's
        # shared percentile calibration -- no transform applied here.
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
# Loss
# ---------------------------------------------------------------------------
class KLAMetrologyLoss(nn.Module):
    """Charbonnier + edge (Sobel or CD-aware) + optional frequency + optional SSIM loss.
    
    Args:
        edge_weight: Weight for edge loss (Sobel or CD)
        freq_weight: Weight for frequency-domain loss (0 = disabled)
        ssim_weight: Weight for SSIM loss (0 = disabled)
        use_cd_loss: If True, use CDMetrologyLoss instead of Sobel edge loss
        cd_edge_percentile: Percentile threshold for CD edge extraction
        cd_max_distance: Maximum distance clamp for CD loss
        use_multiscale_ssim: If True, compute SSIM at multiple scales
        ms_ssim_scales: List of scales for MS-SSIM (e.g., [1.0, 0.5, 0.25])
        ms_ssim_weights: Weights for each scale in MS-SSIM
        charbonnier_weight: Weight for Charbonnier fidelity loss
    """

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
        self.charbonnier_weight = charbonnier_weight
        
        # Multi-scale SSIM config
        self.use_multiscale_ssim = use_multiscale_ssim
        self.ms_ssim_scales = ms_ssim_scales or [1.0, 0.5, 0.25]
        self.ms_ssim_weights = ms_ssim_weights or [0.5, 0.3, 0.2]
        
        # Sobel kernels
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # CD-aware edge loss
        self.use_profile_loss = use_profile_loss
        self.profile_loss = IntensityProfileLoss(
            num_profiles=num_profiles,
            profile_width=profile_width,
        ) if use_profile_loss else None
        
        # Gaussian windows for SSIM
        self.register_buffer('_ssim_window', self._make_gaussian_window(11, 1.5))
        self.register_buffer('_ssim_window_small', self._make_gaussian_window(7, 1.0))

    @staticmethod
    def _make_gaussian_window(window_size, sigma):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
        return window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

    def charbonnier_loss(self, pred, target, eps=1e-6):
        return torch.mean(torch.sqrt((pred - target) ** 2 + eps))

    def edge_loss(self, pred, target):
        """Original Sobel edge loss — penalizes gradient magnitude differences."""
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
        """SSIM loss at a single scale (extracted for reuse in MS-SSIM)."""
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
        """MS-SSIM: same SSIM computed at multiple resolutions."""
        total = 0.0
        for scale, w in zip(self.ms_ssim_scales, self.ms_ssim_weights):
            if scale == 1.0:
                p, t = pred.float(), target.float()
            else:
                h = int(pred.shape[2] * scale)
                w_new = int(pred.shape[3] * scale)
                p = F.interpolate(pred.float(), size=(h, w_new), mode='bilinear', align_corners=False)
                t = F.interpolate(target.float(), size=(h, w_new), mode='bilinear', align_corners=False)
            
            # Use smaller window for smaller scales
            window = self._ssim_window if scale >= 0.5 else self._ssim_window_small
            total += w * self.ssim_loss_single_scale(p, t, window, data_range)
        return total

    def ssim_loss(self, pred, target, data_range=1.0):
        """Overloaded: calls multi-scale or single-scale based on config."""
        if self.use_multiscale_ssim:
            return self.multiscale_ssim_loss(pred, target, data_range)
        return self.ssim_loss_single_scale(pred, target, self._ssim_window, data_range)

    def forward(self, pred, target):
        l_char = self.charbonnier_weight * self.charbonnier_loss(pred, target)
        loss = l_char
        
        # Only compute edge loss if weight > 0
        if self.edge_weight > 0:
            if self.use_cd_loss and self.cd_loss is not None:
                l_edge = self.cd_loss(pred, target)
            else:
                l_edge = self.edge_loss(pred, target)
            loss = loss + self.edge_weight * l_edge
        
        if self.freq_weight > 0:
            loss = loss + self.freq_weight * self.frequency_loss(pred, target)
        if self.ssim_weight > 0:
            loss = loss + self.ssim_weight * self.ssim_loss(pred, target)
        return loss
class IntensityProfileLoss(nn.Module):
    """
    Metrology-grade loss based on intensity profile matching.
    
    Instead of binarizing edges and computing distances, this loss:
    1. Detects line features via horizontal/vertical projections
    2. Extracts cross-section intensity profiles
    3. Penalizes profile mismatch — which naturally captures CD bias,
       edge slope, and LER in a single, physically meaningful loss.
    
    This is how real CD-SEM metrology algorithms work.
    """
    def __init__(self, num_profiles=32, profile_width=5, edge_zone_weight=2.0):
        super().__init__()
        self.num_profiles = num_profiles
        self.profile_width = profile_width
        self.edge_zone_weight = edge_zone_weight
    
    def extract_profiles(self, img, axis='both'):
        """
        Extract intensity profiles perpendicular to line features.
        
        For SEM images with Manhattan geometry, lines run either
        horizontal or vertical. We take profiles in both directions.
        """
        B, C, H, W = img.shape
        profiles_pred = []
        profiles_target = []
        
        # Horizontal profiles (crossing vertical lines)
        # Sample random rows and extract full-width intensity profiles
        if H > self.num_profiles:
            indices = torch.randperm(H)[:self.num_profiles]
        else:
            indices = torch.arange(H)
        
        for idx in indices:
            # Profile is the intensity along this row
            profiles_pred.append(img[:, :, idx, :])   # (B, C, W)
        
        # Vertical profiles (crossing horizontal lines)
        if W > self.num_profiles:
            indices = torch.randperm(W)[:self.num_profiles]
        else:
            indices = torch.arange(W)
        
        for idx in indices:
            profiles_pred.append(img[:, :, :, idx])   # (B, C, H)
        
        return profiles_pred  # list of (B, C, length) tensors
    
    def profile_gradient_loss(self, pred_profile, target_profile):
        """
        Penalize differences in the GRADIENT of the profile.
        
        The gradient highlights edge transitions. Matching gradients
        means matching edge positions AND edge sharpness simultaneously.
        
        This is the key insight: CD bias shows up as shifted gradient peaks.
        Slope errors show up as broadened/narrowed gradient peaks.
        """
        # Compute gradients along the profile
        pred_grad = pred_profile[:, :, 1:] - pred_profile[:, :, :-1]
        target_grad = target_profile[:, :, 1:] - target_profile[:, :, :-1]
        
        # L1 loss on gradients — directly penalizes edge position mismatch
        return F.l1_loss(pred_grad, target_grad)
    
    def forward(self, pred, target):
        """
        Extract profiles from prediction and target, compute gradient
        matching loss. This naturally enforces CD accuracy without
        fragile binarization or distance transforms.
        """
        B = pred.shape[0]
        
        # Extract profiles from both images
        pred_profiles = self.extract_profiles(pred)
        target_profiles = self.extract_profiles(target)
        
        total_loss = 0.0
        for p_prof, t_prof in zip(pred_profiles, target_profiles):
            total_loss += self.profile_gradient_loss(p_prof, t_prof)
        
        return total_loss / len(pred_profiles)
# ---------------------------------------------------------------------------
# Optimizer with weight decay excluded from norm/bias/temperature params
# ---------------------------------------------------------------------------
def build_optimizer(model, lr=2e-4, weight_decay=1e-4):
    """CHANGE: weight_decay=1e-4 was previously applied uniformly via plain
    AdamW, which also decays GroupNorm affine params and the attention
    temperature scalar -- standard practice excludes 1-D params (norms,
    biases, temperature) from weight decay."""
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
# Standalone sanity check (run `python model.py` directly). Uses a plain
# training step with ReduceLROnPlateau, matching what train.py /
# train_baseline.py / finetune_ssim.py actually use via HardwareEngine --
# this block does NOT reflect a separate scheduler choice, it's just a
# quick check that the model/loss/optimizer wire together correctly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SemiRestoreNet_V2(dim=64, num_blocks=2, scale_factor=2).to(device)
    criterion = KLAMetrologyLoss().to(device)
    optimizer = build_optimizer(model, lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

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
    scheduler.step(loss.item())

    print(f"Sanity check step ok. Loss: {loss.item():.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params / 1e6:.2f}M")