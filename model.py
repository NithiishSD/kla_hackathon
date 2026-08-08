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

        return torch.clamp(out + base, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class KLAMetrologyLoss(nn.Module):
    """Charbonnier + edge (Sobel) loss. If you want the Fourier Unit's
    frequency-domain filtering to actually be supervised (rather than only
    implicitly learned via the pixel/edge losses), add frequency_loss back
    in -- see the commented-out block below."""

    def __init__(self, edge_weight=0.5, freq_weight=0.0):
        super().__init__()
        self.edge_weight = edge_weight
        self.freq_weight = freq_weight  # 0.0 by default; set >0 to re-enable
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

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

    def forward(self, pred, target):
        l_char = self.charbonnier_loss(pred, target)
        l_edge = self.edge_loss(pred, target)
        loss = l_char + self.edge_weight * l_edge
        if self.freq_weight > 0:
            loss = loss + self.freq_weight * self.frequency_loss(pred, target)
        return loss


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
# H100 training step (bf16, no GradScaler)
# ---------------------------------------------------------------------------
def train_one_step(model, optimizer, criterion, lr_scheduler, input_img, gt_img):
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        restored_img = model(input_img)
        loss = criterion(restored_img, gt_img)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    return loss.item()


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SemiRestoreNet_V2(dim=64, num_blocks=2, scale_factor=2).to(device)
    criterion = KLAMetrologyLoss().to(device)
    optimizer = build_optimizer(model, lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5,     # Reduce LR by half
    patience=3,     # Wait 3 epochs before dropping
    verbose=True
)

    dummy_lr = torch.rand(2, 1, 128, 128, device=device)
    dummy_gt = torch.rand(2, 1, 256, 256, device=device)

    loss_val = train_one_step(model, optimizer, criterion, scheduler, dummy_lr, dummy_gt)
    print(f"Sanity check step ok. Loss: {loss_val:.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params / 1e6:.2f}M")