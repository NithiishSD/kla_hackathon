"""
AirNet (Li et al., CVPR 2022, "All-in-One Image Restoration for Unknown
Corruptions") -- chosen specifically for this task's OOD-generalization
requirement, not for speed or raw PSNR like the SwinIR/Restormer
comparison. Core idea: instead of assuming one fixed degradation profile,
a self-supervised Contrastive-Based Degradation Encoder (CBDE) learns a
representation of "what kind of corruption is this image" and conditions
the restoration backbone (DGRN) on it.

Self-supervised, not label-based: positive pairs are two independently
augmented views of the SAME degraded image (same degradation
characteristics); negatives come from other images via a MoCo-style
momentum encoder + queue. This fits this task exactly -- there's no
discrete degradation-type label (unlike AirNet's original rain/haze/noise
setup), only continuously varying noise/texture characteristics across
the diverse data sources and OOD test set.

Reuses KLAMetrologyLoss / build_optimizer from model.py for the
restoration loss, same as swinir_model.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Contrastive-Based Degradation Encoder (CBDE)
# ---------------------------------------------------------------------------
class SmallEncoder(nn.Module):
    """Shared shape for both the online and momentum encoders -- three
    stride-2 convs down to a global-pooled feature vector."""

    def __init__(self, feat_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, feat_dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class Projector(nn.Module):
    """Maps the encoder feature to a smaller, L2-normalized space for the
    contrastive loss only -- the restoration backbone conditions on the
    raw encoder feature, not this projection (standard SimCLR/MoCo split
    between representation and contrastive-loss space)."""

    def __init__(self, feat_dim=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def _random_augment_batch(x):
    """Bit-exact (flip/rot90 only, no interpolation) -- same policy as
    SEMPairDataset._augment, applied here to build the two contrastive
    views instead of at data-loading time."""
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-1])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-2])
    k = torch.randint(0, 4, (1,)).item()
    if k:
        x = torch.rot90(x, k, dims=[-2, -1])
    return x


class CBDE(nn.Module):
    def __init__(self, feat_dim=256, proj_dim=128, queue_size=1024,
                 momentum=0.999, temperature=0.07):
        super().__init__()
        self.feat_dim = feat_dim
        self.m = momentum
        self.T = temperature
        self.K = queue_size

        self.encoder_q = SmallEncoder(feat_dim)
        self.proj_q = Projector(feat_dim, proj_dim)
        self.encoder_k = SmallEncoder(feat_dim)
        self.proj_k = Projector(feat_dim, proj_dim)

        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad_(False)
        for p_q, p_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad_(False)

        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def encode(self, x):
        """Online encoder -- gradient flows, feat conditions the restoration
        backbone, proj is used only for the contrastive loss."""
        feat = self.encoder_q(x)
        proj = self.proj_q(feat)
        return feat, proj

    @torch.no_grad()
    def _encode_key(self, x):
        feat = self.encoder_k(x)
        proj = self.proj_k(feat)
        return feat, proj

    @torch.no_grad()
    def _momentum_update(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1.0 - self.m)
        for p_q, p_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            first = self.K - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :batch_size - first] = keys[first:].T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def contrastive_loss(self, x1, x2):
        """x1, x2: two independently augmented views of the same batch of
        degraded images. Standard MoCo v2 InfoNCE -- also updates the
        momentum encoder and queue as a side effect."""
        _, q = self.encode(x1)
        self._momentum_update()
        _, k = self._encode_key(x2)

        l_pos = torch.sum(q * k, dim=1, keepdim=True)
        l_neg = q.float() @ self.queue.clone().detach().float()
        logits = torch.cat([l_pos.float(), l_neg], dim=1) / self.T
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)

        self._dequeue_and_enqueue(k)
        return loss


# ---------------------------------------------------------------------------
# Degradation-Guided Restoration Network (DGRN)
# ---------------------------------------------------------------------------
class DGFT(nn.Module):
    """Degradation-guided Feature Transform: FiLM-style per-channel affine
    modulation of conv features, conditioned on the degradation feature."""

    def __init__(self, feat_dim, channels):
        super().__init__()
        self.to_scale_shift = nn.Linear(feat_dim, channels * 2)

    def forward(self, x, deg_feat):
        gamma, beta = self.to_scale_shift(deg_feat).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class DGRB(nn.Module):
    """Degradation-Guided Restoration Block: conv -> DGFT -> ReLU -> conv
    -> DGFT, residual."""

    def __init__(self, channels, feat_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dgft1 = DGFT(feat_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dgft2 = DGFT(feat_dim, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, deg_feat):
        res = self.act(self.dgft1(self.conv1(x), deg_feat))
        res = self.dgft2(self.conv2(res), deg_feat)
        return x + res


class DGRN(nn.Module):
    def __init__(self, dim=64, num_blocks=6, feat_dim=256, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv_first = nn.Conv2d(1, dim, 3, padding=1)
        self.blocks = nn.ModuleList([DGRB(dim, feat_dim) for _ in range(num_blocks)])
        self.conv_after_body = nn.Conv2d(dim, dim, 3, padding=1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(dim, dim * scale_factor ** 2, 3, padding=1),
            nn.PixelShuffle(scale_factor),
        )
        self.conv_last = nn.Conv2d(dim, 1, 3, padding=1)
        # Same zero-init trick as the Restormer/SwinIR models -- step-0
        # output is the bicubic baseline exactly, not noise on top of it.
        nn.init.zeros_(self.conv_last.weight)
        nn.init.zeros_(self.conv_last.bias)

    def forward(self, x, deg_feat):
        feat = self.conv_first(x)
        res = feat
        for blk in self.blocks:
            res = blk(res, deg_feat)
        res = self.conv_after_body(res) + feat

        out = self.conv_last(self.upsampler(res))
        base = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        return torch.clamp(out + base, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------
class AirNet(nn.Module):
    def __init__(self, dim=64, num_blocks=6, feat_dim=256, proj_dim=128,
                 queue_size=1024, momentum=0.999, temperature=0.07, scale_factor=2):
        super().__init__()
        self.cbde = CBDE(feat_dim, proj_dim, queue_size, momentum, temperature)
        self.dgrn = DGRN(dim, num_blocks, feat_dim, scale_factor)

    def forward(self, x):
        """Restoration only -- what evaluate()/infer.py actually call."""
        feat = self.cbde.encoder_q(x)
        return self.dgrn(x, feat)

    def contrastive_loss(self, x):
        """Training-time auxiliary loss. Builds two augmented views of x
        internally -- doesn't affect the restoration forward pass above."""
        v1 = _random_augment_batch(x)
        v2 = _random_augment_batch(x)
        return self.cbde.contrastive_loss(v1, v2)


# ---------------------------------------------------------------------------
# Standalone sanity check (run `python airnet_model.py` directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from model import KLAMetrologyLoss, build_optimizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AirNet(dim=64, num_blocks=6, feat_dim=256, proj_dim=128,
                    queue_size=256, scale_factor=2).to(device)
    criterion = KLAMetrologyLoss().to(device)
    optimizer = build_optimizer(model, lr=2e-4, weight_decay=1e-4)

    dummy_lr = torch.rand(4, 1, 128, 128, device=device)
    dummy_gt = torch.rand(4, 1, 256, 256, device=device)

    optimizer.zero_grad(set_to_none=True)
    if device == "cuda":
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            restored_img = model(dummy_lr)
            restoration_loss = criterion(restored_img, dummy_gt)
            contrastive_loss = model.contrastive_loss(dummy_lr)
            loss = restoration_loss + 0.1 * contrastive_loss
    else:
        restored_img = model(dummy_lr)
        restoration_loss = criterion(restored_img, dummy_gt)
        contrastive_loss = model.contrastive_loss(dummy_lr)
        loss = restoration_loss + 0.1 * contrastive_loss
    loss.backward()
    optimizer.step()

    print(f"Sanity check step ok. Output shape: {restored_img.shape}.")
    print(f"Restoration loss: {restoration_loss.item():.4f}, "
          f"Contrastive loss: {contrastive_loss.item():.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    n_params_no_momentum = n_params - sum(
        p.numel() for p in list(model.cbde.encoder_k.parameters()) + list(model.cbde.proj_k.parameters()))
    print(f"Total params (incl. momentum copies): {n_params / 1e6:.2f}M")
    print(f"Trainable-relevant params (excl. momentum copies): {n_params_no_momentum / 1e6:.2f}M")
