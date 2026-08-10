"""
SwinIR backbone for the same task as model.py's SemiRestoreNet_V2 (joint
denoise + scale_factor-x SR on 1-channel images), swapped in as an
alternative architecture -- shifted-window self-attention instead of
Restormer's channel attention, to compare a genuinely different inductive
bias on the same data/loss/training pipeline.

Reuses KLAMetrologyLoss / build_optimizer from model.py rather than
duplicating them -- they're architecture-agnostic.

Same global-residual + zero-init-last-conv trick as V2: forward() at
step 0 returns clamp(0 + bicubic(x), 0, 1), not noise, so training starts
from the bicubic baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import KLAMetrologyLoss, build_optimizer  # noqa: F401 (re-exported for train_swinir.py)


def to_2tuple(x):
    return (x, x) if isinstance(x, int) else tuple(x)


# ---------------------------------------------------------------------------
# Window utilities
# ---------------------------------------------------------------------------
def window_partition(x, window_size):
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """(num_windows*B, window_size, window_size, C) -> (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ---------------------------------------------------------------------------
# Windowed multi-head self-attention with relative position bias
# ---------------------------------------------------------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size[0]), torch.arange(window_size[1]), indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SwinTransformerBlock(nn.Module):
    """W-MSA on even blocks, SW-MSA (shifted) on odd blocks -- standard
    Swin alternation so windows see across their own boundaries every
    other block instead of being permanently blind to neighboring windows."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        assert 0 <= shift_size < window_size
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, to_2tuple(window_size), num_heads,
                                     qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def _attn_mask(self, x_size, device):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1), device=device)
        slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for h in slices:
            for w in slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 2)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._attn_mask(x_size, x.device)
        else:
            attn_mask = None

        x_windows = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        attn_windows = self.attn(x_windows, mask=attn_mask).view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, H * W, C)

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class BasicLayer(nn.Module):
    """One RSTB's stack of Swin blocks (alternating shift)."""

    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio, qkv_bias, drop, attn_drop):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, num_heads, window_size,
                                  shift_size=0 if i % 2 == 0 else window_size // 2,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  drop=drop, attn_drop=attn_drop)
            for i in range(depth)
        ])

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block: BasicLayer + conv, with a residual
    skip around the whole thing -- the conv restores local/translation-
    equivariant inductive bias that pure window attention lacks."""

    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio, qkv_bias, drop, attn_drop):
        super().__init__()
        self.residual_group = BasicLayer(dim, depth, num_heads, window_size, mlp_ratio,
                                          qkv_bias, drop, attn_drop)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        res = self.residual_group(x, x_size)
        res = res.transpose(1, 2).contiguous().view(B, C, H, W)
        res = self.conv(res).flatten(2).transpose(1, 2)
        return res + x


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------
class SwinIR(nn.Module):
    def __init__(self, embed_dim=60, depths=(4, 4, 4, 4), num_heads=(6, 6, 6, 6),
                 window_size=8, mlp_ratio=2., qkv_bias=True, scale_factor=2):
        super().__init__()
        self.window_size = window_size
        self.scale_factor = scale_factor

        self.conv_first = nn.Conv2d(1, embed_dim, 3, 1, 1)

        self.layers = nn.ModuleList([
            RSTB(embed_dim, depths[i], num_heads[i], window_size, mlp_ratio, qkv_bias, 0., 0.)
            for i in range(len(depths))
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * scale_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(scale_factor),
        )
        self.conv_last = nn.Conv2d(embed_dim, 1, 3, 1, 1)

        # Same zero-init trick as SemiRestoreNet_V2 -- step-0 output is the
        # bicubic baseline exactly, not noise on top of it.
        nn.init.zeros_(self.conv_last.weight)
        nn.init.zeros_(self.conv_last.bias)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _pad_to_window(self, x):
        """Window attention needs H, W divisible by window_size. Reflect-pad
        up to the next multiple; forward() crops the padding back off at
        the end (scaled up by scale_factor)."""
        _, _, h, w = x.shape
        m = self.window_size
        pad_h, pad_w = (m - h % m) % m, (m - w % m) % m
        return F.pad(x, (0, pad_w, 0, pad_h), mode='reflect'), pad_h, pad_w

    def forward(self, x):
        h_in, w_in = x.shape[2], x.shape[3]
        x_pad, _, _ = self._pad_to_window(x)
        x_size = (x_pad.shape[2], x_pad.shape[3])

        feat = self.conv_first(x_pad)
        deep = feat.flatten(2).transpose(1, 2)  # B,C,H,W -> B,HW,C
        for layer in self.layers:
            deep = layer(deep, x_size)
        deep = self.norm(deep).transpose(1, 2).contiguous().view(feat.shape)
        deep = self.conv_after_body(deep) + feat

        out = self.conv_last(self.upsample(deep))
        base = F.interpolate(x_pad, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        out = torch.clamp(out + base, 0.0, 1.0)

        H_out, W_out = h_in * self.scale_factor, w_in * self.scale_factor
        return out[:, :, :H_out, :W_out]


# ---------------------------------------------------------------------------
# Standalone sanity check (run `python swinir_model.py` directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SwinIR(embed_dim=60, depths=(4, 4, 4, 4), num_heads=(6, 6, 6, 6),
                    window_size=8, mlp_ratio=2., scale_factor=2).to(device)
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

    print(f"Sanity check step ok. Output shape: {restored_img.shape}. Loss: {loss.item():.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params / 1e6:.2f}M")
