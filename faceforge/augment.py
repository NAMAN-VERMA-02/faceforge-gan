"""DiffAugment (Zhao et al., "Differentiable Augmentation for Data-Efficient GAN Training", NeurIPS 2020).

The same random, differentiable augmentations are applied to real AND generated images
right before the discriminator sees them. D can no longer memorise the (small) training
set, and because the ops are differentiable, gradients still flow back into G.
Works on images in [-1, 1], shape B x C x H x W.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + mean


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + mean


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    b, _, h, w = x.shape
    sx, sy = int(h * ratio + 0.5), int(w * ratio + 0.5)
    tx = torch.randint(-sx, sx + 1, size=[b, 1, 1], device=x.device)
    ty = torch.randint(-sy, sy + 1, size=[b, 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(b, device=x.device), torch.arange(h, device=x.device), torch.arange(w, device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    x_pad = F.pad(x, [1, 1, 1, 1])
    return x_pad.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    b, _, h, w = x.shape
    ch, cw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    ox = torch.randint(0, h + (1 - ch % 2), size=[b, 1, 1], device=x.device)
    oy = torch.randint(0, w + (1 - cw % 2), size=[b, 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(b, device=x.device), torch.arange(ch, device=x.device), torch.arange(cw, device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + ox - ch // 2, min=0, max=h - 1)
    gy = torch.clamp(gy + oy - cw // 2, min=0, max=w - 1)
    mask = torch.ones(b, h, w, dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def diff_augment(x: torch.Tensor, policy: str = "color,translation,cutout") -> torch.Tensor:
    """Apply the comma-separated augmentation policy. An empty policy returns x unchanged."""
    for name in [p for p in policy.split(",") if p]:
        for fn in AUGMENT_FNS[name]:
            x = fn(x)
    return x.contiguous()
