"""Conditional DCGAN.

Generator:     (z, attribute vector) -> 3x64x64 image in [-1, 1]
Discriminator: (image, attribute vector) -> real/fake logit

Conditioning works differently on each side:
* The generator concatenates a learned embedding of the attribute vector with z.
* The discriminator uses a *projection* term (Miyato & Koyama, 2018): the logit is
  psi(phi(x)) + <embed(y), phi(x)>. It is more stable than concatenating label maps
  and makes D check that the image actually matches the requested attributes.
Spectral normalisation in D keeps its Lipschitz constant bounded, which removes most
of the mode-collapse and oscillation problems of a vanilla DCGAN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def init_weights(m: nn.Module) -> None:
    """DCGAN initialisation: N(0, 0.02) for convs, N(1, 0.02) for BatchNorm scale."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, num_classes: int = 6, ch: int = 64, embed_dim: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.label_embed = nn.Sequential(nn.Linear(num_classes, embed_dim), nn.LeakyReLU(0.2))

        def up(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(True),
            )

        self.net = nn.Sequential(
            # (z + embed) x 1 x 1 -> 8ch x 4 x 4
            nn.ConvTranspose2d(z_dim + embed_dim, ch * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(True),
            up(ch * 8, ch * 4),   # 8x8
            up(ch * 4, ch * 2),   # 16x16
            up(ch * 2, ch),       # 32x32
            nn.ConvTranspose2d(ch, 3, 4, 2, 1),  # 64x64
            nn.Tanh(),
        )
        self.apply(init_weights)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, self.label_embed(y)], dim=1)
        return self.net(h[:, :, None, None])


class Discriminator(nn.Module):
    def __init__(self, num_classes: int = 6, ch: int = 64):
        super().__init__()

        def down(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(spectral_norm(nn.Conv2d(cin, cout, 4, 2, 1)), nn.LeakyReLU(0.2, True))

        self.features = nn.Sequential(
            down(3, ch),            # 32x32
            down(ch, ch * 2),       # 16x16
            down(ch * 2, ch * 4),   # 8x8
            down(ch * 4, ch * 8),   # 4x4
        )
        feat_dim = ch * 8 * 4 * 4
        self.fc = spectral_norm(nn.Linear(feat_dim, 1))
        self.proj = spectral_norm(nn.Linear(num_classes, feat_dim, bias=False))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        out = self.fc(h).squeeze(1)
        out = out + (self.proj(y) * h).sum(dim=1)  # projection conditioning
        return out


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    """Exponential moving average of weights; EMA generators give visibly cleaner samples."""
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for b_ema, b in zip(ema.buffers(), model.buffers()):
        b_ema.copy_(b)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
