"""Inference helpers shared by the CLI, the Gradio app and the tests."""

from __future__ import annotations

from pathlib import Path

import torch

from faceforge.config import Config
from faceforge.models import Generator

DEFAULT_HUB_REPO = "e-NV-y/faceforge-gan"  # change to your own Hugging Face model repo


def load_generator(path_or_repo: str | Path, device: torch.device | str = "cpu") -> tuple[Generator, Config]:
    """Load a generator from a local `generator.pt` or a Hugging Face model repo id."""
    path = Path(path_or_repo)
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(str(path_or_repo), "generator.pt"))
    ck = torch.load(path, map_location=device, weights_only=False)
    raw = ck["config"]
    cfg = Config(**{k: (tuple(v) if k == "betas" else v) for k, v in raw.items()})
    G = Generator(cfg.z_dim, cfg.num_classes, cfg.g_channels).to(device)
    G.load_state_dict(ck["state_dict"])
    G.eval()
    return G, cfg


@torch.no_grad()
def generate(G: Generator, attrs: torch.Tensor, seed: int = 0, n: int = 1, truncation: float = 1.0) -> torch.Tensor:
    """Generate `n` faces for one attribute vector. Returns uint8 tensor N x H x W x 3.

    `truncation` < 1 shrinks z toward the origin: less variety, higher fidelity.
    """
    device = next(G.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(n, G.z_dim, generator=g).mul_(truncation).to(device)
    y = attrs.float().reshape(1, -1).expand(n, -1).to(device)
    return to_uint8_hwc(G(z, y))


@torch.no_grad()
def interpolate(G: Generator, attrs_a: torch.Tensor, attrs_b: torch.Tensor, seed_a: int, seed_b: int,
                steps: int = 8) -> torch.Tensor:
    """Walk linearly between two (z, attributes) points; returns steps x H x W x 3 uint8."""
    device = next(G.parameters()).device
    za = torch.randn(1, G.z_dim, generator=torch.Generator().manual_seed(seed_a))
    zb = torch.randn(1, G.z_dim, generator=torch.Generator().manual_seed(seed_b))
    t = torch.linspace(0, 1, steps).unsqueeze(1)
    # spherical interpolation keeps intermediate z at a typical Gaussian norm
    z = slerp(za.expand(steps, -1), zb.expand(steps, -1), t)
    y = (1 - t) * attrs_a.float().reshape(1, -1) + t * attrs_b.float().reshape(1, -1)
    return to_uint8_hwc(G(z.to(device), y.to(device)))


def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    a_n, b_n = a / a.norm(dim=1, keepdim=True), b / b.norm(dim=1, keepdim=True)
    omega = torch.acos((a_n * b_n).sum(1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7))
    so = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


def to_uint8_hwc(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu()
