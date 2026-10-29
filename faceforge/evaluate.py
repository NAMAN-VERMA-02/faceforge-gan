"""Evaluation: FID and attribute-control accuracy.

FID (Frechet Inception Distance) compares Inception-v3 feature statistics of real and
generated images. Lower is better; a well-trained 64x64 DCGAN on CelebA lands roughly
in the 15-40 range.

    python -m faceforge.evaluate --checkpoint runs/faceforge/generator.pt
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Dataset

from faceforge.config import Config


def _to_uint8(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


@torch.no_grad()
def compute_fid(G: torch.nn.Module, real_ds: Dataset, cfg: Config, device: torch.device,
                n: int | None = None, batch_size: int = 250) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance

    n = min(n or cfg.fid_samples, len(real_ds))
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    G.eval()

    seen = 0
    real_labels = []
    for x, y in DataLoader(real_ds, batch_size=batch_size, shuffle=True):
        x = x[: n - seen]
        fid.update(_to_uint8(x).to(device), real=True)
        real_labels.append(y[: len(x)])
        seen += len(x)
        if seen >= n:
            break

    # condition generated images on the same label distribution as the real set
    labels = torch.cat(real_labels)
    for i in range(0, n, batch_size):
        y = labels[i : i + batch_size].to(device)
        z = torch.randn(len(y), cfg.z_dim, device=device)
        fid.update(_to_uint8(G(z, y)), real=False)

    score = float(fid.compute())
    print(f"  FID ({n} samples): {score:.2f}")
    return score


def main() -> None:
    from faceforge.data import CelebAFaces, build_cache
    from faceforge.inference import load_generator

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/faceforge/generator.pt")
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--out", default=None, help="optional JSON file to write the score to")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, cfg = load_generator(args.checkpoint, device)
    ds = CelebAFaces(build_cache(cfg.data_dir, cfg.attributes, cfg.num_shards, cfg.image_size), hflip=False)
    score = compute_fid(G, ds, cfg, device, n=args.samples)
    if args.out:
        import json

        with open(args.out, "w") as f:
            json.dump({"fid": round(score, 2), "samples": min(args.samples, len(ds))}, f, indent=2)


if __name__ == "__main__":
    main()
