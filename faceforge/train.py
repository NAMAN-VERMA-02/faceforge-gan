"""Train the conditional DCGAN.

    python -m faceforge.train --epochs 25 --num-shards 6          # real CelebA (GPU)
    python -m faceforge.train --fake-data --epochs 1 --batch-size 16  # CPU smoke test
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from faceforge.config import Config
from faceforge.models import Discriminator, Generator, count_params, update_ema


# ---------------------------------------------------------------- losses
def d_loss_fn(real_logits: torch.Tensor, fake_logits: torch.Tensor, kind: str, smooth: float) -> torch.Tensor:
    if kind == "hinge":
        return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
    real_t = torch.full_like(real_logits, smooth)  # one-sided label smoothing
    return F.binary_cross_entropy_with_logits(real_logits, real_t) + F.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )


def g_loss_fn(fake_logits: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "hinge":
        return -fake_logits.mean()
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


# ---------------------------------------------------------------- helpers
def fixed_sample_inputs(cfg: Config, device: torch.device, n_z: int = 8):
    """A grid where each row is one attribute setting and each column one fixed z.
    Reading down a column shows the same 'person' with different attributes."""
    g = torch.Generator().manual_seed(cfg.seed)
    z = torch.randn(n_z, cfg.z_dim, generator=g)
    rows = [torch.zeros(cfg.num_classes)]           # all attributes off
    for i in range(cfg.num_classes):                 # one attribute on at a time
        y = torch.zeros(cfg.num_classes)
        y[i] = 1.0
        rows.append(y)
    zs = z.repeat(len(rows), 1)
    ys = torch.stack(rows).repeat_interleave(n_z, 0)
    return zs.to(device), ys.to(device), n_z


def save_checkpoint(path: Path, G, D, G_ema, opt_g, opt_d, epoch: int, cfg: Config, history: list) -> None:
    torch.save(
        {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "G_ema": G_ema.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "epoch": epoch,
            "config": cfg.to_dict(),
            "history": history,
        },
        path,
    )


def export_generator(path: Path, G_ema: Generator, cfg: Config) -> None:
    """Small inference-only file (~17 MB) used by the demo app and the Hub."""
    torch.save({"state_dict": G_ema.state_dict(), "config": cfg.to_dict()}, path)


# ---------------------------------------------------------------- training
def train(cfg: Config, loss_kind: str = "hinge", fake_data: bool = False, resume: str | None = None,
          max_steps: int | None = None) -> Path:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    out = Path(cfg.out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    # data
    if fake_data:
        from faceforge.data import FakeFaces

        ds = FakeFaces(n=max(cfg.batch_size * 4, 64), num_classes=cfg.num_classes, size=cfg.image_size)
    else:
        from faceforge.data import CelebAFaces, build_cache

        ds = CelebAFaces(build_cache(cfg.data_dir, cfg.attributes, cfg.num_shards, cfg.image_size))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=0 if fake_data else cfg.num_workers, pin_memory=device.type == "cuda")
    print(f"dataset: {len(ds):,} images | device: {device} | amp: {use_amp}")

    # models
    G = Generator(cfg.z_dim, cfg.num_classes, cfg.g_channels).to(device)
    D = Discriminator(cfg.num_classes, cfg.d_channels).to(device)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    print(f"G params: {count_params(G):,} | D params: {count_params(D):,}")

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=cfg.betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.betas)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, history = 0, []
    if resume:
        ck = torch.load(resume, map_location=device)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); G_ema.load_state_dict(ck["G_ema"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        start_epoch, history = ck["epoch"] + 1, ck.get("history", [])
        print(f"resumed from epoch {ck['epoch']}")

    z_fix, y_fix, ncol = fixed_sample_inputs(cfg, device)
    step = 0
    t_start = time.time()
    deadline = t_start + cfg.time_budget_min * 60 if cfg.time_budget_min > 0 else None
    out_of_time = False

    for epoch in range(start_epoch, cfg.epochs):
        G.train(); D.train()
        t0, d_sum, g_sum, n = time.time(), 0.0, 0.0, 0
        for real, y in loader:
            real, y = real.to(device, non_blocking=True), y.to(device, non_blocking=True)
            b = real.size(0)
            # Generator is asked for attribute combos drawn from the real data, so it
            # learns realistic combinations (e.g. not 'bald + blond').
            y_fake = y[torch.randperm(b, device=device)]

            # ---- discriminator step
            z = torch.randn(b, cfg.z_dim, device=device)
            with torch.autocast(device.type, enabled=use_amp):
                fake = G(z, y_fake)
                d_loss = d_loss_fn(D(real, y), D(fake.detach(), y_fake), loss_kind, cfg.label_smoothing)
            opt_d.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)

            # ---- generator step
            with torch.autocast(device.type, enabled=use_amp):
                g_loss = g_loss_fn(D(fake, y_fake), loss_kind)
            opt_g.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()
            update_ema(G_ema, G, cfg.ema_decay)

            d_sum += d_loss.item(); g_sum += g_loss.item(); n += 1; step += 1
            if step % 200 == 0:
                print(f"  step {step:6d} | D {d_loss.item():.3f} | G {g_loss.item():.3f}")
            if max_steps and step >= max_steps:
                break
            if deadline and time.time() > deadline:
                out_of_time = True
                print(f"  time budget of {cfg.time_budget_min} min reached at step {step}")
                break

        rec = {"epoch": epoch, "d_loss": d_sum / max(n, 1), "g_loss": g_sum / max(n, 1),
               "seconds": round(time.time() - t0, 1)}

        if (epoch + 1) % cfg.sample_every == 0 or epoch == cfg.epochs - 1 or out_of_time:
            with torch.no_grad():
                imgs = G_ema(z_fix, y_fix)
            save_image(imgs, out / "samples" / f"epoch_{epoch + 1:03d}.png", nrow=ncol,
                       normalize=True, value_range=(-1, 1))

        if cfg.fid_every and not fake_data and ((epoch + 1) % cfg.fid_every == 0 or epoch == cfg.epochs - 1):
            try:
                from faceforge.evaluate import compute_fid

                rec["fid"] = compute_fid(G_ema, ds, cfg, device)
            except ImportError as e:  # torchmetrics / torch-fidelity not installed
                print(f"  (skipping FID: {e})")

        history.append(rec)
        print(json.dumps(rec))
        (out / "history.json").write_text(json.dumps(history, indent=2))
        save_checkpoint(out / "last.pt", G, D, G_ema, opt_g, opt_d, epoch, cfg, history)
        export_generator(out / "generator.pt", G_ema, cfg)
        if (max_steps and step >= max_steps) or out_of_time:
            break

    print(f"done. generator exported to {out / 'generator.pt'}")
    return out / "generator.pt"


def parse_args() -> tuple[Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description="Train FaceForge conditional DCGAN")
    cfg = Config()
    for f in fields(Config):
        if f.name in ("attributes", "betas"):
            continue
        flag = "--" + f.name.replace("_", "-")
        default = getattr(cfg, f.name)
        if isinstance(default, bool):
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=default)
        else:
            p.add_argument(flag, type=type(default), default=default)
    p.add_argument("--loss", choices=["hinge", "bce"], default="hinge")
    p.add_argument("--fake-data", action="store_true", help="random tensors instead of CelebA")
    p.add_argument("--resume", type=str, default=None, help="path to last.pt")
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()
    for f in fields(Config):
        if hasattr(args, f.name):
            setattr(cfg, f.name, getattr(args, f.name))
    return cfg, args


if __name__ == "__main__":
    cfg, args = parse_args()
    train(cfg, loss_kind=args.loss, fake_data=args.fake_data, resume=args.resume, max_steps=args.max_steps)
