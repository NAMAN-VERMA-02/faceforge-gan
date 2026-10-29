"""Central configuration for training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# CelebA attributes the generator is conditioned on. Each is a binary on/off switch
# the user can flip in the demo. Order matters: it defines the label vector layout.
ATTRIBUTES: list[str] = [
    "Male",
    "Smiling",
    "Eyeglasses",
    "Blond_Hair",
    "Black_Hair",
    "Young",
]


@dataclass
class Config:
    # Data
    image_size: int = 64
    num_shards: int = 6            # CelebA train shards to download (~8.5k faces each)
    data_dir: str = "data"
    attributes: list[str] = field(default_factory=lambda: list(ATTRIBUTES))

    # Model
    z_dim: int = 128
    g_channels: int = 64           # base width of the generator
    d_channels: int = 64           # base width of the discriminator

    # Optimisation (DCGAN paper settings, plus TTUR: a faster D learning rate)
    epochs: int = 25
    batch_size: int = 128
    lr_g: float = 1e-4             # TTUR (Heusel et al. 2017): D learns faster than G
    lr_d: float = 4e-4
    betas: tuple[float, float] = (0.0, 0.999)  # beta1=0 is standard with hinge loss + TTUR
    label_smoothing: float = 0.9   # one-sided: real targets become 0.9 instead of 1.0
    ema_decay: float = 0.999       # exponential moving average of generator weights
    ema_warmup: bool = True        # ramp EMA decay up from 0 so early weights don't linger
    diffaug: str = "color,translation,cutout"  # DiffAugment policy ("" disables)
    amp: bool = True               # mixed precision on GPU

    # Logging / checkpoints
    out_dir: str = "runs/faceforge"
    sample_every: int = 1          # epochs
    fid_every: int = 5             # epochs (0 disables FID)
    fid_samples: int = 5000
    time_budget_min: float = 0.0   # stop cleanly after this many minutes (0 = no limit)
    seed: int = 42
    num_workers: int = 2

    @property
    def num_classes(self) -> int:
        return len(self.attributes)

    def to_dict(self) -> dict:
        return asdict(self)
