"""CelebA loading.

Images come from the Hugging Face dataset `flwrlabs/celeba` (aligned faces + 40 binary
attributes, stored as parquet shards of ~8.5k images each). We download only as many
shards as needed, centre-crop each 178x218 face to 178x178, resize to 64x64 and cache
the result as one uint8 .npy array, so later epochs read straight from RAM.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HF_REPO = "flwrlabs/celeba"
SHARD_TEMPLATE = "img_align+identity+attr/train-{:05d}-of-00019.parquet"
MAX_SHARDS = 19


def _preprocess(img_bytes: bytes, size: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    side = min(w, h)
    # CelebA faces sit slightly above centre; shift the crop up a little.
    top = max(0, (h - side) // 2 - 10)
    left = (w - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def build_cache(data_dir: str, attributes: list[str], num_shards: int, size: int = 64) -> Path:
    """Download `num_shards` CelebA shards and write images/labels caches. Returns cache dir."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tqdm.auto import tqdm

    num_shards = max(1, min(num_shards, MAX_SHARDS))
    cache = Path(data_dir) / f"celeba{size}_{num_shards}shards"
    img_path, lbl_path = cache / "images.npy", cache / "labels.npy"
    if img_path.exists() and lbl_path.exists():
        return cache
    cache.mkdir(parents=True, exist_ok=True)

    images, labels = [], []
    for i in range(num_shards):
        local = hf_hub_download(HF_REPO, SHARD_TEMPLATE.format(i), repo_type="dataset")
        table = pq.read_table(local, columns=["image", *attributes])
        imgs = table.column("image").to_pylist()
        attr = np.stack([np.asarray(table.column(a).to_pylist(), dtype=np.float32) for a in attributes], 1)
        for rec in tqdm(imgs, desc=f"shard {i + 1}/{num_shards}", leave=False):
            images.append(_preprocess(rec["bytes"], size))
        labels.append(attr)

    np.save(img_path, np.stack(images))
    np.save(lbl_path, np.concatenate(labels))
    return cache


class CelebAFaces(Dataset):
    """Returns (image in [-1, 1] as CxHxW float tensor, attribute vector in {0, 1})."""

    def __init__(self, cache_dir: str | os.PathLike, hflip: bool = True):
        cache_dir = Path(cache_dir)
        self.images = np.load(cache_dir / "images.npy", mmap_mode="r")
        self.labels = np.load(cache_dir / "labels.npy")
        self.hflip = hflip

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.array(self.images[idx])).permute(2, 0, 1).float().div(127.5).sub(1.0)
        if self.hflip and torch.rand(()) < 0.5:
            x = x.flip(-1)  # a mirrored face keeps all six attributes
        return x, torch.from_numpy(self.labels[idx])


class FakeFaces(Dataset):
    """Random tensors with the same shapes as CelebAFaces. Used for smoke tests and CI."""

    def __init__(self, n: int = 256, num_classes: int = 6, size: int = 64):
        g = torch.Generator().manual_seed(0)
        self.x = torch.rand(n, 3, size, size, generator=g) * 2 - 1
        self.y = (torch.rand(n, num_classes, generator=g) > 0.5).float()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
