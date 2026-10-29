"""End-to-end smoke test: train a few steps on fake data, export, reload, generate."""

import json

import torch

from faceforge.config import Config
from faceforge.inference import generate, load_generator
from faceforge.train import train


def test_train_export_reload(tmp_path):
    cfg = Config(out_dir=str(tmp_path), epochs=1, batch_size=8, fid_every=0,
                 g_channels=16, d_channels=16, z_dim=32)
    path = train(cfg, fake_data=True, max_steps=3)

    assert path.exists()
    assert (tmp_path / "samples" / "epoch_001.png").exists()
    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history) == 1 and history[0]["d_loss"] > 0

    G, loaded_cfg = load_generator(path)
    assert loaded_cfg.z_dim == 32
    faces = generate(G, torch.tensor([1, 0, 1, 0, 0, 1]), seed=3, n=2)
    assert faces.shape == (2, 64, 64, 3)
    # same seed -> same face
    assert torch.equal(faces, generate(G, torch.tensor([1, 0, 1, 0, 0, 1]), seed=3, n=2))


def test_bce_loss_variant(tmp_path):
    cfg = Config(out_dir=str(tmp_path), epochs=1, batch_size=8, fid_every=0,
                 g_channels=16, d_channels=16, z_dim=32)
    train(cfg, loss_kind="bce", fake_data=True, max_steps=2)
    assert (tmp_path / "last.pt").exists()


def test_resume(tmp_path):
    cfg = Config(out_dir=str(tmp_path), epochs=1, batch_size=8, fid_every=0,
                 g_channels=16, d_channels=16, z_dim=32)
    train(cfg, fake_data=True, max_steps=2)
    cfg.epochs = 2
    train(cfg, fake_data=True, resume=str(tmp_path / "last.pt"), max_steps=2)
    history = json.loads((tmp_path / "history.json").read_text())
    assert [h["epoch"] for h in history] == [0, 1]
