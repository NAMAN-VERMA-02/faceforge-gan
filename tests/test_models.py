import torch

from faceforge.config import Config
from faceforge.inference import interpolate, slerp
from faceforge.models import Discriminator, Generator, update_ema

B, Z, C = 4, 128, 6


def test_generator_shape_and_range():
    G = Generator(Z, C).eval()
    x = G(torch.randn(B, Z), torch.randint(0, 2, (B, C)).float())
    assert x.shape == (B, 3, 64, 64)
    assert x.min() >= -1 and x.max() <= 1


def test_discriminator_shape():
    D = Discriminator(C)
    out = D(torch.randn(B, 3, 64, 64), torch.randint(0, 2, (B, C)).float())
    assert out.shape == (B,)


def test_conditioning_changes_output():
    torch.manual_seed(0)
    G = Generator(Z, C).eval()
    z = torch.randn(1, Z)
    a = G(z, torch.zeros(1, C))
    b = G(z, torch.ones(1, C))
    assert not torch.allclose(a, b), "attribute vector must influence the image"


def test_gradients_flow_to_both_networks():
    G, D = Generator(Z, C), Discriminator(C)
    y = torch.ones(B, C)
    loss = -D(G(torch.randn(B, Z), y), y).mean()
    loss.backward()
    assert G.label_embed[0].weight.grad is not None
    assert any(p.grad is not None for p in D.parameters())


def test_ema_moves_toward_model():
    G, G_ema = Generator(Z, C), Generator(Z, C)
    before = next(G_ema.parameters()).clone()
    update_ema(G_ema, G, decay=0.5)
    after = next(G_ema.parameters())
    target = next(G.parameters())
    assert torch.allclose(after, 0.5 * before + 0.5 * target, atol=1e-6)


def test_slerp_endpoints():
    a, b = torch.randn(1, Z), torch.randn(1, Z)
    t = torch.tensor([[0.0], [1.0]])
    out = slerp(a.expand(2, -1), b.expand(2, -1), t)
    assert torch.allclose(out[0], a[0], atol=1e-4)
    assert torch.allclose(out[1], b[0], atol=1e-4)


def test_interpolate_output():
    G = Generator(Z, C).eval()
    imgs = interpolate(G, torch.zeros(C), torch.ones(C), 1, 2, steps=5)
    assert imgs.shape == (5, 64, 64, 3) and imgs.dtype == torch.uint8


def test_config_classes():
    assert Config().num_classes == C
