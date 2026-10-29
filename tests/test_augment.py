import torch

from faceforge.augment import diff_augment, rand_cutout, rand_translation


def test_shapes_preserved():
    x = torch.rand(4, 3, 64, 64) * 2 - 1
    for policy in ["", "color", "translation", "cutout", "color,translation,cutout"]:
        assert diff_augment(x, policy).shape == x.shape


def test_empty_policy_is_identity():
    x = torch.rand(2, 3, 64, 64)
    assert torch.equal(diff_augment(x, ""), x)


def test_gradients_flow_through_augment():
    x = torch.rand(2, 3, 64, 64, requires_grad=True)
    diff_augment(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_cutout_zeroes_a_region():
    x = torch.ones(2, 3, 64, 64)
    out = rand_cutout(x)
    zero_frac = (out == 0).float().mean().item()
    assert 0.05 < zero_frac < 0.5


def test_translation_keeps_values_in_range():
    x = torch.rand(2, 3, 64, 64)
    out = rand_translation(x)
    assert out.min() >= 0 and out.max() <= 1
