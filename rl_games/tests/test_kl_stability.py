import math

import pytest
import torch

from rl_games.algos_torch.torch_ext import policy_kl
from rl_games.common.schedulers import AdaptiveScheduler


@pytest.mark.parametrize('log_scale', [-90.0, -45.0, 0.0, 45.0, 85.0])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_kl_matches_float64_reference_at_extreme_scales(log_scale, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    s0 = torch.full((3, 29), log_scale, device=device).exp()
    s1 = torch.full((3, 29), log_scale + 0.01, device=device).exp()
    m0 = s0 * 0.2
    m1 = s1 * 0.3
    reference = torch.distributions.kl_divergence(
        torch.distributions.Normal(m0.double(), s0.double()),
        torch.distributions.Normal(m1.double(), s1.double()),
    ).sum(-1)
    actual = policy_kl(m0, s0, m1, s1, reduce=False)
    torch.testing.assert_close(actual.double(), reference, rtol=1e-5, atol=1e-8)
    assert torch.isfinite(actual).all() and (actual >= 0).all()
    torch.testing.assert_close(policy_kl(m0, s0, m1, s1), actual.mean())


def test_identical_distributions_zero_and_close_distributions_positive():
    m = torch.zeros(2, 29)
    s = torch.ones_like(m)
    assert policy_kl(m, s, m, s).item() == 0.0
    close = policy_kl(m, s, m, s * (1 + 1e-6))
    assert 0 < close.item() < 1e-9


@pytest.mark.parametrize('invalid', [0.0, -1.0, float('nan'), float('inf')])
def test_invalid_scale_is_explicit_infinite_kl(invalid):
    m = torch.zeros(2, 3)
    s0 = torch.ones_like(m)
    s1 = s0.clone()
    s1[0, 1] = invalid
    actual = policy_kl(m, s0, m, s1, reduce=False)
    assert torch.isinf(actual[0]) and actual[1] == 0
    assert torch.isinf(policy_kl(m, s0, m, s1))


def test_promotes_half_and_avoids_large_mean_subtraction_overflow():
    m0 = torch.tensor([[3e38]])
    m1 = -m0
    s = torch.full_like(m0, 3e38)
    torch.testing.assert_close(policy_kl(m0, s, m1, s), torch.tensor(2.0))
    half = torch.ones(2, 3, dtype=torch.float16)
    assert policy_kl(half, half, half, half).dtype == torch.float32


@pytest.mark.parametrize('invalid', [-0.1, -1e-12, float('nan'), float('inf'), -float('inf')])
def test_invalid_kl_always_reduces_lr_and_preserves_entropy(invalid):
    scheduler = AdaptiveScheduler(0.004)
    for lr in [0.01, 0.0001, 1e-6]:
        new_lr, entropy = scheduler.update(lr, 0.25, 1, 0, invalid)
        assert math.isclose(new_lr, max(lr / 1.5, 1e-6))
        assert entropy == 0.25


@pytest.mark.parametrize('kl,factor', [(0.001, 1.5), (0.002, 1), (0.004, 1), (0.008, 1), (0.009, 1 / 1.5)])
def test_scheduler_threshold_boundaries(kl, factor):
    lr, _ = AdaptiveScheduler(0.004).update(1e-4, 0, 1, 0, kl)
    assert math.isclose(lr, 1e-4 * factor)


@pytest.mark.parametrize('kl', [0, 0.004, 0.02, -0.1, float('nan'), float('inf')])
def test_configured_max_lr_enforced_even_for_out_of_bounds_input(kl):
    scheduler = AdaptiveScheduler(0.004, max_lr=1e-3)
    lr, _ = scheduler.update(1e-2, 0, 1, 0, kl)
    assert lr == 1e-3
    lr, _ = scheduler.update(1e-3, 0, 1, 0, kl)
    expected = 1e-3 if math.isfinite(kl) and 0 <= kl <= 0.008 else 1e-3 / 1.5
    assert math.isclose(lr, expected)


@pytest.mark.parametrize('minimum,maximum', [(0, 1e-3), (1e-3, 1e-6), (1e-6, float('inf'))])
def test_invalid_lr_bounds_rejected(minimum, maximum):
    with pytest.raises(ValueError, match='Adaptive LR bounds'):
        AdaptiveScheduler(min_lr=minimum, max_lr=maximum)
