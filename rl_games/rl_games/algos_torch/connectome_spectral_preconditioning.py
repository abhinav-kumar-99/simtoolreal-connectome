"""Spectral gradient preconditioning for signed low-rank synaptic plasticity.

The effective recurrent matrix is factored once per refresh under ``no_grad``.
Adam continues to update ``edge_u`` and ``edge_v`` in their original
coordinates; only those gradients are replaced before the optimizer step.

The SVD source is the CSR operator ``W_eff = W0 * (1 + Delta)`` used by
SpMM and by ``spectral/`` diagnostics. It does not include ``beta``, neuron
gains, or the tanh Jacobian. Rows are postsynaptic (target, ``U`` / ``P``);
columns are presynaptic (source, ``V`` / ``Q``).
"""

from __future__ import annotations

from typing import Dict

import torch


def dense_recurrent_from_csr(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    neuron_count: int,
) -> torch.Tensor:
    """Build a dense ``W[post, pre]`` matrix from CSR edge values.

    The factorization stays in float32. A full float64 SVD of the ~2k neuron
    operator dominated each PPO update on consumer GPUs.
    """
    dense = torch.zeros(
        neuron_count,
        neuron_count,
        dtype=torch.float32,
        device=values.device,
    )
    if values.numel() == 0:
        return dense
    counts = (crow[1:] - crow[:-1]).to(dtype=torch.long)
    rows = torch.repeat_interleave(
        torch.arange(neuron_count, device=values.device),
        counts,
    )
    dense[rows, col.to(dtype=torch.long)] = values.detach().to(dtype=torch.float32)
    return dense


def spectral_multipliers(
    singular_values: torch.Tensor,
    alpha: float,
    floor_ratio: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Return mean-1 multipliers ``(max(sigma, c * sigma_max)) ** alpha``.

    The floor is a numerical safeguard for negative ``alpha``. If the spectrum
    is identically zero, the preconditioner is the identity.
    """
    sigma = singular_values.detach().to(dtype=torch.float64).reshape(-1).abs()
    sigma_max = float(sigma.max().item()) if sigma.numel() else 0.0
    sigma_min = float(sigma.min().item()) if sigma.numel() else 0.0
    if sigma.numel() == 0 or sigma_max <= 0.0 or not torch.isfinite(sigma).all():
        multipliers = torch.ones_like(sigma)
        stats = {
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "sigma_floor": 0.0,
            "multiplier_min": 1.0,
            "multiplier_max": 1.0,
            "multiplier_mean": 1.0,
            "multiplier_std": 0.0,
            "alpha": float(alpha),
        }
        return multipliers, stats

    sigma_floor = float(floor_ratio) * sigma_max
    floored = torch.clamp(sigma, min=sigma_floor)
    multipliers = floored.pow(float(alpha))
    if not torch.isfinite(multipliers).all():
        raise RuntimeError(
            "spectral preconditioner multipliers are non-finite; "
            "increase singular_value_floor_ratio"
        )
    mean = multipliers.mean()
    if not torch.isfinite(mean) or float(mean) <= 0.0:
        raise RuntimeError("spectral preconditioner multiplier mean is invalid")
    multipliers = multipliers / mean
    stats = {
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "sigma_floor": sigma_floor,
        "multiplier_min": float(multipliers.min().item()),
        "multiplier_max": float(multipliers.max().item()),
        "multiplier_mean": float(multipliers.mean().item()),
        "multiplier_std": float(multipliers.std(unbiased=False).item()),
        "alpha": float(alpha),
    }
    return multipliers, stats


def apply_spectral_preconditioner(
    grad: torch.Tensor,
    basis: torch.Tensor,
    multipliers: torch.Tensor,
) -> torch.Tensor:
    """Apply ``basis @ diag(multipliers) @ basis.T`` without forming it.

    ``grad`` has shape ``(neurons, rank)``. ``basis`` is ``P`` for target-side
    ``U`` gradients and ``Q`` for source-side ``V`` gradients.
    """
    spectral = basis.transpose(0, 1).matmul(grad.to(dtype=basis.dtype))
    spectral = spectral * multipliers.to(dtype=basis.dtype).unsqueeze(1)
    return basis.matmul(spectral).to(dtype=grad.dtype)


def gradient_change_stats(
    prefix: str,
    raw: torch.Tensor,
    preconditioned: torch.Tensor,
) -> Dict[str, float]:
    """Cosine and norm ratio between a raw LoRA gradient and its replacement."""
    raw_flat = raw.detach().reshape(-1).to(dtype=torch.float64)
    new_flat = preconditioned.detach().reshape(-1).to(dtype=torch.float64)
    raw_norm = raw_flat.norm()
    new_norm = new_flat.norm()
    denom = raw_norm * new_norm
    if float(denom) == 0.0:
        cosine = 1.0
    else:
        cosine = float((raw_flat.dot(new_flat) / denom).item())
    if float(raw_norm) == 0.0:
        ratio = 1.0
    else:
        ratio = float((new_norm / raw_norm).item())
    return {
        f"{prefix}_gradient_cosine": cosine,
        f"{prefix}_gradient_norm_ratio": ratio,
    }


@torch.no_grad()
def compute_spectral_preconditioner(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    neuron_count: int,
    *,
    alpha: float,
    floor_ratio: float,
    dtype: torch.dtype,
) -> Dict[str, object]:
    """Full SVD of detached ``W_eff`` and the cached mean-1 spectral weights.

    Uses every singular mode. ``P`` and ``Q`` satisfy ``W_eff = P diag(S) Q.T``.
    """
    dense = dense_recurrent_from_csr(crow, col, values, neuron_count)
    left, singular_values, vh = torch.linalg.svd(dense, full_matrices=True)
    right = vh.transpose(0, 1).contiguous()
    multipliers, stats = spectral_multipliers(singular_values, alpha, floor_ratio)
    return {
        "P": left.to(dtype=dtype),
        "Q": right.to(dtype=dtype),
        "multipliers": multipliers.to(dtype=dtype),
        "singular_values": singular_values.to(dtype=dtype),
        **stats,
    }
