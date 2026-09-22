"""Utilities for unified latent probabilistic connectome plasticity.

The runtime orientation is always ``W[post, pre]``.  Production recurrence
generates Bernoulli gates inside Triton; the materialized gate helpers here are
for tiny dense references and detached diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import torch


@dataclass(frozen=True)
class LatentSlices:
    intrinsic: slice
    synaptic_post: slice
    synaptic_pre: slice
    topology_post: slice
    topology_pre: slice
    width: int


def latent_slices(rank: int) -> LatentSlices:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    return LatentSlices(
        intrinsic=slice(0, 3),
        synaptic_post=slice(3, 3 + rank),
        synaptic_pre=slice(3 + rank, 3 + 2 * rank),
        topology_post=slice(3 + 2 * rank, 3 + 3 * rank),
        topology_pre=slice(3 + 3 * rank, 3 + 4 * rank),
        width=3 + 4 * rank,
    )


def csr_rows(crow: torch.Tensor) -> torch.Tensor:
    neuron_count = int(crow.numel() - 1)
    return torch.repeat_interleave(
        torch.arange(neuron_count, device=crow.device, dtype=torch.long),
        (crow[1:] - crow[:-1]).to(dtype=torch.long),
    )


def canonical_pair_ids(
    rows: torch.Tensor, columns: torch.Tensor, neuron_count: int
) -> torch.Tensor:
    return rows.to(dtype=torch.long) * int(neuron_count) + columns.to(dtype=torch.long)


def pair_indices(
    canonical_ids: torch.Tensor, neuron_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = canonical_ids.to(dtype=torch.long)
    return torch.div(ids, int(neuron_count), rounding_mode="floor"), torch.remainder(
        ids, int(neuron_count)
    )


def derive_recurrent_scales(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    neuron_count: int,
) -> Dict[str, torch.Tensor]:
    """Return measured recurrent RMS scales with global zero-degree fallback."""
    rows = csr_rows(crow)
    weights = values.detach().to(dtype=torch.float64)
    global_rms = weights.square().mean().sqrt()
    if not torch.isfinite(global_rms) or float(global_rms) <= 0.0:
        raise ValueError("baseline recurrent weights must have a positive finite RMS")

    in_sum = torch.zeros(neuron_count, dtype=torch.float64, device=values.device)
    out_sum = torch.zeros_like(in_sum)
    in_count = torch.zeros(neuron_count, dtype=torch.long, device=values.device)
    out_count = torch.zeros_like(in_count)
    squared = weights.square()
    in_sum.index_add_(0, rows, squared)
    out_sum.index_add_(0, col.to(dtype=torch.long), squared)
    in_count.index_add_(0, rows, torch.ones_like(rows))
    out_count.index_add_(
        0, col.to(dtype=torch.long), torch.ones_like(col, dtype=torch.long)
    )
    fallback_sq = global_rms.square()
    in_rms = torch.sqrt(
        torch.where(
            in_count > 0,
            in_sum / in_count.clamp_min(1).to(dtype=torch.float64),
            fallback_sq,
        )
    )
    out_rms = torch.sqrt(
        torch.where(
            out_count > 0,
            out_sum / out_count.clamp_min(1).to(dtype=torch.float64),
            fallback_sq,
        )
    )
    return {
        "global_rms": global_rms.to(dtype=values.dtype),
        "in_rms": in_rms.to(dtype=values.dtype),
        "out_rms": out_rms.to(dtype=values.dtype),
        "bias_scale": in_rms.to(dtype=values.dtype),
    }


def allowed_nonedge_ids(
    neuron_count: int,
    original_ids: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Enumerate directed non-self pairs absent from the observed graph."""
    device = original_ids.device if device is None else device
    total = int(neuron_count) * int(neuron_count)
    ids = torch.arange(total, dtype=torch.long, device=device)
    rows, cols = pair_indices(ids, neuron_count)
    allowed = rows != cols
    observed = torch.zeros(total, dtype=torch.bool, device=device)
    observed[original_ids.to(device=device, dtype=torch.long)] = True
    return ids[allowed & ~observed]


def build_support_csr(
    original_ids: torch.Tensor,
    candidate_nonedge_ids: torch.Tensor,
    neuron_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support = torch.cat(
        (
            original_ids.to(dtype=torch.long),
            candidate_nonedge_ids.to(
                device=original_ids.device, dtype=torch.long
            ),
        )
    )
    support = torch.unique(support, sorted=True)
    rows, columns = pair_indices(support, neuron_count)
    counts = torch.bincount(rows, minlength=neuron_count)
    crow = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(dtype=torch.long)
    return crow, columns.contiguous(), support.contiguous()


def support_baselines_and_scales(
    support_ids: torch.Tensor,
    original_ids: torch.Tensor,
    original_values: torch.Tensor,
    in_rms: torch.Tensor,
    out_rms: torch.Tensor,
    neuron_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return W0, S, and observed mask in support order."""
    support_ids = support_ids.to(dtype=torch.long)
    original_ids = original_ids.to(device=support_ids.device, dtype=torch.long)
    insertion = torch.searchsorted(original_ids, support_ids)
    safe = insertion.clamp(max=max(int(original_ids.numel()) - 1, 0))
    observed = (insertion < original_ids.numel()) & (
        original_ids[safe] == support_ids
    )
    baseline = original_values.new_zeros(support_ids.numel())
    if bool(observed.any()):
        baseline[observed] = original_values[safe[observed]]
    rows, cols = pair_indices(support_ids, neuron_count)
    scale = torch.sqrt(
        in_rms[rows].to(dtype=original_values.dtype)
        * out_rms[cols].to(dtype=original_values.dtype)
    )
    scale = torch.where(observed, baseline, scale)
    return baseline, scale, observed


def pairwise_score(
    post: torch.Tensor,
    pre: torch.Tensor,
    canonical_ids: torch.Tensor,
    neuron_count: int,
) -> torch.Tensor:
    rows, cols = pair_indices(canonical_ids, neuron_count)
    return (post[rows] * pre[cols]).sum(dim=-1) / math.sqrt(post.shape[-1])


def bernoulli_kl_from_logits(
    posterior_logits: torch.Tensor, prior_logits: torch.Tensor
) -> torch.Tensor:
    """Elementwise KL(Bern(sigmoid(q)) || Bern(sigmoid(p))) stably."""
    q = torch.sigmoid(posterior_logits)
    return q * (
        torch.nn.functional.logsigmoid(posterior_logits)
        - torch.nn.functional.logsigmoid(prior_logits)
    ) + (1.0 - q) * (
        torch.nn.functional.logsigmoid(-posterior_logits)
        - torch.nn.functional.logsigmoid(-prior_logits)
    )


def gaussian_information(
    plasticity_state: torch.Tensor, log_tau: torch.Tensor
) -> torch.Tensor:
    """Exact KL(N(z,I) || N(0,tau^2 I)), stable around log_tau=0."""
    scalar = log_tau.reshape(())
    inverse_variance = torch.exp(-2.0 * scalar)
    dimension = plasticity_state.numel()
    variance_term = torch.expm1(-2.0 * scalar) + 2.0 * scalar
    return 0.5 * (
        plasticity_state.square().sum() * inverse_variance
        + float(dimension) * variance_term
    )


def _wang_hash_u32(values: torch.Tensor) -> torch.Tensor:
    mask = 0xFFFFFFFF
    x = torch.bitwise_and(values.to(dtype=torch.int64), mask)
    x = torch.bitwise_xor(torch.bitwise_xor(x, 61), torch.bitwise_right_shift(x, 16))
    x = torch.bitwise_and(x + torch.bitwise_left_shift(x, 3), mask)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 4))
    x = torch.bitwise_and(x * 0x27D4EB2D, mask)
    return torch.bitwise_and(
        torch.bitwise_xor(x, torch.bitwise_right_shift(x, 15)), mask
    )


def stateless_uniform(
    topology_seeds: torch.Tensor, canonical_edge_ids: torch.Tensor
) -> torch.Tensor:
    """Counter-based U(0,1), shaped ``[edges, samples]`` for references only."""
    seeds = topology_seeds.to(dtype=torch.int64).reshape(1, -1)
    edge_ids = canonical_edge_ids.to(
        device=seeds.device, dtype=torch.int64
    ).reshape(-1, 1)
    low = torch.bitwise_and(seeds, 0xFFFFFFFF)
    high = torch.bitwise_and(torch.bitwise_right_shift(seeds, 32), 0xFFFFFFFF)
    rotated = torch.bitwise_and(
        torch.bitwise_left_shift(high, 16)
        | torch.bitwise_right_shift(high, 16),
        0xFFFFFFFF,
    )
    mixed = torch.bitwise_xor(
        torch.bitwise_xor(edge_ids, low), rotated ^ 0x9E3779B9
    )
    hashed = _wang_hash_u32(mixed)
    return (hashed.to(dtype=torch.float64) + 0.5) / float(2**32)


def hard_bernoulli_gates(
    posterior_logits: torch.Tensor,
    topology_seeds: torch.Tensor,
    canonical_edge_ids: torch.Tensor,
) -> torch.Tensor:
    probability = torch.sigmoid(posterior_logits).reshape(-1, 1)
    uniform = stateless_uniform(topology_seeds, canonical_edge_ids).to(
        device=probability.device, dtype=probability.dtype
    )
    return (uniform < probability).to(dtype=probability.dtype)


def straight_through_gates(
    posterior_logits: torch.Tensor,
    topology_seeds: torch.Tensor,
    canonical_edge_ids: torch.Tensor,
) -> torch.Tensor:
    probability = torch.sigmoid(posterior_logits).reshape(-1, 1)
    hard = hard_bernoulli_gates(
        posterior_logits, topology_seeds, canonical_edge_ids
    )
    return hard + probability - probability.detach()


def dense_gated_recurrent_reference(
    rows: torch.Tensor,
    columns: torch.Tensor,
    conditional_values: torch.Tensor,
    posterior_logits: torch.Tensor,
    hidden: torch.Tensor,
    topology_seeds: torch.Tensor,
    canonical_edge_ids: torch.Tensor,
    outgoing_gain: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tiny differentiable reference for the intended straight-through rule."""
    gates = straight_through_gates(
        posterior_logits, topology_seeds, canonical_edge_ids
    )
    source = hidden[:, columns.to(dtype=torch.long)].transpose(0, 1)
    if outgoing_gain is not None:
        source = source * outgoing_gain[columns.to(dtype=torch.long)].unsqueeze(1)
    contributions = conditional_values.unsqueeze(1) * gates * source
    recurrent = hidden.new_zeros((hidden.shape[1], hidden.shape[0]))
    recurrent.index_add_(0, rows.to(dtype=torch.long), contributions)
    return recurrent.transpose(0, 1)


def weighted_sample_without_replacement(
    canonical_ids: torch.Tensor,
    probabilities: torch.Tensor,
    budget: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Plackett-Luce weighted sample via Gumbel top-k."""
    count = min(max(int(budget), 0), int(canonical_ids.numel()))
    if count == 0:
        return canonical_ids.new_empty(0)
    uniform = torch.rand(
        probabilities.shape,
        device=probabilities.device,
        dtype=torch.float32,
        generator=generator,
    ).clamp_(min=torch.finfo(torch.float32).tiny, max=1.0 - 1.0e-7)
    gumbel = -torch.log(-torch.log(uniform))
    scores = torch.log(
        probabilities.detach().to(dtype=torch.float32).clamp_min(
            torch.finfo(torch.float32).tiny
        )
    ) + gumbel
    selected = torch.topk(scores, k=count, sorted=False).indices
    return torch.sort(canonical_ids[selected]).values


def uniform_sample_without_replacement(
    canonical_ids: torch.Tensor,
    budget: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    count = min(max(int(budget), 0), int(canonical_ids.numel()))
    if count == 0:
        return canonical_ids.new_empty(0)
    keys = torch.rand(
        canonical_ids.numel(),
        device=canonical_ids.device,
        dtype=torch.float32,
        generator=generator,
    )
    selected = torch.topk(keys, k=count, sorted=False).indices
    return torch.sort(canonical_ids[selected]).values
