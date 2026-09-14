# Connectome adaptation and custom compute

The default actor learns robot adapters and heads; recurrent weight adaptation and learned neuron dynamics are independent choices.

Last updated: 2026-09-13

Related: [Actor](connectome-actor.md), [Backend analysis](../analyses/sparse-backends.md), [Workflow](../workflows/connectome-experiments.md)

## Configuration and semantics

The primary `SimToolRealConnectomeSAPG` profile sets this `params.network.connectome` subtree:

```yaml
adaptation:
  weight_mode: adapters_only
  learn_dynamics: false
  rank: 4
  edge_scale_bounds: [0.0625, 16.0]
operator_backend: triton_fused
```

All modes train input adapters, motor action and actor-value heads, SAPG embeddings and action log standard deviations. The central critic is unchanged. `learn_dynamics` independently enables neuron leaks and recurrent biases (8,620 additional parameters); the default holds leaks at 0.5 and biases at zero.

| Mode | Recurrent parameters | Actor total with frozen dynamics | Meaning |
| --- | ---: | ---: | --- |
| `adapters_only` | 0 | 92,556 | Fixed recurrent operator |
| `neuron_gains` | 8,620 | 101,176 | Positive incoming/outgoing scaling |
| `low_rank`, rank 1 | 8,620 | 101,176 | Factorized edge modulation |
| `low_rank`, rank 4 | 34,480 | 127,036 | Factorized edge modulation |
| `low_rank`, rank 8 | 68,960 | 161,516 | Factorized edge modulation |
| `edgewise` | 118,920 | 211,476 | Independent edge magnitudes |

These are alternative weight parameterizations. Low-rank/edgewise do not add trainable neuron gains. All keep the graph, direction, and extraction-derived sign of every edge. The sign convention is an extraction assumption, not a claim that anatomy fully determines synaptic physiology.

Low-rank scores are `sum(U[dst] * V[src]) / sqrt(rank)`, evaluated only at existing edges. One factor initializes randomly and the other to zero: identity weights with nonzero learning gradients. Edgewise uses one zero-initialized score per edge. Both map scores through a shifted sigmoid into bounded log multipliers and multiply base values by their exponentials. Zero scores map to one, including asymmetric bounds. Neuron gain bounds are square roots of the edge-scale bounds, giving the same overall limits.

The original `[0.25, 4]` neuron-gain interval was introduced as an engineering prior in the initial connectome implementation plan; it was not derived from MaleCNS physiology, the connectome data, or the SimToolReal paper. Its reciprocal endpoints are symmetric around identity in log space. Because incoming and outgoing gains multiply on each edge, it permits total edge scaling from `0.25^2 = 0.0625` through `4^2 = 16`. The later low-rank and edgewise modes inherited `[0.0625, 16]` only to make their allowed per-edge magnitude range comparable, not because that range has biological calibration.

This is a parameter-efficiency comparison, not a strictly nested hierarchy of functions. Nonlinear bounded low-rank modulation is not itself guaranteed to have rank r. Rank 16 would use more factors than independent edge weights. No adaptation penalty is added to SAPG; hard bounds limit weight drift, not dynamical instability. Activation saturation and effective log-weight drift are reported.

## Biological interpretation

All changes are ordinary RL optimizer updates between rollout collections. Parameters stay fixed during rollout/deployment while recurrent activity evolves. There are no per-environment synaptic traces or gain-update rules.

Biologically, gain is the slope or sensitivity of a neural input-output response: how much a neuron's activity changes for a given change in input. It is not directly a measure of how important the neuron is. Gain can be changed through presynaptic release, postsynaptic conductance or excitability, inhibition and disinhibition, or neuromodulators. In this actor, `g_out[j]` is best treated as a cell-wide presynaptic-efficacy proxy for source neuron `j`, while `g_in[i]` is a cell-wide postsynaptic-sensitivity proxy for recurrent input to destination neuron `i`. These are deliberately coarse computational analogies: real modulation can be synapse-, receptor-, state-, and timescale-specific. The learned scalars were not measured for MaleCNS cells and must not be interpreted as inferred fly physiology.

Fly modulation has biological precedents: [Suver et al.](https://pubmed.ncbi.nlm.nih.gov/23142045/) demonstrated octopamine-mediated flight-dependent visual modulation, and [Olsen and Wilson](https://www.nature.com/articles/nature06864) demonstrated olfactory presynaptic gain control. Neither specifies gain dynamics for this MaleCNS front-leg extraction. Online modulation was considered and excluded.

Neuron gains resemble multiplicative adaptation; [standard LoRA](https://arxiv.org/abs/2106.09685) uses an additive low-rank update that would introduce effective off-graph connections. Here `edgewise` means adapting every existing edge under sign/magnitude constraints, not unrestricted dense fine-tuning.

## Compatibility

`SimToolRealConnectomeGainsDynamicsSAPG` preserves the previous primary actor. Random/rewired profiles explicitly retain gains plus learned dynamics; `SimToolRealConnectomeFrozenSAPG` remains an adapters-only alias.

Legacy saved configurations containing only `plasticity_mode` retain their meaning: `frozen_core` maps to adapters-only/frozen dynamics; `neuron_gains` maps to gains/learned dynamics. Specifying both interfaces is an error. Load old checkpoints with their saved configuration or matching explicit profile; the new primary profile intentionally changes trainability.

## Compute implementation

`connectome_ops.py` owns transient CSR/transpose structure, per-stream/shape cuSPARSE plans and autograd. `connectome_cusparse.cpp` uses generic SpMM with reusable preprocessing and SDDMM for edge gradients. First use compiles a C++ extension into PyTorch's external cache; matching CUDA headers/libraries, a compiler and ninja are required.

`connectome_triton.py` fuses tiled CSR accumulation with population drives, gains, tanh and leak interpolation. Backward computes pointwise derivatives, transposed sparse propagation and sampled edge reductions. Dense adapters stay in PyTorch. Recurrence stays FP32 under AMP. These helpers have no CLI; select them through `operator_backend`.

Frozen weights still require hidden-state derivatives to train input adapters. Shared trainable edge values add edge-gradient reductions but retain the shared graph. Values are formed once per sequence forward, reused across timesteps, and never cached detached across optimizer updates. Backend caches are absent from checkpoints and invalidated on model moves and state loads.

`backend_options.cusparse_algorithm` accepts `alg1`, `alg2` (default), or `alg3`; `cusparse_layout` accepts `row` or `column` (column for alg1, row otherwise). Unavailable requested backends fail explicitly.
