# Connectome adaptation and custom compute

The default actor learns robot adapters and heads; recurrent weight adaptation and learned neuron dynamics are independent choices.

Last updated: 2026-09-22

Related: [Actor](connectome-actor.md), [Backend analysis](../analyses/sparse-backends.md), [Probabilistic plasticity](../analyses/probabilistic-plasticity.md), [Workflow](../workflows/connectome-experiments.md)

## Configuration and semantics

The primary `SimToolRealConnectomeSAPG` profile sets this `params.network.connectome` subtree:

```yaml
adaptation:
  weight_mode: adapters_only
  learn_dynamics: false
  rank: 4
  edge_scale_bounds: [0.0625, 16.0]
  # synaptic_plasticity_reg: 1.0e-4   # low_rank only; 0 or omit => off
interface_projections:
  architecture: linear  # or mlp
  hidden_size: 256
  activation: elu
operator_backend: triton_fused
```

All modes train input adapters, motor action and actor-value heads, SAPG embeddings and action log standard deviations. Interface architecture is orthogonal to recurrent adaptation: `linear` preserves the original projections, while `mlp` replaces both input projections and the action-mean projection with one 256-unit hidden layer. The central critic is unchanged. `learn_dynamics` independently enables per-neuron leak \(\lambda_i\), recurrent bias \(b_i\), and intrinsic input-output gain \(a_i=\exp(\texttt{log\_intrinsic\_gain})\) (5,856 additional parameters on the 1,952-cell graph; 12,930 on the historical 4,310-cell graph). Intrinsic gain initializes at \(\texttt{log\_a}=0\) so \(a_i=1\) and the forward pass matches the previous dynamics at initialization. The default holds leaks at 0.5, biases at zero, and intrinsic gains at one.

| Mode | Recurrent parameters | Actor total with frozen dynamics | Meaning |
| --- | ---: | ---: | --- |
| `adapters_only` | 0 | 92,556 | Fixed recurrent operator |
| `neuron_gains` | 8,620 | 101,176 | Positive incoming/outgoing scaling |
| `low_rank`, rank 1 | 8,620 | 101,176 | Factorized signed relative edge modulation |
| `low_rank`, rank 4 | 34,480 | 127,036 | Factorized signed relative edge modulation |
| `low_rank`, rank 8 | 68,960 | 161,516 | Factorized signed relative edge modulation |
| `edgewise` | 118,920 | 211,476 | Independent edge magnitudes |
| `latent_probabilistic`, rank \(r\) | \(N(3+4r)+1\) | graph-dependent | Unified intrinsic, efficacy, and topology state plus `log_tau` |

These are alternative weight parameterizations. Low-rank/edgewise do not add trainable neuron gains. The four legacy modes keep the anatomical graph and direction of every edge. `latent_probabilistic` instead samples edge existence on a shared candidate supergraph while retaining the anatomical graph as its prior. Neuron-gain and edgewise modes keep extraction-derived signs; signed relative low-rank and conditional probabilistic efficacy may weaken through zero and flip signs. The extraction sign convention is a model assumption, not a claim that anatomy fully determines synaptic physiology.

Low-rank scores are \(\Delta_{ij}=\mathrm{sum}(U[i]\odot V[j])/\sqrt{r}\), evaluated only at existing edges. Effective weights use signed relative LoRA:

\[
W^{\mathrm{eff}}_{ij}=W^0_{ij}(1+\Delta_{ij}).
\]

Interpretation: \(\Delta=0\) unchanged; \(\Delta=1\) doubles; \(\Delta=-0.5\) halves; \(\Delta=-1\) zeros; \(\Delta<-1\) flips sign. One factor initializes randomly and the other to zero, so \(\Delta=0\) and \(W^{\mathrm{eff}}=W^0\) at start with nonzero learning gradients. There is no sigmoid map, no `edge_scale_bounds` constraint, and no arbitrary upper/lower bound on \(\Delta\). Global recurrent `dynamics.beta` remains a fixed scalar. (Superseded: the previous low-rank map was \(W^0\exp(\Delta)\) with a numerical \(\Delta\) clamp before `exp`.)

Edgewise still uses one zero-initialized score per edge mapped through a shifted sigmoid into `edge_scale_bounds` log multipliers, then `exp`. Neuron gains keep the same bounded map, with per-cell bounds equal to the square roots of the edge-scale bounds.

Optional `adaptation.synaptic_plasticity_reg` (\(\lambda_{\mathrm{syn}}\), default `0`) adds

\[
L_{\mathrm{syn}}=\lambda_{\mathrm{syn}}\frac{1}{E}\sum_{e=1}^{E}\Delta_e^2
\]

to the continuous PPO/SAPG loss when `weight_mode: low_rank`. Here \(\Delta\) is the **relative synaptic change**, not a log-fold gain. TensorBoard `losses/synaptic_plasticity_reg` logs the unscaled mean \(\Delta^2\) (pre-\(\lambda_{\mathrm{syn}}\)); update-level `adaptation/synaptic_delta_*` / `adaptation/synaptic_relative_scale_*` stats are also recorded. Synaptic-only experiments use `weight_mode: low_rank` with `learn_dynamics: false` (trainable `U,V`; frozen intrinsic \(a,\lambda,b\) and neuron gains).

### Unified probabilistic plasticity

`weight_mode: latent_probabilistic` is isolated from all legacy parameterizations. It stores one zero-initialized `plasticity_state[N, 3+4r]` with intrinsic, efficacy-post/pre, and topology-post/pre blocks, plus an Adam-trained scalar `log_tau`. Fixed random post anchors break bilinear symmetry; the pre blocks start at zero, so continuous intrinsic values and conditional edge weights exactly match the baseline at initialization.

Intrinsic values are read directly:
\[
a_i=a_i^0e^{z_i^a},\quad
\operatorname{logit}\lambda_i^{control}=\operatorname{logit}\lambda_i^0+z_i^\lambda,\quad
b_i=b_i^0+s_i^bz_i^b.
\]
The existing control-to-substep leak conversion is then applied once. \(s_i^b\), nonedge efficacy scales, and zero-degree fallbacks are all derived from baseline recurrent-weight RMS values.

For \(j\to i\), conditional efficacy and topology are:
\[
\mu_{ij}=W^0_{ij}+S_{ij}\frac{(U_i^W)^TV_j^W}{\sqrt r},\qquad
\operatorname{logit}\pi_{ij}=\operatorname{logit}\pi^0_{ij}
+\frac{(U_i^A)^TV_j^A}{\sqrt r}.
\]
Observed edges use \(S=W^0\), recovering signed relative LoRA. Nonedges use the geometric mean of measured incoming/outgoing RMS scales. The runtime circuit samples \(A_{ij}\sim Bernoulli(\pi_{ij})\) independently per environment and uses \(W_{ij}=A_{ij}\mu_{ij}\).

The prepared MaleCNS artifacts contain no calibrated edge-existence probability. `topology_prior_error_rate` therefore gives prior probability \(1-\epsilon_A\) to observed edges and \(\epsilon_A\) to allowed nonedges. It is adjacency uncertainty, not a pruning threshold. Candidate nonedges are selected by weighted Gumbel-top-k only as a compute cache; candidate membership is never edge existence.

PPO stores one compact topology seed per environment trajectory. Triton regenerates hard gates from `(topology_seed, canonical_edge_id)` with no timestep in the key and applies the intended straight-through sigmoid gradient. No edge-by-environment mask or per-environment CSR graph is stored.

The regularizer is one information measure \(I_z+I_A\), divided by neuron count. \(I_z\) is the exact Gaussian KL with learned \(\tau=\exp(\texttt{log_tau})\); \(I_A\) is Bernoulli posterior-to-adjacency-prior KL. A non-Adam dual multiplier tracks `target_information_nats_per_neuron`; no separate intrinsic, synaptic, sparsity, birth, death, or sign-flip coefficient is used. See the [implementation and benchmark analysis](../analyses/probabilistic-plasticity.md).

### Spectral monitoring

Diagnostic-only spectrum of the effective CSR operator \(W^{\mathrm{eff}}\) (same values as `effective_values()` / SpMM), not the tanh Jacobian. Active when synaptic edge weights are trainable (`low_rank`, `edgewise`, or `latent_probabilistic`); ignored for `adapters_only` and intrinsic-only runs even if enabled. The probabilistic mode analyzes the expected supported operator \(\pi\mu\), never one sampled graph. Config under `params.network.connectome`:

```yaml
spectral_monitoring:
  enabled: true
  interval: 100   # training epochs
  top_k: 8
```

Default interval is every 100 epochs. Metrics are written exclusively under `spectral/` (`effective/*`, `baseline/*`, `change/*`, `plasticity/*` with relative-scale stats for the signed `W^{\mathrm{eff}}`). Computation is detached (`torch.no_grad`); compact graphs (\(\le 2048\) neurons) use dense eig/SVD, larger graphs use ARPACK `eigs`/`svds`. Baseline \(\rho(W^0)\) and \(\sigma_{\max}(W^0)\) are cached once. Does not affect the loss.

### Spectral gradient preconditioning

Optional optimization geometry for signed LoRA and unified probabilistic pairwise blocks. It does not change the forward decoder, information/plasticity loss, or trainable parameters. Adam moments are not reset or rotated when the basis refreshes.

```yaml
spectral_preconditioning:
  enabled: false
  alpha: 0.0
  singular_value_floor_ratio: 1.0e-3
  refresh_every_ppo_updates: 1
```

Once per completed PPO/SAPG update (rollout collection plus all optimization epochs; configurable `refresh_every_ppo_updates`, default 1), a detached full float32 SVD is taken of the same CSR \(W^{\mathrm{eff}}\) used by SpMM and by `spectral/` monitoring. Multiplier arithmetic still uses float64. All modes are kept. Singular values are floored at `singular_value_floor_ratio * sigma_max`, raised to `alpha`, then normalized so the multipliers have mean 1. Target-side gradients use the left singular vectors and source-side gradients use the right singular vectors:

\[
\tilde G_U = P \widehat M P^\top G_U, \qquad
\tilde G_V = Q \widehat M Q^\top G_V.
\]

`alpha = 0` or `enabled: false` leaves gradients unchanged. Active only for `low_rank` or `latent_probabilistic`. In the probabilistic mode, the factorized operator is the current expected supported operator \(\bar W=\pi\mu\); `P` acts on both post blocks, `Q` on both pre blocks, and the three intrinsic columns plus `log_tau` remain raw. Defaults are off. Update-level TensorBoard tags live under `spectral_preconditioning/`. No additional spectral loss is added.

On the 33,720-edge MaleCNS matrix, `sigma_max` is 3.643 and 1,361 of 1,952 singular values exceed `1e-3 * sigma_max`. A partial SVD at the current floor therefore keeps about 70% of the modes, so it does not replace the dense float32 factorization. The remaining speed knob that preserves the formula is `refresh_every_ppo_updates`: the float32 SVD is roughly 0.6 s inside a 3.3 s epoch. bfloat16 SVD is unsupported on the RTX 4090.

### Plasticity monitoring

Epoch-interval TensorBoard diagnostics under `synaptic_plasticity/` and `intrinsic_plasticity/`. Content auto-activates by trainability (`low_rank` → synaptic tags; `learn_dynamics` → intrinsic tags). Config:

```yaml
plasticity_monitoring:
  enabled: true
  interval: 100   # training epochs
```

Defaults off in base SAPG; on for `SimToolRealConnectomeLowRankSAPG` and the historical K=4 low-rank suite overrides. Synaptic tags cover \(\Delta\) and relative scale \(1+\Delta\), sign-flip / zero-crossing fractions (zero-crossing uses \(|1+\Delta|<10^{-3}\)), strengthened/weakened/flipped fractions, and optional `sign_flip_fraction_by_nt/<consensus_nt>` from sibling `neurons.csv`. Intrinsic tags use **substep** leaks \(\lambda\) from `substep_leaks()` (actual forward dynamics), retention \(1-\lambda\), timescale \(\tau=-1/\log(1-\lambda)\), control-step retention \((1-\lambda)^K\), \(a\)/\(b\) and deltas vs init, plus sensitivity proxy \(s=\lambda a\) (not a Jacobian criterion). Detached only; no training constraints.

The original `[0.25, 4]` neuron-gain interval was introduced as an engineering prior in the initial connectome implementation plan; it was not derived from MaleCNS physiology, the connectome data, or the SimToolReal paper. Its reciprocal endpoints are symmetric around identity in log space. Because incoming and outgoing gains multiply on each edge, it permits total edge scaling from `0.25^2 = 0.0625` through `4^2 = 16`. Edgewise still uses `[0.0625, 16]` for comparability; low-rank no longer inherits that bound. Checkpoints trained under previous low-rank maps (sigmoid-bounded or \(\exp(\Delta)\)) are not semantically identical under signed relative LoRA.

This is a parameter-efficiency comparison, not a strictly nested hierarchy of functions. Unbounded low-rank relative modulation is not itself guaranteed to have rank \(r\). Rank 16 would use more factors than independent edge weights. Activation saturation remains available as a diagnostic.

## Biological interpretation

All changes are ordinary RL optimizer updates between rollout collections. Parameters stay fixed during rollout/deployment while recurrent activity evolves. There are no per-environment synaptic traces or gain-update rules.

The user subsequently asked about dopamine-driven learning as a cheaper alternative. [The online-learning analysis](../analyses/dopamine-inspired-learning.md) distinguishes a synthetic reward-modulated eligibility rule from native fly dopamine machinery and proposes a checkpoint-warm-started readout-only test. This remains a proposal; the current training mechanism has not changed.

Biologically, gain is the slope or sensitivity of a neural input-output response: how much a neuron's activity changes for a given change in input. It is not directly a measure of how important the neuron is. Gain can be changed through presynaptic release, postsynaptic conductance or excitability, inhibition and disinhibition, or neuromodulators. In this actor, `g_out[j]` is best treated as a cell-wide presynaptic-efficacy proxy for source neuron `j`, while `g_in[i]` is a cell-wide postsynaptic-sensitivity proxy for recurrent input to destination neuron `i`. These are deliberately coarse computational analogies: real modulation can be synapse-, receptor-, state-, and timescale-specific. The learned scalars were not measured for MaleCNS cells and must not be interpreted as inferred fly physiology.

Fly modulation has biological precedents: [Suver et al.](https://pubmed.ncbi.nlm.nih.gov/23142045/) demonstrated octopamine-mediated flight-dependent visual modulation, and [Olsen and Wilson](https://www.nature.com/articles/nature06864) demonstrated olfactory presynaptic gain control. Neither specifies gain dynamics for this MaleCNS front-leg extraction. Online modulation was considered and excluded.

Neuron gains resemble multiplicative adaptation; [standard LoRA](https://arxiv.org/abs/2106.09685) uses an additive low-rank update that would introduce effective off-graph connections. Here `edgewise` means adapting every existing edge under sign/magnitude constraints, not unrestricted dense fine-tuning.

## Compatibility

`SimToolRealConnectomeGainsDynamicsSAPG` preserves the previous primary actor. Random/rewired profiles explicitly retain gains plus learned dynamics; `SimToolRealConnectomeFrozenSAPG` remains an adapters-only alias.

Legacy saved configurations containing only `plasticity_mode` retain their meaning: `frozen_core` maps to adapters-only/frozen dynamics; `neuron_gains` maps to gains/learned dynamics. Specifying both interfaces is an error. Load old checkpoints with their saved configuration or matching explicit profile; the new primary profile intentionally changes trainability.

## Compute implementation

`connectome_ops.py` owns transient CSR/transpose structure, per-stream/shape cuSPARSE plans and autograd. `connectome_cusparse.cpp` uses generic SpMM with reusable preprocessing and SDDMM for edge gradients. First use compiles a C++ extension into PyTorch's external cache; matching CUDA headers/libraries, a compiler and ninja are required.

`connectome_triton.py` fuses tiled CSR accumulation with population drives, gains, intrinsic gain, tanh and leak interpolation. Backward computes pointwise derivatives, transposed sparse propagation and sampled edge reductions. Linear and MLP interface projections stay as ordinary PyTorch dense operations outside the custom recurrent kernel. Selecting an MLP therefore does not change Triton/cuSPARSE correctness or recurrent-kernel coverage, but it adds interface compute and reduces the fraction of total actor time that recurrence can accelerate. Benchmark end-to-end actor steps when comparing architectures. Recurrence stays FP32 under AMP. These helpers have no CLI; select them through `operator_backend`.

Frozen weights still require hidden-state derivatives to train input adapters. Shared trainable edge values add edge-gradient reductions but retain the shared graph. Values are formed once per sequence forward, reused across timesteps, and never cached detached across optimizer updates. Backend caches are absent from checkpoints and invalidated on model moves and state loads.

`backend_options.cusparse_algorithm` accepts `alg1`, `alg2` (default), or `alg3`; `cusparse_layout` accepts `row` or `column` (column for alg1, row otherwise). Unavailable requested backends fail explicitly.

The local-eligibility trainer currently supports only linear projections. Its hand-derived sensory, descending and action eligibility traces address a single weight matrix directly; an MLP requires layerwise traces and activation derivatives. It rejects `architecture: mlp` explicitly instead of silently applying an incomplete update. PPO uses autograd and supports both architectures.
