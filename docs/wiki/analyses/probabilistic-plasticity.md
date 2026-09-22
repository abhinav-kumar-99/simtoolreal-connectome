# Unified probabilistic plasticity

The new `latent_probabilistic` mode unifies intrinsic adaptation, conditional signed efficacy, and sampled topology in one neuron state while preserving all legacy modes.

Last updated: 2026-09-22

Related: [Adaptation concept](../concepts/connectome-adaptation.md), [Experiment workflow](../workflows/connectome-experiments.md), [Spectral preconditioning](spectral-preconditioning-266m.md)

## Data and prior

The prepared 1,952-neuron NPZ contains binary support, signed spectrally normalized contact-count weights, and population masks. Its source audit applied `conf_pre/conf_post >= 0.5` as a hard synapse filter before aggregation; neither the artifact nor `reference_incoming_edges.csv.gz` contains a calibrated edge-existence probability. The topology prior therefore uses the single adjacency uncertainty parameter:

\[
\pi^0_{ij}=1-\epsilon_A
\quad\text{on observed edges},\qquad
\pi^0_{ij}=\epsilon_A
\quad\text{on allowed nonedges}.
\]

The one observed self-edge remains in support. New self-edges are excluded.

## State and decoder

For rank \(r\), each neuron has:
\[
z_i=[z_i^a,z_i^\lambda,z_i^b,z_i^{W,post},z_i^{W,pre},
z_i^{A,post},z_i^{A,pre}]\in\mathbb R^{3+4r}.
\]

`plasticity_state` starts at zero. Fixed post anchors use the existing LoRA `N(0,0.1)` initialization; both pre blocks start at zero. Intrinsic values use direct log/logit/scale offsets. Conditional efficacy is:
\[
\mu_{ij}=W^0_{ij}+S_{ij}
\frac{(U_i^W)^TV_j^W}{\sqrt r}.
\]

Observed edges use \(S=W^0\), while nonedges use
\(\sqrt{s_i^{in}s_j^{out}}\). Bias and nonedge scales are measured from baseline recurrent RMS values with a global RMS fallback for zero degree.

Topology logits are:
\[
\ell_{ij}=\operatorname{logit}\pi^0_{ij}
+\frac{(U_i^A)^TV_j^A}{\sqrt r}.
\]

The recurrent coupling for environment \(b\) is
\(W_{ij}^{(b)}=\mu_{ij}\mathbf1[u(seed_b,id_{ij})<\sigma(\ell_{ij})]\).
The stateless hash excludes timestep, so topology is fixed through recurrent evolution.

## Candidate approximation and PPO conditioning

One shared CSR candidate support contains all 33,720 anatomical edges plus a configurable weighted-without-replacement nonedge cache. Rank 0 selects nonedges by Gumbel-top-k over the all-pair posterior and broadcasts exact canonical IDs. Candidate membership is only a compute cache.

Each rollout environment stores one int64 topology ID. It follows observations/actions through the experience buffer, accumulation, leader/follower augmentation, trajectory shuffle, recurrent sequence slicing, and every PPO minibatch. Triton regenerates gates from `(topology ID, canonical edge ID)` in forward, hidden transpose, and edge/logit backward. No persistent `E x B` gate tensor or per-environment CSR graph exists.

The topology-KL nonedge sample is uniform, fixed for an update, and separate from candidate support. Full topology KL is computed without autograd at update boundaries.

## Information control and spectral geometry

The information objective is:
\[
I_z=\frac12\left[\frac{D+\|z\|^2}{\tau^2}-D+2D\log\tau\right],
\qquad
I_A=\sum_{ij}KL[Bern(\pi_{ij})\Vert Bern(\pi^0_{ij})].
\]

`log_tau` is trained by Adam. A separate non-Adam dual variable applies
\(\alpha_I(I_z+\hat I_A)/N\) and updates after a completed PPO phase:
\[
\eta_I\leftarrow\eta_I+\texttt{dual_lr}
\left((I_z+I_A)/N-\epsilon_I\right).
\]

The selected default is \(\epsilon_I=1.0\) nat per neuron. There are no separate intrinsic, synaptic, topology, birth/death, or sign-flip penalties.

Spectral monitoring and preconditioning use the expected current candidate operator \(\bar W=\pi\mu\), not one sampled circuit. `P` acts on synaptic/topology post columns, `Q` on pre columns, and intrinsic columns plus `log_tau` are untouched. Rank 0 broadcasts contiguous SVD tensors so every DDP rank applies exactly the same geometry.

## Benchmark

Source: `configs/connectome/profiling/probabilistic_plasticity.yaml`, output `profiles/connectome/probabilistic_plasticity.json`. The matched RTX 4090 run used rank 4, 33,720 candidate nonedges, Triton, and synthetic PPO loss. Both physical GPUs were already occupied by long-running jobs, so these numbers are contention-sensitive engineering measurements rather than publication-quality isolated benchmarks.

- Signed LoRA, 33,720 edges: 384-lane forward 2.099 ms; 64x16 forward+backward 21.739 ms; PPO-step throughput 35,947 observations/s; peak 87.4 MiB.
- Probabilistic, group size 1, 67,440 support edges: forward 12.900 ms; forward+backward 31.594 ms; PPO-step throughput 33,434 observations/s; peak 126.6 MiB.
- Probabilistic, group size 32, 67,440 support edges: forward 2.451 ms; forward+backward 34.535 ms; PPO-step throughput 29,775 observations/s; peak 126.6 MiB.

Relative to signed LoRA, independent sampled topology measured about 6.15x forward latency, 1.45x forward+backward latency, 7.0% lower PPO throughput, and 39.2 MiB higher measured peak memory. Group 32 produced a much lower isolated forward median under the contended run but worse forward+backward and 10.9% lower PPO throughput than group 1. Sharing changes seed assignment but does not remove per-lane hidden-state work. The default therefore remains fully independent group size 1.

The support doubled from 33,720 to 67,440 edges, matching the expected main cost driver. No sampled mask storage appears in model state or peak-memory accounting. Warm candidate refreshes were approximately 7.1-28.5 ms in these cases; the first probabilistic refresh included cold allocator/kernel effects and took 107.2 ms.

## Verification

- New focused suite: 16 tests passed, including CPU math/cache/checkpoint/PPO/LF paths, keyed pair-sampling and collision-safe ID checks, exact diagnostics and inference-mode guards, two-rank Gloo support/dual/SVD equality, and CUDA forward/backward parity with the dense straight-through reference.
- Legacy connectome CPU regression: 60 passed, 9 skipped.
- Legacy Triton regression: 12 passed.
- PPO accumulation and configuration suite: 40 tests passed; one unrelated pre-existing evaluation-contract assertion expects three cases while the YAML currently contains five.

The sampled training path deliberately requires tanh plus `operator_backend: triton_fused`; the dense PyTorch implementation is a tiny-graph correctness oracle, not a second production backend. Sampled and MAP inference are supported; expected topology is diagnostic-only.
