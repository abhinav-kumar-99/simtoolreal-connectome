# SimToolReal Connectome Wiki

Durable project knowledge for the MaleCNS-constrained SimToolReal actor.

Last updated: 2026-09-19

Related: [Overview](overview.md), [Log](log.md)

## Overview

- [Project overview](overview.md) — Scope, invariants, and implementation boundaries. Last updated 2026-09-13.

## Sources

- [Quantum-Coded Fruitfly](sources/summaries/quantum-coded-fruitfly.md) — Fixed population encoding, non-learning dopamine pulse, 100-word neural-input ablation, and the resulting structured-adapter direction. Last updated 2026-09-16.

- [MaleCNS front-leg circuit](sources/summaries/malecns-front-leg-circuit.md) — Data identity, provenance, and anatomical-data versus executable-dynamics boundary. Last updated 2026-09-15.
- [SimToolReal training references](sources/summaries/simtoolreal-training-references.md) — Published curve, critic-ablation contract, paper/code value-loss boundary, upstream numerical study export, released checkpoint metadata, and the live seed-42 official-method reproduction. Last updated 2026-09-18.

## Concepts

- [Camera-driven MaleCNS reservoir](concepts/visual-reservoir.md) — All 165,122 traced neurons, 64x36 multirate L1/L2 camera input, fresh proprioception, and motor, descending, or compressed all-neuron readout tradeoffs. Last updated 2026-09-16.

- [Connectome actor](concepts/connectome-actor.md) — Circuit inventory, dynamics, independently sized interface MLPs, structured sensory routing and 6D orientation, exact central-critic inputs, connectome-versus-LSTM auxiliary value gradients, propagation delay and saturation boundaries. Last updated 2026-09-19.
- [Fixed-input reservoir controller](concepts/fixed-reservoir-controller.md) — Parameter-free robot-to-MaleCNS population coding, cached motor features, no-BPTT PPO boundary, optional hard-spiking LIF dynamics, exact channel map and smoke evidence. Last updated 2026-09-19.
- [Adaptation and custom compute](concepts/connectome-adaptation.md) — Independent recurrent/interface controls, adapters-only default, and GPU implementation. Last updated 2026-09-14.
- [Training and evaluation success metrics](concepts/success-metrics.md) — TensorBoard populations, tolerance scaling, policy-ID mismatch, milestone Task Progress, and dual fixed/checkpoint-tolerance videos. Last updated 2026-09-18.

## Workflows

- [Anatomical activation videos](workflows/anatomical-activity-videos.md) — Actual compact-circuit skeletons, enlarged CNS overview, cropped HD robot footage, population labels/bars, 4× FPS and YAML usage. Last updated 2026-09-18.

- [Experiment workflow](workflows/connectome-experiments.md) — Current no-auxiliary fly and official repository LSTM runs, fixed-reservoir suites, milestone videos, LF entropy runs, and the staged handoff. Last updated 2026-09-19.
- [Eligibility training](workflows/eligibility-training.md) — Online trainer, policy-gradient/TD distinction, input-adapter learning alternatives, YAML/video contracts, early gain audit, and stopped-run status. Last updated 2026-09-14.

## Analyses

- [Fly cycles and input routing](analyses/fly-cycles-and-input-routing.md) — Exact directed reachability, current all-neuron shortcut, four-cycle boundary, and input-routing alternatives. Last updated 2026-09-19.
- [Parameter-matched LSTM baseline](analyses/lstm-baseline.md) — The 323-unit coefficient match, exact sparsity-control boundary, official 1,024-unit history, diagnosed scheduler cancellation, and active LF rollout-KL matched pair. Last updated 2026-09-19.
- [Visual reservoir performance](analyses/visual-reservoir-performance.md) — Frozen inference, full-CNS capacity sweep, video-worker memory limits and 2,304-environment continuation. Last updated 2026-09-16.
- [End-to-end system audit](analyses/system-audit-2026-09-15.md) — All 71 event histories, raw-source reproduction, cross-member KL, saturation, proposed Beta parameters and multi-rate neural timing. Last updated 2026-09-15.
- [Minimal dexterity circuits](analyses/minimal-dexterity-circuit.md) — Exact 86/262/408-cell candidates, threshold rationale, and 262 training evidence weakening its full-task recommendation. Last updated 2026-09-15.
- [Dopamine-inspired online learning](analyses/dopamine-inspired-learning.md) — Eligibility rationale, implementation boundary, Wordle source comparison and unproven learning/cost hypotheses. Last updated 2026-09-16.
- [Approved 1,952-neuron training](analyses/compact-1952-training.md) — Exact graph, historical launches, all-neuron readout pair, clipped-policy handoff and scheduler-KL interpretation. Last updated 2026-09-17.
- [Front-leg pathway coverage](analyses/front-leg-pathway-coverage.md) — Public-reference audit, 4,353/4,589/4,778-cell candidates, and evidence-led compact sensory-motor/feedback additions without a size quota. Last updated 2026-09-13.
- [Sparse recurrent backends](analyses/sparse-backends.md) — 160 benchmark cases, 32 capacity probes, memory tradeoffs and ten completed Isaac Gym smoke gates. Last updated 2026-09-13.
- [One-million-step adaptation pilot](analyses/adaptation-1m-pilot.md) — Five timed policies, 40 actor updates each, matched reward/success metrics, and 15 evaluation videos. Last updated 2026-09-13.
- [Billion-step adaptation run](analyses/adaptation-1b-run.md) — Completed adapters-only and neuron-gains jobs, update-count comparison, TensorBoard-axis semantics, and final evaluation videos. Last updated 2026-09-13.
- [100-billion-step update-timing comparison](analyses/adaptation-100b-update-timing.md) — Historical timing comparison, compact-run milestone snapshots, and numerical-failure status. Last updated 2026-09-14.

## Entities

- None yet.
