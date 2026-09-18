# SimToolReal Connectome Wiki

Durable project knowledge for the MaleCNS-constrained SimToolReal actor.

Last updated: 2026-09-17

Related: [Overview](overview.md), [Log](log.md)

## Overview

- [Project overview](overview.md) — Scope, invariants, and implementation boundaries. Last updated 2026-09-13.

## Sources

- [Quantum-Coded Fruitfly](sources/summaries/quantum-coded-fruitfly.md) — Fixed population encoding, non-learning dopamine pulse, 100-word neural-input ablation, and the resulting structured-adapter direction. Last updated 2026-09-16.

- [MaleCNS front-leg circuit](sources/summaries/malecns-front-leg-circuit.md) — Data identity, provenance, and anatomical-data versus executable-dynamics boundary. Last updated 2026-09-15.
- [SimToolReal training references](sources/summaries/simtoolreal-training-references.md) — Published curve, critic-ablation contract, paper/code value-loss boundary, upstream numerical study export, and released checkpoint metadata. Last updated 2026-09-15.

## Concepts

- [Camera-driven MaleCNS reservoir](concepts/visual-reservoir.md) — All 165,122 traced neurons, 64x36 multirate L1/L2 camera input, fresh proprioception, and motor, descending, or compressed all-neuron readout tradeoffs. Last updated 2026-09-16.

- [Connectome actor](concepts/connectome-actor.md) — Circuit inventory, dynamics, independently sized interface MLPs, structured sensory routing and 6D orientation, exact central-critic inputs, connectome-versus-LSTM auxiliary value gradients, propagation delay and saturation boundaries. Last updated 2026-09-17.
- [Fixed-input reservoir controller](concepts/fixed-reservoir-controller.md) — Parameter-free robot-to-MaleCNS population coding, cached motor features, no-BPTT PPO boundary, optional hard-spiking LIF dynamics, exact channel map and smoke evidence. Last updated 2026-09-16.
- [Adaptation and custom compute](concepts/connectome-adaptation.md) — Independent recurrent/interface controls, adapters-only default, and GPU implementation. Last updated 2026-09-14.
- [Training and evaluation success metrics](concepts/success-metrics.md) — TensorBoard populations, tolerance scaling, policy-ID mismatch and dual fixed/checkpoint-tolerance milestone videos. Last updated 2026-09-17.
