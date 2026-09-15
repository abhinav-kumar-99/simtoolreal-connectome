# SimToolReal Connectome Wiki

Durable project knowledge for the MaleCNS-constrained SimToolReal actor.

Last updated: 2026-09-15

Related: [Overview](overview.md), [Log](log.md)

## Overview

- [Project overview](overview.md) — Scope, invariants, and implementation boundaries. Last updated 2026-09-13.

## Sources

- [MaleCNS front-leg circuit](sources/summaries/malecns-front-leg-circuit.md) — Data identity, provenance, and anatomical-data versus executable-dynamics boundary. Last updated 2026-09-15.
- [SimToolReal training references](sources/summaries/simtoolreal-training-references.md) — Published curve, upstream numerical study export, and released checkpoint metadata. Last updated 2026-09-13.

## Concepts

- [Connectome actor](concepts/connectome-actor.md) — Circuit inventory, dynamics, interfaces, exact central-critic inputs, auxiliary actor-value objective, propagation delay and saturation boundaries. Last updated 2026-09-15.
- [Adaptation and custom compute](concepts/connectome-adaptation.md) — Independent recurrent/interface controls, adapters-only default, and GPU implementation. Last updated 2026-09-14.
- [Training and evaluation success metrics](concepts/success-metrics.md) — TensorBoard populations, tolerance scaling, policy-ID mismatch and milestone Task Progress. Last updated 2026-09-15.

## Workflows

- [Experiment workflow](workflows/connectome-experiments.md) — Preparation, profiling, linear/MLP smoke testing, bounded tanh-policy selection, and training. Last updated 2026-09-14.
- [Eligibility training](workflows/eligibility-training.md) — Online trainer, policy-gradient/TD distinction, input-adapter learning alternatives, YAML/video contracts, early gain audit, and stopped-run status. Last updated 2026-09-14.

## Analyses

- [End-to-end system audit](analyses/system-audit-2026-09-15.md) — All 71 event histories, raw-source reproduction, cross-member KL, internal versus action saturation, and proposed multi-rate neural timing. Last updated 2026-09-15.
- [Minimal dexterity circuits](analyses/minimal-dexterity-circuit.md) — Exact 86/262/408-cell candidates, threshold rationale, and 262 training evidence weakening its full-task recommendation. Last updated 2026-09-15.
- [Dopamine-inspired online learning](analyses/dopamine-inspired-learning.md) — Eligibility rationale, historical design, implementation link and unproven learning/cost hypotheses. Last updated 2026-09-13.
- [Approved 1,952-neuron training](analyses/compact-1952-training.md) — Exact graph, historical launches, clipped-policy handoff and corrected interpretation of scheduler KL. Last updated 2026-09-15.
- [Front-leg pathway coverage](analyses/front-leg-pathway-coverage.md) — Public-reference audit, 4,353/4,589/4,778-cell candidates, and evidence-led compact sensory-motor/feedback additions without a size quota. Last updated 2026-09-13.
- [Sparse recurrent backends](analyses/sparse-backends.md) — 160 benchmark cases, 32 capacity probes, memory tradeoffs and ten completed Isaac Gym smoke gates. Last updated 2026-09-13.
- [One-million-step adaptation pilot](analyses/adaptation-1m-pilot.md) — Five timed policies, 40 actor updates each, matched reward/success metrics, and 15 evaluation videos. Last updated 2026-09-13.
- [Billion-step adaptation run](analyses/adaptation-1b-run.md) — Completed adapters-only and neuron-gains jobs, update-count comparison, TensorBoard-axis semantics, and final evaluation videos. Last updated 2026-09-13.
- [100-billion-step update-timing comparison](analyses/adaptation-100b-update-timing.md) — Historical timing comparison, compact-run milestone snapshots, and numerical-failure status. Last updated 2026-09-14.

## Entities

- None yet.
