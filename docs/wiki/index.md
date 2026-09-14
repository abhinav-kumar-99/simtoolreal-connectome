# SimToolReal Connectome Wiki

Durable project knowledge for the MaleCNS-constrained SimToolReal actor.

Last updated: 2026-09-13

Related: [Overview](overview.md), [Log](log.md)

## Overview

- [Project overview](overview.md) — Scope, invariants, and implementation boundaries. Last updated 2026-09-13.

## Sources

- [MaleCNS front-leg circuit](sources/summaries/malecns-front-leg-circuit.md) — Data identity and provenance. Last updated 2026-09-13.
- [SimToolReal training references](sources/summaries/simtoolreal-training-references.md) — Published curve, upstream numerical study export, and released checkpoint metadata. Last updated 2026-09-13.

## Concepts

- [Connectome actor](concepts/connectome-actor.md) — Actor dynamics and robot interfaces. Last updated 2026-09-13.
- [Adaptation and custom compute](concepts/connectome-adaptation.md) — Independent controls, adapters-only default, and GPU implementation. Last updated 2026-09-13.
- [Training and evaluation success metrics](concepts/success-metrics.md) — TensorBoard tag populations, tolerance scaling, exploration blocks, and deterministic Task Progress. Last updated 2026-09-13.

## Workflows

- [Experiment workflow](workflows/connectome-experiments.md) — Preparation, profiling, smoke testing, and training. Last updated 2026-09-13.

## Analyses

- [Dopamine-inspired online learning](analyses/dopamine-inspired-learning.md) — Biological evidence, proposed eligibility-trace/readout experiment and measured limits on PPO-update savings; not implemented. Last updated 2026-09-13.
- [Approved 1,952-neuron training](analyses/compact-1952-training.md) — Exact unpadded graph, old-timing defaults, passed full-size smoke and live GPU-0 run on TensorBoard 6008. Last updated 2026-09-13.
- [Front-leg pathway coverage](analyses/front-leg-pathway-coverage.md) — Public-reference audit, 4,353/4,589/4,778-cell candidates, and evidence-led compact sensory-motor/feedback additions without a size quota. Last updated 2026-09-13.
- [Sparse recurrent backends](analyses/sparse-backends.md) — 160 benchmark cases, 32 capacity probes, memory tradeoffs and ten completed Isaac Gym smoke gates. Last updated 2026-09-13.
- [One-million-step adaptation pilot](analyses/adaptation-1m-pilot.md) — Five timed policies, 40 actor updates each, matched reward/success metrics, and 15 evaluation videos. Last updated 2026-09-13.
- [Billion-step adaptation run](analyses/adaptation-1b-run.md) — Completed adapters-only and neuron-gains jobs, update-count comparison, TensorBoard-axis semantics, and final evaluation videos. Last updated 2026-09-13.
- [100-billion-step update-timing comparison](analyses/adaptation-100b-update-timing.md) — Matched neuron-gains policies, milestone snapshots, and mean-action video watcher. Last updated 2026-09-13.

## Entities

- None yet.
