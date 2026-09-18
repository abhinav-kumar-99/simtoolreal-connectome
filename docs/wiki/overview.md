# Project Overview

This repository tests whether a biologically measured recurrent topology improves sample efficiency in SimToolReal.

Last updated: 2026-09-13

Related: [MaleCNS source](sources/summaries/malecns-front-leg-circuit.md), [Actor](concepts/connectome-actor.md), [Workflow](workflows/connectome-experiments.md)

## Scope

The experiment preserves the legacy Isaac Gym SimToolReal environment, SAPG algorithm, asymmetric critic, observations, actions, rewards, and randomization. Only the recurrent actor changes.

That describes the original PPO/SAPG comparisons. The optional [eligibility experiment](workflows/eligibility-training.md) now deliberately changes the trainer, critic and exploration semantics while retaining the compact actor and rollout/deployment interfaces. It is not a matched optimizer-only comparison and has passed integration smokes, not a learning-quality evaluation.

The primary graph is the 4,310-neuron VNC-restricted front-leg circuit derived from MaleCNS. It is not the complete approximately 166,000-neuron MaleCNS. The 4,604-neuron number belongs to the analogous MANC circuit.

## Comparison

The primary outcome is task performance versus environment interactions. GPU-hours, throughput, memory, and trainable parameters are secondary. The unchanged LSTM, frozen biological graph, degree-preserving rewiring, and random sparse graph are the controls.
