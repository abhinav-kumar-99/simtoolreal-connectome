# Project Log

Append-only record of durable repository work.

Last updated: 2026-09-13

Related: [Index](index.md), [Overview](overview.md)

## [2026-09-13] initialize | MaleCNS connectome actor

Initialized the repository wiki and recorded the front-leg circuit identity, actor boundary, and YAML-driven experiment workflow. Added pinned source configuration and reproducible graph preparation machinery.

## [2026-09-13] implement | Connectome actor and experiment contracts

Added the globally registered sparse recurrent actor, SAPG coefficient routing, biological/frozen/rewired/random Hydra profiles, YAML suite launcher, synthetic profiler, committed provenance manifest, PyTorch3D-compatible rotation fallback, and data/actor/config/deployment tests.

The required 384-environment Isaac Gym smoke completed two epochs with six 64-environment SAPG blocks. It performed optimizer updates, saved a checkpoint with 12 optimizer-state entries, and reloaded that checkpoint through `deployment.RlPlayer` to produce a finite `(1, 29)` action. The second epoch ran at 4,096 total frames per second. A smaller six-environment plumbing suite also passed the same train/save/reload path.

The full RTX 4090 actor profile completed at rollout batch 384 and training shape 384 sequences by 16 steps. The connectome has 109,796 trainable parameters versus 7,811,468 for the LSTM. Median rollout latency was 1.38 ms versus 1.50 ms. At length 16, connectome forward/backward latencies were 20.20/29.49 ms versus 2.18/3.41 ms for the LSTM, with 713.8 MB versus 295.1 MB measured peak GPU memory. These measurements establish that the sparse topology is much smaller in trainable parameter count and competitive for one-step inference, but PyTorch's sparse training path is substantially slower and more memory-intensive at this training shape.

The final focused regression set passes 12 tests. Full multi-seed training remains intentionally unlaunched.

## [2026-09-13] query | Robot-connectome interface

Traced and documented the exact 140-dimensional policy-observation layout, learned sensory and descending adapter shapes, motor-neuron readout, Gaussian action sampling, and the preserved conversion from 29 normalized commands to seven arm and 22 hand position targets. Clarified that the biological circuit supplies recurrent topology rather than a literal fly-to-robot sensor or actuator correspondence.
