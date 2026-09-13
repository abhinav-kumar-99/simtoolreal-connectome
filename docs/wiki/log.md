# Project Log

Append-only record of durable repository work.

Last updated: 2026-09-13

Related: [Index](index.md), [Overview](overview.md)

## [2026-09-13] implementation | Add dual single-GPU capped adaptation training

Changed `SimToolRealConnectomeSAPG` to use the benchmark-winning Triton backend by default. Added the YAML-owned `adaptation_1m.yaml` pilot with five seed-42 adaptation cases, a strict one-million-step ceiling, and two concurrent GPU-owned queues. Extended `run_connectome_suite.py` to record UTC timestamps and training, verification, and total elapsed wall time per run.

The first exact-shape attempt preserved 24,576 environments and 98,304 minibatches but both 24 GB GPUs OOMed at the first asymmetric-critic update after about 29 seconds. The retained timed failure artifacts showed roughly 22.5 GB process use. The pilot was revised to 12,288 environments, 2,048-environment SAPG blocks, 49,152 minibatches, and five epochs/983,040 steps; this keeps the six-block/four-minibatch structure and single-GPU algorithm semantics.

All five revised runs completed and passed optimizer/checkpoint/deployment verification. Added a YAML-owned evaluation parent and a per-case Isaac Gym helper that compute paper 2 cm Task Progress, repository 1 cm `avg_goal_pct`, raw/shaped rollout rewards, and native-camera MP4 videos for three selected evaluation cases per policy.

The final checkpoints each recorded 983,040 frames and 40 actor Adam updates. Training wall times ranged from 29.492 to 31.405 seconds; final raw mean episode rewards ranged from 52.913 to 52.947 and every training success ratio was zero. Thirty closed-loop evaluations completed, with zero Task Progress at both thresholds. Fifteen requested videos were verified as nonempty 100-frame, 10-second simulator recordings.

## [2026-09-13] initialize | MaleCNS connectome actor

Initialized the repository wiki and recorded the front-leg circuit identity, actor boundary, and YAML-driven experiment workflow. Added pinned source configuration and reproducible graph preparation machinery.

## [2026-09-13] implement | Connectome actor and experiment contracts

Added the globally registered sparse recurrent actor, SAPG coefficient routing, biological/frozen/rewired/random Hydra profiles, YAML suite launcher, synthetic profiler, committed provenance manifest, PyTorch3D-compatible rotation fallback, and data/actor/config/deployment tests.

The required 384-environment Isaac Gym smoke completed two epochs with six 64-environment SAPG blocks. It performed optimizer updates, saved a checkpoint with 12 optimizer-state entries, and reloaded that checkpoint through `deployment.RlPlayer` to produce a finite `(1, 29)` action. The second epoch ran at 4,096 total frames per second. A smaller six-environment plumbing suite also passed the same train/save/reload path.

The full RTX 4090 actor profile completed at rollout batch 384 and training shape 384 sequences by 16 steps. The connectome has 109,796 trainable parameters versus 7,811,468 for the LSTM. Median rollout latency was 1.38 ms versus 1.50 ms. At length 16, connectome forward/backward latencies were 20.20/29.49 ms versus 2.18/3.41 ms for the LSTM, with 713.8 MB versus 295.1 MB measured peak GPU memory. These measurements establish that the sparse topology is much smaller in trainable parameter count and competitive for one-step inference, but PyTorch's sparse training path is substantially slower and more memory-intensive at this training shape.

The final focused regression set passes 12 tests. Full multi-seed training remains intentionally unlaunched.

## [2026-09-13] query | Robot-connectome interface

Traced and documented the exact 140-dimensional policy-observation layout, learned sensory and descending adapter shapes, motor-neuron readout, Gaussian action sampling, and the preserved conversion from 29 normalized commands to seven arm and 22 hand position targets. Clarified that the biological circuit supplies recurrent topology rather than a literal fly-to-robot sensor or actuator correspondence.

## [2026-09-13] query | Trainable parameters and sparse backends

Accounted for all 109,796 primary-actor parameters and 2,037,769 asymmetric-critic parameters against the implementation and successful checkpoint. Recorded current official sparse-backend constraints and a benchmark-first optimization order. JAX's experimental sparse module is not the preferred performance migration; native PyTorch tuning, a `torch-sparse` comparison, and a fixed-weight cuSPARSE autograd operator are the prioritized options.

## [2026-09-13] profile | Connectome adaptation and recurrent backends

Documented that neuron gains are topology- and sign-preserving multiplicative diagonal adapters rather than additive LoRA, along with the functional properties they do not guarantee. Added cached native CSR, native COO, dense, and optional `torch_sparse` recurrent backends plus numerical forward/gradient equivalence coverage.

On the local RTX 4090 at rollout batch 384 and training shape `384 x 16`, `torch_sparse` was fastest: 0.780 ms rollout and 17.327 ms forward plus backward, versus native CSR at 1.152 ms and 31.729 ms. It used 822.2 MB peak training memory versus 711.9 MB for CSR. Native CSR remains the dependency-free default pending an Isaac Gym smoke run with `torch_sparse`.

## [2026-09-13] implement | Independent adaptation and custom GPU backends

Made adapters-only/frozen dynamics the primary default; added independent neuron-gain, factorized edge, and edgewise adaptation with optional learned leaks/biases. Preserved legacy configuration interpretation and explicit gains/dynamics topology controls. Added cuSPARSE SpMM/SDDMM autograd and a fused Triton recurrent step/backward. Updated actor/adaptation wiki pages and train profiles. First-order numerical, reset, AMP, optimizer, stream and checkpoint checks passed, including the full 4,310-neuron graph: 69 actor/configuration tests. Custom benchmark and Isaac Gym validation are recorded separately when complete.

## [2026-09-13] evaluate | Custom-kernel benchmark and Isaac Gym gates

Added YAML-owned custom benchmark, full-shape capacity, ten-case smoke, and prepared three-seed learning suites. Profiling now resets model/input seeds per case, records state hashes, cold setup, optimizer-inclusive timings, memory and drift, and persists partial/final status. The launcher supports named case overrides, parent/child vendored imports, verified configuration-matched resumption, and requested-epoch checks. Included the extension source in package data. Updated workflow/backend wiki evidence and kept the old topology-control suite's gains/dynamics behavior explicit.

Completed 160 benchmark cases and 32 capacity cases on physical RTX 4090 GPU 1, all without OOM. Adapters-only Triton measured 14.007 ms F+B at `384 x 16` and 138.432 ms including Adam at `6144 x 16`; native CSR measured 28.853 and 435.252 ms. All ten two-epoch Isaac Gym jobs exited successfully and every checkpoint was reverified at epoch 2 with optimizer state and finite `(1, 29)` deployment output. One parent import-path failure after successful training was repaired and the existing checkpoint preserved. A new partial-checkpoint regression test brings the focused total to 70 passing tests. Full learning runs were prepared but not launched. Generated JSON/checkpoints/logs remain ignored; see the backend analysis for limitations and exact artifact paths.
