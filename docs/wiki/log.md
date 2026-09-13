# Project Log

Append-only record of durable repository work.

Last updated: 2026-09-13

Related: [Index](index.md), [Overview](overview.md)

## [2026-09-13] launch | Release-matched billion-step replacement

Implemented logical-batch-preserving gradient accumulation for the connectome actor and asymmetric critic, including sample-weighted handling of SAPG's enlarged final minibatch. Added rollout accumulation so multiple unchanged-policy horizon batches can be concatenated before normalization, sequence shuffling and optimization. A focused CPU regression verifies accumulated critic parameters and a single optimizer-counter increment against a full update.

At 24,576 simultaneous environments, the full physical batch and 49,152- and 24,576-sample fallbacks OOMed before an optimizer step; smaller training chunks repeatedly exposed a GPU PhysX contact-preparation fault near epoch 24. Replaced the oversized scene with two accumulated 12,288-environment, horizon-16 rollouts per update phase. The two-phase smoke completed, reloaded for deployment, and recorded the expected 24,576 frames and 16 actor Adam steps.

Restarted adapters-only on GPU 0 and neuron-gains on GPU 1 with released-checkpoint exploration and force settings, 999,948,288 capped steps, and 20,344 scheduled actor/critic updates each. Both crossed the prior failure boundary at approximately 19.4--20.3 GB. Their epoch-10 checkpoints recorded frame 3,932,160 and 80 actor Adam steps. TensorBoard follows the fresh directory on port 6007. The LSTM control remains deferred; completion and final timing are pending.

## [2026-09-13] evaluate | High-resolution partial-policy videos

Stopped the exploratory adapters-only and neuron-gains jobs by request, preserving their logs and best checkpoints. Added explicit-checkpoint evaluation sources and configurable Isaac Gym camera resolution. Evaluated three new marker, eraser, and spatula cases per policy at 800x450 and 20 FPS. All six MP4s passed codec, resolution, duration, frame-count, and nonempty-file checks; sampled frames showed rendered simulator state. Both policies reached one spatula waypoint and none on the other cases.

## [2026-09-13] configure | Release perturbations and fair update scheduling

Verified exploration and force settings directly from the released archive. Prepared `adaptation_1b_release_settings.yaml` with exploration scale 0.005, force scale 2, decay 0.99, and disabled torque/velocity impulses; both actor profiles compose and match the archived settings. Documented that previous OOM evidence does not isolate effective minibatch size, that physical perturbations are mass-scaled, and that logical-batch-preserving microbatch accumulation plus a matched LSTM control is the preferred comparison. No replacement run or accumulation implementation was launched.

## [2026-09-13] query | SimToolReal training-progress references

Audited the paper, main repository, released pretrained archive, and upstream transformer-study branch. Paper Figure 8 supplies a five-seed episode-reward curve through 9 billion environment steps but no raw history. The study branch supplies numerical seed-0 `rewards/step` CSV data through 874,217,472 frames on an easier cuboid-only environment. The released checkpoint is a resumed endpoint at frame 86,049,816,576 with `last_mean_rewards` 13,603.997 and no event files.

Recorded that the live connectome series uses a compatible TensorBoard tag and frame axis but is not strictly paper matched: its half-size environment geometry doubles optimizer updates per environment step, and active exploration/perturbation settings differ from paper Table I. No running process or experiment configuration was changed.

## [2026-09-13] launch | Billion-step adapters and neuron-gains run

Added a YAML-owned comparison that maps adapters-only to physical GPU 0 and neuron-gains to physical GPU 1. Each independent single-GPU SAPG child retains the validated 12,288-environment geometry and default Triton recurrence. The 5,086 complete epochs produce 999,948,288 environment steps and schedule 40,688 actor optimizer updates per policy without crossing the one-billion-step ceiling. Launch and completion are tracked separately.

Launched both workers in the durable `connectome-1b` tmux session and TensorBoard on port 6007 in `connectome-tensorboard`. Both children completed multiple epochs, held their intended primary 24 GB GPU allocations, and produced event streams visible to TensorBoard. Initial epoch timing projects roughly 2.7--3.0 hours; final artifacts and verification are pending.

## [2026-09-13] query | Connectome optimization budget

Validated that actor adaptation still uses SAPG/PPO and distinguished fresh rollout collection from repeated optimization passes. Recorded the pilot's low reported KL, rising adaptive learning rate, per-mini-epoch scheduler calls, and refreshed KL reference in `analyses/adaptation-1m-pilot.md`. Proposed a longer fresh-experience baseline followed by a controlled actor mini-epoch comparison; no training configuration was changed or launched.

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
