# Billion-Step Adaptation Run

The active adapters-only and neuron-gains policies reproduce the released checkpoint's effective rollout and logical optimizer schedule with rollout and gradient accumulation. They use the default fused Triton recurrence and run concurrently, with one independent SAPG child on each physical GPU. The original LSTM control remains deferred.

Last updated: 2026-09-13

Related: [Experiment workflow](../workflows/connectome-experiments.md), [One-million-step pilot](adaptation-1m-pilot.md)

## Training contract

The YAML entry point is `configs/connectome/suites/adaptation_1b_release_settings.yaml`. Its ordered case-to-device mapping is:

| Policy | Train profile | Physical GPU |
| --- | --- | ---: |
| adapters-only | `SimToolRealConnectomeSAPG` | 0 |
| neuron-gains | `SimToolRealConnectomeGainsSAPG` | 1 |

Each update phase collects two independent `12,288 x 16` rollouts with unchanged weights, for the released effective batch of 393,216 fresh transitions. The run uses 2,543 update phases, producing 999,948,288 transitions below the strict 1,000,000,000-step cap; another complete phase would exceed it. Each mini-epoch has four logical 98,304-sample minibatches. With two mini-epochs, each actor and critic therefore take eight optimizer steps per phase and 20,344 steps over the run.

Both jobs use seed 42. Each physical rollout has six 2,048-environment SAPG blocks; the same sampled off-policy block is used across both halves, producing the same six-block sample counts as a 24,576-environment phase. A physical training microbatch contains 24,576 samples. The first three logical minibatches use four chunks each; SAPG's enlarged 163,840-sample final minibatch uses six full chunks plus one 16,384-sample remainder. Sample-weighted loss scaling, gradient clipping and Adam stepping preserve one update per logical minibatch, while the adaptive scheduler remains once per PPO mini-epoch. The contract also matches the released checkpoint's exploration scale 0.005, force scale 2, probability range `[0.001, 0.1]`, decay 0.99, decay interval 0.08, lifted-only forces, and zero-default torque/velocity impulses. Generated artifacts live under `train_dir/connectome/adaptation_1b_release_settings_rollout_accumulation/` and remain ignored.

At 24,576 simultaneous environments, the full physical 98,304-sample batch and fallback physical sizes 49,152 and 24,576 each OOMed on both 24 GB RTX 4090s before the first optimizer step. Smaller 12,288- and 6,144-sample training chunks passed updates, but adapters-only repeatedly encountered a GPU PhysX contact-preparation illegal-memory-access near epoch 24; the fault boundary did not move when the training chunk halved. Those diagnostic logs are preserved in sibling `adaptation_1b_release_settings_*` directories. Halving the simultaneous environments and accumulating two rollouts avoids the oversized PhysX scene without changing horizon, fresh samples per update phase or optimizer-step density.

The two-phase smoke gate completed, reloaded its checkpoint for deployment, and recorded 24,576 frames with every actor Adam state at exactly 16 updates: eight updates per phase times two phases. Focused CPU regressions also verify full-versus-accumulated critic equivalence and sample weighting for SAPG's enlarged final logical minibatch.

## Operations

Launch the training parent from the repository root:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/adaptation_1b_release_settings.yaml
```

The suite launcher prepares/verifies the pinned graph, assigns the two ordered cases to separate GPU queues, launches ordinary single-GPU SAPG children, and verifies the final checkpoints. It writes a final suite summary only after both children exit successfully.

TensorBoard watches the suite directory:

```bash
.venv/bin/python -m tensorboard.main \
  --logdir train_dir/connectome/adaptation_1b_release_settings_rollout_accumulation \
  --host 0.0.0.0 --port 6007
```

Port 6007 is used on the development host because port 6006 is occupied. The operational state and final measurements belong in the project log; a launched process is not a completed experiment.

## Live launch

The rollout-accumulated release-matched jobs launched at approximately 2026-09-13 13:25 EDT in the durable tmux session `connectome-1b`; `connectome-tensorboard` was repointed to their fresh output. Both crossed the prior epoch-24 PhysX failure boundary and used approximately 19.4--20.3 GB on their intended GPUs. Epoch-10 checkpoints for both policies record frame 3,932,160 and exactly 80 actor Adam steps, confirming eight updates per phase in the live jobs. Early steady-state phases took about 3.9--4.3 seconds; phases after the first episode-boundary/reset activity took about 10.1--10.7 seconds, so final elapsed time must be measured rather than extrapolated from startup. This remains launch evidence only; final checkpoint verification and measured total elapsed time are pending.

For external progress references and remaining source limitations, see [SimToolReal training references](../sources/summaries/simtoolreal-training-references.md). The active contract matches the release's effective rollout size, horizon, logical minibatches and gradient-update density; accumulation changes physical execution, not the number of weight updates.

The earlier 12,288-environment exploratory jobs were stopped by request after logged frames 139,788,288 for adapters-only and 144,113,664 for neuron-gains. Their latest saved checkpoints are epoch/frame 692/136,052,736 and 734/144,310,272, respectively; the high-resolution evaluation below uses checkpoint metadata rather than the later unsaved log position.

## Partial-checkpoint high-resolution evaluation

`configs/connectome/evaluation/adaptation_1b_partial_highres.yaml` generated three additional 800x450 H.264 videos per policy: `sharpie_marker/write_c`, `flat_eraser/wipe_smile`, and `flat_spatula/flip_over`. Videos contain 118--200 frames at 20 FPS; shorter files reflect early episode termination. Both policies reached one of 45 waypoints on the spatula case (2.22% Task Progress) and zero on the other two, yielding 0.741% mean progress. Mean raw rewards were 47.342 for adapters-only and 32.434 for neuron-gains. All six files are nonempty, and sampled frames were visually inspected.
