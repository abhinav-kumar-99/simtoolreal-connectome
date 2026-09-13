# Billion-Step Adaptation Run

The completed adapters-only and neuron-gains policies reproduce the released checkpoint's effective rollout and logical optimizer schedule with rollout and gradient accumulation. They use the default fused Triton recurrence and ran concurrently, with one independent SAPG child on each physical GPU. The original LSTM control remains deferred.

Last updated: 2026-09-13

Related: [Experiment workflow](../workflows/connectome-experiments.md), [One-million-step pilot](adaptation-1m-pilot.md)

## Training contract

The YAML entry point is `configs/connectome/suites/adaptation_1b_release_settings.yaml`. Its ordered case-to-device mapping is:

| Policy | Train profile | Physical GPU |
| --- | --- | ---: |
| adapters-only | `SimToolRealConnectomeSAPG` | 0 |
| neuron-gains | `SimToolRealConnectomeGainsSAPG` | 1 |

Each update phase collects two consecutive `12,288 x 16` rollouts with unchanged weights, for the released effective batch of 393,216 fresh transitions. GAE remains horizon 16 and bootstraps at each boundary, while simulator and recurrent state continue between the two collections. The run uses 2,543 update phases, producing 999,948,288 transitions below the strict 1,000,000,000-step cap; another complete phase would exceed it. Each mini-epoch has four logical 98,304-sample minibatches. With two mini-epochs, each actor and critic schedule eight optimizer-step calls per phase and 20,344 over the run.

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

## Completed run

Both jobs completed all 2,543 phases and their final deployment reload checks. Adapters-only recorded 999,948,288 frames in 12,258.462 seconds (3:24:18); neuron-gains recorded the same frames in 12,490.291 seconds (3:28:10). End-to-end time including verification was 12,260.070 and 12,490.913 seconds, respectively. `suite_results.json` records both cases as complete.

Each final actor checkpoint has Adam counter 20,335 rather than the 20,344 scheduled calls. Both therefore skipped nine mixed-precision updates through PyTorch `GradScaler`'s non-finite-gradient guard. The central critic does not use `GradScaler`, so its code executes all 20,344 step calls, but its optimizer state is not serialized and cannot be independently counter-verified from the final checkpoint.

Final TensorBoard training telemetry uses the last pre-increment frame coordinate, 999,555,072:

| Policy | Raw `rewards/step` | Shaped reward | `success_ratio/frame` | Peak raw reward |
| --- | ---: | ---: | ---: | ---: |
| adapters-only | 80.1783 | 0.801784 | 0.000001628 | 101.5146 |
| neuron-gains | 107.6453 | 1.076453 | 0.000008138 | 109.6315 |

The training success telemetry logs a base curriculum tolerance of 0.075. The environment multiplies it by `keypointScale: 1.5`, giving an implemented maximum-keypoint threshold of 0.1125 m. It is not deterministic trajectory Task Progress; see [success metrics](../concepts/success-metrics.md) for the tag populations and evaluation thresholds.

For external progress references and remaining source limitations, see [SimToolReal training references](../sources/summaries/simtoolreal-training-references.md). The active contract matches the release's effective rollout size, horizon, logical minibatches and gradient-update density; accumulation changes physical execution, not the number of weight updates.

The earlier 12,288-environment exploratory jobs were stopped by request after logged frames 139,788,288 for adapters-only and 144,113,664 for neuron-gains. Their latest saved checkpoints are epoch/frame 692/136,052,736 and 734/144,310,272, respectively; the high-resolution evaluation below uses checkpoint metadata rather than the later unsaved log position.

## Comparison with the stopped exploratory run

Both contracts schedule eight actor and eight critic optimizer calls per update phase: four logical minibatches repeated for two PPO mini-epochs. Their fresh-data denominators differ. The exploratory contract updates after one `12,288 x 16 = 196,608`-frame rollout, or one scheduled optimizer call per 24,576 frames. The release-matched contract updates after two such rollouts, or one call per 49,152 frames. Had the exploratory contract reached its cap, it would have scheduled 40,688 calls; the release-matched contract scheduled 20,344.

Checkpoint inspection distinguishes scheduled calls from actor Adam steps actually applied:

| Contract and policy | Checkpoint epoch | Checkpoint frames | Scheduled actor calls | Applied actor Adam steps |
| --- | ---: | ---: | ---: | ---: |
| exploratory adapters-only | 692 | 136,052,736 | 5,536 | 5,534 |
| exploratory neuron-gains | 734 | 144,310,272 | 5,872 | 5,870 |
| release-matched adapters-only | 2,543 | 999,948,288 | 20,344 | 20,335 |
| release-matched neuron-gains | 2,543 | 999,948,288 | 20,344 | 20,335 |

The two missed exploratory steps and nine missed release-matched steps are mixed-precision actor updates suppressed by `GradScaler`. The central critic schedules the same number of calls as the actor but does not serialize its optimizer state, so only its call count is established.

At the exploratory logs' final frame coordinates, raw `rewards/step` was 61.548 for exploratory versus 57.330 for release-matched adapters-only at frame 140,771,328. It was 74.112 for exploratory neuron-gains at frame 144,113,664 versus 59.417 for release-matched neuron-gains at the nearest stored coordinate, 143,917,056. The release-matched runs later ended at 80.178 and 107.645. These are not isolated actor comparisons: the exploratory jobs use twice the optimizer-update density, exploration scale 0.002, force scale 20, torque scale 2, and no force decay; the final jobs use the released-checkpoint settings documented above.

All scalar series in the four relevant event files use the integer passed as TensorBoard `global_step`. In this SAPG loop it is the environment-frame counter before the current update phase is added. Thus the Step axis is comparable in environment-interaction units, with a one-phase left offset: 196,608 frames for exploratory points and 393,216 for release-matched points. The latter is also sampled half as often. The tags ending in `/step`, `/iter`, and `/time` are aliases written at exactly the same global-step values; their suffixes do not create iteration or elapsed-time axes. TensorBoard's Wall and Relative display modes compare elapsed clock time rather than sample exposure and should only be used for throughput. Checkpoint `frame` is the exact post-phase count.

## Reward-axis display audit

The earlier explanation that reward points appear half as often described raw logging only and was incomplete for TensorBoard hover behavior. A live audit of port 6007 found 2,543 raw reward events per final policy, with every adjacent gap exactly 393,216 environment frames. The old adapters and gains files contain 716 and 733 reward events, respectively, with every adjacent gap exactly 196,608 frames. There are no missing reward intervals within any of these series.

The running TensorBoard 2.14.0 server uses the fast gRPC data provider and no `samples_per_plugin` override. Its installed `plugins/scalar/scalars_plugin.py` defaults to 1,000 points per scalar series. Direct HTTP queries to `/data/plugin/scalars/scalars` return only 1,000 points for each final reward curve (including `/step`, `/iter`, and `/time`). Their displayed adjacent gaps have minimum 393,216, median 786,432, and maximum 9,437,184 frames. This additional downsampling explains why hover positions can be much farther apart than twice the old spacing. The old histories are below the 1,000-point limit; the current server watches only the final suite, so old-server responses were not observed in this audit.

The raw wall-clock median intervals are 2.248/2.155 seconds for old adapters/gains and 4.173/4.234 seconds for final adapters/gains. These timing intervals are distinct from environment-frame coordinates and display sampling. To expose all existing points, a future TensorBoard launch can add `--samples_per_plugin=scalars=10000`; this changes the visualization budget without changing training or event files. The audit did not restart the server or alter its configuration.

## Partial-checkpoint high-resolution evaluation

`configs/connectome/evaluation/adaptation_1b_partial_highres.yaml` generated three additional 800x450 H.264 videos per policy: `sharpie_marker/write_c`, `flat_eraser/wipe_smile`, and `flat_spatula/flip_over`. Videos contain 118--200 frames at 20 FPS; shorter files reflect early episode termination. Both policies reached one of 45 waypoints on the spatula case (2.22% Task Progress) and zero on the other two, yielding 0.741% mean progress. Mean raw rewards were 47.342 for adapters-only and 32.434 for neuron-gains. All six files are nonempty, and sampled frames were visually inspected.

## Final-checkpoint high-resolution evaluation

`configs/connectome/evaluation/adaptation_1b_final_highres.yaml` evaluates the two epoch-2,543 checkpoints on the same three cases. It sets `action_selection: mean`; the parent emits `deterministic_actions: true`, and the continuous PPO player therefore executes the Gaussian mean `mu` rather than sampling from its exploration distribution. The worker rejects any attempt to record a video with sampled actions. It records video during the configured-base 0.02 Task Progress pass and separately runs the configured-base 0.01 `avg_goal_pct` pass. Because the inherited `keypointScale` is 1.5, the implemented maximum-keypoint thresholds are 0.03 m and 0.015 m. Each cell contains one deterministic-policy episode, although simulator/environment state can still make a one-episode comparison insufficient for statistical policy ranking.

| Metric | adapters-only | neuron-gains |
| --- | ---: | ---: |
| paper Task Progress, mean | 0.741% | 0.741% |
| paper-pass mean raw reward | 51.512 | 207.312 |
| repository `avg_goal_pct`, mean | 0.000% | 0.000% |
| repository-pass mean raw reward | 53.113 | 315.426 |

Both policies again completed one of 45 spatula waypoints and no marker or eraser waypoints at 0.02 m. Relative to the partial checkpoints, the paper-pass raw reward changed from 47.342 to 51.512 for adapters-only and from 32.434 to 207.312 for neuron-gains, while Task Progress did not change. The stricter metric was not run for the partial checkpoints and therefore has no matched earlier value.

The six final MP4s were regenerated after making mean-action selection explicit in all case and result artifacts. Their metrics and frame counts exactly matched the original deterministic outputs. They are nonempty H.264 at 800x450 and 20 FPS. Adapters-only produced three 200-frame, 10-second files; neuron-gains produced 145-, 171-, and 185-frame files because those episodes terminated early. First, middle, and final frames from every regenerated file were visually inspected and show distinct simulator states with the requested tool present.
