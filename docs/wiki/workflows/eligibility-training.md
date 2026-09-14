# Online eligibility training

An experimental YAML-selected local-trace actor-critic trains the compact connectome without recurrent backpropagation while retaining the existing deployment and video interfaces.

Last updated: 2026-09-14

Related: [Scientific boundary and design](../analyses/dopamine-inspired-learning.md), [Experiment workflow](connectome-experiments.md), [Compact circuit](../analyses/compact-1952-training.md)

## What is implemented

`params.algo.name: connectome_eligibility` selects a separate trainer; PPO is unchanged. `SimToolRealConnectome1952Eligibility.yaml` inherits the exact 1,952-neuron / 33,720-edge gains actor and freezes leak, bias, embedding and exploration scale. The actor learns 56,060 sensory/descending adapter weights, 3,904 bounded gain parameters, and 3,944 action-readout parameters, totaling 63,908. The original signed base edges do not change. Gains start at one, with Xavier input adapters and a small random readout. No prior robot skill or dopamine chemistry is assumed.

The online loop, for each fresh transition:

1. Normalize the 140 robot features, append conditioning code 50, and advance the actor by one recurrent step. All environments use this same policy; six stored embedding/sigma rows are retained for player compatibility, not SAPG experience mixing.
2. Sample the latent Gaussian with fixed standard deviation 0.3. Use its score `(sample - mean) / sigma²` for credit. Clip/rescale the sampled action for the environment; the score belongs to the latent sample, not a Gaussian density incorrectly evaluated at the clipped action.
3. Advance per-environment local sensitivities. Retain each destination's own temporal derivative, including autapses; drop cross-neuron temporal derivatives. Outgoing gains require an edge sensitivity, reduced to the shared source-gain parameter after feedback. Both gain derivatives include the bounded raw-to-gain transform.
4. Project action scores through fixed random signed feedback (seed 123, scale 0.1/sqrt(actions)); replace motor rows with the exact direct readout derivative. Accumulate parameter credit `Z = gamma * lambda * Z + q_hat`.
5. Compute a TD target with a separate 64/64 tanh value network on detached normalized observations plus recurrent features. Its critic-only next-state lookahead does not advance the persistent policy state or issue a second action. All `done` transitions, including timeouts, terminate the explicitly finite-horizon objective; `bootstrap_timeouts: true` is rejected.
6. Apply actor ascent using `mean_env(clipped_delta * Z)` with global gradient-norm clipping and group learning rates. Update the critic with one Adam/Huber feedforward step. Clear recurrent state, local sensitivities and action-credit traces at episode boundaries.

This is approximate rate-network eligibility, not exact e-prop, an unbiased full recurrent policy gradient, or a literal dopamine circuit. Trace clipping, changing parameters and online observation statistics introduce additional approximations. There is no actor autograd history, PPO replay, central PPO critic, intrinsic exploration reward, learned sigma, or new anatomical edge. The compatibility actor value head is unused/frozen; the separate learned critic is training-only.

Persistent trace tensors alone cost about 0.594 MiB per environment: approximately 228 MiB at 384 environments, or 7.125 GiB at 12,288, before intermediate tensors, simulator and model state. This implementation favors transparent equations and validation; it is not a demonstrated throughput improvement over PPO.

## Run from the repository root

Short from-scratch integration test (96 environments, 32 transitions per environment, 3,072 total frames):

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/eligibility_smoke.yaml
```

Checkpoint continuation test, after that smoke (adds 1,536 frames to reach 4,608 total):

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/eligibility_resume_smoke.yaml
```

Generate two mean-action videos from the smoke's milestone snapshots:

```bash
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/eligibility_smoke.yaml
```

Prepared from-scratch pilot, **not launched during implementation** (384 environments, 160 logging chunks of 16 transitions, 983,040 total frames, 2,560 actor updates):

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/eligibility_1952.yaml
```

The long adapter+gains eligibility contract uses physical CUDA 0, 384 environments and a nominal 100-billion-frame budget. Its final vectorized collection boundary is 100,000,137,216 frames (137,216 above 100B); milestone videos remain on the exact 250M grid through 100B. The actor updates on every vector step. `horizon_length: 512` is only the logging/checkpoint chunk, yielding one TensorBoard point every 196,608 frames—the same environment-frame cadence as the old-timing PPO jobs—and 508,627 chunks:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/eligibility_1952_100b.yaml

.venv/bin/python scripts/run_connectome_milestone_evaluation.py \
  --config configs/connectome/evaluation/eligibility_1952_100b_milestones.yaml
```

The matched adapters-only eligibility contract uses CUDA 1. Here “adapters only” includes the sensory/descending projections and motor action readout; both gain vectors stay fixed at one. Task, seed, critic, exploration, environment count, update cadence, budget, TensorBoard root and video cases match the CUDA-0 run:

```bash
.venv/bin/python scripts/run_connectome_suite.py \
  --config configs/connectome/suites/eligibility_1952_adapters_100b.yaml

.venv/bin/python scripts/run_connectome_milestone_evaluation.py \
  --config configs/connectome/evaluation/eligibility_1952_adapters_100b_milestones.yaml
```

These suites use `on_existing: fail` to protect outputs. The smoke and continuation directories already exist after validation: use a new output directory to rerun, updating evaluation/checkpoint paths accordingly. `on_existing: skip` only reverifies a matching completed run; it does not continue training. Use `checkpoint.mode: resume` in a new run directory with larger absolute `epochs`/`max_frames` for continuation. The pilot's task disturbances follow the compact PPO run; the smaller integration smoke uses inherited task defaults and is not a matched performance comparison.

## Important YAML settings

The suite owns `training.algorithm`, `train_profiles`, `num_envs`, `sapg_block_size`, `gpu_assignments`, `epochs`, `max_frames`, `checkpoint`, `inference_checkpoint_interval_frames`, `output_directory` and compiler/thread environment. Maintain `num_envs = 6 * sapg_block_size` for the legacy suite/task layout. PPO minibatch fields are unnecessary. `horizon_length` is only a logging/checkpoint chunk; eligibility updates happen each vector step, and `seq_length: 1` does not unroll BPTT.

Learning settings live in `isaacgymenvs/cfg/train/SimToolRealConnectome1952Eligibility.yaml` under `params.config.eligibility`. Override per experiment through suite `training.overrides`, e.g. `train.params.config.eligibility.readout_lr: 0.00005`.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `gamma`, `trace_lambda` | 0.99, 0.95 | TD discount and delayed action-credit decay |
| `adapter_lr`, `readout_lr`, `gain_lr` | 1e-5, 1e-4, 1e-6 | Manual ascent rates after global gradient clipping |
| `critic_lr`, `critic_width` | 3e-4, 64 | Adam learning rate and width of both hidden layers |
| `sigma` | 0.3 | Fixed exploration standard deviation, not log-standard deviation |
| `feedback_seed`, `feedback_scale` | 123, 0.1 | Reproducible auxiliary spatial-credit projection |
| `trace_clip`, `td_clip`, `max_grad_norm` | 10, 10, 1 | Elementwise sensitivity/credit and actor TD bounds, global update-gradient bound |
| `bootstrap_timeouts` | false | Only supported finite-horizon terminal convention |

The selected group set and per-step update cadence are fixed in this first implementation. Do not tune inherited PPO learning-rate/minibatch/GAE keys and expect eligibility behavior to change. Single-process GPU/CPU execution only; DDP is rejected. Changing algorithm settings on full resume is rejected. Use a matching actor configuration and graph, not just matching tensor sizes.

The first live launch briefly used `horizon_length: 4096`, logging every 1,572,864 frames. At the user's request, both trainers/watchers were stopped before a 250M milestone, and all partial artifacts were preserved under names ending `_log4096_stopped_20260914_072647`. The authoritative live contracts now use 512. `save_frequency` changed from 10 to 80 chunks, preserving the same approximately 15.7M-frame recovery-checkpoint spacing rather than increasing checkpoint I/O eightfold.

## Checkpoints, TensorBoard and evaluation

`last.pth` stores the actor and normalization, small critic and its Adam state, feedback matrix, eligibility configuration, update/frame/epoch counters, elapsed training time, and curriculum scalars. Resume deliberately starts fresh episodes with zero recurrent/local/credit traces; it is **not bitwise mid-trajectory recovery** and does not restore RNG state. This avoids transferring unfinished credit to unrelated physics states. Graph CSR/value/population buffers must agree when loading training weights.

Weights-only mode can initialize a compatible actor from PPO, ignoring PPO value-normalization buffers and leaving the new critic/traces fresh. It resets exploration sigma to the YAML setting. An inference-only milestone cannot resume the trainer; it can initialize weights or drive the existing player. No fake PPO optimizer state is inserted: suite verification checks real eligibility and critic updates, completion counters, and a deployment reload.

The default output roots are nested under the existing TensorBoard 6008 log directory. Scalar discovery was verified through `http://localhost:6008/data/plugin/scalars/tags`. Use `eligibility/*` for TD magnitude, trace RMS, per-group update norms, clipping/saturation, critic loss, gains and actual update count. `rewards/transition` is mean immediate raw reward; `rewards/step` is completed-episode return versus environment frames; `rewards/time` uses elapsed training seconds. Existing observer success/tolerance metrics remain available. These are not PPO loss/clip-fraction metrics. The legacy observer's own `/time` tags retain their original behavior; use the explicit elapsed-seconds metric and the trainer's reward-time axis for timing comparisons.

The video YAML owns checkpoint discovery, output root, GPU, task cases, mean action selection, episode count, max steps, resolution and frame interval. Evaluation uses only the standard inference player: recurrent dynamics run, but no eligibility or value updates occur. Use `scripts/run_connectome_evaluation.py` for an explicit checkpoint/config pair; the milestone helper builds that existing evaluator's config for each snapshot.

## Helper responsibilities

- `scripts/run_connectome_suite.py`: YAML orchestration, preparation, subprocess/GPU assignment, algorithm-aware overrides and completion verification. This is the training entrypoint to execute.
- `scripts/prepare_malecns_connectome.py` with `configs/connectome/malecns_1952.yaml`: verifies/prepares the existing compact artifact; the suite invokes it automatically.
- `rl_games/rl_games/algos_torch/connectome_eligibility.py`: imported local sensitivities, action-credit traces and manual actor updates; skips gain/edge traces entirely in adapters-only mode; no standalone command.
- `rl_games/rl_games/algos_torch/connectome_eligibility_agent.py`: imported environment/critic/observer/checkpoint loop; constructed by the registered runner through `isaacgymenvs.train`.
- `scripts/run_connectome_milestone_evaluation.py`: executable YAML-owned snapshot discovery/video orchestration, using the unchanged evaluator/player.

## Validation evidence and limits

- Focused regression command passed **98 tests**: `.venv/bin/python -m pytest rl_games/tests/test_connectome_eligibility.py rl_games/tests/test_connectome_network.py simtoolreal_shared/tests/test_connectome_configuration.py simtoolreal_shared/tests/test_compact_connectome.py simtoolreal_shared/tests/test_milestone_checkpoints.py -q`.
- Eight eligibility tests cover exact one-step derivatives with fixed spatial feedback, shared outgoing gains, autapses and local temporal sensitivity, decay/reset isolation, reward-times-trace aggregation, no actor autograd, terminal/time-limit targets, real critic/actor changes, CPU full-checkpoint continuation, TensorBoard counters, YAML composition and inference-checkpoint rejection.
- Isaac Gym smoke finished at epoch 2/frame 3,072/update 32; continuation finished at epoch 3/frame 4,608/update 48. Each suite's `verification.json` confirms finite `[1, 29]` deployment output and `timing.json` is complete. Logs/artifacts are under the two default smoke output roots.
- Comparing smoke and resumed checkpoints confirmed changes in both input adapters, readout/bias and both raw gain vectors, with identical base graph, leak, recurrent bias, embedding and sigma. Gain changes are intentionally very small at these untuned defaults.
- `evals/connectome/eligibility_smoke/milestone_status.json` is complete with two videos. Both are 800x450, 30 frames/1.5 seconds; a frame was visually inspected. Both short `sharpie_marker/write_c` evaluations report 0% task progress.
- No long eligibility job was launched, no existing PPO process was stopped/reconfigured, and no advantage in learning quality, sample efficiency or time-to-success has been established. The wiki's earlier PPO timing snapshot is not a matched benchmark of this new trainer.
