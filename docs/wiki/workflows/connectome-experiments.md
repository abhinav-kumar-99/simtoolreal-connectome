# Connectome Experiment Workflow

Experiments are owned by YAML contracts and proceed through data, profile, smoke, and full-training gates.

Last updated: 2026-09-14

Related: [Overview](../overview.md), [Actor](../concepts/connectome-actor.md)

## Eligibility alternative

The separately selected `connectome_eligibility` trainer now supports the compact gains actor, online local traces, a small TD critic and the existing TensorBoard/checkpoint/video interfaces. See the [eligibility runbook](eligibility-training.md) for smoke, continuation and prepared-pilot commands and the approximation/finite-horizon limits. Existing PPO profiles and jobs remain unchanged; useful task learning has not been demonstrated for eligibility.

## Adaptation and custom-kernel suites

The primary actor now defaults to adapters-only with frozen leaks/biases and the measured-fastest `triton_fused` recurrent backend. See [adaptation controls](../concepts/connectome-adaptation.md) for all modes, parameter counts and legacy compatibility.

Run the compact two-policy MLP projection smoke gate from the repository root:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/mlp_projection_smoke.yaml
```

This YAML runs the 1,952-cell adapters-only and adapters-plus-gains MLP profiles concurrently, one per GPU, for two short epochs. The important settings are `train_profiles`, `interface_projections` inherited from each profile, `gpu_assignments`, `max_parallel`, `num_envs`, both minibatch sizes, `epochs`, and `max_frames`. `on_existing: fail` prevents accidental reuse of a prior smoke directory. The suite was added as a reproducible gate but was not launched while the long-running jobs occupied both GPUs. For a longer YAML-owned experiment, select `SimToolRealConnectome1952AdaptersMLPSAPG` or `SimToolRealConnectome1952GainsMLPSAPG` as `train_profile`; no architecture CLI argument is required.

The active full-size adapters-only MLP replacement uses a single-policy gate and long-run contract:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_mlp_adapters_kl004_smoke.yaml
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_mlp_adapters_kl004_100b.yaml
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_mlp_adapters_kl004_milestones.yaml
```

The smoke and long-run YAMLs use the same 12,288-environment, 49,152-minibatch geometry, seed, KL/LR feedback, task perturbations and exploration settings; only their frame budget/save cadence differ. The evaluator uses mean actions and three high-resolution object/task videos every 250M frames. `ppo_1952_kl004_gains_milestones.yaml` is the separate watcher for the surviving gains policy and deliberately reuses its original evaluation state tree.

Old timing is now the connectome profile default: one rollout and 49,152-sample actor/critic minibatches, with physical batches following the logical size. Historical suites can explicitly override this. The approved compact gains run pins 12,288 environments and the complete old geometry; see [1,952-cell preparation, smoke, launch and monitoring commands](../analyses/compact-1952-training.md). Its exact body-ID set is validated, not padded to a size target.

Run the matched custom-kernel benchmark from the repository root:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/custom_backends.yaml
```

Run full-training-shape memory/performance probes, with explicit OOM records:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/custom_capacity.yaml
```

Run ten short Isaac Gym jobs (four adaptation modes plus the gains/dynamics control, each with cuSPARSE and Triton):

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/custom_smoke.yaml
```

The new suites use physical GPU 1 sequentially and pin `CC`/`CXX` to system compilers through `runtime_environment`. On this host an inherited conda compiler otherwise fails compiling Triton's Python 3.8 launcher because its sysroot lacks `crypt.h`. Edit these compiler paths on other hosts. No environment package upgrade is required for the checked PyTorch 2.4/CUDA 12.4/Triton 3.0 stack.

The suite entrypoint owns preparation, GPU assignment, child execution, elapsed-time measurement and checkpoint verification. `training.train_profiles` accepts either a profile string or `{name, train_profile, overrides}`; named cases get distinct run directories, and case overrides supersede shared overrides. `training.max_parallel` defaults to one. When larger than one, each worker owns a GPU queue and still launches ordinary `multi_gpu=false` children. The custom smoke suite uses `on_existing: skip`: existing checkpoints must match the resolved configuration, reach the requested epoch count and pass reload verification. `on_existing: fail` remains available. Choose a new output directory to repeat training rather than reverify it.

The profiling helper `scripts/profile_connectome_actors.py` also accepts only `--config`, using `configs/connectome/profiling/custom_backends.yaml` or `custom_capacity.yaml`. Its YAML owns actor profiles, adaptation/backend overrides, shapes, warmups, repetitions, AMP, seed and output path. Every case resets initialization/input seeds; state hashes verify backend-matched parameters. It reports cold setup, forward/backward, optimizer-inclusive timing, memory and adaptation diagnostics. Results are saved incrementally with `status: running`; only `status: complete` is final. Capacity OOMs remain labeled failures, not smaller substituted batches.

Backend helpers `connectome_ops.py`, `connectome_cusparse.cpp`, and `connectome_triton.py` are imported by the actor, not launched separately. [Their responsibilities and prerequisites](../concepts/connectome-adaptation.md#compute-implementation) explain how to select and build them. `scripts/prepare_malecns_connectome.py` remains the YAML-owned data/provenance helper.

Outputs: `profiles/connectome/custom_backends.json`, `profiles/connectome/custom_capacity.json`, and `train_dir/connectome/custom_smoke/`. The smoke launcher writes progress, resolved configurations, logs and checkpoint reload verifications. These generated files remain ignored.

`configs/connectome/suites/adaptation_1m.yaml` is the capped pilot entry point. It runs five adaptation policies at seed 42, with at most two independent single-GPU jobs at once: one pinned to GPU 0 and one to GPU 1. The exact published 24,576-environment geometry OOMed at the first critic update on both 24 GB GPUs. The executable pilot therefore scales environments, SAPG block size, and both minibatches together by two: 12,288 environments, six 2,048-environment blocks, and 49,152 actor/critic minibatches. It preserves horizon/sequence length 16, four minibatches per epoch, two mini-epochs, and all optimizer/reward/randomization settings. Five epochs produce 983,040 environment steps without violating the strict 1,000,000-step ceiling. Every run writes `timing.json` with child-training, checkpoint-verification and total wall time; the suite result repeats those fields.

Run the capped pilot from the repository root:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/adaptation_1m.yaml
```

Run the two-policy billion-step comparison from the repository root:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/adaptation_1b_release_settings.yaml
```

This contract assigns adapters-only to physical GPU 0 and neuron-gains to physical GPU 1, with two concurrent single-GPU children; it does not train the original LSTM. It uses 2,543 complete 393,216-transition update phases, yielding 999,948,288 environment steps per policy below the strict billion-step cap. Each phase collects two consecutive 12,288-environment, horizon-16 rollouts before optimization; this preserves the release's effective rollout size, GAE horizon and update density without the unstable 24,576-environment PhysX scene, though it has half as many simultaneous environment identities. `minibatch_size` and `central_critic_minibatch_size` remain the released logical size 98,304. The actor/critic physical microbatch is 24,576, including sample-weighted handling of SAPG's enlarged final minibatch. Each optimizer schedules 20,344 step calls; the completed actors applied 20,335 because `GradScaler` suppressed nine non-finite mixed-precision updates. See the [billion-step run page](../analyses/adaptation-1b-run.md) for final timing and metrics.

The important training keys are `train_profiles`, `seeds`, `gpu_assignments`, `max_parallel`, `num_envs`, `sapg_block_size`, `rollout_accumulation_steps`, `epochs`, `max_frames`, both logical minibatch sizes, optional actor/critic physical microbatch sizes, and task overrides. `max_parallel: 2` creates two GPU-owned queues, so a device never receives overlapping policies. Set it to `1` for fully serial execution. The launcher accepts only `--config`; all experiment settings stay in YAML. `rollout_accumulation_steps` collects multiple unchanged-policy horizon batches before concatenation; it should not be replaced with a longer horizon when GAE comparability matters.

Run the 100-billion-step neuron-gains update-timing comparison with one independent child per GPU:

```bash
.venv/bin/python scripts/run_connectome_suite.py \
  --config configs/connectome/suites/adaptation_100b_gains_update_timing.yaml
```

In a separate process, watch its inference checkpoints and produce three high-resolution mean-action videos every 250 million nominal frames:

```bash
.venv/bin/python scripts/run_connectome_milestone_evaluation.py \
  --config configs/connectome/evaluation/adaptation_100b_gains_milestones.yaml
```

`inference_checkpoint_interval_frames` enables atomic, inference-only snapshots without duplicating optimizer and simulator state. The milestone evaluator is restartable through its status JSON and records both nominal target and actual post-update frame. Its YAML owns watcher cadence, checkpoint discovery, policy-to-GPU mapping, evaluation cases, paper tolerance, action selection, and video settings. `action_selection: mean` is mandatory for video capture. See the [100-billion-step analysis](../analyses/adaptation-100b-update-timing.md) for update counts, milestone alignment, and the validated smoke contracts.

After training, run the configured closed-loop evaluation and video export:

```bash
.venv/bin/python scripts/run_connectome_evaluation.py --config configs/connectome/evaluation/adaptation_1m.yaml
```

This entry point reads checkpoints directly from the training suite result, extracts the final TensorBoard reward and success telemetry, and runs the configured evaluation cases in two single-GPU queues. Its important YAML keys are `training_suite_directory`, `policies`, `gpu_assignments`, `metrics`, `episodes_per_case`, `max_steps`, `videos`, and `eval_cases`. The checked pilot intentionally selects three DexToolBench cases and one rollout per case: this satisfies the three-video-per-policy request but is not the paper's full 24-case, five-rollout protocol. Increase `episodes_per_case` and list all repository object/task pairs for a paper-scale evaluation.

The helper `dextoolbench/eval_worker_isaacgym.py` is launched once per policy/metric/object/task case from a generated `case.yaml`. It disables training randomization and delays, loads the demonstrated trajectory, advances goals using the requested tolerance, accumulates raw and reward-shaped returns, and captures the native Isaac Gym camera to MP4. It should normally be invoked through the parent script; a generated case can be reproduced with `.venv/bin/python dextoolbench/eval_worker_isaacgym.py --config <case.yaml>`.

Evaluation YAML may use `policy_sources` to name explicit checkpoint and resolved-config paths when a training suite stopped before producing `suite_results.json`. Video resolution is controlled by `videos.camera_resolution_reduction_factor`; factor 2 produces twice the width and height of the default factor 4 camera. `configs/connectome/evaluation/adaptation_1b_partial_highres.yaml` evaluates three additional marker, eraser, and spatula cases from the stopped adapters-only and neuron-gains checkpoints.

The evaluation contract reports both configured-base definitions found in the sources: paper Task Progress uses 0.02, while the repository's existing `eval_isaacgym.py` reports `avg_goal_pct` with 0.01. Both are percentages of demonstrated waypoints reached. The environment multiplies these values by `keypointScale: 1.5` in the actual furthest-keypoint test, producing 0.03 m and 0.015 m implemented thresholds. See [success metrics](../concepts/success-metrics.md) before comparing these evaluations with TensorBoard training curves.

`configs/connectome/suites/adaptation_full_training.yaml` remains the matched three-seed learning comparison and now explicitly selects Triton. It is not launched as part of the capped pilot. The older `full_training.yaml` explicitly selects GainsDynamics for its biological trainable-core control so its frozen control stays distinct after the default change.

## Gates

1. Prepare and validate the pinned graph artifacts.
2. Run unit tests and sparse-versus-dense actor checks.
3. Profile LSTM and connectome forward/backward resource use.
4. Run a two-epoch, 384-environment Isaac Gym SAPG smoke test and reload its checkpoint.
5. Launch matched multi-seed training only after the smoke artifacts are accepted.

## Entry points

The legacy Isaac Gym stack is Python 3.8. A local ignored virtual environment can reuse the installed `diffusion` conda environment while supplying the two missing mesh packages:

```bash
conda run --no-capture-output -n diffusion python -m venv --system-site-packages .venv
.venv/bin/python -m pip install trimesh==3.23.5 yourdfpy
```

Run the commands below from the repository root with `.venv/bin/python`. The suite launcher sets `PYTHONPATH` for its child processes so the vendored SAPG `rl_games` is used.

Prepare or verify the pinned data and deterministic controls:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/prepare.yaml
```

Run full-shape synthetic GPU profiling of the unchanged LSTM and primary connectome actor:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/profiling.yaml
```

Compare native CSR, native COO, dense, and `torch_sparse` recurrence using the same actor and rollout/training shapes. `torch_sparse` is optional and must match the environment's PyTorch and CUDA versions; for the pinned local PyTorch 2.4/CUDA 12.4 stack:

```bash
.venv/bin/python -m pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
.venv/bin/python scripts/run_connectome_suite.py \
  --config configs/connectome/suites/backend_profiling.yaml
```

The suite file owns the GPU and stages. Its profiling helper config owns the backend list, batch/sequence shapes, warmup and measurement counts, AMP choice, seed, and ignored JSON output path. The actor implementation accepts `native_csr`, `native_coo`, `dense`, `torch_sparse`, `cusparse`, and `triton_fused`; the primary production profile defaults to `triton_fused` based on the matched local benchmark.

For a fast plumbing check on CPU, call the profiling helper directly:

```bash
.venv/bin/python scripts/profile_connectome_actors.py --config configs/connectome/profiling/smoke.yaml
```

For a six-environment, one-epoch Isaac Gym plumbing check that still exercises all six SAPG blocks:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/isaac_tiny.yaml
```

Run the two-epoch Isaac Gym/SAPG smoke gate:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/smoke.yaml
```

After smoke acceptance, launch the matched five-profile, three-seed suite:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/full_training.yaml
```

The lower-level preparation helper is also YAML-only:

```bash
.venv/bin/python scripts/prepare_malecns_connectome.py --config configs/connectome/malecns_4310.yaml
```

Evaluate the completed billion-step adapters-only and neuron-gains checkpoints and generate three high-resolution videos per policy:

```bash
.venv/bin/python scripts/run_connectome_evaluation.py \
  --config configs/connectome/evaluation/adaptation_1b_final_highres.yaml
```

The evaluation YAML owns the explicit checkpoint and resolved-policy-config paths, GPU queues, case list, episode count, action selection, success tolerances, output directory, and video resolution/frame rate. `action_selection: mean` maps to `deterministic_actions: true`, which makes the continuous PPO player execute its Gaussian mean `mu`; the evaluator rejects sampled-action video capture. The parent schedules cases across GPUs 0 and 1, generates an auditable `case.yaml` for every policy/metric/task cell, and invokes `dextoolbench/eval_worker_isaacgym.py`. The worker loads one policy and trajectory, runs Isaac Gym, computes raw/shaped reward and waypoint progress, and writes the requested MP4. Video is enabled only for the paper-metric pass so the stricter metric does not create duplicate movies.

## Configuration ownership

`configs/connectome/malecns_4310.yaml` owns source URLs and hashes, expected graph dimensions, normalization, deterministic control seeds, and artifact paths. Hydra train profiles under `isaacgymenvs/cfg/train/` own network construction and SAPG settings. `params.network.connectome` owns graph variant, interface partitions, adapters, recurrent dynamics, plasticity, numerical dtype, and validation counts.

Suite YAML owns the selected train profiles, seeds, GPU assignment, maximum concurrent jobs, environment count, SAPG block size, epochs, frame cap, minibatch, output directory, W&B settings, environment overrides, and checkpoint mode. The launcher deliberately exposes only `--config`; experiment changes belong in YAML.

## Outputs and completion

Raw downloads, NPZ graphs, profiler JSON, W&B data, training directories, evaluation JSON and videos are ignored. Each training run writes its resolved configuration, combined log, checkpoints, `timing.json`, and `verification.json`. A run is complete only after the child process exits zero, a checkpoint is present, its optimizer state is non-empty, and `deployment.RlPlayer` reloads it to produce a finite `(1, 29)` action. The evaluation parent additionally requires its expected count of nonempty MP4 files before writing a final summary.

The checked smoke gate met all four conditions. Its suite result is generated under `train_dir/connectome/smoke/`; the full actor measurements are generated under `profiles/connectome/`. Both locations remain untracked by design.

## Historical profiling interpretation, before the adapters-only default

On the local RTX 4090, the connectome actor uses 109,796 trainable parameters versus 7,811,468 for the LSTM. In the latest matched profile, native CSR measured 1.155 ms for one-step rollout and 31.950 ms for length-16 forward plus backward, compared with 1.092 ms and 4.743 ms for the LSTM. The backend suite measured `torch_sparse` at 0.780 ms and 17.327 ms respectively. These are local synthetic actor-only measurements, not environment throughput or sample-efficiency results.
