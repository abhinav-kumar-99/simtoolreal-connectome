# Approved 1,952-neuron training runs

The exact path-plus-sensory-premotor candidate uses matched neuron-gains and adapters-only policies with old optimizer timing; failed histories are preserved and a fresh stable-KL restart contract is available.

Last updated: 2026-09-14

Related: [Selection evidence](front-leg-pathway-coverage.md), [Timing comparison](adaptation-100b-update-timing.md), [Experiment workflow](../workflows/connectome-experiments.md)

## Exact circuit and interfaces

The user approved **1,952**, without the additional feedback/coordination candidates discussed afterward. Preparation reproduces the 1,285 seed union, the 1,596-cell shortest-path footprint, and all 356 additional direct sensory-to-front-motor intermediates. No count quota, degree ranking, additional feedback cells, or isolate pruning is applied. The result has 33,720 directed edges at >=5 contacts and 44 isolated cells. The broader 4,778 candidate remains a reference, not the training graph.

`configs/connectome/malecns_1952.yaml` pins source hashes, selection masks, expected stage counts, and SHA256 of sorted little-endian int64 body IDs (`d6e3d907840e3f88d17b2c3807e9387c625efd9f3508027001725f410e7c5abc`). A differing shortest-path tie-break result fails identity validation rather than silently changing the candidate. SciPy 1.10.1 reproduced the approved set.

Population interfaces explicitly use the original selection masks: 384 proprioceptive/tactile sensory cells receive the 128 robot sensory features; 157 selected descending cells receive 12 goal features plus the 32-dimensional SAPG embedding; 135 front-leg motor cells feed the 29-action readout. Other retained cells participate through recurrence, not additional direct adapter ports. Dense adapters remain bias-free. The observation vector is unchanged and does not gain true tactile measurements merely by including tactile neurons.

Public unsigned reference counts are signed by source consensus neurotransmitter, with the existing model convention of ACh positive and all other labels negative. The manifest records 1,019 ACh, 604 GABA, 272 glutamate, 53 unclear and four missing labels. Unknown-label signs and glutamatergic inhibition are model assumptions, not per-cell physiological validation. Reference edges retain the audit's explicit VNC ROI/confidence scope; this is not an exact induced subset of the original upstream signed CSV. Global spectral normalization gives raw radius 493.6887786 and scale 0.0020255676113113096, targeting radius one before learned gains.

The artifact is `data/connectomes/processed/malecns_1952/biological.npz`, SHA256 `bdd88d94570b918634ed165486b80915c5fc3ff3bb1e472b4f8a9ee14b6d38e2`. Its directory includes `manifest.json`, annotated `neurons.csv` with stage membership, and unsigned selected `edges.csv.gz`. Preparation refuses to replace an existing different NPZ. The original 4,310 artifact is untouched.

## Training and default timing

`SimToolRealConnectome1952GainsSAPG` inherits adapters plus neuron-gain adaptation, frozen leak/bias dynamics, and fused Triton recurrence. The new job starts fresh at seed 42 on physical GPU 0; it does not transfer the original graph's checkpoint. Task reward, perturbations, exploration, horizon/sequence length 16, and budget match the existing old-timing run.

`SimToolRealConnectome1952AdaptersSAPG` uses the same artifact, interfaces, Triton recurrence, seed, environment contract, and old update timing on physical GPU 1, but freezes incoming/outgoing gains at one as well as leak and bias. Its recurrent operator therefore remains exactly the prepared spectrally normalized matrix while only adapters, heads, SAPG embeddings, and action log standard deviations learn.

- 12,288 environments, six blocks of 2,048.
- One rollout per phase: 196,608 fresh frames; `rollout_accumulation_steps: 1`.
- Actor and critic logical/physical minibatches both 49,152, with two mini-epochs: eight scheduled optimizer calls each per phase.
- 508,626 phases yield 99,999,940,608 frames, below the 100-billion cap.
- Inference milestone every 250 million frames; ordinary recovery saves every 3,000 epochs, best-save eligibility after 100 epochs.

The shared `SimToolRealConnectomeSAPG.yaml` now defaults to accumulation one and 49,152 logical minibatches. Physical microbatch defaults interpolate their corresponding logical minibatch, so small smoke overrides still compose correctly. Historical suites retain their explicit timing overrides; those are opt-in experiment contracts, not the new profile default. The launch YAML explicitly pins the full old environment/update geometry.

## Validation and launch evidence

Twenty-six focused tests passed across compact selection, original graph preparation, profile/suite configuration and front-leg auditing. Tests check seed/isolate retention, graph direction, complete two-edge motif inclusion, absence of padding, input-order stability, explicit sign policies, exact live body IDs/CSR magnitudes/population masks, and resolved old timing.

The full-size smoke used 12,288 environments and the intended 49,152-sample physical minibatches. It completed two phases and 393,216 frames in 33.04 seconds of child training, then reloaded the final recovery checkpoint with finite `(1, 29)` deployment actions. All saved model tensors were finite; all actor Adam states had step 16, and both gain tensors had changed from zero initialization. Evidence: `train_dir/connectome/adaptation_1952_smoke/suite_results.json` and its per-run `verification.json`. This is an execution gate, not evidence of learned dexterity.

The full run started around 2026-09-14 00:10 UTC (September 13 local time), in tmux `connectome-1952`: coordinator PID 3211542, training PID 3211659. At the initial check, TensorBoard served 18 actor/critic loss points through logged frame 3,342,336 with finite values. The old graph's GPU-1 training PID 3172351 remained alive. The stopped new-timing job's artifacts were preserved; its watcher was replaced by an old-only watcher, retaining the existing evaluation history. The compact watcher is in `connectome-1952-eval`, configured for three mean-action videos per 250M-frame milestone. Training is ongoing, not complete.

At the user's direction, the 4,310-cell GPU-1 trainer and its dedicated watcher were stopped after the training log reached frame 928,579,584. Its milestone/checkpoint/evaluation artifacts remain in place; the latest full recovery checkpoint is `last/model.pth` at frame 904,396,800 and the latest inference milestone is 750,059,520. No directory was removed or overwritten.

The fresh adapters-only replacement started in tmux `connectome-1952-adapters`: coordinator PID 3248406 and trainer PID 3248466. The milestone watcher is PID 3248583 in `connectome-1952-adapters-eval`. The resolved configuration confirms `weight_mode: adapters_only`, `learn_dynamics: false`, `operator_backend: triton_fused`, the 1,952/33,720 graph, 12,288 environments, 49,152 minibatches, one rollout per phase, seed 42, and the 100B cap. Its first audit found 160 finite scalar tags and `rewards/step=34.874` at frame 1,769,472. TensorBoard 6008 discovered the new run under the shared parent log directory.

## 2026-09-14 numerical-failure status

The gains trainer exited nonzero after 11,574 phases and 2,275,344,384 logged frames. The immediate exception was `RuntimeError: normal expects all elements of std >= 0.0` while sampling the rollout action. Its last complete recovery checkpoint is epoch 11,400/frame 2,241,331,200 with 91,164 applied actor Adam steps; the last inference milestone is 2,250,178,560 and all nine available milestones have their three mean-action videos.

The adapters-only trainer remained process-alive on GPU 1 after `losses/a_loss` and `info/kl` became `NaN` at approximately frame 3.19 billion. At the user's direction, its trainer and dedicated watcher were interrupted at 2026-09-14 07:20 EDT. The final log coordinate is epoch 17,066/frame 3,355,115,520. Its last complete recovery checkpoint is epoch 17,000/frame 3,342,336,000 with 129,745 applied actor steps versus 136,000 scheduled calls, showing 6,255 mixed-precision-suppressed calls. The applied count had not increased since the earlier epoch-16,800 checkpoint. Its first 13 milestones through 3,250,126,848 have all 39 requested mean-action videos. All artifacts are preserved.

TensorBoard remains on port 6008. Both compact trainers are stopped; the failed gains watcher remains alive waiting for checkpoints that will not arrive, while the adapters watcher is stopped. A separately launched eligibility trainer appeared on physical GPU 0 after the status audit and was not touched by the adapters shutdown.

## Root cause: unbounded exploration log standard deviation

Correction: the saved `a2c_network.sigma` tensor is the Gaussian **log** standard deviation, despite its name. Negative saved entries are valid. `continuous_a2c_logstd` exponentiates this tensor before constructing the Gaussian. The failure message therefore did not establish a negative learned standard deviation; in this run it was produced after the exponentiated value became numerically invalid.

Both policies experienced the same exploration-distribution runaway. Their six coefficient-conditioned log-standard-deviation rows are unconstrained. The positive SAPG entropy terms subtract `entropy_coefficient * entropy` from the minimized PPO loss, so they continually push the higher-entropy rows upward unless the task-policy gradient counteracts them. The adaptive KL scheduler also raised the nominal `1e-4` actor learning rate as high as its hard maximum `1e-2`. The gains block-0 log-standard-deviation mean/max grew from 0.979/1.592 at 250 million frames to 84.800/86.463 at 2.25 billion; adapters-only grew from 1.054/1.501 to 86.393/87.497 by 3.25 billion. The corresponding standard deviations are approximately `exp(log_std)`. Float32 `exp` overflows near 88.7, sampled-action multiplication can overflow before that, and the existing KL calculation squares standard deviations and can overflow once `log_std` is only about 44.4.

The gains policy reached the dangerous range earlier because its PPO/KL trajectory drove the highest-entropy row upward much faster; the extra gain parameters changed that trajectory but were not themselves the failing operation. The adapters-only recurrence was exactly frozen and still developed the same action-distribution problem. All actor-core and critic tensors in the last recovery checkpoints are finite, and the exception occurs in Gaussian sampling after recurrence. This evidence does not implicate the Triton recurrent kernel or a connectome hidden-state overflow.

The old-timing compact contracts schedule eight actor optimizer calls per 196,608 fresh frames, twice the released policy's eight calls per 393,216 fresh frames. This doubles entropy-pressure opportunities per environment frame and is one reason matching only the released entropy coefficient `0.005` did not reproduce the release's training dynamics. At about one billion frames, the matched-release-timing 4,310-cell runs still had block-0 log-standard-deviation means of 1.308 (adapters) and 2.039 (gains), whereas the old-timing compact runs were already at 4.031 and 9.432. Architecture, seed and graph also differ, so this comparison isolates neither timing nor actor by itself.

Mixed-precision `GradScaler` prevented many corrupt gradients from being applied, but it is not a recovery mechanism: after the forward loss became non-finite it skipped optimizer steps while leaving the already extreme distribution parameter in place. This explains why the adapters process kept accumulating frames while its actor step counter stopped. Mean-action evaluation videos do not sample this distribution, so they can remain finite even when stochastic training is broken.

## Commands and configuration ownership

### Stable-KL fresh restarts (2026-09-14)

The user requested a numerically stable KL, LR reduction for negative/non-finite KL, and threshold `0.004`. `torch_ext.policy_kl` now computes Gaussian KL from normalized mean differences and scale ratios using float64 intermediates and `expm1`, without squaring absolute float32 standard deviations or adding biased denominator epsilons. Invalid means/scales produce infinite KL. `AdaptiveScheduler` reduces LR by 1.5 for negative/non-finite KL, bounded below by `1e-6`; every return respects configurable bounds. With threshold `0.004`, finite KL below `0.002` increases LR by 1.5 and above `0.008` decreases it. This is feedback, not a hard KL cap.

`SimToolRealConnectomeSAPG.yaml` owns `params.config.kl_threshold: 0.004`, `min_lr: 1e-6`, and `max_lr: 1e-2`; the paired suite explicitly repeats the threshold and ceiling. The user briefly requested `1e-3`, then restored `1e-2` before the replacement launch. Initial actor LR remains `1e-4`; critic LR and entropy/exploration settings are unchanged. The generic scheduler also retains its `1e-2` default. No log-standard-deviation bound or entropy intervention was added, so this does not guarantee against cumulative variance growth.

New `info/scheduler/mini_epoch_{0,1}/{kl,lr_before,lr_after,invalid_kl}` scalars expose every standard-schedule decision at the same frame coordinate as existing aggregate PPO logs. Existing reward/KL/LR axes are unchanged. The suite starts both 1,952-cell policies from seed 42, without loading already-diverged checkpoints. It retains old timing: 12,288 environments, horizon 16, one rollout per phase, 49,152 logical/physical actor and critic minibatches, and two mini-epochs (eight actor optimizer calls per 196,608 frames). The cap remains 100B frames, with three 800x450, 20-FPS mean-action evaluation videos every 250M frames. Gains uses GPU 0 and adapters-only GPU 1. Separate eligibility jobs remain untouched and share the GPUs; their contention changes throughput, not the configured PPO update density.

The stable-KL implementation previously passed 117 focused tests. After restoring the `1e-2` ceiling, 52 focused KL/configuration tests and the paired YAML linkage check passed. The fresh `ppo_1952_kl004_lr01_smoke` full-batch gate completed 393,216 frames/two phases per policy without OOM, with 16 applied actor Adam steps, finite model tensors, and finite reloaded deployment actions. Child training elapsed 28.13 seconds (gains) and 25.60 seconds (adapters). Evidence is under `train_dir/connectome/adaptation_100b_gains_update_timing/ppo_1952_kl004_lr01_smoke/`; all earlier smoke artifacts remain preserved. Short execution validation is not evidence of long-run learning stability.

Run from the repository root (do not relaunch into populated output directories):

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_kl004_100b.yaml
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_kl004_milestones.yaml
```

The training entrypoint prepares the pinned graph, launches one process per assigned GPU, and records resolved config, progress and elapsed time. The milestone entrypoint watches inference checkpoints and dispatches the three configured eval cases using mean actions. Suite YAML owns batch/update geometry, GPU mapping, caps and LR overrides; evaluation YAML owns snapshot paths, mean-action selection, video settings and cadence. The active YAML content now names the experiment `ppo_1952_kl004_lr01_100b`, so the commands above create new output rather than reusing the earlier `1e-3` directory. For a new smoke, use `configs/connectome/suites/ppo_1952_kl004_smoke.yaml` after choosing a fresh `name` and `output_directory`. The modified `torch_ext.py`, `schedulers.py` and `a2c_common.py` are imported trainer helpers, not separate launch scripts. TensorBoard 6008 watches the shared parent and will discover the new run automatically.

Launch verified at approximately 08:12 EDT: tmux `connectome-ppo-kl004`, coordinator 3449384, gains trainer 3449479 on GPU 0 and adapters trainer 3449478 on GPU 1. The watcher is PID 3449390 in `connectome-ppo-kl004-eval`. Both resolved configs confirm threshold 0.004 and ceiling 0.001. Initial events contain 168 scalar tags, finite actor losses and nonnegative finite KL through five/six phases respectively; both actually reached the new ceiling and stayed at it instead of increasing beyond it (TensorBoard float32 representation is 0.001000000047). `invalid_kl` was zero throughout this initial audit. This confirms active execution and ceiling enforcement, not eventual 100B completion. Existing eligibility PIDs 3431649/3431650 remained active.

That `1e-3` pair and watcher were stopped at the user's direction before any 250M-frame milestone: the last audited aggregate points were frame 39,321,600 (gains, reward 38.057 at 39,124,992) and 44,040,192 (adapters, reward 35.071 at 43,843,584). Their logs remain visible and their empty milestone status remains preserved. The replacement uses fresh initialization and the distinct `ppo_1952_kl004_lr01_100b` artifact identity; it is not a checkpoint continuation.

The `1e-2` replacement launched in tmux `connectome-ppo-kl004-lr01`: coordinator PID 3454679, gains PID 3454787 on GPU 0, and adapters PID 3454786 on GPU 1. Its mean-action watcher is PID 3454681 in `connectome-ppo-kl004-lr01-eval`. Both resolved configurations report KL target 0.004 and max LR 0.01. At the initial audit, both had reached frame 1,769,472 with finite actor loss/KL, `invalid_kl: 0`, and actor LR approximately 0.00865. TensorBoard 6008 discovered both new event paths. The watcher expects 400 milestones per policy and three videos per milestone, with no initial failures. This is a live-launch check, not training completion.

### Interpretation of KL and the baseline comparison

Raw-event audit (all 11,574 gains and 17,066 adapters PPO records, without TensorBoard display sampling): gains `info/kl` becomes permanently NaN at frame 1,765,343,232, and adapters at 2,670,329,856. The remaining 2,595 and 3,484 logged LR values are constant at 0.01 and 0.006666667 respectively. First actor-loss NaNs arrive much later, at 2,259,812,352 and 3,190,554,624 (intermittent initially). Thus KL feedback failed roughly half a billion frames before actor loss failed. Earlier LR decreases are present: gains 0.0044444446 to 0.002962963 after logged KL 0.35763 at 110,690,304; adapters 0.0005852766 to 0.0003901844 after KL 0.47862 at 101,646,336. These examples are consistent with an active scheduler, not a complete replay of its decisions.

Logging prevents an exact per-decision audit: the scheduler is called after each of two mini-epochs, but `info/kl` is their mean and `info/last_lr` is the LR returned by the final minibatch before the second scheduler call. A reduction after one mini-epoch can be offset by an increase after the other. Near numerical failure the computed KL also takes impossible negative values (adapters -0.12280 at frame 2,669,543,424 with LR 0.01), which satisfy the scheduler's increase condition. The 0.016 threshold is an adaptive trigger, not a hard KL limit or early-stop gate; its NaN comparisons both evaluate false. Lowering the threshold alone does not repair that failure.

KL measures the conditional action distributions, summed over the same 29 action dimensions and averaged over training samples; it is not divided by actor parameter count. Architecture changes the parameter-to-distribution Jacobian, gradients, state visitation and the adaptive learning-rate trajectory, but does not change KL units. A restricted mean actor with the same freely learned 6-by-29 log-standard-deviation table can have a different balance between task and entropy gradients. Whether this balance caused the initial divergence needs gradient measurements or a matched intervention; the checkpoints alone do not establish it.

The environment clamps sampled actions to [-1, 1] before rescaling, while the entropy bonus uses the unclipped Gaussian. At large variance the executed commands approach endpoint saturation, so further variance growth can keep increasing the entropy bonus while adding little useful exploration. KL constrains local distribution changes only: increasing every log standard deviation by 0.01 yields approximately 0.002881 exact KL across 29 dimensions irrespective of its starting magnitude. Repeating such small changes does not bound total variance. A CPU reproduction of the actual `policy_kl` arithmetic returns NaN for log standard deviations 45 and 45.01 because it squares their exponentials, despite that small exact KL. The adaptive scheduler compares KL against thresholds with no non-finite handling, so NaN leaves its current LR unchanged.

The release's successful endpoint does not establish that the authors never encountered instability, and available artifacts do not provide their per-block sigma/gradient history. The claim that smaller parameter count or doubled update density definitively caused the runaway is therefore too strong. The observed variance growth and numerical failure are established; architecture-dependent gradient balance, action clipping, entropy pressure and update density provide plausible interacting causes. A controlled comparison needs the same graph, initialization and task with only update timing changed, plus per-block log standard deviation, clipping fraction, stable KL, entropy/task gradient contributions and applied optimizer-step telemetry.

Run from the repository root. Both compact processes are stopped: **do not rerun either launch command into its populated output directory**. `on_existing: fail` protects those directories; a repaired experiment needs a new suite name and output directory or an explicit, validated resume contract.

```bash
# Prepare/verify the exact graph only (does not train).
.venv/bin/python scripts/prepare_malecns_connectome.py --config configs/connectome/malecns_1952.yaml

# Full-size two-phase training and checkpoint reload gate.
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/adaptation_1952_smoke.yaml

# Historical gains launch command; the recorded process failed and is not running.
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/adaptation_1952_100b.yaml

# Persistent gains milestone evaluator, already launched separately.
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/adaptation_1952_milestones.yaml

# Historical adapters-only launch command; the recorded process was stopped after invalid updates.
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/adaptation_1952_adapters_100b.yaml

# Persistent adapters-only milestone evaluator, already launched separately.
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/adaptation_1952_adapters_milestones.yaml
```

The preparation entrypoint dispatches by `preparation_kind`. Its new helper `simtoolreal_shared/compact_connectome.py` performs hash verification, path/motif selection, population indexing, signing, normalization, and artifact/manifest writing; import it rather than launching it directly. Its inputs are the pinned completed audit tables and public neurotransmitter Feather file. Missing sources must be restored from the pinned snapshot; rerunning the audit may change the compressed-file hash even with identical logical rows, so do not bypass the hash check without verifying provenance and selected IDs.

The suite entrypoint handles preparation, device isolation, resolved YAML, logs, checkpoints, and post-exit deployment verification. Important training YAML keys are `train_profiles`, `gpu_assignments`, `num_envs`, `sapg_block_size`, `rollout_accumulation_steps`, logical/physical minibatch sizes, `epochs`, `max_frames`, milestone interval and `output_directory`. The evaluation entrypoint watches snapshots and invokes the existing video evaluator; its YAML owns paths, GPU, polling interval, mean-action cases, and video settings. The stopped large run used `configs/connectome/evaluation/adaptation_100b_old_only_milestones.yaml`; that watcher is no longer active.

TensorBoard remains at **http://localhost:6008**, watching `train_dir/connectome/adaptation_100b_gains_update_timing`. The compact gains and adapters-only runs are nested under `compact_1952/` and `compact_1952_adapters/`; the API confirmed both are visible with 160 scalar tags. All stopped histories remain visible. Training/evaluation artifacts are ignored and are not committed.
