# Connectome Experiment Workflow

Experiments are owned by YAML contracts and proceed through data, profile, smoke, and full-training gates.

Last updated: 2026-09-18

Related: [Overview](../overview.md), [Actor](../concepts/connectome-actor.md)

For source-derived skeleton geometry and recorded recurrent-state videos at four times the rollout FPS, see [Anatomical activation videos](anatomical-activity-videos.md). This optional YAML path supports saved-case replay and future ordinary/milestone evaluations.

## Recovery checkpoint contract

Full PPO recovery checkpoints now preserve three separate central-critic layers:
the critic module state (the legacy `assymetric_vf_nets` key), Adam moments and
parameter groups (`central_value_optimizer`), and non-module trainer metadata
(`central_value_training_state`). The metadata contains the critic epoch, frame,
current scheduler LR and recurrent state. Without it, a restored critic kept its
weights and Adam moments but silently restarted its epoch/frame scheduler inputs
at zero. Legacy checkpoints remain loadable: the loader infers critic epoch/frame
from the actor checkpoint and LR from the restored critic optimizer, while making
clear that no old critic recurrent state exists.

The YAML-owned training entry point remains:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config <suite.yaml>
```

Set `training.checkpoint.mode: resume_training_state` and
`training.checkpoint.path` in the suite YAML to restore actor and critic learning
state while deliberately starting a fresh simulator rollout. The suite's terminal
verification now rejects a newly produced PPO recovery checkpoint if critic
weights, optimizer moments, epoch, frame or LR are absent or invalid, and records
the critic optimizer-entry count and training metadata in its verification JSON.
`mode: resume` additionally attempts simulator and in-flight rollout restoration;
that mode has previously been unsafe for this Isaac environment.

Files named `milestone_target_*.pth` are intentionally small inference snapshots.
They contain the actor policy and frame identity only; they do not contain critic
weights or either optimizer and must not be used with `resume_training_state`.
Use a full file under `nn/` or `last/model.pth` for continuation. Trainers that
were already running when this change was made continue writing the old schema
until restarted, because their Python process has already loaded the old code;
the compatibility inference above makes those files resumable.

## Fixed-input cached-reservoir policies

The full-neuron camera-driven tanh reservoir was stopped at the user's direction after its GPU-1 capacity continuation reached logged frame 24,514,560; its artifacts and last checkpoints remain preserved. The physical-GPU-0 trainer uses the 1,952-neuron no-vision learned-MLP policy described in [compact training](../analyses/compact-1952-training.md), with all 1,952 final tanh states feeding the actor MLP and `use_experimental_cv: false`. On 2026-09-18 the otherwise matched GPU-1 fly job with `use_experimental_cv: true` was stopped at printed frame 9,642,442,752. Its first replacement, the 323-unit [parameter-matched LSTM](../analyses/lstm-baseline.md), was subsequently stopped at frame 261,685,248 with artifacts preserved. A physical-GPU-1 official LSTM run at the repository-default 24,576 environments was then stopped at frame 18,087,936. The active fresh replacement uses the official 1,024-unit LSTM/SAPG method, explicit `use_experimental_cv: true`, user-selected seed 42, and the fly job's 12,288 physical environments with six 2,048-environment SAPG blocks. The former GPU-0 motor-only readout was stopped with artifacts preserved.

The fixed-reservoir contract removes learned input adapters and PPO backpropagation through MaleCNS. The exact fixed population map, memory boundary and biological caveats are in [Fixed-input MaleCNS reservoir controller](../concepts/fixed-reservoir-controller.md). The current replacement retains the 144-value Rotation-6D task, four held-input neural updates, LF/1.0 experience reuse, entropy scale `.005`, actor KL target `.004`, LR range `[1e-6,.001]`, 12,288 environments and the privileged central critic. Its Gaussian distribution, coefficient-conditioned scale, sigma-three ceiling and auxiliary actor value loss match the dense-input Gaussian control.

Run the completed matched-Gaussian two-epoch integration gate from the repository root with an empty output directory:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_fixed_reservoir_rot6d_gaussian_lf_entropy1x_sigma3_smoke.yaml
```

Launch the fresh 100B job with:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_fixed_reservoir_rot6d_gaussian_lf_entropy1x_sigma3_100b.yaml
```

The matched tanh control without the actor-side auxiliary value objective uses:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_fixed_reservoir_rot6d_gaussian_lf_entropy1x_sigma3_noaux_smoke.yaml
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_fixed_reservoir_rot6d_gaussian_lf_entropy1x_sigma3_noaux_100b.yaml
```

The full no-auxiliary contract differs from the existing fixed-reservoir tanh job only in output identity and `use_experimental_cv: false`. It retains the tanh circuit, fixed population map, cached motor features, Gaussian/SAPG distribution, LF reuse, entropy incentive, KL/LR settings, central critic, seed, batch geometry and 100B budget. Disabling the flag removes actor `c_loss`; it does not disable `cval_loss`, GAE, reward learning or the privileged central critic.

The active hard-spiking LIF replacement uses:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_fixed_reservoir_rot6d_gaussian_lif_lf_entropy1x_sigma3_smoke.yaml
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_fixed_reservoir_rot6d_gaussian_lif_lf_entropy1x_sigma3_100b.yaml
```

Those LIF suites select `SimToolRealConnectome1952FixedReservoirRotation6DGaussianSigma3SAPGLIF`. `activation: lif` switches the recurrent state to membrane/refractory/spikes; `control_frequency_hz` and `neural_updates` determine the internal timestep; `membrane_time_constant_ms` controls passive voltage decay; `spike_threshold` and `refractory_period_ms` control events; and `input_current_scale` calibrates the fixed `[0,1]` population-rate drive. All PPO, task and output-readout settings stay matched to the tanh job. Run the independent video watcher with `scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_fixed_reservoir_rot6d_gaussian_lif_lf_entropy1x_sigma3_100b_milestones.yaml`; it polls the matching run for 250M-frame checkpoints and does not block training.

`scripts/run_connectome_suite.py` is the only training entry point. It reads GPU placement, profile, environment count, budgets, checkpoint cadence, optimizer settings and task overrides from the selected YAML. `scripts/prepare_malecns_connectome.py` verifies/prepares the pinned graph artifact. `connectome_network_builder.py` constructs and validates the parameter-free population encoders, runs the live reservoir and exposes cached motor features; `a2c_common.py` stores those 135-value features and presents them as a feed-forward PPO dataset. These helpers are imported by the entry point and are not run separately.

Important settings are:

- `training.gpu_assignments: [1]`: physical GPU used by this replacement job.
- `training.epochs` and `max_frames`: two/393,216 for smoke or 508,626/100B for the full contract.
- `fixed_input_encoder.*_mappings`: fixed observation ranges, cell offsets, signed/opponent encoding and physical scales.
- `reservoir_readout`: selects cached motor features, the 128-unit readout and SAPG conditioning at the readout rather than inside MaleCNS.
- `normalize_input: false`: prevents running normalization from changing the fixed physical encoding.
- `dynamics.neural_updates: 4`: recurrent passes per environment control decision.
- `use_experimental_cv`: `true` trains the separate actor value head on cached motor features plus the SAPG embedding; the matched no-auxiliary suite sets it to `false`. The privileged central critic remains enabled in either case.

The Gaussian smoke completed both epochs, 393,216 frames and checkpoint deployment verification. Its warm update took .263 seconds versus about .92 seconds in the stopped structured/BPTT job. `c_loss` and `cval_loss` were independently nonzero, confirming both the actor-side auxiliary value head and privileged central critic trained. Earlier Beta integration artifacts remain preserved, including two initial failed smoke gates whose auxiliary-buffer and gradient-telemetry bugs were fixed.

The structured Gaussian suite/trainer PIDs 332402/332463 were stopped at the user's direction and their artifacts were preserved. The dense capped-Gaussian suite/trainer/watcher PIDs 261900/261951/262494 on GPU 0 were not signaled.

The first fixed-reservoir Gaussian ran with `use_experimental_cv: true` in tmux `connectome-fixed-reservoir-gaussian-lf-100b` as suite/trainer PIDs 387124/387180, with watcher PID 400161. It was stopped at the user's direction around frame 614.4M after completing deterministic three-video milestones at 250,085,376 and 500,170,752 frames. Its checkpoints, videos and port-6008 TensorBoard history remain preserved.

The matched no-auxiliary tanh smoke then completed two epochs/393,216 frames and deployment reload verification on GPU 1. Its actor `c_loss` was exactly zero while privileged `cval_loss` was `.2063`; entropy `41.1513`, KL `.000845` and all scheduler validity flags were finite. This verifies the intended objective removal without disabling the central critic.

The no-auxiliary tanh run was stopped at the user's direction around frame 301.4M after its watcher completed the 250M deterministic-video milestone. Suite/trainer/watcher PIDs 430585/430658/431147 exited and their checkpoints, videos and port-6008 TensorBoard history remain preserved.

The LIF smoke then completed two epochs/393,216 frames, checkpoint reload and finite deployment inference. Its actor `c_loss` changed from `1.4254` to `.6505`, privileged `cval_loss` changed from `.9956` to `.2105`, entropy remained near `41.15`, KL remained below `.0008`, both scheduler invalid-KL flags were zero and every floating checkpoint tensor was finite. The resolved contract confirms `activation: lif`, threshold `.25`, K=4, the fixed population encoder and `use_experimental_cv: true`.

The fresh full LIF job runs in tmux `connectome-fixed-reservoir-gaussian-lif-100b` as suite/trainer PIDs 450554/450610 on physical GPU 1. Its matching video watcher runs in `connectome-fixed-reservoir-gaussian-lif-videos` as PID 450997 and awaits the first 250M checkpoint. At frame 2,949,120, actor `c_loss` was `.95543`, entropy `41.3040`, KL `.0031157` against the `.004` target and both invalid-KL flags were zero; privileged `cval_loss` was `.06722` at frame 3,145,728. A symlink exposes this run to the existing port-6008 TensorBoard without restarting it. The dense capped-Gaussian suite/trainer/watcher PIDs 261900/261951/262494 on GPU 0 remain live and were not signaled.

## Structured sensory routing with 6D orientation

Two fresh 100B suite contracts match the Gaussian and Beta jobs that were live when the structured interface was implemented. They retain seed 42, 12,288 environments, LF experience reuse at ratio one, entropy scale `.005`, KL target `.004`, LR range `[1e-6,.001]`, auxiliary actor value loss, four neural updates, batch sizes, force settings, checkpoint cadence and action-distribution parameters. They change the observation/interface contract and output identity only: palm/object orientation uses 6D rotation-matrix columns, body/efference signals enter grouped proprioceptor adapters, tactile cells receive no direct fabricated input, and context/goal/SAPG enter a 128-hidden-unit descending MLP.

Do not point these profiles at an existing 140-observation checkpoint. Both start fresh with `checkpoint.mode: none` and `on_existing: fail`. The Gaussian and Beta alternatives now both target physical GPU 1, so they are replacement contracts and must not be run concurrently. The older dense-input capped Gaussian remains separately assigned to GPU 0.

From the repository root, launch the capped-Gaussian experiment with:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_4update_structured_rot6d_gaussian_lf_entropy1x_sigma3_100b.yaml
```

Launch the restricted-Beta experiment with:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_4update_structured_rot6d_restricted_beta_lf_100b.yaml
```

`run_connectome_suite.py` is the executable entry point. Each YAML first invokes the preparation stage and then launches exactly one trainer. The preparation helper `scripts/prepare_malecns_connectome.py` verifies the pinned source hashes and graph identity; when necessary, it atomically upgrades the generated NPZ with the proprioceptor/tactile subpopulation arrays. It does not alter recurrent edges or weights. `simtoolreal_shared/compact_connectome.py` performs the deterministic graph/subpopulation construction, while `connectome_network_builder.py` validates the arrays and builds the grouped adapters at policy initialization; neither helper is launched separately for normal training.

Important YAML parameters are:

- `training.task_profile`: selects the 144-value 6D-orientation environment contract.
- `training.train_profiles`: selects structured capped Gaussian or structured restricted Beta.
- `training.gpu_assignments`: physical GPU assignment; both structured alternatives currently use `[1]` so either can replace the other without colliding with the dense-input Gaussian on GPU 0.
- `training.epochs` and `max_frames`: 508,626 rollouts and the 100B-frame cap.
- `training.checkpoint`: `none` means a fresh policy and fresh normalization state.
- `training.overrides`: owns KL/LR, LF reuse, entropy scale, four neural updates, Beta floor or Gaussian cap, and task randomization.
- `params.network.connectome.observations` in the train profile: owns the exact 102/30/12 sensory/context/goal partition.
- `structured_input_adapter.groups`: owns robot DOF and fingertip grouping. The listed fingertip indices follow environment order index, middle, ring, thumb, pinky.

The structured Gaussian now has a milestone-evaluation YAML that loads the run's resolved task/profile so it reconstructs the 144-value environment. It produces parallel fixed-0.02 and checkpoint-training-tolerance video sets from the preserved milestone checkpoint. External deployment and Isaac Sim observation builders that still emit the released 140-value quaternion vector require the same explicit 6D conversion before they can consume these checkpoints. Training checkpoint verification is dimension-aware. The two suites were composed and both actor distributions completed real 1,952-cell CPU forward/backward checks before launch.

The structured restricted-Beta suite was launched fresh on 2026-09-16 after terminating only the prior dense-input Beta suite/trainer/watcher PIDs 4102855/4102915/4103035. Its artifacts remain preserved. It ran in tmux session `connectome-structured-rot6d-beta-lf-100b` with suite PID 319189 and trainer PID 319244 on physical GPU 1 and stopped at frame 163,381,248 when the user selected Gaussian actions. Its last TensorBoard point was finite: entropy 12.85896, aggregate KL .01079517, actor loss .00127945 and critic loss .07226394.

The structured capped-Gaussian replacement ran in tmux session `connectome-structured-rot6d-gaussian-lf-100b`, with suite PID 332402 and trainer PID 332463 on physical GPU 1. Its saved resolved configuration preserves the new 144-value rotation-6D task, grouped linear proprioceptive adapters, zero direct tactile drive and 128-hidden-unit descending MLP. It changed only the policy family to the same Gaussian contract as the existing dense run: `continuous_a2c_logstd`, coefficient-conditioned log standard deviations, hard-clipped executed actions and a differentiable per-coordinate `max_sigma: 3.0` cap. K=4, auxiliary actor value loss, LF/1.0 reuse, entropy scale .005, KL target .004, LR range, task settings and 100B cap remained matched. The job was later stopped at the user's direction to free GPU 1 for the fixed-reservoir experiment; its artifacts through the 250M inference milestone remain preserved. Its restartable watcher command is:

```bash
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_4update_structured_rot6d_gaussian_lf_entropy1x_sigma3_100b_milestones.yaml
```

The pre-existing dense capped-Gaussian suite/trainer/watcher PIDs 261900/261951/262494 remained live on GPU 0 and were not signaled during the structured training handoff.

## Four-update Beta versus clipped Gaussian (2026-09-15)

**Budget correction:** the user requested **100 billion total environment frames per job**, not 1B. The initial 1B pair was stopped and its epoch-10 / 1,966,080-frame checkpoints are the per-policy resume sources in `configs/connectome/suites/ppo_1952_4update_beta_gaussian_100b.yaml`. Run that YAML with the same suite entrypoint below. Its `epochs: 508626` reaches 99,999,940,608 frames (the largest whole rollout below `max_frames: 100000000000`), with unchanged neural updates, objectives, GPU assignment and save cadence. Per-profile `overrides.checkpoint` and `checkpoint_load_mode: resume_training_state` restore the policy, critic, optimizers and counters while starting fresh simulator episodes; the shared `checkpoint.mode: none` merely avoids adding a second shared checkpoint override. The first full `resume` attempts both exited with SIGSEGV before training; their artifacts remain in the original `_100b` tree. The active retry uses the separate `_100b_training_state` output tree under the existing TensorBoard port 6008. Historical 1B artifacts also remain preserved.

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_4update_beta_gaussian_100b.yaml
```

The user-selected follow-up to the [system audit](../analyses/system-audit-2026-09-15.md) retains auxiliary actor value loss in **both** policies and replaces the proposed tanh/no-aux experiment with this fresh pair:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_4update_beta_gaussian_smoke.yaml
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_4update_beta_gaussian_1b.yaml
```

The long YAML assigns Beta to physical GPU 0 and clipped Gaussian to GPU 1, concurrently. Both use the frozen 1,952-cell MLP actor, seed 42, 12,288 environments, horizon 16, 49,152 logical/physical actor and critic batches, two mini-epochs, on-policy-only samples (`use_others_experience: none`), KL .004, initial LR .0001, maximum LR .001 and SAPG entropy scale .005. `use_experimental_cv: true` retains the actor-side value objective alongside the asymmetric central critic. The 5,086-epoch budget is 999,948,288 frames per policy under the 1B cap; inference checkpoints are due every 250M frames and recovery saves every 200 epochs. `checkpoint.mode: none` starts fresh; `on_existing: fail` prevents rerunning into an existing output tree. Edit the YAML name/output directory to repeat an experiment.

`dynamics.neural_updates: 4` caches observation/goal adapter outputs once per control decision, repeats the recurrence four times, and then reads the motor state. State persists across decisions and resets once at episode boundaries; training differentiates through all four updates. The per-update leak is `1-(1-control_leak)^(1/4)`: base leak .5 becomes approximately .1591036, preserving passive retention over a control interval, not the original nonlinear trajectory. This carries current observations over up to three-edge routes within the same decision; it does not remove all propagation lag or internal saturation.

The Beta profile selects `continuous_a2c_beta`, with two trainable state-conditioned heads `alpha=1+softplus(u)` and `beta=1+softplus(v)`, initialized near (2,2), and maps samples with `a=2*x-1`. It uses exact affine-corrected Beta likelihood/entropy and Beta KL. Rollout storage retains unit-interval samples and shapes, whereas deployment receives actual bounded action means. Gaussian sampling/clipping and its unconstrained log-standard-deviation table remain unchanged; its raw-entropy blowup risk is deliberately not hidden by a new clamp. Equal seeds/configurations do not imply identical shared initial tensors because the Beta model has an additional head.

The exact prediction path is observation- and history-conditioned rather than a per-block alpha/beta lookup. Running-statistics normalization is applied to the 140 policy observations; 128 non-goal features pass through a bias-free `128 -> 256 -> 384` ELU sensory MLP, while 12 goal features are concatenated with the selected learned 32-vector SAPG-ID embedding and pass through a bias-free `44 -> 256 -> 157` ELU descending MLP. Those drives enter the fixed 1,952-neuron sparse recurrent circuit for four neural updates together with its carried state. The resulting 135 motor-neuron activations feed two independent biased `135 -> 256 -> 29` ELU MLPs. The historically named `mu` head emits raw alpha logits `u`; `beta_head` emits raw beta logits `v`. Each action coordinate therefore gets its own state-conditioned `(alpha_i, beta_i)`, while both heads share the entire normalized-observation, embedding and recurrent computation upstream.

For the restricted policy, the raw logits become `alpha_i = 1 + softplus(u_i)` and `beta_i = 1 + softplus(v_i)`. Both final layers start with weights uniformly near zero and bias `inverse_softplus(1) ~= .5413`, giving shapes near `(2,2)` at initialization. Training samples `x_i ~ Beta(alpha_i,beta_i)`, sends `a_i = 2*x_i-1` to the environment, and stores the unit-interval sample plus the old shapes. PPO recomputes the shapes from the stored observation/recurrent sequence and differentiates the affine-corrected Beta log likelihood, entropy and analytic Beta KL. Gradients update both shape heads, the learned sensory/descending interfaces and SAPG embeddings; in the current adapters-only profile the recurrent edge weights, gains, leaks and recurrent biases remain frozen. Deterministic evaluation instead emits the physical mean `(alpha_i-beta_i)/(alpha_i+beta_i)`. Holding that ratio fixed, `alpha_i+beta_i` controls concentration; changing the ratio moves the mean and mode.

The entrypoint prepares the pinned graph, launches isolated GPU children, records resolved configurations and TensorBoard events, and verifies final checkpoint reloads. Imported helpers `connectome_network_builder.py` implement substeps and shape heads; `models.py` implements Beta sampling/likelihood/entropy; `torch_ext.py` implements Beta KL; rollout and PPO helpers route stored shapes and the physical mean correctly. These helpers have no separate command. No new automatic video watcher is part of this pair.

Validation: 92 focused tests passed (59 deselected), including native/Triton four-update forward/gradient parity, current-observation sensitivity, reset/sequence equivalence, trainable Beta heads, likelihood round trips, entropy/KL and YAML composition. Both full-geometry smoke jobs completed two epochs / 393,216 frames and checkpoint deployment reloads with finite actions; actor auxiliary value losses were nonzero and entropy/KL finite. Results are in `train_dir/connectome/adaptation_100b_gains_update_timing/ppo_1952_4update_beta_gaussian_smoke/suite_results.json`. These are execution checks, not evidence of improved learning.

### Scheduler comparison with the historical KL-.004/LR-.01 run

The resolved configurations for `ppo_1952_mlp_adapters_kl004_lr01_100b/...adapters_mlp_seed42` and both active `_100b_training_state` cases share actor `lr_schedule: adaptive`, `schedule_type: standard`, initial LR .0001, minimum LR .000001, KL target .004, two mini-epochs and PPO ratio clip .1. The historical maximum LR is .01; both new maxima are .001. Once per mini-epoch, the scheduler averages minibatch KL, multiplies LR by 1.5 below .002, divides it by 1.5 above .008 (also for invalid KL), otherwise holds it, and clips to the configured LR bounds. This is feedback for subsequent updates, not a hard KL constraint or early-stop gate. The two decisions per rollout can offset each other.

The historical and new clipped policies use raw-Gaussian KL; Beta uses analytic Beta KL, unchanged by the affine action mapping. All sum over action dimensions and average over samples. The historical `use_others_experience: lf` / `off_policy_ratio: 1.0` includes cross-member relabeling; the new `none` setting removes that contribution (the retained ratio value is inactive). Thus equal KL targets do not imply identical KL populations; see the audit's no-update cross-member probe. The dataset also refreshes stored distribution parameters after minibatches, so this is not a strict cumulative KL-to-original-rollout bound. All three separate central critics use constant LR .0001: the inherited critic `kl_threshold: .016` is not an active critic LR control. Auxiliary actor-value gradients, in contrast, share the actor optimizer and its adaptive LR. Sources: each saved `resolved_config.yaml`, `a2c_common.py` scheduler/dataset loop, `a2c_continuous.py` KL dispatch, `torch_ext.py`, `schedulers.py`, and `central_value.py`. This comparison did not change runtime settings.

### What `use_others_experience` actually does

This option belongs to the six-member SAPG exploration population, not to ordinary stale replay. The 12,288 environments are six contiguous blocks of 2,048. Their scalar policy identifiers are `[50, 40, 30, 20, 10, 0]`, and their actor entropy-loss coefficients are `[.0025, .0020, .0015, .0010, .0005, 0]` for `expl_reward_coef_scale: .005`. All six identifiers select learned actor and critic conditioning; in the clipped actor they also select separate log-standard-deviation rows.

With `use_others_experience: none`, every rollout still contains all six blocks. Each member trains from the actions it actually sampled under its own identifier, with its own entropy coefficient. `none` only disables cross-member copying/relabeling. `off_policy_ratio` has no effect on this path.

With the historical `use_others_experience: lf` and `off_policy_ratio: 1.0`, each 16-step rollout first produces `6 * 2,048 * 16 = 196,608` samples. The augmenter chooses one random offset from 1 through 5, copies the rollout, rotates only the coefficient identifier in observations and privileged states, and retains the single 32,768-sample source block that the rotation maps to identifier 0. The resulting dataset has seven blocks / 229,376 samples: all six original blocks plus one randomly selected nonzero member's trajectory presented as zero-entropy member 0. Thus member 0 receives its own data plus one other member's data, while every other member retains only its own data. This is the concrete meaning of leader/follower here.

The copied sample is only partially reinterpreted. Its action, stored behavior negative log likelihood, stored distribution parameters, done flags and recurrent hidden state remain those generated by the source identifier. Values are reevaluated after changing the identifier, and the copied return is rebuilt as a one-step bootstrapped extrinsic target in the current entropy configuration; it is not the original rollout's GAE return. PPO then compares the retained source-policy behavior likelihood with the current target-ID-0 likelihood. This makes the copy fresh but off-policy relative to its relabeled conditioning, rather than replay-buffer data from an old checkpoint.

That mismatch has three important effects:

- The PPO ratio need not start at one for the relabeled block, so clipping can suppress it even before a parameter update.
- Scheduler KL compares the target-conditioned current distribution with source-conditioned stored parameters. At the historical clipped checkpoint, the no-update source-to-ID-0 Gaussian KLs were `[101.862, 6.615, 0.00953, 6.111, 30.201, 0]`; the conditioning switch alone can therefore trigger LR reduction.
- The copied recurrent state encodes the source member's preceding trajectory, not a history rerun under ID 0. Rewriting the current observation identifier does not make that recurrent history target-policy-consistent.

The intended benefit is sensible as a heuristic: use exploratory members to collect diverse behavior and give the zero-entropy exploitation member more task data without another environment rollout. The implementation is not standard on-policy PPO and does not perform a full target-policy reroll, recurrent burn-in, or a separate immutable same-conditioned KL reference. It also calls ID 0 the evaluation policy in a code comment, while `deployment/rl_player.py` currently appends identifier 50; this is a real train/evaluation-member mismatch, not just naming.

The current Beta/Gaussian pair explicitly uses `none` to make the action-family/K=4 comparison cleaner and to prevent cross-conditioning KL from driving the LR scheduler. The cost is losing the extra 1/6 batch of exploratory trajectories for ID 0; it does not turn off the six-member population or their entropy coefficients. In the current 49,152-logical-minibatch setup, `lf` creates five physical forward/backward microbatches but still four optimizer steps per mini-epoch, because the enlarged final logical batch is split for gradient accumulation; `none` uses four physical minibatches and four optimizer steps.

Configuration should spell the value explicitly. The mixed-exploration training loop checks `use_others_experience != 'none'`; an omitted/null value therefore enters augmentation rather than behaving like `none`. Only `lf` invokes leader filtering; another non-`none` value retains the full rotated copies. The active suite's explicit YAML override avoids that ambiguity.

The current four-update clipped-Gaussian case versus the named historical `ppo_1952_mlp_adapters_kl004_lr01_100b/...adapters_mlp_seed42` case is therefore **not a single-variable K ablation**. A recursive leaf comparison of their saved `resolved_config.yaml` files found only run/provenance fields plus these meaningful settings:

| Setting | Historical named run | Current Gaussian |
|---|---:|---:|
| Circuit updates per environment action | 1 (pre-knob behavior) | 4, with held input and rescaled leak |
| Cross-member reuse | `lf`, one selected relabeled follower block | `none`, rollout members only |
| Actor maximum LR | .01 | .001 |
| Actor value auxiliary loss | Implicit default `true` | Explicit `true` |
| Start provenance | Fresh seed 42 | Fresh seed-42 pair, then training-state resume at epoch 10 / frame 1,966,080 with fresh simulator episodes |
| Recovery save frequency | 3,000 epochs | 200 epochs |
| Concurrent suite work | Single configured policy | Paired with Beta on GPU 0; Gaussian remains on GPU 1 |

Both otherwise resolve to `continuous_a2c_logstd` with hard-clipped actions, the same frozen 1,952 graph, adapters-only MLP interfaces, 12,288 environments, horizon/sequence 16, 49,152 actor/critic minibatches and microbatches, two mini-epochs, one rollout per update phase, seed 42, KL .004, initial LR .0001, minimum LR 1e-6, PPO clip .1, critic coefficient 4, SAPG entropy scale .005, central critic, 250M inference checkpoints, 508,626 epochs, 100B cap, reward and perturbation settings. `use_experimental_cv` is absent in the historical YAML but continuous PPO defaults it to true, so both train the actor-side value head as well as the separate central critic. The old leader/follower path adds one 32,768-sample relabeled block to each 196,608-sample rollout; the current path does not. Save cadence and host contention affect operations/throughput rather than the mathematical objective.

At the 2026-09-15 comparison snapshot, the historical run was stopped with its latest rolling checkpoint at epoch 23,140 / frame 4,549,509,120. The current Gaussian was active and had crossed the 1.5B inference milestone. Do not compare their latest rewards as matched-exposure evidence; use a common frame window and remember that K, replay population and LR ceiling all differ.

### Evaluation preserves neural-update timing

`scripts/run_connectome_evaluation.py --config <evaluation.yaml>` passes each policy's `policy_config_path` unchanged to `dextoolbench/eval_worker_isaacgym.py`; use the run's **resolved_config.yaml**, not the base train profile. The milestone entrypoint `scripts/run_connectome_milestone_evaluation.py --config <milestones.yaml>` likewise forwards each policy's configured path. K is owned by `train.params.network.connectome.dynamics.neural_updates` in that saved configuration; do not add an outer K-fold action loop. The imported `deployment/rl_player.py` constructs the same network, whose single forward call performs K internal updates before the motor readout, with the same substep leak.

The player now rejects a K mismatch against the checkpoint's nearest `resolved_config.yaml` within four ancestor directories (covering the suite's `rl_runs/<run>/nn` layout). For standalone exported checkpoints without that saved run config, the supplied export YAML remains authoritative and must travel with the weights; K is not itself a state-dict tensor. Legacy configurations without K retain the historical default of one. The evaluation worker records the instantiated `neural_updates` in each case's result JSON (null for non-connectome networks). Neither helper needs a separate timing CLI parameter.

Validated Beta and Gaussian deployment loading at K=1/4/8, counted exactly K recurrent multiplications per action, verified reset parity, and tested mismatch rejection. Also loaded both actual 100B resolved configs and their 1,966,080-frame resume-source checkpoints through RlPlayer: both executed exactly four updates with finite actions and reset parity. That actual-checkpoint probe used native CSR on CPU to avoid allocating evaluation environments on the training GPUs; it was not a full simulator task evaluation. No training restart or new evaluation watcher was needed for this timing fix.

The two active 100B policies have independent restartable video watchers:

```bash
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_4update_beta_100b_milestones.yaml
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_4update_gaussian_100b_milestones.yaml
```

Each watcher polls every 30 seconds for inference-only checkpoints at 250M-frame targets through the near-cap 100B target, evaluates on the model's corresponding physical GPU, and resumes from `milestone_status.json`. Each target produces three deterministic mean-action, high-resolution videos: Sharpie `write_c`, eraser `wipe_smile`, and spatula `flip_over`, one episode each with the paper Task Progress tolerance of .02 m. Beta outputs go under `evals/connectome/ppo_1952_4update_beta_100b_milestones`; Gaussian outputs use the parallel `_gaussian_` directory. The YAMLs point at the active `_100b_training_state` run's `resolved_config.yaml`, so RlPlayer enforces K=4. The watcher helper discovers checkpoints and manages target/retry state; the evaluation helper expands YAML cases and schedules workers; the worker runs Isaac Gym and writes result JSON/video. These helper files are imported by the entrypoint and have no separate operator command.

The fixed-input, cached-reservoir Gaussian run has its own watcher and artifact identity:

```bash
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_fixed_reservoir_rot6d_gaussian_lf_entropy1x_sigma3_100b_milestones.yaml
```

Its YAML points at the live suite's `fixed_reservoir_gaussian` run name and saved `resolved_config.yaml`, evaluates on physical GPU 1, polls every 30 seconds, and targets every 250M frames through 100B. Each milestone writes the same three deterministic mean-action task videos plus metrics beneath `evals/connectome/ppo_1952_fixed_reservoir_rot6d_gaussian_lf_entropy1x_sigma3_100b_milestones`; `milestone_status.json` is its restart state. `run_connectome_milestone_evaluation.py` owns polling, checkpoint discovery and retries; it imports `run_connectome_evaluation.py` to expand the three YAML cases and schedule one worker at a time, while `dextoolbench/eval_worker_isaacgym.py` loads the checkpoint with the saved K=4 policy contract and produces result JSON and MP4 files. The helpers are not separate entrypoints.

### Guarded Gaussian-to-unrestricted-Beta fallback

Run the one-shot numerical guard from the repository root with:

```bash
.venv/bin/python scripts/run_connectome_numerical_fallback.py --config configs/connectome/handoffs/ppo_1952_gaussian_to_unrestricted_beta.yaml
```

The monitor scans complete TensorBoard TFRecords incrementally and validates each newly observed checkpoint's actor, actor optimizer, asymmetric critic and critic optimizer tensors. It replaces the current clipped-Gaussian job only after an unexpected 90-second trainer absence or a non-finite value in action entropy, actor/value/central-value/bounds loss, policy/scheduler KL, reward, or trainable checkpoint state. A high but finite entropy is recorded, not an arbitrary replacement trigger. `auxiliary_stats/off_on_grad_similarity` remains excluded because it is an optional gradient-comparison diagnostic rather than core training state. Normal completion at frame 99,999,940,608 stops the guard without fallback.

On a trigger, the guard targets only the exact Gaussian trainer and its old video watcher, waits for their process trees to exit, then launches `ppo_1952_4update_unrestricted_beta_100b.yaml` fresh on GPU 1. It verifies the replacement trainer appears before starting its matching milestone-video watcher and writes the decision/PIDs to `gaussian_to_unrestricted_beta_monitor/status.json`. It never stops the restricted-Beta trainer on GPU 0. At the user's 2026-09-15 replacement decision, the fallback retained K=4, auxiliary actor value loss, KL target .004, LR range 1e-6 to .001, full batch geometry, task settings, 250M checkpoints and a 100B budget, but changed experience reuse to explicit leader/follower mode (`use_others_experience: lf`, `off_policy_ratio: 1.0`). Each rollout therefore adds one selected 32,768-sample follower block relabeled as ID 0 to the original 196,608 samples, with the cross-conditioned likelihood, recurrent-state and scheduler-KL limitations documented above.

The standard Beta profile remains `alpha,beta = 1 + softplus(raw)`. The fallback profile alone sets `beta_min_shape: 0.0001`, giving `alpha,beta = 0.0001 + softplus(raw)`: shapes remain numerically positive but can move below one to represent endpoint-peaked and U-shaped policies. It still initializes near Beta(2,2). The lower floor is numerical protection rather than a biological or empirically optimized value, and unrestricted Beta can introduce sharp endpoint densities and difficult gradients. `connectome_network_builder.py` owns this parameterization; the monitor script owns health/process/launch control; the suite and evaluation YAMLs own all fallback experiment and video settings.

The user-triggered replacement launched on 2026-09-15 after Gaussian PID 3968542 was terminated. The guard stopped the matching Gaussian watcher PID 3990855, then launched suite PID 4088170, unrestricted-Beta trainer PID 4088218 and replacement video watcher PID 4088221. The saved `resolved_config.yaml` confirms `continuous_a2c_beta`, minimum shape .0001, initialization 2.0, K=4, auxiliary actor value loss, `lf`/1.0 experience reuse, KL .004 and the 100B cap. By frame 1,769,472, entropy 16.49099, aggregate KL .00089783, both scheduler KLs, actor/value losses and reward were finite; this is launch health, not learning evidence. The restricted-Beta PID 3968541 continued unchanged on GPU 0.

That unrestricted run was superseded and stopped at the user's next decision. Its last logged frame was 193,462,272, where Beta entropy had fallen to -14,694.9385 and aggregate KL had risen to 3,916.8435; the last saved recovery checkpoint is epoch 800. These finite but extreme values are evidence of severe distribution/concentration instability, not a NaN trigger. Its suite PID 4088170, trainer PID 4088218 and watcher PID 4088221 exited, and its artifacts were preserved.

The active GPU-1 replacement restores the restricted parameterization `alpha,beta = 1 + softplus(raw)` while retaining LF/1.0, KL .004, K=4, auxiliary actor value loss and the fresh 100B contract. It has a separate experiment and artifact identity. Launch or restart its YAML-owned training and milestone watcher from the repository root with:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_4update_restricted_beta_lf_100b.yaml
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_4update_restricted_beta_lf_100b_milestones.yaml
```

`ppo_1952_4update_restricted_beta_lf_100b.yaml` owns the Beta floor/initialization, LF population reuse, KL/LR rules, K, auxiliary value loss, GPU, task, checkpoint cadence and budget. The evaluation YAML owns the matching resolved policy path, GPU, 250M-frame cadence and three deterministic video cases. The suite entrypoint prepares the pinned graph and launches the trainer; the watcher discovers inference checkpoints and calls the shared evaluation worker. The first launch used suite/trainer PIDs 4102855/4102915 and watcher PID 4103035. Its saved resolved config confirms minimum shape 1.0 and initialization 2.0. At frame 1,376,256, entropy 16.48605, aggregate KL .00147602, both scheduler KLs, actor/value losses and reward were finite. This was launch health only. At that time the original restricted-Beta on-policy trainer PID 3968541 remained active on GPU 0, yielding a temporary `none`-versus-`lf` comparison; the later replacement below supersedes that live-status description.

### Fivefold-entropy LF replacement on GPU 0

At the user's direction later on 2026-09-15, the longer-running non-LF trainer PID 3968541, its suite PID 3968492 and matching video watcher PID 3990852 were stopped; the GPU-1 LF trainer/watcher PIDs 4102915/4103035 were explicitly preserved. The stopped on-policy job's last complete rolling checkpoint is epoch 16,400/frame 3,224,371,200. Its training and evaluation artifacts were not deleted.

The fresh GPU-0 replacement uses the same restricted-Beta actor, seed, 12,288 environments, K=4, auxiliary actor value loss, LF/1.0 experience reuse, KL target .004, LR range `[1e-6,.001]`, 250M inference milestones and 100B cap as the existing LF job. Its only intended optimization-setting difference is `expl_reward_coef_scale: .025`, five times the baseline `.005`. The six original SAPG blocks therefore receive entropy-loss coefficients `[.0125, .0100, .0075, .0050, .0025, 0]`; the extra LF block remains labeled ID 0 and receives zero entropy coefficient. Fivefold applies to the loss weights, not to the entropy values themselves.

Launch or restart the fresh contracts from the repository root with distinct empty output directories:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_4update_restricted_beta_lf_entropy5x_100b.yaml
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_4update_restricted_beta_lf_entropy5x_100b_milestones.yaml
```

The suite YAML owns the entropy scale, LF policy, GPU 0, optimizer/KL settings, K, batch geometry, checkpoint cadence and 100B budget. The evaluation YAML owns the matching saved policy configuration, deterministic mean actions, three high-resolution task cases and 250M cadence. The launcher prepares the pinned compact graph and runs the trainer; the watcher only polls checkpoints and dispatches evaluation jobs. The composition test passed. The first launch uses suite/trainer PIDs 9553/9610 and watcher PID 9904. Its resolved YAML confirms scale .025 and all retained settings; at frame 1,769,472, entropy 16.71017, actor loss .000831, actor auxiliary value loss .17892, KL .001066 and reward 30.4754 were finite. This establishes launch health, not improved learning or long-run entropy stability. The active comparison is now LF scale `.005` on GPU 1 versus fresh LF scale `.025` on GPU 0, with unmatched training ages.

### Staged 1.25G true-objective handoff: 5x to 3x to capped Gaussian

The corrected handoff does not replace the active 5x run immediately. It first waits until that run has emitted a `true_objective_mean/frame` point at or above 1,250,000,000 TensorBoard frames. It then computes the arithmetic mean of every logged `true_objective_mean/frame` sample whose frame lies in the inclusive matched window `[1,150,000,000,1,250,000,000]` and compares that 100M-frame mean against the same window from the existing 1x LF reference. Replacement uses strict less-than: 5x is stopped only if its window mean is below 1x. If it is equal or higher, 5x is retained and the chain terminates.

If 5x loses, the monitor stops only its exact trainer, suite and video-watcher process trees and launches a fresh restricted-Beta 3x job plus its watcher on GPU 0. The new job starts from frame zero in a distinct output directory; the monitor then independently waits for that 3x job to cross 1.25G and averages its own `[1.15B,1.25B]` window. It compares 3x to the same fixed 1x window. Only if 3x is also lower does it stop 3x and launch the terminal clipped-Gaussian fallback. The GPU-1 1x reference is never stopped by this monitor. The earlier single-point `mean_successes/frame` contract was superseded before any decision or transition occurred.

Run the restartable YAML-owned monitor from the repository root:

```bash
.venv/bin/python scripts/run_connectome_success_handoff.py --config configs/connectome/handoffs/ppo_1952_lf_entropy_success_handoff.yaml
```

The handoff YAML owns the exact TensorBoard tag, 1.25G target, 100M window, arithmetic aggregation, strict comparison rule, event globs, process markers, stage order, replacement suite/video configs, poll cadence and shutdown/startup timeouts. The monitor incrementally scans complete TensorBoard records, deduplicates values by frame, persists window samples/summaries, decisions and launched/stopped PIDs in `train_dir/connectome/adaptation_100b_gains_update_timing/lf_entropy_success_handoff/status.json`, and calls the existing suite and milestone-evaluation entrypoints. If any comparison-contract field changes while still monitoring and before any transition, it archives the prior summary and clears metric offsets so both histories are rescanned under the new contract. `run_connectome_numerical_fallback.py` supplies shared atomic-status, process-tree and subprocess-launch helpers; it is not separately started for this handoff.

The dormant 3x suite uses entropy scale `.015`, corresponding to original-block coefficients `[.0075,.006,.0045,.003,.0015,0]`. It otherwise matches the active LF runs: restricted Beta, K=4, auxiliary actor value loss, LF/1.0 reuse, KL target `.004`, LR range `[1e-6,.001]`, 12,288 environments, 250M video/checkpoint cadence and a fresh 100B budget. Its direct suite and watcher entrypoints are `configs/connectome/suites/ppo_1952_4update_restricted_beta_lf_entropy3x_100b.yaml` and `configs/connectome/evaluation/ppo_1952_4update_restricted_beta_lf_entropy3x_100b_milestones.yaml`; normally the handoff monitor launches them.

The terminal Gaussian fallback retains that same 3x weighting and all other matched settings. Its `SimToolRealConnectome1952AdaptersMLPGaussianSigma3SAPG` profile applies a smooth ceiling to each emitted standard deviation: raw log standard deviation zero still maps to sigma one, and sigma approaches but cannot exceed three with a nonzero gradient below the asymptote. Existing Gaussian profiles omit `max_sigma` and are unchanged. This is a per-coordinate cap: 29 sigmas can sum to at most 87, but 87 is not an entropy bound. At sigma three in every dimension, the unclipped diagonal Gaussian entropy is about 73.01 nats; environment actions are still hard-clipped to `[-1,1]`. The direct suite and watcher configs are `ppo_1952_4update_gaussian_lf_entropy3x_sigma3_100b.yaml` and its matching `_milestones.yaml`, but the handoff launches them only after both Beta candidates lose their own 1.25G comparisons.

The staged handoff completed both comparisons on 2026-09-16. Against the fixed 1x mean `0.0000419941081`, 5x averaged `0.0000221648921` and was replaced by fresh 3x restricted Beta; 3x then averaged `0.0000265286654` and was also lower. The monitor launched the terminal 3x clipped Gaussian, which was later manually superseded at the user's direction. Its final finite TensorBoard point was frame 1,220,345,856 with entropy 55.69760 and aggregate KL .27626; its artifacts remain preserved.

The active GPU-0 replacement is a fresh clipped-Gaussian LF run with baseline entropy scale `.005`, corresponding to coefficients `[.0025,.002,.0015,.001,.0005,0]`. It otherwise retains LF/1.0 reuse, sigma cap 3 per dimension, K=4, KL `.004`, LR ceiling `.001`, auxiliary actor value loss, 250M inference/video cadence and a 100B budget. Its suite/trainer/watcher PIDs at launch were 261900/261951/262494. The saved resolved configuration confirms every retained setting; at frame 1,769,472 entropy 41.20781, aggregate KL .00088270 and actor/value losses were finite. This is launch-health evidence only. The GPU-1 restricted-Beta 1x reference remained live and untouched.

Run the active replacement contracts directly from the repository root with:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_4update_gaussian_lf_entropy1x_sigma3_100b.yaml
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_4update_gaussian_lf_entropy1x_sigma3_100b_milestones.yaml
```

The suite YAML owns the fresh output identity, entropy scale, LF reuse, sigma-capped profile, K, optimizer/KL settings, batch geometry, checkpoint cadence, GPU and budget. The evaluation YAML owns the matching resolved-policy path, deterministic mean action, three task cases and two otherwise matched video metrics. `paper_task_progress` uses fixed tolerance `.02`; `checkpoint_training_tolerance` reads the exact training tolerance logged at each checkpoint frame. The suite helper prepares/verifies the pinned graph and launches training; the watcher polls inference checkpoints, resolves the checkpoint tolerance from TensorBoard and dispatches evaluations. The evaluation helper reuses matching completed cases, so adding the second set backfills only missing videos. Imported network and launcher helpers are not separate entrypoints.

### Interpreting restricted-Beta entropy

`losses/entropy` is the analytic differential entropy of the physical action density after mapping each independent Beta variable from `[0,1]` to `[-1,1]`. The model adds the affine `ln(2)` correction per coordinate, sums all 29 action-coordinate entropies, and the trainer averages the result over samples and PPO minibatches. It is raw policy entropy: the logged number is not multiplied by the six SAPG entropy coefficients. It is also not environment-state visitation entropy, reward, action magnitude, success probability or a direct count of saturated actions.

Because the support is bounded, the maximum is the uniform policy's `29*ln(2) = 20.1013` nats. The Beta(2,2) initialization is 16.4736 nats. For scale only, a hypothetical identical symmetric Beta(4,4) policy would be 8.9508 nats across 29 dimensions and Beta(5,5) would be 6.1627. Real policies are state-dependent, asymmetric and different across joints, so no single alpha/beta pair can be inferred from aggregate entropy. Continuous differential entropy can legitimately be negative: this means a sufficiently narrow density, not negative uncertainty or an invalid probability.

For these restricted policies, rising toward 20.10 means actions are becoming more diffuse; falling means the learned density is becoming more concentrated and/or skewed. The positive Gaussian-style entropy blowup is structurally removed, but concentration can still increase without bound because alpha and beta have no upper limit, driving entropy toward negative infinity. Large negative magnitude, non-finite values, or abrupt coupled KL spikes are therefore the Beta failure signals. The stopped unrestricted run's -14,694.94 was severe concentration/pathology, not extra exploration.

At the 2026-09-15 21:09 UTC snapshot, the on-policy restricted job was near 2.55B frames with a last-100-point entropy mean of 6.03 nats; its six recent block entropies were approximately `[10.22, 9.63, 9.08, 6.23, 4.55, -4.97]` for IDs `[50,40,30,20,10,0]`. This separation is directionally consistent with entropy coefficients decreasing from .0025 to zero: the no-bonus ID-0 member is already much sharper than the high-bonus members. The LF restricted job was near .70B frames with a last-100-point mean of 5.74 and recent blocks approximately `[6.46, 5.55, 4.85, 4.65, 3.64, 4.49]`. Its global metric weights ID 0 twice because the LF batch contains the original ID-0 block plus one relabeled follower block.

Latest-to-latest values are not exposure-matched. Around the LF job's .70B-frame coordinate, the on-policy job had aggregate entropy about 9.50 while LF was about 5.0-5.3. LF is therefore concentrating materially faster at matched environment exposure; this is not by itself evidence of better control. At the same snapshot LF aggregate KL was around .029, above the .008 LR-reduction boundary, so its entropy trajectory is entangled with the known cross-member KL effect. Use reward/task-progress and videos alongside entropy, and add per-action alpha/beta, concentration and physical standard-deviation telemetry before diagnosing which joints are collapsing from the scalar alone.

LF has several mechanisms that can cause this difference. First, it changes the objective population from six equally represented blocks to seven blocks: the six originals plus an extra block labeled ID 0. That extra block gets ID 0's zero entropy coefficient. The nominal sample-average entropy coefficient therefore falls from .00125 to approximately .001071, a 14.3% reduction, while ID 0's raw sample representation doubles. All coefficient IDs share the recurrent circuit, adapters and Beta output heads, so the added zero-entropy policy/value gradients can change the distributions of blocks 0-4 even though their own entropy coefficients are unchanged. LF processes 16.7% more samples per environment rollout, though the current logical-batch scheme still takes four optimizer steps per mini-epoch rather than adding an optimizer step.

Second, the global scalar itself changes population weights. At exact matched step 817,299,456, the on-policy blocks were `[9.712, 8.779, 7.390, 6.107, 3.330, 1.190]`, with global 6.087. Merely counting that same ID-0 value twice would yield approximately 5.385; this accounts for about .70 of the observed 2.50-nat global gap to LF's 3.590. The remaining approximately 1.80 nats reflects actual policy-distribution differences rather than display weighting. On LF, ID 0 was higher than the on-policy ID 0 (`3.394` versus `1.190`) while blocks 0-4 were all lower. A plausible directional mechanism is that ID 0 is trained on actions sampled by more exploratory followers, which can force it to cover more varied actions when their relabeled advantages are positive; this is not behavior cloning and the entropy scalar alone cannot establish that causal explanation.

Third, LF changes more than entropy weighting. The added samples retain follower actions, old likelihoods and recurrent histories but are evaluated as ID 0, use reconstructed one-step returns, and participate in joint advantage normalization. This changes PPO clipping and the normalized advantages for the entire batch. Cross-conditioned KL also drives the adaptive scheduler; the observed LF KL above its upper threshold produces a very different LR trajectory. These effects mean the current comparison demonstrates an LF-associated entropy change, but does not isolate one clean causal term. A proper ablation would retain the same seven-block batch geometry while separately toggling coefficient relabeling, zero-versus-source entropy weight, recurrent reroll, return construction and scheduler-KL inclusion.

### Where the restricted-Beta densities peak

A 2026-09-15 saved-state probe evaluated each checkpoint's 12,288 saved observations and recurrent states for the next policy decision, giving 2,048 environments for each coefficient ID and 29 state-conditioned Beta distributions per environment. The current rolling snapshots were on-policy epoch 14,000/frame 2,752,512,000 and LF epoch 4,600/frame 904,396,800, so their current values are not exposure-matched. The full epoch-3,200 snapshots provide an exact matched checkpoint coordinate of 629,145,600 frames. Loading the same weights with the numerically equivalent CPU native-CSR backend changed no policy parameters.

The formal modes are often near the action bounds and favor the negative side, but mode location substantially overstates how much probability is actually at the bounds:

| snapshot and sharp block | one-state 29-D entropy | median mode | median absolute mode | modes with `abs(mode) >= .8` | negative / positive bound-side modes | probability mass in `[-1,-.8] U [.8,1]` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current on-policy ID 0 | -6.446 | -0.309 | .998 | 75.3% | 42.5% / 32.8% | 38.6% |
| current LF ID 10 | .423 | -0.182 | .872 | 54.6% | 35.3% / 19.2% | 22.9% |
| matched on-policy ID 0 | 2.997 | -0.143 | .926 | 59.3% | 39.0% / 20.3% | 24.5% |
| matched LF ID 10 | 4.307 | -0.207 | .889 | 56.8% | 36.6% / 20.2% | 20.9% |

A uniform action density puts 20% of its mass in that outer-20% region. Thus the current on-policy ID-0 member is genuinely concentrating toward both endpoints, with more negative than positive endpoint pressure: its outer-20% mass is 38.6%, and its outer-10% mass is 24.3% rather than the uniform 10%. The LF ID-10 member's formal modes also tend toward the endpoints, especially `-1`, but its mass is only mildly more endpoint-heavy than uniform: 22.9% in the outer 20% and 12.1% in the outer 10%. At the matched full checkpoint the same qualitative distinction is already present, though weaker.

The apparent contradiction between a near-bound mode and modest bound probability is real. With `alpha` and `beta` close to one, a tiny skew gives an almost-flat Beta density a formal endpoint or near-endpoint maximum. Float32 also weakens the advertised strict-interior guarantee of `1 + softplus(raw)`: in the current on-policy probe, 22.79% of alpha values and 10.47% of beta values rounded exactly to 1.0; LF had .40% and 2.18%, respectively. If one shape equals one and the other exceeds one, the mathematical mode is exactly at one boundary even when the density is shallow. Adding a small representable margin above one would remove exact boundary modes but would not prevent near-bound modes or excessive concentration.

The sharpest individual on-policy ID-0 coordinates show genuine negative endpoint bias: hand actions 8, 19, 9 and 23 had median modes approximately `-1`, `-.995`, `-1` and `-.999`, with 38.5-42.2% outer-20% probability mass. LF ID-10 was milder: arm action 1 had mode `+1` and 32.5% outer mass, while hand actions 10 and 23 had median modes `-.978` and `-.989` with 26.9% and 27.7% outer mass. Indices 0-6 are arm commands and 7-28 are hand commands. Deterministic evaluation executes the Beta mean, not its mode, so a boundary mode does not imply that evaluation videos command an endpoint. The durable telemetry should therefore log per-block physical-action mean and standard deviation, exact-floor frequency, mode, and analytic outer-10%/outer-20% probability mass; entropy or mode alone is insufficient.

## Eligibility alternative

The separately selected `connectome_eligibility` trainer now supports the compact gains actor, online local traces, a small TD critic and the existing TensorBoard/checkpoint/video interfaces. See the [eligibility runbook](eligibility-training.md) for smoke, continuation and prepared-pilot commands and the approximation/finite-horizon limits. Existing PPO profiles and jobs remain unchanged; useful task learning has not been demonstrated for eligibility.

## Adaptation and custom-kernel suites

The primary actor now defaults to adapters-only with frozen leaks/biases and the measured-fastest `triton_fused` recurrent backend. See [adaptation controls](../concepts/connectome-adaptation.md) for all modes, parameter counts and legacy compatibility.

Run the compact two-policy MLP projection smoke gate from the repository root:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/mlp_projection_smoke.yaml
```

This YAML runs the 1,952-cell adapters-only and adapters-plus-gains MLP profiles concurrently, one per GPU, for two short epochs. The important settings are `train_profiles`, `interface_projections` inherited from each profile, `gpu_assignments`, `max_parallel`, `num_envs`, both minibatch sizes, `epochs`, and `max_frames`. `on_existing: fail` prevents accidental reuse of a prior smoke directory. The suite was added as a reproducible gate but was not launched while the long-running jobs occupied both GPUs. For a longer YAML-owned experiment, select `SimToolRealConnectome1952AdaptersMLPSAPG` or `SimToolRealConnectome1952GainsMLPSAPG` as `train_profile`; no architecture CLI argument is required.

The independently sized `128/32/32` sensory/descending/readout MLP option has a
full-geometry two-epoch gate:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_4update_gaussian_lf_entropy1x_sigma3_all_neuron_readout_mlp128x32x32_smoke.yaml
```

The selected train profile inherits the current all-neuron clipped-Gaussian
contract and sets `interface_projections.sensory_hidden_size: 128`,
`descending_hidden_size: 32`, and `readout_hidden_size: 32`. The unchanged
`hidden_size: 256` remains a fallback for any independently sized field that is
omitted. The suite keeps four neural updates, LF reuse, KL `.004`, sigma ceiling
three, auxiliary actor value loss, 12,288 environments and 49,152-sample
minibatches. `scripts/run_connectome_suite.py` prepares the 1,952-neuron graph,
composes the Hydra train profile, writes the resolved contract, and launches
the trainer; the builder constructs the three independently sized interface
MLPs. This gate has not been launched as a training job.

The CUDA-1 long-run counterpart of the live all-neuron no-auxiliary policy is:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_4update_gaussian_lf_entropy1x_sigma3_all_neuron_readout_noaux_mlp128x32x32_100b.yaml
```

Its `gpu_assignments: [1]` selects physical CUDA 1.  It retains the live
policy's 1,952-cell prepared graph, all-neuron readout, four recurrent updates,
seed 42, environment/batch geometry, clipped-Gaussian SAPG/LF objective,
privileged central critic, disabled actor-side auxiliary value loss, task
randomization, checkpoint cadence, and fresh 100B-frame budget.  The only
model change is the selected train profile's sensory/descending/readout MLP
hidden sizes of `128/32/32` rather than the inherited `256/256/256`; output and
experiment names are necessarily distinct to prevent artifact collision.

The superseding synchronized two-GPU small-MLP contract uses 1.25 times the
prior environment population on **each** rank:

```bash
# Two-rank, two-epoch integration and reload gate.
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_noaux_mlp128x32x32_ddp_15360_smoke.yaml

# Fresh two-rank 100B-frame run on physical GPUs 0 and 1.
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_noaux_mlp128x32x32_ddp_15360_mb24576_100b.yaml

# Paper-tolerance and exact checkpoint-training-tolerance videos.
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_noaux_mlp128x32x32_ddp_15360_mb24576_100b_milestones.yaml
```

`training.distributed: true` makes the suite entrypoint launch one synchronized
policy with two `torchrun` ranks over `gpu_assignments: [0, 1]`; it does not
launch two independent policies. Each rank owns 15,360 simulator environments,
exactly 1.25 times the previous 12,288, for 30,720 simultaneous environments
in total. Each rank retains six SAPG blocks (`sapg_block_size: 2560`) and uses
`minibatch_size: 24576`; gradient averaging therefore restores the original
49,152-sample nominal global batch. LF adds a seventh block, so the local
286,720-sample training set produces 11 optimizer steps per mini-epoch and 22
per two-mini-epoch training epoch. One global epoch represents 491,520 frames,
and 203,450 complete epochs produce 99,999,744,000 frames under the 100B cap.
This yields 4,475,900 synchronized actor updates over the full budget. The
earlier 61,440-per-rank launch is preserved as a stopped pilot rather than
mixed into this fresh run.

The production output lives below
`train_dir/connectome/adaptation_100b_gains_update_timing`, the log root already
served by TensorBoard port 6008.

The policy still uses the 1,952-cell all-neuron no-auxiliary clipped-Gaussian
contract and the `128/32/32` sensory/descending/readout interface MLPs. The
milestone watcher polls rank-0 inference checkpoints every 250M global frames.
For each checkpoint it renders all three standard deterministic cases twice:
`paper_task_progress` uses base tolerance 0.02, while
`checkpoint_training_tolerance` requires the exact contemporaneous
`scalars/success_tolerance/frame` value. All checkpoint, action-selection,
coefficient-ID, trajectory, camera, and episode settings remain shared between
the two metric sets.

The suite helper now exposes both physical GPUs to `torchrun`, maps each rank's
Isaac Gym simulator and policy to its local CUDA device, initializes NCCL from
the launcher rendezvous, and counts both ranks when validating the global frame
cap. The two-epoch smoke completed and reload-verified a two-rank checkpoint;
this is an integration check, not evidence of learning.

The active full-size adapters-only MLP replacement uses a single-policy gate and long-run contract:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_mlp_adapters_kl004_smoke.yaml
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_mlp_adapters_kl004_100b.yaml
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_mlp_adapters_kl004_milestones.yaml
```

The smoke and long-run YAMLs use the same 12,288-environment, 49,152-minibatch geometry, seed, KL/LR feedback, task perturbations and exploration settings; only their frame budget/save cadence differ. The evaluator uses mean actions and three high-resolution object/task videos every 250M frames. The historical `ppo_1952_kl004_gains_milestones.yaml` watcher was stopped when its gains policy was replaced; its existing evaluation tree remains preserved.

Run the matched adapters-only MLP policy with the original actor KL target 0.016 using:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_mlp_adapters_kl016_100b.yaml
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_mlp_adapters_kl016_milestones.yaml
```

This contract differs from the KL-0.004 MLP run only in output identity, GPU assignment and actor scheduler threshold. Both use the same MLP profile, seed, optimizer geometry, perturbations, exploration and video cases. `0.016` is validated against the composed original LSTM PPO profile rather than inferred from the later connectome defaults.

The clipped-Gaussian KL-0.016 job above was stopped after its executed-action entropy became badly misaligned with its raw-Gaussian entropy. Its fresh tanh-squashed replacement is selected entirely through YAML:

```bash
# Full-geometry, two-epoch execution and checkpoint-reload gate.
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_mlp_adapters_tanh_kl016_smoke.yaml

# Fresh 100B-cap training job on physical GPU 0.
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_mlp_adapters_tanh_kl016_100b.yaml

# Three deterministic mean-action videos at every 250M-frame milestone.
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_1952_mlp_adapters_tanh_kl016_milestones.yaml
```

`SimToolRealConnectome1952AdaptersMLPTanhSAPG.yaml` inherits the same compact adapters-only MLP actor and changes `params.model.name` to `continuous_a2c_tanh_logstd`. `entropy_samples: 1` selects one reparameterized Monte Carlo sample per state for transformed-entropy optimization; the large training batch supplies the averaging. There is deliberately no `log_std_bounds` setting or latent clamp. The suite YAML retains seed 42, 12,288 environments, horizon 16, 49,152 actor/critic minibatches, two mini-epochs, KL target 0.016, LR ceiling 0.01, SAPG coefficients, perturbations and 250M checkpoint cadence. Change experiment identity, budgets or geometry in the YAML rather than appending Hydra overrides to the command.

The 262-neuron distal-front-leg comparison uses:

```bash
# Rebuild and identity-check the 262/3,194 artifact.
.venv/bin/python scripts/prepare_malecns_connectome.py --config configs/connectome/malecns_262_distal_leg.yaml

# Optional matched full-geometry, two-epoch gate on both GPUs.
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_262_pair_smoke.yaml

# Long runs: clipped KL 0.004 on GPU 0 and tanh KL 0.016 on GPU 1.
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_262_mlp_adapters_kl004_100b.yaml
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_262_mlp_adapters_tanh_kl016_100b.yaml

# Restartable 250M-frame video watchers.
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_262_mlp_adapters_kl004_milestones.yaml
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_262_mlp_adapters_tanh_kl016_milestones.yaml
```

The preparation helper pins source hashes, the five-contact floor, proprioceptor/motor masks, route pruning, expected counts and body-ID hash. The suite helper composes the chosen train profile and owns GPU, policy distribution, KL target, batch geometry, frame budget, TensorBoard directory and checkpoint cadence. The evaluator helper polls inference-only checkpoints and produces the three YAML-listed mean-action videos. The important circuit parameters are `minimum_synapses`, `motor_types`, `expected`, `expected_selection`, transmitter signs and spectral normalization target; the important run parameters are `train_profiles`, `gpu_assignments`, `num_envs`, `sapg_block_size`, minibatch/microbatch sizes, `epochs`, `max_frames`, `kl_threshold` and `inference_checkpoint_interval_frames`. The active fresh directories include `_fresh`; the prior interrupted attempts remain preserved and should not be mixed into the comparison.

The suite entrypoint prepares and verifies the graph, isolates the requested GPU, builds the Hydra child command, records resolved configuration/timing and reload-verifies the terminal checkpoint. The milestone-evaluation helper independently polls the run directory and dispatches the three YAML-owned mean-action cases, so video rendering does not block training. The imported model helper applies `tanh`, evaluates the log-Jacobian-corrected likelihood and transformed entropy, and returns latent actions/means for rollout-buffer PPO and KL bookkeeping; it is not a separate script.

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
