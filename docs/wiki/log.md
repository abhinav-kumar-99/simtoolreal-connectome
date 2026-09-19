# Project Log

Append-only record of durable repository work.

Last updated: 2026-09-19

Related: [Index](index.md), [Overview](overview.md)

## [2026-09-17] analysis | Add parameter-matched LSTM baseline contract

- Added `analyses/lstm-baseline.md` and linked it from the index and connectome actor concept.
- Recorded the 733,796 fly versus 733,790 LSTM coefficient accounting, standard one-update LSTM semantics, YAML entrypoints and single-seed evidence boundary.
- Provenance: current actor/profile code, compact graph artifact contract, and exact composed network parameter shapes.

## [2026-09-16] query | Identify further visual-reservoir speedups

Audited current timing and Triton forward code. Found backward-only `rec`/`z` output allocations and writes remain active under no-grad; documented an inference-only kernel as a concrete candidate, with approximately 4.25 GiB of avoidable nominal writes per nine-update batched control step. Checked that headless viewer synchronization already disables graphics on skipped policy-camera frames. Recorded GPU timing attribution limits, batching/graph-capture candidates and need to benchmark the current full graph. No runtime configuration or implementation changed.

## [2026-09-16] implement and launch | Add multirate lower-resolution fly vision

Added explicit YAML profiles for 64x36 policy cameras rendered once every four 60 Hz control steps. The environment caches luminance between renders while proprioception, goal/previous-action channels and all nine CNS updates remain fresh. The actor observation shrank from 5,283 to 2,403 values while all 3,534 L1/L2 cells retain fixed bilinear sampling. Added matched smoke, 100B, audit and milestone-watcher YAMLs; made the audit derive image dimensions from resolved configuration. All 32 focused tests passed. The two-epoch smoke completed finite checkpoint reload; its warmed epoch reached 4,880 step FPS and 2,934 total FPS. The corrected audit passed distinct/nonblank images, goal-image invariance and measurable response in 115 motor cells.

Stopped only the former 96x54 full-CNS suite/trainer/watcher PIDs 475520/475598/475523 at 1,480,704 frames after the 1M video evaluation completed. Launched the replacement suite/trainer PIDs 484220/484301 and watcher PID 484223 on GPU 1; the GPU-0 learned-adapter Gaussian run remained untouched. Early full-run throughput was about 5,900 step FPS and 3,250 total FPS, roughly 2.2 times the former end-to-end rate. Added `full_cns_tanh_vision64x36_r4_100b` to the existing port-6008 TensorBoard without restarting it.

## [2026-09-16] audit | Decompose full-CNS vision throughput

Traced RL Games timing and the live full-CNS event stream. `step_fps` times only `env_step`, which includes synchronous rendering and GPU access for 384 independent 96x54 cameras; fly inference is outside that timer. The full-CNS run had median 1,913 step FPS and 1,539 step-plus-inference FPS, essentially identical to the preceding visual-path graph's 1,917/1,542. A representative epoch spent 3.18 seconds in environment steps, 3.95 seconds collecting the rollout and only 0.064 seconds updating PPO. Documented that the remaining-neuron expansion is not the throughput cause, the non-camera 12,288-environment run is not a matched control, and a same-size camera-off benchmark is needed to quantify rendering cost precisely. No process or training configuration changed.

## [2026-09-16] query | Explain descending neurons and candidate readout

Updated the visual-reservoir concept and index with the biological brain-to-VNC role (Namiki et al. 2018) and local annotation/config audit: 1,314 descending neurons, 480 type labels, 157 designated input ports, only 53 directly driven by current goal/previous-target mappings. Distinguished all simulated cells from motor-only readout. Documented hybrid readout as an unimplemented experiment, its approximately 30.8 MiB extra rollout feature cost, and the need to control for direct input/readout overlap and visual dependence. No trainer or policy implementation changed.

## [2026-09-16] implement and launch | Include all traced CNS neurons and audit outputs

Added explicit `all_traced_neurons` artifact selection, preserving the earlier visual-path mode and checking that interface cells are retained. The separate SHA-pinned artifact contains exactly 165,122 Traced body IDs and 6,235,682 stored edges; all glia/non-Traced/unannotated endpoints are excluded. Relative to the preceding run it adds 7,955 traced neurons and removes 5,629 other nodes. Verified exact full-population equality and identical visual/proprioceptive/goal-input/motor-output body IDs. The motor audit identified all 135 output cells as T1 VNC motor neurons (68 left, 67 right); documented descending-plus-motor readout as an unimplemented comparison, not a necessary consequence of full-CNS simulation.

The 384-environment two-epoch smoke completed 12,288 frames and finite deployment reload. Camera audit again passed goal-image invariance and distinct environment views; visual sensitivity remained small at about 2.34e-5 maximum motor delta. Stopped only visual-path suite/trainer/watcher 468643/468686/469141, preserving artifacts, and launched the fresh full-CNS tanh suite/watcher as 475520/475523 on GPU 1. The learned-adapter Gaussian trainer on GPU 0 remained untouched. New YAMLs own the full artifact, profile, smoke/full budgets, audit and video watcher; the workflow documents commands and retained parameters.

## [2026-09-16] audit | Correct visual-reservoir neuron coverage denominator

Compared all 162,796 artifact body IDs against the pinned 211,577-row annotation table. Of 165,122 locally Traced entries, 157,167 are included and 7,955 excluded; documented the excluded superclass and type breakdown. Found 4,060 included annotated non-Traced entries (including 64 Glia) and 1,569 included IDs absent from the annotations. The visual extractor has no reconstruction-status filter, so the previous graph-node count must not be presented as confirmed-neuron coverage of the published approximately 166K CNS. Recorded this limitation and the need for an explicitly cleaned artifact. This was a read-only data audit plus documentation; no training process, graph artifact or implementation was changed.

## [2026-09-16] implement and launch | Raw-camera visual CNS reservoir

At the user's correction, implemented actual MaleCNS L1/L2 visual populations and measured optic-to-motor paths rather than an image code injected into the small front-leg circuit. Added SHA-pinned YAML graph extraction (162,796 cells, 6,144,284 stored edges), fixed anatomical-column image sampling, per-environment raw evaluation-angle cameras, hidden goal visualization actors, retained proprioception, explicit desired-goal input and a nine-update tanh profile. Added YAML-owned 96/384-environment smoke gates, full run, camera/sensitivity audit and video watcher. The 93 network/initial-retina tests passed; both smokes passed checkpoint reload. The simulator audit showed byte-identical images under a goal-only change, distinct environment cameras and a small measurable motor response to brightness. Fixed optical alignment and tanh dynamics remain modeling assumptions; the exact evidence and limitations are recorded in the visual-reservoir concept page.

Stopped only LIF suite/trainer/watcher PIDs 450554/450610/450997 and preserved artifacts. Launched full visual tanh suite/trainer PIDs 468643/468686 and watcher 469141 on physical GPU 1. Added `visual_tanh_100b` to the existing port-6008 TensorBoard without restart. The GPU-0 learned-adapter Gaussian trainer 261951 remains active. The warm 384-environment smoke ran at about 1,518 steps/s. Early full-run actor/central value losses, entropy and KL were finite with zero invalid-KL flags. Branch `feature/vision-reservoir` records `feature/malecns-connectome-actor` as its base; unrelated `.vscode/` and `tmp/` remain untouched.

## [2026-09-16] launch | Replace no-auxiliary tanh reservoir with LIF

Stopped only the GPU-1 no-auxiliary fixed-reservoir suite/trainer/watcher PIDs 430585/430658/431147 near frame 301.4M and preserved its artifacts, including the completed 250M three-video evaluation. The dense learned-adapter Gaussian suite/trainer/watcher PIDs 261900/261951/262494 on GPU 0 remained live and were not signaled.

Ran the YAML-owned LIF smoke gate with `use_experimental_cv: true`. It completed two epochs/393,216 frames, saved and reloaded its checkpoint, produced a finite 29-action deployment output and contained only finite floating checkpoint tensors. The resolved contract selected the fixed population encoder, hard LIF dynamics with threshold `.25`, K=4 and the auxiliary actor value path. Actor `c_loss` changed from `1.4254` to `.6505`, privileged `cval_loss` changed from `.9956` to `.2105`, entropy remained near `41.15`, KL remained below `.0008` and both scheduler invalid-KL flags were zero.

Launched the fresh full LIF suite in tmux `connectome-fixed-reservoir-gaussian-lif-100b` as suite/trainer PIDs 450554/450610 on physical GPU 1 and its independent milestone watcher in `connectome-fixed-reservoir-gaussian-lif-videos` as PID 450997. At frame 2,949,120, actor `c_loss` was `.95543`, entropy `41.3040`, KL `.0031157` against the `.004` target and both invalid-KL flags were zero; privileged `cval_loss` was `.06722` at frame 3,145,728. Added the new summaries to the already-running port-6008 TensorBoard by symlink without restarting it. This is launch-health evidence, not convergence evidence.

## [2026-09-16] launch | Replace fixed tanh reservoir with no-auxiliary control

Implemented a YAML-selectable hard-spiking LIF backend for the fixed cached reservoir, using persistent membrane/refractory/spike state, fixed nonnegative rate codes, control-frequency-derived timestep, hard reset, windowed motor spike rates and a fused Triton forward kernel. The dense and fused CUDA paths match, the focused network/configuration suite passes 119 tests, and the implementation is committed but not launched. The design borrows the valid LIF state/reset structure from Quantum-Coded Fruitfly while rejecting its graph-specific drive/weight calibration and final-substep-only readout. A non-training full-graph probe found threshold one left motor features silent despite finite internal spikes; the explicit LIF profile now uses threshold `.25`, which produced nonzero motor rates while remaining finite.

At the user's revised experiment boundary, stopped only the prior GPU-1 fixed-reservoir suite/trainer/watcher PIDs 387124/387180/400161 around 614.4M frames, preserving checkpoints and completed 250M/500M videos. The dense learned-adapter Gaussian PIDs 261900/261951/262494 remained live. A two-epoch tanh fixed-reservoir smoke with `use_experimental_cv: false` completed 393,216 frames and deployment reload; actor `c_loss` was zero, privileged `cval_loss` remained `.2063`, and entropy/KL were finite.

Launched the full fresh no-auxiliary tanh suite in tmux `connectome-fixed-reservoir-gaussian-noaux-100b` as suite/trainer PIDs 430585/430658 on physical GPU 1. Its resolved contract retains the central critic, fixed input map, cached motor readout, K=4, LF/1.0, entropy scale `.005`, capped coefficient-conditioned Gaussian, KL/LR settings, seed, task and 100B budget; only the auxiliary actor value objective is disabled. Watcher PID 431147 awaits 250M milestones, and a symlink makes the run visible to the existing port-6008 TensorBoard without a restart. The spiking full run remains deferred.

## [2026-09-16] query | Audit input normalization across LSTM and connectome policies

Traced raw environment feature construction, RL-Games running-stat behavior, resolved training profiles and saved checkpoint state. The original LSTM, dense/MLP connectome and structured Rotation-6D connectome normalize actor inputs per coordinate; all also normalize privileged critic inputs, values and advantages. The fixed reservoir explicitly disables actor RMS and instead applies fixed scaled `tanh` codes, while retaining critic/value/advantage normalization. A saved fixed-reservoir observation snapshot showed weak velocity coding and comparatively saturated object-scale coding. Documented why online RMS is incompatible with cached fly features and recommended a frozen, provenance-owned per-channel physical calibration plus drive-level controls for learned adapters.

## [2026-09-16] debug | Repair Rotation-6D milestone evaluation dimensions

The fixed-reservoir watcher reached its 250M checkpoint but repeatedly failed before policy construction because `eval_worker_isaacgym.py` hardcoded 140 actor observations. Changed it to derive observation and action dimensions from the instantiated environment, allowing the saved 144-value Rotation-6D policy plus SAPG identifier to build. The live watcher picked up the fix on retry without restarting or touching suite/trainer PIDs 387124/387180. Target 250M then completed at actual frame 250,085,376 with three decodable 800x450, 20-FPS, 200-frame MP4s; the watcher cleared its failure state and resumed polling.

## [2026-09-16] launch | Start fixed-reservoir Gaussian milestone-video watcher

Added a YAML-owned watcher for the live fixed-input, cached-reservoir Gaussian suite. It uses the exact `fixed_reservoir_gaussian` run identity, saved resolved configuration, physical GPU 1, 250M-frame intervals through 100B, and three deterministic mean-action evaluation cases per milestone. Launched it in tmux `connectome-fixed-reservoir-gaussian-videos` as PID 400161; `milestone_status.json` reports running with 400 expected targets, zero completed and zero failures while it awaits the first inference checkpoint. Suite/trainer PIDs 387124/387180 and all other jobs remained untouched.

## [2026-09-16] query | Audit whether four fixed-reservoir updates are sufficient

Traced the exact synchronous recurrence and ran a directed breadth-first search over the prepared 1,952-cell artifact using the fixed encoder's 87 proprioceptor and 98 descending targets. K=4 gives same-decision access from sensory, descending and goal inputs to all 130 reachable motor cells; five motor cells are unreachable from every used input and cannot be recovered by increasing K. Recorded that this establishes structural coverage, not sufficient nonlinear mixing or task performance, and defined a matched K ablation with motor-feature diagnostics. No trainer or live run was changed; unrelated `.vscode/` and `tmp/` remain untouched.

## [2026-09-16] launch | Replace dense-input Beta with structured rotation-6D Beta

Stopped only the prior dense-input restricted-Beta suite, trainer and milestone watcher PIDs 4102855/4102915/4103035, preserving their run artifacts. Launched the fresh structured Beta YAML in tmux session `connectome-structured-rot6d-beta-lf-100b`; its suite PID is 319189 and trainer PID is 319244 on physical GPU 1. No structured milestone watcher was launched because that evaluation contract has not been added.

The saved resolved configuration confirms the 144-value rotation-6D task, grouped linear proprioceptive adapters, no direct fabricated tactile drive, the 128-hidden-unit descending MLP, restricted Beta shapes, K=4, auxiliary actor value loss, LF/1.0 experience reuse, entropy scale .005, KL target .004 and the 100B cap. At frame 3,735,552, entropy 16.47404, aggregate KL .00160879 and both mini-epoch KLs were finite with zero invalid-KL flags. The existing capped-Gaussian suite/trainer/watcher PIDs 261900/261951/262494 remained live and untouched.

## [2026-09-16] ingest | Audit Quantum-Coded Fruitfly sensory routing and dopamine claims

Reviewed upstream commit `e1a2089a1dcc312df446379ca9cd30193575f4db`, the shipped circuit/checkpoint and actual training/evaluation paths. Documented fixed population masks rather than the README's learned projection, a terminal dopamine pulse without a plasticity path, direct candidate features and external constraint memory. A one-thread CPU check on the same seed-42 100-word cohort yielded 99 wins/4.15 guesses with neural simulation, 99/4.13 with zero neural features, and 98/3.87 using the first filtered candidate. Added a pinned source summary with hashes, methodology, limitations, structured adapter proposals and requirements for compartment-specific plasticity. Updated the dopamine analysis and source index. No robot experiment was launched or reconfigured.

## [2026-09-15] query | Correct SAPG embedding gradients in the actor comparison

Corrected the connectome-versus-LSTM auxiliary-value comparison after tracing both `extra_param` builders. Both actors own learned `6 x 32` SAPG embedding tables upstream of their recurrent representations, so the auxiliary value loss updates both embeddings when `use_experimental_cv` is true. Each privileged critic independently owns another `6 x 32` embedding trained by the central-value optimizer. The architecture-specific discrepancy is the value/action readout scope—all recurrent cells versus motor cells for the connectome—not whether SAPG conditioning receives auxiliary gradients.

## [2026-09-15] query | Compare the auxiliary actor value path across connectome and LSTM policies

Traced rollout storage, GAE construction, clipped value loss, both actor forwards, and the active YAML inheritance. For both actors, the privileged central critic supplies stored values, bootstrapping, returns, and advantages. `use_experimental_cv: true` additionally fits the actor model's observation-side value head to those returns with a `2 * c_loss` objective contribution under `critic_coef: 4`; clipping is anchored to the stored privileged-critic prediction. The LSTM loss updates its shared LSTM and post-LSTM MLP, whereas the connectome loss reads all recurrent cells and updates its dense value head plus whichever adapters, conditioning, and circuit parameters are trainable. The connectome action head reads only motor cells, making the all-cell-value versus motor-only-action split architecture-specific.

## [2026-09-15] ingest | Separate the paper critic claim from the auxiliary value loss

Read the complete SimToolReal paper sections describing RL training, the five-seed Figure 8 ablation, the appendix critic state, and Table I. The paper supports a separate privileged critic on clean simulator state: forcing that critic to use the actor's partial observations severely hinders learning. It does not describe or ablate `use_experimental_cv` or the simultaneous actor-side value loss inherited by the released code. Documented that `false` keeps the paper-backed asymmetric critic but differs from exact released-code behavior, while no published result resolves whether the auxiliary objective is beneficial.

## [2026-09-13] query | Align videos with deployed SAPG member

Confirmed from the official SimToolReal source and the local call path that real deployment and evaluation videos both use `deployment.RlPlayer`, which supplies coefficient ID 50 (block 0); video capture additionally enforces the deterministic Gaussian mean. Documented that any future coefficient-ID selection must change deployment and evaluation together and should be chosen on validation objects rather than the final evaluation cohort.

## [2026-09-13] query | Gain-bound provenance and success blocks

Traced `[0.25, 4]` to the initial connectome implementation plan and documented that it is a reciprocal log-symmetric engineering prior, not a biologically calibrated MaleCNS range. Expanded success-metric documentation with the six current 2,048-environment SAPG blocks, coefficient IDs `[50, 40, 30, 20, 10, 0]`, entropy-loss coefficients `[0.0025, 0.0020, 0.0015, 0.0010, 0.0005, 0]`, their shared success definition, and the distinction between block-5 training summaries and coefficient-ID-50 deployment.

## [2026-09-13] query | Biological meaning of gain

Distinguished biological gain—the slope or sensitivity of a neural input-output response—from cell importance. Documented that the actor's outgoing and incoming gains are coarse cell-wide proxies for presynaptic efficacy and postsynaptic recurrent sensitivity, respectively; they are fixed learned RL parameters rather than measurements of MaleCNS physiology or online neuromodulatory state.

## [2026-09-13] query | Interpret learned neuron gains

Clarified the exact gain placement in the recurrent equation. `g_out[j]` attenuates or amplifies source `j` along all recurrent outputs, while `g_in[i]` scales the recurrent sum entering destination `i`; each edge uses their product. Gains are bounded to `[0.25, 4]` and cannot switch a neuron off. Direct drive, leak-carried state, the motor action readout, and the all-neuron actor value head prevent gain magnitude from being a standalone node-importance or causal-ablation measure.

## [2026-09-13] query | TensorBoard success-metric semantics

Traced every live success-related TensorBoard tag through the environment and observer. Documented that `success_ratio` and `mean_success_ratio` are duplicate all-environment means divided by 50, while `successes` and its median/maximum are filtered to the zero-intrinsic-reward exploration block; per-block tags expose all six 2,048-environment populations. All `/frame`, `/iter`, and `/time` variants are same-step aliases. Also corrected the reporting boundary: `success_tolerance` is a base value multiplied by `keypointScale: 1.5` in the actual maximum-keypoint distance test, including during deterministic trajectory evaluation.

## [2026-09-13] launch | 100B gains update-timing comparison

Launched `gains_new_update_timing` on physical GPU 0 and `gains_old_update_timing` on physical GPU 1 in the durable `connectome-100b` tmux session. Both resolved to fused Triton, neuron gains, seed 42, released-checkpoint perturbations, and a 100-billion-frame cap. The children reached live update phases at about 20.2/15.9 GB GPU memory with no startup OOM. Phase times increased from initial 4.1--4.5/1.9--2.1 seconds to about 10.8/5.2 seconds after the first environment-reset boundary, so the live ETA remains unsettled; recompute it after the first milestone.

Launched the restartable mean-action milestone watcher in `connectome-100b-eval`; it will evaluate 400 targets per policy and create three 800x450 videos at each. Launched a separate TensorBoard server in `connectome-100b-tensorboard` on port 6008 with a 10,000-scalar reservoir. No milestone or completion claim has been made yet.

## [2026-09-13] validate | Prepare 100B gains update-timing comparison

Added two seed-42 neuron-gains contracts that use identical Triton actors, released-checkpoint perturbations, and 99,999,940,608 environment frames while isolating the release-matched versus exploratory optimizer timing. The new timing schedules 2,034,504 actor/critic calls; the old timing schedules 4,069,008. Added atomic 250-million-frame inference snapshots and a restartable watcher that generates three 800x450 mean-action videos per policy/milestone using the paper Task Progress evaluation cases.

An end-to-end two-phase smoke created and reloaded two 1.9 MB actor-plus-normalization snapshots, produced two nonempty 800x450/20 FPS deterministic videos, and left normal full recovery checkpoints intact. The focused actor, accumulation, configuration, and milestone tests pass 91 cases. Launch and completion are recorded separately.

## [2026-09-13] query | TensorBoard reward display downsampling

Audited raw reward events and live TensorBoard HTTP responses. Final runs contain 2,543 regularly spaced events per curve, but the server returns only 1,000 under its default scalar sampling limit. Displayed gaps have median 786,432 and maximum 9,437,184 environment frames, versus a raw interval of 393,216. The previous runs have fewer than 1,000 events and raw intervals of 196,608. Corrected the earlier explanation that display spacing was simply doubled; recorded evidence and the optional scalar sampling override in `analyses/adaptation-1b-run.md`. No server or training settings changed.

## [2026-09-13] evaluate | Enforce mean-action video evaluation

Verified that the existing evaluation worker passed `deterministic_actions=True` and that the continuous PPO player maps this path to the predicted Gaussian mean `mu`, not a sampled action. Made `action_selection: mean` explicit in all evaluation YAMLs and generated case/result artifacts, and added parent/worker guards that reject sampled-action video capture or inconsistent flags. Regenerated all six final videos; metrics and frame counts exactly matched the prior deterministic artifacts. All case/result contracts, codecs, resolutions, and sampled visual frames passed reinspection.

## [2026-09-13] evaluate | Final-policy high-resolution videos and run comparison

Evaluated the completed epoch-2,543 adapters-only and neuron-gains checkpoints on matched marker, eraser, and spatula cases. Generated three 800x450, 20 FPS H.264 videos per policy and verified codec, dimensions, nonempty files, frame counts, and first/middle/final rendered states. Both policies averaged 0.741% paper Task Progress at 0.02 m and 0% repository `avg_goal_pct` at 0.01 m. Paper-pass mean raw reward was 51.512 for adapters-only and 207.312 for neuron-gains; these are single stochastic episodes per case and did not translate into higher waypoint completion.

Compared checkpoint Adam counters and event streams with the stopped exploratory jobs. The old checkpoints applied 5,534 adapters-only and 5,870 neuron-gains actor steps; the completed jobs applied 20,335 each. The exploratory schedule used one optimizer call per 24,576 fresh frames, twice the release-matched run's update density of one per 49,152. TensorBoard Step coordinates are comparable environment-frame counts with a one-update-phase left offset, while the `/step`, `/iter`, and `/time` tag variants all receive the same global-step value.

## [2026-09-13] complete | Release-matched billion-step policies

Both rollout-accumulated policies completed 2,543 update phases and 999,948,288 environment frames. Adapters-only trained in 12,258.462 seconds and neuron-gains in 12,490.291 seconds; both final checkpoints passed finite `(1, 29)` deployment reload verification. Final raw `rewards/step` values were 80.1783 and 107.6453, while training `success_ratio/frame` values at the 0.075 m curriculum tolerance were 0.000001628 and 0.000008138.

Final actor Adam counters were 20,335 for both policies, nine below the 20,344 scheduled step calls because mixed-precision `GradScaler` suppressed non-finite updates. The central critic does not use that scaler, but its optimizer counter is not serialized. Post-training 0.02 m paper Task Progress, 0.01 m repository `avg_goal_pct`, and final-policy videos remain pending evaluation.

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

## [2026-09-13] query | Resolve Tyler curriculum enablement

Distinguished updater execution from effective configuration. Legacy Isaac Gym calls and logs the Tyler updater every control step, but current runs, `origin/main`, the transformer-study branch, standard LSTM/SAPG composition, and the released pretrained YAML all leave observation-removal consumers false and controller final targets null. It is therefore functionally disabled in standard paper-era training; no tracked config enables it. The newer Isaac Sim implementation contains no Tyler mechanism. The separate goal-tolerance curriculum remains genuinely enabled.

## [2026-09-13] query | Explain active training curricula

Traced both environment curriculum mechanisms through the resolved compact-run configurations. The active goal-tolerance curriculum requires mean completed-episode successes of at least three before each 0.9 tightening step, with a 3,000-control-step minimum interval; it needs 20 advances and at least 737.28 million frames to move from configured 0.075 to 0.01. Both policies remain at the easiest stage. The separate Tyler scale requires success ratio above 0.6 and five wall-clock minutes per 0.01 increment, but all observation-removal switches and final controller targets are disabled, so it currently has no behavioral effect. Distinguished both from fixed SAPG exploration blocks, randomization, rewards, and optimizer timing.

## [2026-09-13] query | Trace TensorBoard true objective

Traced `true_objective` from the environment formula through reset state and the RL-Games observer. It is a curriculum/PBT ranking diagnostic over current-episode success counts, not reward, success rate, or Task Progress. Documented its piecewise tolerance-progress formula, all-environment mean/max aggregation, difference from completed-episode success tags, and the fact that current PBT-disabled runs remain at initial tolerance so the value is simply `0.01 * current successes`.

## [2026-09-13] experiment | Replace large gains run with compact adapters-only

Stopped the 4,310-cell old-timing trainer and its dedicated milestone watcher without deleting artifacts. Its log reached frame 928,579,584; the latest full recovery checkpoint is frame 904,396,800 and the latest inference milestone is 750,059,520. Added a YAML-owned 1,952-cell adapters-only Triton profile, matched 100B old-timing GPU-1 suite, and deterministic three-video-per-250M milestone evaluator. Twenty-two focused configuration/compact-graph tests passed.

Launched the replacement in tmux `connectome-1952-adapters` with trainer PID 3248466 and its watcher in `connectome-1952-adapters-eval` with PID 3248583. The resolved config confirms frozen gains/dynamics, seed 42, release perturbations, 12,288 environments and 49,152 minibatches. TensorBoard 6008 discovered 160 scalar tags; the first captured reward was 34.874 at frame 1,769,472. The existing compact gains run on GPU 0 continues unchanged.

## [2026-09-13] query | Audit biological attribution of gains actors

Traced the exact frozen/trainable boundary from graph preparation, active gains profiles, actor execution, and stable milestone checkpoints. The policies retain exact prepared edge buffers, support, direction, sign, population masks, and source-derived base magnitudes, but execute diagonally gain-reweighted operators behind fully learned cross-species interfaces and non-biological rate dynamics. At 750.060 million frames the 4,310-cell effective edge array had relative L2 change 0.697 from base with only 10.5% of edges within +/-10%; at 250.085 million frames the 1,952-cell values were 0.604 and 20.0%. Documented why the defensible claim is a MaleCNS-derived recurrent scaffold and why topology attribution still requires matched adapters-only, rewired, and random controls.

## [2026-09-13] query | Digitize paper reward at live training stages

Calibrated the original 1,915 x 1,041 arXiv Figure 8 PNG against its axis ticks and extracted the midpoint of the teal SAPG plus asymmetric-critic mean stroke. At live snapshots of 249.889 million and 681.443 million frames, the plotted paper means are approximately 95 and 295 reward, with approximately +/-20 centerline digitization uncertainty and a more conservative approximately +/-60 full-stroke envelope. The 1,952-neuron actor's trailing-25-million reward of 87.497 is visually tied at this precision; the 4,310-neuron actor's 228.383 is about 77% of the plotted paper mean. Preserved the absence of raw five-seed values and unknown paper-plot smoothing as explicit evidence limits.

## [2026-09-13] query | Audit author-provided training curves

Audited every tracked Git object and the current tips of the author study branches. The repository contains three exact seed-0 reward CSVs; the newest evaluation-member check-in export has 60,277 rows, seven series, and an RL-Games LSTM curve through 1.255 billion frames. The plotting source establishes restart merging, duplicate-step handling, a trailing 25-million-frame mean, and block-5 comparison semantics. No tracked TensorBoard events, W&B histories, ten-run CSV, or raw five-seed Figure 8 values were found.

Compared the two live old-timing gains actors with the study LSTM at matched frame coordinates using the same smoothing. Their smoothed dense rewards were higher at the captured frames, but this is not a controlled actor comparison because the study uses seed 0 and an easier 654-cuboid-only distribution. Recorded separately that deterministic deployment-member Task Progress remains flat at 0.7407% for the available 4,310-neuron milestones.

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

## [2026-09-13] query | Bilateral front-leg pathway coverage

Added `scripts/audit_front_leg_coverage.py`, its YAML contract and three focused tests. Compared the pinned production graph with hash-pinned public MaleCNS v1.0 annotations and streamed individual synaptic partners using explicit VNC compartments. Recovered native lineage/nerve metadata for 4,309 IDs and preserved one unmatched original ID. Confirmed bilateral motor representation and 129 non-motor sources directly connected to both motor soma-side groups.

Persisted the complete analysis in `analyses/front-leg-pathway-coverage.md`, linked from the index and source summary. Explicit expansions have 4,353, 4,589 and 4,778 neurons, with 120,096, 125,611 and 127,432 reference edges at five synapses. Reference-current edges number 119,967; the production graph remains at 118,920. Recorded the snapshot/ROI mismatch, incomplete functional subtype identification, disconnected additions and external-input denominators. Generated exact body-ID lists, annotated membership tables, edge comparisons and final `status: complete` summary remain ignored under `profiles/connectome/front_leg_coverage/`. Full audit, candidate-membership checks, compilation and three tests passed. No actor graph/config or training job was changed.

## [2026-09-13] query | Meaning of omitted cells and connections

Extended the pathway audit's interpretation using its completed edge and cell tables. Partitioned new connections by endpoint membership, identified new 09A-to-proprioceptor connections and actual omitted motor labels, and clarified that the omitted DNg12_a variant has `nt` rather than `fl` subclass. Linked the 23B spatial-touch study while preserving the cross-dataset/subtype evidence boundary. These are anatomical candidates, not proven missing control functions; no code or actor changes were made.

## [2026-09-13] query | Complete-circuit expansion scope

Before implementing an unrestricted completion request, measured incoming closure of the 4,778-cell seed against all traced public MaleCNS neurons and the audit's explicit VNC compartments, using >=1-contact edges and confidence >=0.5. Iterations reached 22,897, 23,960, 23,970 and 23,971 neurons, then stabilized; the induced graph contains 3,864,279 edges. Documented the resulting VNC-wide scope, changed threshold, unknown/nonstandard transmitter labels and need to settle the anatomical boundary before changing the actor. No training, graph replacement or expanded profile was deployed.

## [2026-09-13] query | Neuron participation versus circuit completeness

Checked actual production NPZ support and actor input/output masks: 4,281 neurons are reachable from robot input populations, 4,162 can reach motor readout, and 4,133 lie on an input-to-motor route; ten production neurons are isolated. Documented zero-state behavior for the 29 input-unreachable nodes under frozen zero biases, and the distinction between action connectivity and the all-state actor-value head. Missing external inputs do not imply inactivity. Kept production counts separate from reference-candidate isolates and structural reachability separate from functional validation.

## [2026-09-13] query | Compact circuit size for fine motion

Measured the union of explicitly selected functional candidate groups without preserving all original cells: 993 neurons without tactile afferents, 1,285 with them. The latter induced graph has 7,377 >=5-contact edges and 114 isolated cells, demonstrating that group selection alone omits connecting partners. Documented approximately 2,000 neurons as an untested engineering target, not an established minimum; distinguished the published 803-cell unilateral rhythm model from the desired bilateral dexterous controller. The user prefers the 4,778 candidate as the broader reference. No profile deployment or compact training was authorized or launched in this query.

## [2026-09-13] query | Origin and meaning of five-synapse cutoff

Verified the cutoff in Pugliese's extraction methods and the production NPZ: minimum raw magnitude 5; 17,791 edges exactly at 5; none below 5. Documented anatomical cell-pair contact count versus firing/weight thresholds, explicit reuse in the audit YAML, and the paper's absence of this cutoff for FANC/BANC. Highlighted that same-node threshold changes must be distinguished from unrestricted node expansion. No runtime setting was changed.

## [2026-09-13] query | Evidence-led compact additions without padding

Updated the pathway analysis and index to supersede a fixed approximately 2,000-cell target at the user's direction. Recomputed a 1,596-cell shortest-path footprint and screened excluded cells against explicit directed sensory-motor and GABA feedback motifs. Found 276 proprioceptive-premotor and 119 tactile-premotor candidates, 356 unique (338 VNC intrinsic), plus a broader overlapping 672-cell descending-premotor pool. Recorded actual IN08A006 and IN19A076 connectivity, exact input denominators, source hashes and reproducible traversal semantics. Anatomical candidates are not demonstrated robot-control necessities; neither shortest-path selection nor motif unions constitute a chosen compact controller. This query changed documentation only and launched no training.

## [2026-09-13] experiment | Launch approved 1,952-cell old-timing run

Implemented hash-checked path/motif preparation, explicit 384/157/135 population ports, source-sign uncertainty accounting, normalized 33,720-edge artifact, gains profile, full-size smoke/full-run/evaluation YAMLs and focused tests. No padding, extra feedback candidates or pruning of 44 isolates. Changed shared connectome profile defaults to old timing, preserving historical explicit overrides. All 26 focused tests passed; the 393,216-frame full-size smoke completed, with 16 applied actor Adam steps and finite saved parameters/reloaded actions. Implementation checkpoint: `c44456e0`.

Launched a fresh 100B-capped seed-42 job in `connectome-1952` on GPU 0, confirmed resolved old timing and live scalar data in TensorBoard 6008. Preserved original old-timing PID 3172351 and all canceled new-timing artifacts. Replaced its dual-policy video watcher with old-only configuration and started a separate compact milestone watcher. Added `analyses/compact-1952-training.md` with source identity, evidence, entrypoint/helper usage, parameter ownership, PIDs and limitations; linked it from the index, pathway analysis, timing history and workflow. Long training remains ongoing.

## [2026-09-13] query | Dopamine-inspired learning and training cost

Checked primary fly dopamine/plasticity and e-prop papers, current actor recurrence and the live compact run's timing definitions/logs. Documented absent biochemical/online plasticity machinery, reward-versus-credit-assignment semantics, and a proposed frozen-feature/readout eligibility-trace experiment. A 100-phase snapshot averaged 2.40101 seconds rollout plus 0.55143 seconds updates, limiting update-only savings at unchanged exposure to about 18.7% of timed epoch cost. Added the linked online-learning analysis with explicit sample-efficiency, trace-memory, cross-species and implementation limitations. Documentation only; no training process, runtime config or policy was changed.

## [2026-09-13] query | Eligibility learning from scratch

Extended the online-learning analysis with random adapter/readout initialization, identity gains, learnable input interfaces, per-environment local and delayed-reward traces, Gaussian action credit, approximate neuron-specific feedback, and bounded shared-gain aggregation. Distinguished a proposed rate-network actor-critic from exact e-prop/BPTT and the previous readout-only warm start. Recorded reward/exploration, GPU update cadence, reset masks and future YAML requirements without presenting an untested implementation as ready. Rechecked actual actor step/population configuration and the primary e-prop paper. Documentation only; current runs unchanged.

## [2026-09-13] query | Eligibility trainer infrastructure compatibility

Inspected the actual Isaac Gym entrypoint, algorithm/player factories, GPU observer, suite overrides/budget checks, checkpoint verification, inference export, deployment conditioning and video/milestone loaders. Documented workflow-compatible integration versus a non-drop-in PPO optimizer replacement, including algorithm-aware validation, recovery trace state, frozen evaluation and comparable TensorBoard axes. Preserved the separate trainer/YAML proposal and listed end-to-end acceptance gates. No trainer implementation, executable eligibility config, runtime mutation or new training was performed.

## [2026-09-13] query | Existing RL traces and connectome-specific eligibility hypotheses

Verified the current GAE recurrence and live resolved gamma 0.99, lambda 0.95 and 16-step unroll. Clarified that replacing BPTT with local parameter traces is distinct from the temporal credit already present in PPO. Compared primary GAE/PPO/e-prop evidence with this actor's 3,904 gain parameters, 56,060 adapter weights and restricted motor readout. Documented numerical/credit/hardware tradeoffs, shared benefits to PPO, and a proposed algorithm-by-topology comparison to test a genuinely connectome-specific advantage. Documentation only; no runtime changes or launches.

## [2026-09-13] implement | Local eligibility trainer and infrastructure

Added separate local-sensitivity/action-credit engine and online TD actor-critic, registered trainer/player, compact eligibility Hydra profile, YAML smoke/continuation/prepared-pilot/evaluation contracts, and algorithm-aware suite completion checks. Learned adapters/readout/bounded gains preserve the base graph; fixed random feedback and omitted cross-neuron temporal derivatives are explicit approximations. Per-environment terminal/time-limit credit handling, fresh-episode resume, fixed exploration, small detached-feature critic and atomic inference snapshots are documented in the new workflow page. Updated index, overview, experiment workflow and historical proposal status.

Validated the real simulator through 3,072 frames/32 updates then continuation to 4,608/48; deployment reloads produced finite 29-actions, both runs appeared on TensorBoard 6008, and two 30-frame 800x450 mean-action videos completed. Both short task-progress evaluations were zero. Checkpoint deltas confirm intended parameter groups changed and graph/dynamics/embedding/sigma stayed fixed. No long eligibility job or changes to the existing PPO processes. Integration success is not evidence of faster or better learning.

Final focused regression run passed 98 actor, eligibility, configuration, compact-circuit and milestone tests. CPU checkpoint tests also verify identical inference means and feedback after restore. Existing compact gains/adapters-only PPO PIDs 3211659 and 3248466 remained live at handoff.

## [2026-09-14] experiment | Configure 100B adapter plus gains eligibility run

Added a YAML-owned 1,952-cell adapter+gains eligibility contract pinned to physical CUDA 0 and a persistent mean-action evaluation contract for three task videos on every 250M-frame milestone through 100B. The validated conservative geometry uses 384 environments and six compatibility blocks; eligibility updates remain per vector step. Increased the logging/checkpoint chunk to 4,096 transitions per environment, producing 63,579 summaries and a final vectorized boundary of 100,001,120,256 frames. Outputs remain nested beneath TensorBoard 6008's existing log root. The prior failed gains job on GPU 0 was not restarted, and the running adapters-only PPO job on GPU 1 was not changed.

Extended the same eligibility implementation with an adapters-only mode that omits incoming/outgoing gain and edge traces, while still learning both input adapters and the motor readout. Added a matched CUDA-1 100B suite and 250M milestone-video watcher contract. The earlier adapters-only PPO process was no longer live when this matched eligibility run was requested; its artifacts were preserved.

## [2026-09-14] experiment | Match eligibility TensorBoard frame cadence

At the user's direction, stopped only the newly launched CUDA-0/CUDA-1 eligibility trainers and their milestone watchers. Preserved both partial training trees and both evaluation-status trees with the suffix `_log4096_stopped_20260914_072647`; no artifacts were deleted. Changed both authoritative long-run YAMLs from 4,096 to 512 steps per logging chunk, so 384 environments produce one TensorBoard point per 196,608 frames, exactly matching the old-timing PPO frame cadence. Recomputed the vector-aligned run boundary as 508,627 chunks and 100,000,137,216 frames, and changed recovery save frequency from 10 to 80 chunks to preserve its approximately 15.7M-frame spacing. Eligibility actor updates remain online every vector step; the 250M inference/video schedule is unchanged.

Relaunched adapter+gains on CUDA 0 and adapters-only on CUDA 1 at 07:28 EDT. Coordinators/trainers are PIDs 3431537/3431649 and 3431540/3431650; milestone watchers are 3431831 and 3431834. TensorBoard 6008 discovered both new event paths. Early points were finite and exactly 196,608 frames apart; watcher manifests report no failures and expect 400 targets with three videos each. Both GPU jobs and watchers remained live at handoff.

## [2026-09-14] query | Eligibility traces versus REINFORCE

Traced the implemented actor and critic updates. Documented that the critic uses one-step bootstrapped TD targets and an ordinary Huber loss, while the actor has no autograd loss and applies reward-prediction errors to fading approximate action-credit traces. Distinguished this from Monte Carlo REINFORCE's sampled returns and exact policy score, including the optional-baseline, variance/bias, online timing and recurrent-gradient differences. Corrected the superseded statement that no long eligibility run had launched. No runtime process or configuration changed.

## [2026-09-14] query | Audit live compact-training status

Checked tmux, process ownership, both RTX 4090s, training logs, recovery and inference checkpoints, TensorBoard scalars, and milestone manifests. The 1,952-cell gains trainer failed at 2,275,344,384 logged frames because its learned coefficient-conditioned action standard deviation became negative; its last recovery checkpoint is frame 2,241,331,200 with 91,164 applied actor Adam steps. Nine milestones and 27 mean-action videos are complete.

The adapters-only process remains alive on GPU 1 and passed 3.32 billion logged frames, but actor loss and KL have been `NaN` since about 3.19 billion. Its frame-3,303,014,400 recovery checkpoint has 129,745 applied actor steps versus 134,400 scheduled calls, so simulator/critic activity is continuing without valid actor learning. Thirteen milestones and 39 mean-action videos are complete. GPU 0 is idle and no training/evaluation process was changed during the audit.

## [2026-09-14] experiment | Stop invalid adapters-only run

At the user's direction, sent interrupts to the exact `connectome-1952-adapters` trainer/coordinator and `connectome-1952-adapters-eval` watcher, then verified all three PIDs exited. The last log coordinate is epoch 17,066/frame 3,355,115,520. The last recovery checkpoint remains intact at epoch 17,000/frame 3,342,336,000 with 129,745 applied actor steps; 6,255 of 136,000 scheduled calls had been suppressed, and the applied count had not advanced since epoch 16,800. Preserved all TensorBoard events, checkpoints, milestone manifests, and 39 mean-action videos. A separate eligibility trainer that appeared on GPU 0 was not changed.

## [2026-09-14] query | Correct compact PPO failure diagnosis

Superseded the earlier claim that the gains policy learned a negative action standard deviation. The checkpoint field named `sigma` is log standard deviation and is exponentiated by the selected `continuous_a2c_logstd` model; negative raw entries are valid. Both failed policies instead exhibited runaway positive log standard deviation in the high-entropy SAPG rows, amplified by the adaptive actor learning rate reaching up to `1e-2` and by old timing's doubled optimizer-call density per environment frame. Gains block 0 reached mean/max log standard deviation 84.800/86.463 at 2.25B frames; adapters reached 86.393/87.497 at 3.25B. Float32 exponentiation, sampling and KL-squared arithmetic consequently overflowed to non-finite values.

Verified that all saved actor-core and critic tensors remained finite and that adapters-only had a frozen recurrent matrix, while the exception occurred in Gaussian sampling after recurrence. The shared failure is therefore an action-distribution/optimization instability, not evidence of a Triton recurrent-kernel or connectome-state failure. GradScaler skipped later actor steps but could not restore the already extreme log standard deviation. No process or experiment configuration was changed during this diagnosis.

## [2026-09-14] query | Audit early eligibility learning curves

Updated the eligibility runbook and index from active TensorBoard events at approximately 20M/23M frames and both 15,728,640-frame recovery checkpoints. Metrics are finite and traces stable, but matched-window rewards show no convincing task improvement and successes remain rare. CPU float32 reconstruction finds only two of 3,904 effective gains differ from initialization, by approximately 1.2e-7; logged nonzero gain update norms measure proposed raw-parameter changes, not effective gain movement. Documented startup episode-length confounding, reward/critic interpretation, and the unproven small-update-scale hypothesis. No jobs, training settings, or generated artifacts were modified.

## [2026-09-14] query | Distinguish policy improvement from TD credit estimation

Updated the eligibility runbook and index using actor/critic source and primary PPO/GAE papers. Explained that TD and policy gradients coexist, and separated conventional recurrent differentiation from approximate local eligibility. Recorded the conventional actor-critic baseline recommendation as conceptual, not an empirical winner, plus the limited frozen-core/readout-only control. The interrupted all-actor-1e-3 restart was not executed. Preserved unrelated runtime/config/test changes already in the worktree.

## [2026-09-14] query | Consider alternative input-adapter objectives and optimizers

Updated the eligibility runbook and index with proposed sensory-code supervision, teacher-policy imitation, short-window recurrent gradients, and parameter-space search. Linked primary policy-distillation and evolution-strategy sources, distinguishing published methods from untested connectome-specific adaptations. Explicitly retained teacher cost, simulation cost, propagation horizon, and robot-to-fly correspondence limitations. No runtime or training configuration changes.

## [2026-09-14] implementation | Stabilize PPO KL and configure conservative LR feedback

Replaced overflow-prone Gaussian KL with scale-ratio/normalized-mean float64 arithmetic and cancellation-resistant `expm1`; invalid distributions report infinite KL. Negative/non-finite scheduler input now reduces LR. Exposed adaptive LR bounds through YAML and set connectome threshold 0.004, ceiling 1e-3, floor 1e-6, preserving initial 1e-4. Added exact per-mini-epoch scheduler telemetry and paired fresh-start training/evaluation YAMLs for compact adapters-only and gains with unchanged old update timing, 100B cap and 250M-frame mean-action video cadence.

117 focused tests passed. Both full-size updated smoke jobs completed 393,216 frames with 16 applied actor Adam steps, all finite model tensors and finite deployment actions, without microbatch fallback. New scheduler scalars matched each expected decision. Child training took 27.60 seconds for gains and 25.81 seconds for adapters while sharing GPUs with untouched eligibility jobs. Historical failed checkpoints/logs and both smoke directories remain preserved. These checks establish execution and arithmetic correctness, not long-run stability.

## [2026-09-14] experiment | Launch stable-KL compact PPO pair

Launched suite `ppo_1952_kl004_100b` in tmux `connectome-ppo-kl004` (coordinator 3449384), with gains PID 3449479 on GPU 0 and adapters-only PID 3449478 on GPU 1. Started mean-action milestone watcher PID 3449390 in `connectome-ppo-kl004-eval`. Verified resolved threshold 0.004/max LR 0.001, 168 initial scalar tags, finite actor losses/KL, and actual scheduler saturation at 1e-3 without exceeding the cap. Existing eligibility jobs remained active. Full training is ongoing; no completion or long-run stability claim is made.

## [2026-09-14] experiment | Restore PPO maximum LR and replace compact pair

At the user's direction, restored the connectome maximum adaptive actor LR from `1e-3` to `1e-2` while retaining stable KL arithmetic, target `0.004`, invalid-KL reductions, and floor `1e-6`. Changed the YAML-owned suite/output identity to `ppo_1952_kl004_lr01_100b` and updated its mean-action milestone watcher paths; all batch geometry, update timing, seed, exploration settings, 100B cap and 250M-frame video cadence remain unchanged. The stopped `1e-3` histories are preserved through approximately 39.3M gains and 44.0M adapters frames, before any evaluation milestone. Existing eligibility jobs were not stopped.

Validation after the ceiling change passed 52 focused tests and the paired suite/evaluation linkage check. A new full-batch two-phase smoke completed 393,216 frames per policy without OOM; both checkpoints had 16 applied actor Adam steps, finite model tensors and finite deployment actions. Gains/adapters child training took 28.13/25.60 seconds while sharing the machine with the untouched eligibility jobs.

Launched the fresh replacement in tmux `connectome-ppo-kl004-lr01`: coordinator 3454679, gains trainer 3454787 on GPU 0, adapters trainer 3454786 on GPU 1, and mean-action watcher 3454681. Both resolved configs contain KL target 0.004/max LR 0.01. Initial TensorBoard data was finite through frame 1,769,472, with no invalid KL and actor LR approximately 0.00865; the watcher reported no failures. TensorBoard 6008 discovered both new event paths. Training remains active.

## [2026-09-14] query | Audit rising PPO entropy

Read both live event streams and the exact SAPG loss path. At approximately 72.7M/79.4M frames, aggregate Gaussian entropy had risen from 41.149 to 42.858/43.304 nats for gains/adapters. Block 0 rose to 45.562/46.927, implying geometric-mean standard deviations 1.164/1.220; lagging frame-39.3M checkpoints remained finite with maximum individual block-0 log standard deviations 0.226/0.268. The `[0.0025, 0.0020, 0.0015, 0.0010, 0.0005, 0]` SAPG coefficients directly reward entropy in five blocks, so an early ordered increase is expected. However, there is no entropy target, decay or sigma bound; KL feedback constrains only local change. Recorded this as a current warning rather than a numerical failure or proof of harm. No process or training setting was changed.

## [2026-09-14] query | Explain the compact neural circuit in plain language

Added an accessible overview to the actor concept page and updated the index. Recounted population masks from the prepared CSV: 92 proprioceptors, 292 tactile cells, 157 directly driven descending cells, 135 motor readout cells and 1,276 other cells. Verified the graph manifest, seed/path/motif selection, 33,720 pairwise edges, five-contact filter, 44 isolates and synchronous actor update. Distinguished anatomical family labels from demonstrated functions, body sensing from actual robot tactile availability, and measured wiring from engineered dynamics and learned cross-species interfaces. Clarified the PPO versus eligibility critic distinction. No training changes.

## [2026-09-14] implementation | Add optional MLP interface projections

Added a YAML-selectable shared interface architecture for the sensory, goal/conditioning and motor-action projections. The backward-compatible default remains linear; the new profiles use one 256-unit ELU hidden layer while leaving the recurrent connectome and actor value head unchanged. Added general and compact adapters-only/gains MLP profiles plus a two-GPU smoke-suite contract. The compact MLP shapes are `128 -> 256 -> 384`, `44 -> 256 -> 157` and `135 -> 256 -> 29`, adding 164,793 trainable actor scalars relative to either linear counterpart.

PPO supports the MLPs through autograd with the Triton/cuSPARSE recurrent backend unchanged. The local-eligibility trainer explicitly rejects MLP interfaces because its manual traces currently assume one linear matrix. CPU forward/backward, configuration composition and CUDA dense-versus-cuSPARSE/Triton parity tests cover both architectures. Existing training processes were not restarted or altered.

## [2026-09-14] experiment | Replace linear adapters-only PPO with MLP interfaces

Interrupted only stable-KL linear adapters child PID 3454786 at logged epoch 2,926/frame 575,078,400. Preserved its finite epoch-2,902/frame-570,556,416 checkpoint with 23,206 applied actor Adam steps, both inference milestones and six high-resolution mean-action videos. The gains PPO and both eligibility trainers remained live. Replaced the combined milestone watcher with a gains-only continuation that retains the existing evaluation state.

Added YAML-owned full-batch smoke, 100B training, MLP milestone-evaluation and gains-continuation contracts. Static profile/contract and CUDA actor tests passed 92 tests. The full-size MLP smoke completed 393,216 frames without OOM or microbatch fallback, applying all 16 actor steps in 25.89 seconds of child training; its checkpoint tensors and deployment action were finite. Launched the fresh adapters-only MLP coordinator/trainer as PIDs 3502131/3502258 on GPU 1 and its mean-action watcher as PID 3502135. Initial telemetry was finite through frame 3,735,552. This is a launch/compatibility result, not evidence of learned task performance.

## [2026-09-14] query | Bound attribution for frozen circuits with MLP adapters

Verified compact MLP profile inheritance and source-level observation/recurrence/motor-readout routing. Documented 224,797 learned interface parameters and the absence of a direct observation-to-action bypass, without equating either fact with a percentage of biological contribution. Explained that nonlinear interfaces can learn control-relevant representations around a fixed recurrent transformation, and that inference lesions test dependence rather than biological specificity. Added matched retrained frozen-graph and direct-policy comparison requirements to the actor concept page and index. No training settings or processes changed.

## [2026-09-14] experiment | Replace gains PPO with original-KL MLP adapters

Interrupted only gains PPO child PID 3454787 and its watcher PID 3502127. Preserved its finite epoch-3,600/frame-707,788,800 checkpoint with 28,788 applied actor Adam steps, final frame-711,131,136 TensorBoard telemetry, both inference milestones and six high-resolution mean-action videos. The KL-0.004 MLP trainer/watcher and both eligibility jobs remained live.

Added a YAML-owned GPU-0 100B adapters-only MLP contract with actor KL target 0.016 and its separate mean-action milestone evaluator. A composition/contract test validates 0.016 against the original LSTM PPO profile and verifies that all non-identity/non-GPU/non-threshold training fields match the KL-0.004 MLP run; 19 configuration tests passed. Launched coordinator/trainer PIDs 3515959/3516083 and watcher PID 3515963. Resolved configuration and initial telemetry through frame 6,684,672 were finite with zero invalid-KL flags. This launch does not yet establish relative stability or learning performance.

## [2026-09-14] query | Audit original-KL video watcher

Confirmed watcher PID 3515963 and trainer PID 3516083 remain alive, but the watcher is not discovering the available 250,085,376-frame checkpoint and has produced zero videos. The evaluation policy name is `adapters_mlp_kl016`, while the suite case/run name is `adapters_mlp`; checkpoint discovery constructs its directory from the former and silently sees no files. The status manifest consequently reports zero completed targets and no failures. Recorded this as an open configuration defect without changing or restarting the watcher.

## [2026-09-14] fix | Repair original-KL video checkpoint discovery

Changed the KL-0.016 evaluation policy name to `adapters_mlp`, matching the suite case/run name used by checkpoint discovery, and extended the configuration test to enforce that equality. Nineteen configuration tests passed. Restarted only the milestone watcher; trainer PID 3516083 was uninterrupted.

The replacement watcher immediately discovered and completed the existing target-250M/actual-frame-250,085,376 checkpoint with no failures. All three mean-action videos decode at 800x450, 20 FPS. Marker, eraser and spatula paper Task Progress was 0.0%, 0.0% and 2.2%, respectively. The watcher remains live for future milestones.

## [2026-09-14] query | Audit current MLP entropy and stop eligibility jobs

Verified four live trainers: two fixed-sigma local-eligibility jobs and two matched MLP PPO/SAPG jobs at KL targets 0.004 and 0.016. At the user's direction, interrupted only the eligibility trainers, their suite coordinators and both milestone watchers, then verified all six PIDs exited and the PPO processes remained. Preserved eligibility checkpoints at frames 188,743,680 and 204,472,320; neither run reached its first 250M milestone.

Audited the PPO resolved YAML, Gaussian model, SAPG loss, adaptive scheduler, action clipping, live TensorBoard streams and recovery checkpoints. The KL-0.004 block-0 geometric-mean standard deviation was about 1.51 near 1.30B frames. The KL-0.016 block-0 value was about 87,000 near 1.10B frames, despite finite model tensors. Fixed positive per-block entropy coefficients directly increase an unconstrained trainable `6 x 29` log-standard-deviation table; per-mini-epoch KL LR reductions can be canceled by the following mini-epoch. The sigma gradient bypasses recurrence, and the normal LSTM SAPG profile uses the same mechanism; ordinary LSTM PPO does not have the fixed SAPG entropy push. Recorded bounded/fixed sigma, bounded-action likelihood, coefficient scheduling, non-canceling KL control and clipping telemetry as proposed interventions. No PPO process or training configuration was changed.

## [2026-09-14] query | Audit action-clipping correctness

Traced rollout sampling, buffer storage, PPO likelihood evaluation, trainer preprocessing and the task-base safety clamp. The raw Gaussian sample and its raw log probability are stored before the action is clipped, so the likelihood-ratio estimator is internally consistent for a latent Gaussian followed by a deterministic clip. The actuator bound itself is effective. However, entropy and KL describe the unbounded latent Gaussian rather than the commands seen by physics; at high variance the entropy bonus can grow without useful executed-action diversity. Documented that moving the clamp before an ordinary Gaussian log probability would not fix this many-to-one distribution, and that a bounded distribution requires consistent transformed sampling and likelihood. No runtime process or configuration was changed.

## [2026-09-14] implementation | Replace clipped KL-0.016 policy with unclamped tanh distribution

Stopped only the pathological clipped-Gaussian KL-0.016 trainer/coordinator/watcher and preserved all artifacts; its event stream ended at frame 1,225,850,880 with aggregate/block-0 raw entropy 109.307/438.183 nats. Added YAML-selectable `continuous_a2c_tanh_logstd`: executed and deterministic actions are tanh bounded, PPO log probability includes the stable transform Jacobian, rollout buffers retain latent actions/means, raw-Gaussian KL remains exact under the shared bijection, and entropy is estimated for the transformed action distribution. The SAPG blockwise entropy incentive remains active. Per user direction, there is no `log_std` clamp or `log_std_bounds` configuration.

All 135 focused PPO/connectome/configuration tests passed. The full-size 393,216-frame smoke completed in 21.79 seconds with finite checkpoint reload/actions, entropy 19.42689 to 19.42275 and no invalid KL. Launched the fresh 100B-cap suite on GPU 0 as coordinator/trainer PIDs 3627918/3627974 and watcher PID 3628067. Through frame 7,274,496, transformed entropy decreased to 19.16953 with finite KL and zero invalid flags. The KL-0.004 MLP job on GPU 1 was untouched. This is an integration and early-launch result, not evidence of eventual stability or task learning.

## [2026-09-14] query | Explain tanh interaction with adaptive KL scaling

Verified that rollout and training expose unsquashed Gaussian means/scales to the existing stable KL calculation. Because tanh is the same bijection for old and new policies, its density Jacobians cancel and latent Gaussian KL equals transformed-policy KL in real arithmetic. The target-0.016 scheduler rules remain divide by 1.5 above 0.032, multiply by 1.5 below 0.008, unchanged between them, and reduce on invalid input; SAPG entropy coefficients are not scheduled. At frame 23,003,136 both live mini-epoch values were in the unchanged band and LR remained 0.004444. Recorded that tanh changes the LR trajectory only through policy gradients and does not fix per-mini-epoch cancellation or cumulative drift. No runtime process or configuration changed.

## [2026-09-14] query | Diagnose falling squashed-policy entropy

Audited live block entropy, bounds loss, KL, LR and reward through frame 73,531,392, then inspected stored observations/recurrent states and the six log-standard-deviation rows at the epoch-338/frame-66,453,504 checkpoint. Entropy fell overall from 19.427 to 15.301 nats but oscillated, with sharp troughs aligned to bounds-loss and KL spikes. All blocks followed the trend, including zero-entropy-coefficient block 5. The learned row scales remained moderate near 0.80 to 0.84, while pre-tanh means reached RMS 0.55 to 0.59. State-level quadrature counterfactuals gave 13.72 nats for actual means/scales and 19.59 after zeroing only means, identifying PPO-driven mean motion as the dominant cause. Current transformed-entropy gradients oppose those mean excursions and favor increasing log standard deviation, so SAPG remains active but is outweighed by PPO/task gradients. Recorded that modest entropy reduction can reflect decisive bounded actions, while the observed entropy/bounds/KL oscillations are not yet evidence of useful learning. No runtime process or configuration changed.

## [2026-09-14] query | Audit goal-tolerance tightening gate

Traced `tolerance_curriculum` and the live tanh-squashed run. The 3,000-control-step/36,864,000-frame time gate was passed long ago, but tightening additionally requires all-environment mean `prev_episode_successes >= 3`. At frame 361,955,328 the logged mean was 0.00407 and tolerance remained 0.075; the frame-353,697,792 checkpoint had mean 0.00423, per-environment maximum 2 and `last_curriculum_update: 0`. The failed gate leaves that marker unchanged, so the first qualifying control step will immediately change 0.075 to 0.0675; subsequent advances require another 3,000 steps. Distinguished this gate from the unrelated `curriculumSuccessRatio` used by the functionally disabled Tyler curriculum. No runtime process or configuration changed.

## [2026-09-14] query | Quantify bounds-loss and entropy correlation

Traced raw-mean bounds loss and transformed-entropy formulas, then aligned 3,135 live TensorBoard points through frame 616,169,472. Bounds loss and entropy had Pearson/Spearman correlations -0.928/-0.936; first differences correlated -0.910. The shared cause is pre-tanh mean magnitude: squared bounds loss activates beyond absolute mean 1.1, while tanh entropy falls smoothly as means approach saturation. The bounds penalty itself pushes means inward and therefore opposes, rather than causes, falling entropy. Entropy averaged 14.92 in the lowest positive bounds-loss quartile and 7.83 in the highest; latest values were 6.107 and 1.321. Recorded that the relationship is expected but its magnitude confirms extensive action-mean saturation, not healthy learning by itself. No runtime process or configuration changed.

## [2026-09-14] query | Screen radically smaller dexterity circuits

Compared major fly sensorimotor systems using primary literature and inventoried local whole-CNS motor annotations. Within the existing 4,778-cell reference, recomputed left front-leg proprioceptive-to-motor intermediary unions and route pruning, producing 86/616, 262/3,194 and 408/6,939 neuron/edge candidates, all subsets of the current 1,952 graph. Added a dedicated analysis with reproducible mask/type definitions, body-ID hashes, feedback-component counts, fixed-ID weak-edge restoration and left/right sampling caveats. Recommended 86 only as an aggressive useful-module hypothesis and 262 as a first full-task candidate, not measured minima. Documented missing goal ports in the 86 set and other interface/normalization requirements. No runtime graph, training config, process or generated artifact changed.

## [2026-09-14] query | Compare weak-edge restoration and current training costs

Verified the active frozen-core 1,952-neuron MLP PPO jobs and recomputed fixed-262 edge/contact counts at thresholds one and five. Recorded the cutoff's reference-study/noise-filter provenance, 86.6% reduction in neuron-sized buffers, exact rollout/core-save component shapes, narrower interface parameter counts and backend row-chunk counts. Distinguished 2.97x more edges from nearly unchanged state memory and unmeasured end-to-end timing; current NVIDIA allocations are only a live snapshot. No training process, runtime profile or graph artifact changed.

## [2026-09-14] query | Clarify threshold motivation and uncertain wiring

Updated the minimal-circuit analysis and index after checking CPG, BANC influence and descending-network methods. Distinguished weak-edge pruning from calibrated reconstruction-error rejection, and fixed wiring uncertainty from stochastic forward-pass noise. Given the user's preference to avoid uncertain connections, recommended the 262/3,194 candidate as the initial experiment rather than indiscriminate weak-edge restoration; exceptions require additional evidence. No graph, configuration or process changed.

## [2026-09-14] implementation | Launch matched 262-neuron PPO policies

Added a YAML-owned 262/3,194 selector, pinned preparation contract, ordinary/tanh MLP profiles, full-geometry smoke, long-run suites, milestone evaluators and identity/configuration tests. The exact graph exposes 48 sensory, two descending-command and 31 motor ports. Twenty-seven preparation/configuration tests and 107 broader focused tests passed; both smoke cases completed 393,216 frames with finite checkpoint reload actions.

Initially and incorrectly stopped the two existing 1,952 jobs after interpreting GPU placement as replacement authorization. Preserved their checkpoints, attempted restoration after the user objected, and stopped pursuing restoration when directed to run only the 262 jobs. An initial 262 pair was also interrupted during that correction. Per final direction, launched a clean frame-zero pair: clipped Gaussian actor KL 0.004 on GPU 0 and tanh-squashed actor KL 0.016 on GPU 1, both at the original 12,288-environment geometry. Verified 171 populated TensorBoard scalar tags through steps 10,027,008 and 9,830,400, linked both run directories into the existing port-6008 log tree, confirmed both through its runs endpoint, and started both 250M milestone video watchers. Prior artifacts remain preserved; no 1,952 trainer is active.

## [2026-09-14] implementation | Resume tanh 1,952 policy with matched-reward handoff

Stopped only the clipped 262-neuron GPU-0 trainer/coordinator/watcher and preserved its artifacts; the tanh 262-neuron GPU-1 trainer/watcher remain active. Added `resume_training_state`, which restores actor/critic models, actor optimizer, normalization, trackers, epoch and frame while resetting the unsafe simulator and in-flight rollout state. Historical checkpoints lack critic-optimizer moments, so that optimizer restarts; future checkpoints preserve it. Terminal training exits now refresh the rolling `last/model.pth` checkpoint.

Added YAML-owned catch-up/tanh-continuation/non-tanh-continuation suites and an unattended reward-handoff entrypoint. It resumes tanh from epoch 4,000/frame 786,432,000 to the non-tanh history's reward step 2,378,366,976, compares each final 100-point `rewards/step` mean, continues tanh only on a strict win, and otherwise resumes non-tanh. Both branches append events and checkpoints to their original experiment paths so TensorBoard 6008 shows one history per policy. All 150 focused tests passed. Launched the controller in tmux `connectome-ppo-1952-reward-handoff`; the live log confirmed 12 actor-optimizer entries restored and the new event file populated in the original tanh summaries directory. Restarted both existing 1,952 milestone watchers; their retained manifests prevent old videos from being regenerated and let the selected continuation resume the 250M-frame video cadence.

## [2026-09-15] decision | Continue the non-tanh 1,952 policy

The tanh catch-up completed epoch 12,098 and wrote its terminal recovery checkpoint at frame 2,378,563,584. At the common logged reward coordinate 2,378,366,976, an independent read of the original event histories reproduced the controller's final-100-point window: both span steps 2,358,902,784 through 2,378,366,976, with mean `rewards/step` 386.2670983887 for tanh and 449.7981399536 for non-tanh. Because tanh was not strictly higher, `decision.json` selected `ppo_1952_mlp_adapters_nontanh_resume_continue.yaml`.

The selected clipped-Gaussian KL-0.004 continuation launched on CUDA 0 and restored the non-tanh actor/model, normalization, trackers, epoch/frame and all 12 actor-optimizer state entries from epoch 12,000/frame 2,359,296,000. That historical checkpoint has critic weights but predates critic-optimizer persistence, so the log explicitly records a fresh critic optimizer and fresh simulator rollout. The trainer advanced beyond the comparison coordinate while remaining live. TensorBoard on port 6008 exposes exactly one original non-tanh run entry with a second event file from the continuation. Because the recovery checkpoint precedes the old scalar tail, TensorBoard's restart purge replaces the overlapping visible tail; the pre-resume event file and frozen decision JSON retain the exact comparison window. Both 1,952 milestone watchers remain `running`, each with nine completed 250M-frame evaluations through 2.25B and zero failures.

## [2026-09-15] query | Clarify MaleCNS data versus invented dynamics

Audited the official MaleCNS release inventory, compact preparation path, generated NPZ fields, actor recurrence and primary connectome-simulation literature. MaleCNS provides reconstructed anatomy: cell identities and annotations, directed synaptic partners/contact counts, locations and morphologies, plus predicted presynaptic neurotransmitters. The policy artifact retains only body IDs, sparse edge structure, signed raw/normalized values and sensory/descending/motor masks. It contains no measured activity traces or parameters for membrane/synaptic dynamics, delays, postsynaptic receptor effects, gap junctions, neuromodulation or plasticity. Documented that the rate state, synchronous `tanh` update, leak, recurrent scale, normalization, sign conversion, compact subgraph, robot adapters and action readout are modelling choices. The controller is therefore MaleCNS-derived/connectome-constrained, not a full biophysical MaleCNS simulation. No training process or runtime configuration changed.

## [2026-09-15] query | Trace critic inputs with the actor value loss disabled

Verified that `use_experimental_cv: false` does not change the separate asymmetric critic or its inputs when `central_value_config` remains configured. The environment supplies 162 clean privileged state values: the 140 policy fields plus 22 velocity, episode-history, progress, success and scaled-reward fields. SAPG appends a scalar member identifier that the critic replaces with a learned 32-value embedding, producing the 194-value MLP input. The critic consumes neither connectome hidden state nor actions. The false flag only zeros the actor-side value loss; central rollout values, bootstrapping, advantages and central-value training remain active. No runtime configuration or process changed.

## [2026-09-15] query | Trace the experimental central-value flag provenance

Traced `use_experimental_cv` through local and upstream Git history. It is an inherited `rl_games` compatibility switch, not a connectome or SAPG mechanism. Upstream added `use_old_cv` on 2021-01-08 to default back to training the policy network's value head alongside a configured central critic, renamed it `use_experimental_cv` on 2021-01-09 while extending the path to discrete A2C, and retained continuous PPO's `true` default in the 2021-01-30 asymmetric-critic repair. SimToolReal's initial 2026 release vendored that behavior unchanged. In current continuous PPO the flag simply selects whether the actor/model value head gets an auxiliary critic loss when a separate central-value network exists. No runtime setting or process changed.

## [2026-09-15] query | Audit all training results and the full MaleCNS control pipeline

Read 73 local TensorBoard event files across 71 physical run directories and all 86 evaluation summaries at a fixed 07:12 EDT snapshot. Rehashed pinned raw sources, reconstructed the 4,778 candidate groups, and reaggregated the 6.4 GB partner data to reproduce all 1,110,601 cached directed-pair counts. Verified graph orientation, identities, compact populations, directed reachability, incoming-contact retention and local dynamics. Thirty-three selected CPU tests and 21 configuration tests passed; explicit new-YAML composition assertions and 48 local wiki links also passed. GPU custom-backend tests were not run during active training.

The new `analyses/system-audit-2026-09-15.md` records that the 1,952 clipped policy has the strongest training return but growing raw variance; tanh policies control variance while plateauing mainly at lifting. A no-update checkpoint probe exposed cross-member policy-conditioning KL in SAPG's relabeled batches, weakening the previous optimizer-only interpretation of first-mini-epoch spikes. Saved-state probes also found 94.30% sensory and 99.66% descending preactivation saturation in the 1,952 tanh circuit, an auxiliary actor-value objective despite the separate central critic, and an exact one-step input-to-action delay. Evaluation uses ID 50 while ordinary training return describes ID 0; milestone Task Progress remains almost flat. The 262 graph's two same-type descending ports and limited incoming context weaken its full-task recommendation.

Updated the actor, success-metrics, compact-training, minimal-circuit and index pages with dated findings and historical-status clarifications. Added a separate proposed `configs/connectome/suites/ppo_1952_tanh_onpolicy_noaux_1b.yaml`, composed and validated without launch: frozen 1,952 tanh MLP, on-policy-only samples, auxiliary actor-value loss disabled, KL .004, LR ceiling .001 and a 1B cap. The report prioritizes teacher/imitation/student-state corrections before RL and functionally validated fly feedback modules, with matched topology controls. Temporary audit helpers, JSON results and a visually checked plot remain under `/tmp/connectome-tb-audit-ra21Aa/`. No trainer, evaluation process, existing training YAML or runtime code was changed.

## [2026-09-15] query | Distinguish tanh saturation and propose multi-rate observation processing

Rechecked the bounded-policy likelihood, saved checkpoint probabilities, synchronous recurrence, and 60 Hz task configuration. Clarified in the system audit and actor page that 94–99% saturation describes internal neurons, whereas the 1,952 tanh checkpoint's expected near-bound action fraction is 9.2–13.4% by block. PPO uses fixed stored latent actions for its score objective, so output tanh does not automatically insert a vanishing derivative into that score gradient; internal tanh does attenuate adapter gradients. Tanh remains a bounded-distribution choice rather than a proven reward winner; a Beta policy is an unimplemented alternative.

Specified an unimplemented K=1/4/8 neural-substep comparison with held input drive, cached adapters, persistent/reset-correct state, and matched training/deployment. For alpha=0.5, verified alpha_K=1-(1-alpha)^(1/K), giving 0.15910/0.08300 for four/eight updates and preserving passive one-control-step retention. The current injection convention needs L+1 inner updates for an L-edge route. Recorded that substeps change nonlinear dynamics, cost recurrent compute, do not fix saturated steady inputs, and cannot be replaced by a longer PPO sequence. The earlier proposed YAML contains neither timing nor drive-calibration changes. Updated the index; no runtime implementation, training configuration or process changed.

## [2026-09-15] query | Specify trainable Beta-policy shape parameters

Documented a proposed Beta(2,2) initialization with independently learned, state-conditioned alpha/beta per action, mapped from [0,1] to [-1,1]. Checked the resulting zero mean, 0.447 action standard deviation and 0.5413 inverse-softplus initialization bias. The primary Beta-policy paper supports softplus-plus-one heads; explicitly labeled their above-one shape restriction and the below-one endpoint-peaked alternative. Recorded concentration-collapse and internal-saturation limitations, plus the need for consistent Beta likelihood/entropy/KL if implemented. Updated the audit and index only; no policy implementation, YAML profile or process changed.

## [2026-09-15] implement | Four-update Beta and Gaussian replacement pair

At user request, stopped the prior trainer (3815434), suite (3815379), handoff coordinator (3715392), and obsolete milestone watchers (3212090, 3716627, 3716630), preserving all artifacts and the TensorBoard servers. Verified no remaining trainer/GPU compute process before smoke testing. Implemented YAML-owned repeated neural updates with cached observation encoding and passive-decay-preserving substep leak, trainable Beta heads initialized near (2,2), affine-corrected Beta likelihood/entropy, Beta scheduler KL, and distribution-specific rollout storage. Kept the standard Gaussian path and auxiliary actor value loss in both requested profiles.

Added the two-profile smoke/1B suite YAMLs, Beta train profile, and regression tests; updated the workflow, audit status and index. Ninety-two focused tests passed (59 deselected), including GPU native/Triton four-step gradient parity. Both full-geometry smoke jobs completed 393,216 frames and finite-action deployment reloads, with finite entropy/KL and nonzero actor auxiliary value losses. These checks establish execution correctness, not learning quality or entropy stability over a long run. Pre-existing untracked `tmp/` remains untouched. Long-run launch evidence follows separately.

## [2026-09-15] launch | Extend requested pair to 100 billion frames each

The initial 1B pair ran under suite PID 3966519 with Beta/Gaussian children 3966576/3966577. The user corrected the requested budget to 100B per job. Stopped those exact processes and retained epoch-10 / 1,966,080-frame checkpoints for both. Added the `_100b.yaml` suite with 508,626 total epochs, reaching 99,999,940,608 frames below the 100B cap. All other requested settings remain unchanged. Configuration regression: 22 tests passed; the final training-state-resume variant also passed the targeted composition test.

Both initial full-state resume attempts exited with SIGSEGV before training. Preserved their `_100b` artifact tree and selected the existing `resume_training_state` mode, restoring policy, critic, optimizer states and counters while resetting simulator and recurrent rollout state. The successful retry writes to `_100b_training_state`, runs in tmux `connectome-1952-4update-100b`, and has actual trainer PIDs 3968541 (CUDA_VISIBLE_DEVICES=0, Beta) and 3968542 (CUDA_VISIBLE_DEVICES=1, Gaussian). Live event inspection verified finite actor losses, nonzero auxiliary actor value loss, finite entropy/KL, and progress to at least 2,555,904 / 2,752,512 logged frames respectively. These are launch-health observations, not convergence evidence. The suite entrypoint and imported helper responsibilities are documented in the experiment workflow. Implementation checkpoints: ea393f14; budget correction f3d95874; training-state retry a7301d5f. No merge or push; unrelated `tmp/` preserved.

## [2026-09-15] query | Compare KL-to-LR settings against the named LR-.01 run

Read all three saved resolved configurations and traced actor/critic scheduler construction, minibatch KL dispatch, dataset parameter refresh and mini-epoch scheduler updates. The new pair retains KL target .004 and the .002/.008 LR increase/decrease boundaries, but lowers maximum LR from .01 to .001 and disables historical leader/follower cross-member experience. Recorded exact factor-1.5 rules, two scheduler decisions per rollout, Beta versus raw-Gaussian KL, non-cumulative/no-hard-stop limits, and the independent constant-LR central critic in the experiment workflow. No training configuration or process changed.

## [2026-09-15] implement | Enforce evaluation neural-update consistency

Traced normal and milestone evaluation through the saved policy config, worker, RlPlayer and shared connectome forward implementation. Added player-side K validation against the checkpoint's saved run configuration and recorded instantiated K in evaluation result JSON. Expanded deployment tests across Beta/Gaussian and K=1/4/8, counting recurrent calls, checking reset parity and rejecting mismatched configs. Forty focused network/multirate tests passed (59 deselected). Both real 100B resolved configurations and their resume-source checkpoints also passed a CPU/native-CSR RlPlayer probe with exactly four recurrent calls per action, finite actions and reset parity. The workflow documents entrypoints, policy_config_path and standalone-export limitations. No simulator evaluation was launched and no active training process/configuration changed; unrelated tmp/ preserved.

## [2026-09-15] configure | Add video watchers for both four-update policies

Added separate YAML-owned milestone video jobs for the active Beta/Gaussian 100B training runs. Each polls its matching inference-checkpoint directory every 30 seconds, evaluates every 250M-frame target through the near-cap final target on the corresponding training GPU, and renders deterministic mean-action videos for the three standard high-resolution task cases. Both policy paths select the active run's saved resolved configuration, enforcing K=4. Added composition tests for suite identity, GPU mapping, milestone budget/cadence, evaluation cases, rendering settings and resolved K; two focused tests passed. Launch evidence follows after process verification. Existing trainers and unrelated tmp/ remain untouched.

## [2026-09-15] launch | Start Beta and Gaussian milestone video jobs

Launched independent restartable watchers in tmux sessions `connectome-4update-beta-videos` and `connectome-4update-gaussian-videos`, with watcher PIDs 3990852 and 3990855. Both consumed the target-250M checkpoint saved at actual frame 250,085,376 / epoch 1,272 and completed all three nonempty videos with no recorded failures. Each case JSON reports `neural_updates: 4`; both milestone status files report one of 400 targets complete and remain `running` for future checkpoints. Training PIDs 3968541/3968542 remained active throughout. Generated evaluation artifacts are ignored and were not committed.

## [2026-09-15] implement | Prepare guarded unrestricted-Beta fallback

At user request, added a dormant one-shot monitor for the active clipped-Gaussian K=4 job. The YAML-owned trigger is an unexpected trainer exit or non-finite core TensorBoard/checkpoint training state; it deliberately excludes finite entropy magnitude and the undefined on/off-gradient similarity diagnostic. The replacement is a fresh, otherwise matched 100B unrestricted-Beta policy on GPU 1 with KL .004, auxiliary actor value loss, K=4 and its own milestone-video watcher. Generalized the Beta shape floor while retaining the existing restricted profile's default floor of one; the new profile uses 1e-4 and remains initialized near (2,2). Added incremental TFRecord and checkpoint checks, exact process-tree shutdown, verified-start launch control, configuration coverage, and below-one-shape tests. Twenty-three focused tests passed (21 deselected). At inspection, Gaussian entropy was finite at 180.7924 near frame 1,464,532,992, core metrics/checkpoint tensors were finite, and both trainers remained active; no fallback condition had occurred. Launch evidence follows separately. Unrelated tmp/ remained untouched.

## [2026-09-15] launch | Start Gaussian numerical fallback guard

Launched `run_connectome_numerical_fallback.py` in tmux session `connectome-gaussian-unrestricted-beta-guard` (PID 4070317). Its initial complete event scan excluded only the explicitly unmonitored on/off-gradient similarity diagnostic and recorded `status: monitoring`, Gaussian trainer PID 3968542, entropy 182.6997 and aggregate KL .005345 at frame 1,480,458,240. The latest checked epoch-7,200 / frame-1,415,577,600 checkpoint had no non-finite trainable tensors. No fallback condition fired; both original trainers and video watchers remained active. The persistent status path is `train_dir/connectome/adaptation_100b_gains_update_timing/gaussian_to_unrestricted_beta_monitor/status.json`. Implementation commit: 733feb70.

## [2026-09-15] query | Compare current Gaussian with historical KL-.004 MLP run

Recursively compared both saved resolved configurations and checked runtime/default seams. The meaningful differences are K=4 versus historical K=1, on-policy-only versus leader/follower cross-member reuse, actor maximum LR .001 versus .01, training-state resume provenance versus a fresh start, and 200- versus 3,000-epoch recovery saves. Both are otherwise the same clipped-Gaussian frozen-1,952 adapters-only MLP actor, batch/rollout geometry, task, entropy, PPO/critic, KL-.004 and 100B contract. Confirmed that the historical omission of `use_experimental_cv` still means true by default, so auxiliary actor value loss is not a difference. Recorded the exact comparison and matched-exposure caveat in the experiment workflow; no runtime setting or process changed. Unrelated tmp/ preserved.

## [2026-09-15] query | Explain SAPG cross-member experience reuse

Traced `use_others_experience` from six-block rollout collection through coefficient rotation, leader filtering, return/value reconstruction, recurrent-state copying, PPO likelihood ratios, KL scheduling and logical minibatching. Documented that historical `lf` with ratio 1 adds one 32,768-sample nonzero-member block relabeled as ID 0 to the original 196,608 samples, while retaining source actions, behavior likelihood/distribution and recurrent history. This is fresh cross-conditioned off-policy reuse, not stale replay or all-to-all sharing. It can produce non-unit PPO ratios and large scheduler KL without an optimizer update. The current explicit `none` preserves all six members' own rollouts and entropy coefficients but removes only this extra relabeled block. Also recorded that null is not equivalent to `none`, and that the code-comment evaluation leader ID 0 differs from deployment ID 50. No runtime configuration or process changed; unrelated `tmp/` preserved.

## [2026-09-15] launch | Replace clipped Gaussian with leader/follower unrestricted Beta

At user request, changed only the prepared unrestricted-Beta fallback's cross-member setting from `none` to explicit `lf` with `off_policy_ratio: 1.0`; KL remains .004. Updated its configuration contract test and workflow. The composition test passed. Terminated exact Gaussian trainer PID 3968542 with SIGTERM, leaving restricted-Beta PID 3968541 untouched. The existing guard observed the configured grace interval, stopped Gaussian watcher PID 3990855, and launched unrestricted-Beta suite/trainer PIDs 4088170/4088218 plus video watcher PID 4088221 on GPU 1. The resolved config confirms unrestricted Beta minimum shape .0001, initialization 2.0, K=4, auxiliary actor value loss, LF/1.0, KL .004 and 100B. At frame 1,769,472, entropy 16.49099, aggregate KL .00089783 and core losses/reward were finite. This is launch-health evidence, not convergence evidence. Gaussian artifacts were preserved; unrelated `tmp/` remained untouched.

## [2026-09-15] launch | Replace unrestricted Beta with restricted Beta LF

Stopped exact unrestricted-Beta suite/trainer/watcher PIDs 4088170/4088218/4088221 and preserved their artifacts. Its final TensorBoard point at frame 193,462,272 had entropy -14,694.9385 and aggregate KL 3,916.8435, with recovery artifacts through epoch 800; these values were finite but severely unstable. Added a distinct YAML-owned restricted-Beta LF run and matching milestone-video configuration rather than reuse its output tree. The new contract restores minimum shape 1.0 while retaining Beta(2,2) initialization, `use_others_experience: lf`, ratio 1.0, KL .004, K=4, auxiliary actor value loss, GPU 1 and 100B. The composition test passed. Launched suite/trainer PIDs 4102855/4102915 and watcher PID 4103035; the resolved YAML matches the contract, and by frame 1,376,256 entropy 16.48605, aggregate KL .00147602 and all checked core scalars were finite. Original restricted-Beta PID 3968541 continued unchanged on GPU 0. This establishes launch health, not convergence. Untracked `.vscode/` and `tmp/` were preserved.

## [2026-09-15] query | Interpret live restricted-Beta entropy

Traced the Beta entropy implementation and live TensorBoard histories for the two active jobs without changing either process. Documented that the scalar is affine-corrected physical-action differential entropy in nats, summed over 29 independent coordinates and averaged over PPO data, before SAPG coefficient weighting. Established reference totals 20.1013 for uniform, 16.4736 for Beta(2,2), 8.9508 for symmetric Beta(4,4) and 6.1627 for Beta(5,5), while warning that aggregate entropy cannot identify actual state/joint shapes and may legally be negative. At the 21:09 UTC snapshot, on-policy restricted Beta was near 2.55B frames with last-100 mean 6.03; LF restricted Beta was near .70B with last-100 mean 5.74. At matched .70B exposure, on-policy was about 9.50 versus LF 5.0-5.3, so LF is concentrating faster, not exhibiting positive entropy blowup. LF KL was about .029 and its global entropy double-weights ID 0, preventing a naive scalar-only conclusion. Recorded per-block values and the need for alpha/beta, concentration and physical-standard-deviation telemetry. Untracked `.vscode/` and `tmp/` remained untouched.

## [2026-09-15] query | Diagnose lower LF Beta entropy

Traced the matched-step entropy gap through sample composition, entropy-coefficient selection, shared Beta heads/adapters, PPO relabeling, advantage normalization, return construction and scheduler KL. LF adds one zero-entropy-coefficient ID-0 block, reducing the sample-average entropy coefficient 14.3%, doubling ID-0 representation and processing 16.7% more samples per rollout without adding an optimizer step. At exact step 817,299,456, simply double-weighting the on-policy ID-0 entropy would lower its aggregate from about 6.085 to 5.385, explaining about .70 of the observed 2.50-nat gap to LF 3.590; the remaining roughly 1.80 reflects actual policy differences. LF had higher ID-0 entropy but lower blocks 0-4, consistent with exploratory follower actions potentially broadening ID 0 while its extra zero-regularization shared gradients reduce other blocks, though the scalar does not establish causality. Recorded the confounds from follower likelihood/history, one-step returns, joint advantage normalization and cross-conditioned scheduler KL, plus the ablations needed to isolate them. No trainer or configuration changed; untracked `.vscode/` and `tmp/` remained untouched.

## [2026-09-15] query | Locate restricted-Beta action-density peaks

Probed the rolling saved observations/recurrent states and the exact matched epoch-3,200/frame-629,145,600 full checkpoints for both active restricted-Beta policies. Formal modes frequently lie near `-1` or `+1` and favor the negative side, but almost-flat shapes near one make mode location exaggerate endpoint concentration. In the current sharpest members, on-policy ID 0 placed 38.6% of probability in the outer 20% of the action interval, versus 22.9% for LF ID 10 and 20% for a uniform density. Current on-policy ID 0 is therefore genuinely endpoint-concentrated; LF ID 10 has boundary-leaning formal modes but only mild extra boundary mass. Documented the per-action hotspots, matched-checkpoint corroboration, deterministic-mean evaluation semantics, and float32 rounding of `1 + softplus(raw)` to exact shape one. No trainer or configuration changed; untracked `.vscode/` and `tmp/` remained untouched.

## [2026-09-15] query | Trace Beta-parameter prediction

Traced the exact restricted-Beta actor path from normalized observations and the learned SAPG-ID embedding through sensory/descending MLPs, four recurrent connectome updates, the 135 motor-neuron slice, and two independent 29-output MLP shape heads. Documented initialization near `(2,2)`, softplus-plus-one conversion, unit-interval rollout storage, affine action mapping, PPO likelihood/entropy/KL gradients, frozen recurrent-core boundary and deterministic mean-action evaluation. Alpha and beta are state- and history-conditioned outputs, not fixed constants or coefficient-block lookup tables. No trainer or configuration changed; untracked `.vscode/` and `tmp/` remained untouched.

## [2026-09-15] query | Explain the Gaussian 41-nat entropy baseline

Revalidated that clipped-Gaussian TensorBoard entropy sums raw unbounded Gaussian differential entropy over all 29 action coordinates before averaging samples. Zero-initialized log standard deviations give sigma one and `29 * .5 * log(2*pi*e) = 41.1492` nats, so a start above 40 is the designed baseline rather than evidence of initial numerical blowup. Recorded the distinction from bounded executed-action entropy and the fact that positive SAPG coefficients can subsequently increase the unbounded raw quantity. The interrupted LF-replacement request was not executed; no trainer or configuration changed, and untracked `.vscode/` and `tmp/` remained untouched.

## [2026-09-15] launch | Replace on-policy Beta with fivefold-entropy LF

At the corrected user direction, stopped only the longer-running non-LF restricted-Beta trainer/suite/watcher PIDs 3968541/3968492/3990852 and preserved its artifacts, including the epoch-16,400/frame-3,224,371,200 rolling checkpoint. Left the existing LF trainer/watcher PIDs 4102915/4103035 untouched. Added distinct YAML-owned training and milestone-video contracts for a fresh GPU-0 restricted-Beta LF run with entropy scale `.025`, producing coefficients `[.0125,.0100,.0075,.0050,.0025,0]` while retaining K=4, KL .004, LR ceiling .001, auxiliary actor value loss, LF ratio 1 and 100B. The focused composition test passed. Launched suite/trainer PIDs 9553/9610 and watcher PID 9904; resolved configuration and finite frame-1,769,472 telemetry verified launch health. Implementation commit: `d0f77c20`. Untracked `.vscode/` and `tmp/` remain untouched.

## [2026-09-15] implement | Add staged 1G LF entropy success handoff

Added a YAML-owned monitor that leaves the active 5x LF run untouched until it crosses 1G frames, compares its closest `mean_successes/frame` point to the fixed 1x LF reference at 1G, and replaces it with a fresh 3x restricted-Beta run only on strict underperformance. If launched, 3x receives its own independent 1G gate before a possible terminal clipped-Gaussian replacement. Added distinct 3x and Gaussian suite/video contracts retaining LF, K=4, auxiliary actor value loss, KL .004 and 100B budgets. Added an opt-in differentiable Gaussian `max_sigma: 3.0` ceiling that preserves sigma-one initialization; existing profiles are unchanged. Focused policy, composition and selector tests passed. At implementation time the fixed reference selected frame 999,948,288/value .0069173179, while 5x was at frame 293,535,744 with no above-1G point, so no replacement decision was due. Untracked `.vscode/` and `tmp/` were preserved.

## [2026-09-15] launch | Start staged LF entropy success monitor

Launched `run_connectome_success_handoff.py` in tmux session `connectome-lf-entropy-success-handoff` with PID 35283. The first status persisted `monitoring`, the fixed 1x reference bracket/selection, the latest 5x point at frame 304,152,576/value .0006510417, and exact live candidate suite/trainer/watcher PIDs 9553/9610/9904. Because 5x had no at-or-above-1G point, no comparison or replacement occurred. The 1x suite/trainer/watcher PIDs 4102855/4102915/4103035 remained live and untouched. Generated status is ignored; implementation commit: `b80406c2`.

## [2026-09-15] configure | Raise both LF entropy success gates to 1.25G

Superseded the 1G comparison contract before it made any decision or transition. Changed the shared YAML target to 1,250,000,000 frames, which applies first to 5x and then independently to a newly launched 3x candidate. Added target-aware state migration: on a pre-transition retarget, the monitor archives the prior bracket summary, clears target-dependent metric offsets and rescans both histories, while refusing retargets after a transition or terminal decision. The new fixed 1x reference selects frame 1,250,033,664/value .0086263027; 5x was only near 339M and remained ineligible for comparison. The focused handoff tests passed, and no trainer or video watcher was stopped.

## [2026-09-15] launch | Restart success monitor at 1.25G target

Stopped only the old handoff monitor PID 35283 through its tmux session and restarted the updated contract in the same `connectome-lf-entropy-success-handoff` session as PID 38375. The migrated status records target 1,250,000,000, archives the prior 1G bracket points, selects the fixed reference at frame 1,250,033,664/value .0086263027, and leaves the 5x selection null at frame 343,867,392 because it has not crossed the target. Both candidate and reference suite/trainer/watcher PID triples remained unchanged at 9553/9610/9904 and 4102855/4102915/4103035. No training or video process was stopped. Contract commit: `f768df1a`.

## [2026-09-15] configure | Use 100M-window true objective for entropy handoff

Superseded the single-point `mean_successes/frame` comparison before any decision or transition. The YAML contract now uses the exact `true_objective_mean/frame` tag and an arithmetic mean over all logged samples in the inclusive matched `[1.15B,1.25B]` window, while still requiring each candidate to cross 1.25B and using strict less-than replacement. Generalized state migration to key on the full metric/target/aggregation/window/comparator contract, archive prior point summaries and rescan event histories. The fixed 1x window contains 508 samples from frame 1,150,156,800 through 1,249,837,056 with mean .0000419941081; 5x had not reached the window. Focused tests passed and no training/video process changed during configuration.

## [2026-09-15] launch | Restart handoff with windowed true objective

Stopped only point-metric monitor PID 38375 through its tmux session and restarted the new contract in the same `connectome-lf-entropy-success-handoff` session as PID 41405. Persisted state confirms `true_objective_mean/frame`, arithmetic mean, target 1.25B, window 100M and strict-less comparison. It archived the old `mean_successes/frame` contract, reproduced the 508-sample fixed reference mean .0000419941081, and kept the 5x window/crossing null because that run remained below 1.15B. Candidate and reference suite/trainer/watcher PIDs remained 9553/9610/9904 and 4102855/4102915/4103035. No training or video process was stopped. Contract commit: `e4ae09ba`.

## [2026-09-16] launch | Complete entropy handoff and start capped Gaussian LF

The windowed handoff completed both gates. The 5x restricted-Beta candidate averaged .0000221648921 versus the fixed 1x true-objective mean .0000419941081 and was replaced by fresh 3x restricted Beta. After its independent 1.25B gate, 3x averaged .0000265286654 and was also replaced. The terminal suite/trainer/video watcher are live as PIDs 177271/177336/177338. The saved resolved configuration confirms clipped Gaussian, `expl_reward_coef_scale: .015` (three times the `.005` baseline), `use_others_experience: lf`, `off_policy_ratio: 1.0`, per-coordinate `max_sigma: 3`, K=4, KL .004 and auxiliary actor value loss. The 1x reference trainer PID 4102915 remains live. Generated status and run artifacts remain ignored.

## [2026-09-16] launch | Replace 3x Gaussian LF with fresh 1x run

Added distinct YAML-owned suite and milestone-video contracts for a fresh capped-Gaussian LF run with `expl_reward_coef_scale: .005`, preserving LF/1.0, sigma cap 3, K=4, KL .004, LR ceiling .001, auxiliary actor value loss, GPU 0 and 100B. Thirty-one focused composition tests passed. The first substring-based shutdown attempt matched its own invoking shell and interrupted before stopping the old process; the new suite briefly started alongside it. Corrected immediately using explicit previously verified PIDs, then confirmed no 3x process remained. The stopped 3x run's last finite point was frame 1,220,345,856 with entropy 55.69760 and KL .27626. Launched the 1x suite/trainer/watcher as PIDs 261900/261951/262494; its resolved config matches the contract, and frame-1,769,472 entropy 41.20781, KL .00088270 and actor/value losses were finite. The restricted-Beta reference processes remained untouched. Config commit: `1b7525eb`; unrelated `.vscode/` and `tmp/` remain untracked.

## [2026-09-16] design | Specify structured sensory adapter

Traced the live 140-value policy observation layout, compact 1,952-cell population contract, runtime sensory/descending scatter, robot DOF order and local-eligibility assumptions. Specified a YAML-selected grouped linear profile: 102 body/efference values drive only the 92 proprioceptor-labelled cells; the 292 tactile-labelled cells receive no fabricated direct touch signal; 26 object/context values join 12 goal values and the 32-value SAPG embedding at the descending adapter. Documented the required subpopulation artifact masks, exact observation-partition validation, dense-checkpoint compatibility, eligibility-trainer exclusion, shuffled-assignment control and population-level diagnostics. This is an implementation/evaluation plan, not a completed training result. No trainer or runtime configuration changed; untracked `.vscode/` and `tmp/` were preserved.

## [2026-09-16] design | Separate structured projection and rotation choices

Verified that current policy observations expose absolute palm and object quaternions in Isaac Gym `xyzw` order, with noisy/delayed object orientation also determining observed keypoints, and that rl_games applies componentwise running normalization. Refined the structured proposal to use an exact masked linear body-to-proprioceptor route and a separately configurable small MLP for context/goal/SAPG descending drive. Kept the current raw quaternion representation fixed for the first routing comparison, then proposed a separate fresh-profile ablation using six-dimensional rotation-matrix columns or explicit relative rotation. Recorded quaternion double-cover, linear-relative-rotation and checkpoint/normalization boundaries. No runtime configuration or process changed.

## [2026-09-16] implement | Add structured 6D Gaussian and Beta suites

Superseded the earlier quaternion-first proposal at user direction and implemented a fresh 144-value task contract that converts palm/object `xyzw` quaternions to 6D rotation-matrix columns before running normalization. Extended compact preparation with verified 92-cell proprioceptor and 292-cell tactile artifact masks. Added six independent grouped linear body/efference adapters, exact zero direct tactile drive, a `74 -> 128 -> 157` descending MLP, observation-partition validation, dense-profile compatibility and explicit eligibility-trainer rejection. Added two executable suite YAMLs matched to the currently running capped-Gaussian GPU-0 and restricted-Beta GPU-1 LF contracts; no trainer, watcher or existing process was changed. The generated 1,952 artifact prepared successfully, 119 focused tests passed, and both real-graph structured distributions completed finite CPU forward/backward checks. Implementation commit: `19ed68e5`; untracked `.vscode/` and `tmp/` were preserved.

## [2026-09-16] launch | Replace structured Beta with capped Gaussian

Stopped only the structured Rotation-6D restricted-Beta suite/trainer PIDs 319189/319244 on GPU 1 and preserved its artifacts through frame 163,381,248. Its last entropy 12.85896, aggregate KL .01079517 and actor/critic losses were finite. Reassigned the already implemented structured Gaussian suite from GPU 0 to GPU 1 and launched it fresh in tmux session `connectome-structured-rot6d-gaussian-lf-100b` as suite/trainer PIDs 332402/332463. The resolved contract preserves the 144-value Rotation-6D observation, grouped 92-cell proprioceptive routing, zero direct tactile input, 128-hidden-unit descending adapter, K=4, LF/1.0, entropy scale .005, KL .004, auxiliary actor value loss and 100B budget. Its action path matches the current dense Gaussian: hard-clipped Gaussian actions with coefficient-conditioned scale and differentiable sigma cap 3 per coordinate. At frame 3,735,552, entropy 41.30614, KL .00194340 and actor/critic losses were finite. No structured video watcher was started because no 144-value structured evaluation contract exists yet. The dense Gaussian suite/trainer/watcher PIDs 261900/261951/262494 remained live on GPU 0 and untouched. Focused structured configuration test passed; implementation commit: `1494c83d`. Untracked `.vscode/` and `tmp/` remain untouched.

## [2026-09-16] implement | Add fixed-input cached MaleCNS reservoir

Stopped only the structured Rotation-6D Gaussian suite/trainer PIDs 332402/332463 on GPU 1 and preserved its run artifacts; the dense capped-Gaussian suite/trainer/watcher PIDs 261900/261951/262494 on GPU 0 remained untouched. Added a YAML-owned parameter-free population encoder for the 144-value Rotation-6D observation: 87 channels drive selected proprioceptors, 98 channels drive descending cells, and all tactile cells receive zero direct input. The fixed 1,952-cell circuit runs four updates under `no_grad` during rollout; PPO stores only 135 motor features and trains restricted-Beta `167 -> 128 -> 29` alpha/beta readouts plus the SAPG embedding. `use_experimental_cv: false` leaves the privileged critic authoritative. A successful GPU-1 smoke completed two epochs/393,216 frames and checkpoint deployment verification; warm timing was 1.082 seconds rollout, .267 seconds update and 1.349 seconds total, with an observed trainer footprint near 6.2 GiB. Two failed integration artifacts were preserved and their auxiliary-buffer/gradient-telemetry bugs fixed. All 134 focused tests passed. Added a fresh 100B suite contract; no evaluation watcher exists yet. Untracked `.vscode/` and `tmp/` remain untouched.

## [2026-09-16] launch | Start fixed-input reservoir Beta on GPU 1

Launched the fresh 100B contract in tmux session `connectome-fixed-reservoir-beta-lf-100b` as suite/trainer PIDs 368787/368865 on physical GPU 1. The resolved YAML confirms 12,288 environments, Rotation-6D, the fixed proprioceptor/descending map, cached 135-cell motor features, restricted Beta, readout-only SAPG conditioning, K=4, LF/1.0, entropy scale .005, KL .004, `use_experimental_cv: false` and the 100B budget. At epoch 28/frame 5,308,416 the job was live at 145,946 frames/s with 1.086-second rollout and .261-second update times; its process allocation was about 10.2 GiB after the optimizer had warmed. The dense Gaussian suite/trainer/watcher PIDs 261900/261951/262494 remained untouched on GPU 0. No fixed-reservoir watcher was launched.

## [2026-09-16] launch | Replace fixed-reservoir Beta with matched Gaussian

Verified that the Beta run's zero `losses/c_loss` was the configured `use_experimental_cv: false` actor-side value loss, while its privileged `losses/cval_loss` remained nonzero. Stopped only fixed-reservoir Beta suite/trainer PIDs 368787/368865 near 286M frames and preserved its artifacts and TensorBoard history. Added cached-feature support to the clipped log-standard-deviation wrapper and YAML-owned Gaussian smoke/full profiles exactly matching the live dense Gaussian's coefficient-conditioned sigma, sigma-three ceiling, LF/1.0, entropy scale .005, K=4, KL .004 and `use_experimental_cv: true`. All 136 focused tests passed. The smoke completed two epochs/393,216 frames and deployment reload; warm update time was .263 seconds, with independently nonzero `c_loss` and `cval_loss`. Launched the full replacement in tmux `connectome-fixed-reservoir-gaussian-lf-100b` as suite/trainer PIDs 387124/387180 on GPU 1. At frame 1,769,472 entropy 41.2216, actor `c_loss` .9566, privileged `cval_loss` .07679 and KL .00310 were finite. Added it to the running port-6008 TensorBoard by symlink without restarting the server. Dense Gaussian PIDs 261900/261951/262494 remained untouched; no reservoir watcher was launched.

## [2026-09-16] implement and launch | Optimize full-CNS visual inference

Implemented inference-only tanh buffers, optional CUDA graph capture and fused image/retinal kernels, with defaults unchanged for other policies. All 127 regression tests and seven 12-epoch PPO/checkpoint tests passed. Added YAML-owned isolated kernel/environment/training profiling and an optimized camera audit. Selected 768 environments with frozen buffers and image fusion, no capture: 3,688 median total FPS versus 3,252.5 baseline; 1,536 reached 3,755 but doubled state/camera counts. Full measurements, caveats, scripts and configuration usage are in [Visual reservoir performance](analyses/visual-reservoir-performance.md).

Stopped only the prior visual trainer/watcher and preserved artifacts. Continued its full actor/critic/optimizer checkpoint at frame 1,953,792 with fresh simulator episodes; suite/trainer PIDs 495083/495124 and watcher 494868 now run on GPU 1. Verified TensorBoard 6008 alias `full_cns_tanh_vision_fast_100b` and advancement beyond 2.17M frames, finite losses and valid scheduler KL. The watcher began the new 2M evaluation. GPU-0 Gaussian trainer 261951 and the existing TensorBoard server were untouched. Commits `c3f24946` and `e8bea153`; the latter corrects a preflight-only epoch-budget rejection. No merge or push; unrelated `.vscode/` and `tmp/` preserved.

## [2026-09-16] experiment and launch | Fill GPU 1 and qualify evaluation headroom

Added YAML-owned memory-monitored capacity probes and two passing orchestration tests; 1,920, 2,304 and 2,688 environments each completed 12 PPO epochs and finite-action checkpoint reload. Their warmed median total FPS were 3,647.5, 3,610.5 and 3,510.0, with sampled GPU memory peaks 16,458, 18,894 and 21,376 MiB. A 3,072-env attempt died with native SIGSEGV before training; its exact cause remains unproven. The telemetry monitor now records query timeouts instead of abandoning the child.

A resumed 2,688-env trainer fit alone but a concurrent video worker hit CUDA OOM, so that attempt was stopped with artifacts retained. Resumed the same full actor/critic/optimizer checkpoint at frame 2,359,296 into the validated 2,304-env configuration instead. Suite/trainer PIDs 504111/504155 remain live on GPU 1; a real 60-step video worker passed concurrently at sampled total occupancy 22,270 MiB. Restored the three-case milestone watcher and added `full_cns_tanh_vision_capacity2304_100b` to the existing TensorBoard 6008 server without restart. GPU-0 trainer 261951 was preserved throughout. Commits `b5ba3ed0`, `075d2c67`, `7636a96c`; unrelated `.vscode/` and `tmp/` untouched.

Updated [Visual reservoir performance](analyses/visual-reservoir-performance.md) with raw artifact locations, entrypoint/helper usage, all failed-attempt boundaries, and sourced discussion of GPU-native simulation, tiled cameras and asynchronous RL. More VRAM occupancy did not improve throughput in these short tests; no renderer migration or algorithm change was implemented.

# [2026-09-16] query | Clarify full-CNS readout scope

- Updated `docs/wiki/concepts/visual-reservoir.md` to distinguish simulating all traced neurons from exposing all neuron states to the trainable policy readout.
- Recorded the nine-update reachability caveat, direct-input shortcut risk, and rollout-cache costs for motor-only, all-neuron and non-direct descending-plus-motor readouts.

## [2026-09-16] query | Specify anatomical pooling and compact hybrid readout

- Defined a fixed signed-mean anatomical pooling candidate that excludes externally driven cells and preserves coarse optic topography without adding BPTT.
- Audited the 1,952-neuron artifact: it contains 177 annotated descending neurons, including its 157 input ports, plus 135 motor neurons. Recorded the 312-feature hybrid, memory cost, and direct-input controls.

## [2026-09-16] query | Clarify driven-neuron state semantics

- Corrected the readout discussion: driven neurons receive leak, recurrence, bias and tanh processing on every neural update; their final states are not raw input copies.
- Reframed driven-cell exclusion as a conservative short-path ablation rather than a requirement or evidence that those cells bypass all CNS dynamics.

## [2026-09-16] implement | Add compact all-neuron actor readout

- Added YAML-selectable `motor` or `all` tanh actor feature populations while preserving motor-only defaults and explicitly rejecting unsupported all-neuron LIF output.
- Added a learned-MLP Gaussian train profile plus smoke, 100B and milestone-video YAML entrypoints matched to the live GPU-0 Gaussian contract except for GPU and actor feature population.
- All 94 connectome-network tests passed. The full 12,288-environment smoke completed two epochs/393,216 frames and finite-action checkpoint reload in 24.68 seconds of child training.
- Stopped only the GPU-1 full-CNS vision suite/trainer at logged frame 24,514,560; its artifacts remain preserved. The GPU-0 Gaussian suite/trainer/watcher remained live.

## [2026-09-16] launch | Replace vision with compact all-neuron policy

- Launched the fresh 1,952-neuron no-vision all-neuron-readout suite/trainer as PIDs 543985/544079 on physical GPU 1 and its 250M-frame milestone watcher as PID 543989.
- Initial production telemetry was finite through frame 1,769,472 with approximately 95K-101K warm total FPS, reward 34.7445, entropy 41.1834, KL `.00312`, and both actor and value losses present.
- Confirmed that the existing port-6008 TensorBoard discovered the new production and smoke event streams without restart. GPU-0 trainer/watcher PIDs 261951/262494 were preserved.

## [2026-09-16] query | Clarify actor and critic fly features

- Distinguished the privileged central critic from the actor-side auxiliary value head: the central critic consumes no fly state, while `use_experimental_cv: true` trains the auxiliary head through actor features.
- Recorded that the live motor-readout policy sends 135 fly states to its action MLP but all 1,952 states to its auxiliary value head; the all-neuron policy sends all 1,952 states to both heads.
- Documented privileged-state-plus-detached-recurrent-features as a future recurrent-critic ablation, not an established improvement. No configuration or running process changed.

## [2026-09-16] query | Define the standard SimToolReal critic analogue

- Distinguished asymmetric information from shared representation: privileged state is the defining central-critic contract, while actor memory is only an optional additional conditioning variable for a recurrent policy.
- Defined the paper analogue as privileged central critic only and the released-code analogue as privileged central critic plus an auxiliary actor value head sharing the actor's post-recurrent trunk.
- Recorded that neither current connectome readout is an exact shared-trunk reproduction. No configuration or running process changed.

## [2026-09-17] launch | Replace motor readout with all-neuron no-auxiliary comparison

- Stopped only the GPU-0 motor-readout suite/trainer and its current milestone watcher; its artifacts remain preserved through the completed 5,750,194,176-frame inference milestone.
- Added a fresh YAML-owned GPU-0 run and watcher matched to the GPU-1 all-neuron policy except for `use_experimental_cv: false`; configuration commit `51ee9c72`.
- Launched suite/trainer/watcher PIDs 775098/775189/775104. At frame 1,376,256, reward, entropy, action loss, central-critic loss and KL were finite, actor `c_loss` was correctly zero, and invalid-KL flags were zero.
- Preserved the GPU-1 all-neuron suite/trainer/watcher PIDs 543985/544079/555982 and confirmed the new run is visible in TensorBoard 6008 without restart.

## [2026-09-17] fix | Complete central-critic recovery metadata

- Audited live all-neuron PPO artifacts: full checkpoints already contained critic weights and Adam state, while inference milestones intentionally contained neither.
- Confirmed that full checkpoints omitted the critic trainer's epoch, frame, scheduled-LR variable and optional recurrent state, so a resumed scheduled critic could restart its schedule at zero.
- Added explicit central-value training metadata save/restore, legacy inference from actor counters plus critic optimizer LR, fresh-rollout handling that does not restore critic RNN state, and suite verification of the complete critic resume contract.
- Existing trainers were not interrupted and continue writing the old schema until restarted; those checkpoints remain resumable through the compatibility path.

## [2026-09-17] query | Current all-neuron parameter and edge count

- Verified the live all-neuron MLP actor from its recovery checkpoint shapes: 692,268 declared trainable scalars, versus 7,811,468 for the original SimToolReal LSTM/SAPG actor.
- Distinguished the GPU-0 no-auxiliary job's 690,315 actively gradient-receiving actor scalars from its still-declared 1,953-scalar unused value head; the GPU-1 auxiliary-loss job updates all 692,268.
- Recorded the unchanged separate 2,037,769-scalar asymmetric critic and the current compact MaleCNS graph's 33,720 fixed directed connections, including the no-fly-circuit boundary for the original LSTM.

## [2026-09-17] query | Learned interface MLP bottleneck limits

- Derived the current all-neuron MLP actor counts at shared hidden widths 256/128/96/64 and distinguished algebraic input-rank limits from empirical decoder capacity.
- The existing YAML exposes one shared interface width; 128 is the smallest setting that does not necessarily compress the 128-D sensory input, reducing the actor from 692,268 to 347,308 scalars.
- A 128-sensory/64-descending/64-action 207,596-scalar design would require independently configurable widths; it is a proposed matched capacity ablation, not an implemented or evaluated policy.

## [2026-09-17] docs | Add robot-to-fly mapping infographic

- Added `docs/wiki/assets/robot-to-fly-neuron-mapping.png`, a 1,672 x 941 architecture diagram of the current learned-adapter all-neuron policy.
- Embedded the diagram in the connectome actor concept page beside the exact 1,952-neuron and 33,720-edge dataflow description.

## [2026-09-17] query | Aggressive asymmetric MLP capacity

- Calculated 144,172 actor scalars for proposed `128/64/32` sensory/descending/action hidden widths, or 142,219 actively updated scalars when the auxiliary actor-value loss is disabled.
- Distinguished the coordinate-full-rank `128/44/29` floor (134,206), an SAPG-structure-aware `128/18/29` candidate (128,980), and the already-supported direct-linear interfaces (115,016).
- Widths below 29 on the action decoder impose a rank-limited action-synergy assumption; these counts are architectural bounds, not evidence of retained task performance.

## [2026-09-17] implement | Independently sized 128/32/32 interface MLPs

- Added backward-compatible sensory, descending, and readout hidden-width overrides; existing profiles still use the shared 256-unit width when the new keys are absent.
- Added a clipped-Gaussian all-neuron `128/32/32` train profile and YAML-owned two-epoch full-geometry smoke entrypoint without launching or replacing either live job.
- Verified the exact `128 -> 128 -> 384`, `44 -> 32 -> 157`, and `1,952 -> 32 -> 29` shapes. The resulting actor declares 137,740 trainable scalars, including the auxiliary value head and SAPG parameters.

## [2026-09-17] query | Confirm active success-threshold curriculum

- Revalidated the goal-tolerance curriculum from the raw environment scheduler: the 0.075 base tolerance reduces by 0.9 only after both 3,000 vector control steps and an all-12,288-environment mean of at least three completed goals per most recently completed episode.
- Recorded that failed checks leave the update marker unchanged and therefore make the next qualifying control step advance immediately; `evalSuccessTolerance` is null for training and does not override the scheduler.

## [2026-09-16] implement | Add checkpoint-tolerance milestone videos

- Extended YAML-owned milestone evaluation to generate two otherwise identical deterministic video sets per checkpoint: fixed paper tolerance `.02` and the exact training tolerance from `scalars/success_tolerance/frame` at that checkpoint's actual frame.
- Added strict exact-frame TensorBoard resolution, per-metric video completion accounting and case-level reuse so existing fixed-tolerance videos are not rerendered during backfill.
- Updated the active dense Gaussian watcher contract and added a structured Rotation-6D Gaussian watcher contract for its preserved 250M checkpoint. Both retain mean actions, coefficient ID 50, one-step waypoint scoring, trajectories, simulator overrides and rendering settings across the two sets.
- Focused milestone/configuration tests passed; implementation was isolated on `feature/checkpoint-tolerance-videos` because the primary checkout contained unrelated dirty visual-reservoir work.

## [2026-09-16] launch | Start dual-tolerance video backfill

- Replaced only dense Gaussian watcher PID 262494 with updated watcher PID 552085; dense trainer PID 261951 remained live and untouched. Started structured Rotation-6D watcher PID 552088 against its preserved 250M checkpoint.
- The isolated worktree initially lacked the ignored generated MaleCNS NPZ, so first worker attempts failed before policy construction and were retained as retry diagnostics. Linked the existing read-only processed artifact into the worktree; both watchers then ran within available GPU memory and retried automatically.
- The dense watcher reused all matching fixed-`.02` cases. Its first completed checkpoint-tolerance set at target 1.25B contained six total videos, three per metric; the resolved checkpoint tolerance was `0.07500000298`, mean actions remained deterministic, and an inspected MP4 decoded as H.264 at 800x450 with 200 frames. Dense historical backfill remained active at handoff. The preserved structured 250M checkpoint completed all six videos with three per metric, the same exact-frame tolerance, and no outstanding failure.
- Applied the same dual-tolerance YAML contract to the concurrently launched all-neuron-readout Gaussian watcher so every current Gaussian policy evaluator uses the requested two-set protocol.

## [2026-09-18] experiment | Replace auxiliary fly job with matched LSTM

- Updated `SimToolRealLSTM323MatchedGaussianSigma3SAPG` and its suite contracts to make the 256-unit actor MLP, one-update recurrence, physical GPU 1 placement, and `use_experimental_cv: false` explicit.
- Stopped only the physical-GPU-1 all-neuron fly trainer and watcher at printed frame 9,642,442,752; preserved their artifacts and left the physical-GPU-0 no-auxiliary fly trainer and watcher running.
- Launched the fresh seed-42 LSTM production trainer and milestone watcher. Initial finite telemetry through frame 5,505,024 includes zero actor-side value loss, nonzero privileged critic loss, and zero invalid-KL flags.
- Validated the changed LSTM suite contracts with 11 focused tests and a resolved-configuration audit. Early telemetry proves integration and liveness only, not learning quality.

## [2026-09-18] query | Separate coefficient matching from sparsity matching

- Verified through the actor builder that LSTM widths 323, 976 and 1,952 have 733,790, 4,749,740 and 17,111,756 actor parameters respectively.
- Recorded that width 323 matches instantiated coefficients, width 976 matches the fly's 1,952 scalar state budget when counting LSTM hidden and cell states, and width 1,952 matches recurrent output/unit count but carries 3,904 state scalars.
- Clarified that a 1,952-unit dense LSTM is needed alongside the 323-unit control to evaluate equal-width sparsity costs, while same-dynamics randomized sparse graphs remain necessary for biological-topology attribution. No live process was changed.

## [2026-09-18] experiment | Replace matched LSTM with official repository method

- Stopped the physical-GPU-1 323-unit matched-LSTM trainer and watcher at printed frame 261,685,248, preserving its checkpoints, event history and evaluation artifacts; the GPU-0 fly trainer and watcher were not changed.
- Added exact and memory-preserving YAML contracts for the repository's legacy `SimToolRealLSTMAsymmetricSAPG` launch. Both use the user-selected seed 42; the exact contract otherwise retains 24,576 environments, 4,096/block, 98,304 minibatches, KL `.016`, entropy scale `.002`, force/torque 20/2 and `use_experimental_cv: true`.
- The exact geometry fit GPU 1 and completed multiple optimizer epochs, so the fallback was not launched. Initial telemetry through frame 3,538,944 was finite with nonzero actor and privileged value losses and zero invalid-KL flags.
- Launched the official run's 250M-frame milestone watcher and exposed its summaries on TensorBoard 6008 as `official_repo_lstm_sapg_seed42`. Twelve focused configuration/recurrent tests passed.

## [2026-09-18] experiment | Match official LSTM environment count to fly

- Stopped the 24,576-environment official LSTM trainer and watcher at printed frame 18,087,936 with all artifacts preserved; the physical-GPU-0 fly trainer and watcher were not changed.
- Added and launched a fresh official-profile seed-42 run with the fly job's 12,288 physical environments and six 2,048-environment SAPG blocks. It retains the official 98,304 minibatches, KL `.016`, entropy scale `.002`, force/torque 20/2 and `use_experimental_cv: true` without rollout accumulation.
- Initial telemetry through frame 1,769,472 was finite with nonzero actor and privileged value losses and zero invalid-KL flags. The live watcher polls exact 250M milestones.
- TensorBoard 6008 exposes the active run as `official_repo_lstm_sapg_seed42_env12288`; 13 focused tests passed.

## [2026-09-18] query | Verify TensorBoard time-axis semantics

- Traced the RL Games and task observer writers: `/step`, `/iter` and `/time` aliases all receive the same environment-frame `global_step`; their suffixes do not select a TensorBoard axis.
- Verified the active official-LSTM event file directly. All 771 reward-alias entries had identical values and steps from 196,608 through 151,584,768, while event wall timestamps spanned 2,128.256 seconds.
- Recorded that TensorBoard's `Relative` or `Wall` axis must be selected for time. No logger or live process was changed.

## [2026-09-18] query | Explain fly Gaussian entropy ceiling

- Verified the active fly resolved configuration has `max_sigma: 3.0` and six coefficient-conditioned scale rows.
- Derived the builder's shifted softplus ceiling as `sigma = 3 / (1 + 2 * exp(-r))`, preserving sigma one at initialization and giving a 73.009-nat upper bound for 29-dimensional raw Gaussian entropy.
- Distinguished emitted-scale bounding from entropy-loss clamping and executed-action clipping; recorded that the official LSTM omits this custom ceiling. No live process was changed.

## [2026-09-18] consolidate | Merge active repository worktrees

- Consolidated vision reservoir, checkpoint-tolerance videos, and LSTM baseline branches onto `main`; checkpointed the three user-selected local vision YAML changes.
- Resolved wiki conflicts with explicit user authorization, preserving source-backed findings and both log histories. Evaluator merges retain milestone-target checkpoint resolution and dual-tolerance video reuse.
- Focused evaluator/configuration/checkpoint and baseline checks: 48 passed. The consolidation worktree needed a link to shared ignored training artifacts for one existing configuration test.
- Preserved `.vscode/`, `tmp/`, shared artifact links, existing worktrees, and running processes.

## [2026-09-18] implement | Anatomical circuit activity videos

- Added YAML-owned saved-case replay and trace-only rendering entrypoints, optional ordinary/milestone evaluation recording, and source-derived native X-Z anatomy with overview/VNC detail.
- Recorded every compact tanh neural substep without changing actions/state or checkpoint schema; the CPU/CUDA parity and focused integration/configuration tests passed (55 total).
- Downloaded and validated the exact 1,952 circuit body IDs: 194,668,453 bytes of official MaleCNS v1.0 SWC geometry, with URL/hash provenance and strict missing-geometry failures.
- Added separate/combined video encoding at four times the rollout FPS, episode-local resampling and terminal holds; enabled completion checks require traces, rendered videos and matching successful metadata.
- Documentation: `workflows/anatomical-activity-videos.md`, linked from the index and experiment workflow, explains both scripts, helpers, YAML parameters, timing and modeled-state interpretation.
- The first actual 7B no-auxiliary marker replay produced a valid activity trace and inspected anatomical preview; full batch completion is recorded separately after artifact verification.

## [2026-09-18] fix | Protect geometry caches and expose local video index

- Made projected-raster cache writes atomic for parallel evaluator queues; existing geometry downloads use a cache lock.
- Rejected non-boolean enablement, duplicate output names and disabled standalone rendering.
- Added generated local `index.html` to the replay entrypoint for opening every combined video and downloading circuit-only videos, traces and metadata.
- Focused anatomy/timing/encoding/completion tests passed (4); new modules and entrypoints pass Ruff checks. The first full replay completed with 800 frames at 80 fps and exactly 10 seconds for both outputs.

## [2026-09-18] improve | Interpret anatomical activity with labels and leg guides

- Added source-verified sensory/descending/motor captions, brain/VNC labels and approximate T1/T2/T3 orientation guides; faint leg silhouettes are explicitly schematic, not fabricated anatomical leg geometry.
- Added four population mean-absolute-state bars and the actual all-neuron robot-action readout route. Masks/counts come from the exact hash-verified circuit artifact; sensory/motor anatomical captions additionally use source annotations.
- Added appearance-only replay reuse with checkpoint/config/footage verification, preserving existing simulation recordings while rebuilding videos. New traces save their recording options and footage hash.
- Inspected annotated real-data stills at three seconds. Focused recording, CPU/CUDA parity, timing, encoding, configuration, milestone and baseline checks: 57 passed; Ruff and diff checks passed.
- Verified that all six completed 7B recordings can be rerendered without simulator workers. See `workflows/anatomical-activity-videos.md` for flags and helper/entrypoint usage.

## [2026-09-18] improve | Browse matched tasks at slower playback speeds

- Updated the generated local video gallery to group actors by task, explain state/bar/leg conventions and offer 0.25×/0.5×/1× playback.
- Gallery links respect the requested circuit/combined outputs even when older unrequested MP4s remain. Verified six cards, paired task ordering, existing file targets and playback options; Ruff/diff checks passed.
- Clarified that internal algorithm updates are placed uniformly on the animation clock, rather than measured physiological/wall-clock events.

## [2026-09-18] verify | Complete all six annotated 7B task pairs

- Completed the full six-case replay and interpretability rerender for both compact all-neuron actors at target 7B / actual 7,000,031,232 frames, across marker/eraser/spatula tasks.
- All 12 requested MP4s passed complete-frame decoding and 1600 × 900 / 80 fps / exact frame-count checks. Each video matches its 20 fps source footage's 10 or 10.05 second duration.
- Verified finite states, initial state and episode-local timing, exact ordered body IDs and sensory/descending/motor masks, 1,952-neuron anatomy coverage, hash-verified source checkpoints/configs and all-neuron readout captions. All six original trace/robot MP4 hashes stayed unchanged during annotation rendering.
- Generated `verification.json`, original-recording provenance, per-case decoded previews and a six-case preview montage in ignored `evals/connectome/anatomical_1952_7b/`; cached replay rebuilt the task-paired gallery without simulator workers or additional encoding.
- Implementation remains checkpointed on `feature/anatomical-circuit-video` for user evaluation. Preserved the pre-existing `.vscode/` and `tmp/` directories and existing worktrees/jobs.

## [2026-09-18] fix | Preserve bilateral population callouts

- Investigated the user's observation that sensory/motor label leaders targeted only the screen-right bulb. The cause was a pooled median across separated left/right arbor clusters; 68 versus 67 motor cells selects a location in one bulb despite nearly balanced counts.
- Confirmed bilateral source populations: motor `somaSide` L/R = 68/67, sensory `rootSide` L/R = 220/164, descending `somaSide` L/R = 79/78. Source L-side arbors are screen-right in the unflipped native X-Z projection.
- Replaced the pooled anchor with separate source-side anchors, added endpoint markers and the "Both front legs" caption, and clarified that leaders label groups rather than physical drive sites or activity paths.
- Added renderer-version completion checks to refresh stale videos without repeating simulation. Focused anatomy/timing/encoding/completion/recording-reuse checks: 7 passed, including the one-cell-imbalance regression. Inspected the corrected real-data preview before rerendering all six recorded cases.

## [2026-09-18] verify | Refresh all bilateral-callout videos

- Completed all six version-2 renders (12 MP4s) using three CPU processes and the existing saved traces. No simulator workers were dispatched.
- Verified both source-side anchors/counts for sensory, descending and motor groups, complete-frame decoding, 1600 × 900 resolution, 80 fps, matched per-case duration and unchanged original recording hashes.
- Refreshed decoded previews, the six-case montage, verification metadata and local gallery. The existing replay YAML reuses these complete outputs; the standalone render YAML regenerates the corrected labels with no configuration changes.

## [2026-09-18] improve | Enlarge the simulation and compact video text

- Added YAML-owned `robot_crop` and `robot_panel_fraction`, with normalized crop validation and a 40–70% width range. The six-case presets use `[0.22, 0.06, 0.78, 0.88]` and 58%; sampled source frames across all six rollouts keep the robot, tool and table in view.
- Expanded the shared content height to 708 pixels at 1600 × 900 and scaled the cropped simulation to 860 × 708, preserving its aspect ratio. Timing/counts, population names/roles and legend/credits now use horizontal rows; removed model/task names and the gray-leg explanatory line from encoded frames.
- Added shared viewport geometry, fixed camera cropping and render-layout metadata to the anatomy helper. Renderer version 3 triggers appearance-only refresh without new simulation recordings.
- Inspected actual-data combined and circuit-only stills. All seven existing focused anatomy/timing/encoding/completion/recording-reuse tests passed. Batch encoding and verification are recorded separately when complete.
- Updated `workflows/anatomical-activity-videos.md` with crop/width parameters and helper/entrypoint usage.

## [2026-09-18] improve | Enlarge CNS overview and capture sharper task footage

- The user found the version-3 overview too small. Added `overview_panel_fraction` (preset/default 0.65), reduced the HD simulation allocation to 40%, and tightened side cropping to `[0.30, 0.06, 0.70, 0.88]`.
- Added replay YAML capture overrides for camera reduction factor and source MP4 quality. The HD preset captures native 1600 × 900 (factor 1) at quality 9 and composes 1920 × 1080 output, preserving the 20/80 fps clocks and pinned source checkpoints/cases.
- Changed the header to "1,952 neurons / 33,720 connections" after verifying ordered body-ID and CSR-weight counts in the raw artifact/builder. 33,720 is not a neuron count or individual synaptic-contact count.
- Renderer version 4 and view-rectangle-aware raster caches prevent incompatible split reuse. Inspected actual-data layout and the first native-resolution robot crop. Eight focused tests passed, including capture override routing/source preservation and recording identity invalidation.
- Updated entrypoint/helper/configuration documentation. HD captures and CPU renders target the separate ignored `evals/connectome/anatomical_1952_7b_hd/`; final batch verification is recorded after completion.

## [2026-09-18] verify | Complete native-camera anatomical HD batch

- Completed six native 1600 × 900 task recordings and all 12 version-4 circuit/combined outputs at 1920 × 1080 / 80 fps in `evals/connectome/anatomical_1952_7b_hd/`.
- Fully decoded every output; source/output frame counts are 200→800 or 201→804 and per-case durations match at 10 or 10.05 seconds. The cached replay then regenerated the six-card task-paired gallery without simulation or encoding.
- Verified every source frame's colored robot/tool content stays inside the `[0.30, 0.06, 0.70, 0.88]` crop. Inspected encoded task previews and the six-case montage for robot, tool, table, overview size, labels and header semantics.
- Reverified 1,952 neurons, 33,720 graph connections, all-neuron robot-action readout, population masks and bilateral sensory/descending/motor anchors. `verification.json`, `recording_provenance.json`, per-case previews and `preview_montage.png` preserve the generated evidence.

## [2026-09-18] verify | Complete noaux dual-tolerance anatomical HD batch

- Added a YAML-owned normal-evaluation comparison that fixes the actual 7B noaux checkpoint, mean-action policy, three task cases, native 1600 × 900 capture, quality-9 source footage and version-4 1920 × 1080 / 80 fps circuit layout. It evaluates paper task progress at 0.02 m and checkpoint training tolerance at exactly 0.039858076721429825 m, resolved from `scalars/success_tolerance/frame` at frame 7,000,031,232.
- Completed six cases and 12 MP4s in `evals/connectome/anatomical_1952_7b_noaux_dual_tolerance_hd/`. Every MP4 fully decoded; output frame counts were exactly four times the 20 fps source counts, and crop-content checks passed on every source frame. The six-case montage was inspected for task framing, CNS overview, labels, threshold titles and activity bars.
- Paper tolerance averaged 27.6094% task progress; checkpoint tolerance averaged 77.3333%. The relaxed-threshold eraser episode reached success at 5.6 seconds, while the other cases ran for 10 or 10.05 seconds. The local gallery has six metric/tolerance-labeled cards and was regenerated without another simulator run.

## [2026-09-18] configure | Force fresh noaux video evaluation on CUDA:1

- Added `configs/connectome/evaluation/ppo_1952_7b_noaux_dual_tolerance_anatomical_hd_cuda1.yaml`, preserving the noaux checkpoint, both thresholds, native capture and circuit layout while setting `gpu_assignments: [1]` and a distinct `_cuda1` output directory.
- The distinct destination is intentional: the normal evaluator reuses complete matching cases and otherwise prints the aggregate JSON without launching Isaac Gym workers. The new config forces fresh simulator capture on physical GPU 1 while preserving the previously verified output directory.

## [2026-09-18] configure | Add blue-brush cases to noaux video generation

- Added `blue_brush/sweep_forward` and `blue_brush/sweep_right` to both noaux dual-tolerance evaluation YAMLs. Existing marker, eraser and spatula cases remain unchanged; rerunning reuses their complete artifacts and schedules only the new brush cases.

## [2026-09-18] configure | Launch CUDA-1 small-MLP all-neuron fly control

- Added `configs/connectome/suites/ppo_1952_4update_gaussian_lf_entropy1x_sigma3_all_neuron_readout_noaux_mlp128x32x32_100b.yaml` as the CUDA-1 counterpart to the live no-auxiliary all-neuron fly run.
- The suite preserves the same 1,952-neuron prepared graph, all-neuron actor readout, privileged critic, no-auxiliary loss setting, seed, environment/batch geometry, optimizer/objective, task overrides, checkpoint cadence, and 100B-frame budget. It selects only the pre-existing `128/32/32` sensory/descending/readout MLP train profile; output identity and physical GPU placement are intentionally distinct.

## [2026-09-18] implement | Add synchronized two-GPU small-MLP training

- Stopped the full-size CUDA-0 and small-MLP CUDA-1 trainer process groups while preserving their run artifacts; unrelated milestone watchers were left running.
- Extended `run_connectome_suite.py` with YAML-owned `training.distributed: true`: it launches one `torchrun` policy across all listed GPUs, exposes the complete device list, validates the frame cap against world size, and records all participating devices. Ordinary suite contracts remain single-GPU.
- Corrected the dormant multi-GPU runtime to select rank-local Isaac Gym and PPO devices and use NCCL through the launcher's rendezvous. Previously both ranks were hardwired to logical CUDA 0 and the suite always supplied `multi_gpu=false`.
- Added a two-rank smoke and a fresh 100B production suite using the no-auxiliary all-neuron `128/32/32` MLP profile. Production uses 15,360 environments and six 2,560-environment SAPG blocks per rank, 61,440-sample local actor/critic minibatches, and 203,450 epochs for 99,999,744,000 global frames.
- Added a milestone watcher that generates both fixed paper-tolerance and exact checkpoint-training-tolerance videos for the three standard deterministic cases every 250M global frames.
- The two-epoch smoke completed on both GPUs and reload-verified the two-rank terminal checkpoint with a finite 29-action deployment output.

## [2026-09-18] configure | Halve the DDP minibatch and expose it on TensorBoard 6008

- Stopped the 61,440-per-rank DDP pilot and its milestone watcher with artifacts preserved.
- Added a fresh 24,576-per-rank production contract. NCCL gradient averaging gives the original 49,152-sample nominal global batch; LF's seventh block produces 11 optimizer steps per mini-epoch, 22 per epoch, and 4,475,900 actor updates over 203,450 epochs.
- Added a matching watcher for paper-tolerance and exact checkpoint-training-tolerance videos. The new training output is rooted under `train_dir/connectome/adaptation_100b_gains_update_timing`, which the existing TensorBoard server on port 6008 already watches.

## [2026-09-18] correct | Restore five logical DDP updates per PPO epoch

- Stopped the 24,576-logical-minibatch DDP trainer and matching watcher at epoch 189 / 92,405,760 global frames, preserving their complete run and evaluation artifacts.
- Added a fresh YAML-owned replacement with 15,360 environments per rank, 49,152-sample logical actor/critic minibatches, and 24,576-sample physical microbatches. The unchanged two PPO mini-epochs now apply five synchronized actor steps and five separate critic steps per mini-epoch, or ten of each per training epoch.
- Focused configuration and gradient-accumulation checks passed seven tests. Both live ranks report `num_minibatches: 5`, logical optimizer batch 49,152, physical microbatch 24,576, and nominal accumulation two. Global frame counters advance by exactly 491,520 per epoch.
- Launched the trainer in tmux `connectome-ddp-small-mlp-logical49152` (coordinator PID 363903, torchrun PID 364173, rank PIDs 364247/364249) and the dual paper/checkpoint-tolerance watcher in `connectome-ddp-small-mlp-logical49152-eval` (PID 363908). TensorBoard port 6008 discovered the new summary path.

## [2026-09-18] relaunch | Return DDP to the original environment population

- Stopped the 15,360-environment-per-rank trainer and matching watcher at epoch 168 / 82,083,840 global frames, preserving their training and evaluation artifacts.
- Added and launched the replacement at the original 12,288 environments and six 2,048-environment SAPG blocks per rank. It keeps 49,152-sample logical actor/critic batches and the halved 24,576-sample physical per-rank microbatches, with two-way gradient accumulation before every synchronized step.
- Both ranks report four logical minibatches per PPO mini-epoch, logical optimizer batch 49,152, physical microbatch 24,576, and nominal accumulation two. Two unchanged PPO mini-epochs therefore give eight actor steps and eight separate critic steps per training epoch; the number of synchronized updates matches the original one-GPU schedule while each normal update averages 98,304 global samples.
- The 254,313-epoch contract reaches 99,999,940,608 global frames. The trainer is live in tmux `connectome-ddp-small-mlp-env12288` (coordinator PID 434253, torchrun PID 434536, rank PIDs 434643/434644); the paper/checkpoint-tolerance watcher is live in `connectome-ddp-small-mlp-env12288-eval` (PID 434259), and TensorBoard port 6008 discovered the replacement summary path.

## [2026-09-18] launch | Pair the matched LSTM and asymmetric fly

- Stopped only the two-rank DDP fly trainer and its matching watcher at epoch 753 / 295,698,432 global frames; their artifacts remain preserved. Unrelated historical milestone watchers were not signaled.
- Added one fresh YAML-owned pair: the 323-unit parameter-matched LSTM actor on physical CUDA 0 and the 1,952-neuron all-neuron `128/32/32` fly actor on physical CUDA 1. Both use the identical inherited `[1024, 1024, 512, 512]` privileged asymmetric critic, no actor-side auxiliary value loss, 12,288 environments, 49,152 actor/critic batches, LF/1.0, seed 42 and a fresh 100B-frame budget.
- Seven focused configuration and accumulation tests passed. Live process environments and NVIDIA UUIDs verify PID 671749 on CUDA 0 for the LSTM and PID 671748 on CUDA 1 for the fly; both report four logical minibatches per PPO mini-epoch and 196,608-frame epoch increments.
- The coordinator runs in tmux `connectome-asymmetric-matched-pair-100b` as PID 671476. The dual paper/checkpoint-tolerance watcher runs in `connectome-asymmetric-matched-pair-100b-eval` as PID 671483. Both event streams are linked into and discovered by TensorBoard port 6008.

## [2026-09-19] query | Verify small-MLP fly adaptive-KL scheduler

- Compared the current asymmetric `128/32/32` fly's saved resolved YAML with the specified earlier no-auxiliary all-neuron run. Both use the same adaptive/standard actor LR scheduler, initial LR `1e-4`, bounds `[1e-6, 1e-3]`, KL target `.004`, two mini-epochs, 49,152-sample minibatches and LF/1.0 population.
- Both independent single-GPU runs have four logical minibatches per mini-epoch and one scheduler decision after each mini-epoch. The controller multiplies LR by 1.5 below KL `.002`, divides by 1.5 above `.008` or on invalid KL, and otherwise leaves it unchanged.
- Recorded that identical scheduler semantics do not imply identical KL/LR trajectories because the smaller interface MLP changes the learned policy, gradients and collected rollouts. No process or training configuration was changed.

## [2026-09-19] diagnose | Explain growing fly KL without LR decay

- Read the current small-MLP fly's raw port-6008 scalars through frame 1,186,725,888. Median aggregate KL increased monotonically across six 200M-frame bands from `.00549` to `.04040`; this is not a smoothing artifact.
- Mini-epoch-0 median KL rose from `.00905` to `.07920`, while mini-epoch 1 stayed near `.0016`--`.0019`. In 120 of the latest 200 epochs, the first scheduler decision divided LR by 1.5 and the second multiplied it by 1.5, exactly canceling. The authoritative end-of-epoch LR was `.000197531` at both window endpoints.
- Traced `info/last_lr` to the final actor minibatch return: it precedes the mini-epoch-1 scheduler call and is not necessarily the epoch's final LR. The explicit per-mini-epoch tags show the real transitions. Dataset means/scales are refreshed after each minibatch, and LF KL includes relabeled cross-member samples, so the feedback is local rather than a cumulative trust-region bound. No process or configuration was changed.
