# Project Log

Append-only record of durable repository work.

Last updated: 2026-09-13

Related: [Index](index.md), [Overview](overview.md)

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

## [2026-09-14] query | Audit live compact-training status

Checked tmux, process ownership, both RTX 4090s, training logs, recovery and inference checkpoints, TensorBoard scalars, and milestone manifests. The 1,952-cell gains trainer failed at 2,275,344,384 logged frames because its learned coefficient-conditioned action standard deviation became negative; its last recovery checkpoint is frame 2,241,331,200 with 91,164 applied actor Adam steps. Nine milestones and 27 mean-action videos are complete.

The adapters-only process remains alive on GPU 1 and passed 3.32 billion logged frames, but actor loss and KL have been `NaN` since about 3.19 billion. Its frame-3,303,014,400 recovery checkpoint has 129,745 applied actor steps versus 134,400 scheduled calls, so simulator/critic activity is continuing without valid actor learning. Thirteen milestones and 39 mean-action videos are complete. GPU 0 is idle and no training/evaluation process was changed during the audit.

## [2026-09-14] experiment | Stop invalid adapters-only run

At the user's direction, sent interrupts to the exact `connectome-1952-adapters` trainer/coordinator and `connectome-1952-adapters-eval` watcher, then verified all three PIDs exited. The last log coordinate is epoch 17,066/frame 3,355,115,520. The last recovery checkpoint remains intact at epoch 17,000/frame 3,342,336,000 with 129,745 applied actor steps; 6,255 of 136,000 scheduled calls had been suppressed, and the applied count had not advanced since epoch 16,800. Preserved all TensorBoard events, checkpoints, milestone manifests, and 39 mean-action videos. A separate eligibility trainer that appeared on GPU 0 was not changed.
