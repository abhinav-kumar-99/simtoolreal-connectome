# End-to-end connectome and training audit — 2026-09-15

The strongest training policy still has pathological exploration, while cross-member KL, saturated circuit inputs, evaluation conditioning, and unvalidated functional circuit reduction complicate the learning problem.

Last updated: 2026-09-15

Related: [Training history](compact-1952-training.md), [Circuit selection](front-leg-pathway-coverage.md), [Small circuit](minimal-dexterity-circuit.md), [Actor](../concepts/connectome-actor.md), [Success metrics](../concepts/success-metrics.md), [Source provenance](../sources/summaries/malecns-front-leg-circuit.md)

## Decision

Retain the frozen 1,952-neuron MLP-interface model as the main full-task reference. Its clipped-Gaussian KL-0.004 run has the strongest observed training return and training goal attainment. Its exploration distribution is nevertheless deteriorating; numerical KL repair did not remove the unbounded raw-entropy incentive. The tanh distribution fixes that incentive, but its existing runs have not solved the task.

The next bounded diagnostic is [ppo_1952_tanh_onpolicy_noaux_1b.yaml](../../../configs/connectome/suites/ppo_1952_tanh_onpolicy_noaux_1b.yaml): fresh tanh policy, on-policy samples only, no auxiliary value loss through actor adapters, actor KL target 0.004 and LR ceiling 0.001. This is a proposed bundle of interventions, not a single-factor ablation or empirically established winner. It has passed configuration composition, not a simulator smoke or learning evaluation. No current training configuration or process was changed by this audit.

For the larger research direction, prioritize competent teacher → recurrent imitation → student-state correction → conservative RL. In parallel, test the fly circuit as a low-level feedback/coordination module with meaningful sensory and motor mappings. Current evidence does not establish that training arbitrary interfaces from task reward alone is the best way to exploit biological computation.

## Evidence scope and reproducibility

The TensorBoard snapshot is **2026-09-15 11:12:01 UTC / 07:12 EDT**. All 73 local event files were read across 71 physical run directories, resolving symlink aliases. Seventeen runs have reward histories past 100M frames, 27 have shorter reward histories, and 27 have no completed-episode reward scalar. The latter two groups include smokes, integration failures, short pilots and interrupted/replacement starts; they are not evidence of long-run learning quality. All 86 local evaluation summary files were also read, containing 88 policy-summary cells.

Files were read up to captured byte sizes and incomplete trailing records were excluded. Duplicate tag/step records use the newest wall time. The lightweight TFRecord/protobuf scalar parser agreed exactly with TensorBoard's generated protobuf parser on 991 checked scalar records. Tables use trailing 100 logged reward points, with run-specific end coordinates; the comparison plot uses trailing **25M-frame** means. These are descriptive comparisons, predominantly seed 42, not replicated causal estimates. For PPO the success tag suffixes are frame-coordinate aliases; eligibility `/time` uses a real time axis and was not treated as a frame coordinate.

Temporary reproducibility artifacts are under `/tmp/connectome-tb-audit-ra21Aa/`:

- `results.json`: full run paths, resolved configuration, tag statistics, non-finite counts/onsets, milestones, curves, and every evaluation summary.
- `graphs.json`: current artifact hashes, directed routes, interface counts, incoming-contact retention, and local linear dynamics.
- `checkpoints.json`: CPU probes of saved rollout states and exact Gaussian probabilities. These are not newly collected robot trajectories.
- `curves.png`: rendered and visually inspected four-panel reward, success and entropy comparison. Raw Gaussian and tanh-action entropy are deliberately in separate panels.
- `audit.yaml`: repository root, output locations, milestone coordinates, tail length, checkpoint run indices and 128 samples per exploration block.

The entry-point helpers below all take `--config audit.yaml`; no audit settings require additional CLI flags. `audit_events.py` inventories events/evaluations; `audit_graphs.py` checks routes/coverage and the zero-state linearization; `audit_checkpoints.py` reconstructs saved networks on CPU, measures drives/scales and performs no-update relabel/gradient probes; `verify_sources.py` rehashes and reaggregates the 6.4 GB raw partner file. `plot_audit.py` reads the event inventory and produces the plot. They are temporary diagnostic scripts, not changes to the training implementation. Run from the repository root:

```bash
.venv/bin/python /tmp/connectome-tb-audit-ra21Aa/audit_events.py --config /tmp/connectome-tb-audit-ra21Aa/audit.yaml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python /tmp/connectome-tb-audit-ra21Aa/audit_graphs.py --config /tmp/connectome-tb-audit-ra21Aa/audit.yaml
PYTHONPATH="$PWD/rl_games:$PWD" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python /tmp/connectome-tb-audit-ra21Aa/audit_checkpoints.py --config /tmp/connectome-tb-audit-ra21Aa/audit.yaml
PYTHONPATH="$PWD/rl_games:$PWD" .venv/bin/python /tmp/connectome-tb-audit-ra21Aa/verify_sources.py --config /tmp/connectome-tb-audit-ra21Aa/audit.yaml
.venv/bin/python /tmp/connectome-tb-audit-ra21Aa/plot_audit.py --config /tmp/connectome-tb-audit-ra21Aa/audit.yaml
```

Re-running refreshes the temporary outputs from live files; it does not reproduce the identical time snapshot. Checkpoint index selection refers to the inventory from the first command. These `/tmp` artifacts may be removed by ordinary system cleanup; the durable conclusions and source seams are recorded here.

## What all training histories support

`Return` below is `rewards/step`, ordinarily the zero-direct-entropy-bonus SAPG member (block 5), not an evaluation success rate. End frames are reward coordinates, sometimes one phase behind the checkpoint or optimizer tags.

| Run family | Last recorded frames, B | Return, last 100 | Interpretation |
| --- | ---: | ---: | --- |
| 4,310 gains, new update timing | 0.372 | 72.09 | Shorter exposure; no isolated timing conclusion |
| 4,310 gains, old update timing | 0.929 | 298.38 | Better training trajectory, already extreme raw entropy |
| 1,952 linear + gains, before KL repair | 2.275 | 353.44 | KL invalid from 1.765B; later actor loss invalid and sampling failure |
| 1,952 linear frozen, before KL repair | 3.355 | 335.48 | KL invalid from 2.670B; actor loss invalid from 3.191B |
| 1,952 local eligibility + gains | 0.195 | 25.09 | Stopped; no useful goal-learning evidence |
| 1,952 local eligibility frozen | 0.219 | 25.81 | Stopped; no useful goal-learning evidence |
| 1,952 linear + gains, repaired KL .004 | 0.711 | 235.68 | Finite in recorded history; shorter than main MLP runs |
| 1,952 linear frozen, repaired KL .004 | 0.575 | 141.65 | Slower observed learning than MLP interfaces |
| **1,952 MLP clipped, KL .004** | **4.436** | **560.42** | Strongest training learning, continued raw-sigma growth |
| 1,952 MLP clipped, KL .016 | 1.226 | 344.75 | Stopped after far faster raw-sigma growth |
| 1,952 MLP tanh, KL .016 | 2.378 | 386.27 | Stable moderate scales, mostly lifting plateau |
| 4,310 release-timing frozen | 1.000 | 79.44 | Completed bounded historical run |
| 4,310 release-timing gains | 1.000 | 102.02 | Completed bounded historical run |
| 4,310 early exploratory frozen | 0.141 | 61.90 | Different force/exploration contract |
| 4,310 early exploratory gains | 0.144 | 69.03 | Different force/exploration contract |
| 262 MLP clipped, KL .004, fresh | 0.430 | 74.06 | Stopped well before a billion; poor early learning |
| 262 MLP tanh, KL .016, fresh | 4.593 | 378.30 | Much longer exposure, similar lifting plateau |

At matched **1B** exposure the main MLP returns are approximately 323.1 for 1,952 clipped, 360.9 for 1,952 tanh, and 289.2 for 262 tanh. At 500M they are 274.3, 305.7 and 192.4. Thus tanh did not simply prevent early learning, and 262 did not improve sample efficiency in this comparison. The later 1,952 clipped run learned more training goals. Architecture, KL target and interruption/resume history are not all matched, so this is not an isolated action-distribution experiment.

The five approximately 1M adaptation pilots have almost identical cold-start reward, around 59.71 over only four logged points. They do not establish superiority of edgewise, gain/dynamics, low-rank or frozen adaptation. Short capacity/smoke runs without rewards do not favor an RL method. No substantial matched direct LSTM/MLP, rewired-connectome or random-graph training control exists in this inventory.

### Resume provenance

The immutable handoff decision at `train_dir/connectome/adaptation_100b_gains_update_timing/resume_handoff_controller/decision.json` used the original 100-point window ending at frame 2,378,366,976: clipped **449.79814**, tanh **386.26710**. It correctly selected clipped. The clipped recovery checkpoint was earlier, at 2,359,296,000, so resumed scalars overlap and replace the visible tail. The newest-event version of that window is about 417.17; that does not change the historical decision. Do not merge original and replayed samples as independent evidence.

At process inspection the 1,952 clipped continuation was the sole actual `isaacgymenvs.train` process (PID 3815434), on the existing GPU-0 contract. The 262 tanh stream last recorded at approximately 10:54 UTC, with no matching live trainer found. Its terminal cause was not established; “100b” directory names are budget labels, not completion evidence. Watchers/coordinators can remain alive without a trainer.

## Finding 1: entropy is partly solved, task learning is not

The old model stores a trainable `6 x 29` **log-standard-deviation** table. `fixed_sigma: coef_cond` means state-independent scales for each coefficient ID, not frozen variance. Raw Gaussian entropy grows without bound with log standard deviation, while physics sees hard-clipped normalized commands. The clipping is effective, and the stored latent sample/log likelihood is internally consistent; the problem is the exploration objective's mismatch to executed diversity.

The repaired-KL .004 MLP run's block-0 raw entropy reached 172.93 nats in the event snapshot, corresponding to geometric-mean sigma about 94.1. A slightly later rolling checkpoint at **4,482,662,400** frames gives geometric-mean sigma **116.64**, maximum coordinate sigma **400.47**, and an exact expected **99.19%** of block-0 action coordinates beyond the clipping bounds on sampled saved states. This is a probability over Gaussian samples, averaged over coordinates/states, not a measured fraction of whole actions from a new rollout. Block 0's mean training goals reached is zero.

The two tanh checkpoints have moderate geometric-mean latent sigmas across all blocks, approximately **0.55–0.75**. Their block-0 executed-action entropies are around 7–8 nats, not comparable numerically to raw Gaussian entropy. For a 29-dimensional action density supported on `[-1,1]^29`, maximum differential entropy is `29 * ln(2) = 20.10` nats. Tanh with corrected log likelihood/entropy removes the incentive to inflate latent variance indefinitely merely to increase raw entropy. It is not an absolute numerical bound on the latent parameter or a guarantee of task performance. Do not convert tanh entropy to sigma with the Gaussian formula.

The training plateau is substantive: late all-environment mean goals reached are **0.2331** for 1,952 clipped, **0.00592** for 1,952 tanh and **0.00553** for 262 tanh. These are goals per most recently completed episode, not episode-success probabilities. All retain initial tolerance **0.075**, corresponding to effective keypoint distance 0.1125 m; none reaches the curriculum's mean-three-goals gate. The tanh return near 386 consists mainly of lift bonus (~260), lifting (~76) and fingertip reward (~63), with goal bonus only ~5. The clipped run now has substantial goal bonus (~236), as well as lift bonus (~205). It is inaccurate to dismiss all improvement as lifting, but lifting return alone is insufficient evidence for tool control.

### Clarification: output squashing and internal-neuron saturation are different

The 94–99% saturation figures below concern the **internal neuron activation**, not the action-output tanh. The saved 1,952 tanh checkpoint gives expected `abs(action)>0.95` fractions of 9.2–13.4% across its six blocks. These are conditional Gaussian probability calculations on saved states, not a new rollout. Tanh output can still compress useful control near the bounds and is not proven to give the best reward; it was recommended because transformed entropy describes the bounded distribution, unlike the original unbounded Gaussian entropy paired with hard-clipped commands.

PPO does not differentiate through the simulator or through a freshly sampled action for its likelihood-ratio objective. Here the stored pre-tanh rollout sample is fixed, and the transform Jacobian in its log probability is independent of current policy parameters (`models.py:371–406`). Thus output tanh does not automatically multiply the PPO score gradient by a vanishing tanh derivative. In contrast, the internal recurrent tanh is directly in the adapter-to-policy-parameter derivative path. Output compression can nevertheless make different latent commands behaviorally indistinguishable, weakening the reward signal.

The recommendation is a distribution consistent with actuator bounds, not tanh at every layer. A rescaled [Beta policy](https://proceedings.mlr.press/v70/chou17a.html) is an alternative bounded family; it is not implemented or validated here and does not repair internal saturation. Calibrating sensory/descending drive scale and checking goal sensitivity remain separate requirements. The proposed 1B suite isolates training-objective issues only; it must not be described as an implemented input-saturation or neural-timing fix.

### Proposed Beta-policy parameters

A first Beta-policy experiment could initialize every action dimension with alpha=beta=2, then **learn both as state-conditioned outputs**, not keep two fixed global constants. For each of 29 actions, sample x_i ~ Beta(alpha_i, beta_i), then send a_i=2*x_i-1. The rescaled mean is (alpha-beta)/(alpha+beta), and variance is 4*alpha*beta/((alpha+beta)^2*(alpha+beta+1)). Initial Beta(2,2) therefore gives action mean zero and standard deviation sqrt(0.2), approximately 0.447. This is a proposed initialization, not an empirically selected setting for this task; Beta(1,1) would instead be uniform with standard deviation approximately 0.577.

Following the parameterization in [Chou et al.](https://proceedings.mlr.press/v70/chou17a/chou17a.pdf), two motor-readout heads can output alpha=1+softplus(u) and beta=1+softplus(v). This constrains shapes above one, favoring an interior mode and avoiding singular endpoint densities; it is a deliberate family restriction, not a mathematical requirement of Beta distributions. Small final weights and bias inverse_softplus(1)=log(exp(1)-1), approximately 0.5413, initialize near (2,2). Both heads and upstream trainable interfaces would receive PPO likelihood/entropy gradients; the frozen recurrent graph can remain frozen. SAPG conditioning may feed these heads through the circuit, but shape parameters are distinct from the entropy-loss coefficient. Beta likelihood, analytic entropy and Beta KL must replace the Gaussian calculations consistently, including the affine log-density correction for a=2*x-1. No Beta implementation or YAML profile was added.

The ratio alpha/(alpha+beta) controls the mean, while their sum controls concentration when that ratio is held fixed. Positive shapes below one would allow endpoint-peaked/U-shaped policies if later justified by control requirements. Even with bounded actions, learned concentration can become excessive and exploration can collapse; Beta is not an automatic optimizer-stability or internal-neuron-saturation fix.

## Finding 2: scheduler KL mixes different policies, even with no update

This **weakens the earlier interpretation that large first-mini-epoch KL directly measures aggressive optimizer movement**. The relevant seams are:

1. [`augment_batch_for_mixed_expl`](../../../rl_games/rl_games/common/a2c_common.py) changes the policy coefficient in copied observations (lines 1082–1090), but copies behavior `mu`, `sigma`, `neglogpacs` unchanged (1108–1111). Leader/follower mode retains a relabeled source block for ID 0 alongside the six original blocks.
2. It copies the source member's recurrent states too (1101–1105); these were not generated by the new coefficient-conditioned history.
3. [`calc_gradients`](../../../rl_games/rl_games/algos_torch/a2c_continuous.py), lines 244–248, computes KL against the stored Gaussian on those samples without excluding relabeled observations.
4. `dataset.update_mu_sigma` changes the KL reference after each minibatch (`a2c_common.py:1588`). Mini-epoch two therefore uses a different reference from mini-epoch one, while the behavior action log probabilities used for importance ratios remain fixed.

At the clipped checkpoint, holding weights, observation features and hidden state fixed and changing only the coefficient from each source block to ID 0 gives no-update Gaussian KL values **[101.862, 6.615, 0.00953, 6.111, 30.201, 0]**. Current-step means do not change because of the network's delayed input-to-motor path; this probe isolates the coefficient-dependent scale difference. With a single imported block in seven blocks of augmented data, this difference alone can dominate the scheduler. The tanh checkpoints show the same structural effect with smaller scales: maximum no-update source-to-leader KL 0.862 for 1,952 and 2.259 for 262.

The recorded clipped tail has mini-epoch KL means **4.3163** then **0.001597**, causing alternating LR reduction/increase. It cannot be interpreted solely as an optimizer overshoot followed by recovery. Stable float64 KL arithmetic correctly evaluates the supplied distributions; it does not make those distributions the right same-conditioned old/new comparison.

The existing configuration-only diagnostic is `use_others_experience: none`. A future implementation should separate **behavior-policy likelihoods for importance weighting** from an **immutable same-conditioned rollout reference for update KL**, measure on-policy scheduler KL separately, and handle relabeled recurrent histories with principled reroll/burn-in if cross-member reuse is retained. Do not “fix” this by replacing behavior likelihood denominators with target-policy likelihoods. An explicit update-level KL stop/backoff rule would also be distinct from the current LR feedback, but is not implemented here.

## Finding 3: the tanh circuits' input neurons are nearly saturated

CPU reconstruction used the saved running observation normalization, real sparse operator, and 128 saved environments from each of six blocks. Checkpoint frames differ from the event snapshot because rolling saves occur at different times:

| Policy checkpoint | Frames, B | Sensory preactivation `abs(z)>3` | Descending preactivation `abs(z)>3` | Sensory drive RMS | Descending drive RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,952 clipped | 4.483 | 47.46% | 60.48% | 4.86 | 6.43 |
| 1,952 tanh | 2.379 | **94.30%** | **99.66%** | 52.12 | 45.25 |
| 262 tanh | 4.561 | **96.90%** | **99.09%** | 140.55 | 18.13 |

At `abs(z)=3`, `tanh'(z)` is already below 0.01. The MLP input interfaces can scale their outputs without any corresponding input-drive control. Consequently, a frozen recurrent matrix does not preserve its operating regime: much of the sensory/command interface is being driven into nearly binary responses with weak local sensitivity. Motor states are not themselves saturating in these samples, so this is distinct from both output-mean saturation and output variance blowup.

A 16-step held-input counterfactual also produced much smaller goal sensitivity for the tanh checkpoints, especially 262, than for the clipped checkpoint. This is a local squared-mean-action gradient probe on saved block-0 states, not a general observability test or proof that goal information is completely absent. The one-step zero input gradient, separately, is structural and exact.

There is another relevant actor objective: `use_experimental_cv` defaults true (`a2c_continuous.py:74,118`). Even with a separate privileged central critic supplying rollout values/advantages, the actor's own value head reads every circuit state and trains through its input interfaces. With `critic_coef: 4`, the minimized actor objective includes `2 * actor_value_loss` (`a2c_continuous.py:191–222`). This is verified objective routing; **whether it causes saturation or harms policy learning is untested**. Scalar loss magnitudes do not establish relative gradient influence. The existing `++train.params.config.use_experimental_cv: false` override removes this auxiliary actor-value loss while retaining the central critic.

Recommended follow-up instrumentation: unscaled per-loss adapter-gradient norms/cosines, sensory/descending drive and saturation quantiles, goal perturbation response over time, motor-state variation and effective rank. If saturation remains after the diagnostic, calibrate/normalize input drives or pretrain interfaces against functional responses before further long RL runs. Such drive controls are proposals, not existing configuration keys.

## Finding 4: training selection and evaluation use different SAPG members

Ordinary training reward and block-filtered successes describe **block 5 / coefficient ID 0**, with no direct entropy bonus. [`deployment/rl_player.py:97`](../../../deployment/rl_player.py) hardcodes **ID 50 / block 0**, the highest-entropy member. Deterministic evaluation removes action sampling but still conditions the recurrent policy on this different ID; taking the mean does not make all members the same controller.

The milestone evidence is almost flat. The 1,952 clipped run has 17 milestone evaluations through 4.25B; all but the 2.75B milestone have mean Task Progress **0.74074%**. The 2.75B exception is **1.48148%**. The 1,952 tanh run has nine through 2.25B and the 262 tanh run has 18 through 4.5B, all **0.74074%**. The usual three-case result is marker **0/25**, eraser **0/33**, spatula **1/45** waypoints; the clipped exception reaches **2/45** spatula waypoints. These are only one episode per case, not precise generalization estimates.

The mismatch could hide a better ID-0 policy, but current files do not demonstrate that it does. First evaluate all six IDs and a verified conventional teacher on the same validation object/trajectory/seed cohort, under both training-like and stricter evaluation tolerance where useful for diagnosis. Select an ID on validation and use that same explicit ID in both deployment and final evaluation; reserve a disjoint cohort for reporting. Making that shared ID configurable is a required future code change, not silently done by the new training YAML.

## Raw MaleCNS → initial network: what is correct and what is assumed

### Verified data engineering

The original 4,310 graph comes from the pinned upstream signed adjacency and neuron list. The compact graphs instead come from an official MaleCNS annotation/partner audit. This audit rehashed all source files, reconstructed the 4,778 candidate annotation/group table, reread **4,759 batches** of the approximately **6.4 GB** partner Feather, and exactly reproduced **1,110,601** cached incoming directed-pair counts.

The partner filter uses both pre/post confidence >=0.5 and an explicit primary-postsynaptic VNC ROI list. Counts are summed across batches and ROIs **before** applying the >=5-contact edge cutoff (`scripts/audit_front_leg_coverage.py:53–93`). The cutoff is a modeling/quality filter, not a calibrated guarantee that every retained edge is real or every omitted edge is noise.

Runtime CSR rows are postsynaptic destinations and columns presynaptic sources. The upstream source-row adjacency is correctly transposed (`simtoolreal_shared/connectome_data.py:110–116`). Signed weights use the presynaptic transmitter convention. Current NPZ hashes and body-ID identities match their pinned contracts. No evidence of a reversed recurrent matrix, wrong compact body-ID ordering, duplicate pair aggregation, or failed source provenance was found.

The candidates are not a search of the entire CNS: the pool is the old 4,310 cells plus specified front-leg sensory, motor, descending and hemilineage groups, giving 4,778. Compact paths are restricted to edges with both endpoints in that pool even though the incoming-edge cache retains other presynaptic sources for coverage analysis.

### Reduction does not preserve a demonstrated computation

The 1,952 selection starts with 1,285 seed neurons; multi-source unweighted shortest paths give 1,596; adding every sensory → intermediary → front-motor two-edge motif adds 356. Stable ordering/body-ID hashes make ties reproducible. Shortest paths are not the union of all parallel routes, recurrent loops, or closed-loop behaviors.

The 262 selection starts from left front-leg proprioceptors and distal-leg motor types, takes their two-edge intermediary union, then removes ten cells lacking an input-to-motor route. Its descending ports are only the annotated descending cells incidentally retained by that operation.

| Structural property | 1,952 graph | 262 graph |
| --- | ---: | ---: |
| Directed edges | 33,720 | 3,194 |
| Sensory / descending / motor ports | 384 / 157 / 135 | 48 / 2 / 31 |
| Isolated cells | 44 | 0 |
| Cells on an input-to-motor route | 1,885 | 262 |
| Sensory ports with no route to any motor | 33 | 0 |
| Motor ports unreachable from any input | 5 | 0 |
| Retained incoming contacts, all selected cells | **30.06%** | **15.45%** |
| Retained incoming contacts to motor cells | **42.30%** | **25.38%** |

Retention denominators include all sources into selected targets within the same official confidence/ROI scope, before the edge cutoff. These are synaptic-contact fractions, **not percentages of computation retained**. Sensory axonal incoming contacts are not external receptor signals. The 4,310 upstream extraction has a different count/ROI contract, so dividing its raw weights by the compact reference denominator is invalid and deliberately omitted.

Five unreachable motor states remain zero under the current zero reset and frozen zero recurrent bias, leaving five unused readout inputs. Isolates are modest waste, not a sufficient explanation of the learning failures. More importantly, both compact circuits omit much of their selected neurons' incoming network context. Restoring weak edges indiscriminately would not resolve the functional uncertainty; evaluate missing strong feedback/coordination partners and reconstruction confidence first.

The two 262 descending cells are **DNge061**, body IDs **220203** and **909558**, both left-sided. The 12 goal features plus 32-dimensional SAPG embedding are projected to **two scalar command drives**. This restricts instantaneous goal injection to rank at most two, although recurrence can encode richer histories. There are no dedicated tactile input ports, and the motor population describes one distal leg, not an arm and five-finger hand. This is a defensible **local reflex-module hypothesis**, not a validated minimal whole-task controller.

### Signed counts are not a biophysical model

The convention is ACh positive; GABA, glutamate, unclear and missing negative. The 1,952 graph contains 57 unclear/missing cells, accounting for 189 outgoing edges and **0.275%** of retained synaptic contacts; 262 has three such cells, 13 edges and **0.114%** of contacts. Their signs remain uncertain, but this small fraction is not evidence that it dominates the observed optimization pathology. Glutamatergic sign and target receptor effects require physiological care.

The current artifact retains anatomy-derived IDs, topology, contact-derived weights and transmitter-derived signs. It does not contain measured rates, synaptic efficacy, membrane constants, receptor identities, conductance dynamics, gap junctions, plasticity, neuromodulation or conduction delays. See the [official MaleCNS release inventory](https://male-cns.janelia.org/download/) and [local provenance summary](../sources/summaries/malecns-front-leg-circuit.md).

For the current frozen profile, the implemented recurrence is

```text
h_next = 0.5 h + 0.5 tanh(0.9 W h + sensory_drive + descending_drive)
action_mean = MLP(h_next[motor_ports])
```

`W` is globally normalized to spectral radius one. State is one signed scalar per neuron, initialized at zero, not a measured firing-rate vector. About half the sampled clipped hidden states are negative. This can be interpreted as centered neural features, but then baseline rates/thresholds must be specified before claiming physiological inhibition: a negative source state through a negative edge gives a positive contribution.

The [upstream Pugliese simulation code](https://raw.githubusercontent.com/smpuglie/Pugliese_2026/faee4b06869855ae0164cbf217fb6ec28ef3521b/src/simulation/vnc_sim.py) instead includes rectified nonnegative rate dynamics, thresholds, rate limits, time constants and continuous numerical integration, with neuron parameters supplied separately. Our signed synchronous tanh recurrence is not a reproduction of that executable model. Borrowing its adjacency does not automatically transfer its demonstrated pattern-generation computation.

The zero-state linearization is `J = 0.5 I + 0.45 W`. For 1,952, spectral radius is **0.8555** but operator norm is **1.7673**; for 262 these are **0.7587** and **1.4915**. These imply asymptotically decaying local perturbations at zero, but allow transient amplification and do not prove global contraction of the driven nonlinear system. Global spectral-radius normalization neither calibrates every neuron nor preserves a specific biological operating point.

### Neural timing and body interface

All sensory/descending ports are disjoint from motor ports. Each step reads only the **old** hidden vector for recurrent propagation and injects the current observation into input ports. Therefore the current observation cannot affect the current action mean: the first direct sensory-to-motor response appears on the following control step. The autograd one-step input/goal gradient was exactly zero in all three checkpoint probes, consistent with the code rather than a kernel failure.

At 60 Hz one synaptic hop costs a control interval. Seventy of 135 motor ports in the 1,952 graph require at least two hops from any sensory input; seven need three; five are unreachable. In 262, 17 of 31 motor ports require three hops from the two descending inputs and three require four. Observation/action delays add to this artificial neural propagation delay. BPTT spans 16 control steps (~0.267 s); it is not 16 internal neural substeps within one action decision.

This makes neuron clock, leak and input scaling important design choices. A future multi-substep implementation should calibrate leak against neural `dt`, not repeat the existing half-leak update arbitrarily. Longer sequence/burn-in can be tested afterward; this audit does not prove that sequence length 16 is always insufficient.

The concrete proposed timing experiment is **K = 1, 4 and 8 internal neural updates per 60 Hz control decision**, holding the current encoded observation/goal drive fixed through those updates and reading motor output only afterward. Compute the input adapters once per decision, preserve the recurrent state across decisions, reset once at episode boundaries, and differentiate through all internal updates during PPO training. Training, rollout and deployment must share the same YAML-owned K; there is no such implemented knob yet. With the present simultaneous-injection convention, an input-to-motor route with L edges first receives current input at inner update L+1. Four updates cover up to three edges; eight cover up to seven. They remove the structural whole-control-step delay for those routes, not all neural lag or saturation.

For the current leak alpha=0.5, choose the substep leak `alpha_K = 1 - (1 - alpha) ** (1/K)`: approximately 0.15910 for K=4 and 0.08300 for K=8. This preserves leak-only retention `(1-alpha_K)^K = 0.5` over one control interval, rather than accelerating passive decay to `0.5^K`. Equivalently, neural `dt` becomes 4.17 ms or 2.08 ms while the control interval stays 16.67 ms. This does not preserve the exact nonlinear recurrent transition or all trained behavior; K changes the model's numerical dynamics and must be treated as an architecture/integration experiment, not a transparent hot-swap of a trained checkpoint.

Validate current-observation and goal perturbations against current motor output, reset/sequence parity and backward gradients, then compare response latency, saturation and actual end-to-end throughput. More inner steps increase recurrent computation and training graph depth even when adapters are cached; lower leak cannot desaturate a large steady input by itself. An input-first update is a cheaper alternative for direct sensory-to-motor paths but still leaves longer routes delayed. A direct observation-to-action bypass is a useful control, not the first fix if preserving circuit-mediated action computation is the aim. Increasing PPO sequence length alone cannot change this within-decision dependency. Deliberate observation/action delay randomization is a separate deployment assumption: disable it only in matched diagnostic tests when isolating neural lag, not silently in the production task.

The robot sensory vector contains no new tactile information merely because tactile cells exist in the graph. Dense learned projections do not enforce the tuning or feature semantics of individual fly receptor classes. Position/movement and vibration sensing have distinct fly pathways, as shown by [Lee et al., 2025](https://www.nature.com/articles/s41467-025-59302-3). A meaningful retargeting should distinguish joint-angle/velocity, load and contact sensing; it should not treat all sensory cell labels as interchangeable. Required robot channels must be available under deployment sensing, or explicitly estimated, rather than added as hidden privileged actor information.

## Implementation and efficiency checks

Thirty-three CPU tests passed, with 59 deselected, across graph preparation, compact selection, front-leg auditing and connectome-network tests. Another 21 configuration tests passed, and explicit composition assertions verified every proposed training override and the 999,948,288-frame geometry without launching a simulator. The selected coverage includes population/CSR contracts, recurrent sequence/reset behavior and bounded-distribution checks. GPU custom-backend tests were deliberately excluded from this audit to avoid consuming the active training device. These tests support implementation consistency, not biological correctness or learned competence.

There are avoidable-looking telemetry costs: `a2c_common.py:403` concatenates full actor gradients before AMP unscaling, and `a2c_continuous.py:278–279` transfers full gradient vectors plus a zero off-policy vector to CPU even without `LOG_OFF_POLICY_GRADS`. Such saved “gradient norms” are not unscaled optimizer-gradient diagnostics, and off-policy zeros when detailed logging is disabled do not demonstrate zero off-policy influence. Full per-step recurrent-state buffers are also retained for SAPG reuse although the central critic is feedforward. Make heavy telemetry opt-in and avoid unused state buffers when on-policy-only operation is selected, after profiling.

No isolated end-to-end speedup is claimed: runs have different concurrent GPU use, simulator behavior and evaluation activity. Fewer neurons reduce recurrent memory/compute, but do not by themselves establish better useful-task throughput. Frozen input-to-output recurrence still requires backpropagation through time when learning the input adapters; freezing `W` does not eliminate that derivative path.

A secondary configuration hazard is `connectome_network_builder.py:324–326`: non-legacy adaptation derives gain bounds from `sqrt(adaptation.edge_scale_bounds)`, superseding `dynamics.gain_bounds`. Current profiles agree, so this is not evidence of the present failure, but changing only the latter key would not produce the intended gains. No runtime kernel or optimizer code was edited.

## Better ways to use the fly computation

### First choice for obtaining a useful full-task policy: imitation, then RL

1. Verify a conventional teacher under the exact robot observation/action and evaluation contracts. A published downloadable policy is a candidate, not a locally verified working teacher; no compatible teacher was launched or downloaded in this audit.
2. Collect successful goal-dependent sequences, including recovery and contact transitions. Train the connectome interfaces with dense action targets, correct resets and recurrent burn-in. Compare executed bounded actions/distributions consistently; do not force a clipped teacher's arbitrary latent variance into a tanh student.
3. Roll out the student, query the teacher on student-visited states and aggregate corrections. This addresses closed-loop distribution shift rather than assuming low offline imitation error is sufficient; see [DAgger](https://proceedings.mlr.press/v15/ross11a.html).
4. Fine-tune with conservative bounded-action RL while monitoring held-out progress and internal signal sensitivity.

This directly tests whether the circuit can represent a competent controller before asking sparse policy gradients to discover it. The [FlyGM preprint, v3](https://arxiv.org/html/2602.17997v3), is a relevant precedent: its actual training pipeline uses expert-MLP imitation before PPO, despite an RL-focused abstract. It also uses vector-valued neuron states and learned cell-conditioned processing in a simulated fly, not our frozen scalar MaleCNS model or a robot hand. Its evidence motivates an experiment; it does not establish transfer to our task or validate every biological assumption in that paper.

### Best route for claiming transfer of a fly computation: a functional feedback module

Choose a specific computation first: posture stabilization under load, reciprocal flexor/extensor coordination, contact-triggered transitions, or bilateral coordination. Reproduce its input-response and perturbation behavior in a small native-like benchmark with physiological tuning/operating-point assumptions documented. Retain identified feedback/interneuron loops; reduce only while preserving those measured responses. Then retarget meaningful sensory features and decode antagonistic motor pools into robot joint synergies or posture/impedance corrections.

A higher-level teacher, planner or policy can supply low-dimensional descending commands; the fly-derived module handles local feedback, with a bounded learned residual if needed. Reusing a leg module across fingers would be an explicitly engineered motif replication, not the original unmodified fly circuit. Contact sensing or inferred load must obey real deployment constraints. An oscillator is useful for rhythmic motions, but constant oscillation is not automatically helpful for holding a tool. Which module helps remains an empirical question.

The 262 graph belongs in this module-level comparison. For a whole-hand actor, select diverse command channels, appropriate sensory classes and recurrent/coordination partners using the full reference and functional tests, rather than immediately choosing another neuron-count target. The 1,952 graph is the stronger current reference, not a proven complete circuit.

### Cheap representation test: a fixed reservoir and readout

Freeze both a calibrated input encoding and the recurrent circuit; generate features on shared teacher trajectories; fit a regularized linear readout to actions or state/goal-history targets. Compare the biological, degree-preserving rewired and random graphs with matched ports, scaling and data, plus direct MLP/recurrent controls. This cheaply distinguishes useful memory/features from the difficulty of learning the input embedding. A linear-readout failure does not prove all nonlinear adapters will fail, and success must still be tested closed loop.

Full nonlinear adapter training is no longer a cheap fixed-reservoir readout problem: it again differentiates through the entire recurrent path. Predictive representation training from offline trajectories is another possibility, but should be tested on task-relevant goal/contact information rather than arbitrary reconstruction accuracy.

### Lower priority alternatives

Changing to recurrent off-policy actor-critic learning could improve sample reuse, but introduces replay-state reconstruction and value-extrapolation concerns; it does not repair saturated inputs or the body/circuit mismatch. Current local-eligibility runs provide no empirical reason to promote them above PPO, and cheaper updates did not yield useful early reward. Evolutionary search is more plausible for a handful of low-dimensional module gains/couplings than for all dense interface parameters. These alternatives are untested recommendations, not completed implementations.

## Next-run contract and decision gates

Run the newly added suite only when intentionally starting this fresh diagnostic:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_1952_tanh_onpolicy_noaux_1b.yaml
```

`run_connectome_suite.py` is the existing actual entrypoint: it prepares/verifies the pinned graph, composes the task/profile, launches training, and records the resolved configuration and checkpoint verification. This audit adds only the separate YAML, not a new trainer. Its important settings are:

| Setting | Proposed value | Purpose |
| --- | --- | --- |
| Graph/profile | 1,952 frozen core, MLP interfaces, tanh policy | Stronger current graph; bounded-action entropy |
| `use_others_experience` | `none` | Remove cross-member relabeling and its KL/history confound |
| `use_experimental_cv` | `false` | Remove extra actor-value objective; keep central critic |
| Actor LR / maximum LR | `0.0001` / `0.001` | Conservative initial step and ceiling, not a hard KL cap |
| Actor KL target | `0.004` | Conservative scheduler feedback |
| SAPG entropy scale | `0.005`, six original coefficients | Retain exploration population and proper transformed entropy |
| Environments / block size | `12288` / `2048` | Preserve original geometry |
| Horizon / sequence / mini-epochs | `16` / `16` / `2` | Keep temporal/update contract for this diagnostic |
| Actor and central-critic batches | `49152` each | Eight on-policy actor optimizer calls per phase |
| Fresh seed / physical GPU | `42` / `1` | No checkpoint inheritance; separate from current GPU-0 job |
| Epochs / cap | `5086` / `1,000,000,000` frames | At most 999,948,288 configured frames |
| Inference / recovery saves | every 250M frames / every 200 epochs | Bounded review points and recovery |

The released reward, perturbations, scale noise and initial tolerance are unchanged. No latent log-standard-deviation clamp is introduced. `on_existing: fail` protects an existing output directory. GPU assignment is a launch choice, not a claim the device will be idle later. No milestone video watcher is auto-started by this train-only suite, and it does not implement the missing configurable evaluation ID, internal-saturation telemetry or automatic early stopping.

Before substantial new compute, align/evaluate the six policy IDs on a shared cohort and run a two-phase smoke under a separately named output. At the 250M and 500M diagnostic checkpoints, inspect finite losses/scales, uncontaminated on-policy KL, input saturation and goal response, and compare reward components/per-block goals at common frame coordinates. At 1B, extend only on evidence of goal-dependent progress and improved signal health, not merely lift return. These are manual experiment-review gates, not implemented launcher conditions.

For attribution after a promising diagnostic, add back the auxiliary actor-value loss and cross-member reuse separately, and compare biological/rewired/random/direct controllers with matched data, compute/update budgets, seeds and evaluation members. This is required before attributing improvement to fly topology or declaring any paradigm superior.
