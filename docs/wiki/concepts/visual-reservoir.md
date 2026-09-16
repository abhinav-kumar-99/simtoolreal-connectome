# Camera-driven MaleCNS reservoir

The visual policy drives real L1/L2 optic-column neurons from cached raw-camera luminance, preserves fresh robot proprioception and trains only the readout of a fixed tanh CNS circuit.

Last updated: 2026-09-16

Related: [Fixed reservoir](fixed-reservoir-controller.md), [Experiment workflow](../workflows/connectome-experiments.md), [Fruitfly source audit](../sources/summaries/quantum-coded-fruitfly.md)

## Circuit and evidence boundary

### Current full-neuron replacement

At the user's subsequent direction, `configs/connectome/malecns_full_cns.yaml` selects `all_traced_neurons`: exactly all 165,122 body IDs with `status: Traced` in the pinned annotation table, including isolated neurons. There is no visual-path pruning in this mode; `maximum_path_edges` is used only for reachability diagnostics. The five-synapse edge floor and existing transmitter model remain, so all neurons does not mean every weak synaptic edge is retained. Glia, other reconstruction statuses and endpoints without annotations are excluded explicitly. This adds 7,955 traced neurons and removes 5,629 non-Traced/unannotated nodes from the previous artifact. Exact body-ID equality with the full Traced set was verified, as were unchanged visual, proprioceptive, descending input and motor output identities.

The new artifact has 6,235,682 stored edges, 6,185,843 nonzero fast-current edges, 3,534 L1/L2 input cells and SHA-256 `389953b5b28c70a1cbd84ef0cd32b66c7cda37ead88056b936776ce068b42f04`. Its spectral radius before normalization is 3549.6069472. The current `SimToolRealFullCNSVisualReservoirGaussianSAPGVision64x36R4` profile retains the same proprioception, goal routing, nine tanh updates, cached features, Gaussian settings and auxiliary actor value objective while using the lower-rate camera contract described below. The preceding 96x54 every-step profile remains reproducible, but its live run was stopped at 1,480,704 frames after its 1M evaluation completed.

### Output population audit

All 135 readout cells are annotated `vnc_motor`, `somaNeuromere: T1`: 68 left and 67 right. Their annotated types include tibial/trochanteral flexors and extensors, tarsal levators/depressors and proximal leg muscles. They therefore remain a defensible downstream foreleg motor population after the circuit expands. They are not intrinsically mapped to robot joints: the learned 167-input (135 motor plus 32 SAPG) readout converts activity to 29 robot commands. Using all CNS neurons does not require reading every neuron as an output.

The full annotated population contains 708 VNC motor neurons, 107 central-brain motor neurons and 1,314 descending neurons. A motor-only versus descending-plus-front-motor comparison would test whether higher-level commands retain useful visual information lost before the final motor stage. Reading all motor types would also mix other body functions into the controller. The current 135-cell choice is biologically motivated but not empirically optimal; no readout expansion was silently included in this full-neuron experiment.

### Descending neurons as a possible readout

Biologically, descending neurons link brain processing to ventral nerve cord motor circuitry, where local circuits and sensory feedback help generate coordinated movements. They can initiate or modify behavior without encoding each muscle command; the population is not a list of independent robot-joint commands. See [Namiki et al., 2018](https://elifesciences.org/articles/34272). Anatomical identity does not establish that our frozen tanh approximation reproduces those functions.

An audit of the pinned Traced annotations finds 1,314 descending neurons across 480 non-null type labels (656 left, 648 right, 10 midline). All are simulated in the full artifact. The 157 descending input ports are a subset, not the total descending population. In the current visual profile, only 53 receive direct external drive: 29 previous joint targets and 24 opponent-coded desired-goal channels. The other 104 input ports have no directly mapped external channel in this profile.

A descending readout would expose signals before the fly-specific VNC-to-muscle transformation; a hybrid descending-plus-135-motor readout would retain both stages. This is an experimental hypothesis, not demonstrated robot-control improvement. It requires a new readout population option: the current implementation remains motor-only. All descending cells are already simulated, so reading them adds no recurrent neural updates or need for BPTT. Caching 1,314 additional float32 features for 384 environments and a 16-step rollout would add about 30.8 MiB before other training buffers.

Important confound: reading all descending cells also exposes the 53 directly driven goal/previous-action cells. A readout can exploit that route without relying on visual processing. Compare motor-only, descending-plus-motor, and descending-plus-motor excluding those 53 directly driven cells (1,261 remaining descending cells), with blank/shuffled-camera controls. Exclusion removes direct input/readout overlap, not all indirect nonvisual information. No architecture or running job was changed for this explanation.

The current multirate replacement entrypoints are:

```bash
.venv/bin/python scripts/prepare_malecns_connectome.py --config configs/connectome/malecns_full_cns.yaml
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_full_cns_tanh_vision64x36_r4_smoke.yaml
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/audit_visual_reservoir.py --config configs/connectome/full_cns_visual_64x36_r4_audit.yaml
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_full_cns_tanh_vision64x36_r4_100b.yaml
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_full_cns_tanh_vision64x36_r4_100b_milestones.yaml
```

The helper responsibilities described below are unchanged; `selection_mode` owns the full-neuron selection and the helper rejects any selection omitting an existing interface cell. The full training/watcher contracts retain 384 environments, six 64-environment SAPG blocks, minibatch 1,536, GPU 1, 100B cap and 1M video checkpoints. TensorBoard alias: `full_cns_tanh_vision64x36_r4_100b`. Full runs and smoke runs refuse existing output directories.

### Historical visual-path artifact and count correction

**Count correction from the subsequent annotation audit:** 162,796 is a graph-node count, not a verified-neuron count. The local annotation table has 211,577 entries, of which 165,122 have `status: Traced`. The current artifact includes 157,167 traced entries, 4,060 annotated entries with another status and 1,569 IDs absent from that annotation table. The 4,060 include 2,177 Orphan, 1,317 missing-status, 240 Assign, 220 Anchor, 64 Glia and 42 Unimportant entries. The extractor uses connectivity endpoints without a status/glia filter; therefore the earlier description of every node as a neuron was too strong. The published approximately 166K denominator is not interchangeable with this artifact's endpoint count. No running job was changed during this audit.

Of the 165,122 locally traced entries, 7,955 are excluded: 2,937 optic-lobe intrinsic cells, 950 optic sensory cells, 1,162 central-brain sensory cells, 896 VNC sensory cells, 532 central-brain intrinsic cells, 187 VNC intrinsic cells, 477 VNC motor cells, 43 central-brain motor cells and 771 other/unassigned cells. Common excluded types include T1 (1,659), R1-R6 (715), BM_InOm (495), Dm3b (400), Lawf2 (295), Dm3a (182) and L4 (161). These counts were computed by body-ID set difference against the saved NPZ and `annotations.feather`. Exclusion means failing the chosen L1/L2-to-front-motor path criterion (maximum eight edges after a five-synapse edge threshold), except that the original core is retained explicitly. It does not establish that those neurons are unimportant for vision or behavior. In particular, starting at L1/L2 bypasses photoreceptor processing. A future cleaned artifact should explicitly select permitted reconstruction statuses and resolve missing-annotation endpoints before claiming biological neuron coverage.

`configs/connectome/malecns_visual.yaml` pins the local MaleCNS v1 annotations, full body-to-body synapse counts, neurotransmitters and original 1,952-cell proprioceptive artifact by SHA-256. The extractor selects annotated L1/L2 cells in both eyes with finite `assignedOlHex1/2`, retains nodes on directed anatomical paths of at most eight edges from these inputs to the original motor population, unions the original core, and retains measured induced edges with at least five synapses. No artificial visual-to-motor edges are added.

The artifact contains 162,796 neurons, 6,144,284 stored edges, 3,534 visual inputs (1,767 each L1/L2), 92 original proprioceptors, 292 tactile cells, 157 descending ports and 135 motor cells. Its SHA-256 is `3d2d8e86206c587b5aebe8372dbef6ac4e2fe8b801b8732b704e50db177371cd`. The combined sensory port has 3,918 cells. As in the previous model, ACh is positive and GABA/glutamate are negative; histamine is negative. Unknown/missing labels retain the earlier negative assumption. Dopamine, serotonin and octopamine have **zero fast-current weights**, because no receptor-specific modulation/plasticity rule is implemented. Thus there are 6,094,882 effective nonzero edges. Anatomical reachability is 135 motors, but effective nonzero-weight reachability is 130: 8 at distance three, 103 at four, 19 at five and five unreachable within eight edges. Do not confuse either count with measured activity sensitivity.

The signed operator is globally spectral-normalized to radius one (raw radius 3549.56081767). Tanh dynamics retain beta `.9` and leak `.5` over a control interval; this profile uses nine recurrent updates under `no_grad`. PPO caches only the 135 motor features. These weights and dynamics are engineering approximations of physiology.

Unlike Wordle's five symbolic feedback bins, this interface samples a real image at column positions. Wordle's actual extractor uses visual-superclass/`PN` string masks and X-coordinate quantiles; it neither implements camera vision nor validates retinotopy. Primary references for our column data are the [MaleCNS data release](https://github.com/flyconnectome/2025malecns/blob/main/README.md) and the [visual-system column assignment methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12119369/).

## Image and nonvisual input contract

Each environment has its own 64x36 GPU RGB sensor at the same environment-local pose and default field of view as evaluation video: position `(0,-1,1.03)`, target `(0,0,.53)`. Both camera constructors use `simtoolreal_camera_pose`. The current profile renders once every four 60 Hz control steps, or 15 camera frames/s, and reuses the last luminance image between renders. Proprioception, previous targets, desired goal and all nine recurrent fly updates remain fresh at 60 Hz. Graphics advance before each actual capture. The sensor image is converted to luminance using fixed RGB coefficients `[.299,.587,.114]`; there are no object detectors, segmentation masks or learned image adapters. The 2,304 image samples still exceed the 1,767 cells in either the L1 or L2 input population, although the two eyes continue to sample the same external view.

The green goal visualization actor is parked below the scene before indexed state updates; logical goal tensors remain independent and drive rewards and the explicit task channel. Debug annotations and the blue target robot are rejected for this profile. Image capture does not apply video annotations. Ordinary resets are required; success-state replay and state-file recording are rejected because hidden visualization actors cannot serve as logical goal snapshots.

The actor observation has 2,403 values:

| Range | Meaning | Fixed neural input |
| --- | --- | --- |
| 0:29 | Joint position | Original proprioceptor signed code |
| 29:58 | Joint velocity | Original proprioceptor opponent code |
| 58:87 | Previous joint target | Descending slots 0:29 |
| 87:99 | Desired fixed-size world-space goal keypoints | Descending opponent slots 29:53 |
| 99:2403 | Cached raw camera luminance | L1/L2 visual columns |

Actual object pose, object-relative keypoints and object-to-goal error are absent from actor inputs. The goal keypoints encode desired position/orientation without revealing the current object pose; the privileged central critic retains its original state inputs. The actor has no running input normalization. Fixed encoding and all neural recurrence are frozen, while Gaussian output heads, SAPG embedding and actor auxiliary value head train normally (`use_experimental_cv: true`).

Annotated hex columns are embedded as `(h1-.5*h2, sqrt(3)/2*h2)` and each eye's bounding box is mapped to the same camera image. Bilinear samples drive `tanh(gain*(1-2*luminance))` at L1/L2. This supplies an early-visual luminance code but bypasses photoreceptors. The hex-to-camera alignment, polarity/scale and duplicated monocular image across eyes are explicit engineering assumptions, not calibrated fly optics or physiological L1/L2 responses. Retinotopic positions and neural connectivity come from annotations/data; that does not establish natural visual function.

## Validation and current run

The prior network regression and initial retina tests passed (93 tests); the focused multirate/configuration suite adds another profile-contract check and passed 32 tests. The 64x36/render-every-four smoke completed two PPO epochs and checkpoint reload with finite 29-action outputs. Its warmed 384-environment epoch reached 4,880 environment step FPS and 2,934 total FPS.

`scripts/audit_visual_reservoir.py` now reads policy-camera dimensions from the resolved configuration. The 64x36 audit checked independent environment images, nonblank pixels and byte-identical raw images after moving only the logical goal/goal visualization actor. After 16 constant-image control decisions, changing dark to bright input yielded maximum motor delta about `2.34e-5`, mean `2.23e-6`, with 115 cells exceeding `1e-8`. The response is small: these checks establish a functioning visual path and no tested goal-image leakage, not strong visual features or successful control. Contrast sensitivity and learned task performance remain evaluation questions.

### Throughput decomposition

`performance/step_fps` is environment-only throughput: RL Games starts its timer immediately before `env_step()` and stops it afterward, outside policy inference. In the historical every-step vision profile, observation construction synchronously called `fetch_results`, `step_graphics`, `render_all_camera_sensors`, and image-tensor access on every control step, then stacked 384 separate 96x54 RGBA camera tensors, converted them to float and calculated luminance. Thus the fly recurrence could not directly explain low `step_fps`; it appeared in the difference between step-only and step-plus-inference timing.

At the 2026-09-16 audit snapshot, the 96x54 full-CNS run's median across 153 logged epochs was about 1,913 step FPS and 1,539 step-plus-inference FPS. A representative 6,144-frame epoch spent 3.18 seconds in environment steps, 3.95 seconds for the full rollout, and 0.064 seconds in the PPO update. The optimizer is cheap because cached reservoir features avoid rerunning or backpropagating through the CNS during PPO minibatches. The preceding 162,796-node visual-path run was effectively identical at median 1,917/1,542 FPS, so adding the remaining Traced neurons did not cause the slowdown.

The 64x36/render-every-four replacement completed its two-epoch smoke and finite 29-action checkpoint reload. Its warmed epoch reached 4,880 step FPS and 2,934 total FPS. Early in the full run it stabilized near 5,900 step FPS, 3,390 step-plus-inference FPS and 3,250 total FPS, roughly 2.2 times the old end-to-end rate. Lower resolution and multirate rendering changed together, so this does not attribute their individual gains.

The non-camera 1,952-neuron reservoir is not a matched timing control: it uses 12,288 environments rather than 384. Its latest approximately 202K aggregate step FPS combines 32 times more simultaneous environments with a roughly three-times-shorter environment-step phase. A proper camera-cost measurement requires a same-384-environment, same-policy benchmark with camera rendering toggled off. The code boundary and equal speeds of the two large visual graphs nevertheless show that the step-only bottleneck was on the rendered-environment side, not the extra full-CNS cells. Even at roughly 3.25K total FPS, one billion frames takes approximately 3.6 days and the nominal 100B cap about one year, so the cap remains an upper bound rather than a practical target.

The 96x54 full-CNS suite/trainer/watcher PIDs 475520/475598/475523 were stopped with artifacts preserved. The multirate suite/trainer PIDs 484220/484301 and watcher PID 484223 run on physical GPU 1; the watcher records three videos at each 1M-frame milestone. The GPU-0 learned-adapter Gaussian trainer PID 261951 remains live. Port 6008 exposes `full_cns_tanh_vision64x36_r4_100b` through a summaries symlink without a server restart. At frame 55,296, actor and critic losses were finite, entropy was 41.1281, KL was .009295 and both invalid-KL flags were zero. This is launch validation only.

## Reproduction and helper responsibilities

### Further throughput candidates (not implemented)

After the render-rate change, a live timing sample at frame 878,592 measured 1.185 seconds inside environment steps, 1.971 seconds collecting the rollout, and 0.065 seconds for PPO, about 3,018 total FPS. These host timers are not CUDA-event attribution: synchronization inside environment stepping can charge earlier queued GPU work to that phase. Profile GPU work before assigning exact percentages to camera versus CNS computation.

The tanh Triton `_Step.forward` in `connectome_triton.py` still allocates `out`, `rec`, and `z`, and `_sparse_step` writes the latter two for backward even when the reservoir runs under `no_grad`. A separate inference path could omit those two arrays without changing recurrence equations. Each array at 165,122 neurons and 384 environments is about 242 MiB; eliminating two writes across nine updates avoids about 4.25 GiB of nominal output writes per batched control step. This is a bandwidth opportunity, not a measured speedup or persistent-memory estimate. `VecTask.render()` does not secretly step graphics between scheduled policy renders in current headless training because `_modify_render_settings_if_headless()` sets `enable_viewer_sync=False`.

Other candidates are reusable recurrent buffers and CUDA graph capture of fixed-shape neural inference, fused image conversion/retinal sampling, and a measured environment-count sweep preserving SAPG block ratios. Changes to neural update count or further camera decimation change temporal information and should be evaluated as policy changes. The existing small-circuit backend benchmarks do not establish the fastest backend for this full-CNS forward-only workload.

Run from the repository root (the new full training command is already running; its YAML refuses output-directory reuse):

```bash
.venv/bin/python scripts/prepare_malecns_connectome.py --config configs/connectome/malecns_full_cns.yaml
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_full_cns_tanh_vision64x36_r4_smoke.yaml
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/audit_visual_reservoir.py --config configs/connectome/full_cns_visual_64x36_r4_audit.yaml
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_full_cns_tanh_vision64x36_r4_100b.yaml
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_full_cns_tanh_vision64x36_r4_100b_milestones.yaml
```

The audit YAML references the multirate smoke checkpoint, so that smoke must complete before rerunning the audit. The preparation YAML owns source hashes, input cell types, eye sides, minimum synapses, path depth and transmitter policy; its helper `simtoolreal_shared/visual_connectome.py` extracts and verifies the immutable NPZ. `retina.py` performs fixed bilinear sampling and population scatter and has no trainable parameters. The task YAML owns 64x36 geometry and the four-step render interval; the environment owns image caching and goal-actor suppression. The suite YAML owns 384 environments, six 64-environment SAPG blocks, minibatch 1,536, Gaussian sigma cap three, LF/1 reuse, KL `.004`, entropy scale `.005`, auxiliary CV, a 100B upper budget and 1M inference checkpoints. The train profile owns artifact counts, nine neural updates and matching retinal geometry. The audit entrypoint loads an environment/player, derives camera dimensions from the resolved YAML, runs leakage/sensitivity assertions and writes JSON/PNGs; its `sensitivity_control_steps` controls the static-image probe duration. The watcher YAML owns the matched policy config, GPU and three deterministic task-video cases. Helpers are imported, not separately launched.
