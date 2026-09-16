# Camera-driven MaleCNS reservoir

The visual policy drives real L1/L2 optic-column neurons from raw camera luminance, preserves robot proprioception and trains only the readout of a fixed tanh CNS circuit.

Last updated: 2026-09-16

Related: [Fixed reservoir](fixed-reservoir-controller.md), [Experiment workflow](../workflows/connectome-experiments.md), [Fruitfly source audit](../sources/summaries/quantum-coded-fruitfly.md)

## Circuit and evidence boundary

**Count correction from the subsequent annotation audit:** 162,796 is a graph-node count, not a verified-neuron count. The local annotation table has 211,577 entries, of which 165,122 have `status: Traced`. The current artifact includes 157,167 traced entries, 4,060 annotated entries with another status and 1,569 IDs absent from that annotation table. The 4,060 include 2,177 Orphan, 1,317 missing-status, 240 Assign, 220 Anchor, 64 Glia and 42 Unimportant entries. The extractor uses connectivity endpoints without a status/glia filter; therefore the earlier description of every node as a neuron was too strong. The published approximately 166K denominator is not interchangeable with this artifact's endpoint count. No running job was changed during this audit.

Of the 165,122 locally traced entries, 7,955 are excluded: 2,937 optic-lobe intrinsic cells, 950 optic sensory cells, 1,162 central-brain sensory cells, 896 VNC sensory cells, 532 central-brain intrinsic cells, 187 VNC intrinsic cells, 477 VNC motor cells, 43 central-brain motor cells and 771 other/unassigned cells. Common excluded types include T1 (1,659), R1-R6 (715), BM_InOm (495), Dm3b (400), Lawf2 (295), Dm3a (182) and L4 (161). These counts were computed by body-ID set difference against the saved NPZ and `annotations.feather`. Exclusion means failing the chosen L1/L2-to-front-motor path criterion (maximum eight edges after a five-synapse edge threshold), except that the original core is retained explicitly. It does not establish that those neurons are unimportant for vision or behavior. In particular, starting at L1/L2 bypasses photoreceptor processing. A future cleaned artifact should explicitly select permitted reconstruction statuses and resolve missing-annotation endpoints before claiming biological neuron coverage.

`configs/connectome/malecns_visual.yaml` pins the local MaleCNS v1 annotations, full body-to-body synapse counts, neurotransmitters and original 1,952-cell proprioceptive artifact by SHA-256. The extractor selects annotated L1/L2 cells in both eyes with finite `assignedOlHex1/2`, retains nodes on directed anatomical paths of at most eight edges from these inputs to the original motor population, unions the original core, and retains measured induced edges with at least five synapses. No artificial visual-to-motor edges are added.

The artifact contains 162,796 neurons, 6,144,284 stored edges, 3,534 visual inputs (1,767 each L1/L2), 92 original proprioceptors, 292 tactile cells, 157 descending ports and 135 motor cells. Its SHA-256 is `3d2d8e86206c587b5aebe8372dbef6ac4e2fe8b801b8732b704e50db177371cd`. The combined sensory port has 3,918 cells. As in the previous model, ACh is positive and GABA/glutamate are negative; histamine is negative. Unknown/missing labels retain the earlier negative assumption. Dopamine, serotonin and octopamine have **zero fast-current weights**, because no receptor-specific modulation/plasticity rule is implemented. Thus there are 6,094,882 effective nonzero edges. Anatomical reachability is 135 motors, but effective nonzero-weight reachability is 130: 8 at distance three, 103 at four, 19 at five and five unreachable within eight edges. Do not confuse either count with measured activity sensitivity.

The signed operator is globally spectral-normalized to radius one (raw radius 3549.56081767). Tanh dynamics retain beta `.9` and leak `.5` over a control interval; this profile uses nine recurrent updates under `no_grad`. PPO caches only the 135 motor features. These weights and dynamics are engineering approximations of physiology.

Unlike Wordle's five symbolic feedback bins, this interface samples a real image at column positions. Wordle's actual extractor uses visual-superclass/`PN` string masks and X-coordinate quantiles; it neither implements camera vision nor validates retinotopy. Primary references for our column data are the [MaleCNS data release](https://github.com/flyconnectome/2025malecns/blob/main/README.md) and the [visual-system column assignment methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12119369/).

## Image and nonvisual input contract

Each environment has its own 96x54 GPU RGB sensor at the same environment-local pose and default field of view as evaluation video: position `(0,-1,1.03)`, target `(0,0,.53)`. Both camera constructors use `simtoolreal_camera_pose`. Graphics advance before each capture. The sensor image is converted to luminance using fixed RGB coefficients `[.299,.587,.114]`; there are no object detectors, segmentation masks or learned image adapters.

The green goal visualization actor is parked below the scene before indexed state updates; logical goal tensors remain independent and drive rewards and the explicit task channel. Debug annotations and the blue target robot are rejected for this profile. Image capture does not apply video annotations. Ordinary resets are required; success-state replay and state-file recording are rejected because hidden visualization actors cannot serve as logical goal snapshots.

The actor observation has 5,283 values:

| Range | Meaning | Fixed neural input |
| --- | --- | --- |
| 0:29 | Joint position | Original proprioceptor signed code |
| 29:58 | Joint velocity | Original proprioceptor opponent code |
| 58:87 | Previous joint target | Descending slots 0:29 |
| 87:99 | Desired fixed-size world-space goal keypoints | Descending opponent slots 29:53 |
| 99:5283 | Raw camera luminance | L1/L2 visual columns |

Actual object pose, object-relative keypoints and object-to-goal error are absent from actor inputs. The goal keypoints encode desired position/orientation without revealing the current object pose; the privileged central critic retains its original state inputs. The actor has no running input normalization. Fixed encoding and all neural recurrence are frozen, while Gaussian output heads, SAPG embedding and actor auxiliary value head train normally (`use_experimental_cv: true`).

Annotated hex columns are embedded as `(h1-.5*h2, sqrt(3)/2*h2)` and each eye's bounding box is mapped to the same camera image. Bilinear samples drive `tanh(gain*(1-2*luminance))` at L1/L2. This supplies an early-visual luminance code but bypasses photoreceptors. The hex-to-camera alignment, polarity/scale and duplicated monocular image across eyes are explicit engineering assumptions, not calibrated fly optics or physiological L1/L2 responses. Retinotopic positions and neural connectivity come from annotations/data; that does not establish natural visual function.

## Validation and current run

The network regression and initial retina tests passed (93 tests); the added profile test verifies that privileged object state cannot enter the actor observation. The 96- and 384-environment smoke suites each completed two PPO epochs and checkpoint reload with finite 29-action outputs. At 384 environments the warm epoch took about 4.05 seconds, approximately 1,518 environment steps/s. This much larger circuit plus rendering is substantially slower than the earlier small reservoir.

`scripts/audit_visual_reservoir.py` checked independent environment images, nonblank pixels and byte-identical raw images after moving only the logical goal/goal visualization actor. The PNGs at `profiles/connectome/visual_audit/` were inspected. After 16 constant-image control decisions, changing dark to bright input yielded maximum motor delta about `2.34e-5`, mean `2.24e-6`, with 117 cells exceeding `1e-8` in the latest probe. The response is small: these checks establish a functioning visual path and no tested goal-image leakage, not strong visual features or successful control. Contrast sensitivity and learned task performance remain evaluation questions.

The former LIF suite/trainer/watcher PIDs 450554/450610/450997 were stopped. The fresh full visual tanh suite/trainer are 468643/468686 on physical GPU 1; watcher PID 469141 records three videos at each 1M-frame milestone. The GPU-0 learned-adapter Gaussian trainer PID 261951 remains live. Port 6008 exposes `visual_tanh_100b` through a summaries symlink without a server restart. At frame 55,296, actor value loss was 1.2072, entropy 41.1306 and KL .0101824; central value loss was .110878 at 61,440 and both invalid-KL flags were zero. This is launch validation only.

## Reproduction and helper responsibilities

Run from the repository root (the full training command is already running; its YAML refuses output-directory reuse):

```bash
.venv/bin/python scripts/prepare_malecns_connectome.py --config configs/connectome/malecns_visual.yaml
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_visual_tanh_smoke384.yaml
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/audit_visual_reservoir.py --config configs/connectome/visual_audit.yaml
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/ppo_visual_tanh_100b.yaml
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config configs/connectome/evaluation/ppo_visual_tanh_100b_milestones.yaml
```

The audit YAML currently references the original 96-environment smoke checkpoint, so run `ppo_visual_tanh_smoke.yaml` first for that exact audit, or change both audit paths to a validated replacement. The preparation YAML owns source hashes, input cell types, eye sides, minimum synapses, path depth and transmitter policy; its helper `simtoolreal_shared/visual_connectome.py` extracts and verifies the immutable NPZ. `retina.py` performs fixed bilinear sampling and population scatter and has no trainable parameters. The environment owns rendering and goal-actor suppression. The suite YAML owns 384 environments, six 64-environment SAPG blocks, minibatch 1,536, Gaussian sigma cap three, LF/1 reuse, KL `.004`, entropy scale `.005`, auxiliary CV, a 100B upper budget and 1M inference checkpoints. The train profile owns artifact counts, nine neural updates and retinal geometry. The audit entrypoint loads an environment/player, runs leakage/sensitivity assertions and writes JSON/PNGs; its `sensitivity_control_steps` controls the static-image probe duration. The watcher YAML owns the matched policy config, GPU and three deterministic task-video cases. Helpers are imported, not separately launched.
