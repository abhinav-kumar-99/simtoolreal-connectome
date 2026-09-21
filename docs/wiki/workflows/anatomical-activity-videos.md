# Anatomical activation videos

Record the compact MaleCNS actor's recurrent activity and render it on real anatomical skeletons alongside a synchronized robot rollout.

Last updated: 2026-09-20

Related: [Experiment workflow](connectome-experiments.md), [Connectome actor](../concepts/connectome-actor.md), [MaleCNS source](../sources/summaries/malecns-front-leg-circuit.md)

## Run the entrypoints

From the normal repository checkout, replay the six paper-tolerance cases for both all-neuron actors at the shared 7B milestone:

```bash
.venv/bin/python scripts/replay_connectome_activity.py --config configs/connectome/visualization/replay_1952_7b.yaml
```

The YAML owns `case_paths`, `output_directory`, one physical `gpu`, `runtime_environment`, optional `camera_resolution_reduction_factor` / `video_quality` capture overrides, and `circuit`. Each source `case.yaml` supplies its checkpoint, resolved policy config, trajectory, tolerance, episode count, camera sampling and simulator overrides. Capture overrides apply only to the new destination cases. Replay writes new robot footage, `activity.npz`, `circuit.mp4`, `rollout_with_circuit.mp4`, and `circuit_render.json` into the new directory, plus a local `index.html` at the output root for browsing all pairs. Existing source videos are preserved. Activity matches the newly replayed footage; exact reproduction of an old MP4 is not assumed. Re-running reuses complete cases with matching circuit settings. Appearance changes rerender saved traces without a simulator worker after verifying simulation options, checkpoint/config hashes and footage. Camera resolution or source encoding quality changes require a new recording. New traces embed the recording configuration and footage hash; legacy traces additionally use the previous case YAML and successful render hashes.

To change appearance without rerunning Isaac Gym:

```bash
.venv/bin/python scripts/render_connectome_activity.py --config configs/connectome/visualization/render_1952_example.yaml
```

This YAML owns `trace_path`, `rollout_path`, `output_directory`, and `circuit`. Paths are absolute or relative to the repository root. Generated geometry and videos are ignored by Git.

The current preset writes the HD batch to `evals/connectome/anatomical_1952_7b_hd/`, retaining the earlier `anatomical_1952_7b/` recordings. Open the destination's `index.html` to browse the six pairs, with actors grouped by task. The gallery explains the color/bar/leg conventions and provides 0.25×, 0.5× and normal-speed playback for closer inspection; this does not change encoded FPS or synchronization.

To generate the matched no-auxiliary threshold comparison, use the normal evaluator:

```bash
.venv/bin/python scripts/run_connectome_evaluation.py --config configs/connectome/evaluation/ppo_1952_7b_noaux_dual_tolerance_anatomical_hd.yaml
```

This YAML fixes one no-auxiliary 7B checkpoint, deterministic mean actions, the three task cases, native 1600 × 900 capture, source quality 9, 1920 × 1080 combined output, and both success metrics. `paper_task_progress` uses the paper tolerance of 0.02 m. `checkpoint_training_tolerance` uses 0.039858076721429825 m, resolved from TensorBoard tag `scalars/success_tolerance/frame` at checkpoint frame 7,000,031,232. The resulting gallery places the metric and tolerance in every card title so the two result groups are distinguishable. `scripts/replay_connectome_activity.py` remains the saved-case replay entrypoint; its `write_video_index` helper creates the same local gallery from completed artifacts without rerunning the simulator.

To force a fresh run on physical GPU 1, use `configs/connectome/evaluation/ppo_1952_7b_noaux_dual_tolerance_anatomical_hd_cuda1.yaml`. It keeps `gpu_assignments: [1]` and writes to a new `_cuda1` output directory, so the existing completed directory is not reused. The noaux configs include both blue-brush trajectories, `sweep_forward` and `sweep_right`, in addition to the original three cases.

For the matched asymmetric fly actor, `configs/connectome/evaluation/ppo_lstm323_fly1952_asymmetric_noaux_mlp128x32x32_rollout_kl_100b_fly_checkpoint_lowres.yaml` watches only `fly1952_asymmetric_mlp128x32x32`, uses only the checkpoint TensorBoard success tolerance, assigns physical GPU 1, and captures lower-resolution 400 × 225 source footage with 1280 × 720 circuit output. It processes the available 250M-frame milestones and continues watching the 100B run.

For the K=2 one-third-bounds pair, the same low-res anatomical contract is in `configs/connectome/evaluation/ppo_fly1952_k2_adaptive_lr_third_bounds_100b_checkpoint_lowres.yaml` (control, GPU 0) and `configs/connectome/evaluation/ppo_fly1952_k2_adaptive_lr_third_bounds_intrinsic_plasticity_100b_checkpoint_lowres.yaml` (intrinsic plasticity, GPU 1). Both write separate `_checkpoint_lowres` output trees so the existing robot-only dual-tolerance milestone galleries stay unchanged.

For future ordinary or milestone evaluations, add this block to `videos` (under `evaluation.videos` in watcher configs):

```yaml
circuit:
  enabled: true
  fps_multiplier: 4
  group_labels: true
  leg_shadows: true
  activity_bars: true
  robot_crop: [0.30, 0.06, 0.70, 0.88]
  robot_panel_fraction: 0.40
  overview_panel_fraction: 0.65
  resolution: [1920, 1080]
  geometry_cache: data/connectomes/geometry/malecns_v1
  annotations_path: data/connectomes/raw/malecns_v1_audit/annotations.feather
  projection: [x, z]
  vnc_bounds_um: [240, 560, 400, 1100]
  outputs: [circuit, combined]
```

Run the existing entrypoint with that YAML:

```bash
.venv/bin/python scripts/run_connectome_evaluation.py --config <evaluation.yaml>
.venv/bin/python scripts/run_connectome_milestone_evaluation.py --config <milestone.yaml>
```

The optional block defaults to disabled. Existing jobs are not restarted or enabled automatically. An enabled watcher backfills configured checkpoints whose robot-only outputs lack activity artifacts. Multiple video metrics retain independent rollouts/traces. Existing video FPS, camera sampling/resolution and deterministic mean actions remain authoritative.

For full-resolution capture in ordinary/milestone evaluations, set `videos.camera_resolution_reduction_factor: 1` alongside `videos.circuit` (under `evaluation` in watcher YAMLs). The replay preset instead sets the capture override at the top level. The camera's native size is 1600 × 900: factor 1 captures that size, versus factor 2 at 800 × 450. Changing only `circuit.resolution` scales the composition and cannot recover missing camera detail.

## Important configuration

| Parameter | Meaning |
| --- | --- |
| `fps_multiplier` | Animation FPS relative to robot MP4; 4 makes 20 fps into 80 fps. |
| `resolution` | Even output width/height; general default 1600 × 900, HD presets 1920 × 1080. |
| `robot_crop` | Fixed camera crop `[xmin, ymin, xmax, ymax]`, normalized from 0 to 1. The HD presets keep the robot, tool and table using `[0.30, 0.06, 0.70, 0.88]`; the general default is the full image. |
| `robot_panel_fraction` | Simulation share of the combined view's usable width, from 0.4 to 0.7; general default 0.58, HD presets 0.40. The cropped image is scaled to fit while preserving its aspect ratio. |
| `overview_panel_fraction` | CNS overview share of the anatomy panel width, from 0.4 to 0.8; default/presets 0.65. The remaining width goes to VNC detail. |
| Top-level `camera_resolution_reduction_factor` | Replay-only capture override; positive integer, preset 1 for native 1600 × 900. Omit to retain each source case's setting. |
| Top-level `video_quality` | Replay-only source MP4 quality override, integer 0–10, preset 9. Higher values retain more detail and increase file size. Omit to retain source configuration / worker default 5. |
| `geometry_cache` | SWC files, provenance manifest and cached raster projections. |
| `annotations_path` | Native MaleCNS annotations supplying gray dataset soma context. |
| `projection` | Fixed native EM X–Z view, with equal scale on both axes. |
| `vnc_bounds_um` | Detail crop `[xmin, xmax, zmin, zmax]` in micrometers. |
| `outputs` | `circuit`, `combined`, or both. |
| `group_labels` | Brain/VNC, sensory/descending/motor labels and T1/T2/T3 guides; enabled in the shipped presets. |
| `leg_shadows` | Faint three-pair leg schematic showing anatomical orientation; enabled in the presets. |
| `activity_bars` | Population mean absolute tanh state on a fixed 0–1 scale; enabled in the presets. |

The compact layout puts time/episode/counts alongside the title, population roles alongside their names, and the color key alongside source credits. Model/task names and the gray-leg explanatory sentence are omitted from the encoded frames; case names and interpretation remain in the gallery. The original version-3 compact layout gave the simulation 58% and split anatomy equally, leaving the overview small. The HD presets give the simulation 40% and the overview 65% of the remaining anatomy width. At 1920 × 1080, the simulation viewport is 735 × 849, the anatomy panel is 1103 × 849, and its overview allocation is 717 pixels wide. The tighter camera crop is 640 × 738 from native footage and scales to 735 × 848. Cropping only affects the combined render, preserving the saved camera footage and neural trace. Renderer version 4 invalidates earlier layouts; raster cache identity includes the actual view rectangles so changing the split cannot reuse incompatible geometry.

The header says **1,952 neurons / 33,720 connections**. The builder verifies 1,952 ordered body IDs and 33,720 stored CSR connection weights against the selected circuit artifact. Connections are neuron-to-neuron graph entries, not additional neurons or a count of individual synaptic contacts.

## Timing and interpretation

Four neural updates per 60 Hz control step yield an effective 240 substeps/s. Recording every third robot step at 20 fps and rendering at 80 fps normally selects every third recorded neural state. The renderer uses the latest available state on the video clock and repeats each robot image four times. Duration stays equal. Final camera-frame padding holds the last state; episodes restart from their recorded initial state without blending across resets.

This animation clock places the internal algorithm updates uniformly within each control step. In inference they are computed together before the action; the plotted subframes are not independently measured physiological or wall-clock event times.

The fixed tanh scale is −1 to +1. Cyan/orange indicate positive/negative modeled state, not excitatory/inhibitory anatomical neuron identity. One scalar colors each whole morphology. Overlapping projected neurons blend contributions with fixed density compensation; activity is never normalized per frame. This is real anatomy carrying engineered controller state, with no modeled within-neuron propagation or claim of measured biological activity.

The annotation preset explains the actual controller routes: robot observations drive the sensory input mask, task goals/conditioning drive the descending input mask, and **all 1,952 cells** feed this pair's learned robot action decoder. The four disjoint display groups are 384 sensory input cells, 157 descending input cells, 135 motor cells and 1,276 remaining cells. These are controller interface masks, not an exhaustive physiological classification; additional descending/ascending cells can be in the remaining group. Bars summarize mean absolute state, not causal importance or biological firing rates.

The sensory mask's source classes are tactile/proprioceptive mechanosensory cells; the 135 motor cells all have T1 soma annotations, supporting the front-leg label. Label leaders identify neuron populations rather than physical injection sites or activity-flow paths. They now point separately to median projected arbors for each source-annotated side (`rootSide` for sensory cells, `somaSide` for descending/motor cells). T1/T2/T3 guides use median soma locations in the full annotation dataset (Z = 627.696, 738.136, 977.056 μm), which are approximate orientation guides, not measured segment boundaries. The gray leg shadows are explicitly schematic: the CNS skeleton download does not contain physical leg geometry, and no fly leg is mapped to a particular robot finger.

The first annotated version used one pooled median per population and misleadingly placed both sensory and motor callouts in the screen-right bulb. With two separated clusters, even a one-cell imbalance can select one bulb rather than their midpoint. Source annotations confirm both sides are retained: motor soma sides L/R = 68/67, sensory root sides L/R = 220/164, descending soma sides L/R = 79/78. Native X increases toward the right of the image; the source's L-side arbor clusters have larger median X coordinates, so the screen-right cluster corresponds to the dataset's L side. This static callout placement says nothing about which side is more active. Renderer version 2 preserves both side anchors and marks the motor population as "Both front legs". Completion checks invalidate older render versions while keeping saved recordings available for rerendering.

## Helpers, provenance and limits

- `simtoolreal_shared/activity_trace.py` owns config defaults, completion checks, non-mutating state copies and episode-local video-clock sampling.
- `simtoolreal_shared/anatomical_activity.py` validates/downloads the trace's ordered body IDs, converts 8 nm SWC coordinates to micrometers, caches sparse per-neuron raster masks, streams MP4s and verifies FPS/frame counts with `ffprobe`. Its `frame_layout` shares viewport geometry between anatomy rasterization and frame composition; `crop_robot_frame` applies the fixed camera ROI before aspect-preserving scaling. These helpers are imported by both entrypoints and have no separate CLI.
- `simtoolreal_shared/activity_interpretation.py` verifies the circuit artifact's ordered IDs/hash, derives population masks and source-backed captions, adds static labels/leg guides, and computes the four activity summaries. It is imported by the renderer; it has no separate CLI.
- `dextoolbench/eval_worker_isaacgym.py` records actual recurrent activity while executing the policy. The parent renders after this child exits, freeing GPU memory first. It is normally dispatched by the entrypoints rather than launched manually.
- `connectome_network_builder.py` calls an optional observer after each native/fused tanh substep. No checkpoint parameters/buffers are added. Tests verify exact action/final-state equality on CPU and CUDA.

Geometry comes from the [official MaleCNS v1.0 skeleton inventory](https://male-cns.janelia.org/download/), matched by `bodyId`; FlyEM/Janelia, CC BY 4.0. Every source URL/hash is recorded. Missing or invalid skeletons fail explicitly without artificial positions. Verified compact-circuit coverage: 1,952/1,952 neurons, 194,668,453 source bytes.

Recording supports one-environment, no-gradient tanh inference. LSTM, LIF and frozen CUDA-graph reservoir inference are rejected explicitly. Rendering is an offline CPU stage; full-CNS rendering capacity has not been evaluated.

## Verified first batch

The 2026-09-18 annotated batch completed for both compact all-neuron actors at target 7,000,000,000 / actual 7,000,031,232 frames: marker `write_c`, eraser `wipe_smile`, and spatula `flip_over`, one saved-case replay episode per cell. All six cases have circuit-only and combined MP4s (12 total), 1600 × 900 at 80 fps versus 20 fps robot footage, 800 or 804 output frames, and matched per-case durations of 10 or 10.05 seconds. Recorded states are finite, ordered body IDs/population masks match the circuit artifact, and the six traces contain 2,393–2,405 substep/initial-state rows each.

Every output frame decoded successfully, and previews from all six encoded combined videos plus the circuit-only examples were inspected. The interpretability rerender reused all six original recordings; trace and robot-footage hashes remained unchanged. Generated evidence is in `evals/connectome/anatomical_1952_7b/verification.json`, `recording_provenance.json`, `preview_montage.png`, and each case's `circuit_render.json`. The final cached replay reused completed outputs and regenerated the gallery without simulation or encoding.

The subsequent bilateral-callout correction rerendered all 12 outputs with renderer version 2 using three CPU processes and no simulator workers. All side anchors/counts, complete-frame decoding, 80 fps, matched durations and unchanged trace/robot-footage hashes passed again. Decoded previews, the montage, metadata and gallery now show the corrected callouts; `verification.json` records `bilateral_callouts_verified: true`.

The HD refresh subsequently replayed all six cases with native 1600 × 900 camera capture and generated 12 version-4 outputs at 1920 × 1080 / 80 fps. Every frame decoded, durations remain 10 or 10.05 seconds, and output frame counts remain exactly four times the 20 fps source counts. All source frames passed the configured crop-content check, while decoded previews and the six-case montage were inspected for complete robot/tool/table coverage and the enlarged overview. Metadata reverified the exact 1,952-neuron / 33,720-connection artifact, all-neuron readout, population masks and bilateral source-side anchors. Evidence and the local gallery are under `evals/connectome/anatomical_1952_7b_hd/`; `verification.json` records the layout, probes and checks.

## Verified noaux dual-tolerance batch

The 2026-09-18 no-auxiliary comparison completed one 7B actual-frame-7,000,031,232 rollout for each of three tasks under each threshold: 0.02 m paper task-progress and 0.039858076721429825 m checkpoint-training tolerance. It produced six cases and 12 version-4 circuit/combined MP4s in `evals/connectome/anatomical_1952_7b_noaux_dual_tolerance_hd/`. All combined source footage is native 1600 × 900 at 20 fps; every 1920 × 1080 output is 80 fps with exactly four output frames per source frame. All outputs fully decoded, all source frames kept robot/tool/table content inside the configured crop, and the six-case montage was inspected. The paper group averaged 27.6094% task progress; the checkpoint-tolerance group averaged 77.3333%. The eraser reaches its relaxed success threshold at 5.6 seconds, so that case ends before the 10.05-second horizon.
