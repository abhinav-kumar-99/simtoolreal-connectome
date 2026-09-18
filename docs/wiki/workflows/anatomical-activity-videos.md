# Anatomical activation videos

Record the compact MaleCNS actor's recurrent activity and render it on real anatomical skeletons alongside a synchronized robot rollout.

Last updated: 2026-09-18

Related: [Experiment workflow](connectome-experiments.md), [Connectome actor](../concepts/connectome-actor.md), [MaleCNS source](../sources/summaries/malecns-front-leg-circuit.md)

## Run the entrypoints

From the normal repository checkout, replay the six paper-tolerance cases for both all-neuron actors at the shared 7B milestone:

```bash
.venv/bin/python scripts/replay_connectome_activity.py --config configs/connectome/visualization/replay_1952_7b.yaml
```

The YAML owns `case_paths`, `output_directory`, one physical `gpu`, `runtime_environment`, and `circuit`. Each source `case.yaml` supplies its checkpoint, resolved policy config, trajectory, tolerance, episode count, camera sampling and simulator overrides. Replay writes new robot footage, `activity.npz`, `circuit.mp4`, `rollout_with_circuit.mp4`, and `circuit_render.json` into the new directory, plus a local `index.html` at the output root for browsing all pairs. Existing videos are preserved. Activity matches the newly replayed footage; exact reproduction of an old MP4 is not assumed. Re-running reuses complete cases with matching circuit settings. Appearance changes rerender saved traces without a simulator worker after verifying simulation options, checkpoint/config hashes and footage. New traces embed the recording configuration and footage hash; legacy traces additionally use the previous case YAML and successful render hashes.

To change appearance without rerunning Isaac Gym:

```bash
.venv/bin/python scripts/render_connectome_activity.py --config configs/connectome/visualization/render_1952_example.yaml
```

This YAML owns `trace_path`, `rollout_path`, `output_directory`, and `circuit`. Paths are absolute or relative to the repository root. Generated geometry and videos are ignored by Git.

Open `evals/connectome/anatomical_1952_7b/index.html` in a browser to browse the six pairs, with actors grouped by task. The gallery explains the color/bar/leg conventions and provides 0.25×, 0.5× and normal-speed playback for closer inspection; this does not change encoded FPS or synchronization.

For future ordinary or milestone evaluations, add this block to `videos` (under `evaluation.videos` in watcher configs):

```yaml
circuit:
  enabled: true
  fps_multiplier: 4
  group_labels: true
  leg_shadows: true
  activity_bars: true
  resolution: [1600, 900]
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

## Important configuration

| Parameter | Meaning |
| --- | --- |
| `fps_multiplier` | Animation FPS relative to robot MP4; 4 makes 20 fps into 80 fps. |
| `resolution` | Even output width/height; default 1600 × 900. |
| `geometry_cache` | SWC files, provenance manifest and cached raster projections. |
| `annotations_path` | Native MaleCNS annotations supplying gray dataset soma context. |
| `projection` | Fixed native EM X–Z view, with equal scale on both axes. |
| `vnc_bounds_um` | Detail crop `[xmin, xmax, zmin, zmax]` in micrometers. |
| `outputs` | `circuit`, `combined`, or both. |
| `group_labels` | Brain/VNC, sensory/descending/motor labels and T1/T2/T3 guides; enabled in the shipped presets. |
| `leg_shadows` | Faint three-pair leg schematic showing anatomical orientation; enabled in the presets. |
| `activity_bars` | Population mean absolute tanh state on a fixed 0–1 scale; enabled in the presets. |

## Timing and interpretation

Four neural updates per 60 Hz control step yield an effective 240 substeps/s. Recording every third robot step at 20 fps and rendering at 80 fps normally selects every third recorded neural state. The renderer uses the latest available state on the video clock and repeats each robot image four times. Duration stays equal. Final camera-frame padding holds the last state; episodes restart from their recorded initial state without blending across resets.

This animation clock places the internal algorithm updates uniformly within each control step. In inference they are computed together before the action; the plotted subframes are not independently measured physiological or wall-clock event times.

The fixed tanh scale is −1 to +1. Cyan/orange indicate positive/negative modeled state, not excitatory/inhibitory anatomical neuron identity. One scalar colors each whole morphology. Overlapping projected neurons blend contributions with fixed density compensation; activity is never normalized per frame. This is real anatomy carrying engineered controller state, with no modeled within-neuron propagation or claim of measured biological activity.

The annotation preset explains the actual controller routes: robot observations drive the sensory input mask, task goals/conditioning drive the descending input mask, and **all 1,952 cells** feed this pair's learned robot action decoder. The four disjoint display groups are 384 sensory input cells, 157 descending input cells, 135 motor cells and 1,276 remaining cells. These are controller interface masks, not an exhaustive physiological classification; additional descending/ascending cells can be in the remaining group. Bars summarize mean absolute state, not causal importance or biological firing rates.

The sensory mask's source classes are tactile/proprioceptive mechanosensory cells; the 135 motor cells all have T1 soma annotations, supporting the front-leg label. Label leaders identify neuron populations rather than physical injection sites or activity-flow paths. They now point separately to median projected arbors for each source-annotated side (`rootSide` for sensory cells, `somaSide` for descending/motor cells). T1/T2/T3 guides use median soma locations in the full annotation dataset (Z = 627.696, 738.136, 977.056 μm), which are approximate orientation guides, not measured segment boundaries. The gray leg shadows are explicitly schematic: the CNS skeleton download does not contain physical leg geometry, and no fly leg is mapped to a particular robot finger.

The first annotated version used one pooled median per population and incorrectly placed both sensory and motor callouts in the screen-right bulb. With two separated clusters, even a one-cell imbalance can select one bulb rather than their midpoint. Source annotations confirm both sides are retained: motor soma sides L/R = 68/67, sensory root sides L/R = 220/164, descending soma sides L/R = 79/78. Native X increases toward the right of the image; the source's L-side cells have larger X coordinates, so the screen-right bulb corresponds to the dataset's L side. This static callout placement says nothing about which side is more active. Renderer version 2 preserves both side anchors and marks the motor population as "Both front legs". Completion checks invalidate older render versions while keeping saved recordings available for rerendering.

## Helpers, provenance and limits

- `simtoolreal_shared/activity_trace.py` owns config defaults, completion checks, non-mutating state copies and episode-local video-clock sampling.
- `simtoolreal_shared/anatomical_activity.py` validates/downloads the trace's ordered body IDs, converts 8 nm SWC coordinates to micrometers, caches sparse per-neuron raster masks, streams MP4s and verifies FPS/frame counts with `ffprobe`.
- `simtoolreal_shared/activity_interpretation.py` verifies the circuit artifact's ordered IDs/hash, derives population masks and source-backed captions, adds static labels/leg guides, and computes the four activity summaries. It is imported by the renderer; it has no separate CLI.
- `dextoolbench/eval_worker_isaacgym.py` records actual recurrent activity while executing the policy. The parent renders after this child exits, freeing GPU memory first. It is normally dispatched by the entrypoints rather than launched manually.
- `connectome_network_builder.py` calls an optional observer after each native/fused tanh substep. No checkpoint parameters/buffers are added. Tests verify exact action/final-state equality on CPU and CUDA.

Geometry comes from the [official MaleCNS v1.0 skeleton inventory](https://male-cns.janelia.org/download/), matched by `bodyId`; FlyEM/Janelia, CC BY 4.0. Every source URL/hash is recorded. Missing or invalid skeletons fail explicitly without artificial positions. Verified compact-circuit coverage: 1,952/1,952 neurons, 194,668,453 source bytes.

Recording supports one-environment, no-gradient tanh inference. LSTM, LIF and frozen CUDA-graph reservoir inference are rejected explicitly. Rendering is an offline CPU stage; full-CNS rendering capacity has not been evaluated.

## Verified first batch

The 2026-09-18 annotated batch completed for both compact all-neuron actors at target 7,000,000,000 / actual 7,000,031,232 frames: marker `write_c`, eraser `wipe_smile`, and spatula `flip_over`, one saved-case replay episode per cell. All six cases have circuit-only and combined MP4s (12 total), 1600 × 900 at 80 fps versus 20 fps robot footage, 800 or 804 output frames, and matched per-case durations of 10 or 10.05 seconds. Recorded states are finite, ordered body IDs/population masks match the circuit artifact, and the six traces contain 2,393–2,405 substep/initial-state rows each.

Every output frame decoded successfully, and previews from all six encoded combined videos plus the circuit-only examples were inspected. The interpretability rerender reused all six original recordings; trace and robot-footage hashes remained unchanged. Generated evidence is in `evals/connectome/anatomical_1952_7b/verification.json`, `recording_provenance.json`, `preview_montage.png`, and each case's `circuit_render.json`. The final cached replay reused completed outputs and regenerated the gallery without simulation or encoding.
