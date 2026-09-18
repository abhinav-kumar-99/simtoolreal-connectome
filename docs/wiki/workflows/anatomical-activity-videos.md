# Anatomical activation videos

Record the compact MaleCNS actor's recurrent activity and render it on real anatomical skeletons alongside a synchronized robot rollout.

Last updated: 2026-09-18

Related: [Experiment workflow](connectome-experiments.md), [Connectome actor](../concepts/connectome-actor.md), [MaleCNS source](../sources/summaries/malecns-front-leg-circuit.md)

## Run the entrypoints

From the normal repository checkout, replay the six paper-tolerance cases for both all-neuron actors at the shared 7B milestone:

```bash
.venv/bin/python scripts/replay_connectome_activity.py --config configs/connectome/visualization/replay_1952_7b.yaml
```

The YAML owns `case_paths`, `output_directory`, one physical `gpu`, `runtime_environment`, and `circuit`. Each source `case.yaml` supplies its checkpoint, resolved policy config, trajectory, tolerance, episode count, camera sampling and simulator overrides. Replay writes new robot footage, `activity.npz`, `circuit.mp4`, `rollout_with_circuit.mp4`, and `circuit_render.json` into the new directory. Existing videos are preserved. Activity matches the newly replayed footage; exact reproduction of an old MP4 is not assumed. Re-running reuses complete cases with matching circuit settings.

To change appearance without rerunning Isaac Gym:

```bash
.venv/bin/python scripts/render_connectome_activity.py --config configs/connectome/visualization/render_1952_example.yaml
```

This YAML owns `trace_path`, `rollout_path`, `output_directory`, and `circuit`. Paths are absolute or relative to the repository root. Generated geometry and videos are ignored by Git.

For future ordinary or milestone evaluations, add this block to `videos` (under `evaluation.videos` in watcher configs):

```yaml
circuit:
  enabled: true
  fps_multiplier: 4
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

## Timing and interpretation

Four neural updates per 60 Hz control step yield an effective 240 substeps/s. Recording every third robot step at 20 fps and rendering at 80 fps normally selects every third recorded neural state. The renderer uses the latest available state on the video clock and repeats each robot image four times. Duration stays equal. Final camera-frame padding holds the last state; episodes restart from their recorded initial state without blending across resets.

The fixed tanh scale is −1 to +1. Cyan/orange indicate positive/negative modeled state, not excitatory/inhibitory anatomical neuron identity. One scalar colors each whole morphology. Overlapping projected neurons blend contributions with fixed density compensation; activity is never normalized per frame. This is real anatomy carrying engineered controller state, with no modeled within-neuron propagation or claim of measured biological activity.

## Helpers, provenance and limits

- `simtoolreal_shared/activity_trace.py` owns config defaults, completion checks, non-mutating state copies and episode-local video-clock sampling.
- `simtoolreal_shared/anatomical_activity.py` validates/downloads the trace's ordered body IDs, converts 8 nm SWC coordinates to micrometers, caches sparse per-neuron raster masks, streams MP4s and verifies FPS/frame counts with `ffprobe`.
- `dextoolbench/eval_worker_isaacgym.py` records actual recurrent activity while executing the policy. The parent renders after this child exits, freeing GPU memory first. It is normally dispatched by the entrypoints rather than launched manually.
- `connectome_network_builder.py` calls an optional observer after each native/fused tanh substep. No checkpoint parameters/buffers are added. Tests verify exact action/final-state equality on CPU and CUDA.

Geometry comes from the [official MaleCNS v1.0 skeleton inventory](https://male-cns.janelia.org/download/), matched by `bodyId`; FlyEM/Janelia, CC BY 4.0. Every source URL/hash is recorded. Missing or invalid skeletons fail explicitly without artificial positions. Verified compact-circuit coverage: 1,952/1,952 neurons, 194,668,453 source bytes.

Recording supports one-environment, no-gradient tanh inference. LSTM, LIF and frozen CUDA-graph reservoir inference are rejected explicitly. Rendering is an offline CPU stage; full-CNS rendering capacity has not been evaluated.
