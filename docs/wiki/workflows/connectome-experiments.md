# Connectome Experiment Workflow

Experiments are owned by YAML contracts and proceed through data, profile, smoke, and full-training gates.

Last updated: 2026-09-13

Related: [Overview](../overview.md), [Actor](../concepts/connectome-actor.md)

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

## Configuration ownership

`configs/connectome/malecns_4310.yaml` owns source URLs and hashes, expected graph dimensions, normalization, deterministic control seeds, and artifact paths. Hydra train profiles under `isaacgymenvs/cfg/train/` own network construction and SAPG settings. `params.network.connectome` owns graph variant, interface partitions, adapters, recurrent dynamics, plasticity, numerical dtype, and validation counts.

Suite YAML owns the selected train profiles, seeds, GPU assignment, environment count, SAPG block size, epochs, minibatch, output directory, W&B settings, environment overrides, and checkpoint mode. The launcher deliberately exposes only `--config`; experiment changes belong in YAML.

## Outputs and completion

Raw downloads, NPZ graphs, profiler JSON, W&B data, and training directories are ignored. Each training run writes its resolved configuration, combined log, checkpoints, and `verification.json`. A run is complete only after the child process exits zero, a checkpoint is present, its optimizer state is non-empty, and `deployment.RlPlayer` reloads it to produce a finite `(1, 29)` action.

The checked smoke gate met all four conditions. Its suite result is generated under `train_dir/connectome/smoke/`; the full actor measurements are generated under `profiles/connectome/`. Both locations remain untracked by design.

## Profiling interpretation

On the local RTX 4090, the connectome actor uses 109,796 trainable parameters versus 7,811,468 for the LSTM. At batch 384, median one-step latency was 1.38 ms for the connectome and 1.50 ms for the LSTM. At 384 sequences by 16 steps, connectome forward/backward latencies were 20.20/29.49 ms and peak memory was 713.8 MB; the LSTM measured 2.18/3.41 ms and 295.1 MB. These are local synthetic actor-only measurements, not environment throughput or sample-efficiency results.
