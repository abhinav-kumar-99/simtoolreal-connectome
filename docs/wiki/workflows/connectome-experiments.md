# Connectome Experiment Workflow

Experiments are owned by YAML contracts and proceed through data, profile, smoke, and full-training gates.

Last updated: 2026-09-13

Related: [Overview](../overview.md), [Actor](../concepts/connectome-actor.md)

## Adaptation and custom-kernel suites

The primary actor now defaults to adapters-only with frozen leaks/biases and the measured-fastest `triton_fused` recurrent backend. See [adaptation controls](../concepts/connectome-adaptation.md) for all modes, parameter counts and legacy compatibility.

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

`configs/connectome/suites/adaptation_1m.yaml` is the capped pilot entry point. It runs five adaptation policies at seed 42, with at most two independent single-GPU jobs at once: one pinned to GPU 0 and one to GPU 1. Each child retains the published SimToolReal geometry of 24,576 environments, six 4,096-environment SAPG blocks, horizon/sequence length 16, 98,304 actor and central-critic minibatches, and two mini-epochs. Two epochs produce 786,432 environment steps; a third would produce 1,179,648 and violate the strict 1,000,000-step ceiling. Every run writes `timing.json` with child-training, checkpoint-verification and total wall time; the suite result repeats those fields.

Run the capped pilot from the repository root:

```bash
.venv/bin/python scripts/run_connectome_suite.py --config configs/connectome/suites/adaptation_1m.yaml
```

The important pilot keys are `train_profiles` (the five adaptation cases), `seeds`, `gpu_assignments`, `max_parallel`, `num_envs`, `sapg_block_size`, `epochs`, `max_frames`, both minibatch sizes, and the task overrides. `max_parallel: 2` creates two GPU-owned queues, so a device never receives overlapping policies. Set it to `1` for fully serial execution. The launcher accepts only `--config`; all experiment settings stay in YAML.

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

## Configuration ownership

`configs/connectome/malecns_4310.yaml` owns source URLs and hashes, expected graph dimensions, normalization, deterministic control seeds, and artifact paths. Hydra train profiles under `isaacgymenvs/cfg/train/` own network construction and SAPG settings. `params.network.connectome` owns graph variant, interface partitions, adapters, recurrent dynamics, plasticity, numerical dtype, and validation counts.

Suite YAML owns the selected train profiles, seeds, GPU assignment, maximum concurrent jobs, environment count, SAPG block size, epochs, frame cap, minibatch, output directory, W&B settings, environment overrides, and checkpoint mode. The launcher deliberately exposes only `--config`; experiment changes belong in YAML.

## Outputs and completion

Raw downloads, NPZ graphs, profiler JSON, W&B data, and training directories are ignored. Each training run writes its resolved configuration, combined log, checkpoints, and `verification.json`. A run is complete only after the child process exits zero, a checkpoint is present, its optimizer state is non-empty, and `deployment.RlPlayer` reloads it to produce a finite `(1, 29)` action.

The checked smoke gate met all four conditions. Its suite result is generated under `train_dir/connectome/smoke/`; the full actor measurements are generated under `profiles/connectome/`. Both locations remain untracked by design.

## Historical profiling interpretation, before the adapters-only default

On the local RTX 4090, the connectome actor uses 109,796 trainable parameters versus 7,811,468 for the LSTM. In the latest matched profile, native CSR measured 1.155 ms for one-step rollout and 31.950 ms for length-16 forward plus backward, compared with 1.092 ms and 4.743 ms for the LSTM. The backend suite measured `torch_sparse` at 0.780 ms and 17.327 ms respectively. These are local synthetic actor-only measurements, not environment throughput or sample-efficiency results.
