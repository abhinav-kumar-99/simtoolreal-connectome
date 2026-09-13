# Sparse Recurrent Backend Benchmark

The actor supports six recurrent backends. Fused Triton is the fastest measured custom candidate at the smoke-training shape; native CSR remains the dependency-free default.

Last updated: 2026-09-13

Related: [Actor](../concepts/connectome-actor.md), [Workflow](../workflows/connectome-experiments.md)

## Custom-backend experiment: 2026-09-13

`profiles/connectome/custom_backends.json` completed all 160 cases on physical GPU 1 (RTX 4090), using PyTorch 2.4, CUDA 12.4 and Triton 3.0. Five warmup iterations precede 30 timed samples per case. Each case resets model/input seeds; initial state-dictionary hashes match across all backends within each adaptation/shape group. This supersedes the older performance ranking below, while retaining it as historical evidence.

Median actor forward plus backward at `384 x 16`, milliseconds:

| Adaptation, dynamics frozen unless stated | Native CSR | Dense | torch_sparse | cuSPARSE alg2 | Fused Triton |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adapters only | 28.853 | 17.942 | 17.064 | 17.610 | **14.007** |
| Neuron gains | 30.986 | 19.100 | 15.941 | 16.865 | **15.589** |
| Low-rank, rank 4 | 43.153 | 27.680 | 16.650 | 15.910 | **15.202** |
| Independent edges | 42.805 | 27.406 | 17.662 | 15.528 | **14.891** |
| Gains plus learned dynamics | 31.909 | 20.039 | 16.810 | 17.820 | **15.933** |

For adapters-only, Triton is 2.06x faster than native CSR and 1.22x faster than torch_sparse in this measurement. Its rollout batch-384 forward is 0.730 ms and batch-one forward is 0.678 ms. The optimizer-inclusive `384 x 16` median is 14.147 ms. Peak F+B allocated memory is 675.8 MB versus 501.6 MB native CSR and 602.9 MB torch_sparse: adapters-only Triton trades additional saved activation memory for speed. With neuron gains, Triton instead uses less memory than torch_sparse (676.4 versus 822.2 MB).

The unchanged LSTM measured 4.495 ms F+B and 4.938 ms including an optimizer step. Thus the default Triton connectome remains about 3.12x slower than LSTM for sequence training despite its much smaller parameter count. These are synthetic actor-only losses, not SAPG update throughput, environment throughput or learning-quality results.

cuSPARSE alg1/column layout was slower in this sweep. Alg3 gave modest differences for adapters/gains but was slower than alg2 for trainable edge values. Alg2/row layout remains the explicit backend default. Ranks 1/4/8 produced similar Triton F+B latency (15.13/15.20/15.20 ms), while increasing actor parameters from 101,176 to 127,036 to 161,516. This is evidence for shared fixed-topology compute efficiency, not equal learning capacity.

Cold setup/JIT is reported separately. Layout conversions, edge-value construction, gain operations and backward reductions are included. The synthetic batch-one LSTM case uses its supported no-reset-mask path because its legacy reset implementation squeezes a `1 x 1` mask to a scalar. All other shapes retain zero-valued reset masks. GPU timing does not include competing jobs on the benchmark device.

Reproduction and configuration ownership: [custom suites](../workflows/connectome-experiments.md#adaptation-and-custom-kernel-suites).

### Large-shape capacity probe

`profiles/connectome/custom_capacity.json` completed all 32 cases without OOM: adapters-only, low-rank rank 4 and edgewise across native CSR, dense, torch_sparse, cuSPARSE and Triton, plus LSTM. Each uses one warmup and three timed samples, so these are capacity/performance probes rather than the longer 30-sample benchmark above.

Median forward/backward/Adam-update time at `6144 x 16` (98,304 observations), milliseconds:

| Adaptation | Native CSR | Dense | torch_sparse | cuSPARSE alg2 | Fused Triton |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adapters only | 435.252 | 272.579 | 217.271 | 179.378 | **138.432** |
| Low-rank, rank 4 | 780.330 | 341.336 | 247.133 | 204.465 | **167.160** |
| Independent edges | 780.395 | 341.491 | 247.213 | 204.038 | **167.458** |

At this shape, adapters-only Triton is 3.14x faster than native CSR and 1.57x faster than torch_sparse, with 10.491 GB peak allocated memory versus 7.644 and 9.252 GB respectively. Its rollout batch-24,576 forward is 14.40 ms, compared with 55.27 ms native CSR, 29.50 ms torch_sparse and 24.21 ms cuSPARSE. LSTM measured 54.623 ms per synthetic training update and 3.810 GB; the fastest tested connectome therefore remains about 2.53x slower than LSTM for this update.

Triton is the preferred measured performance candidate for these tested modes/shapes. The production default stays native CSR for optional-dependency compatibility; explicitly select `operator_backend: triton_fused` to use the candidate. Actor memory fitting on a 24 GB GPU does not establish that the full environment plus critic and rollout buffers will fit. Full-training configurations require a combined resource preflight before launch.

### Isaac Gym integration gate

All ten cases in `custom_smoke.yaml` completed two SAPG epochs at 384 environments: adapters-only, gains, low-rank rank 4, edgewise, and gains plus learned dynamics, each with cuSPARSE and Triton. `train_dir/connectome/custom_smoke/suite_results.json` contains ten verified results; every checkpoint has epoch 2, populated optimizer state, and a finite `(1, 29)` deployment action after reload. The launcher completed with exit code zero.

The first cuSPARSE job initially trained successfully but parent-process reload failed because only child processes received the vendored `rl_games` import path. The launcher now configures both paths. Verified resumption preserved that completed checkpoint; subsequent runs check resolved-config equality and reject checkpoints below the requested epoch count. All ten checkpoints were reverified with the strengthened epoch check.

These two-epoch runs finished before complete task episodes and do not establish task success, sample efficiency, or long-run numerical stability. No full learning run was launched.

## Historical matched local results: gains plus learned dynamics

The fixed recurrent matrix is `4310 x 4310` with 118,920 edges, or 0.640% density. Every backend used the same actor shape, FP32 recurrent arithmetic, batch of 384, and three warmup plus ten measured iterations on one local RTX 4090. The `384 x 16` case represents 6,144 observations and includes forward and backward time.

| Recurrent backend | Rollout forward, ms | Rollout obs/s | Train forward + backward, ms | Train obs/s | Peak train memory, MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| native CSR | 1.152 | 333,217 | 31.729 | 193,641 | 711.9 |
| native COO | 1.202 | 319,478 | 34.361 | 178,809 | 715.0 |
| dense `torch.mm` | 0.862 | 445,646 | 19.835 | 309,748 | 788.7 |
| `torch_sparse` | **0.780** | **492,418** | **17.327** | **354,601** | 822.2 |

Against native CSR, `torch_sparse` reduced median rollout latency by 32.3% and length-16 forward-plus-backward latency by 45.4%, at the cost of 15.5% more measured peak training memory. Dense execution was second fastest despite the graph's low density. Native COO was slower than CSR.

The separately matched unchanged LSTM profile measured 1.092 ms rollout latency and 4.743 ms length-16 forward-plus-backward latency with 295.1 MB peak training memory. Thus `torch_sparse` makes the connectome faster than the LSTM for one-step actor inference, but connectome backpropagation remains about 3.65 times slower and uses about 2.79 times the peak memory. These synthetic actor measurements do not include Isaac Gym or measure sample efficiency.

Numerical tests load identical parameters into every backend and compare policy means, variances, values, recurrent state, and parameter gradients against native CSR. The recurrent operator is cached after construction and is rebuilt, rather than checkpointed, after a device or dtype move.

## Historical backend decision, before custom kernels

Native CSR remains the production default because it needs no optional compiled package and has already passed the two-epoch Isaac Gym checkpoint/reload smoke gate. `torch_sparse` is the preferred performance candidate, but it should pass the same Isaac Gym smoke suite before becoming a full-training default. The dense control is a reasonable dependency-free performance alternative when the additional resident matrix and training memory fit.

PyTorch officially supports autograd for CSR-times-dense `torch.sparse.mm`. This matches the recurrence: the biological CSR is fixed and gradients are needed only through the dense hidden state and the surrounding gains. NVIDIA cuSPARSE directly supports CSR SpMM, selectable algorithms, and reusable `cusparseSpMM_preprocess` state.

JAX is not presently the preferred migration target. Its official documentation describes `jax.experimental.sparse` as experimental reference code, not recommended for performance-critical applications, and no longer actively developed; BCSR remains under development. A partial JAX actor would also cross the legacy Isaac Gym/PyTorch boundary every rollout step, while a complete migration would require rewriting SAPG and checkpoint/deployment integration.

`torch-sparse` provides the best measured result here and advertises GPU sparse-dense operations with autograd. CuPy exposes CSR operations but is not a drop-in autograd backend. Triton can implement custom GPU kernels but does not provide a general drop-in irregular-CSR recurrent layer.

## Reproduce

For the repository's PyTorch 2.4 and CUDA 12.4 environment, install the optional matching binary wheels into the ignored virtual environment:

```bash
.venv/bin/python -m pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

Run the YAML-owned preparation and profiling stages:

```bash
.venv/bin/python scripts/run_connectome_suite.py \
  --config configs/connectome/suites/backend_profiling.yaml
```

`configs/connectome/profiling/recurrent_backends.yaml` owns actors, shapes, warmups, measured iterations, AMP, seed, and output path. `configs/connectome/suites/backend_profiling.yaml` owns the preparation stage and GPU assignment. The generated JSON is `profiles/connectome/recurrent_backends.json` and remains ignored.

## Primary references

- [JAX experimental sparse documentation](https://docs.jax.dev/en/latest/jax.experimental.sparse.html)
- [PyTorch `torch.sparse.mm` documentation](https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html)
- [NVIDIA cuSPARSE SpMM documentation](https://docs.nvidia.com/cuda/cusparse/)
- [`torch-sparse` optimized autograd operations](https://github.com/rusty1s/pytorch_sparse)
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
