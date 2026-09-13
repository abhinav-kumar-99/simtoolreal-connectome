# Sparse Recurrent Backend Benchmark

The current PyTorch actor supports four numerically equivalent recurrent operators. `torch_sparse` was the fastest local actor backend, while native CSR remains the dependency-free default.

Last updated: 2026-09-13

Related: [Actor](../concepts/connectome-actor.md), [Workflow](../workflows/connectome-experiments.md)

## Matched local results

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

## Backend decision

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
