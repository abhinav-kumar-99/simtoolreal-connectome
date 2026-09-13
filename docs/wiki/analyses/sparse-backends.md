# Sparse Recurrent Backend Options

The current PyTorch CSR implementation is the lowest-risk baseline; optimize and benchmark its kernel boundary before considering a JAX rewrite.

Last updated: 2026-09-13

Related: [Actor](../concepts/connectome-actor.md), [Workflow](../workflows/connectome-experiments.md)

## Current evidence

The fixed recurrent matrix is `4310 x 4310` with 118,920 edges, or 0.640% density. On the local RTX 4090, the actor-only length-16 profile measured 20.20 ms forward, 29.49 ms backward, and 713.8 MB peak allocated memory for the connectome. The LSTM measured 2.18 ms forward, 3.41 ms backward, and 295.1 MB. Sparse training performance, rather than parameter count, is therefore the current bottleneck.

PyTorch officially supports autograd for CSR-times-dense `torch.sparse.mm`. This matches the recurrence: the biological CSR is fixed and gradients are needed only through the dense hidden state and the surrounding gains. NVIDIA cuSPARSE directly supports CSR SpMM, selectable algorithms, and reusable `cusparseSpMM_preprocess` state.

JAX is not presently the preferred migration target. Its official documentation describes `jax.experimental.sparse` as experimental reference code, not recommended for performance-critical applications, and no longer actively developed; BCSR remains under development. A partial JAX actor would also cross the legacy Isaac Gym/PyTorch boundary every rollout step, while a complete migration would require rewriting SAPG and checkpoint/deployment integration.

`torch-sparse` is the most plausible drop-in package to benchmark. It advertises GPU sparse-dense operations with autograd and provides wheels for PyTorch 2.4, but there is no project-specific evidence yet that it beats native CSR on this matrix and batch shape. CuPy exposes CSR operations but is not a drop-in autograd backend. Triton can implement custom GPU kernels but does not provide a general drop-in irregular-CSR recurrent layer.

## Recommended benchmark order

1. Keep the PyTorch implementation and cache the constructed CSR tensor instead of recreating it on every recurrent step.
2. Benchmark 32-bit CSR indices, native CSR algorithm/layout variations, and a dense `torch.mm` control at the exact rollout and `384 x 16` training shapes. At this matrix size, dense tensor-core execution may be competitive despite the low density and must be measured.
3. Benchmark `torch-sparse` behind the same actor interface and numerical-equivalence tests.
4. If native PyTorch remains limiting, implement a fixed-weight custom autograd operator using cuSPARSE: forward uses `W @ H`, backward uses a prebuilt `W^T @ dY`, with descriptors and preprocessing reused across all steps.
5. Consider Triton only for a fused specialized kernel after profiling shows that SpMM plus gain/leak/activation launches dominate. Consider JAX only as part of an independently justified full training-stack port.

## Primary references

- [JAX experimental sparse documentation](https://docs.jax.dev/en/latest/jax.experimental.sparse.html)
- [PyTorch `torch.sparse.mm` documentation](https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html)
- [NVIDIA cuSPARSE SpMM documentation](https://docs.nvidia.com/cuda/cusparse/)
- [`torch-sparse` optimized autograd operations](https://github.com/rusty1s/pytorch_sparse)
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
