"""Diagnostic spectral monitoring for synaptic-plasticity connectome runs.

Computes top-k eigenvalue magnitudes and singular values of a detached sparse
recurrent operator. Never participates in the training graph.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigs, svds


def _as_csr(
    crow: np.ndarray,
    col: np.ndarray,
    values: np.ndarray,
    neuron_count: int,
) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(
        (values.astype(np.float64, copy=False), col, crow),
        shape=(neuron_count, neuron_count),
    )
    matrix.eliminate_zeros()
    return matrix


def _top_eigenvalue_magnitudes(matrix: sparse.csr_matrix, top_k: int) -> np.ndarray:
    n = matrix.shape[0]
    k = int(min(max(top_k, 1), max(1, n)))
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if matrix.nnz == 0:
        return np.zeros(k, dtype=np.float64)
    # Dense path for compact MaleCNS graphs (~2k); iterative ARPACK for larger CSR.
    if n <= 2048:
        magnitudes = np.abs(np.linalg.eigvals(matrix.toarray()))
        magnitudes.sort()
        magnitudes = magnitudes[::-1][:k]
    else:
        k_iter = int(min(k, max(1, n - 2)))
        try:
            values = eigs(
                matrix,
                k=k_iter,
                which="LM",
                return_eigenvectors=False,
                maxiter=max(300, 6 * n),
                tol=1.0e-5,
            )
            magnitudes = np.abs(values)
            magnitudes.sort()
            magnitudes = magnitudes[::-1]
        except (ArpackNoConvergence, ValueError, RuntimeError):
            magnitudes = np.abs(np.linalg.eigvals(matrix.toarray()))
            magnitudes.sort()
            magnitudes = magnitudes[::-1][:k]
    if magnitudes.size < top_k:
        magnitudes = np.pad(magnitudes, (0, top_k - magnitudes.size))
    return magnitudes[:top_k]


def _top_singular_values(matrix: sparse.csr_matrix, top_k: int) -> np.ndarray:
    n = matrix.shape[0]
    k = int(min(max(top_k, 1), max(1, n)))
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if matrix.nnz == 0:
        return np.zeros(k, dtype=np.float64)
    if n <= 2048:
        singular = np.linalg.svd(matrix.toarray(), compute_uv=False)[:k]
    else:
        k_iter = int(min(k, max(1, n - 1)))
        try:
            _, singular, _ = svds(
                matrix.astype(np.float64),
                k=k_iter,
                which="LM",
                maxiter=max(300, 6 * n),
                tol=1.0e-5,
                return_singular_vectors=True,
            )
            singular = np.sort(np.abs(singular))[::-1]
        except (ArpackNoConvergence, ValueError, RuntimeError):
            singular = np.linalg.svd(matrix.toarray(), compute_uv=False)[:k]
    if singular.size < top_k:
        singular = np.pad(singular, (0, top_k - singular.size))
    return singular[:top_k]


def spectral_metrics(
    crow: np.ndarray,
    col: np.ndarray,
    values: np.ndarray,
    neuron_count: int,
    top_k: int,
) -> Dict[str, float]:
    """Return spectral_radius, max singular value, and top-k lists."""
    matrix = _as_csr(crow, col, values, neuron_count)
    eig_mags = _top_eigenvalue_magnitudes(matrix, top_k)
    singular = _top_singular_values(matrix, top_k)
    return {
        "spectral_radius": float(eig_mags[0]) if eig_mags.size else 0.0,
        "max_singular_value": float(singular[0]) if singular.size else 0.0,
        "top_eigenvalue_magnitudes": [float(x) for x in eig_mags],
        "top_singular_values": [float(x) for x in singular],
    }


def flatten_spectral_tags(
    prefix: str,
    metrics: Mapping[str, object],
    *,
    include_top_lists: bool = True,
) -> Dict[str, float]:
    """Map a metrics dict into TensorBoard tag -> scalar."""
    out: Dict[str, float] = {
        f"{prefix}/spectral_radius": float(metrics["spectral_radius"]),
        f"{prefix}/max_singular_value": float(metrics["max_singular_value"]),
    }
    if include_top_lists:
        for index, value in enumerate(metrics["top_eigenvalue_magnitudes"], start=1):
            out[f"{prefix}/top_eigenvalue_magnitude_{index}"] = float(value)
        for index, value in enumerate(metrics["top_singular_values"], start=1):
            out[f"{prefix}/top_singular_value_{index}"] = float(value)
    return out
