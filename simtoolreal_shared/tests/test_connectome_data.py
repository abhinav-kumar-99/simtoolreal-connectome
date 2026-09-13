from __future__ import annotations

import numpy as np

from simtoolreal_shared.connectome_data import (
    _random_edges,
    _raw_values_for_random,
    _rewire_degree_preserving,
)


def _degrees(sources: np.ndarray, destinations: np.ndarray, count: int):
    return (
        np.bincount(sources, minlength=count),
        np.bincount(destinations, minlength=count),
    )


def test_degree_preserving_rewire_retains_degrees_and_source_weights() -> None:
    sources = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    destinations = np.asarray([1, 2, 2, 3, 0, 3, 0, 1], dtype=np.int64)
    values = np.asarray([1, 2, -3, -4, 5, 6, -7, -8], dtype=np.float32)
    source_weight_pairs = np.stack((sources, values), axis=1).copy()
    rewired = _rewire_degree_preserving(sources, destinations, 7, 2, 100)
    repeated = _rewire_degree_preserving(sources, destinations, 7, 2, 100)
    assert np.array_equal(rewired, repeated)
    assert np.any(rewired != destinations)
    assert np.array_equal(_degrees(sources, rewired, 4)[0], _degrees(sources, destinations, 4)[0])
    assert np.array_equal(_degrees(sources, rewired, 4)[1], _degrees(sources, destinations, 4)[1])
    assert len(set(zip(sources.tolist(), rewired.tolist()))) == len(sources)
    # Rewiring only permutes destinations, so every edge retains both its source
    # neuron and signed weight. This is the source-specific weight-multiset invariant.
    assert np.array_equal(source_weight_pairs, np.stack((sources, values), axis=1))


def test_random_graph_is_deterministic_unique_and_preserves_magnitudes() -> None:
    first = _random_edges(20, 80, 11)
    second = _random_edges(20, 80, 11)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert len(set(zip(first[0].tolist(), first[1].tolist()))) == 80
    assert np.all(first[0] != first[1])
    values = np.arange(1, 81, dtype=np.float32)
    transmitters = ["gaba" if index % 2 else "acetylcholine" for index in range(20)]
    randomized = _raw_values_for_random(first[0], values, transmitters, 13)
    assert np.array_equal(np.sort(np.abs(randomized)), values)
    assert np.all(randomized[np.asarray([transmitters[i] == "gaba" for i in first[0]])] < 0)
