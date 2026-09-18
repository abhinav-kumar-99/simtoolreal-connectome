from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import sparse
from scipy.sparse.linalg import eigs

from simtoolreal_shared.connectome_data import (
    _random_edges,
    _raw_values_for_random,
    _read_adjacency,
    _read_neuron_table,
    _rewire_degree_preserving,
    _validate_signs,
    load_artifact,
    sha256_file,
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
    assert np.array_equal(
        _degrees(sources, rewired, 4)[0], _degrees(sources, destinations, 4)[0]
    )
    assert np.array_equal(
        _degrees(sources, rewired, 4)[1], _degrees(sources, destinations, 4)[1]
    )
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
    transmitters = ["unclear" if index % 2 else "acetylcholine" for index in range(20)]
    randomized = _raw_values_for_random(first[0], values, transmitters, 13)
    assert np.array_equal(np.sort(np.abs(randomized)), values)
    assert np.all(
        randomized[np.asarray([transmitters[i] != "acetylcholine" for i in first[0]])]
        < 0
    )


def _artifact_edges(artifact):
    destinations = np.repeat(
        np.arange(len(artifact["crow_indices"]) - 1), np.diff(artifact["crow_indices"])
    )
    return artifact["col_indices"], destinations


def _sorted_source_weights(sources, values):
    order = np.lexsort((values, sources))
    return np.stack((sources[order], values[order]), axis=1)


def test_prepared_pinned_artifacts_and_controls() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config_path = repository_root / "configs/connectome/malecns_4310.yaml"
    config = yaml.safe_load(config_path.read_text())
    raw_directory = repository_root / config["paths"]["raw_directory"]
    artifact_directory = repository_root / config["paths"]["output_directory"]
    if not artifact_directory.exists():
        pytest.skip(
            "Run the MaleCNS preparation YAML before artifact integration tests"
        )

    for source in config["source"]["files"].values():
        assert sha256_file(raw_directory / source["filename"]) == source["sha256"]
    body_ids, classes, transmitters = _read_neuron_table(
        raw_directory / config["source"]["files"]["neurons"]["filename"]
    )
    source_rows, source_columns, source_values = _read_adjacency(
        raw_directory / config["source"]["files"]["adjacency"]["filename"], body_ids
    )
    _validate_signs(source_rows, source_values, transmitters)

    manifest = json.loads(
        (repository_root / config["paths"]["provenance_manifest"]).read_text()
    )
    artifacts = {
        name: load_artifact(artifact_directory / f"{name}.npz")
        for name in ("biological", "degree_preserving_rewired", "random")
    }
    expected = config["expected"]
    for name, artifact in artifacts.items():
        assert len(artifact["body_ids"]) == expected["neurons"]
        assert len(artifact["values"]) == expected["edges"]
        assert len(artifact["sensory_indices"]) == expected["sensory_neurons"]
        assert len(artifact["descending_indices"]) == expected["descending_neurons"]
        assert len(artifact["motor_indices"]) == expected["motor_neurons"]
        assert (
            sha256_file(artifact_directory / f"{name}.npz")
            == manifest["artifacts"][name]["sha256"]
        )
        matrix = sparse.csr_matrix(
            (artifact["values"], artifact["col_indices"], artifact["crow_indices"]),
            shape=(expected["neurons"], expected["neurons"]),
        )
        radius = np.max(
            np.abs(
                eigs(
                    matrix,
                    k=8,
                    which="LM",
                    v0=np.ones(expected["neurons"]),
                    return_eigenvectors=False,
                    ncv=50,
                    tol=1.0e-10,
                )
            )
        )
        assert radius == pytest.approx(1.0, abs=config["normalization"]["tolerance"])
        variant_sources, _ = _artifact_edges(artifact)
        _validate_signs(variant_sources, artifact["raw_values"], transmitters)
        for population in ("sensory", "descending", "motor"):
            assert np.array_equal(
                artifact[f"{population}_indices"],
                artifacts["biological"][f"{population}_indices"],
            )

    biological_sources, biological_destinations = _artifact_edges(
        artifacts["biological"]
    )
    source_matrix = sparse.coo_matrix(
        (source_values, (source_columns, source_rows)),
        shape=(expected["neurons"], expected["neurons"]),
    ).tocsr()
    source_matrix.sort_indices()
    assert np.array_equal(source_matrix.indices, biological_sources)
    assert np.array_equal(source_matrix.indptr, artifacts["biological"]["crow_indices"])
    assert np.array_equal(source_matrix.data, artifacts["biological"]["raw_values"])

    rewired_sources, rewired_destinations = _artifact_edges(
        artifacts["degree_preserving_rewired"]
    )
    assert np.array_equal(
        _degrees(biological_sources, biological_destinations, expected["neurons"])[0],
        _degrees(rewired_sources, rewired_destinations, expected["neurons"])[0],
    )
    assert np.array_equal(
        _degrees(biological_sources, biological_destinations, expected["neurons"])[1],
        _degrees(rewired_sources, rewired_destinations, expected["neurons"])[1],
    )
    assert np.array_equal(
        _sorted_source_weights(
            biological_sources, artifacts["biological"]["raw_values"]
        ),
        _sorted_source_weights(
            rewired_sources, artifacts["degree_preserving_rewired"]["raw_values"]
        ),
    )
    assert np.array_equal(
        np.sort(np.abs(artifacts["biological"]["raw_values"])),
        np.sort(np.abs(artifacts["random"]["raw_values"])),
    )
