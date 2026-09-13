"""Reproducible MaleCNS graph preparation and artifact loading."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import sparse
from scipy.sparse.linalg import eigs


ARTIFACT_SCHEMA_VERSION = 1


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) == expected_sha256:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    actual = sha256_file(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}"
        )
    temporary.replace(destination)
    return destination


def _read_neuron_table(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    body_ids = np.asarray([int(row["bodyId"]) for row in rows], dtype=np.int64)
    classes = [row["class"].strip() for row in rows]
    transmitters = [row["consensusNt"].strip().lower() for row in rows]
    return body_ids, classes, transmitters


def _read_adjacency(path: Path, expected_body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        column_ids = np.asarray([int(value) for value in header[1:]], dtype=np.int64)
        row_ids: list[int] = []
        sources: list[np.ndarray] = []
        destinations: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for source_index, row in enumerate(reader):
            row_ids.append(int(row[0]))
            dense_row = np.asarray(row[1:], dtype=np.float32)
            nonzero = np.flatnonzero(dense_row)
            if nonzero.size:
                sources.append(np.full(nonzero.size, source_index, dtype=np.int64))
                destinations.append(nonzero.astype(np.int64, copy=False))
                values.append(dense_row[nonzero])
    row_ids_array = np.asarray(row_ids, dtype=np.int64)
    if not np.array_equal(row_ids_array, expected_body_ids):
        raise ValueError("Adjacency row body IDs do not match neuron-table order")
    if not np.array_equal(column_ids, expected_body_ids):
        raise ValueError("Adjacency column body IDs do not match neuron-table order")
    return np.concatenate(sources), np.concatenate(destinations), np.concatenate(values)


def _validate_signs(sources: np.ndarray, values: np.ndarray, transmitters: list[str]) -> None:
    for source, value in zip(sources, values):
        transmitter = transmitters[int(source)]
        if transmitter == "acetylcholine" and value <= 0:
            raise ValueError(f"Non-positive cholinergic edge from neuron index {source}")
        if transmitter != "acetylcholine" and value >= 0:
            raise ValueError(f"Non-negative non-cholinergic edge from neuron index {source}")


def _spectral_normalize(
    sources: np.ndarray,
    destinations: np.ndarray,
    raw_values: np.ndarray,
    neuron_count: int,
    target: float,
) -> tuple[sparse.csr_matrix, float, float]:
    # Source CSV rows encode presynaptic neurons. The runtime operator is W[post, pre].
    matrix = sparse.coo_matrix(
        (raw_values.astype(np.float64), (destinations, sources)),
        shape=(neuron_count, neuron_count),
    ).tocsr()
    eigenvalues = eigs(
        matrix,
        k=8,
        which="LM",
        v0=np.ones(neuron_count, dtype=np.float64),
        return_eigenvectors=False,
        maxiter=max(100000, neuron_count * 20),
        ncv=50,
        tol=1.0e-10,
    )
    # ARPACK can return a subdominant member of a near-degenerate complex
    # spectrum for k=1. Taking the maximum of a fixed candidate set is stable;
    # rounding removes backend-level last-bit differences before FP32 scaling.
    radius = float(round(float(np.max(np.abs(eigenvalues))), 8))
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"Invalid spectral radius: {radius}")
    scale = float(target) / radius
    normalized = matrix.astype(np.float32)
    normalized.data *= np.float32(scale)
    normalized.sort_indices()
    return normalized, radius, scale


def _rewire_degree_preserving(
    sources: np.ndarray,
    destinations: np.ndarray,
    seed: int,
    swaps_per_edge: int,
    maximum_attempts_per_swap: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rewired = destinations.copy()
    edge_set = set(zip(sources.tolist(), rewired.tolist()))
    target_swaps = int(len(sources) * swaps_per_edge)
    maximum_attempts = int(target_swaps * maximum_attempts_per_swap)
    completed = 0
    attempts = 0
    while completed < target_swaps and attempts < maximum_attempts:
        attempts += 1
        first, second = rng.integers(0, len(sources), size=2)
        if first == second:
            continue
        src_a, dst_a = int(sources[first]), int(rewired[first])
        src_b, dst_b = int(sources[second]), int(rewired[second])
        if src_a == src_b or dst_a == dst_b or src_a == dst_b or src_b == dst_a:
            continue
        new_a = (src_a, dst_b)
        new_b = (src_b, dst_a)
        if new_a in edge_set or new_b in edge_set:
            continue
        edge_set.remove((src_a, dst_a))
        edge_set.remove((src_b, dst_b))
        edge_set.add(new_a)
        edge_set.add(new_b)
        rewired[first], rewired[second] = dst_b, dst_a
        completed += 1
    if completed != target_swaps:
        raise RuntimeError(
            f"Completed {completed}/{target_swaps} degree-preserving swaps "
            f"after {attempts} attempts"
        )
    return rewired


def _random_edges(neuron_count: int, edge_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    maximum = neuron_count * neuron_count
    while len(chosen) < edge_count:
        needed = edge_count - len(chosen)
        candidates = rng.integers(0, maximum, size=max(needed * 2, 1024))
        for encoded in candidates.tolist():
            source, destination = divmod(int(encoded), neuron_count)
            if source != destination:
                chosen.add(int(encoded))
            if len(chosen) == edge_count:
                break
    encoded = np.asarray(sorted(chosen), dtype=np.int64)
    return encoded // neuron_count, encoded % neuron_count


def _raw_values_for_random(
    sources: np.ndarray,
    original_values: np.ndarray,
    transmitters: list[str],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    magnitudes = np.abs(original_values).copy()
    rng.shuffle(magnitudes)
    signs = np.ones(len(sources), dtype=np.float32)
    inhibitory = np.asarray([transmitters[int(source)] != "acetylcholine" for source in sources])
    signs[inhibitory] = -1.0
    return magnitudes * signs


def _save_artifact(
    path: Path,
    matrix: sparse.csr_matrix,
    raw_values: np.ndarray,
    body_ids: np.ndarray,
    population_indices: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "schema_version": np.asarray([ARTIFACT_SCHEMA_VERSION], dtype=np.int64),
        "crow_indices": matrix.indptr.astype(np.int64),
        "col_indices": matrix.indices.astype(np.int64),
        "values": matrix.data.astype(np.float32),
        "raw_values": np.asarray(raw_values, dtype=np.float32),
        "body_ids": body_ids,
        "sensory_indices": population_indices["sensory"],
        "descending_indices": population_indices["descending"],
        "motor_indices": population_indices["motor"],
    }
    temporary = path.with_suffix(path.suffix + ".part")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.ascontiguousarray(arrays[name]),
                version=(1, 0),
                allow_pickle=False,
            )
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    temporary.replace(path)


def prepare_connectome(config_path: Path, repository_root: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    raw_dir = repository_root / config["paths"]["raw_directory"]
    output_dir = repository_root / config["paths"]["output_directory"]
    resolved_files = {}
    for key, file_config in config["source"]["files"].items():
        resolved_files[key] = download_verified(
            file_config["url"], raw_dir / file_config["filename"], file_config["sha256"]
        )

    body_ids, classes, transmitters = _read_neuron_table(resolved_files["neurons"])
    sources, destinations, raw_values = _read_adjacency(
        resolved_files["adjacency"], body_ids
    )
    expected = config["expected"]
    population_indices = {
        "sensory": np.flatnonzero(np.asarray(classes) == "sensory neuron").astype(np.int64),
        "descending": np.flatnonzero(np.asarray(classes) == "descending neuron").astype(np.int64),
        "motor": np.flatnonzero(np.asarray(classes) == "motor neuron").astype(np.int64),
    }
    observed = {
        "neurons": int(len(body_ids)),
        "edges": int(len(raw_values)),
        "sensory_neurons": int(len(population_indices["sensory"])),
        "descending_neurons": int(len(population_indices["descending"])),
        "motor_neurons": int(len(population_indices["motor"])),
    }
    if observed != expected:
        raise ValueError(f"MaleCNS count mismatch: expected {expected}, observed {observed}")
    _validate_signs(sources, raw_values, transmitters)

    target = float(config["normalization"]["target"])
    variants: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "biological": (sources, destinations, raw_values)
    }
    controls = config["controls"]
    seed = int(controls["seed"])
    if "degree_preserving_rewired" in controls["variants"]:
        variants["degree_preserving_rewired"] = (
            sources,
            _rewire_degree_preserving(
                sources,
                destinations,
                seed,
                int(controls["rewiring_swaps_per_edge"]),
                int(controls["maximum_attempts_per_swap"]),
            ),
            raw_values,
        )
    if "random" in controls["variants"]:
        random_sources, random_destinations = _random_edges(
            len(body_ids), len(raw_values), seed + 1
        )
        variants["random"] = (
            random_sources,
            random_destinations,
            _raw_values_for_random(random_sources, raw_values, transmitters, seed + 2),
        )

    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source": config["source"],
        "observed": observed,
        "class_counts": dict(sorted(Counter(classes).items())),
        "transmitter_counts": dict(sorted(Counter(transmitters).items())),
        "orientation": "CSR rows are postsynaptic destinations; columns are presynaptic sources",
        "normalization": config["normalization"],
        "controls": config["controls"],
        "artifacts": {},
    }
    for variant, (variant_sources, variant_destinations, variant_raw_values) in variants.items():
        matrix, radius, scale = _spectral_normalize(
            variant_sources,
            variant_destinations,
            variant_raw_values,
            len(body_ids),
            target,
        )
        artifact_path = output_dir / f"{variant}.npz"
        # CSR sorting changes edge order, so store raw values in CSR order too.
        raw_csr = sparse.coo_matrix(
            (variant_raw_values, (variant_destinations, variant_sources)),
            shape=matrix.shape,
        ).tocsr()
        raw_csr.sort_indices()
        _save_artifact(artifact_path, matrix, raw_csr.data, body_ids, population_indices)
        manifest["artifacts"][variant] = {
            "path": str(artifact_path.relative_to(repository_root)),
            "sha256": sha256_file(artifact_path),
            "raw_spectral_radius": radius,
            "normalization_scale": scale,
            "nnz": int(matrix.nnz),
        }
    manifest_path = repository_root / config["paths"]["provenance_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    schema = int(result["schema_version"][0])
    if schema != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported connectome artifact schema {schema}")
    return result
