"""Prepare the approved path-plus-sensory-premotor circuit, without a size quota.

Imported by prepare_malecns_connectome.py; all inputs and policies are YAML-owned.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from simtoolreal_shared.connectome_data import (
    _save_artifact, _spectral_normalize, load_yaml, sha256_file,
)


def group_union(neurons, names):
    """Read explicit audit booleans; never treat strings such as 'False' as true."""
    if not names:
        raise ValueError('Population group list must not be empty')
    for name in names:
        if not neurons[name].isin([True, False]).all():
            raise ValueError(f'Invalid boolean membership in {name}')
    return set(neurons.loc[neurons[names].any(axis=1), 'bodyId'].astype(int))


def select_compact(neurons, reference, selection):
    """Exactly reproduce the audited 1,596 paths plus 356 sensory-premotor cells."""
    if not neurons.bodyId.is_unique:
        raise ValueError('Duplicate neuron IDs')
    if reference.duplicated(['source', 'destination']).any():
        raise ValueError('Reference pairs must be aggregated before thresholding')
    threshold = int(selection['minimum_synapses'])
    if threshold < 1:
        raise ValueError('minimum_synapses must be positive')
    ids = np.sort(neurons.bodyId.to_numpy(dtype=np.int64))
    edges = reference[(reference.synapses >= threshold)
                      & reference.source.isin(ids) & reference.destination.isin(ids)]
    graph = sparse.csr_matrix(
        (np.ones(len(edges)), (np.searchsorted(ids, edges.source),
                               np.searchsorted(ids, edges.destination))),
        shape=(len(ids), len(ids)),
    )
    seeds = group_union(neurons, selection['seed_groups'])
    populations = {name: group_union(neurons, groups)
                   for name, groups in selection['population_groups'].items()}
    selected = set(seeds)
    for adjacency, starts in (
        (graph, populations['sensory'] | populations['descending']),
        (graph.T.tocsr(), populations['motor']),
    ):
        distance, previous, _ = dijkstra(
            adjacency, indices=np.searchsorted(ids, sorted(starts)),
            min_only=True, unweighted=True, return_predecessors=True,
        )
        for body in sorted(seeds):
            i = np.searchsorted(ids, body)
            if np.isfinite(distance[i]):
                while i >= 0:
                    selected.add(int(ids[i]))
                    i = previous[i]
    paths = set(selected)
    # Include ALL such observed two-edge motifs, not the top K strongest cells.
    sensory_targets = set(edges.loc[
        edges.source.isin(populations['sensory']), 'destination'])
    premotor_sources = set(edges.loc[
        edges.destination.isin(populations['motor']), 'source'])
    extra = (sensory_targets & premotor_sources) - paths
    selected |= extra
    body_ids = np.asarray(sorted(selected), dtype=np.int64)
    kept = edges[edges.source.isin(selected) & edges.destination.isin(selected)].copy()
    kept = kept.sort_values(['source', 'destination']).reset_index(drop=True)
    indices = {name: np.flatnonzero(np.isin(body_ids, sorted(members))).astype(np.int64)
               for name, members in populations.items()}
    stats = {
        'seed_neurons': len(seeds), 'path_neurons': len(paths),
        'sensory_premotor_additions': len(extra),
        'isolated_neurons': len(selected - (set(kept.source) | set(kept.destination))),
        'body_ids_sha256': hashlib.sha256(body_ids.astype('<i8').tobytes()).hexdigest(),
    }
    membership = neurons[neurons.bodyId.isin(selected)].copy().sort_values('bodyId')
    membership['compact_seed'] = membership.bodyId.isin(seeds)
    membership['compact_path'] = membership.bodyId.isin(paths)
    membership['compact_sensory_premotor_addition'] = membership.bodyId.isin(extra)
    return body_ids, kept, indices, stats, membership


def source_signs(body_ids, transmitters, policy):
    """Preserve the original model convention; uncertainty is explicit, not biology."""
    if not transmitters.body.is_unique:
        raise ValueError('Duplicate neurotransmitter body IDs')
    labels = (transmitters.set_index('body').consensus_nt.reindex(body_ids)
              .fillna('missing').str.lower())
    unexpected = set(labels) - set(policy)
    if unexpected or any(value not in (-1, 1) for value in policy.values()):
        raise ValueError(f'Invalid or unspecified transmitter sign: {unexpected}')
    return labels.map(policy).to_numpy(dtype=np.float32), labels


def prepare_compact_connectome(config_path: Path, repository_root: Path):
    config = load_yaml(config_path)
    if config.get('schema_version') != 1 or config.get('preparation_kind') != 'compact_paths':
        raise ValueError('Unsupported compact preparation contract')
    paths = {}
    provenance = {}
    for name, spec in config['sources'].items():
        path = repository_root / spec['path']
        if sha256_file(path) != spec['sha256']:
            raise ValueError(f'Unexpected source hash: {path}')
        paths[name] = path
        provenance[name] = spec
    neurons = pd.read_csv(paths['neurons']).fillna('')
    reference = pd.read_csv(paths['edges'])
    ids, edges, populations, selection, membership = select_compact(
        neurons, reference, config['selection'])
    observed = {'neurons': len(ids), 'edges': len(edges), **{
        f'{name}_neurons': len(indices) for name, indices in populations.items()}}
    if observed != config['expected'] or selection != config['expected_selection']:
        raise ValueError(f'Approved circuit identity mismatch: {observed}, {selection}')
    signs, labels = source_signs(ids, pd.read_feather(paths['neurotransmitters']),
                                 config['transmitter_signs'])
    src = np.searchsorted(ids, edges.source)
    dst = np.searchsorted(ids, edges.destination)
    raw = edges.synapses.to_numpy(dtype=np.float32) * signs[src]
    matrix, radius, scale = _spectral_normalize(
        src, dst, raw, len(ids), float(config['normalization']['target']))
    raw_csr = sparse.coo_matrix((raw, (dst, src)), shape=matrix.shape).tocsr()
    raw_csr.sort_indices()
    output = repository_root / config['output_directory']
    artifact = output / 'biological.npz'
    # Never replace a different existing graph underneath a running policy.
    if artifact.exists():
        with np.load(artifact) as old:
            for key, value in {'body_ids': ids, 'values': matrix.data,
                               'raw_values': raw_csr.data, 'col_indices': matrix.indices,
                               'crow_indices': matrix.indptr, **{
                                   f'{name}_indices': ix for name, ix in populations.items()
                               }}.items():
                if not np.array_equal(old[key], value):
                    raise FileExistsError(f'Different artifact already exists: {artifact}')
    else:
        _save_artifact(artifact, matrix, raw_csr.data, ids, populations)
    membership['consensus_nt'] = labels.to_numpy()
    membership.to_csv(output / 'neurons.csv', index=False)
    edges.to_csv(output / 'edges.csv.gz', index=False)
    manifest = {
        'schema_version': 1, 'observed': observed, 'selection': selection,
        'sources': provenance, 'config': str(config_path),
        'config_sha256': sha256_file(config_path), 'scipy_version': scipy.__version__,
        'orientation': 'CSR rows postsynaptic; columns presynaptic',
        'interface': config['selection']['population_groups'],
        'transmitter_counts': {str(k): int(v) for k, v in labels.value_counts().items()},
        'transmitter_signs': config['transmitter_signs'],
        'sign_caveat': 'Non-ACh negative, including unclear/missing, is a model assumption.',
        'normalization': {'target': config['normalization']['target'],
                          'raw_spectral_radius': radius, 'scale': scale},
        'artifact': {'path': str(artifact), 'sha256': sha256_file(artifact)},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest
