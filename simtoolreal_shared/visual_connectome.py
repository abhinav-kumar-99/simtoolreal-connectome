"""Extract real MaleCNS optic-to-motor paths, retaining the proprioceptive core."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from simtoolreal_shared.connectome_data import load_yaml, sha256_file, _spectral_normalize


def distances(matrix, seeds, depth):
    distance = np.full(matrix.shape[0], depth + 1, dtype=np.int16)
    frontier = np.zeros(matrix.shape[0], dtype=bool)
    frontier[seeds] = True
    distance[seeds] = 0
    for step in range(1, depth + 1):
        frontier = np.asarray(matrix @ frontier).ravel().astype(bool) & (distance > depth)
        distance[frontier] = step
        if not frontier.any():
            break
    return distance


def prepare_visual_connectome(config_path, repository_root):
    cfg = load_yaml(config_path)
    sources = {}
    for name, spec in cfg['sources'].items():
        path = repository_root / spec['path']
        if sha256_file(path) != spec['sha256']:
            raise ValueError(f'Source identity changed: {path}')
        sources[name] = path
    annotations = pd.read_feather(sources['annotations']).set_index('bodyId')
    weights = pd.read_feather(sources['weights'])
    weights = weights[weights.weight >= cfg['minimum_synapses']]
    ids = np.union1d(weights.body_pre, weights.body_post)
    src = np.searchsorted(ids, weights.body_pre)
    dst = np.searchsorted(ids, weights.body_post)
    graph = sparse.coo_matrix((np.ones(len(src), dtype=bool), (dst, src)),
                              shape=(len(ids), len(ids))).tocsr()
    visual = annotations[annotations.type.isin(cfg['input_cell_types']) &
                         annotations.somaSide.isin(cfg['eye_sides']) &
                         annotations.assignedOlHex1.notna() & annotations.assignedOlHex2.notna()]
    visual = visual.loc[visual.index.intersection(ids)].sort_index()
    with np.load(sources['core']) as core:
        core_arrays = {k: core[k].copy() for k in core.files}
    motor_ids = core_arrays['body_ids'][core_arrays['motor_indices']]
    depth = int(cfg['maximum_path_edges'])
    forward = distances(graph, np.searchsorted(ids, visual.index), depth)
    backward = distances(graph.T.tocsr(), np.searchsorted(ids, motor_ids), depth)
    keep = (forward.astype(np.int32) + backward <= depth)
    selection_mode = cfg.get('selection_mode', 'visual_paths')
    if selection_mode == 'all_traced_neurons':
        # Include every traced neuron, even isolated cells and cells outside the
        # visual/motor paths. Do not silently include glia or unknown endpoints.
        selected = np.sort(annotations.index[annotations.status.eq('Traced')].to_numpy(dtype=np.int64))
    elif selection_mode == 'visual_paths':
        selected = np.union1d(ids[keep], core_arrays['body_ids'])
    else:
        raise ValueError(f'Unknown selection mode: {selection_mode}')
    visual = visual.loc[visual.index.intersection(selected)].sort_index()
    edge_mask = np.isin(weights.body_pre, selected) & np.isin(weights.body_post, selected)
    edges = weights.loc[edge_mask]
    src = np.searchsorted(selected, edges.body_pre)
    dst = np.searchsorted(selected, edges.body_post)
    labels = pd.read_feather(sources['neurotransmitters']).set_index('body').consensus_nt.reindex(selected).fillna('missing').str.lower()
    if set(labels) - set(cfg['transmitter_signs']):
        raise ValueError('Unspecified neurotransmitter sign')
    signs = labels.map(cfg['transmitter_signs']).to_numpy(dtype=np.float32)
    raw = sparse.coo_matrix((edges.weight.to_numpy(dtype=np.float32) * signs[src], (dst, src)),
                            shape=(len(selected), len(selected))).tocsr()
    matrix, radius, scale = _spectral_normalize(src, dst,
        edges.weight.to_numpy(dtype=np.float32) * signs[src], len(selected), cfg['normalization_target'])
    ports = ['sensory_indices', 'descending_indices', 'motor_indices',
             'front_proprioceptors_indices', 'front_tactile_indices']
    for key in ports:
        if not np.isin(core_arrays['body_ids'][core_arrays[key]], selected).all():
            raise ValueError(f'Selection omits an interface neuron: {key}')
    arrays = {k: np.searchsorted(selected, core_arrays['body_ids'][core_arrays[k]])
              for k in ['sensory_indices', 'descending_indices', 'motor_indices',
                        'front_proprioceptors_indices', 'front_tactile_indices']}
    arrays['visual_indices'] = np.searchsorted(selected, visual.index.to_numpy())
    arrays['sensory_indices'] = np.union1d(arrays['sensory_indices'], arrays['visual_indices'])
    # Column coordinates are anatomical. Mapping their bounding box to this
    # external robot camera is an explicit engineering alignment, not eye optics.
    xy = visual[['assignedOlHex1', 'assignedOlHex2']].to_numpy(dtype=np.float32)
    xy = np.column_stack((xy[:, 0] - .5 * xy[:, 1], np.sqrt(3.) / 2 * xy[:, 1]))
    for side in cfg['eye_sides']:
        mask = visual.somaSide.to_numpy() == side
        if mask.any():
            xy[mask] = 2 * (xy[mask] - xy[mask].min(0)) / np.maximum(np.ptp(xy[mask], axis=0), 1) - 1
    arrays.update(body_ids=selected, crow_indices=matrix.indptr.astype(np.int64),
                  col_indices=matrix.indices.astype(np.int64), values=matrix.data,
                  raw_values=raw.data, visual_grid=xy.astype(np.float32), schema_version=np.array([1]))
    output = repository_root / cfg['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / 'biological.npz'
    if artifact.exists():
        with np.load(artifact) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise FileExistsError(f'Different artifact exists: {artifact}')
    else:
        np.savez_compressed(artifact, **arrays)
    observed = dict(neurons=len(selected), edges=matrix.nnz, sensory_neurons=len(arrays['sensory_indices']),
                    descending_neurons=len(arrays['descending_indices']), motor_neurons=len(arrays['motor_indices']))
    effective = matrix.copy()
    effective.eliminate_zeros()
    effective.data = np.ones(effective.nnz, dtype=bool)
    effective_distance = distances(effective, arrays['visual_indices'], depth)[arrays['motor_indices']]
    manifest = dict(observed=observed, sources=cfg['sources'], artifact_sha256=sha256_file(artifact),
                    selection_mode=selection_mode,
                    status_counts=annotations.reindex(selected).status.fillna('missing').value_counts().to_dict(),
                    visual_neurons=len(visual), visual_types=visual.type.value_counts().to_dict(),
                    maximum_path_edges=depth, raw_spectral_radius=radius, scale=scale,
                    anatomical_reachable_motors=int((forward[np.searchsorted(ids, motor_ids)] <= depth).sum()),
                    effective_edges=effective.nnz,
                    effective_reachable_motors=int((effective_distance <= depth).sum()))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    annotations.reindex(selected).assign(consensus_nt=labels.to_numpy()).to_csv(output / 'neurons.csv')
    return manifest
