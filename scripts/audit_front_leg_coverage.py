#!/usr/bin/env python3
"""Audit a pinned actor subgraph against public MaleCNS annotations/synapses.

Run with --config configs/connectome/audits/front_leg_coverage.yaml.
Paths are repository-relative; output JSON and cell/edge tables are audit
artifacts only. No neuron identity is transferred between connectome datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
from scipy import sparse

from simtoolreal_shared.connectome_data import (
    _read_adjacency, download_verified, load_yaml, sha256_file,
)


def select_groups(annotations, current_ids, selection):
    """Use explicit native annotations; lineage sets are screening supersets."""
    a = annotations
    traced = a.status.eq(selection['status'])
    sensory = (traced & a.superclass.isin(selection['sensory_superclasses'])
               & a.entryNerve.isin(selection['front_entry_nerves']))
    masks = {
        'front_proprioceptors': sensory & a['class'].eq(selection['proprioceptive_class']),
        'front_tactile': sensory & a['class'].eq(selection['tactile_class']),
        'front_motors': (traced & a.superclass.eq(selection['motor_superclass'])
                         & a.subclass.eq(selection['motor_subclass'])),
        'grooming_dn_type_candidates': (traced & a.superclass.eq('descending_neuron')
                                        & a.type.str.contains(selection['descending_type_regex'], na=False)),
        'front_specific_descending': traced & a.superclass.eq('descending_neuron') & a.subclass.eq('fl'),
    }
    for lineage in selection['coordination_hemilineages']:
        masks['T1_' + lineage] = traced & a.somaNeuromere.eq(selection['soma_neuromere']) & a.trumanHl.eq(lineage)
    groups = {name: set(a.loc[mask, 'bodyId'].astype(int)) for name, mask in masks.items()}
    cases = {'current': set(current_ids)}
    cases['proprioceptors_and_motors'] = cases['current'] | groups['front_proprioceptors'] | groups['front_motors']
    cases['coordination_candidates'] = cases['proprioceptors_and_motors'].union(
        *(groups['T1_' + h] for h in selection['coordination_hemilineages']),
        groups['grooming_dn_type_candidates'], groups['front_specific_descending'])
    cases['coordination_and_touch'] = cases['coordination_candidates'] | groups['front_tactile']
    return groups, cases


def aggregate_batch(batch, targets, rois, confidence):
    """Count partner rows only after ROI, confidence and target filtering."""
    mask = pc.and_(pc.is_in(batch.column('body_post'), value_set=targets),
                   pc.is_in(batch.column('primary_post'), value_set=rois))
    mask = pc.and_(mask, pc.greater_equal(batch.column('conf_pre'), confidence))
    mask = pc.and_(mask, pc.greater_equal(batch.column('conf_post'), confidence))
    selected = batch.select(['body_pre', 'body_post']).filter(mask)
    if not selected.num_rows:
        return None
    return (pa.Table.from_batches([selected]).group_by(['body_pre', 'body_post'])
            .aggregate([('body_pre', 'count')]).rename_columns(['source', 'destination', 'synapses']))


def read_reference_edges(path, targets, settings):
    """Stream the large partner table; hold only edges entering audit targets."""
    reader = ipc.open_file(pa.memory_map(str(path)))
    target_array = pa.array(sorted(targets), type=pa.int64())
    roi_array = pa.array(settings['primary_post_rois'])
    observed_rois = set()
    parts = []
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        roi_column = batch.column('primary_post')
        if pa.types.is_dictionary(roi_column.type):
            observed_rois.update(roi_column.dictionary.to_pylist())
        else:
            observed_rois.update(pc.unique(roi_column).to_pylist())
        part = aggregate_batch(batch, target_array, roi_array, settings['minimum_confidence'])
        if part is not None:
            parts.append(part)
        if i % 500 == 0:
            print(f'Synapse batches {i}/{reader.num_record_batches}', flush=True)
    unknown = set(settings['primary_post_rois']) - observed_rois
    if unknown:
        raise ValueError(f'Configured ROI labels absent from reference: {sorted(unknown)}')
    if not parts:
        raise ValueError('No reference edges passed the configured filters')
    # A connection may span record batches and ROIs: sum BEFORE edge threshold.
    result = (pa.concat_tables(parts).group_by(['source', 'destination'])
              .aggregate([('synapses', 'sum')]).rename_columns(['source', 'destination', 'synapses']).to_pandas())
    return result.sort_values(['source', 'destination']).reset_index(drop=True)


def reachability(edges, ids, left_motors, right_motors, hops):
    """Structural reachability only; no claim about signs, strength or behavior."""
    ids = np.asarray(sorted(ids), dtype=np.int64)
    e = edges[edges.source.isin(ids) & edges.destination.isin(ids)]
    graph = sparse.csr_matrix((np.ones(len(e), dtype=np.int32),
                               (np.searchsorted(ids, e.source), np.searchsorted(ids, e.destination))),
                              shape=(len(ids), len(ids)))
    left = np.isin(ids, list(left_motors))
    right = np.isin(ids, list(right_motors))
    rows = []
    for hop in range(1, hops + 1):
        left |= np.asarray(graph @ left.astype(np.int32)).ravel() > 0
        right |= np.asarray(graph @ right.astype(np.int32)).ravel() > 0
        # Exclude output seeds from counts of upstream neurons.
        both = left & right & ~np.isin(ids, list(left_motors | right_motors))
        rows.append({'max_hops': hop, 'neurons_reaching_both_motor_sides': int(both.sum())})
    return rows


def case_metrics(ids, reference, threshold, baseline_ids):
    incoming = reference[reference.destination.isin(ids)]
    internal = incoming[incoming.source.isin(ids)]
    kept = internal[internal.synapses >= threshold]
    added = ids - baseline_ids
    connected = set(kept.source) | set(kept.destination)
    denominator = int(incoming.synapses.sum())
    return {
        'neurons': len(ids), 'added_neurons': len(ids - baseline_ids),
        'edges_at_threshold': len(kept),
        'new_endpoint_edges_at_threshold': int((~kept.source.isin(baseline_ids) | ~kept.destination.isin(baseline_ids)).sum()),
        'added_neurons_with_any_internal_edge': len(added & connected),
        'added_neurons_without_internal_edges': len(added - connected),
        'internal_synapses_before_edge_threshold': int(internal.synapses.sum()),
        'internal_synapses_at_threshold': int(kept.synapses.sum()),
        'all_observed_incoming_synapses': denominator,
        'retained_incoming_fraction_at_threshold': float(kept.synapses.sum() / denominator) if denominator else None,
    }


def run(config_path, repository_root):
    config = load_yaml(config_path)
    if config.get('schema_version') != 1:
        raise ValueError('Expected schema_version: 1')
    out = repository_root / config['output_directory']
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / 'summary.json'
    summary_path.write_text(json.dumps({'status': 'running'}, indent=2) + '\n')
    provenance = {}
    for name, spec in config['sources'].items():
        path = repository_root / spec['path']
        if not path.exists():
            if not config['download_missing']:
                raise FileNotFoundError(path)
            download_verified(spec['url'], path, spec['sha256'])
        actual = sha256_file(path)
        if actual != spec['sha256']:
            raise ValueError(f'Unexpected source hash for {path}')
        provenance[name] = {**spec, 'bytes': path.stat().st_size}

    current_config = load_yaml(repository_root / config['current_config'])
    raw_dir = repository_root / current_config['paths']['raw_directory']
    for spec in current_config['source']['files'].values():
        if sha256_file(raw_dir / spec['filename']) != spec['sha256']:
            raise ValueError('Current graph raw data does not match its pinned config')
    table_path = raw_dir / current_config['source']['files']['neurons']['filename']
    current = pd.read_csv(table_path).fillna('')
    current_ids = set(current.bodyId.astype(int))
    annotation = pd.read_feather(repository_root / config['sources']['annotations']['path'])
    if not annotation.bodyId.is_unique or not current.bodyId.is_unique:
        raise ValueError('Duplicate body IDs')
    groups, cases = select_groups(annotation, current_ids, config['selection'])
    union = set.union(*cases.values())
    reference = read_reference_edges(repository_root / config['sources']['partners']['path'], union, config['connectivity'])
    reference.to_csv(out / 'reference_incoming_edges.csv.gz', index=False)
    threshold = config['connectivity']['minimum_synapses']
    if threshold < 1:
        raise ValueError('minimum_synapses must be positive')
    reference_kept = reference[reference.synapses >= threshold]
    adjacency_path = raw_dir / current_config['source']['files']['adjacency']['filename']
    src, dst, raw_values = _read_adjacency(adjacency_path, current.bodyId.to_numpy(np.int64))
    production = pd.DataFrame({'source': current.bodyId.to_numpy()[src],
                               'destination': current.bodyId.to_numpy()[dst],
                               'production_magnitude': np.abs(raw_values)})
    current_reference = reference_kept[reference_kept.source.isin(current_ids) & reference_kept.destination.isin(current_ids)]
    comparison = production.merge(current_reference, on=['source', 'destination'], how='outer', indicator=True)
    comparison.to_csv(out / 'production_reference_comparison.csv.gz', index=False)
    motors = current[current['class'].eq('motor neuron')]
    left = set(motors.loc[motors.somaSide.eq('L'), 'bodyId'])
    right = set(motors.loc[motors.somaSide.eq('R'), 'bodyId'])

    columns = ['bodyId', 'type', 'mancType', 'mancBodyid', 'superclass', 'class', 'subclass',
               'trumanHl', 'somaNeuromere', 'somaSide', 'rootSide', 'entryNerve', 'status']
    # Keep current IDs missing from this public annotation snapshot as explicit
    # rows with blank native metadata, so case membership counts remain exact.
    records = annotation[columns].set_index('bodyId').reindex(sorted(union)).reset_index()
    records['annotation_matched'] = records.bodyId.isin(annotation.bodyId)
    records['in_current'] = records.bodyId.isin(current_ids)
    for name, ids in groups.items():
        records[name] = records.bodyId.isin(ids)
    for name, ids in cases.items():
        records['case_' + name] = records.bodyId.isin(ids)
        pd.DataFrame({'bodyId': sorted(ids - current_ids)}).to_csv(out / (name + '_added_body_ids.csv'), index=False)
    records.sort_values('bodyId').to_csv(out / 'candidate_neurons.csv', index=False)
    native_current = annotation[annotation.bodyId.isin(current_ids)]
    native_current[native_current.superclass.isin(config['selection']['sensory_superclasses'])].groupby(
        ['entryNerve', 'class', 'subclass'], dropna=False).size().rename('neurons').reset_index().to_csv(
            out / 'current_sensory_annotation_groups.csv', index=False)
    # Retain unmatched production IDs in a separate explicit file.
    current[~current.bodyId.isin(annotation.bodyId)].to_csv(out / 'unmatched_current_neurons.csv', index=False)

    coverage = {}
    for name, ids in groups.items():
        incoming = reference[reference.destination.isin(ids)]
        n = int(incoming.synapses.sum())
        retained_inputs = incoming[incoming.source.isin(current_ids) & incoming.destination.isin(current_ids) & (incoming.synapses >= threshold)]
        coverage[name] = {'reference_neurons': len(ids), 'in_current': len(ids & current_ids),
                          'missing': len(ids - current_ids), 'missing_body_ids': sorted(ids - current_ids),
                          'current_input_fraction_over_complete_group': float(retained_inputs.synapses.sum() / n) if n else None}
    # Externally omitted partners ranked by input to the proposed candidate union.
    missing = reference_kept[~reference_kept.source.isin(union)]
    ranked = missing.groupby('source').synapses.sum().sort_values(ascending=False).head(config['connectivity']['top_missing_partners'])
    ranked = ranked.rename('synapses_to_candidate_union').reset_index().merge(annotation[columns], left_on='source', right_on='bodyId', how='left')
    ranked.to_csv(out / 'top_external_inputs.csv', index=False)

    metrics = {name: case_metrics(ids, reference, threshold, current_ids) for name, ids in cases.items()}
    for value in metrics.values():
        value['node_ratio_to_current'] = value['neurons'] / len(current_ids)
        value['edge_ratio_to_same_reference_current'] = value['edges_at_threshold'] / metrics['current']['edges_at_threshold']
    raw_integer = bool(np.allclose(raw_values, np.round(raw_values)))
    summary = {
        'status': 'complete', 'dataset': config['dataset'], 'config': str(config_path.relative_to(repository_root)),
        'config_sha256': sha256_file(config_path), 'sources': provenance,
        'current_source': current_config['source'], 'connectivity_settings': config['connectivity'],
        'production': {'neurons': len(current_ids), 'edges': len(production),
                       'motor_soma_sides': motors.somaSide.value_counts().to_dict(),
                       'raw_values_are_integers': raw_integer},
        'annotation_join': {'matched': int(current.bodyId.isin(annotation.bodyId).sum()),
                           'unmatched_body_ids': sorted(current_ids - set(annotation.bodyId))},
        'groups': coverage, 'cases': metrics,
        'production_reference': {'edges_shared': int(comparison._merge.eq('both').sum()),
                                  'production_only': int(comparison._merge.eq('left_only').sum()),
                                  'reference_only': int(comparison._merge.eq('right_only').sum()),
                                  'shared_equal_magnitudes': int((comparison._merge.eq('both') & comparison.production_magnitude.eq(comparison.synapses)).sum())},
        'production_bilateral_reachability': reachability(production, current_ids, left, right, config['connectivity']['reachability_hops']),
        'limits': [
            'Lineage candidates do not identify the specific functional 9A, 13A, 13B or 23B subtypes.',
            'Soma side is not axonal target side; reachability is a graph property, not proof of coordination.',
            'Candidate counts are explicit annotation expansions, not a minimal or complete biological controller.',
            'Reference edge magnitudes are unsigned synapse counts, not deployment-ready signed weights.',
            'Input fractions use all observed sources in the listed VNC compartments, including unannotated segments.',
            'Public release and upstream extraction may differ; production/reference mismatch is reported, not repaired.',
            'No claw/hook/club identity is inferred from chordotonal labels or from another dataset body ID.',
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'summary': str(summary_path), 'cases': metrics, 'production_reference': summary['production_reference']}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run((root / args.config).resolve(), root)


if __name__ == '__main__':
    main()
