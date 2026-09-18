"""Small counterexamples for audit orientation, scope and completeness claims."""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import yaml

from scripts.audit_front_leg_coverage import (
    case_metrics, read_reference_edges, reachability, select_groups,
)


def test_reference_counts_across_batches_before_threshold_and_filters_rois(tmp_path):
    batch = pa.record_batch({
        'body_pre': [1, 1, 1, 99, 2, 3],
        'body_post': [2, 2, 2, 2, 1, 2],
        'primary_post': ['VNC', 'brain', 'VNC', 'VNC', 'VNC', 'VNC'],
        'conf_pre': [1., 1., .1, 1., 1., 1.],
        'conf_post': [1., 1., 1., 1., 1., .1],
    })
    path = tmp_path / 'partners.feather'
    with pa.OSFile(str(path), 'wb') as sink:
        with ipc.new_file(sink, batch.schema) as writer:
            for _ in range(3):
                writer.write_batch(batch)
    edges = read_reference_edges(path, {2}, {'primary_post_rois': ['VNC'], 'minimum_confidence': .5})
    assert edges.to_dict('records') == [
        {'source': 1, 'destination': 2, 'synapses': 3},
        {'source': 99, 'destination': 2, 'synapses': 3},
    ]
    metrics = case_metrics({1, 2}, edges, 3, {1, 2})
    assert metrics['edges_at_threshold'] == 1
    assert metrics['all_observed_incoming_synapses'] == 6
    assert metrics['retained_incoming_fraction_at_threshold'] == .5


def test_bilateral_reachability_obeys_edge_direction_and_hop_count():
    # 1 branches to both motor seeds, 4 reaches that branch two hops away.
    # 5 is downstream of a motor and must not be counted as upstream.
    e = pd.DataFrame({'source': [1, 1, 4, 2], 'destination': [2, 3, 1, 5]})
    rows = reachability(e, {1, 2, 3, 4, 5}, {2}, {3}, 2)
    assert [r['neurons_reaching_both_motor_sides'] for r in rows] == [1, 2]


def test_selection_does_not_equate_all_leg_sensors_or_lineage_names_with_function():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / 'configs/connectome/audits/front_leg_coverage.yaml').read_text())
    defaults = dict(status='Traced', superclass='vnc_sensory', entryNerve='ProLN',
                    type='', subclass='leg', somaNeuromere='', trumanHl='')
    rows = [
        {**defaults, 'bodyId': 1, 'class': 'mechanosensory_proprioceptive'},
        {**defaults, 'bodyId': 2, 'class': 'gustatory'},
        {**defaults, 'bodyId': 3, 'class': 'mechanosensory_proprioceptive', 'entryNerve': 'MesoLN'},
        {**defaults, 'bodyId': 4, 'class': '', 'superclass': 'vnc_intrinsic', 'trumanHl': '09A', 'somaNeuromere': 'T1'},
        {**defaults, 'bodyId': 5, 'class': '', 'superclass': 'vnc_intrinsic', 'trumanHl': '09A', 'somaNeuromere': 'T2'},
        {**defaults, 'bodyId': 6, 'class': '', 'superclass': 'descending_neuron', 'type': 'DNg111'},
        {**defaults, 'bodyId': 7, 'class': '', 'superclass': 'descending_neuron', 'type': 'DNg12_e'},
    ]
    groups, cases = select_groups(pd.DataFrame(rows), {99}, config['selection'])
    assert groups['front_proprioceptors'] == {1}
    assert groups['T1_09A'] == {4}
    assert groups['grooming_dn_type_candidates'] == {7}
    assert cases['coordination_candidates'] == {1, 4, 7, 99}
