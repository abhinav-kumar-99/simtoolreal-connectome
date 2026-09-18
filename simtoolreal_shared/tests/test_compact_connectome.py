from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from simtoolreal_shared.compact_connectome import (
    select_compact,
    select_distal_leg,
    source_signs,
)

ROOT = Path(__file__).resolve().parents[2]


def test_paths_and_all_sensory_premotor_motifs_without_padding():
    neurons = pd.DataFrame({'bodyId': range(1, 10)})
    for name, members in {'sens': [1], 'dn': [2], 'motor': [7], 'coord': [3, 9]}.items():
        neurons[name] = neurons.bodyId.isin(members)
    edges = pd.DataFrame([
        (1, 7, 5), (1, 4, 5), (4, 7, 5), (2, 5, 5), (5, 3, 5), (3, 7, 5),
        (1, 8, 5), (8, 6, 5), (6, 7, 5), (1, 6, 4), (99, 7, 50),
    ], columns=['source', 'destination', 'synapses'])
    selection = {'minimum_synapses': 5, 'seed_groups': ['sens', 'dn', 'motor', 'coord'],
                 'population_groups': {'sensory': ['sens'], 'descending': ['dn'], 'motor': ['motor']}}
    ids, kept, populations, stats, _ = select_compact(neurons, edges, selection)
    assert ids.tolist() == [1, 2, 3, 4, 5, 7, 9]
    assert stats['seed_neurons'] == 5
    assert stats['path_neurons'] == 6
    assert stats['sensory_premotor_additions'] == 1
    assert stats['isolated_neurons'] == 1  # Retain the approved disconnected seed.
    assert ids[populations['motor']].tolist() == [7]
    assert len(kept) == 6 and (kept.synapses >= 5).all()
    again = select_compact(neurons.sample(frac=1, random_state=4),
                           edges.sample(frac=1, random_state=5), selection)
    np.testing.assert_array_equal(ids, again[0])
    assert stats == again[3]
    with pytest.raises(ValueError, match='aggregated'):
        select_compact(neurons, pd.concat([edges, edges.iloc[:1]]), selection)
    neurons['sens'] = neurons['sens'].astype(str)
    with pytest.raises(ValueError, match='boolean'):
        select_compact(neurons, edges, selection)


def test_source_sign_policy_is_explicit_and_fails_on_unhandled_labels():
    nt = pd.DataFrame({'body': [1, 2, 3], 'consensus_nt': ['acetylcholine', 'gaba', 'unclear']})
    policy = {'acetylcholine': 1, 'gaba': -1, 'unclear': -1, 'missing': -1}
    signs, labels = source_signs(np.array([1, 2, 3, 4]), nt, policy)
    assert signs.tolist() == [1, -1, -1, -1]
    assert labels.tolist() == ['acetylcholine', 'gaba', 'unclear', 'missing']
    with pytest.raises(ValueError, match='unspecified'):
        source_signs(np.array([1, 2, 3, 4]), nt, {'acetylcholine': 1, 'gaba': -1})


def test_approved_local_graph_identity_and_signed_csr():
    from simtoolreal_shared.connectome_data import load_artifact
    config = yaml.safe_load((ROOT / 'configs/connectome/malecns_1952.yaml').read_text())
    path = ROOT / config['sources']['neurons']['path']
    artifact = ROOT / config['output_directory'] / 'biological.npz'
    if not path.exists() or not artifact.exists():
        pytest.skip('Pinned local audit/artifact not installed')
    ids, edges, ports, stats, _ = select_compact(
        pd.read_csv(path).fillna(''),
        pd.read_csv(ROOT / config['sources']['edges']['path']), config['selection'])
    assert stats == config['expected_selection']
    saved = load_artifact(artifact)
    np.testing.assert_array_equal(ids, saved['body_ids'])
    for name, ix in ports.items():
        np.testing.assert_array_equal(ix, saved[name + '_indices'])
    source_neurons = pd.read_csv(path).fillna('')
    for name, groups in config['selection']['artifact_population_groups'].items():
        members = set(source_neurons.loc[
            source_neurons[groups].any(axis=1), 'bodyId'
        ].astype(int))
        expected_indices = np.flatnonzero(np.isin(ids, sorted(members)))
        np.testing.assert_array_equal(saved[name + '_indices'], expected_indices)
    assert not np.intersect1d(
        saved['front_proprioceptors_indices'], saved['front_tactile_indices']
    ).size
    np.testing.assert_array_equal(
        np.sort(np.concatenate((
            saved['front_proprioceptors_indices'], saved['front_tactile_indices']
        ))),
        saved['sensory_indices'],
    )
    dst = np.repeat(ids, np.diff(saved['crow_indices']))
    src = ids[saved['col_indices']]
    reconstructed = pd.DataFrame({'source': src, 'destination': dst,
                                  'synapses': np.abs(saved['raw_values']).astype(np.int64)})
    pd.testing.assert_frame_equal(
        reconstructed.sort_values(['source', 'destination']).reset_index(drop=True),
        edges[['source', 'destination', 'synapses']], check_dtype=False)
    assert np.isfinite(saved['values']).all()


def test_approved_distal_leg_graph_identity_and_ports():
    from simtoolreal_shared.connectome_data import load_artifact
    config = yaml.safe_load(
        (ROOT / 'configs/connectome/malecns_262_distal_leg.yaml').read_text()
    )
    source = ROOT / config['sources']['neurons']['path']
    artifact = ROOT / config['output_directory'] / 'biological.npz'
    if not source.exists() or not artifact.exists():
        pytest.skip('Pinned local audit/artifact not installed')
    ids, edges, ports, stats, _ = select_distal_leg(
        pd.read_csv(source).fillna(''),
        pd.read_csv(ROOT / config['sources']['edges']['path']),
        config['selection'],
    )
    assert stats == config['expected_selection']
    assert len(ids) == 262 and len(edges) == 3194
    saved = load_artifact(artifact)
    np.testing.assert_array_equal(ids, saved['body_ids'])
    for name, indices in ports.items():
        np.testing.assert_array_equal(indices, saved[name + '_indices'])
    assert ids[ports['descending']].tolist() == [220203, 909558]


def test_compact_profile_and_suite_resolve_old_timing(tmp_path):
    from scripts.run_connectome_suite import _compose_resolved, _training_overrides
    import isaacgymenvs  # noqa: F401
    for filename in ['adaptation_1952_100b.yaml', 'adaptation_1952_smoke.yaml']:
        suite = yaml.safe_load((ROOT / 'configs/connectome/suites' / filename).read_text())
        train = suite['training']
        profile = train['train_profiles'][0]['train_profile']
        config = _compose_resolved(_training_overrides(train, profile, 42, 'test', tmp_path))
        c = config.train.params.config
        assert c.rollout_accumulation_steps == 1
        assert c.minibatch_size == c.microbatch_size == 49152
        assert c.central_value_config.minibatch_size == c.central_value_config.microbatch_size == 49152
        assert c.mini_epochs == c.central_value_config.mini_epochs == 2
        assert c.num_actors == 12288 and c.horizon_length == 16
        assert train['gpu_assignments'] == [0]
        graph = config.train.params.network.connectome
        assert graph.expected.neurons == 1952 and graph.expected.edges == 33720
        assert graph.adaptation.weight_mode == 'neuron_gains'
        assert graph.adaptation.learn_dynamics is False
    assert 12288 * 16 * 508626 == 99_999_940_608


def test_base_connectome_timing_defaults_follow_overridden_minibatches():
    from scripts.run_connectome_suite import _compose_resolved
    import isaacgymenvs  # noqa: F401
    overrides = ['task=SimToolRealLSTMAsymmetric', 'train=SimToolRealConnectomeGainsSAPG']
    c = _compose_resolved(overrides).train.params.config
    assert c.rollout_accumulation_steps == 1 and c.minibatch_size == 49152
    c = _compose_resolved(overrides + ['train.params.config.minibatch_size=1536',
        'train.params.config.central_value_config.minibatch_size=1536']).train.params.config
    assert c.microbatch_size == c.central_value_config.microbatch_size == 1536
