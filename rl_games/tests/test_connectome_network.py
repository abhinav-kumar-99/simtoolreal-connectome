from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.connectome_network_builder import ConnectomeBuilder
from scipy import sparse


@pytest.fixture()
def artifact_path(tmp_path):
    # Stored in runtime orientation: CSR rows are destinations, columns are sources.
    sources = np.asarray([0, 0, 1, 2, 2, 3, 4, 5], dtype=np.int64)
    destinations = np.asarray([2, 4, 3, 4, 5, 6, 5, 6], dtype=np.int64)
    values = np.asarray([0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.4], dtype=np.float32)
    matrix = sparse.coo_matrix((values, (destinations, sources)), shape=(7, 7)).tocsr()
    path = tmp_path / "tiny_connectome.npz"
    np.savez_compressed(
        path,
        schema_version=np.asarray([1], dtype=np.int64),
        crow_indices=matrix.indptr.astype(np.int64),
        col_indices=matrix.indices.astype(np.int64),
        values=matrix.data.astype(np.float32),
        raw_values=matrix.data.astype(np.float32),
        body_ids=np.arange(7, dtype=np.int64),
        sensory_indices=np.asarray([0, 1], dtype=np.int64),
        front_proprioceptors_indices=np.asarray([0], dtype=np.int64),
        front_tactile_indices=np.asarray([1], dtype=np.int64),
        descending_indices=np.asarray([2, 3], dtype=np.int64),
        motor_indices=np.asarray([4, 5], dtype=np.int64),
    )
    return path


def _network_params(
    artifact_path,
    plasticity_mode="neuron_gains",
    operator_backend="native_csr",
    interface_projections=None,
):
    params = {
        "name": "connectome_actor_critic",
        "connectome": {
            "artifact_path": str(artifact_path),
            "graph_variant": "test",
            "operator_backend": operator_backend,
            "dtype": "float32",
            "expected": {
                "neurons": 7,
                "edges": 8,
                "sensory_neurons": 2,
                "descending_neurons": 2,
                "motor_neurons": 2,
            },
            "observations": {
                "policy_size": 5,
                "sensory_size": 2,
                "goal_size": 3,
                "sensory_ranges": [[0, 2]],
                "goal_ranges": [[2, 5]],
            },
            "sapg_embedding_size": 4,
            "population_adapters": {"bias": False},
            "dynamics": {
                "activation": "tanh",
                "beta": 0.9,
                "gain_bounds": [0.25, 4.0],
                "initial_gain": 1.0,
                "initial_leak": 0.5,
                "initial_bias": 0.0,
            },
            "plasticity_mode": plasticity_mode,
        },
        "space": {
            "continuous": {
                "mu_activation": "None",
                "sigma_activation": "None",
                "fixed_sigma": "coef_cond",
            }
        },
    }
    if interface_projections is not None:
        params["connectome"]["interface_projections"] = interface_projections
    return params


def _build(
    artifact_path,
    plasticity_mode="neuron_gains",
    num_seqs=2,
    operator_backend="native_csr",
    adaptation=None,
    interface_projections=None,
    spectral_monitoring=None,
):
    builder = ConnectomeBuilder()
    params = _network_params(
        artifact_path,
        plasticity_mode,
        operator_backend,
        interface_projections,
    )
    if adaptation is not None:
        params["connectome"].pop("plasticity_mode")
        params["connectome"]["adaptation"] = adaptation
    if spectral_monitoring is not None:
        params["connectome"]["spectral_monitoring"] = spectral_monitoring
    builder.load(params)
    return builder.build(
        "connectome",
        actions_num=2,
        input_shape=(6,),
        num_seqs=num_seqs,
        value_size=1,
        type="extra_param",
        coef_ids=torch.tensor([50.0, 0.0]),
        coef_id_idx=5,
    )


def _observations(batch):
    obs = torch.randn(batch, 6)
    obs[:, 5] = torch.tensor([50.0, 0.0] * ((batch + 1) // 2))[:batch]
    return obs


def _build_policy_model(artifact_path, model_config, num_seqs=2):
    params = {
        "model": model_config,
        "network": _network_params(artifact_path),
    }
    return model_builder.ModelBuilder().load(deepcopy(params)).build(
        {
            "actions_num": 2,
            "input_shape": (6,),
            "num_seqs": num_seqs,
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
            "type": "extra_param",
            "coef_ids": torch.tensor([50.0, 0.0]),
            "coef_id_idx": 5,
        }
    )


def test_mlp_interface_projections_have_one_256_unit_hidden_layer(
    artifact_path,
) -> None:
    projection = {"architecture": "mlp", "hidden_size": 256, "activation": "elu"}
    network = _build(
        artifact_path,
        adaptation={"weight_mode": "adapters_only", "learn_dynamics": False},
        interface_projections=projection,
    )
    assert network.projection_architecture == "mlp"
    assert network.projection_hidden_size == 256
    assert network.projection_activation == "elu"
    for module, sizes, biased in (
        (network.sensory_adapter, (2, 256, 2), False),
        (network.descending_adapter, (7, 256, 2), False),
        (network.mu, (2, 256, 2), True),
    ):
        assert isinstance(module, torch.nn.Sequential)
        assert isinstance(module[0], torch.nn.Linear)
        assert isinstance(module[1], torch.nn.ELU)
        assert isinstance(module[2], torch.nn.Linear)
        assert (module[0].in_features, module[0].out_features) == sizes[:2]
        assert (module[2].in_features, module[2].out_features) == sizes[1:]
        assert (module[0].bias is not None) is biased
        assert (module[2].bias is not None) is biased
    assert network.mu[2].weight.abs().max() <= 1e-3
    assert torch.count_nonzero(network.mu[2].bias) == 0
    outputs = network({"obs": _observations(6), "seq_length": 3})
    loss = outputs[0].square().sum() + outputs[1].sum() + outputs[2].square().sum()
    loss.backward()
    for name, parameter in network.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert outputs[0].shape == (6, 2)
    assert outputs[2].shape == (6, 1)


def test_mlp_interface_projections_support_independent_hidden_widths(
    artifact_path,
) -> None:
    projection = {
        "architecture": "mlp",
        "hidden_size": 256,
        "sensory_hidden_size": 128,
        "descending_hidden_size": 32,
        "readout_hidden_size": 32,
        "activation": "elu",
    }
    params = _network_params(
        artifact_path,
        interface_projections=projection,
    )
    params["connectome"]["reservoir_readout"] = {
        "enabled": False,
        "feature_population": "all",
    }
    builder = ConnectomeBuilder()
    builder.load(params)
    network = builder.build(
        "asymmetric_mlp",
        actions_num=2,
        input_shape=(6,),
        num_seqs=2,
        value_size=1,
        type="extra_param",
        coef_ids=torch.tensor([50.0, 0.0]),
        coef_id_idx=5,
    )

    assert network.projection_hidden_size == 256
    assert network.sensory_projection_hidden_size == 128
    assert network.descending_projection_hidden_size == 32
    assert network.readout_projection_hidden_size == 32
    assert network.sensory_adapter[0].out_features == 128
    assert network.sensory_adapter[2].in_features == 128
    assert network.descending_adapter[0].out_features == 32
    assert network.descending_adapter[2].in_features == 32
    assert network.mu[0].in_features == 7
    assert network.mu[0].out_features == 32
    assert network.mu[2].in_features == 32

    outputs = network({"obs": _observations(6), "seq_length": 3})
    loss = outputs[0].square().sum() + outputs[1].sum() + outputs[2].square().sum()
    loss.backward()
    for name, parameter in network.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_mlp_actor_can_read_all_final_neuron_states(artifact_path) -> None:
    projection = {"architecture": "mlp", "hidden_size": 16, "activation": "elu"}
    params = _network_params(
        artifact_path,
        interface_projections=projection,
    )
    params["connectome"]["reservoir_readout"] = {
        "enabled": False,
        "feature_population": "all",
    }
    builder = ConnectomeBuilder()
    builder.load(params)
    network = builder.build(
        "all_neuron_readout",
        actions_num=2,
        input_shape=(6,),
        num_seqs=2,
        value_size=1,
        type="extra_param",
        coef_ids=torch.tensor([50.0, 0.0]),
        coef_id_idx=5,
    )

    assert network.readout_feature_population == "all"
    assert network.reservoir_feature_count == 7
    assert network.readout_input_size == 7
    assert network.mu[0].in_features == 7

    observations = _observations(6)
    initial = (torch.randn(1, 2, 7),)
    result = network(
        {"obs": observations, "rnn_states": initial, "seq_length": 3}
    )

    hidden = initial[0][0]
    outputs = []
    for step_observations in observations.reshape(2, 3, -1).transpose(0, 1):
        hidden = network._step(step_observations, hidden)
        outputs.append(hidden)
    final_states = torch.stack(outputs).transpose(0, 1).reshape(6, 7)
    torch.testing.assert_close(result[0], network.mu(final_states))
    torch.testing.assert_close(result[2], network.value(final_states))

    (result[0].square().mean() + result[2].square().mean()).backward()
    assert network.sensory_adapter[0].weight.grad is not None
    assert network.descending_adapter[0].weight.grad is not None


def test_structured_adapter_drives_only_proprioceptors_and_uses_context(
    artifact_path,
) -> None:
    params = _network_params(
        artifact_path,
        interface_projections={
            "architecture": "mlp",
            "hidden_size": 16,
            "activation": "elu",
        },
    )
    params["connectome"]["observations"] = {
        "policy_size": 9,
        "sensory_size": 6,
        "context_size": 1,
        "goal_size": 2,
        "sensory_ranges": [[0, 6]],
        "context_ranges": [[6, 7]],
        "goal_ranges": [[7, 9]],
    }
    params["connectome"]["structured_input_adapter"] = {
        "mode": "grouped_linear",
        "proprioceptor_artifact_key": "front_proprioceptors_indices",
        "tactile_artifact_key": "front_tactile_indices",
        "dof_count": 1,
        "fingertip_count": 1,
        "groups": [
            {"name": "limb", "dof_range": [0, 1], "fingertip_index": 0}
        ],
        "descending_projection": {
            "architecture": "mlp",
            "hidden_size": 8,
            "activation": "elu",
        },
    }
    params["connectome"].pop("plasticity_mode")
    params["connectome"]["adaptation"] = {
        "weight_mode": "adapters_only",
        "learn_dynamics": False,
    }
    builder = ConnectomeBuilder()
    builder.load(params)
    network = builder.build(
        "structured",
        actions_num=2,
        input_shape=(10,),
        num_seqs=2,
        value_size=1,
        type="extra_param",
        coef_ids=torch.tensor([50.0, 0.0]),
        coef_id_idx=9,
    )
    assert network.sensory_adapter_mode == "grouped_linear"
    assert network.sensory_adapter.group_names == ["limb"]
    assert network.sensory_adapter.group_allocations == (1,)
    assert network.descending_projection_architecture == "mlp"
    assert network.descending_adapter[0].in_features == 7
    assert network.descending_adapter[0].out_features == 8

    observations = torch.randn(6, 10)
    observations[:, 9] = torch.tensor([50.0, 0.0] * 3)
    sensory = network.sensory_adapter(observations[:, :6])
    assert sensory.shape == (6, 2)
    assert torch.count_nonzero(sensory[:, 1]) == 0
    outputs = network({"obs": observations, "seq_length": 3})
    loss = outputs[0].square().sum() + outputs[2].square().sum()
    loss.backward()
    for adapter in network.sensory_adapter.group_adapters:
        assert adapter.weight.grad is not None
        assert torch.isfinite(adapter.weight.grad).all()


def _build_fixed_reservoir(
    artifact_path, distribution="beta", activation="tanh", operator_backend="native_csr"
):
    params = _network_params(
        artifact_path,
        operator_backend=operator_backend,
        interface_projections={
            "architecture": "mlp",
            "hidden_size": 16,
            "activation": "elu",
        },
    )
    params["connectome"]["observations"] = {
        "policy_size": 3,
        "sensory_size": 1,
        "context_size": 1,
        "goal_size": 1,
        "sensory_ranges": [[0, 1]],
        "context_ranges": [[1, 2]],
        "goal_ranges": [[2, 3]],
    }
    params["connectome"]["fixed_input_encoder"] = {
        "mode": "population_code_v1",
        "proprioceptor_artifact_key": "front_proprioceptors_indices",
        "tactile_artifact_key": "front_tactile_indices",
        "sensory_mappings": [
            {
                "source_range": [0, 1],
                "target_offset": 0,
                "encoding": "signed_tanh",
                "scale": 2.0,
            }
        ],
        "descending_mappings": [
            {
                "source_range": [1, 2],
                "target_offset": 0,
                "encoding": "signed_tanh",
                "scale": 1.0,
            },
            {
                "source_range": [2, 3],
                "target_offset": 1,
                "encoding": "signed_tanh",
                "scale": 0.5,
            },
        ],
    }
    params["connectome"]["reservoir_readout"] = {
        "enabled": True,
        "feature_population": "motor",
        "condition_on_sapg": True,
        "architecture": "mlp",
        "hidden_size": 8,
        "activation": "elu",
    }
    params["connectome"].pop("plasticity_mode")
    params["connectome"]["adaptation"] = {
        "weight_mode": "adapters_only",
        "learn_dynamics": False,
    }
    params["connectome"]["dynamics"].update(
        activation=activation,
        neural_updates=4,
    )
    if activation == "lif":
        params["connectome"]["dynamics"].update(
            control_frequency_hz=60.0,
            membrane_time_constant_ms=10.0,
            spike_threshold=1.0,
            refractory_period_ms=2.0,
            input_current_scale=1.5,
        )
    if distribution == "beta":
        params["space"]["continuous"].update(
            distribution="beta",
            fixed_sigma="state_dependent",
            beta_initial_shape=2.0,
            beta_min_shape=1.0,
        )
    else:
        params["space"]["continuous"].update(
            distribution="gaussian",
            fixed_sigma="coef_cond",
            max_sigma=3.0,
        )
    builder = ConnectomeBuilder()
    builder.load(params)
    return builder.build(
        "fixed_reservoir",
        actions_num=2,
        input_shape=(4,),
        num_seqs=2,
        value_size=1,
        type="extra_param",
        coef_ids=torch.tensor([50.0, 0.0]),
        coef_id_idx=3,
    )


@pytest.mark.parametrize('capture', [False, True])
def test_frozen_tanh_matches_rollout_and_invalidates_on_reload(artifact_path, capture):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    reference = _build_fixed_reservoir(artifact_path, operator_backend='triton_fused').cuda()
    optimized = _build_fixed_reservoir(artifact_path, operator_backend='triton_fused').cuda()
    optimized.load_state_dict(reference.state_dict())
    optimized.backend_options.update(frozen_inference=True, cuda_graph=capture)
    obs = torch.tensor([[1., .25, -.5, 50.], [-1., -.25, .5, 0.]], device='cuda')
    a_state = tuple(s.cuda() for s in reference.get_default_rnn_state())
    b_state = tuple(s.cuda() for s in optimized.get_default_rnn_state())
    for step in range(4):
        if step == 2:
            a_state[0][:, 0].zero_()
            b_state[0][:, 0].zero_()
        a = reference({'obs': obs, 'rnn_states': a_state})
        b = optimized({'obs': obs, 'rnn_states': b_state})
        for x, y in zip(a[:3], b[:3]):
            torch.testing.assert_close(x, y)
        torch.testing.assert_close(a[3][0], b[3][0])
        a_state, b_state = a[3], b[3]
    assert optimized._frozen_tanh_runner is not None
    optimized.load_state_dict(reference.state_dict())
    assert optimized._frozen_tanh_runner is None
    cached = optimized.last_reservoir_features.clone()
    output = optimized({'obs': obs, 'reservoir_features': cached})
    output[0].sum().backward()
    assert optimized.extra_params.grad is not None


def test_fixed_reservoir_encodes_inputs_and_trains_only_cached_readout(
    artifact_path, monkeypatch
) -> None:
    network = _build_fixed_reservoir(artifact_path)
    observations = torch.tensor(
        [
            [1.0, 0.25, -0.5, 50.0],
            [-1.0, -0.25, 0.5, 0.0],
        ]
    )
    sensory = network.sensory_adapter(observations)
    descending = network.descending_adapter(observations)
    torch.testing.assert_close(sensory[:, 0], torch.tanh(observations[:, 0] / 2.0))
    assert torch.count_nonzero(sensory[:, 1]) == 0
    torch.testing.assert_close(descending[:, 0], torch.tanh(observations[:, 1]))
    torch.testing.assert_close(descending[:, 1], torch.tanh(observations[:, 2] / 0.5))
    assert not list(network.sensory_adapter.parameters())
    assert not list(network.descending_adapter.parameters())

    online = network(
        {
            "obs": observations,
            "rnn_states": network.get_default_rnn_state(),
        }
    )
    cached = network.last_reservoir_features.clone()
    assert cached.shape == (2, 2)
    assert online[3][0].shape == (1, 2, 7)

    def fail_step(*args, **kwargs):
        raise AssertionError("cached PPO update replayed the reservoir")

    monkeypatch.setattr(network, "_step", fail_step)
    training = network(
        {"obs": observations, "reservoir_features": cached}
    )
    assert training[3] == ()
    loss = training[0].sum() + training[1].sum()
    loss.backward()
    assert network.extra_params.grad is not None
    assert all(parameter.grad is None for parameter in network.sensory_adapter.parameters())
    assert all(parameter.grad is None for parameter in network.descending_adapter.parameters())
    for parameter in (
        network.incoming_gain_raw,
        network.outgoing_gain_raw,
        network.leak_raw,
        network.recurrent_bias,
        network.log_intrinsic_gain,
    ):
        assert not parameter.requires_grad and parameter.grad is None
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in network.mu.parameters()
    )
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in network.beta_head.parameters()
    )


def test_fixed_reservoir_gaussian_reuses_cached_features(
    artifact_path, monkeypatch
) -> None:
    from rl_games.algos_torch.models import ModelA2CContinuousLogStd

    network = _build_fixed_reservoir(artifact_path, distribution="gaussian")
    model = ModelA2CContinuousLogStd.Network(
        network,
        obs_shape=(4,),
        normalize_input=False,
        normalize_value=False,
        value_size=1,
        extra_info_start_idx=3,
    )
    observations = torch.tensor(
        [[1.0, 0.25, -0.5, 50.0], [-1.0, -0.25, 0.5, 0.0]]
    )
    rollout = model(
        {
            "is_train": False,
            "obs": observations,
            "rnn_states": model.get_default_rnn_state(),
        }
    )
    cached = rollout["reservoir_features"].clone()
    assert cached.shape == (2, 2)
    assert rollout["rnn_states"][0].shape == (1, 2, 7)

    def fail_step(*args, **kwargs):
        raise AssertionError("cached Gaussian PPO update replayed the reservoir")

    monkeypatch.setattr(network, "_step", fail_step)
    training = model(
        {
            "is_train": True,
            "obs": observations,
            "prev_actions": rollout["actions"],
            "reservoir_features": cached,
        }
    )
    assert training["rnn_states"] == ()
    loss = training["prev_neglogp"].mean() + training["values"].mean()
    loss.backward()
    assert network.extra_params.grad is not None
    assert network.sigma.grad is not None
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in network.mu.parameters()
    )


def test_fixed_reservoir_lif_uses_rate_codes_and_three_persistent_states(
    artifact_path,
) -> None:
    network = _build_fixed_reservoir(
        artifact_path, distribution="gaussian", activation="lif"
    )
    observations = torch.tensor(
        [[1.0, 0.25, -0.5, 50.0], [-1.0, -0.25, 0.5, 0.0]]
    )
    sensory_rates = network.sensory_adapter.spike_rates(observations)
    torch.testing.assert_close(
        sensory_rates[:, 0], 0.5 * (torch.tanh(observations[:, 0] / 2.0) + 1.0)
    )
    assert torch.count_nonzero(sensory_rates[:, 1]) == 0
    states = network.get_default_rnn_state()
    assert len(states) == 3
    assert all(state.shape == (1, 2, 7) for state in states)

    outputs = network({"obs": observations, "rnn_states": states})
    motor_rates = network.last_reservoir_features
    assert motor_rates.shape == (2, 2)
    assert torch.all((motor_rates >= 0.0) & (motor_rates <= 1.0))
    assert len(outputs[3]) == 3
    membrane, refractory, spikes = outputs[3]
    assert all(state.shape == (1, 2, 7) for state in outputs[3])
    assert torch.isfinite(membrane).all()
    assert torch.all(refractory >= 0.0)
    assert torch.all((spikes == 0.0) | (spikes == 1.0))

    cached = network(
        {"obs": observations, "reservoir_features": motor_rates.clone()}
    )
    assert cached[3] == ()
    (cached[0].sum() + cached[1].sum() + cached[2].sum()).backward()
    assert all(
        parameter.grad is None
        for parameter in (
            network.incoming_gain_raw,
            network.outgoing_gain_raw,
            network.leak_raw,
            network.recurrent_bias,
            network.log_intrinsic_gain,
        )
    )


def test_lif_requires_fixed_cached_reservoir(artifact_path) -> None:
    params = _network_params(artifact_path)
    params["connectome"]["dynamics"]["activation"] = "lif"
    with pytest.raises(ValueError, match="fixed reservoir_readout"):
        ConnectomeBuilder.Network(
            params,
            actions_num=2,
            input_shape=(6,),
            num_seqs=2,
            type="extra_param",
            coef_ids=torch.tensor([50.0, 0.0]),
            coef_id_idx=5,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fixed_reservoir_lif_triton_matches_dense(artifact_path) -> None:
    torch.manual_seed(73)
    reference = _build_fixed_reservoir(
        artifact_path,
        distribution="gaussian",
        activation="lif",
        operator_backend="dense",
    ).cuda()
    fused = _build_fixed_reservoir(
        artifact_path,
        distribution="gaussian",
        activation="lif",
        operator_backend="triton_fused",
    ).cuda()
    fused.load_state_dict(reference.state_dict())
    observations = torch.tensor(
        [[1.0, 0.25, -0.5, 50.0], [-1.0, -0.25, 0.5, 0.0]],
        device="cuda",
    )
    membrane = torch.rand(2, 7, device="cuda")
    refractory = torch.rand(2, 7, device="cuda") * 3.0
    spikes = torch.randint(0, 2, (2, 7), device="cuda").float()
    with torch.no_grad():
        expected = reference._step(
            observations, membrane, refractory=refractory, spikes=spikes
        )
        actual = fused._step(
            observations, membrane, refractory=refractory, spikes=spikes
        )
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(
            actual_tensor, expected_tensor, atol=2.0e-6, rtol=1.0e-5
        )


@pytest.mark.parametrize(
    "projection,error_type,message",
    [
        ({"architecture": "cnn"}, ValueError, "architecture"),
        ({"architecture": "mlp", "hidden_size": True}, TypeError, "hidden_size"),
        ({"architecture": "mlp", "hidden_size": 0}, ValueError, "hidden_size"),
        (
            {"architecture": "mlp", "sensory_hidden_size": True},
            TypeError,
            "sensory_hidden_size",
        ),
        (
            {"architecture": "mlp", "descending_hidden_size": 0},
            ValueError,
            "descending_hidden_size",
        ),
        (
            {"architecture": "mlp", "readout_hidden_size": 0},
            ValueError,
            "readout_hidden_size",
        ),
        ({"architecture": "mlp", "activation": "swish"}, ValueError, "activation"),
    ],
)
def test_invalid_interface_projection_configuration(
    artifact_path, projection, error_type, message
) -> None:
    with pytest.raises(error_type, match=message):
        _build(artifact_path, interface_projections=projection)


def test_sparse_step_matches_dense_reference(artifact_path) -> None:
    torch.manual_seed(1)
    network = _build(artifact_path)
    observations = _observations(2)
    hidden = torch.randn(2, 7)
    sparse_result = network._step(observations, hidden)

    sensory = network.sensory_adapter(observations[:, :2])
    goal = torch.cat(
        (
            observations[:, 2:5],
            network.extra_params[network._coefficient_rows(observations)],
        ),
        dim=-1,
    )
    descending = network.descending_adapter(goal)
    drive = torch.zeros_like(hidden)
    drive[:, network.sensory_indices] += sensory
    drive[:, network.descending_indices] += descending
    recurrent = (
        network.recurrent_matrix().to_dense() @ (network.outgoing_gains() * hidden).T
    ).T
    leak = network.leaks()
    dense_result = (1.0 - leak) * hidden + leak * torch.tanh(
        network.intrinsic_gains()
        * (
            network.recurrent_gain * network.incoming_gains() * recurrent
            + drive
        )
        + network.recurrent_bias
    )
    torch.testing.assert_close(sparse_result, dense_result, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("operator_backend", ["native_coo", "dense", "torch_sparse"])
def test_recurrent_backends_match_native_csr(
    artifact_path,
    operator_backend,
) -> None:
    if operator_backend == "torch_sparse":
        pytest.importorskip("torch_sparse")
    torch.manual_seed(11)
    reference = _build(artifact_path, operator_backend="native_csr")
    candidate = _build(artifact_path, operator_backend=operator_backend)
    candidate.load_state_dict(reference.state_dict())
    observations = _observations(4)
    initial_state = (torch.randn(1, 2, 7),)
    input_dict = {
        "obs": observations,
        "rnn_states": initial_state,
        "seq_length": 2,
    }
    reference_outputs = reference(input_dict)
    candidate_outputs = candidate(input_dict)
    for expected, actual in zip(reference_outputs[:3], candidate_outputs[:3]):
        torch.testing.assert_close(expected, actual, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(
        reference_outputs[3][0],
        candidate_outputs[3][0],
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    assert candidate.recurrent_matrix() is candidate.recurrent_matrix()

    reference_loss = sum(output.square().mean() for output in reference_outputs[:3])
    candidate_loss = sum(output.square().mean() for output in candidate_outputs[:3])
    reference_loss.backward()
    candidate_loss.backward()
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    for name, expected in reference_parameters.items():
        actual = candidate_parameters[name]
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(
                expected.grad,
                actual.grad,
                rtol=1.0e-5,
                atol=1.0e-6,
            )


@pytest.mark.parametrize(
    "mode", ["adapters_only", "neuron_gains", "low_rank", "edgewise"]
)
@pytest.mark.parametrize("dynamics", [False, True])
@pytest.mark.parametrize(
    "backend", ["native_csr", "native_coo", "torch_sparse", "cusparse", "triton_fused"]
)
def test_adaptation_backends_training(artifact_path, mode, dynamics, backend):
    if backend == "torch_sparse":
        pytest.importorskip("torch_sparse")
    if backend in {"cusparse", "triton_fused"} and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda" if backend in {"cusparse", "triton_fused"} else "cpu"
    adaptation = {"weight_mode": mode, "learn_dynamics": dynamics, "rank": 4}
    torch.manual_seed(23)
    reference = _build(
        artifact_path, operator_backend="dense", adaptation=adaptation
    ).to(device)
    # Exercise derivatives away from identity, including both low-rank factors.
    with torch.no_grad():
        for p in reference.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.05)
    candidate = _build(
        artifact_path, operator_backend=backend, adaptation=adaptation
    ).to(device)
    candidate.load_state_dict(reference.state_dict())
    optimizers = [
        torch.optim.Adam(m.parameters(), lr=1e-3) for m in (reference, candidate)
    ]
    for update in range(2):
        obs = _observations(6).to(device)
        h = torch.randn(1, 2, 7, device=device)
        results = []
        gradients = []
        for model, optimizer in zip((reference, candidate), optimizers):
            optimizer.zero_grad(set_to_none=True)
            x = obs.clone().requires_grad_()
            state = h.clone().requires_grad_()
            result = model(
                {
                    "obs": x,
                    "rnn_states": (state,),
                    "seq_length": 3,
                    "dones": torch.tensor(
                        [[0], [0], [1], [0], [1], [0]], device=device
                    ),
                }
            )
            loss = (
                sum(t.square().sum() for t in result[:3]) + result[3][0].square().sum()
            )
            loss.backward()
            results.append((*result[:3], result[3][0]))
            gradients.append((x.grad, state.grad))
        for a, b in zip(results[0] + gradients[0], results[1] + gradients[1]):
            torch.testing.assert_close(a, b, atol=3e-6, rtol=2e-4)
        for (name, a), (_, b) in zip(
            reference.named_parameters(), candidate.named_parameters()
        ):
            if a.grad is None:
                assert b.grad is None, name
            else:
                torch.testing.assert_close(
                    a.grad, b.grad, atol=5e-6, rtol=3e-4, msg=name
                )
        for optimizer in optimizers:
            optimizer.step()
    base = candidate.recurrent_values
    effective = candidate.effective_values()
    if mode == "neuron_gains":
        graph = candidate.backend_graph()
        effective = (
            effective
            * candidate.incoming_gains()[graph.rows]
            * candidate.outgoing_gains()[graph.col]
        )
    assert torch.equal(base.sign(), effective.sign())
    if mode in {"neuron_gains", "edgewise"}:
        assert torch.all(effective / base >= 0.0625)
        assert torch.all(effective / base <= 16.0)
    elif mode == "low_rank":
        # Log-fold gains are unbounded (aside from a numerical clamp).
        gain = effective / base
        assert torch.all(gain > 0)
        torch.testing.assert_close(
            effective, base * candidate.low_rank_delta().clamp(-20, 20).exp()
        )


def test_intrinsic_gain_identity_trainability_and_checkpoint(artifact_path):
    torch.manual_seed(17)
    reference = _build(
        artifact_path,
        adaptation={"weight_mode": "adapters_only", "learn_dynamics": False},
    )
    plastic = _build(
        artifact_path,
        adaptation={"weight_mode": "adapters_only", "learn_dynamics": True},
    )
    plastic.load_state_dict(reference.state_dict())
    assert torch.allclose(plastic.intrinsic_gains(), torch.ones(7))
    assert torch.equal(plastic.log_intrinsic_gain, torch.zeros(7))
    for name in ("leak_raw", "recurrent_bias", "log_intrinsic_gain"):
        assert dict(plastic.named_parameters())[name].requires_grad
    for name in ("incoming_gain_raw", "outgoing_gain_raw"):
        assert not dict(plastic.named_parameters())[name].requires_grad
    assert not plastic.recurrent_values.requires_grad

    observations = _observations(4)
    initial = (torch.randn(1, 2, 7),)
    payload = {
        "obs": observations,
        "rnn_states": tuple(state.clone() for state in initial),
        "seq_length": 2,
    }
    with torch.no_grad():
        expected = reference(payload)
        actual = plastic(
            {
                "obs": observations,
                "rnn_states": tuple(state.clone() for state in initial),
                "seq_length": 2,
            }
        )
    for left, right in zip(expected[:3], actual[:3]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(expected[3][0], actual[3][0], rtol=0, atol=0)

    # Older checkpoints without log_intrinsic_gain reload with a=1.
    checkpoint = reference.state_dict()
    del checkpoint["log_intrinsic_gain"]
    restored = _build(
        artifact_path,
        adaptation={"weight_mode": "adapters_only", "learn_dynamics": True},
    )
    restored.load_state_dict(checkpoint)
    torch.testing.assert_close(
        restored.log_intrinsic_gain, torch.zeros(7), rtol=0, atol=0
    )
    with torch.no_grad():
        reloaded = restored(
            {
                "obs": observations,
                "rnn_states": tuple(state.clone() for state in initial),
                "seq_length": 2,
            }
        )
    for left, right in zip(expected[:3], reloaded[:3]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_adapters_default_and_low_rank_initial_gradient(artifact_path):
    frozen = _build(artifact_path, adaptation={})
    assert frozen.weight_mode == "adapters_only" and not frozen.learn_dynamics
    before = {
        n: p.detach().clone()
        for n, p in frozen.named_parameters()
        if not p.requires_grad
    }
    result = frozen({"obs": _observations(6), "seq_length": 3})
    (result[0].square().sum() + result[2].square().sum()).backward()
    assert frozen.sensory_adapter.weight.grad.abs().sum() > 0
    torch.optim.Adam(frozen.parameters()).step()
    for name, expected in before.items():
        torch.testing.assert_close(
            dict(frozen.named_parameters())[name], expected, rtol=0, atol=0
        )
    model = _build(artifact_path, adaptation={"weight_mode": "low_rank"})
    torch.testing.assert_close(model.effective_values(), model.recurrent_values)
    model.effective_values().sum().backward()
    assert model.edge_v.grad.abs().sum() > 0


def test_low_rank_log_fold_gains_and_regularizer(artifact_path):
    torch.manual_seed(11)
    model = _build(
        artifact_path,
        adaptation={
            "weight_mode": "low_rank",
            "learn_dynamics": False,
            "rank": 4,
            "synaptic_plasticity_reg": 1.0e-4,
        },
    )
    assert model.synaptic_plasticity_reg == pytest.approx(1.0e-4)
    assert not model.incoming_gain_raw.requires_grad
    assert not model.outgoing_gain_raw.requires_grad
    for name in ("leak_raw", "recurrent_bias", "log_intrinsic_gain"):
        assert not dict(model.named_parameters())[name].requires_grad
    assert model.edge_u.requires_grad and model.edge_v.requires_grad
    assert isinstance(model.recurrent_gain, float)

    delta = model.low_rank_delta()
    assert delta.shape == (model.edge_count,)
    torch.testing.assert_close(delta, torch.zeros_like(delta))
    torch.testing.assert_close(model.effective_values(), model.recurrent_values)

    with torch.no_grad():
        model.edge_v.normal_(std=0.5)
        # Drive an existing edge (dst=2, src=0 in the fixture graph) past the
        # old [1/16, 16] sigmoid bounds.
        model.edge_u[2].fill_(4.0)
        model.edge_v[0].fill_(4.0)
    delta = model.low_rank_delta()
    assert delta.shape == (model.edge_count,)
    expected = model.recurrent_values * delta.clamp(-20.0, 20.0).exp()
    torch.testing.assert_close(model.effective_values(), expected)
    gains = model.effective_values() / model.recurrent_values
    assert torch.all(gains > 0)
    assert bool((gains > 16.0).any() or (gains < 0.0625).any())
    assert torch.equal(model.recurrent_values.sign(), model.effective_values().sign())

    mean_sq, stats = model.synaptic_plasticity_penalty()
    torch.testing.assert_close(mean_sq, delta.square().mean())
    assert set(stats) == {
        "synaptic_delta_mean",
        "synaptic_delta_rms",
        "synaptic_delta_std",
        "synaptic_gain_min",
        "synaptic_gain_max",
    }
    assert stats["synaptic_delta_rms"] == pytest.approx(
        float(delta.square().mean().sqrt().item())
    )


def test_edgewise_still_uses_bounded_sigmoid_gains(artifact_path):
    torch.manual_seed(5)
    model = _build(
        artifact_path,
        adaptation={"weight_mode": "edgewise", "learn_dynamics": False},
    )
    with torch.no_grad():
        model.edge_raw.fill_(20.0)
    gains = model.effective_values() / model.recurrent_values
    assert torch.all(gains <= 16.0 + 1.0e-5)
    assert torch.all(gains >= 0.0625 - 1.0e-5)
    with torch.no_grad():
        model.edge_raw.zero_()
    torch.testing.assert_close(model.effective_values(), model.recurrent_values)


def test_spectral_monitoring_activation_and_init_identity(artifact_path):
    spectral = {"enabled": True, "interval": 10, "top_k": 3}
    adapters = _build(
        artifact_path,
        adaptation={"weight_mode": "adapters_only", "learn_dynamics": False},
        spectral_monitoring=spectral,
    )
    intrinsic = _build(
        artifact_path,
        adaptation={"weight_mode": "adapters_only", "learn_dynamics": True},
        spectral_monitoring=spectral,
    )
    low_rank = _build(
        artifact_path,
        adaptation={"weight_mode": "low_rank", "learn_dynamics": False, "rank": 2},
        spectral_monitoring=spectral,
    )
    assert not adapters.synaptic_weights_trainable
    assert not intrinsic.synaptic_weights_trainable
    assert low_rank.synaptic_weights_trainable
    assert adapters.compute_spectral_diagnostics() == {}
    assert intrinsic.compute_spectral_diagnostics() == {}
    assert not adapters.should_run_spectral_monitoring(10)
    assert low_rank.should_run_spectral_monitoring(1)
    assert low_rank.should_run_spectral_monitoring(10)
    assert not low_rank.should_run_spectral_monitoring(11)

    tags = low_rank.compute_spectral_diagnostics()
    assert tags
    assert all(key.startswith("spectral/") for key in tags)
    assert tags["spectral/change/spectral_radius_ratio"] == pytest.approx(1.0, abs=1e-5)
    assert tags["spectral/change/max_singular_value_ratio"] == pytest.approx(
        1.0, abs=1e-5
    )
    assert tags["spectral/effective/spectral_radius"] == pytest.approx(
        tags["spectral/baseline/spectral_radius"], abs=1e-5
    )
    assert tags["spectral/effective/max_singular_value"] == pytest.approx(
        tags["spectral/baseline/max_singular_value"], abs=1e-5
    )
    assert tags["spectral/plasticity/delta_rms"] == pytest.approx(0.0, abs=1e-6)
    assert tags["spectral/plasticity/gain_mean"] == pytest.approx(1.0, abs=1e-5)

    # Analyzed values match forward effective_values / baseline buffer.
    from rl_games.algos_torch.connectome_spectral import spectral_metrics

    crow = low_rank.crow_indices.cpu().numpy()
    col = low_rank.col_indices.cpu().numpy()
    direct = spectral_metrics(
        crow,
        col,
        low_rank.effective_values().detach().cpu().numpy(),
        low_rank.neuron_count,
        3,
    )
    assert tags["spectral/effective/spectral_radius"] == pytest.approx(
        direct["spectral_radius"], abs=1e-5
    )

    with torch.no_grad():
        low_rank.edge_v.normal_(std=0.2)
    moved = low_rank.compute_spectral_diagnostics()
    assert moved["spectral/plasticity/delta_rms"] > 0
    # Baseline stays frozen at W0.
    assert moved["spectral/baseline/spectral_radius"] == pytest.approx(
        tags["spectral/baseline/spectral_radius"], abs=0
    )


@pytest.mark.parametrize("backend", ["cusparse", "triton_fused"])
@pytest.mark.parametrize("batch", [1, 37])
@pytest.mark.parametrize("projection_architecture", ["linear", "mlp"])
def test_cuda_shapes_amp_reload_and_stream(
    artifact_path, backend, batch, projection_architecture
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    opts = {"weight_mode": "edgewise", "learn_dynamics": True}
    projections = {
        "architecture": projection_architecture,
        "hidden_size": 256,
        "activation": "elu",
    }
    a = _build(
        artifact_path,
        num_seqs=batch,
        operator_backend="dense",
        adaptation=opts,
        interface_projections=projections,
    ).cuda()
    b = _build(
        artifact_path,
        num_seqs=batch,
        operator_backend=backend,
        adaptation=opts,
        interface_projections=projections,
    ).cuda()
    b.load_state_dict(a.state_dict())
    obs = _observations(batch * 2).cuda()
    original_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(original_stream)
    with torch.cuda.stream(stream):
        for model in (a, b):
            with torch.autocast("cuda"):
                result = model({"obs": obs, "seq_length": 2})
            (result[0].float().sum() + result[2].float().sum()).backward()
        for module_name in ("sensory_adapter", "descending_adapter", "mu"):
            a_parameters = dict(getattr(a, module_name).named_parameters())
            b_parameters = dict(getattr(b, module_name).named_parameters())
            assert a_parameters.keys() == b_parameters.keys()
            for name in a_parameters:
                torch.testing.assert_close(
                    a_parameters[name].grad,
                    b_parameters[name].grad,
                    atol=2e-4,
                    rtol=3e-3,
                    msg=f"{module_name}.{name}",
                )
    original_stream.wait_stream(stream)
    with torch.no_grad():
        a.recurrent_values.mul_(0.3)
    b.load_state_dict(a.state_dict())  # invalidate already-used plans/operators
    for aa, bb in zip(a({"obs": obs})[:3], b({"obs": obs})[:3]):
        torch.testing.assert_close(aa, bb, atol=3e-6, rtol=2e-4)
    assert not any("plan" in key or "cache" in key for key in b.state_dict())


@pytest.mark.parametrize(
    "algorithm,layout", [("alg1", "column"), ("alg2", "row"), ("alg3", "row")]
)
def test_cusparse_algorithms(artifact_path, algorithm, layout):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = _build(artifact_path, operator_backend="cusparse").cuda()
    model.backend_options = {"cusparse_algorithm": algorithm, "cusparse_layout": layout}
    hidden = torch.randn(37, 7, device="cuda", requires_grad=True)
    actual = model._recurrent_multiply(hidden)
    expected = (model._native_csr_matrix().to_dense() @ hidden.t()).t()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    actual.square().sum().backward()
    assert torch.isfinite(hidden.grad).all()


def test_conflicting_adaptation_configuration(artifact_path):
    params = _network_params(artifact_path)
    params["connectome"]["adaptation"] = {}
    builder = ConnectomeBuilder()
    builder.load(params)
    with pytest.raises(ValueError, match="not both"):
        builder.build("bad", actions_num=2, input_shape=(5,))


@pytest.mark.parametrize("backend", ["cusparse", "triton_fused"])
@pytest.mark.parametrize(
    "mode", ["adapters_only", "neuron_gains", "low_rank", "edgewise"]
)
def test_real_graph_custom_gradients(backend, mode):
    from pathlib import Path

    from scripts.profile_connectome_actors import _compose_network

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    root = Path(__file__).resolve().parents[2]
    if not (root / "data/connectomes/processed/malecns_4310/biological.npz").exists():
        pytest.skip("Prepared biological graph required")
    params = _compose_network("SimToolRealConnectomeSAPG", "SimToolRealLSTMAsymmetric")
    params["connectome"]["adaptation"].update(weight_mode=mode, learn_dynamics=True)
    models = []
    for implementation in ("dense", backend):
        params["connectome"]["operator_backend"] = implementation
        builder = ConnectomeBuilder()
        builder.load(deepcopy(params))
        models.append(
            builder.build(
                "real",
                actions_num=29,
                input_shape=(141,),
                num_seqs=37,
                type="extra_param",
                coef_ids=torch.tensor([50.0, 0.0]),
                coef_id_idx=140,
            ).cuda()
        )
    with torch.no_grad():
        if mode == "low_rank":
            models[0].edge_v.normal_(std=0.05)
        elif mode == "edgewise":
            models[0].edge_raw.normal_(std=0.05)
    models[1].load_state_dict(models[0].state_dict())
    observations = torch.randn(111, 141, device="cuda")
    observations[:, 140] = 50
    initial = torch.randn(1, 37, 4310, device="cuda") * 0.1
    results = []
    for model in models:
        result = model({"obs": observations, "rnn_states": (initial,), "seq_length": 3})
        sum(t.square().mean() for t in result[:3]).backward()
        results.append((*result[:3], result[3][0]))
    for expected, actual in zip(*results):
        torch.testing.assert_close(expected, actual, atol=3e-6, rtol=3e-4)
    for (name, a), (_, b) in zip(
        models[0].named_parameters(), models[1].named_parameters()
    ):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-4, msg=name)


def test_sequence_matches_steps_and_done_resets(artifact_path) -> None:
    torch.manual_seed(2)
    network = _build(artifact_path, num_seqs=2)
    observations = _observations(6)
    initial = (torch.randn(1, 2, 7),)
    sequence_result = network(
        {
            "obs": observations,
            "rnn_states": tuple(state.clone() for state in initial),
            "seq_length": 3,
        }
    )

    hidden = initial[0][0]
    outputs = []
    sequence = observations.reshape(2, 3, -1).transpose(0, 1)
    for step_observations in sequence:
        hidden = network._step(step_observations, hidden)
        outputs.append(hidden)
    output = torch.stack(outputs).transpose(0, 1).reshape(6, 7)
    torch.testing.assert_close(
        sequence_result[0], network.mu(output[:, network.motor_indices])
    )
    torch.testing.assert_close(sequence_result[2], network.value(output))
    torch.testing.assert_close(sequence_result[3][0], hidden.unsqueeze(0))

    reset_network = _build(artifact_path, num_seqs=1)
    reset_network.load_state_dict(network.state_dict())
    pair = observations[:2].clone()
    with_reset = reset_network(
        {
            "obs": pair,
            "rnn_states": (initial[0][:, :1].clone(),),
            "seq_length": 2,
            "dones": torch.tensor([[0.0], [1.0]]),
        }
    )
    from_zero = reset_network(
        {
            "obs": pair[1:],
            "rnn_states": reset_network.get_default_rnn_state(),
            "seq_length": 1,
        }
    )
    torch.testing.assert_close(with_reset[3][0], from_zero[3][0])


def test_sapg_embedding_sigma_and_gradients(artifact_path) -> None:
    torch.manual_seed(3)
    network = _build(artifact_path)
    with torch.no_grad():
        network.extra_params[0].zero_()
        network.extra_params[1].fill_(1.0)
        network.sigma[0].fill_(-1.0)
        network.sigma[1].fill_(0.5)
    observations = torch.zeros(4, 6)
    observations[:, 5] = torch.tensor([50.0, 50.0, 0.0, 0.0])
    mu, sigma, value, state = network(
        {
            "obs": observations,
            "rnn_states": network.get_default_rnn_state(),
            "seq_length": 2,
        }
    )
    assert torch.all(sigma[:2] == -1.0)
    assert torch.all(sigma[2:] == 0.5)
    assert not torch.equal(state[0][:, 0], state[0][:, 1])
    loss = mu.square().sum() + sigma.sum() + value.square().sum()
    loss.backward()
    for name in (
        "incoming_gain_raw",
        "outgoing_gain_raw",
        "leak_raw",
        "recurrent_bias",
        "log_intrinsic_gain",
        "extra_params",
    ):
        gradient = dict(network.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()
    assert torch.allclose(network.incoming_gains(), torch.ones(7))
    assert torch.allclose(network.outgoing_gains(), torch.ones(7))
    assert torch.allclose(network.leaks(), torch.full((7,), 0.5))
    assert torch.allclose(network.intrinsic_gains(), torch.ones(7))


def test_frozen_core_and_global_builder_checkpoint_round_trip(
    artifact_path, tmp_path
) -> None:
    frozen = _build(artifact_path, plasticity_mode="frozen_core")
    for name in (
        "incoming_gain_raw",
        "outgoing_gain_raw",
        "leak_raw",
        "recurrent_bias",
        "log_intrinsic_gain",
    ):
        assert not dict(frozen.named_parameters())[name].requires_grad
    assert frozen.sensory_adapter.weight.requires_grad
    assert frozen.mu.weight.requires_grad

    params = {
        "model": {"name": "continuous_a2c_logstd"},
        "network": _network_params(artifact_path),
    }
    model_factory = model_builder.ModelBuilder()
    first_model = model_factory.load(deepcopy(params)).build(
        {
            "actions_num": 2,
            "input_shape": (6,),
            "num_seqs": 2,
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
            "type": "extra_param",
            "coef_ids": torch.tensor([50.0, 0.0]),
            "coef_id_idx": 5,
        }
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(first_model.state_dict(), checkpoint)

    second_factory = model_builder.ModelBuilder()
    second_model = second_factory.load(deepcopy(params)).build(
        {
            "actions_num": 2,
            "input_shape": (6,),
            "num_seqs": 2,
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
            "type": "extra_param",
            "coef_ids": torch.tensor([50.0, 0.0]),
            "coef_id_idx": 5,
        }
    )
    second_model.load_state_dict(torch.load(checkpoint, weights_only=True))
    observations = _observations(2)
    first = first_model.a2c_network(
        {"obs": observations, "rnn_states": first_model.get_default_rnn_state()}
    )
    second = second_model.a2c_network(
        {"obs": observations, "rnn_states": second_model.get_default_rnn_state()}
    )
    for first_tensor, second_tensor in zip(first[:3], second[:3]):
        torch.testing.assert_close(first_tensor, second_tensor)


def test_tanh_policy_executes_bounded_actions_and_keeps_latent_coordinates(
    artifact_path,
) -> None:
    torch.manual_seed(41)
    model = _build_policy_model(
        artifact_path,
        {"name": "continuous_a2c_tanh_logstd", "entropy_samples": 8},
    )
    observations = _observations(2)
    rollout = model(
        {
            "is_train": False,
            "obs": observations.clone(),
            "rnn_states": model.get_default_rnn_state(),
            "seq_length": 1,
        }
    )

    assert torch.all(rollout["actions"].abs() <= 1.0)
    torch.testing.assert_close(
        rollout["actions"], torch.tanh(rollout["pre_tanh_actions"])
    )
    torch.testing.assert_close(rollout["mus"], torch.tanh(rollout["pre_tanh_mus"]))

    base = torch.distributions.Normal(
        rollout["pre_tanh_mus"], rollout["sigmas"]
    )
    log_jacobian = model.log_abs_det_jacobian(rollout["pre_tanh_actions"])
    expected_neglogp = (
        -base.log_prob(rollout["pre_tanh_actions"]) + log_jacobian
    ).sum(dim=-1)
    torch.testing.assert_close(rollout["neglogpacs"], expected_neglogp)

    training = model(
        {
            "is_train": True,
            "obs": observations.clone(),
            "prev_actions": rollout["pre_tanh_actions"],
            "rnn_states": model.get_default_rnn_state(),
            "seq_length": 1,
        }
    )
    torch.testing.assert_close(training["prev_neglogp"], rollout["neglogpacs"])
    torch.testing.assert_close(training["mus"], rollout["pre_tanh_mus"])
    assert torch.isfinite(training["entropy"]).all()


def test_tanh_policy_does_not_clamp_log_standard_deviation(artifact_path) -> None:
    torch.manual_seed(43)
    model = _build_policy_model(
        artifact_path,
        {"name": "continuous_a2c_tanh_logstd", "entropy_samples": 4096},
    )
    with torch.no_grad():
        model.a2c_network.sigma.fill_(3.25)
        for parameter in model.a2c_network.mu.parameters():
            parameter.zero_()

    observations = _observations(2)
    rollout = model(
        {
            "is_train": False,
            "obs": observations.clone(),
            "rnn_states": model.get_default_rnn_state(),
            "seq_length": 1,
        }
    )
    torch.testing.assert_close(
        rollout["sigmas"], torch.full_like(rollout["sigmas"], np.exp(3.25))
    )
    assert torch.isfinite(rollout["neglogpacs"]).all()
    assert torch.all(rollout["actions"].abs() <= 1.0)

    training = model(
        {
            "is_train": True,
            "obs": observations.clone(),
            "prev_actions": rollout["pre_tanh_actions"],
            "rnn_states": model.get_default_rnn_state(),
            "seq_length": 1,
        }
    )
    training["entropy"].mean().backward()
    sigma_gradient = model.a2c_network.sigma.grad
    assert sigma_gradient is not None and torch.isfinite(sigma_gradient).all()
    assert sigma_gradient.mean() < 0.0


@pytest.mark.parametrize("entropy_samples", [True, 0, -1, 1.5])
def test_tanh_policy_rejects_invalid_entropy_sample_count(
    artifact_path, entropy_samples
) -> None:
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        _build_policy_model(
            artifact_path,
            {
                "name": "continuous_a2c_tanh_logstd",
                "entropy_samples": entropy_samples,
            },
        )


@pytest.mark.parametrize('updates', [1, 4, 8])
@pytest.mark.parametrize('distribution', ['gaussian', 'beta'])
def test_deployment_rl_player_loads_connectome_checkpoint(
    artifact_path, tmp_path, updates, distribution
) -> None:
    pytest.importorskip("gym")
    from deployment.rl_player import RlPlayer

    network_params = _network_params(artifact_path)
    network_params['connectome']['dynamics']['neural_updates'] = updates
    network_params['space']['continuous']['distribution'] = distribution
    params = {
        "seed": 5,
        "algo": {"name": "a2c_continuous"},
        "model": {"name": "continuous_a2c_beta" if distribution == 'beta' else "continuous_a2c_logstd"},
        "network": network_params,
        "config": {
            "env_name": "rlgpu",
            "device_name": "cpu",
            "device": "cpu",
            "num_actors": 1,
            "normalize_input": False,
            "normalize_value": False,
            "expl_type": "mixed_expl_learn_param",
            "expl_reward_coef_embd_size": 4,
            "reward_shaper": {"scale_value": 1.0},
            "player": {"deterministic": True, "games_num": 1, "print_stats": False},
        },
    }
    factory = model_builder.ModelBuilder()
    model = factory.load(
        deepcopy({"model": params["model"], "network": network_params})
    ).build(
        {
            "actions_num": 2,
            "input_shape": (6,),
            "num_seqs": 1,
            "value_size": 1,
            "normalize_input": False,
            "normalize_value": False,
            "type": "extra_param",
            "coef_ids": torch.linspace(50.0, 0.0, 6),
            "coef_id_idx": 5,
        }
    )
    checkpoint_path = tmp_path / "deployment.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(yaml.safe_dump({"train": {"params": params}}))
    (tmp_path / 'resolved_config.yaml').write_text(config_path.read_text())

    player = RlPlayer(
        num_observations=5,
        num_actions=2,
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device="cpu",
        num_envs=1,
    )
    from unittest.mock import patch
    net = player.player.model.a2c_network
    assert net.neural_updates == updates
    with patch.object(net, '_recurrent_multiply', wraps=net._recurrent_multiply) as multiply:
        action = player.get_normalized_action(torch.zeros(1, 5), deterministic_actions=True)
        assert multiply.call_count == updates
    assert action.shape == (1, 2)
    assert torch.isfinite(action).all()
    player.reset()
    torch.testing.assert_close(
        player.get_normalized_action(torch.zeros(1, 5), deterministic_actions=True), action
    )
    params['network']['connectome']['dynamics']['neural_updates'] = updates + 1
    config_path.write_text(yaml.safe_dump({'train': {'params': params}}))
    with pytest.raises(ValueError, match='differs from training'):
        RlPlayer(5, 2, str(config_path), str(checkpoint_path), 'cpu')
