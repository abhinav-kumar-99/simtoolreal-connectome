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
        descending_indices=np.asarray([2, 3], dtype=np.int64),
        motor_indices=np.asarray([4, 5], dtype=np.int64),
    )
    return path


def _network_params(
    artifact_path,
    plasticity_mode="neuron_gains",
    operator_backend="native_csr",
):
    return {
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


def _build(
    artifact_path,
    plasticity_mode="neuron_gains",
    num_seqs=2,
    operator_backend="native_csr",
):
    builder = ConnectomeBuilder()
    builder.load(
        _network_params(
            artifact_path,
            plasticity_mode,
            operator_backend,
        )
    )
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
        network.recurrent_gain * network.incoming_gains() * recurrent
        + drive
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
        "extra_params",
    ):
        gradient = dict(network.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()
    assert torch.allclose(network.incoming_gains(), torch.ones(7))
    assert torch.allclose(network.outgoing_gains(), torch.ones(7))
    assert torch.allclose(network.leaks(), torch.full((7,), 0.5))


def test_frozen_core_and_global_builder_checkpoint_round_trip(
    artifact_path, tmp_path
) -> None:
    frozen = _build(artifact_path, plasticity_mode="frozen_core")
    for name in (
        "incoming_gain_raw",
        "outgoing_gain_raw",
        "leak_raw",
        "recurrent_bias",
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


def test_deployment_rl_player_loads_connectome_checkpoint(
    artifact_path, tmp_path
) -> None:
    pytest.importorskip("gym")
    from deployment.rl_player import RlPlayer

    network_params = _network_params(artifact_path)
    params = {
        "seed": 5,
        "algo": {"name": "a2c_continuous"},
        "model": {"name": "continuous_a2c_logstd"},
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

    player = RlPlayer(
        num_observations=5,
        num_actions=2,
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device="cpu",
        num_envs=1,
    )
    action = player.get_normalized_action(torch.zeros(1, 5), deterministic_actions=True)
    assert action.shape == (1, 2)
    assert torch.isfinite(action).all()
