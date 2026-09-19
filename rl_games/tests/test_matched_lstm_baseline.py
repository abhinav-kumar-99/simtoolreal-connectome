from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import isaacgymenvs  # noqa: F401 - registers OmegaConf resolvers
from rl_games.algos_torch import model_builder


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COEFFICIENT_IDS = torch.tensor([50.0, 40.0, 30.0, 20.0, 10.0, 0.0])


def _params() -> dict:
    with initialize_config_dir(
        version_base="1.1",
        config_dir=str(REPOSITORY_ROOT / "isaacgymenvs/cfg"),
    ):
        config = compose(
            config_name="config",
            overrides=[
                "task=SimToolRealLSTMAsymmetric",
                "train=SimToolRealLSTM323MatchedGaussianSigma3SAPG",
            ],
        )
    return OmegaConf.to_container(config.train.params.network, resolve=True)


def _network(params: dict | None = None, num_seqs: int = 2):
    return model_builder.NetworkBuilder().load(params or _params()).build(
        "matched_lstm",
        actions_num=29,
        input_shape=(141,),
        num_seqs=num_seqs,
        value_size=1,
        type="extra_param",
        coef_ids=COEFFICIENT_IDS,
        coef_id_idx=140,
        param_size=32,
    )


def _observations(num_seqs: int, sequence_length: int) -> torch.Tensor:
    observations = torch.randn(num_seqs, sequence_length, 141)
    observations[:, :, 140] = COEFFICIENT_IDS[
        torch.arange(num_seqs) % len(COEFFICIENT_IDS)
    ].unsqueeze(1)
    return observations.reshape(num_seqs * sequence_length, 141)


def test_matched_lstm_profile_has_exact_actor_count_and_standard_shape() -> None:
    network = _network()
    assert sum(parameter.numel() for parameter in network.parameters()) == 733790
    assert all(parameter.requires_grad for parameter in network.parameters())
    assert network.rnn_units == 323
    assert network.rnn_layers == 1
    assert network.units == [256]
    assert network.is_rnn_before_mlp
    assert network.rnn_ln
    assert "internal_steps" not in _params()["rnn"]


@pytest.mark.parametrize(
    "suite_name",
    [
        "ppo_lstm323_matched_gaussian_lf_entropy1x_sigma3_smoke.yaml",
        "ppo_lstm323_matched_gaussian_lf_entropy1x_sigma3_100b.yaml",
    ],
)
def test_matched_lstm_suites_disable_auxiliary_actor_value_loss(
    suite_name: str,
) -> None:
    suite_path = REPOSITORY_ROOT / "configs/connectome/suites" / suite_name
    with suite_path.open() as stream:
        suite = yaml.safe_load(stream)

    assert suite["training"]["overrides"][
        "++train.params.config.use_experimental_cv"
    ] is False
    assert "no-actor-value-loss" in suite["training"]["wandb"]["tags"]


def test_official_lstm_replacement_reproduces_repo_launch_with_seed_42() -> None:
    suite_path = (
        REPOSITORY_ROOT
        / "configs/connectome/suites/ppo_official_repo_lstm_sapg_seed42.yaml"
    )
    with suite_path.open() as stream:
        suite = yaml.safe_load(stream)

    training = suite["training"]
    assert training["train_profiles"] == [
        {
            "name": "official_lstm_sapg",
            "train_profile": "SimToolRealLSTMAsymmetricSAPG",
        }
    ]
    assert training["seeds"] == [42]
    assert training["gpu_assignments"] == [1]
    assert training["num_envs"] == 24576
    assert training["sapg_block_size"] == 4096
    assert training["minibatch_size"] == 98304
    assert training["central_critic_minibatch_size"] == 98304
    assert training["epochs"] == 1000000
    assert "rollout_accumulation_steps" not in training
    assert "actor_microbatch_size" not in training
    assert training["overrides"][
        "++train.params.config.use_experimental_cv"
    ] is True
    assert training["overrides"]["train.params.config.expl_reward_coef_scale"] == 0.002
    assert training["overrides"]["task.env.forceScale"] == 20.0
    assert training["overrides"]["task.env.torqueScale"] == 2.0


def test_official_lstm_fly_environment_match_retains_six_sapg_blocks() -> None:
    suite_path = (
        REPOSITORY_ROOT
        / "configs/connectome/suites/ppo_official_repo_lstm_sapg_seed42_env12288.yaml"
    )
    with suite_path.open() as stream:
        suite = yaml.safe_load(stream)

    training = suite["training"]
    assert training["seeds"] == [42]
    assert training["num_envs"] == 12288
    assert training["sapg_block_size"] == 2048
    assert training["num_envs"] // training["sapg_block_size"] == 6
    assert training["minibatch_size"] == 98304
    assert "rollout_accumulation_steps" not in training
    assert training["overrides"][
        "++train.params.config.use_experimental_cv"
    ] is True


def test_rollout_kl_matched_pair_retains_lf_and_one_scheduler_decision() -> None:
    suite_path = (
        REPOSITORY_ROOT
        / "configs/connectome/suites/"
        "ppo_lstm323_fly1952_asymmetric_noaux_mlp128x32x32_rollout_kl_100b.yaml"
    )
    with suite_path.open() as stream:
        suite = yaml.safe_load(stream)

    training = suite["training"]
    assert training["gpu_assignments"] == [0, 1]
    assert training["max_parallel"] == 2
    assert training["num_envs"] == 12288
    assert training["minibatch_size"] == 49152
    assert training["actor_microbatch_size"] == 49152
    assert training["overrides"]["train.params.config.schedule_type"] == "rollout"
    assert training["overrides"]["train.params.config.use_others_experience"] == "lf"
    assert training["overrides"]["train.params.config.off_policy_ratio"] == 1.0
    assert training["overrides"][
        "++train.params.config.use_experimental_cv"
    ] is False


def test_forward_is_one_lstm_transition_per_environment_timestep() -> None:
    torch.manual_seed(12)
    num_seqs, sequence_length = 2, 4
    network = _network(num_seqs=num_seqs)
    observations = _observations(num_seqs, sequence_length)
    states = tuple(torch.randn_like(state) for state in network.get_default_rnn_state())

    actual_mu, actual_sigma, actual_value, actual_states = network(
        {
            "obs": observations,
            "rnn_states": tuple(state.clone() for state in states),
            "dones": None,
            "seq_length": sequence_length,
        }
    )

    coefficient_rows = (
        observations[:, 140, None] == COEFFICIENT_IDS[None]
    ).float().argmax(dim=1)
    embedded = torch.cat(
        [observations[:, :140], network.extra_params[coefficient_rows]], dim=1
    )
    recurrent_input = embedded.reshape(num_seqs, sequence_length, -1).transpose(0, 1)
    recurrent_output, expected_states = network.rnn.rnn(
        recurrent_input, tuple(state.clone() for state in states)
    )
    features = recurrent_output.transpose(0, 1).reshape(
        num_seqs * sequence_length, 323
    )
    features = network.actor_mlp(network.layer_norm(features))
    expected_mu = network.mu_act(network.mu(features))
    expected_value = network.value_act(network.value(features))
    expected_sigma = network._bound_log_sigma(network.sigma[coefficient_rows])

    torch.testing.assert_close(actual_mu, expected_mu)
    torch.testing.assert_close(actual_sigma, expected_sigma)
    torch.testing.assert_close(actual_value, expected_value)
    for actual, expected in zip(actual_states, expected_states):
        torch.testing.assert_close(actual, expected)


def test_generic_lstm_sigma_cap_matches_connectome_contract() -> None:
    network = _network()
    observations = _observations(2, 1)
    with torch.no_grad():
        _, initial_log_sigma, _, _ = network(
            {"obs": observations, "rnn_states": network.get_default_rnn_state()}
        )
    torch.testing.assert_close(
        initial_log_sigma.exp(),
        torch.ones_like(initial_log_sigma),
        atol=1e-7,
        rtol=0,
    )

    with torch.no_grad():
        network.sigma.fill_(4.0)
    _, bounded_log_sigma, _, _ = network(
        {"obs": observations, "rnn_states": network.get_default_rnn_state()}
    )
    bounded_sigma = bounded_log_sigma.exp()
    assert torch.all(bounded_sigma < 3.0)
    assert torch.all(bounded_sigma > 2.8)
    bounded_log_sigma.sum().backward()
    assert network.sigma.grad is not None
    assert torch.isfinite(network.sigma.grad).all()
    assert (network.sigma.grad[:2] > 0).all()
    assert torch.count_nonzero(network.sigma.grad[2:]) == 0


@pytest.mark.parametrize("maximum", [1.0, 0.5, float("nan")])
def test_generic_lstm_rejects_invalid_sigma_cap(maximum: float) -> None:
    params = _params()
    params["space"]["continuous"]["max_sigma"] = maximum
    with pytest.raises(ValueError, match="max_sigma"):
        _network(params)
