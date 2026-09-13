from __future__ import annotations

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir


def test_all_actor_profiles_compose_with_sapg_and_asymmetric_critic() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    import isaacgymenvs  # noqa: F401 - registers OmegaConf resolvers

    profiles = {
        "SimToolRealLSTMAsymmetricSAPG": "actor_critic",
        "SimToolRealConnectomeSAPG": "connectome_actor_critic",
        "SimToolRealConnectomeFrozenSAPG": "connectome_actor_critic",
        "SimToolRealConnectomeRewiredSAPG": "connectome_actor_critic",
        "SimToolRealConnectomeRandomSAPG": "connectome_actor_critic",
    }
    with initialize_config_dir(
        version_base="1.1", config_dir=str(repository_root / "isaacgymenvs/cfg")
    ):
        for profile, expected_network in profiles.items():
            config = compose(
                config_name="config",
                overrides=["task=SimToolRealLSTMAsymmetric", f"train={profile}"],
            )
            assert config.train.params.network.name == expected_network
            assert (
                config.train.params.network.space.continuous.fixed_sigma == "coef_cond"
            )
            assert config.train.params.config.expl_type == "mixed_expl_learn_param"
            assert (
                config.train.params.config.central_value_config.network.name
                == "actor_critic"
            )
            if expected_network == "connectome_actor_critic":
                assert (
                    config.train.params.network.connectome.observations.policy_size
                    == 140
                )
                assert config.train.params.network.connectome.expected.neurons == 4310
                assert (
                    config.train.params.network.connectome.operator_backend
                    == "native_csr"
                )


def test_suite_contracts_own_required_execution_settings() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    suite_directory = repository_root / "configs/connectome/suites"
    smoke = yaml.safe_load((suite_directory / "smoke.yaml").read_text())
    full = yaml.safe_load((suite_directory / "full_training.yaml").read_text())
    assert smoke["training"]["num_envs"] == 384
    assert smoke["training"]["sapg_block_size"] == 64
    assert smoke["training"]["epochs"] == 2
    assert smoke["training"]["central_critic_minibatch_size"] == 1536
    assert smoke["training"]["wandb"]["enabled"] is False
    assert full["training"]["num_envs"] == 24576
    assert full["training"]["sapg_block_size"] == 4096
    assert len(full["training"]["train_profiles"]) == 5
    assert len(full["training"]["seeds"]) >= 3
    for suite in (smoke, full):
        training = suite["training"]
        assert training["num_envs"] // training["sapg_block_size"] == 6
        assert "gpu_assignments" in training
        assert "checkpoint" in training
        assert "wandb" in training


def test_backend_profile_is_yaml_owned_and_covers_all_operators() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    suite_directory = repository_root / "configs/connectome/suites"
    profiling_directory = repository_root / "configs/connectome/profiling"
    suite = yaml.safe_load((suite_directory / "backend_profiling.yaml").read_text())
    profile = yaml.safe_load(
        (profiling_directory / "recurrent_backends.yaml").read_text()
    )
    assert suite["stages"] == ["prepare", "profile"]
    assert suite["profiling"]["config"].endswith("recurrent_backends.yaml")
    assert {actor["operator_backend"] for actor in profile["actors"]} == {
        "native_csr",
        "native_coo",
        "dense",
        "torch_sparse",
    }
    assert {
        (shape["num_sequences"], shape["sequence_length"])
        for shape in profile["shapes"]
    } == {
        (384, 1),
        (384, 16),
    }
