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
        "SimToolRealConnectomeGainsSAPG": "connectome_actor_critic",
        "SimToolRealConnectomeLowRankSAPG": "connectome_actor_critic",
        "SimToolRealConnectomeEdgewiseSAPG": "connectome_actor_critic",
        "SimToolRealConnectomeGainsDynamicsSAPG": "connectome_actor_critic",
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
                    == "triton_fused"
                )
                adaptation = config.train.params.network.connectome.adaptation
                assert "plasticity_mode" not in config.train.params.network.connectome
                if profile == "SimToolRealConnectomeSAPG":
                    assert adaptation.weight_mode == "adapters_only"
                    assert adaptation.learn_dynamics is False
                if profile in {
                    "SimToolRealConnectomeRandomSAPG",
                    "SimToolRealConnectomeRewiredSAPG",
                    "SimToolRealConnectomeGainsDynamicsSAPG",
                }:
                    assert adaptation.weight_mode == "neuron_gains"
                    assert adaptation.learn_dynamics is True


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


def test_custom_smoke_matrix_and_case_override_routing(tmp_path):
    from scripts.run_connectome_suite import _training_overrides

    repository_root = Path(__file__).resolve().parents[2]
    suite = yaml.safe_load(
        (repository_root / "configs/connectome/suites/custom_smoke.yaml").read_text()
    )
    training = suite["training"]
    cases = training["train_profiles"]
    assert len(cases) == len({case["name"] for case in cases}) == 10
    assert training["epochs"] == 2 and training["num_envs"] == 384
    for case in cases:
        merged = dict(
            training, overrides={**training["overrides"], **case["overrides"]}
        )
        overrides = _training_overrides(
            merged, case["train_profile"], 42, case["name"], tmp_path
        )
        assert any(
            item.startswith("train.params.network.connectome.operator_backend=")
            for item in overrides
        )


def test_checkpoint_verification_rejects_partial_training(tmp_path):
    import pytest
    import torch

    from scripts.run_connectome_suite import _verify_checkpoint

    checkpoint = tmp_path / "partial.pth"
    torch.save({"epoch": 1, "optimizer": {"state": {0: {"step": 1}}}}, checkpoint)
    config = tmp_path / "resolved.yaml"
    config.write_text("train:\n  params:\n    config:\n      max_epochs: 2\n")
    with pytest.raises(RuntimeError, match="below requested"):
        _verify_checkpoint(checkpoint, config, "cpu")


def test_one_million_step_suite_preserves_single_gpu_paper_batches() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    suite = yaml.safe_load(
        (repository_root / "configs/connectome/suites/adaptation_1m.yaml").read_text()
    )
    training = suite["training"]
    assert training["gpu_assignments"] == [0, 1]
    assert training["max_parallel"] == 2
    assert training["seeds"] == [42]
    assert training["num_envs"] == 12288
    assert training["sapg_block_size"] == 2048
    assert training["minibatch_size"] == 49152
    assert training["central_critic_minibatch_size"] == 49152
    assert training["epochs"] == 5
    assert training["max_frames"] == 1_000_000
    assert training["num_envs"] * 16 * training["epochs"] <= training["max_frames"]
    assert training["num_envs"] * 16 * (training["epochs"] + 1) > training["max_frames"]
    assert len(training["train_profiles"]) == 5


def test_capped_evaluation_contract_owns_metrics_and_videos() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    evaluation = yaml.safe_load(
        (
            repository_root / "configs/connectome/evaluation/adaptation_1m.yaml"
        ).read_text()
    )
    assert evaluation["gpu_assignments"] == [0, 1]
    assert evaluation["max_parallel"] == 2
    assert evaluation["episodes_per_case"] == 1
    assert evaluation["metrics"]["paper_task_progress"]["success_tolerance_m"] == 0.02
    assert (
        evaluation["metrics"]["repository_avg_goal_pct"]["success_tolerance_m"]
        == 0.01
    )
    assert evaluation["videos"]["metric"] == "paper_task_progress"
    assert len(evaluation["eval_cases"]) == 3
