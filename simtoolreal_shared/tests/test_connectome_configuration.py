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
                assert config.train.params.config.kl_threshold == 0.004
                assert config.train.params.config.max_lr == 1e-3
                assert config.train.params.config.min_lr == 1e-6
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


def test_compact_profiles_compose_with_explicit_adaptation_modes() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    import isaacgymenvs  # noqa: F401 - registers OmegaConf resolvers

    profiles = {
        "SimToolRealConnectome1952AdaptersSAPG": "adapters_only",
        "SimToolRealConnectome1952GainsSAPG": "neuron_gains",
    }
    with initialize_config_dir(
        version_base="1.1", config_dir=str(repository_root / "isaacgymenvs/cfg")
    ):
        for profile, weight_mode in profiles.items():
            config = compose(
                config_name="config",
                overrides=["task=SimToolRealLSTMAsymmetric", f"train={profile}"],
            )
            connectome = config.train.params.network.connectome
            assert config.train.params.config.kl_threshold == 0.004
            assert config.train.params.config.max_lr == 1e-3
            assert connectome.operator_backend == "triton_fused"
            assert connectome.artifact_path.endswith("malecns_1952/biological.npz")
            assert dict(connectome.expected) == {
                "neurons": 1952,
                "edges": 33720,
                "sensory_neurons": 384,
                "descending_neurons": 157,
                "motor_neurons": 135,
            }
            assert connectome.adaptation.weight_mode == weight_mode
            assert connectome.adaptation.learn_dynamics is False


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


def test_one_billion_step_suite_maps_one_policy_to_each_gpu() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    suite = yaml.safe_load(
        (repository_root / "configs/connectome/suites/adaptation_1b.yaml").read_text()
    )
    training = suite["training"]
    assert [case["name"] for case in training["train_profiles"]] == [
        "adapters_only",
        "neuron_gains",
    ]
    assert training["gpu_assignments"] == [0, 1]
    assert training["max_parallel"] == 2
    assert training["seeds"] == [42]
    assert training["num_envs"] == 12288
    assert training["sapg_block_size"] == 2048
    assert training["minibatch_size"] == 49152
    assert training["central_critic_minibatch_size"] == 49152
    assert training["epochs"] == 5086
    assert training["max_frames"] == 1_000_000_000
    steps_per_epoch = training["num_envs"] * 16
    assert steps_per_epoch * training["epochs"] == 999_948_288
    assert steps_per_epoch * (training["epochs"] + 1) > training["max_frames"]


def test_release_settings_suite_preserves_original_logical_update_schedule() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    suite = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/suites/adaptation_1b_release_settings.yaml"
        ).read_text()
    )
    training = suite["training"]
    assert training["num_envs"] == 12288
    assert training["sapg_block_size"] == 2048
    assert training["rollout_accumulation_steps"] == 2
    assert training["minibatch_size"] == 98304
    assert training["central_critic_minibatch_size"] == 98304
    assert training["actor_microbatch_size"] == 24576
    assert training["central_critic_microbatch_size"] == 24576
    assert training["epochs"] == 2543
    assert (
        training["num_envs"]
        * training["rollout_accumulation_steps"]
        * 16
        * training["epochs"]
        == 999_948_288
    )
    overrides = training["overrides"]
    assert overrides["train.params.config.expl_reward_coef_scale"] == 0.005
    assert overrides["task.env.forceScale"] == 2.0
    assert overrides["task.env.forceDecay"] == 0.99
    assert overrides["task.env.torqueScale"] == 0.0


def test_release_settings_suite_routes_microbatches_without_changing_logical_batch(
    tmp_path,
) -> None:
    from scripts.run_connectome_suite import _training_overrides

    repository_root = Path(__file__).resolve().parents[2]
    suite = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/suites/adaptation_1b_release_settings.yaml"
        ).read_text()
    )
    training = suite["training"]
    overrides = _training_overrides(
        training,
        training["train_profiles"][0]["train_profile"],
        42,
        "adapters_only",
        tmp_path,
    )
    assert "train.params.config.minibatch_size=98304" in overrides
    assert (
        "train.params.config.central_value_config.minibatch_size=98304" in overrides
    )
    assert "++train.params.config.rollout_accumulation_steps=2" in overrides
    assert "++train.params.config.microbatch_size=24576" in overrides
    assert (
        "++train.params.config.central_value_config.microbatch_size=24576"
        in overrides
    )


def test_rollout_accumulation_smoke_preserves_four_logical_batches() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    suite = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/suites/rollout_accumulation_smoke.yaml"
        ).read_text()
    )
    training = suite["training"]
    frames_per_update_phase = (
        training["num_envs"] * 16 * training["rollout_accumulation_steps"]
    )
    assert frames_per_update_phase == 12288
    assert frames_per_update_phase // training["minibatch_size"] == 4
    assert frames_per_update_phase * training["epochs"] == training["max_frames"]


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
    assert evaluation["action_selection"] == "mean"
    assert evaluation["metrics"]["paper_task_progress"]["success_tolerance_m"] == 0.02
    assert (
        evaluation["metrics"]["repository_avg_goal_pct"]["success_tolerance_m"]
        == 0.01
    )
    assert evaluation["videos"]["metric"] == "paper_task_progress"
    assert len(evaluation["eval_cases"]) == 3


def test_partial_high_resolution_evaluation_uses_explicit_checkpoints() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    evaluation = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/evaluation/adaptation_1b_partial_highres.yaml"
        ).read_text()
    )
    assert set(evaluation["policy_sources"]) == {"adapters_only", "neuron_gains"}
    assert evaluation["action_selection"] == "mean"
    assert evaluation["videos"]["camera_resolution_reduction_factor"] == 2
    assert evaluation["videos"]["frame_interval"] == 3
    assert len(evaluation["eval_cases"]) == 3


def test_final_high_resolution_evaluation_uses_completed_checkpoints() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    evaluation = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/evaluation/adaptation_1b_final_highres.yaml"
        ).read_text()
    )
    assert set(evaluation["policy_sources"]) == {"adapters_only", "neuron_gains"}
    assert evaluation["action_selection"] == "mean"
    assert all(
        "ep_2543" in source["checkpoint_path"]
        for source in evaluation["policy_sources"].values()
    )
    assert set(evaluation["metrics"]) == {
        "paper_task_progress",
        "repository_avg_goal_pct",
    }
    assert evaluation["metrics"]["paper_task_progress"]["success_tolerance_m"] == 0.02
    assert (
        evaluation["metrics"]["repository_avg_goal_pct"]["success_tolerance_m"]
        == 0.01
    )
    assert evaluation["videos"]["camera_resolution_reduction_factor"] == 2
    assert evaluation["videos"]["fps"] == 20
    assert evaluation["videos"]["frame_interval"] == 3
    assert len(evaluation["eval_cases"]) == 3


def test_hundred_billion_gains_suite_isolated_update_timing() -> None:
    from scripts.run_connectome_suite import _training_overrides

    repository_root = Path(__file__).resolve().parents[2]
    suite = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/suites/adaptation_100b_gains_update_timing.yaml"
        ).read_text()
    )
    training = suite["training"]
    assert training["max_frames"] == 100_000_000_000
    assert training["inference_checkpoint_interval_frames"] == 250_000_000
    assert training["gpu_assignments"] == [0, 1]
    assert training["max_parallel"] == 2
    assert training["seeds"] == [42]
    assert all(
        case["train_profile"] == "SimToolRealConnectomeGainsSAPG"
        for case in training["train_profiles"]
    )
    expected = {
        "gains_new_update_timing": (254_313, 2, 98_304),
        "gains_old_update_timing": (508_626, 1, 49_152),
    }
    for case in training["train_profiles"]:
        merged = dict(
            training,
            overrides={**training["overrides"], **case["overrides"]},
        )
        overrides = _training_overrides(
            merged,
            case["train_profile"],
            42,
            case["name"],
            repository_root / "test-output",
        )
        epochs, accumulation, minibatch = expected[case["name"]]
        resolved = {}
        for override in overrides:
            key, value = override.lstrip("+").split("=", 1)
            resolved[key] = value
        assert int(resolved["train.params.config.max_epochs"]) == epochs
        assert (
            int(resolved.get("train.params.config.rollout_accumulation_steps", 1))
            == accumulation
        )
        assert int(resolved["train.params.config.minibatch_size"]) == minibatch
        assert (
            12_288 * 16 * accumulation * epochs == 99_999_940_608
        )


def test_hundred_billion_milestone_evaluation_is_mean_action_high_resolution() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/evaluation/adaptation_100b_gains_milestones.yaml"
        ).read_text()
    )
    assert config["max_frames"] == 100_000_000_000
    assert config["milestone_interval_frames"] == 250_000_000
    assert len(config["policies"]) == 2
    assert config["evaluation"]["action_selection"] == "mean"
    assert (
        config["evaluation"]["videos"]["camera_resolution_reduction_factor"]
        == 2
    )
    assert len(config["evaluation"]["eval_cases"]) == 3


def test_compact_adapters_replacement_owns_matched_training_and_evaluation() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    suite = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/suites/adaptation_1952_adapters_100b.yaml"
        ).read_text()
    )
    training = suite["training"]
    assert training["train_profiles"] == [
        {
            "name": "adapters_only_old_timing",
            "train_profile": "SimToolRealConnectome1952AdaptersSAPG",
        }
    ]
    assert training["gpu_assignments"] == [1]
    assert training["max_parallel"] == 1
    assert training["num_envs"] == 12_288
    assert training["sapg_block_size"] == 2_048
    assert training["rollout_accumulation_steps"] == 1
    assert training["minibatch_size"] == 49_152
    assert training["central_critic_minibatch_size"] == 49_152
    assert training["actor_microbatch_size"] == 49_152
    assert training["central_critic_microbatch_size"] == 49_152
    assert training["max_frames"] == 100_000_000_000
    assert 12_288 * 16 * training["epochs"] == 99_999_940_608

    evaluation = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/evaluation/adaptation_1952_adapters_milestones.yaml"
        ).read_text()
    )
    assert evaluation["training_suite_name"] == suite["name"]
    assert evaluation["training_suite_directory"] == suite["output_directory"]
    assert evaluation["milestone_interval_frames"] == 250_000_000
    assert evaluation["policies"][0]["gpu"] == 1
    assert evaluation["evaluation"]["action_selection"] == "mean"
    assert (
        evaluation["evaluation"]["videos"]["camera_resolution_reduction_factor"]
        == 2
    )
    assert len(evaluation["evaluation"]["eval_cases"]) == 3
