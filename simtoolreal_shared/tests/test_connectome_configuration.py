from __future__ import annotations

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir


def test_four_update_beta_gaussian_suites_preserve_requested_contract():
    from scripts.run_connectome_suite import _compose_resolved, _training_overrides

    root = Path(__file__).resolve().parents[2]
    for suffix, epochs, cap in [('smoke', 2, 393216), ('1b', 5086, 1000000000), ('100b', 508626, 100000000000)]:
        suite = yaml.safe_load((root / f'configs/connectome/suites/ppo_1952_4update_beta_gaussian_{suffix}.yaml').read_text())
        training = suite['training']
        assert training['gpu_assignments'] == [0, 1]
        assert training['max_parallel'] == 2
        assert training['checkpoint']['mode'] == 'none'
        assert training['on_existing'] == 'fail'
        assert [p['name'] for p in training['train_profiles']] == ['beta', 'gaussian']
        for entry in training['train_profiles']:
            case_training = dict(training)
            case_training['overrides'] = {**training['overrides'], **entry.get('overrides', {})}
            cfg = _compose_resolved(_training_overrides(case_training, entry['train_profile'], 42, entry['name'], root / suite['output_directory']))
            if suffix == '100b':
                assert cfg.train.params.config.checkpoint_load_mode == 'resume_training_state'
                assert f"_1b_{entry['name']}_seed42" in cfg.checkpoint
            actor = cfg.train.params.network
            config = cfg.train.params.config
            assert actor.connectome.dynamics.neural_updates == 4
            assert actor.connectome.dynamics.initial_leak == .5
            assert actor.connectome.expected.neurons == 1952
            assert actor.connectome.adaptation.weight_mode == 'adapters_only'
            assert actor.connectome.adaptation.learn_dynamics is False
            assert actor.connectome.interface_projections.architecture == 'mlp'
            assert config.use_experimental_cv is True
            assert config.use_others_experience == 'none'
            assert config.central_value_config is not None
            assert config.max_lr == .001 and config.kl_threshold == .004
            assert config.learning_rate == .0001 and config.expl_reward_coef_scale == .005
            assert config.seq_length == config.horizon_length == 16
            assert config.mini_epochs == 2
            assert config.max_epochs == epochs and config.max_frames == cap
            assert cfg.task.env.numEnvs == 12288
            assert config.minibatch_size == config.microbatch_size == 49152
            assert config.central_value_config.minibatch_size == 49152
            if entry['name'] == 'beta':
                assert cfg.train.params.model.name == 'continuous_a2c_beta'
                assert actor.space.continuous.distribution == 'beta'
                assert actor.space.continuous.beta_initial_shape == 2
            else:
                assert cfg.train.params.model.name == 'continuous_a2c_logstd'
                assert actor.space.continuous.fixed_sigma == 'coef_cond'


def test_four_update_video_watchers_match_training_timing_and_gpu() -> None:
    root = Path(__file__).resolve().parents[2]
    base = root / 'configs/connectome/evaluation'
    expected = {'beta': 0, 'gaussian': 1}
    for name, gpu in expected.items():
        config = yaml.safe_load((base / f'ppo_1952_4update_{name}_100b_milestones.yaml').read_text())
        policy = config['policies'][0]
        assert config['training_suite_name'] == 'ppo_1952_4update_beta_gaussian_100b'
        assert config['max_frames'] == 100_000_000_000
        assert config['milestone_interval_frames'] == 250_000_000
        assert config['watch_until_complete'] is True
        assert policy['name'] == name and policy['gpu'] == gpu
        assert policy['policy_config_path'].endswith(
            f'00_ppo_1952_4update_beta_gaussian_100b_{name}_seed42/resolved_config.yaml'
        )
        resolved = yaml.safe_load((root / policy['policy_config_path']).read_text())
        assert resolved['train']['params']['network']['connectome']['dynamics']['neural_updates'] == 4
        evaluation = config['evaluation']
        assert evaluation['action_selection'] == 'mean'
        assert evaluation['episodes_per_case'] == 1
        assert len(evaluation['eval_cases']) == 3
        assert evaluation['videos']['camera_resolution_reduction_factor'] == 2


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
        "SimToolRealConnectomeMLPSAPG": "connectome_actor_critic",
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
                assert config.train.params.config.max_lr == 1e-2
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
                projections = (
                    config.train.params.network.connectome.interface_projections
                )
                expected_projection = (
                    "mlp" if profile == "SimToolRealConnectomeMLPSAPG" else "linear"
                )
                assert projections.architecture == expected_projection
                assert projections.hidden_size == 256
                assert projections.activation == "elu"
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
        "SimToolRealConnectome1952AdaptersSAPG": ("adapters_only", "linear"),
        "SimToolRealConnectome1952GainsSAPG": ("neuron_gains", "linear"),
        "SimToolRealConnectome1952AdaptersMLPSAPG": ("adapters_only", "mlp"),
        "SimToolRealConnectome1952AdaptersMLPTanhSAPG": (
            "adapters_only",
            "mlp",
        ),
        "SimToolRealConnectome1952GainsMLPSAPG": ("neuron_gains", "mlp"),
    }
    with initialize_config_dir(
        version_base="1.1", config_dir=str(repository_root / "isaacgymenvs/cfg")
    ):
        for profile, (weight_mode, projection_architecture) in profiles.items():
            config = compose(
                config_name="config",
                overrides=["task=SimToolRealLSTMAsymmetric", f"train={profile}"],
            )
            connectome = config.train.params.network.connectome
            assert config.train.params.config.kl_threshold == 0.004
            assert config.train.params.config.max_lr == 1e-2
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
            assert (
                connectome.interface_projections.architecture
                == projection_architecture
            )
            assert connectome.interface_projections.hidden_size == 256
            assert connectome.interface_projections.activation == "elu"
            expected_model = (
                "continuous_a2c_tanh_logstd"
                if profile == "SimToolRealConnectome1952AdaptersMLPTanhSAPG"
                else "continuous_a2c_logstd"
            )
            assert config.train.params.model.name == expected_model
            if expected_model == "continuous_a2c_tanh_logstd":
                assert config.train.params.model.entropy_samples == 1
                assert "log_std_bounds" not in config.train.params.model


def test_asymmetric_mlp_profile_and_smoke_suite_compose() -> None:
    from scripts.run_connectome_suite import _compose_resolved, _training_overrides

    root = Path(__file__).resolve().parents[2]
    suite_path = (
        root
        / "configs/connectome/suites/ppo_1952_4update_gaussian_lf_entropy1x_sigma3_all_neuron_readout_mlp128x32x32_smoke.yaml"
    )
    suite = yaml.safe_load(suite_path.read_text())
    training = suite["training"]
    entry = training["train_profiles"][0]
    config = _compose_resolved(
        _training_overrides(
            training,
            entry["train_profile"],
            training["seeds"][0],
            entry["name"],
            root / suite["output_directory"],
        )
    )

    connectome = config.train.params.network.connectome
    projections = connectome.interface_projections
    assert projections.architecture == "mlp"
    assert projections.hidden_size == 256
    assert projections.sensory_hidden_size == 128
    assert projections.descending_hidden_size == 32
    assert projections.readout_hidden_size == 32
    assert connectome.reservoir_readout.enabled is False
    assert connectome.reservoir_readout.feature_population == "all"
    assert connectome.dynamics.neural_updates == 4
    assert config.train.params.config.use_experimental_cv is True
    assert config.train.params.config.use_others_experience == "lf"
    assert config.train.params.config.kl_threshold == 0.004
    assert config.train.params.config.max_frames == 393_216
    assert config.task.env.numEnvs == 12_288


def test_structured_rotation6d_profiles_and_live_matched_suites_compose() -> None:
    from scripts.run_connectome_suite import _compose_resolved, _training_overrides

    root = Path(__file__).resolve().parents[2]
    suite_root = root / "configs/connectome/suites"
    pairs = (
        (
            "ppo_1952_4update_gaussian_lf_entropy1x_sigma3_100b.yaml",
            "ppo_1952_4update_structured_rot6d_gaussian_lf_entropy1x_sigma3_100b.yaml",
            "gaussian",
        ),
        (
            "ppo_1952_4update_restricted_beta_lf_100b.yaml",
            "ppo_1952_4update_structured_rot6d_restricted_beta_lf_100b.yaml",
            "beta",
        ),
    )
    for live_name, structured_name, distribution in pairs:
        live = yaml.safe_load((suite_root / live_name).read_text())
        structured = yaml.safe_load((suite_root / structured_name).read_text())
        # GPU placement is operational: both structured alternatives target GPU 1
        # because the dense capped-Gaussian control remains active on GPU 0.
        ignored = {"task_profile", "train_profiles", "wandb", "gpu_assignments"}
        assert {
            key: value
            for key, value in structured["training"].items()
            if key not in ignored
        } == {
            key: value
            for key, value in live["training"].items()
            if key not in ignored
        }
        assert structured["preparation"] == live["preparation"]
        assert structured["training"]["gpu_assignments"] == [1]
        assert structured["training"]["task_profile"] == (
            "SimToolRealLSTMAsymmetricRotation6D"
        )
        entry = structured["training"]["train_profiles"][0]
        config = _compose_resolved(
            _training_overrides(
                structured["training"],
                entry["train_profile"],
                42,
                entry["name"],
                root / structured["output_directory"],
            )
        )
        assert config.task.env.orientationObservationRepresentation == "rotation_6d"
        actor = config.train.params.network
        graph = actor.connectome
        assert graph.observations.policy_size == 144
        assert graph.observations.sensory_size == 102
        assert graph.observations.context_size == 30
        assert graph.observations.goal_size == 12
        assert list(graph.observations.sensory_ranges) == [[0, 87], [102, 117]]
        assert list(graph.observations.context_ranges) == [
            [87, 102],
            [117, 129],
            [141, 144],
        ]
        assert list(graph.observations.goal_ranges) == [[129, 141]]
        adapter = graph.structured_input_adapter
        assert adapter.mode == "grouped_linear"
        assert adapter.descending_projection.architecture == "mlp"
        assert adapter.descending_projection.hidden_size == 128
        assert len(adapter.groups) == 6
        assert graph.adaptation.weight_mode == "adapters_only"
        assert graph.adaptation.learn_dynamics is False
        assert graph.dynamics.neural_updates == 4
        assert config.train.params.config.use_experimental_cv is True
        assert config.train.params.config.use_others_experience == "lf"
        assert config.train.params.config.off_policy_ratio == 1.0
        assert config.train.params.config.kl_threshold == 0.004
        assert config.train.params.config.expl_reward_coef_scale == 0.005
        assert actor.space.continuous.distribution == distribution
        if distribution == "gaussian":
            assert actor.space.continuous.max_sigma == 3.0
        else:
            assert actor.space.continuous.beta_min_shape == 1.0
            assert actor.space.continuous.beta_initial_shape == 2.0


def test_fixed_reservoir_rotation6d_beta_suites_compose() -> None:
    from scripts.run_connectome_suite import _compose_resolved, _training_overrides

    root = Path(__file__).resolve().parents[2]
    suite_root = root / "configs/connectome/suites"
    smoke = yaml.safe_load(
        (suite_root / "ppo_1952_fixed_reservoir_rot6d_beta_lf_smoke.yaml").read_text()
    )
    full = yaml.safe_load(
        (suite_root / "ppo_1952_fixed_reservoir_rot6d_beta_lf_100b.yaml").read_text()
    )
    ignored = {
        "epochs",
        "max_frames",
        "inference_checkpoint_interval_frames",
        "save_frequency",
        "save_best_after",
        "wandb",
    }
    assert {
        key: value for key, value in smoke["training"].items() if key not in ignored
    } == {
        key: value for key, value in full["training"].items() if key not in ignored
    }
    assert smoke["preparation"] == full["preparation"]

    entry = smoke["training"]["train_profiles"][0]
    config = _compose_resolved(
        _training_overrides(
            smoke["training"],
            entry["train_profile"],
            42,
            entry["name"],
            root / smoke["output_directory"],
        )
    )
    assert config.task.env.orientationObservationRepresentation == "rotation_6d"
    assert config.task.env.numEnvs == 12288
    actor = config.train.params.network
    graph = actor.connectome
    assert graph.observations.policy_size == 144
    assert graph.expected.neurons == 1952
    assert graph.dynamics.neural_updates == 4
    assert graph.adaptation.weight_mode == "adapters_only"
    assert graph.adaptation.learn_dynamics is False
    assert "structured_input_adapter" not in graph
    fixed = graph.fixed_input_encoder
    assert fixed.mode == "population_code_v1"
    assert len(fixed.sensory_mappings) == 2
    assert list(fixed.sensory_mappings[1].source_range) == [29, 58]
    assert fixed.sensory_mappings[1].encoding == "opponent_tanh"
    assert len(fixed.descending_mappings) == 7
    assert list(fixed.descending_mappings[-1].source_range) == [129, 141]
    assert fixed.descending_mappings[-1].encoding == "opponent_tanh"
    readout = graph.reservoir_readout
    assert readout.enabled is True
    assert readout.feature_population == "motor"
    assert readout.condition_on_sapg is True
    assert readout.architecture == "mlp"
    assert readout.hidden_size == 128
    assert config.train.params.config.normalize_input is False
    assert config.train.params.config.use_experimental_cv is False
    assert config.train.params.config.use_others_experience == "lf"
    assert config.train.params.config.off_policy_ratio == 1.0
    assert config.train.params.config.kl_threshold == 0.004
    assert config.train.params.model.name == "continuous_a2c_beta"
    assert actor.space.continuous.distribution == "beta"
    assert actor.space.continuous.beta_min_shape == 1.0
    assert actor.space.continuous.beta_initial_shape == 2.0


def test_fixed_reservoir_gaussian_matches_live_gaussian_contract() -> None:
    from scripts.run_connectome_suite import _compose_resolved, _training_overrides

    root = Path(__file__).resolve().parents[2]
    suite_root = root / "configs/connectome/suites"
    live = yaml.safe_load(
        (
            suite_root
            / "ppo_1952_4update_gaussian_lf_entropy1x_sigma3_100b.yaml"
        ).read_text()
    )
    full = yaml.safe_load(
        (
            suite_root
            / "ppo_1952_fixed_reservoir_rot6d_gaussian_lf_entropy1x_sigma3_100b.yaml"
        ).read_text()
    )
    smoke = yaml.safe_load(
        (
            suite_root
            / "ppo_1952_fixed_reservoir_rot6d_gaussian_lf_entropy1x_sigma3_smoke.yaml"
        ).read_text()
    )
    ignored = {
        "task_profile",
        "train_profiles",
        "gpu_assignments",
        "wandb",
    }
    assert {
        key: value for key, value in full["training"].items() if key not in ignored
    } == {
        key: value for key, value in live["training"].items() if key not in ignored
    }
    budget_fields = {
        "epochs",
        "max_frames",
        "inference_checkpoint_interval_frames",
        "save_frequency",
        "save_best_after",
        "wandb",
    }
    assert {
        key: value
        for key, value in smoke["training"].items()
        if key not in budget_fields
    } == {
        key: value
        for key, value in full["training"].items()
        if key not in budget_fields
    }

    entry = full["training"]["train_profiles"][0]
    config = _compose_resolved(
        _training_overrides(
            full["training"],
            entry["train_profile"],
            42,
            entry["name"],
            root / full["output_directory"],
        )
    )
    actor = config.train.params.network
    graph = actor.connectome
    assert config.task.env.orientationObservationRepresentation == "rotation_6d"
    assert graph.fixed_input_encoder.mode == "population_code_v1"
    assert graph.reservoir_readout.enabled is True
    assert graph.reservoir_readout.condition_on_sapg is True
    assert graph.dynamics.neural_updates == 4
    assert config.train.params.model.name == "continuous_a2c_logstd"
    assert actor.space.continuous.distribution == "gaussian"
    assert actor.space.continuous.fixed_sigma == "coef_cond"
    assert actor.space.continuous.max_sigma == 3.0
    assert config.train.params.config.normalize_input is False
    assert config.train.params.config.use_experimental_cv is True
    assert config.train.params.config.use_others_experience == "lf"
    assert config.train.params.config.off_policy_ratio == 1.0
    assert config.train.params.config.expl_reward_coef_scale == 0.005
    assert config.train.params.config.kl_threshold == 0.004
    assert config.train.params.config.learning_rate == 0.0001
    assert config.train.params.config.max_lr == 0.001


def test_fixed_reservoir_lif_changes_only_recurrent_dynamics_contract() -> None:
    from scripts.run_connectome_suite import _compose_resolved, _training_overrides

    root = Path(__file__).resolve().parents[2]
    suite_root = root / "configs/connectome/suites"
    tanh = yaml.safe_load(
        (
            suite_root
            / "ppo_1952_fixed_reservoir_rot6d_gaussian_lf_entropy1x_sigma3_100b.yaml"
        ).read_text()
    )
    lif = yaml.safe_load(
        (
            suite_root
            / "ppo_1952_fixed_reservoir_rot6d_gaussian_lif_lf_entropy1x_sigma3_100b.yaml"
        ).read_text()
    )
    smoke = yaml.safe_load(
        (
            suite_root
            / "ppo_1952_fixed_reservoir_rot6d_gaussian_lif_lf_entropy1x_sigma3_smoke.yaml"
        ).read_text()
    )
    identity_fields = {"train_profiles", "wandb"}
    assert {
        key: value
        for key, value in lif["training"].items()
        if key not in identity_fields
    } == {
        key: value
        for key, value in tanh["training"].items()
        if key not in identity_fields
    }
    budget_fields = {
        "epochs",
        "max_frames",
        "inference_checkpoint_interval_frames",
        "save_frequency",
        "save_best_after",
        "on_existing",
        "wandb",
    }
    assert {
        key: value
        for key, value in smoke["training"].items()
        if key not in budget_fields
    } == {
        key: value
        for key, value in lif["training"].items()
        if key not in budget_fields
    }

    entry = lif["training"]["train_profiles"][0]
    config = _compose_resolved(
        _training_overrides(
            lif["training"],
            entry["train_profile"],
            42,
            entry["name"],
            root / lif["output_directory"],
        )
    )
    graph = config.train.params.network.connectome
    assert graph.dynamics.activation == "lif"
    assert graph.dynamics.neural_updates == 4
    assert graph.dynamics.control_frequency_hz == 60.0
    assert graph.dynamics.membrane_time_constant_ms == 10.0
    assert graph.dynamics.spike_threshold == 0.25
    assert graph.dynamics.refractory_period_ms == 2.0
    assert graph.dynamics.input_current_scale == 1.5
    assert graph.fixed_input_encoder.mode == "population_code_v1"
    assert graph.reservoir_readout.enabled is True
    assert config.train.params.model.name == "continuous_a2c_logstd"
    assert config.train.params.network.space.continuous.max_sigma == 3.0
    assert config.train.params.config.normalize_input is False
    assert config.train.params.config.use_experimental_cv is True


def test_fixed_reservoir_noaux_changes_only_auxiliary_value_flag() -> None:
    from scripts.run_connectome_suite import _compose_resolved, _training_overrides

    root = Path(__file__).resolve().parents[2]
    suite_root = root / "configs/connectome/suites"
    baseline = yaml.safe_load(
        (
            suite_root
            / "ppo_1952_fixed_reservoir_rot6d_gaussian_lf_entropy1x_sigma3_100b.yaml"
        ).read_text()
    )
    noaux = yaml.safe_load(
        (
            suite_root
            / "ppo_1952_fixed_reservoir_rot6d_gaussian_lf_entropy1x_sigma3_noaux_100b.yaml"
        ).read_text()
    )
    baseline_training = baseline["training"]
    noaux_training = noaux["training"]
    ignored = {"train_profiles", "wandb"}
    baseline_comparable = {
        key: value for key, value in baseline_training.items() if key not in ignored
    }
    noaux_comparable = {
        key: value for key, value in noaux_training.items() if key not in ignored
    }
    baseline_comparable["overrides"] = dict(baseline_comparable["overrides"])
    noaux_comparable["overrides"] = dict(noaux_comparable["overrides"])
    assert baseline_comparable["overrides"].pop(
        "++train.params.config.use_experimental_cv"
    ) is True
    assert noaux_comparable["overrides"].pop(
        "++train.params.config.use_experimental_cv"
    ) is False
    assert noaux_comparable == baseline_comparable

    entry = noaux_training["train_profiles"][0]
    config = _compose_resolved(
        _training_overrides(
            noaux_training,
            entry["train_profile"],
            42,
            entry["name"],
            root / noaux["output_directory"],
        )
    )
    assert config.train.params.config.use_experimental_cv is False
    assert config.train.params.network.connectome.dynamics.activation == "tanh"
    assert config.train.params.network.connectome.reservoir_readout.enabled is True
    assert config.train.params.config.central_value_config is not None


def test_distal_leg_profiles_compose_with_matched_policy_distributions() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    import isaacgymenvs  # noqa: F401 - registers OmegaConf resolvers

    profiles = {
        "SimToolRealConnectome262AdaptersMLPSAPG": "continuous_a2c_logstd",
        "SimToolRealConnectome262AdaptersMLPTanhSAPG": (
            "continuous_a2c_tanh_logstd"
        ),
    }
    with initialize_config_dir(
        version_base="1.1", config_dir=str(repository_root / "isaacgymenvs/cfg")
    ):
        for profile, expected_model in profiles.items():
            config = compose(
                config_name="config",
                overrides=["task=SimToolRealLSTMAsymmetric", f"train={profile}"],
            )
            graph = config.train.params.network.connectome
            assert dict(graph.expected) == {
                "neurons": 262,
                "edges": 3194,
                "sensory_neurons": 48,
                "descending_neurons": 2,
                "motor_neurons": 31,
            }
            assert graph.adaptation.weight_mode == "adapters_only"
            assert graph.adaptation.learn_dynamics is False
            assert graph.interface_projections.architecture == "mlp"
            assert config.train.params.model.name == expected_model


def test_suite_contracts_own_required_execution_settings() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    suite_directory = repository_root / "configs/connectome/suites"
    smoke = yaml.safe_load((suite_directory / "smoke.yaml").read_text())
    full = yaml.safe_load((suite_directory / "full_training.yaml").read_text())
    mlp_smoke = yaml.safe_load(
        (suite_directory / "mlp_projection_smoke.yaml").read_text()
    )
    assert smoke["training"]["num_envs"] == 384
    assert smoke["training"]["sapg_block_size"] == 64
    assert smoke["training"]["epochs"] == 2
    assert smoke["training"]["central_critic_minibatch_size"] == 1536
    assert smoke["training"]["wandb"]["enabled"] is False
    assert full["training"]["num_envs"] == 24576
    assert full["training"]["sapg_block_size"] == 4096
    assert len(full["training"]["train_profiles"]) == 5
    assert len(full["training"]["seeds"]) >= 3
    assert [
        profile["train_profile"]
        for profile in mlp_smoke["training"]["train_profiles"]
    ] == [
        "SimToolRealConnectome1952AdaptersMLPSAPG",
        "SimToolRealConnectome1952GainsMLPSAPG",
    ]
    assert mlp_smoke["training"]["num_envs"] == 384
    assert mlp_smoke["training"]["minibatch_size"] == 1536
    for suite in (smoke, full):
        training = suite["training"]
        assert training["num_envs"] // training["sapg_block_size"] == 6
        assert "gpu_assignments" in training
        assert "checkpoint" in training
        assert "wandb" in training


def test_mlp_adapters_replacement_matches_stable_ppo_contract() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config_root = repository_root / "configs/connectome"
    original = yaml.safe_load(
        (config_root / "suites/ppo_1952_kl004_100b.yaml").read_text()
    )
    replacement = yaml.safe_load(
        (config_root / "suites/ppo_1952_mlp_adapters_kl004_100b.yaml").read_text()
    )
    smoke = yaml.safe_load(
        (config_root / "suites/ppo_1952_mlp_adapters_kl004_smoke.yaml").read_text()
    )
    matched_keys = {
        "task_profile",
        "seeds",
        "num_envs",
        "sapg_block_size",
        "epochs",
        "max_frames",
        "minibatch_size",
        "central_critic_minibatch_size",
        "rollout_accumulation_steps",
        "actor_microbatch_size",
        "central_critic_microbatch_size",
        "inference_checkpoint_interval_frames",
        "save_frequency",
        "save_best_after",
        "checkpoint",
        "overrides",
    }
    for key in matched_keys:
        assert replacement["training"][key] == original["training"][key]
    assert replacement["training"]["train_profiles"] == [
        {
            "name": "adapters_mlp",
            "train_profile": "SimToolRealConnectome1952AdaptersMLPSAPG",
        }
    ]
    assert replacement["training"]["gpu_assignments"] == [1]
    assert replacement["training"]["max_parallel"] == 1
    assert smoke["training"]["train_profiles"] == replacement["training"][
        "train_profiles"
    ]
    for key in (
        "num_envs",
        "sapg_block_size",
        "minibatch_size",
        "central_critic_minibatch_size",
        "rollout_accumulation_steps",
        "actor_microbatch_size",
        "central_critic_microbatch_size",
        "overrides",
    ):
        assert smoke["training"][key] == replacement["training"][key]
    assert smoke["training"]["epochs"] == 2
    assert smoke["training"]["max_frames"] == 393_216

    evaluation = yaml.safe_load(
        (
            config_root
            / "evaluation/ppo_1952_mlp_adapters_kl004_milestones.yaml"
        ).read_text()
    )
    gains_evaluation = yaml.safe_load(
        (config_root / "evaluation/ppo_1952_kl004_gains_milestones.yaml").read_text()
    )
    assert evaluation["training_suite_name"] == replacement["name"]
    assert evaluation["policies"][0]["name"] == "adapters_mlp"
    assert evaluation["policies"][0]["gpu"] == 1
    assert evaluation["evaluation"]["action_selection"] == "mean"
    assert evaluation["evaluation"]["videos"][
        "camera_resolution_reduction_factor"
    ] == 2
    assert gains_evaluation["training_suite_name"] == original["name"]
    assert [policy["name"] for policy in gains_evaluation["policies"]] == [
        "adapters_gains"
    ]
    assert (
        gains_evaluation["output_directory"]
        == "evals/connectome/ppo_1952_kl004_lr01_100b_milestones"
    )


def test_original_kl_mlp_run_differs_only_in_threshold_and_gpu() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config_root = repository_root / "configs/connectome"
    kl004 = yaml.safe_load(
        (config_root / "suites/ppo_1952_mlp_adapters_kl004_100b.yaml").read_text()
    )
    kl016 = yaml.safe_load(
        (config_root / "suites/ppo_1952_mlp_adapters_kl016_100b.yaml").read_text()
    )
    import isaacgymenvs  # noqa: F401 - registers OmegaConf resolvers

    with initialize_config_dir(
        version_base="1.1", config_dir=str(repository_root / "isaacgymenvs/cfg")
    ):
        original_ppo = compose(
            config_name="config",
            overrides=[
                "task=SimToolRealLSTMAsymmetric",
                "train=SimToolRealLSTMAsymmetricPPO",
            ],
        )
    assert original_ppo.train.params.config.kl_threshold == 0.016
    assert kl016["training"]["gpu_assignments"] == [0]
    assert kl004["training"]["gpu_assignments"] == [1]
    assert kl016["training"]["train_profiles"] == kl004["training"][
        "train_profiles"
    ]
    ignored = {"gpu_assignments", "overrides", "wandb"}
    assert {
        key: value
        for key, value in kl016["training"].items()
        if key not in ignored
    } == {
        key: value
        for key, value in kl004["training"].items()
        if key not in ignored
    }
    kl016_overrides = dict(kl016["training"]["overrides"])
    kl004_overrides = dict(kl004["training"]["overrides"])
    assert kl016_overrides.pop("train.params.config.kl_threshold") == 0.016
    assert kl004_overrides.pop("train.params.config.kl_threshold") == 0.004
    assert kl016_overrides == kl004_overrides

    evaluation = yaml.safe_load(
        (
            config_root
            / "evaluation/ppo_1952_mlp_adapters_kl016_milestones.yaml"
        ).read_text()
    )
    assert evaluation["training_suite_name"] == kl016["name"]
    assert evaluation["policies"] == [
        {
            "name": "adapters_mlp",
            "seed": 42,
            "gpu": 0,
            "policy_config_path": (
                "train_dir/connectome/adaptation_100b_gains_update_timing/"
                "ppo_1952_mlp_adapters_kl016_lr01_100b/"
                "00_ppo_1952_mlp_adapters_kl016_lr01_100b_adapters_mlp_seed42/"
                "resolved_config.yaml"
            ),
        }
    ]
    assert evaluation["evaluation"]["action_selection"] == "mean"
    assert (
        evaluation["policies"][0]["name"]
        == kl016["training"]["train_profiles"][0]["name"]
    )
    assert evaluation["evaluation"]["videos"][
        "camera_resolution_reduction_factor"
    ] == 2


def test_tanh_kl016_run_changes_only_policy_distribution_and_identity() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config_root = repository_root / "configs/connectome"
    clipped = yaml.safe_load(
        (config_root / "suites/ppo_1952_mlp_adapters_kl016_100b.yaml").read_text()
    )
    squashed = yaml.safe_load(
        (
            config_root
            / "suites/ppo_1952_mlp_adapters_tanh_kl016_100b.yaml"
        ).read_text()
    )
    smoke = yaml.safe_load(
        (
            config_root
            / "suites/ppo_1952_mlp_adapters_tanh_kl016_smoke.yaml"
        ).read_text()
    )
    matched_keys = set(clipped["training"]) - {
        "train_profiles",
        "wandb",
    }
    for key in matched_keys:
        assert squashed["training"][key] == clipped["training"][key]
    assert squashed["training"]["train_profiles"] == [
        {
            "name": "adapters_mlp_tanh",
            "train_profile": "SimToolRealConnectome1952AdaptersMLPTanhSAPG",
        }
    ]
    for key in (
        "num_envs",
        "sapg_block_size",
        "minibatch_size",
        "central_critic_minibatch_size",
        "rollout_accumulation_steps",
        "actor_microbatch_size",
        "central_critic_microbatch_size",
        "overrides",
    ):
        assert smoke["training"][key] == squashed["training"][key]
    assert smoke["training"]["epochs"] == 2
    assert smoke["training"]["max_frames"] == 393_216

    evaluation = yaml.safe_load(
        (
            config_root
            / "evaluation/ppo_1952_mlp_adapters_tanh_kl016_milestones.yaml"
        ).read_text()
    )
    assert evaluation["training_suite_name"] == squashed["name"]
    assert evaluation["policies"][0]["name"] == "adapters_mlp_tanh"
    assert evaluation["policies"][0]["gpu"] == 0
    assert evaluation["evaluation"]["action_selection"] == "mean"


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


def test_active_gaussian_watchers_generate_fixed_and_checkpoint_tolerance_videos() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    paths = [
        "ppo_1952_4update_gaussian_lf_entropy1x_sigma3_100b_milestones.yaml",
        "ppo_1952_4update_structured_rot6d_gaussian_lf_entropy1x_sigma3_100b_milestones.yaml",
        "ppo_1952_4update_gaussian_lf_entropy1x_sigma3_all_neuron_readout_100b_milestones.yaml",
    ]
    for name in paths:
        config = yaml.safe_load(
            (repository_root / "configs/connectome/evaluation" / name).read_text()
        )
        evaluation = config["evaluation"]
        assert evaluation["action_selection"] == "mean"
        assert evaluation["metrics"]["paper_task_progress"]["success_tolerance_m"] == 0.02
        checkpoint_metric = evaluation["metrics"]["checkpoint_training_tolerance"]
        assert checkpoint_metric == {
            "success_tolerance_source": "checkpoint_tensorboard",
            "success_tolerance_tag": "scalars/success_tolerance/frame",
        }
        assert evaluation["videos"]["metrics"] == [
            "paper_task_progress",
            "checkpoint_training_tolerance",
        ]
        assert len(evaluation["eval_cases"]) == 3


def test_ddp_small_mlp_uses_half_physical_batches_and_five_logical_updates() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    suite = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/suites/"
            "ppo_1952_noaux_mlp128x32x32_ddp_15360_"
            "logical49152_micro24576_100b.yaml"
        ).read_text()
    )
    training = suite["training"]
    assert training["distributed"] is True
    assert training["gpu_assignments"] == [0, 1]
    assert training["num_envs"] == 15_360
    assert training["sapg_block_size"] == 2_560
    assert training["minibatch_size"] == 49_152
    assert training["central_critic_minibatch_size"] == 49_152
    assert training["actor_microbatch_size"] == 24_576
    assert training["central_critic_microbatch_size"] == 24_576
    assert training["rollout_accumulation_steps"] == 1
    assert training["epochs"] == 203_450

    local_fresh_samples = training["num_envs"] * 16
    local_lf_samples = training["sapg_block_size"] * 16
    local_training_samples = local_fresh_samples + local_lf_samples
    assert local_training_samples == 286_720
    assert local_training_samples // training["minibatch_size"] == 5
    assert 2 * local_fresh_samples * training["epochs"] == 99_999_744_000

    evaluation = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/evaluation/"
            "ppo_1952_noaux_mlp128x32x32_ddp_15360_"
            "logical49152_micro24576_100b_milestones.yaml"
        ).read_text()
    )
    assert evaluation["training_suite_name"] == suite["name"]
    assert evaluation["training_suite_directory"] == suite["output_directory"]
    assert evaluation["evaluation"]["videos"]["metrics"] == [
        "paper_task_progress",
        "checkpoint_training_tolerance",
    ]


def test_ddp_small_mlp_original_env_count_keeps_half_physical_batches() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    suite = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/suites/"
            "ppo_1952_noaux_mlp128x32x32_ddp_12288_"
            "logical49152_micro24576_100b.yaml"
        ).read_text()
    )
    training = suite["training"]
    assert training["distributed"] is True
    assert training["gpu_assignments"] == [0, 1]
    assert training["num_envs"] == 12_288
    assert training["sapg_block_size"] == 2_048
    assert training["minibatch_size"] == 49_152
    assert training["central_critic_minibatch_size"] == 49_152
    assert training["actor_microbatch_size"] == 24_576
    assert training["central_critic_microbatch_size"] == 24_576
    assert training["rollout_accumulation_steps"] == 1
    assert training["epochs"] == 254_313

    local_fresh_samples = training["num_envs"] * 16
    local_lf_samples = training["sapg_block_size"] * 16
    local_training_samples = local_fresh_samples + local_lf_samples
    assert local_training_samples == 229_376
    assert local_training_samples // training["minibatch_size"] == 4
    assert 2 * local_fresh_samples * training["epochs"] == 99_999_940_608

    evaluation = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/evaluation/"
            "ppo_1952_noaux_mlp128x32x32_ddp_12288_"
            "logical49152_micro24576_100b_milestones.yaml"
        ).read_text()
    )
    assert evaluation["training_suite_name"] == suite["name"]
    assert evaluation["training_suite_directory"] == suite["output_directory"]
    assert evaluation["evaluation"]["videos"]["metrics"] == [
        "paper_task_progress",
        "checkpoint_training_tolerance",
    ]


def test_noaux_7b_dual_tolerance_anatomical_hd_contract_is_matched() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/evaluation/"
            "ppo_1952_7b_noaux_dual_tolerance_anatomical_hd.yaml"
        ).read_text()
    )
    assert list(config["policy_sources"]) == [
        "gaussian_lf_entropy1x_sigma3_all_neurons_noaux"
    ]
    assert config["action_selection"] == "mean"
    assert config["metrics"]["paper_task_progress"] == {
        "success_tolerance_m": 0.02
    }
    assert config["metrics"]["checkpoint_training_tolerance"] == {
        "success_tolerance_m": 0.039858076721429825,
        "resolved_from_checkpoint_frame": 7000031232,
        "resolved_from_tensorboard_tag": "scalars/success_tolerance/frame",
    }
    assert config["videos"]["metrics"] == [
        "paper_task_progress",
        "checkpoint_training_tolerance",
    ]
    assert config["videos"]["camera_resolution_reduction_factor"] == 1
    assert config["videos"]["video_quality"] == 9
    assert config["videos"]["circuit"]["resolution"] == [1920, 1080]
    assert len(config["eval_cases"]) == 3


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
