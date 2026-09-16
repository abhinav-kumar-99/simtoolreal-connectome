from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from scripts.run_connectome_numerical_fallback import checkpoint_health, scan_event_file
from scripts.run_connectome_success_handoff import (
    _update_window,
    configure_comparison_state,
    summarize_window,
)


def test_event_scanner_detects_only_new_nonfinite_scalars(tmp_path: Path) -> None:
    writer = SummaryWriter(str(tmp_path))
    writer.add_scalar('losses/entropy', 12.5, 10)
    writer.flush()
    event_path = next(tmp_path.glob('events.out.tfevents.*'))
    offset, latest, nonfinite = scan_event_file(event_path, 0)
    assert not nonfinite
    assert latest['losses/entropy'] == {'step': 10, 'value': 12.5}

    writer.add_scalar('losses/entropy', float('nan'), 20)
    writer.add_scalar('auxiliary_stats/off_on_grad_similarity', float('nan'), 20)
    writer.flush()
    next_offset, latest, nonfinite = scan_event_file(
        event_path, offset, ['losses/*', 'info/kl']
    )
    writer.close()
    assert next_offset > offset
    assert not latest
    assert nonfinite == [{
        'file': str(event_path), 'step': 20, 'tag': 'losses/entropy', 'value': 'nan'
    }]


def test_checkpoint_health_finds_nonfinite_trainable_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / 'model.pth'
    torch.save({0: {
        'epoch': 3, 'frame': 99,
        'model': {'ok': torch.ones(2), 'bad': torch.tensor([float('inf')])},
        'optimizer': {'state': {}},
    }}, checkpoint)
    health = checkpoint_health(checkpoint)
    assert health['epoch'] == 3 and health['frame'] == 99
    assert health['nonfinite_tensors'] == ['model.bad']


def test_unrestricted_beta_fallback_contract_composes() -> None:
    from scripts.run_connectome_suite import _compose_resolved, _training_overrides

    root = Path(__file__).resolve().parents[2]
    suite_path = root / 'configs/connectome/suites/ppo_1952_4update_unrestricted_beta_100b.yaml'
    suite = yaml.safe_load(suite_path.read_text())
    training = suite['training']
    profile = training['train_profiles'][0]
    resolved = _compose_resolved(_training_overrides(
        training, profile['train_profile'], 42, profile['name'],
        root / suite['output_directory'],
    ))
    continuous = resolved.train.params.network.space.continuous
    config = resolved.train.params.config
    assert resolved.train.params.model.name == 'continuous_a2c_beta'
    assert continuous.distribution == 'beta'
    assert continuous.beta_min_shape == 0.0001
    assert continuous.beta_initial_shape == 2.0
    assert resolved.train.params.network.connectome.dynamics.neural_updates == 4
    assert config.kl_threshold == 0.004 and config.max_lr == 0.001
    assert config.use_experimental_cv is True
    assert config.use_others_experience == 'lf'
    assert config.off_policy_ratio == 1.0
    assert training['gpu_assignments'] == [1]
    assert training['max_frames'] == 100_000_000_000

    monitor = yaml.safe_load((
        root / 'configs/connectome/handoffs/ppo_1952_gaussian_to_unrestricted_beta.yaml'
    ).read_text())
    assert monitor['source']['expected_final_frame'] == 99_999_940_608
    assert monitor['fallback']['training_config'].endswith(suite_path.name)
    video = yaml.safe_load((root / monitor['fallback']['video_config']).read_text())
    assert video['policies'][0]['name'] == 'unrestricted_beta'
    assert video['policies'][0]['gpu'] == 1
    assert video['milestone_interval_frames'] == 250_000_000


def test_restricted_beta_lf_replacement_contract_composes() -> None:
    from scripts.run_connectome_suite import _compose_resolved, _training_overrides

    root = Path(__file__).resolve().parents[2]
    suite_path = (
        root
        / 'configs/connectome/suites/ppo_1952_4update_restricted_beta_lf_100b.yaml'
    )
    suite = yaml.safe_load(suite_path.read_text())
    training = suite['training']
    profile = training['train_profiles'][0]
    resolved = _compose_resolved(_training_overrides(
        training, profile['train_profile'], 42, profile['name'],
        root / suite['output_directory'],
    ))
    continuous = resolved.train.params.network.space.continuous
    config = resolved.train.params.config
    assert resolved.train.params.model.name == 'continuous_a2c_beta'
    assert continuous.distribution == 'beta'
    assert continuous.beta_min_shape == 1.0
    assert continuous.beta_initial_shape == 2.0
    assert resolved.train.params.network.connectome.dynamics.neural_updates == 4
    assert config.kl_threshold == 0.004 and config.max_lr == 0.001
    assert config.use_experimental_cv is True
    assert config.use_others_experience == 'lf'
    assert config.off_policy_ratio == 1.0
    assert training['gpu_assignments'] == [1]
    assert training['max_frames'] == 100_000_000_000

    video_path = (
        root
        / 'configs/connectome/evaluation/'
        'ppo_1952_4update_restricted_beta_lf_100b_milestones.yaml'
    )
    video = yaml.safe_load(video_path.read_text())
    assert video['training_suite_name'] == suite['name']
    assert video['policies'][0]['name'] == 'restricted_beta_lf'
    assert video['policies'][0]['gpu'] == 1
    assert video['milestone_interval_frames'] == 250_000_000


def test_restricted_beta_lf_entropy5x_contract_composes() -> None:
    from scripts.run_connectome_suite import _compose_resolved, _training_overrides

    root = Path(__file__).resolve().parents[2]
    suite_path = (
        root
        / 'configs/connectome/suites/'
        'ppo_1952_4update_restricted_beta_lf_entropy5x_100b.yaml'
    )
    suite = yaml.safe_load(suite_path.read_text())
    training = suite['training']
    profile = training['train_profiles'][0]
    resolved = _compose_resolved(_training_overrides(
        training, profile['train_profile'], 42, profile['name'],
        root / suite['output_directory'],
    ))
    continuous = resolved.train.params.network.space.continuous
    config = resolved.train.params.config
    assert resolved.train.params.model.name == 'continuous_a2c_beta'
    assert continuous.distribution == 'beta'
    assert continuous.beta_min_shape == 1.0
    assert continuous.beta_initial_shape == 2.0
    assert resolved.train.params.network.connectome.dynamics.neural_updates == 4
    assert config.kl_threshold == 0.004 and config.max_lr == 0.001
    assert config.use_experimental_cv is True
    assert config.use_others_experience == 'lf'
    assert config.off_policy_ratio == 1.0
    assert config.expl_reward_coef_scale == 0.025
    assert training['gpu_assignments'] == [0]
    assert training['max_frames'] == 100_000_000_000

    video_path = (
        root
        / 'configs/connectome/evaluation/'
        'ppo_1952_4update_restricted_beta_lf_entropy5x_100b_milestones.yaml'
    )
    video = yaml.safe_load(video_path.read_text())
    assert video['training_suite_name'] == suite['name']
    assert video['policies'][0]['name'] == 'restricted_beta_lf_entropy5x'
    assert video['policies'][0]['gpu'] == 0
    assert video['milestone_interval_frames'] == 250_000_000


def test_success_handoff_candidate_contracts_compose() -> None:
    from scripts.run_connectome_suite import _compose_resolved, _training_overrides

    root = Path(__file__).resolve().parents[2]
    cases = [
        (
            'ppo_1952_4update_restricted_beta_lf_entropy3x_100b.yaml',
            'continuous_a2c_beta', 'beta', None,
        ),
        (
            'ppo_1952_4update_gaussian_lf_entropy3x_sigma3_100b.yaml',
            'continuous_a2c_logstd', 'gaussian', 3.0,
        ),
    ]
    for filename, model_name, distribution, max_sigma in cases:
        suite_path = root / 'configs/connectome/suites' / filename
        suite = yaml.safe_load(suite_path.read_text())
        training = suite['training']
        profile = training['train_profiles'][0]
        resolved = _compose_resolved(_training_overrides(
            training, profile['train_profile'], 42, profile['name'],
            root / suite['output_directory'],
        ))
        continuous = resolved.train.params.network.space.continuous
        config = resolved.train.params.config
        assert resolved.train.params.model.name == model_name
        assert continuous.distribution == distribution
        assert getattr(continuous, 'max_sigma', None) == max_sigma
        assert resolved.train.params.network.connectome.dynamics.neural_updates == 4
        assert config.kl_threshold == 0.004 and config.max_lr == 0.001
        assert config.use_experimental_cv is True
        assert config.use_others_experience == 'lf'
        assert config.off_policy_ratio == 1.0
        assert config.expl_reward_coef_scale == 0.015
        assert training['gpu_assignments'] == [0]
        assert training['max_frames'] == 100_000_000_000

    handoff = yaml.safe_load((
        root / 'configs/connectome/handoffs/ppo_1952_lf_entropy_success_handoff.yaml'
    ).read_text())
    assert handoff['comparison'] == {
        'tag': 'true_objective_mean/frame',
        'target_step': 1_250_000_000,
        'aggregation': 'arithmetic_mean',
        'window_steps': 100_000_000,
        'comparator': 'strict_less',
    }
    assert [stage['name'] for stage in handoff['stages']] == [
        'restricted_beta_lf_entropy5x', 'restricted_beta_lf_entropy3x'
    ]
    assert handoff['stages'][0]['replacement']['stage'] == 'restricted_beta_lf_entropy3x'
    assert handoff['stages'][1]['replacement']['stage'] == 'gaussian_lf_entropy3x_sigma3'
    assert handoff['stages'][1]['replacement']['terminal'] is True


def test_success_handoff_window_mean_waits_for_crossing() -> None:
    target, window = 1_250_000_000, 100_000_000
    state = {}
    _update_window(state, [
        {'step': target - window - 1, 'value': 100.0},
        {'step': target - window, 'value': 1.0},
        {'step': target - 50_000_000, 'value': 2.0},
        {'step': target - 1, 'value': 3.0},
    ], target, window)
    assert summarize_window(state, target, window) is None
    _update_window(state, [{'step': target + 10, 'value': 200.0}], target, window)
    summary = summarize_window(state, target, window)
    assert summary == {
        'window_start_step': 1_150_000_000,
        'window_end_step': 1_250_000_000,
        'first_logged_step': 1_150_000_000,
        'last_logged_step': 1_249_999_999,
        'sample_count': 3,
        'mean': 2.0,
        'crossing_step': 1_250_000_010,
    }


def test_success_handoff_contract_change_clears_old_metric_state() -> None:
    state = {
        'status': 'monitoring',
        'comparison_contract': {
            'tag': 'mean_successes/frame',
            'target_step': 1_250_000_000,
            'aggregation': 'arithmetic_mean',
            'window_steps': 100_000_000,
            'comparator': 'strict_less',
        },
        'metrics': {'candidate': {
            'before': {'step': 999_948_288, 'value': 0.1},
            'after': {'step': 1_000_144_896, 'value': 0.2},
            'selected': {'step': 999_948_288, 'value': 0.1},
            'event_offsets': {'events': 123},
        }},
        'transitions': [],
    }
    configure_comparison_state(state, {
        'tag': 'true_objective_mean/frame',
        'target_step': 1_250_000_000,
        'aggregation': 'arithmetic_mean',
        'window_steps': 100_000_000,
        'comparator': 'strict_less',
    })
    assert state['comparison_target_step'] == 1_250_000_000
    assert state['metrics'] == {}
    history = state['comparison_change_history'][-1]
    assert history['previous_contract']['tag'] == 'mean_successes/frame'
    assert history['new_contract']['tag'] == 'true_objective_mean/frame'
    assert history['prior_points']['candidate']['selected']['step'] == 999_948_288
