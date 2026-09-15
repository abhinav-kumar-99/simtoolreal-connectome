from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from scripts.run_connectome_numerical_fallback import checkpoint_health, scan_event_file


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
