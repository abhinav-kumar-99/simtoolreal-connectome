from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


def test_training_state_resume_keeps_optimizer_and_skips_environment(monkeypatch) -> None:
    from rl_games import torch_runner

    checkpoint = {
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {3: {"step": torch.tensor(9.0)}}},
        "epoch": 17,
        "frame": 1234,
        "env_state": {"unsafe": torch.tensor([1.0])},
    }

    class Agent:
        global_rank = 0

        def set_full_state_weights(self, state, **kwargs):
            self.state = state
            self.kwargs = kwargs

    monkeypatch.setattr(
        torch_runner.torch_ext,
        "load_checkpoint",
        lambda _: {0: checkpoint},
    )
    agent = Agent()
    torch_runner._restore(
        agent,
        {
            "checkpoint": "checkpoint.pth",
            "checkpoint_load_mode": "resume_training_state",
        },
    )
    assert agent.state is checkpoint
    assert agent.state["optimizer"]["state"]
    assert agent.kwargs == {"set_epoch": True, "restore_environment": False}


def test_reward_window_requires_exact_target_and_averages_tail(tmp_path: Path) -> None:
    from scripts.run_connectome_reward_handoff import _reward_window

    writer = SummaryWriter(str(tmp_path))
    for index in range(1, 6):
        writer.add_scalar("rewards/step", float(index), index * 10)
    writer.close()

    result = _reward_window(tmp_path, "rewards/step", 50, 3)
    assert result["first_step"] == 30
    assert result["last_step"] == 50
    assert result["mean"] == pytest.approx(4.0)
    with pytest.raises(RuntimeError, match="has not reached"):
        _reward_window(tmp_path, "rewards/step", 60, 3)


def test_1952_reward_handoff_contract_routes_original_tensorboard_runs() -> None:
    from scripts.run_connectome_suite import _training_overrides

    repository_root = Path(__file__).resolve().parents[2]
    handoff = yaml.safe_load(
        (
            repository_root
            / "configs/connectome/handoffs/ppo_1952_tanh_vs_nontanh.yaml"
        ).read_text()
    )
    catch_up_path = repository_root / handoff["catch_up_suite"]
    catch_up = yaml.safe_load(catch_up_path.read_text())
    training = catch_up["training"]
    assert training["checkpoint"]["mode"] == "resume_training_state"
    assert training["epochs"] == 12_098
    assert training["max_frames"] == 2_378_563_584
    assert handoff["comparison"] == {
        **handoff["comparison"],
        "tag": "rewards/step",
        "target_step": 2_378_366_976,
        "window_points": 100,
        "candidate": "tanh",
        "baseline": "non_tanh",
    }

    target = training["artifact_target"]
    overrides = _training_overrides(
        training,
        training["train_profiles"][0]["train_profile"],
        42,
        "bookkeeping_name",
        repository_root / "unused-bookkeeping-directory",
    )
    expected_train_dir = (repository_root / target["train_directory"]).resolve()
    assert f"++train.params.config.train_dir={expected_train_dir}" in overrides
    assert f"experiment={target['experiment_name']}" in overrides

    for continuation_path in handoff["continuation_suites"].values():
        continuation = yaml.safe_load((repository_root / continuation_path).read_text())
        assert continuation["training"]["checkpoint"]["mode"] == "resume_training_state"
        assert continuation["training"]["max_frames"] == 100_000_000_000
        assert continuation["training"]["gpu_assignments"] == [0]
