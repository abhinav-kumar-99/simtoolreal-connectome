from __future__ import annotations

import copy

import torch
from torch import nn

from rl_games.algos_torch.central_value import CentralValueTrain
from rl_games.common.a2c_common import A2CBase
from rl_games.common.datasets import PPODataset


class _TinyValueModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"values": self.linear(batch["obs"])}


class _Tracker:
    def __init__(self) -> None:
        self.mean = torch.zeros(1)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.clone()}

    def load_state_dict(self, state, strict=False) -> None:
        del strict
        self.mean.copy_(state["mean"])


class _CheckpointHarness:
    def __init__(self) -> None:
        self.model = nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.central_value_net = _checkpointable_critic(_TinyValueModel())
        self.has_central_value = True
        self.epoch_num = 9
        self.frame = 1_769_472
        self.last_lr = 3e-4
        self.last_mean_rewards = 12.0
        self.game_rewards = _Tracker()
        self.game_shaped_rewards = _Tracker()
        self.game_lengths = _Tracker()
        self.vec_env = None
        self.intr_reward_model = None
        self.num_actors = 2
        self.rnn_states = None
        self.dones = torch.zeros(2)
        self.obs = torch.zeros(2, 2)
        self.current_rewards = torch.zeros(2, 1)
        self.current_shaped_rewards = torch.zeros(2, 1)
        self.current_lengths = torch.zeros(2)

    def get_weights(self):
        return {"model": self.model.state_dict()}

    def set_weights(self, weights) -> None:
        self.model.load_state_dict(weights["model"])


def _critic(model: nn.Module) -> CentralValueTrain:
    critic = CentralValueTrain.__new__(CentralValueTrain)
    nn.Module.__init__(critic)
    critic.model = model
    critic.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    critic.e_clip = 0.2
    critic.clip_value = False
    critic.is_rnn = False
    critic.seq_length = 1
    critic.multi_gpu = False
    critic.truncate_grads = False
    return critic


def _checkpointable_critic(model: nn.Module) -> CentralValueTrain:
    critic = _critic(model)
    critic.epoch_num = 0
    critic.frame = 0
    critic.lr = 1e-3
    critic.rnn_states = None
    critic.ppo_device = "cpu"
    return critic


def _batch(obs: torch.Tensor, returns: torch.Tensor) -> dict[str, torch.Tensor]:
    count = len(obs)
    return {
        "obs": obs,
        "old_values": torch.zeros(count, 1),
        "returns": returns,
        "actions": torch.zeros(count, 1),
        "dones": torch.zeros(count),
    }


def test_central_critic_microbatches_equal_one_logical_optimizer_step() -> None:
    torch.manual_seed(7)
    full_model = _TinyValueModel()
    accumulated_model = copy.deepcopy(full_model)
    full = _critic(full_model)
    accumulated = _critic(accumulated_model)
    obs = torch.tensor([[1.0, 2.0], [3.0, -1.0], [-2.0, 4.0], [0.5, 1.5]])
    returns = torch.tensor([[0.2], [-0.3], [1.1], [0.7]])

    full.calc_gradients(_batch(obs, returns))
    accumulated.calc_gradients(
        _batch(obs[:2], returns[:2]),
        optimizer_step=False,
        loss_scale=0.5,
    )
    accumulated.calc_gradients(
        _batch(obs[2:], returns[2:]),
        zero_grad=False,
        loss_scale=0.5,
    )

    assert torch.allclose(
        full_model.linear.weight,
        accumulated_model.linear.weight,
        rtol=1e-6,
        atol=1e-7,
    )
    full_step = next(iter(full.optimizer.state.values()))["step"]
    accumulated_step = next(iter(accumulated.optimizer.state.values()))["step"]
    assert int(full_step.item()) == int(accumulated_step.item()) == 1


def test_central_critic_training_state_round_trip() -> None:
    source = _checkpointable_critic(_TinyValueModel())
    source.epoch_num = 37
    source.frame = 7_274_496
    source.lr = 2.5e-5
    source.optimizer.param_groups[0]["lr"] = source.lr
    source.rnn_states = [torch.tensor([[1.0, -2.0]])]

    saved = source.get_training_state()
    restored = _checkpointable_critic(_TinyValueModel())
    restored.set_training_state(saved)

    assert restored.epoch_num == 37
    assert restored.frame == 7_274_496
    assert restored.lr == 2.5e-5
    assert restored.optimizer.param_groups[0]["lr"] == 2.5e-5
    torch.testing.assert_close(restored.rnn_states[0], source.rnn_states[0])


def test_central_critic_fresh_rollout_does_not_restore_rnn_state() -> None:
    restored = _checkpointable_critic(_TinyValueModel())
    restored.rnn_states = [torch.zeros(1, 2)]
    restored.set_training_state(
        {
            "epoch": 4,
            "frame": 128,
            "lr": 4e-4,
            "rnn_states": [torch.ones(1, 2)],
        },
        restore_recurrent_state=False,
    )

    assert restored.epoch_num == 4
    assert restored.frame == 128
    assert restored.lr == 4e-4
    torch.testing.assert_close(restored.rnn_states[0], torch.zeros(1, 2))


def test_full_checkpoint_round_trip_restores_critic_training_metadata() -> None:
    source = _CheckpointHarness()
    source.central_value_net.epoch_num = 9
    source.central_value_net.frame = 1_769_472
    source.central_value_net.lr = 7e-5
    source.central_value_net.optimizer.param_groups[0]["lr"] = 7e-5

    saved = A2CBase.get_full_state_weights(source)
    assert saved["central_value_training_state"]["epoch"] == 9
    assert saved["central_value_training_state"]["frame"] == 1_769_472
    assert saved["central_value_training_state"]["lr"] == 7e-5

    restored = _CheckpointHarness()
    A2CBase.set_full_state_weights(
        restored,
        saved,
        restore_environment=False,
    )

    assert restored.central_value_net.epoch_num == 9
    assert restored.central_value_net.frame == 1_769_472
    assert restored.central_value_net.lr == 7e-5
    assert restored.central_value_net.optimizer.param_groups[0]["lr"] == 7e-5


def test_legacy_full_checkpoint_infers_critic_training_metadata() -> None:
    source = _CheckpointHarness()
    source.central_value_net.optimizer.param_groups[0]["lr"] = 8e-5
    saved = A2CBase.get_full_state_weights(source)
    del saved["central_value_training_state"]

    restored = _CheckpointHarness()
    A2CBase.set_full_state_weights(
        restored,
        saved,
        restore_environment=False,
    )

    assert restored.central_value_net.epoch_num == source.epoch_num
    assert restored.central_value_net.frame == source.frame
    assert restored.central_value_net.lr == 8e-5


def test_microbatch_slices_preserve_enlarged_final_logical_batch() -> None:
    dataset = PPODataset(
        batch_size=32,
        minibatch_size=4,
        is_discrete=False,
        is_rnn=False,
        device="cpu",
        seq_length=2,
        logical_minibatch_size=8,
    )
    returns = torch.arange(38).reshape(38, 1)
    dataset.update_values_dict({"returns": returns})

    reconstructed = []
    logical_scales = []
    current_scales = []
    optimizer_steps = 0
    for index in range(len(dataset)):
        item = dataset[index]
        if dataset.last_zero_grad:
            assert current_scales == []
        reconstructed.append(item["returns"])
        current_scales.append(dataset.last_loss_scale)
        if dataset.last_optimizer_step:
            optimizer_steps += 1
            logical_scales.append(sum(current_scales))
            current_scales = []

    assert optimizer_steps == 4
    assert current_scales == []
    assert all(abs(scale - 1.0) < 1e-12 for scale in logical_scales)
    assert torch.equal(torch.cat(reconstructed), returns)
    assert [entry["end"] - entry["start"] for entry in dataset.slices[-4:]] == [
        4,
        4,
        4,
        2,
    ]
