from __future__ import annotations

import copy

import torch
from torch import nn

from rl_games.algos_torch.central_value import CentralValueTrain
from rl_games.common.datasets import PPODataset


class _TinyValueModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"values": self.linear(batch["obs"])}


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
