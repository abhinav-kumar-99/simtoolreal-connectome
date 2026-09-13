from __future__ import annotations

import copy

import torch
from torch import nn

from rl_games.algos_torch.central_value import CentralValueTrain


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
