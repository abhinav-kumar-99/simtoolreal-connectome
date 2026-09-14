from copy import deepcopy

import pytest
import torch

from .test_connectome_network import artifact_path, _build, _observations
from rl_games.algos_torch.connectome_eligibility import LocalEligibility


def local_network(path):
    return _build(path, adaptation={"weight_mode": "neuron_gains", "learn_dynamics": False})


def settings():
    return dict(gamma=.99, trace_lambda=.95, adapter_lr=1e-3, readout_lr=1e-3,
                gain_lr=1e-3, feedback_seed=7, feedback_scale=.1,
                trace_clip=1e6, td_clip=100., max_grad_norm=1e6)


def test_one_step_local_derivatives_and_shared_outgoing(artifact_path):
    net = local_network(artifact_path)
    reference = deepcopy(net)
    engine = LocalEligibility(net, 2, settings())
    obs, previous, score = _observations(2), torch.randn(2, 7), torch.randn(2, 2)
    hidden = net._step(obs, previous)
    engine.observe(obs, previous, hidden, score)
    signal = score @ engine.feedback.T
    signal[:, net.motor_indices] = score @ net.mu.weight
    ref_hidden = reference._step(obs, previous)
    # Fix the spatial learning signal: tests the local derivative exactly,
    # not a claim that random feedback is the true full actor gradient.
    (ref_hidden * signal.detach()).sum().backward()
    mapping = {"sensory": reference.sensory_adapter.weight,
               "descending": reference.descending_adapter.weight,
               "incoming": reference.incoming_gain_raw,
               "outgoing": reference.outgoing_gain_raw}
    for name, parameter in mapping.items():
        torch.testing.assert_close(engine.credit[name].sum(0), parameter.grad, atol=2e-6, rtol=2e-5)
    reference.zero_grad()
    (reference.mu(ref_hidden.detach()[:, reference.motor_indices]) * score).sum().backward()
    torch.testing.assert_close(engine.credit["readout"].sum(0), reference.mu.weight.grad)
    torch.testing.assert_close(engine.credit["readout_bias"].sum(0), reference.mu.bias.grad)


def test_reward_trace_products_reset_and_frozen_graph(artifact_path):
    net = local_network(artifact_path)
    engine = LocalEligibility(net, 2, settings())
    before = {name: p.clone() for name, p in engine.params.items()}
    edges = net.recurrent_values.clone()
    engine.credit["readout_bias"][0].fill_(1)
    engine.credit["readout_bias"][1].fill_(-1)
    engine.update(torch.zeros(2))
    for name, p in engine.params.items():
        torch.testing.assert_close(p, before[name])
    # Mean(delta) and mean(trace) are both zero, but mean(delta*trace)=1.
    engine.update(torch.tensor([1., -1.]))
    torch.testing.assert_close(net.mu.bias, before["readout_bias"] + .001)
    engine.reset(torch.tensor([0]))
    assert engine.credit["readout_bias"][0].count_nonzero() == 0
    assert engine.credit["readout_bias"][1].count_nonzero() == 2
    torch.testing.assert_close(edges, net.recurrent_values)
    assert all(p.grad is None and not p.requires_grad for p in net.parameters())
    assert all(t.grad_fn is None for t in engine.local.values())


def test_decay_and_resume_contract(artifact_path):
    net = local_network(artifact_path)
    engine = LocalEligibility(net, 2, settings())
    engine.credit["readout_bias"].fill_(1)
    engine.observe(_observations(2), torch.zeros(2, 7), torch.zeros(2, 7), torch.zeros(2, 2))
    torch.testing.assert_close(engine.credit["readout_bias"], torch.full((2, 2), .99 * .95))
    saved = deepcopy(engine.state_dict())
    engine.load_state_dict(saved)
    assert all(t.count_nonzero() == 0 for t in engine.credit.values())
    saved["config"]["gamma"] = .9
    with pytest.raises(ValueError, match="mismatch"):
        engine.load_state_dict(saved)
    with pytest.raises(FloatingPointError):
        engine.update(torch.tensor([float("nan"), 0.]))


def test_local_temporal_sensitivity_with_autapse(artifact_path):
    net = local_network(artifact_path)
    # Change an existing edge to an autapse in this test-only graph.
    net.col_indices[0] = 2
    engine = LocalEligibility(net, 2, settings())
    previous = torch.randn(2, 7)
    expected = torch.zeros(2, 7)
    for _ in range(4):
        obs = _observations(2)
        hidden = net._step(obs, previous)
        derivative = net.leaks() * (1 - ((hidden - (1 - net.leaks()) * previous) / net.leaks()).square())
        recurrence = torch.sparse.mm(net._native_csr_matrix(), (previous * net.outgoing_gains()).T).T
        raw = net.incoming_gain_raw.sigmoid()
        gain_derivative = net.incoming_gains() * net.log_gain_span * raw * (1 - raw)
        own_jacobian = 1 - net.leaks() + derivative * net.recurrent_gain * net.incoming_gains() * engine.diag * net.outgoing_gains()
        expected = own_jacobian * expected + derivative * net.recurrent_gain * recurrence * gain_derivative
        engine.observe(obs, previous, hidden, torch.randn(2, 2))
        torch.testing.assert_close(engine.local["incoming"], expected)
        previous = hidden


class TinyEnv:
    def __init__(self):
        self.progress = torch.zeros(6, dtype=torch.long)
        self.obs = torch.zeros(6, 5)

    def get_env_info(self):
        from gym.spaces import Box
        return {"observation_space": Box(-10., 10., (5,)), "action_space": Box(-1., 1., (2,))}

    def reset(self):
        self.progress.zero_()
        return {"obs": self.obs}

    def reset_done(self):
        ids = (self.progress == 3).nonzero().flatten()
        self.progress[ids] = 0
        self.obs[ids] = 0
        return {"obs": self.obs}, ids

    def step(self, action):
        self.progress += 1
        self.obs[:, :2] = action
        self.obs[:, 2:] = self.progress[:, None].float() / 3
        return {"obs": self.obs}, action[:, 0] + .5, self.progress == 3, {"time_outs": self.progress == 3}

    def set_train_info(self, *args):
        pass

    def get_env_state(self):
        return None


def agent_params(path, tmp_path):
    from .test_connectome_network import _network_params
    from rl_games.common.algo_observer import AlgoObserver
    network = _network_params(path)
    network["connectome"].pop("plasticity_mode")
    network["connectome"]["adaptation"] = {"weight_mode": "neuron_gains", "learn_dynamics": False}
    e = settings()
    e.update(bootstrap_timeouts=False, sigma=.3, critic_width=8, critic_lr=.001)
    return {"model": {"name": "continuous_a2c_logstd"}, "network": network,
            "config": {"eligibility": e, "device": "cpu", "num_actors": 6, "vec_env": TinyEnv(),
                       "normalize_input": True, "features": {"observer": AlgoObserver()},
                       "train_dir": str(tmp_path), "name": "local", "max_epochs": 2,
                       "horizon_length": 3, "max_frames": 36, "save_frequency": 1,
                       "reward_shaper": lambda x: x, "inference_checkpoint_interval_frames": 18}}


def test_agent_train_checkpoint_resume_and_tensorboard(artifact_path, tmp_path):
    from rl_games.algos_torch.connectome_eligibility_agent import ConnectomeEligibilityAgent
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    params = agent_params(artifact_path, tmp_path)
    agent = ConnectomeEligibilityAgent("run", params)
    initial = {name: parameter.clone() for name, parameter in agent.engine.params.items()}
    assert agent.train() == 6
    assert agent.frame == 36 and agent.engine.updates == 6
    assert all(not torch.equal(parameter, initial[name]) for name, parameter in agent.engine.params.items())
    assert all(t.count_nonzero() == 0 for t in agent.engine.credit.values())
    events = EventAccumulator(str(tmp_path / "local/summaries")).Reload()
    assert [e.step for e in events.Scalars("eligibility/updates")] == [18, 36]
    assert "rewards/step" in events.Tags()["scalars"]
    params["config"].update(name="resumed", max_epochs=3, max_frames=54, vec_env=TinyEnv())
    resumed = ConnectomeEligibilityAgent("run", params)
    resumed.restore(str(tmp_path / "local/nn/last.pth"))
    assert resumed.frame == 36 and resumed.engine.updates == 6
    with torch.no_grad():
        probe = torch.zeros(6, 5)
        original_action = agent.net({"obs": agent.normalized(probe)})[0]
        restored_action = resumed.net({"obs": resumed.normalized(probe)})[0]
        torch.testing.assert_close(original_action, restored_action, atol=0, rtol=0)
        torch.testing.assert_close(agent.engine.feedback, resumed.engine.feedback, atol=0, rtol=0)
    assert all(t.count_nonzero() == 0 for t in resumed.engine.credit.values())
    resumed.train()
    assert resumed.frame == 54 and resumed.engine.updates == 9
    assert resumed.critic_optimizer.state


@pytest.mark.parametrize("terminal", [False, True])
def test_td_target_terminal_and_timeout_semantics(artifact_path, tmp_path, terminal):
    from rl_games.algos_torch.connectome_eligibility_agent import ConnectomeEligibilityAgent

    class TargetEnv(TinyEnv):
        def step(self, action):
            obs, reward, done, info = super().step(action)
            self.last_reward = reward.clone()
            done.fill_(terminal)
            return obs, reward, done, {"time_outs": done}

    params = agent_params(artifact_path, tmp_path)
    env = TargetEnv()
    params["config"].update(vec_env=env, horizon_length=1, max_epochs=1, max_frames=6)
    agent = ConnectomeEligibilityAgent("run", params)
    with torch.no_grad():
        for p in agent.critic.parameters():
            p.zero_()
        agent.critic[-1].bias.fill_(2.)
    captured = []
    original_update = agent.engine.update

    def capture(delta):
        captured.append(delta.clone())
        return original_update(delta)

    agent.engine.update = capture
    agent.train()
    expected = env.last_reward - 2. + (0. if terminal else .99 * 2.)
    torch.testing.assert_close(captured[0], expected)


def test_suite_profile_and_checkpoint_validation(tmp_path):
    import yaml
    from pathlib import Path
    from scripts.run_connectome_suite import _compose_resolved, _training_overrides, _verify_checkpoint
    root = Path(__file__).resolve().parents[2]
    for name in ("eligibility_smoke", "eligibility_resume_smoke", "eligibility_1952", "eligibility_1952_100b"):
        suite = yaml.safe_load((root / f"configs/connectome/suites/{name}.yaml").read_text())
        training = suite["training"]
        overrides = _training_overrides(training, training["train_profiles"][0]["train_profile"], 42, "test", tmp_path)
        assert not any("minibatch_size=" in arg for arg in overrides)
        resolved = _compose_resolved(overrides)
        assert resolved.train.params.algo.name == "connectome_eligibility"
        assert resolved.train.params.config.central_value_config is None
        assert not resolved.train.params.config.ppo
        assert resolved.train.params.config.seq_length == 1
        if name == "eligibility_1952_100b":
            assert resolved.train.params.config.num_actors == 384
            assert resolved.train.params.config.horizon_length == 4096
            assert resolved.train.params.config.max_epochs == 63579
            assert resolved.train.params.config.max_frames == 100001120256
            assert resolved.train.params.config.inference_checkpoint_interval_frames == 250000000
    from omegaconf import OmegaConf
    config = tmp_path / "config.yaml"
    OmegaConf.save(resolved, config)
    checkpoint = tmp_path / "inference.pth"
    torch.save({"model": {}, "epoch": 160, "frame": 983040}, checkpoint)
    with pytest.raises(RuntimeError, match="eligibility/critic updates"):
        _verify_checkpoint(checkpoint, config, "cpu")
