import math

import pytest
import torch

from rl_games.algos_torch import model_builder
from rl_games.algos_torch.connectome_network_builder import ConnectomeBuilder
from rl_games.algos_torch.torch_ext import beta_policy_kl
from .test_connectome_network import artifact_path, _network_params, _observations


def network(artifact_path, updates=4, distribution='gaussian', backend='native_csr', beta_min_shape=1.0):
    params = _network_params(artifact_path, 'frozen_core', backend)
    params['connectome']['dynamics']['neural_updates'] = updates
    params['space']['continuous']['distribution'] = distribution
    params['space']['continuous']['beta_min_shape'] = beta_min_shape
    return ConnectomeBuilder.Network(params, actions_num=2, input_shape=(6,), num_seqs=2,
                                     type='extra_param', coef_ids=torch.tensor([50., 0.]), coef_id_idx=5)


def policy(artifact_path):
    params = _network_params(artifact_path, 'frozen_core')
    params['connectome']['dynamics']['neural_updates'] = 4
    params['space']['continuous']['distribution'] = 'beta'
    return model_builder.ModelBuilder().load({'model': {'name': 'continuous_a2c_beta'}, 'network': params}).build(
        {'actions_num': 2, 'input_shape': (6,), 'num_seqs': 2, 'value_size': 1,
         'normalize_input': False, 'normalize_value': False, 'type': 'extra_param',
         'coef_ids': torch.tensor([50., 0.]), 'coef_id_idx': 5})


@pytest.mark.parametrize('updates', [1, 4, 8])
def test_passive_retention_and_manual_step(artifact_path, updates):
    net = network(artifact_path, updates)
    obs = _observations(2).requires_grad_()
    h = torch.randn(2, 7)
    calls = []
    hook = net.sensory_adapter.register_forward_hook(lambda *args: calls.append(1))
    actual = net._step(obs, h)
    hook.remove()
    assert len(calls) == 1  # Encoding is cached across all inner updates.
    torch.testing.assert_close((1-net.substep_leaks())**updates, 1-net.leaks())
    drive = torch.zeros_like(h)
    drive = drive.index_add(1, net.sensory_indices, net.sensory_adapter(obs[:, :2]))
    goal = torch.cat([obs[:, 2:5], net.extra_params[net._coefficient_rows(obs)]], -1)
    drive = drive.index_add(1, net.descending_indices, net.descending_adapter(goal))
    ref = h
    for _ in range(updates):
        pre = .9 * net.incoming_gains() * net._recurrent_multiply(ref) + drive + net.recurrent_bias
        ref = (1-net.substep_leaks())*ref + net.substep_leaks()*pre.tanh()
    torch.testing.assert_close(actual, ref)
    ga = torch.autograd.grad(actual.square().sum(), obs, retain_graph=True)[0]
    gr = torch.autograd.grad(ref.square().sum(), obs)[0]
    torch.testing.assert_close(ga, gr)


def test_current_observation_reaches_motor_with_four_updates(artifact_path):
    net = network(artifact_path)
    obs = _observations(2).requires_grad_()
    for updates in [1, 4]:
        net.neural_updates = updates
        out = net._step(obs, torch.zeros(2, 7))[:, net.motor_indices]
        grad = torch.autograd.grad(out.sum(), obs)[0][:, :5]
        assert bool(grad.abs().sum() > 0) == (updates == 4)


@pytest.mark.parametrize('bad', [0, -1, True, 1.5])
def test_bad_substeps_rejected(artifact_path, bad):
    with pytest.raises((TypeError, ValueError), match='neural_updates'):
        network(artifact_path, bad)


def test_four_update_sequence_reset_parity(artifact_path):
    net = network(artifact_path)
    obs = _observations(8).reshape(2, 4, 6)
    dones = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0.]])
    h = torch.randn(1, 2, 7)
    batched = net({'obs': obs.reshape(8, 6), 'rnn_states': (h,), 'seq_length': 4, 'dones': dones})
    outputs = []
    state = h[0]
    for t in range(4):
        state = state * (1-dones[:, t, None])
        state = net._step(obs[:, t], state)
        outputs.append(net.mu(state[:, net.motor_indices]))
    torch.testing.assert_close(batched[0], torch.stack(outputs, 1).reshape(8, 2))
    torch.testing.assert_close(batched[3][0][0], state)


def test_beta_sampling_likelihood_entropy_and_learning(artifact_path):
    torch.manual_seed(42)
    model = policy(artifact_path)
    obs = _observations(16)
    with torch.no_grad():
        rollout = model({'obs': obs.clone(), 'is_train': False})
    assert (rollout['actions'].abs() <= 1).all()
    alpha, beta = rollout['policy_storage_mus'], rollout['policy_storage_sigmas']
    torch.testing.assert_close(alpha, torch.full_like(alpha, 2), atol=.003, rtol=0)
    torch.testing.assert_close(beta, torch.full_like(beta, 2), atol=.003, rtol=0)
    torch.testing.assert_close(rollout['mus'], (alpha-beta)/(alpha+beta), atol=1e-7, rtol=1e-5)
    stored = rollout['policy_storage_actions']
    train = model({'obs': obs.clone(), 'prev_actions': stored, 'is_train': True})
    torch.testing.assert_close(train['prev_neglogp'], rollout['neglogpacs'])
    ref = torch.distributions.Beta(alpha, beta)
    torch.testing.assert_close(train['entropy'], (ref.entropy()+math.log(2)).sum(-1))
    assert model.policy_kl(train['mus'], train['sigmas'], alpha, beta).item() < 1e-9
    loss = (train['prev_neglogp']*torch.linspace(-1,1,16)).mean() - .01*train['entropy'].mean() + train['values'].square().mean()
    loss.backward()
    net = model.a2c_network
    for module in [net.mu, net.beta_head, net.sensory_adapter, net.descending_adapter, net.value]:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())
    assert not net.leak_raw.requires_grad and not net.recurrent_values.requires_grad
    restored = policy(artifact_path)
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(restored({'obs': obs.clone(), 'is_train': False})['mus'], rollout['mus'])


def test_unrestricted_beta_can_learn_shapes_below_one(artifact_path):
    net = network(artifact_path, distribution='beta', beta_min_shape=1.0e-4)
    for head in (net.mu, net.beta_head):
        layers = [layer for layer in head.modules() if isinstance(layer, torch.nn.Linear)]
        torch.nn.init.zeros_(layers[-1].weight)
        torch.nn.init.constant_(layers[-1].bias, -2.0)
    obs = _observations(2)
    alpha, beta, _, _ = net({'obs': obs, 'rnn_states': None, 'seq_length': 1})
    assert (alpha > 0).all() and (alpha < 1).all()
    assert (beta > 0).all() and (beta < 1).all()


@pytest.mark.parametrize('minimum', [0, -1, float('nan')])
def test_invalid_beta_min_shape_rejected(artifact_path, minimum):
    with pytest.raises(ValueError, match='beta_min_shape'):
        network(artifact_path, distribution='beta', beta_min_shape=minimum)


@pytest.mark.parametrize('scale', [1., 10., 10000.])
def test_beta_kl_matches_reference_and_rejects_invalid(scale):
    a = torch.tensor([[2., 3.], [4., 5.]])*scale
    b = a+1
    c, d = a*1.05, b*1.1
    ref = torch.distributions.kl_divergence(torch.distributions.Beta(a.double(), b.double()), torch.distributions.Beta(c.double(), d.double())).sum(-1)
    torch.testing.assert_close(beta_policy_kl(a,b,c,d,False).double(), ref, rtol=1e-5, atol=1e-7)
    assert beta_policy_kl(a,b,a,b) == 0
    c[0,0] = float('nan')
    assert torch.isinf(beta_policy_kl(a,b,c,d,False)[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_four_update_triton_forward_backward_parity(artifact_path):
    torch.manual_seed(8)
    ref = network(artifact_path).cuda()
    fused = network(artifact_path, backend='triton_fused').cuda()
    fused.load_state_dict(ref.state_dict())
    obs = _observations(6).cuda()
    h = torch.randn(6, 7, device='cuda')
    outs = []
    for net in [ref, fused]:
        x = obs.clone().requires_grad_()
        state = h.clone().requires_grad_()
        out = net._step(x,state)
        out.square().sum().backward()
        outs.append((out.detach(),x.grad,state.grad))
    for a,b in zip(*outs):
        torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
    for (name,a),(_,b) in zip(ref.named_parameters(),fused.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=3e-5)
