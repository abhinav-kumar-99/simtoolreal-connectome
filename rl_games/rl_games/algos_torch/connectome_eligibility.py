"""Approximate local eligibility for the rate connectome (not BPTT/e-prop exact).

Local sensitivities retain the postsynaptic neuron's own temporal derivative,
including autapses, but drop temporal paths through other neurons. A fixed
random action-feedback matrix supplies spatial credit; motor rows use the
exact instantaneous readout derivative. Actor updates are manual ascent.
"""
from __future__ import annotations

import math
import torch

from rl_games.algos_torch.connectome_network_builder import _select_ranges


class LocalEligibility:
    @torch.no_grad()
    def __init__(self, network, num_envs, config):
        self.net, self.config = network, dict(config)
        if network.weight_mode not in {"adapters_only", "neuron_gains"} or network.learn_dynamics:
            raise ValueError("Eligibility requires adapters_only/neuron_gains and frozen dynamics")
        if network.projection_architecture != "linear":
            raise ValueError("Eligibility currently requires linear interface projections")
        if network.sensory_adapter.bias is not None or network.descending_adapter.bias is not None:
            raise ValueError("Eligibility currently requires bias-free input adapters")
        if not isinstance(network.mu_act, torch.nn.Identity):
            raise ValueError("Eligibility requires a linear action mean")
        for name in ("adapter_lr", "readout_lr", "gain_lr", "trace_clip", "td_clip", "max_grad_norm", "feedback_scale"):
            if not math.isfinite(float(config[name])) or float(config[name]) <= 0:
                raise ValueError(f"eligibility.{name} must be finite and positive")
        self.decay = float(config["gamma"]) * float(config["trace_lambda"])
        if not 0 <= float(config["gamma"]) <= 1 or not 0 <= float(config["trace_lambda"]) <= 1:
            raise ValueError("gamma and trace_lambda must be in [0, 1]")
        self.params = {
            "sensory": network.sensory_adapter.weight,
            "descending": network.descending_adapter.weight,
            "readout": network.mu.weight,
            "readout_bias": network.mu.bias,
        }
        if network.weight_mode == "neuron_gains":
            self.params.update(incoming=network.incoming_gain_raw, outgoing=network.outgoing_gain_raw)
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        device = network.recurrent_values.device
        self.src = network.col_indices
        self.dst = torch.repeat_interleave(torch.arange(network.neuron_count, device=device),
                                          network.crow_indices[1:] - network.crow_indices[:-1])
        self.diag = network.recurrent_values.new_zeros(network.neuron_count)
        diagonal = self.src == self.dst
        self.diag.index_add_(0, self.dst[diagonal], network.recurrent_values[diagonal])
        self.local = {name: parameter.new_zeros((num_envs,) + tuple(parameter.shape))
                      for name, parameter in self.params.items() if name in ("sensory", "descending", "incoming")}
        if network.weight_mode == "neuron_gains":
            # Outgoing gains are shared across edges: keep an edge sensitivity,
            # then reduce to the source gain parameter only after spatial feedback.
            self.local["outgoing_edges"] = network.recurrent_values.new_zeros(num_envs, network.edge_count)
        self.credit = {name: parameter.new_zeros((num_envs,) + tuple(parameter.shape))
                       for name, parameter in self.params.items()}
        generator = torch.Generator(device=device).manual_seed(int(config["feedback_seed"]))
        self.feedback = torch.randn(network.neuron_count, network.mu.out_features,
                                    generator=generator, device=device) * (
                                        float(config["feedback_scale"]) / math.sqrt(network.mu.out_features))
        self.updates = 0

    @torch.no_grad()
    def reset(self, indices):
        for tensor in list(self.local.values()) + list(self.credit.values()):
            tensor[indices] = 0

    @torch.no_grad()
    def observe(self, obs, previous, hidden, score):
        n = self.net
        leak, gi, go = n.leaks(), n.incoming_gains(), n.outgoing_gains()
        activation = ((hidden - (1 - leak) * previous) / leak).clamp(-1, 1)
        derivative = leak * (1 - activation.square())
        local_jacobian = 1 - leak + derivative * n.recurrent_gain * gi * self.diag * go
        learning_signal = score @ self.feedback.T
        learning_signal[:, n.motor_indices] = score @ n.mu.weight
        sensory = _select_ranges(obs, n.sensory_ranges)
        descending = torch.cat((_select_ranges(obs, n.goal_ranges),
                                n.extra_params[n._coefficient_rows(obs)]), dim=-1)
        signals = {}
        for name, indices, inputs in (("sensory", n.sensory_indices, sensory),
                                      ("descending", n.descending_indices, descending)):
            trace = self.local[name]
            trace.mul_(local_jacobian[:, indices, None]).add_(derivative[:, indices, None] * inputs[:, None, :])
            trace.clamp_(-self.config["trace_clip"], self.config["trace_clip"])
            signals[name] = learning_signal[:, indices, None] * trace
        if n.weight_mode == "neuron_gains":
            recurrent = torch.sparse.mm(n._native_csr_matrix(), (previous * go).T).T
            sigmoid_in = n.incoming_gain_raw.sigmoid()
            sigmoid_out = n.outgoing_gain_raw.sigmoid()
            dgi = gi * n.log_gain_span * sigmoid_in * (1 - sigmoid_in)
            dgo = go * n.log_gain_span * sigmoid_out * (1 - sigmoid_out)
            self.local["incoming"].mul_(local_jacobian).add_(derivative * n.recurrent_gain * recurrent * dgi)
            edge_trace = self.local["outgoing_edges"]
            edge_trace.mul_(local_jacobian[:, self.dst]).add_(
                derivative[:, self.dst] * n.recurrent_gain * gi[self.dst] * n.recurrent_values
                * previous[:, self.src] * dgo[self.src])
            for name in ("incoming", "outgoing_edges"):
                self.local[name].clamp_(-self.config["trace_clip"], self.config["trace_clip"])
            signals["incoming"] = learning_signal * self.local["incoming"]
            signals["outgoing"] = torch.zeros_like(self.credit["outgoing"])
            signals["outgoing"].index_add_(1, self.src, learning_signal[:, self.dst] * edge_trace)
        signals["readout"] = score[:, :, None] * hidden[:, None, n.motor_indices]
        signals["readout_bias"] = score
        for name, trace in self.credit.items():
            trace.mul_(self.decay).add_(signals[name]).clamp_(-self.config["trace_clip"], self.config["trace_clip"])

    @torch.no_grad()
    def update(self, td_error):
        if not torch.isfinite(td_error).all():
            raise FloatingPointError("Non-finite TD error")
        delta = td_error.flatten().clamp(-self.config["td_clip"], self.config["td_clip"])
        gradients = {name: (trace * delta.reshape((-1,) + (1,) * (trace.ndim - 1))).mean(0)
                     for name, trace in self.credit.items()}
        norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients.values()))
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite eligibility update")
        scale = (self.config["max_grad_norm"] / norm.clamp_min(1e-12)).clamp(max=1)
        update_sq = norm.new_zeros(())
        per_parameter = {}
        for name, parameter in self.params.items():
            group = "adapter" if name in ("sensory", "descending") else "readout" if name.startswith("readout") else "gain"
            change = gradients[name] * scale * self.config[group + "_lr"]
            parameter.add_(change)
            update_sq += change.square().sum()
            per_parameter[name + "_update_norm"] = change.norm().item()
        self.updates += 1
        return {**per_parameter, "gradient_norm": norm.item(), "update_norm": update_sq.sqrt().item(),
                "td_abs": delta.abs().mean().item(),
                "trace_rms": torch.sqrt(sum(t.square().sum() for t in self.credit.values()) /
                                        sum(t.numel() for t in self.credit.values())).item()}

    def state_dict(self):
        # Physics cannot be restored exactly. Restart with fresh episodes and
        # zero hidden/traces; deliberately do not carry stale episode credit.
        return {"version": 1, "config": self.config, "weight_mode": self.net.weight_mode, "feedback": self.feedback,
                "updates": self.updates, "restart_semantics": "fresh_episodes_zero_traces"}

    @torch.no_grad()
    def load_state_dict(self, state):
        if (state["version"] != 1 or state["config"] != self.config
                or state.get("weight_mode", self.net.weight_mode) != self.net.weight_mode):
            raise ValueError("Eligibility checkpoint algorithm/config mismatch")
        self.feedback.copy_(state["feedback"])
        self.updates = int(state["updates"])
        self.reset(slice(None))
