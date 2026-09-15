"""MaleCNS-constrained recurrent actor for the legacy rl_games stack."""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from rl_games.algos_torch import network_builder


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_artifact(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _repository_root() / candidate
    return candidate.resolve()


def _as_ranges(
    values: Iterable[Iterable[int]], name: str
) -> tuple[tuple[int, int], ...]:
    ranges = tuple((int(start), int(stop)) for start, stop in values)
    if not ranges or any(start < 0 or stop <= start for start, stop in ranges):
        raise ValueError(f"Invalid {name}: {ranges}")
    return ranges


def _select_ranges(
    observations: torch.Tensor, ranges: tuple[tuple[int, int], ...]
) -> torch.Tensor:
    return torch.cat([observations[:, start:stop] for start, stop in ranges], dim=-1)


def _interface_activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name]()
    except KeyError as error:
        raise ValueError(
            f"Unknown interface projection activation {name!r}; "
            f"expected one of {sorted(activations)}"
        ) from error


def _interface_projection(
    input_size: int,
    output_size: int,
    architecture: str,
    hidden_size: int,
    activation: str,
    bias: bool,
) -> nn.Module:
    if architecture == "linear":
        return nn.Linear(input_size, output_size, bias=bias)
    if architecture == "mlp":
        return nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=bias),
            _interface_activation(activation),
            nn.Linear(hidden_size, output_size, bias=bias),
        )
    raise ValueError(
        f"Unknown interface projection architecture {architecture!r}; "
        "expected 'linear' or 'mlp'"
    )


class ConnectomeBuilder(network_builder.NetworkBuilder):
    """Builds a continuous actor whose recurrent state is the MaleCNS circuit."""

    def load(self, params: dict[str, Any]) -> None:
        self.params = params

    def build(self, name: str, **kwargs: Any) -> nn.Module:
        return self.Network(self.params, **kwargs)

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params: dict[str, Any], **kwargs: Any) -> None:
            super().__init__()
            self.actions_num = int(kwargs.pop("actions_num"))
            self.value_size = int(kwargs.pop("value_size", 1))
            self.num_seqs = int(kwargs.pop("num_seqs", 1))
            self.net_type = kwargs.pop("type", "simple")
            input_shape = kwargs.pop("input_shape")
            # These flags are consumed by the model wrapper rather than the raw network.
            kwargs.pop("normalize_input", None)
            kwargs.pop("normalize_value", None)

            connectome = params["connectome"]
            if "adaptation" in connectome and "plasticity_mode" in connectome:
                raise ValueError(
                    "Specify adaptation or legacy plasticity_mode, not both"
                )
            legacy = connectome.get("plasticity_mode")
            if legacy is not None and legacy not in {"frozen_core", "neuron_gains"}:
                raise ValueError(f"Unknown plasticity_mode: {legacy}")
            adaptation = connectome.get("adaptation", {})
            self.weight_mode = adaptation.get(
                "weight_mode",
                "neuron_gains" if legacy == "neuron_gains" else "adapters_only",
            )
            if self.weight_mode not in {
                "adapters_only",
                "neuron_gains",
                "low_rank",
                "edgewise",
            }:
                raise ValueError(f"Unknown adaptation weight_mode: {self.weight_mode}")
            self.learn_dynamics = adaptation.get(
                "learn_dynamics", legacy == "neuron_gains"
            )
            if not isinstance(self.learn_dynamics, bool):
                raise TypeError("learn_dynamics must be a YAML boolean")
            self.adaptation_rank = adaptation.get("rank", 4)
            if isinstance(self.adaptation_rank, bool) or not isinstance(
                self.adaptation_rank, int
            ):
                raise TypeError("adaptation rank must be an integer")
            if self.adaptation_rank < 1:
                raise ValueError("adaptation rank must be positive")
            bounds = tuple(
                float(x) for x in adaptation.get("edge_scale_bounds", [0.0625, 16.0])
            )
            if (
                len(bounds) != 2
                or not (0 < bounds[0] < 1 < bounds[1])
                or not all(math.isfinite(x) for x in bounds)
            ):
                raise ValueError(
                    "edge_scale_bounds must be finite positive bounds straddling one"
                )
            self.edge_log_min, self.edge_log_max = map(math.log, bounds)
            self.backend_options = connectome.get("backend_options", {})
            if connectome.get("dtype", "float32") != "float32":
                raise ValueError(
                    "Connectome sparse recurrence currently requires dtype: float32"
                )
            artifact_path = _resolve_artifact(connectome["artifact_path"])
            if not artifact_path.exists():
                raise FileNotFoundError(
                    f"Missing connectome artifact {artifact_path}. "
                    "Run scripts/prepare_malecns_connectome.py first."
                )
            with np.load(artifact_path, allow_pickle=False) as artifact:
                crow_indices = artifact["crow_indices"].astype(np.int64)
                col_indices = artifact["col_indices"].astype(np.int64)
                values = artifact["values"].astype(np.float32)
                sensory_indices = artifact["sensory_indices"].astype(np.int64)
                descending_indices = artifact["descending_indices"].astype(np.int64)
                motor_indices = artifact["motor_indices"].astype(np.int64)
                body_ids = artifact["body_ids"].astype(np.int64)

            expected = connectome["expected"]
            self.neuron_count = int(expected["neurons"])
            self.edge_count = int(expected["edges"])
            observed = {
                "neurons": len(body_ids),
                "edges": len(values),
                "sensory_neurons": len(sensory_indices),
                "descending_neurons": len(descending_indices),
                "motor_neurons": len(motor_indices),
            }
            if observed != {key: int(value) for key, value in expected.items()}:
                raise ValueError(
                    f"Connectome artifact counts differ: {observed} != {expected}"
                )
            if (
                len(crow_indices) != self.neuron_count + 1
                or int(crow_indices[-1]) != self.edge_count
            ):
                raise ValueError("Invalid CSR row pointer")
            if len(np.unique(body_ids)) != self.neuron_count:
                raise ValueError("Neuron body IDs must be unique")
            for population_name, indices in (
                ("sensory", sensory_indices),
                ("descending", descending_indices),
                ("motor", motor_indices),
            ):
                if np.any(indices < 0) or np.any(indices >= self.neuron_count):
                    raise ValueError(f"Invalid {population_name} population indices")

            self.register_buffer("crow_indices", torch.from_numpy(crow_indices))
            self.register_buffer("col_indices", torch.from_numpy(col_indices))
            self.register_buffer("recurrent_values", torch.from_numpy(values))
            self.register_buffer("sensory_indices", torch.from_numpy(sensory_indices))
            self.register_buffer(
                "descending_indices", torch.from_numpy(descending_indices)
            )
            self.register_buffer("motor_indices", torch.from_numpy(motor_indices))
            self.operator_backend = connectome.get("operator_backend", "native_csr")
            supported_backends = {
                "native_csr",
                "native_coo",
                "dense",
                "torch_sparse",
                "cusparse",
                "triton_fused",
            }
            if self.operator_backend not in supported_backends:
                raise ValueError(
                    f"Unknown operator_backend {self.operator_backend!r}; "
                    f"expected one of {sorted(supported_backends)}"
                )
            self._cached_recurrent_operator: Any = None
            self._backend_graph = None

            observations = connectome["observations"]
            self.sensory_ranges = _as_ranges(
                observations["sensory_ranges"], "sensory_ranges"
            )
            self.goal_ranges = _as_ranges(observations["goal_ranges"], "goal_ranges")
            self.policy_observation_size = int(observations["policy_size"])
            sensory_size = sum(stop - start for start, stop in self.sensory_ranges)
            goal_size = sum(stop - start for start, stop in self.goal_ranges)
            if sensory_size != int(observations["sensory_size"]):
                raise ValueError("Sensory observation ranges do not match sensory_size")
            if goal_size != int(observations["goal_size"]):
                raise ValueError("Goal observation ranges do not match goal_size")
            if (
                max(stop for _, stop in self.sensory_ranges + self.goal_ranges)
                > self.policy_observation_size
            ):
                raise ValueError(
                    "Observation range exceeds the policy observation size"
                )

            self.coef_embedding_size = 0
            if self.net_type == "extra_param":
                expected_input_shape = (self.policy_observation_size + 1,)
                if tuple(input_shape) != expected_input_shape:
                    raise ValueError(
                        "Expected SAPG input_shape "
                        f"{expected_input_shape}, got {input_shape}"
                    )
                self.coef_id_idx = int(kwargs.pop("coef_id_idx"))
                if self.coef_id_idx != self.policy_observation_size:
                    raise ValueError(
                        f"SAPG coefficient index {self.coef_id_idx} != policy size "
                        f"{self.policy_observation_size}"
                    )
                coef_ids = torch.as_tensor(
                    kwargs.pop("coef_ids"), dtype=torch.float32
                ).flatten()
                self.register_buffer("coef_ids", coef_ids.detach().clone())
                self.coef_embedding_size = int(connectome["sapg_embedding_size"])
                requested_embedding_size = int(
                    kwargs.pop("param_size", self.coef_embedding_size)
                )
                if requested_embedding_size != self.coef_embedding_size:
                    raise ValueError(
                        f"SAPG embedding size {requested_embedding_size} != configured "
                        f"size {self.coef_embedding_size}"
                    )
                self.extra_params = nn.Parameter(
                    torch.empty(
                        len(coef_ids), self.coef_embedding_size, dtype=torch.float32
                    )
                )
                nn.init.normal_(self.extra_params)
            else:
                self.coef_id_idx = None
                self.register_buffer("coef_ids", torch.empty(0, dtype=torch.float32))
                if tuple(input_shape) != (self.policy_observation_size,):
                    raise ValueError(
                        "Expected input_shape "
                        f"{(self.policy_observation_size,)}, got {input_shape}"
                    )
            if kwargs:
                raise TypeError(
                    f"Unsupported connectome build options: {sorted(kwargs)}"
                )

            adapter = connectome["population_adapters"]
            projections = connectome.get("interface_projections", {})
            self.projection_architecture = str(
                projections.get("architecture", "linear")
            ).lower()
            hidden_size = projections.get("hidden_size", 256)
            if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
                raise TypeError("interface projection hidden_size must be an integer")
            if hidden_size < 1:
                raise ValueError("interface projection hidden_size must be positive")
            self.projection_hidden_size = hidden_size
            self.projection_activation = str(
                projections.get("activation", "elu")
            ).lower()
            # Validate all projection settings even for the linear option so a
            # later YAML architecture switch cannot expose a latent bad value.
            _interface_activation(self.projection_activation)
            input_bias = bool(adapter.get("bias", False))
            self.sensory_adapter = _interface_projection(
                sensory_size,
                len(sensory_indices),
                self.projection_architecture,
                self.projection_hidden_size,
                self.projection_activation,
                bias=input_bias,
            )
            self.descending_adapter = _interface_projection(
                goal_size + self.coef_embedding_size,
                len(descending_indices),
                self.projection_architecture,
                self.projection_hidden_size,
                self.projection_activation,
                bias=input_bias,
            )

            dynamics = connectome["dynamics"]
            self.neural_updates = dynamics.get("neural_updates", 1)
            if isinstance(self.neural_updates, bool) or not isinstance(self.neural_updates, int):
                raise TypeError("dynamics.neural_updates must be an integer")
            if self.neural_updates < 1:
                raise ValueError("dynamics.neural_updates must be positive")
            if dynamics.get("activation") != "tanh":
                raise ValueError("Connectome recurrent activation must be tanh")
            self.recurrent_gain = float(dynamics["beta"])
            gain_min, gain_max = (float(value) for value in dynamics["gain_bounds"])
            if legacy is None:
                gain_min, gain_max = math.sqrt(bounds[0]), math.sqrt(bounds[1])
            initial_gain = float(dynamics["initial_gain"])
            if not 0.0 < gain_min < initial_gain < gain_max:
                raise ValueError(
                    "initial_gain must lie strictly inside positive gain_bounds"
                )
            self.log_gain_min = math.log(gain_min)
            self.log_gain_span = math.log(gain_max) - self.log_gain_min
            initial_fraction = (
                math.log(initial_gain) - self.log_gain_min
            ) / self.log_gain_span
            initial_gain_raw = math.log(initial_fraction / (1.0 - initial_fraction))
            self.incoming_gain_raw = nn.Parameter(
                torch.full((self.neuron_count,), initial_gain_raw, dtype=torch.float32)
            )
            self.outgoing_gain_raw = nn.Parameter(
                torch.full((self.neuron_count,), initial_gain_raw, dtype=torch.float32)
            )
            initial_leak = float(dynamics["initial_leak"])
            if not 0.0 < initial_leak < 1.0:
                raise ValueError("initial_leak must be strictly between zero and one")
            self.leak_raw = nn.Parameter(
                torch.full(
                    (self.neuron_count,),
                    math.log(initial_leak / (1.0 - initial_leak)),
                    dtype=torch.float32,
                )
            )
            self.recurrent_bias = nn.Parameter(
                torch.full(
                    (self.neuron_count,),
                    float(dynamics["initial_bias"]),
                    dtype=torch.float32,
                )
            )

            self.mu = _interface_projection(
                len(motor_indices),
                self.actions_num,
                self.projection_architecture,
                self.projection_hidden_size,
                self.projection_activation,
                bias=True,
            )
            self.value = nn.Linear(self.neuron_count, self.value_size)
            continuous = params["space"]["continuous"]
            self.action_distribution = continuous.get("distribution", "gaussian")
            if self.action_distribution not in {"gaussian", "beta"}:
                raise ValueError("continuous.distribution must be gaussian or beta")
            self.mu_act = self.activations_factory.create(continuous["mu_activation"])
            self.sigma_act = self.activations_factory.create(
                continuous["sigma_activation"]
            )
            self.fixed_sigma = continuous["fixed_sigma"]
            if self.action_distribution == "beta":
                self.beta_initial_shape = float(continuous.get("beta_initial_shape", 2.0))
                if not math.isfinite(self.beta_initial_shape) or self.beta_initial_shape <= 1:
                    raise ValueError("beta_initial_shape must be finite and greater than one")
                self.beta_head = _interface_projection(
                    len(motor_indices), self.actions_num,
                    self.projection_architecture, self.projection_hidden_size,
                    self.projection_activation, bias=True,
                )
            elif self.fixed_sigma == "coef_cond":
                if self.net_type != "extra_param":
                    raise ValueError(
                        "coef_cond sigma requires SAPG extra_param construction"
                    )
                self.sigma = nn.Parameter(
                    torch.zeros(len(self.coef_ids), self.actions_num)
                )
            elif self.fixed_sigma == "fixed":
                self.sigma = nn.Parameter(torch.zeros(self.actions_num))
            else:
                raise ValueError("Connectome actor supports fixed or coef_cond sigma")

            self._initialize_linear_layers()
            for parameter in (self.incoming_gain_raw, self.outgoing_gain_raw):
                parameter.requires_grad_(self.weight_mode == "neuron_gains")
            for parameter in (self.leak_raw, self.recurrent_bias):
                parameter.requires_grad_(self.learn_dynamics)
            if self.weight_mode == "low_rank":
                self.edge_u = nn.Parameter(
                    torch.empty(self.neuron_count, self.adaptation_rank)
                )
                self.edge_v = nn.Parameter(
                    torch.zeros(self.neuron_count, self.adaptation_rank)
                )
                nn.init.normal_(self.edge_u, std=0.1)
            elif self.weight_mode == "edgewise":
                self.edge_raw = nn.Parameter(torch.zeros(self.edge_count))

        def backend_graph(self):
            if self._backend_graph is None:
                from rl_games.algos_torch.connectome_ops import Graph

                self._backend_graph = Graph(self.crow_indices, self.col_indices)
            return self._backend_graph

        def effective_values(self):
            if self.weight_mode == "low_rank":
                graph = self.backend_graph()
                score = (self.edge_u[graph.rows] * self.edge_v[self.col_indices]).sum(
                    -1
                )
                score = score / math.sqrt(self.adaptation_rank)
            elif self.weight_mode == "edgewise":
                score = self.edge_raw
            else:
                return self.recurrent_values
            # Shift the logistic so zero scores are exactly the identity even
            # for asymmetric bounds. Evaluate only the E existing connections.
            span = self.edge_log_max - self.edge_log_min
            fraction = -self.edge_log_min / span
            offset = math.log(fraction / (1 - fraction))
            delta = self.edge_log_min + span * torch.sigmoid(score + offset)
            return self.recurrent_values * delta.exp()

        def _load_from_state_dict(self, *args, **kwargs):
            self._cached_recurrent_operator = None
            self._backend_graph = None
            return super()._load_from_state_dict(*args, **kwargs)

        def _initialize_linear_layers(self) -> None:
            for module in (self.sensory_adapter, self.descending_adapter, self.value):
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            mu_layers = [
                layer for layer in self.mu.modules() if isinstance(layer, nn.Linear)
            ]
            for layer in mu_layers:
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            nn.init.uniform_(mu_layers[-1].weight, -1.0e-3, 1.0e-3)
            nn.init.zeros_(mu_layers[-1].bias)
            if self.action_distribution == "beta":
                layers = [layer for layer in self.beta_head.modules() if isinstance(layer, nn.Linear)]
                for layer in layers:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                nn.init.uniform_(layers[-1].weight, -1.0e-3, 1.0e-3)
                target = self.beta_initial_shape - 1.0
                initial_raw = target + math.log(-math.expm1(-target))
                nn.init.constant_(mu_layers[-1].bias, initial_raw)
                nn.init.constant_(layers[-1].bias, initial_raw)

        def incoming_gains(self) -> torch.Tensor:
            log_gain = self.log_gain_min + self.log_gain_span * torch.sigmoid(
                self.incoming_gain_raw
            )
            return torch.exp(log_gain)

        def outgoing_gains(self) -> torch.Tensor:
            log_gain = self.log_gain_min + self.log_gain_span * torch.sigmoid(
                self.outgoing_gain_raw
            )
            return torch.exp(log_gain)

        def leaks(self) -> torch.Tensor:
            return torch.sigmoid(self.leak_raw)

        def substep_leaks(self) -> torch.Tensor:
            if self.neural_updates == 1:
                return self.leaks()
            # Preserve passive retention over one control interval. Stable even
            # for learned leak logits, without detaching their gradients.
            return -torch.expm1(
                torch.nn.functional.logsigmoid(-self.leak_raw) / self.neural_updates
            )

        def _apply(self, fn, recurse: bool = True):
            # The cached operator is reconstructed from persistent buffers after
            # device or dtype moves, and is deliberately absent from checkpoints.
            self._cached_recurrent_operator = None
            self._backend_graph = None
            return super()._apply(fn, recurse)

        def _native_csr_matrix(self, values=None) -> torch.Tensor:
            return torch.sparse_csr_tensor(
                self.crow_indices,
                self.col_indices,
                self.recurrent_values if values is None else values,
                size=(self.neuron_count, self.neuron_count),
                device=self.recurrent_values.device,
                dtype=torch.float32,
            )

        def recurrent_matrix(self, values=None):
            dynamic = self.weight_mode in {"low_rank", "edgewise"}
            if values is None:
                values = self.effective_values()
            if not dynamic and self._cached_recurrent_operator is not None:
                return self._cached_recurrent_operator
            if self.operator_backend == "native_csr":
                operator = self._native_csr_matrix(values)
            elif self.operator_backend == "native_coo":
                row_counts = self.crow_indices[1:] - self.crow_indices[:-1]
                rows = torch.repeat_interleave(
                    torch.arange(
                        self.neuron_count,
                        device=self.crow_indices.device,
                        dtype=self.col_indices.dtype,
                    ),
                    row_counts,
                )
                operator = torch.sparse_coo_tensor(
                    torch.stack((rows, self.col_indices)),
                    values,
                    size=(self.neuron_count, self.neuron_count),
                    device=self.recurrent_values.device,
                    dtype=torch.float32,
                ).coalesce()
            elif self.operator_backend == "dense":
                operator = self._native_csr_matrix(values).to_dense()
            elif self.operator_backend in {"cusparse", "triton_fused"}:
                return self.backend_graph()
            else:
                try:
                    from torch_sparse import SparseTensor
                except ImportError as exc:
                    raise ImportError(
                        "operator_backend: torch_sparse requires the optional "
                        "torch-scatter and torch-sparse wheels"
                    ) from exc
                operator = SparseTensor(
                    rowptr=self.crow_indices,
                    col=self.col_indices,
                    value=values,
                    sparse_sizes=(self.neuron_count, self.neuron_count),
                    is_sorted=True,
                    trust_data=True,
                )
            if not dynamic:
                self._cached_recurrent_operator = operator
            return operator

        def _recurrent_multiply(
            self, hidden: torch.Tensor, operator=None, values=None
        ) -> torch.Tensor:
            projected = (self.outgoing_gains() * hidden).transpose(0, 1)
            operator = self.recurrent_matrix(values) if operator is None else operator
            if self.operator_backend == "cusparse":
                from rl_games.algos_torch.connectome_ops import cusparse_mm

                recurrent = cusparse_mm(
                    operator,
                    self.effective_values() if values is None else values,
                    projected,
                    self.backend_options,
                )
            elif self.operator_backend == "torch_sparse":
                recurrent = operator.matmul(projected)
            elif self.operator_backend == "dense":
                recurrent = torch.mm(operator, projected)
            else:
                recurrent = torch.sparse.mm(operator, projected)
            return recurrent.transpose(0, 1)

        def _coefficient_rows(self, observations: torch.Tensor) -> torch.Tensor:
            coefficient = observations[:, self.coef_id_idx].float().unsqueeze(1)
            return torch.abs(coefficient - self.coef_ids.unsqueeze(0)).argmin(dim=1)

        def _step(
            self,
            observations: torch.Tensor,
            hidden: torch.Tensor,
            operator=None,
            values=None,
        ) -> torch.Tensor:
            if hidden.device.type == "cuda":
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    autocast_disabled = torch.amp.autocast("cuda", enabled=False)
                else:
                    autocast_disabled = torch.cuda.amp.autocast(enabled=False)
            else:
                autocast_disabled = nullcontext()
            # The adapters, state, gains, and sparse recurrence form one FP32
            # recurrent core even when the surrounding PPO update uses AMP.
            with autocast_disabled:
                observations = observations.float()
                hidden = hidden.float()
                sensory = _select_ranges(observations, self.sensory_ranges)
                goal = _select_ranges(observations, self.goal_ranges)
                if self.net_type == "extra_param":
                    goal = torch.cat(
                        (
                            goal,
                            self.extra_params[self._coefficient_rows(observations)],
                        ),
                        dim=-1,
                    )
                sensory_drive = self.sensory_adapter(sensory)
                descending_drive = self.descending_adapter(goal)
                leak = self.substep_leaks()
                incoming = self.incoming_gains()
                if self.operator_backend == "triton_fused":
                    from rl_games.algos_torch.connectome_triton import fused_step

                    graph = self.backend_graph()
                    step_values = self.effective_values() if values is None else values
                    outgoing = self.outgoing_gains()
                    for _ in range(self.neural_updates):
                        hidden = fused_step(
                            graph, step_values, hidden, incoming, outgoing, leak,
                            self.recurrent_bias, sensory_drive, descending_drive,
                            self.sensory_indices, self.descending_indices,
                            self.recurrent_gain,
                        )
                    return hidden
                drive = hidden.new_zeros(hidden.shape)
                drive = drive.index_add(1, self.sensory_indices, sensory_drive)
                drive = drive.index_add(1, self.descending_indices, descending_drive)
                for _ in range(self.neural_updates):
                    recurrent = self._recurrent_multiply(hidden, operator, values)
                    preactivation = self.recurrent_gain * incoming * recurrent + drive + self.recurrent_bias
                    hidden = (1.0 - leak) * hidden + leak * torch.tanh(preactivation)
                return hidden

        def forward(self, obs_dict: dict[str, Any]):
            observations = obs_dict["obs"]
            if observations.ndim != 2:
                raise ValueError(
                    f"Expected rank-2 observations, got {observations.shape}"
                )
            minimum_features = self.policy_observation_size + int(
                self.net_type == "extra_param"
            )
            if observations.shape[1] < minimum_features:
                raise ValueError(
                    f"Expected at least {minimum_features} observation features, "
                    f"got {observations.shape[1]}"
                )
            sequence_length = int(obs_dict.get("seq_length", 1))
            if observations.shape[0] % sequence_length:
                raise ValueError("Batch size must be divisible by seq_length")
            sequence_count = observations.shape[0] // sequence_length
            sequence = observations.reshape(
                sequence_count, sequence_length, -1
            ).transpose(0, 1)

            states = obs_dict.get("rnn_states")
            if states is None:
                hidden = observations.new_zeros(
                    (sequence_count, self.neuron_count), dtype=torch.float32
                )
            else:
                hidden = states[0] if isinstance(states, (tuple, list)) else states
                if hidden.ndim == 3:
                    hidden = hidden[0]
                hidden = hidden.float()
            dones = obs_dict.get("dones")
            if dones is not None:
                dones = dones.reshape(sequence_count, sequence_length, -1).transpose(
                    0, 1
                )

            outputs = []
            values = self.effective_values()
            operator = self.recurrent_matrix(values)
            for step, step_observations in enumerate(sequence):
                if dones is not None:
                    hidden = hidden * (1.0 - dones[step].float())
                hidden = self._step(step_observations, hidden, operator, values)
                outputs.append(hidden)
            output = (
                torch.stack(outputs).transpose(0, 1).reshape(observations.shape[0], -1)
            )

            value = self.value(output)
            if self.action_distribution == "beta":
                # Shape heads and special-function inputs stay FP32 under AMP.
                with torch.autocast(device_type=output.device.type, enabled=False):
                    motor = output[:, self.motor_indices].float()
                    alpha = 1.0 + torch.nn.functional.softplus(self.mu(motor))
                    beta = 1.0 + torch.nn.functional.softplus(self.beta_head(motor))
                return alpha, beta, value, (hidden.unsqueeze(0),)
            mu = self.mu_act(self.mu(output[:, self.motor_indices]))
            if self.fixed_sigma == "coef_cond":
                sigma = self.sigma_act(self.sigma[self._coefficient_rows(observations)])
            else:
                sigma = self.sigma_act(self.sigma).expand_as(mu)
            return mu, sigma, value, (hidden.unsqueeze(0),)

        def is_rnn(self) -> bool:
            return True

        def get_default_rnn_state(self):
            return (
                torch.zeros((1, self.num_seqs, self.neuron_count), dtype=torch.float32),
            )

        def get_value_layer(self):
            return self.value

        def is_separate_critic(self) -> bool:
            return False
