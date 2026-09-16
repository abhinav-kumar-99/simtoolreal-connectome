"""MaleCNS-constrained recurrent actor for the legacy rl_games stack."""

from __future__ import annotations

import math
from collections.abc import Mapping
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


class _GroupedSensoryAdapter(nn.Module):
    """Block-sparse body/efference projection into proprioceptor rows."""

    def __init__(
        self,
        input_size: int,
        sensory_size: int,
        proprioceptor_positions: np.ndarray,
        groups: list[dict[str, Any]],
        dof_count: int,
        fingertip_count: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if not groups:
            raise ValueError("structured_input_adapter.groups must not be empty")
        if input_size != 3 * dof_count + 3 * fingertip_count:
            raise ValueError(
                "Structured body/efference size must equal three DOF blocks plus "
                "three coordinates per fingertip"
            )
        proprioceptor_positions = np.asarray(
            proprioceptor_positions, dtype=np.int64
        )
        if len(proprioceptor_positions) < len(groups):
            raise ValueError("Need at least one proprioceptor per robot group")
        self.input_size = input_size
        self.sensory_size = sensory_size
        self.register_buffer(
            "proprioceptor_positions", torch.from_numpy(proprioceptor_positions)
        )

        group_features: list[np.ndarray] = []
        covered_dofs: list[int] = []
        covered_fingertips: list[int] = []
        self.group_names: list[str] = []
        for index, group in enumerate(groups):
            name = str(group.get("name", f"group_{index}"))
            if name in self.group_names:
                raise ValueError(f"Duplicate structured input group {name!r}")
            dof_start, dof_stop = (int(value) for value in group["dof_range"])
            if dof_start < 0 or dof_stop <= dof_start or dof_stop > dof_count:
                raise ValueError(f"Invalid DOF range for structured group {name!r}")
            dofs = np.arange(dof_start, dof_stop, dtype=np.int64)
            covered_dofs.extend(dofs.tolist())
            feature_parts = [
                dofs,
                dof_count + dofs,
                2 * dof_count + dofs,
            ]
            fingertip_index = group.get("fingertip_index")
            if fingertip_index is not None:
                fingertip_index = int(fingertip_index)
                if fingertip_index < 0 or fingertip_index >= fingertip_count:
                    raise ValueError(
                        f"Invalid fingertip index for structured group {name!r}"
                    )
                covered_fingertips.append(fingertip_index)
                start = 3 * dof_count + 3 * fingertip_index
                feature_parts.append(np.arange(start, start + 3, dtype=np.int64))
            group_features.append(np.concatenate(feature_parts))
            self.group_names.append(name)
        if sorted(covered_dofs) != list(range(dof_count)):
            raise ValueError("Structured groups must partition every robot DOF once")
        if sorted(covered_fingertips) != list(range(fingertip_count)):
            raise ValueError(
                "Structured finger groups must assign every fingertip exactly once"
            )

        feature_counts = np.asarray(
            [len(features) for features in group_features], dtype=np.float64
        )
        quotas = len(proprioceptor_positions) * feature_counts / feature_counts.sum()
        allocations = np.floor(quotas).astype(np.int64)
        remaining = len(proprioceptor_positions) - int(allocations.sum())
        order = sorted(
            range(len(groups)), key=lambda i: (-(quotas[i] - allocations[i]), i)
        )
        for index in order[:remaining]:
            allocations[index] += 1
        if np.any(allocations < 1) or int(allocations.sum()) != len(
            proprioceptor_positions
        ):
            raise ValueError("Invalid proportional proprioceptor allocation")
        self.group_allocations = tuple(int(value) for value in allocations)

        self.group_adapters = nn.ModuleList()
        for index, (features, output_size) in enumerate(
            zip(group_features, self.group_allocations)
        ):
            self.register_buffer(
                f"group_feature_indices_{index}", torch.from_numpy(features)
            )
            self.group_adapters.append(
                nn.Linear(len(features), output_size, bias=bias)
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        compact_outputs = []
        for index, adapter in enumerate(self.group_adapters):
            features = getattr(self, f"group_feature_indices_{index}")
            compact_outputs.append(adapter(inputs.index_select(1, features)))
        compact = torch.cat(compact_outputs, dim=-1)
        output = inputs.new_zeros((inputs.shape[0], self.sensory_size))
        return output.index_copy(1, self.proprioceptor_positions, compact)


class _FixedPopulationEncoder(nn.Module):
    """Deterministically route policy features into selected fly populations."""

    def __init__(
        self,
        *,
        policy_size: int,
        output_size: int,
        mappings: list[dict[str, Any]],
        target_positions: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if not mappings:
            raise ValueError("fixed population mappings must not be empty")
        if target_positions is None:
            target_positions = np.arange(output_size, dtype=np.int64)
        else:
            target_positions = np.asarray(target_positions, dtype=np.int64)
        if np.any(target_positions < 0) or np.any(target_positions >= output_size):
            raise ValueError("fixed encoder target population contains invalid positions")

        self.policy_size = int(policy_size)
        self.output_size = int(output_size)
        self.encodings: list[str] = []
        self.scales: list[float] = []
        used_targets: list[int] = []
        for index, mapping in enumerate(mappings):
            source_start, source_stop = (
                int(value) for value in mapping["source_range"]
            )
            if (
                source_start < 0
                or source_stop <= source_start
                or source_stop > self.policy_size
            ):
                raise ValueError(
                    f"Invalid fixed encoder source range {source_start, source_stop}"
                )
            encoding = str(mapping.get("encoding", "signed_tanh")).lower()
            if encoding not in {"signed_tanh", "opponent_tanh"}:
                raise ValueError(
                    "fixed encoder encoding must be signed_tanh or opponent_tanh"
                )
            scale = float(mapping.get("scale", 1.0))
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("fixed encoder scale must be finite and positive")
            source_width = source_stop - source_start
            encoded_width = source_width * (2 if encoding == "opponent_tanh" else 1)
            target_offset = int(mapping["target_offset"])
            target_stop = target_offset + encoded_width
            if target_offset < 0 or target_stop > len(target_positions):
                raise ValueError(
                    "fixed encoder target range exceeds its selected population"
                )
            selected_targets = target_positions[target_offset:target_stop]
            used_targets.extend(int(value) for value in selected_targets)
            self.register_buffer(
                f"source_indices_{index}",
                torch.arange(source_start, source_stop, dtype=torch.long),
            )
            self.register_buffer(
                f"target_indices_{index}",
                torch.from_numpy(selected_targets.copy()),
            )
            self.encodings.append(encoding)
            self.scales.append(scale)
        if len(set(used_targets)) != len(used_targets):
            raise ValueError("fixed encoder mappings must not overlap target cells")

    def _encode(
        self, observations: torch.Tensor, *, spike_rates: bool
    ) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] < self.policy_size:
            raise ValueError(
                f"Expected fixed encoder observations [batch, >= {self.policy_size}]"
            )
        output = observations.new_zeros(
            (observations.shape[0], self.output_size), dtype=torch.float32
        )
        for index, (encoding, scale) in enumerate(
            zip(self.encodings, self.scales)
        ):
            source_indices = getattr(self, f"source_indices_{index}")
            target_indices = getattr(self, f"target_indices_{index}")
            encoded = torch.tanh(
                observations.float().index_select(1, source_indices) / scale
            )
            if encoding == "opponent_tanh":
                encoded = torch.cat(
                    (torch.clamp_min(encoded, 0.0), torch.clamp_min(-encoded, 0.0)),
                    dim=-1,
                )
            elif spike_rates:
                # A signed scalar is represented as modulation around a half-rate
                # baseline. Opponent codes are already non-negative rates, and
                # unused population cells remain exactly silent.
                encoded = 0.5 * (encoded + 1.0)
            output.index_copy_(1, target_indices, encoded)
        return output

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self._encode(observations, spike_rates=False)

    def spike_rates(self, observations: torch.Tensor) -> torch.Tensor:
        """Return fixed population codes in the unit interval for LIF drive."""
        return self._encode(observations, spike_rates=True)


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
            structured_input = connectome.get("structured_input_adapter")
            if structured_input is not None and not isinstance(
                structured_input, Mapping
            ):
                raise TypeError("structured_input_adapter must be a YAML mapping")
            fixed_input = connectome.get("fixed_input_encoder")
            if fixed_input is not None and not isinstance(fixed_input, Mapping):
                raise TypeError("fixed_input_encoder must be a YAML mapping")
            if structured_input is not None and fixed_input is not None:
                raise ValueError(
                    "Specify structured_input_adapter or fixed_input_encoder, not both"
                )
            reservoir_readout = connectome.get("reservoir_readout", {})
            if not isinstance(reservoir_readout, Mapping):
                raise TypeError("reservoir_readout must be a YAML mapping")
            self.cache_reservoir_features = bool(
                reservoir_readout.get("enabled", False)
            )
            if self.cache_reservoir_features:
                if fixed_input is None:
                    raise ValueError(
                        "reservoir_readout requires fixed_input_encoder"
                    )
                if self.weight_mode != "adapters_only" or self.learn_dynamics:
                    raise ValueError(
                        "reservoir_readout requires adapters_only weights and frozen dynamics"
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
                visual_indices = artifact['visual_indices'].astype(np.int64) if 'visual_indices' in artifact else np.array([], dtype=np.int64)
                visual_grid = artifact['visual_grid'].astype(np.float32) if 'visual_grid' in artifact else None
                if structured_input is not None or fixed_input is not None:
                    population_config = (
                        structured_input if structured_input is not None else fixed_input
                    )
                    assert population_config is not None
                    proprioceptor_key = str(
                        population_config.get(
                            "proprioceptor_artifact_key",
                            "front_proprioceptors_indices",
                        )
                    )
                    tactile_key = str(
                        population_config.get(
                            "tactile_artifact_key", "front_tactile_indices"
                        )
                    )
                    missing = [
                        key
                        for key in (proprioceptor_key, tactile_key)
                        if key not in artifact
                    ]
                    if missing:
                        raise ValueError(
                            f"Structured input artifact is missing arrays {missing}; "
                            "rerun scripts/prepare_malecns_connectome.py"
                        )
                    proprioceptor_indices = artifact[proprioceptor_key].astype(
                        np.int64
                    )
                    tactile_indices = artifact[tactile_key].astype(np.int64)
                else:
                    proprioceptor_indices = None
                    tactile_indices = None

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
            if structured_input is not None or fixed_input is not None:
                assert proprioceptor_indices is not None and tactile_indices is not None
                if np.intersect1d(proprioceptor_indices, tactile_indices).size:
                    raise ValueError(
                        "Structured proprioceptor and tactile populations overlap"
                    )
                structured_union = np.sort(
                    np.concatenate((proprioceptor_indices, tactile_indices, visual_indices))
                )
                if not np.array_equal(structured_union, np.sort(sensory_indices)):
                    raise ValueError(
                        "Structured proprioceptor and tactile populations must exactly "
                        "partition sensory_indices"
                    )

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
            self.context_ranges = (
                _as_ranges(observations["context_ranges"], "context_ranges")
                if structured_input is not None or fixed_input is not None
                else ()
            )
            self.policy_observation_size = int(observations["policy_size"])
            sensory_size = sum(stop - start for start, stop in self.sensory_ranges)
            goal_size = sum(stop - start for start, stop in self.goal_ranges)
            context_size = sum(stop - start for start, stop in self.context_ranges)
            if sensory_size != int(observations["sensory_size"]):
                raise ValueError("Sensory observation ranges do not match sensory_size")
            if goal_size != int(observations["goal_size"]):
                raise ValueError("Goal observation ranges do not match goal_size")
            if (structured_input is not None or fixed_input is not None) and context_size != int(
                observations["context_size"]
            ):
                raise ValueError("Context observation ranges do not match context_size")
            if (
                max(
                    stop
                    for _, stop in (
                        self.sensory_ranges
                        + self.context_ranges
                        + self.goal_ranges
                    )
                )
                > self.policy_observation_size
            ):
                raise ValueError(
                    "Observation range exceeds the policy observation size"
                )
            covered_observations = [
                index
                for start, stop in (
                    self.sensory_ranges + self.context_ranges + self.goal_ranges
                )
                for index in range(start, stop)
            ]
            if sorted(covered_observations) != list(
                range(self.policy_observation_size)
            ):
                raise ValueError(
                    "Sensory, context, and goal ranges must partition policy observations"
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
            if fixed_input is not None:
                mode = str(fixed_input.get("mode", "population_code_v1")).lower()
                if mode != "population_code_v1":
                    raise ValueError(
                        "fixed_input_encoder.mode must be population_code_v1"
                    )
                assert proprioceptor_indices is not None
                sensory_positions = {
                    int(neuron): position
                    for position, neuron in enumerate(sensory_indices)
                }
                proprioceptor_positions = np.asarray(
                    [sensory_positions[int(neuron)] for neuron in proprioceptor_indices],
                    dtype=np.int64,
                )
                self.sensory_adapter_mode = "fixed_population_code"
                self.sensory_adapter = _FixedPopulationEncoder(
                    policy_size=self.policy_observation_size,
                    output_size=len(sensory_indices),
                    mappings=list(fixed_input["sensory_mappings"]),
                    target_positions=proprioceptor_positions,
                )
                self.descending_projection_architecture = "fixed_population_code"
                retina = fixed_input.get('retina')
                if len(visual_indices):
                    if retina is None or visual_grid is None:
                        raise ValueError('Visual artifact requires a configured retinal encoder')
                    from simtoolreal_shared.retina import RetinalPopulationEncoder
                    self.sensory_adapter = RetinalPopulationEncoder(
                        self.sensory_adapter,
                        [sensory_positions[int(i)] for i in visual_indices],
                        visual_grid, **dict(retina))
                elif retina is not None:
                    raise ValueError('Retinal encoder requires visual neurons in the artifact')
                self.descending_adapter = _FixedPopulationEncoder(
                    policy_size=self.policy_observation_size,
                    output_size=len(descending_indices),
                    mappings=list(fixed_input["descending_mappings"]),
                )
            elif structured_input is None:
                self.sensory_adapter_mode = "dense"
                self.sensory_adapter = _interface_projection(
                    sensory_size,
                    len(sensory_indices),
                    self.projection_architecture,
                    self.projection_hidden_size,
                    self.projection_activation,
                    bias=input_bias,
                )
                descending_architecture = self.projection_architecture
                descending_hidden_size = self.projection_hidden_size
                descending_activation = self.projection_activation
            else:
                self.sensory_adapter_mode = str(
                    structured_input.get("mode", "grouped_linear")
                ).lower()
                if self.sensory_adapter_mode != "grouped_linear":
                    raise ValueError(
                        "structured_input_adapter.mode must be grouped_linear"
                    )
                assert proprioceptor_indices is not None
                sensory_positions = {
                    int(neuron): position
                    for position, neuron in enumerate(sensory_indices)
                }
                proprioceptor_positions = np.asarray(
                    [sensory_positions[int(neuron)] for neuron in proprioceptor_indices],
                    dtype=np.int64,
                )
                self.sensory_adapter = _GroupedSensoryAdapter(
                    input_size=sensory_size,
                    sensory_size=len(sensory_indices),
                    proprioceptor_positions=proprioceptor_positions,
                    groups=list(structured_input["groups"]),
                    dof_count=int(structured_input.get("dof_count", 29)),
                    fingertip_count=int(
                        structured_input.get("fingertip_count", 5)
                    ),
                    bias=input_bias,
                )
                descending_projection = structured_input.get(
                    "descending_projection", {}
                )
                descending_architecture = str(
                    descending_projection.get("architecture", "mlp")
                ).lower()
                descending_hidden_size = int(
                    descending_projection.get("hidden_size", 128)
                )
                if descending_hidden_size < 1:
                    raise ValueError(
                        "structured descending projection hidden_size must be positive"
                    )
                descending_activation = str(
                    descending_projection.get("activation", "elu")
                ).lower()
                _interface_activation(descending_activation)
            if fixed_input is None:
                self.descending_projection_architecture = descending_architecture
                self.descending_adapter = _interface_projection(
                    context_size + goal_size + self.coef_embedding_size,
                    len(descending_indices),
                    descending_architecture,
                    descending_hidden_size,
                    descending_activation,
                    bias=input_bias,
                )

            dynamics = connectome["dynamics"]
            self.neural_updates = dynamics.get("neural_updates", 1)
            if isinstance(self.neural_updates, bool) or not isinstance(self.neural_updates, int):
                raise TypeError("dynamics.neural_updates must be an integer")
            if self.neural_updates < 1:
                raise ValueError("dynamics.neural_updates must be positive")
            self.dynamics_activation = str(
                dynamics.get("activation", "tanh")
            ).lower()
            if self.dynamics_activation not in {"tanh", "lif"}:
                raise ValueError(
                    "Connectome recurrent activation must be tanh or lif"
                )
            if self.dynamics_activation == "lif":
                if not self.cache_reservoir_features:
                    raise ValueError(
                        "LIF dynamics require the fixed reservoir_readout path so "
                        "the hard spikes are never differentiated through"
                    )

                def positive_finite(name: str, default: float) -> float:
                    value = float(dynamics.get(name, default))
                    if not math.isfinite(value) or value <= 0.0:
                        raise ValueError(
                            f"dynamics.{name} must be finite and positive"
                        )
                    return value

                self.control_frequency_hz = positive_finite(
                    "control_frequency_hz", 60.0
                )
                self.membrane_time_constant_ms = positive_finite(
                    "membrane_time_constant_ms", 10.0
                )
                self.spike_threshold = positive_finite("spike_threshold", 1.0)
                self.refractory_period_ms = positive_finite(
                    "refractory_period_ms", 2.0
                )
                self.input_current_scale = positive_finite(
                    "input_current_scale", 1.5
                )
                self.lif_timestep_ms = 1000.0 / (
                    self.control_frequency_hz * self.neural_updates
                )
                self.membrane_decay = math.exp(
                    -self.lif_timestep_ms / self.membrane_time_constant_ms
                )
            else:
                self.control_frequency_hz = None
                self.membrane_time_constant_ms = None
                self.spike_threshold = None
                self.refractory_period_ms = None
                self.input_current_scale = None
                self.lif_timestep_ms = None
                self.membrane_decay = None
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

            self.reservoir_feature_count = len(motor_indices)
            self.readout_input_size = self.reservoir_feature_count
            readout_architecture = self.projection_architecture
            readout_hidden_size = self.projection_hidden_size
            readout_activation = self.projection_activation
            if self.cache_reservoir_features:
                feature_population = str(
                    reservoir_readout.get("feature_population", "motor")
                ).lower()
                if feature_population != "motor":
                    raise ValueError(
                        "reservoir_readout.feature_population must be motor"
                    )
                if bool(reservoir_readout.get("condition_on_sapg", True)) and (
                    self.net_type == "extra_param"
                ):
                    self.readout_input_size += self.coef_embedding_size
                    self.readout_conditions_on_sapg = True
                else:
                    self.readout_conditions_on_sapg = False
                readout_architecture = str(
                    reservoir_readout.get("architecture", "mlp")
                ).lower()
                readout_hidden_size = int(
                    reservoir_readout.get("hidden_size", 128)
                )
                if readout_hidden_size < 1:
                    raise ValueError("reservoir readout hidden_size must be positive")
                readout_activation = str(
                    reservoir_readout.get("activation", "elu")
                ).lower()
                _interface_activation(readout_activation)
            else:
                self.readout_conditions_on_sapg = False
            self.mu = _interface_projection(
                self.readout_input_size,
                self.actions_num,
                readout_architecture,
                readout_hidden_size,
                readout_activation,
                bias=True,
            )
            value_input_size = (
                self.readout_input_size
                if self.cache_reservoir_features
                else self.neuron_count
            )
            self.value = nn.Linear(value_input_size, self.value_size)
            continuous = params["space"]["continuous"]
            self.action_distribution = continuous.get("distribution", "gaussian")
            if self.action_distribution not in {"gaussian", "beta"}:
                raise ValueError("continuous.distribution must be gaussian or beta")
            self.mu_act = self.activations_factory.create(continuous["mu_activation"])
            self.sigma_act = self.activations_factory.create(
                continuous["sigma_activation"]
            )
            self.fixed_sigma = continuous["fixed_sigma"]
            configured_max_sigma = continuous.get("max_sigma")
            self.max_sigma = (
                None if configured_max_sigma is None else float(configured_max_sigma)
            )
            if self.max_sigma is not None:
                if self.action_distribution != "gaussian":
                    raise ValueError("max_sigma is only supported for gaussian policies")
                if not math.isfinite(self.max_sigma) or self.max_sigma <= 1.0:
                    raise ValueError(
                        "max_sigma must be finite and greater than 1 to preserve "
                        "the unit-sigma initialization"
                    )
                max_log_sigma = math.log(self.max_sigma)
                # Shift the soft ceiling so a raw log standard deviation of zero
                # still produces sigma=1, matching existing Gaussian checkpoints.
                self.max_log_sigma = max_log_sigma
                self.max_sigma_offset = (
                    math.log(math.expm1(max_log_sigma)) - max_log_sigma
                )
            if self.action_distribution == "beta":
                self.beta_initial_shape = float(continuous.get("beta_initial_shape", 2.0))
                self.beta_min_shape = float(continuous.get("beta_min_shape", 1.0))
                if not math.isfinite(self.beta_min_shape) or self.beta_min_shape <= 0:
                    raise ValueError("beta_min_shape must be finite and positive")
                if (
                    not math.isfinite(self.beta_initial_shape)
                    or self.beta_initial_shape <= self.beta_min_shape
                ):
                    raise ValueError(
                        "beta_initial_shape must be finite and greater than beta_min_shape"
                    )
                self.beta_head = _interface_projection(
                    self.readout_input_size, self.actions_num,
                    readout_architecture, readout_hidden_size,
                    readout_activation, bias=True,
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
                target = self.beta_initial_shape - self.beta_min_shape
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
            refractory: torch.Tensor | None = None,
            spikes: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
                if self.sensory_adapter_mode == "fixed_population_code":
                    if self.dynamics_activation == "lif":
                        sensory_drive = self.sensory_adapter.spike_rates(observations)
                        descending_drive = self.descending_adapter.spike_rates(
                            observations
                        )
                    else:
                        sensory_drive = self.sensory_adapter(observations)
                        descending_drive = self.descending_adapter(observations)
                else:
                    sensory = _select_ranges(observations, self.sensory_ranges)
                    descending_inputs = []
                    if self.context_ranges:
                        descending_inputs.append(
                            _select_ranges(observations, self.context_ranges)
                        )
                    descending_inputs.append(
                        _select_ranges(observations, self.goal_ranges)
                    )
                    if self.net_type == "extra_param":
                        descending_inputs.append(
                            self.extra_params[self._coefficient_rows(observations)]
                        )
                    descending_input = torch.cat(descending_inputs, dim=-1)
                    sensory_drive = self.sensory_adapter(sensory)
                    descending_drive = self.descending_adapter(descending_input)
                incoming = self.incoming_gains()
                step_values = self.effective_values() if values is None else values
                outgoing = self.outgoing_gains()
                if self.dynamics_activation == "lif":
                    if refractory is None or spikes is None:
                        raise ValueError(
                            "LIF recurrence requires membrane, refractory, and spike state"
                        )
                    refractory = refractory.float()
                    spikes = spikes.float()
                    motor_rates = hidden.new_zeros(
                        (hidden.shape[0], self.reservoir_feature_count)
                    )
                    if self.operator_backend == "triton_fused":
                        from rl_games.algos_torch.connectome_triton import (
                            fused_lif_step,
                        )

                        graph = self.backend_graph()
                        for _ in range(self.neural_updates):
                            hidden, refractory, spikes = fused_lif_step(
                                graph,
                                step_values,
                                hidden,
                                refractory,
                                spikes,
                                incoming,
                                outgoing,
                                self.recurrent_bias,
                                sensory_drive,
                                descending_drive,
                                self.sensory_indices,
                                self.descending_indices,
                                self.recurrent_gain,
                                self.input_current_scale,
                                self.membrane_decay,
                                self.spike_threshold,
                                self.refractory_period_ms,
                                self.lif_timestep_ms,
                            )
                            motor_rates.add_(spikes.index_select(1, self.motor_indices))
                    else:
                        drive = hidden.new_zeros(hidden.shape)
                        drive = drive.index_add(
                            1, self.sensory_indices, sensory_drive
                        )
                        drive = drive.index_add(
                            1, self.descending_indices, descending_drive
                        )
                        for _ in range(self.neural_updates):
                            recurrent = self._recurrent_multiply(
                                spikes, operator, step_values
                            )
                            current = (
                                self.recurrent_gain * incoming * recurrent
                                + self.input_current_scale * drive
                                + self.recurrent_bias
                            )
                            remaining_refractory = torch.clamp_min(
                                refractory - self.lif_timestep_ms, 0.0
                            )
                            available = remaining_refractory <= 0.0
                            candidate = (
                                self.membrane_decay * hidden
                                + (1.0 - self.membrane_decay) * current
                            )
                            fired = (candidate >= self.spike_threshold) & available
                            hidden = torch.where(
                                available, candidate, torch.zeros_like(candidate)
                            )
                            hidden = torch.where(
                                fired, torch.zeros_like(hidden), hidden
                            )
                            refractory = torch.where(
                                fired,
                                torch.full_like(
                                    refractory, self.refractory_period_ms
                                ),
                                remaining_refractory,
                            )
                            spikes = fired.float()
                            motor_rates.add_(
                                spikes.index_select(1, self.motor_indices)
                            )
                    motor_rates.mul_(1.0 / self.neural_updates)
                    return hidden, refractory, spikes, motor_rates

                leak = self.substep_leaks()
                if self.operator_backend == "triton_fused":
                    from rl_games.algos_torch.connectome_triton import fused_step

                    graph = self.backend_graph()
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
            cached_features = obs_dict.get("reservoir_features")
            if cached_features is not None:
                if not self.cache_reservoir_features:
                    raise ValueError(
                        "reservoir_features were provided to a non-reservoir policy"
                    )
                if cached_features.ndim != 2 or tuple(cached_features.shape) != (
                    observations.shape[0],
                    self.reservoir_feature_count,
                ):
                    raise ValueError(
                        "Expected reservoir_features shape "
                        f"{(observations.shape[0], self.reservoir_feature_count)}, "
                        f"got {tuple(cached_features.shape)}"
                    )
                motor = cached_features.float()
                returned_states = ()
            else:
                sequence_length = int(obs_dict.get("seq_length", 1))
                if observations.shape[0] % sequence_length:
                    raise ValueError("Batch size must be divisible by seq_length")
                sequence_count = observations.shape[0] // sequence_length
                sequence = observations.reshape(
                    sequence_count, sequence_length, -1
                ).transpose(0, 1)

                states = obs_dict.get("rnn_states")
                if self.dynamics_activation == "lif":
                    if states is None:
                        hidden, refractory, spikes = (
                            observations.new_zeros(
                                (sequence_count, self.neuron_count),
                                dtype=torch.float32,
                            )
                            for _ in range(3)
                        )
                    else:
                        if not isinstance(states, (tuple, list)) or len(states) != 3:
                            raise ValueError(
                                "LIF recurrence requires three RNN states: "
                                "membrane, refractory, and spikes"
                            )
                        lif_states = []
                        for state in states:
                            if state.ndim == 3:
                                state = state[0]
                            lif_states.append(state.float())
                        hidden, refractory, spikes = lif_states
                else:
                    refractory = spikes = None
                    if states is None:
                        hidden = observations.new_zeros(
                            (sequence_count, self.neuron_count), dtype=torch.float32
                        )
                    else:
                        hidden = (
                            states[0] if isinstance(states, (tuple, list)) else states
                        )
                        if hidden.ndim == 3:
                            hidden = hidden[0]
                        hidden = hidden.float()
                dones = obs_dict.get("dones")
                if dones is not None:
                    dones = dones.reshape(
                        sequence_count, sequence_length, -1
                    ).transpose(0, 1)

                def run_reservoir() -> torch.Tensor:
                    outputs = []
                    values = self.effective_values()
                    operator = self.recurrent_matrix(values)
                    nonlocal hidden, refractory, spikes
                    for step, step_observations in enumerate(sequence):
                        if dones is not None:
                            active = 1.0 - dones[step].float()
                            hidden = hidden * active
                            if self.dynamics_activation == "lif":
                                assert refractory is not None and spikes is not None
                                refractory = refractory * active
                                spikes = spikes * active
                        if self.dynamics_activation == "lif":
                            result = self._step(
                                step_observations,
                                hidden,
                                operator,
                                values,
                                refractory,
                                spikes,
                            )
                            hidden, refractory, spikes, motor_rates = result
                            outputs.append(motor_rates)
                        else:
                            hidden = self._step(
                                step_observations, hidden, operator, values
                            )
                            outputs.append(hidden)
                    return torch.stack(outputs).transpose(0, 1).reshape(
                        observations.shape[0], -1
                    )

                if self.cache_reservoir_features:
                    with torch.no_grad():
                        output = run_reservoir()
                else:
                    output = run_reservoir()
                if self.dynamics_activation == "lif":
                    motor = output.float()
                    assert refractory is not None and spikes is not None
                    returned_states = (
                        hidden.unsqueeze(0),
                        refractory.unsqueeze(0),
                        spikes.unsqueeze(0),
                    )
                else:
                    motor = output[:, self.motor_indices].float()
                    returned_states = (hidden.unsqueeze(0),)

            if self.cache_reservoir_features:
                self.last_reservoir_features = motor.detach()
                readout_parts = [motor]
                if self.readout_conditions_on_sapg:
                    readout_parts.append(
                        self.extra_params[self._coefficient_rows(observations)]
                    )
                readout = torch.cat(readout_parts, dim=-1)
                value = self.value(readout)
            else:
                readout = motor
                value = self.value(output)
            if self.action_distribution == "beta":
                # Shape heads and special-function inputs stay FP32 under AMP.
                with torch.autocast(device_type=readout.device.type, enabled=False):
                    alpha = self.beta_min_shape + torch.nn.functional.softplus(self.mu(readout))
                    beta = self.beta_min_shape + torch.nn.functional.softplus(self.beta_head(readout))
                return alpha, beta, value, returned_states
            mu = self.mu_act(self.mu(readout))
            if self.fixed_sigma == "coef_cond":
                sigma = self.sigma_act(self.sigma[self._coefficient_rows(observations)])
            else:
                sigma = self.sigma_act(self.sigma).expand_as(mu)
            if self.max_sigma is not None:
                sigma = self.max_log_sigma - torch.nn.functional.softplus(
                    self.max_log_sigma - sigma + self.max_sigma_offset
                )
            return mu, sigma, value, returned_states

        def uses_cached_reservoir_features(self) -> bool:
            return self.cache_reservoir_features

        def get_reservoir_feature_count(self) -> int:
            return self.reservoir_feature_count

        def is_rnn(self) -> bool:
            return True

        def get_default_rnn_state(self):
            count = 3 if self.dynamics_activation == "lif" else 1
            return tuple(
                torch.zeros(
                    (1, self.num_seqs, self.neuron_count), dtype=torch.float32
                )
                for _ in range(count)
            )

        def get_value_layer(self):
            return self.value

        def is_separate_critic(self) -> bool:
            return False
