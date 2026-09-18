#!/usr/bin/env python3
"""Profile LSTM and connectome actors from one YAML contract."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return result


def _compose_network(train_profile: str, task_profile: str) -> dict[str, Any]:
    import isaacgymenvs  # noqa: F401 - registers OmegaConf resolvers

    config_directory = REPOSITORY_ROOT / "isaacgymenvs" / "cfg"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_directory)):
        config = compose(
            config_name="config",
            overrides=[f"task={task_profile}", f"train={train_profile}"],
        )
    return OmegaConf.to_container(config.train.params.network, resolve=True)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_case(
    actor: dict[str, Any],
    shape: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    from rl_games.algos_torch import model_builder

    # Reset per case: backend comparisons receive identical parameters and data.
    torch.manual_seed(int(config["seed"]))
    _synchronize(device)
    setup_started = time.perf_counter()
    network_params = _compose_network(actor["train_profile"], config["task_profile"])
    if "operator_backend" in actor:
        network_params["connectome"]["operator_backend"] = actor["operator_backend"]
    for key in ("adaptation", "backend_options"):
        if key in actor:
            network_params["connectome"][key] = {
                **network_params["connectome"].get(key, {}),
                **actor[key],
            }
    builder = model_builder.NetworkBuilder().load(network_params)
    sequence_count = int(shape["num_sequences"])
    sequence_length = int(shape["sequence_length"])
    coefficient_ids = torch.tensor(config["sapg_coefficient_ids"], device=device)
    network = builder.build(
        "profile",
        actions_num=int(config["actions"]),
        # rl_games receives the scalar coefficient as the extra observation;
        # each network builder expands it into its learned embedding internally.
        input_shape=(int(config["policy_observations"]) + 1,),
        num_seqs=sequence_count,
        value_size=1,
        type="extra_param",
        coef_ids=coefficient_ids,
        coef_id_idx=int(config["policy_observations"]),
        param_size=int(config["sapg_embedding_size"]),
    ).to(device)
    network.train(bool(shape["backward"]))
    trainable_parameters = sum(
        parameter.numel()
        for parameter in network.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in network.parameters())
    frozen_parameters = total_parameters - trainable_parameters
    fixed_recurrent_weights = (
        int(network.recurrent_values.numel())
        if hasattr(network, "recurrent_values")
        else 0
    )
    structural_index_elements = sum(
        int(getattr(network, name).numel())
        for name in ("crow_indices", "col_indices")
        if hasattr(network, name)
    )
    instantiated_actor_coefficients = total_parameters + fixed_recurrent_weights
    dense_possible_recurrent_weights = (
        int(network.neuron_count) ** 2 if hasattr(network, "neuron_count") else None
    )
    expected_coefficients = actor.get("expected_instantiated_actor_coefficients")
    if (
        expected_coefficients is not None
        and instantiated_actor_coefficients != int(expected_coefficients)
    ):
        raise RuntimeError(
            f"{actor['name']} instantiated actor coefficient count "
            f"{instantiated_actor_coefficients} != expected {expected_coefficients}"
        )
    state_hash = hashlib.sha256()
    for name, tensor in network.state_dict().items():
        state_hash.update(name.encode())
        state_hash.update(tensor.detach().cpu().numpy().tobytes())

    torch.manual_seed(int(config["seed"]) + 1)
    batch = sequence_count * sequence_length
    observations = torch.randn(
        batch, int(config["policy_observations"]) + 1, device=device
    )
    sequence_coefficients = coefficient_ids[
        torch.arange(sequence_count, device=device) % len(coefficient_ids)
    ]
    observations[:, int(config["policy_observations"])] = (
        sequence_coefficients.repeat_interleave(sequence_length)
    )
    dones = torch.zeros(batch, 1, device=device)
    states = tuple(
        torch.randn_like(state, device=device) * 0.1
        for state in network.get_default_rnn_state()
    )
    input_dict = {
        "obs": observations,
        "rnn_states": states,
        # Legacy LSTM reset code squeezes a 1x1 mask to a scalar. No resets
        # occur in this synthetic case, so use its supported no-mask path.
        "dones": None if batch == 1 else dones,
        "seq_length": sequence_length,
    }
    use_amp = bool(config["amp"]) and device.type == "cuda"
    optimizer = torch.optim.Adam(
        network.parameters(), lr=float(config.get("optimizer_learning_rate", 1e-4))
    )
    do_step = bool(shape.get("optimizer_step", False))
    if do_step and not shape["backward"]:
        raise ValueError("optimizer_step requires backward")

    def forward_pass() -> torch.Tensor:
        network.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            mu, logstd, value, _ = network(input_dict)
            return (
                mu.float().square().mean()
                + logstd.float().square().mean()
                + value.float().square().mean()
            )

    # Cold pass includes graph planning, extension loading and JIT compilation.
    if shape["backward"]:
        forward_pass().backward()
        if do_step:
            optimizer.step()
    else:
        with torch.no_grad():
            forward_pass()
    _synchronize(device)
    setup_seconds = time.perf_counter() - setup_started
    for _ in range(int(config["warmup_iterations"])):
        if shape["backward"]:
            forward_pass().backward()
            if do_step:
                optimizer.step()
        else:
            with torch.no_grad():
                forward_pass()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_latencies = []
    backward_latencies = []
    total_latencies = []
    optimizer_latencies = []
    for _ in range(
        int(config["measure_iterations"]) * int(config.get("repetitions", 1))
    ):
        network.zero_grad(set_to_none=True)
        _synchronize(device)
        started = time.perf_counter()
        if shape["backward"]:
            loss = forward_pass()
            _synchronize(device)
            forward_finished = time.perf_counter()
            loss.backward()
            _synchronize(device)
            backward_finished = time.perf_counter()
            if do_step:
                optimizer.step()
                _synchronize(device)
            finished = time.perf_counter()
            forward_latencies.append(forward_finished - started)
            backward_latencies.append(backward_finished - forward_finished)
            optimizer_latencies.append(finished - backward_finished)
        else:
            with torch.no_grad():
                forward_pass()
            _synchronize(device)
            finished = time.perf_counter()
            forward_latencies.append(finished - started)
        total_latencies.append(finished - started)

    median_forward = statistics.median(forward_latencies)
    median_backward = (
        statistics.median(backward_latencies) if backward_latencies else None
    )
    median_total = statistics.median(total_latencies)
    # Capture benchmark peak before the untimed diagnostic forward.
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    diagnostics = {}
    if hasattr(network, "effective_values"):
        with torch.no_grad():
            weights = network.effective_values()
            if network.weight_mode == "neuron_gains":
                graph = network.backend_graph()
                weights = (
                    weights
                    * network.incoming_gains()[graph.rows]
                    * network.outgoing_gains()[graph.col]
                )
            ratio = weights / network.recurrent_values
            # Saturation of tanh, reconstructed from the public leaky step.
            h0 = states[0][0]
            h1 = network._step(observations[:sequence_count], h0)
            activated = (h1 - (1 - network.leaks()) * h0) / network.leaks()
            diagnostics = {
                "edge_multiplier_min": float(ratio.min()),
                "edge_multiplier_max": float(ratio.max()),
                "edge_log_drift_rms": float(ratio.log().square().mean().sqrt()),
                "activation_saturation_fraction": float(
                    (activated.abs() > 0.99).float().mean()
                ),
            }
    return {
        "status": "complete",
        "actor": actor["name"],
        "train_profile": actor["train_profile"],
        "operator_backend": actor.get("operator_backend"),
        "adaptation": network_params.get("connectome", {}).get("adaptation"),
        "backend_options": actor.get("backend_options", {}),
        "initial_state_sha256": state_hash.hexdigest(),
        "setup_and_cold_pass_ms": setup_seconds * 1000,
        "optimizer_step": do_step,
        "median_optimizer_latency_ms": statistics.median(optimizer_latencies) * 1000
        if do_step
        else None,
        "median_total_latency_ms": median_total * 1000,
        "total_latency_samples_ms": [x * 1000 for x in total_latencies],
        "diagnostics": diagnostics,
        "shape": shape["name"],
        "num_sequences": sequence_count,
        "sequence_length": sequence_length,
        "batch_observations": batch,
        "backward": bool(shape["backward"]),
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
        "total_parameters": total_parameters,
        "fixed_recurrent_weights": fixed_recurrent_weights,
        "instantiated_actor_coefficients": instantiated_actor_coefficients,
        "structural_index_elements_excluded": structural_index_elements,
        "dense_possible_recurrent_weights_excluded": dense_possible_recurrent_weights,
        "median_forward_latency_ms": median_forward * 1000.0,
        "median_backward_latency_ms": (
            median_backward * 1000.0 if median_backward is not None else None
        ),
        "median_forward_backward_latency_ms": statistics.median(
            [f + b for f, b in zip(forward_latencies, backward_latencies)]
        )
        * 1000
        if backward_latencies
        else median_total * 1000,
        "throughput_observations_per_second": batch / median_total,
        "peak_gpu_memory_bytes": peak_memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = (
        args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    )
    config = _load_yaml(config_path)
    for key, value in config.get("runtime_environment", {}).items():
        os.environ[key] = str(value)

    vendored_rl_games = str(REPOSITORY_ROOT / "rl_games")
    if sys.path[0] != vendored_rl_games:
        sys.path.insert(0, vendored_rl_games)
    torch.manual_seed(int(config["seed"]))
    requested_device = config["device"]
    if requested_device == "auto":
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling was requested but CUDA is unavailable")

    results = []
    output_path = Path(config["output_path"])
    if not output_path.is_absolute():
        output_path = REPOSITORY_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 2,
        "status": "running",
        "config": str(config_path),
        "resolved_config": config,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "results": results,
    }
    for actor in config["actors"]:
        for shape in config["shapes"]:
            print(f"Profiling {actor['name']} / {shape['name']}", flush=True)
            try:
                results.append(_profile_case(actor, shape, config, device))
            except torch.cuda.OutOfMemoryError:
                if not shape.get("allow_oom", False):
                    raise
                results.append(
                    {
                        "actor": actor["name"],
                        "shape": shape["name"],
                        "status": "out_of_memory",
                        "num_sequences": shape["num_sequences"],
                        "sequence_length": shape["sequence_length"],
                    }
                )
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report["status"] = "complete"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
