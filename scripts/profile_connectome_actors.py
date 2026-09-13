#!/usr/bin/env python3
"""Profile LSTM and connectome actors from one YAML contract."""

from __future__ import annotations

import argparse
import json
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
        raise ValueError(f"Expected a YAML mapping in {path}")
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

    network_params = _compose_network(actor["train_profile"], config["task_profile"])
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
    states = tuple(state.to(device) for state in network.get_default_rnn_state())
    input_dict = {
        "obs": observations,
        "rnn_states": states,
        "dones": dones,
        "seq_length": sequence_length,
    }
    use_amp = bool(config["amp"]) and device.type == "cuda"

    def forward_pass() -> torch.Tensor:
        network.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            mu, logstd, value, _ = network(input_dict)
            return mu.square().mean() + logstd.square().mean() + value.square().mean()

    for _ in range(int(config["warmup_iterations"])):
        if shape["backward"]:
            forward_pass().backward()
        else:
            with torch.no_grad():
                forward_pass()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_latencies = []
    backward_latencies = []
    total_latencies = []
    for _ in range(int(config["measure_iterations"])):
        network.zero_grad(set_to_none=True)
        _synchronize(device)
        started = time.perf_counter()
        if shape["backward"]:
            loss = forward_pass()
            _synchronize(device)
            forward_finished = time.perf_counter()
            loss.backward()
            _synchronize(device)
            finished = time.perf_counter()
            forward_latencies.append(forward_finished - started)
            backward_latencies.append(finished - forward_finished)
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
    return {
        "actor": actor["name"],
        "train_profile": actor["train_profile"],
        "shape": shape["name"],
        "num_sequences": sequence_count,
        "sequence_length": sequence_length,
        "batch_observations": batch,
        "backward": bool(shape["backward"]),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in network.parameters()
            if parameter.requires_grad
        ),
        "total_parameters": sum(
            parameter.numel() for parameter in network.parameters()
        ),
        "median_forward_latency_ms": median_forward * 1000.0,
        "median_backward_latency_ms": (
            median_backward * 1000.0 if median_backward is not None else None
        ),
        "median_forward_backward_latency_ms": median_total * 1000.0,
        "throughput_observations_per_second": batch / median_total,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = (
        args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    )
    config = _load_yaml(config_path)

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
    for actor in config["actors"]:
        for shape in config["shapes"]:
            results.append(_profile_case(actor, shape, config, device))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report = {
        "schema_version": 1,
        "config": str(config_path),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "results": results,
    }
    output_path = Path(config["output_path"])
    if not output_path.is_absolute():
        output_path = REPOSITORY_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
