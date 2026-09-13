#!/usr/bin/env python3
"""Run preparation, profiling, and/or training from a YAML suite contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _run_streaming(
    command: list[str], environment: dict[str, str], log_path: Path
) -> None:
    print("Running:", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _environment(gpu: int | str | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    python_path = [str(REPOSITORY_ROOT / "rl_games"), str(REPOSITORY_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def _compose_resolved(overrides: list[str]):
    import isaacgymenvs  # noqa: F401 - registers OmegaConf resolvers

    config_directory = REPOSITORY_ROOT / "isaacgymenvs" / "cfg"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_directory)):
        return compose(config_name="config", overrides=overrides)


def _verify_checkpoint(
    checkpoint_path: Path,
    resolved_config_path: Path,
    device: str,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 0 in checkpoint:
        state = checkpoint[0]
    else:
        state = checkpoint
    optimizer_state = state.get("optimizer", {}).get("state", {})
    if not optimizer_state:
        raise RuntimeError(
            "Checkpoint has no optimizer state; no completed update was recorded"
        )

    from deployment.rl_player import RlPlayer

    player = RlPlayer(
        num_observations=140,
        num_actions=29,
        config_path=str(resolved_config_path),
        checkpoint_path=str(checkpoint_path),
        device=device,
        num_envs=1,
    )
    observation = torch.zeros((1, 140), device=device)
    action = player.get_normalized_action(observation, deterministic_actions=True)
    if action.shape != (1, 29) or not torch.isfinite(action).all():
        raise RuntimeError(f"Invalid deployment action after reload: {action.shape}")
    return {
        "checkpoint": str(checkpoint_path),
        "optimizer_state_entries": len(optimizer_state),
        "deployment_action_shape": list(action.shape),
        "deployment_action_finite": True,
    }


def _training_overrides(
    training: dict[str, Any],
    profile: str,
    seed: int,
    run_name: str,
    run_directory: Path,
) -> list[str]:
    overrides = [
        f"task={training['task_profile']}",
        f"train={profile}",
        "headless=true",
        "multi_gpu=false",
        "sim_device=cuda:0",
        "rl_device=cuda:0",
        f"task.env.numEnvs={int(training['num_envs'])}",
        f"train.params.config.expl_coef_block_size={int(training['sapg_block_size'])}",
        f"train.params.config.max_epochs={int(training['epochs'])}",
        f"train.params.config.minibatch_size={int(training['minibatch_size'])}",
        "train.params.config.central_value_config.minibatch_size="
        f"{int(training['central_critic_minibatch_size'])}",
        f"++train.params.config.train_dir={run_directory / 'rl_runs'}",
        f"train.params.config.save_frequency={int(training['save_frequency'])}",
        f"train.params.config.save_best_after={int(training['save_best_after'])}",
        f"seed={seed}",
        f"experiment={run_name}",
        f"hydra.run.dir={run_directory / 'hydra'}",
        f"wandb_activate={str(bool(training['wandb']['enabled'])).lower()}",
        f"wandb_project={training['wandb']['project']}",
        f"wandb_entity={training['wandb']['entity']}",
        f"wandb_group={training['wandb']['group']}",
        f"wandb_tags={_hydra_value(training['wandb']['tags'])}",
    ]
    for key, value in training.get("overrides", {}).items():
        overrides.append(f"{key}={_hydra_value(value)}")
    checkpoint = training["checkpoint"]
    if checkpoint["mode"] != "none":
        overrides.append(f"checkpoint={checkpoint['path']}")
        overrides.append(
            f"++train.params.config.checkpoint_load_mode={checkpoint['mode']}"
        )
    return overrides


def _run_training(
    config: dict[str, Any], suite_directory: Path
) -> list[dict[str, Any]]:
    training = config["training"]
    if int(training["num_envs"]) // int(training["sapg_block_size"]) != 6:
        raise ValueError(
            "The legacy SAPG player requires exactly six exploration blocks"
        )
    results = []
    profiles = training["train_profiles"]
    seeds = [int(seed) for seed in training["seeds"]]
    gpus = training["gpu_assignments"]
    for run_index, (profile, seed) in enumerate(
        (pair for profile in profiles for pair in [(profile, seed) for seed in seeds])
    ):
        gpu = gpus[run_index % len(gpus)]
        # Legacy SAPG parses the leading numeric token as a fallback coefficient.
        run_name = f"00_{config['name']}_{profile}_seed{seed}"
        run_directory = suite_directory / run_name
        if run_directory.exists() and any(run_directory.iterdir()):
            if training["on_existing"] == "skip":
                print(f"Skipping existing run {run_directory}")
                continue
            raise FileExistsError(
                f"Run directory already exists: {run_directory}. "
                "Change output_directory or set on_existing: skip."
            )
        run_directory.mkdir(parents=True, exist_ok=True)
        overrides = _training_overrides(
            training, profile, seed, run_name, run_directory.resolve()
        )
        resolved = _compose_resolved(overrides)
        resolved_config_path = run_directory / "resolved_config.yaml"
        resolved_config_path.write_text(OmegaConf.to_yaml(resolved, resolve=True))
        command = [sys.executable, "-m", "isaacgymenvs.train", *overrides]
        environment = _environment(gpu)
        _run_streaming(command, environment, run_directory / "train.log")

        checkpoint_directory = run_directory / "rl_runs" / run_name / "nn"
        checkpoints = sorted(
            checkpoint_directory.glob("*.pth"),
            key=lambda path: path.stat().st_mtime,
        )
        if not checkpoints:
            raise RuntimeError(
                f"Training exited without a checkpoint in {checkpoint_directory}"
            )
        verification_device = f"cuda:{gpu}" if isinstance(gpu, int) else "cuda:0"
        verification = _verify_checkpoint(
            checkpoints[-1], resolved_config_path, verification_device
        )
        (run_directory / "verification.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n"
        )
        results.append({"profile": profile, "seed": seed, **verification})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = (
        args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    )
    config = _load_yaml(config_path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported suite schema_version")

    output_directory = Path(config["output_directory"])
    if not output_directory.is_absolute():
        output_directory = REPOSITORY_ROOT / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"suite": config["name"], "stages": {}}
    for stage in config["stages"]:
        if stage == "prepare":
            prepare_config = config["preparation"]["config"]
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "prepare_malecns_connectome.py"),
                "--config",
                str(prepare_config),
            ]
            _run_streaming(command, _environment(), output_directory / "prepare.log")
            results["stages"][stage] = "complete"
        elif stage == "profile":
            profile_config = config["profiling"]["config"]
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "profile_connectome_actors.py"),
                "--config",
                str(profile_config),
            ]
            _run_streaming(
                command,
                _environment(config["profiling"].get("gpu")),
                output_directory / "profile.log",
            )
            results["stages"][stage] = "complete"
        elif stage == "train":
            results["stages"][stage] = _run_training(config, output_directory)
        else:
            raise ValueError(f"Unknown suite stage: {stage}")
    (output_directory / "suite_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
