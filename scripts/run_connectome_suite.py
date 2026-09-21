#!/usr/bin/env python3
"""Run preparation, profiling, and/or training from a YAML suite contract."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TENSORBOARD_6008_LOG_ROOT = Path(
    "train_dir/connectome/adaptation_100b_gains_update_timing"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
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
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    label: str | None = None,
) -> None:
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Running: {' '.join(command)}", flush=True)
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
            print(f"{prefix}{line}", end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _environment(
    gpu: int | str | list[int] | tuple[int, ...] | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    python_path = [str(REPOSITORY_ROOT / "rl_games"), str(REPOSITORY_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    if gpu is not None:
        if isinstance(gpu, (list, tuple)):
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in gpu)
        else:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def _register_tensorboard_run(
    training: dict[str, Any],
    run_name: str,
    summaries_directory: Path,
) -> Path:
    """Expose one suite case under the log root served by TensorBoard port 6008.

    A suite may override ``training.tensorboard_log_root`` in YAML, but every
    suite defaults to the long-lived port-6008 aggregation root.  The stable,
    suite-qualified run name prevents two different cases from silently sharing
    one TensorBoard entry.
    """
    configured_root = Path(
        training.get("tensorboard_log_root", TENSORBOARD_6008_LOG_ROOT)
    )
    log_root = (
        configured_root
        if configured_root.is_absolute()
        else REPOSITORY_ROOT / configured_root
    )
    log_root.mkdir(parents=True, exist_ok=True)
    link_path = log_root / run_name
    expected_target = summaries_directory.resolve(strict=False)
    if os.path.lexists(link_path):
        if not link_path.is_symlink():
            raise RuntimeError(
                f"TensorBoard registration path is not a symlink: {link_path}"
            )
        if link_path.resolve(strict=False) != expected_target:
            raise RuntimeError(
                "TensorBoard registration collision: "
                f"{link_path} targets {link_path.resolve(strict=False)}, expected "
                f"{expected_target}"
            )
        return link_path
    link_path.symlink_to(expected_target, target_is_directory=True)
    return link_path


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
    resolved = OmegaConf.load(resolved_config_path)
    algorithm = str(OmegaConf.select(resolved, "train.params.algo.name", default="a2c_continuous"))
    eligibility = algorithm == "connectome_eligibility"
    trainer = state.get("trainer_state", {})
    eligibility_updates = int(trainer.get("eligibility", {}).get("updates", 0))
    if eligibility and (trainer.get("algorithm") != "connectome_eligibility" or eligibility_updates <= 0
                        or not trainer.get("critic_optimizer", {}).get("state", {})):
        raise RuntimeError("Checkpoint has no completed eligibility/critic updates")
    if not eligibility and not optimizer_state:
        raise RuntimeError(
            "Checkpoint has no optimizer state; no completed update was recorded"
        )
    has_central_value = OmegaConf.select(
        resolved,
        "train.params.config.central_value_config",
        default=None,
    ) is not None
    critic_optimizer_entries = 0
    critic_training_state = {}
    critic_training_summary = {}
    if not eligibility and has_central_value:
        missing_critic_state = []
        if not state.get("assymetric_vf_nets"):
            missing_critic_state.append("assymetric_vf_nets")
        critic_optimizer_entries = len(
            state.get("central_value_optimizer", {}).get("state", {})
        )
        if not critic_optimizer_entries:
            missing_critic_state.append("central_value_optimizer")
        critic_training_state = state.get("central_value_training_state", {})
        for key in ("epoch", "frame", "lr"):
            if key not in critic_training_state:
                missing_critic_state.append(f"central_value_training_state.{key}")
        if missing_critic_state:
            raise RuntimeError(
                "Checkpoint is incomplete for central-critic training resume; "
                f"missing {missing_critic_state}"
            )
        if (
            int(critic_training_state["epoch"]) < 0
            or int(critic_training_state["frame"]) < 0
            or not math.isfinite(float(critic_training_state["lr"]))
            or float(critic_training_state["lr"]) <= 0
        ):
            raise RuntimeError(
                "Checkpoint has invalid central-critic training metadata: "
                f"{critic_training_state}"
            )
        critic_rnn_states = critic_training_state.get("rnn_states")
        critic_training_summary = {
            "epoch": int(critic_training_state["epoch"]),
            "frame": int(critic_training_state["frame"]),
            "lr": float(critic_training_state["lr"]),
            "recurrent_state_tensors": (
                0 if critic_rnn_states is None else len(critic_rnn_states)
            ),
        }
    expected_epoch = int(resolved.train.params.config.max_epochs)
    requested_max_frames = int(resolved.train.params.config.get("max_frames", -1))
    checkpoint_epoch = int(state.get("epoch", -1))
    checkpoint_frame = int(state.get("frame", -1))
    if checkpoint_epoch < expected_epoch:
        raise RuntimeError(
            f"Checkpoint epoch {checkpoint_epoch} is below requested {expected_epoch}"
        )
    if requested_max_frames >= 0 and checkpoint_frame > requested_max_frames:
        raise RuntimeError(
            f"Checkpoint frame {checkpoint_frame} exceeds cap {requested_max_frames}"
        )

    from deployment.rl_player import RlPlayer

    num_observations = int(
        OmegaConf.select(
            resolved,
            "train.params.network.connectome.observations.policy_size",
            default=140,
        )
    )
    player = RlPlayer(
        num_observations=num_observations,
        num_actions=29,
        config_path=str(resolved_config_path),
        checkpoint_path=str(checkpoint_path),
        device=device,
        num_envs=1,
    )
    observation = torch.zeros((1, num_observations), device=device)
    action = player.get_normalized_action(observation, deterministic_actions=True)
    if action.shape != (1, 29) or not torch.isfinite(action).all():
        raise RuntimeError(f"Invalid deployment action after reload: {action.shape}")
    return {
        "checkpoint": str(checkpoint_path),
        "optimizer_state_entries": len(optimizer_state),
        "central_critic_optimizer_state_entries": critic_optimizer_entries,
        "central_critic_training_state": critic_training_summary,
        "eligibility_updates": eligibility_updates,
        "algorithm": algorithm,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_frame": checkpoint_frame,
        "requested_epochs": expected_epoch,
        "requested_max_frames": requested_max_frames,
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
    artifact_target = training.get("artifact_target", {})
    train_directory = Path(
        artifact_target.get("train_directory", run_directory / "rl_runs")
    )
    hydra_directory = Path(
        artifact_target.get("hydra_directory", run_directory / "hydra")
    )
    experiment_name = str(artifact_target.get("experiment_name", run_name))
    if not train_directory.is_absolute():
        train_directory = REPOSITORY_ROOT / train_directory
    if not hydra_directory.is_absolute():
        hydra_directory = REPOSITORY_ROOT / hydra_directory
    overrides = [
        f"task={training['task_profile']}",
        f"train={profile}",
        "headless=true",
        f"multi_gpu={str(bool(training.get('distributed', False))).lower()}",
        "sim_device=cuda:0",
        "rl_device=cuda:0",
        f"task.env.numEnvs={int(training['num_envs'])}",
        f"train.params.config.expl_coef_block_size={int(training['sapg_block_size'])}",
        f"train.params.config.max_epochs={int(training['epochs'])}",
        f"++train.params.config.train_dir={train_directory.resolve()}",
        f"train.params.config.save_frequency={int(training['save_frequency'])}",
        f"train.params.config.save_best_after={int(training['save_best_after'])}",
        f"seed={seed}",
        f"experiment={experiment_name}",
        f"hydra.run.dir={hydra_directory.resolve()}",
        f"wandb_activate={str(bool(training['wandb']['enabled'])).lower()}",
        f"wandb_project={training['wandb']['project']}",
        f"wandb_entity={training['wandb']['entity']}",
        f"wandb_group={training['wandb']['group']}",
        f"wandb_tags={_hydra_value(training['wandb']['tags'])}",
    ]
    if training.get("algorithm", "a2c_continuous") != "connectome_eligibility":
        overrides.extend([
            f"train.params.config.minibatch_size={int(training['minibatch_size'])}",
            "train.params.config.central_value_config.minibatch_size="
            f"{int(training['central_critic_minibatch_size'])}",
        ])
    if "max_frames" in training:
        overrides.append(
            f"train.params.config.max_frames={int(training['max_frames'])}"
        )
    if "inference_checkpoint_interval_frames" in training:
        overrides.append(
            "++train.params.config.inference_checkpoint_interval_frames="
            f"{int(training['inference_checkpoint_interval_frames'])}"
        )
    if "rollout_accumulation_steps" in training:
        overrides.append(
            "++train.params.config.rollout_accumulation_steps="
            f"{int(training['rollout_accumulation_steps'])}"
        )
    if "actor_microbatch_size" in training:
        overrides.append(
            "++train.params.config.microbatch_size="
            f"{int(training['actor_microbatch_size'])}"
        )
    if "central_critic_microbatch_size" in training:
        overrides.append(
            "++train.params.config.central_value_config.microbatch_size="
            f"{int(training['central_critic_microbatch_size'])}"
        )
    for key, value in training.get("overrides", {}).items():
        overrides.append(f"{key}={_hydra_value(value)}")
    checkpoint = training["checkpoint"]
    if checkpoint["mode"] != "none":
        overrides.append(f"checkpoint={checkpoint['path']}")
        overrides.append(
            f"++train.params.config.checkpoint_load_mode={checkpoint['mode']}"
        )
    return overrides


def _run_training_case(case: dict[str, Any]) -> dict[str, Any]:
    training = case["training"]
    profile = case["profile"]
    case_name = case["case_name"]
    seed = case["seed"]
    gpu = case["gpu"]
    gpus = case.get("gpus", [gpu])
    run_name = case["run_name"]
    run_directory = case["run_directory"]
    overrides = case["overrides"]
    resolved = case["resolved"]
    resolved_config_path = run_directory / "resolved_config.yaml"
    verification_device = f"cuda:{gpu}" if gpu is not None else "cpu"
    artifact_target = training.get("artifact_target", {})
    train_directory = Path(
        artifact_target.get("train_directory", run_directory / "rl_runs")
    )
    if not train_directory.is_absolute():
        train_directory = REPOSITORY_ROOT / train_directory
    experiment_name = str(artifact_target.get("experiment_name", run_name))
    summaries_directory = train_directory / experiment_name / "summaries"
    training_log = Path(
        artifact_target.get("training_log", run_directory / "train.log")
    )
    if not training_log.is_absolute():
        training_log = REPOSITORY_ROOT / training_log

    if run_directory.exists() and any(run_directory.iterdir()):
        _register_tensorboard_run(training, run_name, summaries_directory)
        if training["on_existing"] != "skip":
            raise FileExistsError(
                f"Run directory already exists: {run_directory}. "
                "Change output_directory or set on_existing: skip."
            )
        if not resolved_config_path.exists() or OmegaConf.to_container(
            OmegaConf.load(resolved_config_path), resolve=True
        ) != OmegaConf.to_container(resolved, resolve=True):
            raise RuntimeError(
                f"Cannot resume mismatched configuration: {run_directory}"
            )
        checkpoints = sorted(
            (run_directory / "rl_runs" / run_name / "nn").glob("*.pth"),
            key=lambda path: path.stat().st_mtime,
        )
        if not checkpoints:
            raise RuntimeError(
                f"Existing run has no checkpoint to verify: {run_directory}"
            )
        verification = _verify_checkpoint(
            checkpoints[-1], resolved_config_path, verification_device
        )
        return {
            "profile": profile,
            "case": case_name,
            "seed": seed,
            "gpu": gpu,
            "reverified": True,
            **verification,
        }

    run_directory.mkdir(parents=True, exist_ok=True)
    tensorboard_link = _register_tensorboard_run(
        training, run_name, summaries_directory
    )
    print(f"TensorBoard 6008 registration: {tensorboard_link}", flush=True)
    resolved_config_path.write_text(OmegaConf.to_yaml(resolved, resolve=True))
    if training.get("distributed", False):
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={len(gpus)}",
            "-m",
            "isaacgymenvs.train",
            *overrides,
        ]
        environment = _environment(gpus)
        label = f"gpus{','.join(str(item) for item in gpus)}:{case_name}"
    else:
        command = [sys.executable, "-m", "isaacgymenvs.train", *overrides]
        environment = _environment(gpu)
        label = f"gpu{gpu}:{case_name}"
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    try:
        _run_streaming(
            command,
            environment,
            training_log,
            label=label,
        )
    except Exception as error:
        failed_timing = {
            "status": "failed",
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_elapsed_seconds": time.monotonic() - started,
            "error": f"{type(error).__name__}: {error}",
        }
        (run_directory / "timing.json").write_text(
            json.dumps(failed_timing, indent=2, sort_keys=True) + "\n"
        )
        raise
    training_elapsed_seconds = time.monotonic() - started
    training_timing = {
        "status": "training_complete",
        "started_at_utc": started_at,
        "training_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_elapsed_seconds": training_elapsed_seconds,
    }
    (run_directory / "timing.json").write_text(
        json.dumps(training_timing, indent=2, sort_keys=True) + "\n"
    )

    checkpoint_directory = train_directory / experiment_name / "nn"
    checkpoints = sorted(
        checkpoint_directory.glob("*.pth"), key=lambda path: path.stat().st_mtime
    )
    if not checkpoints:
        raise RuntimeError(
            f"Training exited without a checkpoint in {checkpoint_directory}"
        )
    verification_started = time.monotonic()
    verification = _verify_checkpoint(
        checkpoints[-1], resolved_config_path, verification_device
    )
    verification_elapsed_seconds = time.monotonic() - verification_started
    timing = {
        "status": "complete",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_elapsed_seconds": training_elapsed_seconds,
        "verification_elapsed_seconds": verification_elapsed_seconds,
        "total_elapsed_seconds": time.monotonic() - started,
    }
    (run_directory / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )
    (run_directory / "timing.json").write_text(
        json.dumps(timing, indent=2, sort_keys=True) + "\n"
    )
    return {
        "profile": profile,
        "case": case_name,
        "seed": seed,
        "gpu": gpu,
        "gpus": gpus,
        **timing,
        **verification,
    }


def _run_training(
    config: dict[str, Any], suite_directory: Path
) -> list[dict[str, Any]]:
    training = config["training"]
    if int(training["num_envs"]) // int(training["sapg_block_size"]) != 6:
        raise ValueError(
            "The legacy SAPG player requires exactly six exploration blocks"
        )
    profiles = training["train_profiles"]
    seeds = [int(seed) for seed in training["seeds"]]
    gpus = training["gpu_assignments"]
    max_parallel = int(training.get("max_parallel", 1))
    distributed = bool(training.get("distributed", False))
    if not 1 <= max_parallel <= len(gpus):
        raise ValueError("training.max_parallel must be between 1 and GPU count")
    if distributed and (len(gpus) < 2 or max_parallel != 1):
        raise ValueError(
            "distributed training requires at least two GPU assignments and "
            "training.max_parallel: 1"
        )

    cases = []
    for run_index, (entry, seed) in enumerate(
        (profile, seed) for profile in profiles for seed in seeds
    ):
        profile = entry["train_profile"] if isinstance(entry, dict) else entry
        case_name = entry["name"] if isinstance(entry, dict) else profile
        case_training = dict(training)
        case_training["overrides"] = dict(training.get("overrides", {}))
        if isinstance(entry, dict):
            case_training["overrides"].update(entry.get("overrides", {}))
        gpu = gpus[run_index % len(gpus)]
        run_name = f"00_{config['name']}_{case_name}_seed{seed}"
        run_directory = suite_directory / run_name
        overrides = _training_overrides(
            case_training, profile, seed, run_name, run_directory.resolve()
        )
        resolved = _compose_resolved(overrides)
        if training.get("algorithm", "a2c_continuous") != str(resolved.train.params.algo.name):
            raise ValueError("training.algorithm must agree with the selected train profile")
        steps_per_epoch = int(resolved.train.params.config.num_actors) * int(
            resolved.train.params.config.horizon_length
        )
        steps_per_epoch *= int(
            resolved.train.params.config.get('rollout_accumulation_steps', 1)
        )
        if distributed:
            steps_per_epoch *= len(gpus)
        requested_steps = steps_per_epoch * int(
            resolved.train.params.config.max_epochs
        )
        max_frames = int(resolved.train.params.config.get("max_frames", -1))
        if max_frames >= 0 and requested_steps > max_frames:
            raise ValueError(
                f"{case_name} requests {requested_steps} steps, above cap {max_frames}"
            )
        cases.append(
            {
                "index": run_index,
                "training": case_training,
                "profile": profile,
                "case_name": case_name,
                "seed": seed,
                "gpu": gpu,
                "gpus": list(gpus) if distributed else [gpu],
                "run_name": run_name,
                "run_directory": run_directory,
                "overrides": overrides,
                "resolved": resolved,
            }
        )

    if distributed:
        results = []
        for case in cases:
            result = _run_training_case(case)
            results.append(result)
            print(
                f"Completed {result['case']} on GPUs {result['gpus']} in "
                f"{result.get('training_elapsed_seconds', 0.0):.2f}s",
                flush=True,
            )
            (suite_directory / "training_progress.json").write_text(
                json.dumps(results, indent=2, sort_keys=True) + "\n"
            )
        return results

    # One worker owns each GPU for its entire queue. This prevents a fast job on
    # one device from causing the executor to start a second job on a busy GPU.
    gpu_queues = {gpu: [] for gpu in gpus[:max_parallel]}
    available_gpus = list(gpu_queues)
    for index, case in enumerate(cases):
        assigned_gpu = available_gpus[index % len(available_gpus)]
        case["gpu"] = assigned_gpu
        gpu_queues[assigned_gpu].append(case)

    def run_gpu_queue(queue: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
        completed = []
        for queued_case in queue:
            completed.append((queued_case["index"], _run_training_case(queued_case)))
        return completed

    results_by_index = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [executor.submit(run_gpu_queue, queue) for queue in gpu_queues.values()]
        for future in concurrent.futures.as_completed(futures):
            for index, result in future.result():
                results_by_index[index] = result
                print(
                    f"Completed {result['case']} on GPU {result['gpu']} in "
                    f"{result.get('training_elapsed_seconds', 0.0):.2f}s",
                    flush=True,
                )
            ordered_results = [
                results_by_index[index] for index in sorted(results_by_index)
            ]
            (suite_directory / "training_progress.json").write_text(
                json.dumps(ordered_results, indent=2, sort_keys=True) + "\n"
            )
    return [results_by_index[index] for index in sorted(results_by_index)]


def main() -> None:
    # Parent checkpoint verification needs the same vendored package as children.
    sys.path.insert(0, str(REPOSITORY_ROOT))
    sys.path.insert(0, str(REPOSITORY_ROOT / "rl_games"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = (
        args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    )
    config = _load_yaml(config_path)
    for key, value in config.get("runtime_environment", {}).items():
        os.environ[key] = str(value)
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
