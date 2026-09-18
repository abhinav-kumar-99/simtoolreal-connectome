#!/usr/bin/env python3
"""Evaluate trained connectome policies and generate videos from YAML."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from simtoolreal_shared.milestone_checkpoints import parse_milestone_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return config


def _latest_scalar(run_directory: Path, tag: str) -> dict | None:
    events = list(run_directory.rglob("events.out.tfevents*"))
    if len(events) != 1:
        raise RuntimeError(f"Expected one event file under {run_directory}, got {events}")
    accumulator = EventAccumulator(str(events[0]))
    accumulator.Reload()
    if tag not in accumulator.Tags()["scalars"]:
        return None
    event = accumulator.Scalars(tag)[-1]
    return {"step": int(event.step), "value": float(event.value)}


def _training_metrics(suite_directory: Path, suite_results: dict) -> dict:
    metrics = {}
    for result in suite_results["stages"]["train"]:
        checkpoint = Path(result["checkpoint"])
        run_directory = checkpoint.parents[3]
        metrics[result["case"]] = {
            "checkpoint_frame": result["checkpoint_frame"],
            "training_elapsed_seconds": result["training_elapsed_seconds"],
            "raw_mean_episode_reward": _latest_scalar(run_directory, "rewards/step"),
            "shaped_mean_episode_reward": _latest_scalar(
                run_directory, "shaped_rewards/step"
            ),
            "mean_successes": _latest_scalar(run_directory, "mean_successes/frame"),
            "success_ratio": _latest_scalar(run_directory, "success_ratio/frame"),
        }
    return metrics


def _repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _resolve_policies(config: dict) -> tuple[dict, Path | None, dict | None]:
    explicit_sources = config.get("policy_sources")
    if explicit_sources is not None:
        policies = {}
        for name, source in explicit_sources.items():
            if "checkpoint_path" in source:
                checkpoint = _repository_path(source["checkpoint_path"])
                policy_config = _repository_path(source["policy_config_path"])
            else:
                required = {
                    "training_suite_name",
                    "training_suite_directory",
                    "policy_name",
                    "seed",
                    "milestone_target_frame",
                }
                missing = sorted(required - set(source))
                if missing:
                    raise ValueError(
                        f"Milestone policy source {name} is missing {missing}"
                    )
                suite_name = str(source["training_suite_name"])
                policy_name = str(source["policy_name"])
                seed = int(source["seed"])
                run_name = f"00_{suite_name}_{policy_name}_seed{seed}"
                run_directory = (
                    _repository_path(source["training_suite_directory"])
                    / run_name
                )
                checkpoint_directory = run_directory / "rl_runs" / run_name / "nn"
                target = int(source["milestone_target_frame"])
                matches = []
                for path in checkpoint_directory.glob("milestone_target_*.pth"):
                    parsed = parse_milestone_checkpoint(path)
                    if parsed is not None and parsed["target"] == target:
                        matches.append(path)
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one target-{target} checkpoint for {name} under "
                        f"{checkpoint_directory}, got {matches}"
                    )
                checkpoint = matches[0]
                policy_config = _repository_path(
                    source.get(
                        "policy_config_path",
                        str(run_directory / "resolved_config.yaml"),
                    )
                )
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint for {name}: {checkpoint}")
            if not policy_config.is_file():
                raise FileNotFoundError(
                    f"Missing policy config for {name}: {policy_config}"
                )
            policies[name] = {
                "case": name,
                "checkpoint": str(checkpoint),
                "policy_config_path": str(policy_config),
            }
        return policies, None, None

    training_directory = _repository_path(config["training_suite_directory"])
    suite_results = json.loads((training_directory / "suite_results.json").read_text())
    policies = {entry["case"]: entry for entry in suite_results["stages"]["train"]}
    return policies, training_directory, suite_results


def video_metric_names(config: dict) -> list[str]:
    """Return the metric directories that must contain rollout videos."""
    videos = config["videos"]
    names = videos.get("metrics")
    if names is None:
        names = [videos["metric"]]
    names = [str(name) for name in names]
    if not names or len(names) != len(set(names)):
        raise ValueError("videos.metrics must contain unique metric names")
    missing = [name for name in names if name not in config["metrics"]]
    if missing:
        raise ValueError(f"Video metrics are not configured evaluation metrics: {missing}")
    return names


def _completed_case(case: dict) -> dict | None:
    """Reuse a matching completed case so adding a metric does not rerun old videos."""
    output_path = Path(case["output_path"])
    if not output_path.is_file():
        return None
    try:
        result = json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        result.get("policy") != case["policy"]
        or result.get("metric") != case["metric"]
        or result.get("action_selection") != case["action_selection"]
        or float(result.get("success_tolerance_m", float("nan")))
        != float(case["success_tolerance"])
    ):
        return None
    if case["record_video"]:
        video_path = Path(case["video_path"])
        if not video_path.is_file() or video_path.stat().st_size == 0:
            return None
    return result


def _run_case(case: dict, gpu: int, environment: dict[str, str]) -> dict:
    case_path = Path(case["case_config_path"])
    log_path = Path(case["log_path"])
    child_environment = environment.copy()
    child_environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "dextoolbench/eval_worker_isaacgym.py"),
        "--config",
        str(case_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=child_environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    output_path = Path(case["output_path"])
    if not output_path.exists():
        raise RuntimeError(
            f"Eval {case['label']} produced no result (exit {result.returncode}); see {log_path}"
        )
    if result.returncode not in (0, 139, -11):
        raise subprocess.CalledProcessError(result.returncode, command)
    evaluation = json.loads(output_path.read_text())
    evaluation["gpu"] = gpu
    print(
        f"[{case['label']}] task progress "
        f"{evaluation['mean_task_progress_pct']:.1f}%",
        flush=True,
    )
    return evaluation


def run(config: dict) -> dict:
    output_directory = REPOSITORY_ROOT / config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    policies, training_directory, suite_results = _resolve_policies(config)
    requested_policies = config.get("policies", list(policies))
    action_selection = str(config.get("action_selection", "mean"))
    if action_selection not in {"mean", "sample"}:
        raise ValueError("action_selection must be 'mean' or 'sample'")
    if action_selection != "mean":
        raise ValueError("Evaluation video capture requires action_selection: mean")
    gpus = [int(gpu) for gpu in config["gpu_assignments"]]
    max_parallel = int(config.get("max_parallel", len(gpus)))
    if not 1 <= max_parallel <= len(gpus):
        raise ValueError("max_parallel must be between one and GPU count")

    video_metrics = video_metric_names(config)

    cases = []
    evaluations = []
    for policy_name in requested_policies:
        policy = policies[policy_name]
        checkpoint = Path(policy["checkpoint"])
        policy_config = Path(
            policy.get("policy_config_path", checkpoint.parents[3] / "resolved_config.yaml")
        )
        for metric_name, metric in config["metrics"].items():
            for task in config["eval_cases"]:
                case_directory = (
                    output_directory
                    / metric_name
                    / policy_name
                    / task["object_name"]
                    / task["task_name"]
                )
                record_video = metric_name in video_metrics
                worker_config = {
                    "policy": policy_name,
                    "metric": metric_name,
                    "policy_config_path": str(policy_config),
                    "checkpoint_path": str(checkpoint),
                    "object_category": task["object_category"],
                    "object_name": task["object_name"],
                    "task_name": task["task_name"],
                    "trajectory_path": str(
                        REPOSITORY_ROOT
                        / "dextoolbench/trajectories"
                        / task["object_category"]
                        / task["object_name"]
                        / f"{task['task_name']}.json"
                    ),
                    "table_urdf": config["table_urdf"],
                    "success_tolerance": metric["success_tolerance_m"],
                    "num_episodes": int(config["episodes_per_case"]),
                    "downsample_factor": int(config["downsample_factor"]),
                    "z_offset": float(config["z_offset"]),
                    "max_steps": int(config["max_steps"]),
                    "action_selection": action_selection,
                    "deterministic_actions": action_selection == "mean",
                    "record_video": record_video,
                    "video_fps": int(config["videos"]["fps"]),
                    "video_frame_interval": int(config["videos"]["frame_interval"]),
                    "camera_resolution_reduction_factor": int(
                        config["videos"].get("camera_resolution_reduction_factor", 4)
                    ),
                    "video_path": str(case_directory / "rollout.mp4"),
                    "output_path": str(case_directory / "eval.json"),
                }
                case_directory.mkdir(parents=True, exist_ok=True)
                case_config_path = case_directory / "case.yaml"
                case_config_path.write_text(yaml.safe_dump(worker_config, sort_keys=False))
                case = {
                    **worker_config,
                    "label": f"{metric_name}:{policy_name}:{task['object_name']}:{task['task_name']}",
                    "case_config_path": str(case_config_path),
                    "log_path": str(case_directory / "eval.log"),
                }
                completed = _completed_case(case)
                if completed is None:
                    cases.append(case)
                else:
                    evaluations.append(completed)

    gpu_queues = {gpu: [] for gpu in gpus[:max_parallel]}
    for index, case in enumerate(cases):
        gpu = list(gpu_queues)[index % len(gpu_queues)]
        gpu_queues[gpu].append(case)
    environment = os.environ.copy()
    python_path = [str(REPOSITORY_ROOT / "rl_games"), str(REPOSITORY_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    for key, value in config.get("runtime_environment", {}).items():
        environment[key] = str(value)

    def run_queue(gpu: int, queue: list[dict]) -> list[dict]:
        return [_run_case(case, gpu, environment) for case in queue]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [
            executor.submit(run_queue, gpu, queue) for gpu, queue in gpu_queues.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            evaluations.extend(future.result())

    aggregate = {}
    for metric_name in config["metrics"]:
        aggregate[metric_name] = {}
        for policy_name in requested_policies:
            selected = [
                item
                for item in evaluations
                if item["metric"] == metric_name and item["policy"] == policy_name
            ]
            aggregate[metric_name][policy_name] = {
                "cases": len(selected),
                "mean_raw_reward": float(np.mean([x["mean_raw_reward"] for x in selected])),
                "mean_shaped_reward": float(
                    np.mean([x["mean_shaped_reward"] for x in selected])
                ),
                "mean_task_progress_pct": float(
                    np.mean([x["mean_task_progress_pct"] for x in selected])
                ),
            }

    expected_videos_per_metric = len(config["eval_cases"])
    video_counts = {}
    video_counts_by_metric = {}
    for policy_name in requested_policies:
        video_counts_by_metric[policy_name] = {}
        policy_total = 0
        for metric_name in video_metrics:
            videos = list(
                (output_directory / metric_name / policy_name).rglob(
                    "rollout.mp4"
                )
            )
            if len(videos) != expected_videos_per_metric or any(
                path.stat().st_size == 0 for path in videos
            ):
                raise RuntimeError(
                    f"Expected {expected_videos_per_metric} nonempty {metric_name} "
                    f"videos for {policy_name}, got {videos}"
                )
            video_counts_by_metric[policy_name][metric_name] = len(videos)
            policy_total += len(videos)
        video_counts[policy_name] = policy_total

    summary = {
        "training": (
            _training_metrics(training_directory, suite_results)
            if training_directory is not None and suite_results is not None
            else {}
        ),
        "evaluation": aggregate,
        "metric_contracts": config["metrics"],
        "video_counts": video_counts,
        "video_counts_by_metric": video_counts_by_metric,
        "video_metrics": video_metrics,
        "evaluated_cases": config["eval_cases"],
        "episodes_per_case": int(config["episodes_per_case"]),
        "action_selection": action_selection,
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    print(json.dumps(run(_load_yaml(config_path)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
