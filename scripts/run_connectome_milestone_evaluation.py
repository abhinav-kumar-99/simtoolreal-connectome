#!/usr/bin/env python3
"""Watch frame-milestone checkpoints and generate configured evaluation videos."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_connectome_evaluation import run as run_evaluation
from simtoolreal_shared.milestone_checkpoints import (
    expected_milestone_targets,
    parse_milestone_checkpoint,
)


def _load_yaml(path: Path) -> dict:
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return config


def _repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _checkpoint_directory(suite_directory: Path, suite_name: str, policy: dict) -> Path:
    run_name = f"00_{suite_name}_{policy['name']}_seed{int(policy['seed'])}"
    return suite_directory / run_name / "rl_runs" / run_name / "nn"


def _available_checkpoints(
    suite_directory: Path,
    suite_name: str,
    policy: dict,
) -> dict[int, tuple[Path, dict[str, int]]]:
    checkpoint_directory = _checkpoint_directory(
        suite_directory, suite_name, policy
    )
    available = {}
    for path in checkpoint_directory.glob("milestone_target_*.pth"):
        parsed = parse_milestone_checkpoint(path)
        if parsed is not None:
            available[parsed["target"]] = (path, parsed)
    return available


def _milestone_output_directory(
    root: Path, policy_name: str, target: int, actual: int
) -> Path:
    return (
        root
        / policy_name
        / f"target_{target:012d}_actual_{actual:012d}"
    )


def _completed_summary(
    output_directory: Path, policy_name: str, expected_videos: int
) -> bool:
    summary_path = output_directory / "summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    return (
        summary.get("action_selection") == "mean"
        and summary.get("video_counts", {}).get(policy_name) == expected_videos
    )


def _evaluation_config(
    config: dict,
    policy: dict,
    checkpoint_path: Path,
    output_directory: Path,
) -> dict:
    evaluation = deepcopy(config["evaluation"])
    evaluation.update(
        {
            "schema_version": 1,
            "output_directory": str(output_directory),
            "policy_sources": {
                policy["name"]: {
                    "checkpoint_path": str(checkpoint_path),
                    "policy_config_path": str(
                        _repository_path(policy["policy_config_path"])
                    ),
                }
            },
            "policies": [policy["name"]],
            "gpu_assignments": [int(policy["gpu"])],
            "max_parallel": 1,
        }
    )
    return evaluation


def run(config: dict) -> dict:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported milestone-evaluation schema_version")
    suite_directory = _repository_path(config["training_suite_directory"])
    output_root = _repository_path(config["output_directory"])
    output_root.mkdir(parents=True, exist_ok=True)
    interval = int(config["milestone_interval_frames"])
    max_frames = int(config["max_frames"])
    expected_targets = expected_milestone_targets(max_frames, interval)
    policies = config["policies"]
    expected_videos = len(config["evaluation"]["eval_cases"])
    state_path = output_root / "milestone_status.json"
    state = {
        "schema_version": 1,
        "status": "running",
        "suite": config["training_suite_name"],
        "milestone_interval_frames": interval,
        "max_frames": max_frames,
        "expected_targets_per_policy": len(expected_targets),
        "expected_videos_per_target": expected_videos,
        "completed": {},
        "failures": {},
    }
    if state_path.is_file():
        previous = json.loads(state_path.read_text())
        state["completed"] = previous.get("completed", {})
        state["failures"] = previous.get("failures", {})
    _write_json(state_path, state)

    poll_seconds = float(config.get("poll_interval_seconds", 30))
    watch = bool(config.get("watch_until_complete", True))
    last_wait_message = 0.0
    while True:
        made_progress = False
        for policy in policies:
            policy_name = policy["name"]
            completed = state["completed"].setdefault(policy_name, {})
            available = _available_checkpoints(
                suite_directory, config["training_suite_name"], policy
            )
            for target in expected_targets:
                target_key = str(target)
                if target_key in completed or target not in available:
                    continue
                checkpoint_path, parsed = available[target]
                output_directory = _milestone_output_directory(
                    output_root, policy_name, target, parsed["actual"]
                )
                if not _completed_summary(
                    output_directory, policy_name, expected_videos
                ):
                    print(
                        f"Evaluating {policy_name} target {target:,} at "
                        f"actual frame {parsed['actual']:,} on GPU {policy['gpu']}",
                        flush=True,
                    )
                    try:
                        run_evaluation(
                            _evaluation_config(
                                config,
                                policy,
                                checkpoint_path,
                                output_directory,
                            )
                        )
                    except Exception as error:
                        state["failures"][f"{policy_name}:{target}"] = {
                            "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
                            "error": f"{type(error).__name__}: {error}",
                        }
                        _write_json(state_path, state)
                        print(
                            f"Milestone evaluation failed and will retry: {error}",
                            flush=True,
                        )
                        continue
                completed[target_key] = {
                    "target_frame": target,
                    "actual_frame": parsed["actual"],
                    "epoch": parsed["epoch"],
                    "checkpoint": str(checkpoint_path),
                    "evaluation_directory": str(output_directory),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                state["failures"].pop(f"{policy_name}:{target}", None)
                made_progress = True
                _write_json(state_path, state)

        complete = all(
            len(state["completed"].get(policy["name"], {}))
            == len(expected_targets)
            for policy in policies
        )
        if complete:
            state["status"] = "complete"
            state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(state_path, state)
            return state
        if not watch:
            _write_json(state_path, state)
            return state
        now = time.monotonic()
        if not made_progress and now - last_wait_message >= 300:
            counts = {
                policy["name"]: len(
                    state["completed"].get(policy["name"], {})
                )
                for policy in policies
            }
            print(f"Waiting for milestone checkpoints; completed {counts}", flush=True)
            last_wait_message = now
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = (
        args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    )
    print(json.dumps(run(_load_yaml(config_path)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
