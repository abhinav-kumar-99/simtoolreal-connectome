#!/usr/bin/env python3
"""Watch frame-milestone checkpoints and generate configured evaluation videos."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.util import tensor_util

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_connectome_evaluation import run as run_evaluation, video_metric_names
from simtoolreal_shared.milestone_checkpoints import (
    expected_milestone_targets,
    parse_milestone_checkpoint,
)


_SCALAR_CACHE: dict[tuple[str, str], dict] = {}


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


def _event_scalar(value, tag: str) -> float | None:
    if value.tag != tag:
        return None
    try:
        if value.HasField("tensor"):
            array = np.asarray(tensor_util.make_ndarray(value.tensor))
            if array.size != 1 or not np.issubdtype(array.dtype, np.number):
                return None
            return float(array.reshape(-1)[0])
        if value.HasField("simple_value"):
            return float(value.simple_value)
    except (TypeError, ValueError):
        return None
    return None


def _scan_scalar_file(path: Path, offset: int, tag: str) -> tuple[int, dict[int, float]]:
    """Incrementally read complete TensorBoard TFRecords for one scalar tag."""
    values: dict[int, float] = {}
    if path.stat().st_size < offset:
        offset = 0
    with path.open("rb") as stream:
        stream.seek(offset)
        while True:
            record_start = stream.tell()
            header = stream.read(12)
            if len(header) < 12:
                return record_start, values
            length = struct.unpack("<Q", header[:8])[0]
            if length > 1_000_000_000:
                raise RuntimeError(f"Implausible TFRecord length {length} at {record_start}")
            payload = stream.read(length)
            footer = stream.read(4)
            if len(payload) < length or len(footer) < 4:
                return record_start, values
            event = Event()
            event.ParseFromString(payload)
            if not event.HasField("summary"):
                continue
            for summary_value in event.summary.value:
                number = _event_scalar(summary_value, tag)
                if number is not None:
                    values[int(event.step)] = number


def _training_scalar_at_checkpoint(
    checkpoint_path: Path, actual_frame: int, tag: str
) -> float:
    """Read the training scalar logged at the exact milestone checkpoint frame."""
    run_directory = checkpoint_path.parent.parent
    event_files = sorted(run_directory.rglob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError(f"No TensorBoard event files under {run_directory}")
    key = (str(run_directory.resolve()), tag)
    cache = _SCALAR_CACHE.setdefault(key, {"offsets": {}, "values": {}})
    for event_path in event_files:
        path_key = str(event_path.resolve())
        offset = int(cache["offsets"].get(path_key, 0))
        next_offset, values = _scan_scalar_file(event_path, offset, tag)
        cache["offsets"][path_key] = next_offset
        cache["values"].update(values)
    if actual_frame not in cache["values"]:
        available = sorted(cache["values"])
        nearest = min(available, key=lambda step: abs(step - actual_frame)) if available else None
        raise RuntimeError(
            f"No exact {tag!r} value at checkpoint frame {actual_frame}; "
            f"nearest logged step is {nearest}"
        )
    value = float(cache["values"][actual_frame])
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(
            f"Invalid {tag!r} value {value!r} at checkpoint frame {actual_frame}"
        )
    return value


def _completed_summary(
    output_directory: Path,
    policy_name: str,
    video_metrics: list[str],
    expected_videos_per_metric: int,
    circuit: dict | None = None,
) -> bool:
    summary_path = output_directory / "summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    counts = summary.get("video_counts_by_metric", {}).get(policy_name, {})
    from simtoolreal_shared.activity_trace import circuit_complete, circuit_settings
    settings = circuit_settings(circuit)
    if settings["enabled"]:
        for metric_name in video_metrics:
            videos = list((output_directory / metric_name / policy_name).rglob("rollout.mp4"))
            if len(videos) != expected_videos_per_metric or any(not circuit_complete(p.parent, settings) for p in videos):
                return False
    return summary.get("action_selection") == "mean" and all(
        counts.get(metric_name) == expected_videos_per_metric
        for metric_name in video_metrics
    )


def _evaluation_config(
    config: dict,
    policy: dict,
    checkpoint_path: Path,
    output_directory: Path,
    actual_frame: int,
) -> dict:
    evaluation = deepcopy(config["evaluation"])
    for metric in evaluation["metrics"].values():
        source = metric.pop("success_tolerance_source", None)
        if source is None:
            continue
        if source != "checkpoint_tensorboard":
            raise ValueError(f"Unknown success_tolerance_source: {source}")
        tag = str(
            metric.pop(
                "success_tolerance_tag", "scalars/success_tolerance/frame"
            )
        )
        metric["success_tolerance_m"] = _training_scalar_at_checkpoint(
            checkpoint_path, actual_frame, tag
        )
        metric["resolved_from_checkpoint_frame"] = int(actual_frame)
        metric["resolved_from_tensorboard_tag"] = tag
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
    configured_video_metrics = video_metric_names(config["evaluation"])
    expected_videos_per_metric = len(config["evaluation"]["eval_cases"])
    expected_videos = expected_videos_per_metric * len(configured_video_metrics)
    state_path = output_root / "milestone_status.json"
    state = {
        "schema_version": 1,
        "status": "running",
        "suite": config["training_suite_name"],
        "milestone_interval_frames": interval,
        "max_frames": max_frames,
        "expected_targets_per_policy": len(expected_targets),
        "expected_videos_per_target": expected_videos,
        "expected_videos_per_metric": expected_videos_per_metric,
        "video_metrics": configured_video_metrics,
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
                if target not in available:
                    continue
                checkpoint_path, parsed = available[target]
                output_directory = _milestone_output_directory(
                    output_root, policy_name, target, parsed["actual"]
                )
                target_complete = _completed_summary(
                    output_directory,
                    policy_name,
                    configured_video_metrics,
                    expected_videos_per_metric,
                    config["evaluation"]["videos"].get("circuit"),
                )
                if target_key in completed and target_complete:
                    continue
                completed.pop(target_key, None)
                if not target_complete:
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
                                parsed["actual"],
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
