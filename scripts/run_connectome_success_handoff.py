#!/usr/bin/env python3
"""Apply YAML-owned, staged success gates to connectome training jobs."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tensorboard.compat.proto.event_pb2 import Event

from scripts.run_connectome_numerical_fallback import (
    REPOSITORY_ROOT,
    _launch,
    _matching_pids,
    _repository_path,
    _stop_processes,
    _summary_numbers,
    _write_json,
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("Expected success-handoff schema_version: 1")
    if not config.get("stages"):
        raise ValueError("At least one candidate stage is required")
    return config


def _scan_metric_file(
    path: Path, offset: int, tag: str
) -> tuple[int, list[dict[str, float | int]]]:
    """Read complete new TFRecord entries and return points for one scalar tag."""
    points: list[dict[str, float | int]] = []
    if path.stat().st_size < offset:
        offset = 0
    with path.open("rb") as stream:
        stream.seek(offset)
        while True:
            record_start = stream.tell()
            header = stream.read(12)
            if len(header) < 12:
                return record_start, points
            length = struct.unpack("<Q", header[:8])[0]
            if length > 1_000_000_000:
                raise RuntimeError(
                    f"Implausible TFRecord length {length} at {record_start}"
                )
            payload = stream.read(length)
            footer = stream.read(4)
            if len(payload) < length or len(footer) < 4:
                return record_start, points
            event = Event()
            event.ParseFromString(payload)
            for observed_tag, number in _summary_numbers(event):
                if observed_tag == tag:
                    points.append({"step": int(event.step), "value": float(number)})


def _update_window(
    metric_state: dict[str, Any], points: list[dict[str, float | int]],
    target: int, window_steps: int,
) -> None:
    window_start = target - window_steps
    values = metric_state.setdefault("window_values", {})
    for point in points:
        step = int(point["step"])
        if window_start <= step <= target:
            values[str(step)] = float(point["value"])
        if step >= target and (
            metric_state.get("crossing") is None
            or step <= int(metric_state["crossing"]["step"])
        ):
            metric_state["crossing"] = point


def summarize_window(
    metric_state: dict[str, Any], target: int, window_steps: int
) -> dict[str, Any] | None:
    """Return an arithmetic mean only after the source has crossed the target."""
    if metric_state.get("crossing") is None:
        return None
    values = {
        int(step): float(value)
        for step, value in metric_state.get("window_values", {}).items()
    }
    if not values:
        raise RuntimeError("No metric samples exist in the configured averaging window")
    steps = sorted(values)
    return {
        "window_start_step": target - window_steps,
        "window_end_step": target,
        "first_logged_step": steps[0],
        "last_logged_step": steps[-1],
        "sample_count": len(steps),
        "mean": math.fsum(values[step] for step in steps) / len(steps),
        "crossing_step": int(metric_state["crossing"]["step"]),
    }


def configure_comparison_state(
    state: dict[str, Any], comparison: dict[str, Any]
) -> None:
    """Reset metric state when any comparison-contract field changes."""
    signature = {
        "tag": str(comparison["tag"]),
        "target_step": int(comparison["target_step"]),
        "aggregation": str(comparison["aggregation"]),
        "window_steps": int(comparison["window_steps"]),
        "comparator": str(comparison["comparator"]),
    }
    if signature["aggregation"] != "arithmetic_mean":
        raise ValueError("Only comparison.aggregation: arithmetic_mean is supported")
    if signature["comparator"] != "strict_less":
        raise ValueError("Only comparison.comparator: strict_less is supported")
    if not 0 < signature["window_steps"] <= signature["target_step"]:
        raise ValueError("comparison.window_steps must be positive and no larger than target_step")

    stored = state.get("comparison_contract")
    if stored is None and state.get("comparison_target_step") is not None:
        # State written by the point-comparison monitor predating contract
        # signatures. Preserve its actual semantics in migration history.
        stored = {
            "tag": "mean_successes/frame",
            "target_step": int(state["comparison_target_step"]),
            "aggregation": "closest_after_crossing",
            "window_steps": 0,
            "comparator": "strict_less",
        }
    if stored == signature:
        return
    if stored is not None or state.get("metrics"):
        if state.get("status") != "monitoring" or state.get("transitions"):
            raise RuntimeError(
                "Cannot change comparison contract after a terminal decision or transition"
            )
        prior_points = {
            name: {
                key: metric.get(key)
                for key in ("before", "after", "selected", "crossing", "window_summary")
                if metric.get(key) is not None
            }
            for name, metric in state.get("metrics", {}).items()
        }
        state.setdefault("comparison_change_history", []).append({
            "previous_contract": stored,
            "new_contract": signature,
            "changed_at_utc": datetime.now(timezone.utc).isoformat(),
            "prior_points": prior_points,
        })
        state["metrics"] = {}
    state["comparison_contract"] = signature
    state["comparison_target_step"] = signature["target_step"]


def _scan_source(
    source: dict[str, Any], metric_state: dict[str, Any], tag: str,
    target: int, window_steps: int,
) -> None:
    offsets = metric_state.setdefault("event_offsets", {})
    event_glob = str(_repository_path(source["event_glob"]))
    for event_path in sorted(Path("/").glob(event_glob.lstrip("/"))):
        key = str(event_path)
        offset, points = _scan_metric_file(event_path, int(offsets.get(key, 0)), tag)
        offsets[key] = offset
        _update_window(metric_state, points, target, window_steps)
    metric_state["window_summary"] = summarize_window(
        metric_state, target, window_steps
    )


def _stop_stage(stage: dict[str, Any], timeout: float) -> dict[str, list[int]]:
    stopped: dict[str, list[int]] = {}
    for marker in stage["stop_process_markers"]:
        pids = _matching_pids(str(marker))
        if pids:
            stopped[str(marker)] = _stop_processes(pids, timeout)
    return stopped


def _launch_replacement(
    replacement: dict[str, Any], output: Path, state: dict[str, Any]
) -> None:
    training_config = _repository_path(replacement["training_config"])
    video_config = _repository_path(replacement["video_config"])
    launch_name = str(replacement["stage"])
    training_pid = _launch(
        training_config,
        "run_connectome_suite.py",
        output / f"{launch_name}_training.log",
    )
    state["pending_launch"] = {
        "stage": launch_name,
        "training_config": str(training_config),
        "video_config": str(video_config),
        "training_pid": training_pid,
    }
    _write_json(output / "status.json", state)

    deadline = time.monotonic() + float(replacement.get("startup_timeout_seconds", 120))
    trainer_pids: list[int] = []
    while time.monotonic() < deadline:
        trainer_pids = _matching_pids(str(replacement["trainer_process_marker"]))
        if trainer_pids:
            break
        if not Path(f"/proc/{training_pid}").exists():
            break
        time.sleep(1)
    if not trainer_pids:
        state["status"] = "replacement_failed_to_start"
        state["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "status.json", state)
        return

    video_pid = _launch(
        video_config,
        "run_connectome_milestone_evaluation.py",
        output / f"{launch_name}_videos.log",
    )
    state["pending_launch"].update(
        {"trainer_pids": trainer_pids, "video_pid": video_pid}
    )
    state["current_stage"] = launch_name
    state["status"] = (
        "monitoring" if not bool(replacement.get("terminal", False))
        else "terminal_replacement_launched"
    )
    state["launched_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output / "status.json", state)


def run(config: dict[str, Any]) -> dict[str, Any]:
    output = _repository_path(config["output_directory"])
    state_path = output / "status.json"
    stage_by_name = {str(stage["name"]): stage for stage in config["stages"]}
    comparison = config["comparison"]
    tag = str(comparison["tag"])
    target = int(comparison["target_step"])
    window_steps = int(comparison["window_steps"])
    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        "schema_version": 1,
        "status": "monitoring",
        "current_stage": str(config["stages"][0]["name"]),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {},
        "transitions": [],
    }
    configure_comparison_state(state, comparison)
    if state.get("status") in {
        "candidate_retained", "terminal_replacement_launched",
        "replacement_failed_to_start",
    }:
        return state

    reference_name = str(config["reference"]["name"])
    poll_seconds = float(config.get("poll_interval_seconds", 30))
    shutdown_timeout = float(config.get("shutdown_timeout_seconds", 60))

    while True:
        current_name = str(state["current_stage"])
        current = stage_by_name[current_name]
        reference_metric = state["metrics"].setdefault(reference_name, {})
        candidate_metric = state["metrics"].setdefault(current_name, {})
        _scan_source(
            config["reference"], reference_metric, tag, target, window_steps
        )
        _scan_source(current, candidate_metric, tag, target, window_steps)
        state["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["candidate_process_pids"] = {
            marker: _matching_pids(str(marker))
            for marker in current["stop_process_markers"]
        }

        reference_point = reference_metric.get("window_summary")
        candidate_point = candidate_metric.get("window_summary")
        if reference_point is not None and candidate_point is not None:
            replace = float(candidate_point["mean"]) < float(reference_point["mean"])
            decision = {
                "stage": current_name,
                "decided_at_utc": datetime.now(timezone.utc).isoformat(),
                "metric_tag": tag,
                "aggregation": "arithmetic_mean",
                "comparison": "strict_less",
                "target_step": target,
                "window_steps": window_steps,
                "reference": reference_point,
                "candidate": candidate_point,
                "replace": replace,
            }
            if not replace:
                state["status"] = "candidate_retained"
                state["decision"] = decision
                _write_json(state_path, state)
                return state

            state["status"] = "transitioning"
            state["decision"] = decision
            _write_json(state_path, state)
            stopped = _stop_stage(current, shutdown_timeout)
            decision["stopped_pids"] = stopped
            state["transitions"].append(decision)
            replacement = current["replacement"]
            _launch_replacement(replacement, output, state)
            if state["status"] != "monitoring":
                return state
            state.pop("decision", None)
            state.pop("pending_launch", None)
            _write_json(state_path, state)
            continue

        _write_json(state_path, state)
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    try:
        print(json.dumps(run(_load_config(path)), indent=2, sort_keys=True), flush=True)
    except KeyboardInterrupt:
        print("Success handoff monitor stopped by operator", flush=True)


if __name__ == "__main__":
    main()
