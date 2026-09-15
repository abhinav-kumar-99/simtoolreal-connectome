#!/usr/bin/env python3
"""Run a catch-up suite, compare matched TensorBoard rewards, and continue a winner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported handoff schema_version")
    return config


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _reward_window(
    summary_directory: Path,
    tag: str,
    target_step: int,
    window_points: int,
) -> dict[str, Any]:
    accumulator = EventAccumulator(
        str(summary_directory), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"No scalar tag {tag!r} in {summary_directory}")
    eligible = [
        event for event in accumulator.Scalars(tag) if int(event.step) <= target_step
    ]
    if not eligible or int(eligible[-1].step) != target_step:
        last_step = int(eligible[-1].step) if eligible else None
        raise RuntimeError(
            f"{summary_directory} has not reached comparison step {target_step}; "
            f"last eligible step is {last_step}"
        )
    window = eligible[-window_points:]
    if len(window) != window_points:
        raise RuntimeError(
            f"Need {window_points} reward points through {target_step}, found {len(window)}"
        )
    return {
        "summary_directory": str(summary_directory),
        "tag": tag,
        "points": len(window),
        "first_step": int(window[0].step),
        "last_step": int(window[-1].step),
        "mean": sum(float(event.value) for event in window) / len(window),
        "last": float(window[-1].value),
    }


def _run_suite(config_path: Path, log_path: Path) -> None:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_connectome_suite.py"),
        "--config",
        str(config_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(command)}", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = _repository_path(args.config)
    config = _load_yaml(config_path)
    output_directory = _repository_path(config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    catch_up_suite = _repository_path(config["catch_up_suite"])
    _run_suite(catch_up_suite, output_directory / "catch_up.log")

    comparison = config["comparison"]
    tag = str(comparison["tag"])
    target_step = int(comparison["target_step"])
    window_points = int(comparison["window_points"])
    candidate_name = str(comparison["candidate"])
    baseline_name = str(comparison["baseline"])
    runs = {
        name: _reward_window(
            _repository_path(run["summary_directory"]),
            tag,
            target_step,
            window_points,
        )
        for name, run in comparison["runs"].items()
    }
    winner = (
        candidate_name
        if runs[candidate_name]["mean"] > runs[baseline_name]["mean"]
        else baseline_name
    )
    decision = {
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "candidate mean must be strictly greater than baseline mean",
        "candidate": candidate_name,
        "baseline": baseline_name,
        "winner": winner,
        "runs": runs,
        "continuation_suite": str(config["continuation_suites"][winner]),
    }
    decision_path = output_directory / "decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)

    continuation_suite = _repository_path(config["continuation_suites"][winner])
    _run_suite(continuation_suite, output_directory / f"continue_{winner}.log")


if __name__ == "__main__":
    main()
