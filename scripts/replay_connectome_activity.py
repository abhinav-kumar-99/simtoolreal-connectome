#!/usr/bin/env python3
"""Replay selected saved evaluation cases and render anatomical activity via YAML."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from scripts.run_connectome_evaluation import _completed_case, _run_case
from simtoolreal_shared.activity_trace import circuit_settings, repository_path


def run(config: dict) -> dict:
    if config.get("schema_version") != 1:
        raise ValueError("Replay schema_version must be 1")
    output = repository_path(config["output_directory"])
    settings = circuit_settings({"enabled": True, **config.get("circuit", {})})
    if not settings["enabled"]:
        raise ValueError("Replay requires circuit.enabled: true")
    if not config.get("case_paths"):
        raise ValueError("Replay requires nonempty case_paths")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "rl_games"), str(ROOT), environment.get("PYTHONPATH", "")]
    )
    for key, value in config.get("runtime_environment", {}).items():
        environment[key] = str(value)
    results, destinations = [], set()
    # One worker at a time: GPU simulation exits before CPU geometry/video rendering.
    for source in config["case_paths"]:
        source_path = repository_path(source).resolve()
        case = yaml.safe_load(source_path.read_text())
        directory = (
            output
            / case["metric"]
            / case["policy"]
            / case["object_name"]
            / case["task_name"]
        )
        if directory.resolve() == source_path.parent:
            raise ValueError(
                "Replay output must not overwrite a source evaluation directory"
            )
        if directory in destinations:
            raise ValueError(
                f"Two source cases map to the same output directory: {directory}"
            )
        destinations.add(directory)
        for key in ("checkpoint_path", "policy_config_path", "trajectory_path"):
            case[key] = str(repository_path(case[key]).resolve())
            if not Path(case[key]).is_file():
                raise FileNotFoundError(case[key])
        case.update(
            {
                "record_video": True,
                "action_selection": "mean",
                "deterministic_actions": True,
                "video_path": str(directory / "rollout.mp4"),
                "output_path": str(directory / "eval.json"),
                "circuit": settings,
            }
        )
        directory.mkdir(parents=True, exist_ok=True)
        case_path = directory / "case.yaml"
        case_path.write_text(yaml.safe_dump(case, sort_keys=False))
        case.update(
            {
                "case_config_path": str(case_path),
                "log_path": str(directory / "eval.log"),
                "label": f"{case['policy']}:{case['object_name']}:{case['task_name']}",
            }
        )
        print(f"Replaying {source_path}", flush=True)
        completed = _completed_case(case)
        result = (
            completed
            if completed is not None
            else _run_case(case, int(config.get("gpu", 0)), environment)
        )
        results.append(
            {
                "source_case": str(source_path),
                "output_directory": str(directory),
                "evaluation": result,
            }
        )
    summary = {
        "status": "complete",
        "cases": len(results),
        "settings": settings,
        "results": results,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(repository_path(args.config).read_text())
    print(json.dumps(run(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
