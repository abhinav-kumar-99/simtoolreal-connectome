#!/usr/bin/env python3
"""Compare fly and LSTM milestone evaluations on their exact common targets."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATTERN = re.compile(r"target_(\d+)_actual_(\d+)$")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return value


def _repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _milestones(source: dict[str, Any], metric: str) -> dict[int, dict[str, Any]]:
    root = _repository_path(source["evaluation_root"])
    policy_name = str(source["policy_name"])
    found: dict[int, dict[str, Any]] = {}
    for summary_path in root.glob(f"{policy_name}/target_*/summary.json"):
        match = TARGET_PATTERN.fullmatch(summary_path.parent.name)
        if match is None:
            continue
        target, actual = (int(value) for value in match.groups())
        summary = json.loads(summary_path.read_text())
        try:
            result = summary["evaluation"][metric][policy_name]
        except KeyError as error:
            raise RuntimeError(
                f"{summary_path} is missing evaluation.{metric}.{policy_name}"
            ) from error
        if target in found:
            raise RuntimeError(f"Duplicate target {target} under {root}")
        found[target] = {
            "target_frame": target,
            "actual_frame": actual,
            "mean_task_progress_pct": float(result["mean_task_progress_pct"]),
            "mean_raw_reward": float(result["mean_raw_reward"]),
            "mean_shaped_reward": float(result["mean_shaped_reward"]),
            "cases": int(result["cases"]),
            "summary_path": str(summary_path),
        }
    return found


def _normalized_auc(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
    )
    return area / (points[-1][0] - points[0][0])


def run(config: dict[str, Any]) -> dict[str, Any]:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported comparison schema_version")
    if len(config.get("series", {})) != 2:
        raise ValueError("Exactly two comparison series are required")
    metric = str(config["metric"])
    reference = str(config["reference"])
    candidate = str(config["candidate"])
    if reference == candidate or {reference, candidate} != set(config["series"]):
        raise ValueError("reference and candidate must name the two configured series")
    milestone_sets = {
        label: _milestones(source, metric)
        for label, source in config["series"].items()
    }
    common_targets = sorted(set.intersection(*(set(x) for x in milestone_sets.values())))
    if not common_targets:
        raise RuntimeError("The configured policies have no completed common targets")

    rows = []
    for target in common_targets:
        for label, milestones in milestone_sets.items():
            rows.append({"policy": label, **milestones[target]})

    labels = [reference, candidate]
    terminal = common_targets[-1]
    auc = {
        label: _normalized_auc(
            [
                (target, milestones[target]["mean_task_progress_pct"])
                for target in common_targets
            ]
        )
        for label, milestones in milestone_sets.items()
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "metric": metric,
        "evidence_scope": "single_training_seed_pilot",
        "reference": reference,
        "candidate": candidate,
        "common_targets": common_targets,
        "common_target_count": len(common_targets),
        "terminal_common_target": terminal,
        "terminal_task_progress_pct": {
            label: milestone_sets[label][terminal]["mean_task_progress_pct"]
            for label in labels
        },
        "terminal_candidate_minus_reference_pct_points": (
            milestone_sets[candidate][terminal]["mean_task_progress_pct"]
            - milestone_sets[reference][terminal]["mean_task_progress_pct"]
        ),
        "normalized_auc_task_progress_pct": auc,
        "rows": rows,
    }

    output = _repository_path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    with (output / "common_milestones.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if bool(config.get("write_plot", True)):
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for label, milestones in milestone_sets.items():
            axis.plot(
                common_targets,
                [milestones[target]["mean_task_progress_pct"] for target in common_targets],
                marker="o",
                label=label,
            )
        axis.set_xlabel("Environment frames (exact common milestone target)")
        axis.set_ylabel("Mean paper Task Progress (%)")
        axis.set_title("Seed-42 compact fly versus parameter-matched LSTM")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.savefig(output / "task_progress_common_milestones.png", dpi=200)
        plt.close(figure)
    return report


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
