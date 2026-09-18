from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import analyze_recurrent_baseline, run_connectome_evaluation


def _summary(path: Path, policy: str, progress: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "evaluation": {
                    "paper_task_progress": {
                        policy: {
                            "cases": 3,
                            "mean_raw_reward": progress * 10,
                            "mean_shaped_reward": progress / 10,
                            "mean_task_progress_pct": progress,
                        }
                    }
                }
            }
        )
    )


def test_comparison_uses_only_exact_common_milestone_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analyze_recurrent_baseline, "REPOSITORY_ROOT", tmp_path)
    _summary(
        tmp_path / "fly/fly/target_000000000100_actual_000000000101/summary.json",
        "fly",
        10.0,
    )
    _summary(
        tmp_path / "fly/fly/target_000000000200_actual_000000000202/summary.json",
        "fly",
        20.0,
    )
    _summary(
        tmp_path / "lstm/lstm/target_000000000100_actual_000000000103/summary.json",
        "lstm",
        15.0,
    )
    _summary(
        tmp_path / "lstm/lstm/target_000000000300_actual_000000000304/summary.json",
        "lstm",
        30.0,
    )
    report = analyze_recurrent_baseline.run(
        {
            "schema_version": 1,
            "metric": "paper_task_progress",
            "output_directory": "out",
            "write_plot": True,
            "reference": "fly",
            "candidate": "lstm",
            "series": {
                "fly": {"evaluation_root": "fly", "policy_name": "fly"},
                "lstm": {"evaluation_root": "lstm", "policy_name": "lstm"},
            },
        }
    )
    assert report["common_targets"] == [100]
    assert report["terminal_candidate_minus_reference_pct_points"] == 5.0
    assert report["normalized_auc_task_progress_pct"] == {
        "fly": None,
        "lstm": None,
    }
    assert (tmp_path / "out/task_progress_common_milestones.png").stat().st_size > 0


def test_milestone_policy_source_resolves_exact_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_connectome_evaluation, "REPOSITORY_ROOT", tmp_path)
    suite = "suite"
    policy = "policy"
    run_name = f"00_{suite}_{policy}_seed42"
    run_directory = tmp_path / "training" / run_name
    checkpoint = (
        run_directory
        / "rl_runs"
        / run_name
        / "nn"
        / "milestone_target_000000000100_actual_000000000101_ep_000002.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (run_directory / "resolved_config.yaml").write_text("train: {}\n")

    policies, training_directory, suite_results = run_connectome_evaluation._resolve_policies(
        {
            "policy_sources": {
                "matched": {
                    "training_suite_name": suite,
                    "training_suite_directory": "training",
                    "policy_name": policy,
                    "seed": 42,
                    "milestone_target_frame": 100,
                }
            }
        }
    )
    assert training_directory is None and suite_results is None
    assert policies["matched"]["checkpoint"] == str(checkpoint)
    assert policies["matched"]["policy_config_path"] == str(
        run_directory / "resolved_config.yaml"
    )


def test_milestone_policy_source_rejects_missing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_connectome_evaluation, "REPOSITORY_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="Expected one target-100 checkpoint"):
        run_connectome_evaluation._resolve_policies(
            {
                "policy_sources": {
                    "matched": {
                        "training_suite_name": "suite",
                        "training_suite_directory": "training",
                        "policy_name": "policy",
                        "seed": 42,
                        "milestone_target_frame": 100,
                    }
                }
            }
        )
