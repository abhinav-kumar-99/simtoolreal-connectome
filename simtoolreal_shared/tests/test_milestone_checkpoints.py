from pathlib import Path

import json
from torch.utils.tensorboard import SummaryWriter

from simtoolreal_shared.milestone_checkpoints import (
    crossed_milestone_targets,
    expected_milestone_targets,
    milestone_checkpoint_name,
    parse_milestone_checkpoint,
)


def test_crossed_milestones_and_near_cap_target() -> None:
    assert crossed_milestone_targets(249_888_768, 250_085_376, 250_000_000) == [
        250_000_000
    ]
    targets = expected_milestone_targets(1_000_000_000, 250_000_000)
    assert targets == [250_000_000, 500_000_000, 750_000_000, 1_000_000_000]
    assert expected_milestone_targets(999_948_288, 250_000_000)[-1] == 999_948_288


def test_milestone_checkpoint_name_round_trip() -> None:
    name = milestone_checkpoint_name(100_000_000_000, 99_999_940_608, 508_626)
    parsed = parse_milestone_checkpoint(Path(name))
    assert parsed == {
        "target": 100_000_000_000,
        "actual": 99_999_940_608,
        "epoch": 508_626,
    }


def test_watcher_writes_waiting_status_before_first_checkpoint(tmp_path: Path) -> None:
    from scripts.run_connectome_milestone_evaluation import run

    output = tmp_path / "evaluations"
    result = run(
        {
            "schema_version": 1,
            "training_suite_name": "not_started",
            "training_suite_directory": str(tmp_path / "training"),
            "output_directory": str(output),
            "max_frames": 1_000,
            "milestone_interval_frames": 250,
            "watch_until_complete": False,
            "policies": [{"name": "gains", "seed": 42, "gpu": 0}],
            "evaluation": {
                "metrics": {"paper": {"success_tolerance_m": 0.02}},
                "videos": {"metrics": ["paper"]},
                "eval_cases": [{"task_name": "unused"}],
            },
        }
    )
    assert result["status"] == "running"
    assert (output / "milestone_status.json").is_file()


def test_checkpoint_tolerance_resolves_exact_tensorboard_frame(tmp_path: Path) -> None:
    from scripts.run_connectome_milestone_evaluation import (
        _SCALAR_CACHE,
        _evaluation_config,
        _training_scalar_at_checkpoint,
    )

    run_directory = tmp_path / "rl_runs" / "run"
    checkpoint = run_directory / "nn" / "milestone_target_000000000250_actual_000000000256_ep_000001.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    writer = SummaryWriter(str(run_directory / "summaries"))
    writer.add_scalar("scalars/success_tolerance/frame", 0.075, 256)
    writer.flush()
    writer.close()
    _SCALAR_CACHE.clear()

    assert _training_scalar_at_checkpoint(
        checkpoint, 256, "scalars/success_tolerance/frame"
    ) == 0.07500000298023224
    config = {
        "evaluation": {
            "metrics": {
                "paper": {"success_tolerance_m": 0.02},
                "checkpoint": {
                    "success_tolerance_source": "checkpoint_tensorboard",
                    "success_tolerance_tag": "scalars/success_tolerance/frame",
                },
            },
            "videos": {"metrics": ["paper", "checkpoint"]},
        }
    }
    resolved = _evaluation_config(
        config,
        {"name": "policy", "gpu": 0, "policy_config_path": "policy.yaml"},
        checkpoint,
        tmp_path / "output",
        256,
    )
    assert resolved["metrics"]["paper"]["success_tolerance_m"] == 0.02
    assert resolved["metrics"]["checkpoint"]["success_tolerance_m"] == 0.07500000298023224
    assert resolved["metrics"]["checkpoint"]["resolved_from_checkpoint_frame"] == 256
    assert resolved["metrics"]["checkpoint"]["resolved_from_tensorboard_tag"] == (
        "scalars/success_tolerance/frame"
    )


def test_completed_milestone_requires_every_video_metric(tmp_path: Path) -> None:
    from scripts.run_connectome_milestone_evaluation import _completed_summary

    summary = {
        "action_selection": "mean",
        "video_counts_by_metric": {
            "policy": {"paper": 3, "checkpoint": 3},
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert _completed_summary(tmp_path, "policy", ["paper", "checkpoint"], 3)
    summary["video_counts_by_metric"]["policy"].pop("checkpoint")
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert not _completed_summary(tmp_path, "policy", ["paper", "checkpoint"], 3)
