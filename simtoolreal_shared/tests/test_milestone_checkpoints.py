from pathlib import Path

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
