"""Naming and scheduling helpers for frame-based evaluation checkpoints."""

from __future__ import annotations

import re
from pathlib import Path


MILESTONE_PATTERN = re.compile(
    r"^milestone_target_(?P<target>\d+)_actual_(?P<actual>\d+)_ep_(?P<epoch>\d+)\.pth$"
)


def crossed_milestone_targets(
    previous_frame: int,
    current_frame: int,
    interval_frames: int,
) -> list[int]:
    if interval_frames <= 0:
        return []
    first_index = previous_frame // interval_frames + 1
    last_index = current_frame // interval_frames
    return [
        index * interval_frames for index in range(first_index, last_index + 1)
    ]


def expected_milestone_targets(max_frames: int, interval_frames: int) -> list[int]:
    if max_frames <= 0 or interval_frames <= 0:
        raise ValueError("max_frames and interval_frames must be positive")
    targets = list(range(interval_frames, max_frames + 1, interval_frames))
    if not targets or targets[-1] != max_frames:
        targets.append(max_frames)
    return targets


def milestone_checkpoint_name(target_frame: int, actual_frame: int, epoch: int) -> str:
    return (
        f"milestone_target_{target_frame:012d}_actual_{actual_frame:012d}"
        f"_ep_{epoch:06d}.pth"
    )


def parse_milestone_checkpoint(path: Path) -> dict[str, int] | None:
    match = MILESTONE_PATTERN.match(path.name)
    if match is None:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}
