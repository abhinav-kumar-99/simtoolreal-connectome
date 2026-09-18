import json

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pytest

from simtoolreal_shared.activity_trace import (
    circuit_complete,
    circuit_settings,
    frame_state_indices,
    sha256,
)
from simtoolreal_shared.anatomical_activity import (
    SWC_URL,
    AnatomicalPanel,
    parse_swc,
    project,
    render_activity,
)


def test_anatomical_units_and_parent_validation():
    segments = parse_swc("1 0 1000 0 2000 1 -1\n2 0 2000 0 3000 1 1\n")
    np.testing.assert_allclose(segments[0], [[16, 0, 24], [8, 0, 16]])
    with pytest.raises(ValueError, match="missing parent"):
        parse_swc("1 0 1000 0 2000 1 -1\n2 0 2000 0 3000 1 3\n")
    p = project(np.array([[0, 0], [1, 0], [0, 1]]), [0, 10, 0, 20], 100, 100)
    assert p[1, 0] - p[0, 0] == p[2, 1] - p[0, 1]


def test_neural_video_clocks_terminal_hold_and_episode_reset():
    trace = {
        "metadata": np.asarray(json.dumps({"neural_updates": 4})),
        "episode": np.array([0] * 9 + [1] * 5),
        "tick": np.r_[np.arange(9), np.arange(5)],
        "frame_episode": np.array([0, 1]),
        "frame_step": np.array([0, 0]),
    }
    assert frame_state_indices(trace, 4, 3).tolist() == [0, 3, 6, 8, 9, 12, 13, 13]
    trace["tick"][4] = 5
    with pytest.raises(ValueError, match="missing or unordered"):
        frame_state_indices(trace, 4, 3)


def _geometry(tmp_path):
    cache = tmp_path / "geometry"
    cache.mkdir()
    files = {}
    for body_id, z in [(1, 1000), (2, 3000)]:
        path = cache / f"{body_id}.swc"
        path.write_text(f"1 0 1000 0 {z} 1 -1\n2 0 4000 0 {z + 1000} 1 1\n")
        files[str(body_id)] = {
            "sha256": sha256(path),
            "url": SWC_URL.format(body_id=body_id),
        }
    (cache / "manifest.json").write_text(
        json.dumps({"files": files, "dataset": "male-cns:v1.0"})
    )
    annotations = tmp_path / "annotations.feather"
    pd.DataFrame(
        {"somaLocation": [np.array([1000, 0, 1000]), np.array([4000, 0, 4000])]}
    ).to_feather(annotations)
    return cache, annotations


def test_signed_activity_and_video_encoding(tmp_path):
    cache, annotations = _geometry(tmp_path)
    body_ids = np.array([1, 2])
    panel = AnatomicalPanel(body_ids, cache, annotations, 300, 200, [0, 40, 0, 40])
    positive = panel.render(np.ones(2))
    negative = panel.render(-np.ones(2))
    active = np.asarray(panel.mask.sum(axis=1)).reshape(200, 300) > 0.5
    assert positive[active, 1].mean() > positive[active, 0].mean()
    assert negative[active, 0].mean() > negative[active, 1].mean()
    rollout = tmp_path / "rollout.mp4"
    with imageio.get_writer(rollout, fps=20, macro_block_size=2) as writer:
        writer.append_data(np.zeros((100, 160, 3), dtype=np.uint8))
        writer.append_data(np.full((100, 160, 3), 100, dtype=np.uint8))
    trace_path = tmp_path / "activity.npz"
    np.savez_compressed(
        trace_path,
        states=np.linspace(-1, 1, 26).reshape(13, 2),
        body_ids=body_ids,
        episode=np.zeros(13, dtype=int),
        tick=np.arange(13),
        frame_episode=np.array([0, 0]),
        frame_step=np.array([0, 3]),
        metadata=np.asarray(
            json.dumps(
                {
                    "neural_updates": 4,
                    "video_fps": 20,
                    "video_frame_interval": 3,
                    "policy": "test",
                    "object_name": "marker",
                    "task_name": "write",
                    "edge_count": 1,
                }
            )
        ),
    )
    settings = circuit_settings(
        {
            "enabled": True,
            "resolution": [800, 600],
            "geometry_cache": str(cache),
            "annotations_path": str(annotations),
            "vnc_bounds_um": [0, 40, 0, 40],
        }
    )
    result = render_activity(
        {
            "trace_path": str(trace_path),
            "rollout_path": str(rollout),
            "circuit": settings,
        }
    )
    assert (
        result["fps"] == 80
        and result["frames"] == 8
        and result["duration_seconds"] == 0.1
    )
    assert circuit_complete(tmp_path, settings)
    (tmp_path / "circuit.mp4").unlink()
    assert not circuit_complete(tmp_path, settings)


def test_old_video_is_not_a_complete_anatomical_case(tmp_path):
    (tmp_path / "rollout.mp4").write_bytes(b"old robot-only video")
    assert not circuit_complete(tmp_path, circuit_settings({"enabled": True}))
