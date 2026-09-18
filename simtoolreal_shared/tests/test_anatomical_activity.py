import json

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pytest

from simtoolreal_shared.activity_interpretation import (
    interpretation_context,
    population_activity,
    population_anchors,
)
from simtoolreal_shared.activity_trace import (
    circuit_complete,
    circuit_settings,
    frame_state_indices,
    recording_matches,
    recording_options,
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
        {
            "bodyId": [1, 2, 3],
            "somaNeuromere": ["T1", "T2", "T3"],
            "class": ["mechanosensory_tactile", "motor", "interneuron"],
            "somaLocation": [
                np.array([1000, 0, 1000]),
                np.array([4000, 0, 4000]),
                np.array([2000, 0, 5000]),
            ],
        }
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
    metadata_path = tmp_path / "circuit_render.json"
    saved = metadata_path.read_text()
    old_metadata = json.loads(saved)
    old_metadata.pop("render_version")
    metadata_path.write_text(json.dumps(old_metadata))
    assert not circuit_complete(tmp_path, settings)
    metadata_path.write_text(saved)
    (tmp_path / "circuit.mp4").unlink()
    assert not circuit_complete(tmp_path, settings)


def test_old_video_is_not_a_complete_anatomical_case(tmp_path):
    (tmp_path / "rollout.mp4").write_bytes(b"old robot-only video")
    assert not circuit_complete(tmp_path, circuit_settings({"enabled": True}))


def test_bilateral_callouts_preserve_both_clusters_despite_one_cell_imbalance():
    # A global median selects the right bulb with 68 vs 67 cells, even though
    # the population is nearly symmetric. Source sides must keep two anchors.
    centers = np.array([[486.0, 571.0]] * 68 + [[310.0, 581.0]] * 67)
    sides = np.array(["L"] * 68 + ["R"] * 67)
    assert np.median(centers[:, 0]) == 486.0
    anchors = population_anchors(centers, sides)
    assert [(a["side"], a["count"]) for a in anchors] == [("L", 68), ("R", 67)]
    np.testing.assert_allclose(
        [a["center_um"] for a in anchors], [[486, 571], [310, 581]]
    )


def test_population_labels_follow_ordered_artifact_and_source_annotations(tmp_path):
    cache, annotations = _geometry(tmp_path)
    artifact = tmp_path / "artifact.npz"
    ids = np.array([2, 1])
    np.savez(
        artifact,
        body_ids=ids,
        sensory_indices=[1],
        descending_indices=[],
        motor_indices=[0],
    )
    metadata = {
        "artifact_path": str(artifact),
        "artifact_sha256": sha256(artifact),
        "readout_feature_population": "all",
    }
    context = interpretation_context(metadata, ids, cache, annotations)
    assert context["readout_label"] == "All 2 cells feed robot actions"
    # Cell 2 belongs to T2; never label it a front-leg cell from its mask alone.
    assert context["motor_detail"] == "Selected motor cells"
    assert context["sensory_detail"] == "Touch + position cells"
    assert population_activity(context, np.array([-0.2, 0.8])) == pytest.approx(
        [0.8, 0, 0.2, 0]
    )
    np.testing.assert_allclose(context["regions"][0]["center_um"], [8, 8])
    with pytest.raises(ValueError, match="ordered neuron IDs"):
        interpretation_context(metadata, ids[::-1], cache, annotations)
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        interpretation_context(metadata, ids, cache, annotations)


def test_appearance_changes_reuse_recording_but_simulation_changes_do_not(
    tmp_path, monkeypatch
):
    import simtoolreal_shared.anatomical_activity as rendering
    from scripts.run_connectome_evaluation import _run_case

    checkpoint = tmp_path / "checkpoint.pth"
    policy_config = tmp_path / "policy.yaml"
    checkpoint.write_bytes(b"checkpoint")
    policy_config.write_text("policy: example\n")
    rollout = tmp_path / "rollout.mp4"
    rollout.write_bytes(b"footage")
    case = {
        "policy": "actor",
        "metric": "progress",
        "action_selection": "mean",
        "success_tolerance": 0.01,
        "checkpoint_path": str(checkpoint),
        "policy_config_path": str(policy_config),
        "record_video": True,
        "video_path": str(rollout),
        "output_path": str(tmp_path / "eval.json"),
        "label": "case",
        "circuit": {"enabled": True, "group_labels": True},
    }
    metadata = {
        "recording_case": recording_options(case),
        "rollout_sha256": sha256(rollout),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "policy_config_path": str(policy_config),
        "policy_config_sha256": sha256(policy_config),
    }
    np.savez(tmp_path / "activity.npz", metadata=np.asarray(json.dumps(metadata)))
    evaluation = {
        "policy": "actor",
        "metric": "progress",
        "action_selection": "mean",
        "success_tolerance_m": 0.01,
        "checkpoint_path": str(checkpoint),
        "mean_task_progress_pct": 10.0,
    }
    (tmp_path / "eval.json").write_text(json.dumps(evaluation))
    assert recording_matches(case)
    changed = {**case, "circuit": {"enabled": True, "leg_shadows": True}}
    assert recording_matches(changed)
    monkeypatch.setattr(rendering, "render_activity", lambda _: {"status": "complete"})
    result = _run_case(changed, 0, {})
    assert (
        result["circuit"]["status"] == "complete"
    )  # no Isaac Gym or subprocess needed
    assert not recording_matches({**case, "success_tolerance": 0.02})
    checkpoint.write_bytes(b"different checkpoint at same path")
    assert not recording_matches(case)
