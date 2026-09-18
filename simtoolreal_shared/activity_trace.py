"""Evaluation activity contracts and a non-invasive recurrent-substep recorder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_RENDER_VERSION = 2


def repository_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recording_options(case: dict) -> dict:
    """Simulation options excluding appearance and output bookkeeping."""
    excluded = {
        "circuit",
        "output_path",
        "video_path",
        "case_config_path",
        "log_path",
        "label",
        "_existing_recording_case",
    }
    options = {key: value for key, value in case.items() if key not in excluded}
    for key in ("checkpoint_path", "policy_config_path", "trajectory_path"):
        if key in options:
            options[key] = str(repository_path(options[key]).resolve())
    return options


def recording_matches(case: dict) -> bool:
    """Allow appearance-only rerenders of a verified existing recording."""
    trace_path = Path(case["output_path"]).parent / "activity.npz"
    rollout_path = Path(case["video_path"])
    try:
        with np.load(trace_path, allow_pickle=False) as trace:
            metadata = json.loads(str(trace["metadata"]))
        previous = metadata.get("recording_case", case.get("_existing_recording_case"))
        if previous is None or recording_options(previous) != recording_options(case):
            return False
        for key in ("checkpoint", "policy_config"):
            path = repository_path(case[f"{key}_path"]).resolve()
            if (
                str(path) != metadata[f"{key}_path"]
                or sha256(path) != metadata[f"{key}_sha256"]
            ):
                return False
        if metadata.get("rollout_sha256"):
            return sha256(rollout_path) == metadata["rollout_sha256"]
        # Legacy traces can be verified against their successful first render.
        render_path = trace_path.parent / "circuit_render.json"
        rendered = (
            json.loads(render_path.read_text())
            if render_path.exists()
            else json.loads(Path(case["output_path"]).read_text())["circuit"]
        )
        return (
            sha256(rollout_path) == rendered["rollout_sha256"]
            and sha256(trace_path) == rendered["trace_sha256"]
        )
    except (OSError, ValueError, KeyError):
        return False


def circuit_settings(value: dict | None) -> dict:
    settings = dict(value or {})
    settings.setdefault("enabled", False)
    settings.setdefault("fps_multiplier", 4)
    settings.setdefault("resolution", [1600, 900])
    settings.setdefault("geometry_cache", "data/connectomes/geometry/malecns_v1")
    settings.setdefault(
        "annotations_path", "data/connectomes/raw/malecns_v1_audit/annotations.feather"
    )
    settings.setdefault("projection", ["x", "z"])
    settings.setdefault("vnc_bounds_um", [240, 560, 400, 1100])
    settings.setdefault("outputs", ["circuit", "combined"])
    for key in ["group_labels", "leg_shadows", "activity_bars"]:
        settings.setdefault(key, False)
        if not isinstance(settings[key], bool):
            raise TypeError(f"circuit.{key} must be a YAML boolean")
    if not isinstance(settings["enabled"], bool):
        raise TypeError("circuit.enabled must be a YAML boolean")
    if not settings["enabled"]:
        return settings
    multiplier = settings["fps_multiplier"]
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, int)
        or multiplier < 1
    ):
        raise ValueError("circuit.fps_multiplier must be a positive integer")
    width, height = settings["resolution"]
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 400 or n % 2
        for n in (width, height)
    ):
        raise ValueError(
            "circuit.resolution must contain even dimensions of at least 400 pixels"
        )
    if settings["projection"] != ["x", "z"]:
        raise ValueError("The anatomical preset currently supports projection: [x, z]")
    bounds = np.asarray(settings["vnc_bounds_um"], dtype=float)
    if (
        bounds.shape != (4,)
        or not np.isfinite(bounds).all()
        or bounds[0] >= bounds[1]
        or bounds[2] >= bounds[3]
    ):
        raise ValueError(
            "vnc_bounds_um must be [xmin, xmax, zmin, zmax] with increasing bounds"
        )
    if len(settings["outputs"]) != len(set(settings["outputs"])):
        raise ValueError("circuit.outputs must contain unique names")
    if not settings["outputs"] or not set(settings["outputs"]) <= {
        "circuit",
        "combined",
    }:
        raise ValueError("circuit.outputs must select circuit and/or combined")
    return settings


def output_paths(directory: Path, settings: dict) -> dict[str, Path]:
    paths = {
        "trace": directory / "activity.npz",
        "metadata": directory / "circuit_render.json",
    }
    if "circuit" in settings["outputs"]:
        paths["circuit"] = directory / "circuit.mp4"
    if "combined" in settings["outputs"]:
        paths["combined"] = directory / "rollout_with_circuit.mp4"
    return paths


def circuit_complete(directory: Path, settings: dict) -> bool:
    """Require matching successful metadata and every requested artifact."""
    if not settings["enabled"]:
        return True
    paths = output_paths(directory, settings)
    if any(not p.is_file() or p.stat().st_size == 0 for p in paths.values()):
        return False
    try:
        metadata = json.loads(paths["metadata"].read_text())
        return (
            metadata.get("status") == "complete"
            and metadata.get("settings") == settings
            and metadata.get("render_version") == ACTIVITY_RENDER_VERSION
        )
    except (ValueError, OSError):
        return False


class ActivityRecorder:
    """Copy states to CPU after each update; the policy never reads these copies."""

    def __init__(self, network, artifact_path: Path):
        if getattr(network, "dynamics_activation", None) != "tanh":
            raise ValueError(
                "Anatomical signed-state recording currently requires a tanh connectome actor"
            )
        if network.backend_options.get("frozen_inference", False):
            raise ValueError(
                "Substep recording is unavailable with frozen_inference/CUDA graphs"
            )
        if getattr(network, "activity_observer", None) is not None:
            raise ValueError("The actor already has an activity observer")
        with np.load(artifact_path, allow_pickle=False) as data:
            self.body_ids = data["body_ids"].copy()
        if len(self.body_ids) != network.neuron_count:
            raise ValueError("Activity neuron count differs from circuit artifact")
        self.network = network
        self.updates = int(network.neural_updates)
        self.states = []
        self.episodes = []
        self.ticks = []
        self.frame_episodes = []
        self.frame_steps = []
        self.episode = 0
        self.step = 0
        self.substep = 0
        network.activity_observer = self.observe

    def _append(self, state, tick: int) -> None:
        array = state.detach().float().cpu().numpy().copy().reshape(-1)
        if len(array) != len(self.body_ids) or not np.isfinite(array).all():
            raise ValueError(
                "Activity recording requires one environment and finite neuron states"
            )
        self.states.append(array)
        self.episodes.append(self.episode)
        self.ticks.append(tick)

    def start_episode(self, episode: int, initial_state) -> None:
        self.episode = episode
        self.step = self.substep = 0
        self._append(initial_state, 0)

    def start_step(self, step: int) -> None:
        self.step = step
        self.substep = 0

    def observe(self, hidden) -> None:
        self.substep += 1
        self._append(hidden, self.step * self.updates + self.substep)

    def finish_step(self) -> None:
        if self.substep != self.updates:
            raise RuntimeError(
                f"Recorded {self.substep} substeps, expected {self.updates}"
            )

    def video_frame(self, step: int) -> None:
        self.frame_episodes.append(self.episode)
        self.frame_steps.append(step)

    def close(self) -> None:
        self.network.activity_observer = None

    def save(self, path: Path, metadata: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.stack(self.states)
        np.savez_compressed(
            path,
            states=values,
            body_ids=self.body_ids,
            episode=np.asarray(self.episodes, dtype=np.int32),
            tick=np.asarray(self.ticks, dtype=np.int64),
            control_step=np.maximum((np.asarray(self.ticks) - 1) // self.updates, 0),
            substep=np.where(
                np.asarray(self.ticks) == 0,
                0,
                (np.asarray(self.ticks) - 1) % self.updates + 1,
            ),
            frame_episode=np.asarray(self.frame_episodes, dtype=np.int32),
            frame_step=np.asarray(self.frame_steps, dtype=np.int64),
            metadata=np.asarray(
                json.dumps(
                    {**metadata, "schema_version": 1, "neural_updates": self.updates}
                )
            ),
        )


def frame_state_indices(trace, multiplier: int, frame_interval: int) -> np.ndarray:
    """Use the latest observed state on the rollout clock, within each episode."""
    metadata = json.loads(str(trace["metadata"]))
    updates = int(metadata["neural_updates"])
    episodes, ticks = trace["episode"], trace["tick"]
    indices = []
    for episode, step in zip(trace["frame_episode"], trace["frame_step"]):
        candidates = np.flatnonzero(episodes == episode)
        if not len(candidates) or ticks[candidates[0]] != 0:
            raise ValueError(f"Episode {episode} is missing its initial activity state")
        local_ticks = ticks[candidates]
        if not np.array_equal(local_ticks, np.arange(len(local_ticks))):
            raise ValueError(
                f"Episode {episode} contains missing or unordered substeps"
            )
        for subframe in range(multiplier):
            # Integer arithmetic avoids choosing the previous tick at an exact boundary.
            target_tick = (
                (int(step) * multiplier + subframe * frame_interval)
                * updates
                // multiplier
            )
            selected = np.searchsorted(local_ticks, target_tick, side="right") - 1
            indices.append(candidates[max(0, selected)])
    return np.asarray(indices, dtype=np.int64)
