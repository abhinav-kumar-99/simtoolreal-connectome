#!/usr/bin/env python3
"""Replay selected saved evaluation cases and render anatomical activity via YAML."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from scripts.run_connectome_evaluation import _completed_case, _run_case
from simtoolreal_shared.activity_trace import circuit_settings, repository_path


def write_video_index(output: Path, results: list[dict], settings: dict) -> None:
    cards = []
    task_keys = [
        (item["evaluation"]["object_name"], item["evaluation"]["task_name"])
        for item in results
    ]
    task_order = {task: i for i, task in enumerate(dict.fromkeys(task_keys))}
    ordered = sorted(
        results,
        key=lambda item: task_order[
            (item["evaluation"]["object_name"], item["evaluation"]["task_name"])
        ],
    )
    for item in ordered:
        evaluation = item["evaluation"]
        directory = Path(item["output_directory"])
        links = []
        video = None
        for name, filename in [
            ("Combined", "rollout_with_circuit.mp4"),
            ("Circuit", "circuit.mp4"),
            ("Activity trace", "activity.npz"),
            ("Metadata", "circuit_render.json"),
        ]:
            path = directory / filename
            if (
                filename == "rollout_with_circuit.mp4"
                and "combined" not in settings["outputs"]
            ):
                continue
            if filename == "circuit.mp4" and "circuit" not in settings["outputs"]:
                continue
            if path.exists():
                relative = html.escape(os.path.relpath(path, output), quote=True)
                links.append(f'<a href="{relative}">{name}</a>')
                if video is None and filename.endswith(".mp4"):
                    video = relative
        metric = evaluation.get("metric")
        label = html.escape(
            f"{evaluation['object_name']} / {evaluation['task_name']}"
            + (f" · {metric}" if metric else "")
        )
        policy = html.escape(evaluation["policy"])
        tolerance = evaluation.get("success_tolerance_m")
        tolerance_text = (
            f" · tolerance {float(tolerance):.8g} m"
            if tolerance is not None
            else ""
        )
        cards.append(
            f'<article><h2>{label}</h2><p>{policy}{tolerance_text}</p><video controls preload="metadata" src="{video}"></video><nav>{" · ".join(links)}</nav></article>'
        )
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>MaleCNS activity videos</title><style>body{background:#0a101b;color:#c3d0db;font:16px system-ui;margin:32px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(480px,100%),1fr));gap:24px}article{background:#111c2c;padding:18px;border-radius:12px}p{overflow-wrap:anywhere;color:#8ca1b6}video{width:100%}a{color:#77ddd3}nav{margin-top:12px}select{background:#111c2c;color:#c3d0db;padding:6px;margin-bottom:20px}</style><h1>MaleCNS anatomical activity</h1>'
    page += f"<p>{len(results)} rollouts · {settings['fps_multiplier']}× rollout FPS · real neuron skeletons and recorded modeled activity</p>"
    notes = ["Cyan/orange show positive/negative controller state."]
    if settings.get("activity_bars"):
        notes.append("Population bars show average state magnitude.")
    if settings.get("leg_shadows"):
        notes.append(
            "Gray leg guides are schematic orientation; the fly legs themselves are not simulated."
        )
    page += f"<p>{' '.join(notes)}</p>"
    page += '<label for="speed">Playback speed </label><select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option></select>'
    page += f"<main>{''.join(cards)}</main>"
    page += '<script>document.getElementById("speed").addEventListener("change",function(){document.querySelectorAll("video").forEach(video=>{video.playbackRate=Number(this.value);});});</script></html>'
    (output / "index.html").write_text(page)


def run(config: dict) -> dict:
    if config.get("schema_version") != 1:
        raise ValueError("Replay schema_version must be 1")
    output = repository_path(config["output_directory"])
    settings = circuit_settings({"enabled": True, **config.get("circuit", {})})
    if not settings["enabled"]:
        raise ValueError("Replay requires circuit.enabled: true")
    if not config.get("case_paths"):
        raise ValueError("Replay requires nonempty case_paths")
    capture_options = {}
    for key, minimum, maximum in [
        ("camera_resolution_reduction_factor", 1, None),
        ("video_quality", 0, 10),
    ]:
        if key in config:
            value = config[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
                or (maximum is not None and value > maximum)
            ):
                raise ValueError(f"Invalid replay {key}: {value!r}")
            capture_options[key] = value
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
        case.update(capture_options)
        directory.mkdir(parents=True, exist_ok=True)
        case_path = directory / "case.yaml"
        previous_case = (
            yaml.safe_load(case_path.read_text()) if case_path.exists() else None
        )
        case_path.write_text(yaml.safe_dump(case, sort_keys=False))
        case.update(
            {
                "case_config_path": str(case_path),
                "log_path": str(directory / "eval.log"),
                "label": f"{case['policy']}:{case['object_name']}:{case['task_name']}",
                "_existing_recording_case": previous_case,
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
    write_video_index(output, results, settings)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(repository_path(args.config).read_text())
    print(json.dumps(run(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
