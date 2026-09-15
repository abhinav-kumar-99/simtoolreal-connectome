#!/usr/bin/env python3
"""Monitor one trainer and launch a YAML-owned numerical fallback once."""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.util import tensor_util


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported numerical-fallback schema_version")
    return config


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _summary_numbers(event: Event) -> Iterator[tuple[str, float]]:
    if not event.HasField("summary"):
        return
    for value in event.summary.value:
        try:
            if value.HasField("tensor"):
                array = np.asarray(tensor_util.make_ndarray(value.tensor))
                if array.size != 1 or not np.issubdtype(array.dtype, np.number):
                    continue
                number = float(array.reshape(-1)[0])
            elif value.HasField("simple_value"):
                number = float(value.simple_value)
            else:
                continue
        except (TypeError, ValueError):
            continue
        yield value.tag, number


def scan_event_file(
    path: Path, offset: int, monitored_tags: list[str] | None = None
) -> tuple[int, dict[str, dict[str, float | int]], list[dict[str, Any]]]:
    """Scan complete TFRecord events, retaining a partial trailing record."""
    latest: dict[str, dict[str, float | int]] = {}
    nonfinite: list[dict[str, Any]] = []
    if path.stat().st_size < offset:
        offset = 0
    with path.open("rb") as stream:
        stream.seek(offset)
        while True:
            record_start = stream.tell()
            header = stream.read(12)
            if len(header) < 12:
                return record_start, latest, nonfinite
            length = struct.unpack("<Q", header[:8])[0]
            if length > 1_000_000_000:
                raise RuntimeError(f"Implausible TFRecord length {length} at {record_start}")
            payload = stream.read(length)
            footer = stream.read(4)
            if len(payload) < length or len(footer) < 4:
                return record_start, latest, nonfinite
            event = Event()
            event.ParseFromString(payload)
            for tag, number in _summary_numbers(event):
                if monitored_tags and not any(
                    fnmatch.fnmatchcase(tag, pattern) for pattern in monitored_tags
                ):
                    continue
                if not math.isfinite(number):
                    nonfinite.append(
                        {"file": str(path), "step": int(event.step), "tag": tag,
                         "value": repr(number)}
                    )
                else:
                    latest[tag] = {"step": int(event.step), "value": number}


def _tensor_items(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensor_items(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensor_items(item, f"{prefix}.{index}" if prefix else str(index))


def checkpoint_health(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get(0, checkpoint) if isinstance(checkpoint, dict) else checkpoint
    bad = []
    for section in ("model", "optimizer", "assymetric_vf_nets", "central_value_optimizer"):
        if not isinstance(state, dict) or section not in state:
            continue
        for name, tensor in _tensor_items(state[section], section):
            if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
                bad.append(name)
                if len(bad) >= 20:
                    break
    return {
        "path": str(path),
        "epoch": int(state.get("epoch", -1)),
        "frame": int(state.get("frame", -1)),
        "nonfinite_tensors": bad,
    }


def _process_table() -> dict[int, tuple[int, str]]:
    table = {}
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            status = (process / "status").read_text()
            parent_line = next(line for line in status.splitlines() if line.startswith("PPid:"))
            table[int(process.name)] = (int(parent_line.split()[1]), command)
        except (OSError, UnicodeDecodeError, StopIteration, ValueError):
            continue
    return table


def _matching_pids(marker: str) -> list[int]:
    return sorted(pid for pid, (_, command) in _process_table().items() if marker in command)


def _process_tree(roots: list[int]) -> list[int]:
    table = _process_table()
    selected = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, (parent, _) in table.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return sorted(selected, reverse=True)


def _stop_processes(roots: list[int], timeout_seconds: float) -> list[int]:
    targets = _process_tree(roots)
    for pid in targets:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        remaining = [pid for pid in targets if Path(f"/proc/{pid}").exists()]
        if not remaining:
            return targets
        time.sleep(1)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + min(timeout_seconds, 30)
    while time.monotonic() < deadline:
        remaining = [pid for pid in targets if Path(f"/proc/{pid}").exists()]
        if not remaining:
            return targets
        time.sleep(1)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return targets


def _launch(config_path: Path, script_name: str, log_path: Path) -> int:
    command = [sys.executable, str(REPOSITORY_ROOT / "scripts" / script_name),
               "--config", str(config_path)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(
            command, cwd=REPOSITORY_ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return int(process.pid)


def _launch_fallback(
    config: dict[str, Any], state: dict[str, Any], reason: dict[str, Any]
) -> None:
    source = config["source"]
    stopped: dict[str, list[int]] = {}
    trainer_pids = _matching_pids(str(source["trainer_process_marker"]))
    if trainer_pids:
        stopped["trainer"] = _stop_processes(
            trainer_pids, float(config.get("shutdown_timeout_seconds", 60))
        )
    for marker in config.get("stop_process_markers", []):
        pids = _matching_pids(str(marker))
        if pids:
            stopped[str(marker)] = _stop_processes(
                pids, float(config.get("shutdown_timeout_seconds", 60))
            )
    output = _repository_path(config["output_directory"])
    training_config = _repository_path(config["fallback"]["training_config"])
    video_config = _repository_path(config["fallback"]["video_config"])
    training_pid = _launch(
        training_config, "run_connectome_suite.py", output / "fallback_training.log"
    )
    state.update(
        {
            "status": "fallback_starting",
            "triggered_at_utc": datetime.now(timezone.utc).isoformat(),
            "trigger": reason,
            "stopped_pids": stopped,
            "fallback_training_pid": training_pid,
            "fallback_training_config": str(training_config),
            "fallback_video_config": str(video_config),
        }
    )
    _write_json(output / "status.json", state)
    marker = str(config["fallback"]["trainer_process_marker"])
    deadline = time.monotonic() + float(
        config["fallback"].get("startup_timeout_seconds", 120)
    )
    while time.monotonic() < deadline:
        fallback_pids = _matching_pids(marker)
        if fallback_pids:
            video_pid = _launch(
                video_config, "run_connectome_milestone_evaluation.py",
                output / "fallback_videos.log",
            )
            state.update(
                {
                    "status": "fallback_launched",
                    "fallback_trainer_pids": fallback_pids,
                    "fallback_video_pid": video_pid,
                    "fallback_started_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _write_json(output / "status.json", state)
            return
        if not Path(f"/proc/{training_pid}").exists():
            break
        time.sleep(1)
    state["status"] = "fallback_failed_to_start"
    state["fallback_failure_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output / "status.json", state)


def run(config: dict[str, Any]) -> dict[str, Any]:
    output = _repository_path(config["output_directory"])
    state_path = output / "status.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        "schema_version": 1,
        "status": "monitoring",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_offsets": {},
        "latest_scalars": {},
    }
    if state.get("status") in {
        "fallback_launched", "fallback_failed_to_start", "source_complete"
    }:
        return state

    source = config["source"]
    event_glob = str(_repository_path(source["event_glob"]))
    checkpoint_glob = str(_repository_path(source["checkpoint_glob"]))
    expected_final_frame = int(source["expected_final_frame"])
    missing_since: float | None = None
    checked_checkpoint: tuple[str, int] | None = None
    poll_seconds = float(config.get("poll_interval_seconds", 30))
    monitored_tags = [str(tag) for tag in source["monitored_scalar_tags"]]

    while True:
        for event_path in sorted(Path("/").glob(event_glob.lstrip("/"))):
            key = str(event_path)
            offset = int(state["event_offsets"].get(key, 0))
            offset, latest, nonfinite = scan_event_file(
                event_path, offset, monitored_tags
            )
            state["event_offsets"][key] = offset
            state["latest_scalars"].update(latest)
            if nonfinite:
                reason = {"kind": "nonfinite_event_scalar", "events": nonfinite[:20]}
                _launch_fallback(config, state, reason)
                return state

        checkpoints = sorted(
            Path("/").glob(checkpoint_glob.lstrip("/")),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if checkpoints:
            latest_path = checkpoints[-1]
            identity = (str(latest_path), latest_path.stat().st_mtime_ns)
            if identity != checked_checkpoint:
                try:
                    health = checkpoint_health(latest_path)
                    state["latest_checkpoint"] = health
                    checked_checkpoint = identity
                    if health["nonfinite_tensors"]:
                        _launch_fallback(
                            config, state,
                            {"kind": "nonfinite_checkpoint", "checkpoint": health},
                        )
                        return state
                except (OSError, RuntimeError, EOFError) as error:
                    state["checkpoint_read_warning"] = f"{type(error).__name__}: {error}"

        trainer_pids = _matching_pids(str(source["trainer_process_marker"]))
        state["trainer_pids"] = trainer_pids
        state["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        if trainer_pids:
            missing_since = None
        else:
            latest_frame = int(state.get("latest_checkpoint", {}).get("frame", -1))
            if latest_frame >= expected_final_frame:
                state["status"] = "source_complete"
                state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(state_path, state)
                return state
            if missing_since is None:
                missing_since = time.monotonic()
            elif time.monotonic() - missing_since >= float(
                source.get("missing_process_grace_seconds", 90)
            ):
                _launch_fallback(
                    config, state,
                    {"kind": "trainer_exit", "latest_checkpoint_frame": latest_frame},
                )
                return state
        _write_json(state_path, state)
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    try:
        result = run(_load_yaml(path))
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    except KeyboardInterrupt:
        print("Numerical fallback monitor stopped by operator", flush=True)


if __name__ == "__main__":
    main()
