"""Real MaleCNS skeleton geometry and cached CPU rasterization for activity movies."""

from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import io
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import sparse

from simtoolreal_shared.activity_trace import (
    ACTIVITY_RENDER_VERSION,
    circuit_settings,
    frame_state_indices,
    output_paths,
    repository_path,
    sha256,
)

SWC_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{body_id}.swc"
BACKGROUND = np.asarray([10, 16, 27], dtype=np.float32)
CYAN = np.asarray([77, 223, 215], dtype=np.float32)
ORANGE = np.asarray([255, 145, 77], dtype=np.float32)


def parse_swc(text: str) -> np.ndarray:
    """Return parent-child segments in micrometers, preserving source geometry."""
    rows = np.loadtxt(io.StringIO(text), comments="#", ndmin=2)
    if rows.shape[1] != 7 or not np.isfinite(rows).all():
        raise ValueError("SWC must contain seven finite numeric columns")
    ids = rows[:, 0].astype(np.int64)
    parents = rows[:, 6].astype(np.int64)
    if not np.array_equal(ids, rows[:, 0]) or not np.array_equal(parents, rows[:, 6]):
        raise ValueError("SWC node and parent IDs must be integers")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("SWC contains duplicate node IDs")
    lookup = {int(node): i for i, node in enumerate(ids)}
    children = np.flatnonzero(parents != -1)
    if not len(children) or any(int(parents[i]) not in lookup for i in children):
        raise ValueError("SWC contains missing parent IDs or no skeleton segments")
    parent_rows = np.asarray([lookup[int(parents[i])] for i in children])
    if np.any(parent_rows == children):
        raise ValueError("SWC contains self-parented nodes")
    # Official native SWC coordinates are in 8 nm units; no per-neuron warping.
    points = rows[:, 2:5].astype(np.float32) * 0.008
    return np.stack([points[children], points[parent_rows]], axis=1)


def ensure_geometry(body_ids: np.ndarray, cache: Path) -> dict:
    """Fetch exactly the requested body IDs; no neuPrint credentials are needed."""
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "geometry.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _ensure_geometry_unlocked(body_ids, cache)


def _ensure_geometry_unlocked(body_ids: np.ndarray, cache: Path) -> dict:
    manifest_path = cache / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {
            "dataset": "male-cns:v1.0",
            "source_template": SWC_URL,
            "source_coordinate_units": "8 nm",
            "display_units": "micrometers",
            "license": "CC BY 4.0",
            "files": {},
        }
    )

    def fetch(body_id):
        body_id = int(body_id)
        path = cache / f"{body_id}.swc"
        previous = manifest["files"].get(str(body_id))
        if path.exists() and previous and sha256(path) == previous["sha256"]:
            return str(body_id), previous
        error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(
                    SWC_URL.format(body_id=body_id), timeout=30
                ) as response:
                    content = response.read()
                segments = parse_swc(content.decode("utf-8"))
                temporary = path.with_suffix(".swc.tmp")
                temporary.write_bytes(content)
                temporary.replace(path)
                return str(body_id), {
                    "url": SWC_URL.format(body_id=body_id),
                    "sha256": sha256(path),
                    "bytes": len(content),
                    "segments": len(segments),
                }
            except (OSError, ValueError) as exc:
                error = exc
                if attempt < 2:
                    time.sleep(0.25 * (attempt + 1))
        raise RuntimeError(
            f"Missing/invalid anatomical geometry for bodyId {body_id}: {error}"
        )

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch, body_id): int(body_id) for body_id in body_ids
        }
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                key, entry = future.result()
                manifest["files"][key] = entry
            except RuntimeError as exc:
                errors.append(str(exc))
            if count % 128 == 0 or count == len(futures):
                print(
                    f"Anatomical geometry: {count}/{len(futures)} neurons checked",
                    flush=True,
                )
                temporary = manifest_path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )
                temporary.replace(manifest_path)
    if errors:
        raise RuntimeError("Anatomical coverage failed:\n" + "\n".join(errors))
    return manifest


def _soma_points(path: Path) -> np.ndarray:
    import pandas as pd

    annotations = pd.read_feather(path, columns=["somaLocation"])
    points = [
        point
        for point in annotations.somaLocation
        if point is not None and np.shape(point) == (3,)
    ]
    return np.stack(points).astype(np.float32)[:, [0, 2]] * 0.008


def project(points: np.ndarray, bounds, width: int, height: int) -> np.ndarray:
    """Equal-scale X-Z projection; positive Z points down toward the VNC."""
    xmin, xmax, zmin, zmax = bounds
    scale = min((width - 12) / (xmax - xmin), (height - 12) / (zmax - zmin))
    center = np.asarray([(xmin + xmax) / 2, (zmin + zmax) / 2])
    return (points - center) * scale + np.asarray([width / 2, height / 2])


class AnatomicalPanel:
    """Pre-rasterize each neuron once, then blend overlapping signed activity."""

    def __init__(
        self,
        body_ids,
        cache: Path,
        annotations: Path,
        width: int,
        height: int,
        vnc_bounds,
        interpretation: dict | None = None,
        settings: dict | None = None,
    ):
        self.width, self.height = width, height
        somas = _soma_points(annotations)
        xmin, zmin = somas.min(axis=0) - 20
        xmax, zmax = somas.max(axis=0) + 20
        overview_bounds = [float(xmin), float(xmax), float(zmin), float(zmax)]
        # Fixed bounds from the dataset, never from activity or the current actor.
        self.bounds = [overview_bounds, list(vnc_bounds)]
        self.views = [
            (8, 0, width // 2 - 12, height),
            (width // 2 + 4, 0, width // 2 - 12, height),
        ]
        provenance = {
            "body_ids": [int(x) for x in body_ids],
            "width": width,
            "height": height,
            "bounds": self.bounds,
            "annotations_sha256": sha256(annotations),
            "raster_version": 1,
        }
        manifest = json.loads((cache / "manifest.json").read_text())
        provenance["skeleton_hashes"] = [
            manifest["files"][str(int(x))]["sha256"] for x in body_ids
        ]
        key = hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()
        ).hexdigest()[:20]
        mask_path = cache / f"projection_{key}.npz"
        if mask_path.exists():
            self.mask = sparse.load_npz(mask_path)
        else:
            rows, columns, weights = [], [], []
            for neuron, body_id in enumerate(body_ids):
                segments = parse_swc((cache / f"{int(body_id)}.swc").read_text())[
                    :, :, [0, 2]
                ]
                canvas = np.zeros((height, width), dtype=np.uint8)
                for bounds, (x, y, w, h) in zip(self.bounds, self.views):
                    projected = np.rint(project(segments, bounds, w, h)).astype(
                        np.int32
                    )
                    # Polylines retain all source segments. Clip within the view ROI.
                    cv2.polylines(
                        canvas[y : y + h, x : x + w],
                        projected,
                        False,
                        255,
                        1,
                        cv2.LINE_AA,
                    )
                pixels = np.flatnonzero(canvas)
                rows.append(pixels.astype(np.int32))
                columns.append(np.full(len(pixels), neuron, dtype=np.int32))
                weights.append(canvas.reshape(-1)[pixels].astype(np.float32) / 255)
                if (neuron + 1) % 256 == 0:
                    print(
                        f"Anatomical projection {width}px: {neuron + 1}/{len(body_ids)}",
                        flush=True,
                    )
            self.mask = sparse.coo_matrix(
                (
                    np.concatenate(weights),
                    (np.concatenate(rows), np.concatenate(columns)),
                ),
                shape=(width * height, len(body_ids)),
                dtype=np.float32,
            ).tocsr()
            # Parallel evaluator queues must never read a partially written cache.
            with tempfile.NamedTemporaryFile(
                dir=cache, suffix=".npz", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                sparse.save_npz(temporary_path, self.mask)
                temporary_path.replace(mask_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        self.density = np.asarray(self.mask.sum(axis=1)).reshape(-1).astype(np.float32)
        self.background = np.broadcast_to(BACKGROUND, (height, width, 3)).copy()
        for bounds, (x, y, w, h) in zip(self.bounds, self.views):
            points = np.rint(project(somas, bounds, w, h)).astype(int)
            valid = (
                (points[:, 0] >= 0)
                & (points[:, 0] < w)
                & (points[:, 1] >= 0)
                & (points[:, 1] < h)
            )
            p = points[valid]
            self.background[p[:, 1] + y, p[:, 0] + x] = [42, 53, 68]
        # Neutral morphology remains visible even for initially zero model state.
        neutral = (1 - np.exp(-0.12 * self.density)).reshape(height, width, 1)
        self.background += neutral * np.asarray([27, 36, 43], dtype=np.float32)
        self.denominator = np.maximum(np.sqrt(self.density), 1)
        self.label_overlay = None
        if interpretation is not None:
            from simtoolreal_shared.activity_interpretation import overlay_layers

            shadows, self.label_overlay = overlay_layers(
                interpretation, self.bounds, self.views, width, height, settings
            )
            image = Image.fromarray(
                np.clip(self.background, 0, 255).astype(np.uint8)
            ).convert("RGBA")
            self.background = np.asarray(
                Image.alpha_composite(image, shadows).convert("RGB")
            ).astype(np.float32)

    def render(self, states: np.ndarray) -> np.ndarray:
        activity = np.clip(states, -1, 1).astype(np.float32)
        positive = self.mask @ np.maximum(activity, 0)
        negative = self.mask @ np.maximum(-activity, 0)
        total = positive + negative
        colors = (positive[:, None] * CYAN + negative[:, None] * ORANGE) / np.maximum(
            total[:, None], 1e-8
        )
        opacity = (1 - np.exp(-0.9 * total / self.denominator))[:, None]
        pixels = self.background.reshape(-1, 3)
        result = (
            np.clip(pixels * (1 - opacity) + colors * opacity, 0, 255)
            .astype(np.uint8)
            .reshape(self.height, self.width, 3)
        )
        if self.label_overlay is not None:
            result = np.asarray(
                Image.alpha_composite(
                    Image.fromarray(result).convert("RGBA"), self.label_overlay
                ).convert("RGB")
            )
        return result


def _font(size: int):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def compose_frame(
    panel: np.ndarray,
    robot: np.ndarray | None,
    width: int,
    height: int,
    metadata: dict,
    seconds: float,
    episode: int,
    activity_values: list[float] | None = None,
) -> np.ndarray:
    canvas = Image.new("RGB", (width, height), tuple(BACKGROUND.astype(int)))
    draw = ImageDraw.Draw(canvas)
    unit = height / 900
    pad = int(24 * unit)

    def text(x, y, label, size=19, color=(195, 208, 219)):
        draw.text(
            (int(x), int(y * unit)),
            label,
            font=_font(max(12, int(size * unit))),
            fill=color,
        )

    text(pad, 20, "MALE CNS / DEXTEROUS TOOL CONTROL", 28)
    interpretation = metadata.get("interpretation")
    subtitle = "Real neuron anatomy  ·  recorded modeled activity  ·  synchronized robot motion"
    if interpretation is not None:
        subtitle = (
            "Robot sensing → sensory cells   ·   Task goals → descending cells   ·   "
            + interpretation["readout_label"]
        )
    text(
        pad,
        64,
        subtitle,
        16,
        (126, 149, 169),
    )
    top, bottom = (
        int(136 * unit),
        int((675 if interpretation is not None else 735) * unit),
    )
    if robot is not None:
        left_width = width // 2 - 2 * pad
        text(pad, 105, "01  ROBOT ROLLOUT", 17, (116, 214, 207))
        image = Image.fromarray(robot)
        image.thumbnail((left_width, bottom - top), Image.Resampling.LANCZOS)
        canvas.paste(
            image,
            (
                pad + (left_width - image.width) // 2,
                top + (bottom - top - image.height) // 2,
            ),
        )
        panel_x = width // 2 + pad
        panel_width = width // 2 - 2 * pad
    else:
        panel_x, panel_width = pad, width - 2 * pad
    text(panel_x, 105, "02  MALE CNS ANATOMICAL ACTIVITY", 17, (116, 214, 207))
    canvas.paste(Image.fromarray(panel), (panel_x, top + int(29 * unit)))
    text(panel_x + 10, 140, "FULL CNS CONTEXT", 15)
    text(panel_x + panel_width // 2 + 10, 140, "VNC DETAIL", 15)
    if activity_values is not None:
        column_width = (width - 2 * pad) / 4
        for i, (group, strength) in enumerate(
            zip(interpretation["groups"], activity_values)
        ):
            x = pad + i * column_width
            text(x, 688, f"{group['label']} ({len(group['indices']):,})", 16)
            text(x, 709, group["role"], 13, (126, 149, 169))
            bar_x, bar_y = int(x), int(733 * unit)
            bar_width = int(column_width - 95 * unit)
            draw.rectangle(
                (bar_x, bar_y, bar_x + bar_width, bar_y + int(8 * unit)),
                fill=(37, 51, 69),
            )
            draw.rectangle(
                (
                    bar_x,
                    bar_y,
                    bar_x + int(bar_width * np.clip(strength, 0, 1)),
                    bar_y + int(8 * unit),
                ),
                fill=(135, 169, 196),
            )
            text(bar_x + bar_width + 8, 725, f"{strength:.2f}", 14)
    text(
        pad,
        755,
        f"{metadata['policy']}  |  {metadata['object_name']} / {metadata['task_name']}",
        17,
    )
    text(
        pad,
        790,
        f"TIME {seconds:05.2f}s   ·   EPISODE {episode + 1}   ·   {metadata['neuron_count']:,} neurons / {metadata['edge_count']:,} connections",
        17,
    )
    text(
        pad,
        828,
        "Cyan = positive state   ·   Orange = negative state   ·   Bars = mean |state| (0–1)"
        if activity_values is not None
        else "Cyan = positive state     Orange = negative state     Fixed state scale: −1 to +1",
        16,
    )
    text(
        pad,
        856,
        "Gray legs = schematic orientation. Gray CNS = dataset somas. Lines label cell groups on both sides."
        if interpretation is not None
        else "Gray = dataset somas / inactive morphology. One modeled state per neuron; overlap colors blend.",
        14,
        (126, 149, 169),
    )
    text(
        pad,
        880,
        "Male CNS v1.0 · FlyEM / Janelia · CC BY 4.0     |     EM X–Z projection · uniform anatomical scale",
        12,
        (126, 149, 169),
    )
    return np.asarray(canvas)


def probe_video(path: Path) -> dict:
    result = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(result)["streams"][0]


def render_activity(config: dict) -> dict:
    settings = circuit_settings({"enabled": True, **config.get("circuit", {})})
    if not settings["enabled"]:
        raise ValueError("Rendering requires circuit.enabled: true")
    trace_path = repository_path(config["trace_path"])
    rollout_path = repository_path(config["rollout_path"])
    directory = repository_path(config.get("output_directory", str(trace_path.parent)))
    directory.mkdir(parents=True, exist_ok=True)
    paths = output_paths(directory, settings)
    # Trace output is owned by recording, and can live outside the render directory.
    paths["trace"] = trace_path
    if paths["metadata"].exists():
        paths["metadata"].unlink()
    with np.load(trace_path, allow_pickle=False) as trace:
        metadata = json.loads(str(trace["metadata"]))
        states = trace["states"]
        body_ids = trace["body_ids"]
        if (
            states.ndim != 2
            or states.shape[1] != len(body_ids)
            or not np.isfinite(states).all()
        ):
            raise ValueError("Trace must contain finite states matching its neuron IDs")
        if len(np.unique(body_ids)) != len(body_ids):
            raise ValueError("Trace contains duplicate neuron IDs")
        video_info = probe_video(rollout_path)
        if int(video_info["nb_frames"]) != len(trace["frame_step"]):
            raise ValueError(
                "Rollout frame count differs from recorded camera-frame timestamps"
            )
        video_fps = int(metadata["video_fps"])
        a, b = video_info["r_frame_rate"].split("/")
        if abs(int(a) / int(b) - video_fps) > 1e-6:
            raise ValueError("Rollout FPS differs from trace video clock")
        multiplier = settings["fps_multiplier"]
        indices = frame_state_indices(
            trace, multiplier, int(metadata["video_frame_interval"])
        )
        frame_episodes = np.repeat(trace["frame_episode"], multiplier)
        frame_steps = np.repeat(trace["frame_step"], multiplier)
        cache = repository_path(settings["geometry_cache"])
        geometry = ensure_geometry(body_ids, cache)
        context = None
        if any(
            settings[key] for key in ["group_labels", "leg_shadows", "activity_bars"]
        ):
            from simtoolreal_shared.activity_interpretation import (
                interpretation_context,
            )

            context = interpretation_context(
                metadata, body_ids, cache, repository_path(settings["annotations_path"])
            )
            metadata["interpretation"] = context
        width, height = settings["resolution"]
        pad = int(24 * height / 900)
        panel_height = int(
            ((675 if context is not None else 735) - 136 - 29) * height / 900
        )
        panels = {}
        for name in settings["outputs"]:
            panel_width = width - 2 * pad if name == "circuit" else width // 2 - 2 * pad
            panels[name] = AnatomicalPanel(
                body_ids,
                cache,
                repository_path(settings["annotations_path"]),
                panel_width,
                panel_height,
                settings["vnc_bounds_um"],
                context,
                settings,
            )
        fps = video_fps * multiplier
        metadata["neuron_count"] = len(body_ids)
        writers = {}
        started = time.monotonic()
        try:
            for name in settings["outputs"]:
                writers[name] = imageio.get_writer(
                    paths[name],
                    fps=fps,
                    codec="libx264",
                    macro_block_size=2,
                    quality=8,
                    ffmpeg_params=["-threads", "2"],
                )
            reader = imageio.get_reader(rollout_path)
            try:
                for frame, robot in enumerate(reader):
                    for subframe in range(multiplier):
                        n = frame * multiplier + subframe
                        seconds = (
                            int(frame_steps[n])
                            + subframe * metadata["video_frame_interval"] / multiplier
                        ) / (video_fps * metadata["video_frame_interval"])
                        strengths = None
                        if context is not None and settings["activity_bars"]:
                            from simtoolreal_shared.activity_interpretation import (
                                population_activity,
                            )

                            strengths = population_activity(context, states[indices[n]])
                        for name, writer in writers.items():
                            panel = panels[name].render(states[indices[n]])
                            writer.append_data(
                                compose_frame(
                                    panel,
                                    robot if name == "combined" else None,
                                    width,
                                    height,
                                    metadata,
                                    seconds,
                                    int(frame_episodes[n]),
                                    strengths,
                                )
                            )
                    if (frame + 1) % 40 == 0:
                        print(
                            f"Rendered {(frame + 1) * multiplier}/{len(indices)} frames ({time.monotonic() - started:.1f}s)",
                            flush=True,
                        )
            finally:
                reader.close()
        finally:
            for writer in writers.values():
                writer.close()
        probes = {name: probe_video(paths[name]) for name in writers}
        for name, probe in probes.items():
            a, b = probe["r_frame_rate"].split("/")
            if int(probe["nb_frames"]) != len(indices) or int(a) / int(b) != fps:
                raise RuntimeError(
                    f"Encoded {name} video has incorrect FPS/frame count: {probe}"
                )
        result = {
            "status": "complete",
            "schema_version": 1,
            "render_version": ACTIVITY_RENDER_VERSION,
            "settings": settings,
            "trace_path": str(trace_path),
            "trace_sha256": sha256(trace_path),
            "rollout_path": str(rollout_path),
            "rollout_sha256": sha256(rollout_path),
            "fps": fps,
            "frames": len(indices),
            "duration_seconds": len(indices) / fps,
            "neuron_count": len(body_ids),
            "geometry_coverage": len(body_ids),
            "dataset": geometry["dataset"],
            "geometry_manifest": str(cache / "manifest.json"),
            "projection": settings["projection"],
            "state_semantics": "one signed tanh model state per whole neuron skeleton",
            "sampling": "latest recorded neural substep on rollout video clock; terminal hold; episode-local",
            "source": metadata,
            "outputs": {name: str(paths[name]) for name in writers},
            "video_probes": probes,
            "render_seconds": time.monotonic() - started,
        }
        paths["metadata"].write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        return result
