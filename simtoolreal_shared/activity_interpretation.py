"""Source-grounded population labels and explicitly schematic leg orientation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

from simtoolreal_shared.activity_trace import repository_path, sha256


def population_anchors(centers: np.ndarray, sides: np.ndarray) -> list[dict]:
    """Keep separate source-annotated sides of a multimodal population."""
    anchors = []
    for side in ["L", "R", "unknown"]:
        mask = sides == side if side != "unknown" else ~np.isin(sides, ["L", "R"])
        if np.any(mask):
            anchors.append(
                {
                    "side": side,
                    "count": int(np.count_nonzero(mask)),
                    "center_um": np.median(centers[mask], axis=0).tolist(),
                }
            )
    return anchors


def interpretation_context(
    metadata: dict, body_ids: np.ndarray, cache: Path, annotations: Path
) -> dict:
    from simtoolreal_shared.anatomical_activity import parse_swc

    artifact = repository_path(metadata["artifact_path"])
    if sha256(artifact) != metadata["artifact_sha256"]:
        raise ValueError("Circuit artifact changed since activity recording")
    with np.load(artifact, allow_pickle=False) as data:
        if not np.array_equal(data["body_ids"], body_ids):
            raise ValueError(
                "Population labels must match the trace's ordered neuron IDs"
            )
        annotations_frame = pd.read_feather(annotations)
        selected_annotations = annotations_frame.set_index("bodyId").reindex(body_ids)
        groups = []
        covered = []
        for key, label, role in [
            ("sensory_indices", "Sensory input", "Robot sensing"),
            ("descending_indices", "Descending", "Task goal / context"),
            ("motor_indices", "Motor cells", "Front-leg motor cells"),
        ]:
            indices = data[key].astype(int)
            covered.extend(indices.tolist())
            centers = []
            for body_id in body_ids[indices]:
                segments = parse_swc((cache / f"{int(body_id)}.swc").read_text())
                centers.append(np.median(segments.reshape(-1, 3)[:, [0, 2]], axis=0))
            side_key = "rootSide" if key == "sensory_indices" else "somaSide"
            sides = (
                selected_annotations.iloc[indices]
                .get(side_key, pd.Series("unknown", index=np.arange(len(indices))))
                .fillna("unknown")
                .to_numpy()
            )
            groups.append(
                {
                    "label": label,
                    "role": role,
                    "indices": indices.tolist(),
                    "center_um": np.mean(centers, axis=0).tolist() if centers else None,
                    "anchors": population_anchors(np.asarray(centers), sides)
                    if centers
                    else [],
                }
            )
        groups.append(
            {
                "label": "Other circuit cells",
                "role": "Processing / feedback",
                "indices": np.setdiff1d(np.arange(len(body_ids)), covered).tolist(),
                "center_um": None,
            }
        )
    motor_rows = selected_annotations.iloc[groups[2]["indices"]]
    front_motor = len(motor_rows) > 0 and motor_rows.somaNeuromere.eq("T1").all()
    groups[2]["role"] = "Front-leg motor cells" if front_motor else "Motor activity"
    sensory_rows = selected_annotations.iloc[groups[0]["indices"]]
    touch_position = (
        len(sensory_rows) > 0
        and sensory_rows["class"].str.startswith("mechanosensory", na=False).all()
    )
    regions = []
    for name, leg in [("T1", "FRONT LEGS"), ("T2", "MIDDLE LEGS"), ("T3", "HIND LEGS")]:
        locations = annotations_frame.loc[
            annotations_frame.somaNeuromere.eq(name), "somaLocation"
        ].dropna()
        if not len(locations):
            continue
        center = np.median(np.stack(locations), axis=0)[[0, 2]] * 0.008
        regions.append({"name": name, "leg": leg, "center_um": center.tolist()})
    if not regions:
        raise ValueError("Anatomical labels require T1/T2/T3 soma annotations")
    readout = metadata.get("readout_feature_population")
    if readout is None and metadata.get("policy_config_path"):
        path = repository_path(metadata["policy_config_path"])
        if sha256(path) == metadata["policy_config_sha256"]:
            config = yaml.safe_load(path.read_text())
            readout = (
                config["train"]["params"]["network"]["connectome"]
                .get("reservoir_readout", {})
                .get("feature_population", "motor")
            )
    return {
        "groups": groups,
        "regions": regions,
        "motor_detail": (
            "Both front legs"
            if {a["side"] for a in groups[2]["anchors"]} >= {"L", "R"}
            else "Front-leg circuit"
        )
        if front_motor
        else "Selected motor cells",
        "sensory_detail": "Touch + position cells"
        if touch_position
        else "Selected sensory cells",
        "readout_population": readout,
        "readout_label": f"All {len(body_ids):,} cells feed robot actions"
        if readout == "all"
        else "Fly activity feeds learned robot actions",
        "leg_semantics": "schematic orientation only; attachment levels estimated from T1/T2/T3 dataset soma medians",
        "callout_semantics": "leaders label populations using separate rootSide/somaSide arbor anchors; they are not injection sites or activity-flow paths",
    }


def overlay_layers(
    context: dict, bounds: list, views: list, width: int, height: int, settings: dict
):
    from simtoolreal_shared.anatomical_activity import project

    shadows = Image.new("RGBA", (width, height))
    labels = Image.new("RGBA", (width, height))
    ghost, draw = ImageDraw.Draw(shadows), ImageDraw.Draw(labels)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    strong = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)

    def boxed_text(x, y, text, bold=False):
        selected = strong if bold else font
        box = draw.textbbox((x, y), text, font=selected)
        draw.rectangle(
            (box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(10, 16, 27, 215)
        )
        draw.text((x, y), text, font=selected, fill=(209, 219, 231, 255))

    def leaders(start, targets, vx, vy):
        for ax, ay in targets:
            end = (vx + int(ax), vy + int(ay))
            draw.line((*start, *end), fill=(209, 219, 231, 180), width=1)
            draw.ellipse(
                (end[0] - 2, end[1] - 2, end[0] + 2, end[1] + 2),
                fill=(209, 219, 231, 230),
            )

    for view_index, (view_bounds, (vx, vy, vw, vh)) in enumerate(zip(bounds, views)):
        for region in context["regions"]:
            cx, cy = project(np.array(region["center_um"]), view_bounds, vw, vh)
            if settings["leg_shadows"]:
                for side in [-1, 1]:
                    spread = [-55, 40, 45][int(region["name"][-1]) - 1]
                    points = [
                        (cx + side * 0.15 * vw, cy),
                        (cx + side * 0.28 * vw, cy + spread * 0.3),
                        (cx + side * 0.36 * vw, cy + spread),
                        (cx + side * 0.46 * vw, cy + spread * 1.25),
                    ]
                    points = [
                        (
                            int(np.clip(x, 5, vw - 5)) + vx,
                            int(np.clip(y, 5, vh - 5)) + vy,
                        )
                        for x, y in points
                    ]
                    ghost.line(
                        points,
                        fill=(78, 91, 111, 65),
                        width=12 if view_index else 8,
                        joint="curve",
                    )
                    ghost.line(points, fill=(108, 122, 143, 70), width=2, joint="curve")
            if view_index == 1 and settings["group_labels"]:
                boxed_text(
                    vx + 6, int(cy) + vy + 14, f"{region['name']}  {region['leg']}"
                )
                draw.line(
                    (vx + 6, int(cy) + vy + 9, vx + vw - 6, int(cy) + vy + 9),
                    fill=(128, 148, 170, 110),
                    width=1,
                )
        if view_index == 0 and settings["group_labels"]:
            anchors = {}
            centers = {}
            for group in context["groups"][:3]:
                if group["center_um"] is not None:
                    centers[group["label"]] = project(
                        np.array(group["center_um"]), view_bounds, vw, vh
                    )
                    anchors[group["label"]] = [
                        project(np.array(a["center_um"]), view_bounds, vw, vh)
                        for a in group["anchors"]
                    ]
            if "Descending" in anchors:
                _, ay = centers["Descending"]
                boxed_text(vx + 8, int(ay) + vy - 49, "BRAIN", True)
                boxed_text(vx + 8, int(ay) + vy - 27, "Descending / goal input")
                leaders((vx + 16, int(ay) + vy - 4), anchors["Descending"], vx, vy)
            if "Sensory input" in anchors:
                _, ay = centers["Sensory input"]
                boxed_text(vx + 6, int(ay) + vy + 19, "SENSORY INPUT", True)
                boxed_text(vx + 6, int(ay) + vy + 39, context["sensory_detail"])
                leaders((vx + 24, int(ay) + vy + 17), anchors["Sensory input"], vx, vy)
            if "Motor cells" in anchors:
                _, ay = centers["Motor cells"]
                tx = vx + max(6, vw - 117)
                boxed_text(tx, int(ay) + vy + 76, "MOTOR CELLS", True)
                boxed_text(tx, int(ay) + vy + 97, context["motor_detail"])
                leaders((tx + 20, int(ay) + vy + 73), anchors["Motor cells"], vx, vy)
            _, tail_y = project(
                np.array([(view_bounds[0] + view_bounds[1]) / 2, view_bounds[3]]),
                view_bounds,
                vw,
                vh,
            )
            boxed_text(
                vx + 8,
                min(height - 27, int(tail_y) + vy + 12),
                "VENTRAL NERVE CORD",
                True,
            )
    return shadows, labels


def population_activity(context: dict, state: np.ndarray) -> list[float]:
    return [
        float(np.mean(np.abs(state[group["indices"]]))) if group["indices"] else 0.0
        for group in context["groups"]
    ]
