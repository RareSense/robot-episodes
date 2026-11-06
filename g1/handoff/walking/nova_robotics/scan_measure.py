"""Real-world dimensions from a creator's 3D scan.

Two kinds of input:

- **RoomPlan USDZ** (Apple's room-scanning framework, used by the "Room" modes of
  3D Scanner App, Polycam, Canvas and others). The file already holds the room as
  parametric parts — walls, doors, windows, openings and labelled furniture
  boxes — in metres. We read every part's world-space box; nothing is guessed.
- **A scan mesh** (GLB/GLTF/OBJ/PLY/USDZ mesh). One welded surface, so we
  compute: units, the up axis, the floor and ceiling planes, the wall outline
  from a horizontal slice, and the furniture clusters standing on the floor.

The report says how sure it is. Anything it cannot settle (units it had to
infer, an up axis it had to guess, a scale that looks implausible for a room)
becomes a question for the creator, so a wrong scale never reaches the build
silently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

MESH_SUFFIXES = {".glb", ".gltf", ".obj", ".ply", ".stl"}
USD_SUFFIXES = {".usdz", ".usd", ".usda", ".usdc"}
SCAN_SUFFIXES = MESH_SUFFIXES | USD_SUFFIXES

# RoomPlan names its prims after their category.
ROOMPLAN_ARCH = ("wall", "door", "window", "opening", "floor")
ROOMPLAN_OBJECTS = (
    "storage", "refrigerator", "stove", "bed", "sink", "washer", "dryer", "toilet",
    "bathtub", "oven", "dishwasher", "table", "sofa", "chair", "fireplace",
    "television", "stairs",
)

# Plausible indoor room extents, metres. Outside this, the units are suspect.
ROOM_MIN_M, ROOM_MAX_M = 1.0, 60.0


def _r(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


# ---------------------------------------------------------------- RoomPlan --


def _roomplan_report(path: Path) -> dict[str, Any] | None:
    """Parametric parts from a RoomPlan USDZ, or None when it holds none."""
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        return None
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        return None
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    up = UsdGeom.GetStageUpAxis(stage)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    parts: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        kind = next((k for k in ROOMPLAN_ARCH + ROOMPLAN_OBJECTS if name.startswith(k)), None)
        if kind is None or not prim.IsA(UsdGeom.Imageable):
            continue
        # count each named part once: skip children of an already-counted part
        if any(str(prim.GetPath()).startswith(p["path"] + "/") for p in parts):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        lo = np.array(box.GetMin()) * meters_per_unit
        hi = np.array(box.GetMax()) * meters_per_unit
        if up == "Y":  # to Z-up: (x, y, z)_Yup -> (x, -z, y)
            lo, hi = (np.array([lo[0], -hi[2], lo[1]]), np.array([hi[0], -lo[2], hi[1]]))
        parts.append({"path": str(prim.GetPath()), "kind": kind, "name": prim.GetName(),
                      "min": lo, "max": hi})
    if not any(p["kind"] in ("wall", "door", "window") for p in parts):
        return None
    walls = [p for p in parts if p["kind"] == "wall"]
    every = np.array([p["min"] for p in parts] + [p["max"] for p in parts])
    floor_z = float(min(p["min"][2] for p in walls)) if walls else float(every[:, 2].min())
    lo, hi = every.min(axis=0), every.max(axis=0)

    def entry(p: dict[str, Any]) -> dict[str, Any]:
        size = p["max"] - p["min"]
        center = (p["max"] + p["min"]) / 2
        return {
            "label": p["kind"],
            "name": p["name"],
            "dims_m": [_r(size[0]), _r(size[1]), _r(size[2])],
            "center_m": [_r(center[0]), _r(center[1]), _r(center[2] - floor_z)],
        }

    return {
        "method": "roomplan_parametric",
        "units": {"value": "m", "source": f"stage metersPerUnit={meters_per_unit}",
                  "confidence": "high"},
        "up_axis": {"value": "Z", "source": f"stage upAxis={up}", "confidence": "high"},
        "room": {
            "footprint_m": [_r(hi[0] - lo[0]), _r(hi[1] - lo[1])],
            "height_m": _r(max(p["max"][2] for p in walls) - floor_z) if walls else None,
            "walls": [entry(p) for p in walls],
        },
        "openings": [entry(p) for p in parts if p["kind"] in ("door", "window", "opening")],
        "objects": [entry(p) for p in parts if p["kind"] in ROOMPLAN_OBJECTS],
    }


# -------------------------------------------------------------------- mesh --


def _load_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(str(path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        geometry = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        if not geometry:
            raise ValueError("The scan holds no mesh surfaces")
        mesh = trimesh.util.concatenate(geometry)
    else:
        mesh = loaded
    if len(mesh.faces) == 0:
        raise ValueError("The scan mesh has no faces")
    return mesh


def _infer_units(extent_max: float) -> tuple[float, str, str]:
    """(scale to metres, unit name, confidence) from the scan's largest extent."""
    if ROOM_MIN_M <= extent_max <= ROOM_MAX_M:
        return 1.0, "m", "medium"
    if ROOM_MIN_M * 100 <= extent_max <= ROOM_MAX_M * 100:
        return 0.01, "cm", "low"
    if ROOM_MIN_M * 1000 <= extent_max <= ROOM_MAX_M * 1000:
        return 0.001, "mm", "low"
    if ROOM_MIN_M / 3.2808 <= extent_max * 0.3048 <= ROOM_MAX_M:
        return 0.3048, "ft", "low"
    return 1.0, "m", "low"


def _level(mesh, suffix: str):
    """Rotate to Z-up. glTF is Y-up by specification; otherwise pick the axis
    the largest share of surface area faces along (floors and ceilings)."""
    import trimesh

    normals, areas = mesh.face_normals, mesh.area_faces
    shares = [float(areas[np.abs(normals[:, k]) > 0.9].sum()) for k in range(3)]
    if suffix in (".glb", ".gltf"):
        axis, source, confidence = 1, "glTF specification (Y-up)", "high"
    else:
        axis = int(np.argmax(shares))
        source = "dominant horizontal-surface direction"
        confidence = "medium" if max(shares) > 0.2 * areas.sum() else "low"
    if axis == 1:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0)))
    elif axis == 0:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, (0, 1, 0)))
    return mesh, {"value": "Z", "source": source, "confidence": confidence,
                  "original_axis": "XYZ"[axis]}


def _planes(mesh, min_ceiling: float = 1.5) -> tuple[float, float | None]:
    """Floor and ceiling heights: the area-weighted peaks of up- and
    down-facing surfaces."""
    normals, areas = mesh.face_normals, mesh.area_faces
    centers = mesh.triangles_center[:, 2]

    def peak(mask, prefer_low):
        if not mask.any():
            return None
        heights, weights = centers[mask], areas[mask]
        lo, hi = heights.min(), heights.max()
        bins = max(10, int((hi - lo) / 0.03) + 1)
        hist, edges = np.histogram(heights, bins=bins, range=(lo, hi + 1e-9), weights=weights)
        strong = np.where(hist >= 0.25 * hist.max())[0]
        if prefer_low:
            # the floor is the LARGEST surface in the lower half of the scan,
            # not merely the lowest one (doubled slabs, scan debris below it)
            lower = strong[edges[strong] <= lo + (hi - lo) * 0.5]
            index = lower[np.argmax(hist[lower])] if len(lower) else strong.min()
        else:
            index = strong.max()
        return float((edges[index] + edges[index + 1]) / 2)

    # Scan exports disagree on face winding, so floors may face down and
    # ceilings up: take horizontal surfaces of either facing, lowest and highest.
    horizontal = np.abs(normals[:, 2]) > 0.9
    floor = peak(horizontal, prefer_low=True)
    ceiling = peak(horizontal, prefer_low=False)
    if floor is None:
        floor = float(centers.min())
    if ceiling is not None and ceiling - floor < min_ceiling:
        ceiling = None  # a table top, not a ceiling
    return floor, ceiling


def _outline(mesh, floor: float) -> dict[str, Any]:
    """The floor plan from the floor itself: every up-facing surface at floor
    height, rasterized, small scan gaps closed, holes under furniture filled.
    Unlike a wall outline this needs no closed walls, so open doorways and
    missing wall patches do not break it."""
    import trimesh
    from scipy import ndimage
    from shapely.geometry import MultiPoint, box
    from shapely.ops import unary_union

    normals = mesh.face_normals
    centers = mesh.triangles_center
    on_floor = (np.abs(normals[:, 2]) > 0.85) & (np.abs(centers[:, 2] - floor) < 0.07)
    if on_floor.sum() < 20:
        on_floor = np.abs(centers[:, 2] - floor) < 0.15
    faces = mesh.faces[on_floor]
    points = np.vstack([mesh.vertices[faces.ravel()][:, :2], centers[on_floor][:, :2]])
    cell = 0.05
    origin = points.min(axis=0) - 0.25
    ij = np.floor((points - origin) / cell).astype(int)
    grid = np.zeros(tuple(ij.max(axis=0) + 10), bool)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = ndimage.binary_closing(grid, structure=np.ones((5, 5)), iterations=2)
    grid = ndimage.binary_fill_holes(grid)
    labels, count = ndimage.label(grid)
    if count:
        largest = 1 + int(np.argmax(ndimage.sum(grid, labels, range(1, count + 1))))
        grid = labels == largest
    cells = np.argwhere(grid)
    floor_xy = origin + (cells + 0.5) * cell
    transform, extents = trimesh.bounds.oriented_bounds_2D(floor_xy)
    yaw = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
    # walls align at any quarter turn; take the smallest so the scan is never mirrored
    yaw = (yaw + 45.0) % 90.0 - 45.0
    width, depth = sorted((float(extents[0]), float(extents[1])), reverse=True)
    shape = unary_union([box(*(origin + c * cell), *(origin + (c + 1) * cell))
                         for c in cells]).simplify(0.08)
    if shape.geom_type != "Polygon":
        shape = max(shape.geoms, key=lambda g: g.area)
    if shape.is_empty:
        shape = MultiPoint(floor_xy).convex_hull
    ring = np.array(shape.exterior.coords)[:-1]
    return {
        "footprint_m": [_r(width), _r(depth)],
        "orientation_deg": _r(yaw, 1),
        "footprint_source": "measured floor surface",
        "floor_area_m2": _r(len(cells) * cell * cell, 2),
        "outline_m": [[_r(x, 2), _r(y, 2)] for x, y in ring],
    }


def _objects(mesh, floor: float, outline: dict[str, Any],
             ceiling: float | None) -> list[dict[str, Any]]:
    """Furniture standing on the floor, separated on an occupancy grid.

    Interior walls, partitions and door frames reach toward the ceiling and
    would weld every piece of furniture into one blob, so any grid column whose
    surface rises above 2 m is structure and is removed first; thin noise
    bridges between pieces are then cut with a morphological opening."""
    from scipy import ndimage

    cell = 0.05
    points = mesh.sample(min(400_000, max(60_000, len(mesh.faces) * 2)))
    points = points[points[:, 2] > floor + 0.08]
    if ceiling is not None:
        # the ceiling spans every column; left in, it would mark all of them structure
        points = points[points[:, 2] < ceiling - 0.15]
    ring = np.array(outline["outline_m"])
    if len(ring) >= 3:
        from shapely import contains_xy
        from shapely.geometry import Polygon

        inner = Polygon(ring).convex_hull.buffer(-0.2)
        if not inner.is_empty:
            points = points[contains_xy(inner, points[:, 0], points[:, 1])]
    if not len(points):
        return []
    origin = points[:, :2].min(axis=0)
    ij = np.floor((points[:, :2] - origin) / cell).astype(int)
    shape = tuple(ij.max(axis=0) + 1)
    top = np.full(shape, -np.inf)
    np.maximum.at(top, (ij[:, 0], ij[:, 1]), points[:, 2] - floor)
    structure_height = 2.0 if ceiling is None else min(2.0, (ceiling - floor) - 0.35)
    structure = ndimage.binary_dilation(top > structure_height, iterations=2)
    occupied = np.isfinite(top) & ~structure
    occupied = ndimage.binary_opening(occupied, structure=np.ones((3, 3)))
    labels, count = ndimage.label(occupied)
    point_labels = labels[ij[:, 0], ij[:, 1]]
    found = []
    for k in range(1, count + 1):
        cluster = points[point_labels == k]
        if len(cluster) < 150:
            continue
        lo, hi = cluster.min(axis=0), cluster.max(axis=0)
        size = hi - lo
        if max(size[0], size[1]) < 0.2 or size[2] < 0.12:
            continue  # floor clutter and scan noise
        if max(size[0], size[1]) > 4.0:
            continue  # a merged region, not one object
        center = (hi + lo) / 2
        top = hi[2] - floor  # it stands on the floor: the band cut as floor noise is part of it
        found.append({
            "label": "object",
            "dims_m": [_r(size[0]), _r(size[1]), _r(top)],
            "center_m": [_r(center[0]), _r(center[1]), _r(top / 2)],
            "top_m": _r(hi[2] - floor),
        })
    found.sort(key=lambda o: -o["dims_m"][0] * o["dims_m"][1])
    return found[:60]


def _label(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labelling without scipy."""
    labels = np.zeros(grid.shape, int)
    count = 0
    for start in zip(*np.nonzero(grid), strict=True):
        if labels[start]:
            continue
        count += 1
        stack = [start]
        labels[start] = count
        while stack:
            i, j = stack.pop()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < grid.shape[0] and 0 <= b < grid.shape[1] \
                        and grid[a, b] and not labels[a, b]:
                    labels[a, b] = count
                    stack.append((a, b))
    return labels, count


def _raw_reference(path: Path, what: str) -> float | None:
    """The anchor's size in file units, before any unit guess — so a creator's one
    real measurement can set the scale of a scan whose units are unknown."""
    mesh, _up = _level(_load_mesh(path), path.suffix.lower())
    # measure at a room-like size (the plane and raster tolerances are in metres),
    # then undo that normalisation
    norm = 8.0 / max(float(mesh.extents.max()), 1e-9)
    mesh.apply_scale(norm)
    what = what.lower()
    floor, ceiling = _planes(mesh, min_ceiling=0.0)
    if what in {"ceiling height", "room height"}:
        return None if ceiling is None or ceiling <= floor else (ceiling - floor) / norm
    width, depth = _outline(mesh, floor)["footprint_m"]
    return {"room width": width, "room length": width, "room depth": depth}.get(what, 0) / norm \
        or None


def _mesh_report(path: Path, stated_scale: float | None,
                 scale_source: str | None = None) -> dict[str, Any]:
    mesh = _load_mesh(path)
    mesh, up = _level(mesh, path.suffix.lower())
    extent = float(mesh.extents.max())
    if stated_scale:
        scale, unit, confidence = stated_scale, scale_source or "creator-stated", "high"
    else:
        scale, unit, confidence = _infer_units(extent)
    if scale != 1.0:
        mesh.apply_scale(scale)
    floor, ceiling = _planes(mesh)
    outline = _outline(mesh, floor)
    # turn the room square to its own walls, so widths, plans and object boxes
    # all read along the walls instead of along the scanner's arbitrary heading
    import trimesh

    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        -math.radians(outline["orientation_deg"]), (0, 0, 1)))
    outline = _outline(mesh, floor)
    return {
        "method": "mesh_geometry",
        "units": {"value": "m", "source_unit": unit, "scale_applied": scale,
                  "confidence": confidence,
                  "source": (scale_source or "creator-stated") if stated_scale else
                  f"inferred from the scan's largest extent ({_r(extent, 2)} file units)"},
        "up_axis": up,
        "room": {
            **outline,
            "height_m": _r(ceiling - floor) if ceiling is not None else None,
            "height_source": "floor-to-ceiling planes" if ceiling is not None
            else "no ceiling captured",
            "bounding_box_m": [_r(v) for v in mesh.extents],
        },
        "openings": [],
        "objects": _objects(mesh, floor, outline, ceiling),
        "objects_note": "approximate furniture boxes separated from a welded scan "
                        "mesh; a RoomPlan (Room mode) scan gives exact per-object boxes",
    }


# ------------------------------------------------------------------- public --


def measure(path: str | Path, *, app: str | None = None,
            stated_scale: float | None = None,
            known: dict[str, float] | None = None) -> dict[str, Any]:
    """Measure one scan. `stated_scale` overrides inferred units (metres per file
    unit); `known` = {"what": metres} lets the creator calibrate from one real
    measurement, e.g. {"room width": 4.2} or {"door height": 2.03}."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SCAN_SUFFIXES:
        raise ValueError(f"Unsupported scan format {suffix}")
    report = None
    if suffix in USD_SUFFIXES:
        report = _roomplan_report(path)
    calibration = None
    if report is None:
        scale_source = None
        if known:
            # scale the mesh itself before measuring, so floor, ceiling and
            # furniture thresholds all work in real metres
            what, value = next(iter(known.items()))
            raw = _raw_reference(path, what)
            if raw:
                stated_scale = float(value) / raw
                scale_source = f"calibrated: {what} = {value} m"
                calibration = {"applied": True, "anchor": what, "anchor_m": float(value),
                               "factor": _r(stated_scale, 6)}
            else:
                calibration = {"applied": False,
                               "reason": f"no measurable counterpart for '{what}'"}
        report = _mesh_report(path, stated_scale, scale_source)
    elif known:
        calibration = _calibrate(report, known)
    report["source"] = {"file": path.name, "format": suffix.lstrip("."),
                        "app": app or "unknown"}
    report["calibration"] = calibration
    report["questions"] = _questions(report, app)
    return report


def _calibrate(report: dict[str, Any], known: dict[str, float]) -> dict[str, Any]:
    """Rescale every dimension so one creator-stated measurement holds exactly."""
    room = report["room"]
    measured = {
        "room width": room["footprint_m"][0],
        "room length": room["footprint_m"][0],
        "room depth": room["footprint_m"][1],
        "room height": room.get("height_m"),
        "ceiling height": room.get("height_m"),
    }
    what, value = next(iter(known.items()))
    reference = measured.get(what.lower())
    if not reference:
        return {"applied": False, "reason": f"no measured counterpart for '{what}'"}
    factor = float(value) / float(reference)

    def scale(values):
        return [_r(v * factor) for v in values]

    room["footprint_m"] = scale(room["footprint_m"])
    if room.get("height_m"):
        room["height_m"] = _r(room["height_m"] * factor)
    for group in ("objects", "openings"):
        for item in report[group]:
            item["dims_m"] = scale(item["dims_m"])
            item["center_m"] = scale(item["center_m"])
    report["units"]["confidence"] = "high"
    return {"applied": True, "anchor": what, "anchor_m": float(value), "factor": _r(factor, 4)}


def _questions(report: dict[str, Any], app: str | None) -> list[str]:
    asks = []
    if not app or app == "unknown":
        asks.append("Which app made this scan (e.g. 3D Scanner App, Polycam, Scaniverse, "
                    "Canvas), and was it a LiDAR / Room mode scan or a photo scan?")
    if report["units"]["confidence"] == "low":
        source_unit = report["units"].get("source_unit", "?")
        asks.append(f"The scan's units look like {source_unit}. Can you confirm one real "
                    "measurement — the room width, the ceiling height or a door height?")
    if report["up_axis"]["confidence"] == "low":
        asks.append("Which way is up in this scan? It looks rotated.")
    if report["room"].get("height_m") is None:
        asks.append("The ceiling was not captured. What is the ceiling height?")
    area = report["room"].get("floor_area_m2")
    box_xy = report["room"].get("bounding_box_m", [0, 0])[:2]
    if area is not None and area < 2.0 and box_xy[0] * box_xy[1] > 6.0:
        asks.append("Only a small patch of floor was found in this scan, so the room size "
                    "is uncertain. What are the room's width and length?")
    width = report["room"]["footprint_m"][0]
    if not ROOM_MIN_M <= width <= ROOM_MAX_M:
        asks.append(f"The scan measures {width} m across, which is unusual for a room. "
                    "What is the real width?")
    return asks


def write_report(scan: str | Path, output: str | Path, **options) -> dict[str, Any]:
    report = measure(scan, **options)
    Path(output).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def floor_plan(report: dict[str, Any], output: str | Path, size: int = 1000) -> Path:
    """A measured top-down plan: wall outline, openings and object boxes with
    their dimensions written on, for the photo-analysis model to read beside
    the creator's photographs."""
    from PIL import Image, ImageDraw

    room = report["room"]
    ring = np.array(room.get("outline_m") or [])
    items = report.get("objects", []) + report.get("openings", [])
    xs = [p[0] for p in ring] + [o["center_m"][0] for o in items]
    ys = [p[1] for p in ring] + [o["center_m"][1] for o in items]
    if not xs:
        raise ValueError("Nothing measured to draw")
    lo = np.array([min(xs), min(ys)]) - 0.6
    hi = np.array([max(xs), max(ys)]) + 0.6
    scale = (size - 80) / max(hi - lo)
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    def px(x, y):
        return (40 + (x - lo[0]) * scale, size - 40 - (y - lo[1]) * scale)

    if len(ring) >= 3:
        draw.polygon([px(x, y) for x, y in ring], outline=(30, 30, 30), width=4)
    for o in items:
        cx, cy = o["center_m"][0], o["center_m"][1]
        w, d = o["dims_m"][0], o["dims_m"][1]
        colour = (200, 90, 40) if o["label"] in ("door", "window", "opening") else (40, 110, 200)
        draw.rectangle([px(cx - w / 2, cy + d / 2), px(cx + w / 2, cy - d / 2)],
                       outline=colour, width=3)
        draw.text(px(cx - w / 2, cy + d / 2 + 0.05),
                  f"{o.get('name', o['label'])} {w:.2f}x{d:.2f}x{o['dims_m'][2]:.2f}m",
                  fill=colour)
    footprint = room["footprint_m"]
    height = room.get("height_m")
    draw.text((40, 12), f"Measured plan: {footprint[0]:.2f} x {footprint[1]:.2f} m"
              + (f", ceiling {height:.2f} m" if height else ", ceiling not captured")
              + f"  ({report['method']})", fill=(0, 0, 0))
    # a 1-metre scale bar
    x0, y0 = size - 40 - scale, size - 20
    draw.line([(x0, y0), (x0 + scale, y0)], fill=(0, 0, 0), width=3)
    draw.text((x0, y0 - 14), "1 m", fill=(0, 0, 0))
    output = Path(output)
    image.save(output)
    return output
