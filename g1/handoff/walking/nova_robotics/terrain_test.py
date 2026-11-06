"""Measured ground: does the robot's foot meet the surface the scene authored?

A hill can look right in the viewport and still be wrong under a foot. MuJoCo collides
a plain mesh through its convex hull, so a valley would fill in; a heightfield written
with its rows transposed would be a different hill entirely; and PyBullet centres a
heightfield on its own data range, so an unshifted field sits at the wrong altitude.
None of that is visible in a render.

So every build drops probe spheres onto each terrain patch at grid nodes and reports
how far the resting contact sits from where that sphere must come to rest on the
authored surface — which is not the height at the node: a sphere rides high on a
slope and rests on the edge beside a riser, and `HeightField.rest_height` predicts
both. Probes collide with terrain only, never with a boulder or a post that happens to
stand on a node. MuJoCo decides; PyBullet is a second opinion recorded beside it.
"""

import importlib.util
import re
from pathlib import Path

import numpy as np

from .configuration import ACCEPTANCE_ENGINE, COMPATIBILITY_ENGINE

PROBE_RADIUS = 0.03  # a toe-sized contact, small enough to enter the roughness
PROBE_MASS = 0.5
DROP_HEIGHT = 0.05
SETTLE_STEPS = 800
TOLERANCE = 0.01  # 10 mm: what is left once the probe's own geometry is accounted for
PROBES_PER_FIELD = 16
# MuJoCo collision bits: probes carry bit 2, terrain accepts bits 1 and 2, everything
# else stays on bit 1. (contype & conaffinity) is then nonzero only for probe-terrain.
PROBE_BIT = 2
ENGINE_LABELS = {"mujoco": "MuJoCo", "pybullet": "PyBullet"}


def _fields(scene):
    return [(index, obj) for index, obj in enumerate(scene.objects) if obj.heightfield is not None]


def _mujoco_pass(scene, model_path):
    import mujoco

    from .mujoco_environment import disable_asset_cache

    disable_asset_cache()
    spec = mujoco.MjSpec.from_file(str(model_path))
    terrain_geoms = {f"object_{index:04d}_terrain" for index, _ in _fields(scene)}
    for geom in spec.geoms:
        if geom.name in terrain_geoms:
            geom.conaffinity = int(geom.conaffinity) | PROBE_BIT
    probes = []
    for index, obj in _fields(scene):
        for number, (x, y, z) in enumerate(obj.heightfield.probe_points(PROBES_PER_FIELD)):
            name = f"nova_terrain_probe_{index:04d}_{number:02d}"
            body = spec.worldbody.add_body(name=name, pos=[x, y, z + DROP_HEIGHT + PROBE_RADIUS])
            body.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], name=name + "_slide")
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[PROBE_RADIUS, 0, 0],
                mass=PROBE_MASS,
                name=name + "_tip",
                contype=PROBE_BIT,
                conaffinity=0,
            )
            probes.append((obj.name, name, x, y, obj.heightfield.rest_height(x, y, PROBE_RADIUS)))
    model = spec.compile()
    data = mujoco.MjData(model)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
    measured = {}
    for object_name, probe, x, y, authored in probes:
        resting = float(data.body(probe).xpos[2]) - PROBE_RADIUS
        measured.setdefault(object_name, []).append((x, y, authored, resting))
    return {
        "engine": f"MuJoCo {mujoco.__version__}",
        "contact": "native heightfield geom",
        "samples": measured,
        "finite": bool(np.isfinite(data.qpos).all()),
        "warnings": int(data.warning.number.sum()),
    }


def _pybullet_pass(scene, package):
    """The same drop in PyBullet, through exactly what the shipped launcher loads.
    Only terrain is loaded, so probes cannot land on anything else."""
    if importlib.util.find_spec("pybullet") is None:
        raise RuntimeError("pybullet is not installed in this runtime")
    launcher_path = Path(__file__).with_name("pybullet_launcher.py")
    module_spec = importlib.util.spec_from_file_location("nova_pybullet_launcher", launcher_path)
    launcher = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(launcher)
    import pybullet as p

    from .urdf_preview import pybullet_item

    package = Path(package)
    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -scene.gravity_m_s2)
        p.setTimeStep(0.002)
        measured = {}
        finite = True
        for index, obj in _fields(scene):
            item = pybullet_item(index, obj)
            if not (package / item["heightfield"]["file"]).is_file():
                raise RuntimeError(f"PyBullet export is incomplete: {item['heightfield']['file']}")
            launcher.load_terrain(p, package, item)
            shape = p.createCollisionShape(p.GEOM_SPHERE, radius=PROBE_RADIUS)
            probes = []
            for x, y, z in obj.heightfield.probe_points(PROBES_PER_FIELD):
                probe = p.createMultiBody(
                    PROBE_MASS, shape, basePosition=[x, y, z + DROP_HEIGHT + PROBE_RADIUS]
                )
                p.changeDynamics(probe, -1, lateralFriction=obj.friction)
                # Pin it to a vertical line, as MuJoCo's slide joint does: a free
                # sphere rolls off a 40-degree ridge and measures the wrong point.
                p.createConstraint(
                    probe, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0], [x, y, 0]
                )
                probes.append((probe, x, y, obj.heightfield.rest_height(x, y, PROBE_RADIUS)))
            for _ in range(SETTLE_STEPS):
                p.stepSimulation()
            for probe, x, y, authored in probes:
                position = p.getBasePositionAndOrientation(probe)[0]
                finite = finite and bool(np.isfinite(position).all())
                measured.setdefault(obj.name, []).append(
                    (x, y, authored, float(position[2]) - PROBE_RADIUS)
                )
                p.removeBody(probe)
        return {
            "engine": f"PyBullet {p.getAPIVersion()}",
            "contact": "native GEOM_HEIGHTFIELD",
            "samples": measured,
            "finite": finite,
            "warnings": 0,
        }
    finally:
        if p.isConnected(client):
            p.disconnect(client)


_PASSES = {"mujoco": _mujoco_pass, "pybullet": _pybullet_pass}


def _measure(engine, scene, model_path, package):
    try:
        measured = _PASSES[engine](scene, model_path if engine == "mujoco" else package)
    except Exception as exc:  # a probe must never fail the build silently
        return None, {"status": "error", "reason": str(exc)[:400]}
    if not measured["finite"]:
        return None, {"status": "error", "reason": "probe simulation produced non-finite state"}
    return measured, {"status": "measured"}


def _summarize(name, rows):
    """Worst error for one terrain patch in one engine, and where it happened."""
    if not rows:
        return None
    errors = [abs(resting - authored) for _, _, authored, resting in rows]
    worst = int(np.argmax(errors))
    x, y, authored, resting = rows[worst]
    return {
        "probes": len(rows),
        "max_error_m": round(float(max(errors)), 4),
        "mean_error_m": round(float(np.mean(errors)), 4),
        "worst_point_m": [round(x, 3), round(y, 3)],
        "authored_z_m": round(float(authored), 4),
        "contact_z_m": round(float(resting), 4),
        "outcome": "exact" if max(errors) <= TOLERANCE else "wrong_surface",
    }


def _overlaps(fields):
    """Patches that cover the same ground and disagree about its height.

    Two grids over one place both collide and the robot walks on whichever is higher,
    so an overlap is only ever harmless when the two surfaces agree.
    """
    findings = []
    for first in range(len(fields)):
        for second in range(first + 1, len(fields)):
            a, b = fields[first][1].heightfield, fields[second][1].heightfield
            x0 = max(a.grid.x[0], b.grid.x[0])
            x1 = min(a.grid.x[-1], b.grid.x[-1])
            y0 = max(a.grid.y[0], b.grid.y[0])
            y1 = min(a.grid.y[-1], b.grid.y[-1])
            if x1 <= x0 or y1 <= y0:
                continue
            xs, ys = np.meshgrid(np.linspace(x0, x1, 17), np.linspace(y0, y1, 17))
            gap = np.abs(
                a.height_at(xs.reshape(-1), ys.reshape(-1))
                - b.height_at(xs.reshape(-1), ys.reshape(-1))
            )
            findings.append(
                {
                    "fields": [fields[first][1].name, fields[second][1].name],
                    "region_m": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
                    "max_height_gap_m": round(float(gap.max()), 4),
                    "outcome": "disagree" if float(gap.max()) > TOLERANCE else "agree",
                }
            )
    return findings


def run_terrain_test(scene, model_path, package, *, acceptance=ACCEPTANCE_ENGINE,
                     compatibility=(COMPATIBILITY_ENGINE,)) -> dict:
    """Drop probes on every terrain patch. The acceptance engine's contact decides;
    a compatibility engine's disagreement is recorded beside it."""
    fields = _fields(scene)
    result = {
        "schema_version": "nova.robotics-terrain-test/v2",
        "acceptance_engine": acceptance,
        "probe_radius_m": PROBE_RADIUS,
        "tolerance_m": TOLERANCE,
        "fields": [],
        "overlaps": _overlaps(fields),
        "coverage": {},
        "compatibility_notes": [],
        "status": "not_applicable",
    }
    if not fields:
        return result
    accepted, result["coverage"][acceptance] = _measure(acceptance, scene, model_path, package)
    opinions = {}
    for engine in compatibility:
        measured, result["coverage"][engine] = _measure(engine, scene, model_path, package)
        if measured is not None:
            opinions[engine] = measured
    worst = 0.0
    for _, obj in fields:
        primary = _summarize(obj.name, accepted["samples"].get(obj.name)) if accepted else None
        engines = {acceptance: primary} if primary else {}
        for engine, measured in opinions.items():
            summary = _summarize(obj.name, measured["samples"].get(obj.name))
            if summary is None:
                continue
            engines[engine] = summary
            if summary["outcome"] == "wrong_surface" and (
                primary is None or primary["outcome"] == "exact"
            ):
                result["compatibility_notes"].append(
                    f"{ENGINE_LABELS.get(engine, engine)}: {obj.name} contact off by "
                    f"{summary['max_error_m'] * 1000:.0f} mm "
                    f"({ENGINE_LABELS.get(acceptance, acceptance)}: "
                    f"{primary['outcome'].replace('_', ' ') if primary else 'not measured'})"
                )
        if primary:
            worst = max(worst, primary["max_error_m"])
        result["fields"].append(
            {
                "name": obj.name,
                "friction": obj.friction,
                **obj.heightfield.statistics(),
                "engines": engines,
                "outcome": primary["outcome"] if primary else "not_run",
            }
        )
    result["max_error_m"] = round(float(worst), 4)
    failed = [f for f in result["fields"] if f["outcome"] == "wrong_surface"]
    clashing = [entry for entry in result["overlaps"] if entry["outcome"] == "disagree"]
    result["status"] = (
        # Overlaps are judged from the authored grids alone, so they fail even when
        # no engine could run.
        "failed"
        if failed or clashing
        else "error"
        if accepted is None
        else "passed"
    )
    return result


def report_lines(result) -> list[str]:
    if not result.get("fields"):
        return []
    lines = [
        f"TERRAIN_TEST status={result['status']} fields={len(result['fields'])} "
        f"max_error_mm={result.get('max_error_m', 0) * 1000:.1f} "
        f"tolerance_mm={result['tolerance_m'] * 1000:.0f}"
    ]
    for engine, item in result.get("coverage", {}).items():
        if item["status"] != "measured":
            lines.append(f"TERRAIN_COVERAGE {engine}: {item['status']} ({item.get('reason', '')})")
    for field in result["fields"]:
        engines = " ".join(
            f"{engine}={entry['max_error_m'] * 1000:.1f}mm/{entry['outcome']}"
            for engine, entry in field["engines"].items()
        )
        lines.append(
            f"TERRAIN {field['name']} {field['nodes'][0]}x{field['nodes'][1]} nodes "
            f"cell={field['cell_m'][0]:.3f}m relief={field['relief_m']:.2f}m "
            f"max_slope={field['max_slope_deg']:.1f}deg step={field['max_node_step_m']:.3f}m "
            f"{engines or 'no_engine'}"
        )
    for entry in result.get("overlaps", []):
        if entry["outcome"] == "disagree":
            first, second = entry["fields"]
            lines.append(
                f"TERRAIN_OVERLAP {first} and {second} cover the same ground over "
                f"{entry['region_m']} and disagree by up to "
                f"{entry['max_height_gap_m'] * 1000:.0f} mm; patches tile edge to edge, "
                "they do not stack — the robot walks on whichever is higher"
            )
    for note in result.get("compatibility_notes", []):
        lines.append(f"COMPATIBILITY {note}; a second-opinion engine, not a repair task")
    return lines


def public_label(raw):
    """A part name safe to show a creator: letters, digits, spaces and dashes only."""
    return raw.replace("_", " ") if re.fullmatch(r"[A-Za-z0-9_ -]{1,80}", raw) else "Terrain patch"
