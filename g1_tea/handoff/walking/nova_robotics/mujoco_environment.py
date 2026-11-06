"""MuJoCo environment export. No inferred dynamics, plugins, or fluid claims."""

import json
import os
import posixpath
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import mujoco
import numpy as np

from .motion import motion_program
from .native_exports import child, environment_meshes, numbers, write_xml
from .scene import validate_environment

# The package's promise to a browser viewer: which model to compile and the exact
# files that compile reads. Viewers mount these and nothing else, so the list must
# be complete — it is proven below by compiling from a folder holding only it.
PREVIEW_SCHEMA = "nova.physics-preview/v1"


def _joint_inertia(body):
    """Inertia (kg m^2) or mass (kg) a hinge or slide joint drives: rotational inertia about
    the hinge axis through the origin (parallel-axis term for the COM offset), or the mass."""
    if body.motion == "slide":
        return float(body.mass)
    axis = np.asarray(body.axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    com = np.asarray(body.center_of_mass, dtype=float)
    radial = com - (com @ axis) * axis
    return float(axis @ np.diag(body.diagonal_inertia) @ axis + body.mass * radial @ radial)


def preview_files(model_path):
    """Every file MuJoCo reads to compile `model_path`, as package-relative paths.

    MuJoCo's own parser names each asset and its folder (meshes, heightfields and skins
    under meshdir; textures, including cube faces, under texturedir). Nothing here is
    inferred from a file extension, so a new asset kind needs no viewer change.
    """
    model_path = Path(model_path)
    spec = mujoco.MjSpec.from_file(str(model_path))
    files = {model_path.name}

    def add(folder, name):
        if not name:
            return
        path = posixpath.normpath(posixpath.join(folder or "", name))
        if PurePosixPath(path).is_absolute() or path == ".." or path.startswith("../"):
            raise ValueError(f"The model reads a file outside its package: {name}")
        files.add(path)

    for asset in (*spec.meshes, *spec.hfields, *spec.skins):
        add(spec.meshdir, asset.file)
    for texture in spec.textures:
        for name in (texture.file, *texture.cubefiles):
            add(spec.texturedir, name)
    return sorted(files)


def compile_preview(root, model_file, files, scratch=None):
    """Compile `model_file` from a fresh folder holding only `files`, as a viewer mounts
    them. Files are hard-linked from `root` (copied only across filesystems), so a large
    scene is not duplicated in the build's memory-backed temporary space. Raises MuJoCo's
    own error when the list is incomplete."""
    root = Path(root)
    with tempfile.TemporaryDirectory(prefix=".nova-preview-", dir=scratch or root) as folder:
        for name in files:
            target = Path(folder, *PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(root / name, target)
            except OSError:
                shutil.copyfile(root / name, target)
        disable_asset_cache()
        return mujoco.MjModel.from_xml_path(str(Path(folder, model_file)))


def declare_preview(output, model_file):
    """What a browser's MuJoCo loads for this package, proven by loading exactly that.

    The model compiles from a fresh folder that holds only the declared files, as a
    viewer will; a missing reference fails here, at build time, in MuJoCo's words, and
    never later in front of a viewer. The download's own MuJoCo load is recorded apart
    (`native_load`): a preview that cannot load does not make the exported file unusable.
    """
    output = Path(output)
    declared = {
        "schema_version": PREVIEW_SCHEMA,
        "engine": "mujoco",
        "engine_version": mujoco.__version__,
        "model": model_file,
        "files": [],
        "status": "unavailable",
        "reason": None,
        "native_load": "failed",
    }
    disable_asset_cache()
    try:
        files = preview_files(output / model_file)
        compile_preview(output, model_file, files)
        # not a compile asset, but the viewer needs it to play prescribed motion
        if (output / "motion_program.json").is_file():
            files = [*files, "motion_program.json"]
        declared.update(files=files, status="ready", native_load="passed")
    except (ValueError, OSError) as error:
        declared["reason"] = f"{type(error).__name__}: {error}"[:300]
        try:
            mujoco.MjModel.from_xml_path(str(output / model_file))
            declared["native_load"] = "passed"
        except (ValueError, OSError):
            pass
    return declared


def disable_asset_cache() -> None:
    """MuJoCo caches compiled mesh assets process-wide, keyed by the mesh's path as
    written in the XML. Every Nova package names its meshes meshes/object_NNNN.obj, so a
    second build in one process (tests, the live worker) would silently reuse the
    previous scene's geometry: a door became a stair tread. Nova never benefits from
    the cache; each model is compiled once."""
    try:
        cache = mujoco.mj_getCache()
        mujoco.mj_setCacheCapacity(cache, 0)
        mujoco.mj_clearCache(cache)
    except AttributeError:  # older MuJoCo without a public cache API
        pass


def export_mujoco_environment(scene, output, entries=None):
    """Static objects are single convex-hull geoms. Objects with a body or a separate
    contact surface get a massless non-colliding visual geom plus convex collision
    geoms under a body with explicit inertial and a passive joint (or freejoint)."""
    validate_environment(scene)
    disable_asset_cache()
    if entries is None:
        entries = environment_meshes(scene, output)
    root = ET.Element("mujoco", model="nova_environment")
    child(root, "compiler", angle="radian", autolimits="true")
    option = child(root, "option", gravity=numbers((0, 0, -scene.gravity_m_s2)), timestep="0.002")
    # Convex pieces multiply contacts; the default 36 MB arena overflows on a furnished
    # house with a robot inside. Scale with the collision geom count.
    pieces = sum(len(e["convex"]) for e in (entries or ()))
    # A heightfield generates contacts per cell under each foot, not per piece.
    pieces += 16 * sum(1 for e in (entries or ()) if e.get("heightfield"))
    child(root, "size", memory=f"{max(64, min(1024, 64 + pieces // 4))}M")
    assets, world = child(root, "asset"), child(root, "worldbody")
    objects = []
    geom_count = 0
    declared = set()

    def asset(file):
        name = file.rsplit("/", 1)[-1][:-4]
        if name not in declared:
            child(assets, "mesh", name=name, file=file)
            declared.add(name)
        return name

    for entry in entries:
        obj, name = entry["object"], entry["id"]
        parent = world
        if obj.body is not None:
            b = obj.body
            if b.motion == "kinematic":
                # Prescribed pose: a mocap body — MuJoCo's native carrier for
                # externally driven motion. The runner (or the Studio preview)
                # sets mocap_pos/quat from the motion program; contacts against
                # dynamic bodies stay real.
                parent = child(
                    world, "body", name=name + "_body", mocap="true", pos=numbers(b.origin)
                )
            else:
                parent = child(world, "body", name=name + "_body", pos=numbers(b.origin))
                child(
                    parent,
                    "inertial",
                    mass=str(b.mass),
                    pos=numbers(b.center_of_mass),
                    diaginertia=numbers(b.diagonal_inertia),
                )
            if b.motion == "free":
                child(parent, "freejoint", name=name + "_joint")
            elif b.motion in {"hinge", "slide"}:
                joint_attrs = dict(
                    name=name + "_joint",
                    type=b.motion,
                    axis=numbers(b.axis),
                    damping=str(b.damping),
                    frictionloss=str(b.frictionloss),
                )
                if b.limits is not None:
                    joint_attrs["range"] = numbers(b.limits)
                else:
                    # a driven spin is continuous
                    joint_attrs["limited"] = "false"
                child(parent, "joint", **joint_attrs)
        visual = asset(entry["visual"])
        if entry.get("heightfield"):
            # Ground: the elevation grid is MuJoCo's own exact primitive. The visible
            # surface is the same grid triangulated, carried as a massless non-colliding
            # mesh so the creator and the robot's feet never see different ground.
            patch = entry["heightfield"]
            elevation = name + "_hfield"
            child(assets, "hfield", name=elevation, nrow="0", ncol="0", file=patch["file"],
                  size=numbers(patch["mujoco_size"]))
            child(parent, "geom", name=name, type="mesh", mesh=visual, rgba=numbers(obj.color),
                  contype="0", conaffinity="0", mass="0", group="2")
            child(
                parent,
                "geom",
                name=name + "_terrain",
                type="hfield",
                hfield=elevation,
                pos=numbers(patch["mujoco_pos"]),
                rgba="0 0 0 0",
                group="3",
                contype="1",
                conaffinity="1",
                condim="6" if obj.rolling_friction else "4" if obj.torsional_friction else "3",
                friction=numbers((obj.friction, obj.torsional_friction, obj.rolling_friction)),
            )
            geom_count += 2
            objects.append(
                {
                    "name": obj.name,
                    "body": None,
                    "collision_pieces": 1,
                    "collision_source": "heightfield",
                    "heightfield": {
                        "nodes": [patch["rows"], patch["cols"]],
                        "size_m": patch["size_m"],
                        "relief_m": patch["statistics"]["relief_m"],
                    },
                    "friction": [obj.friction, obj.torsional_friction, obj.rolling_friction],
                }
            )
            continue
        if not obj.split_contact:
            # Basic static object: one geom is both the visible surface and its hull.
            child(
                parent,
                "geom",
                name=name,
                type="mesh",
                mesh=visual,
                rgba=numbers(obj.color),
                friction=numbers((obj.friction, 0, 0)),
                contype="1" if obj.collidable else "0",
                conaffinity="1" if obj.collidable else "0",
            )
            geom_count += 1
            objects.append(
                {
                    "name": obj.name,
                    "body": None,
                    "collision_pieces": int(obj.collidable),
                    "collision_source": "visual_mesh_convex_hull" if obj.collidable else "none",
                    "friction": [obj.friction, 0, 0],
                }
            )
            continue
        # Visual meshes never contribute a second mass or a second contact surface.
        child(
            parent,
            "geom",
            name=name,
            type="mesh",
            mesh=visual,
            rgba=numbers(obj.color),
            contype="0",
            conaffinity="0",
            mass="0",
            group="2",
        )
        geom_count += 1
        for part, file in enumerate(entry["convex"]):
            child(
                parent,
                "geom",
                name=f"{name}_collision_{part:02d}",
                type="mesh",
                mesh=asset(file),
                mass="0",
                group="3",
                rgba="0 0 0 0",
                contype="1",
                conaffinity="1",
                condim="6" if obj.rolling_friction else "4" if obj.torsional_friction else "3",
                friction=numbers((obj.friction, obj.torsional_friction, obj.rolling_friction)),
            )
            geom_count += 1
        objects.append(
            {
                "name": obj.name,
                "body": asdict(obj.body) if obj.body else None,
                "collision_pieces": len(entry["convex"]),
                "collision_source": "explicit_convex_pieces"
                if obj.collision_meshes
                else "auto_convex_pieces"
                if obj.auto_collision
                else "visual_mesh_convex_hull"
                if obj.collidable
                else "none",
                "friction": [obj.friction, obj.torsional_friction, obj.rolling_friction],
            }
        )
    for name, position in (("spawn", scene.spawn_position), ("goal", scene.goal_position)):
        child(world, "site", name=name, pos=numbers(position), size="0.03", rgba="0 1 0 .4")
    # Prescribed drives: strong position/velocity actuators on driven joints. The
    # motion program supplies ctrl targets; a pushed driven obstacle pushes back
    # with actuator force, which is the intended obstacle behavior.
    driven = [
        e
        for e in entries
        if e["object"].body is not None
        and e["object"].body.drive is not None
        and e["object"].body.motion in {"hinge", "slide"}
    ]
    if driven:
        # Stiff drives on light joints (a 50 mm conveyor roller has 3e-4 kg m^2 about its
        # axis) are far past the explicit-Euler stability limit. implicitfast integrates
        # actuator damping implicitly, so velocity drives stay stiff and stable; position
        # stiffness, still explicit, is capped by the joint's own inertia below.
        option.set("integrator", "implicitfast")
        actuators = child(root, "actuator")
        for entry in driven:
            b = entry["object"].body
            act_name = entry["id"] + "_drive"
            joint = entry["id"] + "_joint"
            if b.drive.kind == "spin":
                child(
                    actuators,
                    "velocity",
                    name=act_name,
                    joint=joint,
                    kv=str(max(50.0, 100.0 * b.mass)),
                )
            else:
                kp = min(max(200.0, 300.0 * b.mass), 1.0 * _joint_inertia(b) / 0.002**2)
                child(
                    actuators,
                    "position",
                    name=act_name,
                    joint=joint,
                    kp=str(kp),
                    # kv damps velocity toward zero, so a large kv fights the moving
                    # target itself: at kp/10 a swinging door lagged 0.12 rad behind
                    # its prescription. kp/70 keeps enough damping to settle without
                    # measurably braking the trajectory.
                    kv=str(round(kp / 70, 1)),
                )
    program = motion_program(entries)
    if program is not None:
        (output / "motion_program.json").write_text(
            json.dumps(program, allow_nan=False, separators=(",", ":"))
        )
    write_xml(
        output / "environment.xml",
        root,
        "SI metres/kg/seconds/radians, Z-up. Bodies attach to world at their explicit "
        "origin; mesh vertices localized from the authored world rest pose. Diagonal "
        "inertias are about COM in world-aligned rest axes. Visual geoms of movable or "
        "split-contact objects are massless and non-colliding. Joint limits use native "
        "soft constraints. No robot, controller, plugins or fluids/soil.",
    )
    loaded = mujoco.MjModel.from_xml_path(str(output / "environment.xml"))
    if loaded.ngeom != geom_count:
        raise ValueError("MuJoCo export lost visual or collision geometry")
    dynamic = sum(o.body is not None for o in scene.objects)
    return {
        "mujoco": {
            "entrypoint": "environment.xml",
            "format": "mjcf",
            "model_load": "passed",
            "mujoco_version": mujoco.__version__,
            "timestep_s": 0.002,
            "simulator_runtime": "not_run",
            "collision": "convex_pieces"
            if any(o.split_contact for o in scene.objects)
            else "per_object_convex_hull",
            "heightfields": sum(1 for e in entries if e.get("heightfield")),
            "restitution": "not_mapped",
            "objects": objects,
            "environment_dofs": loaded.nv,
            "dynamic_bodies": dynamic,
            "driven_bodies": sum(
                1 for o in scene.objects if o.body is not None and o.body.drive is not None
            ),
            "motion_program": "motion_program.json" if program is not None else None,
        },
        "warnings": [
            "Rigid-body approximations, not a calibrated real-world replica. See body provenance.",
            "Native soft contacts/limits apply; restitution is not mapped to a bounce coefficient.",
            "Each collision mesh uses a convex hull; use explicit convex pieces for concavities.",
            "Visuals use solid colors in native MJCF. "
            "No fluid, soil, breaking, thermal or sensor simulation.",
        ],
    }
