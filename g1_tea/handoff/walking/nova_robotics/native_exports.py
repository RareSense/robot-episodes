"""Native environment exports: Gazebo SDF, MuJoCo MJCF and PyBullet bodies.

Every target receives the SAME movable bodies. An object with a RigidBody becomes a
passive hinge or slider attached to world (doors, drawers, lids, gates) or a free
body (loose props a robot can push, lift or clear away). Static objects stay static.
Customer code/plugins are never evaluated; nothing here reads uploaded robots.
"""

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from trimesh.exchange.obj import export_obj

from .collision import ensure_contact
from .motion import motion_program

JOINT_TYPES = {"hinge": "revolute", "slide": "prismatic"}


def child(parent, tag, text=None, **attributes):
    node = ET.SubElement(parent, tag, attributes)
    if text is not None:
        node.text = str(text)
    return node


def numbers(values):
    return " ".join(format(float(value), ".9g") for value in values)


def write_xml(path, root, ledger):
    ET.indent(root)
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<!-- '
        + ledger
        + " -->\n"
        + ET.tostring(root, encoding="unicode")
        + "\n",
        encoding="utf-8",
    )
    ET.parse(path)


def environment_meshes(scene, output: Path) -> list[dict]:
    """Write one OBJ per visual surface and per explicit convex collision piece.

    Movable objects are localized to their body origin (the authored rest pose) so
    every simulator places them through the same origin; static objects keep world
    coordinates. Each entry carries two contact lists: `collision`, the exact surface
    a target may use for static triangle-mesh contact (explicit pieces or the visual
    mesh itself), and `convex`, the convex pieces a target must use where contact has
    to be convex (MuJoCo everywhere, movable bodies everywhere). `convex` comes from
    the author's pieces, else the host's automatic decomposition, else the hull.

    Terrain is the exception: its entry carries `heightfield`, the elevation grid
    written once in MuJoCo's native binary format, which PyBullet reads back too. Its
    `collision` stays the exact triangulated surface, which USD and SDF use directly.
    """
    ensure_contact(scene)
    (output / "meshes").mkdir(parents=True, exist_ok=True)
    entries = []
    for index, obj in enumerate(scene.objects):
        name = f"object_{index:04d}"
        origin = np.asarray(obj.body.origin if obj.body else (0, 0, 0), dtype=float)

        placed = []  # the convex pieces in the body frame, for primitive detection

        def write(stem, mesh, origin=origin, keep=False, placed=placed):
            geometry = mesh.copy()
            geometry.apply_translation(-origin)
            if keep:
                placed.append(geometry)
            file = f"meshes/{stem}.obj"
            # Normals always: Gazebo's DART rejects every OBJ submesh whose normal count
            # does not match its vertex count — zero included — and then segfaults
            # building contact from the empty shapes it is left with.
            (output / file).write_text(
                export_obj(geometry, include_texture=False, include_normals=True),
                encoding="utf-8",
            )
            return file

        visual = write(name, obj.mesh)
        heightfield = _write_heightfield(obj.heightfield, name, output) if obj.is_terrain else None
        if not obj.collidable:
            collision, convex = [], []
        elif obj.collision_meshes is not None:
            collision = [
                write(f"{name}_collision_{part:02d}", mesh, keep=True)
                for part, mesh in enumerate(obj.collision_meshes)
            ]
            convex = collision
        else:
            collision = [visual]
            convex = [
                write(f"{name}_auto_{part:02d}", mesh, keep=True)
                for part, mesh in enumerate(obj.auto_collision or ())
            ]
            if not convex:
                convex = [visual]
                hull = obj.mesh.copy()
                hull.apply_translation(-origin)
                placed.append(hull)
        entries.append(
            {
                "id": name,
                "object": obj,
                "origin": tuple(origin.tolist()),
                "visual": visual,
                "collision": collision,
                # Terrain never collides as convex pieces; each engine takes the grid.
                "convex": [] if heightfield else convex,
                "convex_meshes": [] if heightfield else placed,
                "heightfield": heightfield,
            }
        )
    return entries


def heightfield_spec(patch, name) -> dict:
    """Where the elevation grid lives and how to read it back in metres. Heights in
    the file are normalized to [0, 1]; a reader recovers metres as
    `lowest_m + value * elevation_m`."""
    radius_x, radius_y, elevation, skirt = patch.mujoco_size()
    return {
        "file": f"terrain/{name}.hf",
        "rows": int(patch.grid.rows),
        "cols": int(patch.grid.cols),
        "size_m": [float(patch.grid.size[0]), float(patch.grid.size[1])],
        "center_m": [float(patch.grid.center[0]), float(patch.grid.center[1])],
        "lowest_m": float(patch.base_z + patch.lowest),
        "elevation_m": float(elevation),
        "skirt_m": float(skirt),
        "mujoco_size": [float(radius_x), float(radius_y), float(elevation), float(skirt)],
        "mujoco_pos": [float(v) for v in patch.mujoco_pos()],
        "statistics": patch.statistics(),
    }


def _write_heightfield(patch, name, output: Path) -> dict:
    spec = heightfield_spec(patch, name)
    (output / "terrain").mkdir(parents=True, exist_ok=True)
    (output / spec["file"]).write_bytes(patch.binary())
    return spec


def _box_pose(mesh):
    """(size, pose "x y z roll pitch yaw") when `mesh` is a box — eight corners whose
    oriented bounding box it fills — else None. Gazebo collides a primitive box
    natively; a mesh goes through its slowest, least stable triangle path."""
    if len(mesh.vertices) > 64:
        return None
    mesh = mesh.convex_hull  # judge the shape, not how its vertices were split
    if len(mesh.vertices) > 8 or not mesh.is_volume:
        return None
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    box_volume = float(np.prod(extents))
    if np.min(extents) <= 1e-6 or abs(box_volume - abs(mesh.volume)) > 0.02 * box_volume:
        return None
    box_to_body = np.linalg.inv(to_origin)
    roll, pitch, yaw = trimesh.transformations.euler_from_matrix(box_to_body, "sxyz")
    return extents, numbers((*box_to_body[:3, 3], roll, pitch, yaw))


def _sdf_collision_geometry(collision, uri, mesh, convex):
    box = _box_pose(mesh) if mesh is not None else None
    if box is None:
        _sdf_mesh(collision, uri, convex=convex)
        return
    size, pose = box
    child(collision, "pose", pose)
    child(child(child(collision, "geometry"), "box"), "size", numbers(size))


def _sdf_mesh(parent, uri, convex):
    mesh = child(child(parent, "geometry"), "mesh")
    child(mesh, "uri", uri)
    child(mesh, "scale", "1 1 1")
    # Convex pieces are already hulls on disk. The SDF <optimization> hint is not
    # defined in current Gazebo releases, which warned once per mesh; it is not written.
    del convex


def _sdf_surface(collision, obj):
    surface = child(collision, "surface")
    friction = child(child(surface, "friction"), "ode")
    child(friction, "mu", obj.friction)
    child(friction, "mu2", obj.friction)
    child(child(surface, "bounce"), "restitution_coefficient", obj.restitution)


# Gazebo collides two shapes only when their collide_bitmasks share a bit. Scripted
# (kinematic) bodies follow their program whatever touches them, so their contact with
# the static world and with each other is pure solver cost; they must still hit loose
# props and any robot (default bitmask 0xffff).
STATIC_MASK, KINEMATIC_MASK, DYNAMIC_MASK = "0x01", "0x02", "0x03"


def _sdf_contact_classes(sdf):
    """Merge each patrol group's collisions onto one carrier link and set the bitmasks.

    A walking worker is several parts (boots, trousers, vest, helmet) on one exact
    trajectory; as separate kinematic models their overlapping shapes collided with
    each other every step — on a warehouse that was 5x the whole simulation's cost.
    The carrier keeps every part's shape, so what a robot meets is unchanged."""
    models = {model.get("name"): model for model in sdf.findall("model")}
    groups: dict[str, list] = {}
    for model in models.values():
        drive = model.get("nova_drive")
        if drive:
            groups.setdefault(drive, []).append(model)
    for members in groups.values():
        carrier = members[0].find("link")
        for model in members[1:]:
            link = model.find("link")
            for collision in list(link.findall("collision")):
                link.remove(collision)
                collision.set("name", f"{model.get('name')}_{collision.get('name')}")
                carrier.append(collision)
    for model in models.values():
        model.attrib.pop("nova_drive", None)
        static = (model.findtext("static") or "").strip() == "true"
        for link in model.findall("link"):
            kinematic = (link.findtext("kinematic") or "").strip() == "true"
            mask = KINEMATIC_MASK if kinematic else STATIC_MASK if static else DYNAMIC_MASK
            for collision in link.findall("collision"):
                surface = collision.find("surface")
                if surface is None:
                    surface = child(collision, "surface")
                contact = surface.find("contact")
                if contact is None:
                    contact = child(surface, "contact")
                child(contact, "collide_bitmask", mask)


def sdf_static_model(scene, entries):
    model = ET.Element("model", name="nova_environment")
    child(model, "static", "true")
    link = child(model, "link", name="environment")
    for entry in entries:
        obj = entry["object"]
        if obj.body is not None:
            continue
        name = entry["id"]
        visual = child(link, "visual", name=name + "_visual")
        _sdf_mesh(visual, entry["visual"], convex=False)
        material = child(visual, "material")
        child(material, "ambient", numbers(obj.color))
        child(material, "diffuse", numbers(obj.color))
        # Static contact uses the same convex pieces as MuJoCo (boxes become primitive
        # boxes); exact triangle meshes made every loose prop a trimesh-trimesh pair in
        # Gazebo's ODE detector — about 3 physics steps per second on a warehouse.
        # Terrain keeps its exact surface.
        if entry["heightfield"] or not entry["convex_meshes"]:
            for part, uri in enumerate(entry["collision"]):
                collision = child(link, "collision", name=f"{name}_collision_{part:02d}")
                _sdf_mesh(collision, uri, convex=obj.collision_meshes is not None)
                _sdf_surface(collision, obj)
        else:
            for part, (uri, mesh) in enumerate(
                zip(entry["convex"], entry["convex_meshes"], strict=True)
            ):
                collision = child(link, "collision", name=f"{name}_collision_{part:02d}")
                _sdf_collision_geometry(collision, uri, mesh, convex=True)
                _sdf_surface(collision, obj)
    for name, position in (("spawn", scene.spawn_position), ("goal", scene.goal_position)):
        frame = child(model, "frame", name=name, attached_to="environment")
        child(frame, "pose", numbers((*position, 0, 0, 0)), relative_to="environment")
    return model


def sdf_movable_model(entry):
    """One non-static model per movable object: inertial link, convex contact and,
    for hinges/sliders, a passive joint whose parent is the world."""
    obj, body = entry["object"], entry["object"].body
    model = ET.Element("model", name=f"nova_environment_{entry['id']}")
    child(model, "static", "false")
    child(model, "pose", numbers((*body.origin, 0, 0, 0)))
    link = child(model, "link", name="body")
    inertial = child(link, "inertial")
    child(inertial, "pose", numbers((*body.center_of_mass, 0, 0, 0)))
    child(inertial, "mass", body.mass)
    inertia = child(inertial, "inertia")
    for key, value in zip(("ixx", "iyy", "izz"), body.diagonal_inertia, strict=True):
        child(inertia, key, value)
    for key in ("ixy", "ixz", "iyz"):
        child(inertia, key, 0)
    if body.motion == "kinematic":
        # SDF's native prescribed-motion carrier: a kinematic link. Its pose is
        # set externally (the motion program); dynamic bodies still collide with it.
        child(link, "kinematic", "true")
        if body.drive is not None:
            from .motion import drive_dict

            # Parts sharing one exact trajectory form a patrol group (merged later).
            model.set("nova_drive", json.dumps(
                [drive_dict(body.drive), [round(v, 6) for v in body.origin]], sort_keys=True))
    visual = child(link, "visual", name="visual")
    _sdf_mesh(visual, entry["visual"], convex=False)
    material = child(visual, "material")
    child(material, "ambient", numbers(obj.color))
    child(material, "diffuse", numbers(obj.color))
    meshes = entry.get("convex_meshes") or [None] * len(entry["convex"])
    for part, (uri, mesh) in enumerate(zip(entry["convex"], meshes, strict=True)):
        collision = child(link, "collision", name=f"collision_{part:02d}")
        _sdf_collision_geometry(collision, uri, mesh, convex=True)
        _sdf_surface(collision, obj)
    if body.motion in {"hinge", "slide"}:
        joint = child(model, "joint", name="joint", type=JOINT_TYPES[body.motion])
        child(joint, "parent", "world")
        child(joint, "child", "body")
        axis = child(joint, "axis")
        child(axis, "xyz", numbers(body.axis))
        if body.limits is not None:
            limit = child(axis, "limit")
            child(limit, "lower", body.limits[0])
            child(limit, "upper", body.limits[1])
        dynamics = child(axis, "dynamics")
        child(dynamics, "damping", body.damping)
        child(dynamics, "friction", body.frictionloss)
    return model


def urdf_movable_body(entry):
    """A self-contained URDF for one movable object, loaded by the PyBullet launcher
    at the body origin. Mesh paths are relative to the URDF's own folder."""
    obj, body = entry["object"], entry["object"].body
    robot = ET.Element("robot", name=f"nova_environment_{entry['id']}")
    if body.motion in {"hinge", "slide"}:
        anchor = child(child(robot, "link", name="world_anchor"), "inertial")
        child(anchor, "mass", value="0")
        child(anchor, "inertia", ixx="0", iyy="0", izz="0", ixy="0", ixz="0", iyz="0")
        joint_type = JOINT_TYPES[body.motion]
        if body.limits is None:
            joint_type = "continuous"  # a driven spin has no travel limits
        joint = child(robot, "joint", name="joint", type=joint_type)
        child(joint, "parent", link="world_anchor")
        child(joint, "child", link="body")
        child(joint, "origin", xyz="0 0 0", rpy="0 0 0")
        child(joint, "axis", xyz=numbers(body.axis))
        if body.limits is not None:
            child(
                joint,
                "limit",
                lower=format(body.limits[0], ".9g"),
                upper=format(body.limits[1], ".9g"),
                effort="1000",
                velocity="10",
            )
        child(
            joint,
            "dynamics",
            damping=format(body.damping, ".9g"),
            friction=format(body.frictionloss, ".9g"),
        )
    link = child(robot, "link", name="body")
    inertial = child(link, "inertial")
    child(inertial, "origin", xyz=numbers(body.center_of_mass), rpy="0 0 0")
    child(inertial, "mass", value=format(body.mass, ".9g"))
    ixx, iyy, izz = body.diagonal_inertia
    child(
        inertial,
        "inertia",
        ixx=format(ixx, ".9g"),
        iyy=format(iyy, ".9g"),
        izz=format(izz, ".9g"),
        ixy="0",
        ixz="0",
        iyz="0",
    )
    visual = child(link, "visual")
    child(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    child(child(visual, "geometry"), "mesh", filename=Path(entry["visual"]).name)
    child(child(visual, "material", name="surface"), "color", rgba=numbers(obj.color))
    for uri in entry["convex"]:
        collision = child(link, "collision")
        child(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        child(child(collision, "geometry"), "mesh", filename=Path(uri).name)
    return robot


def export_native(scene, output: Path, targets=None) -> dict:
    """Vertices already use model/world coordinates, metres and right-handed Z up."""
    from .mujoco_environment import export_mujoco_environment

    targets = {"gazebo", "mujoco"} if targets is None else set(targets)
    if not np.isfinite(scene.gravity_m_s2):
        raise ValueError("Non-finite gravity")
    entries = environment_meshes(scene, output)
    movable = [entry for entry in entries if entry["object"].body is not None]
    joints = [entry for entry in movable if entry["object"].body.motion != "free"]
    warnings = [
        "USD/SDF static collision geometry reuses the visual mesh; "
        "collision simplification is not performed.",
        "Contact is convex where a target requires it (MuJoCo everywhere, movable bodies "
        "everywhere): the author's collision_meshes, else the host's automatic convex "
        "decomposition of each concave part, else a single hull for convex parts.",
        "Environment mechanisms are passive world-attached hinges/sliders, free bodies "
        "and prescribed kinematic drives (motion_program.json): no latches, closers, "
        "springs or nested assemblies.",
        "MuJoCo contact response uses its native solver defaults; restitution is not mapped.",
        "Gazebo ODE mu/mu2 are both mapped from the requested friction. "
        "They are directional coefficients, not separate static/dynamic friction.",
    ]
    ledger = (
        "Nova generated environment; SDFormat 1.12; SI metres/kg/radians; right-handed Z-up. "
        "Static geometry is baked into the nova_environment model frame. Each movable object "
        "is its own non-static model posed at its body origin: inertial link (mass, COM, "
        "diagonal inertia about COM), convex contact, and for doors/drawers a passive "
        "revolute/prismatic joint to world with limits, damping and friction. Free props "
        "have no joint. No robot dynamics or plugins synthesized. Gravity/contact "
        "coefficients come from scene source plus optional user overrides. World timestep "
        "is an explicit 0.002 second default. Runtime physics untested."
    )
    if "gazebo" in targets:
        sdf = ET.Element("sdf", version="1.12")
        sdf.append(sdf_static_model(scene, entries))
        for entry in movable:
            sdf.append(sdf_movable_model(entry))
        _sdf_contact_classes(sdf)
        write_xml(output / "model.sdf", sdf, ledger)
        world_sdf = ET.Element("sdf", version="1.12")
        world = child(world_sdf, "world", name="nova_world")
        child(world, "gravity", numbers((0, 0, -scene.gravity_m_s2)))
        child(child(world, "physics", name="default_physics", type="ode"), "max_step_size", "0.002")
        for model in sdf.findall("model"):
            world.append(copy.deepcopy(model))
        light = child(world, "light", name="sun", type="directional")
        child(light, "pose", "0 0 10 0 0 0", relative_to="world")
        child(light, "direction", "-0.5 0.1 -1")
        write_xml(output / "world.sdf", world_sdf, ledger)
    if "pybullet" in targets:
        for entry in movable:
            write_xml(
                output / "meshes" / f"{entry['id']}.urdf",
                urdf_movable_body(entry),
                "Nova movable environment body for the bundled PyBullet launcher. SI metres/"
                "kg/radians. Loaded at its body origin; hinge/slider bodies use a fixed "
                "world_anchor base with one passive joint. Inertia is about COM.",
            )
    report = {
        "gazebo": {
            "entrypoint": "world.sdf",
            "model": "model.sdf",
            "format": "sdf",
            "schema_version": "1.12",
            "xml_reopens": True,
            "simulator_runtime": "not_run",
            "timestep_s": 0.002,
            "movable_models": len(movable),
            "passive_joints": len(joints),
            "dynamic_collision": "convex_hull_or_explicit_pieces",
        },
        "pybullet": {
            "entrypoint": "run_pybullet.py",
            "format": "urdf_bodies+launcher",
            "movable_bodies": len(movable),
            "passive_joints": len(joints),
            "simulator_runtime": "not_run",
            "body_files": [f"meshes/{entry['id']}.urdf" for entry in movable],
        },
        "warnings": warnings,
    }
    if "mujoco" in targets:
        mujoco_report = export_mujoco_environment(scene, output, entries)
        report["mujoco"] = mujoco_report["mujoco"]
        report["mujoco"].update(
            xml_reopens=True, mujoco_version=mujoco.__version__, simulator_runtime="not_run"
        )
        warnings.extend(w for w in mujoco_report["warnings"] if w not in warnings)
    if scene.water_volumes:
        (output / "nova_water.py").write_bytes(
            Path(__file__).with_name("nova_water.py").read_bytes()
        )
        (output / "water_volumes.json").write_text(
            json.dumps([volume.specification() for volume in scene.water_volumes], indent=2),
            encoding="utf-8",
        )
    if any(e["object"].body is not None and e["object"].body.drive is not None
           for e in entries):
        # The exact evaluators every runner uses, shipped beside the package so a
        # customer environment needs nothing but the download.
        (output / "nova_motion.py").write_bytes(
            Path(__file__).with_name("motion.py").read_bytes()
        )
        if "mujoco" in targets:
            # A bare MuJoCo viewer shows a static world; ship the evaluator that
            # plays the program there, exactly like run_pybullet.py does.
            (output / "run_mujoco.py").write_bytes(
                Path(__file__).with_name("mujoco_runner.py").read_bytes()
            )
        if "gazebo" in targets:
            # Same for Gazebo: world.sdf alone leaves kinematic links and joints still.
            (output / "run_gazebo.py").write_bytes(
                Path(__file__).with_name("gazebo_runner.py").read_bytes()
            )
        program = motion_program(entries)
        (output / "motion_program.json").write_text(
            json.dumps(program, allow_nan=False, indent=2), encoding="utf-8"
        )
    if list((output / "textures").glob("*.png")):
        warnings.append(
            "Native SDF/MJCF/PyBullet exports currently use solid colors; image textures "
            "are bound in USD only. Texture originals are included."
        )
    return {key: value for key, value in report.items() if key in targets or key == "warnings"}
