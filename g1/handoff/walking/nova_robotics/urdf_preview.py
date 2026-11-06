"""URDF zero-pose visuals. Source frames and dynamics are never invented or edited."""

import numpy as np
import trimesh

from .assembly import Y_UP
from .uploads import read_xml


def _origin(node):
    origin = node.find("origin")
    xyz = (
        [float(x) for x in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
    )
    rpy = (
        [float(x) for x in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
    )
    if len(xyz) != 3 or len(rpy) != 3 or not np.isfinite([xyz, rpy]).all():
        raise ValueError("Invalid URDF origin")
    transform = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    transform[:3, 3] = xyz
    return transform


def add_urdf_preview(output, robot_path, scene):
    root = read_xml(robot_path.read_bytes())
    links = {n.get("name"): n for n in root.findall("link")}
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    for joint in joints.values():
        kind = joint.get("type")
        if kind not in {"fixed", "continuous", "revolute", "prismatic"}:
            raise ValueError(
                "Unsupported URDF joint type; provide a fixed/continuous/revolute/prismatic model"
            )
        if joint.find("mimic") is not None:
            raise ValueError(
                "URDF mimic joints need a dedicated controller adapter; not silently ignored"
            )
        if kind != "fixed":
            axis = joint.find("axis")
            values = (
                [float(x) for x in axis.get("xyz", "1 0 0").split()]
                if axis is not None
                else [1, 0, 0]
            )
            if len(values) != 3 or not np.isfinite(values).all() or np.linalg.norm(values) == 0:
                raise ValueError("Robot joint has an invalid axis")
        if kind in {"revolute", "prismatic"}:
            limits = joint.find("limit")
            if limits is None:
                raise ValueError("Robot joint limits are required")
            lower, upper = float(limits.get("lower", "nan")), float(limits.get("upper", "nan"))
            if not np.isfinite([lower, upper]).all() or not lower <= 0 <= upper:
                raise ValueError(
                    "The robot's zero pose is outside its supplied limits; "
                    "supply a compatible rest pose"
                )
    frames = {}

    def frame(name):
        if name not in frames:
            j = joints.get(name)
            frames[name] = (
                np.eye(4) if j is None else frame(j.find("parent").get("link")) @ _origin(j)
            )
        return frames[name]

    preview = trimesh.load(output.parent / "model.glb", force="scene", process=False)
    placement = np.eye(4)
    placement[:3, 3] = scene.spawn_position
    count = 0
    for name, link in links.items():
        if link.find("visual") is not None or link.find("collision") is not None:
            if link.find("collision") is None or link.find("inertial") is None:
                raise ValueError(
                    f"Robot link {name} needs collision and inertial data; "
                    "Nova will not invent robot dynamics"
                )
            mass_node = link.find("inertial/mass")
            mass = float(mass_node.get("value", "nan")) if mass_node is not None else float("nan")
            tensor = link.find("inertial/inertia")
            if tensor is None or not np.isfinite(mass) or mass <= 0:
                raise ValueError(f"Robot link {name} has invalid inertial data")
            ixx, iyy, izz, ixy, ixz, iyz = [
                float(tensor.get(k, "nan")) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            ]
            inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
            if not np.isfinite(inertia).all() or np.linalg.eigvalsh(inertia).min() <= 0:
                raise ValueError(f"Robot link {name} has a nonphysical inertia tensor")
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError("Robot visual requires one supported geometry")
            item = geometry[0]
            if item.tag == "mesh":
                reference = item.get("filename", "")
                if ":" in reference:
                    raise ValueError(
                        "PyBullet packaging requires local relative mesh paths; "
                        "export a portable URDF without package:// or remote URIs"
                    )
                path = (robot_path.parent / reference).resolve()
                if not path.is_relative_to((output / "robot").resolve()) or not path.is_file():
                    raise ValueError("Robot mesh escapes its package or is missing")
                if path.suffix.lower() not in {".stl", ".obj", ".ply"}:
                    raise ValueError(
                        "URDF preview supports STL, OBJ and PLY meshes; "
                        "convert other visual mesh formats first"
                    )
                mesh = trimesh.load(path, force="mesh", process=False)
                scale = [float(x) for x in item.get("scale", "1 1 1").split()]
                if len(scale) != 3 or not np.isfinite(scale).all() or min(scale) <= 0:
                    raise ValueError("Invalid robot mesh scale")
                mesh.apply_scale(scale)
            elif item.tag == "box":
                size = [float(x) for x in item.get("size", "").split()]
                if len(size) != 3 or not np.isfinite(size).all() or min(size) <= 0:
                    raise ValueError("Invalid URDF box size")
                mesh = trimesh.creation.box(extents=size)
            elif item.tag == "sphere":
                mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(item.get("radius")))
            elif item.tag == "cylinder":
                mesh = trimesh.creation.cylinder(
                    radius=float(item.get("radius")), height=float(item.get("length")), sections=24
                )
            else:
                raise ValueError(f"Unsupported URDF visual geometry {item.tag}")
            if not np.isfinite(mesh.vertices).all() or mesh.is_empty:
                raise ValueError("Invalid robot visual mesh")
            color = visual.find("material/color")
            if color is not None:
                rgba = np.array([float(x) for x in color.get("rgba", "").split()])
                if len(rgba) != 4 or not np.isfinite(rgba).all() or np.any((rgba < 0) | (rgba > 1)):
                    raise ValueError("Invalid robot visual color")
                mesh.visual.face_colors = np.tile(rgba * 255, (len(mesh.faces), 1)).astype(np.uint8)
            preview.add_geometry(
                mesh,
                node_name=f"nova_robot_{name}_{count}",
                transform=Y_UP @ placement @ frame(name) @ _origin(visual),
            )
            count += 1
    if not count:
        raise ValueError("Robot has no supported visual geometry to show in Studio")
    (output.parent / "model.glb").write_bytes(preview.export(file_type="glb"))


def pybullet_item(index, obj):
    """One object as the bundled launcher loads it: the same record the build's own
    PyBullet checks use, so what is measured here is what the customer runs."""
    from .motion import drive_dict
    from .native_exports import heightfield_spec

    name = f"object_{index:04d}"
    return {
        "name": obj.name,
        "mesh": f"meshes/{name}.obj",
        # Explicit convex pieces of a fixed object, exactly as authored.
        "collision": [
            f"meshes/{name}_collision_{k:02d}.obj" for k in range(len(obj.collision_meshes))
        ]
        if obj.body is None and obj.collidable and obj.collision_meshes
        else [],
        "collidable": obj.collidable,
        "friction": obj.friction,
        "rolling_friction": obj.rolling_friction,
        "torsional_friction": obj.torsional_friction,
        "restitution": obj.restitution,
        "color": list(obj.color),
        # Ground loads as a native heightfield, not a concave mesh: the same grid
        # MuJoCo collides, read back from the same file.
        "heightfield": heightfield_spec(obj.heightfield, name) if obj.heightfield else None,
        # Movable objects load from their own URDF at the body origin; the OBJ above is
        # already localized to that origin.
        "body": {
            "urdf": f"meshes/{name}.urdf",
            "origin": list(obj.body.origin),
            "motion": obj.body.motion,
            "mass": obj.body.mass,
            "damping": obj.body.damping,
            "frictionloss": obj.body.frictionloss,
            "limits": list(obj.body.limits) if obj.body.limits else None,
            "drive": drive_dict(obj.body.drive) if obj.body.drive else None,
        }
        if obj.body is not None
        else None,
    }


def assemble_pybullet(output, robot_path, scene, config):
    if robot_path is not None:
        add_urdf_preview(output, robot_path, scene)
    # This is trusted Nova runtime code; customer scripts are not included or evaluated.
    import json
    from pathlib import Path

    launcher = Path(__file__).with_name("pybullet_launcher.py")
    (output / "run_pybullet.py").write_bytes(launcher.read_bytes())
    (output / "pybullet_scene.json").write_text(
        json.dumps(
            {
                "robot": robot_path.relative_to(output).as_posix() if robot_path else None,
                "fixed_base": config["fixed_base"],
                "spawn": scene.spawn_position,
                "gravity": scene.gravity_m_s2,
                "water_volumes": [volume.specification() for volume in scene.water_volumes],
                "objects": [pybullet_item(i, o) for i, o in enumerate(scene.objects)],
            }
        )
    )
    for i, o in enumerate(scene.objects):
        if o.body is not None and not (output / f"meshes/object_{i:04d}.urdf").is_file():
            raise ValueError(f"PyBullet body file missing for movable object {o.name}")
    movable = sum(o.body is not None for o in scene.objects)
    return "run_pybullet.py", {
        "status": "not_run",
        "engine": "PyBullet",
        "reason": "Run the bundled launcher in a Python environment with pybullet installed",
        "training": "not_run",
        "movable_bodies": movable,
    }
