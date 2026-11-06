"""Bounded native SDF assembly; preserves source frames and refuses unhandled scopes."""

import copy

import numpy as np
import trimesh

from .assembly import Y_UP
from .native_exports import write_xml
from .uploads import read_xml


def assemble_sdf(output, robot_path, scene):
    document = read_xml(robot_path.read_bytes())
    models = document.findall("model")
    if len(models) != 1 or document.find("world") is not None:
        raise ValueError("Gazebo requires one robot model SDF, not a world")
    model = copy.deepcopy(models[0])
    if (
        next(model.iter("plugin"), None) is not None
        or next(model.iter("include"), None) is not None
        or model.find("model") is not None
    ):
        raise ValueError(
            "Gazebo robot plugins, includes and nested models need a dedicated runtime "
            "adapter; upload a standalone plugin-free robot model"
        )
    if model.findtext("static", "false").strip().lower() in {"true", "1"}:
        raise ValueError("A static warehouse is not a dynamic robot")
    nodes = {"__model__": None}
    for tag in ("link", "joint", "frame"):
        for node in model.findall(tag):
            name = node.get("name")
            if not name or name in nodes:
                raise ValueError("SDF robot has duplicate/missing frame names")
            nodes[name] = node
    transforms = {"__model__": np.eye(4)}

    def pose_matrix(pose):
        if pose is None:
            return np.eye(4)
        if (
            pose.get("rotation_format", "euler_rpy") != "euler_rpy"
            or pose.get("degrees", "false") != "false"
        ):
            raise ValueError("SDF preview requires radians and Euler RPY poses")
        values = [float(x) for x in (pose.text or "").split()]
        if len(values) != 6 or not np.isfinite(values).all():
            raise ValueError("Invalid SDF pose")
        result = trimesh.transformations.euler_matrix(*values[3:], axes="sxyz")
        result[:3, 3] = values[:3]
        return result

    def resolve(name, visited=()):
        if name in transforms:
            return transforms[name]
        if name in visited or name not in nodes:
            raise ValueError(f"Unresolved/cyclic SDF frame: {name}")
        node = nodes[name]
        pose = node.find("pose")
        default = (
            node.findtext("child", "__model__")
            if node.tag == "joint"
            else node.get("attached_to", "__model__")
        )
        relative = pose.get("relative_to", default) if pose is not None else default
        transforms[name] = resolve(relative, (*visited, name)) @ pose_matrix(pose)
        return transforms[name]

    preview = trimesh.load(output.parent / "model.glb", force="scene", process=False)
    placement = np.eye(4)
    placement[:3, 3] = scene.spawn_position
    count = 0
    for link in model.findall("link"):
        if link.find("collision") is None or link.find("inertial") is None:
            raise ValueError(
                f"SDF robot link {link.get('name')} requires supplied collision and inertial data"
            )
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError("SDF visual needs one geometry")
            item = geometry[0]
            if item.tag == "box":
                mesh = trimesh.creation.box(
                    extents=[float(x) for x in item.findtext("size", "").split()]
                )
            elif item.tag == "sphere":
                mesh = trimesh.creation.icosphere(
                    subdivisions=2, radius=float(item.findtext("radius"))
                )
            elif item.tag == "cylinder":
                mesh = trimesh.creation.cylinder(
                    radius=float(item.findtext("radius")),
                    height=float(item.findtext("length")),
                    sections=24,
                )
            elif item.tag == "mesh":
                uri = item.findtext("uri", "")
                path = (robot_path.parent / uri).resolve()
                if (
                    ":" in uri
                    or not path.is_relative_to((output / "robot").resolve())
                    or not path.is_file()
                ):
                    raise ValueError("SDF robot needs portable, local relative mesh URIs")
                if (
                    path.suffix.lower() not in {".obj", ".stl", ".ply"}
                    or item.find("submesh") is not None
                ):
                    raise ValueError(
                        "SDF preview supports whole OBJ/STL/PLY meshes; "
                        "convert other visual formats first"
                    )
                mesh = trimesh.load(path, force="mesh", process=False)
                mesh.apply_scale([float(x) for x in item.findtext("scale", "1 1 1").split()])
            else:
                raise ValueError(f"Unsupported SDF robot preview geometry: {item.tag}")
            if mesh.is_empty or not np.isfinite(mesh.vertices).all():
                raise ValueError("Invalid SDF robot visual")
            pose = visual.find("pose")
            relative = (
                pose.get("relative_to", link.get("name")) if pose is not None else link.get("name")
            )
            preview.add_geometry(
                mesh,
                node_name=f"nova_robot_visual_{count}",
                transform=Y_UP @ placement @ resolve(relative) @ pose_matrix(pose),
            )
            count += 1
    if not count:
        raise ValueError("Robot has no supported visuals")
    # URI rebasing only, without changing the canonical uploaded SDF or its mesh bytes.
    for node in model.iter("uri"):
        uri = (node.text or "").strip()
        path = (robot_path.parent / uri).resolve()
        if (
            ":" in uri
            or not path.is_relative_to((output / "robot").resolve())
            or not path.is_file()
        ):
            raise ValueError(f"Unresolved SDF resource: {uri}")
        node.text = path.relative_to(output.resolve()).as_posix()
    import xml.etree.ElementTree as ET

    old_pose = model.find("pose")
    if old_pose is not None:
        model.remove(old_pose)
    ET.SubElement(model, "pose", relative_to="world").text = " ".join(
        map(str, (*scene.spawn_position, 0, 0, 0))
    )
    world = ET.parse(output / "world.sdf").getroot()
    if (model.get("name") or "").startswith("nova_environment"):
        raise ValueError("Robot model uses reserved nova_environment name prefix")
    world.find("world").append(model)
    write_xml(
        output / "simulation.sdf",
        world,
        "Nova robot assembly; SDFormat 1.12; SI metres/radians, Z-up. "
        "Robot internal frames, axes, masses and inertias copied from the uploaded model. "
        "Robot model pose explicitly placed at the scene spawn in world coordinates; "
        "mesh URIs rebased only. Environment mechanisms are passive world-attached joints "
        "and free bodies in their own models; no controllers/plugins synthesized. "
        "Gazebo runtime and joint motion have not been tested.",
    )
    (output.parent / "model.glb").write_bytes(preview.export(file_type="glb"))
    return "simulation.sdf", {
        "status": "not_run",
        "engine": "Gazebo",
        "reason": "Gazebo runtime unavailable in this compiler",
        "training": "not_run",
    }
