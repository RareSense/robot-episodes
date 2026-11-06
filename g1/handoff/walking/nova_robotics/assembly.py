"""Native robot assembly in the compiler cage. No uploaded scripts/plugins execute."""

import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from .nova_water import MuJoCoWater
from .uploads import read_xml, resolve_reference

Y_UP = trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])


def _geometry(model, index):
    kind = model.geom_type[index]
    size = model.geom_size[index]
    types = mujoco.mjtGeom
    if kind == types.mjGEOM_MESH:
        mesh_id = model.geom_dataid[index]
        va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        return trimesh.Trimesh(
            model.mesh_vert[va : va + vn].copy(),
            model.mesh_face[fa : fa + fn].copy(),
            process=False,
        )
    if kind == types.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * size)
    if kind == types.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    if kind == types.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.apply_scale(size)
        return mesh
    if kind == types.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=24)
    if kind == types.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[12, 12])
        # trimesh capsule is centered on the origin in this runtime.
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        return mesh
    raise ValueError(
        f"Robot preview does not support MuJoCo geometry type {kind}; "
        "no incomplete preview will be exported"
    )


def _pose(data, i):
    pose = np.eye(4)
    pose[:3, :3] = data.geom_xmat[i].reshape(3, 3)
    pose[:3, 3] = data.geom_xpos[i]
    return Y_UP @ pose


def _body_pose(data, index):
    pose = np.eye(4)
    pose[:3, :3] = data.xmat[index].reshape(3, 3)
    pose[:3, 3] = data.xpos[index]
    return pose


def add_replay(
    glb_path: Path,
    times,
    tracks,
    name="Physics replay — passive gravity/contact (not training)",
):
    """Embed measured transforms as standard glTF animation, played by Studio's timeline."""
    raw = glb_path.read_bytes()
    json_len = struct.unpack_from("<I", raw, 12)[0]
    document = json.loads(raw[20 : 20 + json_len])
    binary = bytearray(raw[28 + json_len :])
    views, accessors = document.setdefault("bufferViews", []), document.setdefault("accessors", [])

    def accessor(values, kind):
        array = np.asarray(values, dtype="<f4")
        while len(binary) % 4:
            binary.append(0)
        view = len(views)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": array.nbytes})
        binary.extend(array.tobytes())
        item = {"bufferView": view, "componentType": 5126, "count": len(array), "type": kind}
        if kind == "SCALAR":
            item.update(min=[float(array.min())], max=[float(array.max())])
        accessors.append(item)
        return len(accessors) - 1

    time_accessor = accessor(times, "SCALAR")
    animation = {
        "name": name,
        "samplers": [],
        "channels": [],
    }
    for index, node in enumerate(document["nodes"]):
        matrices = tracks.get(node.get("name"))
        if matrices is None:
            continue
        translations = [m[:3, 3].tolist() for m in matrices]
        rotations = []
        for m in matrices:
            q = trimesh.transformations.quaternion_from_matrix(m)
            q = np.roll(q, -1)  # wxyz -> xyzw
            if rotations and np.dot(q, rotations[-1]) < 0:
                q = -q
            rotations.append(q.tolist())
        node.pop("matrix", None)
        node["translation"], node["rotation"] = translations[0], rotations[0]
        node["scale"] = [1, 1, 1]
        for path, values, kind in (
            ("translation", translations, "VEC3"),
            ("rotation", rotations, "VEC4"),
        ):
            sampler = len(animation["samplers"])
            animation["samplers"].append(
                {
                    "input": time_accessor,
                    "output": accessor(values, kind),
                    "interpolation": "LINEAR",
                }
            )
            animation["channels"].append(
                {"sampler": sampler, "target": {"node": index, "path": path}}
            )
    document["animations"] = [animation]
    document["buffers"] = [{"byteLength": len(binary)}]
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    result = struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary))
    result += struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
    result += struct.pack("<II", len(binary), 0x004E4942) + binary
    glb_path.write_bytes(result)


def assemble_mujoco(output, robot_path, scene, controller=None):
    root = read_xml(robot_path.read_bytes())
    if (
        next(root.iter("include"), None) is not None
        or next(root.iter("plugin"), None) is not None
        or root.find("extension") is not None
    ):
        raise ValueError(
            "MuJoCo robot includes/plugins require a flattened, plugin-free MJCF package "
            "before simulation"
        )
    worldbody = root.find("worldbody")
    if worldbody is None or not worldbody.findall("body") or worldbody.findall("geom"):
        raise ValueError("Select a robot-only MJCF with bodies, not a complete world/arena")
    package_root = output / "robot" / robot_path.relative_to(output / "robot").parts[0]
    owner = robot_path.relative_to(package_root).as_posix()
    members = {
        p.relative_to(package_root).as_posix(): p for p in package_root.rglob("*") if p.is_file()
    }
    compiler = root.find("compiler")
    options = dict(compiler.attrib) if compiler is not None else {}
    for index, node in enumerate(root.iter()):
        if "file" not in node.attrib:
            continue
        directory = options.get(
            "meshdir" if node.tag == "mesh" else "texturedir", options.get("assetdir", "")
        )
        ref = str(Path(directory) / node.get("file"))
        resolved = resolve_reference(owner, ref, set(members))
        if resolved is None:
            raise ValueError(f"Unresolved or external robot asset: {ref}")
        # MuJoCo derives unnamed asset names from the original file basename.
        # Relocation must not change those names: geoms/materials still reference them.
        if node.tag in {"mesh", "texture", "hfield", "skin"} and not node.get("name"):
            node.set("name", Path(node.get("file").replace("\\", "/")).stem)
        relative = f"robot_runtime_assets/asset_{index}{Path(resolved).suffix}"
        target = output / relative
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(members[resolved].read_bytes())
        node.set("file", relative)
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir", "strippath"):
            compiler.attrib.pop(key, None)
    # Preserve source body poses, axes, inertias, actuators, solver settings and defaults.
    # The outer frame adds only the requested environment spawn translation.
    frame = ET.Element("frame", pos=" ".join(map(str, scene.spawn_position)))
    for body in list(worldbody.findall("body")):
        worldbody.remove(body)
        frame.append(body)
    worldbody.append(frame)
    # Files are allowlisted, local compiler output; no remote fetch or plugin activation.
    assets = (
        {
            p.relative_to(output).as_posix(): p.read_bytes()
            for p in (output / "robot_runtime_assets").glob("*")
        }
        if (output / "robot_runtime_assets").exists()
        else {}
    )
    from .mujoco_environment import disable_asset_cache

    disable_asset_cache()
    robot = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"), assets=assets)
    robot.option.gravity = [0, 0, -scene.gravity_m_s2]
    robot.option.timestep = 0.002
    env = mujoco.MjSpec.from_file(str(output / "environment.xml"))
    # The attached environment's arena size does not propagate; carry it over.
    if int(env.memory) > 0:
        robot.memory = max(int(robot.memory), int(env.memory))
    if any(
        (g.name or "").startswith("nova_environment_")
        for g in [*robot.geoms, *robot.bodies, *robot.joints]
    ):
        raise ValueError("Robot uses reserved nova_environment_ name prefix")
    robot.attach(env, prefix="nova_environment_", frame=robot.worldbody.add_frame())
    model = robot.compile()
    if model.nv == 0 or model.nbody > 512 or model.ngeom > 8000:
        raise ValueError("Robot requires movable joints and must fit the bounded simulation budget")
    (output / "simulation.xml").write_text(robot.to_xml())
    # Reopen the actual deliverable, not merely the in-memory assembly.
    model = mujoco.MjModel.from_xml_path(str(output / "simulation.xml"))
    mission = None
    if controller == "household_mission_v1":
        from .mission_controller import MissionController, prepare_spec, robot_geometry
        from .mission_planner import MissionPlanner

        probe = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, probe, 0)
        mujoco.mj_forward(model, probe)
        robot_root = next(
            int(model.jnt_bodyid[j])
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(
                "nova_environment_"
            )
        )
        geometry = robot_geometry(model, probe, robot_root)
        plan = MissionPlanner(
            scene, output / "environment.xml", geometry, list(scene.spawn_position)
        ).plan()
        (output / "mission_plan.json").write_text(json.dumps(plan, indent=1))
        (output / "run_mission.py").write_bytes(
            (Path(__file__).parent / "mission_controller.py").read_bytes()
        )
        (output / "requirements-mission.txt").write_text(
            f"mujoco=={mujoco.__version__}\nnumpy>=2,<3\n"
        )
        model, robot_body_name = prepare_spec(output / "simulation.xml", plan)
        mission = (plan, robot_body_name)
    # Preserve actuators in the downloadable model; disable them only for this
    # passive test. Zero position-control commands would still apply joint forces.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    data = mujoco.MjData(model)
    control = None
    if mission is not None:
        control = MissionController(model, data, mission[0], mission[1])
    elif controller:
        from .go2_controller import Go2Controller

        if controller != "go2_walk_v1" or root.get("model", "").lower() != "go2":
            raise ValueError("The selected walking controller is a Go2-only demonstration")
        policy_dir = Path(__file__).parent / "policies/go2_walk_v1"
        (output / "controller").mkdir(exist_ok=True)
        for asset in policy_dir.iterdir():
            if asset.is_file():
                (output / "controller" / asset.name).write_bytes(asset.read_bytes())
        (output / "run_controller.py").write_bytes(
            (Path(__file__).parent / "go2_controller.py").read_bytes()
        )
        (output / "requirements-controller.txt").write_text(
            f"mujoco=={mujoco.__version__}\nnumpy>=2,<3\nonnxruntime>=1.22,<2\n"
        )
        control = Go2Controller(model, data, output / "controller/policy.onnx")
    mujoco.mj_forward(model, data)
    ids = [
        i
        for i in range(model.ngeom)
        if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
            ("nova_environment_", "nova_mission_")
        )
        and not (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i]) or ""
        ).startswith("nova_mission_")
    ]
    glb = output.parent / "model.glb"
    preview = trimesh.load(glb, force="scene", process=False)
    for i in ids:
        geometry = _geometry(model, i)
        geometry.visual.face_colors = np.tile(
            model.geom_rgba[i] * 255, (len(geometry.faces), 1)
        ).astype(np.uint8)
        preview.add_geometry(geometry, node_name=f"nova_robot_geom_{i}", transform=_pose(data, i))
    tracks = {f"nova_robot_geom_{i}": [_pose(data, i)] for i in ids}
    environment_tracks = []
    for index, obj in enumerate(scene.objects):
        if obj.body is None:
            continue
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"nova_environment_object_{index:04d}_body"
        )
        if body_id < 0:
            raise ValueError(f"Moving environment object disappeared: {obj.name}")
        # Match the body's pivot frame used by the preview exporter and live runtime.
        tracks[obj.name] = [Y_UP @ _body_pose(data, body_id) @ Y_UP.T]
        environment_tracks.append((obj.name, body_id, Y_UP.T))
    times = [0.0]
    maximum_contacts = 0
    water = (
        MuJoCoWater([v.specification() for v in scene.water_volumes], scene.gravity_m_s2)
        if scene.water_volumes
        else None
    )
    if mission is not None:
        steps = int(round(mission[0].get("planned_duration_s", 0) / model.opt.timestep)) + 500
        record_every = 25
    else:
        steps = 6000 if control else 1000
        record_every = 10
    for step in range(steps):
        if mission is not None:
            if not control.before_step():
                break
        elif control:
            control.before_step(model, data)
        if water is not None:
            # Neither bundled controller writes xfrc_applied; water owns the buffer.
            data.xfrc_applied[:] = 0.0
            water.apply(model, data)
        mujoco.mj_step(model, data)
        if mission is not None:
            control.after_step()
        elif control:
            control.after_step(model, data)
        maximum_contacts = max(maximum_contacts, data.ncon)
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or np.any(data.warning.number)
        ):
            raise ValueError(
                "Robot physics test became unstable; check inertias, initial pose, "
                "contacts and timestep"
            )
        if (step + 1) % record_every == 0:
            mujoco.mj_forward(model, data)
            times.append(float(data.time))
            for i in ids:
                tracks[f"nova_robot_geom_{i}"].append(_pose(data, i))
            for name, body_id, rest_inverse in environment_tracks:
                tracks[name].append(Y_UP @ _body_pose(data, body_id) @ rest_inverse)
    glb.write_bytes(preview.export(file_type="glb"))
    if mission is not None:
        add_replay(glb, times, tracks, "Household mission — scripted demonstration (not a policy)")
    elif control:
        add_replay(glb, times, tracks, "Go2 controller — learned walking policy (not training)")
    else:
        add_replay(glb, times, tracks)
    report = {
        "status": "passed",
        "engine": f"MuJoCo {mujoco.__version__}",
        "mode": "passive_actuation_disabled",
        "duration_s": round(float(data.time), 6),
        "steps": steps,
        "maximum_contacts": maximum_contacts,
        "robot_dofs": model.nv,
        "actuators": model.nu,
        "training": "not_run",
        "scope": "finite-state smoke test; no traversal, stability or sim-to-real certification",
    }
    if mission is not None:
        report = control.report()
        (output / "mission_report.json").write_text(json.dumps(report, indent=1))
    elif control:
        report = control.report(model, data)
    environment_dofs = sum(
        6 if obj.body.motion == "free" else 1 for obj in scene.objects if obj.body
    )
    report.update(
        robot_dofs=model.nv - environment_dofs,
        environment_dofs=environment_dofs,
        moving_environment_bodies=len(environment_tracks),
        environment_replay="recorded" if environment_tracks else "static",
        real_world_calibrated=False,
    )

    # Long missions produce hundreds of thousands of poses; 4x4 matrices at full float
    # precision made this file 129 MB. Poses are stored as rounded translation plus
    # unit quaternion (wxyz), and the file is rebuilt into matrices on read.
    def compact(matrix):
        q = trimesh.transformations.quaternion_from_matrix(matrix)
        return [round(float(v), 4) for v in matrix[:3, 3]] + [round(float(v), 5) for v in q]

    (output / "physics_replay.json").write_text(
        json.dumps(
            {
                "report": report,
                "time_s": [round(float(t), 4) for t in times],
                "pose_format": "translation_xyz+quaternion_wxyz, Y-up viewport frame",
                "poses_y_up": {name: [compact(m) for m in poses] for name, poses in tracks.items()},
            },
            separators=(",", ":"),
        )
    )
    entry = "simulation.xml"
    if mission is not None:
        entry = "run_mission.py"
    elif control:
        entry = "run_controller.py"
    return entry, report


def assemble_usd(output, robot_path, scene, model):
    if not (
        model.get("has_articulation")
        and model.get("default_prim")
        and model.get("up_axis") == "Z"
        and model.get("meters_per_unit") == 1.0
    ):
        raise ValueError(
            "Isaac Sim requires a complete native articulated USD with defaultPrim, "
            "metre units and Z-up; warehouse USD is not a robot"
        )
    robot_stage = Usd.Stage.Open(str(robot_path))
    if not robot_stage or not any(
        p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in robot_stage.Traverse()
    ):
        raise ValueError("USD robot articulation did not compose")
    if any(p.IsA(UsdPhysics.Scene) for p in robot_stage.Traverse()):
        raise ValueError(
            "Upload a robot-only USD without its own PhysicsScene to avoid conflicting worlds"
        )
    stage = Usd.Stage.CreateNew(str(output / "simulation.usda"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Environment").GetPrim().GetReferences().AddReference(
        "environment.usda"
    )
    robot = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot.GetPrim().GetReferences().AddReference(robot_path.relative_to(output).as_posix())
    robot.AddTranslateOp(opSuffix="novaPlacement").Set(Gf.Vec3d(*scene.spawn_position))
    stage.GetRootLayer().Save()
    preview = trimesh.load(output.parent / "model.glb", force="scene", process=False)
    cache = UsdGeom.XformCache()
    count = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot")):
        if prim.IsInstance() or prim.IsA(UsdGeom.Boundable) and not prim.IsA(UsdGeom.Mesh):
            raise ValueError(
                "USD robot preview requires non-instanced triangle meshes; "
                "unsupported geometry cannot be silently omitted"
            )
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get())
        indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        faces, offset = [], 0
        for n in mesh.GetFaceVertexCountsAttr().Get() or []:
            if n != 3:
                raise ValueError(
                    "Robot USD preview requires triangulated meshes; export a triangulated robot"
                )
            faces.append(indices[offset : offset + n])
            offset += n
        transform = Y_UP @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        geometry = trimesh.Trimesh(vertices, faces, process=False)
        if not np.isfinite(vertices).all():
            raise ValueError("Robot mesh has non-finite vertices")
        preview.add_geometry(geometry, node_name=f"nova_robot_mesh_{count}", transform=transform)
        count += 1
    if not count:
        raise ValueError("No renderable robot meshes found in the USD articulation")
    (output.parent / "model.glb").write_bytes(preview.export(file_type="glb"))
    return "simulation.usda", {
        "status": "not_run",
        "engine": "Isaac Sim",
        "reason": "Isaac Sim runtime unavailable in this compiler",
        "training": "not_run",
    }
