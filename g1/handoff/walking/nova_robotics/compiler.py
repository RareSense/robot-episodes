"""Runs only inside the existing compiler sandbox, including native USD parsing."""

import hashlib
import json
import math
import random
import shutil
import struct
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from . import motion
from .collision import ensure_contact
from .configuration import (
    ACCEPTANCE_ENGINE,
    COMPATIBILITY_ENGINE,
    TARGETS,
    apply_configuration,
    controller_for,
    validate_configuration,
)
from .description import describe
from .mechanism_test import report_lines, run_mechanism_test
from .mujoco_environment import declare_preview
from .native_exports import export_native
from .readiness import COMPATIBLE, require_robot_selection, robot_engine
from .scene import Environment, mechanism_audit, validate_environment
from .support import engine_statuses, limitations, matrix
from .terrain_test import report_lines as terrain_report_lines
from .terrain_test import run_terrain_test
from .uploads import (
    MAX_EXPANDED_BYTES,
    detect_format,
    package_members,
    resolve_reference,
    robot_inventory,
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def normalize_preview_transforms(path: Path) -> None:
    """Use explicit glTF TRS so native viewers and Nova read identical rest poses."""
    raw = path.read_bytes()
    length = struct.unpack_from("<I", raw, 12)[0]
    document = json.loads(raw[20 : 20 + length])
    for node in document.get("nodes", []):
        if "matrix" not in node:
            continue
        matrix = np.asarray(node.pop("matrix")).reshape(4, 4).T
        scale = np.linalg.norm(matrix[:3, :3], axis=0)
        if (
            not np.isfinite(matrix).all()
            or np.any(scale <= 0)
            or np.linalg.det(matrix[:3, :3]) <= 0
        ):
            raise ValueError("Preview transform must have a positive, finite scale")
        rotation = matrix.copy()
        rotation[:3, :3] /= scale
        quaternion = trimesh.transformations.quaternion_from_matrix(rotation)
        node.update(
            translation=matrix[:3, 3].tolist(),
            rotation=np.roll(quaternion, -1).tolist(),
            scale=scale.tolist(),
        )
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    remaining = raw[20 + length :]
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 20 + len(encoded) + len(remaining))
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
        + remaining
    )


def linear_rgb(color) -> np.ndarray:
    """An authored colour is sRGB, as picked from a photo or a swatch; glTF
    baseColorFactor and UsdPreviewSurface diffuseColor are linear. Writing sRGB
    numbers there rendered every scene about three times too light — black chairs
    grey, walnut beige, grey carpet white. MuJoCo shows rgba as authored."""
    rgb = np.clip(np.asarray(color, dtype=float)[:3], 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def validate_preview_materials(scene: Environment, path: Path) -> dict:
    """Check final assembled GLB, not merely the author's material declarations."""
    raw = path.read_bytes()
    length = struct.unpack_from("<I", raw, 12)[0]
    document = json.loads(raw[20 : 20 + length])
    nodes = {n.get("name"): n for n in document["nodes"]}
    checked = 0
    for obj in scene.objects:
        material = getattr(obj.mesh.visual, "material", None)
        if (
            getattr(material, "baseColorTexture", None) is not None
            or getattr(material, "image", None) is not None
        ):
            continue  # Texture appearance needs a separate rendered review.
        node = nodes.get(obj.name)
        if node is None or "mesh" not in node:
            raise ValueError(f"Preview lost environment object: {obj.name}")
        for primitive in document["meshes"][node["mesh"]]["primitives"]:
            if "material" not in primitive:
                raise ValueError(f"Preview lost material: {obj.name}")
            pbr = document["materials"][primitive["material"]].get("pbrMetallicRoughness", {})
            expected = np.round(np.append(linear_rgb(obj.color), obj.color[3]) * 255) / 255
            if not np.allclose(pbr.get("baseColorFactor", [1] * 4), expected, atol=1e-6):
                raise ValueError(f"Preview changed base color: {obj.name}")
            if not math.isclose(pbr.get("roughnessFactor", 1), obj.roughness, abs_tol=1e-6):
                raise ValueError(f"Preview changed roughness: {obj.name}")
        checked += 1
    return {
        "status": "passed",
        "solid_material_objects_checked": checked,
        "photorealism": "not_validated",
        "texture_appearance": "not_validated",
    }


def collect_robot_inputs(input_dir: Path, output: Path, index: list) -> list[dict]:
    if not isinstance(index, list) or len(index) > 8:
        raise ValueError("Invalid robot upload index")
    reports = []
    total = 0
    for entry in index:
        raw_path = input_dir / entry["path"]
        if raw_path.parent != input_dir or raw_path.is_symlink():
            raise ValueError("Robot input path escaped its upload directory")
        data = raw_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("Robot input hash does not match its upload")
        members = package_members(entry["name"], data)
        total += sum(map(len, members.values()))
        if total > MAX_EXPANDED_BYTES:
            raise ValueError("Combined robot assets exceed 128 MB expanded")
        folder = output / "robot" / entry["sha256"][:16]
        for name, content in members.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        models = [m for m in robot_inventory(entry["name"], data)["models"] if m["format"] != "usd"]
        for name, content in members.items():
            if detect_format(name, content) == "usd":
                # Open a layer, not a stage: never follow references/payloads or load plugins
                # requested by a customer's scene during dependency discovery.
                layer = Sdf.Layer.FindOrOpen(str(folder / name))
                if layer is None:
                    raise ValueError(f"Cannot parse robot USD: {name}")
                references = set(layer.GetExternalReferences())
                meshes = 0
                joints = 0
                articulation = False

                def visit(path, layer=layer, references=references):
                    nonlocal meshes, joints, articulation
                    obj = layer.GetObjectAtPath(path)
                    if isinstance(obj, Sdf.PrimSpec):
                        meshes += obj.typeName == "Mesh"
                        joints += "Joint" in obj.typeName
                        if obj.HasInfo("apiSchemas"):
                            articulation |= (
                                "PhysicsArticulationRootAPI"
                                in obj.GetInfo("apiSchemas").GetAppliedItems()
                            )
                    elif isinstance(obj, Sdf.AttributeSpec):
                        value = obj.default
                        if isinstance(value, Sdf.AssetPath):
                            references.add(value.path)
                        elif obj.typeName == Sdf.ValueTypeNames.AssetArray and value:
                            references.update(item.path for item in value)

                layer.Traverse(Sdf.Path.absoluteRootPath, visit)
                # OmniPBR is an Isaac-provided shader, not a customer texture.
                builtins = sorted(
                    ref for ref in references if ref in {"OmniPBR.mdl", "OmniGlass.mdl"}
                )
                missing = sorted(
                    ref
                    for ref in references
                    if ref not in builtins
                    and (
                        ref.startswith("package://")
                        or not resolve_reference(name, ref, set(members))
                    )
                )
                models.append(
                    {
                        "path": name,
                        "format": "usd",
                        "meshes": meshes,
                        "joints": joints,
                        "has_articulation": articulation,
                        "role": "robot" if articulation and joints else "environment",
                        "default_prim": layer.defaultPrim,
                        "missing_dependencies": missing,
                        "simulator_dependencies": builtins,
                        "meters_per_unit": layer.pseudoRoot.GetInfo("metersPerUnit"),
                        "up_axis": str(layer.pseudoRoot.GetInfo("upAxis") or ""),
                    }
                )
        for model in models:
            model["supported_targets"] = sorted(COMPATIBLE.get(model["format"], ()))
            if model["format"] in {"urdf", "mjcf"} and model.get("role") != "environment":
                from .byor.upload_check import inspect_robot

                model["physics"] = inspect_robot(
                    folder / model["path"], folder, output / ".byor-cache" / entry["sha256"][:16]
                )
        reports.append(
            {
                "id": entry.get("id"),
                "name": entry["name"],
                "sha256": entry["sha256"],
                "directory": folder.relative_to(output).as_posix(),
                "models": models,
            }
        )
    return reports


def _axis_rotation(axis) -> Gf.Quatf:
    """Quaternion taking the joint's +X axis onto the requested world axis."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x = np.array([1.0, 0.0, 0.0])
    cosine = float(np.dot(x, axis))
    if cosine > 1 - 1e-9:
        return Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0))
    if cosine < -1 + 1e-9:
        return Gf.Quatf(0.0, Gf.Vec3f(0, 0, 1))
    normal = np.cross(x, axis)
    sine = np.linalg.norm(normal)
    half = math.atan2(sine, cosine) / 2
    return Gf.Quatf(math.cos(half), Gf.Vec3f(*(normal / sine * math.sin(half))))


USD_MOTION_FPS = 24.0


def _usd_motion_times(stage, drive):
    """Sample times for one exact loop, and grow the stage's playback range."""
    duration = motion.loop_duration(motion.drive_dict(drive))
    frames = max(2, int(round(duration * USD_MOTION_FPS)) + 1)
    end = duration * USD_MOTION_FPS
    stage.SetTimeCodesPerSecond(USD_MOTION_FPS)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(max(float(stage.GetEndTimeCode() or 0), end))
    return [(index * duration / (frames - 1)) for index in range(frames)]


def _usd_bake_joint_drive(stage, drive_api, body):
    """A stiff position drive whose target follows the prescribed trajectory."""
    spec = motion.drive_dict(body.drive)
    scale = 180.0 / math.pi if body.motion == "hinge" else 1.0
    stiffness = max(1e4, 1e4 * float(body.mass))
    drive_api.CreateStiffnessAttr(stiffness / scale if body.motion == "hinge" else stiffness)
    drive_api.CreateDampingAttr(
        (stiffness / 10) / scale if body.motion == "hinge" else stiffness / 10
    )
    target = drive_api.CreateTargetPositionAttr()
    for t in _usd_motion_times(stage, body.drive):
        value = motion.joint_position(spec, t)
        target.Set(value * scale if body.motion == "hinge" else value, t * USD_MOTION_FPS)


def _usd_bake_patrol(stage, xform, body):
    """Time-sampled translate/orient on the kinematic body's Xform."""
    spec = motion.drive_dict(body.drive)
    ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
    translate = ops.get("xformOp:translate") or xform.AddTranslateOp()
    orient = xform.AddOrientOp()
    for t in _usd_motion_times(stage, body.drive):
        position, yaw = motion.patrol_pose(spec, t)
        w, x, y, z = motion.yaw_quaternion_wxyz(yaw)
        translate.Set(Gf.Vec3d(*map(float, position)), t * USD_MOTION_FPS)
        orient.Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))), t * USD_MOTION_FPS)


def _usd_mesh(stage, path, geometry, origin):
    """Mesh prim localized to `origin` (a movable body's rest anchor)."""
    vertices = (geometry.vertices - np.asarray(origin, dtype=float)).astype(np.float32)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr([3] * len(geometry.faces))
    mesh.CreateFaceVertexIndicesAttr(geometry.faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(geometry.vertex_normals.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr(
        [Gf.Vec3f(*map(float, v)) for v in (vertices.min(axis=0), vertices.max(axis=0))]
    )
    return mesh


def _usd_body(stage, xform, joint_path, body):
    """RigidBodyAPI + MassAPI on the object, and a passive revolute/prismatic joint to
    the world for hinges/sliders. Free bodies get no joint. A prescribed drive is
    baked natively: kinematic bodies get time-sampled transforms, driven joints get
    a stiff DriveAPI whose targetPosition is time-sampled over one motion loop."""
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim()).CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(xform.GetPrim())
    mass.CreateMassAttr(float(body.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, body.center_of_mass)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, body.diagonal_inertia)))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)))
    if body.motion == "kinematic":
        # The engine moves the body along its authored samples; contacts stay real.
        UsdPhysics.RigidBodyAPI(xform.GetPrim()).CreateKinematicEnabledAttr(True)
        _usd_bake_patrol(stage, xform, body)
        return None
    if body.motion == "free":
        return None
    rotation = _axis_rotation(body.axis)
    spins = body.drive is not None and body.drive.kind == "spin"
    if body.motion == "hinge":
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        if not spins:
            # UsdPhysics revolute limits and angular drive damping are per degree.
            joint.CreateLowerLimitAttr(math.degrees(body.limits[0]))
            joint.CreateUpperLimitAttr(math.degrees(body.limits[1]))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateDampingAttr(float(body.damping) * math.pi / 180)
    else:
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        joint.CreateLowerLimitAttr(float(body.limits[0]))
        joint.CreateUpperLimitAttr(float(body.limits[1]))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateDampingAttr(float(body.damping))
    drive.CreateTypeAttr("force")
    if body.drive is not None:
        _usd_bake_joint_drive(stage, drive, body)
    else:
        drive.CreateStiffnessAttr(0.0)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
    joint.CreateAxisAttr("X")
    # body0 unset = world. The body frame is world-aligned at rest, so both local
    # frames carry the same rotation; the joint sits at the body origin.
    joint.CreateBody1Rel().SetTargets([xform.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, body.origin)))
    joint.CreateLocalRot0Attr(rotation)
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(rotation)
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateJointEnabledAttr(True)
    # PhysX-specific dry joint friction; not part of UsdPhysics, best-effort mapping.
    joint.GetPrim().CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(
        float(body.frictionloss)
    )
    joint.GetPrim().SetCustomDataByKey("nova:frictionloss", float(body.frictionloss))
    return joint


def export_environment(scene: Environment, output: Path) -> tuple[dict, dict]:
    validate_environment(scene)
    contact = ensure_contact(scene)
    output.mkdir(parents=True, exist_ok=True)
    scene_path = output / "environment.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(scene.gravity_m_s2)
    preview = trimesh.Scene()
    descriptions = []
    joints = 0
    for index, obj in enumerate(scene.objects):
        identifier = f"object_{index:04d}"
        path = f"/World/Objects/{identifier}"
        origin = obj.body.origin if obj.body else (0, 0, 0)
        xform = UsdGeom.Xform.Define(stage, path)
        xform.GetPrim().SetDisplayName(obj.name)
        xform.GetPrim().SetCustomDataByKey("nova:semanticClass", obj.semantic_class)
        xform.GetPrim().SetCustomDataByKey("nova:objectName", obj.name)
        if obj.body is not None:
            xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, origin)))
            joints += (
                _usd_body(stage, xform, f"/World/Joints/{identifier}_joint", obj.body) is not None
            )
        mesh = _usd_mesh(stage, f"{path}/visual", obj.mesh, origin)
        mesh.GetPrim().SetCustomDataByKey("nova:objectName", obj.name)
        material = UsdShade.Material.Define(stage, f"/World/Materials/{identifier}")
        shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*(float(v) for v in linear_rgb(obj.color)))
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(obj.roughness)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(obj.color[3])
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        visual = obj.mesh.visual
        image = getattr(getattr(visual, "material", None), "baseColorTexture", None)
        if image is None:
            image = getattr(getattr(visual, "material", None), "image", None)
        uv = getattr(visual, "uv", None)
        if image is not None and uv is not None:
            if np.asarray(uv).shape != (len(obj.mesh.vertices), 2) or not np.isfinite(uv).all():
                raise ValueError(f"{obj.name}: invalid UV coordinates")
            texture_dir = output / "textures"
            texture_dir.mkdir(exist_ok=True)
            texture_path = texture_dir / f"{identifier}.png"
            image.save(texture_path)
            coordinates = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            coordinates.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype=np.float32)))
            reader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/UV")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            texture = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Texture")
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(texture_path.relative_to(output).as_posix())
            )
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.ConnectableAPI(), "result"
            )
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.GetInput("diffuseColor").ConnectToSource(texture.ConnectableAPI(), "rgb")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        if obj.collidable:
            physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            physical.CreateStaticFrictionAttr(obj.friction)
            physical.CreateDynamicFrictionAttr(obj.friction)
            physical.CreateRestitutionAttr(obj.restitution)
            pieces = obj.collision_meshes
            if pieces is None and obj.body is not None and obj.auto_collision:
                pieces = obj.auto_collision  # a moving concave part must not be hulled
            if pieces is None:
                colliders = [mesh]
                # A moving body cannot collide as a raw triangle soup; static ones can.
                approximation = "convexHull" if obj.body is not None else "none"
            else:
                colliders = []
                approximation = "convexHull"
                for part, piece in enumerate(pieces):
                    collider = _usd_mesh(stage, f"{path}/collision_{part:02d}", piece, origin)
                    collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
                    colliders.append(collider)
            for collider in colliders:
                prim = collider.GetPrim()
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        preview_mesh = obj.mesh.copy()
        if image is None or uv is None:
            # ColorVisuals interprets floating-point channels as normalized 0..1.
            # Multiplying floats by 255 clipped every non-dark channel to white.
            # Use an explicit glTF material and retain the authored roughness too.
            preview_mesh.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=np.round(
                        np.append(linear_rgb(obj.color), obj.color[3]) * 255
                    ).astype(np.uint8),
                    roughnessFactor=obj.roughness,
                    metallicFactor=0.0,
                    alphaMode="BLEND" if obj.color[3] < 1 else "OPAQUE",
                ),
            )
        # Meshes are authored in world coordinates. A movable preview node must
        # instead sit at its physical origin, or a slider/hinge rotates the room.
        origin = np.asarray(obj.body.origin) if obj.body else np.zeros(3)
        preview_mesh.apply_translation(-origin)
        preview_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])
        )
        transform = np.eye(4)
        transform[:3, 3] = [origin[0], origin[2], -origin[1]]
        metadata = {}
        if obj.body and obj.body.drive is not None:
            metadata["drive"] = motion.drive_dict(obj.body.drive)
        if obj.body and obj.body.motion in {"hinge", "slide"} and obj.body.limits is not None:
            axis = np.asarray(obj.body.axis)
            converted = np.array([axis[0], axis[2], -axis[1]])
            letters = ["X", "Y", "Z"]
            cardinal = next((i for i in range(3) if np.isclose(abs(converted[i]), 1)), None)
            label = (
                f"local {'-' if converted[cardinal] < 0 else ''}{letters[cardinal]}"
                if cardinal is not None
                else converted.tolist()
            )
            unit = "deg" if obj.body.motion == "hinge" else "m"
            lo, hi = obj.body.limits
            if unit == "deg":
                lo, hi = math.degrees(lo), math.degrees(hi)
            metadata = {
                "joint_type": obj.body.motion,
                "joint_axis": label,
                "joint_axis_space": "local",
                f"joint_min_{unit}": lo,
                f"joint_max_{unit}": hi,
                f"joint_rest_{unit}": 0,
                "joint_note": "Preview posing is kinematic; use Run for physics.",
            }
        preview.add_geometry(
            preview_mesh,
            node_name=obj.name,
            geom_name=identifier,
            transform=transform,
            metadata=metadata,
        )
        descriptions.append(
            {
                "id": identifier,
                "name": obj.name,
                "semantic_class": obj.semantic_class,
                "bounds_m": obj.mesh.bounds.tolist(),
                "static": obj.body is None,
                "body": asdict(obj.body) if obj.body else None,
                "terrain": obj.heightfield.statistics() if obj.heightfield is not None else None,
                "collision": (
                    "heightfield_triangle_mesh"
                    if obj.heightfield is not None
                    else "convex_pieces"
                    if obj.collision_meshes
                    else "auto_convex_pieces"
                    if obj.body is not None and obj.auto_collision
                    else "convex_hull"
                    if obj.body is not None
                    else "triangle_mesh"
                )
                if obj.collidable
                else "none",
                "mujoco_collision": (
                    "heightfield"
                    if obj.heightfield is not None
                    else "convex_pieces"
                    if obj.collision_meshes
                    else "auto_convex_pieces"
                    if obj.auto_collision
                    else "convex_hull"
                )
                if obj.collidable
                else "none",
                "collision_piece_count": len(
                    obj.collision_meshes or obj.auto_collision or (obj.mesh,)
                )
                if obj.collidable
                else 0,
                "torsional_friction": obj.torsional_friction,
                "rolling_friction": obj.rolling_friction,
                "friction": obj.friction,
                "restitution": obj.restitution,
                "static_friction": obj.friction,
                "dynamic_friction": obj.friction,
                "physics_source": "generator_configuration",
            }
        )
    for name, position in (("Spawn", scene.spawn_position), ("Goal", scene.goal_position)):
        marker = UsdGeom.Xform.Define(stage, f"/World/{name}")
        marker.AddTranslateOp().Set(Gf.Vec3d(*position))
    for number, volume in enumerate(scene.water_volumes):
        # The sheet itself is an ordinary non-colliding mesh; this prim records where
        # the water is, so a runtime hook can find it without parsing the scene JSON.
        x0, y0, x1, y1 = volume.bounds
        region = UsdGeom.Xform.Define(stage, f"/World/Water/region_{number:02d}")
        region.AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2, (y0 + y1) / 2, volume.surface_z))
        prim = region.GetPrim()
        prim.SetDisplayName(volume.name)
        for key, value in volume.specification().items():
            prim.SetCustomDataByKey(
                f"nova:water:{key}",
                Vt.DoubleArray([float(v) for v in value]) if isinstance(value, list) else value,
            )
    stage.GetRootLayer().Save()
    # The browser consumes glTF's Y-up convention; USD stays in the authored Z-up frame.
    (output.parent / "model.glb").write_bytes(preview.export(file_type="glb"))
    scenario = {
        "schema_version": "nova.robotics-scene/v1",
        "units": "metres",
        "up_axis": "Z",
        "description": scene.description,
        "scene": "environment.usda",
        "spawn_position": scene.spawn_position,
        "goal_position": scene.goal_position,
        "objects": descriptions,
        "physics": {
            "gravity_m_s2": scene.gravity_m_s2,
            "gravity_direction": [0, 0, -1],
            "bodies": "mixed_rigid" if any(o.body for o in scene.objects) else "static",
            "collision": "convex_pieces"
            if any(o.split_contact for o in scene.objects)
            else "triangle_mesh",
            "mechanisms": "passive world-attached hinges/sliders and free bodies on every "
            "target; no latches, closers, springs or driven joints",
            "contact_geometry": {
                "policy": "explicit pieces, else automatic convex decomposition of every "
                "concave part (MuJoCo: all objects; other targets: movable bodies), else "
                "one hull for convex parts; static objects keep exact triangle meshes "
                "in USD/SDF/PyBullet",
                "objects_resolved": contact["resolved"],
                "objects_decomposed": contact["decomposed_objects"],
                "auto_pieces": contact["pieces"],
            },
            "real_world_calibrated": False,
            "units": "metres",
            "up_axis": "Z",
            "runtime_checks": {},
        },
    }
    # Reopen the bytes we produced; verification is structural, never simulated motion.
    reopened = Usd.Stage.Open(str(scene_path))
    if reopened is None or not reopened.GetDefaultPrim():
        raise ValueError("Exported USD could not be reopened")
    mesh_prims = [p for p in reopened.Traverse() if p.IsA(UsdGeom.Mesh) and p.GetName() == "visual"]
    if len(mesh_prims) != len(scene.objects):
        raise ValueError("Exported USD lost mesh objects")
    rigid = [p for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joint_prims = [p for p in reopened.Traverse() if p.IsA(UsdPhysics.Joint)]
    movable = sum(o.body is not None for o in scene.objects)
    if len(rigid) != movable or len(joint_prims) != joints:
        raise ValueError("Exported USD lost rigid bodies or joints")
    audit = mechanism_audit(scene)
    report = {
        "schema_version": "nova.robotics-validation/v1",
        "status": "passed",
        "scope": "environment_export",
        "checks": {
            "finite_geometry": "passed",
            "usd_reopens": "passed",
            "mesh_count_matches": "passed",
            "units_and_axes": "passed",
            "movable_bodies_exported": "passed" if movable else "not_applicable",
            "mechanism_audit": "failed" if audit["errors"] else "passed",
            "contact_geometry_resolved": "inconclusive"
            if any(
                d.get("decomposition_failures") or d.get("dropped_small_pieces")
                for d in contact["objects"].values()
            )
            else "passed",
            "isaac_sim_runtime": "not_run",
            "robot_control": "not_run",
        },
        "mesh_count": len(mesh_prims),
        "collider_count": sum(o.collidable for o in scene.objects),
        "rigid_body_count": len(rigid),
        "joint_count": len(joint_prims),
        "mechanisms": audit,
        "contact_geometry": {
            "objects_decomposed": contact["decomposed_objects"],
            "auto_pieces": contact["pieces"],
            "details": contact["objects"],
        },
    }
    return scenario, report


def build(source: str = "asset.py", output_dir: str = "out", *, scene, context: dict) -> None:
    source_path = Path(source)
    if source_path.is_absolute() or ".." in source_path.parts:
        raise ValueError("Model source must be workspace-relative")
    config = validate_configuration(context.get("settings"))
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    output = Path(output_dir) / "simulation"
    output.mkdir(parents=True, exist_ok=True)
    robot_reports = (
        collect_robot_inputs(Path(context["input_dir"]), output, context.get("uploads", []))
        if config["mode"] == "robot_environment"
        else []
    )
    model = require_robot_selection(
        config,
        [
            {"id": u["id"], "state": "preserved", "findings": {"models": u["models"]}}
            for u in robot_reports
        ],
    )
    upload = next((u for u in robot_reports if u["id"] == config["robot_upload_id"]), None)
    if upload and any(m.get("missing_dependencies") for m in upload["models"]):
        raise ValueError(
            "Selected robot package has unresolved dependencies; include all referenced assets"
        )
    if model is not None and model.get("role") != "robot":
        raise ValueError(
            "The selected USD is an environment or unconfirmed asset, not an articulated robot"
        )
    validate_environment(scene)
    # Paint and labels must not turn into artificial bumps under a robot's feet.
    for obj in scene.objects:
        if obj.semantic_class in {"floor_marking", "stair_marking", "label"}:
            obj.collidable = False
    constraints, applied = apply_configuration(scene, config)
    audit = mechanism_audit(scene)
    # These are authoring findings. Successful export remains reviewable under Nova's
    # advisory acceptance policy; never disguise a static door as a working mechanism.
    audit["warnings"].extend(audit["errors"])
    scenario, report = export_environment(scene, output)
    scenario["schema_version"] = "nova.robotics-scene/v2"
    scenario["preview_frame"] = "body_origin_y_up/v1"
    scenario["configuration"] = constraints
    scenario["mechanisms"] = audit
    exports = list(config["targets"])
    scenario["exports"] = export_native(scene, output, exports)
    mechanism_result = run_mechanism_test(scene, output / "environment.xml", output)
    scenario["mechanism_test"] = mechanism_result
    report["mechanism_test"] = mechanism_result
    report["checks"]["mechanism_push_test"] = (
        mechanism_result["status"] if mechanism_result["mechanisms"]
        or mechanism_result["status"] != "passed" else "not_applicable"
    )
    # Terrain is measured the same way a mechanism is: probed, judged by the
    # acceptance engine. A hill that collides as a hull fails here, not in the field.
    terrain_result = run_terrain_test(scene, output / "environment.xml", output)
    scenario["terrain_test"] = terrain_result
    report["terrain_test"] = terrain_result
    report["checks"]["terrain_contact_test"] = terrain_result["status"]
    # A photographed place: every copy the photos show is built, every door works.
    from . import twin_check

    twin_result = twin_check.check(scene, mechanism_result, twin_check.load())
    scenario["twin_inventory"] = twin_result
    report["twin_inventory"] = twin_result
    report["checks"]["twin_inventory"] = twin_result["status"]
    write_json(output / "build_configuration.json", constraints)
    write_json(output / "robot_inputs.json", robot_reports)
    robot_path = output / upload["directory"] / model["path"] if upload and model else None
    native = robot_engine(model)
    controller, controller_note = controller_for(config["controller"], model)
    report["controller"] = {
        "requested": config["controller"], "used": controller, "note": controller_note,
    }
    config = {**config, "controller": controller}
    from .assembly import assemble_mujoco, assemble_usd
    from .environment_only import assemble_environment
    from .sdf_assembly import assemble_sdf
    from .urdf_preview import assemble_pybullet

    entrypoints = {}
    simulations = {}
    for target in exports:
        if robot_path is None or target != native:
            reason = (
                None if robot_path is None
                else f"The uploaded robot is native to {TARGETS[native]}, not {TARGETS[target]}"
            )
            entry, simulation = assemble_environment(output, scene, target, config, reason)
        elif target == "mujoco":
            entry, simulation = assemble_mujoco(output, robot_path, scene, config["controller"])
        elif target == "isaac_sim":
            entry, simulation = assemble_usd(output, robot_path, scene, model)
        elif target == "gazebo":
            entry, simulation = assemble_sdf(output, robot_path, scene)
        else:
            entry, simulation = assemble_pybullet(output, robot_path, scene, config)
        entrypoints[target] = entry
        simulations[target] = simulation
        scenario["exports"].setdefault(target, {}).update(
            entrypoint=entry, robot_included=target == native, runtime_check=simulation
        )
    scenario["mode"] = config["mode"]
    scenario["simulations"] = simulations
    scenario["simulation"] = simulations[ACCEPTANCE_ENGINE]
    scenario["robot"] = (
        {
            "name": model.get("name") or model["path"],
            "format": model["format"],
            "engine": native,
            "path": robot_path.relative_to(output).as_posix(),
            "preview": "robot + environment; source rest pose",
            "spawn_position": scene.spawn_position,
        }
        if model
        else None
    )
    report["robot_included"] = model is not None
    report["simulations"] = simulations
    report["simulation"] = scenario["simulation"]
    report["checks"]["robot_control"] = (
        simulations[ACCEPTANCE_ENGINE].get("status", "not_run")
        if config["controller"] and native == ACCEPTANCE_ENGINE
        else "not_run"
    )
    normalize_preview_transforms(output.parent / "model.glb")
    report["appearance"] = validate_preview_materials(scene, output.parent / "model.glb")
    # Physical source lines: the doctrine's geometry-depth requirement (>2,000 for a
    # detailed generation). Reported, never enforced: a small brief is allowed to be small.
    source_lines = sum(
        1 for line in source_path.read_text(errors="replace").splitlines() if line.strip()
    )
    report["source_lines"] = source_lines
    scenario["source_lines"] = source_lines
    report["checks"].update(
        final_glb_solid_materials="passed",
        route_usability="not_run",
        mechanism_task_success="not_run",
        visual_quality="not_run",
    )
    scenario["appearance"] = report["appearance"]
    # The world description and the support matrix are the record every other
    # consumer reads; the README and the limitations are generated from them.
    scenario["world"] = describe(scene, applied)
    tested = {
        ACCEPTANCE_ENGINE: "acceptance",
        COMPATIBILITY_ENGINE: "compatibility"
        if mechanism_result["coverage"].get(COMPATIBILITY_ENGINE, {}).get("status") == "measured"
        else None,
    }
    support = matrix(scene, exports, tested, config["controller"])
    scenario["support"] = support
    scenario["engines"] = engine_statuses(support, native, model is not None)
    scenario["notes"] = list(scene.notes)
    report["support"] = support
    report["engines"] = scenario["engines"]
    report["limitations"] = limitations(support)
    report["authoring_notes"] = list(scene.notes)
    statuses = list(report["checks"].values())
    report["status"] = (
        "failed"
        if "failed" in statuses
        else "inconclusive"
        if "inconclusive" in statuses or "error" in statuses
        else "passed"
    )
    write_json(output / "scene_spec.json", scenario)
    write_json(output / "validation.json", report)
    (output / "README.md").write_text(
        readme_text(config, entrypoints, scenario["engines"], report["limitations"], audit,
                    mechanism_result, scenario["robot"])
    )
    manifest = {
        "schema_version": "nova.robotics-package/v1",
        "targets": exports,
        "acceptance_engine": ACCEPTANCE_ENGINE,
        # A successful passive smoke test is evidence about that check only.
        "runtime_checks": {target: {
            key: result[key] for key in ("status", "mode", "scope", "duration_s", "engine")
            if key in result
        } for target, result in simulations.items()},
        "engines": scenario["engines"],
        "generator": "robotics_v1@1.7.3",
        "openusd_version": ".".join(map(str, Usd.GetVersion())),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "entrypoint": entrypoints[ACCEPTANCE_ENGINE],
        "entrypoints": entrypoints,
        "robot_included": model is not None,
        "files": [],
    }

    # USD is used internally to check geometry/preview, but is not shipped for
    # engines that were filtered out of this build's exports.
    def included(path):
        relative = path.relative_to(output).as_posix()
        return (
            relative != "manifest.json"
            and not (
                "isaac_sim" not in exports
                and (relative == "environment.usda" or relative.startswith("textures/"))
            )
            and not ("mujoco" not in exports and relative == "environment.xml")
            and not ("pybullet" not in exports and relative.endswith(".urdf"))
        )

    for path in sorted(output.rglob("*")):
        if path.is_file() and included(path):
            manifest["files"].append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    # The browser viewer's contract: the MuJoCo model it compiles (the robot assembly
    # when one was attached to MuJoCo, else the environment; never a launcher script)
    # and the exact files that reads, proven loadable from those files alone. Every
    # declared file must also be a hashed package file, so a viewer can verify it.
    preview_model = (
        "simulation.xml"
        if model is not None and native == ACCEPTANCE_ENGINE
        else "environment.xml"
    )
    preview = declare_preview(output, preview_model)
    packaged = {item["path"] for item in manifest["files"]}
    if preview["status"] == "ready" and not set(preview["files"]) <= packaged:
        preview.update(
            status="unavailable", files=[],
            reason="A file the MuJoCo model reads is not part of the package.",
        )
    manifest["preview"] = preview
    write_json(output / "manifest.json", manifest)
    with zipfile.ZipFile(Path(output_dir) / "simulation.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and (included(path) or path.name == "manifest.json"):
                entry = zipfile.ZipInfo(path.relative_to(output).as_posix())
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, path.read_bytes())
    # The ZIP is the deliverable and the only thing the live session, the viewer's
    # physics and the inspector open. The unpacked folder keeps what a reader needs —
    # native scene files, reports, README — and drops the mesh and asset trees, which
    # would otherwise flood the workspace checkpoint with one file per part.
    for folder in ("meshes", "terrain", "robot", "robot_runtime_assets", "textures", "controller"):
        shutil.rmtree(output / folder, ignore_errors=True)
    print(f"ROBOTICS_EXPORT_OK meshes={report['mesh_count']} colliders={report['collider_count']}")
    corners = [obj.mesh.bounds for obj in scene.objects if len(obj.mesh.vertices)]
    if corners:
        lo = np.min([c[0] for c in corners], axis=0)
        hi = np.max([c[1] for c in corners], axis=0)
        size = hi - lo
        print(f"SITE_EXTENT x={size[0]:.1f}m y={size[1]:.1f}m z={size[2]:.1f}m "
              "(built bounds; compare with the plan's stated footprint)")
    print(f"SOURCE_LINES {source_lines} (doctrine depth requirement: more than 2000)")
    contact = report["contact_geometry"]
    print(
        f"CONTACT_GEOMETRY auto_decomposed_objects={contact['objects_decomposed']} "
        f"auto_pieces={contact['auto_pieces']} (see per-object contact_geometry for "
        "decomposition errors, dropped pieces and hull approximations)"
    )
    print(
        f"MECHANISMS hinges={audit['hinges']} sliders={audit['sliders']} "
        f"free_bodies={audit['free_bodies']} static={audit['static_objects']} "
        f"warnings={len(audit['warnings'])}"
    )
    for warning in audit["warnings"]:
        print(f"MECHANISM_WARNING {warning}")
    for line in report_lines(mechanism_result):
        print(line)
    for line in terrain_report_lines(terrain_result):
        print(line)
    for line in twin_result["lines"]:
        print(line)
    for note in scene.notes:
        print(f"SCENE_NOTE {note}")
    for entry in scenario["engines"]:
        print(f"ENGINE {entry['engine']}: {entry['status']}")
    if preview["status"] == "ready":
        print(f"PHYSICS_PREVIEW ready model={preview_model} files={len(preview['files'])}")
    else:
        print(f"PHYSICS_PREVIEW unavailable model={preview_model}: {preview['reason']}")
    if config["controller"]:
        print(f"MUJOCO_CONTROLLER {config['controller']}; training not run; see scene_spec.json")
    for upload in robot_reports:
        for model in upload["models"]:
            print(
                f"ROBOT_INPUT {model['path']} missing={len(model.get('missing_dependencies', []))}"
            )


def readme_text(config, entrypoints, engines, limits, audit, mechanism_result, robot):
    """The package README, generated from the same records the reports carry."""
    status_words = {
        "tested_here": "built and motion-tested here",
        "checked_here": "loaded and motion-checked here as a second opinion",
        "exported_only": "files exported; not run here",
        "not_exported": "no files in this build",
    }
    lines = [
        "# Nova robotics package",
        "",
        "Extract the entire ZIP and keep its directory together: native scene files reference "
        "meshes by relative path. Open only the entrypoint for your simulator.",
        "",
        "| Simulator | Entrypoint | Status in this build |",
        "|---|---|---|",
    ]
    for entry in engines:
        entrypoint = entrypoints.get(entry["engine"], "—")
        robot_note = " (with the uploaded robot)" if entry["robot_included"] else ""
        lines.append(
            f"| {entry['label']} | {entrypoint} | "
            f"{status_words.get(entry['status'], entry['status'])}{robot_note} |"
        )
    lines += [
        "",
        "Isaac Sim: open the .usda. MuJoCo: open the .xml with its viewer. Gazebo: load the "
        ".sdf — with moving obstacles run `python3 run_gazebo.py` and open the viewer with "
        "`gz sim -g`. "
        "MuJoCo with moving obstacles: run `python run_mujoco.py` (a bare viewer shows the "
        "world static). PyBullet: install pybullet, then run `python run_pybullet.py` "
        "(`--headless` for "
        "a bounded passive test).",
        "",
        f"Movable bodies: {audit['hinges']} hinge(s), {audit['sliders']} slider(s), "
        f"{audit['free_bodies']} free prop(s); {audit['static_objects']} static objects. "
        "Hinges and sliders are passive joints to the world with limits, damping and friction; "
        "free props fall, slide and can be picked up. No latches, closers or driven joints.",
        "",
        "Environment mechanisms were settled and pushed in "
        + (", ".join(mechanism_result["engines"]) or "no engine")
        + f" (status {mechanism_result['status']}). This is not a robot task, a training run "
        "or proof of behaviour in another engine.",
    ]
    if config["mode"] == "environment":
        lines += ["", "No robot is included. Robot clearance, controllers and training are "
                  "not validated."]
    else:
        lines += ["", f"Robot: {robot['name']} ({robot['format']}), native to "
                  f"{TARGETS[robot['engine']]}; its source dynamics are preserved, never "
                  "invented. Other simulators receive the environment without it."]
    if config["controller"] == "go2_walk_v1":
        lines += [
            "",
            "## Go2 walking demonstration",
            "",
            "    python -m pip install -r requirements-controller.txt",
            "    python run_controller.py",
            "",
            "For a bounded CPU test without a desktop: `python run_controller.py --headless`. "
            "Opening simulation.xml alone DOES NOT run the walking controller. The controller "
            "is a pinned community Go2 ONNX policy (see controller/PROVENANCE.md and "
            "controller/LICENSE); it supplies bounded joint PD torques.",
        ]
    elif config["controller"] == "household_mission_v1":
        lines += [
            "",
            "## Scripted household mission",
            "",
            "    python -m pip install -r requirements-mission.txt",
            "    python run_mission.py",
            "",
            "Opening simulation.xml alone DOES NOT run the mission. The base is carried and "
            "the joints held; doors, drawers and props respond with real contacts. It is a "
            "demonstration of the environment, never a claim the robot can walk or grasp.",
        ]
    lines += ["", "## Limitations", ""]
    lines += [f"- {sentence}" for sentence in limits] or ["- None recorded for this scene."]
    lines += [
        "",
        "## File map",
        "",
        "In the Nova project, asset.py is the editable geometry source, out/model.glb is "
        "the browser preview, and out/simulation.zip is this entire directory as one download. "
        "Meshes, terrain grids and robot assets live only inside the ZIP. The preview uses Y-up; "
        "native scenes use metres and Z-up.",
        "",
        "| Path in this package | Purpose |",
        "|---|---|",
    ]
    lines += [f"| {entry} | Open this in {target}. |" for target, entry in entrypoints.items()]
    lines += [
        "| meshes/, terrain/ | Visual meshes, collision parts and elevation grids the native "
        "scenes reference. |",
        "| scene_spec.json | The world description: named objects, physical properties with "
        "their sources, joints, terrain, water, support matrix. |",
        "| validation.json | Measured results, findings, limitations and checks that did not "
        "run. |",
        "| build_configuration.json | Exact settings and overrides used for this build. |",
        "| robot_inputs.json, robot/ | Uploaded robot inventory and files, when included. |",
        "| nova_water.py, water_volumes.json | Buoyancy and drag hook for water regions, when "
        "declared. |",
        "| manifest.json | Selected entrypoints, engine statuses and hashes of the packaged "
        "files. |",
        "",
        "Native files can be edited in your simulator; rebuilding from asset.py produces a "
        "new export and does not merge those external edits.",
        "",
    ]
    return "\n".join(lines)
