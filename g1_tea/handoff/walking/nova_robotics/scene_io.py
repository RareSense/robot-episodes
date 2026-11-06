"""A private, bounded handoff from generated source to the reviewed exporter.

Only geometry and physical declarations cross this boundary. Never pickle a generated
Python object: loading it in the exporter would give candidate code another execution.
This format is a compiler intermediate, not a public Nova project contract.
"""

import base64
import io
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from .scene import Environment, PrescribedMotion, RigidBody, validate_environment
from .terrain import Grid, HeightField, WaterVolume


def _mesh(mesh):
    result = {"vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()}
    visual = mesh.visual
    material = getattr(visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if texture is None:
        texture = getattr(material, "image", None)
    if texture is not None and getattr(visual, "uv", None) is not None:
        if texture.width * texture.height > 4_194_304:
            raise ValueError("A scene texture exceeds 4 megapixels")
        stream = io.BytesIO()
        texture.save(stream, format="PNG")
        result["texture"] = base64.b64encode(stream.getvalue()).decode("ascii")
        result["uv"] = np.asarray(visual.uv).tolist()
    return result


def _read_mesh(value):
    vertices, faces = np.asarray(value["vertices"]), np.asarray(value["faces"])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Invalid scene vertices")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not np.issubdtype(faces.dtype, np.integer)
        or faces.min(initial=0) < 0
        or faces.max(initial=0) >= len(vertices)
    ):
        raise ValueError("Invalid scene triangle indices")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if value.get("texture"):
        data = base64.b64decode(value["texture"], validate=True)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError("Scene texture is too large")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.width * image.height > 4_194_304:
                raise ValueError("Invalid scene texture")
            image.load()
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=np.asarray(value["uv"]), image=image.copy()
            )
    return mesh


def _heightfield(patch):
    """The elevation grid itself, never its mesh: the exporter rebuilds the exact
    surface from these nodes, so the grid the probe test judges is the grid that
    collides and the grid the creator sees."""
    return {
        "name": patch.name,
        "grid": {"size": patch.grid.size, "cell": patch.grid.cell, "center": patch.grid.center},
        "heights": patch.heights.tolist(),
        "base_z": patch.base_z,
        "skirt": patch.skirt,
    }


def _read_heightfield(value):
    grid = value["grid"]
    heights = np.asarray(value["heights"], dtype=float)
    if heights.ndim != 2 or not np.isfinite(heights).all():
        raise ValueError("Invalid terrain heights")
    return HeightField(
        name=str(value["name"]),
        grid=Grid(size=tuple(grid["size"]), cell=float(grid["cell"]), center=tuple(grid["center"])),
        heights=heights,
        base_z=float(value["base_z"]),
        skirt=float(value["skirt"]),
    )


def write_scene(scene, path):
    validate_environment(scene)
    result = {
        "schema": 2,
        "description": scene.description,
        "gravity_m_s2": scene.gravity_m_s2,
        "spawn_position": scene.spawn_position,
        "goal_position": scene.goal_position,
        "objects": [],
        "water_volumes": [asdict(volume) for volume in scene.water_volumes],
        "notes": list(scene.notes),
    }
    for obj in scene.objects:
        result["objects"].append(
            {
                "name": obj.name,
                "mesh": None if obj.heightfield is not None else _mesh(obj.mesh),
                "heightfield": _heightfield(obj.heightfield) if obj.heightfield else None,
                "body": asdict(obj.body) if obj.body else None,
                "collision_meshes": (
                    [_mesh(mesh) for mesh in obj.collision_meshes]
                    if obj.collision_meshes is not None
                    else None
                ),
                **{
                    key: getattr(obj, key)
                    for key in (
                        "color",
                        "friction",
                        "restitution",
                        "collidable",
                        "semantic_class",
                        "roughness",
                        "torsional_friction",
                        "rolling_friction",
                    )
                },
            }
        )
    Path(path).write_text(json.dumps(result, allow_nan=False, separators=(",", ":")))


def read_scene(path):
    path = Path(path)
    if path.is_symlink() or not 0 < path.stat().st_size <= 256 * 1024 * 1024:
        raise ValueError("Scene intermediate exceeds its 256 MB limit")
    value = json.loads(path.read_bytes())
    if value.get("schema") != 2 or not 1 <= len(value.get("objects", [])) <= 2000:
        raise ValueError("Invalid scene intermediate")
    if len(value.get("water_volumes", [])) > 32 or len(value.get("notes", [])) > 64:
        raise ValueError("Invalid scene intermediate")
    scene = Environment(
        **{
            key: value[key]
            for key in (
                "description",
                "gravity_m_s2",
                "spawn_position",
                "goal_position",
            )
        }
    )
    for item in value["objects"]:
        record = dict(item)
        patch = record.pop("heightfield", None)
        raw_mesh = record.pop("mesh")
        if patch is not None:
            patch = _read_heightfield(patch)
            mesh = patch.surface_mesh()  # the grid is the authority, never a shipped mesh
        else:
            mesh = _read_mesh(raw_mesh)
        body = record.pop("body")
        if body and body.get("drive"):
            body = {**body, "drive": PrescribedMotion(**body["drive"])}
        pieces = record.pop("collision_meshes")
        scene.add_mesh(
            mesh=mesh,
            body=RigidBody(**body) if body else None,
            collision_meshes=[_read_mesh(piece) for piece in pieces]
            if pieces is not None
            else None,
            **record,
        )
        scene.objects[-1].heightfield = patch
    for volume in value.get("water_volumes", []):
        scene.water_volumes.append(WaterVolume(**volume))
    scene.notes = [str(note) for note in value.get("notes", [])]
    validate_environment(scene)
    return scene
