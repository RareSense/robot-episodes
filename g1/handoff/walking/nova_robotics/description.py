"""The world description: one engine-free record of what the scene is.

Everything downstream reads this record — the exporters' summaries, the Studio's
object panel, the live session, the support matrix — instead of asking each exporter
what it happened to write. Every overridable value carries where it came from: the
authored source, or a creator override applied on top of it. Geometry stays in the
package files; the description points at it and states its size.
"""

from dataclasses import asdict

from .overlay import SCENE_PROPERTIES, property_catalog

SCHEMA = "nova.world-description/v1"


def _collision_mode(obj):
    if not obj.collidable:
        return "none"
    if obj.heightfield is not None:
        return "heightfield"
    if obj.collision_meshes:
        return "convex_pieces"
    if obj.auto_collision:
        return "auto_convex_pieces"
    return "convex_hull" if obj.body is not None else "triangle_mesh"


def _contact_shape(obj):
    """(volume, half height) of what the part collides as: the sum of its convex
    pieces, else its hull. The browser preview's water uses these same numbers, so a
    bucket floats the same on screen as in the build report."""
    pieces = obj.collision_meshes or obj.auto_collision or (obj.mesh.convex_hull,)
    volume = sum(float(abs(piece.volume)) for piece in pieces if piece.is_volume)
    extent = obj.mesh.bounds[1][2] - obj.mesh.bounds[0][2]
    return round(volume, 6), round(float(extent) / 2.0, 6)


def describe_object(index, obj, applied):
    """One object: identity, geometry summary, body, terrain and every property with
    its source. `applied` is the override map returned by apply_overlay."""
    overridden = applied.get(obj.name, {})
    properties = {}
    for key, entry in property_catalog(obj).items():
        properties[key] = {**entry, "source": "override" if key in overridden else "source"}
    volume, half_height = _contact_shape(obj) if obj.collidable else (0.0, 0.0)
    return {
        "id": f"object_{index:04d}",
        "name": obj.name,
        "semantic_class": obj.semantic_class,
        "static": obj.body is None,
        "terrain": obj.heightfield.statistics() if obj.heightfield is not None else None,
        "geometry": {
            "triangles": int(len(obj.mesh.faces)),
            "bounds_m": obj.mesh.bounds.tolist(),
            "contact_volume_m3": volume,
            "half_height_m": half_height,
        },
        "appearance": {"color": list(obj.color), "roughness": obj.roughness},
        "contact": {
            "collidable": obj.collidable,
            "mode": _collision_mode(obj),
            "pieces": (
                len(obj.collision_meshes or obj.auto_collision or (obj.mesh,))
                if obj.collidable and obj.heightfield is None
                else 0
            ),
        },
        "body": asdict(obj.body) if obj.body else None,
        "properties": properties,
    }


def describe(scene, applied=None):
    applied = applied or {}
    return {
        "schema_version": SCHEMA,
        "units": "metres",
        "up_axis": "Z",
        "description": scene.description,
        "spawn_position": list(scene.spawn_position),
        "goal_position": list(scene.goal_position),
        "scene_properties": {
            "gravity_m_s2": {
                "value": float(scene.gravity_m_s2),
                "source": (
                    "override" if "gravity_m_s2" in applied.get("__scene__", {}) else "source"
                ),
                **{
                    key: getattr(SCENE_PROPERTIES["gravity_m_s2"], key)
                    for key in ("unit", "label", "minimum", "maximum", "live")
                },
            }
        },
        "objects": [describe_object(i, obj, applied) for i, obj in enumerate(scene.objects)],
        "fields": {
            "terrain": [
                {"name": obj.name, **obj.heightfield.statistics()}
                for obj in scene.objects
                if obj.heightfield is not None
            ],
            "water": [volume.specification() for volume in scene.water_volumes],
        },
        "notes": list(scene.notes),
        "overrides_applied": applied,
    }
