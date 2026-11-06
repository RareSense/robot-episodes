"""Creator-owned property overrides applied on top of the authored scene.

A typed edit in Studio — "this door weighs 20 kg", "the floor is icy" — must not need
the coding agent. The creator's overrides are saved with the project's run settings,
frozen per task like every other input, and applied here after the author process
has produced the scene and before any exporter sees it. The authored source stays
untouched; the description records which values came from the source and which from
an override, so a later reader can tell them apart.

Every property in PROPERTIES names its owner (the object, its rigid body or its joint),
its unit and the range a value may take. Nothing outside this table is overridable:
an override for a property the object does not have is an error, never ignored.
Scene-wide values (gravity) travel the same way under the reserved name SCENE, so one
saved shape covers everything the creator can change without the agent.
"""

import math
from dataclasses import dataclass

import numpy as np

MAX_OVERRIDDEN_OBJECTS = 512


@dataclass(frozen=True)
class Property:
    owner: str  # "object" | "body" | "joint" | "scene"
    unit: str
    minimum: float
    maximum: float
    # Whether a running MuJoCo session can take the new value without a rebuild.
    live: bool
    label: str


PROPERTIES = {
    "friction": Property("object", "", 0.0, 5.0, True, "Friction"),
    "restitution": Property("object", "", 0.0, 1.0, False, "Bounce"),
    "torsional_friction": Property("object", "", 0.0, 1.0, True, "Torsional friction"),
    "rolling_friction": Property("object", "", 0.0, 1.0, True, "Rolling friction"),
    "mass": Property("body", "kg", 0.001, 100_000.0, True, "Mass"),
    "damping": Property("joint", "N·m·s/rad or N·s/m", 0.0, 1_000.0, True, "Damping"),
    "frictionloss": Property("joint", "N·m or N", 0.0, 1_000.0, True, "Joint friction"),
    "limit_min": Property("joint", "rad or m", -100.0, 0.0, True, "Lower limit"),
    "limit_max": Property("joint", "rad or m", 0.0, 100.0, True, "Upper limit"),
}
SCENE_PROPERTIES = {
    "gravity_m_s2": Property("scene", "m/s²", 0.0, 50.0, True, "Gravity"),
}
SCENE = "__scene__"  # the overrides key that carries SCENE_PROPERTIES; never an object name


def _number(name, key, value, spec):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name}.{key}: override must be a finite number")
    if not spec.minimum <= value <= spec.maximum:
        raise ValueError(
            f"{name}.{key}: {value} is outside {spec.minimum}..{spec.maximum} {spec.unit}".rstrip()
        )
    return float(value)


def validate_overrides(value):
    """Shape check only, for the settings schema and the API: names and numbers."""
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > MAX_OVERRIDDEN_OBJECTS:
        raise ValueError("overrides must map at most 512 object names to property values")
    result = {}
    for name, properties in value.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError("overrides: object names must be short non-empty strings")
        if not isinstance(properties, dict) or not properties:
            raise ValueError(f"{name}: overrides must be a non-empty mapping of properties")
        checked = {}
        table = SCENE_PROPERTIES if name == SCENE else PROPERTIES
        for key, number in properties.items():
            spec = table.get(key)
            if spec is None:
                raise ValueError(f"{name}.{key}: unknown property")
            checked[key] = _number(name, key, number, spec)
        if "limit_min" in checked and "limit_max" in checked and (
            checked["limit_min"] >= checked["limit_max"]
        ):
            raise ValueError(f"{name}: limit_min must be below limit_max")
        result[name] = checked
    return result


def apply_overlay(scene, overrides):
    """Apply validated overrides to the scene. Returns {object: {property: value}} of
    what actually changed, for the description's provenance."""
    overrides = validate_overrides(overrides)
    objects = {obj.name: obj for obj in scene.objects}
    applied = {}
    for name, properties in overrides.items():
        if name == SCENE:
            for key, value in properties.items():
                setattr(scene, key, value)
            applied[SCENE] = dict(properties)
            continue
        obj = objects.get(name)
        if obj is None:
            raise ValueError(f"Override names an object the scene does not have: {name}")
        changed = {}
        for key, value in properties.items():
            spec = PROPERTIES[key]
            if spec.owner == "object":
                if not obj.collidable and key != "restitution":
                    raise ValueError(f"{name}.{key}: this part does not collide")
                setattr(obj, key, value)
            elif obj.body is None:
                raise ValueError(f"{name}.{key}: this part is static and has no body")
            elif spec.owner == "body":
                # Keep the inertia consistent with the new mass: same shape, new weight.
                ratio = value / obj.body.mass
                obj.body.diagonal_inertia = tuple(
                    float(component * ratio) for component in obj.body.diagonal_inertia
                )
                obj.body.mass = value
            elif obj.body.motion == "free":
                raise ValueError(f"{name}.{key}: a loose object has no joint")
            elif key == "limit_min":
                obj.body.limits = (value, float(obj.body.limits[1]))
            elif key == "limit_max":
                obj.body.limits = (float(obj.body.limits[0]), value)
            else:
                setattr(obj.body, key, value)
            changed[key] = value
        if obj.body is not None and obj.body.motion != "free" and (
            obj.body.limits[0] >= obj.body.limits[1]
            or not obj.body.limits[0] <= 0 <= obj.body.limits[1]
        ):
            raise ValueError(f"{name}: joint limits must stay ordered around the rest pose")
        applied[name] = changed
    return applied


def property_catalog(obj):
    """Which overridable properties this object has, with their current values."""
    result = {}
    for key, spec in PROPERTIES.items():
        if spec.owner == "object":
            if not obj.collidable and key != "restitution":
                continue
            value = getattr(obj, key)
        elif obj.body is None:
            continue
        elif spec.owner == "body":
            value = obj.body.mass
        elif obj.body.motion == "free":
            continue
        elif key in {"limit_min", "limit_max"}:
            # A driven spin hinge legitimately has no limits: it turns forever.
            if obj.body.limits is None:
                continue
            value = obj.body.limits[0 if key == "limit_min" else 1]
        else:
            value = getattr(obj.body, key)
        result[key] = {
            "value": float(np.round(value, 6)),
            "unit": spec.unit,
            "label": spec.label,
            "minimum": spec.minimum,
            "maximum": spec.maximum,
            "live": spec.live,
        }
    return result
