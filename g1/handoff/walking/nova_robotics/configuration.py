"""Optional user constraints, shared by the API and isolated exporters.

Exports always cover every engine; MuJoCo is the engine that is built and tested
here. `targets` survives only as an export filter for projects pinned to an older
schema that still saved one; a missing or empty list means every engine.
"""

import math

from .overlay import SCENE, apply_overlay, validate_overrides

COMPLEXITY = {
    "auto": {
        "max_objects": 2000,
        "max_triangles": 2_000_000,
        "guidance": (
            "Default to a medium-complexity, fully developed environment at the requested "
            "dimensions. Prioritize precise structure, materials and working interactions; "
            "not a small scene, rough blockout or maximum object count."
        ),
    },
    "simple": {
        "max_objects": 128,
        "max_triangles": 100_000,
        "guidance": "Focused layout, few obstacles, simple geometry.",
    },
    "standard": {
        "max_objects": 500,
        "max_triangles": 500_000,
        "guidance": (
            "Medium complexity: complete structures, precise dimensions/placement, "
            "correct materials and testable interactions. Preserve the requested footprint."
        ),
    },
    "complex": {
        "max_objects": 2000,
        "max_triangles": 2_000_000,
        "guidance": (
            "Higher fidelity and interaction variety within the requested footprint. "
            "Not a request for a larger building, more clutter or filling the object budget. "
            "Prioritize complete, testable task areas over background detail."
        ),
    },
}
TARGETS = {"isaac_sim": "Isaac Sim", "mujoco": "MuJoCo", "pybullet": "PyBullet", "gazebo": "Gazebo"}
# The engine whose measurements decide a build's status, and the one whose results are
# reported beside it as a second opinion that never fails a build.
ACCEPTANCE_ENGINE = "mujoco"
COMPATIBILITY_ENGINE = "pybullet"
# The current schema saves robot, controller, base and `overrides`. Every other key is
# a setting an older pinned schema may still carry (mode, brief, seed, detail budget,
# first-build strategy, scene-wide gravity/contact values): honoured when saved, never
# shown. The usable-first work order is the only one; it needs no switch.
DEFAULTS = {
    "first_build_strategy": "usable_first",
    "mode": "environment",
    "task_description": "",
    "complexity": "auto",
    "gravity_m_s2": None,
    "friction": None,
    "restitution": None,
    "overrides": {},
    "seed": 0,
    "targets": None,
    "robot_upload_id": None,
    "robot_path": None,
    "fixed_base": None,
    "controller": None,
}


def validate_configuration(value: dict | None = None) -> dict:
    if value is not None and not isinstance(value, dict):
        raise ValueError("Scene settings must be an object")
    if set(value or {}) - DEFAULTS.keys():
        raise ValueError("Unknown scene setting")
    result = {**DEFAULTS, **(value or {})}
    # Older pinned projects still carry mode/brief/seed. New settings infer inclusion
    # from the selected vendor upload; omission must never request a generated robot.
    if value and "mode" not in value:
        result["mode"] = "robot_environment" if result["robot_upload_id"] else "environment"
    if result["first_build_strategy"] not in ("usable_first", "standard"):
        raise ValueError("Choose usable_first or standard for the first build approach")
    brief = result["task_description"]
    if not isinstance(brief, str) or len(brief) > 4000 or "\x00" in brief:
        raise ValueError("Describe the environment or simulation in at most 4000 characters")
    result["task_description"] = brief.strip()
    if result["mode"] not in ("environment", "robot_environment"):
        raise ValueError("Choose environment only or robot + environment")
    if result["mode"] == "environment" and any(
        result[key] is not None
        for key in ("robot_upload_id", "robot_path", "fixed_base", "controller")
    ):
        raise ValueError("Environment-only mode cannot select a robot or robot controller")
    targets = result["targets"]
    if targets is None or targets == []:
        targets = list(TARGETS)
    if (
        not isinstance(targets, list)
        or any(not isinstance(t, str) or t not in TARGETS for t in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("Select supported simulators: Isaac Sim, MuJoCo, PyBullet or Gazebo")
    if ACCEPTANCE_ENGINE not in targets:
        # Every build is tested in MuJoCo; a legacy export filter cannot drop it.
        targets = [ACCEPTANCE_ENGINE, *targets]
    result["targets"] = targets
    if result["controller"] not in (None, "go2_walk_v1", "household_mission_v1"):
        raise ValueError("Unknown simulation controller")
    for key in ("robot_upload_id", "robot_path"):
        item = result[key]
        if item is not None and (not isinstance(item, str) or not item or len(item) > 1024):
            raise ValueError(f"{key} must identify an uploaded robot")
    if result["fixed_base"] is not None and type(result["fixed_base"]) is not bool:
        raise ValueError("Choose a fixed or mobile robot base")
    if not isinstance(result["complexity"], str) or result["complexity"] not in COMPLEXITY:
        raise ValueError("Choose auto, simple, standard or complex")
    for key, maximum in (("gravity_m_s2", 50), ("friction", 5), ("restitution", 1)):
        number = result[key]
        if number is not None and (
            type(number) not in (float, int)
            or not math.isfinite(number)
            or not 0 <= number <= maximum
        ):
            raise ValueError(f"{key} must be blank (automatic) or between 0 and {maximum}")
    result["overrides"] = validate_overrides(result["overrides"])
    if type(result["seed"]) is not int or not 0 <= result["seed"] <= 2**32 - 1:
        raise ValueError("Seed must be an integer between 0 and 4294967295")
    return result


def controller_for(requested, model):
    """The controller a build can honour for this robot, and a note when it differs.

    Studio offers only fitting controllers, but a saved setting can outlive a robot
    change: the Go2 policy only walks a Go2, the household mission needs a base that
    can move. Returns (controller or None, note or None)."""
    if not requested:
        return None, None
    if model is None:
        return None, "No robot is included, so the controller was not run."
    name = str(model.get("name") or "").lower()
    if requested == "go2_walk_v1" and name != "go2":
        return None, (
            "The Go2 walking policy only runs on the Unitree Go2; this robot was placed "
            "without a controller."
        )
    if requested == "household_mission_v1" and model.get("mobile_base") is False:
        return None, (
            "The scripted household demo needs a robot whose base can move; this robot "
            "was placed without a controller."
        )
    return requested, None


def apply_configuration(scene, value: dict) -> tuple[dict, dict]:
    """Enforce the complexity budget and apply the creator's overrides.

    Returns (constraints, applied): the build configuration record and the map of
    every value an override changed, keyed by object name (`__scene__` for gravity),
    which the world description turns into provenance.
    """
    config = validate_configuration(value)
    limits = COMPLEXITY[config["complexity"]]
    triangles = sum(
        len(obj.mesh.faces) + sum(len(m.faces) for m in obj.collision_meshes or ())
        for obj in scene.objects
    )
    if len(scene.objects) > limits["max_objects"] or triangles > limits["max_triangles"]:
        raise ValueError(f"Scene exceeds {config['complexity']} complexity limits: {limits}")
    overrides = {name: dict(properties) for name, properties in config["overrides"].items()}
    if config["gravity_m_s2"] is not None:
        # Legacy scene-wide setting; a saved per-scene override wins over it.
        overrides.setdefault(SCENE, {}).setdefault("gravity_m_s2", config["gravity_m_s2"])
    for key in ("friction", "restitution"):
        if config[key] is not None:
            for obj in scene.objects:
                if obj.collidable or key == "restitution":
                    overrides.setdefault(obj.name, {}).setdefault(key, config[key])
    applied = apply_overlay(scene, overrides)
    return {
        "requested": config,
        "limits": limits,
        "triangle_count": triangles,
        "object_count": len(scene.objects),
        "overrides_applied": applied,
    }, applied
