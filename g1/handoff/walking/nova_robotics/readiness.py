"""Fail closed before spending model time; repeat authoritative checks in the compiler."""

from .configuration import validate_configuration

# Native robot fidelity takes precedence over pretending universal conversion works:
# a robot description goes to the one engine that reads it natively. Every other
# engine still receives the environment, without the robot, and says so.
COMPATIBLE = {"urdf": {"pybullet"}, "usd": {"isaac_sim"}, "mjcf": {"mujoco"}, "sdf": {"gazebo"}}


def robot_engine(model):
    """The single engine an uploaded robot description is native to."""
    return next(iter(COMPATIBLE.get(model["format"], ())), None) if model else None


def require_robot_selection(settings, uploads):
    config = validate_configuration(settings)
    if config["mode"] == "environment":
        return None
    if not config["robot_upload_id"] or not config["robot_path"]:
        raise ValueError(
            "Upload and select an actual robot description (URDF, articulated USD, robot SDF "
            "or MJCF) with its assets. A warehouse/building is an environment, not a robot."
        )
    selected = next((u for u in uploads if u["id"] == config["robot_upload_id"]), None)
    if selected is None or selected.get("state") != "preserved":
        raise ValueError(
            "The selected robot upload is unavailable. Upload the complete robot package."
        )
    models = selected.get("findings", {}).get("models", [])
    model = next((m for m in models if m["path"] == config["robot_path"]), None)
    if model is None or model.get("role") != "robot" and model.get("format") != "usd":
        raise ValueError(
            "This file is not a confirmed articulated robot. Upload a robot description "
            "with links/joints and its meshes, not a warehouse, table or building."
        )
    if model.get("missing_dependencies"):
        raise ValueError(
            "Robot assets are missing. Upload one ZIP containing the robot description "
            "and all referenced meshes/textures: " + ", ".join(model["missing_dependencies"][:5])
        )
    if robot_engine(model) is None:
        raise ValueError("This robot description has no native simulator in this pack.")
    if model["format"] == "urdf" and config["fixed_base"] is None:
        raise ValueError(
            "Choose whether this URDF robot has a fixed or mobile base in Robot setup."
        )
    return model
