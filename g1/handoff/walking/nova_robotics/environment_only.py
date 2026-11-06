"""Environment export only: parse native assets, never step physics or add replay."""

import mujoco


def assemble_environment(output, scene, target, config, reason=None):
    entry = {
        "isaac_sim": "environment.usda",
        "gazebo": "world.sdf",
        "mujoco": "environment.xml",
        "pybullet": "run_pybullet.py",
    }[target]
    if target == "pybullet":
        from .urdf_preview import assemble_pybullet

        entry, _ = assemble_pybullet(output, None, scene, config)
    if target == "mujoco":
        # Loading validates references/topology; no MjData, stepping or replay.
        mujoco.MjModel.from_xml_path(str(output / entry))
    return entry, {
        "status": "not_run",
        "mode": "environment_export_only",
        "engine": target,
        "training": "not_run",
        "robot_included": False,
        "reason": reason or (
            "Environment exported without a robot/controller session; "
            "environment mechanism checks are recorded separately in validation.json"
        ),
        "native_load_check": "passed" if target == "mujoco" else "not_run",
    }
