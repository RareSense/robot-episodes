"""Play this environment, with its prescribed motion, in MuJoCo.

    pip install mujoco
    python run_mujoco.py              # interactive viewer, motion playing
    python run_mujoco.py --headless   # bounded passive check, prints a summary

Opening environment.xml in a bare MuJoCo viewer shows a STATIC world: the
exported position/velocity actuators and mocap bodies are there, but nothing
evaluates the motion program. This runner is that evaluator — the same exact
trajectory functions (nova_motion.py) every other engine's runner uses.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import nova_motion

HERE = Path(__file__).resolve().parent
TIMESTEP = 0.002


def load():
    model = mujoco.MjModel.from_xml_path(str(HERE / "environment.xml"))
    data = mujoco.MjData(model)
    driven = []
    program_path = HERE / "motion_program.json"
    if program_path.is_file():
        for body in json.loads(program_path.read_text())["bodies"]:
            names = body["mujoco"]
            if names.get("mocap"):
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, names["body"])
                if body_id >= 0 and model.body_mocapid[body_id] >= 0:
                    driven.append(("mocap", int(model.body_mocapid[body_id]), body["drive"]))
            elif names.get("actuator"):
                actuator = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, names["actuator"]
                )
                if actuator >= 0:
                    driven.append(("ctrl", actuator, body["drive"]))
    return model, data, driven


def apply_motion(data, driven, t):
    """Prescribed drives at time t: actuator targets for joints, poses for mocap."""
    for mode, index, drive in driven:
        if mode == "mocap":
            position, yaw = nova_motion.patrol_pose(drive, t)
            data.mocap_pos[index] = list(position)
            data.mocap_quat[index] = nova_motion.yaw_quaternion_wxyz(yaw)
        elif drive["kind"] == "spin":
            data.ctrl[index] = drive["rate_rad_s"]
        else:
            data.ctrl[index] = nova_motion.joint_position(drive, t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="headless duration (default 10)")
    args = parser.parse_args()
    model, data, driven = load()
    if args.headless:
        steps = max(1, int(args.seconds / TIMESTEP))
        for step in range(steps):
            apply_motion(data, driven, step * TIMESTEP)
            mujoco.mj_step(model, data)
        print(
            f"Passive MuJoCo playback finished after {steps} steps with "
            f"{len(driven)} driven bodies. No controller or policy was run."
        )
        return
    import mujoco.viewer as mujoco_viewer

    with mujoco_viewer.launch_passive(model, data) as viewer:
        # Motion is evaluated at PHYSICS time (data.time), never the wall clock, so a
        # slow frame can never let the scripted obstacles drift from the bodies they
        # push. The loop then paces physics to real time: it steps until it has
        # caught up, renders once, and sleeps only when it is ahead.
        start = time.monotonic()
        while viewer.is_running():
            target = time.monotonic() - start
            budget = time.monotonic() + 0.05  # never spend more than 50 ms catching up
            while data.time < target and time.monotonic() < budget:
                apply_motion(data, driven, data.time)
                mujoco.mj_step(model, data)
            if data.time < target:
                start += target - data.time  # too slow for real time: slow down, stay exact
            viewer.sync()
            ahead = data.time - (time.monotonic() - start)
            if ahead > 0:
                time.sleep(ahead)


if __name__ == "__main__":
    main()
