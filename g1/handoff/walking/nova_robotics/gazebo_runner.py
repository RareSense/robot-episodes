"""Play this environment, with its prescribed motion, in Gazebo Sim.

    python3 run_gazebo.py              # runs the world in real time; in a second
                                       # terminal, `gz sim -g` opens the viewer
    python3 run_gazebo.py --headless   # bounded check: 1,000 steps, prints a summary

Gazebo loads world.sdf on its own, but nothing in it evaluates motion_program.json:
the patrol bodies are kinematic links and the swinging/spinning parts plain joints,
so `gz sim world.sdf` alone shows the obstacles standing still. This runner is the
evaluator — the same exact trajectory functions (nova_motion.py) every other
engine's runner uses — applied before every physics step at SIMULATION time, so the
obstacles can never drift from the bodies they push.

Needs Gazebo Sim with its Python bindings (gz.sim, gz.math), e.g. Gazebo Jetty.
"""

import argparse
import json
from pathlib import Path

import nova_motion
from gz.math import Pose3d
from gz.sim import Joint, Model, TestFixture, World, world_entity

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="headless step count (default 1000 = 2 s)")
    args = parser.parse_args()
    program = json.loads((HERE / "motion_program.json").read_text())["bodies"]
    fixture = TestFixture(str(HERE / "world.sdf"))
    resolved = {}
    stats = {"steps": 0, "driven": 0}

    def drive(info, ecm):
        t = info.sim_time.total_seconds()
        if not resolved:
            world = World(world_entity(ecm))
            for body in program:
                model = Model(world.model_by_name(ecm, body["sdf"]["model"]))
                if not model.valid(ecm):
                    continue
                joint = None
                if body["motion"] in {"hinge", "slide"}:
                    joint = Joint(model.joint_by_name(ecm, "joint"))
                    joint.enable_position_check(ecm, True)
                resolved[body["sdf"]["model"]] = (body, model, joint)
            stats["driven"] = len(resolved)
        for body, model, joint in resolved.values():
            drive_spec = body["drive"]
            if body["motion"] == "kinematic":
                (x, y, z), yaw = nova_motion.patrol_pose(drive_spec, t)
                model.set_world_pose_cmd(ecm, Pose3d(x, y, z, 0.0, 0.0, yaw))
            elif joint is not None:
                joint.reset_position(ecm, [nova_motion.joint_position(drive_spec, t)])
                joint.reset_velocity(ecm, [nova_motion.joint_velocity(drive_spec, t)])
        stats["steps"] += 1

    fixture.on_pre_update(drive)
    fixture.finalize()
    server = fixture.server()
    if args.headless:
        server.run(True, args.iterations, False)
        print(f"Gazebo playback finished after {stats['steps']} steps with "
              f"{stats['driven']} driven bodies. No controller or policy was run.")
        return
    print("World running. Open the viewer in another terminal:  gz sim -g")
    server.run(True, 0, False)  # 0 iterations = run until interrupted


if __name__ == "__main__":
    main()
