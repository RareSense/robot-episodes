"""Opt-in Go2 walking demo. CPU ONNX inference; simulation only, no hardware I/O.

Policy and settings: Eximius-Labs/efferent go2_walk, commit
f51e1af953afe1016dae9ceeff2b5328fa51781d (rl_sar / robot_lab policy).
Joint order, observation scales, gains and action scales follow its config.yaml.
Source MJCF is never edited. Native actuators are disabled for this runner and
bounded PD torques are applied to joint DOFs, matching the upstream sim backend.
This does not reproduce vendor motor electronics or prove hardware safety.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

POLICY_SHA256 = "61f5174e862659305a7c219026bc02f215034ae26f1fdfdaf18fc907c809c138"
JOINTS = [
    f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")
]
DEFAULT_POSE = np.tile([0.0, 0.8, -1.5], 4)
ACTION_SCALE = np.tile([0.125, 0.25, 0.25], 4)


class Go2Controller:
    def __init__(self, model, data, policy_path, yaw_deg=90.0):
        if hashlib.sha256(Path(policy_path).read_bytes()).hexdigest() != POLICY_SHA256:
            raise ValueError("Go2 demo policy checksum mismatch")
        robot_joints = [
            j
            for j in range(model.njnt)
            if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(
                "nova_environment_"
            )
        ]
        if len(robot_joints) != 13 or model.nu != 12:
            raise ValueError("Go2 walking demo requires a free-base, 12-joint Go2")
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
        if min(joints) < 0 or any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError("Go2 controller joint names/types do not match this robot")
        free = [j for j in robot_joints if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError("Go2 demo requires exactly one free base")
        self.qa = int(model.jnt_qposadr[free[0]])
        self.da = int(model.jnt_dofadr[free[0]])
        self.q = model.jnt_qposadr[joints].copy()
        self.v = model.jnt_dofadr[joints].copy()
        self.ranges = model.jnt_range[joints].copy()
        if np.any(self.ranges[:, 0] > DEFAULT_POSE) or np.any(self.ranges[:, 1] < DEFAULT_POSE):
            raise ValueError("Policy standing pose violates source robot joint limits")
        self.period = round(0.02 / model.opt.timestep)
        if self.period < 1 or not np.isclose(self.period * model.opt.timestep, 0.02):
            raise ValueError("Go2 policy requires physics timestep dividing 20 ms")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(policy_path), opts, providers=["CPUExecutionProvider"]
        )
        if self.session.get_inputs()[0].shape != [1, 45]:
            raise ValueError("Go2 policy observation contract mismatch")
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        mujoco.mj_resetData(model, data)
        data.qpos[self.q] = DEFAULT_POSE
        yaw = math.radians(yaw_deg)
        data.qpos[self.qa + 3 : self.qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.forward = np.array([math.cos(yaw), math.sin(yaw)])
        self.start_xy = data.qpos[self.qa : self.qa + 2].copy()
        self.last_action = np.zeros(12)
        self.target = DEFAULT_POSE.copy()
        self.max_tilt = 0.0
        self.max_torque = 0.0
        self.max_contacts = 0
        self.failed = None
        self.step_count = 0
        self.policy_calls = 0
        mujoco.mj_forward(model, data)

    def before_step(self, model, data, speed=0.4):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[self.qa + 3 : self.qa + 7])
        gravity = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))
        self.max_tilt = max(self.max_tilt, tilt)
        if tilt > 55 and self.failed is None:
            self.failed = "fall_detected"
        settling = data.time < 1.5
        if not settling and self.step_count % self.period == 0 and not self.failed:
            obs = np.concatenate(
                [
                    data.qvel[self.da + 3 : self.da + 6] * 0.25,
                    gravity,
                    [speed, 0.0, 0.0],
                    data.qpos[self.q] - DEFAULT_POSE,
                    data.qvel[self.v] * 0.05,
                    self.last_action,
                ]
            )
            obs = np.clip(obs, -100.0, 100.0).astype(np.float32)[None]
            action = self.session.run(["actions"], {"obs": obs})[0].reshape(-1)
            if action.shape != (12,) or not np.isfinite(action).all():
                raise ValueError("Invalid Go2 policy action")
            self.last_action = np.clip(action, -100.0, 100.0)
            self.target = np.clip(
                DEFAULT_POSE + ACTION_SCALE * self.last_action, self.ranges[:, 0], self.ranges[:, 1]
            )
            self.policy_calls += 1
        kp, kd = (80.0, 3.0) if settling else (20.0, 0.5)
        torque = kp * (self.target - data.qpos[self.q]) - kd * data.qvel[self.v]
        if self.failed:
            torque = -3.0 * data.qvel[self.v]
        torque = np.clip(torque, -23.5, 23.5)
        self.max_torque = max(self.max_torque, float(np.max(np.abs(torque))))
        data.qfrc_applied[self.v] = torque
        self.step_count += 1

    def after_step(self, model, data):
        self.max_contacts = max(self.max_contacts, data.ncon)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Controlled Go2 simulation produced non-finite state")
        if np.any(data.warning.number):
            raise ValueError("Controlled Go2 simulation emitted numerical warnings")

    def report(self, model, data):
        distance = float((data.qpos[self.qa : self.qa + 2] - self.start_xy) @ self.forward)
        if data.time >= 10 and distance < 0.5 and self.failed is None:
            self.failed = "insufficient_forward_progress"
        return {
            "status": "passed" if not self.failed else "failed",
            "engine": f"MuJoCo {mujoco.__version__}",
            "mode": "go2_learned_velocity_policy",
            "controller": "go2_walk_v1",
            "policy_sha256": POLICY_SHA256,
            "duration_s": round(float(data.time), 3),
            "steps": self.step_count,
            "policy_calls": self.policy_calls,
            "policy_rate_hz": 50,
            "command_forward_m_s": 0.4,
            "settling_s": 1.5,
            "forward_distance_m": distance,
            "max_tilt_deg": self.max_tilt,
            "max_applied_joint_torque_nm": self.max_torque,
            "maximum_contacts": self.max_contacts,
            "failure_reason": self.failed,
            "robot_dofs": 18,
            "environment_dofs": model.nv - 18,
            "actuators": model.nu,
            "training": "not_run",
            "scope": "Controller demo only; no course completion or hardware certification",
            "control_interface": "native actuation disabled; bounded PD joint torques",
        }


def main():
    parser = argparse.ArgumentParser(description="Simulation-only Go2 walking policy demo")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=12.0)
    args = parser.parse_args()
    if not 0 < args.duration <= 60:
        parser.error("duration must be between 0 and 60 seconds")
    root = Path(__file__).resolve().parent
    model = mujoco.MjModel.from_xml_path(str(root / "simulation.xml"))
    data = mujoco.MjData(model)
    ctrl = Go2Controller(model, data, root / "controller/policy.onnx")
    viewer = None
    try:
        if not args.headless:
            from mujoco import viewer as gui

            viewer = gui.launch_passive(model, data)
        for _ in range(round(args.duration / model.opt.timestep)):
            begin = time.monotonic()
            ctrl.before_step(model, data)
            mujoco.mj_step(model, data)
            ctrl.after_step(model, data)
            if viewer:
                if not viewer.is_running():
                    break
                if ctrl.step_count % ctrl.period == 0:
                    viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.monotonic() - begin)))
        print(json.dumps(ctrl.report(model, data), indent=2))
    finally:
        if viewer:
            viewer.close()


if __name__ == "__main__":
    main()
