"""Learned locomotion for imported robots: typed velocity commands through real physics.

A walking policy is a function, not a recording: every 20 ms it reads what the robot senses
(base angular velocity, gravity direction in the base frame, joint angles and speeds, the
operator's velocity command) and returns target joint angles for the legs. Those targets go
through the same MotorLayer as everything else in byor: PD torque, clipped at the robot's
real actuator limits, applied by MuJoCo. Nothing is ever keyframed and the base is never
pushed.

Policies are matched to an imported robot by its own leg joint names, never by file name.
A robot nobody trained a policy for simply reports locomotion as unavailable (the written
gait controller for unknown robots is a separate, future layer).

Policy provenance lives next to each .onnx in nova_robotics/policies/<name>/PROVENANCE.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

POLICY_DIR = Path(__file__).resolve().parents[1] / "policies"
POLICY_DT = 0.02                 # every policy here was trained at 50 Hz

# Gait configurations are data, not code: each policy folder carries a gait.json with the
# values its network was trained against (joint order, scales, gains, poses). The code here
# is generic; adding a robot means adding a policy folder, never editing this module.
_ARRAYS = ("default", "kp", "kd", "stand_kp", "stand_kd", "cmd_scale")


def _load_gaits() -> dict:
    out = {}
    for path in sorted(POLICY_DIR.glob("*/gait.json")):
        if not (path.parent / "policy.onnx").is_file():
            continue
        cfg = json.loads(path.read_text())
        for key in _ARRAYS:
            if key in cfg:
                cfg[key] = np.asarray(cfg[key], float)
        cfg["action_scale"] = np.asarray(cfg["action_scale"], float)
        out[path.parent.name] = cfg
    return out


GAITS = _load_gaits()


def _session(path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class WalkingPolicy:
    """Velocity-commanded locomotion for one imported robot, through its MotorLayer."""

    @staticmethod
    def detect(m: mujoco.MjModel) -> str | None:
        """The gait whose full set of leg joint names exists in this model, if any."""
        names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)}
        for name, cfg in GAITS.items():
            if set(cfg["legs"]) <= names and (POLICY_DIR / name / "policy.onnx").is_file():
                return name
        return None

    @staticmethod
    def prepare_spec(spec: mujoco.MjSpec) -> None:
        """Before compiling: where a matched gait was trained with different foot contact
        geometry than the robot file ships, swap the collision feet for the training meshes
        (visuals untouched). Does nothing for robots without a gait or without train_feet."""
        try:
            names = {j.name for j in spec.joints}
        except (AttributeError, TypeError):
            return
        for gait, cfg in GAITS.items():
            if not (set(cfg["legs"]) <= names and (POLICY_DIR / gait / "policy.onnx").is_file()):
                continue
            for body, rel in cfg.get("train_feet", {}).items():
                mesh_file = POLICY_DIR / rel
                b = spec.body(body)
                if b is None or not mesh_file.is_file():
                    continue
                for g in list(b.geoms):
                    if g.contype or g.conaffinity:
                        g.contype = 0
                        g.conaffinity = 0
                mname = f"train_foot_{body}"
                spec.add_mesh(name=mname, file=str(mesh_file))
                b.add_geom(name=f"{body}_sole", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname,
                           contype=1, conaffinity=1, group=3, density=0, mass=0,
                           rgba=[0.2, 0.2, 0.2, 0.0])
            return

    def __init__(self, m: mujoco.MjModel, motors, name: str, feet=()):
        cfg = GAITS[name]
        self.m, self.motors, self.name, self.cfg = m, motors, name, cfg
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["legs"]]
        self.q = m.jnt_qposadr[jid].copy()
        self.v = m.jnt_dofadr[jid].copy()
        self.ranges = m.jnt_range[jid].copy()
        # the robot's own floating base: the free joint whose subtree contains the legs
        # (scene props are free bodies too, so "the one free joint" does not exist)
        from .analysis import subtree
        leg_bodies = set(int(m.jnt_bodyid[j]) for j in jid)
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and leg_bodies <= set(subtree(m, int(m.jnt_bodyid[j])))]
        if len(free) != 1:
            raise ValueError("locomotion needs exactly one floating base above the legs")
        self.qa, self.da = int(m.jnt_qposadr[free[0]]), int(m.jnt_dofadr[free[0]])
        self.base = int(m.jnt_bodyid[free[0]])
        self.period = round(POLICY_DT / m.opt.timestep)
        if self.period < 1 or not np.isclose(self.period * m.opt.timestep, POLICY_DT):
            raise ValueError(f"timestep {m.opt.timestep} does not divide the 50 Hz policy rate")
        if dyn := cfg.get("leg_joint"):
            m.dof_armature[self.v] = dyn["armature"]
            m.dof_damping[self.v] = dyn["damping"]
            m.dof_frictionloss[self.v] = dyn["frictionloss"]
        # map leg joints / held joints onto the MotorLayer's joint motors
        J = motors.joint_idx
        dof_to_k = {int(motors.dof[k]): i for i, k in enumerate(J)}
        leg_dofs = set(self.v.tolist())
        self.leg_k = np.array([dof_to_k[int(v)] for v in self.v], int)
        self.hold_k = np.array([i for i, k in enumerate(J) if int(motors.dof[k]) not in leg_dofs],
                               int)
        self.hold_q = motors.qadr[J[self.hold_k]]
        hname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT,
                                   int(m.dof_jntid[motors.dof[J[i]]])) or "" for i in self.hold_k]
        gains = [next((g for g in cfg["hold_gains"] if g[0] in n), cfg["hold_gains"][-1])
                 for n in hname]
        self.hold_kp = np.array([g[1] for g in gains], float)
        self.hold_kd = np.array([g[2] for g in gains], float)
        self.hold_target = np.array([cfg["hold_pose"].get(n, 0.0) for n in hname], float)
        self.hold_named = np.array([n in cfg["hold_pose"] for n in hname], bool)
        self.session = _session(POLICY_DIR / name / "policy.onnx")
        ins = {i.name: i.shape for i in self.session.get_inputs()}
        self.lstm = "h_in" in ins
        self.h0 = ((np.zeros(ins["h_in"], np.float32), np.zeros(ins["c_in"], np.float32))
                   if self.lstm else None)
        self.command = np.zeros(3)                  # operator: vx, vy [m/s], yaw rate [rad/s]
        self.accel = np.array([1.0, 1.0, 2.0])
        self._cmd = np.zeros(3)
        self.active = False
        self.measured = np.zeros(3)
        self._hist: list[np.ndarray] = []
        self.feet = [b for b in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in feet)
                     if b >= 0]
        self._reset_state()

    # ------------------------------------------------------------------ lifecycle
    def _reset_state(self):
        self.last_action = np.zeros(len(self.q))
        self.target = self.cfg["default"].copy()
        if self.lstm:
            self.h, self.c = self.h0[0].copy(), self.h0[1].copy()
        self.step_count = 0
        self.policy_calls = 0
        self._cmd[:] = 0.0
        self._hist.clear()
        self.phase_state = "blend"
        self.settled = False
        self._idle = 0.0
        self._settle_t = 0.0
        self._quiet = 0.0
        self._fz: list[np.ndarray] = []
        self._bal = None
        self._com_prev = None
        self._comv = np.zeros(2)

    def activate(self, d: mujoco.MjData):
        """Hand control to the policy from whatever pose the robot is standing in. Phases:
        blend (legs ramp from here to the policy's stance), walk (the network runs; its
        command is held at zero for warm_s while the recurrent state builds, the way the
        vendor's own deployment starts), settle (command gone: legs ramp back to the stance
        they stood in and hold, so the handoff back to balance never happens mid-step)."""
        self._reset_state()
        self.blend_from = d.qpos[self.q].copy()
        # un-named held joints keep the pose they have (an operator may have placed the arms);
        # named ones ramp to the pose the policy was trained with
        cur = d.qpos[self.hold_q].copy()
        self.hold_start = cur
        self.hold_target = np.where(self.hold_named, self.hold_target, cur)
        self.active = True

    def deactivate(self):
        self.active = False
        self.settled = False

    # ------------------------------------------------------------------ per-step control
    def base_speed(self, d) -> float:
        return float(np.linalg.norm(d.qvel[self.da:self.da + 2]))

    def _gravity_yaw(self, d):
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, d.qpos[self.qa + 3:self.qa + 7])
        R = rot.reshape(3, 3)
        return R.T @ np.array([0.0, 0.0, -1.0]), math.atan2(R[1, 0], R[0, 0])

    def _measure(self, d, yaw):
        """Gait-averaged (vx, vy, yaw rate) in the heading frame, for the state display."""
        n = int(round((self.cfg["gait_period"] or 0.5) / POLICY_DT))
        if self._hist:
            yaw = self._hist[-1][2] + math.remainder(yaw - self._hist[-1][2], 2 * math.pi)
        self._hist.append(np.array([d.qpos[self.qa], d.qpos[self.qa + 1], yaw]))
        if len(self._hist) > n + 1:
            self._hist.pop(0)
        if len(self._hist) > n:
            a, b = self._hist[0], self._hist[-1]
            T = (len(self._hist) - 1) * POLICY_DT
            mid = 0.5 * (a[2] + b[2])
            cs, sn = math.cos(mid), math.sin(mid)
            dxy = (b[:2] - a[:2]) / T
            self.measured = np.array([cs * dxy[0] + sn * dxy[1], -sn * dxy[0] + cs * dxy[1],
                                      (b[2] - a[2]) / T])

    def _balance_offsets(self, d):
        """Standing ankle strategy (plus lateral hip/ankle-roll shift): PD-target offsets on the
        leg joints that bring the centre of mass over the middle of the feet. A tall humanoid
        held in a frozen stance tips over in about a second; this is the active part real
        robots run while 'standing still'. Ported from the proven film walker."""
        cfg = self.cfg
        out = np.zeros(len(self.q))
        if "balance" not in cfg:
            return out
        _, yaw = self._gravity_yaw(d)
        cs, sn = math.cos(yaw), math.sin(yaw)
        Rt = np.array([[cs, sn], [-sn, cs]])
        com = d.subtree_com[self.base][:2].copy()
        feet = np.mean([d.xpos[b][:2] for b in self.feet], axis=0) if self.feet else com
        e = Rt @ (com - feet) - np.array([cfg["foot_cx"], 0.0])
        if self._com_prev is None:
            self._com_prev = com.copy()
        v = Rt @ (com - self._com_prev) / POLICY_DT
        self._com_prev = com.copy()
        self._comv = 0.7 * self._comv + 0.3 * v
        kx, kv, ky, kvy = cfg["balance"]
        pitch = float(np.clip(kx * e[0] + kv * self._comv[0], -0.15, 0.15))
        roll = float(np.clip(ky * e[1] + kvy * self._comv[1], -0.1, 0.1))
        for i in cfg.get("ankle_pitch_idx", ()):
            out[i] += pitch
        for i in cfg.get("hip_roll_idx", ()):
            out[i] += roll
        for i in cfg.get("ankle_roll_idx", ()):
            out[i] -= roll
        return out

    def tick(self, d: mujoco.MjData):
        """One physics step of walking. Returns (q_des, kp, kd) over the MotorLayer's joint
        motors; the caller passes them to motors.command, which clips at the real limits."""
        cfg = self.cfg
        dt = self.m.opt.timestep
        t = self.step_count * dt
        warm_until = cfg["blend_s"] + cfg["warm_s"]
        # phase transitions
        if self.phase_state == "blend" and t >= cfg["blend_s"]:
            self.phase_state = "walk"
        if self.phase_state == "settle" and np.any(np.abs(self.command) > 1e-3):
            cmd = self.command.copy()
            self.activate(d)                       # fresh start from the held stance
            self.command = cmd
        # policy tick (walk phase only)
        if self.phase_state == "walk" and self.step_count % self.period == 0:
            vmax = np.asarray(cfg["vmax"])
            want = np.zeros(3) if t < warm_until else np.clip(self.command, -vmax, vmax)
            self._cmd += np.clip(want - self._cmd, -self.accel * POLICY_DT, self.accel * POLICY_DT)
            gravity, yaw = self._gravity_yaw(d)
            self._measure(d, yaw)
            # idle: the policy marches in place at zero command; leave the gait only at a
            # double-support instant (every foot at its lowest of the last 0.8 s) with no
            # net progress and a slow base — a timer alone hands over with a foot in the air
            if (np.all(np.abs(self.command) < 1e-3) and np.all(np.abs(self._cmd) < 1e-6)
                    and t >= warm_until):
                self._idle += POLICY_DT
                if self._idle > cfg["idle_s"]:
                    feet_z = (np.array([d.xpos[f][2] for f in self.feet]) if self.feet
                              else np.zeros(1))
                    self._fz.append(feet_z)
                    if len(self._fz) > 40:
                        self._fz.pop(0)
                    low = np.min(self._fz, axis=0)
                    if (len(self._fz) >= 40 and np.all(feet_z - low < 0.004)
                            and self.base_speed(d) < 0.3
                            and np.linalg.norm(self.measured[:2]) < 0.06):
                        self.phase_state = "settle"
                        self._settle_t = 0.0
                        self.settle_from = d.qpos[self.q].copy()
            else:
                self._idle = 0.0
                self._fz.clear()
            obs = [d.qvel[self.da + 3:self.da + 6] * 0.25, gravity,
                   self._cmd * cfg["cmd_scale"],
                   d.qpos[self.q] - cfg["default"], d.qvel[self.v] * 0.05, self.last_action]
            if cfg["obs"] == "lstm_phase":
                phase = (t - cfg["blend_s"]) % cfg["gait_period"] / cfg["gait_period"]
                obs.append([math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)])
            o = np.clip(np.concatenate(obs), -100, 100).astype(np.float32)[None]
            if self.lstm:
                a, self.h, self.c = self.session.run(
                    None, {"obs": o, "h_in": self.h, "c_in": self.c})
            else:
                a = self.session.run(None, {"obs": o})[0]
            a = a.reshape(-1).astype(float)
            if a.shape != self.last_action.shape or not np.isfinite(a).all():
                raise ValueError(f"{self.name}: invalid policy action")
            self.last_action = np.clip(a, -100, 100)
            self.target = np.clip(cfg["default"] + cfg["action_scale"] * self.last_action,
                                  self.ranges[:, 0], self.ranges[:, 1])
            self.policy_calls += 1
        self.step_count += 1
        nj = len(self.motors.joint_idx)
        q_des, kp, kd = np.zeros(nj), np.zeros(nj), np.zeros(nj)
        hold = self.hold_target
        if self.phase_state == "blend":
            s = t / cfg["blend_s"]
            q_des[self.leg_k] = self.blend_from + s * (cfg["default"] - self.blend_from)
            bkp, bkd = cfg.get("blend_gains", (None, None))
            kp[self.leg_k] = bkp if bkp else cfg["kp"]
            kd[self.leg_k] = bkd if bkd else cfg["kd"]
            hold = self.hold_start + s * (self.hold_target - self.hold_start)
        elif self.phase_state == "walk":
            q_des[self.leg_k] = self.target
            kp[self.leg_k] = cfg["kp"]
            kd[self.leg_k] = cfg["kd"]
        else:                                      # settle: back to the stance it stood up in
            self._settle_t += dt
            u = 0.5 - 0.5 * math.cos(math.pi * min(1.0, self._settle_t / cfg["settle_s"]))
            if self._bal is None or self.step_count % self.period == 0:
                self._bal = self._balance_offsets(d)        # active balance, at the policy rate
            q_des[self.leg_k] = (self.settle_from + u * (self.blend_from - self.settle_from)
                                 + u * self._bal)
            kp[self.leg_k] = cfg["stand_kp"]
            kd[self.leg_k] = cfg["stand_kd"]
            # "settled" only after half a second of genuine quiet: a handoff during the
            # residual sway collapses even a controller that holds fine from rest
            if self._settle_t >= cfg["settle_s"]:
                quiet = (self.base_speed(d) < 0.06
                         and float(np.abs(d.qvel[self.da + 3:self.da + 6]).max()) < 0.3
                         and float(np.abs(d.qvel[self.v]).max()) < 0.5)
                self._quiet = self._quiet + dt if quiet else 0.0
                if self._quiet >= 0.4:
                    self.settled = True
        q_des[self.hold_k] = hold
        kp[self.hold_k] = self.hold_kp
        kd[self.hold_k] = self.hold_kd
        return q_des, kp, kd
