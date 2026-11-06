"""Physically faithful teleoperation of an imported robot.

The operator never moves joints. Commands become targets for the whole-body controller (hand
positions, gripper opening); the controller computes torques for the robot's real, limit-clipped
motors; MuJoCo integrates. Every step is checked (torque limits, joint ranges, penetration,
falls) and an episode can be recorded for training data with those checks attached.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .analysis import Report, robot_bodies, robot_roots, subtree
from .control import MotorLayer, WholeBodyController
from .locomotion import WalkingPolicy
from .stance import auto_home, settle, solve_stance, start_pose


def _level_quat(q_now, q_flat):
    """The orientation with the same heading (world yaw) as `q_now` whose sole is level:
    the z-twist of the rotation from `q_flat` (a known flush pose) applied back to it."""
    qi, r = np.empty(4), np.empty(4)
    mujoco.mju_negQuat(qi, q_flat)
    mujoco.mju_mulQuat(r, q_now, qi)
    twist = np.array([r[0], 0.0, 0.0, r[3]])
    n = np.linalg.norm(twist)
    if n < 1e-9:
        return q_flat.copy()
    out = np.empty(4)
    mujoco.mju_mulQuat(out, twist / n, q_flat)
    return out


@dataclass
class Checks:
    steps: int = 0
    torque_limit_violations: int = 0
    peak_torque_pct: float = 0.0
    joint_overshoot: float = 0.0
    max_penetration_mm: float = 0.0
    fell_at: float | None = None

    def as_dict(self):
        return dict(steps=self.steps, torque_limit_violations=self.torque_limit_violations,
                    peak_torque_pct=round(self.peak_torque_pct, 1),
                    joint_overshoot_mm_or_mrad=round(self.joint_overshoot * 1000, 2),
                    max_ground_penetration_mm=round(self.max_penetration_mm, 2), fell_at_s=self.fell_at)


@dataclass
class Episode:
    t: list = field(default_factory=list)
    qpos: list = field(default_factory=list)
    qvel: list = field(default_factory=list)
    torque: list = field(default_factory=list)
    hand_goal: list = field(default_factory=list)
    hand_pos: list = field(default_factory=list)
    grip: list = field(default_factory=list)
    walk_cmd: list = field(default_factory=list)
    walking: list = field(default_factory=list)


class TeleopSession:
    LEASH = 0.08          # metres a hand goal may lead the real hand
    ARM_SPEED = 2.5       # rad/s (m/s for sliding joints) for arm joints under teleoperation

    def __init__(self, spec: mujoco.MjSpec, report: Report, name: str = "robot", record_every: int = 10,
                 reach_fraction: float = 0.7):
        if not 0 < reach_fraction <= 0.95:
            raise ValueError("reach_fraction must be in (0, 0.95]")
        self.report, self.name = report, name
        s = spec.copy()
        if report.base == "floating":
            WalkingPolicy.prepare_spec(s)         # training foot contact where a gait needs it
        self.m = s.compile()
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        start_pose(m, d)
        if report.base == "fixed" and not m.nkey:
            auto_home(m, d, [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in report.end_effectors])
        if report.base == "floating":
            solve_stance(m, d, report)
        self.motors = MotorLayer(m)
        if report.base == "floating":
            settle(m, d, report, self.motors)
        self.wbc = WholeBodyController(m, report, self.motors, d)
        # the flush foot pose (orientation and height), captured while the soles sit flat from
        # solve_stance: the walk -> stand handoff re-levels the feet against this
        self._flat_foot = {f: (d.xquat[f].copy(), float(d.xpos[f][2])) for f in self.wbc.feet}
        self.root = robot_roots(m)[0]
        self.bodies = set(robot_bodies(m))
        self.com_h0 = float(d.subtree_com[self.root][2])
        self.hands = list(self.wbc.ees)
        self.home = {b: self.wbc.ee_goal[b].copy() for b in self.hands}
        root_inverse = d.xquat[self.root].copy()
        root_inverse[1:] *= -1
        self.neutral_hand_rot = {}
        for b in self.hands:
            relative = np.zeros(4)
            mujoco.mju_mulQuat(relative, root_inverse, d.xquat[b])
            self.neutral_hand_rot[b] = relative
        self.shoulder = {b: self._shoulder(b) for b in self.hands}
        # 70 % of the shoulder-to-tool distance with the arm straight: far from the singular straight
        # arm, and a reach a robot standing without stepping can hold (measured on random operators)
        self.reach = {b: reach_fraction * self._arm_length(b) for b in self.hands}
        # arm joints move at collaborative speeds under an operator; fast whips near shoulder
        # singularities are what rock a standing robot off its feet
        self.arm_dofs = sorted({int(m.jnt_dofadr[j]) for b in self.hands for x in self._arm_bodies(b)
                                for j in range(m.body_jntadr[x], m.body_jntadr[x] + m.body_jntnum[x])})
        self.wbc.speed_cap = {dof: self.ARM_SPEED for dof in self.arm_dofs}
        if report.base == "floating" and self.wbc.feet:
            # hand speeds scale with the robot: a 0.5 m robot swinging its arms at full-size speeds
            # is like a person flinging theirs at 2 m/s (reference: CoM height 0.7 m)
            scale = float(np.clip(self.com_h0 / 0.7, 0.3, 1.0))
            self.wbc.ref_vmax *= scale
            self.wbc.ref_amax *= scale
        self.fingers = {b: self._finger_joints(b) for b in self.hands}
        # the hand and its fingers, plus the forearm link it is mounted on
        self.hand_parts = {b: set(subtree(m, b)) | {int(m.body_parentid[b])} for b in self.hands}
        from .analysis import collision_geoms
        arm_parts = {b: set(self._arm_bodies(b)) | self.hand_parts[b] for b in self.hands}
        self.hand_geoms = {b: [g for x in self.hand_parts[b] for g in collision_geoms(m, x)] for b in self.hands}
        self.body_geoms = {b: [g for x in self.bodies - arm_parts[b] for g in collision_geoms(m, x)]
                           for b in self.hands}
        self.grip = {b: 0.0 for b in self.hands}
        self.grip_active = set()
        self._motor_at_qpos = {int(a): k for k, a in enumerate(self.motors.qadr) if a >= 0}
        self.checks = Checks()
        self.support = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in report.feet}
        self.guard_until, self.guard_events = 0.0, 0
        # the guard protects a human operator who cannot feel the balance margin; a motion
        # program that lifts a foot on purpose turns it off and checks the outcome itself
        self.operator_guard = True
        self.nominal_support = max(1, self._support_contacts())
        self.lock = threading.Lock()
        self.push = None
        # locomotion: a walking policy matched to this robot's own joint names, or None
        self.loco = None
        if report.base == "floating":
            gait = WalkingPolicy.detect(m)
            if gait is not None:
                try:
                    self.loco = WalkingPolicy(m, self.motors, gait, feet=report.feet)
                except ValueError:
                    self.loco = None
        self.mode = "stand"                   # "stand" (whole-body controller) or "walk" (policy)
        self._still_since = None
        self.recording: Episode | None = None
        self.record_every = record_every
        # Optional bounded streaming recorder. Hooks observe the actual command interval;
        # they must never change the live physics state or acquire this session's lock.
        self.step_observer = None
        self.limited = [(m.jnt_qposadr[j], m.jnt_range[j]) for j in range(m.njnt)
                        if m.jnt_limited[j] and int(m.jnt_type[j]) in (2, 3) and m.jnt_bodyid[j] in self.bodies]

    # ------------------------------------------------------------------ operator commands
    def hand_names(self):
        return [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.hands]

    def move_hand(self, index: int, delta):
        """Move a hand's goal by `delta` (world frame, metres). Like a real teleop rig, the goal is
        leashed: it never leads the real hand by more than `LEASH`, never extends past the arm's
        safe reach around its shoulder, never pushes into the robot's own body (from 3 cm away) and
        never goes below the floor, so holding a key cannot drag the robot towards a point it
        cannot reach."""
        with self.lock:
            if self.checks.fell_at is not None or self.d.time < self.guard_until:
                return
            if self.mode == "walk":              # while walking, the policy owns the whole body
                return
            if not 0 <= index < len(self.hands):
                return
            b = self.hands[index]
            d, w = self.d, self.wbc
            delta = np.asarray(delta, float)
            away = self._own_body_normals(b)
            for n in away:                         # never push a hand further into its own body
                into = delta @ n
                if into < 0:
                    delta = delta - into * n
            prev = w.ee_goal[b].copy()
            goal = prev + delta
            here = w.ee_pos(d, b)
            for n in away:                         # the goal may not sit inside the body either
                lead_n = (goal - here) @ n
                if lead_n < 0:
                    goal = goal - lead_n * n
            lead = goal - here
            if np.linalg.norm(lead) > self.LEASH:
                goal = here + lead * (self.LEASH / np.linalg.norm(lead))
            shoulder = d.xpos[self.shoulder[b]]
            limit = max(self.reach[b], float(np.linalg.norm(prev - shoulder)))
            off = goal - shoulder                  # never further out than the reach ball (or than
            if np.linalg.norm(off) > limit:        # it already is: a hanging arm is not pulled in)
                goal = shoulder + off * (limit / np.linalg.norm(off))
            goal[2] = max(goal[2], 0.05)
            w.set_ee_target(b, goal)

    def set_grip(self, index: int, value: float):
        """0 = open, 1 = closed. Finger joints follow through their motors (posture targets)."""
        with self.lock:
            if not 0 <= index < len(self.hands):
                return
            b = self.hands[index]
            self.grip[b] = float(np.clip(value, 0, 1))
            self.grip_active.add(b)
            for a, lo, hi, closes_high in self.fingers[b]:
                g = self.grip[b] if closes_high else 1 - self.grip[b]
                self.wbc.qpos0[a] = lo + g * (hi - lo)

    def neutral_hand_orientation(self, index: int):
        """Initial hand orientation, rotated with the robot's current base.

        A task may use this as a grasp hypothesis after a gait has changed the
        wrist orientation. It is a target, never a physical state reset.
        """
        result = np.zeros(4)
        mujoco.mju_mulQuat(result, self.d.xquat[self.root],
                          self.neutral_hand_rot[self.hands[index]])
        return result

    def use_grasp_point(self, index: int, closure: float = 0.6):
        """Track the estimated pinch centre instead of the hand-mesh centroid.

        Forward kinematics of a separate data object estimates the terminal
        finger-pad centre at a partial closure. The live robot state is untouched;
        target offsets are adjusted to preserve the current physical wrist pose.
        This is a geometric estimate, and a lift test must still prove each grasp.
        """
        from .analysis import collision_geoms

        if not 0 <= closure <= 1:
            raise ValueError("closure must be between zero and one")
        with self.lock:
            hand = self.hands[index]
            if not self.fingers[hand]:
                raise ValueError("this end effector has no driven fingers")
            m, d = self.m, self.d
            fk = mujoco.MjData(m)
            fk.qpos[:] = d.qpos
            driven = {a for a, *_ in self.fingers[hand]}
            for a, lo, hi, high in self.fingers[hand]:
                g = closure if high else 1 - closure
                fk.qpos[a] = lo + g * (hi - lo)
            mujoco.mj_kinematics(m, fk)
            finger_bodies = {int(m.jnt_bodyid[j]) for j in range(m.njnt)
                             if int(m.jnt_qposadr[j]) in driven}
            terminal = [b for b in finger_bodies
                        if not (set(subtree(m, b)) - {b}) & finger_bodies]
            pads = [np.mean(fk.geom_xpos[collision_geoms(m, b)], axis=0)
                    for b in terminal if collision_geoms(m, b)]
            if len(pads) < 2:
                raise ValueError("a pinch estimate requires at least two terminal fingers")
            offset = fk.xmat[hand].reshape(3, 3).T @ (np.mean(pads, axis=0) - fk.xpos[hand])
            shift = d.xmat[hand].reshape(3, 3) @ (offset - self.wbc.ee_off[hand])
            self.wbc.ee_off[hand] = offset
            for targets in (self.wbc.ee_goal, self.wbc.ee_targets, self.home):
                targets[hand] = targets[hand] + shift
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, hand)
            self.report.end_effector_point[name] = offset.tolist()
            return offset.copy()

    def shove(self, direction, speed=0.3):
        """A 0.2 s push on the base worth `speed` m/s of the robot's whole mass."""
        mass = float(self.m.body_subtreemass[self.root])
        with self.lock:
            self.push = (time.perf_counter() + 0.2, np.asarray(direction, float) * mass * speed / 0.2)

    def home_hands(self):
        with self.lock:
            for b in self.hands:
                self.wbc.set_ee_target(b, self.home[b].copy())

    def set_walk(self, cmd) -> bool:
        """Velocity command (vx, vy [m/s], yaw rate [rad/s]) in the robot's heading frame.
        A non-zero command hands the legs to the walking policy; once the command is zero and
        the robot has come to rest, the whole-body controller takes balance back where the
        robot now stands. Returns False when this robot has no locomotion."""
        with self.lock:
            if self.loco is None:
                return False
            if self.checks.fell_at is not None:
                return False
            cmd = np.asarray([float(x) for x in cmd][:3], float)
            if self.mode == "stand" and np.any(np.abs(cmd) > 1e-3):
                self.loco.activate(self.d)
                self.mode = "walk"
                self._still_since = None
            self.loco.command = cmd
            return True

    def _to_stand(self):
        """Walk -> stand handoff: a fresh whole-body controller built from the state the robot
        is in now (the same construction a new session uses), with its foot targets set to the
        flush sole pose at the robot's current heading, so it re-seats the slightly tilted
        feet the gait leaves behind. Only safe from a quiet state, which loco.settled
        guarantees: built mid-sway, even a correct controller falls."""
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        w = WholeBodyController(m, self.report, self.motors, d)
        for f in w.feet:                            # level the soles against the flush pose
            if f in self._flat_foot:
                q_flat, z_flat = self._flat_foot[f]
                w.foot_quat0[f] = _level_quat(d.xquat[f], q_flat)
                w.foot_anchor[f][2] = z_flat
                w.foot_goal[f][2] = z_flat
        w.speed_cap = self.wbc.speed_cap
        w.ref_vmax, w.ref_amax = self.wbc.ref_vmax, self.wbc.ref_amax
        self.wbc = w
        for b in self.hands:                        # gripper targets survive the handoff
            for a, lo, hi, closes_high in self.fingers[b]:
                g = self.grip[b] if closes_high else 1 - self.grip[b]
                w.qpos0[a] = lo + g * (hi - lo)
        self.home = {b: w.ee_goal[b].copy() for b in self.hands}
        self.loco.deactivate()
        self.mode = "stand"
        self._still_since = None

    # ------------------------------------------------------------------ physics
    def step(self, n=1):
        m, d = self.m, self.d
        for _ in range(n):
            with self.lock:
                if self.step_observer is not None:
                    self.step_observer.before_step(self)
                mujoco.mj_step1(m, d)
                if self.mode == "walk":
                    q_des, kp, kd = self.loco.tick(d)
                    self.motors.command(d, q_des=q_des, kp=kp, kd=kd)
                else:
                    for hand in self.grip_active:
                        for a, lo, hi, _ in self.fingers[hand]:
                            k = self._motor_at_qpos[a]
                            limit = self.motors.limit[k]
                            if not np.isfinite(limit):
                                continue
                            kp = 4 * limit / max(hi - lo, 0.01)
                            kd = 0.1 * limit
                            force = (kp * (self.wbc.qpos0[a] - d.qpos[a])
                                     - kd * d.qvel[self.motors.dof[k]])
                            self.wbc.force_targets[k] = float(np.clip(force, -limit, limit))
                    tau = self.wbc.step(d)
                    self.motors.command(d, tau_ff=tau)
                d.xfrc_applied[self.root, :3] = 0
                if self.push is not None:
                    if time.perf_counter() < self.push[0]:
                        d.xfrc_applied[self.root, :3] = self.push[1]
                    else:
                        self.push = None
                if self.step_observer is not None:
                    self.step_observer.before_integrate(self)
                mujoco.mj_step2(m, d)
                if self.step_observer is not None:
                    self.step_observer.after_step(self)
                self._check()
                if self.mode == "walk":
                    if self.checks.fell_at is not None:
                        self._to_stand()
                    elif self.loco.settled and self.loco.base_speed(d) < 0.05:
                        self._to_stand()
                elif self.checks.steps % 10 == 0 and self.operator_guard:
                    self._balance_guard()
                if self.recording is not None and self.checks.steps % self.record_every == 0:
                    self._record()

    def _check(self):
        m, d, c = self.m, self.d, self.checks
        c.steps += 1
        ap = np.abs(self.motors.applied(d))
        lim = self.motors.limit
        c.torque_limit_violations += int((ap > lim * 1.001).sum())
        fin = np.isfinite(lim) & (lim > 0)
        if fin.any():
            c.peak_torque_pct = max(c.peak_torque_pct, float((ap[fin] / lim[fin]).max() * 100))
        for a, r in self.limited:
            c.joint_overshoot = max(c.joint_overshoot, r[0] - d.qpos[a], d.qpos[a] - r[1])
        for i in range(d.ncon):
            con = d.contact[i]
            if (m.geom_bodyid[con.geom1] in self.bodies) != (m.geom_bodyid[con.geom2] in self.bodies):
                c.max_penetration_mm = max(c.max_penetration_mm, -con.dist * 1000)
        if self.report.base == "floating" and c.fell_at is None:
            if d.xmat[self.root][8] < 0.6 or d.subtree_com[self.root][2] < 0.6 * self.com_h0:
                c.fell_at = round(d.time, 3)

    def state(self):
        with self.lock:
            d, w = self.d, self.wbc
            ap = np.abs(self.motors.applied(d))
            lim = np.where(np.isfinite(self.motors.limit) & (self.motors.limit > 0), self.motors.limit, np.inf)
            return dict(
                time_s=round(d.time, 3), robot=self.name, robot_class=self.report.robot_class,
                hands=[dict(name=n, position=w.ee_pos(d, b).round(4).tolist(), goal=w.ee_goal[b].round(4).tolist(),
                            error_mm=round(float(np.linalg.norm(w.ee_pos(d, b) - w.ee_goal[b])) * 1000, 1),
                            grip=self.grip[b], has_fingers=bool(self.fingers[b]))
                       for n, b in zip(self.hand_names(), self.hands)],
                torque_now_pct=round(float((ap / lim).max() * 100), 1) if len(ap) else 0.0,
                walk=dict(available=self.loco is not None, mode=self.mode,
                          policy=self.loco.name if self.loco else None,
                          command=self.loco.command.round(2).tolist() if self.loco else [0, 0, 0],
                          measured=self.loco.measured.round(2).tolist() if self.loco else [0, 0, 0]),
                contacts=int(d.ncon), recording=self.recording is not None,
                fallen=self.checks.fell_at is not None,
                guard_active=bool(d.time < self.guard_until), guard_events=self.guard_events,
                checks=self.checks.as_dict())

    # ------------------------------------------------------------------ recording
    def start_recording(self):
        with self.lock:
            self.recording = Episode()
            self._rec_checks0 = Checks(**{k: getattr(self.checks, k) for k in Checks.__dataclass_fields__})

    def stop_recording(self, out_dir: Path) -> Path | None:
        with self.lock:
            ep, self.recording = self.recording, None
        if ep is None or not ep.t:
            return None
        out_dir = Path(out_dir) / time.strftime("episode_%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "episode.npz", **{k: np.asarray(v) for k, v in ep.__dict__.items()})
        meta = dict(robot=self.name, robot_class=self.report.robot_class, timestep_s=self.m.opt.timestep,
                    record_every_steps=self.record_every, hands=self.hand_names(),
                    joints=[mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(self.m.njnt)],
                    motors=[mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in self.motors.ids],
                    motor_limits=[None if not np.isfinite(x) else float(x) for x in self.motors.limit],
                    locomotion_policy=self.loco.name if self.loco is not None else None,
                    physics_checks=self.checks.as_dict(),
                    faithful=self.checks.torque_limit_violations == 0 and self.checks.fell_at is None,
                    note="torques are the forces the motors actually applied (after MuJoCo's limit clamp)")
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        return out_dir

    def _record(self):
        d, w, ep = self.d, self.wbc, self.recording
        ep.t.append(d.time)
        ep.qpos.append(d.qpos.copy())
        ep.qvel.append(d.qvel.copy())
        ep.torque.append(self.motors.applied(d).copy())
        ep.hand_goal.append(np.array([w.ee_goal[b] for b in self.hands]).reshape(-1))
        ep.hand_pos.append(np.array([w.ee_pos(d, b) for b in self.hands]).reshape(-1))
        ep.grip.append(np.array([self.grip[b] for b in self.hands]))
        ep.walk_cmd.append(self.loco.command.copy() if self.loco is not None else np.zeros(3))
        ep.walking.append(self.mode == "walk")

    # ------------------------------------------------------------------ helpers
    def _support_contacts(self):
        m, d = self.m, self.d
        return sum(1 for i in range(d.ncon)
                   if (m.geom_bodyid[d.contact[i].geom1] in self.support or m.geom_bodyid[d.contact[i].geom2] in self.support)
                   and d.contact[i].dist < 0.002)

    def _balance_guard(self):
        """Operator safety layer for robots that balance: when the feet start to lose the ground or
        the centre of mass starts to run, the hands stop where they are and commands pause for
        0.5 s so the balance controller can recover (an operator cannot feel the margin)."""
        if not self.wbc.balance or not self.wbc.feet or self.checks.fell_at is not None:
            return
        mujoco.mj_subtreeVel(self.m, self.d)
        speed = float(np.linalg.norm(self.d.subtree_linvel[self.root][:2]))
        whip = bool(self.arm_dofs) and float(np.abs(self.d.qvel[self.arm_dofs]).max()) > 4.0
        if self._support_contacts() < 0.6 * self.nominal_support or speed > 0.15 or whip:
            for b in self.hands:
                self.wbc.set_ee_target(b, self.wbc.ee_pos(self.d, b))
            if self.d.time >= self.guard_until:
                self.guard_events += 1
            self.guard_until = self.d.time + 0.5

    def _shoulder(self, hand):
        """First body of the arm: walking up from the hand, the child of the body where the arm
        branches off the rest of the robot."""
        m = self.m
        jointed = lambda b: any(m.body_jntnum[x] > 0 for x in subtree(m, b))  # noqa: E731
        x = hand
        while x not in robot_roots(m):
            p = int(m.body_parentid[x])
            if sum(1 for c in range(m.nbody) if m.body_parentid[c] == p and jointed(c)) > 1:
                return x
            x = p
        return x

    def _arm_bodies(self, hand):
        """Bodies from the shoulder down to the hand (not the fingers)."""
        out, x = [], hand
        while True:
            out.append(x)
            if x == self.shoulder[hand] or x in robot_roots(self.m):
                return out
            x = int(self.m.body_parentid[x])

    def _arm_length(self, hand):
        """Shoulder-to-tool-point distance with every joint of the arm at its straightest."""
        m = self.m
        length, x = float(np.linalg.norm(self.wbc.ee_off[hand])), hand
        while x != self.shoulder[hand]:
            length += float(np.linalg.norm(m.body_pos[x]))
            x = int(m.body_parentid[x])
        return max(length, 0.05)

    def _own_body_normals(self, hand, margin=0.03):
        """Unit directions that move this hand away from parts of the robot it is touching or is
        within `margin` of (hands slow down before they reach the body, not after the impact)."""
        m, d = self.m, self.d
        out = []
        fromto = np.zeros(6)
        own = set(subtree(m, hand))
        for g1 in self.hand_geoms[hand]:
            # the link the hand is mounted on sits next to the body by construction (at the
            # shoulder): it counts only when actually touching
            reach_margin = margin if int(m.geom_bodyid[g1]) in own else 0.0
            for g2 in self.body_geoms[hand]:
                reach = m.geom_rbound[g1] + m.geom_rbound[g2] + reach_margin
                if np.sum((d.geom_xpos[g1] - d.geom_xpos[g2]) ** 2) > reach * reach:
                    continue
                dist = mujoco.mj_geomDistance(m, d, g1, g2, max(reach_margin, 1e-4), fromto)
                if dist >= reach_margin:
                    continue
                n = fromto[:3] - fromto[3:]            # from the body's point to the hand's point
                norm = np.linalg.norm(n)
                if norm > 1e-9:
                    out.append(n / norm if dist > 0 else -n / norm)
        return out

    def _finger_joints(self, hand):
        """Driven joints below a hand (its fingers / gripper), with the direction that closes them:
        the end of the range that brings the joint's body nearer the hand's tool point."""
        m, d = self.m, self.d
        out = []
        tool = self.wbc.ee_pos(d, hand)
        driven = set(int(self.motors.dof[k]) for k in self.motors.joint_idx)
        for b in subtree(m, hand):
            if b == hand:
                continue
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                if int(m.jnt_dofadr[j]) not in driven or not m.jnt_limited[j]:
                    continue
                a = int(m.jnt_qposadr[j])
                lo, hi = (float(x) for x in m.jnt_range[j])
                dd = mujoco.MjData(m)
                dd.qpos[:] = d.qpos
                from .analysis import collision_geoms
                # A terminal finger's body origin is at its hinge and does not
                # move when it bends. Measure its geometry instead; comparing
                # origins silently classified every distal hinge as closes-low.
                tips = [g for c in subtree(m, b) for g in collision_geoms(m, c)]
                if not tips:
                    continue
                dist = []
                for q in (lo, hi):
                    dd.qpos[a] = q
                    mujoco.mj_kinematics(m, dd)
                    dist.append(np.mean([np.linalg.norm(dd.geom_xpos[g] - tool) for g in tips]))
                out.append((a, lo, hi, dist[1] < dist[0]))
        return out
