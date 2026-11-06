"""Whole-body motion primitives for an imported robot: stances, weight shifts, steps, strikes,
kicks. Generic over robots (feet, hands and joints come from the analysis report).

Every primitive works the way a real humanoid's motion stack does: a kinematic planner
solves a reference pose (feet where they should be, hips at the requested height, hands at
their targets) on a scratch copy of the model; that pose becomes the whole-body controller's
references (centre of mass, feet, hands, posture). The controller then computes motor torques
from the full dynamics, within the motor limits, and MuJoCo integrates them. The live robot
state is never written: if the robot cannot physically reach a pose, it does not get there,
and the primitive reports what it actually achieved.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .analysis import robot_roots
from .control import quat_err
from .stance import foot_contact_points

# np.int32 `in` a tuple of mujoco enums is silently False: compare plain ints
_SCALAR = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))


def _yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


class Moves:
    """Motion primitives on a TeleopSession in stand mode. `on_tick(session)` runs every
    control tick (rendering hooks). All methods return measured outcomes."""

    TICK = 0.02
    # Direct pose tracking (WholeBodyController.pose_weight): strong while sinking from straight
    # knees, where the balance task alone cannot see the bend; light otherwise, so a nearly
    # unloaded foot is held by its contact task instead of being dragged by the leg posture.
    POSE_WEIGHT, SINK_POSE_WEIGHT = 3.0, 8.0

    def __init__(self, session, on_tick=None):
        self.s, self.on_tick = session, on_tick
        m = session.m
        session.operator_guard = False          # the program lifts feet on purpose
        session.wbc.pose_weight = self.POSE_WEIGHT  # follow planned poses (see WholeBodyController)
        # programmed hand motion: track the moving reference (fed forward), turn orientation
        # at a bounded rate, and let a blocked hand pull with bounded effort only
        session.wbc.ee_feedforward = True
        session.wbc.ee_rot_wmax = 3.0
        session.wbc.ee_leash = 0.06
        self.w = session.wbc
        self.root = robot_roots(m)[0]
        roots = robot_roots(m)
        self.feet = list(self.w.feet)
        self.hands = list(session.hands)
        self.foot_dofs = {f: self._chain(f, roots) for f in self.feet}
        # a hand is placed with its own arm (shoulder to wrist): letting the planner bend the
        # waist to reach leans the whole torso and moves the centre of mass with it
        self.hand_dofs = {h: sorted(int(m.jnt_dofadr[j]) for b in session._arm_bodies(h)
                                    for j in range(m.body_jntadr[b],
                                                   m.body_jntadr[b] + m.body_jntnum[b]))
                          for h in self.hands}
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and int(m.jnt_bodyid[j]) == self.root]
        if not free:
            raise ValueError("moves need a floating-base robot")
        self.qa = int(m.jnt_qposadr[free[0]])
        self.scratch = mujoco.MjData(m)
        self.robot_q = np.array([int(m.jnt_qposadr[j]) for j in range(m.njnt)
                                 if int(m.jnt_bodyid[j]) in session.bodies
                                 and int(m.jnt_type[j]) in _SCALAR], int)
        self.arm_cap = dict(session.wbc.speed_cap)
        # the chest: the body both arms branch from. Held upright relative to the base only
        # while the arms move (so hand tasks use the arms, not the waist); in stances it is
        # left to the posture, as a held chest fights the hips and rocks the feet
        chests = {int(m.body_parentid[session.shoulder[h]]) for h in self.hands}
        if len(chests) == 1 and self.root not in chests:
            chest = chests.pop()
            rel = np.zeros(4)
            inv = session.d.xquat[self.root].copy()
            inv[1:] *= -1
            mujoco.mju_mulQuat(rel, inv, session.d.xquat[chest])
            self.chest, self.w.torso_rel = chest, rel
        else:
            self.chest = None
        self.hand_limits = (self.w.ref_vmax, self.w.ref_amax)
        self.log = []

    # ------------------------------------------------------------------ helpers
    def _chain(self, body, roots):
        """dof indices between the floating base and `body` (excluding the base)."""
        m, out, x = self.s.m, [], body
        while x not in roots and x > 0:
            for j in range(m.body_jntadr[x], m.body_jntadr[x] + m.body_jntnum[x]):
                if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                    out.append(int(m.jnt_dofadr[j]))
            x = int(m.body_parentid[x])
        return sorted(out)

    def _tick(self, seconds):
        n = max(1, int(round(self.TICK / self.s.m.opt.timestep)))
        for _ in range(max(1, int(round(seconds / self.TICK)))):
            self.s.step(n)
            if self.on_tick is not None:
                self.on_tick(self.s)
            if self.s.checks.fell_at is not None:
                raise RuntimeError(f"robot fell at {self.s.checks.fell_at} s")

    def foot_center(self, f, d=None):
        d = self.s.d if d is None else d
        bodies, local, _ = foot_contact_points(self.s.m, d, f)
        pts = [d.xmat[b].reshape(3, 3) @ p + d.xpos[b] for b, p in zip(bodies, local, strict=True)]
        return np.mean(pts, axis=0)

    def heading(self):
        q = self.s.d.xquat[self.root]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def frame(self):
        """(origin between the feet on the floor, forward unit, left unit) at the base heading."""
        c = np.mean([self.foot_center(f) for f in self.feet], axis=0)
        yaw = self.heading()
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        return c, fwd, left

    # ------------------------------------------------------------------ reference pose planner
    def plan_pose(self, root_pos, root_quat, feet=None, hands=None, iters=120):
        """Kinematic reference pose on a scratch copy: base at (root_pos, root_quat), each foot
        body at feet[f] = (pos, quat), each hand tool point at hands[h] = pos. Damped least
        squares over the leg / arm chains, pulled towards the current posture. Returns
        (qpos, residual metres)."""
        m, d = self.s.m, self.scratch
        d.qpos[:] = self.s.d.qpos
        d.qpos[self.qa:self.qa + 3] = root_pos
        d.qpos[self.qa + 3:self.qa + 7] = root_quat
        feet = feet or {}
        hands = hands or {}
        q_rest = self.s.d.qpos.copy()
        # seed the leg chains a little towards mid-range: a joint sitting at a straight singularity
        # (a fully straight knee) has no first-order effect on the error, so least squares
        # would never bend it
        chain = {dof for f in feet for dof in self.foot_dofs[f]}      # legs only: arms keep
        for dof in chain:                                             # their wrists neutral
            j = int(m.dof_jntid[dof])
            if m.jnt_limited[j]:
                a = int(m.jnt_qposadr[j])
                d.qpos[a] += 0.1 * (m.jnt_range[j].mean() - d.qpos[a])
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        res = 0.0
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rows, errs = [], []
            for f, (p, q) in feet.items():
                mujoco.mj_jacBody(m, d, jacp, jacr, f)
                cols = self.foot_dofs[f]
                J = np.zeros((6, m.nv))
                J[:3, cols], J[3:, cols] = jacp[:, cols], jacr[:, cols]
                rows.append(J)
                errs.append(np.r_[p - d.xpos[f], quat_err(d.xquat[f], q)])
            for h, p in hands.items():
                tool = self.w.ee_pos(d, h)
                mujoco.mj_jac(m, d, jacp, None, tool, h)
                cols = self.hand_dofs[h]
                J = np.zeros((3, m.nv))
                J[:, cols] = jacp[:, cols]
                rows.append(J)
                errs.append(p - tool)
            if not rows:
                break
            J, e = np.vstack(rows), np.concatenate(errs)
            res = float(np.abs(e).max())
            if res < 1e-4:
                break
            lam = 1e-2
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(len(e)), e)
            # pull unused freedom back towards the current posture (null space)
            rest = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, rest, 1.0, d.qpos, q_rest)
            rest[:6] = 0
            N = np.eye(m.nv) - J.T @ np.linalg.solve(J @ J.T + lam * np.eye(len(e)), J)
            dq += 0.05 * N @ rest
            dq[:6] = 0
            mujoco.mj_integratePos(m, d.qpos, dq, 1.0)
            for j in range(m.njnt):                     # respect joint ranges
                if m.jnt_limited[j] and int(m.jnt_type[j]) in _SCALAR:
                    a = m.jnt_qposadr[j]
                    d.qpos[a] = np.clip(d.qpos[a], *m.jnt_range[j])
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        return d.qpos.copy(), res

    def plan_balanced(self, com_xy, feet, *, root_z=None, root_quat=None, hands=None, iters=4):
        """plan_pose, with the base shifted horizontally until the planned centre of mass sits
        over `com_xy` (the hips move over the stance foot, like a person shifting weight)."""
        d = self.s.d
        root = d.xpos[self.root].copy()
        if root_z is not None:
            root[2] = root_z
        quat = d.xquat[self.root].copy() if root_quat is None else root_quat
        q, res = None, 0.0
        for _ in range(iters):
            q, res = self.plan_pose(root, quat, feet=feet, hands=hands)
            err = np.asarray(com_xy[:2]) - self.scratch.subtree_com[self.root][:2]
            root[:2] += err
            if np.linalg.norm(err) < 2e-3:
                break
        return q, res

    def _adopt(self, qpos, com_xy=None):
        """Make a planned pose the controller's reference: posture, centre of mass (height from
        the plan, ground position from `com_xy` or the plan) and every hand, which moves with
        the planned body (its target and orientation come from the plan)."""
        d = self.scratch
        w = self.w
        w.qpos0[self.robot_q] = qpos[self.robot_q]
        com = d.subtree_com[self.root].copy()
        w.set_com_target(xy=com[:2] if com_xy is None else com_xy, z=com[2])
        for h in self.hands:
            w.set_ee_target(h, self.w.ee_pos(d, h), quat=d.xquat[h].copy())

    def _com_vel(self):
        mujoco.mj_subtreeVel(self.s.m, self.s.d)
        return self.s.d.subtree_linvel[self.root].copy()

    def _wait_com(self, tol=0.015, vtol=0.03, timeout=4.0):
        """Wait until the centre of mass is at its goal and nearly still (a foot is only
        released from a quiet, centred stance)."""
        s, w = self.s, self.w
        t0 = s.d.time
        while s.d.time - t0 < timeout:
            e = np.linalg.norm(s.d.subtree_com[self.root][:2] - w.com_goal[:2])
            if (e < tol and np.linalg.norm(self._com_vel()[:2]) < vtol
                    and np.allclose(w.com_ref, w.com_goal, atol=1e-3)):
                return float(e)
            self._tick(self.TICK)
        return float(np.linalg.norm(s.d.subtree_com[self.root][:2] - w.com_goal[:2]))

    def _wait_foot(self, foot, tol=0.02, timeout=3.0):
        """Until the foot reference has arrived and the real foot is within `tol` of it."""
        s, w = self.s, self.w
        t0 = s.d.time
        while s.d.time - t0 < timeout:
            if w.foot_settled(foot) and np.linalg.norm(s.d.xpos[foot] - w.foot_goal[foot]) < tol:
                break
            self._tick(self.TICK)

    def _swing(self, foot, waypoints, stance_xy, root_z, wait=3.0):
        """Move a lifted foot through world waypoints (pos, quat), replanning the balanced pose
        for each so the stance leg and hips take the matching posture."""
        w = self.w
        for pos in waypoints:
            feet = self._planted()
            feet[foot] = (np.asarray(pos, float), w.foot_quat0[foot].copy())
            q, _ = self.plan_balanced(stance_xy, feet, root_z=root_z)
            self._adopt(q, com_xy=stance_xy)
            w.set_foot_target(foot, pos)
            self._wait_foot(foot, timeout=wait)

    # ------------------------------------------------------------------ primitives
    def stance(self, drop, *, settle=1.0):
        """Sink the hips `drop` metres (relative to now) with both feet where they are: a horse
        stance once the feet are wide. Arms keep their pose relative to the torso."""
        d, w = self.s.d, self.w
        z0 = float(d.xpos[self.root][2])
        # sink straight down over the ground point the weight is on now: moving the centre of
        # mass sideways or forwards at the same time rocks the feet onto their edges
        here = d.subtree_com[self.root][:2].copy()
        q, res = self.plan_balanced(here, self._planted(), root_z=z0 - drop)
        self._adopt(q, com_xy=here)
        w.com_vmax, w.com_amax = 0.06, 0.4
        w.pose_weight = self.SINK_POSE_WEIGHT
        self._wait_com(tol=0.015)
        self._tick(settle)
        w.pose_weight = self.POSE_WEIGHT
        got = z0 - float(d.xpos[self.root][2])
        out = dict(move="stance", drop_m=drop, achieved_m=round(got, 3),
                   plan_residual_m=round(res, 4))
        self.log.append(out)
        return out

    def shift_weight(self, onto=None, *, ratio=0.9, timeout=5.0, vmax=0.12, amax=0.5):
        """Move the centre of mass `ratio` of the way from between the feet to the centre of
        foot `onto` (or back between the feet). Aiming a little inside the foot's centre keeps
        a margin against overshoot: past the centre the robot rolls over the foot's outer edge.
        Returns the result with the ground point used ("com_xy")."""
        d, w = self.s.d, self.w
        mid = self._mid()
        xy = mid if onto is None else mid + ratio * (self.foot_center(onto)[:2] - mid)
        q, _ = self.plan_balanced(xy, self._planted(), root_z=float(d.xpos[self.root][2]))
        self._adopt(q, com_xy=xy)
        w.com_vmax, w.com_amax = vmax, amax
        err = self._wait_com(timeout=timeout)
        out = dict(move="shift_weight", onto=None if onto is None else self.s.m.body(onto).name,
                   com_error_m=round(err, 3), com_xy=xy.round(4).tolist())
        self.log.append(out)
        return out

    def step(self, foot, delta, *, clearance=0.05, ratio=1.0, speed=0.25, accel=1.5,
             shift_speed=0.12, shift_accel=0.5, wait=3.0):
        """Lift `foot`, move it by `delta` (world xy metres), set it down flat, take weight back
        between the feet. Real single support: the foot's contacts stop being held while it is
        in the air, and the robot balances on the other foot alone. A sideways swing carries the
        leg's mass outwards, so the weight goes fully over the stance foot first (`ratio`).
        `speed`/`accel` bound the swing; `wait` bounds each phase (a quick step does not wait
        for the robot to come fully to rest)."""
        w, d = self.w, self.s.d
        stance_foot = next(f for f in self.feet if f != foot)
        stance_xy = np.asarray(self.shift_weight(stance_foot, ratio=ratio, vmax=shift_speed,
                                                 amax=shift_accel,
                                                 timeout=max(wait, 0.8))["com_xy"])
        z = float(d.xpos[self.root][2])
        start = w.foot_goal[foot].copy()
        end = start + np.r_[np.asarray(delta, float)[:2], 0.0]
        w.set_support(foot, False)
        w.foot_vmax, w.foot_amax = speed, accel
        self._swing(foot, [start + [0, 0, clearance], end + [0, 0, clearance], end], stance_xy, z,
                    wait=wait)
        self._touchdown(foot)
        w.foot_vmax, w.foot_amax = 0.6, 3.0
        self._tick(0.1)
        self.shift_weight(None, vmax=shift_speed, amax=shift_accel, timeout=max(wait, 0.8))
        landed = float(np.linalg.norm(d.xpos[foot][:2] - end[:2]))
        out = dict(move="step", foot=self.s.m.body(foot).name, delta_m=list(np.round(delta, 3)),
                   landing_error_m=round(landed, 3))
        self.log.append(out)
        return out

    def strike(self, hand, target, *, hold=0.25, speed=2.0, accel=14.0, retract_to=None):
        """Punch / palm strike: drive the hand's tool point to `target` at striking speed, hold,
        then return to `retract_to` (or where it started). Speed and acceleration are reference
        limits; the motors' torque limits decide what the arm really does."""
        d = self.s.d
        start = self.w.ee_pos(d, hand).copy()
        self.w.torso_body = self.chest
        self._set_hand_speed(True, speed, accel)
        q, res = self.plan_pose(d.xpos[self.root].copy(), d.xquat[self.root].copy(),
                                feet=self._planted(), hands={hand: np.asarray(target, float)})
        self._adopt_arm(q, hand)
        t0, best = d.time, np.inf
        peak_speed = 0.0
        while d.time - t0 < 1.2:
            prev = self.w.ee_pos(d, hand).copy()
            self._tick(self.TICK)
            now = self.w.ee_pos(d, hand)
            peak_speed = max(peak_speed, float(np.linalg.norm(now - prev) / self.TICK))
            best = min(best, float(np.linalg.norm(now - target)))
            if best < 0.03 and np.linalg.norm(self.w.ee_ref_vel[hand]) < 1e-3:
                break
        self._tick(hold)
        back = start if retract_to is None else np.asarray(retract_to, float)
        q, _ = self.plan_pose(d.xpos[self.root].copy(), d.xquat[self.root].copy(),
                              feet=self._planted(), hands={hand: back})
        self._adopt_arm(q, hand)
        self._tick(0.5)
        self._set_hand_speed(False)
        self.w.torso_body = None
        out = dict(move="strike", hand=self.s.m.body(hand).name, miss_m=round(best, 3),
                   peak_hand_speed_ms=round(peak_speed, 2), plan_residual_m=round(res, 4))
        self.log.append(out)
        return out

    def hands_to(self, targets, *, speed=0.6, accel=4.0, timeout=2.5):
        """Move several hands together (a guard, chambered fists, a double palm strike):
        `targets` maps hand body id -> world point for its tool point."""
        d = self.s.d
        fast = speed > self.hand_limits[0]
        self._set_hand_speed(True, speed, accel) if fast else self._set_hand_speed(False)
        self.w.torso_body = self.chest
        q, res = self.plan_pose(d.xpos[self.root].copy(), d.xquat[self.root].copy(),
                                feet=self._planted(),
                                hands={h: np.asarray(p, float) for h, p in targets.items()})
        for h in targets:
            self._adopt_arm(q, h)
        t0 = d.time
        while d.time - t0 < timeout:
            self._tick(self.TICK)
            if all(np.linalg.norm(self.w.ee_pos(d, h) - np.asarray(p)) < 0.03
                   for h, p in targets.items()):
                break
        miss = max(float(np.linalg.norm(self.w.ee_pos(d, h) - np.asarray(p)))
                   for h, p in targets.items())
        self._set_hand_speed(False)
        self.w.torso_body = None
        out = dict(move="hands_to", hands=len(targets), miss_m=round(miss, 3),
                   plan_residual_m=round(res, 4))
        self.log.append(out)
        return out

    def combo(self, strikes, *, speed=2.5, accel=15.0, out_time=0.35, back_time=0.12,
              yaw_rate=2.0):
        """A chain of strikes without stopping: each entry is (hand, target, retract_to, yaw).
        The hand snaps to `target` (until it arrives or `out_time` passes), then is sent back
        to `retract_to` while the next strike already starts; `yaw` turns the hips into the
        strike (radians about vertical, relative to the stance heading), at most `yaw_rate`
        rad/s so the twist never wrenches the feet loose. Reference speed and acceleration are
        limits; the motors' torque limits decide what the arms really do."""
        d, w = self.s.d, self.w
        base0 = w.base_quat0.copy()
        yaw_now = [0.0]

        def tick_towards(yaw_goal, seconds, until=None):
            t0 = d.time
            while d.time - t0 < seconds:
                step = np.clip(yaw_goal - yaw_now[0], -yaw_rate * self.TICK, yaw_rate * self.TICK)
                yaw_now[0] += step
                w.base_quat0 = _mul(_yaw_quat(yaw_now[0]), base0)
                self._tick(self.TICK)
                if until is not None and until():
                    break

        w.torso_body = self.chest
        self._set_hand_speed(True, speed, accel)
        results = []
        for hand, target, retract, yaw in strikes:
            target = np.asarray(target, float)
            q, _ = self.plan_pose(d.xpos[self.root].copy(), _mul(_yaw_quat(yaw), base0),
                                  feet=self._planted(), hands={hand: target})
            self._adopt_arm(q, hand)
            t0 = d.time
            track = dict(best=np.inf, peak=0.0, prev=w.ee_pos(d, hand).copy())

            def arrived(hand=hand, target=target, track=track):
                now = w.ee_pos(d, hand)
                track["peak"] = max(track["peak"],
                                    float(np.linalg.norm(now - track["prev"]) / self.TICK))
                track["prev"] = now.copy()
                track["best"] = min(track["best"], float(np.linalg.norm(now - target)))
                return track["best"] < 0.04

            tick_towards(yaw, out_time, until=arrived)
            q, _ = self.plan_pose(d.xpos[self.root].copy(), w.base_quat0.copy(),
                                  feet=self._planted(), hands={hand: np.asarray(retract, float)})
            self._adopt_arm(q, hand)
            tick_towards(yaw, back_time)
            results.append(dict(hand=self.s.m.body(hand).name, miss_m=round(track["best"], 3),
                                peak_hand_speed_ms=round(track["peak"], 2),
                                time_s=round(d.time - t0, 2)))
        tick_towards(0.0, 0.4)
        w.base_quat0 = base0
        self._set_hand_speed(False)
        w.torso_body = None
        self._resettle_feet()
        out = dict(move="combo", strikes=results)
        self.log.append(out)
        return out

    def _resettle_feet(self):
        """Planted feet are anchored where they now are (fast work can creep them a few mm;
        pulling them back would fight their own contacts)."""
        for f in self.feet:
            if f in self.w.support_bodies:
                self._touchdown(f)

    def raise_foot(self, foot, waypoints, *, hold=1.0, speed=0.4, accel=3.0,
                   shift_speed=0.12, shift_accel=0.5, ratio=0.9, wait=3.0):
        """Single-leg balance: weight onto the other foot, carry `foot` through `waypoints`
        (offsets from its planted position, world metres), hold at the last one, bring it back
        and set it down. A knee raise, a slow kick or (with high `speed`/`accel`, a short
        `hold` and a short per-waypoint `wait`) a snap kick, depending on the waypoints."""
        w, d = self.w, self.s.d
        stance_foot = next(f for f in self.feet if f != foot)
        stance_xy = np.asarray(self.shift_weight(stance_foot, ratio=ratio, vmax=shift_speed,
                                                 amax=shift_accel,
                                                 timeout=max(wait, 0.8))["com_xy"])
        z = float(d.xpos[self.root][2])
        home = w.foot_goal[foot].copy()
        path = [home + [0, 0, 0.04]] + [home + np.asarray(r, float) for r in waypoints]
        w.set_support(foot, False)
        w.foot_vmax, w.foot_amax = speed, accel
        self._swing(foot, path, stance_xy, z, wait=wait)
        self._tick(hold)
        reached = float(np.linalg.norm(d.xpos[foot] - path[-1]))
        height = float(d.xpos[foot][2] - home[2])
        self._swing(foot, list(reversed(path[:-1])) + [home], stance_xy, z, wait=max(wait, 0.6))
        self._touchdown(foot)
        w.foot_vmax, w.foot_amax = 0.6, 3.0
        self._tick(0.1)
        self.shift_weight(None, vmax=shift_speed, amax=shift_accel)
        out = dict(move="raise_foot", foot=self.s.m.body(foot).name,
                   target_rel_m=list(np.round(waypoints[-1], 3)), miss_m=round(reached, 3),
                   foot_height_m=round(height, 3))
        self.log.append(out)
        return out

    def turn_torso(self, yaw, *, seconds=0.8):
        """Turn the base about the vertical by `yaw` radians relative to the feet heading,
        feet planted (the hips' yaw joints do it)."""
        w = self.w
        q0 = w.base_quat0.copy()
        target = _mul(_yaw_quat(yaw), q0)
        n = max(1, int(seconds / self.TICK))
        for k in range(1, n + 1):
            a = 0.5 - 0.5 * math.cos(math.pi * k / n)
            q = (1 - a) * q0 + a * target       # normalised lerp: fine for turns under 90 deg
            w.base_quat0 = q / np.linalg.norm(q)
            self._tick(self.TICK)
        return dict(move="turn_torso", yaw_rad=yaw)

    def hold(self, seconds):
        self._tick(seconds)

    # ------------------------------------------------------------------ internals
    def _touchdown(self, foot):
        """The foot is down: it carries weight again and is anchored where it actually landed
        horizontally (holding it to the planned spot would fight its own ground contact)."""
        w, d = self.w, self.s.d
        landed = w.foot_goal[foot].copy()                 # planned floor height, real x/y
        landed[:2] = d.xpos[foot][:2]
        w.foot_goal[foot] = landed
        w.foot_anchor[foot] = landed.copy()
        w.foot_ref_vel[foot] = np.zeros(3)
        w.set_support(foot, True)

    def _mid(self):
        return np.mean([self.foot_center(f) for f in self.feet], axis=0)[:2]

    def _planted(self):
        return {f: (self.w.foot_goal[f].copy(), self.w.foot_quat0[f].copy()) for f in self.feet}

    def _adopt_arm(self, qpos, hand):
        d = self.scratch
        for dof in self.hand_dofs[hand]:
            j = int(self.s.m.dof_jntid[dof])
            a = int(self.s.m.jnt_qposadr[j])
            self.w.qpos0[a] = qpos[a]
        self.w.set_ee_target(hand, self.w.ee_pos(d, hand), quat=d.xquat[hand].copy())

    def _set_hand_speed(self, fast, speed=2.0, accel=14.0):
        w = self.w
        if fast:
            w.ref_vmax, w.ref_amax = speed, accel
            w.speed_cap = {dof: max(cap, 10.0) for dof, cap in self.arm_cap.items()}
        else:
            w.ref_vmax, w.ref_amax = self.hand_limits
            w.speed_cap = dict(self.arm_cap)
