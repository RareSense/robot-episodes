"""Motion layer for the tea programs: whole-body reaches, grips, holds, and a 60 fps state recorder.

Nothing here moves the robot directly. A reach plans a balanced whole-body pose on a scratch copy
(byor Moves.plan_balanced: planted feet, pelvis leaned as needed, the hand on its target, the
centre of mass over the feet) and then blends the whole-body controller's REFERENCES there
(pelvis orientation, centre of mass, posture, hand goal). The controller turns references into
torques for the torque-limited motors and MuJoCo integrates. The recorder only reads the state.
"""
from __future__ import annotations
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401

import math
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):   # before numpy loads
    os.environ.setdefault(_v, "1")

import mujoco  # noqa: E402
import numpy as np

sys.path.insert(0, str(HANDOFF / "walking"))
from nova_robotics.byor.moves import Moves  # noqa: E402

FRAME_STEPS = 10                                   # 600 Hz physics -> one 60 fps frame per 10 steps


def quat_from_axes(x, y, z):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.c_[x, y, z].flatten())
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def yaw_pitch_quat(yaw, pitch):
    """Base orientation: heading `yaw`, leaned forward by `pitch` (radians) about its own left axis."""
    qy = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    qp = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
    return quat_mul(qy, qp)


def slerp(a, b, u):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.dot(a, b) < 0:
        b = -b
    dot = float(np.clip(np.dot(a, b), -1, 1))
    if dot > 0.9995:
        q = a + u * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(dot)
    return (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)


def curve(p0, via, p1, u):
    """Point at parameter u on the path p0 -> p1: straight, or a quadratic Bezier bent towards `via`
    (passes through `via` at u = 0.5 when given as the curve's midpoint)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if via is None:
        return p0 + u * (p1 - p0)
    c = 2 * np.asarray(via, float) - 0.5 * (p0 + p1)       # control point so that B(0.5) = via
    return (1 - u) ** 2 * p0 + 2 * (1 - u) * u * c + u * u * p1


def smooth(u):
    """Minimum-jerk blend 0 -> 1 (zero velocity and acceleration at both ends)."""
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


def path(p0, via, p1, n=400):
    """The hand path p0 -> p1 (bent towards `via`) by arc length: path(s) is the point a fraction s
    of the way along it. Timing the fraction with `smooth` gives the hand one bell-shaped speed
    profile; timing the curve's own parameter speeds up and slows down where it bends."""
    us = np.linspace(0, 1, n)
    pts = np.array([curve(p0, via, p1, u) for u in us])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if s[-1] < 1e-9:
        return lambda f: pts[-1]
    s /= s[-1]
    return lambda f: curve(p0, via, p1, float(np.interp(f, s, us)))


class Recorder:
    """Session step observer: the full state at 60 fps, and per-motor load statistics."""

    def __init__(self, s):
        self.s = s
        self.t, self.qpos, self.tau = [], [], []
        self.peak = np.zeros(len(s.motors.ids))
        self.n = 0
        lim = s.motors.limit
        self.lim = np.where(np.isfinite(lim) & (lim > 0), lim, np.inf)

    def before_step(self, s):
        pass

    def before_integrate(self, s):
        pass

    def after_step(self, s):
        self.n += 1
        ap = np.abs(s.motors.applied(s.d))
        self.peak = np.maximum(self.peak, ap / self.lim)
        if self.n % FRAME_STEPS == 0:
            self.t.append(s.d.time)
            self.qpos.append(s.d.qpos.copy())
            self.tau.append(ap.astype(np.float32))

    def save(self, path, **extra):
        np.savez_compressed(path, t=np.array(self.t), qpos=np.array(self.qpos), tau=np.array(self.tau),
                            peak=self.peak, motor_names=np.array([self.s.m.actuator(i).name for i in self.s.motors.ids]),
                            limits=self.s.motors.limit, **extra)


class Program:
    def __init__(self, s, on_tick=None):
        self.s, self.m, self.d = s, s.m, s.d
        self.mv = Moves(s, on_tick=on_tick)
        self.w = s.wbc
        self.root = self.mv.root
        self.rec = Recorder(s)
        s.step_observer = self.rec
        self.log = []

    # ------------------------------------------------------------------ time
    def hold(self, seconds):
        self.mv._tick(seconds)

    def note(self, what, **kv):
        e = dict(t=round(self.d.time, 2), what=what, **kv)
        self.log.append(e)
        print(e, flush=True)

    # ------------------------------------------------------------------ reach
    def plan_reach(self, hand, target, leans=(0, 4, 8, 12, 16, 20, 24, 28), tol=0.008):
        """Smallest forward lean (degrees) at which the planned balanced pose puts the hand's tool
        point on `target`. Returns (qpos plan, lean, residual, base quat)."""
        mv = self.mv
        yaw = mv.heading()
        planted = mv._planted()
        support = np.mean([mv.foot_center(f) for f in mv.feet], axis=0)[:2]
        best = None
        for lean in leans:
            quat = yaw_pitch_quat(yaw, math.radians(lean))
            root_z = float(self.d.xpos[self.root][2]) - 0.02 * lean / 10   # hips sink a little as it leans
            q, res = mv.plan_balanced(support, planted, root_z=root_z, root_quat=quat,
                                      hands={self.s.hands[hand]: np.asarray(target, float)})
            if best is None or res < best[2]:
                best = (q, lean, res, quat)
            if res < tol:
                return q, lean, res, quat
        return best

    def reach(self, hand, target, quat=None, seconds=1.5, plan=None, settle=0.3, via=None):
        """Blend the whole body to the planned reach pose over `seconds` (minimum-jerk), the hand's
        goal to `target` (and orientation `quat`). With `via`, the hand's path is a smooth curve
        bending towards that point (e.g. up in front of the body, over a counter's edge, not
        through it). Returns the measured hand error (m)."""
        w, d, b = self.w, self.d, self.s.hands[hand]
        q_plan, lean, res, base_q = plan if plan is not None else self.plan_reach(hand, target)
        rq = self.mv.robot_q
        q0 = w.qpos0[rq].copy()
        bq0 = w.base_quat0.copy()
        com0 = w.com_goal.copy()
        sd = self.mv.scratch
        com1 = np.r_[np.mean([self.mv.foot_center(f) for f in self.mv.feet], axis=0)[:2], sd.subtree_com[self.root][2]]
        ee0 = w.ee_goal[b].copy()
        r0 = w.ee_rot[b].copy()
        # the other arm and every finger keep their own targets (fingers belong to the grip);
        # legs, waist and this arm blend to the plan
        fingers = [a for h in self.s.hands for a, *_ in self.s.fingers[h]]
        keep = np.isin(rq, self._other_arm_q(hand)) | np.isin(rq, fingers)
        # the moving arm follows its hand path: its joint posture only fills the spare freedom
        # (a joint-space blend towards the final pose would pull it off the curved path)
        go = path(ee0, via, target)
        n = max(1, int(round(seconds / self.mv.TICK)))
        for k in range(1, n + 1):
            u = smooth(k / n)
            self._ease_arm(hand, k * self.mv.TICK)
            qq = q0 + u * (q_plan[rq] - q0)
            qq[keep] = w.qpos0[rq][keep]
            w.qpos0[rq] = qq
            w.base_quat0 = slerp(bq0, base_q, u)
            w.set_com_target(xy=com0[:2] + u * (com1[:2] - com0[:2]), z=com0[2] + u * (com1[2] - com0[2]))
            w.set_ee_target(b, go(u), quat=slerp(r0, quat, u) if quat is not None else None)
            self.mv._tick(self.mv.TICK)
        self.hold(settle)
        err = float(np.linalg.norm(w.ee_pos(d, b) - np.asarray(target)))
        self.note("reach", hand=hand, lean_deg=lean, plan_residual_cm=round(res * 100, 2), hand_error_cm=round(err * 100, 2))
        return err

    def set_tool(self, hand, offset):
        """The hand point the controller places (hand frame): e.g. the pinch centre between thumb and
        fingertips for small things, or palm-side between palm and finger bases for a wrap around a
        handle. Targets are shifted so the wrist does not jump."""
        w, b, d = self.w, self.s.hands[hand], self.d
        offset = np.asarray(offset, float)
        shift = d.xmat[b].reshape(3, 3) @ (offset - w.ee_off[b])
        w.ee_off[b] = offset
        for targets in (w.ee_goal, w.ee_targets, self.s.home):
            targets[b] = targets[b] + shift

    EASE_S = 0.3                                   # posture hand-over time when an arm starts a hand task

    def _set_arm_weights(self):
        w = self.w
        P = w.posture_dofs
        pw = np.full(len(P), float(self.mv.POSE_WEIGHT))
        for h, wt in self._armw.items():
            pw[np.isin(P, self.mv.hand_dofs[self.s.hands[h]])] = wt
        w.pose_weight = pw

    def _ease_arm(self, hand, t):
        """Direct posture tracking of this hand's arm eases off over EASE_S as its hand task takes
        it over (a step in the controller's weights is a step in the torques, felt as a jolt)."""
        if not hasattr(self, "_armw"):
            self._armw = {h: float(self.mv.POSE_WEIGHT) for h in range(len(self.s.hands))}
        if self._armw[hand] <= 0.0:
            return
        if t <= self.mv.TICK * 1.5:
            self._ease0 = getattr(self, "_ease0", {})
            self._ease0[hand] = self._armw[hand]
        w0 = self._ease0.get(hand, float(self.mv.POSE_WEIGHT))
        self._armw[hand] = w0 * (1.0 - smooth(t / self.EASE_S))
        self._set_arm_weights()

    def _other_arm_q(self, hand):
        other = self.s.hands[1 - hand]
        return [int(self.m.jnt_qposadr[int(self.m.dof_jntid[dof])]) for dof in self.mv.hand_dofs[other]]

    def move_hand(self, hand, target, quat=None, seconds=1.0, via=None):
        """Hand only (body references unchanged), minimum-jerk path to `target` (curved with `via`)."""
        w, b = self.w, self.s.hands[hand]
        p0, r0 = w.ee_goal[b].copy(), w.ee_rot[b].copy()
        go = path(p0, via, target)
        n = max(1, int(round(seconds / self.mv.TICK)))
        for k in range(1, n + 1):
            u = smooth(k / n)
            self._ease_arm(hand, k * self.mv.TICK)
            w.set_ee_target(b, go(u), quat=slerp(r0, quat, u) if quat is not None else None)
            self.mv._tick(self.mv.TICK)
        return float(np.linalg.norm(w.ee_pos(self.d, b) - np.asarray(target)))

    def settle_hand(self, hand, target, tol=0.004, timeout=1.2):
        """Hold until the hand's tool point is within `tol` of `target` (precise grasp placement)."""
        b = self.s.hands[hand]
        t0 = self.d.time
        while self.d.time - t0 < timeout:
            if np.linalg.norm(self.w.ee_pos(self.d, b) - np.asarray(target)) < tol:
                break
            self.mv._tick(self.mv.TICK)
        return float(np.linalg.norm(self.w.ee_pos(self.d, b) - np.asarray(target)))

    def lower_until(self, hand, done, speed=0.03, max_dist=0.10, quat=None, ease_in=0.2, ease_out=0.15,
                    steer=None):
        """Hand straight down until done() turns true, e.g. the held object touching the counter
        (see move_until)."""
        return self.move_until(hand, (0.0, 0.0, -1.0), done, speed, max_dist, quat, ease_in, ease_out, steer)

    def move_until(self, hand, direction, done, speed=0.03, max_dist=0.10, quat=None, ease_in=0.2,
                   ease_out=0.15, steer=None):
        """Guarded move: the hand along `direction` at `speed` (easing in) until done() turns true,
        e.g. the palm touching a handle or a held object touching the counter, then easing to a
        stop. Speed changes smoothly, so the arm does not jolt; where it stops comes from the
        contact, not from a planned position. With `steer` (a function giving a measured xy error,
        e.g. of the held object), the hand also drifts to cancel that error on the way (time
        constant 0.4 s). Returns metres moved."""
        w, b, T = self.w, self.s.hands[hand], self.mv.TICK
        u = np.asarray(direction, float) / np.linalg.norm(direction)
        p = w.ee_goal[b].copy()
        t, dist, v, stop_t, v_stop = 0.0, 0.0, 0.0, None, 0.0
        while dist < max_dist:
            t += T
            if stop_t is None and done():
                stop_t, v_stop = 0.0, v
            if stop_t is None:
                v = speed * smooth(t / ease_in)
                if steer is not None:
                    dxy = np.asarray(steer(), float) * (T / 0.4)
                    n = float(np.linalg.norm(dxy))
                    p[:2] += dxy * min(1.0, speed * T / max(n, 1e-9))
            else:
                stop_t += T
                v = v_stop * (1.0 - smooth(stop_t / ease_out))
            p += u * v * T
            dist += v * T
            w.set_ee_target(b, p, quat=quat)
            self.mv._tick(T)
            if stop_t is not None and stop_t >= ease_out:
                break
        return dist

    def finger_pose(self, hand, pose, seconds=0.4):
        """Each finger to its own closure (0 open .. 1 closed), e.g. pointing: {"index": 0,
        "middle": 1, "thumb": 0.7}. The fingers' motors follow their targets, as in grip()."""
        b, m, w = self.s.hands[hand], self.m, self.w
        start, goal = {}, {}
        for a, lo, hi, high in self.s.fingers[b]:
            name = m.joint(int(np.nonzero(m.jnt_qposadr == a)[0][0])).name
            c = next(v for k, v in pose.items() if k in name)
            start[a], goal[a] = float(w.qpos0[a]), lo + (c if high else 1 - c) * (hi - lo)
        self.s.grip_active.add(b)
        n = max(1, int(round(seconds / self.mv.TICK)))
        for k in range(1, n + 1):
            u = smooth(k / n)
            for a in goal:
                w.qpos0[a] = start[a] + u * (goal[a] - start[a])
            self.mv._tick(self.mv.TICK)

    def grip(self, hand, value, seconds=0.5):
        """Fingers close (1) or open (0) through their motors; the grip ramps over `seconds`."""
        b = self.s.hands[hand]
        g0 = self.s.grip[b]
        n = max(1, int(round(seconds / self.mv.TICK)))
        for k in range(1, n + 1):
            self.s.set_grip(hand, g0 + smooth(k / n) * (value - g0))
            self.mv._tick(self.mv.TICK)
