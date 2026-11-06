"""Choreographed whole-body motion for an imported legged robot: fights, drills, dances.

A move script is written once and is robot-independent. It holds key poses for the hips, chest,
hands and feet in the robot's own proportions (hands in arm lengths from their shoulder, hips
and feet in leg lengths), plus the time of every foot lift and touchdown. `Compiler.compile`
turns it into a whole-body reference, the way position-controlled humanoids were made to dance
and fight before learned control:

1. A full-body inverse-kinematics pose every 10 ms on a scratch copy of the model.
2. A dynamics filter. From the reference's centroidal momentum it computes the zero-moment
   point, which is where the ground reaction must act. Wherever that point leaves the support
   polygon of the feet on the ground (shrunk by a margin), the filter shifts the hips by the
   solution of the cart-table equation over the whole timeline, then re-solves the poses. It
   repeats until every instant can be balanced by the feet that are actually down. The filter
   looks ahead, so the hips start moving before a fast strike or a kick needs them to.
3. Joint velocities and accelerations, plus the CoM, base, hand and foot trajectories.

`Tracker` makes that reference the whole-body controller's targets, with velocity and
acceleration fed forward. The controller computes motor torques from the full dynamics within
the motor limits, and MuJoCo integrates them. The plan is only a reference: the robot does what
the physics allows, and the episode records what that was.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial import ConvexHull, cKDTree

from .moves import _SCALAR, Moves, _mul
from .stance import foot_contact_points

GRAVITY = 9.81
SIDES = ("L", "R")


def _axis_quat(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), float(angle))
    return q


def _ypr_quat(yaw, pitch=0.0, roll=0.0):
    """Yaw about world z, then pitch about the yawed y, then roll about the result's x."""
    return _mul(_mul(_axis_quat([0, 0, 1], yaw), _axis_quat([0, 1, 0], pitch)),
                _axis_quat([1, 0, 0], roll))


def _inv(q):
    out = q.copy()
    out[1:] *= -1
    return out


# ----------------------------------------------------------------------------------- script
@dataclass
class _Key:
    t: float
    v: np.ndarray
    through: bool = False


@dataclass
class Script:
    """Key poses on named channels, in the robot's own proportions and its start heading.

    Channels, each as (forward, left, up) along the start heading unless noted:
      pelvis    offset of the base from where it starts, in leg lengths
      turn      (yaw, pitch, roll) of the base relative to its start, radians
      chest     (yaw, pitch) of the chest relative to the base, radians (robots with a waist;
                without one the yaw is added to the base)
      hand_L/R  tool point of the hand relative to its own shoulder, in arm lengths
      foot_L/R  (forward, left, up, yaw, pitch) of the foot relative to its start pose: leg
                lengths, radians

    Between keys each channel follows a quintic with zero acceleration at the keys. The
    velocity at a key is zero, unless the key is passed `through` (a hook's arc, a kick's
    chamber), where it is the Catmull-Rom tangent.
    """

    DIMS = dict(pelvis=3, turn=3, chest=2, hand_L=3, hand_R=3, foot_L=5, foot_R=5)
    keys: dict = field(default_factory=dict)
    supports: dict = field(default_factory=lambda: {s: [] for s in SIDES})
    labels: list = field(default_factory=list)
    # optional zero-phase Gaussian smoothing per channel (seconds): the hips and torso of a
    # fighter carry the strike through as a wave, not with the corners of the hand's keys
    smooth: dict = field(default_factory=dict)

    def key(self, channel, t, value, *, through=False):
        v = np.asarray(value, float)
        if v.shape != (self.DIMS[channel],):
            raise ValueError(f"{channel} takes {self.DIMS[channel]} values")
        ks = self.keys.setdefault(channel, [])
        if ks and t < ks[-1].t - 1e-9:
            raise ValueError(f"{channel}: keys must be in time order")
        ks.append(_Key(float(t), v, through))

    def last(self, channel, default=None):
        ks = self.keys.get(channel)
        return ks[-1].v.copy() if ks else default

    def move(self, channel, start, duration, value, *, through=False, default=None):
        """Hold the channel's last value until `start`, then reach `value` at start + duration."""
        prev = self.last(channel, default)
        ks = self.keys.get(channel)
        if prev is not None and (not ks or ks[-1].t < start - 1e-6):
            self.key(channel, start, prev)
        self.key(channel, start + duration, value, through=through)

    def lift(self, side, t):
        self.supports[side].append((float(t), False))

    def land(self, side, t):
        self.supports[side].append((float(t), True))

    def label(self, t, text):
        self.labels.append((float(t), text))

    @property
    def duration(self):
        ends = [ks[-1].t for ks in self.keys.values() if ks]
        ends += [e[0] for evs in self.supports.values() for e in evs]
        return max(ends, default=0.0)

    def sample(self, channel, times, start_value):
        """Channel values at `times` (a key at t=0 with `start_value` is implied)."""
        ks = [k for k in self.keys.get(channel, []) if k.t > 1e-9]
        ks = [_Key(0.0, np.asarray(start_value, float))] + ks
        vel = []
        for i, k in enumerate(ks):
            if k.through and 0 < i < len(ks) - 1:
                vel.append((ks[i + 1].v - ks[i - 1].v) / max(ks[i + 1].t - ks[i - 1].t, 1e-6))
            else:
                vel.append(np.zeros_like(k.v))
        sigma = self.smooth.get(channel, 0.0)
        if sigma <= 0 or len(times) < 2:
            return self._sample_raw(ks, vel, times, len(start_value))
        # smoothed in script time (on a uniform grid), whatever the sampling
        step = 0.005
        g = np.arange(0.0, float(np.max(times)) + 2 * step, step)
        raw = gaussian_filter1d(self._sample_raw(ks, vel, g, len(start_value)), sigma / step,
                                axis=0, mode="nearest")
        return np.stack([np.interp(times, g, raw[:, i]) for i in range(raw.shape[1])], 1)

    @staticmethod
    def _sample_raw(ks, vel, times, dim):
        out = np.zeros((len(times), dim))
        tk = np.array([k.t for k in ks])
        for n, t in enumerate(times):
            i = int(np.searchsorted(tk, t, side="right")) - 1
            if i >= len(ks) - 1:
                out[n] = ks[-1].v
                continue
            a, b = ks[i], ks[i + 1]
            h = b.t - a.t
            if h < 1e-9:
                out[n] = b.v
                continue
            s = (t - a.t) / h
            s3, s4, s5 = s ** 3, s ** 4, s ** 5
            h00 = 1 - 10 * s3 + 15 * s4 - 6 * s5
            h10 = s - 6 * s3 + 8 * s4 - 3 * s5
            h01 = 10 * s3 - 15 * s4 + 6 * s5
            h11 = -4 * s3 + 7 * s4 - 3 * s5
            out[n] = h00 * a.v + h10 * h * vel[i] + h01 * b.v + h11 * h * vel[i + 1]
        return out

    def support_at(self, side, times):
        on = np.ones(len(times), bool)
        for t, flag in sorted(self.supports[side]):
            on[times >= t - 1e-9] = flag
        return on


# ----------------------------------------------------------------------------------- plan
@dataclass
class Plan:
    """A compiled whole-body reference, sampled every `dt` seconds."""
    dt: float
    times: np.ndarray
    q: np.ndarray            # (N, nq)
    v: np.ndarray            # (N, nv)
    a: np.ndarray            # (N, nv)
    com: np.ndarray          # (N, 3) and its derivatives
    com_v: np.ndarray
    com_a: np.ndarray
    root_quat: np.ndarray    # (N, 4) and world angular velocity / acceleration
    root_w: np.ndarray
    root_dw: np.ndarray
    hands: dict              # body -> dict(pos, vel, acc, quat)
    feet: dict               # body -> dict(pos, vel, acc, quat, support)
    zmp: np.ndarray          # (N, 2) of the final reference
    zmp_margin: np.ndarray   # (N,) signed distance inside the shrunk support polygon (m)
    ik_residual: np.ndarray  # (N,) worst task error after IK (m or rad)
    hip_shift: np.ndarray    # (N, 2) total horizontal correction of the dynamics filter (m)
    twist_load: np.ndarray   # (N,) vertical moment needed / friction moment available
    slip_load: np.ndarray    # (N,) horizontal ground force needed / friction available
    labels: list
    report: dict
    motor_load: np.ndarray = None   # (N,) largest torque needed / motor limit (see compile)

    @property
    def duration(self):
        return float(self.times[-1])


# ----------------------------------------------------------------------------------- compiler
class Compiler:
    """Compiles a `Script` for the robot of a `Moves` instance (TeleopSession in stand mode)."""

    REACH = 0.85            # hand targets at most this share of the arm's length from its shoulder
    BASE_TILT_WEIGHT = 3.0  # IK weight of the planned base orientation (feet and hands: 1)

    def __init__(self, mv: Moves, *, dt=0.01, zmp_margin=0.7):
        s, m, d = mv.s, mv.s.m, mv.s.d
        self.mv, self.s, self.m = mv, s, m
        self.dt, self.margin = dt, zmp_margin
        root = mv.root
        self.root = root
        Rb = d.xmat[root].reshape(3, 3)
        side = lambda b: float((d.xpos[b] - d.xpos[root]) @ Rb[:, 1])  # noqa: E731
        self.hand = dict(zip(SIDES, sorted(mv.hands, key=side, reverse=True), strict=True))
        self.foot = dict(zip(SIDES, sorted(mv.feet, key=side, reverse=True), strict=True))
        self.heading = mv.heading()
        fwd = np.array([math.cos(self.heading), math.sin(self.heading), 0.0])
        left = np.array([-math.sin(self.heading), math.cos(self.heading), 0.0])
        self.axes = np.c_[fwd, left, [0.0, 0.0, 1.0]]
        q_head = _axis_quat([0, 0, 1], self.heading)
        self.root0 = d.xpos[root].copy()
        self.root_tilt0 = _mul(_inv(q_head), d.xquat[root])          # start pose minus heading
        self.foot0 = {k: d.xpos[f].copy() for k, f in self.foot.items()}
        self.foot_tilt0 = {k: _mul(_inv(q_head), d.xquat[f]) for k, f in self.foot.items()}
        self.leg = float(d.xpos[root][2] - np.mean([d.xpos[f][2] for f in mv.feet]))
        self.arm = {k: float(s._arm_length(h)) for k, h in self.hand.items()}
        self.mass = float(m.body_subtreemass[root])
        self.chest = mv.chest
        self.waist = mv._chain(mv.chest, [root]) if mv.chest is not None else []
        self.chest_rel0 = mv.w.torso_rel.copy() if mv.chest is not None else None
        # joints the plan moves: every driven robot joint except fingers (their grip is the
        # session's) - the base is placed, not solved
        fingers = {int(a) for h in mv.hands for a, *_ in s.fingers[h]}
        jnt_of_q = {int(m.jnt_qposadr[j]): j for j in range(m.njnt)}
        self.finger_dofs = sorted(int(m.jnt_dofadr[jnt_of_q[a]]) for a in fingers)
        self.ik_dofs = np.array(sorted(set(mv.w.posture_dofs.tolist()) - set(self.finger_dofs)),
                                int)
        self.track_q = np.array([a for a in mv.robot_q if a not in fingers], int)
        free = next(j for j in range(m.njnt) if int(m.jnt_bodyid[j]) == root
                    and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        self.root_rot = np.arange(3, 6) + int(m.jnt_dofadr[free])
        self.q_start = d.qpos.copy()
        # Planning range of every robot hinge / slider: its own range, or, for a joint the model
        # leaves unlimited, the side it is bent to at the start (a crouched knee stays a knee:
        # it must not fold the other way) or a full turn either way if it starts straight.
        self.range = {}
        for j in range(m.njnt):
            if int(m.jnt_type[j]) not in _SCALAR or int(m.jnt_bodyid[j]) not in s.bodies:
                continue
            q0 = float(self.q_start[int(m.jnt_qposadr[j])])
            if m.jnt_limited[j]:
                self.range[j] = (float(m.jnt_range[j][0]), float(m.jnt_range[j][1]))
            elif abs(q0) > 0.1:
                self.range[j] = (0.0, math.pi) if q0 > 0 else (-math.pi, 0.0)
            else:
                self.range[j] = (-math.pi, math.pi)
        # rest posture for unused freedom: the start pose, with the legs a little towards
        # mid-range (a straight knee is a singularity: least squares would never bend it)
        self.q_rest = d.qpos.copy()
        for f in mv.feet:
            for dof in mv.foot_dofs[f]:
                j = int(m.dof_jntid[dof])
                if j in self.range:
                    a = int(m.jnt_qposadr[j])
                    self.q_rest[a] += 0.1 * (np.mean(self.range[j]) - self.q_rest[a])
        self.limits = [(int(m.jnt_qposadr[j]), lo, hi) for j, (lo, hi) in self.range.items()]
        # A hand is placed with the first four joints of its arm (a shoulder and an elbow);
        # joints beyond them (wrists) only orient the hand and stay at their rest angle, so a
        # fist stays in line with the forearm. Letting a wrist help place the tool point bends
        # it into a claw.
        self.place_dofs = {h: self._proximal(h, 4) for h in mv.hands}
        # fallback IK seeds: the rest posture, and the same with the arms at mid-range
        mid = self.q_rest.copy()
        for h in mv.hands:
            for dof in self.place_dofs[h]:
                j = int(m.dof_jntid[dof])
                if j in self.range:
                    mid[int(m.jnt_qposadr[j])] = float(np.mean(self.range[j]))
        self.seeds = [self.q_rest, mid]
        self._sample_reach()
        self.sole = {k: foot_contact_points(m, d, f)[:2] for k, f in self.foot.items()}
        # ankle motors: hinge joints of the foot and the link above it, with their torque limits
        lim = {int(s.motors.dof[k]): float(s.motors.limit[k]) for k in range(len(s.motors.ids))
               if s.motors.dof[k] >= 0 and np.isfinite(s.motors.limit[k])}
        self.ankles = {f: [(j, lim[int(m.jnt_dofadr[j])]) for j in range(m.njnt)
                           if int(m.jnt_bodyid[j]) in (f, int(m.body_parentid[f]))
                           and int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
                           and int(m.jnt_dofadr[j]) in lim]
                       for f in mv.feet}
        self._ground = min(self._sole_z(d, f) for f in mv.feet)      # floor under the soles
        # friction the controller plans with (MuJoCo takes the larger of the two geoms')
        floor = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == 0]
        sole = [g for g in range(m.ngeom) for f in mv.feet if int(m.geom_bodyid[g]) == f]
        self.mu = 0.8 * max(float(m.geom_friction[floor + sole, 0].max()), 1e-3)
        self.scratch = mujoco.MjData(m)

    def _sample_reach(self, n=20000, seed=0):
        """Where each hand's tool point can be, relative to its shoulder in the frame of the
        body the arm hangs from (arm joints sampled over their ranges, wrists at rest)."""
        m, mv, s = self.m, self.mv, self.s
        d = mujoco.MjData(m)
        rng = np.random.default_rng(seed)
        self.reach = {}
        for h in mv.hands:
            pts = np.zeros((n, 3))
            parent = int(m.body_parentid[s.shoulder[h]])
            for i in range(n):
                d.qpos[:] = self.q_start
                for dof in self.place_dofs[h]:
                    j = int(m.dof_jntid[dof])
                    if j in self.range:
                        d.qpos[int(m.jnt_qposadr[j])] = rng.uniform(*self.range[j])
                mujoco.mj_kinematics(m, d)
                R = d.xmat[parent].reshape(3, 3)
                pts[i] = R.T @ (mv.w.ee_pos(d, h) - d.xpos[s.shoulder[h]])
            self.reach[h] = (cKDTree(pts), pts, parent)
        # each foot relative to the base, in the base frame (leg joints over their ranges)
        self.foot_reach = {}
        for f in mv.feet:
            pts = np.zeros((n, 3))
            for i in range(n):
                d.qpos[:] = self.q_start
                for dof in mv.foot_dofs[f]:
                    j = int(m.dof_jntid[dof])
                    if j in self.range:
                        d.qpos[int(m.jnt_qposadr[j])] = rng.uniform(*self.range[j])
                mujoco.mj_kinematics(m, d)
                R = d.xmat[self.root].reshape(3, 3)
                pts[i] = R.T @ (d.xpos[f] - d.xpos[self.root])
            self.foot_reach[f] = (cKDTree(pts), pts)
        # each arm parent's orientation relative to the base at the start (for parents that are
        # not the planned chest)
        sd = s.d
        self._from_root = {int(m.body_parentid[s.shoulder[h]]):
                           sd.xmat[self.root].reshape(3, 3).T
                           @ sd.xmat[int(m.body_parentid[s.shoulder[h]])].reshape(3, 3)
                           for h in mv.hands}

    def _reachable_hands(self, ch, times):
        """Hand channels moved, where needed, to the nearest place the arm can really put the
        hand (a guard closer to the shoulder than a short forearm folds, for example), then
        smoothed in time. Done before the poses are solved: an unreachable target drives a
        least-squares solver into its joint limits, where it stays until it flips."""
        out = dict(ch)
        for k, h in self.hand.items():
            tree, pts, parent = self.reach[h]
            tol = 0.02 * self.arm[k]
            local = np.zeros((len(times), 3))
            rots = []
            on_chest = parent == self.chest and self.chest is not None
            for n in range(len(times)):
                # the arm's parent turns with the planned chest (or base) orientation
                _, rq, _, _, chest = self._frame_targets(ch, n, np.zeros(2))
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, chest if on_chest else rq)
                R = R.reshape(3, 3) @ (np.eye(3) if on_chest else self._from_root[parent])
                rots.append(R)
                rel = self.axes @ (self.arm[k] * ch[f"hand_{k}"][n])
                loc = R.T @ rel
                local[n] = _toward_reach(tree, pts, loc, tol)
            local = gaussian_filter1d(local, 0.03 / self.dt, axis=0, mode="nearest")
            world = np.array([R @ x for R, x in zip(rots, local, strict=True)])
            out[f"hand_{k}"] = (world @ self.axes) / self.arm[k]
        return out

    def _reachable_feet(self, ch, support):
        """A foot in the air (a kick, a knee) is moved, where needed, to the nearest place its
        leg can really put it, so each robot kicks as high as its own hip and knee ranges allow.
        Planted feet are left alone."""
        out = dict(ch)
        for k, f in self.foot.items():
            tree, pts = self.foot_reach[f]
            delta = np.zeros((len(support[f]), 3))
            for n in np.where(~support[f])[0]:
                pel, rq, feet, _, _ = self._frame_targets(ch, n, np.zeros(2))
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, rq)
                R = R.reshape(3, 3)
                rel = R.T @ (feet[f][0] - pel)
                # a leg's sampled reach is sparse (6+ joints): only clear misses are moved
                fixed = _toward_reach(tree, pts, rel, 0.04 * self.leg)
                delta[n] = self.axes.T @ (R @ (fixed - rel)) / self.leg
            delta = gaussian_filter1d(delta, 0.03 / self.dt, axis=0, mode="nearest")
            delta[support[f]] = 0.0
            fv = ch[f"foot_{k}"].copy()
            fv[:, :3] += delta
            out[f"foot_{k}"] = fv
        return out

    def can_stand_on_one_foot(self, tol=0.01, share=0.85):
        """Whether the robot can bring its hips `share` of the way over either foot with both
        soles flat (needed before a foot can lift without the stance foot rolling onto its
        edge). A leg without an ankle roll stays vertical side-on while its foot is flat, so
        wide-set hips cannot move
        over one foot: such a robot can only stand on one leg dynamically."""
        d = self.s.d
        feet = {f: (d.xpos[f].copy(), d.xquat[f].copy()) for f in self.mv.feet}
        for f in self.mv.feet:
            pel = self.root0.copy()
            over = self.mv.foot_center(f)[:2] + (self.root0[:2] - d.subtree_com[self.root][:2])
            pel[:2] += share * (over - pel[:2])
            pel[2] -= 0.08 * self.leg
            _, res = self._ik(self.q_rest, pel, d.xquat[self.root].copy(), feet, {}, None, set())
            if res > tol:
                return False
        return True

    def _proximal(self, hand, n):
        """The `n` arm dofs nearest the shoulder (all of them for arms with n or fewer)."""
        m, s = self.m, self.s
        chain = []
        for b in reversed(s._arm_bodies(hand)):                  # shoulder first
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                if int(m.jnt_type[j]) in _SCALAR:
                    chain.append(int(m.jnt_dofadr[j]))
        return chain[:n]

    # ------------------------------------------------------------------ start values
    def start_values(self):
        """Every channel's value in the start pose (what an unkeyed channel holds)."""
        d = self.s.d
        out = dict(pelvis=np.zeros(3), turn=np.zeros(3), chest=np.zeros(2),
                   foot_L=np.zeros(5), foot_R=np.zeros(5))
        for k, h in self.hand.items():
            rel = self.mv.w.ee_pos(d, h) - d.xpos[self.s.shoulder[h]]
            out[f"hand_{k}"] = self.axes.T @ rel / self.arm[k]
        return out

    # ------------------------------------------------------------------ targets
    def _frame_targets(self, ch, n, shift):
        A = self.axes
        pel = self.root0 + A @ (self.leg * ch["pelvis"][n]) + np.r_[shift, 0.0]
        yaw, pitch, roll = ch["turn"][n]
        chest_yaw, chest_pitch = ch["chest"][n]
        if self.chest is None:                       # no waist: the base turns for the chest
            yaw, pitch = yaw + chest_yaw, pitch + chest_pitch
        rq = _mul(_ypr_quat(self.heading + yaw, pitch, roll), self.root_tilt0)
        feet = {}
        for k, f in self.foot.items():
            fv = ch[f"foot_{k}"][n]
            pos = self.foot0[k] + A @ (self.leg * fv[:3])
            quat = _mul(_ypr_quat(self.heading + fv[3], fv[4]), self.foot_tilt0[k])
            feet[f] = (pos, quat)
        hands = {}
        for k, h in self.hand.items():
            rel = A @ (self.arm[k] * ch[f"hand_{k}"][n])
            r = float(np.linalg.norm(rel))
            if r > self.REACH * self.arm[k]:             # a straight arm is a singularity
                rel *= self.REACH * self.arm[k] / r
            hands[h] = rel
        chest = None
        if self.chest is not None:
            chest = _mul(_mul(rq, _ypr_quat(chest_yaw, chest_pitch)), self.chest_rel0)
        return pel, rq, feet, hands, chest

    # ------------------------------------------------------------------ inverse kinematics
    def _ik(self, q_seed, pel, rq, feet, hands, chest, lifted, iters=60, tol=2e-5):
        m, d, mv, s = self.m, self.scratch, self.mv, self.s
        d.qpos[:] = q_seed
        d.qpos[mv.qa:mv.qa + 3] = pel
        d.qpos[mv.qa + 3:mv.qa + 7] = rq
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        # the base orientation is a strong preference, not a hard constraint: where a pose is out
        # of the legs' reach the body may tilt a little rather than leave a foot hanging
        rr = self.root_rot
        cols = np.r_[self.ik_dofs, rr]
        res = 0.0
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rows, errs, wts = [], [], []
            for f, (p, q) in feet.items():
                mujoco.mj_jacBody(m, d, jacp, jacr, f)
                J = np.zeros((6, m.nv))
                fc = list(mv.foot_dofs[f]) + list(rr)
                J[:3, fc], J[3:, fc] = jacp[:, fc], jacr[:, fc]
                rows.append(J)
                errs.append(np.r_[p - d.xpos[f], _quat_err(d.xquat[f], q)])
                # a planted foot must be flat; a foot in the air only drifts towards its planned
                # angle (an ankle at its range limit must not snap between solutions)
                wts += [1.0] * 3 + [0.05 if f in lifted else 1.0] * 3
            for h, rel in hands.items():
                tool = mv.w.ee_pos(d, h)
                target = d.xpos[s.shoulder[h]] + rel
                mujoco.mj_jac(m, d, jacp, None, tool, h)
                J = np.zeros((3, m.nv))
                hc = self.place_dofs[h]
                J[:, hc] = jacp[:, hc]
                rows.append(J)
                errs.append(target - tool)
                wts += [1.0] * 3
            if chest is not None and self.waist:
                mujoco.mj_jacBody(m, d, None, jacr, self.chest)
                J = np.zeros((3, m.nv))
                J[:, self.waist] = jacr[:, self.waist]
                rows.append(J)
                errs.append(_quat_err(d.xquat[self.chest], chest))
                wts += [0.5] * 3
            res = float(np.abs(np.concatenate(errs)).max())
            mujoco.mj_jacBody(m, d, None, jacr, self.root)
            J = np.zeros((3, m.nv))
            J[:, rr] = jacr[:, rr]
            rows.append(J)
            errs.append(_quat_err(d.xquat[self.root], rq))
            wts += [self.BASE_TILT_WEIGHT] * 3
            J, e, W = np.vstack(rows), np.concatenate(errs), np.array(wts)
            Jc = J[:, cols] * W[:, None]
            We = e * W
            lam = 1e-4 + 1e-2 * float(We @ We)          # Levenberg-Marquardt style damping
            JJ = Jc @ Jc.T + lam * np.eye(len(e))
            dq_c = Jc.T @ np.linalg.solve(JJ, We)
            # unused freedom drifts back to the rest posture (null space of the tasks)
            rest = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, rest, 1.0, d.qpos, self.q_rest)
            rest[rr] = 0.0
            N = np.eye(len(cols)) - Jc.T @ np.linalg.solve(JJ, Jc)
            pull = 0.3 * N @ rest[cols]
            # done when the tasks are met and the spare freedom has settled at the rest
            # posture (stopping at the tasks alone lets a 7-joint arm drift frame to frame:
            # wrists curl, elbows flip)
            if res < tol and np.abs(pull).max() < 1e-4:
                break
            dq_c += pull
            big = np.abs(dq_c).max()
            if big > 0.2:                                # no leaps across the joint space
                dq_c *= 0.2 / big
            dq = np.zeros(m.nv)
            dq[cols] = dq_c
            mujoco.mj_integratePos(m, d.qpos, dq, 1.0)
            for a, lo, hi in self.limits:
                d.qpos[a] = min(max(d.qpos[a], lo), hi)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        return d.qpos.copy(), res

    # ------------------------------------------------------------------ compile
    def compile(self, script: Script, *, passes=6, tol=0.004, retime=4, load_target=0.85,
                twist_target=0.6, log=None):
        """Compile `script` into a Plan.

        Each round solves the poses and runs the ZMP filter (`passes` times at most, until the
        ZMP is within `tol` of the shrunk support polygon). Then comes the time-scaling.
        Inverse dynamics of the reference gives every frame's torque load on the motors of the
        arms, the waist and any leg in the air. The support legs are left out, because their
        share depends on how the ground carries the robot, and that is the controller's
        decision. It also gives the twist load on the planted feet. Wherever either is over
        its limit (`load_target` of a motor's torque, `twist_target` of the feet's friction
        moment, which is an estimate and so gets a wider margin), the
        script's clock is slowed there, by about the square root of the excess (torque goes
        with acceleration, which goes as 1/time^2), and the reference is compiled again. The rest
        of the script keeps its speed: the result is as fast as the robot's own motors and
        feet allow."""
        dt = self.dt
        end = script.duration + 1.0                                  # rest at the end
        grid = np.arange(0.0, end + dt / 2, dt)
        slow = np.ones(len(grid))
        retimes = []
        plan = None
        for r in range(retime + 1):
            tau = _clock(grid, slow, end, dt)
            plan = self._compile_at(script, tau, passes, tol, log)
            load = self._motor_load(plan)
            plan.motor_load = load
            over = np.maximum.reduce([load / load_target, plan.twist_load / twist_target,
                                      plan.slip_load / twist_target])
            retimes.append(dict(round=r, duration_s=round(plan.duration, 2),
                                max_motor_load=round(float(load.max()), 2),
                                max_twist_load=round(float(plan.twist_load.max()), 2),
                                max_slip_load=round(float(plan.slip_load.max()), 2)))
            if log:
                log(f"round {r}: {plan.duration:.2f} s, peak motor load "
                    f"{load.max() * 100:.0f}% of limit, twist {plan.twist_load.max() * 100:.0f}%"
                    f" and slip {plan.slip_load.max() * 100:.0f}% of friction")
            if over.max() <= 1.0 + 1e-3 or r == retime:
                break
            # a little more than the square root: the slowed stretch also changes the hip
            # correction around it, so the bare estimate converges slowly from above
            f = np.maximum(over, 1.0) ** 0.6
            f = maximum_filter1d(f, size=max(1, int(round(0.12 / dt))))   # whole movement
            f = gaussian_filter1d(f, 0.04 / dt)
            slow *= np.interp(grid, tau, f)
        plan.report.update(retime=retimes, script_duration_s=round(float(end), 2),
                           max_motor_load=round(float(plan.motor_load.max()), 2))
        return plan

    def _compile_at(self, script, tau, passes, tol, log):
        """One reference with the script sampled at script times `tau` (one per plan frame)."""
        dt = self.dt
        N = len(tau)
        times = np.arange(N) * dt
        start = self.start_values()
        ch = {c: script.sample(c, tau, start[c]) for c in Script.DIMS}
        ch = self._reachable_hands(ch, times)
        support = {self.foot[k]: script.support_at(k, tau) for k in SIDES}
        ch = self._reachable_feet(ch, support)
        shift = np.zeros((N, 2))
        report = dict(passes=[])
        for it in range(passes):
            Q, res = self._solve_all(ch, shift, support)
            cen = self._centroidal(Q)
            zmp = self._zmp(cen)
            polys = self._polygons(Q, support)
            des, margin = zip(*(_project(z, p) for z, p in zip(zmp, polys, strict=True)),
                              strict=True)
            err = np.array(des) - zmp
            worst = float(np.abs(err).max())
            report["passes"].append(dict(pass_=it, worst_zmp_violation_m=round(worst, 4),
                                         max_hip_shift_m=round(float(np.abs(shift).max()), 4)))
            if log:
                log(f"  filter pass {it}: worst ZMP outside margin {worst * 1000:.1f} mm, "
                    f"hip shift up to {np.abs(shift).max() * 1000:.0f} mm")
            if worst < tol:
                break
            z = cen["com"][:, 2] - self._ground
            zdd = cen["com_a"][:, 2]
            for ax in range(2):
                shift[:, ax] += 0.9 * _cart_table(err[:, ax], z, zdd, dt)
        margin = np.array(margin)
        twist = self._twist_load(Q, support, cen, zmp)
        # horizontal ground force the reference needs, over what friction can give
        fz = self.mass * np.maximum(cen["com_a"][:, 2] + GRAVITY, 0.2 * GRAVITY)
        slip = self.mass * np.linalg.norm(cen["com_a"][:, :2], axis=1) / (self.mu * fz)
        hands, feet = self._end_effectors(Q, support)
        rw, rdw = self._root_rates(Q)
        v, a = self._joint_rates(Q)
        report.update(duration_s=round(float(times[-1]), 2), frames=N,
                      min_vertical_load_g=round(float((cen["com_a"][:, 2] / GRAVITY + 1).min()),
                                                3),
                      max_ik_residual=round(float(res.max()), 4),
                      min_zmp_margin_m=round(float(margin.min()), 4),
                      max_twist_load=round(float(twist.max()), 2),
                      twist_overload_s=round(float((twist > 1).sum() * dt), 2),
                      max_hip_shift_m=round(float(np.abs(shift).max()), 4))
        labels = [(float(np.interp(t, tau, times)), txt) for t, txt in sorted(script.labels)]
        return Plan(dt=dt, times=times, q=Q, v=v, a=a, com=cen["com"], com_v=cen["com_v"],
                    com_a=cen["com_a"], root_quat=Q[:, self.mv.qa + 3:self.mv.qa + 7].copy(),
                    root_w=rw, root_dw=rdw, hands=hands, feet=feet, zmp=zmp, zmp_margin=margin,
                    ik_residual=res, hip_shift=shift, twist_load=twist, slip_load=slip,
                    labels=labels, report=report)

    def _motor_load(self, plan):
        """Per frame, the largest |torque needed| / limit over the motors of the arms, the waist
        and the legs whose foot is in the air (inverse dynamics of the reference, no contacts)."""
        m, d, s, mv = self.m, self.scratch, self.s, self.mv
        V = self._velocities(plan.q)
        A = np.gradient(V, self.dt, axis=0)
        lim = {int(s.motors.dof[k]): float(s.motors.limit[k]) for k in range(len(s.motors.ids))
               if s.motors.dof[k] >= 0 and np.isfinite(s.motors.limit[k])
               and s.motors.limit[k] > 0}
        upper = sorted(({v for h in mv.hands for v in mv.hand_dofs[h]} | set(self.waist))
                       & set(lim))
        legs = {f: [v for v in mv.foot_dofs[f] if v in lim] for f in mv.feet}
        flags = m.opt.disableflags
        m.opt.disableflags = flags | int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(
            mujoco.mjtDisableBit.mjDSBL_EQUALITY)
        out = np.zeros(len(plan.q))
        try:
            for n in range(len(plan.q)):
                d.qpos[:], d.qvel[:], d.qacc[:] = plan.q[n], V[n], A[n]
                mujoco.mj_inverse(m, d)
                dofs = upper + [v for f, e in plan.feet.items() if not e["support"][n]
                                for v in legs[f]]
                out[n] = max((abs(d.qfrc_inverse[v]) / lim[v] for v in dofs), default=0.0)
        finally:
            m.opt.disableflags = flags
        return out

    def _solve_all(self, ch, shift, support):
        N = len(shift)
        Q = np.zeros((N, self.m.nq))
        res = np.zeros(N)
        q = self.q_start.copy()
        for n in range(N):
            pel, rq, feet, hands, chest = self._frame_targets(ch, n, shift[n])
            lifted = {f for f, on in support.items() if not on[n]}
            q, res[n] = self._ik(q, pel, rq, feet, hands, chest, lifted)
            if res[n] > 0.01:
                # warm starts can walk into a dead end against joint limits and stay there:
                # try fresh seeds and keep a clearly better solution
                for seed in self.seeds:
                    q2, r2 = self._ik(seed, pel, rq, feet, hands, chest, lifted)
                    if r2 < 0.7 * res[n]:
                        q, res[n] = q2, r2
            Q[n] = q
        return Q, res

    def _velocities(self, Q):
        m, dt = self.m, self.dt
        N = len(Q)
        V = np.zeros((N, m.nv))
        for n in range(N):
            a, b = max(n - 1, 0), min(n + 1, N - 1)
            if b > a:
                mujoco.mj_differentiatePos(m, V[n], (b - a) * dt, Q[a], Q[b])
        return V

    def _centroidal(self, Q):
        m, d = self.m, self.scratch
        V = self._velocities(Q)
        N = len(Q)
        com, lin, ang = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3))
        for n in range(N):
            d.qpos[:], d.qvel[:] = Q[n], V[n]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_comVel(m, d)
            mujoco.mj_subtreeVel(m, d)
            com[n] = d.subtree_com[self.root]
            lin[n] = d.subtree_linvel[self.root]
            ang[n] = d.subtree_angmom[self.root]
        g = lambda x: np.gradient(x, self.dt, axis=0)  # noqa: E731
        return dict(com=com, com_v=lin, com_a=g(lin), dL=g(ang))

    def _sole_z(self, d, f):
        bodies, local, _, radii = foot_contact_points(self.m, d, f, with_radius=True)
        return min(float((d.xmat[b].reshape(3, 3) @ p + d.xpos[b])[2]) - r
                   for b, p, r in zip(bodies, local, radii, strict=True))

    def _zmp(self, cen):
        c, ca, dL = cen["com"], cen["com_a"], cen["dL"]
        M = self.mass
        z = c[:, 2] - self._ground
        # a reference that accelerates down faster than gravity would need the feet to pull on
        # the floor: it is clamped here (and reported by compile as a vertical violation)
        fz = M * np.maximum(ca[:, 2] + GRAVITY, 0.2 * GRAVITY)
        px = c[:, 0] - (M * z * ca[:, 0] + dL[:, 1]) / fz
        py = c[:, 1] - (M * z * ca[:, 1] - dL[:, 0]) / fz
        return np.c_[px, py]

    def _polygons(self, Q, support):
        m, d = self.m, self.scratch
        out = []
        for n in range(len(Q)):
            d.qpos[:] = Q[n]
            mujoco.mj_kinematics(m, d)
            pts = []
            for k, f in self.foot.items():
                if not support[f][n]:
                    continue
                bodies, local = self.sole[k]
                sole = np.array([(d.xmat[b].reshape(3, 3) @ p + d.xpos[b])[:2]
                                 for b, p in zip(bodies, local, strict=True)])
                pts += list(self._ankle_clip(d, f, sole))
            pts = np.array(pts)
            c = pts.mean(0)
            out.append(c + (1 - self.margin) * (pts - c))
        return out

    def _ankle_clip(self, d, f, sole):
        """The part of a sole the ankle motors can put the centre of pressure on: an ankle joint
        of torque limit tau holds the body's weight at most tau / (m g) from its axis (with a
        20 % reserve). Small robots with weak ankles get a much smaller support area than their
        soles."""
        out = sole.copy()
        w = self.mass * GRAVITY
        for j, tau in self.ankles[f]:
            axis = d.xaxis[j]
            u = np.cross([0.0, 0.0, 1.0], axis)[:2]
            if np.linalg.norm(u) < 0.5:                  # a vertical axis does not tilt the sole
                continue
            u /= np.linalg.norm(u)
            reach = 0.8 * tau / w
            off = (out - d.xanchor[j][:2]) @ u
            out -= np.outer(off - np.clip(off, -reach, reach), u)
        return out

    def _twist_load(self, Q, support, cen, zmp):
        """Vertical moment the reference needs (rate of its angular momentum about the vertical)
        over the friction moment the supporting soles can give about the zero-moment point.
        Above 1 the planted feet spin: the ZMP filter cannot fix that by moving the hips (a
        twist is limited by friction, not by balance), so it is reported for the script."""
        m, d = self.m, self.scratch
        mu = self.mu
        out = np.zeros(len(Q))
        for n in range(len(Q)):
            d.qpos[:] = Q[n]
            mujoco.mj_kinematics(m, d)
            pts = []
            for k, f in self.foot.items():
                if support[f][n]:
                    bodies, local = self.sole[k]
                    pts += [(d.xmat[b].reshape(3, 3) @ p + d.xpos[b])[:2]
                            for b, p in zip(bodies, local, strict=True)]
            r = 0.7 * float(np.mean(np.linalg.norm(np.array(pts) - zmp[n], axis=1)))
            fz = self.mass * max(cen["com_a"][n, 2] + GRAVITY, 0.2 * GRAVITY)
            out[n] = abs(cen["dL"][n, 2]) / max(mu * fz * r, 1e-6)
        return out

    def _end_effectors(self, Q, support):
        m, d, mv = self.m, self.scratch, self.mv
        N = len(Q)
        hands = {h: dict(pos=np.zeros((N, 3)), quat=np.zeros((N, 4))) for h in mv.hands}
        feet = {f: dict(pos=np.zeros((N, 3)), quat=np.zeros((N, 4)), support=support[f])
                for f in mv.feet}
        for n in range(N):
            d.qpos[:] = Q[n]
            mujoco.mj_kinematics(m, d)
            for h in mv.hands:
                hands[h]["pos"][n] = mv.w.ee_pos(d, h)
                hands[h]["quat"][n] = d.xquat[h]
            for f in mv.feet:
                feet[f]["pos"][n] = d.xpos[f]
                feet[f]["quat"][n] = d.xquat[f]
        for e in list(hands.values()) + list(feet.values()):
            e["vel"] = np.gradient(e["pos"], self.dt, axis=0)
            e["acc"] = np.gradient(e["vel"], self.dt, axis=0)
        return hands, feet

    def _root_rates(self, Q):
        qa, dt, N = self.mv.qa, self.dt, len(Q)
        w = np.zeros((N, 3))
        for n in range(N):
            a, b = max(n - 1, 0), min(n + 1, N - 1)
            if b > a:
                w[n] = _quat_err(Q[a, qa + 3:qa + 7], Q[b, qa + 3:qa + 7]) / ((b - a) * dt)
        return w, np.gradient(w, dt, axis=0)

    def _joint_rates(self, Q):
        V = self._velocities(Q)
        A = np.gradient(V, self.dt, axis=0)
        V[:, :6] = 0.0
        A[:, :6] = 0.0
        V[:, self.finger_dofs] = 0.0
        A[:, self.finger_dofs] = 0.0
        return V, A


def _subtree_set(m, body):
    out = {body}
    for b in range(body + 1, m.nbody):
        if int(m.body_parentid[b]) in out:
            out.add(b)
    return out


def _quat_err(q_cur, q_des):
    v = np.zeros(3)
    mujoco.mju_subQuat(v, q_des, q_cur)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q_cur)
    return R.reshape(3, 3) @ v


def _toward_reach(tree, pts, p, tol):
    """`p` moved towards the sampled reachable points near it: not at all within `tol` of a
    sample, fully onto their mean from 2 * tol, blended in between (no jumps as a target
    crosses the edge of the sampled set)."""
    dist, _ = tree.query(p)
    w = min(max((dist - tol) / tol, 0.0), 1.0)
    if w == 0.0:
        return p
    _, idx = tree.query(p, k=8)
    return p + w * (pts[idx].mean(0) - p)


def _clock(grid, slow, end, dt):
    """Script time at each plan frame: the script's clock runs 1/slow(script time) as fast."""
    out = [0.0]
    while out[-1] < end:
        out.append(out[-1] + dt / float(np.interp(out[-1], grid, slow)))
    out[-1] = end
    return np.array(out)


def _cart_table(dp, z, zdd, dt):
    """Horizontal CoM shift x (zero at both ends) that moves the zero-moment point by dp:
    dp = x - z / (g + zdd) * x'' (the cart-table model, solved over the whole timeline)."""
    N = len(dp)
    if N < 3:
        return np.zeros(N)
    a = z / ((GRAVITY + zdd) * dt * dt)
    n = N - 2
    ab = np.zeros((3, n))
    ab[0, 1:] = -a[1:-1][:-1]
    ab[1, :] = 1 + 2 * a[1:-1]
    ab[2, :-1] = -a[1:-1][1:]
    x = np.zeros(N)
    x[1:-1] = solve_banded((1, 1), ab, dp[1:-1])
    return x


def _project(p, poly):
    """(nearest point of the convex polygon to p, signed distance inside it)."""
    try:
        hull = ConvexHull(poly)
        verts = poly[hull.vertices]
    except Exception:                                  # degenerate: a segment / point
        verts = poly
    A = []
    inside = True
    best, bd = p, np.inf
    for i in range(len(verts)):
        a, b = verts[i], verts[(i + 1) % len(verts)]
        e = b - a
        L2 = max(float(e @ e), 1e-12)
        t = min(max(float((p - a) @ e) / L2, 0.0), 1.0)
        y = a + t * e
        dist = float(np.linalg.norm(p - y))
        if dist < bd:
            best, bd = y, dist
        cross = e[0] * (p - a)[1] - e[1] * (p - a)[0]  # hull vertices run counter-clockwise
        A.append(cross)
        if cross < 0:
            inside = False
    if inside and len(verts) >= 3:
        return p, bd
    return best, -bd


# ----------------------------------------------------------------------------------- tracking
class Tracker:
    """Feeds a Plan to the whole-body controller as its references (WholeBodyController.reference):
    joint posture with velocity / acceleration feed-forward, CoM, base orientation, hands and
    feet. A foot's ground support follows the plan's lift-off and touchdown times. A foot is
    held where it really landed: its reference is offset by the landing error until it lifts
    again, because pulling a planted foot to the planned spot would fight its own contact."""

    TOUCHDOWN_WAIT = 0.15
    DRIFT_TAU = 0.3          # s: how quickly the plan follows the planted feet's drift
    LATE_WAIT = 0.3          # s: longest wait for an overdue foot to touch before it counts
    PRESS = 0.015            # m: an overdue foot's reference goes this far below its landing
    SWING_GAINS = (600.0, 50.0)

    def __init__(self, compiler: Compiler, plan: Plan, t0: float):
        self.c, self.p, self.t0 = compiler, plan, t0
        self.offset = {f: np.zeros(3) for f in plan.feet}
        self.late = {f: None for f in plan.feet}
        self.early = {f: None for f in plan.feet}      # planned landing frame of an early one
        # how far the planted feet have drifted from their planned spots (landing errors, small
        # slips): the rest of the plan is carried along with them, so the weight is shifted
        # over where the feet really are and a planted foot is never hauled back
        self.drift = np.zeros(3)
        self.done = False
        s = compiler.s
        self._foot_geoms = {f: {g for g in range(s.m.ngeom) if int(s.m.geom_bodyid[g]) in
                                 _subtree_set(s.m, f)} for f in plan.feet}

    def _touching(self, d, f):
        geoms = self._foot_geoms[f]
        return any((d.contact[i].geom1 in geoms) != (d.contact[i].geom2 in geoms)
                   and d.contact[i].dist < 0.002 for i in range(d.ncon))

    def _at(self, t):
        p = self.p
        x = min(max(t / p.dt, 0.0), len(p.times) - 1.0)
        k = min(int(x), len(p.times) - 2)
        return k, x - k

    def __call__(self, w, d):
        p, c = self.p, self.c
        t = d.time - self.t0
        if t >= p.duration:
            self.done = True
        k, u = self._at(t)
        lerp = lambda arr: (1 - u) * arr[k] + u * arr[k + 1]  # noqa: E731
        q = lerp(p.q)
        w.qpos0[c.track_q] = q[c.track_q]
        w.q_ref_vel[:] = lerp(p.v)
        w.q_ref_acc[:] = lerp(p.a)
        w.com_ref = lerp(p.com) + self.drift
        w.com_ref_vel = lerp(p.com_v)
        w.com_ref_acc = lerp(p.com_a)
        rq = lerp(p.root_quat)
        w.base_quat0 = rq / np.linalg.norm(rq)
        w.base_ref_w, w.base_ref_dw = lerp(p.root_w), lerp(p.root_dw)
        for h, e in p.hands.items():
            w.ee_targets[h] = lerp(e["pos"]) + self.drift
            w.ee_ref_vel[h] = lerp(e["vel"])
            w.ee_ref_acc[h] = lerp(e["acc"])
            hq = lerp(e["quat"])
            w.ee_rot_ref[h] = hq / np.linalg.norm(hq)
        n = k if u < 0.5 else k + 1
        wait = int(round(self.TOUCHDOWN_WAIT / p.dt))
        planted = []
        for f, e in p.feet.items():
            on = bool(e["support"][n])
            if on:
                self.early[f] = None
            elif self.early[f] is None and f not in w.support_bodies and self._touching(d, f):
                # an early touchdown (the foot meets the ground shortly before its planned
                # landing) lands it there, at the planned landing pose, until the plan catches
                # up: dragging it on through the rest of its swing shoves the body
                ahead = np.nonzero(e["support"][n:n + wait + 1])[0]
                if len(ahead):
                    self.early[f] = n + int(ahead[0])
            on = on or self.early[f] is not None
            ref = self.early[f]
            pos = e["pos"][ref] if ref is not None else lerp(e["pos"])
            fq = e["quat"][ref] if ref is not None else lerp(e["quat"])
            if on and f not in w.support_bodies:
                # touchdown when the foot really meets the ground (a foot trailing its swing
                # reference keeps reaching down for the planned spot for up to LATE_WAIT s);
                # then it is held where it is
                if self.late[f] is None:
                    self.late[f] = t
                if self._touching(d, f) or t - self.late[f] > self.LATE_WAIT:
                    w.set_support(f, True)
                    self.late[f] = None
                    self.offset[f] = np.r_[d.xpos[f][:2] - pos[:2], 0.0]
                else:
                    on = False
                    pos = pos - [0.0, 0.0, self.PRESS]  # an overdue foot reaches for the floor
            elif not on and f in w.support_bodies:
                w.set_support(f, False)
                self.offset[f] = np.zeros(3)
            if f in w.support_bodies:
                # a planted foot is held where it is (it follows its own small slips)
                slip = np.r_[d.xpos[f][:2] - (pos[:2] + self.offset[f][:2]), 0.0]
                self.offset[f] += slip * min(1.0, w.dt / self.DRIFT_TAU)
                w.foot_anchor[f] = pos + self.offset[f]
                planted.append(self.offset[f])
            else:
                w.foot_anchor[f] = pos + self.drift
            w.foot_ref_vel[f] = np.zeros(3) if on else lerp(e["vel"])
            w.foot_ref_acc[f] = np.zeros(3) if on else lerp(e["acc"])
            w.foot_quat0[f] = fq / np.linalg.norm(fq)
        if planted:
            self.drift += (np.mean(planted, axis=0) - self.drift) * min(1.0, w.dt / self.DRIFT_TAU)

    def release(self, w):
        """Hand back to goal-driven control, holding the final pose (no reference jump)."""
        w.reference = None
        w.com_goal = w.com_ref.copy()
        w.com_ref_vel = np.zeros(3)
        w.com_ref_acc = np.zeros(3)
        w.base_ref_w, w.base_ref_dw = np.zeros(3), np.zeros(3)
        w.q_ref_vel[:] = 0.0
        w.q_ref_acc[:] = 0.0
        for h in w.ees:
            w.ee_goal[h] = w.ee_targets[h].copy()
            w.ee_ref_vel[h] = np.zeros(3)
            w.ee_ref_acc[h] = np.zeros(3)
            w.ee_rot[h] = w.ee_rot_ref[h].copy()
        for f in w.feet:
            w.foot_goal[f] = w.foot_anchor[f].copy()
            w.foot_ref_vel[f] = np.zeros(3)
            w.foot_ref_acc[f] = np.zeros(3)


def perform(mv: Moves, compiler: Compiler, plan: Plan, *, on_tick=None, pose_weight=50.0,
            leg_pose_weight=5.0, posture_gains=(200.0, 28.0), joint_speed_cap=None,
            limit_brake=400.0):
    """Run a compiled plan on the live robot through the whole-body controller. Returns the
    measured outcome (falls raise RuntimeError from the session checks).

    The plan's joint trajectory is the main reference for the arms and waist (stiff, fed
    forward); the legs follow it lightly (`leg_pose_weight`), because a planted leg is placed
    by the balance tasks (CoM, base, feet) and a foot that landed a little off its planned
    spot must not be fought by the leg's planned joint angles. The hands' and feet's Cartesian
    tasks agree with the plan. The hand orientation task is off: the planned wrist
    angles already orient the fists, and a second, slightly inconsistent orientation goal makes
    the controller hold whole arms back."""
    s, w = mv.s, mv.w
    tr = Tracker(compiler, plan, s.d.time)
    saved = (w.pose_weight, dict(w.speed_cap), w.ee_feedforward, w.ee_leash, w.torso_body,
             w.weights["ee_rot"], w.gains["posture"], w.limit_brake)
    legs = {v for f in mv.feet for v in mv.foot_dofs[f]}
    w.pose_weight = np.array([leg_pose_weight if v in legs else pose_weight
                              for v in w.posture_dofs])
    w.limit_brake = limit_brake
    w.weights["ee_rot"] = 0.0
    w.gains["posture"] = posture_gains
    w.speed_cap = ({} if joint_speed_cap is None
                   else {int(v): joint_speed_cap for v in compiler.ik_dofs})
    w.ee_feedforward, w.ee_leash, w.torso_body = True, 0.08, None
    swing = w.gains["swing"]
    w.gains["swing"] = Tracker.SWING_GAINS
    w.reference = tr
    n = max(1, int(round(mv.TICK / s.m.opt.timestep)))
    hand_err = {h: 0.0 for h in mv.hands}
    peak = {h: 0.0 for h in mv.hands}
    foot_peak = {f: 0.0 for f in mv.feet}
    jv = 0.0
    try:
        while not tr.done:
            s.step(n)
            if on_tick is not None:
                on_tick(s)
            if s.checks.fell_at is not None:
                raise RuntimeError(f"robot fell at {s.checks.fell_at} s")
            mujoco.mj_subtreeVel(s.m, s.d)
            for h in mv.hands:
                hand_err[h] = max(hand_err[h], float(np.linalg.norm(w.ee_pos(s.d, h)
                                                                    - w.ee_targets[h])))
                jac = np.zeros((3, s.m.nv))
                mujoco.mj_jac(s.m, s.d, jac, None, w.ee_pos(s.d, h), h)
                peak[h] = max(peak[h], float(np.linalg.norm(jac @ s.d.qvel)))
            for f in mv.feet:
                jac = np.zeros((3, s.m.nv))
                mujoco.mj_jacBody(s.m, s.d, jac, None, f)
                foot_peak[f] = max(foot_peak[f], float(np.linalg.norm(jac @ s.d.qvel)))
            jv = max(jv, float(np.abs(s.d.qvel[compiler.ik_dofs]).max()))
    finally:
        tr.release(w)
        w.gains["swing"] = swing
        (w.pose_weight, w.speed_cap, w.ee_feedforward, w.ee_leash, w.torso_body,
         w.weights["ee_rot"], w.gains["posture"], w.limit_brake) = saved
    return dict(peak_hand_speed_ms={s.m.body(h).name: round(v, 2) for h, v in peak.items()},
                peak_foot_speed_ms={s.m.body(f).name: round(v, 2) for f, v in foot_peak.items()},
                worst_hand_tracking_error_m={s.m.body(h).name: round(v, 3)
                                             for h, v in hand_err.items()},
                peak_joint_speed_rads=round(jv, 2))
