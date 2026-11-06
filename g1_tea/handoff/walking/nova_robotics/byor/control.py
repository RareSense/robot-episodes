"""Generic real-physics robot control: motor layer + whole-body QP controller.

Motor layer
    Every actuator becomes a force input with the robot's real limit. Joint motors expose the
    interface real robots have (e.g. Unitree LowCmd): tau = kp (q* - q) + kd (dq* - dq) + tau_ff.
    Tendon and site actuators (cables, telescoping arms, propellers) become force inputs acting
    through MuJoCo's configuration-dependent moment arms. MuJoCo clips every input at its limit.

Whole-body controller
    One QP per physics step, identical for every robot class:
        x = [ddq (nv), u (inputs), lambda (4 per contact, friction-pyramid edges), f_eq]
        min  w_dyn |M ddq + h - S(q) u - Jc' E lambda - Jeq' f_eq|^2      equations of motion
           + w_eq |Jeq ddq + ...|^2                                      couplings stay closed
           + sum_tasks w |J ddq + dJ dq - a*|^2                          no-slip, CoM, torso,
                                                                         feet, hands
           + w_post |N (ddq - a_posture)|^2                              posture in the null space
        s.t. input limits, lambda >= 0, joint-limit velocity dampers on ddq
    Task priorities are separated by orders of magnitude: physics, then contacts, then balance,
    then hands, then posture. The plan keeps the centre of pressure off the sole edge. Tasks are
    built from the analysis report (feet, hands), so nothing is written per robot.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .analysis import robot_roots, subtree as _subtree_raw


# ----------------------------------------------------------------------------------- motor layer

class MotorLayer:
    """Every actuator becomes a force-controlled input with its real limit.

    joint actuators   ctrl = joint torque/force (LowCmd PD + feed-forward available)
    tendon / site     ctrl = actuator force along the tendon or site axis (cables, telescoping
                      arms, propellers); its effect on the joints comes from MuJoCo's moment arms
    """

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        ids, dof, qadr, kp, kd, limit, is_joint = [], [], [], [], [], [], []
        estimated = []
        for i in range(m.nu):
            trn = m.actuator_trntype[i]
            gear = float(m.actuator_gear[i, 0])
            b1, b2 = float(m.actuator_biasprm[i, 1]), float(m.actuator_biasprm[i, 2])
            affine = m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE
            joint = trn == mujoco.mjtTrn.mjTRN_JOINT
            j = int(m.actuator_trnid[i, 0])
            if joint and int(m.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                continue
            # force limit, in the units of ctrl after the rewrite below
            scale = abs(gear) if joint else 1.0          # joint motors: ctrl is joint torque = force * gear
            if joint and m.jnt_actfrclimited[j]:
                lim = float(m.jnt_actfrcrange[j, 1])
            elif m.actuator_forcelimited[i]:
                lim = float(np.abs(m.actuator_forcerange[i]).max() * scale)
            elif not affine and m.actuator_ctrllimited[i]:
                lim = float(np.abs(m.actuator_ctrlrange[i]).max() * abs(m.actuator_gainprm[i, 0]) * scale)
            elif affine and b1 < 0 and m.actuator_ctrllimited[i]:
                # position servo with no declared limit: it can never exceed kp x (setpoint span)
                span = float(m.actuator_ctrlrange[i, 1] - m.actuator_ctrlrange[i, 0])
                lim = float(-b1 * span * scale)
                estimated.append(i)
            else:
                lim = float("inf")                       # reported as an error by the analysis
            m.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_gainprm[i, :] = 0
            m.actuator_gainprm[i, 0] = (1.0 / gear if gear else 1.0) if joint else 1.0
            m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
            m.actuator_biasprm[i, :] = 0
            m.actuator_dyntype[i] = mujoco.mjtDyn.mjDYN_NONE
            m.actuator_forcelimited[i] = 0               # the limit is enforced on ctrl
            if math.isfinite(lim):
                lo = 0.0 if (not joint and m.actuator_ctrllimited[i] and m.actuator_ctrlrange[i, 0] >= 0) else -lim
                m.actuator_ctrllimited[i] = 1
                m.actuator_ctrlrange[i] = (lo, lim)      # one-way actuators (propellers, cables) stay one-way
            else:
                m.actuator_ctrllimited[i] = 0
            ids.append(i); is_joint.append(joint)
            dof.append(int(m.jnt_dofadr[j]) if joint else -1)
            qadr.append(int(m.jnt_qposadr[j]) if joint else -1)
            kp.append(-b1 * gear if joint and affine and b1 < 0 else 0.0)
            kd.append(-b2 * gear if joint and affine and b2 < 0 else 0.0)
            limit.append(lim)
        self.ids = np.array(ids, int)
        self.is_joint = np.array(is_joint, bool)
        self.dof, self.qadr = np.array(dof, int), np.array(qadr, int)
        self.kp, self.kd = np.array(kp, float), np.array(kd, float)
        self.limit = np.array(limit, float)
        self.lower = np.array([m.actuator_ctrlrange[i, 0] if m.actuator_ctrllimited[i] else -np.inf for i in ids])
        self.upper = np.array([m.actuator_ctrlrange[i, 1] if m.actuator_ctrllimited[i] else np.inf for i in ids])
        self.joint_idx = np.where(self.is_joint)[0]
        self.estimated_limits = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in estimated]

    def S(self, d):
        """nv x n_inputs map from inputs to generalized forces at the current configuration."""
        m = self.m
        S = np.zeros((m.nv, len(self.ids)))
        moment = np.zeros((m.nu, m.nv))
        mujoco.mju_sparse2dense(moment, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
        for k, i in enumerate(self.ids):
            if self.is_joint[k]:
                S[self.dof[k], k] = 1.0
            else:
                S[:, k] = moment[i]
        return S

    def command(self, d, q_des=None, dq_des=None, tau_ff=None, kp=None, kd=None):
        """LowCmd for joint motors (target angle/speed, gains, feed-forward); force for the rest."""
        u = np.zeros(len(self.ids)) if tau_ff is None else np.array(tau_ff, float)
        J = self.joint_idx
        if q_des is not None:
            kp = self.kp[J] if kp is None else np.broadcast_to(kp, J.shape)
            kd = self.kd[J] if kd is None else np.broadcast_to(kd, J.shape)
            q, dq = d.qpos[self.qadr[J]], d.qvel[self.dof[J]]
            u[J] += kp * (np.asarray(q_des) - q) + kd * ((0 if dq_des is None else dq_des) - dq)
        d.ctrl[self.ids] = np.clip(u, self.lower, self.upper)
        return d.ctrl[self.ids]

    def applied(self, d):
        """Force each input actually applies after MuJoCo's limit clamp (joint: at its joint)."""
        out = d.actuator_force[self.ids].copy()
        out[self.joint_idx] = d.qfrc_actuator[self.dof[self.joint_idx]]
        return out


# ----------------------------------------------------------------------------------- helpers

def _chain_dofs(m, body, roots):
    """Joint dofs between a hand and the robot's root (the arm, plus any waist joints)."""
    n, x = 0, body
    while x not in roots and x != 0:
        n += sum(1 for j in range(m.body_jntadr[x], m.body_jntadr[x] + m.body_jntnum[x])
                 if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)))
        x = int(m.body_parentid[x])
    return n


def pyramid(frame, mu):
    """4 friction-pyramid edge directions (world) for a contact with normal frame[0]."""
    n, t1, t2 = frame[0], frame[1], frame[2]
    E = np.stack([n + mu * t1, n - mu * t1, n + mu * t2, n - mu * t2], axis=1)
    return E / np.linalg.norm(E, axis=0)


def quat_err(q_cur, q_des):
    """World-frame rotation vector taking q_cur to q_des."""
    v = np.zeros(3)
    mujoco.mju_subQuat(v, q_des, q_cur)          # body-frame error
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q_cur)
    return R.reshape(3, 3) @ v


# ----------------------------------------------------------------------------------- WBC

class WholeBodyController:
    def __init__(self, m: mujoco.MjModel, report, motors: MotorLayer, d0: mujoco.MjData):
        self.m, self.motors = m, motors
        self.nv, self.nt = m.nv, len(motors.ids)
        roots = robot_roots(m)
        self.root = roots[0]
        self.robot = {b for r in roots for b in _subtree_raw(m, r)}
        self.floating = report.base == "floating"
        # balance tasks are physical only for robots that can fall over their support: legged robots
        # (CoM over the feet) and flying ones (hover). A wheeled base rests statically on its wheels.
        self.balance = self.floating and (bool(report.feet) or report.robot_class == "aerial")
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.feet = [bid(n) for n in report.feet]
        self.ees = [bid(n) for n in report.end_effectors]
        # hand orientation is only held where the arm has the joints for it (6 for full pose)
        self.ee_dof = {b: _chain_dofs(m, b, roots) for b in self.ees}
        # nominal posture + targets captured from the starting state
        self.qpos0 = d0.qpos.copy()
        self.posture_dofs = np.array([int(m.jnt_dofadr[j]) + k for j in range(m.njnt)
                                      if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
                                      and m.jnt_bodyid[j] in self.robot
                                      for k in range(3 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1)], int)
        # posture only where a motor can act (directly or through a tendon): passive joints such as
        # casters or compliant fingertip pads cannot be commanded, and their errors must not leak
        # into the other joints through the null-space projection
        S0 = motors.S(d0)
        driven = np.abs(S0).sum(axis=1) > 1e-9
        self.posture_dofs = self.posture_dofs[driven[self.posture_dofs]]
        self.com0 = d0.subtree_com[self.root].copy()
        self.base_quat0 = d0.xquat[self.root].copy()
        self.base_pos_target = d0.xpos[self.root].copy()
        pts = getattr(report, "end_effector_point", {}) or {}
        self.ee_off = {bid(n): np.asarray(pts.get(n, [0.0, 0.0, 0.0]), float) for n in report.end_effectors}
        self.ee_targets = {b: self.ee_pos(d0, b) for b in self.ees}
        self.ee_rot = {b: d0.xquat[b].copy() for b in self.ees}
        # tracked orientation reference: turns towards ee_rot at a bounded rate (when set), so a
        # new orientation goal never asks for an instant flip (which the controller would make
        # with the strongest joints it has, e.g. by throwing the torso)
        self.ee_rot_ref = {b: d0.xquat[b].copy() for b in self.ees}
        # Motion-program hand tracking, off by default so operator teleoperation keeps its
        # validated behaviour: orientation slew rate (rad/s), the hand error that may pull (m),
        # and feeding the reference velocity/acceleration forward. byor.moves turns them on.
        self.ee_rot_wmax = np.inf
        self.ee_leash = np.inf
        self.ee_feedforward = False
        # Loads hanging on a hand (an object it holds): {hand body: force on the hand (N, world)}.
        # The plan then holds them up instead of sagging until the position error balances them;
        # a motion program keeps it to what a wrist force sensor would read (zero once set down).
        self.payload = {}
        self.ee_goal = {b: self.ee_pos(d0, b) for b in self.ees}
        self.ee_ref_vel = {b: np.zeros(3) for b in self.ees}
        self.ee_ref_acc = {b: np.zeros(3) for b in self.ees}
        self.ref_vmax, self.ref_amax = 0.5, 2.0          # m/s, m/s^2 for hand references
        self.foot_anchor = {f: d0.xpos[f].copy() for f in self.feet}
        self.foot_quat0 = {f: d0.xquat[f].copy() for f in self.feet}
        self.com_xy_target, self.support_points = (self._support_polygon(d0) if self.feet
                                                   else (self.com0[:2].copy(), np.zeros((0, 2))))
        # Whole-body references a motion program may move: centre of mass (sink into a stance,
        # shift weight onto one foot) and feet (lift, swing, set down). Like the hand references
        # they slew with bounded speed and acceleration, so a new goal never becomes an impulse.
        self.com_goal = np.r_[self.com_xy_target, self.com0[2]]
        self.com_ref, self.com_ref_vel = self.com_goal.copy(), np.zeros(3)
        self.com_vmax, self.com_amax = 0.25, 1.0
        self.foot_goal = {f: p.copy() for f, p in self.foot_anchor.items()}
        self.foot_ref_vel = {f: np.zeros(3) for f in self.feet}
        self.foot_vmax, self.foot_amax = 0.6, 3.0
        # Feed-forward of a planned whole-body motion (zero unless a reference sets them):
        # CoM and foot accelerations, base angular velocity / acceleration (world), joint
        # velocities / accelerations for the posture task.
        self.com_ref_acc = np.zeros(3)
        self.foot_ref_acc = {f: np.zeros(3) for f in self.feet}
        self.base_ref_w, self.base_ref_dw = np.zeros(3), np.zeros(3)
        self.q_ref_vel, self.q_ref_acc = np.zeros(m.nv), np.zeros(m.nv)
        # Optional reference source, called as reference(controller, d) once per step in place
        # of the bounded-rate goal slews: it sets every reference above from a precomputed
        # dynamically consistent motion (byor.choreo). None for teleoperation.
        self.reference = None
        self.prev = {}
        self.dt = m.opt.timestep
        self.limited = [(int(m.jnt_dofadr[j]), int(m.jnt_qposadr[j]), m.jnt_range[j].copy()) for j in range(m.njnt)
                        if m.jnt_limited[j] and m.jnt_bodyid[j] in self.robot
                        and int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
        self.lim_dof = np.array([x[0] for x in self.limited], int)
        self.lim_q = np.array([x[1] for x in self.limited], int)
        self.lim_lo = np.array([x[2][0] for x in self.limited], float)
        self.lim_hi = np.array([x[2][1] for x in self.limited], float)
        self.weights = dict(dyn=1e5, contact=1e4, com=3e3, base=1e3, foot=3e2, touch=2e2, ee=1e2, ee_rot=10.0, posture=1.0,
                            reg_ddq=1e-4, reg_tau=1e-5, reg_lam=1e-6)
        self.gains = dict(com=(80.0, 18.0), base=(180.0, 26.0), ee=(120.0, 22.0), posture=(60.0, 15.5))
        # a foot in the air (not a support) carries only its own leg: a motion program may track
        # it more stiffly than the balance tasks (same as a planted foot by default)
        self.gains["swing"] = self.gains["base"]
        self.res = None
        self.force_targets = {}  # optional motor-space effort objectives, e.g. gripper preload
        self.last = {}
        self.cop_margin = 0.15
        self.speed_cap = {}                       # dof -> rad/s (or m/s), set by a teleop session
        # deceleration (rad/s^2) the joint-limit damper assumes: gentle for an operator; a planned
        # motion that already stays inside the ranges may brake as hard as its motors can
        self.limit_brake = 30.0
        self.null_damping = 0.002                 # see the posture projector in _solve
        # Optional direct joint-space tracking of qpos0 (not projected). A motion program sets it
        # while following a planned pose: from near-straight knees a centre-of-mass task alone
        # cannot see that bending them lowers the body, and the null-space posture is projected
        # away exactly there. Off (0) for teleoperation.
        self.pose_weight = 0.0
        # Optional upper-body (chest) orientation task, relative to the base target: set by a
        # motion program so hand tasks are done with the arms, not by bending the waist.
        self.torso_body, self.torso_rel = None, None
        self.weights["torso"] = 3e2
        self._dJ_now = {}
        self._rank = 0
        sup = list(report.feet) + list(getattr(report, "support", []))
        self.support_bodies = {bid(n) for n in sup if bid(n) >= 0}
        robot_dofs = [i for i in range(m.nv) if m.dof_bodyid[i] in self.robot
                      and int(m.jnt_type[m.dof_jntid[i]]) != int(mujoco.mjtJoint.mjJNT_FREE)]
        self.robot_dofs = np.array(robot_dofs, int)
        self.frictionloss = np.where(np.isin(np.arange(m.nv), robot_dofs), m.dof_frictionloss, 0.0)
        self.friction_dofs = np.where(self.frictionloss > 0)[0]

    # ------------------------------------------------------------------ public API
    def ee_pos(self, d, b):
        """World position of a hand's tool point."""
        return d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.ee_off[b]

    def set_ee_target(self, body_id, pos, quat=None):
        """Operator command. The tracked reference moves towards it with bounded speed and
        acceleration (like a robot's trajectory generator), so jumps never become impulses."""
        self.ee_goal[body_id] = np.asarray(pos, float)
        if quat is not None:
            self.ee_rot[body_id] = np.asarray(quat, float)

    def set_com_target(self, xy=None, z=None):
        """Move the centre-of-mass goal: `xy` over the support (weight shift), `z` its height
        (sink into a stance). Unchanged components keep their current goal."""
        if xy is not None:
            self.com_goal[:2] = np.asarray(xy, float)[:2]
        if z is not None:
            self.com_goal[2] = float(z)

    def set_foot_target(self, foot, pos, quat=None):
        """Move a foot's goal (world position of the foot body, optionally its orientation).
        A foot that is still a support stays planted wherever its reference is; lift it with
        set_support(foot, False) before swinging it."""
        self.foot_goal[foot] = np.asarray(pos, float).copy()
        if quat is not None:
            self.foot_quat0[foot] = np.asarray(quat, float).copy()

    def set_support(self, foot, on: bool):
        """Declare whether `foot` carries the robot. A support foot's ground contacts are held
        (no slip, no sink); a swing foot's contacts are only kept from penetrating."""
        if on:
            self.support_bodies.add(foot)
        else:
            self.support_bodies.discard(foot)

    def foot_settled(self, foot, tol=0.005):
        """The foot reference has arrived (the bounded-acceleration slew chatters at a few mm/s
        around its goal, so arrival is judged with a velocity band, not zero)."""
        return (np.linalg.norm(self.foot_goal[foot] - self.foot_anchor[foot]) < tol
                and np.linalg.norm(self.foot_ref_vel[foot]) < 0.02)

    @staticmethod
    def _slew(p, v, goal, vmax, amax, dt):
        """One step of a bounded-speed, bounded-acceleration reference towards `goal`."""
        e = goal - p
        dist = np.linalg.norm(e)
        # speed that can still stop at the goal, capped at vmax
        vdes = (e / dist) * min(vmax, np.sqrt(2 * amax * dist)) if dist > 1e-9 else np.zeros(3)
        dv = vdes - v
        nd = np.linalg.norm(dv)
        if nd > amax * dt:
            dv *= amax * dt / nd
        v = v + dv
        return p + v * dt, v

    def _advance_references(self):
        vmax, amax, dt = self.ref_vmax, self.ref_amax, self.dt
        for b, goal in self.ee_goal.items():
            v0 = self.ee_ref_vel[b]
            self.ee_targets[b], self.ee_ref_vel[b] = self._slew(
                self.ee_targets[b], v0, goal, vmax, amax, dt)
            self.ee_ref_acc[b] = (self.ee_ref_vel[b] - v0) / dt
        for b, goal in self.ee_rot.items():
            turn = np.zeros(3)
            mujoco.mju_subQuat(turn, goal, self.ee_rot_ref[b])   # goal relative to ref, local
            angle = np.linalg.norm(turn)
            if angle > self.ee_rot_wmax * dt:
                turn *= self.ee_rot_wmax * dt / angle
            if angle > 1e-9:
                mujoco.mju_quatIntegrate(self.ee_rot_ref[b], turn, 1.0)
        self.com_ref, self.com_ref_vel = self._slew(
            self.com_ref, self.com_ref_vel, self.com_goal, self.com_vmax, self.com_amax, dt)
        for f, goal in self.foot_goal.items():
            self.foot_anchor[f], self.foot_ref_vel[f] = self._slew(
                self.foot_anchor[f], self.foot_ref_vel[f], goal, self.foot_vmax, self.foot_amax, dt)

    def support_center(self, d):
        """Centre-of-mass ground target: where it started, if that is safely inside the support
        polygon (sole corners / foot points); otherwise the nearest safe point inside it. A motion
        program moves it with set_com_target."""
        return self.com_ref[:2]

    def _support_polygon(self, d0, margin=0.35):
        from scipy.spatial import ConvexHull
        from .stance import foot_contact_points
        pts = []
        for f in self.feet:
            bodies, local, _ = foot_contact_points(self.m, d0, f)
            pts += [d0.xmat[b].reshape(3, 3) @ p + d0.xpos[b] for b, p in zip(bodies, local)]
        pts = np.array(pts)[:, :2]
        c = pts.mean(0)
        shrunk = c + (1 - margin) * (pts - c)
        try:
            hull = ConvexHull(shrunk)
        except Exception:                         # degenerate (collinear) support: use its centre
            return c, pts
        x = self.com0[:2]
        A, b = hull.equations[:, :2], hull.equations[:, 2]
        if np.all(A @ x + b <= 0):
            return x, pts
        # project onto the shrunk polygon (closest point on its boundary)
        poly = shrunk[hull.vertices]
        best, bd = c, np.inf
        for i in range(len(poly)):
            p, q = poly[i], poly[(i + 1) % len(poly)]
            t = np.clip(np.dot(x - p, q - p) / max(np.dot(q - p, q - p), 1e-12), 0, 1)
            y = p + t * (q - p)
            if np.linalg.norm(y - x) < bd:
                best, bd = y, np.linalg.norm(y - x)
        return best, pts

    # ------------------------------------------------------------------ internals
    def _dJ(self, key, J, dq):
        """dJ/dt * dq by finite difference across physics steps (computed once per step, so a
        second solve within the same step sees the same value)."""
        if key in self._dJ_now:
            return self._dJ_now[key]
        prev = self.prev.get(key)
        self.prev[key] = J.copy()
        val = ((J - prev) / self.dt) @ dq if prev is not None else np.zeros(J.shape[0])
        self._dJ_now[key] = val
        return val

    def _contacts(self, d):
        """Contacts between the robot and the world or props: (body, None, point, frame, mu), with
        the force on `body` = +E lambda. Contacts of the robot with itself are left to the physics:
        planning them hands the optimiser free internal forces (a hand pushing on its own hip) that
        the real contact cannot deliver, and it then tips the robot over. MuJoCo still keeps every
        part from passing through every other."""
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            r1, r2 = b1 in self.robot, b2 in self.robot
            if r1 == r2 or c.dist > 0.005:
                continue
            frame = c.frame.reshape(3, 3).copy()        # normal points geom1 -> geom2: pushes geom2
            mu = float(c.friction[0]) * 0.8
            # Movable scene bodies must receive the opposite contact force in the
            # dynamics model. Treating a door or a held object as fixed ground
            # prevents manipulation and invents an external support for the hand.
            other = b1 if r2 else b2
            if self.m.body_dofnum[other] == 0:
                ancestor = other
                while ancestor and self.m.body_dofnum[ancestor] == 0:
                    ancestor = int(self.m.body_parentid[ancestor])
                if ancestor == 0:
                    other = None
            if r2:
                out.append((b2, other, c.pos.copy(), frame, mu))
            else:
                frame[0] = -frame[0]
                out.append((b1, other, c.pos.copy(), frame, mu))
        # centre-of-pressure margin: plan with each body's ground contact points pulled 15% towards
        # their centroid. Any force on the shrunk polygon is realisable by the real one, and the plan
        # never puts the CoP on the very edge, where the foot starts to tip.
        by_body = {}
        for k, c in enumerate(out):
            if c[1] is None:
                by_body.setdefault(c[0], []).append(k)
        for ks in by_body.values():
            if len(ks) > 1:
                cen = np.mean([out[k][2] for k in ks], axis=0)
                for k in ks:
                    b, o, p, fr, mu = out[k]
                    out[k] = (b, o, p + self.cop_margin * (cen - p), fr, mu)
        return out

    def _ddq_limits(self, d, margin=0.005, tc=0.02, recover=0.2):
        """Velocity-damper joint-limit bounds: approaching a limit, the joint may move only as fast
        as it can still stop (at `limit_brake` per s^2) before the safety margin; past the margin
        it returns at a gentle rate. Bounds act on ddq over the time constant `tc`."""
        brake = self.limit_brake
        lo, hi = np.full(self.nv, -1e4), np.full(self.nv, 1e4)
        if not len(self.lim_dof):
            return self._speed_caps(d, lo, hi, tc)
        q, v = d.qpos[self.lim_q], d.qvel[self.lim_dof]
        dh = self.lim_hi - margin - q
        dl = q - (self.lim_lo + margin)
        vmax = np.where(dh > 0, np.sqrt(2 * brake * np.maximum(dh, 0)), dh / recover)
        vmin = np.where(dl > 0, -np.sqrt(2 * brake * np.maximum(dl, 0)), -dl / recover)
        amax, amin = (vmax - v) / tc, (vmin - v) / tc
        clash = amin > amax
        mid = 0.5 * (amin + amax)
        amin, amax = np.where(clash, mid, amin), np.where(clash, mid, amax)
        lo[self.lim_dof] = np.clip(amin, -1e4, 1e4)
        hi[self.lim_dof] = np.clip(amax, -1e4, 1e4)
        return self._speed_caps(d, lo, hi, tc)

    def _speed_caps(self, d, lo, hi, tc):
        """Optional joint speed caps (an operator-safety layer, e.g. collaborative arm speeds)."""
        if not self.speed_cap:
            return lo, hi
        dofs = np.fromiter(self.speed_cap, int)
        cap = np.fromiter(self.speed_cap.values(), float)
        v = d.qvel[dofs]
        lo[dofs] = np.maximum(lo[dofs], (-cap - v) / tc)
        hi[dofs] = np.minimum(hi[dofs], (cap - v) / tc)
        return lo, hi

    def _friction_feedforward(self, d, S, ddq, horizon=0.02, v_eps=0.01):
        """Dry joint friction (frictionloss) is not in the QP's smooth model; real controllers add it
        as feed-forward in the direction each joint is MEANT to move (the planned acceleration, never
        the measured drift, which would turn compensation into positive feedback). Holding still
        needs none: static friction then helps, as on hardware. Friction on joints that follow a
        coupling (gripper fingers) is supplied by the driving input through that coupling."""
        fl = self.frictionloss
        if not fl.any():
            return np.zeros(self.nt)
        tf = fl * np.tanh(ddq * horizon / v_eps)
        Jeq, _ = self._equality_rows(d)
        A = np.hstack([S, Jeq.T]) if len(Jeq) else S
        rows = self.robot_dofs                  # every joint, so coupling reactions at the driver balance
        sol, *_ = np.linalg.lstsq(A[rows], tf[rows], rcond=None)
        return sol[:self.nt]

    def _equality_rows(self, d):
        """Jacobian rows and violations of the model's active equality constraints."""
        m = self.m
        rows = np.where(d.efc_type[:d.nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))[0]
        if not len(rows):
            return np.zeros((0, m.nv)), np.zeros(0)
        if mujoco.mj_isSparse(m):
            J = np.zeros((d.nefc, m.nv))
            mujoco.mju_sparse2dense(J, d.efc_J, d.efc_J_rownnz, d.efc_J_rowadr, d.efc_J_colind)
        else:
            J = d.efc_J[:d.nefc * m.nv].reshape(d.nefc, m.nv)
        return J[rows].copy(), d.efc_pos[rows].copy()

    def step(self, d):
        """One control step: returns the force/torque for every input (MotorLayer order).

        Contacts are one-sided. Support contacts (feet, wheels, a base on its caster) are always
        held. Any other contact with the world (a hand on a table or a wall) is first left
        out; if that plan would push the two surfaces into each other, the contact is enforced and
        the problem solved again, so surfaces can always separate but never pass through."""
        nv = self.nv
        self._dJ_now = {}
        if self.reference is not None:
            self.reference(self, d)
        else:
            self._advance_references()
        contacts = self._contacts(d)
        held = [c for c in contacts if c[1] is None and c[0] in self.support_bodies]
        optional = [c for c in contacts if not (c[1] is None and c[0] in self.support_bodies)]
        x, S = self._solve(d, held)
        if optional:
            jacp, jacq = np.zeros((3, nv)), np.zeros((3, nv))
            active = []
            for c in optional:
                body, other, p, frame, _ = c
                mujoco.mj_jac(self.m, d, jacp, None, p, body)
                J = jacp.copy()
                if other is not None:
                    mujoco.mj_jac(self.m, d, jacq, None, p, other)
                    J -= jacq
                if frame[0] @ (J @ x[:nv]) < -1e-3:             # planned to push into the surface
                    active.append(c)
            if active:
                x, S = self._solve(d, held + active)
                contacts = held + active
            else:
                contacts = held
        else:
            contacts = held
        nt, nl = self.nt, 4 * len(contacts)
        tau = x[nv:nv + nt] + self._friction_feedforward(d, S, x[:nv])
        self.last = dict(rank=self._rank, n_contacts=len(contacts), tau=tau, lam=x[nv + nt:nv + nt + nl])
        return tau

    def _solve(self, d, contacts):
        m, nv, nt = self.m, self.nv, self.nt
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(m, d, M)
        h = d.qfrc_bias - d.qfrc_passive
        if self.payload:                          # held loads: M ddq + h = S tau + Jc' lam + Jh' F
            jac_l = np.zeros((3, nv))
            for b, F in self.payload.items():
                mujoco.mj_jac(m, d, jac_l, None, self.ee_pos(d, b), b)
                h = h - jac_l.T @ np.asarray(F, float)
        dq = d.qvel.copy()
        nl = 4 * len(contacts)
        Jeq, eq_pos = self._equality_rows(d)
        ne = len(eq_pos)
        n = nv + nt + nl + ne
        H = np.zeros((n, n))
        g = np.zeros(n)

        primary = []

        def add(A, b, w, main=True):
            # residual A ddq - b (tasks act on accelerations only); main tasks also define the null
            # space the posture lives in
            if main:
                primary.append(A)
            H[:nv, :nv] += w * (A.T @ A)
            g[:nv] -= w * (A.T @ b)

        # equations of motion:  M ddq + h - S tau - Jc' E lam = 0
        A_dyn = np.zeros((nv, n))
        A_dyn[:, :nv] = M
        S = self.motors.S(d)
        A_dyn[:, nv:nv + nt] = -S
        jacp, jacr = np.zeros((3, nv)), np.zeros((3, nv))
        cJ = []
        jacq = np.zeros((3, nv))
        for k, (body, other, p, frame, mu) in enumerate(contacts):
            mujoco.mj_jac(m, d, jacp, None, p, body)
            J = jacp.copy()
            if other is not None:                 # self-contact: relative motion, opposite force
                mujoco.mj_jac(m, d, jacq, None, p, other)
                J -= jacq
            E = pyramid(frame, mu)
            A_dyn[:, nv + nt + 4 * k: nv + nt + 4 * k + 4] = -(J.T @ E)
            cJ.append((body, J, p))
        if ne:                                    # coupled joints / loop closures / welds: their force is free
            A_dyn[:, nv + nt + nl:] = -Jeq.T
        H += self.weights["dyn"] * A_dyn.T @ A_dyn
        g -= self.weights["dyn"] * A_dyn.T @ (-h)
        if ne:                                    # ...and the planned motion keeps them closed
            kp, kd = self.gains["base"]
            add(Jeq, -kp * eq_pos - kd * (Jeq @ dq) - self._dJ(("eq", ne), Jeq, dq), self.weights["dyn"])

        # contact points neither slip nor sink (for self-contacts: relative to each other). Support
        # contacts outrank balance; any other contact (hand on hip, hand on table) ranks below it, so
        # a hand pressed against something can never drag the robot over
        for (body, other, _, _, _), (_, J, _) in zip(contacts, cJ):
            support = other is None and body in self.support_bodies
            add(J, np.zeros(3), self.weights["contact"] if support else self.weights["touch"])
        if self.floating:
            # support feet stay planted and flat where they were set down (also while unloaded,
            # so a foot that rocks off the ground is brought back rather than left hanging)
            for f in self.feet:
                kp, kd = self.gains["base" if f in self.support_bodies else "swing"]
                mujoco.mj_jacBody(m, d, jacp, jacr, f)
                e = self.foot_anchor[f] - d.xpos[f]
                if np.linalg.norm(e) > 0.04:      # a foot lagging its reference pulls with
                    e *= 0.04 / np.linalg.norm(e)  # bounded effort, never at the cost of balance
                a = kp * e + kd * (self.foot_ref_vel[f] - jacp @ dq) + self.foot_ref_acc[f]
                add(jacp.copy(), a - self._dJ(("footp", f), jacp, dq), self.weights["foot"])
                a = kp * quat_err(d.xquat[f], self.foot_quat0[f]) - kd * (jacr @ dq)
                add(jacr.copy(), a - self._dJ(("foot", f), jacr, dq), self.weights["foot"])
        if self.balance:
            # centre of mass over the support area, at its starting height
            Jc = np.zeros((3, nv))
            mujoco.mj_jacSubtreeCom(m, d, Jc, self.root)
            com = d.subtree_com[self.root]
            target = np.r_[self.support_center(d), self.com_ref[2]]
            kp, kd = self.gains["com"]
            a = kp * (target - com) + kd * (self.com_ref_vel - Jc @ dq) + self.com_ref_acc
            add(Jc, a - self._dJ("com", Jc, dq), self.weights["com"])
            # torso stays upright with its starting heading
            mujoco.mj_jacBody(m, d, None, jacr, self.root)
            kp, kd = self.gains["base"]
            a = (kp * quat_err(d.xquat[self.root], self.base_quat0)
                 + kd * (self.base_ref_w - jacr @ dq) + self.base_ref_dw)
            add(jacr.copy(), a - self._dJ("base", jacr, dq), self.weights["base"])
            if self.torso_body is not None:
                tq = np.zeros(4)
                mujoco.mju_mulQuat(tq, self.base_quat0, self.torso_rel)
                mujoco.mj_jacBody(m, d, None, jacr, self.torso_body)
                a = kp * quat_err(d.xquat[self.torso_body], tq) - kd * (jacr @ dq)
                add(jacr.copy(), a - self._dJ("torso", jacr, dq), self.weights["torso"])
        elif self.floating:
            # a wheeled base goes where it is driven (here: stays put); without this the plan would
            # "move" the base to help the hands, which the wheels and casters do not actually allow
            mujoco.mj_jacBody(m, d, jacp, jacr, self.root)
            kp, kd = self.gains["base"]
            a = kp * (self.base_pos_target - d.xpos[self.root]) - kd * (jacp @ dq)
            add(jacp.copy(), a - self._dJ("basep", jacp, dq), self.weights["base"])
            a = kp * quat_err(d.xquat[self.root], self.base_quat0) - kd * (jacr @ dq)
            add(jacr.copy(), a - self._dJ("base", jacr, dq), self.weights["base"])
        # hands / end-effectors follow their targets
        for b in self.ees:
            p = self.ee_pos(d, b)
            mujoco.mj_jac(m, d, jacp, jacr, p, b)
            kp, kd = self.gains["ee"]
            # track the moving reference: its acceleration and velocity are fed forward, so a
            # fast hand is not held back to the speed at which its own damping balances the error
            e = self.ee_targets[b] - p
            if np.linalg.norm(e) > self.ee_leash:  # a hand held back (blocked, out of reach)
                e *= self.ee_leash / np.linalg.norm(e)  # pulls with bounded effort, not the body
            a = kp * e - kd * (jacp @ dq)
            if self.ee_feedforward:
                a = a + self.ee_ref_acc[b] + kd * self.ee_ref_vel[b]
            add(jacp.copy(), a - self._dJ(("ee", b), jacp, dq), self.weights["ee"])
            if self.ee_dof[b] >= 6:
                a = kp * quat_err(d.xquat[b], self.ee_rot_ref[b]) - kd * (jacr @ dq)
                add(jacr.copy(), a - self._dJ(("eer", b), jacr, dq), self.weights["ee_rot"])
        # posture: every joint drifts back towards its nominal angle, strictly in the null space
        # of the tasks above (it can use spare joints but never pull a hand or the balance off)
        kp, kd = self.gains["posture"]
        err = np.zeros(nv)
        mujoco.mj_differentiatePos(m, err, 1.0, d.qpos, self.qpos0)
        P = self.posture_dofs
        a_full = np.zeros(nv)
        a_full[P] = kp * err[P] + kd * (self.q_ref_vel[P] - dq[P]) + self.q_ref_acc[P]
        if ne:
            # joints that only follow a coupling (gripper fingers) move with their driver
            F = np.setdiff1d(np.nonzero(np.abs(Jeq).sum(axis=0) > 1e-9)[0], P)
            F = F[np.isin(F, self.robot_dofs)]
            if len(F):
                a_full[F] = -np.linalg.pinv(Jeq[:, F]) @ (Jeq[:, P] @ a_full[P])
        if primary:
            # damped projector: a direction is taken from the posture only in proportion to how much
            # it moves a primary task (s^2 / (s^2 + damping^2)), so a joint that barely touches one
            # (a head barely moving the CoM) keeps its posture instead of being left uncontrolled
            Jh = np.vstack(primary)
            N = np.eye(nv) - Jh.T @ np.linalg.solve(Jh @ Jh.T + self.null_damping ** 2 * np.eye(len(Jh)), Jh)
        else:
            N = np.eye(nv)
        DN = N[P]
        add(DN, DN @ a_full, self.weights["posture"], main=False)
        if np.any(np.asarray(self.pose_weight) > 0):
            # scalar, or one weight per posture dof (a motion program may hold the arms to a
            # planned pose more firmly than legs that the balance tasks already place)
            sw = np.sqrt(np.broadcast_to(np.asarray(self.pose_weight, float), (len(P),)))
            A = np.zeros((len(P), nv))
            A[np.arange(len(P)), P] = sw
            add(A, sw * a_full[P], 1.0, main=False)
        # regularisation
        H[np.arange(nv), np.arange(nv)] += self.weights["reg_ddq"]
        H[nv + np.arange(nt), nv + np.arange(nt)] += self.weights["reg_tau"]
        # A closed gripper needs preload even when contact constraints prohibit
        # further finger acceleration. A posture target in the null space alone
        # cannot request that squeezing force. Keep it inside the same bounded
        # dynamics optimisation so the rest of the body reacts to the load.
        for k, target in self.force_targets.items():
            H[nv + k, nv + k] += 50.0
            g[nv + k] -= 50.0 * target
        if nl + ne:
            H[nv + nt + np.arange(nl + ne), nv + nt + np.arange(nl + ne)] += self.weights["reg_lam"]
        qlo, qhi = self._ddq_limits(d)
        lo = np.r_[qlo, self.motors.lower, np.zeros(nl), np.full(ne, -1e6)]
        hi = np.r_[qhi, self.motors.upper, np.full(nl, 1e5), np.full(ne, 1e6)]
        lo = np.where(np.isfinite(lo), lo, -1e6)
        hi = np.where(np.isfinite(hi), hi, 1e6)
        hi = np.maximum(hi, lo + 1e-9)            # the solver needs strictly ordered bounds
        # scale for conditioning
        s = 1.0 / np.sqrt(np.maximum(np.diag(H), 1e-9))
        Hs = H * s[:, None] * s[None, :]
        gs = g * s
        x0 = np.zeros(n) if self.res is None or len(self.res) != n else self.res / s
        res = x0.copy()
        R = np.zeros((n, n + 7))
        rank = mujoco.mju_boxQP(res, R, None, Hs, gs, lo / s, hi / s)
        x = res * s
        self.res = x
        self._rank = rank
        return x, S
