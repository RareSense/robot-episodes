"""Last step of the tea film: the RIGHT hand takes the full mug (object_0316, 0.48 kg with the tea) by its handle,
lifts it ~5 cm, the robot turns left and walks ~1 m along the counter toward -x carrying it in front of the body,
then stands still holding it.

Physically simulated: MuJoCo 600 Hz, every G1 motor torque-limited (byor MotorLayer). Standing work = byor whole-body
controller (reach / move_hand / guarded finger closing) with payload compensation from the hand-mug contact forces
(what a wrist force sensor reads). Walking = the walking network's leg targets through the motors, commanded only with
velocity commands from a planned speed profile with position/heading feedback (mugfast's follower); the carrying right
arm keeps its pose through the walking controller's PD set-points (plus the gravity sag of the arm and the mug, no
torque added), its fingers keep squeezing with the torque they had when standing. The mug is a free body held only by
finger contact: a three-point pinch of the handle - thumb on the near face, index and middle curled round onto the
far face (found by grip_kin2.py / grip_kin3.py, kinematics on the real collision shapes). Nothing writes qpos/qvel;
no welds; no external forces; no limits raised; friction as in the model.

run(p, s, **kw) -> dict of measured results ('ok' = every gate passed). Starts from the robot standing near
(1.86, 1.90) facing +y with the mug at (1.72, 2.24) handle +x (plans from the MEASURED stance and mug pose); leaves the
robot standing at about (0.9, 1.85) facing -x, the mug held in front of it.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from motion import quat_from_axes, quat_mul, smooth
from nova_robotics.byor.moves import Moves
from nova_robotics.byor.skills import Tasks
from tea_sim import MUG

R = 1                                             # right hand
# the pinch (grip_kin2.py): hand frame tool point on the handle plate's centre plane, E in from the handle's outer
# edge, at the handle's mid-height; the handshake orientation rolled PHI about x_h (palm turned a little down: keeps
# the palm's lower edge 13 mm off the counter top)
PHI_DEG, TX, TY, E = -10.0, 0.13, 0.075, 0.012
H_MID = 0.044                                     # handle mid-height above the mug's base (mug frame)
OUT_EDGE = 0.0545                                 # handle's outer edge from the mug axis (mug frame -y)
PRE = {"thumb_0": 0.5, "thumb_1": 0.45, "thumb_2": 0.65, "index": 0.0, "middle": 0.0}   # thumb >= 4 mm off, fingers straight
THUMB_SHUT = {"thumb_1": 0.75}                    # thumb closes (t1) until it meets the near face
FINGERS_SHUT = 0.85                               # index / middle close until each meets the far face (touch ~0.55)


# right-arm via poses (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw) for the reach from the hanging arm
T1_POSE = (0.2, -1.5, 0.0, 1.28, 0.0, 0.0, 0.0)          # arm out to the side
S1_POSE = (-0.3, -1.5, 0.4, -0.5, 0.3, 0.4, -0.6)        # forearm forward at shoulder height, over the tray's wall


def chaikin(pts, it=3):
    """Corner-cut a joint-space polyline (keeps both ends): a smooth path through the via poses."""
    P = [np.asarray(x, float) for x in pts]
    for _ in range(it):
        Q_ = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            Q_ += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        Q_.append(P[-1])
        P = Q_
    return np.array(P)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _dsmooth(u):
    u = min(max(u, 0.0), 1.0)
    return 30 * u * u * (1 - u) * (1 - u)


def rotx(a):
    return np.array([math.cos(a / 2), math.sin(a / 2), 0.0, 0.0])


class CupSkill:
    def __init__(self, p, s, **kw):
        self.p, self.s, self.kw = p, s, kw
        m, d = s.m, s.d
        self.m, self.d = m, d
        self.b = s.hands[R]
        self.mb = m.body("env/" + MUG + "_body").id
        self.robot = {i for i in range(1, m.nbody) if not m.body(i).name.startswith("env/")}
        self.HAND = s.hand_parts[self.b]
        self.WEIGHT = float(m.body_subtreemass[self.mb]) * 9.81
        self.hold_on = False
        self.est = np.zeros(3)
        self.rel0 = None
        self.stats = dict(slip_max=0.0, slip_walk_max=0.0, tilt_max=0.0, tilt_walk_max=0.0, z_min_held=9.0)
        self.phase = {}
        self.qa = s.loco.qa
        self.arm_dofs = sorted(int(m.jnt_dofadr[j]) for x in s._arm_bodies(self.b)
                               for j in range(m.body_jntadr[x], m.body_jntadr[x] + m.body_jntnum[x]))
        self.arm_q = np.array([int(m.jnt_qposadr[int(m.dof_jntid[v])]) for v in self.arm_dofs], int)
        self.fnames = {a: m.joint(int(np.nonzero(m.jnt_qposadr == a)[0][0])).name for a, *_ in s.fingers[self.b]}
        self.arm_bodies = {i for i in self.robot if m.body(i).name.startswith("right_")
                           and any(k in m.body(i).name for k in ("shoulder", "elbow", "wrist", "hand"))}
        self.on_tick0 = p.mv.on_tick
        p.mv.on_tick = self.payload_tick

    # ------------------------------------------------------------------ measurements
    def mark(self, name):
        self.phase[name] = round(float(self.d.time), 2)

    def contact_force(self, part=None, other=None):
        """Total normal contact force (N) between right-hand bodies (name containing `part`) and the mug, and the
        deepest of those contacts (mm)."""
        m, d = self.m, self.d
        f, deep, buf = 0.0, 0.0, np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for a, o in ((b1, b2), (b2, b1)):
                if a in self.HAND and o == self.mb and (part is None or part in m.body(a).name):
                    mujoco.mj_contactForce(m, d, i, buf)
                    f += abs(buf[0])
                    deep = max(deep, -c.dist * 1000)
        return f, deep

    def support(self):
        """Upward force (N) the mug gets from anything that is not the robot."""
        m, d, mb = self.m, self.d, self.mb
        f, buf = 0.0, np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 == mb) == (b2 == mb) or b1 in self.robot or b2 in self.robot:
                continue
            mujoco.mj_contactForce(m, d, i, buf)
            f += abs(buf[0] * c.frame[2])
        return f

    def hand_load(self):
        """Force (N, world) the mug puts on the hand: the sum of their contact forces (wrist F/T sensor)."""
        m, d = self.m, self.d
        F, buf = np.zeros(3), np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b2 in self.HAND and b1 == self.mb:
                sign = 1.0
            elif b1 in self.HAND and b2 == self.mb:
                sign = -1.0
            else:
                continue
            mujoco.mj_contactForce(m, d, i, buf)
            F += sign * (c.frame.reshape(3, 3).T @ buf[:3])
        return F

    def rel(self):
        d, b = self.d, self.b
        return d.xmat[b].reshape(3, 3).T @ (d.xpos[self.mb] - d.xpos[b])

    def slip(self):
        return 0.0 if self.rel0 is None else float(np.linalg.norm(self.rel() - self.rel0)) * 1000

    def tilt(self):
        return float(np.degrees(np.arccos(np.clip(self.d.xmat[self.mb][8], -1, 1))))

    def mug_yaw(self):
        q = self.d.xquat[self.mb]
        return math.degrees(2 * math.atan2(q[3], q[0]))

    def track(self, walking=False):
        st = self.stats
        t = self.tilt()
        st["tilt_max"] = max(st["tilt_max"], t)
        if walking:
            st["tilt_walk_max"] = max(st["tilt_walk_max"], t)
        if self.hold_on and self.rel0 is not None:
            sl = self.slip()
            st["slip_max"] = max(st["slip_max"], sl)
            st["z_min_held"] = min(st["z_min_held"], float(self.d.xpos[self.mb][2]))
            if walking:
                st["slip_walk_max"] = max(st["slip_walk_max"], sl)

    def payload_tick(self, _s):
        """The controller plans for the mug in the hand: the force it puts on the hand, filtered 0.1 s."""
        s = self.s
        F = self.hand_load() if self.hold_on else np.zeros(3)
        F = np.array([0.0, 0.0, min(F[2], 0.0)])
        self.est += (F - self.est) * min(1.0, 0.02 / 0.1)
        if np.linalg.norm(self.est) > 0.05:
            s.wbc.payload[self.b] = self.est.copy()
        else:
            s.wbc.payload.pop(self.b, None)
        self.track()

    def note(self, what, **kv):
        self.p.note(what, **kv)

    # ------------------------------------------------------------------ fingers
    def targets(self, pose):
        out = {}
        for a, lo, hi, high in self.s.fingers[self.b]:
            c = next((v for k, v in pose.items() if k in self.fnames[a]), None)
            if c is not None:
                out[a] = lo + (c if high else 1 - c) * (hi - lo)
        return out

    def ramp_fingers(self, pose, seconds):
        """Only the named finger joints move (minimum-jerk) to their closures; the others keep their targets."""
        w, goal = self.p.w, self.targets(pose)
        start = {a: float(w.qpos0[a]) for a in goal}
        self.s.grip_active.add(self.b)
        n = max(1, int(round(seconds / self.p.mv.TICK)))
        for k in range(1, n + 1):
            u = smooth(k / n)
            for a in goal:
                w.qpos0[a] = start[a] + u * (goal[a] - start[a])
            self.p.mv._tick(self.p.mv.TICK)

    def close_to_contact(self, groups, seconds, f_touch=0.4):
        """Each group (name -> (pose, contact part)) ramps its finger targets toward `pose` and stops where its own
        contact with the mug turns on (a force-limited gripper closing to contact). Returns {group: closure frac}."""
        w, p = self.p.w, self.p
        n = max(1, int(round(seconds / p.mv.TICK)))
        goals = {g: self.targets(pose) for g, (pose, _) in groups.items()}
        starts = {g: {a: float(w.qpos0[a]) for a in goals[g]} for g in groups}
        done = {}
        self.s.grip_active.add(self.b)
        for k in range(1, n + 1):
            u = (k / n) ** 1.0
            u = u * u * (3 - 2 * u)                         # smoothstep: no jolt at the start
            for g, (pose, part) in groups.items():
                if g in done:
                    continue
                for a in goals[g]:
                    w.qpos0[a] = starts[g][a] + u * (goals[g][a] - starts[g][a])
            p.mv._tick(p.mv.TICK)
            for g, (pose, part) in groups.items():
                if g not in done and self.contact_force(part)[0] > f_touch:
                    done[g] = round(u, 3)
            if len(done) == len(groups):
                break
        return done

    def squeeze(self, extra, seconds):
        """Finger targets on past where the fingers now are by `extra` (closure units) - the squeeze."""
        w, p = self.p.w, self.p
        goal = {}
        for a, lo, hi, high in self.s.fingers[self.b]:
            c = next((v for k, v in extra.items() if k in self.fnames[a]), None)
            if c is not None:
                goal[a] = float(self.d.qpos[a]) + (1 if high else -1) * c * (hi - lo)
        start = {a: float(w.qpos0[a]) for a in goal}
        n = max(1, int(round(seconds / p.mv.TICK)))
        for k in range(1, n + 1):
            u = smooth(k / n)
            for a in goal:
                w.qpos0[a] = start[a] + u * (goal[a] - start[a])
            p.mv._tick(p.mv.TICK)

    # ------------------------------------------------------------------ arm paths (controller references)
    def elbow_goal(self):
        """Where the right elbow should sit (out to the side and a little forward of the body): the arm's one
        redundant freedom is resolved toward it. Without it the shoulder rolls the upper arm into the torso for a
        reach across the body (IK poses 45 mm into torso_link), and the controller ends in a contorted pose."""
        yaw = self.yaw()
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
        g = self.d.qpos[:3] + right * self.kw.get("elbow_right", 0.29) + fwd * self.kw.get("elbow_fwd", 0.05)
        g[2] = self.kw.get("elbow_z", 0.95)
        return g

    def ik_arm(self, sd, tgt, quat, tool, q_seed, iters=60, wrot=0.3, elbow=None):
        """Right-arm joint angles putting the hand's `tool` point at `tgt` with orientation `quat` (scratch data `sd`,
        body as it stands now), the elbow pulled toward `elbow` in the null space. Kinematics only."""
        m, b = self.m, self.b
        cols = self.arm_dofs
        lo = np.array([m.jnt_range[int(m.dof_jntid[v])][0] for v in cols]) + 0.08
        hi = np.array([m.jnt_range[int(m.dof_jntid[v])][1] for v in cols]) - 0.08
        el = m.body("right_elbow_link").id
        jacp, jacr, jace = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros((3, m.nv))
        sd.qpos[self.arm_q] = q_seed
        ep = er = np.zeros(3)
        for _ in range(iters):
            mujoco.mj_kinematics(m, sd)
            mujoco.mj_comPos(m, sd)
            Rh = sd.xmat[b].reshape(3, 3)
            pt = sd.xpos[b] + Rh @ np.asarray(tool, float)
            mujoco.mj_jac(m, sd, jacp, jacr, pt, b)
            ep = np.asarray(tgt, float) - pt
            r = np.zeros(3)
            mujoco.mju_subQuat(r, np.asarray(quat, float), sd.xquat[b])
            er = Rh @ r
            J = np.vstack([jacp[:, cols], wrot * jacr[:, cols]])
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
            dq = Jp @ np.r_[ep, wrot * er]
            if elbow is not None:
                mujoco.mj_jac(m, sd, jace, None, sd.xpos[el], el)
                dq += (np.eye(len(cols)) - Jp @ J) @ (0.5 * jace[:, cols].T @ (elbow - sd.xpos[el]))
            n = float(np.abs(dq).max())
            if n > 0.1:
                dq *= 0.1 / n
            sd.qpos[self.arm_q] = np.clip(sd.qpos[self.arm_q] + dq, lo, hi)
        return sd.qpos[self.arm_q].copy(), float(np.linalg.norm(ep)), float(np.linalg.norm(er))

    def plan_arm(self, p_to, q_to, via=None, n=40, tool=None, p_from=None, r_from=None, q_from=None):
        """Waypoints (arm joint angles) for the hand's tool point from where it is to `p_to` (orientation slerped
        to `q_to`), along a curve bent toward `via`, each solved from the previous one (continuous), elbow out."""
        from motion import path, slerp
        s, d, b = self.s, self.d, self.b
        tool = s.wbc.ee_off[b] if tool is None else np.asarray(tool, float)
        sd = mujoco.MjData(self.m)
        sd.qpos[:] = d.qpos
        p0 = s.wbc.ee_pos(d, b).copy() if p_from is None else np.asarray(p_from, float)
        r0 = d.xquat[b].copy() if r_from is None else np.asarray(r_from, float)
        go = path(p0, via, p_to)
        el_goal = self.elbow_goal()
        el0 = d.xpos[self.m.body("right_elbow_link").id].copy() if q_from is None else None
        q = d.qpos[self.arm_q].copy() if q_from is None else np.asarray(q_from, float).copy()
        if el0 is None:
            sd.qpos[self.arm_q] = q
            mujoco.mj_kinematics(self.m, sd)
            el0 = sd.xpos[self.m.body("right_elbow_link").id].copy()
        qs, errs, hits = [q.copy()], [], []
        self.plan_hits = hits
        arm = self.arm_bodies
        for k in range(1, n + 1):
            u = k / n
            elbow = el0 + smooth(u) * (el_goal - el0)       # the elbow moves out gradually, not in one pull
            q, e, er = self.ik_arm(sd, go(u), slerp(r0, q_to, min(1.0, u)), tool, q, iters=40 if k < n else 200, elbow=elbow)
            qs.append(q.copy())
            errs.append(e)
            mujoco.mj_collision(self.m, sd)                 # what the planned arm would touch (planning check only)
            for c in sd.contact[:sd.ncon]:
                b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
                if (b1 in arm) != (b2 in arm) and c.dist < -0.001:
                    a, o = (b1, b2) if b1 in arm else (b2, b1)
                    if o != self.mb:
                        hits.append((round(u, 2), self.m.body(a).name[-16:], self.m.body(o).name[-16:], round(-c.dist * 1000, 1)))
        if hits:
            self.note("planned arm path touches", n=len(hits), worst=sorted(hits, key=lambda h: -h[3])[:3])
        return np.array(qs), max(errs)

    def follow(self, qs, seconds, tool=None):
        """Move along the planned waypoints: the controller's posture reference for the right arm and its hand target
        (position + orientation = forward kinematics of that reference) advance together, minimum-jerk timing. The
        whole-body controller turns them into torques; nothing is set in the state."""
        s, d, b, p, m = self.s, self.d, self.b, self.p, self.m
        w = p.w
        tool = s.wbc.ee_off[b] if tool is None else np.asarray(tool, float)
        sd = mujoco.MjData(m)
        sd.qpos[:] = d.qpos
        qs = np.asarray(qs, float)
        N = len(qs) - 1
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(qs, axis=0), axis=1))]
        arc = arc / max(arc[-1], 1e-9) * N                  # joint-space arc length, in waypoint units
        # the hand's path length -> duration so that its peak speed (min-jerk: 1.875 L / T) stays under vmax_hand;
        # the controller's hand-reference slew limit (ref_vmax, ours) is set just above that for the move
        pts = []
        for q in qs:
            sd.qpos[self.arm_q] = q
            mujoco.mj_kinematics(m, sd)
            pts.append(sd.xpos[b] + sd.xmat[b].reshape(3, 3) @ tool)
        L = float(np.sum(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1)))
        vh = self.kw.get("vmax_hand", 0.6)
        seconds = max(seconds, 1.875 * L / vh)
        vmax0 = w.ref_vmax
        w.ref_vmax = max(vmax0, 1.3 * vh)
        self.last_follow = dict(path_m=round(L, 3), seconds=round(seconds, 2))
        n = max(1, int(round(seconds / p.mv.TICK)))
        for k in range(1, n + 1):
            u = float(np.interp(smooth(k / n) * N, arc, np.arange(N + 1)))
            i = min(int(u), N - 1)
            q = qs[i] + (u - i) * (qs[i + 1] - qs[i])
            sd.qpos[self.arm_q] = q
            mujoco.mj_kinematics(m, sd)
            pos = sd.xpos[b] + sd.xmat[b].reshape(3, 3) @ tool
            p._ease_arm(R, k * p.mv.TICK)
            w.qpos0[self.arm_q] = q
            w.set_ee_target(b, pos, quat=sd.xquat[b].copy())
            p.mv._tick(p.mv.TICK)
        w.ref_vmax = vmax0
        return float(np.linalg.norm(w.ee_pos(d, b) - pos))

    def level_quat(self):
        """The hand's commanded orientation turned by the mug's measured tilt (as a camera on the robot would see it),
        so the held mug comes back upright."""
        mz = self.d.xmat[self.mb].reshape(3, 3)[:, 2]
        axis = np.cross(mz, [0.0, 0.0, 1.0])
        sa = float(np.linalg.norm(axis))
        cmd = self.p.w.ee_rot[self.b].copy()
        if sa < 1e-6:
            return cmd
        ang = math.atan2(sa, mz[2]) * self.kw.get("level_gain", 0.5)
        return quat_mul(np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis / sa], cmd)

    def level_move(self, target, seconds, chunk=0.1):
        """move_hand to `target` in short pieces; each piece also turns the hand toward levelling the mug."""
        p, b = self.p, self.b
        start = p.w.ee_goal[b].copy()
        n = max(1, int(round(seconds / chunk)))
        for k in range(1, n + 1):
            pt = start + smooth(k / n) * (np.asarray(target, float) - start)
            p.move_hand(R, pt, quat=self.level_quat(), seconds=chunk)

    def level_hold(self, seconds, chunk=0.1):
        p, b = self.p, self.b
        for _ in range(max(1, int(round(seconds / chunk)))):
            p.move_hand(R, p.w.ee_goal[b].copy(), quat=self.level_quat(), seconds=chunk)

    def force_squeeze(self, targets, seconds, gain=0.004, rate=0.6):
        """Squeeze to measured pad forces (Dex3 fingertips carry tactile sensors): each finger group's joint targets
        move (at most `rate` closure/s) until its contact force with the mug reaches its target (N), so the thumb and
        the two fingers push equally hard and the handle is not shoved out of the pinch."""
        w, p = self.p.w, self.p
        groups = {g: [a for a, *_ in self.s.fingers[self.b] if g in self.fnames[a] and (g != "thumb" or "thumb_0" not in self.fnames[a])]
                  for g in targets}
        sign = {a: (1 if high else -1) * (hi - lo) for a, lo, hi, high in self.s.fingers[self.b]}
        n = max(1, int(round(seconds / p.mv.TICK)))
        for _ in range(n):
            for g, ft in targets.items():
                f = self.contact_force(g)[0]
                dc = float(np.clip(gain * (ft - f), -rate * p.mv.TICK, rate * p.mv.TICK))
                for a in groups[g]:
                    w.qpos0[a] += dc * sign[a]
            p.mv._tick(p.mv.TICK)
        return {g: round(self.contact_force(g)[0], 1) for g in targets}

    def finger_closures(self):
        out = {}
        for a, lo, hi, high in self.s.fingers[self.b]:
            c = (self.d.qpos[a] - lo) / (hi - lo)
            out[self.fnames[a].replace("right_hand_", "").replace("_joint", "")] = round(float(c if high else 1 - c), 3)
        return out

    # ------------------------------------------------------------------ walking
    def standing_again(self, finger_q0):
        """After walking: a fresh whole-body controller took balance back; plan from where we stand."""
        s, p = self.s, self.p
        s.set_walk((0.0, 0.0, 0.0))
        stood = Tasks(s)._wait_stand()
        for attr in ("_armw", "_ease0"):
            if hasattr(p, attr):
                delattr(p, attr)
        p.mv, p.w = Moves(s, on_tick=self.payload_tick), s.wbc
        p.root = p.mv.root
        for a, v in finger_q0.items():
            s.wbc.qpos0[a] = v
        s.grip_active.add(self.b)
        return stood

    def yaw(self):
        q = self.d.qpos[self.qa + 3:self.qa + 7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def sag_offset(self, q_arm, load_tool=None):
        """Set-point offset that makes the walking controller's arm PD hold `q_arm` against gravity (the arm's own
        weight plus the mug hanging at `load_tool`, hand frame): offset = tau_gravity / kp, same PD gains."""
        m, b, loco = self.m, self.b, self.s.loco
        sd = mujoco.MjData(m)
        sd.qpos[:] = self.d.qpos
        sd.qpos[self.arm_q] = q_arm
        sd.qvel[:] = 0.0
        mujoco.mj_kinematics(m, sd)
        mujoco.mj_comPos(m, sd)
        tau = np.zeros(m.nv)
        mujoco.mj_rne(m, sd, 0, tau)
        g = tau[self.arm_dofs].copy()
        if load_tool is not None:
            pt = sd.xpos[b] + sd.xmat[b].reshape(3, 3) @ np.asarray(load_tool, float)
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, sd, jacp, None, pt, b)
            g += jacp[:, self.arm_dofs].T @ np.array([0.0, 0.0, self.WEIGHT])
        idx = np.array([int(np.nonzero(loco.hold_q == q)[0][0]) for q in self.arm_q], int)
        return g / loco.hold_kp[idx]

    def drive(self, goal_xy, goal_yaw, *, vmax=0.4, amax=0.4, kp=0.8, kyaw=1.5, tol=0.03, yaw_tol=math.radians(4),
              extra=1.0, warm=0.4, idle=0.05, stop_short=0.0, lead=0.5, turn_frac=1.0, grip_other=0.8, bow=0.0, wmax=0.6, name="walk"):
        """mugfast's follower (agents/mugfast/skill_mug.py MugSkill.drive) for the RIGHT carrying arm: the walking
        network gets velocity commands from a minimum-jerk straight-line profile (heading reaches its goal by
        `turn_frac` of it) plus position and heading feedback. The right arm keeps its pose (PD set-points = where it
        is + the gravity sag of arm and mug); its fingers keep the torque they squeeze with now."""
        s, d, m = self.s, self.d, self.m
        loco = s.loco
        cfg = loco.cfg
        saved_cfg = {k: cfg[k] for k in ("warm_s", "idle_s")}
        cfg["warm_s"], cfg["idle_s"] = warm, idle
        names = [m.joint(int(np.nonzero(m.jnt_qposadr == q)[0][0])).name for q in loco.hold_q]
        hand_i = getattr(self, "hand_index", R)
        keep = np.array([n.startswith(("left_", "right_")[hand_i]) and any(k in n for k in ("shoulder", "elbow", "wrist")) for n in names])
        named0, target0 = loco.hold_named.copy(), loco.hold_target.copy()
        loco.hold_named = loco.hold_named & ~keep
        tool = self.rel() if self.hold_on else None         # the mug's centre, hand frame (if it is held)
        off = self.sag_offset(d.qpos[self.arm_q].copy(), load_tool=tool)
        for q, o in zip(self.arm_q, off):
            i = int(np.nonzero(loco.hold_q == q)[0][0])
            loco.hold_named[i], loco.hold_target[i] = True, float(d.qpos[q] + o)
        # carrying fingers: the walking controller's finger PD (kp_w) set so that it squeezes with the torque the
        # standing controller squeezes with now (kp_s (target - q), clipped to the motor limit)
        w = s.wbc
        finger_q0 = {a: float(w.qpos0[a]) for a, *_ in s.fingers[self.b]}
        for a, lo, hi, high in s.fingers[self.b]:
            k = s._motor_at_qpos[a]
            lim = float(s.motors.limit[k])
            kp_s = 4 * lim / max(hi - lo, 0.01)
            tau = float(np.clip(kp_s * (w.qpos0[a] - d.qpos[a]), -lim, lim))
            i = int(np.nonzero(loco.hold_q == a)[0][0])
            loco.hold_named[i], loco.hold_target[i] = True, float(d.qpos[a] + tau / loco.hold_kp[i])
        if getattr(self, "keep_waist", False):              # the waist holds where it is (no torso swing at the start)
            for i, nm in enumerate(names):
                if nm.startswith("waist"):
                    loco.hold_named[i], loco.hold_target[i] = True, float(d.qpos[int(loco.hold_q[i])])
        if getattr(self, "keep_other_arm", False):          # the free arm stays as it hangs (not the gait's elbow-forward pose)
            other = ("left_", "right_")[1 - hand_i]
            for i, nm in enumerate(names):
                if nm.startswith(other) and any(k in nm for k in ("shoulder", "elbow", "wrist")):
                    loco.hold_named[i], loco.hold_target[i] = True, float(d.qpos[int(loco.hold_q[i])])
        hook = getattr(self, "walk_hook", None)             # e.g. levelling the carried mug through the wrist set-points
        hb = s.hands[1 - hand_i]
        for a, lo, hi, high in s.fingers[hb]:              # the other hand relaxed, curled
            i = int(np.nonzero(loco.hold_q == a)[0][0])
            loco.hold_named[i], loco.hold_target[i] = True, lo + (grip_other if high else 1 - grip_other) * (hi - lo)
        t_start = d.time
        s.set_walk((0.01, 0.0, 0.0))                        # activates the walking controller
        if getattr(self, "no_hold_blend", False):
            # its non-leg PD set-points blend from "where the joints are" to their targets over blend_s; for the held
            # (carrying) joints that means the gravity offset comes in late and the loaded arm sags first (g01: 7 deg
            # tilt spike). Start them at their targets.
            fixed = loco.hold_named & ~named0 | keep
            loco.hold_start = np.where(fixed | np.isin(loco.hold_q, [a for a, *_ in s.fingers[self.b]]), loco.hold_target, loco.hold_start)
        dt, n = m.opt.timestep, loco.period
        vlim = np.asarray(cfg["vmax"], float)
        warm_until = cfg["blend_s"] + cfg["warm_s"]

        def tick(cmd):
            s.set_walk(cmd)
            if hook is not None:
                hook(n * dt)
            for _ in range(n):
                s.step(1)
                self.track(walking=True)
            if s.checks.fell_at is not None:
                raise RuntimeError(f"robot fell at {s.checks.fell_at} s")

        while loco.step_count * dt < warm_until - 0.02:
            tick((0.01, 0.0, 0.0))
        xy0, yaw0 = d.qpos[self.qa:self.qa + 2].copy(), self.yaw()
        aim = np.asarray(goal_xy, float) - stop_short * np.array([math.cos(goal_yaw), math.sin(goal_yaw)])
        D = aim - xy0
        Ld = float(np.linalg.norm(D))
        dyaw = _wrap(goal_yaw - yaw0)
        T = max(1.875 * Ld / vmax, math.sqrt(5.774 * Ld / amax), 1.875 * abs(dyaw) / (wmax * turn_frac), 0.6)

        def ref(tt):
            f = min(max(tt / T, 0.0), 1.0)
            sm, smd = smooth(f), (_dsmooth(f) / T if 0.0 < f < 1.0 else 0.0)
            fy = min(f / turn_frac, 1.0)
            psi = yaw0 + dyaw * smooth(fy)
            wy = dyaw * _dsmooth(fy) / (T * turn_frac) if 0.0 < fy < 1.0 else 0.0
            # `bow`: the path bends that far to the right of the walking direction mid-way (< 0: to the left)
            nrm = np.array([D[1], -D[0]]) / max(Ld, 1e-9)
            pos = xy0 + D * sm + nrm * bow * math.sin(math.pi * sm)
            vel = (D + nrm * bow * math.pi * math.cos(math.pi * sm)) * smd
            return pos, vel, psi, wy

        t_move = d.time
        t = 0.0
        err_max = 0.0
        while True:
            f = min(t / T, 1.0)
            p_ref, _, psi_ref, _ = ref(t)
            _, v_ff, _, w_ff = ref(t + lead)
            xy, yaw = d.qpos[self.qa:self.qa + 2], self.yaw()
            err_max = max(err_max, float(np.linalg.norm(p_ref - xy)))
            vw = v_ff + kp * (p_ref - xy)
            c, sn = math.cos(yaw), math.sin(yaw)
            cmd = np.array([c * vw[0] + sn * vw[1], -sn * vw[0] + c * vw[1], w_ff + kyaw * _wrap(psi_ref - yaw)])
            cmd = np.clip(cmd, -vlim, vlim)
            if f >= 1.0:
                cmd[:2] = np.clip(cmd[:2], -0.12, 0.12)
                close = np.linalg.norm(aim - xy) < tol and abs(_wrap(goal_yaw - yaw)) < yaw_tol
                if close or t > T + extra:
                    break
            if np.all(np.abs(cmd) < 2e-3):
                cmd[0] = 2e-3
            tick(cmd)
            t += n * dt
        t_cmd0 = d.time
        s.set_walk((0.0, 0.0, 0.0))
        k_stop = 0
        while s.mode != "stand" and d.time - t_cmd0 < 8.0:
            if hook is not None and k_stop % n == 0:
                hook(n * dt)
            k_stop += 1
            s.step(1)                                       # one step at a time: the moment the standing controller
            self.track(walking=True)                        # takes over, its finger targets are set back to the grip
            if s.mode == "stand":
                for a, v in finger_q0.items():
                    s.wbc.qpos0[a] = v
            if s.checks.fell_at is not None:
                raise RuntimeError(f"robot fell at {s.checks.fell_at} s")
        t_stand = d.time
        stood = self.standing_again(finger_q0)
        loco.hold_named, loco.hold_target = named0, target0
        cfg.update(saved_cfg)
        out = dict(start=round(t_start, 2), profile_s=round(T, 2), startup_s=round(t_move - t_start, 2),
                   moving_s=round(t_cmd0 - t_move, 2), stop_s=round(t_stand - t_cmd0, 2), stood=stood,
                   track_err_max_m=round(err_max, 3), at=np.round(d.qpos[self.qa:self.qa + 2], 3).tolist(),
                   goal=list(np.round(goal_xy, 3)), heading_deg=round(math.degrees(self.yaw()), 1),
                   goal_heading_deg=round(math.degrees(goal_yaw), 1))
        self.note(name, **out)
        return out

    # ------------------------------------------------------------------ the step
    def run(self):
        p, s, d, m, b, kw = self.p, self.s, self.d, self.m, self.b, self.kw
        res = dict(ok=False)
        self.mark("start")
        others = [int(m.jnt_bodyid[j]) for j in range(m.njnt)
                  if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and int(m.jnt_bodyid[j]) not in self.robot
                  and int(m.jnt_bodyid[j]) != self.mb]
        others0 = {fb: d.xpos[fb].copy() for fb in others}
        res["stance"] = dict(xy=np.round(d.qpos[:2], 3).tolist(), heading_deg=round(math.degrees(self.yaw()), 1))
        # the grasp, planned from the mug's MEASURED pose: the hand frame's base orientation is the mug's own
        # (x across the handle plate, y from the handle into the mug, z up), rolled PHI about x
        Rm = d.xmat[self.mb].reshape(3, 3)
        ex = Rm[:, 0] - Rm[2, 0] * np.array([0, 0, 1.0])
        ex /= np.linalg.norm(ex)
        ez = np.array([0.0, 0.0, 1.0])
        q_base = quat_from_axes(ex, np.cross(ez, ex), ez)
        Q = quat_mul(q_base, rotx(math.radians(kw.get("phi", PHI_DEG))))
        out_dir = -Rm[:, 1]                                 # the handle points this way
        grasp = d.xpos[self.mb] + Rm @ np.array([0.0, -OUT_EDGE + kw.get("e", E), H_MID + kw.get("h", 0.0)])
        res["mug_start"] = dict(xyz=np.round(d.xpos[self.mb], 4).tolist(), handle_yaw_deg=round(self.mug_yaw(), 1))
        p.set_tool(R, (kw.get("tx", TX), kw.get("ty", TY), 0.0))

        # 1. hand to the handle: pre-shape, up over the counter edge to a point 6 cm out along the handle, slide in
        self.mark("reach")
        pre_shape = dict(PRE)
        pre_shape.update(kw.get("pre_shape", {}))
        self.ramp_fingers(pre_shape, kw.get("preshape_s", 0.4))
        # straight down onto the handle from 6 cm above: coming in from the side, the straight fingers would sweep
        # through the jar beside the mug (and the tray's wall); from above the thumb stays >= 4.7 mm off (grip_kin3.py)
        pre = grasp + out_dir * kw.get("slide", 0.0) + np.array([0.0, 0.0, kw.get("pre_up", 0.06)])
        yaw0 = self.yaw()
        fwd0 = np.array([math.cos(yaw0), math.sin(yaw0), 0.0])
        right0 = np.array([math.sin(yaw0), -math.cos(yaw0), 0.0])
        # up in front of the right side of the chest first (the hanging hand would otherwise swing into the cabinet
        # front), then over the counter's edge; the fingers sit 7.5 cm to the right of the tool point, so they pass high
        # over the tray's wall (x >= 1.985, top 0.925): up to a point well above the handle, then down onto it.
        # Every leg of the path: IK waypoints with the elbow out to the side (plan_arm), followed by the controller.
        # The robot stands 15 cm from the counter's edge: a hand raised in front swings into the cabinet front /
        # counter top. So the arm rises out to the side first (T1), the elbow bends the forearm forward at shoulder
        # height (S1), then in over the handle (dbg_mid3.py: >= 12 mm clear of everything along this joint path)
        high = pre + np.array([0.0, 0.0, kw.get("high_up", 0.07)]) - fwd0 * 0.03
        sd = mujoco.MjData(m)
        sd.qpos[:] = d.qpos
        q_seed = np.array(kw.get("ik_seed", (-0.79, 0.0, 0.70, -0.38, 0.73, 0.96, -1.35)))
        el = self.elbow_goal()
        q_high, e1, _ = self.ik_arm(sd, high, Q, s.wbc.ee_off[b], q_seed, iters=400, elbow=el)
        way = [d.qpos[self.arm_q].copy(), np.array(kw.get("t1", T1_POSE)), np.array(kw.get("s1", S1_POSE)), q_high]
        self.follow(chaikin(way, 3), kw.get("reach_s", 2.4))
        e_high = p.settle_hand(R, high, tol=0.01, timeout=kw.get("settle_high", 0.4))
        self.note("over the handle", ik_err_mm=round(e1 * 1000, 1), hand_err_mm=round(e_high * 1000, 1), **self.last_follow)
        qs, e2 = self.plan_arm(pre, Q, n=15)
        self.follow(qs, kw.get("pre_s", 0.6))
        e_pre = p.settle_hand(R, pre, tol=0.005, timeout=kw.get("settle_pre", 0.5))
        qs, e3 = self.plan_arm(grasp, Q, n=15)
        self.follow(qs, kw.get("down_s", 0.8))
        p.settle_hand(R, grasp, tol=0.003, timeout=kw.get("settle_grasp", 0.4))
        # closed loop on the measured tool point (as a camera / the arm's encoders would give): the loaded arm sits
        # several mm off its target, and the pinch needs the handle within ~2 mm of the planned spot
        for _ in range(kw.get("servo_n", 4)):
            err = grasp - s.wbc.ee_pos(d, b)
            if np.linalg.norm(err) < 0.002:
                break
            p.move_hand(R, s.wbc.ee_goal[b] + err, quat=Q, seconds=0.3)
            p.hold(0.15)
        e_grasp = float(np.linalg.norm(grasp - s.wbc.ee_pos(d, b)))
        res["ik_err_mm"] = [round(e * 1000, 1) for e in (e1, e2, e3)]
        pushed = float(np.linalg.norm(d.xpos[self.mb] - np.asarray(res["mug_start"]["xyz"])))
        res["approach"] = dict(pre_err_mm=round(e_pre * 1000, 1), grasp_err_mm=round(e_grasp * 1000, 1),
                               mug_pushed_mm=round(pushed * 1000, 1), touching_N=round(self.contact_force()[0], 2))
        self.note("at the handle", **res["approach"])

        # 2. pinch: thumb onto the near face, then index and middle round onto the far face, then squeeze
        self.mark("grasp")
        th = self.close_to_contact({"thumb": (dict(THUMB_SHUT, **kw.get("thumb_shut", {})), "thumb")},
                                   kw.get("thumb_s", 0.5))
        fi = self.close_to_contact({"index": ({"index": kw.get("fingers_shut", FINGERS_SHUT)}, "index"),
                                    "middle": ({"middle": kw.get("fingers_shut", FINGERS_SHUT)}, "middle")},
                                   kw.get("close_s", 0.8))
        fsq = kw.get("fsq", 12.0)                          # thumb force (N); index and middle half each
        res["squeezed_N"] = self.force_squeeze({"thumb": fsq, "index": fsq / 2, "middle": fsq / 2}, kw.get("squeeze_s", 1.0))
        p.hold(kw.get("grip_hold", 0.2))
        self.hold_on = True
        p.hold(0.1)
        self.rel0 = self.rel()
        f_all, deep = self.contact_force()
        res["grip"] = dict(touch=dict(th, **fi), closures=self.finger_closures(), force_N=round(f_all, 1),
                           deepest_mm=round(deep, 2), parts={k: round(self.contact_force(k)[0], 1) for k in ("thumb", "index", "middle", "wrist")},
                           mug_moved_mm=round(float(np.linalg.norm(d.xpos[self.mb] - np.asarray(res["mug_start"]["xyz"]))) * 1000, 1))
        self.note("pinched", **res["grip"])

        # 3. lift ~5 cm, straight up
        self.mark("lift")
        z0 = float(d.xpos[self.mb][2])
        self.Q = Q
        self.level_move(s.wbc.ee_goal[b] + [0.0, 0.0, kw.get("lift", 0.05)], kw.get("lift_s", 1.5))
        self.level_hold(kw.get("lift_hold", 0.4))
        res["lifted"] = dict(rise_mm=round((float(d.xpos[self.mb][2]) - z0) * 1000, 1), slip_mm=round(self.slip(), 1),
                             tilt_deg=round(self.tilt(), 1), support_N=round(self.support(), 2),
                             load_N=np.round(self.hand_load(), 2).tolist())
        self.note("lifted", **res["lifted"])
        empty = False
        if self.support() > 0.3 * self.WEIGHT or res["lifted"]["rise_mm"] < 20 or self.slip() > 30:
            res.update(failed="grasp: mug not lifted in the hand")
            if not kw.get("walk_anyway"):
                return self.finish(res, others0)
            # development only: exercise the carry pose and the walk with the EMPTY hand (the mug stays behind)
            empty = True
            self.hold_on = False
            self.rel0 = None
            self.ramp_fingers({"index": 0.3, "middle": 0.3, "thumb_1": 0.3, "thumb_2": 0.3}, 0.4)
            p.move_hand(R, s.wbc.ee_goal[b] + [0.0, 0.0, 0.04], seconds=0.5)
        if kw.get("stop_after") == "lift":
            self.level_hold(kw.get("test_hold", 2.0))
            res["held"] = dict(slip_mm=round(self.slip(), 1), tilt_deg=round(self.tilt(), 1))
            return self.finish(res, others0)

        # 4. into the carry pose: in front of the body, close in, body upright (no lean: the walking controller
        #    holds the waist straight)
        self.mark("carry")
        yaw = self.yaw()
        FWD = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        LEFT = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        pel = d.qpos[:3].copy()
        mug_c = pel + FWD * kw.get("carry_fwd", 0.30) + LEFT * kw.get("carry_left", 0.0)
        mug_c[2] = kw.get("carry_z", 0.95)                  # the mug's base height
        tool_off = s.wbc.ee_pos(d, b) - d.xpos[self.mb]     # hand tool point - mug base, world
        carry = mug_c + tool_off
        if empty:
            carry = pel + FWD * kw.get("carry_fwd", 0.30) - LEFT * 0.04
            carry[2] = kw.get("carry_z", 0.95) + 0.044
        # IK waypoints with the elbow out (as for the reach): the controller's own redundancy resolution folded the
        # arm into the torso on cross-body moves (runs c02-c04)
        qs, e_c = self.plan_arm(carry, s.wbc.ee_rot[b].copy(), n=20,
                                via=0.5 * (s.wbc.ee_pos(d, b) + carry) + np.array([0.0, 0.0, 0.02]))
        self.follow(qs, kw.get("carry_s", 1.6))
        self.level_hold(kw.get("carry_hold", 0.4))
        res["carry"] = dict(slip_mm=round(self.slip(), 1), tilt_deg=round(self.tilt(), 1),
                            mug_at=np.round(d.xpos[self.mb], 3).tolist())
        self.note("carry pose", **res["carry"])
        if kw.get("stop_after") == "carry":
            p.hold(kw.get("test_hold", 1.0))
            return self.finish(res, others0)

        # 5. turn left and walk away along the counter
        self.mark("walk")
        goal = kw.get("goal", (0.90, 1.85, 180.0))
        res["walk"] = self.drive(np.asarray(goal[:2], float), math.radians(goal[2]), vmax=kw.get("v", 0.4),
                                 amax=kw.get("a", 0.4), turn_frac=kw.get("turn_frac", 0.6), stop_short=kw.get("short", 0.04),
                                 warm=kw.get("warm_s", 0.4), idle=kw.get("idle_s", 0.05), name="walked away")
        res["walked"] = dict(slip_mm=round(self.slip(), 1), tilt_deg=round(self.tilt(), 1),
                             mug_at=np.round(d.xpos[self.mb], 3).tolist())
        self.note("walked", **res["walked"])

        # 6. stand still holding it (the person looks down at it)
        self.mark("stand")
        p.hold(kw.get("end_hold", 1.5))
        self.mark("end")
        res["end"] = dict(slip_mm=round(self.slip(), 1), tilt_deg=round(self.tilt(), 1),
                          mug_at=np.round(d.xpos[self.mb], 3).tolist(), support_N=round(self.support(), 2),
                          robot_xy=np.round(d.qpos[:2], 3).tolist(), robot_heading_deg=round(math.degrees(self.yaw()), 1),
                          held=bool(self.contact_force()[0] > 1.0 and self.support() < 0.1))
        self.note("end", **res["end"])
        return self.finish(res, others0)

    def finish(self, res, others0):
        s, d, m = self.s, self.d, self.m
        if "end" not in self.phase:
            self.mark("end")
        st = self.stats
        res["slip_max_mm"] = round(st["slip_max"], 1)
        res["slip_walk_max_mm"] = round(st["slip_walk_max"], 1)
        res["tilt_max_deg"] = round(st["tilt_max"], 1)
        res["tilt_walk_max_deg"] = round(st["tilt_walk_max"], 1)
        moved = sorted(((float(np.linalg.norm(d.xpos[fb] - x0)) * 1000, m.body(fb).name) for fb, x0 in others0.items()),
                       reverse=True)
        res["others_moved_mm"] = [(n, round(v, 2)) for v, n in moved[:3]]
        res["phases"] = self.phase
        ph = self.phase
        keys = [k for k in ("reach", "grasp", "lift", "carry", "walk", "stand", "end") if k in ph]
        res["durations_s"] = {a: round(ph[b_] - ph[a], 2) for a, b_ in zip(keys, keys[1:])}
        res["total_s"] = round(ph["end"] - ph["start"], 2)
        res["checks"] = s.checks.as_dict()
        held = res.get("end", {}).get("held", False)
        res["ok"] = bool(s.checks.fell_at is None and held and st["slip_max"] <= 5.0 and st["tilt_max"] <= 8.0)
        self.p.mv.on_tick = self.on_tick0 if self.kw.get("restore_tick", False) else self.p.mv.on_tick
        return res


def run(p, s, **kw) -> dict:
    """The cup step inside a continuous run. p: motion.Program on session s. Options (all optional): phi, e, h,
    tx, ty (the grip), slide, reach_s, close_s, squeeze, lift, lift_s, carry_fwd, carry_z, goal (x, y, heading deg),
    v, a, turn_frac, end_hold, stop_after ('lift' | 'carry'). Returns measured results; 'ok' = no fall, mug held at
    the end, slip <= 5 mm and tilt <= 8 deg over the whole step (contacts: probe_pen.py on the recording)."""
    return CupSkill(p, s, **kw).run()
