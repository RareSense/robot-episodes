"""Step 1 of the tea film at the person's pace: the G1 walks in, the LEFT hand takes mug object_0316 off the
rack's middle tier, carries it close to the chest, the robot turns right and walks along the counter, and
sets the mug down upright at (1.72, 2.24), handle +x. Fast version of test_mug3.py (same grip, same
set-down logic).

Physically simulated: MuJoCo 600 Hz, every G1 motor torque-limited (byor MotorLayer). Walking = the
walking network's leg targets through the motors, commanded only with velocity commands (vx, vy, yaw
rate) at its own 50 Hz rate, from a planned straight-line speed profile with position/heading feedback
(our own follower in place of byor goto's pure pursuit). Non-leg joints while walking: the walking
controller's PD hold targets (the carrying arm's targets are moved smoothly by us, joint space, from an
IK solution; no torque added). Standing work = byor whole-body controller with payload compensation from
the hand-mug contact forces (what a wrist force sensor reads). The mug is a free body held only by finger
contact. Nothing writes qpos/qvel; no welds; no external forces; no limits raised.

run(p, s, **kw) -> dict of measured results. Starts from the robot standing near (0.62, 1.15) facing +y
(the session start); leaves it standing at the set-down spot with the left hand back by its body.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from motion import quat_from_axes, smooth
from nova_robotics.byor.moves import Moves
from nova_robotics.byor.skills import Tasks
from tea_sim import JAR, KETTLE, MUG

TX, TY, CURL = 0.13, -0.072, 0.4                  # the grip (mug_search.py), as test_mug3
PLACE = (1.72, 2.24)
RACK_STAND = (0.555, 1.790, 92.0)                 # pelvis x, y, heading (deg) for the grasp (test_mug3 stood here)
SPOT_STAND = (1.875, 1.905, 84.0)                 # pelvis for the set-down (mug7 stood at 1.879, 1.900, 81.6 deg)
Q = quat_from_axes([0, 1, 0], [-1, 0, 0], [0, 0, 1])   # fingers +y, palm (-y_h) to the mug, finger stack vertical
L = 0                                             # left hand
# natural left-arm postures (shoulder p/r/y, elbow, wrist r/p/y) the whole-body controller used in test_mug3's
# run (mug7_rec.npz): hand up in front of the rack, and over the set-down spot holding the mug
SEED_PRE = [-1.09, 0.27, -0.05, 0.50, -0.45, 0.55, 0.21]
SEED_SET = [-0.24, 0.34, -0.57, -0.88, -0.59, 0.83, 0.84]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _dsmooth(u):
    """d/du of the minimum-jerk blend."""
    u = min(max(u, 0.0), 1.0)
    return 30 * u * u * (1 - u) * (1 - u)


class MugSkill:
    def __init__(self, p, s, log=print, **kw):
        self.p, self.s, self.log = p, s, log
        self.kw = kw
        m, d = s.m, s.d
        self.m, self.d = m, d
        self.b = s.hands[L]
        self.mb = m.body("env/" + MUG + "_body").id
        self.robot = {i for i in range(1, m.nbody) if not m.body(i).name.startswith("env/")}
        self.ign = [m.body("env/" + o + "_body").id for o in (KETTLE, MUG, JAR)]
        self.tk = Tasks(s)
        self.WEIGHT = float(m.body_subtreemass[self.mb]) * 9.81
        self.HAND = s.hand_parts[self.b]
        self.hold_on = False
        self.est = np.zeros(3)
        self.rel0 = None
        self.slip_max = 0.0
        self.slip_walk_max = 0.0
        self.phase = {}
        self.anim = None
        self.qa = s.loco.qa
        # arm dofs (shoulder .. wrist) and their qpos addresses
        self.arm_dofs = sorted(int(m.jnt_dofadr[j]) for x in s._arm_bodies(self.b)
                               for j in range(m.body_jntadr[x], m.body_jntadr[x] + m.body_jntnum[x]))
        self.arm_q = np.array([int(m.jnt_qposadr[int(m.dof_jntid[v])]) for v in self.arm_dofs], int)
        self.on_tick0 = p.mv.on_tick
        p.mv.on_tick = self.payload_tick

    # ------------------------------------------------------------------ measurements
    def mark(self, name):
        self.phase[name] = round(float(self.d.time), 2)

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

    def hand_on_counter(self):
        m, d = self.m, self.d
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for a, o in ((b1, b2), (b2, b1)):
                if a in self.HAND and o not in self.robot and o != self.mb:
                    return True
        return False

    def slip(self):
        d, b = self.d, self.b
        if self.rel0 is None:
            return 0.0
        return float(np.linalg.norm(d.xmat[b].reshape(3, 3).T @ (d.xpos[self.mb] - d.xpos[b]) - self.rel0)) * 1000

    def tilt(self):
        return float(np.degrees(np.arccos(np.clip(self.d.xmat[self.mb][8], -1, 1))))

    def mug_yaw(self):
        q = self.d.xquat[self.mb]
        return math.degrees(2 * math.atan2(q[3], q[0]))

    def track_slip(self, walking=False):
        if self.hold_on and self.rel0 is not None:
            sl = self.slip()
            self.slip_max = max(self.slip_max, sl)
            if walking:
                self.slip_walk_max = max(self.slip_walk_max, sl)

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
        self.track_slip()

    def note(self, what, **kv):
        kv.pop("t", None)
        self.p.note(what, **kv)

    # ------------------------------------------------------------------ IK for the walking hold targets
    def arm_ik(self, base_xy, base_yaw, target, quat, tool, seed=None, iters=200, wrot=0.05):
        """Arm joint angles (shoulder..wrist) that put the hand's `tool` point (hand frame) at `target` with
        orientation `quat`, the pelvis at (base_xy, heading base_yaw), the rest of the body as now. Solved
        on a scratch copy from `seed` (a natural arm posture for the task, pulled towards in the null
        space); nothing in the live state is touched."""
        m, b = self.m, self.b
        sd = mujoco.MjData(m)
        sd.qpos[:] = self.d.qpos
        qa = self.qa
        sd.qpos[qa:qa + 2] = base_xy
        sd.qpos[qa + 3:qa + 7] = [math.cos(base_yaw / 2), 0, 0, math.sin(base_yaw / 2)]
        if seed is not None:
            sd.qpos[self.arm_q] = seed
        q_nat = sd.qpos[self.arm_q].copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        cols = self.arm_dofs
        lo = np.array([m.jnt_range[int(m.dof_jntid[v])][0] for v in cols]) + 0.05
        hi = np.array([m.jnt_range[int(m.dof_jntid[v])][1] for v in cols]) - 0.05
        tool = np.asarray(tool, float)
        err = 1.0
        for _ in range(iters):
            mujoco.mj_kinematics(m, sd)
            mujoco.mj_comPos(m, sd)
            R = sd.xmat[b].reshape(3, 3)
            pt = sd.xpos[b] + R @ tool
            mujoco.mj_jac(m, sd, jacp, jacr, pt, b)
            ep = np.asarray(target, float) - pt
            r = np.zeros(3)
            mujoco.mju_subQuat(r, np.asarray(quat, float), sd.xquat[b])
            er = R @ r
            e = np.r_[ep, wrot * er]
            err = float(np.linalg.norm(ep))
            if err < 1e-3 and np.linalg.norm(er) < 0.02:
                break
            J = np.vstack([jacp[:, cols], wrot * jacr[:, cols]])
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6))
            dq = Jp @ e + 0.2 * (np.eye(len(cols)) - Jp @ J) @ (q_nat - sd.qpos[self.arm_q])
            n = float(np.abs(dq).max())
            if n > 0.1:
                dq *= 0.1 / n
            q = sd.qpos[self.arm_q] + dq
            sd.qpos[self.arm_q] = np.clip(q, lo, hi)
        return sd.qpos[self.arm_q].copy(), err

    def sag_offset(self, q_arm, load_tool=None, base_xy=None, base_yaw=None):
        """Set-point offset that makes the walking controller's arm PD hold `q_arm` against gravity (the
        arm's own weight plus the mug hanging at `load_tool`, hand frame): offset = tau_gravity / kp, the
        same PD gains, no torque added."""
        m, b, loco = self.m, self.b, self.s.loco
        sd = mujoco.MjData(m)
        sd.qpos[:] = self.d.qpos
        if base_xy is not None:
            sd.qpos[self.qa:self.qa + 2] = base_xy
            sd.qpos[self.qa + 3:self.qa + 7] = [math.cos(base_yaw / 2), 0, 0, math.sin(base_yaw / 2)]
        sd.qpos[self.arm_q] = q_arm
        sd.qvel[:] = 0.0
        mujoco.mj_kinematics(m, sd)
        mujoco.mj_comPos(m, sd)
        tau = np.zeros(m.nv)
        mujoco.mj_rne(m, sd, 0, tau)                    # gravity at rest
        g = tau[self.arm_dofs].copy()
        if load_tool is not None:
            pt = sd.xpos[b] + sd.xmat[b].reshape(3, 3) @ np.asarray(load_tool, float)
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, sd, jacp, None, pt, b)
            g += jacp[:, self.arm_dofs].T @ np.array([0.0, 0.0, self.WEIGHT])
        idx = np.array([int(np.nonzero(loco.hold_q == q)[0][0]) for q in self.arm_q], int)
        return g / loco.hold_kp[idx]

    def start_anim(self, q1, seconds):
        """Move the walking controller's PD hold targets of the left arm to q1 (minimum-jerk, joint space)."""
        loco = self.s.loco
        idx = np.array([int(np.nonzero(loco.hold_q == q)[0][0]) for q in self.arm_q], int)
        self.anim = dict(idx=idx, q0=loco.hold_target[idx].copy(), q1=np.asarray(q1, float),
                         t0=float(self.d.time), T=float(seconds))

    def step_anim(self):
        a = self.anim
        if a is None or self.s.mode != "walk":
            return
        u = smooth((self.d.time - a["t0"]) / a["T"])
        self.s.loco.hold_target[a["idx"]] = a["q0"] + u * (a["q1"] - a["q0"])
        if u >= 1.0:
            self.anim = None

    # ------------------------------------------------------------------ walking
    def standing_again(self):
        """After walking: a fresh whole-body controller took balance back; plan from where we stand."""
        s, p = self.s, self.p
        s.set_walk((0.0, 0.0, 0.0))
        stood = self.tk._wait_stand()
        for attr in ("_armw", "_ease0"):
            if hasattr(p, attr):
                delattr(p, attr)
        p.mv, p.w = Moves(s, on_tick=self.payload_tick), s.wbc
        p.root = p.mv.root
        return stood

    def yaw(self):
        q = self.d.qpos[self.qa + 3:self.qa + 7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def drive(self, goal_xy, goal_yaw, *, vmax=0.5, amax=0.6, turn=0.0, carry=False, grip_walk=0.8,
              kp=0.8, kyaw=1.5, tol=0.03, yaw_tol=math.radians(4), extra=1.0, warm=None, idle=None, stop_short=0.0, bow=0.0, lead=None,
              during=None, name="walk"):
        """Walk the pelvis to goal_xy, heading goal_yaw. The walking network gets velocity commands from a
        minimum-jerk straight-line profile (peak speed <= vmax, acceleration <= amax) plus position and
        heading feedback; `turn` (rad) turns the body right by that much mid-way (walking along the
        counter) and back. `carry`: the left arm keeps its pose, its fingers keep squeezing.
        `during(f, t)` runs every policy tick (f = fraction of the profile, t = s since the profile
        started; < 0 during the start-up)."""
        s, d, m = self.s, self.d, self.m
        loco = s.loco
        cfg = loco.cfg
        saved_cfg = {k: cfg[k] for k in ("warm_s", "idle_s")}
        if warm is not None:
            cfg["warm_s"] = warm
        if idle is not None:
            cfg["idle_s"] = idle
        names = [m.joint(int(np.nonzero(m.jnt_qposadr == q)[0][0])).name for q in loco.hold_q]
        keep = np.array([carry and n.startswith("left_") and any(k in n for k in ("shoulder", "elbow", "wrist"))
                         for n in names])
        named0, target0 = loco.hold_named.copy(), loco.hold_target.copy()
        loco.hold_named = loco.hold_named & ~keep
        if carry and self.kw.get("gravity_comp", True) and self.hand_load()[2] < -1.0:   # only with the mug in hand
            # the carrying arm keeps its pose: its PD set-points (blended in over the start-up) are where it
            # is now plus the sag its own weight and the mug's would cause
            tool = d.xmat[self.b].reshape(3, 3).T @ (d.xpos[self.mb] - d.xpos[self.b])
            off = self.sag_offset(d.qpos[self.arm_q].copy(), load_tool=tool)
            for q, o in zip(self.arm_q, off):
                i = int(np.nonzero(loco.hold_q == q)[0][0])
                loco.hold_named[i], loco.hold_target[i] = True, float(d.qpos[q] + o)
        for h, hb in enumerate(s.hands):
            g = 1.0 if (carry and h == L) else grip_walk
            tgt = {a: lo + (g if high else 1 - g) * (hi - lo) for a, lo, hi, high in s.fingers[hb]}
            for i, q in enumerate(loco.hold_q):
                if int(q) in tgt:
                    loco.hold_named[i], loco.hold_target[i] = True, tgt[int(q)]
        t_start = d.time
        s.set_walk((0.01, 0.0, 0.0))                        # hands the legs to the walking network
        dt, n = m.opt.timestep, loco.period
        vlim = np.asarray(cfg["vmax"], float)
        warm_until = cfg["blend_s"] + cfg["warm_s"]

        def tick(cmd, f, t):
            s.set_walk(cmd)
            if during is not None:
                during(f, t)
            self.step_anim()
            s.step(n)
            self.track_slip(walking=True)
            if s.checks.fell_at is not None:
                raise RuntimeError(f"robot fell at {s.checks.fell_at} s")

        while loco.step_count * dt < warm_until - 0.02:     # blend + warm-up: the network ignores commands
            tick((0.01, 0.0, 0.0), -1.0, -1.0)
        xy0, yaw0 = d.qpos[self.qa:self.qa + 2].copy(), self.yaw()
        # the gait keeps stepping ~2 s after the last command and creeps forward 4-8 cm meanwhile (measured):
        # the commanded walk ends that much short of the stance, along the final heading
        aim = np.asarray(goal_xy, float) - stop_short * np.array([math.cos(goal_yaw), math.sin(goal_yaw)])
        D = aim - xy0
        Ld = float(np.linalg.norm(D))
        dyaw = _wrap(goal_yaw - yaw0)
        T = max(1.875 * Ld / vmax, math.sqrt(5.774 * Ld / amax), 1.875 * abs(dyaw) / 0.6, 0.6)
        if turn:
            T = max(T, math.pi * abs(turn) / 0.85)
        nrm = np.array([D[1], -D[0]]) / max(Ld, 1e-9)       # right of the walking direction

        def ref(tt):
            f = min(max(tt / T, 0.0), 1.0)
            sm, smd = smooth(f), (_dsmooth(f) / T if 0.0 < f < 1.0 else 0.0)
            psi = yaw0 + dyaw * sm - turn * math.sin(math.pi * f) ** 2
            w = (dyaw * smd - turn * math.pi / T * math.sin(2 * math.pi * f)) if 0.0 < f < 1.0 else 0.0
            # `bow`: the path bends that far to the right mid-way (away from the counter when walking along it)
            pos = xy0 + D * sm + nrm * bow * math.sin(math.pi * sm)
            vel = (D + nrm * bow * math.pi * math.cos(math.pi * sm)) * smd
            return pos, vel, psi, w

        t_move = d.time
        t = 0.0
        lead = self.kw.get("lead_s", 0.5) if lead is None else lead   # the gait answers a command ~0.5 s later
        err_max = 0.0
        while True:
            f = min(t / T, 1.0)
            p_ref, _, psi_ref, _ = ref(t)
            _, v_ff, _, w_ff = ref(t + lead)
            xy, yaw = d.qpos[self.qa:self.qa + 2], self.yaw()
            err_max = max(err_max, float(np.linalg.norm(p_ref - xy)))
            vw = v_ff + kp * (p_ref - xy)
            c, sn = math.cos(yaw), math.sin(yaw)
            cmd = np.array([c * vw[0] + sn * vw[1], -sn * vw[0] + c * vw[1],
                            w_ff + kyaw * _wrap(psi_ref - yaw)])
            cmd = np.clip(cmd, -vlim, vlim)
            if f >= 1.0:                                    # past the profile: a slow last correction only
                cmd[:2] = np.clip(cmd[:2], -0.12, 0.12)
                close = (np.linalg.norm(aim - xy) < tol and abs(_wrap(goal_yaw - yaw)) < yaw_tol)
                if close or t > T + extra:
                    break
            if np.all(np.abs(cmd) < 2e-3):
                cmd[0] = 2e-3
            tick(cmd, f, t)
            t += n * dt
        t_cmd0 = d.time
        at_cmd0 = (np.round(d.qpos[self.qa:self.qa + 2], 3).tolist(), round(math.degrees(self.yaw()), 1))
        s.set_walk((0.0, 0.0, 0.0))
        while s.mode != "stand" and d.time - t_cmd0 < 8.0:  # stepping stops at a double-support instant,
            self.step_anim()                                # settles, then the whole-body controller
            s.step(n)                                       # takes balance back
            self.track_slip(walking=True)
            if s.checks.fell_at is not None:
                raise RuntimeError(f"robot fell at {s.checks.fell_at} s")
        t_stand = d.time
        stood = self.standing_again()
        loco.hold_named, loco.hold_target = named0, target0
        cfg.update(saved_cfg)
        self.anim = None
        out = dict(start=round(t_start, 2), profile_s=round(T, 2), startup_s=round(t_move - t_start, 2),
                   moving_s=round(t_cmd0 - t_move, 2), stop_s=round(t_stand - t_cmd0, 2), stood=stood,
                   track_err_max_m=round(err_max, 3), at_stop_cmd=at_cmd0,
                   at=np.round(d.qpos[self.qa:self.qa + 2], 3).tolist(), goal=list(np.round(goal_xy, 3)),
                   heading_deg=round(math.degrees(self.yaw()), 1), goal_heading_deg=round(math.degrees(goal_yaw), 1))
        self.note(name, **out)
        return out

    # ------------------------------------------------------------------ the step
    def run(self):
        p, s, d, m, b, kw = self.p, self.s, self.d, self.m, self.b, self.kw
        res = {}
        self.mark("start")
        # every other free object in the scene (the other mugs, kettle, jar, ...): did anything else move?
        free = [int(m.jnt_bodyid[j]) for j in range(m.njnt)
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and int(m.jnt_bodyid[j]) not in self.robot
                and int(m.jnt_bodyid[j]) != self.mb]
        others0 = {fb: d.xpos[fb].copy() for fb in free}
        yaw_start = self.mug_yaw()
        rack = kw.get("rack_stand", RACK_STAND)
        spot = kw.get("spot_stand", SPOT_STAND)
        place = np.asarray(kw.get("place", PLACE), float)
        warm, idle = kw.get("warm_s", 0.4), kw.get("idle_s", 0.05)
        home = d.xpos[self.mb].copy()
        c = home + np.array([0.0, 0.0, kw.get("grip_h", 0.050)])   # the mug's axis, a little above mid-wall
        # the hand comes down `side_off` further out in the gap (away from the mug wall): with the round thin-wall
        # mug collision the curled middle finger otherwise lands on the rim's top; then it closes in sideways
        side = np.array([-kw.get("side_off", 0.015), 0.0, 0.0])
        c_out = c + side
        above = c_out + np.array([0.0, 0.0, 0.10])       # over the gap, above the side panel and the rims
        front = above + np.array([0.0, -0.14, 0.0])
        pre = front + np.array([0.0, -0.09, 0.02])      # where the hand goes up to while the walk-in ends

        # 1. walk in to the rack, fingers curled (reach_walk=True also raises the left hand during the last steps:
        #    tried, the stop then creeps 8-10 cm and the grasp fails; off by default)
        for h in (0, 1):
            s.set_grip(h, 0.8)
        reach_walk = kw.get("reach_walk", False)
        q_pre, ik_err = self.arm_ik(rack[:2], math.radians(rack[2]), pre, Q, (TX, TY, 0.0), seed=SEED_PRE)
        res["ik_pre_err_mm"] = round(ik_err * 1000, 1)

        def during1(f, t):
            if reach_walk and self.anim is None and f >= kw.get("reach_at", 0.3) and not getattr(self, "_a1", False):
                self._a1 = True
                off = self.sag_offset(q_pre, None, rack[:2], math.radians(rack[2]))
                self.start_anim(q_pre + off, kw.get("reach_walk_s", 1.6))

        self.mark("walk_in")
        res["walk_in"] = self.drive(rack[:2], math.radians(rack[2]), vmax=kw.get("v1", 0.5), amax=kw.get("a1", 0.55),
                                    stop_short=kw.get("short1", 0.06), tol=0.02,
                                    warm=warm, idle=idle, during=during1, name="walked in")
        # 2. the mug off the middle tier (whole-body reach for the lean, then hand-only moves)
        self.mark("grasp")
        p.set_tool(L, (TX, TY, 0.0))
        p.grip(L, CURL, kw.get("curl_s", 0.25))
        if reach_walk:
            p.reach(L, front, quat=Q, seconds=kw.get("front_s", 1.0), settle=0.0)
        else:
            sh = d.xpos[s.shoulder[b]].copy()
            # lean into the reach as test_mug3's run did (16 deg): the arm then takes the same posture it
            # had there, which the descent into the gap beside the mug needs
            plan = p.plan_reach(L, front, leans=kw.get("leans", (16, 20, 24, 28)))
            p.reach(L, front, quat=Q, seconds=kw.get("front_s", 1.4), settle=0.0, plan=plan,
                    via=np.array([sh[0], 1.98, front[2] - 0.02]))
        e_front = p.settle_hand(L, front, tol=0.008, timeout=kw.get("settle_front", 0.0))
        p.move_hand(L, above, quat=Q, seconds=kw.get("above_s", 0.5))
        e_above = p.settle_hand(L, above, tol=0.006, timeout=kw.get("settle_above", 0.3))
        res["approach_err_mm"] = dict(front=round(e_front * 1000, 1), above=round(e_above * 1000, 1))
        if side[0] != 0.0:
            p.move_hand(L, c_out, quat=Q, seconds=kw.get("down_s", 0.6))       # down beside the mug
            p.move_hand(L, c, quat=Q, seconds=kw.get("in_s", 0.3))             # palm in against its side
        else:
            p.move_hand(L, c, quat=Q, seconds=kw.get("down_s", 0.6))
        err = p.settle_hand(L, c, tol=0.004, timeout=kw.get("settle_to", 0.6))
        p.grip(L, 1.0, kw.get("close_s", 0.5))
        self.hold_on = True
        p.hold(kw.get("grip_hold", 0.15))
        self.rel0 = d.xmat[b].reshape(3, 3).T @ (d.xpos[self.mb] - d.xpos[b])
        self.mark("lift")
        p.move_hand(L, c + [0.0, 0.0, 0.035], quat=Q, seconds=kw.get("lift_s", 0.4))      # straight up off the tier
        # out over the front row, straight into the carry pose close in front of the chest
        carry = np.array([d.qpos[0] + 0.03, d.qpos[1] + kw.get("carry_fwd", 0.13), c[2] + kw.get("carry_up", 0.05)])
        p.move_hand(L, carry, quat=Q, seconds=kw.get("out_s", 1.0), via=c + [0.0, -0.12, 0.07])
        res["out_of_rack"] = dict(t=round(d.time, 2), place_err_mm=round(err * 1000, 1), slip_mm=round(self.slip(), 1),
                                  tilt_deg=round(self.tilt(), 1), mug_z=round(float(d.xpos[self.mb][2]), 3))
        self.note("out of the rack, carry pose", **res["out_of_rack"])
        lifted = float(d.xpos[self.mb][2] - home[2])
        if self.hand_load()[2] > -1.0 or lifted < -0.03 or self.slip() > 40.0:
            # the mug is not in the hand (missed grasp): say so and stop here, robot standing at the rack
            res.update(ok=False, failed="grasp: mug not in the hand", mug_rise_mm=round(lifted * 1000, 1),
                       phases=self.phase, checks=s.checks.as_dict(), total_s=round(d.time - self.phase["start"], 2))
            self.hold_on = False
            s.wbc.payload.pop(b, None)
            p.mv.on_tick = self.on_tick0
            self.note("grasp failed", mug_rise_mm=res["mug_rise_mm"])
            return res
        if kw.get("stop_after") == "carry":                # development: first half only
            res["phases"], res["checks"], res["slip_max_mm"] = self.phase, s.checks.as_dict(), round(self.slip_max, 1)
            return res

        # 3. walk along the counter carrying it; as the walk ends the mug comes over its spot
        set_walk = kw.get("set_walk", True)
        target_hi = np.r_[place, 0.86 + kw.get("pre_height", 0.06)]

        def during2(f, t):
            if set_walk and self.anim is None and f >= kw.get("set_at", 0.6) and not getattr(self, "_a2", False):
                self._a2 = True
                tool = d.xmat[b].reshape(3, 3).T @ (d.xpos[self.mb] - d.xpos[b])     # the mug, in hand frame
                q_set, e2 = self.arm_ik(spot[:2], math.radians(spot[2]), target_hi, Q, tool, seed=SEED_SET)
                res["ik_set_err_mm"] = round(e2 * 1000, 1)
                off = self.sag_offset(q_set, tool, spot[:2], math.radians(spot[2]))
                self.start_anim(q_set + off, kw.get("set_walk_s", 1.4))

        self.mark("walk_along")
        res["walk_along"] = self.drive(spot[:2], math.radians(spot[2]), vmax=kw.get("v2", 0.65), amax=kw.get("a2", 0.6),
                                       turn=math.radians(kw.get("turn_deg", 45.0)), carry=True, warm=warm, idle=idle,
                                       stop_short=kw.get("short2", 0.05), bow=kw.get("bow", 0.12), tol=0.03, extra=0.6,
                                       during=during2, name="walked along the counter")
        res["carried"] = dict(slip_mm=round(self.slip(), 1), tilt_deg=round(self.tilt(), 1),
                              mug_at=np.round(d.xpos[self.mb], 3).tolist())
        self.note("carried", **res["carried"])

        # 4. set it down at the spot, handle right: over it, steered by where the mug is, then down by force
        self.mark("set_down")
        target = np.r_[place, 0.86]
        e = target - d.xpos[self.mb]
        p.reach(L, s.wbc.ee_goal[b] + [e[0], e[1], e[2] + 0.03], quat=Q, seconds=kw.get("set_reach_s", 0.8), settle=0.0)
        for sec in kw.get("corr_s", (0.35,)):
            e = target - d.xpos[self.mb]
            p.move_hand(L, s.wbc.ee_goal[b] + [e[0], e[1], e[2] + 0.03], quat=Q, seconds=sec)
            p.hold(kw.get("corr_hold", 0.1))
        went = p.lower_until(L, lambda: self.support() > 0.3 or self.hand_on_counter(), speed=kw.get("lower_v", 0.06),
                             max_dist=0.08, quat=Q, steer=lambda: (target - d.xpos[self.mb])[:2])
        if not self.hand_on_counter():
            went += p.lower_until(L, lambda: self.support() > 0.8 * self.WEIGHT or self.hand_on_counter(), speed=0.012,
                                  max_dist=0.012, quat=Q, ease_in=0.08)
        p.hold(kw.get("down_hold", 0.15))
        res["set_down"] = dict(t=round(d.time, 2), place_err_mm=round(float(np.linalg.norm(d.xpos[self.mb][:2] - place)) * 1000, 1),
                               tilt_deg=round(self.tilt(), 1), support_pct=round(100 * self.support() / self.WEIGHT),
                               handle_yaw_deg=round(self.mug_yaw(), 1), lowered_mm=round(went * 1000, 1),
                               slip_mm=round(self.slip(), 1))
        self.note("set down", **res["set_down"])
        # 5. let go: unload, open, palm off its side, the hand back in by the body
        self.mark("release")
        seated = d.xpos[self.mb].copy()
        p.move_hand(L, s.wbc.ee_pos(d, b) + [0.0, 0.0, 0.005], quat=Q, seconds=0.25)
        self.hold_on = False
        p.grip(L, 0.0, 0.35)
        p.move_hand(L, s.wbc.ee_pos(d, b) + [-0.04, 0.0, 0.01], quat=Q, seconds=0.35)
        p.reach(L, np.array([d.qpos[0] - 0.12, d.qpos[1] + 0.18, 1.0]), quat=Q, seconds=kw.get("back_s", 0.7),
                via=s.wbc.ee_goal[b] + [-0.03, -0.08, 0.06], settle=0.0)
        p.grip(L, 0.3, 0.3)
        p.hold(kw.get("end_hold", 0.1))
        self.mark("end")
        s.wbc.payload.pop(b, None)
        p.mv.on_tick = self.on_tick0
        res["end"] = dict(t=round(d.time, 2), mug_moved_on_release_mm=round(float(np.linalg.norm(d.xpos[self.mb] - seated)) * 1000, 1),
                          place_err_mm=round(float(np.linalg.norm(d.xpos[self.mb][:2] - place)) * 1000, 1),
                          tilt_deg=round(self.tilt(), 1), handle_yaw_deg=round(self.mug_yaw(), 1),
                          handle_yaw_start_deg=round(yaw_start, 1),
                          mug_xyz=np.round(d.xpos[self.mb], 4).tolist(),
                          robot_xy=np.round(d.qpos[self.qa:self.qa + 2], 3).tolist(),
                          robot_heading_deg=round(math.degrees(self.yaw()), 1))
        self.note("end", **res["end"])
        res["slip_max_mm"] = round(self.slip_max, 1)
        moved = sorted(((float(np.linalg.norm(d.xpos[fb] - x0)) * 1000, m.body(fb).name) for fb, x0 in others0.items()),
                       reverse=True)
        res["others_moved_mm"] = [(n, round(v, 2)) for v, n in moved[:4]]
        res["slip_walk_max_mm"] = round(self.slip_walk_max, 1)
        res["phases"] = self.phase
        ph = self.phase
        res["durations_s"] = {k: round(ph[n] - ph[k], 2) for k, n in zip(
            ("walk_in", "grasp", "lift", "walk_along", "set_down", "release"),
            ("grasp", "lift", "walk_along", "set_down", "release", "end"))}
        res["total_s"] = round(ph["end"] - ph["start"], 2)
        res["checks"] = s.checks.as_dict()
        res["ok"] = bool(s.checks.fell_at is None and res["end"]["place_err_mm"] <= 12 and res["end"]["tilt_deg"] < 5)
        return res


def run(p, s, **kw) -> dict:
    """Step 1 (mug off the rack to its spot) inside a continuous run. p: motion.Program on session s.
    Keyword options (all optional): rack_stand, spot_stand, place, v1, v2, a1, a2, turn_deg, warm_s, idle_s,
    reach_walk, set_walk and the per-motion durations (see MugSkill.run). Returns measured results."""
    return MugSkill(p, s, **kw).run()
