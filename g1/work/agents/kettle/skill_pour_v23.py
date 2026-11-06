"""Chunk 3 skill (physics): the RIGHT hand takes the kettle off its base, carries it to the mug (side-steps
along the counter with it), the LEFT hand steadies the mug, the kettle tilts so the spout pours into the
mug, tilts back, and is set down behind the mug at about (1.58, 2.38), handle toward +x; the hand lets go
and retreats.

What is simulated: everything moves through MuJoCo physics at 600 Hz. The G1 only through its
torque-limited motors (our byor whole-body controller standing, the byor walking network for the
side-steps); the kettle (1.25 kg = jug + ~0.4 l water, constant: the water itself is simulated later in
Isaac) and the mug are free bodies held only by finger contact and friction. Nothing is posed, welded or
pushed. The controller is told the load in the hand from the hand's contact forces with the kettle (what a
wrist force sensor reads: `payload`), never more. Targets are planned from the objects' MEASURED poses
(what the robot's camera gives), with slow closed-loop corrections, never by writing the state.

Start state: robot standing at about (1.86, 1.90) facing +y, the mug at (1.72, 2.24) handle +x, the kettle
on its base at (2.57, 2.36). Ends standing, hands free, at about (1.80, 1.93).
    from skill_pour import run;  result = run(p, s)          # p = motion.Program(s)
"""
from __future__ import annotations
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401

import math
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(WORK))
from motion import quat_from_axes, quat_mul, smooth  # noqa: E402
from nova_robotics.byor.moves import Moves  # noqa: E402
from nova_robotics.byor.skills import Tasks  # noqa: E402
from tea_sim import JAR, KETTLE, MUG, SWITCH  # noqa: E402

R_, L_ = 1, 0                                          # hand indices: right, left
KETTLE_STANCE = (2.56, 1.92)                           # the grip was tuned standing at ~(2.56, 1.92)
POUR_STANCE = (1.80, 1.86)                             # in front of the mug, a little right of it (the
                                                       # walking network drifts ~3-5 cm forward on stopping)
PLACE = (1.87, 2.21)                                   # kettle set-down, right of the mug (lead, 29 Oct): clear of
PLACE_YAW = -45.0                                      # mug 2 cm, tray 2 cm, jar 3 cm; handle turned 45 deg toward
                                                       # the robot so it and the lever stay in front of the tray
BEHIND_MUG = (1.58, 2.38)                              # the earlier spot: the wrist then sits over the mug (pour20)
WRAP = (0.105, 0.03, 0.0)                              # grip tool point in the hand frame (grasp_search)
GZ = 0.158                                             # grip height on the handle bar (kettle frame)
BAR_X = 0.126                                          # handle bar centre line (kettle frame)
TIP = np.array([-0.0985, 0.0, 0.200])                  # spout mouth, lowest lip (kettle frame)
PIVOT = np.array([0.02, 0.0, 0.10])                    # the kettle turns about this point (pour_path.py 90)
TILT_DEG = 90.0                                        # water agent (PhysX fluid, 0.4 l): pours from ~86 deg, full mug at 85-93 deg for ~2.5 s
PSI_DEG = -15.0                                        # kettle yaw while pouring: body swung toward the robot
TIP_ABOVE_RIM = 0.04
MUG_RIM = 0.085                                        # mug height above its base (m)
CARRY_BOTTOM_Z = 1.070                                 # kettle bottom while carried: the sachet tray top is 0.925 and the
                                                       # kettle sinks ~5 cm when the walking network takes the legs
MUG_TOOL = (0.13, -0.072, 0.0)                         # left-hand wrap point of the mug grip (test_mug3)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def _mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q / np.linalg.norm(q)


def _quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _rotvec(R):
    """Rotation vector (axis * angle) of a rotation matrix."""
    q = _mat2quat(R)
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def _axis_angle_mat(v):
    q = np.zeros(4)
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.eye(3)
    mujoco.mju_axisAngle2Quat(q, v / ang, ang)
    return _quat2mat(q)


def run(p, s, place=PLACE, place_yaw=PLACE_YAW, pour_s=2.5, tilt_deg=TILT_DEG, walk_speed=0.18, log=print, walk_in=True,
        stop_after=None, steady_mug=False):
    """steady_mug: try the LEFT-hand steadying of the mug (off by default: in this stance the left hand
    cannot be brought from hanging to the mug - see REPORT - and the mug is not pushed by anything)."""
    m, d = s.m, s.d
    kb = m.body("env/" + KETTLE + "_body").id
    swb = m.body("env/" + SWITCH + "_body").id
    mb = m.body("env/" + MUG + "_body").id
    jb = m.body("env/" + JAR + "_body").id
    bR, bL = s.hands[R_], s.hands[L_]
    HAND_R, HAND_L = s.hand_parts[bR], s.hand_parts[bL]
    robot = {i for i in range(1, m.nbody) if not m.body(i).name.startswith("env/")}
    ign = [kb, mb, jb]
    WEIGHT = float(m.body_subtreemass[kb]) * 9.81
    res = dict(kettle_kg=round(float(m.body_subtreemass[kb]), 3), t_start=round(d.time, 2))
    phase_t = {}

    def note(what, **kv):
        p.note(what, **kv)
        phase_t[what] = round(d.time, 2)

    # ------------------------------------------------------------------ sensing (contact forces = F/T sensor)
    def contact_force(parts, objs):
        """Force (N, world) the objects put on the parts: the sum of their contact forces."""
        F, buf = np.zeros(3), np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b2 in parts and b1 in objs:
                sign = 1.0
            elif b1 in parts and b2 in objs:
                sign = -1.0
            else:
                continue
            mujoco.mj_contactForce(m, d, i, buf)
            F += sign * (c.frame.reshape(3, 3).T @ buf[:3])
        return F

    def support(objs):
        """Upward force (N) the objects get from anything that is not the robot."""
        f, buf = 0.0, np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 in objs) == (b2 in objs) or b1 in robot or b2 in robot:
                continue
            mujoco.mj_contactForce(m, d, i, buf)
            f += abs(buf[0] * c.frame[2])
        return f

    def touching(parts, objs):
        return any((int(m.geom_bodyid[c.geom1]) in parts and int(m.geom_bodyid[c.geom2]) in objs)
                   or (int(m.geom_bodyid[c.geom2]) in parts and int(m.geom_bodyid[c.geom1]) in objs)
                   for c in d.contact[:d.ncon])

    KOBJ = {kb, swb}
    Rk = lambda: d.xmat[kb].reshape(3, 3)                                       # noqa: E731
    K = lambda local: d.xpos[kb] + Rk() @ np.asarray(local, float)              # noqa: E731
    tilt = lambda: float(np.degrees(np.arccos(np.clip(Rk()[2, 2], -1, 1))))     # noqa: E731
    kyaw = lambda: math.degrees(math.atan2(Rk()[1, 0], Rk()[0, 0]))             # noqa: E731

    # ------------------------------------------------------------------ payload compensation (right hand)
    HOLD = {"on": False, "est": np.zeros(3)}

    def payload_tick(_s):
        F = contact_force(HAND_R, KOBJ) if HOLD["on"] else np.zeros(3)
        F = np.array([0.0, 0.0, min(F[2], 0.0)])       # the weight hanging on the hand only, never a push back
        HOLD["est"] += (F - HOLD["est"]) * min(1.0, 0.02 / 0.1)
        if np.linalg.norm(HOLD["est"]) > 0.05:
            s.wbc.payload[bR] = HOLD["est"].copy()
        else:
            s.wbc.payload.pop(bR, None)

    old_tick = p.mv.on_tick
    p.mv.on_tick = payload_tick
    tk = Tasks(s, on_tick=payload_tick)

    # ------------------------------------------------------------------ slip of the kettle in the hand
    SL = {"rel0": None, "max": 0.0}

    def rel():
        return d.xmat[bR].reshape(3, 3).T @ (d.xpos[kb] - d.xpos[bR])

    def slip():
        if SL["rel0"] is None:
            return 0.0
        v = float(np.linalg.norm(rel() - SL["rel0"])) * 1000
        SL["max"] = max(SL["max"], v)
        return round(v, 1)

    def stance():
        xy, yaw = tk._xy_yaw()
        return np.round(xy, 3).tolist(), round(math.degrees(yaw), 1)

    def standing_again():
        s.set_walk((0.0, 0.0, 0.0))
        stood = tk._wait_stand()
        for attr in ("_armw", "_ease0"):
            if hasattr(p, attr):
                delattr(p, attr)
        p.mv, p.w = Moves(s, on_tick=payload_tick), s.wbc
        p.root = p.mv.root
        return stood

    # the right arm's motors (shoulder, elbow, wrist) and fingers, by MotorLayer index
    mot = s.motors
    jname = lambda k: m.joint(int(m.dof_jntid[mot.dof[k]])).name if mot.dof[k] >= 0 else ""   # noqa: E731
    ARM_K = np.array([k for k in range(len(mot.ids)) if jname(k).startswith("right_")
                      and any(x in jname(k) for x in ("shoulder", "elbow", "wrist"))], int)
    ARM_KL = np.array([k for k in range(len(mot.ids)) if jname(k).startswith("left_")
                       and any(x in jname(k) for x in ("shoulder", "elbow", "wrist"))], int)
    WAIST_Q = [int(m.joint(n).qposadr[0]) for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")]

    def neutral_waist(seconds=1.0):
        """Before walking with the kettle: the controller's posture reference for the waist eases to zero
        (torso upright and square, as the walking network expects); the hand keeps its target, so the
        arm takes up the difference. The walking network then holds the waist where it is."""
        w = s.wbc
        q0 = np.array([w.qpos0[a] for a in WAIST_Q])
        n = max(1, int(round(seconds / p.mv.TICK)))
        for k in range(1, n + 1):
            u = smooth(k / n)
            for a, v in zip(WAIST_Q, q0):
                w.qpos0[a] = (1 - u) * v
            p.mv._tick(p.mv.TICK)
        p.hold(0.3)
        return [round(float(d.qpos[a]), 3) for a in WAIST_Q]

    def hold_arm_while_walking(carry):
        """While the walking network has the legs, the arms are held by its joint PD. The carrying (right)
        arm keeps its pose; our controller adds the feed-forward torque a real arm controller uses to
        hold a load (gravity of the arm + the measured load in the hand, J' F), and the fingers keep
        squeezing with the same finger gains as standing (closed target). All through the motors,
        clipped at their real limits. The other arm takes the walking pose."""
        loco = s.loco
        names = [m.joint(int(np.nonzero(m.jnt_qposadr == q)[0][0])).name for q in loco.hold_q]
        keep = np.array([carry and ((n.startswith("right_") and any(k in n for k in ("shoulder", "elbow", "wrist")))
                                    or n.startswith("waist")) for n in names])     # the torso stays as it is too
        saved = loco.hold_named.copy(), loco.hold_target.copy(), loco.hold_kp.copy(), loco.hold_kd.copy()
        loco.hold_named = loco.hold_named & ~keep
        if carry:
            # the free (left) arm hangs as in the maker's standing pose: the walking pose's forearm reaches
            # under the counter top into the open cabinet and catches on its divider panel (x 2.07)
            home_q = m.key("home").qpos
            for i, n in enumerate(names):
                if n.startswith("left_") and any(k in n for k in ("shoulder", "elbow", "wrist")):
                    loco.hold_named[i], loco.hold_target[i] = True, float(home_q[m.joint(n).qposadr[0]])
            closed = {a: (hi if high else lo) for a, lo, hi, high in s.fingers[bR]}
            rng = {a: (hi - lo) for a, lo, hi, high in s.fingers[bR]}
            for i, q in enumerate(loco.hold_q):
                q = int(q)
                if q in closed:
                    k = int(np.nonzero(mot.qadr == q)[0][0])
                    loco.hold_named[i], loco.hold_target[i] = True, closed[q]
                    loco.hold_kp[i] = 4 * mot.limit[k] / max(rng[q], 0.01)      # as standing (teleop.step)
                    loco.hold_kd[i] = 0.1 * mot.limit[k]
            orig = mot.command
            jac = np.zeros((3, m.nv))

            def command(d_, q_des=None, dq_des=None, tau_ff=None, kp=None, kd=None):
                if s.mode == "walk" and q_des is not None and tau_ff is None:
                    u = np.zeros(len(mot.ids))
                    dofs = mot.dof[ARM_K]
                    mujoco.mj_jac(m, d, jac, None, s.wbc.ee_pos(d, bR), bR)
                    u[ARM_K] = d.qfrc_bias[dofs] - (jac.T @ HOLD["est"])[dofs]
                    tau_ff = u
                return orig(d_, q_des, dq_des, tau_ff, kp, kd)
            mot.command = command
        return saved

    def release_walk_hold(saved):
        s.loco.hold_named, s.loco.hold_target, s.loco.hold_kp, s.loco.hold_kd = saved
        if "command" in mot.__dict__:
            del mot.command                                  # back to the class method

    def sidestep(goal_xy, yaw_goal=math.pi / 2, carry=False, vmax=walk_speed, tol=0.03, timeout=25.0):
        """Walk sideways (and a little forward/back) to goal_xy keeping the heading: the walking network's
        velocity command, ramped in, like byor goto's final approach. No turning with the kettle in hand."""
        saved = hold_arm_while_walking(carry)
        goal = np.asarray(goal_xy, float)
        t0 = d.time
        reason = "arrived"
        last = -1
        while True:
            xy, yaw = tk._xy_yaw()
            e = goal - xy
            if np.linalg.norm(e) < tol:
                break
            if s.checks.fell_at is not None:
                reason = "fell"
                break
            if d.time - t0 > timeout:
                reason = "timeout"
                break
            cy, sy = math.cos(yaw), math.sin(yaw)
            local = np.array([cy * e[0] + sy * e[1], -sy * e[0] + cy * e[1]])
            ramp = smooth((d.time - t0) / 0.8)
            v = np.clip(1.5 * local, [-0.10, -vmax], [0.10, vmax]) * ramp
            wz = float(np.clip(1.2 * ((yaw_goal - yaw + math.pi) % (2 * math.pi) - math.pi), -0.4, 0.4))
            s.set_walk((float(v[0]), float(v[1]), wz))
            tk._tick(0.1)
            if int(d.time - t0) != last:
                last = int(d.time - t0)
                log(f"   walk t {d.time - t0:4.1f} at {np.round(xy, 3)} yaw {math.degrees(yaw):.1f} cmd {np.round(v, 3)} "
                    f"mode {s.mode} kettle_z {d.xpos[kb][2]:.3f} slip {slip()}")
        stood = standing_again()
        release_walk_hold(saved)
        at, hd = stance()
        note("walked", to=list(map(float, goal)), reason=reason, stood=stood, at=at, heading_deg=hd,
             carry=carry, slip_mm=slip() if carry else None)
        return reason == "arrived" and stood

    ARM_R = {i for i in robot if m.body(i).name.startswith("right_") and any(k in m.body(i).name for k in ("shoulder", "elbow", "wrist", "hand"))}

    def what_touches():
        """Non-robot bodies touching the right arm or the kettle (other than the hand on the kettle)."""
        out = set()
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for a, o, g in ((b1, b2, c.geom2), (b2, b1, c.geom1)):
                if (a in ARM_R and o not in robot and o not in KOBJ) or (a in KOBJ and o not in robot and o not in KOBJ):
                    out.add(f"{m.body(a).name[-18:]}~{(m.geom(g).name or m.body(o).name)[-24:]}")
        return sorted(out)

    # ------------------------------------------------------------------ kettle pose -> hand target
    def hand_target_for(pk, Rkd):
        """Hand tool-point target and orientation that put the kettle at (pk, Rkd), from the kettle's
        MEASURED pose in the hand (it hangs a little turned in the grip)."""
        Rh = d.xmat[bR].reshape(3, 3)
        R_hk = Rh.T @ Rk()
        p_hk = Rh.T @ (d.xpos[kb] - d.xpos[bR])
        Rh_d = Rkd @ R_hk.T
        ph_d = pk - Rh_d @ p_hk
        return ph_d + Rh_d @ s.wbc.ee_off[bR], _mat2quat(Rh_d)

    def move_kettle(pk, Rkd, seconds, whole_body=False, via=None, fix=(0.6, 0.4), tol=0.004, leans=(0, 4, 8, 12)):
        """Kettle to a pose: hand move (whole-body balanced reach if asked: a planned whole-arm pose, a small
        forward lean at most), then corrections from the kettle's measured position (the loaded arm sits a
        little off its target)."""
        if Rkd is None:                       # position only: the hand keeps its turn (the kettle hangs ~15-25 deg
            Rkd = Rk().copy()                 # turned in the grip; levelling it drove the wrist onto its stop)
        tgt, q = hand_target_for(pk, Rkd)
        if whole_body:
            p.reach(R_, tgt, quat=q, seconds=seconds, via=via, settle=0.2, plan=p.plan_reach(R_, tgt, leans=leans))
        else:
            p.move_hand(R_, tgt, quat=q, seconds=seconds, via=via)
            p.hold(0.2)
        for sec in fix:
            e = pk - d.xpos[kb]
            if np.linalg.norm(e) < tol:
                break
            tgt, q = hand_target_for(pk, Rkd)
            p.move_hand(R_, s.wbc.ee_goal[bR] + e, quat=q, seconds=sec)
            p.hold(0.15)
        return float(np.linalg.norm(pk - d.xpos[kb]))

    # =========================================================== 1. walk to the kettle (empty hands, curled)
    for h in (0, 1):
        s.set_grip(h, 0.8)
    tk._tick(0.4)
    note("start", at=stance()[0], heading_deg=stance()[1])
    if walk_in:                                      # side-step right along the counter (no turning near it)
        sidestep(KETTLE_STANCE, vmax=0.25)
    at, hd = stance()
    note("at the kettle", at=at, heading_deg=hd)

    # =========================================================== 2. grip the handle (test_kettle5 grip)
    p.set_tool(R_, WRAP)
    Rk0 = Rk().copy()
    yaw0 = math.atan2(Rk0[1, 0], Rk0[0, 0])
    Q = quat_mul(_mat2quat(_rz(yaw0)), quat_from_axes([0, 1, 0], [-1, 0, 0], [0, 0, 1]))   # fingers +y, palm to the kettle
    home = d.xpos[kb].copy()
    g = K([BAR_X, 0.0, GZ])
    pre = g + _rz(yaw0) @ np.array([0.0, -0.10, 0.02])
    sh = d.xpos[s.shoulder[bR]].copy()
    p.grip(R_, 0.0, 0.3)
    p.reach(R_, pre, quat=Q, seconds=2.0, via=np.array([sh[0] + 0.05, d.qpos[1] + 0.04, 1.02]))
    p.settle_hand(R_, pre)
    p.move_hand(R_, g, quat=Q, seconds=1.1)
    err = p.settle_hand(R_, g)
    p.grip(R_, 1.0, 1.0)
    HOLD["on"] = True
    p.hold(0.4)
    SL["rel0"] = rel()
    note("gripped", place_err_mm=round(err * 1000, 1), on_bar=touching(HAND_R, KOBJ))

    # =========================================================== 3. lift off the base, carry pose
    z0 = float(d.xpos[kb][2])
    up = s.wbc.ee_goal[bR] + np.array([0.0, -0.01, 0.05])
    p.reach(R_, up, quat=Q, seconds=1.1)                       # straight up off the base first
    note("lifted", rise_cm=round((d.xpos[kb][2] - z0) * 100, 1), slip_mm=slip(), tilt_deg=round(tilt(), 1),
         base_support_N=round(support(KOBJ), 2))
    # close in front of the chest, upright, high enough to pass over the sachet tray
    carry_pk = np.array([d.xpos[kb][0] - 0.02, d.qpos[1] + 0.25, CARRY_BOTTOM_Z])
    # not levelled: levelling the kettle that hangs in the grip drove the wrist pitch onto its stop (1.58 rad)
    # for the rest of the step (no reach left to pour from another stance or to set it down); left as it
    # hangs (spout side down, it settles ~40-60 deg: no water reaches the spout below ~85 deg)
    e_c = move_kettle(carry_pk, None, 1.3, whole_body=True)
    if stop_after == "carry":
        return res
    note("carry pose", err_mm=round(e_c * 1000, 1), slip_mm=slip(), tilt_deg=round(tilt(), 1),
         kettle=np.round(d.xpos[kb], 3).tolist())
    p.hold(0.3)

    # the free (left) hand to its ready pose before walking: forearm forward, hand over the counter edge
    QL = quat_from_axes([0, 1, 0], [-1, 0, 0], [0, 0, 1])        # fingers forward, palm (-y_h) toward +x
    RQL = _quat2mat(QL)
    off_L0 = s.wbc.ee_off[bL].copy()
    p.set_tool(L_, (0.0, 0.0, 0.0))                               # the wrist itself is placed
    q_hang = s.wbc.ee_rot[bL].copy()
    s.set_grip(L_, 0.6)

    # =========================================================== 4. side-step to the mug carrying it
    ok = sidestep(POUR_STANCE, carry=True)
    if stop_after == "carried" or not ok:
        res.update(ok=False if not ok else None, error=None if ok else "did not arrive at the pour stance",
                   checks=s.checks.as_dict(), phases=phase_t)
        p.mv.on_tick = old_tick
        return res
    note("carried", ok=ok, slip_mm=slip(), tilt_deg=round(tilt(), 1), kettle=np.round(d.xpos[kb], 3).tolist())
    # =========================================================== 5. kettle to the pour start (beside the mug, upright)
    mug = d.xpos[mb].copy()
    T = np.array([mug[0], mug[1] - 0.004, mug[2] + MUG_RIM + TIP_ABOVE_RIM])    # spout tip target over the mug
    psi = math.radians(PSI_DEG)
    th_max = math.radians(tilt_deg)
    Rf = _rz(psi) @ _ry(-th_max)
    Pw = T - Rf @ (TIP - PIVOT)                                   # the pivot, fixed in the world while tilting

    def pose_at(th):
        R = _rz(psi) @ _ry(-th)
        return Pw - R @ PIVOT, R

    # the pour starts from the tilt the kettle already hangs at in the grip (spout side down), not from upright
    Rrel = _rz(psi).T @ Rk()
    th0 = float(np.clip(math.atan2(Rrel[2, 0], Rrel[2, 2]), 0.0, math.radians(70.0)))
    pk0, R0 = pose_at(th0)
    # the carry ends with the kettle over the mug: out to the right first (above the rim), then to the start
    move_kettle(np.array([pk0[0], pk0[1], max(pk0[2], d.xpos[kb][2]) + 0.02]), R0, 1.0)
    e0 = move_kettle(pk0, R0, 0.8)
    note("ready to pour", err_mm=round(e0 * 1000, 1), slip_mm=slip(), tilt_deg=round(tilt(), 1),
         yaw_deg=round(kyaw(), 1))

    # =========================================================== 6. left hand steadies the mug (palm on its left side)
    mug_steady = d.xpos[mb].copy()
    if steady_mug:
        rx, ry = float(d.qpos[0]), float(d.qpos[1])
        c_mug = mug + np.array([-0.022, 0.0, 0.045])                  # wrap point 2.2 cm short of the mug's axis
        wrist_at = c_mug - RQL @ np.asarray(MUG_TOOL)                 # where the wrist is then (palm beside the mug)
        front = wrist_at + np.array([-0.01, -0.07, 0.02])             # in front of that, fingers forward
        p.grip(L_, 0.25, 0.3)
        # hand only (the body stays as it stands with the kettle; a whole-body replan here tipped it over):
        # up at the side with the fingers still down (the stance is close to the counter, whose top is at
        # z 0.86 with the open cabinet under it), forward over the counter edge turning the fingers forward,
        # then in along the mug's left side, under the kettle's spout side (it waits up and to the right)
        p.move_hand(L_, front, quat=QL, seconds=0.9, via=(s.wbc.ee_goal[bL] + front) / 2 + [0.0, 0.0, 0.02])
        p.settle_hand(L_, front, tol=0.008, timeout=0.6)
        p.move_hand(L_, wrist_at, quat=QL, seconds=0.8)
        p.settle_hand(L_, wrist_at, tol=0.006, timeout=0.6)
        went = p.move_until(L_, (1.0, 0.0, 0.0), lambda: np.linalg.norm(contact_force(HAND_L, {mb})) > 1.5,
                            speed=0.015, max_dist=0.035, quat=QL, ease_in=0.15, ease_out=0.1)
        p.grip(L_, 0.4, 0.4)                                          # fingers round its front, lightly
        p.hold(0.2)
        mug_steady = d.xpos[mb].copy()
        note("mug steadied", palm_moved_mm=round(went * 1000, 1), mug_force_N=round(float(np.linalg.norm(contact_force(HAND_L, {mb}))), 2),
             mug_moved_mm=round(float(np.linalg.norm(d.xpos[mb] - mug)) * 1000, 1), lwrist=np.round(d.xpos[bL], 3).tolist(),
             lwrist_goal=np.round(wrist_at, 3).tolist())

    # =========================================================== 7. pour
    # tilt about the pivot with a slow closed loop on the kettle's measured pose (position and turn)
    corr = {"p": np.zeros(3), "r": np.zeros(3)}
    TICK = p.mv.TICK

    def follow(th, tau=0.35):
        pk, R = pose_at(th)
        ep = pk - d.xpos[kb]                                      # where the kettle still has to go
        et = (pk + R @ TIP) - K(TIP)                              # and its spout tip (half weight, sideways only)
        ep[:2] = 0.5 * ep[:2] + 0.5 * et[:2]
        er = _rotvec(R @ Rk().T)                                  # turn still missing (world)
        corr["p"] = np.clip(corr["p"] + ep * TICK / tau, -0.05, 0.05)
        corr["r"] = np.clip(corr["r"] + er * TICK / tau, -0.35, 0.35)
        tgt, q = hand_target_for(pk + corr["p"], _axis_angle_mat(corr["r"]) @ R)
        s.wbc.set_ee_target(bR, tgt, quat=q)
        p.mv._tick(TICK)

    def tip_err():
        return float(np.linalg.norm((K(TIP) - T)[:2])) * 1000

    GUARD = {"bad": 0.0, "abort": None}

    def unsafe():
        """A robot watching its spout: stop pouring if the kettle touches the mug, or the spout tip sits
        lower than 5 mm over the rim or outside the opening for 0.3 s (the arm cannot place it)."""
        if tilt() < 55.0:
            return False
        if touching(KOBJ, {mb}):
            GUARD["abort"] = "kettle touched the mug"
            return True
        tip = K(TIP)
        bad = (tip[2] - mug[2] - MUG_RIM) < 0.005 or (tilt() > 84.0 and float(np.linalg.norm((tip - mug)[:2])) > 0.0295)
        # (outside the opening matters only where water can flow: from ~85 deg, water agent)
        GUARD["bad"] = GUARD["bad"] + TICK if bad else 0.0
        if GUARD["bad"] > 0.3:
            GUARD["abort"] = "spout not over the mug opening (arm cannot place it)"
            return True
        return False

    pour = dict(t0=None, t1=None, tilt=[], tip_xy_mm=[], tip_above_rim_mm=[], slip=[])
    st = lambda xs, f: round(float(f(xs)), 1) if xs else None   # noqa: E731
    if e0 > 0.04:
        GUARD["abort"] = f"pour start not reached ({e0 * 1000:.0f} mm off)"
    t_tilt = 2.6
    n = int(round(t_tilt / TICK))
    th_now = th0
    for k in range(1, n + 1):
        if GUARD["abort"] or unsafe():
            break
        th_now = th0 + (th_max - th0) * smooth(k / n)
        follow(th_now)
    pour["t0"] = round(d.time, 2)
    k = 0
    while d.time - pour["t0"] < pour_s and not GUARD["abort"]:
        if unsafe():
            break
        follow(th_max)
        k += 1
        if k % 25 == 0:
            log(f"   pour t {d.time - pour['t0']:4.2f} tilt {tilt():5.1f} tip-mug {np.round((K(TIP) - mug)[:2] * 1000)} "
                f"above_rim {(K(TIP)[2] - mug[2] - MUG_RIM) * 1000:5.0f} corr {np.round(corr['p'] * 1000)} "
                f"rcorr {np.round(np.degrees(corr['r']), 1)} slip {slip()}")
        if k % 5 == 0:
            pour["tilt"].append(tilt())
            pour["tip_xy_mm"].append(tip_err())
            pour["tip_above_rim_mm"].append((K(TIP)[2] - mug[2] - MUG_RIM) * 1000)
            pour["slip"].append(slip())
    pour["t1"] = d.time
    note("poured", seconds=round(pour["t1"] - pour["t0"], 2), aborted=GUARD["abort"], tilt_deg=st(pour["tilt"], np.mean),
         tilt_min=st(pour["tilt"], np.min), tip_off_mug_centre_mm=st(pour["tip_xy_mm"], np.max),
         tip_above_rim_mm=st(pour["tip_above_rim_mm"], np.mean), slip_mm=slip(),
         mug_moved_mm=round(float(np.linalg.norm(d.xpos[mb] - mug_steady)) * 1000, 1))
    n = int(round(1.8 / TICK))
    th_back = math.radians(min(tilt(), tilt_deg))
    for k in range(1, n + 1):
        follow(th_back + (th0 - th_back) * smooth(k / n))
    p.hold(0.2)
    note("tilted back", tilt_deg=round(tilt(), 1), slip_mm=slip())
    res["pour"] = dict(aborted=GUARD["abort"], seconds=round(pour["t1"] - pour["t0"], 2),
                       tilt_mean_deg=st(pour["tilt"], np.mean), tilt_min_deg=st(pour["tilt"], np.min),
                       tip_off_centre_max_mm=st(pour["tip_xy_mm"], np.max),
                       tip_above_rim_mean_mm=st(pour["tip_above_rim_mm"], np.mean),
                       tip_above_rim_min_mm=st(pour["tip_above_rim_mm"], np.min))

    # =========================================================== 7. left hand lets go of the mug
    if steady_mug:
        p.grip(L_, 0.0, 0.4)
        p.move_hand(L_, s.wbc.ee_goal[bL] + [-0.04, 0.0, 0.01], quat=QL, seconds=0.5)        # palm off its side
        p.move_hand(L_, s.wbc.ee_goal[bL] + [0.0, 0.0, 0.07], quat=QL, seconds=0.5)          # up, clear of the rim
        p.move_hand(L_, np.array([d.qpos[0] - 0.20, d.qpos[1] - 0.02, 0.86]), quat=q_hang, seconds=1.0)
        p.grip(L_, 0.5, 0.3)
        note("mug released", mug_moved_mm=round(float(np.linalg.norm(d.xpos[mb] - mug_steady)) * 1000, 1))

    # =========================================================== 8. set the kettle down right of the mug
    # kettle up first (clear of the mug rim while the robot steps), then a short step back: the spot is close
    # in front of the robot and the hand needs room in front of the chest
    move_kettle(np.array([d.xpos[kb][0] + 0.03, d.xpos[kb][1] - 0.03, CARRY_BOTTOM_Z]), None, 0.8)
    rx, ry = float(d.qpos[0]), float(d.qpos[1])
    if not sidestep((rx + 0.02, min(ry, 1.95) - 0.06), carry=True, vmax=0.08):
        res.update(ok=False, error="step back before the set-down failed: kettle still held (no set-down tried)",
                   checks=s.checks.as_dict(), phases=phase_t, kettle_at=np.round(d.xpos[kb], 4).tolist(), t_end=round(d.time, 2))
        p.mv.on_tick = old_tick
        return res
    over = np.array([place[0], place[1], max(d.xpos[kb][2], 0.86 + 0.13)])
    move_kettle(np.array([d.xpos[kb][0], d.xpos[kb][1], over[2]]), None, 0.7)       # up: clear of the mug rim
    # whole-body (planned pose: knees, hips, lean) rather than the arm alone, which stalls above its target
    # with the 1.25 kg load
    e_o = move_kettle(over, None, 1.4, whole_body=True, fix=(0.7, 0.5))
    e_o = move_kettle(np.array([place[0], place[1], 0.86 + 0.04]), None, 1.2, whole_body=True, fix=(0.6, 0.4))
    note("over the spot", err_mm=round(e_o * 1000, 1), tilt_deg=round(tilt(), 1), yaw_deg=round(kyaw(), 1), slip_mm=slip())

    def hand_pen():
        """Deepest contact (m) between the palm/wrist link and the kettle: the righting stops before the
        handle is pressed into the hand (MuJoCo's soft contact let it sink 18 mm in pour20). The finger pads'
        own squeeze on the bar (a few mm in every grip here) is not counted."""
        dd = 0.0
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 == bR and b2 in KOBJ) or (b2 == bR and b1 in KOBJ):
                dd = max(dd, -float(c.dist))
        return dd

    def level_cmd():
        kz = Rk()[:, 2]
        axis = np.cross(kz, [0.0, 0.0, 1.0])
        sa = float(np.linalg.norm(axis))
        cmd = s.wbc.ee_rot[bR].copy()
        if sa < 1e-6:
            return cmd
        ang = math.atan2(sa, kz[2])
        return quat_mul(np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis / sa], cmd)

    # no levelling by the wrist: the kettle hangs a little tilted (spout side low); it is lowered by
    # contact, its low edge meets the counter first and it rights itself as the hand comes down
    target_xy = np.array(place, float)
    seated_ok = False
    for attempt in range(5):
        qd = s.wbc.ee_rot[bR].copy()
        went = p.lower_until(R_, lambda: support(KOBJ) > 0.5, speed=0.02, max_dist=0.06, quat=qd,
                             steer=lambda: (target_xy - d.xpos[kb][:2]))
        if support(KOBJ) > 0.5 and tilt() > 4.0:
            # its low edge rests on the counter: the hand is brought to where it is when the kettle stands
            # upright at the spot, with the wrist's turn left compliant (our controller's orientation weight
            # lowered and its reference following the hand: only damping), so the kettle turns upright about
            # its resting edge and the hand turns with it - the wrist is never driven onto its stop
            rel_t = Rk().T @ (s.wbc.ee_pos(d, bR) - d.xpos[kb])        # tool point in the kettle frame (measured)
            tool_up = np.array([target_xy[0], target_xy[1], 0.8605]) + _rz(math.radians(place_yaw)) @ rel_t
            w_rot0 = s.wbc.weights["ee_rot"]
            s.wbc.weights["ee_rot"] = 2.0
            p0 = s.wbc.ee_goal[bR].copy()

            def ref():
                # the hand's own turn (compliant tilt), its heading nudged so the handle turns to place_yaw
                # (left free, the kettle swung 45-50 deg)
                e = math.radians(float(np.clip(place_yaw - kyaw(), -15.0, 15.0)))
                return quat_mul(np.array([math.cos(e / 2), 0.0, 0.0, math.sin(e / 2)]), d.xquat[bR].copy())
            n = int(round(3.2 / p.mv.TICK))
            u, pen_max, held_back = 0.0, 0.0, 0
            pen0 = hand_pen()
            for k in range(1, n + 1):
                pen = hand_pen()
                pen_max = max(pen_max, pen)
                if pen > max(0.002, pen0 + 0.0015):   # the handle starts to press into the palm: do not advance,
                    held_back += 1               # let the hand ease back to where it is
                    s.wbc.set_ee_target(bR, 0.7 * s.wbc.ee_goal[bR] + 0.3 * s.wbc.ee_pos(d, bR), quat=ref())
                else:
                    u = min(1.0, u + 1.0 / n)
                    s.wbc.set_ee_target(bR, p0 + smooth(u) * (tool_up + [0.0, 0.0, 0.004] - p0), quat=ref())
                p.mv._tick(p.mv.TICK)
                if support(KOBJ) > 0.8 * WEIGHT and tilt() < 5.0:
                    break
            for _ in range(int(0.4 / p.mv.TICK)):
                s.wbc.set_ee_target(bR, s.wbc.ee_goal[bR], quat=ref())
                p.mv._tick(p.mv.TICK)
            s.wbc.weights["ee_rot"] = w_rot0
            s.wbc.set_ee_target(bR, s.wbc.ee_goal[bR], quat=d.xquat[bR].copy())
            qd = s.wbc.ee_rot[bR].copy()
            note("righted", progress=round(u, 2), held_back_ticks=held_back, palm_pen_start_mm=round(pen0 * 1000, 2),
                 palm_pen_max_mm=round(pen_max * 1000, 2),
                 tilt_deg=round(tilt(), 1), yaw_deg=round(kyaw(), 1), support_pct=round(100 * support(KOBJ) / WEIGHT),
                 hand_err_mm=round(float(np.linalg.norm(s.wbc.ee_pos(d, bR) - s.wbc.ee_goal[bR])) * 1000, 1))
        went += p.lower_until(R_, lambda: support(KOBJ) > 0.8 * WEIGHT and tilt() < 6.0, speed=0.01, max_dist=0.03, quat=qd,
                              ease_in=0.1)
        p.hold(0.4)
        off = float(np.linalg.norm(d.xpos[kb][:2] - target_xy))
        note("set down", attempt=attempt + 1, touches=what_touches(),
             hand_err_mm=round(float(np.linalg.norm(s.wbc.ee_pos(d, bR) - s.wbc.ee_goal[bR])) * 1000, 1),
             kettle_z=round(float(d.xpos[kb][2]), 3), lowered_mm=round(went * 1000, 1), support_pct=round(100 * support(KOBJ) / WEIGHT),
             off_spot_mm=round(off * 1000, 1), tilt_deg=round(tilt(), 1), yaw_deg=round(kyaw(), 1), slip_mm=slip())
        if support(KOBJ) > 0.5 * WEIGHT and tilt() < 7.0:     # resting on the counter (on its foot rim within 7 deg)
            seated_ok = True
            break
        p.move_hand(R_, s.wbc.ee_pos(d, bR) + [0.0, 0.0, 0.01], quat=qd, seconds=0.6)   # from where the hand IS
        for sec in (0.6, 0.5):
            e = np.r_[target_xy, 0.0] - d.xpos[kb]
            e[2] = 0.03 - (d.xpos[kb][2] - 0.86)
            p.move_hand(R_, s.wbc.ee_pos(d, bR) + e, quat=qd, seconds=sec)
            p.hold(0.25)
    res["slip_max_mm"] = round(SL["max"], 1)
    seated = d.xpos[kb].copy()
    if not seated_ok:                                 # never let go of a kettle the counter does not carry
        res.update(ok=False, error="kettle not seated after 5 tries: still held", checks=s.checks.as_dict(), phases=phase_t,
                   kettle_at=np.round(d.xpos[kb], 4).tolist(), t_end=round(d.time, 2))
        p.mv.on_tick = old_tick
        return res

    # =========================================================== 9. let go: unload first, then open, out, back
    HOLD["on"] = False
    p.move_hand(R_, s.wbc.ee_pos(d, bR), quat=s.wbc.ee_rot[bR].copy(), seconds=0.4)
    p.grip(R_, 0.0, 0.6)
    p.hold(0.2)
    p.move_hand(R_, s.wbc.ee_goal[bR] + [0.03, 0.0, 0.0], seconds=0.5)
    p.move_hand(R_, s.wbc.ee_goal[bR] + [0.01, -0.07, 0.03], seconds=0.7)
    p.move_hand(R_, s.wbc.ee_goal[bR] + [0.05, -0.05, 0.06], seconds=0.6)
    p.move_hand(R_, np.array([d.qpos[0] + 0.14, d.qpos[1] + 0.16, 0.98]), seconds=1.0)
    p.grip(R_, 0.3, 0.4)
    p.set_tool(L_, off_L0)
    p.hold(0.5)
    kyaw_end = kyaw()
    note("released", kettle_moved_on_release_mm=round(float(np.linalg.norm(d.xpos[kb] - seated)) * 1000, 1),
         kettle_at=np.round(d.xpos[kb], 3).tolist(), off_spot_mm=round(float(np.linalg.norm(d.xpos[kb][:2] - target_xy)) * 1000, 1),
         handle_yaw_deg=round(kyaw_end, 1), tilt_deg=round(tilt(), 1),
         mug_moved_total_mm=round(float(np.linalg.norm(d.xpos[mb][:2] - np.asarray(mug_steady[:2]))) * 1000, 1))
    res.update(kettle_at=np.round(d.xpos[kb], 4).tolist(), kettle_off_spot_mm=round(float(np.linalg.norm(d.xpos[kb][:2] - target_xy)) * 1000, 1),
               kettle_handle_yaw_deg=round(kyaw_end, 1), kettle_tilt_deg=round(tilt(), 1),
               kettle_moved_on_release_mm=round(float(np.linalg.norm(d.xpos[kb] - seated)) * 1000, 1),
               mug_at=np.round(d.xpos[mb], 4).tolist(), robot_at=stance()[0], robot_heading_deg=stance()[1],
               mug_moved_total_mm=round(float(np.linalg.norm(d.xpos[mb][:2] - np.asarray(mug_steady[:2]))) * 1000, 1),
               phases=phase_t, t_end=round(d.time, 2), seconds=round(d.time - res["t_start"], 2),
               checks=s.checks.as_dict())
    ck = res["checks"]
    res["ok"] = bool(ck["fell_at_s"] is None and ck["torque_limit_violations"] == 0 and seated_ok
                     and not res["pour"]["aborted"] and (res["pour"]["tilt_min_deg"] or 0) >= 85.0 and res["pour"]["tip_off_centre_max_mm"] <= 29.5
                     and res["pour"]["tip_above_rim_mean_mm"] > 15.0 and res["kettle_moved_on_release_mm"] <= 10.0
                     and res["mug_moved_total_mm"] <= 3.0
                     and res["kettle_tilt_deg"] < 4.0)
    res["steady_mug"] = steady_mug
    res["ok_note"] = "ok = no fall, no torque violation, pour >= 85 deg with the tip over the mug opening, kettle seated "                      "and released upright. NOT included (reported separately): slip <= 5 mm and contact depth <= 2 mm "                      "gates, and the left-hand steadying of the mug (not done unless steady_mug=True, which does not reach)."
    p.mv.on_tick = old_tick
    return res
