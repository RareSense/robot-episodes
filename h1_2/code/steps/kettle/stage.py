"""Kettle stage for the H1-2 tea episode, PhysX only. run(s, ctx) -> dict.

Takes a live h12_physx.Sim standing near the counter, both hands free, the mug on the counter at ctx["mug_xy"]. Uses only
world-measured poses. Never closes the app or exits. Tries to end standing with hands free whatever happens, and returns
status PASS or FAIL with its numbers. Hard rules: contact-only grip, mimic=True, URDF torque limits, no welds, no state
writes, no external forces. It only sets joint targets, gains on the hand's own drives, and walking commands.

The Sim must be built with mimic=True and with the kettle reporting contacts, i.e. watch=("object_0295", ...) or the
kettle in gates.contact_pre: PhysX cannot add a contact report after physics has loaded.

ctx: mug_xy (x, y) required. Optional: gates (a gates.Gates: hold/release/allow_move are called on it), mug ("object_0316"),
params {name: value} overriding P below, water (fluids.Fluids already attached; without it spill is NOT MEASURED).
Dry runs never return PASS: the step is to pour water, and a dry tilt pours nothing.

Pour (port of the G1 pour35 method): one continuous 50 Hz closed loop on the measured spout tip and kettle turn, with the kettle yawed psi deg toward the robot,
because yaw 0 ran wrist pitch, wrist yaw and elbow into their stops (gpu_kettle01). The arm IK pulls joints away from their limits in the null space.
Aim is corrected from live particle positions at rim height. The pour stops on the live mug reading, eased, and the kettle is set down left of the mug.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))    # release_paths.py lives at the code root
import release_paths as RP
import os
import sys
import time

sys.dont_write_bytecode = True
for _p in (RP.code("balance_support"), RP.code()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import h12_physx as H
import numpy as np
from crouch_support import CrouchSupport

K = "object_0295"
TIP = np.array([-0.094, 0.0, 0.205])       # spout mouth, kettle frame
BARX = 0.1175                              # handle bar centre, kettle frame
P = dict(stance_dx=-0.13, stance_dy=-0.57, drop=0.07, hx=0.155, hy=-0.128, hz=0.150, del_deg=-6.0, sq=0.6, low=1.0, tsq=0.5, tyaw=1.3, tp0=0.0,
         carry_y=2.21, carry_z=1.01, over=0.045, pour_dx=-0.12, pour_y=1.80, tilt_max=92.0, psi=28.0, aim_back=0.015, flow_from=72.0, flow_s=4.0, serve_ml=100.0,
         set_dx=-0.24, set_dy=0.04, set_crouch=0.05)
SLIP_MM = 5.0


class Abort(Exception):
    pass


def _yawq(deg):
    a = np.radians(deg) / 2
    return np.array([0, 0, np.sin(a), np.cos(a)])


def _ktilt(deg):
    a = -np.radians(deg) / 2
    return np.array([0.0, np.sin(a), 0.0, np.cos(a)])


def run(s, ctx):
    t_wall = time.time()
    par = dict(P, **ctx.get("params", {}))
    G = ctx.get("gates")
    W = ctx.get("water")
    mug_name = ctx.get("mug", "object_0316")
    mug_xy = np.asarray(ctx["mug_xy"], float)
    a = s.art
    out = dict(status="FAIL", reason="", stages={}, numbers={}, spill="NOT MEASURED" if W is None else None, water="dry" if W is None else "live",
               hand="right, side grip on the handle bar", params=par)
    num = out["numbers"]
    if not s.mimic:
        out["reason"] = "Sim built without mimic: results would not count"
        return out
    kpath = s.objs[K].GetPath().pathString
    kv = s.bodies[K] if K in s.bodies else s.sv.create_rigid_body_view(kpath)
    mv = s.bodies[mug_name] if mug_name in s.bodies else s.sv.create_rigid_body_view(s.objs[mug_name].GetPath().pathString)
    rh = [x for x in a.link_paths[0] if "/R_hand_base_link" in x or x.endswith("right_wrist_yaw_link")]
    try:
        rview = s.sv.create_rigid_contact_view([kpath], [rh], 512)
    except Exception as e:
        out["reason"] = f"no contact view on the kettle (build the Sim with it watched): {e!r}"
        return out
    jv = s.sv.create_rigid_body_view("/World/Objects/object_0309")
    mug_in, jar_in = mv.get_transforms()[0].astype(float)[:3].copy(), jv.get_transforms()[0].astype(float)[:3].copy()
    hold0 = s.hold.copy()                  # arm and hand targets at entry: the pose to come back to
    ty = s.finger_idx("R", "thumb_yaw")[0][0]
    kp_ty0 = float(s.kp[ty])
    st = dict(sup=None, held=False, rel0=None, q0=None, peak_slip=0.0)
    k0_z = float(kv.get_transforms()[0][2])

    def kettle():
        return kv.get_transforms()[0].astype(float)

    def mug():
        return mv.get_transforms()[0].astype(float)

    def tilt():
        return float(np.degrees(np.arccos(np.clip(H.qrot(kettle()[3:7], np.array([0, 0, 1.0]))[2], -1, 1))))

    def rc():
        m = rview.get_normal_contact_force_matrix(H.DT)[0].astype(float) if hasattr(rview, "get_normal_contact_force_matrix") else rview.get_contact_force_matrix(H.DT)[0].astype(float)
        return {rh[i].rsplit("/", 1)[-1].replace("R_", ""): round(float(np.linalg.norm(m[i])), 1) for i in range(len(rh)) if np.linalg.norm(m[i]) > 0.01}

    def hand_rel():
        h, k = s.hand("R"), kettle()
        return H.qrot(H.qinv(h[3:7]), k[:3] - h[:3]), H.qmul(H.qinv(h[3:7]), k[3:7])

    def slip():
        r, q = hand_rel()
        v = float(np.linalg.norm(r - st["rel0"])) * 1000
        st["peak_slip"] = max(st["peak_slip"], v)
        return round(v, 1), round(float(np.degrees(np.linalg.norm(H.qerr(st["q0"], q)))), 1)

    def stage(name, err=None):
        k = kettle()
        row = dict(t=round(s.t, 2), err_mm=None if err is None else round(err * 1000, 1), pelvis=np.round(s.pose()[0], 3).tolist(), hand=np.round(s.hand("R")[:3], 3).tolist(),
                   kettle=np.round(k[:3], 3).tolist(), tilt=round(tilt(), 1), force_n=round(sum(rc().values()), 1), slip=slip() if st["rel0"] is not None else None)
        out["stages"][name] = row
        print("[kettle]", name, row, flush=True)
        if s.fallen():
            raise Abort("robot fell at " + name)

    def set_gain(j, value):
        kp_ = a.get_dof_stiffnesses()[0].astype(float).copy()
        kp_[j] = value
        s.kp[j] = value                    # Sim.to_stand() restores s.kp: a live-only change is lost at the walk-to-stand switch
        a.set_dof_stiffnesses(s.f(kp_), s.idx)

    def to_walk():
        s.h[:], s.c[:], s.last[:] = 0, 0, 0
        s.target = a.get_dof_positions()[0].astype(float)[s.legs].copy()
        kp_, kd_ = a.get_dof_stiffnesses()[0].astype(float).copy(), a.get_dof_dampings()[0].astype(float).copy()
        kp_[s.legs], kd_[s.legs] = H.G["kp"], H.G["kd"]
        a.set_dof_stiffnesses(s.f(kp_), s.idx)
        a.set_dof_dampings(s.f(kd_), s.idx)
        s.want[:], s.cmd[:] = 0, 0
        s.mode = "walk"
        for _ in range(300):
            s.step()

    def walk_to(x, y, yaw=90.0, tol=0.025):
        if not s.go(x, y, yaw, tol=tol, log=lambda *z, **k: None):
            raise Abort(f"walk timed out on the way to {x:.2f}, {y:.2f}")
        gy = np.radians(yaw)
        fz = []
        for n in range(240):
            p, q = s.pose()
            yw = H.yaw_of(q)
            e = np.array([x, y]) - p[:2]
            c, sn = np.cos(yw), np.sin(yw)
            s.want = np.array([H.min_cmd(c * e[0] + sn * e[1], 1.0, 0.14, tol), H.min_cmd(-sn * e[0] + c * e[1], 1.0, 0.14, tol), np.clip(1.2 * H.wrap(gy - yw), -0.3, 0.3)])
            for _ in range(H.PERIOD):
                s.step()
                z = s.feet_xy()[:, 2]
                fz.append(z.copy())
            if s.fallen():
                raise Abort(f"fell walking to {x:.2f}, {y:.2f}")
            if n >= 60 and float(np.linalg.norm(e)) < 0.02 and np.all(z - np.min(fz[-120:], axis=0) < 0.004):
                break
        s.want = np.zeros(3)
        s.to_stand(0.6)
        s.hold_for(1.5)
        num.setdefault("stance_err_mm", []).append(round(float(np.linalg.norm(np.array([x, y]) - s.pose()[0][:2])) * 1000, 1))

    def side_to(x, y, tol=0.2, timeout=16.0):
        """Long move with the kettle in hand: side-step, heading kept. Sim.go faces the path, which swung the kettle in an arc over the tray to y 2.42 beside the jar
        (gpu_kettle01, aim01), and aim01 lost its grip as that turn ended."""
        gy = np.pi / 2
        t0 = s.t
        while s.t - t0 < timeout:
            p, q = s.pose()
            yw = H.yaw_of(q)
            e = np.array([x, y]) - p[:2]
            if float(np.linalg.norm(e)) < tol:
                return
            c, sn = np.cos(yw), np.sin(yw)
            s.want = np.array([H.min_cmd(c * e[0] + sn * e[1], 1.2, 0.15, 0.03), H.min_cmd(-sn * e[0] + c * e[1], 1.2, 0.2, 0.03), np.clip(1.2 * H.wrap(gy - yw), -0.3, 0.3)])
            for _ in range(H.PERIOD):
                s.step()
            if s.fallen():
                raise Abort("fell side-stepping with the kettle")
        print("[kettle] side-step timed out at", np.round(s.pose()[0], 3), flush=True)

    def crouch(drop, seconds=2.5):
        if st["sup"] is None:
            st["sup"] = CrouchSupport(s)
        r = st["sup"].crouch(drop, seconds=seconds, settle=0.8)
        num.setdefault("crouch", []).append(dict(drop=drop, passed=bool(r.get("passed")), err_mm=round(float(r.get("height_error_mm", -1)), 1)))
        return r

    def stand_up():
        if st["sup"] is not None:
            crouch(0.0)
            st["sup"].uninstall()
            st["sup"] = None

    def links_of(key):
        return ("thumb_proximal", "thumb_intermediate", "thumb_distal") if key == "thumb_pitch" else (f"{key}_proximal", f"{key}_intermediate", f"{key}_distal")

    def close(keys, rate=0.5, thresh=0.4, top=1.7, timeout=5.0):
        cur = {k: s.hold[s.finger_idx("R", k)[0][0]] for k in keys}
        done, t = {}, 0.0
        while len(done) < len(keys) and t < timeout:
            c = rc()
            for k in keys:
                if k not in done:
                    if any(c.get(n, 0.0) > thresh for n in links_of(k)):
                        done[k] = cur[k]
                    else:
                        cur[k] = min(top, cur[k] + rate * 0.02)
            s.fingers("R", {k: cur[k] for k in keys}, 0.02)
            t += 0.02
        return cur

    def relation():
        h, k = s.hand("R"), kettle()
        return H.qrot(H.qinv(k[3:7]), h[:3] - k[:3]), H.qmul(H.qinv(k[3:7]), h[3:7])

    def kservo(name, goal_p, goal_q, seconds=2.0, passes=2, rel=None):
        """Hand goal from the MEASURED kettle-in-hand relation: a shift in the grip is compensated and still counted as slip.
        rel: a relation frozen earlier. While the kettle is turning in the grip, re-measuring each pass makes the hand chase it (run s2)."""
        goal_p, goal_q = np.asarray(goal_p, float), np.asarray(goal_q, float)
        for n in range(1 if rel is not None else passes):
            p_hk, q_hk = rel if rel is not None else relation()
            s.ik_to(goal_p + H.qrot(goal_q, p_hk), H.qmul(goal_q, q_hk), seconds if n == 0 else 1.0, side="R", settle=30)
        e = float(np.linalg.norm(kettle()[:3] - goal_p))
        stage(name, e)
        return e

    def arm_home(seconds=2.5):
        """Right arm and hand back to the targets held at entry, as a joint-target ramp."""
        ai = s.arm_idx("R")
        q_from = s.hold[ai].copy()
        n = max(1, int(seconds / (H.PERIOD * H.DT)))
        for i in range(n):
            u = (i + 1) / n
            s.hold[ai] = q_from + u * u * (3 - 2 * u) * (hold0[ai] - q_from)
            for _ in range(H.PERIOD):
                s.step()
            if s.fallen():
                return

    def open_hand():
        s.fingers("R", dict(index=0.0, middle=0.0, ring=0.0, pinky=0.0), 1.5)
        s.fingers("R", dict(thumb_pitch=0.0), 0.6)

    ai = s.arm_idx("R")
    li = s.links.index("R_hand_base_link")
    lim_ = a.get_dof_limits()[0]
    jlo, jhi = lim_[ai, 0].astype(float), lim_[ai, 1].astype(float)
    jmid, jrng = (jlo + jhi) / 2, (jhi - jlo) / 2
    WROT = 0.5
    WV = np.r_[1, 1, 1, WROT, WROT, WROT]
    TICK = H.PERIOD * H.DT
    sv = dict(pp=None, pq=None, rp=None, rq=None, cp=np.zeros(3), cr=np.zeros(3))

    def servo_reset():
        sv.update(pp=None, pq=None, rp=None, rq=None, cp=np.zeros(3), cr=np.zeros(3))

    def servo(ref_p, ref_q, gain=0.12, null=0.02):
        """One 50 Hz arm tick: feedforward plus partial feedback on the measured hand, joint-limit avoidance in the null space. Joint targets only."""
        h = s.hand("R")
        if sv["pp"] is None:
            sv["pp"], sv["pq"] = ref_p.copy(), ref_q.copy()
        J = a.get_jacobians().reshape(-1, 6, a.max_dofs + 6)[li][:, 6 + ai].astype(float) * WV[:, None]
        ffp, ffo = ref_p - sv["pp"], H.qerr(ref_q, sv["pq"])
        sv["pp"], sv["pq"] = ref_p.copy(), ref_q.copy()
        e = np.r_[ffp + gain * np.clip(ref_p - h[:3], -0.02, 0.02), WROT * (ffo + gain * np.clip(H.qerr(ref_q, h[3:7]), -0.15, 0.15))]
        A = J @ J.T + 2e-3 * np.eye(6)
        qd = s.hold[ai].copy()
        dq = J.T @ np.linalg.solve(A, e) + (np.eye(7) - J.T @ np.linalg.solve(A, J)) @ (-null * (qd - jmid) / jrng ** 2)
        s.hold[ai] = s.bounded_arm_reference(ai, np.clip(qd + np.clip(dq, -0.025, 0.025), jlo, jhi))
        for _ in range(H.PERIOD):
            s.step()
        if s.fallen():
            raise Abort("robot fell in the arm servo")

    def rel_filtered(freeze=False):
        """Hand in the kettle frame, measured and low-passed, so slow turning in the grip is followed without chasing noise."""
        p_, q_ = relation()
        if sv["rp"] is None:
            sv["rp"], sv["rq"] = p_, q_
        elif not freeze:
            if np.dot(q_, sv["rq"]) < 0:
                q_ = -q_
            sv["rp"] = sv["rp"] + 0.08 * (p_ - sv["rp"])
            sv["rq"] = sv["rq"] + 0.08 * (q_ - sv["rq"])
            sv["rq"] = sv["rq"] / np.linalg.norm(sv["rq"])
        return sv["rp"], sv["rq"]

    def qvec(v):
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return np.r_[v / n * np.sin(n / 2), np.cos(n / 2)]

    def ktarget(pk, qk, tau=0.8, freeze=False):
        """One tick toward kettle pose (pk, qk). Measured errors feed a leaky bounded integrator (a wound-up correction drove the G1 spout onto the rim)."""
        k = kettle()
        sv["cp"] = np.clip(sv["cp"] * (1 - TICK / 3.0) + (pk - k[:3]) * TICK / tau, -0.05, 0.05)
        sv["cr"] = np.clip(sv["cr"] + H.qerr(qk, k[3:7]) * TICK / tau, -0.3, 0.3)
        q2 = H.qmul(qvec(sv["cr"]), qk)
        p_hk, q_hk = rel_filtered(freeze)
        servo(pk + sv["cp"] + H.qrot(q2, p_hk), H.qmul(q2, q_hk))

    def kmove(name, p_goal, q_goal, seconds, settle=0.4, stop=None, freeze=False):
        """Kettle origin to a pose on a smooth line, as a stream of servo ticks."""
        k = kettle()
        p0, q0 = k[:3].copy(), k[3:7].copy()
        q_goal = np.asarray(q_goal, float)
        if np.dot(q0, q_goal) < 0:
            q_goal = -q_goal
        p_goal = np.asarray(p_goal, float)
        n = max(1, int(seconds / TICK))
        for i in range(n + int(settle / TICK)):
            u = min(1.0, (i + 1) / n)
            u = u * u * (3 - 2 * u)
            q = q0 + u * (q_goal - q0)
            ktarget(p0 + u * (p_goal - p0), q / np.linalg.norm(q), freeze=freeze)
            if stop is not None and stop():
                break
        e = float(np.linalg.norm(kettle()[:3] - p_goal))
        stage(name, e)
        return e

    try:
        if s.mode != "walk":
            to_walk()
        k0 = kettle().copy()
        num["kettle_start"] = np.round(k0[:3], 4).tolist()
        spot = k0[:2] + np.array([par["stance_dx"], par["stance_dy"]])
        walk_to(spot[0], spot[1])
        stage("at_kettle")
        crouch(par["drop"], 2.0)
        QG = H.qmul(_yawq(par["del_deg"]), np.array([1.0, 0.0, 0.0, 0.0]))
        grasp = k0[:3] + np.array([BARX, 0.0, par["hz"]]) + H.qrot(_yawq(par["del_deg"]), np.array([par["hx"] - BARX, par["hy"], 0.0]))
        front = grasp + H.qrot(_yawq(par["del_deg"]), np.array([0, -0.20, 0.0])) + np.array([0.0, 0.0, 0.02])
        set_gain(ty, 10.0)
        s.fingers("R", dict(index=0.0, middle=0.0, ring=0.0, pinky=0.0, thumb_yaw=par["tyaw"], thumb_pitch=par["tp0"]), 1.2)
        stage("front", s.ik_to(front, QG, 2.5, side="R"))
        if float(np.degrees(np.linalg.norm(H.qerr(QG, s.hand("R")[3:7])))) > 3.0:
            raise Abort("hand off its angle before the approach")
        stage("approach", s.ik_to(grasp, QG, 2.0, side="R", stop=lambda: sum(rc().values()) > 0.5))
        cur = close(["thumb_pitch"], rate=0.25, top=0.6)
        cur2 = close(["index", "middle", "ring", "pinky"], rate=0.5)
        tgt = {k: min(1.7, cur2[k] + par["sq"] * (1.0 if k in ("index", "middle") else par["low"])) for k in cur2}   # soft ring and pinky: finger force in line with the thumb, measured 33 -> 14.6 deg turn
        tgt["thumb_pitch"] = cur["thumb_pitch"] + par["tsq"]
        s.fingers("R", tgt, 0.8)
        s.hold_for(0.5)
        if tilt() > 5.0 or float(np.linalg.norm(kettle()[:3] - k0[:3])) > 0.03:
            raise Abort(f"grip moved the kettle: tilt {tilt():.1f}, {np.linalg.norm(kettle()[:3] - k0[:3]) * 1000:.0f} mm")
        st["rel0"], st["q0"] = hand_rel()
        st["held"] = True
        stage("gripped")
        if G:
            G.allow_move(K)
            G.hold(K, "R_hand_base_link")
        h = s.hand("R")
        stage("lift", s.ik_to(h[:3] + np.array([0.0, 0.0, 0.03]), h[3:7], 2.0, side="R"))
        h = s.hand("R")
        stage("lift2", s.ik_to(h[:3] + np.array([0.0, 0.0, 0.06]), h[3:7], 2.0, side="R"))
        s.hold_for(1.5)
        stage("held")
        num["slip_after_lift"] = slip()
        num["rise_mm"] = round((kettle()[2] - k0[2]) * 1000, 1)
        if sum(rc().values()) < 1.0 or kettle()[2] < k0[2] + 0.02:
            raise Abort("kettle not held after the lift")
        if par["low"] < 1.0:               # lift soft, then lock: two fingers alone do not hold through the levelling (run r2)
            s.fingers("R", {k: min(1.7, cur2[k] + par["sq"]) for k in ("ring", "pinky")}, 1.0)
            s.hold_for(0.5)
            stage("locked")
        # Level before standing. Carrying it hanging keeps the hand 10 cm higher and stood the right elbow into object_0323 (x 2.73-2.88, y 1.83-1.98, z above 1.15): k_gpu02 was
        # thrown forward and lost the kettle. The levelled stand-up passed in gpu_kettle01, aim01 and aim02.
        num["hang_tilt"] = round(tilt(), 1)
        kservo("level", [kettle()[0], par["carry_y"], par["carry_z"]], _ktilt(0.0), 2.5)
        if tilt() > 12.0:
            raise Abort(f"could not level the kettle: tilt {tilt():.1f}")
        # object_0323 guard. Passing runs stood from elbow (2.74, 1.90, 1.11) and cleared it by about 1 mm in y. Only outside that does the kettle move in first, so a stance that
        # drifted right or high does not stand the elbow into it. Not triggered at the nominal stance.
        if "right_elbow_link" in s.links:
            ez = s.link_poses()[s.links.index("right_elbow_link")][:3].copy()
            num["elbow_before_stand"] = np.round(ez, 3).tolist()
            if ez[0] > 2.75 or ez[2] > 1.13:
                num["elbow_guard"] = True
                kservo("elbow_clear", [kettle()[0] - float(np.clip(ez[0] - 2.72, 0.02, 0.08)), par["carry_y"], par["carry_z"] - float(np.clip(ez[2] - 1.11, 0.0, 0.03))], _ktilt(0.0), 1.5)
        p_pre = s.pose()[0].copy()
        stand_up()
        stage("stood_up")
        num["stand_shift_mm"] = round(float(np.linalg.norm((s.pose()[0] - p_pre)[:2])) * 1000, 1)
        if num["stand_shift_mm"] > 120.0:
            raise Abort(f"thrown {num['stand_shift_mm']} mm while standing up")
        to_walk()
        px = mug_xy[0] + par["pour_dx"]
        side_to(px, par["pour_y"])
        walk_to(px, par["pour_y"])
        stage("at_pour_stance")
        num["slip_after_carry"] = slip()
        if sum(rc().values()) < 1.0 or kettle()[2] < 0.93:
            raise Abort("kettle lost in the carry")
        servo_reset()
        m0 = mug().copy()
        mxy = m0[:2].copy()
        rim = m0[2] + 0.085
        psi = float(par["psi"])
        sp_back = np.array([np.cos(np.radians(psi)), np.sin(np.radians(psi))])     # away from the way the spout points: water leaves toward the spout
        aim = sp_back * float(os.environ.get("KETTLE_AIM", par["aim_back"]))
        num["aim_start_mm"] = np.round(aim * 1000, 1).tolist()
        pz = rim + par["over"]
        log = num.setdefault("pour_log", [])
        fl = dict(off=[], top=0.0, abort=None, served=False, bad=0.0)

        def q_at(th):
            return H.qmul(_yawq(psi), _ktilt(th))

        def tip_now():
            k = kettle()
            return k[:3] + H.qrot(k[3:7], TIP)

        def pour_tick(th, z):
            """One tick with the spout on its aim point at height z and tilt th."""
            q = q_at(th)
            T = np.r_[mxy + aim, z]
            ktarget(T - H.qrot(q, TIP), q)
            if float(np.linalg.norm(mug()[:2] - m0[:2])) > 0.002:
                fl["abort"] = fl["abort"] or "mug moved more than 2 mm"
            return T

        def read_water(th):
            """Live reading: mug and spill, and the stream's position at rim height, which trims the aim."""
            tp = tip_now()
            row = dict(t=round(s.t, 2), cmd=round(th, 1), tilt=round(tilt(), 1), off_mm=round(float(np.linalg.norm(tp[:2] - mxy - aim)) * 1000, 1), axis_mm=round(float(np.linalg.norm(tp[:2] - mxy)) * 1000, 1),
                       over_mm=round((tp[2] - rim) * 1000, 1), slip=slip())
            if W is not None:
                W.sample(s.step)
                r = W.report()
                row.update(mug=r["mug_ml"], spill=r["spill_ml"], kettle=r["kettle_ml"], other=r["other_ml"])
                c, n = W.stream(rim, rim + 0.03)
                if n >= 8:
                    e = mxy - c[:2]
                    row.update(stream_mm=np.round(-e * 1000, 1).tolist(), n=n)
                    aim[:] = aim + np.clip(0.35 * e, -0.004, 0.004)
                    nrm = float(np.linalg.norm(aim))
                    if nrm > 0.045:
                        aim[:] = aim * (0.045 / nrm)
            log.append(row)
            print("[kettle] pour", row, flush=True)
            return row

        # A: spout over the mug, high, tilt below any flow. B: down to pour height. No water can leave here (flow starts near 78 deg with 400 ml).
        tp0 = tip_now()
        th0 = tilt()
        hi_z = pz + 0.07
        n = int(4.5 / TICK)
        q_start = kettle()[3:7].copy()
        for i in range(n):
            u = (i + 1) / n
            u = u * u * (3 - 2 * u)
            th = th0 + u * (55.0 - th0)
            qg = q_at(th)
            q = q_start + u * ((qg if np.dot(q_start, qg) >= 0 else -qg) - q_start)
            q /= np.linalg.norm(q)
            T = tp0 + u * (np.r_[mxy + aim, hi_z] - tp0)
            ktarget(T - H.qrot(q, TIP), q)
            if float(np.linalg.norm(mug()[:2] - m0[:2])) > 0.002:
                raise Abort("mug moved during the approach")
        stage("over_mug")
        n = int(2.5 / TICK)
        for i in range(n):
            u = (i + 1) / n
            u = u * u * (3 - 2 * u)
            pour_tick(55.0 + u * (par["flow_from"] - 55.0), hi_z + u * (pz - hi_z))
        for i in range(int(3.0 / TICK)):                # settle over the aim point before any flow; leave as soon as it is good
            pour_tick(par["flow_from"], pz)
            if i >= 25 and i % 5 == 0:
                r = read_water(par["flow_from"])
                if r["off_mm"] < 5.0:
                    break
        stage("ready_to_pour")
        num["ready_off_mm"] = round(float(np.linalg.norm(tip_now()[:2] - mxy - aim)) * 1000, 1)
        if fl["abort"]:
            raise Abort(fl["abort"])
        if num["ready_off_mm"] > 15.0:
            raise Abort(f"spout not over the mug before pouring: {num['ready_off_mm']} mm off")
        # C: eased climb (G1 pour35), stopped by the live mug reading. D: hold at the top if the mug still lacks water.
        th_a, th_b = float(par["flow_from"]), float(par["tilt_max"])
        n = int(par["flow_s"] / TICK)
        ease = 0.35
        th = th_a
        i = 0
        t_lim = s.t + par["flow_s"] + 7.0
        while s.t < t_lim:
            i += 1
            u = min(1.0, i / n)
            if u <= 1 - ease:
                f = u / (1 - ease / 2)
            else:
                v = (u - (1 - ease)) / ease
                f = ((1 - ease) + ease * (v - v * v / 2)) / (1 - ease / 2)
            th = th_a + (th_b - th_a) * f
            pour_tick(th, pz)
            off = float(np.linalg.norm(tip_now()[:2] - mxy - aim))
            if tilt() > 76.0:
                fl["off"].append(off * 1000)
            fl["bad"] = fl["bad"] + TICK if (tilt() > 76.0 and off > 0.025) else 0.0
            fl["top"] = max(fl["top"], tilt())
            if fl["bad"] > 0.25:
                fl["abort"] = fl["abort"] or f"spout {off * 1000:.0f} mm off its aim point while pouring"
            if fl["abort"]:
                break
            if i % 6 == 0:
                r = read_water(th)
                if W is not None and r["mug"] >= par["serve_ml"]:
                    fl["served"] = True
                    break
                if W is None and u >= 1.0:
                    break
        num.update(served_by_reading=fl["served"], pour_abort=fl["abort"], pour_tilt_reached=round(fl["top"], 1), aim_end_mm=np.round(aim * 1000, 1).tolist(),
                   flow_off_max_mm=round(max(fl["off"]), 1) if fl["off"] else None, flow_off_mean_mm=round(float(np.mean(fl["off"])), 1) if fl["off"] else None)
        stage("pour_top")
        # E: back below the flow angle fast with the spout still on the mug, then let the spout drain into it
        th1 = min(tilt(), th)
        n = int(0.8 / TICK)
        for i in range(n):
            u = (i + 1) / n
            pour_tick(th1 + (78.0 - th1) * u * u * (3 - 2 * u), pz)
        n = int(1.2 / TICK)
        for i in range(n):
            u = (i + 1) / n
            pour_tick(78.0 + (62.0 - 78.0) * u * u * (3 - 2 * u), pz)
        for i in range(int(1.0 / TICK)):
            pour_tick(62.0, pz)
        read_water(62.0)
        stage("tilted_back")
        if fl["abort"] == "mug moved more than 2 mm":
            raise Abort(fl["abort"])
        # F: up and away, across above the mug, down on free counter to its left. Rescue path of gpu_kettle01, which left mug and jar untouched.
        k = kettle()
        kmove("raise", k[:3] + np.array([0.0, 0.0, 0.10]), q_at(35.0), 2.5)
        k = kettle()
        z_hi = max(float(k[2]), 1.09)
        kmove("upright", [k[0], par["carry_y"], z_hi], _ktilt(0.0), 3.0)
        fx, fy = float(mxy[0] + par["set_dx"]), float(mxy[1] + par["set_dy"])
        kmove("across", [fx, fy, z_hi], _ktilt(0.0), 3.5)
        num["water_after_across"] = read_water(0.0)
        crouch(par["set_crouch"], 2.0)
        servo_reset()
        ztop = float(m0[2])
        stage("crouched")
        kmove("over_spot", [fx, fy, ztop + 0.05], _ktilt(0.0), 3.0)
        kmove("down", [fx, fy, ztop], _ktilt(0.0), 4.0, settle=0.8, stop=lambda: kettle()[2] < ztop + 0.003, freeze=True)
        s.hold_for(0.4)
        stage("placed")
        num["placed_tilt"] = round(tilt(), 1)
        if G:
            G.release(K)
        open_hand()
        st["held"] = False
        s.hold_for(0.4)
        stage("released")
        k1 = kettle().copy()
        h = s.hand("R")
        s.ik_to(h[:3] + np.array([0.012, 0.0, 0.0]), h[3:7], 0.8, side="R")
        h = s.hand("R")
        stage("hand_back", s.ik_to(h[:3] + np.array([0.0, -0.20, 0.06]), h[3:7], 2.0, side="R"))
        s.hold_for(0.5)
        k2 = kettle()
        jar = jv.get_transforms()[0].astype(float)
        num.update(moved_on_release_mm=round(float(np.linalg.norm(k2[:3] - k1[:3])) * 1000, 1), final_tilt=round(tilt(), 1), set_xy=np.round(k2[:2], 3).tolist(),
                   clear_mug_mm=round((float(np.linalg.norm(k2[:2] - mug()[:2])) - 0.142 - 0.034) * 1000, 1), clear_jar_mm=round((float(np.linalg.norm(k2[:2] - jar[:2])) - 0.142 - 0.05) * 1000, 1),
                   mug_moved_mm=round(float(np.linalg.norm(mug()[:2] - m0[:2])) * 1000, 2))
        num["set_down_ok"] = bool(tilt() < 3.0 and abs(k2[2] - ztop) < 0.006 and num["clear_mug_mm"] > 0 and num["clear_jar_mm"] > 0 and num["mug_moved_mm"] < 5.0)
        out["reason"] = "sequence finished"
    except Abort as e:
        out["reason"] = str(e)
    # always: hand open, arm home, standing. Runs on failure too, so a dropped kettle is reported, not hidden.
    try:
        if not s.fallen():
            if st["held"]:
                # do not just open the hand in the air (run s1: the kettle fell to the floor and hit the hip). Try free counter first.
                try:
                    if s.mode == "stand" and sum(rc().values()) > 1.0 and st["sup"] is None:
                        px_ = s.pose()[0][0]
                        fx = mug_xy[0] - 0.24 if px_ < mug_xy[0] + 0.3 else k0[0]    # left of the mug and jar, or the base when still at the kettle
                        fy = 2.21 if px_ < mug_xy[0] + 0.3 else k0[1]
                        ztop = 0.86 if px_ < mug_xy[0] + 0.3 else k0[2]
                        # up first, then across, then down: run s2 went straight across and pushed the mug 141 mm and the jar 68 mm
                        kk_ = kettle()
                        kservo("rescue_up", [kk_[0], fy, 1.08], _ktilt(0.0), 2.5, passes=1)
                        kservo("rescue_across", [fx, fy, 1.08], _ktilt(0.0), 3.0, passes=1)
                        kservo("rescue_over", [fx, fy, ztop + 0.05], _ktilt(0.0), 2.5, passes=1)
                        h = s.hand("R")
                        s.ik_to(h[:3] + np.array([0, 0, -0.07]), h[3:7], 2.5, side="R", settle=20, stop=lambda: kettle()[2] < ztop + 0.004)
                        s.hold_for(0.4)
                        stage("rescue_down")
                        num["rescue"] = dict(kettle=np.round(kettle()[:3], 3).tolist(), tilt=round(tilt(), 1))
                except Abort as e:
                    out["reason"] += f" | rescue: {e}"
                if G:
                    G.release(K)
                open_hand()
                h = s.hand("R")
                s.ik_to(h[:3] + np.array([0.012, -0.10, 0.08]), h[3:7], 1.5, side="R")
            s.fingers("R", dict(thumb_yaw=-0.1), 1.2)
            set_gain(ty, kp_ty0)
            arm_home()
            stand_up()
            if s.mode != "stand":
                s.want = np.zeros(3)
                s.to_stand(0.6)
            s.hold_for(1.0)
    except Exception as e:
        out["reason"] += f" | recovery failed: {e!r}"
    k = kettle()
    num.update(peak_slip_mm=round(st["peak_slip"], 1), end_kettle=np.round(k[:3], 3).tolist(), end_tilt=round(tilt(), 1), end_force_n=round(sum(rc().values()), 1), fallen=bool(s.fallen()),
               standing=bool(s.mode == "stand" and not s.fallen()), wall_s=round(time.time() - t_wall, 1), sim_t=round(s.t, 2))
    water_ok = False
    if W is not None:
        W.sample(s.step)
        out["water_report"] = W.report()
        # A wrist tilt is not evidence of a pour. Require a measured serving.
        num["minimum_serving_ml"] = 100.0
        water_ok = (out["water_report"].get("mug_ml", 0.0) >= num["minimum_serving_ml"]
                    and out["water_report"].get("spill_ml", float("inf")) <= 3.0)
    num["strict_slip_mm"] = round(st["peak_slip"], 1)
    num["strict_slip_pass"] = bool(st["peak_slip"] <= SLIP_MM)
    # status stays the strict stage result. task_status is the task alone, computed on every exit, and is reported beside it.
    ok = (out["reason"] == "sequence finished" and num.get("set_down_ok") and not num["fallen"] and st["peak_slip"] <= SLIP_MM and num.get("pour_tilt_reached", 0) >= 85.0 and water_ok)
    if out["reason"] == "sequence finished" and not water_ok:
        out["reason"] = "measured water serving or spill check did not pass"
    elif out["reason"] == "sequence finished" and not num.get("set_down_ok"):
        out["reason"] = "kettle not upright and clear of mug and jar"
    wr = out.get("water_report") or {}
    task = dict(mug_ml=wr.get("mug_ml"), spill_ml=wr.get("spill_ml"), kettle_tilt=round(tilt(), 1), kettle_released=bool(not st["held"]), kettle_on_surface=bool(k[2] < k0_z + 0.006),
                mug_moved_mm=round(float(np.linalg.norm(mv.get_transforms()[0].astype(float)[:3] - mug_in)) * 1000, 2),
                jar_moved_mm=round(float(np.linalg.norm(jv.get_transforms()[0].astype(float)[:3] - jar_in)) * 1000, 2))
    task_ok = bool(W is not None and water_ok and task["kettle_tilt"] < 3.0 and task["kettle_released"] and task["kettle_on_surface"] and task["mug_moved_mm"] < 5.0 and task["jar_moved_mm"] < 5.0)
    out["task"] = task
    out["task_status"] = "PASS" if task_ok else "FAIL"
    out["status"] = "PASS" if ok else "FAIL"
    print("[kettle] END status", out["status"], "task_status", out["task_status"], out["reason"], out["task"], num, flush=True)
    return out


# ---- late binding: if stage_live.py exists beside this file when the step starts, use it. Same contract, same rules.
def _live(fn_name, s, ctx, fallback):
    import importlib.util
    import os
    p = os.path.join(RP.code('steps', 'kettle'), "stage_live.py")
    if not os.path.exists(p):
        return fallback(s, ctx)
    sp = importlib.util.spec_from_file_location("stage_live_" + fn_name, p)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    print("[late-bind] using", p, "for", fn_name, flush=True)
    return getattr(m, fn_name)(s, ctx)


_orig_run = run


def run(s, ctx):
    return _live("run", s, ctx, _orig_run)
