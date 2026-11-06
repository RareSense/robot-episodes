"""Tea bag and stir stages for the H1-2 tea episode, PhysX only. run_teabag(s, ctx) -> dict, run_stir(s, ctx) -> dict.

Each takes a live h12_physx.Sim standing near the counter, both hands free, the mug on the counter at ctx["mug_xy"]. Only
world-measured poses are used (mug, bag, spoon read from PhysX). Neither calls app.close or sys.exit. Each tries to end standing with
hands free whatever happens and returns status PASS or FAIL with its numbers. Hard rules: contact-only grip, mimic=True, URDF torque
limits, no welds, no state writes, no external forces: only joint targets and walking commands are set. No gains are changed.

The Sim must be built by props.make_sim with mimic=True and gates.contact_pre on the props, the mug and the R_ hand links:
PhysX cannot add contact reports after physics has loaded.

ctx: mug_xy (x, y) required. gates: a gates.Gates, attached (hold/release/allow_move are called on it; its contact view is reused).
Without gates, give contact_paths (the hook's paths); gates are then NOT MEASURED. Optional: mug ("object_0316"), params {name: value}.
status PASS means the task checks passed, and, when gates are given, that slip stayed within 5 mm and the mug within 5 mm. The caller's
gates decide the episode: PASS here is not a claim about gates that were not read."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))    # release_paths.py lives at the code root
import release_paths as RP
import math
import sys
import time

sys.dont_write_bytecode = True
for _p in (RP.code("balance_support"), RP.code(), RP.code('props')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import h12_physx as H
import numpy as np
import props
from crouch_support import CrouchSupport

P = dict(stance_x=2.05, stance_y=1.86, walk_dy=-0.053, ikg=0.12, tooly_bag=0.156, tooly_spoon=0.154, sq_bag=0.12, sq_spoon=0.25, pdown=0.005, bag_pitch=0.0, bag_roll=0.0,
         spoon_pitch=0.0, spoon_roll=0.0, drop=0.08, pinch_l=0.122, offrim=0.008, out_l=0.12, out_s=2.5, over_z=0.02, inside=0.03, stir_z=0.066, stir_r=0.007, turns=3.5,
         stir_period=2.2, bowl_sub=0.010, stir_z_min=0.030, spoon_thumb_pitch=0.3, put_dx=0.10, put_y=2.085, tip_deg=50.0, bag_pre=(3.5, 0.08, 2.27), spoon_pre=(0.0, 3.9, 0.0))
OPEN_A = dict(thumb_yaw=1.1, thumb_pitch=0.3, index=0.35, middle=1.6, ring=1.6, pinky=1.6)
FREE = dict(thumb_yaw=-0.1, thumb_pitch=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0)
RIM, HOLD_MM = 0.085, 5.0


class Abort(Exception):
    pass


def mat_of(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_of(R):
    q = props.q_from_R(R)
    i = q.GetImaginary()
    return np.array([i[0], i[1], i[2], q.GetReal()])


def pinch_frame(nrm, roll_deg, pitch_deg):
    """Right hand: fingers (-y_h) along nrm, palm down at roll 0 (x_h up), pitch > 0 tips the fingers down. Columns x_h y_h z_h."""
    up = np.array([0.0, 0.0, 1.0])
    yh = -np.asarray(nrm, float) / np.linalg.norm(nrm)
    xh = up - (up @ yh) * yh
    xh /= np.linalg.norm(xh)
    zh = np.cross(xh, yh)
    r = math.radians(roll_deg)
    xh, zh = math.cos(r) * xh + math.sin(r) * zh, -math.sin(r) * xh + math.cos(r) * zh
    t = math.radians(pitch_deg)
    yh2 = math.cos(t) * yh + math.sin(t) * xh
    xh2 = math.cos(t) * xh - math.sin(t) * yh
    return np.c_[xh2, yh2, np.cross(xh2, yh2)]


def pre_rot(Rq, deg):
    """Hand frame turned by a rotation vector (degrees, hand axes) before the pinch, so the object meets the finger pad as it will sit once held instead of turning into place at the lift."""
    v = np.radians(np.asarray(deg, float))
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return Rq
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return Rq @ (np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K)


def to_tool(tool_w, Rq, tool):
    return np.asarray(tool_w, float) - Rq @ np.asarray(tool, float)


def ik_free(s, goal_p, seconds=2.0, side="R", tol=0.003, settle=60, stop=None, gain=0.12, axis_h=0, ref=(0, 0, 1.0), w_rot=0.5):
    """Sim.ik_to with the hand free to turn about `ref`: position plus hand axis axis_h kept on ref. Arm joint targets only."""
    a = s.art
    ai = s.arm_idx(side)
    li = s.links.index(f"{side}_hand_base_link")
    lim = a.get_dof_limits()[0]
    lo, hi = lim[ai, 0].astype(float), lim[ai, 1].astype(float)
    ref = np.asarray(ref, float) / np.linalg.norm(ref)
    Pj = np.eye(3) - np.outer(ref, ref)
    p0 = s.hand(side)[:3].copy()
    goal_p = np.asarray(goal_p, float)
    return _ik_loop(s, lambda u: p0 + u * u * (3 - 2 * u) * (goal_p - p0), seconds, settle, side, ai, li, lo, hi, ref, Pj, axis_h, w_rot, gain, stop, tol, goal_p)


def ik_path(s, fn, seconds, side="R", stop=None, gain=0.12, axis_h=0, ref=(0, 0, 1.0), w_rot=0.5):
    a = s.art
    ai = s.arm_idx(side)
    li = s.links.index(f"{side}_hand_base_link")
    lim = a.get_dof_limits()[0]
    lo, hi = lim[ai, 0].astype(float), lim[ai, 1].astype(float)
    ref = np.asarray(ref, float) / np.linalg.norm(ref)
    return _ik_loop(s, fn, seconds, 0, side, ai, li, lo, hi, ref, np.eye(3) - np.outer(ref, ref), axis_h, w_rot, gain, stop, 0.0, None)


def _ik_loop(s, fn, seconds, settle, side, ai, li, lo, hi, ref, Pj, axis_h, w_rot, gain, stop, tol, goal_p):
    a = s.art
    steps = max(1, int(seconds / (H.PERIOD * H.DT)))
    qd = s.hold[ai].copy()
    prev = np.asarray(fn(0.0), float)
    worst, err = 0.0, 1.0
    for k in range(steps + settle):
        ref_p = np.asarray(fn(min(1.0, (k + 1) / steps)), float)
        h = s.hand(side)
        Jf = a.get_jacobians().reshape(-1, 6, a.max_dofs + 6)[li][:, 6 + ai].astype(float)
        J = np.r_[Jf[:3], w_rot * (Pj @ Jf[3:])]
        ep = ref_p - h[:3]
        worst = max(worst, float(np.linalg.norm(ep)))
        eo = Pj @ np.cross(mat_of(h[3:7])[:, axis_h], ref)
        ff = ref_p - prev
        prev = ref_p
        e = np.r_[ff + gain * np.clip(ep, -0.02, 0.02), w_rot * gain * np.clip(eo, -0.15, 0.15)]
        dq = J.T @ np.linalg.solve(J @ J.T + 2e-3 * np.eye(6), e)
        qd = np.clip(qd + np.clip(dq, -0.025, 0.025), lo, hi)
        s.hold[ai] = qd
        for _ in range(H.PERIOD):
            s.step()
        if s.fallen():
            raise Abort("robot fell during an arm move")
        if stop is not None:
            stop()
        if goal_p is not None:
            err = float(np.linalg.norm(goal_p - s.hand(side)[:3]))
            if k >= steps and err < tol:
                break
    return err if goal_p is not None else worst


class Ctx:
    """Everything one stage reads, from the live Sim."""

    def __init__(self, s, ctx, name):
        self.s, self.ctx, self.name = s, ctx, name
        self.par = dict(P, **ctx.get("params", {}))
        self.g = ctx.get("gates")
        self.mug = ctx.get("mug", "object_0316")
        self.out = dict(status="FAIL", reason="", task_status="FAIL", task_reason="stage did not reach its task check", step=name, stages={}, numbers={}, gates_given=self.g is not None,
                        hand="right, thumb-index tip pinch")
        self.num = self.out["numbers"]
        self.t0 = time.time()
        self.t_sim0 = s.t
        pp = props.prop_paths()
        self.V = {k: s.sv.create_rigid_body_view(pp[v]) for k, v in dict(bag="teabag_bag", tag="teabag_tag", spoon="stir_spoon").items()}
        self.paths = pp
        self.mv = s.sv.create_rigid_body_view("/World/Objects/" + self.mug)
        if self.g is not None and self.g.cv is not None:
            self.cv, self.sensors = self.g.cv, [str(x) for x in self.g.sensor_paths]
        else:
            self.cv = s.sv.create_rigid_contact_view(list(ctx["contact_paths"]), max_contact_data_count=4096)
            self.sensors = [str(x) for x in np.ravel(self.cv.sensor_paths)]
        self.sup = None
        self.q_start = s.hold.copy()
        self.mug0 = self.mug_pose().copy()
        self.mug_max = 0.0
        self.tilt_max = 0.0
        self.num["mug_start"] = np.round(self.mug0[:3], 4).tolist()

    def pose(self, k):
        return self.V[k].get_transforms()[0].astype(float)

    def mug_pose(self):
        return self.mv.get_transforms()[0].astype(float)

    def tilt(self):
        return float(np.degrees(np.arccos(np.clip(H.qrot(self.mug_pose()[3:7], np.array([0, 0, 1.0]))[2], -1, 1))))

    def pairs(self):
        f, pt, nrm, sep, cnt, start, other = self.cv.get_raw_contact_data(H.DT)
        sep = np.asarray(sep).reshape(-1)
        f = np.asarray(f).reshape(len(sep), -1)
        cnt, start, other = np.asarray(cnt).reshape(-1), np.asarray(start).reshape(-1), np.asarray(other).reshape(-1)
        out = {}
        for i in np.nonzero(cnt)[0]:
            a_, b_ = int(start[i]), int(start[i] + cnt[i])
            for j, n in enumerate(self.cv.get_other_actor_paths_from_ids(other[a_:b_])):
                key = (self.sensors[i].rsplit("/", 1)[-1], str(n).rsplit("/", 1)[-1])
                F, d = out.get(key, (0.0, 0.0))
                out[key] = (F + float(np.linalg.norm(f[a_ + j])), max(d, -float(sep[a_ + j]) * 1000))
        return out

    def on(self, name, prefix="R_"):
        return {b: round(v[0], 3) for (a_, b), v in self.pairs().items() if a_ == name and b.startswith(prefix)}

    def stage(self, name, err=None):
        s = self.s
        d = float(np.linalg.norm(self.mug_pose()[:3] - self.mug0[:3])) * 1000
        self.mug_max = max(self.mug_max, d)
        self.tilt_max = max(self.tilt_max, self.tilt())
        row = dict(t=round(s.t, 2), wall=round(time.time() - self.t0), err_mm=None if err is None else round(err * 1000, 1), hand=np.round(s.hand("R")[:3], 3).tolist(),
                   bag=np.round(self.pose("bag")[:3], 4).tolist(), spoon=np.round(self.pose("spoon")[:3], 4).tolist(), mug=np.round(self.mug_pose()[:3], 4).tolist(), pelvis=np.round(s.pose()[0], 3).tolist())
        self.out["stages"][name] = row
        print(self.name, name, row, flush=True)
        if s.fallen():
            raise Abort("robot fell at " + name)

    def creep(self, who, need, rate=0.25, top=1.3, thresh=0.05, timeout=4.0):
        s = self.s
        i = s.finger_idx("R", "index")[0][0]
        cur, t = float(s.hold[i]), 0.0
        while t < timeout and cur < top:
            c = self.on(who)
            if all(any(v > thresh for k, v in c.items() if n in k) for n in need):
                return cur, c
            cur += rate * 0.02
            s.fingers("R", dict(index=cur), 0.02)
            t += 0.02
        return cur, self.on(who)

    def walk_to_stance(self):
        """Walk to the stance only if the robot is not there already. Same walk as mug_episode."""
        s = self.s
        gx, gy_ = self.par["stance_x"], self.par["stance_y"]             # wanted pelvis position once standing
        x, y = gx, gy_ + self.par["walk_dy"]                             # measured: the pelvis ends about 5 cm ahead of the walk target after to_stand
        e0 = float(np.linalg.norm(np.array([gx, gy_]) - s.pose()[0][:2]))
        yaw_e = abs(math.degrees(H.wrap(math.radians(90.0) - H.yaw_of(s.pose()[1]))))
        self.num["stance_err_before_mm"] = round(e0 * 1000, 1)
        if e0 < 0.03 and yaw_e < 3.0:
            return
        a = s.art
        s.h[:], s.c[:], s.last[:] = 0, 0, 0
        s.target = a.get_dof_positions()[0].astype(float)[s.legs].copy()
        kp, kd = a.get_dof_stiffnesses()[0].astype(float).copy(), a.get_dof_dampings()[0].astype(float).copy()
        kp[s.legs], kd[s.legs] = H.G["kp"], H.G["kd"]
        a.set_dof_stiffnesses(s.f(kp), s.idx)
        a.set_dof_dampings(s.f(kd), s.idx)
        s.want[:], s.cmd[:] = 0, 0
        s.mode = "walk"
        for _ in range(300):
            s.step()
        if not s.go(x, y, 90.0, tol=0.025, log=lambda *z, **k: None):
            raise Abort("walk to the stance timed out or fell")
        gy = math.radians(90.0)
        fz = []
        for n in range(240):
            p, q = s.pose()
            yw = H.yaw_of(q)
            e = np.array([x, y]) - p[:2]
            c, sn = math.cos(yw), math.sin(yw)
            s.want = np.array([H.min_cmd(c * e[0] + sn * e[1], 1.0, 0.14, 0.025), H.min_cmd(-sn * e[0] + c * e[1], 1.0, 0.14, 0.025), float(np.clip(1.2 * H.wrap(gy - yw), -0.3, 0.3))])
            for _ in range(H.PERIOD):
                s.step()
                z = s.feet_xy()[:, 2]
                fz.append(z.copy())
            if n >= 60 and float(np.linalg.norm(e)) < 0.02 and np.all(z - np.min(fz[-120:], axis=0) < 0.004):
                break
        s.want = np.zeros(3)
        s.to_stand(0.6)
        s.hold_for(1.5)
        self.num["stance_err_after_mm"] = round(float(np.linalg.norm(np.array([gx, gy_]) - s.pose()[0][:2])) * 1000, 1)
        if s.fallen():
            raise Abort("fell at the stance")

    def left_aside(self):
        s = self.s
        st = s.pose()[0]
        self.stage("left_aside", s.ik_to(np.array([st[0] - 0.33, st[1] + 0.08, 1.10]), None, 1.5, side="L", gain=self.par["ikg"]))

    def finish(self):
        """Hands free and standing, whatever happened: open the pinch, hand up, rise, arms back to where this stage found them."""
        s = self.s
        out = self.out
        try:
            for k in ("bag", "spoon"):
                if self.g is not None:
                    self.g.release(self.paths["teabag_bag" if k == "bag" else "stir_spoon"])
            s.fingers("R", dict(index=0.35), 0.6)
            h = s.hand("R")[:3]
            z = max(h[2], 1.17)
            ik_free(s, np.array([h[0], h[1] - 0.03, z]), 1.5, ref=mat_of(s.hand("R")[3:7])[:, 0], gain=self.par["ikg"])
            if self.sup is not None:
                self.num["rise"] = self.sup.crouch(0.0, seconds=2.5, settle=0.8)
                self.sup.uninstall()
                self.sup = None
            q0 = s.hold.copy()
            arms = np.r_[s.arm_idx("L"), s.arm_idx("R")]
            n = int(2.5 / (H.PERIOD * H.DT))
            for i in range(n):
                u = (i + 1) / n
                u = u * u * (3 - 2 * u)
                s.hold[arms] = q0[arms] + u * (self.q_start[arms] - q0[arms])
                for _ in range(H.PERIOD):
                    s.step()
            s.fingers("R", FREE, 1.0)
            s.hold_for(2.5)
        except Exception as e:
            out["finish_error"] = repr(e)
        self.mug_max = max(self.mug_max, float(np.linalg.norm(self.mug_pose()[:3] - self.mug0[:3])) * 1000)
        touching = {k: round(v[0], 2) for k, v in self.pairs().items() if k[0].startswith("R_") and v[0] > 0.01}
        self.num.update(hands_touching=touching, fallen=bool(s.fallen()), pelvis_end=np.round(s.pose()[0], 3).tolist(), mug_end=np.round(self.mug_pose()[:3], 4).tolist(), mug_moved_mm=round(self.mug_max, 2),
                        mug_tilt_deg=round(self.tilt_max, 2), sim_s=round(s.t - self.t_sim0, 2), wall_s=round(time.time() - self.t0, 1))
        if s.fallen():
            out["status"] = "FAIL"
            out["reason"] = (out["reason"] + "; robot fell").strip("; ")
        if s.fallen() or self.mug_max >= HOLD_MM:          # task_status also covers the clean-up, which runs after the task check
            out["task_status"] = "FAIL"
            out["task_reason"] = (out["task_reason"] + f"; fallen {bool(s.fallen())}, mug moved {self.mug_max:.1f} mm by the end").strip("; ")
        return out


def run_teabag(s, ctx):
    c = Ctx(s, ctx, "teabag")
    out, num, par = c.out, c.num, c.par
    if not s.mimic:
        out["reason"] = "Sim built without mimic: results would not count"
        return out
    try:
        _teabag(s, c)
    except Abort as e:
        out["reason"] = str(e)
    return c.finish()


def _teabag(s, c):
    par, num, out, g = c.par, c.num, c.out, c.g
    IKG = par["ikg"]
    TOOL = np.array([-0.058, -par["tooly_bag"], -0.0257])
    BAG = c.paths["teabag_bag"]
    c.walk_to_stance()
    b = c.pose("bag")
    Rb = mat_of(b[3:7])
    hx = props.BAGF_HALF[0]
    up = Rb[:, 0] * np.sign(Rb[2, 0])
    top = b[:3] + up * hx
    nrm = Rb[:, 2].copy()
    nrm[2] = 0
    nrm /= np.linalg.norm(nrm)
    if nrm[1] < 0:
        nrm = -nrm
    Rq = pre_rot(pinch_frame(nrm, par["bag_roll"], par["bag_pitch"]), par["bag_pre"])
    QA = quat_of(Rq)
    tgt = top - up * par["pdown"]
    st = s.pose()[0]
    s.fingers("R", OPEN_A, 1.2)
    c.left_aside()
    s.hold_for(1.0)
    c.stage("up_side", s.ik_to(to_tool((st[0] + 0.22, st[1] + 0.12, 1.05), Rq, TOOL), QA, 1.6, side="R", gain=IKG))
    c.stage("up_front", s.ik_to(to_tool((st[0] + 0.21, st[1] + 0.23, 1.10), Rq, TOOL), QA, 1.4, side="R", gain=IKG))
    c.stage("over_bag", s.ik_to(to_tool(tgt + np.array([0, 0, 0.05]), Rq, TOOL), QA, 1.8, side="R", gain=IKG))
    e = s.ik_to(to_tool(tgt, Rq, TOOL), QA, 1.3, side="R", gain=IKG)
    c.stage("at_bag", e)
    s.hold_for(0.3)
    if e > 0.006:
        raise Abort(f"hand {e * 1000:.1f} mm off at the bag")
    cur, cc = c.creep("teabag_bag", ("index", "thumb"))
    num["creep_index"] = round(cur, 3)
    if not ("R_index_intermediate" in cc or "R_index_proximal" in cc) or not any("thumb" in k for k in cc):
        raise Abort(f"pinch not closed on the bag: {cc}")
    s.fingers("R", dict(index=cur + par["sq_bag"]), 0.4)
    s.hold_for(0.3)
    num["pinch_N"] = c.on("teabag_bag")
    c.stage("pinched")
    if g is not None:
        g.hold(BAG, "R_hand_base_link")
    rel = lambda: H.qrot(H.qinv(s.hand("R")[3:7]), c.pose("bag")[:3] - s.hand("R")[:3])
    rel0 = rel()
    sm = [0.0]
    spot_b = np.array([float(np.sign(Rb[2, 0])) * (hx - par["pdown"]), 0.0, 0.0])    # nominal pinched spot, bag frame

    def in_hand():
        hh, bb = s.hand("R"), c.pose("bag")
        Rbb = mat_of(bb[3:7])
        return H.qrot(H.qinv(hh[3:7]), bb[:3] + Rbb @ spot_b - hh[:3]), H.qrot(H.qinv(hh[3:7]), Rbb[:, 0])

    spot0, ax0 = in_hand()
    sp = [0.0, 0.0]

    def track():
        sm[0] = max(sm[0], float(np.linalg.norm(rel() - rel0)) * 1000)
        p_, a_ = in_hand()                       # read only: separates the bag turning in the pinch from the pinch sliding on the bag
        sp[0] = max(sp[0], float(np.linalg.norm(p_ - spot0)) * 1000)
        sp[1] = max(sp[1], float(np.degrees(np.arccos(np.clip(a_ @ ax0, -1, 1)))))
        return False

    h = s.hand("R")[:3]
    c.stage("lift1", s.ik_to(h + np.array([0, 0, 0.08]), QA, 1.6, side="R", gain=IKG, stop=track))
    s.hold_for(0.4)
    h = s.hand("R")[:3]
    c.stage("lift2", s.ik_to(h + np.array([0, -0.03, 0.08]), QA, 1.4, side="R", gain=IKG, stop=track))
    s.hold_for(0.6)
    num["bag_rise_mm"] = round((c.pose("bag")[2] - b[2]) * 1000, 1)
    if c.pose("bag")[2] < b[2] + 0.05:
        raise Abort("bag not lifted")
    m = c.mug_pose()
    rim = m[2] + RIM
    bot = lambda: float(c.pose("bag")[2] - hx)

    def move_bag(xy, z, seconds, n=3):
        for i in range(n):
            d = np.array([xy[0] - c.pose("bag")[0], xy[1] - c.pose("bag")[1], z - bot()])
            if np.linalg.norm(d) < 0.002:
                break
            ik_free(s, s.hand("R")[:3] + d, seconds if i == 0 else 0.8, stop=track, gain=IKG)
            s.hold_for(0.3)
        return float(np.linalg.norm([xy[0] - c.pose("bag")[0], xy[1] - c.pose("bag")[1], z - bot()]))

    h = s.hand("R")[:3]
    c.stage("carry_mid", ik_free(s, h + np.array([0.5 * (m[0] - c.pose("bag")[0]), 0.0, 0.0]), 1.8, stop=track, gain=IKG))
    front = (m[0], m[1] - 0.075)
    c.stage("carry_front", move_bag(front, rim + 0.05, 1.8))
    s.hold_for(0.6)
    c.stage("low_front", move_bag(front, rim + par["over_z"], 1.2))
    s.hold_for(0.5)
    c.stage("across", move_bag(m[:2], rim + par["over_z"], 1.6))
    s.hold_for(1.0)
    num["bag_off_axis_over_mm"] = round(float(np.linalg.norm(c.pose("bag")[:2] - m[:2])) * 1000, 1)
    c.stage("lowered", move_bag(m[:2], rim - par["inside"], 1.4))
    s.hold_for(0.6)
    num["bag_slip_mm"] = round(sm[0], 2)
    num["bag_pinch_spot_slip_mm"] = round(sp[0], 2)      # reported beside bag_slip_mm; used by neither status
    num["bag_turn_in_pinch_deg"] = round(sp[1], 2)
    if g is not None:
        g.release(BAG)
    s.fingers("R", dict(index=0.35), 1.0)
    s.hold_for(1.0)
    c.stage("released")
    h = s.hand("R")[:3]
    c.stage("hand_up", ik_free(s, h + np.array([0, 0, 0.12]), 1.5, gain=IKG))
    h = s.hand("R")[:3]
    c.stage("hand_back", ik_free(s, h + np.array([0.15, -0.08, 0.0]), 1.5, gain=IKG))
    s.hold_for(1.0)
    bg, tg, m2 = c.pose("bag"), c.pose("tag"), c.mug_pose()
    inside = bool(np.linalg.norm(bg[:2] - m2[:2]) < 0.0295 and bg[2] < rim and bg[2] > m2[2])
    num.update(bag_end=np.round(bg[:3], 4).tolist(), bag_off_axis_mm=round(float(np.linalg.norm(bg[:2] - m2[:2])) * 1000, 1), rim_z=round(float(rim), 4), bag_in_mug=inside,
               tag_end=np.round(tg[:3], 4).tolist(), tag_from_axis_mm=round(float(np.linalg.norm(tg[:2] - m2[:2])) * 1000, 1),
               tag_touches={k[1]: round(v[0], 3) for k, v in c.pairs().items() if k[0] == "teabag_tag"})
    moved = float(np.linalg.norm(m2[:3] - c.mug0[:3])) * 1000
    ok = inside and moved < HOLD_MM and sm[0] <= HOLD_MM and c.tilt() < 5
    out["status"] = "PASS" if ok else "FAIL"
    out["reason"] = "" if ok else f"bag_in_mug {inside}, mug moved {moved:.1f} mm, slip {sm[0]:.1f} mm, tilt {c.tilt():.1f} deg"
    task = inside and moved < HOLD_MM and c.tilt() < 5 and not s.fallen()      # the task only: strict status above is unchanged
    out["task_status"] = "PASS" if task else "FAIL"
    out["task_reason"] = "" if task else f"bag_in_mug {inside}, mug moved {moved:.1f} mm, tilt {c.tilt():.1f} deg, fallen {bool(s.fallen())}"


def run_stir(s, ctx):
    c = Ctx(s, ctx, "stir")
    out = c.out
    if not s.mimic:
        out["reason"] = "Sim built without mimic: results would not count"
        return out
    try:
        _stir(s, c)
    except Abort as e:
        out["reason"] = str(e)
    return c.finish()


def _stir(s, c):
    par, num, out, g = c.par, c.num, c.out, c.g
    IKG = par["ikg"]
    TOOL = np.array([-0.058, -par["tooly_spoon"], -0.0257])
    SP = c.paths["stir_spoon"]
    L = par["pinch_l"]
    arms = np.r_[s.arm_idx("L"), s.arm_idx("R")]         # powder leaves the left hand high over the counter: end on the standing arm pose instead, the one the cup step was tested from
    c.q_start[arms] = np.array([H.G["hold_pose"].get(x, 0.0) for x in s.names])[arms]
    c.walk_to_stance()
    sp0 = c.pose("spoon")
    R0 = mat_of(sp0[3:7])
    ax = R0[:, 0].copy()
    nz = R0[:, 2].copy()
    if nz[1] < 0:
        nz = -nz
    pt = lambda l, z=0.0: (lambda p_: p_[:3] + mat_of(p_[3:7]) @ np.array([l, 0, z]))(c.pose("spoon"))
    Rq = pre_rot(pinch_frame(nz, par["spoon_roll"], par["spoon_pitch"]), par["spoon_pre"])
    QS = quat_of(Rq)
    REST = np.array([-0.3, 0.1, 0.2, 0.9, -1.4, 0.2, -0.4])
    s.fingers("R", dict(OPEN_A, thumb_pitch=par["spoon_thumb_pitch"]), 1.2)
    c.left_aside()
    c.sup = CrouchSupport(s)
    num["crouch"] = c.sup.crouch(par["drop"], seconds=2.5, settle=0.8)
    if not num["crouch"]["passed"]:
        raise Abort("crouch failed its own gates")
    tgt = pt(L, props.HZ)
    pre = tgt + ax * 0.07
    c.stage("b_up", s.ik_to(to_tool(pre + np.array([0.0, -0.10, 0.03]), Rq, TOOL), QS, 2.0, side="R", gain=IKG, rest=REST))
    c.stage("b_over", s.ik_to(to_tool(pre, Rq, TOOL), QS, 1.8, side="R", gain=IKG, rest=REST))
    e = s.ik_to(to_tool(tgt, Rq, TOOL), QS, 1.6, side="R", gain=IKG, rest=REST)
    c.stage("b_at", e)
    s.hold_for(0.4)
    ang = float(np.degrees(np.linalg.norm(H.qerr(QS, s.hand("R")[3:7]))))
    num["hand_at_spoon"] = dict(err_mm=round(e * 1000, 1), ang_deg=round(ang, 2))
    if e > 0.006 or ang > 3.0:
        raise Abort(f"hand {e * 1000:.1f} mm, {ang:.1f} deg off at the spoon")
    if g is not None:
        g.allow_move("stir_spoon")
    cur, cc = c.creep("stir_spoon", ("index", "thumb"))
    num["creep_index"] = round(cur, 3)
    if not any("index" in k for k in cc) or not any("thumb" in k for k in cc):
        raise Abort(f"pinch not closed on the spoon: {cc}")
    s.fingers("R", dict(index=cur + par["sq_spoon"]), 0.4)
    s.hold_for(0.4)
    num["pinch_N"] = c.on("stir_spoon")
    c.stage("b_pinched")
    if g is not None:
        g.hold(SP, "R_hand_base_link")
    rel = lambda: H.qrot(H.qinv(s.hand("R")[3:7]), c.pose("spoon")[:3] - s.hand("R")[:3])
    grip = lambda: H.qrot(H.qinv(s.hand("R")[3:7]), pt(L, props.HZ) - s.hand("R")[:3])
    rel0, grip0 = rel(), grip()
    gm = [0.0]

    def gtrack():
        gm[0] = max(gm[0], float(np.linalg.norm(grip() - grip0)) * 1000)
        return False

    h = s.hand("R")[:3]
    c.stage("b_off_rim", s.ik_to(h + ax * 0.02 + nz * par["offrim"], QS, 1.0, side="R", gain=IKG, stop=gtrack))
    h = s.hand("R")[:3]
    c.stage("b_out", s.ik_to(h + ax * par["out_l"], QS, par["out_s"], side="R", gain=IKG, stop=gtrack))
    s.hold_for(0.4)
    num["spoon_rise_mm"] = round((c.pose("spoon")[2] - sp0[2]) * 1000, 1)
    if c.pose("spoon")[2] < sp0[2] + 0.05:
        raise Abort("spoon not lifted")
    Rc = mat_of(s.hand("R")[3:7])
    tool_now = s.hand("R")[:3] + Rc @ TOOL
    Rv = pinch_frame(np.array([0, 1.0, 0]), 0, 0)
    c.stage("b_upright", s.ik_to(to_tool(tool_now, Rv, TOOL), quat_of(Rv), 2.0, side="R", gain=IKG, stop=gtrack))
    s.hold_for(0.5)
    m = c.mug_pose()
    rim = m[2] + RIM
    bowl = lambda: pt(0.0175)
    tip = lambda: pt(0.0)
    lean = float(np.degrees(np.arccos(np.clip(mat_of(c.pose("spoon")[3:7])[2, 0], -1, 1))))
    num["spoon_lean_deg"] = round(lean, 2)

    def move_bowl(goal, seconds, n=7):
        goal = np.asarray(goal, float)
        for i in range(n):
            if np.linalg.norm(goal - bowl()) < 0.002:
                break
            ik_free(s, s.hand("R")[:3] + (goal - bowl()), seconds if i == 0 else 0.8, stop=gtrack, gain=IKG)
            s.hold_for(0.3)
        return float(np.linalg.norm(goal - bowl()))

    c.stage("b_raise", move_bowl(np.r_[bowl()[:2], max(bowl()[2], rim + 0.055)], 1.2, n=1))
    c.stage("b_over_mug", move_bowl(np.r_[m[:2], rim + 0.055], 2.4))
    s.hold_for(0.6)
    off = float(np.linalg.norm(bowl()[:2] - m[:2]))
    num["bowl_over_off_axis_mm"] = round(off * 1000, 1)
    if off > 0.006:
        raise Abort(f"bowl {off * 1000:.1f} mm off the mug axis before lowering")
    bag_xy = c.pose("bag")[:2]
    cen = m[:2].copy()
    d_ = m[:2] - bag_xy
    if np.linalg.norm(d_) < 0.03:                # the bag is in the mug: stir on its far side
        cen = m[:2] + 0.005 * d_ / max(np.linalg.norm(d_), 1e-6)
    z_stir = par["stir_z"]
    W = c.ctx.get("water")
    if W is not None:                            # measured surface: the bowl must be in the water, not above it
        W.sample(s.step)
        wr = W.report()
        num["water_before_stir"] = {k: wr.get(k) for k in ("mug_ml", "level_mm", "spill_ml")}
        if wr.get("level_mm"):
            z_stir = float(np.clip(wr["level_mm"] / 1000 - par["bowl_sub"], par["stir_z_min"], par["stir_z"]))
    num["stir_z_mm"] = round(z_stir * 1000, 1)
    c.stage("b_lowered", move_bowl(np.r_[cen, m[2] + z_stir], 2.2))
    s.hold_for(0.6)
    path, sm = [], dict(f=0.0, n=0, bag=0.0, zmax=-1.0)

    def watch():
        path.append(bowl()[:2].copy())
        sm["zmax"] = max(sm["zmax"], float(bowl()[2]))
        pr = c.pairs()
        f = sum(v[0] for k, v in pr.items() if k[0] == "stir_spoon" and k[1] == c.mug)
        sm["f"] = max(sm["f"], f)
        sm["bag"] = max(sm["bag"], sum(v[0] for k, v in pr.items() if k[0] == "stir_spoon" and k[1] == "teabag_bag"))
        sm["n"] += int(f > 0)
        gtrack()

    c0 = s.hand("R")[:3].copy()
    RAD, TURNS, PER = par["stir_r"], par["turns"], par["stir_period"]

    def circ(u):
        r = RAD * max(0.0, min(1.0, u / 0.2, (1 - u) / 0.12))
        th = 2 * math.pi * TURNS * u
        return c0 + r * np.array([math.cos(th), math.sin(th), 0.0])

    w = ik_path(s, circ, TURNS * PER, stop=watch, gain=IKG)
    s.hold_for(0.5)
    Pp = np.array(path)
    mid = Pp[len(Pp) // 5:-len(Pp) // 8]
    cc_ = mid.mean(0)
    an = np.unwrap(np.arctan2(*(mid - cc_).T[::-1]))
    num["stir"] = dict(cmd_radius_mm=RAD * 1000, cmd_turns=TURNS, hand_worst_track_mm=round(w * 1000, 1), bowl_circle_diam_mm=round(float(2 * np.linalg.norm(mid - cc_, axis=1).mean()) * 1000, 1),
                       bowl_turns_full_radius=round(float(abs(an[-1] - an[0]) / (2 * math.pi)), 2), bowl_centre_off_axis_mm=round(float(np.linalg.norm(cc_ - m[:2])) * 1000, 1), spoon_mug_contact_steps=sm["n"],
                       spoon_mug_force_N=round(sm["f"], 2), spoon_bag_force_N=round(sm["bag"], 2), mug_tilt_deg=round(c.tilt(), 2), bowl_highest_above_rim_mm=round((sm["zmax"] - rim) * 1000, 1))
    c.stage("b_stirred")
    c.stage("b_out_of_mug", move_bowl(np.r_[bowl()[:2], rim + 0.055], 2.0, n=1))
    GX, GY = m[0] + par["put_dx"], par["put_y"]
    c.stage("b_to_spot", ik_free(s, s.hand("R")[:3] + (np.array([GX, GY, 0.905]) - tip()), 2.4, gain=IKG, stop=gtrack))
    s.hold_for(0.4)
    c.stage("b_over_spot", ik_free(s, s.hand("R")[:3] + (np.array([GX, GY, 0.872]) - tip()), 1.4, gain=IKG, stop=gtrack))
    s.hold_for(0.4)
    hh = s.hand("R")
    Rh = mat_of(hh[3:7])
    v_h = Rh.T @ (tip() - hh[:3])
    goal_tip = tip() + np.array([0, 0, -0.004])
    th = math.radians(par["tip_deg"])
    Ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    Rt = Ry @ Rh
    c.stage("b_tipped", s.ik_to(to_tool(goal_tip, Rt, v_h), quat_of(Rt), 2.4, side="R", gain=IKG, stop=gtrack))
    s.hold_for(0.4)
    num["spoon_grip_point_slip_mm"] = round(gm[0], 2)
    num["spoon_origin_slip_mm"] = round(float(np.linalg.norm(rel() - rel0)) * 1000, 2)
    if g is not None:
        g.release(SP)
    s.fingers("R", dict(index=0.35), 1.2)
    s.hold_for(1.2)
    c.stage("b_released")
    h = s.hand("R")[:3]
    c.stage("b_hand_up", ik_free(s, h + np.array([0.0, -0.04, 0.12]), 1.6, gain=IKG, ref=mat_of(s.hand("R")[3:7])[:, 0]))
    s.hold_for(0.5)
    sp, m2 = c.pose("spoon"), c.mug_pose()
    ax2 = mat_of(sp[3:7])[:, 0]
    on_counter = bool(sp[2] < 0.875 and abs(ax2[2]) < 0.1)
    moved = float(np.linalg.norm(m2[:3] - c.mug0[:3])) * 1000
    num.update(spoon_end=np.round(sp[:3], 4).tolist(), spoon_axis_end=np.round(ax2, 3).tolist(), spoon_on_counter=on_counter, bag_end=np.round(c.pose("bag")[:3], 4).tolist())
    ok = on_counter and moved < HOLD_MM and c.tilt() < 5 and num["stir"]["bowl_turns_full_radius"] >= 2 and num["spoon_origin_slip_mm"] <= HOLD_MM
    out["status"] = "PASS" if ok else "FAIL"
    out["reason"] = "" if ok else f"spoon_on_counter {on_counter}, mug moved {moved:.1f} mm, tilt {c.tilt():.1f} deg, turns {num['stir']['bowl_turns_full_radius']}, origin slip {num['spoon_origin_slip_mm']} mm, grip point slip {num['spoon_grip_point_slip_mm']} mm"
    st_ = num["stir"]
    lifted = num["spoon_rise_mm"] >= 50.0
    stirred = st_["bowl_turns_full_radius"] >= 2 and st_["bowl_highest_above_rim_mm"] < 0 and st_["bowl_centre_off_axis_mm"] < 29.5     # bowl below the rim the whole stir, circling inside the cavity
    task = lifted and stirred and on_counter and moved < HOLD_MM and c.tilt() < 5 and not s.fallen()      # the task only: strict status above is unchanged
    out["task_status"] = "PASS" if task else "FAIL"
    out["task_reason"] = "" if task else f"lifted {lifted}, stirred {stirred}, spoon_on_counter {on_counter}, mug moved {moved:.1f} mm, tilt {c.tilt():.1f} deg, fallen {bool(s.fallen())}"
