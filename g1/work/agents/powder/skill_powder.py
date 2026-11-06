"""Skill: whitener powder from the glass jar into the mug.

Physics only (MuJoCo 600 Hz, torque-limited G1 through the byor whole-body controller; jar, lid, spoon and
powder grains are free bodies held only by contact): the LEFT hand steadies the jar from its left side (palm
on the glass, fingers round its back - there is no room for a hand between the jar and the tray on its right:
4.0 cm gap, the hand is 4.1 cm thick), the RIGHT hand unscrews the lid anticlockwise (seen from above) with a
top grasp (palm down over the lid, index + middle curled onto its far edge, thumb on its near edge), in
~70 deg twists with regrasps, until the thread lets go; lifts the lid off and sets it down on the counter in
front of the jar; takes the spoon from the tray's FRESH tub (the spoon agent's proven pinch from above, lift
upright, three-finger grip), scoops powder from the jar (dip, push the bowl into the pile while tipping it
back ~45 deg), lifts it out, the left hand lets go of the jar, the spoon goes over the mug and is tipped over
until the grains fall in; the spoon is then left in the mug (bowl on the floor, handle on the rim) or kept in
the hand (kw end="hold") for the stirring step.

run(p, s, **kw) -> dict with 'ok' and measured results. Starts with the robot standing at the mug spot
(~(1.86, 1.90) facing +y), plans from the MEASURED stance and object poses.
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
sys.path.insert(0, str(WORK / "agents/powder"))
import props_powder as pp  # noqa: E402
import props_pspoon as ps  # noqa: E402
from motion import quat_from_axes, quat_mul, smooth  # noqa: E402
from tea_sim import JAR, MUG  # noqa: E402

L, R = 0, 1
# spoon agent's proven pickup (agents\spoon\skill_stir.py, run stir20): pinch from above, three-point grip
PINCH_L = 0.122
OPEN_A = {"thumb_0": 0.75, "thumb_1": 0.45, "thumb_2": 0.45, "index": 0.0, "middle": 0.0}
SHUT_A = {"thumb_0": 0.75, "thumb_1": 0.68, "thumb_2": 0.68, "index": 0.68, "middle": 0.0}
MIDDLE3 = {"middle": 0.62}
THUMB3 = {"thumb_0": 0.50, "thumb_1": 0.54, "thumb_2": 0.67}
INDEX3 = {"index": 0.62}
TOOL_A = (0.1257, 0.0711, 0.0264)
# lid top grasp (thumb_search.py, kinematics, mirrored for the right hand): lid axis at x_h 0.13, lid top
# 3.5 cm under the hand origin on the palm side; index/middle at closure 0.3 touch its far edge, thumb_1 flexed
# fully brings the thumb within 3 mm of its near edge
LID_TOOL = (0.13, 0.035 + 0.011, 0.0)          # the lid's centre at mid-height, right hand frame
LID_OPEN = {"thumb_0": 0.0, "thumb_1": 1.0, "thumb_2": 0.6, "index": 0.0, "middle": 0.0}   # thumb already flexed: an open thumb hangs 5-9 cm under the palm and lands on the lid top (lid9)
LID_FING = {"index": 0.7, "middle": 0.7}
LID_THUMB = {"thumb_1": 1.0, "thumb_2": 0.6}
STEADY_TOOL = (0.10, -0.0173 - 0.045 - 0.002, 0.0)   # the jar's axis when the left palm is on its side


class Ctx:
    pass


def _finger_names(s, b):
    m = s.m
    return {a: m.joint(int(np.nonzero(m.jnt_qposadr == a)[0][0])).name for a, *_ in s.fingers[b]}


def _targets(s, b, pose):
    out = {}
    names = _finger_names(s, b)
    for a, lo, hi, high in s.fingers[b]:
        c = next((v for k, v in pose.items() if k in names[a]), None)
        if c is not None:
            out[a] = lo + (c if high else 1 - c) * (hi - lo)
    return out


def close_until(p, hand, pose, done, seconds=1.2):
    s, w = p.s, p.w
    b = s.hands[hand]
    goal = _targets(s, b, pose)
    start = {a: float(w.qpos0[a]) for a in goal}
    s.grip_active.add(b)
    n = max(1, int(round(seconds / p.mv.TICK)))
    for k in range(1, n + 1):
        u = smooth(k / n)
        for a in goal:
            w.qpos0[a] = start[a] + u * (goal[a] - start[a])
        p.mv._tick(p.mv.TICK)
        if done is not None and done():
            return k / n
    return 1.0


def ramp_fingers(p, hand, pose, seconds):
    close_until(p, hand, pose, None, seconds)


def axis_quat(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis]


def run(p, s, **kw) -> dict:
    m, d = s.m, s.d
    out = {"ok": False}
    br, bl = s.hands[R], s.hands[L]
    HR, HL = s.hand_parts[br], s.hand_parts[bl]
    robot = {i for i in range(1, m.nbody) if not m.body(i).name.startswith("env/")}
    jb = m.body("env/" + JAR + "_body").id
    lb = m.body("env/" + pp.LID).id
    mb = m.body("env/" + MUG + "_body").id
    sb = m.body("env/" + ps.BODY).id
    gids = np.array(pp.grain_ids(m), dtype=int)
    gset = set(gids.tolist())
    spoon_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == sb}
    props = {jb, lb, sb, mb} | gset
    t_start = d.time
    th = pp.install(s)                                   # the thread model's step hook
    yaw = p.mv.heading()
    FWD = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    LEFT = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    RIGHT, UP = -LEFT, np.array([0.0, 0.0, 1.0])
    out["stance"] = dict(xy=np.round(d.qpos[:2], 3).tolist(), heading_deg=round(math.degrees(yaw), 1))
    jar0, mug0 = d.xpos[jb].copy(), d.xpos[mb].copy()
    gz0 = d.xmat[jb].reshape(3, 3)[:, 2].copy()
    marks = {}

    def mark(name):
        marks[name] = round(float(d.time - t_start), 2)

    # ------------------------------------------------------------------ measurements
    geom_body = m.geom_bodyid
    stats = {"prop_contact_mm": 0.0, "prop_contact_what": None}

    def tick_hook(_s):
        n = d.ncon
        if n:
            dist = d.contact.dist[:n]
            g = d.contact.geom[:n]
            b1, b2 = geom_body[g[:, 0]], geom_body[g[:, 1]]
            p1 = np.isin(b1, list(props))
            p2 = np.isin(b2, list(props))
            r1 = np.isin(b1, list(robot))
            r2 = np.isin(b2, list(robot))
            sel = (p1 | p2) & ~r1 & ~r2
            if sel.any():
                k = int(np.argmin(np.where(sel, dist, 1.0)))
                if -dist[k] * 1000 > stats["prop_contact_mm"]:
                    stats["prop_contact_mm"] = float(-dist[k] * 1000)
                    stats["prop_contact_what"] = (round(float(d.time), 2), m.geom(int(g[k, 0])).name[-26:], m.geom(int(g[k, 1])).name[-26:])
        if prev_tick is not None:
            prev_tick(_s)

    prev_tick = p.mv.on_tick
    p.mv.on_tick = tick_hook

    def forces(set_a, set_b):
        f, deep, buf = 0.0, 0.0, np.zeros(6)
        for i in range(d.ncon):
            cc = d.contact[i]
            b1, b2 = int(m.geom_bodyid[cc.geom1]), int(m.geom_bodyid[cc.geom2])
            if (b1 in set_a and b2 in set_b) or (b2 in set_a and b1 in set_b):
                mujoco.mj_contactForce(m, d, i, buf)
                f += abs(buf[0])
                deep = max(deep, -cc.dist * 1000)
        return f, deep

    def fingers_on(hand_parts, body_set, names=("thumb", "index", "middle")):
        hit = set()
        for cc in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[cc.geom1]), int(m.geom_bodyid[cc.geom2])
            for a, o in ((b1, b2), (b2, b1)):
                if a in hand_parts and o in body_set:
                    hit.add(m.body(a).name)
        return {n for n in names if any(n in h for h in hit)}

    def support(body):
        """Upward force (N) on `body` from anything that is not the robot."""
        f, buf = 0.0, np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 == body) == (b2 == body) or b1 in robot or b2 in robot:
                continue
            mujoco.mj_contactForce(m, d, i, buf)
            f += abs(buf[0] * c.frame[2])
        return f

    def jar_tilt():
        return math.degrees(math.acos(np.clip(d.xmat[jb].reshape(3, 3)[:, 2] @ gz0, -1, 1)))

    def lid_pt(local):
        return d.xpos[lb] + d.xmat[lb].reshape(3, 3) @ np.asarray(local, float)

    def sp(local):
        return d.xpos[sb] + d.xmat[sb].reshape(3, 3) @ np.asarray(local, float)

    def grains_where():
        G = d.xpos[gids].reshape(-1, 3)
        Rj, Rm = d.xmat[jb].reshape(3, 3), d.xmat[mb].reshape(3, 3)
        J = (G - d.xpos[jb]) @ Rj
        M = (G - d.xpos[mb]) @ Rm
        in_jar = (np.hypot(J[:, 0], J[:, 1]) < pp.R_WALL_IN + 0.001) & (J[:, 2] > 0) & (J[:, 2] < pp.NECK[3] + 0.003)
        in_mug = (np.hypot(M[:, 0], M[:, 1]) < 0.030) & (M[:, 2] > 0.004) & (M[:, 2] < 0.090)
        in_spoon = ps.in_bowl(d, m, G)
        return in_jar, in_mug, in_spoon

    def done_fail(why):
        out["failed"] = why
        p.note("FAILED", why=why)
        finish()
        return out

    def finish():
        in_jar, in_mug, in_spoon = grains_where()
        out["grains"] = dict(total=int(len(gids)), in_jar=int(in_jar.sum()), in_mug=int(in_mug.sum()),
                             in_spoon=int(in_spoon.sum()),
                             elsewhere=int((~in_jar & ~in_mug & ~in_spoon).sum()))
        out["jar"] = dict(moved_mm=round(float(np.linalg.norm((d.xpos[jb] - jar0)[:2])) * 1000, 1),
                          tilt_deg=round(jar_tilt(), 2))
        out["mug_moved_mm"] = round(float(np.linalg.norm((d.xpos[mb] - mug0)[:2])) * 1000, 1)
        out["lid"] = dict(turned_deg=round(math.degrees(th.angle()), 1), thread_released_at=th.released_at,
                          pos=np.round(d.xpos[lb], 4).tolist(),
                          tilt_deg=round(math.degrees(math.acos(np.clip(d.xmat[lb].reshape(3, 3)[2, 2], -1, 1))), 1),
                          support_N=round(support(lb), 2))
        out["prop_contact_deepest_mm"] = round(stats["prop_contact_mm"], 2)
        out["prop_contact_deepest_what"] = stats["prop_contact_what"]
        out["marks_s"] = marks
        out["seconds"] = round(float(d.time - t_start), 2)
        p.mv.on_tick = prev_tick

    rot_w0 = s.wbc.weights["ee_rot"]

    # ================================================================== 1. left hand steadies the jar
    if not kw.get("lid_off", False):
        if kw.get("steady", 1):
          QL = quat_from_axes(FWD, LEFT, UP)                   # fingers forward, left palm (-y_h) facing right
          p.set_tool(L, STEADY_TOOL)
          cj = d.xpos[jb].copy()
          cL = np.r_[cj[:2], cj[2] + 0.07]
          c_out = cL - 0.018 * RIGHT
          aboveL = c_out + 0.10 * UP
          frontL = aboveL - 0.14 * FWD
          shl = d.xpos[s.shoulder[bl]].copy()
          # two stages: a whole-body reach to the spoon agent's proven left "ready" pose in front of the mug (a
          # planned reach straight to the jar picks an arm pose the controller cannot follow from hanging: the
          # shoulder stays down, runs lid1-5), then hand-only moves (the arm follows its hand continuously)
          mc0 = d.xpos[mb].copy()
          p.grip(L, 0.4, 0.25)          # as the spoon agent before its left reach (the hanging hand's default thumb pins it, lid1-7)
          ready = np.r_[(mc0 - 0.003 * RIGHT - 0.14 * FWD)[:2], mc0[2] + 0.15]
          p.reach(L, ready, quat=QL, seconds=kw.get("left_s", 1.6), settle=0.2,
                  via=np.r_[(shl + 0.08 * FWD)[:2], ready[2] - 0.02])
          p.move_hand(L, frontL, quat=QL, seconds=0.8)
          p.grip(L, 0.15, 0.25)                            # fingers open only once the hand is up in the air (hanging, an
          p.move_hand(L, aboveL, quat=QL, seconds=0.6)     # opening thumb can catch the thigh and pin the hand)
          p.settle_hand(L, aboveL, tol=0.006, timeout=0.3)
          p.move_hand(L, c_out, quat=QL, seconds=0.7)
          p.move_until(L, RIGHT, lambda: forces(HL, {jb})[0] > 0.8, speed=0.015, max_dist=0.03, quat=QL, ease_out=0.1)
          p.move_hand(L, s.wbc.ee_pos(d, bl), quat=QL, seconds=0.2)
          p.grip(L, kw.get("steady_grip", 0.55), 0.5)
          p.hold(0.2)
          out["steady"] = dict(force_N=round(forces(HL, {jb})[0], 2), fingers=sorted(fingers_on(HL, {jb})),
                               jar_moved_mm=round(float(np.linalg.norm((d.xpos[jb] - jar0)[:2])) * 1000, 1))
          p.note("left hand steadies the jar", **out["steady"])
          mark("jar_steadied")

        # ============================================================== 2. right hand unscrews the lid
        def QR(psi):
            f = np.array([math.cos(yaw + psi), math.sin(yaw + psi), 0.0])
            return quat_from_axes(f, -UP, np.cross(f, -UP))

        psi0, psi1 = math.radians(kw.get("psi0", -30.0)), math.radians(kw.get("psi1", 40.0))
        tgt = lid_pt((0, 0, pp.LID_Z0 + 0.011))
        pre = tgt + np.array([0.0, 0.0, 0.05])
        sh = d.xpos[s.shoulder[br]].copy()
        # two stages (as the left): the spoon agent's proven reach to over the FRESH tub, then hand-only to the lid
        # EXACTLY the spoon agent's first reach (its tool point, orientation, via, timing) - variants of it with
        # another tool point or target fail from the hanging arm (runs lid2-8) - then the lid tool and hand-only
        nrm0 = d.xmat[sb].reshape(3, 3)[:, 2].copy()
        nrm0[2] = 0.0
        nrm0 /= np.linalg.norm(nrm0)
        if nrm0 @ FWD < 0:
            nrm0 = -nrm0
        p.set_tool(R, np.array(TOOL_A))
        p.finger_pose(R, OPEN_A, 0.4)
        ready = sp((PINCH_L, 0.0, ps.HZ)) + np.array([0.0, 0.0, 0.06])
        p.reach(R, ready, quat=quat_from_axes(nrm0, -UP, np.cross(nrm0, -UP)), seconds=kw.get("reach_s", 1.8),
                via=np.r_[(sh + 0.16 * FWD + 0.05 * RIGHT)[:2], ready[2] + 0.04])
        p.set_tool(R, LID_TOOL)
        p.move_hand(R, pre + [0, 0, 0.02], quat=QR(psi0), seconds=1.2)
        p.move_hand(R, pre, quat=QR(psi0), seconds=0.4)
        p.finger_pose(R, LID_OPEN, 0.4)                  # opened in the air, over the lid
        p.settle_hand(R, pre, tol=0.005, timeout=0.5)
        lid_set = {lb}

        def touching_lid():
            return forces(HR, lid_set | {jb})[0] > 0.3

        twists = []
        s.wbc.weights["ee_rot"] = kw.get("rot_w_lid", 30.0)
        for twist in range(kw.get("max_twists", 7)):
            tgt = lid_pt((0, 0, pp.LID_Z0 + 0.011))
            q0 = QR(psi0)
            if twist == 0:
                p.move_hand(R, tgt + [0, 0, 0.012], quat=q0, seconds=0.6)
                p.lower_until(R, touching_lid, speed=0.02, max_dist=0.02, quat=q0)
                for _ in range(3):                           # servo the measured hand point onto the lid centre
                    e = tgt - s.wbc.ee_pos(d, br)
                    e[2] = max(e[2], -0.002)
                    if np.linalg.norm(e) < 0.002:
                        break
                    p.move_hand(R, s.wbc.ee_goal[br] + e, quat=q0, seconds=0.25)
            fr = close_until(p, R, LID_FING, lambda: {"index", "middle"} <= fingers_on(HR, lid_set)
                             and forces(HR, lid_set)[0] > 4.0, seconds=0.6)
            close_until(p, R, LID_THUMB, lambda: "thumb" in fingers_on(HR, lid_set), seconds=0.4)
            # take the weight of the hand off the lid (pressing down loads the lid on the rim: friction that
            # no grip can beat, lid10: 55 N on the lid, 1.8 deg per twist) - the fingers keep their hold
            p.move_hand(R, s.wbc.ee_pos(d, br) + [0, 0, kw.get("unload_mm", 3.0) / 1000.0], seconds=0.3)
            p.hold(0.1)
            f_grip, deep = forces(HR, lid_set)
            a0 = th.angle()
            here = s.wbc.ee_goal[br].copy()
            # twist anticlockwise about the lid's axis: the hand point stays on it, the lid rises with the thread
            n = int(round(kw.get("twist_s", 1.1) / p.mv.TICK))
            qa = s.wbc.ee_rot[br].copy()
            for k in range(1, n + 1):
                u = smooth(k / n)
                s.wbc.set_ee_target(br, here + [0, 0, u * pp.K_THREAD * (psi1 - psi0)],
                                    quat=quat_mul(axis_quat(UP, u * (psi1 - psi0)), qa))
                p.mv._tick(p.mv.TICK)
                if th.released_at is not None:
                    break
            p.hold(0.1)
            gained = math.degrees(th.angle() - a0)
            twists.append(dict(grip_N=round(f_grip, 1), deepest_mm=round(deep, 2), fingers=sorted(fingers_on(HR, lid_set)),
                               turned_deg=round(gained, 1), total_deg=round(math.degrees(th.angle()), 1),
                               jar_turn_deg=round(math.degrees(math.atan2(d.xmat[jb].reshape(3, 3)[1, 0], d.xmat[jb].reshape(3, 3)[0, 0])), 1)))
            p.note("twist", n=twist + 1, **twists[-1])
            if th.released_at is not None:
                break
            # regrasp: open, up off the lid a little, turn back, down again
            ramp_fingers(p, R, LID_OPEN, 0.3)
            p.move_hand(R, s.wbc.ee_goal[br] + [0, 0, 0.006], seconds=0.25)
            tgt = lid_pt((0, 0, pp.LID_Z0 + 0.011))
            p.move_hand(R, tgt + [0, 0, 0.006], quat=QR(psi0), seconds=kw.get("back_s", 0.8))
            p.move_hand(R, tgt + [0, 0, 0.001], quat=QR(psi0), seconds=0.2)
        out["unscrew"] = dict(twists=twists, released=th.released_at is not None,
                              turned_deg=round(math.degrees(th.angle()), 1))
        mark("lid_unscrewed")
        if th.released_at is None:
            s.wbc.weights["ee_rot"] = rot_w0
            return done_fail("thread not released after the twists")
        # ============================================================== 3. lid off, onto the counter
        lid0 = d.xpos[lb].copy()
        p.move_hand(R, s.wbc.ee_goal[br] + [0, 0, 0.05], seconds=0.8)
        p.hold(0.15)
        lifted = float(d.xpos[lb][2] - lid0[2])
        p.note("lid lifted", rise_mm=round(lifted * 1000, 1), fingers=sorted(fingers_on(HR, lid_set)))
        if lifted < 0.02:
            s.wbc.weights["ee_rot"] = rot_w0
            return done_fail("lid did not come up with the hand")
        spot = np.r_[(jar0 + kw.get("lid_spot_back", 0.19) * -FWD + kw.get("lid_spot_right", 0.0) * RIGHT)[:2], 0.0]
        off = s.wbc.ee_pos(d, br) - lid_pt((0, 0, pp.LID_Z0 + 0.011))          # hand point vs lid centre now
        goal = np.r_[spot[:2], 0.86 + 0.011 + 0.015] + off
        here = s.wbc.ee_goal[br].copy()
        p.move_hand(R, goal, seconds=1.2, via=np.r_[((here + goal) / 2)[:2], max(here[2], goal[2]) + 0.03])
        p.hold(0.15)
        e = spot[:2] - lid_pt((0, 0, 0))[:2]
        p.move_hand(R, s.wbc.ee_goal[br] + np.r_[e, 0.0], seconds=0.4)
        p.lower_until(R, lambda: support(lb) > 0.15, speed=0.02, max_dist=0.04,
                      steer=lambda: (spot[:2] - lid_pt((0, 0, 0))[:2]))
        p.hold(0.15)
        p.move_hand(R, s.wbc.ee_pos(d, br), seconds=0.2)                         # unload
        ramp_fingers(p, R, LID_OPEN, 0.35)
        p.hold(0.1)
        lid_rest = d.xpos[lb].copy()
        p.move_hand(R, s.wbc.ee_goal[br] + [0, 0, 0.06], seconds=0.6)
        s.wbc.weights["ee_rot"] = rot_w0
        p.hold(0.2)
        out["lid_down"] = dict(off_spot_mm=round(float(np.linalg.norm(lid_pt((0, 0, 0))[:2] - spot[:2])) * 1000, 1),
                               moved_after_release_mm=round(float(np.linalg.norm(d.xpos[lb] - lid_rest)) * 1000, 1),
                               support_N=round(support(lb), 2),
                               tilt_deg=round(math.degrees(math.acos(np.clip(d.xmat[lb].reshape(3, 3)[2, 2], -1, 1))), 1))
        p.note("lid on the counter", **out["lid_down"])
        mark("lid_down")
    if kw.get("stop_after_lid", False):
        finish()
        out["ok"] = bool(out.get("unscrew", {}).get("released") and s.checks.fell_at is None)
        return out

    # ================================================================== 4. spoon from the FRESH tub (spoon agent)
    def axis_pt(L_):
        return sp((L_, 0.0, ps.HZ))

    def tilt():
        a = axis_pt(ps.LENGTH) - axis_pt(0.0)
        return math.degrees(math.acos(np.clip(a[2] / np.linalg.norm(a), -1, 1)))

    sbody = {sb}
    nrm = d.xmat[sb].reshape(3, 3)[:, 2].copy()
    nrm[2] = 0.0
    nrm /= np.linalg.norm(nrm)
    if nrm @ FWD < 0:
        nrm = -nrm
    QA = quat_from_axes(nrm, -UP, np.cross(nrm, -UP))
    p.set_tool(R, np.array(TOOL_A))
    p.finger_pose(R, OPEN_A, 0.4)
    target = axis_pt(PINCH_L)
    pre = target + np.array([0.0, 0.0, 0.06])
    sh = d.xpos[s.shoulder[br]].copy()
    p.reach(R, pre, quat=QA, seconds=kw.get("spoon_reach_s", 1.6),
            via=np.r_[(sh + 0.16 * FWD + 0.05 * RIGHT)[:2], pre[2] + 0.04])
    p.settle_hand(R, pre, tol=0.005, timeout=0.6)
    p.move_hand(R, target, quat=QA, seconds=0.8)
    p.settle_hand(R, target, tol=0.003, timeout=0.4)
    for _ in range(4):
        e = target - s.wbc.ee_pos(d, br)
        if np.linalg.norm(e) < 0.002:
            break
        p.move_hand(R, s.wbc.ee_goal[br] + e, quat=QA, seconds=0.3)
        p.hold(0.15)
    frac = close_until(p, R, SHUT_A, lambda: {"thumb", "index"} <= fingers_on(HR, sbody, ("thumb", "index"))
                       and forces(HR, sbody)[0] > 2.0, seconds=0.9)
    p.hold(0.25)
    f, deep = forces(HR, sbody)
    out["pinch"] = dict(closure=round(frac, 2), force_N=round(f, 2), deepest_mm=round(deep, 2),
                        fingers=sorted(fingers_on(HR, sbody)))
    p.note("pinched", **out["pinch"])
    C = axis_pt(0.0)
    P0 = s.wbc.ee_pos(d, br)
    lean = (P0 - C) * np.array([1.0, 1.0, 0.0])
    hl = float(np.linalg.norm(lean))
    lean = lean / hl if hl > 1e-6 else -FWD
    ang0 = math.atan2(hl, (P0 - C)[2])
    r_ = float(np.linalg.norm(P0 - C))
    upright = C + r_ * (math.sin(math.radians(4.0)) * lean + math.cos(math.radians(4.0)) * UP)
    mid_ang = 0.5 * (ang0 + math.radians(4.0))
    via = C + r_ * (math.sin(mid_ang) * lean + math.cos(mid_ang) * UP)
    p.move_hand(R, upright, quat=QA, seconds=0.8, via=via)
    p.hold(0.2)
    p.move_hand(R, s.wbc.ee_goal[br] + [0.0, 0.0, 0.13], quat=QA, seconds=1.2)
    p.hold(0.3)
    if not {"thumb", "index"} <= fingers_on(HR, sbody, ("thumb", "index")):
        return done_fail("spoon not in the pinch after the lift")
    QH = quat_from_axes(FWD, LEFT, UP)
    p.move_hand(R, s.wbc.ee_goal[br] + 0.03 * LEFT - 0.04 * FWD, quat=QH, seconds=1.0)
    p.hold(0.2)
    ramp_fingers(p, R, MIDDLE3, 0.5)
    ramp_fingers(p, R, THUMB3, 0.7)
    ramp_fingers(p, R, INDEX3, 0.3)
    if "grip_index" in kw or "grip_middle" in kw:              # then a firmer squeeze (closing the middle to 0.66
        ramp_fingers(p, R, {"index": kw.get("grip_index", INDEX3["index"]),   # first knocks the spoon out, scoop12)
                            "middle": kw.get("grip_middle", MIDDLE3["middle"])}, 0.4)
    p.hold(0.2)
    f, deep = forces(HR, sbody)
    out["grip3"] = dict(force_N=round(f, 2), deepest_mm=round(deep, 2), tilt_deg=round(tilt(), 1),
                        fingers=sorted(fingers_on(HR, sbody)))
    p.note("three-point grip", **out["grip3"])
    if "middle" not in out["grip3"]["fingers"]:
        return done_fail("no three-point grip")
    mark("spoon_in_hand")

    # ================================================================== 5. scoop
    Rh = d.xmat[br].reshape(3, 3)
    bowl_local = np.array(ps.BOWL_CENTRE)
    p.set_tool(R, Rh.T @ (sp(bowl_local) - d.xpos[br]))          # the bowl centre is the controller's tool point
    pinch_off = Rh.T @ (axis_pt(PINCH_L) - d.xpos[br])

    def rel_hand():
        Rh_ = d.xmat[br].reshape(3, 3)
        return Rh_.T @ (sp(bowl_local) - d.xpos[br])

    rel0 = rel_hand()
    slip = {"max": 0.0}

    def track_slip():
        slip["max"] = max(slip["max"], float(np.linalg.norm(rel_hand() - rel0)) * 1000)

    n_c = d.xmat[sb].reshape(3, 3)[:, 2].copy()                  # the bowl's open side, measured
    n_h = n_c * np.array([1.0, 1.0, 0.0])
    n_h /= np.linalg.norm(n_h)
    tilt_axis = np.cross(n_h, UP)                                 # +angle about it turns the open side up
    s.wbc.weights["ee_rot"] = kw.get("rot_w", 100.0)
    cj = d.xpos[jb].copy()
    rim_z = cj[2] + pp.NECK[3]
    bx = np.r_[(cj - kw.get("dip_back", 0.010) * n_h)[:2], rim_z + 0.035]
    here = s.wbc.ee_goal[br].copy()
    p.move_hand(R, bx, quat=QH, seconds=kw.get("to_jar_s", 1.4), via=np.r_[((here + bx) / 2)[:2], max(here[2], bx[2]) + 0.04])
    p.hold(0.2)
    for _ in range(2):
        e = bx - sp(bowl_local)
        if np.linalg.norm(e) < 0.004:
            break
        p.move_hand(R, s.wbc.ee_goal[br] + e, quat=QH, seconds=0.4)
        p.hold(0.1)
    track_slip()
    _, _, in_spoon = grains_where()
    G = d.xpos[gids]
    if len(gids):                                                 # the powder surface, as a camera sees it
        pile_top = float(np.max(G[np.hypot(G[:, 0] - cj[0], G[:, 1] - cj[1]) < 0.03][:, 2]))
    else:                                                         # grains simulated in powder_pass.py
        pile_top = float(cj[2] + np.load(pp.GRAINS_FILE)[:, 2].max())
    fill_z = cj[2] + pp.FILL_TOP
    grain_geoms = set(np.nonzero(np.isin(m.geom_bodyid, gids))[0].tolist())

    def bowl_in_pile():
        # the bowl tip 4 mm over the crater floor, or the spoon meets the jar / the fill / the grains firmly
        if axis_pt(0.0)[2] < fill_z + kw.get("tip_clear", 0.004):
            return True
        return forces(sbody, {jb})[0] > 0.02 or forces(sbody, gset)[0] > 0.3

    p.lower_until(R, bowl_in_pile, speed=0.025, max_dist=0.09, quat=QH,
                  steer=lambda: (bx - sp(bowl_local))[:2])
    p.move_hand(R, s.wbc.ee_goal[br] + [0, 0, -kw.get("dig", 0.0)], quat=QH, seconds=0.4)
    p.hold(0.1)
    out["dip"] = dict(bowl_z_above_pile_mm=round((sp(bowl_local)[2] - pile_top) * 1000, 1),
                      spoon_on_powder_N=round(forces(sbody, gset)[0], 2), spoon_on_jar_N=round(forces(sbody, {jb})[0], 2))
    p.note("in the powder", **out["dip"])
    track_slip()
    # the spoon tips back about the point of its shaft in the jar's mouth (a lever on the rim, as a person
    # scoops): the bowl swings forward and up through the pile, the shaft stays in the 66 mm opening.
    # (Pivoting about the bowl, scoop3/4, swung the shaft into the neck: 91 kN contact, jar thrown 1.1 m.)
    L_piv = float(rim_z - axis_pt(0.0)[2])
    Rh = d.xmat[br].reshape(3, 3)
    p.set_tool(R, Rh.T @ (axis_pt(L_piv) - d.xpos[br]))
    start = s.wbc.ee_goal[br].copy()
    q_start = s.wbc.ee_rot[br].copy()
    sweep, tip = kw.get("sweep", 0.0), math.radians(kw.get("scoop_tilt", 40.0))
    n = int(round(kw.get("scoop_s", 1.6) / p.mv.TICK))
    for k in range(1, n + 1):
        u = smooth(k / n)
        s.wbc.set_ee_target(br, start + u * sweep * n_h,
                            quat=quat_mul(axis_quat(tilt_axis, u * tip), q_start))
        p.mv._tick(p.mv.TICK)
        track_slip()
    p.hold(0.2)
    Rh = d.xmat[br].reshape(3, 3)
    p.set_tool(R, Rh.T @ (sp(bowl_local) - d.xpos[br]))          # the bowl again for the lift and the carry
    q_scoop = s.wbc.ee_rot[br].copy()
    qerr = np.zeros(3)
    mujoco.mju_subQuat(qerr, q_scoop, d.xquat[br])
    p.note("swept", spoon_tilt_deg=round(tilt(), 1), hand_rot_err_deg=round(math.degrees(float(np.linalg.norm(qerr))), 1),
           slip_mm=round(slip["max"], 1), spoon_on_jar_N=round(forces(sbody, {jb})[0], 2),
           jar_moved_mm=round(float(np.linalg.norm((d.xpos[jb] - jar0)[:2])) * 1000, 1))
    # out of the jar ALONG the spoon's shaft (up and back toward the handle), so the shaft keeps passing
    # through the middle of the mouth and the loaded bowl never scrapes the neck (scoop7 lifted straight up:
    # the tilted shaft swept toward the near rim and the grains were shaken out), then straight up
    xs = d.xmat[sb].reshape(3, 3)[:, 0].copy()
    xs /= np.linalg.norm(xs)
    out_len = max(0.0, (rim_z + 0.012 - sp(bowl_local)[2]) / max(xs[2], 0.3))
    p.move_hand(R, s.wbc.ee_goal[br] + out_len * xs, quat=q_scoop, seconds=kw.get("lift_s", 1.4))
    lift_to = np.r_[s.wbc.ee_goal[br][:2], max(s.wbc.ee_goal[br][2], rim_z + 0.045)]
    p.move_hand(R, lift_to, quat=q_scoop, seconds=0.8)
    p.hold(0.3)
    track_slip()
    _, _, in_spoon = grains_where()
    out["scoop"] = dict(grains_in_spoon=int(in_spoon.sum()), spoon_tilt_deg=round(tilt(), 1), slip_mm=round(slip["max"], 1),
                        jar_moved_mm=round(float(np.linalg.norm((d.xpos[jb] - jar0)[:2])) * 1000, 1))
    p.note("scooped", **out["scoop"])
    mark("scooped")
    if kw.get("stop_after_scoop", 0):
        s.wbc.weights["ee_rot"] = rot_w0
        finish()
        return out
    # the left hand lets go of the jar (open, out to the left, up and back)
    if not kw.get("lid_off", False):
        p.grip(L, 0.0, 0.4)
        p.hold(0.15)
        QL = s.wbc.ee_rot[bl].copy()
        p.move_hand(L, s.wbc.ee_goal[bl] - 0.025 * RIGHT, quat=QL, seconds=0.4)
        p.move_hand(L, s.wbc.ee_goal[bl] + 0.07 * UP, quat=QL, seconds=0.5)
        p.move_hand(L, s.wbc.ee_goal[bl] - 0.03 * RIGHT - 0.12 * FWD + 0.02 * UP, seconds=0.7)

    # ================================================================== 6. over the mug, tip the powder in
    mc = d.xpos[mb].copy()
    # the bowl a little behind the mug's axis, so the tip edge the powder leaves over is above the axis
    over = np.r_[(mc - kw.get("pour_back", 0.0) * n_h)[:2], mc[2] + 0.085 + kw.get("pour_h", 0.035)]
    here = s.wbc.ee_goal[br].copy()
    p.move_hand(R, over, quat=q_scoop, seconds=kw.get("carry_s", 1.6),
                via=np.r_[((here + over) / 2)[:2], max(here[2], over[2]) + 0.03])
    p.hold(0.2)
    for _ in range(2):
        e = over - sp(bowl_local)
        if np.linalg.norm(e) < 0.004:
            break
        p.move_hand(R, s.wbc.ee_goal[br] + e, quat=q_scoop, seconds=0.6)
        p.hold(0.1)
    track_slip()
    _, _, in_spoon = grains_where()
    carried = int(in_spoon.sum())
    p.note("over the mug", grains_in_spoon=carried, bowl_off_axis_mm=round(float(np.linalg.norm((sp(bowl_local) - mc)[:2])) * 1000, 1))
    mark("over_mug")
    pour = math.radians(kw.get("pour_tilt", -80.0))
    q0 = s.wbc.ee_rot[br].copy()
    here = s.wbc.ee_goal[br].copy()
    n = int(round(kw.get("pour_s", 1.5) / p.mv.TICK))
    for k in range(1, n + 1):
        u = smooth(k / n)
        s.wbc.set_ee_target(br, here, quat=quat_mul(axis_quat(tilt_axis, u * pour), q0))
        p.mv._tick(p.mv.TICK)
        track_slip()
    p.hold(kw.get("pour_hold", 1.0))
    in_jar, in_mug, in_spoon = grains_where()
    out["pour"] = dict(grains_carried=carried, grains_in_mug=int(in_mug.sum()), still_in_spoon=int(in_spoon.sum()),
                       spoon_slip_mm=round(slip["max"], 2))
    p.note("poured", **out["pour"])
    mark("poured")
    # back to the hanging spoon (open side forward) over the mug
    q0 = s.wbc.ee_rot[br].copy()
    p.move_hand(R, s.wbc.ee_goal[br], quat=quat_mul(axis_quat(tilt_axis, -pour - tip), q0), seconds=kw.get("back_s", 1.2))
    p.hold(0.2)
    track_slip()
    if kw.get("end", "mug") == "hold":                     # the spoon raised over the mug, still in the hand
        p.move_hand(R, s.wbc.ee_goal[br] + [0, 0, kw.get("raise", 0.06)], seconds=1.0)
        p.hold(0.5)
        track_slip()
        out["hold"] = dict(spoon_slip_mm=round(slip["max"], 2), tilt_deg=round(tilt(), 1),
                           fingers=sorted(fingers_on(HR, sbody)),
                           bowl_over_mug_mm=round(float(np.linalg.norm((sp(bowl_local) - d.xpos[mb])[:2])) * 1000, 1))
        p.note("spoon raised over the mug", **out["hold"])
        mark("spoon_raised")
    s.wbc.weights["ee_rot"] = rot_w0 if kw.get("end", "mug") != "hold" else s.wbc.weights["ee_rot"]

    # ================================================================== 7. spoon left in the mug (or held)
    if kw.get("end", "mug") == "mug":
        Rh = d.xmat[br].reshape(3, 3)
        p.set_tool(R, pinch_off)
        BOWL_C = (ps.DISH_C, 0.0, 0.0)
        mc = d.xpos[mb].copy()

        def bowl_down():
            buf = np.zeros(6)
            for i in range(d.ncon):
                cc = d.contact[i]
                b1, b2 = int(m.geom_bodyid[cc.geom1]), int(m.geom_bodyid[cc.geom2])
                if {b1, b2} == {sb, mb} and abs(cc.frame[2]) > 0.8:
                    mujoco.mj_contactForce(m, d, i, buf)
                    if abs(buf[0]) > 0.08:
                        return True
            return False

        def rim():
            for cc in d.contact[:d.ncon]:
                g1, g2 = cc.geom1, cc.geom2
                if (g1 in spoon_geoms and int(m.geom_bodyid[g2]) == mb) or (g2 in spoon_geoms and int(m.geom_bodyid[g1]) == mb):
                    if cc.pos[2] > mc[2] + 0.05:
                        return True
            return False

        QH2 = s.wbc.ee_rot[br].copy()
        e = np.r_[(mc - sp(BOWL_C))[:2], (mc[2] + 0.009 + 0.012) - axis_pt(0.0)[2]]
        p.move_hand(R, s.wbc.ee_goal[br] + e, quat=QH2, seconds=0.9)
        p.hold(0.15)
        p.lower_until(R, bowl_down, speed=0.015, max_dist=0.04, quat=QH2)
        p.hold(0.15)
        tipp = axis_pt(0.0)
        D = FWD + 0.5 * RIGHT
        D /= np.linalg.norm(D)
        axt = np.cross(UP, D)
        st = np.r_[math.cos(math.radians(1.5)), math.sin(math.radians(1.5)) * axt]
        Rst = np.zeros(9)
        mujoco.mju_quat2Mat(Rst, st)
        Rst = Rst.reshape(3, 3)
        goal, qg = s.wbc.ee_pos(d, br).copy(), s.wbc.ee_rot[br].copy()
        rim_hit = False
        for _ in range(14):
            goal = tipp + Rst @ (goal - tipp)
            qg = quat_mul(st, qg)
            p.move_hand(R, goal, quat=qg, seconds=0.2)
            if rim():
                rim_hit = True
                break
        p.hold(0.2)
        p.move_hand(R, s.wbc.ee_pos(d, br), quat=s.wbc.ee_rot[br].copy(), seconds=0.3)
        p.finger_pose(R, {"thumb_0": 0.5, "thumb_1": 0.3, "thumb_2": 0.3, "index": 0.1, "middle": 0.1}, 0.6)
        p.hold(0.3)
        rest = axis_pt(0.0).copy()
        p.move_hand(R, s.wbc.ee_goal[br] + 0.01 * RIGHT - 0.04 * FWD + 0.01 * UP, seconds=0.5)
        rw = s.wbc.weights["ee_rot"]
        nr = int(round(0.6 / p.mv.TICK))
        for k in range(1, nr + 1):
            s.wbc.weights["ee_rot"] = rot_w0 + (rw - rot_w0) * (1.0 - smooth(k / nr))
            p.mv._tick(p.mv.TICK)
        s.wbc.weights["ee_rot"] = rot_w0
        p.move_hand(R, s.wbc.ee_goal[br] - 0.02 * FWD + 0.07 * UP, seconds=0.6)
        p.move_hand(R, np.r_[(d.qpos[:2] + 0.14 * RIGHT[:2] + 0.20 * FWD[:2]), 1.02], seconds=1.0)
        p.hold(0.4)
        inside = float(np.linalg.norm((axis_pt(0.0) - d.xpos[mb])[:2])) < 0.03 and axis_pt(0.0)[2] < d.xpos[mb][2] + 0.03
        out["leave"] = dict(rim_contact=rim_hit, spoon_tip_in_mug=bool(inside), tilt_deg=round(tilt(), 1),
                            spoon_moved_after_release_mm=round(float(np.linalg.norm(axis_pt(0.0) - rest)) * 1000, 1))
        p.note("spoon left in the mug", **out["leave"])
        mark("spoon_left")
    finish()
    g = out["grains"]
    out["ok"] = bool(s.checks.fell_at is None and out.get("unscrew", {"released": True})["released"]
                     and g["in_mug"] >= 1 and out["jar"]["tilt_deg"] < 5 and out["mug_moved_mm"] <= 5.0
                     and out["prop_contact_deepest_mm"] <= 2.0 and s.checks.torque_limit_violations == 0)
    return out
















