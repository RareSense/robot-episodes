"""Tea-bag step as a skill: the RIGHT hand takes the tea bag out of the tray's loose-bag compartment by the
bag itself (a two-finger pinch on its top, as on the spoon handle), lifts it out with its string and tag
trailing, carries it over the mug, lowers it to just above the rim and lets it drop in; the hand retreats.

Physics only: every motion is a target for the whole-body controller (torque-limited motors); bag, string and
tag are free bodies (props_teabag, START="stand"), the bag held by finger contact alone. The pinch is the one
proven on the spoon (agents/spoon/skill_stir.py, pinch_search.py): palm down, fingers along the bag's face
normal; the thumb tip on the near (south) face, the index curling over the bag's top edge onto the far face;
both close together until both touch and squeeze > 2 N. Why not the tag: the Dex3's pinch cannot get a flat
or hanging card; why not a flat bag: two point contacts cannot stop a flat bag from swinging down about the
pinch, and it slips out past ~22 deg (friction 0.4); a standing bag hangs by turning about the pinch axis,
like the spoon. (The earlier tag-pinch skill is kept in skill_teabag_tagpinch.py.)

run(p, s, mug_id=MUG, **kw) -> dict (measured results, 'ok').
Start: robot standing e.g. at (1.90, 1.88) facing +y (everything planned from the measured stance and the
measured bag and mug poses); bag standing in the slot at the front of the loose-bag pile."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from motion import quat_from_axes, smooth

R_ = 1                                             # the right hand
TOOL_A = (0.1257, 0.0711, 0.0264)                  # spoon pinch point (hand frame): thumb tip / index pad midpoint
OPEN_A = {"thumb_0": 0.75, "thumb_1": 0.58, "thumb_2": 0.58, "index": 0.0, "middle": 1.0}   # thumb tip ~2.5 mm short of the near face, high (an open thumb hangs 2-3 cm lower and scrapes the divider: bag3-5)
SHUT_A = {"thumb_0": 0.75, "thumb_1": 0.68, "thumb_2": 0.68, "index": 0.68, "middle": 1.0}
PINCH_DOWN = 0.004                                 # pinch point this far below the bag's top edge
EXTRA = 0.05                                       # closure past the contact (squeeze)
F_SQUEEZE = 2.0                                    # N


def _names(s, b):
    m = s.m
    return {a: m.joint(int(np.nonzero(m.jnt_qposadr == a)[0][0])).name for a, *_ in s.fingers[b]}


def _targets(s, b, pose):
    out, nm = {}, _names(s, b)
    for a, lo, hi, high in s.fingers[b]:
        c = next((v for k, v in pose.items() if k in nm[a]), None)
        if c is not None:
            out[a] = lo + (c if high else 1 - c) * (hi - lo)
    return out


def close_until(p, hand, pose, done, seconds=1.2):
    """Finger targets ramp (minimum-jerk) toward `pose`, stopping where done() turns true."""
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


def run(p, s, mug_id="object_0316", bag="prop_teabag_bag", tag="prop_teabag_tag", carry_z=1.08, stop_after=None,
        log=print, **kw):
    m, d = s.m, s.d
    b = s.hands[R_]
    HAND = s.hand_parts[b]
    bb = m.body("env/" + bag).id
    tb = m.body("env/" + tag).id
    mb = m.body("env/" + mug_id + "_body").id
    bag_geom = m.geom("env/" + bag + "_geom").id
    hx = float(m.geom_size[bag_geom][0])
    robot = {i for i in range(1, m.nbody) if not m.body(i).name.startswith("env/")}
    prop = {i for i in range(m.nbody) if m.body(i).name.startswith("env/prop_teabag")}
    out = {"ok": False}
    t0 = d.time
    yaw = p.mv.heading()
    FWD = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    RIGHT = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    UP = np.array([0.0, 0.0, 1.0])
    out["stance"] = dict(xy=np.round(d.qpos[:2], 3).tolist(), heading_deg=round(math.degrees(yaw), 1))

    def bag_pt(local):
        return d.xpos[bb] + d.xmat[bb].reshape(3, 3) @ np.asarray(local, float)

    def forces(objs):
        f, deep, buf = 0.0, 0.0, np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 in HAND and b2 in objs) or (b2 in HAND and b1 in objs):
                mujoco.mj_contactForce(m, d, i, buf)
                f += abs(buf[0])
                deep = max(deep, -c.dist * 1000)
        return f, deep

    def fingers_on(objs, names=("thumb", "index", "middle")):
        hit = set()
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for a_, o in ((b1, b2), (b2, b1)):
                if a_ in HAND and o in objs:
                    hit.add(m.body(a_).name)
        return {n for n in names if any(n in h for h in hit)}

    def hand_load(objs):
        F, buf = np.zeros(3), np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b2 in HAND and b1 in objs:
                sign = 1.0
            elif b1 in HAND and b2 in objs:
                sign = -1.0
            else:
                continue
            mujoco.mj_contactForce(m, d, i, buf)
            F += sign * (c.frame.reshape(3, 3).T @ buf[:3])
        return F

    HOLD = {"on": False, "est": np.zeros(3)}

    def payload_tick(_s):
        F = hand_load(prop) if HOLD["on"] else np.zeros(3)
        F = np.array([0.0, 0.0, min(F[2], 0.0)])
        HOLD["est"] += (F - HOLD["est"]) * min(1.0, p.mv.TICK / 0.1)
        if np.linalg.norm(HOLD["est"]) > 0.005:
            s.wbc.payload[b] = HOLD["est"].copy()
        else:
            s.wbc.payload.pop(b, None)

    p.mv.on_tick = payload_tick

    def servo(goal, quat, tol=0.002, iters=4):
        for _ in range(iters):
            e = goal - s.wbc.ee_pos(d, b)
            if np.linalg.norm(e) < tol:
                break
            p.move_hand(R_, s.wbc.ee_goal[b] + e, quat=quat, seconds=0.3)
            p.hold(0.15)
        return float(np.linalg.norm(goal - s.wbc.ee_pos(d, b)))

    # ------------------------------------------------ 1. pinch from above (the spoon's pinch)
    nrm = d.xmat[bb].reshape(3, 3)[:, 2].copy()                # the bag's face normal, measured
    nrm[2] = 0.0
    nrm /= np.linalg.norm(nrm)
    if nrm @ FWD < 0:
        nrm = -nrm                                             # away from the robot: thumb on the near face
    QA = quat_from_axes(nrm, -UP, np.cross(nrm, -UP))          # palm down, fingers along the face normal
    p.set_tool(R_, TOOL_A)
    p.finger_pose(R_, OPEN_A, 0.4)
    target = bag_pt((hx - PINCH_DOWN, 0.0, 0.0))               # on the bag's mid-plane, 7 mm below its top edge
    pre = target + np.array([0.0, 0.0, 0.06])
    sh = d.xpos[s.shoulder[b]].copy()
    # up in front of the chest first, fingers pointing up, palm already facing forward (turned forward any
    # lower, the open index catches under the counter's front edge), then over and down to above the bag
    QU = quat_from_axes(UP, nrm, np.cross(UP, nrm))
    Ru = np.zeros(9)
    mujoco.mju_quat2Mat(Ru, QU)
    wrist_up = np.r_[(sh + 0.12 * FWD + 0.02 * RIGHT)[:2], 1.03]
    p.reach(R_, wrist_up + Ru.reshape(3, 3) @ np.asarray(TOOL_A), quat=QU, seconds=1.5)
    p.reach(R_, pre, quat=QA, seconds=1.8)
    p.settle_hand(R_, pre, tol=0.005, timeout=0.6)
    servo(pre, QA, tol=0.003)
    target = bag_pt((hx - PINCH_DOWN, 0.0, 0.0))
    p.move_hand(R_, target, quat=QA, seconds=0.9)
    err = servo(target, QA)
    p.note("at the bag", err_mm=round(err * 1000, 1), touching=sorted(fingers_on({bb})))
    if stop_after == "approach":
        return out
    # the index first: it comes down over the top edge onto the far face and holds the bag against the
    # divider it leans on (closing both together, the thumb pushed the bag's top away from the index: run bag2);
    # then the thumb, until both touch and the squeeze passes 2 N
    close_until(p, R_, {"index": SHUT_A["index"]}, lambda: "index" in fingers_on({bb}, ("index",))
                and forces({bb})[0] > 0.3, seconds=0.7)
    frac = close_until(p, R_, {k: v for k, v in SHUT_A.items() if "thumb" in k},
                       lambda: {"thumb", "index"} <= fingers_on({bb}, ("thumb", "index")) and forces({bb})[0] > F_SQUEEZE,
                       seconds=0.9)
    # keep squeezing: targets past the contact (a target AT the contact holds only ~0.2 N once the fingers stop:
    # run bag4 dropped the bag on the lift)
    # (only a little: closed further, the thumb tip and the index pad curl past each other and twist the bag out,
    # as the spoon notes say; run bag6 squeezed at 0.78 and dropped it on the lift)
    more = {}
    for a, lo, hi, high in s.fingers[b]:
        nm = _names(s, b)[a]
        if any(k in nm for k in ("thumb_1", "thumb_2", "index")):
            c_now = (p.w.qpos0[a] - lo) / (hi - lo)
            c_now = c_now if high else 1 - c_now
            more[nm.replace("right_hand_", "").replace("left_hand_", "").replace("_joint", "")] = min(1.0, c_now + EXTRA)
    close_until(p, R_, more, None, seconds=0.4)
    p.hold(0.25)
    f, deep = forces({bb})
    HOLD["on"] = True
    # slip = how far the grip point (the hand's pinch point) moves on the bag, in the bag's own frame; the bag
    # turning about the pinch axis (it hangs and swings, like the spoon) is not slip
    grip_on_bag = lambda: d.xmat[bb].reshape(3, 3).T @ (s.wbc.ee_pos(d, b) - d.xpos[bb])   # noqa: E731
    rel0 = grip_on_bag()
    out["pinch"] = dict(closure=round(frac, 2), force_N=round(f, 2), deepest_mm=round(deep, 2),
                        fingers=sorted(fingers_on({bb})), err_mm=round(err * 1000, 1))
    p.note("pinched", **out["pinch"])
    worst = {"slip": 0.0}

    def slip():
        g = grip_on_bag() - rel0
        g[2] = 0.0                                             # (across the 4 mm thickness: the fingers' squeeze, not slip)
        return float(np.linalg.norm(g)) * 1000

    def tick(_s):
        payload_tick(_s)
        worst["slip"] = max(worst["slip"], slip())

    p.mv.on_tick = tick
    if stop_after == "pinch":
        return out

    # ------------------------------------------------ 2. lift: straight up out of the slot, then to carry height
    here = s.wbc.ee_goal[b].copy()
    p.move_hand(R_, here + np.array([0.0, 0.0, 0.08]), quat=QA, seconds=1.2)
    p.move_hand(R_, np.r_[here[:2] - 0.03 * nrm[:2], carry_z], quat=QA, seconds=1.4)
    p.hold(0.5)
    held = {"thumb", "index"} <= fingers_on({bb}, ("thumb", "index"))
    out["lift"] = dict(held=held, bag_bottom_z=round(float(bag_pt((-hx, 0, 0))[2]), 3), slip_mm=round(slip(), 1),
                       tag_z=round(float(d.xpos[tb][2]), 3))
    p.note("lifted", **out["lift"])
    if not held:
        out["failed"] = "bag not in the pinch after the lift"
        return out
    if stop_after == "lift":
        return out

    # ------------------------------------------------ 3. over the mug, steered by the bag's measured position
    mug = d.xpos[mb].copy()
    rim = float(mug[2] + 0.085)
    bag_c = lambda: bag_pt((0.0, 0.0, 0.0))                     # noqa: E731
    off = bag_c()[:2] - s.wbc.ee_pos(d, b)[:2]
    goal = np.r_[mug[:2] - off, carry_z]
    p.reach(R_, goal, quat=QA, seconds=2.4, via=0.5 * (s.wbc.ee_goal[b] + goal) + np.array([0.0, -0.03, 0.03]))
    p.hold(0.6)
    for sec in (0.6, 0.5, 0.5):
        e = mug[:2] - bag_c()[:2]
        if np.linalg.norm(e) < 0.003:
            break
        p.move_hand(R_, s.wbc.ee_goal[b] + np.r_[e, 0.0], quat=QA, seconds=sec)
        p.hold(0.5)
    out["over_mug"] = dict(bag_off_axis_mm=round(float(np.linalg.norm(bag_c()[:2] - mug[:2])) * 1000, 1),
                           bag_bottom_z=round(float(bag_pt((-hx, 0, 0))[2]), 3), slip_mm=round(slip(), 1))
    p.note("over the mug", **out["over_mug"])

    # ------------------------------------------------ 4. down until the fingers are just above the rim (the
    # held hand reaches ~1.1 cm below the pinch point), steered; the bag is then well inside the mug
    def hand_low():
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for a_, o in ((b1, b2), (b2, b1)):
                if a_ in HAND and o not in robot and o not in prop:
                    return True
        return False

    def low_enough():
        return s.wbc.ee_pos(d, b)[2] < rim + 0.020 or hand_low()

    went = p.lower_until(R_, low_enough, speed=0.03, max_dist=0.25, quat=QA, steer=lambda: (mug[:2] - bag_c()[:2]))
    p.hold(0.4)
    out["lowered"] = dict(lowered_mm=round(went * 1000, 1), bag_bottom_z=round(float(bag_pt((-hx, 0, 0))[2]), 3),
                          bag_off_axis_mm=round(float(np.linalg.norm(bag_c()[:2] - mug[:2])) * 1000, 1),
                          slip_mm=round(slip(), 1), hand_touch=hand_low())
    p.note("lowered", **out["lowered"])
    out["max_slip_mm"] = round(worst["slip"], 1)

    # ------------------------------------------------ 5. let go: the index opens (the thumb stays: opening it
    # would swing its tip down onto the rim); the bag drops the last few cm into the mug
    p.move_hand(R_, s.wbc.ee_pos(d, b), quat=QA, seconds=0.3)                  # unload: target to where it is
    HOLD["on"] = False
    p.mv.on_tick = payload_tick
    close_until(p, R_, {"index": 0.0}, None, seconds=0.5)
    p.hold(1.0)
    # ------------------------------------------------ 6. retreat: up, then back by the body
    p.move_hand(R_, s.wbc.ee_goal[b] + np.array([0.0, 0.0, 0.08]), quat=QA, seconds=0.8)
    p.reach(R_, np.r_[(d.qpos[:2] + 0.20 * FWD[:2] + 0.12 * RIGHT[:2]), 1.0], quat=QA, seconds=1.4)
    p.finger_pose(R_, {"thumb_0": 0.5, "thumb_1": 0.3, "thumb_2": 0.3, "index": 0.2, "middle": 0.3}, 0.5)
    p.hold(1.0)
    in_r = float(np.linalg.norm(bag_c()[:2] - d.xpos[mb][:2]))
    in_mug = bool(in_r < 0.021 and d.xpos[mb][2] + 0.005 < bag_pt((-hx, 0, 0))[2] < d.xpos[mb][2] + 0.085)
    out["end"] = dict(bag_in_mug=in_mug, bag_off_axis_mm=round(in_r * 1000, 1), bag_bottom_z=round(float(bag_pt((-hx, 0, 0))[2]), 3),
                      tag=np.round(d.xpos[tb], 3).tolist(), mug_moved_mm=round(float(np.linalg.norm(d.xpos[mb] - mug)) * 1000, 1),
                      seconds=round(d.time - t0, 1))
    p.note("end", **out["end"])
    out["ok"] = bool(in_mug and out["pinch"]["force_N"] > 1.0 and out["max_slip_mm"] <= 5.0)
    return out
