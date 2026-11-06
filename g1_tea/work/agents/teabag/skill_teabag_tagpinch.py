"""Tea-bag step as a skill: the LEFT hand pinches the tag (thumb tip against the index finger pad), lifts the
bag out on its string, carries it over the mug, lowers it into the mug (guarded by the bag's measured
support), lays the tag down at the mug's front-left and retreats.

Physics only: the hand moves through the whole-body controller and torque-limited motors; the tag is held
only by finger contact; bag, string and tag are free bodies (props_teabag). The controller is told the
load the hand carries from the hand-tag contact forces (wrist-sensor estimate), as in test_mug3.

run(p, s, mug_id=MUG, tag_body=..., ...) -> dict of measured results.
Start state: robot standing (stance used in the tests: (2.14, 1.88) facing +y), tag reachable and hanging
free (props_teabag START="rig"; from the tray start the pinch is not reachable - see REPORT.md)."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from motion import quat_from_axes

L = 1                                              # hand index (0 left, 1 right); right: free and in front of the tray
PINCH = {0: (0.1525, -0.035, 0.022), 1: (0.1532, 0.0365, 0.0192)}   # pinch centre, hand frame (probe_pinch/probe_tip_r)
# finger joint angles (rad) of the pinch (probe_pinch.py): thumb tip meets the index pad at PINCH
# at the pinch pose every thumb joint moves the thumb tip SIDEWAYS along the card (it swipes it away); only the
# index finger's curl moves its pad toward the thumb. So: the thumb sits in its contact pose from the start (a
# fixed anvil 2 mm in front of the tag), the index is opened 8 mm behind it, and the index curls to squeeze.
_A = {"thumb_0": -0.25, "thumb_1": 1.0, "thumb_2": 0.73, "index_0": -0.5, "index_1": -0.70, "middle_0": -1.5, "middle_1": -1.7}
INDEX_SQUEEZE = 1.25                               # |index_1| target for the squeeze (contact ~0.97)
APPROACH = {0: _A, 1: {k: (v if k == "thumb_0" else -v) for k, v in _A.items()}}   # right hand: mirrored
THUMB_CONTACT = {0: 0.73, 1: -0.73}                # thumb_2 where the tip meets a 1 mm card
TILT = 30.0                                        # fingers this much below horizontal: the palm clears the tray's front wall
TAG_LEN = 0.025                                    # tag length (m), props_teabag
F_PINCH = 0.8                                      # N squeeze


def _closures(s, b, angles):
    """finger_pose closures (0 open .. 1 closed) that put each finger joint at the given angle."""
    m = s.m
    out = {"ok": False}
    for a, lo, hi, high in s.fingers[b]:
        name = m.joint(int(np.nonzero(m.jnt_qposadr == a)[0][0])).name
        key = next(k for k in angles if k in name)
        c = (angles[key] - lo) / (hi - lo)
        out[key] = float(np.clip(c if high else 1 - c, 0, 1))
    return out


def run(p, s, mug_id="object_0316", tag="prop_teabag_tag", bag="prop_teabag_bag", place_offset=(-0.050, -0.050),
        lift_z=1.18, hand=L, stop_after=None, mug_stance=None, log=print):
    L = hand
    m, d = s.m, s.d
    b = s.hands[L]
    HAND = s.hand_parts[b]
    tb = m.body("env/" + tag).id
    bb = m.body("env/" + bag).id
    bb2 = m.body("env/" + bag + "_b").id
    bag_parts = {bb, bb2}
    m_bag = float(m.body_mass[bb] + m.body_mass[bb2])

    def bag_com():
        return (d.xipos[bb] * m.body_mass[bb] + d.xipos[bb2] * m.body_mass[bb2]) / m_bag
    mb = m.body("env/" + mug_id + "_body").id
    prop = {i for i in range(m.nbody) if m.body(i).name.startswith("env/prop_teabag")}
    robot = {i for i in range(1, m.nbody) if not m.body(i).name.startswith("env/")}
    tag_geom = m.geom("env/" + tag + "_geom").id
    out = {}

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

    def finger_normal(part_substr):
        """Normal force (N) between a finger link (name contains part_substr) and the tag."""
        f, buf = 0.0, np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if tag_geom not in (g1, g2):
                continue
            other = g2 if g1 == tag_geom else g1
            if part_substr not in m.body(m.geom_bodyid[other]).name:
                continue
            mujoco.mj_contactForce(m, d, i, buf)
            f += buf[0]
        return f

    def support(body_set, exclude=()):
        """Upward force on body_set from anything that is not the robot or the prop itself."""
        f, buf = 0.0, np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 in body_set) == (b2 in body_set):
                continue
            o = b2 if b1 in body_set else b1
            if o in robot or o in prop or o in exclude:
                continue
            mujoco.mj_contactForce(m, d, i, buf)
            f += abs(buf[0] * c.frame[2])
        return f

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

    # ---- 1. pre-shape and approach: handshake orientation (fingers forward, palm to the right, index on
    # top); the curled index goes behind the tag past its west edge, the thumb tucked in front
    R_t = d.xmat[tb].reshape(3, 3)
    # the grip frame is fixed to the tray (fingers toward the divider, +y): the hanging tag turns freely on its
    # string (it settles 0-20 deg off), and the closing pinch squares it up against the index pad
    nt = R_t[:, 2] * (1 if R_t[1, 2] > 0 else -1)                      # measured tag face normal (away from the robot)
    ang = 0.0     # (turning the grip with the tag's yaw swings the hand's 8 cm footprint into the tray: kept square to +y)
    n = np.array([math.sin(ang), math.cos(ang), 0.0])                  # grip square to the tag, within +-20 deg of +y
    # thumb-down grip: fingers along n tilted TILT down, the finger stack vertical with the middle finger on
    # top (it stays above the tray), the index finger's pad behind the tag, the thumb tip in front
    xh = n * math.cos(math.radians(TILT)) + np.array([0.0, 0.0, -math.sin(math.radians(TILT))])
    zh = np.array([0.0, 0.0, -1.0]) - (-xh[2]) * xh
    zh /= np.linalg.norm(zh)
    yh = np.cross(zh, xh)
    Q = quat_from_axes(xh, yh, zh)
    side = -1.0 if L == 1 else 1.0                                     # the index goes round the tag's west edge
    west = side * yh                                                   # (right hand: -y_h; left hand: +y_h)
    n = xh
    centre = lambda: d.xpos[tb] + d.xmat[tb].reshape(3, 3) @ m.geom_pos[tag_geom]   # noqa: E731 tag centre
    p.set_tool(L, PINCH[L])
    p.finger_pose(L, _closures(s, b, APPROACH[L]), 0.5)
    T = centre()
    # the tilted grip's index finger sticks out 1.3 cm toward the thumb below the pinch point, so it slides in
    # with its top 2 mm under the tag's lower edge (and its bottom above the next compartment's sticks), then
    # rises behind the tag; the pinch closes on the tag's lower part
    def tag_bottom():
        """Lowest corner of the tag (it may hang turned in its own plane)."""
        Rg = d.geom_xmat[tag_geom].reshape(3, 3)
        hs = m.geom_size[tag_geom]
        return float(d.geom_xpos[tag_geom][2] - abs(Rg[2, 0]) * hs[0] - abs(Rg[2, 1]) * hs[1] - abs(Rg[2, 2]) * hs[2])
    # slab start: the tag hangs in free air 3 cm in front of the heap; the pinch closes 1 cm above its lower
    # edge (the curled middle finger then stays above the slab, the index above the next compartment)
    slide_z = tag_bottom() + 0.010
    target = np.r_[(T - 0.002 * n + 0.002 * west)[:2], slide_z]       # index pad 5 mm behind the tag's back face
    pre = target + 0.004 * n + 0.035 * west
    sh = d.xpos[s.shoulder[b]].copy()
    t0 = d.time
    # as test_mug3: a point high in front of the tag, reached over the counter's edge through a via point
    # in front of the shoulder, then down onto the approach point
    # forward high (the index, 2.5 cm below the pinch, clears the tray's front wall), straight down to the
    # side of the bag (the index lies behind the tag, under the bag's overhang), then sideways behind the tag
    pre2 = target + 0.0485 * west                          # the index (5.9 cm across) fits between the next
                                                                       # compartment's divider and the bag: go down there
    back = 0.028 * np.array([n[0], n[1], 0.0]) / max(np.linalg.norm(n[:2]), 1e-9)   # 1 cm toward the robot: the heap's
    low = pre2 - back                                                  # face is only ~5 mm beyond the index
    over = np.r_[low[:2], max(pre2[2] + 0.065, 0.955 + 0.035)]         # the index clears the tray wall and the heap

    def touching_other(depth=0.0005):
        """Any part of the hand pressing (deeper than `depth`) into anything but the robot, the tag or the string."""
        for c in d.contact[:d.ncon]:
            if c.dist > -depth:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for a_, o in ((b1, b2), (b2, b1)):
                if a_ in HAND and o not in robot and o != tb and not m.body(o).name.startswith("env/prop_teabag_s"):
                    return m.body(o).name or m.geom(c.geom2 if a_ == b1 else c.geom1).name
        return None

    def servo(goal, tol=0.0025, iters=4):
        """Hand pinch point onto `goal` from its measured position (the loaded arm sits off its target by up
        to ~1 cm): the commanded target is shifted by the measured error, a few times."""
        for _ in range(iters):
            e = goal - s.wbc.ee_pos(d, b)
            if np.linalg.norm(e) < tol or touching_other(0.0015) is not None:
                break
            p.move_hand(L, s.wbc.ee_goal[b] + e, quat=Q, seconds=0.35)
            p.hold(0.15)
        return float(np.linalg.norm(goal - s.wbc.ee_pos(d, b)))

    # (a) hand up in front of the chest, fingers pointing up, palm already turned the way it will grip: the
    #     fingers never sweep low toward the counter's edge
    QA = quat_from_axes([0.0, 0.0, 1.0], yh, np.cross([0.0, 0.0, 1.0], yh))
    yaw0 = 2 * math.atan2(d.qpos[6], d.qpos[3])                        # measured heading
    fwd, rgt = np.array([math.cos(yaw0), math.sin(yaw0)]), np.array([math.sin(yaw0), -math.cos(yaw0)])
    wrist_up = np.r_[sh[:2] + 0.12 * fwd + (0.02 if L == 1 else -0.02) * rgt, 1.03]
    Rq = np.zeros(9)
    mujoco.mju_quat2Mat(Rq, QA)
    p.reach(L, wrist_up + Rq.reshape(3, 3) @ np.asarray(PINCH[L]), quat=QA, seconds=1.6)
    # (b) in the air: turn into the thumb-down grip and go over the tray at ~0.97 m, beside the bag
    plan_tag = p.plan_reach(L, target)                                 # whole body posed for the hand AT the tag
    p.reach(L, over + 0.015 * west + np.array([0.0, 0.0, 0.01]), quat=Q, seconds=1.8, plan=plan_tag)   # coarse (+-1.5 cm), clear
    servo(over)
    p.note("over the tray", hand_err_mm=round(float(np.linalg.norm(s.wbc.ee_pos(d, b) - over)) * 1000, 1), touching=touching_other())
    # (c) straight down beside the bag (guarded: a touch stops it)
    went = p.lower_until(L, lambda: touching_other() is not None, speed=0.02, max_dist=float(over[2] - low[2]), quat=Q)
    if touching_other() is None:
        servo(low)
        p.move_until(L, pre2 - s.wbc.ee_pos(d, b), lambda: touching_other() is not None, speed=0.02,
                     max_dist=float(np.linalg.norm(pre2 - s.wbc.ee_pos(d, b))), quat=Q)   # in under the slab
        servo(pre2)
    p.note("down", went_mm=round(went * 1000, 1), touching=touching_other(),
           hand_err_mm=round(float(np.linalg.norm(s.wbc.ee_pos(d, b) - pre2)) * 1000, 1))
    # (d) sideways behind the tag (guarded)
    T = centre()
    target = np.r_[(T - 0.002 * n + 0.002 * west)[:2], slide_z]
    here = s.wbc.ee_goal[b].copy()
    dirn = target - here
    dirn = target - s.wbc.ee_pos(d, b)
    p.move_until(L, dirn, lambda: touching_other() is not None, speed=0.02, max_dist=float(np.linalg.norm(dirn)), quat=Q)
    err = servo(target) if touching_other() is None else float(np.linalg.norm(target - s.wbc.ee_pos(d, b)))
    # (e) up behind the tag: the pinch point 7 mm above the tag's lower edge
    pinch_pt = target.copy()
    p.move_until(L, (0.0, 0.0, 1.0), lambda: touching_other() is not None, speed=0.015,
                 max_dist=float(max(pinch_pt[2] - s.wbc.ee_goal[b][2], 0.001)), quat=Q)
    if touching_other() is None:
        servo(pinch_pt, tol=0.002)
    err = float(np.linalg.norm(pinch_pt - s.wbc.ee_pos(d, b)))
    p.note("at the tag", hand_err_mm=round(err * 1000, 1), tag_moved_mm=round(float(np.linalg.norm(centre() - T)) * 1000, 1),
           touching=touching_other())
    if stop_after == "approach":
        return out

    # ---- 2. pinch: the thumb uncurls onto the tag's front face until it squeezes F_PINCH against the index
    ang = dict(APPROACH[L])
    # the index curls: its pad presses the tag back onto the thumb tip
    ang["index_1"] = INDEX_SQUEEZE * (1.0 if APPROACH[L]["index_1"] > 0 else -1.0)
    th = ang["thumb_2"]
    p.finger_pose(L, _closures(s, b, ang), 0.6)
    p.hold(0.3)
    HOLD["on"] = True
    f_th, f_ix = finger_normal("thumb"), finger_normal("index")
    Rb = d.xmat[b].reshape(3, 3)
    rel0 = Rb.T @ (centre() - d.xpos[b])
    p.note("pinched", thumb2=round(th, 3), thumb_N=round(f_th, 2), index_N=round(f_ix, 2))
    out["pinch"] = dict(thumb_N=round(f_th, 2), index_N=round(f_ix, 2), at_tag_err_mm=round(err * 1000, 1))
    if stop_after == "pinch":
        return out

    def slip():
        return float(np.linalg.norm(d.xmat[b].reshape(3, 3).T @ (centre() - d.xpos[b]) - rel0)) * 1000

    worst = {"slip": 0.0}

    def track(_s=None):
        worst["slip"] = max(worst["slip"], slip())

    def tick_tracking(_s):
        payload_tick(_s)
        track()

    p.mv.on_tick = tick_tracking

    # ---- 3. lift: straight up, the string draws the bag off its support; wait for the swing to die down
    # out from under the slab first (toward the robot), then up: the string slides over the slab's edge and
    # draws the bag off the heap
    here = s.wbc.ee_goal[b].copy()
    p.move_hand(L, here - 0.035 * np.array([n[0], n[1], 0.0]) + np.array([0.0, 0.0, 0.003]), quat=Q, seconds=0.9)
    here = s.wbc.ee_goal[b].copy()
    up = np.array([here[0], here[1] - 0.02, lift_z])
    p.reach(L, up, quat=Q, seconds=1.8)
    p.hold(1.2)
    out["lift"] = dict(bag_z=round(float(bag_com()[2]), 3), slip_mm=round(slip(), 1))
    p.note("lifted", **out["lift"], thumb_N=round(finger_normal("thumb"), 2))
    if stop_after == "lift":
        return out

    # ---- 3b. walk to the mug stance holding the tag up (walking network; the arm keeps its pose, the
    # fingers keep the pinch), then the standing controller takes over again
    if mug_stance is not None:
        from walk_carry import drive_carry
        out["walk"] = drive_carry(p, s, mug_stance[:2], math.radians(mug_stance[2]), hand=L, on_tick=tick_tracking,
                                  note=p.note)
        p.set_tool(L, PINCH[L])
        p.hold(0.8)
        out["after_walk"] = dict(slip_mm=round(slip(), 1), thumb_N=round(finger_normal("thumb"), 2),
                                 bag_z=round(float(bag_com()[2]), 3))
        p.note("after the walk", **out["after_walk"])
        if stop_after == "walk":
            return out

    # ---- 4. carry over the mug: the hand moves the bag (not the tag) over the mug's axis, slowly so the
    # bag does not swing; then small corrections from the bag's measured position
    mug = d.xpos[mb].copy()
    off = bag_com()[:2] - s.wbc.ee_pos(d, b)[:2]
    goal = np.r_[mug[:2] - off, lift_z]
    p.reach(L, goal, quat=Q, seconds=2.6, via=0.5 * (s.wbc.ee_goal[b] + goal) + np.array([0, -0.04, 0.01]))
    p.hold(1.0)
    for sec in (0.8, 0.6):
        e = mug[:2] - bag_com()[:2]
        p.move_hand(L, s.wbc.ee_goal[b] + np.r_[e, 0.0], quat=Q, seconds=sec)
        p.hold(0.8)
    out["over_mug"] = dict(bag_off_axis_mm=round(float(np.linalg.norm(bag_com()[:2] - mug[:2])) * 1000, 1),
                           bag_bottom_z=round(float(bag_com()[2] - 0.03), 3), slip_mm=round(slip(), 1))
    p.note("over the mug", **out["over_mug"])

    # ---- 5. lower into the mug, steered by the bag's measured position, until the mug carries the bag
    rim = float(mug[2] + 0.085)
    for attempt in range(3):
        went = p.lower_until(L, lambda: support(bag_parts) > 0.5 * m_bag * 9.81, speed=0.025,
                             max_dist=0.20, quat=Q, steer=lambda: (mug[:2] - bag_com()[:2]))
        p.hold(0.4)
        inside = bag_com()[2] < rim - 0.015 and np.linalg.norm(bag_com()[:2] - mug[:2]) < 0.018
        p.note("bag lowered", attempt=attempt + 1, inside=bool(inside), bag_z=round(float(bag_com()[2]), 3),
               off_axis_mm=round(float(np.linalg.norm(bag_com()[:2] - mug[:2])) * 1000, 1))
        if inside:
            break
        # caught on the rim: up 3 cm, line it up again from where it hangs, lower again
        p.move_hand(L, s.wbc.ee_goal[b] + [0.0, 0.0, 0.03 + went], quat=Q, seconds=0.8)
        p.hold(0.6)
        for sec in (0.6, 0.5):
            e = mug[:2] - bag_com()[:2]
            p.move_hand(L, s.wbc.ee_goal[b] + np.r_[e, 0.0], quat=Q, seconds=sec)
            p.hold(0.6)
    r_bag = float(np.linalg.norm(bag_com()[:2] - mug[:2]))
    out["lowered"] = dict(lowered_mm=round(went * 1000, 1), bag_z=round(float(bag_com()[2]), 3),
                          bag_off_axis_mm=round(r_bag * 1000, 1), mug_support_pct=round(100 * support(bag_parts) / (m_bag * 9.81)),
                          slip_mm=round(slip(), 1))
    p.note("lowered", **out["lowered"])
    # a little further so the string goes slack and the bag settles
    p.move_hand(L, s.wbc.ee_goal[b] + [0.0, 0.0, -0.015], quat=Q, seconds=0.6)
    p.hold(0.4)

    # ---- 6. the tag over the rim to the mug's front-left, down to the counter, let go
    spot = mug[:2] + np.asarray(place_offset)
    hz = float(s.wbc.ee_goal[b][2])
    p.move_hand(L, np.r_[spot + (s.wbc.ee_goal[b][:2] - centre()[:2]) * 0, max(hz - 0.03, 1.0)], quat=Q, seconds=1.6,
                via=np.r_[0.5 * (s.wbc.ee_goal[b][:2] + spot), max(hz, 1.03)])
    p.hold(0.3)

    def tag_on_counter():
        for c in d.contact[:d.ncon]:
            if tag_geom in (c.geom1, c.geom2):
                o = int(m.geom_bodyid[c.geom2 if c.geom1 == tag_geom else c.geom1])
                if o == 0 and "0278" in (m.geom(c.geom2 if c.geom1 == tag_geom else c.geom1).name or ""):
                    return True
        return False

    def hand_low():
        """Some part of the hand touching anything that is not the robot or the tea bag (counter, mug)."""
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for a, o in ((b1, b2), (b2, b1)):
                if a in HAND and o not in robot and o not in prop:
                    return True
        return False

    went2 = p.lower_until(L, lambda: tag_on_counter() or hand_low(), speed=0.03, max_dist=0.20, quat=Q)
    out["tag_down"] = dict(touched=tag_on_counter(), lowered_mm=round(went2 * 1000, 1),
                           tag_z=round(float(centre()[2]), 3), slip_mm=round(slip(), 1))
    p.note("tag at the counter", **out["tag_down"])
    out["max_slip_mm"] = round(worst["slip"], 1)
    p.move_hand(L, s.wbc.ee_pos(d, b), quat=Q, seconds=0.3)                  # unload: target to where the hand is
    HOLD["on"] = False
    p.mv.on_tick = payload_tick
    ang["thumb_2"] = APPROACH[L]["thumb_2"]
    p.finger_pose(L, _closures(s, b, ang), 0.5)                             # thumb back: the tag drops free
    p.hold(0.6)
    # ---- 7. retreat: back and up, out over the counter edge, fingers relaxed
    p.move_hand(L, s.wbc.ee_goal[b] + [0.0, -0.06, 0.05], quat=Q, seconds=0.8)
    p.reach(L, np.array([d.qpos[0] - 0.12, d.qpos[1] + 0.18, 1.0]), quat=Q, seconds=1.2)
    p.grip(L, 0.3, 0.4)
    p.hold(0.5)
    in_mug = (np.linalg.norm(bag_com()[:2] - d.xpos[mb][:2]) < 0.022 and d.xpos[mb][2] + 0.009 < bag_com()[2] < d.xpos[mb][2] + 0.10)
    out["end"] = dict(bag_in_mug=bool(in_mug), bag=np.round(bag_com(), 3).tolist(), tag=np.round(centre(), 3).tolist(),
                      tag_from_mug_mm=round(float(np.linalg.norm(centre()[:2] - d.xpos[mb][:2])) * 1000),
                      mug_moved_mm=round(float(np.linalg.norm(d.xpos[mb] - mug)) * 1000, 1), seconds=round(d.time - t0, 1))
    p.note("end", **out["end"])
    out["ok"] = bool(out["end"]["bag_in_mug"] and out.get("pinch", {}).get("thumb_N", 0) > 0.3
                     and out.get("max_slip_mm", 99) <= 5.0 and out["tag_down"]["touched"])
    return out
