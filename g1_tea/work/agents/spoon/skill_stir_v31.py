"""Skill: the RIGHT hand takes the spoon from the tray's FRESH tub, brings it to the mug, the LEFT hand
steadies the mug (palm on its left side), the spoon stirs in circles inside the mug and is left standing in
it (bowl on the floor, handle leaning on the rim); the hands let go and retreat.

Physics only: every motion is a target for the whole-body controller (torque-limited motors); the spoon
is a free 17.5 g body held by finger contact alone. Steps:
1. pinch from above: palm down, fingers forward; the thumb comes down on the near flat face of the handle
   end, the index curls onto the far face (the Dex3 pinch closes along the fingers, here along the face
   normal); fingers close until both touch and squeeze ~2 N;
2. lift: forward over the tub's inside and up - the spoon swings upright in the two-finger pinch and hangs;
   straight up until the bowl clears the tub;
3. the hand turns to a handshake (palm left, index up) round the pinch while the spoon keeps hanging, then
   the middle finger closes on the handle 5.7 cm below the index and the thumb moves between them: a
   three-point grip, rigid in the hand;
4. over the mug; the left hand comes to the mug's left side by the proven step-1 path (front, above, down
   beside it, palm in), fingers lightly round it;
5. the spoon goes down (steered by its measured tip) until the bowl is 3 cm over the floor and circles
   (radius ramps 0 -> 5 mm -> 0: the mug's cavity is 42 mm wide, the bowl 25 mm) - slip and contacts measured;
6. down until the bowl touches the floor, the top tipped back until the handle rests on the back rim,
   the hand unloads (target to where it is), opens and backs off; the left hand opens and backs off.
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
sys.path.insert(0, str(WORK / "agents/spoon"))
import props_spoon as ps  # noqa: E402
from motion import quat_from_axes, quat_mul, smooth  # noqa: E402
from tea_sim import MUG  # noqa: E402

L, R = 0, 1
PINCH_L = 0.122                      # where along the spoon (from the bowl tip) the index/thumb pinch sits (1.8 cm below the end)
OPEN_A = {"thumb_0": 0.75, "thumb_1": 0.45, "thumb_2": 0.45, "index": 0.0, "middle": 0.0}  # thumb back, index straight over
SHUT_A = {"thumb_0": 0.75, "thumb_1": 0.68, "thumb_2": 0.68, "index": 0.68, "middle": 0.0}  # both close together
# pinch_search.py (kinematics): at thumb 0.60 / index 0.58 the thumb tip and the index pad face each other
# along the pinch axis 3.1 mm apart (misalignment 0.08); closed further they curl past each other: the thumb
# rides up to the handle's end corner (run stir5), the index meets the far face 3 cm lower (run stir7)
# handshake three-point grip (grip3_search.py, kinematics, handle through the pinch point (0.1257, 0.0711)):
# index and middle pads reach the far face at closure 0.59 (contacts z +-0.0285); the thumb tip is centred
# between them on the near face (front 0.8 mm into it, z -0.0002) at thumb_0/1/2 = 0.50/0.54/0.66
MIDDLE3 = {"middle": 0.62}   # touches at 0.59; 0.60 (stir15) left the index barely touching and the spoon fell out
THUMB3 = {"thumb_0": 0.50, "thumb_1": 0.54, "thumb_2": 0.67}
INDEX3 = {"index": 0.62}
TOOL_A = (0.1257, 0.0711, 0.0264)   # midpoint of that pinch (hand frame)


class Ctx:
    pass


def _finger_names(s, b):
    m = s.m
    return {a: m.joint(int(np.nonzero(m.jnt_qposadr == a)[0][0])).name for a, *_ in s.fingers[b]}


def _targets(s, b, pose):
    """Joint targets for the finger joints named (by substring) in `pose`; joints not named are left out."""
    out = {}
    for a, lo, hi, high in s.fingers[b]:
        name = _finger_names(s, b)[a]
        c = next((v for k, v in pose.items() if k in name), None)
        if c is not None:
            out[a] = lo + (c if high else 1 - c) * (hi - lo)
    return out


def ramp_fingers(p, hand, pose, seconds):
    """Only the named finger joints move (minimum-jerk) to their closures; the others keep their targets."""
    close_until(p, hand, pose, None, seconds)


def close_until(p, hand, pose, done, seconds=1.2):
    """Fingers ramp from where their targets are toward `pose` (minimum-jerk), stopping where done()
    turns true (a force-limited gripper closing to contact): the targets stay where they reached."""
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


def run(p, s, **kw) -> dict:
    m, d = s.m, s.d
    c = Ctx()
    out = {}
    br, bl = s.hands[R], s.hands[L]
    sb = m.body("env/" + ps.BODY).id
    mb = m.body("env/" + MUG + "_body").id
    spoon_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == sb}
    HR, HL = s.hand_parts[br], s.hand_parts[bl]
    robot = {i for i in range(1, m.nbody) if not m.body(i).name.startswith("env/")}
    t_start = d.time
    mug0 = d.xpos[mb].copy()
    out["ok"] = False
    # everything is planned from the MEASURED stance (walking lands 3-5 cm / 5-8 deg off its goal) and the
    # measured spoon and mug poses: heading frame of the robot as it stands now
    yaw = p.mv.heading()
    FWD = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    LEFT = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    RIGHT, UP = -LEFT, np.array([0.0, 0.0, 1.0])
    out["stance"] = dict(xy=np.round(d.qpos[:2], 3).tolist(), heading_deg=round(math.degrees(yaw), 1))

    def sp(local):
        return d.xpos[sb] + d.xmat[sb].reshape(3, 3) @ np.asarray(local, float)

    def axis_pt(L_):
        return sp((L_, 0.0, ps.HZ))

    def forces(body_set_a, body_set_b):
        """Total contact normal force (N) between two body sets and the deepest contact (mm)."""
        f, deep, buf = 0.0, 0.0, np.zeros(6)
        for i in range(d.ncon):
            cc = d.contact[i]
            b1, b2 = int(m.geom_bodyid[cc.geom1]), int(m.geom_bodyid[cc.geom2])
            if (b1 in body_set_a and b2 in body_set_b) or (b2 in body_set_a and b1 in body_set_b):
                mujoco.mj_contactForce(m, d, i, buf)
                f += abs(buf[0])
                deep = max(deep, -cc.dist * 1000)
        return f, deep

    def fingers_on_spoon(names):
        hit = set()
        for cc in d.contact[:d.ncon]:
            g1, g2 = cc.geom1, cc.geom2
            for g, o in ((g1, g2), (g2, g1)):
                if o in spoon_geoms and int(m.geom_bodyid[g]) in HR:
                    hit.add(m.body(int(m.geom_bodyid[g])).name)
        return {n for n in names if any(n in h for h in hit)}

    def rel_hand():
        Rh = d.xmat[br].reshape(3, 3)
        return Rh.T @ (axis_pt(0.0) - d.xpos[br]), Rh.T @ (axis_pt(ps.LENGTH) - d.xpos[br])

    def tilt():
        a = axis_pt(ps.LENGTH) - axis_pt(0.0)
        return math.degrees(math.acos(np.clip(a[2] / np.linalg.norm(a), -1, 1)))

    # ---------------------------------------------------------------- 1. pinch from above
    nrm = d.xmat[sb].reshape(3, 3)[:, 2].copy()                # the handle's face normal, measured
    nrm[2] = 0.0
    nrm /= np.linalg.norm(nrm)
    if nrm @ FWD < 0:
        nrm = -nrm                                            # pointing away from the robot: thumb on the near face
    QA = quat_from_axes(nrm, -UP, np.cross(nrm, -UP))         # palm down, fingers along the face normal
    tool_a = np.array(TOOL_A)
    p.note("pinch tool point (hand frame)", tool_a=list(TOOL_A))
    p.set_tool(R, tool_a)
    p.finger_pose(R, OPEN_A, 0.4)
    target = axis_pt(PINCH_L)
    pre = target + np.array([0.0, 0.0, 0.06])
    sh = d.xpos[s.shoulder[br]].copy()
    p.reach(R, pre, quat=QA, seconds=kw.get("reach_s", 1.8), via=np.r_[(sh + 0.16 * FWD + 0.05 * RIGHT)[:2], pre[2] + 0.04])
    p.settle_hand(R, pre, tol=0.005, timeout=0.6)
    p.move_hand(R, target, quat=QA, seconds=0.8)
    err = p.settle_hand(R, target, tol=0.003, timeout=0.4)
    # the loaded arm sits ~8 mm off its target (runs stir4-6): correct the command by the measured error
    # of the hand point (joint encoders), as a robot servoing its hand would
    for _ in range(4):
        e = target - s.wbc.ee_pos(d, br)
        if np.linalg.norm(e) < 0.002:
            break
        p.move_hand(R, s.wbc.ee_goal[br] + e, quat=QA, seconds=0.3)
        p.hold(0.15)
    err = float(np.linalg.norm(target - s.wbc.ee_pos(d, br)))
    p.note("at the handle end", err_mm=round(err * 1000, 1), touching=sorted(fingers_on_spoon(["thumb", "index", "middle"])))
    frac = close_until(p, R, SHUT_A, lambda: {"thumb", "index"} <= fingers_on_spoon(["thumb", "index"])
                       and forces(HR, {sb})[0] > 2.0, seconds=0.9)
    p.hold(0.25)
    f, deep = forces(HR, {sb})
    out["pinch"] = dict(closure=round(frac, 2), force_N=round(f, 2), deepest_mm=round(deep, 2),
                        fingers=sorted(fingers_on_spoon(["thumb", "index", "middle"])))
    p.note("pinched", **out["pinch"])

    # ---------------------------------------------------------------- 2. lift: upright, then out
    # the pinch goes round the bowl (resting on the tub floor) to 4 deg from vertical: the spoon turns
    # upright in the pinch and its handle leaves the rim; moving straight forward-and-up instead slid the
    # pinch up the handle and off its end (run stir4)
    C = axis_pt(0.0)
    P0 = s.wbc.ee_pos(d, br)
    lean = (P0 - C) * np.array([1.0, 1.0, 0.0])
    hl = float(np.linalg.norm(lean))
    lean = lean / hl if hl > 1e-6 else -FWD                   # horizontal direction the spoon leans to (measured)
    ang0 = math.atan2(hl, (P0 - C)[2])
    r_ = float(np.linalg.norm(P0 - C))
    upright = C + r_ * (math.sin(math.radians(4.0)) * lean + math.cos(math.radians(4.0)) * UP)
    mid_ang = 0.5 * (ang0 + math.radians(4.0))
    via = C + r_ * (math.sin(mid_ang) * lean + math.cos(mid_ang) * UP)
    p.move_hand(R, upright, quat=QA, seconds=0.8, via=via)
    p.hold(0.2)
    p.note("upright", tilt_deg=round(tilt(), 1), fingers=sorted(fingers_on_spoon(["thumb", "index", "middle"])))
    p.move_hand(R, s.wbc.ee_goal[br] + [0.0, 0.0, 0.13], quat=QA, seconds=1.2)
    p.hold(0.3)
    held = {"thumb", "index"} <= fingers_on_spoon(["thumb", "index"])
    p.note("out of the tub", tilt_deg=round(tilt(), 1), tip_z=round(float(axis_pt(0.0)[2]), 3), held=held,
           fingers=sorted(fingers_on_spoon(["thumb", "index", "middle"])))
    if not held:
        out["failed"] = "spoon not in the pinch after the lift"
        return out

    # ---------------------------------------------------------------- 3. handshake, three-point grip
    QH = quat_from_axes(FWD, LEFT, UP)                        # palm left, index up (robot's heading)
    p.move_hand(R, s.wbc.ee_goal[br] + 0.03 * LEFT - 0.04 * FWD, quat=QH, seconds=1.0)
    p.hold(0.2)                                       # the hanging spoon settles
    p.note("handshake", tilt_deg=round(tilt(), 1), fingers=sorted(fingers_on_spoon(["thumb", "index", "middle"])))
    ramp_fingers(p, R, MIDDLE3, 0.5)                          # middle onto the far face, 5.7 cm below the index
    ramp_fingers(p, R, THUMB3, 0.7)                           # thumb slides down the near face to between them
    ramp_fingers(p, R, INDEX3, 0.3)
    p.hold(0.2)
    f, deep = forces(HR, {sb})
    out["grip3"] = dict(force_N=round(f, 2), deepest_mm=round(deep, 2), tilt_deg=round(tilt(), 1),
                        fingers=sorted(fingers_on_spoon(["thumb", "index", "middle"])))
    p.note("three-point grip", **out["grip3"])
    if "middle" not in out["grip3"]["fingers"]:
        out["failed"] = "no three-point grip"
        return out
    Rh = d.xmat[br].reshape(3, 3)
    p.set_tool(R, Rh.T @ (axis_pt(PINCH_L) - d.xpos[br]))   # the spoon axis where it is pinched, measured now

    def tool_for_tip(tip_target):
        """Hand tool target that puts the spoon tip at tip_target (rigid grip, measured offset)."""
        return np.asarray(tip_target) + (s.wbc.ee_pos(d, br) - axis_pt(0.0))

    # ---------------------------------------------------------------- 4. left hand steadies the mug, then the spoon over it
    mc = d.xpos[mb].copy()

    def level_cmd():
        """The hand's commanded orientation turned by the spoon's measured tilt, so the spoon hangs plumb
        (as test_kettle5 levels the kettle): the loaded arm otherwise leaves the hand ~25-30 deg off (stir10)."""
        kz = axis_pt(ps.LENGTH) - axis_pt(0.0)
        kz /= np.linalg.norm(kz)
        ax = np.cross(kz, [0.0, 0.0, 1.0])
        sa = float(np.linalg.norm(ax))
        cmd = s.wbc.ee_rot[br].copy()
        if sa < 1e-6:
            return cmd
        ang = math.atan2(sa, kz[2])
        qc = np.r_[math.cos(ang / 2), math.sin(ang / 2) * ax / sa]
        return quat_mul(qc, cmd)

    # the spoon waits to the right of the mug, tip 14 cm over the counter, while the left hand comes in
    # (stir10: the left hand's approach passes 10 cm over the mug and knocked a spoon held there out of the grip)
    here = s.wbc.ee_goal[br].copy()
    wait = tool_for_tip(np.r_[(mc + 0.23 * RIGHT - 0.05 * FWD)[:2], 1.00])   # in front of the right shoulder (stir11: at x 1.83 the arms met and it fell)
    p.move_hand(R, wait, quat=QH, seconds=kw.get("carry_s", 1.8),
                via=np.r_[((wait + here) / 2 - 0.02 * FWD)[:2], max(wait[2], here[2]) + 0.05])
    p.hold(0.2)
    for _ in range(2):                                        # level it here, clear of the mug and the left hand
        p.move_hand(R, s.wbc.ee_goal[br], quat=level_cmd(), seconds=0.6)
        p.hold(0.1)
    QH = s.wbc.ee_rot[br].copy()
    p.note("spoon waiting by the mug", tilt_deg=round(tilt(), 1), tip=np.round(axis_pt(0.0), 3).tolist(),
           fingers=sorted(fingers_on_spoon(["thumb", "index", "middle"])))
    # left hand: test_mug3 / mugfast's proven side approach (front, above the mug's left side, down beside
    # it 1.5 cm out, palm in), fingers only lightly round it
    QL = quat_from_axes(FWD, LEFT, UP)                        # fingers forward, palm (-y_h) to the mug
    p.set_tool(L, (0.13, -0.072, 0.0))                        # the mug's axis when the hand is round it
    p.grip(L, 0.4, 0.25)
    cL = mc + np.array([0.0, 0.0, 0.050])
    c_out = cL - 0.015 * RIGHT
    aboveL = c_out + np.array([0.0, 0.0, 0.10])
    frontL = aboveL - 0.14 * FWD
    shl = d.xpos[s.shoulder[bl]].copy()
    plan = p.plan_reach(L, frontL, leans=(0, 4, 8, 12, 16, 20, 24, 28))
    p.reach(L, frontL, quat=QL, seconds=kw.get("left_s", 1.4), settle=0.0, plan=plan,
            via=np.r_[(shl + 0.08 * FWD)[:2], frontL[2] - 0.02])
    p.move_hand(L, aboveL, quat=QL, seconds=0.6)
    p.settle_hand(L, aboveL, tol=0.006, timeout=0.3)
    p.move_hand(L, c_out, quat=QL, seconds=0.7)
    def left_touch():
        return forces(HL, {mb})[0] > 0.5

    p.move_until(L, RIGHT, left_touch, speed=0.015, max_dist=0.03, quat=QL, ease_out=0.1)
    # rest against it, do not press: the target 2 mm back from where it touched (stir8: pressing moved it 9 mm;
    # stir23/24: gripping it with the side grip saturated the left arm and pushed it 6-36 mm)
    p.move_hand(L, s.wbc.ee_pos(d, bl) - 0.002 * RIGHT, quat=QL, seconds=0.3)
    p.grip(L, 0.35, 0.4)
    p.hold(0.2)
    out["steady"] = dict(left_force_N=round(forces(HL, {mb})[0], 2),
                         mug_moved_mm=round(float(np.linalg.norm(d.xpos[mb] - mug0)) * 1000, 1),
                         spoon_still_held=sorted(fingers_on_spoon(["thumb", "index", "middle"])))
    p.note("left hand on the mug", **out["steady"])
    # the spoon over the mug (hand only), leveled from its measured axis, one slow tip correction
    mc = d.xpos[mb].copy()
    # tip 5.5 cm over the rim (stir13: 1.04 put the right arm high across the body and it fell during the
    # correction); no re-level here (stir12: re-leveling over the mug swung the bowl into the rim)
    # the hand orientation task is weighted 10 against 100 for position; with the spoon 12 cm under the pinch
    # the orientation drift (12-25 deg) moved the bowl more than the stir did (stir14: 2 mm, 0.46 turns).
    # Weighted as the position task from here until the stir ends (a controller setting, restored after).
    rot_w0 = s.wbc.weights["ee_rot"]
    s.wbc.weights["ee_rot"] = kw.get("rot_w", 100.0)
    tip_goal = np.array([mc[0], mc[1], 1.00])
    above = tool_for_tip(tip_goal)
    p.move_hand(R, above, quat=QH, seconds=1.2, via=(above + s.wbc.ee_goal[br]) / 2 + [0.0, 0.0, 0.03])
    p.hold(0.15)
    e = tip_goal - axis_pt(0.0)
    if np.linalg.norm(e) > 0.03:
        e *= 0.03 / np.linalg.norm(e)                         # one slow correction, at most 3 cm
    p.move_hand(R, s.wbc.ee_goal[br] + e, quat=QH, seconds=0.7)
    p.hold(0.15)
    out["carry"] = dict(tip_err_before_mm=round(float(np.linalg.norm(e)) * 1000, 1),
                        tip_err_after_mm=round(float(np.linalg.norm(tip_goal - axis_pt(0.0))) * 1000, 1),
                        tilt_deg=round(tilt(), 1))
    p.note("over the mug", **out["carry"], fingers=sorted(fingers_on_spoon(["thumb", "index", "middle"])))
    if not {"thumb", "index", "middle"} <= fingers_on_spoon(["thumb", "index", "middle"]):
        out["failed"] = "grip lost before the mug"
        s.wbc.weights["ee_rot"] = rot_w0
        return out

    # ---------------------------------------------------------------- 5. into the mug and stir
    mc = d.xpos[mb].copy()
    floor_z = mc[2] + 0.009

    def tip_low():
        return axis_pt(0.0)[2] < floor_z + 0.030 or forces({sb}, {mb})[0] > 0.05

    p.lower_until(R, tip_low, speed=0.03, max_dist=0.17, quat=QH, steer=lambda: (mc - axis_pt(0.0))[:2])
    p.hold(0.15)
    # VISIBLE stir. Approach as run 26 (pinch point = tool, bowl lowered inside the mug, tip ~3 cm over the
    # floor). Only now the BOWL becomes the controller's tool point; the hand-position task is weighted 300
    # against orientation 100 so the long tool point does not make the controller turn the hand instead of
    # moving it (stir29: equal weights, the bowl ended 63-108 mm off its target). Level about the bowl, slowly,
    # then the bowl centre circles the mug's axis; the commanded circle's radius and centre adapt every tick
    # from the MEASURED bowl (the arm attenuates a commanded circle ~0.5x).
    # Cavity (thick-wall twin mug, default scene): wall hulls' inner flats 27.25 mm from the axis (cavity.py;
    # the 20.9 mm figure was a bounding-box edge, not a face); bowl half-width 12.5 mm -> centre <= ~11.75 mm
    # keeps 3 mm clearance.
    BOWL_C = (ps.BOWL["pos"][0], 0.0, 0.0)
    pinch_tool = s.wbc.ee_off[br].copy()
    ee_w0 = s.wbc.weights["ee"]
    s.wbc.weights["ee"] = kw.get("pos_w", 300.0)
    p.set_tool(R, d.xmat[br].reshape(3, 3).T @ (sp(BOWL_C) - d.xpos[br]))
    centre = s.wbc.ee_pos(d, br).copy()
    p.hold(0.2)
    p.move_hand(R, centre, quat=level_cmd(), seconds=1.2)     # level about the bowl (the bowl stays put)
    p.hold(0.2)
    p.move_hand(R, centre, quat=level_cmd(), seconds=0.6)     # and the remainder
    p.hold(0.15)
    QH = s.wbc.ee_rot[br].copy()
    # the bowl centre onto the mug's axis, closed loop on the measured bowl (stir30: leveling moved the bowl
    # 18-22 mm off the axis and one correction left it 14 mm off; the circle then hit the wall)
    for _ in range(5):
        e = (mc - sp(BOWL_C))[:2]
        if np.linalg.norm(e) < 0.002:
            break
        p.move_hand(R, s.wbc.ee_goal[br] + np.r_[e, 0.0], quat=QH, seconds=0.4)
        p.hold(0.15)
    centre = s.wbc.ee_goal[br].copy()
    out["centred_mm"] = round(float(np.linalg.norm((mc - sp(BOWL_C))[:2])) * 1000, 1)
    tilt_start = tilt()
    rel0 = rel_hand()
    hand_mug0 = d.xpos[mb].copy()
    turns, period, rad = kw.get("turns", 4.5), kw.get("period", 1.75), kw.get("radius", 0.011)
    r_cap = kw.get("radius_cap", 0.013)
    freq = 1.0 / period
    wr = [k for k, i in enumerate(s.motors.ids) if m.actuator(int(i)).name in ("right_wrist_pitch_joint", "right_wrist_yaw_joint")]
    fing = [k for k, i in enumerate(s.motors.ids) if m.actuator(int(i)).name.startswith("right_hand_")]
    wr_peak, fing_peak = 0.0, 0.0
    ramp_s = 0.5 * period                                     # half a turn to ramp the radius up, and down
    T = turns * period + ramp_s
    n = int(round(T / p.mv.TICK))
    slip_tip, slip_end, sm_force, sm_deep, hand_mug_f = 0.0, 0.0, 0.0, 0.0, 0.0
    track, track_tip, track_end, tilts = [], [], [], []
    gain = kw.get("ff_gain", 1.0)                             # commanded/desired radius, adapted below
    c_cmd = centre[:2].copy()
    win = max(1, int(round(period / p.mv.TICK)))
    t_stir = d.time
    for k in range(1, n + 1):
        t = k * p.mv.TICK
        rr = smooth(t / ramp_s) * smooth((T - t) / ramp_s)
        ang = 2 * math.pi * freq * t
        u = np.array([math.cos(ang), math.sin(ang)])
        cmd = centre.copy()
        cmd[:2] = c_cmd + gain * rad * rr * u
        s.wbc.set_ee_target(br, cmd, quat=QH)
        p.mv._tick(p.mv.TICK)
        bw = sp(BOWL_C)[:2] - d.xpos[mb][:2]                   # bowl centre about the mug's (measured) axis
        track.append(bw.copy())
        track_tip.append((axis_pt(0.0)[:2] - d.xpos[mb][:2]).copy())
        track_end.append((axis_pt(ps.LENGTH)[:2] - d.xpos[mb][:2]).copy())
        tilts.append(tilt())
        if k > win and rr > 0.99:
            last = np.array(track[-win:])
            r_meas = float(np.linalg.norm(last - last.mean(axis=0), axis=1).mean())
            gain = float(np.clip(gain * (1.0 + 0.015 * (rad - r_meas) / rad), 0.5, 4.0))
            c_cmd -= 0.01 * last.mean(axis=0)                 # the circle's centre onto the axis
        if np.linalg.norm(bw) > r_cap:                        # never toward the wall
            gain = max(0.5, gain * 0.9)
        r0, r1 = rel_hand()
        slip_tip = max(slip_tip, float(np.linalg.norm(r0 - rel0[0])) * 1000)
        slip_end = max(slip_end, float(np.linalg.norm(r1 - rel0[1])) * 1000)
        f, dp = forces({sb}, {mb})
        sm_force, sm_deep = max(sm_force, f), max(sm_deep, dp)
        hand_mug_f = max(hand_mug_f, forces(HR, {mb})[0])
        ap = np.abs(s.motors.applied(d)) / s.motors.limit
        wr_peak = max(wr_peak, float(ap[wr].max()))
        fing_peak = max(fing_peak, float(ap[fing].max()))
    cmd = centre.copy()
    cmd[:2] = c_cmd
    s.wbc.set_ee_target(br, cmd, quat=QH)
    p.hold(0.2)
    s.wbc.weights["ee"] = ee_w0

    def circ(tr, full_only=True):
        tr = np.array(tr)
        rads = np.linalg.norm(tr, axis=1)
        ang = np.unwrap(np.arctan2(tr[:, 1], tr[:, 0]))
        sel = slice(int(len(tr) * 0.1), int(len(tr) * 0.9)) if full_only else slice(None)
        return dict(median_radius_mm=round(float(np.median(rads[sel])) * 1000, 2), max_radius_mm=round(float(rads.max()) * 1000, 2),
                    turns=round(float(abs(ang[-1] - ang[0])) / (2 * math.pi), 2))

    out["stir"] = dict(turns_cmd=turns, seconds_per_turn=period, radius_cmd_mm=rad * 1000, final_ff_gain=round(gain, 2),
                       seconds=round(d.time - t_stir, 2), bowl_centre=circ(track), spoon_tip=circ(track_tip),
                       handle_end=circ(track_end),
                       tilt_deg=dict(at_start=round(tilt_start, 1), mean=round(float(np.mean(tilts)), 1),
                                     std=round(float(np.std(tilts)), 2), max=round(float(np.max(tilts)), 1)),
                       slip_tip_mm=round(slip_tip, 2), slip_end_mm=round(slip_end, 2),
                       spoon_mug_peak_force_N=round(sm_force, 2), spoon_mug_deepest_mm=round(sm_deep, 2),
                       right_hand_mug_force_N=round(hand_mug_f, 2),
                       right_wrist_peak_load_pct=round(wr_peak * 100, 1), right_finger_peak_load_pct=round(fing_peak * 100, 1),
                       mug_moved_mm=round(float(np.linalg.norm(d.xpos[mb] - hand_mug0)) * 1000, 2))
    p.note("stirred", **out["stir"])
    p.set_tool(R, pinch_tool)                                 # back to the pinch point for the release

    # ---------------------------------------------------------------- 6. leave it in the mug, let go
    # Simplest release (no tipping, which dragged the free mug 10-95 mm in stir18-24): with the hand
    # orientation still weighted 100, the bowl goes to the mug floor near the back wall, the fingers open and
    # the spoon leans against the rim by itself; the hand goes up and back without touching the mug, and only
    # then the orientation weight eases back.
    def bowl_down():
        """The bowl rests on the mug's FLOOR: a spoon-mug contact with a vertical normal carrying > 0.08 N."""
        buf = np.zeros(6)
        for i in range(d.ncon):
            cc = d.contact[i]
            b1, b2 = int(m.geom_bodyid[cc.geom1]), int(m.geom_bodyid[cc.geom2])
            if {b1, b2} == {sb, mb} and abs(cc.frame[2]) > 0.8:
                mujoco.mj_contactForce(m, d, i, buf)
                if abs(buf[0]) > 0.08:
                    return True
        return False

    def on_rim():
        for cc in d.contact[:d.ncon]:
            g1, g2 = cc.geom1, cc.geom2
            if (g1 in spoon_geoms and int(m.geom_bodyid[g2]) == mb) or (g2 in spoon_geoms and int(m.geom_bodyid[g1]) == mb):
                if cc.pos[2] > mc[2] + 0.05:
                    return True
        return False

    mc = d.xpos[mb].copy()
    # bowl ~1 cm toward the back wall (its thickness, +-4 mm, lies along the robot's forward here; the
    # cavity's inner faces are 20.9 mm from the axis)
    e = np.r_[(mc + kw.get("back_mm", 5.0) / 1000.0 * FWD - sp(BOWL_C))[:2], 0.0]
    p.move_hand(R, s.wbc.ee_goal[br] + e, quat=QH, seconds=0.6)
    p.hold(0.15)
    # stop at the FIRST touch of the mug (floor or wall), never press: stir25 waited for a floor contact while
    # the tilted bowl met the wall, pressed ~2 cm on, and the pressed spoon sprang when let go (mug +2.6 mm)
    went = p.lower_until(R, lambda: forces({sb}, {mb})[0] > 0.1, speed=0.01, max_dist=0.035, quat=QH, ease_out=0.1)
    p.hold(0.15)
    p.move_hand(R, s.wbc.ee_pos(d, br), quat=s.wbc.ee_rot[br].copy(), seconds=0.3)   # unload: hold where it is
    mug_before_open = d.xpos[mb].copy()
    p.finger_pose(R, {"thumb_0": 0.5, "thumb_1": 0.3, "thumb_2": 0.3, "index": 0.0, "middle": 0.0}, 0.6)
    p.hold(0.6)                                               # the spoon settles against the rim by itself
    rim_hit = {"v": on_rim()}
    tilt_leveled = tilt()
    spoon_rest = axis_pt(0.0).copy()
    # the open hand: up first (the middle finger sits near the rim's height), then back toward the body
    p.move_hand(R, s.wbc.ee_goal[br] + 0.05 * UP, quat=QH, seconds=0.6)
    p.move_hand(R, s.wbc.ee_goal[br] - 0.06 * FWD + 0.02 * RIGHT + 0.02 * UP, quat=QH, seconds=0.6)
    p.set_tool(R, pinch_tool)
    nramp = int(round(0.8 / p.mv.TICK))
    for k in range(1, nramp + 1):
        s.wbc.weights["ee_rot"] = rot_w0 + (kw.get("rot_w", 100.0) - rot_w0) * (1.0 - smooth(k / nramp))
        p.mv._tick(p.mv.TICK)
    s.wbc.weights["ee_rot"] = rot_w0
    p.move_hand(R, np.r_[(d.qpos[:2] + 0.14 * RIGHT[:2] + 0.20 * FWD[:2]), 1.02], quat=QH, seconds=1.0)
    # the left hand last: open, straight up first, then out and back
    p.grip(L, 0.0, 0.4)
    p.hold(0.2)
    p.move_hand(L, s.wbc.ee_goal[bl] - 0.005 * RIGHT + 0.05 * UP, seconds=0.5)
    p.move_hand(L, s.wbc.ee_goal[bl] - 0.05 * RIGHT - 0.02 * FWD + 0.01 * UP, seconds=0.5)
    p.move_hand(L, s.wbc.ee_goal[bl] - 0.03 * RIGHT - 0.10 * FWD + 0.04 * UP, seconds=0.7)
    p.hold(0.4)
    out["release"] = dict(lowered_mm=round(went * 1000, 1), bowl_on_floor=bool(bowl_down()),
                          mug_moved_while_opening_mm=round(float(np.linalg.norm(d.xpos[mb] - mug_before_open)) * 1000, 1))
    inside = float(np.linalg.norm((axis_pt(0.0) - d.xpos[mb])[:2])) < 0.03 and axis_pt(0.0)[2] < d.xpos[mb][2] + 0.03
    out["leave"] = dict(rim_contact=rim_hit["v"], spoon_tip_in_mug=bool(inside), tilt_deg=round(tilt(), 1),
                        tilt_after_opening_deg=round(tilt_leveled, 1),
                        spoon_moved_after_release_mm=round(float(np.linalg.norm(axis_pt(0.0) - spoon_rest)) * 1000, 1),
                        mug_moved_total_mm=round(float(np.linalg.norm(d.xpos[mb] - mug0)) * 1000, 1))
    p.note("left in the mug", **out["leave"])
    out["seconds"] = round(d.time - t_start, 2)
    st = out.get("stir", {})
    out["ok"] = bool(s.checks.fell_at is None and st.get("slip_tip_mm", 1e9) <= 5.0 and st.get("slip_end_mm", 1e9) <= 5.0
                     and st.get("spoon_mug_deepest_mm", 1e9) <= 2.0 and out["leave"]["rim_contact"]
                     and out["leave"]["spoon_tip_in_mug"] and out["leave"]["mug_moved_total_mm"] <= 5.0
                     and s.checks.max_penetration_mm <= 2.0
                     and st.get("bowl_centre", {}).get("turns", 0) >= 4.0
                     and st.get("bowl_centre", {}).get("median_radius_mm", 0) >= 7.0
                     and st.get("bowl_centre", {}).get("max_radius_mm", 99) <= 14.0)
    return out
