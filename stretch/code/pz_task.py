"""The pizza task, written from what worked live (pz_kit10 to pz_kit13). AI-operated. Every move is guarded; a failed guard stops the run where it is, sim kept open.
No welds, teleports or forces on objects. Heating is a timed colour change, not thermal physics."""
import math

import numpy as np

import pizza_kitchen as K
import pizza_oven as O
import pz

LINE_C, LINE_D = 1.675, 1.40        # counter line, door line
HX = 1.19                          # hand x at the door and rack
HEAD_TILT = -.60
GRIP_Z = .037                      # pad above the plate base: the inner strips then pass over the rim, and the plate hangs under the pinch
CARRY = 1.056
S = {}


class Stop(Exception):
    pass


def ok(v, what):
    if v is False:
        raise Stop(what)


def stage(s, name, views=("close", "front")):
    r = pz.snap_row(s)
    print("STAGE", name, dict(t=r["t"], g=r["g"], base=r["base"], tilt=s.report()["tilt"], door=r["door"], plate=r["pl"], rd=pz.rd(s)["loc"], knobs=pz.oven_q(s)), flush=True)
    pz.save_log(S["tag"])
    if pz.VID or pz.R.get("ann"):
        S["path"] = pz.shot(s, name, views=list(views))


def cams(a, b=None, c=None):
    pz.view("close", *a)
    if b:
        pz.view("front", *b)


def counter_cam(x):
    cams((28.0, (x - .36, 1.62, 1.14), (x, 2.1, .92)), (28.0, (x + .34, 1.62, 1.12), (x, 2.1, .92)))


def oven_cam():
    cams((40.0, (1.72, 1.62, 1.32), (1.19, 2.12, 1.03)), (40.0, (.70, 1.55, 1.30), (1.19, 2.12, 1.03)))


def pick(s, x, y):
    """Hand already over the plate at z .97. Guarded descent, pinch the rim, lift in small steps."""
    ok(pz.move(s, [x, y, K.COUNTER_Z + GRIP_Z], 8.0, z_lag=.006, tilt_max=.3), "descent to rim")
    s.close_on(pad_n=14.0, rate=.005, timeout=10.0)
    g = s.link("link_grasp_center")[:3].copy()
    for dz in (.0045, .009, .013, .019, .026, .036):
        ok(pz.move(s, [g[0], g[1], g[2] + dz], 1.0, z_lag=.02, tilt_max=.6, chunk=.1), "first lift")
    ok(pz.move(s, [g[0], g[1], CARRY], 6.0, z_lag=.02, tilt_max=.6), "lift to carry height")


def put_down(s, stop_z, z_near, floor):
    g = s.link("link_grasp_center")[:3].copy()
    ok(pz.move(s, [g[0], g[1], z_near], 7.0, z_lag=.012, tilt_max=.6), "lower")
    z = z_near
    off = s.link("link_grasp_center")[2] - s.obj(K.PLATE)[2]
    while s.obj(K.PLATE)[2] > stop_z and z > floor:
        z -= .002
        ok(pz.move(s, [g[0], g[1], z], .5, z_lag=.008, tilt_max=.4, chunk=.1), "lower to contact")
        if off - (s.link("link_grasp_center")[2] - s.obj(K.PLATE)[2]) > .002:      # plate has stopped going down with the hand: first contact, it hangs tilted so one edge touches first
            break
    I = s.ix["joint_gripper_slide"]
    s.ramp({"joint_gripper_slide": float(s.q()[I])}, 2.0)
    s.ramp({"joint_gripper_slide": .04}, 2.0)
    s.wait(.6)


def to_line(s, y, heading_back=180.0):
    """Base turn, drive along y, turn back. Hand folded. Slow, proven with the plate in hand at x 1.17."""
    pz.turn_to(s, 90.0, wmax=.2, timeout=20.0)
    pz.drive_y(s, y, vmax=.08)
    pz.turn_to(s, heading_back, wmax=.2, timeout=20.0)
    if s.report()["tilt"] > 1.0 or abs(((math.degrees(pz_yaw(s)) - 180 + 180) % 360) - 180) > 3.0:
        raise Stop("turn or drive")


def pz_yaw(s):
    from stretch_core import yaw_of
    return yaw_of(s.pose()[1])


def door_open(s):
    s.ramp({"joint_gripper_slide": .04}, 1.5)
    s.ramp({"joint_wrist_roll": math.pi / 2}, 3.0)
    ok(pz.move(s, [HX, 1.97, 1.147], 6.0, base=True, z_lag=.03, y_lag=.06, tilt_max=.5), "to the bar")
    s.hand_to([HX, 1.97, 1.147], 2.0, tol=.0015)
    ok(pz.move(s, [HX, 2.046, 1.147], 3.0, z_lag=.02), "onto the bar")
    s.close_on(pad_n=15.0, rate=.006, timeout=9.0)
    s.ramp({"joint_gripper_slide": -.016}, 1.5)
    s.wait(.3)
    for a in (10, 20, 30, 40):
        d = pz.door_step(s, float(a))
        if s.report()["tilt"] > 1.0 or abs(d - a) > 6.0:
            raise Stop("door pull")
    stage(s, "door_40")
    A = s.ix["joint_arm_l3"]
    s.qd[A] = float(s.q()[A])
    s.ramp({"joint_gripper_slide": .02}, 2.0)
    s.wait(.8)
    g = s.link("link_grasp_center")[:3].copy()
    ok(pz.move(s, [HX, 1.817, g[2]], 3.0, z_lag=.03, y_lag=.03, tilt_max=.5), "out of the bar")
    ok(pz.move(s, [HX, 1.82, 1.16], 4.0, z_lag=.03, tilt_max=.5), "rise")
    s.ramp({"joint_wrist_roll": 0., "joint_gripper_slide": 0.}, 3.0)
    ok(pz.move(s, [HX, 1.90, 1.15], 3.0, z_lag=.03), "over the door")
    for z in (1.08, 1.03, 1.0, .975, .96, .945):
        ok(pz.move(s, [HX, 1.90, z], 2.0, z_lag=.015, tilt_max=.8), "press")
        if pz.oven_q(s)["door_j"] >= 86:
            break
    ok(pz.move(s, [HX, 1.90, 1.0], 2.5, z_lag=.03), "off the door")
    ok(pz.move(s, [HX, 1.90, 1.07], 2.5, z_lag=.03), "off the door")
    s.ramp({"joint_gripper_slide": .04}, 1.5)
    s.wait(.5)
    if pz.oven_q(s)["door_j"] < 86:
        raise Stop("door not flat")
    stage(s, "door_flat")


def door_close(s):
    s.ramp({"joint_gripper_slide": 0.}, 1.5)
    ok(pz.move(s, [HX, 1.818, 1.06], 2.5, z_lag=.02, y_lag=.06), "back")
    ok(pz.move(s, [HX, 1.818, .80], 7.0, z_lag=.02, tilt_max=.5), "under")
    ok(pz.move(s, [HX, 1.852, .82], 2.5, z_lag=.02, tilt_max=.5), "under the bar")
    for a in (86, 80, 74, 68, 62, 56, 50, 44, 38, 32):
        off = (0., -.036) if a > 40 else (-.012, -.034) if a > 34 else (-.02, -.03)
        y, z = pz.door_arc(float(a), off)
        lag = .02 if a > 40 else .06             # the hand rides up round the bar near 32 deg, seen every time
        ok(pz.move(s, [HX, y, z], 1.5, z_lag=lag, y_lag=max(lag, .03), tilt_max=.6), "door up")
    stage(s, "door_up")


def door_close_b(s):
    # at about 32 deg the hand rides up round the bar by itself (seen twice). Stand off in front of the bar, then push along the arc
    d = pz.oven_q(s)["door_j"]
    g = s.link("link_grasp_center")[:3].copy()
    y, z = pz.door_arc(d, (-.045, 0.))
    ok(pz.move(s, [HX, min(g[1], y - .03), g[2]], 2.0, z_lag=.08, y_lag=.08, tilt_max=.9), "back off the bar")      # pz_final05: hand rode over the bar and pressed the door top
    ok(pz.move(s, [HX, y, z], 2.5, z_lag=.08, y_lag=.08, tilt_max=.9), "in front of the bar")
    door_walk(s)
    door_push(s)


def door_pinch_close(s):
    """Close by pinching the bar and walking the opening arc backwards, the motion that opens the door. Pushing a bar that rises away from the hand slipped three times; a pinch cannot slip off."""
    ok(pz.move(s, [HX, 1.83, s.link("link_grasp_center")[2]], 3.0, z_lag=.06, y_lag=.08, tilt_max=1.0), "clear of the bar")
    s.ramp({"joint_gripper_slide": .04}, 1.5)
    d = pz.oven_q(s)["door_j"]
    y, z = pz.door_arc(d, (0., 0.))
    ok(pz.move(s, [HX, 1.83, z], 3.0, z_lag=.03, y_lag=.06, tilt_max=.6), "level with the bar")
    s.ramp({"joint_wrist_roll": math.pi / 2}, 3.0)
    d = pz.oven_q(s)["door_j"]
    y, z = pz.door_arc(d, (.006, 0.))
    ok(pz.move(s, [HX, y, z], 3.0, z_lag=.02, y_lag=.03, tilt_max=.6), "onto the bar")
    s.close_on(pad_n=15.0, rate=.006, timeout=9.0)
    s.ramp({"joint_gripper_slide": -.016}, 1.5)
    s.wait(.3)
    for a in (20, 10, 5, 2, 0):
        if a < pz.oven_q(s)["door_j"]:
            pz.door_step(s, float(a))
        if s.report()["tilt"] > 2.0:
            raise Stop("door pinch close")
    if pz.oven_q(s)["door_j"] > .05:
        for i in range(6):
            y, z = pz.door_arc(0., (.006 + .001 * (i + 1), 0.))
            ok(pz.move(s, [HX, y, z], 1.0, z_lag=.02, y_lag=.02, tilt_max=1.5), "seat the door")
            if pz.oven_q(s)["door_j"] <= .05:
                break
    A = s.ix["joint_arm_l3"]
    s.qd[A] = float(s.q()[A])
    L = s.ix["joint_lift"]
    s.ramp({"joint_gripper_slide": .02}, 2.0)
    s.qd[L] = min(float(s.qd[L]), float(s.q()[L]) + .10)     # the lift target had wound up against the bar; the hand jumped up to the stop when let go
    s.wait(.5)
    g = s.link("link_grasp_center")[:3].copy()
    ok(pz.move(s, [HX, 1.96, g[2]], 3.0, z_lag=.03, y_lag=.03, tilt_max=.6), "off the bar")
    s.ramp({"joint_wrist_roll": 0.}, 3.0)
    print("STAGE door_pinched", pz.oven_q(s)["door_j"], s.report()["tilt"], flush=True)
    stage(s, "door_shut")


def door_walk(s):
    """Push the bar shut with the hand kept level with the bar's real height: targets come from the door angle read now, not a planned angle. pz_final05: with planned angles the hand rose over the bar and pressed it down."""
    for i in range(70):
        d = pz.oven_q(s)["door_j"]
        if d < 7.:
            return
        y = pz.door_arc(max(d - 3., 0.), (-.012, 0.))[0]
        z = pz.door_arc(d, (0., 0.))[1]
        ok(pz.move(s, [HX, y, z], 1.0, z_lag=.05, y_lag=.06, tilt_max=2.5), "door shut")      # pushing the door up leans the robot a little: 1.5 deg seen, 6 deg recovered by itself earlier
    raise Stop("door walk did not finish")


def door_push(s):
    g = s.link("link_grasp_center")[:3].copy()
    ok(pz.move(s, [HX, min(g[1], 1.995), 1.147], 2.0, z_lag=.03, y_lag=.03, tilt_max=.6), "to the bar")
    for y in (2.01, 2.018, 2.024, 2.03, 2.032, 2.034, 2.036, 2.038, 2.04):
        ok(pz.move(s, [HX, y, 1.147], 1.5, z_lag=.02, y_lag=.02, tilt_max=.6), "door shut")
        if pz.oven_q(s)["door_j"] < .05:
            break
    ok(pz.move(s, [HX, 1.96, 1.147], 2.5, z_lag=.02, y_lag=.03), "off the door")
    stage(s, "door_shut")


def knob(s, z):
    ok(pz.move(s, [1.44, 2.0, z], 5.0, base=True, z_lag=.03, y_lag=.06, tilt_max=.5), "to the knob")
    s.hand_to([1.44, 2.0, z], 2.0, tol=.0015)
    ok(pz.move(s, [1.44, 2.068, z], 3.0, z_lag=.02, y_lag=.03, tilt_max=.5), "onto the knob")
    s.close_on(pad_n=8.0, rate=.004, timeout=6.0)
    s.ramp({"joint_wrist_roll": 1.5}, 5.0)
    s.wait(.3)
    s.ramp({"joint_gripper_slide": .0085}, 2.0)
    s.wait(.3)
    ok(pz.move(s, [1.44, 2.0, z], 3.0, z_lag=.02, y_lag=.08, tilt_max=.5), "off the knob")
    s.ramp({"joint_wrist_roll": 0.}, 4.0)


def seg_a(s):
    s.ramp({"joint_head_pan": -math.pi / 2, "joint_head_tilt": HEAD_TILT}, 3.0)        # head turns to the arm side and looks down at the work: real joint moves
    pz.vid_cam(40.0, (.95, 1.25, 1.25), (1.67, 2.12, .95))
    counter_cam(1.67)
    stage(s, "s01_start")
    ok(pz.move(s, [1.67, 2.12, 1.0], 5.0, y_lag=.03, z_lag=.03), "over the plate")
    pick(s, 1.67, 2.12)
    stage(s, "s02_picked")
    g = s.link("link_grasp_center")[:3].copy()
    ok(pz.move(s, [g[0], 2.094, CARRY], 3.0, z_lag=.02, y_lag=.03, tilt_max=.6), "arm in")      # pz_run01: with the arm out, the unfold at x .68 met the mug rack


def seg_b(s):
    pz.vid_cam(55.0, (.10, 1.00, 1.50), (1.15, 2.05, .95))
    ok(pz.yaw_to(s, -1.35, 7.0), "fold")
    s.wait(.5)
    ok(pz.drive(s, .68, LINE_C), "carry past the oven")


def seg_c(s):
    ok(pz.yaw_to(s, 0., 7.0), "unfold")
    s.wait(.5)
    s.hand_to([.86, 2.12, CARRY], 6.0, tol=.002)
    counter_cam(.86)
    stage(s, "s03_over_set")
    put_down(s, .8625, .92, .895)
    g = s.link("link_grasp_center")[:3].copy()
    ok(pz.move(s, [g[0], g[1], 1.0], 5.0, z_lag=.02, tilt_max=.6), "rise")
    stage(s, "s04_set_down")


def seg_d(s):
    pz.vid_cam(50.0, (.45, 1.05, 1.50), (1.15, 1.95, .98))
    g = s.link("link_grasp_center")[:3].copy()
    ok(pz.move(s, [g[0], 2.094, 1.0], 3.0, z_lag=.02, y_lag=.03, tilt_max=.6), "arm in")
    ok(pz.drive(s, .68, LINE_C), "left of the plate")
    ok(pz.yaw_to(s, -1.35, 5.0), "fold")
    ok(pz.drive(s, 1.169, LINE_C, vmax=.1), "to the oven")
    to_line(s, LINE_D)
    ok(pz.yaw_to(s, 0., 5.0), "unfold")
    oven_cam()
    door_open(s)


def seg_e(s):
    ok(pz.yaw_to(s, -1.35, 4.0), "fold")
    to_line(s, LINE_C)
    s.set_ext(0., 3.0)                # pz_run01: with the arm out, the unfold met the mug rack
    ok(pz.drive(s, .68, LINE_C, vmax=.1), "to the plate")


def seg_i(s):
    pz.vid_cam(48.0, (.15, 1.10, 1.45), (.95, 2.05, .95))
    ok(pz.yaw_to(s, 0., 5.0), "unfold")
    p = s.obj(K.PLATE)[:2].copy()
    ok(pz.move(s, [p[0], p[1], .97], 6.0, base=True, z_lag=.03, y_lag=.06), "over the plate")
    s.hand_to([p[0], p[1], .97], 3.0, tol=.0015)
    pick(s, p[0], p[1])
    counter_cam(.86)
    stage(s, "s05_repicked")
    g = s.link("link_grasp_center")[:3].copy()
    ok(pz.move(s, [g[0], 2.094, CARRY], 3.0, z_lag=.02, y_lag=.03, tilt_max=.6), "arm in")
    ok(pz.drive(s, .68, LINE_C), "left of the rack")
    ok(pz.yaw_to(s, -1.35, 7.0), "fold")
    s.wait(.5)
    ok(pz.drive(s, 1.169, LINE_C), "to the oven")
    to_line(s, LINE_D)
    ok(pz.yaw_to(s, 0., 7.0), "unfold")
    s.wait(.5)
    s.hand_to([HX, 1.82, CARRY], 4.0, tol=.002)
    oven_cam()
    stage(s, "s06_facing_oven")


def seg_f(s):
    pz.vid_cam(42.0, (1.90, 1.52, 1.32), (1.22, 2.14, 1.03))
    for y in (2.02, 2.10, 2.17, 2.23, 2.28):
        ok(pz.move(s, [HX, y, CARRY], 3.0, z_lag=.015, y_lag=.03, tilt_max=.6), "into the oven")
    stage(s, "s07_inside")
    put_down(s, 1.0015, 1.05, 1.03)
    stage(s, "s08_on_rack")
    ok(pz.move(s, [HX, 2.10, 1.058], 6.0, z_lag=.02, y_lag=.03, tilt_max=.6), "hand out")
    ok(pz.move(s, [HX, 1.87, 1.058], 6.0, z_lag=.02, y_lag=.03, tilt_max=.6), "hand out")
    pz.vid_cam(48.0, (.50, 1.45, 1.40), (1.19, 1.98, 1.02))
    door_close(s)


def seg_j(s):
    door_pinch_close(s)


def seg_g(s):
    pz.vid_cam(30.0, (1.95, 1.72, 1.25), (1.44, 2.07, 1.05))
    s.ramp({"joint_gripper_slide": .0085}, 1.5)
    pz.view("close", 26.0, (1.82, 1.72, 1.25), (1.44, 2.07, 1.08))
    pz.view("front", 26.0, (1.12, 1.68, 1.28), (1.44, 2.07, 1.08))
    knob(s, O.CZ + O.KNOB_Z[0])
    stage(s, "s09_temp_knob")


def seg_k(s):
    knob(s, O.CZ + O.KNOB_Z[1])
    stage(s, "s09_knob")


def seg_l(s):
    knob(s, O.CZ + O.KNOB_Z[3])
    ok(pz.move(s, [1.44, 1.92, 1.0], 3.0, z_lag=.02, tilt_max=.5), "away")
    pz.vid_cam(24.0, (1.19, 1.60, 1.28), (1.19, 2.27, 1.03))
    pz.view("close", 30.0, (1.12, 1.25, 1.2), (1.25, 2.2, 1.0))
    pz.view("front", 22.0, (1.19, 1.60, 1.28), (1.19, 2.27, 1.03))
    stage(s, "s10_started")
    O.set_on(s, True)
    pz.HEAT["on"] = 1
    for k in range(1, 6):
        pz.heat_set(s, k / 5)
        s.wait(1.5)
        if k in (1, 3, 5):
            stage(s, f"s11_heat{k}")
    O.set_on(s, False)
    pz.HEAT["on"] = 0


def seg_h(s):
    pz.vid_cam(46.0, (1.90, 1.48, 1.34), (1.22, 2.05, 1.02))
    oven_cam()
    door_open(s)
    p = s.obj(K.PLATE)[:3].copy()
    s.hand_to([p[0], 1.90, 1.058], 3.0, tol=.0015)              # pz_final05: plate sat 9 mm off the hand's line, one pad pushed it. Square up with the base before going in: the arm cannot move in x
    for y in (1.95, 2.05, 2.12, 2.18, 2.23, p[1] - .01, p[1] + .005):
        ok(pz.move(s, [p[0], y, 1.058], 2.0, z_lag=.012, y_lag=.02, tilt_max=.5), "reach in")
    ok(pz.move(s, [p[0], p[1] + .005, p[2] + GRIP_Z], 6.0, z_lag=.006, y_lag=.01, tilt_max=.3), "to the rim")
    s.close_on(pad_n=14.0, rate=.005, timeout=10.0)
    g = s.link("link_grasp_center")[:3].copy()
    for dz in (.004, .008, .012, .016):
        ok(pz.move(s, [g[0], g[1], g[2] + dz], 1.0, z_lag=.02, tilt_max=.6, chunk=.1), "off the rack")
    for y in (2.2, 2.12, 2.04, 1.96, 1.87):
        ok(pz.move(s, [g[0], y, CARRY], 2.5, z_lag=.015, y_lag=.03, tilt_max=.6), "out of the oven")
    stage(s, "s12_out")
    pz.vid_cam(55.0, (.75, .95, 1.55), (1.50, 2.0, .95))
    ok(pz.yaw_to(s, -1.35, 7.0), "fold")
    s.wait(.5)
    to_line(s, LINE_C)
    ok(pz.drive(s, 1.65, LINE_C), "carry back")
    ok(pz.yaw_to(s, 0., 7.0), "unfold")
    s.wait(.5)
    s.hand_to([1.67, 2.12, CARRY], 4.0, tol=.002)
    counter_cam(1.67)
    stage(s, "s13_over_start")
    put_down(s, .8625, .92, .895)
    g = s.link("link_grasp_center")[:3].copy()
    ok(pz.move(s, [g[0], g[1], 1.02], 6.0, z_lag=.02, tilt_max=.6), "rise")
    stage(s, "s14_back")


SEGS = "abcdeifjgklh"


def run(s, tag, frm="a"):
    S["tag"] = tag
    for c in SEGS[SEGS.index(frm):]:
        globals()["seg_" + c](s)
    print("END status DONE", flush=True)
