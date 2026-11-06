"""Live helpers for the pizza task. Reloaded in the open sim with importlib.reload(pz), so they change without a restart. Debug side only."""
import math
import os

import numpy as np
from PIL import Image
from pxr import Gf, UsdGeom, UsdShade

import pizza_kitchen as Z

if "VID" not in globals():
    VID = {}
    HEAT = {"k": 0, "on": 0}

OUT = "D:/tea_robot/stretch/previews/pz"
W, H = 800, 450
if "VIEWS" not in globals():
    VIEWS = {"wide": (58.0, (.35, .85, 1.55), (1.40, 2.30, .90)),
             "front": (34.0, (.95, 1.25, 1.18), (1.25, 2.20, 1.02)),
             "close": (26.0, (1.75, 1.45, 1.20), (1.33, 2.15, 1.02)),
             "plate": (22.0, (1.35, 1.72, 1.22), (1.67, 2.22, .885))}
    R = {}


def set_cam(st, path, hfov, eye, tgt):
    c = UsdGeom.Camera.Define(st, path)
    c.CreateHorizontalApertureAttr(36.0)
    c.CreateVerticalApertureAttr(36.0 * H / W)
    c.CreateClippingRangeAttr(Gf.Vec2f(.01, 100))
    c.CreateFocalLengthAttr(float(36.0 / (2 * math.tan(math.radians(hfov) / 2))))
    x = UsdGeom.Xformable(c)
    x.ClearXformOpOrder()
    x.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1)).GetInverse())


def hand_view(s):
    g = s.link("link_grasp_center")[:3]
    return 45.0, (g[0] - .35, g[1] - .45, g[2] + .35), g


def render_setup(s):
    import omni.replicator.core as rep
    os.makedirs(OUT, exist_ok=True)
    rep.orchestrator.set_capture_on_play(False)
    R["rep"] = rep
    R["ann"] = {}
    for k, v in VIEWS.items():
        v = hand_view(s) if v is None else v
        set_cam(s.stage, "/World/cam_" + k, *v)
        rp = rep.create.render_product("/World/cam_" + k, (W, H))
        a = rep.AnnotatorRegistry.get_annotator("rgb")
        a.attach([rp])
        R["ann"][k] = a
    R["n"] = 0


def view(name, hfov, eye, tgt):
    """Replace one camera for the next renders. tgt None on the hand view goes back to following the hand."""
    VIEWS[name] = None if eye is None else (hfov, eye, tgt)


def shot(s, label="", views=None, spp=3):
    """Render the current state and write one sheet. Returns its path."""
    if VID:                                  # final take: only the video camera exists
        import shutil
        vid_frame()
        p = f"{OUT}/{VID['dir'].rsplit('/', 1)[-1]}_{label}.jpg"
        shutil.copy(f"{VID['dir']}/{VID['n'] - 1:06d}.jpg", p)
        print(p)
        return p
    rep = R["rep"]
    for k, c in VIEWS.items():
        set_cam(s.stage, "/World/cam_" + k, *(hand_view(s) if c is None else c))
    for _ in range(spp):                     # captureOnPlay off, or this plays the timeline and kills the simulation view (pz_live01)
        rep.orchestrator.step(rt_subframes=8, pause_timeline=False)
    names = views or list(R["ann"])
    ims = [Image.fromarray(np.asarray(R["ann"][k].get_data())[:, :, :3]) for k in names]
    cols = 2 if len(ims) > 1 else 1
    rows = (len(ims) + cols - 1) // cols
    S = Image.new("RGB", (cols * W, rows * H))
    for i, im in enumerate(ims):
        S.paste(im, ((i % cols) * W, (i // cols) * H))
    R["n"] += 1
    p = f"{OUT}/{R['n']:03d}_{label}.jpg"
    S.save(p, quality=82)
    print(p, "t", round(s.t, 2), "mean", round(float(np.asarray(ims[0]).mean()), 1))
    return p


def oven_setup(s, door_k=1.0, door_c=.8, door_d=.3, kn_f=.06, kn_d=.01):
    """Door: spring hinge, friction and damping, so it stays where it is left. Knobs: friction only. Values are authored, not measured from a real oven."""
    a = s.sv.create_articulation_view(Z.BODY)
    s.oven = a
    names = list(a.shared_metatype.dof_names)
    print("oven dofs", names, flush=True)
    n = a.max_dofs
    idx = np.arange(1, dtype=np.uint32)
    k, d, f = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, nm in enumerate(names):
        if nm.startswith("door"):
            k[i], d[i], f[i] = door_k, door_d, door_c
        else:
            d[i], f[i] = kn_d, kn_f
    a.set_dof_stiffnesses(np.asarray(k, np.float32)[None], idx)
    a.set_dof_dampings(np.asarray(d, np.float32)[None], idx)
    a.set_dof_max_forces(np.full((1, n), 1000., np.float32), idx)
    fp = np.array(a.get_dof_friction_properties())
    fp[0, :, 0], fp[0, :, 1] = f, f
    a.set_dof_friction_properties(fp, idx)
    a.set_dof_position_targets(np.zeros((1, n), np.float32), idx)
    s.oven_names = names


def oven_q(s):
    q = s.oven.get_dof_positions()[0]
    return {n: round(float(math.degrees(v)), 2) for n, v in zip(s.oven_names, q)}


def st(s):
    """Numbers to read at a stop."""
    r = s.report()
    r = {k: r[k] for k in ("t", "base", "yaw", "tilt", "lift", "yaw_w", "pitch", "roll", "slide", "grasp")}
    r["ext"] = round(float(s.q()[s.ix["joint_arm_l3"]]) * 4, 4)
    if hasattr(Z, "PLATE"):
        pl, pi = s.obj(Z.PLATE), s.obj(Z.PIZZA)
        r["plate"], r["pizza"] = np.round(pl[:3], 4).tolist(), np.round(pi[:3], 4).tolist()
        r["pizza_on_plate_mm"] = round(float(np.hypot(*(pi[:2] - pl[:2]))) * 1000, 1)
    r["oven"] = oven_q(s)
    r["oven_body"] = np.round(s.obj(Z.BODY)[:3], 4).tolist()
    return r


def heat_set(s, u):
    """Timed visual colour change, NOT thermal physics. u 0 raw to 1 baked, in six texture steps."""
    k = int(round(max(0., min(1., u)) * 5))
    HEAT["k"] = k
    UsdShade.Shader(s.stage.GetPrimAtPath("/World/Looks/pz_pizza_look/tex")).GetInput("file").Set(Z.PIZZA_TEX + f"albedo_{k}.png")
    return k


def turn_to(s, yaw_deg, wmax=.35, tol=.6, timeout=30.0):
    """Turn on the spot to a heading. Closed loop on the base yaw read from the sim."""
    from stretch_core import yaw_of, DT
    y = math.radians(yaw_deg)
    for i in range(int(timeout / DT)):
        if i % 6 == 0:
            e = (y - yaw_of(s.pose()[1]) + math.pi) % (2 * math.pi) - math.pi
            if abs(e) < math.radians(tol):
                break
            w = max(-wmax, min(wmax, 1.8 * e))
            if abs(w) < .05:
                w = math.copysign(.05, e)
            s.wheels(0., w)
        s.step()
    s.wheels(0., 0.)
    s.wait(.5)
    return round(math.degrees(yaw_of(s.pose()[1])), 2)


def drive_y(s, y, yaw_deg=90.0, vmax=.12, tol=.004, timeout=40.0):
    """Base at heading +y (yaw 90), drive to a world y. Holds the heading."""
    from stretch_core import yaw_of, DT
    h = math.radians(yaw_deg)
    for i in range(int(timeout / DT)):
        if i % 6 == 0:
            p, q = s.pose()
            e = y - p[1]
            if abs(e) < tol:
                break
            v = max(-vmax, min(vmax, 1.2 * e))
            if abs(v) < .012:
                v = math.copysign(.012, e)
            s.wheels(v, max(-.4, min(.4, 3.0 * ((h - yaw_of(q) + math.pi) % (2 * math.pi) - math.pi))))
        s.step()
    s.wheels(0., 0.)
    s.wait(.5)
    return np.round(s.pose()[0], 3).tolist()


def door_arc(a_deg, off=(.006, 0.)):
    """Hand point for the door at angle a: bar centre on its circle about the hinge, plus the offset seen at the grip."""
    import pizza_oven as O
    hy, hz = O.FY + O.HY, O.CZ + O.HZ
    by, bz = O.BAR_Y - O.HY, O.BAR_Z - O.HZ
    a = math.radians(a_deg)
    return hy + by * math.cos(a) - bz * math.sin(a) + off[0], hz + by * math.sin(a) + bz * math.cos(a) + off[1]


def door_step(s, a_to, x=1.19, step=3.0, secs=1.0, off=(.006, 0.)):
    """Walk the hand along the door's arc from the angle now read, step by step. Stops if the door stops following."""
    import pizza_oven as O
    a = oven_q(s)["door_j"]
    n = max(1, int(abs(a_to - a) / step + .999))
    for i in range(n):
        t = a + (a_to - a) * (i + 1) / n
        y, z = door_arc(t, off)
        s.hand_to([x, y, z], secs, base=False, settle=.3)
        d = oven_q(s)["door_j"]
        if abs(d - t) > 8.0:
            print("door not following", round(t, 1), d)
            break
    return oven_q(s)["door_j"]


if "LOG" not in globals():
    LOG = []


def snap_row(s):
    q = s.q()
    pl, pi = s.obj(Z.PLATE), s.obj(Z.PIZZA)
    return dict(pl=np.round(pl[:3], 4).tolist(), off=round(float(np.linalg.norm(pi[:3] - pl[:3] - [0, 0, Z.WELL_Z])) * 1000, 1), t=round(s.t, 2), g=np.round(s.link("link_grasp_center")[:3], 4).tolist(), base=np.round(s.pose()[0], 3).tolist(), tilt=round(s.report()["tilt"], 2), lift=round(float(q[s.ix["joint_lift"]]), 4),
                ext=round(float(q[s.ix["joint_arm_l3"]]) * 4, 4), pitch=round(float(q[s.ix["joint_wrist_pitch"]]), 3), slide=round(float(q[s.ix["joint_gripper_slide"]]), 4), door=oven_q(s)["door_j"])


def move(s, target, secs, base=False, chunk=.25, tilt_max=1.2, z_lag=.012, y_lag=.012):
    """Guarded hand_to: small pieces along a line, a trace row each, and a stop as soon as the base tilts or the hand stops following in z. Contact on the way down stops the descent."""
    g0 = s.link("link_grasp_center")[:3].copy()
    t = np.asarray(target, float)
    n = max(1, int(secs / chunk))
    t0 = s.report()["tilt"]
    for i in range(n):
        w = g0 + (t - g0) * (i + 1) / n
        s.hand_to(w, chunk, base=base, settle=0.)
        r = snap_row(s)
        LOG.append(r)
        if r["tilt"] - t0 > tilt_max or abs(r["g"][2] - w[2]) > z_lag or abs(r["g"][1] - w[1]) > y_lag:
            s.wheels(0., 0.)
            print("STOPPED", i, "of", n, r, "wanted", np.round(w, 3).tolist())
            return False
    return True


def save_log(tag):
    import json
    json.dump(LOG, open(f"D:/tea_robot/stretch/logs/{tag}_trace.json", "w"))


def rd(s):
    """Plate and pizza as seen in the plate's own frame."""
    from stretch_core import rmat
    r = snap_row(s)
    P, Q = s.obj(Z.PLATE), s.obj(Z.PIZZA)
    R_ = rmat(P[3:7])
    return dict(pl=r["pl"], ptilt=round(math.degrees(math.acos(min(1, R_[2, 2]))), 2), loc=np.round(R_.T @ (Q[:3] - P[:3]) * 1000, 1).tolist(), pitch=r["pitch"], g=r["g"], base=r["base"], door=r["door"])


def yaw_to(s, y, secs, tol=.06):
    """Guarded wrist yaw: small pieces, stops if the joint lags its target, which means something is in the way."""
    I = s.ix["joint_wrist_yaw"]
    y0 = float(s.qd[I])
    n = max(1, int(secs / .25))
    for i in range(n):
        s.ramp({"joint_wrist_yaw": y0 + (y - y0) * (i + 1) / n}, .25)
        if abs(float(s.q()[I]) - float(s.qd[I])) > tol:
            s.qd[I] = float(s.q()[I])
            print("YAW STOPPED", i, "of", n, snap_row(s))
            return False
    return True


def drive(s, x, line_y, vmax=.08, chunk=.5, heading=180.0, tol=.004, timeout=40.0):
    """Guarded drive along x: short pieces, stops on base tilt or heading error."""
    from stretch_core import yaw_of
    for i in range(int(timeout / chunk)):
        p, q = s.pose()
        if abs(p[0] - x) < tol:
            break
        step = max(-vmax * chunk, min(vmax * chunk, x - p[0]))
        s.drive_x(p[0] + step, line_y=line_y, vmax=vmax, timeout=chunk * 3)
        yaw = math.degrees(yaw_of(s.pose()[1]))
        if s.report()["tilt"] > 1.0 or abs(((yaw - heading + 180) % 360) - 180) > 4.0:
            s.wheels(0., 0.)
            print("DRIVE STOPPED", snap_row(s), "yaw", round(yaw, 1))
            return False
    return True




def rep_setup():
    import omni.replicator.core as rep
    rep.orchestrator.set_capture_on_play(False)


def vid_on(s, tag, every=200, size=(1280, 720)):
    """Final take: one render product, one frame every `every` physics steps, written as it runs. Frames only; nothing here touches the sim."""
    import omni.replicator.core as rep
    d = f"E:/stretch_assets/video/{tag}"
    os.makedirs(d, exist_ok=True)
    set_cam(s.stage, "/World/cam_vid", 40.0, (1., 1.2, 1.3), (1.3, 2.2, 1.))
    rp = rep.create.render_product("/World/cam_vid", size)
    a = rep.AnnotatorRegistry.get_annotator("rgb")
    a.attach([rp])
    VID.update(dir=d, a=a, n=0, c=0, every=every, rep=rep, cam=None, st=s.stage)
    orig = s.step

    def step():
        orig()
        VID["c"] += 1
        if VID["c"] % VID["every"] == 0:
            vid_frame()

    s.step = step


def vid_cam(hfov, eye, tgt):
    if VID:
        set_cam(VID["st"], "/World/cam_vid", hfov, eye, tgt)


def vid_frame():
    VID["rep"].orchestrator.step(rt_subframes=3, pause_timeline=False)
    im = np.asarray(VID["a"].get_data())
    if im.ndim == 3:
        Image.fromarray(im[:, :, :3]).save(f"{VID['dir']}/{VID['n']:06d}.jpg", quality=88)
        VID["n"] += 1
