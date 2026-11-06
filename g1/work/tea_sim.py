"""Tea robot, physically simulated: the twin, the G1 and the handled objects in MuJoCo physics.

Everything moves through physics: the G1 only through its torque-limited motors (byor MotorLayer),
commanded by our whole-body controller (byor WholeBodyController); the kettle, mug and jar are free
bodies with real weights and friction, held only by finger contact. Physics runs at 600 Hz, so a
60 fps video takes one frame every 10 steps, straight from the simulation state.
"""
from __future__ import annotations
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401

import os

# one math thread per simulation: the controller's matrices are small, and parallel runs would
# otherwise each start a thread per CPU core and fight over them
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

PKG = Path(str(HANDOFF))
sys.path.insert(0, str(PKG / "walking"))
from nova_robotics.byor import analyze_model, load_robot  # noqa: E402
from nova_robotics.byor.scene import prepare_environment  # noqa: E402
from nova_robotics.byor.teleop import TeleopSession  # noqa: E402

SCENE = PKG / "project" / "tea_twin" / "scene_v3"
G1_XML = PKG / "robots" / "unitree_g1" / "g1_with_hands.xml"
DT = 1.0 / 600.0                                   # 10 physics steps per 60 fps video frame

MUG, KETTLE, JAR = "object_0316", "object_0295", "object_0309"
# real weights (kg): the kettle with about 0.4 l of water in it, a ceramic mug, the glass jar of whitener
MASS = {KETTLE: 1.25, MUG: 0.30, JAR: 0.45}
import os as _os  # noqa: E402
if _os.environ.get("TEA_KETTLE_KG"):                   # experiments: kettle with less or no water
    MASS[KETTLE] = float(_os.environ["TEA_KETTLE_KG"])
if _os.environ.get("TEA_MUG_KG"):                      # the full cup at the end: mug 0.30 + ~0.18 kg of tea
    MASS[MUG] = float(_os.environ["TEA_MUG_KG"])
# eye camera: Unitree's d435 mount on torso_link, pitched down 0.8308 rad; vertical field of view of
# the wearer's iPhone 0.5x crop (f = 869 px at 720 x 1280 portrait)
EYE_POS = (0.0576, 0.0175, 0.4299)
EYE_PITCH = 0.8308
EYE_FOVY = math.degrees(2 * math.atan(640 / 869))


def fix_kettle_handle(body):
    """Geometry correction: the twin's convex pieces for the kettle handle fill most of the loop (a
    1.7 cm slot); the visual handle is a thin loop with a 3.6-4.4 cm opening that fingers go through.
    The handle collision is rebuilt from the visual mesh: vertical bar x 0.114-0.138, y +-0.011,
    z 0.07-0.18; top arm z 0.18-0.205; bottom arm z 0.035-0.07 (kettle body frame, metres)."""
    for g in list(body.findall("geom")):
        if g.get("name") in ("object_0295_collision_02", "object_0295_collision_03", "object_0295_collision_04"):
            body.remove(g)
    for name, pos, half in (("handle_bar", (0.126, 0.0, 0.125), (0.012, 0.011, 0.055)),
                            ("handle_top", (0.097, 0.0, 0.1925), (0.039, 0.010, 0.0125)),
                            ("handle_bottom", (0.100, 0.0, 0.0525), (0.030, 0.011, 0.0175))):
        ET.SubElement(body, "geom", name=f"{KETTLE}_{name}", type="box", pos=" ".join(map(str, pos)),
                      size=" ".join(map(str, half)), mass="0", group="3", rgba="0 0 0 0",
                      contype="1", conaffinity="1", condim="3", friction="0.5 0 0")


# sole-on-tile contact time constant (s): MuJoCo's default 0.02 lets heel strikes sink 6 mm; 0.005 fixed that
# but made standing under a one-arm load fall (3 of 3 kettle runs)
FOOT_TC = float(_os.environ.get("TEA_FOOT_TC", "0.01"))
def _drop_geoms(body, names):
    for g in list(body.findall("geom")):
        if g.get("name") in names:
            body.remove(g)


def _coll_box(body, name, pos, half, quat=(1, 0, 0, 0), friction="0.55 0 0"):
    ET.SubElement(body, "geom", name=name, type="box", pos=" ".join(f"{x:.5f}" for x in pos),
                  size=" ".join(f"{x:.5f}" for x in half), quat=" ".join(f"{x:.6f}" for x in quat), mass="0",
                  group="3", rgba="0 0 0 0", contype="1", conaffinity="1", condim="3", friction=friction)


def fix_mug_walls(body, oid):
    """Geometry correction: the twin's 8 wall hulls are 13 mm thick (inner face 20.9 mm from the axis);
    the real mug's wall is ~4.5 mm (inner face 29.5 mm), so a spoon or tea bag inside stopped 8 mm short
    of the wall you see. The twin's exact outside is kept - 8 flat wall pieces, each covering a side from
    its middle out to 24 mm, faces at +-34 mm, corners notched (the left-hand grasp is tuned on that shape:
    a round ring let the mug twist in the hand, full corners caught the curled finger) - only thinner."""
    _drop_geoms(body, {f"{oid}_collision_0{k}" for k in range(1, 9)})
    r_in, r_out, z0, z1, span = 0.0295, 0.034, 0.009, 0.085, 0.024
    rc, ht, hz, hs = (r_in + r_out) / 2, (r_out - r_in) / 2, (z1 - z0) / 2, span / 2
    k = 0
    for axis in (0, 1):
        for side in (-1, 1):
            for half_side in (-1, 1):
                pos, half = [0.0, 0.0, z0 + hz], [0.0, 0.0, hz]
                pos[axis], half[axis] = side * rc, ht
                pos[1 - axis], half[1 - axis] = half_side * hs, hs
                _coll_box(body, f"{oid}_wall_{k}", pos, half)
                k += 1


def fix_tub(body, oid, hx, hy, h=0.045, t=0.002):
    """Geometry correction: the tray's tubs were solid blocks (a spoon would stand on top, 4 cm above the
    floor you see); open tubs: a floor and four 2 mm walls."""
    _drop_geoms(body, {f"{oid}_collision_00"})
    _coll_box(body, f"{oid}_floor", (0, 0, t / 2), (hx, hy, t / 2), friction="0.5 0 0")
    for nm, pos, half in (("wx0", (-hx + t / 2, 0, h / 2), (t / 2, hy, h / 2)), ("wx1", (hx - t / 2, 0, h / 2), (t / 2, hy, h / 2)),
                          ("wy0", (0, -hy + t / 2, h / 2), (hx, t / 2, h / 2)), ("wy1", (0, hy - t / 2, h / 2), (hx, t / 2, h / 2))):
        _coll_box(body, f"{oid}_{nm}", pos, half, friction="0.5 0 0")


SWITCH = KETTLE + "_switch"                        # slide joint of the kettle's on/off lever
SWITCH_ON = -0.0035                                # pressed past 3.5 mm: on (blue light)
SWITCH_LEVER = ((0.140, 0.0, 0.030), (0.035, 0.012, 0.004))   # lever centre and half-size, kettle frame


def add_kettle_switch(body):
    """The on/off lever at the handle's foot as a part of its own: a slide joint along the kettle's
    z (0 to -5 mm) held by 2.5 N of friction, so it takes a real push to move and stays pressed,
    like a toggle. It sticks out 3.7 cm past the handle bar, where a fingertip can press it from above
    (the render adds the same lever, SWITCH_LEVER, to the twin's kettle mesh)."""
    sw = ET.SubElement(body, "body", name=f"{SWITCH}_body", pos=" ".join(map(str, SWITCH_LEVER[0])))
    # Stiffness given directly (negative solref = N/m, N s/m): MuJoCo's default softness scales with the
    # lighter body's mass, so a 5 g lever let a fingertip sink 2 mm into it at well under 2.5 N, and its
    # end stop gave way under load. Hard plastic: 20 N per mm; the stop at -5 mm is as stiff.
    ET.SubElement(sw, "joint", name=SWITCH, type="slide", axis="0 0 1", range="-0.005 0", limited="true",
                  frictionloss="2.5", damping="0.5", armature="0.1", solreflimit="-30000 -200")   # 0.1 kg: the switch linkage; keeps both springs 8+ steps per period
    ET.SubElement(sw, "geom", name=f"{SWITCH}_lever", type="box", size=" ".join(map(str, SWITCH_LEVER[1])),
                  mass="0.005", group="3", rgba="0 0 0 0", contype="1", conaffinity="1", condim="3", friction="0.8 0 0",
                  solref="-20000 -150", priority="1")


def corrected_environment(extras=(), poses=None) -> Path:
    """The twin with the scene corrections (objects where the video has them) and real object
    weights, written next to environment.xml so its mesh paths still resolve.
    `extras`: prop modules, each with add(root, bodies) putting its bodies (named prop_*) into the
    environment XML and ACTIVE (its joint names). `poses`: {object id: (x, y, z[, qw, qx, qy, qz])}
    start-pose overrides, e.g. the mug already where an earlier step of the flow left it."""
    src = SCENE / "environment.xml"
    root = ET.parse(src).getroot()
    corr = {k: v for k, v in json.loads((SCENE / "corrections.json").read_text()).items() if k.startswith("object")}
    for oid, pose in (poses or {}).items():
        pose = [float(x) for x in pose]
        corr[oid] = pose if len(pose) == 7 else pose[:3] + [1.0, 0.0, 0.0, 0.0]
    bodies = {b.get("name"): b for b in root.iter("body")}
    for oid, pose in corr.items():
        b = bodies.get(oid + "_body")
        if b is None:
            continue
        b.set("pos", " ".join(f"{x:.6f}" for x in pose[:3]))
        b.set("quat", " ".join(f"{x:.6f}" for x in pose[3:7]))
    for oid, mass in MASS.items():
        b = bodies[oid + "_body"]
        inertial = b.find("inertial")
        old = float(inertial.get("mass"))
        inertial.set("mass", f"{mass}")
        diag = np.fromstring(inertial.get("diaginertia"), sep=" ") * (mass / old)
        inertial.set("diaginertia", " ".join(f"{x:.8f}" for x in diag))
        if oid == KETTLE:                              # water sits low in the kettle
            inertial.set("pos", "0.0 0 0.075")
    fix_kettle_handle(bodies[KETTLE + "_body"])
    add_kettle_switch(bodies[KETTLE + "_body"])
    if _os.environ.get("TEA_THIN_MUG"):
        # off by default: with real thin walls the curled finger hooks the rim and the squeeze sinks 2.9 mm
        # into the wall; step 1's grasp passes on the twin's thick walls (cavity 8 mm too small - noted)
        fix_mug_walls(bodies[MUG + "_body"], MUG)
    fix_tub(bodies["object_0306_body"], "object_0306", 0.048, 0.0363, h=0.090)   # tray tubs FRESH (90 mm deep,
                                                                                 # as the real ones) and USED
    fix_tub(bodies["object_0307_body"], "object_0307", 0.047, 0.0363)
    _drop_geoms(bodies["object_0310_body"], {"object_0310_collision_06"})  # the rack has no roof board
    for ex in extras:
        ex.add(root, bodies)
    out = SCENE / f"environment_physics_{_os.getpid()}.xml"   # one per process: parallel runs must not share it
    ET.ElementTree(root).write(out, encoding="unicode")
    return out


def build(robot_xy=(2.50, 1.90), robot_yaw_deg=88.0, active=(KETTLE, MUG, JAR), extras=(), poses=None):
    loaded = load_robot(G1_XML)
    report = analyze_model(loaded.model)
    spec = loaded.spec
    # the maker's standing pose (arms hanging: shoulder pitch 0.2, roll +-0.2, elbow 1.28); with every
    # joint at zero the elbows sit at 90 degrees and the forearms reach forward into the counter
    rm = loaded.model
    stand = {rm.joint(j).name: float(rm.key_qpos[0][rm.jnt_qposadr[j]]) for j in range(rm.njnt)
             if int(rm.jnt_type[j]) in (2, 3)} if rm.nkey else {}
    for key in list(spec.keys):
        spec.delete(key)
    # Dex3 finger and palm pads are firm silicone, the soles firm rubber: contact impedance rising to
    # 0.99 within 1 mm (MuJoCo's default 0.95 lets a hard squeeze sink 2-3 mm into a handle and a heel
    # strike 6 mm into the floor)
    for body in spec.bodies:
        if "_hand_" in body.name or "ankle_roll" in body.name:
            for g in body.geoms:
                g.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
                if "ankle_roll" in body.name and not _os.environ.get("TEA_SOFT_FEET"):
                    g.solref = [FOOT_TC, 1.0]          # rubber sole on tile
    root = spec.body("pelvis")
    root.pos[:2] = robot_xy
    yaw = math.radians(robot_yaw_deg)
    root.quat[:] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    joints = [o + "_joint" for o in active] + ([SWITCH] if KETTLE in active else [])
    joints += [j for ex in extras for j in ex.ACTIVE]
    env, provenance = prepare_environment(corrected_environment(extras, poses), active_joints=joints)
    spec.attach(env, prefix="env/", frame=spec.worldbody.add_frame())
    for g in spec.geoms:                               # the tiled floor slab is rigid (firm contact, as the soles)
        if g.name.startswith("env/object_0000_"):
            g.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
            if not _os.environ.get("TEA_SOFT_FEET"):
                g.solref = [FOOT_TC, 1.0]              # heel strikes (~0.3 m/s) sank 6 mm with the default 20 ms
    spec.option.timestep = DT
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1920
    spec.worldbody.add_light(pos=[2.0, 1.5, 2.6], dir=[0, 0.3, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    torso = spec.body("torso_link")
    s, c = math.sin(EYE_PITCH), math.cos(EYE_PITCH)
    torso.add_camera(name="eye", pos=list(EYE_POS), xyaxes=[0, -1, 0, s, 0, c], fovy=EYE_FOVY)
    # start keyframe: the scene as authored, the robot in the maker's standing pose
    m0 = spec.compile()
    q = m0.qpos0.copy()
    for name, v in stand.items():
        j = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            q[m0.jnt_qposadr[j]] = v
    spec.add_key(name="home", qpos=q.tolist())
    return spec, report, provenance


def session(**kw) -> TeleopSession:
    spec, report, _ = build(**kw)
    s = TeleopSession(spec, report, "tea", reach_fraction=0.9)
    s.operator_guard = False                           # a motion program checks its own outcome
    return s


if __name__ == "__main__":
    s = session()
    m, d = s.m, s.d
    print("timestep", m.opt.timestep, "nu", m.nu, "nq", m.nq, "hands", s.hand_names(),
          "walking policy", None if s.loco is None else s.loco.name)
    for k in range(int(3.0 / DT)):
        s.step()
    print("after 3 s standing:", s.checks.as_dict(), "pelvis", np.round(d.qpos[:3], 3))
    kb = m.body("env/" + KETTLE + "_body").id
    print("kettle at", np.round(d.xpos[kb], 3), "mass", round(float(m.body_mass[kb]), 3))




