"""Prop module: the glass whitener jar (object_0309) made hollow, a real screw lid, and the whitener powder.

Use as a tea_sim.session extra: session(..., extras=[props_powder]). Every part is a physical body; the
skill must call `install(s)` once (it hooks the thread model into the physics step, see THREAD below).

JAR (measured from the twin's compiled collision mesh, measure_jar.py): 90 mm across, base z 0 (counter
top), glass body r 45 mm up to z 0.099, shoulder to the neck r 36 mm, lid r 38 mm z 0.110-0.132. The twin's
one solid convex hull is replaced by a hollow glass shell of boxes (the body keeps the twin's 0.45 kg inertial):
  - bottom ring of 12 boxes (the jar stands on its rim: spread contacts give real twisting friction on the
    counter; one hull contact in the middle had none) and a floor 0.5 mm above it,
  - 16 wall boxes 4 mm thick (r 41-45 mm), a flat shoulder ring (r 33-45 mm, z 0.097-0.103), a 16-box neck
    (outer face 35.5 mm, inner 33 mm, rim z 0.1285): the opening is 66 mm across.
POWDER: the bulk of the powder is a rigid fill (a cylinder to z 0.064 and a 12-segment funnel rising at 32 deg
to the wall at z 0.082: the crater a used jar has); the loose top powder is N_GRAINS (120) free spheres r 2.5 mm
(lumps of spray-dried creamer, 600 kg/m3 -> 39 mg each), friction 0.7, free-joint damping 2e-6 (negligible for
sliding and falling, b/m = 0.05/s, but it stops free rolling, b/I = 1.6e4/s: the cohesion of a powder lump;
brief: "some friction/cohesion via damping"), explicit contact stiffness (BRIEF lesson 1). They start settled
in the crater (settle_powder.py drop -> grains_rest.npy, jar frame). Grains may sleep (MuJoCo sleeping, like
PhysX's: an island at rest is not integrated until something touches it) - only if POWDER_SLEEP=1.
LID: 30 g steel lid (top disc + 20-sided skirt r 37-38.5 mm, z 0.110-0.132), firm contacts (solref direct,
priority 1); the lid top has condim 4 (torsional friction 12 mm: a palm pad pressing on it has a patch).
THREAD (how the lid is held, and why): the lid body hangs on 6 unlimited joints relative to the jar (x, y, z
slides, the turn about the jar axis, two tilts) - i.e. it is a free body. While the thread is engaged, five
joint equalities make it a screw: x, y and both tilts held at 0, the turn coupled to the rise (3 mm per turn,
the lid rises as it turns anticlockwise seen from above); the thread friction is the turn's frictionloss
(FRICTION_NM, a lightly closed lid); the turn's lower limit 0 is the closed lid (and the lid rests on the
neck rim, explicit contact pairs, since MuJoCo filters parent-child contacts). When the turn reaches
THREAD_END (330 deg, ~1 turn) the thread is out: the step hook installed by install(s) switches the five
equalities off and the thread friction to 0, once; from then on the lid is a free body held by nothing but
contacts: it lifts off only because the hand carries it. No qpos/qvel writes.
Why not a geometric thread: sub-mm helical ridges of boxes chatter and jam under soft contacts (0.1-0.3 mm
penetration is the size of the thread clearance); the coupled joints are exact and stable. (First version: a
light 'nut' body welded to a free lid - it went unstable; the joint chain is well conditioned.)
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

JAR = "object_0309"
NAME = "prop_powder"
LID = f"{NAME}_lid"
NUT = f"{NAME}_nut"
TURN = f"{NAME}_thread_turn"
RISE = f"{NAME}_thread_rise"
LID_X, LID_Y, LID_TX, LID_TY = (f"{NAME}_lid_x", f"{NAME}_lid_y", f"{NAME}_lid_tx", f"{NAME}_lid_ty")
EQ_THREAD = f"{NAME}_thread"
EQ_HELD = [f"{n}_held" for n in (LID_X, LID_Y, LID_TX, LID_TY)]
EQ_SOLREF = "0.005 1"
EQ_SOLIMP = "0.99 0.999 0.001 0.5 2"
HERE = Path(__file__).resolve().parent

PITCH = 0.003                                     # m per turn
K_THREAD = PITCH / (2 * math.pi)                  # m per rad
THREAD_END = math.radians(float(os.environ.get("POWDER_THREAD_DEG", "330")))
FRICTION_NM = float(os.environ.get("POWDER_LID_NM", "0.15"))   # thread friction torque (N m)
LID_OFF = bool(int(os.environ.get("POWDER_LID_OFF", "0")))      # test fixture: start with the lid on the counter
LID_OFF_XY = (1.52, 2.17)
N_GRAINS = int(os.environ.get("POWDER_GRAINS", "120"))
GRAIN_R = 0.0025
GRAIN_RHO = 600.0
GRAIN_SOLREF = "-30000 -400"                      # direct (acceleration units): ~22 steps per period
GRAIN_SOLIMP = "0.95 0.99 0.001 0.5 2"
GRAIN_FRICTION = os.environ.get("POWDER_GFRICTION", "1.0 0.02 0.005")   # cohesive powder lumps: sliding 1.0, torsional and
                                                   # rolling friction (rolls only past ~63 deg; repose ~45 deg)
GRAIN_CONDIM = int(os.environ.get("POWDER_CONDIM", "6"))
GRAIN_DAMP = float(os.environ.get("POWDER_GDAMP", "2e-6"))
GRAIN_SLEEP = bool(int(os.environ.get("POWDER_SLEEP", "1")))
LID_MASS = 0.030
LID_SOLREF = "-50000 -900"                        # as the spoon: firm, ~17 steps per period
LID_SOLIMP = "0.95 0.99 0.001 0.5 2"
GLASS_FR = "0.5 0 0"

# jar-frame geometry (m)
R_OUT, R_WALL_IN = 0.045, 0.041
Z_WALL0, Z_WALL1 = 0.003, 0.099
SHOULDER = (0.033, 0.045, 0.097, 0.103)           # r_in, r_out, z0, z1
NECK = (0.033, 0.0355, 0.103, 0.1285)            # rim at the lid top's underside: the closed lid rests on it
LID_Z0, LID_Z1, LID_TOP_T = 0.110, 0.132, 0.0035
LID_R_IN, LID_R_OUT = 0.037, 0.0385
FILL_TOP = 0.080                                   # crater floor (a well-filled jar: the Dex3's three-finger grip
                                                   # leaves only 6.5 cm of spoon under the middle finger)
FUNNEL = (0.012, 0.041, 0.080, 0.094)              # r_in, r_out, z at r_in, z at r_out (crater wall)
GRAINS_FILE = HERE / "grains_rest.npy"

ACTIVE = [LID_X, LID_Y, RISE, TURN, LID_TX, LID_TY] + [f"{NAME}_g{i}_joint" for i in range(N_GRAINS)]


def _s(v):
    return " ".join(f"{x:.6g}" for x in v)


def _ring(parent, name, r_in, r_out, z0, z1, n, **kw):
    """n boxes forming a polygonal ring (apothem r_in .. r_out) between z0 and z1."""
    rc, ht = (r_in + r_out) / 2, (r_out - r_in) / 2
    half_t = r_out * math.tan(math.pi / n) * 1.02
    for k in range(n):
        a = 2 * math.pi * k / n
        q = (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))
        ET.SubElement(parent, "geom", name=f"{name}_{k}", type="box",
                      pos=_s((rc * math.cos(a), rc * math.sin(a), (z0 + z1) / 2)),
                      size=_s((ht, half_t, (z1 - z0) / 2)), quat=_s(q), **kw)


def _funnel(parent, name, n, **kw):
    r0, r1, z0, z1 = FUNNEL
    slope = math.atan2(z1 - z0, r1 - r0)
    L = math.hypot(r1 - r0, z1 - z0)
    t = 0.004
    rc = (r0 + r1) / 2
    zc = (z0 + z1) / 2
    half_t = r1 * math.tan(math.pi / n) * 1.05
    for k in range(n):
        a = 2 * math.pi * k / n
        # box frame: x radial (tilted up by `slope` toward the wall), y tangential, z its surface normal
        qa = np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])
        qs = np.array([math.cos(-slope / 2), 0.0, math.sin(-slope / 2), 0.0])      # about y: x up toward +z
        q = _qmul(qa, qs)
        nrm = np.array([-math.sin(slope) * math.cos(a), -math.sin(slope) * math.sin(a), math.cos(slope)])
        c = np.array([rc * math.cos(a), rc * math.sin(a), zc]) - nrm * t / 2         # top face on the cone
        ET.SubElement(parent, "geom", name=f"{name}_{k}", type="box", pos=_s(c),
                      size=_s((L / 2, half_t, t / 2)), quat=_s(q), **kw)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def grain_start_local(n=N_GRAINS):
    """Grain centres in the jar frame: the settled rest positions if found, else a loose drop grid above
    the crater (settle_powder.py drops them and saves the rest positions)."""
    if GRAINS_FILE.exists() and not os.environ.get("POWDER_DROP"):
        P = np.load(GRAINS_FILE)
        if len(P) >= n:
            return P[:n]
    pts = []
    d = 2 * GRAIN_R + 0.0012
    z = 0.0935                                     # over the crater, under the shoulder (z 0.097), inside the neck
    layer = 0
    while len(pts) < n:
        off = 0.5 * d * (layer % 2)
        layer += 1
        for i in range(-9, 10):
            for j in range(-9, 10):
                x, y = i * d + off, j * d + off
                if math.hypot(x, y) < 0.028 and len(pts) < n:
                    pts.append((x, y, z))
        z += d
    return np.array(pts[:n])


def add(root, bodies):
    world = root.find("worldbody")
    jar = bodies[JAR + "_body"]
    for g in list(jar.findall("geom")):            # the twin's solid hull goes; its visual mesh stays
        if g.get("name") == f"{JAR}_collision_00":
            jar.remove(g)
    jp = np.array([float(x) for x in jar.get("pos").split()])
    jq = np.array([float(x) for x in (jar.get("quat") or "1 0 0 0").split()])
    Rj = _quat_mat(jq / np.linalg.norm(jq))
    glass = dict(contype="1", conaffinity="1", condim="3", friction=GLASS_FR, mass="0", group="3", rgba="0.8 0.9 0.9 0.15")
    for k in range(12):                            # the bottom rim the jar stands on
        a = 2 * math.pi * k / 12
        ET.SubElement(jar, "geom", name=f"{NAME}_foot_{k}", type="box",
                      pos=_s((0.038 * math.cos(a), 0.038 * math.sin(a), 0.0015)),
                      size=_s((0.003, 0.0102, 0.0015)), quat=_s((math.cos(a / 2), 0, 0, math.sin(a / 2))), **glass)
    ET.SubElement(jar, "geom", name=f"{NAME}_floor", type="cylinder", pos="0 0 0.0055", size="0.041 0.002", **glass)
    _ring(jar, f"{NAME}_wall", R_WALL_IN, R_OUT, Z_WALL0, Z_WALL1, 16, **glass)
    _ring(jar, f"{NAME}_shoulder", *SHOULDER, 16, **glass)
    _ring(jar, f"{NAME}_neck", *NECK, 16, **glass)
    fill = dict(contype="1", conaffinity="1", condim="3", friction="0.7 0 0", mass="0", group="3", rgba="0.96 0.94 0.88 1")
    ET.SubElement(jar, "geom", name=f"{NAME}_fill", type="cylinder", pos=_s((0, 0, (0.0075 + FILL_TOP) / 2)),
                  size=_s((R_WALL_IN - 0.0003, (FILL_TOP - 0.0075) / 2)), **fill)
    _funnel(jar, f"{NAME}_crater", 12, **fill)
    # the lid: 6 unlimited joints relative to the jar (x, y, z slides, then the turn about the jar axis, then
    # two tilts) = a free body; while the thread is engaged, joint equalities hold x, y and the tilts at 0 and
    # couple the rise to the turn (3 mm per turn); the thread friction is the turn's frictionloss
    lid_pos = np.zeros(3)
    if LID_OFF:                                    # test fixture: the lid already on the counter, top up
        lid_pos = Rj.T @ (np.array([LID_OFF_XY[0], LID_OFF_XY[1], 0.86 - LID_Z0 + 0.0002]) - jp)
    lid = ET.SubElement(jar, "body", name=LID, pos=_s(lid_pos))
    for nm, typ, ax in ((LID_X, "slide", "1 0 0"), (LID_Y, "slide", "0 1 0"), (RISE, "slide", "0 0 1"),
                        (TURN, "hinge", "0 0 1"), (LID_TX, "hinge", "1 0 0"), (LID_TY, "hinge", "0 1 0")):
        extra = dict(frictionloss=f"{FRICTION_NM}", limited="true", range="0 1000") if nm == TURN else dict(limited="false")
        ET.SubElement(lid, "joint", name=nm, type=typ, axis=ax, pos="0 0 0.121", damping="0",
                      armature="0", **extra)
    ET.SubElement(lid, "inertial", pos=_s((0, 0, 0.125)), mass=f"{LID_MASS}", diaginertia="1.3e-5 1.3e-5 2.2e-5")
    steel = dict(contype="1", conaffinity="1", condim="3", friction="0.6 0.005 0.0001", mass="0", group="2",
                 rgba="0.85 0.86 0.88 1", solref=LID_SOLREF, solimp=LID_SOLIMP, priority="1")
    top = dict(steel, condim="4", friction="0.6 0.012 0.0001")
    ET.SubElement(lid, "geom", name=f"{NAME}_lid_top", type="cylinder",
                  pos=_s((0, 0, LID_Z1 - LID_TOP_T / 2)), size=_s((LID_R_OUT, LID_TOP_T / 2)), **top)
    _ring(lid, f"{NAME}_lid_skirt", LID_R_IN, LID_R_OUT, LID_Z0, LID_Z1 - LID_TOP_T, 20, **steel)
    eq = root.find("equality")
    if eq is None:
        eq = ET.SubElement(root, "equality")
    if os.environ.get("POWDER_NOEQ"):                # diagnostics only
        eq = ET.Element("equality")
    # the lid is a child of the jar (its 6 joints), and MuJoCo filters parent-child contacts: explicit contact
    # pairs between the lid's geoms and two proxies of the jar's neck (r 35.5 mm, rim z 0.1285) and shoulder
    # (r 45 mm) let the closed lid rest on the rim and a loose lid knock against the jar
    prox = dict(contype="0", conaffinity="0", mass="0", group="4", rgba="0 0 0 0")
    ET.SubElement(jar, "geom", name=f"{NAME}_neck_proxy", type="cylinder",
                  pos=_s((0, 0, (NECK[2] + NECK[3]) / 2)), size=_s((NECK[1], (NECK[3] - NECK[2]) / 2)), **prox)
    ET.SubElement(jar, "geom", name=f"{NAME}_shoulder_proxy", type="cylinder",
                  pos=_s((0, 0, (SHOULDER[2] + SHOULDER[3]) / 2)), size=_s((SHOULDER[1], (SHOULDER[3] - SHOULDER[2]) / 2)), **prox)
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for lg in [f"{NAME}_lid_top"] + [f"{NAME}_lid_skirt_{k}" for k in range(20)]:
        for jg in (f"{NAME}_neck_proxy", f"{NAME}_shoulder_proxy"):
            ET.SubElement(contact, "pair", geom1=lg, geom2=jg, condim="3", friction="0.6 0.6 0.005 0.0001 0.0001",
                          solref=LID_SOLREF, solimp=LID_SOLIMP)
    act = "false" if LID_OFF else "true"
    # turn = rise / k (this way round: MuJoCo's approximate constraint mass for a joint coupling ignores the
    # coefficient, and with rise = k turn it came out 1000x too soft - the lid sagged 0.75 mm on the thread)
    ET.SubElement(eq, "joint", name=EQ_THREAD, joint1=TURN, joint2=RISE, polycoef=f"0 {1.0 / K_THREAD:.6f} 0 0 0",
                  solref=EQ_SOLREF, solimp=EQ_SOLIMP, active=act)
    for nm in (LID_X, LID_Y, LID_TX, LID_TY):
        ET.SubElement(eq, "joint", name=f"{nm}_held", joint1=nm, polycoef="0 0 0 0 0",
                      solref=EQ_SOLREF, solimp=EQ_SOLIMP, active=act)
    # the loose powder: free grains
    gm = GRAIN_RHO * 4.0 / 3.0 * math.pi * GRAIN_R ** 3
    gi = 2.0 / 5.0 * gm * GRAIN_R ** 2
    P = grain_start_local()
    for i, pl in enumerate(P):
        pw = jp + Rj @ pl
        gb = ET.SubElement(world, "body", name=f"{NAME}_g{i}", pos=_s(pw))
        if GRAIN_SLEEP:
            gb.set("sleep", "allowed")
        # free joint with a little damping: on a 29 mg lump it is negligible for sliding/falling (b/m = 0.07/s)
        # but stops free rolling at once (b/I = 3.4e4/s) - the cohesion of a powder lump (brief: "some
        # friction/cohesion via damping"); the rolling friction of condim 6 cost 2.5x the solver rows
        ET.SubElement(gb, "joint", type="free", name=f"{NAME}_g{i}_joint", damping=f"{GRAIN_DAMP:g}")
        ET.SubElement(gb, "inertial", pos="0 0 0", mass=f"{gm:.4e}", diaginertia=_s((gi, gi, gi)))
        ET.SubElement(gb, "geom", name=f"{NAME}_g{i}_geom", type="sphere", size=f"{GRAIN_R}", mass="0",
                      contype="1", conaffinity="1", condim=str(GRAIN_CONDIM), friction=GRAIN_FRICTION, solref=GRAIN_SOLREF,
                      solimp=GRAIN_SOLIMP, priority="2", group="2", rgba="0.97 0.95 0.90 1")


# ---------------------------------------------------------------- runtime: the thread's end
class Thread:
    """Step hook: when the lid has turned to the top of the thread, the thread lets go: its five joint
    equalities (rise coupled to turn; x, y and the tilts held) are switched off and the thread friction ends
    (the turn's frictionloss -> 0), once. Chains the session's existing step observer (the recorder)."""

    def __init__(self, s, prefix="env/"):
        import mujoco
        m = s.m
        self.s = s
        j = m.joint(prefix + TURN).id
        self.turn_q = int(m.jnt_qposadr[j])
        self.turn_dof = int(m.jnt_dofadr[j])
        self.eqs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, prefix + n) for n in [EQ_THREAD] + EQ_HELD]
        self.prev = s.step_observer
        self.released_at = None if (min(self.eqs) >= 0 and s.d.eq_active[self.eqs[0]]) else -1.0
        self.track, self.log_t, self.log_p, self.log_q = [], [], [], []   # 600 Hz poses (powder_pass.py)

    def save_poses(self, path, names):
        np.savez_compressed(path, t=np.array(self.log_t), pos=np.array(self.log_p), quat=np.array(self.log_q),
                            names=np.array(names), released_at=-2.0 if self.released_at is None else self.released_at)

    def angle(self):
        return float(self.s.d.qpos[self.turn_q])

    def release(self, s):
        for e in self.eqs:
            s.d.eq_active[e] = 0
        s.m.dof_frictionloss[self.turn_dof] = 0.0
        self.released_at = float(s.d.time)

    def before_step(self, s):
        if self.prev is not None:
            self.prev.before_step(s)

    def before_integrate(self, s):
        if self.prev is not None:
            self.prev.before_integrate(s)

    def after_step(self, s):
        if self.released_at is None and s.d.qpos[self.turn_q] >= THREAD_END - math.radians(0.5):
            self.release(s)
        if self.track:                                   # read-only: body poses every physics step
            self.log_t.append(s.d.time)
            self.log_p.append(s.d.xpos[self.track].copy())
            self.log_q.append(s.d.xquat[self.track].copy())
        if self.prev is not None:
            self.prev.after_step(s)


def install(s):
    """Hook the thread model into the session's physics step (wraps the current step observer)."""
    th = getattr(s, "_powder_thread", None)
    if th is None or s.step_observer is not th:
        th = Thread(s)
        s.step_observer = th
        s._powder_thread = th
    if GRAIN_SLEEP:                                  # on in all scoop runs: the spoon pickup was tuned with it (off: pinch 7.8 N, spoon lost, scoop10/11)
        # MuJoCo's sleeping (like PhysX's): a grain island at rest for a while stops being integrated until
        # something touches it. Only the grains are allowed to sleep (body sleep="allowed"); turned on only if
        # the robot's own tree can never sleep.
        import mujoco
        m = s.m
        pel = m.body("pelvis").id
        pol = int(m.tree_sleep_policy[m.body_treeid[pel]])
        if pol in (int(mujoco.mjtSleepPolicy.mjSLEEP_NEVER), int(mujoco.mjtSleepPolicy.mjSLEEP_AUTO_NEVER)):
            m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
            th.sleep = True
        else:
            th.sleep = f"not enabled: robot tree sleep policy {pol}"
    return th


def grain_ids(m, prefix="env/"):
    return [m.body(f"{prefix}{NAME}_g{i}").id for i in range(N_GRAINS)]












