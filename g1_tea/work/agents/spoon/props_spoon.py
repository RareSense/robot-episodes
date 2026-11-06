"""Prop module: a stainless teaspoon standing in the sachet tray's FRESH spoon tub (object_0306).

The tub itself is opened by tea_sim.py (lead's scene correction: 2 mm floor and walls, inner +-0.046 x
+-0.034 m, 45 mm deep; it stays fixed in its tray compartment). This module only adds the spoon.

The spoon: real teaspoon size and weight - 14.0 cm long, bowl 3.5 x 2.5 cm (8 mm deep envelope), neck 5 mm,
handle 6 -> 8 -> 10 mm wide and 1.8 mm thick; stainless steel (7900 kg/m3) -> 17.5 g, centre of mass 6.7 cm
from the bowl tip. It is a free body; it stands in the tub (90 mm deep since the lead's 29 Oct fix) with the bowl on the
tub floor and the handle up, leaning on the tub's front rim toward the robot, 22.6 deg from vertical; the
handle end is 4.3 cm above the rim at (2.063, 2.333, 0.999).

Contacts: a 17.5 g part is far lighter than everything it touches, and MuJoCo's default contact softness
scales with the lighter body's mass (BRIEF lesson 1), so the spoon's geoms carry an explicit contact
stiffness (negative solref, direct) and firm impedance; solmix makes its impedance win the mix while
friction stays the pair's maximum (MuJoCo's rule), i.e. finger pad 1.0, mug 0.55, tub 0.5. Point contacts
(condim 3, as every other object here): a two-finger pinch is a free pivot, which the skill uses to let the
spoon swing to hang below the pinch; the stirring grip is a three-point grip (no torsion needed).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

TUB = "object_0306"                      # r8i1_tub_left, "FRESH" (lead, 29 Oct): corrected centre (2.063, 2.3837), top z 0.911
NAME = "prop_spoon"
ACTIVE = [f"{NAME}_joint"]
BODY = f"{NAME}_body"                    # after attaching: "env/prop_spoon_body"

# ---------------------------------------------------------------- spoon geometry (spoon frame)
# x_s: along the spoon from the bowl tip (x=0) to the handle end (x=LENGTH); z_s: normal of the bowl's
# open (concave) side; y_s: across the width. The handle runs at z_s = +3 mm (the bowl's rim level).
LENGTH = 0.140
HZ = 0.003                               # handle plane height over the bowl's centre plane
HT = 0.0009                              # handle half-thickness (1.8 mm)
STEEL = 7900.0
BOWL = dict(pos=(0.0175, 0.0, 0.0), size=(0.0175, 0.0125, 0.0040), mass=0.0060)
SEGMENTS = (                             # (name, x0, x1, half-width)
    ("neck", 0.030, 0.050, 0.0025),
    ("handle1", 0.050, 0.085, 0.0030),
    ("handle2", 0.085, 0.110, 0.0040),
    ("handle3", 0.110, 0.140, 0.0050),
)
FRICTION = "0.4 0.005 0.0001"            # steel; the pair takes the max (finger 1.0, mug 0.55, tub 0.5)
import os as _os
SOLREF = _os.environ.get("SPOON_SOLREF", "-50000 -900")                  # direct stiffness/damping (acceleration units): ~17 steps per period
SOLIMP = "0.95 0.99 0.001 0.5 2"         # as the robot's finger pads and soles
STEEL_RGBA = "0.78 0.79 0.81 1"

# ---------------------------------------------------------------- start pose (world)
# Rest pose found in physics (settle_spoon.py 20 3 270 with TIP_Y=2.384: dropped 3 mm above the analytic
# pose, settled 3 s, spoon alone in the scene; handle-end drift over the last 1 s 0.018 mm); position of the spoon frame origin (bowl tip) and its quaternion (w, x, y, z).
REST_POS = (2.063, 2.3837, 0.8681)
REST_QUAT = (0.5877, -0.3924, -0.3924, -0.5888)
POSE = None                              # override: (x, y, z, qw, qx, qy, qz)


def analytic_pose(theta_deg=55.0, tip=(2.21, 2.4155, 0.8725), psi_deg=270.0):
    """Leaning by theta from vertical with the handle pointing toward heading psi (270: -y, the robot;
    180: -x), concave side up: x_s = sin(theta) h + cos(theta) z, z_s = -cos(theta) h + sin(theta) z."""
    t, ps_ = math.radians(theta_deg), math.radians(psi_deg)
    h = np.array([math.cos(ps_), math.sin(ps_), 0.0])
    xs = math.sin(t) * h + math.cos(t) * np.array([0.0, 0.0, 1.0])
    zs = -math.cos(t) * h + math.sin(t) * np.array([0.0, 0.0, 1.0])
    ys = np.cross(zs, xs)
    R = np.c_[xs, ys, zs]
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return tuple(tip), tuple(q)


def spoon_mass():
    m = BOWL["mass"]
    for _, x0, x1, hw in SEGMENTS:
        m += STEEL * (x1 - x0) * 2 * hw * 2 * HT
    return m


def _geom(parent, **kw):
    return ET.SubElement(parent, "geom", **{k: (" ".join(map(str, v)) if isinstance(v, (tuple, list)) else str(v))
                                            for k, v in kw.items()})


def add(root, bodies):
    world = root.find("worldbody")
    pose = POSE if POSE is not None else tuple(REST_POS) + tuple(REST_QUAT)
    b = ET.SubElement(world, "body", name=BODY, pos=" ".join(f"{v:.6f}" for v in pose[:3]),
                      quat=" ".join(f"{v:.6f}" for v in pose[3:7]))
    ET.SubElement(b, "freejoint", name=f"{NAME}_joint")
    common = dict(contype=1, conaffinity=1, condim=3, friction=FRICTION, solref=SOLREF, solimp=SOLIMP, solmix=100,
                  group=2, rgba=STEEL_RGBA)
    _geom(b, name=f"{NAME}_bowl", type="ellipsoid", pos=BOWL["pos"], size=BOWL["size"], mass=BOWL["mass"], **common)
    for n, x0, x1, hw in SEGMENTS:
        _geom(b, name=f"{NAME}_{n}", type="box", pos=((x0 + x1) / 2, 0.0, HZ), size=((x1 - x0) / 2, hw, HT),
              density=STEEL, **common)


# handy constants for skills (spoon frame)
HANDLE_END = np.array([LENGTH, 0.0, HZ])


def point(d, m, local):
    """World position of a point given in the spoon frame."""
    try:
        bid = m.body("env/" + BODY).id
    except KeyError:                     # the environment alone (no robot attached)
        bid = m.body(BODY).id
    return d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ np.asarray(local, float)
