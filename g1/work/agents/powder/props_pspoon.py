"""Prop module: the stainless teaspoon in the tray's FRESH tub, with a real (concave) bowl that can carry powder.

Drop-in replacement for agents\\spoon\\props_spoon.py: same body/joint names (prop_spoon_*), same spoon frame
(x_s from the bowl tip to the handle end, z_s the normal of the bowl's open side, handle plane at z_s = +3 mm),
same handle (14.0 cm, 6 -> 8 -> 10 mm wide, 1.8 mm thick), same constants (BODY, HZ, LENGTH, BOWL, point,
HANDLE_END), so skill_stir.py works with it unchanged. Only the bowl differs: props_spoon's bowl is a SOLID
ellipsoid (a grain put on it rolls off its convex top); here the bowl is a dish - a 23 x 13 mm floor plate
and 12 thin (0.8 mm) steel wall segments flaring out to the 34 x 24.4 mm rim, 5.5 mm deep - so grains of
powder can lie in it. Mass 17.2 g (bowl 5.4 g), centre of mass 6.8 cm from the tip.
Contacts as props_spoon: explicit stiffness (17 g part), firm impedance, steel friction 0.4.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
import os as _os

import numpy as np

TUB = "object_0306"
NAME = "prop_spoon"
ACTIVE = [f"{NAME}_joint"]
BODY = f"{NAME}_body"

LENGTH = 0.140
HZ = 0.003
HT = 0.0009
STEEL = 7900.0
BOWL = dict(pos=(0.0175, 0.0, 0.0), size=(0.0175, 0.0125, 0.0040), mass=0.0054)   # envelope (for skills)
SEGMENTS = (
    ("neck", 0.030, 0.050, 0.0025),
    ("handle1", 0.050, 0.085, 0.0030),
    ("handle2", 0.085, 0.110, 0.0040),
    ("handle3", 0.110, 0.140, 0.0050),
)
FRICTION = "0.4 0.005 0.0001"
SOLREF = _os.environ.get("SPOON_SOLREF", "-50000 -900")
SOLIMP = "0.95 0.99 0.001 0.5 2"
STEEL_RGBA = "0.78 0.79 0.81 1"

# dish: bottom ellipse (semi-axes) at z_b, rim ellipse at z_t, centred on the bowl centre
DISH_C = 0.0175
A_B, B_B, Z_B = 0.0105, 0.0060, -0.0025
A_T, B_T, Z_T = 0.0170, 0.0122, 0.0030
PLATE_T = 0.0012
WALL_T = 0.0008
N_WALL = 12

# rest pose in the 90 mm FRESH tub (settle_powder.py: dropped from the ellipsoid spoon's rest pose, settled 3 s)
REST_POS = (2.063, 2.3837, 0.8681)
REST_QUAT = (0.5877, -0.3924, -0.3924, -0.5888)
_rest = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "pspoon_rest.npy")
if _os.path.exists(_rest) and not _os.environ.get("PSPOON_DROP"):
    _r = np.load(_rest)
    REST_POS, REST_QUAT = tuple(_r[:3]), tuple(_r[3:7])
POSE = None


def _s(v):
    return " ".join(f"{x:.6g}" for x in v)


def _mat2quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def dish_walls():
    """(centre, quat, half-size) of each wall segment in the spoon frame."""
    out = []
    for k in range(N_WALL):
        p0, p1 = 2 * math.pi * k / N_WALL, 2 * math.pi * (k + 1) / N_WALL
        pm = (p0 + p1) / 2
        B = np.array([DISH_C + A_B * math.cos(pm), B_B * math.sin(pm), Z_B])
        T = np.array([DISH_C + A_T * math.cos(pm), B_T * math.sin(pm), Z_T])
        t = np.array([-A_T * math.sin(pm), B_T * math.cos(pm), 0.0])
        t /= np.linalg.norm(t)
        u = T - B
        u -= (u @ t) * t
        hl = float(np.linalg.norm(u)) / 2
        u /= np.linalg.norm(u)
        n = np.cross(t, u)                           # outward or inward: make it outward (away from the axis)
        radial = np.array([math.cos(pm) * A_T, math.sin(pm) * B_T, 0.0])
        if n @ radial < 0:
            n = -n
        seg = math.hypot(A_T * (math.cos(p1) - math.cos(p0)), B_T * (math.sin(p1) - math.sin(p0)))
        c = (B + T) / 2 + n * WALL_T / 2
        R = np.c_[u, t, n]                           # box x up the wall, y along the rim, z through the wall
        out.append((c, _mat2quat(R), (hl + 0.0006, seg / 2 * 1.18, WALL_T / 2)))
    return out


def spoon_mass():
    m = STEEL * (2 * A_B) * (2 * B_B) * PLATE_T
    for c, q, h in dish_walls():
        m += STEEL * 8 * h[0] * h[1] * h[2]
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
    _geom(b, name=f"{NAME}_bowl", type="box", pos=(DISH_C, 0.0, Z_B - PLATE_T / 2),
          size=(A_B + 0.0008, B_B + 0.0008, PLATE_T / 2), density=STEEL, **common)
    for k, (c, q, h) in enumerate(dish_walls()):
        _geom(b, name=f"{NAME}_bowl_w{k}", type="box", pos=_s(c), quat=_s(q), size=_s(h), density=STEEL, **common)
    for n, x0, x1, hw in SEGMENTS:
        _geom(b, name=f"{NAME}_{n}", type="box", pos=((x0 + x1) / 2, 0.0, HZ), size=((x1 - x0) / 2, hw, HT),
              density=STEEL, **common)


HANDLE_END = np.array([LENGTH, 0.0, HZ])
BOWL_CENTRE = np.array([DISH_C, 0.0, Z_B + 0.002])     # a point just above the dish floor


def point(d, m, local):
    try:
        bid = m.body("env/" + BODY).id
    except KeyError:
        bid = m.body(BODY).id
    return d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ np.asarray(local, float)


def in_bowl(d, m, gpos, margin=0.001):
    """Which of the world points gpos (N x 3, grain centres) lie inside the dish (above its floor, under its
    rim plane + one grain radius, within the rim ellipse)."""
    try:
        bid = m.body("env/" + BODY).id
    except KeyError:
        bid = m.body(BODY).id
    R = d.xmat[bid].reshape(3, 3)
    L = (np.asarray(gpos) - d.xpos[bid]) @ R
    ex = ((L[:, 0] - DISH_C) / (A_T + margin)) ** 2 + (L[:, 1] / (B_T + margin)) ** 2
    return (ex < 1.0) & (L[:, 2] > Z_B - 0.001) & (L[:, 2] < Z_T + 0.006)
