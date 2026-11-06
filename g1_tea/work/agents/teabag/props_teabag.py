"""The tea bag as a physical prop: bag + cotton thread (a cable) + paper tag, free bodies coupled only by
joints and contacts. Use as a tea_sim.session extra: session(..., extras=[props_teabag]).

  bag    sachet 6.5 x 5 x 0.4 cm, 2.0 g, one rigid body (filter paper filled with tea: a stiff pouch; a
         MuJoCo flex at 2 g would need its own stiffness tuning and risks instability - see REPORT.md)
  thread cotton string 16 cm, 0.8 mm thick, 0.1 g: a chain of 20 capsules (8 mm) joined by pairs of
         bending hinges (no twist dof: a 0.8 mm string has ~1e-13 kg m^2 twist inertia, numerically
         singular), small damping, no bending stiffness (a limp string)
  tag    card 2.5 x 2.0 x 0.1 cm, 0.5 g, on a ball joint at the string's end (the staple, 2 mm from one
         end)

Start pose (tray geometry measured from the compiled twin, measure_scene.py): the loose-tea-bag
compartment is x 2.285-2.429, y 2.231-2.329; the twin's pile of loose bags fills it to z 0.8955, 2.2 cm
below the divider tops (0.9175), leaving an 8 mm slot between the pile's front face (y 2.239) and the front
divider (y 2.231) down to the tray floor (z 0.8655). Our bag stands in that slot on its bottom edge,
leaning back against the divider's top edge (bottom edge in the floor/pile corner: it needs no friction to
stand, so it cannot creep), its top 1.2 cm above the divider. The tag leans against the bag's back (north)
face, its lower end (staple) on the pile; the string runs from the staple on the bag's top down the bag
and in a loose loop on the pile to the tag.

Why not lying flat: the Dex3's thumb-index pinch has the index finger lying along the card for 2-3 cm past
the contact (probe_pinch.py), so a flat card on a surface cannot be pinched; a standing tag can.

Contacts: the parts are light (0.1-2 g) and MuJoCo's contact softness grows with the lighter body's
inverse mass, so every geom has explicit stiffness (solref -K -D) and a hard impedance (solimp 0.999),
priority >= 1 (its own friction/stiffness win against the scene's defaults). The bag has priority 2 so a
tag sliding on it uses paper-on-paper friction (0.4).
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

import numpy as np

NAME = "prop_teabag"
BAG = f"{NAME}_bag"
TAG = f"{NAME}_tag"
N_SEG = 20
SEG_L = float(os.environ.get("TB_SEGL", "0.006"))   # 20 x 6 mm = 12 cm string (shorter than 16: see report)
THREAD_R = 0.0004                    # 0.8 mm cotton string
BAG_HALF = (0.0325, 0.025, 0.002)    # 6.5 x 5 x 0.4 cm (bag frame: +x toward the top edge, z its back-face normal)
TAG_HALF = (0.0125, 0.010, 0.0005)   # 2.5 x 2.0 x 0.1 cm (tag frame: +x from the staple toward the free end)
TAG_CENTRE = (0.0105, float(os.environ.get("TB_TAGOFF", "0.0")), 0.0)   # staple 2 mm from the top, 7 mm east of the middle
BAG_MASS, TAG_MASS, THREAD_MASS = 0.0020, 0.0005, 0.0001
ATTACH = (0.0290, -0.015, BAG_HALF[2] + THREAD_R + 0.0002)   # string stapled on the bag's back face near the top

# contact: explicit stiffness (MuJoCo direct solref: -stiffness -damping) and a hard impedance
SOLIMP = os.environ.get("TB_SOLIMP", "0.999 0.9995 0.001 0.5 2")
SOL_TAG = os.environ.get("TB_SOLTAG", "-100000 -1000")       # card between finger pads
SOL_THREAD = os.environ.get("TB_SOLTHREAD", "-50000 -600")
SOL_BAG = os.environ.get("TB_SOLBAG", "-30000 -1000")
FR_TAG, FR_THREAD, FR_BAG = "0.9 0.005 0.0001", "0.5 0.005 0.0001", "0.6 0.005 0.0001"
THREAD_DAMP = float(os.environ.get("TB_TDAMP", "2e-6"))      # N m s/rad per bending hinge
THREAD_ARM = float(os.environ.get("TB_TARM", "2e-9"))        # kg m^2 per hinge (>= 6 steps per period)

# tray geometry (measured from the compiled twin: measure_scene.py)
PILE_TOP, PILE_FRONT = 0.8955, 2.239
FLOOR = 0.8655
DIVIDER_TOP, DIVIDER_N = 0.9175, 2.231          # front divider of the compartment: top z, north face y
BAG_X = 2.315                                   # bag centre line (compartment x 2.285-2.429)
TAG_X = 2.326

# which start pose: "tray" (in the loose-tea-bag compartment; the Dex3 cannot pinch the tag there, see
# REPORT.md) or "rig" (TEST FIXTURE, not in the twin: the bag on a small static shelf, its tag hanging in
# free air below the shelf's edge, so the pinch, lift, bag-into-mug and tag-over-rim can be tested)
START = os.environ.get("TB_START", "stand")
# "bag" (lead, 30 Oct night): the bag is picked by its own overhanging front, pinched from above. It lies FOLDED
# IN HALF lengthwise (two 2 mm layers, 6.5 x 2.5 cm: the twin's mug cavity is 42 mm wide), flat on the heap's
# slab, its front 2.5 cm out in free air; string from its back end over the heap, tag lying on the heap
BAGF_HALF = (0.0325, 0.0125, 0.002)
BAG_X, BAG_OVER = 2.3225, 0.025
STAND_X = 2.303                                    # standing bag centre x (compartment x 2.285-2.429)
# "slab" (lead, 30 Oct): the full heap's top layer is a 1 cm slab overhanging the front divider by 3 cm; the
# bag lies back on it, the string runs over the slab's front edge and the tag hangs in free air below it
SLAB_TOP, SLAB_T, SLAB_OVER = 0.957, 0.005, 0.030
SLAB_BAG_BACK, SLAB_TAG_X = 0.004, 2.3925          # bag's front edge this far behind the divider face; string x
# "heap" (lead's decision 30 Oct: the real tray is full): a static block fills the loose-bag compartment
# (and covers its front divider), mounded HEAP_TOP; our bag lies on it overhanging the front by OVERHANG,
# the string pinned under the bag, looped back over its top and over its front edge; the tag hangs in free
# air above the next compartment, facing TAG_YAW (deg from +y toward +x: the right shoulder's direction)
HEAP_LO, HEAP_HI, HEAP_TOP = (2.2855, 2.227, 0.8655), (2.4285, 2.3285), 0.955
OVERHANG, HINGE_X, TAG_YAW = 0.023, 2.405, float(os.environ.get("TB_TAGYAW", "-75"))
FOLD = float(os.environ.get("TB_FOLD", "1.5"))           # crease rest angle (rad) of the rig bag: hangs ~3.6 cm wide
CREASE_K, CREASE_D = 0.003, 2e-5                    # N m/rad, N m s/rad: a soft paper crease
SHELF_TOP, SHELF_C, SHELF_HALF = 1.15, (2.36, 2.255), (0.05, 0.045, 0.0025)

ACTIVE = [f"{BAG}_joint"] + ([f"{BAG}_crease"] if START not in ("bag", "stand") else []) + [f"{NAME}_s{i}_{a}" for i in range(N_SEG) for a in ("b1", "b2")] + [f"{TAG}_b1", f"{TAG}_b2"]


def _quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q / np.linalg.norm(q)


def _fmt(v):
    return " ".join(f"{x:.7g}" for x in v)


def _frame(lean_south):
    """Axes of a plate standing up, leaning south by `lean_south` rad: x up along it, z its north face normal."""
    x = np.array([0.0, -math.sin(lean_south), math.cos(lean_south)])
    z = np.array([0.0, math.cos(lean_south), math.sin(lean_south)])
    return np.c_[np.cross(z, x), x, z][:, [1, 0, 2]] if False else np.c_[x, np.cross(z, x), z]


def start_pose():
    """Bag (pos, R), string polyline (N_SEG+1 points, world), tag (pos, R)."""
    t = 2 * BAG_HALF[2]
    # bag: bottom-north corner against the pile's front face just above the floor; south face through the
    # divider's top-north corner
    nb = np.array([PILE_FRONT - 0.0003, FLOOR + 0.0004])                    # (y, z) bottom-north corner
    beta = 0.0
    for _ in range(50):                                                     # solve the lean angle
        sb = nb - t * np.array([math.cos(beta), math.sin(beta)])            # bottom-south corner
        s = (DIVIDER_TOP - sb[1]) / math.cos(beta)
        y_at = sb[0] - s * math.sin(beta)
        beta += (y_at - (DIVIDER_N - 0.0003)) / s * 0.8
    R = _frame(beta)
    centre_yz = nb + BAG_HALF[0] * np.array([-math.sin(beta), math.cos(beta)]) - BAG_HALF[2] * np.array([math.cos(beta), math.sin(beta)])
    pos = np.array([BAG_X, centre_yz[0], centre_yz[1]])
    face = lambda u, v, h=0.0: pos + R @ np.array([u, v, BAG_HALF[2] + h])  # noqa: E731 point on the back face

    # tag: leaning south against the bag's back face, its lower end on the pile just behind the pile's edge
    bot = np.array([PILE_FRONT + 0.0006, PILE_TOP + 0.0003])                # (y, z) tag's bottom-south corner
    L = 2 * TAG_HALF[0]
    gam = 0.0
    for _ in range(60):
        top = bot + L * np.array([-math.sin(gam), math.cos(gam)])
        # bag back-face y at that height
        s = (top[1] - (nb[1])) / math.cos(beta)
        yb = nb[0] - s * math.sin(beta)
        gam += (top[0] - (yb + 0.0004)) / L * 0.8
    tR = _frame(gam)
    tag_pos = np.array([TAG_X, bot[0], bot[1]]) + tR @ np.array([0.002, 0.0, TAG_HALF[2]])   # staple, mid-plane

    # string: from the staple on the bag's back face down the bag, over onto the pile, a loose loop on the
    # pile, to the tag's staple (approached from behind, on the pile)
    a = face(ATTACH[0], ATTACH[1], THREAD_R + 0.0002)
    down = face(-0.0005, ATTACH[1], THREAD_R + 0.0002)
    on = np.array([a[0], PILE_FRONT + 0.0025, PILE_TOP + THREAD_R + 0.0002])
    loop = np.array([[a[0], 2.252], [a[0] - 0.002, 2.275], [2.318, 2.290], [2.343, 2.282], [2.352, 2.262],
                     [2.345, 2.248]])
    end_xy = np.array([TAG_X, PILE_FRONT + 0.0045])
    zp = PILE_TOP + THREAD_R + 0.0002

    def poly(lam):
        c = np.array([on[0], on[1]])
        lp = c + lam * (loop - c)
        pts = [a, down, on] + [np.r_[p, zp] for p in lp] + [np.r_[end_xy, zp], tag_pos]
        return np.array(pts)

    lo, hi = 0.2, 1.6
    for _ in range(60):
        lam = 0.5 * (lo + hi)
        Lp = np.linalg.norm(np.diff(poly(lam), axis=0), axis=1).sum()
        lo, hi = (lam, hi) if Lp < N_SEG * SEG_L else (lo, lam)
    W = poly(lam)
    sW = np.r_[0, np.cumsum(np.linalg.norm(np.diff(W, axis=0), axis=1))]
    P = np.array([np.array([np.interp(si, sW, W[:, k]) for k in range(3)]) for si in np.linspace(0, sW[-1], N_SEG + 1)])
    return pos, R, P, tag_pos, tR


def rig_pose():
    """TEST FIXTURE start: bag flat on the shelf, its top end 5 mm from the shelf's south edge, string over
    the edge, tag hanging in free air."""
    edge_y = SHELF_C[1] - SHELF_HALF[1]
    R = np.c_[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]          # bag x south (top end), z up
    pos = np.array([SHELF_C[0], edge_y + 0.005 + BAG_HALF[0], SHELF_TOP + BAG_HALF[2] + 0.0002])
    a = pos + R @ np.array([ATTACH[0], 0.0, ATTACH[2]])
    p1 = np.array([pos[0], edge_y + 0.005 - 0.0002, SHELF_TOP + 2 * BAG_HALF[2] + THREAD_R])
    p2 = np.array([pos[0], edge_y - THREAD_R - 0.0002, SHELF_TOP + THREAD_R])
    used = np.linalg.norm(p1 - a) + np.linalg.norm(p2 - p1)
    p3 = p2 - np.array([0.0, 0.0, N_SEG * SEG_L - used])
    W = np.array([a, p1, p2, p3])
    sW = np.r_[0, np.cumsum(np.linalg.norm(np.diff(W, axis=0), axis=1))]
    P = np.array([[np.interp(si, sW, W[:, k]) for k in range(3)] for si in np.linspace(0, sW[-1], N_SEG + 1)])
    tR = np.c_[[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]       # tag x down (staple on top), z north
    return pos, R, P, P[-1].copy(), tR


def heap_pose():
    """Bag on the heap (panel A flat, panel B folded up at the crease), string pinned under A, looped onto
    A's top and over its front (south) edge, tag hanging below it."""
    R = np.c_[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]          # bag x north, y west, z up
    r = THREAD_R
    centre_y = HEAP_LO[1] - OVERHANG + BAG_HALF[0]
    pos = np.array([HINGE_X, centre_y, HEAP_TOP + 2 * r + BAG_HALF[2] + 0.0002])
    hx, t = BAG_HALF[0], BAG_HALF[2]
    zb, zt = -t - r - 0.0001, t + r + 0.0002                               # string under / on top of panel A
    yA, yT = 0.0125, 0.005                                                  # bag-y of the string under / on top
    def loop(k):
        return [(hx + 0.004, yA, zb), (hx + 0.004 + 0.006 * k, yA + 0.004, zb), (hx + 0.004 + 0.009 * k, 0.5 * (yA + yT), zb),
                (hx + 0.004 + 0.006 * k, yT - 0.002, zb), (hx + 0.003, yT, zb + 0.0015), (hx - 0.001, yT, zt)]
    def poly(k, h):
        pts = [(-hx + 0.0195, yA, zb)] + loop(k) + [(-hx, yT, zt), (-hx - r - 0.0003, yT, 0.0), (-hx - r - 0.0003, yT, -t - h)]
        return np.array([pos + R @ np.array(p) for p in pts])
    h = 0.003
    lo, hi = 0.0, 4.0
    for _ in range(60):
        k = 0.5 * (lo + hi)
        Lp = np.linalg.norm(np.diff(poly(k, h), axis=0), axis=1).sum()
        lo, hi = (k, hi) if Lp < N_SEG * SEG_L else (lo, k)
    W = poly(k, h)
    sW = np.r_[0, np.cumsum(np.linalg.norm(np.diff(W, axis=0), axis=1))]
    P = np.array([[np.interp(si, sW, W[:, j]) for j in range(3)] for si in np.linspace(0, sW[-1], N_SEG + 1)])
    a = math.radians(TAG_YAW)
    n = np.array([math.sin(a), math.cos(a), 0.0])
    tR = np.c_[[0.0, 0.0, -1.0], [-n[1], n[0], 0.0], n]                    # tag x down (staple on top), z its face normal
    return pos, R, P, P[-1].copy(), tR


def slab_pose():
    """Bag on the slab well back from its edge; string pinned under panel A, looped over its top, off its
    front edge, along the slab and over the slab's front edge; tag hanging in free air below the slab."""
    R = np.c_[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]          # bag x north, y west, z up
    r, hx, t = THREAD_R, BAG_HALF[0], BAG_HALF[2]
    yT = 0.005
    hinge_x = SLAB_TAG_X + yT                                               # string on top of A runs at x = SLAB_TAG_X
    front_y = HEAP_LO[1] + SLAB_BAG_BACK
    pos = np.array([hinge_x, front_y + hx, SLAB_TOP + 2 * r + t + 0.0002])
    zb, zt, yA = -t - r - 0.0001, t + r + 0.0002, 0.0125
    edge_y = HEAP_LO[1] - SLAB_OVER
    w = lambda u, v, z: pos + R @ np.array([u, v, z])                       # noqa: E731 bag frame -> world
    zs = SLAB_TOP + r + 0.0002
    def poly(k, h):
        pts = [w(-hx + 0.033, yA, zb), w(hx + 0.004, yA, zb), w(hx + 0.004 + 0.006 * k, yA + 0.004, zb),
               w(hx + 0.004 + 0.009 * k, 0.5 * (yA + yT), zb), w(hx + 0.004 + 0.006 * k, yT - 0.002, zb),
               w(hx + 0.003, yT, zb + 0.0015), w(hx - 0.001, yT, zt), w(-hx, yT, zt), w(-hx - r - 0.0003, yT, 0.0),
               np.array([SLAB_TAG_X, front_y - 0.004, zs]), np.array([SLAB_TAG_X, edge_y + 0.001, zs]),
               np.array([SLAB_TAG_X, edge_y - r - 0.0003, SLAB_TOP - 0.002]),
               np.array([SLAB_TAG_X, edge_y - r - 0.0003, SLAB_TOP - SLAB_T - h])]
        return np.array(pts)
    h = 0.004 + (0.010 - SLAB_T)                                            # the tag at the same height for any slab
    lo, hi = 0.0, 6.0
    for _ in range(60):
        k = 0.5 * (lo + hi)
        Lp = np.linalg.norm(np.diff(poly(k, h), axis=0), axis=1).sum()
        lo, hi = (k, hi) if Lp < N_SEG * 0.008 else (lo, k)
    W = poly(k, h)
    sW = np.r_[0, np.cumsum(np.linalg.norm(np.diff(W, axis=0), axis=1))]
    P = np.array([[np.interp(si, sW, W[:, j]) for j in range(3)] for si in np.linspace(0, sW[-1], N_SEG + 1)])
    a = math.radians(TAG_YAW)
    n = np.array([math.sin(a), math.cos(a), 0.0])
    tR = np.c_[[0.0, 0.0, -1.0], [-n[1], n[0], 0.0], n]
    return pos, R, P, P[-1].copy(), tR


def bag_pose():
    """Folded bag flat on the slab, front 2.5 cm over the slab's edge; string from the staple at its back end
    (top face) back over the heap in a loose S; tag lying flat on the heap."""
    R = np.c_[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]          # bag x south (front), y east, z up
    r, hx, t = THREAD_R, BAGF_HALF[0], BAGF_HALF[2]
    edge_y = HEAP_LO[1] - SLAB_OVER
    pos = np.array([BAG_X, edge_y - BAG_OVER + hx, SLAB_TOP + t + 0.0002])
    zs = SLAB_TOP + r + 0.0002
    a = pos + R @ np.array([-hx + 0.003, 0.0, t + r + 0.0002])          # staple on the top face at the back end
    b = pos + R @ np.array([-hx - r - 0.0003, 0.0, 0.0])
    def poly(k):
        pts = [a, np.r_[b[:2], zs] + [0, 0, 0.0], np.r_[BAG_X, b[1] + 0.012, zs], np.r_[BAG_X + 0.012 * k, b[1] + 0.03, zs],
               np.r_[BAG_X + 0.02 * k, b[1] + 0.05, zs], np.r_[BAG_X + 0.01 * k, b[1] + 0.068, zs],
               np.r_[BAG_X - 0.01 * k, b[1] + 0.078, zs]]
        return np.array(pts)
    lo, hi = 0.0, 4.0
    for _ in range(60):
        k = 0.5 * (lo + hi)
        Lp = np.linalg.norm(np.diff(poly(k), axis=0), axis=1).sum()
        lo, hi = (k, hi) if Lp < N_SEG * 0.008 else (lo, k)
    W = poly(k)
    sW = np.r_[0, np.cumsum(np.linalg.norm(np.diff(W, axis=0), axis=1))]
    P = np.array([[np.interp(si, sW, W[:, j]) for j in range(3)] for si in np.linspace(0, sW[-1], N_SEG + 1)])
    d_ = P[-1] - P[-2]
    d_[2] = 0.0
    d_ /= np.linalg.norm(d_)
    tR = np.c_[d_, np.cross([0.0, 0.0, 1.0], d_), [0.0, 0.0, 1.0]]       # tag flat on the heap, along the string
    tag_pos = P[-1] + np.array([0.0, 0.0, TAG_HALF[2] - r + 0.0002])
    return pos, R, P, tag_pos, tR


def stand_pose():
    """Folded bag (6.5 x 2.5 x 0.4 cm) standing in the 8 mm slot between the twin's pile of loose bags and the
    compartment's front divider: bottom edge in the floor/pile corner, leaning south on the divider's top edge
    (statically determinate: no friction needed, no creep); top 1.35 cm above the divider - like the spoon's
    handle end standing out of its tub. String from a staple at the top of its back (north) face, down the
    face at its west edge, a loose loop on the pile; tag lying flat on the pile."""
    hx, hy, t = BAGF_HALF
    th = 2 * t
    nb = np.array([PILE_FRONT - 0.0003, FLOOR + 0.0004])                    # (y, z) bottom-north corner
    beta = 0.0
    for _ in range(50):
        sb = nb - th * np.array([math.cos(beta), math.sin(beta)])
        s_ = (DIVIDER_TOP - sb[1]) / math.cos(beta)
        y_at = sb[0] - s_ * math.sin(beta)
        beta += (y_at - (DIVIDER_N - 0.0003)) / s_ * 0.8
    R = _frame(beta)                                                        # x up the bag, z its north face normal
    cyz = nb + hx * np.array([-math.sin(beta), math.cos(beta)]) - t * np.array([math.cos(beta), math.sin(beta)])
    pos = np.array([STAND_X, cyz[0], cyz[1]])
    r = THREAD_R
    face = lambda u, v, h=0.0: pos + R @ np.array([u, v, t + r + h])         # noqa: E731 on the north face
    wv = -hy + 0.003                                                        # west edge (bag y axis = +x world)
    a = face(hx - 0.004, wv, 0.0002)
    down = face(-hx + 0.031, wv, 0.0002)                                    # where the face meets the pile's edge
    zp = PILE_TOP + r + 0.0002
    on = np.array([a[0], PILE_FRONT + 0.003, zp])
    def poly(k):
        return np.array([a, down, on, [a[0], on[1] + 0.012, zp], [a[0] + 0.012 * k, on[1] + 0.03, zp],
                         [a[0] + 0.03 * k, on[1] + 0.04, zp], [a[0] + 0.05 * k, on[1] + 0.035, zp]])
    lo, hi = 0.0, 4.0
    for _ in range(60):
        k = 0.5 * (lo + hi)
        Lp = np.linalg.norm(np.diff(poly(k), axis=0), axis=1).sum()
        lo, hi = (k, hi) if Lp < N_SEG * 0.008 else (lo, k)
    W = poly(k)
    sW = np.r_[0, np.cumsum(np.linalg.norm(np.diff(W, axis=0), axis=1))]
    P = np.array([[np.interp(si, sW, W[:, j]) for j in range(3)] for si in np.linspace(0, sW[-1], N_SEG + 1)])
    d_ = P[-1] - P[-2]
    d_[2] = 0.0
    d_ /= np.linalg.norm(d_)
    tR = np.c_[d_, np.cross([0.0, 0.0, 1.0], d_), [0.0, 0.0, 1.0]]
    tag_pos = P[-1] + np.array([0.0, 0.0, TAG_HALF[2] - r + 0.0002])
    return pos, R, P, tag_pos, tR


def add(root, bodies):
    world = root.find("worldbody")
    pos, R, P, tag_pos, tag_R = {"rig": rig_pose, "heap": heap_pose, "slab": slab_pose, "bag": bag_pose, "stand": stand_pose}.get(START, start_pose)()
    if START in ("slab", "bag"):
        hb = ET.SubElement(world, "body", name=f"{NAME}_heap", pos="0 0 0")
        lo = np.array([HEAP_LO[0], HEAP_LO[1], HEAP_LO[2]])
        hi = np.array([HEAP_HI[0], HEAP_HI[1], SLAB_TOP - SLAB_T])
        sl = np.array([HEAP_LO[0], HEAP_LO[1] - SLAB_OVER, SLAB_TOP - SLAB_T])
        sh_ = np.array([HEAP_HI[0], HEAP_HI[1], SLAB_TOP])
        for nm, a_, b_ in (("heap_geom", lo, hi), ("slab_geom", sl, sh_)):
            ET.SubElement(hb, "geom", name=f"{NAME}_{nm}", type="box", pos=_fmt((a_ + b_) / 2), size=_fmt((b_ - a_) / 2),
                          rgba="0.85 0.75 0.55 1", contype="1", conaffinity="1", condim="3", group="1",
                          friction="0.5 0.005 0.0001")
    if START == "heap":
        hb = ET.SubElement(world, "body", name=f"{NAME}_heap", pos="0 0 0")
        lo = np.array([HEAP_LO[0], HEAP_LO[1], HEAP_LO[2]])
        hi = np.array([HEAP_HI[0], HEAP_HI[1], HEAP_TOP])
        ET.SubElement(hb, "geom", name=f"{NAME}_heap_geom", type="box", pos=_fmt((lo + hi) / 2), size=_fmt((hi - lo) / 2),
                      rgba="0.85 0.75 0.55 1", contype="1", conaffinity="1", condim="3", group="1",
                      friction="0.5 0.005 0.0001")
    if START == "rig":
        ET.SubElement(world, "geom", name=f"{NAME}_rig_shelf", type="box", size=_fmt(SHELF_HALF),
                      pos=_fmt((SHELF_C[0], SHELF_C[1], SHELF_TOP - SHELF_HALF[2])), rgba="0.3 0.5 0.9 1",
                      contype="1", conaffinity="1", condim="3", group="1", friction="0.5 0.005 0.0001")
    bag = ET.SubElement(world, "body", name=BAG, pos=_fmt(pos), quat=_fmt(_quat(R)))
    ET.SubElement(bag, "freejoint", name=f"{BAG}_joint")
    common = dict(contype="1", conaffinity="1", condim="3", solimp=SOLIMP, group="1")
    # two panels joined along the bag's long centre line by a soft crease (hinge with a weak spring): panel A
    # (+y half) carries the string's staple, panel B (-y half) folds; at rest the bag hangs folded by FOLD
    if START in ("bag", "stand"):
        ET.SubElement(bag, "geom", name=f"{BAG}_geom", type="box", size=_fmt(BAGF_HALF), mass=f"{BAG_MASS}",
                      rgba="0.93 0.90 0.82 1", solref=SOL_BAG, friction=FR_BAG, priority="2", **common)
    half = (BAG_HALF[0], BAG_HALF[1] / 2, BAG_HALF[2])
    if START not in ("bag", "stand"):
      ET.SubElement(bag, "geom", name=f"{BAG}_geom", type="box", size=_fmt(half), pos=_fmt((0, half[1], 0)),
                  mass=f"{BAG_MASS / 2}", rgba="0.93 0.90 0.82 1", solref=SOL_BAG, friction=FR_BAG, priority="2", **common)
    if START not in ("bag", "stand"):
        fold = -FOLD if START in ("rig", "heap", "slab") else 0.0
        pb = ET.SubElement(bag, "body", name=f"{BAG}_b", pos="0 0 0", quat=_fmt((math.cos(fold / 2), math.sin(fold / 2), 0, 0)))
        ET.SubElement(pb, "joint", name=f"{BAG}_crease", type="hinge", axis="1 0 0", stiffness=f"{CREASE_K}",
                      damping=f"{CREASE_D}", armature="1e-7")
        ET.SubElement(pb, "geom", name=f"{BAG}_b_geom", type="box", size=_fmt(half), pos=_fmt((0, -half[1], 0)),
                      mass=f"{BAG_MASS / 2}", rgba="0.93 0.90 0.82 1", solref=SOL_BAG, friction=FR_BAG, priority="2", **common)
    # string: segment i is a body at P[i], its x axis toward P[i+1]; frames carried along without twist
    parent, R_par, p_par = bag, R, pos
    zprev = R[:, 2]
    m_seg = THREAD_MASS / N_SEG
    for i in range(N_SEG):
        x = P[i + 1] - P[i]
        seg_len = float(np.linalg.norm(x))
        x /= seg_len
        z = zprev - (zprev @ x) * x
        if np.linalg.norm(z) < 1e-6:
            z = np.cross(x, [1.0, 0.0, 0.0])
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
        Ri = np.c_[x, y, z]
        b = ET.SubElement(parent, "body", name=f"{NAME}_s{i}", pos=_fmt(R_par.T @ (P[i] - p_par)),
                          quat=_fmt(_quat(R_par.T @ Ri)))
        for ax, nm in (("0 1 0", "b1"), ("0 0 1", "b2")):
            ET.SubElement(b, "joint", name=f"{NAME}_s{i}_{nm}", type="hinge", axis=ax, damping=f"{THREAD_DAMP}",
                          armature=f"{THREAD_ARM}")
        ET.SubElement(b, "geom", name=f"{NAME}_s{i}_geom", type="capsule", fromto=_fmt((0, 0, 0, seg_len, 0, 0)),
                      size=f"{THREAD_R}", mass=f"{m_seg}", rgba="0.96 0.96 0.94 1", solref=SOL_THREAD,
                      friction=FR_THREAD, priority="1", **common)
        parent, R_par, p_par, zprev = b, Ri, P[i], z
    tag = ET.SubElement(parent, "body", name=TAG, pos=_fmt(R_par.T @ (tag_pos - p_par)),
                        quat=_fmt(_quat(R_par.T @ tag_R)))
    # staple: bends like the string, no twist (the string has none either, so the tag cannot spin freely)
    for ax, nm in (("0 1 0", "b1"), ("0 0 1", "b2")):
        ET.SubElement(tag, "joint", name=f"{TAG}_{nm}", type="hinge", axis=ax, damping="2e-5", armature="2e-8")
    ET.SubElement(tag, "geom", name=f"{TAG}_geom", type="box", pos=_fmt(TAG_CENTRE), size=_fmt(TAG_HALF),
                  mass=f"{TAG_MASS}", rgba="0.85 0.12 0.10 1", solref=SOL_TAG, friction=FR_TAG, priority="1",
                  **common)


if __name__ == "__main__":
    pos, R, P, tp, tR = start_pose()
    print("bag", np.round(pos, 4), "lean south deg", round(math.degrees(math.atan2(-R[1, 0], R[2, 0])), 2),
          "top edge z", round(float((pos + R @ [BAG_HALF[0], 0, 0])[2]), 4))
    print("string length", round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), 4),
          "segment lengths", np.round(np.linalg.norm(np.diff(P, axis=0), axis=1) * 1000, 1))
    print("string pts", np.round(P, 4).tolist())
    print("tag staple", np.round(tp, 4), "lean deg", round(math.degrees(math.atan2(-tR[1, 0], tR[2, 0])), 1),
          "tag top", np.round(tp + tR @ np.array([TAG_CENTRE[0] + TAG_HALF[0], 0, 0]), 4))
