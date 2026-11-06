"""Moulded one-piece measuring scoop for the milk powder step. ADDED prop (replaces the cup plus box tool): session layer only, no source USD touched.
Visual: smooth mesh, bowl (rounded bottom, thin wall, rolled rim, inside and outside) and a long tapered handle growing out of the rim.
Collision: simple convex pieces that follow the visual: the same floor and 16-wall bowl as the old cup (the dose came from it) and convex handle segments.
Local frame: bowl axis +z, origin at the centre of the bowl base, handle along +x. GRIP is the middle of the thick part of the handle."""
import math
import os

import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Vt

R_IN, H_IN, WALL = 0.020, 0.028, 0.0016      # inner size as the old cup
L = float(os.environ.get("P_HANDLE", 0.130))  # bowl axis to the middle of the grip
GRIP_HALF = 0.030                             # thick part of the handle, half length
TAIL = 0.040                                  # handle beyond the grip: the tail rests on the counter and holds the grip up where fingers can reach round it
MASS = 0.045
MODE = os.environ.get("P_SPAWN_MODE", "foot")  # foot: level on a moulded rest at the tail. overhang: handle over the counter front edge. tub: stands in an added open tub
SEG = 96
RGB = (0.80, 0.79, 0.74)


def centre_z(x):
    """Handle centre line: leaves the rim and curves gently upward."""
    u = max(0.0, (x - R_IN) / (L + TAIL - R_IN))
    return 0.0235 + 0.020 * u ** 1.6


def section(x):
    """Half width (y), half height (z) of the handle at x: flared neck into the rim, slim shaft, thick grip, rounded end."""
    end = L + TAIL
    neck = math.exp(-((x - R_IN) / 0.016) ** 2)
    g = 0.5 * (1 + math.tanh((x - (L - GRIP_HALF - 0.006)) / 0.008))
    w = 0.0075 + 0.0075 * neck + 0.0020 * g
    h = 0.0028 + 0.0012 * neck + 0.0092 * g
    r = 1.0
    if x > end - 0.012:
        r = math.sqrt(max(0.0, 1 - ((x - (end - 0.012)) / 0.012) ** 2))
    return w * max(r, 0.02), h * max(r, 0.02)


def bowl_profile():
    """(r, z) from the centre of the outside base, round the rim, to the centre of the inside base."""
    ro, ri, f = R_IN + WALL, R_IN, 0.0065
    p = [(0.0, 0.0), (ro - f, 0.0)]
    p += [(ro - f + f * math.sin(a), f - f * math.cos(a)) for a in np.linspace(0, math.pi / 2, 10)[1:]]
    p += [(ro, z) for z in np.linspace(f, H_IN - 0.0012, 6)[1:]]
    c, b = (ri + WALL / 2, H_IN - 0.0004), WALL * 0.72
    p += [(c[0] + b * math.cos(a), c[1] + b * math.sin(a)) for a in np.linspace(0.0, math.pi, 12)]
    fi = 0.0052
    p += [(ri, z) for z in np.linspace(H_IN - 0.0012, WALL + fi, 6)[1:]]
    p += [(ri - fi + fi * math.cos(a), WALL + fi - fi * math.sin(a)) for a in np.linspace(0, math.pi / 2, 10)[1:]]
    p += [(0.0, WALL)]
    return p


def bowl_mesh():
    prof = bowl_profile()
    m = len(prof)
    pts, faces = [], []
    for k in range(SEG):
        a = 2 * math.pi * k / SEG
        pts += [(r * math.cos(a), r * math.sin(a), z) for r, z in prof]
    for k in range(SEG):
        for i in range(m - 1):
            a0, a1 = k * m + i, ((k + 1) % SEG) * m + i
            faces.append((a0, a1, a1 + 1, a0 + 1))
    return np.array(pts), faces


def handle_ring(x, n=28, wrap=False):
    w, h = section(x)
    zc = centre_z(x)
    e = 0.62                                   # superellipse: flat sides with soft edges
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    y = w * np.sign(c) * np.abs(c) ** e
    z = h * np.sign(s) * np.abs(s) ** e
    if MODE == "foot":                         # the tail deepens downward into a rest, so the scoop stands level with the grip up off the counter
        end = L + TAIL
        u = min(1.0, max(0.0, (x - (end - 0.030)) / 0.016))
        z = np.where(s < 0, z - (zc - h - 0.0005) * u * u * (3 - 2 * u), z)
        y = y * (1 - 0.42 * u * u * (3 - 2 * u))   # slim heel, narrower than the grip
    xs = np.full(n, x)
    if wrap:                                   # the root follows the round of the bowl, so the neck grows out of it with no corners
        ro = R_IN + WALL
        xs = xs + (np.sqrt(np.maximum(ro * ro - y * y, 0.0)) - ro) * math.exp(-((x - R_IN) / 0.010) ** 2)
    return np.c_[xs, y, zc + z]


def handle_mesh(x0=None, n_along=90, n=28):
    x0 = R_IN + WALL * 0.3 if x0 is None else x0
    end = L + TAIL
    xs = np.r_[np.linspace(x0, end - 0.012, n_along), end - 0.012 + 0.012 * np.sin(np.linspace(0, math.pi / 2, 12))[1:]]
    pts, faces = [], []
    for x in xs:
        pts += handle_ring(x, n, wrap=True).tolist()
    for i in range(len(xs) - 1):
        for k in range(n):
            a, b = i * n + k, i * n + (k + 1) % n
            faces.append((a, b, b + n, a + n))
    tip = len(pts)
    pts.append([xs[-1] + 1e-4, 0.0, centre_z(xs[-1])])
    last = (len(xs) - 1) * n
    for k in range(n):
        faces.append((last + k, last + (k + 1) % n, tip))
    return np.array(pts), faces


def smooth_normals(pts, faces):
    nrm = np.zeros_like(pts)
    for f in faces:
        v = pts[list(f[:3])]
        c = np.cross(v[1] - v[0], v[2] - v[0])
        if len(f) > 3 and not np.any(c):
            v = pts[[f[0], f[2], f[3]]]
            c = np.cross(v[1] - v[0], v[2] - v[0])
        for i in f:
            nrm[i] += c
    n = np.linalg.norm(nrm, axis=1, keepdims=True)
    return nrm / np.where(n > 0, n, 1)


def write_mesh(st, path, pts, faces):
    m = UsdGeom.Mesh.Define(st, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr([int(i) for f in faces for i in f])
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, n)) for n in smooth_normals(pts, faces)]))
    m.SetNormalsInterpolation("vertex")
    m.CreateDisplayColorAttr([Gf.Vec3f(*RGB)])
    return m


def install(st, P, PP):
    """P is props, PP is pwprops. Lying on its tail the grip centre was 12 mm up and the thumb never reached it (run p2_w130); leaning on the jar it slid flat (run p2_gr1)."""
    plastic = P._material(st, "scoop_plastic", 0.5, RGB, metallic=0.0, rough=0.62)
    path = PP.SCOOP
    if MODE == "overhang":
        P._body(st, path, (1.70, 2.125, PP.TOP + 0.002), Gf.Quatd(math.cos(-math.pi / 4), 0, 0, math.sin(-math.pi / 4)), MASS)
    elif MODE == "tub":
        tub = P._material(st, "scoop_tub", 0.5, (0.45, 0.47, 0.5), rough=0.7)
        UsdGeom.Xform.Define(st, P.ROOT + "/scoop_tub")
        PP.prism(st, P.ROOT + "/scoop_tub/floor", 0.034, 0.034, 0.0, 0.003, tub, xy=(1.745, 2.30), visible=True)
        for k in range(24):
            a = 2 * math.pi * k / 24
            P._box(st, f"{P.ROOT}/scoop_tub/w{k}", (1.745 + 0.034 * math.cos(a), 2.30 + 0.034 * math.sin(a), PP.TOP + 0.03), (0.002, 0.0048, 0.03), tub, quat=Gf.Quatd(math.cos(a / 2), 0, 0, math.sin(a / 2)))
        P._body(st, path, (1.740, 2.30, PP.TOP + 0.03), Gf.Quatd(math.cos(math.radians(36)), 0, -math.sin(math.radians(36)), 0), MASS)
    else:
        P._body(st, path, (1.788 - L, 2.30, PP.TOP + 0.001), Gf.Quatd(1, 0, 0, 0), MASS)
    PP.prism(st, path + "/floor", R_IN + WALL, R_IN + WALL, 0.0, 0.002, plastic)
    PP.ring(st, path + "/wall", R_IN, R_IN + WALL, 0.0, H_IN, 16, plastic, purpose="guide")
    edges = np.r_[R_IN + WALL * 0.5, np.linspace(R_IN + 0.012, L - GRIP_HALF - 0.012, 4), L - GRIP_HALF, L, L + GRIP_HALF, L + TAIL - 0.004]
    n = 12
    faces = [[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)] + [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    for i in range(len(edges) - 1):
        pts = np.r_[handle_ring(edges[i], n), handle_ring(edges[i + 1], n)]
        m = UsdGeom.Mesh.Define(st, f"{path}/handle_{i}")
        m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
        m.CreateFaceVertexCountsAttr([len(f) for f in faces])
        m.CreateFaceVertexIndicesAttr([j for f in faces for j in f])
        m.CreateSubdivisionSchemeAttr("none")
        q = m.GetPrim()
        P._collider(q, 0.001)
        UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr("convexHull")
        P._bind(q, plastic)
        m.CreatePurposeAttr("guide")
    bp, bf = bowl_mesh()
    hp, hf = handle_mesh()
    look = UsdShade.Material(st.GetPrimAtPath(P.ROOT + "/Looks/scoop_plastic"))
    for name, a, b in (("visual_bowl", bp, bf), ("visual_handle", hp, hf)):
        v = write_mesh(st, f"{path}/{name}", a, b)
        UsdShade.MaterialBindingAPI.Apply(v.GetPrim()).Bind(look)
    print("REAL SCOOP L", L, "mass", MASS, "bowl faces", len(bf), "handle faces", len(hf), flush=True)
