"""Tea bag, stir spoon and powder spoon as real PhysX rigid bodies in the H1-2 tea room. Session layer only.

Sizes, masses, friction and start poses come from the MuJoCo-era modules (props_teabag.py stand_pose, props_spoon.py,
props_pspoon.py). No welds, no state writes, no forces: every prop is a free body held up by contacts.

Use (h12_physx.Sim authors and loads physics in one go, so extras must go in before it loads):

    import props
    s = props.make_sim(app, xy=(0.62, 1.15))      # same arguments as H.Sim; adds the props
    paths = props.prop_paths()                     # {"teabag_bag":..., "teabag_tag":..., "stir_spoon":..., "powder_spoon":...}

Notes
  * string: bag and tag are joined by a UsdPhysics.DistanceJoint with a max length only (a limp inextensible 12 cm string,
    slack below that). There is no capsule chain and no string geometry: nothing can snag on it.
  * tubs: object_0306 / object_0307 are solid blocks in the twin. As tea_sim.fix_tub did, their solid collider is disabled in
    the session layer and replaced by a 2 mm floor and walls; object_0306 is 90 mm tall as in the MuJoCo scene.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))    # release_paths.py lives at the code root
import release_paths as RP
import math
import sys

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

sys.path.insert(0, RP.code())
ROOT = "/World/Props"
STEEL = 7900.0
# spoons (spoon frame: x from bowl tip to handle end, z the bowl's open side)
HZ, HT = 0.003, 0.0009
SEGMENTS = (("neck", 0.030, 0.050, 0.0025), ("handle1", 0.050, 0.085, 0.0030), ("handle2", 0.085, 0.110, 0.0040), ("handle3", 0.110, 0.140, 0.0050))
LENGTH = 0.140
BOWL_MASS = 0.0060
REST_POS = (2.063, 2.3837, 0.8681)              # bowl tip on the FRESH tub's floor, leaning on the tub's front rim
REST_QUAT = (0.5877, -0.3924, -0.3924, -0.5888)  # w x y z
STIR_DX, POWDER_DX = +0.022, -0.022             # two spoons side by side in the one tall tub
DISH_C, A_B, B_B, Z_B, A_T, B_T, Z_T = 0.0175, 0.0105, 0.0060, -0.0025, 0.0170, 0.0122, 0.0030
PLATE_T, WALL_T, N_WALL = 0.0012, 0.0008, 12
# tea bag
BAGF_HALF = (0.0325, 0.0125, 0.002)             # folded bag 6.5 x 2.5 x 0.4 cm
TAG_HALF = (0.0125, 0.010, 0.0005)              # card 2.5 x 2.0 x 0.1 cm
TAG_CENTRE = (0.0105, 0.0, 0.0)                 # staple = tag body origin
BAG_MASS, TAG_MASS, STRING_MASS = 0.0020, 0.0005, 0.0001
STRING_LEN = 0.12
# measured in this room (tray_raw.json): tray floor top, pile front face, front divider north face and top
FLOOR, PILE_FRONT, DIVIDER_N, DIVIDER_TOP, PILE_TOP = 0.8655, 2.2387, 2.2307, 0.9175, 0.8955
STAND_X = 2.303
TAG_XY = (2.345, 2.281)


def q_from_R(R):
    """Rotation matrix (columns = body axes in world) -> Gf.Quatd."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        w, (x, y, z) = (R[k, j] - R[j, k]) / s, v
    return Gf.Quatd(float(w), float(x), float(y), float(z))


def _material(st, name, mu, rgb, metallic=0.0, rough=0.5):
    path = f"{ROOT}/Looks/{name}"
    m = UsdShade.Material.Define(st, path)
    sh = UsdShade.Shader.Define(st, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    pm = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    pm.CreateStaticFrictionAttr(mu)
    pm.CreateDynamicFrictionAttr(mu)
    pm.CreateRestitutionAttr(0.0)
    return m


def _bind(prim, mat):
    b = UsdShade.MaterialBindingAPI.Apply(prim)
    b.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    b.Bind(mat)


def _collider(prim, offset=0.001):
    UsdPhysics.CollisionAPI.Apply(prim)
    c = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    c.CreateContactOffsetAttr(offset)
    c.CreateRestOffsetAttr(0.0)


def _box(st, path, pos, half, mat, quat=None, purpose=None, offset=0.001):
    c = UsdGeom.Cube.Define(st, path)
    c.CreateSizeAttr(2.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    c.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat if quat is not None else Gf.Quatd(1, 0, 0, 0))
    c.AddScaleOp().Set(Gf.Vec3d(*map(float, half)))
    p = c.GetPrim()
    _collider(p, offset)
    if mat is not None:
        _bind(p, mat)
    if purpose:
        c.CreatePurposeAttr(purpose)
    return p


def _body(st, path, pos, quat, mass):
    x = UsdGeom.Xform.Define(st, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat)
    p = x.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(float(mass))
    r = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
    r.CreateSolverPositionIterationCountAttr(16)
    r.CreateSolverVelocityIterationCountAttr(1)
    r.CreateMaxDepenetrationVelocityAttr(0.5)
    return p


def _ellipsoid_mesh(st, path, c, a, mat, n=10, m=6):
    """Convex-hull mesh ellipsoid (PhysX has no non-uniformly scaled sphere)."""
    pts, faces, counts = [], [], []
    for i in range(m + 1):
        th = math.pi * i / m
        for j in range(n):
            ph = 2 * math.pi * j / n
            pts.append((c[0] + a[0] * math.sin(th) * math.cos(ph), c[1] + a[1] * math.sin(th) * math.sin(ph), c[2] + a[2] * math.cos(th)))
    for i in range(m):
        for j in range(n):
            q = (i * n + j, i * n + (j + 1) % n, (i + 1) * n + (j + 1) % n, (i + 1) * n + j)
            faces += q
            counts.append(4)
    me = UsdGeom.Mesh.Define(st, path)
    me.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    me.CreateFaceVertexCountsAttr(counts)
    me.CreateFaceVertexIndicesAttr(faces)
    me.CreateSubdivisionSchemeAttr("none")
    p = me.GetPrim()
    _collider(p)
    UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
    _bind(p, mat)
    return p


def _handle(st, path, mat):
    vol = 0.0
    for n, x0, x1, hw in SEGMENTS:
        _box(st, f"{path}/{n}", ((x0 + x1) / 2, 0, HZ), ((x1 - x0) / 2, hw, HT), mat)
        vol += (x1 - x0) * 2 * hw * 2 * HT
    return vol * STEEL


def dish_walls():
    """Powder spoon's dish walls in the spoon frame: (centre, R, half size), as props_pspoon.dish_walls."""
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
        n = np.cross(t, u)
        if n @ np.array([math.cos(pm) * A_T, math.sin(pm) * B_T, 0.0]) < 0:
            n, t = -n, -t                                  # keep (u, t, n) right-handed
        seg = math.hypot(A_T * (math.cos(p1) - math.cos(p0)), B_T * (math.sin(p1) - math.sin(p0)))
        out.append(((B + T) / 2 + n * WALL_T / 2, np.c_[u, t, n], (hl + 0.0006, seg / 2 * 1.18, WALL_T / 2)))
    return out


def bag_stand_pose():
    """props_teabag.stand_pose for this room's measured tray: bag centre and rotation (x up the bag, z its north normal)."""
    hx, hy, t = BAGF_HALF
    th = 2 * t
    nb = np.array([PILE_FRONT - 0.0003, FLOOR + 0.0004])
    beta = 0.0
    for _ in range(50):
        sb = nb - th * np.array([math.cos(beta), math.sin(beta)])
        s_ = (DIVIDER_TOP - sb[1]) / math.cos(beta)
        beta += (sb[0] - s_ * math.sin(beta) - (DIVIDER_N - 0.0003)) / s_ * 0.8
    x = np.array([0.0, -math.sin(beta), math.cos(beta)])
    z = np.array([0.0, math.cos(beta), math.sin(beta)])
    R = np.c_[x, np.cross(z, x), z]
    cyz = nb + hx * np.array([-math.sin(beta), math.cos(beta)]) - t * np.array([math.cos(beta), math.sin(beta)])
    return np.array([STAND_X, cyz[0], cyz[1]]), R


def _open_tub(st, oid, hx, hy, h, t=0.002):
    """Replace a tub's solid hull with a floor and four walls (same rigid body, session layer)."""
    tub = next(p for p in st.Traverse() if p.GetName() == oid and p.GetTypeName() == "Xform")
    path = tub.GetPath().pathString
    UsdPhysics.CollisionAPI(st.GetPrimAtPath(path + "/collision_00")).CreateCollisionEnabledAttr().Set(False)
    mat = _material(st, f"tub_{oid}", 0.5, (0.45, 0.46, 0.48))
    _box(st, f"{path}/open_floor", (0, 0, t / 2), (hx, hy, t / 2), mat, purpose="guide")
    for nm, pos, half in (("wx0", (-hx + t / 2, 0, h / 2), (t / 2, hy, h / 2)), ("wx1", (hx - t / 2, 0, h / 2), (t / 2, hy, h / 2)),
                          ("wy0", (0, -hy + t / 2, h / 2), (hx, t / 2, h / 2)), ("wy1", (0, hy - t / 2, h / 2), (hx, t / 2, h / 2))):
        _box(st, f"{path}/open_{nm}", pos, half, mat, purpose="guide")


def _spoon_quat():
    w, x, y, z = REST_QUAT
    q = Gf.Quatd(w, x, y, z)
    return q.GetNormalized()


def install(st):
    """Author everything in the stage's current edit target (Sim leaves that on the session layer)."""
    UsdGeom.Xform.Define(st, ROOT)
    steel = _material(st, "steel", 0.4, (0.78, 0.79, 0.81), metallic=1.0, rough=0.3)
    paper = _material(st, "bag_paper", 0.6, (0.93, 0.90, 0.82), rough=0.9)
    card = _material(st, "tag_card", 0.9, (0.85, 0.12, 0.10), rough=0.7)
    _open_tub(st, "object_0306", 0.048, 0.0363, 0.090)
    _open_tub(st, "object_0307", 0.047, 0.0363, 0.045)

    # stir spoon: solid ellipsoid bowl (props_spoon)
    p = f"{ROOT}/stir_spoon"
    handle_mass = sum((x1 - x0) * 2 * hw * 2 * HT for _, x0, x1, hw in SEGMENTS) * STEEL
    _body(st, p, (REST_POS[0] + STIR_DX, REST_POS[1], REST_POS[2]), _spoon_quat(), BOWL_MASS + handle_mass)
    _ellipsoid_mesh(st, p + "/bowl", (0.0175, 0, 0), (0.0175, 0.0125, 0.0040), steel)
    _handle(st, p, steel)

    # powder spoon: a dish that can carry grains (props_pspoon)
    p = f"{ROOT}/powder_spoon"
    walls = dish_walls()
    mass = handle_mass + STEEL * (2 * A_B) * (2 * B_B) * PLATE_T + sum(STEEL * 8 * h[0] * h[1] * h[2] for _, _, h in walls)
    _body(st, p, (REST_POS[0] + POWDER_DX, REST_POS[1], REST_POS[2]), _spoon_quat(), mass)
    _box(st, p + "/bowl", (DISH_C, 0, Z_B - PLATE_T / 2), (A_B + 0.0008, B_B + 0.0008, PLATE_T / 2), steel)
    for k, (c, R, h) in enumerate(walls):
        _box(st, f"{p}/bowl_w{k}", c, h, steel, quat=q_from_R(R))
    _handle(st, p, steel)

    # tea bag, tag, string
    pos, R = bag_stand_pose()
    pb = f"{ROOT}/teabag_bag"
    _body(st, pb, pos, q_from_R(R), BAG_MASS + STRING_MASS / 2)
    _box(st, pb + "/geom", (0, 0, 0), BAGF_HALF, paper)
    pt = f"{ROOT}/teabag_tag"
    tag_z = PILE_TOP + TAG_HALF[2] + 0.0003
    _body(st, pt, (TAG_XY[0], TAG_XY[1], tag_z), Gf.Quatd(1, 0, 0, 0), TAG_MASS + STRING_MASS / 2)
    _box(st, pt + "/geom", TAG_CENTRE, TAG_HALF, card)
    j = UsdPhysics.DistanceJoint.Define(st, f"{ROOT}/teabag_string")
    j.CreateBody0Rel().SetTargets([pb])
    j.CreateBody1Rel().SetTargets([pt])
    j.CreateLocalPos0Attr(Gf.Vec3f(BAGF_HALF[0] - 0.004, -BAGF_HALF[1] + 0.003, BAGF_HALF[2]))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateMinDistanceAttr(-1.0)                          # negative = no minimum: the string can go slack
    j.CreateMaxDistanceAttr(STRING_LEN)
    return prop_paths()


def prop_paths():
    return {"teabag_bag": f"{ROOT}/teabag_bag", "teabag_tag": f"{ROOT}/teabag_tag", "stir_spoon": f"{ROOT}/stir_spoon", "powder_spoon": f"{ROOT}/powder_spoon"}


def make_sim(app, pre=(), add_props=True, sim_class=None, **kw):
    """H.Sim with extra authoring done just before physics loads. pre: callables f(stage), run after the props are added.
    Works by wrapping H.get_physx_interface for the duration of Sim.__init__, so h12_physx.py itself stays untouched."""
    import h12_physx as H
    import omni.usd
    orig = H.get_physx_interface

    class Hook:
        def __init__(self):
            self._i = orig()

        def __getattr__(self, k):
            return getattr(self._i, k)

        def force_load_physics_from_usd(self):
            st = omni.usd.get_context().get_stage()
            if add_props:
                install(st)
            for f in pre:
                f(st)
            self._i.force_load_physics_from_usd()

    H.get_physx_interface = Hook
    try:
        return (sim_class or H.Sim)(app, **kw)
    finally:
        H.get_physx_interface = orig
