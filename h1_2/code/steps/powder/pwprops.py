"""Milk powder scene additions, session layer only. Nothing here touches a source USD.

object_0309 in the twin is ONE rigid body with ONE revolved visual mesh and ONE convex hull: there is no lid body.
What this module adds (geometry from the MuJoCo-era props_powder.py, measured from the twin):
  jar shell    solid hull disabled, replaced by floor, 16 wall boxes, shoulder ring, neck ring (opening 66 mm across).
  rigid fill   the packed bulk of the powder: a cylinder and a funnel crater, part of the jar body, 0.25 kg added.
  LID          a NEW free rigid body /World/Props/jar_lid: 30 g, top disc + 20-box skirt, RESTING FIT on the neck rim with 3 mm
               radial clearance, with an ADDED 30 x 20 x 48 mm grab knob on top. No joint, no thread: nothing holds it but gravity and contact. The twin's real jar has a
               screw lid; the screw is NOT modelled here.
  mug          object_0316 authored at the counter spot before physics loads (this step is run on its own), hollowed with
               fluids.Fluids._hollow (read only, that folder is not changed).
Everything is authored before physics loads. No welds, no state writes, no forces.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))    # release_paths.py lives at the code root
import release_paths as RP
import math
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, RP.code('props'))
import numpy as np
import props as P
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

JAR = "/World/Objects/object_0309"
MUG = "/World/Objects/object_0316"
LID = "/World/Props/jar_lid"
JAR_XY, TOP = (1.90, 2.36), 0.86
MUG_XY = (1.90, 2.17)
R_OUT, R_IN = 0.045, 0.041
NECK_IN, NECK_OUT, NECK_TOP = 0.033, 0.0355, 0.1285
LID_Z0, LID_Z1, LID_TOP_T, LID_R_IN, LID_R_OUT, LID_MASS = 0.118, 0.132, 0.0035, 0.0385, 0.0400, 0.030     # 3 mm radial clearance: a tighter fit jammed on the neck (run l4)
FILL_TOP, FUNNEL = 0.090, (0.012, 0.041, 0.090, 0.100)    # a nearly full jar: the straight handle must clear the rim while the bowl is in the powder
FILL_MASS = 0.25
KNOB_HALF = (0.015, 0.010, 0.024)      # ADDED grab knob on the lid, flat faces: the hand cannot span the 77 mm rim without shoving the lid


def prism(st, path, r0, r1, z0, z1, mat, n=20, xy=(0.0, 0.0), offset=0.001, visible=False):
    """Convex-hull collider: frustum, r0 at z0, r1 at z1."""
    pts = [(xy[0] + r * math.cos(2 * math.pi * k / n), xy[1] + r * math.sin(2 * math.pi * k / n), z) for r, z in ((r0, z0), (r1, z1)) for k in range(n)]
    faces, counts = [], []
    for k in range(n):
        faces += [k, (k + 1) % n, n + (k + 1) % n, n + k]
        counts.append(4)
    faces += list(range(n - 1, -1, -1)) + list(range(n, 2 * n))
    counts += [n, n]
    m = UsdGeom.Mesh.Define(st, path)
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(faces)
    m.CreateSubdivisionSchemeAttr("none")
    p = m.GetPrim()
    P._collider(p, offset)
    UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
    if mat is not None:
        P._bind(p, mat)
    if not visible:
        m.CreatePurposeAttr("guide")
    return p


def ring(st, path, r_in, r_out, z0, z1, n, mat, purpose="guide"):
    rc, ht = (r_in + r_out) / 2, (r_out - r_in) / 2
    half_t = r_out * math.tan(math.pi / n) * 1.02
    for k in range(n):
        a = 2 * math.pi * k / n
        q = Gf.Quatd(math.cos(a / 2), 0, 0, math.sin(a / 2))
        P._box(st, f"{path}_{k}", (rc * math.cos(a), rc * math.sin(a), (z0 + z1) / 2), (ht, half_t, (z1 - z0) / 2), mat, quat=q, purpose=purpose)


def funnel(st, path, n, mat):
    r0, r1, z0, z1 = FUNNEL
    slope = math.atan2(z1 - z0, r1 - r0)
    L, t = math.hypot(r1 - r0, z1 - z0), 0.004
    rc, zc = (r0 + r1) / 2, (z0 + z1) / 2
    half_t = r1 * math.tan(math.pi / n) * 1.05
    for k in range(n):
        a = 2 * math.pi * k / n
        qa = Gf.Quatd(math.cos(a / 2), 0, 0, math.sin(a / 2))
        qs = Gf.Quatd(math.cos(-slope / 2), 0, math.sin(-slope / 2), 0)
        nrm = np.array([-math.sin(slope) * math.cos(a), -math.sin(slope) * math.sin(a), math.cos(slope)])
        c = np.array([rc * math.cos(a), rc * math.sin(a), zc]) - nrm * t / 2
        P._box(st, f"{path}_{k}", c, (L / 2, half_t, t / 2), mat, quat=qa * qs, purpose="guide")


def revolve(st, path, profile, n=48, xy=(0.0, 0.0), rgb=None):
    """Visual mesh from an (r, z) profile."""
    pts, faces, counts = [], [], []
    m_ = len(profile)
    for k in range(n):
        a = 2 * math.pi * k / n
        for r, z in profile:
            pts.append((xy[0] + r * math.cos(a), xy[1] + r * math.sin(a), z))
    for k in range(n):
        for i in range(m_ - 1):
            a0, a1 = k * m_ + i, ((k + 1) % n) * m_ + i
            faces += [a0, a1, a1 + 1, a0 + 1]
            counts.append(4)
    me = UsdGeom.Mesh.Define(st, path)
    me.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    me.CreateFaceVertexCountsAttr(counts)
    me.CreateFaceVertexIndicesAttr(faces)
    me.CreateSubdivisionSchemeAttr("none")
    if rgb:
        me.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    return me


def install_jar(st, with_lid=True):
    jar = st.GetPrimAtPath(JAR)
    UsdPhysics.CollisionAPI(st.GetPrimAtPath(JAR + "/collision_00")).CreateCollisionEnabledAttr().Set(False)
    glass = P._material(st, "jar_glass", 0.5, (0.8, 0.88, 0.88))
    fill = P._material(st, "jar_fill", 0.7, (0.95, 0.93, 0.86), rough=0.9)
    steel = P._material(st, "lid_steel", 0.5, (0.88, 0.88, 0.90), metallic=0.6, rough=0.4)
    g = JAR + "/shell"
    UsdGeom.Xform.Define(st, g)
    prism(st, g + "/floor", R_OUT - 0.001, R_OUT - 0.001, 0.0, 0.003, glass)
    ring(st, g + "/wall", R_IN, R_OUT, 0.003, 0.099, 16, glass)
    ring(st, g + "/shoulder", NECK_IN, R_OUT, 0.097, 0.103, 16, glass)
    ring(st, g + "/neck", NECK_IN, NECK_OUT, 0.103, NECK_TOP, 16, glass)
    prism(st, g + "/fill", R_IN - 0.0005, R_IN - 0.0005, 0.003, FILL_TOP, fill)
    funnel(st, g + "/funnel", 12, fill)
    UsdPhysics.MassAPI(jar).GetMassAttr().Set(0.3 + FILL_MASS)
    # visual: the twin's one revolved mesh includes the lid: keep the jar below the lid line, the lid is its own mesh
    prof = [(0.0, 0.0), (0.04, 0.0), (0.044, 0.006), (0.045, 0.095), (0.04, 0.104), (0.036, 0.108), (NECK_OUT, NECK_TOP), (NECK_IN, NECK_TOP), (NECK_IN, 0.104), (R_IN, 0.098), (R_IN, FUNNEL[3])]
    UsdGeom.Imageable(st.GetPrimAtPath(JAR + "/visual")).MakeInvisible()
    b = revolve(st, JAR + "/visual_open", prof, rgb=(0.8, 0.88, 0.88))
    UsdShade.MaterialBindingAPI.Apply(b.GetPrim()).Bind(UsdShade.Material(st.GetPrimAtPath("/World/Materials/object_0309")))
    pv = revolve(st, JAR + "/visual_fill", [(FUNNEL[1], FUNNEL[3]), (FUNNEL[0], FUNNEL[2]), (0.0, FUNNEL[2])], rgb=(0.95, 0.93, 0.86))
    UsdShade.MaterialBindingAPI.Apply(pv.GetPrim()).Bind(UsdShade.Material(st.GetPrimAtPath(P.ROOT + "/Looks/jar_fill")))
    if not with_lid:
        return
    z0 = TOP + NECK_TOP                                     # underside of the lid's top disc rests on the neck rim
    P._body(st, LID, (JAR_XY[0], JAR_XY[1], z0), Gf.Quatd(1, 0, 0, 0), LID_MASS)
    prism(st, LID + "/top", LID_R_OUT, LID_R_OUT, 0.0, LID_TOP_T, steel)
    ring(st, LID + "/skirt", LID_R_IN, LID_R_OUT, LID_Z0 - NECK_TOP, 0.0, 20, steel)
    P._box(st, LID + "/knob", (0, 0, LID_TOP_T + KNOB_HALF[2]), KNOB_HALF, steel)     # visible and collides: an added grab knob
    v = revolve(st, LID + "/visual", [(LID_R_IN, LID_Z0 - NECK_TOP), (LID_R_OUT, LID_Z0 - NECK_TOP), (LID_R_OUT, 0.0), (LID_R_OUT - 0.003, LID_TOP_T), (0.0, LID_TOP_T)], rgb=(0.88, 0.88, 0.90))
    UsdShade.MaterialBindingAPI.Apply(v.GetPrim()).Bind(UsdShade.Material(st.GetPrimAtPath(P.ROOT + "/Looks/lid_steel")))


def install_mug(st, hollow=True):
    """Mug 0316 at the counter spot (the mug step's end state), hollow so grains can lie in it."""
    m = st.GetPrimAtPath(MUG)
    xf = UsdGeom.Xformable(m)
    q = xf.GetOrderedXformOps()[1].Get()
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(MUG_XY[0], MUG_XY[1], TOP))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q))
    if hollow:
        sys.path.insert(0, RP.code('fluids'))
        import fluids
        F = fluids.Fluids()
        ms = fluids.wl.mug_shapes()
        ms[-1] = ("cyl", (0, 0, fluids.wl.MUG_FLOOR / 2), 0.031, fluids.wl.MUG_FLOOR / 2)
        F._hollow(st, MUG, ms, 0.03, "mug")


def install(st, lid=True, mug=True):
    install_jar(st, with_lid=lid)
    if mug:
        install_mug(st)
GRAIN_R = 0.0025


def grains_pre(n, damp=8.0, spread=0.016, sleep=0.0005):
    """Powder lumps via fluids.Fluids.add_grains (read only): free rigid spheres in the crater. PhysX has no rolling friction, so each
    grain also gets angular damping, a stand-in for powder cohesion. Returns the pre-physics hook."""
    def f(st):
        from pxr import Usd
        sys.path.insert(0, RP.code('fluids'))
        import fluids
        F = fluids.Fluids()
        F.add_grains(st, (JAR_XY[0], JAR_XY[1], TOP + FUNNEL[2] + 0.0045), n=n, r=GRAIN_R, spread=spread)
        for p in Usd.PrimRange(st.GetPrimAtPath("/World/fluid/grains")):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                b = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
                b.CreateAngularDampingAttr(damp)
                b.CreateSolverPositionIterationCountAttr(8)
                b.CreateSleepThresholdAttr(sleep)
    return f


# ---- fallback tool: the spoon's 1.8 mm handle cannot be carried through a wrist turn (runs g8, g9), so powder goes by a small cup with the
# same flat grab block as the lid knob. ADDED prop: a free rigid body, nothing joins it to anything.
SCOOP = "/World/Props/powder_scoop"
SCOOP_XY = (1.74, 2.30)
SC_R, SC_H, SC_T = 0.020, 0.028, 0.0012
SC_BLOCK = (0.0362, 0.0, 0.036)
REAL_SCOOP = os.environ.get("P_SCOOP") == "real"       # moulded scoop with a long handle (pw_scoop.py). Off until it is proven: the old tool stays the default


def install_scoop(st):
    if REAL_SCOOP:
        import pw_scoop
        return pw_scoop.install(st, P, sys.modules[__name__])
    mat = P._material(st, "scoop_steel", 0.5, (0.85, 0.86, 0.88), metallic=0.6, rough=0.4)
    P._body(st, SCOOP, (SCOOP_XY[0], SCOOP_XY[1], TOP + 0.0005), Gf.Quatd(1, 0, 0, 0), 0.030)
    prism(st, SCOOP + "/floor", SC_R + SC_T, SC_R + SC_T, 0.0, 0.002, mat, visible=True)
    ring(st, SCOOP + "/wall", SC_R, SC_R + SC_T, 0.0, SC_H, 16, mat, purpose="default")
    P._box(st, SCOOP + "/block", SC_BLOCK, KNOB_HALF, mat)


if REAL_SCOOP:
    sys.path.insert(0, RP.code('steps', 'powder'))
    import pw_scoop as _ps
    SC_BLOCK = (_ps.L, 0.0, _ps.centre_z(_ps.L))      # now the middle of the handle grip
