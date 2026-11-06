"""Countertop electric oven, modelled here to match the user's reference photo: stainless front, wide glass door, long bar handle across the top, wire rack inside, four round knobs on the right panel, side vents, small feet. No brand name.
Physics: a free articulation standing on its feet on the counter, no fixed joint. Door: revolute, 0-90 deg. Knobs: revolute along their axis, 0-270 deg stops. Rack: real bars. Colliders are simple hidden shapes under the visual mesh.
Oven frame: x along the front, y into the oven, z up, origin on the counter under the middle of the front face."""
import math

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from pizza_scene import box, cyl, hide, mat, phys, place

TEX = "E:/stretch_assets/oven_tex/"
FX, FY, CZ = 1.25, 2.10, .86           # front centre on the counter
W_, H_, D_ = .50, .33, .38             # overall width, height, depth
FOOT = .022
X0, X1 = -.245, .125                   # door span
Z0, Z1 = .040, .310                    # door opening band
PX0, PX1 = .135, .245                  # control panel
CX0, CX1, CZ0, CZ1, CY0, CY1 = -.228, .108, .058, .298, .012, .352    # cavity
HY, HZ = -.0012, .042                  # hinge. Was -.006: the door stood 5.5 mm proud of the frame even at 0 deg
DOOR_T = .018
BAR_Z, BAR_Y, BAR_R = .287, -.052, .0105
RACK_Z = .135
RACK = (-.215, .095, .035, .335)       # x0 x1 y0 y1
KNOB_X, KNOB_Z = .190, (.285, .215, .145, .080)
KNOB_NAMES = ("temp", "sel", "func", "timer")
OVEN = "/World/oven"
BODY, DOOR = OVEN + "/body", OVEN + "/door"
KNOBS = tuple(f"{OVEN}/knob_{n}" for n in KNOB_NAMES)


def W(x, y, z):
    return (FX + x, FY + y, CZ + z)


def rbox_data(size, r, seg=3):
    """Rounded box mesh: grid on the six faces pushed out from an inner box. Returns points, normals, faces, st."""
    hx, hy, hz = [s / 2 for s in size]
    r = min(r, hx, hy, hz)
    cs = [np.r_[-h, -h + r * (1 - np.cos(np.linspace(0, math.pi / 2, seg + 1)[1:])), h - r * (1 - np.cos(np.linspace(0, math.pi / 2, seg + 1)[1:]))[::-1], h] for h in (hx, hy, hz)]
    inner = np.array([hx - r, hy - r, hz - r])
    P, Nn, F, S = [], [], [], []
    for ax in range(3):
        u_, v_ = (ax + 1) % 3, (ax + 2) % 3
        for sg in (-1, 1):
            base = len(P)
            A, B = cs[u_], cs[v_]
            for a in A:
                for b in B:
                    p = np.zeros(3)
                    p[ax], p[u_], p[v_] = sg * (hx, hy, hz)[ax], a, b
                    q = np.clip(p, -inner, inner)
                    d = p - q
                    n = d / (np.linalg.norm(d) + 1e-12)
                    P.append(q + n * r)
                    Nn.append(n)
                    S.append(((a + (hx, hy, hz)[u_]) * 3, (b + (hx, hy, hz)[v_]) * 3))
            nb = len(B)
            for i in range(len(A) - 1):
                for j in range(nb - 1):
                    q4 = [base + i * nb + j, base + (i + 1) * nb + j, base + (i + 1) * nb + j + 1, base + i * nb + j + 1]
                    F.append(q4 if sg > 0 else q4[::-1])
    return np.array(P), np.array(Nn), F, np.array(S)


def mesh(st, path, P, Nn, F, S, look_, centre=(0, 0, 0)):
    m = UsdGeom.Mesh.Define(st, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(P, np.float32)))
    m.CreateFaceVertexCountsAttr([len(f) for f in F])
    m.CreateFaceVertexIndicesAttr([i for f in F for i in f])
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(Nn, np.float32)))
    m.SetNormalsInterpolation("vertex")
    if S is not None:
        UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(np.asarray(S, np.float32)))
    place(m, centre)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(look_)
    return m


def rbox(st, path, lo, hi, r, look_):
    size = [b - a for a, b in zip(lo, hi)]
    P, Nn, F, S = rbox_data(size, r)
    return mesh(st, path, P, Nn, F, S, look_, [(a + b) / 2 for a, b in zip(lo, hi)])


def lathe(st, path, prof, look_, n=64, flute=0, centre=(0, 0, 0), axis_y=True):
    """Smooth turned part with per-vertex normals. prof (r, z). flute: grooves round the side, for knob grip. axis_y turns its z onto -y."""
    pr = np.array(prof, float)
    t = np.gradient(pr, axis=0)
    nr = np.c_[t[:, 1], -t[:, 0]]
    nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-12
    P, Nn, F = [], [], []
    k = len(pr)
    for i in range(n):
        a = 2 * math.pi * i / n
        c, s_ = math.cos(a), math.sin(a)
        for q in range(k):
            r, z = pr[q]
            if flute and q in range(2, k - 2):
                r = r * (1 + .035 * math.cos(flute * a))
            P.append((r * c, r * s_, z))
            Nn.append((nr[q, 0] * c, nr[q, 0] * s_, nr[q, 1]))
    for i in range(n):
        j = (i + 1) % n
        for q in range(k - 1):
            F.append([i * k + q, j * k + q, j * k + q + 1, i * k + q + 1])
    P, Nn = np.array(P), np.array(Nn)
    if axis_y:                                # (x, y, z) -> (x, -z, y): model z points out of the front, towards -y
        P = np.c_[P[:, 0], -P[:, 2], P[:, 1]]
        Nn = np.c_[Nn[:, 0], -Nn[:, 2], Nn[:, 1]]
    return mesh(st, path, P, Nn, F, None, look_, centre)


def tube(st, path, pts, r, look_, n=12):
    """Round tube along a polyline, sampled densely by the caller."""
    pts = np.array(pts, float)
    T = np.gradient(pts, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    up = np.array([0, 0, 1.]) if abs(T[0, 2]) < .9 else np.array([0, 1., 0])
    P, Nn, F = [], [], []
    for p, t in zip(pts, T):
        a = np.cross(t, up)
        a /= np.linalg.norm(a) + 1e-12
        b = np.cross(a, t)
        up = b
        for i in range(n):
            th = 2 * math.pi * i / n
            d = a * math.cos(th) + b * math.sin(th)
            P.append(p + d * r)
            Nn.append(d)
    for s_ in range(len(pts) - 1):
        for i in range(n):
            j = (i + 1) % n
            F.append([s_ * n + i, (s_ + 1) * n + i, (s_ + 1) * n + j, s_ * n + j])
    return mesh(st, path, np.array(P), np.array(Nn), F, None, look_)


def surf(st, name, rgb, metal, rough, tex=None, nrm=None, rtex=None, emissive=None):
    m = UsdShade.Material.Define(st, "/World/Looks/" + name)
    sh = UsdShade.Shader.Define(st, f"/World/Looks/{name}/s")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if emissive is not None:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    rd = None
    if tex or nrm or rtex:
        rd = UsdShade.Shader.Define(st, f"/World/Looks/{name}/st")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    def t_(nm, f, cs_, chan, ty, inp, ity, scale=None, bias=None):
        t = UsdShade.Shader.Define(st, f"/World/Looks/{name}/{nm}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f)
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(cs_)
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        if scale:
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        sh.CreateInput(inp, ity).ConnectToSource(t.CreateOutput(chan, ty))

    if tex:
        t_("tex", tex, "sRGB", "rgb", Sdf.ValueTypeNames.Float3, "diffuseColor", Sdf.ValueTypeNames.Color3f)
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    if nrm:
        t_("nrm", nrm, "raw", "rgb", Sdf.ValueTypeNames.Float3, "normal", Sdf.ValueTypeNames.Normal3f, (2, 2, 2, 1), (-1, -1, -1, 0))
    if rtex:
        t_("rgh", rtex, "raw", "r", Sdf.ValueTypeNames.Float, "roughness", Sdf.ValueTypeNames.Float, (rough * 2, rough * 2, rough * 2, 1), (0, 0, 0, 0))
    else:
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


def glass(st, name):
    m = UsdShade.Material.Define(st, "/World/Looks/" + name)
    sh = UsdShade.Shader.Define(st, f"/World/Looks/{name}/s")
    sh.SetSourceAsset("OmniGlass.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.50, .53, .54))
    sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.45)
    sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
    sh.CreateInput("reflection_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
    m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return m


def hbox(st, path, lo, hi, pm, look_):
    hide(box(st, path, [(a + b) / 2 for a, b in zip(lo, hi)], [b - a for a, b in zip(lo, hi)], look_, pm))


def rbody(st, path, pos, mass):
    x = UsdGeom.Xform.Define(st, path)
    place(x, pos)
    p = x.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(float(mass))
    return p


def wires(st, parent, x0, x1, y0, y1, z, look_):
    """Wire rack: frame, cross wires, two carrier rails."""
    r = .0024
    g = np.linspace(0, 1, 2)
    pts = []
    n = 6
    c = .012
    for a, b in (((x0 + c, y0), (x1 - c, y0)), ((x1, y0 + c), (x1, y1 - c)), ((x1 - c, y1), (x0 + c, y1)), ((x0, y1 - c), (x0, y0 + c))):
        pts += [np.array(a) + (np.array(b) - np.array(a)) * t for t in np.linspace(0, 1, 4)]
    pts = [(p[0], p[1], z) for p in pts]
    pts.append(pts[0])
    tube(st, parent + "/frame", pts, r, look_, n=10)
    xs = np.linspace(x0 + .012, x1 - .012, 21)
    for i, x in enumerate(xs):
        tube(st, f"{parent}/w{i}", [(x, y0, z + .0025), (x, y1, z + .0025)], .0017, look_, n=8)
    for i, y in enumerate((y0 + .07, y1 - .07)):
        tube(st, f"{parent}/x{i}", [(x0, y, z - .002), (x1, y, z - .002)], .0022, look_, n=8)


def element(st, path, z, look_):
    """Heating element: a long U of dark tube across the cavity."""
    y0, y1 = CY0 + .05, CY1 - .05
    x0, x1 = CX0 + .012, CX1 - .012
    pts = [(x0, y0, z)]
    pts += [(x, y0, z) for x in np.linspace(x0, x1 - .03, 6)]
    pts += [(x1 - .03 + .03 * math.sin(a), (y0 + y1) / 2 - (y1 - y0) / 2 * math.cos(a), z) for a in np.linspace(0, math.pi, 14)]
    pts += [(x, y1, z) for x in np.linspace(x1 - .03, x0, 6)]
    tube(st, path, pts, .0042, look_, n=12)


def oven(st):
    steel = surf(st, "ov_steel", (.76, .76, .77), 1., .30, nrm=TEX + "brushed_n2.png", rtex=TEX + "brushed_r2.png")
    bright = surf(st, "ov_bright", (.84, .84, .85), 1., .16)
    panel = surf(st, "ov_panel", None, .85, .34, tex=TEX + "panel.png", rtex=TEX + "brushed_r2.png")
    blackp = surf(st, "ov_blackpaint", (.022, .022, .024), .3, .42)
    slot = surf(st, "ov_slot", (.004, .004, .004), 0., .7)
    rubber = surf(st, "ov_rubber", (.012, .012, .012), 0., .7)
    liner = surf(st, "ov_liner", (.78, .78, .75), .55, .34)
    wire = surf(st, "ov_wire", (.80, .80, .81), 1., .22)
    elem = surf(st, "ov_element", (.13, .11, .10), .4, .55, emissive=(0, 0, 0))
    led = surf(st, "ov_led", (.35, .02, .02), 0., .25, emissive=(0, 0, 0))
    window = glass(st, "ov_glass")
    inv = mat(st, "ov_col", (.5, .5, .5))
    pm, rub, foot = phys(st, "ov_body", .5), phys(st, "ov_handle", .9), phys(st, "ov_foot", .9)
    UsdGeom.Xform.Define(st, OVEN)
    b = rbody(st, BODY, W(0, 0, 0), 7.0)
    # black shell with vents
    wx = W_ / 2
    rbox(st, BODY + "/shell_top", (-wx, .002, H_ - .012), (wx, D_, H_), .006, blackp)
    rbox(st, BODY + "/shell_bottom", (-wx, .002, FOOT), (wx, D_, FOOT + .012), .004, blackp)
    rbox(st, BODY + "/shell_left", (-wx, .002, FOOT), (-wx + .012, D_, H_), .006, blackp)
    rbox(st, BODY + "/shell_right", (wx - .012, .002, FOOT), (wx, D_, H_), .006, blackp)
    rbox(st, BODY + "/shell_back", (-wx, D_ - .012, FOOT), (wx, D_, H_), .004, blackp)
    for c, y in enumerate((.10, .17, .24, .31)):                 # pressed louvres on the side
        for i in range(9):
            z = .070 + i * .024
            for k, sx in enumerate((1, -1)):
                rbox(st, f"{BODY}/louvre{k}_{c}_{i}", (sx * wx - .0012, y - .024, z), (sx * wx + .0012, y + .024, z + .0085), .001, slot)
    for i in range(11):                                          # ribs along the top
        x = -.20 + i * .04
        rbox(st, f"{BODY}/rib{i}", (x - .004, .06, H_ - .0006), (x + .004, D_ - .05, H_ + .0012), .0008, slot)
    # stainless front: frame and panel
    fy0, fy1 = -.0005, .010
    rbox(st, BODY + "/f_top", (-wx, fy0, Z1), (wx, fy1, H_ + .001), .003, steel)
    rbox(st, BODY + "/f_bot", (-wx, fy0, FOOT - .001), (wx, fy1, Z0), .003, steel)
    rbox(st, BODY + "/f_left", (-wx, fy0, Z0), (X0, fy1, Z1), .002, steel)
    rbox(st, BODY + "/f_mid", (X1, fy0, Z0), (PX0, fy1, Z1), .002, steel)
    pl = rbox(st, BODY + "/f_panel", (PX0, fy0 - .0015, Z0 - .010), (wx - .004, fy1, Z1 + .010), .002, panel)
    P_ = np.array(pl.GetPointsAttr().Get())
    sts = np.c_[(P_[:, 0] / (wx - .004 - PX0)) + .5, (P_[:, 2] / (Z1 - Z0 + .020)) + .5]
    UsdGeom.PrimvarsAPI(pl).GetPrimvar("st").Set(Vt.Vec2fArray.FromNumpy(sts.astype(np.float32)))
    # cavity liner
    rbox(st, BODY + "/l_floor", (CX0, CY0, CZ0 - .004), (CX1, CY1, CZ0), .001, liner)
    rbox(st, BODY + "/l_roof", (CX0, CY0, CZ1), (CX1, CY1, CZ1 + .004), .001, liner)
    rbox(st, BODY + "/l_back", (CX0, CY1, CZ0), (CX1, CY1 + .004, CZ1), .001, liner)
    rbox(st, BODY + "/l_left", (CX0 - .004, CY0, CZ0), (CX0, CY1, CZ1), .001, liner)
    rbox(st, BODY + "/l_right", (CX1, CY0, CZ0), (CX1 + .004, CY1, CZ1), .001, liner)
    for i, z in enumerate((.090, .135, .180, .225)):            # pressed rack runners
        for k, x in enumerate((CX0, CX1)):
            s_ = 1 if k == 0 else -1
            rbox(st, f"{BODY}/run{k}_{i}", (min(x, x + s_ * .007), CY0 + .015, z - .006), (max(x, x + s_ * .007), CY1 - .02, z - .0015), .001, liner)
    element(st, BODY + "/elem_top", CZ1 - .018, elem)
    element(st, BODY + "/elem_bot", CZ0 + .016, elem)
    wires(st, BODY + "/rack", RACK[0], RACK[1], RACK[2], RACK[3], RACK_Z, wire)
    # baking tray on the lowest runner, as in the reference photo
    tray = surf(st, "ov_tray", (.03, .03, .032), .4, .3)
    rbox(st, BODY + "/tray", (RACK[0] + .002, RACK[2] - .004, .086), (RACK[1] - .002, RACK[3] + .004, .0885), .001, tray)
    for k, (lo, hi) in enumerate((((RACK[0], RACK[2] - .006, .086), (RACK[1], RACK[2] - .003, .098)), ((RACK[0], RACK[3] + .003, .086), (RACK[1], RACK[3] + .006, .098)),
                                  ((RACK[0], RACK[2] - .006, .086), (RACK[0] + .003, RACK[3] + .006, .098)), ((RACK[1] - .003, RACK[2] - .006, .086), (RACK[1], RACK[3] + .006, .098)))):
        rbox(st, f"{BODY}/tray_lip{k}", lo, hi, .001, tray)
    # feet
    for i, (sx, y) in enumerate(((-1, .03), (1, .03), (-1, D_ - .03), (1, D_ - .03))):
        x = sx * (wx - .04)
        lathe(st, f"{BODY}/foot{i}", [(0, 0), (.013, 0), (.0145, .002), (.0155, FOOT - .002), (.015, FOOT), (0, FOOT)], rubber, n=32, centre=(x, y, 0), axis_y=False)
        hide(cyl(st, f"{BODY}/c_foot{i}", (x, y, FOOT / 2), .0145, FOOT, inv, foot))
    # LED under the knobs
    lathe(st, BODY + "/led", [(0, .0035), (.0030, .0032), (.0040, .001), (.0042, 0)], led, n=24, centre=(KNOB_X + .036, -.002, .030))
    # colliders of the case
    hbox(st, BODY + "/c_floor", (-wx, 0, FOOT), (wx, D_, CZ0), pm, inv)
    hbox(st, BODY + "/c_roof", (-wx, 0, CZ1), (wx, D_, H_), pm, inv)
    hbox(st, BODY + "/c_back", (-wx, CY1, CZ0), (wx, D_, CZ1), pm, inv)
    hbox(st, BODY + "/c_left", (-wx, 0, CZ0), (CX0, CY1, CZ1), pm, inv)
    hbox(st, BODY + "/c_right", (CX1, 0, CZ0), (wx, CY1, CZ1), pm, inv)
    for i in range(11):
        x = RACK[0] + .010 + (RACK[1] - RACK[0] - .02) * i / 10
        hbox(st, f"{BODY}/c_rack{i}", (x - .002, RACK[2], RACK_Z - .002), (x + .002, RACK[3], RACK_Z + .004), pm, inv)
    lamp = UsdLux.SphereLight.Define(st, BODY + "/lamp")
    lamp.CreateRadiusAttr(.02)
    lamp.CreateIntensityAttr(100000.0)
    lamp.CreateColorAttr(Gf.Vec3f(1., .90, .74))
    place(lamp, (CX1 - .05, CY1 - .10, CZ1 - .03))
    lamp2 = UsdLux.SphereLight.Define(st, BODY + "/lamp2")
    lamp2.CreateRadiusAttr(.02)
    lamp2.CreateIntensityAttr(100000.0)
    lamp2.CreateColorAttr(Gf.Vec3f(1., .90, .74))
    place(lamp2, (CX0 + .05, CY0 + .12, CZ1 - .03))
    UsdPhysics.ArticulationRootAPI.Apply(b)
    PhysxSchema.PhysxArticulationAPI.Apply(b).CreateEnabledSelfCollisionsAttr(False)
    # door: frame on the hinge
    rbody(st, DOOR, W(0, HY, HZ), 1.3)
    dh = Z1 - HZ - .001
    xc = (X0 + X1) / 2
    hw = (X1 - X0) / 2 - .001
    t0, t1 = -DOOR_T, 0.

    def dp(n_, lo, hi, r, l_):
        rbox(st, f"{DOOR}/{n_}", (xc + lo[0], lo[1], lo[2]), (xc + hi[0], hi[1], hi[2]), r, l_)

    dp("frame_b", (-hw, t0, 0.), (hw, t1, .040), .003, steel)
    dp("frame_t", (-hw, t0, dh - .046), (hw, t1, dh), .003, steel)
    dp("frame_l", (-hw, t0, .040), (-hw + .022, t1, dh - .046), .002, steel)
    dp("frame_r", (hw - .022, t0, .040), (hw, t1, dh - .046), .002, steel)
    g = rbox(st, DOOR + "/glass", (xc - hw + .018, t0 + .006, .036), (xc + hw - .018, t0 + .009, dh - .042), .0006, window)
    # handle: long bar on two brackets
    bz = BAR_Z - HZ
    by = BAR_Y - HY
    xs = (X0 + .022, X1 - .022)
    lathe(st, DOOR + "/bar", [(0, 0), (.006, 0), (.0095, .0015), (BAR_R, .0045), (BAR_R, X1 - X0 - .016), (.0095, X1 - X0 - .0130), (.006, X1 - X0 - .0115), (0, X1 - X0 - .0115)], bright, n=48, axis_y=False)
    q = UsdGeom.Xformable(st.GetPrimAtPath(DOOR + "/bar"))
    q.ClearXformOpOrder()
    q.AddTranslateOp().Set(Gf.Vec3d(X0 + .006, by, bz))
    q.AddRotateYOp().Set(90.0)
    for i, x in enumerate(xs):
        pts = [(x, y, bz - .0 + .010 * math.sin((y - t0) / (by - t0) * math.pi / 2) * 0) for y in np.linspace(t0 - .002, by, 8)]
        tube(st, f"{DOOR}/bracket{i}", pts, .0078, bright, n=20)
    hbox(st, DOOR + "/c_panel", (X0, t0, 0.), (X1, t1, dh), pm, inv)
    hbox(st, DOOR + "/c_bar", (X0 + .006, by - BAR_R, bz - BAR_R), (X1 - .006, by + BAR_R, bz + BAR_R), rub, inv)
    for i, x in enumerate(xs):
        hbox(st, f"{DOOR}/c_br{i}", (x - .007, by, bz - .007), (x + .007, t0, bz + .007), rub, inv)
    j = UsdPhysics.RevoluteJoint.Define(st, OVEN + "/door_j")
    j.CreateBody0Rel().SetTargets([BODY])
    j.CreateBody1Rel().SetTargets([DOOR])
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(0, HY, HZ))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(88.0)
    UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular").CreateTypeAttr("force")
    # knobs: revolute along y, stops 0-270 deg
    for path, z, nm in zip(KNOBS, KNOB_Z, KNOB_NAMES):
        rbody(st, path, W(KNOB_X, -.002, z), .03)
        lathe(st, path + "/skirt", [(0, 0), (.0225, 0), (.0225, .002), (.0205, .004)], bright, n=64, axis_y=True)
        lathe(st, path + "/cap", [(.0195, .002), (.0195, .004), (.0190, .010), (.0188, .018), (.0183, .0215), (.0172, .0235), (.0, .0245)], steel, n=96, flute=30, axis_y=True)
        rbox(st, path + "/fin", (-.0155, -.0292, -.0035), (.0155, -.0235, .0035), .0018, bright)
        rbox(st, path + "/mark", (-.0015, -.0296, .0095), (.0015, -.0290, .0165), .0003, slot)
        hide(cyl(st, path + "/c_knob", (0, -.012, 0), .019, .024, inv, rub, axis="Y"))
        n = path.rsplit("/", 1)[-1]
        p = UsdPhysics.RevoluteJoint.Define(st, f"{OVEN}/{n}_j")
        p.CreateBody0Rel().SetTargets([BODY])
        p.CreateBody1Rel().SetTargets([path])
        p.CreateAxisAttr("Y")
        p.CreateLocalPos0Attr(Gf.Vec3f(KNOB_X, -.002, z))
        p.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        p.CreateLowerLimitAttr(0.0)
        p.CreateUpperLimitAttr(270.0)
        UsdPhysics.DriveAPI.Apply(p.GetPrim(), "angular").CreateTypeAttr("force")


def set_on(s, on):
    """Element glow, LED and cavity light. Visual only: no thermal physics. The elements sit out of sight of the door view, so the cavity light carries the look."""
    st = s.stage
    UsdShade.Shader(st.GetPrimAtPath("/World/Looks/ov_element/s")).GetInput("emissiveColor").Set(Gf.Vec3f(*((9., 2.2, .3) if on else (0, 0, 0))))
    UsdShade.Shader(st.GetPrimAtPath("/World/Looks/ov_led/s")).GetInput("emissiveColor").Set(Gf.Vec3f(*((30., 1., 1.) if on else (0, 0, 0))))
    for n in ("lamp", "lamp2"):
        L = UsdLux.SphereLight(st.GetPrimAtPath(f"{BODY}/{n}"))
        L.GetColorAttr().Set(Gf.Vec3f(*((1., .42, .12) if on else (1., .90, .74))))
        L.GetIntensityAttr().Set(190000.0 if on else 100000.0)
