"""Pizza task in a copy of the H1-2 kitchen (E:/stretch_assets/kitchen_copy). Everything is authored in the session layer of a wrapper stage: no source USD is edited.
Oven look: NVIDIA ArchVis wall_oven.usd, split into a case and a door by deactivating prims in two references. Table: SimReady appleseed_endtable. Source and licence in NOTES.md. Plate and pizza modelled here; pizza textures from analysis/pizza_gen.py.
Physics is my own: the assets have none. Case, door, rack, sliders and button are an articulation with hidden simple colliders under the visual mesh.
Spot, from frames: east wall between the kitchen door and the stairs side, door facing -x. Plate starts on the real kitchen counter.
Oven frame: origin at floor of the case, centre of the door face, y into the oven, z up. Model cm with +y front, so oven = (-x, .275 - y, z) / 100. Oven y is world +x, oven x is world -y."""
import math

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from pizza_scene import box, cyl, hide, mat, phys, place, revolve

ASSETS = "E:/stretch_assets"
STAGE = ASSETS + "/kitchen_copy/stage.usda"
OVEN_USD = ASSETS + "/archvis_oven/wall_oven.usd"
TABLE_USD = ASSETS + "/sr/common_assets/props/appleseed_endtable/appleseed_endtable.usd"
FX, FY, PZ = 2.25, -.40, .40           # oven front plane x, centre y, plinth height
HY, HZ = .015, .08                     # hinge in the oven frame
RACK_TOP = .174
SL_X, SL_Z0, SL_TRAVEL = (-.29, .29), .625, .07
BTN_X, BTN_Z = -.215, .665
T_H = .635
TABLE_XY = (2.55, .28)
PLATE_XY, COUNTER_Z = (1.67, 2.15), .86
SET_XY = (.86, 2.15)                 # where the plate is put down on the counter
BODY = "/World/oven/body"
PLATE, PIZZA = "/World/pz_plate", "/World/pz_pizza"
GRP = "wall_oven_grp/wall_oven_trans_grp"
PLATE_R, FOOT_H, WELL_Z, RIM_Z = .066, .012, .016, .036     # pz_kit08: pads rest on a 150 mm rim, real gap at full open is about 152 mm, not the 174 mm of pad_gap.py
PIZZA_R = .0505
PS = PLATE_R / .075
PIZZA_TEX = ASSETS + "/pizza2/"


def W(x, y, z):
    return (FX + y, FY - x, PZ + z)


def stage_file():
    subs = ["./looks/" + x for x in ("looks_props.usda", "looks_room.usda", "looks_exterior.usda", "set_fix.usda", "kettle_fix.usda", "looks.usda")] + ["./scene_v3/environment.usda"]
    body_ = ",".join("@" + x + "@" for x in subs)
    with open(STAGE, "w") as f:
        f.write('#usda 1.0\n(\n    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Z"\n    subLayers = [' + body_ + ']\n)\n')
    return STAGE


def rbody(st, path, pos, mass, yaw=-90.0):
    x = UsdGeom.Xform.Define(st, path)
    a = UsdGeom.Xformable(x)
    a.ClearXformOpOrder()
    a.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    if yaw:
        a.AddRotateZOp().Set(float(yaw))
    p = x.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(float(mass))
    return p


def look(st, path, t, off):
    """Reference the whole oven file with `off` prims deactivated. Whole file, so its materials stay in reach."""
    x = UsdGeom.Xform.Define(st, path)
    a = UsdGeom.Xformable(x)
    a.AddTranslateOp().Set(Gf.Vec3d(*t))
    a.AddRotateZOp().Set(180.0)
    a.AddScaleOp().Set(Gf.Vec3f(.01, .01, .01))
    x.GetPrim().GetReferences().AddReference(OVEN_USD)
    for o in off:
        st.GetPrimAtPath(f"{path}/{GRP}/{o}").SetActive(False)


def hbox(st, path, lo, hi, pm, wire):
    c = [(a + b) / 2 for a, b in zip(lo, hi)]
    s_ = [b - a for a, b in zip(lo, hi)]
    hide(box(st, path, c, s_, wire, pm))


def ybar(st, path, pos, r, ln, look_, n=40):
    """Round knob or cap along y, smooth: lathe about z, turned onto y."""
    g = revolve(st, path, [(0, -ln), (r * .9, -ln), (r, -ln + r * .12), (r, 0)], look_, n=n)
    x = UsdGeom.Xformable(g)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x.AddRotateXOp().Set(-90.0)
    return g


def oven(st):
    steel = mat(st, "pz_steel", (.78, .78, .79), 1., .28)
    chrome = mat(st, "pz_chrome", (.86, .86, .87), 1., .14)
    black = mat(st, "pz_black", (.015, .015, .018), .2, .35)
    invis = mat(st, "pz_col", (.5, .5, .5))
    pm, rub = phys(st, "pz_oven", .6), phys(st, "pz_handle", .9)
    UsdGeom.Xform.Define(st, OVEN)
    b = rbody(st, BODY, W(0, 0, 0), 30.0)
    look(st, BODY + "/look", (0, .275, 0), ["wall_oven_door", "wall_oven_racks_grp/wall_oven_grill2_SUBD0"])
    hbox(st, BODY + "/c_floor", (-.368, -.01, 0), (.368, .60, .08), pm, invis)
    hbox(st, BODY + "/c_roof", (-.368, -.01, .56), (.368, .60, .74), pm, invis)
    hbox(st, BODY + "/c_back", (-.368, .50, .08), (.368, .60, .56), pm, invis)
    hbox(st, BODY + "/c_left", (-.368, -.01, .08), (-.345, .50, .56), pm, invis)
    hbox(st, BODY + "/c_right", (.345, -.01, .08), (.368, .50, .56), pm, invis)
    for i in range(15):
        x = -.315 + .045 * i
        hbox(st, f"{BODY}/c_rack{i}", (x - .003, .12, RACK_TOP - .006), (x + .003, .397, RACK_TOP), pm, invis)
    lamp = UsdLux.SphereLight.Define(st, BODY + "/lamp")
    lamp.CreateRadiusAttr(.03)
    lamp.CreateIntensityAttr(130000.0)
    lamp.CreateColorAttr(Gf.Vec3f(1., .82, .58))
    place(lamp, (0, .20, .50))
    fx = UsdPhysics.FixedJoint.Define(st, OVEN + "/root")
    fx.CreateBody1Rel().SetTargets([BODY])
    fx.CreateLocalPos0Attr(Gf.Vec3f(*map(float, W(0, 0, 0))))
    fx.CreateLocalRot0Attr(Gf.Quatf(math.cos(math.radians(-45)), 0, 0, math.sin(math.radians(-45))))
    fx.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    UsdPhysics.ArticulationRootAPI.Apply(b)
    PhysxSchema.PhysxArticulationAPI.Apply(b).CreateEnabledSelfCollisionsAttr(False)
    rbody(st, DOOR, W(0, HY, HZ), 4.0)
    look(st, DOOR + "/look", (0, .275 - HY, -HZ), ["wall_oven_hinges_SUBD0", "wall_oven_racks_grp", "wall_oven_panel_grp", "wall_oven_case_grp"])
    y0, y1 = -.002 - HY, .047 - HY
    hbox(st, DOOR + "/c_panel", (-.367, y0, .063 - HZ), (.367, y1, .604 - HZ), pm, invis)
    hbox(st, DOOR + "/c_handle", (-.30, -.066 - HY, .577 - HZ), (.30, y0, .603 - HZ), rub, invis)
    j = UsdPhysics.RevoluteJoint.Define(st, OVEN + "/door_j")
    j.CreateBody0Rel().SetTargets([BODY])
    j.CreateBody1Rel().SetTargets([DOOR])
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(0, HY, HZ))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(90.0)
    UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular").CreateTypeAttr("force")
    for i, (path, x) in enumerate(zip(SLIDERS, SL_X)):
        box(st, f"{BODY}/track{i}", (x, -.0105, SL_Z0 + SL_TRAVEL / 2), (.006, .002, SL_TRAVEL + .018), black, collide=False)
        for k in range(8):
            w = .014 if k % 7 == 0 else .009
            box(st, f"{BODY}/tick{i}_{k}", (x + (.0125 + w / 2) * (1 if x > 0 else -1), -.0108, SL_Z0 + SL_TRAVEL * k / 7), (w, .0006, .0012), black, collide=False)
        rbody(st, path, W(x, -.0105, SL_Z0), .05)
        ybar(st, path + "/cap", (0, 0, 0), .014, .032, chrome)
        ybar(st, path + "/ring", (0, 0, 0), .0165, .004, steel)
        hbox(st, path + "/c_cap", (-.012, -.034, -.012), (.012, 0, .012), rub, invis)
        n = path.rsplit("/", 1)[-1]
        p = UsdPhysics.PrismaticJoint.Define(st, f"{OVEN}/{n}_j")
        p.CreateBody0Rel().SetTargets([BODY])
        p.CreateBody1Rel().SetTargets([path])
        p.CreateAxisAttr("Z")
        p.CreateLocalPos0Attr(Gf.Vec3f(x, -.0105, SL_Z0))
        p.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        p.CreateLowerLimitAttr(0.0)
        p.CreateUpperLimitAttr(SL_TRAVEL)
        UsdPhysics.DriveAPI.Apply(p.GetPrim(), "linear").CreateTypeAttr("force")
    rbody(st, BUTTON, W(BTN_X, -.0105, BTN_Z), .03)
    ybar(st, BUTTON + "/cap", (0, 0, 0), .017, .012, chrome)
    hbox(st, BUTTON + "/c_cap", (-.015, -.012, -.015), (.015, 0, .015), rub, invis)
    p = UsdPhysics.PrismaticJoint.Define(st, OVEN + "/button_j")
    p.CreateBody0Rel().SetTargets([BODY])
    p.CreateBody1Rel().SetTargets([BUTTON])
    p.CreateAxisAttr("Y")
    p.CreateLocalPos0Attr(Gf.Vec3f(BTN_X, -.0105, BTN_Z))
    p.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    p.CreateLowerLimitAttr(0.0)
    p.CreateUpperLimitAttr(.008)
    UsdPhysics.DriveAPI.Apply(p.GetPrim(), "linear").CreateTypeAttr("force")


def plinth(st):
    """Cabinet the oven stands on. Static. Dark satin, to match the kitchen's black units."""
    pm = phys(st, "pz_cab", .6)
    cab = mat(st, "pz_cab_look", (.035, .035, .04), .1, .42)
    toe = mat(st, "pz_toe", (.012, .012, .012), 0., .6)
    box(st, "/World/pz_plinth", (FX + .305, FY, (PZ + .07) / 2), (.61, .74, PZ - .07), cab, pm)
    box(st, "/World/pz_toe", (FX + .33, FY, .035), (.56, .70, .07), toe, pm)


def table(st, path, xy):
    p = rbody(st, path, (xy[0], xy[1], 0), 22.0, yaw=0)
    v = st.DefinePrim(path + "/look", "Xform")
    v.GetReferences().AddReference(TABLE_USD)
    pm = phys(st, "pz_table", .6)
    invis = mat(st, "pz_col2", (.5, .5, .5))
    hbox(st, path + "/c_top", (-.279, -.29, .46), (.279, .29, T_H), pm, invis)
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        hbox(st, f"{path}/c_leg{i}", (sx * .22 - .03, sy * .24 - .03, 0), (sx * .22 + .03, sy * .24 + .03, .46), pm, invis)


def tex_mat(st, name, albedo, normal, rough=.62):
    m = UsdShade.Material.Define(st, "/World/Looks/" + name)
    sh = UsdShade.Shader.Define(st, f"/World/Looks/{name}/s")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    rd = UsdShade.Shader.Define(st, f"/World/Looks/{name}/st")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    for nm, f, cs, out, inp, ty in (("tex", albedo, "sRGB", "rgb", "diffuseColor", Sdf.ValueTypeNames.Float3), ("nrm", normal, "raw", "rgb", "normal", Sdf.ValueTypeNames.Float3)):
        t = UsdShade.Shader.Define(st, f"/World/Looks/{name}/{nm}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f)
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(cs)
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        if nm == "nrm":
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        o = t.CreateOutput(out, ty)
        sh.CreateInput(inp, Sdf.ValueTypeNames.Color3f if nm == "tex" else Sdf.ValueTypeNames.Normal3f).ConnectToSource(o)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


def pizza_mesh(st, path, look_):
    """Surface from the height field, polar grid, with an underside."""
    H = np.load(PIZZA_TEX + "height.npy")
    nr, nt = 44, 160
    pts, uv = [], []
    for i in range(nr + 1):
        rr = i / nr
        for j in range(nt):
            a = 2 * math.pi * j / nt
            u, v = rr * math.cos(a), rr * math.sin(a)
            hx, hy = (u * .5 + .5) * 255, (v * .5 + .5) * 255
            z = float(H[int(hy), int(hx)]) * .016 + .0005
            if rr > .90:                       # round the outer edge down to the plate
                z *= math.sqrt(max(0., 1 - ((rr - .90) / .10) ** 2))
            pts.append((PIZZA_R * u, PIZZA_R * v, z))
            uv.append((u * .5 + .5, 1 - (v * .5 + .5)))
    P = np.array(pts).reshape(nr + 1, nt, 3)
    dr = np.gradient(P, axis=0)
    dt = np.roll(P, -1, 1) - np.roll(P, 1, 1)
    nrm = np.cross(dr.reshape(-1, 3), dt.reshape(-1, 3))
    nrm[:nt] = (0, 0, 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    nrm[nrm[:, 2] < 0] *= -1
    fv, fi = [], []
    for i in range(nr):
        for j in range(nt):
            k = (j + 1) % nt
            fv.append(4)
            fi += [i * nt + j, (i + 1) * nt + j, (i + 1) * nt + k, i * nt + k]
    m = UsdGeom.Mesh.Define(st, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pts, np.float32)))
    m.CreateFaceVertexCountsAttr(fv)
    m.CreateFaceVertexIndicesAttr(fi)
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)))
    m.SetNormalsInterpolation("vertex")
    m.CreateDoubleSidedAttr(True)
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray.FromNumpy(np.array(uv, np.float32)))
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(look_)


def plate_pizza(st):
    white = mat(st, "pz_plate_look", (.92, .91, .87), 0., .16)
    pp = phys(st, "pz_plate", .7)
    invis = mat(st, "pz_col3", (.5, .5, .5))
    x, y = PLATE_XY
    rbody(st, PLATE, (x, y, COUNTER_Z), .35, yaw=0)
    hide(cyl(st, PLATE + "/c_foot", (0, 0, FOOT_H / 2), .048 * PS, FOOT_H, invis, pp))
    hide(cyl(st, PLATE + "/c_well", (0, 0, WELL_Z / 2 + .0005), .070 * PS, WELL_Z - .001, invis, pp))        # thick: a 4 mm disc under the pizza threw it in pz_kit09 (lead)
    n = 24
    for i in range(n):
        a = 360.0 * i / n
        c = box(st, f"{PLATE}/c_rim{i}", (0, 0, 0), (.007, .0185, RIM_Z - FOOT_H), invis, pp)
        q = UsdGeom.Xformable(c)
        q.ClearXformOpOrder()
        q.AddTranslateOp().Set(Gf.Vec3d((PLATE_R - .0035) * math.cos(math.radians(a)), (PLATE_R - .0035) * math.sin(math.radians(a)), (RIM_Z + FOOT_H) / 2))
        q.AddRotateZOp().Set(a)
        q.AddScaleOp().Set(Gf.Vec3f(.007, .0185, RIM_Z - FOOT_H))
        hide(c)
    prof = [(0, WELL_Z), (.060, WELL_Z), (.064, WELL_Z + .002), (.070, RIM_Z - .006), (.0735, RIM_Z - .001), (.0748, RIM_Z - .0005), (.0752, RIM_Z - .002), (.0745, RIM_Z - .005),
            (.066, WELL_Z - .0005), (.052, FOOT_H - .002), (.049, .002), (.047, 0), (.044, 0), (.043, .002), (0, .003)]
    revolve(st, PLATE + "/look", [(r * PS, z) for r, z in prof], white, n=128)
    rbody(st, PIZZA, (x, y, COUNTER_Z + WELL_Z), .15, yaw=0)
    hide(cyl(st, PIZZA + "/c_base", (0, 0, .006), PIZZA_R, .012, invis, pp))
    pizza_mesh(st, PIZZA + "/look", tex_mat(st, "pz_pizza_look", PIZZA_TEX + "albedo_0.png", PIZZA_TEX + "normal.png"))
    for q in (PLATE, PIZZA):
        PhysxSchema.PhysxRigidBodyAPI.Apply(st.GetPrimAtPath(q)).CreateMaxDepenetrationVelocityAttr(.5)


def fill(st):
    d = UsdLux.DistantLight.Define(st, "/World/RecFill")
    d.CreateIntensityAttr(2500.0)
    d.CreateAngleAttr(30.0)
    d.CreateColorAttr(Gf.Vec3f(1.0, .97, .92))
    UsdGeom.Xformable(d).AddRotateXYZOp().Set(Gf.Vec3f(75.0, 0.0, -20.0))
    r = UsdLux.RectLight.Define(st, "/World/RecSoft")        # soft light over the oven side of the room: steel only looks like steel with something bright to reflect
    r.CreateWidthAttr(1.6)
    r.CreateHeightAttr(1.6)
    r.CreateIntensityAttr(1800.0)
    r.CreateColorAttr(Gf.Vec3f(1.0, .96, .90))
    x = UsdGeom.Xformable(r)
    x.AddTranslateOp().Set(Gf.Vec3d(1.1, -.1, 2.3))
    x.AddRotateXYZOp().Set(Gf.Vec3f(0.0, -55.0, 0.0))


def build(st):
    import pizza_oven
    fill(st)
    pizza_oven.oven(st)
    plate_pizza(st)
