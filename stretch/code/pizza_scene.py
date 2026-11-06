"""Pizza task scene, authored before physics. A simple room, a stand, a table, a plate, a pizza and an oven. All built here, nothing from the tea room.
Sizes come from the Stretch: pad face gap 174 mm at full open, so plate 150 mm across, pinched across its rim. Pads are 40 mm tall, so the plate stands on a 35 mm foot and the pads clear the table.
Oven door drops down on a hinge along x, so it opens in the plane of the arm and lift. Door counterbalance spring and joint friction are authored in pz.oven_setup. Heating is a timed colour change only: no thermal physics exists here."""
import math

from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

Y0 = .60                     # oven front plane
TOP = .60                    # table and stand height
HINGE_Z, DOOR_H, DOOR_W, DOOR_T = .53, .31, .44, .03
CAV_X, CAV_Z0, CAV_Z1, CAV_D = .22, .54, .84, .42
RACK_Z = .64
HANDLE_Z, HANDLE_Y = .27, -.06           # in the door frame
SL_X, SL_Z0, SL_TRAVEL = (.29, .39), .70, .10
BTN_X, BTN_Z = .34, .60
TABLE_XY = (-.70, .70)
STAND_XY = (-1.55, .62)
PLATE_R, FOOT_R, FOOT_H, RIM_H = .075, .05, .035, .020
PIZZA_R, PIZZA_H = .060, .012
DOUGH, CHEESE, PEPPERONI = (.86, .70, .42), (.97, .90, .62), (.72, .20, .10)
OVEN, PLATE, PIZZA, TABLE, STAND = "/World/oven", "/World/plate", "/World/pizza", "/World/table", "/World/stand"
DOOR, BODY = OVEN + "/door", OVEN + "/body"
SLIDERS = (OVEN + "/slider_temp", OVEN + "/slider_time")
BUTTON = OVEN + "/button"


def mat(st, name, rgb, metal=0., rough=.6):
    m = UsdShade.Material.Define(st, "/World/Looks/" + name)
    sh = UsdShade.Shader.Define(st, f"/World/Looks/{name}/s")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


def phys(st, name, fr, rest=0.):
    m = UsdShade.Material.Define(st, "/World/Phys/" + name)
    a = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    a.CreateStaticFrictionAttr(fr)
    a.CreateDynamicFrictionAttr(fr)
    a.CreateRestitutionAttr(rest)
    return m


def place(g, pos, scale=None):
    x = UsdGeom.Xformable(g)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    if scale is not None:
        x.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def _finish(st, c, look, pm, collide):
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(look)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        if pm is not None:
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    return c


def box(st, path, centre, size, look, pm=None, collide=True):
    c = UsdGeom.Cube.Define(st, path)
    c.CreateSizeAttr(1.0)
    place(c, centre, size)
    return _finish(st, c, look, pm, collide)


def cyl(st, path, centre, r, h, look, pm=None, collide=True, axis="Z"):
    c = UsdGeom.Cylinder.Define(st, path)
    c.CreateRadiusAttr(float(r))
    c.CreateHeightAttr(float(h))
    c.CreateAxisAttr(axis)
    place(c, centre)
    return _finish(st, c, look, pm, collide)


def body(st, path, pos, mass):
    x = UsdGeom.Xform.Define(st, path)
    place(x, pos)
    p = x.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(float(mass))
    return p


def room(st):
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(st.GetPrimAtPath("/World"))
    sc = UsdPhysics.Scene.Define(st, "/World/physics")
    sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr(9.81)
    floor_l, wall_l = mat(st, "floor", (.55, .42, .30)), mat(st, "wall", (.82, .80, .76))
    fl = phys(st, "floor", 1.0)
    box(st, "/World/floor", (0, 0, -.05), (8, 8, .1), floor_l, fl)
    box(st, "/World/wall_n", (0, 1.60, 1.2), (8, .1, 2.4), wall_l, fl)
    box(st, "/World/wall_e", (1.6, 0, 1.2), (.1, 8, 2.4), wall_l, fl)
    d = UsdLux.DomeLight.Define(st, "/World/dome")
    d.CreateIntensityAttr(260.0)
    d.CreateColorAttr(Gf.Vec3f(1, .98, .95))
    sun = UsdLux.DistantLight.Define(st, "/World/sun")
    sun.CreateIntensityAttr(700.0)
    sun.CreateAngleAttr(20.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(55.0, 0.0, -25.0))


def table_like(st, path, xy, size, mass, look, pm):
    p = body(st, path, (xy[0], xy[1], 0), mass)
    w, d = size
    box(st, path + "/top", (0, 0, TOP - .015), (w, d, .03), look, pm)
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        box(st, f"{path}/leg{i}", (sx * (w / 2 - .03), sy * (d / 2 - .03), (TOP - .03) / 2), (.04, .04, TOP - .03), look, pm)
    return p


def oven(st):
    steel, dark, glass = mat(st, "steel", (.72, .72, .74), .7, .35), mat(st, "oven_dark", (.08, .08, .09), .1, .5), mat(st, "oven_glass", (.10, .12, .13), .1, .1)
    wire, knob = mat(st, "wire", (.8, .8, .82), .9, .3), mat(st, "knob", (.75, .10, .08), 0., .5)
    pm = phys(st, "oven", .6)
    rub = phys(st, "handle", .9)
    UsdGeom.Xform.Define(st, OVEN)
    b = body(st, BODY, (0, Y0, 0), 25.0)
    d_ = CAV_D
    xr = CAV_X + .22                     # right end: control column
    xl = -CAV_X - .02
    w = xr - xl
    xc = (xr + xl) / 2
    box(st, BODY + "/cabinet", (xc, d_ / 2 + .02, .26), (w, d_ + .04, .52), steel, pm)
    box(st, BODY + "/floor", (xc, d_ / 2 + .02, .53), (w, d_ + .04, .02), dark, pm)
    box(st, BODY + "/roof", (xc, d_ / 2 + .02, CAV_Z1 + .02), (w, d_ + .04, .04), steel, pm)
    box(st, BODY + "/back", (xc, d_ + .03, .69), (w, .02, .30), dark, pm)
    box(st, BODY + "/left", (xl + .01, d_ / 2, .69), (.02, d_, .30), steel, pm)
    box(st, BODY + "/mid", (CAV_X + .01, d_ / 2, .69), (.02, d_, .30), steel, pm)
    box(st, BODY + "/column", (CAV_X + .12, d_ / 2, .69), (.20, d_, .30), steel, pm)
    for i, x in enumerate(SL_X):
        box(st, f"{BODY}/slot{i}", (x, -.0005, SL_Z0 + SL_TRAVEL / 2), (.012, .001, SL_TRAVEL + .03), dark, collide=False)
    # rack: real bars, fixed in the cavity
    for i in range(9):
        box(st, f"{BODY}/rack_y{i}", (-.18 + .045 * i, .21, RACK_Z), (.006, .38, .006), wire, pm)
    for i, y in enumerate((.05, .21, .37)):
        box(st, f"{BODY}/rack_x{i}", (0, y, RACK_Z - .006), (.44, .006, .006), wire, pm)
    box(st, BODY + "/lamp", (BTN_X, -.0005, .86), (.10, .001, .02), mat(st, "lamp", (.05, .05, .05)), collide=False)
    fx = UsdPhysics.FixedJoint.Define(st, OVEN + "/root")
    fx.CreateBody1Rel().SetTargets([BODY])
    fx.CreateLocalPos0Attr(Gf.Vec3f(0, Y0, 0))
    fx.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    UsdPhysics.ArticulationRootAPI.Apply(b)
    PhysxSchema.PhysxArticulationAPI.Apply(b).CreateEnabledSelfCollisionsAttr(False)
    # door
    dz = DOOR_H / 2
    body(st, DOOR, (0, Y0 - DOOR_T / 2, HINGE_Z), 2.5)
    box(st, DOOR + "/panel", (0, 0, dz), (DOOR_W, DOOR_T, DOOR_H), steel, pm)
    box(st, DOOR + "/window", (0, -DOOR_T / 2 - .0005, dz - .02), (DOOR_W - .12, .001, DOOR_H - .14), glass, collide=False)
    cyl(st, DOOR + "/bar", (0, HANDLE_Y, HANDLE_Z), .011, .34, wire, rub, axis="X")
    for i, sx in enumerate((-1, 1)):
        box(st, f"{DOOR}/post{i}", (sx * .15, (HANDLE_Y - DOOR_T / 2) / 2, HANDLE_Z), (.016, abs(HANDLE_Y) - DOOR_T / 2, .016), wire, rub)
    j = UsdPhysics.RevoluteJoint.Define(st, OVEN + "/door_j")
    j.CreateBody0Rel().SetTargets([BODY])
    j.CreateBody1Rel().SetTargets([DOOR])
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(0, -DOOR_T / 2, HINGE_Z))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(90.0)
    UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular").CreateTypeAttr("force")
    # sliders: real prismatic joints, up the front of the control column
    for path, x in zip(SLIDERS, SL_X):
        body(st, path, (x, Y0 - .03, SL_Z0), .05)
        box(st, path + "/knob", (0, 0, 0), (.022, .06, .03), knob, rub)
        n = path.rsplit("/", 1)[-1]
        p = UsdPhysics.PrismaticJoint.Define(st, f"{OVEN}/{n}_j")
        p.CreateBody0Rel().SetTargets([BODY])
        p.CreateBody1Rel().SetTargets([path])
        p.CreateAxisAttr("Z")
        p.CreateLocalPos0Attr(Gf.Vec3f(x, -.03, SL_Z0))
        p.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        p.CreateLowerLimitAttr(0.0)
        p.CreateUpperLimitAttr(SL_TRAVEL)
        UsdPhysics.DriveAPI.Apply(p.GetPrim(), "linear").CreateTypeAttr("force")
    # start button: pushed in along +y, spring back
    body(st, BUTTON, (BTN_X, Y0 - .012, BTN_Z), .03)
    box(st, BUTTON + "/cap", (0, 0, 0), (.06, .024, .06), mat(st, "button", (.10, .55, .18), 0., .5), rub)
    p = UsdPhysics.PrismaticJoint.Define(st, OVEN + "/button_j")
    p.CreateBody0Rel().SetTargets([BODY])
    p.CreateBody1Rel().SetTargets([BUTTON])
    p.CreateAxisAttr("Y")
    p.CreateLocalPos0Attr(Gf.Vec3f(BTN_X, -.012, BTN_Z))
    p.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    p.CreateLowerLimitAttr(0.0)
    p.CreateUpperLimitAttr(.012)
    UsdPhysics.DriveAPI.Apply(p.GetPrim(), "linear").CreateTypeAttr("force")


def revolve(st, path, prof, look, n=96):
    """Smooth lathe mesh, visual only. prof: (r, z) from the axis outwards and round. The colliders stay simple cylinders."""
    m = UsdGeom.Mesh.Define(st, path)
    pts, fv, fi = [], [], []
    k = len(prof)
    for i in range(n):
        a = 2 * math.pi * i / n
        pts += [Gf.Vec3f(r * math.cos(a), r * math.sin(a), z) for r, z in prof]
    for i in range(n):
        j = (i + 1) % n
        for q in range(k - 1):
            fv.append(4)
            fi += [i * k + q, j * k + q, j * k + q + 1, i * k + q + 1]
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(fv)
    m.CreateFaceVertexIndicesAttr(fi)
    m.CreateSubdivisionSchemeAttr("catmullClark")
    m.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(look)
    return m


def hide(p):
    UsdGeom.Imageable(p).MakeInvisible()


def build(st):
    import random
    wood, white = mat(st, "wood", (.42, .27, .15), 0., .45), mat(st, "plate", (.90, .90, .87), 0., .22)
    tp = phys(st, "table", .6)
    table_like(st, TABLE, TABLE_XY, (.60, .60), 15.0, wood, tp)
    table_like(st, STAND, STAND_XY, (.40, .40), 8.0, wood, tp)
    oven(st)
    pp = phys(st, "plate", .7)
    body(st, PLATE, (STAND_XY[0], STAND_XY[1], TOP), .30)
    hide(cyl(st, PLATE + "/foot", (0, 0, FOOT_H / 2), FOOT_R, FOOT_H, white, pp))
    hide(cyl(st, PLATE + "/disc", (0, 0, FOOT_H + RIM_H / 2), PLATE_R, RIM_H, white, pp))
    zt = FOOT_H + RIM_H
    revolve(st, PLATE + "/look", [(0, 0), (.046, 0), (.05, .002), (.05, .03), (.052, FOOT_H), (.073, FOOT_H + .002), (.075, FOOT_H + .006), (.075, zt - .004), (.0735, zt), (.066, zt + .001), (.0625, zt), (0, zt)], white)
    dough, cheese, pep = mat(st, "pizza_dough", DOUGH, 0., .75), mat(st, "pizza_cheese", CHEESE, 0., .55), mat(st, "pizza_top", PEPPERONI, 0., .5)
    body(st, PIZZA, (STAND_XY[0], STAND_XY[1], TOP + zt), .15)
    hide(cyl(st, PIZZA + "/base", (0, 0, PIZZA_H / 2), PIZZA_R, PIZZA_H, dough, pp))
    revolve(st, PIZZA + "/crust", [(.048, .009), (.052, .012), (.0565, .0135), (.0595, .011), (.060, .006), (.058, .001), (.05, 0), (0, 0)], dough)
    revolve(st, PIZZA + "/cheese", [(0, .0105), (.04, .0105), (.049, .0102), (.0525, .0085)], cheese)
    rnd = random.Random(4)
    spots = [(.0, .0), (.030, .008), (-.012, .031), (-.031, -.006), (.006, -.033), (.024, -.024), (-.024, .022), (.021, .026)]
    for i, (x, y) in enumerate(spots):
        r = .0085 + .0015 * rnd.random()
        g = revolve(st, f"{PIZZA}/pep{i}", [(0, .0122), (r * .8, .0120), (r, .0112), (r, .0100)], pep, n=32)
        place(g, (x, y, 0))
