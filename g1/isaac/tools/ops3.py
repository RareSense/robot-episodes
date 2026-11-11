# lights + whole-robot cameras for the walking tasks (exec after ops2.py)
from pxr import UsdLux, Gf, UsdGeom
for nm, pos, r, I in (('LiveL1', (2.0, 1.6, 2.1), .15, 60000.), ('LiveL2', (1.2, 2.0, 2.0), .15, 60000.), ('LiveL4', (2.6, 1.2, 2.0), .15, 60000.), ('LiveL5', (0.8, 1.2, 2.0), .15, 60000.)):
    if not s.stage.GetPrimAtPath('/World/' + nm):
        l = UsdLux.SphereLight.Define(s.stage, '/World/' + nm); l.CreateRadiusAttr(r); l.CreateIntensityAttr(I)
        UsdGeom.Xformable(l.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos))


def wcam(name, back=1.9, side=0.0, h=1.15):
    """whole robot from behind (-y) and from its side (+x)"""
    p = s.link("pelvis")[:3]
    shot(name + "_back", (p[0] + side, p[1] - back, h), (p[0], p[1], .65), 55)
    shot(name + "_sidex", (p[0] + 1.7, p[1] - .25, h), (p[0], p[1], .65), 55)


def wstat(tag=""):
    i = int(np.clip(round((s.tr - R.t[0]) * 60), 0, len(R.t) - 1))
    L = s.link_poses()
    rt = L[s.links.index("pelvis")]
    r, p, y = C.rpy(rt[3:7])
    qr = R.xquat[i, BN.index("pelvis")]; rr, pr, yr = C.rpy(np.r_[qr[1:], qr[0]])
    f = [L[s.links.index(x), :3] for x in FEET]; fr = [R.xpos[i, BN.index(x)] for x in FEET]
    o = {k: s.obj(k)[:3].round(3).tolist() for k in OBJK}
    print(tag, f"tr {s.tr:.2f} pel {rt[:3].round(3).tolist()} rec {R.xpos[i, BN.index('pelvis')].round(3).tolist()} rpy {math.degrees(r):.1f} {math.degrees(p):.1f} {math.degrees(y):.1f} rec {math.degrees(rr):.1f} {math.degrees(pr):.1f} {math.degrees(yr):.1f} | L {f[0].round(3).tolist()} rec {fr[0].round(3).tolist()} | R {f[1].round(3).tolist()} rec {fr[1].round(3).tolist()}")
    return rt
