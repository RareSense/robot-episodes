"""Milk powder visuals for tea_render.py / water_render.py: the twin's jar opened at its lid line (the twin mesh has
the lid built in; the lid is drawn as its own prop, prop_powder_lid, from props_visual.npz), the powder inside the jar
(the crater-shaped fill of agents\\powder\\props_powder.py) and the simulated grains (one-way grain pass,
agents\\powder\\powder_pass.py -> <run>_grains.npz) drawn on every frame."""
import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

JAR = "/World/Objects/object_0309"
POWDER_RGB = (0.95, 0.93, 0.86)
FUNNEL = (0.012, 0.041, 0.080, 0.094)       # props_powder.FUNNEL: r_in, r_out, z at r_in, z at r_out (jar frame)
FLOOR_Z = 0.0075                             # props_powder: top of the jar's glass floor


def _material(stage):
    key = "/World/Looks/milk_powder"
    if stage.GetPrimAtPath(key):
        return UsdShade.Material(stage.GetPrimAtPath(key))
    mt = UsdShade.Material.Define(stage, key)
    sh = UsdShade.Shader.Define(stage, key + "/Shader")
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*POWDER_RGB))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.95)
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    for o in ("surface", "displacement", "volume"):
        getattr(mt, f"Create{o.capitalize()}Output")("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mt


def _lathe(profile, n=48):
    """Triangle mesh of a profile [(r, z), ...] turned about z; a first point with r = 0 closes the centre."""
    pts, faces = [], []
    rings = []
    for r, z in profile:
        if r == 0:
            rings.append([len(pts)])
            pts.append((0.0, 0.0, z))
            continue
        rings.append(list(range(len(pts), len(pts) + n)))
        pts += [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for i in range(n)]
    for a, b in zip(rings, rings[1:]):
        for i in range(n):
            j = (i + 1) % n
            if len(a) == 1:
                faces.append((a[0], b[i], b[j]))
            else:
                faces += [(a[i], b[i], b[j]), (a[i], b[j], a[j])]
    return np.array(pts, np.float32), np.array(faces, np.int32)


def open_jar(stage, cut_z=0.109):
    """Drop the twin jar mesh's faces above the lid line (jar frame) and put the powder inside the jar."""
    jar = stage.GetPrimAtPath(JAR)
    if not jar:
        return False
    for p in Usd.PrimRange(jar):
        if not p.IsA(UsdGeom.Mesh) or p.GetName().startswith("collision"):
            continue
        me = UsdGeom.Mesh(p)                    # the twin's meshes sit untransformed under the object prim
        pts = np.asarray(me.GetPointsAttr().Get(), float)
        cnt = np.asarray(me.GetFaceVertexCountsAttr().Get(), int)
        idx = np.asarray(me.GetFaceVertexIndicesAttr().Get(), int)
        zc = np.add.reduceat(pts[idx, 2], np.r_[0, np.cumsum(cnt)[:-1]]) / cnt
        keep = zc < cut_z
        me.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(cnt[keep].astype(np.int32)))
        me.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(idx[np.repeat(keep, cnt)].astype(np.int32)))
        me.CreateDoubleSidedAttr(True)          # the inside of the jar's wall is now in view
    r_in, r_out, z_in, z_out = FUNNEL
    v, f = _lathe([(0.0, z_in), (r_in, z_in), (r_out, z_out), (r_out, FLOOR_Z)])
    fill = UsdGeom.Mesh.Define(stage, JAR + "/powder_fill")
    fill.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    fill.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
    fill.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
    fill.CreateSubdivisionSchemeAttr("none")
    fill.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(fill.GetPrim()).Bind(_material(stage))
    return True


class Grains:
    """The grain pass's spheres as one point instancer, posed per film frame (nearest grain-pass time)."""

    def __init__(self, stage, path, shot_t):
        G = np.load(path, allow_pickle=True)
        self.pos = np.asarray(G["grain_pos"], np.float32)
        tg = np.asarray(G["t"], float)
        self.map = np.clip(np.searchsorted(tg, np.asarray(shot_t, float) - 1e-6), 0, len(tg) - 1)
        inst = UsdGeom.PointInstancer.Define(stage, "/World/Powder/grains")
        proto = UsdGeom.Sphere.Define(stage, "/World/Powder/grains/grain")
        proto.CreateRadiusAttr(float(G["grain_radius"]))
        UsdShade.MaterialBindingAPI.Apply(proto.GetPrim()).Bind(_material(stage))
        inst.CreatePrototypesRel().SetTargets([proto.GetPath()])
        inst.CreateProtoIndicesAttr(Vt.IntArray([0] * self.pos.shape[1]))
        self.positions = inst.CreatePositionsAttr()
        self.update(0)

    def update(self, k):
        self.positions.Set(Vt.Vec3fArray.FromNumpy(self.pos[self.map[k]]))
