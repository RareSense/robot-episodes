"""Looks layer for the tea twin (USD override; the twin file itself is untouched): Poly Haven CC0 PBR
materials by semantic class with box-mapped world-scale UVs, glossy plastic / ceramic / glass for
the props, the Poly Haven HDRI sky dome and interior lights at the ceiling fixtures.
Writes work/isaac/looks.usda (a sublayer stacked over environment.usda)."""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import json
import shutil
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

TWIN = Path(str(HANDOFF / "project/tea_twin/scene_v3/environment.usda"))
OUT = Path(str(WORK / "isaac"))
PH = {  # material -> (Poly Haven set folder, metres per texture tile, fallback roughness, colour tint)
    "floor": (str(WORK / "isaac/pbr_src" / "floor_tiles_06"), 1.6, 0.45, None),
    # the real worktop: amber wood laminate with a fine straight grain (oak veneer grain, tinted to the video)
    "oak": (str(WORK / "isaac/pbr_src" / "oak_veneer_03"), 1.2, 0.35, (0.95, 0.60, 0.30)),
    "wall": (str(WORK / "isaac/pbr_src" / "beige_wall_001"), 2.0, 0.8, None),
}
# sky only (no ground or buildings in it): the twin's own exterior (trees, rooftops) stays the view
HDRI = str(WORK / "isaac/pbr_src" / "kloofendal_48d_partly_cloudy_puresky_4k.hdr")
names = json.load(open(OUT / "prim_names.json"))

stage = Usd.Stage.Open(str(TWIN))
(OUT / "looks.usda").unlink(missing_ok=True)
layer = Sdf.Layer.CreateNew(str(OUT / "looks.usda"))
layer.pseudoRoot.SetInfo("metersPerUnit", 1.0)          # same units as the twin (Isaac's metrics check)
layer.pseudoRoot.SetInfo("upAxis", "Z")
stage.GetRootLayer().subLayerPaths.insert(0, str(OUT / "looks.usda"))
stage.SetEditTarget(layer)
xc = UsdGeom.XformCache()
tex_dir = OUT / "pbr"


def copy_set(name, folder):
    """Copy a Poly Haven set's maps next to the layer (diff / rough / nor / ao)."""
    dst = tex_dir / name
    dst.mkdir(parents=True, exist_ok=True)
    maps = {}
    for f in Path(folder).iterdir():
        low = f.name.lower()
        for key in ("diff", "rough", "nor", "ao"):
            if key in low and not low.endswith(".png") and (key != "nor" or "gl" in low or low.startswith("nor")):
                if not (dst / f.name).exists():
                    shutil.copy2(f, dst / f.name)
                maps[key] = f"pbr/{name}/{f.name}"
    return maps


def tex_node(mpath, name, file, color_space, st_out):
    t = UsdShade.Shader.Define(stage, mpath.AppendChild(name))
    t.CreateIdAttr("UsdUVTexture")
    t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file)
    t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
    t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
    return t


def pbr_material(name, folder, rough, tint=None):
    maps = copy_set(name, folder)
    mpath = Sdf.Path(f"/World/Looks/{name}")
    m = UsdShade.Material.Define(stage, mpath)
    rd = UsdShade.Shader.Define(stage, mpath.AppendChild("st"))
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_out = rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    ps = UsdShade.Shader.Define(stage, mpath.AppendChild("S"))
    ps.CreateIdAttr("UsdPreviewSurface")
    d = tex_node(mpath, "diff", maps["diff"], "sRGB", st_out)
    if tint is not None:                                   # the real surface's colour over the Poly Haven grain
        d.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*tint, 1.0))
    ps.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(d.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    if "rough" in maps:
        r = tex_node(mpath, "rough", maps["rough"], "raw", st_out)
        ps.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(r.CreateOutput("r", Sdf.ValueTypeNames.Float))
    else:
        ps.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    if "nor" in maps:
        n = tex_node(mpath, "nor", maps["nor"], "raw", st_out)
        n.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        n.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        ps.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(n.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    ps.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    m.CreateSurfaceOutput().ConnectToSource(ps.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return m


def plain(name, rgb, rough, metal=0.0, opacity=1.0, ior=1.5, clearcoat=0.0, emissive=None):
    mpath = Sdf.Path(f"/World/Looks/{name}")
    m = UsdShade.Material.Define(stage, mpath)
    ps = UsdShade.Shader.Define(stage, mpath.AppendChild("S"))
    ps.CreateIdAttr("UsdPreviewSurface")
    ps.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    if emissive is not None:
        ps.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    ps.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    ps.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    ps.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(clearcoat)
    if opacity < 1:
        ps.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        ps.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    m.CreateSurfaceOutput().ConnectToSource(ps.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return m


def flat_normals(prim):
    """Per-face normals: the twin's window pane has a few big faces with smoothed vertex normals,
    which render as a warped, bulging mirror."""
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    F = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    n = np.cross(pts[F[:, 1]] - pts[F[:, 0]], pts[F[:, 2]] - pts[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, v)) for v in n.repeat(3, axis=0)])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)


def box_uv(prim, tile):
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    W = np.array(xc.GetLocalToWorldTransform(prim))
    pts = pts @ W[:3, :3] + W[3, :3]
    F = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    tri = pts[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    dom = np.argmax(np.abs(nrm), axis=1)
    axes = np.array([[1, 2], [0, 2], [0, 1]])
    corners = tri.reshape(-1, 3)
    ax = axes[dom].repeat(3, axis=0)
    uv = np.stack([corners[np.arange(len(corners)), ax[:, 0]], corners[np.arange(len(corners)), ax[:, 1]]], axis=1) / tile
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set([Gf.Vec2f(float(a), float(b)) for a, b in uv])


looks = {k: pbr_material(k, folder, rough, tint) for k, (folder, tile, rough, tint) in PH.items()}
looks.update(
    cabinet=plain("cabinet_dark", (0.045, 0.045, 0.05), 0.45, clearcoat=0.3),
    ceiling=plain("ceiling_white", (0.86, 0.86, 0.84), 0.9),
    glass=plain("window_glass", (0.85, 0.9, 0.92), 0.02, opacity=0.08),
    black_plastic=plain("black_plastic", (0.025, 0.025, 0.028), 0.28, clearcoat=0.5),
    black_ceramic=plain("black_ceramic", (0.02, 0.02, 0.022), 0.12, clearcoat=0.8),
    tray_plastic=plain("tray_plastic", (0.03, 0.03, 0.03), 0.5),
    jar_glass=plain("jar_glass", (0.9, 0.9, 0.88), 0.05, opacity=0.35),
    steel=plain("brushed_steel", (0.62, 0.63, 0.65), 0.35, metal=1.0),
    led=plain("led_diffuser", (0.95, 0.95, 0.95), 0.5, emissive=(6.0, 5.8, 5.4)),
)
CLASS_LOOK = {"floor": "floor", "worktop": "oak", "wall": "wall", "wall_finish": "wall", "ceiling": "ceiling",
              "cabinet": "cabinet", "glass": "glass"}
NAME_LOOK = {"led_diffuser": "led", "kettle_jug": "black_plastic", "kettle_base": "black_plastic", "mug": "black_ceramic",
             "sachet_tray": "tray_plastic", "glass_jar": "jar_glass", "window_frame": "cabinet", "mullions": "cabinet"}
TILE = {"floor": PH["floor"][1], "oak": PH["oak"][1], "wall": PH["wall"][1]}
n = 0
for path, (name, cls, col) in names.items():
    vis = stage.GetPrimAtPath(path + "/visual")
    if not vis:
        continue
    look = next((v for k, v in NAME_LOOK.items() if k in name), None) or CLASS_LOOK.get(cls)
    if cls == "sky":                                   # the HDRI dome is the sky
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).CreateVisibilityAttr().Set("invisible")
        continue
    if look is None:
        continue
    if look in TILE:
        box_uv(vis, TILE[look])
    if look in ("glass", "cabinet", "oak", "wall", "floor", "ceiling"):
        flat_normals(vis)
    UsdGeom.Mesh(vis).CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    UsdShade.MaterialBindingAPI.Apply(vis).Bind(looks[look], UsdShade.Tokens.strongerThanDescendants)
    n += 1
# collision meshes never render
for p in stage.Traverse():
    if p.IsA(UsdGeom.Mesh) and p.GetName().startswith("collision"):
        UsdGeom.Imageable(p).CreateVisibilityAttr().Set("invisible")
# sky and daylight: the Poly Haven HDRI dome; interior lights at the ceiling fixtures near the tea corner
dome = UsdLux.DomeLight.Define(stage, "/World/Lights/hdri_sky")
dome.CreateTextureFileAttr().Set(HDRI)
dome.CreateTextureFormatAttr().Set("latlong")
dome.CreateIntensityAttr(1000.0)
nl = 0
for path, (name, cls, col) in names.items():
    if cls != "light" or "diffuser" not in name:
        continue
    vis = stage.GetPrimAtPath(path + "/visual")
    ext = vis.GetAttribute("extent").Get() if vis else None
    if not ext:
        continue
    W = np.array(xc.GetLocalToWorldTransform(vis))
    lo, hi = np.array(ext[0]) @ W[:3, :3] + W[3, :3], np.array(ext[1]) @ W[:3, :3] + W[3, :3]
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    c = (lo + hi) / 2
    L = UsdLux.RectLight.Define(stage, f"/World/Lights/fixture_{nl}")
    UsdGeom.Xformable(L).AddTranslateOp().Set(Gf.Vec3d(float(c[0]), float(c[1]), float(lo[2]) - 0.01))
    L.CreateWidthAttr(float(max(hi[0] - lo[0], 0.1))); L.CreateHeightAttr(float(max(hi[1] - lo[1], 0.1)))
    L.CreateIntensityAttr(30000.0)
    L.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    L.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)   # the diffuser mesh shows
    nl += 1
# the white ceiling bouncing the strips' light back down over the tea corner (soft fill, unseen by cameras)
B = UsdLux.RectLight.Define(stage, "/World/Lights/ceiling_bounce")
UsdGeom.Xformable(B).AddTranslateOp().Set(Gf.Vec3d(1.9, 1.5, 2.75))
B.CreateWidthAttr(2.6); B.CreateHeightAttr(2.2)
B.CreateIntensityAttr(1500.0)
B.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
B.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
layer.Save()
print("materials bound:", n, "fixture lights:", nl, "->", OUT / "looks.usda")
