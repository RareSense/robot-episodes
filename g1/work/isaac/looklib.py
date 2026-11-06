"""Shared helpers for the look layers of the tea scene (USD overrides; the twin file itself is never edited).

    import looklib as L
    stage, layer = L.open_layer("looks_room.usda")    # composed tea_scene.usda + this layer as the strongest edit target
    m = L.pbr("room_floor", L.polyhaven("floor_tiles_06"), tile=1.2)           # Poly Haven CC0 set
    m = L.pbr("room_floor", L.nano_set(r"...\\texgen\\out", "floor_tiles"))     # Nano Banana set (nano_texture.py)
    L.bind(stage, "/World/Objects/object_0001", m, tile=1.2)                    # box UVs (metres per tile) + binding
    layer.Save()

Materials are UsdPreviewSurface with sRGB albedo, raw roughness, OpenGL normal maps; texture files are copied to
isaac/pbr/<material>/ so the layers stay relocatable. Box UVs are world-space (faceVarying primvars:st), so the
texture scale is real (metres per tile) on every face."""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

ISAAC = Path(str(WORK / "isaac"))
SCENE = ISAAC / "tea_scene.usda"
PH_DIR = ISAAC / "pbr_src"
NAMES = json.load(open(ISAAC / "prim_names.json"))     # "/World/Objects/object_xxxx": [name, class, colour]
_stage = None


def open_layer(filename):
    """The composed scene with a fresh layer `filename` (in isaac/) as the strongest sublayer and edit target."""
    global _stage
    path = ISAAC / filename
    path.unlink(missing_ok=True)
    layer = Sdf.Layer.CreateNew(str(path))
    layer.pseudoRoot.SetInfo("metersPerUnit", 1.0)
    layer.pseudoRoot.SetInfo("upAxis", "Z")
    stage = Usd.Stage.Open(str(SCENE))
    subs = list(stage.GetRootLayer().subLayerPaths)
    subs = [s for s in subs if Path(s).name != filename]
    stage.GetRootLayer().subLayerPaths = [str(path)] + subs     # in memory only: tea_scene.usda is not saved here
    stage.SetEditTarget(layer)
    _stage = stage
    return stage, layer


def _get(url):
    """Poly Haven answers 403 to Python's default User-Agent."""
    req = urllib.request.Request(url, headers={"User-Agent": "tea-twin-looks/1.0 "})
    return urllib.request.urlopen(req, timeout=120).read()


def polyhaven(asset, res="2k"):
    """Download a Poly Haven texture set (CC0) once; returns {"diff","rough","nor","ao"} -> local file paths."""
    dst = PH_DIR / asset
    dst.mkdir(parents=True, exist_ok=True)
    files = json.loads(_get(f"https://api.polyhaven.com/files/{asset}"))
    want = {"diff": ["Diffuse", "diff"], "rough": ["Rough", "rough"], "nor": ["nor_gl"], "ao": ["AO", "ao"]}
    out = {}
    for key, cands in want.items():
        for c in cands:
            if c in files and res in files[c]:
                fmt = "png" if "png" in files[c][res] and key != "diff" else "jpg"
                url = files[c][res][fmt]["url"]
                f = dst / Path(url).name
                if not f.exists():
                    f.write_bytes(_get(url))
                out[key] = str(f)
                break
    if "diff" not in out:
        raise KeyError(f"{asset}: no diffuse map at {res} (keys {list(files)})")
    return out


def nano_set(folder, name):
    """A texture set written by texgen/nano_texture.py."""
    f = Path(folder)
    out = {"diff": str(f / f"{name}_diff.png")}
    for k, suf in (("rough", "_rough.png"), ("nor", "_nor_gl.png")):
        if (f / f"{name}{suf}").exists():
            out[k] = str(f / f"{name}{suf}")
    return out


def _tex(m, name, file, cs, st):
    t = UsdShade.Shader.Define(_stage, m.GetPath().AppendChild(name))
    t.CreateIdAttr("UsdUVTexture")
    t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file)
    t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(cs)
    t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
    return t


def pbr(name, maps, rough=0.5, tint=None, metal=0.0, clearcoat=0.0, normal_scale=1.0, opacity=1.0, ior=1.5):
    """Textured material /World/Looks/<name>; `maps` from polyhaven() or nano_set(). tint multiplies the albedo."""
    dst = ISAAC / "pbr" / name
    dst.mkdir(parents=True, exist_ok=True)
    rel = {}
    for k, f in maps.items():
        shutil.copy2(f, dst / Path(f).name)
        rel[k] = f"./pbr/{name}/{Path(f).name}"
    m = UsdShade.Material.Define(_stage, f"/World/Looks/{name}")
    rd = UsdShade.Shader.Define(_stage, m.GetPath().AppendChild("st"))
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st = rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    ps = UsdShade.Shader.Define(_stage, m.GetPath().AppendChild("S"))
    ps.CreateIdAttr("UsdPreviewSurface")
    d = _tex(m, "diff", rel["diff"], "sRGB", st)
    if tint is not None:
        d.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*tint, 1.0))
    ps.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(d.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    if "rough" in rel:
        r = _tex(m, "rough", rel["rough"], "raw", st)
        ps.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(r.CreateOutput("r", Sdf.ValueTypeNames.Float))
    else:
        ps.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    if "nor" in rel:
        n = _tex(m, "nor", rel["nor"], "raw", st)
        s = 2.0 * normal_scale
        n.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(s, s, 2.0, 1))
        n.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-normal_scale, -normal_scale, -1, 0))
        ps.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(n.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    if "ao" in rel:
        a = _tex(m, "ao", rel["ao"], "raw", st)
        ps.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(a.CreateOutput("r", Sdf.ValueTypeNames.Float))
    ps.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    ps.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(clearcoat)
    if opacity < 1:
        ps.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        ps.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    m.CreateSurfaceOutput().ConnectToSource(ps.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return m


def plain(name, rgb, rough, metal=0.0, opacity=1.0, ior=1.5, clearcoat=0.0, emissive=None):
    m = UsdShade.Material.Define(_stage, f"/World/Looks/{name}")
    ps = UsdShade.Shader.Define(_stage, m.GetPath().AppendChild("S"))
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


def box_uv(prim, tile):
    """World-space box projection, `tile` metres per texture repeat (faceVarying st on a triangle mesh)."""
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    W = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    pts = pts @ W[:3, :3] + W[3, :3]
    cnt = np.array(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    p0, p1, p2 = pts[idx[start]], pts[idx[start + 1]], pts[idx[start + 2]]
    dom = np.argmax(np.abs(np.cross(p1 - p0, p2 - p0)), axis=1)          # dominant normal axis per face
    axes = np.array([[1, 2], [0, 2], [0, 1]])[np.repeat(dom, cnt)]
    c = pts[idx]
    uv = np.stack([c[np.arange(len(c)), axes[:, 0]], c[np.arange(len(c)), axes[:, 1]]], axis=1) / tile
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set([Gf.Vec2f(float(a), float(b)) for a, b in uv])


def flat_normals(prim):
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    cnt = np.array(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    n = np.cross(pts[idx[start + 1]] - pts[idx[start]], pts[idx[start + 2]] - pts[idx[start]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, v)) for v in np.repeat(n, cnt, axis=0)])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)


def meshes(path):
    """The renderable meshes under an object prim (its 'visual' mesh and any other non-collision mesh)."""
    return [p for p in Usd.PrimRange(_stage.GetPrimAtPath(path))
            if p.IsA(UsdGeom.Mesh) and not p.GetName().startswith("collision")]


def bind(stage, path, material, tile=None, flat=False):
    """Bind `material` to every renderable mesh under `path`; tile (metres) adds box UVs."""
    n = 0
    for p in meshes(path):
        if tile:
            box_uv(p, tile)
        if flat:
            flat_normals(p)
        UsdShade.MaterialBindingAPI.Apply(p).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        n += 1
    return n


def objects(cls=None, name_has=None):
    """Object prim paths by semantic class and/or a name substring (from prim_names.json)."""
    return [p for p, (n, c, _) in NAMES.items()
            if (cls is None or c == cls) and (name_has is None or name_has in n)]
