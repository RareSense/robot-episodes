"""Exterior looks for the tea twin: everything seen through the windows (street pavers, cars, buildings, walls,
gate, poles, trees) plus the sky. Rebuilds work/isaac/looks_exterior.usda from scratch.

  python work/isaac\\make_looks_exterior.py

Sources
- Street pavers: Nano Banana texture from the person's video (frames 14 s and 30 s: grey zigzag 'uni' pavers in
  stretcher bond, ~22.5 x 11 cm). The model's raw output is periodic (10 blocks x 20 rows), so it is used without
  the cross-fade (which ghosts a regular joint pattern) and tiled at its real size (2.25 m).
- Trees: Poly Haven bark + a Nano Banana leaf canopy texture from the shrub in the video (frame 14 s).
- Buildings / walls / ground: Poly Haven CC0 sets tinted to the twin's own colours (sampled from the video).
- Cars, glass, rubber, plastic, metal: procedural PBR sets (uniform paint with a slight orange-peel normal and a
  dust-varied roughness), written to pbr_src/exterior_proc/.
- Sky: Poly Haven qwantani_noon_puresky (clear midday, sky only). The sun disc is cut out of the HDRI
  (pbr_src/qwantani_noon_puresky_4k_nosun.hdr, made once with OpenCV from the system Python) and replaced by a
  distant light /World/Lights/exterior_sun, so the hard sunlight's direction and strength are set here: behind
  the building, as in the video, where the building's shadow covers the near street and the far street is in full
  sun; the strength was tuned with match_view so the sunlit pavers do not clip (the HDRI's own sun did)."""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import json
import math
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(WORK / "isaac"))
sys.path.insert(0, str(WORK / "isaac/texgen"))
import looklib as L  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

ISAAC = L.ISAAC
NANO = ISAAC / "texgen" / "out_exterior"
PROC = ISAAC / "pbr_src" / "exterior_proc"
HDRI_SRC = ISAAC / "pbr_src" / "qwantani_noon_puresky_4k.hdr"
HDRI = ISAAC / "pbr_src" / "qwantani_noon_puresky_4k_nosun.hdr"
HDRI_SUN = PROC / "qwantani_noon_sun.json"
SYS_PY = (sys.executable)   # has OpenCV (.hdr I/O)

# ---- lighting (tuned against the video with texgen/match_view.py)
DOME = 2000.0                      # dome (sky without the sun) intensity
DOME_COLOR = (1.0, 0.88, 0.72)     # the clean blue South-African sky greyed towards the video's dusty, hazy one
SUN = 3500.0                       # distant light intensity (r1: the HDRI's own sun, 9500, blew the street out;
                                   # r2: 4500 still clipped the sunlit pavers at ~230)
SUN_ELEV = 30.0                    # degrees (lead, 30 Oct: was 52; lower = the twin building's shadow reaches the
                                   # near street, which lies in shade in the video)
SUN_FROM = (-0.25, -0.97)          # horizontal direction towards the sun: behind the building (-y), so its
                                   # shadow covers the near street as in the video
SUN_TINT = (1.0, 0.9, 0.76)        # warm, dusty light: the video's sunlit pavers are (234, 215, 165)
DOME_ROT_Z = 0.0

PAVER_TILE = 2.25                  # 10 blocks x 0.225 m


# ------------------------------------------------------------------ texture preparation
def srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def mean_albedo(path):
    a = np.asarray(Image.open(path).convert("RGB").resize((64, 64), Image.BILINEAR), np.float64) / 255
    return lin(a).reshape(-1, 3).mean(0)


def noise(size, scale, seed):
    """Tileable smooth noise (FFT-filtered white noise), zero mean, unit std."""
    r = np.random.default_rng(seed)
    f = np.fft.fftfreq(size)
    k = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    n = np.real(np.fft.ifft2(np.fft.fft2(r.standard_normal((size, size))) * np.exp(-(k * scale) ** 2)))
    return (n - n.mean()) / (n.std() + 1e-9)


def proc_set(name, rgb, rough, var=0.05, bump=0.3, size=512, seed=1):
    """A procedural PBR set: albedo `rgb` (linear) with 3% mottling, roughness `rough` +- `var` (dust/wear),
    normal map from fine noise (orange peel / grain) of strength `bump`."""
    PROC.mkdir(parents=True, exist_ok=True)
    out = {"diff": PROC / f"{name}_diff.png", "rough": PROC / f"{name}_rough.png", "nor": PROC / f"{name}_nor_gl.png"}
    m = noise(size, 40, seed)
    a = np.asarray(rgb)[None, None, :] * (1 + 0.03 * m[..., None])
    Image.fromarray((srgb(a) * 255 + 0.5).astype(np.uint8)).save(out["diff"])
    rr = np.clip(rough + var * noise(size, 25, seed + 1), 0.02, 1)
    Image.fromarray((rr * 255 + 0.5).astype(np.uint8)).save(out["rough"])
    h = noise(size, 3, seed + 2) * bump * 0.02 + noise(size, 12, seed + 3) * bump * 0.05
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * size / 64      # r3: /8 made paint look like snow
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * size / 64
    n = np.dstack([-dx, dy, np.ones_like(dx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    Image.fromarray(((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)).save(out["nor"])
    return {k: str(v) for k, v in out.items()}


def pavers_seamless():
    """The Nano Banana paver texture without the cross-fade: the raw output is already periodic (10 x 20 blocks).
    Its joints are ~6 mm: seen from the 2nd floor (a block is ~12 px in a 540x960 view) they filter away and the
    street reads as flat white, while in the video the dusty joints + their shadow read clearly. So the joints are
    widened to ~1.5 cm and darkened (dirt) before the normal / roughness maps are derived."""
    import nano_texture as NT
    from PIL import ImageFilter
    name = "ext_pavers_seamless"
    src = Image.open(NANO / "ext_pavers_src.png").convert("RGB").resize((2048, 2048), Image.LANCZOS)
    g = src.convert("L").filter(ImageFilter.GaussianBlur(2))
    d = np.asarray(g.filter(ImageFilter.GaussianBlur(12)), np.float32) - np.asarray(g, np.float32)
    joint = Image.fromarray(((d > 10) * 255).astype(np.uint8))
    joint = joint.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(11))   # lines stay, pores go
    joint = joint.filter(ImageFilter.GaussianBlur(2))
    j = np.asarray(joint, np.float32)[..., None] / 255
    dirt = np.array([95, 92, 86], np.float32)                     # dusty dark-grey joint fill (sRGB)
    alb = Image.fromarray(np.clip(np.asarray(src, np.float32) * (1 - 0.55 * j) + dirt * 0.55 * j, 0, 255)
                          .astype(np.uint8))
    alb.save(NANO / f"{name}_diff.png")
    nor, rgh = NT.derived_maps(alb, 0.82, 0.12, 3.0)
    nor.save(NANO / f"{name}_nor_gl.png")
    rgh.save(NANO / f"{name}_rough.png")
    return L.nano_set(NANO, name)


CLIP = r'''
import cv2, json, numpy as np, sys
src, dst, js = sys.argv[1:4]
im = cv2.imread(src, cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float64)
H, W, _ = im.shape
Y = im @ [0.2126, 0.7152, 0.0722]
r, c = np.unravel_index(np.argmax(Y), Y.shape)
mask = (Y > 20).astype(np.uint8)
mask = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
ring = (cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~mask
fill = im[ring].mean(0)
th = (np.arange(H) + .5) / H * np.pi
dO = (np.pi / H) * (2 * np.pi / W) * np.sin(th)[:, None]
E = ((im - fill) * (dO * mask)[..., None]).sum((0, 1))
out = im.copy(); out[mask] = fill
cv2.imwrite(dst, out[..., ::-1].astype(np.float32))
json.dump({"E_sun_rgb": E.tolist(), "elev_deg": 90 - (r + .5) / H * 180, "u": (c + .5) / W,
           "pixels": int(mask.sum())}, open(js, "w"), indent=1)
'''


def hdri_without_sun():
    if not (HDRI.exists() and HDRI_SUN.exists()):
        PROC.mkdir(parents=True, exist_ok=True)
        subprocess.run([SYS_PY, "-I", "-c", CLIP, str(HDRI_SRC), str(HDRI), str(HDRI_SUN)], check=True)
    return json.load(open(HDRI_SUN))


# ------------------------------------------------------------------ UVs for the dense tree meshes
def vertex_uv(prim, tile):
    """Per-vertex box projection (dominant axis of the vertex normal), `tile` metres per repeat: a quarter of the
    size of the faceVarying UVs on the 800k-face trees, and seams do not show in foliage or bark."""
    mesh = UsdGeom.Mesh(prim)
    W = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64) @ W[:3, :3] + W[3, :3]
    nrm = mesh.GetNormalsAttr().Get()
    if nrm is not None and len(nrm) == len(pts):
        n = np.array(nrm, dtype=np.float64) @ W[:3, :3]
    else:
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        fn = np.cross(pts[idx[:, 1]] - pts[idx[:, 0]], pts[idx[:, 2]] - pts[idx[:, 0]])
        n = np.zeros_like(pts)
        for j in range(3):
            np.add.at(n, idx[:, j], fn)
    dom = np.argmax(np.abs(n), axis=1)
    ax = np.array([[1, 2], [0, 2], [0, 1]])[dom]
    uv = np.stack([pts[np.arange(len(pts)), ax[:, 0]], pts[np.arange(len(pts)), ax[:, 1]]], 1) / tile
    uv = np.round(uv, 3).astype(np.float32)
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray.FromNumpy(uv))


# ------------------------------------------------------------------ what each object is made of
CREAM = (0.55, 0.44, 0.28)        # video 14 s: the band is 1.4 / 1.17 / 0.9 x the grey pavers (linear)
ASPHALT = (0.10, 0.10, 0.095)
SOIL = (0.24, 0.20, 0.15)
RULES = [  # (name substring, kind, colour: None = the twin's own colour for that object)
    ("paver_field", "pavers", "tex"), ("paver_joints", "pavers", "tex"), ("cream_bands", "pavers", CREAM),
    ("drain_channel", "panel", (0.02, 0.02, 0.02)),
    ("road_tarmac", "asphalt", ASPHALT), ("back_lane", "asphalt", ASPHALT), ("surrounding_ground", "soil", SOIL),
    ("r47i", None, None),                                   # cars: by part below
    ("motorbikes_chrome", "metal", (0.62, 0.62, 0.62)), ("motorbikes_frames", "plastic", (0.02, 0.02, 0.022)),
    ("yellow_vehicle", "carpaint", None), ("vehicle_wheels", "rubber", (0.025, 0.025, 0.025)),
    ("facade_glass", "glass", None), ("window_bands", "glass", None), ("side_windows", "glass", None),
    ("r53i1_windows", "glass", None),
    ("ac_casing", "panel", (0.75, 0.75, 0.72)), ("ac_fan_grille", "plastic", None),
    ("acp_fascia", "panel", None), ("clad_wall", "panel", None), ("shutter_guide_frame", "panel", None),
    ("gate", "panel", None), ("facade_fins", "panel", None), ("crossarms", "panel", None),
    ("logo_disc", "plastic", None), ("signboard", "plastic", None),
    ("logo_letter", "paint", None), ("sign_lettering", "paint", None),
    ("roof_sheets", "corrugated", None), ("roof_ribs", "corrugated", None), ("container_shell", "corrugated", None),
    ("container_doors", "corrugated", None), ("blue_roof_strip", "panel", None),
    ("water_tanks", "plastic", None),
    ("brick_courses", "brick", None), ("r53i1_main_block", "brick", None), ("r53i1_roof_parapet", "brick", None),
    ("roof_slab", "concrete", None), ("coping", "concrete", None), ("umbrella_pole", "plastic", None),
    ("_pole", "concrete", None), ("roof_structures", "concrete", None), ("rooftop_stair_box", "concrete", None),
    ("insulators", "ceramic", None), ("overhead_wires", "plastic", None),
    ("umbrella_canopy", "fabric", None), ("bench_frame", "plastic", None), ("bench_slats", "wood", None),
]
CAR = [("_body", "carpaint", None), ("glazing", "glass", (0.012, 0.014, 0.016)),
       ("tyres_and_trim", "rubber", (0.022, 0.022, 0.022)), ("wheels_and_lamps", "metal", (0.55, 0.55, 0.56))]
# kind -> (texture source, metres per tile, pbr() keywords)
KINDS = {
    "pavers": ("pavers", PAVER_TILE, {}),
    "asphalt": ("ph:asphalt_02", 4.0, {}),
    "soil": ("ph:dry_ground_01", 6.0, {}),
    "plaster": ("ph:plastered_wall_04", 3.0, {}),
    "concrete": ("ph:concrete_wall_008", 3.0, {}),
    "brick": ("ph:red_brick_03", 1.6, {}),
    "panel": ("ph:metal_plate", 1.5, {"metal": 0.3}),
    "corrugated": ("ph:corrugated_iron_02", 1.2, {"metal": 0.4}),
    "bark": ("ph:bark_platanus", 1.2, {}),
    "foliage": ("nano:ext_foliage", 1.6, {}),
    "fabric": ("ph:rough_linen", 0.6, {}),
    "wood": ("ph:oak_wood_planks", 1.0, {}),
    # procedural: (rough, rough variation, bump), tile, pbr keywords
    "carpaint": ((0.28, 0.05, 0.15), 2.0, {"clearcoat": 1.0, "metal": 0.15}),
    "glass": ((0.04, 0.02, 0.02), 2.0, {"clearcoat": 0.0}),
    "rubber": ((0.85, 0.06, 0.6), 0.5, {}),
    "plastic": ((0.45, 0.08, 0.3), 1.0, {}),
    "metal": ((0.3, 0.06, 0.2), 1.0, {"metal": 1.0}),
    "ceramic": ((0.15, 0.03, 0.1), 0.5, {"clearcoat": 0.5}),
    "paint": ((0.5, 0.06, 0.2), 1.0, {}),
}
DENSE = {"bark", "foliage"}     # tree meshes: vertex UVs


def classify(name, cls, col):
    if cls == "foliage":
        return "foliage", col
    if cls == "tree":
        return "bark", (0.09, 0.065, 0.045)
    for sub, kind, c in RULES:
        if sub in name:
            if kind is None:
                kind, c = next((k, cc) for s, k, cc in CAR if s in name)
            return kind, (col if c is None else c)
    if cls in ("exterior", "gate"):
        return ("panel" if cls == "gate" else "plaster"), col
    return None, None


_maps = {}


def polyhaven(asset, res="2k"):
    """looklib.polyhaven, but with a browser User-Agent (the Poly Haven API answers 403 to Python-urllib) and
    offline reuse of a set that is already in pbr_src/."""
    dst = L.PH_DIR / asset
    local = {}
    for key, pat in (("diff", f"*_diff_{res}.jpg"), ("rough", f"*_rough_{res}.*"), ("nor", f"*_nor_gl_{res}.*"),
                     ("ao", f"*_ao_{res}.*")):
        hit = sorted(dst.glob(pat)) if dst.exists() else []
        if hit:
            local[key] = str(hit[0])
    if {"diff", "rough", "nor"} <= set(local):
        return local
    dst.mkdir(parents=True, exist_ok=True)
    get = lambda url: urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),  # noqa: E731
                                             timeout=120).read()
    files = json.loads(get(f"https://api.polyhaven.com/files/{asset}"))
    want = {"diff": ["Diffuse", "diff"], "rough": ["Rough", "rough"], "nor": ["nor_gl"], "ao": ["AO", "ao"]}
    out = {}
    for key, cands in want.items():
        for c in cands:
            if c in files and res in files[c]:
                fmt = "png" if "png" in files[c][res] and key != "diff" else "jpg"
                url = files[c][res][fmt]["url"]
                f = dst / Path(url).name
                if not f.exists():
                    f.write_bytes(get(url))
                out[key] = str(f)
                break
    return out


def maps_for(src):
    if src not in _maps:
        typ, _, ident = src.partition(":")
        _maps[src] = (pavers_seamless() if typ == "pavers" else polyhaven(ident) if typ == "ph"
                      else L.nano_set(NANO, ident))
    return _maps[src]


_mats = {}


def material(kind, col):
    src, tile, kw = KINDS[kind]
    key = (kind, "tex" if col == "tex" else tuple(round(float(v), 3) for v in col))
    if key in _mats:
        return _mats[key], tile
    tag = "" if col == "tex" else "_" + "".join(f"{int(round(float(v) * 255)):02x}" for v in col)
    name = f"exterior_{kind}{tag}"
    if isinstance(src, tuple):                              # procedural set in the object's colour
        rough, var, bump = src
        c = np.asarray(col, np.float64) * min(1.0, 0.72 / max(max(col), 1e-6))   # white paint ~0.72, not 0.83
        m = L.pbr(name, proc_set(name, c, rough, var, bump, seed=len(_mats) + 1), **kw)
    else:
        maps = maps_for(src)
        tint = None
        if col != "tex":
            want = np.asarray(col, np.float64)
            if kind == "foliage":                           # the twin's sun/shade split, softened (RT shades too)
                want = np.sqrt(want / np.array([0.15, 0.29, 0.06])) * np.array([0.055, 0.11, 0.025])
            have = mean_albedo(maps["diff"])
            want = want * min(1.0, 0.7 / max(want.max(), 1e-6))          # no albedo above 0.7
            tint = tuple(float(v) for v in np.clip(want / np.maximum(have, 1e-4), 0, 4))
        m = L.pbr(name, maps, tint=tint, **kw)
        share(m, name, f"exterior_{kind}")
    _mats[key] = m
    return m, tile


def share(m, name, folder):
    """Tinted variants of one texture set read ONE copy (pbr/exterior_<kind>/): pbr() copies the maps per material
    (1.3 GB and 63 sets of 2k textures on the GPU otherwise)."""
    if name == folder:
        return
    src, dst = ISAAC / "pbr" / name, ISAAC / "pbr" / folder
    dst.mkdir(exist_ok=True)
    for f in src.iterdir():
        if not (dst / f.name).exists():
            shutil.move(str(f), dst / f.name)
    shutil.rmtree(src)
    for c in ("diff", "rough", "nor", "ao"):
        prim = L._stage.GetPrimAtPath(m.GetPath().AppendChild(c))
        if prim:
            inp = UsdShade.Shader(prim).GetInput("file")
            inp.Set(Sdf.AssetPath(f"./pbr/{folder}/{Path(inp.Get().path).name}"))


# ------------------------------------------------------------------ build the layer
if __name__ == "__main__":
    sun = hdri_without_sun()
    for old in (ISAAC / "pbr").glob("exterior_*"):          # this script's own texture copies: rebuilt below
        shutil.rmtree(old)
    stage, layer = L.open_layer("looks_exterior.usda")
    table = {}
    for path, (name, cls, col) in L.NAMES.items():
        if cls not in ("exterior", "ground", "tree", "foliage", "gate"):
            continue
        kind, c = classify(name, cls, col)
        if kind is None:
            continue
        m, tile = material(kind, c)
        if kind in DENSE:
            for p in L.meshes(path):
                vertex_uv(p, tile)
                UsdShade.MaterialBindingAPI.Apply(p).Bind(m, UsdShade.Tokens.strongerThanDescendants)
        else:
            L.bind(stage, path, m, tile=tile)
        table.setdefault(m.GetPath().name, []).append(name)

    # sky: Poly Haven pure sky without its sun disc + the sun as a distant light with the same irradiance
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Lights/hdri_sky"))
    dome.CreateTextureFileAttr().Set(Sdf.AssetPath("./pbr_src/" + HDRI.name))
    dome.CreateTextureFormatAttr().Set("latlong")
    dome.CreateIntensityAttr().Set(DOME)
    dome.CreateColorAttr().Set(Gf.Vec3f(*DOME_COLOR))
    if DOME_ROT_Z:
        UsdGeom.Xformable(dome).AddRotateZOp(opSuffix="exterior").Set(DOME_ROT_Z)
    E = np.array(sun["E_sun_rgb"])
    el = math.radians(SUN_ELEV)
    h = np.array(SUN_FROM) / np.linalg.norm(SUN_FROM)
    to_sun = np.array([h[0] * math.cos(el), h[1] * math.cos(el), math.sin(el)])
    S = UsdLux.DistantLight.Define(stage, "/World/Lights/exterior_sun")
    q = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*map(float, to_sun))).GetQuat()   # the light shines along its -Z
    UsdGeom.Xformable(S).AddOrientOp().Set(Gf.Quatf(q))
    S.CreateIntensityAttr(SUN)
    S.CreateAngleAttr(0.53)
    S.CreateColorAttr(Gf.Vec3f(*map(float, E / E.max() * np.array(SUN_TINT))))
    layer.Save()
    for k, v in sorted(table.items()):
        print(f"{k:34s} {len(v):3d}  {', '.join(sorted(set(v)))[:150]}")
    print(f"{len(_mats)} materials, sun {SUN:.0f} (HDRI sun x dome would be {DOME * E.mean():.0f}, {E.round(2)} at "
          f"{sun['elev_deg']:.1f} deg), dome {DOME:.0f} ->", ISAAC / "looks_exterior.usda")
