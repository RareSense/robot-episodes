"""Props looks layer: the tea things on and around the counter, matched to the person's video (vid.mp4).

  python make_looks_props.py            rebuild looks_props.usda (no API calls)
  python make_looks_props.py --gen a,b   (re)generate Nano Banana prints first

Writes work/isaac/looks_props.usda (strongest sublayer of tea_scene.usda), every material named
props_*. Texture sources:
  - Nano Banana (texgen/nano_texture.py helpers, OpenRouter google/gemini-nano-banana-2.1): flat front-view
    packaging prints made from crops of the 4K reference frames (texgen/ref/f_NNN.jpg = t 2*(N-1) s), saved to
    texgen/out/props/<name>_src.png; this script crops/resizes them to the meshes' UV layouts and derives
    normal + roughness maps (nano_texture.derived_maps).
  - Poly Haven CC0 sets (looklib.polyhaven), tinted to the video colours.
  - Procedural PBR sets (numpy: periodic micro-noise / brushed streaks) for glazes, plastics, steel, glass.
Tray layout of the twin (seen from the person, window at the back, jar on the left) and what each slot shows,
following the video's compartments as closely as the twin's 3x3 grid allows:
    back:   tub_left (grey holder, FRESH)  | tub_right (grey holder + spoons, USED) | snack_pack (pink Danedar)
    middle: green_tea_bags -> napkins      | coffee_sticks (Nescafe Classic)        | tea_bags_loose (paper + envelopes)
    front:  tea_envelopes -> Chashnik      | sugar_sticks (gold sticks)             | stirrers_and_sticks -> white/green
The kettle (kettle_fix.usda mesh) and the jar are single meshes: the kettle's lid disc is a GeomSubset (faces
never change); the jar uses a height-banded RGBA texture on a per-POINT st (survives powder_visual.open_jar,
which drops the lid faces but keeps the points)."""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import argparse
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ISAAC = Path(str(WORK / "isaac"))
sys.path.insert(0, str(ISAAC))
sys.path.insert(0, str(ISAAC / "texgen"))
import nano_texture as NT  # noqa: E402

REF = ISAAC / "texgen" / "ref"
OUT = ISAAC / "texgen" / "out" / "props"
OUT.mkdir(parents=True, exist_ok=True)

LABEL_STYLE = ("Output ONE image: the flat printed artwork of this product package, laid flat and photographed "
               "straight on (orthographic, no perspective), filling the whole frame edge to edge. Even, shadowless "
               "light: no reflections, no glare, no wrinkles, no hands, no background, no other objects. Reproduce "
               "the real colours, logo, pictures and text layout of the package in the reference photo(s) as "
               "exactly as possible; text must be crisp and correctly spelled. ")


def ref(n, box):
    return f"{REF / f'f_{n:03d}.jpg'}@{','.join(str(v) for v in box)}"


# name -> (prompt, refs, aspect ratio, tileable?)
GEN = {
    "nescafe": ("A NESCAFE CLASSIC instant coffee single-serve sachet (Pakistan market): glossy black film with a "
                "diagonal red band, the white NESCAFE logo and the word CLASSIC printed large and running "
                "horizontally along the long side, a red cup of black coffee seen from above. Landscape layout.",
                [ref(7, (1000, 2600, 1660, 3000)), ref(6, (1050, 2580, 1480, 2980))], "16:9", False),
    "danedar": ("A pink-red translucent crinkly plastic snack pack (biscuits / Danedar sachet strip) as seen in the "
                "photo: glossy magenta-pink film with the green and white word DANEDAR printed in bold italic "
                "capitals, small white text. Landscape layout.",
                [ref(7, (1680, 2620, 2160, 3110)), ref(6, (1450, 2500, 2160, 2900))], "4:3", False),
    "chashnik": ("A Chashnik white refined sugar stick sachet: white paper with bright green mint-leaf graphics "
                 "at both ends, the green script logo 'Chashnik' and 'WHITE REFINED SUGAR' in small green capitals, "
                 "a small web address. Long landscape strip layout, the text running along the length.",
                 [ref(6, (780, 3050, 1200, 3500))], "16:9", False),
    "sogarte": ("A golden-yellow sugar / sweetener stick sachet: mustard-yellow film with thin blue and green "
                "stripes across both ends and a blue logo in the middle, small dark text. Long thin landscape "
                "strip, the logo running along the length.",
                [ref(6, (1170, 3000, 1560, 3500))], "21:9", False),
    "greentea": ("A white green-tea sachet: white paper with fresh green tea-leaf graphics along one side and "
                 "green brand text (Urdu and English), as in the photo. Landscape layout.",
                 [ref(7, (1741, 3051, 2106, 3456)), ref(6, (1540, 2950, 2000, 3420))], "16:9", False),
    "tea_envelope": ("A small paper tea-bag envelope standing in a white paper tray: beige / cream paper with a red "
                     "brand logo and red text. Show the flat front of one envelope. Landscape layout.",
                     [ref(6, (1370, 2150, 1910, 2500))], "4:3", False),
    "napkin": ("Seamless tileable texture of a stack of plain cream-white paper napkins seen from straight above: "
               "soft tissue paper with a faint embossed pattern and fibres, warm off-white as in the photo.",
               [ref(6, (850, 2150, 1230, 2500))], "1:1", True),
}


def nano_generate(prompt, refs, aspect, tries=3):
    """Like nano_texture.generate(), with a free aspect ratio and no tileable-texture preamble for labels."""
    content = [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": NT.crop_b64(r)}}
                                                    for r in refs]
    body = {"model": NT.MODEL, "modalities": ["image", "text"], "messages": [{"role": "user", "content": content}],
            "image_config": {"aspect_ratio": aspect}}
    for i in range(tries):
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {NT.key()}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read())
            msg = out["choices"][0]["message"]
            for img in msg.get("images") or []:
                return Image.open(io.BytesIO(base64.b64decode(img["image_url"]["url"].split(",", 1)[1]))).convert("RGB")
            print("no image in the reply:", (msg.get("content") or "")[:300], file=sys.stderr)
        except Exception as e:                                              # network / rate limit: retry
            print(f"try {i + 1}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        time.sleep(5 * (i + 1))
    raise SystemExit("no image from the model")


def run_gen(names):
    for n in names:
        prompt, refs, aspect, tile = GEN[n]
        full = (NT.STYLE if tile else LABEL_STYLE) + prompt
        im = nano_generate(full, refs, aspect)
        im.save(OUT / f"{n}_src.png")
        (OUT / f"{n}_prompt.txt").write_text(full + "\nrefs: " + " ".join(refs) + f"\nmodel: {NT.MODEL}\n")
        print("generated", n, im.size, flush=True)


if __name__ == "__main__" and "--gen" in sys.argv:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="")
    A = ap.parse_args()
    run_gen([n for n in A.gen.split(",") if n] or list(GEN))
    sys.exit(0)

# ====================================================================================== build the layer
# Poly Haven's API refuses urllib's default User-Agent (403): send one for looklib.polyhaven()
_op = urllib.request.build_opener()
_op.addheaders = [("User-Agent", "tea-twin-looks/1.0")]
urllib.request.install_opener(_op)
import looklib as L  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdShade  # noqa: E402

PROC = OUT / "proc"
PROC.mkdir(parents=True, exist_ok=True)


def to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def periodic_noise(size, sigma_x, sigma_y, seed):
    """Unit-variance tileable noise (white noise low-passed in the Fourier domain; sigmas in cycles/pixel)."""
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((size, size)))
    fy, fx = np.fft.fftfreq(size)[:, None], np.fft.fftfreq(size)[None, :]
    h = np.real(np.fft.ifft2(F * np.exp(-0.5 * ((fx / sigma_x) ** 2 + (fy / sigma_y) ** 2))))
    return (h - h.mean()) / (h.std() + 1e-9)


def proc_set(name, rgb, rough, rough_var=0.04, alb_var=0.03, normal=0.5, brushed=False, size=512, seed=1):
    """A procedural PBR set: albedo (sRGB) with faint mottling, roughness (raw) and an OpenGL normal map."""
    h = periodic_noise(size, 0.004, 0.35, seed) if brushed else periodic_noise(size, 0.05, 0.05, seed)
    fine = periodic_noise(size, 0.3, 0.3, seed + 7)
    h = 0.8 * h + 0.2 * fine
    alb = to_srgb(np.array(rgb)[None, None, :] * (1 + alb_var * h[..., None]))
    Image.fromarray((alb * 255 + 0.5).astype(np.uint8)).save(PROC / f"{name}_diff.png")
    Image.fromarray((np.clip(rough + rough_var * h, 0.02, 1) * 255 + 0.5).astype(np.uint8)).save(PROC / f"{name}_rough.png")
    dy, dx = np.gradient(h)
    n = np.dstack([-dx * normal, dy * normal, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    Image.fromarray(((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)).save(PROC / f"{name}_nor_gl.png")
    return L.nano_set(PROC, name)


def label_set(name, src, box=None, size=None, rough=0.4, rough_var=0.1, normal=1.0, tint=None):
    """A non-tiling print (Nano Banana output or an existing decal): optional crop (fractions x0,y0,x1,y1) and
    resize to the mesh's UV aspect, plus derived normal / roughness maps."""
    im = Image.open(src).convert("RGB")
    if box:
        w, h = im.size
        im = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
    if size:
        im = im.resize(size, Image.LANCZOS)
    if tint is not None:
        a = np.asarray(im, np.float32) / 255
        im = Image.fromarray((to_srgb(((a + 0.055) / 1.055) ** 2.4 * np.array(tint)) * 255 + 0.5).astype(np.uint8))
    im.save(OUT / f"{name}_diff.png")
    nor, rgh = NT.derived_maps(im, rough, rough_var, normal)
    nor.save(OUT / f"{name}_nor_gl.png")
    rgh.save(OUT / f"{name}_rough.png")
    return L.nano_set(OUT, name)


def tint_to(maps, target):
    """tint that brings a Poly Haven albedo's mean (linear) to the video colour `target`."""
    a = np.asarray(Image.open(maps["diff"]).convert("RGB").resize((256, 256)), np.float32) / 255
    mean = (((a + 0.055) / 1.055) ** 2.4).reshape(-1, 3).mean(0)
    return tuple(float(t / max(m, 1e-4)) for t, m in zip(target, mean))


def banded(name, rows, height=512):
    """Material from a vertical RGBA strip read with st.y: rows = [(v0, v1, rgb_linear, alpha, rough, metal)].
    Opacity from the strip's alpha (real transparency with ior 1.5), roughness / metallic from raw strips."""
    rgba = np.zeros((height, 8, 4), np.float32)
    rm = np.zeros((height, 8, 3), np.float32)
    v = (height - 1 - np.arange(height)) / (height - 1)          # image row -> v (row 0 is v = 1)
    for v0, v1, rgb, a, r, m in rows:
        s = (v >= v0) & (v <= v1)
        rgba[s, :, :3] = to_srgb(np.array(rgb))
        rgba[s, :, 3] = a
        rm[s, :, 0], rm[s, :, 1] = r, m
    d = ISAAC / "pbr" / name
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), "RGBA").save(d / f"{name}_rgba.png")
    Image.fromarray((rm * 255 + 0.5).astype(np.uint8)).save(d / f"{name}_rm.png")
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    rd = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("st"))
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st = rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    ps = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("S"))
    ps.CreateIdAttr("UsdPreviewSurface")

    def tex(nm, f, cs):
        t = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild(nm))
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./pbr/{name}/{f}")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(cs)
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        return t
    c = tex("diff", f"{name}_rgba.png", "sRGB")
    ps.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(c.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    ps.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(c.CreateOutput("a", Sdf.ValueTypeNames.Float))
    r = tex("rm", f"{name}_rm.png", "raw")
    ps.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(r.CreateOutput("r", Sdf.ValueTypeNames.Float))
    ps.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(r.CreateOutput("g", Sdf.ValueTypeNames.Float))
    ps.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
    ps.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(1.0)
    ps.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.02)
    mat.CreateSurfaceOutput().ConnectToSource(ps.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat


def bind_one(path, mat, tile=None):
    """Bind on one mesh prim (or every renderable mesh under an object); tile -> world box UVs (metres)."""
    n = L.bind(stage, path, mat, tile=tile)
    assert n, f"nothing to bind under {path}"
    return n


def bind_weak(mesh_path, mat, tile):
    """Mesh-level binding that lets its GeomSubsets' bindings win (looklib binds strongerThanDescendants)."""
    prim = stage.GetPrimAtPath(mesh_path)
    L.box_uv(prim, tile)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants)


def face_subset(mesh_path, name, keep, mat):
    """Bind `mat` to the faces `keep` (bool per face) of a mesh through a materialBind GeomSubset."""
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
    sub = UsdGeom.Subset.CreateGeomSubset(mesh, name, UsdGeom.Tokens.face, np.flatnonzero(keep).astype(int).tolist(),
                                          familyName="materialBind")
    UsdGeom.Subset.SetFamilyType(mesh, "materialBind", UsdGeom.Tokens.nonOverlapping)
    UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mat)
    return int(keep.sum())


def face_data(mesh_path):
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
    P = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
    cnt = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    cen = np.add.reduceat(P[idx], start) / cnt[:, None]
    nrm = np.cross(P[idx[start + 1]] - P[idx[start]], P[idx[start + 2]] - P[idx[start]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    return P, cen, nrm


O = lambda i: f"/World/Objects/object_{i:04d}"          # noqa: E731
SV = lambda i, m: f"{O(i)}/set_visual/{m}"              # noqa: E731  set_fix.usda's detail meshes

stage, layer = L.open_layer("looks_props.usda")
M = {}
# ---------------------------------------------------------------- mugs: black glaze, tan unglazed rim and foot (video t 2-4 s)
M["glaze"] = L.pbr("props_mug_glaze", proc_set("props_mug_glaze", (0.010, 0.010, 0.011), 0.06, 0.02, 0.15, 0.15, seed=2),
                   clearcoat=1.0)
M["rim"] = L.pbr("props_mug_rim", proc_set("props_mug_rim", (0.50, 0.36, 0.21), 0.6, 0.08, 0.10, 0.8, seed=3))
M["foot"] = L.pbr("props_mug_foot", proc_set("props_mug_foot", (0.45, 0.37, 0.27), 0.85, 0.06, 0.10, 1.0, seed=4))
for i in range(311, 322):
    for part, look in (("body", "glaze"), ("handle", "glaze"), ("rim", "rim"), ("foot_ring", "foot")):
        bind_one(SV(i, part), M[look], tile=0.05)
# ---------------------------------------------------------------- mug rack: matte black wood, cream cloth liners
wood = L.polyhaven("fine_grained_wood", "2k")
M["rack"] = L.pbr("props_rack_black_wood", wood, tint=tint_to(wood, (0.016, 0.016, 0.018)), normal_scale=0.6, clearcoat=0.1)
linen = L.polyhaven("rough_linen", "2k")
M["cloth"] = L.pbr("props_rack_cloth", linen, tint=tint_to(linen, (0.50, 0.39, 0.25)))
bind_one(SV(310, "side_panels"), M["rack"], tile=0.4)
bind_one(SV(310, "boards"), M["rack"], tile=0.4)
bind_one(SV(310, "cloth_liners"), M["cloth"], tile=0.2)
# ---------------------------------------------------------------- tray and contents (video t 8-14 s, 24-30 s)
M["tray"] = L.pbr("props_tray_black", proc_set("props_tray_black", (0.014, 0.014, 0.016), 0.55, 0.06, 0.2, 0.6, seed=5),
                  clearcoat=0.05)
bind_one(SV(299, "box"), M["tray"], tile=0.1)
for lab in ("fresh", "used"):                     # the tray's FRESH / USED tabs: set_fix's decals, made PBR
    M[lab] = L.pbr(f"props_label_{lab}", label_set(f"props_label_{lab}", ISAAC / "set_tex" / f"label_{lab}.png",
                                                     rough=0.35, normal=0.5))
    bind_one(SV(299, f"label_{lab}"), M[lab])
M["tub"] = L.pbr("props_tub_plastic", proc_set("props_tub_plastic", (0.60, 0.61, 0.63), 0.3, 0.05, 0.03, 0.3, seed=6),
                 opacity=0.45, ior=1.49)
bind_one(SV(306, "tub"), M["tub"], tile=0.05)
bind_one(SV(307, "tub"), M["tub"], tile=0.05)
M["spoon"] = L.pbr("props_spoon_steel", proc_set("props_spoon_steel", (0.78, 0.78, 0.80), 0.16, 0.05, 0.03, 0.4,
                                                   brushed=True, seed=7), metal=1.0)
bind_one(SV(307, "cutlery"), M["spoon"], tile=0.05)
# printed packs (Nano Banana, from the 4K frames); stick UVs: u along the stick, v across it (1024 x 96 decals)
M["nescafe"] = L.pbr("props_nescafe", label_set("props_nescafe", OUT / "nescafe_src.png", (0.0, 0.0, 1.0, 1.0),
                                                (1024, 96), rough=0.25), clearcoat=0.3)
bind_one(SV(304, "sticks"), M["nescafe"])
M["sugar"] = L.pbr("props_gold_sugar", label_set("props_gold_sugar", OUT / "sogarte_src.png", (0.0, 0.22, 1.0, 0.78),
                                                 (1024, 96), rough=0.35))
bind_one(SV(301, "sticks"), M["sugar"])
M["chashnik"] = L.pbr("props_chashnik_stick", label_set("props_chashnik_stick", OUT / "chashnik_src.png",
                                                        (0.045, 0.31, 0.96, 0.68), (1024, 96), rough=0.5))
bind_one(SV(302, "sticks"), M["chashnik"])
UsdGeom.Imageable(stage.GetPrimAtPath(SV(302, "stirrers"))).CreateVisibilityAttr().Set("invisible")  # none in the video
M["greentea"] = L.pbr("props_greentea_sachet", label_set("props_greentea_sachet", OUT / "greentea_src.png",
                                                         (0.0, 0.0, 1.0, 1.0), (1024, 632), rough=0.5))
bind_one(SV(300, "envelopes"), M["greentea"])
M["napkin"] = L.pbr("props_napkin", label_set("props_napkin", OUT / "napkin_src.png", None, (1024, 1024), rough=0.9,
                                              rough_var=0.05, normal=3.0, tint=(1.08, 1.06, 1.06)))
bind_one(SV(303, "sachets"), M["napkin"])
M["danedar"] = L.pbr("props_danedar_pack", label_set("props_danedar_pack", OUT / "danedar_src.png", None, (1024, 768),
                                                     rough=0.18, rough_var=0.08), clearcoat=0.6)
bind_one(SV(308, "packs"), M["danedar"])
M["crimp"] = L.pbr("props_danedar_crimp", proc_set("props_danedar_crimp", (0.62, 0.05, 0.22), 0.25, 0.08, 0.1, 1.5,
                                                    brushed=True, seed=8), clearcoat=0.5)
bind_one(SV(308, "crimps"), M["crimp"], tile=0.02)
M["paper"] = L.pbr("props_paper_white", label_set("props_paper_white", OUT / "napkin_src.png", None, (1024, 1024),
                                                  rough=0.85, rough_var=0.05, normal=2.0, tint=(1.15, 1.13, 1.16)))
bind_one(SV(305, "bags"), M["paper"], tile=0.08)
bind_one(SV(305, "strings"), M["paper"], tile=0.08)
M["envelope"] = L.pbr("props_tea_envelope", label_set("props_tea_envelope", OUT / "tea_envelope_src.png", None,
                                                      (512, 384), rough=0.6))
bind_one(SV(305, "tags"), M["envelope"])
# ---------------------------------------------------------------- kettle (kettle_fix.usda's closed mesh) and its base
M["kettle"] = L.pbr("props_kettle_black", proc_set("props_kettle_black", (0.018, 0.018, 0.02), 0.42, 0.03, 0.03, 0.08,
                                                     seed=9), clearcoat=0.25)
M["kettle_lid"] = L.pbr("props_kettle_lid_steel", proc_set("props_kettle_lid_steel", (0.80, 0.80, 0.81), 0.40, 0.06,
                                                             0.05, 0.3, brushed=True, seed=10), metal=1.0)
kv = O(295) + "/visual"
bind_weak(kv, M["kettle"], tile=0.08)
_, cen, nrm = face_data(kv)                       # the lid disc: the up-facing top within 5 cm of the axis
n_lid = face_subset(kv, "props_lid_disc", (cen[:, 2] > 0.205) & (nrm[:, 2] > 0.5) & (np.hypot(cen[:, 0], cen[:, 1]) < 0.05),
                    M["kettle_lid"])
bind_one(O(294), M["kettle"], tile=0.08)
# ---------------------------------------------------------------- jar: clear glass, cream milk powder, brushed lid
jv = O(309) + "/visual"
JAR_H = 0.132
M["jar"] = banded("props_jar", [
    (0.000, 0.070, (0.86, 0.88, 0.86), 0.25, 0.03, 0.0),     # thick glass base
    (0.070, 0.660, (0.74, 0.60, 0.42), 1.00, 0.45, 0.0),     # milk powder seen through the wall
    (0.660, 0.825, (0.90, 0.92, 0.90), 0.10, 0.02, 0.0),     # clear glass shoulder above the powder
    (0.825, 1.000, (0.82, 0.82, 0.83), 1.00, 0.50, 0.85)])    # brushed aluminium lid (z > 0.109 m)
jm = UsdGeom.Mesh(stage.GetPrimAtPath(jv))
jp = np.asarray(jm.GetPointsAttr().Get(), np.float64)
UsdGeom.PrimvarsAPI(jm).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
    [Gf.Vec2f(0.5, float(np.clip(z / JAR_H, 0.002, 0.998))) for z in jp[:, 2]])
UsdShade.MaterialBindingAPI.Apply(jm.GetPrim()).Bind(M["jar"], UsdShade.Tokens.strongerThanDescendants)
# ---------------------------------------------------------------- misc: spare 19 l water bottles, flat mop
M["bottle"] = L.pbr("props_bottle_blue_pc", proc_set("props_bottle_blue_pc", (0.30, 0.55, 0.85), 0.05, 0.02, 0.02, 0.2,
                                                       seed=11), opacity=0.35, ior=1.58, clearcoat=1.0)
for i in (353, 354):
    bind_one(O(i), M["bottle"], tile=0.2)
vel = L.polyhaven("velour_velvet", "2k")
M["mop_pad"] = L.pbr("props_mop_pad", vel, tint=tint_to(vel, (0.02, 0.05, 0.13)))
M["mop_pole"] = L.pbr("props_mop_pole", proc_set("props_mop_pole", (0.62, 0.63, 0.65), 0.3, 0.05, 0.02, 0.4,
                                                   brushed=True, seed=12), metal=1.0)
mv = O(29) + "/visual"
bind_weak(mv, M["mop_pole"], tile=0.2)
_, cen, _ = face_data(mv)
zmin = cen[:, 2].min()
n_pad = face_subset(mv, "props_pad", cen[:, 2] < zmin + 0.06, M["mop_pad"])
layer.Save()
print(f"materials: {len(M)}; kettle lid faces {n_lid}; mop pad faces {n_pad} ->", ISAAC / "looks_props.usda")