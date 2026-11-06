"""Room look layer for the tea twin: every fixed surface of the room interior (floors, walls, ceilings, built-ins,
openings, signs, fittings, furniture, decor, stairwell), matched to the person's video (vid.mp4).

  python work/isaac\\make_looks_room.py
rebuilds work/isaac/looks_room.usda from scratch (looklib.open_layer) and copies the textures it uses
to isaac/pbr/room_*/. Every material is /World/Looks/room_*.

Sources
- Nano Banana (texgen/nano_texture.py, refs = video frames): floor tiles (texgen/out/floor_tiles_*, 2 x 2 tiles of
  60 cm per image), worktop oak (texgen/out_room/room_worktop_*), the three black window signs
  (texgen/out_room/room_sign_{env,cups,self}.png, flat front views, mapped 0..1 onto each card face).
- Poly Haven CC0 sets (downloaded by looklib.polyhaven into isaac/pbr_src): their luminance detail, roughness
  detail and normal maps are kept, the colour is replaced by the colour measured in the video (derive()), so the
  albedo is the real surface's colour and not the stock photo's.
- plain UsdPreviewSurface for small fittings (sockets, switch plates, lens domes, cables, ...).
Interior fixture lights (/World/Lights/fixture_*, ceiling_bounce, defined in looks.usda) are only re-coloured /
re-scaled here (warmer, like the video)."""
import urllib.request

_op = urllib.request.build_opener()                      # api.polyhaven.com answers 403 to urllib's default agent
_op.addheaders = [("User-Agent", "tea-twin-looks/1.0")]
urllib.request.install_opener(_op)

from pathlib import Path                                  # noqa: E402

import numpy as np                                        # noqa: E402
from PIL import Image                                     # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade       # noqa: E402

import looklib as L                                       # noqa: E402

ISAAC = L.ISAAC
NANO = ISAAC / "texgen" / "out"
OUTR = ISAAC / "texgen" / "out_room"
OUTR.mkdir(parents=True, exist_ok=True)


def lin(srgb):
    c = np.asarray(srgb, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def _load01(path, size):
    im = Image.open(path)
    a = np.asarray(im)
    if a.dtype == np.uint16 or im.mode.startswith("I"):
        a = a.astype(np.float64) / 65535.0
    else:
        a = a.astype(np.float64) / 255.0
    if a.ndim == 3:
        a = a[..., :3] if a.shape[2] >= 3 else a[..., 0]
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    return np.asarray(img, np.float64) / 255.0


def derive(name, ph_id, target_lin, rough, rough_var=0.06, contrast=1.0, size=1024):
    """Poly Haven set `ph_id` re-coloured: albedo = target linear colour x the set's own luminance detail
    (contrast-scaled), roughness = `rough` +- rough_var x its own roughness detail, its OpenGL normal map."""
    src = L.polyhaven(ph_id, "2k")
    alb = lin(_load01(src["diff"], size) * 255.0)
    Y = alb @ np.array([0.2126, 0.7152, 0.0722])
    d = 1.0 + contrast * (Y / Y.mean() - 1.0)
    out = np.asarray(target_lin)[None, None, :] * d[..., None]
    files = {"diff": OUTR / f"{name}_diff.png", "rough": OUTR / f"{name}_rough.png", "nor": OUTR / f"{name}_nor_gl.png"}
    Image.fromarray((to_srgb(out) * 255 + 0.5).astype(np.uint8)).save(files["diff"])
    r = _load01(src["rough"], size)
    r = r if r.ndim == 2 else r[..., 0]
    r = np.clip(rough + rough_var * (r - r.mean()) / (r.std() + 1e-6), 0.03, 1.0)
    Image.fromarray((r * 255 + 0.5).astype(np.uint8)).save(files["rough"])
    n = _load01(src["nor"], size)
    Image.fromarray((n * 255 + 0.5).astype(np.uint8)).save(files["nor"])
    return {k: str(v) for k, v in files.items()}


def const_rough(name, value, size=64):
    f = OUTR / f"{name}_rough_const.png"
    Image.fromarray(np.full((size, size), int(value * 255 + 0.5), np.uint8)).save(f)
    return str(f)


def box_uv_off(prim, tile, off):
    """looklib.box_uv with a world offset (so the 80 cm tile grid lands on the twin's own grout lines)."""
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    W = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    pts = pts @ W[:3, :3] + W[3, :3] - np.asarray(off, np.float64)
    cnt = np.array(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    p0, p1, p2 = pts[idx[start]], pts[idx[start + 1]], pts[idx[start + 2]]
    dom = np.argmax(np.abs(np.cross(p1 - p0, p2 - p0)), axis=1)
    axes = np.array([[1, 2], [0, 2], [0, 1]])[np.repeat(dom, cnt)]
    c = pts[idx]
    uv = np.stack([c[np.arange(len(c)), axes[:, 0]], c[np.arange(len(c)), axes[:, 1]]], axis=1) / tile
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set([Gf.Vec2f(float(a), float(b)) for a, b in uv])


def tuned(src_set, name, contrast=1.0, rough_scale=1.0):
    """Copy of a Nano Banana set with the albedo's luminance detail scaled by `contrast` (colour kept) and the
    roughness scaled by `rough_scale`."""
    out = dict(src_set)
    if contrast != 1.0:
        a = lin(np.asarray(Image.open(src_set["diff"]).convert("RGB"), np.float64))
        Y = a @ np.array([0.2126, 0.7152, 0.0722])
        k = 1.0 + contrast * (Y / Y.mean() - 1.0)
        a = a * (k / np.maximum(Y / Y.mean(), 1e-4))[..., None]
        out["diff"] = str(OUTR / f"{name}_diff.png")
        Image.fromarray((to_srgb(a) * 255 + 0.5).astype(np.uint8)).save(out["diff"])
    if rough_scale != 1.0 and "rough" in src_set:
        r = np.asarray(Image.open(src_set["rough"]).convert("L"), np.float64) * rough_scale
        out["rough"] = str(OUTR / f"{name}_rough.png")
        Image.fromarray(np.clip(r + 0.5, 0, 255).astype(np.uint8)).save(out["rough"])
    return out


def plane_uv(prim, box):
    """Decal UVs for a sign face in the x-z plane (seen from -y, i.e. from the room): u = x, v = z over `box`."""
    x0, x1, z0, z1 = box
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    W = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    pts = pts @ W[:3, :3] + W[3, :3]
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    c = pts[idx]
    uv = np.stack([(c[:, 0] - x0) / (x1 - x0), (c[:, 2] - z0) / (z1 - z0)], axis=1)
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set([Gf.Vec2f(float(a), float(b)) for a, b in uv])


stage, layer = L.open_layer("looks_room.usda")
O = "/World/Objects/object_"

# ---------------------------------------------------------------- materials
M = {}
# floor: cream polished porcelain, 60 cm tiles, thin light grout, glossy (Nano Banana from f_093 + f_001)
# (tiles are 80 cm in the video and in the twin's grout geometry: image = 2 x 2 tiles -> 1.6 m; warm beige tint and
#  ~0.09 roughness measured against the video at t = 1 / 184 s)
M["floor"] = L.pbr("room_floor", tuned(L.nano_set(NANO, "floor_tiles"), "room_floor", rough_scale=0.6),
                   tint=(0.74, 0.46, 0.28))
# worktop: honey-orange crown-cut oak veneer, lacquered (Nano Banana from f_008 + f_013)
M["oak"] = L.pbr("room_worktop_oak", tuned(L.nano_set(OUTR, "room_worktop"), "room_worktop_tuned", contrast=1.5),
                 tint=(0.78, 0.86, 1.8), normal_scale=0.6, clearcoat=0.25)
# walls / ceiling: smooth painted plaster, off-white cream (colour from the video's walls, f_011 / f_013)
M["wall"] = L.pbr("room_wall_paint", derive("room_wall_paint", "painted_plaster_wall", lin((226, 212, 190)), 0.8, 0.05,
                                              contrast=0.6), normal_scale=0.5)
M["ceiling"] = L.pbr("room_ceiling_paint", derive("room_ceiling_paint", "painted_plaster_wall", lin((232, 230, 224)), 0.85,
                                                    0.04, contrast=0.5), normal_scale=0.4)
M["kitchen_tile"] = L.pbr("room_kitchen_tile", derive("room_kitchen_tile", "painted_plaster_wall", (0.49, 0.41, 0.28), 0.2,
                                                        0.04, contrast=0.3), normal_scale=0.2, clearcoat=0.5)
# black satin furniture board (cabinets, mug-rack style units, planter boxes, bench base)
M["black_board"] = L.pbr("room_black_board", derive("room_black_board", "fine_grained_wood", (0.022, 0.022, 0.024), 0.5,
                                                      0.08, contrast=0.5), normal_scale=0.4)
# black powder-coated aluminium / steel (window mullions, frames, railings, stair structure, cable trays)
M["black_metal"] = L.pbr("room_black_powdercoat", derive("room_black_powdercoat", "painted_plaster_wall",
                                                         (0.016, 0.016, 0.017), 0.38, 0.05, contrast=0.3),
                         normal_scale=0.25)
# dark grey bench cushions (upholstery fabric) and the black piping
M["cushion"] = L.pbr("room_cushion_fabric", derive("room_cushion_fabric", "rough_linen", lin((58, 58, 61)), 0.9, 0.05,
                                                     contrast=0.8))
M["piping"] = L.pbr("room_piping_fabric", derive("room_piping_fabric", "rough_linen", (0.012, 0.012, 0.013), 0.85, 0.05,
                                                   contrast=0.6))
# other floors (out of the video's view: kitchen planks, restroom tiles, stair treads / half landings)
M["kitchen_floor"] = L.pbr("room_kitchen_floor", derive("room_kitchen_floor", "laminate_floor_02", (0.15, 0.06, 0.03), 0.4,
                                                          0.08))
M["restroom_floor"] = L.pbr("room_restroom_floor", derive("room_restroom_floor", "grey_tiles", (0.26, 0.25, 0.22), 0.35,
                                                            0.08))
M["stair_wood"] = L.pbr("room_stair_wood", derive("room_stair_wood", "oak_wood_planks", (0.25, 0.13, 0.05), 0.45, 0.08))
# the three black acrylic window signs (Nano Banana flat front views from f_007 / f_008 / t=14 s)
for key, img in (("sign_env", "room_sign_env.png"), ("sign_cups", "room_sign_cups.png"), ("sign_self", "room_sign_self.png")):
    M[key] = L.pbr(f"room_{key}", {"diff": str(OUTR / img), "rough": const_rough(f"room_{key}", 0.35)}, clearcoat=0.1)
# plain fittings
P = L.plain
M["glass"] = P("room_clear_glass", (0.85, 0.9, 0.92), 0.02, opacity=0.08)
M["frosted"] = P("room_frosted_glass", (0.9, 0.92, 0.92), 0.35, opacity=0.55)
M["white_plastic"] = P("room_white_plastic", (0.79, 0.77, 0.71), 0.35, clearcoat=0.2)
M["housing"] = P("room_light_housing", (0.7, 0.7, 0.68), 0.35, metal=0.3)
M["black_plastic"] = P("room_black_plastic", (0.02, 0.02, 0.022), 0.3, clearcoat=0.4)
M["black_gloss"] = P("room_black_gloss", (0.012, 0.012, 0.013), 0.08, clearcoat=0.8)
M["steel"] = P("room_steel", (0.6, 0.6, 0.62), 0.4, metal=1.0)
M["led"] = P("room_led_diffuser", (0.95, 0.93, 0.9), 0.5, emissive=(6.0, 5.3, 4.4))
M["print_white"] = P("room_sign_print_white", (0.86, 0.86, 0.84), 0.4)
M["print_orange"] = P("room_sign_print_orange", (0.81, 0.48, 0.03), 0.4)
M["exit_green"] = P("room_exit_green", (0.03, 0.38, 0.07), 0.3, emissive=(0.05, 0.9, 0.15))
M["card"] = P("room_card_paper", (0.89, 0.89, 0.85), 0.7)
M["poster_mount"] = P("room_poster_mount", (0.69, 0.71, 0.75), 0.5, metal=0.5)
M["poster_red"] = P("room_poster_red", (0.55, 0.05, 0.07), 0.6)
M["red_gloss"] = P("room_extinguisher_red", (0.38, 0.01, 0.02), 0.2, clearcoat=0.7)
M["leaf"] = P("room_plant_leaf", (0.06, 0.2, 0.03), 0.45)
M["leaf_sun"] = P("room_plant_leaf_light", (0.1, 0.28, 0.04), 0.45)
M["stem"] = P("room_plant_stem", (0.07, 0.11, 0.02), 0.6)
M["soil"] = P("room_soil", (0.03, 0.02, 0.012), 0.95)
M["ceramic_white"] = P("room_white_ceramic", (0.87, 0.87, 0.85), 0.25, clearcoat=0.5)
M["balloon_pink"] = P("room_balloon_pink", (0.87, 0.38, 0.46), 0.2, clearcoat=0.8)
M["balloon_white"] = P("room_balloon_white", (0.91, 0.91, 0.91), 0.2, clearcoat=0.8)
M["ribbon"] = P("room_ribbon", (0.87, 0.87, 0.85), 0.4)
M["mural_yellow"] = P("room_mural_yellow", (0.89, 0.46, 0.03), 0.8)
M["mural_blue"] = P("room_mural_blue", (0.01, 0.27, 0.46), 0.8)
M["relief"] = P("room_relief_grey", (0.2, 0.25, 0.27), 0.7)
M["bottle"] = P("room_bottle_blue", (0.3, 0.55, 0.85), 0.08, opacity=0.35)
M["grout"] = P("room_kitchen_grout", (0.45, 0.37, 0.26), 0.8)

# ---------------------------------------------------------------- bindings: object id -> (material, tile metres | None)
FLOOR = 1.6           # floor_tiles image = 2 x 2 tiles -> 80 cm tiles
FLOOR_OFF = (0.4, 0.6, 0.0)   # the twin's grout lines: x = 0.4 + 0.8 k, y = 0.6 + 0.8 k (main floor)
B = {
    # floors (the twin's 80 cm grout strips get the floor material too: the 60 cm grout is in the texture)
    "0000": ("floor", FLOOR), "0001": ("floor", FLOOR), "0036": ("floor", FLOOR), "0005": ("floor", FLOOR),
    "0020": ("kitchen_floor", 1.5), "0021": ("kitchen_floor", 1.5),
    "0030": ("restroom_floor", 1.2), "0031": ("restroom_floor", 1.2),
    "0246": ("stair_wood", 1.0), "0249": ("stair_wood", 1.0), "0252": ("stair_wood", 1.0),
    # walls, ceilings, columns, beam
    "0024": ("kitchen_tile", 0.6), "0025": ("grout", None), "0015": ("wall", 2.0),
    "0003": ("ceiling", 2.0), "0006": ("ceiling", 2.0), "0023": ("ceiling", 2.0), "0037": ("ceiling", 2.0),
    "0261": ("ceiling", 2.0), "0262": ("ceiling", 2.0),
    "0004": ("black_metal", 0.5), "0258": ("black_metal", 0.5), "0002": ("black_metal", 0.5),
    # built-ins
    "0278": ("oak", 1.0), "0279": ("oak", 1.0), "0026": ("black_board", 1.0),
    "0027": ("black_board", 1.0), "0033": ("black_board", 1.0), "0280": ("black_board", 1.0), "0281": ("black_board", 1.0),
    "0011": ("black_board", 1.0), "0028": ("black_plastic", None), "0284": ("piping", 0.3),
    # openings
    "0013": ("black_metal", 0.5), "0272": ("black_metal", 0.5), "0275": ("black_metal", 0.5),
    "0019": ("black_metal", 0.5), "0034": ("black_metal", 0.5), "0035": ("black_board", 1.0),
    "0014": ("glass", None), "0273": ("glass", None), "0276": ("glass", None),
    "0274": ("frosted", None), "0277": ("frosted", None),
    # signs and cards
    "0224": ("white_plastic", None), "0225": ("exit_green", None), "0226": ("print_white", None),
    "0325": ("card", None), "0326": ("black_plastic", None),
    "0327": ("black_gloss", None), "0328": ("print_white", None), "0329": ("print_orange", None),
    "0330": ("black_gloss", None), "0331": ("print_white", None), "0332": ("print_orange", None),
    "0343": ("black_plastic", None), "0344": ("poster_mount", None), "0345": ("poster_red", None),
    "0346": ("black_plastic", None), "0347": ("black_gloss", None), "0348": ("print_white", None),
    # switches and fixtures
    "0333": ("white_plastic", None), "0334": ("black_plastic", None), "0335": ("white_plastic", None),
    "0336": ("black_plastic", None), "0296": ("white_plastic", None), "0297": ("black_plastic", None),
    "0298": ("black_plastic", None), "0323": ("red_gloss", None), "0324": ("black_plastic", None),
    "0349": ("white_plastic", None), "0350": ("white_plastic", None),
    "0385": ("white_plastic", None), "0386": ("black_gloss", None), "0387": ("white_plastic", None),
    "0388": ("black_gloss", None),
    "0041": ("led", None),
    # furniture, planters, plants, decor, paint
    "0039": ("oak", 1.0), "0040": ("black_plastic", None), "0282": ("black_board", 1.0), "0283": ("cushion", 0.3),
    "0269": ("ceramic_white", None), "0285": ("black_board", 1.0), "0290": ("black_board", 1.0),
    "0286": ("soil", None), "0291": ("soil", None),
    "0270": ("stem", None), "0287": ("stem", None), "0292": ("stem", None),
    "0271": ("leaf", None), "0288": ("leaf", None), "0289": ("leaf", None), "0293": ("leaf_sun", None),
    "0389": ("balloon_pink", None), "0392": ("balloon_pink", None), "0390": ("balloon_white", None),
    "0393": ("balloon_white", None), "0391": ("ribbon", None), "0394": ("ribbon", None),
    "0266": ("mural_yellow", None), "0267": ("mural_blue", None), "0268": ("relief", None),
    # stairwell + water dispenser
    "0351": ("black_plastic", None), "0352": ("bottle", None),
}
for p, (n, c, _) in L.NAMES.items():
    k = p[-4:]
    if c == "wall":
        B.setdefault(k, ("wall", 2.0))
    elif c in ("stair", "railing"):
        B.setdefault(k, ("black_metal", 0.5))
    elif c == "stair_step":
        B.setdefault(k, ("stair_wood", 1.0))
    elif c == "cable_tray":
        B.setdefault(k, ("steel", None) if "threaded_rod" in n else ("black_metal", 0.5))
    elif c == "light":
        B.setdefault(k, ("led", None) if "diffuser" in n else ("housing", None))

nb = 0
for k, (mk, tile) in sorted(B.items()):
    if mk == "floor":
        for m in L.meshes(O + k):
            box_uv_off(m, tile, FLOOR_OFF)
        tile = None
    nb += L.bind(stage, O + k, M[mk], tile=tile)
# window signs: decal UVs over each card's face (the print geometry gets the same mapping, so it merges)
for card, prt, mk in (("0337", "0338", "sign_env"), ("0339", "0340", "sign_cups"), ("0341", "0342", "sign_self")):
    xs, zs = [], []
    for m in L.meshes(O + card):
        pts = np.array(UsdGeom.Mesh(m).GetPointsAttr().Get(), dtype=np.float64)
        W = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(m))
        w = pts @ W[:3, :3] + W[3, :3]
        xs += [w[:, 0].min(), w[:, 0].max()]
        zs += [w[:, 2].min(), w[:, 2].max()]
    box = (min(xs), max(xs), min(zs), max(zs))
    for m in L.meshes(O + card):
        plane_uv(m, box)
        UsdShade.MaterialBindingAPI.Apply(m).Bind(M[mk], UsdShade.Tokens.strongerThanDescendants)
        nb += 1
    # the twin's modelled lettering ('CONSERVE!' + bars) sits 1 mm in front: the decal carries the real text now
    UsdGeom.Imageable(stage.GetPrimAtPath(O + prt)).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

# ---------------------------------------------------------------- interior lights (defined in looks.usda)
WARM = Gf.Vec3f(1.0, 0.82, 0.62)          # ~3000 K, the video's warm strips
nl = 0
for prim in stage.GetPrimAtPath("/World/Lights").GetChildren():
    if prim.GetName().startswith("fixture_"):
        Lr = UsdLux.RectLight(prim)
        Lr.GetColorAttr().Set(WARM)
        Lr.GetIntensityAttr().Set(25000.0)
        nl += 1
cb = UsdLux.RectLight(stage.GetPrimAtPath("/World/Lights/ceiling_bounce"))
cb.GetColorAttr().Set(Gf.Vec3f(1.0, 0.9, 0.78))
cb.GetIntensityAttr().Set(1500.0)
layer.Save()
print("room materials:", len(M), "meshes bound:", nb, "fixture lights:", nl, "->", ISAAC / "looks_room.usda")
