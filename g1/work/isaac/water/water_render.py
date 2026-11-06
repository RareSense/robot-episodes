"""Water agent's copy of isaac\\tea_render.py: the same render plus the simulated water (water_layer.py).
Render a baked tea shot in Isaac Sim 6.2 (RTX): the twin with its Poly Haven looks (tea_scene.usda),
the G1's meshes and the handled objects posed on every frame from the physics recording (bake.py),
seen through the robot's own eye camera or the film camera; the PhysX fluid is stepped to every frame.
  <isaac-sim>/python work/isaac\\water\\water_render.py <shot_dir> <out.mp4> --cam eye|film
      [--res WxH] [--mode pt|rt] [--spp 32] [--frames a:b] [--still 0,300,600]
      [--water-cache dir] [--fill 400] [--tea-from FRAME] [--no-water] [--hide-robot] [--view ex,ey,ez:tx,ty,tz:hfov]
Changes against tea_render.py are marked  # WATER."""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import argparse
import json
import math
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("shot")
ap.add_argument("out")
ap.add_argument("--cam", default="eye")
ap.add_argument("--res", default="")
ap.add_argument("--mode", default="pt")
ap.add_argument("--spp", type=int, default=32)
ap.add_argument("--frames", default="")
ap.add_argument("--still", default="")
ap.add_argument("--exposure", type=float, default=0.0)
ap.add_argument("--hide", default="", help="checking: comma-separated object-name pieces to hide")
ap.add_argument("--cands", default="", help="checking: film cameras 'ex,ey,ez:tx,ty,tz:hfov;...' as stills")
# WATER: options
ap.add_argument("--no-water", action="store_true")
ap.add_argument("--water-cache", default="", help="fluid surfaces per frame: written on the first render, replayed after")
ap.add_argument("--fill", type=float, default=400.0, help="ml of water in the kettle at frame 0")
ap.add_argument("--substeps", type=int, default=4)
ap.add_argument("--tea-from", type=int, default=-1, help="frame from which the water in the mug steeps to tea")
ap.add_argument("--hide-robot", action="store_true")
ap.add_argument("--view", default="", help="film camera override 'ex,ey,ez:tx,ty,tz:hfov'")
ap.add_argument("--water-look", default="glass", help="glass (OmniGlass MDL) | preview (UsdPreviewSurface) | debug")
ap.add_argument("--water-looks", default="", help="stills: render each still with each of these looks")
ap.add_argument("--water-kw", default="{}", help="extra WaterLayer keyword arguments as JSON (tuning)")
ap.add_argument("--jar-open", action="store_true", help="milk powder: the twin jar without its built-in lid, powder inside")
ap.add_argument("--grains", default="", help="milk powder: <run>_grains.npz from the grain pass")
A = ap.parse_args()
W, H = map(int, (A.res or ("720x1280" if A.cam == "eye" else "1920x1080")).split("x"))
SCENE = str(WORK / "isaac/tea_scene.usda")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": W, "height": H,
                     "renderer": "PathTracing" if A.mode == "pt" else "RaytracedLighting"})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402

S = dict(np.load(os.path.join(A.shot, "shot.npz"), allow_pickle=True))
V = np.load(os.path.join(A.shot, "robot_visual.npz"), allow_pickle=True)
st = carb.settings.get_settings()
SETTINGS = {"/rtx/rendermode": "PathTracing" if A.mode == "pt" else "RaytracedLighting",
            "/rtx/pathtracing/spp": A.spp, "/rtx/pathtracing/totalSpp": A.spp,
            "/rtx/pathtracing/maxBounces": 6, "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 8,
            "/rtx/pathtracing/optixDenoiser/enabled": True, "/rtx/pathtracing/clampSpp": 0,
            "/rtx/post/aa/op": 3, "/rtx/post/tonemap/op": 6, "/rtx/post/tonemap/filmIso": 100.0 * 2 ** A.exposure,
            "/rtx/post/histogram/enabled": False, "/rtx/directLighting/sampledLighting/enabled": True,
            # WATER: refraction through the water surface in the real-time mode
            "/rtx/translucency/enabled": True, "/rtx/translucency/maxRefractionBounces": 8,
            "/rtx/translucency/reflectAtAllBounce": True,
            "/omni/replicator/RTSubframes": 1,
            "/app/viewport/grid/enabled": False, "/persistent/app/viewport/grid/enabled": False,
            "/app/viewport/show/grid": False, "/persistent/app/viewport/displayOptions": 0,
            "/app/viewport/displayOptions": 0, "/app/viewport/outline/enabled": False}


def apply_settings():
    for k, v in SETTINGS.items():
        st.set(k, v)


apply_settings()
ctx = omni.usd.get_context()
ctx.open_stage(SCENE)
for _ in range(30):
    app.update()
stage = ctx.get_stage()
# a playback of states the MuJoCo physics already computed: Isaac's own physics stays off
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Scene):
        prim.SetActive(False)
    elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(False)
if A.hide:
    for path, (name, cls, col) in json.load(open(str(WORK / "isaac/prim_names.json"))).items():
        if any(h in name or h == cls for h in A.hide.split(",")):
            UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()


# ------------------------------------------------------------------ the G1 (meshes from the MuJoCo model)
def robot_material(rgba):
    """The G1's finish: light parts satin aluminium, dark parts black anodised / plastic."""
    lum = 0.3 * rgba[0] + 0.59 * rgba[1] + 0.11 * rgba[2]
    key = f"/World/Looks/g1_{'light' if lum > 0.45 else 'dark'}"
    if stage.GetPrimAtPath(key):
        return UsdShade.Material(stage.GetPrimAtPath(key))
    m = UsdShade.Material.Define(stage, key)
    s = UsdShade.Shader.Define(stage, key + "/Shader")
    s.SetSourceAsset("OmniPBR.mdl", "mdl")
    s.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    col, rough, metal = ((0.62, 0.63, 0.65), 0.34, 0.75) if lum > 0.45 else ((0.035, 0.035, 0.04), 0.4, 0.25)
    s.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col))
    s.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(rough)
    s.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metal)
    s.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.5)
    for o in ("surface", "displacement", "volume"):
        getattr(m, f"Create{o.capitalize()}Output")("mdl").ConnectToSource(s.ConnectableAPI(), "out")
    return m


names = [] if A.hide_robot else [str(n) for n in S["robot_names"]]       # WATER: --hide-robot
ops = {}
for n in names:
    x = UsdGeom.Xform.Define(stage, f"/World/Robot/{n}")
    ops[n] = (x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble))
mesh_names = V["mesh_names"] if "mesh_names" in V.files else [""] * len(V["bodies"])
for gi, (b, mk) in enumerate(zip(V["bodies"], V["mesh"])):
    if A.hide_robot:
        break
    if A.cam == "eye" and "head" in str(mesh_names[gi]):
        continue                                # the eye camera is in the head: its own head is not in view
    b = str(b)
    v, f = V[f"v{int(mk)}"].astype(np.float32), V[f"f{int(mk)}"].astype(np.int32)
    me = UsdGeom.Mesh.Define(stage, f"/World/Robot/{b}/g{gi}")
    me.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    me.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
    me.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
    me.CreateSubdivisionSchemeAttr("none")
    p = V["pose"][gi]
    xf = UsdGeom.Xformable(me)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, p[:3])))
    xf.AddOrientOp().Set(Gf.Quatf(float(p[3]), float(p[4]), float(p[5]), float(p[6])))
    UsdShade.MaterialBindingAPI.Apply(me.GetPrim()).Bind(robot_material(V["rgba"][gi]))


# ------------------------------------------------------------------ props added to the scene (tea bag, spoon, ...)
# (as tea_render.py: this copy had lost them, so the spoon was missing from every water render)
def prop_material(name, rgba):
    metal = "spoon" in name
    key = f"/World/Looks/prop_{'steel' if metal else ''}_{int(rgba[0]*255)}_{int(rgba[1]*255)}_{int(rgba[2]*255)}"
    if stage.GetPrimAtPath(key):
        return UsdShade.Material(stage.GetPrimAtPath(key))
    mt = UsdShade.Material.Define(stage, key)
    sh = UsdShade.Shader.Define(stage, key + "/Shader")
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in rgba[:3]]))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.22 if metal else 0.6)
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(1.0 if metal else 0.0)
    for o in ("surface", "displacement", "volume"):
        getattr(mt, f"Create{o.capitalize()}Output")("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mt


def usd_name(n):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in str(n).replace("env/", ""))


prop_ops = {}
if os.path.exists(os.path.join(A.shot, "props_visual.npz")) and "prop_names" in S and len(S["prop_names"]):
    PV = np.load(os.path.join(A.shot, "props_visual.npz"), allow_pickle=True)
    for n in S["prop_names"]:
        x = UsdGeom.Xform.Define(stage, f"/World/Props/{usd_name(n)}")
        prop_ops[usd_name(n)] = (x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble))
    for gi, (bn, mk) in enumerate(zip(PV["bodies"], PV["mesh"])):
        n = usd_name(bn)
        if n not in prop_ops:
            continue
        v, f = PV[f"v{int(mk)}"].astype(np.float32), PV[f"f{int(mk)}"].astype(np.int32)
        me = UsdGeom.Mesh.Define(stage, f"/World/Props/{n}/g{gi}")
        me.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
        me.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
        me.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
        me.CreateSubdivisionSchemeAttr("none")
        pp = PV["pose"][gi]
        xf = UsdGeom.Xformable(me)
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pp[:3])))
        xf.AddOrientOp().Set(Gf.Quatf(float(pp[3]), float(pp[4]), float(pp[5]), float(pp[6])))
        UsdShade.MaterialBindingAPI.Apply(me.GetPrim()).Bind(prop_material(n, PV["rgba"][gi]))
prop_index = {usd_name(n): j for j, n in enumerate(S["prop_names"])} if prop_ops else {}

# ------------------------------------------------------------------ milk powder: the open jar and the grains
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import powder_visual  # noqa: E402
if A.jar_open:
    powder_visual.open_jar(stage)
grains = powder_visual.Grains(stage, A.grains, S["t"]) if A.grains else None

# ------------------------------------------------------------------ objects posed from physics
obj_paths = [str(p) for p in S["obj_paths"] if stage.GetPrimAtPath(str(p))]
obj_rows = [i for i, p in enumerate(S["obj_paths"]) if stage.GetPrimAtPath(str(p))]
for path in obj_paths:
    UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform()

# ------------------------------------------------------------------ the kettle's on/off lever and its blue light
# (the lever is a part of its own in the physics: it rides on its slide joint; the light follows its state)
lever_op = led_op = glow_op = glow_i = led_emit = None
if "lever" in S and stage.GetPrimAtPath("/World/Objects/object_0295"):
    (cx, cy, cz), (hx, hy, hz) = S["lever"]
    lv = UsdGeom.Cube.Define(stage, "/World/Objects/object_0295/switch_lever")
    lever_op = UsdGeom.Xformable(lv).AddTranslateOp()
    UsdGeom.Xformable(lv).AddScaleOp().Set(Gf.Vec3f(float(hx), float(hy), float(hz)))
    look = stage.GetPrimAtPath("/World/Looks/black_plastic")
    if look:
        UsdShade.MaterialBindingAPI.Apply(lv.GetPrim()).Bind(UsdShade.Material(look))
    led = UsdGeom.Sphere.Define(stage, "/World/Objects/object_0295/switch_led")
    led.CreateRadiusAttr(0.0035)
    led_op = UsdGeom.Xformable(led).AddTranslateOp()
    mat = UsdShade.Material.Define(stage, "/World/Looks/switch_led")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/switch_led/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.08, 0.2))
    led_emit = sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(led.GetPrim()).Bind(mat)
    led_glow = UsdLux.SphereLight.Define(stage, "/World/Objects/object_0295/switch_glow")
    led_glow.CreateRadiusAttr(0.004)
    led_glow.CreateColorAttr(Gf.Vec3f(0.25, 0.5, 1.0))
    glow_i = led_glow.CreateIntensityAttr(0.0)
    glow_op = UsdGeom.Xformable(led_glow).AddTranslateOp()
    led_glow.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)


def set_switch(k):
    if lever_op is None:
        return
    (cx, cy, cz), (hx, hy, hz) = S["lever"]
    z = float(S["switch_z"][k])
    lever_op.Set(Gf.Vec3d(float(cx), float(cy), float(cz) + z))
    tip = Gf.Vec3d(float(cx - hx + 0.012), float(cy), float(cz + hz) + z + 0.001)
    led_op.Set(tip)
    glow_op.Set(tip + Gf.Vec3d(0, 0, 0.004))
    on = bool(S["switch_on"][k])
    led_emit.Set(Gf.Vec3f(1.5, 3.0, 12.0) if on else Gf.Vec3f(0.0, 0.0, 0.0))
    glow_i.Set(4.0e4 if on else 0.0)


# ------------------------------------------------------------------ WATER: the simulated fluid
water = None
if not A.no_water:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from water_layer import WaterLayer  # noqa: E402
    water = WaterLayer(S, stage, fill_ml=A.fill, substeps=A.substeps, cache_dir=A.water_cache or None,
                       tea_from_frame=(A.tea_from if A.tea_from >= 0 else None), look=A.water_look, **json.loads(A.water_kw))
    # (WaterLayer blocks the timeline events itself: the replicator step's play/pause must not step the fluid)

# ------------------------------------------------------------------ camera
cam = UsdGeom.Camera.Define(stage, "/World/RenderCam")
HA = 20.955
cam.CreateHorizontalApertureAttr(HA)
cam.CreateVerticalApertureAttr(HA * H / W)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 500))
VIEW = None
if A.view:                                                                 # WATER: --view
    e_, t_, hf_ = A.view.split(":")
    VIEW = ([float(x) for x in e_.split(",")], [float(x) for x in t_.split(",")], float(hf_))
if A.cam == "eye":
    cam.CreateFocalLengthAttr(float((HA * H / W) / (2 * math.tan(math.radians(float(S["eye_fovy"])) / 2))))
else:
    hf = VIEW[2] if VIEW else float(S["film_hfov"])
    cam.CreateFocalLengthAttr(float(HA / (2 * math.tan(math.radians(hf) / 2))))
cam_op = UsdGeom.Xformable(cam).AddTransformOp()


def set_frame(k):
    if A.cam == "eye":
        cam_op.Set(Gf.Matrix4d(S["eye_mats"][k].tolist()))
    else:
        e, t = (VIEW[0], VIEW[1]) if VIEW else (S["film_eye"][k], S["film_tgt"][k])
        e, t = Gf.Vec3d(*map(float, e)), Gf.Vec3d(*map(float, t))
        cam_op.Set(Gf.Matrix4d().SetLookAt(e, t, Gf.Vec3d(0, 0, 1)).GetInverse())
    set_switch(k)
    if grains:
        grains.update(k)
    for n, j in prop_index.items():
        prop_ops[n][0].Set(Gf.Vec3d(*map(float, S["prop_xpos"][k, j])))
        q = S["prop_xquat"][k, j]
        prop_ops[n][1].Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    for path, i in zip(obj_paths, obj_rows):          # looked up by path: handles go stale if the stage recomposes
        stage.GetPrimAtPath(path).GetAttribute("xformOp:transform").Set(Gf.Matrix4d(S["obj_mats"][k, i].tolist()))
    for j, n in enumerate(names):
        ops[n][0].Set(Gf.Vec3d(*map(float, S["xpos"][k, j])))
        q = S["xquat"][k, j]
        ops[n][1].Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    if water is not None:                                                  # WATER: fluid to frame k
        water.update(k)


rp = rep.create.render_product("/World/RenderCam", (W, H))
for _ in range(5):
    app.update()
apply_settings()
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])
F = len(S["xpos"])
a, b = 0, F - 1
if A.frames:
    a, b = [int(x) for x in A.frames.split(":")]
set_frame(a)
for _ in range(20):
    app.update()
if water is not None and not water.cached:                               # WATER: check nothing else steps the fluid
    p0 = water.positions()
    rep.orchestrator.step(rt_subframes=1, pause_timeline=True)
    moved = float(np.abs(water.positions() - p0).max())
    print(f"[water] fluid moved {moved * 1e3:.3f} mm during a render step (must be 0)", flush=True)
if A.cands:
    from PIL import Image
    k = int(A.still or 0)
    set_frame(k)
    for i, c in enumerate(A.cands.split(";")):
        e, t, hf = c.split(":")
        cam_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, e.split(","))), Gf.Vec3d(*map(float, t.split(","))),
                                           Gf.Vec3d(0, 0, 1)).GetInverse())
        cam.GetFocalLengthAttr().Set(float(HA / (2 * math.tan(math.radians(float(hf)) / 2))))
        rep.orchestrator.step(rt_subframes=1, pause_timeline=True)
        Image.fromarray(np.asarray(rgb.get_data())[:, :, :3]).save(A.out.replace(".mp4", f"_cand{i}.png"))
        print("cand", i, c, flush=True)
elif A.still:
    from PIL import Image
    for k in [int(x) for x in A.still.split(",")]:
        set_frame(k)
        for look in (A.water_looks.split(",") if A.water_looks and water is not None else [""]):   # WATER: looks
            if look:
                water.set_look(look)
            # WATER: the RT capture comes back a few frames late after a jump (the stills came out showing
            # an earlier frame): render a few times before saving
            for _ in range(4 if (A.mode != "pt" or look) else 1):
                rep.orchestrator.step(rt_subframes=1, pause_timeline=True)
            Image.fromarray(np.asarray(rgb.get_data())[:, :, :3]).save(
                A.out.replace(".mp4", f"_{A.cam}_{k:04d}{'_' + look if look else ''}.png"))
        if water is not None and not water.cached:
            np.save(A.out.replace(".mp4", f"_{k:04d}_particles.npy"), water.positions().astype(np.float32))
        print("still", k, (water.stats[min(k, len(water.stats) - 1)] if water is not None and water.stats else ""),
              flush=True)
else:
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(int(S["fps"])), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                           "-pix_fmt", "yuv420p", A.out], stdin=subprocess.PIPE)
    t_start = time.time()
    for k in range(a, b + 1):
        set_frame(k)
        rep.orchestrator.step(rt_subframes=1, pause_timeline=True)
        ff.stdin.write(np.ascontiguousarray(np.asarray(rgb.get_data())[:, :, :3]).tobytes())
        if k % 60 == 0:
            ws = ""
            if water is not None and water.stats:
                s = water.stats[min(k, len(water.stats) - 1)]
                ws = f"  water: mug {s['mug_ml']:.0f} ml, kettle {s['kettle_ml']:.0f} ml, other {s['other_ml']:.0f} ml"
            print(f"frame {k}/{b}  {(time.time() - t_start) / max(1, k - a + 1):.2f} s/frame{ws}", flush=True)
    ff.stdin.close()
    ff.wait()
if water is not None:                                                      # WATER: stats + cache
    water.finish(complete=(a == 0 and b == F - 1 and not A.still))
    json.dump(water.stats, open(os.path.splitext(A.out)[0] + "_water_stats.json", "w"))
    tm = water.timing
    if tm["frames"]:
        print(f"[water] fluid {tm['sim'] / tm['frames'] * 1000:.0f} ms/frame, surface copy "
              f"{tm['copy'] / tm['frames'] * 1000:.0f} ms/frame", flush=True)
print("done", A.out, flush=True)
app.close()
