"""Render a baked tea shot in Isaac Sim 6.2 (RTX): the twin with its Poly Haven looks (tea_scene.usda),
the G1's meshes and the handled objects posed on every frame from the physics recording (bake.py),
seen through the robot's own eye camera or the film camera.
  D:\\isaacsim62\\python.bat D:\\tea_robot\\work\\isaac\\tea_render.py <shot_dir> <out.mp4> --cam eye|film
      [--res WxH] [--mode pt|rt] [--spp 32] [--frames a:b] [--still 0,300,600]"""
import argparse
import math
import os
import subprocess
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
ap.add_argument("--jar-open", action="store_true", help="milk powder: the twin jar without its built-in lid, powder inside")
ap.add_argument("--grains", default="", help="milk powder: <run>_grains.npz from the grain pass")
ap.add_argument("--serve", default="")
ap.add_argument("--scene", default="D:/tea_robot/work/isaac/tea_scene_before_textures.usda")
A = ap.parse_args()
W, H = map(int, (A.res or ("720x1280" if A.cam == "eye" else "1920x1080")).split("x"))
SCENE = A.scene
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
    import json
    for path, (name, cls, col) in json.load(open(r"D:\tea_robot\work\isaac\prim_names.json")).items():
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


names = [str(n) for n in S["robot_names"]]
ops = {}
for n in names:
    x = UsdGeom.Xform.Define(stage, f"/World/Robot/{n}")
    ops[n] = (x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble))
mesh_names = V["mesh_names"] if "mesh_names" in V.files else [""] * len(V["bodies"])
for gi, (b, mk) in enumerate(zip(V["bodies"], V["mesh"])):
    b = str(b)
    if A.cam == "eye" and "head" in str(mesh_names[gi]):
        continue                                # the eye camera is in the head: its own head is not in view
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
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\tea_robot\work\isaac")
sys.dont_write_bytecode = True
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


# ------------------------------------------------------------------ camera
cam = UsdGeom.Camera.Define(stage, "/World/RenderCam")
HA = 20.955
cam.CreateHorizontalApertureAttr(HA)
cam.CreateVerticalApertureAttr(HA * H / W)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 500))
if A.cam == "eye":
    cam.CreateFocalLengthAttr(float((HA * H / W) / (2 * math.tan(math.radians(float(S["eye_fovy"])) / 2))))
else:
    cam.CreateFocalLengthAttr(float(HA / (2 * math.tan(math.radians(float(S["film_hfov"])) / 2))))
cam_op = UsdGeom.Xformable(cam).AddTransformOp()


def set_frame(k):
    if A.cam == "eye":
        cam_op.Set(Gf.Matrix4d(S["eye_mats"][k].tolist()))
    else:
        e, t = Gf.Vec3d(*map(float, S["film_eye"][k])), Gf.Vec3d(*map(float, S["film_tgt"][k]))
        cam_op.Set(Gf.Matrix4d().SetLookAt(e, t, Gf.Vec3d(0, 0, 1)).GetInverse())
    set_switch(k)
    if grains:
        grains.update(k)
    for n, j in prop_index.items():
        if S["prop_xpos"].shape[1] <= j:
            prop_ops[n][0].Set(Gf.Vec3d(0, 0, -10))
            continue
        prop_ops[n][0].Set(Gf.Vec3d(*map(float, S["prop_xpos"][k, j])))
        q = S["prop_xquat"][k, j]
        prop_ops[n][1].Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    for path, i in zip(obj_paths, obj_rows):          # looked up by path: handles go stale if the stage recomposes
        stage.GetPrimAtPath(path).GetAttribute("xformOp:transform").Set(Gf.Matrix4d(S["obj_mats"][k, i].tolist()))
    for j, n in enumerate(names):
        ops[n][0].Set(Gf.Vec3d(*map(float, S["xpos"][k, j])))
        q = S["xquat"][k, j]
        ops[n][1].Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))


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
def run_job(J):
    Sn = dict(np.load(os.path.join(J["shot"], "shot.npz"), allow_pickle=True))
    for k_ in ("xpos", "xquat", "obj_mats", "eye_mats", "film_eye", "film_tgt", "prop_xpos", "prop_xquat", "switch_z", "switch_on"):
        S[k_] = Sn[k_]
    S["film_hfov"] = Sn["film_hfov"]
    W2, H2 = (720, 1280) if J["cam"] == "eye" else (1920, 1080)
    A.cam = J["cam"]
    heads = [f"/World/Robot/{b_}/g{gi_}" for gi_, b_ in enumerate(V["bodies"]) if "head" in str(mesh_names[gi_])]
    if J["cam"] == "eye":
        for h_ in heads:
            UsdGeom.Imageable(stage.GetPrimAtPath(h_)).MakeInvisible()
    cam.GetVerticalApertureAttr().Set(HA * H2 / W2)
    if J["cam"] == "eye":
        cam.GetFocalLengthAttr().Set(float((HA * H2 / W2) / (2 * math.tan(math.radians(float(Sn["eye_fovy"])) / 2))))
    else:
        cam.GetFocalLengthAttr().Set(float(HA / (2 * math.tan(math.radians(float(Sn["film_hfov"])) / 2))))
    rp2 = rep.create.render_product("/World/RenderCam", (W2, H2))
    an2 = rep.AnnotatorRegistry.get_annotator("rgb")
    an2.attach([rp2])
    a_, b_ = J.get("a", 0), J.get("b", len(S["xpos"]) - 1)
    b_ = min(b_, len(S["xpos"]) - 1)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W2}x{H2}", "-r", "60", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", J["out"]], stdin=subprocess.PIPE)
    t_s = time.time()
    wat = None
    if J.get("water"):
        WI = np.load(J["water"] + "/water_index.npz")
        wn = int(WI["n"])
        WP = np.memmap(J["water"] + "/water.f32", np.float32, "r").reshape(-1, wn, 3)
        wt_ = WI["t"]
        if not stage.GetPrimAtPath("/World/RecWater"):
            wat = UsdGeom.Points.Define(stage, "/World/RecWater")
            wat.CreateWidthsAttr([0.003])
            wat.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            wm = UsdShade.Material.Define(stage, "/World/Looks/recwater")
            ws = UsdShade.Shader.Define(stage, "/World/Looks/recwater/S")
            ws.CreateIdAttr("UsdPreviewSurface")
            ws.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.45, 0.72, 0.95))
            ws.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.1)
            wm.CreateSurfaceOutput().ConnectToSource(ws.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(wat.GetPrim()).Bind(wm)
        else:
            wat = UsdGeom.Points(stage.GetPrimAtPath("/World/RecWater"))
        ki_ = [str(x) for x in Sn["obj_paths"]].index("/World/Objects/object_0295")
    for k in range(a_, b_ + 1):
        set_frame(k)
        if wat is not None:
            j_ = int(np.argmin(np.abs(wt_ - float(Sn["t"][k]))))
            Pw = np.asarray(WP[j_], float)
            Mk_ = S["obj_mats"][k, ki_]
            lc = (Pw - Mk_[3, :3]) @ Mk_[:3, :3].T
            ins = (np.hypot(lc[:, 0], lc[:, 1]) < 0.08) & (lc[:, 2] > 0) & (lc[:, 2] < 0.215)
            Pv = Pw[~ins].astype(np.float32)
            wat.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(Pv) if len(Pv) else Vt.Vec3fArray())
        rep.orchestrator.step(rt_subframes=1, pause_timeline=True)
        ff.stdin.write(np.ascontiguousarray(np.asarray(an2.get_data())[:, :, :3]).tobytes())
        if k % 120 == 0:
            print(f"job frame {k}/{b_}  {(time.time() - t_s) / max(1, k - a_ + 1):.2f} s/frame", flush=True)
    ff.stdin.close()
    ff.wait()
    an2.detach()
    rp2.destroy()
    if wat is not None:
        wat.GetPointsAttr().Set(Vt.Vec3fArray())
    A.cam = "film"
    for h_ in heads:
        UsdGeom.Imageable(stage.GetPrimAtPath(h_)).MakeVisible()
    cam.GetVerticalApertureAttr().Set(HA * H / W)
    print("job done", J["out"], flush=True)


if A.serve:
    from PIL import Image
    import glob
    print("RENDER READY", flush=True)
    import json as J_
    while True:
        for fn in sorted(glob.glob(A.serve + "/*.job.json")):
            Jn = J_.load(open(fn))
            os.remove(fn)
            try:
                run_job(Jn)
            except Exception as ex:
                print("job failed", repr(ex)[:400], flush=True)
            open(fn.replace(".job.json", ".done"), "w").write("x")
        for fn in sorted(glob.glob(A.serve + "/*.req.npz")):
            with np.load(fn, allow_pickle=True) as Z_:
                Q = {k_: Z_[k_] for k_ in Z_.files}
            S["xpos"] = Q["xpos"][None]; S["xquat"] = Q["xquat"][None]
            S["obj_mats"] = Q["obj"][None]
            if "prop_xpos" in Q:
                S["prop_xpos"] = Q["prop_xpos"][None]; S["prop_xquat"] = Q["prop_xquat"][None]
            set_frame(0)
            cam_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, Q["eye"])), Gf.Vec3d(*map(float, Q["tgt"])), Gf.Vec3d(0, 0, 1)).GetInverse())
            cam.GetFocalLengthAttr().Set(float(HA / (2 * math.tan(math.radians(float(Q["hfov"])) / 2))))
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=1, pause_timeline=True)
            out = fn.replace(".req.npz", ".png")
            Image.fromarray(np.asarray(rgb.get_data())[:, :, :3]).save(out)
            os.remove(fn)
            print("still", out, flush=True)
        time.sleep(.1)
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
        rep.orchestrator.step(rt_subframes=1, pause_timeline=True)
        Image.fromarray(np.asarray(rgb.get_data())[:, :, :3]).save(A.out.replace(".mp4", f"_{A.cam}_{k:04d}.png"))
        print("still", k, flush=True)
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
            print(f"frame {k}/{b}  {(time.time() - t_start) / max(1, k - a + 1):.2f} s/frame", flush=True)
    ff.stdin.close()
    ff.wait()
print("done", A.out, flush=True)
app.close()
