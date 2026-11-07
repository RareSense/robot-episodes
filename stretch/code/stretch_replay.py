"""Renders a recorded Stretch pizza take from its recording. REPLAY: recorded poses only, physics is off, nothing is simulated here. Labelled so in every output.
Three views, frame file name = sample number: wrist (link d405_cam), head (link realsense), fixed (one fixed camera, no cuts).
Resumes from the frames already on disk. usage (through stretch_launch.py): stretch_replay.py <tag> <rec tag>   env RP_MAX (stop after n samples, for a check), RP_STRIDE (default 1)."""
import json
import math
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

TAG, REC = sys.argv[1], sys.argv[2]
from isaacsim import SimulationApp

W, H = 640, 480
app = SimulationApp({"headless": True, "width": W, "height": H, "renderer": "RaytracedLighting"})
import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

import pizza_kitchen as K
import pizza_oven as O
import stretch_core as C

cs = carb.settings.get_settings()
for k, v in {"/rtx/rendermode": "RaytracedLighting", "/rtx/post/aa/op": 3, "/rtx/post/tonemap/op": 6, "/rtx/post/tonemap/filmIso": 100.0 * 2 ** 1.5, "/rtx/post/histogram/enabled": False,
             "/rtx/directLighting/sampledLighting/enabled": True, "/omni/replicator/RTSubframes": 1, "/app/viewport/grid/enabled": False, "/app/viewport/show/grid": False,
             "/app/viewport/outline/enabled": False, "/rtx/translucency/enabled": True, "/rtx/raytracing/fractionalCutoutOpacity": True}.items():
    cs.set(k, v)
D = os.environ.get("RP_DIR", f"E:/stretch_assets/data/{REC}_views")
LOOP = os.environ.get("RP_LOOP", "rep")
UPD = int(os.environ.get("RP_UPD", 2))
VIEWS = {"wrist": ("d405_cam", 87.0), "head": ("realsense", 75.0)}
THIRD = (58.0, (2.45, .80, 1.72), (1.22, 2.20, .92))
S = np.load(f"E:/stretch_assets/data/{REC}_rec.npz")
LINKN = [str(x) for x in S["links"]]
LT, OV, PL, PI, BD, HT = S["link"].astype(float), S["oven"].astype(float), S["plate"].astype(float), S["pizza"].astype(float), S["body"].astype(float), S["heat"]
N = len(S["t"])
ctx = omni.usd.get_context()
ctx.open_stage(K.stage_file())
for _ in range(5):
    app.update()
st = ctx.get_stage()
st.SetEditTarget(st.GetSessionLayer())
objs = {}
for p in st.Traverse():
    if p.GetName().startswith("object_") and p.GetTypeName() == "Xform":
        objs.setdefault(p.GetName(), p)
for k, v in json.load(open(C.CORR)).items():
    p = objs.get(k)
    if k.startswith("_") or p is None or len(v) != 7:
        continue
    xf = UsdGeom.Xformable(p)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*v[:3]))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(v[3], v[4], v[5], v[6]))
r = st.DefinePrim(C.PRIM, "Xform")
r.GetReferences().AddReference(C.USD)
r.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
app.update()
K.build(st)
app.update()


def mat(T):
    return Gf.Matrix4d().SetTransform(Gf.Rotation(Gf.Quatd(float(T[6]), float(T[3]), float(T[4]), float(T[5]))), Gf.Vec3d(float(T[0]), float(T[1]), float(T[2])))


paths = {}
for p in Usd.PrimRange(r, Usd.TraverseInstanceProxies()):
    if p.GetName() in LINKN and p.HasAPI(UsdPhysics.RigidBodyAPI):
        paths[p.GetName()] = p.GetPath().pathString
missing = [n for n in LINKN if n not in paths]
print("links found", len(paths), "missing", missing, flush=True)
order = sorted(paths, key=lambda n: paths[n].count("/"))
pose = []
xc = UsdGeom.XformCache()
for n in order:
    par = paths[n].rsplit("/", 1)[0]
    anc = None
    while par:
        hit = [m for m in paths if paths[m] == par]
        if hit:
            anc = hit[0]
            break
        par = par.rsplit("/", 1)[0] if "/" in par else ""
    pr = st.GetPrimAtPath(paths[n])
    static = None if anc else xc.GetLocalToWorldTransform(pr.GetParent())
    op = UsdGeom.Xformable(pr).MakeMatrixXform()
    pose.append((LINKN.index(n), LINKN.index(anc) if anc else None, static, op))


def xop(path):
    return UsdGeom.Xformable(st.GetPrimAtPath(path)).MakeMatrixXform()


op_body, op_door, op_plate, op_pizza = xop(O.BODY), xop(O.DOOR), xop(K.PLATE), xop(K.PIZZA)
op_knob = [xop(p) for p in O.KNOBS]
T_DOOR = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, O.HY, O.HZ))
T_KNOB = [Gf.Matrix4d().SetTranslate(Gf.Vec3d(O.KNOB_X, -.002, z)) for z in O.KNOB_Z]
RCAM = {"wrist": Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 180.0)),      # d405_cam looks down +z, x left, y up. USD cameras look down -z
        "head": Gf.Matrix4d(0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, .03, 0, 0, 1)}        # realsense link looks down +x, y up; 30 mm forward clears the head shell


def cam(path, hfov):
    c = UsdGeom.Camera.Define(st, path)
    c.CreateHorizontalApertureAttr(36.0)
    c.CreateVerticalApertureAttr(36.0 * H / W)
    c.CreateClippingRangeAttr(Gf.Vec2f(.02, 100))
    c.CreateFocalLengthAttr(float(36.0 / (2 * math.tan(math.radians(hfov) / 2))))
    return UsdGeom.Xformable(c).MakeMatrixXform()


cops = {k: (LINKN.index(v[0]), cam("/World/rp_" + k, v[1])) for k, v in VIEWS.items()}
FOLLOW = [float(x) for x in os.environ["RP_FOLLOW"].split(",")] if os.environ.get("RP_FOLLOW") else None
cam_fixed = cam("/World/rp_fixed", float(os.environ.get("RP_HFOV", THIRD[0])))
cam_fixed.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*THIRD[1]), Gf.Vec3d(*THIRD[2]), Gf.Vec3d(0, 0, 1)).GetInverse())
last = {}


def frame(f):
    L = LT[f]
    for i, j, static, op in pose:
        key = (L[i].tobytes(), None if j is None else L[j].tobytes())
        if last.get(i) == key:
            continue
        last[i] = key
        op.Set(mat(L[i]) * (static.GetInverse() if j is None else mat(L[j]).GetInverse()))
    MB = mat(BD[f])
    op_body.Set(MB)
    op_door.Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), math.degrees(float(OV[f, 0])))) * T_DOOR * MB)
    for i, o in enumerate(op_knob):
        o.Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(float(OV[f, 1 + i])))) * T_KNOB[i] * MB)
    op_plate.Set(mat(PL[f]))
    if FOLLOW:                                # check frames only: fixed camera near the plate
        e = PL[f][:3] + np.array(FOLLOW[:3])
        tg = PL[f][:3] + np.array(FOLLOW[3:6])
        cam_fixed.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, e)), Gf.Vec3d(*map(float, tg)), Gf.Vec3d(0, 0, 1)).GetInverse())
    op_pizza.Set(mat(PI[f]))
    for k, (i, o) in cops.items():
        o.Set(RCAM[k] * mat(L[i]))
    h = (int(HT[f, 0]), int(HT[f, 1]))
    if last.get("heat") != h:
        last["heat"] = h
        UsdShade.Shader(st.GetPrimAtPath("/World/Looks/pz_pizza_look/tex")).GetInput("file").Set(K.PIZZA_TEX + f"albedo_{h[0]}.png")
        O.set_on(types.SimpleNamespace(stage=st), bool(h[1]))


rep.orchestrator.set_capture_on_play(False)
ann = {}
for k in ("wrist", "fixed", "head"):
    os.makedirs(f"{D}/{k}", exist_ok=True)
    rp = rep.create.render_product("/World/rp_" + k, (W, H))
    a = rep.AnnotatorRegistry.get_annotator("rgb")
    a.attach([rp])
    ann[k] = a
stride = int(os.environ.get("RP_STRIDE", 1))
start = 0
stop = min(N, start + int(os.environ.get("RP_MAX", N)))
print("replay samples", start, "to", stop, "of", N, flush=True)
frame(start)
for _ in range(6):
    rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
if os.environ.get("RP_PICK"):            # check frames only
    for f in map(int, os.environ["RP_PICK"].split(",")):
        frame(f)
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=2, pause_timeline=False)
        for k, a in ann.items():
            Image.fromarray(np.asarray(a.get_data())[:, :, :3]).save(f"E:/stretch_assets/audit/{os.environ.get('RP_NAME', 'rp')}_{f}_{k}.jpg", quality=85)
        print("picked", f, flush=True)
    print("END status PICK DONE", flush=True)
    os._exit(0)
pool = ThreadPoolExecutor(6)


def save(im, p):
    Image.fromarray(im).save(p, quality=90)


EVERY = int(os.environ.get("RP_EVERY", 1))
todo = [f for f in range(0, N, EVERY) if not all(os.path.exists(f"{D}/{k}/{f:06d}.jpg") for k in ann)]
print("to render", len(todo), "of", len(range(0, N, EVERY)), flush=True)
t0 = time.time()
for n, f in enumerate(todo):
    frame(f)
    rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
    for k, a in ann.items():
        im = np.asarray(a.get_data())
        if im.ndim == 3:
            pool.submit(save, np.ascontiguousarray(im[:, :, :3]), f"{D}/{k}/{f:06d}.jpg")
    if n % 300 == 0:
        print("frame", f, "done", n, "of", len(todo), "rate", round((n + 1) / (time.time() - t0), 2), "per s", flush=True)
pool.shutdown()
print("END status REPLAY DONE", flush=True)
os._exit(0)
