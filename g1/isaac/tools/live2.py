"""Live session, one Isaac process: the sim stays open, commands come in through live/<tag>/cmd.py, output goes back to out_<n>.txt, and frames are
rendered in this same process (shot()), so no second Isaac process is needed. Debug tool, AI-operated. Uses the read-only tools in E:/g1_isaac/tools.
usage: live2.py <tag> <task> cpu|gpu"""
import contextlib, ctypes, math, os, sys, time, traceback
TAG, TASK, MODE = sys.argv[1:4]
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)   # below normal
except Exception:
    pass
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
sys.path.insert(0, "E:/g1_isaac/tools")
import g1_core as C
import numpy as np
from pxr import Gf, UsdGeom, UsdLux
D = f"E:/g1_isaac_s2/live/{TAG}"
os.makedirs(D + "/frames", exist_ok=True)
s = C.Sim(app, TASK, gpu=MODE == "gpu")
ns = dict(s=s, C=C, np=np, app=app, D=D, TAG=TAG, os=os, math=math, time=time)
exec(open("E:/g1_isaac/tools/g1_ops.py", encoding="utf-8").read(), ns)
exec(open("E:/g1_isaac/tools/g1_walk.py", encoding="utf-8").read(), ns)

# in-process camera
import omni.replicator.core as rep
st = s.stage
CAM = "/World/LiveCam"
cam = UsdGeom.Camera.Define(st, CAM)
cam.CreateHorizontalApertureAttr(20.955)
cam.CreateVerticalApertureAttr(20.955 * 540 / 960)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 500))
xf = UsdGeom.Xformable(cam.GetPrim())
xf.ClearXformOpOrder()
XOP = xf.AddTransformOp()
if os.environ.get("LIVE_DOME", "1") == "1":
    dl = UsdLux.DomeLight.Define(st, "/World/LiveDome")
    dl.CreateIntensityAttr(float(os.environ.get("LIVE_DOME_I", "600")))
RP = rep.create.render_product(CAM, (960, 540))
ANN = rep.AnnotatorRegistry.get_annotator("rgb")
ANN.attach([RP])


def shot(name, eye, tgt, hfov=60.0, n=8):
    """render the state now from eye to tgt, save live/<tag>/frames/<name>.png"""
    from PIL import Image
    eye = np.array(eye, float); tgt = np.array(tgt, float)
    z = eye - tgt; z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = Gf.Matrix4d(x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0, eye[0], eye[1], eye[2], 1)
    XOP.Set(M)
    cam.GetFocalLengthAttr().Set(20.955 / (2 * math.tan(math.radians(hfov) / 2)))
    for _ in range(n):
        app.update()
    d = ANN.get_data()
    a = np.asarray(d["data"] if isinstance(d, dict) else d)
    p = f"{D}/frames/{name}.png"
    Image.fromarray(a[:, :, :3]).save(p)
    print("shot", p, a.shape, "mean", round(float(a[:, :, :3].mean()), 1))


ns["shot"] = shot
print("READY", flush=True)
open(D + "/ready.txt", "w").write(time.strftime("%H:%M:%S"))
n = 0
while True:
    cmd = D + "/cmd.py"
    if os.path.exists(D + "/quit.txt"):
        print("QUIT", flush=True)
        os._exit(0)
    if not os.path.exists(cmd):
        time.sleep(.1)
        continue
    n += 1
    run = f"{D}/cmd_{n}.py"
    os.replace(cmd, run)
    tmp = f"{D}/out_{n}.tmp"
    with open(tmp, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        try:
            exec(compile(open(run, encoding="utf-8").read(), run, "exec"), ns)
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc(file=f)
    os.replace(tmp, f"{D}/out_{n}.txt")
