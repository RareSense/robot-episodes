# session helpers for live2.py (exec'd into the live namespace). Frames are rendered in this same process.
import omni.replicator.core as rep
from PIL import Image
from pxr import Gf, UsdGeom
from omni.physx import get_physx_interface as _gpi
_pi = _gpi()
_cam = UsdGeom.Camera(s.stage.GetPrimAtPath("/World/LiveCam"))
_xop = UsdGeom.Xformable(_cam.GetPrim()).GetOrderedXformOps()[0]
if "ANN2" not in globals():
    RP2 = rep.create.render_product("/World/LiveCam", (960, 540))
    ANN2 = rep.AnnotatorRegistry.get_annotator("rgb")
    ANN2.attach([RP2])
    for _ in range(10):
        app.update()


def shot(name, eye, tgt, hfov=60.0, n=6, tries=6):
    """render the state now from eye to tgt, save live/<tag>/frames/<name>.png"""
    eye = np.array(eye, float); tgt = np.array(tgt, float)
    z = eye - tgt; z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    _xop.Set(Gf.Matrix4d(x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0, eye[0], eye[1], eye[2], 1))
    _cam.GetFocalLengthAttr().Set(20.955 / (2 * math.tan(math.radians(hfov) / 2)))
    _pi.update_transformations(False, True, False, False)      # physics poses to USD, so this process can draw them
    a = None
    for k in range(tries):
        for _ in range(n):
            app.update()
        try:
            rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
        except Exception as e:
            if k == 0:
                print("orchestrator.step:", repr(e)[:200])
        d = ANN2.get_data()
        a = np.asarray(d["data"] if isinstance(d, dict) else d)
        if a.ndim == 3 and a.size:
            break
    if a is None or a.ndim != 3:
        print("shot FAILED: annotator returned", None if a is None else a.shape)
        return
    p = f"{D}/frames/{name}.png"
    Image.fromarray(a[:, :, :3]).save(p)
    print("shot", name, a.shape, "mean", round(float(a[:, :, :3].mean()), 1))


def bagstat(tag=""):
    i = int(np.clip(round((s.tr - R.t[0]) * 60), 0, len(R.t) - 1))
    P = s.props()
    L = s.link_poses()
    hr = L[s.links.index(HANDS[1]), :3]
    print(tag, f"tr {s.tr:.2f} bag", P[0, :3].round(3).tolist(), "rec", np.array(R.prop_xpos[i, 0]).round(3).tolist(), "tag", P[-1, :3].round(3).tolist(), "handR", hr.round(3).tolist(), "rec", R.xpos[i, BN.index(HANDS[1])].round(3).tolist(), "mug", s.obj("object_0316")[:3].round(3).tolist())


def cams(name, c=None, d=.38):
    """two close views of the right-hand pinch: from the robot's left (-x) and from above-front"""
    P = s.props()
    c = P[0, :3] if c is None else np.array(c, float)
    shot(name + "_side", c + np.array([-d, -.06, .10]), c, 40)
    shot(name + "_top", c + np.array([.10, -.12, d]), c, 40)
