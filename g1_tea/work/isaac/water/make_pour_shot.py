"""Synthetic pour shot for testing the water layer (the robot's pour step is being built in parallel).
The mug stands where chunk 1 sets it down (last frame of mug7); the kettle is lifted off its base, carried
over the mug with its spout over the opening, tilted to pour, tilted back and set down again.
Everything else (robot, cameras) comes from mug7's last frame; written in bake.py's shot.npz format.
usage: venv python make_pour_shot.py [out_dir]"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

SRC = Path(str(WORK / "isaac/shots/mug7"))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else str(WORK / "isaac/water/shots/pour_test"))
OUT.mkdir(parents=True, exist_ok=True)
S = dict(np.load(SRC / "shot.npz", allow_pickle=True))
paths = [str(p) for p in S["obj_paths"]]
iK, iM = paths.index("/World/Objects/object_0295"), paths.index("/World/Objects/object_0316")
last = len(S["t"]) - 1
MUG = S["obj_mats"][last, iM]
mug_p = MUG[3, :3]
TIP = np.array([-0.094, 0.0, 0.205])           # spout mouth, kettle frame
BASE = np.array([2.57, 2.36, 0.885])
FPS = 60


def rot_tilt(th, yaw=0.0):
    """kettle rotation: yaw about z, then tip the spout (-x) down by th about the kettle's y axis"""
    c, s = math.cos(-th), math.sin(-th)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry


def pose(tip_w, th):
    R = rot_tilt(th)
    p = tip_w - R @ TIP
    M = np.eye(4)
    M[:3, :3] = R.T
    M[3, :3] = p
    return M


base_tip = BASE + TIP
target = np.array([mug_p[0] + 0.012, mug_p[1], mug_p[2] + 0.085 + 0.045])    # 4.5 cm over the rim
D = math.radians
# waypoints: (time s, tip position, tilt)
PROFILE = sys.argv[2] if len(sys.argv) > 2 else "film"
if PROFILE == "curve":          # slow tilt ramp to measure the pour (outflow vs tilt angle)
    POUR = [(4.0, target + [0.0, 0, 0.01], D(70)), (12.0, target, D(118))]
else:                           # the test pour "a0,a1,ramp_s,hold_s": tilt a0 at 4.0 s, ramp to a1, hold
    a0, a1, ramp, hold = [float(x) for x in (PROFILE if "," in PROFILE else "85,98,3,1").split(",")]
    POUR = [(4.0, target + [0.0, 0, 0.01], D(a0)), (4.0 + ramp, target, D(a1)), (4.0 + ramp + hold, target, D(a1))]
tp = POUR[-1][0]
WP = [(0.0, base_tip, 0.0),
      (0.5, base_tip, 0.0),
      (1.2, base_tip + [0, 0, 0.15], 0.0),
      (2.6, target + [0.05, 0, 0.17], D(12)),
      *POUR,
      (tp + 1.2, target + [0.05, 0, 0.17], D(10)),
      (tp + 2.6, base_tip + [0, 0, 0.15], 0.0),
      (tp + 3.3, base_tip + [0, 0, 0.003], 0.0),
      (tp + 3.6, base_tip, 0.0),
      (tp + 4.2, base_tip, 0.0)]


def minjerk(u):
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


T = WP[-1][0]
F = int(round(T * FPS)) + 1
tips, tilts = np.zeros((F, 3)), np.zeros(F)
for f in range(F):
    t = f / FPS
    for (t0, p0, a0), (t1, p1, a1) in zip(WP[:-1], WP[1:]):
        if t0 <= t <= t1:
            u = minjerk((t - t0) / (t1 - t0))
            tips[f] = np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * u
            tilts[f] = a0 + (a1 - a0) * u
            break
kmats = np.array([pose(tips[f], tilts[f]) for f in range(F)])

# clearance: kettle visual vertices vs the mug (radius 3.4 cm up to the rim) and the counter
try:
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(HANDOFF / "project/tea_twin/scene_v3/environment.usda"))
    V = np.array(UsdGeom.Mesh(st.GetPrimAtPath("/World/Objects/object_0295/visual")).GetPointsAttr().Get())
    worst_mug, worst_ctr = 1.0, 1.0
    for f in range(0, F, 3):
        W = V @ kmats[f, :3, :3] + kmats[f, 3, :3]
        lm = (W - mug_p) @ MUG[:3, :3].T
        r = np.hypot(lm[:, 0], lm[:, 1])
        above = lm[:, 2] - 0.085
        d = np.where(r < 0.034, above, np.hypot(r - 0.034, np.maximum(above, 0)))
        worst_mug = min(worst_mug, d.min())
        if 0.6 < f / FPS < 10.6:
            worst_ctr = min(worst_ctr, W[:, 2].min() - 0.86)
    print(f"clearance: kettle to mug {worst_mug * 100:.1f} cm, to counter (in the air) {worst_ctr * 100:.1f} cm")
except ImportError:
    pass

shot = {k: v for k, v in S.items()}
rep = lambda a: np.repeat(a[last:last + 1], F, axis=0)  # noqa: E731
for key in ("xpos", "xquat", "eye_mats", "switch_z", "switch_on"):
    shot[key] = rep(S[key])
if "prop_xpos" in S:
    shot["prop_xpos"], shot["prop_xquat"] = rep(S["prop_xpos"]), rep(S["prop_xquat"])
om = rep(S["obj_mats"]).copy()
om[:, iK] = kmats
shot["obj_mats"] = om
shot["t"] = np.arange(F) / FPS
# film camera: in front of the counter, a little left of the mug, looking at the mug and the pour
eye, tgt = np.array([1.50, 1.80, 1.14]), np.array([1.80, 2.26, 0.97])
shot["film_eye"], shot["film_tgt"], shot["film_hfov"] = np.repeat([eye], F, 0), np.repeat([tgt], F, 0), 42.0
np.savez(OUT / "shot.npz", **shot)
shutil.copy(SRC / "robot_visual.npz", OUT / "robot_visual.npz")
json.dump({"synthetic": True, "frames": F, "mug": mug_p.tolist(), "pour_target": target.tolist(),
           "waypoints": [(t, list(map(float, p)), math.degrees(a)) for t, p, a in WP]},
          open(OUT / "bake.json", "w"), indent=1)
print("frames", F, f"({T:.1f} s) ->", OUT, "| mug", np.round(mug_p, 4), "target tip", np.round(target, 4))
