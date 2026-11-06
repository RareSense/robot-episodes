"""Replay a recorded physics run exactly as it was simulated (the full MuJoCo state at 60 fps): in the MuJoCo viewer,
or as an MP4 from a camera that follows the robot (or from the robot's own eye camera).

  python work/replay.py recordings/3_pour_pour35.npz                 # interactive viewer
  python work/replay.py recordings/3_pour_pour35.npz --video pour.mp4 [--cam follow|eye] [--size 1280x720]

The scene (twin room + G1 + the step's props) is rebuilt with tea_sim.build(), the same builder the runs used, so
qpos maps 1:1. Video export needs ffmpeg on PATH."""
import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tea_paths  # noqa: E402

STEPS = {  # recording name prefix -> (prop modules, env)
    "1": ([], {}),
    "2": (["agents/teabag/props_teabag.py"], {"TB_START": "stand"}),
    "3": ([], {}),
    "4": (["agents/spoon/props_spoon.py"], {}),
    "5": ([], {"TEA_MUG_KG": "0.48"}),
}

ap = argparse.ArgumentParser()
ap.add_argument("rec")
ap.add_argument("--video", default="")
ap.add_argument("--cam", default="follow", choices=["follow", "eye"])
ap.add_argument("--size", default="1280x720")
ap.add_argument("--extras", default="", help="prop modules (comma-separated), when the file name has no step prefix")
A = ap.parse_args()

mods, env = STEPS.get(Path(A.rec).name[:1], ([], {}))
if A.extras:
    mods = A.extras.split(",")
os.environ.update(env)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from tea_sim import build  # noqa: E402
from nova_robotics.byor.locomotion import WalkingPolicy  # noqa: E402


def load(path):
    path = Path(path) if Path(path).is_absolute() else tea_paths.WORK / path
    sp = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(sp)
    sys.path.insert(0, str(path.parent))
    sp.loader.exec_module(mod)
    return mod


r = np.load(A.rec, allow_pickle=True)
spec, _, _ = build(extras=[load(p) for p in mods])
WalkingPolicy.prepare_spec(spec)
m = spec.compile()
# the twin includes the street block (extent ~680 m) and MuJoCo scales its near clip with the extent (~7 m), which
# would cut the whole room away: clip at ~1 cm instead
m.vis.map.znear = 0.01 / m.stat.extent
d = mujoco.MjData(m)
Q, t = r["qpos"], r["t"]
if Q.shape[1] != m.nq:
    raise SystemExit(f"recording has {Q.shape[1]} state values, the rebuilt scene {m.nq}: wrong --extras for this step?")
print(f"{len(Q)} frames, {t[-1] - t[0]:.1f} s, {m.nu} motors, {m.nbody} bodies")


def pose(k):
    d.qpos[:] = Q[k]
    mujoco.mj_forward(m, d)


if A.video:
    W, H = map(int, A.size.split("x"))
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, 640), max(H, 480)
    ren = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    if A.cam == "eye":
        cam.type, cam.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, m.camera("eye").id
    else:
        cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, m.body("pelvis").id
        cam.distance, cam.azimuth, cam.elevation = 1.9, 90.0, -25.0       # behind the robot, facing the counter (+y)
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", "60", "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", A.video],
                          stdin=subprocess.PIPE)
    for k in range(len(Q)):
        pose(k)
        ren.update_scene(d, camera=cam)
        ff.stdin.write(ren.render().tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", A.video)
else:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as v:
        t0 = time.time()
        for k in range(len(Q)):
            if not v.is_running():
                break
            pose(k)
            v.sync()
            time.sleep(max(0.0, (t[k] - t[0]) - (time.time() - t0)))
