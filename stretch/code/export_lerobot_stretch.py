"""Stretch pizza take as one LeRobot v3.0 episode. Per 30 Hz sample of sim time: joint state, velocities, base pose, oven joints, plate and pizza poses, the commanded targets as action.
Camera views are added when a views dir is given: <views>/<head|wrist|fixed>/<sample index, 6 digits>.jpg, one image per sample.
usage: <lerobot venv python> export_lerobot_stretch.py <rec.npz> <out_dir, must not exist> [views dir] [stride, default 1]
stride 3 keeps every 3rd sample, so 10 fps from the 30 Hz recording. Action then stays the commanded target at each kept sample."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
VIEWS = Path(sys.argv[3]) if len(sys.argv) > 3 else None
STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 1
TASK = "Put the pizza plate in the oven, set the knobs, then take it out and carry it back."
d = dict(np.load(REC))                 # one read: indexing an npz directly decompresses the array every time
names = [str(x) for x in d["names"]]
POSE = ["x", "y", "z", "qx", "qy", "qz", "qw"]
f32 = "float32"
features = {
    "observation.state": {"dtype": f32, "shape": (len(names),), "names": names},
    "observation.velocity": {"dtype": f32, "shape": (len(names),), "names": names},
    "observation.base_pose": {"dtype": f32, "shape": (7,), "names": POSE},
    "observation.oven_joints": {"dtype": f32, "shape": (d["oven"].shape[1],), "names": ["door"] + [f"knob{i}" for i in range(d["oven"].shape[1] - 1)]},
    "observation.environment_state": {"dtype": f32, "shape": (14,), "names": [f"{o}_{k}" for o in ("plate", "pizza") for k in POSE]},
    "action": {"dtype": f32, "shape": (2 * len(names),), "names": [f"pos_target_{n}" for n in names] + [f"vel_target_{n}" for n in names]},
}
cams = []
if VIEWS:
    cams = [c for c in ("head", "wrist", "fixed") if (VIEWS / c).is_dir() and any((VIEWS / c).glob("*.jpg"))]
    for c in cams:
        im = Image.open(sorted((VIEWS / c).glob("*.jpg"))[0])
        features[f"observation.images.{c}"] = {"dtype": "video", "shape": (im.height, im.width, 3), "names": ["height", "width", "channels"]}
ds = LeRobotDataset.create("abdullahanxie/stretch_pizza_sim_demo", fps=30 // STRIDE, root=OUT, robot_type="hello_robot_stretch_3", features=features, use_videos=bool(cams))
n = len(d["t"])
for f in range(0, n, STRIDE):
    row = {
        "observation.state": d["q"][f].astype(f32),
        "observation.velocity": d["dq"][f].astype(f32),
        "observation.base_pose": d["root"][f].astype(f32),
        "observation.oven_joints": d["oven"][f].astype(f32),
        "observation.environment_state": np.r_[d["plate"][f], d["pizza"][f]].astype(f32),
        "action": np.r_[d["qd"][f], d["vd"][f]].astype(f32),
        "task": TASK,
    }
    for c in cams:
        row[f"observation.images.{c}"] = np.asarray(Image.open(VIEWS / c / f"{f:06d}.jpg").convert("RGB"))
    ds.add_frame(row)
    if f % (5000 * STRIDE) == 0:
        print(f, "of", n, flush=True)
ds.save_episode()
ds.finalize()
print("wrote", OUT, n, "frames", cams, flush=True)
