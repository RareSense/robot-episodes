"""Stretch pizza task, live session in the kitchen copy. The sim opens, stays open and frozen. Commands come in through live_tap. Debug only: not a final run.
Rendered in this same kit. Replicator captureOnPlay is off, or its step plays the timeline and kills the simulation view.
usage (through stretch_launch.py, LIVE_PAUSE=*): stretch_pizza.py <tag> [gpu]"""
import importlib
import json
import math
import os
import sys

TAG = sys.argv[1]
GPU = len(sys.argv) > 2 and sys.argv[2] == "gpu"
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 800, "height": 450, "renderer": "RaytracedLighting"})
import carb
import numpy as np
import omni.physx.bindings._physx as pb

import stretch_core as C
import pizza_kitchen as K

cs = carb.settings.get_settings()
for k, v in {"/rtx/rendermode": "RaytracedLighting", "/rtx/post/aa/op": 3, "/rtx/post/tonemap/op": 6, "/rtx/post/tonemap/filmIso": 100.0 * 2 ** 1.5, "/rtx/post/histogram/enabled": False,
             "/rtx/directLighting/sampledLighting/enabled": True, "/omni/replicator/RTSubframes": 1, "/app/viewport/grid/enabled": False, "/app/viewport/show/grid": False,
             "/app/viewport/outline/enabled": False, "/rtx/translucency/enabled": True, "/rtx/raytracing/fractionalCutoutOpacity": True}.items():
    cs.set(k, v)
cs.set(pb.SETTING_UPDATE_TO_USD, True)
START = os.environ.get("PZ_START", "plate")
if START in ("door", "insert"):         # test starts: plate already on its set-down spot. Not a task result
    K.PLATE_XY = K.SET_XY
if START == "takeout":                 # test start: plate on the rack, door at its stop, pizza baked. Not a task result
    K.PLATE_XY, K.COUNTER_Z = (1.1796, 2.2641), .9995
s = C.Sim(app, xy={"plate": (1.649, 1.675), "door": (1.169, 1.40), "insert": (.841, 1.70), "takeout": (1.169, 1.40)}[START], q={"joint_lift": 1.05, "joint_gripper_slide": .04}, pre=[K.build], add_props=False, gpu=GPU, room=K.stage_file())
import pz

pz.oven_setup(s)
if START in ("insert", "takeout"):      # test start only: door authored at its stop, so the insert can be practised without replaying the door
    q_ = s.oven.get_dof_positions().copy()
    q_[0, 0] = math.radians(88.0)
    i_ = np.arange(1, dtype=np.uint32)
    s.oven.set_dof_positions(q_, i_)
    s.oven.set_dof_position_targets(np.zeros_like(q_), i_)
if os.environ.get("PZ_DOOR0"):         # test start only: door authored part open for the short door check
    q_ = s.oven.get_dof_positions().copy()
    q_[0, 0] = math.radians(float(os.environ["PZ_DOOR0"]))
    s.oven.set_dof_positions(q_, np.arange(1, dtype=np.uint32))
s.wait(.5)
if not os.environ.get("PZ_VIDEO") and not os.environ.get("PZ_RUN"):
    pz.render_setup(s)
if START == "takeout":
    pz.heat_set(s, 1.0)
print("BUILT", pz.st(s), flush=True)
if os.environ.get("PZ_RUN"):             # the whole task from the plate start, written from what worked live
    import pz_task
    if os.environ.get("PZ_VIDEO"):
        pz.rep_setup()
        pz.vid_on(s, TAG)
    if os.environ.get("PZ_REC"):         # read-only full recording, wraps s.step after the video
        import pz_rec
        pz_rec.on(s, TAG)
    try:
        pz_task.run(s, TAG)
    except pz_task.Stop as e:
        print("END status STOPPED", str(e), pz.snap_row(s), flush=True)
    if os.environ.get("PZ_REC"):
        pz_rec.save()
        print("REC END", pz.snap_row(s), flush=True)
        sys.stdout.flush()
        os._exit(0)
while True:
    print("STAGE", "live", {"t": round(s.t, 2)}, flush=True)
