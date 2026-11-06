"""One continuous take of a task on Isaac Sim PhysX, AI-operated. Script written from what worked live. Starts from the recording's first frame; after that nothing writes state:
joint drive targets only (recorded path + small live fixes), motor limits on, no welds, no teleports, no external forces.
usage: g1_take.py <tag> <task> cpu|gpu   output E:/g1_isaac/<task>/takes/<tag>/"""
import json, os, sys, time, traceback
TAG, TASK, MODE = sys.argv[1:4]
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
sys.path.insert(0, "E:/g1_isaac/tools")
import g1_core as C
import numpy as np
OUT = f"E:/g1_isaac/{TASK}/takes/{TAG}"
os.makedirs(OUT, exist_ok=True)
s = C.Sim(app, TASK, gpu=MODE == "gpu")
ns = dict(s=s, C=C, np=np, app=app, D=OUT, TAG=TAG, os=os)
exec(open("E:/g1_isaac/tools/g1_ops.py", encoding="utf-8").read(), ns)
exec(open("E:/g1_isaac/tools/g1_walk.py", encoding="utf-8").read(), ns)
fix = f"E:/g1_isaac/tools/fix_{TASK}.py"
if os.path.exists(fix):
    exec(open(fix, encoding="utf-8").read(), ns)
R, FR, advance, status = ns["R"], ns["FR"], ns["advance"], ns["status"]
T1 = float(os.environ.get("TAKE_END", R.t[-1]))
print("TAKE", TAG, TASK, MODE, flush=True)
t0 = time.time()
nx = s.tr + 1
peak = np.zeros(s.n)
try:
    while s.tr < T1:
        advance(6)
        f = np.abs(s.art.get_dof_projected_joint_forces()[0].astype(float)) if hasattr(s.art, "get_dof_projected_joint_forces") else peak
        peak = np.maximum(peak, f)
        if s.tr >= nx:
            status(); sys.stdout.flush()
            nx += 1
            print("wall", round(time.time() - t0), flush=True)
        if s.link("pelvis")[2] < .5:
            print("FALLEN at", s.tr, flush=True)
            break
except Exception:
    traceback.print_exc()
status("END")
A = {k: np.array(v) for k, v in FR.items()}
np.savez_compressed(OUT + "/rec60.npz", names=np.array(s.names), links=np.array(s.links), objects=np.array(ns["OBJK"]), peak_force=peak, limits=s.lim, **A)
print("SAVED", len(A["t"]), "frames", flush=True)
print("ALL DONE", flush=True)
sys.stdout.flush()
os._exit(0)
