"""Chunk 1 in physics at the person's pace (fast version of work/test_mug3.py): the G1 walks
in from (0.62, 1.15) to the mug rack, the LEFT hand takes mug object_0316 off the middle tier, draws it
out into a carry pose close in front of the chest, the robot turns right and walks along the counter,
and sets the mug down upright, handle right, at (1.72, 2.24). The motion itself is skill_mug.run().

Physically simulated: MuJoCo 600 Hz, torque-limited G1 motors, walking network + byor whole-body
controller, mug a free body held by finger contact. Nothing writes qpos/qvel after the initial state.
usage: test_mug_fast.py [out.npz] [key=value ...]   (options go to skill_mug.run, e.g. v2=0.6 turn_deg=50)
"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import ast
import atexit
import json
import sys
import time

sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "agents/mugfast"))
from motion import Program  # noqa: E402
from tea_sim import session  # noqa: E402

import skill_mug  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 and "=" not in sys.argv[1] else "mugfast_rec.npz"
kw = {}
for a in sys.argv[1:]:
    if "=" in a:
        k, v = a.split("=", 1)
        try:
            kw[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kw[k] = v
t0 = time.time()
s = session(robot_xy=(0.62, 1.15), robot_yaw_deg=90.0)
p = Program(s)
atexit.register(lambda: p.rec.save(OUT))                 # keep the recording even if the run stops early
res = skill_mug.run(p, s, **kw)
m = s.m
peak = sorted(zip(p.rec.peak, [m.actuator(i).name for i in s.motors.ids]), reverse=True)[:8]
res["peak_motor_loads"] = [(n, f"{v * 100:.0f}%") for v, n in peak]
res["wall_s"] = round(time.time() - t0)
res["sim_s"] = round(float(s.d.time), 2)
res["options"] = kw
print("RESULT", json.dumps(res, default=str))
with open(OUT.replace(".npz", "_result.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
