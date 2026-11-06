"""The cup step on its own: the full mug (TEA_MUG_KG=0.48) stands at (1.72, 2.24) handle +x, the G1 stands at
(1.86, 1.90) facing +y; skill_cup.run() takes it by the handle with the right hand, lifts it, turns left, walks ~1 m
toward -x carrying it and stands holding it. Saves the 60 fps recording (also on a crash) and the result JSON.

Physically simulated: MuJoCo 600 Hz, torque-limited G1 motors, walking network + byor whole-body controller, mug a
free body held by finger contact. Nothing writes qpos/qvel after the initial state.
usage: test_cup.py [out.npz] [key=value ...]   (options go to skill_cup.run, e.g. stop_after='lift' v=0.35)"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import ast
import atexit
import json
import os
import sys
import time

os.environ.setdefault("TEA_FOOT_TC", "0.007")
os.environ.setdefault("TEA_MUG_KG", "0.48")
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "agents/cup"))
from motion import Program  # noqa: E402
from tea_sim import MUG, session  # noqa: E402

import props_cup  # noqa: E402
import skill_cup  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 and "=" not in sys.argv[1] else str(WORK / "agents/cup/cup_rec.npz")
kw = {}
for a in sys.argv[1:]:
    if "=" in a:
        k, v = a.split("=", 1)
        try:
            kw[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kw[k] = v
start = kw.pop("start", (1.86, 1.90, 90.0))
extras = [props_cup] if kw.pop("props", False) else []      # props_cup: condim-4 handle (handle-pinch tests only)
poses = {MUG: (1.72, 2.24, 0.86, 0.7071068, 0.0, 0.0, 0.7071068)}
t0 = time.time()
s = session(robot_xy=start[:2], robot_yaw_deg=start[2], poses=poses, extras=extras)
p = Program(s)
atexit.register(lambda: p.rec.save(OUT))                 # keep the recording even if the run stops early
p.hold(kw.pop("settle", 0.5))                            # the scene settles (robot standing, mug on the counter)
res = skill_cup.run(p, s, **kw)
m = s.m
peak = sorted(zip(p.rec.peak, [m.actuator(i).name for i in s.motors.ids]), reverse=True)[:8]
res["peak_motor_loads"] = [(n, f"{v * 100:.0f}%") for v, n in peak]
res["wall_s"] = round(time.time() - t0)
res["sim_s"] = round(float(s.d.time), 2)
res["options"] = kw
res["mug_kg"] = float(os.environ["TEA_MUG_KG"])
print("RESULT", json.dumps(res, default=str))
with open(OUT.replace(".npz", "_result.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
