"""Test: start state (robot standing at (1.86, 1.90) facing +y, mug at (1.72, 2.24) handle +x, spoon in the
FRESH tub), run skill_stir, save the recording, report checks.
usage: test_stir.py [out.npz] [x y yaw_deg]   (e.g. a perturbed stance as walking leaves it: 1.83 1.93 82)"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import atexit
import json
import sys
import time
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "agents/spoon"))
import numpy as np
import props_spoon as ps
import skill_stir
from motion import Program
from tea_sim import MUG, session

OUT = sys.argv[1] if len(sys.argv) > 1 else str(WORK / "agents/spoon/stir1_rec.npz")
POSES = {MUG: (1.72, 2.24, 0.86, 0.7071, 0, 0, 0.7071)}
t0 = time.time()
XY = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 4 else (1.86, 1.90)
YAW = float(sys.argv[4]) if len(sys.argv) > 4 else 90.0
s = session(robot_xy=XY, robot_yaw_deg=YAW, extras=[ps], poses=POSES)
p = Program(s)
atexit.register(lambda: p.rec.save(OUT))
m, d = s.m, s.d
sb = m.body("env/" + ps.BODY).id
end0 = ps.point(d, m, ps.HANDLE_END)
p.hold(1.0)                                   # the spoon must rest by itself before anything touches it
p.note("spoon at rest", end_drift_mm=round(float(np.linalg.norm(ps.point(d, m, ps.HANDLE_END) - end0)) * 1000, 2))
res = {}
try:
    res = skill_stir.run(p, s)
finally:
    print("RESULT", json.dumps(res))
    print("checks", s.checks.as_dict())
    peak = sorted(zip(p.rec.peak, [m.actuator(i).name for i in s.motors.ids]), reverse=True)[:6]
    print("peak motor loads", [(n, f"{v*100:.0f}%") for v, n in peak])
    print(f"sim {d.time:.1f} s in {time.time() - t0:.0f} s wall")
