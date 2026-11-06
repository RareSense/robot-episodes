"""Test of skill_pour (chunk 3 in physics): builds the start state (robot at (1.86, 1.90) facing +y, the mug
at (1.72, 2.24) handle +x via `poses`, the kettle on its base), runs the skill, saves the 60 fps recording
(also on a crash: atexit) and prints the gates.
usage (PowerShell): $env:TEA_FOOT_TC = "0.007"; python test_pour.py [out.npz] [kwargs-json] [x,y,heading]"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import atexit
import json
import sys
import time

sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "agents/kettle"))
import numpy as np  # noqa: E402

from motion import Program  # noqa: E402
from tea_sim import MUG, SWITCH, session  # noqa: E402
import skill_pour  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else str(WORK / "agents/kettle/pour_rec.npz")
kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
# start stance "x,y,heading_deg" (the film run reaches it only within ~3-5 cm / 5-8 deg, e.g. 1.83,1.93,80)
sx, sy, syaw = (float(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "1.86,1.90,90").split(","))
t0 = time.time()
s = session(robot_xy=(sx, sy), robot_yaw_deg=syaw, poses={MUG: (1.72, 2.24, 0.86, 0.7071, 0.0, 0.0, 0.7071)})
p = Program(s)
m, d = s.m, s.d
swq = int(m.joint("env/" + SWITCH).qposadr[0])
atexit.register(lambda: p.rec.save(OUT, switch_qadr=swq))
res = skill_pour.run(p, s, **kw)
print("RESULT", json.dumps(res))
peak = sorted(zip(p.rec.peak, [m.actuator(i).name for i in s.motors.ids]), reverse=True)[:8]
print("peak motor loads", [(n, f"{v*100:.0f}%") for v, n in peak])
print("checks", s.checks.as_dict())
print(f"sim {d.time:.1f} s in {time.time() - t0:.0f} s wall")
json.dump(dict(result=res, log=p.log, peak=[(n, round(float(v), 3)) for v, n in peak]),
          open(OUT.replace(".npz", ".json"), "w"), indent=1, default=float)
