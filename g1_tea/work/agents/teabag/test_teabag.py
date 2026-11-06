"""Tea-bag step in physics (test): session at the stance with the tea-bag prop and the mug where step 1 left
it, run skill_teabag, save the recording (also on a crash), print the checks.
usage: test_teabag.py [out.npz]      (env TB_START=rig|tray selects the prop's start pose)"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import atexit
import os
import sys
import time

sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "agents/teabag"))
os.environ.setdefault("TB_START", "stand")
import numpy as np
import props_teabag as tb
import skill_teabag
from motion import Program
from tea_sim import MUG, session

OUT = sys.argv[1] if len(sys.argv) > 1 else str(WORK / "agents/teabag/teabag_rec.npz")
STANCE = tuple(float(v) for v in os.environ.get("TB_STANCE", "1.90,1.88").split(","))
YAW = float(os.environ.get("TB_YAW", "90"))
MUG_POSE = (1.72, 2.24, 0.86, 0.7071068, 0, 0, 0.7071068)
t0 = time.time()
s = session(robot_xy=STANCE, robot_yaw_deg=YAW, extras=[tb], poses={MUG: MUG_POSE})
p = Program(s)
atexit.register(lambda: p.rec.save(OUT))
p.hold(1.0)                                         # the prop settles where it lies (robot standing still)
ms = os.environ.get('TB_MUGSTANCE')
res = skill_teabag.run(p, s, stop_after=os.environ.get('TB_STOP'), mug_stance=tuple(float(v) for v in ms.split(',')) if ms else None)
p.hold(0.5)
print("RESULT", res)
print("checks", s.checks.as_dict())
peak = sorted(zip(p.rec.peak, [s.m.actuator(i).name for i in s.motors.ids]), reverse=True)[:6]
print("peak motor loads", [(n, f"{v*100:.0f}%") for v, n in peak])
print(f"sim {s.d.time:.1f} s in {time.time() - t0:.0f} s wall")
