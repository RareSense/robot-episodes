"""Test: start state (robot standing at (1.86, 1.90) facing +y, mug at (1.72, 2.24) handle +x, the glass jar with
its screw lid and the powder at (1.90, 2.36), the dish spoon in the FRESH tub), run skill_powder, save the
recording (atexit), print the results and the checks.
usage: test_powder.py out.npz [key=value ...]   (skill keywords, e.g. stop_after_lid=1 lid_off=1 end=hold)
env: POWDER_LID_OFF=1 starts with the lid already on the counter (test fixture for the scoop part),
     POWDER_GRAINS=n fewer grains (fast tests of the lid part)."""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import atexit
import json
import os
import sys
import time

sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "agents/powder"))
import numpy as np  # noqa: E402
import props_powder as pp  # noqa: E402
import props_pspoon as ps  # noqa: E402
import skill_powder  # noqa: E402
from motion import Program  # noqa: E402
from tea_sim import MUG, session  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else str(WORK / "agents/powder/powder1_rec.npz")
kw = {}
for a in sys.argv[2:]:
    k, v = a.split("=")
    try:
        v = float(v)
        v = int(v) if v.is_integer() and k not in ("psi0", "psi1") else v
    except ValueError:
        pass
    kw[k] = v
if pp.LID_OFF:
    kw["lid_off"] = True
t0 = time.time()
POSES = {MUG: (1.72, 2.24, 0.86, 0.7071, 0, 0, 0.7071)}
s = session(robot_xy=(1.86, 1.90), robot_yaw_deg=90.0, extras=[pp, ps], poses=POSES)
p = Program(s)
atexit.register(lambda: p.rec.save(OUT))
m, d = s.m, s.d
print("model: nbody", m.nbody, "nv", m.nv, "npair", m.npair, "neq", m.neq, "grains", pp.N_GRAINS, flush=True)
th = pp.install(s)
print("sleep:", getattr(th, "sleep", None), flush=True)
TRACK = ["env/object_0309_body", "env/" + pp.LID, "env/" + ps.BODY, "env/" + MUG + "_body"]
th.track = [m.body(n).id for n in TRACK]                 # 600 Hz poses for the powder pass (read-only)
atexit.register(lambda: th.save_poses(OUT.replace("_rec.npz", "_poses600.npz"), TRACK))
end0 = ps.point(d, m, ps.HANDLE_END)
p.hold(float(os.environ.get("POWDER_HOLD0", "1.0")))
p.note("props at rest", spoon_end_drift_mm=round(float(np.linalg.norm(ps.point(d, m, ps.HANDLE_END) - end0)) * 1000, 2))
def _stopper(_s):                       # touch <out>.stop to end a run early (the recording is still saved)
    if os.path.exists(OUT + ".stop"):
        print("stopped by", OUT + ".stop", flush=True)
        sys.exit(0)


p.mv.on_tick = _stopper
res = {}
try:
    res = skill_powder.run(p, s, **kw)
finally:
    print("RESULT", json.dumps(res, default=str))
    print("checks", s.checks.as_dict())
    peak = sorted(zip(p.rec.peak, [m.actuator(i).name for i in s.motors.ids]), reverse=True)[:8]
    print("peak motor loads", [(n, f"{v*100:.0f}%") for v, n in peak])
    print(f"sim {d.time:.1f} s in {time.time() - t0:.0f} s wall")
    with open(OUT.replace(".npz", ".json"), "w") as f:
        json.dump(dict(result=res, checks={k: (float(v) if v is not None else None) for k, v in s.checks.as_dict().items()},
                       peak=[(n, round(float(v), 3)) for v, n in peak], sim_s=d.time, wall_s=time.time() - t0, kw=kw), f, indent=1, default=str)




