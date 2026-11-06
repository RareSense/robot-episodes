"""The tea bag prop alone: does it rest stably in the tray compartment? (robot standing still)
usage: test_settle.py [seconds]"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import sys
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "agents/teabag"))
import time
import mujoco
import numpy as np
import props_teabag as tb
from tea_sim import session, MUG

T = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
t0 = time.time()
s = session(robot_xy=(2.08, 1.88), robot_yaw_deg=90.0, extras=[tb],
            poses={MUG: (1.72, 2.24, 0.86, 0.7071068, 0, 0, 0.7071068)})
m, d = s.m, s.d
print("build", round(time.time() - t0, 1), "s; nq", m.nq, "nv", m.nv, "nbody", m.nbody)
bag = m.body("env/" + tb.BAG).id
tag = m.body("env/" + tb.TAG).id
segs = [m.body(f"env/{tb.NAME}_s{i}").id for i in range(tb.N_SEG)]
prop = {bag, tag, *segs}
print("masses bag", m.body_mass[bag], "tag", m.body_mass[tag], "seg", m.body_mass[segs[0]], "subtree", m.body_subtreemass[bag])
p0 = {b: d.xpos[b].copy() for b in (bag, tag)}
worst = 0.0
t1 = time.time()
for k in range(int(T / 0.1)):
    s.step(60)
    for c in d.contact[:d.ncon]:
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if b1 in prop or b2 in prop:
            if -c.dist > worst:
                worst = -c.dist
                who = (m.geom(c.geom1).name, m.geom(c.geom2).name, round(d.time, 2))
    if k % 5 == 4:
        print(f"t {d.time:.1f} bag moved {np.linalg.norm(d.xpos[bag]-p0[bag])*1000:.2f} mm  tag moved "
              f"{np.linalg.norm(d.xpos[tag]-p0[tag])*1000:.2f} mm  bag z {d.xpos[bag][2]:.4f} tag {np.round(d.xpos[tag],4)}"
              f"  deepest prop contact {worst*1000:.2f} mm", flush=True)
print("deepest", round(worst * 1000, 2), "mm", who if worst > 0 else "")
print("sim", round(d.time, 2), "s in", round(time.time() - t1, 1), "s wall; checks", s.checks.as_dict())
