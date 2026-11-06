"""The film as ONE continuous physics run: (1) mug from the rack to its spot, (2) tea bag into the mug,
(3) kettle off its base, pour, kettle down behind the mug, (4) spoon from the tub, stir.
Each step is an agent's skill module (`agents\<name>\skill_*.py`, run(p, s, **kw) -> dict); between steps
the robot walks with the walking network (mugfast's follower) to the stance the next skill needs. Steps
whose skill is not delivered yet are skipped and logged. Recording at 60 fps, saved even on a crash.
usage: flow.py [out.npz] [--until step]"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import atexit
import json
import math
import os
import sys
import time

os.environ.setdefault("TEA_FOOT_TC", "0.007")
W = str(WORK)
for sub in ("", r"\agents\mugfast", r"\agents\teabag", r"\agents\kettle", r"\agents\spoon"):
    sys.path.insert(0, W + sub)

import numpy as np  # noqa: E402

from motion import Program  # noqa: E402
from tea_sim import session  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(W, "flow_rec.npz")
UNTIL = sys.argv[sys.argv.index("--until") + 1] if "--until" in sys.argv else None
SKIP = set(sys.argv[sys.argv.index("--skip") + 1].split(",")) if "--skip" in sys.argv else set()


def optional(name):
    try:
        return __import__(name)
    except ImportError as e:
        print(f"[flow] {name} not available ({e})", flush=True)
        return None


props = [m for m in (optional("props_teabag"), optional("props_spoon")) if m is not None]
skill_mug, skill_teabag = optional("skill_mug"), optional("skill_teabag")
skill_pour, skill_stir = optional("skill_pour"), optional("skill_stir")

t0 = time.time()
s = session(robot_xy=(0.62, 1.15), robot_yaw_deg=90.0, extras=props)
p = Program(s)
atexit.register(lambda: p.rec.save(OUT))
results, marks = {}, []


def mark(name):
    marks.append((name, round(float(s.d.time), 3)))
    print(f"[flow] {name} at {s.d.time:.2f} s", flush=True)


def walk_to(xy, yaw_deg):
    """Walk to the stance a skill needs (the mug skill's walking follower, nothing carried)."""
    w = skill_mug.MugSkill(p, s)
    r = w.drive(np.asarray(xy, float), math.radians(yaw_deg), name=f"walk to {xy}")   # ends standing, Moves rebuilt
    p.mv.on_tick = None                         # the next skill installs its own payload estimate
    return r


STEPS = [("mug", skill_mug, None), ("teabag", skill_teabag, "TEABAG_STANCE"),
         ("pour", skill_pour, "POUR_STANCE"), ("stir", skill_stir, "STIR_STANCE")]
STANCES = {"teabag": (1.90, 1.88, 90.0), "pour": (1.86, 1.90, 90.0), "stir": (1.86, 1.90, 90.0)}   # from the agents
KW = {"mug": {"side_off": 0.0}}                 # step 1 passes on the twin's thick-wall mug with no side offset (f18)
for name, mod, stance_attr in STEPS:
    if mod is None or name in SKIP:
        mark(f"{name} skipped ({'not delivered' if mod is None else 'not passing yet'})")
        continue
    stance = getattr(mod, stance_attr, None) if stance_attr else None
    stance = stance or STANCES.get(name)
    if stance is not None:
        here = np.array(s.d.qpos[:2])
        q = s.d.qpos[3:7]
        heading = math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        if np.linalg.norm(here - np.asarray(stance[:2])) > 0.02 or abs((heading - stance[2] + 180) % 360 - 180) > 4:
            mark(f"walk to {name} stance {stance} (from {np.round(here, 3).tolist()}, {heading:.1f} deg)")
            walk_to(stance[:2], stance[2])
    mark(f"{name} start")
    r = mod.run(p, s, **KW.get(name, {}))
    results[name] = r
    mark(f"{name} end ok={r.get('ok', '?')}")
    if r.get("ok") is False:
        print(f"[flow] {name} failed: {r.get('failed')}; stopping", flush=True)
        break
    if UNTIL == name:
        break

print("checks", s.checks.as_dict())
print(f"sim {s.d.time:.1f} s in {time.time() - t0:.0f} s wall")
json.dump({"marks": marks, "results": results, "checks": s.checks.as_dict()},
          open(OUT.replace(".npz", "_result.json"), "w"), indent=1, default=float)
