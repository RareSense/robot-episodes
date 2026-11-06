"""Read-only: per phase of a cup-step recording (phases from its _result.json): the mug's horizontal acceleration
(p99 / max, from recorded positions at 60 fps with 3-frame smoothing), tilt max, and slip in the LEFT hand's frame
(from the grip). Also the whole carried stretch (lift-off to the end of the recording) and the tilt once lifted.
usage: phase_check.py <rec.npz>"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import json
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(WORK))
from tea_sim import MUG, build  # noqa: E402
from nova_robotics.byor.locomotion import WalkingPolicy  # noqa: E402

rec = np.load(sys.argv[1])
ph = json.load(open(sys.argv[1].replace(".npz", "_result.json")))["phases"]
spec, _, _ = build()
WalkingPolicy.prepare_spec(spec)
m = spec.compile()
d = mujoco.MjData(m)
mb = m.body("env/" + MUG + "_body").id
hb = m.body("left_wrist_yaw_link").id
T = rec["t"]
P, TL, S, Z = [], [], [], []
rel0 = None
for i, t in enumerate(T):
    d.qpos[:] = rec["qpos"][i]
    mujoco.mj_kinematics(m, d)
    R = d.xmat[hb].reshape(3, 3)
    rel = R.T @ (d.xpos[mb] - d.xpos[hb])
    if rel0 is None and t >= ph["lift"] - 0.05:
        rel0 = rel
    S.append(0.0 if rel0 is None else float(np.linalg.norm(rel - rel0)) * 1000)
    P.append(d.xpos[mb].copy())
    TL.append(float(np.degrees(np.arccos(np.clip(d.xmat[mb][8], -1, 1)))))
P, TL, S = np.array(P), np.array(TL), np.array(S)
dt = float(np.diff(T).mean())
ps = np.array([np.convolve(P[:, k], np.ones(3) / 3, mode="same") for k in range(3)]).T
a = np.r_[0.0, np.linalg.norm(np.diff(ps[:, :2], 2, axis=0), axis=1) / dt ** 2, 0.0]
a[-2:] = 0.0                                                         # edge of the smoothing window
z0 = P[0, 2]
lifted = np.nonzero(P[:, 2] > z0 + 0.003)[0]
t_off = T[lifted[0]] if len(lifted) else T[-1]
names = sorted((k for k in ph if k not in ("start", "reach", "grasp")), key=lambda k: ph[k])
for a_, b_ in zip(names, names[1:]):
    s = (T >= ph[a_]) & (T < ph[b_])
    if s.sum() < 3:
        continue
    print(f"{a_:6s} {ph[a_]:6.2f}-{ph[b_]:6.2f}: acc p99 {np.percentile(a[s], 99):5.2f} max {a[s].max():5.2f} m/s2 | "
          f"tilt max {TL[s].max():4.1f} deg | slip {S[s][0]:.1f} -> {S[s][-1]:.1f} (max {S[s].max():.1f}) mm")
c = (T >= t_off) & (T <= T[-1] - 2 * dt)
print(f"lift-off at t {t_off:.2f}: once lifted tilt max {TL[c].max():.1f} deg (at t {T[c][np.argmax(TL[c])]:.2f}); "
      f"carried acc p99 {np.percentile(a[c], 99):.2f} m/s2, max {a[c].max():.2f}; slip max {S.max():.1f} mm")
