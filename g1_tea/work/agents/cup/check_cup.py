"""Read-only check of a cup-step recording (forward kinematics of the recorded 60 fps states, nothing simulated):
mug tilt over the whole run, the mug's position in the LEFT hand's frame (slip) from t_grip on, the mug's horizontal
and vertical acceleration while carried (finite differences of recorded positions, 3-frame smoothing for the report
only), pelvis height (fall), and how far every other free object moved.
usage: check_cup.py <rec.npz> t_grip t_lift_done [t_end]"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(WORK))
from tea_sim import MUG, build  # noqa: E402
from nova_robotics.byor.locomotion import WalkingPolicy  # noqa: E402

rec = np.load(sys.argv[1])
t_grip, t_lift = float(sys.argv[2]), float(sys.argv[3])
t_end = float(sys.argv[4]) if len(sys.argv) > 4 else 1e9
spec, _, _ = build()
WalkingPolicy.prepare_spec(spec)
m = spec.compile()
d = mujoco.MjData(m)
mb = m.body("env/" + MUG + "_body").id
hb = m.body("left_wrist_yaw_link").id
robot = {i for i in range(1, m.nbody) if not m.body(i).name.startswith("env/")}
free = [int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j] == 0 and int(m.jnt_bodyid[j]) not in robot and int(m.jnt_bodyid[j]) != mb]
T, Qp = rec["t"], rec["qpos"]
tilt, rel, pos, pel_z, t_used = [], [], [], [], []
first = None
for i, t in enumerate(T):
    if t > t_end:
        break
    d.qpos[:] = Qp[i]
    mujoco.mj_kinematics(m, d)
    if first is None:
        first = {fb: d.xpos[fb].copy() for fb in free}
    tilt.append(np.degrees(np.arccos(np.clip(d.xmat[mb][8], -1, 1))))
    rel.append(d.xmat[hb].reshape(3, 3).T @ (d.xpos[mb] - d.xpos[hb]))
    pos.append(d.xpos[mb].copy())
    pel_z.append(d.qpos[2])
    t_used.append(t)
t_used, tilt, rel, pos = np.array(t_used), np.array(tilt), np.array(rel), np.array(pos)
moved = sorted(((float(np.linalg.norm(d.xpos[fb] - first[fb])) * 1000, m.body(fb).name) for fb in free), reverse=True)[:3]
g = t_used >= t_grip
i0 = int(np.argmax(g))
slip = np.linalg.norm(rel - rel[i0], axis=1) * 1000
car = t_used >= t_lift
dt = np.diff(t_used).mean()
ps = np.array([np.convolve(pos[:, k], np.ones(3) / 3, mode="same") for k in range(3)]).T
acc = np.diff(ps, 2, axis=0) / dt ** 2
acc_c = acc[car[1:-1]][2:-2]
print(f"frames {len(t_used)}  t {t_used[0]:.2f}..{t_used[-1]:.2f}")
print(f"mug tilt max {tilt.max():.1f} deg at t {t_used[np.argmax(tilt)]:.2f}; after grip max {tilt[g].max():.1f}; carried max {tilt[car].max():.1f}")
print(f"slip (mug in left-hand frame) from t {t_grip}: max {slip[g].max():.1f} mm at t {t_used[g][np.argmax(slip[g])]:.2f}; final {slip[-1]:.1f} mm")
print(f"carried mug acceleration: horizontal p99 {np.percentile(np.linalg.norm(acc_c[:, :2], axis=1), 99):.2f} m/s2, "
      f"max {np.linalg.norm(acc_c[:, :2], axis=1).max():.2f}; vertical p99 {np.percentile(np.abs(acc_c[:, 2]), 99):.2f} m/s2")
print(f"mug lowest z while carried {pos[car][:, 2].min():.3f} m; pelvis z min {min(pel_z):.3f} m")
print("other free objects moved (mm):", [(n, round(v, 2)) for v, n in moved])
