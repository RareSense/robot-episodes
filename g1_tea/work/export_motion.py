"""Export a physics recording as motion data for engineers: per 60 fps frame the robot's floating base and every
joint (legs, waist, arms, fingers), joint velocities, the torque of every motor (with its limit), the pose of
every handled object, the robot eye camera pose; CSV + NPZ + metadata, plus the MuJoCo model it ran on.
Velocities are central differences of the recorded 60 fps positions (the exact 600 Hz velocities, IMU and contact
forces need the extended recorder and a re-run - see metadata 'not_included').
usage: export_motion.py <recording.npz> <out_dir> [--extras a.py,b.py] [--name step]"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(WORK))
from tea_sim import DT, G1_XML, JAR, KETTLE, MUG, build  # noqa: E402
from nova_robotics.byor.locomotion import WalkingPolicy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("rec")
ap.add_argument("out")
ap.add_argument("--extras", default="")
ap.add_argument("--name", default="")
a = ap.parse_args()


def load_module(path):
    sp = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(sp)
    sys.path.insert(0, str(Path(path).parent))
    sp.loader.exec_module(mod)
    return mod


extras = [load_module(x) for x in a.extras.split(",") if x]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
r = np.load(a.rec, allow_pickle=True)
spec, _, _ = build(extras=extras)
WalkingPolicy.prepare_spec(spec)
m = spec.compile()
d = mujoco.MjData(m)
assert r["qpos"].shape[1] == m.nq, (r["qpos"].shape, m.nq)
t = np.asarray(r["t"], float)
Q = np.asarray(r["qpos"], float)
F = len(t)
fps = round(1.0 / float(np.median(np.diff(t))))
jn = lambda j: m.joint(j).name  # noqa: E731
robot_j = [j for j in range(m.njnt) if not jn(j).startswith("env/")]
free_robot = [j for j in robot_j if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
hinge_robot = [j for j in robot_j if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
obj_j = [j for j in range(m.njnt) if jn(j).startswith("env/")]

# robot base and joints
qa = m.jnt_qposadr
base = Q[:, qa[free_robot[0]]:qa[free_robot[0]] + 7]
joints = Q[:, [qa[j] for j in hinge_robot]]
jvel = np.gradient(joints, t, axis=0)
jnames = [jn(j) for j in hinge_robot]
# motors: recorded applied torques (60 fps) and limits
mnames = [str(x) for x in r["motor_names"]] if "motor_names" in r.files else []
tau = np.asarray(r["tau"], float) if "tau" in r.files else np.zeros((F, 0))
limits = np.asarray(r["limits"], float) if "limits" in r.files else np.zeros(0)
# objects: every free body in the scene (mug, kettle, jar, props) and other env joints (e.g. the kettle switch)
objs, obj_pose, other = [], [], {}
nice = {KETTLE: "kettle", MUG: "mug", JAR: "milk_powder_jar"}


def readable(name):
    name = name.replace("env/", "").replace("_body", "")
    for k, v in nice.items():
        name = name.replace(k, v)
    return name.replace("prop_", "")


for j in obj_j:
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        objs.append(readable(m.body(m.jnt_bodyid[j]).name))
        obj_pose.append(Q[:, qa[j]:qa[j] + 7])
    elif m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        other[readable(jn(j))] = Q[:, qa[j]]
obj_pose = np.stack(obj_pose, axis=1) if obj_pose else np.zeros((F, 0, 7))
# robot eye camera pose (forward kinematics of the recorded state)
eye = m.camera("eye").id
cam = np.zeros((F, 12))
for f in range(F):
    d.qpos[:] = Q[f]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    cam[f, :3], cam[f, 3:] = d.cam_xpos[eye], d.cam_xmat[eye]

np.savez_compressed(out / "motion.npz", t=t, base_pos_quat=base, joint_names=np.array(jnames), joint_pos=joints,
                    joint_vel=jvel, motor_names=np.array(mnames), motor_torque=tau, motor_limit=limits,
                    object_names=np.array(objs), object_pos_quat=obj_pose,
                    other_joint_names=np.array(list(other)), other_joint_pos=np.array(list(other.values())).T if other else np.zeros((F, 0)),
                    eye_cam_pos_rotmat=cam)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


write_csv(out / "base.csv", ["t", "x", "y", "z", "qw", "qx", "qy", "qz"], np.c_[t, base].round(6))
write_csv(out / "joint_positions.csv", ["t"] + jnames, np.c_[t, joints].round(6))
write_csv(out / "joint_velocities.csv", ["t"] + jnames, np.c_[t, jvel].round(5))
if mnames:
    write_csv(out / "motor_torques.csv", ["t"] + mnames, np.c_[t, tau].round(4))
hdr = ["t"] + [f"{o}_{k}" for o in objs for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
write_csv(out / "objects.csv", hdr, np.c_[t, obj_pose.reshape(F, -1)].round(6))
write_csv(out / "eye_camera.csv", ["t", "x", "y", "z"] + [f"R{i}{j}" for i in range(3) for j in range(3)], np.c_[t, cam].round(6))
shutil.copy2(G1_XML, out / "g1_with_hands.xml")
meta = dict(step=a.name or Path(a.rec).stem, recording=str(Path(a.rec).resolve()), frames=F, fps=fps,
            physics="MuJoCo 600 Hz (dt %.6f s), G1 torque-limited motors, byor whole-body controller / walking network" % DT,
            units=dict(length="m", angle="rad", torque="N m", time="s", quat="w x y z"),
            joints=jnames, motors=mnames, motor_limits_Nm=dict(zip(mnames, [float(x) for x in limits])),
            objects=objs, other_joints=list(other), eye_camera="d435 mount on torso_link; rows of R are world axes of the camera (MuJoCo: -z forward, +y up)",
            robot_model="g1_with_hands.xml (copied); the scene is the Nova twin handoff/project/tea_twin/scene_v3",
            joint_velocities="central differences of the 60 fps positions",
            not_included=["exact 600 Hz joint velocities", "IMU (gyro/accelerometer)", "foot and hand contact forces / wrist F/T",
                          "eye camera depth/segmentation images (RGB video is in isaac/renders)"])
json.dump(meta, open(out / "metadata.json", "w"), indent=1)
print("exported", F, "frames ->", out)
