"""Package the 5 tea demos (dataset/step*/motion.npz) as one LeRobot dataset (v3.0 format, LeRobot 0.4.x).
Per 60 fps frame: whole-body joint state, base pose, object poses, eye camera pose, |motor torque|, the task text.
action = the next frame's 43 joint positions (joint position targets; the recorder kept torque magnitudes only, so
signed torques cannot be the action). No camera images yet (state-only dataset).
usage: <lerobot venv python> export_lerobot.py [out_dir]   (out_dir must not exist)"""
import json
import sys
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SRC = Path(r"D:\tea_robot\work\dataset")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\tea_robot\work\dataset_lerobot")
REPO_ID = "abdullahanxie/g1_tea_sim_demos"
STEPS = [  # folder, task text
    ("step1_mug", "Take the mug from the rack."),
    ("step2_teabag", "Put the tea bag into the mug."),
    ("step3_pour", "Pour hot water from the kettle into the mug."),
    ("step4_stir", "Stir the tea in the mug with the spoon."),
    ("step5_cup", "Pick up the cup of tea and walk away with it."),
]
OBJECTS = ["kettle", "milk_powder_jar", "mug", "teabag_bag", "spoon"]  # fixed slots; absent objects are zeros
POSE = ["x", "y", "z", "qw", "qx", "qy", "qz"]

first = np.load(SRC / STEPS[0][0] / "motion.npz")
joints = [str(x) for x in first["joint_names"]]
motors = [str(x) for x in first["motor_names"]]
f32 = "float32"
features = {
    "observation.state": {"dtype": f32, "shape": (len(joints),), "names": joints},
    "observation.velocity": {"dtype": f32, "shape": (len(joints),), "names": joints},
    "observation.base_pose": {"dtype": f32, "shape": (7,), "names": POSE},
    "observation.environment_state": {"dtype": f32, "shape": (7 * len(OBJECTS),),
                                      "names": [f"{o}_{k}" for o in OBJECTS for k in POSE]},
    "observation.object_present": {"dtype": f32, "shape": (len(OBJECTS),), "names": OBJECTS},
    "observation.eye_camera_pose": {"dtype": f32, "shape": (12,),
                                    "names": ["x", "y", "z"] + [f"r{i}{j}" for i in range(3) for j in range(3)]},
    "observation.effort_abs": {"dtype": f32, "shape": (len(motors),), "names": motors},
    "action": {"dtype": f32, "shape": (len(joints),), "names": joints},
}

ds = LeRobotDataset.create(REPO_ID, fps=60, root=OUT, robot_type="unitree_g1_with_hands", features=features,
                           use_videos=False)
for folder, task in STEPS:
    r = np.load(SRC / folder / "motion.npz")
    assert [str(x) for x in r["joint_names"]] == joints and [str(x) for x in r["motor_names"]] == motors, folder
    q = r["joint_pos"]
    act = np.vstack([q[1:], q[-1:]])                    # target = where the joints are one frame later
    names = [str(x) for x in r["object_names"]]
    env = np.zeros((len(q), len(OBJECTS), 7))
    present = np.zeros(len(OBJECTS))
    for k, o in enumerate(OBJECTS):
        if o in names:
            env[:, k] = r["object_pos_quat"][:, names.index(o)]
            present[k] = 1.0
    for f in range(len(q)):
        ds.add_frame({
            "observation.state": q[f].astype(f32),
            "observation.velocity": r["joint_vel"][f].astype(f32),
            "observation.base_pose": r["base_pos_quat"][f].astype(f32),
            "observation.environment_state": env[f].reshape(-1).astype(f32),
            "observation.object_present": present.astype(f32),
            "observation.eye_camera_pose": r["eye_cam_pos_rotmat"][f].astype(f32),
            "observation.effort_abs": r["motor_torque"][f].astype(f32),
            "action": act[f].astype(f32),
            "task": task,
        })
    ds.save_episode()
    print(folder, len(q), "frames:", task, flush=True)
ds.finalize()

(OUT / "README.md").write_text(f"""# G1 tea-making sim demos (LeRobot v3.0)

5 whole-body demonstrations of a Unitree G1 (with 3-finger hands) making tea in a digital twin of a real kitchen,
physically simulated in MuJoCo (600 Hz, torque-limited motors), logged at 60 Hz. One episode per sub-task:

| episode | task |
|---|---|
""" + "".join(f"| {i} | {t} |\n" for i, (_, t) in enumerate(STEPS)) + f"""
## Features (per frame)

- `observation.state` - 43 joint positions (rad): legs, waist, arms, fingers
- `observation.velocity` - 43 joint velocities (rad/s), central differences of the 60 Hz positions
- `observation.base_pose` - floating base (pelvis) position (m) and quaternion (w x y z), world frame
- `observation.environment_state` - pose (x y z qw qx qy qz) of {', '.join(OBJECTS)}; zeros when absent
- `observation.object_present` - 1 if that object is in the episode's scene
- `observation.eye_camera_pose` - head camera position + 3x3 rotation (rows = camera axes in world; MuJoCo camera: -z forward, +y up)
- `observation.effort_abs` - |torque| of each of the 43 motors (N m). **Magnitude only** (the recorder kept absolute values)
- `action` - joint position targets: the 43 joint positions of the next frame (last frame repeats)

## Not included

- camera images (no RGB in this version)
- signed motor torques, IMU, contact forces
- variations: one demo per sub-task

Robot model, scene and the code that produced the runs: https://github.com/abdullahanxie/tea-robot-physics
""", encoding="utf-8")
print("wrote", OUT)
