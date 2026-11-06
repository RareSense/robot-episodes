"""Load one G1 step and print a summary.
usage: python examples/load_g1.py [data/per_step/step3_pour]"""
import sys
from pathlib import Path

import numpy as np

step = Path(sys.argv[1] if len(sys.argv) > 1 else "data/per_step/step3_pour")
d = dict(np.load(step / "motion.npz"))
joints = [str(n) for n in d["joint_names"]]
print(f"{step.name}: {len(d['t'])} frames, {d['t'][0]:.2f} to {d['t'][-1]:.2f} s, {len(joints)} joints")
print("objects:", [str(n) for n in d["object_names"]])
print("largest torque magnitude:", round(float(d["motor_torque"].max()), 2), "N m")
