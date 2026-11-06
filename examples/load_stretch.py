"""Load the Stretch pizza recording and print a summary.
usage: python examples/load_stretch.py [stretch_pizza/data/pz_final03_rec.npz]"""
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "stretch_pizza/data/pz_final03_rec.npz"
d = dict(np.load(path))                     # one read: indexing an npz directly decompresses the array each time
names = [str(n) for n in d["names"]]
t, q = d["t"], d["q"]
print(f"{len(t)} samples, {t[0]:.2f} to {t[-1]:.2f} s, {len(names)} joints, {len(d['links'])} links")
lift = names.index("joint_lift")
print(f"lift range {q[:, lift].min():.3f} to {q[:, lift].max():.3f} m")
print(f"oven door range {np.degrees(d['oven'][:, 0]).min():.1f} to {np.degrees(d['oven'][:, 0]).max():.1f} deg")
print("plate start", np.round(d["plate"][0, :3], 3), "end", np.round(d["plate"][-1, :3], 3))
