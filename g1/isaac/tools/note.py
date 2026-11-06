"""Hand-to-object numbers and the matching frame pair. usage: note.py <task> <tag> <object key in rec60> <hand link> <t_from> <t_to> <pair time> <clip> <film mp4>"""
import json, subprocess, sys
import numpy as np
from PIL import Image
task, tag, ok, hand, a, b, tp, clip, film = sys.argv[1:10]
a, b, tp = float(a), float(b), float(tp)
Z = np.load(f"E:/g1_isaac/{task}/takes/{tag}/rec60.npz", allow_pickle=True)
links = [str(x) for x in Z["links"]]; objs = [str(x) for x in Z["objects"]]
tr = Z["tr"]; m = (tr >= a) & (tr <= b)
O = Z["obj"][m, objs.index(ok)].astype(float)
H = Z["link"][m, links.index(hand)].astype(float)


def rm(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


rel = np.array([rm(h[3:7]).T @ (o[:3] - h[:3]) for o, h in zip(O, H)])
slip = np.linalg.norm(rel - rel[0], axis=1) * 1000
tilt = np.degrees([np.arccos(np.clip(rm(o[3:7])[2, 2], -1, 1)) for o in O])
d = dict(slip_mm_max=round(float(slip.max()), 1), slip_mm_end=round(float(slip[-1]), 1), tilt_deg_max=round(float(tilt.max()), 1), window=[a, b])
f = "C:/Users/raresense/AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe"
for src, out in ((film, "E:/g1_isaac/work/_i.png"), (clip, "E:/g1_isaac/work/_m.png")):
    subprocess.run([f, "-y", "-loglevel", "error", "-ss", str(tp), "-i", src, "-frames:v", "1", "-vf", "scale=960:-1", out], check=True)
A, B = Image.open("E:/g1_isaac/work/_m.png").convert("RGB"), Image.open("E:/g1_isaac/work/_i.png").convert("RGB")
S = Image.new("RGB", (A.width + B.width, A.height))
S.paste(A, (0, 0)); S.paste(B, (A.width, 0))
S.save(f"E:/g1_isaac/{task}/match_pair_mujoco_left_isaac_right.png")
M = json.load(open(f"E:/g1_isaac/{task}/metrics.json"))
M["object_in_hand"] = d
json.dump(M, open(f"E:/g1_isaac/{task}/metrics.json", "w"), indent=1)
print(d)
