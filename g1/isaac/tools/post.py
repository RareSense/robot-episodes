"""Recording -> deliverables. usage: post.py <task> <tag>. Pure numpy: no sim. Writes <task>_rec.npz (30 Hz, Stretch pz_rec keys), shot/ (poses for the renderer), metrics.json"""
import json, math, os, shutil, sys
import numpy as np
task, tag = sys.argv[1:3]
SH = "D:/tea_robot/work/isaac/shots/"
TASKS = {"mug": ("step1_mug", SH + "step1"), "teabag": ("step2_teabag", SH + "tb8h"), "pour": ("step3_pour", "D:/tea_robot/work/agents/kettle/shots/pour35"), "stir": ("step4_stir", SH + "stir34h"), "cup": ("step5_cup", SH + "cupg2h")}
VIEW = {"pour": ((1.40, 2.20, 1.10), (1.72, 2.24, 1.00), 65.0), "stir": ((1.72, 2.40, 1.12), (1.72, 2.22, .94), 70.0)}
d, sh = TASKS[task]
OUT = f"E:/g1_isaac/{task}"
Z = np.load(f"{OUT}/takes/{tag}/rec60.npz", allow_pickle=True)
M = np.load(f"D:/tea_robot/work/dataset/{d}/motion.npz", allow_pickle=True)
S = dict(np.load(sh + "/shot.npz", allow_pickle=True))
names = [str(x) for x in Z["names"]]; links = [str(x) for x in Z["links"]]; objs = [str(x) for x in Z["objects"]]
F = len(Z["t"])
idx = np.clip(np.round((Z["tr"] - M["t"][0]) * 60).astype(int), 0, len(M["t"]) - 1)
e = slice(0, None, 2)      # 60 Hz sampled -> 30 Hz
PROP = Z["prop"] if "prop" in Z.files else np.zeros((F, 0, 7), np.float32)
np.savez_compressed(f"{OUT}/{task}_rec.npz", t=Z["t"][e], tr=Z["tr"][e], names=Z["names"], links=Z["links"], video_frame=np.arange(F)[e], steps_per_sample=20, q=Z["q"][e], dq=Z["dq"][e], qd=Z["qd"][e], vd=Z["vd"][e], root=Z["root"][e], link=Z["link"][e], prop=PROP[e], object_names=Z["objects"], objects=Z["obj"][e], physics="Isaac Sim PhysX, AI-operated")
jn = [str(x) for x in M["joint_names"]]
perm = [jn.index(n) for n in names]
Qr = M["joint_pos"][idx][:, perm]
dq = np.abs(Z["q"] - Qr)
body = np.array(["_hand_" not in n for n in names])
bn = [str(x) for x in S["robot_names"]]


def lp(n):
    return Z["link"][:, links.index(n), :3]


def rp(n):
    return S["xpos"][idx, bn.index(n)]


m = dict(task=task, take=tag, frames=F, duration_s=round(float(Z["t"][-1]), 2), physics="Isaac Sim PhysX, AI-operated")
m["max_joint_diff_rad_body"] = round(float(dq[:, body].max()), 4)
m["joint_worst_body"] = names[int(np.where(body)[0][int(dq[:, body].max(0).argmax())])]
m["max_joint_diff_rad_fingers"] = round(float(dq[:, ~body].max()), 4)
m["rms_joint_diff_rad_body"] = round(float(np.sqrt((dq[:, body] ** 2).mean())), 5)
for n, k in (("left_wrist_yaw_link", "left"), ("right_wrist_yaw_link", "right")):
    hd = np.linalg.norm(lp(n) - rp(n), axis=1) * 1000
    m[f"{k}_hand_diff_mm_max"] = round(float(hd.max()), 1)
    m[f"{k}_hand_diff_mm_mean"] = round(float(hd.mean()), 1)
    m[f"{k}_hand_diff_worst_t"] = round(float(Z["tr"][hd.argmax()]), 2)
bd = np.linalg.norm(lp("pelvis") - rp("pelvis"), axis=1) * 1000
m["pelvis_diff_mm_max"] = round(float(bd.max()), 1)
m["pelvis_diff_mm_end"] = round(float(bd[-1]), 1)
on = [str(x) for x in M["object_names"]]
OM = {"object_0295": "kettle", "object_0309": "milk_powder_jar", "object_0316": "mug"}
for i, o in enumerate(objs):
    r = M["object_pos_quat"][idx, on.index(OM[o]), :3]
    dd = np.linalg.norm(Z["obj"][:, i, :3] - r, axis=1) * 1000
    m[OM[o] + "_diff_mm_max"] = round(float(dd.max()), 1)
    m[OM[o] + "_diff_mm_end"] = round(float(dd[-1]), 1)
    m[OM[o] + "_end"] = Z["obj"][-1, i, :3].round(4).tolist()
    m[OM[o] + "_rec_end"] = r[-1].round(4).tolist()
pk, lim = Z["peak_force"], Z["limits"]
r = pk / lim
m["peak_joint_force_over_limit_top5"] = [(names[i], round(float(r[i]), 3)) for i in np.argsort(-r)[:5]]
m["pelvis_min_z"] = round(float(lp("pelvis")[:, 2].min()), 3)
sd = f"{OUT}/shot"
os.makedirs(sd, exist_ok=True)
I = [links.index(b) for b in bn]
L = Z["link"]
xpos = L[:, I, :3]
xq = L[:, I][:, :, [6, 3, 4, 5]]
om = np.repeat(S["obj_mats"][:1], F, axis=0).copy()
op = [str(x) for x in S["obj_paths"]]


def R_(q):
    x, y, z, w = [q[:, i] for i in range(4)]
    Rm = np.empty((len(q), 3, 3))
    Rm[:, 0, 0] = 1 - 2 * (y * y + z * z); Rm[:, 0, 1] = 2 * (x * y - z * w); Rm[:, 0, 2] = 2 * (x * z + y * w)
    Rm[:, 1, 0] = 2 * (x * y + z * w); Rm[:, 1, 1] = 1 - 2 * (x * x + z * z); Rm[:, 1, 2] = 2 * (y * z - x * w)
    Rm[:, 2, 0] = 2 * (x * z - y * w); Rm[:, 2, 1] = 2 * (y * z + x * w); Rm[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return Rm


for i, o in enumerate(objs):
    j = op.index("/World/Objects/" + o)
    T = Z["obj"][:, i].astype(float)
    om[:, j, :3, :3] = R_(T[:, 3:7]).transpose(0, 2, 1)
    om[:, j, 3, :3] = T[:, :3]


def rowT(p, q):
    A = np.tile(np.eye(4), (len(p), 1, 1))
    A[:, :3, :3] = R_(q).transpose(0, 2, 1)
    A[:, 3, :3] = p
    return A


tk = bn.index("torso_link")
Ti = rowT(L[:, I[tk], :3].astype(float), L[:, I[tk], 3:7].astype(float))
qr = S["xquat"][idx, tk]
Tr = rowT(S["xpos"][idx, tk].astype(float), qr[:, [1, 2, 3, 0]].astype(float))
eye = S["eye_mats"][idx] @ np.linalg.inv(Tr) @ Ti
out = dict(S)
fe, ft, fh = S["film_eye"][idx], S["film_tgt"][idx], S["film_hfov"]
if task in VIEW:
    a, b, h = VIEW[task]
    fe, ft, fh = np.tile(a, (F, 1)), np.tile(b, (F, 1)), np.float64(h)
P = PROP
out.update(t=Z["tr"], xpos=xpos, xquat=xq, obj_mats=om, eye_mats=eye, film_eye=fe, film_tgt=ft, film_hfov=fh, switch_z=S["switch_z"][idx], switch_on=S["switch_on"][idx],
           prop_xpos=P[:, :, :3] if P.size else S["prop_xpos"][idx], prop_xquat=P[:, :, [6, 3, 4, 5]] if P.size else S["prop_xquat"][idx])
np.savez(sd + "/shot.npz", **out)
for f in ("robot_visual.npz", "props_visual.npz"):
    if os.path.exists(sh + "/" + f):
        shutil.copy(sh + "/" + f, sd + "/" + f)
json.dump(m, open(f"{OUT}/metrics.json", "w"), indent=1)
print(json.dumps(m, indent=0))
