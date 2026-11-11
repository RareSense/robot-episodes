"""Motor audit from a take: PD force the drive asks for (kp*(target-q)+kd*(vtarget-dq)) against the MJCF limit. PhysX clamps the drive at the limit, so this counts saturated frames. usage: sat.py <task> <tag>"""
import json, sys
import numpy as np
task, tag = sys.argv[1:3]
SUF = sys.argv[3] if len(sys.argv) > 3 else ""
GAIN = {"hip": (3000, 150), "knee": (3000, 150), "ankle": (3000, 200), "waist": (2500, 120), "shoulder": (600, 15), "elbow": (600, 15), "wrist_roll": (300, 6), "wrist_pitch": (200, 3), "wrist_yaw": (200, 3), "thumb": (40, .6), "middle": (40, .6), "index": (40, .6)}


def key(n):
    for k, v in GAIN.items():
        if k in n:
            return v


Z = np.load(f"E:/g1_isaac/{task}/takes/{tag}/rec60.npz", allow_pickle=True)
n = [str(x) for x in Z["names"]]
kp = np.array([key(x)[0] for x in n]); kd = np.array([key(x)[1] for x in n])
f = np.abs(kp * (Z["qd"] - Z["q"]) + kd * (Z["vd"] - Z["dq"]))
r = f / Z["limits"]
sat = (r >= 1).mean(0)
o = np.argsort(-sat)[:6]
res = dict(frames=len(r), saturated_frame_share={n[i]: round(float(sat[i]), 4) for i in o if sat[i] > 0}, any_joint_saturated_share=round(float((r >= 1).any(1).mean()), 4))
print(json.dumps(res, indent=0))
m = json.load(open(f"E:/g1_isaac/{task}/metrics{SUF}.json"))
m["motor_saturation"] = res
json.dump(m, open(f"E:/g1_isaac/{task}/metrics{SUF}.json", "w"), indent=1)
