"""Full recording for the Stretch final take. Read only: it wraps s.step and reads state, it never writes to the sim. 30 Hz of sim time, saved to E: every 60 sim s so a crash keeps what ran."""
import os

import numpy as np

R = {}


def on(s, tag, every=20, save_every=36000):
    d = "E:/stretch_assets/data"
    os.makedirs(d, exist_ok=True)
    R.update(path=f"{d}/{tag}_rec.npz", s=s, every=every, save_every=save_every, c=0, t=[], q=[], dq=[], qd=[], vd=[], root=[], link=[], oven=[], plate=[], pizza=[], body=[], heat=[], vid=[])
    R["names"] = np.array(s.names)
    R["links"] = np.array(list(s.art.shared_metatype.link_names))
    import pizza_kitchen as Z
    import pz
    R["Z"], R["pz"] = Z, pz
    orig = s.step

    def step():
        orig()
        R["c"] += 1
        if R["c"] % R["every"] == 0:
            sample()
        if R["c"] % R["save_every"] == 0:
            save()

    s.step = step


def sample():
    s, Z = R["s"], R["Z"]
    R["t"].append(s.t)
    R["q"].append(s.art.get_dof_positions()[0].astype(np.float32))
    R["dq"].append(s.art.get_dof_velocities()[0].astype(np.float32))
    R["qd"].append(np.array(s.qd, np.float32))
    R["vd"].append(np.array(s.vd, np.float32))
    R["root"].append(s.art.get_root_transforms()[0].astype(np.float32))
    R["link"].append(s.art.get_link_transforms()[0].astype(np.float32))
    R["oven"].append(s.oven.get_dof_positions()[0].astype(np.float32))
    R["plate"].append(s.obj(Z.PLATE).astype(np.float32))
    R["pizza"].append(s.obj(Z.PIZZA).astype(np.float32))
    R["body"].append(s.obj(Z.BODY).astype(np.float32))
    R["heat"].append((R["pz"].HEAT["k"], R["pz"].HEAT["on"]))
    R["vid"].append(R["pz"].VID.get("n", -1))


def save():
    tmp = R["path"] + ".tmp.npz"
    k = ("q", "dq", "qd", "vd", "root", "link", "oven", "plate", "pizza", "body", "heat")
    np.savez_compressed(tmp, t=np.array(R["t"]), names=R["names"], links=R["links"], video_frame=np.array(R["vid"]), steps_per_sample=R["every"], **{x: np.array(R[x]) for x in k})
    os.replace(tmp, R["path"])
    print("REC saved", len(R["t"]), "samples to t", round(R["t"][-1], 2) if R["t"] else None, flush=True)
