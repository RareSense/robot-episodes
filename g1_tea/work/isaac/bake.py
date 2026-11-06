"""Bake a physics recording for the Isaac Sim render: the G1's visual meshes, and per 60 fps frame the
world pose of every robot body, of every twin object the physics moved or the corrections placed, the
robot's eye camera (d435 mount on torso_link) and a film camera.

The recorded frames are the simulation states saved every 10 physics steps. Here they only go through
forward kinematics (body poses from joint angles); nothing is stepped, edited or interpolated.
usage: bake.py <recording.npz> <out_dir> [--follow az,el,dist]"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(WORK))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tea_sim import EYE_FOVY, SCENE, SWITCH, SWITCH_LEVER, SWITCH_ON, build  # noqa: E402
from nova_robotics.byor.locomotion import WalkingPolicy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("rec")
ap.add_argument("out")
ap.add_argument("--follow", default="180,-18,1.9",
                help="film camera azimuth from the robot's heading (deg, 180 = behind), elevation (deg), distance (m)")
ap.add_argument("--fixed", default="", help="film camera standing still: 'ex,ey,ez:tx,ty,tz' (eye, aim point)")
ap.add_argument("--world", action="store_true", help="--follow azimuth is a world direction (deg from +x), not from the heading")
ap.add_argument("--hfov", type=float, default=55.0)
ap.add_argument("--headcam", default="", help="'height,pitch_deg': eye camera raised to the person's head-cam height and "
                                               "tilted to their pitch (rides on the robot's own eye motion)")
ap.add_argument("--extras", default="", help="prop modules the recording was made with (comma-separated .py paths)")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def load_module(path):
    import importlib.util
    sp = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(sp)
    sys.path.insert(0, str(Path(path).parent))
    sp.loader.exec_module(mod)
    return mod


extras = [load_module(x) for x in a.extras.split(",") if x]
r = np.load(a.rec)
spec, report, _ = build(extras=extras)
WalkingPolicy.prepare_spec(spec)
m = spec.compile()
d = mujoco.MjData(m)
assert r["qpos"].shape[1] == m.nq, (r["qpos"].shape, m.nq)
bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""  # noqa: E731

# ---------------------------------------------------------------- robot visual meshes (body frame)
rows, data = [], {}
for g in range(m.ngeom):
    b = int(m.geom_bodyid[g])
    if b == 0 or bname(b).startswith("env/"):
        continue
    if m.geom_contype[g] or m.geom_conaffinity[g] or m.geom_group[g] > 2 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    k = int(m.geom_dataid[g])
    v0, nv, f0, nf = m.mesh_vertadr[k], m.mesh_vertnum[k], m.mesh_faceadr[k], m.mesh_facenum[k]
    key = len(rows)
    rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
    rows.append((bname(b), key, *m.geom_pos[g], *m.geom_quat[g], *rgba))
    data[f"v{key}"] = np.asarray(m.mesh_vert[v0:v0 + nv], np.float32)
    data[f"f{key}"] = np.asarray(m.mesh_face[f0:f0 + nf], np.int32)
    data.setdefault("_names", []).append(m.mesh(k).name)
mesh_names = np.array(data.pop("_names"))
np.savez(out / "robot_visual.npz", bodies=np.array([x[0] for x in rows]), mesh=np.array([x[1] for x in rows]),
         pose=np.array([x[2:9] for x in rows]), rgba=np.array([x[9:] for x in rows]), mesh_names=mesh_names, **data)
robot_bodies = sorted({x[0] for x in rows}, key=lambda n: m.body(n).id)
rb = [m.body(n).id for n in robot_bodies]
print("robot visual geoms", len(rows), "bodies", len(rb))


# ---------------------------------------------------------------- props added to the scene (tea bag, spoon, ...)
def tessellate(g, nu=24, nv=12):
    """Triangle mesh (geom frame) of a geom: meshes as compiled, primitives tessellated."""
    t, s = int(m.geom_type[g]), m.geom_size[g]
    T = mujoco.mjtGeom
    if t == T.mjGEOM_MESH:
        k = int(m.geom_dataid[g])
        return (m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]],
                m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]])
    if t == T.mjGEOM_BOX:
        V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * s[:3]
        F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        return V, F
    if t in (T.mjGEOM_SPHERE, T.mjGEOM_ELLIPSOID, T.mjGEOM_CAPSULE):
        rad = np.array([s[0], s[0], s[0]]) if t != T.mjGEOM_ELLIPSOID else s[:3]
        half = s[1] if t == T.mjGEOM_CAPSULE else 0.0
        V = []
        for i in range(nv + 1):
            th = math.pi * i / nv
            dz = half if i <= nv // 2 else -half
            for j in range(nu):
                ph = 2 * math.pi * j / nu
                V.append([rad[0] * math.sin(th) * math.cos(ph), rad[1] * math.sin(th) * math.sin(ph),
                          rad[2] * math.cos(th) + dz])
        F = []
        for i in range(nv):
            for j in range(nu):
                a0, a1 = i * nu + j, i * nu + (j + 1) % nu
                b0, b1 = a0 + nu, a1 + nu
                F += [[a0, b0, b1], [a0, b1, a1]]
        return np.array(V), np.array(F)
    if t == T.mjGEOM_CYLINDER:
        ring = [[s[0] * math.cos(2 * math.pi * j / nu), s[0] * math.sin(2 * math.pi * j / nu)] for j in range(nu)]
        V = [[x, y, -s[1]] for x, y in ring] + [[x, y, s[1]] for x, y in ring] + [[0, 0, -s[1]], [0, 0, s[1]]]
        F = []
        for j in range(nu):
            j1 = (j + 1) % nu
            F += [[j, j1, nu + j1], [j, nu + j1, nu + j], [2 * nu, j1, j], [2 * nu + 1, nu + j, nu + j1]]
        return np.array(V), np.array(F)
    return None


prop_b = [b for b in range(m.nbody) if bname(b).startswith("env/prop_")]
prows, pdata = [], {}
for b in prop_b:
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
    vis = [g for g in gs if m.geom_rgba[g][3] > 0.01 and m.geom_group[g] <= 2] or gs   # visual geoms, else collision
    for g in vis:
        tm = tessellate(g)
        if tm is None:
            continue
        key = len(prows)
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        prows.append((bname(b), key, *m.geom_pos[g], *m.geom_quat[g], *rgba))
        pdata[f"v{key}"] = np.asarray(tm[0], np.float32)
        pdata[f"f{key}"] = np.asarray(tm[1], np.int32)
if prows:
    np.savez(out / "props_visual.npz", bodies=np.array([x[0] for x in prows]), mesh=np.array([x[1] for x in prows]),
             pose=np.array([x[2:9] for x in prows]), rgba=np.array([x[9:] for x in prows]), **pdata)
prop_names = sorted({x[0] for x in prows}, key=lambda n: m.body(n).id)
pb = [m.body(n).id for n in prop_names]
print("prop visual geoms", len(prows), "bodies", len(pb))

# ---------------------------------------------------------------- twin objects whose pose differs from the USD
authored = {}
for b in ET.parse(SCENE / "environment.xml").getroot().iter("body"):
    if b.get("name", "").startswith("object_"):
        q = np.fromstring(b.get("quat") or "1 0 0 0", sep=" ")
        authored[b.get("name")] = (np.fromstring(b.get("pos"), sep=" "), q / np.linalg.norm(q))
F = len(r["qpos"])
env_b = [b for b in range(m.nbody) if bname(b).startswith("env/object_") and bname(b)[4:] in authored]
P = np.zeros((F, m.nbody, 3))
Q = np.zeros((F, m.nbody, 4))
eye_id = m.camera("eye").id
torso = m.body("torso_link").id
CP = np.zeros((F, 3))
CR = np.zeros((F, 3, 3))
for f, q in enumerate(r["qpos"]):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    P[f], Q[f] = d.xpos, d.xquat
    CP[f], CR[f] = d.cam_xpos[eye_id], d.cam_xmat[eye_id].reshape(3, 3)


def placed(b):
    p0, q0 = authored[bname(b)[4:]]
    moved = np.abs(P[:, b] - p0).max() > 5e-4
    turned = (1 - np.abs(Q[:, b] @ q0)).max() > 1e-6
    return moved or turned


objs = [b for b in env_b if placed(b)]


def usd_matrix(p, q):
    """USD row-vector matrix: rows are the frame's axes, the last row its position."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    M = np.eye(4)
    M[:3, :3] = R.reshape(3, 3).T
    M[3, :3] = p
    return M


obj_paths = ["/World/Objects/" + bname(b)[4:].removesuffix("_body") for b in objs]
obj_mats = np.array([[usd_matrix(P[f, b], Q[f, b]) for b in objs] for f in range(F)])
print("objects posed from physics:", [n.split("/")[-1] for n in obj_paths])

# ---------------------------------------------------------------- cameras
# eye: MuJoCo cameras look down their -z with +y up, as USD cameras do; the frame maps over directly
if a.headcam:
    # head-cam match: the person's camera is ~1.62 m up looking ~67 deg down (median over their video,
    # project/headpath/headpath_twin_final.npz); the G1's eye is ~1.25 m up at 47.6 deg. Same motion as the
    # robot's eye (its sway and turns), raised to their height and tilted further down about its right axis.
    from tea_sim import EYE_PITCH
    h, pitch = (float(x) for x in a.headcam.split(","))
    dth = -(math.radians(pitch) - EYE_PITCH)
    Rx = np.array([[1, 0, 0], [0, math.cos(dth), -math.sin(dth)], [0, math.sin(dth), math.cos(dth)]])
    CR = CR @ Rx
    CP = CP + np.array([0.0, 0.0, h - float(np.median(CP[:, 2]))])
    fwd = -CR[:, :, 2] * np.array([1.0, 1.0, 0.0])                   # forehead: 8 cm ahead of the head's centre
    CP = CP + 0.08 * fwd / np.linalg.norm(fwd, axis=1, keepdims=True)
eye_mats = np.zeros((F, 4, 4))
eye_mats[:, 3, 3] = 1
eye_mats[:, :3, :3] = np.transpose(CR, (0, 2, 1))
eye_mats[:, 3, :3] = CP
# film camera: an operator keeping the robot's working area in frame, at fixed azimuth/elevation/distance
# (a render camera, not part of the robot: its aim eases after the torso, like the MuJoCo previews)
az, el, dist = (float(x) for x in a.follow.split(","))
look, tgts, eyes = None, [], []
for f in range(F):
    yaw = math.atan2(2 * (Q[f, 1, 0] * Q[f, 1, 3] + Q[f, 1, 1] * Q[f, 1, 2]), 1 - 2 * (Q[f, 1, 2] ** 2 + Q[f, 1, 3] ** 2))
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    tgt = P[f, torso] + 0.25 * fwd + np.array([0, 0, -0.15])
    look = tgt if look is None else look + 0.06 * (tgt - look)
    ang = math.radians(az) if a.world else yaw + math.radians(az)
    off = dist * np.array([math.cos(math.radians(el)) * math.cos(ang), math.cos(math.radians(el)) * math.sin(ang),
                           -math.sin(math.radians(el))])
    tgts.append(look.copy())
    eyes.append(look + off)
if a.fixed:                                      # the twin's walls are close here: a placed camera instead
    e, t = ([float(x) for x in part.split(",")] for part in a.fixed.split(":"))
    eyes, tgts = [e] * F, [t] * F
# the kettle's on/off lever: its travel on the slide joint, and on/off (the blue light) from it
sj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "env/" + SWITCH)
switch_z = r["qpos"][:, m.jnt_qposadr[sj]] if sj >= 0 else np.zeros(F)
np.savez(out / "shot.npz", fps=60, t=r["t"], robot_names=np.array(robot_bodies), xpos=P[:, rb], xquat=Q[:, rb],
         obj_paths=np.array(obj_paths), obj_mats=obj_mats, eye_mats=eye_mats, eye_fovy=EYE_FOVY,
         film_eye=np.array(eyes), film_tgt=np.array(tgts), film_hfov=a.hfov,
         switch_z=switch_z, switch_on=switch_z < SWITCH_ON, lever=np.array(SWITCH_LEVER, float),
         prop_names=np.array([n.replace("env/", "") for n in prop_names]), prop_xpos=P[:, pb], prop_xquat=Q[:, pb])
print("frames", F, f"({F / 60:.1f} s) ->", out)
json.dump({"rec": str(Path(a.rec).resolve()), "frames": F, "objects": obj_paths}, open(out / "bake.json", "w"), indent=1)
