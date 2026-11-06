# helpers inside the live session (exec'd; reload with exec(open(...).read()))
import math, time, json
R = s.rec
BN = R.bn
LI = [s.links.index(b) for b in BN]
HANDS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
FR = dict(t=[], tr=[], q=[], dq=[], qd=[], vd=[], root=[], link=[], obj=[], prop=[])
OBJK = [C.OBJ[n] for n in R.on if n in C.OBJ]
REC_ON = [True]
hooks = []            # callables run every step with (tr): live fixes


AX = np.array([i for i, n in enumerate(s.names) if "_hand_" not in n])
KI = [3.0]


def integ(tr):
    """joint-space integral: takes out the sag a stiff PD leaves under load. Motor limits stay in force."""
    s.off[AX] = np.clip(s.off[AX] + KI[0] * C.DT * (s.qd[AX] - s.q()[AX]), -.12, .12)


hooks.append(integ)
# fingers: the recording holds the positions where contact stopped them, so a replay alone squeezes nothing. The recorded motor torque goes back in as a closing offset.
FI, FS = [], []
for i_, n_ in enumerate(s.names):
    if "_hand_" in n_ and "thumb_0" not in n_:
        left = n_.startswith("left")
        cl = (1 if left else -1) if "thumb" in n_ else (-1 if left else 1)
        FI.append(i_); FS.append(cl)
FI, FS = np.array(FI), np.array(FS, float)
GF = [1.0]


def fgrip(tr):
    i = int(np.clip(round((tr - R.t[0]) * 60), 0, len(R.t) - 1))
    s.off[FI] = FS * GF[0] * R.tau[i][s.perm][FI] / s.kp[FI]


hooks.append(fgrip)


def sample():
    FR["t"].append(s.t); FR["tr"].append(s.tr)
    FR["q"].append(s.q().astype(np.float32)); FR["dq"].append(s.v().astype(np.float32))
    FR["qd"].append((s.qd + s.off).astype(np.float32)); FR["vd"].append(s.vd.astype(np.float32))
    FR["root"].append(s.art.get_root_transforms()[0].astype(np.float32))
    FR["link"].append(s.link_poses().astype(np.float32))
    FR["obj"].append(np.array([s.obj(k) for k in OBJK], np.float32))
    FR["prop"].append(s.props().astype(np.float32))
    FR["prop"].append(s.props().astype(np.float32))


def advance(n=1, speed=1.0):
    """n physics steps following the recording"""
    for _ in range(n):
        qr, dr, i = R.at(s.tr)
        s.qd, s.vd = qr[s.perm], dr[s.perm] * speed
        for h in hooks:
            h(s.tr)
        s.step()
        s.tr += C.DT * speed
        if REC_ON[0] and round(s.t * 600) % 10 == 0:
            sample()


def hold(secs):
    """stay on the current targets, recording time stopped"""
    for _ in range(int(secs * 600)):
        s.step()


def status(tag=""):
    i = int(np.clip(round((s.tr - R.t[0]) * 60), 0, len(R.t) - 1))
    L = s.link_poses()
    rt = L[s.links.index("pelvis")]
    r, p, y = C.rpy(rt[3:7])
    b = R.base[i]
    qe = np.abs(s.q() - R.q[i][s.perm]); w = int(np.argmax(qe))
    hd = [float(np.linalg.norm(L[s.links.index(h), :3] - R.xpos[i, BN.index(h)])) for h in HANDS]
    ft = [(np.round(L[s.links.index(f), :3], 3).tolist()) for f in FEET]
    o = {k: s.obj(k)[:3].round(3).tolist() for k in OBJK}
    ro = {R.on[j]: R.obj[i, j, :3].round(3).tolist() for j in range(len(R.on)) if R.on[j] in C.OBJ}
    qa = qe[AX]; wa = AX[int(np.argmax(qa))]
    fp = [round(math.degrees(C.rpy(L[s.links.index(f), 3:7])[1]), 1) for f in FEET]
    print(tag, f"tr {s.tr:.2f} pelvis {rt[:3].round(3).tolist()} rec {b[:3].round(3).tolist()} rpy {round(math.degrees(r),1)} {round(math.degrees(p),1)} {round(math.degrees(y),1)} qerr body {qa.max():.3f} {s.names[wa]} hand {qe[w]:.3f} {s.names[w]} handerr mm {hd[0]*1000:.0f} {hd[1]*1000:.0f} feet {ft} footpitch {fp}")
    print("   obj", o, "rec", ro)
    if s.prop_view is not None:
        Pp = s.props()
        print("   prop0", Pp[0, :3].round(3).tolist(), "rec", np.array(R.prop_xpos[i, 0]).round(3).tolist(), "last", Pp[-1, :3].round(3).tolist(), "rec", np.array(R.prop_xpos[i, -1]).round(3).tolist())
    if s.prop_view is not None:
        Pp = s.props()
        print("   prop0", Pp[0, :3].round(3).tolist(), "rec", np.array(R.prop_xpos[i, 0]).round(3).tolist(), "last", Pp[-1, :3].round(3).tolist(), "rec", np.array(R.prop_xpos[i, -1]).round(3).tolist())


def run_to(t, every=2.0, stop_fall=True):
    nxt = s.tr + every
    t0 = time.time(); s0 = s.t
    while s.tr < t:
        advance(6)
        if s.tr >= nxt:
            status(); nxt += every
        if stop_fall and s.link("pelvis")[2] < .55:
            print("FALLEN"); break
    status("END")
    print("wall", round(time.time() - t0, 1), "sim", round(s.t - s0, 2))


def still(name, cam="film", eye=None, tgt=None, hfov=None, reqdir="E:/g1_isaac/live/render"):
    """ask the render session for a frame of the state now"""
    os.makedirs(reqdir, exist_ok=True)
    i = int(np.clip(round((s.tr - R.t[0]) * 60), 0, len(R.t) - 1))
    L = s.link_poses()
    xpos = L[LI, :3]; xq = L[LI][:, [6, 3, 4, 5]]
    M = np.array(R.obj_mats[0])
    for j, path in enumerate(R.obj_paths):
        k = path.rsplit("/", 1)[-1]
        if k in OBJK:
            T = s.obj(k)
            M[j] = np.eye(4); M[j, :3, :3] = C.rmat(T[3:7]).T; M[j, 3, :3] = T[:3]
    e = R.film_eye[i] if eye is None else eye
    g = R.film_tgt[i] if tgt is None else tgt
    tmp = f"{reqdir}/{name}.tmp.npz"
    P = s.props()
    np.savez(tmp, xpos=xpos, xquat=xq, obj=M, eye=np.array(e, float), tgt=np.array(g, float), hfov=float(hfov or R.film_hfov), prop_xpos=P[:, :3], prop_xquat=P[:, [6, 3, 4, 5]])
    os.replace(tmp, f"{reqdir}/{name}.req.npz")
    print("req", name, "rec frame", i)


# state snapshot for live what-ifs (debug only, never inside a delivered take)
from pxr import UsdPhysics as _UP
ALLP = [v.GetPath().pathString for k, v in s.objs.items() if v.HasAPI(_UP.RigidBodyAPI) and "object_0294" < k < "object_0322"]
ALLV = s.sv.create_rigid_body_view(ALLP)
print("snapshot bodies", len(ALLP))


def snap():
    return dict(allb=(ALLV.get_transforms().copy(), ALLV.get_velocities().copy()), q=s.q(), v=s.v(), root=s.art.get_root_transforms().copy(), rv=s.art.get_root_velocities().copy(), tr=s.tr, t=s.t, off=s.off.copy(), o={k: (s.body(k).get_transforms().copy(), s.body(k).get_velocities().copy()) for k in OBJK}, nf=len(FR["t"]), pv=(s.prop_view.get_transforms().copy(), s.prop_view.get_velocities().copy()) if s.prop_view is not None else None)


def restore(z):
    a = s.art
    a.set_root_transforms(z["root"], s.idx); a.set_root_velocities(z["rv"], s.idx)
    a.set_dof_positions(s.f(z["q"]), s.idx); a.set_dof_velocities(s.f(z["v"]), s.idx)
    ixa = np.arange(ALLV.count, dtype=np.uint32)
    ALLV.set_transforms(z["allb"][0], ixa)
    ALLV.set_velocities(z["allb"][1], ixa)
    for k, (T, V) in z["o"].items():
        b = s.body(k); ix = np.arange(1, dtype=np.uint32)
        b.set_transforms(T, ix); b.set_velocities(V, ix)
    if z["pv"] is not None:
        ix = np.arange(s.prop_view.count, dtype=np.uint32)
        s.prop_view.set_transforms(z["pv"][0], ix); s.prop_view.set_velocities(z["pv"][1], ix)
    if z["pv"] is not None:
        ix = np.arange(s.prop_view.count, dtype=np.uint32)
        s.prop_view.set_transforms(z["pv"][0], ix)
        s.prop_view.set_velocities(z["pv"][1], ix)
    s.tr, s.t = z["tr"], z["t"]; s.off = z["off"].copy()
    for kk in FR:
        del FR[kk][z["nf"]:]


Z0 = snap()
