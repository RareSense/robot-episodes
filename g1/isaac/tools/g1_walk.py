# Live fix for walking: swing-foot world servo. Stance-foot slip in PhysX differs from MuJoCo, so a pure joint replay drifts. While a foot is in the air in the recording, its joint targets get a small
# correction (Jacobian, damped) that lands it where the recording lands it; the correction fades out once it stands. Targets only.
LEGJ = {f: np.array([s.ix[n] for n in s.names if n.startswith(side) and any(k in n for k in ("hip", "knee", "ankle"))]) for f, side in zip(FEET, ("left", "right"))}
FL = {f: s.links.index(f) for f in FEET}
FB = {f: R.bn.index(f) for f in FEET}
OI = s.off.copy() * 0
FO = np.zeros(s.n)
WK = dict(k=.3, lam=.02, dec=.25, ztol=.0375, every=6, on=True, clip=.25, c=0)


def qmulw(a, b):  # x y z w
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz])


def rec_foot(f, tr):
    u = (tr - R.t[0]) * 60
    i = int(np.clip(math.floor(u), 0, len(R.t) - 2)); a = float(np.clip(u - i, 0, 1))
    p = R.xpos[i, FB[f]] * (1 - a) + R.xpos[i + 1, FB[f]] * a
    q = R.xquat[i + 1 if a > .5 else i, FB[f]]
    return p, np.array([q[1], q[2], q[3], q[0]])


def footservo(tr):
    WK["c"] += 1
    if WK["c"] % WK["every"]:
        return
    if not WK["on"]:
        return
    J = None
    L = s.link_poses()
    dt = WK["every"] * C.DT
    for f in FEET:
        p, q = rec_foot(f, tr)
        idx = LEGJ[f]
        if p[2] > WK["ztol"]:
            if J is None:
                J = s.art.get_jacobians()[0]
            T = L[FL[f]]
            e = np.zeros(6)
            e[:3] = p - T[:3]
            qc = T[3:7] * np.array([-1, -1, -1, 1.])
            d = qmulw(q, qc)
            if d[3] < 0:
                d = -d
            e[3:] = 2 * d[:3]
            Jl = J[JROW(FL[f])][:, 6 + idx].astype(float)
            dq = Jl.T @ np.linalg.solve(Jl @ Jl.T + WK["lam"] ** 2 * np.eye(6), e)
            FO[idx] = np.clip(FO[idx] + WK["k"] * dq, -WK["clip"], WK["clip"])
        else:
            FO[idx] *= math.exp(-dt / WK["dec"])
            if WK.get("pel", 0) > 0:
                # stance leg: pelvis servo. To move the pelvis by e with the foot planted, the foot must move by -e relative to the pelvis.
                if J is None:
                    J = s.art.get_jacobians()[0]
                pb = R.bn.index("pelvis")
                u = (tr - R.t[0]) * 60
                i = int(np.clip(round(u), 0, len(R.t) - 1))
                Tp = L[s.links.index("pelvis")]
                qr = R.xquat[i, pb]
                d = qmulw(np.array([qr[1], qr[2], qr[3], qr[0]]), Tp[3:7] * np.array([-1, -1, -1, 1.]))
                if d[3] < 0:
                    d = -d
                e = np.r_[R.xpos[i, pb] - Tp[:3], 2 * d[:3]]
                Jl = J[JROW(FL[f])][:, 6 + idx].astype(float)
                dq = Jl.T @ np.linalg.solve(Jl @ Jl.T + WK["lam"] ** 2 * np.eye(6), -e)
                FO[idx] = np.clip(FO[idx] + WK["pel"] * dq, -WK["clip"], WK["clip"])


ARMJ = {h: np.array([s.ix[n] for n in s.names if n.startswith(side) and any(k in n for k in ("shoulder", "elbow", "wrist"))]) for h, side in zip(HANDS, ("left", "right"))}
HB = {h: R.bn.index(h) for h in HANDS}
HL = {h: s.links.index(h) for h in HANDS}
HS = dict(win=[], k=.2, lam=.03, dec=.3, clip=.3)        # win: (t0, t1, hand link): in that window the hand is servoed to where the recording has it in the world


def rec_hand(h, tr):
    u = (tr - R.t[0]) * 60
    i = int(np.clip(math.floor(u), 0, len(R.t) - 2)); a = float(np.clip(u - i, 0, 1))
    p = R.xpos[i, HB[h]] * (1 - a) + R.xpos[i + 1, HB[h]] * a
    q = R.xquat[i + 1 if a > .5 else i, HB[h]]
    return p, np.array([q[1], q[2], q[3], q[0]])


def handservo(tr):
    if WK["c"] % WK["every"]:
        return
    dt = WK["every"] * C.DT
    L = None
    for h in HANDS:
        idx = ARMJ[h]
        if any(a <= tr <= b and hh == h for a, b, hh in HS["win"]):
            L = s.link_poses() if L is None else L
            J = s.art.get_jacobians()[0]
            p, q = rec_hand(h, tr)
            if HS.get('bias'):
                p = p + HS['bias'](h, tr)
            if HS.get('qb'):
                q = qmulw(HS['qb'](h, tr), q)
            T = L[HL[h]]
            e = np.zeros(6)
            e[:3] = p - T[:3]
            d = qmulw(q, T[3:7] * np.array([-1, -1, -1, 1.]))
            if d[3] < 0:
                d = -d
            e[3:] = 2 * d[:3]
            Jl = J[JROW(HL[h])][:, 6 + idx].astype(float)
            dq = Jl.T @ np.linalg.solve(Jl @ Jl.T + HS["lam"] ** 2 * np.eye(6), e)
            FO[idx] = np.clip(FO[idx] + HS["k"] * dq, -HS["clip"], HS["clip"])
        else:
            FO[idx] *= math.exp(-dt / HS["dec"])


def JROW(l):
    return l - 1 if s.art.get_jacobians().shape[1] == len(s.links) - 1 else l


BAL = np.zeros(s.n)
BK = dict(kp=0., kd=0., win=(0., 1e9), lim=.25)
ANK = {f: (s.ix[side + "_ankle_pitch_joint"], s.ix[side + "_ankle_roll_joint"]) for f, side in zip(FEET, ("left", "right"))}


def balance(tr):
    """Pelvis attitude against the recording, fed to the stance ankles. Targets only."""
    if WK["c"] % WK["every"]:
        return
    BAL[:] = 0
    if BK["kp"] <= 0 or not BK["win"][0] <= tr <= BK["win"][1]:
        return
    i = int(np.clip(round((tr - R.t[0]) * 60), 0, len(R.t) - 1))
    pb = R.bn.index("pelvis")
    T = s.link("pelvis")
    qr = R.xquat[i, pb]
    d = qmulw(np.array([qr[1], qr[2], qr[3], qr[0]]), T[3:7] * np.array([-1, -1, -1, 1.]))
    if d[3] < 0:
        d = -d
    Rm = C.rmat(T[3:7])
    e = Rm.T @ (2 * d[:3])
    w = Rm.T @ s.art.get_root_velocities()[0, 3:6].astype(float)
    for f in FEET:
        p, _ = rec_foot(f, tr)
        if p[2] <= WK["ztol"]:
            a, b = ANK[f]
            BAL[a] = np.clip(-BK["kp"] * e[1] + BK["kd"] * w[1], -BK["lim"], BK["lim"])
            BAL[b] = np.clip(-BK["kp"] * e[0] + BK["kd"] * w[0], -BK["lim"], BK["lim"])


def integ2(tr):
    OI[AX] = np.clip(OI[AX] + KI[0] * C.DT * (s.qd[AX] + FO[AX] + BAL[AX] - s.q()[AX]), -.12, .12)
    s.off[AX] = OI[AX] + FO[AX] + BAL[AX]


hooks[:] = [footservo, handservo, balance, integ2, fgrip]


def mkb(d, t0=7.8, t1=8.2, t2=9.2, t3=9.8):
    d = np.array(d, float)

    def b(h, tr):
        u = 0.
        if t0 < tr < t3:
            u = min(1, (tr - t0) / (t1 - t0), (t3 - tr) / (t3 - t2))
        return d * u
    return b
