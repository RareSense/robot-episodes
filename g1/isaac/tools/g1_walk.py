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


def JROW(l):
    return l - 1 if s.art.get_jacobians().shape[1] == len(s.links) - 1 else l


def integ2(tr):
    OI[AX] = np.clip(OI[AX] + KI[0] * C.DT * (s.qd[AX] + FO[AX] - s.q()[AX]), -.12, .12)
    s.off[AX] = OI[AX] + FO[AX]


hooks[:] = [footservo, integ2, fgrip]
