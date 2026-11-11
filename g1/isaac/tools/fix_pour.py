# Pour take, stage from t 31.5 with the kettle in the hand. Seen in take pgW2: from t 41 the moving water turned the kettle further in the grip (tilt 104 deg against 91.7 recorded) and the spout moved 5 cm past the mug.
# Fix: while pouring, the hand follows the recording corrected by the measured error of the kettle: position by the spout error, orientation by the tilt error. Hand targets only.
if s.F is not None:
    SPOUT = np.array([-0.1, 0, .17])
    KI_ = R.on.index("kettle")
    KST = dict(dp=np.zeros(3), da=0.)
    KW = (38.5, 44.5)

    def _rm_wxyz(q):
        return C.rmat(np.r_[q[1:], q[0]])

    def kettle_err(tr):
        if WK["c"] % WK["every"] or not KW[0] <= tr <= KW[1]:
            return
        i = int(np.clip(round((tr - R.t[0]) * 60), 0, len(R.t) - 1))
        T = s.obj(C.OBJ["kettle"])
        Rm = C.rmat(T[3:7])
        q = R.obj[i, KI_, 3:7]
        Rr = _rm_wxyz(q)
        e = (R.obj[i, KI_, :3] + Rr @ SPOUT) - (T[:3] + Rm @ SPOUT)
        ti = math.acos(max(-1, min(1, Rm[2, 2])))
        tr_ = math.acos(max(-1, min(1, Rr[2, 2])))
        KST["dp"] += .02 * e
        KST["da"] += .02 * (ti - tr_)
        KST["dp"] = np.clip(KST["dp"], -.08, .08)
        KST["da"] = float(np.clip(KST["da"], -.35, .35))

    def fade(tr):
        return min(1., max(0., (tr - KW[0]) / .4), max(0., (KW[1] - tr) / .4))

    HS["win"][:] = [(KW[0], KW[1], HANDS[1])]
    HS["bias"] = lambda h, tr: KST["dp"] * fade(tr)
    HS["qb"] = lambda h, tr: np.array([0, math.sin(KST["da"] * fade(tr) / 2), 0, math.cos(KST["da"] * fade(tr) / 2)])
    hooks.insert(0, kettle_err)
if s.F is None:
    BK.update(kp=1.0, kd=.1, win=(1.0, 1e9))        # stage A has walks: same stance-ankle balance as the mug
