# Tea bag take, stage from t 13 with the bag already pinched (the pick from the tray was not reached).
# Live findings: recorded torque x1.5 held the bag to t 21, then it fell as the recorded torque dropped to about the joint friction. x2.5 squeezed the bag out at the start.
# Fix: right-hand fingers keep the torque of t 14 x1.5 until the recorded release.
GF[0] = 1.5
TH = dict(t0=13.0, t1=float(os.environ.get("TB_REL", "26.9")), g=float(os.environ.get("TB_G", "1.5")))
_i14 = int(round((14.0 - R.t[0]) * 60))
_rf = np.array([k for k, n in enumerate(FI) if s.names[n].startswith("right_hand")])
_hold = R.tau[_i14][s.perm][FI][_rf]


def tb_hold(tr):
    if TH["t0"] <= tr <= TH["t1"]:
        cur = FS[_rf] * s.off[FI[_rf]] * s.kp[FI[_rf]]
        s.off[FI[_rf]] = FS[_rf] * np.maximum(cur, TH["g"] * _hold) / s.kp[FI[_rf]]


hooks.append(tb_hold)
