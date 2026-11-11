def trial(z, t1, pel=0., bkp=1.0, bkd=.1, wk=.3, every=.5, ki=3.0, quiet=False, shots=None):
    """what-if from a snapshot (debug only): restore, set walk parameters, run to t1, report where the pelvis is against the recording"""
    restore(z)
    OI[:] = 0; OI[AX] = np.clip(s.off[AX], -.12, .12); FO[:] = 0; BAL[:] = 0
    WK["pel"] = pel; WK["k"] = wk; BK.update(kp=bkp, kd=bkd); KI[0] = ki
    nxt = s.tr + every
    worst = 0.
    while s.tr < t1:
        advance(6)
        i = int(np.clip(round((s.tr - R.t[0]) * 60), 0, len(R.t) - 1))
        p = s.link("pelvis")
        e = float(np.linalg.norm(p[:3] - R.xpos[i, BN.index("pelvis")]))
        worst = max(worst, e)
        if s.tr >= nxt:
            if not quiet:
                wstat()
            if shots and round(s.tr, 1) in shots:
                wcam("%s_%03d" % (shots[0], round(s.tr * 10)))
            nxt += every
        if p[2] < .6:
            print("FALLEN at", round(s.tr, 2), "worst pelvis err mm", round(worst * 1000))
            return False
    print("OK to", round(s.tr, 2), "worst pelvis err mm", round(worst * 1000), "params", dict(pel=pel, bkp=bkp, bkd=bkd, wk=wk, ki=ki))
    return True
