# Live fixes for the mug take, from what worked live.
# 1. Left hand servoed to its recorded world path from t 6.4 to 11, with a 20 mm pull toward +y from t 7.8 to 9.8: the fingertips landed on the mug rim in PhysX; this takes them behind the mug as in the recording.
# 2. Stance-ankle balance from t 11.5, gains 1.0 / 0.1 (stage A). Stage B starts standing at t 18 and needed no balance loop.
HS["win"][:] = [(6.4, 11.0, HANDS[0])]
HS["bias"] = mkb((0, .02, 0))
BK.update(kp=0. if s.i0 > 0 else 1.0, kd=.1, win=(11.5, 1e9))
