# Stir take, stage from t 15.5 with the spoon in the hand (the pick from the tub does not hold in PhysX yet).
# Live finding: at the recorded finger torque the spoon dropped straight through the grip at the start; x1.5 held and tracked it to 4 mm through the stir and release.
GF[0] = 1.5
# Release: with no correction the handle falls 34 mm along the rim from where the recording leaves it. It falls one side or the other of a corner in the mug's collider.
# A +18 mm move of the right hand in x at the moment it opens puts the handle on the recorded side (17 mm off) and moves the mug about 12 mm.
if os.environ.get("STIR_REL", "1") == "1":
    HS["win"][:] = [(42.9, 44.6, HANDS[1])]
    HS["bias"] = mkb((.018, 0, 0), 43.2, 43.6, 44.0, 44.4)
