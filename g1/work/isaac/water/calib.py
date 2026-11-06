"""Calibrate the PBD particle volume: fill the (static) mug with N particles, settle, measure the water
surface (isosurface top inside the mug) and the mug's inner volume below it -> ml per particle.
usage: <isaac-sim>/python calib.py [iters] [substeps] [count_ml]"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import math
import os
import sys
import time
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import water_layer as wl  # noqa: E402

iters = int(sys.argv[1]) if len(sys.argv) > 1 else 8
subs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
amounts = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "60,150").split(",")]
S0 = dict(np.load(str(WORK / "isaac/water/shots/pour_test/shot.npz"), allow_pickle=True))
S = {"fps": 60, "obj_paths": S0["obj_paths"], "obj_mats": S0["obj_mats"][:2], "t": S0["t"][:2]}
mug = S["obj_mats"][0, [str(p) for p in S["obj_paths"]].index(wl.MUG)]


def mug_volume(h):
    """inner volume (m^3) of the mug from its floor up to height h (mug frame)"""
    zs = np.linspace(wl.MUG_FLOOR, h, 200)
    r = np.interp(zs, [z for z, _ in wl.MUG_INNER], [r for _, r in wl.MUG_INNER])
    return np.trapezoid(math.pi * r ** 2, zs)


for ml in amounts:
    t0 = time.time()
    W = wl.WaterLayer(S, None, fill_ml=ml, substeps=subs, settle_s=2.0, iters=iters, fill_in="mug")
    surf = W._read_surface()
    p = surf["p"].astype(float)
    lm = (p - mug[3, :3]) @ mug[:3, :3].T
    inside = np.hypot(lm[:, 0], lm[:, 1]) < 0.02
    top = np.median(lm[inside & (lm[:, 2] > wl.MUG_FLOOR + 0.004), 2]) if inside.any() else float("nan")
    P = W.positions()
    lp = (P - mug[3, :3]) @ mug[:3, :3].T
    ptop = np.percentile(lp[:, 2], 99)
    V = mug_volume(top)
    print(f"CALIB iters {iters} subs {subs}: N {W.n} (nominal {ml:.0f} ml) iso surface z {top * 1e3:.1f} mm, "
          f"particle top {ptop * 1e3:.1f} mm -> volume {V * 1e6:.1f} ml -> {V / W.n * 1e9:.3f} mm^3/particle, "
          f"VP_FACTOR {V / W.n / W.fro ** 3:.3f} ({time.time() - t0:.0f} s)", flush=True)
    edges = np.arange(wl.MUG_FLOOR, 0.09, 0.004)
    cnt, _ = np.histogram(lp[:, 2], edges)
    rr = np.interp((edges[:-1] + edges[1:]) / 2, [z for z, _ in wl.MUG_INNER], [r for _, r in wl.MUG_INNER])
    dens = cnt / (math.pi * rr ** 2 * 0.004) * 1e-9          # particles per mm^3
    ic, _ = np.histogram(lm[inside, 2], edges)
    print(f"  film above the surface: {((lp[:, 2] > top + 0.003)).sum()} particles")
    print("  z(mm)  particles/mm^3  iso-verts(r<2cm)")
    for z, d_, c_ in zip(edges[:-1], dens, ic):
        print(f"  {z * 1e3:5.0f}  {d_:7.4f}  {c_:6d}")
    np.savez(f"{WORK}/isaac/water/logs/calib_{iters}_{int(ml)}.npz", P=lp, iso=lm)
    W.sim.detach_stage()
app.close()
