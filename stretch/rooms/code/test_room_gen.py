"""Offline check of the room generator in rooms_body.txt (no simulator): runs it for N rooms on CPU and draws them, so a bug shows before a GPU run. usage: python test_room_gen.py [N] [seed]"""
import math
import os
import sys
import textwrap
import time
import types

import numpy as np
import torch
from PIL import Image, ImageDraw

D = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
args = types.SimpleNamespace(seed=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
dev = "cpu"
src = open(f"{D}/rooms_body.txt", encoding="utf-8").read()
a, b = src.index("        CZ, RZ, NFX"), src.index("        T = lambda a: torch.tensor")
code = textwrap.dedent(src[a:b])
exec(code)
print("ok rooms", sum(okroom), "of", N)
S = 70                                    # pixels per metre
cols = 6
rows = (N + cols - 1) // cols
W = int(7.6 * S)
img = Image.new("RGB", (cols * W, rows * W), (255, 255, 255))
dr = ImageDraw.Draw(img)
for e in range(N):
    ox, oy = (e % cols) * W + W / 2, (e // cols) * W + W / 2
    P = lambda x, y: (ox + x * S, oy - y * S)
    Lx, Ly = R["L"][e]
    dr.rectangle([P(-Lx / 2, Ly / 2), P(Lx / 2, -Ly / 2)], outline=(60, 60, 60), width=3)
    th, c = R["th"][e], R["c"][e]
    ct, st = math.cos(th), math.sin(th)
    l2w = lambda x, y: (c[0] + ct * x - st * y, c[1] + st * x + ct * y)
    poly = lambda x0, x1, y0, y1: [P(*l2w(x, y)) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    dr.polygon(poly(.15, 2.9, 2.05, 2.6), fill=(205, 160, 100), outline=(120, 80, 40))
    fx_, fy_ = float(fx[e]), float(fy[e])
    dr.polygon(poly(fx_ - .25, fx_ + .25, fy_, fy_ + .38), fill=(70, 70, 75))
    dr.polygon(poly(fx_ - .25, fx_ + .25, fy_ - .29, fy_), outline=(150, 150, 160))
    px_, py_ = float(V["plate_x"][e]), float(V["plate_y"][e])
    q = P(*l2w(px_, py_))
    dr.ellipse([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5], fill=(230, 60, 40))
    colr = [(140, 95, 60), (200, 200, 190), (50, 90, 150), (150, 55, 50), (85, 85, 92), (50, 130, 75)]
    for k in range(6):
        x, y, a_, on = R["obs"][e, k]
        if on > .5:
            w, h = OBS_FOOT[k]
            ca, sa = math.cos(a_), math.sin(a_)
            dr.polygon([P(x + ca * u - sa * v, y + sa * u + ca * v) for u, v in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))], fill=colr[k])
    s = R["start"][e]
    pts = [P(s[0], s[1])] + [P(*l2w(*R["wp"][e, k])) for k in range(R["nwp"][e])]
    dr.line(pts, fill=(20, 140, 220), width=3)
    dr.ellipse([pts[0][0] - 8, pts[0][1] - 8, pts[0][0] + 8, pts[0][1] + 8], outline=(0, 0, 0), width=3)
    dr.line([pts[0], (pts[0][0] + 16 * math.cos(s[2]), pts[0][1] - 16 * math.sin(s[2]))], fill=(0, 0, 0), width=3)
    dr.text((ox - W / 2 + 6, oy - W / 2 + 4), f"{e}: {Lx:.1f}x{Ly:.1f} m, counter {math.degrees(th):.0f} deg, route {R['path_m'][e]:.1f} m", fill=(0, 0, 0))
out = "E:/isaaclab_mimic_test/rooms_plan.jpg"
os.makedirs(os.path.dirname(out), exist_ok=True)
img.save(out, quality=88)
print(out, "room size", R["L"].min().round(2), R["L"].max().round(2), "route m", R["path_m"].min().round(2), R["path_m"].max().round(2), "tries max", int(R["tries"].max()), "on wall", int(R["wall"].sum()), "furniture", (R["obs"][:, :, 3] > .5).sum(1).tolist())
