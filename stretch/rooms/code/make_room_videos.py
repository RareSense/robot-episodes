"""Builds the videos of a rooms run from its frames (rooms/<tag>/vid/eNN_KKKK.jpg, left = whole room from above, right = close on the oven).
One video per room with a label bar, plus two grid videos of all rooms (top views, oven views). H.264 previews, 10 frames/s = 5x speed at 0.5 s per frame. AI-operated runs.
usage: python make_room_videos.py <tag>"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

tag = sys.argv[1]
RUN = f"E:/isaaclab_mimic_test/rooms/{tag}"
VD, OUT = f"{RUN}/vid", f"{RUN}/video"
os.makedirs(OUT, exist_ok=True)
T = json.load(open(f"{VD}/times.json"))
ts, dt = T["t"], T["vid_dt"]
res = json.load(open(f"{RUN}/results.json"))["results"]
N, K = len(res), len(ts)
FPS = 10
speed = round(dt * FPS, 1)
try:
    F1, F2 = ImageFont.truetype("arial.ttf", 20), ImageFont.truetype("arialbd.ttf", 15)
except OSError:
    F1 = F2 = ImageFont.load_default()


def ff(path, w, h):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
                            stdin=subprocess.PIPE)


def frame(e, k):
    return Image.open(f"{VD}/e{e:02d}_{k:04d}.jpg")


def state(r):
    return ("PASS" if r["ok"] else "FAIL: " + "; ".join(r["fails"]))


for e, r in enumerate(res):                                   # one video per room
    te = r["t_done"] if r["t_done"] > 0 else ts[-1]
    ke = min(K, int(np.searchsorted(ts, te + 3.0)) + 1)
    rm = r["room"]
    lab = (f"Room {e:02d}   {rm['size_m'][0]:.1f} x {rm['size_m'][1]:.1f} m   counter at {rm['counter_angle_deg']:.0f} deg, {'on a wall' if rm['against_wall'] else 'free-standing'}   "
           f"{rm['furniture']} furniture   route {rm['route_m']:.1f} m   {state(r)}")
    p = ff(f"{OUT}/room_{e:02d}.mp4", 1280, 520)
    for k in range(ke):
        im = Image.new("RGB", (1280, 520), (18, 18, 20))
        im.paste(frame(e, k), (0, 40))
        d = ImageDraw.Draw(im)
        d.text((10, 8), lab, fill=(240, 240, 240), font=F1)
        d.text((1010, 8), f"t {ts[k]:5.1f} s   {speed:g}x   AI-operated", fill=(170, 200, 255), font=F1)
        p.stdin.write(im.tobytes())
    for _ in range(FPS):                                       # hold the last frame 1 s
        p.stdin.write(im.tobytes())
    p.stdin.close()
    p.wait()
print("room videos", N, flush=True)

cols = 6
rows = (N + cols - 1) // cols
cw, ch = 320, 240
for name, x0 in (("all_rooms_top", 0), ("all_rooms_oven", 640)):   # all rooms at once
    p = ff(f"{OUT}/{name}.mp4", cols * cw, rows * ch + 40)
    for k in range(K):
        im = Image.new("RGB", (cols * cw, rows * ch + 40), (18, 18, 20))
        d = ImageDraw.Draw(im)
        d.text((10, 8), f"{N} generated rooms, same skill, motion computed per room   t {ts[k]:5.1f} s   {speed:g}x   AI-operated", fill=(240, 240, 240), font=F1)
        for e, r in enumerate(res):
            c = frame(e, k).crop((x0, 0, x0 + 640, 480)).resize((cw, ch), Image.LANCZOS)
            ox, oy = (e % cols) * cw, (e // cols) * ch + 40
            im.paste(c, (ox, oy))
            done = r["t_done"] > 0 and (ts[k] >= r["t_done"] or k == K - 1)
            d.rectangle([ox + 4, oy + 4, ox + (92 if done else 34), oy + 24], fill=(30, 130, 60) if done and r["ok"] else (20, 20, 22))
            d.text((ox + 8, oy + 6), f"{e:02d}" + ("  DONE" if done else ""), fill=(255, 255, 255), font=F2)
        p.stdin.write(im.tobytes())
    for _ in range(2 * FPS):
        p.stdin.write(im.tobytes())
    p.stdin.close()
    p.wait()
    print(name, flush=True)
print("out", OUT)
