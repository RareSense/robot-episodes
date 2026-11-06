"""Cut the film panel (840 x 1080) out of a 1920 x 1080 film-camera clip with a crop window that follows the robot:
per frame the robot's torso and both wrists (from the baked shot) are projected through the film camera and the
window is centred on them, smoothed over ~1.5 s so it pans like an operator instead of jittering.
usage: follow_crop.py <shot_dir> <film.mp4> <out.mp4> [--first FRAME] [--width 840]
  --first: the shot frame the clip starts at (the clip's frames are the shot's frames from there on)."""
import argparse
import math
import subprocess

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("shot")
ap.add_argument("film")
ap.add_argument("out")
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--width", type=int, default=840)
A = ap.parse_args()
S = np.load(A.shot + r"\shot.npz", allow_pickle=True)
W, H, CW = 1920, 1080, A.width
names = [str(n) for n in S["robot_names"]]
idx = [names.index(n) for n in ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")]
probe = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", A.film], capture_output=True, text=True)
n = int(probe.stdout.strip())
f = 0.5 * W / math.tan(math.radians(float(S["film_hfov"])) / 2)
xs = []
for k in range(A.first, A.first + n):
    e, t = S["film_eye"][k], S["film_tgt"][k]
    fwd = (t - e) / np.linalg.norm(t - e)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    p = S["xpos"][k, idx].mean(0)                       # torso + both wrists: the body and the working hands
    d = p - e
    xs.append(W / 2 + f * (d @ right) / max(d @ fwd, 1e-3))
xs = np.array(xs)
win = 90                                                 # 1.5 s at 60 fps
pad = np.pad(xs, win, mode="edge")
smooth = np.convolve(pad, np.ones(2 * win + 1) / (2 * win + 1), mode="same")[win:-win]
x0 = np.clip(smooth - CW / 2, 0, W - CW).round().astype(int)
print(f"{n} frames, window left edge {x0.min()}..{x0.max()} px (centre crop would be {(W - CW) // 2})")
dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", A.film, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CW}x{H}",
                        "-r", "60", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", A.out],
                       stdin=subprocess.PIPE)
for k in range(n):
    buf = dec.stdout.read(W * H * 3)
    if len(buf) < W * H * 3:
        break
    img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    enc.stdin.write(np.ascontiguousarray(img[:, x0[k]:x0[k] + CW]).tobytes())
enc.stdin.close()
enc.wait()
dec.wait()
print("wrote", A.out)
