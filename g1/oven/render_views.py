"""Render three views (main, head, hand) of a recorded G1 oven episode by setting the recorded MuJoCo state into the
same scene and rendering offscreen. Nothing is simulated: every recorded value is set, rows are linearly
interpolated (10 Hz recording -> 30 fps video).
usage: python render_views.py <episode_dir> <out_dir> [--views main,head,hand] [--frames a:b] [--stills t1,t2,..]
Needs the urc package (src/urc/tasks/g1oven/scene.py) to build the scene MJCF from scene.json."""
import sys, os, json, math, argparse
import numpy as np, mujoco, cv2
sys.path.insert(0, os.environ.get("URC_SRC", "E:/urc_g1_multiply/src"))
from urc.tasks.g1oven import scene as SC

HEAD = dict(body="torso_link", pos=(0.0576, 0.0175, 0.4299), pitch_deg=47.6)      # G1 head depth camera, torso frame
HAND = dict(body="left_wrist_yaw_link", pos=(-0.03, -0.12, 0.0), look=(0.14, 0.0, 0.0), up=(0, 0, 1))

def xyaxes(fwd, up):
    f = np.asarray(fwd, float); f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return list(r) + list(u)

def build(sd, hand=HAND, head=HEAD):
    spec = mujoco.MjSpec.from_string(SC.scene_xml(sd))
    th = math.radians(head["pitch_deg"])
    spec.body(head["body"]).add_camera(name="headcam", pos=list(head["pos"]), fovy=75,
        xyaxes=xyaxes((math.cos(th), 0, -math.sin(th)), (math.sin(th), 0, math.cos(th))))
    fwd = np.array(hand["look"]) - np.array(hand["pos"])
    spec.body(hand["body"]).add_camera(name="handcam", pos=list(hand["pos"]), fovy=80, xyaxes=xyaxes(fwd, hand["up"]))
    return spec.compile()

class Ep:
    def __init__(self, ep_dir, hand=HAND, head=HEAD):
        self.sd = json.load(open(os.path.join(ep_dir, "scene.json")))
        self.tr = np.load(os.path.join(ep_dir, "recording.npz"))["trace"].astype(np.float64)
        self.m = build(self.sd, hand, head); self.d = mujoco.MjData(self.m)
        m = self.m; J = lambda n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
        self.names = self.sd["oven"].get("knob", {}).get("names", [])
        self.nk = len(self.names)
        self.a_oven, self.a_door, self.a_plate, self.a_food = J("oven_free"), J("oven_door_j"), J("plate_free"), J("plate_food_free")
        self.a_knob = [J(f"oven_knob_{n}_j") for n in self.names]
        assert self.tr.shape[1] == 1 + 50 + 7 + 3 + 1 + self.nk + 3 + 1, (self.tr.shape, self.nk)
        self.t = self.tr[:, 0]
        self.ren = {}
        self.pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.oven_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "oven")
        self.cam_state = None

    def row_at(self, t):
        t = min(max(t, self.t[0]), self.t[-1]); i = min(np.searchsorted(self.t, t, side="right") - 1, len(self.t) - 2)
        w = (t - self.t[i]) / (self.t[i + 1] - self.t[i]); a, b = self.tr[i], self.tr[i + 1]
        r = a * (1 - w) + b * w
        for s in (slice(4, 8), slice(1 + 50 + 3, 1 + 50 + 7)):               # quaternions: base (q[3:7]) and plate (plate_qpos[3:7])
            qa, qb = a[s].copy(), b[s].copy()
            if qa @ qb < 0: qb = -qb
            q = qa * (1 - w) + qb * w; r[s] = q / np.linalg.norm(q)
        return r

    def set_state(self, t):
        r = self.row_at(t); d = self.d; c = 1
        d.qpos[:50] = r[c:c + 50]; c += 50
        d.qpos[self.a_plate:self.a_plate + 7] = r[c:c + 7]; c += 7
        d.qpos[self.a_food:self.a_food + 3] = r[c:c + 3]; d.qpos[self.a_food + 3:self.a_food + 7] = [1, 0, 0, 0]; c += 3
        d.qpos[self.a_door] = math.radians(r[c]); c += 1
        for a in self.a_knob: d.qpos[a] = math.radians(r[c]); c += 1
        op, yaw = r[c:c + 3], r[c + 3]
        d.qpos[self.a_oven:self.a_oven + 3] = op; d.qpos[self.a_oven + 3:self.a_oven + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.m, d)

    def renderer(self, w, h):
        if (w, h) not in self.ren: self.ren[(w, h)] = mujoco.Renderer(self.m, h, w)
        return self.ren[(w, h)]

    def main_cam(self):
        pel = self.d.xpos[self.pel] + [0, 0, 0.55]; ov = self.d.xpos[self.oven_b]
        near = np.linalg.norm(pel[:2] - ov[:2]) < 1.6
        tgt = np.array((pel + ov + [0, 0, 0.3]) / 2 if near else pel); dist = 2.4 if near else 3.3
        if self.cam_state is None: self.cam_state = np.append(tgt, dist)
        self.cam_state = 0.9 * self.cam_state + 0.1 * np.append(tgt, dist)
        base = SC.oven_camera(self.sd)
        return SC.camera(self.cam_state[:3], self.cam_state[3], base.azimuth, -22.0)

    def frame(self, view, label, w, h):
        self.set_state(self.cur_t)
        r = self.renderer(w, h)
        if view == "main":
            r.update_scene(self.d, self.main_cam()); im = r.render().copy()
            r2 = self.renderer(320, 240); r2.update_scene(self.d, SC.room_camera(self.sd)); top = r2.render()
            im[28:268, w - 324:w - 4] = top; cv2.rectangle(im, (w - 325, 27), (w - 4, 268), (255, 255, 255), 1)
        else:
            r.update_scene(self.d, "headcam" if view == "head" else "handcam"); im = r.render().copy()
        s = max(0.5, w / 1280 * 0.7)
        cv2.rectangle(im, (0, 0), (w, int(26 * s * 1.4)), (0, 0, 0), -1)
        cv2.putText(im, f"AI-operated | {label} | {view} | t={self.cur_t:6.1f}s", (6, int(19 * s * 1.4)), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)
        return im

SIZE = {"main": (1280, 720), "head": (960, 720), "hand": (960, 720)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ep"); ap.add_argument("out"); ap.add_argument("--label", default="")
    ap.add_argument("--views", default="main,head,hand"); ap.add_argument("--stills", default=""); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--limit", type=float, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ep = Ep(a.ep); label = a.label or os.path.basename(a.ep.rstrip("/\\"))
    if a.stills:
        for v in a.views.split(","):
            for ts in a.stills.split(","):
                ep.cur_t = float(ts); w, h = SIZE[v]
                cv2.imwrite(os.path.join(a.out, f"still_{v}_{float(ts):.0f}.jpg"), ep.frame(v, label, w, h)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
        sys.exit()
    import imageio
    T = ep.t[-1] if not a.limit else min(a.limit, ep.t[-1]); n = int(T * a.fps)
    for v in a.views.split(","):
        ep.cam_state = None; w, h = SIZE[v]
        wr = imageio.get_writer(os.path.join(a.out, f"{v}.mp4"), fps=a.fps, codec="libx264", macro_block_size=8,
                                ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "24", "-preset", "medium"])
        for k in range(n):
            ep.cur_t = k / a.fps; wr.append_data(ep.frame(v, label, w, h))
        wr.close(); print("done", v, n, "frames", flush=True)
