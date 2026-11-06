"""Read a <tag>_rec.npz recording and its particle file. Needs numpy only: no simulator.

    python examples/load_rec.py runs/<tag>/<tag>_rec.npz
    python examples/load_rec.py runs/<tag>/<tag>_rec.npz --particles <folder holding <tag>_rec.npz.particles.f32>
    python examples/load_rec.py --self-test

Format: docs/DATA_FORMAT.md.
"""
import argparse
import os
import sys
import tempfile

import numpy as np


class Recording:
    """Poses at 30 Hz. Every pose is (x, y, z, qx, qy, qz, qw) in the world frame, metres, scalar-last quaternion."""

    def __init__(self, path, particle_file=None):
        d = np.load(path, allow_pickle=False)
        self.t = d["t"]                                     # (F,) float64, sim seconds
        self.fps = float(d["fps"])                          # scalar, 30.0
        self.link_paths = [str(x) for x in d["link_paths"]]  # (L,) robot link prim paths
        self.link_T = d["link_T"]                           # (F, L, 7) float32
        self.body_paths = [str(x) for x in d["body_paths"]]  # (B,) free objects, then powder grains
        self.body_T = d["body_T"]                           # (F, B, 7) float32
        self.p_t = d["p_t"] if "p_t" in d.files else np.zeros(0)                  # (S,) float64, sim seconds of each particle sample
        self.p_n = d["p_n"] if "p_n" in d.files else np.zeros(0, np.int64)        # (S,) int64, points in each sample
        self.p_off = np.r_[0, np.cumsum(self.p_n)][:-1].astype(np.int64)          # first point of each sample
        self.particles = None
        pf = particle_file or str(path) + ".particles.f32"
        if len(self.p_n) and os.path.exists(pf):
            self.particles = np.memmap(pf, dtype="<f4", mode="r").reshape(-1, 3)  # all samples end to end; stays on disk

    def link(self, name):
        """(F, 7) poses of the link whose prim path ends with name."""
        return self.link_T[:, next(i for i, p in enumerate(self.link_paths) if p.endswith(name))]

    def body(self, name):
        """(F, 7) poses of the free body whose prim path ends with name, for example object_0316 (the mug)."""
        return self.body_T[:, next(i for i, p in enumerate(self.body_paths) if p.endswith(name))]

    def grains(self):
        """(F, G, 3) powder grain positions."""
        i = [k for k, p in enumerate(self.body_paths) if "/grains/" in p]
        return self.body_T[:, i, :3]

    def water(self, i):
        """(N, 3) water positions of particle sample i. Sample times are p_t[i]; there are 10 samples per sim second."""
        if self.particles is None:
            raise FileNotFoundError("particle file not found: it is hosted apart from the repo, see docs/DATA_HOSTING.md")
        a = int(self.p_off[i])
        return np.asarray(self.particles[a:a + int(self.p_n[i])])

    def water_at(self, t):
        """The latest particle sample at or before sim time t."""
        return self.water(max(0, int(np.searchsorted(self.p_t, t, "right")) - 1))


def quat_to_R(q):
    """(..., 4) x, y, z, w to (..., 3, 3)."""
    x, y, z, w = np.moveaxis(np.asarray(q, float), -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(x.shape + (3, 3))


def tilt_deg(q):
    """Angle between a body's +z and world +z."""
    return np.degrees(np.arccos(np.clip(quat_to_R(q)[..., 2, 2], -1, 1)))


def summary(r):
    print(f"{len(r.t)} frames at {r.fps} fps, t {r.t[0]:.2f} to {r.t[-1]:.2f} s")
    print(f"{len(r.link_paths)} links, {len(r.body_paths)} bodies, of which {sum('/grains/' in p for p in r.body_paths)} grains")
    print(f"{len(r.p_n)} particle samples, last {int(r.p_n[-1]) if len(r.p_n) else 0} points, file {'found' if r.particles is not None else 'not found'}")
    if any(p.endswith("object_0316") for p in r.body_paths):
        m = r.body("object_0316")
        print("mug end xyz", np.round(m[-1, :3], 3), "peak tilt", round(float(tilt_deg(m[:, 3:7]).max()), 2), "deg")
    if r.particles is not None:
        need = int(r.p_n.sum())
        print("particle points in file", len(r.particles), "listed in the recording", need)
        print("last sample z range", np.round([r.water(len(r.p_n) - 1)[:, 2].min(), r.water(len(r.p_n) - 1)[:, 2].max()], 3))


def self_test():
    """Writes a tiny recording in the real layout and reads it back."""
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "demo_rec.npz")
        F, L, B = 4, 2, 3
        T = np.zeros((F, L, 7), np.float32)
        T[..., 6] = 1
        Bd = np.zeros((F, B, 7), np.float32)
        Bd[..., 6] = 1
        Bd[:, 0, 0] = np.arange(F)
        np.savez_compressed(p, t=np.arange(F) / 30, fps=30.0, link_paths=np.array(["/World/h1_2/pelvis", "/World/h1_2/L_hand_base_link"]), link_T=T,
                            body_paths=np.array(["/World/Objects/object_0316", "/World/Props/jar_lid", "/World/fluid/grains/g000"]), body_T=Bd,
                            p_t=np.array([0.1, 0.2]), p_n=np.array([2, 3], np.int64))
        pts = np.arange(15, dtype="<f4")
        pts.tofile(p + ".particles.f32")
        r = Recording(p)
        assert r.body("object_0316")[3, 0] == 3 and r.grains().shape == (F, 1, 3) and r.link("pelvis").shape == (F, 7)
        assert r.water(0).shape == (2, 3) and r.water(1)[0, 0] == 6 and r.water_at(0.15).shape == (2, 3)
        assert abs(tilt_deg(np.array([0, 0, 0, 1.0]))) < 1e-9
        assert abs(tilt_deg(np.array([np.sin(np.pi / 4), 0, 0, np.cos(np.pi / 4)])) - 90) < 1e-6
        r.particles = None                # release the memory map before the folder goes
        del r
    print("self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", nargs="?")
    ap.add_argument("--particles", default="", help="folder or file, if the particle file is not beside the recording")
    ap.add_argument("--self-test", action="store_true")
    A = ap.parse_args()
    if A.self_test:
        self_test()
        sys.exit(0)
    if not A.rec:
        ap.error("give a recording, or --self-test")
    pf = A.particles
    if pf and os.path.isdir(pf):
        pf = os.path.join(pf, os.path.basename(A.rec) + ".particles.f32")
    summary(Recording(A.rec, pf or None))
