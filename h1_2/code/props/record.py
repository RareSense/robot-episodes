r"""Video for headless Isaac: log during physics, render afterwards, so physics speed is unaffected.

Physics side (inside the episode runner, no rendering, only reads):
    import record
    rec = record.Recorder(s, bodies=record.default_bodies(s), fps=30).attach()     # every s.step() is sampled
    ...run the episode...
    rec.save(RP.out("props_logs", 'run1.npz'))

Render side (separate process, after the physics process has exited):
    <isaac-sim>/python.bat record.py run1.npz run1 [--res 1280x720] [--frames 0:150] [--eye x,y,z --target x,y,z]
  -> <repo>/code\previews\run1_fixed.mp4 and run1_hands.mp4, frames in previews\frames_run1\{fixed,hands}\
Log contents: t, link_paths, link_T (frames, links, 7), body_paths, body_T (frames, bodies, 7), poses x y z qx qy qz qw.

Water and powder (optional):
    rec = record.Recorder(s, bodies=record.default_bodies(s), fps=30, particles=F.sample, particle_fps=10, views=[F.gv]).attach()
    particles: a callable returning an (N, 3) array. With no argument it is called after a step. With one argument (F.sample) it is
    handed the step function and must run that step itself, so sampling adds no extra physics step. views: tensor rigid body views
    logged with the bodies (F.gv = the 120 grains, one view, fast). Particles stream to <log>.particles.f32 beside the npz, so a long
    run does not hold them in RAM: 45.8k particles at 10 fps is 5.5 MB per sim second.
Fast long render:
    record.py run1.npz run1 --res 960x540 --no-frames --loop update [--stride 2] [--second close --close-eye x,y,z --close-target x,y,z]
  Measured 12.49 frames/s for 8932 frames, two videos. --loop update repeats a few frames and lags about one frame; --loop rep does not, at 6.2 frames/s."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))    # release_paths.py lives at the code root
import release_paths as RP
import inspect
import os
import sys

import numpy as np

PREV = RP.out_dir("previews")
LOOKS = RP.looks()
LAYERS = ("looks_props.usda", "looks_room.usda", "looks_exterior.usda", "set_fix.usda", "kettle_fix.usda", "looks.usda")   # strongest first, as tea_scene.usda


def default_bodies(s):
    """Every free object near the tea corner, plus the props."""
    import props
    from pxr import UsdPhysics
    out = [x for x in props.prop_paths().values() if s.stage.GetPrimAtPath(x)]
    for k, p in s.objs.items():
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and "object_0294" < k < "object_0322":
            out.append(p.GetPath().pathString)
    return out


class Recorder:
    def __init__(self, sim, bodies=(), fps=30, particles=None, particle_fps=10, views=(), particle_file=None):
        # h12_physx, or the frozen copy the sim was built from; walk the classes so a subclass in the runner script also works
        H = next(sys.modules[c.__module__] for c in type(sim).__mro__ if hasattr(sys.modules.get(c.__module__), "DT"))
        self.s = sim
        self.every = max(1, round(1 / (fps * H.DT)))
        self.fps = 1 / (self.every * H.DT)
        self.k = 0
        self.t, self.L, self.B = [], [], []
        self.link_paths = list(sim.art.link_paths[0])
        self.body_paths = list(bodies)
        self.body_view = sim.sv.create_rigid_body_view(self.body_paths) if self.body_paths else None
        actual = list(map(str, self.body_view.prim_paths)) if self.body_view is not None else []
        self.body_order = [actual.index(p) for p in self.body_paths]
        self.group_views = [v for v in views if v is not None]
        for v in self.group_views:
            self.body_paths += [str(x) for x in v.prim_paths]
        self._orig = None
        self.particles = particles
        self.pevery = max(1, round(1 / (particle_fps * H.DT)))
        self.p_takes_step = particles is not None and len(inspect.signature(particles).parameters) >= 1
        self.p_t, self.p_n, self.p_bad = [], [], 0
        self.p_path = particle_file or os.path.join(RP.out_dir("props_logs"), f"_particles_{os.getpid()}.f32")
        self.p_file = open(self.p_path, "wb") if particles is not None else None

    def attach(self):
        s = self.s
        if self._orig is None:
            self._orig = s.step

            def step():
                due = self.particles is not None and (self.k + 1) % self.pevery == 0
                if due and self.p_takes_step:
                    P = self.particles(self._orig)
                else:
                    self._orig()
                    P = self.particles() if due else None
                if due:
                    self._log_particles(P)
                self.tick()
            s.step = step
        self.sample()
        return self

    def _log_particles(self, P):
        P = None if P is None else np.ascontiguousarray(P, np.float32)
        if P is None or P.ndim != 2 or P.shape[1] != 3 or not np.isfinite(P).all():
            self.p_bad += 1
            return
        self.p_file.write(P.tobytes())
        self.p_t.append(float(self.s.t))
        self.p_n.append(len(P))

    def sample(self):
        self.t.append(float(self.s.t))
        poses = self.s.link_poses() if hasattr(self.s, 'link_poses') else np.asarray(self.s.art.get_link_transforms())[0]
        self.L.append(poses.astype(np.float32).copy())
        B = [np.asarray(self.body_view.get_transforms())[self.body_order].astype(np.float32).copy()
             if self.body_view is not None else np.empty((0,7),np.float32)]
        B += [np.asarray(v.get_transforms()).astype(np.float32).reshape(-1, 7) for v in self.group_views]
        self.B.append(np.concatenate(B))

    def tick(self):
        self.k += 1
        if self.k % self.every == 0:
            self.sample()
            # Live frames: save as the run goes, so stills can be rendered without waiting for the end. REC_AUTOSAVE_S sim seconds, 0 turns it off.
            # Path follows the particle file when it has one, else REC_AUTOSAVE_PATH.
            gap = float(os.environ.get("REC_AUTOSAVE_S", "15"))
            pth = self.p_path[:-len(".particles.f32")] if (self.p_file is not None and self.p_path.endswith(".particles.f32")) else os.environ.get("REC_AUTOSAVE_PATH")
            if not hasattr(self, "_saved_t"):
                self._saved_t = float(self.s.t)
            if gap > 0 and pth and float(self.s.t) - self._saved_t >= gap:
                self._saved_t = float(self.s.t)
                self.save(pth)

    def save(self, path):
        """Safe to call more than once, so a long run can save as it goes."""
        extra = {}
        if self.p_file is not None:
            self.p_file.flush()
            final = path + ".particles.f32"
            if os.path.normcase(self.p_path) != os.path.normcase(final):
                self.p_file.close()
                os.replace(self.p_path, final)
                self.p_path = final
                self.p_file = open(final, "ab")
            extra = dict(p_t=np.array(self.p_t), p_n=np.array(self.p_n, np.int64))
        np.savez_compressed(path, t=np.array(self.t), fps=self.fps, link_paths=np.array(self.link_paths), link_T=np.array(self.L),
                            body_paths=np.array(self.body_paths), body_T=np.array(self.B), **extra)
        print("recorder saved", len(self.t), "frames at", round(self.fps, 2), "fps to", path, flush=True)
        if self.p_file is not None:
            print("particles:", len(self.p_t), "samples, last count", self.p_n[-1] if self.p_n else 0, "bad samples", self.p_bad, flush=True)


def _render():
    import argparse
    import json
    import math
    import subprocess
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("name")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--frames", default="")
    ap.add_argument("--still", default="", help="comma frame numbers: PNGs only, no video")
    ap.add_argument("--eye", default="-0.55,0.55,1.75")
    ap.add_argument("--target", default="0.62,2.0,0.95")
    ap.add_argument("--hfov", type=float, default=55.0)
    ap.add_argument("--left-offset", default="-0.36,-0.28,0.20", help="left close-up camera, offset from the smoothed left hand")
    ap.add_argument("--right-offset", default="0.36,-0.28,0.20", help="right close-up camera, offset from the smoothed right hand")
    ap.add_argument("--fill", type=float, default=600.0, help="preview fill light from the room side (intensity, 0 = none): the robot is black and backlit by the window")
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--no-looks", action="store_true")
    ap.add_argument("--stride", type=int, default=1, help="render every Nth logged frame; the video keeps real-time speed at fps/N")
    ap.add_argument("--no-frames", action="store_true", help="no per-frame images, video only (frames go straight to ffmpeg)")
    ap.add_argument("--loop", default="rep", choices=("rep", "update"), help="update: app.update() instead of replicator's step, faster")
    ap.add_argument("--updates", type=int, default=2, help="--loop update: app.update() calls per frame; 1 is fastest but repeats some frames")
    ap.add_argument("--second", default="hands", choices=("hands", "close"), help="second video: both hands side by side, or one fixed close camera")
    ap.add_argument("--close-eye", default="1.55,1.75,1.25")
    ap.add_argument("--close-target", default="1.90,2.17,1.00")
    ap.add_argument("--close-hfov", type=float, default=35.0)
    ap.add_argument("--water-width", type=float, default=3.0, help="drawn particle diameter, mm")
    ap.add_argument("--water-opacity", type=float, default=0.55)
    ap.add_argument("--keep-kettle-water", action="store_true", help="also draw the water hidden inside the kettle (slower)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many rendered frames (speed tests)")
    A = ap.parse_args()
    W, Hh = map(int, A.res.split("x"))
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": W, "height": Hh, "renderer": "RaytracedLighting"})
    import carb
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
    sys.path.insert(0, RP.code())
    sys.path.insert(0, RP.code('props'))
    import h12_physx as H
    import props

    log_path = A.log if os.path.isabs(A.log) else os.path.join(RP.out_dir("props_logs"), A.log)
    S = np.load(log_path, allow_pickle=False)
    cs = carb.settings.get_settings()
    SET = {"/rtx/rendermode": "RaytracedLighting", "/rtx/post/aa/op": 3, "/rtx/post/tonemap/op": 6, "/rtx/post/tonemap/filmIso": 100.0 * 2 ** A.exposure,
           "/rtx/post/histogram/enabled": False, "/rtx/directLighting/sampledLighting/enabled": True, "/omni/replicator/RTSubframes": 1,
           "/app/viewport/grid/enabled": False, "/persistent/app/viewport/grid/enabled": False, "/app/viewport/show/grid": False,
           "/persistent/app/viewport/displayOptions": 0, "/app/viewport/displayOptions": 0, "/app/viewport/outline/enabled": False,
           "/rtx/translucency/enabled": True, "/rtx/raytracing/fractionalCutoutOpacity": True}
    for k, v in SET.items():
        cs.set(k, v)

    ctx = omni.usd.get_context()
    # Render stage: a small file of our own that sublayers the looks and the room; edits go to its root layer, in memory only.
    # The source USD files are not touched.
    wrap = RP.out("props_logs", 'render_stage.usda')
    subs = [] if A.no_looks else [os.path.join(LOOKS, x) for x in LAYERS]
    subs.append(H.ROOM)
    body = ",".join("@" + x.replace(chr(92), "/") + "@" for x in subs)
    with open(wrap, "w") as fh:
        fh.write('#usda 1.0\n(\n    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Z"\n    subLayers = [' + body + ']\n)\n')
    ctx.open_stage(wrap)
    for _ in range(5):
        app.update()
    st = ctx.get_stage()
    objs = {}
    for p in st.Traverse():
        if p.GetName().startswith("object_") and p.GetTypeName() == "Xform":
            objs.setdefault(p.GetName(), p)
    for k, v in json.load(open(H.CORR)).items():
        p = objs.get(k)
        if k.startswith("_") or p is None or len(v) != 7:
            continue
        UsdGeom.Xformable(p).MakeMatrixXform().Set(Gf.Matrix4d().SetTransform(Gf.Rotation(Gf.Quatd(v[3], v[4], v[5], v[6])), Gf.Vec3d(*v[:3])))
    r = st.DefinePrim("/World/h1_2", "Xform")
    r.GetReferences().AddReference(H.ROBOT)
    r.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    if any(str(x).startswith(props.ROOT) for x in S["body_paths"]):
        props.install(st)
    # Rebuild the same visible free tools that existed during the recorded run.
    # Their motion below comes only from recorded PhysX body transforms.
    recorded_paths = set(map(str, S["body_paths"]))
    if recorded_paths.intersection({"/World/Props/powder_scoop", "/World/Props/jar_lid"}):
        sys.path.insert(0, RP.code('steps', 'powder'))
        import pwprops
        pwprops.install_jar(st, with_lid=pwprops.LID in recorded_paths)
        if pwprops.SCOOP in recorded_paths:
            pwprops.install_scoop(st)
    app.update()
    # replicator's step runs the timeline: with physics live, PhysX takes over the bodies and drops the xform ops set below. Render with physics off.
    print("robot Physics variants", r.GetVariantSets().GetVariantSet("Physics").GetVariantNames(), flush=True)
    nb = 0
    for p in Usd.PrimRange(st.GetPseudoRoot()):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr().Set(False)
            nb += 1
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(p).CreateArticulationEnabledAttr().Set(False)
        if p.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(p).CreateJointEnabledAttr().Set(False)
    print("rigid bodies disabled for rendering", nb, flush=True)

    # posable prims: local matrix = world * inverse(parent world); parent is the nearest logged ancestor, or a static prim
    lp, bp = [str(x) for x in S["link_paths"]], [str(x) for x in S["body_paths"]]
    LT, BT = S["link_T"], S["body_T"]
    allp = lp + bp
    GRAIN = "/World/fluid/grains/"
    gi = np.array([i for i, x in enumerate(bp) if x.startswith(GRAIN)], int)
    xc = UsdGeom.XformCache()
    pose = []
    for i, path in enumerate(allp):
        if path.startswith(GRAIN):
            continue
        prim = st.GetPrimAtPath(path)
        if not prim:
            print("missing prim", path, flush=True)
            continue
        par = path.rsplit("/", 1)[0]
        j = None
        while par:
            if par in allp:
                j = allp.index(par)
                break
            par = par.rsplit("/", 1)[0] if "/" in par else ""
        static = xc.GetLocalToWorldTransform(prim.GetParent()) if j is None else None
        UsdGeom.Xformable(prim).MakeMatrixXform()
        pose.append((i, j, static, path + ".xformOp:transform"))
    print("posing", len(pose), "prims", flush=True)

    def mat(T):
        return Gf.Matrix4d().SetTransform(Gf.Rotation(Gf.Quatd(float(T[6]), float(T[3]), float(T[4]), float(T[5]))), Gf.Vec3d(float(T[0]), float(T[1]), float(T[2])))

    nL = len(lp)
    hi_ = [lp.index(next(x for x in lp if x.endswith(n))) for n in ("L_hand_base_link", "R_hand_base_link")]
    w = max(1, int(float(S["fps"])))
    F = LT.shape[0]

    def smooth(x):
        return np.array([x[max(0, i - w):i + w + 1].mean(0) for i in range(F)])

    hands = {"handL": (smooth(LT[:, hi_[0], :3].astype(float)), np.array(list(map(float, A.left_offset.split(","))))),
             "handR": (smooth(LT[:, hi_[1], :3].astype(float)), np.array(list(map(float, A.right_offset.split(",")))))}

    def camera(path, hfov, aspect=None):
        c = UsdGeom.Camera.Define(st, path)
        HA = 36.0
        c.CreateHorizontalApertureAttr(HA)
        c.CreateVerticalApertureAttr(HA / (aspect or W / Hh))
        c.CreateClippingRangeAttr(Gf.Vec2f(0.01, 500))
        c.CreateFocalLengthAttr(float(HA / (2 * math.tan(math.radians(hfov) / 2))))
        UsdGeom.Xformable(c).MakeMatrixXform()

    def look(path, e, t):
        M = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, e)), Gf.Vec3d(*map(float, t)), Gf.Vec3d(0, 0, 1)).GetInverse()
        st.GetAttributeAtPath(path + ".xformOp:transform").Set(M)

    camera("/World/RecFixed", A.hfov)
    look("/World/RecFixed", A.eye.split(","), A.target.split(","))
    Wh = W // 2 // 2 * 2
    rp = {"fixed": rep.create.render_product("/World/RecFixed", (W, Hh))}
    if A.second == "close":
        hands = {}
        camera("/World/RecClose", A.close_hfov)
        look("/World/RecClose", A.close_eye.split(","), A.close_target.split(","))
        rp["close"] = rep.create.render_product("/World/RecClose", (W, Hh))
    for k, (c, o) in hands.items():
        camera("/World/Rec" + k, 42.0, Wh / Hh)
        look("/World/Rec" + k, c[0] + o, c[0])
        rp[k] = rep.create.render_product("/World/Rec" + k, (Wh, Hh))

    # water and powder: two point clouds, redrawn from the log. Water holds its last sample between logged particle frames.
    pt = S["p_t"] if "p_t" in S.files else np.zeros(0)
    T = S["t"]
    water = grains = None
    if len(pt):
        pn = S["p_n"].astype(np.int64)
        poff = np.r_[0, np.cumsum(pn)][:-1]
        PF = np.memmap(log_path + ".particles.f32", np.float32, "r").reshape(-1, 3)
        water = UsdGeom.Points.Define(st, "/World/RecWater")
        water.CreateWidthsAttr([A.water_width / 1000.0])
        water.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        m = UsdShade.Material.Define(st, "/World/RecLooks/water")
        sh = UsdShade.Shader.Define(st, "/World/RecLooks/water/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.45, 0.72, 0.95))
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(A.water_opacity)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.15)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(water.GetPrim()).Bind(m)
        print("water:", len(pt), "particle samples, up to", int(pn.max()), "particles", flush=True)
    if len(gi):
        grains = UsdGeom.Points.Define(st, "/World/RecGrains")
        grains.CreateWidthsAttr([0.005])
        grains.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        m = UsdShade.Material.Define(st, "/World/RecLooks/powder")
        sh = UsdShade.Shader.Define(st, "/World/RecLooks/powder/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.93, 0.89, 0.78))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(grains.GetPrim()).Bind(m)
        print("powder:", len(gi), "grains", flush=True)
    ki = next((i for i, x in enumerate(bp) if x.endswith("/object_0295")), None)
    wstate = {"i": -1}

    def set_fluid(f):
        if water is not None:
            i = max(0, int(np.searchsorted(pt, T[f], "right")) - 1)
            if i != wstate["i"]:
                wstate["i"] = i
                P = np.asarray(PF[poff[i]:poff[i] + pn[i]])
                if ki is not None and not A.keep_kettle_water:          # the kettle is opaque: skip what cannot be seen
                    k = BT[f, ki].astype(float)
                    u, w_ = -k[3:6], k[6]
                    v = P.astype(float) - k[:3]
                    L = v + 2 * np.cross(u, np.cross(u, v) + w_ * v)
                    P = P[~((np.hypot(L[:, 0], L[:, 1]) < 0.074) & (L[:, 2] > 0.004) & (L[:, 2] < 0.205))]
                wstate["n"] = len(P)
                water.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(P)))
        if grains is not None:
            grains.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(BT[f, gi, :3])))
    if A.fill > 0:
        d = UsdLux.DistantLight.Define(st, "/World/RecFill")
        d.CreateIntensityAttr(A.fill)
        d.CreateAngleAttr(30.0)
        d.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
        UsdGeom.Xformable(d).AddRotateXYZOp().Set(Gf.Vec3f(75.0, 0.0, -20.0))     # shines toward +y and down: onto the robot's back and the counter
    for k, v in SET.items():
        cs.set(k, v)
    ann = {}
    for k, v in rp.items():
        ann[k] = rep.AnnotatorRegistry.get_annotator("rgb")
        ann[k].attach([v])

    attrs = {}
    last = {}

    def set_frame(f):
        for i, j, static, op in pose:
            Tw = LT[f, i] if i < nL else BT[f, i - nL]
            P = None if j is None else (LT[f, j] if j < nL else BT[f, j - nL])
            key = Tw.tobytes() if P is None else Tw.tobytes() + P.tobytes()
            if last.get(i) == key:                       # asleep or unchanged: nothing to send to the renderer
                continue
            last[i] = key
            M = mat(Tw)
            if j is None:
                M = M * static.GetInverse()
            else:
                M = M * mat(P).GetInverse()
            st.GetAttributeAtPath(op).Set(M)
        for k, (c, o) in hands.items():
            look("/World/Rec" + k, c[f] + o, c[f])
        set_fluid(f)

    a, b = 0, F
    if A.frames:
        a, b = map(int, A.frames.split(":"))
    second = "close" if A.second == "close" else "hands"
    out = {k: os.path.join(PREV, f"frames_{A.name}", k) for k in ("fixed", second)}
    if not A.no_frames or A.still:
        for d in out.values():
            os.makedirs(d, exist_ok=True)
    set_frame(a)
    t0 = time.time()
    for _ in range(8):
        rep.orchestrator.step(rt_subframes=4, pause_timeline=True)
    print("warm-up", round(time.time() - t0, 1), "s", flush=True)
    frames = [int(x) for x in A.still.split(",")] if A.still else list(range(a, b, A.stride))
    if A.max_frames:
        frames = frames[:A.max_frames]
    vfps = float(S["fps"]) / A.stride
    enc = {}
    if not A.still:
        for k in out:
            mp4 = os.path.join(PREV, f"{A.name}_{k}.mp4")
            enc[k] = (mp4, subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W if k == 'fixed' or A.second == 'close' else 2 * Wh}x{Hh}",
                                             "-r", f"{vfps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", mp4], stdin=subprocess.PIPE))

    def pose_error(f):
        """Does the posed stage match the physics log? World position of every link, USD against log."""
        c = UsdGeom.XformCache()
        e = 0.0
        for i, path in enumerate(lp):
            m = c.GetLocalToWorldTransform(st.GetPrimAtPath(path)).ExtractTranslation()
            e = max(e, float(np.linalg.norm(np.array(m) - LT[f, i, :3])))
        return e * 1000

    worst = 0.0
    tm = {"pose": 0.0, "render": 0.0, "read": 0.0, "out": 0.0}
    t1 = time.time()
    for n, f in enumerate(frames):
        c0 = time.time()
        set_frame(f)
        if n % 30 == 0:
            worst = max(worst, pose_error(f))
        c1 = time.time()
        if A.loop == "update" and not A.still:
            for _ in range(A.updates):
                app.update()
        else:
            rep.orchestrator.step(rt_subframes=1, pause_timeline=True)
        c2 = time.time()
        img = {k: np.asarray(ann[k].get_data())[:, :, :3] for k in rp}
        c3 = time.time()
        pics = {"fixed": img["fixed"], second: img["close"] if A.second == "close" else np.hstack([img["handL"], img["handR"]])}
        for k, v in pics.items():
            if v.size == 0:
                continue
            if A.still:
                Image.fromarray(v).save(os.path.join(out[k], f"still_{f:05d}.png"))
                continue
            enc[k][1].stdin.write(np.ascontiguousarray(v).tobytes())
            if not A.no_frames:
                Image.fromarray(v).save(os.path.join(out[k], f"{n:05d}.jpg"), quality=93)
        c4 = time.time()
        tm["pose"] += c1 - c0
        tm["render"] += c2 - c1
        tm["read"] += c3 - c2
        tm["out"] += c4 - c3
        if n % 30 == 0:
            print("frame", f, "wall", round(time.time() - t0, 1), "water drawn", wstate.get("n"), flush=True)
    el = time.time() - t1
    print("worst link pose error, render vs physics log:", round(worst, 3), "mm", flush=True)
    print(f"SPEED {len(frames)} frames in {el:.1f} s = {len(frames) / el:.2f} frames/s; per frame ms " + " ".join(f"{k} {1000 * v / len(frames):.0f}" for k, v in tm.items()), flush=True)
    for k, (mp4, pr) in enc.items():
        pr.stdin.close()
        pr.wait()
        print("WROTE", mp4, os.path.getsize(mp4), flush=True)
    app.close()


if __name__ == "__main__":
    _render()
