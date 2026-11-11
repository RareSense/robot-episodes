"""Unitree G1 + Dex3 in the tea room on Isaac Sim PhysX. AI-operated. Replays the recorded MuJoCo joint path as joint drive targets.
No welds, attachments, object teleports, external forces. Motor limits = MJCF actuatorfrcrange. Source USD never edited: all room edits go to the session layer."""
import json, math, os, sys
sys.dont_write_bytecode = True
for p in ("D:/tea_robot/isaac_h12/pylib", "D:/tea_robot/isaac_delivery", "D:/tea_robot/isaac_delivery/props"):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import omni.physics.tensors
import omni.usd
from omni.physx import get_physx_interface, get_physx_simulation_interface
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Sdf

ROOM = "D:/tea_robot/tea_robot_handoff/project/tea_twin/scene_v3/environment.usda"
CORR = "D:/tea_robot/tea_robot_handoff/project/tea_twin/scene_v3/corrections.json"
USD = "E:/g1_isaac/usd/g1_with_hands/g1_with_hands.usda"
DT = 1 / 600
PRIM = "/World/g1"
DS = "D:/tea_robot/work/dataset/"
SH = "D:/tea_robot/work/isaac/shots/"
TASKS = {"mug": ("step1_mug", SH + "step1"), "teabag": ("step2_teabag", SH + "tb8h"), "pour": ("step3_pour", "D:/tea_robot/work/agents/kettle/shots/pour35"),
         "stir": ("step4_stir", SH + "stir34h"), "cup": ("step5_cup", SH + "cupg2h")}
OBJ = {"kettle": "object_0295", "milk_powder_jar": "object_0309", "mug": "object_0316"}
MASS = {"object_0295": 1.25, "object_0316": 0.30, "object_0309": 0.45}
LIMIT = {"hip_pitch": 88, "hip_roll": 139, "hip_yaw": 88, "knee": 139, "ankle": 50, "waist_yaw": 88, "waist_roll": 50, "waist_pitch": 50, "shoulder": 25, "elbow": 25, "wrist_roll": 25, "wrist_pitch": 5, "wrist_yaw": 5,
         "thumb_0": 2.45, "thumb_1": 1.4, "thumb_2": 1.4, "middle": 1.4, "index": 1.4}
GAIN = {"hip": (3000, 150), "knee": (3000, 150), "ankle": (3000, 200), "waist": (2500, 120), "shoulder": (600, 15), "elbow": (600, 15), "wrist_roll": (300, 6), "wrist_pitch": (200, 3), "wrist_yaw": (200, 3),
        "thumb": (40, .6), "middle": (40, .6), "index": (40, .6)}


def scene_fix(sim, st):
    """Same corrections as tea_sim.py, in the session layer: open tubs, kettle handle loop, rack roof strip off. Contact materials as the MJCF: hands friction 1.0 (MuJoCo takes the pair maximum), soles 0.6."""
    import props
    objs = sim.objs
    props._open_tub(st, "object_0306", 0.048, 0.0363, 0.090)
    props._open_tub(st, "object_0307", 0.047, 0.0363, 0.045)
    UsdPhysics.CollisionAPI(st.GetPrimAtPath(objs["object_0310"].GetPath().pathString + "/collision_06")).CreateCollisionEnabledAttr().Set(False)
    kp = objs["object_0295"].GetPath().pathString
    for k in ("02", "03", "04"):
        UsdPhysics.CollisionAPI(st.GetPrimAtPath(f"{kp}/collision_{k}")).CreateCollisionEnabledAttr().Set(False)
    km = props._material(st, "kettle_handle", 0.5, (0.1, 0.1, 0.1))
    for n, pos, half in (("handle_bar", (0.126, 0.0, 0.125), (0.012, 0.011, 0.055)), ("handle_top", (0.097, 0.0, 0.1925), (0.039, 0.010, 0.0125)), ("handle_bottom", (0.100, 0.0, 0.0525), (0.030, 0.011, 0.0175))):
        props._box(st, f"{kp}/{n}", pos, half, km, purpose="guide")
    UsdPhysics.MassAPI.Apply(objs["object_0295"]).CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, .075))

    def mat(name, mu, comb):
        m = UsdShade.Material.Define(st, "/World/PhysMat/" + name)
        a = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        a.CreateStaticFrictionAttr(mu); a.CreateDynamicFrictionAttr(mu); a.CreateRestitutionAttr(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim()).CreateFrictionCombineModeAttr(comb)
        return m
    hand, sole = mat("hand", 1.0, "max"), mat("sole", .6, "min")
    n = 0
    for p in Usd.PrimRange(sim.robot, Usd.TraverseInstanceProxies()):
        if p.IsInstanceProxy() or not p.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        nm = p.GetName()
        if "_hand_" in nm or "wrist_yaw" in nm:
            UsdShade.MaterialBindingAPI.Apply(p).Bind(hand, UsdShade.Tokens.strongerThanDescendants, "physics"); n += 1
        elif "ankle_roll" in nm:
            UsdShade.MaterialBindingAPI.Apply(p).Bind(sole, UsdShade.Tokens.strongerThanDescendants, "physics"); n += 1
    print("contact materials on", n, "links", flush=True)


def add_spoon(sim, st):
    """The stir spoon as in the recording: props_spoon sizes and mass, at the recorded rest pose (session layer)."""
    import props
    P = sim.rec.prop0
    UsdGeom.Xform.Define(st, props.ROOT)
    steel = props._material(st, "steel", 0.4, (0.78, 0.79, 0.81), metallic=1.0, rough=0.3)
    p = props.ROOT + "/stir_spoon"
    handle_mass = sum((x1 - x0) * 2 * hw * 2 * props.HT for _, x0, x1, hw in props.SEGMENTS) * props.STEEL
    pos, q = P["pos"][0], P["quat"][0]
    props._body(st, p, pos, Gf.Quatd(*map(float, q)).GetNormalized(), props.BOWL_MASS + handle_mass)
    props._ellipsoid_mesh(st, p + "/bowl", (0.0175, 0, 0), (0.0175, 0.0125, 0.0040), steel)
    props._handle(st, p, steel)
    sim.prop_paths = [p]


def add_teabag(sim, st):
    """Bag and tag as free bodies with props_teabag sizes and masses, at the recorded poses, joined by a limp 12 cm distance joint (no string geometry, as the H1-2 props).
    The recorded 20-segment string went NaN at 5 mg per segment on GPU PhysX and stood stiff at 20 mg, levering the bag in the pinch. The string drawn in the video is a render-only curve between staple and tag."""
    import props
    P = sim.rec.prop0
    UsdGeom.Xform.Define(st, props.ROOT)
    paper = props._material(st, "bag_paper", 0.6, (0.93, 0.90, 0.82), rough=0.9)
    card = props._material(st, "tag_card", 0.9, (0.85, 0.12, 0.10), rough=0.7)
    pb, pt = props.ROOT + "/prop_teabag_bag", props.ROOT + "/prop_teabag_tag"
    props._body(st, pb, P["pos"][0], Gf.Quatd(*map(float, P["quat"][0])).GetNormalized(), .00205)
    props._box(st, pb + "/geom", (0, 0, 0), props.BAGF_HALF, paper)
    props._body(st, pt, P["pos"][-1], Gf.Quatd(*map(float, P["quat"][-1])).GetNormalized(), .00055)
    props._box(st, pt + "/geom", props.TAG_CENTRE, props.TAG_HALF, card)
    qa = P["quat"][0]
    sim.staple = rmat(np.r_[qa[1:], qa[0]]).T @ (P["pos"][1] - P["pos"][0])
    j = UsdPhysics.DistanceJoint.Define(st, props.ROOT + "/teabag_string")
    j.CreateBody0Rel().SetTargets([pb])
    j.CreateBody1Rel().SetTargets([pt])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, sim.staple)))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateMinDistanceAttr(-1.0)
    j.CreateMaxDistanceAttr(.12)
    sim.prop_paths = [pb, pt]


def add_water(sim, st):
    """GPU PhysX particle water in the kettle, from the H1-2 water tool (read only). Must run before scene_fix: it reads the kettle's mesh hulls."""
    sys.path.insert(0, "D:/tea_robot/isaac_delivery/fluids")
    import fluids
    sim.F = fluids.Fluids(fill_ml=float(os.environ.get("WATER_ML", "400")))
    if os.environ.get("WATER_LEVEL"):
        # The take starts with the kettle already tilted. The tool fills it as if upright, so the first second poured 20.9 ml out of the spout. Here the same lattice is cut level with gravity at the start tilt.
        wl = fluids.wl
        orig = wl.WaterLayer._fill_points
        R_ = sim.rec
        q = R_.obj[sim.i0, R_.on.index("kettle"), 3:7]
        Rk = rmat(np.r_[q[1:], q[0]])

        def level(self_, ml):
            pts = orig(self_, float(os.environ.get("WATER_LATTICE", "900")))
            n = int(round(ml * 1e-6 / self_.vp))
            zw = pts @ Rk[2]
            keep = np.argsort(zw)[:n]
            print("[water] level fill", n, "of", len(pts), "lattice points; top in world", round(float(zw[keep].max()), 3), flush=True)
            return pts[keep]
        wl.WaterLayer._fill_points = level
    sim.F.pre_physics(st, sim.scene)


def key(tbl, name):
    for k, v in tbl.items():
        if k in name:
            return v
    raise KeyError(name)


def rmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rpy(q):
    x, y, z, w = q
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), math.asin(max(-1, min(1, 2 * (w * y - z * x)))), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Rec:
    """The recorded MuJoCo take for one task."""
    def __init__(self, task):
        d, sh = TASKS[task]
        z = np.load(DS + d + "/motion.npz", allow_pickle=True)
        self.t = z["t"].astype(float)
        self.jn = [str(x) for x in z["joint_names"]]
        self.q, self.dq, self.base = z["joint_pos"].astype(float), z["joint_vel"].astype(float), z["base_pos_quat"].astype(float)
        mn = [str(x) for x in z["motor_names"]]
        self.tau = z["motor_torque"].astype(float)[:, [mn.index(n) for n in self.jn]]     # magnitudes, in joint order
        self.on = [str(x) for x in z["object_names"]]
        self.obj = z["object_pos_quat"].astype(float)
        self.sh = sh
        S = np.load(sh + "/shot.npz", allow_pickle=True)
        self.bn = [str(x) for x in S["robot_names"]]
        self.xpos, self.xquat = S["xpos"].astype(float), S["xquat"].astype(float)
        self.eye, self.film_eye, self.film_tgt, self.film_hfov = S["eye_mats"], S["film_eye"], S["film_tgt"], float(S["film_hfov"])
        self.obj_paths, self.obj_mats = [str(x) for x in S["obj_paths"]], S["obj_mats"]
        self.prop_names = [str(x) for x in S["prop_names"]]
        self.prop0 = dict(pos=S["prop_xpos"][0].astype(float), quat=S["prop_xquat"][0].astype(float))
        self.prop_xpos, self.prop_xquat = S["prop_xpos"], S["prop_xquat"]
        self.prop_names = [str(x) for x in S["prop_names"]]
        self.prop0 = dict(pos=S["prop_xpos"][0].astype(float), quat=S["prop_xquat"][0].astype(float))
        self.prop_xpos, self.prop_xquat = S["prop_xpos"], S["prop_xquat"]

    def at(self, t):
        u = (t - self.t[0]) * 60.0
        i = int(np.clip(math.floor(u), 0, len(self.t) - 2))
        a = float(np.clip(u - i, 0, 1))
        return self.q[i] * (1 - a) + self.q[i + 1] * a, self.dq[i] * (1 - a) + self.dq[i + 1] * a, i


class Sim:
    def __init__(self, app, task, gpu=False, extra=(scene_fix,), iters=None):
        self.app, self.task = app, task
        self.rec = R = Rec(task)
        ts = float(os.environ.get("START_T", "0"))          # stance start: the take begins at this recorded moment, then nothing writes state
        self.i0 = int(np.clip(round((ts - R.t[0]) * 60), 0, len(R.t) - 1)) if ts > 0 else 0
        R.prop0 = dict(pos=np.array(R.prop_xpos[self.i0], float), quat=np.array(R.prop_xquat[self.i0], float))
        ctx = omni.usd.get_context()
        ctx.open_stage(ROOM)
        for _ in range(5):
            app.update()
        st = self.stage = ctx.get_stage()
        st.SetEditTarget(st.GetSessionLayer())
        self.objs = {}
        for p in st.Traverse():
            if p.GetName().startswith("object_") and p.GetTypeName() == "Xform":
                self.objs.setdefault(p.GetName(), p)
        corr = {k: v for k, v in json.load(open(CORR)).items() if not k.startswith("_")}
        for i, n in enumerate(R.on):                                # start of this take: objects where the recording starts (scene set-up before physics loads)
            if n in OBJ:
                corr[OBJ[n]] = list(R.obj[self.i0, i])
        for k, v in corr.items():
            p = self.objs.get(k)
            if p is None or len(v) != 7:
                continue
            xf = UsdGeom.Xformable(p)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*v[:3]))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(v[3], v[4], v[5], v[6]))
        for k, m in MASS.items():
            if k == "object_0295" and os.environ.get("WATER"):
                m = m - float(os.environ.get("WATER_ML", "400")) / 1000.0     # 1.25 kg includes 0.4 kg of water; with real particles the water carries its own mass
            UsdPhysics.MassAPI.Apply(self.objs[k]).CreateMassAttr().Set(float(m))
        sc = self.scene = next(p for p in st.Traverse() if p.IsA(UsdPhysics.Scene))
        px = PhysxSchema.PhysxSceneAPI.Apply(sc)
        px.CreateEnableGPUDynamicsAttr(bool(gpu))
        px.CreateBroadphaseTypeAttr("GPU" if gpu else "MBP")
        px.CreateSolverTypeAttr("TGS")
        px.CreateTimeStepsPerSecondAttr(600)
        r = self.robot = st.DefinePrim(PRIM, "Xform")
        r.GetReferences().AddReference(USD)
        vs = r.GetVariantSets().GetVariantSet("Physics")
        print("Physics variants", vs.GetVariantNames(), flush=True)
        vs.SetVariantSelection("physx")
        app.update()
        self.prop_paths = []
        self.F = None
        pre = (add_water,) if os.environ.get("WATER") else ()
        for f in pre + tuple(extra) + ({"stir": (add_spoon,), "teabag": (add_teabag,)}.get(task, ())):
            f(self, st)
        root = next(p.GetPath().pathString for p in Usd.PrimRange(r, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.ArticulationRootAPI))
        print("art root", root, flush=True)
        rp = st.GetPrimAtPath(root)
        if not rp.IsInstanceProxy():
            api = PhysxSchema.PhysxArticulationAPI.Apply(rp)
            api.CreateSolverPositionIterationCountAttr(int(iters or 16))
            api.CreateSolverVelocityIterationCountAttr(1)
            api.CreateEnabledSelfCollisionsAttr(False)
        sid = UsdUtils.StageCache.Get().Insert(st).ToLongInt()
        self.sim = get_physx_simulation_interface()
        self.sim.attach_stage(sid)
        get_physx_interface().force_load_physics_from_usd()
        get_physx_interface().start_simulation()
        self.sim.simulate(DT, 0.0)
        self.sim.fetch_results()
        self.sv = omni.physics.tensors.create_simulation_view("numpy", stage_id=sid)
        self.sv.set_subspace_roots("/")
        a = self.art = self.sv.create_articulation_view(root)
        self.names = list(a.shared_metatype.dof_names)
        self.n = a.max_dofs
        self.idx = np.arange(1, dtype=np.uint32)
        self.links = [p.rsplit("/", 1)[-1] for p in a.link_paths[0]]
        self.ix = {n: i for i, n in enumerate(self.names)}
        self.lim = np.array([key(LIMIT, n) for n in self.names], float)
        self.kp = np.array([key(GAIN, n)[0] for n in self.names], float)
        self.kd = np.array([key(GAIN, n)[1] for n in self.names], float)
        self.perm = np.array([R.jn.index(n) for n in self.names])     # dof order <- recorded order
        a.set_dof_armatures(self.f(np.full(self.n, .01)), self.idx)
        try:
            fp = np.array(a.get_dof_friction_properties())
            fp[0, :, 0], fp[0, :, 1] = .3, .3
            a.set_dof_friction_properties(fp, self.idx)
            print("joint friction 0.3 Nm as MJCF", flush=True)
        except Exception as e:
            print("WARNING joint friction not set", repr(e)[:200], flush=True)
        a.set_dof_max_forces(self.f(self.lim), self.idx)
        a.set_dof_stiffnesses(self.f(self.kp), self.idx)
        a.set_dof_dampings(self.f(self.kd), self.idx)
        self.mass = a.get_masses()[0].astype(float)
        print("robot mass", round(float(self.mass.sum()), 3), flush=True)
        self.tr = float(R.t[self.i0])                                 # time in the recording
        self.t = 0.0
        self.off = np.zeros(self.n)                                   # live fix: target offset added to the recorded target
        self.bodies = {}
        if self.F is not None:
            self.F.attach(self.sv)
        self.prop_view = self.sv.create_rigid_body_view(self.prop_paths) if self.prop_paths else None
        if self.prop_view is not None:
            act = list(map(str, self.prop_view.prim_paths))
            self.prop_order = [act.index(x) for x in self.prop_paths]
        self.spawn()

    def f(self, v):
        return np.asarray(v, np.float32)[None]

    def spawn(self, dz=0.0):
        R, a = self.rec, self.art
        q0 = R.q[self.i0][self.perm]
        self.qd, self.vd = q0.copy(), np.zeros(self.n)
        a.set_dof_positions(self.f(q0), self.idx)
        a.set_dof_velocities(self.f(np.zeros(self.n)), self.idx)
        a.set_dof_position_targets(self.f(q0), self.idx)
        b = R.base[self.i0]
        tf = a.get_root_transforms().copy()
        tf[0] = [b[0], b[1], b[2] + dz, b[4], b[5], b[6], b[3]]
        a.set_root_transforms(tf, self.idx)
        a.set_root_velocities(np.zeros((1, 6), np.float32), self.idx)

    def q(self):
        return self.art.get_dof_positions()[0].astype(float)

    def v(self):
        return self.art.get_dof_velocities()[0].astype(float)

    def link_poses(self):
        return self.art.get_link_transforms()[0].astype(float)

    def link(self, name):
        return self.link_poses()[self.links.index(name)]

    def body(self, k):
        if k not in self.bodies:
            path = k if k.startswith("/") else self.objs[k].GetPath().pathString
            self.bodies[k] = self.sv.create_rigid_body_view(path)
        return self.bodies[k]

    def obj(self, k):
        return self.body(k).get_transforms()[0].astype(float)

    def props(self):
        """(n,7) x y z qx qy qz qw in recorded prop order"""
        if self.prop_view is None:
            return np.zeros((0, 7))
        A = np.asarray(self.prop_view.get_transforms())[self.prop_order].astype(float)
        if self.task != "teabag":
            return A
        b, g = A[0], A[1]                                   # 20 string pieces for drawing only: a sagging curve from staple to tag, no physics
        a = b[:3] + rmat(b[3:7]) @ self.staple
        c = g[:3]
        n = 20
        L = float(np.linalg.norm(c - a))
        sag = .5 * math.sqrt(max(0., .12 ** 2 - L ** 2))
        out = [b]
        pts = [a + (c - a) * (k / n) - np.array([0, 0, sag * 4 * (k / n) * (1 - k / n)]) for k in range(n + 1)]
        for k in range(n):
            d = pts[k + 1] - pts[k]
            d = d / max(1e-9, np.linalg.norm(d))
            ax = np.cross([1., 0, 0], d)
            w = 1 + d[0]
            q = np.r_[ax, w]
            q = q / max(1e-9, np.linalg.norm(q))
            out.append(np.r_[pts[k], q])
        out.append(g)
        return np.array(out)

    def props(self):
        """(n,7) x y z qx qy qz qw in recorded prop order"""
        if self.prop_view is None:
            return np.zeros((0, 7))
        A = np.asarray(self.prop_view.get_transforms())[self.prop_order].astype(float)
        if self.task != "teabag":
            return A
        b, g = A[0], A[1]                                   # 20 string pieces for drawing only: a sagging curve from staple to tag, no physics
        a = b[:3] + rmat(b[3:7]) @ self.staple
        c = g[:3]
        n = 20
        L = float(np.linalg.norm(c - a))
        sag = .5 * math.sqrt(max(0., .12 ** 2 - L ** 2))
        out = [b]
        pts = [a + (c - a) * (k / n) - np.array([0, 0, sag * 4 * (k / n) * (1 - k / n)]) for k in range(n + 1)]
        for k in range(n):
            d = pts[k + 1] - pts[k]
            d = d / max(1e-9, np.linalg.norm(d))
            ax = np.cross([1., 0, 0], d)
            w = 1 + d[0]
            q = np.r_[ax, w]
            q = q / max(1e-9, np.linalg.norm(q))
            out.append(np.r_[pts[k], q])
        out.append(g)
        return np.array(out)

    def step(self):
        self.art.set_dof_position_targets(self.f(self.qd + self.off), self.idx)
        self.art.set_dof_velocity_targets(self.f(self.vd), self.idx)
        self.sim.simulate(DT, self.t)
        self.sim.fetch_results()
        self.t += DT
