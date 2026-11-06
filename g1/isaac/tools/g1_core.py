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
    """Bag, 20-segment string and tag as in the recording (props_teabag sizes and masses, recorded start poses). One small free articulation, spherical joints, held by nothing but contacts."""
    import props
    P = sim.rec.prop0
    UsdGeom.Xform.Define(st, props.ROOT)
    paper = props._material(st, "bag_paper", 0.6, (0.93, 0.90, 0.82), rough=0.9)
    card = props._material(st, "tag_card", 0.9, (0.85, 0.12, 0.10), rough=0.7)
    thread = props._material(st, "thread", 0.5, (0.96, 0.96, 0.94), rough=0.9)
    names = sim.rec.prop_names
    paths = []
    for i, n in enumerate(names):
        path = props.ROOT + "/" + n.replace("env/", "")
        pos, q = P["pos"][i], P["quat"][i]
        if n.endswith("bag"):
            b = props._body(st, path, pos, Gf.Quatd(*map(float, q)).GetNormalized(), .002)
            props._box(st, path + "/geom", (0, 0, 0), props.BAGF_HALF, paper)
        elif n.endswith("tag"):
            b = props._body(st, path, pos, Gf.Quatd(*map(float, q)).GetNormalized(), .0005)
            props._box(st, path + "/geom", props.TAG_CENTRE, props.TAG_HALF, card)
        else:
            b = props._body(st, path, pos, Gf.Quatd(*map(float, q)).GetNormalized(), 5e-6)
            UsdPhysics.MassAPI(b).CreateDiagonalInertiaAttr().Set(Gf.Vec3f(2e-9, 2e-9, 2e-9))
            nxt = P["pos"][i + 1]
            L = float(np.linalg.norm(nxt - pos))
            c = UsdGeom.Capsule.Define(st, path + "/geom")
            c.CreateAxisAttr("X"); c.CreateRadiusAttr(.0004); c.CreateHeightAttr(L)
            c.AddTranslateOp().Set(Gf.Vec3d(L / 2, 0, 0))
            props._collider(c.GetPrim())
            props._bind(c.GetPrim(), thread)
        paths.append(path)
    UsdPhysics.ArticulationRootAPI.Apply(st.GetPrimAtPath(paths[0]))
    pa = PhysxSchema.PhysxArticulationAPI.Apply(st.GetPrimAtPath(paths[0]))
    pa.CreateSolverPositionIterationCountAttr(16)
    pa.CreateEnabledSelfCollisionsAttr(False)
    for i in range(1, len(paths)):
        qa, qb = P["quat"][i - 1], P["quat"][i]
        Ra = rmat(np.r_[qa[1:], qa[0]])
        lp = Ra.T @ (P["pos"][i] - P["pos"][i - 1])
        Qa, Qb = Gf.Quatd(*map(float, qa)).GetNormalized(), Gf.Quatd(*map(float, qb)).GetNormalized()
        rel = (Qa.GetInverse() * Qb)
        j = UsdPhysics.SphericalJoint.Define(st, paths[i] + "/j")
        j.CreateBody0Rel().SetTargets([paths[i - 1]]); j.CreateBody1Rel().SetTargets([paths[i]])
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, lp))); j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(rel)); j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    sim.prop_paths = paths


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
    """Bag, 20-segment string and tag as in the recording (props_teabag sizes and masses, recorded start poses). One small free articulation, spherical joints, held by nothing but contacts."""
    import props
    P = sim.rec.prop0
    UsdGeom.Xform.Define(st, props.ROOT)
    paper = props._material(st, "bag_paper", 0.6, (0.93, 0.90, 0.82), rough=0.9)
    card = props._material(st, "tag_card", 0.9, (0.85, 0.12, 0.10), rough=0.7)
    thread = props._material(st, "thread", 0.5, (0.96, 0.96, 0.94), rough=0.9)
    names = sim.rec.prop_names
    paths = []
    for i, n in enumerate(names):
        path = props.ROOT + "/" + n.replace("env/", "")
        pos, q = P["pos"][i], P["quat"][i]
        Q = Gf.Quatd(*map(float, q)).GetNormalized()
        if n.endswith("bag"):
            b = props._body(st, path, pos, Q, .002)
            props._box(st, path + "/geom", (0, 0, 0), props.BAGF_HALF, paper)
        elif n.endswith("tag"):
            b = props._body(st, path, pos, Q, .0005)
            props._box(st, path + "/geom", props.TAG_CENTRE, props.TAG_HALF, card)
        else:
            b = props._body(st, path, pos, Q, 5e-6)
            UsdPhysics.MassAPI(b).CreateDiagonalInertiaAttr().Set(Gf.Vec3f(2e-9, 2e-9, 2e-9))
            L = float(np.linalg.norm(P["pos"][i + 1] - pos))
            c = UsdGeom.Capsule.Define(st, path + "/geom")
            c.CreateAxisAttr("X")
            c.CreateRadiusAttr(.0004)
            c.CreateHeightAttr(L)
            c.AddTranslateOp().Set(Gf.Vec3d(L / 2, 0, 0))
            props._collider(c.GetPrim())
            props._bind(c.GetPrim(), thread)
        paths.append(path)
    UsdPhysics.ArticulationRootAPI.Apply(st.GetPrimAtPath(paths[0]))
    pa = PhysxSchema.PhysxArticulationAPI.Apply(st.GetPrimAtPath(paths[0]))
    pa.CreateSolverPositionIterationCountAttr(16)
    pa.CreateEnabledSelfCollisionsAttr(False)
    for i in range(1, len(paths)):
        qa, qb = P["quat"][i - 1], P["quat"][i]
        Ra = rmat(np.r_[qa[1:], qa[0]])
        lp = Ra.T @ (P["pos"][i] - P["pos"][i - 1])
        Qa, Qb = Gf.Quatd(*map(float, qa)).GetNormalized(), Gf.Quatd(*map(float, qb)).GetNormalized()
        rel = Qa.GetInverse() * Qb
        j = UsdPhysics.SphericalJoint.Define(st, paths[i] + "/j")
        j.CreateBody0Rel().SetTargets([paths[i - 1]])
        j.CreateBody1Rel().SetTargets([paths[i]])
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, lp)))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(rel))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    sim.prop_paths = paths


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
                corr[OBJ[n]] = list(R.obj[0, i])
        for k, v in corr.items():
            p = self.objs.get(k)
            if p is None or len(v) != 7:
                continue
            xf = UsdGeom.Xformable(p)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*v[:3]))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(v[3], v[4], v[5], v[6]))
        for k, m in MASS.items():
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
        for f in tuple(extra) + ({"stir": (add_spoon,), "teabag": (add_teabag,)}.get(task, ())):
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
        self.tr = float(R.t[0])                                       # time in the recording
        self.t = 0.0
        self.off = np.zeros(self.n)                                   # live fix: target offset added to the recorded target
        self.bodies = {}
        self.prop_view = self.sv.create_rigid_body_view(self.prop_paths) if self.prop_paths else None
        if self.prop_view is not None:
            act = list(map(str, self.prop_view.prim_paths))
            self.prop_order = [act.index(x) for x in self.prop_paths]
        self.spawn()

    def f(self, v):
        return np.asarray(v, np.float32)[None]

    def spawn(self, dz=0.0):
        R, a = self.rec, self.art
        q0 = R.q[0][self.perm]
        self.qd, self.vd = q0.copy(), np.zeros(self.n)
        a.set_dof_positions(self.f(q0), self.idx)
        a.set_dof_velocities(self.f(np.zeros(self.n)), self.idx)
        a.set_dof_position_targets(self.f(q0), self.idx)
        b = R.base[0]
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
        return np.asarray(self.prop_view.get_transforms())[self.prop_order].astype(float) if self.prop_view is not None else np.zeros((0, 7))

    def props(self):
        """(n,7) x y z qx qy qz qw in recorded prop order"""
        return np.asarray(self.prop_view.get_transforms())[self.prop_order].astype(float) if self.prop_view is not None else np.zeros((0, 7))

    def step(self):
        self.art.set_dof_position_targets(self.f(self.qd + self.off), self.idx)
        self.art.set_dof_velocity_targets(self.f(self.vd), self.idx)
        self.sim.simulate(DT, self.t)
        self.sim.fetch_results()
        self.t += DT
