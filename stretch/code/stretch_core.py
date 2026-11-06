"""Stretch 3 live sim in the tea room. Reuses props, gates, record, fluids from D:/tea_robot/isaac_delivery by import only.
No welds, attachments, object teleports, external forces or state writes after the spawn.
Telescope: 4 real prismatic joints, coupled by PhysX mimic constraints as in the MJCF equalities. Fingers: real revolute joints, mimic gear 10 to the gripper slide.
Limits. From the MJCF: lift 70 N, arm 56 N at the tip (= 224 N on the one driven telescope joint, by virtual work).
From motor data, because the MJCF leaves them blank (labelled): wrist pitch 6.0 Nm (XM540-W270 at the 2.5 A setting), roll 1.8 Nm (XM430-W350 at 1 A), yaw 4.6 Nm (XC430-W240 x 2.4).
Gripper slide force: ASSUMED, real gripper force unverified, see GRIP_N."""
import json
import math
import os
import sys

sys.dont_write_bytecode = True
for p in ("D:/tea_robot/isaac_h12/pylib", "D:/tea_robot/isaac_delivery", "D:/tea_robot/isaac_delivery/props"):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import omni.physics.tensors
import omni.usd
from omni.physx import get_physx_interface, get_physx_simulation_interface
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

ROOT = "D:/tea_robot/stretch"
ROOM = "D:/tea_robot/tea_robot_handoff/project/tea_twin/scene_v3/environment.usda"
CORR = "D:/tea_robot/tea_robot_handoff/project/tea_twin/scene_v3/corrections.json"
MJCF = "E:/robot_controller_big/mujoco_menagerie/hello_robot_stretch_3/stretch.xml"
USD = ROOT + "/usd2/stretch/stretch.usda"
DT = 1 / 600
PRIM = "/World/stretch"
TELE = ["joint_arm_l3", "joint_arm_l2", "joint_arm_l1", "joint_arm_l0"]
GRIP_N = 142.0      # ASSUMED slide force = about 40 N at each pad (slide to pad ratio 3.56 from finger angle = 10 x slide, 178 mm finger)
LIMITS = {"joint_lift": 70.0, "joint_arm_l3": 224.0, "joint_wrist_yaw": 4.6, "joint_wrist_pitch": 6.0, "joint_wrist_roll": 1.8, "joint_gripper_slide": GRIP_N,
          "joint_head_pan": 2.0, "joint_head_tilt": 2.0, "joint_left_wheel": 33.3, "joint_right_wheel": 33.3}
GAINS = {"joint_lift": (400., 135.), "joint_arm_l3": (2400., 160.), "joint_wrist_yaw": (20., 2.), "joint_wrist_pitch": (50., 3.), "joint_wrist_roll": (20., 2.),
         "joint_gripper_slide": (4000., 124.), "joint_head_pan": (10., 1.), "joint_head_tilt": (10., 1.)}


def rmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def couple(st, root):
    """MJCF equalities as PhysX mimic constraints: position + gearing * reference = 0. The instance token must match a revolute joint's own axis."""
    J = {p.GetName(): p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.IsA(UsdPhysics.Joint)}
    out = []
    for name, ref, ratio in (("joint_arm_l2", "joint_arm_l3", 1.), ("joint_arm_l1", "joint_arm_l3", 1.), ("joint_arm_l0", "joint_arm_l3", 1.),
                             ("joint_gripper_finger_left_open", "joint_gripper_slide", 10.), ("joint_gripper_finger_right_open", "joint_gripper_slide", 10.)):
        jp, rp = J[name], J[ref]
        ax = "rot" + jp.GetAttribute("physics:axis").Get() if jp.IsA(UsdPhysics.RevoluteJoint) else "rotX"
        api = PhysxSchema.PhysxMimicJointAPI.Apply(jp, ax)
        api.GetReferenceJointRel().SetTargets([rp.GetPath()])
        api.CreateReferenceJointAxisAttr("rotX")
        # measured: slide 0.0275 m gave finger 0.0048 rad = 0.275 deg, so a revolute-from-prismatic gearing is deg per metre
        api.CreateGearingAttr(-ratio * (180.0 / math.pi if jp.IsA(UsdPhysics.RevoluteJoint) else 1.0))
        api.CreateOffsetAttr(0.0)
        out.append((name, ax))
    print("mimic constraints", out, flush=True)
    return out


def mjcf_masses():
    import re
    out = {}
    for line in open(ROOT + "/analysis/model_dump.txt", encoding="utf-8"):
        m_ = re.match(r"^(\S+)\s+<- (\S+)\s+m\s+([0-9.]+)", line)
        if m_:
            out[m_.group(1)] = float(m_.group(3))
    return out


def fix_model(st, root):
    """Importer gave massless frames 1 kg and added mass from colliders. Author the MJCF mass on every body; massless frames 1 g (labelled)."""
    mj = mjcf_masses()
    n = filled = 0
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not p.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        m = mj.get(p.GetName())
        if m is None:
            print("no MJCF mass for", p.GetName(), flush=True)
            continue
        api = UsdPhysics.MassAPI.Apply(p)
        if m < 1e-4:
            api.CreateMassAttr().Set(.001)
            api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1e-7, 1e-7, 1e-7))
            filled += 1
        else:
            api.CreateMassAttr().Set(float(m))
        n += 1
    print("masses authored on", n, "bodies,", filled, "massless frames at 1 g", flush=True)
    # caster: frictionless in the MJCF (condim 1)
    mat = UsdShade.Material.Define(st, "/World/StretchCaster")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(0.0)
    pm.CreateDynamicFrictionAttr(0.0)
    pm.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr("min")
    c = st.GetPrimAtPath(PRIM + "/Geometry/base_link/Sphere")
    UsdShade.MaterialBindingAPI.Apply(c).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
    # pad gimbal: the MJCF has x and y hinges only
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if p.GetName().startswith("rubber_") and p.IsA(UsdPhysics.Joint):
            lim = UsdPhysics.LimitAPI.Apply(p, "rotZ")
            lim.CreateLowAttr(1.0)
            lim.CreateHighAttr(-1.0)
            # measured: pads tumbled freely. A D6 joint needs a drive per axis; the importer left only "angular", which does nothing here.
            # MJCF: stiffness 1 Nm/rad, damping 0.01. USD angular drives are per degree.
            for ax in ("rotX", "rotY"):
                dr = UsdPhysics.DriveAPI.Apply(p, ax)
                dr.CreateTypeAttr("force")
                dr.CreateStiffnessAttr(math.pi / 180.0)
                dr.CreateDampingAttr(.01 * math.pi / 180.0)
                dr.CreateTargetPositionAttr(0.0)
                dr.CreateMaxForceAttr(5.0)
            print("rotZ locked, rotX rotY springs on", p.GetName(), flush=True)


class Sim:
    """Same surface as h12_physx.Sim where gates.py and record.py read it: stage, objs, names, art, sv, links, t, step, pose, link_poses."""

    def __init__(self, app, xy, yaw_deg=180.0, z=.002, q=None, pre=(), add_props=True, gpu=False, iters=None, room=None):
        self.app = app
        ctx = omni.usd.get_context()
        if room is None or isinstance(room, str):
            ctx.open_stage(room or ROOM)              # a wrapper stage on a copy of the kitchen, or the room itself
        else:
            ctx.new_stage()                         # simple room: room(stage) authors floor, physics scene and lights
        for _ in range(5):
            app.update()
        st = self.stage = ctx.get_stage()
        if callable(room):
            room(st)
        st.SetEditTarget(st.GetSessionLayer())
        self.objs = {}
        for p in st.Traverse():
            if p.GetName().startswith("object_") and p.GetTypeName() == "Xform":
                self.objs.setdefault(p.GetName(), p)
        for k, v in (json.load(open(CORR)) if not callable(room) else {}).items():
            p = self.objs.get(k)
            if k.startswith("_") or p is None or len(v) != 7:
                continue
            xf = UsdGeom.Xformable(p)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*v[:3]))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(v[3], v[4], v[5], v[6]))
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
        self.mimic = couple(st, r)
        fix_model(st, r)
        if add_props:
            import props
            props.install(st)
        for f in pre:
            f(st)
        root = next(p.GetPath().pathString for p in Usd.PrimRange(r) if p.HasAPI(UsdPhysics.ArticulationRootAPI))
        if iters:
            PhysxSchema.PhysxArticulationAPI.Apply(st.GetPrimAtPath(root)).CreateSolverPositionIterationCountAttr(int(iters))
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
        self.lim = np.ones(self.n) * 5.0
        self.kp = np.zeros(self.n)
        self.kd = np.zeros(self.n)
        for k, v in LIMITS.items():
            if k in self.ix:
                self.lim[self.ix[k]] = v
        for k, (p_, d_) in GAINS.items():
            if k in self.ix:
                self.kp[self.ix[k]], self.kd[self.ix[k]] = p_, d_
        for k in ("joint_left_wheel", "joint_right_wheel"):
            self.kd[self.ix[k]] = 30.0                    # velocity drive, kv from the MJCF wheel class scaled by gear
        arm_, fr_ = np.zeros(self.n), np.zeros(self.n)
        for k, (am, fc) in {"joint_left_wheel": (.1, 1.), "joint_right_wheel": (.1, 1.), "joint_lift": (.01, 1.5), "joint_arm_l3": (.01, 1.5), "joint_arm_l2": (.01, 1.5), "joint_arm_l1": (.01, 1.5), "joint_arm_l0": (.01, 1.5),
                            "joint_wrist_yaw": (.01, .1), "joint_wrist_pitch": (.01, .1), "joint_wrist_roll": (.01, .1), "joint_gripper_slide": (.1, 0.)}.items():
            if k in self.ix:
                arm_[self.ix[k]], fr_[self.ix[k]] = am, fc
        for i, nm in enumerate(self.names):
            if nm.startswith("rubber_"):                  # pad gimbal, MJCF stiffness 1, damping 0.01
                self.kp[i], self.kd[i], self.lim[i] = 1.0, .01, 5.0
        a.set_dof_armatures(self.f(arm_), self.idx)
        try:
            fp = np.array(a.get_dof_friction_properties())   # static, dynamic (force units), viscous. The old coefficient call stalled the telescope.
            fp[0, :, 0], fp[0, :, 1] = fr_, fr_
            a.set_dof_friction_properties(fp, self.idx)
            print("joint friction set in force units", flush=True)
        except Exception as e:
            print("WARNING joint friction NOT set, model differs from the MJCF:", repr(e)[:200], flush=True)
        a.set_dof_max_forces(self.f(self.lim), self.idx)
        a.set_dof_stiffnesses(self.f(self.kp), self.idx)
        a.set_dof_dampings(self.f(self.kd), self.idx)
        q0 = np.zeros(self.n)
        for k, v in (q or {}).items():
            q0[self.ix[k]] = v
        self.qd = q0.copy()
        self.vd = np.zeros(self.n)
        a.set_dof_positions(self.f(q0), self.idx)
        a.set_dof_velocities(self.f(np.zeros(self.n)), self.idx)
        a.set_dof_position_targets(self.f(q0), self.idx)
        tf = a.get_root_transforms().copy()
        y = math.radians(yaw_deg)
        tf[0] = [xy[0], xy[1], z, 0, 0, math.sin(y / 2), math.cos(y / 2)]
        a.set_root_transforms(tf, self.idx)
        a.set_root_velocities(np.zeros((1, 6), np.float32), self.idx)
        self.t = 0.0
        self.bodies = {}

    def f(self, v):
        return np.asarray(v, np.float32)[None]

    def q(self):
        return self.art.get_dof_positions()[0].astype(float)

    def v(self):
        return self.art.get_dof_velocities()[0].astype(float)

    def link_poses(self):
        return self.art.get_link_transforms()[0].astype(float)

    def pose(self):
        r = self.art.get_root_transforms()[0].astype(float)
        return r[:3], r[3:7]

    def link(self, name):
        return self.link_poses()[self.links.index(name)]

    def body(self, key):
        if key not in self.bodies:
            path = key if key.startswith("/") else self.objs[key].GetPath().pathString
            self.bodies[key] = self.sv.create_rigid_body_view(path)
        return self.bodies[key]

    def obj(self, key):
        return self.body(key).get_transforms()[0].astype(float)

    def step(self):
        self.art.set_dof_position_targets(self.f(self.qd), self.idx)
        self.art.set_dof_velocity_targets(self.f(self.vd), self.idx)
        self.sim.simulate(DT, self.t)
        self.sim.fetch_results()
        self.t += DT

    def wait(self, s):
        for _ in range(int(round(s / DT))):
            self.step()

    # --- motion. All of it is joint drive targets; nothing writes state.
    def wheels(self, v, w):
        """Body forward speed v (m/s), turn rate w (rad/s). Wheel radius 0.05, half track 0.1704, from the MJCF."""
        I = self.ix
        self.vd[I["joint_left_wheel"]] = (v - w * .1704) / .05
        self.vd[I["joint_right_wheel"]] = (v + w * .1704) / .05

    def drive_x(self, x, line_y=None, yaw_deg=180.0, vmax=.12, tol=.004, timeout=40.0):
        """Drive along world x with the base at yaw 180 (body forward = -x). Closed loop on heading and on the line y. Slows into the target, stops, returns the error."""
        p0 = self.pose()[0]
        ly = p0[1] if line_y is None else line_y
        n = int(timeout / DT)
        y0 = math.radians(yaw_deg)
        for i in range(n):
            if i % 6 == 0:
                p, q = self.pose()
                e = p[0] - x                       # moving -x is forward
                if abs(e) < tol:
                    break
                v = max(-vmax, min(vmax, 1.2 * e))
                if abs(v) < .012:
                    v = math.copysign(.012, e)
                ey = p[1] - ly
                yerr = (y0 + math.copysign(1., v) * max(-.25, min(.25, 6.0 * ey)) - yaw_of(q) + math.pi) % (2 * math.pi) - math.pi      # at yaw 180: vy = -v * delta
                self.wheels(v, max(-.5, min(.5, 3.0 * yerr)))
            self.step()
        self.wheels(0., 0.)
        self.wait(.5)
        p, q = self.pose()
        return dict(x_err_mm=round((p[0] - x) * 1000, 1), y_err_mm=round((p[1] - ly) * 1000, 1), heading=round(math.degrees(yaw_of(q)), 2))

    def ramp(self, joints, secs):
        """joints: {name: target}. Smooth move on the drives."""
        q0 = {k: self.qd[self.ix[k]] for k in joints}
        n = max(1, int(round(secs / DT)))
        for i in range(n):
            u = (i + 1) / n
            u = u * u * (3 - 2 * u)
            for k, v in joints.items():
                self.qd[self.ix[k]] = q0[k] + (v - q0[k]) * u
            self.step()

    def set_ext(self, ext, secs):
        """Arm extension 0-0.52 m: one driven joint at ext / 4, the mimic constraints carry the other three."""
        self.ramp({"joint_arm_l3": ext / 4}, secs)

    def hand_to(self, target, secs, k=.12, base=True, tol=.0015, settle=1.5):
        """Hand to a world point on the drives: lift for z, telescope for y, base wheels for x, all together. Targets integrate the measured error, so the lift's spring sag is found, not assumed.
        Force stays limited by the drives. Returns the final error in mm. Follows a straight smooth path, then holds up to `settle` seconds."""
        I = self.ix
        L, A = I["joint_lift"], I["joint_arm_l3"]
        g0 = self.link("link_grasp_center")[:3].copy()
        t = np.asarray(target, float)
        n, m = int(round(secs / DT)), int(round(settle / DT))
        for i in range(n + m):
            if i % 12 == 0:
                u = min(1., (i + 12) / n)
                u = u * u * (3 - 2 * u)
                want = g0 + (t - g0) * u
                g = self.link("link_grasp_center")[:3]
                e = want - g
                self.qd[L] += k * e[2]
                self.qd[A] = min(.13, max(0., self.qd[A] + k * e[1] / 4))
                if base:
                    p, q = self.pose()
                    yerr = (math.pi - yaw_of(q) + math.pi) % (2 * math.pi) - math.pi
                    self.wheels(max(-.1, min(.1, -2.0 * e[0])), max(-.4, min(.4, 3.0 * yerr)))
                if i >= n and np.abs(e).max() < tol:
                    break
            self.step()
        self.wheels(0., 0.)
        self.wait(.2)
        return dict(err_mm=np.round((self.link("link_grasp_center")[:3] - t) * 1000, 1).tolist())

    def grip_open(self, slide, secs=1.0):
        self.ramp({"joint_gripper_slide": slide}, secs)

    def close_on(self, pad_n=20.0, rate=.008, timeout=6.0):
        """Creep shut until the slide lags (touch), then build a chosen pad force as a slide target offset. Slide force = 3.56 x pad force, drive kp 4000 N/m.
        Measured force comes from the drive error. pad_n is a chosen setting; the real gripper force is unverified."""
        I = self.ix["joint_gripper_slide"]
        touch = None
        for i in range(int(timeout / DT)):
            self.qd[I] -= rate * DT
            self.step()
            if self.q()[I] - self.qd[I] > .002 and i > 60:
                touch = float(self.q()[I])
                break
        off = pad_n * 3.56 / self.kp[I]
        q = float(self.q()[I])
        self.ramp({"joint_gripper_slide": q - off}, 1.0)
        self.wait(.3)
        return dict(touch_slide=touch, final_slide=round(float(self.q()[I]), 4), cmd=round(float(self.qd[I]), 4), slide_force_n=round(float(self.kp[I] * (self.q()[I] - self.qd[I])), 1), pad_force_est_n=round(float(self.kp[I] * (self.q()[I] - self.qd[I])) / 3.56, 1))

    def report(self):
        """Numbers to read at a stop."""
        p, q = self.pose()
        qq = self.q()
        return dict(t=round(self.t, 2), base=np.round(p, 3).tolist(), yaw=round(math.degrees(yaw_of(q)), 1), tilt=round(math.degrees(math.acos(max(-1., min(1., rmat(q)[2, 2])))), 2),
                    lift=round(float(qq[self.ix["joint_lift"]]), 4), arm=[round(float(qq[self.ix[j]]), 4) for j in TELE if j in self.ix], yaw_w=round(float(qq[self.ix["joint_wrist_yaw"]]), 3),
                    pitch=round(float(qq[self.ix["joint_wrist_pitch"]]), 3), roll=round(float(qq[self.ix["joint_wrist_roll"]]), 3), slide=round(float(qq[self.ix["joint_gripper_slide"]]), 4),
                    grasp=np.round(self.link("link_grasp_center")[:3], 3).tolist() if "link_grasp_center" in self.links else None)


def contact_pre(robot_links=("link_gripper", "rubber_tip", "link_SG3", "link_wrist", "link_DW3", "link_arm"), objects=(), extra_paths=()):
    """gates.contact_pre with this robot's prim: the original hard-codes /World/h1_2."""
    import gates

    def f(st):
        done = []
        for name in list(objects) + list(extra_paths):
            q = st.GetPrimAtPath(gates._find(st, name))
            if q and q.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(q).CreateThresholdAttr(0.0)
                done.append(q.GetPath().pathString)
        for q in Usd.PrimRange(st.GetPrimAtPath(PRIM), Usd.TraverseInstanceProxies()):
            if q.HasAPI(UsdPhysics.RigidBodyAPI) and q.GetName().startswith(tuple(robot_links)) and not q.IsInstanceProxy():
                PhysxSchema.PhysxContactReportAPI.Apply(q).CreateThresholdAttr(0.0)
                done.append(q.GetPath().pathString)
        f.paths = done
        print("contact_pre: report API on", len(done), "bodies", flush=True)
    f.paths = []
    return f


def gates_for_stretch(sim_names):
    """gates.py reads limits from h12_physx.URDF and a humanoid fall height. Point both at this robot, in this process only: no file changed. Mimic followers and pad joints get wide limits: their constraint force is not a motor."""
    import gates
    import h12_physx as H
    path = ROOT + "/logs/limits.urdf"
    with open(path, "w") as fh:
        fh.write('<robot name="stretch_limits">')
        for n in sim_names:
            fh.write(f'<joint name="{n}"><limit effort="{LIMITS.get(n, 1000.0)}"/></joint>')
        fh.write("</robot>")
    H.URDF = path
    gates.FALL_Z = -.05
    return gates
