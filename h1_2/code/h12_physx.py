"""H1-2 in the tea room, PhysX only (no MuJoCo). Import after SimulationApp is created.
Room edits go to the session layer: the source USD files are never changed."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))    # release_paths.py lives at the code root
import release_paths as RP
import json
import math
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, RP.pylib())
import numpy as np
import onnxruntime as ort
import omni.physics.tensors
import omni.usd
from omni.physx import get_physx_interface, get_physx_simulation_interface
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

ROOM = RP.scene('environment.usda')
CORR = RP.scene('corrections.json')
ROBOT = RP.asset("robot", "h1_2", "h1_2.usda")
URDF = RP.asset("robot", "h1_2.urdf")
G = json.load(open(RP.asset("policy", "h1_2_walk_v1", 'gait.json')))
DT, PERIOD = 1 / 600, 12


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def qrot(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def qinv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def min_cmd(e, k, vmax, tol, vmin=0.14):
    """Velocity command with a floor: the walking network ignores commands below about 0.1 m/s."""
    if abs(e) < 0.6 * tol:
        return 0.0
    return float(math.copysign(min(vmax, max(vmin, k * abs(e))), e))


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def qerr(goal, cur):
    """World-frame rotation vector taking `cur` to `goal` (both x,y,z,w)."""
    d = qmul(goal, qinv(cur))
    if d[3] < 0:
        d = -d
    return 2 * d[:3]


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Sim:
    def __init__(self, app, xy=(0.62, 1.15), yaw_deg=90.0, z=1.05, watch=(), mimic=False, pad=0.0, pad_radius=0.0):
        self.app = app
        self.mimic = mimic
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
        done = 0
        for k, v in corr.items():
            p = self.objs.get(k)
            if p is None or len(v) != 7:
                continue
            xf = UsdGeom.Xformable(p)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*v[:3]))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(v[3], v[4], v[5], v[6]))
            done += 1
        print("corrections applied", done, "of", len(corr), flush=True)
        for k in watch:
            PhysxSchema.PhysxContactReportAPI.Apply(self.objs[k]).CreateThresholdAttr(0.0)
        sc = next(p for p in st.Traverse() if p.IsA(UsdPhysics.Scene))
        px = PhysxSchema.PhysxSceneAPI.Apply(sc)
        px.CreateEnableGPUDynamicsAttr(False)
        px.CreateBroadphaseTypeAttr("MBP")
        px.CreateSolverTypeAttr("TGS")
        px.CreateTimeStepsPerSecondAttr(600)
        r = st.DefinePrim("/World/h1_2", "Xform")
        r.GetReferences().AddReference(ROBOT)
        r.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
        app.update()
        self.passive = []
        if mimic:
            joints = {p.GetName(): p for p in Usd.PrimRange(r, Usd.TraverseInstanceProxies()) if p.IsA(UsdPhysics.RevoluteJoint)}
            for j in ET.parse(URDF).getroot().findall("joint"):
                m = j.find("mimic")
                if m is None:
                    continue
                jp, ref = joints[j.get("name")], joints[m.get("joint")]
                if jp.IsInstanceProxy():
                    raise RuntimeError("joint is an instance proxy: " + j.get("name"))
                api = PhysxSchema.PhysxMimicJointAPI.Apply(jp, "rotX")
                api.GetReferenceJointRel().SetTargets([ref.GetPath()])
                api.CreateReferenceJointAxisAttr("rotX")
                api.CreateGearingAttr(-float(m.get("multiplier", 1.0)))
                api.CreateOffsetAttr(-float(m.get("offset", 0.0)))
                self.passive.append(j.get("name"))
            print("mimic joints", len(self.passive), flush=True)
        if pad > 0:
            mat = UsdShade.Material.Define(st, "/World/HandPad")
            ma = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            ma.CreateStaticFrictionAttr(pad)
            ma.CreateDynamicFrictionAttr(pad)
            nb = 0
            for p in Usd.PrimRange(r, Usd.TraverseInstanceProxies()):
                nm = p.GetName()
                if nm[:2] in ("L_", "R_") and any(k in nm for k in ("index", "middle", "ring", "pinky", "thumb", "hand_base")) and p.HasAPI(UsdPhysics.RigidBodyAPI):
                    if p.IsInstanceProxy():
                        continue
                    UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
                    nb += 1
            print("pad friction", pad, "bound on links", nb, flush=True)
            if pad_radius > 0:
                instances = set()
                for p in Usd.PrimRange(r, Usd.TraverseInstanceProxies()):
                    path = str(p.GetPath())
                    if ('/L_hand_base_link/' in path or '/R_hand_base_link/' in path) and p.HasAPI(UsdPhysics.CollisionAPI) and p.IsInstanceProxy():
                        ancestor = p
                        while ancestor.IsInstanceProxy():
                            ancestor = ancestor.GetParent()
                        if ancestor.IsInstance():
                            instances.add(str(ancestor.GetPath()))
                for path in instances:
                    st.GetPrimAtPath(path).SetInstanceable(False)
                count = 0
                for p in Usd.PrimRange(r, Usd.TraverseInstanceProxies()):
                    path = str(p.GetPath())
                    if ('/L_hand_base_link/' in path or '/R_hand_base_link/' in path) and p.HasAPI(UsdPhysics.CollisionAPI) and not p.IsInstanceProxy():
                        api = PhysxSchema.PhysxCollisionAPI.Apply(p)
                        api.CreateTorsionalPatchRadiusAttr(float(pad_radius))
                        api.CreateMinTorsionalPatchRadiusAttr(float(pad_radius))
                        count += 1
                print('ASSUMPTION finite fingertip contact radius', pad_radius, 'm on', count, 'shapes', flush=True)
                if count == 0:
                    raise RuntimeError('requested fingertip contact radius was not applied to any collider')
        root = next(p.GetPath().pathString for p in Usd.PrimRange(r) if p.HasAPI(UsdPhysics.ArticulationRootAPI))
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
        n = self.n = a.max_dofs
        self.idx = np.arange(1, dtype=np.uint32)
        ur = ET.parse(URDF).getroot()
        eff = {j.get("name"): float(j.find("limit").get("effort")) for j in ur.findall("joint") if j.find("limit") is not None}
        lim = np.array([eff.get(x, 1.0) for x in self.names])
        lim[lim <= 0] = 1.0
        self.legs = np.array([self.names.index(x) for x in G["legs"]])
        self.default = np.array(G["default"], float)
        hold, kp, kd = np.zeros(n), np.zeros(n), np.zeros(n)
        for i, x in enumerate(self.names):
            for key, s_, d_ in G["hold_gains"]:
                if key in x:
                    kp[i], kd[i] = s_, d_
                    break
            hold[i] = G["hold_pose"].get(x, 0.0)
        kp[self.legs], kd[self.legs] = G["kp"], G["kd"]
        # The Cartesian servo changes drive references at 50 Hz. The inherited
        # low arm damping permits recoil after the first target crossing.
        # Damping changes the controller, never the URDF effort caps.
        for i, x in enumerate(self.names):
            if x.startswith(("left_", "right_")) and any(j in x for j in ("shoulder", "elbow", "wrist")):
                kd[i] *= 2.5
        for i, x in enumerate(self.names):
            if x[:2] in ("L_", "R_"):
                kp[i], kd[i] = (0.0, 0.0) if x in self.passive else (3.0, 0.05)
        self.hold, self.kp, self.kd = hold, kp, kd

        def f(v):
            return np.asarray(v, np.float32)[None]

        self.f = f
        a.set_dof_max_forces(f(lim), self.idx)
        a.set_dof_stiffnesses(f(kp), self.idx)
        a.set_dof_dampings(f(kd), self.idx)
        arm = np.full(n, 0.001)
        arm[self.legs] = 0.01
        a.set_dof_armatures(f(arm), self.idx)
        q0 = hold.copy()
        q0[self.legs] = self.default
        a.set_dof_positions(f(q0), self.idx)
        a.set_dof_velocities(f(np.zeros(n)), self.idx)
        tf = a.get_root_transforms().copy()
        y = math.radians(yaw_deg)
        tf[0] = [xy[0], xy[1], z, 0, 0, math.sin(y / 2), math.cos(y / 2)]
        a.set_root_transforms(tf, self.idx)
        a.set_root_velocities(np.zeros((1, 6), np.float32), self.idx)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.ses = ort.InferenceSession(RP.asset("policy", "h1_2_walk_v1", 'policy.onnx'), so,
                                        providers=["CPUExecutionProvider"])
        self.h = np.zeros((1, 1, 64), np.float32)
        self.c = np.zeros((1, 1, 64), np.float32)
        self.last = np.zeros(12)
        self.target = self.default.copy()
        self.cmd = np.zeros(3)
        self.want = np.zeros(3)
        self.t = 0.0
        self.k = 0
        self.mode = "walk"
        self.links = list(a.shared_metatype.link_names)
        self.mass = a.get_masses()[0].astype(float)
        self.feet = [self.links.index(x) for x in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.comv = np.zeros(2)
        self.com_prev = None
        self.off = np.zeros(12)
        self.gc_idx = np.r_[self.arm_idx("L"), self.arm_idx("R")]
        self.gc = np.zeros(n)
        self.gc_on = True
        self.lim = lim
        self.bodies, self.touch = {}, {}
        for k in watch:
            path = self.objs[k].GetPath().pathString
            self.bodies[k] = self.sv.create_rigid_body_view(path)
            hand = [x for x in a.link_paths[0] if "/L_hand_base_link" in x or x.endswith("left_wrist_yaw_link")]
            self.touch[k] = (self.sv.create_rigid_contact_view([path], [hand], 512), hand)

    def obj(self, k):
        """Measured pose of a watched rigid body: x,y,z,qx,qy,qz,qw."""
        return self.bodies[k].get_transforms()[0].astype(float)

    def contact(self, k):
        """Force (N) on watched body k from each left hand link: {link name: magnitude}, only links touching."""
        v, hand = self.touch[k]
        m = v.get_contact_force_matrix(DT)[0].astype(float)
        return {hand[i].rsplit("/", 1)[-1]: round(float(np.linalg.norm(m[i])), 2) for i in range(len(hand)) if np.linalg.norm(m[i]) > 0.01}

    def com(self):
        """Whole-robot centre of mass from PhysX link poses, masses and link COM offsets."""
        T = self.link_poses()
        C = self.art.get_coms()[0].astype(float)
        out = np.zeros(3)
        for i in range(len(self.links)):
            out += self.mass[i] * (T[i, :3] + qrot(T[i, 3:7], C[i, :3]))
        return out / self.mass.sum()

    def feet_xy(self):
        T = self.link_poses()
        return np.array([T[f, :3] for f in self.feet])

    def to_stand(self, ramp=0.6):
        """Legs from the walking network to a stiff standing stance with an ankle/hip balance loop."""
        a = self.art
        kp, kd = self.kp.copy(), self.kd.copy()
        kp[self.legs], kd[self.legs] = G["stand_kp"], G["stand_kd"]
        a.set_dof_stiffnesses(self.f(kp), self.idx)
        a.set_dof_dampings(self.f(kd), self.idx)
        self.stand_from = a.get_dof_positions()[0].astype(float)[self.legs].copy()
        self.stand_goal = self.default.copy()
        self.stand_ramp, self.stand_t = ramp, 0.0
        self.mode = "stand"
        self.com_prev = None

    def balance(self):
        _, q = self.pose()
        yaw = yaw_of(q)
        c, s = math.cos(yaw), math.sin(yaw)
        Rt = np.array([[c, s], [-s, c]])
        com = self.com()[:2]
        e = Rt @ (com - self.feet_xy()[:, :2].mean(0)) - np.array([G["foot_cx"], 0.0])
        if self.com_prev is None:
            self.com_prev = com.copy()
        v = Rt @ (com - self.com_prev) / (PERIOD * DT)
        self.com_prev = com.copy()
        self.comv = 0.7 * self.comv + 0.3 * v
        kx, kv, ky, kvy = G["balance"]
        pitch = float(np.clip(kx * e[0] + kv * self.comv[0], -0.15, 0.15))
        roll = float(np.clip(ky * e[1] + kvy * self.comv[1], -0.1, 0.1))
        o = np.zeros(12)
        for i in G["ankle_pitch_idx"]:
            o[i] += pitch
        for i in G["hip_roll_idx"]:
            o[i] += roll
        for i in G["ankle_roll_idx"]:
            o[i] -= roll
        self.off = o
        self.e_com = e

    def link_poses(self):
        """One readback per physics tick, shared by control and measurement."""
        if getattr(self, '_link_pose_tick', -1) != self.k:
            self._link_pose_snapshot = self.art.get_link_transforms()[0].astype(float).copy()
            self._link_pose_tick = self.k
        return self._link_pose_snapshot

    def hand(self, side="L"):
        i = self.links.index(f"{side}_hand_base_link")
        return self.link_poses()[i].copy()

    def arm_idx(self, side="L"):
        pre = "left_" if side == "L" else "right_"
        return np.array([self.names.index(pre + x) for x in ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
                                                            "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint")])

    def reach(self, goal, seconds=2.0, side="L", tol=0.004):
        """Hand position to `goal` along a straight line: damped least squares on the PhysX Jacobian, arm targets only."""
        a = self.art
        ai = self.arm_idx(side)
        li = self.links.index(f"{side}_hand_base_link")
        lo, hi = a.get_dof_limits()[0, ai, 0].astype(float), a.get_dof_limits()[0, ai, 1].astype(float)
        p0 = self.hand(side)[:3].copy()
        goal = np.asarray(goal, float)
        steps = int(seconds / (PERIOD * DT))
        qd = a.get_dof_positions()[0].astype(float)[ai]
        for k in range(steps + 50):
            u = min(1.0, (k + 1) / steps)
            u = u * u * (3 - 2 * u)
            ref = p0 + u * (goal - p0)
            J = a.get_jacobians().reshape(-1, 6, a.max_dofs + 6)[li - 1 if a.get_jacobians().reshape(-1, 6, a.max_dofs + 6).shape[0] < len(self.links) else li, :3, :][:, 6 + ai].astype(float)
            e = ref - self.hand(side)[:3]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), np.clip(e, -0.02, 0.02))
            qd = np.clip(qd + np.clip(dq, -0.03, 0.03), lo, hi)
            self.hold[ai] = qd
            for _ in range(PERIOD):
                self.step()
            if self.fallen():
                return False
            if k >= steps and np.linalg.norm(goal - self.hand(side)[:3]) < tol:
                break
        return True

    def bounded_arm_reference(self, ai, target):
        """Prevent IK reference windup against contacts and joint stops.

        A position target farther ahead than the drive's available torque can
        produce stores error without moving the arm. Bound that lead using the
        measured joint state and current drive budget; never write joint state.
        """
        measured = self.art.get_dof_positions()[0, ai].astype(float)
        budget = self.art.get_dof_max_forces()[0, ai].astype(float)
        lead = np.clip(budget / np.maximum(self.kp[ai], 1.0) + .02, .03, .5)
        limits = self.art.get_dof_limits()[0, ai]
        return np.clip(target, np.maximum(limits[:, 0], measured - lead),
                       np.minimum(limits[:, 1], measured + lead))

    def ik_to(self, goal_p, goal_q=None, seconds=2.0, side="L", tol=0.003, w_rot=0.5, settle=100, rest=None, stop=None, gain=0.12, tol_ang=0.035):
        """Hand frame to a position and (x,y,z,w) orientation on a straight line: damped least squares on the PhysX
        Jacobian, closed on the measured hand pose. Only arm joint targets change. Returns the final position error (m)."""
        a = self.art
        ai = self.arm_idx(side)
        li = self.links.index(f"{side}_hand_base_link")
        lim = a.get_dof_limits()[0]
        lo, hi = lim[ai, 0].astype(float), lim[ai, 1].astype(float)
        h0 = self.hand(side)
        p0, q0 = h0[:3].copy(), h0[3:7].copy()
        goal_p = np.asarray(goal_p, float)
        goal_q = q0 if goal_q is None else np.asarray(goal_q, float)
        if np.dot(q0, goal_q) < 0:
            goal_q = -goal_q
        steps = max(1, int(seconds / (PERIOD * DT)))
        qd = self.hold[ai].copy()
        W = np.r_[1, 1, 1, w_rot, w_rot, w_rot]
        err = 1.0
        prev_p, prev_q = p0.copy(), q0.copy()
        for k in range(steps + settle):
            u = min(1.0, (k + 1) / steps)
            u = u * u * (3 - 2 * u)
            ref_p = p0 + u * (goal_p - p0)
            ref_q = q0 + u * (goal_q - q0)
            ref_q /= np.linalg.norm(ref_q)
            h = self.hand(side)
            J = a.get_jacobians().reshape(-1, 6, a.max_dofs + 6)[li][:, 6 + ai].astype(float) * W[:, None]
            ep = ref_p - h[:3]
            eo = qerr(ref_q, h[3:7])
            ffp, ffo = ref_p - prev_p, qerr(ref_q, prev_q)
            prev_p, prev_q = ref_p, ref_q
            e = np.r_[ffp + gain * np.clip(ep, -0.02, 0.02), w_rot * (ffo + gain * np.clip(eo, -0.15, 0.15))]
            A = J @ J.T + 2e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(A, e)
            if rest is not None:
                N = np.eye(7) - J.T @ np.linalg.solve(A, J)
                dq += N @ (0.02 * (np.asarray(rest, float) - qd))
            qd = np.clip(qd + np.clip(dq, -0.025, 0.025), lo, hi)
            qd = self.bounded_arm_reference(ai, qd)
            self.hold[ai] = qd
            for _ in range(PERIOD):
                self.step()
            if self.fallen():
                return 9.0
            err = float(np.linalg.norm(goal_p - self.hand(side)[:3]))
            if stop is not None and stop():
                return err
            self.ang_err = float(np.linalg.norm(qerr(goal_q, self.hand(side)[3:7])))
            speed = float(np.max(np.abs(a.get_dof_velocities()[0, ai])))
            if k >= steps and err < tol and self.ang_err < tol_ang and speed < 0.08:
                break
        return err

    def finger_idx(self, side, key):
        """Joints of one finger with the URDF mimic ratios (PhysX does not enforce them, so each is driven)."""
        S = side
        if key == "thumb_yaw":
            return [(self.names.index(f"{S}_thumb_proximal_yaw_joint"), 1.0)]
        if key == "thumb_pitch":
            return [(self.names.index(f"{S}_thumb_proximal_pitch_joint"), 1.0), (self.names.index(f"{S}_thumb_intermediate_joint"), 1.6),
                    (self.names.index(f"{S}_thumb_distal_joint"), 2.4)]
        return [(self.names.index(f"{S}_{key}_proximal_joint"), 1.0), (self.names.index(f"{S}_{key}_intermediate_joint"), 1.0)]

    def fingers(self, side, vals, seconds=0.5):
        """Finger targets ramped over `seconds`. vals: {index, middle, ring, pinky, thumb_yaw, thumb_pitch: rad}."""
        start = {k: [self.hold[i] for i, _ in self.finger_idx(side, k)] for k in vals}
        steps = max(1, int(seconds / (PERIOD * DT)))
        for n_ in range(steps):
            u = (n_ + 1) / steps
            for k, v in vals.items():
                for (i, m), s0 in zip(self.finger_idx(side, k), start[k]):
                    self.hold[i] = s0 + u * (v * m - s0)
            for _ in range(PERIOD):
                self.step()

    def finger_q(self, side, key):
        return float(self.art.get_dof_positions()[0, self.finger_idx(side, key)[0][0]])

    def crouch(self, a_rad, seconds=1.5):
        """Knee bend about the standing pose: hip pitch -a, knee +2a, ankle pitch -a keeps the pelvis over the feet."""
        goal = self.default.copy()
        for j, name in enumerate(G["legs"]):
            if "hip_pitch" in name:
                goal[j] -= a_rad
            elif "knee" in name:
                goal[j] += 2 * a_rad
            elif "ankle_pitch" in name:
                goal[j] -= a_rad
        u = min(1.0, self.stand_t / self.stand_ramp)
        self.stand_from = self.stand_from + u * u * (3 - 2 * u) * (self.stand_goal - self.stand_from)
        self.stand_goal, self.stand_ramp, self.stand_t = goal, seconds, 0.0
        for _ in range(int(seconds / DT) + 120):
            self.step()
            if self.fallen():
                return False
        return True

    def hold_for(self, seconds):
        for _ in range(int(seconds / DT)):
            self.step()
        return not self.fallen()

    def pose(self):
        r = self.art.get_root_transforms()[0].astype(float)
        return r[:3], r[3:7]

    def step(self):
        a = self.art
        if self.mode == "walk" and self.k % PERIOD == 0:
            q, v = a.get_dof_positions()[0].astype(float), a.get_dof_velocities()[0].astype(float)
            _, qq = self.pose()
            rv = a.get_root_velocities()[0].astype(float)
            grav, wb = qrot(qinv(qq), np.array([0, 0, -1.0])), qrot(qinv(qq), rv[3:6])
            self.cmd += np.clip(self.want - self.cmd, -0.008, 0.008)
            ph = (self.t % G["gait_period"]) / G["gait_period"]
            L = self.legs
            o = np.r_[wb * 0.25, grav, self.cmd * np.array(G["cmd_scale"]), q[L] - self.default, v[L] * 0.05, self.last,
                      math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)]
            act, self.h, self.c = self.ses.run(None, {"obs": np.clip(o, -100, 100).astype(np.float32)[None],
                                                      "h_in": self.h, "c_in": self.c})
            self.last = np.clip(act.reshape(-1).astype(float), -100, 100)
            self.target = self.default + 0.25 * self.last
        qd = self.hold.copy()
        if self.mode == "stand":
            if self.k % PERIOD == 0:
                self.balance()
            self.stand_t += DT
            u = min(1.0, self.stand_t / self.stand_ramp)
            u = u * u * (3 - 2 * u)
            qd[self.legs] = self.stand_from + u * (self.stand_goal - self.stand_from) + self.off
        else:
            qd[self.legs] = self.target
        a.set_dof_position_targets(self.f(qd), self.idx)
        if self.gc_on:
            if self.k % PERIOD == 0:
                tau = a.get_gravity_compensation_forces()[0].astype(float)
                self.gc[:] = 0.0
                self.gc[self.gc_idx] = np.clip(tau[6 + self.gc_idx], -0.8 * self.lim[self.gc_idx], 0.8 * self.lim[self.gc_idx])
                a.set_dof_max_forces(self.f(self.lim - np.abs(self.gc)), self.idx)
            a.set_dof_actuation_forces(self.f(self.gc), self.idx)
        self.sim.simulate(DT, self.t)
        self.sim.fetch_results()
        self.t += DT
        self.k += 1

    def fallen(self):
        p, q = self.pose()
        if not np.isfinite(np.r_[p, q]).all():
            return True
        up = qrot(q, np.array([0., 0., 1.]))
        return bool(p[2] < 0.6 or up[2] < np.cos(np.radians(55.)))

    def go(self, x, y, yaw_deg, vmax=0.3, tol=0.05, timeout=40.0, log=print):
        """Walk to (x, y, yaw): face the path on long moves, position + heading feedback on the velocity command."""
        gy = math.radians(yaw_deg)
        t0 = self.t
        nlog = 0
        arrived = False
        while self.t - t0 < timeout:
            p, q = self.pose()
            yaw = yaw_of(q)
            e = np.array([x, y]) - p[:2]
            dist = float(np.linalg.norm(e))
            if dist < tol and abs(wrap(gy - yaw)) < math.radians(6):
                arrived = True
                break
            c, s = math.cos(yaw), math.sin(yaw)
            ex, ey = c * e[0] + s * e[1], -s * e[0] + c * e[1]
            head = math.atan2(e[1], e[0]) if dist > 0.35 else gy
            self.want = np.array([min_cmd(ex, 1.2, vmax, tol), min_cmd(ey, 1.2, 0.2, tol),
                                  float(np.clip(1.2 * wrap(head - yaw), -0.6, 0.6))])
            for _ in range(PERIOD):
                self.step()
            if self.fallen():
                log("FELL at", round(self.t, 2), np.round(p, 3), flush=True)
                return False
            if self.t - t0 > nlog:
                nlog += 2
                log(f"t {self.t:.1f} pelvis {np.round(p, 3)} yaw {math.degrees(yaw):.1f} err {dist:.3f}", flush=True)
        self.want = np.zeros(3)
        p, q = self.pose()
        log("ARRIVED" if arrived else "TIMED OUT", np.round(p, 3), "yaw", round(math.degrees(yaw_of(q)), 1), "t", round(self.t, 2), flush=True)
        return arrived
