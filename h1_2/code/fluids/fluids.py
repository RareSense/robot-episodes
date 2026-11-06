"""Live PhysX water (and powder) in the same scene as the H1-2. No MuJoCo, no separate stage.

Use (Sim setup, after room + robot are in the stage, BEFORE force_load_physics_from_usd):
    F = fluids.Fluids(fill_ml=400)
    F.pre_physics(stage, scene_prim)      # hollow kettle + mug colliders, particle system, water in the kettle
After the tensor view exists:
    F.attach(sim_view)
During the run (each call costs one physics step's worth of particle readback, so do it a few times a second):
    F.sample(step_fn)                     # runs step_fn once with particle writeback on, reads positions
    F.report() -> {'kettle_ml','mug_ml','other_ml','spill_ml', ...}

Kettle and mug stay ordinary dynamic rigid bodies. The kettle's main hull is solid, so water cannot be inside it:
the hulls that fill the cavity are disabled in the session layer and replaced by thin box walls under the same
body (geometry from water_layer.py, measured from the twin). The handle hulls stay. The mug keeps its own hulls. The kettle pose used for the
reports comes from the tensor rigid body view, so it follows real motion.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))    # release_paths.py lives at the code root
import release_paths as RP
import math
import sys

import numpy as np

sys.path.insert(0, RP.code("water"))
import water_layer as wl  # noqa: E402  (geometry only; its class is never instantiated)

KETTLE = "/World/Objects/object_0295"
MUG = "/World/Objects/object_0316"
ROOT = "/World/fluid"


def rot_inv(q, v):
    """v rotated by the inverse of q (x, y, z, w); v is (n, 3)."""
    u = -np.asarray(q[:3], float)
    w = float(q[3])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


class Fluids:
    def __init__(self, fill_ml=400.0, fro=0.00125, iters=4, friction=0.02, cohesion=0.1, vp_factor=4.47,
                 solid_rest=0.0009, max_velocity=4.0, kettle=KETTLE, mug=MUG, keep_material=True):
        self.fill_ml, self.fro, self.iters, self.friction, self.cohesion = fill_ml, fro, iters, friction, cohesion
        self.vp = vp_factor * fro ** 3            # one particle's share of the water volume (m^3)
        self.solid_rest, self.max_velocity = solid_rest, max_velocity
        self.kettle, self.mug = kettle, mug
        self.keep_material = keep_material
        self.solid_mat = None                     # read by water_layer's shape builder
        self.P = None
        self.n = 0

    # ------------------------------------------------------------ colliders
    def _hollow(self, st, path, shapes, r_cavity, name):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
        root = st.GetPrimAtPath(path)
        xc = UsdGeom.XformCache()
        off, kept, mat = [], [], None
        for p in root.GetChildren():
            if not p.HasAPI(UsdPhysics.CollisionAPI):
                continue
            pts = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), float)
            M = np.array(xc.ComputeRelativeTransform(p, root)[0], float)
            pts = pts @ M[:3, :3] + M[3, :3]
            lo, hi = pts.min(0), pts.max(0)
            near = math.hypot(max(lo[0], 0.0, -hi[0]), max(lo[1], 0.0, -hi[1]))   # bbox point nearest the body axis
            b, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
            mat = b if b else mat
            print(f"[fluids]   {p.GetName()} bbox {np.round(lo, 3)} {np.round(hi, 3)} axis distance {near:.3f}", flush=True)
            if near < r_cavity or (name == "kettle" and hi[0] < 0):   # fills the cavity, or plugs the spout (-x side)
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
                off.append(p.GetName())
            else:
                kept.append(p.GetName())
        print(f"[fluids] {name}: hulls disabled {off}, kept {kept}, material {mat.GetPath() if mat else None}", flush=True)
        self.solid_mat = mat if self.keep_material else self._low
        UsdGeom.Xform.Define(st, f"{path}/fluid_col")
        pco = self.fro / 0.6
        for i, sh in enumerate(shapes):
            pth = f"{path}/fluid_col/s{i}"
            wl.WaterLayer._shape(self, st, pth, sh, pco + 0.001, self.solid_rest)
            UsdGeom.Imageable(st.GetPrimAtPath(pth)).MakeInvisible()
        M = np.array(xc.GetLocalToWorldTransform(root), float)
        print(f"[fluids] {name}: {len(shapes)} wall shapes, origin {np.round(M[3, :3], 4)}, scale {np.round(np.linalg.norm(M[:3, :3], axis=1), 4)}", flush=True)
        return M

    # ------------------------------------------------------------ particle-compatible hulls
    def gpu_safe_hulls(self, st, limit=32):
        """Particles ignore a convex hull PhysX cannot run on the GPU ("Non-GPU-compatible convex mesh"). The default hull limit of
        64 vertices still allows more than 64 polygons on round shapes, such as the mug handle (86 hull vertices, 139 faces) and the jar.
        A 32-vertex hull has at most 60 faces. Session layer only; the robot is left alone. Hulls already small are unchanged."""
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
        done = []
        for p in st.Traverse():
            path = p.GetPath().pathString
            if path.startswith("/World/h1_2") or p.GetTypeName() != "Mesh" or not p.HasAPI(UsdPhysics.CollisionAPI) or p.IsInstanceProxy():
                continue
            if not p.HasAPI(UsdPhysics.MeshCollisionAPI) or UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() != "convexHull":
                continue
            if UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is False:
                continue
            pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if pts is not None and len(pts) > limit:
                PhysxSchema.PhysxConvexHullCollisionAPI.Apply(p).CreateHullVertexLimitAttr(int(limit))
                done.append(f"{path.replace('/World/Objects/', '')}:{len(pts)}")
        print(f"[fluids] hull vertex limit {limit} on {len(done)} colliders: {done}", flush=True)
        return done

    # ------------------------------------------------------------ build
    def pre_physics(self, st, sc):
        from omni.physx.scripts import particleUtils, physicsUtils
        from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade, Vt
        self.stage = st
        UsdGeom.Xform.Define(st, ROOT)
        self._low = UsdShade.Material.Define(st, f"{ROOT}/wall_mat")
        pm = UsdPhysics.MaterialAPI.Apply(self._low.GetPrim())
        pm.CreateStaticFrictionAttr(self.friction)
        pm.CreateDynamicFrictionAttr(self.friction)
        pm.CreateRestitutionAttr(0.0)
        ks = [("cyl", (0, 0, wl.KETTLE_FLOOR / 2), 0.079, wl.KETTLE_FLOOR / 2) if (s[0] == "cyl" and s[2] == 0.079) else s for s in wl.kettle_shapes()]   # floor reaches z 0: same rest height as the twin
        Mk = self._hollow(st, self.kettle, ks, 0.05, "kettle")
        self.gpu_safe_hulls(st)
        # The mug needs no change: its twin hulls are already a hollow shell (a floor disc and 8 wall segments, cavity apothem
        # 27 mm). Replacing them with box walls lost the robot's grip in the same test (see NOTES.md), so they stay.
        fro = self.fro
        pco = fro / 0.6
        ps = particleUtils.add_physx_particle_system(
            st, f"{ROOT}/ps", simulation_owner=sc.GetPath(), particle_contact_offset=pco, contact_offset=pco + 0.001,
            fluid_rest_offset=fro, solid_rest_offset=self.solid_rest, rest_offset=self.solid_rest, enable_ccd=True,
            solver_position_iterations=self.iters, max_neighborhood=96, max_velocity=self.max_velocity)
        UsdShade.Material.Define(st, f"{ROOT}/water_pbd")
        particleUtils.add_pbd_particle_material(st, f"{ROOT}/water_pbd", cohesion=self.cohesion, damping=0.0, friction=self.friction,
                                                surface_tension=0.0074, viscosity=1.7e-6, density=1000.0, adhesion=0.0)
        physicsUtils.add_physics_material_to_prim(st, ps.GetPrim(), f"{ROOT}/water_pbd")
        pts = wl.WaterLayer._fill_points(self, self.fill_ml)
        w = pts @ Mk[:3, :3] + Mk[3, :3]
        self.n = len(w)
        particleUtils.add_physx_particleset_pointinstancer(
            st, Sdf.Path(f"{ROOT}/water"), Vt.Vec3fArray.FromNumpy(w.astype(np.float32)), Vt.Vec3fArray.FromNumpy(np.zeros_like(w, np.float32)),
            f"{ROOT}/ps", self_collision=True, fluid=True, particle_group=0, particle_mass=1000.0 * self.vp, density=0.0)
        UsdGeom.Sphere.Get(st, f"{ROOT}/water/particlePrototype0").CreateRadiusAttr(fro)
        self.inst = UsdGeom.PointInstancer(st.GetPrimAtPath(f"{ROOT}/water"))
        self.ps_path = f"{ROOT}/ps"
        print(f"[fluids] water {self.n} particles = {self.n * self.vp * 1e6:.0f} ml, particle radius {fro * 1e3:.2f} mm, "
              f"solver iterations {self.iters}", flush=True)

    def add_grains(self, st, center, n=120, r=0.0025, rho=600.0, friction=0.7, spread=0.018):
        """Powder as free rigid spheres (the props_powder.py lumps: r 2.5 mm, 600 kg/m3, 39 mg, friction 0.7).
        center: world position of the lowest layer's middle. Grains start on a loose lattice, at rest."""
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
        mat = UsdShade.Material.Define(st, f"{ROOT}/grain_mat")
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(friction)
        pm.CreateDynamicFrictionAttr(friction)
        pm.CreateRestitutionAttr(0.0)
        sp = 2 * r + 0.0012
        g = np.mgrid[-spread:spread + 1e-6:sp, -spread:spread + 1e-6:sp].reshape(2, -1).T
        g = g[np.hypot(g[:, 0], g[:, 1]) <= spread]
        pts = [np.array(center) + [x, y, k * sp] for k in range(n // len(g) + 1) for x, y in g][:n]
        UsdGeom.Xform.Define(st, f"{ROOT}/grains")
        m = rho * 4 / 3 * math.pi * r ** 3
        for i, p in enumerate(pts):
            s_ = UsdGeom.Sphere.Define(st, f"{ROOT}/grains/g{i:03d}")
            s_.CreateRadiusAttr(r)
            s_.AddTranslateOp().Set(Gf.Vec3d(*map(float, p)))
            q = s_.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(q)
            UsdPhysics.CollisionAPI.Apply(q)
            UsdPhysics.MassAPI.Apply(q).CreateMassAttr(m)
            c = PhysxSchema.PhysxCollisionAPI.Apply(q)
            c.CreateContactOffsetAttr(0.002)
            c.CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(q).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        self.n_grains = len(pts)
        print(f"[fluids] powder: {len(pts)} rigid grains, r {r * 1e3:.1f} mm, {m * 1e6:.0f} mg each, total {m * len(pts) * 1e3:.1f} g", flush=True)

    # ------------------------------------------------------------ live state
    def attach(self, sv):
        """sv: the omni.physics.tensors simulation view. Creates views for the kettle, the mug and the grains."""
        self.kv = sv.create_rigid_body_view(self.kettle)
        self.mv = sv.create_rigid_body_view(self.mug)
        self.gv = sv.create_rigid_body_view(f"{ROOT}/grains/g*") if getattr(self, "n_grains", 0) else None
        return self

    def grains_report(self):
        """Live grain counts: in the mug (inside its wall, below the rim), the rest split into on/above the rim and fallen."""
        G = np.asarray(self.gv.get_transforms(), float)[:, :3]
        _, m = self.poses()
        lm = rot_inv(m[3:7], G - m[:3])
        ins = (np.hypot(lm[:, 0], lm[:, 1]) < 0.0300) & (lm[:, 2] > 0.0) & (lm[:, 2] < wl.MUG_RIM + 0.004)
        return {"grains_total": int(len(G)), "grains_in_mug": int(ins.sum()), "grains_out": int((~ins).sum()), "grain_zmin": round(float(G[:, 2].min()), 3)}

    def poses(self):
        return (np.asarray(self.kv.get_transforms(), float)[0], np.asarray(self.mv.get_transforms(), float)[0])

    def sample(self, step):
        """step(): one simulation step, run with particle writeback on. Then positions are read from USD."""
        import carb
        import omni.physx.bindings._physx as pb
        cs = carb.settings.get_settings()
        cs.set(pb.SETTING_UPDATE_PARTICLES_TO_USD, True)
        step()
        cs.set(pb.SETTING_UPDATE_PARTICLES_TO_USD, False)
        self.P = np.asarray(self.inst.GetPositionsAttr().Get(), float)
        self.P_pose = self.poses()      # kettle and mug at this same instant: a sample can be 0.2 s old when read, and a moving mug then read as spill (cup_gpu02: 113 ml "spilled", 109.9 ml in the mug)
        return self.P

    def stream(self, z0, z1):
        """Water in the air between heights z0 and z1, outside the kettle: (centroid xyz, particle count). Live aim feedback for the pour."""
        P = self.P
        k, _ = self.poses()
        lk = rot_inv(k[3:7], P - k[:3])
        in_k = ((np.hypot(lk[:, 0], lk[:, 1]) < 0.08) & (lk[:, 2] > 0.0) & (lk[:, 2] < 0.215)) | \
               ((lk[:, 0] < -0.04) & (lk[:, 0] > -0.1) & (np.abs(lk[:, 1]) < 0.03) & (lk[:, 2] > 0.14) & (lk[:, 2] < 0.216))
        b = ~in_k & (P[:, 2] > z0) & (P[:, 2] < z1)
        n = int(b.sum())
        return (P[b].mean(0) if n else np.zeros(3)), n

    def report(self):
        P = self.P
        k, m = getattr(self, "P_pose", None) or self.poses()
        lk = rot_inv(k[3:7], P - k[:3])
        lm = rot_inv(m[3:7], P - m[:3])
        in_mug = (np.hypot(lm[:, 0], lm[:, 1]) < 0.0335) & (lm[:, 2] > 0.0) & (lm[:, 2] < wl.MUG_RIM + 0.004)
        in_k = ((np.hypot(lk[:, 0], lk[:, 1]) < 0.08) & (lk[:, 2] > 0.0) & (lk[:, 2] < 0.215)) | \
               ((lk[:, 0] < -0.04) & (lk[:, 0] > -0.1) & (np.abs(lk[:, 1]) < 0.03) & (lk[:, 2] > 0.14) & (lk[:, 2] < 0.216))
        rest = ~in_mug & ~in_k
        low = rest & (P[:, 2] < m[2] + wl.MUG_RIM)      # outside both and below the rim: cannot land in the mug
        ml = 1e6 * self.vp
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (k[3] ** 2 + k[4] ** 2)))))
        out = {"kettle_ml": round(float(in_k.sum() * ml), 1), "mug_ml": round(float(in_mug.sum() * ml), 1),
               "other_ml": round(float(rest.sum() * ml), 1), "spill_ml": round(float(low.sum() * ml), 1), "tilt": round(tilt, 1),
               "kettle_xyz": np.round(k[:3], 3).tolist(), "mug_xyz": np.round(m[:3], 3).tolist(),
               "zmin": round(float(P[:, 2].min()), 3)}
        if in_mug.any():
            out["level_mm"] = round(float(np.percentile(lm[in_mug, 2], 98) * 1e3), 1)
        return out
