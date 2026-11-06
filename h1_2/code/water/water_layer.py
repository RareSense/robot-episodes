"""Simulated water for the tea film: PhysX position-based fluid (PBD particles) in Isaac Sim 6.2.

How it works
- The fluid lives in its own in-memory USD stage that omni.physx simulates (GPU PBD particle system).
  The render stage (the twin + looks) stays physics-free, exactly as tea_render.py wants it.
- The solids are kinematic colliders that follow the MuJoCo recording: every film frame k the kettle,
  the mug (and spoon / tea bag when given) get the world matrix baked in shot.npz. Between two film frames
  the fluid takes `substeps` fixed PhysX steps; the colliders' kinematic targets move along the straight
  line / slerp between the two recorded poses (a boundary condition for the fluid, not a change to the
  recording). One-way coupling: water does not push back on the MuJoCo bodies.
- The kettle and the mug are hollow colliders built from the twin's own measurements (kettle visual
  mesh: body profile, spout beak on -x at z 0.198-0.212; mug (set_fix.usda): inner wall r 29.5 mm with a floor fillet,
  floor z 9 mm, rim z 85 mm). Counter top and kettle base are static colliders.
- Rendering: PhysX builds an isosurface mesh of the fluid (anisotropy + smoothing + isosurface APIs) on
  the GPU; after each film frame its points/normals/triangles are copied into /World/Water in the render
  stage, with an OmniGlass water material (IOR 1.333). Optional: the part inside the mug is split off and
  gets a tea colour that steeps in over time.

Use from a renderer (after the render stage is open):
    from water_layer import WaterLayer
    water = WaterLayer(S, stage)              # S = dict(np.load(shot.npz)), stage = render stage
    ...per frame k (increasing):  water.update(k)   # steps the fluid to frame k, updates /World/Water
    water.stats[-1] -> {'k', 't', 'mug_ml', 'kettle_ml', 'other_ml'}
Sim-only bake (no render) to a cache the renderer can replay (eye and film cameras then see identical water):
    <isaac-sim>/python.bat water_layer.py <shot_dir> <cache_dir> [--frames a:b]
"""
import json
import math
import os
import time

import numpy as np

KETTLE = "/World/Objects/object_0295"
MUG = "/World/Objects/object_0316"
BASE_BOX = ((2.49, 2.28, 0.86), (2.65, 2.44, 0.885))          # kettle base (twin collision box)
COUNTER_TOP = 0.86

# ---------------------------------------------------------------- geometry (object frames, metres)
# kettle body: outer radius of the twin's visual mesh by height; the cavity is 4 mm inside it
KETTLE_OUTER = [(0.015, 0.077), (0.066, 0.078), (0.12, 0.073), (0.155, 0.0695), (0.192, 0.066), (0.205, 0.0645)]
KETTLE_WALL = 0.004
KETTLE_FLOOR = 0.015
KETTLE_LID = 0.205
SPOUT_ROOT_Z = 0.155           # wall is open above this on the -x side (into the spout beak)
SPOUT_HALF_ANGLE = math.radians(24)
SPOUT_TIP = np.array([-0.094, 0.0, 0.205])                   # middle of the beak's mouth
BEAK_V = 28.0          # deg: the beak's lip is a V (the visual beak narrows to its tip), not a flat channel
# mug: inner wall (set_fix.usda visual mug, = the lead's physics mug), floor, rim
MUG_INNER = [(0.009, 0.0275), (0.017, 0.0288), (0.081, 0.0295), (0.085, 0.0298)]   # set_fix.usda mug (29.5 mm)
MUG_FLOOR = 0.009
MUG_RIM = 0.085

# PBD particle volume at rest, as a multiple of fluidRestOffset^3 (calibrated: settle test in the mug)
VP_FACTOR = 4.2

WATER_COLOR = (0.97, 0.99, 1.0)
TEA_COLOR = (0.085, 0.028, 0.006)        # steeped black tea seen from above (dark amber)


def quat_from_axis(axis, ang):
    s = math.sin(ang / 2)
    return (math.cos(ang / 2), axis[0] * s, axis[1] * s, axis[2] * s)


def usd_matrix(p, q):
    """USD row-vector matrix from position + quaternion (w,x,y,z)."""
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R.T
    M[3, :3] = p
    return M


def ring_band(z0, r0, z1, r1, n, thick, skip=None):
    """Boxes forming the wall of a frustum band (inner face from (r0,z0) to (r1,z1)), n segments.
    Returns [(center, half_extents, quat)] in the object frame; skip(phi) -> True leaves a segment out."""
    out = []
    dr, dz = r1 - r0, z1 - z0
    L = math.hypot(dr, dz)
    tilt = math.atan2(dr, dz)       # inner face's angle from vertical (rotation about the tangential axis)
    rmax = max(r0, r1)
    half_w = rmax * math.tan(math.pi / n) * 1.25 + thick * math.tan(math.pi / n)
    for i in range(n):
        phi = 2 * math.pi * (i + 0.5) / n
        if skip and skip(phi):
            continue
        rm, zm = (r0 + r1) / 2, (z0 + z1) / 2
        # outward normal of the inner face in the (radial, z) plane: (dz, -dr)/L
        nr, nz = dz / L, -dr / L
        cr, cz = rm + nr * thick / 2, zm + nz * thick / 2
        c = (cr * math.cos(phi), cr * math.sin(phi), cz)
        # box local axes: x = radial normal, y = tangential, z = along the face; rotation = Rz(phi) * Ry(tilt)
        qz = quat_from_axis((0, 0, 1), phi)
        qy = quat_from_axis((0, 1, 0), tilt)
        out.append((c, (thick / 2, half_w, L / 2 + 0.0015), qmul(qz, qy)))
    return out


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def box_between(p0, p1, half_w, thick, side):
    """A slab whose top (side=+1) or bottom (side=-1) face runs from p0 to p1 in the x-z plane (y = 0)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    ang = math.atan2(-d[2], d[0])          # rotation about +y that maps +x onto d
    n = np.array([-d[2], 0, d[0]]) / L     # normal (points up for a left-to-right... sign fixed below)
    if n[2] < 0:
        n = -n
    c = (p0 + p1) / 2 - side * n * thick / 2
    return (tuple(c), (L / 2, half_w, thick / 2), quat_from_axis((0, 1, 0), ang))


def quat_from_matrix(R):
    """quaternion (w,x,y,z) of a rotation matrix whose columns are the rotated x, y, z axes"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


def v_groove(p0, p1, ang, half_w, thick):
    """two slabs forming a V-shaped gutter whose bottom line runs from p0 to p1 (y = 0), sides rising at ang:
    funnels the water to the middle of the beak's lip so it leaves as a rope, not a 2.5 cm sheet"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.cross(u, [0.0, 1.0, 0.0])
    n = n if n[2] > 0 else -n
    out = []
    for s in (-1.0, 1.0):
        a = s * np.array([0.0, 1.0, 0.0]) * math.cos(ang) + n * math.sin(ang)
        x = u
        z = np.cross(x, a)
        if z[2] < 0:                       # keep the frame right-handed with the slab's top face up
            x = -u
            z = np.cross(x, a)
        c = (p0 + p1) / 2 + a * half_w - z * thick / 2
        R = np.stack([x, a, z], axis=1)
        out.append(("box", tuple(c), (np.linalg.norm(p1 - p0) / 2, half_w, thick / 2), quat_from_matrix(R)))
    return out


def kettle_shapes():
    shapes = []
    inner = [(z, r - KETTLE_WALL) for z, r in KETTLE_OUTER]
    spout = lambda phi: abs(math.atan2(math.sin(phi - math.pi), math.cos(phi - math.pi))) < SPOUT_HALF_ANGLE  # noqa
    for (z0, r0), (z1, r1) in zip(inner[:-1], inner[1:]):
        shapes += [("box",) + s for s in ring_band(z0, r0, z1, r1, 32, 0.008,
                                                   skip=spout if z0 >= SPOUT_ROOT_Z - 1e-6 else None)]
    shapes.append(("cyl", (0, 0, KETTLE_FLOOR - 0.006), 0.079, 0.006))        # floor
    shapes.append(("cyl", (0, 0, KETTLE_LID + 0.005), 0.068, 0.005))          # lid
    # spout beak: floor slopes up from the wall opening to the mouth; roof just under the visual top lip
    # (the floor slab starts inside the cavity and is thick, so no gap opens under it at the wall)
    shapes.append(("box",) + box_between((-0.045, 0, 0.142), (-0.093, 0, 0.197), 0.03, 0.012, side=+1))
    shapes.append(("box",) + box_between((-0.050, 0, 0.207), (-0.097, 0, 0.2115), 0.03, 0.005, side=-1))
    if BEAK_V > 0:                       # V-shaped lip on top of the flat beak floor (z on the floor line)
        zf = lambda x: 0.142 + (x + 0.045) * (0.197 - 0.142) / (-0.093 + 0.045)  # noqa: E731
        shapes += v_groove((-0.060, 0, zf(-0.060)), (-0.094, 0, zf(-0.094)), math.radians(BEAK_V), 0.016, 0.006)
    for sgn in (-1, 1):                                                       # beak side walls (taper)
        a, b = np.array([-0.050, sgn * 0.027]), np.array([-0.097, sgn * 0.0125])
        d = b - a
        L = np.linalg.norm(d)
        nrm = np.array([-d[1], d[0]]) / L * sgn
        if nrm[1] * sgn < 0:
            nrm = -nrm
        c = (a + b) / 2 + nrm * 0.0025
        shapes.append(("box", (c[0], c[1], 0.182), (L / 2 + 0.002, 0.0025, 0.036),
                       quat_from_axis((0, 0, 1), math.atan2(d[1], d[0]))))
    return shapes


def mug_shapes():
    shapes = []
    for (z0, r0), (z1, r1) in zip(MUG_INNER[:-1], MUG_INNER[1:]):
        shapes += [("box",) + s for s in ring_band(z0, r0 + 0.0005, z1, r1 + 0.0005, 32, 0.006)]
    shapes.append(("cyl", (0, 0, MUG_FLOOR - 0.007), 0.031, 0.007))
    return shapes


# example spoon / tea bag colliders (object frames) - replace with the real prop geometry when it exists
SPOON_EXAMPLE = [("capsule", (0.0, 0.0, 0.07), 0.003, 0.06, "Z"), ("box", (0.0, 0.0, 0.005), (0.009, 0.0015, 0.014), (1, 0, 0, 0))]
TEABAG_EXAMPLE = [("box", (0.0, 0.0, 0.0), (0.025, 0.003, 0.03), (1, 0, 0, 0))]


# ---------------------------------------------------------------- the layer
class WaterLayer:
    def __init__(self, S, render_stage=None, *, fill_ml=400.0, fro=0.00125, substeps=4, settle_s=1.0,
                 cache_dir=None, props=None, tea_from_frame=None, tea_steep_s=40.0, iso_spacing=None,
                 iters=16, fill_in="kettle", wall_friction=0.02, look="glass", opacity=0.15, cohesion=0.1, aniso=(1.5, 1.0, 2.0), surf_dist=1.3, mesh_smooth=12,
                 verbose=True):
        """S: shot dict (bake.py format). render_stage: the open render stage (None = sim-only bake).
        props: {name: (path_in_shot, shapes)} extra kinematic colliders (spoon, tea bag).
        tea_from_frame: frame from which the water in the mug steeps to tea colour (None = plain water)."""
        self.S, self.rstage, self.verbose = S, render_stage, verbose
        self.fps = float(S["fps"])
        self.n_sub = int(substeps)
        self.dt = 1.0 / (self.fps * self.n_sub)
        self.fro = fro
        self.vp = VP_FACTOR * fro ** 3
        self.iters, self.fill_in, self.wall_friction = iters, fill_in, wall_friction
        self.look, self.opacity = look, opacity
        self.cohesion, self.aniso, self.surf_dist, self.mesh_smooth = cohesion, aniso, surf_dist, mesh_smooth
        self.solid_mat = None
        self.cache_dir = cache_dir
        self.tea_from, self.tea_steep = tea_from_frame, tea_steep_s
        self.stats = []
        self.k = None
        self.t_sim = 0.0
        self.timing = {"sim": 0.0, "copy": 0.0, "frames": 0}
        self.paths = [str(p) for p in S["obj_paths"]]
        self.bodies = {"kettle": (KETTLE, kettle_shapes()), "mug": (MUG, mug_shapes())}
        for name, spec in (props or {}).items():
            self.bodies[name] = spec
        self.cached = bool(cache_dir) and os.path.exists(os.path.join(cache_dir, "complete.json"))
        if render_stage is not None:
            self._make_render_prims()
        if self.cached:
            self.stats = json.load(open(os.path.join(cache_dir, "stats.json")))
            if verbose:
                print(f"[water] replaying cache {cache_dir} ({len(self.stats)} frames)", flush=True)
            return
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._build_sim(fill_ml, iso_spacing)
        self._settle(settle_s)

    # ------------------------------------------------------------ poses from the shot
    def body_matrix(self, name, k):
        path = self.bodies[name][0]
        if path in self.paths:
            return np.asarray(self.S["obj_mats"][k, self.paths.index(path)], float)
        names = [str(n) for n in self.S.get("prop_names", [])]
        if path in names:
            i = names.index(path)
            return usd_matrix(self.S["prop_xpos"][k, i], self.S["prop_xquat"][k, i])
        raise KeyError(f"{name}: {path} is not in the shot")

    # ------------------------------------------------------------ physics stage
    def _build_sim(self, fill_ml, iso_spacing):
        import carb
        import omni.physx
        import omni.physx.bindings._physx as pb
        from omni.physx.scripts import particleUtils, physicsUtils
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt
        self.Gf, self.Vt, self.UsdGeom = Gf, Vt, UsdGeom
        st = carb.settings.get_settings()
        st.set(pb.SETTING_UPDATE_TO_USD, True)
        st.set(pb.SETTING_UPDATE_PARTICLES_TO_USD, True)
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/W").GetPrim())
        sc = UsdPhysics.Scene.Define(stage, "/W/scene")
        sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        sc.CreateGravityMagnitudeAttr(9.81)
        px = PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim())
        px.CreateEnableGPUDynamicsAttr(True)
        px.CreateBroadphaseTypeAttr("GPU")
        px.CreateSolverTypeAttr("TGS")
        px.CreateTimeStepsPerSecondAttr(int(round(1 / self.dt)))
        self.pstage = stage
        fro = self.fro
        pco = fro / 0.6
        solid_rest = 0.0009
        # glazed ceramic / plastic: low friction for the water
        self.solid_mat = UsdShade.Material.Define(stage, "/W/solid_mat")
        pm = UsdPhysics.MaterialAPI.Apply(self.solid_mat.GetPrim())
        pm.CreateStaticFrictionAttr(self.wall_friction)
        pm.CreateDynamicFrictionAttr(self.wall_friction)
        pm.CreateRestitutionAttr(0.0)
        # ---- kinematic bodies
        self.ops = {}
        for name, (path, shapes) in self.bodies.items():
            x = UsdGeom.Xform.Define(stage, f"/W/{name}")
            rb = UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
            rb.CreateKinematicEnabledAttr(True)
            op = x.AddTransformOp()
            op.Set(Gf.Matrix4d(self.body_matrix(name, 0).tolist()))
            self.ops[name] = op
            for i, sh in enumerate(shapes):
                self._shape(stage, f"/W/{name}/s{i}", sh, pco + 0.001, solid_rest)
        # ---- statics: counter top, kettle base
        for nm, (lo, hi) in {"counter": ((0.9, 2.05, COUNTER_TOP - 0.04), (2.9, 2.66, COUNTER_TOP)),
                             "base": BASE_BOX, "floor": ((-1, -1, -0.1), (4, 4, 0.0))}.items():
            lo, hi = np.array(lo), np.array(hi)
            self._shape(stage, f"/W/static_{nm}", ("box", tuple((lo + hi) / 2), tuple((hi - lo) / 2), (1, 0, 0, 0)),
                        pco + 0.001, solid_rest)
        # ---- particle system
        ps = particleUtils.add_physx_particle_system(
            stage, "/W/ps", simulation_owner=sc.GetPath(), particle_contact_offset=pco, contact_offset=pco + 0.001,
            fluid_rest_offset=fro, solid_rest_offset=solid_rest, rest_offset=solid_rest, enable_ccd=True,
            solver_position_iterations=self.iters, max_neighborhood=96, max_velocity=4.0)
        UsdShade.Material.Define(stage, "/W/water_pbd")
        particleUtils.add_pbd_particle_material(stage, "/W/water_pbd", cohesion=self.cohesion, damping=0.0,
                                                friction=self.wall_friction,
                                                surface_tension=0.0074, viscosity=1.7e-6, density=1000.0,
                                                adhesion=0.0)
        physicsUtils.add_physics_material_to_prim(stage, ps.GetPrim(), "/W/water_pbd")
        sm = PhysxSchema.PhysxParticleSmoothingAPI.Apply(ps.GetPrim())
        sm.CreateStrengthAttr(0.8)
        an = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(ps.GetPrim())
        an.CreateScaleAttr(self.aniso[0])
        an.CreateMinAttr(self.aniso[1])
        an.CreateMaxAttr(self.aniso[2])
        iso = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(ps.GetPrim())
        gs = iso_spacing or fro * 1.2
        iso.CreateGridSpacingAttr(gs)
        iso.CreateSurfaceDistanceAttr(fro * self.surf_dist)
        iso.CreateGridSmoothingRadiusAttr(fro * 2.0)
        iso.CreateNumMeshSmoothingPassesAttr(self.mesh_smooth)
        iso.CreateNumMeshNormalSmoothingPassesAttr(4)
        iso.CreateMaxSubgridsAttr(8192)
        iso.CreateMaxVerticesAttr(1024 * 1024)
        iso.CreateMaxTrianglesAttr(2 * 1024 * 1024)
        # ---- water: fill the kettle cavity (or, for calibration, the mug) from the bottom up
        pts = self._fill_points(fill_ml) if self.fill_in == "kettle" else self._fill_mug(fill_ml)
        M0 = self.body_matrix(self.fill_in, 0)
        world = pts @ M0[:3, :3] + M0[3, :3]
        self.n = len(world)
        particleUtils.add_physx_particleset_pointinstancer(
            stage, Sdf.Path("/W/water"), Vt.Vec3fArray.FromNumpy(world.astype(np.float32)),
            Vt.Vec3fArray.FromNumpy(np.zeros_like(world, np.float32)), "/W/ps", self_collision=True, fluid=True,
            particle_group=0, particle_mass=1000.0 * self.vp, density=0.0)
        UsdGeom.Sphere.Get(stage, "/W/water/particlePrototype0").CreateRadiusAttr(fro)
        self.inst = UsdGeom.PointInstancer(stage.GetPrimAtPath("/W/water"))
        cache = UsdUtils.StageCache.Get()
        cache.Insert(stage)
        self.sim = omni.physx.get_physx_simulation_interface()
        ok = self.sim.attach_stage(cache.GetId(stage).ToLongInt())
        # the replicator step plays/pauses the timeline; keep that from stepping the fluid on its own
        # (checked in water_render.py: the fluid moves 0.000 mm during a render step)
        try:
            from omni.physics.stageupdate import get_physics_stage_update_node_interface
            get_physics_stage_update_node_interface().block_timeline_events(True)
        except Exception as e:  # noqa: BLE001
            print("[water] WARNING: could not block timeline events:", e, flush=True)
        if self.verbose:
            print(f"[water] {self.n} particles ({self.n * self.vp * 1e6:.0f} ml), r {fro * 1e3:.2f} mm, "
                  f"{self.n_sub} substeps/frame ({1 / self.dt:.0f} Hz), attached {ok}", flush=True)

    def _shape(self, stage, path, sh, contact, rest):
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
        kind = sh[0]
        if kind == "box":
            _, c, h, q = sh
            g = UsdGeom.Cube.Define(stage, path)
            g.CreateSizeAttr(2.0)
            g.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
            g.AddOrientOp().Set(Gf.Quatf(*map(float, q)))
            g.AddScaleOp().Set(Gf.Vec3f(*map(float, h)))
        elif kind == "cyl":
            _, c, r, hh = sh
            g = UsdGeom.Cylinder.Define(stage, path)
            g.CreateRadiusAttr(float(r))
            g.CreateHeightAttr(float(2 * hh))
            g.CreateAxisAttr("Z")
            g.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
        elif kind == "capsule":
            _, c, r, hh, axis = sh
            g = UsdGeom.Capsule.Define(stage, path)
            g.CreateRadiusAttr(float(r))
            g.CreateHeightAttr(float(2 * hh))
            g.CreateAxisAttr(axis)
            g.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
        else:
            raise ValueError(kind)
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        if self.solid_mat is not None:
            from pxr import UsdShade
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(self.solid_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        pc = PhysxSchema.PhysxCollisionAPI.Apply(g.GetPrim())
        pc.CreateContactOffsetAttr(float(contact))
        pc.CreateRestOffsetAttr(0.0)

    def _fill_points(self, fill_ml):
        """Particles at their rest spacing inside the kettle cavity (kettle frame), lowest first."""
        s = self.vp ** (1 / 3)
        n = int(round(fill_ml * 1e-6 / self.vp))
        m = self.fro + 0.0008
        r_in = KETTLE_OUTER[0][1] - KETTLE_WALL - m
        g = np.mgrid[-r_in:r_in:s, -r_in:r_in:s, KETTLE_FLOOR + m:0.12:s].reshape(3, -1).T
        zr = np.interp(g[:, 2], [z for z, _ in KETTLE_OUTER], [r - KETTLE_WALL - m for _, r in KETTLE_OUTER])
        g = g[np.hypot(g[:, 0], g[:, 1]) < zr]
        g = g[np.argsort(g[:, 2], kind="stable")][:n]
        rng = np.random.default_rng(1)
        return g + rng.uniform(-0.05, 0.05, g.shape) * s

    def _fill_mug(self, fill_ml):
        s = self.vp ** (1 / 3)
        n = int(round(fill_ml * 1e-6 / self.vp))
        m = self.fro + 0.0006
        g = np.mgrid[-0.03:0.03:s, -0.03:0.03:s, MUG_FLOOR + m:0.2:s].reshape(3, -1).T
        zr = np.interp(g[:, 2], [z for z, _ in MUG_INNER], [r - m for _, r in MUG_INNER])
        g = g[np.hypot(g[:, 0], g[:, 1]) < zr]
        return g[np.argsort(g[:, 2], kind="stable")][:n]

    def _set_bodies(self, ka, kb, a):
        Gf = self.Gf
        for name, op in self.ops.items():
            Ma, Mb = self.body_matrix(name, ka), self.body_matrix(name, kb)
            if a >= 1.0 or np.allclose(Ma, Mb):
                op.Set(Gf.Matrix4d(Mb.tolist()))
                continue
            ga, gb = Gf.Matrix4d(Ma.tolist()), Gf.Matrix4d(Mb.tolist())
            qa, qb = ga.ExtractRotationQuat(), gb.ExtractRotationQuat()
            q = Gf.Slerp(a, qa, qb)
            p = ga.ExtractTranslation() * (1 - a) + gb.ExtractTranslation() * a
            op.Set(Gf.Matrix4d().SetTransform(Gf.Rotation(q), p))

    def _step(self):
        self.sim.simulate(self.dt, self.t_sim)
        self.sim.fetch_results()
        self.t_sim += self.dt

    def _settle(self, secs):
        t0 = time.time()
        self._set_bodies(0, 0, 1.0)
        for _ in range(int(round(secs / self.dt))):
            self._step()
        self.k = 0
        self._record(0)
        if self.verbose:
            s = self.stats[-1]
            print(f"[water] settled {secs:.1f} s in {time.time() - t0:.1f} s: kettle {s['kettle_ml']:.0f} ml, "
                  f"mug {s['mug_ml']:.0f} ml, other {s['other_ml']:.0f} ml", flush=True)

    # ------------------------------------------------------------ measurements
    def positions(self):
        return np.asarray(self.inst.GetPositionsAttr().Get(), float)

    def _local(self, name, k, P):
        M = self.body_matrix(name, k)
        return (P - M[3, :3]) @ M[:3, :3].T

    def _record(self, k):
        P = self.positions()
        lm = self._local("mug", k, P)
        in_mug = (np.hypot(lm[:, 0], lm[:, 1]) < 0.0335) & (lm[:, 2] > 0.0) & (lm[:, 2] < MUG_RIM + 0.004)
        lk = self._local("kettle", k, P)
        in_k = (np.hypot(lk[:, 0], lk[:, 1]) < 0.08) & (lk[:, 2] > 0.0) & (lk[:, 2] < 0.215) | \
               ((lk[:, 0] < -0.04) & (lk[:, 0] > -0.1) & (np.abs(lk[:, 1]) < 0.03) & (lk[:, 2] > 0.14) & (lk[:, 2] < 0.216))
        ml = 1e6 * self.vp
        rec = {"k": int(k), "t": round(k / self.fps, 4), "mug_ml": round(float(in_mug.sum() * ml), 1),
               "kettle_ml": round(float(in_k.sum() * ml), 1),
               "other_ml": round(float((~in_mug & ~in_k).sum() * ml), 1),
               "zmin": round(float(P[:, 2].min()), 4),
               "tilt_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, self.body_matrix("kettle", k)[2, 2])))), 1)}
        if in_mug.any():
            rec["mug_level_mm"] = round(float(np.percentile(lm[in_mug, 2], 98) * 1e3), 1)
        low = P[:, 2] < COUNTER_TOP - 0.01
        if low.any():
            rec["below_counter_ml"] = round(float(low.sum() * ml), 1)
            rec["below_counter_xy"] = [round(float(v), 3) for v in P[low, :2].mean(0)]
        self.stats.append(rec)

    # ------------------------------------------------------------ per film frame
    def update(self, k):
        """Bring the water to film frame k (frames must not go backwards) and update /World/Water."""
        if self.cached:
            self._load_cached(k)
            return
        if k < self.k:
            raise ValueError(f"water is at frame {self.k}; it cannot go back to {k}")
        t0 = time.time()
        while self.k < k:
            ka, kb = self.k, self.k + 1
            for j in range(1, self.n_sub + 1):
                self._set_bodies(ka, kb, j / self.n_sub)
                self._step()
            self.k = kb
            self._record(kb)
        t1 = time.time()
        surf = self._read_surface()
        if self.cache_dir:
            np.savez_compressed(os.path.join(self.cache_dir, f"surf_{k:05d}.npz"), **surf)
        if self.rstage is not None:
            self._show(surf, k)
        self.timing["sim"] += t1 - t0
        self.timing["copy"] += time.time() - t1
        self.timing["frames"] += 1

    def finish(self, complete=True):
        """Write stats (and mark the cache complete so later renders replay it)."""
        if self.cache_dir and not self.cached:
            json.dump(self.stats, open(os.path.join(self.cache_dir, "stats.json"), "w"))
            if complete:
                json.dump({"frames": [self.stats[0]["k"], self.stats[-1]["k"]], "particles": self.n,
                           "fro": self.fro, "substeps": self.n_sub}, open(os.path.join(self.cache_dir, "complete.json"), "w"))

    def _read_surface(self):
        m = self.UsdGeom.Mesh(self.pstage.GetPrimAtPath("/W/ps/Isosurface"))
        pts = np.asarray(m.GetPointsAttr().Get() or [], np.float32).reshape(-1, 3)
        nrm = np.asarray(m.GetNormalsAttr().Get() or [], np.float32).reshape(-1, 3)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get() or [], np.int32).reshape(-1, 3)
        return {"p": pts, "n": nrm.astype(np.float16), "i": idx}

    def _load_cached(self, k):
        f = os.path.join(self.cache_dir, f"surf_{k:05d}.npz")
        if not os.path.exists(f):
            return
        d = np.load(f)
        self._show({"p": d["p"], "n": d["n"], "i": d["i"]}, k)

    # ------------------------------------------------------------ render prims
    def _material(self, name, look, col):
        """water looks: 'glass' = OmniGlass MDL (IOR 1.333), 'preview' = UsdPreviewSurface translucent
        (the way looks.usda does the glass jar), 'debug' = opaque blue (to see where the fluid is)."""
        from pxr import Gf, Sdf, UsdShade
        st = self.rstage
        mp = f"/World/Water/Looks/{name}_{look}"
        mat = UsdShade.Material.Define(st, mp)
        sh = UsdShade.Shader.Define(st, mp + "/Shader")
        if name == "tea" and look != "debug":
            # tea: clear at first, then a glossy amber-brown liquid as it steeps (in a black mug a clear
            # glass-like tea would just look black); colour and opacity follow the steeping strength
            sh.CreateIdAttr("UsdPreviewSurface")
            c = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            self.tea_opacity = sh.CreateInput("opacity", Sdf.ValueTypeNames.Float)
            self.tea_opacity.Set(self.opacity)
            sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.333)
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.02)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(1.0)
            sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.01)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            c.Set(Gf.Vec3f(*col))
            return mat, c, look
        if look == "glass":
            sh.SetSourceAsset("OmniGlass.mdl", "mdl")
            sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
            sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.333)
            c = sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f)
            sh.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.001)
            sh.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.0)
            sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
            for o in ("surface", "volume"):
                getattr(mat, f"Create{o.capitalize()}Output")("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        else:
            sh.CreateIdAttr("UsdPreviewSurface")
            c = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            if look == "debug":
                sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
            else:
                sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.333)
                sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(self.opacity)
                sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.01)
                sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        c.Set(Gf.Vec3f(*((0.1, 0.35, 1.0) if look == "debug" else col)))
        return mat, c, look

    def _make_render_prims(self):
        from pxr import Sdf, UsdGeom, UsdShade
        st = self.rstage
        UsdGeom.Xform.Define(st, "/World/Water")
        self.rmesh = {}
        for name in ("water", "tea"):
            me = UsdGeom.Mesh.Define(st, f"/World/Water/{name}")
            me.CreateSubdivisionSchemeAttr("none")
            me.CreateDoubleSidedAttr(False)
            mat, c, _ = self._material(name, self.look, WATER_COLOR)
            UsdShade.MaterialBindingAPI.Apply(me.GetPrim()).Bind(mat)
            self.rmesh[name] = (me, c)

    def set_look(self, look):
        """rebind the water meshes to another look (checking stills)"""
        from pxr import UsdShade
        for name in ("water", "tea"):
            me, _ = self.rmesh[name]
            mat, c, _ = self._material(name, look, WATER_COLOR)
            UsdShade.MaterialBindingAPI.Apply(me.GetPrim()).Bind(mat)
            self.rmesh[name] = (me, c)
        self.look = look

    def _tea_strength(self, k):
        if self.tea_from is None or k < self.tea_from:
            return 0.0
        return 1.0 - math.exp(-(k - self.tea_from) / self.fps / (self.tea_steep / 3))

    def _show(self, surf, k):
        Vt, Gf = __import__("pxr").Vt, __import__("pxr").Gf
        p, n, idx = surf["p"], surf["n"].astype(np.float32), surf["i"]
        parts = {"water": idx, "tea": idx[:0]}
        tea = self._tea_strength(k)
        if tea > 0 and len(idx):
            c = p[idx].mean(1)
            lm = self._local("mug", k, c.astype(float))
            inside = (np.hypot(lm[:, 0], lm[:, 1]) < 0.0335) & (lm[:, 2] > 0) & (lm[:, 2] < MUG_RIM + 0.002)
            parts = {"water": idx[~inside], "tea": idx[inside]}
        for name, tri in parts.items():
            me, colattr = self.rmesh[name]
            if len(tri) == 0:
                me.GetPointsAttr().Set(Vt.Vec3fArray())
                me.GetFaceVertexIndicesAttr().Set(Vt.IntArray())
                me.GetFaceVertexCountsAttr().Set(Vt.IntArray())
                continue
            used = np.unique(tri)
            remap = np.full(len(p), -1, np.int32)
            remap[used] = np.arange(len(used), dtype=np.int32)
            pp, nn = p[used], n[used]
            me.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pp)))
            me.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(nn)))
            me.SetNormalsInterpolation("vertex")
            me.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.ascontiguousarray(remap[tri].ravel())))
            me.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(tri), 3, np.int32)))
            me.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, pp.min(0))), Gf.Vec3f(*map(float, pp.max(0)))]))
            if name == "tea" and self.look != "debug":
                colattr.Set(Gf.Vec3f(*[(1 - tea) * w + tea * t for w, t in zip(WATER_COLOR, TEA_COLOR)]))
                self.tea_opacity.Set(float(self.opacity + (0.9 - self.opacity) * tea))


# ---------------------------------------------------------------- sim-only bake
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("shot")
    ap.add_argument("cache")
    ap.add_argument("--frames", default="")
    ap.add_argument("--fill", type=float, default=400.0)
    ap.add_argument("--fro", type=float, default=0.00125)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=1, help="save the surface every n-th frame only (tuning)")
    ap.add_argument("--log", type=int, default=30)
    ap.add_argument("--fill-in", default="kettle", help="kettle | mug: where the water starts")
    A = ap.parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    S = dict(np.load(os.path.join(A.shot, "shot.npz"), allow_pickle=True))
    F = len(S["t"])
    a, b = (0, F - 1) if not A.frames else map(int, A.frames.split(":"))
    W = WaterLayer(S, None, fill_ml=A.fill, fro=A.fro, substeps=A.substeps, settle_s=A.settle, cache_dir=A.cache,
                   fill_in=A.fill_in)
    t0 = time.time()
    for k in range(1, b + 1):
        if A.every > 1 and k % A.every and k != b:
            W.cache_dir, cd = None, W.cache_dir
            W.update(k)
            W.cache_dir = cd
        else:
            W.update(k)
        if k % A.log == 0:
            s = W.stats[-1]
            print(f"frame {k} t {s['t']:.2f} tilt {s['tilt_deg']:5.1f}: mug {s['mug_ml']:.0f} ml (level {s.get('mug_level_mm', 0)} mm) "
                  f"kettle {s['kettle_ml']:.0f} other {s['other_ml']:.0f} zmin {s['zmin']:.3f} "
                  f"| {(time.time() - t0) / k * 1000:.0f} ms/frame", flush=True)
    W.finish(complete=(A.every == 1))
    print("done", A.cache, flush=True)
    app.close()
