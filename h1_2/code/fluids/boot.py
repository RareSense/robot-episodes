"""Start the H1-2 Sim with scene flags changed and a hook that runs just before physics loads.

Uses h12_base.py, a frozen copy of h12_physx.py, so the numbers here do not move while the main
controller file is being edited. Nothing in Sim's logic is changed: get_physx_interface() is wrapped so
that force_load_physics_from_usd() first runs `hook(sim_stage, scene_prim)`.
In the real controller the same thing is two lines in Sim.__init__ (see NOTES.md).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))    # release_paths.py lives at the code root
import release_paths as RP
import importlib
import sys

sys.path.insert(0, RP.code('fluids'))


class _Shim:
    def __init__(self, real, fn):
        self.real, self.fn = real, fn

    def force_load_physics_from_usd(self):
        self.fn()
        return self.real.force_load_physics_from_usd()

    def __getattr__(self, k):
        return getattr(self.real, k)


def set_scene(st, gpu=True, broadphase="GPU", solver="TGS", hz=600):
    from pxr import PhysxSchema, UsdPhysics
    sc = next(p for p in st.Traverse() if p.IsA(UsdPhysics.Scene))
    px = PhysxSchema.PhysxSceneAPI.Apply(sc)
    px.CreateEnableGPUDynamicsAttr(bool(gpu))
    px.CreateBroadphaseTypeAttr(broadphase)
    px.CreateSolverTypeAttr(solver)
    px.CreateTimeStepsPerSecondAttr(int(hz))
    return sc


def set_iters(st, art=8, bodies=8):
    """Solver position iterations. The robot's USD default is 32 (scene GPU cost scales with the largest count)."""
    from pxr import PhysxSchema, UsdPhysics
    n = 0
    for p in st.Traverse():
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            a = PhysxSchema.PhysxArticulationAPI.Apply(p)
            a.CreateSolverPositionIterationCountAttr(int(art))
            a.CreateSolverVelocityIterationCountAttr(1)
        elif p.HasAPI(UsdPhysics.RigidBodyAPI) and not p.GetPath().pathString.startswith("/World/h1_2"):
            r = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
            r.CreateSolverPositionIterationCountAttr(int(bodies))
            r.CreateSolverVelocityIterationCountAttr(1)
            n += 1
    print("solver iterations: articulation", art, "bodies", bodies, "on", n, flush=True)


def make_sim(app, hook=None, gpu=True, broadphase="GPU", solver="TGS", xy=(0.62, 1.15), yaw_deg=90.0, z=1.05,
             module="h12_base", **kw):
    """hook(stage, scene_prim) runs after the room + robot are in the stage, before PhysX parses it."""
    import omni.usd
    H = importlib.import_module(module)
    real = H.__dict__.get("_real_gpi") or H.get_physx_interface
    H._real_gpi = real

    def fn():
        st = omni.usd.get_context().get_stage()
        sc = set_scene(st, gpu, broadphase, solver)
        if hook:
            hook(st, sc)

    H.get_physx_interface = lambda: _Shim(real(), fn)
    s = H.Sim(app, xy=xy, yaw_deg=yaw_deg, z=z, **kw)
    H.get_physx_interface = real
    return H, s
