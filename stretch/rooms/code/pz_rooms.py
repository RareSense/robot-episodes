"""Stretch plate-to-rack skill in N different rooms at once on GPU PhysX, Isaac Lab. AI-operated.
Every copy gets its own room, generated in code: room size, counter anywhere at any angle, oven and plate anywhere on it, furniture on the floor, robot starting anywhere at any heading.
The route to the counter is planned on a floor grid of that room; the hand work is computed in the counter's own frame. Door authored open (test start).
State is written once at the spawn (the room), never after. No welds. Walls and furniture are kinematic boxes: they block, they cannot be pushed.
usage: pz_rooms.py --tag T --num_envs 30 --spacing 9 [--seed 1] [--cam] [--shots 30]"""
import argparse
import json
import math
import os
import time

from isaaclab.app import add_launcher_args, launch_simulation

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--num_envs", type=int, default=1)
ap.add_argument("--spacing", type=float, default=4.0)
ap.add_argument("--secs", type=float, default=60.0)
ap.add_argument("--iters", type=int, default=32)
ap.add_argument("--cam", action="store_true")
ap.add_argument("--joints", action="store_true")
ap.add_argument("--probe", action="store_true")
ap.add_argument("--track", default="")
ap.add_argument("--shot_from", type=float, default=0.)
ap.add_argument("--patches", type=int, default=0)
ap.add_argument("--contacts", type=int, default=0)
ap.add_argument("--cam_w", type=int, default=640)
ap.add_argument("--cam_h", type=int, default=480)
ap.add_argument("--row0", type=int, default=0)
ap.add_argument("--gear", type=float, default=0.)
ap.add_argument("--mimic_nf", type=float, default=0.)
ap.add_argument("--keep_newton", action="store_true")
ap.add_argument("--mimic_dr", type=float, default=0.)
ap.add_argument("--asset", default="stretch_pz")
ap.add_argument("--print_rows", type=int, default=150)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--shots", type=int, default=32)
ap.add_argument("--rec_every", type=int, default=5)
ap.add_argument("--vid_dt", type=float, default=0.)
ap.add_argument("--vid_r", type=int, default=3)
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = args.cam

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

A = "C:/Users/raresense/robot_dataset/assets"
RUN = f"E:/isaaclab_mimic_test/rooms/{args.tag}"
os.makedirs(RUN, exist_ok=True)
DT = 1 / 600
EVERY = 20                                  # the recording holds one target row per 20 physics steps
# Gains, efforts, armature, friction: stretch_core.py values, as in stretch_live.py
GAINS = {"joint_lift": (400., 135.), "joint_arm_l3": (2400., 160.), "joint_wrist_yaw": (20., 2.), "joint_wrist_pitch": (50., 3.), "joint_wrist_roll": (20., 2.),
         "joint_gripper_slide": (4000., 124.), "joint_head_pan": (10., 1.), "joint_head_tilt": (10., 1.), "joint_left_wheel": (0., 30.), "joint_right_wheel": (0., 30.)}
EFFORT = {"joint_lift": 70.0, "joint_arm_l3": 224.0, "joint_wrist_yaw": 4.6, "joint_wrist_pitch": 6.0, "joint_wrist_roll": 1.8, "joint_gripper_slide": 142.0,
          "joint_head_pan": 2.0, "joint_head_tilt": 2.0, "joint_left_wheel": 33.3, "joint_right_wheel": 33.3}
ARM = {"joint_left_wheel": .1, "joint_right_wheel": .1, "joint_lift": .01, "joint_arm_l3": .01, "joint_wrist_yaw": .01, "joint_wrist_pitch": .01, "joint_wrist_roll": .01, "joint_gripper_slide": .1}
FRIC = {"joint_left_wheel": 1., "joint_right_wheel": 1., "joint_lift": 1.5, "joint_arm_l3": 1.5, "joint_wrist_yaw": .1, "joint_wrist_pitch": .1, "joint_wrist_roll": .1}

ROBOT = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=(f"{A}/{args.asset}.usda" if args.asset != "lab" else "C:/Users/raresense/isaaclab_work/stretch_asset/stretch_lab.usda"), activate_contact_sensors=False),
    init_state=ArticulationCfg.InitialStateCfg(pos=(1.649, 1.675, .002), rot=(0., 0., 1., 0.), joint_pos={"joint_lift": 1.05, "joint_gripper_slide": .04}),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=list(GAINS), stiffness={k: v[0] for k, v in GAINS.items()}, damping={k: v[1] for k, v in GAINS.items()},
                                          joint_effort_limit=EFFORT, armature=ARM, friction=FRIC, dynamic_friction=FRIC),
               # as Sim: every joint outside the gains table has no drive and a 5 N limit, pad gimbals kp 1, kd .01. Left alone, the USD drives hold the mimic fingers, and the slide with them, at 0.
               "free": ImplicitActuatorCfg(joint_names_expr=["joint_gripper_finger_.*", "joint_arm_l[012]", "joint_head_nav_cam"], stiffness=0., damping=0., effort_limit_sim=5., armature=0., friction=0., dynamic_friction=0., viscous_friction=0.),
               "gimbal": ImplicitActuatorCfg(joint_names_expr=["rubber_.*"], stiffness=1., damping=.01, effort_limit_sim=5., armature=0., friction=0., dynamic_friction=0., viscous_friction=0.)},
)
OVEN = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Oven",
    spawn=sim_utils.UsdFileCfg(usd_path=f"{A}/oven.usda", activate_contact_sensors=False),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 0.)),
    actuators={"door": ImplicitActuatorCfg(joint_names_expr=["door_j"], stiffness=1.0, damping=.3, friction=.8, dynamic_friction=.8, effort_limit_sim=1000.),
               "knobs": ImplicitActuatorCfg(joint_names_expr=["knob_.*"], stiffness=0., damping=.01, friction=.06, dynamic_friction=.06, effort_limit_sim=1000.)},
)
SOLID = dict(collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.5, dynamic_friction=.5), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.55, .32, .12)))


KIN = dict(rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.5, dynamic_friction=.5))


def kbox(name, size, rgb, pos=(0., 0., -6.)):
    """A box that can be placed per copy before the first step and then stays put (kinematic)."""
    return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/" + name, spawn=sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb), **KIN), init_state=RigidObjectCfg.InitialStateCfg(pos=pos))


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(.9, .9, .9)))
    ctop: RigidObjectCfg = kbox("CounterTop", (2.75, .55, .04), (.55, .32, .12), (1.525, 2.325, .84))
    cbody: RigidObjectCfg = kbox("CounterBody", (2.41, .48, .82), (.42, .38, .34), (1.355, 2.36, .41))
    w0: RigidObjectCfg = kbox("Wall0", (8.0, .1, 1.1), (.86, .86, .82))
    w1: RigidObjectCfg = kbox("Wall1", (8.0, .1, 1.1), (.86, .86, .82))
    w2: RigidObjectCfg = kbox("Wall2", (8.0, .1, 1.1), (.80, .82, .86))
    w3: RigidObjectCfg = kbox("Wall3", (8.0, .1, 1.1), (.80, .82, .86))
    ob0: RigidObjectCfg = kbox("Table", (1.2, .8, .75), (.45, .30, .20))
    ob1: RigidObjectCfg = kbox("Island", (1.0, .6, .9), (.78, .78, .74))
    ob2: RigidObjectCfg = kbox("ChairA", (.45, .45, .9), (.20, .35, .60))
    ob3: RigidObjectCfg = kbox("ChairB", (.45, .45, .9), (.60, .22, .20))
    ob4: RigidObjectCfg = kbox("Cabinet", (.5, 1.0, 1.8), (.33, .33, .36))
    ob5: RigidObjectCfg = kbox("Crate", (.5, .5, .5), (.20, .50, .30))
    plates = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Plates", spawn=sim_utils.UsdFileCfg(usd_path=f"{A}/plate.usda"))
    plate: RigidObjectCfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Plates/pz_plate", spawn=None, init_state=RigidObjectCfg.InitialStateCfg(pos=(1.67, 2.12, .86)))
    pizza: RigidObjectCfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Plates/pz_pizza", spawn=None, init_state=RigidObjectCfg.InitialStateCfg(pos=(1.67, 2.12, .876)))
    oven: ArticulationCfg = OVEN
    robot: ArticulationCfg = ROBOT
    cam: CameraCfg = None if not args.cam else CameraCfg(prim_path="{ENV_REGEX_NS}/LiveCam", update_period=0, height=args.cam_h, width=args.cam_w, data_types=["rgb"], renderer_cfg=IsaacRtxRendererCfg(),
                                                         spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(.05, 200.)))


def main():
    rec = np.load("E:/stretch_assets/data/pz_final06_rec.npz", allow_pickle=True)
    rn = [str(x) for x in rec["names"]]
    cfg = sim_utils.SimulationCfg(device=args.device or "cuda:0", dt=DT, render_interval=600, physics=PhysxCfg(solver_type=1, **({"gpu_max_rigid_patch_count": args.patches} if args.patches else {}), **({"gpu_max_rigid_contact_count": args.contacts} if args.contacts else {})))
    with launch_simulation(cfg, args):
        from pxr import PhysxSchema, UsdPhysics
        sim = SimulationContext(cfg)
        scene = instantiate(SceneCfg(num_envs=args.num_envs, env_spacing=args.spacing, replicate_physics=True))
        st = sim.stage
        for p in st.Traverse():
            if p.IsA(UsdPhysics.Scene):
                PhysxSchema.PhysxSceneAPI.Apply(p).CreateDisableSleepingAttr(True)
                print("sleeping off on", p.GetPath(), flush=True)
            if args.gear and p.GetName().startswith("joint_gripper_finger_"):
                at = p.GetAttribute("physxMimicJoint:rotY:gearing")
                print("gearing", p.GetName(), at.Get(), "->", args.gear, flush=True)
                at.Set(args.gear)
            if not args.keep_newton and p.HasAPI("NewtonMimicAPI"):      # the joints carry two mimics: PhysX 10 rad/m and Newton 10, read as 10 deg/m. Both only hold at slide 0, so the gripper froze
                print("newton mimic off", p.GetName(), p.GetAttribute("newton:mimicEnabled").Get(), p.GetAttribute("newton:mimicCoef1").Get(), flush=True)
                p.GetAttribute("newton:mimicEnabled").Set(False)
                p.RemoveAPI("NewtonMimicAPI")
            if args.mimic_nf and p.GetName().startswith("joint_gripper_finger_"):
                p.GetAttribute("physxMimicJoint:rotY:naturalFrequency").Set(args.mimic_nf)
                p.GetAttribute("physxMimicJoint:rotY:dampingRatio").Set(args.mimic_dr)
                print("mimic soft", p.GetName(), args.mimic_nf, args.mimic_dr, flush=True)
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                a = PhysxSchema.PhysxArticulationAPI.Apply(p)
                a.CreateSolverPositionIterationCountAttr(args.iters)
                a.CreateSolverVelocityIterationCountAttr(1)
        sim.reset()
        scene.reset()
        rb, ov, pl, pi = scene["robot"], scene["oven"], scene["plate"], scene["pizza"]
        names = list(rb.joint_names)
        N, dev, org = scene.num_envs, sim.device, scene.env_origins
        JL, JA, JS, JLW, JRW = (names.index(n) for n in ("joint_lift", "joint_arm_l3", "joint_gripper_slide", "joint_left_wheel", "joint_right_wheel"))
        GB = rb.body_names.index("link_grasp_center")
        col = [names.index(n) for n in rn]
        q = rb.data.default_joint_pos.torch.clone()
        q[:, col] = torch.tensor(rec["q"][0], device=dev, dtype=torch.float32)       # seed's first row: fingers consistent with the open slide
        v = torch.zeros_like(q)
        scene.update(DT)
        CZ, RZ, NFX, NFY, NPX, NPY = .86, .135, 1.25, 2.10, 1.67, 2.12              # counter top, rack height, and the seed scene's oven and plate
        gen = torch.Generator(device="cpu").manual_seed(args.seed)

        def strat(lo, hi):                    # one value per equal slice of the range, shuffled
            u = (torch.randperm(N, generator=gen).float() + torch.rand(N, generator=gen)) / N
            return (lo + (hi - lo) * u).to(dev)

        # ---- what stands on the counter, in the counter's own frame (x along it, y into it): oven anywhere along it, plate on either side at any distance
        fx = strat(.75, 2.30)
        gap = strat(.14, .60)                                                        # plate centre to the oven's side wall; .14 = open fingers clear of the wall
        side = torch.where(strat(0., 1.) < .5, -1., 1.)
        px = fx + side * (.25 + gap)
        px = torch.where((px < .30) | (px > 2.75), fx - side * (.25 + gap), px)     # off the counter: the other side
        fy = NFY + strat(-.02, .06)
        ry = torch.maximum(strat(1.38, 1.48), fy - .72)                             # the line the robot works from. Reach check: the hand must still reach the back of the rack
        V = dict(plate_x=px, plate_y=strat(2.11, 2.28), goal_dx=strat(-.35, .35), robot_y=ry, oven_dx=fx - NFX, oven_dy=fy - NFY)
        # ---- the room around it: every copy its own room. Counter frame placed anywhere at any angle, furniture on the floor, robot starting anywhere, route planned on a floor grid
        rng = np.random.default_rng(args.seed)
        OBS_FOOT = [(1.2, .8), (1.0, .6), (.45, .45), (.45, .45), (.5, 1.0), (.5, .5)]   # table, island, chair, chair, cabinet, crate: as OBS in the scene
        OBS_ZC = [.375, .45, .45, .45, .9, .25]
        KW, INF, RES = 10, .55, .05                                                  # waypoints kept, clearance from the base centre (arm tucked sweeps .52), grid cell
        WORK = (-.40, 3.45, .78, 2.66)                                               # counter plus the strip the robot works in, counter frame: x0 x1 y0 y1
        R = dict(L=np.zeros((N, 2)), c=np.zeros((N, 2)), th=np.zeros(N), start=np.zeros((N, 3)), wp=np.zeros((N, KW, 2)), nwp=np.zeros(N, int), obs=np.zeros((N, 6, 4)), path_m=np.zeros(N), tries=np.zeros(N, int), wall=np.zeros(N, int))

        def astar(occ, s, g):
            import heapq
            H, W = occ.shape
            if occ[s] or occ[g]:
                return None
            dist = np.full(occ.shape, np.inf)
            dist[s] = 0.
            prev = {}
            pq = [(0., s)]
            nb = [(-1, 0, 1.), (1, 0, 1.), (0, -1, 1.), (0, 1, 1.), (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
            while pq:
                f, u = heapq.heappop(pq)
                if u == g:
                    out = [u]
                    while u in prev:
                        u = prev[u]
                        out.append(u)
                    return out[::-1]
                if f - math.hypot(u[0] - g[0], u[1] - g[1]) > dist[u] + 1e-9:
                    continue
                for di, dj, c in nb:
                    w = (u[0] + di, u[1] + dj)
                    if 0 <= w[0] < H and 0 <= w[1] < W and not occ[w] and dist[u] + c < dist[w]:
                        dist[w] = dist[u] + c
                        prev[w] = u
                        heapq.heappush(pq, (dist[w] + math.hypot(w[0] - g[0], w[1] - g[1]), w))
            return None

        def make_room(e):
            px_, ry_, gdx = float(V["plate_x"][e]), float(V["robot_y"][e]), float(V["goal_dx"][e])
            for attempt in range(400):
                Lx, Ly = rng.uniform(5.0, 7.2, 2)
                on_wall = rng.random() < .5
                th = float(rng.integers(4)) * math.pi / 2 if on_wall else rng.uniform(0, 2 * math.pi)
                c, s = math.cos(th), math.sin(th)
                cxy = np.array([rng.uniform(-Lx / 2, Lx / 2), rng.uniform(-Ly / 2, Ly / 2)])
                cor = np.array([[WORK[0], WORK[2]], [WORK[1], WORK[2]], [WORK[1], WORK[3]], [WORK[0], WORK[3]]])
                l2w = lambda p_: cxy + np.c_[c * p_[:, 0] - s * p_[:, 1], s * p_[:, 0] + c * p_[:, 1]]
                w2l = lambda p_: np.c_[c * (p_[:, 0] - cxy[0]) + s * (p_[:, 1] - cxy[1]), -s * (p_[:, 0] - cxy[0]) + c * (p_[:, 1] - cxy[1])]
                cw = l2w(cor)
                if (np.abs(cw[:, 0]) > Lx / 2 - .06).any() or (np.abs(cw[:, 1]) > Ly / 2 - .06).any():
                    continue
                if on_wall:                                                          # slide the counter back against the wall behind it
                    d = np.array([-s, c])
                    ts = []
                    for p_ in cw:
                        for k_, half in ((0, Lx / 2 - .06), (1, Ly / 2 - .06)):
                            if abs(d[k_]) > 1e-6:
                                ts.append(((half if d[k_] > 0 else -half) - p_[k_]) / d[k_])
                    cxy = cxy + d * max(0., min(ts))
                gx, gy = np.arange(-Lx / 2 + RES / 2, Lx / 2, RES), np.arange(-Ly / 2 + RES / 2, Ly / 2, RES)
                X, Y = np.meshgrid(gx, gy, indexing="ij")
                P = np.c_[X.ravel(), Y.ravel()]
                Pl = w2l(P)
                occ = (np.abs(P[:, 0]) > Lx / 2 - INF) | (np.abs(P[:, 1]) > Ly / 2 - INF)
                occ |= (Pl[:, 0] > .15 - INF) & (Pl[:, 0] < 2.90 + INF) & (Pl[:, 1] > 2.05 - INF) & (Pl[:, 1] < 2.60 + INF)
                obs = np.zeros((6, 4))
                placed = []
                for k_ in rng.permutation(6)[:int(rng.integers(3, 7))]:
                    w_, h_ = OBS_FOOT[k_]
                    r_ = math.hypot(w_, h_) / 2
                    for _ in range(40):
                        o = np.array([rng.uniform(-Lx / 2 + r_ + .1, Lx / 2 - r_ - .1), rng.uniform(-Ly / 2 + r_ + .1, Ly / 2 - r_ - .1)])
                        ol = w2l(o[None])[0]
                        if WORK[0] - r_ - .1 < ol[0] < WORK[1] + r_ + .1 and WORK[2] - r_ - .1 < ol[1] < WORK[3] + r_ + .1:
                            continue
                        if any(np.hypot(*(o - q_[:2])) < r_ + q_[2] + .15 for q_ in placed):
                            continue
                        a = rng.uniform(0, math.pi)
                        obs[k_] = (o[0], o[1], a, 1.)
                        placed.append((o[0], o[1], r_))
                        ca, sa = math.cos(a), math.sin(a)
                        u_ = ca * (P[:, 0] - o[0]) + sa * (P[:, 1] - o[1])
                        v_ = -sa * (P[:, 0] - o[0]) + ca * (P[:, 1] - o[1])
                        occ |= (np.abs(u_) < w_ / 2 + INF) & (np.abs(v_) < h_ / 2 + INF)
                        break
                occ = occ.reshape(X.shape)
                goal_l = np.array([min(max(px_ + gdx, -.05), 3.15), ry_])
                goal_w = l2w(goal_l[None])[0]
                cell = lambda p_: (int(np.clip((p_[0] + Lx / 2) / RES, 0, X.shape[0] - 1)), int(np.clip((p_[1] + Ly / 2) / RES, 0, X.shape[1] - 1)))
                free = np.argwhere(~occ)
                if len(free) < 50 or occ[cell(goal_w)]:
                    continue
                st = tuple(free[rng.integers(len(free))])
                sw = np.array([gx[st[0]], gy[st[1]]])
                if np.hypot(*(sw - goal_w)) < 1.2:
                    continue
                path = astar(occ, st, cell(goal_w))
                if path is None:
                    continue
                pts = np.array([[gx[i], gy[j]] for i, j in path])
                keep, i0 = [0], 0                                                    # shortcut: keep a point only where the straight line to the next one would cross something
                while i0 < len(pts) - 1:
                    j0 = len(pts) - 1
                    while j0 > i0 + 1:
                        seg = pts[i0] + (pts[j0] - pts[i0]) * np.linspace(0, 1, int(np.hypot(*(pts[j0] - pts[i0])) / (RES / 2)) + 2)[:, None]
                        if not any(occ[cell(p_)] for p_ in seg):
                            break
                        j0 -= 1
                    keep.append(j0)
                    i0 = j0
                wps = np.vstack([pts[keep[1:-1]].reshape(-1, 2), goal_w[None]])
                if len(wps) > KW:
                    continue
                R["L"][e], R["c"][e], R["th"][e], R["obs"][e], R["tries"][e], R["wall"][e] = (Lx, Ly), cxy, th, obs, attempt + 1, int(on_wall)
                R["start"][e] = (sw[0], sw[1], rng.uniform(-math.pi, math.pi))
                R["wp"][e, :len(wps)] = w2l(wps)                                     # stored in the counter frame, as everything the controller reads
                R["nwp"][e] = len(wps)
                R["path_m"][e] = float(np.hypot(*np.diff(np.vstack([sw[None], wps]), axis=0).T).sum())
                return True
            return False

        t_gen = time.time()
        okroom = [make_room(e) for e in range(N)]
        print("rooms generated", sum(okroom), "of", N, "in", round(time.time() - t_gen, 1), "s; tries max", int(R["tries"].max()), "path m min/mean/max", round(float(R["path_m"].min()), 2), round(float(R["path_m"].mean()), 2), round(float(R["path_m"].max()), 2), flush=True)
        if not all(okroom):
            raise RuntimeError("no room found for copies " + str([e for e, k_ in enumerate(okroom) if not k_][:10]))
        T = lambda a: torch.tensor(a, device=dev, dtype=torch.float32)
        CXY, TH, WP, NWP = T(R["c"]), T(R["th"]), T(R["wp"]), torch.tensor(R["nwp"], device=dev)
        cth, sth = torch.cos(TH), torch.sin(TH)

        def l2w(xl, yl):                      # counter frame -> this copy's room
            return CXY[:, 0] + cth * xl - sth * yl, CXY[:, 1] + sth * xl + cth * yl

        def loc(p):                           # room -> counter frame; p is (N, 3) without the env origin
            dx, dy = p[:, 0] - CXY[:, 0], p[:, 1] - CXY[:, 1]
            return torch.stack([cth * dx + sth * dy, -sth * dx + cth * dy, p[:, 2]], dim=1)

        def put(obj, x, y, z, a):             # one state write before the first step
            p = torch.zeros(N, 7, device=dev)
            p[:, 0], p[:, 1], p[:, 2] = org[:, 0] + x, org[:, 1] + y, org[:, 2] + z
            p[:, 5], p[:, 6] = torch.sin(a / 2), torch.cos(a / 2)
            obj.write_root_pose_to_sim_index(root_pose=p)

        ST = T(R["start"])
        put(rb, ST[:, 0], ST[:, 1], torch.full((N,), .002, device=dev), ST[:, 2])
        rb.write_joint_position_to_sim_index(position=q)
        rb.write_joint_velocity_to_sim_index(velocity=v)
        for o_ in (pl, pi):
            x_, y_ = l2w(V["plate_x"], V["plate_y"])
            put(o_, x_, y_, o_.data.root_pos_w.torch[:, 2] - org[:, 2], TH)
        x_, y_ = l2w(fx, fy)
        put(ov, x_, y_, torch.full((N,), CZ, device=dev), TH)
        oq = ov.data.default_joint_pos.torch.clone()
        oq[:, 0] = math.radians(88.0)
        ov.write_joint_position_to_sim_index(position=oq)
        ov.set_joint_position_target_index(target=oq)
        for o_, (lx_, ly_, z_) in ((scene["ctop"], (1.525, 2.325, .84)), (scene["cbody"], (1.355, 2.36, .41))):
            x_, y_ = l2w(torch.full((N,), lx_, device=dev), torch.full((N,), ly_, device=dev))
            put(o_, x_, y_, torch.full((N,), z_, device=dev), TH)
        Lx_, Ly_ = T(R["L"][:, 0]), T(R["L"][:, 1])
        zN, hN = torch.zeros(N, device=dev), torch.full((N,), .55, device=dev)
        put(scene["w0"], zN, Ly_ / 2 + .05, hN, zN)
        put(scene["w1"], zN, -Ly_ / 2 - .05, hN, zN)
        put(scene["w2"], Lx_ / 2 + .05, zN, hN, zN + math.pi / 2)
        put(scene["w3"], -Lx_ / 2 - .05, zN, hN, zN + math.pi / 2)
        OB = T(R["obs"])
        for k_ in range(6):
            act = OB[:, k_, 3] > .5
            put(scene[f"ob{k_}"], OB[:, k_, 0], OB[:, k_, 1], torch.where(act, torch.full((N,), OBS_ZC[k_], device=dev), torch.full((N,), -6., device=dev)), OB[:, k_, 2])
        FXE, FYE = fx, fy
        MX, MY, PXT, PYT = FXE, FYE, V["plate_x"] + .0025, V["plate_y"].clone()       # c1000: all 22 lift failures had the hand at or behind the plate centre in x, none 0 to 6 mm ahead
        CARRY = 1.056
        CIN = min(CZ + RZ + .061, CZ + .200)                                        # under the top heating element (poc2_s06)
        OUTZ = min(CZ + RZ + .063, CZ + .202)
        rb.set_joint_position_target_index(target=q)
        rb.set_joint_velocity_target_index(target=v)
        for _ in range(300):                                                        # settle .5 s
            scene.write_data_to_sim()
            sim.step(render=False)
        scene.update(300 * DT)
        CT = 12
        dtc = CT * DT
        K = .12
        ph = torch.zeros(N, dtype=torch.long, device=dev)
        tp = torch.zeros(N, device=dev)
        g0, tg, dur = torch.zeros(N, 3, device=dev), torch.zeros(N, 3, device=dev), torch.ones(N, device=dev)
        qL, qA, qS = q[:, JL].clone(), q[:, JA].clone(), q[:, JS].clone()
        hold = torch.zeros(N, 2, device=dev)
        touch = torch.zeros(N, dtype=torch.bool, device=dev)
        t_touch, sfin, s0, zp0 = (torch.zeros(N, device=dev) for _ in range(4))
        failc = torch.zeros(N, dtype=torch.long, device=dev)
        t_done, t_nav = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
        wpi = torch.zeros(N, dtype=torch.long, device=dev)
        PH = ["over_plate", "descend", "close", "lift", "carry", "retract", "to_oven", "height", "into", "lower", "open", "out1", "out2", "done", "square", "drive_to_plate", "nav"]
        DONE, SQUARE, DRIVE, NAV, FAIL = 13, 14, 15, 16, 99
        FAILS = {1: "plate did not rise with the hand", 2: "gripper never touched the plate", 3: "not finite", 4: "did not reach the counter"}
        # phase: seconds (None = from the distance), settle seconds, tolerance m, base drives
        MOVE = {0: (6., 4., .0015, True), 1: (8., 1., .003, False), 3: (6., .5, .003, False), 4: (6., 1., .003, False), 5: (4., 1., .004, False), 6: (None, 3., .003, True),
                7: (2., 1., .003, False), 8: (15., 1., .003, False), 9: (6., .5, .003, False), 11: (6., .5, .004, False), 12: (6., .5, .004, False), 15: (None, 3., .01, True)}
        BASE_PH = torch.tensor([0, 6, 14, 15, 16], device=dev)
        SIMT = [0.5]
        wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi

        def read():                           # everything in the counter frame, so the skill is the same in every room
            gp = loc(rb.data.body_pos_w.torch[:, GB] - org)
            rpos = loc(rb.data.root_pos_w.torch - org)
            rq = rb.data.root_quat_w.torch
            yw = wrap(torch.atan2(2 * (rq[:, 3] * rq[:, 2] + rq[:, 0] * rq[:, 1]), 1 - 2 * (rq[:, 1] ** 2 + rq[:, 2] ** 2)) - TH)
            return gp, rpos, yw, loc(pl.data.root_pos_w.torch - org), rb.data.joint_pos.torch

        def enter(k, i, gp, rpos, pp, jq):
            """Copies i start phase k. Targets are computed here, from where things are in each copy."""
            if i.numel() == 0:
                return
            hb = i[torch.isin(ph[i], BASE_PH) | (tp[i] < 0)]                           # wheel hold angle: taken only after the base has driven (c1000: taking it at every phase let the 3 mm GPU creep add up)
            hold[hb, 0], hold[hb, 1] = jq[hb, JLW], jq[hb, JRW]
            ph[i], tp[i], g0[i] = k, 0., gp[i]
            if k in MOVE:
                t = gp[i].clone()
                if k == 0:
                    t[:, 0], t[:, 1], t[:, 2] = PXT[i], PYT[i], .97
                elif k == 1:
                    t[:, 0], t[:, 1], t[:, 2] = PXT[i], PYT[i], CZ + .035
                elif k == 3:
                    t[:, 2] += .036
                    zp0[i] = pp[i, 2]
                elif k == 4:
                    t[:, 2] = CARRY
                elif k == 5:
                    t[:, 1] = torch.maximum(MY[i] - .28, rpos[i, 1] + .42)
                elif k == 6:
                    t[:, 0] = MX[i] - .06
                elif k == 7:
                    t[:, 2] = CIN
                elif k == 8:
                    t[:, 1] = MY[i] + .18
                elif k == 9:
                    t[:, 2] = CZ + RZ + .045
                elif k == 11:
                    t[:, 1], t[:, 2] = MY[i], OUTZ
                elif k == 12:
                    t[:, 1], t[:, 2] = torch.maximum(MY[i] - .23, rpos[i, 1] + .42), OUTZ
                elif k == 15:
                    t[:, 0], t[:, 2] = PXT[i], CARRY                                # arm stays in, above the open door, while the base drives to the plate
                tg[i] = t
                d = MOVE[k][0]
                dur[i] = d if d else torch.clamp((t[:, 0] - gp[i, 0]).abs() / .07, min=4.)
            if k == 2:
                touch[i] = False
                qS[i] = jq[i, JS]
            if k == 10:
                s0[i] = qS[i]
                sfin[i] = jq[i, JS]
            if k == SQUARE:
                t_nav[i] = SIMT[0]
            if k == DONE:
                t_done[i] = SIMT[0]

        def tick():
            gp, rpos, yw, pp, jq = read()
            tp.add_(dtc)
            bad = ~(torch.isfinite(gp).all(dim=1) & torch.isfinite(jq).all(dim=1))
            if bad.any():
                ib = torch.nonzero(bad & (ph != FAIL)).flatten()
                failc[ib], ph[ib] = 3, FAIL
            vl = torch.clamp(-15. * (jq[:, JLW] - hold[:, 0]), -2., 2.)             # wheel hold: grip force turned a free base 19 deg (pz_gpu18)
            vr = torch.clamp(-15. * (jq[:, JRW] - hold[:, 1]), -2., 2.)
            i = torch.nonzero(ph == NAV).flatten()
            if i.numel():                                                           # drive the planned route: turn to face the next point, drive to it. Arm tucked, hand at carry height
                d = WP[i, torch.clamp(wpi[i], max=KW - 1)] - rpos[i, :2]
                dist = d.norm(dim=1)
                eh = wrap(torch.atan2(d[:, 1], d[:, 0]) - yw[i])
                turn = eh.abs() > math.radians(12.)
                vv = torch.where(turn, torch.zeros_like(dist), torch.clamp(1.2 * dist, max=.20) * torch.cos(eh))
                ww = torch.clamp(torch.where(turn, 1.8 * eh, 2.5 * eh), -.5, .5)
                ww = torch.where(turn & (ww.abs() < .08), torch.sign(eh) * .08, ww)
                vl[i], vr[i] = (vv - ww * .1704) / .05, (vv + ww * .1704) / .05
                qL[i] += K * (CARRY - gp[i, 2])
                wpi[i[dist < .04]] += 1
                late = i[tp[i] > 150.]
                failc[late], ph[late] = 4, FAIL
                enter(SQUARE, i[(wpi[i] >= NWP[i]) & (tp[i] <= 150.)], gp, rpos, pp, jq)
            for k, (d, settle, tol, base) in MOVE.items():
                i = torch.nonzero(ph == k).flatten()
                if i.numel() == 0:
                    continue
                u = torch.clamp(tp[i] / dur[i], max=1.)
                u = u * u * (3 - 2 * u)
                e = g0[i] + (tg[i] - g0[i]) * u[:, None] - gp[i]
                qL[i] += K * e[:, 2]
                qA[i] = torch.clamp(qA[i] + K * e[:, 1] / 4, 0., .13)
                if base:
                    vv = torch.clamp(-2.0 * e[:, 0], -.1, .1)
                    ww = torch.clamp(3.0 * wrap(math.pi - yw[i]), -.4, .4)
                    vl[i], vr[i] = (vv - ww * .1704) / .05, (vv + ww * .1704) / .05
                fin = (tp[i] >= dur[i]) & ((e.abs().max(dim=1).values < tol) | (tp[i] >= dur[i] + settle))
                nxt = i[fin]
                if k == 15:
                    enter(0, nxt, gp, rpos, pp, jq)
                    continue
                if k == 3 and nxt.numel():
                    low = pp[nxt, 2] < zp0[nxt] + .02
                    lo_ = nxt[low]
                    failc[lo_], ph[lo_] = 1, FAIL
                    nxt = nxt[~low]
                enter(DONE if k == 12 else k + 1, nxt, gp, rpos, pp, jq)
            i = torch.nonzero(ph == SQUARE).flatten()
            if i.numel():                                                           # turn on the spot to face along the counter
                ye = wrap(math.pi - yw[i])
                ww = torch.clamp(1.8 * ye, -.35, .35)
                ww = torch.where(ww.abs() < .05, torch.sign(ye) * .05, ww)
                vl[i], vr[i] = -ww * .1704 / .05, ww * .1704 / .05
                enter(DRIVE, i[(ye.abs() < math.radians(.6)) | (tp[i] > 40.)], gp, rpos, pp, jq)
            i = torch.nonzero(ph == 2).flatten()
            if i.numel():
                nt = ~touch[i]
                qS[i[nt]] -= .005 * dtc
                new = i[nt & ((jq[i, JS] - qS[i]) > .002) & (tp[i] > .1)]
                touch[new], t_touch[new], s0[new] = True, tp[new], qS[new]
                sfin[new] = jq[new, JS] - 14.0 * 3.56 / 4000.
                it = i[touch[i]]
                uu = torch.clamp((tp[it] - t_touch[it]) / 1.0, max=1.)
                qS[it] = s0[it] + (sfin[it] - s0[it]) * uu * uu * (3 - 2 * uu)
                enter(3, it[(tp[it] - t_touch[it]) > 1.3], gp, rpos, pp, jq)
                to = i[(~touch[i]) & (tp[i] > 10.)]
                failc[to], ph[to] = 2, FAIL
            i = torch.nonzero(ph == 10).flatten()
            if i.numel():
                a = torch.clamp(tp[i] / 2.0, max=1.)
                b = torch.clamp((tp[i] - 2.0) / 2.0, min=0., max=1.)
                qS[i] = s0[i] + (sfin[i] - s0[i]) * a + (.04 - sfin[i]) * b
                enter(11, i[tp[i] > 4.6], gp, rpos, pp, jq)
            q[:, JL], q[:, JA], q[:, JS] = qL, qA, qS
            v[:, JLW], v[:, JRW] = vl, vr
            rb.set_joint_position_target_index(target=q)
            rb.set_joint_velocity_target_index(target=v)

        gp, rpos, yw, pp, jq = read()
        print("start copy 0 (counter frame): grasp", [round(float(a), 3) for a in gp[0]], "base", [round(float(a), 3) for a in rpos[0]], "yaw deg", round(float(torch.rad2deg(yw[0])), 1), "plate", [round(float(a), 3) for a in pp[0]],
              "oven", [round(float(a), 3) for a in loc(ov.data.root_pos_w.torch - org)[0]], "door deg", round(float(torch.rad2deg(ov.data.joint_pos.torch[0, 0])), 1),
              "room", [round(float(a), 2) for a in R["L"][0]], "counter angle deg", round(math.degrees(R["th"][0]), 1), "route points", int(R["nwp"][0]), flush=True)
        tp.fill_(-1.)                                                               # marks the first entry: take the wheel angles
        enter(NAV, torch.arange(N, device=dev), gp, rpos, pp, jq)
        O0 = loc(ov.data.root_pos_w.torch - org).clone()
        OQ0 = ov.data.root_quat_w.torch.clone()
        LG = {k_: [] for k_ in ("t", "grasp", "plate", "base_room", "phase")}

        def shot(name, top=True):
            import PIL.Image as I
            cam = scene["cam"]
            if top:                                                                 # the whole room from above
                eye = org + torch.tensor([[0., -.05, 10.5]], device=dev)
                at = org.clone()
            else:                                                                   # close on each copy's oven, from the robot's side
                ex_, ey_ = l2w(fx + 1.25, torch.full((N,), .75, device=dev))
                ax_, ay_ = l2w(fx + .05, torch.full((N,), 2.15, device=dev))
                eye = org + torch.stack([ex_, ey_, torch.full((N,), 1.85, device=dev)], dim=1)
                at = org + torch.stack([ax_, ay_, torch.full((N,), .95, device=dev)], dim=1)
            cam.set_world_poses_from_view(eye, at)
            for _ in range(6):
                sim.render()
                cam.update(DT, force_recompute=True)
            for e in range(min(N, args.shots)):
                I.fromarray(cam.data.output["rgb"][e, ..., :3].cpu().numpy().astype(np.uint8)).save(f"{RUN}/{name}_env{e:02d}.png")

        if args.cam:
            shot("top_start")
        VID = bool(args.cam and args.vid_dt > 0)
        vid_t = []
        if VID:                                                                     # one camera per room, pointed at each view in turn: the whole room from above, then close on that room's oven
            import PIL.Image as PI_
            VD = f"{RUN}/vid"
            os.makedirs(VD, exist_ok=True)
            cam1 = scene["cam"]
            ex_, ey_ = l2w(fx + 1.55, torch.full((N,), .40, device=dev))
            ax_, ay_ = l2w(fx - .10, torch.full((N,), 2.10, device=dev))
            VEYE = (org + torch.tensor([[0., -.05, 10.5]], device=dev), org + torch.stack([ex_, ey_, torch.full((N,), 2.05, device=dev)], dim=1))
            VAT = (org.clone(), org + torch.stack([ax_, ay_, torch.full((N,), .90, device=dev)], dim=1))
        EPREC = {k_: [] for k_ in ("link", "q", "dq", "qd", "vd", "root", "plate", "pizza", "oven", "body", "t")}
        next_vid = 0.
        t0 = time.time()
        nblk = int(args.secs / dtc)
        last_print, next_shot = 0., 25.
        for b in range(nblk):
            tick()
            for _ in range(CT):
                scene.write_data_to_sim()
                sim.step(render=False)
            scene.update(dtc)
            SIMT[0] += dtc
            if b % 5 == 0:
                gp, rpos, yw, pp, jq = read()
                LG["t"].append(SIMT[0])
                LG["grasp"].append(gp.cpu().numpy().astype(np.float32))
                LG["plate"].append(pp.cpu().numpy().astype(np.float32))
                LG["base_room"].append((rb.data.root_pos_w.torch - org)[:, :2].cpu().numpy().astype(np.float32))
                LG["phase"].append(ph.cpu().numpy().astype(np.int8))
            if SIMT[0] - last_print >= 5.0:
                last_print = SIMT[0]
                torch.cuda.synchronize()
                w = time.time() - t0
                cnt = torch.bincount(torch.clamp(ph, max=17), minlength=18).tolist()
                gp, rpos, yw, pp, jq = read()
                print(json.dumps(dict(sim_t=round(SIMT[0], 1), wall=round(w, 1), per_env_sim_s_per_min=round((SIMT[0] - .5) / max(w, 1e-6) * 60, 2), phases={PH[k_]: c for k_, c in enumerate(cnt[:17]) if c},
                                      failed=int((ph == FAIL).sum()), copy0=dict(grasp=[round(float(a), 3) for a in gp[0]], plate=[round(float(a), 3) for a in pp[0]], base=[round(float(a), 3) for a in rpos[0, :2]],
                                                                                yaw=round(float(torch.rad2deg(yw[0])), 1), route_point=int(wpi[0])))), flush=True)
            if b % args.rec_every == 0:                                             # the full episode of every room, positions in the room's own frame
                EPREC["link"].append(torch.cat([rb.data.body_pos_w.torch - org[:, None, :], rb.data.body_quat_w.torch], dim=2).cpu().numpy().astype(np.float32))
                EPREC["q"].append(rb.data.joint_pos.torch.cpu().numpy().astype(np.float32))
                EPREC["dq"].append(rb.data.joint_vel.torch.cpu().numpy().astype(np.float32))
                EPREC["qd"].append(q.cpu().numpy().astype(np.float32))
                EPREC["vd"].append(v.cpu().numpy().astype(np.float32))
                EPREC["root"].append(torch.cat([rb.data.root_pos_w.torch - org, rb.data.root_quat_w.torch], dim=1).cpu().numpy().astype(np.float32))
                EPREC["plate"].append(torch.cat([pl.data.root_pos_w.torch - org, pl.data.root_quat_w.torch], dim=1).cpu().numpy().astype(np.float32))
                EPREC["pizza"].append(torch.cat([pi.data.root_pos_w.torch - org, pi.data.root_quat_w.torch], dim=1).cpu().numpy().astype(np.float32))
                EPREC["oven"].append(ov.data.joint_pos.torch.cpu().numpy().astype(np.float32))
                EPREC["body"].append(torch.cat([ov.data.root_pos_w.torch - org, ov.data.root_quat_w.torch], dim=1).cpu().numpy().astype(np.float32))
                EPREC["t"].append(SIMT[0])
            if VID and SIMT[0] >= next_vid:
                next_vid += args.vid_dt
                fr = []
                for j_ in (0, 1):
                    cam1.set_world_poses_from_view(VEYE[j_], VAT[j_])
                    for _ in range(args.vid_r):
                        sim.render()
                        cam1.update(DT, force_recompute=True)
                    fr.append(cam1.data.output["rgb"][..., :3].clone())
                fr = torch.cat(fr, dim=2).to(torch.uint8).cpu().numpy()
                for e in range(N):
                    PI_.fromarray(fr[e]).save(f"{VD}/e{e:02d}_{len(vid_t):04d}.jpg", quality=88)
                vid_t.append(SIMT[0])
            if args.cam and not VID and SIMT[0] >= next_shot:
                next_shot += 25.
                shot(f"top_t{int(SIMT[0]):04d}")
            if bool(((ph == DONE) | (ph == FAIL)).all()):
                break
        for _ in range(600):                                                        # 1 s to come to rest
            scene.write_data_to_sim()
            sim.step(render=False)
        scene.update(600 * DT)
        torch.cuda.synchronize()
        wall = time.time() - t0
        if args.cam:
            shot("top_end")
            shot("oven_end", top=False)
        # ---- the same checks as the CPU proof of concept, in each copy's counter frame
        gp, rpos, yw, pp, jq = read()
        pq, zz = pl.data.root_quat_w.torch, loc(pi.data.root_pos_w.torch - org)
        ob = loc(ov.data.root_pos_w.torch - org)
        oq_ = ov.data.root_quat_w.torch
        ex, ey, ez = FXE - .06, FYE + .175, CZ + RZ + .004
        xy = torch.hypot(pp[:, 0] - ex, pp[:, 1] - ey) * 1000
        dz = (pp[:, 2] - ez) * 1000
        tilt = torch.rad2deg(torch.acos(torch.clamp(1 - 2 * (pq[:, 0] ** 2 + pq[:, 1] ** 2), -1., 1.)))
        poff = torch.hypot(zz[:, 0] - pp[:, 0], zz[:, 1] - pp[:, 1]) * 1000
        pab = (zz[:, 2] - pp[:, 2]) * 1000
        odr = torch.hypot(ob[:, 0] - O0[:, 0], ob[:, 1] - O0[:, 1]) * 1000
        oyaw = torch.rad2deg(wrap(2 * torch.atan2(oq_[:, 2], oq_[:, 3]) - 2 * torch.atan2(OQ0[:, 2], OQ0[:, 3]))).abs()
        hout = (FYE - gp[:, 1]) * 1000
        furn = []                                                                    # furniture and walls are kinematic: they cannot be pushed. Counted here only as how many each room had
        res = []
        for e in range(N):
            fails = []
            if int(ph[e]) == FAIL:
                fails.append(FAILS.get(int(failc[e]), "failed"))
            elif int(ph[e]) != DONE:
                fails.append("ran out of time in " + PH[int(ph[e])])
            vals = [float(x[e]) for x in (xy, dz, tilt, poff, pab, odr, oyaw, hout)]
            fin_ = all(math.isfinite(a) for a in vals)
            if not fin_:
                fails.append("not finite")
            else:
                if vals[0] > 20 or abs(vals[1]) > 8:
                    fails.append("plate not on the rack spot")
                if vals[2] > 5:
                    fails.append("plate tilted")
                if vals[3] > 20 or not (0 < vals[4] < 40):
                    fails.append("pizza not on the plate")
                if vals[5] > 5 or vals[6] > .5:
                    fails.append("oven moved")
                if vals[7] < 150:
                    fails.append("hand still in the oven")
            res.append(dict(env=e, ok=not fails, fails=fails, phase=PH[int(ph[e])] if int(ph[e]) != FAIL else "failed", t_reached_counter=round(float(t_nav[e]), 2), t_done=round(float(t_done[e]), 2),
                            room=dict(size_m=[round(float(a), 2) for a in R["L"][e]], counter_xy=[round(float(a), 2) for a in R["c"][e]], counter_angle_deg=round(math.degrees(R["th"][e]), 1), against_wall=bool(R["wall"][e]),
                                      furniture=int((R["obs"][e, :, 3] > .5).sum()), start_xy=[round(float(a), 2) for a in R["start"][e, :2]], start_heading_deg=round(math.degrees(R["start"][e, 2]), 1),
                                      route_points=int(R["nwp"][e]), route_m=round(float(R["path_m"][e]), 2)),
                            counter=dict(oven_x=round(float(fx[e]), 3), oven_y=round(float(fy[e]), 3), plate_x=round(float(V["plate_x"][e]), 3), plate_y=round(float(V["plate_y"][e]), 3), work_line_y=round(float(V["robot_y"][e]), 3)),
                            checks=dict(plate_xy_err_mm=round(vals[0], 1), plate_z_err_mm=round(vals[1], 1), plate_tilt_deg=round(vals[2], 2), pizza_off_plate_mm=round(vals[3], 1), pizza_above_plate_mm=round(vals[4], 1),
                                        oven_drift_mm=round(vals[5], 1), oven_yaw_deg=round(vals[6], 2), hand_out_mm=round(vals[7], 1)) if fin_ else None))
        npass = sum(r["ok"] for r in res)
        why = {}
        for r in res:
            for f in r["fails"]:
                why[f] = why.get(f, 0) + 1
        summ = dict(tag=args.tag, rooms=N, passed=npass, pass_rate=round(npass / N, 4), fail_reasons=why, sim_s=round(SIMT[0], 1), wall_s=round(wall, 1), wall_min=round(wall / 60, 2), seed=args.seed,
                    room_size_m=[round(float(R["L"].min()), 2), round(float(R["L"].max()), 2)], route_m=[round(float(R["path_m"].min()), 2), round(float(R["path_m"].max()), 2)], against_wall=int(R["wall"].sum()))
        json.dump(dict(summary=summ, results=res), open(f"{RUN}/results.json", "w"))
        np.savez_compressed(f"{RUN}/log.npz", **{k_: np.array(x) for k_, x in LG.items()}, room_L=R["L"], room_c=R["c"], room_th=R["th"], room_obs=R["obs"], room_start=R["start"], room_wp=R["wp"], room_nwp=R["nwp"])
        if VID:
            json.dump(dict(t=[round(a, 3) for a in vid_t], vid_dt=args.vid_dt, views=["top", "oven"]), open(f"{VD}/times.json", "w"))
        A_ = {k_: np.array(x) for k_, x in EPREC.items()}
        n_ = len(A_["t"])
        ED = f"E:/isaaclab_mimic_test/episodes/{args.tag}"
        os.makedirs(ED, exist_ok=True)
        for e in range(N):                                                           # one file per room: the episode, the room it happened in, and its label. Each stops at that room's own end
            te = res[e]["t_done"] if res[e]["t_done"] > 0 else A_["t"][-1]
            m_ = int(np.searchsorted(A_["t"], te + 1.0)) or n_
            np.savez_compressed(f"{ED}/e{e:04d}.npz", t=A_["t"][:m_], names=np.array(names), links=np.array(rb.body_names), oven_names=np.array(ov.joint_names), steps_per_sample=CT * args.rec_every,
                                hz=round(1 / (dtc * args.rec_every), 3), q=A_["q"][:m_, e], dq=A_["dq"][:m_, e], qd=A_["qd"][:m_, e], vd=A_["vd"][:m_, e], root=A_["root"][:m_, e], link=A_["link"][:m_, e],
                                oven=A_["oven"][:m_, e], plate=A_["plate"][:m_, e], pizza=A_["pizza"][:m_, e], body=A_["body"][:m_, e], ok=res[e]["ok"], fails=np.array(res[e]["fails"]),
                                params=json.dumps(dict(room=res[e]["room"], counter=res[e]["counter"], checks=res[e]["checks"], plate_mass=0.35)), frame="room: metres from the room centre",
                                room_size=R["L"][e], counter_xy=R["c"][e], counter_angle=R["th"][e], furniture=R["obs"][e], start=R["start"][e], route=R["wp"][e][:int(R["nwp"][e])])
        json.dump(dict(summary=summ, results=res), open(f"{ED}/index.json", "w"))
        print("EPISODES saved", N, "to", ED, "samples up to", n_, "hz", round(1 / (dtc * args.rec_every), 2), "video frames", len(vid_t), flush=True)
        print("ROOMS SUMMARY", json.dumps(summ), flush=True)
        print("SKILL DONE", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
