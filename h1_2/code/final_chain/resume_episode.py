r"""Resume the H1-2 tea episode from a full-run recording, just before a step, and run the remaining steps.

This RESTORES STATE. The result is a disclosed join, not one continuous physics run: <tag>.json and <tag>_rec.npz record the
source run, the resume time and the tag, and full_episode_passed is always False. All state writes happen before the first
step after the resume moment. After it: no welds, no forces, no state writes, only the same joint references the stages use.

Usage (a new tag each time; check the machine rules first):
    powershell -File D:\tea_robot\isaac_delivery\run_test.ps1 -Script D:\tea_robot\isaac_delivery\resume_episode.py -Tag <tag> -Extra1 <source run> -Extra2 <step or sim time>
    e.g. -Tag rs_stir1 -Extra1 final07 -Extra2 stir          from the STEP START stir in logs\final07.log
         -Tag rs_pw1   -Extra1 final07 -Extra2 276.3         from the last still sample at or before t 276.3
Env (defaults are the final-run values, as run_live_episode.py; REC_AUTOSAVE_S=30 P_SCOOP=real):
    MODE=gpu|cpu             cpu: no water, for joint and pose checks only
    RESUME_SETTLE_S=0.5      sim seconds under its own controllers before the first step starts (recorded velocities do not exist)
    RESUME_SIM_S=0           stop this many sim seconds after the resume moment (0 = run to the end of the episode)
    RESUME_HOLD=1            run no stage: stand under the controllers until RESUME_SIM_S, measures the restore alone
    RESUME_STEPS=a,b         steps to run, default the source order from the resume step on
    RESUME_CUP=1             also run cup_walk_away at the end (default 1)
    RESUME_OVERRIDE=step=path[,step=path]   run this file for a step instead of stage_live.py, without installing it (step: teabag kettle powder stir cup)
    RESUME_Q=auto|links      joint source: auto = q in logs\<source>.npz if present, links = recovered from link_T
    RESUME_PARTICLE_FPS=10   water recording rate; 9.6 MB per sample at 700 ml
    RESUME_MAX_PARTICLE_GB=0 stop recording water past this size (the run goes on), 0 = no cap
    RESUME_FORCE=1           accept a moment that is not still
Output: logs\<tag>.json, <tag>_gates.json, <tag>_rec.npz (+ .particles.f32), <tag>.npz (q at 30 Hz, so a resumed run can itself be resumed).
Then: python resume_compare.py <tag> for numbers against the source, live_still.py for frames.
"""
import hashlib
import json
import os
import re
import sys
import time
import traceback

ROOT = r"D:\tea_robot\isaac_delivery"
TAG = sys.argv[1]
SRC = sys.argv[2]
WHERE = sys.argv[3]
for _k, _v in dict(MODE="gpu", EP_ITERS="32", PAD="1", GRIP_ONLY="0", KEEP_GOING="1", FILL_ML="700", SQ="0.5", LOW="1.0", TSQ="0.5",
                   STEPS="teabag,kettle,powder,stir", LEVEL="0", REC_AUTOSAVE_S="30", P_SCOOP="real").items():
    os.environ.setdefault(_k, _v)

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
sys.path.insert(0, ROOT)
import h12_physx as H
import numpy as np

sys.dont_write_bytecode = True          # read the other agent's module, leave nothing in its folder
sys.path.insert(0, r"D:\tea_robot\isaac_native\balance_support")
sys.path.insert(0, ROOT + r"\props")
import gates as GT
import mug_place
import props as PR
import record as RC

MUG = "object_0316"
OUT = rf"{ROOT}\logs\{TAG}"
SRC_BASE = rf"{ROOT}\logs\{SRC}"
MODE = os.environ["MODE"]
PAD = float(os.environ["PAD"])
ALL_STEPS = [x for x in os.environ["STEPS"].split(",") if x]
if os.path.exists(ROOT + r"inal_steps.txt"):               # the same order file full_episode.py honours
    ALL_STEPS = [x.strip() for x in open(ROOT + r"inal_steps.txt").read().split(",") if x.strip()]
SETTLE = float(os.environ.get("RESUME_SETTLE_S", 0.5))
SIM_S = float(os.environ.get("RESUME_SIM_S", 0))
HOLD = os.environ.get("RESUME_HOLD") == "1"
CUP = os.environ.get("RESUME_CUP", "1") == "1"
FORCE = os.environ.get("RESUME_FORCE") == "1"
MAX_P_GB = float(os.environ.get("RESUME_MAX_PARTICLE_GB", 0))
P_FPS = float(os.environ.get("RESUME_PARTICLE_FPS", 10))
STILL_LINK, STILL_BODY = 0.05, 0.02     # m/s: above this the moment is not still and the zero velocities below would be a real jump
OVERRIDE = dict(x.split("=", 1) for x in os.environ.get("RESUME_OVERRIDE", "").split(",") if x)
t_wall = time.time()


class StopResume(BaseException):
    """The requested sim time is reached. BaseException, so the stages' own 'except Exception' clean-ups do not swallow it."""


# ---------------------------------------------------------------- the recording
def read_log(path):
    raw = open(path, "rb").read()
    return raw.decode("utf-16", "ignore") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "ignore")


log_text = read_log(SRC_BASE + ".log")
starts = {m.group(1): float(m.group(2)) for m in re.finditer(r"STEP START (\w+) t ([\d.]+)", log_text)}
m_cup = re.search(r"CUP start \{'t': ([\d.]+)", log_text)
if m_cup:
    starts["cup"] = float(m_cup.group(1))
S = np.load(SRC_BASE + "_rec.npz", allow_pickle=False)
T, LT, BT = S["t"], S["link_T"], S["body_T"]
link_paths, body_paths = [str(x) for x in S["link_paths"]], [str(x) for x in S["body_paths"]]
order = ALL_STEPS + ["cup"]
if WHERE in order:
    step0, t_req = WHERE, starts[WHERE]
else:
    t_req = float(WHERE)
    after = [k for k in order if k in starts and starts[k] >= t_req - 0.2]
    step0 = min(after, key=lambda k: starts[k]) if after else None
    step0 = os.environ.get("RESUME_STEP0") or step0        # mid-step restart: name the step the moment belongs to
    if step0 is None:
        raise SystemExit(f"t {t_req} is after every step start in {SRC}: {starts}")
def speeds(g):
    """Peak link and body speed over the 0.1 s up to g, from the recorded transforms."""
    fps = float(S["fps"])
    return (float(max(np.linalg.norm(LT[x, :, :3].astype(float) - LT[x - 1, :, :3], axis=1).max() for x in range(g - 2, g + 1)) * fps),
            float(max(np.linalg.norm(BT[x, :, :3].astype(float) - BT[x - 1, :, :3], axis=1).max() for x in range(g - 2, g + 1)) * fps))


# Snap to a water sample. These fall on multiples of 0.1 s, so the step count is a multiple of 60 and the 50 Hz control tick and gait clock line up.
# Log times are rounded to 0.1 s, so a frame at a step's printed start can already be inside that step: given a step name, go back up to 1 s to the nearest still sample.
has_water = "p_t" in S.files and len(S["p_t"]) > 0
if has_water:
    pt, pn = S["p_t"], S["p_n"].astype(np.int64)
    pi = int(np.searchsorted(pt, t_req + 1e-6, "right")) - 1
    frame_of = lambda i: int(np.argmin(np.abs(T - pt[i])))
else:
    pi = int(np.searchsorted(T, t_req + 1e-6, "right")) // 3
    frame_of = lambda i: 3 * i
f = frame_of(pi)
link_v, body_v = speeds(f)
tries = 10 if WHERE in order else 0
while tries and (link_v > STILL_LINK or body_v > STILL_BODY):
    tries -= 1
    pi -= 1
    f = frame_of(pi)
    link_v, body_v = speeds(f)
t_res = float(T[f])
if has_water:
    if abs(T[f] - pt[pi]) > 1e-6:
        raise SystemExit(f"no frame on water sample {pt[pi]}")
    P_rec = np.fromfile(SRC_BASE + "_rec.npz.particles.f32", np.float32, int(pn[pi]) * 3, offset=int(pn[:pi].sum()) * 12).reshape(-1, 3)
    P_prev = np.fromfile(SRC_BASE + "_rec.npz.particles.f32", np.float32, int(pn[pi - 1]) * 3, offset=int(pn[:pi - 1].sum()) * 12).reshape(-1, 3)
    water_v = np.linalg.norm(P_rec.astype(float) - P_prev, axis=1) / (pt[pi] - pt[pi - 1])
else:
    P_rec, water_v = None, np.zeros(1)
print(f"RESUME source {SRC} frame {f} t {t_res:.3f} (asked {t_req}), step {step0}; last 0.1 s max link speed {link_v:.4f} m/s, max body speed {body_v:.4f} m/s, "
      f"water speed mean {water_v.mean():.5f} p99 {np.percentile(water_v, 99):.4f} max {water_v.max():.4f} m/s", flush=True)
if (link_v > STILL_LINK or body_v > STILL_BODY) and not FORCE:
    raise SystemExit(f"not a still moment (limits {STILL_LINK}, {STILL_BODY} m/s): velocities are not stored, pick another time or set RESUME_FORCE=1")


def pose_matrix(Tw, scale):
    from pxr import Gf
    R = Gf.Matrix4d().SetTransform(Gf.Rotation(Gf.Quatd(float(Tw[6]), float(Tw[3]), float(Tw[4]), float(Tw[5]))), Gf.Vec3d(float(Tw[0]), float(Tw[1]), float(Tw[2])))
    return Gf.Matrix4d().SetScale(Gf.Vec3d(*scale)) * R


def restore_scene(st):
    """Last pre-physics hook: every recorded body and all the water are authored at the recorded moment, session layer, before PhysX loads.
    PhysX therefore starts from them at rest. Nothing has to be moved later, and the water never exists without its kettle and mug around it."""
    from pxr import UsdGeom, Vt
    xc = UsdGeom.XformCache()
    done, scaled = 0, []
    for i, path in enumerate(body_paths):
        prim = st.GetPrimAtPath(path)
        if not prim:
            raise SystemExit("recorded body missing from the scene: " + path)
        Mw = xc.GetLocalToWorldTransform(prim)
        sc = [float(np.linalg.norm([Mw[r][0], Mw[r][1], Mw[r][2]])) for r in range(3)]
        if max(abs(x - 1) for x in sc) > 1e-4:
            scaled.append((path, np.round(sc, 4).tolist()))
        M = pose_matrix(BT[f, i], sc) * xc.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(M)
        done += 1
    print("RESUME bodies authored", done, "scaled", scaled, flush=True)
    if F is not None:
        if F.n != len(P_rec):
            raise SystemExit(f"water count {F.n} differs from the recording {len(P_rec)}: FILL_ML must match the source run")
        F.inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(P_rec)))
        F.inst.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.zeros_like(P_rec)))
        print("RESUME water authored", len(P_rec), "particles from sample", pi, "t", round(float(pt[pi]), 3), flush=True)


def q_from_links(st, T_links, names):
    """Joint angles from recorded link transforms and the USD joint frames, for a source run with no <source>.npz (killed before finish)."""
    from pxr import Usd, UsdPhysics
    tail = [x.rsplit("/", 1)[-1] for x in link_paths]
    q = np.full(len(names), np.nan)
    for p in Usd.PrimRange(st.GetPrimAtPath("/World/h1_2"), Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdPhysics.RevoluteJoint) or p.GetName() not in names:
            continue
        j = UsdPhysics.RevoluteJoint(p)
        b0, b1 = j.GetBody0Rel().GetTargets()[0].name, j.GetBody1Rel().GetTargets()[0].name
        qa = lambda g: np.array([g.GetImaginary()[0], g.GetImaginary()[1], g.GetImaginary()[2], g.GetReal()], float)
        A = H.qmul(T_links[tail.index(b0), 3:7].astype(float), qa(j.GetLocalRot0Attr().Get()))
        B = H.qmul(T_links[tail.index(b1), 3:7].astype(float), qa(j.GetLocalRot1Attr().Get()))
        r = H.qmul(H.qinv(A), B)
        ax = {"X": 0, "Y": 1, "Z": 2}[j.GetAxisAttr().Get()]
        q[names.index(p.GetName())] = H.wrap(2 * np.arctan2(r[ax], r[3]))
    return q


# ---------------------------------------------------------------- the same scene as full_episode.py
import importlib.util as _iu


def load_file(path, name):
    sp = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


pw0 = load_file(ROOT + r"\steps\powder\stage.py", "st_pw0") if "powder" in ALL_STEPS else None
hook = GT.contact_pre(robot_links=GT.HAND_AND_FEET + ("right_elbow", "left_elbow", "right_knee", "left_knee", "right_shoulder", "left_shoulder"),
                      objects=GT.scene_sensors(), extra_paths=list(PR.prop_paths().values()) + ["/World/Props/jar_lid", "/World/Props/powder_scoop"])
pre = (list(pw0.pre_hooks()) if pw0 else []) + [hook]
F = None
if MODE == "gpu":
    sys.path.insert(0, ROOT + r"\fluids")
    import carb
    import omni.physx.bindings._physx as pb
    import boot
    import fluids
    carb.settings.get_settings().set(pb.SETTING_UPDATE_TO_USD, False)
    carb.settings.get_settings().set(pb.SETTING_UPDATE_PARTICLES_TO_USD, False)
    F = fluids.Fluids(fill_ml=float(os.environ["FILL_ML"]))

    def gpu_hook(st):
        sc = boot.set_scene(st, True, "GPU", "TGS")
        F.pre_physics(st, sc)
        boot.set_iters(st, int(os.environ["EP_ITERS"]))

    pre.append(gpu_hook)
pre.append(restore_scene)
s = PR.make_sim(app, pre=pre, add_props=True, xy=(0.62, 1.15), watch=(MUG, "object_0295"), mimic=True, pad=PAD)
if F is not None:
    F.attach(s.sv)
a = s.art
if link_paths != list(a.link_paths[0]):
    raise SystemExit("link order differs from the recording")

# ---------------------------------------------------------------- joint positions
q_note, Q = None, None
NPZ = SRC_BASE + ".npz"
q_links = q_from_links(s.stage, LT[f], s.names)
if os.environ.get("RESUME_Q", "auto") != "links" and os.path.exists(NPZ):
    N = np.load(NPZ, allow_pickle=False)
    j = int(np.argmin(np.abs(N["t"] - t_res)))
    if abs(N["t"][j] - t_res) < 1e-6 and [str(x) for x in N["dof_names"]] == s.names:
        Q = N["q"][j].astype(float)
        q_note = f"stored q from {SRC}.npz row {j}"
if Q is None:
    Q = q_links.copy()
    q_note = "recovered from link_T"
    if np.isnan(Q).any():
        raise SystemExit("joint recovery incomplete: " + str([n for n, v in zip(s.names, Q) if np.isnan(v)]))
d_q = np.abs(Q - q_links)
print("RESUME q from", q_note, "| stored against recovered: max", round(float(np.nanmax(d_q)), 5), "rad at", s.names[int(np.nanargmax(d_q))], "nan", int(np.isnan(q_links).sum()), flush=True)
root = LT[f, 0].astype(np.float32)


def write_robot():
    a.set_dof_positions(s.f(Q), s.idx)
    a.set_dof_velocities(s.f(np.zeros(s.n)), s.idx)
    tf = a.get_root_transforms().copy()
    tf[0] = root
    a.set_root_transforms(tf, s.idx)
    a.set_root_velocities(np.zeros((1, 6), np.float32), s.idx)
    a.set_dof_position_targets(s.f(Q), s.idx)


def link_err():
    return float(np.linalg.norm(a.get_link_transforms()[0].astype(float)[:, :3] - LT[f, :, :3], axis=1).max())


# ---------------------------------------------------------------- Python state the earlier steps leave behind. Gains, balance and references are not recorded: each line rebuilds what full_episode.py and the stages leave at a step boundary.
ty = s.finger_idx("L", "thumb_yaw")[0][0]
s.kp[ty] = 10.0                                              # full_episode.py at the mug grasp: stays for the whole run
s.hold[:] = Q                                                # arm and hand references: not recorded; measured pose, so no drive error at the join
if os.environ.get("RESUME_PINCH_INDEX"):                     # mid-grip restart: commanded squeeze, not the measured pose, or the held object drops on the first steps
    for i_, m_ in s.finger_idx("R", "index"):
        s.hold[i_] = float(os.environ["RESUME_PINCH_INDEX"]) * m_
write_robot()
s.to_stand(0.6)                                              # stand mode, stand gains, every step boundary ends here
s.stand_from, s.stand_goal = s.default.copy(), s.default.copy()
s.stand_ramp, s.stand_t = 0.6, 0.6                           # ramp long finished: reference is the default stance plus the balance offset, as at each boundary
s.balance = mug_place.stand_balance(s, H.Sim.balance.__get__(s))    # installed at mug_done, handed back by every CrouchSupport.uninstall()
s.want[:], s.cmd[:] = 0, 0
s.t = t_res
s.k = int(round(t_res / H.DT))
e_write = link_err()
try:
    s.sv.update_articulations_kinematic()
except Exception as e:
    print("RESUME update_articulations_kinematic:", repr(e), flush=True)
e_kin = link_err()
e_raw = None
if e_kin > 1e-3:                                             # links not refreshed by the write: one raw physics step at rest so the balance loop never reads the spawn pose
    s.sim.simulate(H.DT, s.t)
    s.sim.fetch_results()
    e_raw = link_err()
    write_robot()
s._link_pose_tick = -1
print("RESUME link position error against the recording, m: after write", round(e_write, 5), "after kinematic update", round(e_kin, 5), "after raw step", e_raw, flush=True)

# ---------------------------------------------------------------- recording, gates, step context: rebuilt new, they cover only what runs after the join
rec = dict(t=[], q=[])
orig = s.step


def step():
    orig()
    if s.k % 20 == 0:
        rec["t"].append(s.t)
        rec["q"].append(a.get_dof_positions()[0].astype(np.float32).copy())


s.step = step
near = {k: k for k in ("object_0310", "object_0312", "object_0317", "object_0318", "object_0319", "object_0295", "object_0309")}
G = GT.Gates(s, still=near, contact_paths=hook.paths, name=TAG,
             spill=(lambda: F.report().get("spill_ml") if F.P is not None else None) if F is not None else None,
             torque_spike_steps=int(os.environ.get("SPIKE", 0))).attach()
G.baseline()
G.watch_still(MUG)
bodies = RC.default_bodies(s)
for x in ("/World/Props/jar_lid", "/World/Props/powder_scoop"):
    if s.stage.GetPrimAtPath(x):
        bodies.append(x)
gv = s.sv.create_rigid_body_view("/World/fluid/grains/g*") if s.stage.GetPrimAtPath("/World/fluid/grains") else None


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


META = dict(kind="RESUMED_RUN", disclosure="State restored from a recording: a disclosed join, not one continuous physics run.", source_run=SRC, source_rec=SRC_BASE + "_rec.npz",
            source_frame=f, resume_t=t_res, asked=WHERE, tag=TAG, first_step=step0, joint_source=q_note, settle_s=SETTLE, mode=MODE,
            overrides=OVERRIDE, env={k: os.environ.get(k) for k in ("MODE", "EP_ITERS", "PAD", "FILL_ML", "SQ", "LOW", "TSQ", "P_SCOOP", "REC_AUTOSAVE_S", "KEEP_GOING", "LEVEL")})


class ResumeRecorder(RC.Recorder):
    def save(self, path):
        """record.Recorder.save with the join written into the file itself."""
        extra = {}
        if self.p_file is not None:
            self.p_file.flush()
            extra = dict(p_t=np.array(self.p_t), p_n=np.array(self.p_n, np.int64))
        np.savez_compressed(path, t=np.array(self.t), fps=self.fps, link_paths=np.array(self.link_paths), link_T=np.array(self.L), body_paths=np.array(self.body_paths),
                            body_T=np.array(self.B), resume_meta=np.array(json.dumps(META)), **extra)
        print("recorder saved", len(self.t), "frames to", path, "particles", len(self.p_t), "bad", self.p_bad, flush=True)


REC = ResumeRecorder(s, bodies=bodies, fps=30, particles=(F.sample if F is not None else None), particle_fps=P_FPS, views=[gv] if gv is not None else [],
                     particle_file=OUT + "_rec.npz.particles.f32").attach()
if REC.body_paths != body_paths:
    raise SystemExit("recorded bodies differ from this scene")
d_b = np.linalg.norm(REC.B[0][:, :3].astype(float) - BT[f, :, :3], axis=1)
d_l = np.linalg.norm(REC.L[0][:, :3].astype(float) - LT[f, :, :3], axis=1)
i_b = int(d_b.argmax())
restore = dict(link_mm_max=round(float(d_l.max()) * 1000, 3), body_mm_max=round(float(d_b.max()) * 1000, 3), body_worst=body_paths[i_b], link_err_after_write_m=e_write, link_err_after_kinematic_m=e_kin, link_err_after_raw_step_m=e_raw,
               source_link_speed=link_v, source_body_speed=body_v, source_water_speed_mean=float(water_v.mean()), source_water_speed_max=float(water_v.max()), q_stored_vs_recovered_rad=float(np.nanmax(d_q)))
print("RESUME restore", restore, flush=True)


# ---------------------------------------------------------------- run
def snapshot():
    p, q = s.pose()
    k, m = s.obj("object_0295"), s.obj(MUG)
    row = dict(t=round(s.t, 3), pelvis=np.round(p, 4).tolist(), mug=np.round(m[:3], 4).tolist(), kettle=np.round(k[:3], 4).tolist(), fallen=bool(s.fallen()))
    if F is not None and F.P is not None:
        w = F.report()
        row["water"] = {x: w.get(x) for x in ("kettle_ml", "mug_ml", "other_ml", "spill_ml", "level_mm", "zmin")}
    return row


inner = s.step
watch = dict(next=t_res + 1.0, rows=[], capped=False)
t_stop = t_res + SIM_S if SIM_S > 0 else None


def outer_step():
    inner()
    if s.t >= watch["next"]:
        watch["next"] += 1.0
        r = snapshot()
        watch["rows"].append(r)
        print("RESUME at", r, "wall", round(time.time() - t_wall), flush=True)
    if MAX_P_GB and not watch["capped"] and REC.p_file is not None and sum(REC.p_n) * 12 / 1e9 > MAX_P_GB:
        watch["capped"] = True
        REC.particles = None                                 # stops the recording only: the stages sample water themselves
        print("RESUME water recording capped at", MAX_P_GB, "GB, t", round(s.t, 2), flush=True)
    if t_stop is not None and s.t >= t_stop - 1e-9:
        raise StopResume()


s.step = outer_step
gates_steps = {}
why = "finished"


def write_result(why):
    res = dict(META, result=why, sim_t_end=round(s.t, 3), sim_s_after_resume=round(s.t - t_res, 3), wall_s=round(time.time() - t_wall, 1), restore=restore, watch=watch["rows"], steps=gates_steps,
               full_episode_passed=False, overall="NOT A FULL RUN: disclosed join", simulation=dict(engine="Isaac Sim PhysX", mode=MODE, pad_friction=PAD, live_water=F is not None))
    REC.save(OUT + "_rec.npz")                               # the recording first: nothing below may cost it
    np.savez_compressed(OUT + ".npz", t=np.array(rec["t"]), q=np.array(rec["q"]), dof_names=np.array(s.names), resume_meta=np.array(json.dumps(META)))
    src_json = SRC_BASE + ".json"
    if os.path.exists(src_json):                             # what the source run reported for the steps before the join; these are its numbers, not re-run here
        before = ["mug"] + order[:order.index(step0)]
        src_steps = json.load(open(src_json)).get("gates", {}).get("steps", {})
        res["steps_before_the_join_from_source"] = {k: {x: src_steps[k].get(x) for x in ("status", "task_status")} for k in before if k in src_steps}
    res["stage_files"] = {p: sha(p) for p in [ROOT + r"\steps\teabag_stir\stage_live.py", ROOT + r"\steps\kettle\stage_live.py", ROOT + r"\steps\powder\stage_live.py", ROOT + r"\cup_walk_away.py"] + list(OVERRIDE.values())}
    try:
        res["physics_gates_after_join"] = G.report(OUT + "_gates.json")
    except Exception as e:
        res["gates_error"] = repr(e)
    json.dump(res, open(OUT + ".json", "w"), indent=1, default=float)
    print("RESULT", why, "sim", round(s.t, 2), "wall", round(time.time() - t_wall), flush=True)


try:
    if F is not None:
        F.sample(s.step)
        w0 = F.report()
        restore["water_first_step"] = w0
        print("RESUME water after the first step", w0, flush=True)
    if SETTLE > 0:
        s.hold_for(SETTLE)
        print("RESUME settled", snapshot(), flush=True)
    if HOLD:
        while True:
            s.hold_for(0.5)
    REC.save(OUT + "_rec.npz")

    steps_todo = [x for x in os.environ.get("RESUME_STEPS", ",".join(order[order.index(step0):])).split(",") if x]
    if not CUP and "cup" in steps_todo:
        steps_todo.remove("cup")
    ROOT_S = ROOT + r"\steps"
    tb = load_file(ROOT_S + r"\teabag_stir\stage.py", "st_tb")
    kt = load_file(ROOT_S + r"\kettle\stage.py", "st_kt")
    pw = load_file(ROOT_S + r"\powder\stage.py", "st_pw")
    RUN = dict(teabag=tb.run_teabag, stir=tb.run_stir, kettle=kt.run, powder=pw.run)
    FN = dict(teabag="run_teabag", stir="run_stir", kettle="run", powder="run", cup="run")
    for name, path in OVERRIDE.items():
        RUN[name] = getattr(load_file(path, "override_" + name), FN[name])
        print("RESUME override", name, path, flush=True)
    for name in steps_todo:
        if s.fallen():
            gates_steps[name] = dict(status="NOT RUN", reason="robot had fallen")
            continue
        t1, w1 = s.t, time.time()
        if name == "cup":
            import cup_walk_away
            fn = RUN.get("cup", cup_walk_away.run)
            ctx = dict(gates=G, water=F)
        else:
            fn = RUN[name]
            ctx = dict(mug_xy=s.obj(MUG)[:2].tolist(), gates=G, mug=s.objs[MUG].GetPath().pathString if name == "powder" else MUG, water=F, fluids=F, contact_paths=hook.paths)
        print("STEP START", name, "t", round(s.t, 1), "mug", s.obj(MUG)[:2].tolist(), flush=True)
        try:
            r = fn(s, ctx)
        except Exception as e:
            traceback.print_exc()
            r = dict(status="FAIL", error=repr(e)[:300])
        r = dict(r or {}, sim_s=round(s.t - t1, 1), wall_s=round(time.time() - w1))
        gates_steps[name] = r
        print("STEP END", name, r.get("status"), "sim", r["sim_s"], "wall", r["wall_s"], "fallen", s.fallen(), json.dumps(r, default=str)[:600], flush=True)
        if F is not None:
            F.sample(s.step)
            gates_steps[name]["water"] = F.report()
            print("WATER", gates_steps[name]["water"], flush=True)
        json.dump(dict(META, steps=gates_steps), open(OUT + "_partial.json", "w"), indent=1, default=float)
        REC.save(OUT + "_rec.npz")
except StopResume:
    why = f"stopped at the requested {SIM_S} sim s"
except Exception as e:
    traceback.print_exc()
    why = "crashed: " + repr(e)[:200]
s.step = inner                                               # nothing below may stop on the time limit
print("RESUME end", snapshot(), flush=True)
write_result(why)
app.close()
sys.exit(0)
