"""The whole H1-2 tea episode in one continuous PhysX run: mug, then each later step on the same live robot.
MODE=cpu has no water (rehearsal). MODE=gpu has live water and grains. STEPS picks the later steps, in order.
Contact-only grip, URDF torque limits, no welds, no state writes after the initial spawn."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))    # release_paths.py lives at the code root
import release_paths as RP
import json
import os
import sys
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
sys.path.insert(0, RP.code())
import h12_physx as H
import numpy as np

sys.dont_write_bytecode = True          # leave no bytecode beside the shared modules
sys.path.insert(0, RP.code("balance_support"))
from crouch_support import CrouchSupport

sys.path.insert(0, RP.code('props'))
import gates as GT
import props as PR
import record as RC

MUG = "object_0316"
TAG = sys.argv[1] if len(sys.argv) > 1 else "e1"
START = sys.argv[2] if len(sys.argv) > 2 else "walk"     # "walk" = from the door, "stand" = spawn at the rack (fast debug)
MIMIC = len(sys.argv) > 3 and sys.argv[3] == "mimic"     # real PhysX finger linkage: one motor per finger
SPOT = np.array([1.90, 2.17])
Q = np.array([0.0, 0.0, 1.0, 0.0])
TX, TY, GH = 0.047, 0.165, 0.048
SQ, LOW, TSQ = float(os.environ.get("SQ", 0.3)), float(os.environ.get("LOW", 0.7)), float(os.environ.get("TSQ", 0.15))
PAD = float(os.environ.get("PAD", 0))
GRIP_ONLY = os.environ.get("GRIP_ONLY") == "1"
DROP_RACK, DROP_SPOT = 0.035, 0.13      # pelvis drop in metres, measured and corrected by CrouchSupport
OUT = RP.out("logs", TAG)

MODE = os.environ.get("MODE", "cpu")
STEPS = [x for x in os.environ.get("STEPS", "teabag,kettle,powder,stir").split(",") if x]
import importlib.util as _iu


def load0(path, name):
    sp = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


pw0 = load0(RP.code('steps', 'powder', 'stage.py'), "st_pw0") if "powder" in STEPS else None
hook = GT.contact_pre(robot_links=GT.HAND_AND_FEET + ("right_elbow", "left_elbow", "right_knee", "left_knee", "right_shoulder", "left_shoulder"),
                      objects=GT.scene_sensors(), extra_paths=list(PR.prop_paths().values()) + ["/World/Props/jar_lid", "/World/Props/powder_scoop"])
pre = (list(pw0.pre_hooks()) if pw0 else []) + [hook]
F = None
if MODE == "gpu":
    sys.path.insert(0, RP.code('fluids'))
    import carb
    import omni.physx.bindings._physx as pb
    import boot
    import fluids
    carb.settings.get_settings().set(pb.SETTING_UPDATE_TO_USD, False)
    carb.settings.get_settings().set(pb.SETTING_UPDATE_PARTICLES_TO_USD, False)
    F = fluids.Fluids(fill_ml=float(os.environ.get("FILL_ML", "400")))

    def gpu_hook(st):
        sc = boot.set_scene(st, True, "GPU", "TGS")
        F.pre_physics(st, sc)
        boot.set_iters(st, int(os.environ.get("EP_ITERS", "32")))

    pre.append(gpu_hook)
s = PR.make_sim(app, pre=pre, add_props=True, xy=(0.62 + float(os.environ.get("MDX", 0)), (1.15 if START == "walk" else 1.78) + float(os.environ.get("MDY", 0))), watch=(MUG, "object_0295"), mimic=True, pad=PAD)
if F is not None:
    F.attach(s.sv)
a = s.art
t0 = time.time()
rec = dict(t=[], links=[], q=[], mug=[], root=[])
log, gates = {}, dict(peak_torque_ratio=0.0)
orig = s.step
lim = a.get_dof_max_forces()[0].astype(float)


def step():
    orig()
    if s.k % 20 == 0:
        rec["t"].append(s.t)
        rec["links"].append(a.get_link_transforms()[0].astype(np.float32).copy())
        rec["q"].append(a.get_dof_positions()[0].astype(np.float32).copy())
        rec["mug"].append(s.obj(MUG).astype(np.float32).copy())
        tq = np.abs(a.get_dof_projected_joint_forces()[0].astype(float))
        gates["peak_torque_ratio"] = max(gates["peak_torque_ratio"], float((tq / np.maximum(lim, 1e-6)).max()))


s.step = step


def mug_in_hand():
    h = s.hand("L")
    return H.qrot(H.qinv(h[3:7]), s.obj(MUG)[:3] - h[:3])


def tilt():
    return float(np.degrees(np.arccos(np.clip(H.qrot(s.obj(MUG)[3:7], np.array([0, 0, 1.0]))[2], -1, 1))))


def stage(name, err=None):
    h, m = s.hand("L"), s.obj(MUG)
    row = dict(t=round(s.t, 2), wall=round(time.time() - t0), err_mm=None if err is None else round(err * 1000, 1), hand=np.round(h[:3], 3).tolist(),
               pelvis=np.round(s.pose()[0], 3).tolist(), mug=np.round(m[:3], 3).tolist(), tilt=round(tilt(), 1), touch=s.contact(MUG), fallen=bool(s.fallen()))
    log[name] = row
    print(name, row, flush=True)
    if s.fallen():
        finish("FELL at " + name)


def finish(why):
    res = dict(result=why, sim_s=round(s.t, 2), wall_s=round(time.time() - t0, 1), gates=gates, stages=log,
               full_episode_passed=False, overall="FAIL",
               simulation=dict(engine="Isaac Sim PhysX", mode=MODE, pad_friction=PAD,
                               continuous_from_entrance=START == "walk", live_water=F is not None))
    try:
        R = G.report(OUT + "_gates.json")
        res["physics_gates"] = R
        required = ("mug", "teabag", "kettle", "powder", "stir", "cup_walk_away")
        stages_ok = all(gates.get("steps", {}).get(k, {}).get("status") == "PASS" for k in required)
        res["full_episode_passed"] = bool(why == "finished" and START == "walk" and F is not None and R["overall"] == "PASS" and stages_ok)
        res["overall"] = "PASS" if res["full_episode_passed"] else "FAIL"
        gg = R["gates"]
        motor = float(gg.get("torque", {}).get("motor_estimate_ratio", 9.0))
        task_ok = {k: gates.get("steps", {}).get(k, {}).get("task_status", gates.get("steps", {}).get(k, {}).get("status")) for k in required}
        res["task_steps"] = task_ok
        checks = dict(all_steps_task_done=all(v == "PASS" for v in task_ok.values()), finished=why == "finished", from_entrance=START == "walk", live_water=F is not None,
                      fall=gg.get("fall", {}).get("status") == "PASS", penetration=gg.get("penetration", {}).get("status") == "PASS",
                      moved=gg.get("moved", {}).get("status") == "PASS", spill=gg.get("spill", {}).get("status") == "PASS", motors_within_limit=motor <= 1.0)
        res["task_done"] = dict(passed=all(checks.values()), checks=checks, rule="working rule, not confirmed by the user; strict verdict is full_episode_passed",
                                strict_slip=gg.get("slip", {}).get("status"), strict_torque=gg.get("torque", {}).get("status"))
        print("TASK_DONE", res["task_done"], flush=True)
        res["required_steps"] = {k: gates.get("steps", {}).get(k, {}).get("status", "NOT RUN") for k in required}
        print("GATES", json.dumps(R, default=float)[:1500], flush=True)
        REC.save(OUT + "_rec.npz")
    except Exception as exc:
        res["report_error"] = repr(exc)
    print("RESULT", why, gates, "sim", round(s.t, 1), "wall", round(time.time() - t0), flush=True)
    json.dump(res, open(OUT + ".json", "w"), indent=1, default=float)
    np.savez_compressed(OUT + ".npz", **{k: np.array(v) for k, v in rec.items()}, link_names=np.array(s.links), dof_names=np.array(s.names))
    app.close()
    sys.exit(0 if res["full_episode_passed"] else 2)


def walk_to(x, y, yaw=90.0, tol=0.025, gentle=False):
    """Walk to the stance, keep correcting position until the stand switch, stand on a low-foot moment."""
    if not s.go(x, y, yaw, tol=tol, log=lambda *z, **k: None):
        finish(f"walk timed out on the way to {x}, {y}")
    gy = np.radians(yaw)
    fz = []
    for n in range(240):
        p, q = s.pose()
        yw = H.yaw_of(q)
        e = np.array([x, y]) - p[:2]
        c, sn = np.cos(yw), np.sin(yw)
        s.want = np.array([H.min_cmd(c * e[0] + sn * e[1], 1.0, 0.14, tol), H.min_cmd(-sn * e[0] + c * e[1], 1.0, 0.14, tol),
                           np.clip(1.2 * H.wrap(gy - yw), -0.3, 0.3)])
        for _ in range(H.PERIOD):
            s.step()
            z = s.feet_xy()[:, 2]
            fz.append(z.copy())
        if n >= 60 and float(np.linalg.norm(e)) < 0.02 and np.all(z - np.min(fz[-120:], axis=0) < 0.004):
            break
    s.want = np.zeros(3)
    if gentle:                                               # carrying: let the speed die out, blend slowly; costs a few cm of stance
        fz = []
        for _ in range(900):
            s.step()
            z = s.feet_xy()[:, 2]
            fz.append(z.copy())
            if len(fz) > 480 and np.all(z - np.min(fz[-120:], axis=0) < 0.004):
                break
        s.to_stand(1.5)
        s.hold_for(2.2)
    else:
        s.to_stand(0.6)
        s.hold_for(1.5)
    gates.setdefault("stance_err_mm", []).append(round(float(np.linalg.norm(np.array([x, y]) - s.pose()[0][:2])) * 1000, 1))


def to_walk():
    """Back to the walking network from the stance: fresh memory, walking leg gains, zero command warm-up."""
    s.h[:], s.c[:], s.last[:] = 0, 0, 0
    s.target = s.art.get_dof_positions()[0].astype(float)[s.legs].copy()
    kp = a.get_dof_stiffnesses()[0].astype(float).copy()
    kd = a.get_dof_dampings()[0].astype(float).copy()
    kp[s.legs], kd[s.legs] = H.G["kp"], H.G["kd"]
    a.set_dof_stiffnesses(s.f(kp), s.idx)
    a.set_dof_dampings(s.f(kd), s.idx)
    s.want[:], s.cmd[:] = 0, 0
    s.mode = "walk"


def links_of(key):
    return ("L_thumb_proximal", "L_thumb_intermediate", "L_thumb_distal") if key == "thumb_pitch" else (f"L_{key}_proximal", f"L_{key}_intermediate", f"L_{key}_distal")


def close(keys, rate=0.5, thresh=0.4, top=1.4, timeout=4.0):
    cur = {k: s.hold[s.finger_idx("L", k)[0][0]] for k in keys}
    done, t = {}, 0.0
    while len(done) < len(keys) and t < timeout:
        c = s.contact(MUG)
        for k in keys:
            if k in done:
                continue
            if any(c.get(n, 0.0) > thresh for n in links_of(k)):
                done[k] = cur[k]
            else:
                cur[k] = min(top, cur[k] + rate * 0.02)
        s.fingers("L", {k: cur[k] for k in keys}, 0.02)
        t += 0.02
    return done, cur


# ---- walk in
for _ in range(300):
    s.step()
near = {k: k for k in ("object_0310", "object_0312", "object_0317", "object_0318", "object_0319", "object_0295", "object_0309")}
G = GT.Gates(s, still=near, contact_paths=hook.paths, name=TAG,
             spill=(lambda: F.report().get("spill_ml") if F.P is not None else None) if F is not None else None,
             torque_spike_steps=int(os.environ.get("SPIKE", 0))).attach()
G.baseline()
bodies = RC.default_bodies(s)
for x in ("/World/Props/jar_lid", "/World/Props/powder_scoop"):
    if s.stage.GetPrimAtPath(x):
        bodies.append(x)
gv = s.sv.create_rigid_body_view("/World/fluid/grains/g*") if s.stage.GetPrimAtPath("/World/fluid/grains") else None
REC = RC.Recorder(s, bodies=bodies, fps=30, particles=(F.sample if F is not None else None), particle_fps=10, views=[gv] if gv is not None else []).attach()
print("recorder bodies", len(bodies), "grains view", gv is not None, "particles", F is not None, flush=True)
if START == "walk":
    walk_to(0.62, 1.785)
else:
    s.to_stand(ramp=0.3)
    s.hold_for(1.0)
m0 = s.obj(MUG).copy()
stage("at_rack")

# ---- grasp
sup = CrouchSupport(s)
gates["crouch_rack"] = sup.crouch(DROP_RACK, seconds=2.0, settle=0.8)
axis = np.array([m0[0], m0[1], m0[2] + GH])
hand_at = axis - np.array([TX, TY, 0.0])
out = hand_at + np.array([-0.020, 0.0, 0.0])
s.fingers("L", dict(index=0.0, middle=0.0, ring=0.0, pinky=0.0, thumb_yaw=-0.1, thumb_pitch=0.0), 0.2)
stage("side", s.ik_to(np.array([0.40, 1.92, 1.10]), Q, 1.5))
stage("front", s.ik_to(out + np.array([0.0, -0.17, 0.0]), Q, 1.2))
ang = float(np.degrees(np.linalg.norm(H.qerr(Q, s.hand("L")[3:7]))))
print("hand angle at front", round(ang, 2), "pelvis", np.round(s.pose()[0], 3), flush=True)
if ang > 3.0:
    finish(f"stance wrong: hand {ang:.1f} deg off before the approach")
stage("beside", s.ik_to(out, Q, 1.6))
stage("palm", s.ik_to(hand_at, Q, 1.5, stop=lambda: sum(s.contact(MUG).values()) > 0.2))
kp = a.get_dof_stiffnesses()[0].astype(float).copy()
kp[s.finger_idx("L", "thumb_yaw")[0][0]] = 10.0
s.kp[s.finger_idx("L", "thumb_yaw")[0][0]] = 10.0     # to_stand() restores s.kp: the thumb must stay stiff through it
a.set_dof_stiffnesses(s.f(kp), s.idx)
s.fingers("L", dict(thumb_yaw=1.3), 1.2)
s.hold_for(0.6)
_, cur = close(["thumb_pitch"], rate=0.25, top=0.55)
_, cur2 = close(["index", "middle", "ring", "pinky"], rate=0.5)
tgt = {k: cur2[k] + (SQ if k in ("index", "middle") else LOW * SQ) for k in cur2}
tgt["thumb_pitch"] = cur["thumb_pitch"] + TSQ
s.fingers("L", tgt, 0.6)
s.hold_for(0.4)
stage("gripped")
rel0 = mug_in_hand()
G.allow_move(MUG)
G.hold(MUG, "L_hand_base_link")
h = s.hand("L")[:3]
stage("lift", s.ik_to(h + np.array([0.0, 0.0, 0.06]), Q, 2.8))
gates["lift_rise_mm"] = round((s.obj(MUG)[2] - m0[2]) * 1000, 1)
h = s.hand("L")[:3]
stage("out", s.ik_to(h + np.array([0.0, -0.17, 0.0]), Q, 3.2))
s.hold_for(0.5)
gates["slip_after_lift_mm"] = round(float(np.linalg.norm(mug_in_hand() - rel0)) * 1000, 1)
if s.obj(MUG)[2] < m0[2] + 0.02:
    finish("mug not lifted")
if GRIP_ONLY:
    z1 = s.obj(MUG)[2]
    s.hold_for(3.0)
    stage("held3s")
    gates["creep_3s_mm"] = round(float(abs(s.obj(MUG)[2] - z1)) * 1000, 1)
    gates["slip_after_hold_mm"] = round(float(np.linalg.norm(mug_in_hand() - rel0)) * 1000, 1)
    finish("grip test held" if s.obj(MUG)[2] > m0[2] + 0.02 else "grip test dropped")
gates["rise_rack"] = sup.crouch(0.0, seconds=2.0, settle=0.8)
sup.uninstall()
stage("stood_up")

# ---- carry
from cup_walk_away import CarryStabilizer
carry_stabilizer = CarryStabilizer(s)
to_walk()
for _ in range(300):
    s.step()
stage("walk_start")
gx = SPOT[0] + 0.12
walk_to(gx, 1.81, gentle=True)
carry_stabilizer.detach()
stage("at_spot")
REC.save(OUT + "_rec.npz")
gates["slip_after_carry_mm"] = round(float(np.linalg.norm(mug_in_hand() - rel0)) * 1000, 1)
if float(np.linalg.norm(mug_in_hand() - rel0)) > 0.05:
    finish("mug lost in carry")

# ---- set down: measured descent and balanced release live in mug_place.py (only joint references change)
import mug_place
TOP = 0.86
# The walk ends with uneven feet and to_stand() commands the default pose against them: friction alone holds that strain,
# and it snapped the right foot 34 mm on the way back up (m_cpu01/02: 1471 Nm spike, 3.4 mm into the floor). Standing
# references move onto the measured stance first. Joint references only.
s.stand_from = s.stand_goal.copy()
s.stand_goal = a.get_dof_positions()[0].astype(float)[s.legs].copy()
s.stand_ramp, s.stand_t = 1.0, 0.0
s.hold_for(1.2)
sup = CrouchSupport(s)
gates["crouch_spot"] = sup.crouch(DROP_SPOT, seconds=4.0, settle=2.5)
stage("crouched_spot")


def at_contact():
    gates["slip_when_set_down_mm"] = round(float(np.linalg.norm(mug_in_hand() - rel0)) * 1000, 1)


pl = mug_place.run(s, G, spot=SPOT, top=TOP, name=MUG, on_contact=at_contact)
gates["place"] = {k: v for k, v in pl.items() if k != "stages"}
log["place_stages"] = pl["stages"]
stage("released")
m1 = s.obj(MUG).copy()
gates["rise_spot"] = sup.crouch(0.0, seconds=3.5, settle=1.0)
s.hold_for(2.0)
stage("end")
m2 = s.obj(MUG)
gates.update(slip_final_note="see gate checker", place_err_mm=round(float(np.linalg.norm(m2[:2] - SPOT)) * 1000, 1), mug_moved_on_release_mm=round(float(np.linalg.norm(m2[:3] - m1[:3])) * 1000, 1),
             mug_tilt_deg=round(tilt(), 1), mug_on_counter=bool(abs(m2[2] - TOP) < 0.01), fallen=bool(s.fallen()))
gates["task_status"] = "PASS" if gates["mug_on_counter"] and gates["place_err_mm"] <= 25 and gates["mug_tilt_deg"] < 5 and not gates["fallen"] else "FAIL"   # upright, on the counter within 25 mm, not dropped; strict gates are reported beside it
gates["place_within_5mm"] = bool(gates["place_err_mm"] <= 5)
sup.old_balance = mug_place.stand_balance(s, sup.old_balance)   # sup.uninstall() hands this back: lateral feedback on the true stance
sup.uninstall()
ai = s.arm_idx("L")
s.fingers("L", dict(index=0.0, middle=0.0, ring=0.0, pinky=0.0, thumb_yaw=-0.1, thumb_pitch=0.0), 0.5)
qa, qb = s.hold[ai].copy(), np.array([H.G["hold_pose"].get(s.names[j], 0.0) for j in ai])
for n_ in range(100):                                        # left arm back to the walking pose
    u = (n_ + 1) / 100
    s.hold[ai] = qa + (u * u * (3 - 2 * u)) * (qb - qa)
    for _ in range(H.PERIOD):
        s.step()
stage("mug_done")
G.watch_still(MUG)

# ---- every later step on the same live robot; a step that fails or is not built is labelled, and the run goes on
import importlib.util
import traceback


def load(path, name):
    if not os.path.exists(path):
        return None
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


ROOT = RP.code('steps')
tb = load(os.path.join(ROOT, "teabag_stir", "stage.py"), "st_tb")
kt = load(os.path.join(ROOT, "kettle", "stage.py"), "st_kt")
pw = load(os.path.join(ROOT, "powder", "stage.py"), "st_pw")
RUN = dict(teabag=tb and tb.run_teabag, stir=tb and tb.run_stir, kettle=kt and kt.run, powder=pw and pw.run)
gates["steps"] = {"mug": dict(status="PASS" if gates["mug_on_counter"] and gates["place_err_mm"] <= 5 and gates["mug_tilt_deg"] < 5 else "FAIL",
                             place_err_mm=gates["place_err_mm"], tilt_deg=gates["mug_tilt_deg"], task_status=gates["task_status"])}
KEEP_GOING = os.environ.get("KEEP_GOING", "0") == "1"
if gates["steps"]["mug"]["status"] != "PASS" and not KEEP_GOING:
    finish("mug placement did not pass; dependent tasks not started")
_order = RP.code('final_steps.txt')
if os.path.exists(_order):
    STEPS = [x.strip() for x in open(_order).read().split(",") if x.strip()]
    print("STEP ORDER from file", STEPS, flush=True)
for name in STEPS:
    if s.fallen():
        gates["steps"][name] = dict(status="NOT RUN", reason="robot had fallen")
        continue
    if RUN.get(name) is None:
        gates["steps"][name] = dict(status="NOT BUILT")
        print("STEP", name, "NOT BUILT", flush=True)
        continue
    t1, w1 = s.t, time.time()
    ctx = dict(mug_xy=s.obj(MUG)[:2].tolist(), gates=G, mug=s.objs[MUG].GetPath().pathString if name == "powder" else MUG, water=F, fluids=F, contact_paths=hook.paths)
    print("STEP START", name, "t", round(s.t, 1), "mug", ctx["mug_xy"], flush=True)
    try:
        r = RUN[name](s, ctx)
    except Exception as e:
        traceback.print_exc()
        r = dict(status="FAIL", error=repr(e)[:300])
    r = dict(r or {}, sim_s=round(s.t - t1, 1), wall_s=round(time.time() - w1))
    gates["steps"][name] = r
    print("STEP END", name, r.get("status"), "sim", r["sim_s"], "wall", r["wall_s"], "fallen", s.fallen(), json.dumps(r, default=str)[:600], flush=True)
    if F is not None:
        F.sample(s.step)
        gates["steps"][name]["water"] = F.report()
        print("WATER", gates["steps"][name]["water"], flush=True)
    json.dump(dict(gates=gates, stages=log), open(OUT + "_partial.json", "w"), indent=1, default=float)
    REC.save(OUT + "_rec.npz")

# ---- pick up the prepared cup, then walk away while physically holding it
import importlib
import cup_walk_away
importlib.reload(cup_walk_away)
if not s.fallen() and (KEEP_GOING or all(gates["steps"].get(n, {}).get("status") == "PASS" for n in ("teabag", "kettle", "powder", "stir"))):
    try:
        gates["steps"]["cup_walk_away"] = cup_walk_away.run(s, dict(gates=G, water=F))
        stage("cup_walk_away_end")
    except Exception as e:
        traceback.print_exc()
        gates["steps"]["cup_walk_away"] = dict(status="FAIL", reason=repr(e)[:200])
else:
    gates["steps"]["cup_walk_away"] = dict(status="NOT RUN", reason="tea preparation did not pass or robot fell")
finish("finished")
