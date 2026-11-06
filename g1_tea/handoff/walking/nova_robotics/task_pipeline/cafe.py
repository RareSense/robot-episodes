"""Run the robot task in the database-exported Cafe Robotique scene."""

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from nova_robotics.byor import analyze_model, load_robot
from nova_robotics.byor.navigate import goto
from nova_robotics.byor.scene import prepare_environment
from nova_robotics.byor.skills import Tasks
from nova_robotics.byor.teleop import TeleopSession

from .fixture import BIN_BOUNDS, CLUTTER, DROP_POINT, add_cleanup_fixture
from .recording import TraceRecorder
from .spec import write_json


def build(request, seed):
    loaded = load_robot(
        request.robot,
    )
    report = analyze_model(loaded.model)
    spec = loaded.spec
    for key in list(spec.keys):
        spec.delete(key)
    root = spec.body("pelvis")
    root.pos[:2] = [2.85, 4.35]
    root.quat[:] = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
    if request.task_family == "tabletop_cleanup":
        root.pos[:2] = [5.95, 7.0]
    env, provenance = prepare_environment(
        request.scene,
        active_joints=["object_0074_joint", "object_0076_joint"],
        rigid_groups=[
            ("object_0075_body", "object_0074_body"),
            ("object_0077_body", "object_0076_body"),
        ],
    )
    provenance["task_fixture"] = add_cleanup_fixture(
        env, seed=seed, variation_m=request.variation_m
    )
    spec.attach(env, prefix="env/", frame=spec.worldbody.add_frame())
    spec.visual.global_.offwidth = 960
    spec.visual.global_.offheight = 720
    spec.worldbody.add_light(pos=[3, 4, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    light = float(
        np.random.default_rng(seed + 1).uniform(
            1 - request.lighting_variation, 1 + request.lighting_variation
        )
    )
    for lamp in spec.lights:
        lamp.diffuse[:] = np.clip(np.array(lamp.diffuse) * light, 0, 1)
    # Fixed virtual camera mounts, not a cinematic tracking/cutaway view.
    head = spec.body("torso_link")
    head.add_camera(
        name="sim_head_rgb", pos=[0.12, 0, 0.4], xyaxes=[0, -1, 0, 0.65, 0, 0.76], fovy=85
    )
    wrist = spec.body("right_wrist_yaw_link")
    wrist.add_camera(
        name="sim_right_wrist_rgb", pos=[0.07, 0, 0.08], xyaxes=[0, -1, 0, 0.55, 0, 0.835], fovy=100
    )
    provenance.update(
        seed=seed,
        lighting_multiplier=light,
        scene_adapter="cafe_g1_v1",
        task_family=request.task_family,
    )
    provenance.update(
        project_id="prj_01a12fae51c60f193db4911cd4fa",
        project_name="Cafe Robotique",
        database="nova_sync",
        branch_id="br_01a12fae51c685a81889ae595717",
        task=request.task_family,
    )
    return spec, report, provenance


def run_episode(request, seed, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    spec, report, provenance = build(request, seed)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (out / "scene.xml").write_text(spec.to_xml())
    s = TeleopSession(spec, report, "cafe-cleanup", reach_fraction=0.9)
    mujoco.mj_saveModel(s.m, str(out / "scene.mjb"), None)
    neutral_posture = s.wbc.qpos0.copy()
    recorder = TraceRecorder(
        s, out / "trace.h5", max_sim_seconds=request.max_sim_seconds, policy_hz=request.policy_hz
    )
    s.step_observer = recorder
    t = Tasks(s)
    stages = []
    j = s.m.joint("env/object_0074_joint").qposadr[0]
    ignore = [s.m.body(f"env/object_{n:04d}_body").id for n in (74, 75, 76, 77)]

    def log(name, result):
        entry = {
            "stage": name,
            "time_s": round(s.d.time, 3),
            "result": result,
            "position": s.d.qpos[:3].tolist(),
            "door_angle": float(s.d.qpos[j]),
            "checks": s.checks.as_dict(),
        }
        stages.append(entry)
        (out / "stages.json").write_text(json.dumps(stages, indent=2))
        print(json.dumps(entry), flush=True)

    def walk(name, xy, yaw, timeout=30, extra_ignore=()):
        t._hands_home(1)
        r = goto(
            s,
            xy,
            final_yaw=yaw,
            speed=0.15,
            inflate=0.20,
            ignore_bodies=ignore + list(extra_ignore),
            goal_tol=0.08,
            timeout=timeout,
        )
        r["stood"] = t._wait_stand()
        r["settled_position_error_m"] = float(np.linalg.norm(s.d.qpos[:2] - xy))
        log(name, r)
        if not r["arrived"] or not r["stood"]:
            raise RuntimeError(name + " failed")

    success = False
    try:
        if request.task_family == "door_and_cleanup":
            walk("approach door", [2.85, 5.05], np.pi / 2)
            for _ in range(8):
                p = s.wbc.ee_pos(s.d, s.hands[0]).copy()
                p[1] += 0.06
                p[2] = 0.95
                t._hand_to(0, p, timeout=0.7, speed=0.12)
            log("push door", {"angle": float(s.d.qpos[j])})
            walk("walk through doorway", [2.95, 6.1], np.pi / 2)
            opened = float(s.d.qpos[j]) > 1.0
            log("door opened", {"ok": opened})
            if not opened:
                raise RuntimeError("door opening insufficient")
            walk("go behind door", [2.45, 6.65], np.pi)
            walk("approach closing side", [2.33, 6.3], 0)
            for _ in range(6):
                p = s.wbc.ee_pos(s.d, s.hands[1]).copy()
                p[0] += 0.06
                p[2] = 0.95
                t._hand_to(1, p, timeout=0.6, speed=0.12)
            log("start closing", {"angle": float(s.d.qpos[j])})
            log(
                "follow closing arc",
                t.push_hinge("env/object_0074_joint", [0.6, 0, 0.95], 0, timeout=7.2),
            )
            if abs(s.d.qpos[j]) > 0.08:
                walk("follow door to closed position", [2.75, 6.05], 0)
                log(
                    "finish closing arc",
                    t.push_hinge(
                        "env/object_0074_joint",
                        [0.6, 0, 0.95],
                        0,
                        timeout=16,
                        tolerance=0.02,
                        lead=0.012,
                        speed=0.04,
                    ),
                )
            t._tick(3)
            t._hands_home(1)
            closed = abs(float(s.d.qpos[j])) < 0.10
            log("door closed", {"ok": closed})
            if not closed:
                raise RuntimeError("door closure insufficient")
            walk("clear doorway", [3.3, 6.65], 0)
            walk("pass the storage rack", [5.95, 7.0], np.pi / 2, timeout=60)
            walk(
                "walk to cleanup station",
                [6.98, 7.1],
                -np.pi / 2,
                timeout=60,
                extra_ignore=[
                    s.m.body("env/" + n).id for n in ["cleanup_table", "cleanup_bin"] + CLUTTER
                ],
            )
        if request.task_family == "tabletop_cleanup":
            walk(
                "walk to cleanup station",
                [6.98, 7.1],
                -np.pi / 2,
                timeout=60,
                extra_ignore=[
                    s.m.body("env/" + n).id for n in ["cleanup_table", "cleanup_bin"] + CLUTTER
                ],
            )
        t._tick(1)
        s.use_grasp_point(1)
        h = s.wbc.ee_pos(s.d, s.hands[1]).copy()
        for phase, target in [
            ("retract hand", [h[0], 7.06, h[2]]),
            ("move hand outward", [6.65, 7.06, 0.8]),
            ("raise clear of table", [6.65, 7.06, 1.08]),
        ]:
            error = t._hand_to(1, target, timeout=5, speed=0.12)
            log(phase, {"ok": error < 0.06, "hand_error_m": error})
            if error > 0.06:
                raise RuntimeError(phase + " failed")
        for b in s._arm_bodies(s.hands[1]):
            for joint in range(s.m.body_jntadr[b], s.m.body_jntadr[b] + s.m.body_jntnum[b]):
                adr = s.m.jnt_qposadr[joint]
                s.wbc.qpos0[adr] = neutral_posture[adr]
        orientation = t.orient_hand(1, s.neutral_hand_orientation(1), speed=0.3)
        log("orient wrist for grasp", orientation)
        if not orientation["ok"]:
            raise RuntimeError("grasp orientation failed")
        log("ready for cleanup", {"ok": True})
        for name in CLUTTER:
            for attempt, closure in enumerate([0.65, 0.7], 1):
                result = t.grasp("env/" + name, hand=1, closure=closure)
                log("pick " + name, dict(attempt=attempt, **result))
                if result["ok"] or s.checks.fell_at is not None:
                    break
            if not result["ok"]:
                raise RuntimeError("grasp failed: " + name)
            result = t.release_into("env/" + name, DROP_POINT, BIN_BOUNDS)
            log("place " + name, result)
            if not result["ok"]:
                raise RuntimeError("placement failed: " + name)
        t._tick(2)
        positions = {n: s.d.xpos[s.m.body("env/" + n).id].tolist() for n in CLUTTER}
        inside = all(
            np.all(np.array(p) >= BIN_BOUNDS[0]) and np.all(np.array(p) <= BIN_BOUNDS[1])
            for p in positions.values()
        )
        success = bool(
            inside
            and abs(s.d.qpos[j]) < 0.1
            and s.checks.fell_at is None
            and s.checks.torque_limit_violations == 0
        )
        log(
            "task complete",
            {
                "ok": success,
                "objects": positions,
            },
        )
    except Exception as exc:  # noqa: BLE001 — retain failure evidence for every task error
        log("task stopped", {"ok": False, "reason": str(exc)})
    finally:
        s.step_observer = None
        recorder.close(s)
        np.savez(out / "final_state.npz", qpos=s.d.qpos, qvel=s.d.qvel)
        from .validation import validate_episode

        validation = validate_episode(s, out, request, stages)
        write_json(out / "validation.json", validation)

    return validation
