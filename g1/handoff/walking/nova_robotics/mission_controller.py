"""Scripted household mission for any uploaded MuJoCo humanoid or mobile robot.

This is a DEMONSTRATION controller, not a learned policy. The robot's floating base is
carried along the planned route by a mocap weld (no balance, no gait, no learned
walking), its joints are held in the source keyframe pose, and each mechanism is
worked through a real physical constraint: a weld between the robot's hand body and
the door leaf, drawer or prop, driven by moving the hand. Doors, drawers and props
therefore respond with their own mass, hinge friction and contacts; the environment
is exercised for real even though the robot's locomotion is scripted.

Everything it does is measured and reported: which mechanisms opened and how far,
which props were carried and where they came to rest, which steps were skipped and
why. The report is evidence about the environment, never a claim of robot capability.
"""

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

CONTROLLER_ID = "household_mission_v1"
HAND_TOKENS = ("hand", "wrist", "gripper", "palm", "ee", "tool")
LIFT_ABOVE_FLOOR = 0.0


def robot_geometry(model, data, robot_root):
    geoms = [g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == robot_root]
    base = data.xpos[robot_root]
    lo = min(data.geom_xpos[g][2] - model.geom_rbound[g] for g in geoms)
    hi = max(data.geom_xpos[g][2] + model.geom_rbound[g] for g in geoms)
    radius = max(
        float(np.linalg.norm(data.geom_xpos[g][:2] - base[:2]) + model.geom_rbound[g])
        for g in geoms
    )
    return {
        "radius": min(radius, 0.45),
        "height": float(hi - lo),
        "base_above_feet": float(base[2] - lo),
    }


def _find_hands(model, robot_root):
    """Rightmost/leftmost distal bodies named like hands; else the two bodies farthest
    from the base at roughly arm height. Returns [right, left] body ids."""
    names = {}
    for b in range(model.nbody):
        if model.body_rootid[b] != robot_root or b == robot_root:
            continue
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").lower()
        names[b] = name
    hands = {}
    finger = ("thumb", "index", "middle", "ring", "pinky", "finger")
    for side in ("right", "left"):
        best = None
        for b, name in names.items():
            if side not in name or not any(t in name for t in HAND_TOKENS):
                continue
            if any(f in name for f in finger):
                continue  # a fingertip is not the palm
            # Prefer the most distal non-finger link: the wrist/palm nearest the hand.
            depth = 0
            parent = b
            while parent > 0:
                parent = int(model.body_parentid[parent])
                depth += 1
            score = (-depth, name.count("_"))
            if best is None or score < best[0]:
                best = (score, b)
        if best is not None:
            hands[side] = best[1]
    if len(hands) == 2:
        return [hands["right"], hands["left"]]
    if len(hands) == 1:
        return [next(iter(hands.values()))] * 2
    leaves = [b for b in names if not any(model.body_parentid[c] == b for c in range(model.nbody))]
    leaves.sort(key=lambda b: -np.linalg.norm(model.body_pos[b]))
    return (leaves[:2] or [robot_root]) * 2


def prepare_spec(simulation_xml, plan):
    """Add the mocap carrier, the weld to the robot base and one inactive weld per
    hand for each object the plan touches. Returns the compiled model."""
    try:  # the package's meshes share names with every other Nova package
        cache = mujoco.mj_getCache()
        mujoco.mj_setCacheCapacity(cache, 0)
        mujoco.mj_clearCache(cache)
    except AttributeError:
        pass
    spec = mujoco.MjSpec.from_file(str(simulation_xml))
    robot_body = None
    for body in spec.bodies:
        if (body.name or "").startswith("nova_environment_") or body.name == "world":
            continue
        for joint in body.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                robot_body = body
                break
        if robot_body is not None:
            break
    if robot_body is None:
        raise ValueError("Mission controller needs a robot with a floating base")
    carrier = spec.worldbody.add_body(name="nova_mission_carrier", mocap=True)
    carrier.add_site(name="nova_mission_carrier_site", size=[0.01, 0, 0])
    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1="nova_mission_carrier",
        name2=robot_body.name,
        name="nova_mission_carry",
        active=False,
        # Soft enough that a door leaf or wall visibly resists the base instead of
        # being ploughed through; a real robot cannot walk through a closed door.
        solref=[0.08, 1],
    )
    hand_carrier = spec.worldbody.add_body(name="nova_mission_hand", mocap=True)
    hand_carrier.add_site(name="nova_mission_hand_site", size=[0.01, 0, 0])
    # Hinged and sliding parts are driven through a CONNECT (ball) constraint at the
    # handle so the part stays free to rotate about its own hinge; a weld would fight
    # the joint. Free props are welded so they do not spin in the hand.
    kinds = {}
    for step in plan["steps"]:
        if step.get("body"):
            kinds[step["body"]] = step["kind"]
    for target, kind in sorted(kinds.items()):
        spec.add_equality(
            type=mujoco.mjtEq.mjEQ_CONNECT
            if kind in ("hinge", "slide")
            else mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1="nova_mission_hand",
            name2=target,
            name=f"nova_mission_grasp_{target}",
            active=False,
            solref=[0.02, 1],
        )
    return spec.compile(), robot_body.name


def _quat_yaw(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class MissionController:
    def __init__(self, model, data, plan, robot_body_name):
        self.model, self.data, self.plan = model, data, plan
        self.robot = model.body(robot_body_name).id
        self.free_joint = next(
            j
            for j in range(model.njnt)
            if model.jnt_bodyid[j] == self.robot and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        )
        self.qa = int(model.jnt_qposadr[self.free_joint])
        self.hands = _find_hands(model, self.robot)
        self.carrier = model.body("nova_mission_carrier").mocapid[0]
        self.hand_carrier = model.body("nova_mission_hand").mocapid[0]
        self.carry_eq = model.equality("nova_mission_carry").id
        prefix = "nova_mission_grasp_"
        self.grasp_eq = {
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, e) or "")[len(prefix) :]: e
            for e in range(model.neq)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, e) or "").startswith(prefix)
        }
        if not self.grasp_eq and any(s.get("body") for s in plan["steps"]):
            raise ValueError("Mission model has no grasp welds for the planned objects")
        # Robot joints held at the source keyframe pose (or the initial pose). Only the
        # robot's own slice of the keyframe is used: the environment's free bodies and
        # mechanisms keep qpos0 (a keyframe from the robot file has zeros for them,
        # which would pile every prop at the world origin).
        hinge, slide = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
        robot_joints = [
            j
            for j in range(model.njnt)
            if int(model.body_rootid[model.jnt_bodyid[j]]) == int(self.robot)
            and int(model.jnt_type[j]) in (hinge, slide)
        ]
        self.joint_q = [int(model.jnt_qposadr[j]) for j in robot_joints]
        self.joint_v = [int(model.jnt_dofadr[j]) for j in robot_joints]
        mujoco.mj_resetData(model, data)
        if model.nkey:
            key = model.key_qpos[0]
            for q in self.joint_q:
                data.qpos[q] = key[q]
            data.qpos[self.qa : self.qa + 7] = key[self.qa : self.qa + 7]
        self.hold = data.qpos[self.joint_q].copy()
        self.base_offset = plan["robot"]["base_above_feet"]
        self.yaw = 0.0
        self.step_index = -1
        self.step_time = 0.0
        self.active_grasp = None
        self.records = []
        self.current = None
        self.contacts_max = 0
        self.failed = None
        self.begin_state = None
        start = (
            plan["steps"][0]["path"][0]
            if plan["steps"] and plan["steps"][0]["kind"] == "walk"
            else None
        )
        base = np.array(start if start else data.qpos[self.qa : self.qa + 3])
        if start:
            base[2] += self.base_offset
        self.position = base.astype(float)
        data.qpos[self.qa : self.qa + 3] = self.position
        data.qpos[self.qa + 3 : self.qa + 7] = _quat_yaw(self.yaw)
        data.qvel[:] = 0
        data.mocap_pos[self.carrier] = self.position
        data.mocap_quat[self.carrier] = _quat_yaw(self.yaw)
        # A body weld stores the relative pose captured at compile time (carrier at the
        # origin, robot at its authored spawn). The carrier must coincide with the base.
        model.eq_data[self.carry_eq, 0:3] = 0.0
        model.eq_data[self.carry_eq, 3:6] = 0.0
        model.eq_data[self.carry_eq, 6:10] = [1.0, 0.0, 0.0, 0.0]
        data.eq_active[self.carry_eq] = 1
        mujoco.mj_forward(model, data)

    # ----- helpers -----
    def _object_state(self, body_name):
        """Joint coordinate for a hinge/slide body, world position for a free one."""
        body = self.model.body(body_name)
        for j in range(self.model.njnt):
            if int(self.model.jnt_bodyid[j]) != int(body.id):
                continue
            kind = int(self.model.jnt_type[j])
            if kind in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                return float(self.data.qpos[self.model.jnt_qposadr[j]])
            break
        return [float(v) for v in self.data.xpos[body.id]]

    def _face(self, target_xy):
        dx, dy = target_xy[0] - self.position[0], target_xy[1] - self.position[1]
        if math.hypot(dx, dy) > 0.05:
            self.yaw = math.atan2(dy, dx)

    def _hand_to(self, point):
        self.data.mocap_pos[self.hand_carrier] = point
        self.data.mocap_quat[self.hand_carrier] = _quat_yaw(self.yaw)

    def _grasp(self, body_name, point):
        eq = self.grasp_eq.get(body_name)
        if eq is None:
            raise ValueError(f"No grasp weld prepared for {body_name}")
        # Place the carrier at the grasp point and set the weld's relative pose so the
        # object is held where it is now; no snap.
        self._hand_to(point)
        mujoco.mj_forward(self.model, self.data)
        body = self.model.body(body_name).id
        point = np.asarray(point, float)
        if int(self.model.eq_type[eq]) == int(mujoco.mjtEq.mjEQ_CONNECT):
            # Anchor at the carrier origin; matching anchor expressed in the part's frame.
            inv = np.empty(4)
            mujoco.mju_negQuat(inv, self.data.xquat[body])
            local = np.empty(3)
            mujoco.mju_rotVecQuat(local, point - self.data.xpos[body], inv)
            self.model.eq_data[eq, 0:3] = 0.0
            self.model.eq_data[eq, 3:6] = local
        else:
            rel = self.data.xpos[body] - point
            quat = np.empty(4)
            mujoco.mju_negQuat(quat, self.data.mocap_quat[self.hand_carrier])
            local = np.empty(3)
            mujoco.mju_rotVecQuat(local, rel, quat)
            bq = np.empty(4)
            mujoco.mju_mulQuat(bq, quat, self.data.xquat[body])
            self.model.eq_data[eq, 0:3] = 0.0  # anchor at the carrier origin
            self.model.eq_data[eq, 3:6] = local
            self.model.eq_data[eq, 6:10] = bq
        self.data.eq_active[eq] = 1
        self.active_grasp = eq
        return True

    def _release(self):
        if self.active_grasp is not None:
            self.data.eq_active[self.active_grasp] = 0
            self.active_grasp = None

    # ----- stepping -----
    def _begin(self, step):
        self.current = step
        self.step_time = 0.0
        self.begin_state = self._object_state(step["body"]) if step.get("body") else None
        kind = step["kind"]
        if kind == "walk":
            self.path = np.asarray(step["path"], float)
            self.path[:, 2] += self.base_offset
            self.cum = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
            )
        elif kind == "hinge":
            origin, axis = np.asarray(step["origin"]), np.asarray(step["axis"])
            self.handle = np.asarray(step["handle"])
            state = self._object_state(step["body"])
            self.start_angle = state
            self.goal_angle = step["angle"] if step["target"] == "open" else 0.0
            self._face(self.handle)
            # The hand starts where the handle is NOW (leaf may be part open).
            current = self._rotate(self.handle, origin, axis, state)
            self._grasp(step["body"], current)
        elif kind == "slide":
            self.pull = np.asarray(step["pull"])
            axis = np.asarray(step["axis"])
            state = self._object_state(step["body"])
            self.start_travel = state
            self.goal_travel = step["travel"] if step["target"] == "open" else 0.0
            self._face(self.pull)
            self._grasp(step["body"], self.pull + axis * state)
        elif kind == "pick":
            grasp = np.asarray(self._object_state(step["body"]))
            self._face(grasp)
            self._grasp(step["body"], grasp)
            self.pick_from = grasp.copy()
            self.pick_to = grasp + np.array([0, 0, step["lift"]])
        elif kind == "place":
            target = np.asarray(step["target"])
            self._face(target)
            held = np.asarray(self._object_state(step["body"]))
            self.place_from = held.copy()
            self.place_to = np.array([target[0], target[1], target[2] + 0.02])

    @staticmethod
    def _rotate(point, origin, axis, angle):
        axis = axis / np.linalg.norm(axis)
        rel = point - origin
        c, s = math.cos(angle), math.sin(angle)
        return origin + rel * c + np.cross(axis, rel) * s + axis * (axis @ rel) * (1 - c)

    def _finish(self, step):
        end = self._object_state(step["body"]) if step.get("body") else None
        record = {
            "label": step["label"],
            "kind": step["kind"],
            "object": step.get("object"),
            "start_state": self.begin_state,
            "end_state": end,
            "duration_s": round(self.step_time, 2),
        }
        if step["kind"] == "hinge":
            moved = abs(end - self.begin_state)
            wanted = abs(self.goal_angle - self.start_angle)
            record["fraction"] = round(moved / wanted, 3) if wanted > 1e-6 else 1.0
            record["outcome"] = "done" if record["fraction"] > 0.6 else "blocked"
        elif step["kind"] == "slide":
            moved = abs(end - self.begin_state)
            wanted = abs(self.goal_travel - self.start_travel)
            record["fraction"] = round(moved / wanted, 3) if wanted > 1e-6 else 1.0
            record["outcome"] = "done" if record["fraction"] > 0.6 else "blocked"
        elif step["kind"] == "pick":
            lifted = end[2] - self.begin_state[2]
            record["outcome"] = "done" if lifted > 0.5 * step["lift"] else "blocked"
        elif step["kind"] == "place":
            target = np.asarray(step["target"])
            record["distance_to_target_m"] = round(
                float(np.linalg.norm(np.asarray(end) - target)), 3
            )
            record["outcome"] = "done" if record["distance_to_target_m"] < 0.35 else "missed"
        else:
            record["outcome"] = "done"
        self.records.append(record)
        if step["kind"] in ("hinge", "slide", "place"):
            self._release()
        if step["kind"] == "walk" and self.active_grasp is not None:
            pass  # carrying a prop: keep holding it through the walk

    def before_step(self):
        dt = self.model.opt.timestep
        steps = self.plan["steps"]
        if self.current is None or self.step_time >= self.current["duration"]:
            if self.current is not None:
                self._finish(self.current)
            self.step_index += 1
            if self.step_index >= len(steps):
                self.current = None
                self.data.mocap_pos[self.carrier] = self.position
                return False
            self._begin(steps[self.step_index])
        step = self.current
        u = min(1.0, self.step_time / step["duration"])
        smooth = 0.5 - 0.5 * math.cos(math.pi * u)
        if step["kind"] == "walk":
            s = smooth * self.cum[-1]
            k = int(np.searchsorted(self.cum, s, side="right") - 1)
            k = max(0, min(k, len(self.path) - 2))
            seg = self.cum[k + 1] - self.cum[k]
            t = (s - self.cum[k]) / seg if seg > 1e-9 else 0.0
            target = self.path[k] * (1 - t) + self.path[k + 1] * t
            self._face(self.path[min(k + 1, len(self.path) - 1)])
            self.position = target
            if self.active_grasp is not None:
                # Carry the held prop in front of the chest.
                fwd = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
                self._hand_to(self.position + 0.45 * fwd + np.array([0, 0, 0.1]))
        elif step["kind"] == "hinge":
            angle = self.start_angle + smooth * (self.goal_angle - self.start_angle)
            point = self._rotate(
                self.handle, np.asarray(step["origin"]), np.asarray(step["axis"]), angle
            )
            self._hand_to(point)
        elif step["kind"] == "slide":
            travel = self.start_travel + smooth * (self.goal_travel - self.start_travel)
            self._hand_to(self.pull + np.asarray(step["axis"]) * travel)
        elif step["kind"] == "pick":
            self._hand_to(self.pick_from * (1 - smooth) + self.pick_to * smooth)
        elif step["kind"] == "place":
            self._hand_to(self.place_from * (1 - smooth) + self.place_to * smooth)
        self.data.mocap_pos[self.carrier] = self.position
        self.data.mocap_quat[self.carrier] = _quat_yaw(self.yaw)
        # Hold the robot's joints in the keyframe pose (stiff PD, no gait).
        q = self.data.qpos[self.joint_q]
        v = self.data.qvel[self.joint_v]
        self.data.qfrc_applied[self.joint_v] = 200.0 * (self.hold - q) - 5.0 * v
        self.step_time += dt
        return True

    def after_step(self):
        self.contacts_max = max(self.contacts_max, int(self.data.ncon))
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise ValueError("Mission simulation produced non-finite state")
        if np.any(self.data.warning.number):
            raise ValueError("Mission simulation emitted numerical warnings")

    def report(self):
        if self.current is not None:
            self._finish(self.current)
            if self.step_time < self.current["duration"]:
                self.records[-1]["outcome"] = "incomplete"
            self.current = None
        done = sum(r["outcome"] == "done" for r in self.records)
        # Failed attempted actions are failures; skipped/unattempted work cannot
        # turn a successful subset into a completed mission.
        if any(record["outcome"] not in {"done", "incomplete"} for record in self.records):
            status = "failed"
        elif not self.records:
            status = "not_run"
        elif done != len(self.plan["steps"]) or self.plan.get("skipped"):
            status = "partial"
        else:
            status = "passed"
        return {
            "status": status,
            "engine": f"MuJoCo {mujoco.__version__}",
            "mode": "scripted_household_mission",
            "controller": CONTROLLER_ID,
            "duration_s": round(float(self.data.time), 3),
            "steps_planned": len(self.plan["steps"]),
            "steps_completed": done,
            "steps": self.records,
            "skipped": self.plan.get("skipped", []),
            "maximum_contacts": self.contacts_max,
            "locomotion": "scripted base carrier (mocap weld); no balance, gait or learned "
            "walking; joints held at the source keyframe pose",
            "manipulation": "hand-to-object weld driven along the mechanism's own axis; the "
            "door, drawer or prop responds with its real mass, friction and contacts",
            "training": "not_run",
            "scope": "Environment interactability demonstration. Not a robot capability, "
            "task-success or sim-to-real claim.",
        }


def run_mission(simulation_xml, plan, record_every=10, viewer=None):
    model, robot_body = prepare_spec(simulation_xml, plan)
    data = mujoco.MjData(model)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    controller = MissionController(model, data, plan, robot_body)
    times, tracks = [0.0], {}
    return model, data, controller, times, tracks


def main():
    parser = argparse.ArgumentParser(description="Scripted household mission demonstration")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = json.loads((root / "mission_plan.json").read_text())
    model, robot_body = prepare_spec(root / "simulation.xml", plan)
    data = mujoco.MjData(model)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    ctrl = MissionController(model, data, plan, robot_body)
    viewer = None
    try:
        if not args.headless:
            from mujoco import viewer as gui

            viewer = gui.launch_passive(model, data)
        running = True
        while running:
            begin = time.monotonic()
            running = ctrl.before_step()
            mujoco.mj_step(model, data)
            ctrl.after_step()
            if viewer:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.monotonic() - begin)))
        print(json.dumps(ctrl.report(), indent=2))
    finally:
        if viewer:
            viewer.close()


if __name__ == "__main__":
    main()
