"""Trusted, isolated MuJoCo worker. JSON commands only; no user code is imported."""

import json
import math
import signal
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from nova_robotics.experiments import EXPERIMENTS, validate_program
from nova_robotics.nova_water import MuJoCoWater
from nova_robotics.overlay import PROPERTIES, SCENE, SCENE_PROPERTIES
from nova_robotics.uploads import read_xml

# Pack-owned drive policy: force-limited position control, never teleport a body.
JOINT_DRIVE = {
    "response_hz": 1.3,
    "slide_speed_m_s": 0.15,
    "hinge_speed_rad_s": 0.75,
    "slide_force_n": 100.0,
    "hinge_torque_nm": 60.0,
}

Y_UP = trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])


class LiveWorld:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.scene = json.loads((self.directory / "scene_spec.json").read_text())
        self.has_robot = bool(self.scene.get("robot"))
        source = self.directory / ("simulation.xml" if self.has_robot else "environment.xml")
        root = read_xml(source.read_bytes())
        if root.tag != "mujoco" or any(
            n.tag in {"include", "extension", "plugin"} for n in root.iter()
        ):
            raise ValueError("Live mode needs a flattened plugin-free MJCF export")
        for node in root.iter():
            for key in ("file", "meshdir", "texturedir", "assetdir"):
                if key in node.attrib:
                    path = (self.directory / node.attrib[key]).resolve()
                    if not path.is_relative_to(self.directory.resolve()):
                        raise ValueError("Live model references an asset outside the package")
        # This file runs as a standalone script: no relative imports. The process-wide
        # asset cache is keyed by mesh path, which every Nova package shares.
        try:
            cache = mujoco.mj_getCache()
            mujoco.mj_setCacheCapacity(cache, 0)
            mujoco.mj_clearCache(cache)
        except AttributeError:
            pass
        self.model = mujoco.MjModel.from_xml_path(str(source))
        m = self.model
        volumes = self.scene.get("world", {}).get("fields", {}).get("water", [])
        self.water = MuJoCoWater(volumes, float(-m.opt.gravity[2])) if volumes else None
        if m.nbody > 512 or m.ngeom > 8000 or m.nu > 64 or m.nv > 1024:
            raise ValueError("Model exceeds live simulation budget")
        if not np.isfinite(m.opt.timestep) or not 0.0005 <= m.opt.timestep <= 0.01:
            raise ValueError("Live timestep must be between 0.5 and 10 ms")
        self.data = mujoco.MjData(m)
        self.disabled = int(m.opt.disableflags)
        self.actuators = {}
        for i in range(m.nu):
            lo, hi = m.actuator_ctrlrange[i]
            if (
                m.actuator_ctrllimited[i]
                and np.isfinite([lo, hi]).all()
                and -10000 <= lo < hi <= 10000
            ):
                self.actuators[f"actuator:{i}"] = {
                    "id": f"actuator:{i}",
                    "name": m.actuator(i).name or f"Actuator {i}",
                    "min": float(lo),
                    "max": float(hi),
                    "value": float(np.clip(0, lo, hi)),
                }
        # Enabling global actuation must not silently enable unbounded other motors.
        if len(self.actuators) != m.nu:
            self.actuators.clear()
        self.targets = []
        for j in range(m.njnt):
            kind = m.jnt_type[j]
            name = m.joint(j).name or m.body(m.jnt_bodyid[j]).name or f"joint:{j}"
            environment = name.startswith("nova_environment_") or not self.has_robot
            requirement = (
                "free"
                if kind == mujoco.mjtJoint.mjJNT_FREE
                else (
                    "hinge"
                    if environment and kind == mujoco.mjtJoint.mjJNT_HINGE
                    else "slide"
                    if environment and kind == mujoco.mjtJoint.mjJNT_SLIDE
                    else None
                )
            )
            if requirement:
                self.targets.append({"id": str(j), "name": name, "kind": requirement})
        self.track_bodies = []
        mujoco.mj_forward(m, self.data)
        prefix = "nova_environment_" if self.has_robot else ""
        for i, obj in enumerate(self.scene["objects"]):
            if obj.get("body"):
                bid = m.body(f"{prefix}object_{i:04d}_body").id
                self.track_bodies.append((obj["name"], bid, np.linalg.inv(self.body_pose(bid))))
                for target in self.targets:
                    if m.jnt_bodyid[int(target["id"])] == bid:
                        target["name"] = f"{obj['name']} ({target['kind']})"
        self.joint_controls = []
        for name, bid, _ in self.track_bodies:
            for target in self.targets:
                j = int(target["id"])
                if m.jnt_bodyid[j] != bid or target["kind"] not in {"hinge", "slide"}:
                    continue
                lo, hi = m.jnt_range[j]
                if m.jnt_limited[j] and np.isfinite([lo, hi]).all() and lo < hi:
                    self.joint_controls.append(
                        {
                            "id": target["id"],
                            "node_name": name,
                            "kind": target["kind"],
                            "min": float(lo),
                            "max": float(hi),
                        }
                    )
        self.robot_geoms = [
            i
            for i in range(m.ngeom)
            if self.has_robot and not (m.geom(i).name or "").startswith("nova_environment_")
        ]
        # Live property changes: a creator's override tried on the running model
        # without a rebuild. The same table the build applies; nothing else moves.
        self.live_objects = {}
        for i, obj in enumerate(self.scene["objects"]):
            entry = {"geoms": [], "body": None, "dof": None}
            for g in range(m.ngeom):
                gname = m.geom(g).name or ""
                if gname.startswith(f"{prefix}object_{i:04d}") and (
                    m.geom_contype[g] or m.geom_conaffinity[g]
                ):
                    entry["geoms"].append(g)
            if obj.get("body"):
                bid = m.body(f"{prefix}object_{i:04d}_body").id
                entry["body"] = bid
                for j in range(m.njnt):
                    if int(m.jnt_bodyid[j]) == bid and int(m.jnt_type[j]) in (
                        int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
                    ):
                        entry["dof"] = int(m.jnt_dofadr[j])
                        entry["joint"] = j
            self.live_objects[obj["name"]] = entry
        self.reset({"experiment": "passive"})

    def set_properties(self, request):
        """Apply live-adjustable overrides to the running model, then re-derive the
        model constants. Values outside the table's range or on a part that does not
        have the property are refused, exactly as the build refuses them."""
        m = self.model
        overrides = request.get("overrides")
        if not isinstance(overrides, dict) or len(overrides) > 512:
            raise ValueError("overrides must map object names to property values")
        for name, properties in overrides.items():
            if name == SCENE:
                for key, value in properties.items():
                    spec = SCENE_PROPERTIES.get(key)
                    number = type(value) in (int, float) and np.isfinite(value)
                    if spec is None or not number or not spec.minimum <= value <= spec.maximum:
                        raise ValueError(f"{name}.{key}: value outside its range")
                    m.opt.gravity[2] = -float(value)
                    if self.water is not None:
                        self.water.gravity = float(value)
                continue
            entry = self.live_objects.get(name)
            if entry is None or not isinstance(properties, dict):
                raise ValueError(f"Unknown object for a live override: {name}")
            for key, value in properties.items():
                spec = PROPERTIES.get(key)
                if spec is None or not spec.live:
                    raise ValueError(f"{name}.{key}: not adjustable in a running session")
                if type(value) not in (int, float) or not np.isfinite(value) or not (
                    spec.minimum <= value <= spec.maximum
                ):
                    raise ValueError(f"{name}.{key}: value outside its range")
                if spec.owner == "object":
                    column = {"friction": 0, "torsional_friction": 1, "rolling_friction": 2}[key]
                    for g in entry["geoms"]:
                        m.geom_friction[g][column] = value
                elif entry["body"] is None:
                    raise ValueError(f"{name}.{key}: this part is static")
                elif key == "mass":
                    ratio = value / max(float(m.body_mass[entry["body"]]), 1e-9)
                    m.body_mass[entry["body"]] = value
                    m.body_inertia[entry["body"]] *= ratio
                elif entry["dof"] is None:
                    raise ValueError(f"{name}.{key}: a loose object has no joint")
                elif key == "damping":
                    m.dof_damping[entry["dof"]] = value
                elif key == "frictionloss":
                    m.dof_frictionloss[entry["dof"]] = value
                else:
                    low, high = m.jnt_range[entry["joint"]]
                    m.jnt_range[entry["joint"]] = (
                        (value, high) if key == "limit_min" else (low, value)
                    )
        mujoco.mj_setConst(m, self.data)
        mujoco.mj_forward(m, self.data)
        return self.frame()

    def body_pose(self, i):
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[i].reshape(3, 3)
        pose[:3, 3] = self.data.xpos[i]
        return pose

    def controls(self, values):
        if not isinstance(values, dict) or len(values) > 64:
            raise ValueError("Expected at most 64 actuator commands")
        for key, value in values.items():
            a = self.actuators.get(key)
            if a is None or type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError("Unknown/unbounded actuator or invalid command")
            if not a["min"] <= value <= a["max"]:
                raise ValueError(f"Command outside model bounds for {key}")
            self.data.ctrl[int(key.split(":")[1])] = value

    def reset(self, request):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for key, a in self.actuators.items():
            d.ctrl[int(key.split(":")[1])] = a["value"]
        m.opt.disableflags = self.disabled | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        self.drives = {}
        self.experiment = request.get("experiment", "passive")
        self.target = request.get("target")
        self.program = request.get("program")
        self.program_index = 0
        self.builtin = None
        self.magnitude = request.get("magnitude", 1.0)
        if (
            type(self.magnitude) not in (int, float)
            or not np.isfinite(self.magnitude)
            or not 0 <= self.magnitude <= 100
        ):
            raise ValueError("Experiment magnitude must be between 0 and 100")
        if self.experiment == "builtin":
            from nova_robotics.go2_controller import Go2Controller

            self.builtin = Go2Controller(m, d, self.directory / "controller/policy.onnx")
        elif self.experiment == "program":
            if not self.actuators:
                raise ValueError("All native actuators must declare finite control bounds")
            self.program = validate_program(self.program)
            for f in self.program["frames"]:
                self.controls(f["controls"])
            for key, a in self.actuators.items():
                d.ctrl[int(key.split(":")[1])] = a["value"]
            m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        elif self.experiment == "manual":
            if not self.actuators:
                raise ValueError(
                    "Manual control requires all native actuators to have finite bounds"
                )
            m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        else:
            choice = next((x for x in EXPERIMENTS if x["id"] == self.experiment), None)
            if choice is None:
                raise ValueError("Unknown experiment")
            requirement = choice["requires"]
            if requirement == "actuator":
                if self.target not in self.actuators:
                    raise ValueError("Select a bounded native actuator")
                m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            elif requirement != "none":
                target = next(
                    (
                        t
                        for t in self.targets
                        if t["id"] == self.target and t["kind"] == requirement
                    ),
                    None,
                )
                if target is None:
                    raise ValueError(f"Select a {requirement} environment/robot target")
                j = int(self.target)
                if self.experiment == "drop":
                    if self.magnitude > 2:
                        raise ValueError("Drop offset is limited to 2 metres")
                    d.qpos[m.jnt_qposadr[j] + 2] += self.magnitude
                elif self.experiment == "slide":
                    if self.magnitude > 2:
                        raise ValueError("Initial sliding speed is limited to 2 m/s")
                    d.qvel[m.jnt_dofadr[j]] = self.magnitude
        mujoco.mj_forward(m, d)
        return self.frame()

    def joint_target(self, request):
        control = next((c for c in self.joint_controls if c["id"] == request.get("target")), None)
        value = request.get("position")
        if (
            control is None
            or type(value) not in (int, float)
            or not np.isfinite(value)
            or not control["min"] <= value <= control["max"]
        ):
            raise ValueError("Select a supported joint and a finite target within its limits")
        if self.experiment != "joint_position":
            self.drives = {}
        self.experiment = "joint_position"
        self.builtin = None
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        j = int(control["id"])
        drive = self.drives.setdefault(
            j, {"position": float(self.data.qpos[self.model.jnt_qposadr[j]])}
        )
        drive["target"] = float(value)
        return self.frame()

    def _drive_joints(self):
        m, d = self.model, self.data
        omega = 2 * math.pi * JOINT_DRIVE["response_hz"]
        for j, drive in self.drives.items():
            slide = m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
            speed = JOINT_DRIVE["slide_speed_m_s" if slide else "hinge_speed_rad_s"]
            cap = JOINT_DRIVE["slide_force_n" if slide else "hinge_torque_nm"]
            drive["position"] += float(
                np.clip(
                    drive["target"] - drive["position"],
                    -speed * m.opt.timestep,
                    speed * m.opt.timestep,
                )
            )
            dof = m.jnt_dofadr[j]
            inertia = max(float(m.dof_M0[dof]), 1e-6)
            error = drive["position"] - d.qpos[m.jnt_qposadr[j]]
            # Contacts remain authoritative even when the requested pose is unreachable.
            force = inertia * (omega * omega * error - 2 * omega * d.qvel[dof]) + d.qfrc_bias[dof]
            d.qfrc_applied[dof] = np.clip(force, -cap, cap)

    def step(self, request):
        m, d = self.model, self.data
        if self.experiment == "manual":
            self.controls(request.get("controls", {}))
        steps = request.get("steps", round(0.05 / m.opt.timestep))
        if type(steps) is not int or not 0 <= steps <= 200:
            raise ValueError("Step count must be 0–200")
        if d.time >= 30:
            return self.frame()
        for _ in range(steps):
            if d.time >= 30:
                break
            d.qfrc_applied[:] = 0
            d.xfrc_applied[:] = 0
            if self.water is not None:
                self.water.apply(m, d)
            if self.experiment == "joint_position":
                self._drive_joints()
            elif self.builtin:
                self.builtin.before_step(m, d)
            elif self.experiment == "program":
                frames = self.program["frames"]
                while (
                    self.program_index < len(frames)
                    and frames[self.program_index]["time_s"] <= d.time
                ):
                    self.controls(frames[self.program_index]["controls"])
                    self.program_index += 1
                if d.time > frames[-1]["time_s"]:
                    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            elif self.experiment in {"actuator_step", "actuator_sweep", "release"}:
                a = self.actuators[self.target]
                fraction = 0.1 * (
                    math.sin(2 * math.pi * d.time) if self.experiment == "actuator_sweep" else 1
                )
                self.controls(
                    {
                        self.target: float(
                            np.clip(
                                a["value"] + fraction * (a["max"] - a["min"]), a["min"], a["max"]
                            )
                        )
                    }
                )
                if self.experiment == "release" and d.time >= 0.5:
                    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            elif self.experiment in {"hinge", "slider"}:
                d.qfrc_applied[m.jnt_dofadr[int(self.target)]] = self.magnitude
            elif self.experiment in {"push", "load"}:
                body = m.jnt_bodyid[int(self.target)]
                if self.experiment == "load":
                    d.xfrc_applied[body, 2] = -self.magnitude
                elif d.time < 0.3:
                    d.xfrc_applied[body, 0] = self.magnitude
            mujoco.mj_step(m, d)
            if self.builtin:
                self.builtin.after_step(m, d)
            if (
                not np.isfinite(d.qpos).all()
                or not np.isfinite(d.qvel).all()
                or d.warning.number.any()
            ):
                raise ValueError(
                    "Simulation became unstable; stopped. Check source physics and commands."
                )
        mujoco.mj_forward(m, d)
        return self.frame()

    def frame(self):
        d = self.data
        poses = {}
        for i in self.robot_geoms:
            pose = np.eye(4)
            pose[:3, :3] = d.geom_xmat[i].reshape(3, 3)
            pose[:3, 3] = d.geom_xpos[i]
            poses[f"nova_robot_geom_{i}"] = Y_UP @ pose
        for name, body, inverse in self.track_bodies:
            # Older immutable packages stored world-space mesh vertices; new ones
            # store body-local, Y-up vertices and a real physical pivot.
            poses[name] = (
                Y_UP
                @ self.body_pose(body)
                @ (Y_UP.T if self.scene.get("preview_frame") == "body_origin_y_up/v1" else inverse)
            )
        result = {}
        for name, matrix in poses.items():
            q = trimesh.transformations.quaternion_from_matrix(matrix)
            result[name] = {"position": matrix[:3, 3].tolist(), "rotation": np.roll(q, -1).tolist()}
        return {
            "time_s": float(d.time),
            "contacts": int(d.ncon),
            "joints": {
                c["id"]: {
                    "position": float(d.qpos[self.model.jnt_qposadr[int(c["id"])]]),
                    "target": self.drives.get(int(c["id"]), {}).get("target"),
                }
                for c in self.joint_controls
            },
            "poses": result,
            "experiment": self.experiment,
            "training": "not_run",
        }

    def info(self):
        return {
            "timestep_s": float(self.model.opt.timestep),
            "duration_limit_s": 30,
            "actuators": list(self.actuators.values()),
            "targets": self.targets,
            "joint_controls": self.joint_controls,
            "experiments": EXPERIMENTS,
            "builtin_controller": (self.directory / "controller/policy.onnx").is_file(),
            "live_properties": sorted(key for key, spec in PROPERTIES.items() if spec.live),
            "water": self.water is not None,
            "frame": self.frame(),
        }


def main():
    signal.alarm(180)
    world = LiveWorld(".")
    print(json.dumps(world.info()), flush=True)
    for line in sys.stdin:
        try:
            if len(line) > 128 * 1024:
                raise ValueError("Command too large")
            request = json.loads(line)
            if request.get("op") == "reset":
                result = world.reset(request)
            elif request.get("op") == "joint_target":
                result = world.joint_target(request)
            elif request.get("op") == "set_properties":
                result = world.set_properties(request)
            elif request.get("op") == "step":
                result = world.step(request)
            else:
                raise ValueError("Unknown command")
            print(json.dumps(result, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)
            return


if __name__ == "__main__":
    main()
