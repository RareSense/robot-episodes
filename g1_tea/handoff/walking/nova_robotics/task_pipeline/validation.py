"""Measured success and physics checks, independent of the runner's success flag."""

from __future__ import annotations

from pathlib import Path

import h5py
import mujoco
import numpy as np

from .fixture import CLUTTER
from .spec import sha256, write_json


def box_inside(model, data, body_name, bounds):
    body = model.body(body_name).id
    geoms = range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body])
    for geom in geoms:
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Café object containment requires box geometry")
        extent = np.abs(data.geom_xmat[geom].reshape(3, 3)) @ model.geom_size[geom]
        if np.any(data.geom_xpos[geom] - extent < np.asarray(bounds[0]) - 0.003):
            return False
        if np.any(data.geom_xpos[geom] + extent > np.asarray(bounds[1]) + 0.003):
            return False
    return True


def bin_interior_bounds(model, data):
    """Use the actual tray floor and rim, not the old center-placement target bounds."""
    floor = model.geom("env/cleanup_bin_floor").id
    center = data.geom_xpos[floor]
    extent = np.abs(data.geom_xmat[floor].reshape(3, 3)) @ model.geom_size[floor]
    body = model.body("env/cleanup_bin").id
    rim = max(
        data.geom_xpos[g, 2] + model.geom_size[g, 2]
        for g in range(
            model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body]
        )
    )
    return [
        [center[0] - extent[0], center[1] - extent[1], center[2] + extent[2]],
        [center[0] + extent[0], center[1] + extent[1], rim],
    ]


def validate_episode(session, out, request, stages):
    m = session.m
    d = mujoco.MjData(m)
    objects = {name: {"stable_in_bin": True, "max_speed_m_s": 0.0} for name in CLUTTER}
    finite, commands_in_limits, stable_door = True, True, True
    with h5py.File(out / "trace.h5", "r") as f:
        steps = len(f["time"])
        for start in range(0, steps, 2048):
            for key in ("qpos", "qvel", "ctrl", "applied"):
                finite &= bool(np.isfinite(f[key][start : start + 2048]).all())
            commands = f["ctrl"][start : start + 2048]
            commands_in_limits &= bool(
                np.all(commands >= m.actuator_ctrlrange[:, 0] - 1e-8)
                and np.all(commands <= m.actuator_ctrlrange[:, 1] + 1e-8)
            )
        dwell_steps = round(1.0 / m.opt.timestep)
        for index in range(max(0, steps - dwell_steps), steps, 20):
            d.qpos[:] = f["qpos"][index]
            d.qvel[:] = f["qvel"][index]
            mujoco.mj_kinematics(m, d)
            stable_door &= abs(d.qpos[m.joint("env/object_0074_joint").qposadr[0]]) < 0.1
            for name, status in objects.items():
                status["stable_in_bin"] &= box_inside(
                    m, d, "env/" + name, bin_interior_bounds(m, d)
                )
                adr = m.joint("env/" + name + "_free").dofadr[0]
                status["max_speed_m_s"] = max(
                    status["max_speed_m_s"], float(np.linalg.norm(d.qvel[adr : adr + 3]))
                )
        final_door = float(f["final_qpos"][m.joint("env/object_0074_joint").qposadr[0]])
        interval_count = int(f.attrs["complete_intervals"])
    for name, status in objects.items():
        picks = [
            s["result"] for s in stages if s["stage"] == "pick " + name and s["result"].get("ok")
        ]
        status["verified_grasp"] = bool(picks)
        status["grasp_evidence"] = picks[-1] if picks else None
        status["stable_in_bin"] &= status["max_speed_m_s"] < 0.03
    door_open = any(s["stage"] == "door opened" and s["result"].get("ok") for s in stages)
    door_ok = bool(stable_door and (door_open or request.task_family == "tabletop_cleanup"))
    task_success = bool(
        steps >= dwell_steps
        and door_ok
        and all(v["stable_in_bin"] and v["verified_grasp"] for v in objects.values())
        and any(s["stage"] == "task complete" and s["result"].get("ok") for s in stages)
    )
    checks = session.checks.as_dict()
    physics_ok = bool(
        finite
        and commands_in_limits
        and checks["fell_at_s"] is None
        and checks["torque_limit_violations"] == 0
        and checks["max_ground_penetration_mm"] <= 20
        and checks["joint_overshoot_mm_or_mrad"] <= 10
        and m.neq == 0
    )
    source_ok = sha256(request.scene) == request.source_sha256
    return dict(
        schema="nova.task-validation/v1",
        task_success=task_success,
        simulation_valid=physics_ok and source_ok,
        export_compatible=False,
        learning_evaluated=False,
        usable=task_success and physics_ok and source_ok,
        source_unchanged=source_ok,
        physics_steps=steps,
        simulation_seconds=steps * float(m.opt.timestep),
        observation_intervals=interval_count,
        final_door_angle_rad=final_door,
        door_opened=door_open,
        door_stable_closed=bool(stable_door),
        objects=objects,
        checks=checks,
        physics_profile="cafe_demo_v1",
        thresholds={
            "penetration_mm": 20,
            "joint_overshoot_mm_or_mrad": 10,
            "placement_dwell_s": 1,
            "object_speed_m_s": 0.03,
        },
        limitations=[
            "Simulation-only data; learning and hardware transfer not evaluated",
            "Scene-specific door/navigation adapter; staged small blocks only",
            "Configured NoSlip solver; no default-solver robustness claim",
        ],
    )


def replay(out, *, windows=12):
    """Replay sampled command windows from recorded states, including solver warmstarts."""
    out = Path(out)
    m = mujoco.MjModel.from_binary_path(str(out / "scene.mjb"))
    d = mujoco.MjData(m)
    max_error = 0.0
    with h5py.File(out / "trace.h5", "r") as f:
        period = int(f.attrs["commands_per_observation"])
        count = len(f["ctrl"])
        if count <= period:
            result = dict(passed=False, reason="Insufficient trace for replay")
        else:
            for start in np.linspace(
                0, count - period - 1, min(windows, count // period), dtype=int
            ):
                mujoco.mj_setState(
                    m, d, f["initial_integration_state"][:], int(f.attrs["initial_state_kind"])
                )
                d.time = f["time"][start]
                d.qpos[:] = f["qpos"][start]
                d.qvel[:] = f["qvel"][start]
                d.qacc_warmstart[:] = f["warmstart"][start]
                for ctrl in f["ctrl"][start : start + period]:
                    mujoco.mj_step1(m, d)
                    d.ctrl[:] = ctrl
                    mujoco.mj_step2(m, d)
                error = float(np.max(np.abs(d.qpos - f["qpos"][start + period])))
                max_error = max(max_error, error)
            result = dict(
                passed=max_error < 1e-5,
                windows=min(windows, count // period),
                horizon_seconds=period * float(m.opt.timestep),
                max_qpos_error=max_error,
                tolerance=1e-5,
                method="Every recorded 2 ms motor command; reset at each sampled window",
                continuous_replay=False,
            )
    write_json(out / "replay_validation.json", result)
    return result
