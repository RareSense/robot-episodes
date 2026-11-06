"""Bounded-memory, pre-action recording of every actual physics command."""

from __future__ import annotations

import json

import h5py
import mujoco
import numpy as np


class TraceRecorder:
    def __init__(self, session, path, *, max_sim_seconds=300, policy_hz=25):
        m, d = session.m, session.d
        if m.nmocap or m.neq or np.any(m.body_gravcomp):
            raise ValueError("This recorder profile requires an unassisted motor-only model")
        period = 1 / (policy_hz * m.opt.timestep)
        if abs(period - round(period)) > 1e-8:
            raise ValueError("Policy period must be an integer number of physics steps")
        self.period = round(period)
        self.start = float(d.time)
        self.max_sim_seconds = max_sim_seconds
        self.count = 0
        self.buffer = []
        self.file = h5py.File(path, "x")
        self.file.attrs.update(
            schema="nova.physics-trace/v1",
            timestep=float(m.opt.timestep),
            mujoco_version=mujoco.__version__,
            policy_hz=policy_hz,
            commands_per_observation=self.period,
            action_adapter="g1_motor_chunk_500hz_v1",
            time_convention="observation(t), then commands[t:t+period], then next observation",
            noslip_iterations=int(m.opt.noslip_iterations),
            integrator=int(m.opt.integrator),
            solver=int(m.opt.solver),
            iterations=int(m.opt.iterations),
            sensor_note="Ideal simulated joint encoders; audit sensors are post-solver values",
        )
        state_kind = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(m, state_kind))
        mujoco.mj_getState(m, d, state, state_kind)
        self.file.create_dataset("initial_integration_state", data=state)
        self.file.attrs["initial_state_kind"] = int(state_kind)
        self.file.create_dataset("motor_qpos_indices", data=session.motors.qadr)
        self.file.create_dataset("motor_qvel_indices", data=session.motors.dof)
        self.file.create_dataset("motor_limits", data=session.motors.limit)
        self.file.attrs["motor_names"] = json.dumps([m.actuator(i).name for i in range(m.nu)])
        self.file.attrs["sensor_names"] = json.dumps([m.sensor(i).name for i in range(m.nsensor)])
        sizes = {
            "time": (),
            "qpos": (m.nq,),
            "qvel": (m.nv,),
            "ctrl": (m.nu,),
            "applied": (m.nu,),
            "warmstart": (m.nv,),
            "sensors_after": (m.nsensordata,),
        }
        for name, shape in sizes.items():
            if shape == (0,):
                continue
            self.file.create_dataset(
                name,
                shape=(0, *shape),
                maxshape=(None, *shape),
                chunks=(256, *shape),
                compression="lzf",
                dtype="f8",
            )

    def before_step(self, session):
        d = session.d
        if d.time - self.start >= self.max_sim_seconds:
            raise TimeoutError("Simulation-time budget exhausted")
        self.pending = dict(
            time=float(d.time),
            qpos=d.qpos.copy(),
            qvel=d.qvel.copy(),
            warmstart=d.qacc_warmstart.copy(),
        )

    def before_integrate(self, session):
        d = session.d
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied) or session.push is not None:
            raise ValueError("Undeclared external assistance is forbidden by this task profile")
        self.pending["ctrl"] = d.ctrl.copy()

    def after_step(self, session):
        self.pending["applied"] = session.motors.applied(session.d)
        if session.m.nsensordata:
            self.pending["sensors_after"] = session.d.sensordata.copy()
        self.buffer.append(self.pending)
        self.count += 1
        if len(self.buffer) >= 256:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        for key in self.buffer[0]:
            dataset = self.file[key]
            start = len(dataset)
            dataset.resize(start + len(self.buffer), axis=0)
            dataset[start:] = np.asarray([row[key] for row in self.buffer])
        self.buffer.clear()
        self.file.flush()

    def close(self, session):
        self.flush()
        self.file.create_dataset("final_qpos", data=session.d.qpos)
        self.file.create_dataset("final_qvel", data=session.d.qvel)
        self.file.attrs["complete_intervals"] = self.count // self.period
        self.file.attrs["unexported_tail_steps"] = self.count % self.period
        self.file.attrs["checks"] = json.dumps(session.checks.as_dict())
        self.file.close()


def execute_motor_chunk(model, data, action):
    """The exact exported action interface, also usable by a learned policy.

    One flattened action is 20 successive all-motor commands at 500 Hz. No WBC is
    applied on top. This is a torque policy interface, not a Cartesian target policy.
    """
    commands = np.asarray(action).reshape(-1, model.nu)
    if commands.shape != (20, model.nu) or not np.isfinite(commands).all():
        raise ValueError("Expected 20 finite motor command vectors")
    if not np.isclose(model.opt.timestep, 0.002):
        raise ValueError("Motor-chunk adapter requires a 2 ms physics timestep")
    for command in commands:
        mujoco.mj_step1(model, data)
        data.ctrl[:] = np.clip(
            command, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        )
        mujoco.mj_step2(model, data)
