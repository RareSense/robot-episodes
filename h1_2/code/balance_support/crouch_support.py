"""Measured-height crouch and COM balance for the native PhysX H1-2 Sim.

This is an add-on to the existing implicit force-limited leg drives. It only
changes leg joint targets, never physical states or forces on the free base.
No MuJoCo dependency. Arm/hand commands and their ownership are unchanged.
"""
from __future__ import annotations
import math
import numpy as np


def qrot(q, v):
    u, w = q[:3], q[3]
    return v + 2*np.cross(u, np.cross(u, v) + w*v)


def yaw(q):
    x, y, z, w = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def smooth(u):
    u = float(np.clip(u, 0, 1))
    return u*u*u*(10 + u*(-15+6*u))


class CrouchSupport:
    def __init__(self, sim, lateral_kp=1.2, max_drop=.14):
        if sim.mode != 'stand':
            raise ValueError('Enter standing mode and settle before installing crouch support')
        self.s = sim
        self.old_balance = sim.balance
        self.kp_roll = lateral_kp
        self.max_drop = max_drop
        self.dt_control = 12/600
        self.leg_names = [sim.names[i] for i in sim.legs]
        self.nominal = sim.stand_goal.copy()
        self.leg_sign = np.array([-1 if ('hip_pitch' in n or 'ankle_pitch' in n) else
                                  2 if 'knee' in n else 0 for n in self.leg_names], float)
        self.base0 = sim.pose()[0].copy()
        self.foot0 = sim.feet_xy().copy()
        self.root_q0 = sim.pose()[1].copy()
        self.z_ref = float(self.base0[2])
        self.trajectory = None
        self.angle, self.integral = 0., 0.
        self.prev_com = None
        self.velocity = np.zeros(2)
        self.max_foot_drift = 0.
        self.max_tilt_deg = 0.
        self.max_com_error = 0.
        self.rows = []
        # Height sensitivity comes from a two-link leg model; feedback corrects
        # unequal thigh/shin lengths, stance angle, compliance and payload.
        self.leg_length = float(self.base0[2] - self.foot0[:, 2].mean())
        knees = [sim.stand_goal[i] for i, n in enumerate(self.leg_names) if 'knee' in n]
        self.theta0 = .5*float(np.mean(knees))
        sim.balance = self.update
        sim.stand_t = max(sim.stand_t, sim.stand_ramp)

    def set_height(self, drop_m, seconds=2.5):
        if not 0 <= drop_m <= self.max_drop:
            raise ValueError(f'Crouch drop must be in [0, {self.max_drop}] m')
        self.trajectory = (self.s.t, max(float(seconds), .5), self.z_ref, self.base0[2]-drop_m)

    def update(self):
        s = self.s
        if s.mode != 'stand':
            raise RuntimeError('Crouch support is for standing; uninstall before walking')
        p, q = s.pose()
        if self.trajectory:
            t0, seconds, z0, z1 = self.trajectory
            self.z_ref = z0 + smooth((s.t-t0)/seconds)*(z1-z0)
        drop = float(self.base0[2] - self.z_ref)
        cosine = np.clip(math.cos(self.theta0) - drop/self.leg_length, -.95, .9999)
        feed = max(0., math.acos(cosine) - self.theta0)
        error_z = float(p[2]-self.z_ref)
        sensitivity = max(.20, self.leg_length*math.sin(self.theta0+feed))
        # Integral compensation provides the offset required by force-driven
        # joints under load; it does not change link weights or gravity.
        self.integral = float(np.clip(self.integral + .5*error_z*self.dt_control/sensitivity, -.12, .12))
        requested = float(np.clip(feed + .6*error_z/sensitivity + self.integral, -.035, .60))
        self.angle += float(np.clip(requested-self.angle, -.15*self.dt_control, .15*self.dt_control))
        s.stand_goal = self.nominal + self.leg_sign*self.angle

        h = yaw(q)
        c, sn = math.cos(h), math.sin(h)
        world_to_body = np.array([[c, sn], [-sn, c]])
        com = s.com()[:2]
        feet = s.feet_xy()
        target = feet[:, :2].mean(0) + np.array([c, sn])*.036
        e = world_to_body @ (com-target)
        if self.prev_com is None:
            self.prev_com = com.copy()
        velocity = world_to_body @ (com-self.prev_com)/self.dt_control
        self.prev_com = com.copy()
        self.velocity = .7*self.velocity + .3*velocity
        pitch = float(np.clip(1.0*e[0] + .30*self.velocity[0], -.15, .15))
        roll = float(np.clip(self.kp_roll*e[1] + .40*self.velocity[1], -.12, .12))
        s.off[:] = 0.
        for i, name in enumerate(self.leg_names):
            if 'ankle_pitch' in name:
                s.off[i] += pitch
            elif 'hip_roll' in name:
                s.off[i] += roll
            elif 'ankle_roll' in name:
                s.off[i] -= roll
        s.e_com = e
        drift = float(np.linalg.norm(feet-self.foot0, axis=1).max())
        tilt = math.degrees(math.acos(float(np.clip(qrot(q, np.array([0.,0,1]))[2], -1, 1))))
        self.max_foot_drift = max(self.max_foot_drift, drift)
        self.max_tilt_deg = max(self.max_tilt_deg, tilt)
        self.max_com_error = max(self.max_com_error, float(np.linalg.norm(e)))
        self.rows.append([s.t, *p, self.z_ref, self.angle, *e, drift, tilt])

    def crouch(self, drop_m, seconds=2.5, settle=1.0):
        self.set_height(drop_m, seconds)
        n = int(round((seconds+settle)*600))
        for _ in range(n):
            self.s.step()
            if self.s.fallen():
                break
        return self.result()

    def result(self):
        p, _ = self.s.pose()
        error = abs(float(p[2]-self.z_ref))
        return dict(no_fall=not self.s.fallen(), requested_drop_mm=1000*(self.base0[2]-self.z_ref),
                    measured_drop_mm=1000*(self.base0[2]-p[2]), height_error_mm=1000*error,
                    foot_drift_mm=1000*self.max_foot_drift, peak_tilt_deg=self.max_tilt_deg,
                    peak_com_error_mm=1000*self.max_com_error, knee_bend_offset_rad=self.angle,
                    passed=bool(not self.s.fallen() and error <= .005 and self.max_foot_drift <= .005
                                and self.max_tilt_deg <= 8))

    def uninstall(self):
        self.s.balance = self.old_balance
