"""Last step of the tea film: the robot takes the full mug (object_0316, 0.48 kg with the tea) off the counter and walks
away with it along the counter toward -x, then stands holding it.

Grasp (lead's decision, 29 Oct): the step-1 body grasp (agents/mugfast/skill_mug.py, MugSkill) - the LEFT hand comes
down beside the mug on its -x side (the handle points +x, to the robot's right), fingers forward, palm to the mug, and
closes round the body. The Dex3 pinch of the handle itself cannot hold the full mug (skill_cup_handle.py, runs c08-c13).

Physically simulated: MuJoCo 600 Hz, every G1 motor torque-limited (byor MotorLayer). Standing work = byor whole-body
controller with payload compensation from the hand-mug contact forces (what a wrist force sensor reads); the hand's
orientation is corrected by the mug's measured tilt (as a camera would see it). Walking = the walking network through the
motors, commanded only with velocity commands (MugSkill.drive); the carrying left arm keeps its pose through the walking
controller's PD set-points (+ gravity sag of arm and mug), its fingers keep squeezing. The mug is a free body held only
by finger contact. Nothing writes qpos/qvel; no welds; no external forces; no limits raised; friction as in the model.

run(p, s, **kw) -> dict ('ok' = no fall, mug held at the end, slip <= 5 mm, tilt <= 8 deg, nothing else moved > 5 mm;
contacts: probe_pen.py on the recording). Starts from the robot standing near (1.86, 1.90) facing +y, the mug at
(1.72, 2.24) handle +x; plans from the MEASURED stance and mug pose.
"""
from __future__ import annotations
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401

import math
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(WORK / "agents/mugfast"))
from motion import quat_mul, smooth  # noqa: E402
from skill_mug import CURL, Q, TX, TY, MugSkill  # noqa: E402
import skill_cup_handle as H  # noqa: E402   (finger closing to contact, force-balanced squeeze, walking follower)

L = 0
GOAL = (0.876, 1.781, 180.0)          # where the d02 walk ended: along the counter toward -x, facing -x


class CupBody(MugSkill):
    hand_index = L
    # from the handle version (same physics, hand-generic): closing each finger group until it touches, squeezing to
    # balanced measured pad forces (Dex3 tactile), and the walking follower whose carrying fingers keep the torque they
    # squeeze with (MugSkill.drive closes them fully: an unbalanced squeeze at the walk start)
    contact_force = H.CupSkill.contact_force
    targets = H.CupSkill.targets
    close_to_contact = H.CupSkill.close_to_contact
    force_squeeze = H.CupSkill.force_squeeze
    standing_again = H.CupSkill.standing_again
    drive = H.CupSkill.drive

    def __init__(self, p, s, **kw):
        self.tilt_max = 0.0
        self.tilt_walk_max = 0.0
        self.tilt_t = 0.0
        self.tilt_lifted_max = 0.0
        self.z_min = 9.0
        super().__init__(p, s, **kw)
        self.fnames = {a: s.m.joint(int(np.nonzero(s.m.jnt_qposadr == a)[0][0])).name for a, *_ in s.fingers[self.b]}

    def rel(self):
        d, b = self.d, self.b
        return d.xmat[b].reshape(3, 3).T @ (d.xpos[self.mb] - d.xpos[b])

    def track(self, walking=False):
        self.track_slip(walking)

    # tilt is tracked every control tick (standing) and every policy tick (walking), the whole step
    def track_slip(self, walking=False):
        super().track_slip(walking)
        t = self.tilt()
        if t > self.tilt_max:
            self.tilt_max, self.tilt_t = t, float(self.d.time)
        if walking:
            self.tilt_walk_max = max(self.tilt_walk_max, t)
        if self.hold_on:
            self.z_min = min(self.z_min, float(self.d.xpos[self.mb][2]))
            if self.support() < 0.05 * self.WEIGHT:           # lifted: the counter carries nothing
                self.tilt_lifted_max = max(self.tilt_lifted_max, t)

    # ------------------------------------------------------------------ levelling: closed loop on the mug's tilt
    # (as a camera on the robot would measure it). Standing: an integrating correction on the hand's commanded
    # orientation, updated every control tick (smooth; b05 used 0.1 s stop-and-go pieces = 3 m/s^2 on the mug).
    # Walking: the same correction through the carrying wrist's PD set-points (walk_level).
    def tilt_vec(self):
        """World rotation (axis * angle) that brings the mug's axis back to vertical."""
        mz = self.d.xmat[self.mb].reshape(3, 3)[:, 2]
        axis = np.cross(mz, [0.0, 0.0, 1.0])
        sa = float(np.linalg.norm(axis))
        return np.zeros(3) if sa < 1e-9 else axis / sa * math.atan2(sa, mz[2])

    def level_reset(self, q_base):
        self.q_base = np.asarray(q_base, float).copy()
        self.q_corr = np.array([1.0, 0.0, 0.0, 0.0])

    def level_step(self, dt):
        w = self.tilt_vec() * dt / self.kw.get("level_tau", 0.25)
        ang = float(np.linalg.norm(w))
        if ang > 1e-12:
            self.q_corr = quat_mul(np.r_[math.cos(ang / 2), math.sin(ang / 2) * w / ang], self.q_corr)
            self.q_corr /= np.linalg.norm(self.q_corr)
        lim = math.radians(self.kw.get("level_max_deg", 20.0))             # anti-windup
        a = 2 * math.acos(min(1.0, abs(float(self.q_corr[0]))))
        if a > lim:
            v = self.q_corr[1:] / max(np.linalg.norm(self.q_corr[1:]), 1e-12) * np.sign(self.q_corr[0])
            self.q_corr = np.r_[math.cos(lim / 2), math.sin(lim / 2) * v]

    def level_move(self, target, seconds, via=None):
        """Hand tool point along one minimum-jerk path (one bell-shaped speed profile), the orientation levelled
        every control tick."""
        from motion import path
        p, b = self.p, self.b
        w = p.w
        go = path(w.ee_goal[b].copy(), via, np.asarray(target, float))
        T = p.mv.TICK
        n = max(1, int(round(seconds / T)))
        for k in range(1, n + 1):
            self.level_step(T)
            p._ease_arm(L, k * T)
            w.set_ee_target(b, go(smooth(k / n)), quat=quat_mul(self.q_corr, self.q_base))
            p.mv._tick(T)

    def level_hold(self, seconds):
        self.level_move(self.p.w.ee_goal[self.b].copy(), seconds)

    def walk_level(self, dt):
        """Walking: the carrying hand holds a fixed WORLD orientation through the wrist's PD set-points (roll/pitch/
        yaw), so the torso's roll with every step (g02: +-6 deg, which the mug followed one to one) is taken out at the
        wrist: set-point += J_wrist^+ (K * orientation error), every policy tick (20 ms). The goal itself is levelled
        slowly by the mug's measured tilt."""
        m, d, b, loco = self.m, self.d, self.b, self.s.loco
        if not self.kw.get("walk_servo", False):
            # default (run g02): slow integral levelling only - set-point += J^+ (mug tilt * dt / tau). The fast servo
            # below (walk_servo=True, run g03) halved the mean tilt but disturbed the gait (stopped 30 cm short).
            wv = self.tilt_vec() * dt / self.kw.get("walk_int_tau", 0.3)
            jacr = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, None, jacr, b)
            J = jacr[:, self.arm_dofs[-3:]]
            dq = np.clip(J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), wv), -0.01, 0.01)
            for qa, x in zip(self.arm_q[-3:], dq):
                loco.hold_target[int(np.nonzero(loco.hold_q == qa)[0][0])] += float(x)
            return
        if getattr(self, "walk_goal", None) is None:
            self.walk_goal = d.xquat[b].copy()
        w = self.tilt_vec() * dt / self.kw.get("walk_level_tau", 0.5)
        ang = float(np.linalg.norm(w))
        if ang > 1e-12:
            self.walk_goal = quat_mul(np.r_[math.cos(ang / 2), math.sin(ang / 2) * w / ang], self.walk_goal)
        r = np.zeros(3)
        mujoco.mju_subQuat(r, self.walk_goal, d.xquat[b])
        err = d.xmat[b].reshape(3, 3) @ r                                   # world
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, None, jacr, b)
        J = jacr[:, self.arm_dofs[-3:]]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), self.kw.get("walk_wrist_k", 0.5) * err)
        dq = np.clip(dq, -0.03, 0.03)
        for qa, x in zip(self.arm_q[-3:], dq):
            i = int(np.nonzero(loco.hold_q == qa)[0][0])
            loco.hold_target[i] += float(x)

    def heading_vectors(self):
        yaw = self.yaw()
        return np.array([math.cos(yaw), math.sin(yaw), 0.0]), np.array([math.sin(yaw), -math.cos(yaw), 0.0])

    def run(self):
        p, s, d, m, b, kw = self.p, self.s, self.d, self.m, self.b, self.kw
        res = dict(ok=False)
        self.mark("start")
        free = [int(m.jnt_bodyid[j]) for j in range(m.njnt)
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and int(m.jnt_bodyid[j]) not in self.robot
                and int(m.jnt_bodyid[j]) != self.mb]
        others0 = {fb: d.xpos[fb].copy() for fb in free}
        res["stance"] = dict(xy=np.round(d.qpos[:2], 3).tolist(), heading_deg=round(math.degrees(self.yaw()), 1))
        home = d.xpos[self.mb].copy()
        res["mug_start"] = dict(xyz=np.round(home, 4).tolist(), handle_yaw_deg=round(self.mug_yaw(), 1))
        FWD, RIGHT = self.heading_vectors()

        # 1. the step-1 grasp, on the counter: the left hand over the gap on the mug's -x side, down beside it, close
        c = home + np.array([0.0, 0.0, kw.get("grip_h", 0.050)])     # the mug's axis, a little above mid-wall
        # down `side` further out from the mug (b01: straight above the grasp the curled middle finger lands on the rim),
        # then in sideways against the body
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, Q)
        out = Rq.reshape(3, 3)[:, 1]                                 # hand frame y: from the mug toward the hand
        c_out = c + out * kw.get("side", 0.02)
        above = c_out + np.array([0.0, 0.0, kw.get("above", 0.10)])
        front = above - FWD * kw.get("front_back", 0.10)
        self.mark("reach")
        p.set_tool(L, (TX, TY, 0.0))
        p.grip(L, CURL, kw.get("curl_s", 0.3))
        sh = d.xpos[s.shoulder[b]].copy()
        via = sh + FWD * kw.get("via_fwd", 0.10)
        via[2] = front[2] - 0.02
        p.reach(L, front, quat=Q, seconds=kw.get("reach_s", 1.4), settle=0.0, via=via)
        e_front = p.settle_hand(L, front, tol=0.008, timeout=kw.get("settle_front", 0.3))
        p.move_hand(L, above, quat=Q, seconds=kw.get("above_s", 0.5))
        e_above = p.settle_hand(L, above, tol=0.006, timeout=kw.get("settle_above", 0.3))
        p.move_hand(L, c_out, quat=Q, seconds=kw.get("down_s", 0.7))
        p.settle_hand(L, c_out, tol=0.005, timeout=0.4)
        p.move_hand(L, c, quat=Q, seconds=kw.get("in_s", 0.4))
        e_c = p.settle_hand(L, c, tol=0.004, timeout=kw.get("settle_to", 0.6))
        res["approach_err_mm"] = dict(front=round(e_front * 1000, 1), above=round(e_above * 1000, 1), at_mug=round(e_c * 1000, 1),
                                      mug_pushed_mm=round(float(np.linalg.norm(d.xpos[self.mb] - home)) * 1000, 1))
        self.note("beside the mug", **res["approach_err_mm"])
        self.mark("grasp")
        # close each finger group until it touches the mug, then squeeze to balanced pad forces (thumb on the near side
        # against index + middle round the far side): b02 closed all fingers fully at once and tipped the mug 12 deg
        touch = self.close_to_contact({"index": ({"index": 0.95}, "index"), "middle": ({"middle": 0.95}, "middle"),
                                       "thumb": ({"thumb_1": 0.95, "thumb_2": 0.95}, "thumb")}, kw.get("close_s", 0.8))
        fsq = kw.get("fsq", 20.0)
        sq = self.force_squeeze({"thumb": fsq, "index": fsq / 2, "middle": fsq / 2}, kw.get("squeeze_s", 0.8))
        self.hold_on = True
        p.hold(kw.get("grip_hold", 0.2))
        res["pinch"] = dict(touch=touch, forces_N=sq)
        self.rel0 = d.xmat[b].reshape(3, 3).T @ (d.xpos[self.mb] - d.xpos[b])
        res["grip"] = dict(mug_moved_mm=round(float(np.linalg.norm(d.xpos[self.mb] - home)) * 1000, 1), tilt_deg=round(self.tilt(), 1))

        # 2. lift straight up off the counter (slowly: the controller takes the load from the wrist force), level it
        self.mark("lift")
        self.level_reset(Q)
        self.level_move(s.wbc.ee_goal[b] + [0.0, 0.0, kw.get("lift", 0.04)], kw.get("lift_s", 1.5))
        self.level_hold(kw.get("lift_hold", 0.3))
        res["lifted"] = dict(rise_mm=round(float(d.xpos[self.mb][2] - home[2]) * 1000, 1), slip_mm=round(self.slip(), 1),
                             tilt_deg=round(self.tilt(), 1), support_N=round(self.support(), 2),
                             load_N=np.round(self.hand_load(), 2).tolist())
        self.note("lifted", **res["lifted"])
        if self.hand_load()[2] > -1.0 or res["lifted"]["rise_mm"] < 15 or self.slip() > 30:
            res.update(failed="grasp: mug not lifted in the hand")
            return self.finish(res, others0)

        # 3. carry pose close in front of the chest (as step 1), the mug levelled by its measured tilt
        self.mark("carry")
        if kw.get("plan", "g02") == "turn_first":            # h01: failed (drift, elbow at its stop); g02 is the default
            return self.turn_first(res, others0, FWD, RIGHT)
        pel = d.qpos[:3].copy()
        carry = pel + FWD * kw.get("carry_fwd", 0.13) + RIGHT * kw.get("carry_right", 0.03)
        carry[2] = c[2] + kw.get("carry_up", 0.05)
        self.level_move(carry, kw.get("out_s", 2.6), via=0.5 * (s.wbc.ee_goal[b] + carry) + np.array([0.0, 0.0, 0.03]))
        self.level_hold(kw.get("carry_hold", 0.3))
        res["carry"] = dict(slip_mm=round(self.slip(), 1), tilt_deg=round(self.tilt(), 1), mug_at=np.round(d.xpos[self.mb], 3).tolist())
        self.note("carry pose", **res["carry"])
        if kw.get("stop_after") == "carry":
            self.level_hold(kw.get("test_hold", 1.0))
            return self.finish(res, others0)

        # 4. turn left and walk away along the counter (path bowed away from the counter: `bow` < 0 is to the left of
        #    the walking direction)
        self.mark("walk")
        goal = kw.get("goal", GOAL)
        self.walk_hook = self.walk_level if kw.get("walk_level", True) else None
        self.walk_goal = None
        self.keep_waist = kw.get("keep_waist", True)
        self.keep_other_arm = kw.get("keep_other_arm", True)    # g01: the free right hand brushed the cabinet (2.1 mm)
        self.no_hold_blend = kw.get("no_hold_blend", True)
        if kw.get("back", 0.0) > 0:                          # optional short straight back-step off the counter first
            back = d.qpos[:2] - FWD[:2] * kw["back"]
            res["back_step"] = self.drive(back, self.yaw(), vmax=0.25, amax=0.3, warm=kw.get("warm_s", 0.4),
                                          idle=kw.get("idle_s", 0.05), tol=0.03, extra=0.5, name="stepped back")
        res["walk"] = self.drive(np.asarray(goal[:2], float), math.radians(goal[2]), vmax=kw.get("v", 0.3), amax=kw.get("a", 0.3),
                                 warm=kw.get("warm_s", 0.4), idle=kw.get("idle_s", 0.05),
                                 stop_short=kw.get("short", 0.04), bow=kw.get("bow", -0.08), tol=0.03, extra=0.6, turn_frac=kw.get("turn_frac", 1.0), grip_other=kw.get("idle_grip", 0.5),
                                 name="walked away")
        res["walked"] = dict(slip_mm=round(self.slip(), 1), tilt_deg=round(self.tilt(), 1), mug_at=np.round(d.xpos[self.mb], 3).tolist())
        self.note("walked", **res["walked"])

        # 5. stand still holding it in front (the person looks down at it); the hand levels the mug again
        self.mark("stand")
        self.level_reset(s.wbc.ee_rot[b].copy())             # the fresh standing controller holds the hand as it is
        self.level_hold(kw.get("end_hold", 1.5))
        self.mark("end")
        res["end"] = dict(slip_mm=round(self.slip(), 1), tilt_deg=round(self.tilt(), 1), mug_at=np.round(d.xpos[self.mb], 3).tolist(),
                          support_N=round(self.support(), 2), robot_xy=np.round(d.qpos[:2], 3).tolist(),
                          robot_heading_deg=round(math.degrees(self.yaw()), 1),
                          held=bool(self.hand_load()[2] < -0.5 * self.WEIGHT and self.support() < 0.1))
        self.note("end", **res["end"])
        return self.finish(res, others0)

    def walk_options(self):
        kw = self.kw
        self.walk_hook = self.walk_level if kw.get("walk_level", True) else None
        self.walk_goal = None
        self.keep_waist = kw.get("keep_waist", True)
        self.keep_other_arm = kw.get("keep_other_arm", True)    # g01: the free right hand brushed the cabinet (2.1 mm)
        self.no_hold_blend = kw.get("no_hold_blend", True)

    def turn_first(self, res, others0, FWD, RIGHT):
        """Plan 'turn_first' (lead, 30 Oct): pull the mug in a little (small arm change), turn in place slowly with the
        tea at rest, lower the mug in front of the stomach (short lever: the gait's torso roll moves it less), then a
        short straight walk at <= 0.25 m/s."""
        p, s, d, b, kw = self.p, self.s, self.d, self.b, self.kw
        # 3. pull in toward the left chest at the lift height (b05/g02: the long move to the chest's middle swung the
        #    wrist through ~0.5 rad and the mug slipped 2 mm at t ~10.6 s)
        pel = d.qpos[:3].copy()
        pre = pel + FWD * kw.get("pre_fwd", 0.17) - RIGHT * kw.get("pre_left", 0.06)
        pre[2] = float(s.wbc.ee_pos(d, b)[2]) + kw.get("pre_up", 0.0)
        self.level_move(pre, kw.get("out_s", 3.0))
        self.level_hold(kw.get("carry_hold", 0.3))
        res["carry"] = dict(slip_mm=round(self.slip(), 1), tilt_deg=round(self.tilt(), 1), mug_at=np.round(d.xpos[self.mb], 3).tolist())
        self.note("pulled in", **res["carry"])
        # 4. turn in place (and 5 cm back off the counter) to face -x, slowly
        self.mark("turn")
        goal = kw.get("goal", GOAL)
        self.walk_options()
        turn_xy = d.qpos[:2] - FWD[:2] * kw.get("turn_back", 0.05)
        res["turn"] = self.drive(turn_xy, math.radians(goal[2]), vmax=0.15, amax=0.2, wmax=kw.get("wmax", 0.4),
                                 warm=kw.get("warm_s", 0.4), idle=kw.get("idle_s", 0.05), tol=0.04, yaw_tol=math.radians(5),
                                 extra=0.8, grip_other=kw.get("idle_grip", 0.5), name="turned")
        # 5. lower it in front of the stomach, forearm about horizontal
        self.mark("lower")
        self.level_reset(s.wbc.ee_rot[b].copy())
        FWD2, RIGHT2 = self.heading_vectors()
        pel = d.qpos[:3].copy()
        low = pel + FWD2 * kw.get("low_fwd", 0.15) + RIGHT2 * kw.get("low_right", -0.02)
        low[2] = kw.get("low_z", 0.93)                     # the mug's axis at grip height; its base ~5 cm lower
        self.level_move(low, kw.get("lower_s", 2.2), via=0.5 * (s.wbc.ee_goal[b] + low) + np.array([0.0, 0.0, 0.01]))
        self.level_hold(kw.get("low_hold", 0.4))
        res["lowered"] = dict(slip_mm=round(self.slip(), 1), tilt_deg=round(self.tilt(), 1), mug_at=np.round(d.xpos[self.mb], 3).tolist())
        self.note("lowered", **res["lowered"])
        # 6. short straight walk away
        self.mark("walk")
        self.walk_options()
        FWD3, _ = self.heading_vectors()
        target = d.qpos[:2] + FWD3[:2] * kw.get("walk_dist", 0.65)
        res["walk"] = self.drive(target, self.yaw(), vmax=kw.get("v", 0.25), amax=kw.get("a", 0.25), warm=kw.get("warm_s", 0.4),
                                 idle=kw.get("idle_s", 0.05), stop_short=kw.get("short", 0.04), tol=0.03, extra=0.6,
                                 grip_other=kw.get("idle_grip", 0.5), name="walked away")
        res["walked"] = dict(slip_mm=round(self.slip(), 1), tilt_deg=round(self.tilt(), 1), mug_at=np.round(d.xpos[self.mb], 3).tolist())
        self.note("walked", **res["walked"])
        self.mark("stand")
        self.level_reset(s.wbc.ee_rot[b].copy())
        self.level_hold(kw.get("end_hold", 1.5))
        self.mark("end")
        res["end"] = dict(slip_mm=round(self.slip(), 1), tilt_deg=round(self.tilt(), 1), mug_at=np.round(d.xpos[self.mb], 3).tolist(),
                          support_N=round(self.support(), 2), robot_xy=np.round(d.qpos[:2], 3).tolist(),
                          robot_heading_deg=round(math.degrees(self.yaw()), 1),
                          held=bool(self.hand_load()[2] < -0.5 * self.WEIGHT and self.support() < 0.1))
        self.note("end", **res["end"])
        return self.finish(res, others0)

    def finish(self, res, others0):
        s, d, m = self.s, self.d, self.m
        if "end" not in self.phase:
            self.mark("end")
        res["slip_max_mm"] = round(self.slip_max, 1)
        res["slip_walk_max_mm"] = round(self.slip_walk_max, 1)
        res["tilt_max_deg"] = round(self.tilt_max, 1)
        res["tilt_max_at_s"] = round(self.tilt_t, 2)
        res["tilt_walk_max_deg"] = round(self.tilt_walk_max, 1)
        res["tilt_lifted_max_deg"] = round(self.tilt_lifted_max, 1)
        res["mug_lowest_held_z"] = round(self.z_min, 3)
        moved = sorted(((float(np.linalg.norm(d.xpos[fb] - x0)) * 1000, m.body(fb).name) for fb, x0 in others0.items()), reverse=True)
        res["others_moved_mm"] = [(n, round(v, 2)) for v, n in moved[:3]]
        ph = self.phase
        keys = sorted((k for k in ph if k != "start"), key=lambda k: ph[k])
        res["phases"] = ph
        res["durations_s"] = {a: round(ph[b_] - ph[a], 2) for a, b_ in zip(keys, keys[1:])}
        res["total_s"] = round(ph["end"] - ph["start"], 2)
        res["checks"] = s.checks.as_dict()
        held = res.get("end", {}).get("held", False)
        res["ok"] = bool(s.checks.fell_at is None and held and self.slip_max <= 5.0 and self.tilt_max <= 8.0
                         and self.tilt_lifted_max <= self.kw.get("tilt_gate", 5.0)
                         and (not moved or moved[0][0] <= 5.0))
        return res


def run(p, s, **kw) -> dict:
    """The cup step inside a continuous run. p: motion.Program on session s. Options: grip_h, close, lift, lift_s,
    carry_fwd, carry_right, carry_up, out_s, goal (x, y, heading deg), v, a, bow, back, end_hold, level_gain,
    stop_after ('carry'). Returns measured results."""
    return CupBody(p, s, **kw).run()
