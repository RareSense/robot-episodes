"""Typed task skills for an imported robot: open/close a door, pick an object, place or
drop it, clear a set of objects into a container.

Every skill is composed from the two honest primitives that already exist — goto (walking
velocity commands into the locomotion policy) and move_hand (leashed hand targets into the
whole-body controller) — so a skill can only ever act through the robot's torque-limited
motors. Success is measured from the physics state (the door's own joint angle, the
object's own pose), never assumed, and each skill returns a report saying what actually
happened. A robot that cannot physically do a step (reach exhausted, grasp slipped twice)
gets a failure report, not a faked success.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .navigate import goto


class Tasks:
    """Task skills bound to one TeleopSession. `on_tick(session)` runs every control step
    (rendering hooks)."""

    def __init__(self, session, on_tick=None):
        self.s = session
        self.on_tick = on_tick
        self.m, self.d = session.m, session.d
        self.held = {}

    # ------------------------------------------------------------------ small helpers
    def _tick(self, seconds):
        n = max(1, int(round(seconds / self.m.opt.timestep)))
        chunk = max(1, int(round(0.02 / self.m.opt.timestep)))
        done = 0
        while done < n:
            self.s.step(min(chunk, n - done))
            done += chunk
            if self.on_tick is not None:
                self.on_tick(self.s)

    def _xy_yaw(self):
        qa = self.s.loco.qa
        q = self.d.qpos[qa + 3:qa + 7]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return self.d.qpos[qa:qa + 2].copy(), yaw

    def _wait_stand(self, timeout=10.0):
        t0 = self.d.time
        while self.s.mode != "stand" and self.d.time - t0 < timeout:
            if self.s.checks.fell_at is not None:
                return False
            self._tick(0.1)
        return self.s.mode == "stand" and self.s.checks.fell_at is None

    def _goto(self, xy, yaw=None, speed=0.2, inflate=0.30, ignore=(), goal_tol=0.08):
        try:
            r = goto(self.s, xy, final_yaw=yaw, speed=speed, inflate=inflate,
                     on_step=self.on_tick, ignore_bodies=ignore, goal_tol=goal_tol)
        except ValueError as e:
            return dict(arrived=False, reason=str(e))
        if r.get("arrived"):
            r["stood"] = self._wait_stand()
            if not r["stood"]:
                r.update(arrived=False, reason="failed to reach stable standing")
        return r

    def _standoff(self, point, distance, from_xy=None):
        """Where to stand to work on `point`: `distance` back along the approach line."""
        here = self._xy_yaw()[0] if from_xy is None else np.asarray(from_xy, float)
        v = np.asarray(point[:2], float) - here
        n = v / max(np.linalg.norm(v), 1e-6)
        stand = np.asarray(point[:2], float) - n * distance
        return stand, math.atan2(n[1], n[0])

    def _hand(self, target):
        """Hand index nearest to a world point."""
        w = self.s.wbc
        return int(np.argmin([np.linalg.norm(w.ee_pos(self.d, b) - target) for b in self.s.hands]))

    def _hand_to(self, i, target, tol=0.025, timeout=6.0, speed=0.25):
        """Walk a hand's goal to `target` through move_hand (leash, reach and body guards
        stay active). Returns the final error in metres."""
        w, b = self.s.wbc, self.s.hands[i]
        t0 = self.d.time
        step = speed * 0.02
        while self.d.time - t0 < timeout:
            if self.s.checks.fell_at is not None or self.s.mode != "stand":
                break
            err = np.asarray(target, float) - w.ee_pos(self.d, b)
            n = np.linalg.norm(err)
            if n < tol:
                break
            # Advance the reference towards the requested point. Integrating
            # actual tracking error into the previous reference winds the goal
            # past the object and makes a slow push overshoot by the whole leash.
            lead = np.asarray(target, float) - w.ee_goal[b]
            length = np.linalg.norm(lead)
            if length > 1e-9:
                self.s.move_hand(i, lead * min(1.0, step / length))
            self._tick(0.02)
        return float(np.linalg.norm(np.asarray(target, float) - w.ee_pos(self.d, b)))

    def _hands_home(self, settle=0.8):
        self.s.home_hands()
        self._tick(settle)

    def orient_hand(self, hand, quaternion, *, speed=0.4, timeout=10.0):
        """Rotate a stationary hand target gradually, in radians per second.

        The caller first moves to a clear pose. This changes controller targets
        only; actual orientation error and falls determine the result.
        """
        goal = np.asarray(quaternion, float)
        if goal.shape != (4,) or not np.all(np.isfinite(goal)) or np.linalg.norm(goal) < 1e-8:
            raise ValueError("hand orientation must be a finite, nonzero quaternion")
        if speed <= 0 or timeout <= 0:
            raise ValueError("orientation speed and timeout must be positive")
        goal = goal / np.linalg.norm(goal)
        body = self.s.hands[hand]
        start = self.s.wbc.ee_rot[body].copy()
        if np.dot(start, goal) < 0:
            goal = -goal
        angle = math.acos(float(np.clip(np.dot(start, goal), -1, 1)))
        duration = max(0.02, 2 * angle / speed)
        for elapsed in np.arange(0.02, min(duration, timeout) + 0.01, 0.02):
            fraction = min(1.0, elapsed / duration)
            if angle < 1e-6:
                target = goal
            else:
                target = (math.sin((1 - fraction) * angle) * start
                          + math.sin(fraction * angle) * goal) / math.sin(angle)
            self.s.wbc.ee_rot[body] = target
            self._tick(0.02)
            if self.s.checks.fell_at is not None or self.s.mode != "stand":
                break
        error = 2 * math.acos(float(np.clip(abs(np.dot(self.d.xquat[body], goal)), 0, 1)))
        return dict(ok=error < 0.15 and self.s.checks.fell_at is None,
                    orientation_error_rad=error, fell=self.s.checks.fell_at)

    def _body_pos(self, name):
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            raise ValueError(f"no body named {name!r}")
        return b, self.d.xpos[b].copy()

    # ------------------------------------------------------------------ doors
    def push_hinge(self, joint, contact_local, target_angle, *, hand=None,
                   timeout=8.0, tolerance=0.08, lead=0.04, speed=0.15):
        """Push a passive hinge along its changing tangent from the current stance.

        ``contact_local`` identifies a reachable point on the panel in its body
        frame. Callers may reposition between attempts; an exhausted reach is a
        reported partial attempt, never a successful door operation.
        """
        j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if j < 0 or self.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"not a hinge joint: {joint}")
        if self.m.jnt_limited[j] and not self.m.jnt_range[j, 0] <= target_angle <= self.m.jnt_range[j, 1]:
            raise ValueError("requested hinge angle is outside the joint limits")
        adr, body = int(self.m.jnt_qposadr[j]), int(self.m.jnt_bodyid[j])
        local = np.asarray(contact_local, float)
        start = float(self.d.qpos[adr])
        t0, last_progress = self.d.time, self.d.time
        best = abs(start - target_angle)
        error = None
        while self.d.time - t0 < timeout:
            angle = float(self.d.qpos[adr])
            if abs(angle - target_angle) <= tolerance:
                return dict(ok=True, angle=angle, start_angle=start, hand=hand)
            if self.s.checks.fell_at is not None or self.s.mode != "stand":
                break
            point = self.d.xpos[body] + self.d.xmat[body].reshape(3, 3) @ local
            tangent = np.cross(self.d.xaxis[j], point - self.d.xanchor[j])
            norm = np.linalg.norm(tangent)
            if norm < 1e-6:
                raise ValueError("hinge contact point must be away from the axis")
            target = point + np.sign(target_angle - angle) * lead * tangent / norm
            if hand is None:
                hand = self._hand(target)
            error = self._hand_to(hand, target, timeout=0.2, speed=speed, tol=0.005)
            remaining = abs(float(self.d.qpos[adr]) - target_angle)
            if remaining < best - 0.015:
                best, last_progress = remaining, self.d.time
            if error < 0.06 and self.d.time - last_progress > 2.0:
                break
        angle = float(self.d.qpos[adr])
        return dict(ok=abs(angle - target_angle) <= tolerance and self.s.checks.fell_at is None,
                    angle=angle, start_angle=start, hand=hand, hand_error_m=error,
                    reason="target not reached from this stance", fell=self.s.checks.fell_at)

    def _door(self, joint):
        j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if j < 0:
            raise ValueError(f"no joint named {joint!r}")
        a = int(self.m.jnt_qposadr[j])
        body = int(self.m.jnt_bodyid[j])
        geoms = [g for g in range(self.m.ngeom) if int(self.m.geom_bodyid[g]) == body]
        handles = [g for g in geoms
                   if "handle" in (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        panel = max(geoms, key=lambda g: float(np.prod(self.m.geom_size[g])))
        return a, body, panel, handles

    def _panel_normal(self, panel, away_from):
        """The panel's thin-axis direction in world, pointing away from `away_from`."""
        R = self.d.geom_xmat[panel].reshape(3, 3)
        n = R[:, int(np.argmin(self.m.geom_size[panel]))]
        to_panel = self.d.geom_xpos[panel] - np.asarray(away_from, float)
        return n if n @ to_panel > 0 else -n

    def open_door(self, joint, open_to=1.1, push_clear=None):
        """Open a hinged door from the side the robot is on: walk to the handle, push the
        panel at the handle along its own swinging arc until the arm's reach is used up,
        then walk through the doorway, letting the body push it the rest of the way.
        `push_clear` is a point past the doorway to walk to (default: mirrored stand point).
        Success = the door's own hinge angle reached `open_to`."""
        adr, body, panel, handles = self._door(joint)
        grab = (np.mean([self.d.geom_xpos[g] for g in handles], axis=0) if handles
                else self.d.geom_xpos[panel].copy())
        stand, face = self._standoff(grab, 0.52)
        r = self._goto(stand, face)
        if not r.get("arrived"):
            return dict(ok=False, stage="goto", **r)
        here = self._xy_yaw()[0]
        # approach just off the panel face, at handle height, then push along the moving arc
        n0 = self._panel_normal(panel, np.r_[here, 1.0])
        i = self._hand(grab)
        self._hand_to(i, grab - 0.10 * n0, tol=0.03)
        start = float(self.d.qpos[adr])
        stall, last = self.d.time, start
        while abs(float(self.d.qpos[adr]) - start) < open_to:
            fresh = (np.mean([self.d.geom_xpos[g] for g in handles], axis=0) if handles
                     else self.d.geom_xpos[panel].copy())
            n = self._panel_normal(panel, np.r_[here, 1.0])
            err = self._hand_to(i, fresh + 0.05 * n, tol=0.02, timeout=0.6)
            ang = float(self.d.qpos[adr])
            if abs(ang - last) > 0.03:
                stall, last = self.d.time, ang
            if self.d.time - stall > 3.0 or err > 0.35:
                break                              # reach used up from this stance
        pushed_to = float(self.d.qpos[adr])
        self._hands_home()
        opened = abs(pushed_to - start) >= open_to
        walked = None
        if not opened:
            # finish by walking through: the body pushes the panel on the way
            if push_clear is None:
                push_clear = grab[:2] + (grab[:2] - here) * (0.9 / max(np.linalg.norm(grab[:2] - here), 1e-6))
            # plan through the door itself (it is openable, that is the point), with the
            # robot's true body radius rather than the comfortable default; walking into
            # the panel pushes it open the way a person with full arms does
            walked = self._goto(push_clear, speed=0.35, inflate=0.20, ignore=(body,))
            opened = abs(float(self.d.qpos[adr]) - start) >= 0.9 * open_to
        return dict(ok=bool(opened), angle=round(float(self.d.qpos[adr]), 2),
                    pushed_to=round(pushed_to, 2), walked_through=walked,
                    fell=self.s.checks.fell_at)

    def close_door(self, joint, closed_at=0.1):
        """Push the door back towards its closed stop from wherever the robot is."""
        adr, body, panel, handles = self._door(joint)
        edge = self.d.geom_xpos[panel].copy()
        stand, face = self._standoff(edge, 0.5)
        r = self._goto(stand, face)
        if not r.get("arrived"):
            return dict(ok=False, stage="goto", **r)
        i = self._hand(edge)
        stall, last = self.d.time, float(self.d.qpos[adr])
        while abs(float(self.d.qpos[adr])) > closed_at:
            n = self._panel_normal(panel, np.r_[self._xy_yaw()[0], 1.0])
            target = self.d.geom_xpos[panel] + 0.05 * n
            target[2] = 1.0
            self._hand_to(i, target, tol=0.02, timeout=0.6)
            ang = float(self.d.qpos[adr])
            if abs(ang - last) > 0.03:
                stall, last = self.d.time, ang
            if self.d.time - stall > 3.0:
                break
        self._hands_home()
        return dict(ok=abs(float(self.d.qpos[adr])) <= closed_at * 1.5,
                    angle=round(float(self.d.qpos[adr]), 2), fell=self.s.checks.fell_at)

    # ------------------------------------------------------------------ objects
    def _grasp_contacts(self, hand, body):
        from .analysis import subtree

        objects = set(subtree(self.m, body))
        parts = self.s.hand_parts[self.s.hands[hand]]
        touching = set()
        for contact in self.d.contact:
            a, b = (int(self.m.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if contact.dist <= 0.001:
                if a in objects and b in parts:
                    touching.add(b)
                elif b in objects and a in parts:
                    touching.add(a)
        return len(touching)

    def grasp(self, body_name, *, hand=None, opening=0.35, closure=0.6,
              grasp_offset=(0, 0, 0), clearance=0.15, lift=0.22, orientation=None):
        """Grasp from the current stance and prove a sustained, contacted lift.

        Opening/closure and the object-local grasp point belong to the task's
        grasp hypothesis. Geometry alone does not guarantee a feasible grasp.
        """
        body, position = self._body_pos(body_name)
        if not self.m.body_jntnum[body] or self.m.jnt_type[self.m.body_jntadr[body]] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("grasp requires a free scene object")
        if hand is None:
            hand = self._hand(position)
        h = self.s.hands[hand]
        self.s.use_grasp_point(hand, closure)
        point = position + self.d.xmat[body].reshape(3, 3) @ np.asarray(grasp_offset, float)
        self.s.set_grip(hand, opening)
        self._tick(0.7)
        high = self.s.wbc.ee_pos(self.d, h).copy()
        high[2] = max(high[2], point[2] + clearance + 0.1)
        for phase, target in (("raise", high), ("approach", point + [0, 0, clearance]),
                              ("descend", point)):
            error = self._hand_to(hand, target, tol=0.005, timeout=4, speed=0.12)
            # The first waypoint provides clearance, not a grasp alignment.
            # Body guards may limit its horizontal component after walking.
            clear = (phase == "raise" and error < 0.12
                     and self.s.wbc.ee_pos(self.d, h)[2] >= point[2] + 0.08)
            if (error > 0.035 and not clear) or self.s.checks.fell_at is not None:
                return dict(ok=False, stage=phase, hand_error_m=error, fell=self.s.checks.fell_at)
            if phase == "raise" and orientation is not None:
                result = self.orient_hand(hand, orientation)
                if not result["ok"]:
                    return dict(stage="orient", **result)
        self.s.set_grip(hand, closure)
        self._tick(1.0)
        z0 = float(self.d.xpos[body, 2])
        self._hand_to(hand, point + [0, 0, lift], tol=0.015, timeout=5, speed=0.12)
        self._tick(0.5)
        rise = float(self.d.xpos[body, 2]) - z0
        contacts = self._grasp_contacts(hand, body)
        ok = rise >= 0.08 and contacts >= 2 and self.s.checks.fell_at is None
        if ok:
            self.held[body_name] = dict(hand=hand, body=body,
                offset=self.d.xpos[body].copy() - self.s.wbc.ee_pos(self.d, h))
        return dict(ok=bool(ok), lifted_m=rise, contact_bodies=contacts, hand=hand,
                    object_position=self.d.xpos[body].tolist(), fell=self.s.checks.fell_at)

    def release_into(self, body_name, drop_point, bounds, *, timeout=6.0):
        """Place a proven grasp into a 3-D collection region from this stance."""
        if body_name not in self.held:
            return dict(ok=False, reason="no verified grasp for this object")
        held = self.held[body_name]
        hand, body = held["hand"], held["body"]
        target = np.asarray(drop_point, float) - held["offset"]
        error = self._hand_to(hand, target, tol=0.015, timeout=timeout, speed=0.10)
        contacts = self._grasp_contacts(hand, body)
        lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
        actual = self.d.xpos[body]
        above_region = (np.all(actual[:2] >= lo[:2] + 0.025)
                        and np.all(actual[:2] <= hi[:2] - 0.025)
                        and actual[2] >= hi[2] + 0.05)
        if not above_region or contacts < 2 or self.s.checks.fell_at is not None:
            return dict(ok=False, reason="transfer failed or grasp lost", hand_error_m=error,
                        contact_bodies=contacts, object_position=actual.tolist(),
                        fell=self.s.checks.fell_at)
        self.s.set_grip(hand, 0.0)
        self._tick(0.8)
        self.held.pop(body_name)
        retreat = self.s.wbc.ee_pos(self.d, self.s.hands[hand]).copy() + [0, 0, 0.08]
        self._hand_to(hand, retreat, tol=0.02, timeout=2, speed=0.1)
        self._tick(1.0)
        position = self.d.xpos[body].copy()
        stable = float(np.linalg.norm(self.d.cvel[body])) < 0.1
        ok = bool(np.all(position >= lo) and np.all(position <= hi) and stable
                  and self.s.checks.fell_at is None)
        return dict(ok=ok, object_position=position.tolist(), stable=stable,
                    fell=self.s.checks.fell_at)

    def pick(self, body_name, retries=1):
        """Grasp a free object: stand at it, hand above, descend, close the fingers, lift,
        and prove the grasp by the object's own motion (it must rise with the hand)."""
        b, p = self._body_pos(body_name)
        stand, face = self._standoff(p, 0.42)
        r = self._goto(stand, face)
        if not r.get("arrived"):
            return dict(ok=False, stage="goto", **r)
        for attempt in range(retries + 1):
            b, p = self._body_pos(body_name)
            i = self._hand(p)
            self.s.set_grip(i, 0.0)
            self._hand_to(i, p + [0, 0, 0.14], tol=0.03)
            self._hand_to(i, p + [0, 0, 0.015], tol=0.015, speed=0.12)
            self.s.set_grip(i, 1.0)
            self._tick(0.7)
            z0 = float(self.d.xpos[b][2])
            self._hand_to(i, p + [0, 0, 0.22], tol=0.03, speed=0.15)
            lifted = float(self.d.xpos[b][2]) - z0
            if lifted > 0.08:
                return dict(ok=True, hand=i, lifted_m=round(lifted, 3), attempts=attempt + 1,
                            fell=self.s.checks.fell_at)
            self.s.set_grip(i, 0.0)
            self._tick(0.4)
        self._hands_home()
        return dict(ok=False, stage="grasp", lifted_m=round(lifted, 3), attempts=retries + 1,
                    fell=self.s.checks.fell_at)

    def place(self, body_name, where, drop_from=0.30, region=0.25):
        """Carry the held object to `where` (xy or xyz) and release it above; success =
        the object's own final position inside `region` of the target."""
        tgt = np.asarray(where, float)
        if tgt.shape[0] == 2:
            tgt = np.r_[tgt, 0.0]
        b, _ = self._body_pos(body_name)
        i = self._hand(self.d.xpos[b])
        stand, face = self._standoff(tgt, 0.48)
        r = self._goto(stand, face, speed=0.35)
        if not r.get("arrived"):
            return dict(ok=False, stage="goto", **r)
        held = float(self.d.xpos[b][2]) > 0.15
        self._hand_to(i, tgt + [0, 0, max(drop_from, 0.25)], tol=0.03, speed=0.15)
        self.s.set_grip(i, 0.0)
        self._tick(1.0)
        self._hands_home()
        err = float(np.linalg.norm(self.d.xpos[b][:2] - tgt[:2]))
        return dict(ok=err <= region, was_held_on_arrival=held, offset_m=round(err, 3),
                    fell=self.s.checks.fell_at)

    def drop(self, body_name=None):
        """Open every gripper where it is."""
        for i in range(len(self.s.hands)):
            self.s.set_grip(i, 0.0)
        self._tick(0.5)
        return dict(ok=True)

    def clear(self, body_names, where, region=0.25):
        """Pick each object and place it at `where`; continues past failures and reports
        each object's outcome."""
        out = {}
        for name in body_names:
            got = self.pick(name)
            if not got.get("ok"):
                out[name] = dict(picked=False, **got)
                continue
            put = self.place(name, where, region=region)
            out[name] = dict(picked=True, placed=bool(put.get("ok")), **put)
            if self.s.checks.fell_at is not None:
                break
        done = sum(1 for v in out.values() if v.get("placed"))
        return dict(ok=done == len(body_names), cleared=done, of=len(body_names), detail=out)
