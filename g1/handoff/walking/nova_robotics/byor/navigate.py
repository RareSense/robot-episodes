"""Floor navigation for an imported robot: goto(session, target) plans on the scene's own
collision geometry and walks the robot there through its locomotion policy.

The chain stays physically honest end to end: the planner only ever produces walking
velocity commands (vx, vy, yaw rate), which the policy turns into joint targets for the
torque-limited motors. Nothing steers or slides the base; if the robot cannot physically
follow the path, goto reports failure instead of forcing it.

- occupancy(): a boolean grid of everything in the scene a walking robot could hit,
  rasterised from the world-frame bounding boxes of the static collision geoms, inflated
  by the robot's own radius so the planner treats the robot as a point.
- plan(): A* over the grid (8-connected), then line-of-sight shortcutting.
- PathFollower: pure pursuit over the smoothed path; turns in place for large heading
  errors, walks the lookahead point otherwise.
- goto(): plan + follow, stepping the TeleopSession until arrival, fall or timeout.
"""
from __future__ import annotations

import heapq
import math

import mujoco
import numpy as np


def _world_aabbs(m, d, exclude_bodies):
    """World-frame (min_xyz, max_xyz) of every collision geom outside `exclude_bodies`.
    Planes are skipped (they are the floor); everything else uses its local AABB rotated
    into the world, which is conservative and cheap."""
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b in exclude_bodies or b == 0 and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        centre = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
        half = np.abs(R) @ m.geom_aabb[g, 3:]
        out.append((centre - half, centre + half))
    return out


def occupancy(m, d, exclude_bodies, resolution=0.05, inflate=0.30, zband=(0.10, 1.6),
              margin=1.0, include=()):
    """Boolean obstacle grid of the walkable plane. A geom blocks a cell when its vertical
    extent overlaps `zband` (what a walking robot's body sweeps); `inflate` grows every
    obstacle by the robot's radius so the planner can treat the robot as a point. The grid
    always covers the `include` points (the robot and its goal) plus the margin."""
    boxes = [(lo, hi) for lo, hi in _world_aabbs(m, d, exclude_bodies)
             if hi[2] > zband[0] and lo[2] < zband[1]]
    if not boxes:
        raise ValueError("no obstacles found: check exclude_bodies")
    pts = ([b[0][:2] for b in boxes] + [b[1][:2] for b in boxes]
           + [np.asarray(p[:2], float) for p in include])
    lo = np.min(pts, axis=0) - margin
    hi = np.max(pts, axis=0) + margin
    nx, ny = (np.ceil((hi - lo) / resolution)).astype(int)
    grid = np.zeros((nx, ny), bool)
    for blo, bhi in boxes:
        i0 = np.floor((blo[:2] - inflate - lo) / resolution).astype(int)
        i1 = np.ceil((bhi[:2] + inflate - lo) / resolution).astype(int)
        i0, i1 = np.clip(i0, 0, [nx, ny]), np.clip(i1, 0, [nx, ny])
        grid[i0[0]:i1[0], i0[1]:i1[1]] = True
    return grid, lo, resolution


def _cell(xy, lo, res, grid):
    c = np.floor((np.asarray(xy[:2]) - lo) / res).astype(int)
    return (int(np.clip(c[0], 0, grid.shape[0] - 1)), int(np.clip(c[1], 0, grid.shape[1] - 1)))


def _line_free(grid, a, b):
    """Every cell on the segment a-b is free (sampled at sub-cell steps)."""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1)) * 2
    for t in np.linspace(0, 1, n + 1):
        i = int(round(a[0] + t * (b[0] - a[0])))
        j = int(round(a[1] + t * (b[1] - a[1])))
        if grid[i, j]:
            return False
    return True


def _nearest_free(grid, c, radius=40):
    """`c` itself, or the nearest free cell (a goal next to furniture lands inside its
    inflated footprint; stand at the closest reachable spot instead of failing)."""
    if not grid[c]:
        return c
    for r in range(1, radius):
        best = None
        for di in range(-r, r + 1):
            for dj in (-r, r) if abs(di) < r else range(-r, r + 1):
                i, j = c[0] + di, c[1] + dj
                if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and not grid[i, j]:
                    dd = di * di + dj * dj
                    if best is None or dd < best[0]:
                        best = (dd, (i, j))
        if best:
            return best[1]
    raise ValueError("no free cell near the goal")


def plan(grid, lo, res, start_xy, goal_xy):
    """A* (8-connected) + line-of-sight shortcutting -> waypoints in world metres."""
    start = _nearest_free(grid, _cell(start_xy, lo, res, grid))
    goal = _nearest_free(grid, _cell(goal_xy, lo, res, grid))
    h = lambda c: math.hypot(c[0] - goal[0], c[1] - goal[1])  # noqa: E731
    open_q = [(h(start), 0.0, start)]
    came, gsc = {start: None}, {start: 0.0}
    steps = [(i, j, math.hypot(i, j)) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
    while open_q:
        _, g, c = heapq.heappop(open_q)
        if c == goal:
            break
        if g > gsc.get(c, np.inf):
            continue
        for di, dj, w in steps:
            n = (c[0] + di, c[1] + dj)
            if not (0 <= n[0] < grid.shape[0] and 0 <= n[1] < grid.shape[1]) or grid[n]:
                continue
            if di and dj and (grid[c[0] + di, c[1]] or grid[c[0], c[1] + dj]):
                continue  # a point path must not cut through an obstacle corner
            ng = g + w
            if ng < gsc.get(n, np.inf):
                gsc[n], came[n] = ng, c
                heapq.heappush(open_q, (ng + h(n), ng, n))
    if goal not in came:
        raise ValueError("no path: the goal is walled off from the robot")
    cells = [goal]
    while came[cells[-1]] is not None:
        cells.append(came[cells[-1]])
    cells.reverse()
    short = [cells[0]]
    i = 0
    while i < len(cells) - 1:                      # furthest visible cell from here
        j = len(cells) - 1
        while j > i + 1 and not _line_free(grid, cells[i], cells[j]):
            j -= 1
        short.append(cells[j])
        i = j
    return [lo + (np.array(c) + 0.5) * res for c in short]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PathFollower:
    """Pure pursuit over the waypoints: walk towards a lookahead point on the path, turn in
    place when the heading error is large, finish by turning to `final_yaw` if given."""

    def __init__(self, path, speed=0.5, lookahead=0.5, goal_tol=0.30, final_yaw=None):
        self.path = [np.asarray(p, float) for p in path]
        self.speed, self.lookahead, self.goal_tol = speed, lookahead, goal_tol
        self.final_yaw = final_yaw
        self._i = 0

    def command(self, xy, yaw):
        """-> ((vx, vy, wz), done) in the robot's heading frame."""
        xy = np.asarray(xy[:2], float)
        goal = self.path[-1]
        distance = np.linalg.norm(goal - xy)
        if distance < self.goal_tol:
            if self.final_yaw is None:
                return (0.0, 0.0, 0.0), True
            e = _wrap(self.final_yaw - yaw)
            if abs(e) < 0.15:
                return (0.0, 0.0, 0.0), True
            return (0.0, 0.0, float(np.clip(1.5 * e, -0.8, 0.8))), False
        if distance < self.lookahead and self.final_yaw is not None:
            # Near a manipulation stance, translate sideways/backwards while
            # facing the object. Circling a nearby waypoint cannot converge.
            delta = goal - xy
            cy, sy = math.cos(yaw), math.sin(yaw)
            local = np.array([cy * delta[0] + sy * delta[1],
                              -sy * delta[0] + cy * delta[1]])
            vel = np.clip(0.8 * local, -min(self.speed, 0.2), min(self.speed, 0.2))
            e = _wrap(self.final_yaw - yaw)
            return (float(vel[0]), float(vel[1]), float(np.clip(e, -0.5, 0.5))), False
        while (self._i + 1 < len(self.path)
               and np.linalg.norm(self.path[self._i] - xy) < self.lookahead):
            self._i += 1
        target = self.path[self._i]
        e = _wrap(math.atan2(*(target - xy)[::-1]) - yaw)
        wz = float(np.clip(1.5 * e, -0.8, 0.8))
        if abs(e) > 1.0:                           # face the path first
            return (0.0, 0.0, wz), False
        vx = float(np.clip(self.speed * math.cos(e), 0.15, self.speed))
        return (vx, 0.0, wz), False


def goto(session, target_xy, final_yaw=None, speed=0.5, timeout=90.0, resolution=0.05,
         inflate=0.30, on_step=None, ignore_bodies=(), goal_tol=0.30):
    """Walk the session's robot to `target_xy` (world metres). Plans on the current scene,
    then feeds the follower's commands to the walking policy at 10 Hz. Returns a report;
    raises ValueError when there is no path. `on_step(session)` runs every control tick
    (rendering hooks). `ignore_bodies` are planned through rather than around — an openable
    door the robot intends to push, never a wall."""
    if session.loco is None:
        raise ValueError("this robot has no locomotion; goto needs a walking policy")
    m, d = session.m, session.d
    qa = session.loco.qa
    grid, lo, res = occupancy(m, d, set(session.bodies) | set(ignore_bodies),
                              resolution=resolution, inflate=inflate,
                              include=(d.qpos[qa:qa + 2], target_xy))
    path = plan(grid, lo, res, d.qpos[qa:qa + 2], target_xy)
    follow = PathFollower(path, speed=speed, final_yaw=final_yaw, goal_tol=goal_tol)
    t0 = d.time
    steps_per_tick = max(1, int(round(0.1 / m.opt.timestep)))
    while True:
        q = d.qpos[qa + 3:qa + 7]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        cmd, done = follow.command(d.qpos[qa:qa + 2], yaw)
        if done:
            session.set_walk([0, 0, 0])
            break
        session.set_walk(cmd)
        session.step(steps_per_tick)
        if on_step is not None:
            on_step(session)
        if session.checks.fell_at is not None:
            return dict(arrived=False, reason="fell", at=session.checks.fell_at,
                        walked_s=round(d.time - t0, 2), waypoints=len(path))
        if d.time - t0 > timeout:
            session.set_walk([0, 0, 0])
            return dict(arrived=False, reason="timeout", walked_s=round(d.time - t0, 2),
                        waypoints=len(path))
    err = float(np.linalg.norm(np.asarray(target_xy[:2]) - d.qpos[qa:qa + 2]))
    # A blocked endpoint may have been snapped to a distant free cell. Reaching
    # that cell is not arrival at the requested manipulation stance. Allow only
    # the normal half-diagonal quantization error of the occupancy grid.
    arrived = err <= goal_tol + resolution / math.sqrt(2)
    result = dict(arrived=arrived, position_error_m=round(err, 3),
                  walked_s=round(d.time - t0, 2), waypoints=len(path))
    if not arrived:
        result["reason"] = "planned endpoint is too far from the requested goal"
    return result
