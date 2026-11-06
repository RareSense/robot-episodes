"""Mission planning for the scripted household demo: where to walk, what to open.

The plan is derived from the environment alone: a walkability grid probed with a
robot-sized capsule against the compiled environment MJCF (so walls, furniture and
closed drawers block, doors are passable because the mission opens them), floor
levels found from the grid, A* routes between them, and a tour over the passage
doors, drawers, cabinet doors, loose props and stairs the brief asked for. The output
is plain JSON the controller executes; nothing here is learned or robot-specific.
"""

import heapq
import math

import mujoco
import numpy as np

PITCH = 0.12
STEP_HEIGHT = 0.22
ROBOT_MARGIN = 0.02
MISSION_TIME_LIMIT_S = 150.0
WALK_SPEED = 0.55
STAIR_SPEED = 0.3
TIDY_TOKENS = ("table", "desk", "worktop", "counter", "bench", "shelf", "dresser", "unit")


def _tokens(name):
    return {t for t in "".join(c if c.isalnum() else " " for c in name.lower()).split() if t}


def _tall_leaf(obj):
    if obj.body is None or obj.body.motion != "hinge":
        return False
    size = obj.mesh.bounds[1] - obj.mesh.bounds[0]
    return size[2] > 1.6 and max(size[0], size[1]) > 0.6 and abs(obj.body.axis[2]) > 0.9


def _handle_point(obj):
    """A grasp point on a hinged leaf: the region farthest from the hinge line at
    about handle height (or mid-height for cabinet doors and lids)."""
    body = obj.body
    origin, axis = np.asarray(body.origin, float), np.asarray(body.axis, float)
    v = obj.mesh.vertices
    rel = v - origin
    along = rel @ axis
    radial = rel - np.outer(along, axis)
    dist = np.linalg.norm(radial, axis=1)
    far = dist >= 0.85 * dist.max()
    zmin, zmax = v[:, 2].min(), v[:, 2].max()
    if zmax - zmin > 1.5:
        band = far & (v[:, 2] > zmin + 0.85) & (v[:, 2] < zmin + 1.15)
        if band.sum() >= 3:
            far = band
    point = v[far].mean(axis=0)
    # Pull the point slightly back toward the hinge so the hand sits on the leaf.
    return point - 0.03 * (point - origin) / max(np.linalg.norm(point - origin), 1e-6)


def _swing_direction(obj, point):
    body = obj.body
    axis = np.asarray(body.axis, float)
    sign = 1.0 if body.limits[1] > 1e-6 else -1.0
    velocity = sign * np.cross(axis, point - np.asarray(body.origin, float))
    norm = np.linalg.norm(velocity)
    return velocity / norm if norm > 1e-9 else np.array([1.0, 0, 0])


def _rotate(point, origin, axis, angle):
    axis = axis / np.linalg.norm(axis)
    rel = point - origin
    c, s = math.cos(angle), math.sin(angle)
    return origin + rel * c + np.cross(axis, rel) * s + axis * (axis @ rel) * (1 - c)


class NavGrid:
    """Walkable cells on every floor level, probed against the environment model."""

    def __init__(self, environment_xml, scene, radius, height):
        from .mujoco_environment import disable_asset_cache

        disable_asset_cache()
        spec = mujoco.MjSpec.from_file(str(environment_xml))
        probe = spec.worldbody.add_body(name="nova_nav_probe", mocap=True)
        # Two probes: a shin capsule from the floor (so plinth gaps and low bars still
        # block) and a body capsule above step height (so treads and thresholds pass).
        probe.add_geom(
            name="nova_nav_probe",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[radius, 0, 0],
            fromto=[0, 0, STEP_HEIGHT, 0, 0, max(height - 0.05, STEP_HEIGHT + 0.3)],
            contype=1,
            conaffinity=1,
        )
        probe.add_geom(
            name="nova_nav_shin",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[max(radius - 0.08, 0.12), 0, 0],
            fromto=[0, 0, 0.08, 0, 0, STEP_HEIGHT + 0.02],
            contype=1,
            conaffinity=1,
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.probe_body = self.model.body("nova_nav_probe").id
        self.scene = scene
        self.radius = radius
        self.height = height
        movable = {}
        for index, obj in enumerate(scene.objects):
            if obj.body is not None:
                movable[self.model.body(f"object_{index:04d}_body").id] = obj
        self.movable = movable
        self.stair_geoms = set()
        self.floor_geoms = set()
        for index, obj in enumerate(scene.objects):
            if obj.body is None and obj.collidable:
                size = obj.mesh.bounds[1] - obj.mesh.bounds[0]
                if size[0] * size[1] >= 1.5 and size[2] <= 0.45:
                    name = f"object_{index:04d}"
                    for g in range(self.model.ngeom):
                        gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                        if gname == name or gname.startswith(name + "_"):
                            self.floor_geoms.add(g)
            if obj.body is None and (
                obj.semantic_class in {"stair_step", "stairs", "landing", "ramp", "step"}
                or "stair" in obj.name.lower()
                or "step" in obj.name.lower()
                or "landing" in obj.name.lower()
            ):
                name = f"object_{index:04d}"
                for g in range(self.model.ngeom):
                    gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                    if gname == name or gname.startswith(name + "_"):
                        self.stair_geoms.add(g)
        static = [o.mesh for o in scene.objects if o.body is None and o.collidable]
        lo = np.min([m.bounds[0] for m in static], axis=0)
        hi = np.max([m.bounds[1] for m in static], axis=0)
        self.origin = lo[:2] - PITCH
        self.shape = (
            int(math.ceil((hi[0] - lo[0]) / PITCH)) + 3,
            int(math.ceil((hi[1] - lo[1]) / PITCH)) + 3,
        )
        self.top = float(hi[2]) + 1.0
        self.levels = self._levels(static)
        self.nodes = {}  # (i, j, k) -> (x, y, z)
        self.doors_at = {}  # node -> object name of the hinge leaf occupying it
        self._probe_all()

    def _levels(self, meshes):
        """Storey heights: tops of large horizontal surfaces. Each contributing object
        must itself offer at least 4 m² of upward-facing area at that height (a slab or
        floor boards), so wall tops and worktops never form a level."""
        area = {}
        for mesh in meshes:
            up = mesh.face_normals[:, 2] > 0.9
            if not up.any():
                continue
            centers = mesh.triangles_center[up]
            weights = mesh.area_faces[up]
            bins = np.round(centers[:, 2] / 0.05).astype(int)
            per_object = {}
            for b, w in zip(bins, weights, strict=True):
                per_object[b] = per_object.get(b, 0.0) + float(w)
            for b, a in per_object.items():
                if a >= 4.0:
                    area[b] = area.get(b, 0.0) + a
        levels = sorted(b * 0.05 for b in area)
        merged = []
        for z in levels:
            if not merged or z - merged[-1] > 1.2:
                merged.append(z)
        return merged or [0.0]

    def _cell_xy(self, i, j):
        return self.origin[0] + i * PITCH, self.origin[1] + j * PITCH

    def _ray_down(self, x, y, z):
        geomid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(
            self.model,
            self.data,
            np.array([x, y, z]),
            np.array([0.0, 0.0, -1.0]),
            None,
            1,
            self.probe_body,
            geomid,
        )
        return (z - dist, int(geomid[0])) if dist >= 0 else (None, -1)

    def _clearance(self, x, y, z, on_stair):
        """None if blocked by static geometry; else the name of a door leaf standing in
        the cell, or '' when free."""
        self.data.mocap_pos[0] = [x, y, z]
        mujoco.mj_forward(self.model, self.data)
        door = ""
        shin = self.model.geom("nova_nav_shin").id
        for c in self.data.contact[: self.data.ncon]:
            bodies = (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2])
            if self.probe_body not in bodies:
                continue
            mine = c.geom1 if bodies[0] == self.probe_body else c.geom2
            other = bodies[0] if bodies[1] == self.probe_body else bodies[1]
            other_geom = c.geom2 if bodies[0] == self.probe_body else c.geom1
            if mine == shin and (on_stair or other_geom in self.stair_geoms):
                continue  # the shin brushes the next riser on a staircase
            if other == 0:
                return None
            obj = self.movable.get(int(other))
            if obj is None:
                return None
            if obj.body.motion == "hinge":
                door = obj.name
            elif obj.body.motion == "slide":
                return None
        return door

    def _probe_all(self):
        self.data.mocap_pos[0] = [1e6, 1e6, 1e6]
        mujoco.mj_forward(self.model, self.data)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                x, y = self._cell_xy(i, j)
                found = []
                # Walk the ray chain from just under each ceiling down to the floor.
                # A cell is walkable only when nothing static (except stair treads)
                # sits between its floor and the robot's head: the base board of a
                # wardrobe or the underside of a table both disqualify it.
                tops = [*self.levels[1:], self.top]
                for level, ceiling in zip(self.levels, tops, strict=True):
                    chain = []
                    z, geom = self._ray_down(x, y, ceiling - 0.05)
                    while z is not None and z >= level - 0.05 and len(chain) < 12:
                        chain.append((float(z), int(geom)))
                        z, geom = self._ray_down(x, y, z - 0.01)
                    for hit_index, (z, geom) in enumerate(chain):
                        movable = int(self.model.geom_bodyid[geom]) in self.movable
                        on_floor = z <= level + 0.12 and geom in self.floor_geoms
                        on_stair = geom in self.stair_geoms
                        if movable or not (on_floor or on_stair):
                            continue
                        above = [
                            (az, ag)
                            for az, ag in chain[:hit_index]
                            if az > z + 0.02
                            and az < z + self.height
                            and ag not in self.stair_geoms
                            and int(self.model.geom_bodyid[ag]) not in self.movable
                        ]
                        if above:
                            continue  # enclosed: inside furniture or under a worktop
                        if not any(abs(z - f) < 0.05 for f in found):
                            found.append(z)
                for k, z in enumerate(sorted(found)):
                    on_stair = not any(abs(z - lv) <= 0.12 for lv in self.levels)
                    door = self._clearance(x, y, z, on_stair)
                    if door is None:
                        continue
                    self.nodes[(i, j, k)] = (x, y, float(z))
                    if door:
                        self.doors_at[(i, j, k)] = door
        self.data.mocap_pos[0] = [1e6, 1e6, 1e6]
        mujoco.mj_forward(self.model, self.data)

    def nearest(self, point, max_distance=0.6, level=None):
        """Closest walkable cell in plan view. `level` restricts to one floor height,
        'any' ignores height, None keeps cells within a step of the point's z."""
        best, best_d = None, max_distance
        px, py, pz = point
        for node, (x, y, z) in self.nodes.items():
            if level == "any":
                pass
            elif level is not None:
                if abs(z - level) > 0.3:
                    continue
            elif abs(z - pz) > 0.35:
                continue
            d = math.hypot(x - px, y - py)
            if d < best_d:
                best, best_d = node, d
        return best

    def neighbours(self, node):
        i, j, k = node
        x, y, z = self.nodes[node]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                for kk in range(4):
                    other = (i + di, j + dj, kk)
                    if other not in self.nodes:
                        continue
                    ox, oy, oz = self.nodes[other]
                    if abs(oz - z) <= STEP_HEIGHT:
                        cost = math.hypot(ox - x, oy - y) + 2.0 * abs(oz - z)
                        yield other, cost

    def route(self, start, goal):
        if start is None or goal is None:
            return None
        gx, gy, gz = self.nodes[goal]
        frontier = [(0.0, start)]
        came, cost = {start: None}, {start: 0.0}
        while frontier:
            _, node = heapq.heappop(frontier)
            if node == goal:
                break
            for other, step in self.neighbours(node):
                new = cost[node] + step
                if new < cost.get(other, float("inf")):
                    cost[other] = new
                    came[other] = node
                    x, y, z = self.nodes[other]
                    heapq.heappush(frontier, (new + math.hypot(gx - x, gy - y), other))
        if goal not in came:
            return None
        path, node = [], goal
        while node is not None:
            path.append(node)
            node = came[node]
        path.reverse()
        return path

    def region_area(self, start, blocked, limit):
        """Walkable area (m²) connected to `start` without entering `blocked` cells,
        stopping early once `limit` is exceeded."""
        seen, frontier, area = {start}, [start], 0.0
        cell = PITCH * PITCH
        while frontier and area < limit:
            node = frontier.pop()
            area += cell
            for other, _ in self.neighbours(node):
                if other not in seen and other not in blocked:
                    seen.add(other)
                    frontier.append(other)
        return area

    def headroom(self, x, y, floor):
        """Height of the first static surface above (x, y, floor + 0.3)."""
        geomid = np.zeros(1, dtype=np.int32)
        self.data.mocap_pos[0] = [1e6, 1e6, 1e6]
        mujoco.mj_forward(self.model, self.data)
        dist = mujoco.mj_ray(
            self.model,
            self.data,
            np.array([x, y, floor + 0.3]),
            np.array([0.0, 0.0, 1.0]),
            None,
            1,
            self.probe_body,
            geomid,
        )
        while dist >= 0 and int(self.model.geom_bodyid[geomid[0]]) in self.movable:
            z = floor + 0.3 + dist + 0.01
            dist2 = mujoco.mj_ray(
                self.model,
                self.data,
                np.array([x, y, z]),
                np.array([0.0, 0.0, 1.0]),
                None,
                1,
                self.probe_body,
                geomid,
            )
            if dist2 < 0:
                return 10.0
            dist = z - floor - 0.3 + dist2
        return 10.0 if dist < 0 else float(0.3 + dist)

    def walk_distance(self, start, goal, blocked, limit):
        """Shortest walk from start to goal avoiding `blocked`, or None if it exceeds
        `limit` metres (or is impossible)."""
        frontier = [(0.0, start)]
        cost = {start: 0.0}
        while frontier:
            c, node = heapq.heappop(frontier)
            if node == goal:
                return c
            if c > limit:
                return None
            for other, step in self.neighbours(node):
                if other in blocked:
                    continue
                new = c + step
                if new < cost.get(other, float("inf")):
                    cost[other] = new
                    heapq.heappush(frontier, (new, other))
        return None

    def level_of(self, node):
        z = self.nodes[node][2]
        return min(self.levels, key=lambda f: abs(f - z))


def _polyline(grid, nodes):
    points = [grid.nodes[n] for n in nodes]
    # Drop collinear interior points so the base does not jitter on a lattice.
    kept = [points[0]]
    for a, b, c in zip(points, points[1:], points[2:], strict=False):
        va = np.array(b[:2]) - np.array(a[:2])
        vb = np.array(c[:2]) - np.array(b[:2])
        if abs(va[0] * vb[1] - va[1] * vb[0]) > 1e-6 or abs(b[2] - a[2]) > 1e-3:
            kept.append(b)
    if len(points) > 1:
        kept.append(points[-1])
    return [[float(v) for v in p] for p in kept]


def _path_length(points):
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1]) + abs(b[2] - a[2])
        for a, b in zip(points, points[1:], strict=False)
    )


class MissionPlanner:
    def __init__(self, scene, environment_xml, robot, start_xyz):
        self.scene = scene
        self.robot = robot
        self.grid = NavGrid(
            environment_xml, scene, min(robot["radius"], 0.32) + ROBOT_MARGIN, robot["height"]
        )
        self.start = start_xyz
        self.steps = []
        self.skipped = []
        self.time = 0.0
        self.node = self.grid.nearest(start_xyz, max_distance=1.5, level="any")
        self.position = list(self.grid.nodes[self.node]) if self.node else list(start_xyz)
        self.body_name = {
            obj.name: f"nova_environment_object_{i:04d}_body"
            for i, obj in enumerate(scene.objects)
            if obj.body is not None
        }

    def _is_passage(self, obj):
        """A tall hinged leaf hung in a wall-height opening with standing room on both
        sides: a room door. A wardrobe or tall-cupboard door hangs in a carcass whose
        top is below the ceiling and whose interior has no headroom cells."""
        if not _tall_leaf(obj):
            return False
        handle = _handle_point(obj)
        normal = _swing_direction(obj, handle)
        floor = obj.mesh.bounds[0][2]
        # Walkable floor at 0.25, 0.6 and 1.2 m out on both sides, on this floor. The
        # 0.25 m probe sits inside the opening: a threshold for a room door, the carcass
        # interior (never walkable) for a cabinet door.
        for sign in (1.0, -1.0):
            for reach in (0.25, 0.6, 1.2):
                probe = handle + sign * reach * normal
                probe[2] = floor
                if self.grid.nearest(probe, max_distance=0.15, level=floor) is None:
                    return False
        # The surround: static geometry within 0.35 m of the hinge line must reach at
        # least 2.2 m above the floor (a wall or door frame), not stop at cabinet height.
        origin = np.asarray(obj.body.origin, float)
        surround_top = floor
        for other in self.scene.objects:
            if other.body is not None or not other.collidable:
                continue
            lo, hi = other.mesh.bounds
            near_x = lo[0] - 0.35 <= origin[0] <= hi[0] + 0.35
            near_y = lo[1] - 0.35 <= origin[1] <= hi[1] + 0.35
            if near_x and near_y and hi[2] - lo[2] > 1.0:
                surround_top = max(surround_top, float(hi[2]))
        if surround_top - floor < 2.2:
            return False
        # Headroom: from the far-side standing point the ray up must not hit anything
        # below 2.2 m (a wardrobe's top shelf/carcass top is lower than that).
        for sign in (1.0, -1.0):
            probe = handle + sign * 0.6 * normal
            if self.grid.headroom(probe[0], probe[1], floor) < 2.2:
                return False
        return True

    def _walk(self, target_point, label, level=None):
        if level is None:
            level = self.grid.level_of(self.node)
        goal = self.grid.nearest(target_point, max_distance=0.75, level=level)
        if goal is None:
            self.skipped.append(f"{label}: no walkable cell near the standoff point")
            return False
        route = self.grid.route(self.node, goal)
        if route is None:
            self.skipped.append(f"{label}: no route from the robot's position")
            return False
        points = _polyline(self.grid, route)
        stairs = abs(points[-1][2] - points[0][2]) > 0.5
        speed = STAIR_SPEED if stairs else WALK_SPEED
        duration = max(0.5, _path_length(points) / speed)
        if self.time + duration > MISSION_TIME_LIMIT_S:
            self.skipped.append(f"{label}: mission time limit reached")
            return False
        self.steps.append(
            {
                "kind": "walk",
                "label": label,
                "path": points,
                "speed": speed,
                "duration": round(duration, 3),
                "stairs": stairs,
            }
        )
        self.time += duration
        self.node = goal
        self.position = list(self.grid.nodes[goal])
        return True

    def _add(self, step):
        if self.time + step["duration"] > MISSION_TIME_LIMIT_S:
            self.skipped.append(f"{step['label']}: mission time limit reached")
            return False
        self.steps.append(step)
        self.time += step["duration"]
        return True

    def door(self, obj, passage):
        body = obj.body
        handle = _handle_point(obj)
        swing = _swing_direction(obj, handle)
        # Prefer the PULL side: the leaf swings away from the robot there, so the
        # scripted base never ploughs it open before the hand reaches the handle. On
        # the push side stand back a full arm's reach plus the leaf's swept depth.
        normal = swing / np.linalg.norm(swing)
        floor = self.grid.level_of(self.node)
        leaf_depth = float(np.linalg.norm((handle - np.asarray(body.origin))[:2]))
        sides = [(normal, "pull", 0.7), (-normal, "push", 0.45 + leaf_depth)]
        chosen = None
        for u, mode, reach in sides:
            standoff = handle + reach * u
            standoff[2] = floor
            if self.grid.nearest(standoff, max_distance=0.45, level=floor) is not None:
                chosen = (u, mode, standoff)
                break
        if chosen is None:
            self.skipped.append(f"{obj.name}: no standing room on either side of the leaf")
            return
        u, mode, standoff = chosen
        if not self._walk(standoff, f"walk to {obj.name}"):
            return
        angle = 0.85 * (body.limits[1] if body.limits[1] > 1e-6 else body.limits[0])
        duration = 2.5 + 1.5 * abs(angle)
        step = {
            "kind": "hinge",
            "label": f"{'open' if passage else 'open cabinet'} {obj.name} ({mode})",
            "object": obj.name,
            "body": self.body_name[obj.name],
            "origin": [float(v) for v in body.origin],
            "axis": [float(v) for v in body.axis],
            "handle": [float(v) for v in handle],
            "side": [float(v) for v in u],
            "angle": float(angle),
            "duration": round(duration, 3),
            "target": "open",
        }
        if not self._add(step):
            return
        opened_handle = _rotate(handle, np.asarray(body.origin), np.asarray(body.axis), angle)
        if passage:
            # Walk through: aim 1.2 m beyond the doorway on the far side, then stand
            # off the OPEN leaf's handle on that far side to swing it shut.
            beyond = handle - 1.2 * u
            beyond[2] = floor
            if self._walk(beyond, f"walk through {obj.name}", level=floor):
                open_normal = _swing_direction(obj, opened_handle)
                far = opened_handle - 0.7 * open_normal
                far[2] = floor
                if self._walk(far, f"turn to close {obj.name}", level=floor):
                    self._add(
                        {
                            **step,
                            "label": f"close {obj.name}",
                            "side": [float(-v) for v in u],
                            "target": "closed",
                        }
                    )
        else:
            self._add({**step, "label": f"close {obj.name}", "target": "closed"})

    def drawer(self, obj):
        body = obj.body
        axis = np.asarray(body.axis, float)
        v = obj.mesh.vertices
        along = (v - np.asarray(body.origin)) @ axis
        front = v[along >= along.max() - 0.03].mean(axis=0)
        floor = self.grid.level_of(self.node)
        standoff = front + 0.65 * axis
        standoff[2] = floor
        if not self._walk(standoff, f"walk to {obj.name}", level=floor):
            return
        travel = 0.85 * body.limits[1]
        step = {
            "kind": "slide",
            "label": f"pull {obj.name}",
            "object": obj.name,
            "body": self.body_name[obj.name],
            "axis": [float(a) for a in axis],
            "pull": [float(p) for p in front],
            "travel": float(travel),
            "duration": round(2.0 + 4.0 * travel, 3),
            "target": "open",
        }
        if self._add(step):
            self._add({**step, "label": f"push {obj.name} shut", "target": "closed"})

    def prop(self, obj, destination):
        body = obj.body
        centre = np.asarray(body.origin, float) + np.asarray(body.center_of_mass, float)
        floor = self.grid.level_of(self.node)
        standoff = centre.copy()
        standoff[2] = floor
        approach = self.grid.nearest(standoff, max_distance=0.8, level=floor)
        if approach is None:
            self.skipped.append(f"{obj.name}: no standing room next to it")
            return
        if not self._walk(self.grid.nodes[approach], f"walk to {obj.name}"):
            return
        size = obj.mesh.bounds[1] - obj.mesh.bounds[0]
        if not self._add(
            {
                "kind": "pick",
                "label": f"pick up {obj.name}",
                "object": obj.name,
                "body": self.body_name[obj.name],
                "grasp": [float(c) for c in centre],
                "lift": 0.25,
                "duration": 3.0,
            }
        ):
            return
        point, label = destination
        standoff = np.asarray(point, float).copy()
        standoff[2] = floor
        if not self._walk(standoff, f"carry {obj.name} to {label}", level=floor):
            return
        self._add(
            {
                "kind": "place",
                "label": f"put {obj.name} on {label}",
                "object": obj.name,
                "body": self.body_name[obj.name],
                "target": [float(point[0]), float(point[1]), float(point[2] + size[2] / 2)],
                "duration": 3.0,
            }
        )

    def tidy_destination(self):
        best = None
        for obj in self.scene.objects:
            if obj.body is not None or not obj.collidable:
                continue
            tokens = _tokens(obj.name) | _tokens(obj.semantic_class)
            if not tokens & set(TIDY_TOKENS):
                continue
            top = obj.mesh.bounds[1][2]
            if not 0.5 < top - self.grid.level_of(self.node) < 1.2:
                continue
            centre = obj.mesh.bounds.mean(axis=0)
            point = np.array([centre[0], centre[1], top])
            d = math.hypot(point[0] - self.position[0], point[1] - self.position[1])
            if best is None or d < best[0]:
                best = (d, point, obj.name)
        if best is None:
            goal = self.scene.goal_position
            return (np.array([goal[0], goal[1], self.grid.level_of(self.node)]), "the goal marker")
        return best[1], best[2]

    def stairs(self):
        here = self.grid.level_of(self.node)
        others = [lv for lv in self.grid.levels if abs(lv - here) > 0.5]
        if not others:
            self.skipped.append("stairs: the environment has a single floor level")
            return
        target_level = min(others, key=lambda lv: abs(lv - here))
        candidates = [
            (n, xyz) for n, xyz in self.grid.nodes.items() if abs(xyz[2] - target_level) < 0.15
        ]
        if not candidates:
            self.skipped.append("stairs: no walkable cell on the other level")
            return
        cx = np.mean([xyz[0] for _, xyz in candidates])
        cy = np.mean([xyz[1] for _, xyz in candidates])
        candidates.sort(key=lambda c: math.hypot(c[1][0] - cx, c[1][1] - cy))
        before = self.node
        up = target_level > here
        for node, xyz in candidates[:: max(1, len(candidates) // 12)][:12]:
            if self.grid.route(self.node, node) is not None:
                if self._walk(xyz, "go upstairs" if up else "go downstairs", level=target_level):
                    self._walk(
                        self.grid.nodes[before],
                        "come back downstairs" if up else "come back upstairs",
                        level=here,
                    )
                return
        self.skipped.append("stairs: no route between the floor levels")

    def plan(self):
        scene = self.scene
        if self.node is None:
            return {
                "robot": self.robot,
                "steps": [],
                "skipped": ["the robot's spawn position is not on a walkable cell"],
                "levels": self.grid.levels,
                "walkable_cells": len(self.grid.nodes),
            }

        def distance(obj):
            c = obj.mesh.bounds.mean(axis=0)
            return math.hypot(c[0] - self.position[0], c[1] - self.position[1])

        passages = {o.name for o in scene.objects if self._is_passage(o)}
        doors = sorted((o for o in scene.objects if o.name in passages), key=distance)
        for obj in doors[:2]:
            self.door(obj, passage=True)
        self.stairs()
        drawers = sorted(
            (o for o in scene.objects if o.body is not None and o.body.motion == "slide"),
            key=distance,
        )
        for obj in drawers[:2]:
            self.drawer(obj)
        cabinets = sorted(
            (
                o
                for o in scene.objects
                if o.body is not None and o.body.motion == "hinge" and o.name not in passages
            ),
            key=distance,
        )
        for obj in cabinets[:2]:
            self.door(obj, passage=False)
        props = sorted(
            (
                o
                for o in scene.objects
                if o.body is not None
                and o.body.motion == "free"
                and o.body.mass <= 5.0
                and -0.2 < o.body.origin[2] - self.grid.level_of(self.node) < 1.2
            ),
            key=distance,
        )
        destination = self.tidy_destination()
        for obj in props[:2]:
            self.prop(obj, destination)
        return {
            "robot": self.robot,
            "steps": self.steps,
            "skipped": self.skipped,
            "levels": self.grid.levels,
            "walkable_cells": len(self.grid.nodes),
            "planned_duration_s": round(self.time, 2),
            "time_limit_s": MISSION_TIME_LIMIT_S,
        }
