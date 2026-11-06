"""Buoyancy and drag for the water regions in this package. Copy-free, no dependencies.

None of the targets simulates a fluid. What this module does is the part that can be
done honestly from rigid-body state: while a body is inside a water region it receives
an upward force for the volume it displaces and a resisting force for the speed it
moves at. Each collision geometry is treated as a column of its own volume and its
own height, so a crate displaces a crate's worth of water, not the sphere around it.
Orientation is ignored: a tilted part is judged by its upright height.

Not modelled: waves, currents, surface tension, added mass, splash, wetting, changes
in friction, or any effect of water on sensors. Treat this as a wading model, useful
for training a robot to cross shallow water — not as a swimming simulator.

    from nova_water import MuJoCoWater
    water = MuJoCoWater(json.loads(Path("water_volumes.json").read_text()), 9.81)
    while running:
        data.xfrc_applied[:] = 0.0    # your other applied forces go here
        water.apply(model, data)      # adds to xfrc_applied; never clears it
        mujoco.mj_step(model, data)
"""

import math


class WaterField:
    """The scene's water regions and the force they put on a submerged part."""

    def __init__(self, volumes, gravity=9.81):
        self.volumes = [dict(volume) for volume in volumes]
        self.gravity = float(gravity)

    def region_at(self, x, y, z):
        for volume in self.volumes:
            x0, y0, x1, y1 = volume["bounds_m"]
            inside = x0 <= x <= x1 and y0 <= y <= y1
            if inside and volume["bed_z_m"] - 1.0 <= z <= volume["surface_z_m"]:
                return volume
        return None

    def part_force(self, position, half_height, volume_m3, velocity):
        """Force on one part: buoyancy up, drag against its motion. Newtons.

        Buoyancy is the weight of the water displaced by the submerged fraction of the
        part's volume, the part standing as a column `2 * half_height` tall. Drag is
        `-(linear + quadratic * speed) * area * velocity`, with `area` the part's mid
        cross-section (its waterplane once any of it is under): a blunt-body
        approximation that also damps the heave of a floating part.
        """
        x, y, z = position
        bottom = z - half_height
        # A part counts as in the water as soon as its LOWEST point is under the
        # surface; testing its top would give a half-submerged body no lift at all.
        region = self.region_at(x, y, bottom)
        if region is None or half_height <= 0 or volume_m3 <= 0:
            return (0.0, 0.0, 0.0)
        fraction = max(0.0, min(1.0, (region["surface_z_m"] - bottom) / (2.0 * half_height)))
        if fraction <= 0.0:
            return (0.0, 0.0, 0.0)
        lift = region["density_kg_m3"] * self.gravity * volume_m3 * fraction
        area = volume_m3 / (2.0 * half_height)
        linear = region["linear_drag"] * area
        quadratic = region["quadratic_drag"] * area
        speed = math.sqrt(sum(component * component for component in velocity))
        drag = [-(linear + quadratic * speed) * component for component in velocity]
        return (drag[0], drag[1], drag[2] + lift)


def _mujoco_geom_shape(model, geom):
    """(volume, half height) of one MuJoCo collision geom in its rest orientation:
    exact for primitives, the convex hull for meshes (which is what the geom
    collides as)."""
    kind, size = int(model.geom_type[geom]), model.geom_size[geom]
    sphere, capsule, ellipsoid, cylinder, box, mesh = 2, 3, 4, 5, 6, 7
    if kind == sphere:
        return 4.0 / 3.0 * math.pi * float(size[0]) ** 3, float(size[0])
    if kind == capsule:
        r, h = float(size[0]), float(size[1])
        return math.pi * r * r * 2 * h + 4.0 / 3.0 * math.pi * r**3, r + h
    if kind == ellipsoid:
        volume = 4.0 / 3.0 * math.pi * float(size[0]) * float(size[1]) * float(size[2])
        return volume, float(size[2])
    if kind == cylinder:
        return math.pi * float(size[0]) ** 2 * 2 * float(size[1]), float(size[1])
    if kind == box:
        return 8.0 * float(size[0]) * float(size[1]) * float(size[2]), float(size[2])
    if kind == mesh:
        mesh_id = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        points = model.mesh_vert[start : start + count]
        extents = points.max(axis=0) - points.min(axis=0)
        half_height = float(extents[2]) / 2.0
        try:
            from scipy.spatial import ConvexHull

            return float(ConvexHull(points).volume), half_height
        except Exception:  # degenerate mesh: fall back to its bounding box
            return float(extents[0] * extents[1] * extents[2]), half_height
    return 0.0, 0.0


class MuJoCoWater(WaterField):
    """Adds the force to `data.xfrc_applied`; call once per step after clearing it.

    Only contact geoms displace water: the massless visual geom every movable body
    carries (contype and conaffinity zero) would otherwise count the same body twice.
    Geom volumes are computed once per model and cached by model identity.
    """

    def __init__(self, volumes, gravity=9.81):
        super().__init__(volumes, gravity)
        self._volumes_for = None

    def _geoms(self, model):
        if self._volumes_for is None or self._volumes_for[0] is not model:
            entries = []
            for geom in range(model.ngeom):
                body = int(model.geom_bodyid[geom])
                if model.body_dofnum[body] == 0 and model.body_jntnum[body] == 0:
                    continue  # static scenery displaces nothing
                if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0:
                    continue  # visual-only geom of a body already counted through its contact
                volume, half_height = _mujoco_geom_shape(model, geom)
                entries.append((geom, body, volume, half_height))
            self._volumes_for = (model, entries)
        return self._volumes_for[1]

    def apply(self, model, data):
        for geom, body, volume, half_height in self._geoms(model):
            position = [float(value) for value in data.geom_xpos[geom]]
            velocity = [float(value) for value in data.cvel[body][3:6]]
            fx, fy, fz = self.part_force(position, half_height, volume, velocity)
            if fx or fy or fz:
                data.xfrc_applied[body][0] += fx
                data.xfrc_applied[body][1] += fy
                data.xfrc_applied[body][2] += fz

    def lift(self, model, data, body):
        """Upward water force on `body` right now, in newtons."""
        total = 0.0
        for geom, owner, volume, half_height in self._geoms(model):
            if owner == body:
                position = [float(value) for value in data.geom_xpos[geom]]
                total += self.part_force(position, half_height, volume, (0.0, 0.0, 0.0))[2]
        return total



class PyBulletWater(WaterField):
    """Applies the force through `applyExternalForce`; call once per step.

    PyBullet exposes no geom volumes, so a part's volume is estimated as 60 % of its
    axis-aligned bounding box (a box-to-sphere middle ground), and its height as that
    box's vertical extent.
    """

    def apply(self, p):
        for body in range(p.getNumBodies()):
            unique = p.getBodyUniqueId(body)
            for link in range(-1, p.getNumJoints(unique)):
                if p.getDynamicsInfo(unique, link)[0] <= 0:
                    continue  # mass 0: static scenery
                if link < 0:
                    position, _ = p.getBasePositionAndOrientation(unique)
                    velocity, _ = p.getBaseVelocity(unique)
                else:
                    state = p.getLinkState(unique, link, computeLinkVelocity=1)
                    position, velocity = state[0], state[6]
                low, high = p.getAABB(unique, link)
                extents = [high[axis] - low[axis] for axis in range(3)]
                volume = 0.6 * extents[0] * extents[1] * extents[2]
                force = self.part_force(position, extents[2] / 2.0, volume, velocity)
                if any(force):
                    p.applyExternalForce(unique, link, force, position, p.WORLD_FRAME)

    def lift(self, p, unique):
        """Upward water force on the base of `unique` right now, in newtons."""
        low, high = p.getAABB(unique, -1)
        extents = [high[axis] - low[axis] for axis in range(3)]
        position = [(low[axis] + high[axis]) / 2.0 for axis in range(3)]
        volume = 0.6 * extents[0] * extents[1] * extents[2]
        return self.part_force(position, extents[2] / 2.0, volume, (0.0, 0.0, 0.0))[2]
