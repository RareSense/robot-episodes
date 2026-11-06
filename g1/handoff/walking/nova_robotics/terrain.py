"""Heightfield terrain: one elevation grid, exact contact on every target.

A hillside, a rubble field or a river bed is not furniture. Sending it through the
convex decomposition in collision.py would smear the centimetre-scale roughness a
locomotion policy learns from, and MuJoCo would otherwise collide the whole field as
a single convex hull: a dome. Terrain therefore keeps its elevation grid all the way
to the exporters, and each engine uses its own exact primitive for it — a MuJoCo
`<hfield>`, a PyBullet `GEOM_HEIGHTFIELD`, and the triangulated surface itself for
USD/Isaac Sim and SDF/Gazebo, which collide static triangle meshes exactly. The grid
is the only authority: the visual mesh is rebuilt from it on both sides of the
author→export boundary, so what the creator sees and what the robot's feet touch are
one surface.

`Grid` is metre-native: every generator takes metres and degrees, never cell counts,
so a 6 m hill spacing stays 6 m when the resolution changes. Generators return plain
arrays; combine them with ordinary arithmetic.
"""

import math
from dataclasses import dataclass

import numpy as np
import trimesh

MIN_NODES = 4
MAX_NODES = 1025
MAX_TOTAL_NODES = 400_000


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _value_noise(shape, period_nodes, rng):
    """Smooth noise in [-1, 1] with features about `period_nodes` nodes across.

    A random lattice interpolated with smoothstep: deterministic for a seed, no
    external noise dependency, and C1 enough that a foot does not catch on a seam.
    """
    rows, cols = shape
    period = max(2.0, float(period_nodes))
    lattice_rows = int(math.ceil(rows / period)) + 2
    lattice_cols = int(math.ceil(cols / period)) + 2
    lattice = rng.uniform(-1.0, 1.0, size=(lattice_rows, lattice_cols))
    y = np.arange(rows, dtype=float) / period
    x = np.arange(cols, dtype=float) / period
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    fy, fx = _smoothstep(y - y0)[:, None], _smoothstep(x - x0)[None, :]
    c00 = lattice[np.ix_(y0, x0)]
    c01 = lattice[np.ix_(y0, x0 + 1)]
    c10 = lattice[np.ix_(y0 + 1, x0)]
    c11 = lattice[np.ix_(y0 + 1, x0 + 1)]
    top = c00 * (1 - fx) + c01 * fx
    bottom = c10 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bottom * fy


def _fractal(shape, period_nodes, octaves, seed, gain, lacunarity, ridged):
    rng = np.random.default_rng(seed)
    total = np.zeros(shape, dtype=float)
    amplitude, weight, period = 1.0, 0.0, float(period_nodes)
    for _ in range(max(1, int(octaves))):
        layer = _value_noise(shape, period, rng)
        if ridged:
            # 1 - |noise| turns smooth hills into creased ridges and gullies: the
            # crests are sharp, the valleys rounded, which is what rock reads as.
            layer = 2.0 * (1.0 - np.abs(layer)) - 1.0
        total += amplitude * layer
        weight += amplitude
        amplitude *= gain
        period = max(2.0, period / lacunarity)
    return total / max(weight, 1e-9)


@dataclass
class Grid:
    """Sampling lattice for one terrain patch, in world metres.

    `size` is the full extent (x, y) and `cell` the target node spacing; the node
    count is rounded so the patch keeps exactly the requested extent. X and Y are
    world coordinate arrays of shape (rows, cols), rows running along +Y.
    """

    size: tuple[float, float]
    cell: float = 0.1
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        size = np.asarray(self.size, dtype=float)
        if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError("Terrain size must be two positive metre extents")
        if not np.isfinite(self.cell) or self.cell <= 0:
            raise ValueError("Terrain cell spacing must be a positive number of metres")
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("Terrain center must be a finite XY position in metres")
        self.size = (float(size[0]), float(size[1]))
        self.center = (float(center[0]), float(center[1]))
        self.cols = int(round(size[0] / self.cell)) + 1
        self.rows = int(round(size[1] / self.cell)) + 1
        if not (MIN_NODES <= self.rows <= MAX_NODES and MIN_NODES <= self.cols <= MAX_NODES):
            raise ValueError(
                f"Terrain grid is {self.rows}x{self.cols} nodes; keep each axis between "
                f"{MIN_NODES} and {MAX_NODES} (raise `cell` for a larger patch)"
            )
        if self.rows * self.cols > MAX_TOTAL_NODES:
            raise ValueError("Terrain grid exceeds 400,000 nodes; use a coarser cell spacing")
        self.x = np.linspace(
            self.center[0] - self.size[0] / 2, self.center[0] + self.size[0] / 2, self.cols
        )
        self.y = np.linspace(
            self.center[1] - self.size[1] / 2, self.center[1] + self.size[1] / 2, self.rows
        )
        self.X, self.Y = np.meshgrid(self.x, self.y)

    @property
    def shape(self):
        return (self.rows, self.cols)

    @property
    def spacing(self):
        """Actual node spacing (dx, dy) after rounding to whole nodes."""
        return (self.size[0] / (self.cols - 1), self.size[1] / (self.rows - 1))

    @property
    def bounds(self):
        """(x0, y0, x1, y1) of the patch in world metres."""
        return (float(self.x[0]), float(self.y[0]), float(self.x[-1]), float(self.y[-1]))

    def _period(self, feature_m):
        if not np.isfinite(feature_m) or feature_m <= 0:
            raise ValueError("Terrain feature size must be a positive number of metres")
        dx, dy = self.spacing
        return feature_m / max(min(dx, dy), 1e-9)

    def flat(self, height=0.0):
        return np.full(self.shape, float(height))

    def hills(self, *, amplitude, feature=6.0, octaves=4, seed=0, gain=0.5, lacunarity=2.0):
        """Rolling ground: smooth fractal noise spanning about -amplitude..+amplitude."""
        field = _fractal(
            self.shape, self._period(feature), octaves, seed, gain, lacunarity, ridged=False
        )
        return float(amplitude) * field

    def ridges(self, *, amplitude, feature=10.0, octaves=5, seed=0, gain=0.55, lacunarity=2.1):
        """Rock: creased ridged-multifractal crests with rounded gullies between."""
        field = _fractal(
            self.shape, self._period(feature), octaves, seed, gain, lacunarity, ridged=True
        )
        return float(amplitude) * field

    def rubble(self, *, amplitude, patch=0.25, seed=0):
        """Broken ground: independent bumps about `patch` metres across. This is the
        high-frequency roughness a vision policy has to read; keep `patch` near the
        robot's foot size and `amplitude` under its ankle travel."""
        rng = np.random.default_rng(seed)
        return float(amplitude) * _value_noise(self.shape, self._period(patch), rng)

    def _along(self, angle):
        radians = math.radians(float(angle))
        return (self.X - self.center[0]) * math.cos(radians) + (
            self.Y - self.center[1]
        ) * math.sin(radians)

    def waves(self, *, amplitude, wavelength=2.0, angle=0.0, phase=0.0):
        """One sinusoid: the repeatable wavy floor used to compare gaits."""
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("Wave length must be a positive number of metres")
        along = self._along(angle)
        return float(amplitude) * np.sin(2 * math.pi * along / float(wavelength) + float(phase))

    def slope(self, *, grade, angle=0.0):
        """Constant incline; `grade` is rise over run (0.2 = 20 %, about 11°)."""
        return float(grade) * self._along(angle)

    def steps(self, *, rise, tread, angle=0.0):
        """Discrete terraces: stair-like ground with no ramp between treads."""
        if not np.isfinite(tread) or tread <= 0:
            raise ValueError("Step tread must be a positive number of metres")
        return float(rise) * np.floor(self._along(angle) / float(tread))

    def bowl(self, *, depth, radius, center=None, rim=0.0):
        """Smooth basin, `depth` metres below the rim at its centre. A water bed."""
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Bowl radius must be a positive number of metres")
        cx, cy = self.center if center is None else center
        distance = np.hypot(self.X - cx, self.Y - cy) / float(radius)
        profile = np.clip(1.0 - distance**2, 0.0, 1.0)
        return float(rim) - float(depth) * _smoothstep(profile)

    def potholes(self, *, count, depth, radius, seed=0, raised=0.0):
        """Scattered pits (and, with `raised`, mounds): trip hazards for vision."""
        rng = np.random.default_rng(seed)
        field = np.zeros(self.shape)
        for _ in range(max(0, int(count))):
            cx = rng.uniform(self.x[0], self.x[-1])
            cy = rng.uniform(self.y[0], self.y[-1])
            span = float(radius) * rng.uniform(0.6, 1.4)
            drop = float(depth) if rng.uniform() >= float(raised) else -float(depth)
            profile = np.clip(1.0 - (np.hypot(self.X - cx, self.Y - cy) / span) ** 2, 0.0, 1.0)
            field -= drop * _smoothstep(profile)
        return field

    def flatten(self, heights, *, center, radius, feather=0.5, level=None):
        """Level a disc — a spawn pad, a landing, a machine base — into the terrain.

        The disc is flat to `radius` and blends back to the original ground over
        `feather` metres, so the seam cannot catch a toe.
        """
        heights = np.array(heights, dtype=float)
        if heights.shape != self.shape:
            raise ValueError("Height grid does not match this terrain grid")
        cx, cy = center
        if level is None:
            level = float(self.sample(heights, cx, cy))
        distance = np.hypot(self.X - cx, self.Y - cy)
        blend = np.clip((distance - float(radius)) / max(float(feather), 1e-6), 0.0, 1.0)
        return heights * _smoothstep(blend) + float(level) * (1.0 - _smoothstep(blend))

    def flatten_rect(self, heights, *, bounds, feather=0.5, level=None):
        """Level an axis-aligned rectangle (x0, y0, x1, y1) into the terrain."""
        heights = np.array(heights, dtype=float)
        if heights.shape != self.shape:
            raise ValueError("Height grid does not match this terrain grid")
        x0, y0, x1, y1 = (float(v) for v in bounds)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Rectangle bounds must be (x0, y0, x1, y1) with x1 > x0, y1 > y0")
        if level is None:
            level = float(self.sample(heights, (x0 + x1) / 2, (y0 + y1) / 2))
        outside = np.maximum.reduce(
            [x0 - self.X, self.X - x1, y0 - self.Y, self.Y - y1, np.zeros(self.shape)]
        )
        blend = _smoothstep(np.clip(outside / max(float(feather), 1e-6), 0.0, 1.0))
        return heights * blend + float(level) * (1.0 - blend)

    def contains(self, x, y):
        x0, y0, x1, y1 = self.bounds
        tolerance = 1e-9
        return (
            (np.asarray(x, dtype=float) >= x0 - tolerance)
            & (np.asarray(x, dtype=float) <= x1 + tolerance)
            & (np.asarray(y, dtype=float) >= y0 - tolerance)
            & (np.asarray(y, dtype=float) <= y1 + tolerance)
        )

    def sample(self, heights, x, y, *, clamp=False):
        """Height at a world position on the TRIANGULATED surface.

        Not a bilinear patch: every target splits each cell into the same two triangles
        (the diagonal runs from the low corner to the high corner in index order), and
        on a steep cell the two differ by centimetres. This returns what the engines
        actually collide, so a predicted contact can be compared with a measured one.

        A point outside the patch is an authoring error and raises, because the edge
        height is not the ground there: beyond the boundary there is nothing. Internal
        callers that sample a footprint straddling the edge pass `clamp=True`.
        """
        heights = np.asarray(heights, dtype=float)
        if not clamp and not self.contains(x, y).all():
            x0, y0, x1, y1 = self.bounds
            raise ValueError(
                f"Position outside the terrain patch ({x0:.2f}..{x1:.2f}, {y0:.2f}..{y1:.2f} m)"
            )
        dx, dy = self.spacing
        u = np.clip((np.asarray(x, dtype=float) - self.x[0]) / dx, 0, self.cols - 1)
        v = np.clip((np.asarray(y, dtype=float) - self.y[0]) / dy, 0, self.rows - 1)
        i0 = np.clip(np.floor(u).astype(int), 0, self.cols - 2)
        j0 = np.clip(np.floor(v).astype(int), 0, self.rows - 2)
        fu, fv = u - i0, v - j0
        corner = heights[j0, i0]
        along_x = heights[j0, i0 + 1]
        opposite = heights[j0 + 1, i0 + 1]
        along_y = heights[j0 + 1, i0]
        lower = fu >= fv  # the triangle on the +X side of the diagonal
        return np.where(
            lower,
            corner + (along_x - corner) * fu + (opposite - along_x) * fv,
            corner + (along_y - corner) * fv + (opposite - along_y) * fu,
        )


@dataclass
class HeightField:
    """An elevation grid pinned to the world: heights are metres above `base_z`.

    `skirt` is the solid thickness carried below the lowest node so the patch is a
    closed volume wherever an engine collides triangle meshes (USD, SDF); native
    heightfields (MuJoCo, PyBullet) are open below and have no underside to fall
    through in the first place.
    """

    name: str
    grid: Grid
    heights: np.ndarray
    base_z: float = 0.0
    skirt: float = 0.5

    def __post_init__(self):
        heights = np.array(self.heights, dtype=float)
        if heights.shape != self.grid.shape:
            raise ValueError(
                f"{self.name}: height grid is {heights.shape}, expected {self.grid.shape}"
            )
        if not np.isfinite(heights).all():
            raise ValueError(f"{self.name}: terrain heights must all be finite")
        if not np.isfinite(self.base_z) or not np.isfinite(self.skirt) or self.skirt <= 0:
            raise ValueError(f"{self.name}: terrain base_z must be finite and skirt positive")
        if self.skirt > 100 or heights.max() - heights.min() > 1000:
            raise ValueError(f"{self.name}: terrain relief and skirt must stay under 1 km")
        self.heights = heights

    @property
    def lowest(self):
        return float(self.heights.min())

    @property
    def highest(self):
        return float(self.heights.max())

    @property
    def elevation(self):
        """Peak-to-trough range; never zero, so a flat patch still has a valid hfield."""
        return max(self.highest - self.lowest, 1e-3)

    @property
    def floor_z(self):
        """World z of the underside of the solid patch."""
        return self.base_z + self.lowest - self.skirt

    def height_at(self, x, y):
        """World z of the surface at a world XY inside the patch."""
        return self.base_z + self.grid.sample(self.heights, x, y)

    def normalized(self):
        """Heights mapped to [0, 1]: MuJoCo and the packaged binary both store this."""
        return ((self.heights - self.lowest) / self.elevation).astype(np.float32)

    def mujoco_size(self):
        """`<hfield size>`: x radius, y radius, elevation range, base thickness."""
        sx, sy = self.grid.size
        return (sx / 2.0, sy / 2.0, self.elevation, self.skirt)

    def mujoco_pos(self):
        """Geom position: the hfield's zero level sits at the lowest node."""
        cx, cy = self.grid.center
        return (cx, cy, self.base_z + self.lowest)

    def binary(self):
        """MuJoCo's native hfield payload: nrow, ncol as int32, then float32 rows."""
        rows, cols = self.grid.shape
        header = np.array([rows, cols], dtype=np.int32).tobytes()
        return header + np.ascontiguousarray(self.normalized()).tobytes()

    def slope_degrees(self):
        """Per-node steepest slope from the central-difference gradient, in degrees."""
        dx, dy = self.grid.spacing
        gy, gx = np.gradient(self.heights, dy, dx)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    def statistics(self):
        slopes = self.slope_degrees()
        dx, dy = self.grid.spacing
        return {
            "nodes": [int(self.grid.rows), int(self.grid.cols)],
            "size_m": [round(self.grid.size[0], 4), round(self.grid.size[1], 4)],
            "cell_m": [round(dx, 4), round(dy, 4)],
            "center_m": [round(self.grid.center[0], 4), round(self.grid.center[1], 4)],
            "lowest_z_m": round(self.base_z + self.lowest, 4),
            "highest_z_m": round(self.base_z + self.highest, 4),
            "relief_m": round(self.highest - self.lowest, 4),
            "mean_slope_deg": round(float(slopes.mean()), 2),
            "max_slope_deg": round(float(slopes.max()), 2),
            # Node-to-node step: what a foot actually trips over between samples.
            "max_node_step_m": round(
                float(
                    max(
                        np.abs(np.diff(self.heights, axis=0)).max(initial=0.0),
                        np.abs(np.diff(self.heights, axis=1)).max(initial=0.0),
                    )
                ),
                4,
            ),
            "skirt_m": round(float(self.skirt), 4),
        }

    def surface_mesh(self):
        """Closed triangle mesh of the patch: the surface, its four walls and a base.

        This is the visual surface everywhere and the contact surface wherever an
        engine collides static triangle meshes exactly (USD/Isaac Sim, SDF/Gazebo).
        """
        rows, cols = self.grid.shape
        top = np.column_stack(
            [
                self.grid.X.reshape(-1),
                self.grid.Y.reshape(-1),
                (self.base_z + self.heights).reshape(-1),
            ]
        )
        index = np.arange(rows * cols).reshape(rows, cols)
        a = index[:-1, :-1].reshape(-1)
        b = index[:-1, 1:].reshape(-1)
        c = index[1:, 1:].reshape(-1)
        d = index[1:, :-1].reshape(-1)
        faces = [np.column_stack([a, b, c]), np.column_stack([a, c, d])]
        floor = self.floor_z
        base = top.copy()
        base[:, 2] = floor
        offset = len(top)
        vertices = [top, base]
        # One closed boundary loop, counter-clockwise seen from above. The walls hang
        # from it and the base cap fans across it, so every edge belongs to exactly two
        # faces and the patch is a solid nothing can fall into from below.
        loop = np.concatenate(
            [index[0, :-1], index[:-1, -1], index[-1, :0:-1], index[::-1, 0][:-1]]
        )
        following = np.roll(loop, -1)
        faces.append(np.column_stack([loop, following + offset, following]))
        faces.append(np.column_stack([loop, loop + offset, following + offset]))
        # The base fans from its own centre, not from a corner: three points along one
        # straight edge of the rectangle would otherwise make zero-area triangles.
        hub = len(top) + len(base)
        vertices.append(np.array([[self.grid.center[0], self.grid.center[1], floor]], dtype=float))
        ring = loop + offset
        faces.append(np.column_stack([np.full(len(ring), hub), np.roll(ring, -1), ring]))
        mesh = trimesh.Trimesh(
            vertices=np.vstack(vertices), faces=np.vstack(faces).astype(np.int64), process=False
        )
        if mesh.volume < 0:  # every face is wound the same way; flip the whole shell
            mesh.invert()
        return mesh

    def _footprint(self, x, y, radius, samples):
        radius = float(radius)
        offsets = np.linspace(-radius, radius, 2 * int(samples) + 1)
        dx, dy = np.meshgrid(offsets, offsets)
        distance = np.hypot(dx, dy)
        inside = distance <= radius
        surface = self.base_z + self.grid.sample(
            self.heights, x + dx[inside], y + dy[inside], clamp=True
        )
        return surface, distance[inside]

    def rest_height(self, x, y, radius, samples=6):
        """Where a sphere of `radius` dropped at (x, y) comes to rest, as a surface
        height (its centre minus the radius).

        A sphere does not touch the ground directly beneath its centre: on a slope it
        rides high by r/cos(theta) - r, and beside a riser it rests on the edge. Both
        are geometry, not engine error, so the probe test compares against this value
        instead of the height at the node.
        """
        surface, distance = self._footprint(x, y, radius, samples)
        centre = surface + np.sqrt(np.maximum(float(radius) ** 2 - distance**2, 0.0))
        return float(centre.max()) - float(radius)

    def support_height(self, x, y, radius, samples=6):
        """Highest ground within `radius` of (x, y): what a flat-bottomed object of
        that half-width rests on. A boulder placed by the height at its centre alone
        starts buried wherever the ground rises under its edge."""
        surface, _ = self._footprint(x, y, radius, samples)
        return float(surface.max())

    def probe_points(self, limit=24, margin=2):
        """Grid nodes spread over the patch, used to measure contact against the
        authored surface. Nodes, not arbitrary points: there the triangulated surface
        equals the authored height exactly, so any error measured is the engine's."""
        rows, cols = self.grid.shape
        margin = int(min(margin, (rows - 1) // 2, (cols - 1) // 2))
        side = max(2, int(math.isqrt(max(1, int(limit)))))
        js = np.unique(np.linspace(margin, rows - 1 - margin, side).astype(int))
        is_ = np.unique(np.linspace(margin, cols - 1 - margin, side).astype(int))
        points = []
        for j in js:
            for i in is_:
                points.append(
                    (
                        float(self.grid.x[i]),
                        float(self.grid.y[j]),
                        float(self.base_z + self.heights[j, i]),
                    )
                )
        return points[: max(1, int(limit))]


# Fresh water against a blunt body. The quadratic coefficient is about rho * Cd / 2
# for Cd ~ 0.47; the linear term covers the low-speed regime a wading robot lives in.
WATER_DENSITY = 1000.0
WATER_LINEAR_DRAG = 60.0
WATER_QUADRATIC_DRAG = 235.0


@dataclass
class WaterVolume:
    """A body of water: where its surface is, how deep, and what it does to a body.

    No simulator in this pack's targets runs fluid dynamics. What is exported is
    honest and useful: the visible surface, the submerged region, and the buoyancy and
    drag coefficients the packaged nova_water.py hook applies to whatever is inside it
    in MuJoCo and PyBullet. Waves, currents, splashes and wetting are not simulated.
    """

    name: str
    bounds: tuple[float, float, float, float]
    surface_z: float
    bed_z: float
    density: float = WATER_DENSITY
    # Drag on a submerged part, per square metre of its submerged cross-section:
    # force = -(linear_drag + quadratic_drag * speed) * area * velocity.
    linear_drag: float = WATER_LINEAR_DRAG
    quadratic_drag: float = WATER_QUADRATIC_DRAG

    def __post_init__(self):
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError(f"{self.name}: water bounds must be (x0, y0, x1, y1) in metres")
        x0, y0, x1, y1 = bounds
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"{self.name}: water bounds must have positive extent")
        if not np.isfinite([self.surface_z, self.bed_z]).all() or self.surface_z <= self.bed_z:
            raise ValueError(f"{self.name}: water surface must sit above its bed")
        if (
            not np.isfinite([self.density, self.linear_drag, self.quadratic_drag]).all()
            or min(self.density, self.linear_drag, self.quadratic_drag) < 0
            or self.density > 20_000
        ):
            raise ValueError(f"{self.name}: water density and drag must be non-negative")
        self.bounds = (float(x0), float(y0), float(x1), float(y1))

    @property
    def depth(self):
        return float(self.surface_z - self.bed_z)

    def surface_mesh(self, thickness=0.02):
        x0, y0, x1, y1 = self.bounds
        mesh = trimesh.creation.box(extents=(x1 - x0, y1 - y0, thickness))
        mesh.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, self.surface_z - thickness / 2))
        return mesh

    def specification(self):
        return {
            "name": self.name,
            "bounds_m": list(self.bounds),
            "surface_z_m": round(float(self.surface_z), 4),
            "bed_z_m": round(float(self.bed_z), 4),
            "depth_m": round(self.depth, 4),
            "density_kg_m3": float(self.density),
            "linear_drag": float(self.linear_drag),
            "quadratic_drag": float(self.quadratic_drag),
        }


def rock_mesh(radius, seed=0, roughness=0.45, flatten=0.62):
    """One convex boulder: a jittered icosphere, squashed so it sits like a rock.

    Convex on purpose — a boulder then collides exactly on every engine with a single
    hull, costs one contact pair, and never needs decomposition. Coarse on purpose too:
    a smooth ball rolls down any slope forever, while broad facets let it find a seat.
    """
    rng = np.random.default_rng(seed)
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=float(radius))
    scale = 1.0 + rng.uniform(-float(roughness), float(roughness), size=(len(mesh.vertices), 1))
    mesh.vertices = mesh.vertices * scale
    mesh.vertices[:, 2] *= float(flatten)
    mesh.vertices = mesh.vertices * rng.uniform(0.8, 1.2, size=(1, 3))
    mesh = mesh.convex_hull
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh
