"""Code-native environments, shared by native exports and Studio preview.

All geometry uses metres, a right-handed frame, and Z up. Physical surfaces are
static by default; visual color/roughness never determines contact friction.
Objects given a RigidBody are exported as movable bodies on EVERY target: a hinge
or slider joint to world (doors, drawers, lids) or a free body (loose props a robot
can pick up or push aside). Static-only exports are never produced silently.
"""

import re
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .terrain import Grid, HeightField, WaterVolume, rock_mesh

MAX_NOTES = 64
NOTE_LENGTH = 400

# Semantic classes that only make sense on a movable body. An object tagged with one
# of these but left static is an authoring error, not a style choice.
MOVABLE_CLASSES = frozenset({"door", "drawer", "lid", "gate", "hatch", "movable_prop"})
# Name tokens that usually denote the moving part of a mechanism ...
_MECHANISM_TOKENS = frozenset({"door", "drawer", "lid", "hatch", "gate", "shutter", "flap", "sash"})
# ... unless another token says the object is the fixed surround or a fitting.
_FIXED_TOKENS = frozenset(
    {
        "frame",
        "jamb",
        "trim",
        "sill",
        "threshold",
        "way",
        "doorway",
        "arch",
        "opening",
        "casing",
        "header",
        "stop",
        "stopper",
        "track",
        "rail",
        "hinge",
        "handle",
        "knob",
        "pull",
        "latch",
        "lock",
        "closer",
        "mat",
        "sign",
        "label",
        "marking",
        "bell",
        "step",
        "slot",
        "cavity",
        "post",
        "pillar",
        "column",
        "wall",
        "carcass",
        "housing",
        "surround",
        "stile",
        "lintel",
        "bar",
        "bumper",
    }
)


@dataclass
class PrescribedMotion:
    """Externally prescribed kinematic motion for an environment body.

    This is scripted world behavior (a swaying branch, a patrolling obstacle, a
    rotating hazard), not force-driven dynamics: the trajectory is the authority
    and the robot must react to it. Contact against the robot is still real on
    every engine that simulates the drive. Three kinds:

    - "oscillate": sinusoidal joint motion on a hinge (radians) or slide (metres):
      q(t) = amplitude * sin(2*pi * (t + phase_s) / period_s). Amplitude must stay
      inside the joint limits.
    - "spin": constant-rate hinge rotation, rate_rad_s, no limits (a fan, a
      sweeping boom). Hinge only.
    - "patrol": the body's origin follows world-space waypoints at speed_m_s.
      loop="cycle" retraces the path back and forth; loop="wrap" closes it into a
      ring. face_travel yaws the body along the direction of travel. Requires
      motion="kinematic".
    """

    kind: str
    amplitude: float = 0.0
    period_s: float = 0.0
    phase_s: float = 0.0
    rate_rad_s: float = 0.0
    waypoints: tuple[tuple[float, float, float], ...] = ()
    speed_m_s: float = 0.0
    loop: str = "cycle"
    face_travel: bool = False

    @classmethod
    def oscillate(cls, *, amplitude, period_s, phase_s=0.0):
        return cls("oscillate", amplitude=amplitude, period_s=period_s, phase_s=phase_s)

    @classmethod
    def spin(cls, *, rate_rad_s):
        return cls("spin", rate_rad_s=rate_rad_s)

    @classmethod
    def patrol(cls, *, waypoints, speed_m_s, loop="cycle", face_travel=False):
        return cls(
            "patrol",
            waypoints=tuple(tuple(point) for point in waypoints),
            speed_m_s=speed_m_s,
            loop=loop,
            face_travel=face_travel,
        )


@dataclass
class RigidBody:
    """Single body attached to world. Inertia is about COM in world-aligned axes.

    origin is the world-space body/joint anchor at q=0. center_of_mass is relative
    to origin. Meshes remain world-space at q=0; the exporter localizes them.
    No nested mechanisms, deformables or inferred material density. A `drive`
    prescribes kinematic motion (see PrescribedMotion); motion="kinematic" is a
    world-attached body whose pose the drive alone determines.
    """

    mass: float
    diagonal_inertia: tuple[float, float, float]
    provenance: str
    origin: tuple[float, float, float] = (0, 0, 0)
    center_of_mass: tuple[float, float, float] = (0, 0, 0)
    motion: str = "free"
    axis: tuple[float, float, float] = (0, 0, 1)
    limits: tuple[float, float] | None = None
    damping: float = 0.0
    frictionloss: float = 0.0
    drive: PrescribedMotion | None = None

    @classmethod
    def box(cls, *, mass, size, provenance, **properties):
        """Uniform solid-box approximation, not an inferred real-object inertia."""
        x, y, z = size
        if not np.isfinite(size).all() or min(size) <= 0:
            raise ValueError("Inertial box dimensions must be positive and finite")
        return cls(
            mass,
            tuple(mass / 12 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])),
            provenance,
            **properties,
        )


@dataclass
class SceneObject:
    name: str
    mesh: trimesh.Trimesh
    color: tuple[float, float, float, float]
    friction: float
    restitution: float
    collidable: bool
    semantic_class: str
    roughness: float
    collision_meshes: tuple[trimesh.Trimesh, ...] | None = None
    body: RigidBody | None = None
    torsional_friction: float = 0.0
    rolling_friction: float = 0.0
    # Elevation grid for ground the robot walks on. Terrain never goes through convex
    # decomposition: every target collides it with its own exact primitive instead.
    heightfield: HeightField | None = None
    # Host-derived convex pieces (see collision.py): None = not resolved yet, () = the
    # object is convex and one hull is exact. Never authored; never validated as input.
    auto_collision: tuple[trimesh.Trimesh, ...] | None = field(default=None, repr=False)
    auto_collision_report: dict | None = field(default=None, repr=False)

    @property
    def movable(self):
        return self.body is not None

    @property
    def is_terrain(self):
        return self.heightfield is not None

    @property
    def split_contact(self):
        """Visual and contact surfaces differ: explicit or derived convex pieces,
        extended friction or a movable body (whose contact must be convex everywhere).
        Terrain is separate again: its contact primitive is the elevation grid."""
        if self.heightfield is not None:
            return False
        return (
            self.body is not None
            or self.collision_meshes is not None
            or bool(self.auto_collision)
            or self.torsional_friction != 0
            or self.rolling_friction != 0
        )


@dataclass
class Environment:
    objects: list[SceneObject] = field(default_factory=list)
    water_volumes: list[WaterVolume] = field(default_factory=list)
    # Short authoring findings the build prints as SCENE_NOTE lines and keeps in the
    # validation report: a scatter that placed fewer rocks than asked, for example.
    notes: list[str] = field(default_factory=list)
    spawn_position: tuple[float, float, float] = (0.0, -2.0, 0.6)
    goal_position: tuple[float, float, float] = (0.0, 2.0, 0.0)
    description: str = "Generated robot environment"
    gravity_m_s2: float = 9.81

    def add_mesh(
        self,
        name,
        mesh,
        *,
        position=(0, 0, 0),
        transform=None,
        color=(0.55, 0.60, 0.65, 1.0),
        friction=0.7,
        restitution=0.0,
        collidable=True,
        semantic_class="structure",
        roughness=0.8,
        collision_meshes=None,
        body=None,
        torsional_friction=0.0,
        rolling_friction=0.0,
    ):
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("add_mesh requires a trimesh.Trimesh")
        geometry = mesh.copy()
        if transform is not None:
            geometry.apply_transform(np.asarray(transform, dtype=float))
        geometry.apply_translation(position)
        collisions = None
        if collision_meshes is not None:
            collisions = []
            for collider in collision_meshes:
                if not isinstance(collider, trimesh.Trimesh):
                    raise ValueError("collision_meshes requires trimesh meshes")
                collider = collider.copy()
                if transform is not None:
                    collider.apply_transform(np.asarray(transform, dtype=float))
                collider.apply_translation(position)
                collisions.append(collider)
            collisions = tuple(collisions)
        self.objects.append(
            SceneObject(
                name,
                geometry,
                tuple(color),
                friction,
                restitution,
                collidable,
                semantic_class,
                roughness,
                collisions,
                body,
                torsional_friction,
                rolling_friction,
            )
        )
        return geometry

    def box(self, name, size, center, **properties):
        size = np.asarray(size, dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError("Box dimensions must be three positive metre values")
        geometry = trimesh.creation.box(extents=size)
        # Boxes and stairs have hard edges; shared corner normals make them look rounded.
        geometry.unmerge_vertices()
        return self.add_mesh(name, geometry, position=center, **properties)

    def stairs(
        self,
        *,
        name="stairs",
        count=8,
        width=1.8,
        rise=0.16,
        tread=0.30,
        origin=(0, 0, 0),
        **properties,
    ):
        if not isinstance(count, int) or not 1 <= count <= 256:
            raise ValueError("Stair count must be between 1 and 256")
        x, y, z = origin
        for index in range(count):
            height = (index + 1) * rise
            self.box(
                f"{name}_{index + 1:03d}",
                (width, tread, height),
                (x, y + (index + 0.5) * tread, z + height / 2),
                semantic_class="stair_step",
                **properties,
            )

    def note(self, text):
        """Record an authoring finding for the build report; never a silent fallback."""
        text = str(text).strip()
        if not text:
            return
        if len(self.notes) < MAX_NOTES:
            self.notes.append(text[:NOTE_LENGTH])

    def terrain(
        self,
        name,
        grid,
        heights,
        *,
        base_z=0.0,
        skirt=0.5,
        color=(0.38, 0.35, 0.30, 1.0),
        friction=0.9,
        restitution=0.0,
        semantic_class="terrain",
        roughness=0.95,
    ):
        """Ground built from an elevation grid: hills, rock, rubble, ramps, a river bed.

        Returns the HeightField so the same surface can be queried while the rest of
        the scene is placed: `field.height_at(x, y)` is the exact z a foot, a boulder
        or the robot's spawn should sit on. Terrain is always static and is never
        convex-decomposed; each target collides it with its own exact primitive
        (MuJoCo hfield, PyBullet heightfield, triangle mesh in USD and SDF).
        """
        if not isinstance(grid, Grid):
            raise ValueError("terrain() needs a terrain.Grid describing the patch")
        patch = HeightField(name=name, grid=grid, heights=heights, base_z=base_z, skirt=skirt)
        self.add_mesh(
            name,
            patch.surface_mesh(),
            color=color,
            friction=friction,
            restitution=restitution,
            semantic_class=semantic_class,
            roughness=roughness,
        )
        self.objects[-1].heightfield = patch
        return patch

    def water(
        self,
        name,
        *,
        bounds,
        surface_z,
        bed_z=None,
        density=None,
        linear_drag=None,
        quadratic_drag=None,
        color=(0.13, 0.36, 0.46, 0.55),
    ):
        """A body of water over terrain already placed: pond, flooded bay, ford.

        The surface is exported as a visible, non-colliding sheet, and the region is
        exported with the buoyancy and drag coefficients the packaged runtime hook
        applies to anything inside it. No target simulates fluid: there are no waves,
        currents or splashes, and the robot is not wetted. The bed is whatever
        collidable ground already lies under the region; water creates none.
        """
        x0, y0, x1, y1 = (float(v) for v in bounds)
        if bed_z is None:
            floors = []
            for obj in self.objects:
                if obj.heightfield is None:
                    continue
                grid = obj.heightfield.grid
                inside = (x0 <= grid.X) & (x1 >= grid.X) & (y0 <= grid.Y) & (y1 >= grid.Y)
                if inside.any():
                    lowest = float(obj.heightfield.heights[inside].min())
                    floors.append(obj.heightfield.base_z + lowest)
            if not floors:
                raise ValueError(
                    f"{name}: no terrain under this water region; pass bed_z explicitly"
                )
            bed_z = min(floors)
        arguments = {
            key: value
            for key, value in (
                ("density", density),
                ("linear_drag", linear_drag),
                ("quadratic_drag", quadratic_drag),
            )
            if value is not None
        }
        volume = WaterVolume(
            name=name, bounds=(x0, y0, x1, y1), surface_z=float(surface_z), bed_z=float(bed_z),
            **arguments,
        )
        self.add_mesh(
            name,
            volume.surface_mesh(),
            color=color,
            collidable=False,
            semantic_class="water",
            roughness=0.1,
            friction=0.0,
        )
        self.water_volumes.append(volume)
        return volume

    def scatter_rocks(
        self,
        name,
        patch,
        *,
        count,
        radius=(0.12, 0.35),
        seed=0,
        sink=None,
        area=None,
        max_slope_deg=None,
        movable=False,
        density=2600.0,
        friction=0.95,
        rolling_friction=0.03,
        torsional_friction=0.01,
        color=(0.33, 0.31, 0.29, 1.0),
    ):
        """Boulders bedded into a terrain patch, the loose part of rocky ground.

        Each rock is convex, so it collides exactly on every engine with one hull.
        `movable=True` makes them free bodies the robot can shift; the mass follows
        from the rock's own volume and the stated density, which is an assumption.

        `sink` is the fraction of the rock's height buried in the ground. A fixed
        boulder is bedded in (0.35 by default); a free one rests ON the surface (0.0),
        because a body that starts inside the terrain is ejected by the solver and the
        build reports it, correctly, as a prop that was not resting on a support.

        Returns how many rocks were placed. Fewer than `count` means the area was too
        crowded or too steep; that shortfall is recorded as a scene note, never hidden.
        """
        if not isinstance(patch, HeightField):
            raise ValueError("scatter_rocks() needs the HeightField returned by terrain()")
        if not isinstance(count, int) or not 0 <= count <= 512:
            raise ValueError(f"{name}: rock count must be an integer between 0 and 512")
        rng = np.random.default_rng(seed)
        sink = (0.0 if movable else 0.35) if sink is None else float(sink)
        low, high = (
            (float(radius[0]), float(radius[1])) if np.ndim(radius) else (float(radius),) * 2
        )
        if not np.isfinite([low, high]).all() or low <= 0 or high < low:
            raise ValueError(f"{name}: rock radius must be positive and ordered")
        px0, py0, px1, py1 = patch.grid.bounds
        x0, y0, x1, y1 = (px0, py0, px1, py1) if area is None else (float(v) for v in area)
        # A rock never straddles the patch edge: beyond it there is nothing to rest on.
        x0, y0 = max(x0, px0) + high, max(y0, py0) + high
        x1, y1 = min(x1, px1) - high, min(y1, py1) - high
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"{name}: scatter area is smaller than one rock")
        slopes = patch.slope_degrees()
        # Nothing may be scattered into something already standing there. Terrain and
        # water span the whole patch, so they are not obstacles; everything else is.
        occupied = [
            obj.mesh.bounds for obj in self.objects if obj.heightfield is None and obj.collidable
        ]
        placed = 0
        for _ in range(count * 12):
            if placed >= count:
                break
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if max_slope_deg is not None and float(
                patch.grid.sample(slopes, x, y)
            ) > float(max_slope_deg):
                continue
            size = float(rng.uniform(low, high))
            mesh = rock_mesh(size, seed=int(rng.integers(0, 2**31 - 1)))
            half = float(mesh.extents[2]) / 2
            # Bedded by `sink` of its own height; a free rock sits just clear so it
            # settles into contact instead of being pushed out of the ground. The
            # ground under the whole footprint decides, not the height at one point:
            # on broken ground the rock's own width straddles a rise.
            span = float(max(mesh.extents[0], mesh.extents[1])) / 2
            support = patch.support_height(x, y, span)
            clearance = 0.0 if sink > 0 else 0.002
            center = (x, y, support + half - sink * 2 * half + clearance)
            body = None
            if movable:
                mass = max(0.2, float(mesh.volume) * float(density))
                body = RigidBody.box(
                    mass=mass,
                    size=tuple(float(e) for e in mesh.extents),
                    origin=center,
                    motion="free",
                    provenance=(
                        f"Assumed solid rock at {density:.0f} kg/m3 over the generated "
                        "boulder volume; box inertia approximation"
                    ),
                )
            box = np.array([[*center] - mesh.extents / 2, [*center] + mesh.extents / 2])
            if any(
                (box[0] < other[1] + 1e-6).all() and (other[0] < box[1] + 1e-6).all()
                for other in occupied
            ):
                continue  # a rock inside a wall, a post or another rock explodes
            occupied.append(box)
            self.add_mesh(
                f"{name}_{placed + 1:03d}",
                mesh,
                position=center,
                color=color,
                friction=friction,
                semantic_class="movable_prop" if movable else "rock",
                roughness=0.95,
                body=body,
                rolling_friction=rolling_friction,
                torsional_friction=torsional_friction,
            )
            placed += 1
        if placed < count:
            self.note(
                f"{name}: placed {placed} of {count} rocks; the area is too crowded or too "
                "steep for the rest"
            )
        return placed


def _validate_drive(name: str, body: RigidBody) -> None:
    drive = body.drive
    if drive is None:
        if body.motion == "kinematic":
            raise ValueError(f"{name}: kinematic bodies exist to be driven; give a drive")
        return
    if not isinstance(drive, PrescribedMotion):
        raise ValueError(f"{name}: drive must be a PrescribedMotion")
    if drive.kind == "oscillate":
        if body.motion not in {"hinge", "slide"}:
            raise ValueError(f"{name}: oscillate drives a hinge or slide joint")
        if (
            not np.isfinite([drive.amplitude, drive.period_s, drive.phase_s]).all()
            or drive.amplitude == 0
            or drive.period_s <= 0
        ):
            raise ValueError(f"{name}: oscillation needs a nonzero amplitude and period")
        low, high = body.limits
        if not (low <= -abs(drive.amplitude) and abs(drive.amplitude) <= high):
            raise ValueError(f"{name}: oscillation amplitude exceeds the joint limits")
    elif drive.kind == "spin":
        if body.motion != "hinge":
            raise ValueError(f"{name}: spin drives a hinge joint")
        if not np.isfinite(drive.rate_rad_s) or drive.rate_rad_s == 0:
            raise ValueError(f"{name}: spin needs a nonzero rate_rad_s")
    elif drive.kind == "patrol":
        if body.motion != "kinematic":
            raise ValueError(f"{name}: patrol requires motion='kinematic'")
        points = np.asarray(drive.waypoints, dtype=float)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
            raise ValueError(f"{name}: patrol needs at least two XYZ waypoints")
        if not np.isfinite(points).all():
            raise ValueError(f"{name}: patrol waypoints must be finite")
        if (np.linalg.norm(np.diff(points, axis=0), axis=1) < 1e-6).any():
            raise ValueError(f"{name}: consecutive patrol waypoints coincide")
        if not np.isfinite(drive.speed_m_s) or drive.speed_m_s <= 0:
            raise ValueError(f"{name}: patrol needs a positive speed_m_s")
        if not np.allclose(points[0], body.origin, atol=1e-6):
            raise ValueError(
                f"{name}: the first patrol waypoint must equal the body origin, so the "
                "authored pose and t=0 coincide"
            )
        if drive.loop not in {"cycle", "wrap"}:
            raise ValueError(f"{name}: patrol loop must be 'cycle' or 'wrap'")
    else:
        raise ValueError(f"{name}: unknown drive kind {drive.kind!r}")


def validate_environment(scene: Environment) -> None:
    if not isinstance(scene, Environment) or not 1 <= len(scene.objects) <= 2000:
        raise ValueError("build() must return an Environment with 1–2000 objects")
    if not np.isfinite(scene.gravity_m_s2) or not 0 <= scene.gravity_m_s2 <= 50:
        raise ValueError("Downward gravity must be between 0 and 50 m/s²")
    names = set()
    triangles = 0
    for obj in scene.objects:
        if not isinstance(obj.name, str) or not obj.name.strip() or obj.name in names:
            raise ValueError("Object names must be nonempty and unique")
        names.add(obj.name)
        if obj.name.startswith("nova_robot_geom_"):
            raise ValueError("Object name uses reserved robot preview prefix")
        if obj.body is not None:
            body = obj.body
            if not isinstance(body, RigidBody):
                raise ValueError(f"{obj.name}: body must be a RigidBody")
            inertia = np.asarray(body.diagonal_inertia)
            if (
                not np.isfinite(body.mass)
                or body.mass <= 0
                or inertia.shape != (3,)
                or not np.isfinite(inertia).all()
                or (inertia <= 0).any()
                or 2 * inertia.max() > inertia.sum() + 1e-12
            ):
                raise ValueError(f"{obj.name}: invalid mass or physically impossible inertia")
            if not isinstance(body.provenance, str) or not body.provenance.strip():
                raise ValueError(f"{obj.name}: physical assumptions/provenance are required")
            for vector in (body.origin, body.center_of_mass, body.axis):
                if np.asarray(vector).shape != (3,) or not np.isfinite(vector).all():
                    raise ValueError(f"{obj.name}: invalid body frame")
            if body.motion not in {"free", "hinge", "slide", "kinematic"}:
                raise ValueError(f"{obj.name}: unsupported rigid-body motion")
            if (
                not np.isfinite([body.damping, body.frictionloss]).all()
                or min(body.damping, body.frictionloss) < 0
            ):
                raise ValueError(f"{obj.name}: invalid joint damping/frictionloss")
            spins = body.drive is not None and body.drive.kind == "spin"
            if body.motion == "free":
                if body.limits is not None or body.damping or body.frictionloss:
                    raise ValueError(f"{obj.name}: free bodies cannot have joint limits/friction")
            elif body.motion == "kinematic":
                if body.limits is not None or body.damping or body.frictionloss:
                    raise ValueError(
                        f"{obj.name}: kinematic bodies take no joint limits/friction"
                    )
            elif spins:
                if body.limits is not None:
                    raise ValueError(
                        f"{obj.name}: a spinning hinge is continuous; omit limits"
                    )
                if not np.isclose(np.linalg.norm(body.axis), 1):
                    raise ValueError(f"{obj.name}: unit joint axis required")
            elif (
                not np.isclose(np.linalg.norm(body.axis), 1)
                or body.limits is None
                or np.asarray(body.limits).shape != (2,)
                or not np.isfinite(body.limits).all()
                or not body.limits[0] <= 0 <= body.limits[1]
                or body.limits[0] >= body.limits[1]
            ):
                raise ValueError(
                    f"{obj.name}: unit axis and ordered limits containing zero required"
                )
            _validate_drive(obj.name, body)
        if obj.heightfield is not None:
            if not isinstance(obj.heightfield, HeightField):
                raise ValueError(f"{obj.name}: heightfield must come from Environment.terrain")
            if obj.body is not None or obj.collision_meshes is not None:
                raise ValueError(
                    f"{obj.name}: terrain is static ground; it takes no rigid body and no "
                    "collision pieces"
                )
            if not obj.collidable:
                raise ValueError(f"{obj.name}: terrain must be collidable")
        if obj.collision_meshes is not None:
            if not obj.collidable or not 1 <= len(obj.collision_meshes) <= 32:
                raise ValueError(f"{obj.name}: use 1–32 collision pieces on a collidable object")
            for collider in obj.collision_meshes:
                if (
                    not isinstance(collider, trimesh.Trimesh)
                    or not collider.is_volume
                    or not collider.is_convex
                    or not np.isfinite(collider.vertices).all()
                    or (collider.area_faces <= 1e-14).any()
                ):
                    raise ValueError(f"{obj.name}: collision pieces must be finite convex volumes")
                triangles += len(collider.faces)
        if (
            not np.isfinite([obj.torsional_friction, obj.rolling_friction]).all()
            or min(obj.torsional_friction, obj.rolling_friction) < 0
        ):
            raise ValueError(f"{obj.name}: invalid torsional/rolling friction")
        mesh = obj.mesh
        if (
            not len(mesh.vertices)
            or not len(mesh.faces)
            or not np.isfinite(mesh.vertices).all()
            or not np.isfinite(mesh.area_faces).all()
            or (mesh.area_faces <= 1e-14).any()
        ):
            raise ValueError(f"{obj.name}: empty, non-finite or degenerate mesh")
        triangles += len(mesh.faces)
        if triangles > 2_000_000:
            raise ValueError("Environment exceeds two million triangles")
        color = np.asarray(obj.color)
        if color.shape != (4,) or not np.isfinite(color).all() or ((color < 0) | (color > 1)).any():
            raise ValueError(f"{obj.name}: color must be RGBA values between zero and one")
        if not np.isfinite([obj.friction, obj.restitution, obj.roughness]).all():
            raise ValueError(f"{obj.name}: invalid surface properties")
        if obj.friction < 0 or not 0 <= obj.restitution <= 1 or not 0 <= obj.roughness <= 1:
            raise ValueError(f"{obj.name}: invalid friction, restitution or roughness")
    if len(scene.water_volumes) > 32:
        raise ValueError("Use at most 32 water volumes in one environment")
    water_names = set()
    for volume in scene.water_volumes:
        if not isinstance(volume, WaterVolume):
            raise ValueError("Water volumes must come from Environment.water")
        if volume.name in water_names:
            raise ValueError("Water volume names must be unique")
        water_names.add(volume.name)
    if len(scene.notes) > MAX_NOTES or any(
        not isinstance(note, str) or len(note) > NOTE_LENGTH for note in scene.notes
    ):
        raise ValueError("Scene notes must be short strings")
    for position in (scene.spawn_position, scene.goal_position):
        if np.asarray(position).shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Spawn and goal must be finite XYZ positions in metres")


def _name_tokens(name):
    return {token for token in re.split(r"[^a-z0-9]+", name.lower()) if token}


def mechanism_audit(scene: Environment) -> dict:
    """Host-side check that mechanisms are real. Returns counts, hard errors (a
    movable semantic class without a body) and warnings (a name such as
    `kitchen_door` on a static object). Never guesses a body for the author."""
    errors, warnings = [], []
    counts = {"hinge": 0, "slide": 0, "free": 0, "kinematic": 0, "static": 0, "driven": 0}
    for obj in scene.objects:
        if obj.body is not None:
            counts[obj.body.motion] += 1
            if obj.body.drive is not None:
                counts["driven"] += 1
            continue
        counts["static"] += 1
        tokens = _name_tokens(obj.name)
        if obj.semantic_class in MOVABLE_CLASSES:
            errors.append(
                f"{obj.name}: semantic_class {obj.semantic_class!r} is static; give it a "
                "RigidBody (motion='hinge'/'slide' for doors, drawers and lids, 'free' for "
                "loose props) or use a structural semantic class"
            )
        elif tokens & _MECHANISM_TOKENS and not tokens & _FIXED_TOKENS:
            warnings.append(
                f"{obj.name}: named like a moving part but exported static; if it should "
                "open, attach a RigidBody hinge/slide, otherwise rename it (e.g. *_frame)"
            )
    return {
        "movable_bodies": counts["hinge"] + counts["slide"] + counts["free"]
        + counts["kinematic"],
        "hinges": counts["hinge"],
        "sliders": counts["slide"],
        "free_bodies": counts["free"],
        "kinematic_bodies": counts["kinematic"],
        "driven_bodies": counts["driven"],
        "static_objects": counts["static"],
        "errors": errors,
        "warnings": warnings,
    }
