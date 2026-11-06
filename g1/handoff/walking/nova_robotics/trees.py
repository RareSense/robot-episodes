"""Grown trees for robotics scenes: ten hardcoded BASE FORMS, dressed per species.

A base form is the proven skeleton — trunk, scaffolds, laterals, twigs, grown
branch by branch with thousands of individual closed leaves — ported from the
farm environments and the world-building tree set. It is a starting point, not
a product: the species the brief names decides the dressing (leaf size and
density, fruit, colours at placement), so "an apple tree" is `round_crown` or
`tall_spindle` dressed with apple leaves and hanging apples — never a cone with
a sphere on top, and never a stock model handed over unchanged.

    from nova_robotics import trees
    tree = trees.grow("round_crown", (x, y, z), seed=7, height=4.5,
                      fruit={"chance": 0.5, "radius": 0.04})
    trees.place(world, "apple_04", tree, (x, y, z),
                sun_color=(0.36, 0.54, 0.22, 1), shade_color=(0.20, 0.36, 0.13, 1),
                fruit_color=(0.72, 0.10, 0.08, 1))

For physics each tree collides as its trunk plus its crown envelope — what a
ground robot actually meets — while leaves and fruit stay visual. A swaying
tree is the same parts as synced hinge bodies sharing one prescribed drive.
Forests stay inside the triangle budget by density: `density=0.4` quarters the
leaf count for background rows while keeping the silhouette.
"""

from __future__ import annotations

import numpy as np
import trimesh

BARK = (0.30, 0.23, 0.17, 1.0)

BASE_FORMS = (
    "round_crown",    # garden/orchard broadleaf: short trunk, scaffolds, round canopy
    "tall_spindle",   # trellised orchard row: central leader, short fruit laterals
    "open_vase",      # stone fruit: low trunk splitting into an open goblet
    "broadleaf_high", # mature park tree: tall clear trunk, wide lifted crown
    "conifer_spire",  # fir/spruce: whorls of near-level branches to a spire
    "young_pine",     # juvenile pine: sparse upswept whorls, tufted foliage
    "columnar",       # poplar/cypress: tight upright branches, narrow silhouette
    "weeping",        # willow: arching scaffolds trailing long curtains
    "sapling",        # thin staked newcomer: one leader, a few twigs
    "dead_snag",      # bare weathered trunk and broken limbs, no leaves
)


# ------------------------------------------------------------------ primitives --


class _Buffer:
    """Raw vertex/face accumulation: one trimesh at the end, not thousands."""

    def __init__(self):
        self.vertices, self.faces, self.count = [], [], 0

    def add(self, vertices, faces):
        self.vertices.append(np.asarray(vertices, float))
        self.faces.append(np.asarray(faces, int) + self.count)
        self.count += len(vertices)

    def mesh(self):
        if not self.vertices:
            return None
        mesh = trimesh.Trimesh(np.vstack(self.vertices), np.vstack(self.faces), process=True)
        mesh.fix_normals(multibody=True)
        return mesh


def _frame(direction):
    d = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, helper)
    u /= np.linalg.norm(u)
    return d, u, np.cross(d, u)


def _tube(buffer, points, radii, sides):
    """A smooth tapered tube along a polyline, capped at both ends. Ring frames are
    parallel-transported so the tube never twists or pinches at bends."""
    points = np.asarray(points, float)
    n = len(points)
    _, u, v = _frame(points[1] - points[0])
    rings = []
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    for i in range(n):
        tangent = points[min(i + 1, n - 1)] - points[max(i - 1, 0)]
        tangent /= np.linalg.norm(tangent)
        u = u - np.dot(u, tangent) * tangent
        u /= np.linalg.norm(u)
        v = np.cross(tangent, u)
        rings.append(points[i] + radii[i] * (np.outer(np.cos(angles), u)
                                             + np.outer(np.sin(angles), v)))
    vertices = np.vstack(rings + [points[:1], points[-1:]])
    faces = []
    for i in range(n - 1):
        for k in range(sides):
            a, b = i * sides + k, i * sides + (k + 1) % sides
            c, d = a + sides, b + sides
            faces += [(a, b, d), (a, d, c)]
    base, tip = n * sides, n * sides + 1
    for k in range(sides):
        faces.append((base, (k + 1) % sides, k))
        last = (n - 1) * sides
        faces.append((tip, last + k, last + (k + 1) % sides))
    buffer.add(vertices, faces)


def _leaf(buffer, base, direction, normal, length, width, thickness=0.0025):
    """One leaf: an elliptic blade with a raised midrib, closed so it is a volume."""
    d = direction / np.linalg.norm(direction)
    n = normal - np.dot(normal, d) * d
    n /= np.linalg.norm(n)
    side = np.cross(n, d)
    local = np.array([
        (0.0, 0.0, 0.0),
        (0.28 * length, 0.44 * width, 0.002),
        (0.62 * length, 0.50 * width, 0.0),
        (length, 0.0, -0.004),
        (0.62 * length, -0.50 * width, 0.0),
        (0.28 * length, -0.44 * width, 0.002),
        (0.45 * length, 0.0, thickness + 0.003),
    ])
    world = base + local[:, :1] * d + local[:, 1:2] * side + local[:, 2:3] * n
    top = [(0, 1, 6), (1, 2, 6), (2, 3, 6), (3, 4, 6), (4, 5, 6), (5, 0, 6)]
    bottom = [(0, 5, 1), (1, 5, 4), (1, 4, 2), (2, 4, 3)]
    buffer.add(world, top + bottom)


def _curved_path(start, direction, length, rng, segments, bend, droop=0.0, wander=0.12):
    """A gently curving branch: bends toward `bend` (unit vector) along its length,
    optionally drooping under its own weight near the tip."""
    points = [np.asarray(start, float)]
    d = direction / np.linalg.norm(direction)
    step = length / segments
    for i in range(segments):
        t = (i + 1) / segments
        d = d + bend * 0.18 / segments * 4 + rng.normal(0, wander / segments * 2, 3)
        d[2] -= droop * t * 0.25
        d /= np.linalg.norm(d)
        points.append(points[-1] + d * step)
    return np.array(points)


class Tree:
    """Grown geometry split by material: wood, sunlit leaves, shaded leaves, fruit."""

    def __init__(self):
        self.wood, self.sun, self.shade, self.fruit = _Buffer(), _Buffer(), _Buffer(), _Buffer()
        self.leaf_points = []

    def meshes(self):
        return {k: getattr(self, k).mesh() for k in ("wood", "sun", "shade", "fruit")}

    def canopy_hull(self):
        if len(self.leaf_points) < 4:
            return None
        return trimesh.convex.convex_hull(np.array(self.leaf_points))


def _foliate(tree, path, rng, center, leaves_per_m, leaf_len, leaf_w, fruit=None,
             droop_leaves=0.0):
    """Leaves along the outer part of a twig/branch, each angled toward the light."""
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = lengths.sum()
    count = max(1, int(total * leaves_per_m))
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    for _ in range(count):
        s = rng.uniform(0.1, 1.0) * total
        k = min(np.searchsorted(cumulative, s) - 1, len(lengths) - 1)
        k = max(k, 0)
        u = (s - cumulative[k]) / max(lengths[k], 1e-9)
        point = path[k] * (1 - u) + path[k + 1] * u
        axis = path[k + 1] - path[k]
        axis /= np.linalg.norm(axis)
        outward = point - center
        outward[2] *= 0.5
        outward /= max(np.linalg.norm(outward), 1e-6)
        side = np.cross(axis, rng.normal(0, 1, 3))
        side /= max(np.linalg.norm(side), 1e-6)
        direction = 0.55 * side + 0.35 * outward + 0.25 * axis + rng.normal(0, 0.15, 3)
        direction[2] -= droop_leaves
        normal = np.array([0.0, 0.0, 1.0]) + 0.6 * outward + rng.normal(0, 0.35, 3)
        size = rng.uniform(0.8, 1.2)
        radial = np.linalg.norm((point - center)[:2])
        sunlit = (point[2] - center[2]) + 0.6 * radial + rng.normal(0, 0.25) > 0.35
        _leaf(tree.sun if sunlit else tree.shade, point + side * 0.01, direction, normal,
              leaf_len * size, leaf_w * size)
        tree.leaf_points.append(point)
    if fruit and rng.random() < fruit["chance"]:
        u = rng.uniform(0.4, 0.95)
        k = min(int(u * (len(path) - 1)), len(path) - 2)
        hang = path[k] + np.array([0.0, 0.0, -fruit["radius"] * 1.3])
        ball = trimesh.creation.uv_sphere(radius=fruit["radius"], count=(7, 8))
        ball.vertices *= (1.0, 1.0, fruit.get("squash", 0.92))
        tree.fruit.add(ball.vertices + hang, ball.faces)


# ------------------------------------------------------------------ base forms --


def _grow_tall_spindle(tree, base, rng, height, spread, leaf, density, fruit):
    """Trellised orchard row: a central leader with whorls of short fruit laterals."""
    leader = _curved_path(base + (0, 0, -0.05), np.array([0, 0, 1.0]), height, rng, 14,
                          bend=np.array([0, 0, 1.0]), wander=0.05)
    _tube(tree.wood, leader, np.linspace(0.075, 0.012, len(leader)), 10)
    center = base + (0, 0, height * 0.6)
    z, whorl = 0.75, 0
    while z < height - 0.25:
        k = int(z / height * (len(leader) - 1))
        origin = leader[k]
        reach = spread * (0.75 * (1.0 - (z - 0.75) / height) + 0.2)
        for j in range(int(rng.integers(3, 5))):
            azimuth = whorl * 1.1 + j * np.pi / 2 + rng.uniform(-0.4, 0.4)
            out = np.array([np.cos(azimuth), 0.55 * np.sin(azimuth), rng.uniform(-0.05, 0.25)])
            lateral = _curved_path(origin, out, reach * rng.uniform(0.8, 1.15), rng, 6,
                                   bend=np.array([0, 0, 1.0]) * 0.3, droop=0.9)
            _tube(tree.wood, lateral, np.linspace(0.028, 0.006, len(lateral)), 6)
            _foliate(tree, lateral, rng, center, 66 * density, *leaf, fruit=fruit)
            for _ in range(int(rng.integers(3, 6))):
                p = lateral[int(rng.integers(1, len(lateral) - 1))]
                twig = _curved_path(p, rng.normal(0, 1, 3) + (0, 0, 0.6),
                                    rng.uniform(0.15, 0.32), rng, 3,
                                    bend=np.array([0, 0, 1.0]), wander=0.3)
                _tube(tree.wood, twig, np.linspace(0.008, 0.003, len(twig)), 4)
                _foliate(tree, twig, rng, center, 80 * density, *leaf,
                         fruit=_lighter(fruit))
        z += rng.uniform(0.28, 0.38)
        whorl += 1
    _foliate(tree, leader[-8:], rng, center, 96 * density, *leaf)


def _grow_round_crown(tree, base, rng, height, spread, leaf, density, fruit,
                      trunk_range=(0.45, 0.65), gloss_low=True):
    """Garden/orchard broadleaf: short trunk, upright scaffolds, dense round crown."""
    trunk_h = rng.uniform(*trunk_range)
    stem = _curved_path(base + (0, 0, -0.05), np.array([0, 0, 1.0]), trunk_h, rng, 4,
                        bend=np.array([0, 0, 1.0]), wander=0.05)
    _tube(tree.wood, stem, np.linspace(0.10, 0.085, len(stem)), 10)
    crown_c = base + (0, 0, trunk_h + (height - trunk_h) * 0.5)
    half_h = (height - trunk_h) * 0.55

    def inside(p):
        q = (p - crown_c) / np.array([spread, spread, half_h])
        return np.dot(q, q) <= 1.0

    for s in range(int(rng.integers(4, 7))):
        azimuth = s * 2 * np.pi / 5 + rng.uniform(-0.3, 0.3)
        up = np.array([np.cos(azimuth) * 0.6, np.sin(azimuth) * 0.6, 1.0])
        scaffold = _curved_path(stem[-1], up, rng.uniform(0.75, 1.1) * (height - trunk_h) * 0.7,
                                rng, 7, bend=np.array([0, 0, 0.4]))
        _tube(tree.wood, scaffold, np.linspace(0.06, 0.018, len(scaffold)), 8)
        _foliate(tree, scaffold[2:], rng, crown_c, 48 * density, *leaf)
        for _ in range(int(rng.integers(8, 12))):
            p = scaffold[int(rng.integers(2, len(scaffold)))]
            out = np.array([np.cos(azimuth), np.sin(azimuth), 0.0]) + rng.normal(0, 0.7, 3)
            out[2] += 0.25
            branch = _curved_path(p, out, rng.uniform(0.5, 0.9) * max(spread, 0.8), rng, 5,
                                  bend=np.array([0, 0, 0.2]), droop=0.5 if gloss_low else 0.2)
            keep = [pt for pt in branch if inside(pt)] or [branch[0], branch[1]]
            branch = np.array(keep if len(keep) > 1 else branch[:2])
            _tube(tree.wood, branch, np.linspace(0.02, 0.006, len(branch)), 6)
            _foliate(tree, branch, rng, crown_c, 82 * density, *leaf, fruit=fruit)
            for _ in range(int(rng.integers(2, 4))):
                q = branch[int(rng.integers(0, len(branch)))]
                twig = _curved_path(q, rng.normal(0, 1, 3) + (0, 0, 0.3),
                                    rng.uniform(0.2, 0.35), rng, 3,
                                    bend=np.array([0, 0, 0.3]), wander=0.3)
                twig = np.array([pt for pt in twig if inside(pt)] or twig[:2])
                if len(twig) < 2:
                    continue
                _tube(tree.wood, twig, np.linspace(0.007, 0.003, len(twig)), 4)
                _foliate(tree, twig, rng, crown_c, 90 * density, *leaf,
                         fruit=_lighter(fruit))


def _grow_open_vase(tree, base, rng, height, spread, leaf, density, fruit):
    """Stone fruit: a low trunk splitting into 3–5 scaffolds around an open centre."""
    trunk_h = rng.uniform(0.5, 0.7)
    stem = _curved_path(base + (0, 0, -0.05), np.array([0, 0, 1.0]), trunk_h, rng, 4,
                        bend=np.array([0, 0, 1.0]), wander=0.04)
    _tube(tree.wood, stem, np.linspace(0.09, 0.075, len(stem)), 10)
    crown_c = base + (0, 0, height * 0.65)
    count = int(rng.integers(3, 6))
    for s in range(count):
        azimuth = s * 2 * np.pi / count + rng.uniform(-0.2, 0.2)
        out = np.array([np.cos(azimuth), np.sin(azimuth), 1.15])  # up and outward: the vase
        scaffold = _curved_path(stem[-1], out, (height - trunk_h) * rng.uniform(0.9, 1.1),
                                rng, 8, bend=np.array([np.cos(azimuth), np.sin(azimuth), -0.2]))
        _tube(tree.wood, scaffold, np.linspace(0.055, 0.012, len(scaffold)), 8)
        _foliate(tree, scaffold[3:], rng, crown_c, 50 * density, *leaf, fruit=fruit)
        for _ in range(int(rng.integers(5, 9))):
            p = scaffold[int(rng.integers(3, len(scaffold)))]
            branch = _curved_path(p, rng.normal(0, 0.6, 3) + out * 0.7,
                                  rng.uniform(0.4, 0.8) * max(spread, 0.8), rng, 4,
                                  bend=np.array([0, 0, -0.1]), droop=0.5)
            _tube(tree.wood, branch, np.linspace(0.015, 0.005, len(branch)), 5)
            _foliate(tree, branch, rng, crown_c, 85 * density, *leaf, fruit=fruit)


def _grow_broadleaf_high(tree, base, rng, height, spread, leaf, density, fruit):
    """Mature park broadleaf: tall clear trunk, wide crown lifted well off the ground."""
    trunk_h = height * rng.uniform(0.35, 0.45)
    stem = _curved_path(base + (0, 0, -0.05), np.array([0, 0, 1.0]), trunk_h, rng, 6,
                        bend=np.array([0, 0, 1.0]), wander=0.07)
    _tube(tree.wood, stem, np.linspace(0.16, 0.11, len(stem)), 12)
    crown_c = base + (0, 0, trunk_h + (height - trunk_h) * 0.55)
    half_h = (height - trunk_h) * 0.6

    def inside(p):
        q = (p - crown_c) / np.array([spread, spread, half_h])
        return np.dot(q, q) <= 1.05

    for s in range(int(rng.integers(5, 8))):
        azimuth = s * 2 * np.pi / 6 + rng.uniform(-0.3, 0.3)
        up = np.array([np.cos(azimuth), np.sin(azimuth), rng.uniform(0.5, 1.1)])
        limb = _curved_path(stem[-1], up, rng.uniform(0.7, 1.0) * spread * 1.2, rng, 8,
                            bend=np.array([0, 0, 0.25]))
        _tube(tree.wood, limb, np.linspace(0.085, 0.02, len(limb)), 8)
        _foliate(tree, limb[4:], rng, crown_c, 40 * density, *leaf)
        for _ in range(int(rng.integers(10, 15))):
            p = limb[int(rng.integers(2, len(limb)))]
            branch = _curved_path(p, rng.normal(0, 0.8, 3) + up * 0.6,
                                  rng.uniform(0.6, 1.1), rng, 5,
                                  bend=np.array([0, 0, 0.15]), droop=0.3)
            branch = np.array([pt for pt in branch if inside(pt)] or branch[:2])
            if len(branch) < 2:
                continue
            _tube(tree.wood, branch, np.linspace(0.022, 0.006, len(branch)), 5)
            _foliate(tree, branch, rng, crown_c, 75 * density, *leaf, fruit=fruit)


def _grow_conifer_spire(tree, base, rng, height, spread, leaf, density, fruit):
    """Fir/spruce: whorls of near-level branches shrinking steadily to a spire."""
    del fruit  # cones read as texture at robot scale; none are grown
    leader = _curved_path(base + (0, 0, -0.05), np.array([0, 0, 1.0]), height, rng, 12,
                          bend=np.array([0, 0, 1.0]), wander=0.03)
    _tube(tree.wood, leader, np.linspace(0.10, 0.01, len(leader)), 10)
    center = base + (0, 0, height * 0.5)
    needle = (leaf[0] * 0.5, leaf[1] * 0.28)
    z = height * 0.12
    while z < height * 0.94:
        t = z / height
        reach = spread * (1.0 - t) + 0.08
        k = int(t * (len(leader) - 1))
        for j in range(int(rng.integers(5, 8))):
            azimuth = j * 2 * np.pi / 6 + rng.uniform(-0.35, 0.35) + z * 2.1
            out = np.array([np.cos(azimuth), np.sin(azimuth), rng.uniform(-0.12, 0.02)])
            branch = _curved_path(leader[k], out, reach * rng.uniform(0.85, 1.1), rng, 4,
                                  bend=np.array([0, 0, 0.12]), wander=0.08)
            _tube(tree.wood, branch, np.linspace(0.02 * (1 - t) + 0.006, 0.003, len(branch)), 4)
            _foliate(tree, branch, rng, center, 150 * density, *needle)
        z += rng.uniform(0.3, 0.42)
    _foliate(tree, leader[-6:], rng, center, 170 * density, *needle)


def _grow_young_pine(tree, base, rng, height, spread, leaf, density, fruit):
    """Juvenile pine: sparse upswept whorls, foliage tufted near the branch tips."""
    del fruit
    leader = _curved_path(base + (0, 0, -0.05), np.array([0, 0, 1.0]), height, rng, 8,
                          bend=np.array([0, 0, 1.0]), wander=0.06)
    _tube(tree.wood, leader, np.linspace(0.055, 0.01, len(leader)), 8)
    center = base + (0, 0, height * 0.55)
    needle = (leaf[0] * 0.6, leaf[1] * 0.25)
    for z in np.linspace(height * 0.3, height * 0.9, int(rng.integers(3, 5))):
        k = int(z / height * (len(leader) - 1))
        for j in range(int(rng.integers(3, 5))):
            azimuth = j * 2 * np.pi / 4 + rng.uniform(-0.5, 0.5)
            out = np.array([np.cos(azimuth), np.sin(azimuth), rng.uniform(0.15, 0.4)])
            branch = _curved_path(leader[k], out, spread * rng.uniform(0.6, 1.0), rng, 4,
                                  bend=np.array([0, 0, 0.3]), wander=0.15)
            _tube(tree.wood, branch, np.linspace(0.014, 0.004, len(branch)), 4)
            _foliate(tree, branch[-2:], rng, center, 260 * density, *needle)
    _foliate(tree, leader[-4:], rng, center, 240 * density, *needle)


def _grow_columnar(tree, base, rng, height, spread, leaf, density, fruit):
    """Poplar/cypress: many tight upright branches hugging a narrow silhouette."""
    del fruit
    leader = _curved_path(base + (0, 0, -0.05), np.array([0, 0, 1.0]), height, rng, 12,
                          bend=np.array([0, 0, 1.0]), wander=0.03)
    _tube(tree.wood, leader, np.linspace(0.09, 0.015, len(leader)), 10)
    center = base + (0, 0, height * 0.55)
    width = min(spread, height * 0.16)
    z = height * 0.08
    while z < height * 0.92:
        k = int(z / height * (len(leader) - 1))
        for _j in range(int(rng.integers(2, 4))):
            azimuth = rng.uniform(0, 2 * np.pi)
            out = np.array([np.cos(azimuth) * 0.5, np.sin(azimuth) * 0.5, 1.4])
            branch = _curved_path(leader[k], out,
                                  rng.uniform(0.6, 1.2) * min(height - z, height * 0.3),
                                  rng, 5, bend=np.array([0, 0, 0.9]), wander=0.07)
            branch[:, :2] = np.clip(branch[:, :2] - base[:2], -width, width) + base[:2]
            _tube(tree.wood, branch, np.linspace(0.015, 0.004, len(branch)), 4)
            _foliate(tree, branch, rng, center, 110 * density, leaf[0] * 0.8, leaf[1] * 0.8)
        z += rng.uniform(0.3, 0.5)


def _grow_weeping(tree, base, rng, height, spread, leaf, density, fruit):
    """Willow: arching scaffolds whose tips trail long drooping leaf curtains."""
    del fruit
    trunk_h = height * rng.uniform(0.35, 0.5)
    stem = _curved_path(base + (0, 0, -0.05), np.array([0, 0, 1.0]), trunk_h, rng, 5,
                        bend=np.array([0, 0, 1.0]), wander=0.08)
    _tube(tree.wood, stem, np.linspace(0.13, 0.09, len(stem)), 10)
    crown_c = base + (0, 0, trunk_h)
    for s in range(int(rng.integers(5, 8))):
        azimuth = s * 2 * np.pi / 6 + rng.uniform(-0.3, 0.3)
        up = np.array([np.cos(azimuth), np.sin(azimuth), 1.2])
        arch = _curved_path(stem[-1], up, rng.uniform(0.8, 1.2) * spread, rng, 7,
                            bend=np.array([np.cos(azimuth), np.sin(azimuth), -1.2]))
        _tube(tree.wood, arch, np.linspace(0.05, 0.012, len(arch)), 6)
        for _ in range(int(rng.integers(5, 8))):
            p = arch[int(rng.integers(3, len(arch)))]
            fall = _curved_path(p, np.array([rng.normal(0, 0.2), rng.normal(0, 0.2), -1.0]),
                                rng.uniform(0.5, min(p[2] - base[2] - 0.2, height * 0.6)),
                                rng, 6, bend=np.array([0, 0, -1.0]), wander=0.05)
            _tube(tree.wood, fall, np.linspace(0.006, 0.002, len(fall)), 3)
            _foliate(tree, fall, rng, crown_c, 120 * density, leaf[0] * 0.9, leaf[1] * 0.45,
                     droop_leaves=0.8)


def _grow_sapling(tree, base, rng, height, spread, leaf, density, fruit):
    """A thin newcomer: one leader, a handful of twigs, believable in any verge."""
    leader = _curved_path(base + (0, 0, -0.02), np.array([0, 0, 1.0]), height, rng, 7,
                          bend=np.array([0, 0, 1.0]), wander=0.09)
    _tube(tree.wood, leader, np.linspace(0.022, 0.005, len(leader)), 6)
    center = base + (0, 0, height * 0.65)
    for _ in range(int(rng.integers(4, 7))):
        k = int(rng.uniform(0.35, 0.95) * (len(leader) - 1))
        twig = _curved_path(leader[k], rng.normal(0, 1, 3) + (0, 0, 0.7),
                            rng.uniform(0.2, 0.45) * max(spread, 0.5), rng, 3,
                            bend=np.array([0, 0, 0.4]), wander=0.25)
        _tube(tree.wood, twig, np.linspace(0.006, 0.002, len(twig)), 4)
        _foliate(tree, twig, rng, center, 70 * density, *leaf, fruit=_lighter(fruit))
    _foliate(tree, leader[-4:], rng, center, 80 * density, *leaf)


def _grow_dead_snag(tree, base, rng, height, spread, leaf, density, fruit):
    """A bare weathered trunk with broken limbs — habitat and obstacle, no leaves."""
    del leaf, density, fruit
    stem = _curved_path(base + (0, 0, -0.05), np.array([0, 0, 1.0]), height, rng, 9,
                        bend=np.array([0, 0, 1.0]), wander=0.09)
    _tube(tree.wood, stem, np.linspace(0.14, 0.03, len(stem)), 9)
    for _ in range(int(rng.integers(3, 6))):
        k = int(rng.uniform(0.3, 0.9) * (len(stem) - 1))
        out = rng.normal(0, 1, 3)
        out[2] = abs(out[2]) * 0.4
        limb = _curved_path(stem[k], out, rng.uniform(0.3, 0.8) * max(spread, 0.6), rng, 3,
                            bend=np.array([0, 0, -0.2]), wander=0.2)
        _tube(tree.wood, limb, np.linspace(0.045, 0.012, len(limb)), 5)
    # the snag's "crown" for collision purposes is its own wood
    tree.leaf_points.extend(stem[-4:].tolist())


_GROWERS = {
    "round_crown": _grow_round_crown,
    "tall_spindle": _grow_tall_spindle,
    "open_vase": _grow_open_vase,
    "broadleaf_high": _grow_broadleaf_high,
    "conifer_spire": _grow_conifer_spire,
    "young_pine": _grow_young_pine,
    "columnar": _grow_columnar,
    "weeping": _grow_weeping,
    "sapling": _grow_sapling,
    "dead_snag": _grow_dead_snag,
}

_DEFAULTS = {  # (height m, spread m, leaf_len m, leaf_w m)
    "round_crown": (4.0, 1.6, 0.10, 0.055),
    "tall_spindle": (3.0, 1.0, 0.09, 0.053),
    "open_vase": (3.2, 1.7, 0.095, 0.05),
    "broadleaf_high": (9.0, 3.4, 0.10, 0.06),
    "conifer_spire": (8.0, 1.8, 0.09, 0.05),
    "young_pine": (3.0, 1.1, 0.10, 0.05),
    "columnar": (7.0, 1.0, 0.075, 0.04),
    "weeping": (6.0, 2.6, 0.10, 0.05),
    "sapling": (1.8, 0.5, 0.08, 0.045),
    "dead_snag": (5.0, 1.0, 0.0, 0.0),
}


def _lighter(fruit):
    if not fruit:
        return None
    softened = dict(fruit)
    softened["chance"] = fruit["chance"] * 0.4
    return softened


def grow(form, base, *, seed=0, height=None, spread=None, leaf_scale=1.0,
         density=1.0, fruit=None):
    """Grow one tree of a hardcoded base form, dressed for the species at hand.

    `leaf_scale` sizes the blades (citrus ≈ 1.2, olive ≈ 0.6); `density` scales the
    leaf count (background rows at 0.3–0.5 keep forests inside the triangle budget);
    `fruit` = {"chance": 0..1, "radius": m, "squash": 1.0} hangs fruit where the form
    supports it. The same seed grows the same tree."""
    if form not in _GROWERS:
        raise ValueError(f"unknown base form {form!r}; forms: {', '.join(BASE_FORMS)}")
    d_height, d_spread, leaf_len, leaf_w = _DEFAULTS[form]
    if not 0.05 <= float(density) <= 3.0:
        raise ValueError("density must be within 0.05..3")
    tree = Tree()
    rng = np.random.default_rng(seed)
    _GROWERS[form](
        tree, np.asarray(base, float),
        rng,
        float(height or d_height),
        float(spread or d_spread),
        (leaf_len * leaf_scale, leaf_w * leaf_scale),
        float(density),
        fruit,
    )
    return tree


def place(world, name, tree, base, *, sun_color, shade_color, bark_color=BARK,
          fruit_color=None, fruit_class="fruit", drive=None, sway_axis=(1.0, 0.0, 0.0),
          wood_mass=30.0, foliage_mass=6.0, fruit_mass=2.0, trunk_radius=0.09):
    """Add a grown tree to the scene: wood collides as trunk + crown envelope (what a
    ground robot meets); leaves and fruit are visual. A swaying tree is the same
    parts as synced hinge bodies sharing one prescribed drive."""
    from nova_robotics.scene import RigidBody

    parts = tree.meshes()
    x, y, z = base
    trunk_piece = trimesh.creation.cylinder(radius=trunk_radius, height=0.9, sections=10)
    trunk_piece.apply_translation((x, y, z + 0.45))
    hull = tree.canopy_hull()
    colliders = [trunk_piece] + ([hull] if hull is not None else [])
    layers = (("sun", sun_color, "foliage", foliage_mass / 2),
              ("shade", shade_color, "foliage", foliage_mass / 2),
              ("fruit", fruit_color, fruit_class, fruit_mass))
    if drive is None:
        world.add_mesh(f"{name}_wood", parts["wood"], color=bark_color,
                       semantic_class="tree", collision_meshes=colliders, roughness=0.95)
        for key, color, cls, _mass in layers:
            if parts[key] is not None and color is not None:
                world.add_mesh(f"{name}_{key}", parts[key], color=color, semantic_class=cls,
                               collidable=False, roughness=0.6)
        return

    def body(mass, what):
        return RigidBody.box(
            mass=mass, size=(0.9, 0.9, 3.0), origin=(x, y, z + 0.05), motion="hinge",
            axis=sway_axis, limits=(-0.2, 0.2), damping=3.0, drive=drive,
            provenance=f"swaying tree {what}: {mass:g} kg, uniform box inertia about "
                       "the base hinge; wind motion is prescribed, not simulated")

    world.add_mesh(f"{name}_wood", parts["wood"], color=bark_color,
                   semantic_class="movable_prop", collision_meshes=colliders,
                   body=body(wood_mass, "wood"))
    for key, color, _cls, mass in layers:
        if parts[key] is not None and color is not None:
            world.add_mesh(f"{name}_{key}", parts[key], color=color,
                           semantic_class="movable_prop", collidable=False,
                           body=body(mass, key))
