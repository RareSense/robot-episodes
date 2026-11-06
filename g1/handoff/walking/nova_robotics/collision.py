"""Host-side contact geometry: convex pieces for every concave part, on every target.

MuJoCo collides a mesh through its convex hull, and every target hulls a movable body
that has no explicit pieces. A wall with a doorway therefore became a solid slab and a
desk carcass swallowed its own drawers unless the author hand-built collision pieces.
This module derives those pieces automatically: connected components that are already
convex become one hull each, concave components are decomposed with CoACD, tiny bits
(screw heads) are dropped, and the result is capped so a part stays simulable.
Explicit `collision_meshes` always win; automatic pieces never replace an author's.
"""

import numpy as np
import trimesh

MAX_PIECES = 32
# A static assembly of many separate convex members (a rack row's beams, a fence's
# pickets) keeps one piece per member, merged where members line up; a robot must be
# able to reach between them. Static geoms are cheap in every engine's broadphase.
MAX_STATIC_PIECES = 400
MERGE_SLACK = 0.12
CONVEX_TOLERANCE = 1.03
TINY_FRACTION = 0.001
TINY_VOLUME = 1e-6
_COACD = {
    "threshold": 0.06,
    "max_convex_hull": 12,
    "seed": 0,
    "preprocess_mode": "auto",
    "preprocess_resolution": 30,
    "mcts_nodes": 10,
    "mcts_iterations": 60,
    "mcts_max_depth": 2,
}


def _volume(mesh) -> float:
    return float(abs(mesh.volume)) if mesh.is_volume else 0.0


def _components(mesh):
    merged = mesh.copy()
    merged.merge_vertices(merge_tex=True, merge_norm=True)
    parts = merged.split(only_watertight=False, repair=False)
    return list(parts) if len(parts) else [merged]


def _decompose(component):
    import coacd

    coacd.set_log_level("off")
    vertices = np.ascontiguousarray(component.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(component.faces, dtype=np.int64)
    parts = coacd.run_coacd(coacd.Mesh(vertices, faces), **_COACD)
    hulls = []
    for v, f in parts:
        hull = trimesh.Trimesh(v, f, process=False).convex_hull
        if hull.is_volume and _volume(hull) > 0:
            hulls.append(hull)
    return hulls


def _merge_aligned(pieces, slack=MERGE_SLACK, limit=MAX_STATIC_PIECES):
    """Merge neighbouring convex pieces whose joint hull adds almost no empty volume:
    collinear beam segments become one bar, stacked slats one board. Never merges
    across a real gap — the joint hull must stay within `slack` of the parts' sum."""
    pieces = list(pieces)
    for _round in range(4):
        before = len(pieces)
        for axis in (0, 1, 2):
            others = [a for a in (0, 1, 2) if a != axis]
            pieces.sort(key=lambda p, axis=axis, others=others: (
                *np.round(p.centroid[others] / 0.05), p.bounds[0][axis]))
            merged = [pieces[0]]
            for piece in pieces[1:]:
                last = merged[-1]
                lo = np.maximum(last.bounds[0], piece.bounds[0])
                hi = np.minimum(last.bounds[1], piece.bounds[1])
                if np.any(lo - hi > 0.02):  # not touching: never bridge a gap
                    merged.append(piece)
                    continue
                joint = trimesh.convex.convex_hull(np.vstack([last.vertices, piece.vertices]))
                total = _volume(last) + _volume(piece)
                if joint.is_volume and _volume(joint) <= (1 + slack) * total:
                    merged[-1] = joint
                else:
                    merged.append(piece)
            pieces = merged
        if len(pieces) == before or len(pieces) <= limit:
            break
    return pieces


OBJECT_SCALE_M = 1.5


def _clusters(components, tolerance=0.01):
    """Groups of components that touch (bounding boxes within `tolerance`): one pallet's
    slats and blocks, one frame's uprights and braces."""
    lows = np.array([c.bounds[0] for c in components]) - tolerance
    highs = np.array([c.bounds[1] for c in components]) + tolerance
    parent = list(range(len(components)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(components)):
        touching = np.all((lows[i] <= highs[i + 1:]) & (lows[i + 1:] <= highs[i]), axis=1)
        for j in np.nonzero(touching)[0] + i + 1:
            parent[find(i)] = find(int(j))
    groups: dict[int, list] = {}
    for i, component in enumerate(components):
        groups.setdefault(find(i), []).append(component)
    return list(groups.values())


def _assembly_pieces(components):
    """Pieces for a part made of many separate members (a rack row, a pallet stack).
    An object-sized cluster (a pallet, a carton) collides as its own hull — what a
    gripper or fork meets; a larger cluster (a frame, a beam run) keeps its members,
    merged where they line up, so the openings between them stay open."""
    pieces = []
    for group in _clusters(components):
        joint = trimesh.convex.convex_hull(np.vstack([c.vertices for c in group]))
        if float(np.max(joint.extents)) <= OBJECT_SCALE_M:
            pieces.append(joint)
            continue
        material = sum(_volume(c) for c in group)
        members = [c.convex_hull for c in group
                   if _volume(c.convex_hull) >= max(TINY_VOLUME, TINY_FRACTION * material)]
        pieces.extend(_merge_aligned(members) if len(members) > 1 else members)
    return pieces


def convex_pieces(mesh) -> tuple[list, dict]:
    """Convex volumes that cover `mesh` without filling its cavities.

    Returns ([], report) when the whole mesh is already convex: one hull is exact and
    the exporters keep their single-geom path. Otherwise the pieces are convex hulls
    of components (or CoACD parts of concave components), largest first.
    """
    hull = mesh.convex_hull
    merged = mesh.copy()
    merged.merge_vertices(merge_tex=True, merge_norm=True)
    whole = _volume(merged)
    if whole > 0 and _volume(hull) <= CONVEX_TOLERANCE * whole + 1e-9:
        return [], {"source": "convex", "pieces": 0, "components": 1, "decomposed": 0}
    pieces, decomposed, failed = [], 0, 0
    components = _components(merged)
    sample = components[:200]
    convex_share = sum(
        1 for c in sample
        if _volume(c) > 0 and _volume(c.convex_hull) <= 1.25 * _volume(c)
    ) / max(1, len(sample))
    if len(components) > MAX_PIECES and convex_share >= 0.8:
        # An assembly of convex members: cluster it instead of decomposing each one.
        kept = _assembly_pieces(components)
        kept.sort(key=_volume, reverse=True)
        dropped = max(0, len(kept) - MAX_STATIC_PIECES)
        kept = kept[:MAX_STATIC_PIECES]
        return kept, {
            "source": "auto_assembly_pieces", "pieces": len(kept),
            "components": len(components), "decomposed": 0,
            "decomposition_failures": 0, "dropped_small_pieces": dropped,
        }
    for component in components:
        volume = _volume(component)
        if volume == 0:
            # Open shell: CoACD remeshes it; fall back to its hull if that fails.
            try:
                parts = _decompose(component)
                decomposed += 1
            except Exception:
                parts, failed = [], failed + 1
            pieces.extend(parts or [component.convex_hull])
            continue
        if _volume(component.convex_hull) <= CONVEX_TOLERANCE * volume + 1e-9:
            pieces.append(component.convex_hull)
            continue
        try:
            parts = _decompose(component)
            decomposed += 1
        except Exception:
            parts, failed = [], failed + 1
        pieces.extend(parts or [component.convex_hull])
    # "Tiny" is measured against the part's own material, not its bounding hull: a
    # 30 m rack row's hull is mostly air, and against it every beam looked like a
    # screw head — all 480 were dropped and the row collided as one solid wall.
    reference = max(sum(_volume(p) for p in pieces if p.is_volume), TINY_VOLUME)
    kept = [
        p
        for p in pieces
        if p.is_volume and _volume(p) >= max(TINY_VOLUME, TINY_FRACTION * reference)
    ]
    dropped = len(pieces) - len(kept)
    many = len(kept) > MAX_PIECES
    if many:
        kept = _merge_aligned(kept)
    limit = MAX_STATIC_PIECES if many else MAX_PIECES
    kept.sort(key=_volume, reverse=True)
    if len(kept) > limit:
        dropped += len(kept) - limit
        kept = kept[:limit]
    if not kept:
        # Only a single body may fall back to its hull; an assembly of members never
        # becomes one slab.
        kept = [hull] if len(_components(merged)) == 1 else [
            p.convex_hull for p in _components(merged)][:MAX_STATIC_PIECES]
    return kept, {
        "source": "auto_convex_pieces",
        "pieces": len(kept),
        "components": len(_components(merged)),
        "decomposed": decomposed,
        "decomposition_failures": failed,
        "dropped_small_pieces": dropped,
    }


def ensure_contact(scene) -> dict:
    """Fill `auto_collision` for every collidable object that has no explicit pieces.

    Idempotent: an object already resolved (tuple, possibly empty) is skipped, so the
    compiler, the native exporters and the USD exporter can each call this safely.
    """
    report = {
        "resolved": 0, "convex": 0, "decomposed_objects": 0, "pieces": 0, "terrain": 0,
        "objects": {},
    }
    for obj in scene.objects:
        if not obj.collidable or obj.collision_meshes is not None:
            continue
        if obj.heightfield is not None:
            # Ground keeps its elevation grid: decomposing a hill into convex blobs is
            # exactly the loss of detail terrain exists to avoid.
            obj.auto_collision = ()
            obj.auto_collision_report = {"source": "heightfield", "pieces": 0}
            report["terrain"] += 1
            continue
        if obj.auto_collision is None:
            pieces, detail = convex_pieces(obj.mesh)
            obj.auto_collision = tuple(pieces)
            obj.auto_collision_report = detail
        detail = obj.auto_collision_report or {}
        report["resolved"] += 1
        if obj.auto_collision:
            report["decomposed_objects"] += 1
            report["pieces"] += len(obj.auto_collision)
            report["objects"][obj.name] = detail
        else:
            report["convex"] += 1
    return report


def contact_pieces(obj):
    """The convex pieces a target must use for this object, or None when its single
    convex hull is exact (or the object is not collidable)."""
    if not obj.collidable:
        return None
    if obj.collision_meshes is not None:
        return obj.collision_meshes
    if obj.auto_collision:
        return obj.auto_collision
    return None
