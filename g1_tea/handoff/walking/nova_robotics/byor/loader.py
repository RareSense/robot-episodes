"""Load any uploaded robot (MJCF or URDF) into a MuJoCo model that is ready for real control.

Everything this module adds to the customer's description is recorded in `LoadedRobot.added`, so
the report can say exactly what came from the file and what Nova supplied:

- URDF has no actuators: every driven joint gets a torque motor whose limit is the URDF `effort`
  (MuJoCo stores it as the joint's actuatorfrcrange). Mimic joints become joint couplings.
- URDF velocity limits are kept (MuJoCo's importer drops them) for the motor speed check.
- A floor plane is added when the scene has none.
- Base: an MJCF says what it is (free joint or not). A URDF cannot express a free-floating base,
  so `base="auto"` decides from geometry: if, released, the robot would stand on legs or wheels it
  is mobile, otherwise it is a mounted arm.
- Collision pairs that already overlap in the rest pose are approximations in the collision
  meshes (MoveIt's setup assistant disables the same pairs); they are excluded and listed.
"""
from __future__ import annotations

import posixpath
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from ..uploads import read_xml, resolve_reference

MESH_FORMATS = {".stl", ".obj", ".msh"}


@dataclass
class LoadedRobot:
    model: mujoco.MjModel
    spec: mujoco.MjSpec
    source_format: str
    base: str
    added: list[str] = field(default_factory=list)
    velocity_limits: dict[str, float] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def xml(self) -> str:
        return self.spec.to_xml()


def _convert_mesh(src: Path, cache: Path) -> Path:
    import trimesh

    out = cache / (src.stem + "_" + str(abs(hash(str(src))) % 10**8) + ".stl")
    if not out.exists():
        mesh = trimesh.load(src, force="mesh")
        mesh.export(out)
    return out


def _urdf_to_mjcf_ready(path: Path, root_dir: Path, cache: Path):
    """URDF text with every mesh resolved to an absolute, MuJoCo-readable file, plus
    the joint velocity limits (MuJoCo's importer drops them) and the mimic couplings."""
    import xml.etree.ElementTree as ET

    tree = read_xml(path.read_bytes())
    # physics only needs collision geometry; visual meshes (often COLLADA) are rendered elsewhere
    for link in tree.iter("link"):
        for vis in list(link.findall("visual")):
            link.remove(vis)
    names = {p.relative_to(root_dir).as_posix() for p in root_dir.rglob("*") if p.is_file()}
    owner = path.relative_to(root_dir).as_posix()
    problems = []
    for mesh in tree.iter("mesh"):
        ref = mesh.get("filename", "")
        rel = resolve_reference(owner, ref, names) or _nearest_match(owner, ref, names)
        if rel is None:
            problems.append(f"mesh not found in the upload: {ref}")
            continue
        file = root_dir / rel
        if file.suffix.lower() not in MESH_FORMATS:
            try:
                file = _convert_mesh(file, cache)
            except Exception as exc:                       # noqa: BLE001 - reported to the user
                raise ValueError(f"Collision mesh {rel} could not be read ({type(exc).__name__}: {exc}). "
                                 "Export the robot's collision meshes as STL or OBJ.") from exc
        mesh.set("filename", str(file))
    velocity, mimic = {}, []
    for joint in tree.iter("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("velocity") not in (None, "", "0"):
            velocity[joint.get("name")] = float(limit.get("velocity"))
        mm = joint.find("mimic")
        if mm is not None:
            mimic.append((joint.get("name"), mm.get("joint"), float(mm.get("multiplier", 1)), float(mm.get("offset", 0))))
    ext = tree.find("mujoco")
    if ext is None:
        ext = ET.SubElement(tree, "mujoco")
    comp = ext.find("compiler")
    if comp is None:
        comp = ET.SubElement(ext, "compiler")
    comp.set("strippath", "false")
    comp.set("discardvisual", "true")
    comp.set("fusestatic", "false")          # keep the base link its own body (it may float)
    return ET.tostring(tree, encoding="unicode"), velocity, mimic, problems


def _nearest_match(owner: str, ref: str, names: set[str]) -> str | None:
    """package:// reference matching several copies: take the one sharing the longest directory
    prefix with the URDF itself (the package it ships with)."""
    if not ref.startswith("package://"):
        return None
    tail = ref[len("package://"):]
    cands = [p for p in names if p.endswith("/" + tail) or p == tail]
    if not cands:
        low = tail.lower()
        cands = [p for p in names if p.lower().endswith("/" + low) or p.lower() == low]
    if not cands:
        return None
    parts = owner.split("/")
    def shared(p):
        k = 0
        for a, b in zip(parts, p.split("/")):
            if a != b:
                break
            k += 1
        return k
    cands.sort(key=shared, reverse=True)
    if len(cands) > 1 and shared(cands[0]) == shared(cands[1]):
        return None
    return cands[0]


def _convex_hull_copy(src: Path, cache: Path) -> Path:
    """Collision in MuJoCo uses a mesh's convex hull, so a hull with far fewer faces is the same
    collision shape (inertia comes from the description's explicit values, not the mesh)."""
    import trimesh

    out = cache / (src.stem + "_hull_" + str(abs(hash(str(src))) % 10**8) + ".stl")
    if not out.exists():
        trimesh.load(src, force="mesh").convex_hull.export(out)
    return out


def _compile_with_repairs(spec: mujoco.MjSpec, cache: Path, problems: list, added: list):
    """Compile; on known recoverable errors make the minimal, reported change and retry."""
    import re

    for _ in range(64):
        try:
            return spec.compile()
        except ValueError as exc:
            msg = str(exc)
            bad = re.search(r"(?:decoder failed for|number of faces).*?mesh file '([^']+)'", msg) or \
                re.search(r"mesh file '([^']+)'.*(?:faces|decoder)", msg)
            if bad:
                path = bad.group(1)
                for mesh in spec.meshes:
                    if mesh.file == path or mesh.file.endswith(Path(path).name):
                        mesh.file = str(_convex_hull_copy(Path(mesh.file), cache))
                added.append(f"mesh {Path(path).name} is too detailed for MuJoCo; used its convex hull "
                             "(the same collision shape)")
                continue
            if "positive eigenvalues" in msg or "inertia" in msg and spec.compiler.balanceinertia:
                raise ValueError("The robot description has a physically impossible inertia matrix "
                                 f"({msg.replace('Error: ', '')}). Supply real inertia values.") from exc
            if "inertia must satisfy" in msg and not spec.compiler.balanceinertia:
                spec.compiler.balanceinertia = True
                problems.append("physically impossible inertia (violates A + B >= C) in the description; "
                                "balanced only so the robot can be analysed. Supply the real inertia values.")
                continue
            raise
    raise ValueError("robot description could not be compiled")


def _has_floor(spec: mujoco.MjSpec) -> bool:
    return any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms)


def _add_floor(spec: mujoco.MjSpec):
    g = spec.worldbody.add_geom()
    g.name = "nova_floor"
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [0, 0, 0.05]
    g.friction = [1.0, 0.005, 0.0001]


def _add_motors(spec: mujoco.MjSpec, m: mujoco.MjModel, skip: set[str]) -> int:
    actuated = {spec.actuators[i].target for i in range(len(spec.actuators))}
    n = 0
    for j in range(m.njnt):
        if int(m.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jn in actuated or jn in skip or not m.jnt_actfrclimited[j] or m.jnt_actfrcrange[j, 1] <= 0:
            continue                               # no effort limit in the URDF: a passive joint
        a = spec.add_actuator()
        a.name = f"{jn}_motor"
        a.target = jn
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gainprm[0] = 1.0
        a.ctrllimited = True
        a.ctrlrange = m.jnt_actfrcrange[j].tolist()
        n += 1
    return n


def _would_stand(spec_xml: str, root_name: str) -> bool:
    """Released, would the robot rest on wheels, or on legs that hold its body clear of the floor?
    A mounted arm rests on its own base instead (its base is among the lowest parts)."""
    from .analysis import analyze_model, collision_geoms, geom_min_z, rest_pose

    s = mujoco.MjSpec.from_string(spec_xml)
    s.body(root_name).add_freejoint()
    if not _has_floor(s):
        _add_floor(s)
    m = s.compile()
    rep = analyze_model(m, source="probe")
    if rep.wheels:
        return True
    if not rep.feet:
        return False
    d = rest_pose(m)
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_name)
    low = min(geom_min_z(m, d, g) for b in range(m.nbody) for g in collision_geoms(m, b) if m.body_rootid[b] == root)
    welded, frontier = {root}, [root]
    while frontier:                               # the base and everything rigidly fixed to it
        p = frontier.pop()
        for b in range(m.nbody):
            if m.body_parentid[b] == p and m.body_jntnum[b] == 0 and b not in welded:
                welded.add(b)
                frontier.append(b)
    base_low = min([geom_min_z(m, d, g) for b in welded for g in collision_geoms(m, b)] or [np.inf])
    return base_low - low > 0.05


def _exclude_rest_overlaps(spec: mujoco.MjSpec) -> list[str]:
    from .stance import start_pose

    m = spec.compile()
    d = mujoco.MjData(m)
    start_pose(m, d)
    pairs = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if b1 == 0 or b2 == 0 or m.body_weldid[b1] == m.body_weldid[b2]:
            continue
        r1, r2 = int(m.body_rootid[b1]), int(m.body_rootid[b2])
        if r1 != r2 or c.dist > -1e-4:
            continue
        pairs.add(tuple(sorted((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1),
                                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2)))))
    for a, b in sorted(pairs):
        e = spec.add_exclude()
        e.bodyname1, e.bodyname2 = a, b
    return [f"{a} / {b}" for a, b in sorted(pairs)]


def load_robot(path: str | Path, root_dir: str | Path | None = None, base: str = "auto",
               cache_dir: str | Path | None = None) -> LoadedRobot:
    path = Path(path).resolve()
    root_dir = Path(root_dir).resolve() if root_dir else path.parent
    cache = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="nova_byor_"))
    cache.mkdir(parents=True, exist_ok=True)
    added, problems, velocity, mimic = [], [], {}, []
    if path.suffix.lower() == ".urdf" or read_xml(path.read_bytes()).tag == "robot":
        text, velocity, mimic, problems = _urdf_to_mjcf_ready(path, root_dir, cache)
        spec = mujoco.MjSpec.from_string(text)
        fmt = "urdf"
    else:
        spec = mujoco.MjSpec.from_file(str(path))
        fmt = "mjcf"
    if problems and any("not found" in p for p in problems):
        raise ValueError("; ".join(problems))
    m = _compile_with_repairs(spec, cache, problems, added)

    # ---- base
    tops = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0]
    floating = any(int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) and m.jnt_bodyid[j] in tops
                   for j in range(m.njnt))
    if not floating and fmt == "urdf" and tops:
        root = max(tops, key=lambda b: m.body_subtreemass[b])
        rname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, root)
        want = base == "floating" or (base == "auto" and _would_stand(spec.to_xml(), rname))
        if want:
            spec.body(rname).add_freejoint()
            floating = True
            added.append(f"free-floating base on '{rname}' (the robot stands on its own legs or wheels)")
        else:
            added.append(f"'{rname}' bolted to the floor (mounted manipulator)")
    elif base == "fixed" and floating:
        raise ValueError("This robot description has a free-floating base; it cannot be mounted as fixed.")

    # ---- mimic joints -> couplings, never separately driven
    skip = set()
    existing = {frozenset((e.name1, e.name2)) for e in spec.equalities if e.type == mujoco.mjtEq.mjEQ_JOINT}
    for follower, leader, mult, off in mimic:
        if spec.joint(follower) is None or spec.joint(leader) is None:
            continue
        skip.add(follower)
        if frozenset((follower, leader)) in existing:     # MuJoCo's importer already coupled them
            continue
        e = spec.add_equality()
        e.type = mujoco.mjtEq.mjEQ_JOINT
        e.name1, e.name2 = follower, leader
        e.data[:5] = [off, mult, 0, 0, 0]
        added.append(f"coupling {follower} = {mult} x {leader} + {off} (URDF mimic)")
    m = _compile_with_repairs(spec, cache, problems, added)

    # ---- motors
    if fmt == "urdf":
        n = _add_motors(spec, m, skip)
        if n:
            added.append(f"{n} torque motors, limits from the URDF effort values "
                         "(joints without an effort limit are left passive)")
    if not _has_floor(spec):
        _add_floor(spec)
        added.append("floor plane")
    if fmt == "urdf":
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.timestep = 0.002
    # MuJoCo friction is a soft constraint: a sustained sideways load makes feet creep millimetres
    # per second, which real feet below their friction limit do not. MuJoCo's no-slip post-solver
    # removes that creep without changing the model's friction cone or contact parameters.
    if spec.option.noslip_iterations < 10:
        spec.option.noslip_iterations = 10
        added.append("no-slip friction pass (10 iterations): stops numerical foot creep")
    excluded = _exclude_rest_overlaps(spec)
    if excluded:
        added.append(f"{len(excluded)} collision pairs excluded because they overlap at rest: " + ", ".join(excluded[:12])
                     + (" ..." if len(excluded) > 12 else ""))
    m = spec.compile()
    return LoadedRobot(model=m, spec=spec, source_format=fmt, base="floating" if floating else "fixed",
                       added=added, velocity_limits=velocity, problems=problems)
