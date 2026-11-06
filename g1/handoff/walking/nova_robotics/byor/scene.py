"""Prepare an existing Nova simulation for a reproducible manipulation task.

The source export is immutable. All task setup (movable objects, initial placements,
and rigid assemblies) is explicit in the returned provenance. Objects only receive
new initial poses here, before an episode; task execution uses physical contacts.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def prepare_environment(path, *, active_joints=None, placements=None, rigid_groups=()):
    path = Path(path).resolve()
    raw = path.read_bytes()
    root = ET.fromstring(raw)
    bodies = {b.get("name"): b for b in root.iter("body")}
    changes = []
    for child_name, parent_name in rigid_groups:
        child, parent = bodies[child_name], bodies[parent_name]
        # Generated parts with the same hinge frame form one rigid assembly.
        # Refuse a guessed transform: callers must provide aligned source frames.
        for attr, default in (("pos", "0 0 0"), ("quat", "1 0 0 0")):
            if not np.allclose(
                np.fromstring(child.get(attr, default), sep=" "),
                np.fromstring(parent.get(attr, default), sep=" "),
            ):
                raise ValueError(f"rigid parts have different {attr}: {child_name}, {parent_name}")
        owner = next(x for x in root.iter() if child in list(x))
        if parent not in list(owner):
            raise ValueError("rigid parts must share a source parent")
        owner.remove(child)
        child.set("pos", "0 0 0")
        child.set("quat", "1 0 0 0")
        for joint in list(child):
            if joint.tag in ("joint", "freejoint"):
                child.remove(joint)
        parent.append(child)
        changes.append({"rigid_part": child_name, "parent": parent_name})
    if active_joints is not None:
        active = set(active_joints)
        known = {
            j.get("name") for b in root.iter("body") for j in b if j.tag in ("joint", "freejoint")
        }
        if active - known:
            raise ValueError(f"unknown active joints: {sorted(active - known)}")
        for body in root.iter("body"):
            for joint in list(body):
                if joint.tag in ("joint", "freejoint") and joint.get("name") not in active:
                    body.remove(joint)
                    changes.append(
                        {"fixed_at_initial_pose": body.get("name"), "joint": joint.get("name")}
                    )
    for name, position in (placements or {}).items():
        body = bodies[name]
        if body.find("freejoint") is None:
            raise ValueError(f"placement requires a movable free body: {name}")
        pos = np.asarray(position, float)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError(f"invalid placement for {name}")
        changes.append({"initial_placement": name, "from": body.get("pos"), "to": pos.tolist()})
        body.set("pos", " ".join(map(str, pos)))
    # Scene drives must never become robot motors when the robot is analysed.
    # Autonomous scene machinery needs its own controller, outside this task fixture.
    for tag in ("actuator", "keyframe"):
        node = root.find(tag)
        if node is not None:
            if len(node):
                changes.append({"removed_scene_section": tag, "count": len(node)})
            root.remove(node)
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""
    for item in root.findall("./asset/*"):
        if item.get("file"):
            base = texturedir if item.tag == "texture" else meshdir
            item.set("file", str((path.parent / base / item.get("file")).resolve()))
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    provenance = {
        "source": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "changes": changes,
        "active_joints": sorted(active_joints) if active_joints is not None else None,
    }
    return spec, provenance
