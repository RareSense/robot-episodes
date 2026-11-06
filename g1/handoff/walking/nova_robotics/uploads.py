"""Bounded, passive robot-package inspection. Never execute uploaded code."""

import io
import posixpath
import stat
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import PurePosixPath
from urllib.parse import unquote

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_MEMBERS = 512
# Common extensionless notices are passive package data, not executable inputs.
NOTICE_NAMES = {"license", "licence", "copying", "notice", "authors", "readme"}
ASSET_SUFFIXES = {
    ".sdf",
    ".world",
    ".mjcf",
    ".config",
    ".xacro",
    ".srdf",
    ".msh",
    ".ply",
    ".tif",
    ".tiff",
    ".bmp",
    ".hdr",
    ".usd",
    ".usda",
    ".usdc",
    ".urdf",
    ".stl",
    ".obj",
    ".mtl",
    ".dae",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".exr",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".xml",
    ".glb",
    ".gltf",
    ".bin",
}


def package_members(name: str, data: bytes) -> dict[str, bytes]:
    """Keep archive paths intact; no filesystem extraction or URL fetching here."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Robot uploads must be at most 64 MB each.")
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = {}
            seen = set()
            expanded = 0
            if len(archive.infolist()) > MAX_MEMBERS:
                raise ValueError("Robot archive has too many entries")
            for info in archive.infolist():
                path = unicodedata.normalize("NFKC", info.filename).replace("\\", "/")
                parts = PurePosixPath(path)
                kind = stat.S_IFMT(info.external_attr >> 16)
                if (
                    parts.is_absolute()
                    or ".." in parts.parts
                    or not parts.parts
                    or ":" in parts.parts[0]
                    or any(ord(c) < 32 for c in path)
                    or info.flag_bits & 1
                    or kind not in {0, stat.S_IFREG, stat.S_IFDIR}
                ):
                    raise ValueError("Robot archive has an unsafe path, link or encrypted entry")
                key = parts.as_posix().casefold()
                if key in seen:
                    raise ValueError("Robot archive has duplicate portable paths")
                seen.add(key)
                expanded += info.file_size
                if (
                    expanded > MAX_EXPANDED_BYTES
                    or info.file_size > max(info.compress_size, 1) * 100
                ):
                    raise ValueError("Robot archive exceeds expanded size or compression limits")
                if info.is_dir():
                    continue
                if (
                    parts.suffix.lower() not in ASSET_SUFFIXES
                    and parts.name.casefold() not in NOTICE_NAMES
                ):
                    raise ValueError(f"Unsupported robot-package file: {path}")
                members[parts.as_posix()] = archive.read(info)
            return members
    if PurePosixPath(name).suffix.lower() not in ASSET_SUFFIXES and detect_format(
        name, data
    ) not in {"usd", "urdf", "sdf", "mjcf"}:
        raise ValueError("Unsupported data format. Use USD, URDF, SDF, MJCF/XML or an asset ZIP.")
    return {PurePosixPath(name).name: data}


def resolve_reference(owner: str, reference: str, names: set[str]) -> str | None:
    ref = unquote(reference).replace("\\", "/")
    if ref.startswith("package://"):
        tail = ref[len("package://") :]
        candidates = [p for p in names if p == tail or p.endswith("/" + tail)]
        if not candidates:
            # A package ZIP may omit its outer package-name directory.
            tail = tail.partition("/")[2]
            candidates = [p for p in names if p == tail]
        return candidates[0] if len(candidates) == 1 else None
    if ":" in ref or ref.startswith("/"):
        return None
    path = posixpath.normpath(posixpath.join(posixpath.dirname(owner), ref))
    return path if path in names else None


def inspect_urdf(data: bytes, path: str, names: set[str]) -> dict:
    root = read_xml(data)
    if root.tag != "robot" or not root.get("name"):
        raise ValueError("URDF must declare a named robot.")
    return _inspect_urdf_root(root, path, names)


def read_xml(data: bytes):
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("XML descriptions must be at most 10 MB.")
    text = data.decode("utf-8-sig")
    if "\x00" in text or "<!doctype" in text.lower() or "<!entity" in text.lower():
        raise ValueError("XML must not contain DTDs or entities.")
    return ET.fromstring(text)


def _inspect_urdf_root(root, path, names):
    links = [link.get("name") for link in root.findall("link")]
    if not links or None in links or len(set(links)) != len(links):
        raise ValueError("URDF link names must be present and unique.")
    joints = root.findall("joint")
    joint_names = [joint.get("name") for joint in joints]
    if None in joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError("URDF joint names must be present and unique.")
    parents = {}
    for joint in joints:
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            raise ValueError("URDF joint is missing a parent or child link.")
        p, c = parent.get("link"), child.get("link")
        if p not in links or c not in links or c in parents or p == c:
            raise ValueError("URDF has invalid joint link references.")
        parents[c] = p
    for link in links:
        seen = set()
        current = link
        while current in parents:
            if current in seen:
                raise ValueError("URDF joint graph contains a cycle.")
            seen.add(current)
            current = parents[current]
    if len(set(links) - set(parents)) != 1:
        raise ValueError("URDF must have one connected root link.")
    references = sorted(
        {node.get("filename", "") for node in root.iter() if node.tag in {"mesh", "texture"}}
    )
    return {
        "path": path,
        "name": root.get("name"),
        "format": "urdf",
        "role": "robot"
        if any(j.get("type") not in {None, "fixed"} for j in joints)
        else "unconfirmed",
        "links": len(links),
        "joints": len(joints),
        "missing_dependencies": [
            ref for ref in references if not resolve_reference(path, ref, names)
        ],
        "note": (
            "Source preserved. Import with Isaac Sim's URDF importer; dynamics are not verified."
        ),
    }


def detect_format(path: str, data: bytes) -> str:
    if data.startswith(b"PXR-USDC") or data.lstrip().startswith(b"#usda"):
        return "usd"
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".dae":
        return "asset"  # Large Collada geometry is companion data, not a robot XML schema.
    if suffix in {
        ".sdf",
        ".world",
        ".mjcf",
        ".urdf",
        ".xml",
        ".config",
        ".srdf",
        ".xacro",
    } or data.lstrip().startswith(b"<"):
        root = read_xml(data)
        if root.tag.rsplit("}", 1)[-1] == "COLLADA":
            return "asset"
        if suffix == ".xacro" or any("xacro" in e.tag for e in root.iter()):
            return "xacro"
        return {"robot": "urdf", "sdf": "sdf", "mujoco": "mjcf"}.get(root.tag, "xml")
    return "asset"


def inspect_simulation_xml(data: bytes, path: str, names: set[str], kind: str) -> dict:
    root = read_xml(data)
    warnings = ["Source context only; robot conversion and simulator execution are not performed."]
    references = []
    physics = []
    if kind == "sdf":
        if not root.get("version"):
            raise ValueError("SDF must declare its version")
        references = [e.text.strip() for e in root.iter("uri") if e.text and e.text.strip()]
        for world in root.findall("world"):
            for tag in ("gravity", "physics/max_step_size"):
                element = world.find(tag)
                if element is not None and element.text:
                    physics.append(
                        {
                            "field": f"{world.get('name', 'world')}/{tag}",
                            "value": element.text.strip(),
                            "source": "uploaded SDF",
                        }
                    )
        for collision in root.iter("collision"):
            for tag in (
                "surface/friction/ode/mu",
                "surface/friction/ode/mu2",
                "surface/bounce/restitution_coefficient",
            ):
                element = collision.find(tag)
                if element is not None and element.text:
                    physics.append(
                        {
                            "field": f"{collision.get('name', 'collision')}/{tag}",
                            "value": element.text.strip(),
                            "source": "uploaded SDF",
                        }
                    )
        if any(True for _ in root.iter("plugin")):
            warnings.append("SDF plugins are preserved but never executed or converted.")
    elif kind == "mjcf":
        compiler = root.find("compiler")
        options = compiler.attrib if compiler is not None else {}
        for element in root.iter():
            if "file" not in element.attrib:
                continue
            reference = element.attrib["file"]
            directory = options.get(
                "meshdir" if element.tag == "mesh" else "texturedir", options.get("assetdir", "")
            )
            if element.tag not in {"mesh", "texture", "hfield", "skin"}:
                directory = ""
            if options.get("strippath") == "true" and element.tag in {"mesh", "texture"}:
                reference = posixpath.basename(reference)
            references.append(posixpath.join(directory, reference))
        for element in root.iter("option"):
            physics.extend(
                {"field": key, "value": value, "source": "uploaded MJCF"}
                for key, value in element.attrib.items()
            )
        if root.find("include") is not None:
            warnings.append(
                "MJCF includes are inventoried, not expanded; "
                "inherited defaults need simulator validation."
            )
        if root.find("extension") is not None:
            warnings.append("MuJoCo extensions/plugins are not executed or converted.")
    else:
        warnings.append(
            "Auxiliary XML/Xacro preserved only. Macros and unknown schemas are not evaluated."
        )
    missing = []
    for ref in sorted(set(references)):
        resolved = resolve_reference(path, ref, names)
        if ref.startswith("model://"):
            tail = ref[len("model://") :].rstrip("/")
            candidates = [
                p
                for p in names
                if p == tail
                or p.endswith("/" + tail)
                or p == tail + "/model.sdf"
                or p.endswith("/" + tail + "/model.sdf")
            ]
            resolved = candidates[0] if len(candidates) == 1 else None
        if not resolved:
            missing.append(ref)
    return {
        "path": path,
        "format": kind,
        "role": (
            "robot"
            if kind == "sdf"
            and any(
                m.findtext("static", "false").strip().lower() not in {"true", "1"}
                and any(j.get("type") not in {None, "fixed"} for j in m.findall("joint"))
                for m in root.findall("model")
            )
            else "robot"
            if kind == "mjcf"
            and any(
                b.find("joint") is not None or b.find("freejoint") is not None
                for b in root.findall("worldbody/body") + root.findall("worldbody/body//body")
            )
            else "environment"
            if kind in {"sdf", "mjcf"}
            else "auxiliary"
        ),
        "name": root.get("name") or root.get("model"),
        # A free root joint means the base can move: what a walking or carried-base
        # controller needs. None where the format leaves it to the loader.
        "mobile_base": (
            any(
                b.find("freejoint") is not None
                or any(j.get("type") == "free" for j in b.findall("joint"))
                for b in root.findall("worldbody/body")
            )
            if kind == "mjcf"
            else None
        ),
        "missing_dependencies": missing,
        "physics_values": physics[:100],
        "warnings": warnings,
        "conversion": "not_performed",
    }


def robot_inventory(name: str, data: bytes) -> dict:
    members = package_members(name, data)
    models = []
    for path, content in members.items():
        kind = detect_format(path, content)
        if kind == "urdf":
            models.append(inspect_urdf(content, path, set(members)))
        elif kind == "usd":
            models.append(
                {
                    "path": path,
                    "format": "usd",
                    "role": "unconfirmed",
                    "dependency_check": "pending_build",
                }
            )
        elif kind in {"sdf", "mjcf", "xml", "xacro"}:
            models.append(inspect_simulation_xml(content, path, set(members), kind))
        elif PurePosixPath(path).suffix.lower() in {".usd", ".usda", ".usdc"}:
            raise ValueError(f"Invalid USD header: {path}")
    return {
        "purpose": "robot",
        "files": len(members),
        "models": models,
        "expanded_bytes": sum(map(len, members.values())),
        "ready_for_execution": False,
        "note": "Source saved. Description-level dependencies are inspected; transitive "
        "mesh/material/plugin dependencies may require simulator validation. "
        "USD inspection runs inside the isolated build.",
    }
