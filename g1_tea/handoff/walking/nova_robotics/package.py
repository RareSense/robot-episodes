"""Validate portable simulator packages before any native parser sees their assets."""

import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath


def unpack_package(data, directory):
    directory = Path(directory)
    if len(data) > 256 * 1024 * 1024 or not zipfile.is_zipfile(io.BytesIO(data)):
        raise ValueError("Expected a simulation ZIP of at most 256 MB")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 10000 or sum(e.file_size for e in entries) > 512 * 1024 * 1024:
            raise ValueError("Simulation package exceeds extraction limits")
        names = set()
        for entry in entries:
            path = PurePosixPath(entry.filename)
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if (
                path.is_absolute() or ".." in path.parts or "\\" in entry.filename
                or ":" in entry.filename or str(path) in names or not path.parts
                or any(ord(c) < 32 for c in entry.filename) or entry.flag_bits & 1
                or kind not in {0, stat.S_IFREG, stat.S_IFDIR}
            ):
                raise ValueError("Unsafe or duplicate simulation-package path")
            names.add(str(path))
            if entry.is_dir():
                continue
            if entry.file_size > 128 * 1024 * 1024:
                raise ValueError("Simulation asset exceeds its size limit")
            with archive.open(entry) as stream:
                content = stream.read(entry.file_size + 1)
            if len(content) != entry.file_size:
                raise ValueError("Truncated simulation asset")
            destination = directory.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    if manifest.get("schema_version") != "nova.robotics-package/v1":
        raise ValueError("Unsupported simulation package")
    declared = set()
    for item in manifest.get("files", []):
        name = item["path"]
        path = (directory / name).resolve()
        if (
            name in declared or not path.is_relative_to(directory.resolve())
            or not path.is_file() or name == "manifest.json"
        ):
            raise ValueError("Unsafe or duplicate manifest entry")
        declared.add(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Simulation package checksum mismatch")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != declared | {"manifest.json"}:
        raise ValueError("Simulation package contains unlisted files")
    return manifest
