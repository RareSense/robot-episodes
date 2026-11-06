"""The digital-twin photo analysis: its inventory contract, counts and crops.

The jewelry workflow's lesson, applied to rooms: exactness comes from a dedicated
analysis pass whose structured output the builder receives as ground truth, not from
hoping the builder studies the photographs hard enough. One tool-free provider call
inventories EVERY visible object — counts (including duplicates), dimensions with their
anchors, colours, materials, placement and mobility class. The pack's evidence step
(twin_evidence.py) asks Nova's host to make that call and hands the reply to these
functions, inside the build cage: the result lands in the workspace as
`refs/twin_manifest.json`, which the pack doctrine binds the builder to, beside
`twin_inventory.json` — the per-row counts the build checks every copy against — and
`refs/crops/`, each object cut from the photo that shows it best.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_FENCE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)```", re.S)


def strip_fences(text):
    """Return the artifact inside a markdown fence when the model added one.

    Providers regularly wrap a JSON object or a Python module in ```json / ```python
    fences, sometimes with a sentence before or after. The longest fenced block is the
    artifact; text without any fence is returned unchanged.
    """
    text = text.strip()
    blocks = _FENCE.findall(text)
    if not blocks:
        return text
    return max(blocks, key=len).strip()


MANIFEST_NAME = "refs/twin_manifest.json"
INVENTORY_NAME = "twin_inventory.json"
CROPS_DIR = "refs/crops"
MAX_CROPS = 300
CROPS_PER_ROW = 3
MAX_OUTPUT_TOKENS = 64_000
MAX_MANIFEST_BYTES = 2_000_000
MAX_REQUEST_IMAGES = 100  # the provider's per-request image limit

REFERENCE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _references(workspace_root: Path) -> tuple[Path, ...]:
    root = workspace_root.resolve()
    refs = workspace_root / "refs"
    return tuple(
        p
        for p in sorted(refs.glob("*"))
        if p.is_file()
        and not p.is_symlink()
        and p.resolve().is_relative_to(root)
        and p.suffix.lower() in REFERENCE_SUFFIXES
        and not p.name.startswith(("generated_", "scan_plan_"))
    )


def parse_manifest(text: str) -> dict[str, Any]:
    raw = strip_fences(text)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The analysis reply contains no JSON object")
    value = json.loads(raw[start : end + 1])
    objects = value.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("The analysis lists no objects")
    total = 0
    for row in objects:
        if not isinstance(row, dict) or not str(row.get("name", "")).strip():
            raise ValueError("Every inventory row needs a name")
        count = row.get("count", 1)
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"{row.get('name')}: count must be a positive integer")
        instances = row.get("instances")
        if isinstance(instances, list) and len(instances) > count:
            # a located copy is proof of a copy: the count can only go up
            row["count"] = count = len(instances)
        total += count
    value["total_object_instances"] = total
    return value


def inventory(manifest: dict[str, Any]) -> dict[str, Any]:
    """What the build checks: one entry per inventory row, every copy counted."""

    rows = []
    for index, row in enumerate(manifest["objects"], start=1):
        try:
            n = int(row.get("n", index))
        except (TypeError, ValueError):
            n = index
        rows.append({
            "n": n,
            "name": str(row["name"])[:120],
            "count": int(row.get("count", 1)),
            "class": str(row.get("class") or "fixed"),
            "mechanism_detail": row.get("mechanism_detail") or None,
        })
    return {
        "source": MANIFEST_NAME,
        "naming": "name every part of copy i of row n 'r<n>i<i>_<part>', e.g. r32i3_top",
        "rows": rows,
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:32] or "object"


def write_crops(workspace_root: Path, manifest: dict[str, Any],
                references: tuple[Path, ...]) -> list[str]:
    """Each object cut from the photos that show it largest, for modelling by eye."""

    from PIL import Image

    folder = workspace_root / CROPS_DIR
    if folder.is_symlink():
        raise ValueError("Unsafe crops directory")
    folder.mkdir(parents=True, exist_ok=True)
    opened: dict[int, Image.Image] = {}
    written: list[str] = []
    try:
        for index, row in enumerate(manifest["objects"], start=1):
            boxes = []
            for instance in row.get("instances") or []:
                if not isinstance(instance, dict):
                    continue
                for box in instance.get("boxes") or []:
                    try:
                        photo = int(box["photo"])
                        x0, y0, x1, y1 = (float(v) for v in box["bbox"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if 1 <= photo <= len(references) and 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1:
                        boxes.append(((x1 - x0) * (y1 - y0), photo, (x0, y0, x1, y1)))
            used_photos: set[int] = set()
            for _area, photo, (x0, y0, x1, y1) in sorted(boxes, reverse=True):
                if photo in used_photos or len(used_photos) >= CROPS_PER_ROW:
                    continue
                if len(written) >= MAX_CROPS:
                    return written
                if photo not in opened:
                    opened[photo] = Image.open(references[photo - 1]).convert("RGB")
                image = opened[photo]
                w, h = image.size
                pad_x, pad_y = 0.15 * (x1 - x0) * w, 0.15 * (y1 - y0) * h
                box = (max(0, int(x0 * w - pad_x)), max(0, int(y0 * h - pad_y)),
                       min(w, int(x1 * w + pad_x)), min(h, int(y1 * h + pad_y)))
                if box[2] - box[0] < 24 or box[3] - box[1] < 24:
                    continue
                crop = image.crop(box)
                crop.thumbnail((512, 512))
                n = row.get("n", index)
                name = f"r{n}_{_slug(str(row['name']))}_photo{photo}.jpg"
                crop.save(folder / name, quality=88)
                written.append(f"{CROPS_DIR}/{name}")
                used_photos.add(photo)
    finally:
        for image in opened.values():
            image.close()
    return written
