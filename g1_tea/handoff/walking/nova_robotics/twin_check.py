"""Digital-twin inventory check: every photographed copy is built, every door works.

`twin_inventory.json` (written by Nova's photo analysis at the workspace root)
lists each inventory row with its exact count. The builder names every part of
copy i of row n `r<n>i<i>_<part>` (e.g. `r32i3_top`), so the build can count the
copies it actually contains instead of trusting a summary:

    TWIN_COUNT r32 "desk table" expected=6 built=4 MISSING 2
    TWIN_MECHANISM r10 "sliding glass door" copy 1: static — no hinge or slide body
    TWIN_COUNTS status=failed rows=61 complete=58 missing_rows=2 untagged_rows=1

A row the photographs cannot support may be excused only by a scene note
`MANIFEST_GAP r<n>: <reason>`; it is reported as a gap, never as complete.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

INVENTORY = "twin_inventory.json"
TAG = re.compile(r"^r(\d{1,4})i(\d{1,4})(?:_|$)")
GAP = re.compile(r"^MANIFEST_GAP\s+r(\d{1,4})\s*:\s*(.+)$")
JOINTED = {"hinged", "sliding", "pocket", "folding", "drawer", "lid", "gate"}
WORKING = {"opens_to_stop", "opens_but_touching"}


def load(root: Path = Path(".")) -> dict | None:
    path = root / INVENTORY
    if not path.is_file() or path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and isinstance(value.get("rows"), list) else None


def check(scene, mechanism_result: dict | None, inventory: dict | None) -> dict:
    if inventory is None:
        return {"status": "not_applicable", "rows": [], "lines": []}
    built: dict[int, set[int]] = {}
    jointed: dict[tuple[int, int], list[str]] = {}
    for obj in scene.objects:
        match = TAG.match(obj.name)
        if not match:
            continue
        n, i = int(match.group(1)), int(match.group(2))
        built.setdefault(n, set()).add(i)
        if obj.body is not None and obj.body.motion in {"hinge", "slide"}:
            jointed.setdefault((n, i), []).append(obj.name)
    gaps = {}
    for note in scene.notes:
        match = GAP.match(note.strip())
        if match:
            gaps[int(match.group(1))] = match.group(2).strip()
    outcomes = {r["name"]: r["outcome"] for r in (mechanism_result or {}).get("mechanisms", [])}

    rows, lines = [], []
    missing_rows = untagged_rows = mechanism_failures = 0
    for row in inventory["rows"]:
        try:
            n, expected = int(row["n"]), int(row.get("count", 1))
        except (KeyError, TypeError, ValueError):
            continue
        name = str(row.get("name", ""))[:60]
        copies = built.get(n, set())
        entry = {"n": n, "name": name, "expected": expected, "built": len(copies)}
        if n in gaps:
            entry["status"] = "gap"
            entry["gap"] = gaps[n]
            lines.append(f'TWIN_COUNT r{n} "{name}" expected={expected} built={len(copies)} '
                         f"GAP {gaps[n][:160]}")
        elif not copies:
            entry["status"] = "untagged"
            untagged_rows += 1
            lines.append(f'TWIN_COUNT r{n} "{name}" expected={expected} built=0 MISSING '
                         f"{expected} (no part named r{n}i<copy>_...)")
        elif len(copies) < expected:
            entry["status"] = "missing"
            missing_rows += 1
            lines.append(f'TWIN_COUNT r{n} "{name}" expected={expected} built={len(copies)} '
                         f"MISSING {expected - len(copies)}")
        else:
            entry["status"] = "ok"
            extra = len(copies) - expected
            lines.append(f'TWIN_COUNT r{n} "{name}" expected={expected} built={len(copies)} ok'
                         + (f" (+{extra} beyond the photo count: justify or remove)"
                            if extra else ""))
        detail = row.get("mechanism_detail")
        kind = str(detail.get("type", "")).lower() if isinstance(detail, dict) else ""
        if kind in JOINTED and n not in gaps:
            for i in sorted(copies) or [1]:
                parts = jointed.get((n, i), [])
                if not parts:
                    mechanism_failures += 1
                    lines.append(f'TWIN_MECHANISM r{n} "{name}" copy {i}: static — photographed '
                                 f"as {kind} ({_describe(detail)}) but built with no hinge or "
                                 "slide body")
                    continue
                results = [outcomes.get(part) for part in parts]
                if not any(result in WORKING for result in results):
                    mechanism_failures += 1
                    lines.append(f'TWIN_MECHANISM r{n} "{name}" copy {i}: '
                                 f"{', '.join(str(r) for r in results)} — it must open through "
                                 "its full travel in MECHANISM_TEST")
        rows.append(entry)
    failed = missing_rows or untagged_rows or mechanism_failures
    status = "failed" if failed else "passed"
    lines.append(
        f"TWIN_COUNTS status={status} rows={len(rows)} "
        f"complete={sum(r['status'] == 'ok' for r in rows)} missing_rows={missing_rows} "
        f"untagged_rows={untagged_rows} gaps={sum(r['status'] == 'gap' for r in rows)} "
        f"mechanism_failures={mechanism_failures}"
    )
    return {"status": status, "rows": rows, "lines": lines}


def _describe(detail: dict) -> str:
    parts = [str(detail.get(key)) for key in ("wall_face", "opens", "hinge_or_track")
             if detail.get(key)]
    return "; ".join(parts)[:160]
