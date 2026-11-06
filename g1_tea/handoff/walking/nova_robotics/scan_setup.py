"""Scan setup: the one question about a creator's 3D scan, and its answer.

The pack's evidence step (twin_evidence.py) measures every scan first, then asks the
creator ONE question that bundles what the measurement could not settle — the scanning
app, units, one real measurement for calibration — and what to populate the scene with
(their photos, or a description). On the answer it measures again with the stated app,
units and known measurement and writes the results for the builder:

    refs/scan_measurements.json   real dimensions, per scan
    refs/scan_plan_<n>.png        measured top-down plan with dimensions
    refs/scan_answers.json        the creator's answers, verbatim and parsed

The doctrine binds the builder to these numbers. A failed measurement never fails the
turn: the builder falls back to photo-anchored estimates and says so.
"""

from __future__ import annotations

import re
from typing import Any

MEASUREMENTS = "refs/scan_measurements.json"
ANSWERS = "refs/scan_answers.json"

APP_OPTIONS = [
    ("opt_roomplan", "3D Scanner App / Polycam — Room mode",
     "A LiDAR Room-mode scan (exact walls, doors and furniture boxes)."),
    ("opt_lidar", "LiDAR mesh scan (Scaniverse, Polycam, 3D Scanner App)",
     "A LiDAR scan exported as a mesh. Already in real metres."),
    ("opt_photo", "Photo scan (photogrammetry)",
     "Built from photos only — its scale is unknown until you give one measurement."),
    ("opt_other", "Other / not sure", "Nova will infer what it can and tell you."),
]
UNIT_SCALE = {"mm": 0.001, "cm": 0.01, "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
              "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0}
MEASURE_WORDS = {
    "room width": "room width", "width": "room width", "room length": "room length",
    "length": "room length", "room depth": "room depth", "depth": "room depth",
    "ceiling height": "ceiling height", "room height": "room height",
    "ceiling": "ceiling height", "height": "ceiling height",
}


def parse_answer(text: str, option_id: str | None) -> dict[str, Any]:
    """Pull app, units and one calibration measurement out of a free answer."""
    lowered = (text or "").lower()
    app = dict((o[0], o[1]) for o in APP_OPTIONS).get(option_id or "")
    for word, name in (("3d scanner", "3D Scanner App"), ("polycam", "Polycam"),
                       ("scaniverse", "Scaniverse"), ("canvas", "Canvas"),
                       ("matterport", "Matterport"), ("room mode", "Room mode"),
                       ("roomplan", "Room mode")):
        if word in lowered:
            app = (app + "; " if app else "") + name
    units = None
    match = re.search(r"\b(?:units?|in)\s*(?:are|is|:)?\s*(mm|cm|ft|feet|foot|meters?|metres?|m)\b",
                      lowered)
    if match:
        units = match.group(1)
    known = None
    pattern = (r"(room width|room length|room depth|ceiling height|room height|width|length|"
               r"depth|ceiling|height)\s*(?:is|=|:|of|about|~)?\s*(\d+(?:\.\d+)?)\s*"
               r"(mm|cm|ft|feet|foot|meters?|metres?|m)?\b")
    found = re.search(pattern, lowered)
    if found:
        value = float(found.group(2)) * UNIT_SCALE.get(found.group(3) or "m", 1.0)
        known = {MEASURE_WORDS[found.group(1)]: round(value, 4)}
    populate = None
    pop = re.search(r"(?:populate|fill|furnish|put|add)\b(.{0,400})", text or "", re.IGNORECASE)
    if pop:
        populate = pop.group(0).strip()
    return {"app": app, "units": units, "known": known, "populate": populate,
            "verbatim": text or ""}


def question(reports: list[dict[str, Any]], has_photos: bool) -> dict[str, Any]:
    lines = []
    for report in reports:
        if report.get("error"):
            lines.append(f"• {report['upload_name']}: could not be measured ({report['error']}).")
            continue
        room = report["room"]
        height = room.get("height_m")
        lines.append(
            f"• {report['upload_name']}: measured {room['footprint_m'][0]} × "
            f"{room['footprint_m'][1]} m"
            + (f", ceiling {height} m" if height else ", ceiling not captured")
            + f" ({len(report.get('objects', []))} furniture pieces found)."
        )
        lines.extend(f"  – {q}" for q in report.get("questions", []))
    populate = ("Should the scene be filled from your photos, or describe what to place?"
                if has_photos else
                "What should the scene contain? Describe it, or upload photos of the place.")
    text = (
        "I measured your scan:\n" + "\n".join(lines) + "\n\n"
        "To make the twin exact, tell me in one message (pick the app below, or type it all):\n"
        "1. Which app made the scan, and was it Room mode / LiDAR or a photo scan?\n"
        "2. If the units are not metres, which (cm, mm, ft)?\n"
        "3. One real measurement to calibrate, e.g. 'room width 4.2 m' or 'ceiling 2.7 m'.\n"
        f"4. {populate}"
    )
    return {
        "question": text[:2000],
        "options": [{"id": i, "label": label, "description": desc}
                    for i, label, desc in APP_OPTIONS],
        "allow_free_text": True,
    }


def announcement(reports: list[dict[str, Any]], *, calibrated: bool) -> str:
    good = [r for r in reports if not r.get("error")]
    if not good:
        message = "The scan could not be measured; the builder will estimate from photos."
    else:
        room = good[0]["room"]
        message = (f"Scan measured{' and calibrated' if calibrated else ''}: "
                   f"{room['footprint_m'][0]} × {room['footprint_m'][1]} m"
                   + (f", ceiling {room['height_m']} m." if room.get("height_m") else "."))
    return message
