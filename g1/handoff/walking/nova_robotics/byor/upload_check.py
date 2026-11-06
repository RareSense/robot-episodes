"""Upload-time robot inspection: import + analysis + the fast physical checks.

Runs inside the inspect_input job, which must answer within the runtime command timeout, so it
does only what takes seconds: import, analysis, description validity and the passive drop test.
The full acceptance suite (hold, joint sweep, shoves, reach) is `run_acceptance`, a longer job.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def inspect_robot(path: Path, root_dir: Path, cache_dir: Path) -> dict:
    from .acceptance import TestResult, check_drop_and_settle
    from .analysis import analyze_model
    from .loader import load_robot

    t0 = time.perf_counter()
    try:
        loaded = load_robot(path, root_dir=root_dir, cache_dir=cache_dir)
    except Exception as exc:                       # noqa: BLE001 - reported to the user
        return {"status": "rejected", "reason": str(exc)[:2000]}
    report = analyze_model(loaded.model, source=path.name)
    errors = [i["detail"] for i in report.issues if i["severity"] == "error"] + list(loaded.problems)
    tests = [TestResult("description_valid", not errors,
                        "no errors in the analysis" if not errors else "; ".join(errors[:6]), {})]
    if not errors:
        tests.append(check_drop_and_settle(loaded.spec, report))
    level = 1 if all(t.passed for t in tests) else 0
    out = {
        "status": "analysed",
        "source_format": loaded.source_format,
        "base": loaded.base,
        "added_by_nova": loaded.added,
        "analysis": report.as_dict(),
        "checks": [{"name": t.name, "passed": t.passed, "detail": t.detail, "metrics": t.metrics} for t in tests],
        "trust_level": level,
        "trust_note": ("physically valid; hold, joint sweep, balance and reach run in the full acceptance job"
                       if level else "not usable yet: fix the listed problems in the robot description"),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    safe = _json_safe(out)
    if len(json.dumps(safe)) > 200_000:            # keep the inspect response small
        safe["analysis"]["motors"] = safe["analysis"]["motors"][:64]
        safe["analysis"]["issues"] = safe["analysis"]["issues"][:64]
    return safe
