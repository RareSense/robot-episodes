"""Measured interactability: does every declared mechanism actually move?

The build settles the environment alone (no robot, actuation irrelevant), records
every movable part that starts touching or inside static geometry, then pushes each
hinge and slider in its allowed direction with a force sized to its mass and reports
whether it opened to its stop, opened partly, or jammed and on what. Free props that
fall away from where they were placed are reported as unsupported; props that slide
or roll into a seat during settling are reported as settled.

One engine decides. MuJoCo is the acceptance engine: its measurements are the
findings the agent repairs and the status the build carries. PyBullet runs the same
push as a compatibility opinion: where it disagrees, the disagreement is recorded
beside the finding and never becomes a failure or a repair task on its own. Water
regions apply their buoyancy and drag in both engines, so a prop meant to float is
judged floating.
"""

import importlib.util
from pathlib import Path

import numpy as np

from .configuration import ACCEPTANCE_ENGINE, COMPATIBILITY_ENGINE
from .nova_water import MuJoCoWater, PyBulletWater

CLEARANCE_TOLERANCE = 0.0005  # closer than this at rest counts as touching
PENETRATION_TOLERANCE = 0.003
DRIFT_TOLERANCE = 0.3
FALL_TOLERANCE = 0.3  # dropping this far means nothing was holding it up
STILL_SPEED = 0.05  # a prop still moving this fast at the end has not come to rest
FLOAT_SHARE = 0.8  # averaged lift at least this share of the weight: the water holds it
SETTLE_STEPS = 500
PUSH_STEPS = 1500
RELEASE_STEPS = 250
ENGINE_LABELS = {"mujoco": "MuJoCo", "pybullet": "PyBullet"}


def push_force(kind, mass):
    """Enough to beat gravity on a lid hinged at its back edge, or a stiff drawer.
    A torque for hinges (N·m), a force for sliders (N)."""
    if kind == "hinge":
        return min(60.0, 6.0 + 2.5 * mass)
    return min(120.0, 30.0 + 3.0 * mass)


def _outcome(kind, rest, pushed, lower, upper, sign):
    limit = upper if sign > 0 else lower
    travel = abs(pushed - rest)
    span = abs(limit)
    threshold = 0.05 if kind == "hinge" else 0.02
    if travel < min(threshold, 0.5 * span):
        return "jammed"
    if abs(pushed - limit) < 0.1 * span + 0.02:
        return "opens_to_stop"
    return "opens_partly"


def _support(drift, fall, speed, floating=False):
    """What held a free prop up. Falling or never coming to rest is a defect; sliding
    or rolling into a seat is not — a boulder on rough ground settles where it settles.
    A settled prop is still reported: the source did not put it where it ends up.
    A prop that is receiving lift when the settle window ends is held by the water;
    it may still be bobbing, which the drag approximation damps only slowly."""
    if floating:
        return "floats"
    if fall > FALL_TOLERANCE or speed > STILL_SPEED:
        return "unsupported"
    return "settled" if drift > DRIFT_TOLERANCE else "rests"


def _mujoco_pass(scene, model_path):
    import mujoco

    from .mujoco_environment import disable_asset_cache

    disable_asset_cache()
    name_by_id = {f"object_{i:04d}": obj.name for i, obj in enumerate(scene.objects)}
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    water = MuJoCoWater([v.specification() for v in scene.water_volumes], scene.gravity_m_s2)
    hinge, slide = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
    mechanisms = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        kind = int(model.jnt_type[j])
        if not name.startswith("object_") or kind not in (hinge, slide):
            continue
        index = int(name[7:11])
        obj = scene.objects[index]
        if obj.body is not None and obj.body.drive is not None:
            # A prescribed drive owns this joint; pushing against its actuator
            # proves nothing about passive mechanics. Reported separately.
            continue
        lower, upper = (float(v) for v in model.jnt_range[j])
        sign = 1.0 if upper > 1e-6 else -1.0
        mechanisms.append(
            {
                "joint": j,
                "index": index,
                "name": obj.name,
                "kind": "hinge" if kind == hinge else "slide",
                "range": (lower, upper),
                "sign": sign,
                "push": push_force("hinge" if kind == hinge else "slide", float(obj.body.mass)),
                "body": int(model.jnt_bodyid[j]),
            }
        )
    body_index = {}
    for i, obj in enumerate(scene.objects):
        if obj.body is not None:
            body_index[int(model.body(f"object_{i:04d}_body").id)] = i

    def object_of(geom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        return name_by_id.get(gname[:11], gname) if gname.startswith("object_") else "world"

    def contacts(threshold, static_only=True):
        """{movable body: {static part: worst distance}} for contacts closer than
        `threshold` (negative = penetration)."""
        found = {}
        for c in data.contact[: data.ncon]:
            if c.dist >= threshold:
                continue
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            for mine, other, g_other in ((b1, b2, c.geom2), (b2, b1, c.geom1)):
                if mine in body_index and other != mine and (not static_only or other == 0):
                    entry = found.setdefault(mine, {})
                    against = object_of(g_other)
                    entry[against] = min(entry.get(against, 1.0), float(c.dist))
        return found

    def step():
        data.xfrc_applied[:] = 0.0
        if scene.water_volumes:
            water.apply(model, data)
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    rest_contacts = contacts(CLEARANCE_TOLERANCE)
    prop_penetration = contacts(-PENETRATION_TOLERANCE, static_only=False)
    # A floating prop still bobs when the window ends (the drag model damps heave
    # only slowly); judge its lift averaged over the second half of the settle
    # window, which spans several bobs, not at one instant.
    window = SETTLE_STEPS // 2
    lift = {body: 0.0 for body in body_index}
    for index in range(SETTLE_STEPS):
        step()
        if scene.water_volumes and index >= SETTLE_STEPS - window:
            for body in lift:
                lift[body] += water.lift(model, data, body) / window
    mujoco.mj_forward(model, data)
    # Support is a passive test. A drawer carrying its contents during the later
    # forced push is expected motion, not evidence that those contents floated.
    settled_positions = data.xpos.copy()
    settled_velocities = data.cvel.copy()
    floating = {
        body: lift[body] >= FLOAT_SHARE * float(model.body_subtreemass[body]) * scene.gravity_m_s2
        for body in body_index
    } if scene.water_volumes else {}
    rest = {m["joint"]: float(data.qpos[model.jnt_qposadr[m["joint"]]]) for m in mechanisms}
    for _ in range(PUSH_STEPS):
        for m in mechanisms:
            data.qfrc_applied[model.jnt_dofadr[m["joint"]]] = m["sign"] * m["push"]
        step()
    mujoco.mj_forward(model, data)
    pushed = {m["joint"]: float(data.qpos[model.jnt_qposadr[m["joint"]]]) for m in mechanisms}
    push_contacts = contacts(-PENETRATION_TOLERANCE)
    for m in mechanisms:
        data.qfrc_applied[model.jnt_dofadr[m["joint"]]] = 0.0
    for _ in range(RELEASE_STEPS):
        step()
    results = []
    for m in mechanisms:
        touching = rest_contacts.get(m["body"]) or {}
        blockers = push_contacts.get(m["body"]) or {}
        results.append(
            {
                "name": m["name"],
                "kind": m["kind"],
                "range": list(m["range"]),
                "rest": rest[m["joint"]],
                "pushed": pushed[m["joint"]],
                "push": m["push"],
                "outcome": _outcome(
                    m["kind"], rest[m["joint"]], pushed[m["joint"]], *m["range"], m["sign"]
                ),
                "touching_at_rest": {k: round(v, 4) for k, v in touching.items()},
                "blocked_by": sorted(blockers, key=lambda k: blockers[k])[:4],
            }
        )
    props = []
    for body, index in body_index.items():
        obj = scene.objects[index]
        if obj.body.motion != "free":
            continue
        drift = float(np.linalg.norm(settled_positions[body] - np.asarray(obj.body.origin)))
        fall = float(obj.body.origin[2] - settled_positions[body][2])
        speed = float(np.linalg.norm(settled_velocities[body][3:6]))
        props.append(
            {
                "name": obj.name,
                "drift_m": round(drift, 3),
                "fall_m": round(fall, 3),
                "after_push_displacement_m": round(
                    float(np.linalg.norm(data.xpos[body] - settled_positions[body])), 3
                ),
                "outcome": _support(drift, fall, speed, floating.get(body, False)),
                "touching_at_rest": {
                    k: round(v, 4)
                    for k, v in (prop_penetration.get(body) or {}).items()
                    if v < -PENETRATION_TOLERANCE
                },
            }
        )
    return {
        "engine": f"MuJoCo {mujoco.__version__}",
        "mechanisms": results,
        "props": props,
        "finite": bool(np.isfinite(data.qpos).all()),
        "warnings": int(data.warning.number.sum()),
    }


def _pybullet_pass(scene, package):
    """The same push in PyBullet, loading exactly what the shipped launcher loads."""
    if importlib.util.find_spec("pybullet") is None:
        raise RuntimeError("pybullet is not installed in this runtime")
    launcher_path = Path(__file__).with_name("pybullet_launcher.py")
    spec = importlib.util.spec_from_file_location("nova_pybullet_launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    import pybullet as p

    from .urdf_preview import pybullet_item

    package = Path(package)
    items = [pybullet_item(i, obj) for i, obj in enumerate(scene.objects)]
    for item in items:
        for file in (
            [item["body"]["urdf"]] if item["body"] else []
        ) + ([item["heightfield"]["file"]] if item["heightfield"] else []):
            if not (package / file).is_file():
                raise RuntimeError(f"PyBullet export is incomplete: {file}")
    water = PyBulletWater([v.specification() for v in scene.water_volumes], scene.gravity_m_s2)
    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -scene.gravity_m_s2)
        p.setTimeStep(0.002)
        handles = {}
        for item in items:
            if item["heightfield"]:
                handles[item["name"]] = launcher.load_terrain(p, package, item)
            elif item["body"]:
                handles[item["name"]] = launcher.load_movable(p, package, item)
            else:
                handles[item["name"]] = launcher.load_static(p, package, item)
        joints = []
        for item in items:
            body = item["body"]
            if not body or body["motion"] not in {"hinge", "slide"} or body.get("drive"):
                continue
            handle = handles[item["name"]]
            info = p.getJointInfo(handle, 0)
            kind = "hinge" if info[2] == p.JOINT_REVOLUTE else "slide"
            lower, upper = float(info[8]), float(info[9])
            sign = 1.0 if upper > 1e-6 else -1.0
            joints.append((handle, item["name"], kind, lower, upper, sign,
                           push_force(kind, body["mass"]), np.array(info[13])))

        def step():
            if scene.water_volumes:
                water.apply(p)
            p.stepSimulation()

        free = [item["name"] for item in items if item["body"] and item["body"]["motion"] == "free"]
        window = SETTLE_STEPS // 2
        lift = {name: 0.0 for name in free}
        for index in range(SETTLE_STEPS):
            step()
            if scene.water_volumes and index >= SETTLE_STEPS - window:
                for name in free:
                    lift[name] += water.lift(p, handles[name]) / window
        rest = {h: p.getJointState(h, 0)[0] for h, *_ in joints}
        settled_positions = {
            name: np.asarray(p.getBasePositionAndOrientation(handles[name])[0]) for name in free
        }
        settled_speeds = {
            name: float(np.linalg.norm(np.asarray(p.getBaseVelocity(handles[name])[0])))
            for name in free
        }
        floating = {
            name: lift[name] >= FLOAT_SHARE * float(p.getDynamicsInfo(handles[name], -1)[0])
            * scene.gravity_m_s2
            for name in free
        } if scene.water_volumes else {}
        owner = {h: n for n, h in handles.items()}
        static_handles = {handles[item["name"]] for item in items if not item["body"]}
        prop_penetration = {}
        for name in free:
            found = {}
            for c in p.getContactPoints(bodyA=handles[name]):
                if c[2] != handles[name] and c[8] < -PENETRATION_TOLERANCE:
                    other = owner.get(c[2], "world")
                    found[other] = min(found.get(other, 0.0), round(float(c[8]), 4))
            prop_penetration[name] = found
        touching = {}
        for handle, name, *_ in joints:
            for c in p.getContactPoints(bodyA=handle):
                if c[8] >= CLEARANCE_TOLERANCE or c[2] not in static_handles:
                    continue
                other = owner.get(c[2], "world")
                entry = touching.setdefault(name, {})
                entry[other] = min(entry.get(other, 1.0), float(c[8]))
        for _ in range(PUSH_STEPS):
            for handle, _name, kind, _lower, _upper, sign, push, axis in joints:
                if kind == "hinge":
                    p.applyExternalTorque(handle, 0, (sign * push * axis).tolist(), p.WORLD_FRAME)
                else:
                    p.applyExternalForce(
                        handle, 0, (sign * push * axis).tolist(), [0, 0, 0], p.LINK_FRAME
                    )
            step()
        results = []
        for handle, name, kind, lower, upper, sign, push, _axis in joints:
            q = p.getJointState(handle, 0)[0]
            results.append({
                "name": name, "kind": kind, "range": [lower, upper], "push": push,
                "blocked_by": [], "rest": float(rest[handle]), "pushed": float(q),
                "outcome": _outcome(kind, rest[handle], q, lower, upper, sign),
                "touching_at_rest": {k: round(v, 4) for k, v in touching.get(name, {}).items()},
            })
        props = []
        for item in items:
            body = item["body"]
            if not body or body["motion"] != "free":
                continue
            name = item["name"]
            pos = p.getBasePositionAndOrientation(handles[name])[0]
            settled = settled_positions[name]
            drift = float(np.linalg.norm(settled - np.asarray(body["origin"])))
            fall = float(body["origin"][2] - settled[2])
            props.append({
                "name": name,
                "touching_at_rest": prop_penetration[name],
                "drift_m": round(drift, 3),
                "fall_m": round(fall, 3),
                "after_push_displacement_m": round(
                    float(np.linalg.norm(np.asarray(pos) - settled)), 3
                ),
                "outcome": _support(
                    drift, fall, settled_speeds[name], floating.get(name, False)
                ),
            })
        return {
            "engine": f"PyBullet {p.getAPIVersion()}", "mechanisms": results, "props": props,
            "warnings": 0,
            "finite": all(np.isfinite(r["pushed"]) for r in results)
            and all(np.isfinite(r["drift_m"]) for r in props),
        }
    finally:
        p.disconnect(client)


RANK = {"opens_to_stop": 0, "opens_but_touching": 1, "opens_partly": 2, "jammed": 3}
_PASSES = {"mujoco": _mujoco_pass, "pybullet": _pybullet_pass}


def _measure(engine, scene, model_path, package):
    try:
        measured = _PASSES[engine](scene, model_path if engine == "mujoco" else package)
    except Exception as exc:  # the engine's failure is the finding, never a pass
        return None, {"status": "error", "reason": str(exc)[:400]}
    return measured, {"status": "measured"}


def _finalize(measured):
    for r in measured["mechanisms"]:
        if r["touching_at_rest"] and r["outcome"] == "opens_to_stop":
            r["outcome"] = "opens_but_touching"
    return measured


def run_mechanism_test(scene, model_path, package=None, *, acceptance=ACCEPTANCE_ENGINE,
                       compatibility=(COMPATIBILITY_ENGINE,)) -> dict:
    """Push every mechanism in the acceptance engine and, separately, in each
    compatibility engine. Only the acceptance engine's measurements decide."""
    package = Path(package) if package is not None else Path(model_path).parent
    coverage = {}
    accepted, coverage[acceptance] = _measure(acceptance, scene, model_path, package)
    opinions = {}
    for engine in compatibility:
        measured, coverage[engine] = _measure(engine, scene, model_path, package)
        if measured is not None:
            opinions[engine] = _finalize(measured)
    results, props, warnings, finite = [], [], 0, True
    if accepted is not None:
        accepted = _finalize(accepted)
        results, props = accepted["mechanisms"], accepted["props"]
        warnings, finite = accepted["warnings"], accepted["finite"]
    # A compatibility engine that saw a WORSE outcome for the same part: the creator
    # will run this package in that engine too, so a door that jams or opens only
    # partly there IS the finding — it fails the test, with that engine's evidence.
    # (Three warehouse doors opened in MuJoCo, jammed against their frames in
    # PyBullet, and the summary still said "passed".)
    notes = []
    for engine, measured in opinions.items():
        label = ENGINE_LABELS.get(engine, engine)
        theirs = {r["name"]: r for r in measured["mechanisms"]}
        for r in results:
            other = theirs.get(r["name"])
            r.setdefault("engines", {})[acceptance] = r["outcome"]
            if other is None:
                continue
            r["engines"][engine] = other["outcome"]
            if RANK[other["outcome"]] > RANK[r["outcome"]] and other["outcome"] in {
                "jammed", "opens_partly",
            }:
                for key in ("outcome", "touching_at_rest", "blocked_by", "range", "rest", "pushed"):
                    if key in other:
                        r[key] = other[key]
                r["outcome_engine"] = engine
            elif RANK[other["outcome"]] > RANK[r["outcome"]]:
                notes.append(
                    f"{label}: {r['name']} {other['outcome'].replace('_', ' ')} "
                    f"({ENGINE_LABELS.get(acceptance, acceptance)}: "
                    f"{r['outcome'].replace('_', ' ')})"
                )
        their_props = {pr["name"]: pr for pr in measured["props"]}
        for pr in props:
            other = their_props.get(pr["name"])
            pr.setdefault("engines", {})[acceptance] = pr["outcome"]
            if other is None:
                continue
            pr["engines"][engine] = other["outcome"]
            if other["outcome"] == "unsupported" and pr["outcome"] != "unsupported":
                notes.append(
                    f"{label}: {pr['name']} unsupported "
                    f"({ENGINE_LABELS.get(acceptance, acceptance)}: {pr['outcome']})"
                )
    # Fill in the acceptance engine's own outcome where no second opinion ran. Never
    # overwrite it: after a compatibility engine's jam decides `outcome`, this map
    # still says what each engine measured.
    for r in results:
        r.setdefault("engines", {}).setdefault(acceptance, r["outcome"])
    for pr in props:
        pr.setdefault("engines", {}).setdefault(acceptance, pr["outcome"])
    jammed = sum(r["outcome"] == "jammed" for r in results)
    partial = sum(r["outcome"] == "opens_partly" for r in results)
    touching = sum(r["outcome"] == "opens_but_touching" for r in results)
    unsupported = sum(pr["outcome"] == "unsupported" for pr in props)
    settled = sum(pr["outcome"] == "settled" for pr in props)
    floating = sum(pr["outcome"] == "floats" for pr in props)
    interfering = sorted(
        {pr["name"] for pr in props if pr["touching_at_rest"]}
        | {
            r["name"]
            for r in results
            if any(v < -PENETRATION_TOLERANCE for v in r["touching_at_rest"].values())
        }
    )
    status = (
        "error"
        if accepted is None
        else "failed"
        if jammed or partial or unsupported or interfering or not finite
        else "inconclusive"
        if touching or settled or warnings
        else "passed"
    )
    return {
        "acceptance_engine": acceptance,
        "engines": [accepted["engine"]] if accepted else [],
        "coverage": coverage,
        "scope": "environment only, no robot: rest clearance, forced push in the allowed "
        "direction, free-prop displacement during passive settling before the push, water "
        "buoyancy and drag where declared. Support beyond the settle window is not "
        "certified. Not a task or controller test.",
        "mechanisms": results,
        "props": props,
        "compatibility": {
            engine: {"engine": measured["engine"], "mechanisms": measured["mechanisms"],
                     "props": measured["props"], "finite": measured["finite"]}
            for engine, measured in opinions.items()
        },
        "compatibility_notes": notes,
        "summary": {
            "mechanisms": len(results),
            "opens_to_stop": sum(r["outcome"] == "opens_to_stop" for r in results),
            "opens_but_touching": touching,
            "opens_partly": partial,
            "jammed": jammed,
            "free_props": len(props),
            "unsupported_props": unsupported,
            "settled_props": settled,
            "floating_props": floating,
            "parts_interfering_at_rest": interfering,
        },
        "status": status,
        "finite": finite,
        "warnings": warnings,
    }


def report_lines(result) -> list[str]:
    lines = [f"MECHANISM_COVERAGE {engine}: {item['status']}"
             + (f" ({item['reason']})" if item.get("reason") else "")
             for engine, item in result.get("coverage", {}).items()]
    default_label = ENGINE_LABELS.get(result.get("acceptance_engine", ""), "MuJoCo")
    for r in result["mechanisms"]:
        # The engine whose evidence decided this outcome (a compatibility engine wins
        # when it saw the part jam or stop short).
        acceptance = ENGINE_LABELS.get(r.get("outcome_engine", ""), default_label)
        if r["outcome"] == "jammed":
            detail = (
                ", ".join(
                    f"{k} ({-v * 1000:.0f} mm inside at rest)"
                    if v < 0
                    else f"{k} (touching at rest)"
                    for k, v in r["touching_at_rest"].items()
                )
                or ", ".join(r["blocked_by"])
                or "nothing measurable; check limits, axis sign and joint resistance"
            )
            lines.append(f"MECHANISM_TEST {r['name']}: JAMMED in {acceptance}, blocked by {detail}")
        elif r["outcome"] == "opens_partly":
            span = max(abs(v) for v in r["range"])
            travel = abs(r["pushed"] - r["rest"])
            lines.append(
                f"MECHANISM_TEST {r['name']}: opens partly in {acceptance} ({travel:.2f} "
                f"of {span:.2f}); check the swept volume and the clearance to "
                + (", ".join(r["blocked_by"]) or "its frame or carcass")
            )
        elif r["outcome"] == "opens_but_touching":
            worst = min(r["touching_at_rest"].items(), key=lambda kv: kv[1])
            lines.append(
                f"MECHANISM_TEST {r['name']}: opens to its stop in {acceptance}, but touches "
                f"static part {worst[0]} ({max(0.0, -worst[1]) * 1000:.1f} mm inside at rest). "
                "Check whether this is an intended support/stop or unwanted rubbing."
            )
    for pr in result["props"]:
        if pr["outcome"] == "unsupported":
            lines.append(
                f"MECHANISM_TEST {pr['name']}: prop fell {pr.get('fall_m', 0):.2f} m and moved "
                f"{pr['drift_m']:.2f} m during passive settling before any push; it was not "
                "resting on a support"
            )
        elif pr["outcome"] == "settled":
            lines.append(
                f"MECHANISM_TEST {pr['name']}: prop settled {pr['drift_m']:.2f} m from where it "
                "was placed during passive settling; it is supported, but it does not stay "
                "where the source put it"
            )
        elif pr["outcome"] == "floats":
            lines.append(
                f"MECHANISM_TEST {pr['name']}: prop floats in water "
                f"({-pr.get('fall_m', 0):+.2f} m from where it was placed); buoyancy is the "
                "packaged approximation, not a fluid solver"
            )
        elif pr["touching_at_rest"]:
            worst = min(pr["touching_at_rest"].items(), key=lambda kv: kv[1])
            lines.append(
                f"MECHANISM_TEST {pr['name']}: prop starts {-worst[1] * 1000:.0f} mm inside "
                f"{worst[0]}; place its bottom face on the support"
            )
    for note in result.get("compatibility_notes", []):
        lines.append(f"COMPATIBILITY {note}; a second-opinion engine, not a repair task")
    s = result["summary"]
    lines.append(
        f"MECHANISM_TEST_SUMMARY engine={result.get('acceptance_engine', 'mujoco')} "
        f"opens_to_stop={s['opens_to_stop']} opens_but_touching={s['opens_but_touching']} "
        f"opens_partly={s['opens_partly']} jammed={s['jammed']} "
        f"unsupported_props={s['unsupported_props']}/{s['free_props']} "
        f"settled_props={s['settled_props']} floating_props={s['floating_props']} "
        f"interfering_parts={len(s['parts_interfering_at_rest'])} status={result['status']}"
    )
    return lines
