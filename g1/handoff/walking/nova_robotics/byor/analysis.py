"""Generic robot analysis: structure, class, feet, wheels, hands, motors and physical validity.

Nothing is robot-specific. The class comes from the kinematic tree and the geometry of the
robot's own rest pose, never from names, so a robot nobody has seen before is handled the same
way as a known one.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

HINGE, SLIDE, FREE = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE),
                      int(mujoco.mjtJoint.mjJNT_FREE))
BODY = mujoco.mjtObj.mjOBJ_BODY


@dataclass
class Report:
    source: str
    total_mass_kg: float = 0.0
    n_bodies: int = 0
    n_joints: int = 0
    n_dof: int = 0
    n_actuators: int = 0
    base: str = "?"                 # floating | fixed
    robot_class: str = "?"          # fixed_arm | humanoid | quadruped | multi_legged | wheeled | aerial | floating_other
    feet: list[str] = field(default_factory=list)
    support: list[str] = field(default_factory=list)      # every body the robot rests on (feet, wheels, casters, skids)
    wheels: list[str] = field(default_factory=list)
    end_effectors: list[str] = field(default_factory=list)
    end_effector_reach_m: dict = field(default_factory=dict)
    end_effector_point: dict = field(default_factory=dict)  # tool point in the hand body's frame
    motors: list[dict] = field(default_factory=list)
    unactuated_joints: list[str] = field(default_factory=list)
    keyframes: list[str] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)

    def issue(self, severity: str, code: str, detail: str):
        self.issues.append({"severity": severity, "code": code, "detail": detail})

    def as_dict(self):
        return asdict(self)


# ------------------------------------------------------------------------------------- tree helpers

def name(m, objtype, i):
    return mujoco.mj_id2name(m, objtype, i) or f"#{i}"


def subtree(m, root):
    out = []
    for b in range(m.nbody):
        x = b
        while x and x != root:
            x = int(m.body_parentid[x])
        if x == root:
            out.append(b)
    return out


def robot_roots(m):
    """World children that carry actuated joints or thrusters (several for e.g. bimanual rigs)."""
    act = set()
    for i in range(m.nu):
        trn, t = int(m.actuator_trntype[i]), int(m.actuator_trnid[i, 0])
        if trn == int(mujoco.mjtTrn.mjTRN_JOINT):
            act.add(int(m.jnt_bodyid[t]))
        elif trn == int(mujoco.mjtTrn.mjTRN_SITE):
            act.add(int(m.site_bodyid[t]))
        elif trn == int(mujoco.mjtTrn.mjTRN_TENDON):
            adr, num = m.tendon_adr[t], m.tendon_num[t]
            for k in range(num):
                if m.wrap_type[adr + k] == mujoco.mjtWrap.mjWRAP_JOINT:
                    act.add(int(m.jnt_bodyid[m.wrap_objid[adr + k]]))
    tops = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0]
    roots = [t for t in tops if act & set(subtree(m, t))]
    if roots:
        return roots
    return [max(tops, key=lambda b: m.body_subtreemass[b])] if tops else [1]


def robot_bodies(m):
    return sorted({b for r in robot_roots(m) for b in subtree(m, r)})


def collision_geoms(m, body):
    """Geoms that can touch anything: by contype/conaffinity flags, or named in an explicit pair."""
    paired = set(m.pair_geom1.tolist()) | set(m.pair_geom2.tolist()) if m.npair else set()
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and
            (m.geom_contype[g] or m.geom_conaffinity[g] or g in paired)]


def path_to(m, b, top):
    out = [b]
    while out[-1] != top and out[-1] != 0:
        out.append(int(m.body_parentid[out[-1]]))
    return out


def lca(m, bodies, root):
    paths = [path_to(m, b, root)[::-1] for b in bodies]
    common = root
    for level in zip(*paths):
        if all(x == level[0] for x in level):
            common = level[0]
        else:
            break
    return common


def joints_between(m, lo, hi):
    """Hinge/slide joints on the path from body lo up to (excluding) ancestor hi."""
    n, x = 0, lo
    while x != hi and x != 0:
        n += sum(1 for j in range(m.body_jntadr[x], m.body_jntadr[x] + m.body_jntnum[x])
                 if int(m.jnt_type[j]) in (HINGE, SLIDE))
        x = int(m.body_parentid[x])
    return n


def geom_min_z(m, d, g):
    """Lowest world z of a geom."""
    t = int(m.geom_type[g])
    if t == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return float(((d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g])[:, 2].min())
    if t == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return float("inf")
    R = d.geom_xmat[g].reshape(3, 3)
    size = m.geom_size[g]
    z = d.geom_xpos[g][2]
    if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(z - size[0])
    if t == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(z - np.abs(R[2, :]) @ size)
    if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return float(z - abs(R[2, 2]) * size[1] - size[0])
    if t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return float(z - abs(R[2, 2]) * size[1] - math.sqrt(max(0.0, 1 - R[2, 2] ** 2)) * size[0])
    if t == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return float(z - math.sqrt(((R[2, :] * size) ** 2).sum()))
    return float(z - m.geom_rbound[g])


def rest_pose(m):
    from .stance import start_pose

    d = mujoco.MjData(m)
    start_pose(m, d)
    return d


# ------------------------------------------------------------------------------------- checks

def motor_table(m, rep: Report):
    for i in range(m.nu):
        trn, j = int(m.actuator_trntype[i]), int(m.actuator_trnid[i, 0])
        aname = name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        gear = float(m.actuator_gear[i, 0])
        b1, b2 = m.actuator_biasprm[i, 1], m.actuator_biasprm[i, 2]
        gt, bt = int(m.actuator_gaintype[i]), int(m.actuator_biastype[i])
        affine = bt == int(mujoco.mjtBias.mjBIAS_AFFINE)
        if affine and b1 < 0:
            kind, kp, kd = "position", -b1 * gear, -b2 * gear
        elif affine and b1 == 0 and b2 < 0:
            kind, kp, kd = "velocity", 0.0, -b2 * gear
        elif bt == int(mujoco.mjtBias.mjBIAS_NONE) and gt == int(mujoco.mjtGain.mjGAIN_FIXED):
            kind, kp, kd = "torque", 0.0, 0.0
        else:
            kind, kp, kd = "other", 0.0, 0.0
        target = {int(mujoco.mjtTrn.mjTRN_JOINT): ("joint", mujoco.mjtObj.mjOBJ_JOINT),
                  int(mujoco.mjtTrn.mjTRN_TENDON): ("tendon", mujoco.mjtObj.mjOBJ_TENDON),
                  int(mujoco.mjtTrn.mjTRN_SITE): ("site", mujoco.mjtObj.mjOBJ_SITE)}.get(trn, ("other", None))
        lim, src = None, "none"
        if trn == int(mujoco.mjtTrn.mjTRN_JOINT) and m.jnt_actfrclimited[j]:
            lim, src = float(m.jnt_actfrcrange[j, 1]), "joint actuatorfrcrange"
        elif m.actuator_forcelimited[i]:
            lim, src = float(abs(m.actuator_forcerange[i]).max() * abs(gear)), "actuator forcerange"
        elif kind == "torque" and m.actuator_ctrllimited[i]:
            lim, src = float(abs(m.actuator_ctrlrange[i]).max() * abs(gear * m.actuator_gainprm[i, 0])), "actuator ctrlrange"
        elif kind == "position" and m.actuator_ctrllimited[i]:
            span = float(m.actuator_ctrlrange[i, 1] - m.actuator_ctrlrange[i, 0])
            lim, src = float(kp * span), "estimated (servo gain x setpoint span)"
        rep.motors.append(dict(name=aname, drives=target[0],
                               target=name(m, target[1], j) if target[1] is not None else "",
                               kind=kind, kp=round(float(kp), 3), kd=round(float(kd), 3),
                               force_limit=lim, force_limit_source=src))


def rigid_mass(m, b):
    """Mass of a body plus everything welded to it (children without joints, recursively)."""
    total = float(m.body_mass[b])
    for c in range(m.nbody):
        if c != b and m.body_parentid[c] == b and m.body_jntnum[c] == 0:
            total += rigid_mass(m, c)
    return total


def check_inertials(m, rep: Report, bodies):
    for b in bodies:
        bn = name(m, BODY, b)
        mass, I = float(m.body_mass[b]), m.body_inertia[b]
        if m.body_dofnum[b] > 0 and rigid_mass(m, b) <= 0:
            rep.issue("error", "zero_mass_moving_body", f"{bn} moves but it and everything welded to it have no mass")
        if mass > 0 and (I <= 0).any():
            # a point mass welded to a moving link (sensor frames) is harmless; a body that moves on
            # its own with no rotational inertia is not physical
            sev = "error" if m.body_jntnum[b] > 0 else "info"
            rep.issue(sev, "non_positive_inertia", f"{bn} inertia {I.round(6).tolist()}")
        if mass > 0 and (I > 0).all():
            a, b2, c = sorted(I)
            if a + b2 < c * (1 - 1e-6):
                rep.issue("error", "inertia_violates_triangle", f"{bn}: {sorted(I)} cannot belong to any physical body")
            ext = max([m.geom_rbound[g] for g in range(m.ngeom) if m.geom_bodyid[g] == b] or [0.0])
            if ext > 0.01:
                rg = math.sqrt(max(I) / mass)
                if rg > 3.0 * ext or rg < 0.02 * ext:
                    rep.issue("warning", "inertia_implausible_for_size",
                              f"{bn}: radius of gyration {rg:.3f} m vs geometry size {ext:.3f} m")
        if 0 < mass < 1e-4 and m.body_dofnum[b] > 0:
            rep.issue("warning", "tiny_mass", f"{bn} mass {mass:.2e} kg makes the solver stiff")


def check_joints_and_motors(m, rep: Report, bodies):
    bodies = set(bodies)
    actuated = set()
    for i in range(m.nu):
        trn, t = int(m.actuator_trntype[i]), int(m.actuator_trnid[i, 0])
        if trn == int(mujoco.mjtTrn.mjTRN_JOINT):
            actuated.add(t)
        elif trn == int(mujoco.mjtTrn.mjTRN_TENDON):
            adr, num = m.tendon_adr[t], m.tendon_num[t]
            actuated |= {int(m.wrap_objid[adr + k]) for k in range(num) if m.wrap_type[adr + k] == mujoco.mjtWrap.mjWRAP_JOINT}
    coupled = {int(m.eq_obj1id[e]) for e in range(m.neq) if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT}
    no_armature = []
    for j in range(m.njnt):
        if m.jnt_bodyid[j] not in bodies or int(m.jnt_type[j]) == FREE:
            continue
        jn = name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if int(m.jnt_type[j]) in (HINGE, SLIDE) and not m.jnt_limited[j]:
            rep.issue("info", "joint_unlimited", f"{jn} has no position limits (fine for wheels / continuous joints)")
        if j not in actuated and j not in coupled:
            rep.unactuated_joints.append(jn)
        if m.dof_armature[m.jnt_dofadr[j]] == 0 and j in actuated:
            no_armature.append(jn)
    if no_armature:
        rep.issue("warning", "no_armature",
                  f"{len(no_armature)} driven joints have no rotor inertia (armature), e.g. {', '.join(no_armature[:3])}: "
                  "geared motors will feel too light; URDF cannot express it, add it from the motor data sheet")
    for mt in rep.motors:
        if mt["kind"] != "other" and mt["force_limit"] is None:
            rep.issue("error", "no_torque_limit", f"{mt['name']}: no torque limit, the motor could apply unlimited force")
        elif mt["force_limit_source"].startswith("estimated"):
            rep.issue("warning", "estimated_torque_limit",
                      f"{mt['name']}: no declared limit; using {mt['force_limit']:.1f} from the servo gain x range")
    kps = [mt["kp"] for mt in rep.motors if mt["kind"] == "position"]
    if len(kps) > 6 and len(set(round(k, 3) for k in kps)) == 1:
        rep.issue("warning", "generic_gains", f"all {len(kps)} position motors share kp={kps[0]}: placeholder gains")


def check_rest_contacts(m, rep: Report, bodies):
    """Self-collisions present in the rest pose (MuJoCo does not filter bodies welded to the world)."""
    d = rest_pose(m)
    bodies = set(bodies)
    seen = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if b1 in bodies and b2 in bodies and c.dist < -1e-3:
            key = tuple(sorted((name(m, BODY, b1), name(m, BODY, b2))))
            if key not in seen:
                seen.add(key)
                rep.issue("warning", "self_collision_at_rest",
                          f"{key[0]} and {key[1]} overlap by {-c.dist * 1000:.1f} mm in the rest pose")


def check_contact_softness(m, rep: Report, bodies):
    soft = []
    for b in bodies:
        for g in collision_geoms(m, b):
            width = float(m.geom_solimp[g][2])
            if width > 0.005:
                soft.append((name(m, BODY, b), width))
    if soft:
        widest = max(soft, key=lambda x: x[1])
        rep.issue("info", "compliant_contacts",
                  f"{len(soft)} contact surfaces are modelled as compliant (e.g. rubber feet); "
                  f"widest compliance band {widest[1] * 1000:.0f} mm on {widest[0]}")


# ------------------------------------------------------------------------------------- classification

def is_wheel(m, d, b):
    """An unlimited hinge on this body whose ground geometry is round about that hinge axis."""
    js = [j for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])
          if int(m.jnt_type[j]) == HINGE and not m.jnt_limited[j]]
    if not js:
        return False
    axis = d.xaxis[js[0]]
    for g in collision_geoms(m, b):
        t = int(m.geom_type[g])
        if t in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)):
            return True
        if t == int(mujoco.mjtGeom.mjGEOM_CYLINDER) and abs(d.geom_xmat[g].reshape(3, 3)[:, 2] @ axis) > 0.9:
            return True
        if t == int(mujoco.mjtGeom.mjGEOM_MESH):
            # round about the axis: vertex distances from the axis vary by < 10 %
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            vw = (d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g] - d.xanchor[js[0]]
            r = np.linalg.norm(vw - np.outer(vw @ axis, axis), axis=1)
            rim = r[r > 0.8 * r.max()]
            if r.max() > 0.02 and rim.std() < 0.1 * rim.mean() and (r > 0.8 * r.max()).mean() > 0.2:
                return True
    return False


def classify(m, rep: Report, roots, bodies):
    root = roots[0]
    floating = any(int(m.jnt_type[j]) == FREE and m.jnt_bodyid[j] in roots for j in range(m.njnt))
    rep.base = "floating" if floating else "fixed"
    sub = bodies
    ch = {b: [] for b in range(m.nbody)}
    for b in range(1, m.nbody):
        ch[int(m.body_parentid[b])].append(b)
    d = rest_pose(m)
    support, resting = [], []
    if floating:
        mins = {}
        for b in sub:
            zs = [geom_min_z(m, d, g) for g in collision_geoms(m, b)]
            if zs:
                mins[b] = min(zs)
        if mins:
            floor = min(mins.values())
            resting = [b for b, z in mins.items() if z < floor + 0.025]
            support = [b for b in resting if b != root]
    rep.support = [name(m, BODY, b) for b in resting]
    wheels = [b for b in support if is_wheel(m, d, b)]
    legs = [b for b in support if b not in wheels]
    feet = []
    if legs:
        top = lca(m, legs, root) if len(legs) > 1 else root
        groups = {}
        for b in legs:
            p = path_to(m, b, top)
            groups.setdefault(p[-2] if len(p) >= 2 else b, []).append(b)
        for bs in groups.values():
            feet.append(max(bs, key=lambda x: joints_between(m, x, top)))
    rep.feet = [name(m, BODY, b) for b in feet]
    rep.wheels = [name(m, BODY, b) for b in wheels]
    thrusters = any(int(m.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_SITE) for i in range(m.nu))
    joints_driven = any(int(m.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT) for i in range(m.nu))
    if not floating:
        rep.robot_class = "fixed_arm"
    elif thrusters and not joints_driven:
        rep.robot_class = "aerial"
    elif wheels:
        rep.robot_class = "wheeled"
    elif len(feet) == 2:
        rep.robot_class = "humanoid"
    elif len(feet) == 4:
        rep.robot_class = "quadruped"
    elif len(feet) > 4:
        rep.robot_class = "multi_legged"
    else:
        rep.robot_class = "floating_other"

    # ---- hands: clustered chain tips away from the support limbs
    support_set = set(support)
    jointed = {b: any(m.body_jntnum[x] > 0 for x in subtree(m, b)) for b in sub}
    jch = {b: [c for c in ch[b] if jointed.get(c)] for b in sub}
    leaves = [b for b in sub if b not in roots and m.body_jntnum[b] > 0 and not jch[b]]
    leaves = [b for b in leaves if not any(s in path_to(m, b, 0) or b in path_to(m, s, 0) for s in support_set)]
    clusters = []
    for b in leaves:
        r = int(m.body_rootid[b])
        for c in clusters:
            if int(m.body_rootid[c[0]]) != r:
                continue
            top = lca(m, [b, c[0]], r)
            if joints_between(m, b, top) <= 4 and joints_between(m, c[0], top) <= 4 and joints_between(m, top, r) >= 3:
                c.append(b)
                break
        else:
            clusters.append([b])
    branch_points = {b for b in sub if len(jch[b]) > 1}
    # joints tied together by couplings (equality constraints), e.g. a gripper drive and its fingers
    couples = {}
    for e in range(m.neq):
        if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT):
            j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            couples.setdefault(j1, set()).add(j2)
            couples.setdefault(j2, set()).add(j1)

    def is_gripper_drive(b):
        """Body whose joint is coupled to joints below it: part of the hand mechanism, not the hand."""
        below = {j for x in subtree(m, b) if x != b
                 for j in range(m.body_jntadr[x], m.body_jntadr[x] + m.body_jntnum[x])}
        return any(couples.get(j, set()) & below for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]))

    cands = []
    for c in clusters:
        r = int(m.body_rootid[c[0]])
        ee = lca(m, c, r) if len(c) > 1 else c[0]
        while ee != r and is_gripper_drive(ee):
            ee = int(m.body_parentid[ee])
        x, n, length = ee, 0, 0.0
        while x != r and x != 0:
            p = int(m.body_parentid[x])
            n += joints_between(m, x, p)
            length += float(np.linalg.norm(m.body_pos[x]))
            if p in branch_points and (any(o is not c and lca(m, [o[0], ee], r) == p for o in clusters)
                                       or any(lca(m, [s, ee], r) == p for s in support_set)):
                length -= float(np.linalg.norm(m.body_pos[x]))     # the mount offset is not reach
                break
            x = p
        if n >= 3:
            cands.append((ee, length))
    longest = max([l for _, l in cands], default=0.0)
    keep = {name(m, BODY, ee): round(l, 3) for ee, l in cands if l >= 0.4 * longest}
    rep.end_effectors = sorted(keep)
    rep.end_effector_reach_m = keep
    rep.end_effector_point = {n: tool_point(m, d, mujoco.mj_name2id(m, BODY, n)) for n in keep}


def tool_point(m, d, b):
    """The hand's working point, in the hand body's frame: the centroid of the collision geometry
    the hand carries (palm + fingers, a gripper's jaws, or a forearm's end when there is no hand)."""
    pts = [d.geom_xpos[g] for x in subtree(m, b) for g in collision_geoms(m, x)]
    if not pts:
        pts = [d.geom_xpos[g] for x in subtree(m, b) for g in range(m.ngeom) if m.geom_bodyid[g] == x]
    if not pts:
        return [0.0, 0.0, 0.0]
    c = np.mean(pts, axis=0)
    return (d.xmat[b].reshape(3, 3).T @ (c - d.xpos[b])).round(4).tolist()


# ------------------------------------------------------------------------------------- entry point

def analyze_model(m: mujoco.MjModel, source: str = "") -> Report:
    roots = robot_roots(m)
    bodies = robot_bodies(m)
    bset = set(bodies)
    rep = Report(source=source,
                 total_mass_kg=round(float(sum(m.body_mass[b] for b in bodies)), 3),
                 n_bodies=len(bodies),
                 n_joints=int(sum(1 for j in range(m.njnt) if m.jnt_bodyid[j] in bset)),
                 n_dof=int(m.nv), n_actuators=int(m.nu),
                 keyframes=[name(m, mujoco.mjtObj.mjOBJ_KEY, k) for k in range(m.nkey)])
    motor_table(m, rep)
    if not any(collision_geoms(m, b) for b in bodies):
        rep.issue("error", "no_collision_geometry", "no robot geom can touch anything: it would fall through the floor")
    check_inertials(m, rep, bodies)
    check_joints_and_motors(m, rep, bodies)
    check_rest_contacts(m, rep, bodies)
    check_contact_softness(m, rep, bodies)
    classify(m, rep, roots, bodies)
    return rep
