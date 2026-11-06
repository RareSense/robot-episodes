"""Generic spawn stance: put a floating robot on the floor with every flat sole flush.

Real robots are set down by an operator (or a stand-up routine) with the soles flat
before the balance controller takes over. In simulation that is a static IK problem,
solved here for any robot from its own collision geometry:

1. For each foot (from the analysis report) take the convex hull of its collision
   geometry in the foot frame and pick the hull face whose outward normal points most
   nearly straight down in the start pose: that is the sole. Feet whose lowest face is
   smaller than a few cm^2 (balls, capsules, point feet) are treated as point contacts.
2. Gauss-Newton on (floating base + leg joints): every sole vertex at floor height,
   point feet touching it, gently regularised to the start pose and within joint limits.

No forces are applied and nothing is dropped. The result is only an initial
configuration, which MuJoCo then simulates normally.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from .analysis import collision_geoms, robot_roots, subtree as _subtree_raw

POINT_FOOT_AREA = 4e-4          # m^2: below this the lowest face is a contact point, not a sole


def _geom_points_local(m, g, n=64):
    """Sample points of a geom's surface in its own frame (exact for boxes and meshes)."""
    t, s = int(m.geom_type[g]), m.geom_size[g]
    G = mujoco.mjtGeom
    if t == G.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        return m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    if t == G.mjGEOM_BOX:
        return np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if t == G.mjGEOM_CYLINDER:
        ring = np.c_[s[0] * np.cos(a), s[0] * np.sin(a)]
        return np.vstack([np.c_[ring, np.full(n, z)] for z in (-s[1], s[1])])
    # spheres, capsules, ellipsoids: a fibonacci sphere, stretched / split into the capsule ends
    k = np.arange(4 * n) + 0.5
    phi, th = np.arccos(1 - 2 * k / (4 * n)), np.pi * (1 + 5 ** 0.5) * k
    u = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
    if t == G.mjGEOM_SPHERE:
        return np.zeros((1, 3))
    if t == G.mjGEOM_ELLIPSOID:
        return u * s[:3]
    if t == G.mjGEOM_CAPSULE:
        return np.linspace([0, 0, -s[1]], [0, 0, s[1]], 5)
    return m.geom_rbound[g] * u


def _geom_radius(m, g):
    """Rounded geoms are represented by their centre (line) plus this radius."""
    t = int(m.geom_type[g])
    return float(m.geom_size[g][0]) if t in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE)) else 0.0


def _foot_geoms(m, foot):
    """A foot's contact geometry: the foot and anything mounted below it (toes). For the robot's
    root body (a base resting on a skid or caster) only its own geometry counts."""
    if foot in robot_roots(m):
        return collision_geoms(m, foot)
    return [g for b in _subtree_raw(m, foot) for g in collision_geoms(m, b)]


def foot_contact_points(m, d, foot, with_radius=False):
    """Return (body_ids, local_points, is_sole[, radii]). Points are in each geom's body frame;
    a rounded geom contributes its centre / axis points, which touch the floor at radius below."""
    out = _foot_contact_points(m, d, foot)
    return out if with_radius else out[:3]


def _foot_contact_points(m, d, foot):
    bodies, pts_body, pts_world, radii = [], [], [], []
    for g in _foot_geoms(m, foot):
        P = _geom_points_local(m, g)
        b = int(m.geom_bodyid[g])
        # geom frame -> body frame
        R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3, 3)
        pb = P @ R.T + m.geom_pos[g]
        r = _geom_radius(m, g)
        pw = pb @ d.xmat[b].reshape(3, 3).T + d.xpos[b] - [0, 0, r]   # lowest point if level
        bodies += [b] * len(pb); pts_body.append(pb); pts_world.append(pw); radii += [r] * len(pb)
    bodies = np.array(bodies); pb = np.vstack(pts_body); pw = np.vstack(pts_world); radii = np.array(radii)
    # hull in world (start pose); the face pointing most nearly down, ties broken by area
    try:
        hull = ConvexHull(pw)
    except Exception:
        # all contact points in one plane (e.g. four sole spheres): that plane is the sole if it
        # faces down and has area
        c = pw.mean(0)
        U, S, Vt = np.linalg.svd(pw - c)
        nrm = Vt[2] if Vt[2][2] < 0 else -Vt[2]
        if len(pw) >= 3 and nrm[2] < -0.85:
            try:
                q2 = (pw - c) @ Vt[:2].T
                h2 = ConvexHull(q2)
                if h2.volume >= POINT_FOOT_AREA:
                    idx = h2.vertices
                    return bodies[idx], pb[idx], True, radii[idx]
            except Exception:
                pass
        i = np.argmin(pw[:, 2])
        return bodies[[i]], pb[[i]], False, radii[[i]]
    best, best_score = None, -1.0
    normals, offs = hull.equations[:, :3], hull.equations[:, 3]
    for f, (nrm, off) in enumerate(zip(normals, offs)):
        if nrm[2] > -0.85:                      # face must point within ~32 deg of straight down
            continue
        on = np.abs(pw @ nrm + off) < 1.5e-3    # every vertex lying in this face's plane
        if on.sum() < 3:
            continue
        try:
            q = pw[on] - pw[on].mean(0)
            B = np.linalg.svd(q)[2][:2]
            area = ConvexHull(q @ B.T).volume
        except Exception:
            area = 0.0
        score = area * (-nrm[2])
        if score > best_score:
            best, best_score = on, score
    if best is not None and best_score >= POINT_FOOT_AREA:
        idx = np.where(best)[0]
        # keep the polygon's extreme vertices only
        q = pw[idx]; B = np.linalg.svd(q - q.mean(0))[2][:2]
        hv = ConvexHull((q - q.mean(0)) @ B.T).vertices
        idx = idx[hv]
        return bodies[idx], pb[idx], True, radii[idx]
    i = np.argmin(pw[:, 2])
    return bodies[[i]], pb[[i]], False, radii[[i]]


def start_pose(m, d, key=0, margin_rad=0.01, margin_m=0.001):
    """Initial joint configuration. Call BEFORE MotorLayer rewrites the actuators.

    - Keyframe qpos when the robot ships one.
    - If the keyframe only stores servo setpoints (ctrl) and leaves the joints at their
      defaults, the setpoints ARE the intended pose: copy them onto their joints / fixed tendons.
    - Every limited joint starts just inside its range, so no motor begins pressed on a hard stop.
    Returns a short description of what was done.
    """
    notes = []
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, key)
        notes.append("keyframe '%s'" % (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, key) or key))
        ctrl = m.key_ctrl[key]
        moved = 0
        for i in range(m.nu):
            servo = (m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE and m.actuator_biasprm[i, 1] < 0
                     and abs(m.actuator_gainprm[i, 0] + m.actuator_biasprm[i, 1]) < 1e-9 * max(1, abs(m.actuator_gainprm[i, 0])))
            if not servo or ctrl[i] == 0:
                continue
            gear = m.actuator_gear[i, 0] or 1.0
            target = ctrl[i] / gear
            trn, tid = int(m.actuator_trntype[i]), int(m.actuator_trnid[i, 0])
            if trn == int(mujoco.mjtTrn.mjTRN_JOINT):
                a = m.jnt_qposadr[tid]
                if abs(d.qpos[a] - m.qpos0[a]) < 1e-9:
                    d.qpos[a] = target; moved += 1
            elif trn == int(mujoco.mjtTrn.mjTRN_TENDON) and m.tendon_num[tid] and \
                    all(m.wrap_type[m.tendon_adr[tid] + k] == mujoco.mjtWrap.mjWRAP_JOINT for k in range(m.tendon_num[tid])):
                adr, num = m.tendon_adr[tid], m.tendon_num[tid]
                coef = m.wrap_prm[adr:adr + num]
                for k in range(num):
                    a = m.jnt_qposadr[m.wrap_objid[adr + k]]
                    d.qpos[a] = target * coef[k] / float(np.dot(coef, coef))
                moved += 1
        if moved:
            notes.append("%d joints placed at the keyframe's servo setpoints" % moved)
    nudged = 0
    for j in range(m.njnt):
        t = int(m.jnt_type[j])
        if not m.jnt_limited[j] or t not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        lo, hi = m.jnt_range[j]
        mg = min(margin_rad if t == int(mujoco.mjtJoint.mjJNT_HINGE) else margin_m, 0.25 * (hi - lo))
        a = m.jnt_qposadr[j]
        q = float(np.clip(d.qpos[a], lo + mg, hi - mg))
        if q != d.qpos[a]:
            d.qpos[a] = q; nudged += 1
    if nudged:
        notes.append("%d joints moved off their hard stops" % nudged)
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    return notes


def _pose_quality(m, d, bodies, hands):
    """(in_contact, conditioning) of the current configuration of d."""
    mujoco.mj_forward(m, d)
    touching = any((m.geom_bodyid[d.contact[i].geom1] in bodies or m.geom_bodyid[d.contact[i].geom2] in bodies)
                   and d.contact[i].dist < 0 for i in range(d.ncon))
    jacp = np.zeros((3, m.nv))
    cond = []
    for b in hands:
        mujoco.mj_jacBody(m, d, jacp, None, b)
        sv = np.linalg.svd(jacp, compute_uv=False)
        cond.append(sv[-1] / max(sv[0], 1e-9))
    return touching, (min(cond) if cond else 1.0)


def _is_gripper_joint(m, j):
    """Finger and gripper joints legitimately rest at a limit (closed); they do not make a bad pose."""
    b = int(m.jnt_bodyid[j])
    coupled = any(int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT) and j in (m.eq_obj1id[e], m.eq_obj2id[e])
                  for e in range(m.neq))
    leaf = not any(m.body_parentid[c] == b and m.body_jntnum[c] > 0 for c in range(m.nbody))
    return coupled or (leaf and int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE))


def auto_home(m, d, hands, samples=400, seed=0):
    """A collision-free, well-conditioned home pose for a mounted arm whose file has no keyframe
    and whose all-zero pose is in collision or singular (common in URDFs: arms lying along the
    floor or pointing straight up). Random configurations within the joint limits, scaled towards
    the zero pose, are scored on: no contact, conditioning of every hand, distance from limits.
    Returns a note, or None if the zero pose was already fine."""
    from .analysis import robot_bodies

    bodies = set(robot_bodies(m))
    q0 = d.qpos.copy()
    joints = [j for j in range(m.njnt) if m.jnt_bodyid[j] in bodies
              and int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
    touching, cond = _pose_quality(m, d, bodies, hands)
    pinned = any(min(q0[m.jnt_qposadr[j]] - m.jnt_range[j][0], m.jnt_range[j][1] - q0[m.jnt_qposadr[j]])
                 < 0.02 * (m.jnt_range[j][1] - m.jnt_range[j][0]) for j in joints if m.jnt_limited[j]
                 and not _is_gripper_joint(m, j))
    if not touching and cond > 0.05 and not pinned:
        return None
    rng = np.random.default_rng(seed)
    best, best_q = -np.inf, None
    for k in range(samples):
        q = q0.copy()
        spread = 0.15 + 0.85 * k / samples            # start close to the zero pose, widen
        for j in joints:
            a = m.jnt_qposadr[j]
            lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi))
            mid = np.clip(q0[a], lo, hi)
            q[a] = np.clip(mid + spread * rng.uniform(lo - mid, hi - mid), lo, hi)
        d.qpos[:] = q
        touching, cond = _pose_quality(m, d, bodies, hands)
        if touching:
            continue
        margin = min([min(q[m.jnt_qposadr[j]] - m.jnt_range[j][0], m.jnt_range[j][1] - q[m.jnt_qposadr[j]])
                      / max(m.jnt_range[j][1] - m.jnt_range[j][0], 1e-6) for j in joints if m.jnt_limited[j]] or [0.5])
        dist = np.linalg.norm(q - q0) / max(len(joints), 1)
        score = cond + 0.5 * margin - 0.2 * dist
        if score > best:
            best, best_q = score, q.copy()
    if best_q is None:
        d.qpos[:] = q0
        mujoco.mj_forward(m, d)
        return "no collision-free home pose found; starting from the file's zero pose"
    d.qpos[:] = best_q
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    return ("no keyframe, and the zero pose is in collision, singular or pinned against a joint limit: "
            "chose a collision-free home pose")


def solve_stance(m, d, report, clearance=-1e-4, iters=60, verbose=False):
    """Modify d.qpos in place. Returns a dict describing the stance."""
    names = report.feet
    if report.base != "floating" or not names:
        return dict(applied=False)
    mujoco.mj_forward(m, d)
    root = robot_roots(m)[0]
    floor = 0.0
    planes = [d.geom_xpos[g][2] for g in range(m.ngeom)
              if m.geom_bodyid[g] == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    if planes:
        floor = min(planes)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    feet = [bid(f) for f in names]
    contacts = [foot_contact_points(m, d, f, with_radius=True) for f in feet]
    # dofs allowed to move: floating base + joints on root->foot paths
    free = [j for j in range(m.njnt) if m.jnt_bodyid[j] == root and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    dofs = set(range(m.jnt_dofadr[free], m.jnt_dofadr[free] + 6))
    for f in feet:
        b = f
        while b != root and b != 0:
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                dofs.add(int(m.jnt_dofadr[j]))
            b = int(m.body_parentid[b])
    dofs = np.array(sorted(dofs))
    q0 = d.qpos.copy()
    jacp = np.zeros((3, m.nv))
    lam_reg = 1e-3
    for it in range(iters):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        r, J = [], []
        for bodies, pts, _, rad in contacts:
            for b, p, rr in zip(bodies, pts, rad):
                pw = d.xmat[b].reshape(3, 3) @ p + d.xpos[b]
                mujoco.mj_jac(m, d, jacp, None, pw, b)
                r.append(pw[2] - rr - floor - clearance); J.append(jacp[2, dofs].copy())
        r, J = np.array(r), np.array(J)
        # regularise towards the start pose (in velocity space)
        dq0 = np.zeros(m.nv); mujoco.mj_differentiatePos(m, dq0, 1.0, d.qpos, q0)
        A = J.T @ J + lam_reg * np.eye(len(dofs))
        step = -np.linalg.solve(A, J.T @ r - lam_reg * dq0[dofs])
        v = np.zeros(m.nv); v[dofs] = step
        mujoco.mj_integratePos(m, d.qpos, v, 1.0)
        # joint limits
        for j in range(m.njnt):
            if m.jnt_limited[j] and int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                a = m.jnt_qposadr[j]
                d.qpos[a] = np.clip(d.qpos[a], *m.jnt_range[j])
        if np.abs(r).max() < 1e-5 and np.linalg.norm(step) < 1e-6:
            break
    mujoco.mj_forward(m, d)
    d.qvel[:] = 0
    out = dict(applied=True, iterations=it + 1, max_residual_mm=float(np.abs(r).max() * 1000),
               feet={names[i]: ("sole, %d corners" % len(c[1]) if c[2] else "point contact")
                     for i, c in enumerate(contacts)},
               base_shift_mm=float(np.linalg.norm(d.qpos[m.jnt_qposadr[free]:m.jnt_qposadr[free] + 3]
                                                  - q0[m.jnt_qposadr[free]:m.jnt_qposadr[free] + 3]) * 1000))
    if verbose:
        print(out)
    return out


def settle(m, d, report, motors, seconds=0.5):
    """Let a placed robot settle onto its supports under gravity, motors holding its posture
    (wheels and casters find their own contact, a legged robot loads its soles). Real robots are
    set down the same way. Returns the settled state in d."""
    from .control import WholeBodyController

    if report.base != "floating":
        return
    w = WholeBodyController(m, report, motors, d)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step1(m, d)
        motors.command(d, tau_ff=w.step(d))
        mujoco.mj_step2(m, d)
