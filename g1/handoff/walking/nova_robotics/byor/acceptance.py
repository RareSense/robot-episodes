"""Acceptance tests that every imported robot goes through before it is offered for teleoperation.

All tests run the robot through its real motors (torque-limited by the robot's own file) and
MuJoCo physics; nothing is puppeted. The result is a trust level:

  level 0  the description is not physically usable (errors in the analysis, or unstable physics)
  level 1  physically valid: masses, inertias, limits and contacts make sense; passive drop is stable
  level 2  controllable: holds its pose, and every motor can move its joint through its range
  level 3  teleop-ready: balances against shoves (floating robots) and its hands reach targets

Shoves are sized from the robot's own physics: the largest velocity change a standing robot can
absorb without stepping is about omega x (CoM-to-sole-edge distance), omega = sqrt(g / CoM height)
(linear inverted pendulum). Each robot is shoved at a fixed fraction of its own bound.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from .analysis import Report, analyze_model, robot_bodies
from .control import MotorLayer, WholeBodyController
from .stance import auto_home, settle, solve_stance, start_pose

SHOVE_FRACTION = 0.4          # of the robot's own linear-inverted-pendulum bound


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str
    metrics: dict = field(default_factory=dict)


@dataclass
class Acceptance:
    trust_level: int
    summary: str
    tests: list[TestResult]
    seconds: float

    def as_dict(self):
        return dict(trust_level=self.trust_level, summary=self.summary, seconds=round(self.seconds, 1),
                    tests=[asdict(t) for t in self.tests])


# ------------------------------------------------------------------------------------ rig

class Rig:
    """A fresh model + data + motors + controller, placed in its start stance.

    gantry=True mounts a floating robot on a rigid test stand: its free joint is removed and the
    base is fixed in the stance pose, exactly like hanging a real robot on a gantry for bench tests.
    """

    def __init__(self, spec: mujoco.MjSpec, report: Report, gantry=False, control=True):
        s = mujoco.MjSpec.from_string(spec.to_xml()) if not hasattr(spec, "copy") else spec.copy()
        m = s.compile()
        d = mujoco.MjData(m)
        start_pose(m, d)
        if report.base == "fixed" and not m.nkey:
            hands = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in report.end_effectors]
            auto_home(m, d, [h for h in hands if h >= 0])
        if report.base == "floating":
            solve_stance(m, d, report)
        if gantry and report.base == "floating":
            m, d = self._mount(s, m, d)
            report = _fixed_view(report)
        self.m, self.d, self.report = m, d, report
        self.motors = MotorLayer(m)
        if control and not gantry and report.base == "floating":
            settle(m, d, report, self.motors)
        self.wbc = WholeBodyController(m, report, self.motors, d) if control else None
        self.bodies = set(robot_bodies(m))
        self.over = 0.0
        self.viol = 0
        self.peak = 0.0
        self.pen = 0.0

    @staticmethod
    def _mount(s, m, d):
        root = robot_roots_of(m)[0]
        rname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, root)
        joints = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 1].copy()
                  for j in range(m.njnt) if int(m.jnt_type[j]) in (2, 3)}
        body = s.body(rname)
        free = [j for j in body.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
        body.pos = d.xpos[root].copy() + [0, 0, 0.15]   # hung clear of the floor, like a real gantry
        body.quat = d.xquat[root].copy()
        for j in free:
            s.delete(j)
        for k in list(s.keys):                    # keyframes still describe the free joint
            s.delete(k)
        m2 = s.compile()
        d2 = mujoco.MjData(m2)
        for jn, q in joints.items():
            j = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if j >= 0:
                d2.qpos[m2.jnt_qposadr[j]] = q[0]
        mujoco.mj_forward(m2, d2)
        return m2, d2

    def step(self, push=None):
        m, d = self.m, self.d
        mujoco.mj_step1(m, d)
        if self.wbc is not None:
            self.motors.command(d, tau_ff=self.wbc.step(d))
        else:
            d.ctrl[:] = 0
        if push is not None:
            d.xfrc_applied[push[0], :3] = push[1]
        mujoco.mj_step2(m, d)
        self._record()

    def _record(self):
        m, d = self.m, self.d
        ap = np.abs(self.motors.applied(d))
        lim = np.where(np.isfinite(self.motors.limit), self.motors.limit, np.inf)
        self.viol += int((ap > lim * 1.001).sum())
        finite = np.isfinite(lim) & (lim > 0)
        if finite.any():
            self.peak = max(self.peak, float((ap[finite] / lim[finite]).max()))
        for j in range(m.njnt):
            if m.jnt_limited[j] and int(m.jnt_type[j]) in (2, 3) and m.jnt_bodyid[j] in self.bodies:
                a = m.jnt_qposadr[j]
                self.over = max(self.over, m.jnt_range[j][0] - d.qpos[a], d.qpos[a] - m.jnt_range[j][1])
        for i in range(d.ncon):
            c = d.contact[i]
            if (m.geom_bodyid[c.geom1] in self.bodies) != (m.geom_bodyid[c.geom2] in self.bodies):
                self.pen = max(self.pen, -c.dist)

    def upright(self):
        root = robot_roots_of(self.m)[0]
        return float(self.d.xmat[root][8])

    def com_height(self):
        return float(self.d.subtree_com[robot_roots_of(self.m)[0]][2])

    def common_metrics(self):
        return dict(peak_torque_pct=round(100 * self.peak, 1), torque_limit_violations=self.viol,
                    joint_overshoot_mm_or_mrad=round(max(self.over, 0) * 1000, 2),
                    max_ground_penetration_mm=round(self.pen * 1000, 2))


def _fixed_view(report: Report) -> Report:
    from copy import deepcopy
    r = deepcopy(report)
    r.base, r.feet, r.wheels, r.end_effectors = "fixed", [], [], []
    return r


def robot_roots_of(m):
    from .analysis import robot_roots
    return robot_roots(m)


# ------------------------------------------------------------------------------------ tests

def check_drop_and_settle(spec, report, seconds=2.0) -> TestResult:
    """Motors off, released 5 cm up (floating robots) or simply let go (mounted ones). Energy must
    not come from nowhere: once settled it may not exceed the start, and brief impact transients
    (MuJoCo's soft contacts store impact energy for a moment) must stay small next to the robot's
    own potential energy. An unstable model keeps gaining energy and fails both."""
    rig = Rig(spec, report, control=False)
    m, d = rig.m, rig.d
    m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    root = robot_roots_of(m)[0]
    if report.base == "floating":
        d.qpos[m.jnt_qposadr[m.body_jntadr[root]] + 2] += 0.05
    mujoco.mj_forward(m, d)
    e0 = float(d.energy.sum())
    mass = float(sum(m.body_mass[b] for b in rig.bodies))
    scale = mass * 9.81 * max(float(d.subtree_com[root][2]), 0.05)
    worst, vmax = 0.0, 0.0
    for _ in range(int(seconds / m.opt.timestep)):
        rig.step()
        if not np.isfinite(d.qacc).all():
            return TestResult("drop_and_settle", False, "simulation diverged (NaN)", {})
        worst = max(worst, float(d.energy.sum()) - e0)
        vmax = max(vmax, float(np.abs(d.qvel).max()))
    final = float(d.energy.sum()) - e0
    ok = final <= 0.01 * scale + 1e-3 and worst <= 0.25 * scale
    return TestResult("drop_and_settle", ok,
                      f"settled with {-final:.2f} J less energy than it started with; largest impact "
                      f"transient {worst:.2f} J ({100 * worst / scale:.1f} % of its potential energy)" if ok else
                      f"energy grew: {final:+.2f} J after settling, transient {worst:.2f} J "
                      f"({100 * worst / scale:.1f} % of its potential energy)",
                      dict(final_energy_change_J=round(final, 4), peak_transient_J=round(worst, 4),
                           max_joint_speed=round(vmax, 2), max_ground_penetration_mm=round(rig.pen * 1000, 2)))


def check_hold(spec, report, seconds=3.0, settle=1.0) -> TestResult:
    """Hold the rest pose. The controller may first move the CoM to a safer spot inside the
    support area; after `settle` seconds the robot must be still."""
    rig = Rig(spec, report)
    root = robot_roots_of(rig.m)[0]
    for _ in range(int(settle / rig.m.opt.timestep)):
        rig.step()
    p0 = rig.d.xpos[root].copy()
    for _ in range(int((seconds - settle) / rig.m.opt.timestep)):
        rig.step()
    drift = float(np.linalg.norm(rig.d.xpos[root] - p0))
    hand = max([float(np.linalg.norm(rig.wbc.ee_pos(rig.d, b) - rig.wbc.ee_goal[b])) for b in rig.wbc.ees] or [0.0])
    ok = rig.upright() > 0.8 and drift < 0.005 and hand < 0.01 and rig.viol == 0
    return TestResult("hold_pose", ok,
                      f"held {seconds:.0f} s; once settled the base moved {drift * 1000:.1f} mm, "
                      f"hands are {hand * 1000:.1f} mm from target",
                      dict(base_drift_mm=round(drift * 1000, 2), hand_error_mm=round(hand * 1000, 2),
                           **rig.common_metrics()))


def check_joint_sweep(spec, report, fraction=0.8, max_seconds=6.0, reach=np.pi) -> TestResult:
    """On a test stand, each driven joint moves to `fraction` of its range both ways (at most
    `reach` from home), each joint from a fresh rig so one joint's result cannot affect the next.
    A target counts as reached when the joint arrives; the attempt ends early when the joint
    stalls (no motion for 0.3 s), so slow but honest motors are not penalised."""
    view = _fixed_view(report) if report.base == "fixed" else report
    probe = Rig(spec, view, gantry=True)
    driven = []
    for k in probe.motors.joint_idx:
        j = int(probe.m.dof_jntid[int(probe.motors.dof[k])])
        if probe.m.jnt_limited[j]:
            driven.append(mujoco.mj_id2name(probe.m, mujoco.mjtObj.mjOBJ_JOINT, j))
    failed, blocked, reached = [], [], 0
    peak, viol, over = 0.0, 0, 0.0
    for jn in driven:
        for side in (1, -1):
            rig = Rig(spec, view, gantry=True)
            m, d, w = rig.m, rig.d, rig.wbc
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            a, dof = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            lo, hi = m.jnt_range[j]
            mid, home = 0.5 * (lo + hi), float(w.qpos0[a])
            tol = 0.05 if int(m.jnt_type[j]) == 3 else 0.005
            target = mid + fraction * ((hi if side > 0 else lo) - mid)
            target = float(np.clip(target, home - reach, home + reach))
            w.qpos0[a] = target
            still, dt = 0.0, m.opt.timestep
            for _ in range(int(max_seconds / dt)):
                rig.step()
                err = abs(d.qpos[a] - target)
                if err < tol and abs(d.qvel[dof]) < 0.05:
                    break
                still = still + dt if abs(d.qvel[dof]) < 0.02 else 0.0
                if still > 0.3:                    # stopped short: stalled against something
                    break
            err = abs(d.qpos[a] - target)
            if err <= tol:
                reached += 1
            elif _moving_part_touches(rig, j):
                blocked.append(jn)                 # stopped by the floor or its own body: geometry
            else:
                failed.append(f"{jn} ({err:.3f})")
            peak, viol, over = max(peak, rig.peak), viol + rig.viol, max(over, rig.over)
    ok = not failed and viol == 0
    return TestResult("joint_sweep", ok,
                      f"{reached} of {2 * len(driven)} sweep targets reached" +
                      (f"; blocked by contact (geometry, not motors): {', '.join(sorted(set(blocked))[:8])}"
                       if blocked else "") +
                      (f"; motors too weak or controller short: {', '.join(failed[:8])}" if failed else ""),
                      dict(joints=len(driven), targets_reached=reached, blocked_by_contact=len(blocked),
                           short=len(failed), peak_torque_pct=round(100 * peak, 1),
                           torque_limit_violations=viol, joint_overshoot_mm_or_mrad=round(max(over, 0) * 1000, 2)))


def _moving_part_touches(rig, joint) -> bool:
    """Is anything this joint moves (its body and everything below it, plus joints coupled to it,
    such as a mimicking second finger) in contact with something that does not move with it?"""
    from .analysis import subtree
    m, d = rig.m, rig.d
    joints = {joint}
    for e in range(m.neq):
        if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT) and joint in (m.eq_obj1id[e], m.eq_obj2id[e]):
            joints |= {int(m.eq_obj1id[e]), int(m.eq_obj2id[e])}
    moving = {b for jj in joints for b in subtree(m, int(m.jnt_bodyid[jj]))}
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (b1 in moving) != (b2 in moving) and c.dist < 1e-3:
            return True
    return False


def _touching(rig) -> set:
    """Pairs of bodies (robot-world or robot-robot) currently in contact."""
    m, d = rig.m, rig.d
    out = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if b1 in rig.bodies or b2 in rig.bodies:
            out.add((min(b1, b2), max(b1, b2)))
    return out


def lip_bounds(rig: Rig) -> dict:
    w = rig.wbc
    om = np.sqrt(9.81 / w.com0[2])
    P, c = w.support_points, w.com0[:2]
    return {"forward": om * max(P[:, 0].max() - c[0], 0), "backward": om * max(c[0] - P[:, 0].min(), 0),
            "sideways": om * max(min(P[:, 1].max() - c[1], c[1] - P[:, 1].min()), 0)}


def check_shoves(spec, report, seconds=3.0) -> TestResult:
    dirs = {"forward": (1, 0, 0), "backward": (-1, 0, 0), "sideways": (0, 1, 0)}
    probe = Rig(spec, report)
    if not probe.wbc.feet:
        dirs = {"forward": (1, 0, 0), "sideways": (0, 1, 0)}
        bounds = {k: 0.25 for k in dirs}
        basis = "fixed 0.25 m/s (no feet to bound it)"
    else:
        bounds = {k: SHOVE_FRACTION * v for k, v in lip_bounds(probe).items()}
        basis = f"{int(SHOVE_FRACTION * 100)} % of this robot's own standing limit"
    results, metrics, ok = [], {}, True
    for name, v in bounds.items():
        rig = Rig(spec, report)
        root = robot_roots_of(rig.m)[0]
        mass = float(rig.m.body_subtreemass[root])
        h0 = rig.com_height()
        fell = None
        for k in range(int(seconds / rig.m.opt.timestep)):
            t = k * rig.m.opt.timestep
            force = np.array(dirs[name]) * mass * v / 0.2 if 1.0 <= t < 1.2 else np.zeros(3)
            rig.step((root, force))
            if fell is None and (rig.upright() < 0.6 or rig.com_height() < 0.6 * h0):
                fell = t
        survived = fell is None and rig.viol == 0
        ok &= survived
        results.append(f"{name} {v:.2f} m/s {'held' if survived else 'FELL'}")
        metrics[name] = dict(shove_m_per_s=round(v, 3), survived=survived, **rig.common_metrics())
    return TestResult("shoves", ok, f"{'; '.join(results)} ({basis})", metrics)


def _ik_residual(rig, body, offset, target, cols, iters=40):
    """Kinematic check: how close can the arm's own joints (within limits) bring the tool point to
    the target, holding the hand's orientation when the controller does (arms with >= 6 joints)?
    Works at singular poses where a single Jacobian step cannot tell up from down."""
    from .control import quat_err

    m = rig.m
    d = mujoco.MjData(m)
    d.qpos[:] = rig.d.qpos
    keep_rot = rig.wbc.ee_dof.get(body, 0) >= 6
    quat0 = rig.d.xquat[body].copy()
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p = d.xpos[body] + d.xmat[body].reshape(3, 3) @ offset
        e = target - p
        if np.linalg.norm(e) < 1e-4:
            break
        mujoco.mj_jac(m, d, jacp, jacr, p, body)
        J = jacp[:, cols]
        if keep_rot:
            J = np.vstack([J, jacr[:, cols]])
            e = np.r_[e, quat_err(d.xquat[body], quat0)]
        dq = np.zeros(m.nv)
        dq[cols] = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
        mujoco.mj_integratePos(m, d.qpos, dq, 1.0)
        for j in range(m.njnt):
            if m.jnt_limited[j] and int(m.jnt_type[j]) in (2, 3):
                a = m.jnt_qposadr[j]
                d.qpos[a] = np.clip(d.qpos[a], *m.jnt_range[j])
    mujoco.mj_kinematics(m, d)
    return float(np.linalg.norm(target - (d.xpos[body] + d.xmat[body].reshape(3, 3) @ offset)))


def check_reach(spec, report, seconds=3.0) -> TestResult:
    """Each hand moves forward and up by a fixed share of its own arm's reach (8 cm / 5 cm for a
    typical 0.4 m arm), as a sudden command; the robot must stay balanced and arrive within 2 cm."""
    rig = Rig(spec, report)
    w = rig.wbc
    if not w.ees:
        return TestResult("reach", True, "no hands", {})
    arm = min(report.end_effector_reach_m.values() or [0.4])
    forward, up = min(0.08, 0.2 * arm), min(0.05, 0.125 * arm)
    root = robot_roots_of(rig.m)[0]
    R = rig.d.xmat[root].reshape(3, 3)
    fwd = np.array([R[0, 0], R[1, 0], 0.0])
    fwd /= np.linalg.norm(fwd) or 1
    side = np.array([-fwd[1], fwd[0], 0.0])
    arm_cols = [i for i in range(rig.m.nv) if int(rig.m.jnt_type[rig.m.dof_jntid[i]]) != 0]
    goal, chosen = {}, {}
    jacp = np.zeros((3, rig.m.nv))
    for b in w.ees:
        # the horizontal direction this arm reaches most easily with its own joints (forward for
        # most robots; along the arm for a side-mounted telescoping arm), always combined with "up"
        mujoco.mj_jac(rig.m, rig.d, jacp, None, w.ee_pos(rig.d, b), b)
        Ja = jacp[:, arm_cols]
        costs = {}
        for name_, u, vz in [(f"{h} and {v}", hv, z) for v, z in (("up", up), ("down", -up))
                             for h, hv in (("forward", fwd), ("left", side), ("right", -side), ("back", -fwd))]:
            step = forward * u + np.array([0.0, 0.0, vz])
            costs[name_] = (_ik_residual(rig, b, w.ee_off[b], w.ee_goal[b] + step, arm_cols), step)
        easiest = min(costs, key=lambda k: costs[k][0])
        pick = "forward and up" if costs["forward and up"][0] <= costs[easiest][0] + 0.002 else easiest
        best = (costs[pick][0], pick, costs[pick][1])
        chosen[b], goal[b] = best[1], w.ee_goal[b] + best[2]
    h0 = rig.com_height()
    fell = None
    for k in range(int(seconds / rig.m.opt.timestep)):
        if k * rig.m.opt.timestep >= 0.5:
            for b in w.ees:
                w.set_ee_target(b, goal[b])
        rig.step()
        if fell is None and (rig.upright() < 0.6 or rig.com_height() < 0.6 * h0):
            fell = k * rig.m.opt.timestep
    err = max(float(np.linalg.norm(w.ee_pos(rig.d, b) - goal[b])) for b in w.ees)
    ok = fell is None and err < 0.02 and rig.viol == 0
    return TestResult("reach", ok,
                      f"hands moved {forward * 100:.0f} cm {' / '.join(sorted(set(chosen.values())))} "
                      f"({up * 100:.0f} cm vertically): "
                      f"{'error %.1f cm' % (err * 100) if fell is None else 'robot fell'}",
                      dict(hand_error_mm=round(err * 1000, 1), **rig.common_metrics()))


# ------------------------------------------------------------------------------------ driver

def run_acceptance(spec: mujoco.MjSpec, report: Report | None = None, quick=False,
                   load_problems: list | None = None) -> Acceptance:
    t0 = time.perf_counter()
    if report is None:
        report = analyze_model(spec.compile())
    tests = []
    errors = [i for i in report.issues if i["severity"] == "error"]
    errors += [{"detail": p} for p in (load_problems or [])]
    tests.append(TestResult("description_valid", not errors,
                            "no errors in the analysis" if not errors else
                            f"{len(errors)} errors: " + "; ".join(e["detail"] for e in errors[:4]), {}))
    tests.append(check_drop_and_settle(spec, report))
    level = 1 if all(t.passed for t in tests) else 0
    if level == 1:
        hold = check_hold(spec, report)
        tests.append(hold)
        if not quick:
            tests.append(check_joint_sweep(spec, report))
        if all(t.passed for t in tests):
            level = 2
            more = []
            if report.base == "floating":
                more.append(check_shoves(spec, report))
            more.append(check_reach(spec, report))
            tests += more
            if all(t.passed for t in more):
                level = 3
    summaries = {0: "not usable yet: fix the listed problems in the robot description",
                 1: "physically valid, but the controller could not hold or drive it",
                 2: "controllable; balance or reach did not pass yet, so teleop is limited",
                 3: "teleop-ready: balances and reaches with its real motor limits"}
    return Acceptance(level, summaries[level], tests, time.perf_counter() - t0)
