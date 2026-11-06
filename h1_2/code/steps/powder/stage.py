r"""Milk powder stage for the integrated episode. run(s, ctx) -> dict.

s    live h12_physx.Sim, built with mimic=True, standing near the counter, both hands free. Build it with
     pre=stage.pre_hooks() in props.make_sim so the jar shell, lid, cup and grains exist: they must be authored before physics loads.
ctx  {"mug_xy": (x, y), "mug": mug prim path (optional), "grains": True (optional)}

Uses only poses measured from the live sim. Never calls app.close or sys.exit. No welds, no state writes, no forces.
Ends with the robot standing and the left hand free. Returns status PASS or FAIL with its numbers; a failed step returns, it does not raise.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))    # release_paths.py lives at the code root
import release_paths as RP
import os
import sys

sys.dont_write_bytecode = True
for _p in (RP.code(), RP.code('props'), RP.code('steps', 'powder'), RP.code("balance_support")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import h12_physx as H
import numpy as np
import pwprops as PP
from crouch_support import CrouchSupport
from arm_servo import ArmPoseHold, configure_fingers

N_GRAINS = 300                                           # MODEL VARIATION (runs scoop12, scoop13): 300 loose grains, not 120; total powder mass preserved
GRAIN_SPREAD = 0.027
GRAIN_DAMP = float(os.environ.get("P_DAMP", 60.0))        # ASSUMPTION: angular damping as a stand-in for rolling resistance, PhysX has none. Was 8; at 8 grains roll out of the lifting scoop (run p_c1: 8 kept)
GRAIN_SLEEP = 0.0                                        # grains never sleep: with 0.0005 they froze in mid air under the tipped scoop (bench runs p_lab01-03)
TARGET_GRAINS = 20                                       # working target, not confirmed by the user
MAX_DIPS = int(os.environ.get("P_DIPS", 3))
POUR_Z = 0.24                                            # scoop origin above the mug base at full tilt. Lower (0.185, run p_c1) brings the open pinky and ring onto the mug rim
ROCKS = 2
DROP = 0.08
STANCE_XY = (1.95, 1.90)
LSQ, CSQ = 0.22, 0.05                                    # lid grip: lifted in p_c1 and p_c3 (jar shifted 4.4 and 5.2 mm); 0.12 dropped the lid (run p_c4, 6 mm rise)
CUP_LSQ, CUP_CSQ = 0.18, 0.02                            # scoop grip. 0.14 squeezed the block out at the lift in 2 of 3 runs (p_c2, p_c3)
HAND_TO_BLOCK = np.array([-0.016, -0.1614, 0.0555])      # hand origin from the grab block's centre, palm down (measured grip, runs l4, c1)
CUP_TO_BLOCK = np.array([-0.016, -0.1614, 0.0705])       # scoop grip sits 15 mm higher: the lower one touched the jar with the pinky (run scoop13)
QL = None


def pre_hooks(grains=True, lid=True):
    """Authoring hooks for props.make_sim(pre=...). The mug is NOT moved here: the mug step puts it on the counter."""
    hooks = [lambda st: PP.install_jar(st, with_lid=lid), PP.install_scoop]
    if grains:
        hooks.append(PP.grains_pre(N_GRAINS, damp=GRAIN_DAMP, spread=GRAIN_SPREAD, sleep=GRAIN_SLEEP))
        hooks.append(_preserve_powder_mass)
    return hooks


def _preserve_powder_mass(st):
    """The extra loose grains come out of the packed rigid fill, so jar plus powder keeps the mass of the 120-grain model."""
    from pxr import UsdPhysics
    m = UsdPhysics.MassAPI(st.GetPrimAtPath(PP.JAR)).GetMassAttr()
    m.Set(float(m.Get()) - (N_GRAINS - 120) * 600.0 * 4.0 / 3.0 * np.pi * PP.GRAIN_R ** 3)
    print(f"MODEL VARIATION: {N_GRAINS} loose grains, total powder mass preserved", flush=True)


def _rotv(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[axis * np.sin(ang / 2), np.cos(ang / 2)]


def _q_palm_down():
    # columns x=(0,0,1), y=(0,-1,0), z=(1,0,0)
    return np.array([0.70710678, 0.0, 0.70710678, 0.0])


class _Fail(Exception):
    pass


def _to_walk(s):
    """Stance back to the walking network: fresh memory, walking leg gains, zero command warm-up (mug_episode.to_walk)."""
    a = s.art
    s.h[:], s.c[:], s.last[:] = 0, 0, 0
    s.target = a.get_dof_positions()[0].astype(float)[s.legs].copy()
    kp = a.get_dof_stiffnesses()[0].astype(float).copy()
    kd = a.get_dof_dampings()[0].astype(float).copy()
    kp[s.legs], kd[s.legs] = H.G["kp"], H.G["kd"]
    a.set_dof_stiffnesses(s.f(kp), s.idx)
    a.set_dof_dampings(s.f(kd), s.idx)
    s.want[:], s.cmd[:] = 0, 0
    s.mode = "walk"


def _walk_to(s, x, y, yaw=90.0, tol=0.025):
    """mug_episode.walk_to: walk, correct until the feet are planted, then stand."""
    if not s.go(x, y, yaw, tol=tol, log=lambda *z, **k: None):
        return False
    gy = np.radians(yaw)
    fz = []
    for n in range(240):
        p, q = s.pose()
        yw = H.yaw_of(q)
        e = np.array([x, y]) - p[:2]
        c, sn = np.cos(yw), np.sin(yw)
        s.want = np.array([H.min_cmd(c * e[0] + sn * e[1], 1.0, 0.14, tol), H.min_cmd(-sn * e[0] + c * e[1], 1.0, 0.14, tol), np.clip(1.2 * H.wrap(gy - yw), -0.3, 0.3)])
        for _ in range(H.PERIOD):
            s.step()
            z = s.feet_xy()[:, 2]
            fz.append(z.copy())
        if n >= 60 and float(np.linalg.norm(e)) < 0.02 and np.all(z - np.min(fz[-120:], axis=0) < 0.004):
            break
    s.want = np.zeros(3)
    s.to_stand(0.6)
    s.hold_for(1.5)
    return True


def run(s, ctx):
    out = dict(status="FAIL", task_status="FAIL", stages={}, nums={}, why=None)     # task_status: lid off and at least 20 grains staying in the mug. Separate from status and from the strict gates
    a = s.art
    views, bodies = {}, {}
    q_down = _q_palm_down()

    def body(path):
        if path not in bodies:
            bodies[path] = s.sv.create_rigid_body_view(path)
        return bodies[path]

    def pose(path):
        return body(path).get_transforms()[0].astype(float)

    def touch(path):
        if path not in views:
            hand = [x for x in a.link_paths[0] if "/L_hand_base_link" in x or x.endswith("left_wrist_yaw_link")]
            views[path] = (s.sv.create_rigid_contact_view([path], [hand], 1024), hand)
        v, hand = views[path]
        m = v.get_contact_force_matrix(H.DT)[0].astype(float)
        return {hand[i].rsplit("/", 1)[-1]: round(float(np.linalg.norm(m[i])), 2) for i in range(len(hand)) if np.linalg.norm(m[i]) > 0.01}

    def mark(name, err=None, **more):
        out["stages"][name] = dict(t=round(s.t, 2), err_mm=None if err is None else round(err * 1000, 1), hand=np.round(s.hand("L")[:3], 3).tolist(), scoop=pose(PP.SCOOP).tolist(), **more)
        print("STAGE", name, out["stages"][name], flush=True)
        if s.fallen():
            raise _Fail("fell at " + name)

    def go(name, p, q, sec=1.6):
        err = s.ik_to(p, q, sec, side="L", w_rot=1.0, settle=80, gain=.10)
        mark(name, err)
        return err

    links = {k: (f"L_{k}_proximal", f"L_{k}_intermediate", f"L_{k}_distal") for k in ("index", "middle")}

    def creep(path):
        cur = {k: s.hold[s.finger_idx("L", k)[0][0]] for k in links}
        done, t = {}, 0.0
        while len(done) < 2 and t < 4.0:
            c = touch(path)
            for k in links:
                if k not in done:
                    if any(c.get(n, 0.0) > 0.1 for n in links[k]):
                        done[k] = cur[k]
                    else:
                        cur[k] = min(1.2, cur[k] + 0.004)
            s.fingers("L", dict(cur), 0.02)
            t += 0.02
        return done, cur

    def grip(path, block, name, lift=0.05, object_q=None, to_block=HAND_TO_BLOCK, side_fingers=0.45, thumb_pitch=0.3, squeeze=LSQ):
        object_q = np.array([0., 0., 0., 1.]) if object_q is None else object_q
        hp = block + H.qrot(object_q, to_block)
        q_grip = H.qmul(object_q, q_down)
        s.fingers("L", dict(index=0.45, middle=0.45, ring=side_fingers, pinky=side_fingers, thumb_yaw=1.3, thumb_pitch=thumb_pitch), 1.0)
        go(name + "_above", hp + np.array([0, 0, 0.06]), q_grip, 2.4)
        go(name + "_down", hp, q_grip, 1.8)
        holder = ArmPoseHold(s)
        holder.hold('L', hp, q_grip)
        try:
            done, cur = creep(path)
            s.fingers("L", {k: cur[k] + squeeze for k in cur}, 0.6)
            s.hold_for(0.3)
            out["nums"][name + "_grip_N"] = touch(path)
        finally:
            holder.detach()
        if gt is not None:
            gt.hold(path, "L_hand_base_link")
        return cur

    def grains_counts(mug_path, jar_xy):
        v = ctx.get("_gv")
        if v is None:
            return {}
        G = np.asarray(v.get_transforms(), float)[:, :3]
        m = pose(mug_path)
        rm = np.hypot(G[:, 0] - m[0], G[:, 1] - m[1])
        jp, cp = pose(PP.JAR), pose(PP.SCOOP)
        def local_points(p):
            v = G - p[:3]
            u, w = -p[3:6], p[6]
            return v + 2 * np.cross(u, np.cross(u, v) + w * v)
        j, c = local_points(jp), local_points(cp)
        in_mug = (rm < 0.031) & (G[:, 2] > m[2]) & (G[:, 2] < m[2] + 0.09)
        in_scoop = (np.linalg.norm(c[:, :2], axis=1) < PP.SC_R + PP.GRAIN_R) & (c[:, 2] > 0.001) & (c[:, 2] < PP.SC_H + PP.GRAIN_R)
        in_jar = (np.linalg.norm(j[:, :2], axis=1) < PP.R_IN + PP.GRAIN_R) & (j[:, 2] > 0.003) & (j[:, 2] < PP.NECK_TOP + PP.GRAIN_R) & ~in_scoop
        return dict(grains_total=len(G), grains_in_mug=int(in_mug.sum()),
                    grains_in_scoop=int(in_scoop.sum()), grains_in_jar=int(in_jar.sum()),
                    grains_elsewhere=int((~in_mug & ~in_scoop & ~in_jar).sum()))

    sup = None
    saved_gains = None
    gt = ctx.get("gates")                                 # optional props.gates.Gates: slip and moved are then judged inside this stage
    try:
        p0 = s.pose()[0]
        d0 = float(np.linalg.norm(p0[:2] - np.array(STANCE_XY)))
        out["nums"]["stance_err_mm"] = round(d0 * 1000, 1)
        if d0 > 0.04:                                     # the mug step ends elsewhere: walk to the stance, same pattern as mug_episode.py
            _to_walk(s)
            for _ in range(300):
                s.step()
            if not _walk_to(s, STANCE_XY[0], STANCE_XY[1]):
                raise _Fail("walk to the stance timed out")
            p0 = s.pose()[0]
            d0 = float(np.linalg.norm(p0[:2] - np.array(STANCE_XY)))
            out["nums"]["stance_err_after_walk_mm"] = round(d0 * 1000, 1)
            mark("at_stance")
        if d0 > 0.06:
            raise _Fail(f"stance {np.round(p0[:2], 3).tolist()} is {d0 * 1000:.0f} mm from {STANCE_XY}")
        mug_path = ctx.get("mug", PP.MUG)
        mug_xy = np.asarray(ctx["mug_xy"], float)
        if ctx.get("grains", True):
            ctx["_gv"] = s.sv.create_rigid_body_view("/World/fluid/grains/g*")
        saved_gains = (a.get_dof_stiffnesses()[0].copy(), a.get_dof_dampings()[0].copy(), s.kp.copy(), s.kd.copy())
        configure_fingers(s, "L", kp_value=10.0, kd_value=0.2)   # same 1 Nm motor caps; restored at the end of the stage
        k0 = a.get_dof_stiffnesses()[0].astype(float).copy()
        ty = s.finger_idx("L", "thumb_yaw")[0][0]
        k0[ty] = 10.0
        s.kp[ty] = 10.0                                   # in s.kp as well: to_stand() restores s.kp
        a.set_dof_stiffnesses(s.f(k0), s.idx)
        sup = CrouchSupport(s)
        c = sup.crouch(DROP, seconds=2.0, settle=0.6)
        out["nums"]["crouch"] = bool(c.get("passed"))
        jar = pose(PP.JAR)[:3]
        lid0 = pose(PP.LID).copy() if ctx.get("only") != "cup" else None

        if ctx.get("only") != "cup":                      # ctx "only" is a test switch: "lid" or "cup" runs half the stage
            # ---- lid
            knob = lid0[:3] + np.array([0, 0, PP.LID_TOP_T + PP.KNOB_HALF[2]])
            if gt is not None:
                gt.allow_move("lid")
            grip(PP.LID, knob, "lid")
            h = s.hand("L").copy()
            e = jar[:2] - pose(PP.LID)[:2]                # closing the hand shoves the lid against the neck (3 mm deep at the lift, runs t2, p_c1): centre it from the measured pose first
            out["nums"]["lid_off_centre_mm"] = round(float(np.linalg.norm(e)) * 1000, 2)
            go("lid_centre", h[:3] + np.r_[e, 0.0], q_down, 0.8)
            h = s.hand("L").copy()
            go("lid_lift", h[:3] + np.array([0, 0, 0.07]), q_down, 3.0)
            rise = (pose(PP.LID)[2] - lid0[2]) * 1000
            out["nums"]["lid_rise_mm"] = round(float(rise), 1)
            if rise < 12:
                raise _Fail("lid not lifted")
            lp = pose(PP.LID)[:3]
            goal = np.array([jar[0] - 0.15, mug_xy[1] - 0.04])
            h = s.hand("L").copy()
            go("lid_side", h[:3] + np.r_[goal - lp[:2], 0.0], q_down, 2.2)
            h = s.hand("L").copy()
            rest_z = jar[2] + PP.NECK_TOP - PP.LID_Z0
            go("lid_low", h[:3] + np.array([0, 0, rest_z + 0.004 - pose(PP.LID)[2]]), q_down, 2.2)
            h = s.hand("L").copy()
            go("lid_touch", h[:3] + np.array([0, 0, -0.005]), q_down, 1.0)
            if gt is not None:
                gt.release(PP.LID)
            s.fingers("L", dict(index=0.3, middle=0.3), 1.0)
            s.hold_for(0.3)
            h = s.hand("L").copy()
            go("lid_clear", h[:3] + np.array([0, 0, 0.09]), q_down, 1.6)
            lf = pose(PP.LID)
            out["nums"]["lid_final"] = np.round(lf[:3], 4).tolist()
            out["nums"]["lid_on_counter"] = bool(abs(lf[2] - rest_z) < 0.004)
            print("LID", out["nums"], flush=True)
            if ctx.get("only") == "lid":
                out["status"] = "PASS" if out["nums"]["lid_on_counter"] else "FAIL"
                return out

        # ---- powder by the cup
        c0 = pose(PP.SCOOP).copy()
        # This approach offset was empirically calibrated on the resting tool.
        # Use measured full tool orientation for transport after the grasp.
        for attempt in (1, 2):                            # the pinch can squeeze the block out sideways (run p_c2): one regrip from the measured pose
            cn = pose(PP.SCOOP).copy()
            cur = grip(PP.SCOOP, cn[:3] + H.qrot(cn[3:7], np.array(PP.SC_BLOCK)), "cup", to_block=CUP_TO_BLOCK, side_fingers=0.0, thumb_pitch=0.40, squeeze=CUP_LSQ)
            h = s.hand("L").copy()
            go("cup_lift0", h[:3] + np.array([0, 0, 0.015]), h[3:7], 1.5)     # break it off the counter slowly and tighten there, before the load swings
            s.fingers("L", {k: cur[k] + CUP_LSQ + CUP_CSQ for k in cur}, 0.6)
            s.hold_for(0.3)
            go("cup_lift", h[:3] + np.array([0, 0, 0.07]), h[3:7], 1.8)
            rise = (pose(PP.SCOOP)[2] - cn[2]) * 1000
            out["nums"]["cup_rise_mm"] = round(float(rise), 1)
            out["nums"]["cup_grip_attempts"] = attempt
            if rise >= 15:
                break
            if attempt == 2:
                raise _Fail("cup not lifted")
            if gt is not None:
                gt.release(PP.SCOOP)
            s.fingers("L", dict(index=0.3, middle=0.3), 0.8)
            h = s.hand("L").copy()
            go("cup_regrip_clear", h[:3] + np.array([0, 0, 0.03]), h[3:7], 1.0)
        hh = s.hand("L")
        cp = pose(PP.SCOOP)
        r0 = H.qrot(H.qinv(hh[3:7]), cp[:3] - hh[:3])
        q_hand_in_cup = H.qmul(H.qinv(cp[3:7]), hh[3:7])

        def cup_to(name, o, rho, sec=2.0, fix=False):
            o = np.asarray(o, float)
            q = H.qmul(_rotv([0, -1.0, 0], np.radians(rho)), q_hand_in_cup)
            result = go(name, o - H.qrot(q, r0), q, sec)
            if fix:                                       # the tool sits off its grip model by a measured amount: one correction from the measured pose
                e = o - pose(PP.SCOOP)[:3]
                result = go(name, o + e - H.qrot(q, r0), q, 0.8)
            tool_error = float(np.linalg.norm(pose(PP.SCOOP)[:3] - o))
            out['stages'][name]['tool_error_mm'] = tool_error * 1000
            if tool_error > .02:
                raise _Fail(f'scoop missed {name} by {tool_error*1000:.1f} mm; stopping before further contact')
            return result

        rim = jar[2] + PP.NECK_TOP
        if ctx.get('_gv') is not None:
            grain_xyz = ctx["_gv"].get_transforms()[:, :3]
            out["nums"]["grain_bounds_before"] = [grain_xyz.min(0).tolist(), grain_xyz.max(0).tolist()]
        oz = jar[2] + PP.FILL_TOP + PP.SC_R - 0.001
        dips = []
        for d in range(1, MAX_DIPS + 1):                  # repeat scoops, stop at the dose: each one is the same motion
            t = f"d{d}_"
            cup_to(t + "over_jar", (jar[0] + 0.01, jar[1], rim + 0.06), 0, 3.5 if d == 1 else 2.6, fix=True)
            cup_to(t + "tilt", (jar[0] + 0.018, jar[1], rim + 0.045), 75, 2.2)
            cup_to(t + "dip", (jar[0] + 0.018, jar[1], oz), 75, 2.8)
            cup_to(t + "drag", (jar[0] - 0.002, jar[1], oz), 75, 2.5)
            cup_to(t + "up1", (jar[0] - 0.002, jar[1], oz), 30, 2.8)
            cup_to(t + "up2", (jar[0], jar[1], rim + 0.01), 8, 2.5)
            cup_to(t + "out", (jar[0], jar[1], rim + 0.05), 0, 1.6)
            out["nums"]["jar_moved_mm"] = round(float(np.linalg.norm(pose(PP.JAR)[:3] - jar)) * 1000, 2)
            after_scoop = grains_counts(mug_path, jar[:2])
            print("COUNTS after scoop", d, after_scoop, "jar moved mm", out["nums"]["jar_moved_mm"], flush=True)
            m = pose(mug_path)                            # measured mug, not the spot it was meant to be at
            pz = m[2] + POUR_Z
            cup_to(t + "to_mug", (m[0] + 0.012, m[1], pz + 0.02), 0, 3.0)
            cup_to(t + "pour1", (m[0] + 0.012, m[1], pz + 0.02), 60, 2.0)
            cup_to(t + "pour2", (m[0] + 0.008, m[1], pz + 0.01), 105, 1.5)
            cup_to(t + "pour3", (m[0] + 0.003, m[1], pz), 150, 1.2, fix=True)
            for k in range(ROCKS):                        # tap the pile loose: rock the tilt, all above the mug
                cup_to(t + f"rock{k}a", (m[0] + 0.004, m[1], pz + 0.004), 125, 0.5)
                cup_to(t + f"rock{k}b", (m[0] + 0.003, m[1], pz), 150, 0.5)
            s.hold_for(0.8)
            after_pour = grains_counts(mug_path, jar[:2])
            dips.append(dict(after_scoop=after_scoop, after_pour=after_pour))
            out["nums"]["dips"] = dips
            print("COUNTS after pour", d, after_pour, "mug moved mm", round(float(np.linalg.norm(pose(mug_path)[:2] - m[:2])) * 1000, 2), flush=True)
            cup_to(t + "back", (m[0] + 0.012, m[1], pz + 0.03), 0, 2.2)
            if after_pour.get("grains_in_mug", 0) >= TARGET_GRAINS or ctx.get('_gv') is None:
                break
        cup_to("cup_home_up", (c0[0], c0[1], c0[2] + 0.05), 0, 3.0)
        cup_to("cup_home", (c0[0], c0[1], c0[2] + 0.002), 0, 2.0)
        if gt is not None:
            gt.release(PP.SCOOP)
        s.fingers("L", dict(index=0.3, middle=0.3), 1.0)
        s.hold_for(0.3)
        h = s.hand("L").copy()
        go("cup_clear", h[:3] + np.array([0, 0, 0.09]), h[3:7], 1.6)
        s.hold_for(1.0)
        out["nums"].update(grains_counts(mug_path, jar[:2]))
        n = out["nums"].get("grains_in_mug")
        lid_ok = bool(out["nums"].get("lid_on_counter", ctx.get("only") == "cup"))
        ok = lid_ok and n is not None and n > 0
        out["status"] = "PASS" if ok else "FAIL"
        out["task_status"] = "PASS" if lid_ok and n is not None and n >= TARGET_GRAINS else "FAIL"
        if not ok:
            out["why"] = "lid not on the counter" if not lid_ok else "no grains in the mug"
    except _Fail as e:
        out["why"] = str(e)
    except Exception as e:                              # never take the caller's run down with this stage
        out["why"] = "exception: " + repr(e)
    finally:
        try:
            if not s.fallen():
                if gt is not None:
                    # Stop retention measurement at the intentional release.
                    # Previously measured slip remains in the gate report.
                    gt.release(PP.SCOOP)
                    gt.release(PP.LID)
                s.fingers("L", dict(index=0.0, middle=0.0, ring=0.0, pinky=0.0, thumb_pitch=0.0), 1.0)
                s.fingers("L", dict(thumb_yaw=-0.1), 0.8)
                if sup is not None:
                    out["nums"]["rise"] = bool(sup.crouch(0.0, seconds=2.0, settle=0.6).get("passed"))
                    sup.uninstall()
                if saved_gains is not None:
                    s.kp[:], s.kd[:] = saved_gains[2], saved_gains[3]
                    a.set_dof_stiffnesses(s.f(saved_gains[0]), s.idx)
                    a.set_dof_dampings(s.f(saved_gains[1]), s.idx)
        except Exception as e:                          # report, never kill the caller's run
            out["why"] = (out["why"] or "") + f" cleanup error: {e!r}"
        out["nums"]["fallen"] = bool(s.fallen())
        if out["nums"]["fallen"]:
            out["status"] = "FAIL"
    return out


# ---- late binding: if stage_live.py exists beside this file when the step starts, use it. Same contract, same rules.
def _live(fn_name, s, ctx, fallback):
    import importlib.util
    import os
    p = os.path.join(RP.code('steps', 'powder'), "stage_live.py")
    if not os.path.exists(p):
        return fallback(s, ctx)
    sp = importlib.util.spec_from_file_location("stage_live_" + fn_name, p)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    print("[late-bind] using", p, "for", fn_name, flush=True)
    return getattr(m, fn_name)(s, ctx)


_orig_run = run


def run(s, ctx):
    return _live("run", s, ctx, _orig_run)
