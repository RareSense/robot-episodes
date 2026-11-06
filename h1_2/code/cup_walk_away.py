"""Last step of the tea episode: pick up the prepared mug and carry it to the exit.

Contact-only grip with the rack grasp recipe (mug_episode.py), URDF torque limits, joint references only.
No object or base pose writes, no attachments, no external forces. Every goal comes from measured world poses.

run(s, ctx) ends standing in the exit area with the left hand still holding the mug, the one step that does
not end with hands free. ctx: gates, water (fluids.Fluids or None), exit_xy (optional).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))    # release_paths.py lives at the code root
import release_paths as RP
import math
import sys

import numpy as np

import h12_physx as H
from arm_servo import ArmPoseHold, configure_fingers

sys.path.insert(0, RP.code("balance_support"))
from crouch_support import CrouchSupport

MUG = 'object_0316'
Q = np.array([0., 0., 1., 0.])            # hand frame of the rack grasp: palm toward +x, fingers toward +y
TX, TY, GH = .047, .165, .054             # hand origin from the mug axis. Rack height .048 + 6 mm: the open pinky tip clears the counter by only 1 cm
SQ, LOW, TSQ = .09, .7, .07               # squeeze at finger gain 10: the rack's torque (.3 and .15 at gain 3) with 3x the stiffness
FINGER_KP = 10.                           # c_cpu08 vs c_cpu09: at .33 kg gain 3 lets the mug pivot out at 0.7 deg/s; gain 10 holds it, 1 Nm caps unchanged
STANCE = np.array([.13, -.42])            # pelvis from the mug: same hand-to-pelvis offset as the rack grasp
DROP = .125                               # crouch for the 0.86 m counter: at .09 the wrist runs out below hand z .925 and pitches the pinky into the counter
CARRY_FWD, CARRY_LEFT, CARRY_UP = .095, .16, .10   # hand from the pelvis in the carry that held in live01
EXIT_XY = (.75, 1.25)
VMAX, WMAX, ACC = .20, .35, .004          # carry walk: m/s, rad/s, command change per 20 ms tick
EZ = np.array([0., 0., 1.])


class Failed(RuntimeError):
    pass


class CarryStabilizer:
    """Track an upright hand frame relative to the moving pelvis yaw. full_episode.py uses this default form for the empty
    mug. With level=MUG the hand goal is also turned by the measured mug axis in the hand, because the mug creeps in the
    grip (live01: hand within 2 deg while the mug axis drifted 6 deg in the hand)."""

    def __init__(self, s, level=None):
        self.s = s
        self.orig = s.step
        self.ai = s.arm_idx('L')
        self.li = s.links.index('L_hand_base_link')
        self.level, self.axis = level, None
        self.retarget()
        self.qd = s.hold[self.ai].copy()
        self.limits = s.art.get_dof_limits()[0, self.ai]
        s.step = self.step

    def _yaw_q(self):
        p, q = self.s.pose()
        yaw = H.yaw_of(q)
        return p, np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)])

    def retarget(self, offset=None):
        """Hold the present hand pose relative to the pelvis, or `offset` (pelvis yaw frame) if given."""
        p, yq = self._yaw_q()
        h = self.s.hand('L')
        self.offset = H.qrot(H.qinv(yq), h[:3] - p) if offset is None else np.asarray(offset, float)
        self.qrel = H.qmul(H.qinv(yq), h[3:])

    def _goal_q(self, yq, h):
        g = H.qmul(yq, self.qrel)
        if self.level is None:
            return g
        a = H.qrot(H.qinv(h[3:]), H.qrot(self.s.obj(self.level)[3:], EZ))
        self.axis = a if self.axis is None else .98 * self.axis + .02 * a      # about 1 s: follows the creep, not the gait
        w = H.qrot(g, self.axis / np.linalg.norm(self.axis))
        ax = np.cross(w, EZ)
        sn = float(np.linalg.norm(ax))
        if sn < 1e-6:
            return g
        ang = math.atan2(sn, float(w @ EZ))
        return H.qmul(np.r_[ax / sn * math.sin(ang / 2), math.cos(ang / 2)], g)

    def step(self):
        s = self.s
        if s.k % H.PERIOD == 0:
            p, yq = self._yaw_q()
            h = s.hand('L')
            goalp = p + H.qrot(yq, self.offset)
            goalq = self._goal_q(yq, h)
            J = s.art.get_jacobians()[0, self.li][:, 6 + self.ai].astype(float)
            J[3:] *= .5
            error = np.r_[.08 * np.clip(goalp - h[:3], -.02, .02), .04 * np.clip(H.qerr(goalq, h[3:]), -.15, .15)]
            dq = J.T @ np.linalg.solve(J @ J.T + 2e-3 * np.eye(6), error)
            q = s.art.get_dof_positions()[0, self.ai].astype(float)      # a joint at its stop drops out, so the others take over: c2_cpu01 sat at wrist pitch -.44, 7 deg short for 8 s
            stuck = ((q < self.limits[:, 0] + .03) & (dq < 0)) | ((q > self.limits[:, 1] - .03) & (dq > 0))
            if stuck.any():
                J[:, stuck] = 0.
                dq = J.T @ np.linalg.solve(J @ J.T + 2e-3 * np.eye(6), error)
            self.qd = np.clip(self.qd + np.clip(dq, -.012, .012), self.limits[:, 0], self.limits[:, 1])
            self.qd = s.bounded_arm_reference(self.ai, self.qd)
            s.hold[self.ai] = self.qd
        self.orig()

    def detach(self):
        self.s.step = self.orig


def _to_walk(s):
    """Stance back to the walking network: fresh memory, walking leg gains, zero command warm-up (mug_episode.to_walk)."""
    a = s.art
    s.h[:], s.c[:], s.last[:] = 0, 0, 0
    s.target = a.get_dof_positions()[0].astype(float)[s.legs].copy()
    kp = a.get_dof_stiffnesses()[0].astype(float).copy()
    kd = a.get_dof_dampings()[0].astype(float).copy()
    kp[s.legs], kd[s.legs] = H.G['kp'], H.G['kd']
    a.set_dof_stiffnesses(s.f(kp), s.idx)
    a.set_dof_dampings(s.f(kd), s.idx)
    s.want[:], s.cmd[:] = 0, 0
    s.mode = 'walk'


def _walk(s, x, y, yaw_deg=None, vmax=VMAX, wmax=WMAX, tol=.06, timeout=45., tick=None):
    """Walk with a limited command rate. Turns first, then goes straight. yaw_deg None keeps the travel heading."""
    t0 = s.t
    gy = None if yaw_deg is None else math.radians(yaw_deg)
    keep = None
    while s.t - t0 < timeout:
        p, q = s.pose()
        yaw = H.yaw_of(q)
        e = np.array([x, y]) - p[:2]
        dist = float(np.linalg.norm(e))
        if gy is None and dist > .35:
            keep = math.atan2(e[1], e[0])
        goal = gy if gy is not None else (keep if keep is not None else yaw)    # no spin on arrival
        if dist < tol and abs(H.wrap(goal - yaw)) < math.radians(8):
            return True
        c, sn = math.cos(yaw), math.sin(yaw)
        ex, ey = c * e[0] + sn * e[1], -sn * e[0] + c * e[1]
        head = math.atan2(e[1], e[0]) if dist > .35 else goal
        yerr = H.wrap(head - yaw)
        go = float(np.clip(1 - (abs(yerr) - .3) / .5, 0, 1))
        vx, vy = H.min_cmd(ex, 1.2, vmax, tol) * go, H.min_cmd(ey, 1.2, .14, tol) * go
        want = np.array([vx if abs(vx) >= .1 else 0., vy if abs(vy) >= .1 else 0., float(np.clip(1.2 * yerr, -wmax, wmax))])
        s.want = s.want + np.clip(want - s.want, -ACC, ACC)
        for _ in range(H.PERIOD):
            s.step()
        if s.fallen():
            raise Failed('robot fell while walking')
        if tick:
            tick()
    return False


def _stop(s):
    """Let the speed die out, wait for planted feet, then blend slowly to the stance (mug_episode gentle stop)."""
    while np.abs(s.want).max() > 1e-3:
        s.want = s.want + np.clip(-s.want, -ACC, ACC)
        for _ in range(H.PERIOD):
            s.step()
    s.want = np.zeros(3)
    fz = []
    for _ in range(900):
        s.step()
        fz.append(s.feet_xy()[:, 2].copy())
        if len(fz) > 480 and np.all(fz[-1] - np.min(fz[-120:], axis=0) < .004):
            break
    s.to_stand(1.5)
    s.stand_goal = s.art.get_dof_positions()[0].astype(float)[s.legs].copy()    # stay on the measured stance: the blend to the default pose pitched the body 3 deg and the mug 10 (c2_cpu01)
    s.hold_for(2.2)


def _plant(s, x, y, yaw_deg, tol=.025):
    """Empty-handed stance: keep correcting until the feet are planted, then stand (mug_episode.walk_to)."""
    gy = math.radians(yaw_deg)
    fz = []
    for n in range(240):
        p, q = s.pose()
        yw = H.yaw_of(q)
        e = np.array([x, y]) - p[:2]
        c, sn = math.cos(yw), math.sin(yw)
        s.want = np.array([H.min_cmd(c * e[0] + sn * e[1], 1., .14, tol), H.min_cmd(-sn * e[0] + c * e[1], 1., .14, tol), float(np.clip(1.2 * H.wrap(gy - yw), -.3, .3))])
        for _ in range(H.PERIOD):
            s.step()
            z = s.feet_xy()[:, 2]
            fz.append(z.copy())
        if n >= 60 and float(np.linalg.norm(e)) < .02 and np.all(z - np.min(fz[-120:], axis=0) < .004):
            break
    s.want = np.zeros(3)
    s.to_stand(1.5)                                   # .6 s spiked the ankle (1117 Nm) and the foot then drifted 9 mm in the crouch (c2_cpu05); crouching straight from the walking pose leaned 6 deg (c2_cpu06)
    s.hold_for(2.2)
    s.stand_from = s.stand_goal.copy()                # then references onto the measured stance, as full_episode does before its set-down crouch
    s.stand_goal = s.art.get_dof_positions()[0].astype(float)[s.legs].copy()
    s.stand_ramp, s.stand_t = 1., 0.
    s.hold_for(1.2)


def _links_of(key):
    if key == 'thumb_pitch':
        return ('L_thumb_proximal', 'L_thumb_intermediate', 'L_thumb_distal')
    return (f'L_{key}_proximal', f'L_{key}_intermediate', f'L_{key}_distal')


def _close(s, keys, rate=.5, thresh=.4, top=1.4, timeout=4.):
    """Creep each finger to measured contact (mug_episode.close)."""
    cur = {k: s.hold[s.finger_idx('L', k)[0][0]] for k in keys}
    done, t = {}, 0.
    while len(done) < len(keys) and t < timeout:
        c = s.contact(MUG)
        for k in keys:
            if k in done:
                continue
            if any(c.get(n, 0.) > thresh for n in _links_of(k)):
                done[k] = cur[k]
            else:
                cur[k] = min(top, cur[k] + rate * .02)
        s.fingers('L', {k: cur[k] for k in keys}, .02)
        t += .02
    return cur


def run(s, ctx):
    G, W = ctx.get('gates'), ctx.get('water')
    exit_xy = ctx.get('exit_xy', EXIT_XY)
    sq, low, tsq, fkp = ctx.get('sq', SQ), ctx.get('low', LOW), ctx.get('tsq', TSQ), ctx.get('finger_kp', FINGER_KP)
    # task_status: carried to the exit upright, not dropped, spill 3 ml or less. status and strict are reported beside it, unchanged.
    out = {'status': 'FAIL', 'task_status': 'FAIL', 'task_reason': 'step did not finish', 'strict': None, 'stages': {}, 'reason': '', 'exit_xy': list(exit_xy)}
    sup = stab = rel0 = holder = None
    peak = {'tilt_carry_deg': 0., 'slip_mm': 0.}
    last = {'water_t': -1.}
    initial = s.obj(MUG).copy()

    def tilt():
        return float(np.degrees(np.arccos(np.clip(H.qrot(s.obj(MUG)[3:], EZ)[2], -1, 1))))

    def rel():
        h = s.hand('L')
        return H.qrot(H.qinv(h[3:]), s.obj(MUG)[:3] - h[:3])

    def water():
        if W is None:
            return None
        W.sample(s.step)
        return W.report()

    def mark(name, err=None):
        slip = None if rel0 is None else float(np.linalg.norm(rel() - rel0)) * 1000
        if slip is not None:
            peak['slip_mm'] = max(peak['slip_mm'], slip)
        row = {'t': round(s.t, 2), 'err_mm': None if err is None else round(err * 1000, 1), 'mug': np.round(s.obj(MUG)[:3], 3).tolist(),
               'hand': np.round(s.hand('L')[:3], 3).tolist(), 'pelvis': np.round(s.pose()[0], 3).tolist(), 'tilt_deg': round(tilt(), 2),
               'touch': s.contact(MUG), 'slip_mm': None if slip is None else round(slip, 1)}
        w = water()
        if w is not None:
            row['water'] = {k: w.get(k) for k in ('mug_ml', 'spill_ml', 'other_ml', 'level_mm')}
        out['stages'][name] = row
        print('CUP', name, row, flush=True)
        if s.fallen():
            raise Failed('robot fell at ' + name)
        return w

    def tick():
        """Carry watch: peak tilt every control tick, loss of the mug, water once a second."""
        peak['tilt_carry_deg'] = max(peak['tilt_carry_deg'], tilt())
        if float(np.linalg.norm(rel() - rel0)) > .03:
            raise Failed('mug slid %.0f mm in the hand' % (1000 * float(np.linalg.norm(rel() - rel0))))
        if s.t - last['water_t'] >= 1.:
            last['water_t'] = s.t
            w = water()
            if w is not None:
                print('CUP carry t', round(s.t, 2), 'tilt', round(tilt(), 2), 'water', w.get('mug_ml'), w.get('spill_ml'), w.get('other_ml'), flush=True)
            else:
                print('CUP carry t', round(s.t, 2), 'tilt', round(tilt(), 2), 'pelvis', np.round(s.pose()[0], 2).tolist(), flush=True)

    def move(p, q=Q, seconds=2., limit=.015, **kw):
        e = s.ik_to(np.asarray(p, float), q, seconds, **kw)
        if e > limit:
            raise Failed('hand missed by %.1f mm' % (1000 * e))
        return e

    try:
        w0 = mark('start')
        out['mug_ml_start'] = None if w0 is None else w0.get('mug_ml')

        # ---- stance beside the mug, from its measured pose
        stance = initial[:2] + STANCE
        if float(np.linalg.norm(s.pose()[0][:2] - stance)) > .03 or s.mode != 'stand':
            _to_walk(s)
            for _ in range(300):
                s.step()
            if not s.go(stance[0], stance[1], 90., tol=.025, timeout=25., log=lambda *z, **k: None) and float(np.linalg.norm(s.pose()[0][:2] - stance)) > .08:
                raise Failed('walk to the mug timed out')
            _plant(s, stance[0], stance[1], 90.)
        mark('at_stance')

        # ---- rack grasp, placed from the measured mug pose
        sup = CrouchSupport(s)
        c = sup.crouch(DROP, seconds=4., settle=2.5)     # place06: this depth passes only with the slow timing
        out['crouch'] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()}
        if not c['passed']:
            raise Failed('crouch failed: %s' % out['crouch'])
        m = s.obj(MUG)
        hand_at = np.array([m[0] - TX, m[1] - TY, m[2] + GH])
        beside = hand_at + [-.02, 0., 0.]
        s.fingers('L', dict(index=0., middle=0., ring=0., pinky=0., thumb_yaw=-.1, thumb_pitch=0.), .3)
        mark('side', move(hand_at + [-.073, -.10, .077], seconds=1.8))
        high = beside + [0., 0., .03]                 # the open pinky would drag on the counter if the hand tilts at grip height
        mark('front', move(high + [0., -.07, 0.], seconds=1.4, w_rot=1.))
        ang = float(np.degrees(np.linalg.norm(H.qerr(Q, s.hand('L')[3:7]))))
        if ang > 3.:
            raise Failed('hand %.1f deg off before the approach' % ang)
        mark('above', move(high, seconds=1.8, w_rot=1.))
        mark('beside', move(beside, seconds=1.4, w_rot=1.))
        s.ik_to(hand_at, Q, 1.5, stop=lambda: sum(s.contact(MUG).values()) > .2)
        mark('palm')
        ai = s.arm_idx('L')
        s.hold[ai] = s.art.get_dof_positions()[0].astype(float)[ai]    # no stored push: at the rack the mug could not slide, here it can
        holder = ArmPoseHold(s)
        holder.hold('L')
        if G:
            G.allow_move(MUG)                         # the grasp itself seats the mug about 14 mm deeper in the hand
        configure_fingers(s, 'L', 3., .05)            # earlier steps may leave other gains: the close below is tuned at the default
        ty = s.finger_idx('L', 'thumb_yaw')[0][0]
        kp = s.art.get_dof_stiffnesses()[0].astype(float).copy()
        kp[ty] = 10.
        s.kp[ty] = 10.                                # to_stand() restores s.kp
        s.art.set_dof_stiffnesses(s.f(kp), s.idx)
        s.fingers('L', dict(thumb_yaw=1.3), 1.2)
        s.hold_for(.6)
        thumb = _close(s, ['thumb_pitch'], rate=.25, top=.55)
        fingers = _close(s, ['index', 'middle', 'ring', 'pinky'], rate=.5)
        print('CUP closed at', {k: round(float(v), 3) for k, v in {**thumb, **fingers}.items()}, s.contact(MUG), flush=True)
        if fkp:
            configure_fingers(s, 'L', fkp, .2)        # stiffer references, same 1 Nm caps: the squeeze below is scaled to the same torque
        tgt = {k: fingers[k] + (sq if k in ('index', 'middle') else low * sq) for k in fingers}
        tgt['thumb_pitch'] = thumb['thumb_pitch'] + tsq
        s.fingers('L', tgt, .6)
        s.hold_for(.4)
        holder.detach()
        holder = None
        touch = s.contact(MUG)
        if not any('thumb' in k for k in touch) or not any('index' in k or 'middle' in k for k in touch):
            raise Failed('no opposing grip: %s' % touch)
        rel0 = rel()
        if G:
            G.hold(MUG, 'L_hand_base_link')
        mark('gripped')

        # ---- lift, draw in to the carry pose, rise
        h = s.hand('L')[:3]
        mark('lifted', move(h + [0., 0., .07], seconds=3.2))
        if s.obj(MUG)[2] < initial[2] + .04:
            raise Failed('mug did not lift')
        p, q = s.pose()
        yaw = H.yaw_of(q)
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.])
        h = s.hand('L')[:3]
        back = float(np.clip((h[:2] - p[:2]) @ fwd[:2] - CARRY_FWD, 0., .2))
        mark('drawn_in', move(h - back * fwd, seconds=max(2., back / .05)))
        stab = CarryStabilizer(s, level=MUG)
        c = sup.crouch(0., seconds=3.5, settle=1.)
        sup.uninstall()
        sup = None
        if not c['passed']:
            raise Failed('rise with the mug failed')
        mark('risen')
        if s.obj(MUG)[2] < initial[2] + .04 or float(np.linalg.norm(rel() - rel0)) > .03:
            raise Failed('mug lost before the walk')

        # ---- carry
        _to_walk(s)
        for _ in range(300):
            s.step()
        mark('walk_start')
        if not _walk(s, exit_xy[0], exit_xy[1], None, tol=.08, tick=tick):
            raise Failed('carry to the exit timed out')
        mark('at_exit_moving')
        _stop(s)
        tick()
        mark('stopped')
        settled, t_end = 0, s.t + 8.                  # stabilizer stays on: the stop leans the mug to 4-7 deg and it was cut off 2 s in (c_cpu09 ended at 5.6).
        while s.t < t_end and settled < 10:           # c_cpu10: a one-shot ik_to on the mug axis instead ran the arm into the wrist stop, 27 deg
            s.hold_for(.1)
            settled = settled + 1 if tilt() < 2. else 0
        tick()
        mark('levelled')
        stab.detach()
        stab = None
        s.hold_for(1.)
        w1 = mark('walked_away_with_cup')

        # ---- result
        if G:
            peak['slip_mm'] = max(peak['slip_mm'], G.slip.get(MUG, {}).get('mm', 0.))
        pel = s.pose()[0]
        held = bool(sum(s.contact(MUG).values()) > .2 and s.obj(MUG)[2] > .9 and float(np.linalg.norm(rel() - rel0)) < .03)
        out.update(slip_mm=round(peak['slip_mm'], 2), tilt_peak_carry_deg=round(peak['tilt_carry_deg'], 2), tilt_deg=round(tilt(), 2), held=held,
                   exit_err_m=round(float(np.linalg.norm(pel[:2] - np.asarray(exit_xy))), 3), standing=not s.fallen(), mug=s.obj(MUG).tolist())
        ok = held and out['standing'] and out['tilt_deg'] < 5. and out['exit_err_m'] < .2
        if w1 is not None:
            out['mug_ml_end'] = w1.get('mug_ml')
            out['carry_loss_ml'] = round(float(out['mug_ml_start'] - w1.get('mug_ml')), 1)
            out['spill_ml_scene'] = w1.get('spill_ml')
            ok = ok and out['carry_loss_ml'] <= 3.
        else:
            out['water_note'] = 'no water in this scene: spill not measured'
        out['status'] = 'PASS' if ok else 'FAIL'
        if not ok:
            out['reason'] = 'end check failed: held, upright, at exit or spill'
        why = [n for n, good in (('dropped', held), ('fell', out['standing']), ('not upright', out['tilt_deg'] < 5.), ('not at exit', out['exit_err_m'] < .2),
                                 ('spill not measured', w1 is not None), ('spill over 3 ml', w1 is None or out['carry_loss_ml'] <= 3.)) if not good]
        out['task_status'] = 'FAIL' if why else 'PASS'
        out['task_reason'] = ', '.join(why)
    except Exception as e:
        out['reason'] = repr(e)
        out['task_reason'] = repr(e)
        out['tilt_deg'] = round(tilt(), 2)
        out['slip_mm'] = round(peak['slip_mm'], 2)
        out['tilt_peak_carry_deg'] = round(peak['tilt_carry_deg'], 2)
    finally:
        if holder:
            holder.detach()
        if stab:
            stab.detach()
        if sup:
            sup.uninstall()
        if G:
            try:
                j = G.judge()['gates']
                out['strict'] = {k: {'status': j[k]['status'], 'value': j[k]['value']} for k in j}
                out['strict']['torque']['motor_estimate'] = j['torque'].get('motor_estimate_ratio')
            except Exception as e:
                out['strict'] = 'not read: ' + repr(e)
    return out
