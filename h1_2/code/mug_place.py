"""Place a held mug using its measured bottom pose and a contact-aware descent.

Only joint references change. No object pose writes, attachments or forces.
"""
import numpy as np
import h12_physx as H
from arm_servo import ArmPoseHold

def stand_balance(s, base, kp=1.2, limit=.1):
    """Standing balance for any measured stance. The stock balance has no lateral
    position term (gait.json ky=0) and leans on the default stance's width; on a
    narrower true stance the robot tips. Adds CrouchSupport's lateral gain, leg
    joint references only. Clears velocity memory left stale during a crouch."""
    s.com_prev=None; s.comv[:]=0.
    hip,ankle=H.G['hip_roll_idx'],H.G['ankle_roll_idx']
    def balance():
        base()
        roll=float(np.clip(kp*s.e_com[1],-limit,limit))
        for i in hip:s.off[i]+=roll
        for i in ankle:s.off[i]-=roll
    return balance

def run(s, gates, spot=(1.90,2.17), top=.86, name='object_0316', unload_s=1.5, on_contact=None):
    out={'status':'FAIL','stages':{},'reason':''}
    spot=np.asarray(spot,float)
    def tilt():
        z=H.qrot(s.obj(name)[3:],np.array([0.,0.,1.]))
        return float(np.degrees(np.arccos(np.clip(z[2],-1.,1.))))
    def lowest():
        return s.obj(name)[2]-.034*np.sin(np.radians(tilt()))
    def mark(label):
        out['stages'][label]={'t':s.t,'mug':s.obj(name).tolist(),
                             'hand':s.hand('L').tolist(),'tilt_deg':tilt()}
        print('PLACE',label,out['stages'][label],flush=True)
        if s.fallen():raise RuntimeError('robot fell during placement')
    def level_q():
        h=s.hand('L');m=s.obj(name)
        z=H.qrot(m[3:],np.array([0.,0.,1.]));axis=np.cross(z,[0.,0.,1.])
        sn=np.linalg.norm(axis)
        if sn<1e-7:return h[3:]
        angle=min(np.arctan2(sn,z[2]),np.radians(3.))
        rot=np.r_[axis/sn*np.sin(angle/2),np.cos(angle/2)]
        return H.qmul(rot,h[3:])
    holder=None
    try:
        # Correct actual tool position while still clear of the counter.
        for _ in range(5):
            m=s.obj(name);h=s.hand('L')
            delta=np.r_[spot-m[:2],top+.060-m[2]]
            s.ik_to(h[:3]+delta,level_q(),max(1.4,float(np.linalg.norm(delta))/.05),
                    settle=30,gain=.08,w_rot=.8,
                    stop=lambda:lowest()<top+.040)
            if np.linalg.norm(s.obj(name)[:2]-spot)<.002 and tilt()<1.:break
        mark('aligned_above_counter')
        lower_q=s.hand('L')[3:].copy()
        # Near its stop the wrist cannot hold the hand's angle: the rotation term then
        # drags the elbow down at about 3x the commanded speed (m_cpu01: 36 mm/s, 6 deg
        # at contact). So near the stop, stop levelling and let position lead.
        wj=s.names.index('left_wrist_pitch_joint')
        wlo,whi=[float(x) for x in s.art.get_dof_limits()[0,wj]]
        for _ in range(90):
            clearance=lowest()-top
            if clearance<.0015:break
            h=s.hand('L');m=s.obj(name)
            wq=float(s.art.get_dof_positions()[0,wj])
            near=min(wq-wlo,whi-wq)<.04
            out['wrist_near_stop']=bool(out.get('wrist_near_stop') or near)
            if near:
                lower_q=h[3:].copy()
            elif clearance>.015 and tilt()>1.:
                lower_q=level_q()
            dz=-min(.002,max(.0003,clearance-.001))
            delta=np.r_[np.clip(spot-m[:2],-.001,.001),dz]
            s.ik_to(h[:3]+delta,lower_q,.16,settle=3,gain=.08,w_rot=.25 if near else 1.,
                    stop=lambda:lowest()<top+.0015)
            if s.fallen():raise RuntimeError('fell during mug descent')
        if lowest()>top+.003:
            raise RuntimeError('mug did not reach the counter')
        mark('counter_contact')
        if on_contact:on_contact()
        holder=ArmPoseHold(s);holder.hold('L')
        if gates:gates.release(name)
        # Balanced unload: each preload (reference minus measured finger angle) falls
        # to zero over the same ramp, so no finger outlasts the others and drags the
        # mug (it slides at about 0.9 N). Only then do the fingers open.
        before=s.obj(name).copy()
        keys=('index','middle','ring','pinky','thumb_pitch')
        s.fingers('L',{k:s.finger_q('L',k) for k in keys},unload_s)
        s.hold_for(.2)
        out['unload_shift_mm']=float(np.linalg.norm(s.obj(name)[:2]-before[:2])*1000)
        mark('unloaded')
        s.fingers('L',dict(index=0.,middle=0.,ring=0.,pinky=0.,thumb_pitch=0.),1.5)
        s.hold_for(.3)
        s.fingers('L',dict(thumb_yaw=-.1),.8)
        s.hold_for(.3)
        mark('released')
        holder.detach();holder=None
        h=s.hand('L')
        s.ik_to(h[:3]+[-.05,0,0],h[3:],1.2,gain=.08,settle=20)
        h=s.hand('L')
        s.ik_to(h[:3]+[0,-.20,.08],h[3:],1.8,gain=.08,settle=20)
        s.hold_for(.5);mark('hand_clear')
        m=s.obj(name)
        out.update(place_err_mm=float(np.linalg.norm(m[:2]-spot)*1000),
                   tilt_deg=tilt(),on_counter=bool(abs(m[2]-top)<.004))
        out['status']='PASS' if out['place_err_mm']<=5 and out['tilt_deg']<5 and out['on_counter'] else 'FAIL'
        if out['status']!='PASS':out['reason']='placement pose outside tolerance'
    except Exception as exc:
        out['reason']=repr(exc)
    finally:
        if holder:holder.detach()
    return out
