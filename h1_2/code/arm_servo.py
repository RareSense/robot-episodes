"""Maintain measured-world arm goals while fingers close or the other arm moves.

Only torque-limited articulation position-drive references are changed.
"""
import numpy as np
import h12_physx as H

def configure_fingers(s,side,kp_value=10.,kd_value=.2):
    """Stiffer references around contact, with original 1 Nm motor limits."""
    kp=s.art.get_dof_stiffnesses()[0].copy();kd=s.art.get_dof_dampings()[0].copy()
    for i,name in enumerate(s.names):
        if name.startswith(side+'_') and name not in s.passive:
            kp[i]=kp_value;kd[i]=kd_value;s.kp[i]=kp_value;s.kd[i]=kd_value
    s.art.set_dof_stiffnesses(s.f(kp),s.idx);s.art.set_dof_dampings(s.f(kd),s.idx)

class ArmPoseHold:
    def __init__(self,s):
        self.s=s;self.orig=s.step;self.goals={};s.step=self.step
    def hold(self,side,p=None,q=None):
        h=self.s.hand(side);ai=self.s.arm_idx(side)
        self.goals[side]={'p':h[:3].copy() if p is None else np.array(p,float),
                          'q':h[3:].copy() if q is None else np.array(q,float),
                          'ai':ai,'li':self.s.links.index(side+'_hand_base_link'),
                          'target':self.s.hold[ai].copy(),'limits':self.s.art.get_dof_limits()[0,ai]}
    def step(self):
        s=self.s
        if s.k % H.PERIOD==0 and self.goals:
            Jall=s.art.get_jacobians()[0]
            for side,g in self.goals.items():
                h=s.hand(side); J=Jall[g['li']][:,6+g['ai']].astype(float);J[3:]*=.5
                error=np.r_[.08*np.clip(g['p']-h[:3],-.02,.02),.04*np.clip(H.qerr(g['q'],h[3:]),-.15,.15)]
                dq=J.T@np.linalg.solve(J@J.T+2e-3*np.eye(6),error)
                g['target']=np.clip(g['target']+np.clip(dq,-.012,.012),g['limits'][:,0],g['limits'][:,1])
                g['target']=s.bounded_arm_reference(g['ai'],g['target'])
                s.hold[g['ai']]=g['target']
        self.orig()
    def detach(self):self.s.step=self.orig
