"""Bounded position-target compensation for free motion on the real arm (layer 2).

Joint friction can hold the hand a few millimetres short of a free-motion goal; the stock controller is
not changed, so the workstation adds a small, bounded lead to the target instead.
"""
import numpy as np


def clipped(value,limit):
    value=np.asarray(value,dtype=float)
    return value*min(1.,limit/max(float(np.linalg.norm(value)),1e-12))


class StallCompensatedTarget:
    """Add up to 4 mm/s of lead only after 0.3 s without 0.2 mm progress.

    The returned target stays within 8 mm of measured position. Once within
    the internal 1 mm goal, retract the added offset at 8 mm/s. The robot side separately
    enforces target speed, flange lead, force, workspace and timing bounds.
    """
    def __init__(self,goal,position,*,tolerance=.001,lead=.008):
        self.goal=np.asarray(goal,dtype=float).copy()
        self.anchor=np.asarray(position,dtype=float).copy()
        if (self.goal.shape!=(3,) or self.anchor.shape!=(3,)
            or not np.isfinite(self.goal).all() or not np.isfinite(self.anchor).all()):
            raise ValueError('Target and position require three finite values')
        if not 0<tolerance<=.003 or not 0<lead<=.008:
            raise ValueError('Invalid task tolerance or lead')
        self.tolerance,self.lead=tolerance,lead
        self.bias=np.zeros(3)
        self.stationary_s=0.
        self.peak_bias_m=0.
        self.at_cap_s=0.
        self.stalled_short=False

    def update(self,position,nominal,dt,*,enabled=True,contact=False):
        position,nominal=np.asarray(position,dtype=float),np.asarray(nominal,dtype=float)
        if (position.shape!=(3,) or nominal.shape!=(3,) or not np.isfinite(position).all()
            or not np.isfinite(nominal).all() or not np.isfinite(dt) or not 0<dt<=.15):
            raise ValueError('Invalid target compensation sample')
        if contact:
            self.bias[:]=0.
            self.stationary_s=0.
            self.at_cap_s=0.
            self.stalled_short=False
            return position.copy()
        progressed=np.linalg.norm(position-self.anchor)>=.0002
        if progressed:
            self.anchor=position.copy()
            self.stationary_s=0.
        else:
            self.stationary_s+=dt
        error=self.goal-position
        if np.linalg.norm(error)<=self.tolerance:
            self.bias-=clipped(self.bias,.008*dt)
        elif enabled and self.stationary_s>=.3:
            self.bias=clipped(self.bias+clipped(error,.004*dt),self.lead)
        target=position+clipped(nominal+self.bias-position,self.lead)
        # Anti-windup: remember only the offset that survives the lead clamp.
        self.bias=clipped(target-nominal,self.lead)
        self.peak_bias_m=max(self.peak_bias_m,float(np.linalg.norm(self.bias)))
        if (enabled and not progressed and np.linalg.norm(error)>self.tolerance
            and np.linalg.norm(target-position)>=self.lead-1e-7):
            self.at_cap_s+=dt
        else:
            self.at_cap_s=0.
        self.stalled_short=self.at_cap_s>=1.-1e-9
        return target

    @property
    def retracted(self):
        return np.linalg.norm(self.bias)<1e-6
