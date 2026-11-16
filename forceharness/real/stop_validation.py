"""Stop check after every primitive (layer 2): distinguish a reverse rebound from unintended motion.

When the controller releases its target at the end of a primitive, the hand may spring back a few
millimetres against the direction it was moving. That rebound is expected; motion forward, sideways or
farther than the tolerance is not, and ends the episode. All distances are from the last active position,
including the normal target release. No changes to controller, gains, force guards or target generation.
"""
import numpy as np


def classify_stop(displacements,motion_direction,*,normal_soft_stop):
    d=np.asarray(displacements,dtype=float).reshape(-1,3)
    direction=np.asarray(motion_direction,dtype=float)
    if direction.shape!=(3,) or not np.isfinite(d).all() or not np.isfinite(direction).all():
        raise ValueError('Finite 3-D stop positions and preceding motion required')
    length=float(np.linalg.norm(direction))
    u=direction/length if length>1e-9 else np.zeros(3)
    distance=np.linalg.norm(d,axis=1)
    along=d@u
    reverse=bool(normal_soft_stop and length>1e-9)
    # Opposite half-space, without inventing an angular acceptance threshold.
    opposite=(along<0) & reverse
    upward=bool(u[2]>0 and u[2]>=np.linalg.norm(u[:2]))
    tolerance=np.where(opposite & upward,.006,.005)
    # Remove only the reverse component: forward and lateral motion retain
    # the 5 mm safety boundary even when reverse motion is also present.
    other=d-np.minimum(along,0)[:,None]*u if reverse else d
    other_distance=np.linalg.norm(other,axis=1)
    unsafe=(distance>.008+1e-12)|(other_distance>.005+1e-12)
    return dict(protocol='r9',normal_soft_stop=bool(normal_soft_stop),
                preceding_motion_direction=u.tolist(),upward_motion=upward,
                max_displacement_mm=float(distance.max(initial=0)*1000),
                max_other_direction_mm=float(other_distance.max(initial=0)*1000),
                rebound_tolerance_mm=6. if upward and reverse else 5.,
                rebound_passed=bool(np.all(distance<=tolerance+1e-12) and not unsafe.any()),
                safety_event=bool(unsafe.any()),
                classification='unexpected_motion' if unsafe.any() else
                    'rebound_over_tolerance' if np.any(distance>tolerance+1e-12) else
                    'within_stop_tolerance')


class UnexpectedMotion(RuntimeError):
    def __init__(self,check):
        self.check=check
        super().__init__('r9 unexpected motion: '+str(check))


def path_deviation(position,start,goal):
    """Distance to the planned Cartesian segment, not impedance target lead."""
    p,a,c=(np.asarray(v,dtype=float) for v in (position,start,goal))
    delta=c-a
    t=np.clip(float((p-a)@delta)/max(float(delta@delta),1e-20),0.,1.)
    return float(np.linalg.norm(p-(a+t*delta)))
