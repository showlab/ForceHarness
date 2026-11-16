"""Real-robot contact debounce on top of the shared event detector (layer 2); force/torque caps stay instantaneous."""
import numpy as np

from forceharness.primitives.events import EventDetector

FREE_COLLISION_N = 8.
FIND_CONTACT_N = 5.


class RealEventDetector(EventDetector):
    def __init__(self,clock,*,f_contact=FIND_CONTACT_N,f_cap=25.,t_cap=3.):
        self.clock=clock
        super().__init__(f_contact=f_contact,f_release=f_contact*.5,
                         f_cap=f_cap,t_cap=t_cap,min_samples=3,filter_n=1)

    def reset(self):
        super().reset()
        self.started_at=self.clock()
        self.contact_sample=None

    def update(self,t,reaction,torque,pos,commanded_step,actual_step,grip_closed,
               suppress_contact=False):
        suppressed=suppress_contact or t-self.started_at < .3-1e-9
        # The base detector retains its counter in the hysteresis band; on the
        # real arm a contact needs three genuinely consecutive readings >= threshold.
        if suppressed or np.linalg.norm(reaction)<self.f_contact:
            self._above=0
        events=super().update(t,reaction,torque,pos,commanded_step,actual_step,
                              grip_closed,suppress_contact=suppressed)
        if suppressed:
            self._above=0
        if any(e.kind=='contact_start' for e in events):
            self.contact_sample=dict(t=float(t),pos=np.asarray(pos).tolist(),
                                     force=np.asarray(reaction).tolist())
        return events
