"""A pressing slide/push may lead the flange by up to 25 mm once the force has risen 3 N; free motion keeps 8 mm."""
import numpy as np
import pytest

from forceharness.real.config import BackendConfig
from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real.plant import DemoPort
from forceharness.real.primitives import CONTACT_LEAD_M


def session():
    p = DemoPort(BackendConfig(motion_enabled=True,f_cap=25.))
    p.driver.frozen = True                     # a blocked hand: the target can only lead it
    p.session.begin(30.)
    return p,p.session


def steps(p,s,n,lead=None):
    for _ in range(n):
        s.command([.03,0,0],[0,0,0],[0,0,0],[0,0,0,1],lead)
        p.clock.sleep(.05);s.tick()


def press(p,newtons):
    p.driver.override = {'wrench':(p.driver.bias+np.array([0,0,newtons,0,0,0])).tolist()}


def test_limits_default_and_cannot_be_raised():
    from forceharness.real.plant import core
    assert core.Limits().contact_lead==CONTACT_LEAD_M==.025 and core.Limits().lead==.008
    with pytest.raises(ValueError,match='Cannot increase'):
        core.Limits(contact_lead=.03)
    with pytest.raises(ValueError,match='at least lead'):
        core.Limits(contact_lead=.005)


def test_contact_lead_needs_a_force_rise_on_the_nuc():
    p,s = session()
    x0 = p.driver.p[0]
    steps(p,s,30,lead=.015)                    # asked, but nothing pressed yet
    assert s.desired[0]-x0==pytest.approx(.008,abs=1e-6)
    press(p,2.9)
    steps(p,s,5,lead=.015)
    assert s.desired[0]-x0==pytest.approx(.008,abs=1e-6)
    press(p,4.)
    steps(p,s,25,lead=.015)
    assert s.desired[0]-x0==pytest.approx(.015,abs=1e-6)
    assert s.fault is None and s.lead_now==pytest.approx(.015)


def test_granted_lead_shrinks_at_target_speed_then_release_converges():
    p,s = session()
    x0 = p.driver.p[0]
    press(p,6.)
    steps(p,s,45,lead=.015)
    p.driver.override = {}                     # contact lost: the grant ends
    steps(p,s,1)
    assert .0146-1e-9<=s.desired[0]-x0<.015       # at most 0.4 mm per step, no jump to 8 mm
    steps(p,s,20)
    assert s.desired[0]-x0==pytest.approx(.008,abs=1e-6) and s.lead_now==pytest.approx(.008)
    steps(p,s,25)                              # target still for 1 s: the frozen fixture hand is not a stall
    press(p,6.)
    steps(p,s,20,lead=.015)
    s.finish(sleep=p.clock.sleep)              # 15 mm back at 8 mm/s, inside the 2.5 s release
    assert s.fault is None and not s.active
    assert np.linalg.norm(s.desired[:3]-p.driver.p)<=.0001


def test_tracking_slack_follows_the_granted_lead():
    p,s = session()
    press(p,6.)
    steps(p,s,45,lead=.015)
    p.driver.p = p.driver.p-[.006,0,0]         # 21 mm behind: inside 15 + 7 mm
    s.tick()
    assert s.fault is None
    p.driver.p = p.driver.p-[.002,0,0]         # 23 mm
    with pytest.raises(RuntimeError,match='Tracking error'):
        s.tick()


def test_lead_outside_contact_lead_is_rejected():
    p,s = session()
    with pytest.raises(ValueError,match='contact_lead'):
        s.command([.03,0,0],[0,0,0],[0,0,0],[0,0,0,1],.03)


def test_slide_asks_for_contact_lead_only_while_pressing_and_presses_past_8mm():
    p = DemoPort(BackendConfig(motion_enabled=True,f_cap=25.),plane=.33)
    leads,request = [],p.request
    def spy(op,**args):
        if op=='step':
            leads.append(args.get('lead'))
        return request(op,**args)
    p.request = spy
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        assert prim.move_until_contact([0,0,-1],max_dist=.03,f_threshold=2.).termination=='contact'
        assert set(leads)=={None}
        leads.clear()
        rep = prim.slide([1,0,0],.05,normal_force=15.)
        assert b.step_lead is None
    assert CONTACT_LEAD_M in leads and leads[0] is None
    # fixture: 750 N/m plane. 8 mm could press 6 N; the 80% share of 25 mm presses ~15 N
    fz = np.array(rep.wrench_rows)[-20:,3]-p.driver.bias[2]
    assert 14.<fz.mean()<=15.
    assert rep.termination=='completed'
