"""A hop search that starts above a port: the last nearby support contact judges the first probe, every hop rises
above the shallowest probe so far, and a hand that stays put through several unreached points stops as stuck (it had
dipped into the same port again and again). Offline only."""
import numpy as np
import pytest

from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real import primitives as cp
from forceharness.real.plant import DemoPort

from test_search_hop import TOP, board


@pytest.fixture(autouse=True)
def demo_probe_force(monkeypatch):
    monkeypatch.setattr(cp, 'HOP_PROBE_N', 4.)


def search(prim, **kw):
    args = dict(normal=[0, 0, -1], radius_max=.004, pitch=.0015, drop=.0015, mode='hop', hop=.003)
    args.update(kw)
    return prim.call('search', **args)


def test_first_probe_into_the_hole_is_judged_by_the_earlier_touch():
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        assert prim.call('move_until_contact', direction=[0, 0, -1], max_dist=.003, f_threshold=.5).termination == 'contact'
        board(p, hole=(0., 0.), radius=.002, depth=.005)
        up = np.asarray(b.eef_pos)+[0, 0, .014]
        prim.call('move_to', x=float(up[0]), y=float(up[1]), z=float(up[2]))
        rep = search(prim)
        assert not b.fault
    assert rep.termination == 'dropped' and rep.derived['probes'] == 0
    assert rep.derived['prior_support_depth_m'] == pytest.approx(.014, abs=.002)


def test_a_far_touch_is_not_a_reference():
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        prim.call('move_until_contact', direction=[0, 0, -1], max_dist=.003, f_threshold=.5)
        prim.last_support['pos'][0] += .03
        board(p, hole=None)
        rep = search(prim, radius_max=.002)
    assert rep.derived['prior_support_depth_m'] is None and rep.termination != 'dropped'


def test_a_hand_held_in_place_stops_as_stuck():
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        prim.call('move_until_contact', direction=[0, 0, -1], max_dist=.003, f_threshold=.5)
        board(p, hole=None, stick=.02)          # sideways the hand never leaves the spot
        rep = search(prim)
        assert not b.fault
    assert rep.termination == 'stuck' and rep.derived['held_in_place']
    assert cp.HOP_STUCK_POINTS <= rep.derived['probes'] <= cp.HOP_STUCK_POINTS+3   # the first points lie within 1 mm
