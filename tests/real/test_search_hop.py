"""Hop search: a slide search drags a screw tip sideways and tilts it in the two-pad grip, so hop probes instead. It
pushes past joint friction (the simulation's hop waits at a point the hand cannot reach), zeroes the force before every
probe, and stops as soon as a probe is deeper than the surface so far by drop. Offline only."""
import numpy as np
import pytest

from forceharness.primitives.library import Primitives
from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real import primitives as cp
from forceharness.real.plant import DemoPort

TOP = .349


@pytest.fixture(autouse=True)
def demo_probe_force(monkeypatch):
    # the demo plant is 750 N/m, so an 8 mm lead gives 6.0 N at most; the FR3 gives 8-10 N
    monkeypatch.setattr(cp, 'HOP_PROBE_N', 4.)


def board(port, hole=(.002, 0.), radius=.0015, depth=.002, stick=0., drift=0., tight=False):
    """Flat board with a hole near the start; optional sideways stiction (the hand stays put unless the
    target is `stick` away), a z force estimate drifting `drift` N per update, and a tight mouth that
    lets the tip in only while it is pressed with more than 2 N (1 mm/s); lifting resets it."""
    plant = port.driver
    centre = None if hole is None else plant.p[:2]+np.array(hole)
    original, n, sunk = plant.update, [0], [0.]

    def update(pose, kp, kd, heartbeat):
        pose = np.array(pose, dtype=float)
        if stick and np.linalg.norm(pose[:2]-plant.p[:2]) < stick:
            pose[:2] = plant.p[:2]
        inside = centre is not None and np.linalg.norm(pose[:2]-centre) <= radius
        if tight:
            pressed = (plant.plane or TOP)-pose[2] > 2./kp[2]
            lifted = pose[2] > TOP
            sunk[0] = (min(depth, sunk[0]+.00005) if inside and pressed else
                       0. if lifted or not inside else sunk[0])
            plant.plane = TOP-sunk[0]
        else:
            plant.plane = TOP-depth if inside else TOP
        out = original(pose, kp, kd, heartbeat)
        n[0] += 1
        plant.w[2] += drift*n[0]
        return out
    plant.update = update
    return centre


def run(sim_hop=False, radius_max=.004, pitch=.001, **kw):
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        prim.call('move_until_contact', direction=[0, 0, -1], max_dist=.003, f_threshold=.5)
        centre = board(p, **kw)
        if sim_hop:
            rep = Primitives._search_hop(prim, np.array([0., 0., -1.]), radius_max, pitch, .001, None, .003)
        else:
            rep = prim.call('search', normal=[0, 0, -1], radius_max=radius_max, pitch=pitch, drop=.001,
                            mode='hop', hop=.003)
        end = np.asarray(p.driver.p[:2]).copy()
        assert not b.fault
    return rep, centre, end


def test_finds_the_hole_and_stops_at_once():
    rep, centre, end = run()
    assert rep.termination == 'dropped' and rep.derived['probes'] >= 2
    assert .001 <= rep.derived['drop_depth_m'] < .0015             # stopped just past drop, not at the 2 mm bottom
    assert np.linalg.norm(end-centre) <= .0015


def test_joint_friction_does_not_stall_it():
    old, _, _ = run(sim_hop=True, hole=None, radius_max=.003, stick=.002)
    new, _, _ = run(hole=None, radius_max=.003, stick=.002)
    assert old.termination == 'timeout'                                     # got stuck waiting at a point
    assert new.termination == 'completed' and new.derived['probes'] > old.derived['probes']


def test_drifting_force_estimate_does_not_fake_contacts():
    rep, centre, end = run(drift=.01)                              # +3 N after 300 updates
    assert rep.termination == 'dropped' and np.linalg.norm(end-centre) <= .0015


def test_a_tight_mouth_found_by_pressing():
    rep, centre, end = run(tight=True)
    assert rep.termination == 'dropped' and np.linalg.norm(end-centre) <= .0015
    assert rep.derived['probe_N'] == 4.


def test_too_many_points_is_refused():
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        with pytest.raises(ValueError, match='larger pitch'):
            ContinuousPrimitives(b).call('search', radius_max=.01, pitch=.0005, mode='hop')
