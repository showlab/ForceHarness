"""Search sensitivity: a shallow dip must already pause the spiral; on the FR3 the catching side force is friction,
so the simulation's guided hops are off, and arming needs the board to stop the hand (a force rise drifts in the air).
Offline only."""
import numpy as np
import pytest

from forceharness.primitives.library import Primitives
from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real.plant import DemoPort

TOP = .349


def mouth_edge(port, offset=(.003, 0.), radius=.0015, light=.00035, pressed=.0015, rate=.00005):
    """Over the mouth the tip catches the edge: light pressing lets it in `light` deep, pressing harder
    than 5 N lets it in `pressed` deep (1 mm/s); off the mouth it is the flat board."""
    plant, original = port.driver, port.driver.update
    centre = plant.p[:2]+np.array(offset)
    sunk = [0.]

    def update(pose, kp, kd, heartbeat):
        inside = np.linalg.norm(np.asarray(pose[:2])-centre) <= radius
        force = (TOP-sunk[0]-pose[2])*kp[2]
        limit = pressed if force > 5. else light
        sunk[0] = min(limit, sunk[0]+rate) if inside else 0.
        plant.plane = TOP-sunk[0]
        return original(pose, kp, kd, heartbeat)
    plant.update = update
    return centre


def slide(**kw):
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        centre = mouth_edge(p)
        rep = prim.call('search', normal=[0, 0, -1], radius_max=.005, pitch=.001, normal_force=2.,
                        drop=.001, mode='slide', **kw)
        end = np.asarray(p.driver.p[:2]).copy()
        assert not b.fault
    return rep, centre, end


def test_a_shallow_catch_is_paused_and_pressed_in():
    rep, centre, end = slide()
    assert rep.termination == 'dropped' and rep.derived['sink_pauses'] >= 1
    depth = rep.derived['found_by_pause_at_m'] or rep.derived['advance_along_normal_m']   # or past drop while pressed
    assert depth >= .0008 and np.linalg.norm(end-centre) <= .0015


def test_the_old_half_millimetre_pause_misses_it(monkeypatch):
    monkeypatch.setattr(ContinuousPrimitives, 'search_settle', (.0005, .0008, 2., 6.))
    rep, _, _ = slide()
    assert rep.termination != 'dropped' and rep.derived['sink_pauses'] == 0


def test_sim_keeps_hops_and_support_arming():
    assert Primitives.search_guided_hops == 8 and Primitives.search_arm_rise is None
    assert Primitives.search_arm_blocked is None and Primitives.search_fall_rule is True
    assert ContinuousPrimitives.search_guided_hops == 0 and ContinuousPrimitives.search_arm_rise is None
    assert ContinuousPrimitives.search_arm_blocked == (.002, .0002, .3) and ContinuousPrimitives.search_fall_rule is False


@pytest.mark.parametrize('arming', ['support', 'rise', 'blocked'])
def test_force_drift_in_the_air_no_longer_arms_it(monkeypatch, arming):
    # a held screw's estimate can rise ~2 N by itself while descending 9 mm above the board
    if arming != 'blocked':
        monkeypatch.setattr(ContinuousPrimitives, 'search_arm_blocked', None)
        monkeypatch.setattr(ContinuousPrimitives, 'search_arm_rise', 2. if arming == 'rise' else None)
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        plant, original, n = p.driver, p.driver.update, [0]
        plant.plane = float(plant.p[2])-.009                        # the board is 9 mm below the tip

        def drifting(pose, kp, kd, heartbeat):
            out = original(pose, kp, kd, heartbeat)
            n[0] += 1
            plant.w[2] += 1.+min(.05*n[0], 2.5)                     # 1 N at once, then +2.5 N while it moves
            return out
        plant.update = drifting
        rep = prim.call('search', normal=[0, 0, -1], radius_max=.003, pitch=.001, normal_force=2.,
                        drop=.001, mode='slide')
        assert not b.fault
    if arming == 'blocked':
        assert rep.termination == 'completed' and rep.derived['advance_along_normal_m'] < .001
    else:
        assert rep.termination in ('dropped', 'lost_contact')       # armed in the air: a bogus outcome


def bump(port, offset=(.0015, 0.), radius=.001, height=.0011):
    """A 1.1 mm bump the slide rides over and drops back off (as logged); coming off it the force
    estimate dips for a moment, as on the FR3."""
    plant, original = port.driver, port.driver.update
    centre = plant.p[:2]+np.array(offset)
    st = dict(on=False, dip=0)

    def update(pose, kp, kd, heartbeat):
        on = np.linalg.norm(np.asarray(pose[:2])-centre) <= radius
        if st['on'] and not on:
            st['dip'] = 5
        st['on'] = on
        plant.plane = TOP+height if on else TOP
        out = original(pose, kp, kd, heartbeat)
        if st['dip']:
            st['dip'] -= 1
            plant.w[2] -= 3.
        return out
    plant.update = update


@pytest.mark.parametrize('fall_rule', [True, False])
def test_riding_off_a_bump_is_not_a_hole(monkeypatch, fall_rule):
    monkeypatch.setattr(ContinuousPrimitives, 'search_fall_rule', fall_rule)
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        bump(p)
        rep = prim.call('search', normal=[0, 0, -1], radius_max=.003, pitch=.001, normal_force=2.,
                        drop=.001, mode='slide')
    assert (rep.termination == 'dropped') == fall_rule


class Hand:
    eef_pos = [.40, 0., .25]
    _state = {}


def test_wider_search_and_the_reach_it_reports():
    from forceharness.real import runner as free
    assert free.validate('search', dict(radius_max=.015), Hand(), 'A')[0]['radius_max'] == .015
    with pytest.raises(ValueError):
        free.validate('search', dict(radius_max=.016), Hand(), 'A')
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        rep = ContinuousPrimitives(b).call('search', normal=[0, 0, -1], radius_max=.004, pitch=.002, drop=.001)
    assert rep.params['radius_max'] == .004 and 'reached' in rep.derived['note']
    assert 0 < rep.derived['radius_reached_m'] <= .0045
