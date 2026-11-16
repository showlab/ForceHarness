"""Once the tip dips into a hole mouth, search thrusts it in with about 15 N and finishes the 2 s press before
judging; a 6 N pause press could not push a stiff thread mouth open. Offline only."""
import numpy as np
import pytest

from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real.plant import DemoPort

TOP = .349


def stiff_mouth(port, offset=(.003, 0.), radius=.0015, light=.00035, deep=.002, above_n=10., rate=.00005):
    """Over the mouth a light touch lets the tip in `light` deep; only pressing harder than above_n opens it
    to `deep` (1 mm/s). Off the mouth it is the flat board."""
    plant, original = port.driver, port.driver.update
    centre = plant.p[:2]+np.array(offset)
    sunk = [0.]

    def update(pose, kp, kd, heartbeat):
        inside = np.linalg.norm(np.asarray(pose[:2])-centre) <= radius
        force = (TOP-sunk[0]-pose[2])*kp[2]
        goal = deep if force > above_n else light
        sunk[0] = max(sunk[0], min(goal, sunk[0]+rate)) if inside else 0.     # once in, it stays in
        plant.plane = TOP-sunk[0]
        return original(pose, kp, kd, heartbeat)
    plant.update = update
    return centre


def slide(**patch):
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        for k, v in patch.items():
            setattr(prim, k, v)
        centre = stiff_mouth(p)
        rep = prim.call('search', normal=[0, 0, -1], radius_max=.005, pitch=.001, normal_force=2.,
                        drop=.001, mode='slide')
        end = np.asarray(p.driver.p[:2]).copy()
        assert not b.fault and prim.lead_override is None
    return rep, centre, end


def test_the_thrust_presses_a_stiff_mouth_open():
    rep, centre, end = slide()
    assert rep.termination == 'dropped' and rep.derived['sink_pauses'] >= 1
    assert rep.derived['found_by_pause_at_m'] >= .0015                # pressed well in, the whole 2 s
    assert np.linalg.norm(end-centre) <= .0015


def test_a_six_newton_pause_press_could_not():
    rep, _, _ = slide(search_settle=(.00025, .0008, 2., 6.), search_press=None)
    assert rep.termination != 'dropped' and rep.derived['sink_pauses'] >= 1


def test_the_sim_pause_is_unchanged():
    from forceharness.primitives.library import Primitives
    assert Primitives.search_press is None and Primitives.lead_override is None
    assert ContinuousPrimitives.search_press == (.025, 23., .0015)                     # stop raising the press at 23 N raw
