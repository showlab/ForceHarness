"""A slide search that catches a threaded hole mouth pauses there instead of letting the spiral drag the tip back
out, and judges the hole only after a full press. Offline only."""
import numpy as np

from forceharness.primitives.library import Primitives
from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real.plant import DemoPort

TOP = .349


def hole_mouth(port, offset=(.003, 0.), radius=.0015, depth=.001, rate=.00005):
    """A mouth the tip sinks into while it stays over it (1 mm/s, at most `depth`); leaving resets it."""
    plant = port.driver
    centre = plant.p[:2]+np.array(offset)
    state = dict(sink=0.)
    original = plant.update

    def update(pose, kp, kd, heartbeat):
        inside = float(np.hypot(*(np.asarray(pose[:2])-centre))) <= radius
        state['sink'] = min(depth, state['sink']+rate) if inside else 0.
        plant.plane = TOP-state['sink']
        return original(pose, kp, kd, heartbeat)
    plant.update = update
    return centre


def run(settle=True, **hole):
    p = DemoPort(plane=TOP)
    centre = hole_mouth(p, **hole) if hole else None
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        if not settle:
            prim.search_settle = None
        z0 = b.eef_pos[2]
        rep = prim.call('search', normal=[0, 0, -1], radius_max=.005, pitch=.001, normal_force=2.,
                        drop=.0015, mode='slide')
        zmin = min(r[2] for r in rep.pos_rows)
        end = np.asarray(p.driver.p[:2])
    return rep, centre, end, z0-zmin


def test_old_spiral_drags_the_tip_back_out():
    rep, _, _, deepest = run(settle=False, depth=.001)
    assert rep.termination != 'dropped' and deepest > .0003     # it went in, then the spiral moved on


def test_pause_keeps_it_in_and_calls_it_a_hole():
    rep, centre, end, _ = run(depth=.002)                       # a hole is >= 1.5 mm after the thrust
    assert rep.termination == 'dropped' and rep.derived['sink_pauses'] >= 1
    assert .0015 <= rep.derived['found_by_pause_at_m'] <= .0021
    assert rep.derived['found_from_start_m'] < .0045 and 'found' in rep.derived['note']
    assert np.linalg.norm(end-centre) <= .0015                  # stopped over the mouth


def test_a_flat_press_sink_under_the_thrust_is_not_a_hole():
    rep, _, _, _ = run(depth=.001)                              # 1 mm: what a hard press gives on a flat spot
    assert rep.termination != 'dropped' and rep.derived['sink_pauses'] >= 1


def test_flat_surface_and_shallow_dent_slide_on():
    rep, _, _, _ = run()
    assert rep.termination == 'completed' and rep.derived['sink_pauses'] == 0
    rep, _, _, _ = run(depth=.0006)                             # a dent: pauses, then slides on
    assert rep.termination == 'completed' and 1 <= rep.derived['sink_pauses'] <= 4


def test_sim_spiral_unchanged():
    assert Primitives.search_settle is None and ContinuousPrimitives.search_settle == (.00025, .0008, 2., 15.)
