"""On the real robot a slide search that sinks deeper than drop stops as dropped (a threaded hole never lets the tip
fall free, and the spiral dragged a sunk screw back out). Offline only."""
import numpy as np

from forceharness.primitives.library import Primitives
from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real.plant import DemoPort

TOP = .349


def threaded_hole(port, offset=(-.0025, 0.), radius=.0015, depth=.0025, rate=.0000125):
    """Threaded hole mouth as observed: once the tip is over the hole it sinks slowly while
    still pressed (0.25 mm/s, never losing support) and the hole wall keeps it from leaving."""
    plant = port.driver
    centre = plant.p[:2] + np.array(offset)
    state = dict(sink=0.)
    original = plant.update

    def update(pose, kp, kd, heartbeat):
        pose = np.array(pose, dtype=float)
        rel = pose[:2] - centre
        d = float(np.hypot(*rel))
        if state['sink'] > .0005 and d > radius:
            pose[:2] = centre + rel / d * radius
            d = radius
        if d <= radius:
            state['sink'] = min(depth, state['sink'] + rate)
        plant.plane = TOP - state['sink'] if d <= radius else TOP
        return original(pose, kp, kd, heartbeat)
    plant.update = update


def run_search(sink_is_drop):
    p = DemoPort(plane=TOP)
    threaded_hole(p)
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        prim.search_sink_is_drop = sink_is_drop
        if not sink_is_drop:
            prim.search_settle = None          # the plain rule without the pause
        rep = prim.call('search', normal=[0, 0, -1], radius_max=.004, pitch=.001,
                        normal_force=1., drop=.002, mode='slide')
    return rep


def test_sinking_past_drop_ends_search_as_dropped():
    rep = run_search(True)
    assert rep.termination == 'dropped'
    assert rep.derived['advance_along_normal_m'] > .002
    assert rep.derived['radius_searched_m'] < .004       # stopped at once, no more dragging
    flags = next(q for q in rep.quantities if q['name'] == 'insertion_depth')['flags']
    assert 'dropped_in' in flags


def test_old_rule_reproduces_no_hole_found():
    rep = run_search(False)
    assert rep.termination == 'completed'
    flags = next(q for q in rep.quantities if q['name'] == 'insertion_depth')['flags']
    assert 'no_hole_found' in flags


def test_sim_keeps_the_blocked_rule():
    assert Primitives.search_sink_is_drop is False
    assert ContinuousPrimitives.search_sink_is_drop is True
