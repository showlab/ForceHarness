"""A slide search pressed at 5-8 N may not slide at all, its target circling while the hand stays put, until the
robot side's tracking-stall fault ends the session. The runner stops such a primitive at 3.5 s with a tool result and
the episode goes on. Offline only."""
import numpy as np
import pytest

from forceharness.real import runner as free
from forceharness.real.rotation_progress import TrackingProgress


def test_tracking_progress_matches_the_nuc_rule_earlier():
    g = TrackingProgress()
    hand = np.zeros(3)
    out = [g.observe(t, hand, .003*np.array([np.cos(t), np.sin(t), 0.])) for t in np.arange(0, 4, .05)]
    assert not any(out[:int(3.5/.05)]) and out[-1]                       # circling target, stuck hand
    g = TrackingProgress()
    assert not any(g.observe(t, hand, [0, 0, -.008]) for t in np.arange(0, 6, .05))   # pressing, target still
    g = TrackingProgress()
    assert not any(g.observe(t, [.001*t, 0, 0], [.001*t+.004, 0, 0]) for t in np.arange(0, 6, .05))   # moving


def sticky(driver, stick_n=3.):
    """Static friction: pressed harder than stick_n onto the plane, the hand does not move sideways."""
    original = driver.update

    def update(pose, kp, kd, heartbeat):
        pose = np.array(pose, dtype=float)
        if driver.plane is not None and (driver.plane-pose[2])*kp[2] > stick_n:
            pose[:2] = driver.p[:2]
        return original(pose, kp, kd, heartbeat)
    driver.update = update


@pytest.mark.parametrize('force, stuck', [(5., True), (1., False)])
def test_pressed_slide_search_stops_with_a_receipt_not_a_fault(force, stuck):
    b, _, _ = free.demo()
    with b:
        prim = free.FreePrimitives(b, confirm=lambda kind, state: True, tick=lambda: None, stop_protocol='r9',
                                   interface='force')
        prim.target_quat = np.asarray(b.eef_quat, dtype=float).copy()
        b.port.driver.plane = float(b.port.driver.p[2])-.001
        sticky(b.port.driver)
        rep = prim.call('search', normal=[0, 0, -1], radius_max=.006, pitch=.001, normal_force=force,
                        drop=.002, mode='slide')
        assert not b.fault
    if stuck:
        assert rep.termination == 'stalled' and rep.derived['relay_guard']['kind'] == 'stalled'
        assert rep.derived['tracking_progress']['version'] == 'stall1' and 'pressed too hard' in rep.derived['note']
        assert 3.5 < rep.derived['tracking_progress']['stuck_s'] < 4.
    else:
        assert 'tracking_progress' not in rep.derived
