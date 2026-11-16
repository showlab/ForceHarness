"""The first press seats the screw: screw presses with the whole contact lead until the screw stops sinking before it
turns, and the press stops rising at 23 N raw and backs off above 24 N, ahead of a fast rise. Offline only."""
import numpy as np

from forceharness.real import primitives as cp
from test_screw import TOP, held_on_surface


def soft_mouth(port, depth=.002, rate=.00005, above_n=5.):
    """Pressed harder than above_n, the screw goes in at 1 mm/s until `depth` (then it is seated)."""
    plant, original, sunk = port.driver, port.driver.update, [0.]

    def update(pose, kp, kd, heartbeat):
        if (TOP-sunk[0]-pose[2])*kp[2] > above_n:
            sunk[0] = min(depth, sunk[0]+rate)
        plant.plane = TOP-sunk[0]
        return original(pose, kp, kd, heartbeat)
    plant.update = update


def test_presses_until_seated_then_turns():
    p, b, prim = held_on_surface(plane=False)
    p.driver.plane = TOP
    soft_mouth(p)
    rep = prim.call('screw', angle_deg=-10)
    b.__exit__(None, None, None)
    d = rep.derived
    assert d['press_seated'] and d['press_sink_m'] > .0018          # pressed all the way in first
    assert d['press_s'] > 15/8+1.                                   # longer than ramp + 1 s: it waited
    assert rep.params['push'] == cp.SCREW_PUSH_N == 15.


def test_press_is_no_longer_held_below_a_raw_reading():
    # the raw reading does not stop the press; only the robot side's absolute 25 N cap does
    p, b, prim = held_on_surface()
    plant, original = p.driver, p.driver.update

    def heavy(pose, kp, kd, heartbeat):
        out = original(pose, kp, kd, heartbeat)
        plant.w[2] += 5.                                            # the raw reading sits 5 N higher
        return out
    plant.update = heavy
    rep = prim.call('screw', angle_deg=-10)
    b.__exit__(None, None, None)
    raw = np.linalg.norm(np.asarray(rep.wrench_rows)[:, 1:4], axis=1)
    assert raw.max() > 19. and not b.fault          # the hold may act for a moment on the rise; it still gets there


class Hand:
    eef_pos = [.40, 0., .25]
    _state = {}


def test_push_may_press_up_to_fifteen_newton():
    # push may press with up to 15 N (to press a part in after a search)
    import pytest
    from forceharness.real import runner as free
    assert free.validate('push', dict(direction=[0, 0, -1], target_force=15), Hand(), 'A')[0]['target_force'] == 15
    with pytest.raises(ValueError):
        free.validate('push', dict(direction=[0, 0, -1], target_force=15.5), Hand(), 'A')
    assert 'target_force (≤ 15 N)' in free.menu_text('A', free.GRIP_RULES['auto'])


def test_hold2_keeps_a_drifting_reading_off_the_nuc_cap():
    # a press can drive the raw reading into the robot side's 25 N cap: stop at 23 N, back off above 24 N,
    # even while the reading keeps creeping up (+3.7 N while held was seen)
    p, b, prim = held_on_surface()
    plant, original, n = p.driver, p.driver.update, [0]

    def drifting(pose, kp, kd, heartbeat):
        out = original(pose, kp, kd, heartbeat)
        n[0] += 1
        plant.w[2] += 10.+min(.02*n[0], 6.)                          # 10 N high, creeping up to +6 N more
        return out
    plant.update = drifting
    rep = prim.call('screw', angle_deg=-30)
    b.__exit__(None, None, None)
    raw = np.linalg.norm(np.asarray(rep.wrench_rows)[:, 1:4], axis=1)
    assert not b.fault and rep.termination == 'completed'
    assert rep.derived['press_force_capped'] and raw.max() < 25.


def test_hold3_stops_a_steep_late_press_short_of_the_stops():
    # a press on a jammed, crooked screw can rise by ~11 N/s while the reading lags ~0.25 s, so a plain 23 N
    # hold overshoots past 25 N: hold 0.3 s ahead of the rise
    p, b, prim = held_on_surface(plane=False)
    plant, original, felt = p.driver, p.driver.update, []

    def jammed(pose, kp, kd, heartbeat):
        out = original(pose, kp, kd, heartbeat)
        plant.p[2] = max(plant.p[2], TOP)
        felt.append(max(0., TOP-pose[2])*kp[2]*1.9)                 # ~1.4 N/mm, as the jam
        f = felt[-26] if len(felt) > 25 else 0.                      # read ~5 steps (0.25 s) late
        plant.w[:3] = plant.bias[:3]+[-.43*f, 0., f]                 # ~23 deg off vertical
        return out
    plant.update = jammed
    rep = prim.call('screw', angle_deg=-30)
    b.__exit__(None, None, None)
    net = np.linalg.norm(np.asarray(rep.wrench_rows)[:, 1:4], axis=1)
    assert not b.fault and rep.termination == 'completed'
    assert rep.derived['press_force_capped'] and net.max() < 24.
