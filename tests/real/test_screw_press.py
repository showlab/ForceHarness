"""screw presses first and hard (1-3 N never pressed a screw into the thread), then turns; the commanded yaw waits for
a slowed wrist instead of running into the robot-side IK continuity check. Offline only."""
import numpy as np
import pytest

from forceharness.real import runner as free
from forceharness.real import validation
from forceharness.primitives.library import _yaw_deg_xyzw
from forceharness.real import primitives as cp
from test_screw import held_on_surface


def update_yaws(port, start):
    return [_yaw_deg_xyzw(np.asarray(u['pose'][3:])) for u in port.driver.updates[start:]]


def test_presses_first_then_turns():
    p, b, prim = held_on_surface()
    n0 = len(p.driver.updates)
    rep = prim.call('screw', angle_deg=-30)
    b.__exit__(None, None, None)
    press = rep.derived['press_s']
    assert 2.4 <= press <= 4.2                      # 20 mm at 8 mm/s, then held >= 1 s
    yaws = update_yaws(p, n0)                       # flange yaw (the TCP is turned 45 deg from it)
    first_turn = next(i for i, y in enumerate(yaws) if abs(y-yaws[0]) > .2)
    assert first_turn*b.dt >= press-.2               # no turning while it presses
    assert rep.yaw_deg == pytest.approx(-30, abs=1) and prim.screw_phase is None
    q = next(q for q in rep.quantities if q['name'] == 'screw_advance')
    assert 'while pressing before turning' in q['unit'] and q['extra']['turn_sink_mm'] == pytest.approx(.25, abs=.05)


def test_high_push_asks_for_the_contact_lead_only_while_pressing():
    p, b, prim = held_on_surface()
    leads, request = [], p.request

    def spy(op, **kw):
        if op == 'step':
            leads.append(kw.get('lead'))
        return request(op, **kw)
    p.request = spy
    rep = prim.call('screw', angle_deg=-10, push=15)
    assert rep.derived['push_lead_m'] == pytest.approx(15/cp.SCREW_KZ_N_PER_M)        # 20 mm, within 25
    assert cp.CONTACT_LEAD_M in leads and rep.peak_force > 9.      # 15 mm on the 750 N/m demo plant
    assert b.step_lead is None and not b.fault
    b.__exit__(None, None, None)
    p, b, prim = held_on_surface(plane=False)                       # nothing underneath: plain 8 mm lead
    leads.clear()
    request = p.request
    p.request = spy
    rep = prim.call('screw', angle_deg=-10, push=15)
    b.__exit__(None, None, None)
    assert rep.termination == 'sink_limit' and set(leads) == {None}


def test_commanded_yaw_waits_for_a_stuck_wrist():
    p, b, prim = held_on_surface()
    plant, original = p.driver, p.driver.update
    quat = np.asarray(plant.r, dtype=float).copy()

    def stuck(pose, kp, kd, heartbeat):
        out = original(pose, kp, kd, heartbeat)
        plant.r = quat.copy()                                       # the wrist does not turn
        return out
    plant.update = stuck
    n0 = len(plant.updates)
    rep = prim.call('screw', angle_deg=-30)
    b.__exit__(None, None, None)
    yaws = update_yaws(p, n0)
    assert not b.fault and rep.derived['yaw_wait_steps'] > 0
    assert max(abs(y-yaws[0]) for y in yaws) <= np.degrees(cp.SCREW_YAW_LEAD_RAD)+.5


def test_runner_rotation_stall_guard_waits_for_the_turn(monkeypatch):
    monkeypatch.setattr(cp, 'SCREW_PRESS_HOLD_S', 4.)               # a long press: no wrist motion for ~5 s
    b, _, _ = free.demo()
    with b:
        prim = free.FreePrimitives(b, confirm=lambda kind, state: True, tick=lambda: None, stop_protocol='r9',
                                   interface='force')
        prim.target_quat = np.asarray(b.eef_quat, dtype=float).copy()
        b.port.driver.plane = float(b.port.driver.p[2])-.0005
        rep = prim.call('screw', angle_deg=-20)
    assert rep.derived['press_s'] >= 4. and rep.yaw_deg == pytest.approx(-20, abs=1)
    assert 'relay_guard' not in rep.derived and rep.derived['rotation_progress']['version'] == 'r33c'


def test_screw_collision_stop_sits_above_the_push(monkeypatch):
    monkeypatch.setattr(validation, 'rotation_reach', lambda *a, **k: (90., None))
    assert 'screw' not in free.COLLISION_GUARDED and cp.SCREW_PRESS_RAW_N == 23.     # no workstation stop; the press holds at 23 N
    assert free.validate('screw', dict(angle_deg=-10, push=15), SimpleHand(), 'A')[0]['push'] == 15
    with pytest.raises(ValueError):
        free.validate('screw', dict(angle_deg=-10, push=15.5), SimpleHand(), 'A')


class SimpleHand:
    eef_pos = [.40, 0., .25]
