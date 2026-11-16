"""screw: turn like rotate_z while pressing about `push` N down the axis and following the thread. rotate_z keeps the
hand still, so a thread drags it against the controller (1.5-1.7 mm per 30 deg was seen). Offline only."""
import json

import numpy as np
import pytest

from forceharness.real import runner as free
from forceharness.real import validation
from forceharness.primitives.library import _yaw_deg_xyzw
from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real.plant import DemoPort

TOP = .349
PITCH = .003


def thread(port, z, pitch=PITCH, floor=None):
    """Surface under the screw tip that goes down `pitch` per clockwise turn of the hand (down to `floor`)."""
    plant = port.driver
    yaw0 = _yaw_deg_xyzw(plant.r)
    original = plant.update

    def update(pose, kp, kd, heartbeat):
        turned = ((_yaw_deg_xyzw(np.asarray(pose[3:]))-yaw0+180.) % 360.)-180.
        plant.plane = max(z+pitch*turned/360., floor if floor is not None else -1.)
        return original(pose, kp, kd, heartbeat)
    plant.update = update


def held_on_surface(floor=None, plane=True):
    p = DemoPort(plane=TOP, payload=.01)
    b = p.backend()
    b.__enter__()
    b.calibration['orientation_scope_record_only'] = True     # as configs/real/force_calibration.json
    prim = ContinuousPrimitives(b, confirm=lambda *_: True)
    prim.grasp()
    prim.move_until_contact([0, 0, -1], max_dist=.003, f_threshold=.5)
    if plane:
        thread(p, TOP, floor=floor)
    else:
        p.driver.plane = None
    return p, b, prim


def advance(rep):
    return next(q for q in rep.quantities if q['name'] == 'screw_advance')


def test_screw_follows_the_thread_down():
    p, b, prim = held_on_surface()
    rep = prim.call('screw', angle_deg=-30)
    b.__exit__(None, None, None)
    q = advance(rep)
    assert rep.termination == 'completed' and rep.yaw_deg == pytest.approx(-30, abs=1)
    assert rep.derived['sink_m'] == pytest.approx(PITCH/12, abs=5e-5)
    assert q['flags'] == ['advancing'] and q['extra']['per_turn_mm'] == pytest.approx(3., abs=.3)
    assert 13. < rep.peak_force < 16.                       # 15 N as a 20 mm lead on the 750 N/m demo plant


def test_rotate_z_keeps_the_hand_still_on_the_same_thread():
    p, b, prim = held_on_surface()
    z0 = b.eef_pos[2]
    rep = prim.call('rotate_z', angle_deg=-30)
    b.__exit__(None, None, None)
    assert rep.termination == 'completed' and abs(b.eef_pos[2]-z0) < 1e-4


def test_seated_screw_reports_no_advance():
    p, b, prim = held_on_surface(floor=TOP)                  # already at the bottom
    rep = prim.call('screw', angle_deg=-30)
    b.__exit__(None, None, None)
    assert rep.termination == 'completed' and advance(rep)['flags'] == ['not_advancing']


def test_nothing_underneath_stops_after_ten_millimetres():
    p, b, prim = held_on_surface(plane=False)
    rep = prim.call('screw', angle_deg=-30)
    b.__exit__(None, None, None)
    assert rep.termination == 'sink_limit' and .01 <= rep.derived['sink_m'] < .0115


def test_screw_rejects_out_of_range_arguments():
    p, b, prim = held_on_surface()
    for args in (dict(angle_deg=-40), dict(angle_deg=-30, push=16), dict(angle_deg=-30, push=0)):
        with pytest.raises(ValueError):
            prim.call('screw', **args)
    b.__exit__(None, None, None)


class Hand:
    eef_pos = [.40, 0., .25]


def test_menu_is_force_only_and_validated(monkeypatch):
    monkeypatch.setattr(validation, 'rotation_reach', lambda *a, **k: (90., None))
    assert 'screw' in free.CONDITIONS['A']['menu'] and 'screw' not in free.CONDITIONS['B']['menu']
    assert free.validate('screw', dict(angle_deg=-90, push=3), Hand(), 'A')[0] == dict(angle_deg=-90, push=3)
    for args in (dict(angle_deg=-181), dict(angle_deg=-30, push=15.5)):
        with pytest.raises(ValueError):
            free.validate('screw', args, Hand(), 'A')
    with pytest.raises(ValueError, match='not on the menu'):
        free.validate('screw', dict(angle_deg=-30), Hand(), 'B')
    assert 'screw(angle_deg, push?)' in free.menu_text('A', free.GRIP_RULES['auto'])


def test_runner_segments_a_long_screw_and_sums_the_advance(tmp_path, monkeypatch):
    original = free.demo

    def plant():
        b, world, calls = original()
        driver, upd, st = b.port.driver, b.port.driver.update, {}

        def update(pose, kp, kd, heartbeat):
            if world.held is not None:                   # a threaded hole just under the grasped part
                yaw = _yaw_deg_xyzw(np.asarray(pose[3:]))
                if 'z' not in st:
                    st.update(z=driver.p[2], yaw=yaw)
                driver.plane = st['z']+PITCH*(((yaw-st['yaw']+180.) % 360.)-180.)/360.
            return upd(pose, kp, kd, heartbeat)
        driver.update = update
        return b, world, calls[:5]+[dict(name='screw', args=dict(angle_deg=-60)),
                                    dict(name='done', args=dict(success=True, note='x'))]
    monkeypatch.setattr(free, 'demo', plant)
    code = free.main(['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'), '--instruction', 'x'])
    r = json.loads((tmp_path/'ep'/'episode.json').read_text())
    assert code == 0 and r['fault'] is None
    i = [a['name'] for a in r['actions']].index('screw')
    rep = json.loads((tmp_path/'ep'/f'action_{i:02d}.json').read_text())
    assert rep['termination'] == 'completed' and len(rep['derived']['segments']) == 2
    q = next(q for q in rep['quantities'] if q['name'] == 'screw_advance')
    assert q['extra']['turned_deg'] == pytest.approx(-60, abs=2) and len(q['extra']['segment_sink_mm']) == 2
    assert q['flags'] == ['advancing'] and q['extra']['per_turn_mm'] == pytest.approx(3., abs=.5)
