"""wiggle: press while rocking a plug caught on the port mouth; once the port walls hold it sideways it only presses,
and depth counts from the support surface. Offline only: synthetic ports."""
import numpy as np
import pytest

from forceharness.primitives.library import _yaw_deg_xyzw
from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real import primitives as cp
from forceharness.real.plant import DemoPort

from forceharness.real import runner as free
from real_helpers import run_task

from test_search_hop import TOP, board

DOWN = np.array([0., 0., -1.])


def port(p, yaw_ok=2., side=(.0003, 0.), tol_deg=1., tol_m=.0004, depth=.01):
    """The mouth lets the plug in (and holds it sideways) once it was within tol of yaw_ok and side."""
    plant = p.driver
    start, yaw0, entered, original = plant.p[:2].copy(), _yaw_deg_xyzw(plant.r), [False], plant.update

    def update(pose, kp, kd, heartbeat):
        pose = np.array(pose, dtype=float)
        yaw = ((_yaw_deg_xyzw(pose[3:])-yaw0+180.) % 360.)-180.
        if abs(yaw-yaw_ok) < tol_deg and np.linalg.norm(pose[:2]-start-np.array(side)) < tol_m:
            entered[0] = True
        if entered[0]:
            pose[:2] = plant.p[:2]                  # the port walls hold it sideways
        plant.plane = TOP-depth if entered[0] else TOP
        return original(pose, kp, kd, heartbeat)
    plant.update = update
    return entered


def touched(p, b):
    prim = ContinuousPrimitives(b)
    prim.call('move_until_contact', direction=[0, 0, -1], max_dist=.003, f_threshold=.5)
    return prim


def test_rocking_finds_the_lined_up_pose_and_presses_it_home():
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        prim = touched(p, b)
        entered = port(p)
        rep = prim.call('wiggle')
        assert not b.fault
    assert entered[0] and rep.termination == 'seated' and rep.derived['surface_known']
    assert rep.derived['depth_below_surface_m'] >= .0095 and rep.derived['held_sideways_after_s'] is not None


def test_a_part_already_in_is_judged_from_the_surface_and_pressed_past_the_detent():
    # a search can leave the plug ~6 mm below the strip; the last mm needs a harder press
    def run(force):
        p = DemoPort(plane=TOP)
        with p.backend() as b:
            prim = touched(p, b)
            surface = np.asarray(b.eef_pos).copy()
            plant, original = p.driver, p.driver.update
            plant.plane = TOP-.006
            prim.call('move_until_contact', direction=[0, 0, -1], max_dist=.008, f_threshold=.5)
            prim.last_support = dict(pos=surface, normal=DOWN)        # what the slide search records

            def update(pose, kp, kd, heartbeat):
                pose = np.array(pose, dtype=float)
                pose[:2] = plant.p[:2]                                 # in the port: held sideways
                if (plant.plane-pose[2])*kp[2] > 12.:                  # the detent gives at 12 N
                    plant.plane = TOP-.0095
                return original(pose, kp, kd, heartbeat)
            plant.update = update
            rep = prim.call('wiggle', force=force)
            assert not b.fault
        return rep
    weak, strong = run(10.), run(18.)
    assert weak.termination == 'jammed' and weak.derived['depth_below_surface_m'] == pytest.approx(.006, abs=.0006)
    assert strong.termination == 'seated' and strong.derived['depth_below_surface_m'] >= .009
    assert strong.derived['advance_m'] < .008                          # its own advance alone would have failed


def test_held_on_the_mouth_without_a_hole_is_jammed():
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        prim = touched(p, b)
        board(p, hole=None, stick=.02)
        rep = prim.call('wiggle')
        assert not b.fault
    assert rep.termination == 'jammed' and '180' in rep.derived['note']
    assert rep.derived['held_sideways_after_s'] == pytest.approx(1., abs=.3)


def test_out_of_reach_reports_not_seated():
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        prim = touched(p, b)
        entered = port(p, yaw_ok=20.)                  # e.g. the plug is turned too far (or upside down)
        rep = prim.call('wiggle', duration=10.)
    assert not entered[0] and rep.termination == 'not_seated' and not rep.derived['seated']


def test_shift_zero_keeps_rocking_a_part_held_sideways():
    # a circle on a pressed plug freezes the rocking early, so the screw task rocks without one
    def held(**kw):
        p = DemoPort(plane=TOP)
        with p.backend() as b:
            prim = touched(p, b)
            port(p, yaw_ok=0., side=(0., 0.), tol_deg=90., tol_m=.01, depth=.002)
            rep = prim.call('wiggle', duration=8., **kw)
            assert not b.fault
        return rep
    circling, rocking = held(), held(shift=0., yaw_deg=5.)
    assert circling.derived['held_sideways_after_s'] is not None and circling.termination == 'jammed'
    assert rocking.derived['held_sideways_after_s'] is None and rocking.derived['yaw_amplitude_deg'] == 5.
    assert rocking.termination == 'not_seated' and rocking.duration >= 8.


def test_bad_arguments_are_refused():
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        for bad in (dict(yaw_deg=6.), dict(shift=.003), dict(force=25.), dict(depth=.03), dict(duration=40.)):
            with pytest.raises(ValueError):
                prim.call('wiggle', **bad)


@pytest.mark.parametrize('task_id', ['usb_insert', 'screw_insert'])
def test_only_the_tasks_that_add_it_offer_it_to_condition_a(tmp_path, task_id):
    assert 'wiggle' not in free.CONDITIONS['A']['menu'] and 'wiggle' not in free.CONDITIONS['B']['menu']
    run_task(task_id, ['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay')])
    assert 'wiggle(' in (tmp_path/'relay'/'obs_0.md').read_text()
    assert 'wiggle' not in free.CONDITIONS['A']['menu']                # the module-level menu is never changed
    run_task(task_id, ['--condition', 'B', '--out', str(tmp_path/'b'), '--relay-dir', str(tmp_path/'b_relay')])
    assert 'wiggle(' not in (tmp_path/'b_relay'/'obs_0.md').read_text()
