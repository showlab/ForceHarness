"""A screw pinched by two pads pivots in the grip: holding the hand rigidly sideways while turning built up the side
load and tipped it, and a fully free hand let the press tip it over. Pressing holds sideways, turning is soft (70%
stiffness) and follows the thread. Offline only."""
import numpy as np
import pytest

from forceharness.primitives.library import _yaw_deg_xyzw
from forceharness.real import primitives as cp
from test_screw import held_on_surface

K = 750.          # demo plant stiffness, N/m


def hole_pulls_sideways(port, mm_per_30deg=2.):
    """The hole wants the hand mm_per_30deg further +x per 30 deg turned (an off-axis screw); hand and
    controller target meet halfway, like two equal springs. Returns the per-update side loads (N)."""
    plant = port.driver
    yaw0, x0 = _yaw_deg_xyzw(plant.r), float(plant.p[0])
    original, loads = plant.update, []

    def update(pose, kp, kd, heartbeat):
        out = original(pose, kp, kd, heartbeat)
        turned = abs(((_yaw_deg_xyzw(np.asarray(pose[3:]))-yaw0+180.) % 360.)-180.)
        hole_x = x0+mm_per_30deg/1000.*turned/30.
        plant.p[0] = (float(pose[0])+hole_x)/2.
        loads.append(K*abs(plant.p[0]-float(pose[0])))
        return out
    plant.update = update
    return loads


def test_hand_follows_the_screw_sideways():
    p, b, prim = held_on_surface()
    x0 = b.eef_pos[0]
    loads = hole_pulls_sideways(p)
    rep = prim.call('screw', angle_deg=-30)
    b.__exit__(None, None, None)
    # 70% stiffness while turning: between held stiffly (1.0 mm of the 2) and fully free (2.0 mm)
    assert .00105 < b.eef_pos[0]-x0 < .0019
    assert rep.derived['hand_followed_sideways_m'] > .001 and not b.fault


def test_press_holds_sideways_so_the_screw_cannot_tip():
    p, b, prim = held_on_surface()
    plant, original, x_cmd = p.driver, p.driver.update, []

    def shoved(pose, kp, kd, heartbeat):
        out = original(pose, kp, kd, heartbeat)
        plant.p[0] = float(pose[0])+.001                 # a tipping screw drags the hand 1 mm sideways
        x_cmd.append(float(pose[0]))
        return out
    plant.update = shoved
    rep = prim.call('screw', angle_deg=-30)
    b.__exit__(None, None, None)
    k = int(rep.derived['press_s']/b.dt)
    assert np.ptp(x_cmd[:k-2]) < 2e-4                     # pressing: the target keeps the start
    assert x_cmd[-1]-x_cmd[k] > .001                      # turning: the soft target drifts with the hand


def test_full_push_presses_then_turning_eases_to_three_newton():
    p, b, prim = held_on_surface()
    rep = prim.call('screw', angle_deg=-30, push=8)
    b.__exit__(None, None, None)
    fz = np.asarray(rep.wrench_rows)[:, 3]
    assert rep.derived['turn_push_N'] == [8., 8.]          # a full press turns at the full push
    assert abs(fz.max()-fz[-5:].mean()) < 1.5             # no drop to a light turning push any more
    assert rep.derived['turn_sink_m'] == pytest.approx(.00025, abs=.00005)
