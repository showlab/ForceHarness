"""Joint 7 stays within -0.8..2.4 rad: turning it further pulls the wrist cable tight and the robot side's IK
continuity check ends the session. The clip inherited from simulation assumed d(j7) = +d(yaw) and let a clockwise turn
through. Offline only."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from forceharness.real import runner as free
from forceharness.primitives.library import Primitives
from forceharness.real.primitives import WRIST_Q7_RAD, wrist_turn_range
from test_screw import held_on_surface

# robot-side states of a logged screw episode: start of the last screw, and the fault
BEFORE = dict(joints=[-.17361, -.22993, .14780, -2.67256, .12881, 2.41165, 2.24764],
              flange_pos=[.40298, -.01205, .30798], flange_quat=[-.37370, .92707, .02976, -.00289])
STUCK = dict(joints=[-.22910, -.23911, .20179, -2.68651, .17068, 2.43906, 2.67310],
             flange_pos=[.40115, -.01407, .30762], flange_quat=[-.15151, .98796, .03124, .00291])


def nuc_backend(state):
    tcp = np.asarray(state['flange_pos'])+free.R.from_quat(state['flange_quat']).apply(free.TCP_TRANSLATION)
    return SimpleNamespace(_state=state, eef_pos=tcp, eef_quat=np.asarray(state['flange_quat']))


def test_real_range_turns_the_right_way():
    lo, hi = wrist_turn_range(2.24764)                 # clockwise (negative) raises joint 7
    assert lo == pytest.approx(math.degrees(2.24764-WRIST_Q7_RAD[1])) and -10 < lo < -8
    assert hi == pytest.approx(math.degrees(2.24764-WRIST_Q7_RAD[0]))
    assert wrist_turn_range(2.6731)[0] == 0 < wrist_turn_range(2.6731)[1]   # past the end: only back
    assert wrist_turn_range(-1.)[1] == 0 > wrist_turn_range(-1.)[0]


def test_sim_clip_is_unchanged():
    sim = SimpleNamespace(sim=SimpleNamespace(wrist_joint_rad=.5))
    assert Primitives._wrist_turn_range(sim) == pytest.approx((-math.degrees(3.1), math.degrees(2.1)))


def test_runner_refuses_the_turn_that_faulted():
    b = nuc_backend(BEFORE)
    for name in ('screw', 'rotate_z'):
        with pytest.raises(ValueError, match='the wrist') as e:
            free.validate(name, dict(angle_deg=-30), b, 'A')
        left = float(e.value.args[0].split('at most about ')[1].split('°')[0])
        assert 5 <= left <= 12                         # ~9 deg clockwise were left up to 2.4 rad
    assert free.validate('rotate_z', dict(angle_deg=90), b, 'A')[0] == dict(angle_deg=90)


def test_stuck_wrist_may_turn_back_and_move():
    b = nuc_backend(STUCK)
    with pytest.raises(ValueError, match='the wrist'):
        free.validate('screw', dict(angle_deg=-5), b, 'A')
    assert free.validate('rotate_z', dict(angle_deg=90), b, 'A')[0] == dict(angle_deg=90)
    assert free.validate('lift', dict(height=.02), b, 'A')[0]['height'] == .02


@pytest.mark.parametrize('name', ['screw', 'rotate_z'])
def test_primitive_clips_at_the_wrist_end(name):
    p, b, prim = held_on_surface()
    p.driver.q[6] = 2.3
    b._request('state')
    rep = prim.call(name, angle_deg=-30)
    assert rep.derived['wrist_limit_clipped'] and rep.derived['executed_deg'] == pytest.approx(-math.degrees(.1))
    assert rep.yaw_deg == pytest.approx(-math.degrees(.1), abs=1.)
    rep = prim.call(name, angle_deg=30)                 # back the other way is not clipped
    b.__exit__(None, None, None)
    assert not rep.derived.get('wrist_limit_clipped') and rep.yaw_deg == pytest.approx(30, abs=1.)
