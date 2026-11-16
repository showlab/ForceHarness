"""The robot-side gripper width checks can be switched off for one task (a tool clamped by the operator, whose width
reading means nothing); every other guard stays on."""
import inspect
import json
import math

import numpy as np
import pytest

from forceharness.real.config import BackendConfig
from real_helpers import robot_side
from forceharness.real.transport import worker_command
from forceharness.real import runner as free
from forceharness.real.tasks import TASKS

guard = robot_side('guard')


class Driver:
    def __init__(self, width):
        self.width = width

    def metadata(self):
        return dict(default_Kx=[750.]*3+[15.]*3, default_Kxd=[37.]*3+[2.]*3,
                    joint_low=[-2.8, -1.66, -2.8, -2.97, -2.8, .44, -2.8], joint_high=[2.8, 1.66, 2.8, -.17, 2.8, 3.65, 2.8])

    def read(self, wrench=(1., 0., 0., 0., 0., 0.)):
        return dict(flange_pos=[.5, 0., .3], flange_quat=[0., 0., 0., 1.], joints=[0., 0., 0., -1.5, 0., 1.5, 0.],
                    velocities=[0.]*7, wrench=list(wrench), gripper_width=self.width, timestamp_ns=1, smin=.2, residual=.1,
                    policy_running=True)


def session(width, **limits):
    return guard.GuardedSession(Driver(width), limits=guard.Limits(**limits), clock=lambda: 0.)


def moving(g, initial_width):
    s = g.driver.read()
    g.active, g.initial, g.initial_width, g.budget = True, dict(s, flange_pos=np.array(s['flange_pos'])), initial_width, 100.
    g.desired = np.array(s['flange_pos'])
    return g


@pytest.mark.parametrize('width', [-.0001, .09])
def test_width_range_checked_by_default_and_skipped_when_off(width):
    with pytest.raises(RuntimeError, match='Invalid gripper width'):
        session(width).read_checked()
    assert session(width, width_check=0).read_checked()['gripper_width'] == width


def test_width_change_checked_by_default_and_skipped_when_off():
    with pytest.raises(RuntimeError, match='Gripper width changed'):
        moving(session(.0026), .0102).read_checked()
    assert moving(session(.0026, width_check=0), .0102).read_checked()['gripper_width'] == .0026


def test_a_reading_that_is_not_a_number_still_stops():
    with pytest.raises(RuntimeError, match='Invalid gripper width'):
        session(math.nan, width_check=0).read_checked()


def test_turning_it_off_leaves_the_force_guard_on():
    g = session(-.001, width_check=0)
    g.driver.read = lambda wrench=(26., 0., 0., 0., 0., 0.): Driver.read(g.driver, wrench)
    with pytest.raises(RuntimeError, match='Raw force/torque cap'):
        g.read_checked()


@pytest.mark.parametrize('value', [-1, True, math.nan, 2.])
def test_only_zero_or_the_default_is_accepted(value):
    with pytest.raises(ValueError):
        guard.Limits(width_check=value)
    with pytest.raises(ValueError):
        guard.Limits(t_cap=0)                     # the exception to "must be positive" is this limit alone


def limits_json(config):
    command = worker_command(config)
    return json.loads(command.split('--limits-json ')[1].split("' ")[0].strip("'"))


def test_worker_gets_the_switch_only_when_a_task_asks():
    assert 'width_check' not in limits_json(BackendConfig(motion_enabled=True))
    assert limits_json(BackendConfig(motion_enabled=True, gripper_width_check=False))['width_check'] == 0
    with pytest.raises(ValueError):
        BackendConfig(gripper_width_check=0)


def test_only_the_peel_task_turns_it_off():
    assert TASKS['peel_vegetable'].gripper_width_check is False
    assert all(task.gripper_width_check for name, task in TASKS.items() if name != 'peel_vegetable')


class Resume:
    resume_holding = free.free_backend(object).resume_holding
    fault, active, _closed = None, False, None

    def __init__(self, width, check):
        self.gripper_opening, self.config = width, BackendConfig(gripper_width_check=check)


def test_resume_accepts_any_closed_reading_only_when_off():
    assert Resume(-.00001, False).resume_holding() == -.00001
    with pytest.raises(free.BackendError, match='does not look like holding'):
        Resume(-.00001, True).resume_holding()
    with pytest.raises(free.BackendError, match='does not look like holding'):
        Resume(.079, False).resume_holding()     # wide open still is not holding anything
