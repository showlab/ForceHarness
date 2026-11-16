"""The 20 N pull through the offline robot-side guard, without robot or network access."""
import numpy as np
import pytest

from forceharness.real import runner as free


@pytest.mark.parametrize('direction', [[0, 0, 1], [1, 0, 0]])
def test_twenty_newton_pull_is_not_clamped_or_stopped_at_old_fifteen(direction):
    backend, _, _ = free.demo()
    port = backend.port
    update = port.driver.update

    def resisting_handle(*args):
        update(*args)
        port.driver.w[:3] = -20*np.asarray(direction, dtype=float)

    with backend:
        backend.command_gripper(True, confirmed=True)
        port.driver.frozen = True
        port.driver.update = resisting_handle
        prim = free.FreePrimitives(backend, tick=lambda: None)
        args, _ = free.validate('pull', dict(direction=direction, force=20, distance=.005), backend, 'A')
        rep = prim.call('pull', **args, max_time=3.)
        assert rep.params['force'] == 20
        assert rep.termination == 'stalled' and backend.fault is None
        assert backend.config.f_cap == port.limits['f_cap'] == 25
        assert backend._closed is True
    assert port.driver.grips == [True] and not port.driver.running


def test_default_pull_uses_eight_newtons_in_execution():
    backend, _, _ = free.demo()
    with backend:
        backend.command_gripper(True, confirmed=True)
        prim = free.FreePrimitives(backend, tick=lambda: None)
        args, _ = free.validate('pull', dict(direction=[0, 0, 1], distance=.001), backend, 'A')
        rep = prim.call('pull', **args)
        assert rep.termination == 'completed' and rep.params['force'] == 8


@pytest.mark.parametrize('direction', [[0, 0, 1], [1, 0, 0]])
def test_force_above_twenty_is_refused_before_motion(direction):
    backend, _, _ = free.demo()
    with backend:
        args = dict(direction=direction, force=20.1, distance=.005)
        with pytest.raises(ValueError, match='force'):
            free.validate('pull', args, backend, 'A')
        prim = free.FreePrimitives(backend, tick=lambda: None)
        with pytest.raises(ValueError, match='force'):
            prim.call('pull', **args)
        assert backend.port.driver.starts == 0
        with pytest.raises(ValueError, match='target_force'):
            free.validate('push', dict(direction=direction, target_force=15.1), backend, 'A')
        with pytest.raises(ValueError, match='not on the menu'):
            free.validate('pull', args, backend, 'B')


def test_raw_twenty_five_newton_guard_still_aborts_without_opening_gripper():
    backend, _, _ = free.demo()
    port = backend.port
    update = port.driver.update

    def overload(*args):
        update(*args)
        port.driver.w[2] = -25.1

    with backend:
        backend.command_gripper(True, confirmed=True)
        port.driver.update = overload
        prim = free.FreePrimitives(backend, tick=lambda: None)
        with pytest.raises(RuntimeError, match='cap'):
            prim.call('pull', direction=[0, 0, 1], force=20, distance=.005)
        assert backend.fault and backend._closed is True
    assert port.driver.grips == [True] and not port.driver.running
