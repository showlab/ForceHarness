"""Straight 3-D pull with a nonzero tracking error; offline fixtures, never real motion.

The deadband is a counterexample to a target-distance cap, not a fitted drawer model or evidence that the hardware
will reach the requested travel.
"""
import numpy as np
import pytest

from forceharness.real.plant import DemoPort
from forceharness.real.primitives import ContinuousPrimitives


def deadband_follower(port, direction, lag):
    """Only motion beyond the tracking deadband moves the synthetic hand."""
    u = np.asarray(direction, dtype=float)
    u /= np.linalg.norm(u)
    original = port.driver.update
    samples = []

    def update(pose, kp, kd, token):
        before = port.driver.p.copy()
        original(pose, kp, kd, token)
        along = float((pose[:3]-before)@u)
        displacement = np.sign(along)*max(0., abs(along)-lag)
        port.driver.p = before+u*displacement
        lead = pose[:3]-port.driver.p
        port.driver.w[:3] = port.driver.bias[:3]-750*lead
        samples.append((pose[:3].copy(), port.driver.p.copy()))

    port.driver.update = update
    return u, samples


@pytest.mark.parametrize('direction', [[0, 0, 1], [0, 0, -1], [1, 0, 1], [1, -1, -.5]])
@pytest.mark.parametrize('lag,force', [(.003, 3.), (.005, 5.)])
def test_requested_actual_travel_is_reached_despite_tracking_deadband(direction, lag, force):
    port = DemoPort(payload=0)
    u, samples = deadband_follower(port, direction, lag)
    request = port.request

    def with_latency(op, **args):
        reply = request(op, **args)
        if op == 'step':
            port.clock.sleep(.02)  # real-like 70 ms samples, not ideal 50 ms
        return reply

    port.request = with_latency
    with port.backend() as backend:
        backend.command_gripper(True, confirmed=True)
        primitive = ContinuousPrimitives(backend)
        start, quat = backend.eef_pos.copy(), backend.eef_quat.copy()
        rep = primitive.call('pull', direction=direction, force=force, distance=.005, speed=.002)
        delta = backend.eef_pos-start
        assert rep.termination == 'completed' and backend.fault is None
        assert .0048-1e-9 <= delta@u <= .005+1e-9
        np.testing.assert_allclose(delta-u*(delta@u), 0, atol=1e-9)
        np.testing.assert_allclose(backend.eef_quat, quat, atol=1e-9)
        assert rep.derived['pull_distance_reference'] == 'measured_tcp'
        assert rep.derived['pull_target_travel_max_m'] > .005
        assert rep.derived['pull_target_lead_limit_m'] == .006
        assert rep.derived['pull_target_lead_max_m'] <= .006+1e-9

    # Leading the measured point does not permit increasing speed or lead.
    targets, actual = (np.array(x) for x in zip(*samples))
    assert np.max(np.linalg.norm(targets-actual, axis=1)) <= .006+1e-9
    assert np.max(np.diff(targets, axis=0)@u) <= .002*.05+1e-9
    assert rep.derived['pull_target_travel_max_m'] <= .005+.006+1e-9
    # On reaching actual travel, unload the lead instead of continuing to pull.
    np.testing.assert_allclose(targets[-1], actual[-1], atol=1e-9)
    assert not port.driver.running and port.driver.grips == [True]


def test_extended_target_is_checked_against_existing_workspace_before_motion():
    port = DemoPort(payload=0)
    port.driver.p[2] = .545
    with port.backend() as backend:
        primitive = ContinuousPrimitives(backend)
        with pytest.raises(ValueError, match='workspace'):
            primitive.pull([0, 0, 1], force=3, distance=.002)
    assert port.driver.starts == 0


def test_stuck_mechanism_cannot_exceed_contact_lead_to_reach_force():
    port = DemoPort(payload=0)
    port.driver.frozen = True
    original = port.driver.update
    leads = []

    def spring(pose, kp, kd, token):
        original(pose, kp, kd, token)
        lead = pose[:3]-port.driver.p
        leads.append(np.linalg.norm(lead))
        port.driver.w[:3] = port.driver.bias[:3]-300*lead

    port.driver.update = spring
    with port.backend() as backend:
        backend.command_gripper(True, confirmed=True)
        rep = ContinuousPrimitives(backend).call(
            'pull', direction=[0, 0, 1], force=5, distance=.002, max_time=3.)
        assert rep.termination == 'timeout' and backend.fault is None
        np.testing.assert_allclose(rep.actual_disp, 0, atol=1e-9)
    assert max(leads) == pytest.approx(.006)
    assert not port.driver.running and port.driver.grips == [True]
