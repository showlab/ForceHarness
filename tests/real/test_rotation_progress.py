"""Regression: moving pivot turns must not look like a stuck wrist."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from forceharness.real.rotation_progress import RotationProgress

IDENTITY = [0., 0., 0., 1.]
FIXTURE = Path(__file__).parent/'fixtures/pivot_turn_traces.json'


@pytest.mark.parametrize('timing', [{}, dict(window_s=2., earliest_s=2.5)])
def test_recorded_pivot_turns_do_not_stall_even_with_the_original_timing(timing):
    for trace in json.loads(FIXTURE.read_text())['traces']:
        params, samples = trace['params'], trace['samples']
        guard = RotationProgress(axis=params['axis'], pivot=params['pivot'], **timing)
        q0 = R.from_quat(samples[0]['quat'])
        old_window, old_stops = [], []
        for sample in samples:
            t, pos, quat = sample['t'], sample['pos'], sample['quat']
            assert not guard.observe(t, pos, quat)
            turned = np.degrees((R.from_quat(quat)*q0.inv()).magnitude())
            old_window.append((t, turned))
            old_window = [row for row in old_window if row[0] >= t-2.]
            if t >= 2.5 and turned < abs(params['angle_deg'])-1 and turned-old_window[0][1] < .3:
                old_stops.append(t)
        assert old_stops, 'The fixture must reproduce the original false stall'
        assert guard.metrics['arc_span_mm'] > 5


def test_slow_arc_with_stationary_wrist_keeps_progressing():
    guard = RotationProgress(axis=[0, 1, 0], pivot=[0, 0, 0])
    for t in np.arange(0, 8, .05):
        pos = R.from_euler('y', .1*t, degrees=True).apply([-.15, 0, 0])
        assert not guard.observe(t, pos, IDENTITY)
    assert .5 < guard.metrics['arc_span_mm'] < 1


def test_wrist_correction_crossing_the_start_is_motion():
    guard = RotationProgress()
    for t in np.arange(0, 3.5, .05):
        # First rotate away, then back past the start: subtracting unsigned
        # angles gives a negative value despite continuous physical movement.
        deg = t if t <= 1.5 else 3-t
        assert not guard.observe(t, [0, 0, 0], R.from_euler('y', deg, degrees=True).as_quat())
    assert guard.metrics['wrist_span_deg'] > 1


def test_tiny_sensor_noise_does_not_hide_a_truly_blocked_turn():
    guard = RotationProgress(axis=[0, 1, 0], pivot=[0, 0, 0])
    for t in np.arange(0, 4, .05):
        pos = [-.15, 0, 1e-5*np.sin(9*t)]
        quat = R.from_euler('y', .01*np.sin(7*t), degrees=True).as_quat()
        if guard.observe(t, pos, quat):
            assert 3.2 <= t < 3.4  # before the unchanged robot-side 4-second stall guard
            break
    else:
        pytest.fail('A blocked turn was never stopped')


def test_progress_then_block_still_stops_after_a_stationary_window():
    guard = RotationProgress(axis=[0, 1, 0], pivot=[0, 0, 0])
    for t in np.arange(0, 9, .05):
        turn = R.from_euler('y', min(t, 4.), degrees=True)
        if guard.observe(t, turn.apply([-.15, 0, 0]), turn.as_quat()):
            assert 6.7 <= t < 7.2
            break
    else:
        pytest.fail('Past progress permanently disabled the stall guard')


def test_radial_drift_does_not_count_as_pivot_progress():
    guard = RotationProgress(axis=[0, 1, 0], pivot=[0, 0, 0])
    stopped = [t for t in np.arange(0, 3.5, .05)
               if guard.observe(t, [-.15+.001*t, 0, 0], IDENTITY)]
    assert stopped and stopped[0] == pytest.approx(3.2)
