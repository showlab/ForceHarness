"""The 8 mm/s target bound applies over the real time between steps (capped at 80 ms), not per nominal 50 ms step,
so free motion is not held to about 5 mm/s."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real.plant import DemoPort, core

OFFSET = np.array([0, 0, .1185])


def tcp(desired):
    return desired[:3]+R.from_quat(desired[3:]).apply(OFFSET)


def wait(p, s, seconds):
    for _ in range(round(seconds/.01)):
        p.clock.sleep(.01); s.tick()


def test_target_advances_8mm_s_over_the_real_interval_capped_at_the_window():
    p = DemoPort(); s = p.session; s.begin(30.)
    old = s.desired.copy()
    new = s.command([.03, 0, 0], [0, 0, 0], OFFSET, [0, 0, 0, 1])
    assert np.linalg.norm(tcp(new)-tcp(old)) == pytest.approx(.0004, abs=1e-7)   # first step: one 50 ms step as before
    for gap, expected in ((.07, .00056), (.03, .00024), (.2, .008*core.RATE_WINDOW_S)):
        wait(p, s, gap)
        old = s.desired.copy()
        new = s.command([.03, 0, 0], [0, 0, 0], OFFSET, [0, 0, 0, 1])
        assert np.linalg.norm(tcp(new)-tcp(old)) == pytest.approx(expected, abs=1e-7)


def test_rotation_shares_the_bound_and_uses_all_of_it():
    p = DemoPort(); s = p.session; s.begin(30.)
    s.command([0, 0, 0], [0, 0, 0], OFFSET, [0, 0, 0, 1])
    wait(p, s, .05)
    old = s.desired.copy()
    new = s.command([.03, 0, 0], [0, .05, 0], OFFSET, [0, 0, 0, 1])
    moved = max(np.linalg.norm(new[:3]-old[:3]), np.linalg.norm(tcp(new)-tcp(old)))
    assert moved == pytest.approx(.0004, rel=1e-3)          # the old 1, .8, .64 ... search stopped near .32 mm


class SlowLink(DemoPort):
    """Steps every 70 ms like the real workstation-to-robot round trip, not 50 ms."""
    def request(self, op, **args):
        out = super().request(op, **args)
        if op == 'step':
            wait(self, self.session, .02)
        return out


def test_move_by_runs_near_8mm_s_with_70ms_steps_and_never_faster():
    events = []
    p = SlowLink(emit=events.append)
    with p.backend(keep_controller=True) as b:
        prim = ContinuousPrimitives(b)
        t0 = p.clock()
        rep = prim.move_by(.05, 0, 0)
        elapsed = p.clock()-t0
    assert rep.termination == 'completed'
    targets = [(e['time'], tcp(np.array(e['desired_flange']))) for e in events if e['type'] == 'target']
    speeds = [np.linalg.norm(b2-b1)/(t2-t1) for (t1, b1), (t2, b2) in zip(targets, targets[1:]) if t2 > t1]
    assert max(speeds) <= .008+1e-6
    assert elapsed < 8.                                     # was 0.4 mm per 70 ms step: ~8.8 s of travel alone
