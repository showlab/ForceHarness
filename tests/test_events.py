import numpy as np

from forceharness.primitives.events import EventDetector


def feed(det, forces, cmd=(0, 0, -0.002), act=None, grip=False, t0=0.0, dt=0.05):
    evs = []
    for i, f in enumerate(forces):
        f = np.asarray(f, dtype=float)
        a = np.asarray(act if act is not None else cmd, dtype=float)
        evs += det.update(t0 + i * dt, f, np.zeros(3), np.array([0, 0, 1.0]), np.asarray(cmd, float), a, grip)
    return evs


def test_no_contact_no_events():
    det = EventDetector()
    assert feed(det, [[0, 0, 0.3]] * 20) == []


def test_contact_start_and_end():
    det = EventDetector(f_contact=2.0, f_release=1.0)
    forces = [[0, 0, 0]] * 5 + [[0, 0, 5.0]] * 5 + [[0, 0, 0]] * 5
    evs = feed(det, forces)
    kinds = [e.kind for e in evs]
    assert kinds[0] == "contact_start" and "contact_end" in kinds
    assert "+z" in evs[0].detail


def test_force_cap():
    det = EventDetector(f_cap=25.0)
    evs = feed(det, [[0, 0, 0]] * 3 + [[0, 0, 30.0]] * 3)
    assert any(e.kind == "force_cap" for e in evs)


def test_stall_when_commanded_but_not_moving():
    det = EventDetector(stall_window=6)
    forces = [[0, 0, 8.0]] * 20
    evs = feed(det, forces, cmd=(0, 0, -0.003), act=(0, 0, -0.0002))
    assert any(e.kind == "stall" for e in evs)
    assert sum(e.kind == "stall" for e in evs) == 1  # reported only once


def test_resistance_drop():
    det = EventDetector(breakaway_min_peak=4.0, breakaway_drop=0.35, filter_n=1)
    forces = [[0, 0, 0]] * 3 + [[0, 0, 3.0], [0, 0, 6.0], [0, 0, 9.0], [0, 0, 9.5], [0, 0, 5.0], [0, 0, 4.0], [0, 0, 3.5]]
    evs = feed(det, forces, cmd=(0, 0, -0.002), act=(0, 0, -0.002))
    assert any(e.kind == "resistance_drop" for e in evs)


def test_slip_when_holding():
    det = EventDetector(slip_window=6, slip_drop=0.5)
    forces = [[0, 0, -6.0]] * 8 + [[0, 0, -0.5]] * 3
    evs = feed(det, forces, cmd=(0, 0, 0), act=(0, 0, 0), grip=True)
    assert any(e.kind == "slip" for e in evs)
