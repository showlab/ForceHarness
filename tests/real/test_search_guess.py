"""search in hop mode with a guess: the model may say roughly where the hole is relative to the hand; the search
probes the start once as the surface reference, then spirals out from the guess. Offline only."""
import numpy as np
import pytest

from forceharness.real import runner as free
from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real import primitives as cp
from forceharness.real.plant import DemoPort
from test_search_hop import TOP, board


@pytest.fixture(autouse=True)
def demo_probe_force(monkeypatch):
    monkeypatch.setattr(cp, 'HOP_PROBE_N', 4.)       # the demo plant is 750 N/m: 6 N is its 8 mm-lead cap


def search(**args):
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        prim.call('move_until_contact', direction=[0, 0, -1], max_dist=.003, f_threshold=.5)
        centre = board(p, hole=(.003, 0.))
        rep = prim.call('search', mode='hop', **args)
        assert not b.fault
    return rep, centre


def test_a_good_guess_finds_it_in_a_few_probes():
    rep, _ = search(guess=[.003, 0.])
    plain, _ = search()
    assert rep.termination == 'dropped' and rep.derived['probes'] <= 2      # the start, then the guess
    assert np.linalg.norm(rep.derived['found_from_guess_m']) < .0015
    assert plain.termination == 'dropped' and plain.derived['probes'] > rep.derived['probes']


def test_hop_defaults_are_the_screw_tuned_ones():
    rep, _ = search()
    d = cp.SEARCH_DEFAULTS['hop']
    assert [rep.params[k] for k in ('radius_max', 'pitch', 'drop', 'hop')] == [d['radius_max'], d['pitch'], d['drop'], d['hop']]
    assert rep.derived['probe_N'] == 4.


def test_guess_is_for_hop_only():
    p = DemoPort(plane=TOP)
    with p.backend() as b:
        with pytest.raises(ValueError, match='hop'):
            ContinuousPrimitives(b).call('search', guess=[.001, 0.])


class Hand:
    eef_pos = [.40, 0., .25]
    _state = {}


def test_runner_validates_guess():
    assert 'guess?' in free.signature('search', 'A')
    assert free.validate('search', dict(mode='hop', guess=[.003, -.001]), Hand(), 'A')[0]['guess'] == [.003, -.001]
    for args, msg in ((dict(guess=[.003, 0]), 'hop'), (dict(mode='hop', guess=[.03, 0]), '0.02'),
                      (dict(mode='hop', guess=[.003]), 'two finite numbers')):
        with pytest.raises(ValueError, match=msg):
            free.validate('search', args, Hand(), 'A')
