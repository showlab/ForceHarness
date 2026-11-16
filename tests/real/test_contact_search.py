"""One move_until_contact may search up to 0.30 m, run as segments of at most 0.10 m (each its own robot-side motion
with every guard), with a fresh still zero before each held segment. Offline only."""
import json

import pytest

from forceharness.real import runner as free
from forceharness.real import validation


class Hand:
    eef_pos = [.40, 0., .25]


def test_validate_allows_a_long_search_checked_per_segment(monkeypatch):
    seen = []
    monkeypatch.setattr(validation, '_reach_check', lambda backend, delta, what, segment_m=None: seen.append(segment_m))
    args = dict(direction=[0, 0, -1], max_dist=.25)
    assert free.validate('move_until_contact', args, Hand(), 'A')[0]['max_dist'] == .25
    assert seen == [free.MOVE_MAX_M]                       # the joint-travel check restarts with each robot-side motion
    with pytest.raises(ValueError, match='max_dist'):
        free.validate('move_until_contact', dict(args, max_dist=.31), Hand(), 'A')
    with pytest.raises(ValueError, match='amplitude'):     # probe is unchanged
        free.validate('probe', dict(direction=[0, 0, -1], amplitude=.15), Hand(), 'A')
    assert '≤ 0.30 m' in free.menu_text('A', free.GRIP_RULES['auto'])


SURFACE_Z = .08      # a raised work surface (TCP z), inside a segment; the synthetic table is only 3.5 mm above the floor


def run(tmp_path, monkeypatch, calls, width=None, extra=()):
    original = free.demo

    def plant():
        b, world, _ = original()
        world.objects = []
        b.port.driver.plane = SURFACE_Z+free.TCP_TRANSLATION[2]
        if width is not None:
            b.port.driver.width = width
        return b, world, calls
    monkeypatch.setattr(free, 'demo', plant)
    code = free.main(['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'), '--instruction', 'x', *extra])
    return code, json.loads((tmp_path/'ep'/'episode.json').read_text())


def test_one_call_comes_down_to_the_table_in_segments(tmp_path, monkeypatch):
    # the synthetic surface tops out near 4.5 N, so 3 N is the threshold here (an unsegmented call behaves the same)
    calls = [dict(name='move_until_contact', args=dict(direction=[0, 0, -1], max_dist=.25, f_threshold=3)),
             dict(name='done', args=dict(success=True, note='x'))]
    code, r = run(tmp_path, monkeypatch, calls)
    assert code == 0 and r['fault'] is None and r['actions'][0]['termination'] == 'contact'
    rep = json.loads((tmp_path/'ep'/'action_00.json').read_text())
    segments = rep['derived']['segments']
    assert len(segments) == 2 and [s['termination'] for s in segments] == ['no_contact_within_max_dist', 'contact']
    assert rep['start_pos'][2] == pytest.approx(.25, abs=.005) and rep['end_pos'][2] == pytest.approx(SURFACE_Z, abs=.01)
    assert rep['params']['max_dist'] == .25


def test_a_held_search_reads_the_zero_again_before_each_segment(tmp_path, monkeypatch):
    calls = [dict(name='move_until_contact', args=dict(direction=[0, 0, -1], max_dist=.25)),
             dict(name='done', args=dict(success=True, note='x'))]
    code, r = run(tmp_path, monkeypatch, calls, width=.0196, extra=('--resume-held', 'peeler'))
    rep = json.loads((tmp_path/'ep'/'action_00.json').read_text())
    assert code == 0 and len(rep['derived']['segments']) >= 2
    assert len(rep['derived']['held_zero_shifts_N']) == len(rep['derived']['segments'])-1
