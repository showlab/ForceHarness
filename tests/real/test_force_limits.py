"""A task may raise the contact-force targets and insert up to 0.8 x the 25 N raw cap; the cap and every other guard
are unchanged, and tasks that do not ask keep 10 N / 12 N. Offline only."""
import json

import pytest

from forceharness.real import runner as free
from forceharness.real import validation

GRIP = free.GRIP_RULES['auto']


def test_defaults_are_unchanged_and_the_menu_says_so():
    menu = free.menu_text('A', GRIP)
    assert 'f_threshold (default 5 N, ≤ 10)' in menu and 'normal_force (≤ 10 N' in menu and 'max_force (≤ 12 N' in menu
    assert free.TARGET_FORCE_CEILING_N == pytest.approx(.8*free.RAW_FORCE_CAP_N) == 20


class Hand:
    eef_pos = [.40, 0., .20]


def test_a_task_can_ask_for_up_to_twenty_newtons(tmp_path, monkeypatch):
    limits = dict(contact=20., insert=20.)
    menu = free.menu_text('A', GRIP, limits)
    assert 'normal_force (≤ 20 N' in menu and 'target_force (≤ 20 N)' in menu and 'max_force (≤ 20 N' in menu
    monkeypatch.setattr(validation, '_reach_check', lambda *args, **kwargs: None)     # only the force range is under test
    slide = dict(direction=[1, 0, 0], distance=.05, normal_force=18)
    with pytest.raises(ValueError, match='normal_force'):
        free.validate('slide', slide, Hand(), 'A')                    # 18 N is above the default 10 N
    assert free.validate('slide', slide, Hand(), 'A', limits)[0]['normal_force'] == 18
    assert free.validate('insert', dict(max_force=18), Hand(), 'A', limits)[0]['max_force'] == 18
    code = free.main(['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'), '--instruction', 'x',
                      '--force-max', '20', '--insert-force-max', '20'])
    r = json.loads((tmp_path/'ep'/'episode.json').read_text())
    assert code == 0 and r['force_limits_N'] == dict(limits, slide_default=1.)
    assert '≤ 20 N' in (tmp_path/'relay'/'obs_0.md').read_text()


@pytest.mark.parametrize('flag,value', [('--force-max', '25'), ('--insert-force-max', '21'), ('--force-max', '5')])
def test_limits_stay_between_the_old_default_and_the_ceiling(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        free.main(['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'), '--instruction', 'x', flag, value])


def test_a_task_can_start_slide_at_a_higher_default(monkeypatch):
    # peeling starts slide at 10 N
    limits = dict(contact=20., insert=12., slide_default=10.)
    assert 'default 10) while moving' in free.menu_text('A', GRIP, limits)
    assert 'default 1) while moving' in free.menu_text('A', GRIP)
    monkeypatch.setattr(validation, '_reach_check', lambda *args, **kwargs: None)
    sent = dict(direction=[1, 0, 0], distance=.05)
    assert free.validate('slide', sent, Hand(), 'A', limits)[0]['normal_force'] == 10.
    assert free.validate('slide', sent, Hand(), 'A')[0]['normal_force'] == 1.         # other tasks: unchanged
    assert 'normal_force' not in sent                                                   # the model's call is kept as sent
    assert free.validate('slide', dict(sent, normal_force=4), Hand(), 'A', limits)[0]['normal_force'] == 4


def test_the_slide_default_stays_within_the_contact_limit(tmp_path):
    with pytest.raises(SystemExit):
        free.main(['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'), '--instruction', 'x',
                   '--slide-force-default', '12'])                                      # above the default 10 N limit
