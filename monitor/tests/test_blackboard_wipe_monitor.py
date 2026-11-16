"""Wiping launch contract and separate records, without starting a robot episode."""
import json

import pytest

from conftest import make_free
from fhmon import episodes, live
from fhmon.sessions import LaunchError
from test_server_boundary import server  # noqa: F401


def test_fixed_goal_is_identical_for_both_conditions_and_live_still_requires_checklist(server):
    app, _ = server
    task = next(t for t in app.info()['tasks'] if t['id'] == 'blackboard_wipe')
    assert task['name'] == 'Wipe the whiteboard' and task['fixed_instruction'].startswith('Goal of blackboard_wipe')
    assert task['guards'][1]['value'] == 25
    req = dict(task='blackboard_wipe', conditions=['A', 'B'], mode='demo', model=dict(source='external'))
    a, b = app.sessions.plan(req)
    assert a.instruction == b.instruction == task['fixed_instruction']
    assert a.command[1].endswith('scripts/real/blackboard_wipe.py') and a.command[1] == b.command[1]
    assert '--live' not in a.command and '--execute' not in a.command
    with pytest.raises(LaunchError, match='is fixed in the task file'):
        app.sessions.plan(dict(req, instruction='a different task'))
    with pytest.raises(LaunchError, match='real-robot mode is off'):
        app.sessions.plan(dict(req, conditions=['A'], mode='live'))
    app.cfg.allow_live = True
    real = dict(req, conditions=['A'], mode='live', model=dict(source='manual'))
    with pytest.raises(LaunchError, match='checklist'):
        app.sessions.plan(real)
    planned = app.sessions.plan(dict(real, checklist=[True]*3))[0]
    assert '--live' in planned.command and '--auto-confirm' in planned.command


def test_wiping_results_are_not_mixed_with_free_tasks(runs):
    for name, condition, outcome in [('wipe_A', 'phys2_force', 'success'), ('wipe_B', 'none_guard', 'partial')]:
        folder = make_free(runs, name, condition=condition, task='Wipe the board', scored=outcome)
        path = folder/'episode.json'
        record = json.loads(path.read_text())
        record.update(task_id='blackboard_wipe')
        path.write_text(json.dumps(record, ensure_ascii=False))
        assert episodes.adapter_for(record).family == 'blackboard_wipe'
    make_free(runs, 'old_free_wipe', task='Wipe the board', scored='failed')
    rows = live.Store([runs]).scan()
    comparison = live.compare(rows, 'blackboard_wipe')
    assert len(comparison['episodes']) == 2
    assert len(live.compare(rows, 'free')['episodes']) == 1
    assert comparison['arms']['harness']['n_success'] == 1
    assert comparison['arms']['baseline']['n_success'] == 0
